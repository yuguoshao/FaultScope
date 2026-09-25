"""Deterministic raw-run aggregation."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def row_case_id(row: dict[str, Any]) -> str:
    if row.get("case_id"):
        return str(row["case_id"])
    if "distance" in row:
        distance = int(row["distance"])
        rounds = int(row.get("rounds", distance))
        return f"rotated_memory_z_d{distance}_r{rounds}"
    return "<missing-case-id>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q3


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    terminal: dict[tuple[str, str, int], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not {"engine", "shots"}.issubset(row):
            continue
        key = (str(row.get("engine")), row_case_id(row), int(row.get("shots", -1)))
        if row.get("record_type") == "config_status":
            terminal[key] = row
        if (
            row.get("included_in_summary") is True
            and row.get("status") == "ok"
            and row.get("seconds_per_e2e") is not None
        ):
            grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, samples in sorted(grouped.items()):
        # Partial successful repetitions preceding a timeout/OOM/error are kept
        # in raw_runs.jsonl for auditability but are not reported as a completed
        # grid point.
        if terminal.get(key, {}).get("status") != "ok":
            continue
        values = [float(sample["seconds_per_e2e"]) for sample in samples]
        q1, q3 = _quartiles(values)
        first = samples[0]
        summary_rows.append(
            {
                "engine": key[0],
                "engine_label": first["engine_label"],
                "engine_version": first.get("engine_version", "unknown"),
                "case_id": key[1],
                "family": first.get("family", "surface_code:rotated_memory_z"),
                "distance": first.get("distance", ""),
                "rounds": first.get("rounds", ""),
                "width": first.get("width", ""),
                "depth": first.get("depth", ""),
                "instance": first.get("instance", ""),
                "circuit_seed": first.get("circuit_seed", ""),
                "shots": key[2],
                "repetitions": len(values),
                "median_seconds": statistics.median(values),
                "q1_seconds": q1,
                "q3_seconds": q3,
                "min_seconds": min(values),
                "max_seconds": max(values),
                "cumulative_seconds": sum(float(sample["wall_seconds"]) for sample in samples),
                "max_peak_rss_bytes": max(int(sample.get("peak_rss_bytes", 0)) for sample in samples),
                "num_qubits": int(first["num_qubits"]),
                "num_measurements": int(first["num_measurements"]),
                "two_qubit_operations": int(first["two_qubit_operations"]),
                "low_sample": bool(terminal.get(key, {}).get("low_sample", len(values) == 1)),
            }
        )

    status_rows: list[dict[str, Any]] = []
    for key, row in sorted(terminal.items()):
        status_rows.append(
            {
                "engine": key[0],
                "engine_label": row["engine_label"],
                "case_id": key[1],
                "family": row.get("family", "surface_code:rotated_memory_z"),
                "distance": row.get("distance", ""),
                "rounds": row.get("rounds", ""),
                "width": row.get("width", ""),
                "depth": row.get("depth", ""),
                "instance": row.get("instance", ""),
                "circuit_seed": row.get("circuit_seed", ""),
                "shots": key[2],
                "status": row["status"],
                "repetitions": int(row.get("repetitions", 0)),
                "low_sample": bool(row.get("low_sample", False)),
                "error": row.get("error", ""),
            }
        )
    return summary_rows, status_rows


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = [
    "engine",
    "engine_label",
    "engine_version",
    "case_id",
    "family",
    "distance",
    "rounds",
    "width",
    "depth",
    "instance",
    "circuit_seed",
    "shots",
    "repetitions",
    "median_seconds",
    "q1_seconds",
    "q3_seconds",
    "min_seconds",
    "max_seconds",
    "cumulative_seconds",
    "max_peak_rss_bytes",
    "num_qubits",
    "num_measurements",
    "two_qubit_operations",
    "low_sample",
]

STATUS_FIELDS = [
    "engine",
    "engine_label",
    "case_id",
    "family",
    "distance",
    "rounds",
    "width",
    "depth",
    "instance",
    "circuit_seed",
    "shots",
    "status",
    "repetitions",
    "low_sample",
    "error",
]


def summarize_results(results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from paper_runtime import write_json
    raw_rows = read_jsonl(results_dir / "raw_runs.jsonl")
    summary_rows, status_rows = aggregate_rows(raw_rows)
    write_csv(results_dir / "summary.csv", summary_rows, fieldnames=SUMMARY_FIELDS)
    write_csv(results_dir / "status.csv", status_rows, fieldnames=STATUS_FIELDS)
    write_json(results_dir / "summary.json", {"rows": summary_rows, "status": status_rows})
    return summary_rows, status_rows

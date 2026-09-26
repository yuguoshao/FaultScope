"""Small, dependency-free statistics and LaTeX tables for the paper benchmarks."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


METHODS = ("no_events", "with_events", "score")
TIMING_FIELDS = ("preparation_seconds", "sampling_seconds", "decoding_seconds",
                 "scoring_seconds", "total_seconds")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty summary")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def attribution_summary(records: list[dict], distances: tuple[int, ...],
                        repeats: int) -> tuple[list[dict], list[dict]]:
    """Use formal trials only, with paired seeds and complete unique repeats."""
    formal = [row for row in records if not row["warmup"]]
    expected = {(d, method, repeat) for d in distances
                for method in METHODS for repeat in range(repeats)}
    keys = [(row["distance"], row["method"], row["repeat"]) for row in formal]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("formal trials must contain every unique distance/method/repeat")
    for d in distances:
        for repeat in range(repeats):
            paired = [row for row in formal if row["distance"] == d and row["repeat"] == repeat]
            if (any(row["status"] != "completed" for row in paired)
                    or len({row["seed"] for row in paired}) != 1
                    or len({row["loss_count"] for row in paired}) != 1):
                raise ValueError(f"failed or inconsistent paired trial d={d}, repeat={repeat}")
    summary, comparisons = [], []
    for distance in distances:
        medians = {}
        for method in METHODS:
            subset = [row for row in formal
                      if row["distance"] == distance and row["method"] == method]
            item = {"distance": distance, "method": method, "repeats": repeats}
            for field in TIMING_FIELDS:
                values = [float(row[field]) for row in subset]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"invalid timing in {field}")
                quartiles = statistics.quantiles(values, n=4, method="inclusive") if repeats > 1 else values * 3
                item[f"median_{field}"] = statistics.median(values)
                item[f"q1_{field}"] = quartiles[0]
                item[f"q3_{field}"] = quartiles[2]
            item["max_peak_process_rss_bytes"] = max(row["peak_process_rss_bytes"] for row in subset)
            summary.append(item)
            medians[method] = item["median_total_seconds"]
        if min(medians.values()) <= 0:
            raise ValueError("total times must be positive")
        comparisons.append({
            "distance": distance,
            "score_over_no_events_percent": 100 * (medians["score"] / medians["no_events"] - 1),
            "with_events_over_no_events_percent": 100 * (medians["with_events"] / medians["no_events"] - 1),
            "score_over_with_events_percent": 100 * (medians["score"] / medians["with_events"] - 1),
            "median_direct_scoring_seconds": next(row["median_scoring_seconds"] for row in summary
                                                  if row["distance"] == distance and row["method"] == "score"),
        })
    return summary, comparisons


def summarize_attribution(records: list[dict], distances: tuple[int, ...], repeats: int) -> dict:
    """Keep valid distances when another distance has failed or missing trials."""
    rows, comparisons = [], []
    for distance in distances:
        selected = [row for row in records if row["distance"] == distance]
        try:
            if any(row["status"] != "completed" for row in selected):
                raise ValueError("failed trial prevents a complete distance result")
            for repeat in {row["repeat"] for row in selected}:
                paired = [row for row in selected if row["repeat"] == repeat]
                if (len({row["seed"] for row in paired}) != 1
                        or len({row["loss_count"] for row in paired}) != 1):
                    raise ValueError(f"inconsistent paired trial d={distance}, repeat={repeat}")
            summary, comparison = attribution_summary(selected, (distance,), repeats)
        except ValueError as error:
            failures = [row for row in selected if row["status"] != "completed"]
            reason = str(error)
            if failures:
                reason += "; " + "; ".join(dict.fromkeys(
                    f"{row['status']}: {row.get('error', 'trial failed')}" for row in failures))
            rows.extend({"distance": distance, "method": method, "repeats": repeats,
                         "status": "incomplete", "error": reason} for method in METHODS)
            comparisons.append({"distance": distance, "status": "incomplete", "error": reason})
        else:
            rows.extend(dict(row, status="completed") for row in summary)
            comparisons.extend(dict(row, status="completed") for row in comparison)
    return {"complete": all(row["status"] == "completed" for row in rows),
            "rows": rows, "comparisons": comparisons}


def write_table3(path: Path, rows: list[dict]) -> None:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Circuit & Simulator & Batch shots & Mshots/s & Preparation (s) & Complete (s) \\",
             r"\midrule"]
    for row in rows:
        prefix = "Random $n=" if row["family"] == "random_clifford" else "Surface $d="
        circuit = prefix + str(row["size"]) + "$"
        simulator = {"faultscope": "FaultScope", "stim": "Stim"}[row["engine"]]
        if row["status"] == "ok":
            values = (f"{row['batch_shots']:,}", f"{row['median_shots_per_second'] / 1e6:.3g}",
                      f"{row['median_setup_seconds']:.3g}", f"{row['median_end_to_end_seconds']:.3g}")
        else:
            values = ("--", "--", "--", "--")
        lines.append(" & ".join((circuit, simulator, *values)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table4(path: Path, summary: list[dict], comparisons: list[dict]) -> None:
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$d=M$ & LER & LER+events & Attribution & $\Delta$ (\%) \\", r"\midrule"]
    for comparison in comparisons:
        distance = comparison["distance"]
        if comparison.get("status", "completed") != "completed":
            lines.append(f"{distance} & -- & -- & -- & --" + r" \\")
            continue
        values = {row["method"]: row["median_total_seconds"] * 1000 for row in summary
                  if row["distance"] == distance}
        fields = [str(distance), *(f"{values[method]:.3f}" for method in METHODS),
                  f"${comparison['score_over_no_events_percent']:+.2f}\\%$"]
        lines.append(" & ".join(fields) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Fig. 2 and Table 2 built only from compatible completed paper runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import ALL_ENGINES, ENGINE_LABELS
from .random_report import ENGINE_STYLES, aggregate_random_instances


def _segments(values: list[int], successful: set[int]) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for value in values:
        if value in successful:
            current.append(value)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _status_marker(status: str) -> str | None:
    return {"timeout": "^", "oom": "X", "error": "P", "unsupported": "P"}.get(status)


def _plot_combined(
    *,
    output_dir: Path,
    surface_manifest: dict[str, Any],
    surface_summary: list[dict[str, str]],
    surface_status: list[dict[str, str]],
    random_manifest: dict[str, Any],
    random_aggregate: list[dict[str, Any]],
    surface_shots: int = 1_000_000,
    random_shots: int = 1_000_000,
    surface_title: str = r"(b) Surface code · $p=10^{-3}$ · 1,000,000 shots",
    random_title: str = "(a) Random Clifford · depth 64 · 1,000,000 shots",
    figure_title: str = "CPU end-to-end stabilizer simulation · 1,000,000 shots",
    output_stem: str = "fig2",
    schema: str = "fsbench-combined-figure-manifest-v1",
    comparison_scope: str | None = None,
    random_first: bool = True,
    engines: tuple[str, ...] = ALL_ENGINES,
    engine_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    cache_dir = output_dir / ".cache" / "combined-paper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {**ENGINE_STYLES, "symft": {
        "color": "#6F4C9B", "linestyle": (0, (3, 1, 1, 1)), "marker": "p",
    }}
    surface_distances = [int(value) for value in surface_manifest["selection"]["distances"]]
    widths = [int(value) for value in random_manifest["selection"]["widths"]]
    surface = {
        (row["engine"], int(row["distance"])): row
        for row in surface_summary
        if int(row["shots"]) == surface_shots
    }
    surface_states = {
        (row["engine"], int(row["distance"])): row
        for row in surface_status
        if int(row["shots"]) == surface_shots
    }
    random_rows = {
        (row["engine"], int(row["width"])): row
        for row in random_aggregate
        if int(row["shots"]) == random_shots
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.72, wspace=0.20)
    status_counts: Counter[str] = Counter()
    for panel, (axis, x_values, title) in enumerate(
        zip(
            axes[::-1] if random_first else axes,
            (surface_distances, widths),
            (surface_title, random_title),
            strict=True,
        )
    ):
        if panel == 0:
            numeric = [float(row["median_seconds"]) for row in surface.values()]
        else:
            numeric = [
                float(row["median_seconds"])
                for row in random_rows.values()
                if row["status"] == "ok"
            ]
        y_bottom = max(min(numeric) / 3.0, 1e-9) if numeric else 1e-6
        y_top = max(numeric) * 4.0 if numeric else 1_800.0
        for engine_index, engine in enumerate(engines):
            style = styles[engine]
            if panel == 0:
                successful = {x for x in x_values if (engine, x) in surface}
                values = lambda x: float(surface[(engine, x)]["median_seconds"])
            else:
                successful = {
                    x for x in x_values if random_rows[(engine, x)]["status"] == "ok"
                }
                values = lambda x: float(random_rows[(engine, x)]["median_seconds"])
            for segment in _segments(x_values, successful):
                axis.plot(
                    segment,
                    [values(x) for x in segment],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    linewidth=1.7,
                    markersize=4.8,
                )
            for x in successful:
                if panel == 1:
                    row = random_rows[(engine, x)]
                    medians = json.loads(str(row["instance_medians_seconds"]))
                    axis.scatter([x] * len(medians), medians, s=13, color=style["color"], alpha=0.35)
            for x in x_values:
                if x in successful:
                    continue
                status = (
                    surface_states.get((engine, x), {}).get("status", "error")
                    if panel == 0
                    else str(random_rows[(engine, x)]["status"])
                )
                status_counts[status] += 1
                marker = _status_marker(status)
                if marker:
                    axis.scatter(
                        [x], [y_top / (1.10**engine_index)], marker=marker, s=38,
                        facecolors="none" if status == "timeout" else style["color"],
                        edgecolors=style["color"],
                    )
        axis.set_xscale("log", base=2 if panel else 10)
        axis.set_yscale("log")
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(value) for value in x_values], rotation=32, ha="right")
        axis.set_ylim(y_bottom, y_top * 1.15)
        axis.grid(True, which="major", color="0.87", linewidth=0.55)
        axis.set_title(title, loc="left", fontsize=10.5)
        axis.set_xlabel("Code distance $d$" if panel == 0 else "Qubits $n$")
        if panel == (1 if random_first else 0):
            axis.set_ylabel("End-to-end wall time (s)")
    handles = [
        Line2D(
            [0], [0], color=styles[engine]["color"],
            linestyle=styles[engine]["linestyle"], marker=styles[engine]["marker"],
            markerfacecolor="white", label=(engine_labels or {}).get(engine, ENGINE_LABELS[engine]),
        )
        for engine in engines
    ]
    status_handles = [
        Line2D(
            [0], [0], color="0.25", marker="^", markerfacecolor="none",
            linestyle="None", label="Timeout",
        ),
        Line2D(
            [0], [0], color="0.25", marker="X", linestyle="None",
            label="Out of memory",
        ),
        Line2D(
            [0], [0], color="0.25", marker="P", linestyle="None",
            label="Error / unavailable",
        ),
    ]
    figure.legend(
        handles=[*handles, *status_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(figure_title, y=0.99, fontsize=12.5)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{output_stem}.png"
    pdf_path = figures_dir / f"{output_stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    manifest: dict[str, Any] = {
        "schema": schema,
        "engines": list(engines),
        "panel_shots": {
            "surface_code": surface_shots,
            "random_clifford": random_shots,
        },
        "panels": (
            ["random_clifford", "surface_code"]
            if random_first
            else ["surface_code", "random_clifford"]
        ),
        "status_counts": dict(status_counts),
        "censored_points_connected": False,
        "error_bars_drawn": False,
        "surface_iqr_drawn": False,
        "random_instance_range_drawn": False,
        "random_instance_points_drawn": True,
        "random_aggregation": (
            "median of three instance medians with individual instance points and no "
            "interval lines"
        ),
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    if surface_shots == random_shots:
        manifest["shots"] = surface_shots
    if comparison_scope is not None:
        manifest["comparison_scope"] = comparison_scope
    return manifest



def compatibility_errors(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """A host/build/protocol mismatch prevents a combined artifact."""
    errors: list[str] = []
    for field in ("benchmark_source_sha256", "dependency_file_sha256", "master_seed", "paper_protocol"):
        if not left.get(field) or left.get(field) != right.get(field):
            errors.append(f"protocol/source differs: {field}")
    for field in ("engines", "shots"):
        if left.get("selection", {}).get(field) != right.get("selection", {}).get(field):
            errors.append(f"selection differs: {field}")
    for field in ("hostname", "machine_id", "platform", "system", "release", "machine", "cpu_model", "total_memory_bytes", "cpu_controls", "cpu_affinity"):
        if left.get("host", {}).get(field) != right.get("host", {}).get(field):
            errors.append(f"host differs: {field}")
    for field in ("worker_cpu", "worker_cpus", "configuration_parallelism", "timeout_seconds", "memory_limit_bytes", "max_batch_shots", "rss_poll_interval_seconds", "fresh_worker_per_sample", "page_cache_prefetch_after_ready", "thread_environment"):
        if left.get("resource_control", {}).get(field) != right.get("resource_control", {}).get(field):
            errors.append(f"execution differs: {field}")
    for engine in ALL_ENGINES:
        for field in ("version", "aer_version", "julia_version", "source_commit", "native_module_sha256", "native_modules_sha256", "dependency_lock", "runtime"):
            if left.get("engine_probes", {}).get(engine, {}).get("details", {}).get(field) != right.get("engine_probes", {}).get(engine, {}).get("details", {}).get(field):
                errors.append(f"software differs: {engine}.{field}")
    return errors


def _table2_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "ERR"
    status = row.get("status", "ok")
    if status != "ok":
        return {"timeout": "TO", "oom": "OOM", "error": "ERR", "unsupported": "ERR"}.get(status, "ERR")
    value = float(row["median_seconds"])
    # Four significant digits retain the precision used for small timings;
    # large values retain one decimal place for readable seconds.
    text = f"{value:.1f}" if value >= 100 else f"{value:.4g}"
    low = row.get("any_low_sample", row.get("low_sample", False))
    if low is True or str(low).lower() == "true":
        text += r"\textsuperscript{$\dagger$}"
    return text


def write_table2(output: Path, *, random_aggregate: list[dict[str, Any]], surface_summary: list[dict[str, Any]], surface_status: list[dict[str, Any]]) -> None:
    random = {(r["engine"], int(r["width"])): r for r in random_aggregate}
    surface = {(r["engine"], int(r["distance"])): dict(r) for r in surface_status}
    for row in surface_summary:
        key = (row["engine"], int(row["distance"]))
        # A failed terminal point never inherits a partial timing.
        if surface.get(key, {}).get("status") == "ok":
            surface[key].update(row)
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Median end-to-end CPU wall time (seconds), using 1,000,000 shots in batches of at most 1,000. Timings include circuit preparation, compilation, sampling, and output processing. TO denotes a 3600-second timeout; ERR an execution error or unavailable engine; OOM a memory-limit failure. A $\dagger$ marks a result with at least one circuit instance measured only once.}",
        r"\label{tab:stabilizer-benchmark-times}", r"\begin{tabular}{lrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{Random depth 64, $p=0$} & \multicolumn{2}{c}{Surface code, $p=10^{-3}$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Simulator & $n=128$ & $n=1024$ & $d=20$ & $d=100$ \\", r"\midrule",
    ]
    labels = {"qiskit-aer": "Qiskit Aer (CPU)", "symft": "SymFT (CPU)"}
    for engine in ALL_ENGINES:
        cells = [random.get((engine, 128)), random.get((engine, 1024)), surface.get((engine, 20)), surface.get((engine, 100))]
        lines.append(labels.get(engine, ENGINE_LABELS[engine]) + " & " + " & ".join(_table2_cell(row) for row in cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def build_combined_report(*, surface_results_dir: Path, random_results_dir: Path, output_dir: Path) -> dict[str, Any]:
    from .validation import read_csv, validate_paper_run
    from paper_runtime import file_sha256, write_json

    runs = {"surface": surface_results_dir, "random": random_results_dir}
    manifests = {name: json.loads((directory / "run_manifest.json").read_text(encoding="utf-8")) for name, directory in runs.items()}
    for name, manifest in manifests.items():
        if manifest.get("complete") is not True:
            raise ValueError(f"latest {name} run is not complete")
        validation = validate_paper_run(runs[name])
        if not validation["passed"]:
            raise ValueError(f"latest {name} run failed validation: {validation['errors']}")
        recorded = manifest.get("paper_validation", {}).get("validated_inputs_sha256", {})
        if recorded != validation["validated_inputs_sha256"]:
            raise ValueError(f"latest {name} raw/summary/status hashes changed after completion")
    errors = compatibility_errors(manifests["surface"], manifests["random"])
    if errors:
        raise ValueError("cannot combine incompatible runs: " + "; ".join(errors))
    random_summary = read_csv(random_results_dir / "summary.csv")
    random_status = read_csv(random_results_dir / "status.csv")
    surface_summary = read_csv(surface_results_dir / "summary.csv")
    surface_status = read_csv(surface_results_dir / "status.csv")
    aggregate = aggregate_random_instances(manifest=manifests["random"], summary_rows=random_summary, status_rows=random_status)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the completed source manifests before the caller records these
    # combined outputs in its own final manifest. This avoids a hash cycle.
    snapshots = output_dir / "source_manifests"
    snapshots.mkdir(exist_ok=True)
    for name, directory in runs.items():
        (snapshots / f"{name}.json").write_bytes((directory / "run_manifest.json").read_bytes())
    figure = _plot_combined(
        output_dir=output_dir, surface_manifest=manifests["surface"], surface_summary=surface_summary,
        surface_status=surface_status, random_manifest=manifests["random"], random_aggregate=aggregate,
        surface_shots=1_000_000, random_shots=1_000_000,
        surface_title=r"(b) Surface code · $p=10^{-3}$ · 1,000,000 shots",
        random_title="(a) Random Clifford · depth 64 · 1,000,000 shots",
        figure_title="CPU end-to-end stabilizer simulation · 1,000,000 shots",
        output_stem="fig2", schema="paper-fig2-v1", comparison_scope="same-host-six-engines-per-workload",
        random_first=True, engines=ALL_ENGINES,
        engine_labels={"qiskit-aer": "Qiskit Aer (CPU)", "symft": "SymFT (CPU)"},
    )
    write_table2(output_dir / "table2.tex", random_aggregate=aggregate, surface_summary=surface_summary, surface_status=surface_status)
    write_json(output_dir / "random_instance_aggregate.json", aggregate)
    manifest = {
        "schema": "paper-fig2-table2-v1", "complete": True, "figure": figure,
        "source_runs": {name: str(directory.resolve()) for name, directory in runs.items()},
        "input_manifests": {name: f"source_manifests/{name}.json" for name in runs},
        "input_sha256": {
            **{f"source_manifests/{name}.json": file_sha256(snapshots / f"{name}.json") for name in runs},
            **{f"{name}/{filename}": file_sha256(directory / filename) for name, directory in runs.items() for filename in ("raw_runs.jsonl", "summary.csv", "status.csv")},
        },
        "output_sha256": {name: file_sha256(output_dir / name) for name in ("figures/fig2.pdf", "figures/fig2.png", "table2.tex", "random_instance_aggregate.json")},
    }
    write_json(output_dir / "combined_manifest.json", manifest)
    return manifest

"""Hierarchical reports for the three-instance random-Clifford workload."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import ENGINE_LABELS, ALL_ENGINES, selection_depth_by_width


ENGINE_STYLES: dict[str, dict[str, Any]] = {
    "symft": {"color": "#6F4C9B", "linestyle": (0, (3, 1, 1, 1)), "marker": "p"},
    "faultscope": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "stim": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    "qiskit-aer": {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    "cirq": {"color": "#CC79A7", "linestyle": ":", "marker": "D"},
    "quantumclifford": {
        "color": "#E69F00",
        "linestyle": (0, (5, 1, 1, 1)),
        "marker": "v",
    },
}

AGGREGATE_FIELDS = [
    "engine",
    "engine_label",
    "width",
    "depth",
    "shots",
    "status",
    "instances_expected",
    "instances_completed",
    "median_seconds",
    "min_instance_median_seconds",
    "max_instance_median_seconds",
    "instance_medians_seconds",
    "any_low_sample",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def aggregate_random_instances(
    *,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, str]],
    status_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selection = manifest["selection"]
    instances = tuple(int(value) for value in selection["instances"])
    depths = selection_depth_by_width(selection)
    summary = {
        (
            row["engine"],
            int(row.get("width") or row["num_qubits"]),
            int(row["instance"]),
            int(row["shots"]),
        ): row
        for row in summary_rows
    }
    statuses = {
        (
            row["engine"],
            int(row.get("distance") or row.get("width") or 0),
            int(row["instance"]),
            int(row["shots"]),
        ): row
        for row in status_rows
    }
    # v2 status rows intentionally keep width in the case identity metadata;
    # recover it from the stable case id when no explicit width column exists.
    for row in status_rows:
        if row.get("distance") or row.get("width"):
            continue
        case_id = row["case_id"]
        width_text = case_id.split("_n", 1)[1].split("_d", 1)[0]
        statuses[(row["engine"], int(width_text), int(row["instance"]), int(row["shots"]))] = row

    output: list[dict[str, Any]] = []
    precedence = ("oom", "timeout", "error", "unsupported")
    for engine in selection["engines"]:
        for width in selection["widths"]:
            for shots in selection["shots"]:
                keys = [(engine, int(width), instance, int(shots)) for instance in instances]
                completed = [summary[key] for key in keys if key in summary]
                terminal_statuses = [
                    statuses.get(key, {}).get("status", "error") for key in keys
                ]
                all_complete = (
                    len(completed) == len(instances)
                    and all(status == "ok" for status in terminal_statuses)
                )
                medians = [float(row["median_seconds"]) for row in completed]
                status = "ok"
                if not all_complete:
                    status = next(
                        (candidate for candidate in precedence if candidate in terminal_statuses),
                        "error",
                    )
                output.append(
                    {
                        "engine": engine,
                        "engine_label": ENGINE_LABELS[engine],
                        "width": int(width),
                        "depth": depths[int(width)],
                        "shots": int(shots),
                        "status": status,
                        "instances_expected": len(instances),
                        "instances_completed": len(completed),
                        "median_seconds": statistics.median(medians) if all_complete else "",
                        "min_instance_median_seconds": min(medians) if all_complete else "",
                        "max_instance_median_seconds": max(medians) if all_complete else "",
                        "instance_medians_seconds": (
                            json.dumps(medians, separators=(",", ":")) if all_complete else ""
                        ),
                        "any_low_sample": any(_bool(row.get("low_sample")) for row in completed),
                    }
                )
    return output


def _segments(values: list[int], successful: set[int]) -> list[list[int]]:
    output: list[list[int]] = []
    current: list[int] = []
    for value in values:
        if value in successful:
            current.append(value)
        elif current:
            output.append(current)
            current = []
    if current:
        output.append(current)
    return output


def _depth_text(selection: dict[str, Any]) -> tuple[str, str]:
    """Return a plot title and manuscript sentence for the selected depths."""

    depths = selection_depth_by_width(selection)
    if selection.get("depth_rule") == "width" and all(
        depth == width for width, depth in depths.items()
    ):
        return (
            "Noiseless $n$-qubit, $n$-layer random-Clifford benchmark",
            "A circuit on $n$ qubits has exactly $n$ logical layers.",
        )
    unique_depths = sorted(set(depths.values()))
    if len(unique_depths) == 1:
        depth = unique_depths[0]
        return (
            f"Noiseless $n$-qubit, depth-{depth} random-Clifford benchmark",
            f"Every circuit has exactly {depth} logical layers, independent of width.",
        )
    mapping = ", ".join(f"$n={width}$: {depth}" for width, depth in depths.items())
    return (
        "Noiseless random-Clifford benchmark with width-dependent depth",
        f"The logical depths are {mapping}.",
    )


def generate_random_scaling_figure(
    *, results_dir: Path, manifest: dict[str, Any], aggregate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    cache_dir = results_dir / ".cache" / "random-report"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    widths = [int(value) for value in manifest["selection"]["widths"]]
    panels = [int(value) for value in manifest["selection"]["shots"]]
    figure_title, _ = _depth_text(manifest["selection"])
    rows = {
        (row["engine"], int(row["width"]), int(row["shots"])): row
        for row in aggregate_rows
    }
    column_count = 1 if len(panels) == 1 else 2
    row_count = math.ceil(len(panels) / column_count)
    figure_height = 4.5 if row_count == 1 else 3.5 * row_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(8.5, figure_height),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.13 if row_count == 1 else 0.085,
        top=0.72 if row_count == 1 else 0.79,
        wspace=0.20,
        hspace=0.42,
    )
    status_counter: Counter[str] = Counter()
    flat_axes = list(axes.flat)
    for panel_index, (axis, shots) in enumerate(
        zip(flat_axes, panels, strict=False)
    ):
        successful_values = [
            float(row["median_seconds"])
            for row in aggregate_rows
            if int(row["shots"]) == shots and row["status"] == "ok"
        ]
        y_bottom = max(min(successful_values) / 3.0, 1e-9) if successful_values else 1e-6
        y_top = max(successful_values) * 4.0 if successful_values else 1_800.0
        for engine_index, engine in enumerate(ALL_ENGINES):
            style = ENGINE_STYLES[engine]
            successful = {
                width
                for width in widths
                if rows.get((engine, width, shots), {}).get("status") == "ok"
            }
            for segment in _segments(widths, successful):
                axis.plot(
                    segment,
                    [float(rows[(engine, width, shots)]["median_seconds"]) for width in segment],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    linewidth=1.5,
                    markersize=4.5,
                )
            for width in successful:
                row = rows[(engine, width, shots)]
                medians = json.loads(str(row["instance_medians_seconds"]))
                axis.scatter(
                    [width] * len(medians),
                    medians,
                    s=12,
                    color=style["color"],
                    alpha=0.35,
                    zorder=2,
                )
            for width in widths:
                row = rows.get((engine, width, shots))
                if row is None:
                    continue
                if row["status"] == "ok":
                    continue
                status_counter[str(row["status"])] += 1
                marker = {"timeout": "^", "oom": "X", "error": "P", "unsupported": "P"}.get(str(row["status"]))
                if marker:
                    axis.scatter(
                        [width],
                        [y_top / (1.10**engine_index)],
                        marker=marker,
                        facecolors="none" if row["status"] == "timeout" else style["color"],
                        edgecolors=style["color"],
                        s=34,
                    )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(widths)
        axis.set_xticklabels([str(width) for width in widths], rotation=35)
        axis.set_ylim(y_bottom, y_top * 1.15)
        axis.grid(True, which="major", color="0.88", linewidth=0.55)
        axis.set_title("1 shot" if shots == 1 else f"{shots:,} shots", loc="left")
        axis.set_xlabel("Qubits $n$")
        if panel_index % column_count == 0:
            axis.set_ylabel("End-to-end wall time (s)")
    for axis in flat_axes[len(panels) :]:
        axis.set_visible(False)
    handles = [
        Line2D(
            [0], [0], color=ENGINE_STYLES[engine]["color"],
            linestyle=ENGINE_STYLES[engine]["linestyle"],
            marker=ENGINE_STYLES[engine]["marker"], markerfacecolor="white",
            label=ENGINE_LABELS[engine],
        )
        for engine in ALL_ENGINES
    ]
    handles.extend([
        Line2D([0], [0], color="0.25", marker="^", markerfacecolor="none", linestyle="None", label="Timeout"),
        Line2D([0], [0], color="0.25", marker="X", linestyle="None", label="Out of memory"),
        Line2D([0], [0], color="0.25", marker="P", linestyle="None", label="Error / unavailable"),
    ])
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    figure.suptitle(
        "(a) " + figure_title,
        y=0.84 if row_count == 1 else 0.86,
        fontsize=12,
    )
    png_path = figures_dir / "fig2a_random_clifford.png"
    pdf_path = figures_dir / "fig2a_random_clifford.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    output = {
        "schema": "fsbench-random-figure-manifest-v1",
        "panels": panels,
        "depth_rule": manifest["selection"].get("depth_rule"),
        "depth_by_width": manifest["selection"].get("depth_by_width"),
        "styles": {engine: ENGINE_STYLES[engine] for engine in ALL_ENGINES},
        "status_counts": dict(status_counter),
        "censored_points_connected": False,
        "error_bars_drawn": False,
        "instance_range_drawn": False,
        "individual_instance_points_drawn": True,
        "single_sample_iqr_drawn": False,
        "labels_complete": True,
        "aggregation": (
            "median of three per-instance timing medians; individual instance points "
            "shown without interval lines"
        ),
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    (results_dir / "figure_manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output

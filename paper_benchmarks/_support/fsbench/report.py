"""Static benchmark figures, tables, and manuscript-ready LaTeX fragments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import ENGINE_LABELS, ALL_ENGINES


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_affinity_text(resources: dict[str, Any]) -> str:
    cpus = list(resources.get("worker_cpus", [resources.get("worker_cpu", "unknown")]))
    if len(cpus) > 16:
        if all(cpu is None for cpu in cpus):
            return f"{len(cpus)} unpinned worker slots"
        return (
            f"{len(cpus)} physical CPUs; base {cpus[0]}; "
            "exact list in run manifest"
        )
    return ", ".join(str(value) for value in cpus)


def _segments(
    distances: list[int], successful: set[int]
) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for distance in distances:
        if distance in successful:
            current.append(distance)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _status_marker(status: str) -> str | None:
    return {"timeout": "^", "oom": "X", "error": "P", "unsupported": "P"}.get(status)


def generate_scaling_figure(
    *,
    results_dir: Path,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, str]],
    status_rows: list[dict[str, str]],
) -> dict[str, Any]:
    cache_dir = results_dir / ".cache"
    matplotlib_cache = cache_dir / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    distances = [int(value) for value in manifest["selection"]["distances"]]
    panels = [int(value) for value in manifest["selection"]["shots"]]
    summary = {
        (row["engine"], int(row["distance"]), int(row["shots"])): row
        for row in summary_rows
    }
    statuses = {
        (row["engine"], int(row["distance"]), int(row["shots"])): row
        for row in status_rows
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if len(panels) == 1:
        figure, axis = plt.subplots(
            1, 1, figsize=(7.2, 4.8), constrained_layout=False
        )
        axes = [axis]
        columns = 1
        figure.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.70)
    elif len(panels) == 2:
        figure, raw_axes = plt.subplots(
            1, 2, figsize=(9.2, 4.8), constrained_layout=False
        )
        axes = list(raw_axes)
        columns = 2
        figure.subplots_adjust(
            left=0.085, right=0.985, bottom=0.16, top=0.70, wspace=0.22
        )
    else:
        rows = (len(panels) + 1) // 2
        figure, raw_axes = plt.subplots(
            rows, 2, figsize=(8.5, 3.5 * rows), constrained_layout=False
        )
        axes = list(raw_axes.flat)
        columns = 2
        figure.subplots_adjust(
            left=0.09,
            right=0.985,
            bottom=0.085,
            top=0.775,
            wspace=0.20,
            hspace=0.42,
        )
        for unused_axis in axes[len(panels) :]:
            unused_axis.set_visible(False)
    timeout_limit = float(manifest.get("resource_control", {}).get("timeout_seconds", 1_800.0))
    status_counter: Counter[str] = Counter()

    for panel_index, (axis, shots) in enumerate(
        zip(axes[: len(panels)], panels, strict=True)
    ):
        panel_values = [
            float(row["median_seconds"])
            for row in summary_rows
            if int(row["shots"]) == shots and float(row["median_seconds"]) > 0
        ]
        minimum_values = [
            float(row["median_seconds"])
            for row in summary_rows
            if int(row["shots"]) == shots and float(row["median_seconds"]) > 0
        ]
        if panel_values:
            y_top = max(panel_values) * 4.0
            y_bottom = max(min(minimum_values) / 3.0, 1e-9)
        else:
            y_top = max(timeout_limit, 1.0)
            y_bottom = max(y_top / 1e6, 1e-9)
        if y_top <= y_bottom:
            y_top = y_bottom * 10.0

        for engine_index, engine in enumerate(ALL_ENGINES):
            style = ENGINE_STYLES[engine]
            successful = {
                distance
                for distance in distances
                if (engine, distance, shots) in summary
            }
            for segment in _segments(distances, successful):
                medians = [float(summary[(engine, distance, shots)]["median_seconds"]) for distance in segment]
                axis.plot(
                    segment,
                    medians,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    marker=style["marker"],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    zorder=3,
                )
            censor_height = y_top / (1.10 ** engine_index)
            for distance in distances:
                status_row = statuses.get((engine, distance, shots))
                if not status_row or status_row["status"] == "ok":
                    continue
                status = status_row["status"]
                status_counter[status] += 1
                marker = _status_marker(status)
                if marker is None:
                    continue
                axis.scatter(
                    [distance],
                    [censor_height],
                    marker=marker,
                    s=34,
                    facecolors="none" if status == "timeout" else style["color"],
                    edgecolors=style["color"],
                    linewidths=1.1,
                    zorder=4,
                )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(min(distances) / 1.15, max(distances) * 1.15)
        axis.set_ylim(y_bottom, y_top * 1.15)
        axis.set_xticks(distances)
        axis.set_xticklabels([str(distance) for distance in distances], rotation=35)
        axis.grid(True, which="major", color="0.88", linewidth=0.55)
        axis.grid(True, which="minor", axis="y", color="0.94", linewidth=0.35)
        title = "1 shot" if shots == 1 else f"{shots:,} shots"
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("Code distance $d$")
        if panel_index % columns == 0:
            axis.set_ylabel("End-to-end wall time (s)")

    engine_handles = [
        Line2D(
            [0],
            [0],
            color=ENGINE_STYLES[engine]["color"],
            linestyle=ENGINE_STYLES[engine]["linestyle"],
            marker=ENGINE_STYLES[engine]["marker"],
            markerfacecolor="white",
            linewidth=1.5,
            markersize=4.5,
            label=ENGINE_LABELS[engine],
        )
        for engine in ALL_ENGINES
    ]
    status_handles = [
        Line2D([0], [0], color="0.25", marker="^", markerfacecolor="none", linestyle="None", label="Timeout"),
        Line2D([0], [0], color="0.25", marker="X", linestyle="None", label="Out of memory"),
        Line2D([0], [0], color="0.25", marker="P", linestyle="None", label="Error / unavailable"),
    ]
    figure.legend(
        handles=[*engine_handles, *status_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        handlelength=2.7,
    )
    figure.suptitle(
        r"(b) Rotated surface-code memory-Z · $p=10^{-3}$",
        y=0.84 if len(panels) <= 2 else 0.855,
        fontsize=12,
        fontweight="semibold",
    )
    png_path = figures_dir / "fig2b_surface_code.png"
    pdf_path = figures_dir / "fig2b_surface_code.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    style_manifest = {
        engine: {
            "color": style["color"],
            "linestyle": str(style["linestyle"]),
            "marker": style["marker"],
        }
        for engine, style in ENGINE_STYLES.items()
    }
    figure_manifest = {
        "schema": "fsbench-figure-manifest-v1",
        "panels": panels,
        "styles": style_manifest,
        "status_counts": dict(status_counter),
        "censored_points_connected": False,
        "error_bars_drawn": False,
        "timing_iqr_drawn": False,
        "single_sample_iqr_drawn": False,
        "labels_complete": True,
        "sources": {
            "raw_runs_sha256": _sha256(results_dir / "raw_runs.jsonl"),
            "summary_sha256": _sha256(results_dir / "summary.csv"),
            "status_sha256": _sha256(results_dir / "status.csv"),
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
    }
    (results_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return figure_manifest

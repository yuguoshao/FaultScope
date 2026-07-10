"""Small analysis helpers for collection results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from pathlib import Path
from typing import Any

from faultscope.collection._types import TaskStats


def error_rate_points(
    stats: Iterable[TaskStats],
    *,
    x_key: str,
    group_key: str | None = None,
    count_key: str | None = None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for stat in stats:
        metadata = dict(stat.metadata)
        x = metadata[x_key]
        group = metadata[group_key] if group_key is not None else None
        errors = int(stat.custom_counts.get(count_key, 0)) if count_key is not None else stat.errors
        shots = stat.accepted_shots
        rate = errors / shots if shots else math.nan
        points.append(
            {
                "x": x,
                "group": group,
                "errors": errors,
                "shots": shots,
                "rate": rate,
                "stderr": _binomial_stderr(rate, shots),
                "stat": stat,
            }
        )
    return points


def fit_log_error_rate_lines(
    points: Iterable[Mapping[str, Any]],
    *,
    x_key: str,
    group_key: str,
) -> dict[Any, dict[str, float | int | Any]]:
    del x_key, group_key
    grouped: dict[Any, list[tuple[float, float]]] = defaultdict(list)
    for point in points:
        rate = float(point["rate"])
        if not math.isfinite(rate) or rate <= 0:
            continue
        grouped[point.get("group")].append((float(point["x"]), math.log(rate)))

    fits: dict[Any, dict[str, float | int | Any]] = {}
    for group, values in grouped.items():
        values.sort()
        if len(values) == 1:
            slope = 0.0
            intercept = values[0][1]
        else:
            xs = [x for x, _ in values]
            ys = [y for _, y in values]
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            denom = sum((x - x_mean) ** 2 for x in xs)
            slope = (
                0.0 if denom == 0.0 else sum((x - x_mean) * (y - y_mean) for x, y in values) / denom
            )
            intercept = y_mean - slope * x_mean
        fits[group] = {
            "group": group,
            "slope": slope,
            "intercept": intercept,
            "points": len(values),
        }
    return fits


def predict_error_rate(fit: Mapping[str, Any], x: float) -> float:
    return math.exp(float(fit["intercept"]) + float(fit["slope"]) * float(x))


def plot_error_rates(
    stats: Iterable[TaskStats],
    *,
    x_key: str,
    group_key: str | None = None,
    output: str | Path | None = None,
    ax: Any | None = None,
    count_key: str | None = None,
) -> Any:
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    points = error_rate_points(
        stats,
        x_key=x_key,
        group_key=group_key,
        count_key=count_key,
    )
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[point["group"]].append(point)

    for group, group_points in grouped.items():
        group_points.sort(key=lambda point: point["x"])
        label = str(group) if group_key is not None else None
        ax.errorbar(
            [point["x"] for point in group_points],
            [point["rate"] for point in group_points],
            yerr=[point["stderr"] for point in group_points],
            marker="o",
            linestyle="-",
            label=label,
        )
    ax.set_xlabel(x_key)
    ax.set_ylabel("logical error rate")
    if group_key is not None:
        ax.legend(title=group_key)
    if output is not None:
        ax.figure.savefig(output, bbox_inches="tight")
    return ax


def _binomial_stderr(rate: float, shots: int) -> float:
    if shots == 0 or not math.isfinite(rate):
        return math.nan
    return math.sqrt(rate * (1.0 - rate) / shots)


def _import_matplotlib() -> Any:
    import os
    import tempfile

    cache_root = Path(tempfile.gettempdir()) / "faultscope-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_error_rates requires matplotlib; install faultscope[collection]"
        ) from exc
    return plt

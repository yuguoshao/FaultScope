"""Threshold analysis for collection statistics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

from ._types import TaskStats


__all__ = [
    "ThresholdPoint",
    "ThresholdEstimate",
    "PairwiseCrossing",
    "FiniteSizeScalingFit",
    "ThresholdAnalysisResult",
    "analyze_thresholds",
    "plot_threshold_analysis",
]


@dataclass(frozen=True)
class ThresholdPoint:
    x: float
    distance: float
    shots: int
    errors: int
    rate: float
    stderr: float


@dataclass(frozen=True)
class ThresholdEstimate:
    value: float
    ci_low: float | None
    ci_high: float | None
    confidence_level: float
    bootstrap_samples: int
    bootstrap_successes: int


@dataclass(frozen=True)
class PairwiseCrossing:
    lower_distance: float
    upper_distance: float
    status: Literal["ok", "no_crossing", "ambiguous"]
    candidates: tuple[float, ...]
    estimate: ThresholdEstimate | None


@dataclass(frozen=True)
class FiniteSizeScalingFit:
    status: Literal["ok", "insufficient_data", "fit_failed", "bootstrap_unstable"]
    threshold: ThresholdEstimate | None
    critical_exponent: ThresholdEstimate | None
    coefficients: tuple[float, ...]
    reduced_chi_squared: float | None
    message: str | None


@dataclass(frozen=True)
class ThresholdAnalysisResult:
    series: Mapping[str, object]
    points: tuple[ThresholdPoint, ...]
    crossings: tuple[PairwiseCrossing, ...]
    pairwise_threshold: ThresholdEstimate | None
    scaling_fit: FiniteSizeScalingFit

    def to_dict(self) -> dict[str, object]:
        return _json_safe(
            {
                "series": self.series,
                "points": self.points,
                "crossings": self.crossings,
                "pairwise_threshold": self.pairwise_threshold,
                "scaling_fit": self.scaling_fit,
            }
        )


def analyze_thresholds(
    stats: Iterable[TaskStats],
    *,
    x_key: str,
    distance_key: str,
    series_keys: Iterable[str] = (),
    count_key: str | None = None,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
    scaling_order: Literal[1, 2, 3] = 2,
) -> tuple[ThresholdAnalysisResult, ...]:
    """Analyze threshold crossings and finite-size scaling by series."""

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if scaling_order not in (1, 2, 3):
        raise ValueError("scaling_order must be 1, 2, or 3")

    series_key_tuple = tuple(series_keys)
    grouped = _aggregate_points(
        stats,
        x_key=x_key,
        distance_key=distance_key,
        series_keys=series_key_tuple,
        count_key=count_key,
    )

    ordered_series = sorted(grouped.items(), key=lambda item: item[0])
    child_seeds: dict[str, int] = {}
    if bootstrap_samples:
        np = _import_numpy()
        root_rng = np.random.default_rng(seed)
        high = 2**63 - 1
        for series_id, _ in ordered_series:
            child_seeds[series_id] = int(root_rng.integers(0, high))

    results: list[ThresholdAnalysisResult] = []
    for series_id, aggregate in ordered_series:
        points = tuple(
            _make_point(x=x, distance=distance, shots=shots, errors=errors)
            for (_series_id, distance, x), (shots, errors) in sorted(
                aggregate["counts"].items(), key=lambda item: (item[0][1], item[0][2])
            )
        )
        crossings, pairwise = _pairwise_summary(
            points,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            pair_bootstrap_values={},
            aggregate_bootstrap_values=(),
        )
        scaling_fit = _fit_scaling(
            points,
            scaling_order=scaling_order,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            threshold_bootstrap_values=(),
            nu_bootstrap_values=(),
        )

        if bootstrap_samples:
            pair_bootstrap: dict[tuple[float, float], list[float]] = {}
            pairwise_bootstrap: list[float] = []
            scaling_threshold_bootstrap: list[float] = []
            scaling_nu_bootstrap: list[float] = []
            np = _import_numpy()
            rng = np.random.default_rng(child_seeds[series_id])
            for _ in range(bootstrap_samples):
                boot_points = _bootstrap_points(points, rng)
                boot_crossings, boot_pairwise = _pairwise_summary(
                    boot_points,
                    confidence_level=confidence_level,
                    bootstrap_samples=0,
                    pair_bootstrap_values={},
                    aggregate_bootstrap_values=(),
                )
                boot_scaling = _fit_scaling(
                    boot_points,
                    scaling_order=scaling_order,
                    confidence_level=confidence_level,
                    bootstrap_samples=0,
                    threshold_bootstrap_values=(),
                    nu_bootstrap_values=(),
                )
                for crossing in boot_crossings:
                    if crossing.status == "ok" and crossing.estimate is not None:
                        key = (crossing.lower_distance, crossing.upper_distance)
                        pair_bootstrap.setdefault(key, []).append(crossing.estimate.value)
                if boot_pairwise is not None:
                    pairwise_bootstrap.append(boot_pairwise.value)
                if boot_scaling.threshold is not None:
                    scaling_threshold_bootstrap.append(boot_scaling.threshold.value)
                if boot_scaling.critical_exponent is not None:
                    scaling_nu_bootstrap.append(boot_scaling.critical_exponent.value)

            crossings, pairwise = _pairwise_summary(
                points,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                pair_bootstrap_values=pair_bootstrap,
                aggregate_bootstrap_values=pairwise_bootstrap,
            )
            scaling_fit = _fit_scaling(
                points,
                scaling_order=scaling_order,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                threshold_bootstrap_values=scaling_threshold_bootstrap,
                nu_bootstrap_values=scaling_nu_bootstrap,
            )

        results.append(
            ThresholdAnalysisResult(
                series=aggregate["series"],
                points=points,
                crossings=crossings,
                pairwise_threshold=pairwise,
                scaling_fit=scaling_fit,
            )
        )
    return tuple(results)


def plot_threshold_analysis(
    results: ThresholdAnalysisResult | Iterable[ThresholdAnalysisResult],
    *,
    output: str | Path | None = None,
    axes: Any | None = None,
    log_y: bool = True,
) -> tuple[Any, Any]:
    """Plot threshold diagnostics as one rate/collapse row per result series."""

    normalized_results = _normalize_threshold_results(results)
    plt = _import_matplotlib()
    if axes is None:
        figure, axes = plt.subplots(
            len(normalized_results),
            2,
            squeeze=False,
            figsize=(11.0, 4.0 * len(normalized_results)),
        )
    else:
        axes, figure = _validate_threshold_axes(axes, len(normalized_results))

    for row, result in enumerate(normalized_results):
        left = _axis_at(axes, row, 0)
        right = _axis_at(axes, row, 1)
        _plot_threshold_rates(left, result, log_y=log_y)
        _plot_scaling_collapse(right, result)

    figure.tight_layout()
    if output is not None:
        figure.savefig(output, bbox_inches="tight")
    return figure, axes


def _aggregate_points(
    stats: Iterable[TaskStats],
    *,
    x_key: str,
    distance_key: str,
    series_keys: tuple[str, ...],
    count_key: str | None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for stat in stats:
        metadata = stat.metadata
        x = _read_finite_float(metadata, x_key, "x")
        distance = _read_finite_float(metadata, distance_key, "distance")
        if distance <= 0.0:
            raise ValueError("distance must be positive")

        shots = int(stat.accepted_shots)
        if shots <= 0:
            raise ValueError("accepted shots must be positive")
        errors = _read_error_count(stat, count_key=count_key, shots=shots)

        series = {
            key: stat.decoder if key == "decoder" else _read_metadata(metadata, key)
            for key in series_keys
        }
        series_id = _canonical_json(series)
        if series_id not in grouped:
            grouped[series_id] = {
                "series": series,
                "counts": {},
            }
        key = (series_id, distance, x)
        old_shots, old_errors = grouped[series_id]["counts"].get(key, (0, 0))
        grouped[series_id]["counts"][key] = (
            int(old_shots) + shots,
            int(old_errors) + errors,
        )
    return grouped


def _normalize_threshold_results(
    results: ThresholdAnalysisResult | Iterable[ThresholdAnalysisResult],
) -> tuple[ThresholdAnalysisResult, ...]:
    normalized: tuple[ThresholdAnalysisResult, ...]
    if isinstance(results, ThresholdAnalysisResult):
        normalized = (results,)
    else:
        normalized = tuple(results)
    if not normalized:
        raise ValueError("plot_threshold_analysis requires at least one result")
    return normalized


def _validate_threshold_axes(axes: Any, rows: int) -> tuple[Any, Any]:
    expected = (rows, 2)
    if hasattr(axes, "shape"):
        if tuple(axes.shape) != expected:
            raise ValueError(f"axes must have exact shape {expected}")
        return axes, axes[0, 0].figure

    try:
        nested = [list(row) for row in axes]
    except TypeError as exc:
        raise ValueError(f"axes must have exact shape {expected}") from exc
    if len(nested) != rows or any(len(row) != 2 for row in nested):
        raise ValueError(f"axes must have exact shape {expected}")
    return nested, nested[0][0].figure


def _axis_at(axes: Any, row: int, column: int) -> Any:
    try:
        return axes[row, column]
    except TypeError:
        return axes[row][column]


def _plot_threshold_rates(ax: Any, result: ThresholdAnalysisResult, *, log_y: bool) -> None:
    by_distance: dict[float, list[ThresholdPoint]] = {}
    for point in result.points:
        by_distance.setdefault(point.distance, []).append(point)

    for distance in sorted(by_distance):
        points = sorted(by_distance[distance], key=lambda point: point.x)
        ax.errorbar(
            [point.x for point in points],
            [point.rate for point in points],
            yerr=[point.stderr for point in points],
            marker="o",
            linestyle="-",
            label=_format_distance(distance),
        )

    for crossing in result.crossings:
        color = "0.55" if crossing.status == "ok" else "0.75"
        for candidate in crossing.candidates:
            lower_points = tuple(
                sorted(by_distance[crossing.lower_distance], key=lambda point: point.x)
            )
            upper_points = tuple(
                sorted(by_distance[crossing.upper_distance], key=lambda point: point.x)
            )
            marker_rate = _logistic(
                0.5
                * (
                    _interpolate_logit(lower_points, candidate)
                    + _interpolate_logit(upper_points, candidate)
                )
            )
            ax.axvline(candidate, color=color, linestyle=":", linewidth=1.0, alpha=0.8)
            ax.plot(
                [candidate],
                [marker_rate],
                marker="x",
                color=color,
                linestyle="None",
            )
        if crossing.estimate is not None:
            ax.axvline(
                crossing.estimate.value,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
            )

    threshold = result.scaling_fit.threshold
    if threshold is not None:
        ax.axvline(
            threshold.value,
            color="black",
            linestyle="-",
            linewidth=1.2,
            label="scaling threshold",
        )
        if threshold.ci_low is not None and threshold.ci_high is not None:
            ax.axvspan(threshold.ci_low, threshold.ci_high, color="black", alpha=0.12)

    if log_y:
        if any(point.rate == 0.0 for point in result.points):
            positive_rates = [point.rate for point in result.points if point.rate > 0.0]
            if positive_rates:
                linthresh = min(positive_rates) / 10.0
            else:
                linthresh = min(1.0 / (point.shots + 1.0) for point in result.points)
            ax.set_yscale("symlog", linthresh=linthresh)
        else:
            ax.set_yscale("log")
    ax.set_xlabel("physical error rate")
    ax.set_ylabel("logical error rate")
    ax.set_title(_format_series_label(result.series))
    ax.legend(title="distance")


def _plot_scaling_collapse(ax: Any, result: ThresholdAnalysisResult) -> None:
    fit = result.scaling_fit
    threshold = fit.threshold
    exponent = fit.critical_exponent
    if threshold is None or exponent is None or not fit.coefficients:
        message = f"scaling fit: {fit.status}"
        if fit.message:
            message = f"{message}\n{fit.message}"
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    scaled = [
        ((point.x - threshold.value) * point.distance ** (1.0 / exponent.value), point.rate)
        for point in result.points
    ]
    scaled.sort(key=lambda item: item[0])
    ax.scatter(
        [x for x, _ in scaled],
        [rate for _, rate in scaled],
        marker="o",
        label="points",
    )

    min_x = scaled[0][0]
    max_x = scaled[-1][0]
    if math.isclose(min_x, max_x, rel_tol=0.0, abs_tol=1e-15):
        curve_x = [min_x]
    else:
        steps = 160
        curve_x = [min_x + (max_x - min_x) * index / (steps - 1) for index in range(steps)]
    curve_y = [_logistic(_evaluate_polynomial(fit.coefficients, x)) for x in curve_x]
    ax.plot(curve_x, curve_y, color="black", linewidth=1.2, label="fit")
    ax.set_xlabel("scaled distance variable")
    ax.set_ylabel("logical error rate")
    ax.set_title(f"finite-size collapse: {fit.status}")
    ax.legend()


def _evaluate_polynomial(coefficients: tuple[float, ...], x: float) -> float:
    total = 0.0
    power = 1.0
    for coefficient in coefficients:
        total += coefficient * power
        power *= x
    return total


def _format_distance(distance: float) -> str:
    return f"d={distance:g}"


def _format_series_label(series: Mapping[str, object]) -> str:
    if not series:
        return "all series"
    return ",".join(f"{key}={value}" for key, value in series.items())


def _read_metadata(metadata: Mapping[str, object], key: str) -> object:
    if key not in metadata:
        raise ValueError(f"missing metadata key {key!r}")
    return metadata[key]


def _read_finite_float(metadata: Mapping[str, object], key: str, label: str) -> float:
    value = _read_metadata(metadata, key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _read_error_count(stat: TaskStats, *, count_key: str | None, shots: int) -> int:
    if count_key is None:
        value = stat.errors
    else:
        if count_key not in stat.custom_counts:
            raise ValueError(f"missing custom count {count_key!r}")
        value = stat.custom_counts[count_key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("count must be an integer")
    errors = int(value)
    if errors < 0 or errors > shots:
        raise ValueError("count must be between 0 and accepted shots")
    return errors


def _make_point(*, x: float, distance: float, shots: int, errors: int) -> ThresholdPoint:
    rate = float(errors) / float(shots)
    stderr = math.sqrt(rate * (1.0 - rate) / float(shots))
    return ThresholdPoint(
        x=float(x),
        distance=float(distance),
        shots=int(shots),
        errors=int(errors),
        rate=rate,
        stderr=stderr,
    )


def _continuity_corrected_rate(point: ThresholdPoint) -> float:
    return (float(point.errors) + 0.5) / (float(point.shots) + 1.0)


def _continuity_corrected_stderr(point: ThresholdPoint) -> float:
    rate = _continuity_corrected_rate(point)
    return math.sqrt(rate * (1.0 - rate) / float(point.shots))


def _pairwise_summary(
    points: tuple[ThresholdPoint, ...],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    pair_bootstrap_values: Mapping[tuple[float, float], Iterable[float]],
    aggregate_bootstrap_values: Iterable[float],
) -> tuple[tuple[PairwiseCrossing, ...], ThresholdEstimate | None]:
    by_distance: dict[float, list[ThresholdPoint]] = {}
    for point in points:
        by_distance.setdefault(point.distance, []).append(point)
    distances = sorted(by_distance)
    crossings: list[PairwiseCrossing] = []
    aggregate_candidates: list[float] = []
    for lower, upper in zip(distances, distances[1:]):
        lower_points = tuple(sorted(by_distance[lower], key=lambda point: point.x))
        upper_points = tuple(sorted(by_distance[upper], key=lambda point: point.x))
        candidates = _crossing_candidates(lower_points, upper_points)
        status: Literal["ok", "no_crossing", "ambiguous"] = (
            "no_crossing" if not candidates else "ok" if len(candidates) == 1 else "ambiguous"
        )
        aggregate_candidates.extend(candidates)
        estimate = None
        if status == "ok":
            estimate = _estimate_from_value(
                candidates[0],
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                bootstrap_values=pair_bootstrap_values.get((lower, upper), ()),
            )
        crossings.append(
            PairwiseCrossing(
                lower_distance=lower,
                upper_distance=upper,
                status=status,
                candidates=tuple(candidates),
                estimate=estimate,
            )
        )

    pairwise = None
    unique_candidates = _dedupe_sorted(aggregate_candidates)
    if unique_candidates:
        pairwise = _estimate_from_value(
            _median(unique_candidates),
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_values=aggregate_bootstrap_values,
        )
    return tuple(crossings), pairwise


def _crossing_candidates(
    lower_points: tuple[ThresholdPoint, ...],
    upper_points: tuple[ThresholdPoint, ...],
) -> list[float]:
    if len(lower_points) < 2 or len(upper_points) < 2:
        return []
    lo = max(lower_points[0].x, upper_points[0].x)
    hi = min(lower_points[-1].x, upper_points[-1].x)
    if lo > hi:
        return []
    knots = sorted(
        {
            lo,
            hi,
            *[point.x for point in lower_points if lo <= point.x <= hi],
            *[point.x for point in upper_points if lo <= point.x <= hi],
        }
    )
    if len(knots) == 1:
        diff = _interpolate_logit(lower_points, knots[0]) - _interpolate_logit(
            upper_points, knots[0]
        )
        return [knots[0]] if _is_close_zero(diff) else []
    if len(knots) < 2:
        return []
    diffs = [
        _interpolate_logit(lower_points, x) - _interpolate_logit(upper_points, x) for x in knots
    ]
    candidates: list[float] = []
    for left_x, right_x, left_diff, right_diff in zip(knots, knots[1:], diffs, diffs[1:]):
        if _is_close_zero(left_diff):
            candidates.append(left_x)
        if _is_close_zero(right_diff):
            candidates.append(right_x)
        if left_diff * right_diff < 0.0:
            fraction = abs(left_diff) / (abs(left_diff) + abs(right_diff))
            candidates.append(left_x + fraction * (right_x - left_x))
    if _is_close_zero(diffs[-1]):
        candidates.append(knots[-1])
    return _dedupe_sorted(candidate for candidate in candidates if lo <= candidate <= hi)


def _interpolate_logit(points: tuple[ThresholdPoint, ...], x: float) -> float:
    if x <= points[0].x:
        return _logit(_continuity_corrected_rate(points[0]))
    for left, right in zip(points, points[1:]):
        if x <= right.x:
            left_y = _logit(_continuity_corrected_rate(left))
            right_y = _logit(_continuity_corrected_rate(right))
            if right.x == left.x:
                return left_y
            fraction = (x - left.x) / (right.x - left.x)
            return left_y + fraction * (right_y - left_y)
    return _logit(_continuity_corrected_rate(points[-1]))


def _fit_scaling(
    points: tuple[ThresholdPoint, ...],
    *,
    scaling_order: Literal[1, 2, 3],
    confidence_level: float,
    bootstrap_samples: int,
    threshold_bootstrap_values: Iterable[float],
    nu_bootstrap_values: Iterable[float],
) -> FiniteSizeScalingFit:
    by_distance: dict[float, list[ThresholdPoint]] = {}
    for point in points:
        by_distance.setdefault(point.distance, []).append(point)
    if len(by_distance) < 3 or any(
        len(distance_points) < 2 for distance_points in by_distance.values()
    ):
        return FiniteSizeScalingFit(
            status="insufficient_data",
            threshold=None,
            critical_exponent=None,
            coefficients=(),
            reduced_chi_squared=None,
            message="scaling fit requires at least three distances and two x points for each distance",
        )
    usable = {
        distance: sorted(distance_points, key=lambda point: point.x)
        for distance, distance_points in by_distance.items()
    }

    lower_bound = max(points_for_distance[0].x for points_for_distance in usable.values())
    upper_bound = min(points_for_distance[-1].x for points_for_distance in usable.values())
    if not lower_bound < upper_bound:
        return FiniteSizeScalingFit(
            status="insufficient_data",
            threshold=None,
            critical_exponent=None,
            coefficients=(),
            reduced_chi_squared=None,
            message="scaling fit requires common x support",
        )

    fit_points = tuple(point for distance_points in usable.values() for point in distance_points)
    parameter_count = scaling_order + 3
    degrees_of_freedom = len(fit_points) - parameter_count
    if degrees_of_freedom <= 0:
        return FiniteSizeScalingFit(
            status="insufficient_data",
            threshold=None,
            critical_exponent=None,
            coefficients=(),
            reduced_chi_squared=None,
            message="scaling fit requires positive residual degrees of freedom",
        )

    try:
        fit = _solve_scaling_fit(
            fit_points,
            scaling_order=scaling_order,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    except ImportError:
        raise
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return FiniteSizeScalingFit(
            status="fit_failed",
            threshold=None,
            critical_exponent=None,
            coefficients=(),
            reduced_chi_squared=None,
            message=str(exc),
        )

    if fit is None:
        return FiniteSizeScalingFit(
            status="fit_failed",
            threshold=None,
            critical_exponent=None,
            coefficients=(),
            reduced_chi_squared=None,
            message="least_squares did not converge",
        )

    threshold, nu, coefficients, chi_squared = fit
    threshold_estimate = _estimate_from_value(
        threshold,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_values=threshold_bootstrap_values,
    )
    nu_estimate = _estimate_from_value(
        nu,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_values=nu_bootstrap_values,
    )
    status: Literal["ok", "bootstrap_unstable"] = "ok"
    if bootstrap_samples and (
        threshold_estimate.bootstrap_successes < _minimum_bootstrap_successes(bootstrap_samples)
        or nu_estimate.bootstrap_successes < _minimum_bootstrap_successes(bootstrap_samples)
    ):
        status = "bootstrap_unstable"
    return FiniteSizeScalingFit(
        status=status,
        threshold=threshold_estimate,
        critical_exponent=nu_estimate,
        coefficients=tuple(coefficients),
        reduced_chi_squared=chi_squared / degrees_of_freedom,
        message=None,
    )


def _solve_scaling_fit(
    points: tuple[ThresholdPoint, ...],
    *,
    scaling_order: Literal[1, 2, 3],
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float, tuple[float, ...], float] | None:
    np = _import_numpy()
    least_squares = _import_least_squares()

    x = np.asarray([point.x for point in points], dtype=float)
    distance = np.asarray([point.distance for point in points], dtype=float)
    corrected_rates = [_continuity_corrected_rate(point) for point in points]
    y = np.asarray([_logit(rate) for rate in corrected_rates], dtype=float)
    sigma = np.asarray(
        [
            max(
                _continuity_corrected_stderr(point) / max(rate * (1.0 - rate), 1e-12),
                1e-12,
            )
            for point, rate in zip(points, corrected_rates)
        ],
        dtype=float,
    )
    y_mean = float(np.mean(y))

    def residual(parameters: Any) -> Any:
        threshold = parameters[0]
        nu = parameters[1]
        coeffs = parameters[2:]
        z = (x - threshold) * np.power(distance, 1.0 / nu)
        prediction = np.zeros_like(z) + coeffs[0]
        for power in range(1, scaling_order + 1):
            prediction = prediction + coeffs[power] * np.power(z, power)
        return (prediction - y) / sigma

    width = upper_bound - lower_bound
    threshold_guesses = [
        lower_bound + 0.5 * width,
        lower_bound + 0.25 * width,
        lower_bound + 0.75 * width,
    ]
    nu_guesses = [1.0, 1.5, 2.5]
    best: Any | None = None
    for threshold_guess in threshold_guesses:
        for nu_guess in nu_guesses:
            initial = np.zeros(scaling_order + 3, dtype=float)
            initial[0] = threshold_guess
            initial[1] = nu_guess
            initial[2] = y_mean
            lower = np.asarray(
                [lower_bound, 0.1, *([-np.inf] * (scaling_order + 1))],
                dtype=float,
            )
            upper = np.asarray(
                [upper_bound, 10.0, *([np.inf] * (scaling_order + 1))],
                dtype=float,
            )
            result = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                max_nfev=2000,
            )
            if result.success and (best is None or result.cost < best.cost):
                best = result
    if best is None:
        return None

    parameters = best.x
    threshold = float(parameters[0])
    nu = float(parameters[1])
    coefficients = tuple(float(value) for value in parameters[2:])
    chi_squared = float(2.0 * best.cost)
    return threshold, nu, coefficients, chi_squared


def _bootstrap_points(points: tuple[ThresholdPoint, ...], rng: Any) -> tuple[ThresholdPoint, ...]:
    boot_points: list[ThresholdPoint] = []
    for point in points:
        errors = int(rng.binomial(point.shots, _continuity_corrected_rate(point)))
        boot_points.append(
            _make_point(
                x=point.x,
                distance=point.distance,
                shots=point.shots,
                errors=errors,
            )
        )
    return tuple(boot_points)


def _estimate_from_value(
    value: float,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_values: Iterable[float],
) -> ThresholdEstimate:
    values = sorted(float(candidate) for candidate in bootstrap_values)
    ci_low = None
    ci_high = None
    successes = len(values)
    if bootstrap_samples and successes >= _minimum_bootstrap_successes(bootstrap_samples):
        alpha = 1.0 - confidence_level
        ci_low = _quantile(values, alpha / 2.0)
        ci_high = _quantile(values, 1.0 - alpha / 2.0)
    return ThresholdEstimate(
        value=float(value),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=float(confidence_level),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_successes=int(successes),
    )


def _minimum_bootstrap_successes(samples: int) -> int:
    return max(40, math.ceil(samples / 2.0))


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires values")
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _dedupe_sorted(values: Iterable[float]) -> list[float]:
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not result or not math.isclose(value, result[-1], rel_tol=0.0, abs_tol=1e-12):
            result.append(value)
    return result


def _logit(rate: float) -> float:
    return math.log(rate / (1.0 - rate))


def _logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _is_close_zero(value: float) -> bool:
    return math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)


def _canonical_json(value: object) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _json_safe(value: object) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "__dataclass_fields__"):
        return {field: _json_safe(getattr(value, field)) for field in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "threshold analysis requires optional dependencies; install faultscope[collection]"
        ) from exc
    return np


def _import_least_squares() -> Any:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError(
            "threshold analysis requires optional dependencies; install faultscope[collection]"
        ) from exc
    return least_squares


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
            "plot_threshold_analysis requires matplotlib; install faultscope[collection]"
        ) from exc
    return plt

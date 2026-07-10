from __future__ import annotations

import builtins
import dataclasses
import importlib
import json
import math
from pathlib import Path
from typing import Literal, get_type_hints
from unittest import mock

import pytest

from faultscope.collection._types import TaskStats


def stat(
    *,
    x: float,
    distance: int,
    errors: int,
    shots: int = 1000,
    discards: int = 0,
    decoder: str | None = "mwpm",
    custom_counts: dict[str, int] | None = None,
    **metadata: object,
) -> TaskStats:
    return TaskStats(
        task_id=f"{distance}-{x}-{errors}-{shots}-{discards}",
        shots=shots,
        errors=errors,
        discards=discards,
        seconds=0.0,
        decoder=decoder,
        metadata={"p": x, "d": distance, **metadata},
        custom_counts=custom_counts or {},
    )


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def make_scaling_stats(
    *,
    threshold: float = 0.031,
    nu: float = 1.4,
    shots: int = 8000,
) -> list[TaskStats]:
    stats: list[TaskStats] = []
    for distance in (5, 7, 9, 11):
        for x in (0.021, 0.026, 0.031, 0.036, 0.041):
            z = (x - threshold) * distance ** (1.0 / nu)
            rate = logistic(-1.1 + 52.0 * z + 4.0 * z * z)
            stats.append(
                stat(
                    x=x,
                    distance=distance,
                    errors=round(rate * shots),
                    shots=shots,
                    decoder="mwpm",
                )
            )
    return stats


def test_collection_exports_threshold_api_without_top_level_exports() -> None:
    import faultscope
    import faultscope.collection as collection
    import faultscope.collection.threshold as threshold

    exported = [
        "FiniteSizeScalingFit",
        "PairwiseCrossing",
        "ThresholdAnalysisResult",
        "ThresholdEstimate",
        "ThresholdPoint",
        "analyze_thresholds",
        "plot_threshold_analysis",
    ]
    for name in exported:
        assert getattr(collection, name) is getattr(threshold, name)
        assert not hasattr(faultscope, name)


def test_threshold_module_all_controls_wildcard_api() -> None:
    import faultscope.collection.threshold as threshold

    assert threshold.__all__ == [
        "ThresholdPoint",
        "ThresholdEstimate",
        "PairwiseCrossing",
        "FiniteSizeScalingFit",
        "ThresholdAnalysisResult",
        "analyze_thresholds",
        "plot_threshold_analysis",
    ]


def test_public_dataclass_fields_are_frozen_and_ordered() -> None:
    from faultscope.collection.threshold import (
        FiniteSizeScalingFit,
        PairwiseCrossing,
        ThresholdAnalysisResult,
        ThresholdEstimate,
        ThresholdPoint,
    )

    assert dataclasses.fields(ThresholdPoint)
    assert [field.name for field in dataclasses.fields(ThresholdPoint)] == [
        "x",
        "distance",
        "shots",
        "errors",
        "rate",
        "stderr",
    ]
    assert [field.name for field in dataclasses.fields(ThresholdEstimate)] == [
        "value",
        "ci_low",
        "ci_high",
        "confidence_level",
        "bootstrap_samples",
        "bootstrap_successes",
    ]
    assert [field.name for field in dataclasses.fields(PairwiseCrossing)] == [
        "lower_distance",
        "upper_distance",
        "status",
        "candidates",
        "estimate",
    ]
    assert [field.name for field in dataclasses.fields(FiniteSizeScalingFit)] == [
        "status",
        "threshold",
        "critical_exponent",
        "coefficients",
        "reduced_chi_squared",
        "message",
    ]
    assert [field.name for field in dataclasses.fields(ThresholdAnalysisResult)] == [
        "series",
        "points",
        "crossings",
        "pairwise_threshold",
        "scaling_fit",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ThresholdPoint(1.0, 3, 10, 2, 0.2, 0.1).rate = 0.4


def test_public_annotations_match_approved_api() -> None:
    import faultscope.collection.threshold as threshold

    pairwise_hints = get_type_hints(threshold.PairwiseCrossing)
    scaling_hints = get_type_hints(threshold.FiniteSizeScalingFit)
    analyze_hints = get_type_hints(threshold.analyze_thresholds)

    assert pairwise_hints["status"] == Literal["ok", "no_crossing", "ambiguous"]
    assert (
        scaling_hints["status"]
        == Literal["ok", "insufficient_data", "fit_failed", "bootstrap_unstable"]
    )
    assert scaling_hints["threshold"] == threshold.ThresholdEstimate | None
    assert scaling_hints["critical_exponent"] == threshold.ThresholdEstimate | None
    assert scaling_hints["message"] == str | None
    assert analyze_hints["scaling_order"] == Literal[1, 2, 3]


def test_duplicate_points_aggregate_accepted_shots_and_raw_rates() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    (result,) = analyze_thresholds(
        [
            stat(x=0.01, distance=3, shots=100, discards=10, errors=9),
            stat(x=0.01, distance=3, shots=50, discards=5, errors=6),
        ],
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    assert len(result.points) == 1
    point = result.points[0]
    assert point.shots == 135
    assert point.errors == 15
    assert point.rate == pytest.approx(15.0 / 135.0)
    assert point.stderr == pytest.approx(math.sqrt(point.rate * (1.0 - point.rate) / point.shots))


def test_custom_count_key_is_strict_and_uses_accepted_shots_denominator() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    (result,) = analyze_thresholds(
        [
            stat(
                x=0.01,
                distance=3,
                shots=200,
                discards=20,
                errors=99,
                custom_counts={"logical_x": 18},
            )
        ],
        x_key="p",
        distance_key="d",
        count_key="logical_x",
        bootstrap_samples=0,
    )

    assert result.points[0].shots == 180
    assert result.points[0].errors == 18
    with pytest.raises(ValueError, match="custom count"):
        analyze_thresholds(
            [stat(x=0.01, distance=3, shots=100, errors=4)],
            x_key="p",
            distance_key="d",
            count_key="logical_x",
            bootstrap_samples=0,
        )


def test_series_partitioning_decoder_key_and_json_safe_to_dict() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    stats = [
        stat(x=0.01, distance=3, errors=1, decoder="b", label={"name": "beta"}),
        stat(x=0.02, distance=3, errors=2, decoder="a", label={"name": "alpha"}),
    ]
    results = analyze_thresholds(
        reversed(stats),
        x_key="p",
        distance_key="d",
        series_keys=("decoder", "label"),
        bootstrap_samples=0,
    )

    assert [result.series["decoder"] for result in results] == ["a", "b"]
    as_dict = results[0].to_dict()
    json.dumps(as_dict)
    assert as_dict["series"] == {"decoder": "a", "label": {"name": "alpha"}}
    assert isinstance(as_dict["points"], list)


def test_to_dict_preserves_series_and_nested_mapping_order() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    (result,) = analyze_thresholds(
        [
            stat(
                x=0.01,
                distance=3,
                errors=1,
                decoder="mwpm",
                alpha={"z": 1, "a": 2},
                beta=3,
            )
        ],
        x_key="p",
        distance_key="d",
        series_keys=("beta", "alpha", "decoder"),
        bootstrap_samples=0,
    )

    series = result.to_dict()["series"]
    assert list(series) == ["beta", "alpha", "decoder"]
    assert list(series["alpha"]) == ["z", "a"]


def test_pairwise_crossings_handle_unique_none_multiple_and_unequal_grids() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    stats = [
        stat(x=0.01, distance=3, errors=110),
        stat(x=0.02, distance=3, errors=190),
        stat(x=0.03, distance=3, errors=350),
        stat(x=0.04, distance=3, errors=420),
        stat(x=0.015, distance=5, errors=150),
        stat(x=0.025, distance=5, errors=250),
        stat(x=0.035, distance=5, errors=330),
        stat(x=0.01, distance=7, errors=400),
        stat(x=0.04, distance=7, errors=450),
        stat(x=0.01, distance=9, errors=100),
        stat(x=0.02, distance=9, errors=600),
        stat(x=0.03, distance=9, errors=150),
        stat(x=0.04, distance=9, errors=700),
    ]

    (result,) = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    by_pair = {
        (crossing.lower_distance, crossing.upper_distance): crossing
        for crossing in result.crossings
    }
    assert by_pair[(3.0, 5.0)].status == "ok"
    assert by_pair[(3.0, 5.0)].estimate.value == pytest.approx(0.022, abs=0.002)
    assert by_pair[(5.0, 7.0)].status == "no_crossing"
    assert by_pair[(7.0, 9.0)].status == "ambiguous"
    assert len(by_pair[(7.0, 9.0)].candidates) > 1
    for crossing in result.crossings:
        assert all(0.01 <= candidate <= 0.04 for candidate in crossing.candidates)
    all_candidates = sorted(
        {candidate for crossing in result.crossings for candidate in crossing.candidates}
    )
    assert result.pairwise_threshold.value == pytest.approx(
        0.5 * (all_candidates[1] + all_candidates[2])
    )


def test_ambiguous_pair_contributes_all_candidates_to_global_median() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    stats = [
        stat(x=0.01, distance=3, errors=100),
        stat(x=0.02, distance=3, errors=400),
        stat(x=0.03, distance=3, errors=100),
        stat(x=0.04, distance=3, errors=400),
        stat(x=0.01, distance=5, errors=300),
        stat(x=0.02, distance=5, errors=200),
        stat(x=0.03, distance=5, errors=300),
        stat(x=0.04, distance=5, errors=200),
    ]

    (result,) = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    crossing = result.crossings[0]
    assert crossing.status == "ambiguous"
    assert crossing.estimate is None
    assert len(crossing.candidates) == 3
    assert result.pairwise_threshold is not None
    assert result.pairwise_threshold.value == pytest.approx(crossing.candidates[1])


def test_global_pairwise_candidates_are_deduplicated_before_median() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    rates_by_distance = {
        3: (100, 200, 400, 500),
        5: (50, 200, 500, 600),
        7: (30, 200, 600, 700),
        9: (100, 300, 600, 600),
    }
    stats = [
        stat(x=x, distance=distance, errors=errors)
        for distance, errors_by_x in rates_by_distance.items()
        for x, errors in zip((0.01, 0.02, 0.04, 0.05), errors_by_x)
    ]

    (result,) = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    assert len(result.crossings) == 3
    for crossing, expected in zip(result.crossings, (0.02, 0.02, 0.04)):
        assert crossing.candidates == pytest.approx((expected,))
    assert result.pairwise_threshold is not None
    assert result.pairwise_threshold.value == pytest.approx(0.03)


def test_single_shared_endpoint_crossing_is_detected_only_when_logits_match() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    equal_endpoint = [
        stat(x=0.01, distance=3, errors=100),
        stat(x=0.02, distance=3, errors=200),
        stat(x=0.02, distance=5, errors=200),
        stat(x=0.03, distance=5, errors=300),
    ]
    (equal_result,) = analyze_thresholds(
        equal_endpoint,
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    assert equal_result.crossings[0].status == "ok"
    assert equal_result.crossings[0].candidates == pytest.approx((0.02,))

    unequal_endpoint = [
        stat(x=0.01, distance=3, errors=100),
        stat(x=0.02, distance=3, errors=200),
        stat(x=0.02, distance=5, errors=210),
        stat(x=0.03, distance=5, errors=300),
    ]
    (unequal_result,) = analyze_thresholds(
        unequal_endpoint,
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    assert unequal_result.crossings[0].status == "no_crossing"
    assert unequal_result.crossings[0].candidates == ()


@pytest.mark.parametrize("order", [1, 2, 3])
def test_scaling_fit_recovers_known_synthetic_threshold_and_nu(order: int) -> None:
    from faultscope.collection.threshold import analyze_thresholds

    (result,) = analyze_thresholds(
        make_scaling_stats(),
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
        scaling_order=order,
    )

    assert result.scaling_fit.status == "ok"
    assert result.scaling_fit.threshold.value == pytest.approx(0.031, abs=0.003)
    assert result.scaling_fit.critical_exponent.value == pytest.approx(1.4, abs=0.45)
    assert result.scaling_fit.message is None
    assert len(result.scaling_fit.coefficients) == order + 1
    assert math.isfinite(result.scaling_fit.reduced_chi_squared)


def test_scaling_fit_reports_insufficient_data_without_raising() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    (result,) = analyze_thresholds(
        [
            stat(x=0.01, distance=3, errors=10),
            stat(x=0.02, distance=3, errors=20),
            stat(x=0.01, distance=5, errors=15),
            stat(x=0.02, distance=5, errors=25),
        ],
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    assert result.scaling_fit.status == "insufficient_data"
    assert result.scaling_fit.threshold is None


def test_public_rates_are_raw_while_zero_and_one_internal_rates_remain_finite() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    (result,) = analyze_thresholds(
        [
            stat(x=0.01, distance=3, errors=0, shots=10),
            stat(x=0.02, distance=3, errors=10, shots=10),
            stat(x=0.01, distance=5, errors=10, shots=10),
            stat(x=0.02, distance=5, errors=0, shots=10),
        ],
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    assert [point.rate for point in result.points] == [0.0, 1.0, 1.0, 0.0]
    assert [point.stderr for point in result.points] == [0.0, 0.0, 0.0, 0.0]
    assert all(math.isfinite(point.stderr) for point in result.points)
    assert result.crossings[0].status == "ok"
    assert math.isfinite(result.crossings[0].candidates[0])

    (bootstrapped,) = analyze_thresholds(
        [
            stat(x=0.01, distance=3, errors=0, shots=10),
            stat(x=0.02, distance=3, errors=10, shots=10),
            stat(x=0.01, distance=5, errors=10, shots=10),
            stat(x=0.02, distance=5, errors=0, shots=10),
        ],
        x_key="p",
        distance_key="d",
        bootstrap_samples=40,
        seed=17,
    )
    assert bootstrapped.pairwise_threshold is not None
    assert math.isfinite(bootstrapped.pairwise_threshold.value)
    assert bootstrapped.pairwise_threshold.bootstrap_successes == 40
    assert math.isfinite(bootstrapped.pairwise_threshold.ci_low)
    assert math.isfinite(bootstrapped.pairwise_threshold.ci_high)


def test_scaling_rejects_any_distance_with_fewer_than_two_points() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    stats = make_scaling_stats()
    stats.append(stat(x=0.031, distance=13, errors=250, shots=1000))

    (result,) = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    assert result.scaling_fit.status == "insufficient_data"
    assert result.scaling_fit.threshold is None
    assert "each distance" in result.scaling_fit.message


def test_bootstrap_is_deterministic_input_order_independent_and_counts_successes() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    stats = make_scaling_stats(shots=1200)
    result_a = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        bootstrap_samples=48,
        seed=123,
    )
    result_b = analyze_thresholds(
        list(reversed(stats)),
        x_key="p",
        distance_key="d",
        bootstrap_samples=48,
        seed=123,
    )
    result_c = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        bootstrap_samples=48,
        seed=124,
    )

    assert result_a[0].to_dict() == result_b[0].to_dict()
    assert result_a[0].scaling_fit.threshold.bootstrap_samples == 48
    assert result_a[0].scaling_fit.threshold.bootstrap_successes >= 40
    assert result_a[0].scaling_fit.threshold.ci_low is not None
    assert result_a[0].scaling_fit.threshold.ci_high is not None
    assert result_a[0].scaling_fit.critical_exponent.bootstrap_samples == 48
    assert result_a[0].scaling_fit.critical_exponent.bootstrap_successes >= 40
    assert result_a[0].scaling_fit.critical_exponent.ci_low is not None
    assert result_a[0].scaling_fit.critical_exponent.ci_high is not None
    assert result_a[0].scaling_fit.threshold.value == result_c[0].scaling_fit.threshold.value
    assert (
        result_a[0].scaling_fit.critical_exponent.value
        == result_c[0].scaling_fit.critical_exponent.value
    )
    assert (
        result_a[0].scaling_fit.threshold.ci_low,
        result_a[0].scaling_fit.threshold.ci_high,
    ) != (
        result_c[0].scaling_fit.threshold.ci_low,
        result_c[0].scaling_fit.threshold.ci_high,
    )
    assert (
        result_a[0].scaling_fit.critical_exponent.ci_low,
        result_a[0].scaling_fit.critical_exponent.ci_high,
    ) != (
        result_c[0].scaling_fit.critical_exponent.ci_low,
        result_c[0].scaling_fit.critical_exponent.ci_high,
    )


def test_unique_pairwise_crossings_receive_pair_specific_bootstrap_metadata() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    stats = [
        stat(x=0.01, distance=3, errors=480, shots=10000),
        stat(x=0.02, distance=3, errors=840, shots=10000),
        stat(x=0.03, distance=3, errors=1560, shots=10000),
        stat(x=0.04, distance=3, errors=1920, shots=10000),
        stat(x=0.01, distance=5, errors=680, shots=10000),
        stat(x=0.02, distance=5, errors=960, shots=10000),
        stat(x=0.03, distance=5, errors=1360, shots=10000),
        stat(x=0.04, distance=5, errors=1800, shots=10000),
        stat(x=0.01, distance=7, errors=880, shots=10000),
        stat(x=0.02, distance=7, errors=1080, shots=10000),
        stat(x=0.03, distance=7, errors=1280, shots=10000),
        stat(x=0.04, distance=7, errors=1680, shots=10000),
    ]

    (result,) = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        bootstrap_samples=48,
        seed=321,
    )

    unique_crossings = [crossing for crossing in result.crossings if crossing.status == "ok"]
    assert len(unique_crossings) == 2
    assert result.pairwise_threshold.bootstrap_samples == 48
    assert result.pairwise_threshold.bootstrap_successes >= 40
    for crossing in unique_crossings:
        assert crossing.estimate.bootstrap_samples == 48
        assert crossing.estimate.bootstrap_successes >= 40
        assert crossing.estimate.ci_low is not None
        assert crossing.estimate.ci_high is not None


def test_validation_rejects_bad_inputs() -> None:
    from faultscope.collection.threshold import analyze_thresholds

    with pytest.raises(ValueError, match="x"):
        analyze_thresholds(
            [stat(x=math.nan, distance=3, errors=1)],
            x_key="p",
            distance_key="d",
        )
    with pytest.raises(ValueError, match="distance"):
        analyze_thresholds(
            [stat(x=0.01, distance=0, errors=1)],
            x_key="p",
            distance_key="d",
        )
    with pytest.raises(ValueError, match="accepted shots"):
        analyze_thresholds(
            [stat(x=0.01, distance=3, shots=10, discards=10, errors=0)],
            x_key="p",
            distance_key="d",
        )
    with pytest.raises(ValueError, match="count"):
        analyze_thresholds(
            [stat(x=0.01, distance=3, shots=10, errors=11)],
            x_key="p",
            distance_key="d",
        )
    with pytest.raises(ValueError, match="x"):
        analyze_thresholds(
            [stat(x=True, distance=3, errors=1)],
            x_key="p",
            distance_key="d",
        )
    with pytest.raises(ValueError, match="count"):
        analyze_thresholds(
            [
                stat(
                    x=0.01,
                    distance=3,
                    shots=10,
                    errors=1,
                    custom_counts={"logical_x": True},
                )
            ],
            x_key="p",
            distance_key="d",
            count_key="logical_x",
        )
    with pytest.raises(ValueError, match="bootstrap_samples"):
        analyze_thresholds(
            [stat(x=0.01, distance=3, errors=1)],
            x_key="p",
            distance_key="d",
            bootstrap_samples=-1,
        )
    with pytest.raises(ValueError, match="confidence_level"):
        analyze_thresholds(
            [stat(x=0.01, distance=3, errors=1)],
            x_key="p",
            distance_key="d",
            confidence_level=1.0,
        )
    with pytest.raises(ValueError, match="scaling_order"):
        analyze_thresholds(
            [stat(x=0.01, distance=3, errors=1)],
            x_key="p",
            distance_key="d",
            scaling_order=4,
        )


def test_threshold_module_imports_without_numpy_or_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"numpy", "scipy"} or name.startswith(("numpy.", "scipy.")):
            raise AssertionError(f"eager optional import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module("faultscope.collection.threshold")


def test_missing_numpy_or_scipy_dependency_error_mentions_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import faultscope.collection.threshold as threshold

    real_import = builtins.__import__

    def missing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("blocked numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    with pytest.raises(ImportError, match=r"faultscope\[collection\]"):
        threshold.analyze_thresholds(
            make_scaling_stats(),
            x_key="p",
            distance_key="d",
            bootstrap_samples=1,
        )


def test_plot_threshold_analysis_single_series_and_output_png(tmp_path: Path) -> None:
    from faultscope.collection.threshold import analyze_thresholds, plot_threshold_analysis

    (result,) = analyze_thresholds(
        make_scaling_stats(),
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )
    output = tmp_path / "threshold.png"

    figure, axes = plot_threshold_analysis(result, output=output)

    assert output.read_bytes().startswith(b"\x89PNG")
    assert axes.shape == (1, 2)
    assert axes[0, 0].figure is figure
    assert axes[0, 0].get_yscale() == "log"
    assert axes[0, 0].get_xlabel() == "physical error rate"
    assert axes[0, 1].get_xlabel() == "scaled distance variable"
    assert len(axes[0, 0].lines) >= 4
    assert len(axes[0, 1].lines) >= 1


def test_plot_threshold_analysis_keeps_zero_rates_visible_with_log_y() -> None:
    from faultscope.collection.threshold import analyze_thresholds, plot_threshold_analysis

    (result,) = analyze_thresholds(
        [
            stat(x=0.01, distance=3, errors=0, shots=100),
            stat(x=0.02, distance=3, errors=0, shots=100),
            stat(x=0.01, distance=5, errors=0, shots=100),
            stat(x=0.02, distance=5, errors=0, shots=100),
        ],
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    _, axes = plot_threshold_analysis(result)

    assert axes[0, 0].get_yscale() == "symlog"
    rate_lines = [line for line in axes[0, 0].lines if line.get_marker() == "o"]
    assert rate_lines
    assert any(0.0 in line.get_ydata() for line in rate_lines)


def test_plot_threshold_analysis_multiple_series_and_supplied_axes() -> None:
    from matplotlib import pyplot as plt

    from faultscope.collection.threshold import analyze_thresholds, plot_threshold_analysis

    stats = [
        *make_scaling_stats(threshold=0.029),
        *[
            stat(
                x=sample.metadata["p"],
                distance=sample.metadata["d"],
                errors=sample.errors + 10,
                shots=sample.shots,
                decoder="bposd",
            )
            for sample in make_scaling_stats(threshold=0.033)
        ],
    ]
    results = analyze_thresholds(
        stats,
        x_key="p",
        distance_key="d",
        series_keys=("decoder",),
        bootstrap_samples=0,
    )
    figure, supplied_axes = plt.subplots(2, 2, squeeze=False)

    returned_figure, returned_axes = plot_threshold_analysis(
        results,
        axes=supplied_axes,
        log_y=False,
    )

    assert returned_figure is figure
    assert returned_axes is supplied_axes
    assert returned_axes.shape == (2, 2)
    assert all(axis.get_yscale() == "linear" for axis in returned_axes[:, 0])
    assert {axis.get_title() for axis in returned_axes[:, 0]} == {
        "decoder=bposd",
        "decoder=mwpm",
    }


def test_plot_threshold_analysis_pairwise_marker_uses_crossing_rate() -> None:
    from faultscope.collection.threshold import analyze_thresholds, plot_threshold_analysis

    (result,) = analyze_thresholds(
        [
            stat(x=0.01, distance=3, errors=100),
            stat(x=0.03, distance=3, errors=500),
            stat(x=0.01, distance=5, errors=500),
            stat(x=0.03, distance=5, errors=100),
        ],
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    _, axes = plot_threshold_analysis(result, log_y=False)
    crossing_marker = next(
        line for line in axes[0, 0].lines if line.get_marker() == "x" and len(line.get_xdata()) == 1
    )

    assert crossing_marker.get_xdata()[0] == pytest.approx(0.02)
    low_rate = (100.0 + 0.5) / (1000.0 + 1.0)
    high_rate = (500.0 + 0.5) / (1000.0 + 1.0)
    low_logit = math.log(low_rate / (1.0 - low_rate))
    high_logit = math.log(high_rate / (1.0 - high_rate))
    expected_marker_rate = logistic(0.5 * (low_logit + high_logit))
    assert crossing_marker.get_ydata()[0] == pytest.approx(expected_marker_rate)


def test_plot_threshold_analysis_validates_axes_shape_and_empty_results() -> None:
    from matplotlib import pyplot as plt

    from faultscope.collection.threshold import analyze_thresholds, plot_threshold_analysis

    (result,) = analyze_thresholds(
        make_scaling_stats(),
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )
    _, flat_axes = plt.subplots(1, 2)
    _, wrong_axes = plt.subplots(1, 3, squeeze=False)

    with pytest.raises(ValueError, match=r"shape \(1, 2\)"):
        plot_threshold_analysis(result, axes=flat_axes)
    with pytest.raises(ValueError, match=r"shape \(1, 2\)"):
        plot_threshold_analysis(result, axes=wrong_axes)
    with pytest.raises(ValueError, match="at least one"):
        plot_threshold_analysis([])


def test_plot_threshold_analysis_annotates_insufficient_fit_status() -> None:
    from faultscope.collection.threshold import analyze_thresholds, plot_threshold_analysis

    (result,) = analyze_thresholds(
        [
            stat(x=0.01, distance=3, errors=10),
            stat(x=0.02, distance=3, errors=20),
            stat(x=0.01, distance=5, errors=15),
            stat(x=0.02, distance=5, errors=25),
        ],
        x_key="p",
        distance_key="d",
        bootstrap_samples=0,
    )

    _, axes = plot_threshold_analysis(result)

    assert len(axes[0, 1].lines) == 0
    assert any("insufficient_data" in text.get_text() for text in axes[0, 1].texts)


def test_plot_threshold_analysis_missing_matplotlib_has_install_hint() -> None:
    import faultscope.collection.threshold as threshold

    original_import = builtins.__import__

    def import_hook(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("mocked missing matplotlib")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=import_hook):
        with pytest.raises(ImportError, match=r"faultscope\[collection\]"):
            threshold.plot_threshold_analysis(
                threshold.analyze_thresholds(
                    make_scaling_stats(),
                    x_key="p",
                    distance_key="d",
                    bootstrap_samples=0,
                )
            )

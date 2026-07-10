"""Surface-code threshold comparison benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/surface_code_threshold.py

The benchmark compares logical failure estimates from four paths on Stim
standard rotated surface-code memory circuits:

* Stim circuit detector sampling + PyMatching.
* Stim DEM sampling + PyMatching.
* FaultScope native forward sampling + PyMatching.
* FaultScope native DEM sampling + PyMatching.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faultscope.dem import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.collection import (
    FiniteSizeScalingFit,
    PairwiseCrossing,
    TaskStats,
    ThresholdAnalysisResult,
    ThresholdEstimate,
    analyze_thresholds,
)
from faultscope.runtime import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    compile_native_sampler,
)
from faultscope.runtime.loss import logical_residual_loss_mask
from faultscope.io import StimImportResult, parse_stim_circuit


DEFAULT_RATES = (0.002, 0.004, 0.006, 0.008, 0.010, 0.012)
PATHS = ("stim", "stim-dem", "faultscope-forward", "faultscope-dem")


@dataclass(frozen=True)
class LogicalFailureStats:
    shots: int
    failures: int

    @property
    def rate(self) -> float:
        return self.failures / self.shots

    @property
    def stderr(self) -> float:
        p = self.rate
        return math.sqrt(p * (1.0 - p) / self.shots)


@dataclass(frozen=True)
class TimedLogicalFailureStats:
    stats: LogicalFailureStats
    compile_s: float
    sample_s: float
    decode_s: float

    @property
    def total_s(self) -> float:
        return self.compile_s + self.sample_s + self.decode_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--basis", choices=("x", "z"), default="x")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    args = parser.parse_args()

    stim, pymatching, np = _load_required_modules()
    distances = tuple(sorted(set(args.distances)))
    rates = tuple(sorted(set(float(rate) for rate in args.rates)))
    if not distances:
        raise SystemExit("at least one distance is required")
    if not rates:
        raise SystemExit("at least one physical error rate is required")
    if args.shots <= 0:
        raise SystemExit("shots must be positive")
    if args.bootstrap_samples < 0:
        raise SystemExit("bootstrap samples must be non-negative")

    print(
        "basis\tdistance\trounds\tp\tshots\tpath\tlogical_failure\t"
        "failures\tstderr\tcompile_s\tsample_s\tdecode_s\ttotal_s\tstatus",
        flush=True,
    )

    threshold_stats: list[TaskStats] = []
    timings_by_path: dict[str, list[TimedLogicalFailureStats]] = {
        path: [] for path in PATHS
    }
    for distance in distances:
        rounds = args.rounds if args.rounds is not None else distance
        for rate_index, p in enumerate(rates):
            circuit = stim.Circuit.generated(
                code_task=f"surface_code:rotated_memory_{args.basis}",
                distance=distance,
                rounds=rounds,
                after_clifford_depolarization=p,
            )
            stim_dem = circuit.detector_error_model(
                decompose_errors=True,
                flatten_loops=True,
            )
            matcher = pymatching.Matching.from_detector_error_model(stim_dem)
            imported = parse_stim_circuit(str(circuit.flattened()))
            faultscope_dem = stim_dem_to_faultscope_dem(stim_dem)

            path_fns = (
                (
                    "stim",
                    lambda seed: timed_stim_logical_failure(
                        circuit,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
                (
                    "stim-dem",
                    lambda seed: timed_stim_dem_logical_failure(
                        stim_dem,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
                (
                    "faultscope-forward",
                    lambda seed: timed_faultscope_forward_logical_failure(
                        imported,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
                (
                    "faultscope-dem",
                    lambda seed: timed_faultscope_dem_logical_failure(
                        faultscope_dem,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
            )
            for path_index, (path, fn) in enumerate(path_fns):
                seed = args.seed + 1_000_000 * distance + 1_000 * rate_index + path_index
                started = time.perf_counter()
                try:
                    timed_stats = fn(seed)
                except (ImportError, UnsupportedNativeCircuitError, ValueError) as exc:
                    elapsed_s = time.perf_counter() - started
                    print(
                        f"{args.basis}\t{distance}\t{rounds}\t{p:.17g}\t"
                        f"{args.shots}\t{path}\tNA\tNA\tNA\t"
                        f"NA\tNA\tNA\t{elapsed_s:.6f}\t"
                        f"skip:{type(exc).__name__}",
                        flush=True,
                    )
                    continue
                stats = timed_stats.stats
                task_id = (
                    f"surface-code:{args.basis}:{path}:"
                    f"d={distance}:p={p:.17g}"
                )
                threshold_stats.append(
                    TaskStats(
                        task_id=task_id,
                        strong_id=task_id,
                        shots=stats.shots,
                        errors=stats.failures,
                        discards=0,
                        seconds=timed_stats.total_s,
                        decoder=path,
                        metadata={
                            "basis": args.basis,
                            "path": path,
                            "distance": distance,
                            "p": p,
                        },
                    )
                )
                timings_by_path[path].append(timed_stats)
                print(
                    f"{args.basis}\t{distance}\t{rounds}\t{p:.17g}\t"
                    f"{args.shots}\t{path}\t{stats.rate:.17g}\t"
                    f"{stats.failures}\t{stats.stderr:.17g}\t"
                    f"{timed_stats.compile_s:.6f}\t{timed_stats.sample_s:.6f}\t"
                    f"{timed_stats.decode_s:.6f}\t{timed_stats.total_s:.6f}\tok",
                    flush=True,
                )

    threshold_results = analyze_thresholds(
        threshold_stats,
        x_key="p",
        distance_key="distance",
        series_keys=("path", "basis"),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _print_threshold_results_for_paths(
        threshold_results,
        basis=args.basis,
        paths=PATHS,
        distances=distances,
    )

    for path in PATHS:
        timed_values = timings_by_path[path]
        if not timed_values:
            print(f"timing\t{args.basis}\t{path}\t0\tNA\tNA\tNA\tNA\tNA", flush=True)
            continue
        compile_s = sum(value.compile_s for value in timed_values)
        sample_s = sum(value.sample_s for value in timed_values)
        decode_s = sum(value.decode_s for value in timed_values)
        total_s = sum(value.total_s for value in timed_values)
        average_total_s = total_s / len(timed_values)
        print(
            f"timing\t{args.basis}\t{path}\t{len(timed_values)}\t"
            f"{compile_s:.6f}\t{sample_s:.6f}\t{decode_s:.6f}\t"
            f"{total_s:.6f}\t{average_total_s:.6f}",
            flush=True,
        )


def timed_stim_logical_failure(
    circuit: Any,
    matcher: Any,
    shots: int,
    seed: int,
) -> TimedLogicalFailureStats:
    started = time.perf_counter()
    sampler = circuit.compile_detector_sampler(seed=seed)
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    detectors, observables = sampler.sample(
        shots,
        separate_observables=True,
    )
    sample_s = time.perf_counter() - started
    started = time.perf_counter()
    stats = _logical_failure_stats_from_arrays(detectors, observables, matcher, shots)
    decode_s = time.perf_counter() - started
    return TimedLogicalFailureStats(stats, compile_s, sample_s, decode_s)


def timed_stim_dem_logical_failure(
    stim_dem: Any,
    matcher: Any,
    shots: int,
    seed: int,
) -> TimedLogicalFailureStats:
    started = time.perf_counter()
    sampler = stim_dem.compile_sampler(seed=seed)
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    detectors, observables, _ = sampler.sample(shots)
    sample_s = time.perf_counter() - started
    started = time.perf_counter()
    stats = _logical_failure_stats_from_arrays(detectors, observables, matcher, shots)
    decode_s = time.perf_counter() - started
    return TimedLogicalFailureStats(stats, compile_s, sample_s, decode_s)


def timed_faultscope_forward_logical_failure(
    imported: StimImportResult,
    matcher: Any,
    shots: int,
    seed: int,
) -> TimedLogicalFailureStats:
    started = time.perf_counter()
    sampler = compile_native_sampler(imported.circuit)
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    batch = sampler.sample(shots=shots, seed=seed)
    sample_s = time.perf_counter() - started
    started = time.perf_counter()
    detector_ids = tuple(detector.id for detector in imported.detectors)
    observable_ids = tuple(observable.id for observable in imported.observables)
    corrections = _decode_batch_masks_with_matching(
        matcher,
        batch.detectors,
        detector_ids,
        observable_ids,
        shots,
    )
    stats = _logical_failure_stats_from_masks(
        batch.observables,
        corrections,
        observable_ids,
        shots,
    )
    decode_s = time.perf_counter() - started
    return TimedLogicalFailureStats(stats, compile_s, sample_s, decode_s)


def timed_faultscope_dem_logical_failure(
    faultscope_dem: DetectorErrorModel,
    matcher: Any,
    shots: int,
    seed: int,
) -> TimedLogicalFailureStats:
    started = time.perf_counter()
    sampler = compile_native_dem_sampler(faultscope_dem)
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    batch = sampler.run_batch(
        shots=shots,
        seed=seed,
        return_edge_events=False,
    )
    sample_s = time.perf_counter() - started
    started = time.perf_counter()
    detector_ids = tuple(detector.id for detector in faultscope_dem.detectors)
    observable_ids = tuple(observable.id for observable in faultscope_dem.observables)
    corrections = _decode_batch_masks_with_matching(
        matcher,
        batch.detectors,
        detector_ids,
        observable_ids,
        shots,
    )
    stats = _logical_failure_stats_from_masks(
        batch.observables,
        corrections,
        observable_ids,
        shots,
    )
    decode_s = time.perf_counter() - started
    return TimedLogicalFailureStats(stats, compile_s, sample_s, decode_s)


def stim_dem_to_faultscope_dem(stim_dem: Any) -> DetectorErrorModel:
    detectors_by_id: dict[int, Detector] = {}
    observable_ids: set[int] = set()
    edges: list[DetectorErrorEdge] = []
    detector_offset = 0

    for instruction in stim_dem:
        instruction_type = instruction.type
        if instruction_type == "error":
            detectors: list[int] = []
            observables: list[int] = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors.append(detector_offset + int(target.val))
                elif target.is_logical_observable_id():
                    observable_id = int(target.val)
                    observables.append(observable_id)
                    observable_ids.add(observable_id)
                elif target.is_separator():
                    continue
                else:
                    raise ValueError(f"unsupported Stim DEM target {target!r}")
            edge_index = len(edges)
            edges.append(
                DetectorErrorEdge(
                    probability=float(instruction.args_copy()[0]),
                    detectors=tuple(detectors),
                    observables=tuple(observables),
                    location_id=f"stim_dem_edge_{edge_index}",
                    event=f"stim_dem_edge_{edge_index}",
                    tags={"source": "stim_dem"},
                )
            )
        elif instruction_type == "detector":
            coords = tuple(float(coord) for coord in instruction.args_copy())
            for target in instruction.targets_copy():
                if not target.is_relative_detector_id():
                    raise ValueError(f"unsupported detector target {target!r}")
                detector_id = detector_offset + int(target.val)
                detectors_by_id.setdefault(
                    detector_id,
                    Detector(
                        id=detector_id,
                        measurement_keys=(),
                        coords=coords,
                    ),
                )
        elif instruction_type == "shift_detectors":
            detector_offset += sum(int(target) for target in instruction.targets_copy())
        elif instruction_type == "logical_observable":
            for target in instruction.targets_copy():
                if not target.is_logical_observable_id():
                    raise ValueError(f"unsupported logical observable target {target!r}")
                observable_ids.add(int(target.val))
        else:
            raise ValueError(f"unsupported Stim DEM instruction {instruction_type!r}")

    for edge in edges:
        for detector_id in edge.detectors:
            detectors_by_id.setdefault(
                detector_id,
                Detector(id=detector_id, measurement_keys=()),
            )
        observable_ids.update(edge.observables)

    return DetectorErrorModel(
        detectors=tuple(detectors_by_id[key] for key in sorted(detectors_by_id)),
        observables=tuple(
            LogicalObservable(id=observable_id)
            for observable_id in sorted(observable_ids)
        ),
        edges=tuple(edges),
    )


def _logical_failure_stats_from_arrays(
    detectors: Any,
    observables: Any,
    matcher: Any,
    shots: int,
) -> LogicalFailureStats:
    corrections = matcher.decode_batch(detectors)
    corrections = _ensure_2d_bool_array(corrections, shots)
    observables = _ensure_2d_bool_array(observables, shots)
    if corrections.shape != observables.shape:
        raise ValueError(
            f"correction shape {corrections.shape} does not match "
            f"observable shape {observables.shape}"
        )
    residual = observables ^ corrections
    failures = int(residual.any(axis=1).sum())
    return LogicalFailureStats(shots=shots, failures=failures)


def _logical_failure_stats_from_masks(
    observable_masks: Mapping[int, int],
    correction_masks: Mapping[int, int],
    observable_ids: Iterable[int],
    shots: int,
) -> LogicalFailureStats:
    all_mask = (1 << shots) - 1
    residual_loss_mask = logical_residual_loss_mask(
        observable_masks,
        correction_masks,
        observable_ids=observable_ids,
        all_mask=all_mask,
    )
    failures = (residual_loss_mask & all_mask).bit_count()
    return LogicalFailureStats(shots=shots, failures=failures)


def _decode_batch_masks_with_matching(
    matcher: Any,
    detector_masks: Mapping[int, int],
    detector_ids: Iterable[int],
    observable_ids: Iterable[int],
    shots: int,
) -> dict[int, int]:
    detector_ids = tuple(detector_ids)
    observable_ids = tuple(observable_ids)
    packed_shots = _masks_to_packed_shots(detector_masks, detector_ids, shots)
    try:
        predictions = matcher.decode_batch(
            packed_shots,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        return _packed_predictions_to_masks(predictions, observable_ids, shots)
    except TypeError:
        dense_shots = _packed_shots_to_dense(packed_shots, len(detector_ids))
        predictions = matcher.decode_batch(dense_shots)
        return _dense_predictions_to_masks(predictions, observable_ids, shots)


def _masks_to_packed_shots(
    masks_by_id: Mapping[int, int],
    ids: Iterable[int],
    shots: int,
) -> Any:
    _, _, np = _load_required_modules()
    ids = tuple(ids)
    packed = np.zeros((shots, (len(ids) + 7) // 8), dtype=np.uint8)
    if shots == 0 or not ids:
        return packed
    byte_count = (shots + 7) // 8
    all_mask = (1 << shots) - 1
    for col, item_id in enumerate(ids):
        mask = int(masks_by_id.get(item_id, 0)) & all_mask
        shot_bits = np.unpackbits(
            np.frombuffer(mask.to_bytes(byte_count, "little"), dtype=np.uint8),
            bitorder="little",
        )[:shots]
        packed[:, col // 8] |= shot_bits.astype(np.uint8) << (col % 8)
    return packed


def _packed_shots_to_dense(packed: Any, detector_count: int) -> Any:
    _, _, np = _load_required_modules()
    if detector_count == 0:
        return np.zeros((packed.shape[0], 0), dtype=np.uint8)
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        bitorder="little",
        axis=1,
    )[:, :detector_count].astype(np.uint8, copy=False)


def _packed_predictions_to_masks(
    predictions: Any,
    observable_ids: Iterable[int],
    shots: int,
) -> dict[int, int]:
    _, _, np = _load_required_modules()
    observable_ids = tuple(observable_ids)
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape((shots, -1))
    if predictions.ndim != 2 or predictions.shape[0] != shots:
        raise ValueError(f"unexpected PyMatching prediction shape {predictions.shape}")
    out = {observable_id: 0 for observable_id in observable_ids}
    for col, observable_id in enumerate(observable_ids):
        if col // 8 >= predictions.shape[1]:
            raise ValueError(
                "PyMatching prediction width does not match observable count"
            )
        bits = (predictions[:, col // 8] >> (col % 8)) & 1
        out[observable_id] = int.from_bytes(
            np.packbits(bits.astype(np.uint8), bitorder="little").tobytes(),
            "little",
        )
    return out


def _dense_predictions_to_masks(
    predictions: Any,
    observable_ids: Iterable[int],
    shots: int,
) -> dict[int, int]:
    _, _, np = _load_required_modules()
    observable_ids = tuple(observable_ids)
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape((shots, 1))
    if predictions.ndim != 2 or predictions.shape[0] != shots:
        raise ValueError(f"unexpected PyMatching prediction shape {predictions.shape}")
    if predictions.shape[1] != len(observable_ids):
        raise ValueError("PyMatching prediction length does not match observable count")
    out = {observable_id: 0 for observable_id in observable_ids}
    for col, observable_id in enumerate(observable_ids):
        out[observable_id] = int.from_bytes(
            np.packbits(predictions[:, col].astype(np.uint8), bitorder="little").tobytes(),
            "little",
        )
    return out


def _ensure_2d_bool_array(value: Any, shots: int) -> Any:
    _, _, np = _load_required_modules()
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim == 1:
        array = array.reshape((shots, 1))
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")
    if array.shape[0] != shots:
        raise ValueError(f"expected {shots} shots, got shape {array.shape}")
    return array


def _print_threshold_results(results: Iterable[ThresholdAnalysisResult]) -> None:
    for result in results:
        basis = str(result.series["basis"])
        path = str(result.series["path"])
        for crossing in result.crossings:
            estimate = _estimate_cells(crossing.estimate)
            candidates = ",".join(f"{value:.17g}" for value in crossing.candidates)
            print(
                f"threshold-pairwise\t{basis}\t{path}\t"
                f"{crossing.lower_distance:g}-{crossing.upper_distance:g}\t"
                f"{crossing.status}\t{candidates or 'NA'}\t"
                f"{estimate[0]}\t{estimate[1]}\t{estimate[2]}\t"
                f"{estimate[3]}",
                flush=True,
            )

        pairwise = _estimate_cells(result.pairwise_threshold)
        print(
            f"threshold-pairwise-summary\t{basis}\t{path}\t"
            f"{pairwise[0]}\t{pairwise[1]}\t{pairwise[2]}\t{pairwise[3]}",
            flush=True,
        )

        fit = result.scaling_fit
        threshold = _estimate_cells(fit.threshold)
        exponent = _estimate_cells(fit.critical_exponent)
        reduced_chi_squared = (
            "NA"
            if fit.reduced_chi_squared is None
            else f"{fit.reduced_chi_squared:.17g}"
        )
        message = (
            "NA"
            if fit.message is None
            else fit.message.replace("\t", " ").replace("\n", " ")
        )
        print(
            f"threshold-scaling\t{basis}\t{path}\t{fit.status}\t"
            f"{threshold[0]}\t{threshold[1]}\t{threshold[2]}\t{threshold[3]}\t"
            f"{exponent[0]}\t{exponent[1]}\t{exponent[2]}\t{exponent[3]}\t"
            f"{reduced_chi_squared}\t{message}",
            flush=True,
        )


def _print_threshold_results_for_paths(
    results: Iterable[ThresholdAnalysisResult],
    *,
    basis: str,
    paths: Iterable[str],
    distances: Iterable[int],
) -> None:
    results_by_path = {str(result.series["path"]): result for result in results}
    distance_values = tuple(sorted(set(float(distance) for distance in distances)))
    for path in paths:
        result = results_by_path.get(path)
        if result is None:
            result = ThresholdAnalysisResult(
                series={"basis": basis, "path": path},
                points=(),
                crossings=(),
                pairwise_threshold=None,
                scaling_fit=FiniteSizeScalingFit(
                    status="insufficient_data",
                    threshold=None,
                    critical_exponent=None,
                    coefficients=(),
                    reduced_chi_squared=None,
                    message="no sampled data for path",
                ),
            )
        crossings_by_pair = {
            (crossing.lower_distance, crossing.upper_distance): crossing
            for crossing in result.crossings
        }
        expected_pairs = tuple(zip(distance_values, distance_values[1:]))
        normalized_crossings = tuple(
            crossings_by_pair.get(
                (lower, upper),
                PairwiseCrossing(
                    lower_distance=lower,
                    upper_distance=upper,
                    status="no_crossing",
                    candidates=(),
                    estimate=None,
                ),
            )
            for lower, upper in expected_pairs
        )
        observed_distances = {point.distance for point in result.points}
        if normalized_crossings != result.crossings or observed_distances != set(
            distance_values
        ):
            retained_candidates = sorted(
                {
                    candidate
                    for crossing in normalized_crossings
                    for candidate in crossing.candidates
                }
            )
            expected_pair_set = set(expected_pairs)
            all_original_pairs_retained = all(
                (crossing.lower_distance, crossing.upper_distance)
                in expected_pair_set
                for crossing in result.crossings
            )
            pairwise_threshold = None
            if all_original_pairs_retained:
                pairwise_threshold = result.pairwise_threshold
            retained_candidate_crossings = tuple(
                crossing for crossing in normalized_crossings if crossing.candidates
            )
            if (
                pairwise_threshold is None
                and len(retained_candidate_crossings) == 1
                and retained_candidate_crossings[0].estimate is not None
            ):
                pairwise_threshold = retained_candidate_crossings[0].estimate
            if retained_candidates and pairwise_threshold is None:
                midpoint = len(retained_candidates) // 2
                if len(retained_candidates) % 2:
                    value = retained_candidates[midpoint]
                else:
                    value = 0.5 * (
                        retained_candidates[midpoint - 1]
                        + retained_candidates[midpoint]
                    )
                confidence_level = (
                    result.pairwise_threshold.confidence_level
                    if result.pairwise_threshold is not None
                    else 0.95
                )
                bootstrap_samples = (
                    result.pairwise_threshold.bootstrap_samples
                    if result.pairwise_threshold is not None
                    else 0
                )
                pairwise_threshold = ThresholdEstimate(
                    value=value,
                    ci_low=None,
                    ci_high=None,
                    confidence_level=confidence_level,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_successes=0,
                )
            result = ThresholdAnalysisResult(
                series=result.series,
                points=result.points,
                crossings=normalized_crossings,
                pairwise_threshold=pairwise_threshold,
                scaling_fit=result.scaling_fit,
            )
        _print_threshold_results((result,))


def _estimate_cells(
    estimate: ThresholdEstimate | None,
) -> tuple[str, str, str, str]:
    if estimate is None:
        return "NA", "NA", "NA", "0"
    ci_low = "NA" if estimate.ci_low is None else f"{estimate.ci_low:.17g}"
    ci_high = "NA" if estimate.ci_high is None else f"{estimate.ci_high:.17g}"
    return (
        f"{estimate.value:.17g}",
        ci_low,
        ci_high,
        str(estimate.bootstrap_successes),
    )


def _load_required_modules() -> tuple[Any, Any, Any]:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "faultscope-matplotlib-cache"),
    )
    try:
        import stim
    except ImportError as exc:
        raise SystemExit("Stim is required for this benchmark") from exc
    try:
        import pymatching
    except ImportError as exc:
        raise SystemExit("PyMatching is required for this benchmark") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required for this benchmark") from exc
    return stim, pymatching, np


if __name__ == "__main__":
    main()

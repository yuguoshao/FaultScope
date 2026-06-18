"""Surface-code threshold comparison benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/surface_code_threshold.py

The benchmark compares logical failure estimates from four paths on Stim
standard rotated surface-code memory circuits:

* Stim circuit detector sampling + PyMatching.
* Stim DEM sampling + PyMatching.
* NPSim native forward sampling + PyMatching.
* NPSim native DEM sampling + PyMatching.
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

from npsim.dem import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from npsim.runtime import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    compile_native_sampler,
)
from npsim.runtime.loss import logical_residual_loss_mask
from npsim.io import StimImportResult, parse_stim_circuit


DEFAULT_RATES = (0.002, 0.004, 0.006, 0.008, 0.010, 0.012)
PATHS = ("stim", "stim-dem", "npsim-forward", "npsim-dem")


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

    print(
        "basis\tdistance\trounds\tp\tshots\tpath\tlogical_failure\t"
        "failures\tstderr\tcompile_s\tsample_s\tdecode_s\ttotal_s\tstatus",
        flush=True,
    )

    results: dict[str, dict[int, dict[float, float]]] = {
        path: {distance: {} for distance in distances}
        for path in PATHS
    }
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
            npsim_dem = stim_dem_to_npsim_dem(stim_dem)

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
                    "npsim-forward",
                    lambda seed: timed_npsim_forward_logical_failure(
                        imported,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
                (
                    "npsim-dem",
                    lambda seed: timed_npsim_dem_logical_failure(
                        npsim_dem,
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
                results[path][distance][p] = stats.rate
                timings_by_path[path].append(timed_stats)
                print(
                    f"{args.basis}\t{distance}\t{rounds}\t{p:.17g}\t"
                    f"{args.shots}\t{path}\t{stats.rate:.17g}\t"
                    f"{stats.failures}\t{stats.stderr:.17g}\t"
                    f"{timed_stats.compile_s:.6f}\t{timed_stats.sample_s:.6f}\t"
                    f"{timed_stats.decode_s:.6f}\t{timed_stats.total_s:.6f}\tok",
                    flush=True,
                )

    for path in PATHS:
        for small, large in zip(distances, distances[1:]):
            p_cross = _crossing_rate(results[path][small], results[path][large], rates)
            if p_cross is None:
                p_cell = "NA"
                status = "no-crossing"
            else:
                p_cell = f"{p_cross:.17g}"
                status = "ok"
            print(
                f"threshold\t{args.basis}\t{small}-{large}\t{path}\t"
                f"{p_cell}\t{status}",
                flush=True,
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


def timed_npsim_forward_logical_failure(
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


def timed_npsim_dem_logical_failure(
    npsim_dem: DetectorErrorModel,
    matcher: Any,
    shots: int,
    seed: int,
) -> TimedLogicalFailureStats:
    started = time.perf_counter()
    sampler = compile_native_dem_sampler(npsim_dem)
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    batch = sampler.run_batch(
        shots=shots,
        seed=seed,
        return_edge_events=False,
    )
    sample_s = time.perf_counter() - started
    started = time.perf_counter()
    detector_ids = tuple(detector.id for detector in npsim_dem.detectors)
    observable_ids = tuple(observable.id for observable in npsim_dem.observables)
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


def stim_dem_to_npsim_dem(stim_dem: Any) -> DetectorErrorModel:
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


def _crossing_rate(
    small_distance_rates: Mapping[float, float],
    large_distance_rates: Mapping[float, float],
    rates: Iterable[float],
) -> float | None:
    previous_rate: float | None = None
    previous_delta: float | None = None
    for rate in rates:
        if rate not in small_distance_rates or rate not in large_distance_rates:
            continue
        delta = small_distance_rates[rate] - large_distance_rates[rate]
        if previous_rate is not None and previous_delta is not None:
            # Below threshold, larger distance should have lower logical failure.
            if previous_delta > 0.0 >= delta:
                fraction = -previous_delta / (delta - previous_delta)
                return previous_rate + fraction * (rate - previous_rate)
        previous_rate = rate
        previous_delta = delta
    return None


def _load_required_modules() -> tuple[Any, Any, Any]:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "npsim-matplotlib-cache"),
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

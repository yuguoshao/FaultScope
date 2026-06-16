"""Surface-code threshold comparison benchmark.

Run from the repository root after building the optional native extension:

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npsim.dem import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from npsim.native import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    compile_native_sampler,
)
from npsim.stim_import import StimImportResult, parse_stim_circuit


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
        "failures\tstderr\tstatus",
        flush=True,
    )

    results: dict[str, dict[int, dict[float, float]]] = {
        path: {distance: {} for distance in distances}
        for path in PATHS
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
                    lambda seed: sample_stim_logical_failure(
                        circuit,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
                (
                    "stim-dem",
                    lambda seed: sample_stim_dem_logical_failure(
                        stim_dem,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
                (
                    "npsim-forward",
                    lambda seed: sample_npsim_forward_logical_failure(
                        imported,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
                (
                    "npsim-dem",
                    lambda seed: sample_npsim_dem_logical_failure(
                        npsim_dem,
                        matcher,
                        args.shots,
                        seed,
                    ),
                ),
            )
            for path_index, (path, fn) in enumerate(path_fns):
                seed = args.seed + 1_000_000 * distance + 1_000 * rate_index + path_index
                try:
                    stats = fn(seed)
                except (ImportError, UnsupportedNativeCircuitError, ValueError) as exc:
                    print(
                        f"{args.basis}\t{distance}\t{rounds}\t{p:.17g}\t"
                        f"{args.shots}\t{path}\tNA\tNA\tNA\t"
                        f"skip:{type(exc).__name__}",
                        flush=True,
                    )
                    continue
                results[path][distance][p] = stats.rate
                print(
                    f"{args.basis}\t{distance}\t{rounds}\t{p:.17g}\t"
                    f"{args.shots}\t{path}\t{stats.rate:.17g}\t"
                    f"{stats.failures}\t{stats.stderr:.17g}\tok",
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


def sample_stim_logical_failure(
    circuit: Any,
    matcher: Any,
    shots: int,
    seed: int,
) -> LogicalFailureStats:
    detectors, observables = circuit.compile_detector_sampler(seed=seed).sample(
        shots,
        separate_observables=True,
    )
    return _logical_failure_stats(detectors, observables, matcher, shots)


def sample_stim_dem_logical_failure(
    stim_dem: Any,
    matcher: Any,
    shots: int,
    seed: int,
) -> LogicalFailureStats:
    detectors, observables, _ = stim_dem.compile_sampler(seed=seed).sample(shots)
    return _logical_failure_stats(detectors, observables, matcher, shots)


def sample_npsim_forward_logical_failure(
    imported: StimImportResult,
    matcher: Any,
    shots: int,
    seed: int,
) -> LogicalFailureStats:
    batch = compile_native_sampler(
        imported.circuit,
        backend="native",
    ).sample(shots=shots, seed=seed)
    detector_ids = tuple(detector.id for detector in imported.detectors)
    observable_ids = tuple(observable.id for observable in imported.observables)
    detectors = _masks_to_bool_matrix(batch.detectors, detector_ids, shots)
    observables = _masks_to_bool_matrix(batch.observables, observable_ids, shots)
    return _logical_failure_stats(detectors, observables, matcher, shots)


def sample_npsim_dem_logical_failure(
    npsim_dem: DetectorErrorModel,
    matcher: Any,
    shots: int,
    seed: int,
) -> LogicalFailureStats:
    batch = compile_native_dem_sampler(
        npsim_dem,
        backend="native",
    ).run_batch(shots=shots, seed=seed)
    detector_ids = tuple(detector.id for detector in npsim_dem.detectors)
    observable_ids = tuple(observable.id for observable in npsim_dem.observables)
    detectors = _masks_to_bool_matrix(batch.detectors, detector_ids, shots)
    observables = _masks_to_bool_matrix(batch.observables, observable_ids, shots)
    return _logical_failure_stats(detectors, observables, matcher, shots)


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
                    event=edge_index,
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


def _logical_failure_stats(
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


def _masks_to_bool_matrix(
    masks_by_id: Mapping[int, int],
    ids: Iterable[int],
    shots: int,
) -> Any:
    _, _, np = _load_required_modules()
    ids = tuple(ids)
    matrix = np.zeros((shots, len(ids)), dtype=np.bool_)
    for col, item_id in enumerate(ids):
        mask = int(masks_by_id.get(item_id, 0))
        for shot in range(shots):
            matrix[shot, col] = bool((mask >> shot) & 1)
    return matrix


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

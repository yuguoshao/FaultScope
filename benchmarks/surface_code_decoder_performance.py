"""Surface-code decoder performance comparison benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/surface_code_decoder_performance.py

The benchmark compares PyMatching and optional NPSim native decoder backends on
Stim standard rotated surface-code DEMs. It reports TSV rows for machine
consumption and does not estimate threshold crossings.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from npsim.decoders import (
    NativeFusionBlossomDecoder,
    NativePyMatchingDecoder,
)
from npsim.dem import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from npsim.runtime import compile_native_dem_sampler
from surface_code_threshold import (
    _decode_batch_masks_with_matching,
    _load_required_modules,
    _logical_failure_stats_from_arrays,
    _logical_failure_stats_from_masks,
)


PATHS = (
    "stim-dem-pymatching",
    "npsim-dem-pymatching",
    "npsim-dem-pymatching-native",
    "npsim-dem-fusion-blossom",
)


@dataclass(frozen=True)
class ProblemMetadata:
    dem_edges: int
    detectors: int
    observables: int


@dataclass(frozen=True)
class BenchmarkRow:
    basis: str
    distance: int
    rounds: int
    p: float
    shots: int
    path: str
    metadata: ProblemMetadata
    solver_edges: int | None
    merged_edges: int | None
    construct_s: float | None
    sample_s: float | None
    decode_or_estimate_s: float | None
    mean_loss: float | None
    python_decode_calls: int | None
    status: str

    @property
    def total_s(self) -> float | None:
        if self.construct_s is None or self.decode_or_estimate_s is None:
            return None
        total = self.construct_s + self.decode_or_estimate_s
        if self.sample_s is not None:
            total += self.sample_s
        return total

    @property
    def shots_per_second(self) -> float | None:
        total = self.total_s
        if total is None or total <= 0.0:
            return None
        return self.shots / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.01])
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--basis", choices=("x", "z"), default="x")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--paths", nargs="+", default=list(PATHS))
    parser.add_argument(
        "--split-native-baseline",
        action="store_true",
        help=(
            "For native decoder paths, report sample_s as a no-decoder native "
            "estimate baseline and decode_or_estimate_s as the decoder delta. "
            "This preserves the TSV schema while exposing whether remaining "
            "time is sampler/hotspot baseline or decoder work."
        ),
    )
    args = parser.parse_args()

    stim, pymatching, _np = _load_required_modules()
    distances = tuple(sorted(set(args.distances)))
    rates = tuple(sorted(set(float(rate) for rate in args.rates)))
    paths = parse_paths(args.paths)
    if not distances:
        raise SystemExit("at least one distance is required")
    if not rates:
        raise SystemExit("at least one physical error rate is required")
    if args.shots <= 0:
        raise SystemExit("shots must be positive")

    print(
        "basis\tdistance\trounds\tp\tshots\tpath\t"
        "dem_edges\tdetectors\tobservables\tsolver_edges\tmerged_edges\t"
        "construct_s\tsample_s\tdecode_or_estimate_s\ttotal_s\t"
        "shots_per_second\tmean_loss\tpython_decode_calls\tstatus",
        flush=True,
    )

    failed = False
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
            npsim_dem = stim_dem_to_graphlike_npsim_dem(stim_dem)
            metadata = ProblemMetadata(
                dem_edges=len(npsim_dem.edges),
                detectors=len(npsim_dem.detectors),
                observables=len(npsim_dem.observables),
            )
            runners: dict[str, Callable[[int], BenchmarkRow]] = {
                "stim-dem-pymatching": lambda seed: run_stim_dem_pymatching(
                    stim_dem,
                    pymatching,
                    args.basis,
                    distance,
                    rounds,
                    p,
                    args.shots,
                    metadata,
                    seed,
                ),
                "npsim-dem-pymatching": lambda seed: run_npsim_dem_pymatching(
                    stim_dem,
                    npsim_dem,
                    pymatching,
                    args.basis,
                    distance,
                    rounds,
                    p,
                    args.shots,
                    metadata,
                    seed,
                ),
                "npsim-dem-pymatching-native": lambda seed: run_npsim_dem_pymatching_native(
                    npsim_dem,
                    args.basis,
                    distance,
                    rounds,
                    p,
                    args.shots,
                    metadata,
                    seed,
                    args.split_native_baseline,
                ),
                "npsim-dem-fusion-blossom": lambda seed: run_npsim_dem_fusion_blossom(
                    npsim_dem,
                    args.basis,
                    distance,
                    rounds,
                    p,
                    args.shots,
                    metadata,
                    seed,
                ),
            }
            for path_index, path in enumerate(paths):
                seed = args.seed + 1_000_000 * distance + 1_000 * rate_index + path_index
                row = safe_run(
                    runners[path],
                    seed,
                    args.basis,
                    distance,
                    rounds,
                    p,
                    args.shots,
                    path,
                    metadata,
                )
                print(format_row(row), flush=True)
                if row.status == "python-callback-used":
                    failed = True

    if failed:
        raise SystemExit("native decoder path used Python decode callback")


def run_stim_dem_pymatching(
    stim_dem: Any,
    pymatching: Any,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    metadata: ProblemMetadata,
    seed: int,
) -> BenchmarkRow:
    started = time.perf_counter()
    matcher = pymatching.Matching.from_detector_error_model(stim_dem)
    sampler = stim_dem.compile_sampler(seed=seed)
    construct_s = time.perf_counter() - started

    started = time.perf_counter()
    detectors, observables, _errors = sampler.sample(shots)
    sample_s = time.perf_counter() - started

    started = time.perf_counter()
    stats = _logical_failure_stats_from_arrays(detectors, observables, matcher, shots)
    decode_s = time.perf_counter() - started

    return BenchmarkRow(
        basis,
        distance,
        rounds,
        p,
        shots,
        "stim-dem-pymatching",
        metadata,
        solver_edges=None,
        merged_edges=None,
        construct_s=construct_s,
        sample_s=sample_s,
        decode_or_estimate_s=decode_s,
        mean_loss=stats.rate,
        python_decode_calls=None,
        status="ok",
    )


def run_npsim_dem_pymatching(
    stim_dem: Any,
    npsim_dem: Any,
    pymatching: Any,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    metadata: ProblemMetadata,
    seed: int,
) -> BenchmarkRow:
    started = time.perf_counter()
    matcher = pymatching.Matching.from_detector_error_model(stim_dem)
    sampler = compile_native_dem_sampler(npsim_dem)
    construct_s = time.perf_counter() - started

    started = time.perf_counter()
    batch = sampler.run_batch(
        shots=shots,
        seed=seed,
        return_edge_events=False,
    )
    sample_s = time.perf_counter() - started

    detector_ids = tuple(detector.id for detector in npsim_dem.detectors)
    observable_ids = tuple(observable.id for observable in npsim_dem.observables)
    started = time.perf_counter()
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

    return BenchmarkRow(
        basis,
        distance,
        rounds,
        p,
        shots,
        "npsim-dem-pymatching",
        metadata,
        solver_edges=None,
        merged_edges=None,
        construct_s=construct_s,
        sample_s=sample_s,
        decode_or_estimate_s=decode_s,
        mean_loss=stats.rate,
        python_decode_calls=None,
        status="ok",
    )


def run_npsim_dem_pymatching_native(
    npsim_dem: Any,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    metadata: ProblemMetadata,
    seed: int,
    split_baseline: bool,
) -> BenchmarkRow:
    started = time.perf_counter()
    sampler = compile_native_dem_sampler(npsim_dem)
    decoder = NativePyMatchingDecoder.from_dem(npsim_dem)
    construct_s = time.perf_counter() - started

    baseline_s = None
    if split_baseline:
        started = time.perf_counter()
        sampler.estimate(shots=shots, seed=seed, aggregate_hotspots=False)
        baseline_s = time.perf_counter() - started

    started = time.perf_counter()
    result = sampler.estimate(
        shots=shots,
        seed=seed,
        decoder=decoder,
        aggregate_hotspots=False,
    )
    estimate_s = time.perf_counter() - started
    decoder_delta_s = (
        max(0.0, estimate_s - baseline_s) if baseline_s is not None else estimate_s
    )

    python_decode_calls = int(getattr(decoder, "python_decode_call_count", -1))
    status = "ok" if python_decode_calls == 0 else "python-callback-used"
    summary = decoder.build_summary

    return BenchmarkRow(
        basis,
        distance,
        rounds,
        p,
        shots,
        "npsim-dem-pymatching-native",
        metadata,
        solver_edges=int(getattr(decoder, "solver_edge_count")),
        merged_edges=int(summary.get("merged_parallel_edge_count", 0)),
        construct_s=construct_s,
        sample_s=baseline_s,
        decode_or_estimate_s=decoder_delta_s,
        mean_loss=float(result.mean_loss),
        python_decode_calls=python_decode_calls,
        status=status,
    )


def run_npsim_dem_fusion_blossom(
    npsim_dem: Any,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    metadata: ProblemMetadata,
    seed: int,
) -> BenchmarkRow:
    started = time.perf_counter()
    sampler = compile_native_dem_sampler(npsim_dem)
    decoder = NativeFusionBlossomDecoder.from_dem(npsim_dem)
    construct_s = time.perf_counter() - started

    started = time.perf_counter()
    result = sampler.estimate(shots=shots, seed=seed, decoder=decoder)
    estimate_s = time.perf_counter() - started

    python_decode_calls = int(getattr(decoder, "python_decode_call_count", -1))
    status = "ok" if python_decode_calls == 0 else "python-callback-used"
    summary = decoder.build_summary

    return BenchmarkRow(
        basis,
        distance,
        rounds,
        p,
        shots,
        "npsim-dem-fusion-blossom",
        metadata,
        solver_edges=int(getattr(decoder, "solver_edge_count")),
        merged_edges=int(summary.get("merged_parallel_edge_count", 0)),
        construct_s=construct_s,
        sample_s=None,
        decode_or_estimate_s=estimate_s,
        mean_loss=float(result.mean_loss),
        python_decode_calls=python_decode_calls,
        status=status,
    )


def stim_dem_to_graphlike_npsim_dem(stim_dem: Any) -> DetectorErrorModel:
    detectors_by_id: dict[int, Detector] = {}
    observable_ids: set[int] = set()
    edges: list[DetectorErrorEdge] = []
    detector_offset = 0

    for instruction in stim_dem:
        instruction_type = instruction.type
        if instruction_type == "error":
            probability = float(instruction.args_copy()[0])
            groups: list[tuple[list[int], list[int]]] = [([], [])]
            for target in instruction.targets_copy():
                if target.is_separator():
                    groups.append(([], []))
                elif target.is_relative_detector_id():
                    groups[-1][0].append(detector_offset + int(target.val))
                elif target.is_logical_observable_id():
                    observable_id = int(target.val)
                    groups[-1][1].append(observable_id)
                    observable_ids.add(observable_id)
                else:
                    raise ValueError(f"unsupported Stim DEM target {target!r}")
            for detectors, observables in groups:
                if not detectors and not observables:
                    continue
                edge_index = len(edges)
                edges.append(
                    DetectorErrorEdge(
                        probability=probability,
                        detectors=tuple(detectors),
                        observables=tuple(observables),
                        location_id=f"stim_dem_edge_{edge_index}",
                        event=f"stim_dem_edge_{edge_index}",
                        tags={"source": "stim_dem_graphlike"},
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


def safe_run(
    fn: Callable[[int], BenchmarkRow],
    seed: int,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    path: str,
    metadata: ProblemMetadata,
) -> BenchmarkRow:
    try:
        return fn(seed)
    except (ImportError, ValueError, RuntimeError) as exc:
        return BenchmarkRow(
            basis,
            distance,
            rounds,
            p,
            shots,
            path,
            metadata,
            solver_edges=None,
            merged_edges=None,
            construct_s=None,
            sample_s=None,
            decode_or_estimate_s=None,
            mean_loss=None,
            python_decode_calls=None,
            status=f"skip:{type(exc).__name__}:{sanitize_status(str(exc))}",
        )


def format_row(row: BenchmarkRow) -> str:
    return "\t".join(
        (
            row.basis,
            str(row.distance),
            str(row.rounds),
            f"{row.p:.17g}",
            str(row.shots),
            row.path,
            str(row.metadata.dem_edges),
            str(row.metadata.detectors),
            str(row.metadata.observables),
            format_optional_int(row.solver_edges),
            format_optional_int(row.merged_edges),
            format_optional_float(row.construct_s),
            format_optional_float(row.sample_s),
            format_optional_float(row.decode_or_estimate_s),
            format_optional_float(row.total_s),
            format_optional_float(row.shots_per_second),
            format_optional_float(row.mean_loss, precision=17),
            format_optional_int(row.python_decode_calls),
            row.status,
        )
    )


def format_optional_float(value: float | None, *, precision: int = 6) -> str:
    if value is None:
        return "NA"
    if precision == 17:
        return f"{value:.17g}"
    return f"{value:.{precision}f}"


def format_optional_int(value: int | None) -> str:
    if value is None:
        return "NA"
    return str(value)


def sanitize_status(value: str) -> str:
    return " ".join(value.replace("\t", " ").split())


def parse_paths(raw_paths: list[str]) -> tuple[str, ...]:
    paths: list[str] = []
    for raw in raw_paths:
        for path in raw.split(","):
            path = path.strip()
            if path:
                paths.append(path)
    unknown = sorted(set(paths) - set(PATHS))
    if unknown:
        raise SystemExit(
            "unknown path(s): "
            + ", ".join(unknown)
            + "; expected one of "
            + ", ".join(PATHS)
        )
    return tuple(dict.fromkeys(paths))


if __name__ == "__main__":
    main()

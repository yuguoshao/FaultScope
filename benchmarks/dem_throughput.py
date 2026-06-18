"""DEM generation and sampling throughput benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21

The benchmark compares native Rust DEM generation / default DEM hotspot
estimation against Stim detector-error-model generation and detector sampling on
deterministic repetition-code and rotated surface-code detector error models.
Stim is required.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npsim.core import (
    BernoulliPauliNoise,
    Circuit,
    MeasurementBitFlip,
    NoiseLocation,
    Operation,
)
from npsim.dem import Detector, LogicalObservable
from npsim.experiments import make_repetition_code_experiment
from npsim.runtime import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    generate_native_dem,
)
from tests.stim_helpers import to_stim_circuit, with_dem_declarations
from tests.surface_code_examples import _data_index, _rotated_surface_code_checks


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    distance: int
    rounds: int
    circuit: Circuit
    detectors: tuple[Detector, ...]
    observables: tuple[LogicalObservable, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[9, 13, 21])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    _validate_distances(args.distances)
    _load_stim()
    print(
        "case\tdistance\trounds\tedges\tnative_gen_s\tnative_det_gen_s\t"
        "stim_gen_s\tstim_gen_ratio\tnative_est_sps\t"
        "native_det_sps\tstim_det_sps\tdet_ratio\tstatus",
        flush=True,
    )
    for case in _make_cases(args.distances, args.rounds):
        _run_case(case, shots=args.shots, repeats=args.repeats)


def _make_cases(distances: list[int], rounds: int) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for distance in distances:
        cases.append(_make_repetition_case(distance, rounds))
        cases.extend(_make_surface_memory_cases(distance, rounds))
    return cases


def _validate_distances(distances: list[int]) -> None:
    invalid = [distance for distance in distances if distance < 3 or distance % 2 != 1]
    if invalid:
        values = ", ".join(str(distance) for distance in invalid)
        raise SystemExit(
            "surface-code DEM benchmark distances must be odd integers >= 3; "
            f"got {values}"
        )


def _make_repetition_case(distance: int, rounds: int) -> BenchmarkCase:
    experiment = make_repetition_code_experiment(
        distance=distance,
        rounds=rounds,
        data_error_rate=0.025,
        measurement_error_rate=0.015,
    )
    return BenchmarkCase(
        label=f"repetition-d{distance}",
        distance=distance,
        rounds=rounds,
        circuit=experiment.circuit,
        detectors=tuple(experiment.detectors),
        observables=tuple(experiment.observables),
    )


def _run_case(case: BenchmarkCase, *, shots: int, repeats: int) -> None:
    try:
        native_gen_s, native_dem = _time_once(
            lambda: generate_native_dem(
                case.circuit,
                detectors=case.detectors,
                observables=case.observables,
            )
        )
        native_sampler = compile_native_dem_sampler(native_dem)
        stim_circuit, _ = to_stim_circuit(
            with_dem_declarations(
                case.circuit,
                detectors=case.detectors,
                observables=(),
            )
        )
        stim_gen_s, _ = _time_once(stim_circuit.detector_error_model)
        native_detector_gen_s, native_detector_dem = _time_once(
            lambda: generate_native_dem(
                case.circuit,
                detectors=case.detectors,
                observables=(),
            )
        )
        native_detector_sampler = compile_native_dem_sampler(native_detector_dem)
        stim_sampler = stim_circuit.compile_detector_sampler()
        native_det_sps = _median_samples_per_second(
            lambda seed: native_detector_sampler.run_batch(
                shots=shots,
                seed=seed,
            ),
            shots=shots,
            repeats=repeats,
        )
        stim_det_sps = _median_samples_per_second(
            lambda seed: _sample_stim_detectors(
                stim_sampler,
                shots,
                seed,
            ),
            shots=shots,
            repeats=repeats,
        )
        native_est_sps = _median_samples_per_second(
            lambda seed: native_sampler.estimate_default(
                shots=shots,
                seed=seed,
            ),
            shots=shots,
            repeats=repeats,
        )
        stim_gen_ratio = native_detector_gen_s / stim_gen_s if stim_gen_s else float("inf")
        det_ratio = native_det_sps / stim_det_sps if stim_det_sps else float("inf")
        print(
            f"{case.label}\t{case.distance}\t{case.rounds}\t"
            f"{len(native_dem.edges)}\t{native_gen_s:.6f}\t"
            f"{native_detector_gen_s:.6f}\t{stim_gen_s:.6f}\t"
            f"{stim_gen_ratio:.3f}\t{native_est_sps:.3f}\t"
            f"{native_det_sps:.3f}\t{stim_det_sps:.3f}\t"
            f"{det_ratio:.3f}\tok",
            flush=True,
        )
    except UnsupportedNativeCircuitError as exc:
        print(
            f"{case.label}\t{case.distance}\t{case.rounds}\t"
            f"NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tnative-skip:{type(exc).__name__}",
            flush=True,
        )


def _make_surface_memory_cases(distance: int, rounds: int) -> list[BenchmarkCase]:
    return [
        _make_surface_memory_case("z", distance, rounds),
        _make_surface_memory_case("x", distance, rounds),
    ]


def _make_surface_memory_case(memory: str, distance: int, rounds: int) -> BenchmarkCase:
    x_checks, z_checks = _rotated_surface_code_checks(distance)
    operations: list[Operation] = []
    if memory == "z":
        checks = z_checks
        basis = "Z"
        data_error = "X"
        logical_qubits = tuple(_data_index(distance, row, 0) for row in range(distance))
    elif memory == "x":
        checks = x_checks
        basis = "X"
        data_error = "Z"
        logical_qubits = tuple(_data_index(distance, 0, col) for col in range(distance))
        operations.extend(Operation.h(qubit) for qubit in range(distance * distance))
    else:
        raise ValueError(f"unknown surface memory case {memory!r}")

    _append_surface_memory_checks(
        operations,
        distance=distance,
        round_idx=0,
        checks=checks,
        basis=basis,
        noise=False,
    )
    for round_idx in range(1, rounds + 1):
        for qubit in range(distance * distance):
            operations.append(
                Operation.noise(
                    NoiseLocation(
                        id=f"surface_{memory}_data_r{round_idx}_q{qubit}",
                        model=BernoulliPauliNoise(data_error),
                        rate=0.04,
                        qubits=(qubit,),
                        tags={
                            "layout": "rotated_surface_code",
                            "memory": memory,
                            "round": round_idx,
                            "operation": "data_noise",
                        },
                    )
                )
            )
        _append_surface_memory_checks(
            operations,
            distance=distance,
            round_idx=round_idx,
            checks=checks,
            basis=basis,
            noise=True,
        )
    for idx, qubit in enumerate(logical_qubits):
        operations.append(Operation.measure(qubit, key=f"logical_{idx}", basis=basis))

    detectors = tuple(
        Detector(
            id=idx,
            measurement_keys=(f"r{rounds}_{check['id']}", f"r0_{check['id']}"),
            coords=(float(check["x"]), float(check["y"])),
        )
        for idx, check in enumerate(checks)
    )
    observables = (
        LogicalObservable(
            id=0,
            measurement_keys=_path_keys("logical", len(logical_qubits)),
        ),
    )
    return BenchmarkCase(
        label=f"surface-{memory}-d{distance}",
        distance=distance,
        rounds=rounds,
        circuit=Circuit(n_qubits=distance * distance, operations=tuple(operations)),
        detectors=detectors,
        observables=observables,
    )


def _append_surface_memory_checks(
    operations: list[Operation],
    *,
    distance: int,
    round_idx: int,
    checks: list[dict[str, object]],
    basis: str,
    noise: bool,
) -> None:
    for check in checks:
        check_id = str(check["id"])
        qubits = tuple(
            _data_index(distance, row, col)
            for row, col in check["data"]
        )
        location = None
        if noise:
            location = NoiseLocation(
                id=f"surface_{basis.lower()}_meas_r{round_idx}_{check_id}",
                model=MeasurementBitFlip(),
                rate=0.02,
                qubits=qubits[:1],
                tags={
                    "layout": "rotated_surface_code",
                    "round": round_idx,
                    "operation": "measurement_noise",
                },
            )
        operations.append(
            Operation.measure_pauli(
                qubits,
                basis * len(qubits),
                key=f"r{round_idx}_{check_id}",
                noise=location,
            )
        )


def _path_keys(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{idx}" for idx in range(count))


def _time_once(fn: Any) -> tuple[float, Any]:
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    return elapsed, value


def _median_samples_per_second(fn: Any, *, shots: int, repeats: int) -> float:
    values: list[float] = []
    for repeat in range(repeats):
        seed = 20_000 + repeat
        start = time.perf_counter()
        fn(seed)
        elapsed = time.perf_counter() - start
        values.append(shots / elapsed)
    return statistics.median(values)


def _load_stim() -> Any:
    try:
        import stim
    except ImportError as exc:
        raise SystemExit("Stim is required for dem_throughput.py") from exc
    return stim


def _sample_stim_detectors(stim_sampler: Any, shots: int, seed: int) -> Any:
    del seed
    try:
        return stim_sampler.sample(shots=shots, bit_packed=True)
    except TypeError:
        return stim_sampler.sample(shots, bit_packed=True)


if __name__ == "__main__":
    main()

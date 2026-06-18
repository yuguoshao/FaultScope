"""DEM generation and sampling throughput benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21

The benchmark compares native Rust DEM generation / default DEM hotspot
estimation against Stim detector-error-model generation and detector sampling on
deterministic repetition-code and surface-code detector error models.  Stim is
required.
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

from npsim.core import Circuit, NoiseLocation, Operation
from npsim.dem import Detector, LogicalObservable
from npsim.runtime import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    compile_native_dem_sampler_from_circuit,
    generate_native_dem,
)
from npsim.core import BernoulliPauliNoise, MeasurementBitFlip, PauliChannel
from npsim.experiments import make_repetition_code_experiment
from tests.surface_code_examples import _data_index, _rotated_surface_code_checks


@dataclass(frozen=True)
class DemBenchmarkCase:
    label: str
    family: str
    distance: int
    rounds: int
    circuit: Circuit
    detectors: tuple[Detector, ...]
    observables: tuple[LogicalObservable, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=("all", "repetition", "surface"),
        default="all",
        help="Which DEM benchmark family to run.",
    )
    parser.add_argument("--distances", nargs="+", type=int, default=[9, 13, 21])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    stim_module = _load_stim()
    print(
        "case\tdistance\trounds\tedges\tnative_gen_s\tnative_det_gen_s\t"
        "native_det_compile_s\tstim_gen_s\tstim_gen_ratio\tnative_est_sps\t"
        "native_det_sps\tstim_det_sps\tdet_ratio\tstatus",
        flush=True,
    )
    for case in _make_cases(args):
        try:
            native_gen_s, native_dem = _time_once(
                lambda: generate_native_dem(
                    case.circuit,
                    detectors=case.detectors,
                    observables=case.observables,
                )
            )
            native_sampler = compile_native_dem_sampler(native_dem)
            stim_circuit = stim_module.Circuit(
                _to_stim_text_with_detectors(
                    case.circuit,
                    case.detectors,
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
            native_detector_compile_s, native_detector_sampler = _time_once(
                lambda: compile_native_dem_sampler_from_circuit(
                    case.circuit,
                    detectors=case.detectors,
                    observables=(),
                    materialize_dem=False,
                )
            )
            stim_sampler = stim_circuit.compile_detector_sampler()
            native_det_sps = _median_samples_per_second(
                lambda seed: native_detector_sampler.run_batch(
                    shots=args.shots,
                    seed=seed,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )
            stim_det_sps = _median_samples_per_second(
                lambda seed: _sample_stim_detectors(
                    stim_sampler,
                    args.shots,
                    seed,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )
            native_est_sps = _median_samples_per_second(
                lambda seed: native_sampler.estimate_default(
                    shots=args.shots,
                    seed=seed,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )
            stim_gen_ratio = native_detector_gen_s / stim_gen_s if stim_gen_s else float("inf")
            det_ratio = native_det_sps / stim_det_sps if stim_det_sps else float("inf")
            print(
                f"{case.label}\t{case.distance}\t{case.rounds}\t"
                f"{len(native_dem.edges)}\t{native_gen_s:.6f}\t"
                f"{native_detector_gen_s:.6f}\t{native_detector_compile_s:.6f}\t"
                f"{stim_gen_s:.6f}\t"
                f"{stim_gen_ratio:.3f}\t{native_est_sps:.3f}\t"
                f"{native_det_sps:.3f}\t{stim_det_sps:.3f}\t"
                f"{det_ratio:.3f}\tok",
                flush=True,
            )
        except UnsupportedNativeCircuitError as exc:
            print(
                f"{case.label}\t{case.distance}\t{case.rounds}\t"
                f"NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tnative-skip:{type(exc).__name__}",
                flush=True,
            )


def _make_cases(args: argparse.Namespace) -> list[DemBenchmarkCase]:
    cases: list[DemBenchmarkCase] = []
    if args.family in {"all", "repetition"}:
        cases.extend(_make_repetition_cases(args.distances, args.rounds))
    if args.family in {"all", "surface"}:
        cases.extend(_make_surface_cases(args.distances, args.rounds))
    return cases


def _make_repetition_cases(distances: list[int], rounds: int) -> list[DemBenchmarkCase]:
    cases = []
    for distance in distances:
        experiment = make_repetition_code_experiment(
            distance=distance,
            rounds=rounds,
            data_error_rate=0.025,
            measurement_error_rate=0.015,
        )
        cases.append(
            DemBenchmarkCase(
                label=f"repetition-d{distance}",
                family="repetition",
                distance=distance,
                rounds=rounds,
                circuit=experiment.circuit,
                detectors=tuple(experiment.detectors),
                observables=tuple(experiment.observables),
            )
        )
    return cases


def _make_surface_cases(distances: list[int], rounds: int) -> list[DemBenchmarkCase]:
    cases = []
    for distance in distances:
        cases.append(_make_surface_memory_case(distance, rounds, memory="z"))
        cases.append(_make_surface_memory_case(distance, rounds, memory="x"))
    return cases


def _make_surface_memory_case(distance: int, rounds: int, *, memory: str) -> DemBenchmarkCase:
    if distance < 3 or distance % 2 != 1:
        raise ValueError("surface-code distance must be an odd integer >= 3")
    x_checks, z_checks = _rotated_surface_code_checks(distance)
    operations: list[Operation] = []
    if memory == "z":
        checks = tuple(z_checks)
        basis = "Z"
        data_error = "X"
        logical_qubits = tuple(_data_index(distance, row, 0) for row in range(distance))
    elif memory == "x":
        checks = tuple(x_checks)
        basis = "X"
        data_error = "Z"
        logical_qubits = tuple(_data_index(distance, 0, col) for col in range(distance))
        for qubit in range(distance * distance):
            operations.append(Operation.h(qubit))
    else:
        raise ValueError(f"unknown surface memory {memory!r}")

    _append_surface_memory_checks(
        operations,
        distance=distance,
        round_idx=0,
        checks=checks,
        basis=basis,
        measurement_noise=False,
    )
    for round_idx in range(1, rounds + 1):
        for qubit in range(distance * distance):
            operations.append(
                Operation.noise(
                    NoiseLocation(
                        id=f"surface_{memory}_data_r{round_idx}_q{qubit}",
                        model=BernoulliPauliNoise(data_error),
                        rate=0.025,
                        qubits=(qubit,),
                        tags={
                            "layout": "rotated_surface_code",
                            "family": f"surface_{memory}_memory",
                            "round": round_idx,
                            "qubit": qubit,
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
            measurement_noise=True,
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
            measurement_keys=tuple(f"logical_{idx}" for idx in range(len(logical_qubits))),
        ),
    )
    return DemBenchmarkCase(
        label=f"surface-{memory}-d{distance}",
        family="surface",
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
    checks: tuple[dict[str, object], ...],
    basis: str,
    measurement_noise: bool,
) -> None:
    for check in checks:
        check_id = str(check["id"])
        qubits = tuple(_data_index(distance, row, col) for row, col in check["data"])
        location = None
        if measurement_noise:
            location = NoiseLocation(
                id=f"surface_{basis.lower()}_meas_r{round_idx}_{check_id}",
                model=MeasurementBitFlip(),
                rate=0.015,
                qubits=qubits[:1],
                tags={
                    "layout": "rotated_surface_code",
                    "round": round_idx,
                    "check": check_id,
                    "basis": basis,
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


def _to_stim_text_with_detectors(circuit: Circuit, detectors: tuple[Any, ...]) -> str:
    lines: list[str] = []
    measurement_order: list[str] = []
    for operation in circuit.operations:
        lines.extend(_operation_to_stim_lines(operation, measurement_order))
    measurement_index = {key: idx for idx, key in enumerate(measurement_order)}
    current = len(measurement_order)
    for detector in detectors:
        coords = ""
        if detector.coords:
            coords = "(" + ", ".join(f"{coord:.17g}" for coord in detector.coords) + ")"
        targets = []
        for key in detector.measurement_keys:
            idx = measurement_index[key]
            targets.append(f"rec[-{current - idx}]")
        lines.append(f"DETECTOR{coords} {' '.join(targets)}".rstrip())
    return "\n".join(lines)


def _operation_to_stim_lines(
    operation: Operation,
    measurement_order: list[str],
) -> list[str]:
    kind = operation.kind
    if kind == "h":
        return [f"H {operation.qubits[0]}"]
    if kind == "s":
        return [f"S {operation.qubits[0]}"]
    if kind == "s_dag":
        return [f"S_DAG {operation.qubits[0]}"]
    if kind == "cx":
        return [f"CX {operation.qubits[0]} {operation.qubits[1]}"]
    if kind == "cz":
        return [f"CZ {operation.qubits[0]} {operation.qubits[1]}"]
    if kind == "swap":
        return [f"SWAP {operation.qubits[0]} {operation.qubits[1]}"]
    if kind == "pauli":
        return [f"{operation.pauli} {' '.join(str(q) for q in operation.qubits)}"]
    if kind == "noise":
        return [_noise_to_stim_line(operation.noise_location)]
    if kind == "measure":
        measurement_order.append(operation.key or f"m{len(measurement_order)}")
        return [_measurement_to_stim_line(operation)]
    if kind == "measure_pauli":
        measurement_order.append(operation.key or f"m{len(measurement_order)}")
        return [_mpp_to_stim_line(operation)]
    if kind == "reset":
        basis = operation.basis.upper()
        name = {"Z": "R", "X": "RX", "Y": "RY"}[basis]
        if operation.key is not None:
            measurement_order.append(operation.key)
        return [f"{name} {operation.qubits[0]}"]
    if kind in {"detector", "observable_include"}:
        return []
    raise ValueError(f"unsupported benchmark Stim conversion op {kind!r}")


def _noise_to_stim_line(location: Any) -> str:
    if location is None:
        raise ValueError("noise operation missing location")
    model = location.model
    targets = " ".join(str(qubit) for qubit in location.qubits)
    if isinstance(model, BernoulliPauliNoise) and len(model.pauli) == 1:
        return f"{model.pauli}_ERROR({location.rate:.17g}) {targets}"
    if isinstance(model, PauliChannel) and len(location.qubits) == 1:
        probs = {pauli: 0.0 for pauli in ("X", "Y", "Z")}
        total = model.total_weight
        for pauli, weight in model.weights.items():
            probs[pauli] = location.rate * weight / total
        return (
            "PAULI_CHANNEL_1"
            f"({probs['X']:.17g},{probs['Y']:.17g},{probs['Z']:.17g}) {targets}"
        )
    raise ValueError(f"unsupported benchmark noise model {type(model).__name__}")


def _measurement_to_stim_line(operation: Operation) -> str:
    basis = operation.basis.upper()
    name = {"Z": "M", "X": "MX", "Y": "MY"}[basis]
    if operation.noise_location is None:
        return f"{name} {operation.qubits[0]}"
    if not isinstance(operation.noise_location.model, MeasurementBitFlip):
        raise ValueError("benchmark conversion supports only measurement bit-flip noise")
    return f"{name}({operation.noise_location.rate:.17g}) {operation.qubits[0]}"


def _mpp_to_stim_line(operation: Operation) -> str:
    if operation.pauli is None:
        raise ValueError("MPP conversion requires a Pauli string")
    prefix = "MPP"
    if operation.noise_location is not None:
        if not isinstance(operation.noise_location.model, MeasurementBitFlip):
            raise ValueError("benchmark conversion supports only measurement bit-flip noise")
        prefix += f"({operation.noise_location.rate:.17g})"
    targets = "*".join(
        f"{pauli}{qubit}" for qubit, pauli in zip(operation.qubits, operation.pauli)
    )
    return f"{prefix} {targets}"


if __name__ == "__main__":
    main()

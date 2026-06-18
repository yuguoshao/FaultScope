"""DEM generation and default-estimate throughput benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21

The benchmark compares native Rust DEM generation / default DEM hotspot
estimation against the test-only Python reference implementation on deterministic
repetition-code detector error models.  When Stim is installed, it also reports
Stim detector-error-model generation and detector-sampler throughput.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npsim.core import Circuit, Operation
from npsim.runtime import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    generate_native_dem,
)
from npsim.core import BernoulliPauliNoise, MeasurementBitFlip, PauliChannel
from npsim.experiments import make_repetition_code_experiment
from tests.reference.dem import (
    DetectorErrorModelGenerator as ReferenceDetectorErrorModelGenerator,
)
from tests.reference.dem_sampler import (
    DemBatchHotspotSimulator as ReferenceDemBatchHotspotSimulator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[9, 13, 21])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    stim_module = _load_stim()
    print(
        "case\tdistance\trounds\tedges\tnative_gen_s\treference_gen_s\t"
        "gen_ratio\tnative_det_gen_s\tstim_gen_s\tnative_vs_stim_gen\t"
        "native_est_sps\treference_est_sps\test_ratio\t"
        "native_det_sps\tstim_det_sps\tdet_ratio\tstatus",
        flush=True,
    )
    for distance in args.distances:
        experiment = make_repetition_code_experiment(
            distance=distance,
            rounds=args.rounds,
            data_error_rate=0.025,
            measurement_error_rate=0.015,
        )
        reference_generator = ReferenceDetectorErrorModelGenerator(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=experiment.observables,
        )

        try:
            native_gen_s, native_dem = _time_once(
                lambda: generate_native_dem(
                    experiment.circuit,
                    detectors=experiment.detectors,
                    observables=experiment.observables,
                )
            )
            reference_gen_s, reference_dem = _time_once(reference_generator.generate)
            native_sampler = compile_native_dem_sampler(native_dem)
            reference_sampler = ReferenceDemBatchHotspotSimulator(reference_dem)
            stim_gen_s: float | None = None
            native_vs_stim_gen = float("nan")
            native_det_sps: float | None = None
            stim_det_sps: float | None = None
            det_ratio = float("nan")
            native_detector_gen_s: float | None = None
            if stim_module is not None:
                stim_circuit = stim_module.Circuit(
                    _to_stim_text_with_detectors(
                        experiment.circuit,
                        experiment.detectors,
                    )
                )
                stim_gen_s, _ = _time_once(stim_circuit.detector_error_model)
                native_detector_gen_s, native_detector_dem = _time_once(
                    lambda: generate_native_dem(
                        experiment.circuit,
                        detectors=experiment.detectors,
                        observables=(),
                    )
                )
                native_detector_sampler = compile_native_dem_sampler(native_detector_dem)
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
                native_vs_stim_gen = (
                    stim_gen_s / native_detector_gen_s
                    if native_detector_gen_s
                    else float("inf")
                )
                det_ratio = native_det_sps / stim_det_sps if stim_det_sps else float("inf")
            native_est_sps = _median_samples_per_second(
                lambda seed: native_sampler.estimate_default(
                    shots=args.shots,
                    seed=seed,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )
            reference_est_sps = _median_samples_per_second(
                lambda seed: reference_sampler.estimate(
                    shots=args.shots,
                    seed=seed,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )
            gen_ratio = reference_gen_s / native_gen_s if native_gen_s else float("inf")
            est_ratio = native_est_sps / reference_est_sps if reference_est_sps else float("inf")
            status = "pass" if gen_ratio >= 2.0 and est_ratio >= 5.0 else "below-target"
            stim_gen_cell = f"{stim_gen_s:.6f}" if stim_gen_s is not None else "NA"
            native_detector_gen_cell = (
                f"{native_detector_gen_s:.6f}"
                if native_detector_gen_s is not None
                else "NA"
            )
            native_vs_stim_gen_cell = (
                f"{native_vs_stim_gen:.6f}" if stim_gen_s is not None else "NA"
            )
            native_det_cell = f"{native_det_sps:.3f}" if native_det_sps is not None else "NA"
            stim_det_cell = f"{stim_det_sps:.3f}" if stim_det_sps is not None else "NA"
            det_ratio_cell = f"{det_ratio:.3f}" if stim_det_sps is not None else "NA"
            print(
                f"repetition-d{distance}\t{distance}\t{args.rounds}\t"
                f"{len(native_dem.edges)}\t{native_gen_s:.6f}\t{reference_gen_s:.6f}\t"
                f"{gen_ratio:.3f}\t{native_detector_gen_cell}\t"
                f"{stim_gen_cell}\t{native_vs_stim_gen_cell}\t"
                f"{native_est_sps:.3f}\t{reference_est_sps:.3f}\t{est_ratio:.3f}\t"
                f"{native_det_cell}\t{stim_det_cell}\t{det_ratio_cell}\t{status}",
                flush=True,
            )
        except UnsupportedNativeCircuitError as exc:
            print(
                f"repetition-d{distance}\t{distance}\t{args.rounds}\tNA\tNA\tNA\tNA\t"
                f"NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tnative-skip:{type(exc).__name__}",
                flush=True,
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


def _load_stim() -> Any | None:
    try:
        import stim
    except ImportError:
        return None
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

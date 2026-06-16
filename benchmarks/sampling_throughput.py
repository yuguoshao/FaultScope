"""Packed sampling throughput benchmark for NPSim native sampler.

Run from the repository root after building the native extension in release
mode:

    .venv/bin/python benchmarks/sampling_throughput.py --distances 15 21 31
    .venv/bin/python benchmarks/sampling_throughput.py --family random-clifford \
        --qubits 128 256 512 --depth 20
    .venv/bin/python benchmarks/sampling_throughput.py --family random-clifford \
        --qubits 128 256 512 --depth 20 --noise-rate 0.001

Stim is optional.  When it is not installed, the script reports only NPSim
throughput and marks the Stim comparison as skipped.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.native import compile_native_sampler
from npsim.noise import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
)
from tests.surface_code_examples import make_large_rotated_surface_code_memory_example


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    circuit: Circuit
    qubits: int
    depth: int | None
    rounds: int | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=("surface-code", "random-clifford"),
        default="surface-code",
    )
    parser.add_argument("--distances", nargs="+", type=int, default=[15, 21, 31])
    parser.add_argument("--qubits", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument(
        "--noise-model",
        choices=("depolarizing1", "x"),
        default="depolarizing1",
    )
    parser.add_argument("--noise-rate", type=float, default=0.0)
    parser.add_argument("--measurement-noise-rate", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend", choices=("auto", "native"), default="auto")
    args = parser.parse_args()

    stim_module = _load_stim()
    print(
        "case\tqubits\tdepth\trounds\tnoise\tmeas_noise\tshots\tbackend\t"
        "npsim_sps\tstim_sps\tratio\tstatus",
        flush=True,
    )
    for case in _make_cases(args):
        _run_case(case, args=args, stim_module=stim_module)


def _make_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if args.family == "surface-code":
        cases: list[BenchmarkCase] = []
        for distance in args.distances:
            example = make_large_rotated_surface_code_memory_example(
                distance=distance,
                rounds=args.rounds,
            )
            cases.append(
                BenchmarkCase(
                    label=f"surface-d{distance}",
                    circuit=example.circuit,
                    qubits=example.circuit.n_qubits,
                    depth=None,
                    rounds=args.rounds,
                )
            )
        return cases

    return [
        BenchmarkCase(
            label=f"random-clifford-{n_qubits}q",
            circuit=_make_random_clifford_circuit(
                n_qubits=n_qubits,
                depth=args.depth,
                seed=args.seed + n_qubits,
                noise_model=args.noise_model,
                noise_rate=args.noise_rate,
                measurement_noise_rate=args.measurement_noise_rate,
            ),
            qubits=n_qubits,
            depth=args.depth,
            rounds=None,
        )
        for n_qubits in args.qubits
    ]


def _run_case(
    case: BenchmarkCase,
    *,
    args: argparse.Namespace,
    stim_module: Any | None,
) -> None:
    sampler = compile_native_sampler(case.circuit, backend=args.backend)
    npsim_sps = _median_samples_per_second(
        lambda seed: sampler.sample_measurements(shots=args.shots, seed=seed),
        shots=args.shots,
        repeats=args.repeats,
    )

    stim_sps: float | None = None
    ratio = float("nan")
    status = "stim-skip"
    if stim_module is not None:
        try:
            stim_circuit = stim_module.Circuit(_to_stim_text(case.circuit))
            stim_sampler = stim_circuit.compile_sampler()
            stim_sps = _median_samples_per_second(
                lambda seed: _sample_stim_packed(stim_sampler, args.shots, seed),
                shots=args.shots,
                repeats=args.repeats,
            )
            ratio = npsim_sps / stim_sps if stim_sps else float("inf")
            status = "pass" if ratio >= 1.10 else "below-target"
        except Exception as exc:
            status = f"stim-error:{type(exc).__name__}"

    stim_cell = f"{stim_sps:.3f}" if stim_sps is not None else "NA"
    ratio_cell = f"{ratio:.3f}" if stim_sps is not None else "NA"
    depth_cell = str(case.depth) if case.depth is not None else "NA"
    rounds_cell = str(case.rounds) if case.rounds is not None else "NA"
    noise_cell = "circuit" if case.depth is None else _noise_label(args)
    print(
        f"{case.label}\t{case.qubits}\t{depth_cell}\t{rounds_cell}\t"
        f"{noise_cell}\t{args.measurement_noise_rate:.17g}\t"
        f"{args.shots}\t{sampler.backend_name}\t{npsim_sps:.3f}\t"
        f"{stim_cell}\t{ratio_cell}\t{status}",
        flush=True,
    )


def _make_random_clifford_circuit(
    *,
    n_qubits: int,
    depth: int,
    seed: int,
    noise_model: str,
    noise_rate: float,
    measurement_noise_rate: float,
) -> Circuit:
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise-rate must be in [0, 1]")
    if not 0.0 <= measurement_noise_rate <= 1.0:
        raise ValueError("measurement-noise-rate must be in [0, 1]")
    rng = random.Random(seed)
    operations: list[Operation] = []
    single_qubit_gates = ("h", "s", "s_dag", "none")
    two_qubit_gates = ("cx", "cz", "swap", "none")

    for layer in range(depth):
        for qubit in range(n_qubits):
            gate = rng.choice(single_qubit_gates)
            if gate == "h":
                operations.append(Operation.h(qubit))
            elif gate == "s":
                operations.append(Operation.s(qubit))
            elif gate == "s_dag":
                operations.append(Operation.s_dag(qubit))

        pairing = list(range(n_qubits))
        rng.shuffle(pairing)
        for left, right in zip(pairing[0::2], pairing[1::2]):
            gate = rng.choice(two_qubit_gates)
            if gate == "cx":
                operations.append(Operation.cx(left, right))
            elif gate == "cz":
                operations.append(Operation.cz(left, right))
            elif gate == "swap":
                operations.append(Operation.swap(left, right))

        if noise_rate > 0.0:
            for qubit in range(n_qubits):
                operations.append(
                    Operation.noise(
                        NoiseLocation(
                            id=f"rc_l{layer}_q{qubit}_{noise_model}",
                            model=_random_clifford_noise(noise_model),
                            rate=noise_rate,
                            qubits=(qubit,),
                            tags={
                                "family": "random_clifford",
                                "layer": layer,
                                "qubit": qubit,
                                "operation": "layer_noise",
                                "noise_model": noise_model,
                            },
                        )
                    )
                )

    for qubit in range(n_qubits):
        measurement_noise = None
        if measurement_noise_rate > 0.0:
            measurement_noise = NoiseLocation(
                id=f"rc_m_q{qubit}",
                model=MeasurementBitFlip(),
                rate=measurement_noise_rate,
                qubits=(qubit,),
                tags={
                    "family": "random_clifford",
                    "qubit": qubit,
                    "operation": "measurement_noise",
                },
            )
        operations.append(
            Operation.measure(
                qubit,
                key=f"m{qubit}",
                noise=measurement_noise,
            )
        )

    return Circuit(n_qubits=n_qubits, operations=tuple(operations))


def _random_clifford_noise(noise_model: str) -> object:
    if noise_model == "depolarizing1":
        return SingleQubitDepolarizing()
    if noise_model == "x":
        return BernoulliPauliNoise("X")
    raise ValueError(f"unsupported random Clifford noise model {noise_model!r}")


def _noise_label(args: argparse.Namespace) -> str:
    if args.noise_rate <= 0:
        return "none"
    return f"{args.noise_model}:{args.noise_rate:.17g}"


def _median_samples_per_second(fn: Any, *, shots: int, repeats: int) -> float:
    values: list[float] = []
    for repeat in range(repeats):
        seed = 10_000 + repeat
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


def _sample_stim_packed(stim_sampler: Any, shots: int, seed: int) -> Any:
    del seed
    try:
        return stim_sampler.sample(shots=shots, bit_packed=True)
    except TypeError:
        return stim_sampler.sample(shots, bit_packed=True)


def _to_stim_text(circuit: Circuit) -> str:
    lines: list[str] = []
    for operation in circuit.operations:
        lines.extend(_operation_to_stim_lines(operation))
    return "\n".join(lines)


def _operation_to_stim_lines(operation: Operation) -> list[str]:
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
        return [_measurement_to_stim_line(operation)]
    if kind == "measure_pauli":
        return [_mpp_to_stim_line(operation)]
    if kind == "reset":
        basis = operation.basis.upper()
        name = {"Z": "R", "X": "RX", "Y": "RY"}[basis]
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
    if isinstance(model, SingleQubitDepolarizing) and len(location.qubits) == 1:
        return f"DEPOLARIZE1({location.rate:.17g}) {targets}"
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

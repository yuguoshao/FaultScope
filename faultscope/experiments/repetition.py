"""Repetition-code experiment builder for hotspot smoke tests and examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from faultscope.core import Circuit, NoiseLocation, Operation
from faultscope.decoders import RepetitionCodeDecoder
from faultscope.dem import Detector, LogicalObservable
from faultscope.core import BernoulliPauliNoise, MeasurementBitFlip


RateSpec = float | Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class RepetitionCodeExperiment:
    circuit: Circuit
    data_qubits: tuple[int, ...]
    ancilla_qubits: tuple[int, ...]
    detector_fn: Callable[[Any], list[int]]
    decoder: RepetitionCodeDecoder
    loss_fn: Callable[[Any, list[int]], float]
    detectors: tuple[Detector, ...]
    observables: tuple[LogicalObservable, ...]


def make_repetition_code_experiment(
    *,
    distance: int,
    rounds: int,
    data_error_rate: RateSpec,
    measurement_error_rate: RateSpec,
) -> RepetitionCodeExperiment:
    """Build a bit-flip repetition memory experiment.

    ``RateSpec`` can be a scalar or a mapping keyed by ``(round, index)``.
    Data-noise indices address data qubits, measurement-noise indices address
    check measurements.
    """

    if distance < 1 or distance % 2 != 1:
        raise ValueError("distance must be a positive odd integer")
    if rounds < 1:
        raise ValueError("rounds must be positive")

    data = tuple(range(distance))
    ancilla = tuple(range(distance, distance + distance - 1))
    operations: list[Operation] = []

    for round_idx in range(rounds):
        for data_idx, qubit in enumerate(data):
            rate = _lookup_rate(data_error_rate, round_idx, data_idx)
            location = NoiseLocation(
                id=f"data_r{round_idx}_q{data_idx}",
                model=BernoulliPauliNoise("X"),
                rate=rate,
                qubits=(qubit,),
                tags={
                    "round": round_idx,
                    "qubit": qubit,
                    "data_index": data_idx,
                    "gate": "idle",
                    "operation": "data_noise",
                },
            )
            operations.append(Operation.noise(location))

        for check_idx, measure_qubit in enumerate(ancilla):
            operations.append(Operation.reset(measure_qubit))
            operations.append(Operation.cx(data[check_idx], measure_qubit))
            operations.append(Operation.cx(data[check_idx + 1], measure_qubit))

            rate = _lookup_rate(measurement_error_rate, round_idx, check_idx)
            location = NoiseLocation(
                id=f"meas_r{round_idx}_c{check_idx}",
                model=MeasurementBitFlip(),
                rate=rate,
                qubits=(measure_qubit,),
                tags={
                    "round": round_idx,
                    "qubit": measure_qubit,
                    "check": check_idx,
                    "gate": "measure",
                    "operation": "measurement_noise",
                },
            )
            operations.append(
                Operation.measure(
                    measure_qubit,
                    key=f"r{round_idx}_c{check_idx}",
                    basis="Z",
                    noise=location,
                )
            )

    final_round = rounds - 1
    final_measurement_keys = tuple(
        f"r{final_round}_c{check_idx}"
        for check_idx in range(distance - 1)
    )
    circuit = Circuit(n_qubits=distance + distance - 1, operations=operations)
    decoder = RepetitionCodeDecoder(
        distance,
        measurement_keys=final_measurement_keys,
        observable_id=0,
    )
    detectors = tuple(
        Detector(
            id=round_idx * (distance - 1) + check_idx,
            measurement_keys=(
                (f"r{round_idx}_c{check_idx}",)
                if round_idx == 0
                else (
                    f"r{round_idx - 1}_c{check_idx}",
                    f"r{round_idx}_c{check_idx}",
                )
            ),
            coords=(check_idx + 0.5, round_idx),
        )
        for round_idx in range(rounds)
        for check_idx in range(distance - 1)
    )
    observables = (
        LogicalObservable(
            id=0,
            pauli_qubits=(data[0],),
            pauli="Z",
        ),
    )

    def detector_fn(trajectory: Any) -> list[int]:
        return [
            trajectory.measurement_by_key[key].bit
            for key in final_measurement_keys
        ]

    def loss_fn(trajectory: Any, correction: list[int]) -> float:
        return float(trajectory.frame.x[data[0]] ^ int(correction[0]))

    return RepetitionCodeExperiment(
        circuit=circuit,
        data_qubits=data,
        ancilla_qubits=ancilla,
        detector_fn=detector_fn,
        decoder=decoder,
        loss_fn=loss_fn,
        detectors=detectors,
        observables=observables,
    )


def _lookup_rate(rate_spec: RateSpec, round_idx: int, index: int) -> float:
    if isinstance(rate_spec, Mapping):
        return float(rate_spec.get((round_idx, index), 0.0))
    return float(rate_spec)

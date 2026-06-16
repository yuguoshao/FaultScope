"""Repetition-code experiment builder for hotspot smoke tests and examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from npsim.batch import BatchTrajectory
from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.decoders import RepetitionCodeDecoder
from npsim.dem import Detector, LogicalObservable
from npsim.noise import BernoulliPauliNoise, MeasurementBitFlip
from npsim.simulator import Trajectory


RateSpec = float | Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class RepetitionCodeExperiment:
    circuit: Circuit
    data_qubits: tuple[int, ...]
    ancilla_qubits: tuple[int, ...]
    detector_fn: Callable[[Trajectory], list[int]]
    decoder: RepetitionCodeDecoder
    loss_fn: Callable[[Trajectory, list[int]], float]
    batch_loss_mask_fn: Callable[[BatchTrajectory], int]
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

    circuit = Circuit(n_qubits=distance + distance - 1, operations=operations)
    decoder = RepetitionCodeDecoder(distance)
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

    def detector_fn(trajectory: Trajectory) -> list[int]:
        final_round = rounds - 1
        return [
            trajectory.measurement_by_key[f"r{final_round}_c{check_idx}"].bit
            for check_idx in range(distance - 1)
        ]

    def loss_fn(trajectory: Trajectory, correction: list[int]) -> float:
        residual = [
            trajectory.frame.x[qubit] ^ int(correction[data_idx])
            for data_idx, qubit in enumerate(data)
        ]
        return float(sum(residual) > distance // 2)

    def batch_loss_mask_fn(batch: BatchTrajectory) -> int:
        loss_mask = 0
        final_round = rounds - 1
        measurements = batch.measurements
        x_frame = batch.x_frame
        for shot in range(batch.shots):
            syndrome = [
                (measurements[f"r{final_round}_c{check_idx}"] >> shot) & 1
                for check_idx in range(distance - 1)
            ]
            correction = _decode_repetition_shot(syndrome)
            residual_weight = 0
            for data_idx, qubit in enumerate(data):
                residual_weight += ((x_frame[qubit] >> shot) & 1) ^ correction[data_idx]
            if residual_weight > distance // 2:
                loss_mask |= 1 << shot
        return loss_mask

    return RepetitionCodeExperiment(
        circuit=circuit,
        data_qubits=data,
        ancilla_qubits=ancilla,
        detector_fn=detector_fn,
        decoder=decoder,
        loss_fn=loss_fn,
        batch_loss_mask_fn=batch_loss_mask_fn,
        detectors=detectors,
        observables=observables,
    )


def _lookup_rate(rate_spec: RateSpec, round_idx: int, index: int) -> float:
    if isinstance(rate_spec, Mapping):
        return float(rate_spec.get((round_idx, index), 0.0))
    return float(rate_spec)


def _decode_repetition_shot(syndrome: list[int]) -> list[int]:
    candidate = [0] * (len(syndrome) + 1)
    for idx, bit in enumerate(syndrome):
        candidate[idx + 1] = candidate[idx] ^ int(bit)
    complement = [bit ^ 1 for bit in candidate]
    return candidate if sum(candidate) <= sum(complement) else complement

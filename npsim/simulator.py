"""Forward noise-aware stabilizer trajectory simulator."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.noise import MeasurementBitFlip
from npsim.pauli import PauliFrame, sparse_pauli_to_xz
from npsim.stabilizer import StabilizerState


class Decoder(Protocol):
    def decode(
        self,
        detector_record: Any,
        measurements: Mapping[str, "MeasurementRecord"],
        trajectory: "Trajectory",
    ) -> Any:
        ...


DetectorFn = Callable[["Trajectory"], Any]
LossFn = Callable[["Trajectory", Any], float]


@dataclass(frozen=True)
class MeasurementRecord:
    key: str
    bit: int
    raw_bit: int
    qubits: tuple[int, ...]
    basis: str
    flipped: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoiseEvent:
    location: NoiseLocation
    event: object
    score: float


@dataclass
class Trajectory:
    state: StabilizerState
    frame: PauliFrame
    measurements: list[MeasurementRecord]
    measurement_by_key: dict[str, MeasurementRecord]
    noise_events: list[NoiseEvent]
    scores: dict[str, float]
    detector_record: Any = None
    decoded: Any = None
    loss: float = 0.0


@dataclass(frozen=True)
class HotspotRow:
    location_id: str
    sensitivity: float
    hotspot: float
    qubits: tuple[int, ...]
    tags: Mapping[str, Any]


@dataclass
class SimulationResult:
    shots: int
    mean_loss: float
    baseline: float
    sensitivities: dict[str, float]
    hotspots: dict[str, float]
    by_qubit: dict[int, float]
    by_round: dict[Any, float]
    by_gate: dict[Any, float]
    by_operation: dict[Any, float]
    locations: dict[str, NoiseLocation]
    losses: list[float]

    @property
    def logical_failure_rate(self) -> float:
        return self.mean_loss

    def top_hotspots(self, top_k: int = 10) -> list[HotspotRow]:
        rows = [
            HotspotRow(
                location_id=location_id,
                sensitivity=self.sensitivities[location_id],
                hotspot=self.hotspots[location_id],
                qubits=self.locations[location_id].qubits,
                tags=self.locations[location_id].tags,
            )
            for location_id in self.hotspots
        ]
        rows.sort(key=lambda row: row.hotspot, reverse=True)
        return rows[:top_k]

    def hotspot_table(self, top_k: int = 10) -> str:
        lines = ["location_id\tsensitivity\thotspot\tqubits\ttags"]
        for row in self.top_hotspots(top_k):
            lines.append(
                f"{row.location_id}\t{row.sensitivity:.6g}\t"
                f"{row.hotspot:.6g}\t{row.qubits}\t{dict(row.tags)}"
            )
        return "\n".join(lines)


class ForwardNoiseAwareSimulator:
    """Execute stabilizer-compatible noisy circuits forward in time."""

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.locations = circuit.noise_locations()

    def run_shot(
        self,
        *,
        rng: random.Random,
        detector_fn: DetectorFn | None = None,
        decoder: Decoder | None = None,
        loss_fn: LossFn | None = None,
    ) -> Trajectory:
        state = StabilizerState.zero(self.circuit.n_qubits)
        frame = PauliFrame.zero(self.circuit.n_qubits)
        measurements: list[MeasurementRecord] = []
        measurement_by_key: dict[str, MeasurementRecord] = {}
        noise_events: list[NoiseEvent] = []
        scores: dict[str, float] = defaultdict(float)

        for operation in self.circuit.operations:
            self._apply_operation(
                operation,
                state,
                frame,
                measurements,
                measurement_by_key,
                noise_events,
                scores,
                rng,
            )

        trajectory = Trajectory(
            state=state,
            frame=frame,
            measurements=measurements,
            measurement_by_key=measurement_by_key,
            noise_events=noise_events,
            scores=dict(scores),
        )
        if detector_fn is not None:
            trajectory.detector_record = detector_fn(trajectory)
        if decoder is not None:
            trajectory.decoded = decoder.decode(
                trajectory.detector_record,
                trajectory.measurement_by_key,
                trajectory,
            )
        if loss_fn is not None:
            trajectory.loss = float(loss_fn(trajectory, trajectory.decoded))
        return trajectory

    def estimate(
        self,
        *,
        shots: int,
        seed: int | None = None,
        detector_fn: DetectorFn | None = None,
        decoder: Decoder | None = None,
        loss_fn: LossFn | None = None,
        baseline: str | float = "mean",
    ) -> SimulationResult:
        if shots <= 0:
            raise ValueError("shots must be positive")

        rng = random.Random(seed)
        losses: list[float] = []
        shot_scores: list[dict[str, float]] = []

        for _ in range(shots):
            trajectory = self.run_shot(
                rng=rng,
                detector_fn=detector_fn,
                decoder=decoder,
                loss_fn=loss_fn,
            )
            losses.append(trajectory.loss)
            shot_scores.append(trajectory.scores)

        if baseline == "mean":
            baseline_value = sum(losses) / shots
        elif isinstance(baseline, (int, float)):
            baseline_value = float(baseline)
        else:
            raise ValueError("baseline must be 'mean' or a numeric value")

        sensitivities = {location_id: 0.0 for location_id in self.locations}
        for loss, scores in zip(losses, shot_scores):
            centered_loss = loss - baseline_value
            for location_id in sensitivities:
                sensitivities[location_id] += centered_loss * scores.get(location_id, 0.0)
        for location_id in sensitivities:
            sensitivities[location_id] /= shots

        hotspots = {
            location_id: abs(sensitivity)
            for location_id, sensitivity in sensitivities.items()
        }

        return SimulationResult(
            shots=shots,
            mean_loss=sum(losses) / shots,
            baseline=baseline_value,
            sensitivities=sensitivities,
            hotspots=hotspots,
            by_qubit=self._aggregate_by_qubit(hotspots),
            by_round=self._aggregate_by_tag(hotspots, "round"),
            by_gate=self._aggregate_by_tag(hotspots, "gate"),
            by_operation=self._aggregate_by_tag(hotspots, "operation"),
            locations=self.locations.copy(),
            losses=losses,
        )

    def _apply_operation(
        self,
        operation: Operation,
        state: StabilizerState,
        frame: PauliFrame,
        measurements: list[MeasurementRecord],
        measurement_by_key: dict[str, MeasurementRecord],
        noise_events: list[NoiseEvent],
        scores: dict[str, float],
        rng: random.Random,
    ) -> None:
        kind = operation.kind
        if kind == "h":
            (qubit,) = operation.qubits
            state.apply_h(qubit)
            frame.apply_h(qubit)
            return
        if kind == "s":
            (qubit,) = operation.qubits
            state.apply_s(qubit)
            frame.apply_s(qubit)
            return
        if kind == "s_dag":
            (qubit,) = operation.qubits
            state.apply_s_dag(qubit)
            frame.apply_s_dag(qubit)
            return
        if kind == "cx":
            control, target = operation.qubits
            state.apply_cx(control, target)
            frame.apply_cx(control, target)
            return
        if kind == "cz":
            left, right = operation.qubits
            state.apply_cz(left, right)
            frame.apply_cz(left, right)
            return
        if kind == "swap":
            left, right = operation.qubits
            state.apply_swap(left, right)
            frame.apply_swap(left, right)
            return
        if kind == "pauli":
            if operation.pauli is None:
                raise ValueError("pauli operation requires a Pauli string")
            x, z = sparse_pauli_to_xz(
                self.circuit.n_qubits,
                operation.qubits,
                operation.pauli,
            )
            state.apply_pauli_string(x, z)
            return
        if kind == "noise":
            if operation.noise is None:
                raise ValueError("noise operation requires a noise location")
            self._sample_noise(operation.noise, state, frame, noise_events, scores, rng)
            return
        if kind == "measure":
            self._measure(
                operation,
                state,
                frame,
                measurements,
                measurement_by_key,
                noise_events,
                scores,
                rng,
            )
            return
        if kind == "measure_pauli":
            self._measure_pauli(
                operation,
                state,
                frame,
                measurements,
                measurement_by_key,
                noise_events,
                scores,
                rng,
            )
            return
        if kind == "reset":
            (qubit,) = operation.qubits
            basis = operation.basis.upper()
            if basis == "Z":
                bit = state.reset_z(qubit, rng)
            elif basis == "X":
                bit = state.reset_x(qubit, rng)
            elif basis == "Y":
                bit = state.reset_y(qubit, rng)
            else:
                raise ValueError(f"unsupported reset basis {operation.basis!r}")
            frame.reset(qubit)
            if operation.key is not None:
                self._record_measurement(
                    measurements,
                    measurement_by_key,
                    MeasurementRecord(
                        key=operation.key,
                        bit=bit,
                        raw_bit=bit,
                        qubits=operation.qubits,
                        basis=f"reset_{basis.lower()}",
                        metadata=operation.metadata,
                    ),
                )
            return
        raise ValueError(f"unsupported operation kind {kind!r}")

    def _sample_noise(
        self,
        location: NoiseLocation,
        state: StabilizerState,
        frame: PauliFrame,
        noise_events: list[NoiseEvent],
        scores: dict[str, float],
        rng: random.Random,
    ) -> object:
        event = location.model.sample(rng, location.rate)
        score = location.model.score(event, location.rate)
        location.model.apply(event, state, frame, location.qubits)
        scores[location.id] += score
        noise_events.append(NoiseEvent(location, event, score))
        return event

    def _measure(
        self,
        operation: Operation,
        state: StabilizerState,
        frame: PauliFrame,
        measurements: list[MeasurementRecord],
        measurement_by_key: dict[str, MeasurementRecord],
        noise_events: list[NoiseEvent],
        scores: dict[str, float],
        rng: random.Random,
    ) -> None:
        (qubit,) = operation.qubits
        basis = operation.basis.upper()
        if basis == "Z":
            raw_bit = state.measure_z(qubit, rng)
        elif basis == "X":
            raw_bit = state.measure_x(qubit, rng)
        elif basis == "Y":
            raw_bit = state.measure_y(qubit, rng)
        else:
            raise ValueError(f"unsupported measurement basis {operation.basis!r}")

        bit, flipped = self._apply_measurement_noise(
            raw_bit,
            operation.noise,
            state,
            frame,
            noise_events,
            scores,
            rng,
        )
        key = operation.key or f"m{len(measurements)}"
        self._record_measurement(
            measurements,
            measurement_by_key,
            MeasurementRecord(
                key=key,
                bit=bit,
                raw_bit=raw_bit,
                qubits=operation.qubits,
                basis=basis,
                flipped=flipped,
                metadata=operation.metadata,
            ),
        )

    def _measure_pauli(
        self,
        operation: Operation,
        state: StabilizerState,
        frame: PauliFrame,
        measurements: list[MeasurementRecord],
        measurement_by_key: dict[str, MeasurementRecord],
        noise_events: list[NoiseEvent],
        scores: dict[str, float],
        rng: random.Random,
    ) -> None:
        if operation.pauli is None:
            raise ValueError("measure_pauli operation requires a Pauli string")
        x, z = sparse_pauli_to_xz(self.circuit.n_qubits, operation.qubits, operation.pauli)
        raw_bit = state.measure_pauli(x, z, rng)
        bit, flipped = self._apply_measurement_noise(
            raw_bit,
            operation.noise,
            state,
            frame,
            noise_events,
            scores,
            rng,
        )
        key = operation.key or f"m{len(measurements)}"
        self._record_measurement(
            measurements,
            measurement_by_key,
            MeasurementRecord(
                key=key,
                bit=bit,
                raw_bit=raw_bit,
                qubits=operation.qubits,
                basis=operation.pauli,
                flipped=flipped,
                metadata=operation.metadata,
            ),
        )

    def _apply_measurement_noise(
        self,
        raw_bit: int,
        location: NoiseLocation | None,
        state: StabilizerState,
        frame: PauliFrame,
        noise_events: list[NoiseEvent],
        scores: dict[str, float],
        rng: random.Random,
    ) -> tuple[int, bool]:
        if location is None:
            return raw_bit, False
        event = location.model.sample(rng, location.rate)
        score = location.model.score(event, location.rate)
        if isinstance(location.model, MeasurementBitFlip):
            bit = location.model.apply_to_bit(raw_bit, event)
        elif hasattr(location.model, "apply_to_bit"):
            bit = location.model.apply_to_bit(raw_bit, event)
        else:
            location.model.apply(event, state, frame, location.qubits)
            bit = raw_bit
        scores[location.id] += score
        noise_events.append(NoiseEvent(location, event, score))
        return bit, bit != raw_bit

    @staticmethod
    def _record_measurement(
        measurements: list[MeasurementRecord],
        measurement_by_key: dict[str, MeasurementRecord],
        record: MeasurementRecord,
    ) -> None:
        if record.key in measurement_by_key:
            raise ValueError(f"duplicate measurement key {record.key!r}")
        measurements.append(record)
        measurement_by_key[record.key] = record

    def _aggregate_by_qubit(self, hotspots: Mapping[str, float]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for location_id, hotspot in hotspots.items():
            for qubit in self.locations[location_id].qubits:
                out[qubit] += hotspot
        return dict(out)

    def _aggregate_by_tag(self, hotspots: Mapping[str, float], tag: str) -> dict[Any, float]:
        out: dict[Any, float] = defaultdict(float)
        for location_id, hotspot in hotspots.items():
            value = self.locations[location_id].tags.get(tag)
            if value is not None:
                out[value] += hotspot
        return dict(out)

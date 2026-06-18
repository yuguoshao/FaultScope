"""Test-only Python bit-packed batch sampler reference."""

from __future__ import annotations

import random
import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from npsim.core import Circuit, NoiseLocation, Operation
from npsim.core import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from npsim.core import pauli_to_xz, sparse_pauli_to_xz
from npsim.runtime import SimulationResult
from tests.reference.batch_stabilizer import BatchStabilizerState


class UnsupportedBatchCircuitError(ValueError):
    """Raised when a circuit needs per-shot tableau branching."""


BatchLossMaskFn = Callable[..., int]
BatchCorrectionMaskFn = Callable[["BatchTrajectory"], Mapping[Any, int]]


@dataclass(frozen=True)
class BatchTrajectory:
    """Bit-packed result of executing many forward trajectories.

    The least-significant bit corresponds to shot 0. A bit value of 1 in
    ``x_frame[q]`` means shot ``k`` has an X component on qubit ``q`` in the
    final Pauli frame.
    """

    shots: int
    all_mask: int
    x_frame: tuple[int, ...]
    z_frame: tuple[int, ...]
    measurements: Mapping[str, int]
    detectors: Mapping[int, int]
    observables: Mapping[int, int]
    noise_event_masks: Mapping[str, int]

    def bit(self, mask: int, shot: int) -> int:
        return (mask >> shot) & 1

    def measurement_bit(self, key: str, shot: int) -> int:
        return self.bit(self.measurements[key], shot)

    def measurement_mask(self, key: str) -> int:
        return self.measurements[key]

    def detector_bit(self, detector_id: int, shot: int) -> int:
        return self.bit(self.detectors[detector_id], shot)

    def observable_bit(self, observable_id: int, shot: int) -> int:
        return self.bit(self.observables[observable_id], shot)

    def x_bit(self, qubit: int, shot: int) -> int:
        return self.bit(self.x_frame[qubit], shot)

    def x_mask(self, qubit: int) -> int:
        return self.x_frame[qubit]

    def z_bit(self, qubit: int, shot: int) -> int:
        return self.bit(self.z_frame[qubit], shot)

    def z_mask(self, qubit: int) -> int:
        return self.z_frame[qubit]


@dataclass
class _BatchState:
    ideal_state: BatchStabilizerState
    x_frame: list[int]
    z_frame: list[int]
    measurements: dict[str, int]
    detectors: dict[int, int]
    observables: dict[int, int]
    event_masks: dict[str, int]


class BatchForwardNoiseAwareSimulator:
    """Fast bit-packed sampler for stabilizer-compatible QEC circuits.

    This engine keeps one ideal stabilizer tableau and tracks all noisy shots in
    bit-packed stabilizer signs and Pauli frames. It supports random Pauli
    measurements as long as all shots share the same Clifford/stabilizer support
    evolution.
    """

    def __init__(self, circuit: Circuit, *, observables: Any | None = None):
        self.circuit = circuit
        self.observables = tuple(observables) if observables is not None else ()
        self.locations = circuit.noise_locations()
        self._ensure_unique_noise_location_ids()

    def estimate(
        self,
        *,
        shots: int,
        loss_mask_fn: BatchLossMaskFn | None = None,
        decoder: Any | None = None,
        correction_mask_fn: BatchCorrectionMaskFn | None = None,
        seed: int | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> SimulationResult:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if decoder is not None and correction_mask_fn is not None:
            raise ValueError("supply either decoder or correction_mask_fn, not both")

        rng = random.Random(seed)
        batch = self.run_batch(shots=shots, rng=rng)
        corrections = _correction_masks(batch, decoder, correction_mask_fn)
        if loss_mask_fn is None:
            loss_mask = _default_loss_mask(batch, corrections)
        else:
            loss_mask = _call_loss_mask_fn(loss_mask_fn, batch, corrections)
        loss_mask &= batch.all_mask
        loss_count = loss_mask.bit_count()
        mean_loss = loss_count / shots

        if baseline == "mean":
            baseline_value = mean_loss
        elif isinstance(baseline, (int, float)):
            baseline_value = float(baseline)
        else:
            raise ValueError("baseline must be 'mean' or a numeric value")

        sensitivities: dict[str, float] = {}
        for location_id, location in self.locations.items():
            event_mask = batch.noise_event_masks.get(location_id, 0) & batch.all_mask
            event_score, no_event_score = _score_pair(location)
            event_count = event_mask.bit_count()
            loss_event_count = (loss_mask & event_mask).bit_count()
            loss_no_event_count = loss_count - loss_event_count
            no_event_count = shots - event_count

            sum_loss_score = (
                loss_event_count * event_score
                + loss_no_event_count * no_event_score
            )
            sum_score = event_count * event_score + no_event_count * no_event_score
            sensitivities[location_id] = (
                sum_loss_score - baseline_value * sum_score
            ) / shots

        hotspots = {
            location_id: abs(sensitivity)
            for location_id, sensitivity in sensitivities.items()
        }
        return SimulationResult(
            shots=shots,
            mean_loss=mean_loss,
            baseline=baseline_value,
            sensitivities=sensitivities,
            hotspots=hotspots,
            by_qubit=self._aggregate_by_qubit(hotspots),
            by_round=self._aggregate_by_tag(hotspots, "round"),
            by_gate=self._aggregate_by_tag(hotspots, "gate"),
            by_operation=self._aggregate_by_tag(hotspots, "operation"),
            locations=self.locations.copy(),
            losses=[],
        )

    def run_batch(self, *, shots: int, rng: random.Random) -> BatchTrajectory:
        if shots <= 0:
            raise ValueError("shots must be positive")
        all_mask = (1 << shots) - 1
        state = _BatchState(
            ideal_state=BatchStabilizerState.zero(
                self.circuit.n_qubits,
                shots,
                all_mask,
            ),
            x_frame=[0] * self.circuit.n_qubits,
            z_frame=[0] * self.circuit.n_qubits,
            measurements={},
            detectors={},
            observables={},
            event_masks={location_id: 0 for location_id in self.locations},
        )

        for operation in self.circuit.operations:
            self._apply_operation(operation, state, shots, all_mask, rng)

        for observable in self.observables:
            state.observables[int(observable.id)] = (
                _observable_mask(observable, state) & all_mask
            )

        return BatchTrajectory(
            shots=shots,
            all_mask=all_mask,
            x_frame=tuple(mask & all_mask for mask in state.x_frame),
            z_frame=tuple(mask & all_mask for mask in state.z_frame),
            measurements=state.measurements,
            detectors=state.detectors,
            observables=state.observables,
            noise_event_masks=state.event_masks,
        )

    def _apply_operation(
        self,
        operation: Operation,
        state: _BatchState,
        shots: int,
        all_mask: int,
        rng: random.Random,
    ) -> None:
        kind = operation.kind
        if kind == "h":
            (qubit,) = operation.qubits
            state.ideal_state.apply_h(qubit)
            state.x_frame[qubit], state.z_frame[qubit] = (
                state.z_frame[qubit],
                state.x_frame[qubit],
            )
            return
        if kind == "s":
            (qubit,) = operation.qubits
            state.ideal_state.apply_s(qubit)
            state.z_frame[qubit] ^= state.x_frame[qubit]
            return
        if kind == "s_dag":
            (qubit,) = operation.qubits
            state.ideal_state.apply_s_dag(qubit)
            state.z_frame[qubit] ^= state.x_frame[qubit]
            return
        if kind == "cx":
            control, target = operation.qubits
            state.ideal_state.apply_cx(control, target)
            state.x_frame[target] ^= state.x_frame[control]
            state.z_frame[control] ^= state.z_frame[target]
            return
        if kind == "cz":
            left, right = operation.qubits
            state.ideal_state.apply_cz(left, right)
            state.z_frame[left] ^= state.x_frame[right]
            state.z_frame[right] ^= state.x_frame[left]
            return
        if kind == "swap":
            left, right = operation.qubits
            state.ideal_state.apply_swap(left, right)
            state.x_frame[left], state.x_frame[right] = (
                state.x_frame[right],
                state.x_frame[left],
            )
            state.z_frame[left], state.z_frame[right] = (
                state.z_frame[right],
                state.z_frame[left],
            )
            return
        if kind == "pauli":
            if operation.pauli is None:
                raise ValueError("pauli operation requires a Pauli string")
            x, z = sparse_pauli_to_xz(
                self.circuit.n_qubits,
                operation.qubits,
                operation.pauli,
            )
            state.ideal_state.apply_pauli_string(x, z)
            return
        if kind == "noise":
            if operation.noise_location is None:
                raise ValueError("noise operation requires a noise location")
            self._sample_noise(operation.noise_location, state, shots, all_mask, rng)
            return
        if kind == "measure":
            self._measure(operation, state, shots, all_mask, rng)
            return
        if kind == "measure_pauli":
            self._measure_pauli(operation, state, shots, all_mask, rng)
            return
        if kind == "reset":
            self._reset(operation, state, all_mask, rng)
            return
        if kind == "detector":
            detector_id = operation.metadata.get("detector_id")
            if detector_id is None:
                detector_id = len(state.detectors)
            state.detectors[int(detector_id)] = _measurement_mask_parity(
                state.measurements,
                operation.measurement_keys,
            ) & all_mask
            return
        if kind == "observable_include":
            if operation.observable_id is None:
                raise ValueError("observable_include operation requires observable_id")
            value = _measurement_mask_parity(
                state.measurements,
                operation.measurement_keys,
            )
            state.observables[operation.observable_id] = (
                state.observables.get(operation.observable_id, 0) ^ value
            ) & all_mask
            return
        raise ValueError(f"unsupported operation kind {kind!r}")

    def _sample_noise(
        self,
        location: NoiseLocation,
        state: _BatchState,
        shots: int,
        all_mask: int,
        rng: random.Random,
    ) -> None:
        event_masks = _sample_noise_event_masks(location, shots, rng)
        error_mask = 0
        for pauli, mask in event_masks.items():
            mask &= all_mask
            error_mask |= mask
            _apply_masked_pauli_to_frame(state, location.qubits, pauli, mask)
        state.event_masks[location.id] ^= error_mask & all_mask

    def _measure(
        self,
        operation: Operation,
        state: _BatchState,
        shots: int,
        all_mask: int,
        rng: random.Random,
    ) -> None:
        basis = operation.basis.upper()
        x, z = sparse_pauli_to_xz(self.circuit.n_qubits, operation.qubits, basis)
        bit_mask = state.ideal_state.measure_pauli_mask(x, z, rng)
        bit_mask ^= _frame_measurement_flip(state, operation.qubits, basis)
        if operation.noise_location is not None:
            bit_mask ^= self._sample_measurement_noise(operation.noise_location, state, shots, all_mask, rng)
        key = operation.key or f"m{len(state.measurements)}"
        self._record_measurement(state.measurements, key, bit_mask & all_mask)

    def _measure_pauli(
        self,
        operation: Operation,
        state: _BatchState,
        shots: int,
        all_mask: int,
        rng: random.Random,
    ) -> None:
        if operation.pauli is None:
            raise ValueError("measure_pauli operation requires a Pauli string")
        x, z = sparse_pauli_to_xz(self.circuit.n_qubits, operation.qubits, operation.pauli)
        bit_mask = state.ideal_state.measure_pauli_mask(x, z, rng)
        bit_mask ^= _frame_measurement_flip(state, operation.qubits, operation.pauli)
        if operation.noise_location is not None:
            bit_mask ^= self._sample_measurement_noise(operation.noise_location, state, shots, all_mask, rng)
        key = operation.key or f"m{len(state.measurements)}"
        self._record_measurement(state.measurements, key, bit_mask & all_mask)

    def _reset(
        self,
        operation: Operation,
        state: _BatchState,
        all_mask: int,
        rng: random.Random,
    ) -> None:
        (qubit,) = operation.qubits
        basis = operation.basis.upper()
        x, z = sparse_pauli_to_xz(self.circuit.n_qubits, operation.qubits, basis)
        outcome_mask = state.ideal_state.measure_pauli_mask(x, z, rng)
        if operation.key is not None:
            bit_mask = outcome_mask ^ _frame_measurement_flip(
                state,
                operation.qubits,
                basis,
            )
            self._record_measurement(state.measurements, operation.key, bit_mask & all_mask)

        if basis == "Z":
            correction = "X"
        elif basis == "X":
            correction = "Z"
        elif basis == "Y":
            correction = "X"
        else:
            raise ValueError(f"unsupported reset basis {operation.basis!r}")
        if outcome_mask:
            x_correction, z_correction = sparse_pauli_to_xz(
                self.circuit.n_qubits,
                operation.qubits,
                correction,
            )
            state.ideal_state.apply_pauli_string_masked(
                x_correction,
                z_correction,
                outcome_mask,
            )
        state.x_frame[qubit] = 0
        state.z_frame[qubit] = 0

    def _sample_measurement_noise(
        self,
        location: NoiseLocation,
        state: _BatchState,
        shots: int,
        all_mask: int,
        rng: random.Random,
    ) -> int:
        if not isinstance(location.model, MeasurementBitFlip):
            raise UnsupportedBatchCircuitError(
                "batch measurement noise currently supports MeasurementBitFlip only"
            )
        flip_mask = _bernoulli_mask(rng, shots, location.rate) & all_mask
        state.event_masks[location.id] ^= flip_mask
        return flip_mask

    @staticmethod
    def _record_measurement(measurements: dict[str, int], key: str, bit_mask: int) -> None:
        if key in measurements:
            raise ValueError(f"duplicate measurement key {key!r}")
        measurements[key] = bit_mask

    def _ensure_unique_noise_location_ids(self) -> None:
        seen: set[str] = set()
        for operation in self.circuit.operations:
            if operation.kind not in {"noise", "measure", "measure_pauli"}:
                continue
            location = operation.noise_location
            if location is None:
                continue
            if location.id in seen:
                raise ValueError(
                    "BatchForwardNoiseAwareSimulator requires one unique id per "
                    f"noise operation; duplicate id {location.id!r}"
                )
            seen.add(location.id)

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


def _sample_noise_event_masks(
    location: NoiseLocation,
    shots: int,
    rng: random.Random,
) -> dict[str, int]:
    _validate_rate(location.rate)
    model = location.model
    if isinstance(model, MeasurementBitFlip):
        raise UnsupportedBatchCircuitError(
            "MeasurementBitFlip must be attached to a measurement operation"
        )
    if isinstance(model, BernoulliPauliNoise):
        return {model.pauli: _bernoulli_mask(rng, shots, location.rate)}
    if isinstance(model, SingleQubitDepolarizing):
        return _sample_weighted_pauli_masks(
            ("X", "Y", "Z"),
            (1.0, 1.0, 1.0),
            location.rate,
            shots,
            rng,
        )
    if isinstance(model, TwoQubitDepolarizing):
        return _sample_weighted_pauli_masks(
            model._events,
            (1.0,) * len(model._events),
            location.rate,
            shots,
            rng,
        )
    if isinstance(model, PauliChannel):
        events = tuple(model.weights.keys())
        weights = tuple(model.weights[event] for event in events)
        return _sample_weighted_pauli_masks(events, weights, location.rate, shots, rng)
    raise UnsupportedBatchCircuitError(
        f"unsupported batch noise model {type(model).__name__}"
    )


def _sample_weighted_pauli_masks(
    events: tuple[str, ...],
    weights: tuple[float, ...],
    rate: float,
    shots: int,
    rng: random.Random,
) -> dict[str, int]:
    _validate_rate(rate)
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weighted Pauli event weights must be positive")
    masks = {event: 0 for event in events}
    for shot in range(shots):
        if rng.random() >= rate:
            continue
        threshold = rng.random() * total_weight
        acc = 0.0
        chosen = events[-1]
        for event, weight in zip(events, weights):
            if weight == 0:
                continue
            acc += weight
            if threshold <= acc:
                chosen = event
                break
        masks[chosen] |= 1 << shot
    return masks


def _bernoulli_mask(rng: random.Random, shots: int, rate: float) -> int:
    _validate_rate(rate)
    if rate <= 0:
        return 0
    if rate >= 1:
        return (1 << shots) - 1
    mask = 0
    for shot in range(shots):
        if rng.random() < rate:
            mask |= 1 << shot
    return mask


def _validate_rate(rate: float) -> None:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"noise rate must be in [0, 1], got {rate}")


def _correction_masks(
    batch: BatchTrajectory,
    decoder: Any | None,
    correction_mask_fn: BatchCorrectionMaskFn | None,
) -> Mapping[int, int]:
    if correction_mask_fn is not None:
        return dict(correction_mask_fn(batch))
    if decoder is None:
        return {}
    if not hasattr(decoder, "decode_batch_masks"):
        raise TypeError("batch decoder must provide decode_batch_masks(batch)")
    return dict(decoder.decode_batch_masks(batch))


def _default_loss_mask(
    batch: BatchTrajectory,
    corrections: Mapping[Any, int],
) -> int:
    observable_ids = set(batch.observables)
    observable_ids.update(corrections)
    loss_mask = 0
    for observable_id in observable_ids:
        if not isinstance(observable_id, int):
            raise TypeError(
                "default batch loss requires observable-id correction masks; "
                "supply loss_mask_fn for data-qubit corrections"
            )
        loss_mask |= batch.observables.get(observable_id, 0) ^ corrections.get(
            observable_id,
            0,
        )
    return loss_mask & batch.all_mask


def _call_loss_mask_fn(
    loss_mask_fn: Callable[..., int],
    batch: BatchTrajectory,
    corrections: Mapping[Any, int],
) -> int:
    if _accepts_positional_args(loss_mask_fn, 2):
        return int(loss_mask_fn(batch, corrections))
    return int(loss_mask_fn(batch))


def _accepts_positional_args(fn: Callable[..., object], count: int) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= count


def _apply_masked_pauli_to_frame(
    state: _BatchState,
    qubits: tuple[int, ...],
    pauli: str,
    mask: int,
) -> None:
    if mask == 0:
        return
    for qubit, local_pauli in zip(qubits, pauli):
        x, z = pauli_to_xz(local_pauli)
        if x:
            state.x_frame[qubit] ^= mask
        if z:
            state.z_frame[qubit] ^= mask


def _frame_measurement_flip(
    state: _BatchState,
    qubits: tuple[int, ...],
    pauli: str,
) -> int:
    flip_mask = 0
    for qubit, local_pauli in zip(qubits, pauli):
        x, z = pauli_to_xz(local_pauli)
        if z:
            flip_mask ^= state.x_frame[qubit]
        if x:
            flip_mask ^= state.z_frame[qubit]
    return flip_mask


def _score_pair(location: NoiseLocation) -> tuple[float, float]:
    model = location.model
    if isinstance(model, BernoulliPauliNoise):
        return model.score(model.pauli, location.rate), model.score(
            "I" * len(model.pauli),
            location.rate,
        )
    if isinstance(model, SingleQubitDepolarizing):
        return model.score("X", location.rate), model.score("I", location.rate)
    if isinstance(model, TwoQubitDepolarizing):
        return model.score("IX", location.rate), model.score("II", location.rate)
    if isinstance(model, MeasurementBitFlip):
        return model.score(True, location.rate), model.score(False, location.rate)
    if isinstance(model, PauliChannel):
        error_event = next(iter(model.weights))
        return model.score(error_event, location.rate), model.score(
            "I" * model.event_length,
            location.rate,
        )
    raise UnsupportedBatchCircuitError(
        f"unsupported batch noise model {type(model).__name__}"
    )


def _measurement_mask_parity(
    measurements: Mapping[str, int],
    keys: tuple[str, ...],
) -> int:
    parity = 0
    for key in keys:
        try:
            parity ^= measurements[key]
        except KeyError as exc:
            raise ValueError(f"unknown measurement key {key!r}") from exc
    return parity


def _observable_mask(observable: Any, state: _BatchState) -> int:
    value = _measurement_mask_parity(
        state.measurements,
        tuple(str(key) for key in observable.measurement_keys),
    )
    if observable.pauli:
        value ^= _frame_measurement_flip(
            state,
            tuple(int(qubit) for qubit in observable.pauli_qubits),
            str(observable.pauli),
        )
    return value

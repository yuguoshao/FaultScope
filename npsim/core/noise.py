"""Stabilizer-compatible stochastic noise models with score functions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from npsim.core.pauli import PauliFrame, sparse_pauli_to_xz
from npsim.core.stabilizer import StabilizerState


MIN_SCORE_RATE = 1e-12


def _validate_rate(rate: float) -> None:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"noise rate must be in [0, 1], got {rate}")


def _score_rate(rate: float) -> float:
    return min(1.0 - MIN_SCORE_RATE, max(MIN_SCORE_RATE, rate))


class StochasticNoise(Protocol):
    """Noise model interface used by the forward trajectory simulator."""

    def sample(self, rng: random.Random, rate: float) -> object:
        ...

    def score(self, event: object, rate: float) -> float:
        ...

    def apply(
        self,
        event: object,
        state: StabilizerState,
        frame: PauliFrame,
        qubits: Sequence[int],
    ) -> None:
        ...


@dataclass(frozen=True)
class BernoulliPauliNoise:
    """Apply a fixed Pauli string with probability ``rate``."""

    pauli: str

    def sample(self, rng: random.Random, rate: float) -> str:
        _validate_rate(rate)
        return self.pauli if rng.random() < rate else "I" * len(self.pauli)

    def score(self, event: object, rate: float) -> float:
        rate = _score_rate(rate)
        no_error = "I" * len(self.pauli)
        return -1.0 / (1.0 - rate) if event == no_error else 1.0 / rate

    def apply(
        self,
        event: object,
        state: StabilizerState,
        frame: PauliFrame,
        qubits: Sequence[int],
    ) -> None:
        pauli = str(event)
        if len(pauli) != len(qubits):
            raise ValueError("event Pauli length does not match qubits")
        x, z = sparse_pauli_to_xz(state.n_qubits, qubits, pauli)
        state.apply_pauli_string(x, z)
        frame.apply_pauli_string(qubits, pauli)


@dataclass(frozen=True)
class PauliChannel:
    """A fixed non-identity Pauli mixture controlled by one total error rate.

    ``weights`` maps Pauli strings to relative probabilities conditioned on an
    error occurring. Identity events are supplied by the no-error branch with
    probability ``1 - rate``.
    """

    weights: dict[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("PauliChannel requires at least one non-identity event")
        total = 0.0
        length: int | None = None
        for pauli, weight in self.weights.items():
            if set(pauli) - {"I", "X", "Y", "Z"}:
                raise ValueError(f"unsupported Pauli string {pauli!r}")
            if set(pauli) == {"I"}:
                raise ValueError("identity should not appear in PauliChannel weights")
            if weight < 0:
                raise ValueError("PauliChannel weights must be non-negative")
            if length is None:
                length = len(pauli)
            elif len(pauli) != length:
                raise ValueError("all PauliChannel events must have the same length")
            total += weight
        if total <= 0:
            raise ValueError("PauliChannel weights must have positive total weight")

    def sample(self, rng: random.Random, rate: float) -> str:
        _validate_rate(rate)
        if rng.random() >= rate:
            return "I" * self.event_length

        threshold = rng.random() * self.total_weight
        acc = 0.0
        for pauli, weight in self.weights.items():
            if weight == 0:
                continue
            acc += weight
            if threshold <= acc:
                return pauli
        return next(reversed(self.weights))

    def score(self, event: object, rate: float) -> float:
        rate = _score_rate(rate)
        return -1.0 / (1.0 - rate) if event == "I" * self.event_length else 1.0 / rate

    def apply(
        self,
        event: object,
        state: StabilizerState,
        frame: PauliFrame,
        qubits: Sequence[int],
    ) -> None:
        pauli = str(event)
        if len(pauli) != len(qubits):
            raise ValueError("event Pauli length does not match qubits")
        x, z = sparse_pauli_to_xz(state.n_qubits, qubits, pauli)
        state.apply_pauli_string(x, z)
        frame.apply_pauli_string(qubits, pauli)

    @property
    def event_length(self) -> int:
        return len(next(iter(self.weights)))

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())


@dataclass(frozen=True)
class SingleQubitDepolarizing:
    """Single-qubit depolarizing channel with total error probability ``rate``."""

    def sample(self, rng: random.Random, rate: float) -> str:
        _validate_rate(rate)
        if rng.random() >= rate:
            return "I"
        return ("X", "Y", "Z")[rng.randrange(3)]

    def score(self, event: object, rate: float) -> float:
        rate = _score_rate(rate)
        return -1.0 / (1.0 - rate) if event == "I" else 1.0 / rate

    def apply(
        self,
        event: object,
        state: StabilizerState,
        frame: PauliFrame,
        qubits: Sequence[int],
    ) -> None:
        if len(qubits) != 1:
            raise ValueError("single-qubit depolarizing noise requires one qubit")
        pauli = str(event)
        x, z = sparse_pauli_to_xz(state.n_qubits, qubits, pauli)
        state.apply_pauli_string(x, z)
        frame.apply_pauli_string(qubits, pauli)


@dataclass(frozen=True)
class TwoQubitDepolarizing:
    """Two-qubit depolarizing channel with total non-identity probability ``rate``."""

    _events: tuple[str, ...] = (
        "IX",
        "IY",
        "IZ",
        "XI",
        "XX",
        "XY",
        "XZ",
        "YI",
        "YX",
        "YY",
        "YZ",
        "ZI",
        "ZX",
        "ZY",
        "ZZ",
    )

    def sample(self, rng: random.Random, rate: float) -> str:
        _validate_rate(rate)
        if rng.random() >= rate:
            return "II"
        return self._events[rng.randrange(len(self._events))]

    def score(self, event: object, rate: float) -> float:
        rate = _score_rate(rate)
        return -1.0 / (1.0 - rate) if event == "II" else 1.0 / rate

    def apply(
        self,
        event: object,
        state: StabilizerState,
        frame: PauliFrame,
        qubits: Sequence[int],
    ) -> None:
        if len(qubits) != 2:
            raise ValueError("two-qubit depolarizing noise requires two qubits")
        pauli = str(event)
        x, z = sparse_pauli_to_xz(state.n_qubits, qubits, pauli)
        state.apply_pauli_string(x, z)
        frame.apply_pauli_string(qubits, pauli)


@dataclass(frozen=True)
class MeasurementBitFlip:
    """Classical measurement bit-flip noise."""

    def sample(self, rng: random.Random, rate: float) -> bool:
        _validate_rate(rate)
        return rng.random() < rate

    def score(self, event: object, rate: float) -> float:
        rate = _score_rate(rate)
        return 1.0 / rate if bool(event) else -1.0 / (1.0 - rate)

    def apply(
        self,
        event: object,
        state: StabilizerState,
        frame: PauliFrame,
        qubits: Sequence[int],
    ) -> None:
        del event, state, frame, qubits

    def apply_to_bit(self, bit: int, event: object) -> int:
        return bit ^ int(bool(event))

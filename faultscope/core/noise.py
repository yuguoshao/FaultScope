"""Stabilizer-compatible stochastic noise model API."""

from __future__ import annotations

import random
from typing import Protocol, Sequence

from faultscope._native import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from faultscope.core.pauli import PauliFrame
from faultscope.core.stabilizer import StabilizerState


class StochasticNoise(Protocol):
    """Noise model interface used by the Python reference simulators."""

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


__all__ = [
    "BernoulliPauliNoise",
    "MeasurementBitFlip",
    "PauliChannel",
    "SingleQubitDepolarizing",
    "StochasticNoise",
    "TwoQubitDepolarizing",
]

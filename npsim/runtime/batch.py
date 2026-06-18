"""Bit-packed batch sampler for forward Pauli-frame QEC trajectories."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from npsim._npsim_native import BatchForwardNoiseAwareSimulator, BatchTrajectory


class UnsupportedBatchCircuitError(ValueError):
    """Raised when a circuit needs per-shot tableau branching."""


BatchLossMaskFn = Callable[..., int]
BatchCorrectionMaskFn = Callable[["BatchTrajectory"], Mapping[Any, int]]

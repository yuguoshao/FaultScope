"""Bit-packed batch sampler for forward Pauli-frame QEC trajectories."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from faultscope._native import FaultScopeSimulator, SampleBatch


class UnsupportedBatchCircuitError(ValueError):
    """Raised when a circuit needs per-shot tableau branching."""


BatchLossMaskFn = Callable[..., int]
BatchCorrectionMaskFn = Callable[["SampleBatch"], Mapping[Any, int]]

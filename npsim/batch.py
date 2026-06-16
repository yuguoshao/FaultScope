"""Bit-packed batch sampler for forward Pauli-frame QEC trajectories."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from npsim.circuit import Circuit
from npsim.simulator import SimulationResult


class UnsupportedBatchCircuitError(ValueError):
    """Raised when a circuit needs per-shot tableau branching."""


BatchLossMaskFn = Callable[["BatchTrajectory"], int]
BatchCorrectionMaskFn = Callable[["BatchTrajectory"], Mapping[int, int]]


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


class BatchForwardNoiseAwareSimulator:
    """Native-backed bit-packed sampler for stabilizer-compatible QEC circuits.

    The Rust extension is required for batch execution.  Passing ``rng`` to
    ``run_batch`` derives a 64-bit native seed from that RNG, preserving the
    requested random distribution without preserving the removed Python
    bit-for-bit sequence.
    """

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.locations = circuit.noise_locations()
        self._native_sampler_cache: Any | None = None
        self._ensure_unique_noise_location_ids()

    def _native_sampler(self) -> Any:
        if self._native_sampler_cache is None:
            from npsim.native import compile_native_sampler

            self._native_sampler_cache = compile_native_sampler(
                self.circuit,
                backend="native",
            )
        return self._native_sampler_cache

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

        return self._native_sampler().estimate(
            shots=shots,
            loss_mask_fn=loss_mask_fn,
            decoder=decoder,
            correction_mask_fn=correction_mask_fn,
            seed=seed,
            baseline=baseline,
            top_k=top_k,
        )

    def run_batch(
        self,
        *,
        shots: int,
        rng: random.Random | None = None,
        seed: int | None = None,
    ) -> BatchTrajectory:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if rng is not None and seed is not None:
            raise ValueError("supply either seed or rng, not both")
        if rng is not None:
            seed = rng.getrandbits(64)
        return self._native_sampler().sample(shots=shots, seed=seed)

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

"""Optional native packed sampler API.

The public entry point in this module is intentionally stable even when the
Rust extension is not installed.  ``backend="auto"`` tries the native extension
first and falls back to the existing Python bit-packed sampler.
"""

from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from typing import Any, Mapping

from npsim.batch import BatchForwardNoiseAwareSimulator, BatchTrajectory
from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.noise import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)


class UnsupportedNativeCircuitError(ValueError):
    """Raised when a circuit cannot be compiled by the native sampler."""


@dataclass(frozen=True)
class NativePackedSampler:
    """Compiled packed sampler wrapper.

    ``backend_name`` is ``"native"`` when the Rust extension is active and
    ``"python"`` when the object is using the compatibility fallback.
    """

    circuit: Circuit
    backend_name: str
    _engine: Any

    @property
    def is_native(self) -> bool:
        return self.backend_name == "native"

    def sample(
        self,
        *,
        shots: int,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> BatchTrajectory:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if self.backend_name == "native":
            if rng is not None:
                raise ValueError("native sampler accepts seed, not a Python rng")
            payload = self._engine.sample(int(shots), seed)
            return _payload_to_batch_trajectory(payload)
        if rng is None:
            rng = random.Random(seed)
        return self._engine.run_batch(shots=shots, rng=rng)

    def sample_measurements(
        self,
        *,
        shots: int,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> dict[str, int]:
        """Return only packed measurement masks for ordinary sampling benchmarks."""

        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if self.backend_name == "native" and hasattr(self._engine, "sample_measurements"):
            if rng is not None:
                raise ValueError("native sampler accepts seed, not a Python rng")
            payload = self._engine.sample_measurements(int(shots), seed)
            return {str(key): int(value) for key, value in payload.items()}
        return self.sample(shots=shots, seed=seed, rng=rng).measurements


def compile_native_sampler(
    circuit: Circuit,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> NativePackedSampler:
    """Compile ``circuit`` into a packed sampler.

    Parameters
    ----------
    backend:
        ``"auto"`` tries the Rust extension and falls back to Python.
        ``"native"`` requires the Rust extension.
        ``"python"`` forces the existing Python bit-packed sampler.
    strict:
        When true with ``backend="auto"``, native compile/import failures are
        surfaced instead of falling back.
    """

    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be 'auto', 'native', or 'python'")
    if backend == "python":
        return _python_sampler(circuit)

    try:
        spec = _serialize_circuit(circuit)
        native_mod = importlib.import_module("npsim._npsim_native")
        engine = native_mod.compile_sampler(spec)
    except Exception as exc:
        if backend == "native" or strict:
            raise UnsupportedNativeCircuitError(str(exc)) from exc
        return _python_sampler(circuit)

    return NativePackedSampler(
        circuit=circuit,
        backend_name="native",
        _engine=engine,
    )


def _python_sampler(circuit: Circuit) -> NativePackedSampler:
    return NativePackedSampler(
        circuit=circuit,
        backend_name="python",
        _engine=BatchForwardNoiseAwareSimulator(circuit),
    )


def _serialize_circuit(circuit: Circuit) -> dict[str, Any]:
    return {
        "n_qubits": int(circuit.n_qubits),
        "operations": [
            _serialize_operation(operation)
            for operation in circuit.operations
        ],
    }


def _serialize_operation(operation: Operation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": operation.kind,
        "qubits": tuple(int(qubit) for qubit in operation.qubits),
        "basis": operation.basis,
        "metadata": dict(operation.metadata),
        "measurement_keys": tuple(operation.measurement_keys),
    }
    if operation.key is not None:
        out["key"] = operation.key
    if operation.pauli is not None:
        out["pauli"] = operation.pauli
    if operation.observable_id is not None:
        out["observable_id"] = int(operation.observable_id)
    if operation.noise_location is not None:
        out["noise_location"] = _serialize_noise_location(operation.noise_location)
    return out


def _serialize_noise_location(location: NoiseLocation) -> dict[str, Any]:
    return {
        "id": location.id,
        "rate": float(location.rate),
        "qubits": tuple(int(qubit) for qubit in location.qubits),
        "tags": dict(location.tags),
        "model": _serialize_noise_model(location.model),
    }


def _serialize_noise_model(model: object) -> dict[str, Any]:
    if isinstance(model, BernoulliPauliNoise):
        return {"type": "bernoulli_pauli", "pauli": model.pauli}
    if isinstance(model, MeasurementBitFlip):
        return {"type": "measurement_bit_flip"}
    if isinstance(model, SingleQubitDepolarizing):
        return {"type": "single_qubit_depolarizing"}
    if isinstance(model, TwoQubitDepolarizing):
        return {"type": "two_qubit_depolarizing"}
    if isinstance(model, PauliChannel):
        return {
            "type": "pauli_channel",
            "weights": tuple(
                (pauli, float(weight))
                for pauli, weight in model.weights.items()
            ),
        }
    raise UnsupportedNativeCircuitError(
        f"unsupported native noise model {type(model).__name__}"
    )


def _payload_to_batch_trajectory(payload: Mapping[str, Any]) -> BatchTrajectory:
    return BatchTrajectory(
        shots=int(payload["shots"]),
        all_mask=int(payload["all_mask"]),
        x_frame=tuple(int(mask) for mask in payload["x_frame"]),
        z_frame=tuple(int(mask) for mask in payload["z_frame"]),
        measurements={
            str(key): int(value)
            for key, value in payload["measurements"].items()
        },
        detectors={int(key): int(value) for key, value in payload["detectors"].items()},
        observables={int(key): int(value) for key, value in payload["observables"].items()},
        noise_event_masks={
            str(key): int(value) for key, value in payload["noise_event_masks"].items()
        },
    )


__all__ = [
    "NativePackedSampler",
    "UnsupportedNativeCircuitError",
    "compile_native_sampler",
]

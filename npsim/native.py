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


@dataclass(frozen=True)
class NativeDemSampler:
    """Compiled DEM sampler wrapper with native/Python fallback parity."""

    dem: Any
    backend_name: str
    _engine: Any

    @property
    def is_native(self) -> bool:
        return self.backend_name == "native"

    def run_batch(
        self,
        *,
        shots: int,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if self.backend_name == "native":
            if rng is not None:
                raise ValueError("native DEM sampler accepts seed, not a Python rng")
            payload = self._engine.run_batch(int(shots), seed)
            return _payload_to_dem_batch(payload)
        if rng is None:
            rng = random.Random(seed)
        return self._engine.run_batch(shots=shots, rng=rng)

    def estimate_default(
        self,
        *,
        shots: int,
        seed: int | None = None,
        baseline: str | float = "mean",
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if self.backend_name == "native":
            native_baseline = _native_baseline_value(baseline)
            payload = self._engine.estimate_default(int(shots), seed, native_baseline)
            return _payload_to_dem_hotspot_result(self.dem, payload)
        estimate_python = getattr(self._engine, "_estimate_python", None)
        if estimate_python is not None:
            return estimate_python(shots=shots, seed=seed, baseline=baseline)
        return self._engine.estimate(shots=shots, seed=seed, baseline=baseline)


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


def generate_native_dem(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
    backend: str = "auto",
    strict: bool = False,
) -> Any:
    """Generate a detector error model through the optional native extension."""

    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be 'auto', 'native', or 'python'")
    detectors, observables = _coerce_dem_declarations(circuit, detectors, observables)
    if backend == "python":
        return _python_generate_dem(circuit, detectors, observables)

    try:
        spec = _serialize_circuit(circuit)
        native_mod = importlib.import_module("npsim._npsim_native")
        payload = native_mod.generate_dem(
            spec,
            [_serialize_dem_detector(detector) for detector in detectors],
            [_serialize_dem_observable(observable) for observable in observables],
        )
        return _payload_to_detector_error_model(
            circuit,
            detectors,
            observables,
            payload,
        )
    except Exception as exc:
        if backend == "native" or strict:
            raise UnsupportedNativeCircuitError(str(exc)) from exc
        return _python_generate_dem(circuit, detectors, observables)


def compile_native_dem_sampler(
    dem: Any,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> NativeDemSampler:
    """Compile a detector error model into a packed native DEM sampler."""

    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be 'auto', 'native', or 'python'")
    if backend == "python":
        return _python_dem_sampler(dem)

    try:
        spec = _serialize_dem(dem)
        native_mod = importlib.import_module("npsim._npsim_native")
        engine = native_mod.compile_dem_sampler(spec)
    except Exception as exc:
        if backend == "native" or strict:
            raise UnsupportedNativeCircuitError(str(exc)) from exc
        return _python_dem_sampler(dem)

    return NativeDemSampler(
        dem=dem,
        backend_name="native",
        _engine=engine,
    )


def _python_sampler(circuit: Circuit) -> NativePackedSampler:
    return NativePackedSampler(
        circuit=circuit,
        backend_name="python",
        _engine=BatchForwardNoiseAwareSimulator(circuit),
    )


def _python_dem_sampler(dem: Any) -> NativeDemSampler:
    from npsim.dem_sampler import DemBatchHotspotSimulator

    return NativeDemSampler(
        dem=dem,
        backend_name="python",
        _engine=DemBatchHotspotSimulator(dem),
    )


def _python_generate_dem(circuit: Circuit, detectors: Any, observables: Any) -> Any:
    from npsim.dem import DetectorErrorModelGenerator

    generator = DetectorErrorModelGenerator(
        circuit,
        detectors=detectors,
        observables=observables,
    )
    generate_python = getattr(generator, "_generate_python", None)
    if generate_python is not None:
        return generate_python()
    return generator.generate()


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


def _coerce_dem_declarations(
    circuit: Circuit,
    detectors: Any | None,
    observables: Any | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from npsim.dem import _detectors_from_circuit, _observables_from_circuit

    return (
        tuple(detectors) if detectors is not None else _detectors_from_circuit(circuit),
        tuple(observables) if observables is not None else _observables_from_circuit(circuit),
    )


def _serialize_dem_detector(detector: Any) -> dict[str, Any]:
    return {
        "id": int(detector.id),
        "measurement_keys": tuple(str(key) for key in detector.measurement_keys),
        "coords": tuple(float(coord) for coord in detector.coords),
    }


def _serialize_dem_observable(observable: Any) -> dict[str, Any]:
    return {
        "id": int(observable.id),
        "measurement_keys": tuple(str(key) for key in observable.measurement_keys),
        "pauli_qubits": tuple(int(qubit) for qubit in observable.pauli_qubits),
        "pauli": str(observable.pauli),
    }


def _serialize_dem(dem: Any) -> dict[str, Any]:
    return {
        "detectors": [_serialize_dem_detector(detector) for detector in dem.detectors],
        "observables": [
            _serialize_dem_observable(observable)
            for observable in dem.observables
        ],
        "edges": [
            {
                "probability": float(edge.probability),
                "detectors": tuple(int(detector_id) for detector_id in edge.detectors),
                "observables": tuple(
                    int(observable_id)
                    for observable_id in edge.observables
                ),
                "location_id": str(edge.location_id),
            }
            for edge in dem.edges
        ],
    }


def _payload_to_detector_error_model(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
    payload: Any,
) -> Any:
    from npsim.dem import DetectorErrorEdge, DetectorErrorModel

    locations = circuit.noise_locations()
    edges = []
    for item in payload:
        location_id = str(item["location_id"])
        location = locations.get(location_id)
        edges.append(
            DetectorErrorEdge(
                probability=float(item["probability"]),
                detectors=tuple(int(value) for value in item["detectors"]),
                observables=tuple(int(value) for value in item["observables"]),
                location_id=location_id,
                event=item["event"],
                tags=dict(location.tags) if location is not None else {},
            )
        )
    return DetectorErrorModel(
        detectors=tuple(detectors),
        observables=tuple(observables),
        edges=tuple(edges),
    )


def _payload_to_dem_batch(payload: Mapping[str, Any]) -> Any:
    from npsim.dem_sampler import DemBatchTrajectory

    return DemBatchTrajectory(
        shots=int(payload["shots"]),
        all_mask=int(payload["all_mask"]),
        detectors={int(key): int(value) for key, value in payload["detectors"].items()},
        observables={
            int(key): int(value)
            for key, value in payload["observables"].items()
        },
        edge_event_masks={
            int(key): int(value)
            for key, value in payload["edge_event_masks"].items()
        },
    )


def _payload_to_dem_hotspot_result(dem: Any, payload: Mapping[str, Any]) -> Any:
    from npsim.dem_sampler import (
        DemHotspotResult,
        DemLocationMetadata,
        _aggregate_by_tag,
        _aggregate_detector_hotspots,
        _edge_sensitivities_to_detector_graph,
    )

    edge_sensitivities = {
        int(key): float(value)
        for key, value in payload["edge_sensitivities"].items()
    }
    edge_hotspots = {
        edge_index: abs(sensitivity)
        for edge_index, sensitivity in edge_sensitivities.items()
    }
    sensitivities = {
        str(key): float(value)
        for key, value in payload["sensitivities"].items()
    }
    hotspots = {
        location_id: abs(sensitivity)
        for location_id, sensitivity in sensitivities.items()
    }
    locations = _dem_location_metadata(dem)
    return DemHotspotResult(
        dem=dem,
        shots=int(payload["shots"]),
        mean_loss=float(payload["mean_loss"]),
        baseline=float(payload["baseline"]),
        edge_sensitivities=edge_sensitivities,
        edge_hotspots=edge_hotspots,
        sensitivities=sensitivities,
        hotspots=hotspots,
        by_detector=_aggregate_detector_hotspots(dem, edge_hotspots),
        by_round=_aggregate_by_tag(hotspots, locations, "round"),
        by_gate=_aggregate_by_tag(hotspots, locations, "gate"),
        by_operation=_aggregate_by_tag(hotspots, locations, "operation"),
        locations=locations,
        detector_graph_hotspots=_edge_sensitivities_to_detector_graph(
            dem,
            edge_sensitivities,
        ),
    )


def _dem_location_metadata(dem: Any) -> dict[str, Any]:
    from npsim.dem_sampler import DemLocationMetadata

    out: dict[str, DemLocationMetadata] = {}
    for edge in dem.edges:
        if edge.location_id not in out:
            out[edge.location_id] = DemLocationMetadata(
                id=edge.location_id,
                tags=edge.tags,
            )
    return out


def _native_baseline_value(baseline: str | float) -> float | None:
    if baseline == "mean":
        return None
    if isinstance(baseline, (int, float)):
        return float(baseline)
    raise ValueError("baseline must be 'mean' or a numeric value")


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
    "NativeDemSampler",
    "NativePackedSampler",
    "UnsupportedNativeCircuitError",
    "compile_native_dem_sampler",
    "compile_native_sampler",
    "generate_native_dem",
]

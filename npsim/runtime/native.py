"""Native packed sampler API.

The Rust extension is the only runtime backend for packed batch sampling and
detector-error-model sampling.  ``backend="auto"`` is kept for API
compatibility, but it now requires the native extension to import and compile
successfully.
"""

from __future__ import annotations

import importlib
import inspect
import random
from dataclasses import dataclass
from typing import Any, Mapping

from npsim.runtime.batch import BatchTrajectory
from npsim.runtime.loss import logical_residual_loss_mask
from npsim.core import Circuit, NoiseLocation, Operation
from npsim.core import (
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

    ``backend_name`` is always ``"native"``.
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
        if rng is not None:
            raise ValueError("native sampler accepts seed, not a Python rng")
        payload = self._engine.sample(int(shots), seed)
        return _payload_to_batch_trajectory(payload)

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
        if rng is not None:
            raise ValueError("native sampler accepts seed, not a Python rng")
        if hasattr(self._engine, "sample_measurements"):
            payload = self._engine.sample_measurements(int(shots), seed)
            return {str(key): int(value) for key, value in payload.items()}
        return self.sample(shots=shots, seed=seed).measurements

    def estimate(
        self,
        *,
        shots: int,
        loss_mask_fn: Any | None = None,
        decoder: Any | None = None,
        correction_mask_fn: Any | None = None,
        seed: int | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if decoder is not None and correction_mask_fn is not None:
            raise ValueError("supply either decoder or correction_mask_fn, not both")
        native_baseline = _native_baseline_value(baseline)
        batch = self._engine.run_native_batch(int(shots), seed)
        corrections = _forward_correction_masks(
            batch,
            decoder,
            correction_mask_fn,
        )
        if loss_mask_fn is None:
            loss_mask = _forward_default_loss_mask(batch, corrections)
        else:
            loss_mask = _call_forward_loss_mask_fn(loss_mask_fn, batch, corrections)
        loss_mask &= int(batch.all_mask)
        payload = self._engine.estimate_hotspots(
            batch,
            loss_mask,
            native_baseline,
            int(top_k),
        )
        return _payload_to_simulation_result(self.circuit, payload)


@dataclass(frozen=True)
class NativeDemSampler:
    """Compiled native DEM sampler wrapper."""

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
        return_edge_events: bool = True,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if rng is not None:
            raise ValueError("native DEM sampler accepts seed, not a Python rng")
        payload = self._engine.run_batch(
            int(shots),
            seed,
            bool(return_edge_events),
        )
        return _payload_to_dem_batch(payload)

    def estimate_default(
        self,
        *,
        shots: int,
        seed: int | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        native_baseline = _native_baseline_value(baseline)
        payload = self._engine.estimate_default(
            int(shots),
            seed,
            native_baseline,
            int(top_k),
        )
        return _payload_to_dem_hotspot_result(self.dem, payload)

    def estimate(
        self,
        *,
        shots: int,
        seed: int | None = None,
        decoder: Any | None = None,
        correction_mask_fn: Any | None = None,
        loss_mask_fn: Any | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if decoder is not None and correction_mask_fn is not None:
            raise ValueError("supply either decoder or correction_mask_fn, not both")
        native_baseline = _native_baseline_value(baseline)
        if decoder is None and correction_mask_fn is None and loss_mask_fn is None:
            payload = self._engine.estimate_default(
                int(shots),
                seed,
                native_baseline,
                int(top_k),
            )
            return _payload_to_dem_hotspot_result(self.dem, payload)

        batch = self._engine.run_native_batch(int(shots), seed)
        if correction_mask_fn is not None:
            corrections = dict(correction_mask_fn(batch))
        elif decoder is not None:
            if not hasattr(decoder, "decode_batch_masks"):
                raise TypeError("DEM decoder must provide decode_batch_masks(batch)")
            corrections = dict(decoder.decode_batch_masks(batch))
        else:
            corrections = {}
        if loss_mask_fn is None:
            loss_mask = logical_residual_loss_mask(
                batch.observables,
                corrections,
                observable_ids=(observable.id for observable in self.dem.observables),
                all_mask=batch.all_mask,
            )
        else:
            loss_mask = loss_mask_fn(batch, corrections)
        loss_mask = int(loss_mask) & int(batch.all_mask)
        payload = self._engine.estimate_hotspots(
            batch,
            loss_mask,
            native_baseline,
            int(top_k),
        )
        return _payload_to_dem_hotspot_result(self.dem, payload)


def compile_native_sampler(
    circuit: Circuit,
    *,
    observables: Any | None = None,
    backend: str = "auto",
    strict: bool = False,
) -> NativePackedSampler:
    """Compile ``circuit`` into a packed sampler.

    Parameters
    ----------
    backend:
        ``"auto"`` and ``"native"`` both require the Rust extension.
        ``"python"`` is no longer supported.
    strict:
        Kept for compatibility. Native compile/import failures are always
        surfaced.
    """

    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be 'auto', 'native', or 'python'")
    if backend == "python":
        raise UnsupportedNativeCircuitError("Python backend is no longer supported")

    try:
        spec = _serialize_circuit(
            circuit,
            observables=tuple(observables) if observables is not None else (),
        )
        native_mod = importlib.import_module("npsim._npsim_native")
        engine = native_mod.compile_sampler(spec)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc

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
    """Generate a detector error model through the native extension."""

    if backend not in {"auto", "native", "python"}:
        raise ValueError("backend must be 'auto', 'native', or 'python'")
    detectors, observables = _coerce_dem_declarations(circuit, detectors, observables)
    if backend == "python":
        raise UnsupportedNativeCircuitError("Python backend is no longer supported")

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
        raise UnsupportedNativeCircuitError(str(exc)) from exc


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
        raise UnsupportedNativeCircuitError("Python backend is no longer supported")

    try:
        spec = _serialize_dem(dem)
        native_mod = importlib.import_module("npsim._npsim_native")
        engine = native_mod.compile_dem_sampler(spec)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc

    return NativeDemSampler(
        dem=dem,
        backend_name="native",
        _engine=engine,
    )


def _serialize_circuit(
    circuit: Circuit,
    *,
    observables: tuple[Any, ...] = (),
) -> dict[str, Any]:
    return {
        "n_qubits": int(circuit.n_qubits),
        "operations": [
            _serialize_operation(operation)
            for operation in circuit.operations
        ],
        "observables": [
            _serialize_dem_observable(observable)
            for observable in observables
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
    from npsim.dem.model import _detectors_from_circuit, _observables_from_circuit

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
                "tags": dict(edge.tags),
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
    from npsim.dem.model import DetectorErrorEdge, DetectorErrorModel

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
    from npsim.dem.sampler import DemBatchTrajectory

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
    from npsim.dem.sampler import (
        DemEdgeHotspotRow,
        DemHotspotResult,
        DemLocationHotspotRow,
        DemLocationMetadata,
    )

    edge_sensitivities = _required_payload_field(payload, "edge_sensitivities")
    edge_hotspots = _required_payload_field(payload, "edge_hotspots")
    sensitivities = _required_payload_field(payload, "sensitivities")
    hotspots = _required_payload_field(payload, "hotspots")
    locations = _dem_location_metadata(dem)
    by_detector = _required_payload_field(payload, "by_detector")
    detector_graph_hotspots = _payload_to_detector_graph_hotspots(
        dem,
        _required_payload_field(payload, "detector_graph_hotspots"),
    )
    top_edges_cache = tuple(
        DemEdgeHotspotRow(
            edge_index=int(row["edge_index"]),
            location_id=dem.edges[int(row["edge_index"])].location_id,
            event=dem.edges[int(row["edge_index"])].event,
            probability=dem.edges[int(row["edge_index"])].probability,
            detectors=dem.edges[int(row["edge_index"])].detectors,
            observables=dem.edges[int(row["edge_index"])].observables,
            sensitivity=float(row["sensitivity"]),
            hotspot=float(row["hotspot"]),
        )
        for row in _required_payload_field(payload, "top_edges")
    )
    top_hotspots_cache = tuple(
        DemLocationHotspotRow(
            location_id=str(row["location_id"]),
            sensitivity=float(row["sensitivity"]),
            hotspot=float(row["hotspot"]),
            qubits=locations[str(row["location_id"])].qubits,
            tags=locations[str(row["location_id"])].tags,
        )
        for row in _required_payload_field(payload, "top_hotspots")
    )
    return DemHotspotResult(
        dem=dem,
        shots=int(payload["shots"]),
        mean_loss=float(payload["mean_loss"]),
        baseline=float(payload["baseline"]),
        edge_sensitivities=edge_sensitivities,
        edge_hotspots=edge_hotspots,
        sensitivities=sensitivities,
        hotspots=hotspots,
        by_detector=by_detector,
        by_round=_required_payload_field(payload, "by_round"),
        by_gate=_required_payload_field(payload, "by_gate"),
        by_operation=_required_payload_field(payload, "by_operation"),
        locations=locations,
        detector_graph_hotspots=detector_graph_hotspots,
        top_edges_cache=top_edges_cache,
        top_hotspots_cache=top_hotspots_cache,
    )


def _required_payload_field(payload: Mapping[str, Any], field: str) -> Any:
    try:
        return payload[field]
    except KeyError as exc:
        raise UnsupportedNativeCircuitError(
            f"native payload is missing {field}"
        ) from exc


def _payload_to_detector_graph_hotspots(dem: Any, payload: Mapping[str, Any]) -> Any:
    from npsim.dem.model import DetectorGraphEdgeHotspot, DetectorGraphHotspots

    edge_hotspots = []
    for row in payload["edge_hotspots"]:
        edge_index = int(row["edge_index"])
        edge = dem.edges[edge_index]
        edge_hotspots.append(
            DetectorGraphEdgeHotspot(
                edge_index=edge_index,
                location_id=edge.location_id,
                event=edge.event,
                probability=edge.probability,
                detectors=edge.detectors,
                observables=edge.observables,
                sensitivity=float(row["sensitivity"]),
                hotspot=float(row["hotspot"]),
                weight=1.0,
            )
        )

    return DetectorGraphHotspots(
        edge_hotspots=tuple(edge_hotspots),
        by_detector_edge=_graph_key_rows_to_dict(payload["by_detector_edge"]),
        signed_by_detector_edge=_graph_key_rows_to_dict(payload["signed_by_detector_edge"]),
        by_detector={int(key): float(value) for key, value in payload["by_detector"].items()},
        signed_by_detector={
            int(key): float(value)
            for key, value in payload["signed_by_detector"].items()
        },
        by_observable={
            int(key): float(value)
            for key, value in payload["by_observable"].items()
        },
        signed_by_observable={
            int(key): float(value)
            for key, value in payload["signed_by_observable"].items()
        },
        by_location={
            str(key): float(value)
            for key, value in payload["by_location"].items()
        },
        signed_by_location={
            str(key): float(value)
            for key, value in payload["signed_by_location"].items()
        },
    )


def _graph_key_rows_to_dict(rows: Any) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    if hasattr(rows, "items"):
        return rows
    return {
        (
            tuple(int(value) for value in row["detectors"]),
            tuple(int(value) for value in row["observables"]),
        ): float(row["value"])
        for row in rows
    }


def _payload_to_simulation_result(circuit: Circuit, payload: Mapping[str, Any]) -> Any:
    from npsim.runtime.results import HotspotRow, SimulationResult

    locations = circuit.noise_locations()
    sensitivities = payload["sensitivities"]
    hotspots = payload["hotspots"]
    top_hotspots_cache = tuple(
        HotspotRow(
            location_id=str(row["location_id"]),
            sensitivity=float(row["sensitivity"]),
            hotspot=float(row["hotspot"]),
            qubits=locations[str(row["location_id"])].qubits,
            tags=locations[str(row["location_id"])].tags,
        )
        for row in _required_payload_field(payload, "top_hotspots")
    )
    return SimulationResult(
        shots=int(payload["shots"]),
        mean_loss=float(payload["mean_loss"]),
        baseline=float(payload["baseline"]),
        sensitivities=sensitivities,
        hotspots=hotspots,
        by_qubit=payload["by_qubit"],
        by_round=payload["by_round"],
        by_gate=payload["by_gate"],
        by_operation=payload["by_operation"],
        locations=locations,
        losses=[],
        top_hotspots_cache=top_hotspots_cache,
    )


def _dem_location_metadata(dem: Any) -> dict[str, Any]:
    from npsim.dem.sampler import DemLocationMetadata

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


def _forward_correction_masks(
    batch: Any,
    decoder: Any | None,
    correction_mask_fn: Any | None,
) -> Mapping[int, int]:
    if correction_mask_fn is not None:
        return dict(correction_mask_fn(batch))
    if decoder is None:
        return {}
    if not hasattr(decoder, "decode_batch_masks"):
        raise TypeError("batch decoder must provide decode_batch_masks(batch)")
    return dict(decoder.decode_batch_masks(batch))


def _forward_default_loss_mask(
    batch: Any,
    corrections: Mapping[Any, int],
) -> int:
    return logical_residual_loss_mask(
        batch.observables,
        corrections,
        all_mask=batch.all_mask,
    )


def _call_forward_loss_mask_fn(
    loss_mask_fn: Any,
    batch: Any,
    corrections: Mapping[Any, int],
) -> int:
    if _accepts_positional_args(loss_mask_fn, 2):
        return int(loss_mask_fn(batch, corrections))
    return int(loss_mask_fn(batch))


def _accepts_positional_args(fn: Any, count: int) -> bool:
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

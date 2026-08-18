"""Canonical collection task identity payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any


SAMPLING_ID_SCHEMA_VERSION = 2
STRONG_ID_SCHEMA_VERSION = 4
SOURCE_DIGEST_SCHEMA_VERSION = 1
DECODER_DIGEST_SCHEMA_VERSION = 1


def canonical_json(value: object) -> str:
    """Return the stable JSON encoding used by collection identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def domain_digest(
    *,
    schema: str,
    schema_version: int,
    payload: object,
) -> str:
    """Hash a canonical payload under an explicit identity domain."""

    envelope = {
        "schema": schema,
        "schema_version": schema_version,
        "payload": payload,
    }
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


def source_identity_digest(payload: Mapping[str, object]) -> str:
    """Return the domain-separated digest for a complete source payload."""

    return domain_digest(
        schema="faultscope.collection.source_digest",
        schema_version=SOURCE_DIGEST_SCHEMA_VERSION,
        payload=dict(payload),
    )


def decoder_identity_digest(payload: Mapping[str, object]) -> str:
    """Return the domain-separated digest for a normalized decoder payload."""

    return domain_digest(
        schema="faultscope.collection.decoder_digest",
        schema_version=DECODER_DIGEST_SCHEMA_VERSION,
        payload=dict(payload),
    )


def canonical_mapping_payload(value: object, *, context: str) -> dict[str, object]:
    """Validate and normalize a JSON-compatible identity mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must return a mapping")
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must return a JSON-serializable mapping") from exc
    if not isinstance(normalized, dict):  # Defensive: mappings encode as JSON objects.
        raise TypeError(f"{context} must return a mapping")
    return normalized


def decoder_identity_payload(
    decoder: object | None,
    *,
    decoder_name: str | None,
) -> dict[str, object]:
    """Return a stable payload for a resolved collection decoder."""

    if decoder is None:
        return {
            "kind": "none",
            "identity_version": 1,
        }

    provider = getattr(decoder, "strong_id_payload", None)
    if not callable(provider):
        decoder_type = _qualified_type_name(decoder)
        raise TypeError(
            "collection decoder objects must be native decoders and provide "
            f"strong_id_payload(); {decoder_type} does not"
        )
    payload = canonical_mapping_payload(
        provider(),
        context=f"{_qualified_type_name(decoder)}.strong_id_payload()",
    )
    context = f"{_qualified_type_name(decoder)}.strong_id_payload()"
    _validate_decoder_payload_fields(payload, context=context)
    return {
        "kind": "decoder",
        "type": _qualified_type_name(decoder),
        "name": decoder_name,
        "detector_ids": _optional_int_ids(decoder, "detector_ids"),
        "observable_ids": _optional_int_ids(decoder, "observable_ids"),
        "payload": payload,
    }


def _validate_decoder_payload_fields(payload: dict[str, object], *, context: str) -> None:
    required_fields = {"backend", "decoder", "implementation_version", "parameters"}
    missing_fields = sorted(required_fields - payload.keys())
    if missing_fields:
        names = ", ".join(missing_fields)
        raise TypeError(f"{context} is missing required field(s): {names}")
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise TypeError(f"{context} parameters must be a mapping")
    children = parameters.get("children")
    if children is None:
        return
    if not isinstance(children, list):
        raise TypeError(f"{context} children must be a sequence of decoder payloads")
    for index, child in enumerate(children):
        if not isinstance(child, dict):
            raise TypeError(f"{context} child {index} must be a decoder payload mapping")
        _validate_decoder_payload_fields(child, context=f"{context} child {index}")


def source_identity_payload(
    *,
    circuit: object | None,
    dem: object,
) -> dict[str, object]:
    """Return the canonical source payload for a collection task."""

    if circuit is None:
        return {
            "kind": "dem",
            "dem": dem_identity_payload(dem),
        }
    return {
        "kind": "circuit",
        "circuit": circuit_identity_payload(circuit),
        "dem": dem_identity_payload(dem),
    }


def forward_source_identity_payload(
    *,
    circuit: object,
    detectors: Iterable[object] | None,
    observables: Iterable[object] | None,
) -> dict[str, object]:
    """Return the source payload for direct forward-circuit collection.

    Declaration overrides deliberately preserve the difference between ``None``
    (use declarations embedded in the circuit) and an explicit empty sequence.
    """

    return {
        "kind": "forward_circuit",
        "circuit": circuit_identity_payload(circuit),
        "detectors": (
            None
            if detectors is None
            else [detector_identity_payload(item) for item in tuple(detectors)]
        ),
        "observables": (
            None
            if observables is None
            else [observable_identity_payload(item) for item in tuple(observables)]
        ),
    }


def detector_identity_payload(detector: object) -> dict[str, object]:
    return {
        "type": _qualified_type_name(detector),
        "id": int(getattr(detector, "id")),
        "measurement_keys": [str(value) for value in tuple(getattr(detector, "measurement_keys"))],
        "coords": [float(value) for value in tuple(getattr(detector, "coords"))],
    }


def observable_identity_payload(observable: object) -> dict[str, object]:
    return {
        "type": _qualified_type_name(observable),
        "id": int(getattr(observable, "id")),
        "measurement_keys": [
            str(value) for value in tuple(getattr(observable, "measurement_keys"))
        ],
        "pauli_qubits": [int(value) for value in tuple(getattr(observable, "pauli_qubits"))],
        "pauli": str(getattr(observable, "pauli")),
    }


def circuit_identity_payload(circuit: object) -> dict[str, object]:
    """Serialize every public behavioral field of a native circuit."""

    return {
        "type": _qualified_type_name(circuit),
        "n_qubits": int(getattr(circuit, "n_qubits")),
        "operations": [
            operation_identity_payload(operation)
            for operation in tuple(getattr(circuit, "operations"))
        ],
    }


def operation_identity_payload(operation: object) -> dict[str, object]:
    """Serialize an operation, including structured repeat/record fields."""

    noise_location = getattr(operation, "noise_location")
    return {
        "type": _qualified_type_name(operation),
        "kind": str(getattr(operation, "kind")),
        "qubits": [int(value) for value in tuple(getattr(operation, "qubits"))],
        "key": getattr(operation, "key"),
        "basis": str(getattr(operation, "basis")),
        "pauli": getattr(operation, "pauli"),
        "measurement_keys": [str(value) for value in tuple(getattr(operation, "measurement_keys"))],
        "observable_id": getattr(operation, "observable_id"),
        "noise_location": (
            None if noise_location is None else noise_location_identity_payload(noise_location)
        ),
        "metadata": dict(getattr(operation, "metadata")),
        "repeat_count": getattr(operation, "repeat_count"),
        "body": [operation_identity_payload(child) for child in tuple(getattr(operation, "body"))],
        "record_lookbacks": [int(value) for value in tuple(getattr(operation, "record_lookbacks"))],
    }


def noise_location_identity_payload(location: object) -> dict[str, object]:
    """Serialize a circuit noise location and its concrete noise model."""

    return {
        "type": _qualified_type_name(location),
        "id": str(getattr(location, "id")),
        "model": noise_model_identity_payload(getattr(location, "model")),
        "rate": float(getattr(location, "rate")),
        "qubits": [int(value) for value in tuple(getattr(location, "qubits"))],
        "tags": dict(getattr(location, "tags")),
    }


def noise_model_identity_payload(model: object) -> dict[str, object]:
    """Serialize the supported native stochastic noise models."""

    type_name = type(model).__name__
    payload: dict[str, object] = {
        "type": _qualified_type_name(model),
        "identity_version": 1,
    }
    if type_name == "BernoulliPauliNoise":
        payload["pauli"] = str(getattr(model, "pauli"))
    elif type_name == "PauliChannel":
        payload["weights"] = {
            str(key): float(value) for key, value in dict(getattr(model, "weights")).items()
        }
    elif type_name == "TwoQubitDepolarizing":
        payload["events"] = [str(value) for value in tuple(getattr(model, "_events"))]
    elif type_name not in {"MeasurementBitFlip", "SingleQubitDepolarizing"}:
        raise TypeError(
            "collection circuit identity does not support noise model "
            f"{_qualified_type_name(model)}"
        )
    return payload


def dem_identity_payload(dem: object) -> dict[str, object]:
    """Serialize every behavioral and provenance field of a native DEM."""

    return {
        "type": _qualified_type_name(dem),
        "detectors": [
            {
                "id": int(getattr(detector, "id")),
                "measurement_keys": [
                    str(value) for value in tuple(getattr(detector, "measurement_keys"))
                ],
                "coords": [float(value) for value in tuple(getattr(detector, "coords"))],
            }
            for detector in tuple(getattr(dem, "detectors"))
        ],
        "observables": [
            {
                "id": int(getattr(observable, "id")),
                "measurement_keys": [
                    str(value) for value in tuple(getattr(observable, "measurement_keys"))
                ],
                "pauli_qubits": [
                    int(value) for value in tuple(getattr(observable, "pauli_qubits"))
                ],
                "pauli": str(getattr(observable, "pauli")),
            }
            for observable in tuple(getattr(dem, "observables"))
        ],
        "edges": [
            {
                "probability": float(getattr(edge, "probability")),
                "detectors": [int(value) for value in tuple(getattr(edge, "detectors"))],
                "observables": [int(value) for value in tuple(getattr(edge, "observables"))],
                "location_id": str(getattr(edge, "location_id")),
                "event": getattr(edge, "event"),
                "tags": dict(getattr(edge, "tags")),
            }
            for edge in tuple(getattr(dem, "edges"))
        ],
    }


def _optional_int_ids(value: object, attribute: str) -> list[int] | None:
    try:
        ids: Any = getattr(value, attribute)
    except AttributeError:
        return None
    return [int(item) for item in tuple(ids)]


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"

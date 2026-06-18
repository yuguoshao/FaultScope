"""Shared Stim comparison helpers for tests and benchmarks."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from npsim.core import (
    BernoulliPauliNoise,
    Circuit,
    MeasurementBitFlip,
    NoiseLocation,
    Operation,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from npsim.dem import Detector, LogicalObservable
from npsim.runtime import BatchTrajectory

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import stim
except ImportError as exc:  # pragma: no cover - dependency guard
    stim = None
    _STIM_IMPORT_ERROR = exc
else:
    _STIM_IMPORT_ERROR = None


def require_numpy() -> Any:
    if np is None:
        raise ImportError("NumPy is required for Stim comparisons") from _NUMPY_IMPORT_ERROR
    return np


def require_stim() -> Any:
    if stim is None:
        raise ImportError("Stim is required for Stim comparisons") from _STIM_IMPORT_ERROR
    return stim


def to_stim_circuit(circuit: Circuit) -> tuple[Any, tuple[str, ...]]:
    stim_module = require_stim()
    out = stim_module.Circuit()
    measurement_keys: list[str] = []
    measurement_index_by_key: dict[str, int] = {}

    for operation in circuit.operations:
        kind = operation.kind
        if kind in {"h", "s", "s_dag"}:
            gate = {"h": "H", "s": "S", "s_dag": "S_DAG"}[kind]
            out.append(gate, operation.qubits)
        elif kind in {"cx", "cz", "swap"}:
            gate = {"cx": "CX", "cz": "CZ", "swap": "SWAP"}[kind]
            out.append(gate, operation.qubits)
        elif kind == "pauli":
            if operation.pauli is None:
                raise ValueError("pauli operation requires a Pauli string")
            _append_pauli_gate(out, operation.qubits, operation.pauli)
        elif kind == "noise":
            if operation.noise_location is None:
                raise ValueError("noise operation requires a noise location")
            append_noise(out, operation.noise_location)
        elif kind == "measure":
            basis = operation.basis.upper()
            gate = {"Z": "M", "X": "MX", "Y": "MY"}[basis]
            append_measurement_gate(out, gate, operation.qubits, operation.noise_location)
            _record_measurement_key(
                operation.key,
                measurement_keys,
                measurement_index_by_key,
            )
        elif kind == "measure_pauli":
            if operation.pauli is None:
                raise ValueError("measure_pauli operation requires a Pauli string")
            append_measurement_gate(
                out,
                "MPP",
                mpp_targets(operation.qubits, operation.pauli),
                operation.noise_location,
            )
            _record_measurement_key(
                operation.key,
                measurement_keys,
                measurement_index_by_key,
            )
        elif kind == "reset":
            (qubit,) = operation.qubits
            basis = operation.basis.upper()
            if operation.key is None:
                out.append({"Z": "R", "X": "RX", "Y": "RY"}[basis], [qubit])
            else:
                out.append({"Z": "MR", "X": "MRX", "Y": "MRY"}[basis], [qubit])
                _record_measurement_key(
                    operation.key,
                    measurement_keys,
                    measurement_index_by_key,
                )
        elif kind == "detector":
            out.append(
                "DETECTOR",
                rec_targets(operation.measurement_keys, measurement_index_by_key),
                operation.metadata.get("coords", ()),
            )
        elif kind == "observable_include":
            if operation.observable_id is None:
                raise ValueError("observable_include requires observable_id")
            out.append(
                "OBSERVABLE_INCLUDE",
                rec_targets(operation.measurement_keys, measurement_index_by_key),
                operation.observable_id,
            )
        else:
            raise ValueError(f"unsupported operation kind {kind!r}")

    return out, tuple(measurement_keys)


def append_measurement_gate(
    circuit: Any,
    gate: str,
    targets: Sequence[Any] | Sequence[int],
    location: NoiseLocation | None,
) -> None:
    if location is None:
        circuit.append(gate, targets)
        return
    if not isinstance(location.model, MeasurementBitFlip):
        raise ValueError("Stim comparison only supports MeasurementBitFlip on measurements")
    circuit.append(gate, targets, location.rate)


def append_noise(circuit: Any, location: NoiseLocation) -> None:
    model = location.model
    if isinstance(model, BernoulliPauliNoise):
        if len(model.pauli) == 1:
            gate = {"X": "X_ERROR", "Y": "Y_ERROR", "Z": "Z_ERROR", "I": None}[
                model.pauli
            ]
            if gate is not None:
                circuit.append(gate, location.qubits, location.rate)
            return
        circuit.append("E", correlated_error_targets(location.qubits, model.pauli), location.rate)
        return
    if isinstance(model, SingleQubitDepolarizing):
        circuit.append("DEPOLARIZE1", location.qubits, location.rate)
        return
    if isinstance(model, TwoQubitDepolarizing):
        circuit.append("DEPOLARIZE2", location.qubits, location.rate)
        return
    if isinstance(model, PauliChannel):
        probabilities = pauli_channel_probabilities(model, location.rate)
        if model.event_length == 1:
            circuit.append(
                "PAULI_CHANNEL_1",
                location.qubits,
                [probabilities.get(pauli, 0.0) for pauli in ("X", "Y", "Z")],
            )
            return
        if model.event_length == 2:
            events = (
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
            circuit.append(
                "PAULI_CHANNEL_2",
                location.qubits,
                [probabilities.get(event, 0.0) for event in events],
            )
            return
    raise ValueError(f"unsupported Stim comparison noise model {type(model).__name__}")


def measurement_batch_from_stim_samples(
    samples: Any,
    key_order: Sequence[str],
) -> BatchTrajectory:
    shots = int(samples.shape[0])
    return BatchTrajectory(
        shots=shots,
        all_mask=(1 << shots) - 1,
        x_frame=(),
        z_frame=(),
        measurements={
            key: stim_column_mask(samples, column)
            for column, key in enumerate(key_order)
        },
        detectors={},
        observables={},
        noise_event_masks={},
    )


def dem_batch_from_stim_samples(
    stim_detectors: Any,
    stim_observables: Any,
    *,
    detectors: Sequence[Detector],
    observables: Sequence[LogicalObservable],
) -> BatchTrajectory:
    shots = int(stim_detectors.shape[0])
    return BatchTrajectory(
        shots=shots,
        all_mask=(1 << shots) - 1,
        x_frame=(),
        z_frame=(),
        measurements={},
        detectors={
            int(detector.id): stim_column_mask(stim_detectors, column)
            for column, detector in enumerate(detectors)
        },
        observables=stim_observable_masks(stim_observables, observables),
        noise_event_masks={},
    )


def stim_observable_masks(
    stim_observables: Any,
    observables: Sequence[LogicalObservable],
) -> dict[int, int]:
    return {
        int(observable.id): stim_column_mask(stim_observables, int(observable.id))
        for observable in observables
    }


def stim_column_mask(samples: Any, column: int) -> int:
    np_module = require_numpy()
    mask = 0
    for shot in np_module.flatnonzero(samples[:, column]):
        mask |= 1 << int(shot)
    return mask


def masks_to_dense_array(
    masks: Mapping[int, int],
    ids: Sequence[int],
    shots: int,
) -> Any:
    np_module = require_numpy()
    out = np_module.zeros((shots, len(ids)), dtype=np_module.uint8)
    if shots == 0 or not ids:
        return out
    byte_count = (shots + 7) // 8
    all_mask = (1 << shots) - 1
    for col, item_id in enumerate(ids):
        mask = int(masks.get(int(item_id), 0)) & all_mask
        out[:, col] = np_module.unpackbits(
            np_module.frombuffer(mask.to_bytes(byte_count, "little"), dtype=np_module.uint8),
            bitorder="little",
        )[:shots]
    return out


def dense_predictions_to_masks(
    predictions: Any,
    observable_ids: Sequence[int],
    shots: int,
) -> dict[int, int]:
    np_module = require_numpy()
    observable_ids = tuple(int(observable_id) for observable_id in observable_ids)
    predictions = np_module.asarray(predictions, dtype=np_module.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape((shots, 1))
    if predictions.ndim != 2 or predictions.shape[0] != shots:
        raise ValueError(f"unexpected PyMatching prediction shape {predictions.shape}")
    if predictions.shape[1] != len(observable_ids):
        raise ValueError("PyMatching prediction length does not match observable count")
    return {
        observable_id: int.from_bytes(
            np_module.packbits(
                predictions[:, col].astype(np_module.uint8),
                bitorder="little",
            ).tobytes(),
            "little",
        )
        for col, observable_id in enumerate(observable_ids)
    }


def npsim_dem_error_edges(dem: Any) -> tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                round(float(edge.probability), 12),
                tuple(int(detector_id) for detector_id in edge.detectors),
                tuple(int(observable_id) for observable_id in edge.observables),
            )
            for edge in dem.edges
            if round(float(edge.probability), 12) > 0.0
        )
    )


def stim_dem_error_edges(stim_dem: Any) -> tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]:
    edges: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    detector_offset = 0
    for instruction in stim_dem:
        if instruction.type == "shift_detectors":
            detector_offset += sum(int(target) for target in instruction.targets_copy())
            continue
        if instruction.type != "error":
            continue
        detectors: list[int] = []
        observables: list[int] = []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(detector_offset + int(target.val))
            elif target.is_logical_observable_id():
                observables.append(int(target.val))
            elif not target.is_separator():
                raise ValueError(f"unsupported Stim DEM target {target!r}")
        edges.append(
            (
                round(float(instruction.args_copy()[0]), 12),
                tuple(detectors),
                tuple(observables),
            )
        )
    return tuple(sorted(edges))


def with_dem_declarations(
    circuit: Circuit,
    *,
    detectors: Sequence[Detector],
    observables: Sequence[LogicalObservable],
) -> Circuit:
    operations = list(circuit.operations)
    for detector in detectors:
        operations.append(
            Operation.detector(
                detector.measurement_keys,
                detector_id=detector.id,
                coords=detector.coords,
            )
        )
    for observable in observables:
        if observable.pauli:
            raise ValueError("Stim comparison needs measurement-only observables")
        operations.append(
            Operation.observable_include(
                observable.id,
                observable.measurement_keys,
            )
        )
    return Circuit(n_qubits=circuit.n_qubits, operations=operations)


def final_data_measurement_circuit(
    circuit: Circuit,
    qubits: Iterable[int],
    *,
    basis: str,
    prefix: str,
) -> Circuit:
    operations = list(circuit.operations)
    for idx, qubit in enumerate(qubits):
        operations.append(Operation.measure(int(qubit), key=f"{prefix}_{idx}", basis=basis))
    return Circuit(n_qubits=circuit.n_qubits, operations=operations)


def _append_pauli_gate(circuit: Any, qubits: Sequence[int], pauli: str) -> None:
    by_gate: dict[str, list[int]] = {"X": [], "Y": [], "Z": []}
    for qubit, local_pauli in zip(qubits, pauli):
        if local_pauli in by_gate:
            by_gate[local_pauli].append(qubit)
        elif local_pauli != "I":
            raise ValueError(f"unsupported Pauli {local_pauli!r}")
    for gate, targets in by_gate.items():
        if targets:
            circuit.append(gate, targets)


def mpp_targets(qubits: Sequence[int], pauli: str) -> list[Any]:
    stim_module = require_stim()
    factors = pauli_targets(qubits, pauli)
    if not factors:
        raise ValueError("Stim MPP comparison does not support empty Pauli products")
    targets: list[Any] = []
    for idx, target in enumerate(factors):
        if idx:
            targets.append(stim_module.target_combiner())
        targets.append(target)
    return targets


def correlated_error_targets(qubits: Sequence[int], pauli: str) -> list[Any]:
    return pauli_targets(qubits, pauli)


def pauli_targets(qubits: Sequence[int], pauli: str) -> list[Any]:
    stim_module = require_stim()
    targets: list[Any] = []
    for qubit, local_pauli in zip(qubits, pauli):
        if local_pauli == "I":
            continue
        if local_pauli == "X":
            targets.append(stim_module.target_x(qubit))
        elif local_pauli == "Y":
            targets.append(stim_module.target_y(qubit))
        elif local_pauli == "Z":
            targets.append(stim_module.target_z(qubit))
        else:
            raise ValueError(f"unsupported Pauli {local_pauli!r}")
    return targets


def pauli_channel_probabilities(model: PauliChannel, rate: float) -> dict[str, float]:
    total_weight = model.total_weight
    return {
        event: rate * weight / total_weight
        for event, weight in model.weights.items()
        if weight > 0
    }


def rec_targets(
    keys: Sequence[str],
    measurement_index_by_key: dict[str, int],
) -> list[Any]:
    stim_module = require_stim()
    current_index = len(measurement_index_by_key)
    return [
        stim_module.target_rec(measurement_index_by_key[key] - current_index)
        for key in keys
    ]


def _record_measurement_key(
    key: str | None,
    measurement_keys: list[str],
    measurement_index_by_key: dict[str, int],
) -> None:
    if key is None:
        key = f"m{len(measurement_keys)}"
    if key in measurement_index_by_key:
        raise ValueError(f"duplicate measurement key {key!r}")
    measurement_index_by_key[key] = len(measurement_keys)
    measurement_keys.append(key)

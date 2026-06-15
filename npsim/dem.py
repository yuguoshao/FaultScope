"""Detector error model generation by single-error propagation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.noise import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from npsim.pauli import PauliFrame, sparse_pauli_to_xz
from npsim.stabilizer import StabilizerState


class UnsupportedDemCircuitError(ValueError):
    """Raised when a circuit cannot be converted by single-error propagation."""


@dataclass(frozen=True)
class Detector:
    """A detector parity over measurement record keys."""

    id: int
    measurement_keys: tuple[str, ...]
    coords: tuple[float, ...] = ()


@dataclass(frozen=True)
class LogicalObservable:
    """A logical parity over measurements and/or the final Pauli frame."""

    id: int
    measurement_keys: tuple[str, ...] = ()
    pauli_qubits: tuple[int, ...] = ()
    pauli: str = ""

    def __post_init__(self) -> None:
        if bool(self.pauli_qubits) != bool(self.pauli):
            raise ValueError("pauli_qubits and pauli must be supplied together")
        if self.pauli and len(self.pauli_qubits) != len(self.pauli):
            raise ValueError("pauli_qubits and pauli must have the same length")


@dataclass(frozen=True)
class DetectorErrorEdge:
    probability: float
    detectors: tuple[int, ...]
    observables: tuple[int, ...]
    location_id: str
    event: object
    tags: Mapping[str, Any] = field(default_factory=dict)

    def to_dem_line(self) -> str:
        targets = [f"D{detector_id}" for detector_id in self.detectors]
        targets.extend(f"L{observable_id}" for observable_id in self.observables)
        joined_targets = " ".join(targets)
        if joined_targets:
            return f"error({self.probability:.17g}) {joined_targets}"
        return f"error({self.probability:.17g})"


@dataclass(frozen=True)
class DetectorGraphEdgeHotspot:
    """Hotspot assigned to one DEM edge."""

    edge_index: int
    location_id: str
    event: object
    probability: float
    detectors: tuple[int, ...]
    observables: tuple[int, ...]
    sensitivity: float
    hotspot: float
    weight: float


@dataclass(frozen=True)
class DetectorGraphHotspots:
    """Hotspots projected from noise locations onto the detector graph."""

    edge_hotspots: tuple[DetectorGraphEdgeHotspot, ...]
    by_detector_edge: Mapping[tuple[tuple[int, ...], tuple[int, ...]], float]
    signed_by_detector_edge: Mapping[tuple[tuple[int, ...], tuple[int, ...]], float]
    by_detector: Mapping[int, float]
    signed_by_detector: Mapping[int, float]
    by_observable: Mapping[int, float]
    signed_by_observable: Mapping[int, float]
    by_location: Mapping[str, float]
    signed_by_location: Mapping[str, float]

    def top_edges(self, top_k: int = 10) -> list[DetectorGraphEdgeHotspot]:
        edges = list(self.edge_hotspots)
        edges.sort(key=lambda edge: edge.hotspot, reverse=True)
        return edges[:top_k]

    def edge_table(self, top_k: int = 10) -> str:
        lines = [
            "edge_index\tlocation_id\tevent\tsensitivity\thotspot\t"
            "detectors\tobservables"
        ]
        for edge in self.top_edges(top_k):
            lines.append(
                f"{edge.edge_index}\t{edge.location_id}\t{edge.event}\t"
                f"{edge.sensitivity:.6g}\t{edge.hotspot:.6g}\t"
                f"{edge.detectors}\t{edge.observables}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class DetectorErrorModel:
    detectors: tuple[Detector, ...]
    observables: tuple[LogicalObservable, ...]
    edges: tuple[DetectorErrorEdge, ...]

    def to_dem_text(self, *, include_detector_coords: bool = True) -> str:
        lines: list[str] = []
        if include_detector_coords:
            for detector in self.detectors:
                if detector.coords:
                    coords = ", ".join(f"{coord:.17g}" for coord in detector.coords)
                    lines.append(f"detector({coords}) D{detector.id}")
                else:
                    lines.append(f"detector D{detector.id}")
        for edge in self.edges:
            lines.append(edge.to_dem_line())
        return "\n".join(lines)

    def edges_by_location(self) -> dict[str, list[DetectorErrorEdge]]:
        out: dict[str, list[DetectorErrorEdge]] = {}
        for edge in self.edges:
            out.setdefault(edge.location_id, []).append(edge)
        return out

    def project_hotspots_to_edges(
        self,
        hotspots: Mapping[str, float],
    ) -> dict[tuple[str, object], float]:
        """Assign a location-level hotspot to that location's DEM edges.

        The location hotspot is split across edges in proportion to their DEM
        probabilities. Locations with zero total edge probability are split
        uniformly.
        """

        projected: dict[tuple[str, object], float] = {}
        for location_id, edges in self.edges_by_location().items():
            hotspot = float(hotspots.get(location_id, 0.0))
            if not edges:
                continue
            total_probability = sum(edge.probability for edge in edges)
            for edge in edges:
                if total_probability > 0:
                    weight = edge.probability / total_probability
                else:
                    weight = 1.0 / len(edges)
                projected[(edge.location_id, edge.event)] = hotspot * weight
        return projected

    def project_result_to_detector_graph(self, result: object) -> DetectorGraphHotspots:
        """Project a SimulationResult-like object's sensitivities onto DEM edges."""

        sensitivities = getattr(result, "sensitivities")
        return self.project_sensitivities_to_detector_graph(sensitivities)

    def project_sensitivities_to_detector_graph(
        self,
        sensitivities: Mapping[str, float],
    ) -> DetectorGraphHotspots:
        """Map location-level signed sensitivities onto the detector graph.

        Each location's signed sensitivity is split across that location's DEM
        edges in proportion to edge probability. The returned object preserves
        signed sums and absolute hotspot sums for detector-graph edges,
        detector nodes, logical observables, and locations.
        """

        edges_by_location = self.edges_by_location()
        location_totals = {
            location_id: sum(edge.probability for edge in edges)
            for location_id, edges in edges_by_location.items()
        }

        edge_hotspots: list[DetectorGraphEdgeHotspot] = []
        by_detector_edge: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        signed_by_detector_edge: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        by_detector: dict[int, float] = {}
        signed_by_detector: dict[int, float] = {}
        by_observable: dict[int, float] = {}
        signed_by_observable: dict[int, float] = {}
        by_location: dict[str, float] = {}
        signed_by_location: dict[str, float] = {}

        for edge_index, edge in enumerate(self.edges):
            location_sensitivity = float(sensitivities.get(edge.location_id, 0.0))
            total_probability = location_totals.get(edge.location_id, 0.0)
            sibling_count = len(edges_by_location.get(edge.location_id, ()))
            if total_probability > 0:
                weight = edge.probability / total_probability
            elif sibling_count:
                weight = 1.0 / sibling_count
            else:
                weight = 0.0

            edge_sensitivity = location_sensitivity * weight
            edge_hotspot = abs(edge_sensitivity)
            edge_hotspots.append(
                DetectorGraphEdgeHotspot(
                    edge_index=edge_index,
                    location_id=edge.location_id,
                    event=edge.event,
                    probability=edge.probability,
                    detectors=edge.detectors,
                    observables=edge.observables,
                    sensitivity=edge_sensitivity,
                    hotspot=edge_hotspot,
                    weight=weight,
                )
            )

            graph_key = (edge.detectors, edge.observables)
            if edge_hotspot:
                _add(by_detector_edge, graph_key, edge_hotspot)
                _add(signed_by_detector_edge, graph_key, edge_sensitivity)
                _add(by_location, edge.location_id, edge_hotspot)
                _add(signed_by_location, edge.location_id, edge_sensitivity)

            detector_share = edge_hotspot / len(edge.detectors) if edge.detectors else 0.0
            signed_detector_share = (
                edge_sensitivity / len(edge.detectors) if edge.detectors else 0.0
            )
            if edge_hotspot:
                for detector_id in edge.detectors:
                    _add(by_detector, detector_id, detector_share)
                    _add(signed_by_detector, detector_id, signed_detector_share)

            observable_share = (
                edge_hotspot / len(edge.observables) if edge.observables else 0.0
            )
            signed_observable_share = (
                edge_sensitivity / len(edge.observables) if edge.observables else 0.0
            )
            if edge_hotspot:
                for observable_id in edge.observables:
                    _add(by_observable, observable_id, observable_share)
                    _add(signed_by_observable, observable_id, signed_observable_share)

        return DetectorGraphHotspots(
            edge_hotspots=tuple(edge_hotspots),
            by_detector_edge=by_detector_edge,
            signed_by_detector_edge=signed_by_detector_edge,
            by_detector=by_detector,
            signed_by_detector=signed_by_detector,
            by_observable=by_observable,
            signed_by_observable=signed_by_observable,
            by_location=by_location,
            signed_by_location=signed_by_location,
        )


@dataclass(frozen=True)
class _NoiseOccurrence:
    op_index: int
    location: NoiseLocation
    is_measurement_noise: bool


@dataclass(frozen=True)
class _RunRecord:
    measurements: Mapping[str, int]
    frame: PauliFrame


class DetectorErrorModelGenerator:
    """Generate a detector error model by propagating each single error event."""

    def __init__(
        self,
        circuit: Circuit,
        *,
        detectors: Sequence[Detector] | None = None,
        observables: Sequence[LogicalObservable] | None = None,
    ):
        self.circuit = circuit
        self.detectors = (
            tuple(detectors)
            if detectors is not None
            else _detectors_from_circuit(circuit)
        )
        self.observables = (
            tuple(observables)
            if observables is not None
            else _observables_from_circuit(circuit)
        )
        self._validate_declarations()
        self._occurrences = self._collect_noise_occurrences()

    def generate(self) -> DetectorErrorModel:
        reference = self._run_with_injection(None, None)
        reference_detectors = self._evaluate_detectors(reference)
        reference_observables = self._evaluate_observables(reference)

        edges: list[DetectorErrorEdge] = []
        for occurrence in self._occurrences:
            for event, probability in _non_identity_events(occurrence.location):
                injected = self._run_with_injection(occurrence.op_index, event)
                detector_flips = _flipped_ids(
                    reference_detectors,
                    self._evaluate_detectors(injected),
                )
                observable_flips = _flipped_ids(
                    reference_observables,
                    self._evaluate_observables(injected),
                )
                if not detector_flips and not observable_flips:
                    continue
                edges.append(
                    DetectorErrorEdge(
                        probability=probability,
                        detectors=detector_flips,
                        observables=observable_flips,
                        location_id=occurrence.location.id,
                        event=event,
                        tags=occurrence.location.tags,
                    )
                )

        return DetectorErrorModel(
            detectors=self.detectors,
            observables=self.observables,
            edges=tuple(edges),
        )

    def _run_with_injection(
        self,
        injected_op_index: int | None,
        injected_event: object | None,
    ) -> _RunRecord:
        state = StabilizerState.zero(self.circuit.n_qubits)
        frame = PauliFrame.zero(self.circuit.n_qubits)
        measurements: dict[str, int] = {}
        rng = random.Random(0)

        for op_index, operation in enumerate(self.circuit.operations):
            self._apply_operation(
                operation,
                op_index,
                injected_op_index,
                injected_event,
                state,
                frame,
                measurements,
                rng,
            )
        return _RunRecord(measurements=measurements, frame=frame)

    def _apply_operation(
        self,
        operation: Operation,
        op_index: int,
        injected_op_index: int | None,
        injected_event: object | None,
        state: StabilizerState,
        frame: PauliFrame,
        measurements: dict[str, int],
        rng: random.Random,
    ) -> None:
        kind = operation.kind
        if kind == "h":
            (qubit,) = operation.qubits
            state.apply_h(qubit)
            frame.apply_h(qubit)
            return
        if kind == "s":
            (qubit,) = operation.qubits
            state.apply_s(qubit)
            frame.apply_s(qubit)
            return
        if kind == "s_dag":
            (qubit,) = operation.qubits
            state.apply_s_dag(qubit)
            frame.apply_s_dag(qubit)
            return
        if kind == "cx":
            control, target = operation.qubits
            state.apply_cx(control, target)
            frame.apply_cx(control, target)
            return
        if kind == "cz":
            left, right = operation.qubits
            state.apply_cz(left, right)
            frame.apply_cz(left, right)
            return
        if kind == "swap":
            left, right = operation.qubits
            state.apply_swap(left, right)
            frame.apply_swap(left, right)
            return
        if kind == "pauli":
            if operation.pauli is None:
                raise ValueError("pauli operation requires a Pauli string")
            x, z = sparse_pauli_to_xz(
                self.circuit.n_qubits,
                operation.qubits,
                operation.pauli,
            )
            state.apply_pauli_string(x, z)
            return
        if kind == "noise":
            if op_index == injected_op_index:
                if operation.noise_location is None:
                    raise ValueError("noise operation requires a noise location")
                operation.noise_location.model.apply(
                    injected_event,
                    state,
                    frame,
                    operation.noise_location.qubits,
                )
            return
        if kind == "measure":
            bit = self._measure(operation, state, rng)
            if op_index == injected_op_index:
                bit ^= int(bool(injected_event))
            key = operation.key or f"m{len(measurements)}"
            _record_measurement(measurements, key, bit)
            return
        if kind == "measure_pauli":
            bit = self._measure_pauli(operation, state, rng)
            if op_index == injected_op_index:
                bit ^= int(bool(injected_event))
            key = operation.key or f"m{len(measurements)}"
            _record_measurement(measurements, key, bit)
            return
        if kind == "reset":
            self._reset(operation, state, frame, measurements, rng)
            return
        if kind in {"detector", "observable_include"}:
            return
        raise ValueError(f"unsupported operation kind {kind!r}")

    def _measure(
        self,
        operation: Operation,
        state: StabilizerState,
        rng: random.Random,
    ) -> int:
        (qubit,) = operation.qubits
        basis = operation.basis.upper()
        x, z = sparse_pauli_to_xz(self.circuit.n_qubits, operation.qubits, basis)
        if not state.is_deterministic_pauli(x, z):
            raise UnsupportedDemCircuitError(
                f"measurement {operation.key or operation.kind!r} is random in "
                "the ideal/single-error circuit"
            )
        if basis == "Z":
            return state.measure_z(qubit, rng)
        if basis == "X":
            return state.measure_x(qubit, rng)
        if basis == "Y":
            return state.measure_y(qubit, rng)
        raise ValueError(f"unsupported measurement basis {operation.basis!r}")

    def _measure_pauli(
        self,
        operation: Operation,
        state: StabilizerState,
        rng: random.Random,
    ) -> int:
        if operation.pauli is None:
            raise ValueError("measure_pauli operation requires a Pauli string")
        x, z = sparse_pauli_to_xz(self.circuit.n_qubits, operation.qubits, operation.pauli)
        if not state.is_deterministic_pauli(x, z):
            raise UnsupportedDemCircuitError(
                f"measurement {operation.key or operation.kind!r} is random in "
                "the ideal/single-error circuit"
            )
        return state.measure_pauli(x, z, rng)

    def _reset(
        self,
        operation: Operation,
        state: StabilizerState,
        frame: PauliFrame,
        measurements: dict[str, int],
        rng: random.Random,
    ) -> None:
        (qubit,) = operation.qubits
        basis = operation.basis.upper()
        if operation.key is not None:
            x, z = sparse_pauli_to_xz(self.circuit.n_qubits, operation.qubits, basis)
            if not state.is_deterministic_pauli(x, z):
                raise UnsupportedDemCircuitError(
                    f"reset measurement {operation.key!r} is random"
                )
            bit = state.deterministic_measurement_bit(x, z)
            _record_measurement(measurements, operation.key, bit)

        if basis == "Z":
            state.reset_z(qubit, rng)
        elif basis == "X":
            state.reset_x(qubit, rng)
        elif basis == "Y":
            state.reset_y(qubit, rng)
        else:
            raise ValueError(f"unsupported reset basis {operation.basis!r}")
        frame.reset(qubit)

    def _evaluate_detectors(self, run: _RunRecord) -> dict[int, int]:
        return {
            detector.id: _parity_from_measurements(
                run.measurements,
                detector.measurement_keys,
            )
            for detector in self.detectors
        }

    def _evaluate_observables(self, run: _RunRecord) -> dict[int, int]:
        values: dict[int, int] = {}
        for observable in self.observables:
            value = _parity_from_measurements(
                run.measurements,
                observable.measurement_keys,
            )
            if observable.pauli:
                value ^= run.frame.measurement_flip(
                    observable.pauli_qubits,
                    observable.pauli,
                )
            values[observable.id] = value
        return values

    def _collect_noise_occurrences(self) -> tuple[_NoiseOccurrence, ...]:
        occurrences: list[_NoiseOccurrence] = []
        seen_ids: set[str] = set()
        for op_index, operation in enumerate(self.circuit.operations):
            if operation.kind == "noise":
                if operation.noise_location is None:
                    raise ValueError("noise operation requires a noise location")
                location = operation.noise_location
                if location.id in seen_ids:
                    raise ValueError(
                        "DetectorErrorModelGenerator requires unique noise "
                        f"location ids; duplicate id {location.id!r}"
                    )
                seen_ids.add(location.id)
                occurrences.append(
                    _NoiseOccurrence(op_index, location, is_measurement_noise=False)
                )
            elif (
                operation.kind in {"measure", "measure_pauli"}
                and operation.noise_location is not None
            ):
                location = operation.noise_location
                if not isinstance(location.model, MeasurementBitFlip):
                    raise UnsupportedDemCircuitError(
                        "DEM generation currently supports MeasurementBitFlip "
                        "on measurement operations"
                    )
                if location.id in seen_ids:
                    raise ValueError(
                        "DetectorErrorModelGenerator requires unique noise "
                        f"location ids; duplicate id {location.id!r}"
                    )
                seen_ids.add(location.id)
                occurrences.append(
                    _NoiseOccurrence(op_index, location, is_measurement_noise=True)
                )
        return tuple(occurrences)

    def _validate_declarations(self) -> None:
        detector_ids = [detector.id for detector in self.detectors]
        if len(detector_ids) != len(set(detector_ids)):
            raise ValueError("detector ids must be unique")
        observable_ids = [observable.id for observable in self.observables]
        if len(observable_ids) != len(set(observable_ids)):
            raise ValueError("logical observable ids must be unique")


def _non_identity_events(location: NoiseLocation) -> tuple[tuple[object, float], ...]:
    model = location.model
    rate = location.rate
    if isinstance(model, BernoulliPauliNoise):
        return ((model.pauli, rate),)
    if isinstance(model, MeasurementBitFlip):
        return ((True, rate),)
    if isinstance(model, SingleQubitDepolarizing):
        return tuple((event, rate / 3.0) for event in ("X", "Y", "Z"))
    if isinstance(model, TwoQubitDepolarizing):
        return tuple((event, rate / len(model._events)) for event in model._events)
    if isinstance(model, PauliChannel):
        total_weight = model.total_weight
        return tuple(
            (event, rate * weight / total_weight)
            for event, weight in model.weights.items()
            if weight > 0
        )
    raise UnsupportedDemCircuitError(
        f"unsupported DEM noise model {type(model).__name__}"
    )


def _record_measurement(measurements: dict[str, int], key: str, bit: int) -> None:
    if key in measurements:
        raise ValueError(f"duplicate measurement key {key!r}")
    measurements[key] = int(bit)


def _parity_from_measurements(
    measurements: Mapping[str, int],
    keys: Sequence[str],
) -> int:
    parity = 0
    for key in keys:
        try:
            parity ^= int(measurements[key])
        except KeyError as exc:
            raise ValueError(f"unknown measurement key {key!r}") from exc
    return parity


def _flipped_ids(
    reference: Mapping[int, int],
    injected: Mapping[int, int],
) -> tuple[int, ...]:
    return tuple(
        item_id
        for item_id in sorted(reference)
        if int(reference[item_id]) ^ int(injected[item_id])
    )


def _add(out: dict[Any, float], key: Any, value: float) -> None:
    out[key] = out.get(key, 0.0) + value


def _detectors_from_circuit(circuit: Circuit) -> tuple[Detector, ...]:
    detectors: list[Detector] = []
    for operation in circuit.operations:
        if operation.kind != "detector":
            continue
        detector_id = operation.metadata.get("detector_id")
        if detector_id is None:
            detector_id = len(detectors)
        detectors.append(
            Detector(
                id=int(detector_id),
                measurement_keys=operation.measurement_keys,
                coords=tuple(operation.metadata.get("coords", ())),
            )
        )
    return tuple(detectors)


def _observables_from_circuit(circuit: Circuit) -> tuple[LogicalObservable, ...]:
    keys_by_id: dict[int, list[str]] = {}
    for operation in circuit.operations:
        if operation.kind != "observable_include":
            continue
        if operation.observable_id is None:
            raise ValueError("observable_include operation requires observable_id")
        keys_by_id.setdefault(operation.observable_id, []).extend(operation.measurement_keys)
    return tuple(
        LogicalObservable(id=observable_id, measurement_keys=tuple(keys))
        for observable_id, keys in sorted(keys_by_id.items())
    )

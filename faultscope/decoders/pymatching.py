"""Optional PyMatching decoder integration."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from faultscope.dem import DetectorErrorModel


class PyMatchingUnavailableError(ImportError):
    """Raised when optional PyMatching dependencies are unavailable."""


class UnsupportedPyMatchingDemError(ValueError):
    """Raised when a DEM cannot be represented as a graphlike matching problem."""


@dataclass(frozen=True)
class PyMatchingDecoder:
    """PyMatching-backed decoder for detector records and batch detector masks."""

    matching: Any
    detector_ids: tuple[int, ...]
    observable_ids: tuple[int, ...]
    edge_count: int

    def strong_id_payload(self) -> Mapping[str, object]:
        """Return the effective PyMatching graph used by this decoder."""

        module_name = type(self.matching).__module__.split(".", 1)[0]
        try:
            backend_module = importlib.import_module(module_name)
        except ImportError:
            backend_version = None
        else:
            backend_version = getattr(backend_module, "__version__", None)
        return {
            "backend": "pymatching-python",
            "backend_module": module_name,
            "backend_version": (None if backend_version is None else str(backend_version)),
            "decoder": "pymatching",
            "implementation_version": 1,
            "detector_ids": list(self.detector_ids),
            "observable_ids": list(self.observable_ids),
            "parameters": {},
            "source_edge_count": self.edge_count,
            "matching_graph": _matching_graph_identity_payload(self.matching),
        }

    @classmethod
    def from_dem(
        cls,
        dem: DetectorErrorModel,
        *,
        pymatching_module: Any | None = None,
        numpy_module: Any | None = None,
        scipy_sparse_module: Any | None = None,
    ) -> "PyMatchingDecoder":
        """Build a PyMatching decoder from a detector error model.

        The DEM must be graphlike: every edge may touch at most two detectors.
        Logical flips are passed through PyMatching's ``faults_matrix``.
        Parallel edges with the same endpoints must have identical logical
        effects because PyMatching cannot preserve conflicting fault IDs when
        it merges those edges.
        """

        pymatching, np, sparse = _load_optional_modules(
            pymatching_module,
            numpy_module,
            scipy_sparse_module,
        )
        problem = _compile_graphlike_problem(dem)
        detector_ids = tuple(int(detector_id) for detector_id in problem.detector_ids)
        observable_ids = tuple(int(observable_id) for observable_id in problem.observable_ids)
        edges = tuple(problem.edges)
        _validate_parallel_logical_effects(edges)

        h_rows: list[int] = []
        h_cols: list[int] = []
        h_data: list[int] = []
        f_rows: list[int] = []
        f_cols: list[int] = []
        f_data: list[int] = []
        weights: list[float] = []
        probabilities: list[float] = []

        for col, edge in enumerate(edges):
            probability = float(edge.probability)
            probabilities.append(probability)
            weights.append(float(edge.weight))
            for detector_index in edge.detectors:
                h_rows.append(int(detector_index))
                h_cols.append(col)
                h_data.append(1)
            for observable_index in edge.fault_observables:
                f_rows.append(int(observable_index))
                f_cols.append(col)
                f_data.append(1)

        h = sparse.csc_matrix(
            (h_data, (h_rows, h_cols)),
            shape=(len(detector_ids), len(edges)),
            dtype=np.uint8,
        )
        faults_matrix = sparse.csc_matrix(
            (f_data, (f_rows, f_cols)),
            shape=(len(observable_ids), len(edges)),
            dtype=np.uint8,
        )
        weights_array = np.array(weights, dtype=float)
        probabilities_array = np.array(probabilities, dtype=float)

        matching = _matching_from_check_matrix(
            pymatching,
            h,
            weights_array,
            probabilities_array,
            faults_matrix,
        )
        return cls(
            matching=matching,
            detector_ids=detector_ids,
            observable_ids=observable_ids,
            edge_count=len(edges),
        )

    def decode_detector_record(
        self,
        detector_record: Mapping[int, int] | Sequence[int],
    ) -> dict[int, int]:
        syndrome = self._coerce_detector_record(detector_record)
        prediction = self.matching.decode(syndrome)
        return self._prediction_to_dict(prediction)

    def decode_batch_detector_records(
        self,
        detector_records: Sequence[Mapping[int, int] | Sequence[int]],
    ) -> list[dict[int, int]]:
        syndromes = [self._coerce_detector_record(record) for record in detector_records]
        predictions = self.matching.decode_batch(syndromes)
        return [self._prediction_to_dict(row) for row in _rows(predictions)]

    def decode_batch_masks(
        self,
        detector_masks: Mapping[int, int] | object,
        *,
        shots: int | None = None,
    ) -> dict[int, int]:
        """Decode bit-packed detector masks into bit-packed observable masks.

        ``detector_masks`` can be a ``SampleBatch`` or any mapping from
        detector id to integer bit mask. The returned dict maps observable id to
        an integer bit mask of predicted logical flips.
        """

        if hasattr(detector_masks, "detectors") and hasattr(detector_masks, "shots"):
            shots = int(getattr(detector_masks, "shots"))
            masks = getattr(detector_masks, "detectors")
        else:
            masks = detector_masks
        if shots is None:
            raise ValueError("shots must be supplied when detector_masks is not a SampleBatch")
        if not isinstance(masks, Mapping):
            raise TypeError("detector_masks must be a mapping or SampleBatch-like object")

        syndromes = _masks_to_packed_shots(masks, self.detector_ids, shots)
        try:
            predictions = self.matching.decode_batch(
                syndromes,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            )
            return _packed_predictions_to_masks(
                predictions,
                self.observable_ids,
                shots,
            )
        except TypeError:
            dense_syndromes = _packed_shots_to_dense(syndromes, len(self.detector_ids))
            predictions = self.matching.decode_batch(dense_syndromes)
            return _dense_predictions_to_masks(
                predictions,
                self.observable_ids,
                shots,
            )

    def _coerce_detector_record(
        self,
        detector_record: Mapping[int, int] | Sequence[int],
    ) -> list[int]:
        if isinstance(detector_record, Mapping):
            return [int(detector_record.get(detector_id, 0)) for detector_id in self.detector_ids]
        syndrome = [int(bit) for bit in detector_record]
        if len(syndrome) != len(self.detector_ids):
            raise ValueError("detector record length does not match decoder detector count")
        return syndrome

    def _prediction_to_dict(self, prediction: Any) -> dict[int, int]:
        prediction_tuple = self._prediction_to_tuple(prediction)
        if len(prediction_tuple) != len(self.observable_ids):
            raise ValueError("PyMatching prediction length does not match decoder observable count")
        return {
            observable_id: bit for observable_id, bit in zip(self.observable_ids, prediction_tuple)
        }

    def _prediction_to_tuple(self, prediction: Any) -> tuple[int, ...]:
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        if isinstance(prediction, (bool, int)):
            prediction = [prediction]
        return tuple(int(bit) for bit in prediction)


def _matching_from_check_matrix(
    pymatching: Any,
    h: Any,
    weights: Any,
    probabilities: Any,
    faults_matrix: Any,
) -> Any:
    kwargs = {
        "weights": weights,
        "faults_matrix": faults_matrix,
        "error_probabilities": probabilities,
        "merge_strategy": "independent",
        "use_virtual_boundary_node": True,
    }
    try:
        return pymatching.Matching.from_check_matrix(h, **kwargs)
    except TypeError:
        kwargs.pop("error_probabilities")
        return pymatching.Matching.from_check_matrix(h, **kwargs)


def _matching_graph_identity_payload(matching: Any) -> dict[str, object]:
    edges_method = getattr(matching, "edges", None)
    if not callable(edges_method):
        raise TypeError("PyMatchingDecoder matching object does not expose stable edges()")

    edges: list[dict[str, object]] = []
    for raw_edge in edges_method():
        if len(raw_edge) != 3:
            raise TypeError("PyMatchingDecoder matching edge must contain two endpoints and data")
        left, right, raw_data = raw_edge
        if not isinstance(raw_data, Mapping):
            raise TypeError("PyMatchingDecoder matching edge data must be a mapping")
        left = int(left)
        right = None if right is None else int(right)
        if right is not None and right < left:
            left, right = right, left
        fault_ids = raw_data.get("fault_ids", ())
        edges.append(
            {
                "left": left,
                "right": right,
                "fault_ids": sorted(int(value) for value in fault_ids),
                "weight": float(raw_data.get("weight", 1.0)),
                "error_probability": float(raw_data.get("error_probability", -1.0)),
            }
        )
    edges.sort(key=lambda edge: json.dumps(edge, sort_keys=True, separators=(",", ":")))
    boundary = getattr(matching, "boundary", ())
    return {
        "num_detectors": int(getattr(matching, "num_detectors")),
        "num_nodes": int(getattr(matching, "num_nodes")),
        "num_fault_ids": int(getattr(matching, "num_fault_ids")),
        "boundary": sorted(int(value) for value in boundary),
        "edges": edges,
    }


def _load_optional_modules(
    pymatching_module: Any | None,
    numpy_module: Any | None,
    scipy_sparse_module: Any | None,
) -> tuple[Any, Any, Any]:
    if pymatching_module is None:
        try:
            import pymatching as imported_pymatching
        except ImportError as exc:
            raise PyMatchingUnavailableError(
                "PyMatching is required for PyMatchingDecoder. "
                "Install it with `pip install pymatching`."
            ) from exc
        pymatching_module = imported_pymatching
    if numpy_module is None:
        try:
            import numpy as imported_numpy
        except ImportError as exc:
            raise PyMatchingUnavailableError("NumPy is required for PyMatchingDecoder.") from exc
        numpy_module = imported_numpy
    if scipy_sparse_module is None:
        try:
            from scipy import sparse as imported_sparse
        except ImportError as exc:
            raise PyMatchingUnavailableError("SciPy is required for PyMatchingDecoder.") from exc
        scipy_sparse_module = imported_sparse
    return pymatching_module, numpy_module, scipy_sparse_module


def _compile_graphlike_problem(dem: DetectorErrorModel) -> Any:
    try:
        return dem.compile_graphlike_problem()
    except ValueError as exc:
        if not dem.is_graphlike():
            raise UnsupportedPyMatchingDemError(str(exc)) from exc
        raise


def _validate_parallel_logical_effects(edges: Sequence[Any]) -> None:
    groups: dict[
        tuple[int, int | None],
        tuple[tuple[int, ...], list[int]],
    ] = {}
    for edge in edges:
        detectors = tuple(int(detector) for detector in edge.detectors)
        if not detectors:
            continue
        endpoint: tuple[int, int | None]
        if len(detectors) == 1:
            endpoint = (detectors[0], None)
        elif len(detectors) == 2:
            endpoint = (min(detectors), max(detectors))
        else:  # pragma: no cover - compile_graphlike_problem rejects this first.
            raise UnsupportedPyMatchingDemError(
                f"pymatching edge {edge.dem_edge_index} has {len(detectors)} detectors; "
                "expected one boundary detector or two graph detectors"
            )

        fault_observables = tuple(int(index) for index in edge.fault_observables)
        dem_edge_index = int(edge.dem_edge_index)
        existing = groups.get(endpoint)
        if existing is None:
            groups[endpoint] = (fault_observables, [dem_edge_index])
            continue

        existing_observables, existing_edge_indices = existing
        if existing_observables != fault_observables:
            raise UnsupportedPyMatchingDemError(
                f"pymatching found ambiguous parallel endpoint {endpoint}: "
                f"DEM edge {dem_edge_index} fault_observables {fault_observables} "
                f"conflicts with DEM edge(s) {tuple(existing_edge_indices)} "
                f"fault_observables {existing_observables}"
            )
        existing_edge_indices.append(dem_edge_index)


def _rows(predictions: Any) -> list[Any]:
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    return list(predictions)


def _masks_to_packed_shots(
    masks: Mapping[int, int],
    detector_ids: Sequence[int],
    shots: int,
) -> Any:
    import numpy as np

    byte_count_by_shots = (shots + 7) // 8
    byte_count_by_detectors = (len(detector_ids) + 7) // 8
    packed = np.zeros((shots, byte_count_by_detectors), dtype=np.uint8)
    if shots == 0 or not detector_ids:
        return packed
    all_mask = (1 << shots) - 1
    for detector_index, detector_id in enumerate(detector_ids):
        mask = int(masks.get(detector_id, 0)) & all_mask
        shot_bytes = mask.to_bytes(byte_count_by_shots, "little")
        shot_bits = np.unpackbits(
            np.frombuffer(shot_bytes, dtype=np.uint8),
            bitorder="little",
        )[:shots]
        packed[:, detector_index // 8] |= shot_bits.astype(np.uint8) << (detector_index % 8)
    return packed


def _packed_shots_to_dense(packed: Any, detector_count: int) -> Any:
    import numpy as np

    if detector_count == 0:
        return np.zeros((packed.shape[0], 0), dtype=np.uint8)
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        bitorder="little",
        axis=1,
    )[:, :detector_count].astype(np.uint8, copy=False)


def _packed_predictions_to_masks(
    predictions: Any,
    observable_ids: Sequence[int],
    shots: int,
) -> dict[int, int]:
    import numpy as np

    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape((shots, -1))
    if predictions.ndim != 2 or predictions.shape[0] != shots:
        raise ValueError(f"unexpected PyMatching prediction shape {predictions.shape}")
    out = {observable_id: 0 for observable_id in observable_ids}
    for observable_index, observable_id in enumerate(observable_ids):
        if observable_index // 8 >= predictions.shape[1]:
            raise ValueError("PyMatching prediction width does not match decoder observable count")
        bits = (predictions[:, observable_index // 8] >> (observable_index % 8)) & 1
        packed_bits = np.packbits(bits.astype(np.uint8), bitorder="little")
        out[observable_id] = int.from_bytes(packed_bits.tobytes(), "little")
    return out


def _dense_predictions_to_masks(
    predictions: Any,
    observable_ids: Sequence[int],
    shots: int,
) -> dict[int, int]:
    import numpy as np

    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape((shots, 1))
    if predictions.ndim != 2 or predictions.shape[0] != shots:
        raise ValueError(f"unexpected PyMatching prediction shape {predictions.shape}")
    if predictions.shape[1] != len(observable_ids):
        raise ValueError("PyMatching prediction length does not match decoder observable count")
    out = {observable_id: 0 for observable_id in observable_ids}
    for observable_index, observable_id in enumerate(observable_ids):
        packed_bits = np.packbits(
            predictions[:, observable_index].astype(np.uint8),
            bitorder="little",
        )
        out[observable_id] = int.from_bytes(packed_bits.tobytes(), "little")
    return out

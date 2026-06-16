"""Optional PyMatching decoder integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from npsim.dem import DetectorErrorModel


class PyMatchingUnavailableError(ImportError):
    """Raised when optional PyMatching dependencies are unavailable."""


class UnsupportedPyMatchingDemError(ValueError):
    """Raised when a DEM cannot be represented as a graphlike matching problem."""


@dataclass(frozen=True)
class PyMatchingBatchDecoder:
    """PyMatching-backed decoder for detector records and batch detector masks."""

    matching: Any
    detector_ids: tuple[int, ...]
    observable_ids: tuple[int, ...]
    edge_count: int

    @classmethod
    def from_dem(
        cls,
        dem: DetectorErrorModel,
        *,
        pymatching_module: Any | None = None,
        numpy_module: Any | None = None,
        scipy_sparse_module: Any | None = None,
    ) -> "PyMatchingBatchDecoder":
        """Build a PyMatching decoder from a detector error model.

        The DEM must be graphlike: every edge may touch at most two detectors.
        Logical flips are passed through PyMatching's ``faults_matrix``.
        """

        pymatching, np, sparse = _load_optional_modules(
            pymatching_module,
            numpy_module,
            scipy_sparse_module,
        )
        detector_ids, observable_ids = _dem_ids(dem)
        _validate_graphlike_dem(dem)

        detector_index = {detector_id: idx for idx, detector_id in enumerate(detector_ids)}
        observable_index = {
            observable_id: idx for idx, observable_id in enumerate(observable_ids)
        }

        h_rows: list[int] = []
        h_cols: list[int] = []
        h_data: list[int] = []
        f_rows: list[int] = []
        f_cols: list[int] = []
        f_data: list[int] = []
        weights: list[float] = []
        probabilities: list[float] = []

        for col, edge in enumerate(dem.edges):
            probability = _clamped_probability(edge.probability)
            probabilities.append(probability)
            weights.append(math.log((1.0 - probability) / probability))
            for detector_id in edge.detectors:
                h_rows.append(detector_index[detector_id])
                h_cols.append(col)
                h_data.append(1)
            for observable_id in edge.observables:
                f_rows.append(observable_index[observable_id])
                f_cols.append(col)
                f_data.append(1)

        h = sparse.csc_matrix(
            (h_data, (h_rows, h_cols)),
            shape=(len(detector_ids), len(dem.edges)),
            dtype=np.uint8,
        )
        faults_matrix = sparse.csc_matrix(
            (f_data, (f_rows, f_cols)),
            shape=(len(observable_ids), len(dem.edges)),
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
            edge_count=len(dem.edges),
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

        ``detector_masks`` can be a ``BatchTrajectory`` or any mapping from
        detector id to integer bit mask. The returned dict maps observable id to
        an integer bit mask of predicted logical flips.
        """

        if hasattr(detector_masks, "detectors") and hasattr(detector_masks, "shots"):
            shots = int(getattr(detector_masks, "shots"))
            masks = getattr(detector_masks, "detectors")
        else:
            masks = detector_masks
        if shots is None:
            raise ValueError("shots must be supplied when detector_masks is not a BatchTrajectory")
        if not isinstance(masks, Mapping):
            raise TypeError("detector_masks must be a mapping or BatchTrajectory-like object")

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
            return [
                int(detector_record.get(detector_id, 0))
                for detector_id in self.detector_ids
            ]
        syndrome = [int(bit) for bit in detector_record]
        if len(syndrome) != len(self.detector_ids):
            raise ValueError(
                "detector record length does not match decoder detector count"
            )
        return syndrome

    def _prediction_to_dict(self, prediction: Any) -> dict[int, int]:
        prediction_tuple = self._prediction_to_tuple(prediction)
        if len(prediction_tuple) != len(self.observable_ids):
            raise ValueError(
                "PyMatching prediction length does not match decoder observable count"
            )
        return {
            observable_id: bit
            for observable_id, bit in zip(self.observable_ids, prediction_tuple)
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


def _load_optional_modules(
    pymatching_module: Any | None,
    numpy_module: Any | None,
    scipy_sparse_module: Any | None,
) -> tuple[Any, Any, Any]:
    if pymatching_module is None:
        try:
            import pymatching as pymatching_module
        except ImportError as exc:
            raise PyMatchingUnavailableError(
                "PyMatching is required for PyMatchingBatchDecoder. "
                "Install it with `pip install pymatching`."
            ) from exc
    if numpy_module is None:
        try:
            import numpy as numpy_module
        except ImportError as exc:
            raise PyMatchingUnavailableError(
                "NumPy is required for PyMatchingBatchDecoder."
            ) from exc
    if scipy_sparse_module is None:
        try:
            from scipy import sparse as scipy_sparse_module
        except ImportError as exc:
            raise PyMatchingUnavailableError(
                "SciPy is required for PyMatchingBatchDecoder."
            ) from exc
    return pymatching_module, numpy_module, scipy_sparse_module


def _dem_ids(dem: DetectorErrorModel) -> tuple[tuple[int, ...], tuple[int, ...]]:
    detector_ids = {detector.id for detector in dem.detectors}
    observable_ids = {observable.id for observable in dem.observables}
    for edge in dem.edges:
        detector_ids.update(edge.detectors)
        observable_ids.update(edge.observables)
    return tuple(sorted(detector_ids)), tuple(sorted(observable_ids))


def _validate_graphlike_dem(dem: DetectorErrorModel) -> None:
    for edge_index, edge in enumerate(dem.edges):
        if len(edge.detectors) > 2:
            raise UnsupportedPyMatchingDemError(
                "PyMatching graph construction requires graphlike DEM edges "
                f"with at most two detectors; edge {edge_index} has {edge.detectors}"
            )
        if not edge.detectors and edge.observables:
            raise UnsupportedPyMatchingDemError(
                "DEM contains an undetectable logical edge with no detectors; "
                f"edge {edge_index} has observables {edge.observables}"
            )


def _clamped_probability(probability: float) -> float:
    eps = 1e-15
    return min(1.0 - eps, max(eps, float(probability)))


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
        packed[:, detector_index // 8] |= (
            shot_bits.astype(np.uint8) << (detector_index % 8)
        )
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
            raise ValueError(
                "PyMatching prediction width does not match decoder observable count"
            )
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
        raise ValueError(
            "PyMatching prediction length does not match decoder observable count"
        )
    out = {observable_id: 0 for observable_id in observable_ids}
    for observable_index, observable_id in enumerate(observable_ids):
        packed_bits = np.packbits(
            predictions[:, observable_index].astype(np.uint8),
            bitorder="little",
        )
        out[observable_id] = int.from_bytes(packed_bits.tobytes(), "little")
    return out

"""Detector-error-model hotspot sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from npsim.dem.model import (
    DetectorErrorModel,
    DetectorGraphEdgeHotspot,
    DetectorGraphHotspots,
)


DemCorrectionMaskFn = Callable[["DemBatchTrajectory"], Mapping[int, int]]
DemLossMaskFn = Callable[["DemBatchTrajectory", Mapping[int, int]], int]


@dataclass(frozen=True)
class DemBatchTrajectory:
    """Bit-packed DEM sampling result.

    The least-significant bit corresponds to shot 0. ``edge_event_masks[e]`` is
    one on shots where DEM edge ``e`` was sampled.
    """

    shots: int
    all_mask: int
    detectors: Mapping[int, int]
    observables: Mapping[int, int]
    edge_event_masks: Mapping[int, int]

    def bit(self, mask: int, shot: int) -> int:
        return (mask >> shot) & 1

    def detector_bit(self, detector_id: int, shot: int) -> int:
        return self.bit(self.detectors[detector_id], shot)

    def observable_bit(self, observable_id: int, shot: int) -> int:
        return self.bit(self.observables[observable_id], shot)

    def edge_event_bit(self, edge_index: int, shot: int) -> int:
        return self.bit(self.edge_event_masks[edge_index], shot)


@dataclass(frozen=True)
class DemLocationMetadata:
    """Location metadata reconstructed from DEM edge tags."""

    id: str
    tags: Mapping[str, Any]
    qubits: tuple[int, ...] = ()


@dataclass(frozen=True)
class DemLocationHotspotRow:
    location_id: str
    sensitivity: float
    hotspot: float
    qubits: tuple[int, ...]
    tags: Mapping[str, Any]


@dataclass(frozen=True)
class DemEdgeHotspotRow:
    edge_index: int
    location_id: str
    event: object
    probability: float
    detectors: tuple[int, ...]
    observables: tuple[int, ...]
    sensitivity: float
    hotspot: float


@dataclass(frozen=True)
class DemHotspotResult:
    """Noise hotspot estimate from detector-error-model sampling."""

    dem: DetectorErrorModel
    shots: int
    mean_loss: float
    baseline: float
    edge_sensitivities: Mapping[int, float]
    edge_hotspots: Mapping[int, float]
    sensitivities: Mapping[str, float]
    hotspots: Mapping[str, float]
    by_detector: Mapping[int, float]
    by_round: Mapping[Any, float]
    by_gate: Mapping[Any, float]
    by_operation: Mapping[Any, float]
    locations: Mapping[str, DemLocationMetadata]
    detector_graph_hotspots: DetectorGraphHotspots
    top_edges_cache: tuple[DemEdgeHotspotRow, ...] = ()
    top_hotspots_cache: tuple[DemLocationHotspotRow, ...] = ()

    @property
    def logical_failure_rate(self) -> float:
        return self.mean_loss

    def top_edges(self, top_k: int = 10) -> list[DemEdgeHotspotRow]:
        if self.top_edges_cache and top_k <= len(self.top_edges_cache):
            return list(self.top_edges_cache[:top_k])
        rows = [
            DemEdgeHotspotRow(
                edge_index=edge_index,
                location_id=edge.location_id,
                event=edge.event,
                probability=edge.probability,
                detectors=edge.detectors,
                observables=edge.observables,
                sensitivity=self.edge_sensitivities[edge_index],
                hotspot=self.edge_hotspots[edge_index],
            )
            for edge_index, edge in enumerate(self.dem.edges)
        ]
        rows.sort(key=lambda row: row.hotspot, reverse=True)
        return rows[:top_k]

    def top_hotspots(self, top_k: int = 10) -> list[DemLocationHotspotRow]:
        if self.top_hotspots_cache and top_k <= len(self.top_hotspots_cache):
            return list(self.top_hotspots_cache[:top_k])
        rows = [
            DemLocationHotspotRow(
                location_id=location_id,
                sensitivity=self.sensitivities[location_id],
                hotspot=self.hotspots[location_id],
                qubits=self.locations[location_id].qubits,
                tags=self.locations[location_id].tags,
            )
            for location_id in self.hotspots
        ]
        rows.sort(key=lambda row: row.hotspot, reverse=True)
        return rows[:top_k]

    def hotspot_table(self, top_k: int = 10) -> str:
        lines = ["location_id\tsensitivity\thotspot\ttags"]
        for row in self.top_hotspots(top_k):
            lines.append(
                f"{row.location_id}\t{row.sensitivity:.6g}\t"
                f"{row.hotspot:.6g}\t{dict(row.tags)}"
            )
        return "\n".join(lines)


class DemBatchHotspotSimulator:
    """Native-backed DEM sampler for logical failure and hotspot estimation."""

    def __init__(self, dem: DetectorErrorModel):
        self.dem = dem
        self._validate_probabilities()

    def run_batch(
        self,
        *,
        shots: int,
        rng: random.Random | None = None,
        seed: int | None = None,
        return_edge_events: bool = True,
    ) -> DemBatchTrajectory:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if rng is not None and seed is not None:
            raise ValueError("supply either seed or rng, not both")
        if rng is not None:
            seed = rng.getrandbits(64)
        from npsim.runtime.native import compile_native_dem_sampler

        return compile_native_dem_sampler(
            self.dem,
            backend="native",
        ).run_batch(
            shots=shots,
            seed=seed,
            return_edge_events=return_edge_events,
        )

    def estimate(
        self,
        *,
        shots: int,
        seed: int | None = None,
        decoder: Any | None = None,
        correction_mask_fn: DemCorrectionMaskFn | None = None,
        loss_mask_fn: DemLossMaskFn | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> DemHotspotResult:
        if decoder is not None and correction_mask_fn is not None:
            raise ValueError("supply either decoder or correction_mask_fn, not both")
        from npsim.runtime.native import compile_native_dem_sampler

        return compile_native_dem_sampler(
            self.dem,
            backend="native",
        ).estimate(
            shots=shots,
            seed=seed,
            decoder=decoder,
            correction_mask_fn=correction_mask_fn,
            loss_mask_fn=loss_mask_fn,
            baseline=baseline,
            top_k=top_k,
        )

    def _validate_probabilities(self) -> None:
        for edge_index, edge in enumerate(self.dem.edges):
            if not 0.0 <= edge.probability <= 1.0:
                raise ValueError(
                    f"DEM edge {edge_index} probability must be in [0, 1], "
                    f"got {edge.probability}"
                )


def _default_loss_mask(
    batch: DemBatchTrajectory,
    corrections: Mapping[int, int],
    dem: DetectorErrorModel,
) -> int:
    observable_ids = {observable.id for observable in dem.observables}
    observable_ids.update(batch.observables)
    observable_ids.update(corrections)
    loss_mask = 0
    for observable_id in observable_ids:
        loss_mask |= batch.observables.get(observable_id, 0) ^ corrections.get(
            observable_id,
            0,
        )
    return loss_mask


def _aggregate_detector_hotspots(
    dem: DetectorErrorModel,
    edge_hotspots: Mapping[int, float],
) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for edge_index, edge in enumerate(dem.edges):
        hotspot = edge_hotspots[edge_index]
        if not hotspot or not edge.detectors:
            continue
        share = hotspot / len(edge.detectors)
        for detector_id in edge.detectors:
            out[detector_id] += share
    return dict(out)


def _aggregate_by_tag(
    hotspots: Mapping[str, float],
    locations: Mapping[str, DemLocationMetadata],
    tag: str,
) -> dict[Any, float]:
    out: dict[Any, float] = defaultdict(float)
    for location_id, hotspot in hotspots.items():
        value = locations[location_id].tags.get(tag)
        if value is not None:
            out[value] += hotspot
    return dict(out)


def _edge_sensitivities_to_detector_graph(
    dem: DetectorErrorModel,
    edge_sensitivities: Mapping[int, float],
) -> DetectorGraphHotspots:
    edge_hotspots: list[DetectorGraphEdgeHotspot] = []
    by_detector_edge: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    signed_by_detector_edge: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    by_detector: dict[int, float] = {}
    signed_by_detector: dict[int, float] = {}
    by_observable: dict[int, float] = {}
    signed_by_observable: dict[int, float] = {}
    by_location: dict[str, float] = {}
    signed_by_location: dict[str, float] = {}

    for edge_index, edge in enumerate(dem.edges):
        sensitivity = float(edge_sensitivities.get(edge_index, 0.0))
        hotspot = abs(sensitivity)
        edge_hotspots.append(
            DetectorGraphEdgeHotspot(
                edge_index=edge_index,
                location_id=edge.location_id,
                event=edge.event,
                probability=edge.probability,
                detectors=edge.detectors,
                observables=edge.observables,
                sensitivity=sensitivity,
                hotspot=hotspot,
                weight=1.0,
            )
        )
        if not hotspot:
            continue

        graph_key = (edge.detectors, edge.observables)
        _add(by_detector_edge, graph_key, hotspot)
        _add(signed_by_detector_edge, graph_key, sensitivity)
        _add(by_location, edge.location_id, hotspot)
        _add(signed_by_location, edge.location_id, sensitivity)

        detector_share = hotspot / len(edge.detectors) if edge.detectors else 0.0
        signed_detector_share = (
            sensitivity / len(edge.detectors) if edge.detectors else 0.0
        )
        for detector_id in edge.detectors:
            _add(by_detector, detector_id, detector_share)
            _add(signed_by_detector, detector_id, signed_detector_share)

        observable_share = hotspot / len(edge.observables) if edge.observables else 0.0
        signed_observable_share = (
            sensitivity / len(edge.observables) if edge.observables else 0.0
        )
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


def _add(out: dict[Any, float], key: Any, value: float) -> None:
    out[key] = out.get(key, 0.0) + value

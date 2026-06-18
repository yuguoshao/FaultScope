"""Detector-error-model hotspot sampling."""

from __future__ import annotations

from typing import Callable, Mapping

from npsim._npsim_native import (
    DemBatchHotspotSimulator,
    DemBatchTrajectory,
    DemEdgeHotspotRow,
    DemHotspotResult,
    DemLocationHotspotRow,
    DemLocationMetadata,
)


DemCorrectionMaskFn = Callable[["DemBatchTrajectory"], Mapping[int, int]]
DemLossMaskFn = Callable[["DemBatchTrajectory", Mapping[int, int]], int]

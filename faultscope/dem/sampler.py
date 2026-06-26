"""Detector-error-model hotspot sampling."""

from __future__ import annotations

from typing import Callable, Mapping

from faultscope._native import (
    DemHotspotEstimator,
    DemSampleBatch,
    DemEdgeHotspot,
    DemHotspotEstimate,
    DemLocationHotspot,
    DemLocationMetadata,
)


DemCorrectionMaskFn = Callable[["DemSampleBatch"], Mapping[int, int]]
DemLossMaskFn = Callable[["DemSampleBatch", Mapping[int, int]], int]

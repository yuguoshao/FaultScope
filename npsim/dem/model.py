"""Detector error model generation and data structures."""

from __future__ import annotations

from npsim._npsim_native import (
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    DetectorErrorModelGenerator,
    DetectorGraphEdgeHotspot,
    DetectorGraphHotspots,
    LogicalObservable,
)


class UnsupportedDemCircuitError(ValueError):
    """Raised when a circuit cannot be converted by single-error propagation."""

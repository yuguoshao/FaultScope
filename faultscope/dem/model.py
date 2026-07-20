"""Detector error model generation and data structures."""

from __future__ import annotations

from faultscope._native import (
    BinaryLinearDecodingProblem,
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    DetectorErrorModelGenerator,
    DetectorGraphEdgeHotspot,
    DetectorGraphHotspots,
    GeneratedDetectorErrorModel,
    GraphlikeDecompositionHints,
    GraphlikeDecodingProblem,
    IndexedDem,
    LogicalObservable,
    SparseBinaryMatrix,
)


class UnsupportedDemCircuitError(ValueError):
    """Raised when a circuit cannot be converted by single-error propagation."""

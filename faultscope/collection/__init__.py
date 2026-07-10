"""Logical error-rate and threshold-style collection APIs."""

from faultscope.collection._collect import collect, iter_collect
from faultscope.collection._types import (
    COLLECTION_CSV_FIELDS,
    COLLECTION_CSV_HEADER,
    CollectionData,
    CollectionOptions,
    CollectionTask,
    Progress,
    TaskStats,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)
from faultscope.collection.threshold import (
    FiniteSizeScalingFit,
    PairwiseCrossing,
    ThresholdAnalysisResult,
    ThresholdEstimate,
    ThresholdPoint,
    analyze_thresholds,
    plot_threshold_analysis,
)

__all__ = [
    "COLLECTION_CSV_FIELDS",
    "COLLECTION_CSV_HEADER",
    "CollectionData",
    "CollectionOptions",
    "CollectionTask",
    "Progress",
    "TaskStats",
    "collect",
    "iter_collect",
    "read_stats_from_csv_files",
    "write_stats_to_csv_file",
    "FiniteSizeScalingFit",
    "PairwiseCrossing",
    "ThresholdAnalysisResult",
    "ThresholdEstimate",
    "ThresholdPoint",
    "analyze_thresholds",
    "plot_threshold_analysis",
]

"""Logical error-rate and threshold-style collection APIs."""

from faultscope.collection._collect import Collector, collect, iter_collect, iter_progress
from faultscope.collection._types import (
    COLLECTION_CSV_FIELDS,
    COLLECTION_CSV_HEADER,
    CollectionData,
    CollectionOptions,
    CollectionRunOptions,
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
    "Collector",
    "CollectionOptions",
    "CollectionRunOptions",
    "CollectionTask",
    "Progress",
    "TaskStats",
    "collect",
    "iter_collect",
    "iter_progress",
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

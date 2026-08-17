"""Legacy detector-error-model collection APIs.

The primary :mod:`faultscope.collection` package executes circuits directly.
This module preserves explicit DEM sampling as an isolated compatibility path.
"""

from faultscope.collection._dem_collect import (
    DemCollector,
    collect,
    collect_hotspots,
    iter_collect,
    iter_progress,
)
from faultscope.collection._dem_types import DemCollectionTask, DemHotspotCollectionResult
from faultscope.collection._types import (
    COLLECTION_COUNTER_SCHEMA_VERSION,
    COLLECTION_CSV_FIELDS,
    COLLECTION_CSV_HEADER,
    CollectionCounterSchema,
    CollectionData,
    CollectionOptions,
    CollectionRunOptions,
    Progress,
    TaskStats,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)

__all__ = [
    "COLLECTION_COUNTER_SCHEMA_VERSION",
    "COLLECTION_CSV_FIELDS",
    "COLLECTION_CSV_HEADER",
    "CollectionCounterSchema",
    "CollectionData",
    "CollectionOptions",
    "CollectionRunOptions",
    "DemCollectionTask",
    "DemCollector",
    "DemHotspotCollectionResult",
    "Progress",
    "TaskStats",
    "collect",
    "collect_hotspots",
    "iter_collect",
    "iter_progress",
    "read_stats_from_csv_files",
    "write_stats_to_csv_file",
]

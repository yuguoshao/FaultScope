//! Logical error-rate collection for FaultScope.

mod api;
mod counting;
mod hotspot;
mod scheduler;
mod worker_decoder;
mod worker_executor;

pub use api::{
    collect_dem_logical_error_stats, collect_dem_logical_error_tasks,
    collect_dem_logical_error_tasks_with_progress, sample_dem_logical_error_stats,
    DemLogicalCollectionOptions, DemLogicalCollectionRunOptions, DemLogicalCollectionStats,
    DemLogicalCollectionTask, DemLogicalCounterSchema, DEM_LOGICAL_COUNTER_SCHEMA_VERSION,
};
pub use hotspot::{collect_dem_hotspot_tasks, DemHotspotCollectionResult};

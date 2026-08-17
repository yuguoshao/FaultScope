//! Logical error-rate collection for FaultScope.

mod api;
mod counting;
mod forward_hotspot;
mod hotspot;
mod scheduler;
mod worker_decoder;
mod worker_executor;

pub use api::{
    collect_dem_logical_error_stats, collect_dem_logical_error_tasks,
    collect_dem_logical_error_tasks_with_progress, collect_forward_logical_error_stats,
    collect_forward_logical_error_tasks, collect_forward_logical_error_tasks_with_progress,
    sample_dem_logical_error_stats, sample_forward_logical_error_stats,
    DemLogicalCollectionOptions, DemLogicalCollectionRunOptions, DemLogicalCollectionStats,
    DemLogicalCollectionTask, DemLogicalCounterSchema, ForwardLogicalCollectionTask,
    DEM_LOGICAL_COUNTER_SCHEMA_VERSION,
};
pub use forward_hotspot::{collect_forward_hotspot_tasks, ForwardHotspotCollectionResult};
pub use hotspot::{collect_dem_hotspot_tasks, DemHotspotCollectionResult};

pub use api::{
    DemLogicalCollectionOptions as LogicalCollectionOptions,
    DemLogicalCollectionRunOptions as LogicalCollectionRunOptions,
    DemLogicalCollectionStats as LogicalCollectionStats,
    DemLogicalCounterSchema as LogicalCounterSchema,
};
pub const LOGICAL_COUNTER_SCHEMA_VERSION: u32 = DEM_LOGICAL_COUNTER_SCHEMA_VERSION;

//! Logical error-rate collection for FaultScope.

mod api;
mod counting;
mod scheduler;

pub use api::{
    collect_dem_logical_error_stats, collect_dem_logical_error_tasks,
    collect_dem_logical_error_tasks_with_progress, sample_dem_logical_error_stats,
    DemLogicalCollectionOptions, DemLogicalCollectionRunOptions, DemLogicalCollectionStats,
    DemLogicalCollectionTask,
};

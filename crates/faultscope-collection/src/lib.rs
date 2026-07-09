//! Logical error-rate collection for FaultScope.

pub mod api;
mod counting;
mod scheduler;

pub use api::*;

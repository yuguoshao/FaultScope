//! Rust core types for NPSim.
//!
//! This crate is the Python-independent home for NPSim's circuit, detector
//! error model, and hotspot data model.  The PyO3 crate is responsible for
//! adapting these types to Python and for the remaining Python callback
//! integration while the algorithms are migrated here.

pub mod compile;
pub mod dem;
pub mod dem_sampling;
pub mod expr;
pub mod hotspot;
pub mod mask;
pub mod model;
pub mod packed;
pub mod pauli;
pub mod rng;
pub mod sampling;
pub mod stabilizer;

pub use compile::*;
pub use dem::*;
pub use dem_sampling::*;
pub use expr::*;
pub use hotspot::*;
pub use mask::*;
pub use model::*;
pub use packed::*;
pub use pauli::*;
pub use rng::*;
pub use sampling::*;
pub use stabilizer::*;

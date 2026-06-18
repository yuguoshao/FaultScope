use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::{Expr, Mask};

pub type NpResult<T> = Result<T, NpError>;

/// Error type returned by the Rust core APIs.
///
/// The core crate is still pre-1.0, so error messages are intended to be
/// readable diagnostics rather than a stable machine-readable taxonomy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpError {
    message: String,
}

impl NpError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for NpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl Error for NpError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TagValue {
    None,
    Bool(bool),
    Int(i64),
    Float(u64),
    String(String),
}

impl TagValue {
    pub fn float(value: f64) -> NpResult<Self> {
        if value.is_nan() {
            return Err(NpError::new("tags do not support NaN"));
        }
        Ok(Self::Float(value.to_bits()))
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NoiseModel {
    BernoulliPauli(String),
    MeasurementBitFlip,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
    PauliChannel(Vec<(String, f64)>),
}

/// A named stochastic noise source attached to a circuit operation.
///
/// Noise location ids must be unique within runtime compilation and DEM
/// generation. Tags are optional metadata used by hotspot aggregation.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseLocation {
    pub id: String,
    pub model: NoiseModel,
    pub rate: f64,
    pub qubits: Vec<usize>,
    pub tags: HashMap<String, TagValue>,
}

impl NoiseLocation {
    pub fn validate(&self) -> NpResult<()> {
        if !(0.0..=1.0).contains(&self.rate) {
            return Err(NpError::new(format!(
                "noise rate must be in [0, 1], got {}",
                self.rate
            )));
        }
        Ok(())
    }
}

/// Typed circuit operation understood by the Rust core.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Pauli {
        qubits: Vec<usize>,
        pauli: String,
    },
    Noise(NoiseLocation),
    Measure {
        qubit: usize,
        key: Option<String>,
        basis: String,
        noise: Option<NoiseLocation>,
    },
    MeasurePauli {
        qubits: Vec<usize>,
        pauli: String,
        key: Option<String>,
        noise: Option<NoiseLocation>,
    },
    Reset {
        qubit: usize,
        key: Option<String>,
        basis: String,
    },
    Detector {
        detector_id: Option<i64>,
        measurement_keys: Vec<String>,
        coords: Vec<f64>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_keys: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunOperation {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Noise(NoiseLocation),
    Measure {
        qubits: Vec<usize>,
        pauli: String,
        key: Option<String>,
        ideal: Expr,
        noise: Option<NoiseLocation>,
    },
    Reset {
        qubit: usize,
        key: Option<String>,
        basis: String,
        ideal: Expr,
    },
    Detector {
        detector_id: i64,
        measurement_keys: Vec<String>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_keys: Vec<String>,
    },
}

impl Operation {
    pub fn noise_locations(&self) -> Vec<&NoiseLocation> {
        match self {
            Self::Noise(location) => vec![location],
            Self::Measure {
                noise: Some(location),
                ..
            }
            | Self::MeasurePauli {
                noise: Some(location),
                ..
            } => vec![location],
            _ => Vec::new(),
        }
    }
}

/// A stabilizer-compatible circuit for forward sampling and DEM generation.
///
/// The Rust API is intentionally typed and Python-independent. It is currently
/// pre-1.0 and may change as the core package stabilizes.
#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    pub n_qubits: usize,
    pub operations: Vec<Operation>,
}

impl Circuit {
    pub fn noise_locations(&self) -> HashMap<String, NoiseLocation> {
        let mut locations = HashMap::new();
        for operation in &self.operations {
            for location in operation.noise_locations() {
                locations.insert(location.id.clone(), location.clone());
            }
        }
        locations
    }
}

/// Detector declaration used by detector error models.
#[derive(Debug, Clone, PartialEq)]
pub struct Detector {
    pub id: i64,
    pub measurement_keys: Vec<String>,
    pub coords: Vec<f64>,
}

/// Logical observable declaration.
///
/// Observables can be defined from measurement keys, from a final Pauli frame
/// projection, or from both.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalObservable {
    pub id: i64,
    pub measurement_keys: Vec<String>,
    pub pauli_qubits: Vec<usize>,
    pub pauli: String,
}

impl LogicalObservable {
    pub fn validate(&self) -> NpResult<()> {
        if self.pauli_qubits.is_empty() != self.pauli.is_empty() {
            return Err(NpError::new(
                "pauli_qubits and pauli must be supplied together",
            ));
        }
        if !self.pauli.is_empty() && self.pauli_qubits.len() != self.pauli.len() {
            return Err(NpError::new(
                "pauli_qubits and pauli must have the same length",
            ));
        }
        Ok(())
    }
}

/// Event label carried by a detector error edge.
#[derive(Debug, Clone, PartialEq)]
pub enum DemEvent {
    Pauli(String),
    Bool(bool),
}

/// One detector error model instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorErrorEdge {
    pub probability: f64,
    pub detectors: Vec<i64>,
    pub observables: Vec<i64>,
    pub location_id: String,
    pub event: DemEvent,
    pub tags: HashMap<String, TagValue>,
}

impl DetectorErrorEdge {
    pub fn validate(&self) -> NpResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(NpError::new(format!(
                "DEM edge probability must be in [0, 1], got {}",
                self.probability
            )));
        }
        Ok(())
    }
}

/// Typed detector error model.
///
/// This model can be produced by [`crate::DetectorErrorModelGenerator`] or
/// supplied directly to [`crate::DemBatchHotspotSimulator`].
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorErrorModel {
    pub detectors: Vec<Detector>,
    pub observables: Vec<LogicalObservable>,
    pub edges: Vec<DetectorErrorEdge>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemSamplerEdge {
    pub probability: f64,
    pub detectors: Vec<i64>,
    pub observables: Vec<i64>,
    pub location_id: String,
    pub event: DemEvent,
    pub tags: HashMap<String, TagValue>,
}

impl DemSamplerEdge {
    pub fn validate(&self) -> NpResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(NpError::new(format!(
                "DEM edge probability must be in [0, 1], got {}",
                self.probability
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemLocationGroup {
    pub location_id: String,
    pub edge_indices: Vec<usize>,
    pub total_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedBatch {
    pub shots: usize,
    pub all_mask_words: Vec<u64>,
    pub x_frame: Vec<Vec<u64>>,
    pub z_frame: Vec<Vec<u64>>,
    pub measurements: HashMap<String, Vec<u64>>,
    pub detectors: HashMap<i64, Vec<u64>>,
    pub observables: HashMap<i64, Vec<u64>>,
    pub noise_event_masks: HashMap<String, Vec<u64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemBatch {
    pub shots: usize,
    pub all_mask: Mask,
    pub detectors: HashMap<i64, Mask>,
    pub observables: HashMap<i64, Mask>,
    pub edge_event_masks: Vec<Mask>,
    pub loss_mask: Mask,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DetectorGraphKey {
    pub detectors: Vec<i64>,
    pub observables: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectorGraphEstimate {
    pub by_detector_edge: HashMap<DetectorGraphKey, f64>,
    pub signed_by_detector_edge: HashMap<DetectorGraphKey, f64>,
    pub by_detector: HashMap<i64, f64>,
    pub signed_by_detector: HashMap<i64, f64>,
    pub by_observable: HashMap<i64, f64>,
    pub signed_by_observable: HashMap<i64, f64>,
    pub by_location: HashMap<String, f64>,
    pub signed_by_location: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HotspotEstimate {
    pub shots: usize,
    pub mean_loss: f64,
    pub baseline: f64,
    pub sensitivities: HashMap<String, f64>,
    pub hotspots: HashMap<String, f64>,
    pub by_qubit: HashMap<usize, f64>,
    pub by_round: HashMap<TagValue, f64>,
    pub by_gate: HashMap<TagValue, f64>,
    pub by_operation: HashMap<TagValue, f64>,
    pub top_locations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemHotspotEstimate {
    pub shots: usize,
    pub mean_loss: f64,
    pub baseline: f64,
    pub edge_sensitivities: Vec<f64>,
    pub edge_hotspots: Vec<f64>,
    pub location_sensitivities: HashMap<String, f64>,
    pub location_hotspots: HashMap<String, f64>,
    pub by_detector: HashMap<i64, f64>,
    pub by_round: HashMap<TagValue, f64>,
    pub by_gate: HashMap<TagValue, f64>,
    pub by_operation: HashMap<TagValue, f64>,
    pub detector_graph: DetectorGraphEstimate,
    pub top_edges: Vec<usize>,
    pub top_locations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise_location(id: &str, rate: f64) -> NoiseLocation {
        NoiseLocation {
            id: id.to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate,
            qubits: vec![0],
            tags: HashMap::new(),
        }
    }

    #[test]
    fn circuit_collects_explicit_and_measurement_noise_locations() {
        let explicit = noise_location("after_h", 0.01);
        let measurement = noise_location("m0_noise", 0.02);
        let circuit = Circuit {
            n_qubits: 1,
            operations: vec![
                Operation::H(0),
                Operation::Noise(explicit.clone()),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m0".to_string()),
                    basis: "Z".to_string(),
                    noise: Some(measurement.clone()),
                },
            ],
        };

        let locations = circuit.noise_locations();

        assert_eq!(locations.len(), 2);
        assert_eq!(locations.get("after_h"), Some(&explicit));
        assert_eq!(locations.get("m0_noise"), Some(&measurement));
    }

    #[test]
    fn noise_location_rejects_invalid_rates() {
        let invalid = noise_location("bad", 1.5);

        let err = invalid.validate().unwrap_err();

        assert!(err.message().contains("noise rate must be in [0, 1]"));
    }

    #[test]
    fn detector_error_edge_rejects_invalid_probabilities() {
        let edge = DetectorErrorEdge {
            probability: -0.1,
            detectors: vec![0],
            observables: Vec::new(),
            location_id: "loc".to_string(),
            event: DemEvent::Bool(true),
            tags: HashMap::new(),
        };

        let err = edge.validate().unwrap_err();

        assert!(err
            .message()
            .contains("DEM edge probability must be in [0, 1]"));
    }

    #[test]
    fn logical_observable_requires_matching_pauli_support() {
        let observable = LogicalObservable {
            id: 0,
            measurement_keys: Vec::new(),
            pauli_qubits: vec![0, 1],
            pauli: "X".to_string(),
        };

        let err = observable.validate().unwrap_err();

        assert!(err
            .message()
            .contains("pauli_qubits and pauli must have the same length"));
    }
}

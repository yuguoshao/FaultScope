use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::Mask;

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

impl NoiseModel {
    /// Validate configuration that is intrinsic to this noise model.
    pub fn validate(&self) -> NpResult<()> {
        validate_noise_model(self, "noise model").map(|_| ())
    }
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
        validate_noise_rate(self.rate)?;
        let context = format!("noise location {:?}", self.id);
        validate_unique_qubits(&self.qubits, &context)?;
        validate_noise_shape(&self.model, &self.qubits, &context)
    }

    pub(crate) fn validate_for_n_qubits<C>(&self, n_qubits: usize, context: &C) -> NpResult<()>
    where
        C: fmt::Display + ?Sized,
    {
        validate_noise_parts(n_qubits, &self.model, self.rate, &self.qubits, context)
    }
}

/// Typed circuit operation understood by the Rust core.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// A semantic no-op used to preserve Stim scheduling boundaries.
    Tick,
    /// Add an offset to subsequent detector coordinates.
    ShiftCoords(Vec<f64>),
    /// Repeat a nested operation body `count` times.
    Repeat {
        count: usize,
        body: Vec<Operation>,
    },
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
    /// Combined measurement and reset with an automatically assigned record key.
    MeasureReset {
        qubit: usize,
        basis: String,
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
    /// Detector declaration using positive measurement-record lookbacks.
    DetectorRec {
        detector_id: Option<i64>,
        lookbacks: Vec<usize>,
        coords: Vec<f64>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_keys: Vec<String>,
    },
    /// Observable declaration using positive measurement-record lookbacks.
    ObservableIncludeRec {
        observable_id: i64,
        lookbacks: Vec<usize>,
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
            Self::Repeat { body, .. } => body.iter().flat_map(Operation::noise_locations).collect(),
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
    /// Validate all qubit-bearing operations without expanding repeat blocks.
    pub fn validate(&self) -> NpResult<()> {
        validate_operations(self.n_qubits, &self.operations)
    }

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
        if !self.pauli.is_empty() && self.pauli_qubits.len() != pauli_len(&self.pauli) {
            return Err(NpError::new(
                "pauli_qubits and pauli must have the same length",
            ));
        }
        let context = ObservableContext { id: self.id };
        validate_pauli_string(&self.pauli, &context)?;
        validate_unique_qubits(&self.pauli_qubits, &context)?;
        Ok(())
    }

    pub fn validate_for_n_qubits(&self, n_qubits: usize) -> NpResult<()> {
        self.validate()?;
        let context = ObservableContext { id: self.id };
        for qubit in &self.pauli_qubits {
            validate_qubit_index(n_qubits, *qubit, &context)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ObservableContext {
    id: i64,
}

impl fmt::Display for ObservableContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "logical observable {}", self.id)
    }
}

#[derive(Clone, Copy)]
struct OperationContext<'a> {
    kind: &'static str,
    path: &'a [usize],
}

impl fmt::Display for OperationContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at operation path ", self.kind)?;
        for (index, component) in self.path.iter().enumerate() {
            if index != 0 {
                f.write_str(".")?;
            }
            component.fmt(f)?;
        }
        Ok(())
    }
}

struct OperationNoiseContext<'a> {
    operation: OperationContext<'a>,
    location_id: &'a str,
}

impl fmt::Display for OperationNoiseContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} noise location {:?}",
            self.operation, self.location_id
        )
    }
}

pub(crate) fn validate_operations(n_qubits: usize, operations: &[Operation]) -> NpResult<()> {
    validate_operation_sequence(n_qubits, operations, &mut Vec::new())
}

fn validate_operation_sequence(
    n_qubits: usize,
    operations: &[Operation],
    path: &mut Vec<usize>,
) -> NpResult<()> {
    for (index, operation) in operations.iter().enumerate() {
        path.push(index);
        validate_operation(n_qubits, operation, path)?;
        path.pop();
    }
    Ok(())
}

fn validate_operation(
    n_qubits: usize,
    operation: &Operation,
    path: &mut Vec<usize>,
) -> NpResult<()> {
    match operation {
        Operation::Tick
        | Operation::ShiftCoords(_)
        | Operation::Detector { .. }
        | Operation::DetectorRec { .. }
        | Operation::ObservableInclude { .. }
        | Operation::ObservableIncludeRec { .. } => return Ok(()),
        Operation::Repeat { count, body } => {
            if *count == 0 {
                return Err(NpError::new("repeat count must be positive"));
            }
            return validate_operation_sequence(n_qubits, body, path);
        }
        _ => {}
    }

    let context = OperationContext {
        kind: operation_kind(operation),
        path,
    };
    match operation {
        Operation::H(qubit) | Operation::S(qubit) | Operation::SDag(qubit) => {
            validate_qubit_index(n_qubits, *qubit, &context)?;
        }
        Operation::Cx(control, target)
        | Operation::Cz(control, target)
        | Operation::Swap(control, target) => {
            validate_qubit_pair(n_qubits, *control, *target, &context)?;
        }
        Operation::Pauli { qubits, pauli } | Operation::MeasurePauli { qubits, pauli, .. } => {
            validate_pauli_targets(n_qubits, qubits, pauli, &context)?;
            if let Operation::MeasurePauli {
                noise: Some(location),
                ..
            } = operation
            {
                let noise_context = OperationNoiseContext {
                    operation: context,
                    location_id: &location.id,
                };
                location.validate_for_n_qubits(n_qubits, &noise_context)?;
            }
        }
        Operation::Noise(location) => {
            let noise_context = OperationNoiseContext {
                operation: context,
                location_id: &location.id,
            };
            location.validate_for_n_qubits(n_qubits, &noise_context)?;
        }
        Operation::Measure { qubit, noise, .. } => {
            validate_qubit_index(n_qubits, *qubit, &context)?;
            if let Some(location) = noise {
                let noise_context = OperationNoiseContext {
                    operation: context,
                    location_id: &location.id,
                };
                location.validate_for_n_qubits(n_qubits, &noise_context)?;
            }
        }
        Operation::MeasureReset { qubit, .. } | Operation::Reset { qubit, .. } => {
            validate_qubit_index(n_qubits, *qubit, &context)?;
        }
        Operation::Tick
        | Operation::ShiftCoords(_)
        | Operation::Repeat { .. }
        | Operation::Detector { .. }
        | Operation::DetectorRec { .. }
        | Operation::ObservableInclude { .. }
        | Operation::ObservableIncludeRec { .. } => unreachable!("handled above"),
    }
    Ok(())
}

fn operation_kind(operation: &Operation) -> &'static str {
    match operation {
        Operation::Tick => "Tick",
        Operation::ShiftCoords(_) => "ShiftCoords",
        Operation::Repeat { .. } => "Repeat",
        Operation::H(_) => "H",
        Operation::S(_) => "S",
        Operation::SDag(_) => "SDag",
        Operation::Cx(_, _) => "CX",
        Operation::Cz(_, _) => "CZ",
        Operation::Swap(_, _) => "SWAP",
        Operation::Pauli { .. } => "Pauli",
        Operation::Noise(_) => "Noise",
        Operation::Measure { .. } => "Measure",
        Operation::MeasurePauli { .. } => "MeasurePauli",
        Operation::MeasureReset { .. } => "MeasureReset",
        Operation::Reset { .. } => "Reset",
        Operation::Detector { .. } => "Detector",
        Operation::DetectorRec { .. } => "DetectorRec",
        Operation::ObservableInclude { .. } => "ObservableInclude",
        Operation::ObservableIncludeRec { .. } => "ObservableIncludeRec",
    }
}

pub(crate) fn validate_qubit_index<C>(n_qubits: usize, qubit: usize, context: &C) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    if qubit >= n_qubits {
        return Err(NpError::new(format!(
            "{context} targets qubit {qubit}, but only {n_qubits} qubits are available"
        )));
    }
    Ok(())
}

pub(crate) fn validate_unique_qubits<C>(qubits: &[usize], context: &C) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    // Most gates and noise events touch very few qubits. A short linear scan
    // avoids allocating a hash table in that overwhelmingly common case.
    const LINEAR_SCAN_LIMIT: usize = 8;
    if qubits.len() <= LINEAR_SCAN_LIMIT {
        for (index, qubit) in qubits.iter().enumerate() {
            if qubits[..index].contains(qubit) {
                return Err(NpError::new(format!(
                    "{context} requires unique qubit targets; duplicate qubit {qubit}"
                )));
            }
        }
        return Ok(());
    }

    let mut seen = HashSet::with_capacity(qubits.len());
    for qubit in qubits {
        if !seen.insert(*qubit) {
            return Err(NpError::new(format!(
                "{context} requires unique qubit targets; duplicate qubit {qubit}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_qubit_targets<C>(
    n_qubits: usize,
    qubits: &[usize],
    context: &C,
) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    for qubit in qubits {
        validate_qubit_index(n_qubits, *qubit, context)?;
    }
    validate_unique_qubits(qubits, context)
}

pub(crate) fn validate_qubit_pair<C>(
    n_qubits: usize,
    left: usize,
    right: usize,
    context: &C,
) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    validate_qubit_index(n_qubits, left, context)?;
    validate_qubit_index(n_qubits, right, context)?;
    if left == right {
        return Err(NpError::new(format!(
            "{context} requires distinct qubit targets; duplicate qubit {left}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_pauli_string<C>(pauli: &str, context: &C) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    if let Some(local) = pauli
        .chars()
        .find(|local| !matches!(local, 'I' | 'X' | 'Y' | 'Z'))
    {
        return Err(NpError::new(format!(
            "{context} contains unsupported Pauli {local:?}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_pauli_targets<C>(
    n_qubits: usize,
    qubits: &[usize],
    pauli: &str,
    context: &C,
) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    if qubits.len() != pauli_len(pauli) {
        return Err(NpError::new(format!(
            "{context} qubit targets and Pauli must have the same length"
        )));
    }
    validate_pauli_string(pauli, context)?;
    validate_qubit_targets(n_qubits, qubits, context)
}

pub(crate) fn validate_noise_parts<C>(
    n_qubits: usize,
    model: &NoiseModel,
    rate: f64,
    qubits: &[usize],
    context: &C,
) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    validate_noise_rate(rate)?;
    validate_qubit_targets(n_qubits, qubits, context)?;
    validate_noise_shape(model, qubits, context)
}

fn validate_noise_rate(rate: f64) -> NpResult<()> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(NpError::new(format!(
            "noise rate must be in [0, 1], got {rate}"
        )));
    }
    Ok(())
}

fn validate_noise_shape<C>(model: &NoiseModel, qubits: &[usize], context: &C) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    let event_length = validate_noise_model(model, context)?;
    match model {
        NoiseModel::MeasurementBitFlip | NoiseModel::SingleQubitDepolarizing => {
            if qubits.len() != 1 {
                return Err(NpError::new(format!(
                    "{context} requires exactly one qubit target, got {}",
                    qubits.len()
                )));
            }
        }
        NoiseModel::TwoQubitDepolarizing => {
            if qubits.len() != 2 {
                return Err(NpError::new(format!(
                    "{context} requires exactly two qubit targets, got {}",
                    qubits.len()
                )));
            }
        }
        NoiseModel::BernoulliPauli(_) => {
            if qubits.len() != event_length {
                return Err(NpError::new(format!(
                    "{context} qubit targets and Pauli event must have the same length"
                )));
            }
        }
        NoiseModel::PauliChannel(_) => {
            if qubits.len() != event_length {
                return Err(NpError::new(format!(
                    "{context} qubit targets and PauliChannel events must have the same length"
                )));
            }
        }
    }
    Ok(())
}

fn validate_noise_model<C>(model: &NoiseModel, context: &C) -> NpResult<usize>
where
    C: fmt::Display + ?Sized,
{
    match model {
        NoiseModel::MeasurementBitFlip | NoiseModel::SingleQubitDepolarizing => Ok(1),
        NoiseModel::TwoQubitDepolarizing => Ok(2),
        NoiseModel::BernoulliPauli(pauli) => {
            validate_non_identity_pauli_event(pauli, context)?;
            Ok(pauli_len(pauli))
        }
        NoiseModel::PauliChannel(weights) => validate_pauli_channel_weights(weights, context),
    }
}

fn validate_non_identity_pauli_event<C>(pauli: &str, context: &C) -> NpResult<()>
where
    C: fmt::Display + ?Sized,
{
    if pauli.is_empty() {
        return Err(NpError::new(format!(
            "{context} Pauli event must not be empty"
        )));
    }
    validate_pauli_string(pauli, context)?;
    if pauli.chars().all(|local| local == 'I') {
        return Err(NpError::new(format!(
            "{context} Pauli event must be non-identity"
        )));
    }
    Ok(())
}

pub(crate) fn validate_pauli_channel_weights<C>(
    weights: &[(String, f64)],
    context: &C,
) -> NpResult<usize>
where
    C: fmt::Display + ?Sized,
{
    if weights.is_empty() {
        return Err(NpError::new(format!(
            "{context} PauliChannel requires at least one non-identity event"
        )));
    }

    let mut total = 0.0;
    let mut event_length = None;
    for (event, weight) in weights {
        validate_non_identity_pauli_event(event, context)?;
        if !weight.is_finite() {
            return Err(NpError::new(format!(
                "{context} PauliChannel weights must be finite"
            )));
        }
        if *weight < 0.0 {
            return Err(NpError::new(format!(
                "{context} PauliChannel weights must be non-negative"
            )));
        }
        match event_length {
            None => event_length = Some(pauli_len(event)),
            Some(length) if pauli_len(event) != length => {
                return Err(NpError::new(format!(
                    "{context} PauliChannel events must have the same length"
                )));
            }
            _ => {}
        }
        total += *weight;
        if !total.is_finite() {
            return Err(NpError::new(format!(
                "{context} PauliChannel total weight must be finite"
            )));
        }
    }

    if total <= 0.0 {
        return Err(NpError::new(format!(
            "{context} PauliChannel weights must have positive total weight"
        )));
    }
    Ok(event_length.expect("non-empty PauliChannel has an event length"))
}

#[inline]
fn pauli_len(pauli: &str) -> usize {
    if pauli.is_ascii() {
        pauli.len()
    } else {
        pauli.chars().count()
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
/// supplied directly to [`crate::DemHotspotEstimator`].
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorErrorModel {
    pub detectors: Vec<Detector>,
    pub observables: Vec<LogicalObservable>,
    pub edges: Vec<DetectorErrorEdge>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemLocationGroup {
    pub location_id: String,
    pub edge_indices: Vec<usize>,
    pub total_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemBatch {
    shots: usize,
    all_mask: Mask,
    pub detectors: HashMap<i64, Mask>,
    pub observables: HashMap<i64, Mask>,
    edge_event_masks: Vec<Mask>,
    pub loss_mask: Mask,
    edge_events_recorded: bool,
    hotspot_layout_identity: Arc<()>,
}

impl DemBatch {
    pub(crate) fn new(
        shots: usize,
        all_mask: Mask,
        detectors: HashMap<i64, Mask>,
        observables: HashMap<i64, Mask>,
        edge_event_masks: Option<Vec<Mask>>,
        loss_mask: Mask,
        hotspot_layout_identity: Arc<()>,
    ) -> Self {
        let edge_events_recorded = edge_event_masks.is_some();
        Self {
            shots,
            all_mask,
            detectors,
            observables,
            edge_event_masks: edge_event_masks.unwrap_or_default(),
            loss_mask,
            edge_events_recorded,
            hotspot_layout_identity,
        }
    }

    pub fn shots(&self) -> usize {
        self.shots
    }

    pub fn all_mask(&self) -> &Mask {
        &self.all_mask
    }

    pub fn edge_event_masks(&self) -> &[Mask] {
        &self.edge_event_masks
    }

    pub fn records_edge_events(&self) -> bool {
        self.edge_events_recorded
    }

    pub(crate) fn matches_hotspot_layout(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.hotspot_layout_identity, identity)
    }
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
    fn noise_models_reject_invalid_pauli_configurations() {
        let invalid_models = vec![
            NoiseModel::BernoulliPauli(String::new()),
            NoiseModel::BernoulliPauli("II".to_string()),
            NoiseModel::BernoulliPauli("A".to_string()),
            NoiseModel::PauliChannel(Vec::new()),
            NoiseModel::PauliChannel(vec![(String::new(), 1.0)]),
            NoiseModel::PauliChannel(vec![("I".to_string(), 1.0)]),
            NoiseModel::PauliChannel(vec![("X".to_string(), f64::NAN)]),
            NoiseModel::PauliChannel(vec![("X".to_string(), f64::INFINITY)]),
            NoiseModel::PauliChannel(vec![("X".to_string(), f64::NEG_INFINITY)]),
            NoiseModel::PauliChannel(vec![
                ("X".to_string(), f64::MAX),
                ("Z".to_string(), f64::MAX),
            ]),
            NoiseModel::PauliChannel(vec![("X".to_string(), -1.0)]),
            NoiseModel::PauliChannel(vec![("X".to_string(), 0.0)]),
            NoiseModel::PauliChannel(vec![("X".to_string(), 1.0), ("ZZ".to_string(), 1.0)]),
        ];

        for model in invalid_models {
            assert!(
                model.validate().is_err(),
                "accepted invalid model {model:?}"
            );
        }

        NoiseModel::BernoulliPauli("XI".to_string())
            .validate()
            .unwrap();
        NoiseModel::PauliChannel(vec![("IX".to_string(), 0.0), ("XI".to_string(), 2.0)])
            .validate()
            .unwrap();
    }

    #[test]
    fn noise_location_rejects_invalid_model_at_zero_rate() {
        let location = NoiseLocation {
            id: "bad".to_string(),
            model: NoiseModel::BernoulliPauli("A".to_string()),
            rate: 0.0,
            qubits: vec![0],
            tags: HashMap::new(),
        };

        assert!(location.validate().is_err());
        let circuit = Circuit {
            n_qubits: 1,
            operations: vec![Operation::Noise(location)],
        };
        assert!(circuit.validate().is_err());
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
    fn circuit_recursively_rejects_out_of_range_operation_targets() {
        let mut invalid_noise = noise_location("bad_target", 0.1);
        invalid_noise.qubits = vec![1];
        let cases = vec![
            Operation::H(1),
            Operation::S(1),
            Operation::SDag(1),
            Operation::Cx(0, 1),
            Operation::Cz(0, 1),
            Operation::Swap(0, 1),
            Operation::Pauli {
                qubits: vec![1],
                pauli: "X".to_string(),
            },
            Operation::Noise(invalid_noise.clone()),
            Operation::Measure {
                qubit: 1,
                key: None,
                basis: "Z".to_string(),
                noise: None,
            },
            Operation::Measure {
                qubit: 0,
                key: None,
                basis: "Z".to_string(),
                noise: Some(invalid_noise.clone()),
            },
            Operation::MeasurePauli {
                qubits: vec![1],
                pauli: "Z".to_string(),
                key: None,
                noise: None,
            },
            Operation::MeasurePauli {
                qubits: vec![0],
                pauli: "Z".to_string(),
                key: None,
                noise: Some(invalid_noise),
            },
            Operation::MeasureReset {
                qubit: 1,
                basis: "Z".to_string(),
            },
            Operation::Reset {
                qubit: 1,
                key: None,
                basis: "Z".to_string(),
            },
        ];

        for operation in cases {
            let circuit = Circuit {
                n_qubits: 1,
                operations: vec![Operation::Repeat {
                    count: 2,
                    body: vec![operation],
                }],
            };
            let err = circuit.validate().unwrap_err();
            assert!(err.message().contains("targets qubit 1"), "{err}");
            assert!(err.message().contains("operation path 0.0"), "{err}");
        }
    }

    #[test]
    fn circuit_rejects_duplicate_targets_for_every_multi_target_shape() {
        let duplicate_noise = NoiseLocation {
            id: "two".to_string(),
            model: NoiseModel::TwoQubitDepolarizing,
            rate: 0.1,
            qubits: vec![0, 0],
            tags: HashMap::new(),
        };
        let cases = vec![
            Operation::Cx(0, 0),
            Operation::Cz(0, 0),
            Operation::Swap(0, 0),
            Operation::Pauli {
                qubits: vec![0, 0],
                pauli: "XZ".to_string(),
            },
            Operation::MeasurePauli {
                qubits: vec![0, 0],
                pauli: "ZZ".to_string(),
                key: None,
                noise: None,
            },
            Operation::Noise(duplicate_noise),
        ];

        for operation in cases {
            let err = Circuit {
                n_qubits: 1,
                operations: vec![operation],
            }
            .validate()
            .unwrap_err();
            assert!(err.message().contains("duplicate qubit 0"), "{err}");
        }
    }

    #[test]
    fn circuit_validates_pauli_width_characters_and_noise_target_arity() {
        let invalid_operations = vec![
            Operation::Pauli {
                qubits: vec![0],
                pauli: "XX".to_string(),
            },
            Operation::MeasurePauli {
                qubits: vec![0],
                pauli: "Q".to_string(),
                key: None,
                noise: None,
            },
            Operation::Noise(NoiseLocation {
                id: "single".to_string(),
                model: NoiseModel::SingleQubitDepolarizing,
                rate: 0.1,
                qubits: vec![0, 1],
                tags: HashMap::new(),
            }),
            Operation::Noise(NoiseLocation {
                id: "two".to_string(),
                model: NoiseModel::TwoQubitDepolarizing,
                rate: 0.1,
                qubits: vec![0],
                tags: HashMap::new(),
            }),
            Operation::Noise(NoiseLocation {
                id: "channel".to_string(),
                model: NoiseModel::PauliChannel(vec![("XX".to_string(), 1.0)]),
                rate: 0.1,
                qubits: vec![0],
                tags: HashMap::new(),
            }),
        ];

        for operation in invalid_operations {
            assert!(Circuit {
                n_qubits: 2,
                operations: vec![operation],
            }
            .validate()
            .is_err());
        }
    }

    #[test]
    fn circuit_accepts_valid_targets_at_the_upper_boundary() {
        let circuit = Circuit {
            n_qubits: 2,
            operations: vec![
                Operation::H(1),
                Operation::Cx(0, 1),
                Operation::Cz(1, 0),
                Operation::Swap(0, 1),
                Operation::Pauli {
                    qubits: vec![0, 1],
                    pauli: "IY".to_string(),
                },
                Operation::Noise(NoiseLocation {
                    id: "two".to_string(),
                    model: NoiseModel::TwoQubitDepolarizing,
                    rate: 0.1,
                    qubits: vec![0, 1],
                    tags: HashMap::new(),
                }),
            ],
        };

        circuit.validate().unwrap();
        Circuit {
            n_qubits: 0,
            operations: Vec::new(),
        }
        .validate()
        .unwrap();
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

    #[test]
    fn logical_observable_rejects_out_of_range_and_duplicate_targets() {
        let out_of_range = LogicalObservable {
            id: 4,
            measurement_keys: Vec::new(),
            pauli_qubits: vec![1],
            pauli: "Z".to_string(),
        };
        let err = out_of_range.validate_for_n_qubits(1).unwrap_err();
        assert!(err.message().contains("logical observable 4"));
        assert!(err.message().contains("targets qubit 1"));

        let duplicate = LogicalObservable {
            id: 5,
            measurement_keys: Vec::new(),
            pauli_qubits: vec![0, 0],
            pauli: "XZ".to_string(),
        };
        let err = duplicate.validate_for_n_qubits(1).unwrap_err();
        assert!(err.message().contains("duplicate qubit 0"));
    }
}

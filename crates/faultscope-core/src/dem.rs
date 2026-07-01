use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{
    sparse_pauli_to_xz, word_count, Circuit, ConcreteStabilizer, DemEvent, DemSamplerEdge,
    Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable, Mask, NoiseLocation,
    NoiseModel, NpError, NpResult, Operation,
};

/// Detector error model generator based on single-error propagation.
///
/// The generator is pure Rust core logic. It can infer detector and observable
/// declarations from circuit operations or use declarations supplied by the
/// caller.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorErrorModelGenerator {
    pub circuit: Arc<Circuit>,
    pub detectors: Vec<Detector>,
    pub observables: Vec<LogicalObservable>,
    event_plan: Arc<DemEventPlan>,
    measurement_plan: Option<DemMeasurementPlan>,
}

impl DetectorErrorModelGenerator {
    /// Create a generator for `circuit`.
    ///
    /// If `detectors` or `observables` are `None`, declarations are inferred
    /// from `Operation::Detector` and `Operation::ObservableInclude` entries in
    /// the circuit.
    pub fn new(
        circuit: Circuit,
        detectors: Option<Vec<Detector>>,
        observables: Option<Vec<LogicalObservable>>,
    ) -> NpResult<Self> {
        let detectors = detectors.unwrap_or_else(|| detectors_from_circuit(&circuit));
        let observables = observables.unwrap_or_else(|| observables_from_circuit(&circuit));
        let event_plan = collect_dem_event_plan(&circuit.operations)?;
        Self::new_with_event_plan(circuit, detectors, observables, event_plan)
    }

    /// Create a generator using a precompiled event plan for this circuit.
    pub fn new_with_event_plan(
        circuit: Circuit,
        detectors: Vec<Detector>,
        observables: Vec<LogicalObservable>,
        event_plan: DemEventPlan,
    ) -> NpResult<Self> {
        Self::new_with_shared_event_plan(
            Arc::new(circuit),
            detectors,
            observables,
            Arc::new(event_plan),
        )
    }

    /// Create a generator using shared circuit and event-plan ownership.
    pub fn new_with_shared_event_plan(
        circuit: Arc<Circuit>,
        detectors: Vec<Detector>,
        observables: Vec<LogicalObservable>,
        event_plan: Arc<DemEventPlan>,
    ) -> NpResult<Self> {
        validate_detector_ids(&detectors)?;
        validate_observable_ids(&observables)?;
        for observable in &observables {
            observable.validate()?;
        }
        let measurement_plan =
            if supports_product_reference_fast_path(circuit.n_qubits, &circuit.operations) {
                Some(compile_dem_measurement_plan(
                    &circuit.operations,
                    &detectors,
                    &observables,
                )?)
            } else {
                None
            };
        Ok(Self {
            circuit,
            detectors,
            observables,
            event_plan,
            measurement_plan,
        })
    }

    /// Generate a typed detector error model.
    pub fn generate(&self) -> NpResult<DetectorErrorModel> {
        Ok(self.generate_lazy()?.materialize())
    }

    /// Generate detector/observable edge structure while deferring metadata clones.
    pub fn generate_lazy(&self) -> NpResult<LazyDetectorErrorModel> {
        let edges = self.generate_edge_refs()?;
        Ok(LazyDetectorErrorModel {
            detectors: self.detectors.clone(),
            observables: self.observables.clone(),
            event_plan: self.event_plan.clone(),
            edges,
        })
    }

    fn generate_edges(&self) -> NpResult<Vec<GeneratedDemEdge>> {
        if let Some(measurement_plan) = &self.measurement_plan {
            return generate_indexed_product_dem_edges_from_plan(
                self.circuit.n_qubits,
                &self.circuit.operations,
                measurement_plan,
                &self.event_plan,
            );
        }
        generate_dem_edges_from_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.detectors,
            &self.observables,
            &self.event_plan,
        )
    }

    fn generate_edge_refs(&self) -> NpResult<Vec<GeneratedDemEdgeRef>> {
        if let Some(measurement_plan) = &self.measurement_plan {
            return generate_indexed_product_dem_edge_refs_from_plan(
                self.circuit.n_qubits,
                &self.circuit.operations,
                measurement_plan,
                &self.event_plan,
            );
        }
        generate_dem_edge_refs_from_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.detectors,
            &self.observables,
            &self.event_plan,
        )
    }

    /// Generate only the edge metadata required by the native DEM sampler.
    pub fn generate_sampler_edges(&self) -> NpResult<Vec<DemSamplerEdge>> {
        let generated_edges = self.generate_edges()?;
        Ok(generated_edges_to_sampler_edges(generated_edges))
    }

    /// Generate sampling-only edges without location/event/tag metadata.
    pub fn generate_sampling_edges(&self) -> NpResult<Vec<DemSamplerEdge>> {
        if let Some(measurement_plan) = &self.measurement_plan {
            return generate_indexed_product_sampling_edges_from_plan(
                self.circuit.n_qubits,
                &self.circuit.operations,
                measurement_plan,
                &self.event_plan,
            );
        }
        generate_sampling_edges_from_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.detectors,
            &self.observables,
            &self.event_plan,
        )
    }
}

fn generated_edges_to_sampler_edges(generated_edges: Vec<GeneratedDemEdge>) -> Vec<DemSamplerEdge> {
    generated_edges
        .into_iter()
        .map(|edge| DemSamplerEdge {
            probability: edge.probability,
            detectors: edge.detectors,
            observables: edge.observables,
            tags: edge.tags,
            location_id: edge.location_id,
            event: edge.event,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemEventPlan {
    fault_events: Vec<DemFaultEvent>,
    fault_events_by_op: Vec<Vec<usize>>,
}

pub fn collect_dem_event_plan(operations: &[Operation]) -> NpResult<DemEventPlan> {
    let (fault_events, fault_events_by_op) = collect_dem_fault_events(operations)?;
    Ok(DemEventPlan {
        fault_events,
        fault_events_by_op,
    })
}

pub fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
) -> NpResult<Vec<GeneratedDemEdge>> {
    let event_plan = collect_dem_event_plan(operations)?;
    generate_dem_edges_from_plan(n_qubits, operations, detectors, observables, &event_plan)
}

pub fn generate_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdge>> {
    if supports_product_reference_fast_path(n_qubits, operations) {
        let measurement_plan = compile_dem_measurement_plan(operations, detectors, observables)?;
        return generate_indexed_product_dem_edges_from_plan(
            n_qubits,
            operations,
            &measurement_plan,
            event_plan,
        );
    }
    let fault_events = &event_plan.fault_events;
    let fault_events_by_op = &event_plan.fault_events_by_op;
    let mut state = DemFaultPropagationState::new(n_qubits, fault_events.len());
    for (op_index, operation) in operations.iter().enumerate() {
        apply_fault_propagation_operation(
            operation,
            op_index,
            fault_events,
            fault_events_by_op,
            &mut state,
        )?;
    }
    let detector_flip_masks =
        evaluate_detector_flip_masks(&state.measurement_flip_masks, detectors, state.event_words)?;
    let observable_flip_masks = evaluate_observable_flip_masks(
        &state.measurement_flip_masks,
        &state.x_frame,
        &state.z_frame,
        observables,
        state.event_words,
    )?;
    Ok(assemble_generated_dem_edges(
        fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn generate_dem_edge_refs_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdgeRef>> {
    if supports_product_reference_fast_path(n_qubits, operations) {
        let measurement_plan = compile_dem_measurement_plan(operations, detectors, observables)?;
        return generate_indexed_product_dem_edge_refs_from_plan(
            n_qubits,
            operations,
            &measurement_plan,
            event_plan,
        );
    }
    let fault_events = &event_plan.fault_events;
    let fault_events_by_op = &event_plan.fault_events_by_op;
    let mut state = DemFaultPropagationState::new(n_qubits, fault_events.len());
    for (op_index, operation) in operations.iter().enumerate() {
        apply_fault_propagation_operation(
            operation,
            op_index,
            fault_events,
            fault_events_by_op,
            &mut state,
        )?;
    }
    let detector_flip_masks =
        evaluate_detector_flip_masks(&state.measurement_flip_masks, detectors, state.event_words)?;
    let observable_flip_masks = evaluate_observable_flip_masks(
        &state.measurement_flip_masks,
        &state.x_frame,
        &state.z_frame,
        observables,
        state.event_words,
    )?;
    Ok(assemble_dem_edge_refs(
        fault_events.len(),
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn generate_sampling_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
    event_plan: &DemEventPlan,
) -> NpResult<Vec<DemSamplerEdge>> {
    if supports_product_reference_fast_path(n_qubits, operations) {
        let measurement_plan = compile_dem_measurement_plan(operations, detectors, observables)?;
        return generate_indexed_product_sampling_edges_from_plan(
            n_qubits,
            operations,
            &measurement_plan,
            event_plan,
        );
    }
    let fault_events = &event_plan.fault_events;
    let fault_events_by_op = &event_plan.fault_events_by_op;
    let mut state = DemFaultPropagationState::new(n_qubits, fault_events.len());
    for (op_index, operation) in operations.iter().enumerate() {
        apply_fault_propagation_operation(
            operation,
            op_index,
            fault_events,
            fault_events_by_op,
            &mut state,
        )?;
    }
    let detector_flip_masks =
        evaluate_detector_flip_masks(&state.measurement_flip_masks, detectors, state.event_words)?;
    let observable_flip_masks = evaluate_observable_flip_masks(
        &state.measurement_flip_masks,
        &state.x_frame,
        &state.z_frame,
        observables,
        state.event_words,
    )?;
    Ok(assemble_sampling_edges(
        fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn assemble_generated_dem_edges(
    fault_events: &[DemFaultEvent],
    detector_flip_masks: HashMap<i64, Mask>,
    observable_flip_masks: HashMap<i64, Mask>,
) -> Vec<GeneratedDemEdge> {
    let detector_flip_masks = sorted_flip_masks(detector_flip_masks);
    let observable_flip_masks = sorted_flip_masks(observable_flip_masks);
    assemble_generated_dem_edges_from_flip_masks(
        fault_events,
        detector_flip_masks,
        observable_flip_masks,
    )
}

fn assemble_generated_dem_edges_from_flip_masks(
    fault_events: &[DemFaultEvent],
    detector_flip_masks: Vec<(i64, Mask)>,
    observable_flip_masks: Vec<(i64, Mask)>,
) -> Vec<GeneratedDemEdge> {
    let mut detector_flips_by_event =
        flip_ids_by_fault_event(&detector_flip_masks, fault_events.len());
    let mut observable_flips_by_event =
        flip_ids_by_fault_event(&observable_flip_masks, fault_events.len());
    let mut edges = Vec::new();

    for (event_index, event) in fault_events.iter().enumerate() {
        let detector_flips = std::mem::take(&mut detector_flips_by_event[event_index]);
        let observable_flips = std::mem::take(&mut observable_flips_by_event[event_index]);
        if detector_flips.is_empty() && observable_flips.is_empty() {
            continue;
        }
        edges.push(GeneratedDemEdge {
            probability: event.probability,
            detectors: detector_flips,
            observables: observable_flips,
            location_id: event.location_id.clone(),
            event: event.event.clone(),
            tags: event.tags.clone(),
        });
    }
    edges
}

fn assemble_generated_dem_edges_from_flat_flip_masks(
    fault_events: &[DemFaultEvent],
    detector_flip_masks: Vec<(i64, Vec<u64>)>,
    observable_flip_masks: Vec<(i64, Vec<u64>)>,
) -> Vec<GeneratedDemEdge> {
    let mut detector_flips_by_event =
        flat_flip_ids_by_fault_event(&detector_flip_masks, fault_events.len());
    let mut observable_flips_by_event =
        flat_flip_ids_by_fault_event(&observable_flip_masks, fault_events.len());
    let mut edges = Vec::new();

    for (event_index, event) in fault_events.iter().enumerate() {
        let detector_flips = std::mem::take(&mut detector_flips_by_event[event_index]);
        let observable_flips = std::mem::take(&mut observable_flips_by_event[event_index]);
        if detector_flips.is_empty() && observable_flips.is_empty() {
            continue;
        }
        edges.push(GeneratedDemEdge {
            probability: event.probability,
            detectors: detector_flips,
            observables: observable_flips,
            location_id: event.location_id.clone(),
            event: event.event.clone(),
            tags: event.tags.clone(),
        });
    }
    edges
}

fn assemble_dem_edge_refs(
    event_count: usize,
    detector_flip_masks: HashMap<i64, Mask>,
    observable_flip_masks: HashMap<i64, Mask>,
) -> Vec<GeneratedDemEdgeRef> {
    let detector_flip_masks = sorted_flip_masks(detector_flip_masks);
    let observable_flip_masks = sorted_flip_masks(observable_flip_masks);
    assemble_dem_edge_refs_from_flip_masks(event_count, detector_flip_masks, observable_flip_masks)
}

fn assemble_dem_edge_refs_from_flip_masks(
    event_count: usize,
    detector_flip_masks: Vec<(i64, Mask)>,
    observable_flip_masks: Vec<(i64, Mask)>,
) -> Vec<GeneratedDemEdgeRef> {
    let mut detector_flips_by_event = flip_ids_by_fault_event(&detector_flip_masks, event_count);
    let mut observable_flips_by_event =
        flip_ids_by_fault_event(&observable_flip_masks, event_count);
    let mut edges = Vec::new();

    for event_index in 0..event_count {
        let detector_flips = std::mem::take(&mut detector_flips_by_event[event_index]);
        let observable_flips = std::mem::take(&mut observable_flips_by_event[event_index]);
        if detector_flips.is_empty() && observable_flips.is_empty() {
            continue;
        }
        edges.push(GeneratedDemEdgeRef {
            event_index,
            detectors: detector_flips,
            observables: observable_flips,
        });
    }
    edges
}

fn assemble_dem_edge_refs_from_flat_flip_masks(
    event_count: usize,
    detector_flip_masks: Vec<(i64, Vec<u64>)>,
    observable_flip_masks: Vec<(i64, Vec<u64>)>,
) -> Vec<GeneratedDemEdgeRef> {
    let mut detector_flips_by_event =
        flat_flip_ids_by_fault_event(&detector_flip_masks, event_count);
    let mut observable_flips_by_event =
        flat_flip_ids_by_fault_event(&observable_flip_masks, event_count);
    let mut edges = Vec::new();

    for event_index in 0..event_count {
        let detector_flips = std::mem::take(&mut detector_flips_by_event[event_index]);
        let observable_flips = std::mem::take(&mut observable_flips_by_event[event_index]);
        if detector_flips.is_empty() && observable_flips.is_empty() {
            continue;
        }
        edges.push(GeneratedDemEdgeRef {
            event_index,
            detectors: detector_flips,
            observables: observable_flips,
        });
    }
    edges
}

fn assemble_sampling_edges(
    fault_events: &[DemFaultEvent],
    detector_flip_masks: HashMap<i64, Mask>,
    observable_flip_masks: HashMap<i64, Mask>,
) -> Vec<DemSamplerEdge> {
    let detector_flip_masks = sorted_flip_masks(detector_flip_masks);
    let observable_flip_masks = sorted_flip_masks(observable_flip_masks);
    assemble_sampling_edges_from_flip_masks(
        fault_events,
        detector_flip_masks,
        observable_flip_masks,
    )
}

fn assemble_sampling_edges_from_flip_masks(
    fault_events: &[DemFaultEvent],
    detector_flip_masks: Vec<(i64, Mask)>,
    observable_flip_masks: Vec<(i64, Mask)>,
) -> Vec<DemSamplerEdge> {
    let mut detector_flips_by_event =
        flip_ids_by_fault_event(&detector_flip_masks, fault_events.len());
    let mut observable_flips_by_event =
        flip_ids_by_fault_event(&observable_flip_masks, fault_events.len());
    let mut edges = Vec::new();

    for (event_index, event) in fault_events.iter().enumerate() {
        let detector_flips = std::mem::take(&mut detector_flips_by_event[event_index]);
        let observable_flips = std::mem::take(&mut observable_flips_by_event[event_index]);
        if detector_flips.is_empty() && observable_flips.is_empty() {
            continue;
        }
        edges.push(DemSamplerEdge {
            probability: event.probability,
            detectors: detector_flips,
            observables: observable_flips,
            location_id: String::new(),
            event: DemEvent::Bool(false),
            tags: HashMap::new(),
        });
    }
    edges
}

fn assemble_sampling_edges_from_flat_flip_masks(
    fault_events: &[DemFaultEvent],
    detector_flip_masks: Vec<(i64, Vec<u64>)>,
    observable_flip_masks: Vec<(i64, Vec<u64>)>,
) -> Vec<DemSamplerEdge> {
    let mut detector_flips_by_event =
        flat_flip_ids_by_fault_event(&detector_flip_masks, fault_events.len());
    let mut observable_flips_by_event =
        flat_flip_ids_by_fault_event(&observable_flip_masks, fault_events.len());
    let mut edges = Vec::new();

    for (event_index, event) in fault_events.iter().enumerate() {
        let detector_flips = std::mem::take(&mut detector_flips_by_event[event_index]);
        let observable_flips = std::mem::take(&mut observable_flips_by_event[event_index]);
        if detector_flips.is_empty() && observable_flips.is_empty() {
            continue;
        }
        edges.push(DemSamplerEdge {
            probability: event.probability,
            detectors: detector_flips,
            observables: observable_flips,
            location_id: String::new(),
            event: DemEvent::Bool(false),
            tags: HashMap::new(),
        });
    }
    edges
}

/// Infer detector declarations from detector operations in a circuit.
pub fn detectors_from_circuit(circuit: &Circuit) -> Vec<Detector> {
    let mut detectors = Vec::new();
    for operation in &circuit.operations {
        let Operation::Detector {
            detector_id,
            measurement_keys,
            coords,
        } = operation
        else {
            continue;
        };
        detectors.push(Detector {
            id: detector_id.unwrap_or(detectors.len() as i64),
            measurement_keys: measurement_keys.clone(),
            coords: coords.clone(),
        });
    }
    detectors
}

/// Infer logical observable declarations from observable include operations.
pub fn observables_from_circuit(circuit: &Circuit) -> Vec<LogicalObservable> {
    let mut keys_by_id = HashMap::<i64, Vec<String>>::new();
    for operation in &circuit.operations {
        let Operation::ObservableInclude {
            observable_id,
            measurement_keys,
        } = operation
        else {
            continue;
        };
        keys_by_id
            .entry(*observable_id)
            .or_default()
            .extend(measurement_keys.clone());
    }
    let mut observable_ids = keys_by_id.keys().copied().collect::<Vec<_>>();
    observable_ids.sort_unstable();
    observable_ids
        .into_iter()
        .map(|observable_id| LogicalObservable {
            id: observable_id,
            measurement_keys: keys_by_id.remove(&observable_id).unwrap_or_default(),
            pauli_qubits: Vec::new(),
            pauli: String::new(),
        })
        .collect()
}

fn validate_detector_ids(detectors: &[Detector]) -> NpResult<()> {
    let mut seen = HashSet::new();
    for detector in detectors {
        if !seen.insert(detector.id) {
            return Err(NpError::new("detector ids must be unique"));
        }
    }
    Ok(())
}

fn validate_observable_ids(observables: &[LogicalObservable]) -> NpResult<()> {
    let mut seen = HashSet::new();
    for observable in observables {
        if !seen.insert(observable.id) {
            return Err(NpError::new("logical observable ids must be unique"));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedDemEdge {
    pub probability: f64,
    pub detectors: Vec<i64>,
    pub observables: Vec<i64>,
    pub location_id: String,
    pub event: DemEvent,
    pub tags: HashMap<String, crate::TagValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedDemEdgeRef {
    pub event_index: usize,
    pub detectors: Vec<i64>,
    pub observables: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LazyDetectorErrorModel {
    pub detectors: Vec<Detector>,
    pub observables: Vec<LogicalObservable>,
    pub event_plan: Arc<DemEventPlan>,
    pub edges: Vec<GeneratedDemEdgeRef>,
}

impl LazyDetectorErrorModel {
    pub fn materialize(&self) -> DetectorErrorModel {
        DetectorErrorModel {
            detectors: self.detectors.clone(),
            observables: self.observables.clone(),
            edges: self
                .edges
                .iter()
                .map(|edge| {
                    let event = &self.event_plan.fault_events[edge.event_index];
                    DetectorErrorEdge {
                        probability: event.probability,
                        detectors: edge.detectors.clone(),
                        observables: edge.observables.clone(),
                        location_id: event.location_id.clone(),
                        event: event.event.clone(),
                        tags: event.tags.clone(),
                    }
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DemFaultEvent {
    location_id: String,
    qubits: Vec<usize>,
    event: DemEvent,
    probability: f64,
    tags: HashMap<String, crate::TagValue>,
}

struct DemFaultPropagationState {
    reference: ConcreteStabilizer,
    x_frame: Vec<Mask>,
    z_frame: Vec<Mask>,
    measurement_flip_masks: HashMap<String, Mask>,
    event_words: usize,
}

impl DemFaultPropagationState {
    fn new(n_qubits: usize, event_count: usize) -> Self {
        let event_words = word_count(event_count);
        Self {
            reference: ConcreteStabilizer::zero(n_qubits),
            x_frame: vec![Mask::zero(event_words); n_qubits],
            z_frame: vec![Mask::zero(event_words); n_qubits],
            measurement_flip_masks: HashMap::new(),
            event_words,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProductAxis {
    X,
    Y,
    Z,
}

impl ProductAxis {
    fn apply_h(self) -> Self {
        match self {
            Self::X => Self::Z,
            Self::Y => Self::Y,
            Self::Z => Self::X,
        }
    }

    fn apply_s_ignoring_sign(self) -> Self {
        match self {
            Self::X => Self::Y,
            Self::Y => Self::X,
            Self::Z => Self::Z,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct IndexedDemDetector {
    id: i64,
    measurement_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct IndexedDemObservable {
    id: i64,
    measurement_indices: Vec<usize>,
    pauli_qubits: Vec<usize>,
    pauli: String,
}

#[derive(Debug, Clone, PartialEq)]
struct DemMeasurementPlan {
    measurement_count: usize,
    measurement_indices_by_op: Vec<Option<usize>>,
    detectors: Vec<IndexedDemDetector>,
    observables: Vec<IndexedDemObservable>,
}

struct IndexedProductFaultPropagationState {
    basis: Vec<ProductAxis>,
    x_frame: Vec<u64>,
    z_frame: Vec<u64>,
    measurement_flip_words: Vec<u64>,
    measurement_recorded: Vec<bool>,
    event_words: usize,
}

impl IndexedProductFaultPropagationState {
    fn new(n_qubits: usize, event_count: usize, measurement_count: usize) -> Self {
        let event_words = word_count(event_count);
        Self {
            basis: vec![ProductAxis::Z; n_qubits],
            x_frame: vec![0; n_qubits * event_words],
            z_frame: vec![0; n_qubits * event_words],
            measurement_flip_words: vec![0; measurement_count * event_words],
            measurement_recorded: vec![false; measurement_count],
            event_words,
        }
    }
}

fn collect_dem_fault_events(
    operations: &[Operation],
) -> NpResult<(Vec<DemFaultEvent>, Vec<Vec<usize>>)> {
    let mut fault_events = Vec::new();
    let mut fault_events_by_op = vec![Vec::new(); operations.len()];
    let mut seen = HashSet::new();
    for (op_index, operation) in operations.iter().enumerate() {
        match operation {
            Operation::Noise(location) => {
                collect_dem_fault_events_for_location(
                    &mut fault_events,
                    &mut fault_events_by_op,
                    &mut seen,
                    op_index,
                    location,
                )?;
            }
            Operation::Measure {
                noise: Some(location),
                ..
            }
            | Operation::MeasurePauli {
                noise: Some(location),
                ..
            } => {
                if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
                    return Err(NpError::new(
                        "DEM generation currently supports MeasurementBitFlip on measurement operations",
                    ));
                }
                collect_dem_fault_events_for_location(
                    &mut fault_events,
                    &mut fault_events_by_op,
                    &mut seen,
                    op_index,
                    location,
                )?;
            }
            _ => {}
        }
    }
    Ok((fault_events, fault_events_by_op))
}

fn collect_dem_fault_events_for_location(
    fault_events: &mut Vec<DemFaultEvent>,
    fault_events_by_op: &mut [Vec<usize>],
    seen: &mut HashSet<String>,
    op_index: usize,
    location: &NoiseLocation,
) -> NpResult<()> {
    if !seen.insert(location.id.clone()) {
        return Err(NpError::new(format!(
            "DetectorErrorModelGenerator requires unique noise location ids; duplicate id {:?}",
            location.id
        )));
    }
    for (event, probability) in non_identity_events(location)? {
        let event_index = fault_events.len();
        fault_events.push(DemFaultEvent {
            location_id: location.id.clone(),
            qubits: location.qubits.clone(),
            event,
            probability,
            tags: location.tags.clone(),
        });
        fault_events_by_op[op_index].push(event_index);
    }
    Ok(())
}

fn non_identity_events(location: &NoiseLocation) -> NpResult<Vec<(DemEvent, f64)>> {
    match &location.model {
        NoiseModel::BernoulliPauli(pauli) => {
            Ok(vec![(DemEvent::Pauli(pauli.clone()), location.rate)])
        }
        NoiseModel::MeasurementBitFlip => Ok(vec![(DemEvent::Bool(true), location.rate)]),
        NoiseModel::SingleQubitDepolarizing => Ok(["X", "Y", "Z"]
            .iter()
            .map(|event| (DemEvent::Pauli((*event).to_string()), location.rate / 3.0))
            .collect()),
        NoiseModel::TwoQubitDepolarizing => {
            let fault_events = [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY",
                "ZZ",
            ];
            Ok(fault_events
                .iter()
                .map(|event| {
                    (
                        DemEvent::Pauli((*event).to_string()),
                        location.rate / fault_events.len() as f64,
                    )
                })
                .collect())
        }
        NoiseModel::PauliChannel(weights) => {
            let total: f64 = weights.iter().map(|(_, weight)| *weight).sum();
            if total <= 0.0 {
                return Err(NpError::new(
                    "PauliChannel weights must have positive total weight",
                ));
            }
            Ok(weights
                .iter()
                .filter(|(_, weight)| *weight > 0.0)
                .map(|(event, weight)| {
                    (
                        DemEvent::Pauli(event.clone()),
                        location.rate * *weight / total,
                    )
                })
                .collect())
        }
    }
}

fn supports_product_reference_fast_path(n_qubits: usize, operations: &[Operation]) -> bool {
    let mut basis = vec![ProductAxis::Z; n_qubits];
    for operation in operations {
        match operation {
            Operation::H(q) => {
                let Some(axis) = basis.get_mut(*q) else {
                    return false;
                };
                *axis = axis.apply_h();
            }
            Operation::S(q) | Operation::SDag(q) => {
                let Some(axis) = basis.get_mut(*q) else {
                    return false;
                };
                *axis = axis.apply_s_ignoring_sign();
            }
            Operation::Swap(left, right) => {
                if *left >= basis.len() || *right >= basis.len() {
                    return false;
                }
                basis.swap(*left, *right);
            }
            Operation::Reset {
                qubit,
                basis: reset_basis,
                ..
            } => {
                if *qubit >= basis.len() {
                    return false;
                }
                let Ok(axis) = product_axis_from_pauli_bytes(reset_basis.as_bytes()) else {
                    return false;
                };
                basis[*qubit] = axis;
            }
            Operation::Cx(control, target) => {
                if !z_product_pair(&basis, *control, *target) {
                    return false;
                }
            }
            Operation::Cz(left, right) => {
                if !z_product_pair(&basis, *left, *right) {
                    return false;
                }
            }
            Operation::Pauli { .. }
            | Operation::Noise(_)
            | Operation::Measure { .. }
            | Operation::MeasurePauli { .. }
            | Operation::Detector { .. }
            | Operation::ObservableInclude { .. } => {}
        }
    }
    true
}

fn z_product_pair(basis: &[ProductAxis], left: usize, right: usize) -> bool {
    matches!(
        (basis.get(left), basis.get(right)),
        (Some(ProductAxis::Z), Some(ProductAxis::Z))
    )
}

fn compile_dem_measurement_plan(
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
) -> NpResult<DemMeasurementPlan> {
    let required_keys = required_measurement_keys(detectors, observables);
    let mut measurement_indices_by_op = vec![None; operations.len()];
    let mut measurement_index_by_key = HashMap::<String, usize>::new();
    let mut seen_measurement_keys = HashSet::<String>::new();
    let mut operation_measurement_count = 0usize;
    let mut indexed_measurement_count = 0usize;

    for (op_index, operation) in operations.iter().enumerate() {
        let key = match operation {
            Operation::Measure { key, .. } | Operation::MeasurePauli { key, .. } => key
                .clone()
                .unwrap_or_else(|| format!("m{operation_measurement_count}")),
            Operation::Reset { key: Some(key), .. } => key.clone(),
            _ => continue,
        };
        operation_measurement_count += 1;
        if !seen_measurement_keys.insert(key.clone()) {
            return Err(NpError::new(format!("duplicate measurement key {key:?}")));
        }
        if required_keys.contains(&key) {
            let measurement_index = indexed_measurement_count;
            indexed_measurement_count += 1;
            measurement_index_by_key.insert(key, measurement_index);
            measurement_indices_by_op[op_index] = Some(measurement_index);
        }
    }

    let mut indexed_detectors = detectors
        .iter()
        .map(|detector| {
            Ok(IndexedDemDetector {
                id: detector.id,
                measurement_indices: measurement_indices_for_keys(
                    &measurement_index_by_key,
                    &detector.measurement_keys,
                )?,
            })
        })
        .collect::<NpResult<Vec<_>>>()?;
    indexed_detectors.sort_by_key(|detector| detector.id);

    let mut indexed_observables = observables
        .iter()
        .map(|observable| {
            Ok(IndexedDemObservable {
                id: observable.id,
                measurement_indices: measurement_indices_for_keys(
                    &measurement_index_by_key,
                    &observable.measurement_keys,
                )?,
                pauli_qubits: observable.pauli_qubits.clone(),
                pauli: observable.pauli.clone(),
            })
        })
        .collect::<NpResult<Vec<_>>>()?;
    indexed_observables.sort_by_key(|observable| observable.id);

    Ok(DemMeasurementPlan {
        measurement_count: indexed_measurement_count,
        measurement_indices_by_op,
        detectors: indexed_detectors,
        observables: indexed_observables,
    })
}

fn required_measurement_keys(
    detectors: &[Detector],
    observables: &[LogicalObservable],
) -> HashSet<String> {
    let mut keys = HashSet::new();
    for detector in detectors {
        keys.extend(detector.measurement_keys.iter().cloned());
    }
    for observable in observables {
        keys.extend(observable.measurement_keys.iter().cloned());
    }
    keys
}

fn measurement_indices_for_keys(
    measurement_index_by_key: &HashMap<String, usize>,
    keys: &[String],
) -> NpResult<Vec<usize>> {
    keys.iter()
        .map(|key| {
            measurement_index_by_key
                .get(key)
                .copied()
                .ok_or_else(|| NpError::new(format!("unknown measurement key {key:?}")))
        })
        .collect()
}

fn generate_indexed_product_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdge>> {
    let mut state = IndexedProductFaultPropagationState::new(
        n_qubits,
        event_plan.fault_events.len(),
        measurement_plan.measurement_count,
    );
    for (op_index, operation) in operations.iter().enumerate() {
        apply_indexed_product_fault_propagation_operation(
            operation,
            op_index,
            &event_plan.fault_events,
            &event_plan.fault_events_by_op,
            measurement_plan,
            &mut state,
        )?;
    }
    let detector_flip_masks = evaluate_indexed_detector_flip_masks(
        &state.measurement_flip_words,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_flip_masks = evaluate_indexed_observable_flip_masks(
        &state.measurement_flip_words,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_generated_dem_edges_from_flat_flip_masks(
        &event_plan.fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn generate_indexed_product_dem_edge_refs_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdgeRef>> {
    let mut state = IndexedProductFaultPropagationState::new(
        n_qubits,
        event_plan.fault_events.len(),
        measurement_plan.measurement_count,
    );
    for (op_index, operation) in operations.iter().enumerate() {
        apply_indexed_product_fault_propagation_operation(
            operation,
            op_index,
            &event_plan.fault_events,
            &event_plan.fault_events_by_op,
            measurement_plan,
            &mut state,
        )?;
    }
    let detector_flip_masks = evaluate_indexed_detector_flip_masks(
        &state.measurement_flip_words,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_flip_masks = evaluate_indexed_observable_flip_masks(
        &state.measurement_flip_words,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_dem_edge_refs_from_flat_flip_masks(
        event_plan.fault_events.len(),
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn generate_indexed_product_sampling_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<DemSamplerEdge>> {
    let mut state = IndexedProductFaultPropagationState::new(
        n_qubits,
        event_plan.fault_events.len(),
        measurement_plan.measurement_count,
    );
    for (op_index, operation) in operations.iter().enumerate() {
        apply_indexed_product_fault_propagation_operation(
            operation,
            op_index,
            &event_plan.fault_events,
            &event_plan.fault_events_by_op,
            measurement_plan,
            &mut state,
        )?;
    }
    let detector_flip_masks = evaluate_indexed_detector_flip_masks(
        &state.measurement_flip_words,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_flip_masks = evaluate_indexed_observable_flip_masks(
        &state.measurement_flip_words,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_sampling_edges_from_flat_flip_masks(
        &event_plan.fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn apply_indexed_product_fault_propagation_operation(
    operation: &Operation,
    op_index: usize,
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    measurement_plan: &DemMeasurementPlan,
    state: &mut IndexedProductFaultPropagationState,
) -> NpResult<()> {
    match operation {
        Operation::H(q) => {
            state.basis[*q] = state.basis[*q].apply_h();
            swap_flat_rows_between_frames(
                &mut state.x_frame,
                &mut state.z_frame,
                *q,
                state.event_words,
            );
        }
        Operation::S(q) | Operation::SDag(q) => {
            state.basis[*q] = state.basis[*q].apply_s_ignoring_sign();
            xor_between_flat_frames(
                &state.x_frame,
                &mut state.z_frame,
                *q,
                *q,
                state.event_words,
            );
        }
        Operation::Swap(left, right) => {
            state.basis.swap(*left, *right);
            swap_flat_rows(&mut state.x_frame, *left, *right, state.event_words);
            swap_flat_rows(&mut state.z_frame, *left, *right, state.event_words);
        }
        Operation::Cx(control, target) => {
            if !z_product_pair(&state.basis, *control, *target) {
                return Err(NpError::new(
                    "product fast path received a Cx that entangles the reference state",
                ));
            }
            xor_within_flat_frame(&mut state.x_frame, *target, *control, state.event_words);
            xor_within_flat_frame(&mut state.z_frame, *control, *target, state.event_words);
        }
        Operation::Cz(left, right) => {
            if !z_product_pair(&state.basis, *left, *right) {
                return Err(NpError::new(
                    "product fast path received a Cz that entangles the reference state",
                ));
            }
            xor_between_flat_frames(
                &state.x_frame,
                &mut state.z_frame,
                *left,
                *right,
                state.event_words,
            );
            xor_between_flat_frames(
                &state.x_frame,
                &mut state.z_frame,
                *right,
                *left,
                state.event_words,
            );
        }
        Operation::Pauli { .. } => {}
        Operation::Noise(_) => {
            apply_fault_events_to_flat_frames(
                fault_events,
                fault_events_by_op,
                op_index,
                &mut state.x_frame,
                &mut state.z_frame,
                state.event_words,
            )?;
        }
        Operation::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = [*qubit];
            ensure_product_deterministic_measurement(&state.basis, &qubits, basis, key.as_deref())?;
            if let Some(measurement_index) =
                optional_indexed_measurement_op(measurement_plan, op_index)
            {
                record_flat_measurement_flip(
                    state,
                    measurement_index,
                    &qubits,
                    basis,
                    fault_events,
                    fault_events_by_op,
                    op_index,
                )?;
            }
        }
        Operation::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            ensure_product_deterministic_measurement(&state.basis, qubits, pauli, key.as_deref())?;
            if let Some(measurement_index) =
                optional_indexed_measurement_op(measurement_plan, op_index)
            {
                record_flat_measurement_flip(
                    state,
                    measurement_index,
                    qubits,
                    pauli,
                    fault_events,
                    fault_events_by_op,
                    op_index,
                )?;
            }
        }
        Operation::Reset { qubit, key, basis } => {
            let qubits = [*qubit];
            if key.is_some() {
                ensure_product_deterministic_measurement(
                    &state.basis,
                    &qubits,
                    basis,
                    key.as_deref(),
                )?;
                if let Some(measurement_index) =
                    optional_indexed_measurement_op(measurement_plan, op_index)
                {
                    record_flat_measurement_flip(
                        state,
                        measurement_index,
                        &qubits,
                        basis,
                        fault_events,
                        fault_events_by_op,
                        op_index,
                    )?;
                }
            }
            state.basis[*qubit] = product_axis_from_pauli_bytes(basis.as_bytes())?;
            zero_flat_row(&mut state.x_frame, *qubit, state.event_words);
            zero_flat_row(&mut state.z_frame, *qubit, state.event_words);
        }
        Operation::Detector { .. } | Operation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn optional_indexed_measurement_op(
    measurement_plan: &DemMeasurementPlan,
    op_index: usize,
) -> Option<usize> {
    measurement_plan
        .measurement_indices_by_op
        .get(op_index)
        .and_then(|index| *index)
}

fn record_flat_measurement_flip(
    state: &mut IndexedProductFaultPropagationState,
    measurement_index: usize,
    qubits: &[usize],
    pauli: &str,
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
) -> NpResult<()> {
    if measurement_index >= state.measurement_recorded.len() {
        return Err(NpError::new(format!(
            "internal DEM measurement index {measurement_index} is out of range"
        )));
    }
    if state.measurement_recorded[measurement_index] {
        return Err(NpError::new(format!(
            "duplicate DEM measurement index {measurement_index}"
        )));
    }
    let range = flat_range(measurement_index, state.event_words);
    state.measurement_flip_words[range.clone()].fill(0);
    xor_flat_frame_measurement_flip_into(
        &mut state.measurement_flip_words[range],
        &state.x_frame,
        &state.z_frame,
        qubits,
        pauli,
        state.event_words,
    )?;
    xor_flat_measurement_noise_events(
        &mut state.measurement_flip_words[flat_range(measurement_index, state.event_words)],
        fault_events,
        fault_events_by_op,
        op_index,
    )?;
    state.measurement_recorded[measurement_index] = true;
    Ok(())
}

fn flat_range(index: usize, words: usize) -> std::ops::Range<usize> {
    let start = index * words;
    start..start + words
}

fn xor_word_slices(target: &mut [u64], source: &[u64]) {
    for (left, right) in target.iter_mut().zip(source) {
        *left ^= *right;
    }
}

fn xor_within_flat_frame(frame: &mut [u64], target: usize, source: usize, words: usize) {
    let target_start = target * words;
    let source_start = source * words;
    for word in 0..words {
        let value = frame[source_start + word];
        frame[target_start + word] ^= value;
    }
}

fn xor_between_flat_frames(
    source: &[u64],
    target: &mut [u64],
    source_index: usize,
    target_index: usize,
    words: usize,
) {
    let source_range = flat_range(source_index, words);
    let target_range = flat_range(target_index, words);
    xor_word_slices(&mut target[target_range], &source[source_range]);
}

fn swap_flat_rows(frame: &mut [u64], left: usize, right: usize, words: usize) {
    if left == right {
        return;
    }
    let left_start = left * words;
    let right_start = right * words;
    for word in 0..words {
        frame.swap(left_start + word, right_start + word);
    }
}

fn swap_flat_rows_between_frames(left: &mut [u64], right: &mut [u64], index: usize, words: usize) {
    let range = flat_range(index, words);
    for offset in range {
        std::mem::swap(&mut left[offset], &mut right[offset]);
    }
}

fn zero_flat_row(frame: &mut [u64], index: usize, words: usize) {
    frame[flat_range(index, words)].fill(0);
}

fn apply_fault_events_to_flat_frames(
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
    x_frame: &mut [u64],
    z_frame: &mut [u64],
    words: usize,
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        if let DemEvent::Pauli(pauli) = &fault_events[*event_index].event {
            apply_fault_event_pauli_string_to_flat_frames(
                x_frame,
                z_frame,
                words,
                &fault_events[*event_index].qubits,
                pauli,
                *event_index,
            )?;
        }
    }
    Ok(())
}

fn apply_fault_event_pauli_string_to_flat_frames(
    x_frame: &mut [u64],
    z_frame: &mut [u64],
    words: usize,
    qubits: &[usize],
    pauli: &str,
    event_index: usize,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("event Pauli length does not match qubits"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.bytes()) {
        match local {
            b'I' => {}
            b'X' => set_flat_event_bit(x_frame, *qubit, words, event_index),
            b'Z' => set_flat_event_bit(z_frame, *qubit, words, event_index),
            b'Y' => {
                set_flat_event_bit(x_frame, *qubit, words, event_index);
                set_flat_event_bit(z_frame, *qubit, words, event_index);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    local as char
                )));
            }
        }
    }
    Ok(())
}

fn xor_flat_measurement_noise_events(
    value: &mut [u64],
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        match &fault_events[*event_index].event {
            DemEvent::Bool(true) => set_event_bit_in_words(value, *event_index),
            DemEvent::Bool(false) => {}
            DemEvent::Pauli(_) => {
                return Err(NpError::new("measurement noise event must be boolean"));
            }
        }
    }
    Ok(())
}

fn xor_flat_frame_measurement_flip_into(
    out: &mut [u64],
    x_frame: &[u64],
    z_frame: &[u64],
    qubits: &[usize],
    pauli: &str,
    words: usize,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.as_bytes()) {
        match *local {
            b'I' => {}
            b'X' => xor_word_slices(out, &z_frame[flat_range(*qubit, words)]),
            b'Z' => xor_word_slices(out, &x_frame[flat_range(*qubit, words)]),
            b'Y' => {
                xor_word_slices(out, &x_frame[flat_range(*qubit, words)]);
                xor_word_slices(out, &z_frame[flat_range(*qubit, words)]);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    *local as char
                )));
            }
        }
    }
    Ok(())
}

fn set_flat_event_bit(frame: &mut [u64], row: usize, words: usize, event_index: usize) {
    let word_index = event_index / 64;
    if word_index >= words {
        return;
    }
    let bit_index = event_index % 64;
    let offset = row * words + word_index;
    if let Some(word) = frame.get_mut(offset) {
        *word |= 1u64 << bit_index;
    }
}

fn set_event_bit_in_words(words: &mut [u64], event_index: usize) {
    let word_index = event_index / 64;
    let bit_index = event_index % 64;
    if let Some(word) = words.get_mut(word_index) {
        *word |= 1u64 << bit_index;
    }
}

fn apply_fault_propagation_operation(
    operation: &Operation,
    op_index: usize,
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    state: &mut DemFaultPropagationState,
) -> NpResult<()> {
    match operation {
        Operation::H(q) => {
            state.reference.apply_h(*q);
            std::mem::swap(&mut state.x_frame[*q], &mut state.z_frame[*q]);
        }
        Operation::S(q) => {
            state.reference.apply_s(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Operation::SDag(q) => {
            state.reference.apply_s_dag(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Operation::Cx(control, target) => {
            state.reference.apply_cx(*control, *target);
            xor_within_frame(&mut state.x_frame, *target, *control);
            xor_within_frame(&mut state.z_frame, *control, *target);
        }
        Operation::Cz(left, right) => {
            state.reference.apply_cz(*left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *right, *left);
        }
        Operation::Swap(left, right) => {
            state.reference.apply_swap(*left, *right);
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        Operation::Pauli { qubits, pauli } => {
            let (x, z) = sparse_pauli_to_xz(state.reference.n_qubits(), qubits, pauli)?;
            state.reference.apply_pauli_string(&x, &z);
        }
        Operation::Noise(_) => {
            apply_fault_events(fault_events, fault_events_by_op, op_index, state)?;
        }
        Operation::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = vec![*qubit];
            ensure_deterministic_dem_measurement(&state.reference, &qubits, basis, key.as_deref())?;
            let mut value = frame_measurement_flip_mask(
                &state.x_frame,
                &state.z_frame,
                &qubits,
                basis,
                state.event_words,
            )?;
            xor_measurement_noise_events(&mut value, fault_events, fault_events_by_op, op_index)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", state.measurement_flip_masks.len()));
            record_measurement_flip_mask(&mut state.measurement_flip_masks, &key, value)?;
        }
        Operation::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            ensure_deterministic_dem_measurement(&state.reference, qubits, pauli, key.as_deref())?;
            let mut value = frame_measurement_flip_mask(
                &state.x_frame,
                &state.z_frame,
                qubits,
                pauli,
                state.event_words,
            )?;
            xor_measurement_noise_events(&mut value, fault_events, fault_events_by_op, op_index)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", state.measurement_flip_masks.len()));
            record_measurement_flip_mask(&mut state.measurement_flip_masks, &key, value)?;
        }
        Operation::Reset { qubit, key, basis } => {
            let qubits = vec![*qubit];
            if let Some(key) = key {
                ensure_deterministic_dem_measurement(&state.reference, &qubits, basis, Some(key))?;
                let value = frame_measurement_flip_mask(
                    &state.x_frame,
                    &state.z_frame,
                    &qubits,
                    basis,
                    state.event_words,
                )?;
                record_measurement_flip_mask(&mut state.measurement_flip_masks, key, value)?;
            }
            state.reference.reset_prepare(*qubit, basis)?;
            state.x_frame[*qubit] = Mask::zero(state.event_words);
            state.z_frame[*qubit] = Mask::zero(state.event_words);
        }
        Operation::Detector { .. } | Operation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn apply_fault_events(
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
    state: &mut DemFaultPropagationState,
) -> NpResult<()> {
    apply_fault_events_to_frames(
        fault_events,
        fault_events_by_op,
        op_index,
        &mut state.x_frame,
        &mut state.z_frame,
    )
}

fn apply_fault_events_to_frames(
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
    x_frame: &mut [Mask],
    z_frame: &mut [Mask],
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        if let DemEvent::Pauli(pauli) = &fault_events[*event_index].event {
            apply_fault_event_pauli_string(
                x_frame,
                z_frame,
                &fault_events[*event_index].qubits,
                pauli,
                *event_index,
            )?;
        }
    }
    Ok(())
}

fn ensure_product_deterministic_measurement(
    basis: &[ProductAxis],
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.as_bytes()) {
        if *local == b'I' {
            continue;
        }
        if product_axis_from_pauli_byte(*local)? != basis[*qubit] {
            return Err(NpError::new(format!(
                "measurement {:?} is random in the ideal/single-error circuit",
                key.unwrap_or("measure")
            )));
        }
    }
    Ok(())
}

fn product_axis_from_pauli_bytes(pauli: &[u8]) -> NpResult<ProductAxis> {
    if pauli.len() != 1 {
        return Err(NpError::new(format!(
            "product reset basis must be a single-qubit Pauli, got {:?}",
            String::from_utf8_lossy(pauli)
        )));
    }
    product_axis_from_pauli_byte(pauli[0])
}

fn product_axis_from_pauli_byte(pauli: u8) -> NpResult<ProductAxis> {
    match pauli {
        b'X' => Ok(ProductAxis::X),
        b'Y' => Ok(ProductAxis::Y),
        b'Z' => Ok(ProductAxis::Z),
        _ => Err(NpError::new(format!(
            "unsupported Pauli {:?}",
            pauli as char
        ))),
    }
}

fn apply_fault_event_pauli_string(
    x_frame: &mut [Mask],
    z_frame: &mut [Mask],
    qubits: &[usize],
    pauli: &str,
    event_index: usize,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("event Pauli length does not match qubits"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.bytes()) {
        match local {
            b'I' => {}
            b'X' => set_event_bit(&mut x_frame[*qubit], event_index),
            b'Z' => set_event_bit(&mut z_frame[*qubit], event_index),
            b'Y' => {
                set_event_bit(&mut x_frame[*qubit], event_index);
                set_event_bit(&mut z_frame[*qubit], event_index);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    local as char
                )))
            }
        }
    }
    Ok(())
}

fn xor_measurement_noise_events(
    value: &mut Mask,
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        match &fault_events[*event_index].event {
            DemEvent::Bool(true) => set_event_bit(value, *event_index),
            DemEvent::Bool(false) => {}
            DemEvent::Pauli(_) => {
                return Err(NpError::new("measurement noise event must be boolean"));
            }
        }
    }
    Ok(())
}

fn frame_measurement_flip_mask(
    x_frame: &[Mask],
    z_frame: &[Mask],
    qubits: &[usize],
    pauli: &str,
    words: usize,
) -> NpResult<Mask> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    let mut flip = Mask::zero(words);
    let pauli_bytes = pauli.as_bytes();
    if pauli_bytes.iter().all(|local| *local == b'Z') {
        for qubit in qubits {
            flip.xor_assign(&x_frame[*qubit]);
        }
        return Ok(flip);
    }
    if pauli_bytes.iter().all(|local| *local == b'X') {
        for qubit in qubits {
            flip.xor_assign(&z_frame[*qubit]);
        }
        return Ok(flip);
    }
    if pauli_bytes.iter().all(|local| *local == b'Y') {
        for qubit in qubits {
            flip.xor_assign(&x_frame[*qubit]);
            flip.xor_assign(&z_frame[*qubit]);
        }
        return Ok(flip);
    }
    for (qubit, local) in qubits.iter().zip(pauli_bytes) {
        match *local {
            b'I' => {}
            b'X' => flip.xor_assign(&z_frame[*qubit]),
            b'Z' => flip.xor_assign(&x_frame[*qubit]),
            b'Y' => {
                flip.xor_assign(&x_frame[*qubit]);
                flip.xor_assign(&z_frame[*qubit]);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    *local as char
                )))
            }
        }
    }
    Ok(flip)
}

fn ensure_deterministic_dem_measurement(
    state: &ConcreteStabilizer,
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> NpResult<()> {
    if !state.is_deterministic_sparse_pauli(qubits, pauli)? {
        return Err(NpError::new(format!(
            "measurement {:?} is random in the ideal/single-error circuit",
            key.unwrap_or("measure")
        )));
    }
    Ok(())
}

fn evaluate_detector_flip_masks(
    measurement_flip_masks: &HashMap<String, Mask>,
    detectors: &[Detector],
    words: usize,
) -> NpResult<HashMap<i64, Mask>> {
    let mut out = HashMap::new();
    for detector in detectors {
        out.insert(
            detector.id,
            measurement_flip_parity(measurement_flip_masks, &detector.measurement_keys, words)?,
        );
    }
    Ok(out)
}

fn evaluate_observable_flip_masks(
    measurement_flip_masks: &HashMap<String, Mask>,
    x_frame: &[Mask],
    z_frame: &[Mask],
    observables: &[LogicalObservable],
    words: usize,
) -> NpResult<HashMap<i64, Mask>> {
    let mut out = HashMap::new();
    for observable in observables {
        let mut value =
            measurement_flip_parity(measurement_flip_masks, &observable.measurement_keys, words)?;
        if !observable.pauli.is_empty() {
            let flip = frame_measurement_flip_mask(
                x_frame,
                z_frame,
                &observable.pauli_qubits,
                &observable.pauli,
                words,
            )?;
            value.xor_assign(&flip);
        }
        out.insert(observable.id, value);
    }
    Ok(out)
}

fn evaluate_indexed_detector_flip_masks(
    measurement_flip_words: &[u64],
    detectors: &[IndexedDemDetector],
    words: usize,
) -> NpResult<Vec<(i64, Vec<u64>)>> {
    let mut out = Vec::with_capacity(detectors.len());
    for detector in detectors {
        out.push((
            detector.id,
            measurement_index_flip_parity(
                measurement_flip_words,
                &detector.measurement_indices,
                words,
            )?,
        ));
    }
    Ok(out)
}

fn evaluate_indexed_observable_flip_masks(
    measurement_flip_words: &[u64],
    x_frame: &[u64],
    z_frame: &[u64],
    observables: &[IndexedDemObservable],
    words: usize,
) -> NpResult<Vec<(i64, Vec<u64>)>> {
    let mut out = Vec::with_capacity(observables.len());
    for observable in observables {
        let mut value = measurement_index_flip_parity(
            measurement_flip_words,
            &observable.measurement_indices,
            words,
        )?;
        if !observable.pauli.is_empty() {
            xor_flat_frame_measurement_flip_into(
                &mut value,
                x_frame,
                z_frame,
                &observable.pauli_qubits,
                &observable.pauli,
                words,
            )?;
        }
        out.push((observable.id, value));
    }
    Ok(out)
}

fn measurement_flip_parity(
    measurement_flip_masks: &HashMap<String, Mask>,
    keys: &[String],
    words: usize,
) -> NpResult<Mask> {
    let mut parity = Mask::zero(words);
    for key in keys {
        let value = measurement_flip_masks
            .get(key)
            .ok_or_else(|| NpError::new(format!("unknown measurement key {key:?}")))?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

fn measurement_index_flip_parity(
    measurement_flip_words: &[u64],
    indices: &[usize],
    words: usize,
) -> NpResult<Vec<u64>> {
    let mut parity = vec![0; words];
    for index in indices {
        let range = flat_range(*index, words);
        let value = measurement_flip_words
            .get(range)
            .ok_or_else(|| NpError::new(format!("unknown measurement index {index}")))?;
        xor_word_slices(&mut parity, value);
    }
    Ok(parity)
}

fn record_measurement_flip_mask(
    measurement_flip_masks: &mut HashMap<String, Mask>,
    key: &str,
    value: Mask,
) -> NpResult<()> {
    if measurement_flip_masks.contains_key(key) {
        return Err(NpError::new(format!("duplicate measurement key {key:?}")));
    }
    measurement_flip_masks.insert(key.to_string(), value);
    Ok(())
}

fn sorted_flip_masks(flip_masks: HashMap<i64, Mask>) -> Vec<(i64, Mask)> {
    let mut out: Vec<(i64, Mask)> = flip_masks.into_iter().collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

fn flip_ids_by_fault_event(flip_masks: &[(i64, Mask)], event_count: usize) -> Vec<Vec<i64>> {
    let mut out = vec![Vec::new(); event_count];
    for (id, mask) in flip_masks {
        for (word_index, word) in mask.words.iter().enumerate() {
            let mut remaining = *word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let event_index = word_index * 64 + bit;
                if event_index < event_count {
                    out[event_index].push(*id);
                }
                remaining &= remaining - 1;
            }
        }
    }
    out
}

fn flat_flip_ids_by_fault_event(
    flip_masks: &[(i64, Vec<u64>)],
    event_count: usize,
) -> Vec<Vec<i64>> {
    let mut out = vec![Vec::new(); event_count];
    for (id, words) in flip_masks {
        for (word_index, word) in words.iter().enumerate() {
            let mut remaining = *word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let event_index = word_index * 64 + bit;
                if event_index < event_count {
                    out[event_index].push(*id);
                }
                remaining &= remaining - 1;
            }
        }
    }
    out
}

fn xor_within_frame(frame: &mut [Mask], target: usize, source: usize) {
    let source_mask = frame[source].clone();
    frame[target].xor_assign(&source_mask);
}

fn xor_between_frames(source: &[Mask], target: &mut [Mask], source_idx: usize, target_idx: usize) {
    let source_mask = source[source_idx].clone();
    target[target_idx].xor_assign(&source_mask);
}

fn set_event_bit(mask: &mut Mask, event_index: usize) {
    let word_index = event_index / 64;
    let bit_index = event_index % 64;
    if let Some(word) = mask.words.get_mut(word_index) {
        *word |= 1u64 << bit_index;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurement_bit_flip_generates_detector_edge() {
        let noise = NoiseLocation {
            id: "m_noise".to_string(),
            model: NoiseModel::MeasurementBitFlip,
            rate: 0.25,
            qubits: vec![0],
            tags: HashMap::new(),
        };
        let operations = vec![
            Operation::Measure {
                qubit: 0,
                key: Some("m0".to_string()),
                basis: "Z".to_string(),
                noise: Some(noise),
            },
            Operation::Detector {
                detector_id: Some(0),
                measurement_keys: vec!["m0".to_string()],
                coords: Vec::new(),
            },
        ];
        let detectors = vec![Detector {
            id: 0,
            measurement_keys: vec!["m0".to_string()],
            coords: Vec::new(),
        }];

        let edges = generate_dem_edges(1, &operations, &detectors, &[]).unwrap();

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].probability, 0.25);
        assert_eq!(edges[0].detectors, vec![0]);
        assert_eq!(edges[0].location_id, "m_noise");
        assert_eq!(edges[0].event, DemEvent::Bool(true));
    }

    #[test]
    fn generator_defaults_declarations_from_circuit_and_carries_tags() {
        let mut tags = HashMap::new();
        tags.insert(
            "gate".to_string(),
            crate::TagValue::String("idle".to_string()),
        );
        let location = NoiseLocation {
            id: "x0".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.125,
            qubits: vec![0],
            tags,
        };
        let circuit = Circuit {
            n_qubits: 1,
            operations: vec![
                Operation::Noise(location),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m0".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Detector {
                    detector_id: None,
                    measurement_keys: vec!["m0".to_string()],
                    coords: vec![1.0],
                },
                Operation::ObservableInclude {
                    observable_id: 0,
                    measurement_keys: vec!["m0".to_string()],
                },
            ],
        };
        let generator = DetectorErrorModelGenerator::new(circuit, None, None).unwrap();

        let dem = generator.generate().unwrap();

        assert_eq!(generator.detectors[0].id, 0);
        assert_eq!(generator.detectors[0].coords, vec![1.0]);
        assert_eq!(generator.observables[0].measurement_keys, vec!["m0"]);
        assert_eq!(dem.edges.len(), 1);
        assert_eq!(dem.edges[0].location_id, "x0");
        assert_eq!(
            dem.edges[0].tags.get("gate"),
            Some(&crate::TagValue::String("idle".to_string()))
        );
    }

    #[test]
    fn generator_sampler_edges_match_full_dem_edges() {
        let mut tags = HashMap::new();
        tags.insert(
            "operation".to_string(),
            crate::TagValue::String("idle".to_string()),
        );
        let location = NoiseLocation {
            id: "x0".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.25,
            qubits: vec![0],
            tags,
        };
        let circuit = Circuit {
            n_qubits: 1,
            operations: vec![
                Operation::Noise(location),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m0".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Detector {
                    detector_id: Some(0),
                    measurement_keys: vec!["m0".to_string()],
                    coords: Vec::new(),
                },
                Operation::ObservableInclude {
                    observable_id: 0,
                    measurement_keys: vec!["m0".to_string()],
                },
            ],
        };
        let generator = DetectorErrorModelGenerator::new(circuit, None, None).unwrap();

        let first = generator.generate().unwrap();
        let second = generator.generate().unwrap();
        let sampler_edges = generator.generate_sampler_edges().unwrap();

        assert_eq!(first, second);
        assert_eq!(sampler_edges.len(), first.edges.len());
        for (sampler_edge, dem_edge) in sampler_edges.iter().zip(first.edges.iter()) {
            assert_eq!(sampler_edge.probability, dem_edge.probability);
            assert_eq!(sampler_edge.detectors, dem_edge.detectors);
            assert_eq!(sampler_edge.observables, dem_edge.observables);
            assert_eq!(sampler_edge.location_id, dem_edge.location_id);
            assert_eq!(sampler_edge.event, dem_edge.event);
            assert_eq!(sampler_edge.tags, dem_edge.tags);
        }
    }

    #[test]
    fn generator_rejects_duplicate_detector_ids() {
        let circuit = Circuit {
            n_qubits: 1,
            operations: Vec::new(),
        };
        let detectors = vec![
            Detector {
                id: 0,
                measurement_keys: Vec::new(),
                coords: Vec::new(),
            },
            Detector {
                id: 0,
                measurement_keys: Vec::new(),
                coords: Vec::new(),
            },
        ];

        let err = DetectorErrorModelGenerator::new(circuit, Some(detectors), None).unwrap_err();

        assert!(err.message().contains("detector ids must be unique"));
    }
}

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

mod assembly;
mod bitset;
mod event_plan;
mod fallback_path;
mod indexed_parity;
mod measurement_plan;
mod product_path;

use assembly::{assemble_dem_edge_refs, materialize_generated_dem_edges, DemFlipMasks};
use event_plan::collect_dem_event_plan_from_program;
pub use event_plan::{collect_dem_event_plan, DemEventPlan};
use fallback_path::generate_fallback_dem_flip_masks_from_plan;
use measurement_plan::{
    compile_dem_measurement_plan, compile_dem_measurement_plan_with_optional_declarations,
    DemMeasurementPlan,
};
use product_path::{
    generate_indexed_product_dem_flip_masks_from_plan, supports_product_reference_fast_path,
};

use crate::dem_canonical::parity_support_len;
use crate::dem_problem::{compile_graphlike_problem_from_edge_views, GraphlikeSourceEdge};
use crate::dem_sampling::{DemHotspotEstimator, DemProgramEdge, DemProgramEdgeMetadata};
use crate::program::{ExpandedOperation, ExpandedProgram, ExpansionMode};
use crate::{
    Circuit, Detector, DetectorErrorEdge, DetectorErrorModel, GraphlikeDecodingProblem,
    LogicalObservable, NpError, NpResult, Operation,
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
    measurement_plan: DemMeasurementPlan,
}

/// A circuit whose targets have been validated and whose DEM event plan is
/// guaranteed to be derived from that exact circuit.
///
/// This type is intentionally opaque.  It lets trusted adapters carry the
/// validation result across API boundaries without allowing callers to pair a
/// circuit with an unrelated event plan.
#[doc(hidden)]
#[derive(Debug)]
pub struct ValidatedDemCircuit {
    circuit: Arc<Circuit>,
    event_plan: OnceLock<Arc<DemEventPlan>>,
}

impl ValidatedDemCircuit {
    /// Validate a circuit once and retain it for DEM compilation.
    #[doc(hidden)]
    pub fn new(circuit: Circuit) -> NpResult<Self> {
        circuit.validate()?;
        Ok(Self {
            circuit: Arc::new(circuit),
            event_plan: OnceLock::new(),
        })
    }

    /// Borrow the validated circuit.
    #[doc(hidden)]
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Return shared ownership of the validated circuit.
    fn shared_circuit(&self) -> Arc<Circuit> {
        Arc::clone(&self.circuit)
    }

    /// Return the event plan derived from this circuit, compiling it once.
    #[doc(hidden)]
    pub fn event_plan(&self) -> NpResult<Arc<DemEventPlan>> {
        if let Some(event_plan) = self.event_plan.get() {
            return Ok(Arc::clone(event_plan));
        }

        let event_plan = Arc::new(collect_dem_event_plan(&self.circuit.operations)?);
        let _ = self.event_plan.set(event_plan);
        Ok(Arc::clone(
            self.event_plan
                .get()
                .expect("DEM event plan must be initialized after set"),
        ))
    }
}

impl DemEventPlan {
    /// Materialize public detector declarations from the indexed IR.
    #[doc(hidden)]
    pub fn inferred_detectors(&self) -> Vec<Detector> {
        detectors_from_program(&self.program)
    }

    /// Materialize public observable declarations from the indexed IR.
    #[doc(hidden)]
    pub fn inferred_observables(&self) -> Vec<LogicalObservable> {
        observables_from_program(&self.program)
    }
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
        circuit.validate()?;
        let (circuit, program) = crate::program::expand_circuit(&circuit)?;
        let measurement_plan = compile_dem_measurement_plan_with_optional_declarations(
            &program,
            detectors.as_deref(),
            observables.as_deref(),
        )?;
        let detectors = detectors.unwrap_or_else(|| detectors_from_program(&program));
        let observables = observables.unwrap_or_else(|| observables_from_program(&program));
        validate_detector_ids(&detectors)?;
        validate_observable_ids(&observables)?;
        validate_observables_for_n_qubits(circuit.n_qubits, &observables)?;
        let event_plan = collect_dem_event_plan_from_program(program)?;
        Ok(Self {
            circuit: Arc::new(circuit),
            detectors,
            observables,
            event_plan: Arc::new(event_plan),
            measurement_plan,
        })
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
        Self::new_with_shared_event_plan_options(
            circuit,
            Some(detectors),
            Some(observables),
            event_plan,
        )
    }

    /// Create a generator from a shared integer event plan, inferring omitted
    /// declarations directly from that plan without a string round trip.
    #[doc(hidden)]
    pub fn new_with_shared_event_plan_options(
        circuit: Arc<Circuit>,
        detectors: Option<Vec<Detector>>,
        observables: Option<Vec<LogicalObservable>>,
        event_plan: Arc<DemEventPlan>,
    ) -> NpResult<Self> {
        circuit.validate()?;
        crate::program::validate_expanded_program_targets(circuit.n_qubits, &event_plan.program)?;
        Self::new_with_prevalidated_shared_event_plan_options(
            circuit,
            detectors,
            observables,
            event_plan,
        )
    }

    /// Create a generator from a circuit carrying proof that validation and
    /// event-plan binding have already been performed.
    #[doc(hidden)]
    pub fn new_with_validated_dem_circuit_options(
        circuit: &ValidatedDemCircuit,
        detectors: Option<Vec<Detector>>,
        observables: Option<Vec<LogicalObservable>>,
    ) -> NpResult<Self> {
        let event_plan = circuit.event_plan()?;
        Self::new_with_prevalidated_shared_event_plan_options(
            circuit.shared_circuit(),
            detectors,
            observables,
            event_plan,
        )
    }

    fn new_with_prevalidated_shared_event_plan_options(
        circuit: Arc<Circuit>,
        detectors: Option<Vec<Detector>>,
        observables: Option<Vec<LogicalObservable>>,
        event_plan: Arc<DemEventPlan>,
    ) -> NpResult<Self> {
        let measurement_plan = compile_dem_measurement_plan_with_optional_declarations(
            &event_plan.program,
            detectors.as_deref(),
            observables.as_deref(),
        )?;
        let detectors = detectors.unwrap_or_else(|| detectors_from_program(&event_plan.program));
        let observables =
            observables.unwrap_or_else(|| observables_from_program(&event_plan.program));
        validate_detector_ids(&detectors)?;
        validate_observable_ids(&observables)?;
        validate_observables_for_n_qubits(circuit.n_qubits, &observables)?;
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

    fn generate_edge_refs(&self) -> NpResult<Vec<GeneratedDemEdgeRef>> {
        // Constructors validate the circuit, compiled plan, and observables.
        // Keep repeated generation scan-free so validation remains a boundary
        // cost instead of scaling with every generated model.
        generate_dem_edge_refs_from_compiled_plan(
            self.circuit.n_qubits,
            &self.event_plan.program,
            &self.measurement_plan,
            &self.event_plan,
        )
    }
}

pub fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
) -> NpResult<Vec<DetectorErrorEdge>> {
    crate::model::validate_operations(n_qubits, operations)?;
    validate_observables_for_n_qubits(n_qubits, observables)?;
    let program = crate::program::expand_operations(operations, ExpansionMode::Dem)?;
    let event_plan = collect_dem_event_plan_from_program(program)?;
    let measurement_plan =
        compile_dem_measurement_plan(&event_plan.program, detectors, observables)?;
    generate_dem_edges_from_compiled_plan(
        n_qubits,
        &event_plan.program,
        &measurement_plan,
        &event_plan,
    )
}

/// Generate DEM edges from a precompiled integer event plan.
pub fn generate_dem_edges_from_event_plan(
    n_qubits: usize,
    detectors: &[Detector],
    observables: &[LogicalObservable],
    event_plan: &DemEventPlan,
) -> NpResult<Vec<DetectorErrorEdge>> {
    crate::program::validate_expanded_program_targets(n_qubits, &event_plan.program)?;
    validate_observables_for_n_qubits(n_qubits, observables)?;
    let measurement_plan =
        compile_dem_measurement_plan(&event_plan.program, detectors, observables)?;
    generate_dem_edges_from_compiled_plan(
        n_qubits,
        &event_plan.program,
        &measurement_plan,
        event_plan,
    )
}

fn generate_dem_edges_from_compiled_plan(
    n_qubits: usize,
    program: &ExpandedProgram,
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<DetectorErrorEdge>> {
    let edges =
        generate_dem_edge_refs_from_compiled_plan(n_qubits, program, measurement_plan, event_plan)?;
    Ok(materialize_generated_dem_edges(event_plan, edges))
}

fn generate_dem_edge_refs_from_compiled_plan(
    n_qubits: usize,
    program: &ExpandedProgram,
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdgeRef>> {
    let flip_masks = generate_dem_flip_masks_from_compiled_plan(
        n_qubits,
        program,
        measurement_plan,
        event_plan,
    )?;
    Ok(assemble_dem_edge_refs(
        event_plan.fault_events.len(),
        flip_masks,
    ))
}

fn generate_dem_flip_masks_from_compiled_plan(
    n_qubits: usize,
    program: &ExpandedProgram,
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<DemFlipMasks> {
    if supports_product_reference_fast_path(n_qubits, &program.operations) {
        return generate_indexed_product_dem_flip_masks_from_plan(
            n_qubits,
            &program.operations,
            measurement_plan,
            event_plan,
        );
    }
    generate_fallback_dem_flip_masks_from_plan(
        n_qubits,
        &program.operations,
        measurement_plan,
        event_plan,
    )
}

/// Infer detector declarations from detector operations in a circuit.
pub fn detectors_from_circuit(circuit: &Circuit) -> Vec<Detector> {
    let program = match crate::program::expand_operations(&circuit.operations, ExpansionMode::Dem) {
        Ok(program) => program,
        Err(_) => return Vec::new(),
    };
    detectors_from_program(&program)
}

fn detectors_from_program(program: &ExpandedProgram) -> Vec<Detector> {
    let mut detectors = Vec::new();
    let mut detector_coords = program.detector_coords.iter();
    for operation in &program.operations {
        let ExpandedOperation::Detector {
            detector_id,
            measurement_ids,
        } = operation
        else {
            continue;
        };
        detectors.push(Detector {
            id: *detector_id,
            measurement_keys: measurement_ids
                .iter()
                .map(|measurement_id| program.measurement_keys[*measurement_id].clone())
                .collect(),
            coords: detector_coords.next().cloned().unwrap_or_default(),
        });
    }
    detectors
}

/// Infer logical observable declarations from observable include operations.
pub fn observables_from_circuit(circuit: &Circuit) -> Vec<LogicalObservable> {
    let program = match crate::program::expand_operations(&circuit.operations, ExpansionMode::Dem) {
        Ok(program) => program,
        Err(_) => return Vec::new(),
    };
    observables_from_program(&program)
}

fn observables_from_program(program: &ExpandedProgram) -> Vec<LogicalObservable> {
    let mut measurement_ids_by_observable = HashMap::<i64, Vec<usize>>::new();
    for operation in &program.operations {
        let ExpandedOperation::ObservableInclude {
            observable_id,
            measurement_ids,
        } = operation
        else {
            continue;
        };
        measurement_ids_by_observable
            .entry(*observable_id)
            .or_default()
            .extend(measurement_ids);
    }
    let mut observable_ids = measurement_ids_by_observable
        .keys()
        .copied()
        .collect::<Vec<_>>();
    observable_ids.sort_unstable();
    observable_ids
        .into_iter()
        .map(|observable_id| LogicalObservable {
            id: observable_id,
            measurement_keys: measurement_ids_by_observable
                .remove(&observable_id)
                .unwrap_or_default()
                .into_iter()
                .map(|measurement_id| program.measurement_keys[measurement_id].clone())
                .collect(),
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

fn validate_observables_for_n_qubits(
    n_qubits: usize,
    observables: &[LogicalObservable],
) -> NpResult<()> {
    for observable in observables {
        observable.validate_for_n_qubits(n_qubits)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratedDemEdgeRef {
    event_index: usize,
    detectors: Vec<i64>,
    observables: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LazyDetectorErrorModel {
    pub detectors: Vec<Detector>,
    pub observables: Vec<LogicalObservable>,
    pub event_plan: Arc<DemEventPlan>,
    edges: Vec<GeneratedDemEdgeRef>,
}

impl LazyDetectorErrorModel {
    /// Test whether parity-reduced lazy DEM edges satisfy graphlike constraints.
    pub fn is_graphlike(&self) -> bool {
        self.edges.iter().all(|edge| {
            let detector_count = parity_support_len(&edge.detectors);
            detector_count <= 2
                && (detector_count != 0 || parity_support_len(&edge.observables) == 0)
        })
    }

    /// Compile directly into the compact graphlike representation without
    /// materializing location labels, events, or tag dictionaries.
    pub fn compile_graphlike_problem(&self) -> NpResult<GraphlikeDecodingProblem> {
        compile_graphlike_problem_from_edge_views(
            &self.detectors,
            &self.observables,
            self.edges.len(),
            || {
                self.edges
                    .iter()
                    .enumerate()
                    .map(|(edge_index, edge)| GraphlikeSourceEdge {
                        probability: self.event_plan.fault_events[edge.event_index].probability,
                        detectors: &edge.detectors,
                        observables: &edge.observables,
                        original_edge_index: edge_index,
                    })
            },
        )
    }

    /// Consume a lazy DEM into the metadata-free sampling representation.
    #[doc(hidden)]
    pub fn into_sampling_estimator(self) -> DemHotspotEstimator {
        let Self {
            detectors,
            observables,
            event_plan,
            edges,
        } = self;
        let program_edges = edges
            .into_iter()
            .map(|edge| DemProgramEdge {
                probability: event_plan.fault_events[edge.event_index].probability,
                detectors: edge.detectors,
                observables: edge.observables,
            })
            .collect();
        DemHotspotEstimator::from_compact_sampling_parts(
            detectors.into_iter().map(|detector| detector.id).collect(),
            observables
                .into_iter()
                .map(|observable| observable.id)
                .collect(),
            program_edges,
        )
    }

    /// Compile the lazy integer DEM directly into a sampler without
    /// materializing location labels or tag dictionaries.
    pub fn compile_hotspot_estimator(&self) -> DemHotspotEstimator {
        let mut edges = Vec::with_capacity(self.edges.len());
        let mut edge_metadata = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            let event = &self.event_plan.fault_events[edge.event_index];
            let location = &self.event_plan.program.noise_locations[event.noise_id];
            edges.push(DemProgramEdge {
                probability: event.probability,
                detectors: edge.detectors.clone(),
                observables: edge.observables.clone(),
            });
            edge_metadata.push(DemProgramEdgeMetadata {
                location_id: location.location_id,
                event: event.event.clone(),
            });
        }
        DemHotspotEstimator::from_program_parts(
            self.detectors.iter().map(|detector| detector.id).collect(),
            self.observables
                .iter()
                .map(|observable| observable.id)
                .collect(),
            edges,
            edge_metadata,
            self.event_plan.program.location_catalog.clone(),
        )
    }

    pub fn materialize(&self) -> DetectorErrorModel {
        DetectorErrorModel {
            detectors: self.detectors.clone(),
            observables: self.observables.clone(),
            edges: self
                .edges
                .iter()
                .map(|edge| {
                    let event = &self.event_plan.fault_events[edge.event_index];
                    let location = &self.event_plan.program.noise_locations[event.noise_id];
                    let catalog = &self.event_plan.program.location_catalog;
                    DetectorErrorEdge {
                        probability: event.probability,
                        detectors: edge.detectors.clone(),
                        observables: edge.observables.clone(),
                        location_id: catalog.label(location.location_id).to_string(),
                        event: event.event.clone(),
                        tags: catalog.tags(location.location_id).clone(),
                    }
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests;

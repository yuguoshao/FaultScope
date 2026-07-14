use std::collections::{HashMap, HashSet};
use std::sync::Arc;

mod assembly;
mod bitset;
mod event_plan;
mod fallback_path;
mod indexed_parity;
mod measurement_plan;
mod product_path;

use assembly::generated_edges_to_sampler_edges;
pub use event_plan::{collect_dem_event_plan, DemEventPlan};
use fallback_path::{
    generate_fallback_dem_edge_refs_from_plan, generate_fallback_dem_edges_from_plan,
    generate_fallback_sampling_edges_from_plan,
};
use measurement_plan::{compile_dem_measurement_plan, DemMeasurementPlan};
use product_path::{
    generate_indexed_product_dem_edge_refs_from_plan, generate_indexed_product_dem_edges_from_plan,
    generate_indexed_product_sampling_edges_from_plan, supports_product_reference_fast_path,
};

use crate::{
    Circuit, DemEvent, DemSamplerEdge, Detector, DetectorErrorEdge, DetectorErrorModel,
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
        let (circuit, _) = crate::program::expand_circuit(&circuit)?;
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
            compile_dem_measurement_plan(&circuit.operations, &detectors, &observables)?;
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
        generate_dem_edges_from_compiled_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.measurement_plan,
            &self.event_plan,
        )
    }

    fn generate_edge_refs(&self) -> NpResult<Vec<GeneratedDemEdgeRef>> {
        generate_dem_edge_refs_from_compiled_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.measurement_plan,
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
        generate_sampling_edges_from_compiled_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
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
) -> NpResult<Vec<GeneratedDemEdge>> {
    let expanded = crate::program::expand_operations(operations)?;
    let operations = expanded.operations;
    let event_plan = collect_dem_event_plan(&operations)?;
    generate_dem_edges_from_plan(n_qubits, &operations, detectors, observables, &event_plan)
}

pub fn generate_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdge>> {
    let measurement_plan = compile_dem_measurement_plan(operations, detectors, observables)?;
    generate_dem_edges_from_compiled_plan(n_qubits, operations, &measurement_plan, event_plan)
}

fn generate_dem_edges_from_compiled_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdge>> {
    if supports_product_reference_fast_path(n_qubits, operations) {
        return generate_indexed_product_dem_edges_from_plan(
            n_qubits,
            operations,
            measurement_plan,
            event_plan,
        );
    }
    generate_fallback_dem_edges_from_plan(n_qubits, operations, measurement_plan, event_plan)
}

fn generate_dem_edge_refs_from_compiled_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdgeRef>> {
    if supports_product_reference_fast_path(n_qubits, operations) {
        return generate_indexed_product_dem_edge_refs_from_plan(
            n_qubits,
            operations,
            measurement_plan,
            event_plan,
        );
    }
    generate_fallback_dem_edge_refs_from_plan(n_qubits, operations, measurement_plan, event_plan)
}

fn generate_sampling_edges_from_compiled_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<DemSamplerEdge>> {
    if supports_product_reference_fast_path(n_qubits, operations) {
        return generate_indexed_product_sampling_edges_from_plan(
            n_qubits,
            operations,
            measurement_plan,
            event_plan,
        );
    }
    generate_fallback_sampling_edges_from_plan(n_qubits, operations, measurement_plan, event_plan)
}

/// Infer detector declarations from detector operations in a circuit.
pub fn detectors_from_circuit(circuit: &Circuit) -> Vec<Detector> {
    let expanded = match crate::program::expand_operations(&circuit.operations) {
        Ok(expanded) => expanded.operations,
        Err(_) => return Vec::new(),
    };
    let mut detectors = Vec::new();
    for operation in &expanded {
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
    let expanded = match crate::program::expand_operations(&circuit.operations) {
        Ok(expanded) => expanded.operations,
        Err(_) => return Vec::new(),
    };
    let mut keys_by_id = HashMap::<i64, Vec<String>>::new();
    for operation in &expanded {
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

#[cfg(test)]
mod tests;

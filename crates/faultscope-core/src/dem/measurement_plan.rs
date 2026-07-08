use std::collections::{HashMap, HashSet};

use crate::{Detector, LogicalObservable, NpError, NpResult, Operation};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct IndexedDemDetector {
    pub(super) id: i64,
    pub(super) measurement_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct IndexedDemObservable {
    pub(super) id: i64,
    pub(super) measurement_indices: Vec<usize>,
    pub(super) pauli_qubits: Vec<usize>,
    pub(super) pauli: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DemMeasurementPlan {
    pub(super) measurement_count: usize,
    pub(super) measurement_indices_by_op: Vec<Option<usize>>,
    pub(super) detectors: Vec<IndexedDemDetector>,
    pub(super) observables: Vec<IndexedDemObservable>,
}

pub(super) fn compile_dem_measurement_plan(
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

pub(super) fn optional_indexed_measurement_op(
    measurement_plan: &DemMeasurementPlan,
    op_index: usize,
) -> Option<usize> {
    measurement_plan
        .measurement_indices_by_op
        .get(op_index)
        .and_then(|index| *index)
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

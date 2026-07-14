use std::collections::HashMap;

use crate::program::{ExpandedOperation, ExpandedProgram};
use crate::{Detector, LogicalObservable, NpError, NpResult, SamplerObservable};

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
    pub(super) measurement_index_by_id: Vec<Option<usize>>,
    pub(super) detectors: Vec<IndexedDemDetector>,
    pub(super) observables: Vec<IndexedDemObservable>,
}

struct DetectorMeasurementIds {
    id: i64,
    measurement_ids: Vec<usize>,
}

pub(super) fn compile_dem_measurement_plan(
    program: &ExpandedProgram,
    detectors: &[Detector],
    observables: &[LogicalObservable],
) -> NpResult<DemMeasurementPlan> {
    compile_dem_measurement_plan_with_optional_declarations(
        program,
        Some(detectors),
        Some(observables),
    )
}

pub(super) fn compile_dem_measurement_plan_with_optional_declarations(
    program: &ExpandedProgram,
    detectors: Option<&[Detector]>,
    observables: Option<&[LogicalObservable]>,
) -> NpResult<DemMeasurementPlan> {
    let measurement_id_by_key = (detectors.is_some() || observables.is_some()).then(|| {
        program
            .measurement_keys
            .iter()
            .enumerate()
            .map(|(measurement_id, key)| (key.as_str(), measurement_id))
            .collect::<HashMap<_, _>>()
    });
    let detectors = match detectors {
        Some(detectors) => detectors
            .iter()
            .map(|detector| {
                Ok(DetectorMeasurementIds {
                    id: detector.id,
                    measurement_ids: measurement_ids_for_keys(
                        measurement_id_by_key
                            .as_ref()
                            .expect("explicit declarations require a key index"),
                        &detector.measurement_keys,
                    )?,
                })
            })
            .collect::<NpResult<Vec<_>>>()?,
        None => inferred_detector_measurement_ids(program),
    };
    let observables = match observables {
        Some(observables) => observables
            .iter()
            .map(|observable| {
                Ok(SamplerObservable {
                    id: observable.id,
                    measurement_ids: measurement_ids_for_keys(
                        measurement_id_by_key
                            .as_ref()
                            .expect("explicit declarations require a key index"),
                        &observable.measurement_keys,
                    )?,
                    pauli_qubits: observable.pauli_qubits.clone(),
                    pauli: observable.pauli.clone(),
                })
            })
            .collect::<NpResult<Vec<_>>>()?,
        None => inferred_observable_measurement_ids(program),
    };

    Ok(compile_dem_measurement_plan_from_ids(
        program,
        detectors,
        observables,
    ))
}

fn inferred_detector_measurement_ids(program: &ExpandedProgram) -> Vec<DetectorMeasurementIds> {
    let mut detectors = Vec::new();
    for operation in &program.operations {
        if let ExpandedOperation::Detector {
            detector_id,
            measurement_ids,
        } = operation
        {
            detectors.push(DetectorMeasurementIds {
                id: *detector_id,
                measurement_ids: measurement_ids.clone(),
            });
        }
    }
    detectors
}

fn inferred_observable_measurement_ids(program: &ExpandedProgram) -> Vec<SamplerObservable> {
    let mut observables_by_id = HashMap::<i64, Vec<usize>>::new();
    for operation in &program.operations {
        if let ExpandedOperation::ObservableInclude {
            observable_id,
            measurement_ids,
        } = operation
        {
            observables_by_id
                .entry(*observable_id)
                .or_default()
                .extend(measurement_ids);
        }
    }
    observables_by_id
        .into_iter()
        .map(|(id, measurement_ids)| SamplerObservable {
            id,
            measurement_ids,
            pauli_qubits: Vec::new(),
            pauli: String::new(),
        })
        .collect()
}

fn compile_dem_measurement_plan_from_ids(
    program: &ExpandedProgram,
    detectors: Vec<DetectorMeasurementIds>,
    observables: Vec<SamplerObservable>,
) -> DemMeasurementPlan {
    let mut required_measurements = vec![false; program.measurement_keys.len()];
    for measurement_id in detectors
        .iter()
        .flat_map(|detector| &detector.measurement_ids)
        .chain(
            observables
                .iter()
                .flat_map(|observable| &observable.measurement_ids),
        )
    {
        required_measurements[*measurement_id] = true;
    }

    let mut measurement_index_by_id = vec![None; program.measurement_keys.len()];
    let mut indexed_measurement_count = 0usize;

    for (measurement_id, required) in required_measurements.into_iter().enumerate() {
        if required {
            measurement_index_by_id[measurement_id] = Some(indexed_measurement_count);
            indexed_measurement_count += 1;
        }
    }

    let mut indexed_detectors = detectors
        .into_iter()
        .map(|detector| IndexedDemDetector {
            id: detector.id,
            measurement_indices: measurement_indices_for_ids(
                &measurement_index_by_id,
                &detector.measurement_ids,
            ),
        })
        .collect::<Vec<_>>();
    indexed_detectors.sort_by_key(|detector| detector.id);

    let mut indexed_observables = observables
        .into_iter()
        .map(|observable| IndexedDemObservable {
            id: observable.id,
            measurement_indices: measurement_indices_for_ids(
                &measurement_index_by_id,
                &observable.measurement_ids,
            ),
            pauli_qubits: observable.pauli_qubits,
            pauli: observable.pauli,
        })
        .collect::<Vec<_>>();
    indexed_observables.sort_by_key(|observable| observable.id);

    DemMeasurementPlan {
        measurement_count: indexed_measurement_count,
        measurement_index_by_id,
        detectors: indexed_detectors,
        observables: indexed_observables,
    }
}

pub(super) fn optional_indexed_measurement(
    measurement_plan: &DemMeasurementPlan,
    measurement_id: usize,
) -> Option<usize> {
    measurement_plan
        .measurement_index_by_id
        .get(measurement_id)
        .and_then(|index| *index)
}

fn measurement_ids_for_keys(
    measurement_id_by_key: &HashMap<&str, usize>,
    keys: &[String],
) -> NpResult<Vec<usize>> {
    keys.iter()
        .map(|key| {
            measurement_id_by_key
                .get(key.as_str())
                .copied()
                .ok_or_else(|| NpError::new(format!("unknown measurement key {key:?}")))
        })
        .collect()
}

fn measurement_indices_for_ids(
    measurement_index_by_id: &[Option<usize>],
    measurement_ids: &[usize],
) -> Vec<usize> {
    measurement_ids
        .iter()
        .map(|measurement_id| {
            measurement_index_by_id[*measurement_id]
                .expect("required DEM measurement must have a dense index")
        })
        .collect()
}

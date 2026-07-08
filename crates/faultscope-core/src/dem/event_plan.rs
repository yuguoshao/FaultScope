use std::collections::{HashMap, HashSet};

use crate::{DemEvent, NoiseLocation, NoiseModel, NpError, NpResult, Operation};

#[derive(Debug, Clone, PartialEq)]
pub struct DemEventPlan {
    pub(super) fault_events: Vec<DemFaultEvent>,
    pub(super) fault_events_by_op: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DemFaultEvent {
    pub(super) location_id: String,
    pub(super) qubits: Vec<usize>,
    pub(super) event: DemEvent,
    pub(super) probability: f64,
    pub(super) tags: HashMap<String, crate::TagValue>,
}

pub fn collect_dem_event_plan(operations: &[Operation]) -> NpResult<DemEventPlan> {
    let (fault_events, fault_events_by_op) = collect_dem_fault_events(operations)?;
    Ok(DemEventPlan {
        fault_events,
        fault_events_by_op,
    })
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

use std::ops::Range;
use std::sync::Arc;

use crate::program::{ExpandedOperation, ExpandedProgram, ExpansionMode};
use crate::{DemEvent, IndexedNoiseLocation, NoiseModel, NpError, NpResult, Operation};

#[derive(Debug, Clone, PartialEq)]
pub struct DemEventPlan {
    pub(super) program: Arc<ExpandedProgram>,
    pub(super) fault_events: Vec<DemFaultEvent>,
    pub(super) fault_event_range_by_noise: Vec<Range<usize>>,
}

impl DemEventPlan {
    pub(super) fn measurement_noise_event_range(&self, noise_id: Option<usize>) -> Range<usize> {
        let range = noise_id
            .map(|noise_id| self.fault_event_range_by_noise[noise_id].clone())
            .unwrap_or(0..0);
        debug_assert!(range.clone().all(|event_index| matches!(
            &self.fault_events[event_index].event,
            DemEvent::Bool(true)
        )));
        range
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DemFaultEvent {
    pub(super) noise_id: usize,
    pub(super) event: DemEvent,
    pub(super) probability: f64,
}

pub fn collect_dem_event_plan(operations: &[Operation]) -> NpResult<DemEventPlan> {
    let program = crate::program::expand_operations(operations, ExpansionMode::Dem)?;
    collect_dem_event_plan_from_program(program)
}

pub(super) fn collect_dem_event_plan_from_program(
    program: ExpandedProgram,
) -> NpResult<DemEventPlan> {
    let program = Arc::new(program);
    let (fault_events, fault_event_range_by_noise) = collect_dem_fault_events(&program)?;
    Ok(DemEventPlan {
        program,
        fault_events,
        fault_event_range_by_noise,
    })
}

fn collect_dem_fault_events(
    program: &ExpandedProgram,
) -> NpResult<(Vec<DemFaultEvent>, Vec<Range<usize>>)> {
    let mut fault_events = Vec::new();
    let mut fault_event_range_by_noise = vec![0..0; program.noise_locations.len()];
    for operation in &program.operations {
        match operation {
            ExpandedOperation::Noise(noise_id) => {
                fault_event_range_by_noise[*noise_id] = collect_dem_fault_events_for_location(
                    &mut fault_events,
                    *noise_id,
                    &program.noise_locations[*noise_id],
                )?;
            }
            ExpandedOperation::MeasureSingle {
                noise: Some(noise_id),
                ..
            }
            | ExpandedOperation::MeasurePauli {
                noise: Some(noise_id),
                ..
            } => {
                let location = &program.noise_locations[*noise_id];
                if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
                    return Err(NpError::new(
                        "DEM generation currently supports MeasurementBitFlip on measurement operations",
                    ));
                }
                fault_event_range_by_noise[*noise_id] =
                    collect_dem_fault_events_for_location(&mut fault_events, *noise_id, location)?;
            }
            _ => {}
        }
    }
    Ok((fault_events, fault_event_range_by_noise))
}

fn collect_dem_fault_events_for_location(
    fault_events: &mut Vec<DemFaultEvent>,
    noise_id: usize,
    location: &IndexedNoiseLocation,
) -> NpResult<Range<usize>> {
    let start = fault_events.len();
    for (event, probability) in non_identity_events(location)? {
        fault_events.push(DemFaultEvent {
            noise_id,
            event,
            probability,
        });
    }
    Ok(start..fault_events.len())
}

fn non_identity_events(location: &IndexedNoiseLocation) -> NpResult<Vec<(DemEvent, f64)>> {
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

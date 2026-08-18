use std::ops::Range;
use std::sync::Arc;

use crate::program::{ExpandedOperation, ExpandedProgram, ExpansionMode};
use crate::{DemEvent, IndexedNoiseLocation, NoiseModel, NpError, NpResult, Operation};

/// Controls circuit-to-DEM conversions that cannot preserve a disjoint noise
/// channel exactly using independent DEM instructions.
///
/// The default forbids fallback approximation. A positive threshold opts into
/// Stim-compatible independent approximation of multi-component
/// [`NoiseModel::PauliChannel`] locations when every component probability is
/// at most the threshold. Stim's exact one-qubit conversion is attempted first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemGenerationOptions {
    approximate_disjoint_errors_threshold: f64,
}

impl DemGenerationOptions {
    /// Construct strict DEM-generation options.
    pub const fn strict() -> Self {
        Self {
            approximate_disjoint_errors_threshold: 0.0,
        }
    }

    /// Construct options permitting independent approximation up to `threshold`.
    pub fn new(threshold: f64) -> NpResult<Self> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(NpError::new(format!(
                "approximate_disjoint_errors must be a finite probability in [0, 1], got {threshold}"
            )));
        }
        Ok(Self {
            approximate_disjoint_errors_threshold: threshold,
        })
    }

    /// Largest disjoint component probability accepted for approximation.
    pub const fn approximate_disjoint_errors_threshold(self) -> f64 {
        self.approximate_disjoint_errors_threshold
    }
}

impl Default for DemGenerationOptions {
    fn default() -> Self {
        Self::strict()
    }
}

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
    pub(super) sibling_semantics: DemFaultEventSiblingSemantics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DemFaultEventSiblingSemantics {
    None,
    Independent,
    Disjoint,
}

pub fn collect_dem_event_plan(operations: &[Operation]) -> NpResult<DemEventPlan> {
    collect_dem_event_plan_with_options(operations, DemGenerationOptions::default())
}

/// Compile a DEM event plan with explicit disjoint-channel handling options.
pub fn collect_dem_event_plan_with_options(
    operations: &[Operation],
    options: DemGenerationOptions,
) -> NpResult<DemEventPlan> {
    let program = crate::program::expand_operations(operations, ExpansionMode::Dem)?;
    collect_dem_event_plan_from_program_with_options(program, options)
}

pub(super) fn collect_dem_event_plan_from_program_with_options(
    program: ExpandedProgram,
    options: DemGenerationOptions,
) -> NpResult<DemEventPlan> {
    let program = Arc::new(program);
    let (fault_events, fault_event_range_by_noise) = collect_dem_fault_events(&program, options)?;
    Ok(DemEventPlan {
        program,
        fault_events,
        fault_event_range_by_noise,
    })
}

fn collect_dem_fault_events(
    program: &ExpandedProgram,
    options: DemGenerationOptions,
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
                    options,
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
                fault_event_range_by_noise[*noise_id] = collect_dem_fault_events_for_location(
                    &mut fault_events,
                    *noise_id,
                    location,
                    options,
                )?;
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
    options: DemGenerationOptions,
) -> NpResult<Range<usize>> {
    let start = fault_events.len();
    for (event, probability, sibling_semantics) in non_identity_events(location, options)? {
        fault_events.push(DemFaultEvent {
            noise_id,
            event,
            probability,
            sibling_semantics,
        });
    }
    Ok(start..fault_events.len())
}

fn non_identity_events(
    location: &IndexedNoiseLocation,
    options: DemGenerationOptions,
) -> NpResult<Vec<(DemEvent, f64, DemFaultEventSiblingSemantics)>> {
    match &location.model {
        NoiseModel::BernoulliPauli(pauli) => Ok(vec![(
            DemEvent::Pauli(pauli.clone()),
            location.rate,
            DemFaultEventSiblingSemantics::None,
        )]),
        NoiseModel::MeasurementBitFlip => Ok(vec![(
            DemEvent::Bool(true),
            location.rate,
            DemFaultEventSiblingSemantics::None,
        )]),
        NoiseModel::SingleQubitDepolarizing => {
            let probability = independent_depolarizing_component_probability(
                location.rate,
                4,
                "SingleQubitDepolarizing",
            )?;
            Ok(["X", "Y", "Z"]
                .iter()
                .map(|event| {
                    (
                        DemEvent::Pauli((*event).to_string()),
                        probability,
                        DemFaultEventSiblingSemantics::Independent,
                    )
                })
                .collect())
        }
        NoiseModel::TwoQubitDepolarizing => {
            let fault_events = [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY",
                "ZZ",
            ];
            let probability = independent_depolarizing_component_probability(
                location.rate,
                16,
                "TwoQubitDepolarizing",
            )?;
            Ok(fault_events
                .iter()
                .map(|event| {
                    (
                        DemEvent::Pauli((*event).to_string()),
                        probability,
                        DemFaultEventSiblingSemantics::Independent,
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
            // A PauliChannel is categorical in the forward sampler.  DEM
            // instructions are independent, so match Stim by requiring either
            // its exact one-qubit conversion or explicit fallback
            // approximation. Combine duplicate Pauli labels first so a Rust
            // caller cannot turn one physical outcome into a false
            // multi-component channel merely by repeating its label.
            let mut event_weights: Vec<(String, f64)> = Vec::new();
            for (event, weight) in weights.iter().filter(|(_, weight)| *weight > 0.0) {
                if let Some((_, existing_weight)) = event_weights
                    .iter_mut()
                    .find(|(existing_event, _)| existing_event == event)
                {
                    *existing_weight += *weight;
                } else {
                    event_weights.push((event.clone(), *weight));
                }
            }
            let events = event_weights
                .into_iter()
                .map(|(event, weight)| (event, location.rate * (weight / total)))
                .collect::<Vec<_>>();

            let positive_probability_count = if location.rate > 0.0 { events.len() } else { 0 };
            if positive_probability_count <= 1 {
                return Ok(events
                    .into_iter()
                    .map(|(event, probability)| {
                        (
                            DemEvent::Pauli(event),
                            probability,
                            DemFaultEventSiblingSemantics::None,
                        )
                    })
                    .collect());
            }

            // Stim first tries to turn PAULI_CHANNEL_1 into independent X, Y,
            // and Z mechanisms exactly.  PAULI_CHANNEL_2 (and FaultScope's
            // wider generic PauliChannel) does not have this special case.
            if events[0].0.len() == 1 {
                let mut disjoint = [0.0; 3];
                for (event, probability) in &events {
                    let index = match event.as_str() {
                        "X" => 0,
                        "Y" => 1,
                        "Z" => 2,
                        _ => unreachable!("validated one-qubit PauliChannel event"),
                    };
                    disjoint[index] = *probability;
                }
                if let Some(independent) =
                    try_disjoint_to_independent_xyz_errors(disjoint[0], disjoint[1], disjoint[2])
                {
                    return Ok(["X", "Y", "Z"]
                        .into_iter()
                        .zip(independent)
                        .filter(|(_, probability)| *probability > 0.0)
                        .map(|(event, probability)| {
                            (
                                DemEvent::Pauli(event.to_string()),
                                probability,
                                DemFaultEventSiblingSemantics::Independent,
                            )
                        })
                        .collect());
                }
            }

            let largest_probability = events
                .iter()
                .map(|(_, probability)| *probability)
                .fold(0.0_f64, f64::max);
            let threshold = options.approximate_disjoint_errors_threshold();
            let comparison_slack = 8.0 * f64::EPSILON * largest_probability.max(threshold);
            let exceeds_threshold = if threshold == 0.0 {
                true
            } else {
                largest_probability - threshold > comparison_slack
            };
            if exceeds_threshold {
                return Err(NpError::new(format!(
                    "PauliChannel has multiple disjoint outcomes, including component probability {largest_probability}, which cannot be represented exactly by independent DEM edges; set approximate_disjoint_errors=True, or set it to a probability threshold at least {largest_probability}, to enable Stim-compatible approximation"
                )));
            }

            Ok(events
                .into_iter()
                .map(|(event, probability)| {
                    (
                        DemEvent::Pauli(event),
                        probability,
                        DemFaultEventSiblingSemantics::Disjoint,
                    )
                })
                .collect())
        }
    }
}

/// Try Stim's exact PAULI_CHANNEL_1 conversion into independent X/Y/Z errors.
///
/// `None` means no sufficiently accurate independent representation was found;
/// callers may then use the explicitly enabled disjoint approximation.
fn try_disjoint_to_independent_xyz_errors(x: f64, y: f64, z: f64) -> Option<[f64; 3]> {
    fn solve(x: f64, y: f64, z: f64, max_steps: usize) -> Option<[f64; 3]> {
        let identity = (1.0 - x - y - z).max(0.0);
        if identity < x {
            let [independent_x, independent_y, independent_z] = solve(identity, z, y, max_steps)?;
            return Some([1.0 - independent_x, independent_y, independent_z]);
        }
        if identity < y {
            let [independent_x, independent_y, independent_z] = solve(z, identity, x, max_steps)?;
            return Some([independent_x, 1.0 - independent_y, independent_z]);
        }
        if identity < z {
            let [independent_x, independent_y, independent_z] = solve(y, x, identity, max_steps)?;
            return Some([independent_x, independent_y, 1.0 - independent_z]);
        }

        if x + z < 0.5 && x + y < 0.5 && y + z < 0.5 {
            let sqrt_xz = (1.0 - 2.0 * x - 2.0 * z).sqrt();
            let sqrt_xy = (1.0 - 2.0 * x - 2.0 * y).sqrt();
            let sqrt_yz = (1.0 - 2.0 * y - 2.0 * z).sqrt();
            let independent_x = 0.5 - 0.5 * sqrt_xz * sqrt_xy / sqrt_yz;
            let independent_y = 0.5 - 0.5 * sqrt_xy * sqrt_yz / sqrt_xz;
            let independent_z = 0.5 - 0.5 * sqrt_xz * sqrt_yz / sqrt_xy;
            if independent_x >= 0.0 && independent_y >= 0.0 && independent_z >= 0.0 {
                return Some([independent_x, independent_y, independent_z]);
            }
        }

        let mut independent_x = x;
        let mut independent_y = y;
        let mut independent_z = z;
        for _ in 0..max_steps {
            let xy = independent_x * independent_y;
            let xz = independent_x * independent_z;
            let yz = independent_y * independent_z;
            let not_x = 1.0 - independent_x;
            let not_y = 1.0 - independent_y;
            let not_z = 1.0 - independent_z;
            let not_xy = not_x * not_y;
            let not_xz = not_x * not_z;
            let not_yz = not_y * not_z;
            let actual_x = independent_x * not_yz + not_x * yz;
            let actual_y = independent_y * not_xz + not_y * xz;
            let actual_z = independent_z * not_xy + not_z * xy;
            let error_x = actual_x - x;
            let error_y = actual_y - y;
            let error_z = actual_z - z;
            if error_x.abs() + error_y.abs() + error_z.abs() < 1e-14 {
                return Some([independent_x, independent_y, independent_z]);
            }

            let derivative_x = not_yz - yz;
            let derivative_y = not_xz - xz;
            let derivative_z = not_xy - xy;
            independent_x -= error_x / derivative_x;
            independent_y -= error_y / derivative_y;
            independent_z -= error_z / derivative_z;
            if independent_x < 0.0 {
                independent_x = 0.0;
            }
            if independent_y < 0.0 {
                independent_y = 0.0;
            }
            if independent_z < 0.0 {
                independent_z = 0.0;
            }
        }
        None
    }

    solve(x, y, z, 50)
}

/// Reparameterize an n-qubit uniform depolarizing channel into independent
/// Bernoulli factors over every non-identity Pauli, as Stim does during DEM
/// analysis.  `group_size` is 4^n.
fn independent_depolarizing_component_probability(
    rate: f64,
    group_size: usize,
    model_name: &str,
) -> NpResult<f64> {
    let group_size = group_size as f64;
    let maximum_exact_rate = (group_size - 1.0) / group_size;
    if rate > maximum_exact_rate {
        return Err(NpError::new(format!(
            "{model_name} rate {rate} exceeds {maximum_exact_rate}, the largest rate representable exactly by independent DEM edges"
        )));
    }
    if rate == 0.0 {
        return Ok(0.0);
    }

    // lambda = 1 - group_size * rate / (group_size - 1) is every
    // non-trivial Fourier coefficient of the categorical depolarizing
    // distribution.  Independent factors give
    // lambda = (1 - 2q)^(group_size / 2).  ln_1p/exp_m1 retain accuracy for
    // very small rates.
    let log_lambda = (-group_size * rate / (group_size - 1.0)).ln_1p();
    Ok(-0.5 * ((2.0 / group_size) * log_lambda).exp_m1())
}

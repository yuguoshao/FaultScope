use std::collections::HashMap;

use super::event_plan::{DemEventPlan, DemFaultEventSiblingSemantics};
use super::{DetectorErrorEdge, GeneratedDemEdgeOverride, GeneratedDemEdgeRef};

pub(super) struct DemFlipMasks {
    pub(super) detector_flip_masks: Vec<(i64, Vec<u64>)>,
    pub(super) observable_flip_masks: Vec<(i64, Vec<u64>)>,
}

pub(super) fn assemble_dem_edge_refs(
    event_count: usize,
    flip_masks: DemFlipMasks,
) -> Vec<GeneratedDemEdgeRef> {
    let mut active_event_words = vec![0u64; event_count.div_ceil(64)];
    for (_, words) in flip_masks
        .detector_flip_masks
        .iter()
        .chain(&flip_masks.observable_flip_masks)
    {
        for (active, word) in active_event_words.iter_mut().zip(words) {
            *active |= *word;
        }
    }

    let active_event_count = active_event_words
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum();
    let mut edge_index_by_event = vec![usize::MAX; event_count];
    let mut edges = Vec::with_capacity(active_event_count);
    for (word_index, word) in active_event_words.into_iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let event_index = word_index * 64 + bit;
            if event_index < event_count {
                edge_index_by_event[event_index] = edges.len();
                edges.push(GeneratedDemEdgeRef {
                    event_index,
                    detectors: Vec::new(),
                    observables: Vec::new(),
                });
            }
            remaining &= remaining - 1;
        }
    }

    append_flip_ids(
        &mut edges,
        &edge_index_by_event,
        &flip_masks.detector_flip_masks,
        FlipTarget::Detector,
    );
    append_flip_ids(
        &mut edges,
        &edge_index_by_event,
        &flip_masks.observable_flip_masks,
        FlipTarget::Observable,
    );
    edges
}

pub(super) fn materialize_generated_dem_edges(
    event_plan: &DemEventPlan,
    edges: Vec<GeneratedDemEdgeRef>,
    edge_overrides: &HashMap<usize, GeneratedDemEdgeOverride>,
) -> Vec<DetectorErrorEdge> {
    let catalog = &event_plan.program.location_catalog;
    edges
        .into_iter()
        .enumerate()
        .map(|(edge_index, edge)| {
            let event = &event_plan.fault_events[edge.event_index];
            let location = &event_plan.program.noise_locations[event.noise_id];
            DetectorErrorEdge {
                probability: edge_overrides
                    .get(&edge_index)
                    .map(|override_| override_.probability)
                    .unwrap_or(event.probability),
                detectors: edge.detectors,
                observables: edge.observables,
                location_id: catalog.label(location.location_id).to_string(),
                event: edge_overrides
                    .get(&edge_index)
                    .map(|override_| override_.event.clone())
                    .unwrap_or_else(|| event.event.clone()),
                tags: catalog.tags(location.location_id).clone(),
            }
        })
        .collect()
}

/// Coalesce sibling events when propagation gives them exactly the same
/// detector/observable effect.
///
/// Stim sums mutually exclusive PauliChannel probabilities and parity-combines
/// independent depolarizing factors before treating distinct effect classes as
/// independent DEM instructions.
pub(super) fn coalesce_sibling_dem_edge_refs(
    event_plan: &DemEventPlan,
    input_edges: Vec<GeneratedDemEdgeRef>,
) -> (
    Vec<GeneratedDemEdgeRef>,
    HashMap<usize, GeneratedDemEdgeOverride>,
) {
    let mut output_edges = Vec::with_capacity(input_edges.len());
    let mut event_groups: Vec<Option<Vec<usize>>> = Vec::with_capacity(input_edges.len());
    let mut output_index_by_effect: HashMap<(usize, Vec<i64>, Vec<i64>), usize> = HashMap::new();
    let mut merged = false;

    for edge in input_edges {
        let event = &event_plan.fault_events[edge.event_index];
        if event.sibling_semantics == DemFaultEventSiblingSemantics::None {
            output_edges.push(edge);
            event_groups.push(None);
            continue;
        }

        let key = (
            event.noise_id,
            edge.detectors.clone(),
            edge.observables.clone(),
        );
        if let Some(&output_index) = output_index_by_effect.get(&key) {
            event_groups[output_index]
                .as_mut()
                .expect("sibling effect map must point to a sibling event group")
                .push(edge.event_index);
            merged = true;
        } else {
            output_index_by_effect.insert(key, output_edges.len());
            event_groups.push(Some(vec![edge.event_index]));
            output_edges.push(edge);
        }
    }

    if !merged {
        return (output_edges, HashMap::new());
    }

    let mut edge_overrides = HashMap::new();
    for (output_index, event_indices) in event_groups.into_iter().enumerate() {
        let Some(event_indices) = event_indices.filter(|indices| indices.len() > 1) else {
            continue;
        };
        let sibling_semantics = event_plan.fault_events[event_indices[0]].sibling_semantics;
        let probability = match sibling_semantics {
            DemFaultEventSiblingSemantics::Independent => {
                event_indices.iter().fold(0.0, |combined, event_index| {
                    let probability = event_plan.fault_events[*event_index].probability;
                    combined + probability - 2.0 * combined * probability
                })
            }
            DemFaultEventSiblingSemantics::Disjoint => event_indices
                .iter()
                .map(|event_index| event_plan.fault_events[*event_index].probability)
                .sum::<f64>()
                .min(1.0),
            DemFaultEventSiblingSemantics::None => {
                unreachable!("non-sibling events are not assigned to event groups")
            }
        };
        let labels = event_indices
            .iter()
            .map(
                |event_index| match &event_plan.fault_events[*event_index].event {
                    crate::DemEvent::Pauli(pauli) => pauli.as_str(),
                    crate::DemEvent::Bool(_) => {
                        unreachable!("only PauliChannel events can be disjoint siblings")
                    }
                },
            )
            .collect::<Vec<_>>();
        edge_overrides.insert(
            output_index,
            GeneratedDemEdgeOverride {
                probability,
                event: crate::DemEvent::Pauli(labels.join(match sibling_semantics {
                    DemFaultEventSiblingSemantics::Independent => "^",
                    DemFaultEventSiblingSemantics::Disjoint => "|",
                    DemFaultEventSiblingSemantics::None => unreachable!(),
                })),
            },
        );
    }

    (output_edges, edge_overrides)
}

#[derive(Clone, Copy)]
enum FlipTarget {
    Detector,
    Observable,
}

fn append_flip_ids(
    edges: &mut [GeneratedDemEdgeRef],
    edge_index_by_event: &[usize],
    flip_masks: &[(i64, Vec<u64>)],
    target: FlipTarget,
) {
    for (id, words) in flip_masks {
        for (word_index, word) in words.iter().enumerate() {
            let mut remaining = *word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let event_index = word_index * 64 + bit;
                if let Some(&edge_index) = edge_index_by_event.get(event_index) {
                    debug_assert_ne!(edge_index, usize::MAX);
                    let ids = match target {
                        FlipTarget::Detector => &mut edges[edge_index].detectors,
                        FlipTarget::Observable => &mut edges[edge_index].observables,
                    };
                    ids.push(*id);
                }
                remaining &= remaining - 1;
            }
        }
    }
}

use std::collections::HashMap;

use super::event_plan::DemFaultEvent;
use super::{GeneratedDemEdge, GeneratedDemEdgeRef};
use crate::{DemEvent, DemSamplerEdge};

pub(super) fn generated_edges_to_sampler_edges(
    generated_edges: Vec<GeneratedDemEdge>,
) -> Vec<DemSamplerEdge> {
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

pub(super) fn assemble_generated_dem_edges_from_flat_flip_masks(
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

pub(super) fn assemble_dem_edge_refs_from_flat_flip_masks(
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

pub(super) fn assemble_sampling_edges_from_flat_flip_masks(
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

pub(super) fn flat_flip_ids_by_fault_event(
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

use super::event_plan::DemEventPlan;
use super::{DetectorErrorEdge, GeneratedDemEdgeRef};

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
) -> Vec<DetectorErrorEdge> {
    let catalog = &event_plan.program.location_catalog;
    edges
        .into_iter()
        .map(|edge| {
            let event = &event_plan.fault_events[edge.event_index];
            let location = &event_plan.program.noise_locations[event.noise_id];
            DetectorErrorEdge {
                probability: event.probability,
                detectors: edge.detectors,
                observables: edge.observables,
                location_id: catalog.label(location.location_id).to_string(),
                event: event.event.clone(),
                tags: catalog.tags(location.location_id).clone(),
            }
        })
        .collect()
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

use std::collections::HashMap;

use crate::{
    DemBatch, DemHotspotEstimate, DemLocationGroup, DemSamplerEdge, DetectorGraphEstimate,
    DetectorGraphKey, HotspotEstimate, Mask, NoiseLocation, RuntimeState, TagValue,
};

pub fn compute_packed_estimate(
    locations: &[NoiseLocation],
    state: &RuntimeState,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> HotspotEstimate {
    let mut clipped_loss = loss_mask.clone();
    clipped_loss.and_assign(&state.all_mask);
    let loss_count = clipped_loss.bit_count();
    let mean_loss = loss_count as f64 / state.shots as f64;
    let baseline_value = baseline.unwrap_or(mean_loss);
    let mut sensitivities = HashMap::new();
    let mut hotspots = HashMap::new();
    let mut by_qubit = HashMap::new();

    for location in locations {
        let zero_mask;
        let event_mask = if let Some(mask) = state.event_masks.get(&location.id) {
            mask
        } else {
            zero_mask = Mask::zero(state.all_mask.words.len());
            &zero_mask
        };
        let event_count = event_mask.bit_count();
        let no_event_count = state.shots - event_count;
        let loss_event_count = clipped_loss.and_count(event_mask);
        let loss_no_event_count = loss_count - loss_event_count;
        let (event_score, no_event_score) = score_pair(location.rate);
        let sum_loss_score =
            loss_event_count as f64 * event_score + loss_no_event_count as f64 * no_event_score;
        let sum_score = event_count as f64 * event_score + no_event_count as f64 * no_event_score;
        let sensitivity = (sum_loss_score - baseline_value * sum_score) / state.shots as f64;
        let hotspot = sensitivity.abs();
        sensitivities.insert(location.id.clone(), sensitivity);
        hotspots.insert(location.id.clone(), hotspot);
        for qubit in &location.qubits {
            add_f64(&mut by_qubit, *qubit, hotspot);
        }
    }

    let by_round = aggregate_location_tag_hotspots(locations, &hotspots, "round");
    let by_gate = aggregate_location_tag_hotspots(locations, &hotspots, "gate");
    let by_operation = aggregate_location_tag_hotspots(locations, &hotspots, "operation");
    let top_locations = top_location_ids(&hotspots, top_k);

    HotspotEstimate {
        shots: state.shots,
        mean_loss,
        baseline: baseline_value,
        sensitivities,
        hotspots,
        by_qubit,
        by_round,
        by_gate,
        by_operation,
        top_locations,
    }
}

pub fn compute_dem_estimate(
    edges: &[DemSamplerEdge],
    location_groups: &[DemLocationGroup],
    batch: &DemBatch,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> DemHotspotEstimate {
    let mut clipped_loss = loss_mask.clone();
    clipped_loss.and_assign(&batch.all_mask);
    let loss_count = clipped_loss.bit_count();
    let mean_loss = loss_count as f64 / batch.shots as f64;
    let baseline_value = baseline.unwrap_or(mean_loss);
    let mut edge_sensitivities = Vec::with_capacity(edges.len());
    for (edge_index, edge) in edges.iter().enumerate() {
        let event_mask = &batch.edge_event_masks[edge_index];
        let event_count = event_mask.bit_count();
        let no_event_count = batch.shots - event_count;
        let loss_event_count = clipped_loss.and_count(event_mask);
        let loss_no_event_count = loss_count - loss_event_count;
        let (event_score, no_event_score) = score_pair(edge.probability);
        let sum_loss_score =
            loss_event_count as f64 * event_score + loss_no_event_count as f64 * no_event_score;
        let sum_score = event_count as f64 * event_score + no_event_count as f64 * no_event_score;
        edge_sensitivities.push((sum_loss_score - baseline_value * sum_score) / batch.shots as f64);
    }

    let location_sensitivities =
        aggregate_dem_location_sensitivities(edges, location_groups, &edge_sensitivities);
    let edge_hotspots = edge_sensitivities
        .iter()
        .map(|sensitivity| sensitivity.abs())
        .collect::<Vec<_>>();
    let location_hotspots = location_sensitivities
        .iter()
        .map(|(location_id, sensitivity)| (location_id.clone(), sensitivity.abs()))
        .collect::<HashMap<_, _>>();
    let by_detector = aggregate_dem_detector_hotspots(edges, &edge_hotspots);
    let by_round = aggregate_dem_tag_hotspots(edges, &location_hotspots, "round");
    let by_gate = aggregate_dem_tag_hotspots(edges, &location_hotspots, "gate");
    let by_operation = aggregate_dem_tag_hotspots(edges, &location_hotspots, "operation");
    let detector_graph = compute_detector_graph_estimate(edges, &edge_sensitivities);
    let top_edges = top_edge_indices(&edge_hotspots, top_k);
    let top_locations = top_location_ids(&location_hotspots, top_k);

    DemHotspotEstimate {
        shots: batch.shots,
        mean_loss,
        baseline: baseline_value,
        edge_sensitivities,
        edge_hotspots,
        location_sensitivities,
        location_hotspots,
        by_detector,
        by_round,
        by_gate,
        by_operation,
        detector_graph,
        top_edges,
        top_locations,
    }
}

fn aggregate_dem_location_sensitivities(
    edges: &[DemSamplerEdge],
    location_groups: &[DemLocationGroup],
    edge_sensitivities: &[f64],
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for group in location_groups {
        let mut value = 0.0;
        for edge_index in &group.edge_indices {
            let weight = if group.total_probability > 0.0 {
                edges[*edge_index].probability / group.total_probability
            } else {
                1.0 / group.edge_indices.len() as f64
            };
            value += edge_sensitivities[*edge_index] * weight;
        }
        out.insert(group.location_id.clone(), value);
    }
    out
}

fn aggregate_location_tag_hotspots(
    locations: &[NoiseLocation],
    hotspots: &HashMap<String, f64>,
    tag: &str,
) -> HashMap<TagValue, f64> {
    let mut out = HashMap::new();
    for location in locations {
        let Some(tag_value) = location.tags.get(tag) else {
            continue;
        };
        let hotspot = *hotspots.get(&location.id).unwrap_or(&0.0);
        add_f64(&mut out, tag_value.clone(), hotspot);
    }
    out
}

fn aggregate_dem_tag_hotspots(
    edges: &[DemSamplerEdge],
    hotspots: &HashMap<String, f64>,
    tag: &str,
) -> HashMap<TagValue, f64> {
    let mut by_location: HashMap<String, &HashMap<String, TagValue>> = HashMap::new();
    for edge in edges {
        by_location
            .entry(edge.location_id.clone())
            .or_insert(&edge.tags);
    }
    let mut out = HashMap::new();
    for (location_id, tags) in by_location {
        let Some(tag_value) = tags.get(tag) else {
            continue;
        };
        let hotspot = *hotspots.get(&location_id).unwrap_or(&0.0);
        add_f64(&mut out, tag_value.clone(), hotspot);
    }
    out
}

fn aggregate_dem_detector_hotspots(
    edges: &[DemSamplerEdge],
    edge_hotspots: &[f64],
) -> HashMap<i64, f64> {
    let mut out = HashMap::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        let hotspot = edge_hotspots.get(edge_index).copied().unwrap_or(0.0);
        if hotspot == 0.0 || edge.detectors.is_empty() {
            continue;
        }
        let share = hotspot / edge.detectors.len() as f64;
        for detector_id in &edge.detectors {
            add_f64(&mut out, *detector_id, share);
        }
    }
    out
}

fn compute_detector_graph_estimate(
    edges: &[DemSamplerEdge],
    edge_sensitivities: &[f64],
) -> DetectorGraphEstimate {
    let mut graph = DetectorGraphEstimate {
        by_detector_edge: HashMap::new(),
        signed_by_detector_edge: HashMap::new(),
        by_detector: HashMap::new(),
        signed_by_detector: HashMap::new(),
        by_observable: HashMap::new(),
        signed_by_observable: HashMap::new(),
        by_location: HashMap::new(),
        signed_by_location: HashMap::new(),
    };
    for (edge_index, edge) in edges.iter().enumerate() {
        let sensitivity = edge_sensitivities.get(edge_index).copied().unwrap_or(0.0);
        let hotspot = sensitivity.abs();
        if hotspot == 0.0 {
            continue;
        }
        let key = DetectorGraphKey {
            detectors: edge.detectors.clone(),
            observables: edge.observables.clone(),
        };
        add_f64(&mut graph.by_detector_edge, key.clone(), hotspot);
        add_f64(&mut graph.signed_by_detector_edge, key, sensitivity);
        add_f64(&mut graph.by_location, edge.location_id.clone(), hotspot);
        add_f64(
            &mut graph.signed_by_location,
            edge.location_id.clone(),
            sensitivity,
        );
        if !edge.detectors.is_empty() {
            let share = hotspot / edge.detectors.len() as f64;
            let signed_share = sensitivity / edge.detectors.len() as f64;
            for detector_id in &edge.detectors {
                add_f64(&mut graph.by_detector, *detector_id, share);
                add_f64(&mut graph.signed_by_detector, *detector_id, signed_share);
            }
        }
        if !edge.observables.is_empty() {
            let share = hotspot / edge.observables.len() as f64;
            let signed_share = sensitivity / edge.observables.len() as f64;
            for observable_id in &edge.observables {
                add_f64(&mut graph.by_observable, *observable_id, share);
                add_f64(
                    &mut graph.signed_by_observable,
                    *observable_id,
                    signed_share,
                );
            }
        }
    }
    graph
}

fn top_location_ids(hotspots: &HashMap<String, f64>, top_k: usize) -> Vec<String> {
    if top_k == 0 || hotspots.is_empty() {
        return Vec::new();
    }
    let limit = top_k.min(hotspots.len());
    let mut rows = hotspots
        .iter()
        .map(|(location_id, hotspot)| (location_id, *hotspot))
        .collect::<Vec<_>>();
    let compare = |left: &(&String, f64), right: &(&String, f64)| {
        right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0))
    };
    if limit < rows.len() {
        rows.select_nth_unstable_by(limit, compare);
    }
    rows.truncate(limit);
    rows.sort_by(compare);
    rows.into_iter()
        .map(|(location_id, _)| location_id.clone())
        .collect()
}

fn top_edge_indices(hotspots: &[f64], top_k: usize) -> Vec<usize> {
    if top_k == 0 || hotspots.is_empty() {
        return Vec::new();
    }
    let limit = top_k.min(hotspots.len());
    let mut rows = hotspots.iter().copied().enumerate().collect::<Vec<_>>();
    let compare = |left: &(usize, f64), right: &(usize, f64)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if limit < rows.len() {
        rows.select_nth_unstable_by(limit, compare);
    }
    rows.truncate(limit);
    rows.sort_by(compare);
    rows.into_iter().map(|(edge_index, _)| edge_index).collect()
}

fn add_f64<K>(values: &mut HashMap<K, f64>, key: K, value: f64)
where
    K: std::hash::Hash + Eq,
{
    values
        .entry(key)
        .and_modify(|existing| *existing += value)
        .or_insert(value);
}

fn score_pair(probability: f64) -> (f64, f64) {
    let p = probability.clamp(1e-12, 1.0 - 1e-12);
    (1.0 / p, -1.0 / (1.0 - p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoiseModel;

    #[test]
    fn packed_estimate_reports_location_hotspot() {
        let location = NoiseLocation {
            id: "n".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.5,
            qubits: vec![0],
            tags: HashMap::new(),
        };
        let mut state = RuntimeState::new(1, 2, &["n".to_string()], true);
        state
            .event_masks
            .insert("n".to_string(), Mask { words: vec![0b01] });
        let loss = Mask { words: vec![0b01] };

        let estimate = compute_packed_estimate(&[location], &state, &loss, None, 1);

        assert_eq!(estimate.top_locations, vec!["n"]);
        assert!(estimate.hotspots["n"] > 0.0);
    }
}

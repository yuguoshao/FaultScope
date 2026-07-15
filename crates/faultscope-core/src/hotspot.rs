use std::collections::HashMap;

use crate::dem_sampling::{DemProgramEdge, DemProgramEdgeMetadata, DemProgramLocationGroup};
use crate::labels::HotspotTagKind;
use crate::{
    DemBatch, DemHotspotEstimate, DetectorGraphEstimate, DetectorGraphKey, HotspotEstimate,
    IndexedNoiseLocation, LocationCatalog, LocationId, Mask, RuntimeState, TagValue,
};

/// Compute forward-sampler hotspots using dense location ids throughout.
///
/// User-facing strings are materialized from `catalog` only when constructing
/// the returned boundary object.
pub fn compute_packed_estimate(
    locations: &[IndexedNoiseLocation],
    catalog: &LocationCatalog,
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
    let mut sensitivities = vec![0.0; catalog.len()];
    let mut hotspots = vec![0.0; catalog.len()];
    let mut by_qubit = HashMap::new();
    let zero_mask = Mask::zero(state.all_mask.words.len());

    for (noise_id, location) in locations.iter().enumerate() {
        let event_mask = state.event_masks.get(noise_id).unwrap_or(&zero_mask);
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
        sensitivities[location.location_id.index()] = sensitivity;
        hotspots[location.location_id.index()] = hotspot;
        for qubit in &location.qubits {
            add_f64(&mut by_qubit, *qubit, hotspot);
        }
    }

    let by_round = aggregate_location_tag_hotspots(catalog, &hotspots, HotspotTagKind::Round);
    let by_gate = aggregate_location_tag_hotspots(catalog, &hotspots, HotspotTagKind::Gate);
    let by_operation =
        aggregate_location_tag_hotspots(catalog, &hotspots, HotspotTagKind::Operation);
    let top_locations = top_location_ids(catalog, &hotspots, top_k);

    HotspotEstimate {
        shots: state.shots,
        mean_loss,
        baseline: baseline_value,
        sensitivities: materialize_location_values(catalog, &sensitivities),
        hotspots: materialize_location_values(catalog, &hotspots),
        by_qubit,
        by_round,
        by_gate,
        by_operation,
        top_locations,
    }
}

pub(crate) struct DemEstimateProgram<'a> {
    pub(crate) edges: &'a [DemProgramEdge],
    pub(crate) edge_metadata: &'a [DemProgramEdgeMetadata],
    pub(crate) location_groups: &'a [DemProgramLocationGroup],
    pub(crate) catalog: &'a LocationCatalog,
}

pub(crate) fn compute_dem_estimate(
    program: DemEstimateProgram<'_>,
    batch: &DemBatch,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> DemHotspotEstimate {
    let DemEstimateProgram {
        edges,
        edge_metadata,
        location_groups,
        catalog,
    } = program;
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
        aggregate_dem_location_sensitivities(edges, location_groups, &edge_sensitivities, catalog);
    let edge_hotspots = edge_sensitivities
        .iter()
        .map(|sensitivity| sensitivity.abs())
        .collect::<Vec<_>>();
    let location_hotspots = location_sensitivities
        .iter()
        .map(|value| value.map(f64::abs))
        .collect::<Vec<_>>();
    let by_detector = aggregate_dem_detector_hotspots(edges, &edge_hotspots);
    let by_round =
        aggregate_optional_tag_hotspots(catalog, &location_hotspots, HotspotTagKind::Round);
    let by_gate =
        aggregate_optional_tag_hotspots(catalog, &location_hotspots, HotspotTagKind::Gate);
    let by_operation =
        aggregate_optional_tag_hotspots(catalog, &location_hotspots, HotspotTagKind::Operation);
    let detector_graph =
        compute_detector_graph_estimate(edges, edge_metadata, catalog, &edge_sensitivities);
    let top_edges = top_edge_indices(&edge_hotspots, top_k);
    let top_locations = top_optional_location_ids(catalog, &location_hotspots, top_k);

    DemHotspotEstimate {
        shots: batch.shots,
        mean_loss,
        baseline: baseline_value,
        edge_sensitivities,
        edge_hotspots,
        location_sensitivities: materialize_optional_location_values(
            catalog,
            &location_sensitivities,
        ),
        location_hotspots: materialize_optional_location_values(catalog, &location_hotspots),
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
    edges: &[DemProgramEdge],
    location_groups: &[DemProgramLocationGroup],
    edge_sensitivities: &[f64],
    catalog: &LocationCatalog,
) -> Vec<Option<f64>> {
    let mut out = vec![None; catalog.len()];
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
        out[group.location_id.index()] = Some(value);
    }
    out
}

fn aggregate_location_tag_hotspots(
    catalog: &LocationCatalog,
    hotspots: &[f64],
    kind: HotspotTagKind,
) -> HashMap<TagValue, f64> {
    let values = hotspots.iter().copied().map(Some).collect::<Vec<_>>();
    aggregate_optional_tag_hotspots(catalog, &values, kind)
}

fn aggregate_optional_tag_hotspots(
    catalog: &LocationCatalog,
    hotspots: &[Option<f64>],
    kind: HotspotTagKind,
) -> HashMap<TagValue, f64> {
    let mut totals = vec![0.0; catalog.tag_value_count()];
    let mut seen = vec![false; catalog.tag_value_count()];
    for (location_index, hotspot) in hotspots.iter().enumerate() {
        let Some(hotspot) = hotspot else {
            continue;
        };
        let location_id = LocationId::new(location_index);
        let Some(tag_value_id) = catalog.hotspot_tag(location_id, kind) else {
            continue;
        };
        totals[tag_value_id.index()] += *hotspot;
        seen[tag_value_id.index()] = true;
    }
    totals
        .into_iter()
        .zip(seen)
        .enumerate()
        .filter(|(_, (_, seen))| *seen)
        .map(|(tag_value_index, (value, _))| {
            (
                catalog
                    .tag_value(crate::labels::TagValueId(tag_value_index))
                    .clone(),
                value,
            )
        })
        .collect()
}

fn aggregate_dem_detector_hotspots(
    edges: &[DemProgramEdge],
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
    edges: &[DemProgramEdge],
    edge_metadata: &[DemProgramEdgeMetadata],
    catalog: &LocationCatalog,
    edge_sensitivities: &[f64],
) -> DetectorGraphEstimate {
    let mut by_detector_edge = HashMap::new();
    let mut signed_by_detector_edge = HashMap::new();
    let mut by_detector = HashMap::new();
    let mut signed_by_detector = HashMap::new();
    let mut by_observable = HashMap::new();
    let mut signed_by_observable = HashMap::new();
    let mut by_location = vec![0.0; catalog.len()];
    let mut signed_by_location = vec![0.0; catalog.len()];
    let mut location_seen = vec![false; catalog.len()];

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
        add_f64(&mut by_detector_edge, key.clone(), hotspot);
        add_f64(&mut signed_by_detector_edge, key, sensitivity);
        if let Some(metadata) = edge_metadata.get(edge_index) {
            by_location[metadata.location_id.index()] += hotspot;
            signed_by_location[metadata.location_id.index()] += sensitivity;
            location_seen[metadata.location_id.index()] = true;
        }
        if !edge.detectors.is_empty() {
            let share = hotspot / edge.detectors.len() as f64;
            let signed_share = sensitivity / edge.detectors.len() as f64;
            for detector_id in &edge.detectors {
                add_f64(&mut by_detector, *detector_id, share);
                add_f64(&mut signed_by_detector, *detector_id, signed_share);
            }
        }
        if !edge.observables.is_empty() {
            let share = hotspot / edge.observables.len() as f64;
            let signed_share = sensitivity / edge.observables.len() as f64;
            for observable_id in &edge.observables {
                add_f64(&mut by_observable, *observable_id, share);
                add_f64(&mut signed_by_observable, *observable_id, signed_share);
            }
        }
    }

    DetectorGraphEstimate {
        by_detector_edge,
        signed_by_detector_edge,
        by_detector,
        signed_by_detector,
        by_observable,
        signed_by_observable,
        by_location: materialize_seen_location_values(catalog, &by_location, &location_seen),
        signed_by_location: materialize_seen_location_values(
            catalog,
            &signed_by_location,
            &location_seen,
        ),
    }
}

fn materialize_location_values(catalog: &LocationCatalog, values: &[f64]) -> HashMap<String, f64> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| (catalog.label(LocationId::new(index)).to_string(), *value))
        .collect()
}

fn materialize_optional_location_values(
    catalog: &LocationCatalog,
    values: &[Option<f64>],
) -> HashMap<String, f64> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            value.map(|value| (catalog.label(LocationId::new(index)).to_string(), value))
        })
        .collect()
}

fn materialize_seen_location_values(
    catalog: &LocationCatalog,
    values: &[f64],
    seen: &[bool],
) -> HashMap<String, f64> {
    values
        .iter()
        .zip(seen)
        .enumerate()
        .filter(|(_, (_, seen))| **seen)
        .map(|(index, (value, _))| (catalog.label(LocationId::new(index)).to_string(), *value))
        .collect()
}

fn top_location_ids(catalog: &LocationCatalog, hotspots: &[f64], top_k: usize) -> Vec<String> {
    let values = hotspots.iter().copied().map(Some).collect::<Vec<_>>();
    top_optional_location_ids(catalog, &values, top_k)
}

fn top_optional_location_ids(
    catalog: &LocationCatalog,
    hotspots: &[Option<f64>],
    top_k: usize,
) -> Vec<String> {
    if top_k == 0 {
        return Vec::new();
    }
    let mut rows = hotspots
        .iter()
        .enumerate()
        .filter_map(|(index, hotspot)| hotspot.map(|hotspot| (LocationId::new(index), hotspot)))
        .collect::<Vec<_>>();
    let limit = top_k.min(rows.len());
    let compare = |left: &(LocationId, f64), right: &(LocationId, f64)| {
        right.1.total_cmp(&left.1).then_with(|| {
            catalog
                .lexical_rank(left.0)
                .cmp(&catalog.lexical_rank(right.0))
        })
    };
    if limit < rows.len() {
        rows.select_nth_unstable_by(limit, compare);
    }
    rows.truncate(limit);
    rows.sort_by(compare);
    rows.into_iter()
        .map(|(location_id, _)| catalog.label(location_id).to_string())
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
    use crate::labels::LocationCatalogBuilder;
    use crate::{NoiseLocation, NoiseModel};

    #[test]
    fn packed_estimate_reports_location_hotspot() {
        let input = NoiseLocation {
            id: "n".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.5,
            qubits: vec![0],
            tags: HashMap::new(),
        };
        let mut builder = LocationCatalogBuilder::with_capacity(1);
        let location_id = builder.intern(input.id.clone(), input.tags.clone());
        let locations = vec![IndexedNoiseLocation::from_input(location_id, input)];
        let catalog = builder.finish();
        let mut state = RuntimeState::new(1, 2, 0, 1, true);
        state.event_masks[0] = Mask { words: vec![0b01] };
        let loss = Mask { words: vec![0b01] };

        let estimate = compute_packed_estimate(&locations, &catalog, &state, &loss, None, 1);

        assert_eq!(estimate.top_locations, vec!["n"]);
        assert!(estimate.hotspots["n"] > 0.0);
    }

    #[test]
    fn integer_location_ranking_preserves_lexical_tie_breaks() {
        let mut builder = LocationCatalogBuilder::with_capacity(2);
        let mut locations = Vec::new();
        for label in ["z", "a"] {
            let input = NoiseLocation {
                id: label.to_string(),
                model: NoiseModel::BernoulliPauli("X".to_string()),
                rate: 0.5,
                qubits: vec![0],
                tags: HashMap::new(),
            };
            let location_id = builder.intern(input.id.clone(), input.tags.clone());
            locations.push(IndexedNoiseLocation::from_input(location_id, input));
        }
        let catalog = builder.finish();
        let mut state = RuntimeState::new(1, 2, 0, 2, true);
        state.event_masks.fill(Mask { words: vec![0b01] });
        let loss = Mask { words: vec![0b01] };

        let estimate = compute_packed_estimate(&locations, &catalog, &state, &loss, None, 2);

        assert_eq!(estimate.top_locations, vec!["a", "z"]);
    }
}

use std::collections::HashMap;

use crate::{
    bernoulli_mask, for_each_bernoulli_event, word_count, DemBatch, DemHotspotEstimate,
    DemLocationGroup, DemSamplerEdge, DetectorErrorModel, Mask, NpError, NpResult, SmallRng,
};

/// Bit-packed detector-error-model sampler and hotspot estimator.
///
/// This simulator samples DEM edges as independent Bernoulli error
/// instructions, computes detector and observable masks, and can aggregate
/// edge and location hotspot estimates entirely in Rust.
#[derive(Debug, Clone, PartialEq)]
pub struct DemHotspotEstimator {
    pub detector_ids: Vec<i64>,
    pub observable_ids: Vec<i64>,
    pub edges: Vec<DemSamplerEdge>,
    pub location_groups: Vec<DemLocationGroup>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedDemShotBatch {
    pub shots: usize,
    pub detector_data: Vec<u8>,
    pub detector_byte_count: usize,
    pub observable_ids: Vec<i64>,
    pub observable_data: Vec<u8>,
    pub observable_byte_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectorEventDemShotBatch {
    pub shots: usize,
    pub offsets: Vec<usize>,
    pub events: Vec<usize>,
    pub observable_ids: Vec<i64>,
    pub observable_data: Vec<u8>,
    pub observable_byte_count: usize,
}

impl DemHotspotEstimator {
    /// Build a simulator from a typed detector error model.
    pub fn new(dem: DetectorErrorModel) -> NpResult<Self> {
        let detector_ids = dem.detectors.iter().map(|detector| detector.id).collect();
        let observable_ids = dem
            .observables
            .iter()
            .map(|observable| observable.id)
            .collect();
        let edges = dem
            .edges
            .into_iter()
            .map(|edge| DemSamplerEdge {
                probability: edge.probability,
                detectors: edge.detectors,
                observables: edge.observables,
                location_id: edge.location_id,
                event: edge.event,
                tags: edge.tags,
            })
            .collect();
        Self::from_parts(detector_ids, observable_ids, edges)
    }

    /// Build a simulator from pre-extracted ids and sampler edges.
    pub fn from_parts(
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        edges: Vec<DemSamplerEdge>,
    ) -> NpResult<Self> {
        for edge in &edges {
            edge.validate()?;
        }
        let location_groups = build_dem_location_groups(&edges);
        Ok(Self {
            detector_ids,
            observable_ids,
            edges,
            location_groups,
        })
    }

    /// Build a simulator for sampling only, without hotspot location metadata.
    pub fn from_sampling_parts(
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        edges: Vec<DemSamplerEdge>,
    ) -> NpResult<Self> {
        for edge in &edges {
            edge.validate()?;
        }
        Ok(Self {
            detector_ids,
            observable_ids,
            edges,
            location_groups: Vec::new(),
        })
    }

    /// Run DEM sampling for `shots`.
    pub fn run_batch(
        &self,
        shots: usize,
        seed: Option<u64>,
        return_edge_events: bool,
    ) -> NpResult<DemBatch> {
        if shots == 0 {
            return Err(NpError::new("shots must be positive"));
        }
        let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
        Ok(self.run_batch_with_rng(shots, &mut rng, return_edge_events))
    }

    /// Run DEM sampling with a caller-owned core RNG.
    pub fn run_batch_with_rng(
        &self,
        shots: usize,
        rng: &mut SmallRng,
        return_edge_events: bool,
    ) -> DemBatch {
        run_dem_batch(
            &self.detector_ids,
            &self.observable_ids,
            &self.edges,
            shots,
            rng,
            return_edge_events,
        )
    }

    pub fn run_packed_shot_batch_with_rng(
        &self,
        shots: usize,
        rng: &mut SmallRng,
        detector_ids: &[i64],
        observable_ids: &[i64],
    ) -> NpResult<PackedDemShotBatch> {
        run_dem_packed_shot_batch(
            &self.detector_ids,
            &self.observable_ids,
            &self.edges,
            detector_ids,
            observable_ids,
            shots,
            rng,
        )
    }

    pub fn run_detector_event_shot_batch_with_rng(
        &self,
        shots: usize,
        rng: &mut SmallRng,
        detector_ids: &[i64],
        observable_ids: &[i64],
    ) -> NpResult<DetectorEventDemShotBatch> {
        run_dem_detector_event_shot_batch(
            &self.detector_ids,
            &self.observable_ids,
            &self.edges,
            detector_ids,
            observable_ids,
            shots,
            rng,
        )
    }

    /// Run a batch and estimate hotspots using the default logical loss mask.
    pub fn estimate_default(
        &self,
        shots: usize,
        seed: Option<u64>,
        baseline: Option<f64>,
        top_k: usize,
    ) -> NpResult<DemHotspotEstimate> {
        let batch = self.run_batch(shots, seed, true)?;
        Ok(self.estimate_from_loss(&batch, &batch.loss_mask, baseline, top_k))
    }

    /// Estimate hotspots from an externally supplied loss mask.
    pub fn estimate_from_loss(
        &self,
        batch: &DemBatch,
        loss_mask: &Mask,
        baseline: Option<f64>,
        top_k: usize,
    ) -> DemHotspotEstimate {
        crate::compute_dem_estimate(
            &self.edges,
            &self.location_groups,
            batch,
            loss_mask,
            baseline,
            top_k,
        )
    }
}

pub fn run_dem_detector_event_shot_batch(
    sampling_detector_ids: &[i64],
    sampling_observable_ids: &[i64],
    edges: &[DemSamplerEdge],
    detector_ids: &[i64],
    observable_ids: &[i64],
    shots: usize,
    rng: &mut SmallRng,
) -> NpResult<DetectorEventDemShotBatch> {
    let sampling_detector_index = sampling_detector_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();
    for detector_id in detector_ids {
        if !sampling_detector_index.contains_key(detector_id) {
            return Err(NpError::new(format!(
                "detector-event DEM sampler requested detector id {detector_id}, but it is not declared by the DEM"
            )));
        }
    }
    let detector_index = detector_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();

    let sampling_observable_index = sampling_observable_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();
    for observable_id in observable_ids {
        if !sampling_observable_index.contains_key(observable_id) {
            return Err(NpError::new(format!(
                "detector-event DEM sampler requested observable id {observable_id}, but it is not declared by the DEM"
            )));
        }
    }
    let observable_index = observable_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();

    let packed_edges = edges
        .iter()
        .map(|edge| {
            let detector_columns = edge
                .detectors
                .iter()
                .map(|detector_id| packed_column_for_id(*detector_id, &detector_index))
                .collect::<NpResult<Vec<_>>>()?;
            let mut observable_columns = Vec::with_capacity(edge.observables.len());
            for observable_id in &edge.observables {
                let column = packed_column_for_id(*observable_id, &observable_index)?;
                observable_columns.push((column >> 3, 1u8 << (column & 7)));
            }
            Ok(EventEdgeColumns {
                probability: edge.probability,
                detector_columns,
                observable_columns,
            })
        })
        .collect::<NpResult<Vec<_>>>()?;

    let observable_byte_count = observable_ids.len().div_ceil(8);
    let mut observable_data = vec![0; shots * observable_byte_count];
    let mut event_counts = vec![0usize; shots];
    let mut raw_events = Vec::<(usize, usize)>::new();

    for columns in &packed_edges {
        if columns.detector_columns.is_empty() && columns.observable_columns.is_empty() {
            continue;
        }
        for_each_bernoulli_event(rng, shots, columns.probability, |shot, _| {
            event_counts[shot] += columns.detector_columns.len();
            raw_events.extend(
                columns
                    .detector_columns
                    .iter()
                    .copied()
                    .map(|detector| (shot, detector)),
            );
            let observable_row = shot * observable_byte_count;
            for (byte_index, bit_mask) in &columns.observable_columns {
                observable_data[observable_row + byte_index] ^= bit_mask;
            }
        });
    }

    let mut offsets = Vec::with_capacity(shots + 1);
    offsets.push(0);
    for count in event_counts {
        offsets.push(offsets.last().copied().unwrap_or(0) + count);
    }

    let mut events = vec![0usize; offsets.last().copied().unwrap_or(0)];
    let mut next = offsets.clone();
    for (shot, detector) in raw_events {
        let index = next[shot];
        events[index] = detector;
        next[shot] += 1;
    }

    Ok(DetectorEventDemShotBatch {
        shots,
        offsets,
        events,
        observable_ids: observable_ids.to_vec(),
        observable_data,
        observable_byte_count,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct EventEdgeColumns {
    probability: f64,
    detector_columns: Vec<usize>,
    observable_columns: Vec<(usize, u8)>,
}

pub fn run_dem_packed_shot_batch(
    sampling_detector_ids: &[i64],
    sampling_observable_ids: &[i64],
    edges: &[DemSamplerEdge],
    detector_ids: &[i64],
    observable_ids: &[i64],
    shots: usize,
    rng: &mut SmallRng,
) -> NpResult<PackedDemShotBatch> {
    let sampling_detector_index = sampling_detector_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();
    for detector_id in detector_ids {
        if !sampling_detector_index.contains_key(detector_id) {
            return Err(NpError::new(format!(
                "packed DEM sampler requested detector id {detector_id}, but it is not declared by the DEM"
            )));
        }
    }
    let detector_index = detector_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();

    let sampling_observable_index = sampling_observable_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();
    for observable_id in observable_ids {
        if !sampling_observable_index.contains_key(observable_id) {
            return Err(NpError::new(format!(
                "packed DEM sampler requested observable id {observable_id}, but it is not declared by the DEM"
            )));
        }
    }
    let observable_index = observable_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();

    let packed_edges = edges
        .iter()
        .map(|edge| {
            let detector_columns = edge
                .detectors
                .iter()
                .map(|detector_id| {
                    packed_column_for_id(*detector_id, &detector_index)
                        .map(|column| (column >> 3, 1u8 << (column & 7)))
                })
                .collect::<NpResult<Vec<_>>>()?;
            let mut observable_columns = Vec::with_capacity(edge.observables.len());
            for observable_id in &edge.observables {
                let column = packed_column_for_id(*observable_id, &observable_index)?;
                observable_columns.push((column >> 3, 1u8 << (column & 7)));
            }
            Ok(PackedEdgeColumns {
                probability: edge.probability,
                detector_columns,
                observable_columns,
            })
        })
        .collect::<NpResult<Vec<_>>>()?;

    let detector_byte_count = detector_ids.len().div_ceil(8);
    let observable_byte_count = observable_ids.len().div_ceil(8);
    let mut detector_data = vec![0; shots * detector_byte_count];
    let mut observable_data = vec![0; shots * observable_byte_count];

    for columns in &packed_edges {
        if columns.detector_columns.is_empty() && columns.observable_columns.is_empty() {
            continue;
        }
        for_each_bernoulli_event(rng, shots, columns.probability, |shot, _| {
            let detector_row = shot * detector_byte_count;
            for (byte_index, bit_mask) in &columns.detector_columns {
                detector_data[detector_row + byte_index] ^= bit_mask;
            }
            let observable_row = shot * observable_byte_count;
            for (byte_index, bit_mask) in &columns.observable_columns {
                observable_data[observable_row + byte_index] ^= bit_mask;
            }
        });
    }

    Ok(PackedDemShotBatch {
        shots,
        detector_data,
        detector_byte_count,
        observable_ids: observable_ids.to_vec(),
        observable_data,
        observable_byte_count,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct PackedEdgeColumns {
    probability: f64,
    detector_columns: Vec<(usize, u8)>,
    observable_columns: Vec<(usize, u8)>,
}

fn packed_column_for_id(id: i64, index: &HashMap<i64, usize>) -> NpResult<usize> {
    index.get(&id).copied().ok_or_else(|| {
        NpError::new(format!(
            "packed DEM sampler edge references unknown id {id}"
        ))
    })
}

pub fn run_dem_batch(
    detector_ids: &[i64],
    observable_ids: &[i64],
    edges: &[DemSamplerEdge],
    shots: usize,
    rng: &mut SmallRng,
    return_edge_events: bool,
) -> DemBatch {
    let words = word_count(shots);
    let all_mask = Mask::all(shots);
    let mut detectors = HashMap::new();
    for detector_id in detector_ids {
        detectors.insert(*detector_id, Mask::zero(words));
    }
    let mut observables = HashMap::new();
    for observable_id in observable_ids {
        observables.insert(*observable_id, Mask::zero(words));
    }
    let mut edge_event_masks = if return_edge_events {
        Vec::with_capacity(edges.len())
    } else {
        Vec::new()
    };

    for edge in edges {
        let event_mask = bernoulli_mask(rng, shots, edge.probability);
        if !event_mask.is_zero() {
            for detector_id in &edge.detectors {
                detectors
                    .entry(*detector_id)
                    .or_insert_with(|| Mask::zero(words))
                    .xor_assign(&event_mask);
            }
            for observable_id in &edge.observables {
                observables
                    .entry(*observable_id)
                    .or_insert_with(|| Mask::zero(words))
                    .xor_assign(&event_mask);
            }
        }
        if return_edge_events {
            edge_event_masks.push(event_mask);
        }
    }

    let mut loss_mask = Mask::zero(words);
    for observable in observables.values() {
        loss_mask.or_assign(observable);
    }
    loss_mask.and_assign(&all_mask);

    DemBatch {
        shots,
        all_mask,
        detectors,
        observables,
        edge_event_masks,
        loss_mask,
    }
}

pub fn build_dem_location_groups(edges: &[DemSamplerEdge]) -> Vec<DemLocationGroup> {
    let mut group_indices = HashMap::<String, usize>::new();
    let mut groups = Vec::<DemLocationGroup>::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        let group_index = if let Some(group_index) = group_indices.get(&edge.location_id) {
            *group_index
        } else {
            let group_index = groups.len();
            group_indices.insert(edge.location_id.clone(), group_index);
            groups.push(DemLocationGroup {
                location_id: edge.location_id.clone(),
                edge_indices: Vec::new(),
                total_probability: 0.0,
            });
            group_index
        };
        let group = &mut groups[group_index];
        group.edge_indices.push(edge_index);
        group.total_probability += edge.probability;
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_edges_by_location() {
        let edges = vec![
            DemSamplerEdge {
                probability: 0.1,
                detectors: vec![0],
                observables: Vec::new(),
                location_id: "a".to_string(),
                event: crate::DemEvent::Pauli("X".to_string()),
                tags: HashMap::new(),
            },
            DemSamplerEdge {
                probability: 0.2,
                detectors: vec![1],
                observables: Vec::new(),
                location_id: "a".to_string(),
                event: crate::DemEvent::Pauli("Z".to_string()),
                tags: HashMap::new(),
            },
        ];

        let groups = build_dem_location_groups(&edges);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].edge_indices, vec![0, 1]);
        assert_eq!(groups[0].total_probability, 0.30000000000000004);
    }

    #[test]
    fn simulator_estimates_default_logical_loss() {
        let simulator = DemHotspotEstimator::from_parts(
            Vec::new(),
            vec![0],
            vec![DemSamplerEdge {
                probability: 1.0,
                detectors: Vec::new(),
                observables: vec![0],
                location_id: "logical".to_string(),
                event: crate::DemEvent::Pauli("L".to_string()),
                tags: HashMap::new(),
            }],
        )
        .unwrap();

        let estimate = simulator.estimate_default(64, Some(7), None, 1).unwrap();

        assert_eq!(estimate.shots, 64);
        assert_eq!(estimate.mean_loss, 1.0);
        assert_eq!(estimate.top_edges, vec![0]);
        assert_eq!(estimate.top_locations, vec!["logical"]);
    }

    #[test]
    fn simulator_rejects_invalid_edge_probability() {
        let err = DemHotspotEstimator::from_parts(
            Vec::new(),
            Vec::new(),
            vec![DemSamplerEdge {
                probability: 1.5,
                detectors: Vec::new(),
                observables: Vec::new(),
                location_id: "bad".to_string(),
                event: crate::DemEvent::Pauli("X".to_string()),
                tags: HashMap::new(),
            }],
        )
        .unwrap_err();

        assert!(err
            .message()
            .contains("DEM edge probability must be in [0, 1]"));
    }
}

use std::collections::{hash_map::Entry, HashMap};

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

/// Precomputed detector and observable layouts for repeated DEM sampling.
///
/// The plan is immutable and independent of [`DemHotspotEstimator`], so it
/// remains valid even when callers mutate the estimator's public fields.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledDemSamplingPlan {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    edges: Vec<CompiledSamplingEdge>,
    detector_byte_count: usize,
    observable_byte_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct CompiledSamplingEdge {
    probability: f64,
    detector_columns: Vec<usize>,
    packed_detector_columns: Vec<(usize, u8)>,
    observable_columns: Vec<(usize, u8)>,
}

/// Precomputed observable layout for counting logical failures without a decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledDemLogicalCountPlan {
    observable_ids: Vec<i64>,
    edges: Vec<CompiledLogicalCountEdge>,
    single_observable_flip_edge_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct CompiledLogicalCountEdge {
    probability: f64,
    observable_columns: Vec<usize>,
    single_observable_flip: bool,
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
        Ok(self
            .compile_sampling_plan(detector_ids, observable_ids)?
            .run_packed_shot_batch_with_rng(shots, rng))
    }

    pub fn run_detector_event_shot_batch_with_rng(
        &self,
        shots: usize,
        rng: &mut SmallRng,
        detector_ids: &[i64],
        observable_ids: &[i64],
    ) -> NpResult<DetectorEventDemShotBatch> {
        Ok(compile_dem_sampling_plan(
            &self.detector_ids,
            &self.observable_ids,
            &self.edges,
            detector_ids,
            observable_ids,
            "detector-event DEM sampler",
        )?
        .run_detector_event_shot_batch_with_rng(shots, rng))
    }

    /// Compile detector and observable layouts for repeated packed DEM batches.
    pub fn compile_sampling_plan(
        &self,
        detector_ids: &[i64],
        observable_ids: &[i64],
    ) -> NpResult<CompiledDemSamplingPlan> {
        compile_dem_sampling_plan(
            &self.detector_ids,
            &self.observable_ids,
            &self.edges,
            detector_ids,
            observable_ids,
            "packed DEM sampler",
        )
    }

    /// Compile the observable-only path used to count default logical failures.
    pub fn compile_logical_count_plan(&self) -> CompiledDemLogicalCountPlan {
        CompiledDemLogicalCountPlan::new(&self.observable_ids, &self.edges)
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
    Ok(compile_dem_sampling_plan(
        sampling_detector_ids,
        sampling_observable_ids,
        edges,
        detector_ids,
        observable_ids,
        "detector-event DEM sampler",
    )?
    .run_detector_event_shot_batch_with_rng(shots, rng))
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
    Ok(compile_dem_sampling_plan(
        sampling_detector_ids,
        sampling_observable_ids,
        edges,
        detector_ids,
        observable_ids,
        "packed DEM sampler",
    )?
    .run_packed_shot_batch_with_rng(shots, rng))
}

fn compile_dem_sampling_plan(
    sampling_detector_ids: &[i64],
    sampling_observable_ids: &[i64],
    edges: &[DemSamplerEdge],
    detector_ids: &[i64],
    observable_ids: &[i64],
    sampler_name: &str,
) -> NpResult<CompiledDemSamplingPlan> {
    let sampling_detector_index = sampling_detector_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();
    for detector_id in detector_ids {
        if !sampling_detector_index.contains_key(detector_id) {
            return Err(NpError::new(format!(
                "{sampler_name} requested detector id {detector_id}, but it is not declared by the DEM"
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
                "{sampler_name} requested observable id {observable_id}, but it is not declared by the DEM"
            )));
        }
    }
    let observable_index = observable_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<HashMap<_, _>>();

    let compiled_edges = edges
        .iter()
        .map(|edge| {
            let detector_columns = edge
                .detectors
                .iter()
                .map(|detector_id| packed_column_for_id(*detector_id, &detector_index))
                .collect::<NpResult<Vec<_>>>()?;
            let packed_detector_columns = detector_columns
                .iter()
                .map(|column| (column >> 3, 1u8 << (column & 7)))
                .collect();
            let mut observable_columns = Vec::with_capacity(edge.observables.len());
            for observable_id in &edge.observables {
                let column = packed_column_for_id(*observable_id, &observable_index)?;
                observable_columns.push((column >> 3, 1u8 << (column & 7)));
            }
            Ok(CompiledSamplingEdge {
                probability: edge.probability,
                detector_columns,
                packed_detector_columns,
                observable_columns,
            })
        })
        .collect::<NpResult<Vec<_>>>()?;

    Ok(CompiledDemSamplingPlan {
        detector_ids: detector_ids.to_vec(),
        observable_ids: observable_ids.to_vec(),
        edges: compiled_edges,
        detector_byte_count: detector_ids.len().div_ceil(8),
        observable_byte_count: observable_ids.len().div_ceil(8),
    })
}

impl CompiledDemSamplingPlan {
    /// Detector ids in the packed output order.
    pub fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    /// Observable ids in the packed output order.
    pub fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    /// Sample a shot-major bit-packed detector and observable batch.
    pub fn run_packed_shot_batch_with_rng(
        &self,
        shots: usize,
        rng: &mut SmallRng,
    ) -> PackedDemShotBatch {
        let mut detector_data = vec![0; shots * self.detector_byte_count];
        let mut observable_data = vec![0; shots * self.observable_byte_count];

        for columns in &self.edges {
            if columns.detector_columns.is_empty() && columns.observable_columns.is_empty() {
                continue;
            }
            for_each_bernoulli_event(rng, shots, columns.probability, |shot, _| {
                let detector_row = shot * self.detector_byte_count;
                for (byte_index, bit_mask) in &columns.packed_detector_columns {
                    detector_data[detector_row + byte_index] ^= bit_mask;
                }
                let observable_row = shot * self.observable_byte_count;
                for (byte_index, bit_mask) in &columns.observable_columns {
                    observable_data[observable_row + byte_index] ^= bit_mask;
                }
            });
        }

        PackedDemShotBatch {
            shots,
            detector_data,
            detector_byte_count: self.detector_byte_count,
            observable_ids: self.observable_ids.clone(),
            observable_data,
            observable_byte_count: self.observable_byte_count,
        }
    }

    /// Sample a detector-event batch using the precomputed detector indices.
    pub fn run_detector_event_shot_batch_with_rng(
        &self,
        shots: usize,
        rng: &mut SmallRng,
    ) -> DetectorEventDemShotBatch {
        let mut observable_data = vec![0; shots * self.observable_byte_count];
        let mut event_counts = vec![0usize; shots];
        let mut raw_events = Vec::<(usize, usize)>::new();

        for columns in &self.edges {
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
                let observable_row = shot * self.observable_byte_count;
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

        DetectorEventDemShotBatch {
            shots,
            offsets,
            events,
            observable_ids: self.observable_ids.clone(),
            observable_data,
            observable_byte_count: self.observable_byte_count,
        }
    }
}

impl CompiledDemLogicalCountPlan {
    fn new(observable_ids: &[i64], edges: &[DemSamplerEdge]) -> Self {
        let mut compiled_observable_ids = Vec::new();
        let mut observable_index = HashMap::new();
        for observable_id in observable_ids
            .iter()
            .chain(edges.iter().flat_map(|edge| edge.observables.iter()))
        {
            let next_index = compiled_observable_ids.len();
            if let Entry::Vacant(entry) = observable_index.entry(*observable_id) {
                entry.insert(next_index);
                compiled_observable_ids.push(*observable_id);
            }
        }
        let compiled_edges = edges
            .iter()
            .map(|edge| {
                let observable_columns = edge
                    .observables
                    .iter()
                    .map(|observable_id| observable_index[observable_id])
                    .collect::<Vec<_>>();
                let single_observable_flip = observable_columns
                    .iter()
                    .filter(|column| **column == 0)
                    .count()
                    % 2
                    != 0;
                CompiledLogicalCountEdge {
                    probability: edge.probability,
                    observable_columns,
                    single_observable_flip,
                }
            })
            .collect::<Vec<_>>();
        let single_observable_flip_edge_count = compiled_edges
            .iter()
            .filter(|edge| edge.single_observable_flip)
            .count();
        Self {
            observable_ids: compiled_observable_ids,
            edges: compiled_edges,
            single_observable_flip_edge_count,
        }
    }

    /// Observable ids included in the default logical loss mask.
    pub fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    /// Count shots where at least one logical observable is set.
    pub fn sample_logical_error_count_with_rng(&self, shots: usize, rng: &mut SmallRng) -> usize {
        if self.observable_ids.len() == 1 {
            return self.sample_single_observable_count(shots, rng);
        }
        let words = word_count(shots);
        let mut observable_words = vec![0u64; self.observable_ids.len() * words];
        let mut event_words = vec![0u64; words];

        for edge in &self.edges {
            fill_bernoulli_words(&mut event_words, shots, edge.probability, rng);
            for observable_column in &edge.observable_columns {
                let start = observable_column * words;
                for (observable, event) in observable_words[start..start + words]
                    .iter_mut()
                    .zip(&event_words)
                {
                    *observable ^= event;
                }
            }
        }

        (0..words)
            .map(|word| {
                (0..self.observable_ids.len())
                    .fold(0u64, |loss, observable| {
                        loss | observable_words[observable * words + word]
                    })
                    .count_ones() as usize
            })
            .sum()
    }

    fn sample_single_observable_count(&self, shots: usize, rng: &mut SmallRng) -> usize {
        if self.single_observable_flip_edge_count <= 1 {
            let mut errors = 0usize;
            for edge in &self.edges {
                if edge.probability <= 0.0 {
                    continue;
                }
                if edge.probability >= 1.0 {
                    if edge.single_observable_flip {
                        errors += shots;
                    }
                    continue;
                }
                if edge.single_observable_flip {
                    for_each_bernoulli_event(rng, shots, edge.probability, |_, _| errors += 1);
                } else {
                    for_each_bernoulli_event(rng, shots, edge.probability, |_, _| {});
                }
            }
            return errors;
        }

        const STACK_WORDS: usize = 8;
        let words = word_count(shots);
        if words <= STACK_WORDS {
            let mut logical_words = [0u64; STACK_WORDS];
            sample_single_observable_words(&self.edges, shots, rng, &mut logical_words[..words]);
            logical_words[..words]
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum()
        } else {
            let mut logical_words = vec![0u64; words];
            sample_single_observable_words(&self.edges, shots, rng, &mut logical_words);
            logical_words
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum()
        }
    }
}

fn sample_single_observable_words(
    edges: &[CompiledLogicalCountEdge],
    shots: usize,
    rng: &mut SmallRng,
    logical_words: &mut [u64],
) {
    for edge in edges {
        if edge.probability <= 0.0 {
            continue;
        }
        if edge.probability >= 1.0 {
            if edge.single_observable_flip {
                for word in logical_words.iter_mut() {
                    *word ^= u64::MAX;
                }
                let remainder = shots % 64;
                if remainder != 0 {
                    *logical_words.last_mut().expect("non-empty shot mask") &=
                        (1u64 << remainder) - 1;
                }
            }
            continue;
        }
        if edge.single_observable_flip {
            for_each_bernoulli_event(rng, shots, edge.probability, |shot, _| {
                logical_words[shot / 64] ^= 1u64 << (shot % 64);
            });
        } else {
            for_each_bernoulli_event(rng, shots, edge.probability, |_, _| {});
        }
    }
}

fn fill_bernoulli_words(words: &mut [u64], shots: usize, probability: f64, rng: &mut SmallRng) {
    words.fill(0);
    if probability <= 0.0 {
        return;
    }
    if probability >= 1.0 {
        words.fill(u64::MAX);
        if let (Some(last), remainder) = (words.last_mut(), shots % 64) {
            if remainder != 0 {
                *last &= (1u64 << remainder) - 1;
            }
        }
        return;
    }
    for_each_bernoulli_event(rng, shots, probability, |shot, _| {
        words[shot / 64] |= 1u64 << (shot % 64);
    });
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

    fn edge(probability: f64, detectors: Vec<i64>, observables: Vec<i64>) -> DemSamplerEdge {
        DemSamplerEdge {
            probability,
            detectors,
            observables,
            location_id: "test".to_string(),
            event: crate::DemEvent::Pauli("X".to_string()),
            tags: HashMap::new(),
        }
    }

    fn packed_reference(
        batch: &DemBatch,
        detector_ids: &[i64],
        observable_ids: &[i64],
    ) -> (Vec<u8>, Vec<u8>) {
        let detector_bytes = detector_ids.len().div_ceil(8);
        let observable_bytes = observable_ids.len().div_ceil(8);
        let mut detectors = vec![0u8; batch.shots * detector_bytes];
        let mut observables = vec![0u8; batch.shots * observable_bytes];
        for shot in 0..batch.shots {
            for (column, detector_id) in detector_ids.iter().enumerate() {
                if batch.detectors[detector_id].words[shot / 64] & (1u64 << (shot % 64)) != 0 {
                    detectors[shot * detector_bytes + column / 8] |= 1u8 << (column % 8);
                }
            }
            for (column, observable_id) in observable_ids.iter().enumerate() {
                if batch.observables[observable_id].words[shot / 64] & (1u64 << (shot % 64)) != 0 {
                    observables[shot * observable_bytes + column / 8] |= 1u8 << (column % 8);
                }
            }
        }
        (detectors, observables)
    }

    #[test]
    fn compiled_sampling_matches_generic_batch_across_word_boundaries() {
        let simulator = DemHotspotEstimator::from_sampling_parts(
            vec![10, 20, 30],
            vec![7, 9],
            vec![
                edge(0.0, vec![10], vec![7]),
                edge(0.13, vec![20, 30], vec![9]),
                edge(0.71, vec![10, 30], vec![7, 9]),
                edge(1.0, vec![20], vec![]),
            ],
        )
        .unwrap();
        let detector_ids = [30, 10, 20];
        let observable_ids = [9, 7];
        let plan = simulator
            .compile_sampling_plan(&detector_ids, &observable_ids)
            .unwrap();

        for shots in [1, 63, 64, 65, 129] {
            for seed in [0, 1, 0xdead_beef] {
                let mut generic_rng = SmallRng::new(seed);
                let generic = simulator.run_batch_with_rng(shots, &mut generic_rng, false);
                let expected = packed_reference(&generic, &detector_ids, &observable_ids);
                let mut compiled_rng = SmallRng::new(seed);
                let actual = plan.run_packed_shot_batch_with_rng(shots, &mut compiled_rng);
                assert_eq!(actual.detector_data, expected.0);
                assert_eq!(actual.observable_data, expected.1);
                assert_eq!(compiled_rng.next_u64(), generic_rng.next_u64());
            }
        }
    }

    #[test]
    fn compiled_event_batch_preserves_event_order_and_duplicates() {
        let simulator = DemHotspotEstimator::from_sampling_parts(
            vec![10, 20],
            vec![7, 9],
            vec![
                edge(1.0, vec![20, 10, 20], vec![9]),
                edge(1.0, vec![10], vec![9, 9, 7]),
            ],
        )
        .unwrap();
        let plan = simulator.compile_sampling_plan(&[20, 10], &[9, 7]).unwrap();
        let batch = plan.run_detector_event_shot_batch_with_rng(2, &mut SmallRng::new(4));

        assert_eq!(batch.offsets, vec![0, 4, 8]);
        assert_eq!(batch.events, vec![0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(batch.observable_data, vec![0b11, 0b11]);
    }

    #[test]
    fn compiled_logical_count_matches_generic_loss_and_rng_sequence() {
        let simulator = DemHotspotEstimator::from_sampling_parts(
            vec![1],
            vec![5],
            vec![
                edge(0.17, vec![1], vec![]),
                edge(0.0, vec![], vec![5]),
                edge(0.23, vec![], vec![5, 9]),
                edge(0.61, vec![], vec![9, 9]),
                edge(1.0, vec![], vec![11]),
            ],
        )
        .unwrap();
        let plan = simulator.compile_logical_count_plan();
        assert_eq!(plan.observable_ids(), &[5, 9, 11]);

        for shots in [0, 1, 63, 64, 65, 129] {
            for seed in [0, 1, 99] {
                let mut generic_rng = SmallRng::new(seed);
                let generic = simulator.run_batch_with_rng(shots, &mut generic_rng, false);
                let mut compiled_rng = SmallRng::new(seed);
                let actual = plan.sample_logical_error_count_with_rng(shots, &mut compiled_rng);
                assert_eq!(actual, generic.loss_mask.bit_count());
                assert_eq!(compiled_rng.next_u64(), generic_rng.next_u64());
            }
        }
    }

    #[test]
    fn single_observable_zero_allocation_path_preserves_xor_and_rng() {
        let simulator = DemHotspotEstimator::from_sampling_parts(
            vec![1],
            vec![5],
            vec![
                edge(0.19, vec![1], vec![]),
                edge(0.23, vec![], vec![5]),
                edge(0.61, vec![], vec![5]),
                edge(0.47, vec![], vec![5, 5]),
                edge(1.0, vec![], vec![5]),
            ],
        )
        .unwrap();
        let plan = simulator.compile_logical_count_plan();

        for shots in [0, 1, 63, 64, 65, 129, 513] {
            for seed in [0, 7, 1234] {
                let mut generic_rng = SmallRng::new(seed);
                let generic = simulator.run_batch_with_rng(shots, &mut generic_rng, false);
                let mut compiled_rng = SmallRng::new(seed);
                let actual = plan.sample_logical_error_count_with_rng(shots, &mut compiled_rng);
                assert_eq!(actual, generic.loss_mask.bit_count());
                assert_eq!(compiled_rng.next_u64(), generic_rng.next_u64());
            }
        }
    }

    #[test]
    fn compiled_sampling_preserves_validation_errors() {
        let simulator = DemHotspotEstimator::from_sampling_parts(
            vec![1],
            vec![2],
            vec![edge(0.5, vec![1], vec![2])],
        )
        .unwrap();
        let detector_error = simulator.compile_sampling_plan(&[9], &[2]).unwrap_err();
        assert_eq!(
            detector_error.message(),
            "packed DEM sampler requested detector id 9, but it is not declared by the DEM"
        );
        let observable_error = simulator.compile_sampling_plan(&[1], &[9]).unwrap_err();
        assert_eq!(
            observable_error.message(),
            "packed DEM sampler requested observable id 9, but it is not declared by the DEM"
        );
    }

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

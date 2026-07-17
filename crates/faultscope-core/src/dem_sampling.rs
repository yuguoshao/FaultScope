use std::collections::HashMap;
use std::sync::Arc;

use crate::dem_canonical::{canonical_id_order, parity_canonicalize};
use crate::labels::LocationCatalogBuilder;
use crate::{
    bernoulli_mask, for_each_bernoulli_event, word_count, DemBatch, DemEvent, DemHotspotEstimate,
    DemLocationGroup, DetectorErrorEdge, DetectorErrorModel, LocationCatalog, LocationId, Mask,
    NpError, NpResult, SmallRng,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DemProgramEdge {
    pub(crate) probability: f64,
    pub(crate) detectors: Vec<i64>,
    pub(crate) observables: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DemProgramEdgeMetadata {
    pub(crate) location_id: LocationId,
    pub(crate) event: DemEvent,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DemProgramLocationGroup {
    pub(crate) location_id: LocationId,
    pub(crate) edge_indices: Vec<usize>,
    pub(crate) total_probability: f64,
}

struct CompiledDemEdges {
    edges: Vec<DemProgramEdge>,
    metadata: Option<Vec<DemProgramEdgeMetadata>>,
    catalog: LocationCatalog,
    location_groups: Vec<DemProgramLocationGroup>,
}

/// Bit-packed detector-error-model sampler and hotspot estimator.
///
/// This simulator samples DEM edges as independent Bernoulli error
/// instructions, computes detector and observable masks, and can aggregate
/// edge and location hotspot estimates entirely in Rust.
#[derive(Debug, Clone, PartialEq)]
pub struct DemHotspotEstimator {
    /// Construction-time detector ID mirror retained for source compatibility.
    ///
    /// Operational methods use [`Self::detector_ids`] instead. Mutating this
    /// field does not recompile or otherwise change the estimator.
    pub detector_ids: Vec<i64>,
    /// Construction-time observable ID mirror retained for source compatibility.
    ///
    /// Operational methods use [`Self::observable_ids`] instead. Mutating this
    /// field does not recompile or otherwise change the estimator.
    pub observable_ids: Vec<i64>,
    canonical_detector_ids: Arc<[i64]>,
    canonical_observable_ids: Arc<[i64]>,
    edges: Vec<DemProgramEdge>,
    edge_metadata: Option<Vec<DemProgramEdgeMetadata>>,
    location_catalog: LocationCatalog,
    location_groups: Vec<DemProgramLocationGroup>,
    hotspot_layout_identity: Arc<()>,
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
    pub(crate) fn from_compact_sampling_parts(
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        edges: Vec<DemProgramEdge>,
    ) -> Self {
        let (detector_ids, observable_ids, edges) =
            canonicalize_program_parts(detector_ids, observable_ids, edges)
                .expect("generated DEM declarations must have unique ids");
        Self {
            canonical_detector_ids: Arc::from(detector_ids.as_slice()),
            canonical_observable_ids: Arc::from(observable_ids.as_slice()),
            detector_ids,
            observable_ids,
            edges,
            edge_metadata: None,
            location_catalog: LocationCatalog::default(),
            location_groups: Vec::new(),
            hotspot_layout_identity: Arc::new(()),
        }
    }

    pub(crate) fn from_program_parts(
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        edges: Vec<DemProgramEdge>,
        edge_metadata: Vec<DemProgramEdgeMetadata>,
        location_catalog: LocationCatalog,
    ) -> Self {
        debug_assert_eq!(edges.len(), edge_metadata.len());
        let (detector_ids, observable_ids, edges) =
            canonicalize_program_parts(detector_ids, observable_ids, edges)
                .expect("generated DEM declarations must have unique ids");
        let location_groups =
            build_dem_program_location_groups(&edges, &edge_metadata, location_catalog.len());
        Self {
            canonical_detector_ids: Arc::from(detector_ids.as_slice()),
            canonical_observable_ids: Arc::from(observable_ids.as_slice()),
            detector_ids,
            observable_ids,
            edges,
            edge_metadata: Some(edge_metadata),
            location_catalog,
            location_groups,
            hotspot_layout_identity: Arc::new(()),
        }
    }

    /// Build a simulator from a typed detector error model.
    pub fn new(dem: DetectorErrorModel) -> NpResult<Self> {
        let detector_ids = dem.detectors.iter().map(|detector| detector.id).collect();
        let observable_ids = dem
            .observables
            .iter()
            .map(|observable| observable.id)
            .collect();
        Self::from_parts(detector_ids, observable_ids, dem.edges)
    }

    /// Build a simulator from pre-extracted ids and sampler edges.
    pub fn from_parts(
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        edges: Vec<DetectorErrorEdge>,
    ) -> NpResult<Self> {
        let compiled = compile_dem_edges(edges, true)?;
        let (detector_ids, observable_ids, edges) =
            canonicalize_program_parts(detector_ids, observable_ids, compiled.edges)?;
        Ok(Self {
            canonical_detector_ids: Arc::from(detector_ids.as_slice()),
            canonical_observable_ids: Arc::from(observable_ids.as_slice()),
            detector_ids,
            observable_ids,
            edges,
            edge_metadata: compiled.metadata,
            location_catalog: compiled.catalog,
            location_groups: compiled.location_groups,
            hotspot_layout_identity: Arc::new(()),
        })
    }

    /// Build a simulator for sampling only, without hotspot location metadata.
    pub fn from_sampling_parts(
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        edges: Vec<DetectorErrorEdge>,
    ) -> NpResult<Self> {
        let compiled = compile_dem_edges(edges, false)?;
        debug_assert!(compiled.metadata.is_none());
        debug_assert!(compiled.catalog.is_empty());
        let (detector_ids, observable_ids, edges) =
            canonicalize_program_parts(detector_ids, observable_ids, compiled.edges)?;
        Ok(Self {
            canonical_detector_ids: Arc::from(detector_ids.as_slice()),
            canonical_observable_ids: Arc::from(observable_ids.as_slice()),
            detector_ids,
            observable_ids,
            edges,
            edge_metadata: None,
            location_catalog: LocationCatalog::default(),
            location_groups: Vec::new(),
            hotspot_layout_identity: Arc::new(()),
        })
    }

    /// Number of compiled DEM edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Detector IDs in the immutable canonical sampling order.
    pub fn detector_ids(&self) -> &[i64] {
        self.canonical_detector_ids.as_ref()
    }

    /// Observable IDs in the immutable canonical sampling order.
    pub fn observable_ids(&self) -> &[i64] {
        self.canonical_observable_ids.as_ref()
    }

    /// Materialize one user-facing DEM edge at the output boundary.
    pub fn edge(&self, edge_index: usize) -> Option<DetectorErrorEdge> {
        self.edges.get(edge_index).map(|edge| {
            materialize_dem_edge(
                edge,
                self.edge_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get(edge_index)),
                &self.location_catalog,
            )
        })
    }

    /// Materialize all user-facing DEM edges at the output boundary.
    pub fn edges(&self) -> Vec<DetectorErrorEdge> {
        self.edges
            .iter()
            .enumerate()
            .map(|(edge_index, edge)| {
                materialize_dem_edge(
                    edge,
                    self.edge_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get(edge_index)),
                    &self.location_catalog,
                )
            })
            .collect()
    }

    /// Materialize grouped location metadata at the output boundary.
    pub fn location_groups(&self) -> Vec<DemLocationGroup> {
        self.location_groups
            .iter()
            .map(|group| DemLocationGroup {
                location_id: self.location_catalog.label(group.location_id).to_string(),
                edge_indices: group.edge_indices.clone(),
                total_probability: group.total_probability,
            })
            .collect()
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
        run_dem_program_batch(
            self.detector_ids(),
            self.observable_ids(),
            &self.edges,
            shots,
            rng,
            return_edge_events,
            &self.hotspot_layout_identity,
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
            self.detector_ids(),
            self.observable_ids(),
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
            self.detector_ids(),
            self.observable_ids(),
            &self.edges,
            detector_ids,
            observable_ids,
            "packed DEM sampler",
        )
    }

    /// Compile the observable-only path used to count default logical failures.
    pub fn compile_logical_count_plan(&self) -> CompiledDemLogicalCountPlan {
        CompiledDemLogicalCountPlan::new(self.observable_ids(), &self.edges)
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
        self.estimate_from_loss(&batch, &batch.loss_mask, baseline, top_k)
    }

    /// Estimate hotspots from an externally supplied loss mask.
    ///
    /// The batch must come from this estimator (or one of its clones) and must
    /// have edge-event recording enabled.
    pub fn estimate_from_loss(
        &self,
        batch: &DemBatch,
        loss_mask: &Mask,
        baseline: Option<f64>,
        top_k: usize,
    ) -> NpResult<DemHotspotEstimate> {
        if !batch.matches_hotspot_layout(&self.hotspot_layout_identity) {
            return Err(NpError::new(
                "hotspot batch layout does not match the current DEM estimator",
            ));
        }
        if batch.shots() == 0 {
            return Err(NpError::new(
                "DEM hotspot estimation requires shots to be positive",
            ));
        }
        if !batch.records_edge_events() {
            return Err(NpError::new(
                "DEM hotspot estimation requires recorded event masks",
            ));
        }
        crate::hotspot::validate_loss_mask_width("DEM hotspot", batch.shots(), loss_mask)?;
        Ok(crate::hotspot::compute_dem_estimate(
            crate::hotspot::DemEstimateProgram {
                edges: &self.edges,
                edge_metadata: self.edge_metadata.as_deref().unwrap_or(&[]),
                location_groups: &self.location_groups,
                catalog: &self.location_catalog,
            },
            batch,
            loss_mask,
            baseline,
            top_k,
        ))
    }
}

fn compile_dem_sampling_plan(
    sampling_detector_ids: &[i64],
    sampling_observable_ids: &[i64],
    edges: &[DemProgramEdge],
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
    fn new(observable_ids: &[i64], edges: &[DemProgramEdge]) -> Self {
        let compiled_observable_ids = observable_ids.to_vec();
        let observable_index = id_index(observable_ids);
        let compiled_edges = edges
            .iter()
            .map(|edge| {
                let observable_columns = edge
                    .observables
                    .iter()
                    .map(|observable_id| {
                        *observable_index
                            .get(observable_id)
                            .expect("canonical DEM layout must include every edge observable")
                    })
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

fn run_dem_program_batch(
    detector_ids: &[i64],
    observable_ids: &[i64],
    edges: &[DemProgramEdge],
    shots: usize,
    rng: &mut SmallRng,
    return_edge_events: bool,
    hotspot_layout_identity: &Arc<()>,
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
    let mut edge_event_masks = return_edge_events.then(|| Vec::with_capacity(edges.len()));

    for edge in edges {
        let event_mask = bernoulli_mask(rng, shots, edge.probability);
        if !event_mask.is_zero() {
            for detector_id in &edge.detectors {
                detectors
                    .get_mut(detector_id)
                    .expect("canonical DEM layout must include every edge detector")
                    .xor_assign(&event_mask);
            }
            for observable_id in &edge.observables {
                observables
                    .get_mut(observable_id)
                    .expect("canonical DEM layout must include every edge observable")
                    .xor_assign(&event_mask);
            }
        }
        if let Some(edge_event_masks) = &mut edge_event_masks {
            edge_event_masks.push(event_mask);
        }
    }

    let mut loss_mask = Mask::zero(words);
    for observable in observables.values() {
        loss_mask.or_assign(observable);
    }
    loss_mask.and_assign(&all_mask);

    DemBatch::new(
        shots,
        all_mask,
        detectors,
        observables,
        edge_event_masks,
        loss_mask,
        hotspot_layout_identity.clone(),
    )
}

fn canonicalize_program_parts(
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    mut edges: Vec<DemProgramEdge>,
) -> NpResult<(Vec<i64>, Vec<i64>, Vec<DemProgramEdge>)> {
    let detector_ids = canonical_id_order(
        &detector_ids,
        edges.iter().map(|edge| edge.detectors.as_slice()),
        "detector ids must be unique",
    )?;
    let observable_ids = canonical_id_order(
        &observable_ids,
        edges.iter().map(|edge| edge.observables.as_slice()),
        "logical observable ids must be unique",
    )?;
    for edge in &mut edges {
        parity_canonicalize(&mut edge.detectors);
        parity_canonicalize(&mut edge.observables);
    }
    Ok((detector_ids, observable_ids, edges))
}

fn id_index(ids: &[i64]) -> HashMap<i64, usize> {
    ids.iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect()
}

fn compile_dem_edges(
    edges: Vec<DetectorErrorEdge>,
    preserve_metadata: bool,
) -> NpResult<CompiledDemEdges> {
    let mut catalog = preserve_metadata.then(|| LocationCatalogBuilder::with_capacity(edges.len()));
    let mut compiled = Vec::with_capacity(edges.len());
    let mut metadata = preserve_metadata.then(|| Vec::with_capacity(edges.len()));
    for edge in edges {
        edge.validate()?;
        let DetectorErrorEdge {
            probability,
            detectors,
            observables,
            location_id,
            event,
            tags,
        } = edge;
        if let (Some(catalog), Some(metadata)) = (&mut catalog, &mut metadata) {
            metadata.push(DemProgramEdgeMetadata {
                location_id: catalog.intern(location_id, tags),
                event,
            });
        }
        compiled.push(DemProgramEdge {
            probability,
            detectors,
            observables,
        });
    }
    let catalog = catalog
        .map(LocationCatalogBuilder::finish)
        .unwrap_or_default();
    let groups = metadata
        .as_deref()
        .map(|metadata| build_dem_program_location_groups(&compiled, metadata, catalog.len()))
        .unwrap_or_default();
    Ok(CompiledDemEdges {
        edges: compiled,
        metadata,
        catalog,
        location_groups: groups,
    })
}

fn build_dem_program_location_groups(
    edges: &[DemProgramEdge],
    edge_metadata: &[DemProgramEdgeMetadata],
    location_count: usize,
) -> Vec<DemProgramLocationGroup> {
    let mut group_indices = vec![None; location_count];
    let mut groups = Vec::with_capacity(location_count);
    for (edge_index, metadata) in edge_metadata.iter().enumerate() {
        let location_index = metadata.location_id.index();
        let group_index = match group_indices[location_index] {
            Some(group_index) => group_index,
            None => {
                let group_index = groups.len();
                group_indices[location_index] = Some(group_index);
                groups.push(DemProgramLocationGroup {
                    location_id: metadata.location_id,
                    edge_indices: Vec::new(),
                    total_probability: 0.0,
                });
                group_index
            }
        };
        let group = &mut groups[group_index];
        group.edge_indices.push(edge_index);
        group.total_probability += edges[edge_index].probability;
    }
    groups
}

fn materialize_dem_edge(
    edge: &DemProgramEdge,
    metadata: Option<&DemProgramEdgeMetadata>,
    catalog: &LocationCatalog,
) -> DetectorErrorEdge {
    let (location_id, event, tags) = match metadata {
        Some(metadata) => (
            catalog.label(metadata.location_id).to_string(),
            metadata.event.clone(),
            catalog.tags(metadata.location_id).clone(),
        ),
        None => (String::new(), DemEvent::Bool(false), HashMap::new()),
    };
    DetectorErrorEdge {
        probability: edge.probability,
        detectors: edge.detectors.clone(),
        observables: edge.observables.clone(),
        location_id,
        event,
        tags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(probability: f64, detectors: Vec<i64>, observables: Vec<i64>) -> DetectorErrorEdge {
        DetectorErrorEdge {
            probability,
            detectors,
            observables,
            location_id: "test".to_string(),
            event: crate::DemEvent::Pauli("X".to_string()),
            tags: HashMap::new(),
        }
    }

    #[test]
    fn sampling_only_edges_keep_metadata_out_of_the_hot_representation() {
        let simulator = DemHotspotEstimator::from_sampling_parts(
            vec![1],
            vec![2],
            vec![edge(0.25, vec![1], vec![2])],
        )
        .unwrap();

        assert!(simulator.edge_metadata.is_none());
        assert!(simulator.location_catalog.is_empty());
        let materialized = simulator.edge(0).unwrap();
        assert_eq!(materialized.probability, 0.25);
        assert_eq!(materialized.detectors, vec![1]);
        assert_eq!(materialized.observables, vec![2]);
        assert!(materialized.location_id.is_empty());
        assert_eq!(materialized.event, DemEvent::Bool(false));
        assert!(materialized.tags.is_empty());
    }

    fn packed_reference(
        batch: &DemBatch,
        detector_ids: &[i64],
        observable_ids: &[i64],
    ) -> (Vec<u8>, Vec<u8>) {
        let detector_bytes = detector_ids.len().div_ceil(8);
        let observable_bytes = observable_ids.len().div_ceil(8);
        let mut detectors = vec![0u8; batch.shots() * detector_bytes];
        let mut observables = vec![0u8; batch.shots() * observable_bytes];
        for shot in 0..batch.shots() {
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
    fn compiled_event_batch_emits_parity_canonical_events() {
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

        assert_eq!(batch.offsets, vec![0, 2, 4]);
        assert_eq!(batch.events, vec![1, 1, 1, 1]);
        assert_eq!(batch.observable_data, vec![0b11, 0b11]);
    }

    #[test]
    fn estimator_canonicalizes_implicit_ids_and_parity_once() {
        let simulator = DemHotspotEstimator::from_sampling_parts(
            Vec::new(),
            Vec::new(),
            vec![edge(1.0, vec![1, 1], vec![9, 9])],
        )
        .unwrap();

        assert_eq!(simulator.detector_ids, vec![1]);
        assert_eq!(simulator.observable_ids, vec![9]);
        assert_eq!(simulator.edge_count(), 1);
        assert!(simulator.edge(0).unwrap().detectors.is_empty());
        assert!(simulator.edge(0).unwrap().observables.is_empty());

        let mut generic_rng = SmallRng::new(13);
        let generic = simulator.run_batch_with_rng(4, &mut generic_rng, true);
        assert_eq!(generic.detectors[&1].bit_count(), 0);
        assert_eq!(generic.observables[&9].bit_count(), 0);
        assert_eq!(generic.edge_event_masks().len(), 1);
        assert_eq!(generic.edge_event_masks()[0].bit_count(), 4);

        let plan = simulator.compile_sampling_plan(&[1], &[9]).unwrap();
        let mut packed_rng = SmallRng::new(13);
        let packed = plan.run_packed_shot_batch_with_rng(4, &mut packed_rng);
        assert_eq!(packed.detector_data, vec![0; 4]);
        assert_eq!(packed.observable_data, vec![0; 4]);
        assert_eq!(generic_rng.next_u64(), packed_rng.next_u64());
    }

    #[test]
    fn parity_cancelled_edges_preserve_rng_across_sampling_paths() {
        for cancelled_probability in [0.0, 0.37, 1.0] {
            let simulator = DemHotspotEstimator::from_sampling_parts(
                Vec::new(),
                Vec::new(),
                vec![
                    edge(cancelled_probability, vec![1, 1], vec![9, 9]),
                    edge(0.43, vec![2], vec![10]),
                ],
            )
            .unwrap();
            let detector_ids = simulator.detector_ids();
            let observable_ids = simulator.observable_ids();
            let plan = simulator
                .compile_sampling_plan(detector_ids, observable_ids)
                .unwrap();

            let mut generic_rng = SmallRng::new(123);
            let generic = simulator.run_batch_with_rng(129, &mut generic_rng, true);
            let expected = packed_reference(&generic, detector_ids, observable_ids);
            let generic_next = generic_rng.next_u64();
            assert_eq!(generic.edge_event_masks().len(), 2);
            assert_eq!(generic.detectors[&1].bit_count(), 0);
            assert_eq!(generic.observables[&9].bit_count(), 0);

            let mut packed_rng = SmallRng::new(123);
            let packed = plan.run_packed_shot_batch_with_rng(129, &mut packed_rng);
            assert_eq!(packed.detector_data, expected.0);
            assert_eq!(packed.observable_data, expected.1);
            assert_eq!(packed_rng.next_u64(), generic_next);

            let mut expected_offsets = Vec::with_capacity(130);
            let mut expected_events = Vec::new();
            expected_offsets.push(0);
            for shot in 0..129 {
                if generic.detectors[&2].words[shot / 64] & (1u64 << (shot % 64)) != 0 {
                    expected_events.push(1);
                }
                expected_offsets.push(expected_events.len());
            }
            let mut event_rng = SmallRng::new(123);
            let events = plan.run_detector_event_shot_batch_with_rng(129, &mut event_rng);
            assert_eq!(events.offsets, expected_offsets);
            assert_eq!(events.events, expected_events);
            assert_eq!(events.observable_data, expected.1);
            assert_eq!(event_rng.next_u64(), generic_next);
        }
    }

    #[test]
    fn public_id_mirrors_do_not_mutate_the_canonical_sampling_layout() {
        let mut simulator = DemHotspotEstimator::from_sampling_parts(
            Vec::new(),
            Vec::new(),
            vec![edge(0.43, vec![7], vec![9])],
        )
        .unwrap();
        let mut expected_rng = SmallRng::new(17);
        let expected = simulator.run_batch_with_rng(129, &mut expected_rng, false);
        let expected_rng_next = expected_rng.next_u64();
        let expected_packed = packed_reference(&expected, &[7], &[9]);

        simulator.detector_ids.clear();
        simulator.detector_ids.push(70);
        simulator.observable_ids.clear();
        simulator.observable_ids.push(90);

        assert_eq!(simulator.detector_ids(), &[7]);
        assert_eq!(simulator.observable_ids(), &[9]);
        assert_eq!(simulator.run_batch(129, Some(17), false).unwrap(), expected);
        assert_eq!(
            simulator.compile_logical_count_plan().observable_ids(),
            &[9]
        );
        let plan = simulator
            .compile_sampling_plan(simulator.detector_ids(), simulator.observable_ids())
            .unwrap();
        assert_eq!(plan.detector_ids(), &[7]);
        assert_eq!(plan.observable_ids(), &[9]);

        let mut packed_rng = SmallRng::new(17);
        let packed = plan.run_packed_shot_batch_with_rng(129, &mut packed_rng);
        assert_eq!(packed.detector_data, expected_packed.0);
        assert_eq!(packed.observable_data, expected_packed.1);
        assert_eq!(packed_rng.next_u64(), expected_rng_next);

        let mut expected_offsets = Vec::with_capacity(130);
        let mut expected_events = Vec::new();
        expected_offsets.push(0);
        for shot in 0..129 {
            if expected.detectors[&7].words[shot / 64] & (1u64 << (shot % 64)) != 0 {
                expected_events.push(0);
            }
            expected_offsets.push(expected_events.len());
        }
        let mut event_rng = SmallRng::new(17);
        let events = plan.run_detector_event_shot_batch_with_rng(129, &mut event_rng);
        assert_eq!(events.offsets, expected_offsets);
        assert_eq!(events.events, expected_events);
        assert_eq!(events.observable_data, expected_packed.1);
        assert_eq!(event_rng.next_u64(), expected_rng_next);
    }

    #[test]
    fn estimator_rejects_duplicate_declared_ids() {
        let err = DemHotspotEstimator::from_sampling_parts(vec![1, 1], Vec::new(), Vec::new())
            .unwrap_err();

        assert_eq!(err.message(), "detector ids must be unique");
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
            DetectorErrorEdge {
                probability: 0.1,
                detectors: vec![0],
                observables: Vec::new(),
                location_id: "a".to_string(),
                event: crate::DemEvent::Pauli("X".to_string()),
                tags: HashMap::new(),
            },
            DetectorErrorEdge {
                probability: 0.2,
                detectors: vec![1],
                observables: Vec::new(),
                location_id: "a".to_string(),
                event: crate::DemEvent::Pauli("Z".to_string()),
                tags: HashMap::new(),
            },
        ];

        let groups = DemHotspotEstimator::from_parts(Vec::new(), Vec::new(), edges)
            .unwrap()
            .location_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].edge_indices, vec![0, 1]);
        assert_eq!(groups[0].total_probability, 0.30000000000000004);
    }

    #[test]
    fn simulator_estimates_default_logical_loss() {
        let simulator = DemHotspotEstimator::from_parts(
            Vec::new(),
            vec![0],
            vec![DetectorErrorEdge {
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
    fn simulator_hotspot_rejects_unrecorded_foreign_and_invalid_loss_batches() {
        let make_simulator = || {
            DemHotspotEstimator::from_parts(
                Vec::new(),
                vec![0],
                vec![edge(0.25, Vec::new(), vec![0])],
            )
            .unwrap()
        };
        let simulator = make_simulator();

        let batch = simulator.run_batch(8, Some(1), false).unwrap();
        let err = simulator
            .estimate_from_loss(&batch, &batch.loss_mask, None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("recorded event masks"));

        let batch = simulator.run_batch(8, Some(1), true).unwrap();
        let foreign = make_simulator();
        let err = foreign
            .estimate_from_loss(&batch, &batch.loss_mask, None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("layout does not match"));
        simulator
            .clone()
            .estimate_from_loss(&batch, &batch.loss_mask, None, 1)
            .unwrap();

        let err = simulator
            .estimate_from_loss(&batch, &Mask::zero(2), None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("loss_mask has 2 words"));

        let mut rng = SmallRng::new(1);
        let zero_shot_batch = simulator.run_batch_with_rng(0, &mut rng, true);
        let err = simulator
            .estimate_from_loss(&zero_shot_batch, &zero_shot_batch.loss_mask, None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("shots to be positive"));
    }

    #[test]
    fn simulator_rejects_invalid_edge_probability() {
        let err = DemHotspotEstimator::from_parts(
            Vec::new(),
            Vec::new(),
            vec![DetectorErrorEdge {
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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::counting::{
    prepare_dem_count_plan, prepare_forward_count_plan, sample_dem_logical_error_stats_with_rng,
    sample_forward_logical_error_stats as sample_forward_logical_error_stats_batch,
    validate_decoder_observable_layout, validate_mask_shape, CountOptions,
};
use crate::scheduler::{
    batch_seed, collect_task_set, collect_task_set_with_progress, next_batch_size,
};
use faultscope_core::{
    validate_decoder_batch_formats, validate_decoder_detector_ids, DemHotspotEstimator,
    DetectorBatchFormat, NativeDecoderFactory, NativeDecoderWorker, NpError, NpResult,
    SamplerOperation, SamplerProgram, SmallRng,
};

pub const DEM_LOGICAL_COUNTER_SCHEMA_VERSION: u32 = 1;

pub(crate) const DETECTION_EVENTS_KEY: &str = "detection_events";
pub(crate) const DETECTORS_CHECKED_KEY: &str = "detectors_checked";
pub(crate) const OBSERVABLE_COMBO_PREFIX: &str = "obs_mistake_mask=";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemLogicalCounterSchema {
    pub schema_version: u32,
    pub count_observable_error_combos: bool,
    pub count_detection_events: bool,
}

impl Default for DemLogicalCounterSchema {
    fn default() -> Self {
        Self {
            schema_version: DEM_LOGICAL_COUNTER_SCHEMA_VERSION,
            count_observable_error_combos: false,
            count_detection_events: false,
        }
    }
}

impl DemLogicalCounterSchema {
    pub fn new(count_observable_error_combos: bool, count_detection_events: bool) -> Self {
        Self {
            schema_version: DEM_LOGICAL_COUNTER_SCHEMA_VERSION,
            count_observable_error_combos,
            count_detection_events,
        }
    }

    pub fn validate(self) -> NpResult<()> {
        if self.schema_version != DEM_LOGICAL_COUNTER_SCHEMA_VERSION {
            return Err(NpError::new(format!(
                "unsupported collection counter schema version {}; expected {}",
                self.schema_version, DEM_LOGICAL_COUNTER_SCHEMA_VERSION
            )));
        }
        Ok(())
    }

    fn empty_custom_counts(self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        if self.count_detection_events {
            counts.insert(DETECTION_EVENTS_KEY.to_string(), 0);
            counts.insert(DETECTORS_CHECKED_KEY.to_string(), 0);
        }
        counts
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemLogicalCollectionOptions {
    pub max_shots: usize,
    pub min_shots: usize,
    pub max_errors: Option<usize>,
    pub batch_size: usize,
    pub seed: Option<u64>,
    pub start_batch_size: Option<usize>,
    pub max_batch_size: Option<usize>,
    pub max_batch_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemLogicalCollectionStats {
    pub task_id: String,
    pub strong_id: String,
    pub decoder: Option<String>,
    pub metadata_json: String,
    pub shots: usize,
    pub errors: usize,
    pub discards: usize,
    pub seconds: f64,
    pub counter_schema: DemLogicalCounterSchema,
    pub custom_counts: HashMap<String, usize>,
}

impl DemLogicalCollectionStats {
    pub fn empty_for_task(
        task: &DemLogicalCollectionTask,
        counter_schema: DemLogicalCounterSchema,
    ) -> Self {
        Self {
            task_id: task.task_id.clone(),
            strong_id: task.strong_id.clone(),
            decoder: task.decoder_name.clone(),
            metadata_json: task.metadata_json.clone(),
            shots: 0,
            errors: 0,
            discards: 0,
            seconds: 0.0,
            counter_schema,
            custom_counts: counter_schema.empty_custom_counts(),
        }
    }

    pub(crate) fn empty_for_logical_task(
        task: &LogicalCollectionTask,
        counter_schema: DemLogicalCounterSchema,
    ) -> Self {
        Self {
            task_id: task.task_id.clone(),
            strong_id: task.strong_id.clone(),
            decoder: task.decoder_name.clone(),
            metadata_json: task.metadata_json.clone(),
            shots: 0,
            errors: 0,
            discards: 0,
            seconds: 0.0,
            counter_schema,
            custom_counts: counter_schema.empty_custom_counts(),
        }
    }

    pub fn with_identity(
        mut self,
        task_id: String,
        strong_id: String,
        decoder: Option<String>,
        metadata_json: String,
    ) -> Self {
        self.task_id = task_id;
        self.strong_id = strong_id;
        self.decoder = decoder;
        self.metadata_json = metadata_json;
        self
    }

    pub fn add_assign_checked(&mut self, other: &Self) -> NpResult<()> {
        self.validate_counter_schema()?;
        other.validate_counter_schema()?;
        if self.strong_id != other.strong_id {
            return Err(NpError::new(format!(
                "cannot merge stats with different strong_id values: {:?} != {:?}",
                self.strong_id, other.strong_id
            )));
        }
        if self.decoder != other.decoder || self.metadata_json != other.metadata_json {
            return Err(NpError::new(
                "stats with the same strong_id have different decoder or metadata",
            ));
        }
        if self.counter_schema != other.counter_schema {
            return Err(NpError::new(
                "stats with the same strong_id have different counter schemas",
            ));
        }
        self.shots += other.shots;
        self.errors += other.errors;
        self.discards += other.discards;
        self.seconds += other.seconds;
        for (key, value) in &other.custom_counts {
            *self.custom_counts.entry(key.clone()).or_insert(0) += *value;
        }
        Ok(())
    }

    pub fn validate_counter_schema(&self) -> NpResult<()> {
        self.counter_schema.validate()?;
        for fixed_key in [DETECTION_EVENTS_KEY, DETECTORS_CHECKED_KEY] {
            let present = self.custom_counts.contains_key(fixed_key);
            if self.counter_schema.count_detection_events != present {
                let expectation = if self.counter_schema.count_detection_events {
                    "must contain"
                } else {
                    "must not contain"
                };
                return Err(NpError::new(format!(
                    "collection stats {expectation} counter {fixed_key:?} for their counter schema"
                )));
            }
        }
        for key in self.custom_counts.keys() {
            if key == DETECTION_EVENTS_KEY || key == DETECTORS_CHECKED_KEY {
                continue;
            }
            let Some(mask) = key.strip_prefix(OBSERVABLE_COMBO_PREFIX) else {
                return Err(NpError::new(format!(
                    "collection stats contain unsupported custom counter {key:?}"
                )));
            };
            if !self.counter_schema.count_observable_error_combos {
                return Err(NpError::new(format!(
                    "collection stats contain observable combo counter {key:?} but their counter schema disables observable combos"
                )));
            }
            validate_observable_combo_mask_syntax(mask)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct DemLogicalCollectionTask {
    pub task_id: String,
    pub strong_id: String,
    pub sampling_id: String,
    pub sampler: Arc<DemHotspotEstimator>,
    pub decoder: Option<Arc<dyn NativeDecoderFactory>>,
    pub decoder_name: Option<String>,
    pub metadata_json: String,
    pub options: DemLogicalCollectionOptions,
    pub postselection_mask: Option<Vec<u8>>,
    pub postselected_observables_mask: Option<Vec<u8>>,
}

/// A forward-circuit logical-error collection task.
#[derive(Clone)]
pub struct ForwardLogicalCollectionTask {
    pub task_id: String,
    pub strong_id: String,
    pub sampling_id: String,
    pub sampler: Arc<SamplerProgram>,
    pub decoder: Option<Arc<dyn NativeDecoderFactory>>,
    pub decoder_name: Option<String>,
    pub metadata_json: String,
    pub options: DemLogicalCollectionOptions,
    pub postselection_mask: Option<Vec<u8>>,
    pub postselected_observables_mask: Option<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) enum CollectionSampler {
    Dem(Arc<DemHotspotEstimator>),
    Forward(Arc<ForwardCollectionSampler>),
}

#[derive(Debug)]
pub(crate) struct ForwardCollectionSampler {
    program: Arc<SamplerProgram>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl ForwardCollectionSampler {
    fn new(program: Arc<SamplerProgram>) -> Self {
        let (detector_ids, observable_ids) = forward_sampler_layout(&program);
        Self {
            program,
            detector_ids,
            observable_ids,
        }
    }

    pub(crate) fn program(&self) -> &SamplerProgram {
        &self.program
    }

    pub(crate) fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    pub(crate) fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    pub(crate) fn view(&self) -> ForwardSamplerView<'_> {
        ForwardSamplerView::new(&self.program, &self.detector_ids, &self.observable_ids)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ForwardSamplerView<'a> {
    program: &'a SamplerProgram,
    detector_ids: &'a [i64],
    observable_ids: &'a [i64],
}

impl<'a> ForwardSamplerView<'a> {
    pub(crate) fn new(
        program: &'a SamplerProgram,
        detector_ids: &'a [i64],
        observable_ids: &'a [i64],
    ) -> Self {
        Self {
            program,
            detector_ids,
            observable_ids,
        }
    }

    pub(crate) fn program(self) -> &'a SamplerProgram {
        self.program
    }

    pub(crate) fn detector_ids(self) -> &'a [i64] {
        self.detector_ids
    }

    pub(crate) fn observable_ids(self) -> &'a [i64] {
        self.observable_ids
    }
}

pub(crate) fn forward_sampler_layout(program: &SamplerProgram) -> (Vec<i64>, Vec<i64>) {
    let mut detector_ids = Vec::new();
    let mut observable_ids = Vec::new();
    let mut seen_detector_ids = HashSet::new();
    let mut seen_observable_ids = HashSet::new();
    for operation in program.operations() {
        match operation {
            SamplerOperation::Detector { detector_id, .. } => {
                if seen_detector_ids.insert(*detector_id) {
                    detector_ids.push(*detector_id);
                }
            }
            SamplerOperation::ObservableInclude { observable_id, .. }
                if seen_observable_ids.insert(*observable_id) =>
            {
                observable_ids.push(*observable_id);
            }
            _ => {}
        }
    }
    for observable in program.compiled_observables() {
        if seen_observable_ids.insert(observable.id) {
            observable_ids.push(observable.id);
        }
    }
    (detector_ids, observable_ids)
}

impl CollectionSampler {
    pub(crate) fn detector_ids(&self) -> &[i64] {
        match self {
            Self::Dem(sampler) => sampler.detector_ids(),
            Self::Forward(sampler) => sampler.detector_ids(),
        }
    }

    pub(crate) fn observable_ids(&self) -> &[i64] {
        match self {
            Self::Dem(sampler) => sampler.observable_ids(),
            Self::Forward(sampler) => sampler.observable_ids(),
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::Dem(_) => "DEM sampler",
            Self::Forward(_) => "forward sampler",
        }
    }
}

#[derive(Clone)]
pub(crate) struct LogicalCollectionTask {
    pub(crate) task_id: String,
    pub(crate) strong_id: String,
    pub(crate) sampling_id: String,
    pub(crate) sampler: CollectionSampler,
    pub(crate) decoder: Option<Arc<dyn NativeDecoderFactory>>,
    pub(crate) decoder_name: Option<String>,
    pub(crate) metadata_json: String,
    pub(crate) options: DemLogicalCollectionOptions,
    pub(crate) postselection_mask: Option<Vec<u8>>,
    pub(crate) postselected_observables_mask: Option<Vec<u8>>,
}

impl From<DemLogicalCollectionTask> for LogicalCollectionTask {
    fn from(task: DemLogicalCollectionTask) -> Self {
        Self {
            task_id: task.task_id,
            strong_id: task.strong_id,
            sampling_id: task.sampling_id,
            sampler: CollectionSampler::Dem(task.sampler),
            decoder: task.decoder,
            decoder_name: task.decoder_name,
            metadata_json: task.metadata_json,
            options: task.options,
            postselection_mask: task.postselection_mask,
            postselected_observables_mask: task.postselected_observables_mask,
        }
    }
}

impl From<ForwardLogicalCollectionTask> for LogicalCollectionTask {
    fn from(task: ForwardLogicalCollectionTask) -> Self {
        Self {
            task_id: task.task_id,
            strong_id: task.strong_id,
            sampling_id: task.sampling_id,
            sampler: CollectionSampler::Forward(Arc::new(ForwardCollectionSampler::new(
                task.sampler,
            ))),
            decoder: task.decoder,
            decoder_name: task.decoder_name,
            metadata_json: task.metadata_json,
            options: task.options,
            postselection_mask: task.postselection_mask,
            postselected_observables_mask: task.postselected_observables_mask,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemLogicalCollectionRunOptions {
    pub num_workers: usize,
    pub seed: Option<u64>,
    pub count_observable_error_combos: bool,
    pub count_detection_events: bool,
    pub custom_error_count_key: Option<String>,
}

impl DemLogicalCollectionRunOptions {
    pub fn counter_schema(&self) -> DemLogicalCounterSchema {
        DemLogicalCounterSchema::new(
            self.count_observable_error_combos,
            self.count_detection_events,
        )
    }
}

pub fn collect_dem_logical_error_stats(
    sampler: &DemHotspotEstimator,
    options: DemLogicalCollectionOptions,
    mut decoder: Option<&mut dyn NativeDecoderWorker>,
) -> NpResult<DemLogicalCollectionStats> {
    validate_collection_options(options)?;
    if let Some(decoder) = decoder.as_deref() {
        validate_decoder_layout(
            sampler.detector_ids(),
            sampler.observable_ids(),
            "DEM sampler",
            decoder.name(),
            decoder.detector_ids(),
            decoder.observable_ids(),
            decoder.batch_formats(),
        )?;
    }

    let decoder_name = decoder.as_ref().map(|decoder| decoder.name().to_string());
    let started = Instant::now();
    let mut shots_done = 0usize;
    let mut errors = 0usize;
    let mut batch_ordinal = 0usize;
    let mut last_batch: Option<(usize, f64)> = None;
    if !collection_limits_reached(options, shots_done, errors) {
        let count_options = CountOptions::default();
        let prepared_plan = prepare_dem_count_plan(
            sampler,
            decoder
                .as_deref()
                .map(|worker| (worker.detector_ids(), worker.batch_formats())),
            &count_options,
            false,
        )?;

        while !collection_limits_reached(options, shots_done, errors) {
            let batch_shots = next_batch_size(options, shots_done, last_batch);
            let mut batch_rng = SmallRng::new(batch_seed(options.seed, 0, batch_ordinal));
            let batch_started = Instant::now();
            let batch_stats = match decoder.as_mut() {
                Some(decoder) => sample_dem_logical_error_stats_with_rng(
                    sampler,
                    batch_shots,
                    &mut batch_rng,
                    Some(&mut **decoder),
                    None,
                    &count_options,
                    &prepared_plan,
                )?,
                None => sample_dem_logical_error_stats_with_rng(
                    sampler,
                    batch_shots,
                    &mut batch_rng,
                    None,
                    None,
                    &count_options,
                    &prepared_plan,
                )?,
            };
            let elapsed = batch_started.elapsed().as_secs_f64();
            shots_done += batch_stats.shots;
            errors += batch_stats.errors;
            last_batch = Some((batch_shots, elapsed));
            batch_ordinal += 1;
        }
    }

    Ok(DemLogicalCollectionStats {
        task_id: String::new(),
        strong_id: String::new(),
        decoder: decoder_name,
        metadata_json: "null".to_string(),
        shots: shots_done,
        errors,
        discards: 0,
        seconds: started.elapsed().as_secs_f64(),
        counter_schema: DemLogicalCounterSchema::default(),
        custom_counts: HashMap::new(),
    })
}

pub fn sample_dem_logical_error_stats(
    sampler: &DemHotspotEstimator,
    shots: usize,
    seed: Option<u64>,
    decoder: Option<&mut dyn NativeDecoderWorker>,
) -> NpResult<DemLogicalCollectionStats> {
    if shots == 0 {
        return Err(NpError::new("shots must be positive"));
    }
    let started = Instant::now();
    let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
    let decoder_name = decoder.as_ref().map(|decoder| decoder.name().to_string());
    let count_options = CountOptions::default();
    let prepared_plan = prepare_dem_count_plan(
        sampler,
        decoder
            .as_deref()
            .map(|worker| (worker.detector_ids(), worker.batch_formats())),
        &count_options,
        false,
    )?;
    let batch = sample_dem_logical_error_stats_with_rng(
        sampler,
        shots,
        &mut rng,
        decoder,
        Some(started),
        &count_options,
        &prepared_plan,
    )?;
    Ok(DemLogicalCollectionStats {
        task_id: String::new(),
        strong_id: String::new(),
        decoder: decoder_name,
        metadata_json: "null".to_string(),
        shots: batch.shots,
        errors: batch.errors,
        discards: batch.discards,
        seconds: batch.seconds,
        counter_schema: DemLogicalCounterSchema::default(),
        custom_counts: batch.custom_counts,
    })
}

/// Collect logical-error statistics by repeatedly executing a forward sampler program.
pub fn collect_forward_logical_error_stats(
    program: &SamplerProgram,
    options: DemLogicalCollectionOptions,
    mut decoder: Option<&mut dyn NativeDecoderWorker>,
) -> NpResult<DemLogicalCollectionStats> {
    validate_collection_options(options)?;
    let (detector_ids, observable_ids) = forward_sampler_layout(program);
    let sampler = ForwardSamplerView::new(program, &detector_ids, &observable_ids);
    if let Some(decoder) = decoder.as_deref() {
        validate_decoder_layout(
            &detector_ids,
            &observable_ids,
            "forward sampler",
            decoder.name(),
            decoder.detector_ids(),
            decoder.observable_ids(),
            decoder.batch_formats(),
        )?;
    }

    let decoder_name = decoder.as_ref().map(|decoder| decoder.name().to_string());
    let started = Instant::now();
    let mut shots_done = 0usize;
    let mut errors = 0usize;
    let mut batch_ordinal = 0usize;
    let mut last_batch: Option<(usize, f64)> = None;
    if !collection_limits_reached(options, shots_done, errors) {
        let count_options = CountOptions::default();
        let prepared_plan = prepare_forward_count_plan(
            decoder
                .as_deref()
                .map(|worker| (worker.detector_ids(), worker.batch_formats())),
            false,
        )?;

        while !collection_limits_reached(options, shots_done, errors) {
            let batch_shots = next_batch_size(options, shots_done, last_batch);
            let batch_started = Instant::now();
            let batch_stats = match decoder.as_mut() {
                Some(decoder) => sample_forward_logical_error_stats_batch(
                    sampler,
                    batch_shots,
                    batch_seed(options.seed, 0, batch_ordinal),
                    Some(&mut **decoder),
                    None,
                    &count_options,
                    &prepared_plan,
                )?,
                None => sample_forward_logical_error_stats_batch(
                    sampler,
                    batch_shots,
                    batch_seed(options.seed, 0, batch_ordinal),
                    None,
                    None,
                    &count_options,
                    &prepared_plan,
                )?,
            };
            let elapsed = batch_started.elapsed().as_secs_f64();
            shots_done += batch_stats.shots;
            errors += batch_stats.errors;
            last_batch = Some((batch_shots, elapsed));
            batch_ordinal += 1;
        }
    }

    Ok(DemLogicalCollectionStats {
        task_id: String::new(),
        strong_id: String::new(),
        decoder: decoder_name,
        metadata_json: "null".to_string(),
        shots: shots_done,
        errors,
        discards: 0,
        seconds: started.elapsed().as_secs_f64(),
        counter_schema: DemLogicalCounterSchema::default(),
        custom_counts: HashMap::new(),
    })
}

/// Execute one seeded forward batch and return its logical-error statistics.
pub fn sample_forward_logical_error_stats(
    program: &SamplerProgram,
    shots: usize,
    seed: Option<u64>,
    mut decoder: Option<&mut dyn NativeDecoderWorker>,
) -> NpResult<DemLogicalCollectionStats> {
    if shots == 0 {
        return Err(NpError::new("shots must be positive"));
    }
    let (detector_ids, observable_ids) = forward_sampler_layout(program);
    let sampler = ForwardSamplerView::new(program, &detector_ids, &observable_ids);
    if let Some(decoder) = decoder.as_deref() {
        validate_decoder_layout(
            &detector_ids,
            &observable_ids,
            "forward sampler",
            decoder.name(),
            decoder.detector_ids(),
            decoder.observable_ids(),
            decoder.batch_formats(),
        )?;
    }
    let started = Instant::now();
    let decoder_name = decoder.as_ref().map(|decoder| decoder.name().to_string());
    let count_options = CountOptions::default();
    let prepared_plan = prepare_forward_count_plan(
        decoder
            .as_deref()
            .map(|worker| (worker.detector_ids(), worker.batch_formats())),
        false,
    )?;
    let batch = match decoder.as_mut() {
        Some(decoder) => sample_forward_logical_error_stats_batch(
            sampler,
            shots,
            seed.unwrap_or(0x95f2_04dc_4291_a715),
            Some(&mut **decoder),
            Some(started),
            &count_options,
            &prepared_plan,
        )?,
        None => sample_forward_logical_error_stats_batch(
            sampler,
            shots,
            seed.unwrap_or(0x95f2_04dc_4291_a715),
            None,
            Some(started),
            &count_options,
            &prepared_plan,
        )?,
    };
    Ok(DemLogicalCollectionStats {
        task_id: String::new(),
        strong_id: String::new(),
        decoder: decoder_name,
        metadata_json: "null".to_string(),
        shots: batch.shots,
        errors: batch.errors,
        discards: batch.discards,
        seconds: batch.seconds,
        counter_schema: DemLogicalCounterSchema::default(),
        custom_counts: batch.custom_counts,
    })
}

/// Collect detector-error-model logical-error tasks.
///
/// Returns an error before worker creation when two tasks share a `strong_id`.
pub fn collect_dem_logical_error_tasks(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
) -> NpResult<Vec<DemLogicalCollectionStats>> {
    collect_task_set(
        tasks.into_iter().map(LogicalCollectionTask::from).collect(),
        run_options,
        existing_data,
    )
}

/// Collect detector-error-model logical-error tasks and report committed deltas.
///
/// Returns an error before worker creation or progress when two tasks share a `strong_id`.
pub fn collect_dem_logical_error_tasks_with_progress<F>(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
    progress_callback: F,
) -> NpResult<Vec<DemLogicalCollectionStats>>
where
    F: FnMut(&DemLogicalCollectionStats) -> NpResult<()>,
{
    let mut progress_callback = progress_callback;
    collect_task_set_with_progress(
        tasks.into_iter().map(LogicalCollectionTask::from).collect(),
        run_options,
        existing_data,
        &mut progress_callback,
    )
}

/// Collect forward-circuit logical-error tasks.
///
/// Returns an error before worker creation when two tasks share a `strong_id`.
pub fn collect_forward_logical_error_tasks(
    tasks: Vec<ForwardLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
) -> NpResult<Vec<DemLogicalCollectionStats>> {
    collect_task_set(
        tasks.into_iter().map(LogicalCollectionTask::from).collect(),
        run_options,
        existing_data,
    )
}

/// Collect forward-circuit logical-error tasks and report committed deltas.
///
/// Returns an error before worker creation or progress when two tasks share a `strong_id`.
pub fn collect_forward_logical_error_tasks_with_progress<F>(
    tasks: Vec<ForwardLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
    progress_callback: F,
) -> NpResult<Vec<DemLogicalCollectionStats>>
where
    F: FnMut(&DemLogicalCollectionStats) -> NpResult<()>,
{
    let mut progress_callback = progress_callback;
    collect_task_set_with_progress(
        tasks.into_iter().map(LogicalCollectionTask::from).collect(),
        run_options,
        existing_data,
        &mut progress_callback,
    )
}

pub(crate) fn validate_task(task: &LogicalCollectionTask) -> NpResult<()> {
    validate_collection_options(task.options)?;
    validate_mask_shape(
        task.postselection_mask.as_deref(),
        task.sampler.detector_ids().len(),
        "postselection_mask",
    )?;
    validate_mask_shape(
        task.postselected_observables_mask.as_deref(),
        task.sampler.observable_ids().len(),
        "postselected_observables_mask",
    )?;
    if let Some(decoder) = &task.decoder {
        validate_decoder_layout(
            task.sampler.detector_ids(),
            task.sampler.observable_ids(),
            task.sampler.description(),
            decoder.name(),
            decoder.detector_ids(),
            decoder.observable_ids(),
            decoder.batch_formats(),
        )?;
    }
    if task.strong_id.is_empty() {
        return Err(NpError::new("strong_id must not be empty"));
    }
    if task.sampling_id.is_empty() {
        return Err(NpError::new("sampling_id must not be empty"));
    }
    Ok(())
}

fn validate_decoder_layout(
    sampler_detector_ids: &[i64],
    sampler_observable_ids: &[i64],
    sampler_description: &str,
    decoder_name: &str,
    detector_ids: &[i64],
    observable_ids: &[i64],
    batch_formats: &[DetectorBatchFormat],
) -> NpResult<()> {
    validate_decoder_detector_ids(detector_ids)?;
    let sampler_detector_ids = sampler_detector_ids.iter().copied().collect::<HashSet<_>>();
    for &detector_id in detector_ids {
        if !sampler_detector_ids.contains(&detector_id) {
            let message = if sampler_description == "DEM sampler" {
                format!(
                    "decoder DEM sampler requested detector id {detector_id}, but it is not declared by the DEM"
                )
            } else {
                format!(
                    "decoder {decoder_name} requested detector id {detector_id}, but it is not declared by the {sampler_description}"
                )
            };
            return Err(NpError::new(message));
        }
    }
    validate_decoder_batch_formats(batch_formats)?;
    validate_decoder_observable_layout(sampler_observable_ids, observable_ids, decoder_name)
}

fn validate_collection_options(options: DemLogicalCollectionOptions) -> NpResult<()> {
    if options.max_shots == 0 {
        return Err(NpError::new("max_shots must be positive"));
    }
    if options.min_shots > options.max_shots {
        return Err(NpError::new("min_shots must not exceed max_shots"));
    }
    if options.batch_size == 0 {
        return Err(NpError::new("batch_size must be positive"));
    }
    if options.start_batch_size.is_some_and(|value| value == 0) {
        return Err(NpError::new("start_batch_size must be positive"));
    }
    if options.max_batch_size.is_some_and(|value| value == 0) {
        return Err(NpError::new("max_batch_size must be positive"));
    }
    if options
        .max_batch_seconds
        .is_some_and(|value| value <= 0.0 || !value.is_finite())
    {
        return Err(NpError::new("max_batch_seconds must be positive"));
    }
    Ok(())
}

pub(crate) fn collection_limits_reached(
    options: DemLogicalCollectionOptions,
    shots: usize,
    error_count: usize,
) -> bool {
    shots >= options.max_shots
        || (shots >= options.min_shots
            && options.max_errors.is_some_and(|limit| error_count >= limit))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValidatedStopCounter {
    Errors,
    DetectionEvents,
    DetectorsChecked,
    ObservableCombo(String),
}

pub(crate) fn validate_stop_counter_for_tasks(
    tasks: &[LogicalCollectionTask],
    run_options: &DemLogicalCollectionRunOptions,
) -> NpResult<ValidatedStopCounter> {
    let schema = run_options.counter_schema();
    schema.validate()?;
    let Some(key) = run_options.custom_error_count_key.as_deref() else {
        return Ok(ValidatedStopCounter::Errors);
    };

    if key == DETECTION_EVENTS_KEY {
        if !schema.count_detection_events {
            return Err(NpError::new(format!(
                "custom_error_count_key {key:?} requires count_detection_events=True"
            )));
        }
        return Ok(ValidatedStopCounter::DetectionEvents);
    }
    if key == DETECTORS_CHECKED_KEY {
        if !schema.count_detection_events {
            return Err(NpError::new(format!(
                "custom_error_count_key {key:?} requires count_detection_events=True"
            )));
        }
        return Ok(ValidatedStopCounter::DetectorsChecked);
    }

    let Some(mask) = key.strip_prefix(OBSERVABLE_COMBO_PREFIX) else {
        return Err(NpError::new(format!(
            "unsupported custom_error_count_key {key:?}; use None for logical errors, {DETECTION_EVENTS_KEY:?}, {DETECTORS_CHECKED_KEY:?}, or {OBSERVABLE_COMBO_PREFIX}<mask>"
        )));
    };
    if !schema.count_observable_error_combos {
        return Err(NpError::new(format!(
            "custom_error_count_key {key:?} requires count_observable_error_combos=True"
        )));
    }
    validate_observable_combo_mask_syntax(mask)?;
    for task in tasks {
        validate_observable_combo_mask_for_task(mask, task)?;
    }
    Ok(ValidatedStopCounter::ObservableCombo(key.to_string()))
}

pub(crate) fn stop_error_count(
    stats: &DemLogicalCollectionStats,
    stop_counter: &ValidatedStopCounter,
) -> NpResult<usize> {
    stats.counter_schema.validate()?;
    match stop_counter {
        ValidatedStopCounter::Errors => Ok(stats.errors),
        ValidatedStopCounter::ObservableCombo(key) => {
            Ok(stats.custom_counts.get(key).copied().unwrap_or(0))
        }
        ValidatedStopCounter::DetectionEvents => required_stop_count(stats, DETECTION_EVENTS_KEY),
        ValidatedStopCounter::DetectorsChecked => required_stop_count(stats, DETECTORS_CHECKED_KEY),
    }
}

pub(crate) fn task_is_complete(
    stats: &DemLogicalCollectionStats,
    options: &DemLogicalCollectionOptions,
    stop_counter: &ValidatedStopCounter,
) -> NpResult<bool> {
    if stats.shots >= options.max_shots {
        return Ok(collection_limits_reached(*options, stats.shots, 0));
    }
    if stats.shots < options.min_shots || options.max_errors.is_none() {
        return Ok(collection_limits_reached(*options, stats.shots, 0));
    }
    let error_count = stop_error_count(stats, stop_counter)?;
    Ok(collection_limits_reached(
        *options,
        stats.shots,
        error_count,
    ))
}

fn required_stop_count(stats: &DemLogicalCollectionStats, key: &str) -> NpResult<usize> {
    stats.custom_counts.get(key).copied().ok_or_else(|| {
        NpError::new(format!(
            "collection stats are missing configured stop counter {key:?}"
        ))
    })
}

fn validate_observable_combo_mask_syntax(mask: &str) -> NpResult<()> {
    if mask.is_empty()
        || !mask.bytes().all(|value| matches!(value, b'E' | b'_'))
        || !mask.as_bytes().contains(&b'E')
    {
        return Err(NpError::new(format!(
            "observable combo mask {mask:?} must be non-empty, contain only 'E' and '_', and include at least one 'E'"
        )));
    }
    Ok(())
}

pub(crate) fn validate_observable_combo_mask_for_task(
    mask: &str,
    task: &LogicalCollectionTask,
) -> NpResult<()> {
    let observable_count = task.sampler.observable_ids().len();
    if mask.len() != observable_count {
        return Err(NpError::new(format!(
            "custom observable combo mask {mask:?} has width {}, but task {:?} has {observable_count} observables",
            mask.len(), task.task_id
        )));
    }
    for (index, value) in mask.bytes().enumerate() {
        if value == b'E' && packed_mask_bit(task.postselected_observables_mask.as_deref(), index) {
            return Err(NpError::new(format!(
                "custom observable combo mask {mask:?} marks postselected observable index {index} as an error for task {:?}",
                task.task_id
            )));
        }
    }
    Ok(())
}

fn packed_mask_bit(mask: Option<&[u8]>, index: usize) -> bool {
    mask.and_then(|bytes| bytes.get(index / 8))
        .is_some_and(|byte| byte & (1 << (index % 8)) != 0)
}

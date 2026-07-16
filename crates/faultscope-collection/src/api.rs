use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::counting::{
    prepare_dem_count_plan, sample_dem_logical_error_stats_with_rng,
    validate_decoder_observable_layout, validate_mask_shape, CountOptions,
};
use crate::scheduler::{
    batch_seed, collect_task_set, collect_task_set_with_progress, next_batch_size,
};
use faultscope_core::{
    DemHotspotEstimator, NativeDecoderFactory, NativeDecoderWorker, NpError, NpResult, SmallRng,
};

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
    pub custom_counts: HashMap<String, usize>,
}

impl DemLogicalCollectionStats {
    pub fn empty_for_task(task: &DemLogicalCollectionTask) -> Self {
        Self {
            task_id: task.task_id.clone(),
            strong_id: task.strong_id.clone(),
            decoder: task.decoder_name.clone(),
            metadata_json: task.metadata_json.clone(),
            shots: 0,
            errors: 0,
            discards: 0,
            seconds: 0.0,
            custom_counts: HashMap::new(),
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
        self.shots += other.shots;
        self.errors += other.errors;
        self.discards += other.discards;
        self.seconds += other.seconds;
        for (key, value) in &other.custom_counts {
            *self.custom_counts.entry(key.clone()).or_insert(0) += *value;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct DemLogicalCollectionTask {
    pub task_id: String,
    pub strong_id: String,
    pub sampler: Arc<DemHotspotEstimator>,
    pub decoder: Option<Arc<dyn NativeDecoderFactory>>,
    pub decoder_name: Option<String>,
    pub metadata_json: String,
    pub options: DemLogicalCollectionOptions,
    pub postselection_mask: Option<Vec<u8>>,
    pub postselected_observables_mask: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemLogicalCollectionRunOptions {
    pub num_workers: usize,
    pub seed: Option<u64>,
    pub count_observable_error_combos: bool,
    pub count_detection_events: bool,
    pub custom_error_count_key: Option<String>,
}

pub fn collect_dem_logical_error_stats(
    sampler: &DemHotspotEstimator,
    options: DemLogicalCollectionOptions,
    mut decoder: Option<&mut dyn NativeDecoderWorker>,
) -> NpResult<DemLogicalCollectionStats> {
    validate_collection_options(options)?;

    let decoder_name = decoder.as_ref().map(|decoder| decoder.name().to_string());
    let started = Instant::now();
    let mut shots_done = 0usize;
    let mut errors = 0usize;
    let mut batch_ordinal = 0usize;
    let mut last_batch: Option<(usize, f64)> = None;
    let count_options = CountOptions::default();
    let prepared_plan = prepare_dem_count_plan(
        sampler,
        decoder.as_deref().map(NativeDecoderWorker::detector_ids),
        &count_options,
    )?;

    while shots_done < options.max_shots
        && !(shots_done >= options.min_shots
            && options.max_errors.is_some_and(|limit| errors >= limit))
    {
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
        if shots_done >= options.min_shots
            && options.max_errors.is_some_and(|limit| errors >= limit)
        {
            break;
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
        decoder.as_deref().map(NativeDecoderWorker::detector_ids),
        &count_options,
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
        custom_counts: batch.custom_counts,
    })
}

pub fn collect_dem_logical_error_tasks(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
) -> NpResult<Vec<DemLogicalCollectionStats>> {
    collect_task_set(tasks, run_options, existing_data)
}

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
    collect_task_set_with_progress(tasks, run_options, existing_data, &mut progress_callback)
}

pub(crate) fn validate_task(task: &DemLogicalCollectionTask) -> NpResult<()> {
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
        validate_decoder_observable_layout(
            task.sampler.observable_ids(),
            decoder.observable_ids(),
            decoder.name(),
        )?;
    }
    if task.strong_id.is_empty() {
        return Err(NpError::new("strong_id must not be empty"));
    }
    Ok(())
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

pub(crate) fn stop_error_count(
    stats: &DemLogicalCollectionStats,
    custom_error_count_key: &Option<String>,
) -> usize {
    match custom_error_count_key {
        Some(key) => stats.custom_counts.get(key).copied().unwrap_or(0),
        None => stats.errors,
    }
}

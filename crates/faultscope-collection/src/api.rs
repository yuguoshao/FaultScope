use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::Arc;
use std::time::Instant;

use crate::counting::{sample_dem_logical_error_stats_with_rng, validate_mask_shape, CountOptions};
use crate::scheduler::{
    batch_seed, collect_task_set, collect_task_set_with_progress, next_batch_size,
};
use faultscope_core::{DemHotspotEstimator, NativeBatchDecoder, NpError, NpResult, SmallRng};

pub const DEM_LOGICAL_COLLECTION_CSV_HEADER: &str =
    "shots,errors,discards,seconds,decoder,strong_id,json_metadata,custom_counts";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemLogicalCollectionOptions {
    pub max_shots: usize,
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
    pub decoder: Option<Arc<dyn NativeBatchDecoder>>,
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
    decoder: Option<&dyn NativeBatchDecoder>,
) -> NpResult<DemLogicalCollectionStats> {
    validate_collection_options(options)?;

    let started = Instant::now();
    let mut shots_done = 0usize;
    let mut errors = 0usize;
    let mut batch_ordinal = 0usize;
    let mut last_batch: Option<(usize, f64)> = None;

    while shots_done < options.max_shots {
        let batch_shots = next_batch_size(options, shots_done, last_batch);
        let mut batch_rng = SmallRng::new(batch_seed(options.seed, 0, batch_ordinal));
        let batch_started = Instant::now();
        let batch_stats = sample_dem_logical_error_stats_with_rng(
            sampler,
            batch_shots,
            &mut batch_rng,
            decoder,
            None,
            &CountOptions::default(),
        )?;
        let elapsed = batch_started.elapsed().as_secs_f64();
        shots_done += batch_stats.shots;
        errors += batch_stats.errors;
        last_batch = Some((batch_shots, elapsed));
        batch_ordinal += 1;
        if options.max_errors.is_some_and(|limit| errors >= limit) {
            break;
        }
    }

    Ok(DemLogicalCollectionStats {
        task_id: String::new(),
        strong_id: String::new(),
        decoder: decoder.map(|decoder| decoder.name().to_string()),
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
    decoder: Option<&dyn NativeBatchDecoder>,
) -> NpResult<DemLogicalCollectionStats> {
    if shots == 0 {
        return Err(NpError::new("shots must be positive"));
    }
    let started = Instant::now();
    let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
    let batch = sample_dem_logical_error_stats_with_rng(
        sampler,
        shots,
        &mut rng,
        decoder,
        Some(started),
        &CountOptions::default(),
    )?;
    Ok(DemLogicalCollectionStats {
        task_id: String::new(),
        strong_id: String::new(),
        decoder: decoder.map(|decoder| decoder.name().to_string()),
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

pub fn write_dem_logical_collection_csv_header<W: Write>(writer: &mut W) -> NpResult<()> {
    writeln!(writer, "{DEM_LOGICAL_COLLECTION_CSV_HEADER}").map_err(io_error)
}

pub fn write_dem_logical_collection_csv_row<W: Write>(
    writer: &mut W,
    stats: &DemLogicalCollectionStats,
) -> NpResult<()> {
    writeln!(
        writer,
        "{},{},{},{:.12},{},{},{},{}",
        stats.shots,
        stats.errors,
        stats.discards,
        stats.seconds,
        csv_escape(stats.decoder.as_deref().unwrap_or("")),
        csv_escape(&stats.strong_id),
        csv_escape(&stats.metadata_json),
        csv_escape(&custom_counts_to_json(&stats.custom_counts)),
    )
    .map_err(io_error)
}

pub fn read_dem_logical_collection_csv<R: Read>(
    reader: R,
) -> NpResult<HashMap<String, DemLogicalCollectionStats>> {
    let mut lines = BufReader::new(reader).lines();
    let Some(header) = lines.next() else {
        return Ok(HashMap::new());
    };
    let header = header.map_err(io_error)?;
    if header.trim_end() != DEM_LOGICAL_COLLECTION_CSV_HEADER {
        return Err(NpError::new("collection CSV header does not match"));
    }

    let mut out = HashMap::<String, DemLogicalCollectionStats>::new();
    for (line_index, line) in lines.enumerate() {
        let line = line.map_err(io_error)?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(&line)?;
        if fields.len() != 8 {
            return Err(NpError::new(format!(
                "collection CSV line {} has {} fields; expected 8",
                line_index + 2,
                fields.len()
            )));
        }
        let decoder = if fields[4].is_empty() {
            None
        } else {
            Some(fields[4].clone())
        };
        let stats = DemLogicalCollectionStats {
            task_id: fields[5].clone(),
            strong_id: fields[5].clone(),
            decoder,
            metadata_json: fields[6].clone(),
            shots: parse_usize(&fields[0], "shots")?,
            errors: parse_usize(&fields[1], "errors")?,
            discards: parse_usize(&fields[2], "discards")?,
            seconds: fields[3]
                .parse::<f64>()
                .map_err(|_| NpError::new("invalid seconds value in collection CSV"))?,
            custom_counts: parse_custom_counts(&fields[7])?,
        };
        if let Some(existing) = out.get_mut(&stats.strong_id) {
            existing.add_assign_checked(&stats)?;
        } else {
            out.insert(stats.strong_id.clone(), stats);
        }
    }
    Ok(out)
}

pub(crate) fn validate_task(task: &DemLogicalCollectionTask) -> NpResult<()> {
    validate_collection_options(task.options)?;
    validate_mask_shape(
        task.postselection_mask.as_deref(),
        task.sampler.detector_ids.len(),
        "postselection_mask",
    )?;
    validate_mask_shape(
        task.postselected_observables_mask.as_deref(),
        task.sampler.observable_ids.len(),
        "postselected_observables_mask",
    )?;
    if task.strong_id.is_empty() {
        return Err(NpError::new("strong_id must not be empty"));
    }
    Ok(())
}

fn validate_collection_options(options: DemLogicalCollectionOptions) -> NpResult<()> {
    if options.max_shots == 0 {
        return Err(NpError::new("max_shots must be positive"));
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

fn csv_escape(value: &str) -> String {
    let escaped = value.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

fn parse_csv_line(line: &str) -> NpResult<Vec<String>> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                current.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(current);
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    if quoted {
        return Err(NpError::new("unterminated quoted field in collection CSV"));
    }
    fields.push(current);
    Ok(fields)
}

fn custom_counts_to_json(counts: &HashMap<String, usize>) -> String {
    let mut keys = counts.keys().collect::<Vec<_>>();
    keys.sort();
    let terms = keys
        .into_iter()
        .map(|key| {
            format!(
                "\"{}\":{}",
                key.replace('\\', "\\\\").replace('"', "\\\""),
                counts[key]
            )
        })
        .collect::<Vec<_>>();
    format!("{{{}}}", terms.join(","))
}

fn parse_custom_counts(text: &str) -> NpResult<HashMap<String, usize>> {
    let text = text.trim();
    if text.is_empty() || text == "{}" {
        return Ok(HashMap::new());
    }
    if !text.starts_with('{') || !text.ends_with('}') {
        return Err(NpError::new("invalid custom_counts JSON in collection CSV"));
    }
    let body = &text[1..text.len() - 1];
    if body.trim().is_empty() {
        return Ok(HashMap::new());
    }
    let mut out = HashMap::new();
    for term in split_json_object_terms(body)? {
        let Some((key, value)) = term.split_once(':') else {
            return Err(NpError::new(
                "invalid custom_counts entry in collection CSV",
            ));
        };
        let key = parse_json_string(key.trim())?;
        let value = parse_usize(value.trim(), "custom_counts value")?;
        out.insert(key, value);
    }
    Ok(out)
}

fn split_json_object_terms(body: &str) -> NpResult<Vec<String>> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for ch in body.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if quoted => {
                current.push(ch);
                escaped = true;
            }
            '"' => {
                quoted = !quoted;
                current.push(ch);
            }
            ',' if !quoted => {
                terms.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if quoted {
        return Err(NpError::new("unterminated custom_counts string"));
    }
    terms.push(current.trim().to_string());
    Ok(terms)
}

fn parse_json_string(text: &str) -> NpResult<String> {
    if !text.starts_with('"') || !text.ends_with('"') {
        return Err(NpError::new("custom_counts key must be a JSON string"));
    }
    let mut out = String::new();
    let mut escaped = false;
    for ch in text[1..text.len() - 1].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped {
        return Err(NpError::new("unterminated escape in custom_counts key"));
    }
    Ok(out)
}

fn parse_usize(text: &str, name: &str) -> NpResult<usize> {
    text.parse::<usize>()
        .map_err(|_| NpError::new(format!("invalid {name} value in collection CSV")))
}

fn io_error(err: std::io::Error) -> NpError {
    NpError::new(err.to_string())
}

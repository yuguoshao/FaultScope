//! Global scheduler for native DEM logical collection.
//!
//! A run owns one worker pool capped by `DemLogicalCollectionRunOptions::num_workers`.
//! Fixed-size tasks are split into deterministic batch work items; the main
//! scheduler commits finished batches by task-local ordinal, so worker completion
//! order cannot affect shots, errors, discards, custom counts, or stop points.
//! Adaptive tasks calibrate serially, freeze a batch size, and then use the same
//! batch-granular worker pool. Batch seeds are derived from the run/task seed,
//! seed stream, and ordinal, not from the worker that executes the work.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::api::{
    stop_error_count, validate_task, DemLogicalCollectionOptions, DemLogicalCollectionRunOptions,
    DemLogicalCollectionStats, DemLogicalCollectionTask,
};
use crate::counting::{sample_dem_logical_error_stats_with_rng, BatchStats, CountOptions};
use faultscope_core::{NpError, NpResult, SmallRng};

#[derive(Clone)]
struct BatchWork {
    state_index: usize,
    ordinal: usize,
    shots: usize,
    seed: Option<u64>,
    seed_stream: usize,
    task: Arc<DemLogicalCollectionTask>,
    run_options: Arc<DemLogicalCollectionRunOptions>,
}

#[derive(Clone)]
struct AdaptiveCalibrationWork {
    state_index: usize,
    task: Arc<DemLogicalCollectionTask>,
    run_options: Arc<DemLogicalCollectionRunOptions>,
    seed_stream: usize,
}

struct AdaptiveCalibrationResult {
    stats: DemLogicalCollectionStats,
    frozen_batch_size: usize,
}

enum Work {
    Batch(BatchWork),
    AdaptiveCalibration(AdaptiveCalibrationWork),
    Shutdown,
}

enum WorkResult {
    Batch {
        state_index: usize,
        ordinal: usize,
        stats: BatchStats,
    },
    AdaptiveDelta {
        stats: DemLogicalCollectionStats,
        ack: mpsc::Sender<NpResult<()>>,
    },
    AdaptiveCalibrated {
        state_index: usize,
        result: AdaptiveCalibrationResult,
    },
}

impl WorkResult {
    fn completes_work(&self) -> bool {
        !matches!(self, WorkResult::AdaptiveDelta { .. })
    }
}

enum TaskPhase {
    Calibrating { scheduled: bool },
    Parallel { seed_stream: usize },
    Complete,
}

struct TaskState {
    output_index: usize,
    task: Arc<DemLogicalCollectionTask>,
    stats: DemLogicalCollectionStats,
    target_shots: usize,
    stop_error_limit: Option<usize>,
    specs: Vec<usize>,
    next_scheduled: usize,
    next_committed: usize,
    pending: HashMap<usize, BatchStats>,
    in_flight: usize,
    phase: TaskPhase,
    seed: Option<u64>,
    resume_shots: usize,
    started: Option<Instant>,
    committed_elapsed: Duration,
}

impl TaskState {
    fn potential_capacity(&self, worker_limit: usize) -> usize {
        match self.phase {
            TaskPhase::Complete => 0,
            TaskPhase::Calibrating { .. } => worker_limit,
            TaskPhase::Parallel { .. } => self.specs.len().saturating_sub(self.next_scheduled),
        }
    }

    fn can_schedule(&self) -> bool {
        match self.phase {
            TaskPhase::Calibrating { scheduled } => !scheduled,
            TaskPhase::Parallel { .. } => self.next_scheduled < self.specs.len(),
            TaskPhase::Complete => false,
        }
    }

    fn is_complete(&self) -> bool {
        matches!(self.phase, TaskPhase::Complete)
    }
}

pub(crate) fn collect_task_set(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
) -> NpResult<Vec<DemLogicalCollectionStats>> {
    collect_task_set_inner(tasks, run_options, existing_data, None)
}

pub(crate) type ProgressCallback<'a> = dyn FnMut(&DemLogicalCollectionStats) -> NpResult<()> + 'a;

pub(crate) fn collect_task_set_with_progress(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
    progress_callback: &mut ProgressCallback<'_>,
) -> NpResult<Vec<DemLogicalCollectionStats>> {
    collect_task_set_inner(tasks, run_options, existing_data, Some(progress_callback))
}

fn collect_task_set_inner(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
    existing_data: HashMap<String, DemLogicalCollectionStats>,
    mut progress_callback: Option<&mut ProgressCallback<'_>>,
) -> NpResult<Vec<DemLogicalCollectionStats>> {
    if run_options.num_workers == 0 {
        return Err(NpError::new("num_workers must be positive"));
    }
    for task in &tasks {
        validate_task(task)?;
    }

    let mut results = vec![None; tasks.len()];
    let mut states = Vec::new();
    for (output_index, task) in tasks.into_iter().enumerate() {
        let existing = existing_data.get(&task.strong_id).cloned();
        if let Some(existing_stats) = &existing {
            validate_existing_stats_for_task(existing_stats, &task)?;
            if task_is_complete(
                existing_stats,
                &task.options,
                &run_options.custom_error_count_key,
            ) {
                results[output_index] = Some(existing_stats.clone());
                continue;
            }
        }

        let stats = existing.unwrap_or_else(|| DemLogicalCollectionStats::empty_for_task(&task));
        let state = make_task_state(output_index, task, stats, &run_options)?;
        if state.is_complete() {
            results[output_index] = Some(state.stats);
        } else {
            states.push(state);
        }
    }

    if states.is_empty() {
        return collect_results(results);
    }

    let runnable_capacity = states.iter().fold(0usize, |capacity, state| {
        capacity.saturating_add(state.potential_capacity(run_options.num_workers))
    });
    let worker_count = run_options.num_workers.min(runnable_capacity).max(1);
    let run_options = Arc::new(run_options);
    let (work_tx, work_rx) = mpsc::channel::<Work>();
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (result_tx, result_rx) = mpsc::channel::<NpResult<WorkResult>>();
    let mut handles = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        let work_rx = work_rx.clone();
        let result_tx = result_tx.clone();
        handles.push(thread::spawn(move || loop {
            let work = {
                let rx = work_rx.lock().unwrap();
                rx.recv()
            };
            let Ok(work) = work else {
                break;
            };
            match work {
                Work::Shutdown => break,
                Work::Batch(work) => {
                    if result_tx.send(run_batch_work(work)).is_err() {
                        break;
                    }
                }
                Work::AdaptiveCalibration(work) => {
                    if result_tx
                        .send(run_adaptive_calibration_work(work, &result_tx))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }));
    }
    drop(result_tx);

    let mut in_flight = 0usize;
    let mut next_state_to_schedule = 0usize;
    let mut first_error: Option<NpError> = None;
    schedule_available_work(
        &mut states,
        &mut next_state_to_schedule,
        worker_count,
        &work_tx,
        &run_options,
        &mut in_flight,
    )?;

    while in_flight > 0 {
        let result = result_rx
            .recv()
            .map_err(|_| NpError::new("collection worker result channel closed"))?;
        let completes_work = match &result {
            Ok(result) => result.completes_work(),
            Err(_) => true,
        };
        if completes_work {
            in_flight -= 1;
        }
        if first_error.is_some() {
            continue;
        }
        match result {
            Ok(result) => {
                if let Err(err) = handle_work_result(
                    result,
                    &mut states,
                    &run_options,
                    &mut results,
                    &mut progress_callback,
                ) {
                    first_error = Some(err);
                    continue;
                }
                if let Err(err) = schedule_available_work(
                    &mut states,
                    &mut next_state_to_schedule,
                    worker_count,
                    &work_tx,
                    &run_options,
                    &mut in_flight,
                ) {
                    first_error = Some(err);
                }
            }
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }

    for _ in 0..worker_count {
        let _ = work_tx.send(Work::Shutdown);
    }
    drop(work_tx);
    for handle in handles {
        if handle.join().is_err() && first_error.is_none() {
            first_error = Some(NpError::new("collection worker thread panicked"));
        }
    }
    if let Some(err) = first_error {
        return Err(err);
    }

    collect_results(results)
}

fn make_task_state(
    output_index: usize,
    task: DemLogicalCollectionTask,
    stats: DemLogicalCollectionStats,
    run_options: &DemLogicalCollectionRunOptions,
) -> NpResult<TaskState> {
    let target_shots = task.options.max_shots;
    let stop_error_limit = task.options.max_errors;
    let resume_shots = stats.shots;
    let remaining_shots = target_shots.saturating_sub(stats.shots);
    if remaining_shots == 0 {
        return Ok(TaskState {
            output_index,
            task: Arc::new(task),
            stats,
            target_shots,
            stop_error_limit,
            specs: Vec::new(),
            next_scheduled: 0,
            next_committed: 0,
            pending: HashMap::new(),
            in_flight: 0,
            phase: TaskPhase::Complete,
            seed: None,
            resume_shots,
            started: None,
            committed_elapsed: Duration::ZERO,
        });
    }

    let adjusted = adjusted_task_for_remaining(task, &stats, run_options)?;
    let adaptive = adjusted.options.max_batch_seconds.is_some();
    let specs = if adaptive {
        Vec::new()
    } else {
        fixed_batch_specs(adjusted.options)
    };
    let phase = if adaptive {
        TaskPhase::Calibrating { scheduled: false }
    } else {
        TaskPhase::Parallel {
            seed_stream: task_seed_stream(resume_shots, 0),
        }
    };
    let seed = task_run_seed(&adjusted, run_options);
    Ok(TaskState {
        output_index,
        task: Arc::new(adjusted),
        stats,
        target_shots,
        stop_error_limit,
        specs,
        next_scheduled: 0,
        next_committed: 0,
        pending: HashMap::new(),
        in_flight: 0,
        phase,
        seed,
        resume_shots,
        started: None,
        committed_elapsed: Duration::ZERO,
    })
}

fn adjusted_task_for_remaining(
    mut task: DemLogicalCollectionTask,
    stats: &DemLogicalCollectionStats,
    run_options: &DemLogicalCollectionRunOptions,
) -> NpResult<DemLogicalCollectionTask> {
    let remaining_shots = task.options.max_shots.saturating_sub(stats.shots);
    task.options.max_shots = remaining_shots;
    if let Some(max_errors) = task.options.max_errors {
        let current = stop_error_count(stats, &run_options.custom_error_count_key);
        if current >= max_errors {
            task.options.max_errors = Some(0);
        } else {
            task.options.max_errors = Some(max_errors - current);
        }
    }
    validate_task(&task)?;
    Ok(task)
}

fn schedule_available_work(
    states: &mut [TaskState],
    next_state_to_schedule: &mut usize,
    worker_count: usize,
    work_tx: &mpsc::Sender<Work>,
    run_options: &Arc<DemLogicalCollectionRunOptions>,
    in_flight: &mut usize,
) -> NpResult<()> {
    while *in_flight < worker_count && states.iter().any(TaskState::can_schedule) {
        let Some(state_index) = next_schedulable_state(states, next_state_to_schedule) else {
            break;
        };
        let work = next_work_for_state(state_index, &mut states[state_index], run_options)?;
        states[state_index].in_flight += 1;
        *in_flight += 1;
        work_tx
            .send(work)
            .map_err(|_| NpError::new("collection worker work channel closed"))?;
    }
    Ok(())
}

fn next_schedulable_state(
    states: &[TaskState],
    next_state_to_schedule: &mut usize,
) -> Option<usize> {
    if states.is_empty() {
        return None;
    }
    for _ in 0..states.len() {
        let index = *next_state_to_schedule % states.len();
        *next_state_to_schedule = (*next_state_to_schedule + 1) % states.len();
        if states[index].can_schedule() {
            return Some(index);
        }
    }
    None
}

fn next_work_for_state(
    state_index: usize,
    state: &mut TaskState,
    run_options: &Arc<DemLogicalCollectionRunOptions>,
) -> NpResult<Work> {
    if let TaskPhase::Calibrating { scheduled } = &mut state.phase {
        *scheduled = true;
        return Ok(Work::AdaptiveCalibration(AdaptiveCalibrationWork {
            state_index,
            task: state.task.clone(),
            run_options: run_options.clone(),
            seed_stream: task_seed_stream(state.resume_shots, 0),
        }));
    }

    let seed_stream = match state.phase {
        TaskPhase::Parallel { seed_stream } => seed_stream,
        TaskPhase::Complete => return Err(NpError::new("cannot schedule completed task")),
        TaskPhase::Calibrating { .. } => {
            return Err(NpError::new("adaptive calibration is already scheduled"));
        }
    };

    if state.started.is_none() {
        state.started = Some(Instant::now());
    }
    let ordinal = state.next_scheduled;
    let shots = *state
        .specs
        .get(ordinal)
        .ok_or_else(|| NpError::new("missing fixed-batch work spec"))?;
    state.next_scheduled += 1;
    Ok(Work::Batch(BatchWork {
        state_index,
        ordinal,
        shots,
        seed: state.seed,
        seed_stream,
        task: state.task.clone(),
        run_options: run_options.clone(),
    }))
}

fn handle_work_result(
    result: WorkResult,
    states: &mut [TaskState],
    run_options: &DemLogicalCollectionRunOptions,
    results: &mut [Option<DemLogicalCollectionStats>],
    progress_callback: &mut Option<&mut ProgressCallback<'_>>,
) -> NpResult<()> {
    match result {
        WorkResult::Batch {
            state_index,
            ordinal,
            stats,
        } => {
            let state = states
                .get_mut(state_index)
                .ok_or_else(|| NpError::new("collection returned invalid task index"))?;
            state.in_flight = state.in_flight.saturating_sub(1);
            if state.is_complete() {
                return Ok(());
            }
            state.pending.insert(ordinal, stats);
            commit_ready_batches(state, run_options, results, progress_callback)
        }
        WorkResult::AdaptiveDelta { stats, ack } => {
            let result = emit_progress(progress_callback, &stats);
            let _ = ack.send(result.clone());
            result
        }
        WorkResult::AdaptiveCalibrated {
            state_index,
            result,
        } => {
            let state = states
                .get_mut(state_index)
                .ok_or_else(|| NpError::new("collection returned invalid task index"))?;
            state.in_flight = state.in_flight.saturating_sub(1);
            if state.is_complete() {
                return Ok(());
            }
            state.stats.add_assign_checked(&result.stats)?;
            if reached_task_limit(state, run_options) {
                return mark_state_complete(state, results);
            }

            let remaining_shots = state.target_shots.saturating_sub(state.stats.shots);
            state.specs = fixed_batch_specs_for_size(remaining_shots, result.frozen_batch_size);
            state.next_scheduled = 0;
            state.next_committed = 0;
            state.pending.clear();
            state.phase = TaskPhase::Parallel {
                seed_stream: task_seed_stream(state.resume_shots, 1),
            };
            state.started = None;
            state.committed_elapsed = Duration::ZERO;
            if state.specs.is_empty() {
                mark_state_complete(state, results)?;
            }
            Ok(())
        }
    }
}

fn commit_ready_batches(
    state: &mut TaskState,
    run_options: &DemLogicalCollectionRunOptions,
    results: &mut [Option<DemLogicalCollectionStats>],
    progress_callback: &mut Option<&mut ProgressCallback<'_>>,
) -> NpResult<()> {
    while let Some(batch_stats) = state.pending.remove(&state.next_committed) {
        let elapsed = state
            .started
            .map(|started| started.elapsed())
            .unwrap_or(Duration::ZERO);
        let delta_seconds = elapsed
            .checked_sub(state.committed_elapsed)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64();
        state.committed_elapsed = elapsed;
        let delta = stats_delta_from_batch(state.task.as_ref(), batch_stats, delta_seconds);
        state.stats.add_assign_checked(&delta)?;
        state.next_committed += 1;
        emit_progress(progress_callback, &delta)?;

        if reached_task_limit(state, run_options) {
            return mark_state_complete(state, results);
        }
    }

    if state.next_committed >= state.specs.len() && state.in_flight == 0 {
        mark_state_complete(state, results)?;
    }
    Ok(())
}

fn reached_task_limit(state: &TaskState, run_options: &DemLogicalCollectionRunOptions) -> bool {
    state.stats.shots >= state.target_shots
        || state.stop_error_limit.is_some_and(|limit| {
            stop_error_count(&state.stats, &run_options.custom_error_count_key) >= limit
        })
}

fn mark_state_complete(
    state: &mut TaskState,
    results: &mut [Option<DemLogicalCollectionStats>],
) -> NpResult<()> {
    state.phase = TaskPhase::Complete;
    let slot = results
        .get_mut(state.output_index)
        .ok_or_else(|| NpError::new("collection returned invalid output index"))?;
    if slot.is_none() {
        *slot = Some(state.stats.clone());
    }
    Ok(())
}

fn run_batch_work(work: BatchWork) -> NpResult<WorkResult> {
    let mut rng = SmallRng::new(batch_seed(work.seed, work.seed_stream, work.ordinal));
    let count_options = CountOptions {
        postselection_mask: work.task.postselection_mask.as_deref(),
        postselected_observables_mask: work.task.postselected_observables_mask.as_deref(),
        count_observable_error_combos: work.run_options.count_observable_error_combos,
        count_detection_events: work.run_options.count_detection_events,
    };
    let stats = sample_dem_logical_error_stats_with_rng(
        &work.task.sampler,
        work.shots,
        &mut rng,
        work.task.decoder.as_deref(),
        None,
        &count_options,
    )?;
    Ok(WorkResult::Batch {
        state_index: work.state_index,
        ordinal: work.ordinal,
        stats,
    })
}

fn run_adaptive_calibration_work(
    work: AdaptiveCalibrationWork,
    result_tx: &mpsc::Sender<NpResult<WorkResult>>,
) -> NpResult<WorkResult> {
    let result = calibrate_adaptive_task(
        work.task.as_ref(),
        &work.run_options,
        work.seed_stream,
        Some(work.state_index),
        Some(result_tx),
    )?;
    Ok(WorkResult::AdaptiveCalibrated {
        state_index: work.state_index,
        result,
    })
}

fn calibrate_adaptive_task(
    task: &DemLogicalCollectionTask,
    run_options: &DemLogicalCollectionRunOptions,
    seed_stream: usize,
    state_index: Option<usize>,
    delta_tx: Option<&mpsc::Sender<NpResult<WorkResult>>>,
) -> NpResult<AdaptiveCalibrationResult> {
    validate_task(task)?;
    let mut stats = DemLogicalCollectionStats::empty_for_task(task);
    let mut shots_done = 0usize;
    let mut batch_ordinal = 0usize;
    let mut observations = Vec::with_capacity(3);
    let seed = task_run_seed(task, run_options);
    let cap = task
        .options
        .max_batch_size
        .unwrap_or(task.options.batch_size);
    let mut batch_shots = task
        .options
        .start_batch_size
        .unwrap_or(task.options.batch_size)
        .min(cap)
        .min(task.options.max_shots)
        .max(1);
    let mut frozen_batch_size = batch_shots;
    let count_options = CountOptions {
        postselection_mask: task.postselection_mask.as_deref(),
        postselected_observables_mask: task.postselected_observables_mask.as_deref(),
        count_observable_error_combos: run_options.count_observable_error_combos,
        count_detection_events: run_options.count_detection_events,
    };

    while shots_done < task.options.max_shots && batch_ordinal < 3 {
        batch_shots = batch_shots.min(task.options.max_shots - shots_done).max(1);
        let mut rng = SmallRng::new(batch_seed(seed, seed_stream, batch_ordinal));
        let batch_started = Instant::now();
        let batch_stats = sample_dem_logical_error_stats_with_rng(
            &task.sampler,
            batch_shots,
            &mut rng,
            task.decoder.as_deref(),
            Some(batch_started),
            &count_options,
        )?;
        let elapsed = batch_stats.seconds;
        let delta = stats_delta_from_batch(task, batch_stats, elapsed);
        stats.add_assign_checked(&delta)?;
        if let (Some(_state_index), Some(delta_tx)) = (state_index, delta_tx) {
            let (ack_tx, ack_rx) = mpsc::channel();
            delta_tx
                .send(Ok(WorkResult::AdaptiveDelta {
                    stats: delta,
                    ack: ack_tx,
                }))
                .map_err(|_| NpError::new("collection worker result channel closed"))?;
            ack_rx
                .recv()
                .map_err(|_| NpError::new("adaptive collection progress cancelled"))??;
        }
        shots_done += batch_shots;
        observations.push((batch_shots, elapsed));
        batch_ordinal += 1;

        if task.options.max_errors.is_some_and(|limit| {
            stop_error_count(&stats, &run_options.custom_error_count_key) >= limit
        }) {
            break;
        }
        if shots_done >= task.options.max_shots {
            break;
        }

        let target = calibration_target_batch_size(task.options, &observations);
        frozen_batch_size = target;
        if should_finish_calibration(batch_ordinal, batch_shots, target) {
            break;
        }
        batch_shots = target;
    }
    Ok(AdaptiveCalibrationResult {
        stats,
        frozen_batch_size,
    })
}

fn stats_delta_from_batch(
    task: &DemLogicalCollectionTask,
    batch_stats: BatchStats,
    seconds: f64,
) -> DemLogicalCollectionStats {
    DemLogicalCollectionStats {
        task_id: task.task_id.clone(),
        strong_id: task.strong_id.clone(),
        decoder: task.decoder_name.clone(),
        metadata_json: task.metadata_json.clone(),
        shots: batch_stats.shots,
        errors: batch_stats.errors,
        discards: batch_stats.discards,
        seconds,
        custom_counts: batch_stats.custom_counts,
    }
}

fn emit_progress(
    progress_callback: &mut Option<&mut ProgressCallback<'_>>,
    delta: &DemLogicalCollectionStats,
) -> NpResult<()> {
    if let Some(callback) = progress_callback.as_deref_mut() {
        callback(delta)?;
    }
    Ok(())
}

pub(crate) fn next_batch_size(
    options: DemLogicalCollectionOptions,
    shots_done: usize,
    last_batch: Option<(usize, f64)>,
) -> usize {
    let remaining = options.max_shots - shots_done;
    let batch = if shots_done == 0 {
        options.start_batch_size.unwrap_or(options.batch_size)
    } else if let (Some(max_seconds), Some((last_shots, last_seconds))) =
        (options.max_batch_seconds, last_batch)
    {
        let estimated = if last_seconds > 0.0 {
            ((last_shots as f64) * max_seconds / last_seconds).floor() as usize
        } else {
            options.batch_size
        };
        estimated.max(1)
    } else {
        options.batch_size
    };
    let cap = options.max_batch_size.unwrap_or(options.batch_size);
    remaining.min(batch.min(cap).max(1))
}

fn calibration_target_batch_size(
    options: DemLogicalCollectionOptions,
    observations: &[(usize, f64)],
) -> usize {
    let mut throughputs = observations
        .iter()
        .filter_map(|(shots, seconds)| {
            if *shots > 0 && seconds.is_finite() && *seconds > 0.0 {
                Some((*shots as f64) / *seconds)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    throughputs.sort_by(f64::total_cmp);
    let estimated = if throughputs.is_empty() {
        options.batch_size
    } else {
        let midpoint = throughputs.len() / 2;
        let median = if throughputs.len() % 2 == 0 {
            0.5 * (throughputs[midpoint - 1] + throughputs[midpoint])
        } else {
            throughputs[midpoint]
        };
        let target_seconds = options.max_batch_seconds.unwrap_or(0.0);
        (median * target_seconds).floor().max(1.0) as usize
    };
    estimated
        .min(options.max_batch_size.unwrap_or(options.batch_size))
        .max(1)
}

fn should_finish_calibration(batch_count: usize, current: usize, target: usize) -> bool {
    if batch_count >= 3 {
        return true;
    }
    batch_count >= 2 && (current.abs_diff(target) as u128) * 5 <= current as u128
}

pub(crate) fn batch_seed(seed: Option<u64>, task_index: usize, batch_ordinal: usize) -> u64 {
    let base = collection_seed(seed);
    mix_seed(
        mix_seed(base, task_index as u64),
        batch_ordinal as u64 ^ 0x517c_c1b7_2722_0a95,
    )
}

fn task_seed_stream(resume_shots: usize, phase: usize) -> usize {
    if resume_shots == 0 {
        return phase;
    }
    (mix_seed(resume_shots as u64, 0x7a6d_4f21_c953_8b17) as usize).wrapping_shl(2)
        | 2
        | (phase & 1)
}

pub(crate) fn task_run_seed(
    task: &DemLogicalCollectionTask,
    run_options: &DemLogicalCollectionRunOptions,
) -> Option<u64> {
    task.options.seed.or_else(|| {
        run_options
            .seed
            .map(|seed| mix_seed(seed, stable_string_hash(&task.strong_id)))
    })
}

fn fixed_batch_specs(options: DemLogicalCollectionOptions) -> Vec<usize> {
    let mut specs = Vec::new();
    let mut shots_done = 0usize;
    while shots_done < options.max_shots {
        let batch_shots = next_batch_size(options, shots_done, None);
        specs.push(batch_shots);
        shots_done += batch_shots;
    }
    specs
}

fn fixed_batch_specs_for_size(shots: usize, batch_size: usize) -> Vec<usize> {
    let mut specs = Vec::new();
    let mut shots_done = 0usize;
    while shots_done < shots {
        let batch_shots = (shots - shots_done).min(batch_size).max(1);
        specs.push(batch_shots);
        shots_done += batch_shots;
    }
    specs
}

fn validate_existing_stats_for_task(
    stats: &DemLogicalCollectionStats,
    task: &DemLogicalCollectionTask,
) -> NpResult<()> {
    if stats.decoder != task.decoder_name || stats.metadata_json != task.metadata_json {
        return Err(NpError::new(
            "existing stats strong_id matched but decoder or metadata differs",
        ));
    }
    Ok(())
}

fn task_is_complete(
    stats: &DemLogicalCollectionStats,
    options: &DemLogicalCollectionOptions,
    custom_error_count_key: &Option<String>,
) -> bool {
    stats.shots >= options.max_shots
        || options
            .max_errors
            .is_some_and(|limit| stop_error_count(stats, custom_error_count_key) >= limit)
}

fn collect_results(
    results: Vec<Option<DemLogicalCollectionStats>>,
) -> NpResult<Vec<DemLogicalCollectionStats>> {
    results
        .into_iter()
        .map(|item| item.ok_or_else(|| NpError::new("missing collection result")))
        .collect()
}

fn collection_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0x95f2_04dc_4291_a715)
    })
}

fn mix_seed(left: u64, right: u64) -> u64 {
    let mut value = left ^ right.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn stable_string_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |acc, byte| {
            (acc ^ (*byte as u64)).wrapping_mul(0x1000_0000_01b3)
        })
}

#[cfg(test)]
mod tests {
    use super::{
        batch_seed, calibration_target_batch_size, fixed_batch_specs, should_finish_calibration,
        task_seed_stream, DemLogicalCollectionOptions,
    };

    fn options() -> DemLogicalCollectionOptions {
        DemLogicalCollectionOptions {
            max_shots: 1_000,
            max_errors: None,
            batch_size: 40,
            seed: Some(1),
            start_batch_size: Some(10),
            max_batch_size: Some(30),
            max_batch_seconds: Some(0.1),
        }
    }

    #[test]
    fn calibration_target_uses_median_positive_throughput_and_cap() {
        assert_eq!(
            calibration_target_batch_size(options(), &[(10, 0.1), (20, 0.1), (30, 0.1)]),
            20
        );
        assert_eq!(
            calibration_target_batch_size(options(), &[(100, 0.1), (200, 0.1)]),
            30
        );
    }

    #[test]
    fn calibration_target_falls_back_when_elapsed_is_not_positive() {
        assert_eq!(
            calibration_target_batch_size(options(), &[(10, 0.0), (20, -1.0)]),
            30
        );
    }

    #[test]
    fn calibration_requires_two_batches_and_stops_by_three() {
        assert!(!should_finish_calibration(1, 100, 100));
        assert!(should_finish_calibration(2, 100, 120));
        assert!(!should_finish_calibration(2, 100, 121));
        assert!(should_finish_calibration(3, 100, 200));
    }

    #[test]
    fn fixed_specs_preserve_start_batch_size_for_only_the_first_batch() {
        let mut options = options();
        options.max_shots = 100;
        options.max_batch_size = None;
        options.max_batch_seconds = None;
        assert_eq!(fixed_batch_specs(options), vec![10, 40, 40, 10]);
    }

    #[test]
    fn resumed_tasks_use_distinct_seed_streams_for_each_phase() {
        assert_eq!(task_seed_stream(0, 0), 0);
        assert_eq!(task_seed_stream(0, 1), 1);
        assert_ne!(task_seed_stream(5, 0), 0);
        assert_ne!(task_seed_stream(5, 1), 1);
        assert_ne!(task_seed_stream(5, 0), task_seed_stream(5, 1));
        assert_ne!(
            batch_seed(Some(7), task_seed_stream(0, 0), 0),
            batch_seed(Some(7), task_seed_stream(5, 0), 0)
        );
    }
}

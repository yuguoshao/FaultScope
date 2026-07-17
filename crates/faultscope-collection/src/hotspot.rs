use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use faultscope_core::{NativeDecoderWorker, NpError, NpResult, SmallRng};

use crate::api::{
    stop_error_count, validate_stop_counter_for_tasks, validate_task,
    DemLogicalCollectionRunOptions, DemLogicalCollectionStats, DemLogicalCollectionTask,
    ValidatedStopCounter,
};
use crate::counting::{count_detailed_batch, CountOptions};
use crate::scheduler::{batch_seed, next_batch_size, task_run_seed};
use crate::worker_decoder::WorkerDecoderCache;
use crate::worker_executor::WorkerExecutor;

#[derive(Debug, Clone, PartialEq)]
pub struct DemHotspotCollectionResult {
    pub stats: DemLogicalCollectionStats,
    pub batch_stats: Vec<DemLogicalCollectionStats>,
    pub edge_sensitivities: Vec<f64>,
}

#[derive(Clone)]
struct HotspotWork {
    task_index: usize,
    ordinal: usize,
    shots: usize,
    seed: Option<u64>,
    task: Arc<DemLogicalCollectionTask>,
    run_options: Arc<DemLogicalCollectionRunOptions>,
}

struct HotspotBatchResult {
    task_index: usize,
    ordinal: usize,
    stats: DemLogicalCollectionStats,
    edge_sensitivities: Vec<f64>,
}

struct HotspotCommitState {
    stats: DemLogicalCollectionStats,
    batch_stats: Vec<DemLogicalCollectionStats>,
    sensitivity_sums: Vec<f64>,
    pending: HashMap<usize, HotspotBatchResult>,
    next_ordinal: usize,
    max_shots: usize,
    min_shots: usize,
    max_errors: Option<usize>,
    complete: bool,
    started: Option<Instant>,
    completed_seconds: Option<f64>,
}

struct HotspotTaskCursor {
    task: Arc<DemLogicalCollectionTask>,
    seed: Option<u64>,
    shots_scheduled: usize,
    next_ordinal: usize,
}

struct HotspotWorkQueue {
    tasks: Vec<HotspotTaskCursor>,
    next_task: usize,
    total_batches: usize,
    run_options: Arc<DemLogicalCollectionRunOptions>,
}

impl HotspotWorkQueue {
    fn new(
        tasks: Vec<DemLogicalCollectionTask>,
        run_options: Arc<DemLogicalCollectionRunOptions>,
    ) -> Self {
        let total_batches = tasks.iter().fold(0usize, |total, task| {
            total.saturating_add(fixed_hotspot_batch_count(task.options))
        });
        let tasks = tasks
            .into_iter()
            .map(|task| {
                let seed = task_run_seed(&task, &run_options);
                HotspotTaskCursor {
                    task: Arc::new(task),
                    seed,
                    shots_scheduled: 0,
                    next_ordinal: 0,
                }
            })
            .collect();
        Self {
            tasks,
            next_task: 0,
            total_batches,
            run_options,
        }
    }

    fn next_work(
        &mut self,
        states: &[HotspotCommitState],
        worker_count: usize,
    ) -> Option<HotspotWork> {
        for _ in 0..self.tasks.len() {
            let task_index = self.next_task;
            let cursor = &mut self.tasks[task_index];
            if states[task_index].complete
                || cursor.shots_scheduled >= cursor.task.options.max_shots
            {
                self.next_task = (self.next_task + 1) % self.tasks.len();
                continue;
            }
            if cursor.next_ordinal >= states[task_index].next_ordinal + worker_count {
                return None;
            }

            let shots = next_batch_size(cursor.task.options, cursor.shots_scheduled, None);
            let work = HotspotWork {
                task_index,
                ordinal: cursor.next_ordinal,
                shots,
                seed: cursor.seed,
                task: cursor.task.clone(),
                run_options: self.run_options.clone(),
            };
            cursor.shots_scheduled += shots;
            cursor.next_ordinal += 1;
            self.next_task = (self.next_task + 1) % self.tasks.len();
            return Some(work);
        }
        None
    }
}

pub fn collect_dem_hotspot_tasks(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
) -> NpResult<Vec<DemHotspotCollectionResult>> {
    if run_options.num_workers == 0 {
        return Err(NpError::new("num_workers must be positive"));
    }
    let counter_schema = run_options.counter_schema();
    counter_schema.validate()?;
    for task in &tasks {
        validate_task(task)?;
        if task.options.max_batch_seconds.is_some() {
            return Err(NpError::new(
                "hotspot collection does not support adaptive batch sizing",
            ));
        }
    }
    let stop_counter = validate_stop_counter_for_tasks(&tasks, &run_options)?;
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let run_options = Arc::new(run_options);
    let mut states = tasks
        .iter()
        .map(|task| HotspotCommitState {
            stats: DemLogicalCollectionStats::empty_for_task(task, counter_schema),
            batch_stats: Vec::new(),
            sensitivity_sums: vec![0.0; task.sampler.edge_count()],
            pending: HashMap::new(),
            next_ordinal: 0,
            max_shots: task.options.max_shots,
            min_shots: task.options.min_shots,
            max_errors: task.options.max_errors,
            complete: task.options.min_shots == 0 && task.options.max_errors == Some(0),
            started: None,
            completed_seconds: None,
        })
        .collect::<Vec<_>>();
    if states.iter().all(|state| state.complete) {
        return states.into_iter().map(finish_hotspot_state).collect();
    }
    let mut work_queue = HotspotWorkQueue::new(tasks, run_options.clone());
    let worker_count = run_options.num_workers.min(work_queue.total_batches).max(1);
    let executor = WorkerExecutor::new(
        worker_count,
        WorkerDecoderCache::new,
        |work, decoder_cache, _| {
            decoder_cache
                .resolve(work.task_index, work.task.decoder.as_ref())
                .and_then(|decoder| run_hotspot_batch(work, decoder))
        },
        hotspot_worker_context,
        hotspot_worker_context,
    );
    let mut in_flight = 0usize;
    schedule_hotspot_work(
        &mut work_queue,
        &mut states,
        worker_count,
        executor.sender(),
        &mut in_flight,
    )?;
    let mut first_error = None;
    executor.drain(
        &mut in_flight,
        &mut first_error,
        "hotspot collection worker result channel closed",
        |_| true,
        |result, work_tx, in_flight| {
            commit_hotspot_result(result, &mut states, &stop_counter)?;
            schedule_hotspot_work(
                &mut work_queue,
                &mut states,
                worker_count,
                work_tx,
                in_flight,
            )
        },
        |_| {},
    );
    executor.finish(first_error, "hotspot collection worker thread panicked")?;

    states.into_iter().map(finish_hotspot_state).collect()
}

fn schedule_hotspot_work(
    work_queue: &mut HotspotWorkQueue,
    states: &mut [HotspotCommitState],
    worker_count: usize,
    work_tx: &mpsc::Sender<HotspotWork>,
    in_flight: &mut usize,
) -> NpResult<()> {
    while *in_flight < worker_count {
        let Some(work) = work_queue.next_work(states, worker_count) else {
            break;
        };
        if states[work.task_index].started.is_none() {
            states[work.task_index].started = Some(Instant::now());
        }
        work_tx
            .send(work)
            .map_err(|_| NpError::new("hotspot collection worker channel closed"))?;
        *in_flight += 1;
    }
    Ok(())
}

fn fixed_hotspot_batch_count(options: crate::api::DemLogicalCollectionOptions) -> usize {
    if options.max_shots == 0 {
        return 0;
    }
    let first_batch = next_batch_size(options, 0, None);
    let regular_batch = options
        .batch_size
        .min(options.max_batch_size.unwrap_or(options.batch_size))
        .max(1);
    1 + (options.max_shots - first_batch).div_ceil(regular_batch)
}

fn run_hotspot_batch(
    work: HotspotWork,
    decoder: Option<&mut dyn NativeDecoderWorker>,
) -> NpResult<HotspotBatchResult> {
    let started = Instant::now();
    let mut rng = SmallRng::new(batch_seed(work.seed, 0, work.ordinal));
    let batch = work
        .task
        .sampler
        .run_batch_with_rng(work.shots, &mut rng, true);
    let count_options = CountOptions {
        postselection_mask: work.task.postselection_mask.as_deref(),
        postselected_observables_mask: work.task.postselected_observables_mask.as_deref(),
        count_observable_error_combos: work.run_options.count_observable_error_combos,
        count_detection_events: work.run_options.count_detection_events,
    };
    let detailed = count_detailed_batch(&work.task.sampler, &batch, decoder, &count_options)?;
    let estimate = work
        .task
        .sampler
        .estimate_from_loss(&batch, &detailed.loss_mask, None, 0)?;
    let stats = DemLogicalCollectionStats {
        task_id: work.task.task_id.clone(),
        strong_id: work.task.strong_id.clone(),
        decoder: work.task.decoder_name.clone(),
        metadata_json: work.task.metadata_json.clone(),
        shots: detailed.stats.shots,
        errors: detailed.stats.errors,
        discards: detailed.stats.discards,
        seconds: started.elapsed().as_secs_f64(),
        counter_schema: work.run_options.counter_schema(),
        custom_counts: detailed.stats.custom_counts,
    };
    Ok(HotspotBatchResult {
        task_index: work.task_index,
        ordinal: work.ordinal,
        stats,
        edge_sensitivities: estimate.edge_sensitivities,
    })
}

fn hotspot_worker_context(work: &HotspotWork) -> String {
    let backend = work
        .task
        .decoder
        .as_deref()
        .map(|factory| factory.name())
        .unwrap_or("none");
    format!(
        "hotspot collection worker failed while processing task key {} (`{}`) with backend `{backend}`",
        work.task_index, work.task.task_id
    )
}

fn commit_hotspot_result(
    result: HotspotBatchResult,
    states: &mut [HotspotCommitState],
    stop_counter: &ValidatedStopCounter,
) -> NpResult<()> {
    let state = states
        .get_mut(result.task_index)
        .ok_or_else(|| NpError::new("hotspot collection returned invalid task index"))?;
    if state.complete {
        return Ok(());
    }
    state.pending.insert(result.ordinal, result);
    while let Some(batch) = state.pending.remove(&state.next_ordinal) {
        if state.sensitivity_sums.is_empty() {
            state.sensitivity_sums = vec![0.0; batch.edge_sensitivities.len()];
        }
        if batch.edge_sensitivities.len() != state.sensitivity_sums.len() {
            return Err(NpError::new(
                "hotspot collection edge sensitivity shape changed between batches",
            ));
        }
        let shots = batch.stats.shots;
        state.stats.add_assign_checked(&batch.stats)?;
        for (sum, sensitivity) in state
            .sensitivity_sums
            .iter_mut()
            .zip(batch.edge_sensitivities)
        {
            *sum += sensitivity * shots as f64;
        }
        state.batch_stats.push(batch.stats);
        state.next_ordinal += 1;
        let reached_error_limit = if state.stats.shots >= state.min_shots {
            match state.max_errors {
                Some(limit) => stop_error_count(&state.stats, stop_counter)? >= limit,
                None => false,
            }
        } else {
            false
        };
        if state.stats.shots >= state.max_shots || reached_error_limit {
            state.complete = true;
            state.completed_seconds = state.started.map(|started| started.elapsed().as_secs_f64());
            break;
        }
    }
    Ok(())
}

fn finish_hotspot_state(mut state: HotspotCommitState) -> NpResult<DemHotspotCollectionResult> {
    if !state.complete {
        return Err(NpError::new("hotspot collection did not complete a task"));
    }
    state.stats.seconds = state.completed_seconds.unwrap_or(0.0);
    let edge_sensitivities = if state.stats.shots == 0 {
        state.sensitivity_sums
    } else {
        state
            .sensitivity_sums
            .into_iter()
            .map(|sum| sum / state.stats.shots as f64)
            .collect()
    };
    Ok(DemHotspotCollectionResult {
        stats: state.stats,
        batch_stats: state.batch_stats,
        edge_sensitivities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::DemLogicalCollectionOptions;

    #[test]
    fn lazy_hotspot_batch_count_matches_eager_sequence() {
        let cases = [
            DemLogicalCollectionOptions {
                max_shots: 0,
                min_shots: 0,
                max_errors: None,
                batch_size: 100,
                seed: None,
                start_batch_size: None,
                max_batch_size: None,
                max_batch_seconds: None,
            },
            DemLogicalCollectionOptions {
                max_shots: 1_001,
                min_shots: 0,
                max_errors: None,
                batch_size: 128,
                seed: Some(7),
                start_batch_size: Some(17),
                max_batch_size: Some(64),
                max_batch_seconds: None,
            },
            DemLogicalCollectionOptions {
                max_shots: 5_000_000,
                min_shots: 0,
                max_errors: None,
                batch_size: 100,
                seed: None,
                start_batch_size: Some(1),
                max_batch_size: Some(1_000),
                max_batch_seconds: None,
            },
        ];

        for options in cases {
            let mut shots_done = 0usize;
            let mut expected = 0usize;
            while shots_done < options.max_shots {
                shots_done += next_batch_size(options, shots_done, None);
                expected += 1;
            }
            assert_eq!(fixed_hotspot_batch_count(options), expected);
        }
    }
}

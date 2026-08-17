use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use faultscope_core::{NativeDecoderWorker, NpError, NpResult};

use crate::api::{
    task_is_complete, validate_stop_counter_for_tasks, validate_task, CollectionSampler,
    DemLogicalCollectionOptions, DemLogicalCollectionRunOptions, DemLogicalCollectionStats,
    ForwardCollectionSampler, ForwardLogicalCollectionTask, LogicalCollectionTask,
    ValidatedStopCounter,
};
use crate::counting::{
    prepare_count_plan, sample_forward_detailed_batch, CountOptions, PreparedCountPlan,
    PreparedForwardCountPlan,
};
use crate::scheduler::{batch_seed, next_batch_size, task_run_seed};
use crate::worker_decoder::WorkerDecoderCache;
use crate::worker_executor::WorkerExecutor;

#[derive(Debug, Clone, PartialEq)]
pub struct ForwardHotspotCollectionResult {
    pub stats: DemLogicalCollectionStats,
    pub batch_stats: Vec<DemLogicalCollectionStats>,
    pub location_sensitivities: HashMap<String, f64>,
}

#[derive(Clone)]
struct ForwardHotspotWork {
    task_index: usize,
    ordinal: usize,
    shots: usize,
    seed: Option<u64>,
    task: Arc<LogicalCollectionTask>,
    run_options: Arc<DemLogicalCollectionRunOptions>,
    prepared_plan: Arc<PreparedForwardCountPlan>,
}

struct ForwardHotspotBatchResult {
    task_index: usize,
    ordinal: usize,
    stats: DemLogicalCollectionStats,
    location_sensitivities: HashMap<String, f64>,
}

struct ForwardHotspotCommitState {
    stats: DemLogicalCollectionStats,
    batch_stats: Vec<DemLogicalCollectionStats>,
    sensitivity_sums: HashMap<String, f64>,
    pending: HashMap<usize, ForwardHotspotBatchResult>,
    next_ordinal: usize,
    completion_options: DemLogicalCollectionOptions,
    complete: bool,
    started: Option<Instant>,
    completed_seconds: Option<f64>,
}

struct ForwardHotspotTaskCursor {
    task: Arc<LogicalCollectionTask>,
    seed: Option<u64>,
    shots_scheduled: usize,
    next_ordinal: usize,
    prepared_plan: Arc<PreparedForwardCountPlan>,
}

struct ForwardHotspotWorkQueue {
    tasks: Vec<ForwardHotspotTaskCursor>,
    next_task: usize,
    total_batches: usize,
    run_options: Arc<DemLogicalCollectionRunOptions>,
}

impl ForwardHotspotWorkQueue {
    fn new(
        tasks: Vec<LogicalCollectionTask>,
        run_options: Arc<DemLogicalCollectionRunOptions>,
    ) -> NpResult<Self> {
        let total_batches = tasks.iter().fold(0usize, |total, task| {
            total.saturating_add(fixed_hotspot_batch_count(task.options))
        });
        let tasks = tasks
            .into_iter()
            .map(|task| {
                let seed = task_run_seed(&task, &run_options);
                let count_options = CountOptions {
                    postselection_mask: task.postselection_mask.as_deref(),
                    postselected_observables_mask: task.postselected_observables_mask.as_deref(),
                    count_observable_error_combos: run_options.count_observable_error_combos,
                    count_detection_events: run_options.count_detection_events,
                };
                let prepared = prepare_count_plan(
                    &task.sampler,
                    task.decoder
                        .as_deref()
                        .map(|factory| (factory.detector_ids(), factory.batch_formats())),
                    &count_options,
                    true,
                )?;
                let PreparedCountPlan::Forward(prepared_plan) = prepared else {
                    return Err(NpError::new(
                        "forward hotspot collection received a non-forward sampling plan",
                    ));
                };
                Ok(ForwardHotspotTaskCursor {
                    task: Arc::new(task),
                    seed,
                    shots_scheduled: 0,
                    next_ordinal: 0,
                    prepared_plan: Arc::new(prepared_plan),
                })
            })
            .collect::<NpResult<Vec<_>>>()?;
        Ok(Self {
            tasks,
            next_task: 0,
            total_batches,
            run_options,
        })
    }

    fn next_work(
        &mut self,
        states: &[ForwardHotspotCommitState],
        worker_count: usize,
    ) -> Option<ForwardHotspotWork> {
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
            let work = ForwardHotspotWork {
                task_index,
                ordinal: cursor.next_ordinal,
                shots,
                seed: cursor.seed,
                task: cursor.task.clone(),
                run_options: self.run_options.clone(),
                prepared_plan: cursor.prepared_plan.clone(),
            };
            cursor.shots_scheduled += shots;
            cursor.next_ordinal += 1;
            self.next_task = (self.next_task + 1) % self.tasks.len();
            return Some(work);
        }
        None
    }
}

pub fn collect_forward_hotspot_tasks(
    tasks: Vec<ForwardLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
) -> NpResult<Vec<ForwardHotspotCollectionResult>> {
    if run_options.num_workers == 0 {
        return Err(NpError::new("num_workers must be positive"));
    }
    let counter_schema = run_options.counter_schema();
    counter_schema.validate()?;
    let tasks = tasks
        .into_iter()
        .map(LogicalCollectionTask::from)
        .collect::<Vec<_>>();
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
        .map(|task| {
            let stats = DemLogicalCollectionStats::empty_for_logical_task(task, counter_schema);
            let complete = task_is_complete(&stats, &task.options, &stop_counter)?;
            let sampler = forward_sampler(&task.sampler)?;
            let sensitivity_sums = sampler
                .program()
                .noise_locations()
                .iter()
                .map(|location| {
                    (
                        sampler
                            .program()
                            .location_catalog()
                            .label(location.location_id)
                            .to_string(),
                        0.0,
                    )
                })
                .collect();
            Ok(ForwardHotspotCommitState {
                stats,
                batch_stats: Vec::new(),
                sensitivity_sums,
                pending: HashMap::new(),
                next_ordinal: 0,
                completion_options: task.options,
                complete,
                started: None,
                completed_seconds: None,
            })
        })
        .collect::<NpResult<Vec<_>>>()?;
    if states.iter().all(|state| state.complete) {
        return states
            .into_iter()
            .map(finish_forward_hotspot_state)
            .collect();
    }

    let mut work_queue = ForwardHotspotWorkQueue::new(tasks, run_options.clone())?;
    let worker_count = run_options.num_workers.min(work_queue.total_batches).max(1);
    let executor = WorkerExecutor::new(
        worker_count,
        WorkerDecoderCache::new,
        |work, decoder_cache, _| {
            decoder_cache
                .resolve(work.task_index, work.task.decoder.as_ref())
                .and_then(|decoder| run_forward_hotspot_batch(work, decoder))
        },
        forward_hotspot_worker_context,
        forward_hotspot_worker_context,
    );
    let mut in_flight = 0usize;
    schedule_forward_hotspot_work(
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
        "forward hotspot collection worker result channel closed",
        |_| true,
        |result, work_tx, in_flight| {
            commit_forward_hotspot_result(result, &mut states, &stop_counter)?;
            schedule_forward_hotspot_work(
                &mut work_queue,
                &mut states,
                worker_count,
                work_tx,
                in_flight,
            )
        },
        |_| {},
    );
    executor.finish(
        first_error,
        "forward hotspot collection worker thread panicked",
    )?;

    states
        .into_iter()
        .map(finish_forward_hotspot_state)
        .collect()
}

fn schedule_forward_hotspot_work(
    work_queue: &mut ForwardHotspotWorkQueue,
    states: &mut [ForwardHotspotCommitState],
    worker_count: usize,
    work_tx: &mpsc::Sender<ForwardHotspotWork>,
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
            .map_err(|_| NpError::new("forward hotspot collection worker channel closed"))?;
        *in_flight += 1;
    }
    Ok(())
}

fn fixed_hotspot_batch_count(options: DemLogicalCollectionOptions) -> usize {
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

fn run_forward_hotspot_batch(
    work: ForwardHotspotWork,
    decoder: Option<&mut dyn NativeDecoderWorker>,
) -> NpResult<ForwardHotspotBatchResult> {
    let started = Instant::now();
    let count_options = CountOptions {
        postselection_mask: work.task.postselection_mask.as_deref(),
        postselected_observables_mask: work.task.postselected_observables_mask.as_deref(),
        count_observable_error_combos: work.run_options.count_observable_error_combos,
        count_detection_events: work.run_options.count_detection_events,
    };
    let sampler = forward_sampler(&work.task.sampler)?;
    let sampled = sample_forward_detailed_batch(
        sampler.view(),
        work.shots,
        batch_seed(work.seed, 0, work.ordinal),
        decoder,
        &count_options,
        &work.prepared_plan,
    )?;
    let estimate = sampler.program().estimate_from_loss(
        &sampled.state,
        &sampled.detailed.loss_mask,
        None,
        0,
    )?;
    let stats = DemLogicalCollectionStats {
        task_id: work.task.task_id.clone(),
        strong_id: work.task.strong_id.clone(),
        decoder: work.task.decoder_name.clone(),
        metadata_json: work.task.metadata_json.clone(),
        shots: sampled.detailed.stats.shots,
        errors: sampled.detailed.stats.errors,
        discards: sampled.detailed.stats.discards,
        seconds: started.elapsed().as_secs_f64(),
        counter_schema: work.run_options.counter_schema(),
        custom_counts: sampled.detailed.stats.custom_counts,
    };
    Ok(ForwardHotspotBatchResult {
        task_index: work.task_index,
        ordinal: work.ordinal,
        stats,
        location_sensitivities: estimate.sensitivities,
    })
}

fn forward_sampler(sampler: &CollectionSampler) -> NpResult<&ForwardCollectionSampler> {
    match sampler {
        CollectionSampler::Forward(sampler) => Ok(sampler),
        CollectionSampler::Dem(_) => Err(NpError::new(
            "forward hotspot collection received a DEM sampler",
        )),
    }
}

fn forward_hotspot_worker_context(work: &ForwardHotspotWork) -> String {
    let backend = work
        .task
        .decoder
        .as_deref()
        .map(|factory| factory.name())
        .unwrap_or("none");
    format!(
        "forward hotspot collection worker failed while processing task key {} (`{}`) with backend `{backend}`",
        work.task_index, work.task.task_id
    )
}

fn commit_forward_hotspot_result(
    result: ForwardHotspotBatchResult,
    states: &mut [ForwardHotspotCommitState],
    stop_counter: &ValidatedStopCounter,
) -> NpResult<()> {
    let state = states
        .get_mut(result.task_index)
        .ok_or_else(|| NpError::new("forward hotspot collection returned invalid task index"))?;
    if state.complete {
        return Ok(());
    }
    state.pending.insert(result.ordinal, result);
    while let Some(batch) = state.pending.remove(&state.next_ordinal) {
        if !state.sensitivity_sums.is_empty()
            && (state.sensitivity_sums.len() != batch.location_sensitivities.len()
                || state
                    .sensitivity_sums
                    .keys()
                    .any(|key| !batch.location_sensitivities.contains_key(key)))
        {
            return Err(NpError::new(
                "forward hotspot location layout changed between batches",
            ));
        }
        let shots = batch.stats.shots;
        state.stats.add_assign_checked(&batch.stats)?;
        for (location, sensitivity) in batch.location_sensitivities {
            *state.sensitivity_sums.entry(location).or_insert(0.0) += sensitivity * shots as f64;
        }
        state.batch_stats.push(batch.stats);
        state.next_ordinal += 1;
        if task_is_complete(&state.stats, &state.completion_options, stop_counter)? {
            state.complete = true;
            state.completed_seconds = state.started.map(|started| started.elapsed().as_secs_f64());
            break;
        }
    }
    Ok(())
}

fn finish_forward_hotspot_state(
    mut state: ForwardHotspotCommitState,
) -> NpResult<ForwardHotspotCollectionResult> {
    if !state.complete {
        return Err(NpError::new(
            "forward hotspot collection did not complete a task",
        ));
    }
    state.stats.seconds = state.completed_seconds.unwrap_or(0.0);
    let location_sensitivities = if state.stats.shots == 0 {
        state.sensitivity_sums
    } else {
        state
            .sensitivity_sums
            .into_iter()
            .map(|(location, sum)| (location, sum / state.stats.shots as f64))
            .collect()
    };
    Ok(ForwardHotspotCollectionResult {
        stats: state.stats,
        batch_stats: state.batch_stats,
        location_sensitivities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faultscope_core::{
        compile_sampler_program_ref, run_sampler_program, NoiseLocation, NoiseModel, Operation,
    };

    #[test]
    fn multi_batch_hotspot_matches_shot_weighted_direct_forward_estimates() {
        let program = Arc::new(
            compile_sampler_program_ref(
                1,
                &[
                    Operation::Noise(NoiseLocation {
                        id: "x0".to_string(),
                        model: NoiseModel::BernoulliPauli("X".to_string()),
                        rate: 0.375,
                        qubits: vec![0],
                        tags: HashMap::new(),
                    }),
                    Operation::Measure {
                        qubit: 0,
                        key: Some("m".to_string()),
                        basis: "Z".to_string(),
                        noise: None,
                    },
                    Operation::ObservableInclude {
                        observable_id: 7,
                        measurement_keys: vec!["m".to_string()],
                    },
                ],
                Vec::new(),
            )
            .unwrap(),
        );
        let shots = 130;
        let batch_size = 64;
        let task_seed = 0xa125_4f09;
        let task = ForwardLogicalCollectionTask {
            task_id: "direct-match".to_string(),
            strong_id: "direct-match-strong".to_string(),
            sampling_id: "direct-match-sampling".to_string(),
            sampler: program.clone(),
            decoder: None,
            decoder_name: None,
            metadata_json: "null".to_string(),
            options: DemLogicalCollectionOptions {
                max_shots: shots,
                min_shots: 0,
                max_errors: None,
                batch_size,
                seed: Some(task_seed),
                start_batch_size: None,
                max_batch_size: None,
                max_batch_seconds: None,
            },
            postselection_mask: None,
            postselected_observables_mask: None,
        };
        let result = collect_forward_hotspot_tasks(
            vec![task],
            DemLogicalCollectionRunOptions {
                num_workers: 1,
                seed: Some(17),
                count_observable_error_combos: false,
                count_detection_events: false,
                custom_error_count_key: None,
            },
        )
        .unwrap()
        .remove(0);

        let mut expected_errors = 0;
        let mut expected_sensitivity_sum = 0.0;
        for (ordinal, batch_shots) in [64, 64, 2].into_iter().enumerate() {
            let state = run_sampler_program(
                &program,
                batch_shots,
                Some(batch_seed(Some(task_seed), 0, ordinal)),
                true,
            )
            .unwrap();
            let loss = state.observables.get(&7).unwrap();
            let direct = program.estimate_from_loss(&state, loss, None, 0).unwrap();
            expected_errors += loss.bit_count();
            expected_sensitivity_sum += direct.sensitivities["x0"] * batch_shots as f64;
        }

        assert_eq!(
            result
                .batch_stats
                .iter()
                .map(|batch| batch.shots)
                .collect::<Vec<_>>(),
            vec![64, 64, 2]
        );
        assert_eq!(result.stats.errors, expected_errors);
        assert_eq!(
            result.location_sensitivities["x0"],
            expected_sensitivity_sum / shots as f64
        );
    }
}

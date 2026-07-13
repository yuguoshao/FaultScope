use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use faultscope_core::{compute_dem_estimate, NativeDecoderWorker, NpError, NpResult, SmallRng};

use crate::api::{
    stop_error_count, validate_task, DemLogicalCollectionRunOptions, DemLogicalCollectionStats,
    DemLogicalCollectionTask,
};
use crate::counting::{count_detailed_batch, CountOptions};
use crate::scheduler::{batch_seed, next_batch_size, task_run_seed};
use crate::worker_decoder::WorkerDecoderCache;

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

pub fn collect_dem_hotspot_tasks(
    tasks: Vec<DemLogicalCollectionTask>,
    run_options: DemLogicalCollectionRunOptions,
) -> NpResult<Vec<DemHotspotCollectionResult>> {
    if run_options.num_workers == 0 {
        return Err(NpError::new("num_workers must be positive"));
    }
    for task in &tasks {
        validate_task(task)?;
        if task.options.max_batch_seconds.is_some() {
            return Err(NpError::new(
                "hotspot collection does not support adaptive batch sizing",
            ));
        }
    }
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let run_options = Arc::new(run_options);
    let mut states = tasks
        .iter()
        .map(|task| HotspotCommitState {
            stats: DemLogicalCollectionStats::empty_for_task(task),
            batch_stats: Vec::new(),
            sensitivity_sums: vec![0.0; task.sampler.edges.len()],
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
    let mut work_by_task = (0..tasks.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    for (task_index, task) in tasks.into_iter().enumerate() {
        let seed = task_run_seed(&task, &run_options);
        let task = Arc::new(task);
        let mut shots_done = 0usize;
        let mut ordinal = 0usize;
        while shots_done < task.options.max_shots {
            let shots = next_batch_size(task.options, shots_done, None);
            work_by_task[task_index].push(HotspotWork {
                task_index,
                ordinal,
                shots,
                seed,
                task: task.clone(),
                run_options: run_options.clone(),
            });
            shots_done += shots;
            ordinal += 1;
        }
    }
    let mut work_items = Vec::new();
    let max_batches = work_by_task.iter().map(Vec::len).max().unwrap_or(0);
    for ordinal in 0..max_batches {
        for task_work in &work_by_task {
            if let Some(work) = task_work.get(ordinal) {
                work_items.push(work.clone());
            }
        }
    }

    let worker_count = run_options.num_workers.min(work_items.len()).max(1);
    let (work_tx, work_rx) = mpsc::channel::<HotspotWork>();
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (result_tx, result_rx) = mpsc::channel::<NpResult<HotspotBatchResult>>();
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let work_rx = work_rx.clone();
        let result_tx = result_tx.clone();
        handles.push(thread::spawn(move || {
            let mut decoder_cache = WorkerDecoderCache::new();
            loop {
                let work = {
                    let receiver = work_rx.lock().unwrap();
                    receiver.recv()
                };
                let Ok(work) = work else {
                    break;
                };
                let context = hotspot_worker_context(&work);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    decoder_cache
                        .resolve(work.task_index, work.task.decoder.as_ref())
                        .and_then(|decoder| run_hotspot_batch(work, decoder))
                }))
                .map_or_else(
                    |payload| {
                        Err(NpError::new(format!(
                            "{context}: {}",
                            panic_payload_message(payload.as_ref())
                        )))
                    },
                    |result| {
                        result.map_err(|err| NpError::new(format!("{context}: {}", err.message())))
                    },
                );
                if result_tx.send(result).is_err() {
                    break;
                }
            }
        }));
    }
    drop(result_tx);
    let mut next_work = 0usize;
    let mut in_flight = 0usize;
    schedule_hotspot_work(
        &work_items,
        &mut next_work,
        &mut states,
        worker_count,
        &work_tx,
        &mut in_flight,
    )?;
    let mut first_error = None;
    while in_flight > 0 {
        match result_rx.recv() {
            Ok(Ok(result)) => {
                in_flight -= 1;
                if first_error.is_none() {
                    if let Err(err) = commit_hotspot_result(result, &mut states, &run_options) {
                        first_error = Some(err);
                    } else if let Err(err) = schedule_hotspot_work(
                        &work_items,
                        &mut next_work,
                        &mut states,
                        worker_count,
                        &work_tx,
                        &mut in_flight,
                    ) {
                        first_error = Some(err);
                    }
                }
            }
            Ok(Err(err)) => {
                in_flight -= 1;
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
            Err(_) => {
                if first_error.is_none() {
                    first_error = Some(NpError::new(
                        "hotspot collection worker result channel closed",
                    ));
                }
                break;
            }
        }
    }
    drop(work_tx);
    for handle in handles {
        if handle.join().is_err() && first_error.is_none() {
            first_error = Some(NpError::new("hotspot collection worker thread panicked"));
        }
    }
    if let Some(err) = first_error {
        return Err(err);
    }

    states.into_iter().map(finish_hotspot_state).collect()
}

fn schedule_hotspot_work(
    work_items: &[HotspotWork],
    next_work: &mut usize,
    states: &mut [HotspotCommitState],
    worker_count: usize,
    work_tx: &mpsc::Sender<HotspotWork>,
    in_flight: &mut usize,
) -> NpResult<()> {
    while *in_flight < worker_count && *next_work < work_items.len() {
        let work = &work_items[*next_work];
        if states[work.task_index].complete {
            *next_work += 1;
            continue;
        }
        if work.ordinal >= states[work.task_index].next_ordinal + worker_count {
            break;
        }
        *next_work += 1;
        if states[work.task_index].started.is_none() {
            states[work.task_index].started = Some(Instant::now());
        }
        work_tx
            .send(work.clone())
            .map_err(|_| NpError::new("hotspot collection worker channel closed"))?;
        *in_flight += 1;
    }
    Ok(())
}

fn run_hotspot_batch(
    work: HotspotWork,
    decoder: Option<&mut (dyn NativeDecoderWorker + 'static)>,
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
    let estimate = compute_dem_estimate(
        &work.task.sampler.edges,
        &work.task.sampler.location_groups,
        &batch,
        &detailed.loss_mask,
        None,
        0,
    );
    let stats = DemLogicalCollectionStats {
        task_id: work.task.task_id.clone(),
        strong_id: work.task.strong_id.clone(),
        decoder: work.task.decoder_name.clone(),
        metadata_json: work.task.metadata_json.clone(),
        shots: detailed.stats.shots,
        errors: detailed.stats.errors,
        discards: detailed.stats.discards,
        seconds: started.elapsed().as_secs_f64(),
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
    let backend = work.task.decoder_name.as_deref().unwrap_or("none");
    format!(
        "hotspot collection worker failed while processing task key {} (`{}`) with backend `{backend}`",
        work.task_index, work.task.task_id
    )
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "non-string panic payload"
    }
}

fn commit_hotspot_result(
    result: HotspotBatchResult,
    states: &mut [HotspotCommitState],
    run_options: &DemLogicalCollectionRunOptions,
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
        if state.stats.shots >= state.max_shots
            || (state.stats.shots >= state.min_shots
                && state.max_errors.is_some_and(|limit| {
                    stop_error_count(&state.stats, &run_options.custom_error_count_key) >= limit
                }))
        {
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

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use faultscope_collection::api::{
    collect_dem_logical_error_stats, collect_dem_logical_error_tasks,
    collect_dem_logical_error_tasks_with_progress, read_dem_logical_collection_csv,
    sample_dem_logical_error_stats, write_dem_logical_collection_csv_header,
    write_dem_logical_collection_csv_row, DemLogicalCollectionOptions,
    DemLogicalCollectionRunOptions, DemLogicalCollectionStats, DemLogicalCollectionTask,
};
use faultscope_core::{
    CorrectionMaskBatch, DemEvent, DemHotspotEstimator, Detector, DetectorErrorEdge,
    DetectorErrorModel, DetectorMaskBatchView, LogicalObservable, NativeBatchDecoder,
    NativeGraphlikeDetectorCopyDecoder,
};

#[derive(Debug)]
struct ThreadRecordingDecoder {
    threads: Mutex<HashSet<ThreadId>>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    sleep: Duration,
}

impl ThreadRecordingDecoder {
    fn new(detector_ids: Vec<i64>, observable_ids: Vec<i64>) -> Self {
        Self::new_with_sleep(detector_ids, observable_ids, Duration::from_millis(5))
    }

    fn new_with_sleep(detector_ids: Vec<i64>, observable_ids: Vec<i64>, sleep: Duration) -> Self {
        Self {
            threads: Mutex::new(HashSet::new()),
            detector_ids,
            observable_ids,
            sleep,
        }
    }

    fn thread_count(&self) -> usize {
        self.threads.lock().unwrap().len()
    }
}

impl NativeBatchDecoder for ThreadRecordingDecoder {
    fn name(&self) -> &str {
        "thread_recording"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        self.threads.lock().unwrap().insert(thread::current().id());
        thread::sleep(self.sleep);
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug)]
struct CountingDecoder {
    calls: Mutex<usize>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl CountingDecoder {
    fn new(detector_ids: Vec<i64>, observable_ids: Vec<i64>) -> Self {
        Self {
            calls: Mutex::new(0),
            detector_ids,
            observable_ids,
        }
    }

    fn call_count(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

impl NativeBatchDecoder for CountingDecoder {
    fn name(&self) -> &str {
        "counting"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        *self.calls.lock().unwrap() += 1;
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug)]
struct FailingDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl NativeBatchDecoder for FailingDecoder {
    fn name(&self) -> &str {
        "failing"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(
        &self,
        _detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        Err(faultscope_core::NpError::new("intentional decoder failure"))
    }
}

fn logical_edge_dem(probability: f64) -> DetectorErrorModel {
    DetectorErrorModel {
        detectors: Vec::new(),
        observables: vec![LogicalObservable {
            id: 0,
            measurement_keys: Vec::new(),
            pauli_qubits: Vec::new(),
            pauli: String::new(),
        }],
        edges: vec![DetectorErrorEdge {
            probability,
            detectors: Vec::new(),
            observables: vec![0],
            location_id: "logical_edge".to_string(),
            event: DemEvent::Bool(true),
            tags: HashMap::new(),
        }],
    }
}

fn graphlike_dem(probability: f64) -> DetectorErrorModel {
    DetectorErrorModel {
        detectors: vec![Detector {
            id: 0,
            measurement_keys: Vec::new(),
            coords: Vec::new(),
        }],
        observables: vec![LogicalObservable {
            id: 0,
            measurement_keys: Vec::new(),
            pauli_qubits: Vec::new(),
            pauli: String::new(),
        }],
        edges: vec![DetectorErrorEdge {
            probability,
            detectors: vec![0],
            observables: vec![0],
            location_id: "edge0".to_string(),
            event: DemEvent::Pauli("X".to_string()),
            tags: HashMap::new(),
        }],
    }
}

#[test]
fn collection_api_collects_logical_error_stats() {
    let simulator = DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap();

    let stats = collect_dem_logical_error_stats(
        &simulator,
        DemLogicalCollectionOptions {
            max_shots: 10,
            max_errors: Some(1),
            batch_size: 4,
            seed: Some(7),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        None,
    )
    .unwrap();

    assert_eq!(stats.shots, 4);
    assert_eq!(stats.errors, 4);
    assert_eq!(stats.discards, 0);

    let sample = sample_dem_logical_error_stats(&simulator, 3, Some(9), None).unwrap();
    assert_eq!(sample.shots, 3);
    assert_eq!(sample.errors, 3);
}

#[test]
fn native_decoder_removes_graphlike_failures() {
    let dem = graphlike_dem(1.0);
    let decoder = NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(
        dem.compile_graphlike_problem().unwrap(),
    )
    .unwrap();
    let simulator = DemHotspotEstimator::new(dem).unwrap();

    let stats = collect_dem_logical_error_stats(
        &simulator,
        DemLogicalCollectionOptions {
            max_shots: 16,
            max_errors: None,
            batch_size: 5,
            seed: Some(11),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        Some(&decoder),
    )
    .unwrap();

    assert_eq!(stats.shots, 16);
    assert_eq!(stats.errors, 0);
}

#[test]
fn validates_options_and_repeats_seeded_runs() {
    let simulator = DemHotspotEstimator::new(logical_edge_dem(0.375)).unwrap();
    let options = DemLogicalCollectionOptions {
        max_shots: 128,
        max_errors: None,
        batch_size: 17,
        seed: Some(123),
        start_batch_size: None,
        max_batch_size: None,
        max_batch_seconds: None,
    };

    let first = collect_dem_logical_error_stats(&simulator, options, None).unwrap();
    let second = collect_dem_logical_error_stats(&simulator, options, None).unwrap();

    assert_eq!(first.shots, 128);
    assert_eq!(first.errors, second.errors);
    assert!(collect_dem_logical_error_stats(
        &simulator,
        DemLogicalCollectionOptions {
            max_shots: 0,
            max_errors: None,
            batch_size: 1,
            seed: None,
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        None,
    )
    .unwrap_err()
    .message()
    .contains("max_shots"));
    assert!(collect_dem_logical_error_stats(
        &simulator,
        DemLogicalCollectionOptions {
            max_shots: 1,
            max_errors: None,
            batch_size: 0,
            seed: None,
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        None,
    )
    .unwrap_err()
    .message()
    .contains("batch_size"));
}

#[test]
fn parallel_task_collection_is_seed_order_stable() {
    let task_a = DemLogicalCollectionTask {
        task_id: "a".to_string(),
        strong_id: "a-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(0.375)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{\"d\":3}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 200,
            max_errors: None,
            batch_size: 25,
            seed: Some(19),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let task_b = DemLogicalCollectionTask {
        task_id: "b".to_string(),
        strong_id: "b-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(0.125)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{\"d\":5}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 200,
            max_errors: None,
            batch_size: 20,
            seed: Some(23),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let one_worker = collect_dem_logical_error_tasks(
        vec![task_a.clone(), task_b.clone()],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: Some(101),
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();
    let two_workers = collect_dem_logical_error_tasks(
        vec![task_a, task_b],
        DemLogicalCollectionRunOptions {
            num_workers: 2,
            seed: Some(101),
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(
        one_worker
            .iter()
            .map(|stats| (
                stats.task_id.as_str(),
                stats.shots,
                stats.errors,
                stats.discards
            ))
            .collect::<Vec<_>>(),
        two_workers
            .iter()
            .map(|stats| (
                stats.task_id.as_str(),
                stats.shots,
                stats.errors,
                stats.discards
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        two_workers
            .iter()
            .map(|stats| stats.task_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
}

#[test]
fn single_fixed_batch_task_uses_multiple_workers() {
    let decoder = Arc::new(ThreadRecordingDecoder::new(vec![0], vec![0]));
    let task = DemLogicalCollectionTask {
        task_id: "single-parallel".to_string(),
        strong_id: "single-parallel-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 64,
            max_errors: None,
            batch_size: 8,
            seed: Some(29),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let stats = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(stats[0].shots, 64);
    assert!(
        decoder.thread_count() > 1,
        "single fixed-batch task should use multiple worker threads"
    );
}

#[test]
fn multiple_fixed_batch_tasks_share_global_worker_pool() {
    let decoder = Arc::new(ThreadRecordingDecoder::new(vec![0], vec![0]));
    let make_task = |index: usize| DemLogicalCollectionTask {
        task_id: format!("global-{index}"),
        strong_id: format!("global-{index}-strong"),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: format!("{{\"index\":{index}}}"),
        options: DemLogicalCollectionOptions {
            max_shots: 64,
            max_errors: None,
            batch_size: 8,
            seed: Some(53 + index as u64),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let stats = collect_dem_logical_error_tasks(
        vec![make_task(0), make_task(1)],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(
        stats
            .iter()
            .map(|stats| (stats.task_id.as_str(), stats.shots))
            .collect::<Vec<_>>(),
        vec![("global-0", 64), ("global-1", 64)]
    );
    assert!(
        decoder.thread_count() > 2,
        "two fixed-batch tasks should share more than one worker per task"
    );
}

#[test]
fn single_fixed_batch_task_matches_serial_stats() {
    let task = DemLogicalCollectionTask {
        task_id: "single-repeatable".to_string(),
        strong_id: "single-repeatable-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(0.375)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 256,
            max_errors: None,
            batch_size: 16,
            seed: Some(41),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let serial = collect_dem_logical_error_tasks(
        vec![task.clone()],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();
    let parallel = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(parallel[0].shots, serial[0].shots);
    assert_eq!(parallel[0].errors, serial[0].errors);
    assert_eq!(parallel[0].discards, serial[0].discards);
    assert_eq!(parallel[0].custom_counts, serial[0].custom_counts);
}

#[test]
fn fixed_batch_scheduler_preserves_committed_order_with_custom_counts() {
    let make_task = |name: &str, probability: f64, seed: u64| DemLogicalCollectionTask {
        task_id: name.to_string(),
        strong_id: format!("{name}-strong"),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(probability)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: format!("{{\"name\":\"{name}\"}}"),
        options: DemLogicalCollectionOptions {
            max_shots: 70,
            max_errors: None,
            batch_size: 9,
            seed: Some(seed),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let tasks = vec![
        make_task("ordered-a", 0.25, 131),
        make_task("ordered-b", 0.5, 137),
    ];
    let run_options = |num_workers| DemLogicalCollectionRunOptions {
        num_workers,
        seed: Some(139),
        count_observable_error_combos: true,
        count_detection_events: true,
        custom_error_count_key: None,
    };

    let serial =
        collect_dem_logical_error_tasks(tasks.clone(), run_options(1), HashMap::new()).unwrap();
    let parallel = collect_dem_logical_error_tasks(tasks, run_options(4), HashMap::new()).unwrap();

    assert_eq!(
        parallel
            .iter()
            .map(|stats| (
                stats.task_id.as_str(),
                stats.shots,
                stats.errors,
                stats.discards,
                stats.custom_counts.clone()
            ))
            .collect::<Vec<_>>(),
        serial
            .iter()
            .map(|stats| (
                stats.task_id.as_str(),
                stats.shots,
                stats.errors,
                stats.discards,
                stats.custom_counts.clone()
            ))
            .collect::<Vec<_>>()
    );
}

#[test]
fn single_parallel_task_stops_after_completed_max_error_batch() {
    let task = DemLogicalCollectionTask {
        task_id: "single-stop".to_string(),
        strong_id: "single-stop-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 64,
            max_errors: Some(1),
            batch_size: 8,
            seed: Some(43),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let stats = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(stats[0].shots, 8);
    assert_eq!(stats[0].errors, 8);
}

#[test]
fn progress_callback_receives_committed_fixed_batch_deltas_in_order() {
    let task = DemLogicalCollectionTask {
        task_id: "fixed-progress".to_string(),
        strong_id: "fixed-progress-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 10,
            max_errors: None,
            batch_size: 4,
            seed: Some(151),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let mut deltas = Vec::new();

    let stats = collect_dem_logical_error_tasks_with_progress(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
        |delta| {
            deltas.push(delta.clone());
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(stats[0].shots, 10);
    assert_eq!(
        deltas
            .iter()
            .map(|delta| (delta.task_id.as_str(), delta.shots, delta.errors))
            .collect::<Vec<_>>(),
        vec![
            ("fixed-progress", 4, 4),
            ("fixed-progress", 4, 4),
            ("fixed-progress", 2, 2),
        ]
    );
}

#[test]
fn fixed_batch_seconds_track_wall_time_and_preserve_resume_seconds() {
    let decoder = Arc::new(ThreadRecordingDecoder::new_with_sleep(
        vec![0],
        vec![0],
        Duration::from_millis(40),
    ));
    let task = DemLogicalCollectionTask {
        task_id: "fixed-wall-time".to_string(),
        strong_id: "fixed-wall-time-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 8,
            max_errors: None,
            batch_size: 1,
            seed: Some(173),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let existing_seconds = 0.25;
    let existing = DemLogicalCollectionStats {
        task_id: "fixed-wall-time".to_string(),
        strong_id: "fixed-wall-time-strong".to_string(),
        decoder: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        shots: 4,
        errors: 4,
        discards: 0,
        seconds: existing_seconds,
        custom_counts: HashMap::new(),
    };
    let mut deltas = Vec::new();

    let started = Instant::now();
    let stats = collect_dem_logical_error_tasks_with_progress(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::from([("fixed-wall-time-strong".to_string(), existing)]),
        |delta| {
            deltas.push(delta.clone());
            Ok(())
        },
    )
    .unwrap();
    let wall_seconds = started.elapsed().as_secs_f64();

    assert_eq!(stats[0].shots, 8);
    assert_eq!(deltas.len(), 4);
    assert_eq!(deltas.iter().map(|delta| delta.shots).sum::<usize>(), 4);
    assert!(
        decoder.thread_count() > 1,
        "fixed batches should overlap across workers"
    );

    let current_run_seconds = stats[0].seconds - existing_seconds;
    let delta_seconds: f64 = deltas.iter().map(|delta| delta.seconds).sum();
    assert!(
        current_run_seconds > 0.0,
        "current run seconds should be positive"
    );
    assert!(
        (delta_seconds - current_run_seconds).abs()
            <= 0.005_f64.max(current_run_seconds * 0.10),
        "progress delta seconds {delta_seconds} should sum to current run seconds {current_run_seconds}"
    );
    assert!(
        current_run_seconds <= wall_seconds * 1.75 + 0.005,
        "fixed task seconds {current_run_seconds} should track wall time {wall_seconds}, not summed worker time"
    );
}

#[test]
fn progress_callback_stops_at_first_max_error_batch() {
    let task = DemLogicalCollectionTask {
        task_id: "fixed-stop-progress".to_string(),
        strong_id: "fixed-stop-progress-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 64,
            max_errors: Some(1),
            batch_size: 8,
            seed: Some(157),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let mut deltas = Vec::new();

    let stats = collect_dem_logical_error_tasks_with_progress(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
        |delta| {
            deltas.push(delta.clone());
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(stats[0].shots, 8);
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0].shots, 8);
}

#[test]
fn progress_callback_receives_adaptive_task_deltas() {
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-progress".to_string(),
        strong_id: "adaptive-progress-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 9,
            max_errors: None,
            batch_size: 3,
            seed: Some(163),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: Some(1.0),
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let mut deltas = Vec::new();

    let stats = collect_dem_logical_error_tasks_with_progress(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 2,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
        |delta| {
            deltas.push(delta.clone());
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(stats[0].shots, 9);
    assert!(deltas.len() > 1);
    assert_eq!(deltas.iter().map(|delta| delta.shots).sum::<usize>(), 9);
}

#[test]
fn adaptive_progress_error_stops_before_next_batch() {
    let decoder = Arc::new(CountingDecoder::new(vec![0], vec![0]));
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-cancel".to_string(),
        strong_id: "adaptive-cancel-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 4,
            max_errors: None,
            batch_size: 1,
            seed: Some(181),
            start_batch_size: Some(1),
            max_batch_size: Some(1),
            max_batch_seconds: Some(1.0),
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let mut calls = 0usize;

    let err = collect_dem_logical_error_tasks_with_progress(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
        |_delta| {
            calls += 1;
            Err(faultscope_core::NpError::new("stop adaptive progress"))
        },
    )
    .unwrap_err();

    assert_eq!(calls, 1);
    assert!(err.message().contains("stop adaptive progress"));
    assert_eq!(
        decoder.call_count(),
        1,
        "adaptive worker must wait for progress acknowledgement before starting another batch"
    );
}

#[test]
fn progress_callback_error_propagates_after_joining_workers() {
    let task = DemLogicalCollectionTask {
        task_id: "progress-error".to_string(),
        strong_id: "progress-error-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 32,
            max_errors: None,
            batch_size: 4,
            seed: Some(167),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let mut calls = 0usize;

    let err = collect_dem_logical_error_tasks_with_progress(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
        |_delta| {
            calls += 1;
            Err(faultscope_core::NpError::new("intentional progress stop"))
        },
    )
    .unwrap_err();

    assert_eq!(calls, 1);
    assert!(err.message().contains("intentional progress stop"));
}

#[test]
fn adaptive_batch_task_keeps_task_granular_execution() {
    let decoder = Arc::new(ThreadRecordingDecoder::new(vec![0], vec![0]));
    let task = DemLogicalCollectionTask {
        task_id: "adaptive".to_string(),
        strong_id: "adaptive-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 32,
            max_errors: None,
            batch_size: 8,
            seed: Some(47),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: Some(0.01),
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let stats = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(stats[0].shots, 32);
    assert_eq!(decoder.thread_count(), 1);
}

#[test]
fn global_scheduler_resumes_partial_existing_stats() {
    let task = DemLogicalCollectionTask {
        task_id: "partial".to_string(),
        strong_id: "partial-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{\"p\":1}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 16,
            max_errors: None,
            batch_size: 4,
            seed: Some(59),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let existing = DemLogicalCollectionStats {
        task_id: "partial".to_string(),
        strong_id: "partial-strong".to_string(),
        decoder: None,
        metadata_json: "{\"p\":1}".to_string(),
        shots: 8,
        errors: 8,
        discards: 0,
        seconds: 0.25,
        custom_counts: HashMap::new(),
    };

    let stats = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::from([("partial-strong".to_string(), existing)]),
    )
    .unwrap();

    assert_eq!(stats[0].shots, 16);
    assert_eq!(stats[0].errors, 16);
}

#[test]
fn global_scheduler_mixes_fixed_and_adaptive_tasks() {
    let fixed = DemLogicalCollectionTask {
        task_id: "fixed".to_string(),
        strong_id: "fixed-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(0.375)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 64,
            max_errors: None,
            batch_size: 8,
            seed: Some(61),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let adaptive = DemLogicalCollectionTask {
        task_id: "adaptive-mixed".to_string(),
        strong_id: "adaptive-mixed-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(0.375)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 64,
            max_errors: None,
            batch_size: 8,
            seed: Some(67),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: Some(0.01),
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let serial = collect_dem_logical_error_tasks(
        vec![fixed.clone(), adaptive.clone()],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();
    let parallel = collect_dem_logical_error_tasks(
        vec![fixed, adaptive],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(
        parallel
            .iter()
            .map(|stats| (stats.task_id.as_str(), stats.shots, stats.errors))
            .collect::<Vec<_>>(),
        serial
            .iter()
            .map(|stats| (stats.task_id.as_str(), stats.shots, stats.errors))
            .collect::<Vec<_>>()
    );
}

#[test]
fn global_scheduler_returns_worker_errors() {
    let decoder = Arc::new(FailingDecoder {
        detector_ids: vec![0],
        observable_ids: vec![0],
    });
    let task = DemLogicalCollectionTask {
        task_id: "failing".to_string(),
        strong_id: "failing-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 16,
            max_errors: None,
            batch_size: 4,
            seed: Some(71),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let err = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 4,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap_err();

    assert!(err.message().contains("intentional decoder failure"));
}

#[test]
fn postselection_and_custom_counts_are_reported() {
    let simulator = DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap();

    let stats = collect_dem_logical_error_stats(
        &simulator,
        DemLogicalCollectionOptions {
            max_shots: 8,
            max_errors: None,
            batch_size: 8,
            seed: Some(5),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        None,
    )
    .unwrap()
    .with_identity(
        "post".to_string(),
        "post-strong".to_string(),
        None,
        "{}".to_string(),
    );

    let selected = collect_dem_logical_error_tasks(
        vec![DemLogicalCollectionTask {
            task_id: "post".to_string(),
            strong_id: "post-strong".to_string(),
            sampler: Arc::new(simulator),
            decoder: None,
            decoder_name: None,
            metadata_json: "{}".to_string(),
            options: DemLogicalCollectionOptions {
                max_shots: 8,
                max_errors: None,
                batch_size: 8,
                seed: Some(5),
                start_batch_size: None,
                max_batch_size: None,
                max_batch_seconds: None,
            },
            postselection_mask: Some(vec![0b0000_0001]),
            postselected_observables_mask: None,
        }],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: true,
            count_detection_events: true,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(stats.errors, 8);
    assert_eq!(selected[0].shots, 8);
    assert_eq!(selected[0].discards, 8);
    assert_eq!(selected[0].errors, 0);
    assert_eq!(selected[0].custom_counts["detection_events"], 8);
    assert_eq!(selected[0].custom_counts["detectors_checked"], 8);
}

#[test]
fn csv_round_trips_and_resume_skips_completed_tasks() {
    let simulator = Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap());
    let task = DemLogicalCollectionTask {
        task_id: "csv".to_string(),
        strong_id: "csv-strong".to_string(),
        sampler: simulator,
        decoder: None,
        decoder_name: None,
        metadata_json: "{\"p\":1}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 4,
            max_errors: None,
            batch_size: 4,
            seed: Some(7),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let stats = collect_dem_logical_error_tasks(
        vec![task.clone()],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::new(),
    )
    .unwrap();

    let mut csv = Vec::new();
    write_dem_logical_collection_csv_header(&mut csv).unwrap();
    write_dem_logical_collection_csv_row(&mut csv, &stats[0]).unwrap();
    let loaded = read_dem_logical_collection_csv(csv.as_slice()).unwrap();

    assert_eq!(loaded["csv-strong"].strong_id, stats[0].strong_id);
    assert_eq!(loaded["csv-strong"].metadata_json, stats[0].metadata_json);
    assert_eq!(loaded["csv-strong"].shots, stats[0].shots);
    assert_eq!(loaded["csv-strong"].errors, stats[0].errors);

    let resumed = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        loaded,
    )
    .unwrap();

    assert_eq!(resumed[0].shots, 4);
    assert_eq!(resumed[0].errors, 4);
}

#[test]
fn custom_error_count_key_controls_stop_condition() {
    let task = DemLogicalCollectionTask {
        task_id: "custom-stop".to_string(),
        strong_id: "custom-stop-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 10,
            max_errors: Some(4),
            batch_size: 3,
            seed: Some(31),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };

    let stats = collect_dem_logical_error_tasks(
        vec![task],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: true,
            custom_error_count_key: Some("detection_events".to_string()),
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(stats[0].shots, 6);
    assert_eq!(stats[0].custom_counts["detection_events"], 6);
}

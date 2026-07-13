use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Condvar, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use faultscope_collection::{
    collect_dem_hotspot_tasks, collect_dem_logical_error_stats, collect_dem_logical_error_tasks,
    collect_dem_logical_error_tasks_with_progress, sample_dem_logical_error_stats,
    DemLogicalCollectionOptions, DemLogicalCollectionRunOptions, DemLogicalCollectionStats,
    DemLogicalCollectionTask,
};
use faultscope_core::{
    CorrectionMaskBatch, DemEvent, DemHotspotEstimator, Detector, DetectorErrorEdge,
    DetectorErrorModel, DetectorMaskBatchView, LogicalObservable, NativeBatchDecoder,
    NativeGraphlikeDetectorCopyDecoder,
};

#[derive(Debug, Default)]
struct InstanceTracker {
    created: AtomicUsize,
    decode_calls: AtomicUsize,
    concurrent_reentries: AtomicUsize,
    cross_thread_uses: AtomicUsize,
    prototype_decode_calls: AtomicUsize,
}

#[derive(Debug)]
struct WorkerOwnedDecoder {
    tracker: Arc<InstanceTracker>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    is_prototype: bool,
    decode_barrier: Option<Arc<Barrier>>,
    active: AtomicBool,
    owner: Mutex<Option<ThreadId>>,
}

impl WorkerOwnedDecoder {
    fn prototype(tracker: Arc<InstanceTracker>) -> Self {
        Self::prototype_with_barrier(tracker, None)
    }

    fn prototype_with_barrier(
        tracker: Arc<InstanceTracker>,
        decode_barrier: Option<Arc<Barrier>>,
    ) -> Self {
        Self {
            tracker,
            detector_ids: vec![0],
            observable_ids: vec![0],
            is_prototype: true,
            decode_barrier,
            active: AtomicBool::new(false),
            owner: Mutex::new(None),
        }
    }

    fn worker(&self) -> Self {
        Self {
            tracker: self.tracker.clone(),
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
            is_prototype: false,
            decode_barrier: self.decode_barrier.clone(),
            active: AtomicBool::new(false),
            owner: Mutex::new(None),
        }
    }
}

impl NativeBatchDecoder for WorkerOwnedDecoder {
    fn name(&self) -> &str {
        "worker_owned"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        self.tracker.created.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(self.worker()))
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        self.tracker.decode_calls.fetch_add(1, Ordering::SeqCst);
        if self.is_prototype {
            self.tracker
                .prototype_decode_calls
                .fetch_add(1, Ordering::SeqCst);
        }
        if self.active.swap(true, Ordering::SeqCst) {
            self.tracker
                .concurrent_reentries
                .fetch_add(1, Ordering::SeqCst);
        }
        if let Some(barrier) = &self.decode_barrier {
            barrier.wait();
        }
        let current = thread::current().id();
        let mut owner = self.owner.lock().unwrap();
        match *owner {
            Some(previous) if previous != current => {
                self.tracker
                    .cross_thread_uses
                    .fetch_add(1, Ordering::SeqCst);
            }
            None => *owner = Some(current),
            _ => {}
        }
        drop(owner);
        thread::sleep(Duration::from_millis(5));
        self.active.store(false, Ordering::SeqCst);
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug, Default)]
struct GatedAdaptiveTracker {
    created: AtomicUsize,
    dropped: AtomicUsize,
}

#[derive(Debug)]
struct GatedAdaptiveDecoder {
    tracker: Arc<GatedAdaptiveTracker>,
    gate: Arc<(Mutex<bool>, Condvar)>,
    started_tx: Arc<Mutex<Option<mpsc::SyncSender<()>>>>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    is_worker: bool,
}

#[derive(Debug, Default)]
struct HotspotFactoryFailureTracker {
    factory_calls: AtomicUsize,
    created: AtomicUsize,
    dropped: AtomicUsize,
}

#[derive(Debug, Default)]
struct HotspotFactoryFailureState {
    first_decode_started: bool,
    release_decode: bool,
}

#[derive(Debug)]
struct SecondInstanceFailingDecoder {
    tracker: Arc<HotspotFactoryFailureTracker>,
    coordination: Arc<(Mutex<HotspotFactoryFailureState>, Condvar)>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    is_worker: bool,
}

impl Drop for SecondInstanceFailingDecoder {
    fn drop(&mut self) {
        if self.is_worker {
            self.tracker.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl NativeBatchDecoder for SecondInstanceFailingDecoder {
    fn name(&self) -> &str {
        "second_instance_failing"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        let call = self.tracker.factory_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 2 {
            let (lock, changed) = self.coordination.as_ref();
            let mut state = lock.lock().unwrap();
            while !state.first_decode_started {
                state = changed.wait(state).unwrap();
            }
            state.release_decode = true;
            changed.notify_all();
            return Err(faultscope_core::NpError::new(
                "intentional hotspot second worker factory failure",
            ));
        }
        self.tracker.created.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(Self {
            tracker: self.tracker.clone(),
            coordination: self.coordination.clone(),
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
            is_worker: true,
        }))
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        if !self.is_worker {
            return Ok(CorrectionMaskBatch::empty(detectors.shots));
        }
        let (lock, changed) = self.coordination.as_ref();
        let mut state = lock.lock().unwrap();
        state.first_decode_started = true;
        changed.notify_all();
        while !state.release_decode {
            state = changed.wait(state).unwrap();
        }
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

impl Drop for GatedAdaptiveDecoder {
    fn drop(&mut self) {
        if self.is_worker {
            self.tracker.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl NativeBatchDecoder for GatedAdaptiveDecoder {
    fn name(&self) -> &str {
        "gated_adaptive"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        self.tracker.created.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(Self {
            tracker: self.tracker.clone(),
            gate: self.gate.clone(),
            started_tx: self.started_tx.clone(),
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
            is_worker: true,
        }))
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        if let Some(started_tx) = self.started_tx.lock().unwrap().take() {
            let _ = started_tx.send(());
        }
        let (lock, ready) = self.gate.as_ref();
        let mut released = lock.lock().unwrap();
        while !*released {
            released = ready.wait(released).unwrap();
        }
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug)]
struct SignalingUnsupportedDecoder {
    factory_called: Arc<(Mutex<bool>, Condvar)>,
    decode_calls: AtomicUsize,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl NativeBatchDecoder for SignalingUnsupportedDecoder {
    fn name(&self) -> &str {
        "signaling_legacy"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        let (lock, called) = self.factory_called.as_ref();
        *lock.lock().unwrap() = true;
        called.notify_all();
        Err(faultscope_core::NpError::new(
            "signaling_legacy does not support collection worker instances",
        ))
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug)]
struct NearMatchFactoryErrorDecoder {
    decode_calls: AtomicUsize,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl NativeBatchDecoder for NearMatchFactoryErrorDecoder {
    fn name(&self) -> &str {
        "near_match"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        Err(faultscope_core::NpError::new(
            "near_match does not support collection worker instances (temporary)",
        ))
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug)]
struct LegacyTrackingDecoder {
    calls: AtomicUsize,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl LegacyTrackingDecoder {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            detector_ids: vec![0],
            observable_ids: vec![0],
        }
    }
}

impl NativeBatchDecoder for LegacyTrackingDecoder {
    fn name(&self) -> &str {
        "legacy_tracking"
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug)]
struct ThreadRecordingDecoder {
    threads: Arc<Mutex<HashSet<ThreadId>>>,
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
            threads: Arc::new(Mutex::new(HashSet::new())),
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

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        Ok(Arc::new(Self {
            threads: self.threads.clone(),
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
            sleep: self.sleep,
        }))
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
    calls: Arc<Mutex<usize>>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl CountingDecoder {
    fn new(detector_ids: Vec<i64>, observable_ids: Vec<i64>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(0)),
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

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        Ok(Arc::new(Self {
            calls: self.calls.clone(),
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
        }))
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
struct MaskRecordingDecoder {
    batches: Mutex<Vec<Vec<u64>>>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl MaskRecordingDecoder {
    fn new() -> Self {
        Self {
            batches: Mutex::new(Vec::new()),
            detector_ids: vec![0],
            observable_ids: vec![0],
        }
    }

    fn batches(&self) -> Vec<Vec<u64>> {
        self.batches.lock().unwrap().clone()
    }
}

impl NativeBatchDecoder for MaskRecordingDecoder {
    fn name(&self) -> &str {
        "mask_recording"
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
        self.batches
            .lock()
            .unwrap()
            .push(detectors.masks[0].words.clone());
        Ok(CorrectionMaskBatch::empty(detectors.shots))
    }
}

#[derive(Debug)]
struct CalibrationCorrectingDecoder {
    calls: Arc<AtomicUsize>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl CalibrationCorrectingDecoder {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            detector_ids: vec![0],
            observable_ids: vec![0],
        }
    }
}

impl NativeBatchDecoder for CalibrationCorrectingDecoder {
    fn name(&self) -> &str {
        "calibration_correcting"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        Ok(Arc::new(Self {
            calls: self.calls.clone(),
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
        }))
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call < 2 {
            return Ok(CorrectionMaskBatch {
                observable_ids: self.observable_ids.clone(),
                masks: vec![detectors.masks[0].clone()],
                shots: detectors.shots,
            });
        }
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

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn NativeBatchDecoder>> {
        Ok(Arc::new(Self {
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
        }))
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

fn decoder_collection_task(
    task_name: &str,
    decoder: Arc<dyn NativeBatchDecoder>,
    max_shots: usize,
    adaptive: bool,
) -> DemLogicalCollectionTask {
    DemLogicalCollectionTask {
        task_id: task_name.to_string(),
        strong_id: format!("{task_name}-strong"),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder_name: Some(decoder.name().to_string()),
        decoder: Some(decoder),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots,
            min_shots: 0,
            max_errors: None,
            batch_size: 1_000,
            seed: Some(251),
            start_batch_size: adaptive.then_some(1_000),
            max_batch_size: adaptive.then_some(1_000),
            max_batch_seconds: adaptive.then_some(1.0),
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    }
}

fn collection_run_options(num_workers: usize) -> DemLogicalCollectionRunOptions {
    DemLogicalCollectionRunOptions {
        num_workers,
        seed: None,
        count_observable_error_combos: false,
        count_detection_events: false,
        custom_error_count_key: None,
    }
}

#[test]
fn collection_uses_worker_local_decoder() {
    let tracker = Arc::new(InstanceTracker::default());
    let prototype: Arc<dyn NativeBatchDecoder> =
        Arc::new(WorkerOwnedDecoder::prototype_with_barrier(
            tracker.clone(),
            Some(Arc::new(Barrier::new(4))),
        ));
    let task = decoder_collection_task("worker-owned-fixed", prototype, 4_000, false);

    let stats =
        collect_dem_logical_error_tasks(vec![task], collection_run_options(4), HashMap::new())
            .unwrap();

    assert_eq!(stats[0].shots, 4_000);
    assert!(tracker.created.load(Ordering::SeqCst) >= 2);
    assert_eq!(tracker.concurrent_reentries.load(Ordering::SeqCst), 0);
    assert_eq!(tracker.cross_thread_uses.load(Ordering::SeqCst), 0);
    assert_eq!(tracker.prototype_decode_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn hotspot_uses_worker_local_decoder() {
    let serial_tracker = Arc::new(InstanceTracker::default());
    let serial_decoder: Arc<dyn NativeBatchDecoder> =
        Arc::new(WorkerOwnedDecoder::prototype(serial_tracker));
    let mut serial_task =
        decoder_collection_task("hotspot-worker-owned", serial_decoder, 4_000, false);
    serial_task.sampler = Arc::new(DemHotspotEstimator::new(graphlike_dem(0.37)).unwrap());
    let serial = collect_dem_hotspot_tasks(vec![serial_task], collection_run_options(1)).unwrap();

    let tracker = Arc::new(InstanceTracker::default());
    let prototype: Arc<dyn NativeBatchDecoder> =
        Arc::new(WorkerOwnedDecoder::prototype_with_barrier(
            tracker.clone(),
            Some(Arc::new(Barrier::new(4))),
        ));
    let mut parallel_task =
        decoder_collection_task("hotspot-worker-owned", prototype, 4_000, false);
    parallel_task.sampler = Arc::new(DemHotspotEstimator::new(graphlike_dem(0.37)).unwrap());

    let parallel =
        collect_dem_hotspot_tasks(vec![parallel_task], collection_run_options(4)).unwrap();

    assert_eq!(parallel[0].stats.shots, 4_000);
    assert!(tracker.created.load(Ordering::SeqCst) >= 2);
    assert_eq!(tracker.concurrent_reentries.load(Ordering::SeqCst), 0);
    assert_eq!(tracker.cross_thread_uses.load(Ordering::SeqCst), 0);
    assert_eq!(tracker.prototype_decode_calls.load(Ordering::SeqCst), 0);
    assert_eq!(parallel[0].batch_stats.len(), 4);
    assert_eq!(
        parallel[0]
            .batch_stats
            .iter()
            .map(|batch| (batch.shots, batch.errors, batch.discards))
            .collect::<Vec<_>>(),
        serial[0]
            .batch_stats
            .iter()
            .map(|batch| (batch.shots, batch.errors, batch.discards))
            .collect::<Vec<_>>()
    );
    assert_eq!(parallel[0].edge_sensitivities, serial[0].edge_sensitivities);
}

#[test]
fn hotspot_worker_factory_failure() {
    let tracker = Arc::new(HotspotFactoryFailureTracker::default());
    let decoder: Arc<dyn NativeBatchDecoder> = Arc::new(SecondInstanceFailingDecoder {
        tracker: tracker.clone(),
        coordination: Arc::new((
            Mutex::new(HotspotFactoryFailureState::default()),
            Condvar::new(),
        )),
        detector_ids: vec![0],
        observable_ids: vec![0],
        is_worker: false,
    });
    let task = decoder_collection_task("hotspot-factory-failure", decoder, 4_000, false);

    let err = collect_dem_hotspot_tasks(vec![task], collection_run_options(4)).unwrap_err();

    assert_eq!(
        err.message(),
        "intentional hotspot second worker factory failure"
    );
    assert!(tracker.factory_calls.load(Ordering::SeqCst) >= 2);
    assert!(tracker.created.load(Ordering::SeqCst) > 0);
    assert_eq!(
        tracker.created.load(Ordering::SeqCst),
        tracker.dropped.load(Ordering::SeqCst),
        "hotspot collection must return only after every worker-local decoder is dropped"
    );
}

#[test]
fn adaptive_collection_uses_worker_local_decoder() {
    let tracker = Arc::new(InstanceTracker::default());
    let prototype: Arc<dyn NativeBatchDecoder> =
        Arc::new(WorkerOwnedDecoder::prototype(tracker.clone()));
    let task = decoder_collection_task("worker-owned-adaptive", prototype, 6_000, true);

    let stats =
        collect_dem_logical_error_tasks(vec![task], collection_run_options(4), HashMap::new())
            .unwrap();

    assert_eq!(stats[0].shots, 6_000);
    assert!(tracker.created.load(Ordering::SeqCst) >= 2);
    assert_eq!(tracker.concurrent_reentries.load(Ordering::SeqCst), 0);
    assert_eq!(tracker.cross_thread_uses.load(Ordering::SeqCst), 0);
    assert_eq!(tracker.prototype_decode_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn legacy_decoder_uses_prototype_with_one_worker() {
    let decoder = Arc::new(LegacyTrackingDecoder::new());
    let task = decoder_collection_task("legacy-single", decoder.clone(), 4_000, false);

    let stats =
        collect_dem_logical_error_tasks(vec![task], collection_run_options(1), HashMap::new())
            .unwrap();

    assert_eq!(stats[0].shots, 4_000);
    assert_eq!(decoder.calls.load(Ordering::SeqCst), 4);
}

#[test]
fn legacy_decoder_fails_with_multiple_workers_before_decoding() {
    let decoder = Arc::new(LegacyTrackingDecoder::new());
    let task = decoder_collection_task("legacy-parallel", decoder.clone(), 4_000, false);

    let err =
        collect_dem_logical_error_tasks(vec![task], collection_run_options(2), HashMap::new())
            .unwrap_err();

    assert!(err
        .message()
        .contains("does not support collection worker instances"));
    assert_eq!(decoder.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn worker_error_drain_cancels_late_adaptive_delta_and_joins_workers() {
    let factory_called = Arc::new((Mutex::new(false), Condvar::new()));
    let unsupported = Arc::new(SignalingUnsupportedDecoder {
        factory_called: factory_called.clone(),
        decode_calls: AtomicUsize::new(0),
        detector_ids: vec![0],
        observable_ids: vec![0],
    });
    let (started_tx, started_rx) = mpsc::sync_channel(0);
    let adaptive_tracker = Arc::new(GatedAdaptiveTracker::default());
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let adaptive: Arc<dyn NativeBatchDecoder> = Arc::new(GatedAdaptiveDecoder {
        tracker: adaptive_tracker.clone(),
        gate: gate.clone(),
        started_tx: Arc::new(Mutex::new(Some(started_tx))),
        detector_ids: vec![0],
        observable_ids: vec![0],
        is_worker: false,
    });
    let tasks = vec![
        decoder_collection_task("drain-error", unsupported.clone(), 1_000, false),
        decoder_collection_task("drain-adaptive", adaptive, 1_000, true),
    ];
    let (done_tx, done_rx) = mpsc::sync_channel(0);
    let progress_calls = Arc::new(AtomicUsize::new(0));
    let worker_progress_calls = progress_calls.clone();
    let collect_handle = thread::spawn(move || {
        let result = collect_dem_logical_error_tasks_with_progress(
            tasks,
            collection_run_options(2),
            HashMap::new(),
            |_delta| {
                worker_progress_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        );
        let _ = done_tx.send(result);
    });

    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("adaptive calibration work did not reach decode");
    {
        let (lock, called) = factory_called.as_ref();
        let (called, timeout) = called
            .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(1), |called| {
                !*called
            })
            .unwrap();
        assert!(*called, "unsupported factory work did not start");
        assert!(!timeout.timed_out(), "unsupported factory signal timed out");
    }
    thread::sleep(Duration::from_millis(100));
    {
        let (lock, ready) = gate.as_ref();
        *lock.lock().unwrap() = true;
        ready.notify_all();
    }

    let result = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("collection did not drain and join workers after the first error");
    collect_handle.join().unwrap();
    let err = result.unwrap_err();
    assert_eq!(
        err.message(),
        "signaling_legacy does not support collection worker instances"
    );
    assert_eq!(unsupported.decode_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        progress_calls.load(Ordering::SeqCst),
        0,
        "the gated adaptive delta must arrive after the original worker error"
    );
    assert!(adaptive_tracker.created.load(Ordering::SeqCst) > 0);
    assert_eq!(
        adaptive_tracker.created.load(Ordering::SeqCst),
        adaptive_tracker.dropped.load(Ordering::SeqCst),
        "collection must return only after every worker-local decoder is dropped"
    );
}

#[test]
fn mixed_fixed_and_adaptive_tasks_keep_decoder_caches_isolated_by_task_key() {
    let fixed_tracker = Arc::new(InstanceTracker::default());
    let adaptive_tracker = Arc::new(InstanceTracker::default());
    let fixed: Arc<dyn NativeBatchDecoder> =
        Arc::new(WorkerOwnedDecoder::prototype(fixed_tracker.clone()));
    let adaptive: Arc<dyn NativeBatchDecoder> =
        Arc::new(WorkerOwnedDecoder::prototype(adaptive_tracker.clone()));
    let tasks = vec![
        decoder_collection_task("isolated-fixed", fixed, 4_000, false),
        decoder_collection_task("isolated-adaptive", adaptive, 6_000, true),
    ];

    let stats =
        collect_dem_logical_error_tasks(tasks, collection_run_options(4), HashMap::new()).unwrap();

    assert_eq!(stats[0].shots, 4_000);
    assert_eq!(stats[1].shots, 6_000);
    for tracker in [&fixed_tracker, &adaptive_tracker] {
        assert!(tracker.created.load(Ordering::SeqCst) > 0);
        assert!(tracker.decode_calls.load(Ordering::SeqCst) > 0);
        assert_eq!(tracker.concurrent_reentries.load(Ordering::SeqCst), 0);
        assert_eq!(tracker.cross_thread_uses.load(Ordering::SeqCst), 0);
        assert_eq!(tracker.prototype_decode_calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn single_worker_does_not_treat_near_match_factory_error_as_legacy_unsupported() {
    let decoder = Arc::new(NearMatchFactoryErrorDecoder {
        decode_calls: AtomicUsize::new(0),
        detector_ids: vec![0],
        observable_ids: vec![0],
    });
    let task = decoder_collection_task("near-match", decoder.clone(), 1_000, false);

    let err =
        collect_dem_logical_error_tasks(vec![task], collection_run_options(1), HashMap::new())
            .unwrap_err();

    assert_eq!(
        err.message(),
        "near_match does not support collection worker instances (temporary)"
    );
    assert_eq!(decoder.decode_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn collection_api_collects_logical_error_stats() {
    let simulator = DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap();

    let stats = collect_dem_logical_error_stats(
        &simulator,
        DemLogicalCollectionOptions {
            max_shots: 10,
            min_shots: 0,
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

    let zero = collect_dem_logical_error_stats(
        &simulator,
        DemLogicalCollectionOptions {
            max_shots: 10,
            min_shots: 0,
            max_errors: Some(0),
            batch_size: 4,
            seed: Some(7),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        None,
    )
    .unwrap();
    assert_eq!(zero.shots, 0);
    assert_eq!(zero.errors, 0);
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
            min_shots: 0,
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
        min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
fn adaptive_batch_task_uses_multiple_workers_after_calibration() {
    let decoder = Arc::new(ThreadRecordingDecoder::new_with_sleep(
        vec![0],
        vec![0],
        Duration::from_millis(5),
    ));
    let task = DemLogicalCollectionTask {
        task_id: "adaptive".to_string(),
        strong_id: "adaptive-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 64,
            min_shots: 0,
            max_errors: None,
            batch_size: 4,
            seed: Some(47),
            start_batch_size: Some(1),
            max_batch_size: Some(4),
            max_batch_seconds: Some(0.005),
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
        "adaptive work should use multiple worker threads after calibration"
    );
}

#[test]
fn adaptive_parallel_phase_stops_at_first_committed_error_batch() {
    let decoder = Arc::new(CalibrationCorrectingDecoder::new());
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-parallel-stop".to_string(),
        strong_id: "adaptive-parallel-stop-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 20,
            min_shots: 0,
            max_errors: Some(1),
            batch_size: 4,
            seed: Some(191),
            start_batch_size: Some(1),
            max_batch_size: Some(4),
            max_batch_seconds: Some(1.0),
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

    assert_eq!(stats[0].shots, 9);
    assert_eq!(stats[0].errors, 4);
    assert_eq!(
        deltas.iter().map(|delta| delta.shots).collect::<Vec<_>>(),
        vec![1, 4, 4]
    );
    assert_eq!(
        deltas.iter().map(|delta| delta.errors).collect::<Vec<_>>(),
        vec![0, 0, 4]
    );
}

#[test]
fn adaptive_calibration_stops_immediately_at_error_limit() {
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-calibration-stop".to_string(),
        strong_id: "adaptive-calibration-stop-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 20,
            min_shots: 0,
            max_errors: Some(1),
            batch_size: 4,
            seed: Some(193),
            start_batch_size: Some(3),
            max_batch_size: Some(4),
            max_batch_seconds: Some(1.0),
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

    assert_eq!(stats[0].shots, 3);
    assert_eq!(stats[0].errors, 3);
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0].shots, 3);
}

#[test]
fn adaptive_task_resumes_partial_existing_stats_to_target() {
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-resume".to_string(),
        strong_id: "adaptive-resume-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(logical_edge_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 20,
            min_shots: 0,
            max_errors: None,
            batch_size: 4,
            seed: Some(197),
            start_batch_size: Some(1),
            max_batch_size: Some(4),
            max_batch_seconds: Some(1.0),
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let existing = DemLogicalCollectionStats {
        task_id: task.task_id.clone(),
        strong_id: task.strong_id.clone(),
        decoder: None,
        metadata_json: task.metadata_json.clone(),
        shots: 5,
        errors: 5,
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
        HashMap::from([("adaptive-resume-strong".to_string(), existing)]),
    )
    .unwrap();

    assert_eq!(stats[0].shots, 20);
    assert_eq!(stats[0].errors, 20);
    assert!(stats[0].seconds >= 0.25);
}

#[test]
fn adaptive_seconds_and_progress_deltas_track_calibration_plus_parallel_wall_time() {
    let decoder = Arc::new(ThreadRecordingDecoder::new_with_sleep(
        vec![0],
        vec![0],
        Duration::from_millis(20),
    ));
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-wall-time".to_string(),
        strong_id: "adaptive-wall-time-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 10,
            min_shots: 0,
            max_errors: None,
            batch_size: 1,
            seed: Some(211),
            start_batch_size: Some(1),
            max_batch_size: Some(1),
            max_batch_seconds: Some(0.02),
        },
        postselection_mask: None,
        postselected_observables_mask: None,
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
        HashMap::new(),
        |delta| {
            deltas.push(delta.clone());
            thread::sleep(Duration::from_millis(10));
            Ok(())
        },
    )
    .unwrap();
    let wall_seconds = started.elapsed().as_secs_f64();

    assert_eq!(stats[0].shots, 10);
    assert_eq!(deltas.iter().map(|delta| delta.shots).sum::<usize>(), 10);
    assert!(decoder.thread_count() > 1);
    let delta_seconds: f64 = deltas.iter().map(|delta| delta.seconds).sum();
    assert!(
        (delta_seconds - stats[0].seconds).abs() <= 0.005,
        "progress seconds {delta_seconds} should sum to final seconds {}",
        stats[0].seconds
    );
    assert!(
        stats[0].seconds <= wall_seconds * 1.75 + 0.01,
        "adaptive seconds {} should track wall time {wall_seconds}",
        stats[0].seconds
    );
}

#[test]
fn adaptive_parallel_phase_honors_custom_error_stop_counter() {
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-custom-stop".to_string(),
        strong_id: "adaptive-custom-stop-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: None,
        decoder_name: None,
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 20,
            min_shots: 0,
            max_errors: Some(6),
            batch_size: 4,
            seed: Some(223),
            start_batch_size: Some(1),
            max_batch_size: Some(4),
            max_batch_seconds: Some(1.0),
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
            count_detection_events: true,
            custom_error_count_key: Some("detection_events".to_string()),
        },
        HashMap::new(),
    )
    .unwrap();

    assert_eq!(stats[0].shots, 9);
    assert_eq!(stats[0].custom_counts["detection_events"], 9);
}

#[test]
fn adaptive_calibration_propagates_decoder_errors() {
    let decoder = Arc::new(FailingDecoder {
        detector_ids: vec![0],
        observable_ids: vec![0],
    });
    let task = DemLogicalCollectionTask {
        task_id: "adaptive-failing".to_string(),
        strong_id: "adaptive-failing-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(1.0)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 20,
            min_shots: 0,
            max_errors: None,
            batch_size: 4,
            seed: Some(199),
            start_batch_size: Some(1),
            max_batch_size: Some(4),
            max_batch_seconds: Some(1.0),
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
            min_shots: 0,
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
fn seeded_resume_does_not_replay_the_original_detector_batch() {
    let decoder = Arc::new(MaskRecordingDecoder::new());
    let make_task = |max_shots| DemLogicalCollectionTask {
        task_id: "seeded-resume".to_string(),
        strong_id: "seeded-resume-strong".to_string(),
        sampler: Arc::new(DemHotspotEstimator::new(graphlike_dem(0.5)).unwrap()),
        decoder: Some(decoder.clone()),
        decoder_name: Some(decoder.name().to_string()),
        metadata_json: "{}".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots,
            min_shots: 0,
            max_errors: None,
            batch_size: 128,
            seed: Some(229),
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    };
    let first = collect_dem_logical_error_tasks(
        vec![make_task(128)],
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

    collect_dem_logical_error_tasks(
        vec![make_task(256)],
        DemLogicalCollectionRunOptions {
            num_workers: 1,
            seed: None,
            count_observable_error_combos: false,
            count_detection_events: false,
            custom_error_count_key: None,
        },
        HashMap::from([("seeded-resume-strong".to_string(), first[0].clone())]),
    )
    .unwrap();

    let batches = decoder.batches();
    assert_eq!(batches.len(), 2);
    assert_ne!(
        batches[0], batches[1],
        "resumed seeded collection must not replay the original random batch"
    );
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
            min_shots: 0,
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
                min_shots: 0,
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
fn existing_stats_resume_skips_completed_tasks() {
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
            min_shots: 0,
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

    let loaded = HashMap::from([("csv-strong".to_string(), stats[0].clone())]);

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
            min_shots: 0,
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

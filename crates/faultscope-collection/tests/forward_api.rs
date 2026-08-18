use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faultscope_collection::{
    collect_forward_hotspot_tasks, collect_forward_logical_error_tasks,
    DemLogicalCollectionOptions, ForwardLogicalCollectionTask, LogicalCollectionRunOptions,
};
use faultscope_core::{
    compile_sampler_program_ref, DetectorBatchFormat, NativeDecoderFactory, NativeDecoderWorker,
    NativeNoCorrectionDecoder, NoiseLocation, NoiseModel, Operation, SamplerProgram,
};

struct CountingDecoderFactory {
    created: Arc<AtomicUsize>,
    inner: NativeNoCorrectionDecoder,
}

impl NativeDecoderFactory for CountingDecoderFactory {
    fn name(&self) -> &str {
        NativeDecoderFactory::name(&self.inner)
    }

    fn detector_ids(&self) -> &[i64] {
        NativeDecoderFactory::detector_ids(&self.inner)
    }

    fn observable_ids(&self) -> &[i64] {
        NativeDecoderFactory::observable_ids(&self.inner)
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        NativeDecoderFactory::batch_formats(&self.inner)
    }

    fn create_worker(&self) -> faultscope_core::NpResult<Box<dyn NativeDecoderWorker>> {
        self.created.fetch_add(1, Ordering::SeqCst);
        NativeDecoderFactory::create_worker(&self.inner)
    }
}

fn forward_program(probability: f64) -> Arc<SamplerProgram> {
    Arc::new(
        compile_sampler_program_ref(
            1,
            &[
                Operation::Noise(NoiseLocation {
                    id: "x0".to_string(),
                    model: NoiseModel::BernoulliPauli("X".to_string()),
                    rate: probability,
                    qubits: vec![0],
                    tags: HashMap::new(),
                }),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Detector {
                    detector_id: Some(3),
                    measurement_keys: vec!["m".to_string()],
                    coords: Vec::new(),
                },
                Operation::ObservableInclude {
                    observable_id: 7,
                    measurement_keys: vec!["m".to_string()],
                },
            ],
            Vec::new(),
        )
        .unwrap(),
    )
}

fn forward_task(program: Arc<SamplerProgram>) -> ForwardLogicalCollectionTask {
    ForwardLogicalCollectionTask {
        task_id: "forward".to_string(),
        strong_id: "forward-strong".to_string(),
        sampling_id: "forward-sampling".to_string(),
        sampler: program,
        decoder: None,
        decoder_name: None,
        metadata_json: "null".to_string(),
        options: DemLogicalCollectionOptions {
            max_shots: 257,
            min_shots: 0,
            max_errors: None,
            batch_size: 64,
            seed: None,
            start_batch_size: None,
            max_batch_size: None,
            max_batch_seconds: None,
        },
        postselection_mask: None,
        postselected_observables_mask: None,
    }
}

fn run_options(num_workers: usize) -> LogicalCollectionRunOptions {
    LogicalCollectionRunOptions {
        num_workers,
        seed: Some(0x1234_5678),
        count_observable_error_combos: true,
        count_detection_events: true,
        custom_error_count_key: None,
    }
}

#[test]
fn fixed_forward_collection_is_worker_count_deterministic() {
    let task = forward_task(forward_program(0.375));
    let serial =
        collect_forward_logical_error_tasks(vec![task.clone()], run_options(1), HashMap::new())
            .unwrap()
            .remove(0);
    let parallel = collect_forward_logical_error_tasks(vec![task], run_options(4), HashMap::new())
        .unwrap()
        .remove(0);

    assert_eq!(serial.shots, 257);
    assert_eq!(serial.errors, parallel.errors);
    assert_eq!(serial.discards, parallel.discards);
    assert_eq!(serial.custom_counts, parallel.custom_counts);
}

#[test]
fn forward_collection_rejects_duplicate_strong_ids() {
    let program = forward_program(0.375);
    let mut first = forward_task(program.clone());
    first.task_id = "forward-a".to_string();
    first.strong_id = "shared-forward-strong".to_string();
    first.sampling_id = "shared-forward-sampling".to_string();
    let mut second = forward_task(program);
    second.task_id = "forward-b".to_string();
    second.strong_id = first.strong_id.clone();
    second.sampling_id = first.sampling_id.clone();

    let err =
        collect_forward_logical_error_tasks(vec![first, second], run_options(4), HashMap::new())
            .unwrap_err();

    assert_eq!(
        err.message(),
        "duplicate collection strong_id \"shared-forward-strong\" at task indices 0 (\"forward-a\") and 1 (\"forward-b\"); task_id is not part of collection identity"
    );
}

#[test]
fn forward_hotspot_rejects_duplicate_strong_ids_before_workers() {
    let program = forward_program(0.375);
    let created = Arc::new(AtomicUsize::new(0));
    let decoder: Arc<dyn NativeDecoderFactory> = Arc::new(CountingDecoderFactory {
        created: created.clone(),
        inner: NativeNoCorrectionDecoder::with_detector_ids(vec![3], vec![7]).unwrap(),
    });
    let mut first = forward_task(program.clone());
    first.task_id = "forward-hotspot-a".to_string();
    first.strong_id = "shared-forward-hotspot-strong".to_string();
    first.sampling_id = "shared-forward-hotspot-sampling".to_string();
    first.decoder = Some(decoder.clone());
    first.decoder_name = Some(decoder.name().to_string());
    let mut second = forward_task(program);
    second.task_id = "forward-hotspot-b".to_string();
    second.strong_id = first.strong_id.clone();
    second.sampling_id = first.sampling_id.clone();
    second.decoder = Some(decoder.clone());
    second.decoder_name = Some(decoder.name().to_string());

    let err = collect_forward_hotspot_tasks(vec![first, second], run_options(4)).unwrap_err();

    assert_eq!(
        err.message(),
        "duplicate collection strong_id \"shared-forward-hotspot-strong\" at task indices 0 (\"forward-hotspot-a\") and 1 (\"forward-hotspot-b\"); task_id is not part of collection identity"
    );
    assert_eq!(created.load(Ordering::SeqCst), 0);
}

#[test]
fn forward_collection_rejects_nonzero_mask_padding_bits() {
    let program = forward_program(0.375);
    for (name, detector_mask, observable_mask) in [
        ("postselection_mask", Some(vec![0x80]), None),
        ("postselected_observables_mask", None, Some(vec![0x80])),
    ] {
        let mut task = forward_task(program.clone());
        task.postselection_mask = detector_mask;
        task.postselected_observables_mask = observable_mask;

        let err = collect_forward_logical_error_tasks(vec![task], run_options(1), HashMap::new())
            .unwrap_err();
        assert_eq!(
            err.message(),
            format!("{name} has non-zero unused padding bits in its final byte")
        );
    }
}

#[test]
fn forward_collection_runs_native_decoder_and_detailed_postselection() {
    let program = forward_program(1.0);
    let mut decoded = forward_task(program.clone());
    decoded.decoder = Some(Arc::new(
        NativeNoCorrectionDecoder::with_detector_ids(vec![3], vec![7]).unwrap(),
    ));
    decoded.decoder_name = Some("no-correction".to_string());
    decoded.options.max_shots = 8;
    decoded.options.batch_size = 4;
    let decoded_stats =
        collect_forward_logical_error_tasks(vec![decoded], run_options(2), HashMap::new())
            .unwrap()
            .remove(0);
    assert_eq!(decoded_stats.errors, 8);
    assert_eq!(decoded_stats.custom_counts["detection_events"], 8);
    assert_eq!(decoded_stats.custom_counts["detectors_checked"], 8);
    assert_eq!(decoded_stats.custom_counts["obs_mistake_mask=E"], 8);

    let mut postselected = forward_task(program);
    postselected.options.max_shots = 8;
    postselected.options.batch_size = 4;
    postselected.postselection_mask = Some(vec![1]);
    let stats =
        collect_forward_logical_error_tasks(vec![postselected], run_options(2), HashMap::new())
            .unwrap()
            .remove(0);
    assert_eq!(stats.errors, 0);
    assert_eq!(stats.discards, 8);
}

#[test]
fn forward_hotspot_aggregates_all_batches_by_location() {
    let mut task = forward_task(forward_program(0.25));
    task.options.max_shots = 130;
    task.options.batch_size = 64;
    let result = collect_forward_hotspot_tasks(vec![task], run_options(3))
        .unwrap()
        .remove(0);

    assert_eq!(result.stats.shots, 130);
    assert_eq!(result.batch_stats.len(), 3);
    assert_eq!(
        result
            .batch_stats
            .iter()
            .map(|batch| batch.shots)
            .sum::<usize>(),
        130
    );
    assert_eq!(result.location_sensitivities.len(), 1);
    assert!(result.location_sensitivities["x0"].is_finite());
}

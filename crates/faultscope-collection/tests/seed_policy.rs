//! The explicit seed policy is shared by Forward/DEM and both hotspot APIs.
//! Capture decoder inputs so equality checks cover actual samples, not just
//! aggregate error counts that could happen to coincide.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use faultscope_collection::{
    collect_dem_hotspot_tasks, collect_dem_logical_error_tasks, collect_forward_hotspot_tasks,
    collect_forward_logical_error_tasks, DemLogicalCollectionOptions,
    DemLogicalCollectionRunOptions, DemLogicalCollectionStats, DemLogicalCollectionTask,
    ForwardLogicalCollectionTask,
};
use faultscope_core::{
    compile_sampler_program_ref, CorrectionMaskBatch, DecoderCorrectionBatch, DemEvent,
    DemHotspotEstimator, Detector, DetectorBatchFormat, DetectorBatchView, DetectorErrorEdge,
    DetectorErrorModel, LogicalObservable, Mask, NativeDecoderFactory, NativeDecoderWorker,
    NoiseLocation, NoiseModel, NpError, NpResult, Operation, SamplerProgram,
};

const FORMATS: [DetectorBatchFormat; 1] = [DetectorBatchFormat::Masks];
const SHOTS: usize = 1025;
const BATCH_SHOTS: usize = 128;
type CapturedBatches = Vec<(usize, Vec<u64>)>;

#[derive(Clone)]
struct RecordingDecoder {
    name: &'static str,
    correct_errors: bool,
    batches: Arc<Mutex<CapturedBatches>>,
}

impl RecordingDecoder {
    fn new(name: &'static str, correct_errors: bool) -> Arc<Self> {
        Arc::new(Self {
            name,
            correct_errors,
            batches: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn take_batches(&self) -> CapturedBatches {
        let mut batches = std::mem::take(&mut *self.batches.lock().unwrap());
        // Workers may finish in a different order. Preserve every complete
        // input batch, including its shot count, and compare their multiset.
        batches.sort();
        assert_eq!(batches.iter().map(|(shots, _)| shots).sum::<usize>(), SHOTS);
        batches
    }
}

impl NativeDecoderFactory for RecordingDecoder {
    fn name(&self) -> &str {
        self.name
    }
    fn detector_ids(&self) -> &[i64] {
        &[0]
    }
    fn observable_ids(&self) -> &[i64] {
        &[0]
    }
    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &FORMATS
    }
    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        Ok(Box::new(self.clone()))
    }
}

impl NativeDecoderWorker for RecordingDecoder {
    fn name(&self) -> &str {
        self.name
    }
    fn detector_ids(&self) -> &[i64] {
        &[0]
    }
    fn observable_ids(&self) -> &[i64] {
        &[0]
    }
    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &FORMATS
    }
    fn decode_batch(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch> {
        let DetectorBatchView::Masks(batch) = detectors else {
            return Err(NpError::new("seed-policy decoder requires masks"));
        };
        self.batches
            .lock()
            .unwrap()
            .push((batch.shots, batch.masks[0].words.clone()));
        let correction = if self.correct_errors {
            batch.masks[0].clone()
        } else {
            Mask::zero(faultscope_core::word_count(batch.shots))
        };
        Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
            vec![0],
            vec![correction],
            batch.shots,
        )?))
    }
}

enum Source {
    Dem(Arc<DemHotspotEstimator>),
    Forward(Arc<SamplerProgram>),
}

fn sources() -> [Source; 2] {
    let dem = DetectorErrorModel {
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
            probability: 0.375,
            detectors: vec![0],
            observables: vec![0],
            location_id: "x0".to_string(),
            event: DemEvent::Pauli("X".to_string()),
            tags: HashMap::new(),
        }],
    };
    let forward = compile_sampler_program_ref(
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
            Operation::Detector {
                detector_id: Some(0),
                measurement_keys: vec!["m".to_string()],
                coords: Vec::new(),
            },
            Operation::ObservableInclude {
                observable_id: 0,
                measurement_keys: vec!["m".to_string()],
            },
        ],
        Vec::new(),
    )
    .unwrap();
    [
        Source::Dem(Arc::new(DemHotspotEstimator::new(dem).unwrap())),
        Source::Forward(Arc::new(forward)),
    ]
}

#[derive(Clone)]
struct Case {
    label: &'static str,
    seed: Option<u64>,
    decoder: Arc<RecordingDecoder>,
}

fn collect(
    source: &Source,
    hotspot: bool,
    cases: &[Case],
    run_seed: Option<u64>,
    workers: usize,
) -> Vec<DemLogicalCollectionStats> {
    let run_options = DemLogicalCollectionRunOptions {
        num_workers: workers,
        seed: run_seed,
        count_observable_error_combos: true,
        count_detection_events: true,
        custom_error_count_key: None,
    };
    let options = |seed| DemLogicalCollectionOptions {
        max_shots: SHOTS,
        min_shots: SHOTS,
        max_errors: None,
        batch_size: BATCH_SHOTS,
        seed,
        start_batch_size: Some(BATCH_SHOTS),
        max_batch_size: Some(BATCH_SHOTS),
        max_batch_seconds: None,
    };
    match source {
        Source::Dem(sampler) => {
            let tasks = cases
                .iter()
                .map(|case| DemLogicalCollectionTask {
                    task_id: case.label.to_string(),
                    strong_id: format!("{}-strong", case.label),
                    sampling_id: format!("{}-sampling", case.label),
                    sampler: sampler.clone(),
                    decoder: Some(case.decoder.clone()),
                    decoder_name: Some(case.decoder.name.to_string()),
                    metadata_json: format!("{{\"label\":\"{}\"}}", case.label),
                    options: options(case.seed),
                    postselection_mask: None,
                    postselected_observables_mask: None,
                })
                .collect();
            if hotspot {
                collect_dem_hotspot_tasks(tasks, run_options)
                    .unwrap()
                    .into_iter()
                    .map(|result| result.stats)
                    .collect()
            } else {
                collect_dem_logical_error_tasks(tasks, run_options, HashMap::new()).unwrap()
            }
        }
        Source::Forward(sampler) => {
            let tasks = cases
                .iter()
                .map(|case| ForwardLogicalCollectionTask {
                    task_id: case.label.to_string(),
                    strong_id: format!("{}-strong", case.label),
                    sampling_id: format!("{}-sampling", case.label),
                    sampler: sampler.clone(),
                    decoder: Some(case.decoder.clone()),
                    decoder_name: Some(case.decoder.name.to_string()),
                    metadata_json: format!("{{\"label\":\"{}\"}}", case.label),
                    options: options(case.seed),
                    postselection_mask: None,
                    postselected_observables_mask: None,
                })
                .collect();
            if hotspot {
                collect_forward_hotspot_tasks(tasks, run_options)
                    .unwrap()
                    .into_iter()
                    .map(|result| result.stats)
                    .collect()
            } else {
                collect_forward_logical_error_tasks(tasks, run_options, HashMap::new()).unwrap()
            }
        }
    }
}

#[test]
fn metadata_and_decoder_identity_do_not_change_samples_in_any_collector() {
    for source in sources() {
        for hotspot in [false, true] {
            let first = RecordingDecoder::new("uncorrected", false);
            let second = RecordingDecoder::new("corrected", true);
            let cases = [
                Case {
                    label: "original",
                    seed: None,
                    decoder: first.clone(),
                },
                Case {
                    label: "changed-identities",
                    seed: None,
                    decoder: second.clone(),
                },
            ];
            let stats = collect(&source, hotspot, &cases, Some(117), 1);
            let expected = first.take_batches();
            assert_eq!(expected, second.take_batches());
            assert!(stats[0].errors > 0);
            assert_eq!(stats[1].errors, 0);

            // Identity/index changes and a different worker assignment cannot
            // alter the input samples, even when the decoders correct differently.
            let reversed = [cases[1].clone(), cases[0].clone()];
            let stats = collect(&source, hotspot, &reversed, Some(117), 4);
            assert_eq!(expected, first.take_batches());
            assert_eq!(expected, second.take_batches());
            assert_eq!(stats[0].errors, 0);
            assert!(stats[1].errors > 0);
        }
    }
}

#[test]
fn explicit_task_seeds_override_the_run_seed_in_every_collector() {
    for source in sources() {
        for hotspot in [false, true] {
            let inherited = RecordingDecoder::new("inherited", false);
            let explicit = RecordingDecoder::new("explicit-zero", false);
            let independent = RecordingDecoder::new("independent", false);
            let cases = [
                Case {
                    label: "inherited",
                    seed: None,
                    decoder: inherited.clone(),
                },
                Case {
                    label: "explicit",
                    seed: Some(0),
                    decoder: explicit.clone(),
                },
                Case {
                    label: "independent",
                    seed: Some(118),
                    decoder: independent.clone(),
                },
            ];
            collect(&source, hotspot, &cases, Some(0), 4);
            let common = inherited.take_batches();
            let other = independent.take_batches();
            assert_eq!(common, explicit.take_batches());
            assert_ne!(common, other);

            collect(&source, hotspot, &cases, Some(119), 2);
            assert_ne!(common, inherited.take_batches());
            assert_eq!(common, explicit.take_batches());
            assert_eq!(other, independent.take_batches());
        }
    }
}

#[test]
fn unseeded_runs_draw_one_fresh_root_shared_by_their_tasks() {
    for source in sources() {
        for hotspot in [false, true] {
            let first = RecordingDecoder::new("first", false);
            let second = RecordingDecoder::new("second", false);
            let cases = [
                Case {
                    label: "first",
                    seed: None,
                    decoder: first.clone(),
                },
                Case {
                    label: "second",
                    seed: None,
                    decoder: second.clone(),
                },
            ];
            collect(&source, hotspot, &cases, None, 4);
            let previous_run = first.take_batches();
            assert_eq!(previous_run, second.take_batches());
            collect(&source, hotspot, &cases, None, 4);
            let next_run = first.take_batches();
            assert_eq!(next_run, second.take_batches());
            assert_ne!(
                previous_run, next_run,
                "unseeded runs must not use a constant root"
            );
        }
    }
}

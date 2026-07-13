use std::collections::HashMap;
use std::mem::{align_of, offset_of, size_of};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faultscope_core::{
    Circuit, CorrectionMaskBatch, DemHotspotEstimator, Detector, DetectorErrorModelGenerator,
    DetectorMaskBatchView, FaultScopeNativeCorrectionMaskBatchMutViewV1,
    FaultScopeNativeDecoderStatusV1, FaultScopeNativeDecoderStringViewV1,
    FaultScopeNativeDecoderV1, FaultScopeNativeDetectorEventShotBatchViewV1,
    FaultScopeNativeDetectorMaskBatchViewV1, FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeSimulator, LogicalObservable, Mask, NativeBatchDecoder, NativeCompositeDecoder,
    NoiseLocation, NoiseModel, NpError, NpResult, Operation, NATIVE_DECODER_PLUGIN_ABI_NAME,
    NATIVE_DECODER_PLUGIN_ABI_VERSION, NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
    NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
};

struct FactoryDecoder {
    created: Arc<AtomicUsize>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl NativeBatchDecoder for FactoryDecoder {
    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
        let id = self.created.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(FactoryWorkerDecoder::new(
            id,
            self.detector_ids.clone(),
            self.observable_ids.clone(),
        )))
    }

    fn decode_batch(&self, _: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        Err(NpError::new("prototype must not decode"))
    }
}

struct FactoryWorkerDecoder {
    id: usize,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

struct SharedNameFactoryDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    fail: bool,
}

impl NativeBatchDecoder for SharedNameFactoryDecoder {
    fn name(&self) -> &str {
        "shared-child"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
        if self.fail {
            return Err(NpError::new("factory exploded"));
        }
        Ok(Arc::new(FactoryWorkerDecoder::new(
            0,
            self.detector_ids.clone(),
            self.observable_ids.clone(),
        )))
    }

    fn decode_batch(&self, _: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        Err(NpError::new("prototype must not decode"))
    }
}

impl FactoryWorkerDecoder {
    fn new(id: usize, detector_ids: Vec<i64>, observable_ids: Vec<i64>) -> Self {
        Self {
            id,
            detector_ids,
            observable_ids,
        }
    }
}

impl NativeBatchDecoder for FactoryWorkerDecoder {
    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        CorrectionMaskBatch::new(
            self.observable_ids.clone(),
            vec![
                Mask {
                    words: vec![self.id as u64 + 1],
                };
                self.observable_ids.len()
            ],
            detectors.shots,
        )
    }
}

fn x_noise(id: &str, rate: f64) -> NoiseLocation {
    NoiseLocation {
        id: id.to_string(),
        model: NoiseModel::BernoulliPauli("X".to_string()),
        rate,
        qubits: vec![0],
        tags: HashMap::new(),
    }
}

#[test]
fn rust_forward_batch_api_samples_and_estimates() {
    let circuit = Circuit {
        n_qubits: 1,
        operations: vec![
            Operation::Noise(x_noise("x0", 1.0)),
            Operation::Measure {
                qubit: 0,
                key: Some("m".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
        ],
    };
    let simulator = FaultScopeSimulator::new(circuit, Vec::new()).unwrap();

    let batch = simulator.run_batch(32, Some(1), true).unwrap();
    let estimate = simulator.estimate_from_loss(&batch, &batch.measurements["m"], None, 1);

    assert_eq!(batch.measurements["m"], batch.all_mask);
    assert_eq!(batch.event_masks["x0"], batch.all_mask);
    assert_eq!(estimate.mean_loss, 1.0);
    assert_eq!(estimate.top_locations, vec!["x0"]);
}

#[test]
fn rust_dem_generator_api_uses_circuit_declarations() {
    let circuit = Circuit {
        n_qubits: 1,
        operations: vec![
            Operation::Noise(x_noise("x0", 0.25)),
            Operation::Measure {
                qubit: 0,
                key: Some("m".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
            Operation::Detector {
                detector_id: Some(0),
                measurement_keys: vec!["m".to_string()],
                coords: vec![1.0],
            },
            Operation::ObservableInclude {
                observable_id: 0,
                measurement_keys: vec!["m".to_string()],
            },
        ],
    };

    let dem = DetectorErrorModelGenerator::new(circuit, None, None)
        .unwrap()
        .generate()
        .unwrap();

    assert_eq!(dem.detectors[0].id, 0);
    assert_eq!(dem.observables[0].id, 0);
    assert_eq!(dem.edges.len(), 1);
    assert_eq!(dem.edges[0].location_id, "x0");
    assert_eq!(dem.edges[0].detectors, vec![0]);
    assert_eq!(dem.edges[0].observables, vec![0]);
}

#[test]
fn rust_dem_hotspot_api_estimates_default_loss() {
    let circuit = Circuit {
        n_qubits: 1,
        operations: vec![
            Operation::Noise(x_noise("x0", 1.0)),
            Operation::Measure {
                qubit: 0,
                key: Some("m".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
        ],
    };
    let dem = DetectorErrorModelGenerator::new(
        circuit,
        Some(vec![Detector {
            id: 0,
            measurement_keys: vec!["m".to_string()],
            coords: Vec::new(),
        }]),
        Some(vec![LogicalObservable {
            id: 0,
            measurement_keys: vec!["m".to_string()],
            pauli_qubits: Vec::new(),
            pauli: String::new(),
        }]),
    )
    .unwrap()
    .generate()
    .unwrap();
    let simulator = DemHotspotEstimator::new(dem).unwrap();

    let estimate = simulator.estimate_default(32, Some(2), None, 1).unwrap();

    assert_eq!(estimate.mean_loss, 1.0);
    assert_eq!(estimate.top_edges, vec![0]);
    assert_eq!(estimate.top_locations, vec!["x0"]);
}

#[test]
fn native_decoder_worker_instances_are_fresh_and_preserve_ids() {
    let prototype = FactoryDecoder {
        created: Arc::new(AtomicUsize::new(0)),
        detector_ids: vec![10, 20],
        observable_ids: vec![2],
    };

    let first = prototype.create_worker_instance().unwrap();
    let second = prototype.create_worker_instance().unwrap();

    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first.detector_ids(), prototype.detector_ids());
    assert_eq!(second.detector_ids(), prototype.detector_ids());
    assert_eq!(first.observable_ids(), prototype.observable_ids());
    assert_eq!(second.observable_ids(), prototype.observable_ids());
    let detector_masks = vec![Mask::zero(1); 2];
    let first_view = DetectorMaskBatchView::new(first.detector_ids(), &detector_masks, 4).unwrap();
    let second_view =
        DetectorMaskBatchView::new(second.detector_ids(), &detector_masks, 4).unwrap();
    assert_eq!(first.decode_batch(first_view).unwrap().masks[0].words, [1]);
    assert_eq!(
        second.decode_batch(second_view).unwrap().masks[0].words,
        [2]
    );
}

#[test]
fn composite_worker_instance_recursively_creates_fresh_children() {
    let created = Arc::new(AtomicUsize::new(0));
    let prototype = NativeCompositeDecoder::new(vec![
        Arc::new(FactoryDecoder {
            created: Arc::clone(&created),
            detector_ids: vec![10],
            observable_ids: vec![2],
        }),
        Arc::new(FactoryDecoder {
            created: Arc::clone(&created),
            detector_ids: vec![20],
            observable_ids: vec![5],
        }),
    ])
    .unwrap();

    let first = prototype.create_worker_instance().unwrap();
    let second = prototype.create_worker_instance().unwrap();

    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(created.load(Ordering::SeqCst), 4);
    assert_eq!(first.detector_ids(), &[10, 20]);
    assert_eq!(second.detector_ids(), &[10, 20]);
    assert_eq!(first.observable_ids(), &[2, 5]);
    assert_eq!(second.observable_ids(), &[2, 5]);
    let detector_masks = vec![Mask::zero(1); 2];
    let first_view = DetectorMaskBatchView::new(first.detector_ids(), &detector_masks, 4).unwrap();
    let second_view =
        DetectorMaskBatchView::new(second.detector_ids(), &detector_masks, 4).unwrap();
    assert_eq!(
        first
            .decode_batch(first_view)
            .unwrap()
            .masks
            .into_iter()
            .map(|mask| mask.words[0])
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        second
            .decode_batch(second_view)
            .unwrap()
            .masks
            .into_iter()
            .map(|mask| mask.words[0])
            .collect::<Vec<_>>(),
        vec![3, 4]
    );
}

#[test]
fn composite_worker_factory_error_identifies_duplicate_named_child_by_index() {
    let prototype = NativeCompositeDecoder::new(vec![
        Arc::new(SharedNameFactoryDecoder {
            detector_ids: vec![10],
            observable_ids: vec![2],
            fail: false,
        }),
        Arc::new(SharedNameFactoryDecoder {
            detector_ids: vec![20],
            observable_ids: vec![5],
            fail: true,
        }),
    ])
    .unwrap();

    let Err(err) = prototype.create_worker_instance() else {
        panic!("composite accepted a child worker factory failure");
    };

    assert_eq!(
        err.message(),
        "composite decoder `composite` child 1 (`shared-child`) worker factory failed: factory exploded"
    );
}

#[test]
fn native_decoder_v1_abi_layout_is_frozen_on_64_bit_targets() {
    assert_eq!(NATIVE_DECODER_PLUGIN_ABI_VERSION, 1);
    assert_eq!(
        NATIVE_DECODER_PLUGIN_ABI_NAME,
        "faultscope.native_decoder_plugin.v1"
    );
    assert_eq!(
        NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
        NATIVE_DECODER_PLUGIN_ABI_NAME
    );
    assert_eq!(
        NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
        "faultscope.native_decoders"
    );

    if cfg!(target_pointer_width = "64") {
        assert_eq!(size_of::<FaultScopeNativeDecoderStringViewV1>(), 16);
        assert_eq!(size_of::<FaultScopeNativeDetectorMaskBatchViewV1>(), 40);
        assert_eq!(
            size_of::<FaultScopeNativeCorrectionMaskBatchMutViewV1>(),
            40
        );
        assert_eq!(
            size_of::<FaultScopeNativePackedDetectorShotBatchViewV1>(),
            40
        );
        assert_eq!(
            size_of::<FaultScopeNativeDetectorEventShotBatchViewV1>(),
            56
        );
        assert_eq!(size_of::<FaultScopeNativeDecoderStatusV1>(), 24);
        assert_eq!(size_of::<FaultScopeNativeDecoderV1>(), 96);
        assert_eq!(align_of::<FaultScopeNativeDecoderV1>(), 8);
        assert_eq!(offset_of!(FaultScopeNativeDecoderV1, abi_version), 0);
        assert_eq!(offset_of!(FaultScopeNativeDecoderV1, struct_size), 8);
        assert_eq!(offset_of!(FaultScopeNativeDecoderV1, flags), 16);
        assert_eq!(offset_of!(FaultScopeNativeDecoderV1, state), 24);
        assert_eq!(offset_of!(FaultScopeNativeDecoderV1, decode_batch), 64);
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderV1, decode_packed_batch),
            72
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderV1, decode_detector_event_batch),
            80
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderV1, create_worker_state),
            88
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderV1, create_worker_state) + size_of::<usize>(),
            size_of::<FaultScopeNativeDecoderV1>()
        );
    }
}

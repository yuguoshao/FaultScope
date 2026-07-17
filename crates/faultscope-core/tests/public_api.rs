use std::collections::HashMap;
use std::mem::{align_of, offset_of, size_of};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faultscope_core::{
    collect_dem_event_plan, generate_dem_edges_from_event_plan, log_likelihood_ratio, Circuit,
    ConcreteStabilizer, CorrectionMaskBatch, DemHotspotEstimator, Detector,
    DetectorErrorModelGenerator, DetectorMaskBatchView,
    FaultScopeNativeCorrectionMaskBatchMutViewV1, FaultScopeNativeDecoderFactoryV3,
    FaultScopeNativeDecoderStatusV1, FaultScopeNativeDecoderStringViewV1,
    FaultScopeNativeDecoderWorkerV3, FaultScopeNativeDetectorEventShotBatchViewV1,
    FaultScopeNativeDetectorMaskBatchViewV1, FaultScopeNativeGraphlikeEdgeV1,
    FaultScopeNativeGraphlikeProblemV1, FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeSimulator, GraphlikeDecodingProblem, GraphlikeEdge, LogicalObservable, Mask,
    NativeCompositeDecoder, NativeDecoderFactory, NativeDecoderWorker, NoiseLocation, NoiseModel,
    NpError, NpResult, Operation, PauliFrame, NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE,
    NATIVE_DECODER_PLUGIN_ABI_NAME, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_CAPSULE_NAME, NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
    NATIVE_GRAPHLIKE_PROBLEM_ABI_NAME, NATIVE_GRAPHLIKE_PROBLEM_ABI_VERSION,
    NATIVE_GRAPHLIKE_PROBLEM_CAPSULE_NAME,
};

#[test]
fn graphlike_public_constructor_derives_canonical_weight() {
    let problem = GraphlikeDecodingProblem::new(
        vec![10],
        vec![vec![]],
        vec![],
        vec![GraphlikeEdge {
            detectors: vec![0],
            fault_observables: vec![],
            probability: 0.25,
            dem_edge_index: 3,
        }],
    )
    .unwrap();

    assert_eq!(
        problem.edge(0).unwrap().weight(),
        log_likelihood_ratio(0.25)
    );
}

#[test]
fn pauli_api_exposes_only_validated_stateful_operations() {
    let mut state = ConcreteStabilizer::zero(1);
    let apply: fn(&mut ConcreteStabilizer, &[u8], &[u8]) -> NpResult<()> =
        ConcreteStabilizer::apply_pauli_string;
    let is_deterministic: fn(&ConcreteStabilizer, &[u8], &[u8]) -> NpResult<bool> =
        ConcreteStabilizer::is_deterministic_pauli;
    let apply_h: fn(&mut ConcreteStabilizer, usize) -> NpResult<()> = ConcreteStabilizer::apply_h;
    apply(&mut state, &[0], &[0]).unwrap();
    assert!(is_deterministic(&state, &[0], &[1]).unwrap());
    apply_h(&mut state, 0).unwrap();

    let frame_apply_h: fn(&mut PauliFrame, usize) -> NpResult<()> = PauliFrame::apply_h;
    let mut frame = PauliFrame::new(vec![0], vec![0]).unwrap();
    frame_apply_h(&mut frame, 0).unwrap();
    frame.apply_pauli_string(&[0], "X").unwrap();
    assert_eq!(frame.pauli_on(&[0]).unwrap(), "X");

    let apply_event: fn(&mut ConcreteStabilizer, &mut PauliFrame, &[usize], &str) -> NpResult<()> =
        ConcreteStabilizer::apply_pauli_event;
    apply_event(&mut state, &mut frame, &[0], "Z").unwrap();

    assert!(state.apply_pauli_string(&[1, 0], &[0]).is_err());
    assert!(state.is_deterministic_pauli(&[1, 0], &[0]).is_err());
}

struct FactoryDecoder {
    created: Arc<AtomicUsize>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl NativeDecoderFactory for FactoryDecoder {
    fn name(&self) -> &str {
        "factory"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        let id = self.created.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FactoryWorkerDecoder::new(
            "factory",
            id,
            self.detector_ids.clone(),
            self.observable_ids.clone(),
        )))
    }
}

struct FactoryWorkerDecoder {
    name: &'static str,
    id: usize,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

struct SharedNameFactoryDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    fail: bool,
}

impl NativeDecoderFactory for SharedNameFactoryDecoder {
    fn name(&self) -> &str {
        "shared-child"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        if self.fail {
            return Err(NpError::new("factory exploded"));
        }
        Ok(Box::new(FactoryWorkerDecoder::new(
            "shared-child",
            0,
            self.detector_ids.clone(),
            self.observable_ids.clone(),
        )))
    }
}

impl FactoryWorkerDecoder {
    fn new(
        name: &'static str,
        id: usize,
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
    ) -> Self {
        Self {
            name,
            id,
            detector_ids,
            observable_ids,
        }
    }
}

impl NativeDecoderWorker for FactoryWorkerDecoder {
    fn name(&self) -> &str {
        self.name
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(
        &mut self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> NpResult<CorrectionMaskBatch> {
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
    assert_eq!(simulator.program().n_qubits(), 1);
    assert_eq!(
        simulator
            .program()
            .location_catalog()
            .label(simulator.program().noise_locations()[0].location_id),
        "x0"
    );

    let batch = simulator.run_batch(32, Some(1), true).unwrap();
    let measurement = batch.measurements[0].as_ref().unwrap();
    let estimate = simulator
        .estimate_from_loss(&batch, measurement, None, 1)
        .unwrap();

    assert_eq!(measurement, batch.all_mask());
    assert_eq!(&batch.event_masks()[0], batch.all_mask());
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
fn rust_plan_only_dem_api_uses_the_precompiled_integer_plan() {
    let operations = vec![
        Operation::Noise(x_noise("x0", 0.25)),
        Operation::Measure {
            qubit: 0,
            key: Some("m".to_string()),
            basis: "Z".to_string(),
            noise: None,
        },
    ];
    let plan = collect_dem_event_plan(&operations).unwrap();
    let edges = generate_dem_edges_from_event_plan(
        1,
        &[Detector {
            id: 0,
            measurement_keys: vec!["m".to_string()],
            coords: Vec::new(),
        }],
        &[],
        &plan,
    )
    .unwrap();

    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].location_id, "x0");
    assert_eq!(edges[0].detectors, vec![0]);
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

    let mut first = prototype.create_worker().unwrap();
    let mut second = prototype.create_worker().unwrap();

    assert!(!std::ptr::eq(first.as_ref(), second.as_ref()));
    assert_eq!(first.name(), prototype.name());
    assert_eq!(second.name(), prototype.name());
    assert_eq!(first.detector_ids(), prototype.detector_ids());
    assert_eq!(second.detector_ids(), prototype.detector_ids());
    assert_eq!(first.observable_ids(), prototype.observable_ids());
    assert_eq!(second.observable_ids(), prototype.observable_ids());
    let detector_masks = vec![Mask::zero(1); 2];
    let first_detector_ids = first.detector_ids().to_vec();
    let second_detector_ids = second.detector_ids().to_vec();
    let first_view = DetectorMaskBatchView::new(&first_detector_ids, &detector_masks, 4).unwrap();
    let second_view = DetectorMaskBatchView::new(&second_detector_ids, &detector_masks, 4).unwrap();
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

    let mut first = prototype.create_worker().unwrap();
    let mut second = prototype.create_worker().unwrap();

    assert!(!std::ptr::eq(first.as_ref(), second.as_ref()));
    assert_eq!(created.load(Ordering::SeqCst), 4);
    assert_eq!(first.detector_ids(), &[10, 20]);
    assert_eq!(second.detector_ids(), &[10, 20]);
    assert_eq!(first.observable_ids(), &[2, 5]);
    assert_eq!(second.observable_ids(), &[2, 5]);
    let detector_masks = vec![Mask::zero(1); 2];
    let first_detector_ids = first.detector_ids().to_vec();
    let second_detector_ids = second.detector_ids().to_vec();
    let first_view = DetectorMaskBatchView::new(&first_detector_ids, &detector_masks, 4).unwrap();
    let second_view = DetectorMaskBatchView::new(&second_detector_ids, &detector_masks, 4).unwrap();
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

    let Err(err) = prototype.create_worker() else {
        panic!("composite accepted a child worker factory failure");
    };

    assert_eq!(
        err.message(),
        "composite decoder `composite` child 1 (`shared-child`) worker factory failed: factory exploded"
    );
}

#[test]
fn native_decoder_v3_abi_layout_is_frozen_on_64_bit_targets() {
    assert_eq!(NATIVE_DECODER_PLUGIN_ABI_VERSION, 3);
    assert_eq!(
        NATIVE_DECODER_PLUGIN_ABI_NAME,
        "faultscope.native_decoder_plugin.v3"
    );
    assert_eq!(
        NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
        NATIVE_DECODER_PLUGIN_ABI_NAME
    );
    assert_eq!(
        NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
        "faultscope.native_decoders"
    );
    assert_eq!(NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE, 1 << 0);

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
        assert_eq!(size_of::<FaultScopeNativeDecoderFactoryV3>(), 72);
        assert_eq!(align_of::<FaultScopeNativeDecoderFactoryV3>(), 8);
        assert_eq!(offset_of!(FaultScopeNativeDecoderFactoryV3, abi_version), 0);
        assert_eq!(offset_of!(FaultScopeNativeDecoderFactoryV3, struct_size), 8);
        assert_eq!(offset_of!(FaultScopeNativeDecoderFactoryV3, flags), 16);
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderFactoryV3, factory_state),
            24
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderFactoryV3, drop_factory_state),
            32
        );
        assert_eq!(offset_of!(FaultScopeNativeDecoderFactoryV3, name), 40);
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderFactoryV3, detector_ids),
            48
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderFactoryV3, observable_ids),
            56
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderFactoryV3, create_worker),
            64
        );

        assert_eq!(size_of::<FaultScopeNativeDecoderWorkerV3>(), 48);
        assert_eq!(align_of::<FaultScopeNativeDecoderWorkerV3>(), 8);
        assert_eq!(offset_of!(FaultScopeNativeDecoderWorkerV3, struct_size), 0);
        assert_eq!(offset_of!(FaultScopeNativeDecoderWorkerV3, worker_state), 8);
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderWorkerV3, drop_worker_state),
            16
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderWorkerV3, decode_batch),
            24
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderWorkerV3, decode_packed_batch),
            32
        );
        assert_eq!(
            offset_of!(FaultScopeNativeDecoderWorkerV3, decode_detector_event_batch),
            40
        );
    }
}

#[test]
fn native_graphlike_problem_v1_abi_layout_is_frozen_on_64_bit_targets() {
    assert_eq!(NATIVE_GRAPHLIKE_PROBLEM_ABI_VERSION, 1);
    assert_eq!(
        NATIVE_GRAPHLIKE_PROBLEM_ABI_NAME,
        "faultscope.native_graphlike_problem.v1"
    );
    assert_eq!(
        NATIVE_GRAPHLIKE_PROBLEM_CAPSULE_NAME,
        NATIVE_GRAPHLIKE_PROBLEM_ABI_NAME
    );

    if cfg!(target_pointer_width = "64") {
        assert_eq!(size_of::<FaultScopeNativeGraphlikeEdgeV1>(), 48);
        assert_eq!(align_of::<FaultScopeNativeGraphlikeEdgeV1>(), 8);
        assert_eq!(
            offset_of!(FaultScopeNativeGraphlikeEdgeV1, dem_edge_index),
            0
        );
        assert_eq!(offset_of!(FaultScopeNativeGraphlikeEdgeV1, probability), 8);
        assert_eq!(offset_of!(FaultScopeNativeGraphlikeEdgeV1, weight), 16);
        assert_eq!(
            offset_of!(FaultScopeNativeGraphlikeEdgeV1, fault_observable_offset),
            24
        );
        assert_eq!(offset_of!(FaultScopeNativeGraphlikeEdgeV1, detector0), 32);
        assert_eq!(
            offset_of!(FaultScopeNativeGraphlikeEdgeV1, detector_count),
            40
        );

        assert_eq!(size_of::<FaultScopeNativeGraphlikeProblemV1>(), 88);
        assert_eq!(align_of::<FaultScopeNativeGraphlikeProblemV1>(), 8);
        assert_eq!(
            offset_of!(FaultScopeNativeGraphlikeProblemV1, abi_version),
            0
        );
        assert_eq!(
            offset_of!(FaultScopeNativeGraphlikeProblemV1, struct_size),
            8
        );
        assert_eq!(offset_of!(FaultScopeNativeGraphlikeProblemV1, edges), 56);
        assert_eq!(
            offset_of!(FaultScopeNativeGraphlikeProblemV1, fault_observables),
            72
        );
    }
}

use std::collections::HashMap;
use std::mem::{align_of, offset_of, size_of};

use faultscope_core::{
    Circuit, DemHotspotEstimator, Detector, DetectorErrorModelGenerator,
    FaultScopeNativeCorrectionMaskBatchMutViewV1, FaultScopeNativeDecoderStatusV1,
    FaultScopeNativeDecoderStringViewV1, FaultScopeNativeDecoderV1,
    FaultScopeNativeDetectorEventShotBatchViewV1, FaultScopeNativeDetectorMaskBatchViewV1,
    FaultScopeNativePackedDetectorShotBatchViewV1, FaultScopeSimulator, LogicalObservable,
    NoiseLocation, NoiseModel, Operation, NATIVE_DECODER_PLUGIN_ABI_NAME,
    NATIVE_DECODER_PLUGIN_ABI_VERSION, NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
    NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
};

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
        assert_eq!(size_of::<FaultScopeNativeDecoderV1>(), 88);
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
    }
}

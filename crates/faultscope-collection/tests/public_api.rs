use std::collections::HashMap;

use faultscope_collection::api::{
    collect_dem_logical_error_stats, sample_dem_logical_error_stats, DemLogicalCollectionOptions,
};
use faultscope_core::{
    DemEvent, DemHotspotEstimator, Detector, DetectorErrorEdge, DetectorErrorModel,
    LogicalObservable, NativeGraphlikeDetectorCopyDecoder,
};

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
        },
        None,
    )
    .unwrap_err()
    .message()
    .contains("batch_size"));
}

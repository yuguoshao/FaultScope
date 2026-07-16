use std::collections::HashMap;

use super::*;
use crate::{DemEvent, NoiseLocation, NoiseModel};

#[test]
fn measurement_bit_flip_generates_detector_edge() {
    let noise = NoiseLocation {
        id: "m_noise".to_string(),
        model: NoiseModel::MeasurementBitFlip,
        rate: 0.25,
        qubits: vec![0],
        tags: HashMap::new(),
    };
    let operations = vec![
        Operation::Measure {
            qubit: 0,
            key: Some("m0".to_string()),
            basis: "Z".to_string(),
            noise: Some(noise),
        },
        Operation::Detector {
            detector_id: Some(0),
            measurement_keys: vec!["m0".to_string()],
            coords: Vec::new(),
        },
    ];
    let detectors = vec![Detector {
        id: 0,
        measurement_keys: vec!["m0".to_string()],
        coords: Vec::new(),
    }];

    let edges = generate_dem_edges(1, &operations, &detectors, &[]).unwrap();

    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].probability, 0.25);
    assert_eq!(edges[0].detectors, vec![0]);
    assert_eq!(edges[0].location_id, "m_noise");
    assert_eq!(edges[0].event, DemEvent::Bool(true));
}

#[test]
fn event_and_measurement_plans_share_integer_program_ids() {
    let location = NoiseLocation {
        id: "x0".to_string(),
        model: NoiseModel::BernoulliPauli("X".to_string()),
        rate: 0.125,
        qubits: vec![0],
        tags: HashMap::new(),
    };
    let operations = vec![
        Operation::Noise(location),
        Operation::Measure {
            qubit: 0,
            key: Some("syndrome".to_string()),
            basis: "Z".to_string(),
            noise: None,
        },
        Operation::Detector {
            detector_id: Some(4),
            measurement_keys: vec!["syndrome".to_string()],
            coords: vec![2.0],
        },
    ];

    let event_plan = collect_dem_event_plan(&operations).unwrap();
    assert!(matches!(
        event_plan.program.operations.as_slice(),
        [
            ExpandedOperation::Noise(0),
            ExpandedOperation::MeasureSingle {
                measurement_id: 0,
                ..
            },
            ExpandedOperation::Detector {
                detector_id: 4,
                measurement_ids,
            }
        ] if measurement_ids.as_slice() == [0]
    ));
    assert_eq!(event_plan.fault_events[0].noise_id, 0);
    assert_eq!(event_plan.fault_event_range_by_noise, vec![0..1]);
    let location = &event_plan.program.noise_locations[0];
    assert_eq!(
        event_plan
            .program
            .location_catalog
            .label(location.location_id),
        "x0"
    );
    assert_eq!(event_plan.program.detector_coords, vec![vec![2.0]]);

    let detectors = event_plan.inferred_detectors();
    let measurement_plan =
        compile_dem_measurement_plan(&event_plan.program, &detectors, &[]).unwrap();
    assert_eq!(measurement_plan.measurement_index_by_id, vec![Some(0)]);
    assert_eq!(measurement_plan.detectors[0].measurement_indices, vec![0]);
}

#[test]
fn dem_entrypoints_reject_invalid_circuit_event_plan_and_observable_targets() {
    let invalid_operations = vec![Operation::H(1)];
    let err = generate_dem_edges(1, &invalid_operations, &[], &[]).unwrap_err();
    assert!(err.message().contains("targets qubit 1"));

    let err = DetectorErrorModelGenerator::new(
        Circuit {
            n_qubits: 1,
            operations: invalid_operations.clone(),
        },
        None,
        None,
    )
    .unwrap_err();
    assert!(err.message().contains("targets qubit 1"));

    let err = ValidatedDemCircuit::new(Circuit {
        n_qubits: 1,
        operations: invalid_operations.clone(),
    })
    .unwrap_err();
    assert!(err.message().contains("targets qubit 1"));

    let event_plan = collect_dem_event_plan(&invalid_operations).unwrap();
    let err = generate_dem_edges_from_event_plan(1, &[], &[], &event_plan).unwrap_err();
    assert!(err.message().contains("targets qubit 1"));

    let err = DetectorErrorModelGenerator::new_with_event_plan(
        Circuit {
            n_qubits: 1,
            operations: Vec::new(),
        },
        Vec::new(),
        Vec::new(),
        event_plan,
    )
    .unwrap_err();
    assert!(err.message().contains("targets qubit 1"));

    let observable = LogicalObservable {
        id: 3,
        measurement_keys: Vec::new(),
        pauli_qubits: vec![1],
        pauli: "Z".to_string(),
    };
    let err = generate_dem_edges(1, &[], &[], &[observable]).unwrap_err();
    assert!(err.message().contains("logical observable 3"));
}

#[test]
fn validated_dem_circuit_reuses_bound_plan_and_matches_checked_generator() {
    let circuit = Circuit {
        n_qubits: 1,
        operations: vec![
            Operation::Noise(NoiseLocation {
                id: "x0".to_string(),
                model: NoiseModel::BernoulliPauli("X".to_string()),
                rate: 0.125,
                qubits: vec![0],
                tags: HashMap::new(),
            }),
            Operation::Measure {
                qubit: 0,
                key: Some("m0".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
            Operation::Detector {
                detector_id: Some(0),
                measurement_keys: vec!["m0".to_string()],
                coords: vec![1.0],
            },
        ],
    };
    let checked = DetectorErrorModelGenerator::new(circuit.clone(), None, None)
        .unwrap()
        .generate()
        .unwrap();
    let validated = ValidatedDemCircuit::new(circuit).unwrap();

    let first_plan = validated.event_plan().unwrap();
    let second_plan = validated.event_plan().unwrap();
    assert!(Arc::ptr_eq(&first_plan, &second_plan));

    let fast =
        DetectorErrorModelGenerator::new_with_validated_dem_circuit_options(&validated, None, None)
            .unwrap()
            .generate()
            .unwrap();
    assert_eq!(fast, checked);
}

#[test]
fn generator_defaults_declarations_from_circuit_and_carries_tags() {
    let mut tags = HashMap::new();
    tags.insert(
        "gate".to_string(),
        crate::TagValue::String("idle".to_string()),
    );
    let location = NoiseLocation {
        id: "x0".to_string(),
        model: NoiseModel::BernoulliPauli("X".to_string()),
        rate: 0.125,
        qubits: vec![0],
        tags,
    };
    let circuit = Circuit {
        n_qubits: 1,
        operations: vec![
            Operation::Noise(location),
            Operation::Measure {
                qubit: 0,
                key: Some("m0".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
            Operation::Detector {
                detector_id: None,
                measurement_keys: vec!["m0".to_string()],
                coords: vec![1.0],
            },
            Operation::ObservableInclude {
                observable_id: 0,
                measurement_keys: vec!["m0".to_string()],
            },
        ],
    };
    let generator = DetectorErrorModelGenerator::new(circuit, None, None).unwrap();

    let dem = generator.generate().unwrap();

    assert_eq!(generator.detectors[0].id, 0);
    assert_eq!(generator.detectors[0].coords, vec![1.0]);
    assert_eq!(generator.observables[0].measurement_keys, vec!["m0"]);
    assert_eq!(dem.edges.len(), 1);
    assert_eq!(dem.edges[0].location_id, "x0");
    assert_eq!(
        dem.edges[0].tags.get("gate"),
        Some(&crate::TagValue::String("idle".to_string()))
    );
}

#[test]
fn generator_sampler_edges_match_full_dem_edges() {
    let mut tags = HashMap::new();
    tags.insert(
        "operation".to_string(),
        crate::TagValue::String("idle".to_string()),
    );
    let location = NoiseLocation {
        id: "x0".to_string(),
        model: NoiseModel::BernoulliPauli("X".to_string()),
        rate: 0.25,
        qubits: vec![0],
        tags,
    };
    let circuit = Circuit {
        n_qubits: 1,
        operations: vec![
            Operation::Noise(location),
            Operation::Measure {
                qubit: 0,
                key: Some("m0".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
            Operation::Detector {
                detector_id: Some(0),
                measurement_keys: vec!["m0".to_string()],
                coords: Vec::new(),
            },
            Operation::ObservableInclude {
                observable_id: 0,
                measurement_keys: vec!["m0".to_string()],
            },
        ],
    };
    let generator = DetectorErrorModelGenerator::new(circuit, None, None).unwrap();

    let first = generator.generate().unwrap();
    let second = generator.generate().unwrap();
    let lazy = generator.generate_lazy().unwrap();
    let lazy_graphlike = lazy.compile_graphlike_problem().unwrap();
    let materialized_graphlike = first.compile_graphlike_problem().unwrap();
    let sampler_edges = lazy.compile_hotspot_estimator().edges();

    assert_eq!(first, second);
    assert_eq!(lazy_graphlike, materialized_graphlike);
    assert_eq!(sampler_edges.len(), first.edges.len());
    for (sampler_edge, dem_edge) in sampler_edges.iter().zip(first.edges.iter()) {
        assert_eq!(sampler_edge.probability, dem_edge.probability);
        assert_eq!(sampler_edge.detectors, dem_edge.detectors);
        assert_eq!(sampler_edge.observables, dem_edge.observables);
        assert_eq!(sampler_edge.location_id, dem_edge.location_id);
        assert_eq!(sampler_edge.event, dem_edge.event);
        assert_eq!(sampler_edge.tags, dem_edge.tags);
    }
}

#[test]
fn consuming_sampling_estimator_skips_metadata_and_preserves_seeded_batches() {
    let circuit = Circuit {
        n_qubits: 1,
        operations: vec![
            Operation::Noise(NoiseLocation {
                id: "x0".to_string(),
                model: NoiseModel::BernoulliPauli("X".to_string()),
                rate: 0.25,
                qubits: vec![0],
                tags: HashMap::new(),
            }),
            Operation::Measure {
                qubit: 0,
                key: Some("m0".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
            Operation::Detector {
                detector_id: Some(0),
                measurement_keys: vec!["m0".to_string()],
                coords: Vec::new(),
            },
        ],
    };
    let generator = DetectorErrorModelGenerator::new(circuit, None, None).unwrap();
    let metadata_estimator = generator
        .generate_lazy()
        .unwrap()
        .compile_hotspot_estimator();
    let sampling_estimator = generator.generate_lazy().unwrap().into_sampling_estimator();

    assert_eq!(sampling_estimator.location_groups(), Vec::new());
    assert_eq!(sampling_estimator.edges()[0].location_id, "");
    assert_eq!(sampling_estimator.edges()[0].event, DemEvent::Bool(false));
    assert_eq!(
        sampling_estimator.run_batch(129, Some(7), true).unwrap(),
        metadata_estimator.run_batch(129, Some(7), true).unwrap()
    );
}

#[test]
fn generator_rejects_duplicate_detector_ids() {
    let circuit = Circuit {
        n_qubits: 1,
        operations: Vec::new(),
    };
    let detectors = vec![
        Detector {
            id: 0,
            measurement_keys: Vec::new(),
            coords: Vec::new(),
        },
        Detector {
            id: 0,
            measurement_keys: Vec::new(),
            coords: Vec::new(),
        },
    ];

    let err = DetectorErrorModelGenerator::new(circuit, Some(detectors), None).unwrap_err();

    assert!(err.message().contains("detector ids must be unique"));
}

#[test]
fn fallback_generator_rejects_unknown_measurement_key_during_construction() {
    let circuit = Circuit {
        n_qubits: 2,
        operations: vec![
            Operation::H(0),
            Operation::Cx(0, 1),
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "ZZ".to_string(),
                key: Some("zz".to_string()),
                noise: None,
            },
        ],
    };
    let detectors = vec![Detector {
        id: 0,
        measurement_keys: vec!["missing".to_string()],
        coords: Vec::new(),
    }];

    let err = DetectorErrorModelGenerator::new(circuit, Some(detectors), None).unwrap_err();

    assert!(err.message().contains("unknown measurement key"));
}

#[test]
fn fallback_generator_rejects_duplicate_measurement_key_during_construction() {
    let circuit = Circuit {
        n_qubits: 2,
        operations: vec![
            Operation::H(0),
            Operation::Cx(0, 1),
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "ZZ".to_string(),
                key: Some("m".to_string()),
                noise: None,
            },
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "ZZ".to_string(),
                key: Some("m".to_string()),
                noise: None,
            },
        ],
    };
    let detectors = vec![Detector {
        id: 0,
        measurement_keys: vec!["m".to_string()],
        coords: Vec::new(),
    }];

    let err = DetectorErrorModelGenerator::new(circuit, Some(detectors), None).unwrap_err();

    assert!(err.message().contains("duplicate measurement key"));
}

#[test]
fn fallback_bell_state_zz_detector_tracks_single_error() {
    let location = NoiseLocation {
        id: "x0".to_string(),
        model: NoiseModel::BernoulliPauli("X".to_string()),
        rate: 0.125,
        qubits: vec![0],
        tags: HashMap::new(),
    };
    let circuit = Circuit {
        n_qubits: 2,
        operations: vec![
            Operation::H(0),
            Operation::Cx(0, 1),
            Operation::Noise(location),
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "ZZ".to_string(),
                key: Some("zz".to_string()),
                noise: None,
            },
        ],
    };
    let detectors = vec![Detector {
        id: 7,
        measurement_keys: vec!["zz".to_string()],
        coords: Vec::new(),
    }];

    let dem = DetectorErrorModelGenerator::new(circuit, Some(detectors), None)
        .unwrap()
        .generate()
        .unwrap();

    assert_eq!(dem.edges.len(), 1);
    assert_eq!(dem.edges[0].location_id, "x0");
    assert_eq!(dem.edges[0].event, DemEvent::Pauli("X".to_string()));
    assert_eq!(dem.edges[0].detectors, vec![7]);
}

#[test]
fn fallback_indexes_required_measurements_and_ignores_unused_measurements() {
    let location = NoiseLocation {
        id: "x0".to_string(),
        model: NoiseModel::BernoulliPauli("X".to_string()),
        rate: 0.25,
        qubits: vec![0],
        tags: HashMap::new(),
    };
    let circuit = Circuit {
        n_qubits: 2,
        operations: vec![
            Operation::H(0),
            Operation::Cx(0, 1),
            Operation::Noise(location),
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "ZZ".to_string(),
                key: Some("zz0".to_string()),
                noise: None,
            },
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "XX".to_string(),
                key: Some("unused".to_string()),
                noise: None,
            },
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "ZZ".to_string(),
                key: Some("zz1".to_string()),
                noise: None,
            },
        ],
    };
    let detectors = vec![Detector {
        id: 2,
        measurement_keys: vec!["zz0".to_string(), "zz1".to_string()],
        coords: Vec::new(),
    }];
    let observables = vec![LogicalObservable {
        id: 3,
        measurement_keys: vec!["zz0".to_string()],
        pauli_qubits: Vec::new(),
        pauli: String::new(),
    }];

    let dem = DetectorErrorModelGenerator::new(circuit, Some(detectors), Some(observables))
        .unwrap()
        .generate()
        .unwrap();

    assert_eq!(dem.edges.len(), 1);
    assert_eq!(dem.edges[0].detectors, Vec::<i64>::new());
    assert_eq!(dem.edges[0].observables, vec![3]);
}

#[test]
fn fallback_measurement_bit_flip_on_indexed_measurement_generates_edge() {
    let location = NoiseLocation {
        id: "mflip".to_string(),
        model: NoiseModel::MeasurementBitFlip,
        rate: 0.2,
        qubits: vec![0],
        tags: HashMap::new(),
    };
    let circuit = Circuit {
        n_qubits: 2,
        operations: vec![
            Operation::H(0),
            Operation::Cx(0, 1),
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "XX".to_string(),
                key: Some("unused".to_string()),
                noise: None,
            },
            Operation::MeasurePauli {
                qubits: vec![0, 1],
                pauli: "ZZ".to_string(),
                key: Some("zz".to_string()),
                noise: Some(location),
            },
        ],
    };
    let detectors = vec![Detector {
        id: 5,
        measurement_keys: vec!["zz".to_string()],
        coords: Vec::new(),
    }];

    let dem = DetectorErrorModelGenerator::new(circuit, Some(detectors), None)
        .unwrap()
        .generate()
        .unwrap();

    assert_eq!(dem.edges.len(), 1);
    assert_eq!(dem.edges[0].location_id, "mflip");
    assert_eq!(dem.edges[0].event, DemEvent::Bool(true));
    assert_eq!(dem.edges[0].detectors, vec![5]);
}

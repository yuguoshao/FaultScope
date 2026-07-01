use std::collections::HashMap;

use super::*;
use crate::{NoiseLocation, NoiseModel};

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
    let sampler_edges = generator.generate_sampler_edges().unwrap();

    assert_eq!(first, second);
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

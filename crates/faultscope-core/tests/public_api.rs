use std::collections::HashMap;

use faultscope_core::{
    FaultScopeSimulator, Circuit, DemHotspotEstimator, Detector,
    DetectorErrorModelGenerator, LogicalObservable, NoiseLocation, NoiseModel, Operation,
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

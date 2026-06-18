use std::collections::{HashMap, HashSet};

use crate::{
    frame_apply_cx, frame_apply_cz, frame_apply_h, frame_apply_pauli_string, frame_apply_s,
    frame_apply_swap, frame_measurement_flip_bits, sparse_pauli_to_xz, Circuit, ConcreteStabilizer,
    DemEvent, Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable, NoiseLocation,
    NoiseModel, NpError, NpResult, Operation,
};

/// Detector error model generator based on single-error propagation.
///
/// The generator is pure Rust core logic. It can infer detector and observable
/// declarations from circuit operations or use declarations supplied by the
/// caller.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorErrorModelGenerator {
    pub circuit: Circuit,
    pub detectors: Vec<Detector>,
    pub observables: Vec<LogicalObservable>,
}

impl DetectorErrorModelGenerator {
    /// Create a generator for `circuit`.
    ///
    /// If `detectors` or `observables` are `None`, declarations are inferred
    /// from `Operation::Detector` and `Operation::ObservableInclude` entries in
    /// the circuit.
    pub fn new(
        circuit: Circuit,
        detectors: Option<Vec<Detector>>,
        observables: Option<Vec<LogicalObservable>>,
    ) -> NpResult<Self> {
        let detectors = detectors.unwrap_or_else(|| detectors_from_circuit(&circuit));
        let observables = observables.unwrap_or_else(|| observables_from_circuit(&circuit));
        validate_detector_ids(&detectors)?;
        validate_observable_ids(&observables)?;
        for observable in &observables {
            observable.validate()?;
        }
        Ok(Self {
            circuit,
            detectors,
            observables,
        })
    }

    /// Generate a typed detector error model.
    pub fn generate(&self) -> NpResult<DetectorErrorModel> {
        let generated_edges = generate_dem_edges(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.detectors,
            &self.observables,
        )?;
        let noise_locations = self.circuit.noise_locations();
        let edges = generated_edges
            .into_iter()
            .map(|edge| DetectorErrorEdge {
                probability: edge.probability,
                detectors: edge.detectors,
                observables: edge.observables,
                tags: noise_locations
                    .get(&edge.location_id)
                    .map(|location| location.tags.clone())
                    .unwrap_or_default(),
                location_id: edge.location_id,
                event: edge.event,
            })
            .collect();
        Ok(DetectorErrorModel {
            detectors: self.detectors.clone(),
            observables: self.observables.clone(),
            edges,
        })
    }
}

/// Infer detector declarations from detector operations in a circuit.
pub fn detectors_from_circuit(circuit: &Circuit) -> Vec<Detector> {
    let mut detectors = Vec::new();
    for operation in &circuit.operations {
        let Operation::Detector {
            detector_id,
            measurement_keys,
            coords,
        } = operation
        else {
            continue;
        };
        detectors.push(Detector {
            id: detector_id.unwrap_or(detectors.len() as i64),
            measurement_keys: measurement_keys.clone(),
            coords: coords.clone(),
        });
    }
    detectors
}

/// Infer logical observable declarations from observable include operations.
pub fn observables_from_circuit(circuit: &Circuit) -> Vec<LogicalObservable> {
    let mut keys_by_id = HashMap::<i64, Vec<String>>::new();
    for operation in &circuit.operations {
        let Operation::ObservableInclude {
            observable_id,
            measurement_keys,
        } = operation
        else {
            continue;
        };
        keys_by_id
            .entry(*observable_id)
            .or_default()
            .extend(measurement_keys.clone());
    }
    let mut observable_ids = keys_by_id.keys().copied().collect::<Vec<_>>();
    observable_ids.sort_unstable();
    observable_ids
        .into_iter()
        .map(|observable_id| LogicalObservable {
            id: observable_id,
            measurement_keys: keys_by_id.remove(&observable_id).unwrap_or_default(),
            pauli_qubits: Vec::new(),
            pauli: String::new(),
        })
        .collect()
}

fn validate_detector_ids(detectors: &[Detector]) -> NpResult<()> {
    let mut seen = HashSet::new();
    for detector in detectors {
        if !seen.insert(detector.id) {
            return Err(NpError::new("detector ids must be unique"));
        }
    }
    Ok(())
}

fn validate_observable_ids(observables: &[LogicalObservable]) -> NpResult<()> {
    let mut seen = HashSet::new();
    for observable in observables {
        if !seen.insert(observable.id) {
            return Err(NpError::new("logical observable ids must be unique"));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedDemEdge {
    pub probability: f64,
    pub detectors: Vec<i64>,
    pub observables: Vec<i64>,
    pub location_id: String,
    pub event: DemEvent,
}

#[derive(Clone)]
struct NoiseOccurrence {
    op_index: usize,
    location: NoiseLocation,
}

struct DemRunRecord {
    measurements: HashMap<String, bool>,
    x_frame: Vec<u8>,
    z_frame: Vec<u8>,
}

pub fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
) -> NpResult<Vec<GeneratedDemEdge>> {
    let occurrences = collect_noise_occurrences(operations)?;
    let reference = run_dem_with_injection(n_qubits, operations, None, None)?;
    let reference_detectors = evaluate_dem_detectors(&reference, detectors)?;
    let reference_observables = evaluate_dem_observables(&reference, observables)?;
    let mut edges = Vec::new();

    for occurrence in occurrences {
        for (event, probability) in non_identity_events(&occurrence.location)? {
            let injected = run_dem_with_injection(
                n_qubits,
                operations,
                Some(occurrence.op_index),
                Some(&event),
            )?;
            let detector_flips = flipped_ids(
                &reference_detectors,
                &evaluate_dem_detectors(&injected, detectors)?,
            );
            let observable_flips = flipped_ids(
                &reference_observables,
                &evaluate_dem_observables(&injected, observables)?,
            );
            if detector_flips.is_empty() && observable_flips.is_empty() {
                continue;
            }
            edges.push(GeneratedDemEdge {
                probability,
                detectors: detector_flips,
                observables: observable_flips,
                location_id: occurrence.location.id.clone(),
                event,
            });
        }
    }
    Ok(edges)
}

fn collect_noise_occurrences(operations: &[Operation]) -> NpResult<Vec<NoiseOccurrence>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (op_index, operation) in operations.iter().enumerate() {
        match operation {
            Operation::Noise(location) => {
                collect_noise_occurrence(&mut out, &mut seen, op_index, location)?;
            }
            Operation::Measure {
                noise: Some(location),
                ..
            }
            | Operation::MeasurePauli {
                noise: Some(location),
                ..
            } => {
                if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
                    return Err(NpError::new(
                        "DEM generation currently supports MeasurementBitFlip on measurement operations",
                    ));
                }
                collect_noise_occurrence(&mut out, &mut seen, op_index, location)?;
            }
            _ => {}
        }
    }
    Ok(out)
}

fn collect_noise_occurrence(
    out: &mut Vec<NoiseOccurrence>,
    seen: &mut HashSet<String>,
    op_index: usize,
    location: &NoiseLocation,
) -> NpResult<()> {
    if !seen.insert(location.id.clone()) {
        return Err(NpError::new(format!(
            "DetectorErrorModelGenerator requires unique noise location ids; duplicate id {:?}",
            location.id
        )));
    }
    out.push(NoiseOccurrence {
        op_index,
        location: location.clone(),
    });
    Ok(())
}

fn non_identity_events(location: &NoiseLocation) -> NpResult<Vec<(DemEvent, f64)>> {
    match &location.model {
        NoiseModel::BernoulliPauli(pauli) => {
            Ok(vec![(DemEvent::Pauli(pauli.clone()), location.rate)])
        }
        NoiseModel::MeasurementBitFlip => Ok(vec![(DemEvent::Bool(true), location.rate)]),
        NoiseModel::SingleQubitDepolarizing => Ok(["X", "Y", "Z"]
            .iter()
            .map(|event| (DemEvent::Pauli((*event).to_string()), location.rate / 3.0))
            .collect()),
        NoiseModel::TwoQubitDepolarizing => {
            let events = [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY",
                "ZZ",
            ];
            Ok(events
                .iter()
                .map(|event| {
                    (
                        DemEvent::Pauli((*event).to_string()),
                        location.rate / events.len() as f64,
                    )
                })
                .collect())
        }
        NoiseModel::PauliChannel(weights) => {
            let total: f64 = weights.iter().map(|(_, weight)| *weight).sum();
            if total <= 0.0 {
                return Err(NpError::new(
                    "PauliChannel weights must have positive total weight",
                ));
            }
            Ok(weights
                .iter()
                .filter(|(_, weight)| *weight > 0.0)
                .map(|(event, weight)| {
                    (
                        DemEvent::Pauli(event.clone()),
                        location.rate * *weight / total,
                    )
                })
                .collect())
        }
    }
}

fn run_dem_with_injection(
    n_qubits: usize,
    operations: &[Operation],
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
) -> NpResult<DemRunRecord> {
    let mut state = ConcreteStabilizer::zero(n_qubits);
    let mut x_frame = vec![0; n_qubits];
    let mut z_frame = vec![0; n_qubits];
    let mut measurements = HashMap::new();
    for (op_index, operation) in operations.iter().enumerate() {
        let mut context = DemOperationContext {
            op_index,
            injected_op_index,
            injected_event,
            state: &mut state,
            x_frame: &mut x_frame,
            z_frame: &mut z_frame,
            measurements: &mut measurements,
        };
        apply_dem_operation(operation, &mut context)?;
    }
    Ok(DemRunRecord {
        measurements,
        x_frame,
        z_frame,
    })
}

struct DemOperationContext<'a> {
    op_index: usize,
    injected_op_index: Option<usize>,
    injected_event: Option<&'a DemEvent>,
    state: &'a mut ConcreteStabilizer,
    x_frame: &'a mut [u8],
    z_frame: &'a mut [u8],
    measurements: &'a mut HashMap<String, bool>,
}

fn apply_dem_operation(
    operation: &Operation,
    context: &mut DemOperationContext<'_>,
) -> NpResult<()> {
    match operation {
        Operation::H(q) => {
            context.state.apply_h(*q);
            frame_apply_h(context.x_frame, context.z_frame, *q);
        }
        Operation::S(q) => {
            context.state.apply_s(*q);
            frame_apply_s(context.x_frame, context.z_frame, *q);
        }
        Operation::SDag(q) => {
            context.state.apply_s_dag(*q);
            frame_apply_s(context.x_frame, context.z_frame, *q);
        }
        Operation::Cx(control, target) => {
            context.state.apply_cx(*control, *target);
            frame_apply_cx(context.x_frame, context.z_frame, *control, *target);
        }
        Operation::Cz(left, right) => {
            context.state.apply_cz(*left, *right);
            frame_apply_cz(context.x_frame, context.z_frame, *left, *right);
        }
        Operation::Swap(left, right) => {
            context.state.apply_swap(*left, *right);
            frame_apply_swap(context.x_frame, context.z_frame, *left, *right);
        }
        Operation::Pauli { qubits, pauli } => {
            let (x, z) = sparse_pauli_to_xz(context.state.n_qubits(), qubits, pauli)?;
            context.state.apply_pauli_string(&x, &z);
        }
        Operation::Noise(location) => {
            if Some(context.op_index) == context.injected_op_index {
                let event = context.injected_event.ok_or_else(|| {
                    NpError::new("missing injected DEM event for noise operation")
                })?;
                apply_dem_noise_event(
                    location,
                    event,
                    context.state,
                    context.x_frame,
                    context.z_frame,
                )?;
            }
        }
        Operation::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = vec![*qubit];
            let bit = deterministic_dem_measurement(context.state, &qubits, basis, key.as_deref())?;
            let bit = maybe_flip_measurement_bit(
                bit,
                context.op_index,
                context.injected_op_index,
                context.injected_event,
            )?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", context.measurements.len()));
            record_dem_measurement(context.measurements, &key, bit)?;
        }
        Operation::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            let bit = deterministic_dem_measurement(context.state, qubits, pauli, key.as_deref())?;
            let bit = maybe_flip_measurement_bit(
                bit,
                context.op_index,
                context.injected_op_index,
                context.injected_event,
            )?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", context.measurements.len()));
            record_dem_measurement(context.measurements, &key, bit)?;
        }
        Operation::Reset { qubit, key, basis } => {
            let qubits = vec![*qubit];
            if let Some(key) = key {
                let bit = deterministic_dem_measurement(context.state, &qubits, basis, Some(key))?;
                record_dem_measurement(context.measurements, key, bit)?;
            }
            context.state.reset_prepare(*qubit, basis)?;
            context.x_frame[*qubit] = 0;
            context.z_frame[*qubit] = 0;
        }
        Operation::Detector { .. } | Operation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn deterministic_dem_measurement(
    state: &ConcreteStabilizer,
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> NpResult<bool> {
    let (x, z) = sparse_pauli_to_xz(state.n_qubits(), qubits, pauli)?;
    if !state.is_deterministic_pauli(&x, &z) {
        return Err(NpError::new(format!(
            "measurement {:?} is random in the ideal/single-error circuit",
            key.unwrap_or("measure")
        )));
    }
    state.deterministic_measurement_bit(&x, &z)
}

fn maybe_flip_measurement_bit(
    bit: bool,
    op_index: usize,
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
) -> NpResult<bool> {
    if Some(op_index) != injected_op_index {
        return Ok(bit);
    }
    match injected_event {
        Some(DemEvent::Bool(value)) => Ok(bit ^ *value),
        Some(DemEvent::Pauli(_)) => Err(NpError::new(
            "measurement injection requires a boolean event",
        )),
        None => Err(NpError::new(
            "missing injected DEM event for measurement operation",
        )),
    }
}

fn apply_dem_noise_event(
    location: &NoiseLocation,
    event: &DemEvent,
    state: &mut ConcreteStabilizer,
    x_frame: &mut [u8],
    z_frame: &mut [u8],
) -> NpResult<()> {
    match event {
        DemEvent::Bool(_) => {
            if matches!(location.model, NoiseModel::MeasurementBitFlip) {
                Ok(())
            } else {
                Err(NpError::new(
                    "non-measurement DEM noise event must be a Pauli string",
                ))
            }
        }
        DemEvent::Pauli(pauli) => {
            let (x, z) = sparse_pauli_to_xz(state.n_qubits(), &location.qubits, pauli)?;
            state.apply_pauli_string(&x, &z);
            frame_apply_pauli_string(x_frame, z_frame, &location.qubits, pauli)
        }
    }
}

fn evaluate_dem_detectors(
    run: &DemRunRecord,
    detectors: &[Detector],
) -> NpResult<HashMap<i64, bool>> {
    let mut out = HashMap::new();
    for detector in detectors {
        out.insert(
            detector.id,
            dem_measurement_parity(&run.measurements, &detector.measurement_keys)?,
        );
    }
    Ok(out)
}

fn evaluate_dem_observables(
    run: &DemRunRecord,
    observables: &[LogicalObservable],
) -> NpResult<HashMap<i64, bool>> {
    let mut out = HashMap::new();
    for observable in observables {
        let mut value = dem_measurement_parity(&run.measurements, &observable.measurement_keys)?;
        if !observable.pauli.is_empty() {
            value ^= frame_measurement_flip_bits(
                &run.x_frame,
                &run.z_frame,
                &observable.pauli_qubits,
                &observable.pauli,
            )?;
        }
        out.insert(observable.id, value);
    }
    Ok(out)
}

fn dem_measurement_parity(measurements: &HashMap<String, bool>, keys: &[String]) -> NpResult<bool> {
    let mut parity = false;
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| NpError::new(format!("unknown measurement key {key:?}")))?;
        parity ^= *value;
    }
    Ok(parity)
}

fn record_dem_measurement(
    measurements: &mut HashMap<String, bool>,
    key: &str,
    bit: bool,
) -> NpResult<()> {
    if measurements.contains_key(key) {
        return Err(NpError::new(format!("duplicate measurement key {key:?}")));
    }
    measurements.insert(key.to_string(), bit);
    Ok(())
}

fn flipped_ids(reference: &HashMap<i64, bool>, injected: &HashMap<i64, bool>) -> Vec<i64> {
    let mut ids: Vec<i64> = reference.keys().copied().collect();
    ids.sort_unstable();
    ids.into_iter()
        .filter(|id| {
            reference.get(id).copied().unwrap_or(false) ^ injected.get(id).copied().unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

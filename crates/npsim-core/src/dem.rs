use std::collections::{HashMap, HashSet};

use crate::{
    sparse_pauli_to_xz, word_count, Circuit, ConcreteStabilizer, DemEvent, DemSamplerEdge,
    Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable, Mask, NoiseLocation,
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
    event_plan: DemEventPlan,
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
        let event_plan = collect_dem_event_plan(&circuit.operations)?;
        Ok(Self {
            circuit,
            detectors,
            observables,
            event_plan,
        })
    }

    /// Generate a typed detector error model.
    pub fn generate(&self) -> NpResult<DetectorErrorModel> {
        let generated_edges = generate_dem_edges_from_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.detectors,
            &self.observables,
            &self.event_plan,
        )?;
        Ok(DetectorErrorModel {
            detectors: self.detectors.clone(),
            observables: self.observables.clone(),
            edges: generated_edges_to_detector_edges(&self.circuit, generated_edges),
        })
    }

    /// Generate only the edge metadata required by the native DEM sampler.
    pub fn generate_sampler_edges(&self) -> NpResult<Vec<DemSamplerEdge>> {
        let generated_edges = generate_dem_edges_from_plan(
            self.circuit.n_qubits,
            &self.circuit.operations,
            &self.detectors,
            &self.observables,
            &self.event_plan,
        )?;
        Ok(generated_edges_to_sampler_edges(
            &self.circuit,
            generated_edges,
        ))
    }
}

fn generated_edges_to_detector_edges(
    circuit: &Circuit,
    generated_edges: Vec<GeneratedDemEdge>,
) -> Vec<DetectorErrorEdge> {
    let noise_locations = circuit.noise_locations();
    generated_edges
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
        .collect()
}

fn generated_edges_to_sampler_edges(
    circuit: &Circuit,
    generated_edges: Vec<GeneratedDemEdge>,
) -> Vec<DemSamplerEdge> {
    let noise_locations = circuit.noise_locations();
    generated_edges
        .into_iter()
        .map(|edge| DemSamplerEdge {
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
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemEventPlan {
    events: Vec<SensitivityEvent>,
    events_by_op: Vec<Vec<usize>>,
}

pub fn collect_dem_event_plan(operations: &[Operation]) -> NpResult<DemEventPlan> {
    let (events, events_by_op) = collect_sensitivity_events(operations)?;
    Ok(DemEventPlan {
        events,
        events_by_op,
    })
}

pub fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
) -> NpResult<Vec<GeneratedDemEdge>> {
    let event_plan = collect_dem_event_plan(operations)?;
    generate_dem_edges_from_plan(n_qubits, operations, detectors, observables, &event_plan)
}

pub fn generate_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    detectors: &[Detector],
    observables: &[LogicalObservable],
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdge>> {
    let events = &event_plan.events;
    let events_by_op = &event_plan.events_by_op;
    let mut state = DemSensitivityState::new(n_qubits, events.len());
    for (op_index, operation) in operations.iter().enumerate() {
        apply_sensitivity_operation(operation, op_index, events, events_by_op, &mut state)?;
    }
    let detector_sensitivities =
        evaluate_sensitivity_detectors(&state.measurements, detectors, state.event_words)?;
    let observable_sensitivities = evaluate_sensitivity_observables(
        &state.measurements,
        &state.x_frame,
        &state.z_frame,
        observables,
        state.event_words,
    )?;
    Ok(assemble_generated_dem_edges(
        events,
        detector_sensitivities,
        observable_sensitivities,
    ))
}

fn assemble_generated_dem_edges(
    events: &[SensitivityEvent],
    detector_sensitivities: HashMap<i64, Mask>,
    observable_sensitivities: HashMap<i64, Mask>,
) -> Vec<GeneratedDemEdge> {
    let detector_sensitivities = sorted_sensitivities(detector_sensitivities);
    let observable_sensitivities = sorted_sensitivities(observable_sensitivities);
    let mut detector_flips_by_event =
        sensitivity_flips_by_event(&detector_sensitivities, events.len());
    let mut observable_flips_by_event =
        sensitivity_flips_by_event(&observable_sensitivities, events.len());
    let mut edges = Vec::new();

    for (event_index, event) in events.iter().enumerate() {
        let detector_flips = std::mem::take(&mut detector_flips_by_event[event_index]);
        let observable_flips = std::mem::take(&mut observable_flips_by_event[event_index]);
        if detector_flips.is_empty() && observable_flips.is_empty() {
            continue;
        }
        edges.push(GeneratedDemEdge {
            probability: event.probability,
            detectors: detector_flips,
            observables: observable_flips,
            location_id: event.location_id.clone(),
            event: event.event.clone(),
        });
    }
    edges
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

#[derive(Debug, Clone, PartialEq)]
struct SensitivityEvent {
    location_id: String,
    qubits: Vec<usize>,
    event: DemEvent,
    probability: f64,
}

struct DemSensitivityState {
    reference: ConcreteStabilizer,
    x_frame: Vec<Mask>,
    z_frame: Vec<Mask>,
    measurements: HashMap<String, Mask>,
    event_words: usize,
}

impl DemSensitivityState {
    fn new(n_qubits: usize, event_count: usize) -> Self {
        let event_words = word_count(event_count);
        Self {
            reference: ConcreteStabilizer::zero(n_qubits),
            x_frame: vec![Mask::zero(event_words); n_qubits],
            z_frame: vec![Mask::zero(event_words); n_qubits],
            measurements: HashMap::new(),
            event_words,
        }
    }
}

fn collect_sensitivity_events(
    operations: &[Operation],
) -> NpResult<(Vec<SensitivityEvent>, Vec<Vec<usize>>)> {
    let occurrences = collect_noise_occurrences(operations)?;
    let mut events = Vec::new();
    let mut events_by_op = vec![Vec::new(); operations.len()];
    for occurrence in occurrences {
        for (event, probability) in non_identity_events(&occurrence.location)? {
            let event_index = events.len();
            events.push(SensitivityEvent {
                location_id: occurrence.location.id.clone(),
                qubits: occurrence.location.qubits.clone(),
                event,
                probability,
            });
            events_by_op[occurrence.op_index].push(event_index);
        }
    }
    Ok((events, events_by_op))
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

fn apply_sensitivity_operation(
    operation: &Operation,
    op_index: usize,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    state: &mut DemSensitivityState,
) -> NpResult<()> {
    match operation {
        Operation::H(q) => {
            state.reference.apply_h(*q);
            std::mem::swap(&mut state.x_frame[*q], &mut state.z_frame[*q]);
        }
        Operation::S(q) => {
            state.reference.apply_s(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Operation::SDag(q) => {
            state.reference.apply_s_dag(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Operation::Cx(control, target) => {
            state.reference.apply_cx(*control, *target);
            xor_within_frame(&mut state.x_frame, *target, *control);
            xor_within_frame(&mut state.z_frame, *control, *target);
        }
        Operation::Cz(left, right) => {
            state.reference.apply_cz(*left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *right, *left);
        }
        Operation::Swap(left, right) => {
            state.reference.apply_swap(*left, *right);
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        Operation::Pauli { qubits, pauli } => {
            let (x, z) = sparse_pauli_to_xz(state.reference.n_qubits(), qubits, pauli)?;
            state.reference.apply_pauli_string(&x, &z);
        }
        Operation::Noise(_) => {
            apply_sensitivity_events(events, events_by_op, op_index, state)?;
        }
        Operation::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = vec![*qubit];
            ensure_deterministic_dem_measurement(&state.reference, &qubits, basis, key.as_deref())?;
            let mut value = sensitivity_frame_measurement_flip(
                &state.x_frame,
                &state.z_frame,
                &qubits,
                basis,
                state.event_words,
            )?;
            xor_measurement_noise_events(&mut value, events, events_by_op, op_index)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", state.measurements.len()));
            record_sensitivity_measurement(&mut state.measurements, &key, value)?;
        }
        Operation::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            ensure_deterministic_dem_measurement(&state.reference, qubits, pauli, key.as_deref())?;
            let mut value = sensitivity_frame_measurement_flip(
                &state.x_frame,
                &state.z_frame,
                qubits,
                pauli,
                state.event_words,
            )?;
            xor_measurement_noise_events(&mut value, events, events_by_op, op_index)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", state.measurements.len()));
            record_sensitivity_measurement(&mut state.measurements, &key, value)?;
        }
        Operation::Reset { qubit, key, basis } => {
            let qubits = vec![*qubit];
            if let Some(key) = key {
                ensure_deterministic_dem_measurement(&state.reference, &qubits, basis, Some(key))?;
                let value = sensitivity_frame_measurement_flip(
                    &state.x_frame,
                    &state.z_frame,
                    &qubits,
                    basis,
                    state.event_words,
                )?;
                record_sensitivity_measurement(&mut state.measurements, key, value)?;
            }
            state.reference.reset_prepare(*qubit, basis)?;
            state.x_frame[*qubit] = Mask::zero(state.event_words);
            state.z_frame[*qubit] = Mask::zero(state.event_words);
        }
        Operation::Detector { .. } | Operation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn apply_sensitivity_events(
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    op_index: usize,
    state: &mut DemSensitivityState,
) -> NpResult<()> {
    for event_index in &events_by_op[op_index] {
        if let DemEvent::Pauli(pauli) = &events[*event_index].event {
            apply_sensitivity_pauli_string(
                &mut state.x_frame,
                &mut state.z_frame,
                &events[*event_index].qubits,
                pauli,
                *event_index,
            )?;
        }
    }
    Ok(())
}

fn apply_sensitivity_pauli_string(
    x_frame: &mut [Mask],
    z_frame: &mut [Mask],
    qubits: &[usize],
    pauli: &str,
    event_index: usize,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("event Pauli length does not match qubits"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.bytes()) {
        match local {
            b'I' => {}
            b'X' => set_event_bit(&mut x_frame[*qubit], event_index),
            b'Z' => set_event_bit(&mut z_frame[*qubit], event_index),
            b'Y' => {
                set_event_bit(&mut x_frame[*qubit], event_index);
                set_event_bit(&mut z_frame[*qubit], event_index);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    local as char
                )))
            }
        }
    }
    Ok(())
}

fn xor_measurement_noise_events(
    value: &mut Mask,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    op_index: usize,
) -> NpResult<()> {
    for event_index in &events_by_op[op_index] {
        match &events[*event_index].event {
            DemEvent::Bool(true) => set_event_bit(value, *event_index),
            DemEvent::Bool(false) => {}
            DemEvent::Pauli(_) => {
                return Err(NpError::new("measurement noise event must be boolean"));
            }
        }
    }
    Ok(())
}

fn sensitivity_frame_measurement_flip(
    x_frame: &[Mask],
    z_frame: &[Mask],
    qubits: &[usize],
    pauli: &str,
    words: usize,
) -> NpResult<Mask> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    let mut flip = Mask::zero(words);
    let pauli_bytes = pauli.as_bytes();
    if pauli_bytes.iter().all(|local| *local == b'Z') {
        for qubit in qubits {
            flip.xor_assign(&x_frame[*qubit]);
        }
        return Ok(flip);
    }
    if pauli_bytes.iter().all(|local| *local == b'X') {
        for qubit in qubits {
            flip.xor_assign(&z_frame[*qubit]);
        }
        return Ok(flip);
    }
    if pauli_bytes.iter().all(|local| *local == b'Y') {
        for qubit in qubits {
            flip.xor_assign(&x_frame[*qubit]);
            flip.xor_assign(&z_frame[*qubit]);
        }
        return Ok(flip);
    }
    for (qubit, local) in qubits.iter().zip(pauli_bytes) {
        match *local {
            b'I' => {}
            b'X' => flip.xor_assign(&z_frame[*qubit]),
            b'Z' => flip.xor_assign(&x_frame[*qubit]),
            b'Y' => {
                flip.xor_assign(&x_frame[*qubit]);
                flip.xor_assign(&z_frame[*qubit]);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    *local as char
                )))
            }
        }
    }
    Ok(flip)
}

fn ensure_deterministic_dem_measurement(
    state: &ConcreteStabilizer,
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> NpResult<()> {
    if !state.is_deterministic_sparse_pauli(qubits, pauli)? {
        return Err(NpError::new(format!(
            "measurement {:?} is random in the ideal/single-error circuit",
            key.unwrap_or("measure")
        )));
    }
    Ok(())
}

fn evaluate_sensitivity_detectors(
    measurements: &HashMap<String, Mask>,
    detectors: &[Detector],
    words: usize,
) -> NpResult<HashMap<i64, Mask>> {
    let mut out = HashMap::new();
    for detector in detectors {
        out.insert(
            detector.id,
            sensitivity_measurement_parity(measurements, &detector.measurement_keys, words)?,
        );
    }
    Ok(out)
}

fn evaluate_sensitivity_observables(
    measurements: &HashMap<String, Mask>,
    x_frame: &[Mask],
    z_frame: &[Mask],
    observables: &[LogicalObservable],
    words: usize,
) -> NpResult<HashMap<i64, Mask>> {
    let mut out = HashMap::new();
    for observable in observables {
        let mut value =
            sensitivity_measurement_parity(measurements, &observable.measurement_keys, words)?;
        if !observable.pauli.is_empty() {
            let flip = sensitivity_frame_measurement_flip(
                x_frame,
                z_frame,
                &observable.pauli_qubits,
                &observable.pauli,
                words,
            )?;
            value.xor_assign(&flip);
        }
        out.insert(observable.id, value);
    }
    Ok(out)
}

fn sensitivity_measurement_parity(
    measurements: &HashMap<String, Mask>,
    keys: &[String],
    words: usize,
) -> NpResult<Mask> {
    let mut parity = Mask::zero(words);
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| NpError::new(format!("unknown measurement key {key:?}")))?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

fn record_sensitivity_measurement(
    measurements: &mut HashMap<String, Mask>,
    key: &str,
    value: Mask,
) -> NpResult<()> {
    if measurements.contains_key(key) {
        return Err(NpError::new(format!("duplicate measurement key {key:?}")));
    }
    measurements.insert(key.to_string(), value);
    Ok(())
}

fn sorted_sensitivities(sensitivities: HashMap<i64, Mask>) -> Vec<(i64, Mask)> {
    let mut out: Vec<(i64, Mask)> = sensitivities.into_iter().collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

fn sensitivity_flips_by_event(sensitivities: &[(i64, Mask)], event_count: usize) -> Vec<Vec<i64>> {
    let mut out = vec![Vec::new(); event_count];
    for (id, mask) in sensitivities {
        for (word_index, word) in mask.words.iter().enumerate() {
            let mut remaining = *word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let event_index = word_index * 64 + bit;
                if event_index < event_count {
                    out[event_index].push(*id);
                }
                remaining &= remaining - 1;
            }
        }
    }
    out
}

fn xor_within_frame(frame: &mut [Mask], target: usize, source: usize) {
    let source_mask = frame[source].clone();
    frame[target].xor_assign(&source_mask);
}

fn xor_between_frames(source: &[Mask], target: &mut [Mask], source_idx: usize, target_idx: usize) {
    let source_mask = source[source_idx].clone();
    target[target_idx].xor_assign(&source_mask);
}

fn set_event_bit(mask: &mut Mask, event_index: usize) {
    let word_index = event_index / 64;
    let bit_index = event_index % 64;
    if let Some(word) = mask.words.get_mut(word_index) {
        *word |= 1u64 << bit_index;
    }
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
}

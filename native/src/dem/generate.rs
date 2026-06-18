use crate::*;

pub(crate) fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> PyResult<Vec<GeneratedDemEdge>> {
    let (events, events_by_op) = collect_sensitivity_events(operations)?;
    let mut state = DemSensitivityState::new(n_qubits, events.len());
    for (op_index, operation) in operations.iter().enumerate() {
        apply_sensitivity_operation(operation, op_index, &events, &events_by_op, &mut state)?;
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
    let mut edges = Vec::new();

    for (event_index, event) in events.iter().enumerate() {
        let detector_flips = sensitive_ids(&detector_sensitivities, event_index);
        let observable_flips = sensitive_ids(&observable_sensitivities, event_index);
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
    Ok(edges)
}

pub(crate) struct SensitivityEvent {
    pub(crate) location_id: String,
    pub(crate) qubits: Vec<usize>,
    pub(crate) event: DemEvent,
    pub(crate) probability: f64,
}

pub(crate) struct DemSensitivityState {
    pub(crate) reference: ConcreteStabilizer,
    pub(crate) x_frame: Vec<Mask>,
    pub(crate) z_frame: Vec<Mask>,
    pub(crate) measurements: HashMap<String, Mask>,
    pub(crate) event_words: usize,
}

impl DemSensitivityState {
    pub(crate) fn new(n_qubits: usize, event_count: usize) -> Self {
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

pub(crate) fn collect_sensitivity_events(
    operations: &[Op],
) -> PyResult<(Vec<SensitivityEvent>, Vec<Vec<usize>>)> {
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

pub(crate) fn apply_sensitivity_operation(
    operation: &Op,
    op_index: usize,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    state: &mut DemSensitivityState,
) -> PyResult<()> {
    match operation {
        Op::H(q) => {
            state.reference.apply_h(*q);
            std::mem::swap(&mut state.x_frame[*q], &mut state.z_frame[*q]);
        }
        Op::S(q) => {
            state.reference.apply_s(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Op::SDag(q) => {
            state.reference.apply_s_dag(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Op::Cx(control, target) => {
            state.reference.apply_cx(*control, *target);
            xor_within_frame(&mut state.x_frame, *target, *control);
            xor_within_frame(&mut state.z_frame, *control, *target);
        }
        Op::Cz(left, right) => {
            state.reference.apply_cz(*left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *right, *left);
        }
        Op::Swap(left, right) => {
            state.reference.apply_swap(*left, *right);
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        Op::Pauli { qubits, pauli } => {
            let (x, z) = sparse_pauli_to_xz(state.reference.n_qubits(), qubits, pauli)?;
            state.reference.apply_pauli_string(&x, &z);
        }
        Op::Noise(_) => {
            apply_sensitivity_events(events, events_by_op, op_index, state)?;
        }
        Op::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = vec![*qubit];
            deterministic_dem_measurement(&state.reference, &qubits, basis, key.as_deref())?;
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
        Op::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            deterministic_dem_measurement(&state.reference, qubits, pauli, key.as_deref())?;
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
        Op::Reset { qubit, key, basis } => {
            let qubits = vec![*qubit];
            if let Some(key) = key {
                deterministic_dem_measurement(&state.reference, &qubits, basis, Some(key))?;
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
        Op::Detector { .. } | Op::ObservableInclude { .. } => {}
    }
    Ok(())
}

pub(crate) fn apply_sensitivity_events(
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    op_index: usize,
    state: &mut DemSensitivityState,
) -> PyResult<()> {
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

pub(crate) fn apply_sensitivity_pauli_string(
    x_frame: &mut [Mask],
    z_frame: &mut [Mask],
    qubits: &[usize],
    pauli: &str,
    event_index: usize,
) -> PyResult<()> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "event Pauli length does not match qubits",
        ));
    }
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        if x != 0 {
            set_shot_bit(&mut x_frame[*qubit], event_index);
        }
        if z != 0 {
            set_shot_bit(&mut z_frame[*qubit], event_index);
        }
    }
    Ok(())
}

pub(crate) fn xor_measurement_noise_events(
    value: &mut Mask,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    op_index: usize,
) -> PyResult<()> {
    for event_index in &events_by_op[op_index] {
        match &events[*event_index].event {
            DemEvent::Bool(true) => set_shot_bit(value, *event_index),
            DemEvent::Bool(false) => {}
            DemEvent::Pauli(_) => {
                return Err(PyValueError::new_err(
                    "measurement noise event must be boolean",
                ))
            }
        }
    }
    Ok(())
}

pub(crate) fn sensitivity_frame_measurement_flip(
    x_frame: &[Mask],
    z_frame: &[Mask],
    qubits: &[usize],
    pauli: &str,
    words: usize,
) -> PyResult<Mask> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "qubits and pauli must have the same length",
        ));
    }
    let mut flip = Mask::zero(words);
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        if z != 0 {
            flip.xor_assign(&x_frame[*qubit]);
        }
        if x != 0 {
            flip.xor_assign(&z_frame[*qubit]);
        }
    }
    Ok(flip)
}

pub(crate) fn record_sensitivity_measurement(
    measurements: &mut HashMap<String, Mask>,
    key: &str,
    value: Mask,
) -> PyResult<()> {
    if measurements.contains_key(key) {
        return Err(PyValueError::new_err(format!(
            "duplicate measurement key {key:?}"
        )));
    }
    measurements.insert(key.to_string(), value);
    Ok(())
}

pub(crate) fn evaluate_sensitivity_detectors(
    measurements: &HashMap<String, Mask>,
    detectors: &[DemDetectorSpec],
    words: usize,
) -> PyResult<HashMap<i64, Mask>> {
    let mut out = HashMap::new();
    for detector in detectors {
        out.insert(
            detector.id,
            sensitivity_measurement_parity(measurements, &detector.measurement_keys, words)?,
        );
    }
    Ok(out)
}

pub(crate) fn evaluate_sensitivity_observables(
    measurements: &HashMap<String, Mask>,
    x_frame: &[Mask],
    z_frame: &[Mask],
    observables: &[DemObservableSpec],
    words: usize,
) -> PyResult<HashMap<i64, Mask>> {
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

pub(crate) fn sensitivity_measurement_parity(
    measurements: &HashMap<String, Mask>,
    keys: &[String],
    words: usize,
) -> PyResult<Mask> {
    let mut parity = Mask::zero(words);
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

pub(crate) fn sensitive_ids(sensitivities: &HashMap<i64, Mask>, event_index: usize) -> Vec<i64> {
    let mut ids: Vec<i64> = sensitivities.keys().copied().collect();
    ids.sort_unstable();
    ids.into_iter()
        .filter(|id| {
            sensitivities
                .get(id)
                .map(|mask| mask_has_bit(mask, event_index))
                .unwrap_or(false)
        })
        .collect()
}

pub(crate) fn mask_has_bit(mask: &Mask, bit_index: usize) -> bool {
    mask.words
        .get(bit_index / 64)
        .map(|word| (word & (1u64 << (bit_index % 64))) != 0)
        .unwrap_or(false)
}

pub(crate) fn collect_noise_occurrences(operations: &[Op]) -> PyResult<Vec<NoiseOccurrence>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (op_index, operation) in operations.iter().enumerate() {
        match operation {
            Op::Noise(location) => {
                if !seen.insert(location.id.clone()) {
                    return Err(PyValueError::new_err(format!(
                        "DetectorErrorModelGenerator requires unique noise location ids; duplicate id {:?}",
                        location.id
                    )));
                }
                out.push(NoiseOccurrence {
                    op_index,
                    location: location.clone(),
                });
            }
            Op::Measure {
                noise: Some(location),
                ..
            }
            | Op::MeasurePauli {
                noise: Some(location),
                ..
            } => {
                if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
                    return Err(PyValueError::new_err(
                        "DEM generation currently supports MeasurementBitFlip on measurement operations",
                    ));
                }
                if !seen.insert(location.id.clone()) {
                    return Err(PyValueError::new_err(format!(
                        "DetectorErrorModelGenerator requires unique noise location ids; duplicate id {:?}",
                        location.id
                    )));
                }
                out.push(NoiseOccurrence {
                    op_index,
                    location: location.clone(),
                });
            }
            _ => {}
        }
    }
    Ok(out)
}

pub(crate) fn non_identity_events(location: &NoiseLocationSpec) -> PyResult<Vec<(DemEvent, f64)>> {
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
                return Err(PyValueError::new_err(
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

pub(crate) fn deterministic_dem_measurement(
    state: &ConcreteStabilizer,
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> PyResult<bool> {
    let (x, z) = sparse_pauli_to_xz(state.n_qubits(), qubits, pauli)?;
    if !state.is_deterministic_pauli(&x, &z) {
        return Err(PyValueError::new_err(format!(
            "measurement {:?} is random in the ideal/single-error circuit",
            key.unwrap_or("measure")
        )));
    }
    state.deterministic_measurement_bit(&x, &z)
}

pub(crate) fn dem_edges_to_py(py: Python<'_>, edges: &[GeneratedDemEdge]) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for edge in edges {
        let dict = PyDict::new(py);
        dict.set_item("probability", edge.probability)?;
        dict.set_item("detectors", edge.detectors.clone())?;
        dict.set_item("observables", edge.observables.clone())?;
        dict.set_item("location_id", edge.location_id.clone())?;
        match &edge.event {
            DemEvent::Pauli(pauli) => dict.set_item("event", pauli)?,
            DemEvent::Bool(value) => dict.set_item("event", *value)?,
        }
        list.append(dict)?;
    }
    Ok(list.into())
}

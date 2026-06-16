use crate::*;

pub(crate) fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> PyResult<Vec<GeneratedDemEdge>> {
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

pub(crate) fn run_dem_with_injection(
    n_qubits: usize,
    operations: &[Op],
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
) -> PyResult<DemRunRecord> {
    let mut state = ConcreteStabilizer::zero(n_qubits);
    let mut x_frame = vec![0; n_qubits];
    let mut z_frame = vec![0; n_qubits];
    let mut measurements = HashMap::new();
    for (op_index, operation) in operations.iter().enumerate() {
        apply_dem_operation(
            operation,
            op_index,
            injected_op_index,
            injected_event,
            &mut state,
            &mut x_frame,
            &mut z_frame,
            &mut measurements,
        )?;
    }
    Ok(DemRunRecord {
        measurements,
        x_frame,
        z_frame,
    })
}

pub(crate) fn apply_dem_operation(
    operation: &Op,
    op_index: usize,
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
    state: &mut ConcreteStabilizer,
    x_frame: &mut [u8],
    z_frame: &mut [u8],
    measurements: &mut HashMap<String, bool>,
) -> PyResult<()> {
    match operation {
        Op::H(q) => {
            state.apply_h(*q);
            frame_apply_h(x_frame, z_frame, *q);
        }
        Op::S(q) => {
            state.apply_s(*q);
            frame_apply_s(x_frame, z_frame, *q);
        }
        Op::SDag(q) => {
            state.apply_s_dag(*q);
            frame_apply_s(x_frame, z_frame, *q);
        }
        Op::Cx(control, target) => {
            state.apply_cx(*control, *target);
            frame_apply_cx(x_frame, z_frame, *control, *target);
        }
        Op::Cz(left, right) => {
            state.apply_cz(*left, *right);
            frame_apply_cz(x_frame, z_frame, *left, *right);
        }
        Op::Swap(left, right) => {
            state.apply_swap(*left, *right);
            frame_apply_swap(x_frame, z_frame, *left, *right);
        }
        Op::Pauli { qubits, pauli } => {
            let (x, z) = sparse_pauli_to_xz(state.n_qubits(), qubits, pauli)?;
            state.apply_pauli_string(&x, &z);
        }
        Op::Noise(location) => {
            if Some(op_index) == injected_op_index {
                let event = injected_event.ok_or_else(|| {
                    PyValueError::new_err("missing injected DEM event for noise operation")
                })?;
                apply_dem_noise_event(location, event, state, x_frame, z_frame)?;
            }
        }
        Op::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = vec![*qubit];
            let bit = deterministic_dem_measurement(state, &qubits, basis, key.as_deref())?;
            let bit = maybe_flip_measurement_bit(bit, op_index, injected_op_index, injected_event)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", measurements.len()));
            record_dem_measurement(measurements, &key, bit)?;
        }
        Op::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            let bit = deterministic_dem_measurement(state, qubits, pauli, key.as_deref())?;
            let bit = maybe_flip_measurement_bit(bit, op_index, injected_op_index, injected_event)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", measurements.len()));
            record_dem_measurement(measurements, &key, bit)?;
        }
        Op::Reset { qubit, key, basis } => {
            let qubits = vec![*qubit];
            if let Some(key) = key {
                let bit = deterministic_dem_measurement(state, &qubits, basis, Some(key))?;
                record_dem_measurement(measurements, key, bit)?;
            }
            state.reset_prepare(*qubit, basis)?;
            x_frame[*qubit] = 0;
            z_frame[*qubit] = 0;
        }
        Op::Detector { .. } | Op::ObservableInclude { .. } => {}
    }
    Ok(())
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

pub(crate) fn maybe_flip_measurement_bit(
    bit: bool,
    op_index: usize,
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
) -> PyResult<bool> {
    if Some(op_index) != injected_op_index {
        return Ok(bit);
    }
    match injected_event {
        Some(DemEvent::Bool(value)) => Ok(bit ^ *value),
        Some(DemEvent::Pauli(_)) => Err(PyValueError::new_err(
            "measurement injection requires a boolean event",
        )),
        None => Err(PyValueError::new_err(
            "missing injected DEM event for measurement operation",
        )),
    }
}

pub(crate) fn apply_dem_noise_event(
    location: &NoiseLocationSpec,
    event: &DemEvent,
    state: &mut ConcreteStabilizer,
    x_frame: &mut [u8],
    z_frame: &mut [u8],
) -> PyResult<()> {
    match event {
        DemEvent::Bool(_) => {
            if matches!(location.model, NoiseModel::MeasurementBitFlip) {
                Ok(())
            } else {
                Err(PyValueError::new_err(
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

pub(crate) fn evaluate_dem_detectors(
    run: &DemRunRecord,
    detectors: &[DemDetectorSpec],
) -> PyResult<HashMap<i64, bool>> {
    let mut out = HashMap::new();
    for detector in detectors {
        out.insert(
            detector.id,
            dem_measurement_parity(&run.measurements, &detector.measurement_keys)?,
        );
    }
    Ok(out)
}

pub(crate) fn evaluate_dem_observables(
    run: &DemRunRecord,
    observables: &[DemObservableSpec],
) -> PyResult<HashMap<i64, bool>> {
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

pub(crate) fn dem_measurement_parity(
    measurements: &HashMap<String, bool>,
    keys: &[String],
) -> PyResult<bool> {
    let mut parity = false;
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        parity ^= *value;
    }
    Ok(parity)
}

pub(crate) fn record_dem_measurement(
    measurements: &mut HashMap<String, bool>,
    key: &str,
    bit: bool,
) -> PyResult<()> {
    if measurements.contains_key(key) {
        return Err(PyValueError::new_err(format!(
            "duplicate measurement key {key:?}"
        )));
    }
    measurements.insert(key.to_string(), bit);
    Ok(())
}

pub(crate) fn flipped_ids(
    reference: &HashMap<i64, bool>,
    injected: &HashMap<i64, bool>,
) -> Vec<i64> {
    let mut ids: Vec<i64> = reference.keys().copied().collect();
    ids.sort_unstable();
    ids.into_iter()
        .filter(|id| {
            reference.get(id).copied().unwrap_or(false) ^ injected.get(id).copied().unwrap_or(false)
        })
        .collect()
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

use crate::*;

pub(crate) fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> PyResult<Vec<GeneratedDemEdge>> {
    let event_plan = collect_dem_event_plan(operations)?;
    generate_dem_edges_from_plan(n_qubits, operations, detectors, observables, &event_plan)
}

pub(crate) fn generate_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
    event_plan: &DemEventPlan,
) -> PyResult<Vec<GeneratedDemEdge>> {
    if supports_product_reference_fast_path(operations) {
        return generate_product_dem_edges(
            n_qubits,
            operations,
            detectors,
            observables,
            &event_plan.events,
            &event_plan.events_by_op,
        );
    }

    let mut state = DemSensitivityState::new(n_qubits, event_plan.events.len());
    for (op_index, operation) in operations.iter().enumerate() {
        apply_sensitivity_operation(
            operation,
            op_index,
            &event_plan.events,
            &event_plan.events_by_op,
            &mut state,
        )?;
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
        &event_plan.events,
        detector_sensitivities,
        observable_sensitivities,
    ))
}

pub(crate) fn generate_sampling_dem_edge_specs(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> PyResult<Vec<DemEdgeSpec>> {
    let event_plan = collect_dem_event_plan(operations)?;
    generate_sampling_dem_edge_specs_from_plan(
        n_qubits,
        operations,
        detectors,
        observables,
        &event_plan,
    )
}

pub(crate) fn generate_sampling_dem_edge_specs_from_plan(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
    event_plan: &DemEventPlan,
) -> PyResult<Vec<DemEdgeSpec>> {
    let (detector_sensitivities, observable_sensitivities) =
        if supports_product_reference_fast_path(operations) {
            let mut state = ProductSensitivityState::new(n_qubits, event_plan.events.len());
            for (op_index, operation) in operations.iter().enumerate() {
                apply_product_sensitivity_operation(
                    operation,
                    op_index,
                    &event_plan.events,
                    &event_plan.events_by_op,
                    &mut state,
                )?;
            }
            (
                evaluate_sensitivity_detectors(&state.measurements, detectors, state.event_words)?,
                evaluate_sensitivity_observables(
                    &state.measurements,
                    &state.x_frame,
                    &state.z_frame,
                    observables,
                    state.event_words,
                )?,
            )
        } else {
            let mut state = DemSensitivityState::new(n_qubits, event_plan.events.len());
            for (op_index, operation) in operations.iter().enumerate() {
                apply_sensitivity_operation(
                    operation,
                    op_index,
                    &event_plan.events,
                    &event_plan.events_by_op,
                    &mut state,
                )?;
            }
            (
                evaluate_sensitivity_detectors(&state.measurements, detectors, state.event_words)?,
                evaluate_sensitivity_observables(
                    &state.measurements,
                    &state.x_frame,
                    &state.z_frame,
                    observables,
                    state.event_words,
                )?,
            )
        };
    Ok(assemble_sampling_dem_edge_specs(
        &event_plan.events,
        detector_sensitivities,
        observable_sensitivities,
    ))
}

pub(crate) struct SensitivityEvent {
    pub(crate) location_id: String,
    pub(crate) qubits: Vec<usize>,
    pub(crate) event: DemEvent,
    pub(crate) probability: f64,
}

pub(crate) struct DemEventPlan {
    pub(crate) events: Vec<SensitivityEvent>,
    pub(crate) events_by_op: Vec<Vec<usize>>,
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProductAxis {
    X,
    Y,
    Z,
}

impl ProductAxis {
    pub(crate) fn apply_h(self) -> Self {
        match self {
            ProductAxis::X => ProductAxis::Z,
            ProductAxis::Y => ProductAxis::Y,
            ProductAxis::Z => ProductAxis::X,
        }
    }

    pub(crate) fn apply_s_ignoring_sign(self) -> Self {
        match self {
            ProductAxis::X => ProductAxis::Y,
            ProductAxis::Y => ProductAxis::X,
            ProductAxis::Z => ProductAxis::Z,
        }
    }
}

pub(crate) struct ProductSensitivityState {
    pub(crate) basis: Vec<ProductAxis>,
    pub(crate) x_frame: Vec<Mask>,
    pub(crate) z_frame: Vec<Mask>,
    pub(crate) measurements: HashMap<String, Mask>,
    pub(crate) event_words: usize,
}

impl ProductSensitivityState {
    pub(crate) fn new(n_qubits: usize, event_count: usize) -> Self {
        let event_words = word_count(event_count);
        Self {
            basis: vec![ProductAxis::Z; n_qubits],
            x_frame: vec![Mask::zero(event_words); n_qubits],
            z_frame: vec![Mask::zero(event_words); n_qubits],
            measurements: HashMap::new(),
            event_words,
        }
    }
}

pub(crate) struct IndexedProductSensitivityState {
    pub(crate) basis: Vec<ProductAxis>,
    pub(crate) x_frame: Vec<Mask>,
    pub(crate) z_frame: Vec<Mask>,
    pub(crate) measurements: Vec<Mask>,
    pub(crate) measurement_recorded: Vec<bool>,
    pub(crate) event_words: usize,
}

impl IndexedProductSensitivityState {
    pub(crate) fn new(n_qubits: usize, event_count: usize, measurement_count: usize) -> Self {
        let event_words = word_count(event_count);
        Self {
            basis: vec![ProductAxis::Z; n_qubits],
            x_frame: vec![Mask::zero(event_words); n_qubits],
            z_frame: vec![Mask::zero(event_words); n_qubits],
            measurements: vec![Mask::zero(event_words); measurement_count],
            measurement_recorded: vec![false; measurement_count],
            event_words,
        }
    }
}

pub(crate) fn supports_product_reference_fast_path(operations: &[Op]) -> bool {
    let n_qubits = operations
        .iter()
        .flat_map(operation_qubits)
        .max()
        .map(|qubit| qubit + 1)
        .unwrap_or(0);
    let mut basis = vec![ProductAxis::Z; n_qubits];
    for operation in operations {
        match operation {
            Op::H(q) => basis[*q] = basis[*q].apply_h(),
            Op::S(q) | Op::SDag(q) => basis[*q] = basis[*q].apply_s_ignoring_sign(),
            Op::Swap(left, right) => basis.swap(*left, *right),
            Op::Reset {
                qubit,
                basis: reset_basis,
                ..
            } => {
                let Ok(axis) = product_axis_from_pauli_bytes(reset_basis.as_bytes()) else {
                    return false;
                };
                basis[*qubit] = axis;
            }
            Op::Cx(control, target) => {
                if !z_product_pair(&basis, *control, *target) {
                    return false;
                }
            }
            Op::Cz(left, right) => {
                if !z_product_pair(&basis, *left, *right) {
                    return false;
                }
            }
            Op::Pauli { .. }
            | Op::Noise(_)
            | Op::Measure { .. }
            | Op::MeasurePauli { .. }
            | Op::Detector { .. }
            | Op::ObservableInclude { .. } => {}
        }
    }
    true
}

pub(crate) fn operation_qubits(operation: &Op) -> Vec<usize> {
    match operation {
        Op::H(q)
        | Op::S(q)
        | Op::SDag(q)
        | Op::Measure { qubit: q, .. }
        | Op::Reset { qubit: q, .. } => {
            vec![*q]
        }
        Op::Cx(a, b) | Op::Cz(a, b) | Op::Swap(a, b) => vec![*a, *b],
        Op::Pauli { qubits, .. } | Op::MeasurePauli { qubits, .. } => qubits.clone(),
        Op::Noise(location) => location.qubits.clone(),
        Op::Detector { .. } | Op::ObservableInclude { .. } => Vec::new(),
    }
}

pub(crate) fn z_product_pair(basis: &[ProductAxis], left: usize, right: usize) -> bool {
    basis[left] == ProductAxis::Z && basis[right] == ProductAxis::Z
}

pub(crate) fn compile_dem_measurement_plan(
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> PyResult<DemMeasurementPlan> {
    let mut measurement_indices_by_op = vec![None; operations.len()];
    let mut measurement_index_by_key = HashMap::<String, usize>::new();
    let mut measurement_count = 0usize;

    for (op_index, operation) in operations.iter().enumerate() {
        let key = match operation {
            Op::Measure { key, .. } | Op::MeasurePauli { key, .. } => key
                .clone()
                .unwrap_or_else(|| format!("m{measurement_count}")),
            Op::Reset { key: Some(key), .. } => key.clone(),
            _ => continue,
        };
        if measurement_index_by_key.contains_key(&key) {
            return Err(PyValueError::new_err(format!(
                "duplicate measurement key {key:?}"
            )));
        }
        let measurement_index = measurement_count;
        measurement_count += 1;
        measurement_index_by_key.insert(key, measurement_index);
        measurement_indices_by_op[op_index] = Some(measurement_index);
    }

    let mut indexed_detectors = detectors
        .iter()
        .map(|detector| {
            Ok(IndexedDemDetectorSpec {
                id: detector.id,
                measurement_indices: measurement_indices_for_keys(
                    &measurement_index_by_key,
                    &detector.measurement_keys,
                )?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    indexed_detectors.sort_by_key(|detector| detector.id);

    let mut indexed_observables = observables
        .iter()
        .map(|observable| {
            Ok(IndexedDemObservableSpec {
                id: observable.id,
                measurement_indices: measurement_indices_for_keys(
                    &measurement_index_by_key,
                    &observable.measurement_keys,
                )?,
                pauli_qubits: observable.pauli_qubits.clone(),
                pauli: observable.pauli.clone(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    indexed_observables.sort_by_key(|observable| observable.id);

    Ok(DemMeasurementPlan {
        measurement_count,
        measurement_indices_by_op,
        detectors: indexed_detectors,
        observables: indexed_observables,
    })
}

pub(crate) fn measurement_indices_for_keys(
    measurement_index_by_key: &HashMap<String, usize>,
    keys: &[String],
) -> PyResult<Vec<usize>> {
    keys.iter()
        .map(|key| {
            measurement_index_by_key
                .get(key)
                .copied()
                .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))
        })
        .collect()
}

pub(crate) fn generate_product_dem_edges(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
) -> PyResult<Vec<GeneratedDemEdge>> {
    let mut state = ProductSensitivityState::new(n_qubits, events.len());
    for (op_index, operation) in operations.iter().enumerate() {
        apply_product_sensitivity_operation(operation, op_index, events, events_by_op, &mut state)?;
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

pub(crate) fn generate_indexed_product_dem_edges(
    n_qubits: usize,
    operations: &[Op],
    measurement_plan: &DemMeasurementPlan,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
) -> PyResult<Vec<GeneratedDemEdge>> {
    let mut state = IndexedProductSensitivityState::new(
        n_qubits,
        events.len(),
        measurement_plan.measurement_count,
    );
    for (op_index, operation) in operations.iter().enumerate() {
        apply_indexed_product_sensitivity_operation(
            operation,
            op_index,
            events,
            events_by_op,
            measurement_plan,
            &mut state,
        )?;
    }
    let detector_sensitivities = evaluate_indexed_sensitivity_detectors(
        &state.measurements,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_sensitivities = evaluate_indexed_sensitivity_observables(
        &state.measurements,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_generated_dem_edges_from_sensitivities(
        events,
        detector_sensitivities,
        observable_sensitivities,
    ))
}

pub(crate) fn generate_indexed_product_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Op],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> PyResult<Vec<GeneratedDemEdge>> {
    generate_indexed_product_dem_edges(
        n_qubits,
        operations,
        measurement_plan,
        &event_plan.events,
        &event_plan.events_by_op,
    )
}

pub(crate) fn generate_indexed_product_sampling_dem_edge_specs(
    n_qubits: usize,
    operations: &[Op],
    measurement_plan: &DemMeasurementPlan,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
) -> PyResult<Vec<DemEdgeSpec>> {
    let mut state = IndexedProductSensitivityState::new(
        n_qubits,
        events.len(),
        measurement_plan.measurement_count,
    );
    for (op_index, operation) in operations.iter().enumerate() {
        apply_indexed_product_sensitivity_operation(
            operation,
            op_index,
            events,
            events_by_op,
            measurement_plan,
            &mut state,
        )?;
    }
    let detector_sensitivities = evaluate_indexed_sensitivity_detectors(
        &state.measurements,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_sensitivities = evaluate_indexed_sensitivity_observables(
        &state.measurements,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_sampling_dem_edge_specs_from_sensitivities(
        events,
        detector_sensitivities,
        observable_sensitivities,
    ))
}

pub(crate) fn generate_indexed_product_sampling_dem_edge_specs_from_plan(
    n_qubits: usize,
    operations: &[Op],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> PyResult<Vec<DemEdgeSpec>> {
    generate_indexed_product_sampling_dem_edge_specs(
        n_qubits,
        operations,
        measurement_plan,
        &event_plan.events,
        &event_plan.events_by_op,
    )
}

pub(crate) fn apply_product_sensitivity_operation(
    operation: &Op,
    op_index: usize,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    state: &mut ProductSensitivityState,
) -> PyResult<()> {
    match operation {
        Op::H(q) => {
            state.basis[*q] = state.basis[*q].apply_h();
            std::mem::swap(&mut state.x_frame[*q], &mut state.z_frame[*q]);
        }
        Op::S(q) | Op::SDag(q) => {
            state.basis[*q] = state.basis[*q].apply_s_ignoring_sign();
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Op::Swap(left, right) => {
            state.basis.swap(*left, *right);
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        Op::Cx(control, target) => {
            if !z_product_pair(&state.basis, *control, *target) {
                return Err(PyValueError::new_err(
                    "product fast path received a Cx that entangles the reference state",
                ));
            }
            xor_within_frame(&mut state.x_frame, *target, *control);
            xor_within_frame(&mut state.z_frame, *control, *target);
        }
        Op::Cz(left, right) => {
            if !z_product_pair(&state.basis, *left, *right) {
                return Err(PyValueError::new_err(
                    "product fast path received a Cz that entangles the reference state",
                ));
            }
            xor_between_frames(&state.x_frame, &mut state.z_frame, *left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *right, *left);
        }
        Op::Pauli { .. } => {}
        Op::Noise(_) => {
            apply_sensitivity_events_to_frames(
                events,
                events_by_op,
                op_index,
                &mut state.x_frame,
                &mut state.z_frame,
            )?;
        }
        Op::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = [*qubit];
            ensure_product_deterministic_measurement(&state.basis, &qubits, basis, key.as_deref())?;
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
            ensure_product_deterministic_measurement(&state.basis, qubits, pauli, key.as_deref())?;
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
            let qubits = [*qubit];
            if let Some(key) = key {
                ensure_product_deterministic_measurement(&state.basis, &qubits, basis, Some(key))?;
                let value = sensitivity_frame_measurement_flip(
                    &state.x_frame,
                    &state.z_frame,
                    &qubits,
                    basis,
                    state.event_words,
                )?;
                record_sensitivity_measurement(&mut state.measurements, key, value)?;
            }
            state.basis[*qubit] = product_axis_from_pauli_bytes(basis.as_bytes())?;
            state.x_frame[*qubit] = Mask::zero(state.event_words);
            state.z_frame[*qubit] = Mask::zero(state.event_words);
        }
        Op::Detector { .. } | Op::ObservableInclude { .. } => {}
    }
    Ok(())
}

pub(crate) fn apply_indexed_product_sensitivity_operation(
    operation: &Op,
    op_index: usize,
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    measurement_plan: &DemMeasurementPlan,
    state: &mut IndexedProductSensitivityState,
) -> PyResult<()> {
    match operation {
        Op::H(q) => {
            state.basis[*q] = state.basis[*q].apply_h();
            std::mem::swap(&mut state.x_frame[*q], &mut state.z_frame[*q]);
        }
        Op::S(q) | Op::SDag(q) => {
            state.basis[*q] = state.basis[*q].apply_s_ignoring_sign();
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        Op::Swap(left, right) => {
            state.basis.swap(*left, *right);
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        Op::Cx(control, target) => {
            if !z_product_pair(&state.basis, *control, *target) {
                return Err(PyValueError::new_err(
                    "product fast path received a Cx that entangles the reference state",
                ));
            }
            xor_within_frame(&mut state.x_frame, *target, *control);
            xor_within_frame(&mut state.z_frame, *control, *target);
        }
        Op::Cz(left, right) => {
            if !z_product_pair(&state.basis, *left, *right) {
                return Err(PyValueError::new_err(
                    "product fast path received a Cz that entangles the reference state",
                ));
            }
            xor_between_frames(&state.x_frame, &mut state.z_frame, *left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *right, *left);
        }
        Op::Pauli { .. } => {}
        Op::Noise(_) => {
            apply_sensitivity_events_to_frames(
                events,
                events_by_op,
                op_index,
                &mut state.x_frame,
                &mut state.z_frame,
            )?;
        }
        Op::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = [*qubit];
            ensure_product_deterministic_measurement(&state.basis, &qubits, basis, key.as_deref())?;
            let mut value = sensitivity_frame_measurement_flip(
                &state.x_frame,
                &state.z_frame,
                &qubits,
                basis,
                state.event_words,
            )?;
            xor_measurement_noise_events(&mut value, events, events_by_op, op_index)?;
            let measurement_index = indexed_measurement_op(measurement_plan, op_index)?;
            record_indexed_sensitivity_measurement(state, measurement_index, value)?;
        }
        Op::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            ensure_product_deterministic_measurement(&state.basis, qubits, pauli, key.as_deref())?;
            let mut value = sensitivity_frame_measurement_flip(
                &state.x_frame,
                &state.z_frame,
                qubits,
                pauli,
                state.event_words,
            )?;
            xor_measurement_noise_events(&mut value, events, events_by_op, op_index)?;
            let measurement_index = indexed_measurement_op(measurement_plan, op_index)?;
            record_indexed_sensitivity_measurement(state, measurement_index, value)?;
        }
        Op::Reset { qubit, key, basis } => {
            let qubits = [*qubit];
            if key.is_some() {
                ensure_product_deterministic_measurement(
                    &state.basis,
                    &qubits,
                    basis,
                    key.as_deref(),
                )?;
                let value = sensitivity_frame_measurement_flip(
                    &state.x_frame,
                    &state.z_frame,
                    &qubits,
                    basis,
                    state.event_words,
                )?;
                let measurement_index = indexed_measurement_op(measurement_plan, op_index)?;
                record_indexed_sensitivity_measurement(state, measurement_index, value)?;
            }
            state.basis[*qubit] = product_axis_from_pauli_bytes(basis.as_bytes())?;
            state.x_frame[*qubit] = Mask::zero(state.event_words);
            state.z_frame[*qubit] = Mask::zero(state.event_words);
        }
        Op::Detector { .. } | Op::ObservableInclude { .. } => {}
    }
    Ok(())
}

pub(crate) fn indexed_measurement_op(
    measurement_plan: &DemMeasurementPlan,
    op_index: usize,
) -> PyResult<usize> {
    measurement_plan
        .measurement_indices_by_op
        .get(op_index)
        .and_then(|index| *index)
        .ok_or_else(|| PyValueError::new_err("internal DEM measurement plan is missing op index"))
}

pub(crate) fn record_indexed_sensitivity_measurement(
    state: &mut IndexedProductSensitivityState,
    measurement_index: usize,
    value: Mask,
) -> PyResult<()> {
    let Some(slot) = state.measurements.get_mut(measurement_index) else {
        return Err(PyValueError::new_err(format!(
            "internal DEM measurement index {measurement_index} is out of range"
        )));
    };
    let Some(recorded) = state.measurement_recorded.get_mut(measurement_index) else {
        return Err(PyValueError::new_err(format!(
            "internal DEM measurement index {measurement_index} is out of range"
        )));
    };
    if *recorded {
        return Err(PyValueError::new_err(format!(
            "duplicate DEM measurement index {measurement_index}"
        )));
    }
    *slot = value;
    *recorded = true;
    Ok(())
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

pub(crate) fn collect_dem_event_plan(operations: &[Op]) -> PyResult<DemEventPlan> {
    let (events, events_by_op) = collect_sensitivity_events(operations)?;
    Ok(DemEventPlan {
        events,
        events_by_op,
    })
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
        Op::MeasurePauli {
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
        Op::Reset { qubit, key, basis } => {
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
    apply_sensitivity_events_to_frames(
        events,
        events_by_op,
        op_index,
        &mut state.x_frame,
        &mut state.z_frame,
    )
}

pub(crate) fn apply_sensitivity_events_to_frames(
    events: &[SensitivityEvent],
    events_by_op: &[Vec<usize>],
    op_index: usize,
    x_frame: &mut [Mask],
    z_frame: &mut [Mask],
) -> PyResult<()> {
    for event_index in &events_by_op[op_index] {
        if let DemEvent::Pauli(pauli) = &events[*event_index].event {
            apply_sensitivity_pauli_string(
                x_frame,
                z_frame,
                &events[*event_index].qubits,
                pauli,
                *event_index,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn ensure_product_deterministic_measurement(
    basis: &[ProductAxis],
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> PyResult<()> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "qubits and pauli must have the same length",
        ));
    }
    for (qubit, local) in qubits.iter().zip(pauli.as_bytes()) {
        if *local == b'I' {
            continue;
        }
        if product_axis_from_pauli_byte(*local)? != basis[*qubit] {
            return Err(PyValueError::new_err(format!(
                "measurement {:?} is random in the ideal/single-error circuit",
                key.unwrap_or("measure")
            )));
        }
    }
    Ok(())
}

pub(crate) fn product_axis_from_pauli_bytes(pauli: &[u8]) -> PyResult<ProductAxis> {
    if pauli.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "product reset basis must be a single-qubit Pauli, got {:?}",
            String::from_utf8_lossy(pauli)
        )));
    }
    product_axis_from_pauli_byte(pauli[0])
}

pub(crate) fn product_axis_from_pauli_byte(pauli: u8) -> PyResult<ProductAxis> {
    match pauli {
        b'X' => Ok(ProductAxis::X),
        b'Y' => Ok(ProductAxis::Y),
        b'Z' => Ok(ProductAxis::Z),
        _ => Err(PyValueError::new_err(format!(
            "unsupported Pauli {:?}",
            pauli as char
        ))),
    }
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
    for (qubit, local) in qubits.iter().zip(pauli.bytes()) {
        match local {
            b'I' => {}
            b'X' => set_shot_bit(&mut x_frame[*qubit], event_index),
            b'Z' => set_shot_bit(&mut z_frame[*qubit], event_index),
            b'Y' => {
                set_shot_bit(&mut x_frame[*qubit], event_index);
                set_shot_bit(&mut z_frame[*qubit], event_index);
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported Pauli {:?}",
                    local as char
                )))
            }
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
                return Err(PyValueError::new_err(format!(
                    "unsupported Pauli {:?}",
                    *local as char
                )))
            }
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

pub(crate) fn evaluate_indexed_sensitivity_detectors(
    measurements: &[Mask],
    detectors: &[IndexedDemDetectorSpec],
    words: usize,
) -> PyResult<Vec<(i64, Mask)>> {
    let mut out = Vec::with_capacity(detectors.len());
    for detector in detectors {
        out.push((
            detector.id,
            sensitivity_measurement_index_parity(
                measurements,
                &detector.measurement_indices,
                words,
            )?,
        ));
    }
    Ok(out)
}

pub(crate) fn evaluate_indexed_sensitivity_observables(
    measurements: &[Mask],
    x_frame: &[Mask],
    z_frame: &[Mask],
    observables: &[IndexedDemObservableSpec],
    words: usize,
) -> PyResult<Vec<(i64, Mask)>> {
    let mut out = Vec::with_capacity(observables.len());
    for observable in observables {
        let mut value = sensitivity_measurement_index_parity(
            measurements,
            &observable.measurement_indices,
            words,
        )?;
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
        out.push((observable.id, value));
    }
    Ok(out)
}

pub(crate) fn sensitivity_measurement_index_parity(
    measurements: &[Mask],
    indices: &[usize],
    words: usize,
) -> PyResult<Mask> {
    let mut parity = Mask::zero(words);
    for index in indices {
        let value = measurements
            .get(*index)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement index {index}")))?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

pub(crate) fn sorted_sensitivities(sensitivities: HashMap<i64, Mask>) -> Vec<(i64, Mask)> {
    let mut out: Vec<(i64, Mask)> = sensitivities.into_iter().collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

pub(crate) fn assemble_generated_dem_edges(
    events: &[SensitivityEvent],
    detector_sensitivities: HashMap<i64, Mask>,
    observable_sensitivities: HashMap<i64, Mask>,
) -> Vec<GeneratedDemEdge> {
    let detector_sensitivities = sorted_sensitivities(detector_sensitivities);
    let observable_sensitivities = sorted_sensitivities(observable_sensitivities);
    assemble_generated_dem_edges_from_sensitivities(
        events,
        detector_sensitivities,
        observable_sensitivities,
    )
}

pub(crate) fn assemble_generated_dem_edges_from_sensitivities(
    events: &[SensitivityEvent],
    detector_sensitivities: Vec<(i64, Mask)>,
    observable_sensitivities: Vec<(i64, Mask)>,
) -> Vec<GeneratedDemEdge> {
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

pub(crate) fn assemble_sampling_dem_edge_specs(
    events: &[SensitivityEvent],
    detector_sensitivities: HashMap<i64, Mask>,
    observable_sensitivities: HashMap<i64, Mask>,
) -> Vec<DemEdgeSpec> {
    let detector_sensitivities = sorted_sensitivities(detector_sensitivities);
    let observable_sensitivities = sorted_sensitivities(observable_sensitivities);
    assemble_sampling_dem_edge_specs_from_sensitivities(
        events,
        detector_sensitivities,
        observable_sensitivities,
    )
}

pub(crate) fn assemble_sampling_dem_edge_specs_from_sensitivities(
    events: &[SensitivityEvent],
    detector_sensitivities: Vec<(i64, Mask)>,
    observable_sensitivities: Vec<(i64, Mask)>,
) -> Vec<DemEdgeSpec> {
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
        edges.push(DemEdgeSpec {
            probability: event.probability,
            detectors: detector_flips,
            observables: observable_flips,
            location_id: String::new(),
            tags: HashMap::new(),
        });
    }
    edges
}

pub(crate) fn sensitivity_flips_by_event(
    sensitivities: &[(i64, Mask)],
    event_count: usize,
) -> Vec<Vec<i64>> {
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

pub(crate) fn ensure_deterministic_dem_measurement(
    state: &ConcreteStabilizer,
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> PyResult<()> {
    if !state.is_deterministic_sparse_pauli(qubits, pauli)? {
        return Err(PyValueError::new_err(format!(
            "measurement {:?} is random in the ideal/single-error circuit",
            key.unwrap_or("measure")
        )));
    }
    Ok(())
}

pub(crate) fn dem_edges_to_py(py: Python<'_>, edges: &[GeneratedDemEdge]) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for edge in edges {
        let event_obj: PyObject = match &edge.event {
            DemEvent::Pauli(pauli) => pauli.into_py(py),
            DemEvent::Bool(value) => value.into_py(py),
        };
        let row = PyTuple::new(
            py,
            [
                edge.probability.into_py(py),
                edge.detectors.clone().into_py(py),
                edge.observables.clone().into_py(py),
                edge.location_id.clone().into_py(py),
                event_obj,
            ],
        )?;
        list.append(row)?;
    }
    Ok(list.into())
}

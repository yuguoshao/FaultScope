use super::assembly::{
    assemble_dem_edge_refs_from_flat_flip_masks, assemble_generated_dem_edges_from_flat_flip_masks,
    assemble_sampling_edges_from_flat_flip_masks,
};
use super::bitset::{
    flat_range, set_event_bit_in_words, set_flat_event_bit, swap_flat_rows,
    swap_flat_rows_between_frames, xor_between_flat_frames, xor_flat_frame_measurement_flip_into,
    xor_within_flat_frame, zero_flat_row,
};
use super::event_plan::{DemEventPlan, DemFaultEvent};
use super::indexed_parity::{
    evaluate_indexed_detector_flip_masks, evaluate_indexed_observable_flip_masks,
};
use super::measurement_plan::{optional_indexed_measurement_op, DemMeasurementPlan};
use super::{GeneratedDemEdge, GeneratedDemEdgeRef};
use crate::{word_count, DemEvent, DemSamplerEdge, NpError, NpResult, Operation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProductAxis {
    X,
    Y,
    Z,
}

impl ProductAxis {
    fn apply_h(self) -> Self {
        match self {
            Self::X => Self::Z,
            Self::Y => Self::Y,
            Self::Z => Self::X,
        }
    }

    fn apply_s_ignoring_sign(self) -> Self {
        match self {
            Self::X => Self::Y,
            Self::Y => Self::X,
            Self::Z => Self::Z,
        }
    }
}

struct IndexedProductFaultPropagationState {
    basis: Vec<ProductAxis>,
    x_frame: Vec<u64>,
    z_frame: Vec<u64>,
    measurement_flip_words: Vec<u64>,
    measurement_recorded: Vec<bool>,
    event_words: usize,
}

impl IndexedProductFaultPropagationState {
    fn new(n_qubits: usize, event_count: usize, measurement_count: usize) -> Self {
        let event_words = word_count(event_count);
        Self {
            basis: vec![ProductAxis::Z; n_qubits],
            x_frame: vec![0; n_qubits * event_words],
            z_frame: vec![0; n_qubits * event_words],
            measurement_flip_words: vec![0; measurement_count * event_words],
            measurement_recorded: vec![false; measurement_count],
            event_words,
        }
    }
}

pub(super) fn supports_product_reference_fast_path(
    n_qubits: usize,
    operations: &[Operation],
) -> bool {
    let mut basis = vec![ProductAxis::Z; n_qubits];
    for operation in operations {
        match operation {
            Operation::H(q) => {
                let Some(axis) = basis.get_mut(*q) else {
                    return false;
                };
                *axis = axis.apply_h();
            }
            Operation::S(q) | Operation::SDag(q) => {
                let Some(axis) = basis.get_mut(*q) else {
                    return false;
                };
                *axis = axis.apply_s_ignoring_sign();
            }
            Operation::Swap(left, right) => {
                if *left >= basis.len() || *right >= basis.len() {
                    return false;
                }
                basis.swap(*left, *right);
            }
            Operation::Reset {
                qubit,
                basis: reset_basis,
                ..
            } => {
                if *qubit >= basis.len() {
                    return false;
                }
                let Ok(axis) = product_axis_from_pauli_bytes(reset_basis.as_bytes()) else {
                    return false;
                };
                basis[*qubit] = axis;
            }
            Operation::Cx(control, target) => {
                if !z_product_pair(&basis, *control, *target) {
                    return false;
                }
            }
            Operation::Cz(left, right) => {
                if !z_product_pair(&basis, *left, *right) {
                    return false;
                }
            }
            Operation::Pauli { .. }
            | Operation::Noise(_)
            | Operation::Measure { .. }
            | Operation::MeasurePauli { .. }
            | Operation::Detector { .. }
            | Operation::ObservableInclude { .. } => {}
        }
    }
    true
}

pub(super) fn generate_indexed_product_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdge>> {
    let state =
        propagate_indexed_product_state(n_qubits, operations, measurement_plan, event_plan)?;
    let detector_flip_masks = evaluate_indexed_detector_flip_masks(
        &state.measurement_flip_words,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_flip_masks = evaluate_indexed_observable_flip_masks(
        &state.measurement_flip_words,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_generated_dem_edges_from_flat_flip_masks(
        &event_plan.fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

pub(super) fn generate_indexed_product_dem_edge_refs_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdgeRef>> {
    let state =
        propagate_indexed_product_state(n_qubits, operations, measurement_plan, event_plan)?;
    let detector_flip_masks = evaluate_indexed_detector_flip_masks(
        &state.measurement_flip_words,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_flip_masks = evaluate_indexed_observable_flip_masks(
        &state.measurement_flip_words,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_dem_edge_refs_from_flat_flip_masks(
        event_plan.fault_events.len(),
        detector_flip_masks,
        observable_flip_masks,
    ))
}

pub(super) fn generate_indexed_product_sampling_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<DemSamplerEdge>> {
    let state =
        propagate_indexed_product_state(n_qubits, operations, measurement_plan, event_plan)?;
    let detector_flip_masks = evaluate_indexed_detector_flip_masks(
        &state.measurement_flip_words,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_flip_masks = evaluate_indexed_observable_flip_masks(
        &state.measurement_flip_words,
        &state.x_frame,
        &state.z_frame,
        &measurement_plan.observables,
        state.event_words,
    )?;
    Ok(assemble_sampling_edges_from_flat_flip_masks(
        &event_plan.fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn propagate_indexed_product_state(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<IndexedProductFaultPropagationState> {
    let mut state = IndexedProductFaultPropagationState::new(
        n_qubits,
        event_plan.fault_events.len(),
        measurement_plan.measurement_count,
    );
    for (op_index, operation) in operations.iter().enumerate() {
        apply_indexed_product_fault_propagation_operation(
            operation,
            op_index,
            &event_plan.fault_events,
            &event_plan.fault_events_by_op,
            measurement_plan,
            &mut state,
        )?;
    }
    Ok(state)
}

fn z_product_pair(basis: &[ProductAxis], left: usize, right: usize) -> bool {
    matches!(
        (basis.get(left), basis.get(right)),
        (Some(ProductAxis::Z), Some(ProductAxis::Z))
    )
}

fn apply_indexed_product_fault_propagation_operation(
    operation: &Operation,
    op_index: usize,
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    measurement_plan: &DemMeasurementPlan,
    state: &mut IndexedProductFaultPropagationState,
) -> NpResult<()> {
    match operation {
        Operation::H(q) => {
            state.basis[*q] = state.basis[*q].apply_h();
            swap_flat_rows_between_frames(
                &mut state.x_frame,
                &mut state.z_frame,
                *q,
                state.event_words,
            );
        }
        Operation::S(q) | Operation::SDag(q) => {
            state.basis[*q] = state.basis[*q].apply_s_ignoring_sign();
            xor_between_flat_frames(
                &state.x_frame,
                &mut state.z_frame,
                *q,
                *q,
                state.event_words,
            );
        }
        Operation::Swap(left, right) => {
            state.basis.swap(*left, *right);
            swap_flat_rows(&mut state.x_frame, *left, *right, state.event_words);
            swap_flat_rows(&mut state.z_frame, *left, *right, state.event_words);
        }
        Operation::Cx(control, target) => {
            if !z_product_pair(&state.basis, *control, *target) {
                return Err(NpError::new(
                    "product fast path received a Cx that entangles the reference state",
                ));
            }
            xor_within_flat_frame(&mut state.x_frame, *target, *control, state.event_words);
            xor_within_flat_frame(&mut state.z_frame, *control, *target, state.event_words);
        }
        Operation::Cz(left, right) => {
            if !z_product_pair(&state.basis, *left, *right) {
                return Err(NpError::new(
                    "product fast path received a Cz that entangles the reference state",
                ));
            }
            xor_between_flat_frames(
                &state.x_frame,
                &mut state.z_frame,
                *left,
                *right,
                state.event_words,
            );
            xor_between_flat_frames(
                &state.x_frame,
                &mut state.z_frame,
                *right,
                *left,
                state.event_words,
            );
        }
        Operation::Pauli { .. } => {}
        Operation::Noise(_) => {
            apply_fault_events_to_flat_frames(
                fault_events,
                fault_events_by_op,
                op_index,
                &mut state.x_frame,
                &mut state.z_frame,
                state.event_words,
            )?;
        }
        Operation::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = [*qubit];
            ensure_product_deterministic_measurement(&state.basis, &qubits, basis, key.as_deref())?;
            if let Some(measurement_index) =
                optional_indexed_measurement_op(measurement_plan, op_index)
            {
                record_flat_measurement_flip(
                    state,
                    measurement_index,
                    &qubits,
                    basis,
                    fault_events,
                    fault_events_by_op,
                    op_index,
                )?;
            }
        }
        Operation::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            ensure_product_deterministic_measurement(&state.basis, qubits, pauli, key.as_deref())?;
            if let Some(measurement_index) =
                optional_indexed_measurement_op(measurement_plan, op_index)
            {
                record_flat_measurement_flip(
                    state,
                    measurement_index,
                    qubits,
                    pauli,
                    fault_events,
                    fault_events_by_op,
                    op_index,
                )?;
            }
        }
        Operation::Reset { qubit, key, basis } => {
            let qubits = [*qubit];
            if key.is_some() {
                ensure_product_deterministic_measurement(
                    &state.basis,
                    &qubits,
                    basis,
                    key.as_deref(),
                )?;
                if let Some(measurement_index) =
                    optional_indexed_measurement_op(measurement_plan, op_index)
                {
                    record_flat_measurement_flip(
                        state,
                        measurement_index,
                        &qubits,
                        basis,
                        fault_events,
                        fault_events_by_op,
                        op_index,
                    )?;
                }
            }
            state.basis[*qubit] = product_axis_from_pauli_bytes(basis.as_bytes())?;
            zero_flat_row(&mut state.x_frame, *qubit, state.event_words);
            zero_flat_row(&mut state.z_frame, *qubit, state.event_words);
        }
        Operation::Detector { .. } | Operation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn record_flat_measurement_flip(
    state: &mut IndexedProductFaultPropagationState,
    measurement_index: usize,
    qubits: &[usize],
    pauli: &str,
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
) -> NpResult<()> {
    if measurement_index >= state.measurement_recorded.len() {
        return Err(NpError::new(format!(
            "internal DEM measurement index {measurement_index} is out of range"
        )));
    }
    if state.measurement_recorded[measurement_index] {
        return Err(NpError::new(format!(
            "duplicate DEM measurement index {measurement_index}"
        )));
    }
    let range = flat_range(measurement_index, state.event_words);
    state.measurement_flip_words[range.clone()].fill(0);
    xor_flat_frame_measurement_flip_into(
        &mut state.measurement_flip_words[range],
        &state.x_frame,
        &state.z_frame,
        qubits,
        pauli,
        state.event_words,
    )?;
    xor_flat_measurement_noise_events(
        &mut state.measurement_flip_words[flat_range(measurement_index, state.event_words)],
        fault_events,
        fault_events_by_op,
        op_index,
    )?;
    state.measurement_recorded[measurement_index] = true;
    Ok(())
}

fn apply_fault_events_to_flat_frames(
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
    x_frame: &mut [u64],
    z_frame: &mut [u64],
    words: usize,
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        if let DemEvent::Pauli(pauli) = &fault_events[*event_index].event {
            apply_fault_event_pauli_string_to_flat_frames(
                x_frame,
                z_frame,
                words,
                &fault_events[*event_index].qubits,
                pauli,
                *event_index,
            )?;
        }
    }
    Ok(())
}

fn apply_fault_event_pauli_string_to_flat_frames(
    x_frame: &mut [u64],
    z_frame: &mut [u64],
    words: usize,
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
            b'X' => set_flat_event_bit(x_frame, *qubit, words, event_index),
            b'Z' => set_flat_event_bit(z_frame, *qubit, words, event_index),
            b'Y' => {
                set_flat_event_bit(x_frame, *qubit, words, event_index);
                set_flat_event_bit(z_frame, *qubit, words, event_index);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    local as char
                )));
            }
        }
    }
    Ok(())
}

fn xor_flat_measurement_noise_events(
    value: &mut [u64],
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        match &fault_events[*event_index].event {
            DemEvent::Bool(true) => set_event_bit_in_words(value, *event_index),
            DemEvent::Bool(false) => {}
            DemEvent::Pauli(_) => {
                return Err(NpError::new("measurement noise event must be boolean"));
            }
        }
    }
    Ok(())
}

fn ensure_product_deterministic_measurement(
    basis: &[ProductAxis],
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.as_bytes()) {
        if *local == b'I' {
            continue;
        }
        if product_axis_from_pauli_byte(*local)? != basis[*qubit] {
            return Err(NpError::new(format!(
                "measurement {:?} is random in the ideal/single-error circuit",
                key.unwrap_or("measure")
            )));
        }
    }
    Ok(())
}

fn product_axis_from_pauli_bytes(pauli: &[u8]) -> NpResult<ProductAxis> {
    if pauli.len() != 1 {
        return Err(NpError::new(format!(
            "product reset basis must be a single-qubit Pauli, got {:?}",
            String::from_utf8_lossy(pauli)
        )));
    }
    product_axis_from_pauli_byte(pauli[0])
}

fn product_axis_from_pauli_byte(pauli: u8) -> NpResult<ProductAxis> {
    match pauli {
        b'X' => Ok(ProductAxis::X),
        b'Y' => Ok(ProductAxis::Y),
        b'Z' => Ok(ProductAxis::Z),
        _ => Err(NpError::new(format!(
            "unsupported Pauli {:?}",
            pauli as char
        ))),
    }
}

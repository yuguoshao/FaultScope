use super::assembly::DemFlipMasks;
use super::bitset::{
    flat_range, set_event_bit_in_words, set_flat_event_bit, swap_flat_rows,
    swap_flat_rows_between_frames, xor_between_flat_frames, xor_flat_frame_measurement_flip_into,
    xor_within_flat_frame, zero_flat_row,
};
use super::event_plan::DemEventPlan;
use super::indexed_parity::{
    evaluate_indexed_detector_flip_masks, evaluate_indexed_observable_flip_masks,
};
use super::measurement_plan::{optional_indexed_measurement, DemMeasurementPlan};
use crate::program::ExpandedOperation;
use crate::{word_count, DemEvent, NpError, NpResult, PauliBasis};

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
    operations: &[ExpandedOperation],
) -> bool {
    let mut basis = vec![ProductAxis::Z; n_qubits];
    for operation in operations {
        match operation {
            ExpandedOperation::H(q) => {
                let Some(axis) = basis.get_mut(*q) else {
                    return false;
                };
                *axis = axis.apply_h();
            }
            ExpandedOperation::S(q) | ExpandedOperation::SDag(q) => {
                let Some(axis) = basis.get_mut(*q) else {
                    return false;
                };
                *axis = axis.apply_s_ignoring_sign();
            }
            ExpandedOperation::Swap(left, right) => {
                if *left >= basis.len() || *right >= basis.len() {
                    return false;
                }
                basis.swap(*left, *right);
            }
            ExpandedOperation::Reset {
                qubit,
                basis: reset_basis,
                ..
            } => {
                if *qubit >= basis.len() {
                    return false;
                }
                basis[*qubit] = product_axis_from_basis(*reset_basis);
            }
            ExpandedOperation::Cx(control, target) => {
                if !z_product_pair(&basis, *control, *target) {
                    return false;
                }
            }
            ExpandedOperation::Cz(left, right) => {
                if !z_product_pair(&basis, *left, *right) {
                    return false;
                }
            }
            ExpandedOperation::Pauli { .. }
            | ExpandedOperation::Noise(_)
            | ExpandedOperation::MeasureSingle { .. }
            | ExpandedOperation::MeasurePauli { .. }
            | ExpandedOperation::Detector { .. }
            | ExpandedOperation::ObservableInclude { .. } => {}
        }
    }
    true
}

pub(super) fn generate_indexed_product_dem_flip_masks_from_plan(
    n_qubits: usize,
    operations: &[ExpandedOperation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<DemFlipMasks> {
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
    Ok(DemFlipMasks {
        detector_flip_masks,
        observable_flip_masks,
    })
}

fn propagate_indexed_product_state(
    n_qubits: usize,
    operations: &[ExpandedOperation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<IndexedProductFaultPropagationState> {
    let mut state = IndexedProductFaultPropagationState::new(
        n_qubits,
        event_plan.fault_events.len(),
        measurement_plan.measurement_count,
    );
    for operation in operations {
        apply_indexed_product_fault_propagation_operation(
            operation,
            event_plan,
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
    operation: &ExpandedOperation,
    event_plan: &DemEventPlan,
    measurement_plan: &DemMeasurementPlan,
    state: &mut IndexedProductFaultPropagationState,
) -> NpResult<()> {
    match operation {
        ExpandedOperation::H(q) => {
            state.basis[*q] = state.basis[*q].apply_h();
            swap_flat_rows_between_frames(
                &mut state.x_frame,
                &mut state.z_frame,
                *q,
                state.event_words,
            );
        }
        ExpandedOperation::S(q) | ExpandedOperation::SDag(q) => {
            state.basis[*q] = state.basis[*q].apply_s_ignoring_sign();
            xor_between_flat_frames(
                &state.x_frame,
                &mut state.z_frame,
                *q,
                *q,
                state.event_words,
            );
        }
        ExpandedOperation::Swap(left, right) => {
            state.basis.swap(*left, *right);
            swap_flat_rows(&mut state.x_frame, *left, *right, state.event_words);
            swap_flat_rows(&mut state.z_frame, *left, *right, state.event_words);
        }
        ExpandedOperation::Cx(control, target) => {
            if !z_product_pair(&state.basis, *control, *target) {
                return Err(NpError::new(
                    "product fast path received a Cx that entangles the reference state",
                ));
            }
            xor_within_flat_frame(&mut state.x_frame, *target, *control, state.event_words);
            xor_within_flat_frame(&mut state.z_frame, *control, *target, state.event_words);
        }
        ExpandedOperation::Cz(left, right) => {
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
        ExpandedOperation::Pauli { .. } => {}
        ExpandedOperation::Noise(noise_id) => {
            apply_fault_events_to_flat_frames(
                event_plan,
                *noise_id,
                &mut state.x_frame,
                &mut state.z_frame,
                state.event_words,
            )?;
        }
        ExpandedOperation::MeasureSingle {
            qubit,
            measurement_id,
            basis,
            noise,
        } => {
            let qubits = [*qubit];
            let pauli = basis.as_str();
            ensure_product_deterministic_measurement(
                &state.basis,
                &qubits,
                pauli,
                *measurement_id,
            )?;
            if let Some(measurement_index) =
                optional_indexed_measurement(measurement_plan, *measurement_id)
            {
                record_flat_measurement_flip(
                    state,
                    measurement_index,
                    &qubits,
                    pauli,
                    event_plan,
                    *noise,
                )?;
            }
        }
        ExpandedOperation::MeasurePauli {
            qubits,
            pauli,
            measurement_id,
            noise,
        } => {
            ensure_product_deterministic_measurement(&state.basis, qubits, pauli, *measurement_id)?;
            if let Some(measurement_index) =
                optional_indexed_measurement(measurement_plan, *measurement_id)
            {
                record_flat_measurement_flip(
                    state,
                    measurement_index,
                    qubits,
                    pauli,
                    event_plan,
                    *noise,
                )?;
            }
        }
        ExpandedOperation::Reset {
            qubit,
            measurement_id,
            basis,
        } => {
            let qubits = [*qubit];
            let pauli = basis.as_str();
            if let Some(measurement_id) = measurement_id {
                ensure_product_deterministic_measurement(
                    &state.basis,
                    &qubits,
                    pauli,
                    *measurement_id,
                )?;
                if let Some(measurement_index) =
                    optional_indexed_measurement(measurement_plan, *measurement_id)
                {
                    record_flat_measurement_flip(
                        state,
                        measurement_index,
                        &qubits,
                        pauli,
                        event_plan,
                        None,
                    )?;
                }
            }
            state.basis[*qubit] = product_axis_from_basis(*basis);
            zero_flat_row(&mut state.x_frame, *qubit, state.event_words);
            zero_flat_row(&mut state.z_frame, *qubit, state.event_words);
        }
        ExpandedOperation::Detector { .. } | ExpandedOperation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn record_flat_measurement_flip(
    state: &mut IndexedProductFaultPropagationState,
    measurement_index: usize,
    qubits: &[usize],
    pauli: &str,
    event_plan: &DemEventPlan,
    noise_id: Option<usize>,
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
        event_plan,
        noise_id,
    );
    state.measurement_recorded[measurement_index] = true;
    Ok(())
}

fn apply_fault_events_to_flat_frames(
    event_plan: &DemEventPlan,
    noise_id: usize,
    x_frame: &mut [u64],
    z_frame: &mut [u64],
    words: usize,
) -> NpResult<()> {
    let qubits = &event_plan.program.noise_locations[noise_id].qubits;
    for event_index in event_plan.fault_event_range_by_noise[noise_id].clone() {
        if let DemEvent::Pauli(pauli) = &event_plan.fault_events[event_index].event {
            apply_fault_event_pauli_string_to_flat_frames(
                x_frame,
                z_frame,
                words,
                qubits,
                pauli,
                event_index,
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
    event_plan: &DemEventPlan,
    noise_id: Option<usize>,
) {
    for event_index in event_plan.measurement_noise_event_range(noise_id) {
        set_event_bit_in_words(value, event_index);
    }
}

fn ensure_product_deterministic_measurement(
    basis: &[ProductAxis],
    qubits: &[usize],
    pauli: &str,
    measurement_id: usize,
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
                "measurement id {measurement_id} is random in the ideal/single-error circuit"
            )));
        }
    }
    Ok(())
}

fn product_axis_from_basis(basis: PauliBasis) -> ProductAxis {
    match basis {
        PauliBasis::X => ProductAxis::X,
        PauliBasis::Y => ProductAxis::Y,
        PauliBasis::Z => ProductAxis::Z,
    }
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

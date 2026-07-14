use super::assembly::DemFlipMasks;
use super::bitset::{flat_range, xor_word_slices};
use super::event_plan::DemEventPlan;
use super::indexed_parity::{evaluate_indexed_detector_flip_masks, measurement_index_flip_parity};
use super::measurement_plan::{
    optional_indexed_measurement, DemMeasurementPlan, IndexedDemObservable,
};
use crate::program::ExpandedOperation;
use crate::{word_count, ConcreteStabilizer, DemEvent, Mask, NpError, NpResult};

struct DemFaultPropagationState {
    reference: ConcreteStabilizer,
    x_frame: Vec<Mask>,
    z_frame: Vec<Mask>,
    measurement_flip_words: Vec<u64>,
    measurement_recorded: Vec<bool>,
    event_words: usize,
}

impl DemFaultPropagationState {
    fn new(n_qubits: usize, event_count: usize, measurement_count: usize) -> Self {
        let event_words = word_count(event_count);
        Self {
            reference: ConcreteStabilizer::zero(n_qubits),
            x_frame: vec![Mask::zero(event_words); n_qubits],
            z_frame: vec![Mask::zero(event_words); n_qubits],
            measurement_flip_words: vec![0; measurement_count * event_words],
            measurement_recorded: vec![false; measurement_count],
            event_words,
        }
    }
}

pub(super) fn generate_fallback_dem_flip_masks_from_plan(
    n_qubits: usize,
    operations: &[ExpandedOperation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<DemFlipMasks> {
    let state = propagate_fallback_state(n_qubits, operations, measurement_plan, event_plan)?;
    let detector_flip_masks = evaluate_indexed_detector_flip_masks(
        &state.measurement_flip_words,
        &measurement_plan.detectors,
        state.event_words,
    )?;
    let observable_flip_masks = evaluate_indexed_observable_flip_masks_from_masks(
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

fn propagate_fallback_state(
    n_qubits: usize,
    operations: &[ExpandedOperation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<DemFaultPropagationState> {
    let mut state = DemFaultPropagationState::new(
        n_qubits,
        event_plan.fault_events.len(),
        measurement_plan.measurement_count,
    );
    for operation in operations {
        apply_fault_propagation_operation(operation, event_plan, measurement_plan, &mut state)?;
    }
    Ok(state)
}

fn record_mask_measurement_flip(
    state: &mut DemFaultPropagationState,
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
    let mut value = frame_measurement_flip_mask(
        &state.x_frame,
        &state.z_frame,
        qubits,
        pauli,
        state.event_words,
    )?;
    xor_measurement_noise_events(&mut value, event_plan, noise_id);
    let range = flat_range(measurement_index, state.event_words);
    state.measurement_flip_words[range].copy_from_slice(&value.words);
    state.measurement_recorded[measurement_index] = true;
    Ok(())
}

fn apply_fault_propagation_operation(
    operation: &ExpandedOperation,
    event_plan: &DemEventPlan,
    measurement_plan: &DemMeasurementPlan,
    state: &mut DemFaultPropagationState,
) -> NpResult<()> {
    match operation {
        ExpandedOperation::H(q) => {
            state.reference.apply_h(*q);
            std::mem::swap(&mut state.x_frame[*q], &mut state.z_frame[*q]);
        }
        ExpandedOperation::S(q) => {
            state.reference.apply_s(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        ExpandedOperation::SDag(q) => {
            state.reference.apply_s_dag(*q);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        ExpandedOperation::Cx(control, target) => {
            state.reference.apply_cx(*control, *target);
            xor_within_frame(&mut state.x_frame, *target, *control);
            xor_within_frame(&mut state.z_frame, *control, *target);
        }
        ExpandedOperation::Cz(left, right) => {
            state.reference.apply_cz(*left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *right, *left);
        }
        ExpandedOperation::Swap(left, right) => {
            state.reference.apply_swap(*left, *right);
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        ExpandedOperation::Pauli { qubits, pauli } => {
            state.reference.apply_sparse_pauli_string(qubits, pauli)?;
        }
        ExpandedOperation::Noise(noise_id) => {
            apply_fault_events(event_plan, *noise_id, state)?;
        }
        ExpandedOperation::MeasureSingle {
            qubit,
            measurement_id,
            basis,
            noise,
        } => {
            let qubits = [*qubit];
            let pauli = basis.as_str();
            ensure_deterministic_dem_measurement(
                &state.reference,
                &qubits,
                pauli,
                *measurement_id,
            )?;
            if let Some(measurement_index) =
                optional_indexed_measurement(measurement_plan, *measurement_id)
            {
                record_mask_measurement_flip(
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
            ensure_deterministic_dem_measurement(&state.reference, qubits, pauli, *measurement_id)?;
            if let Some(measurement_index) =
                optional_indexed_measurement(measurement_plan, *measurement_id)
            {
                record_mask_measurement_flip(
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
                ensure_deterministic_dem_measurement(
                    &state.reference,
                    &qubits,
                    pauli,
                    *measurement_id,
                )?;
                if let Some(measurement_index) =
                    optional_indexed_measurement(measurement_plan, *measurement_id)
                {
                    record_mask_measurement_flip(
                        state,
                        measurement_index,
                        &qubits,
                        pauli,
                        event_plan,
                        None,
                    )?;
                }
            }
            state.reference.reset_prepare(*qubit, pauli)?;
            state.x_frame[*qubit] = Mask::zero(state.event_words);
            state.z_frame[*qubit] = Mask::zero(state.event_words);
        }
        ExpandedOperation::Detector { .. } | ExpandedOperation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn apply_fault_events(
    event_plan: &DemEventPlan,
    noise_id: usize,
    state: &mut DemFaultPropagationState,
) -> NpResult<()> {
    apply_fault_events_to_frames(event_plan, noise_id, &mut state.x_frame, &mut state.z_frame)
}

fn apply_fault_events_to_frames(
    event_plan: &DemEventPlan,
    noise_id: usize,
    x_frame: &mut [Mask],
    z_frame: &mut [Mask],
) -> NpResult<()> {
    let qubits = &event_plan.program.noise_locations[noise_id].qubits;
    for event_index in event_plan.fault_event_range_by_noise[noise_id].clone() {
        if let DemEvent::Pauli(pauli) = &event_plan.fault_events[event_index].event {
            apply_fault_event_pauli_string(x_frame, z_frame, qubits, pauli, event_index)?;
        }
    }
    Ok(())
}

fn apply_fault_event_pauli_string(
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
                )));
            }
        }
    }
    Ok(())
}

fn xor_measurement_noise_events(
    value: &mut Mask,
    event_plan: &DemEventPlan,
    noise_id: Option<usize>,
) {
    for event_index in event_plan.measurement_noise_event_range(noise_id) {
        set_event_bit(value, event_index);
    }
}

fn frame_measurement_flip_mask(
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
                )));
            }
        }
    }
    Ok(flip)
}

fn ensure_deterministic_dem_measurement(
    state: &ConcreteStabilizer,
    qubits: &[usize],
    pauli: &str,
    measurement_id: usize,
) -> NpResult<()> {
    if !state.is_deterministic_sparse_pauli(qubits, pauli)? {
        return Err(NpError::new(format!(
            "measurement id {measurement_id} is random in the ideal/single-error circuit"
        )));
    }
    Ok(())
}

fn evaluate_indexed_observable_flip_masks_from_masks(
    measurement_flip_words: &[u64],
    x_frame: &[Mask],
    z_frame: &[Mask],
    observables: &[IndexedDemObservable],
    words: usize,
) -> NpResult<Vec<(i64, Vec<u64>)>> {
    let mut out = Vec::with_capacity(observables.len());
    for observable in observables {
        let mut value = measurement_index_flip_parity(
            measurement_flip_words,
            &observable.measurement_indices,
            words,
        )?;
        if !observable.pauli.is_empty() {
            let flip = frame_measurement_flip_mask(
                x_frame,
                z_frame,
                &observable.pauli_qubits,
                &observable.pauli,
                words,
            )?;
            xor_word_slices(&mut value, &flip.words);
        }
        out.push((observable.id, value));
    }
    Ok(out)
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

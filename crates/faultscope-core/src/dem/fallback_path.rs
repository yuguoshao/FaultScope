use super::assembly::{
    assemble_dem_edge_refs_from_flat_flip_masks, assemble_generated_dem_edges_from_flat_flip_masks,
    assemble_sampling_edges_from_flat_flip_masks,
};
use super::bitset::{flat_range, xor_word_slices};
use super::event_plan::{DemEventPlan, DemFaultEvent};
use super::indexed_parity::{evaluate_indexed_detector_flip_masks, measurement_index_flip_parity};
use super::measurement_plan::{
    optional_indexed_measurement_op, DemMeasurementPlan, IndexedDemObservable,
};
use super::{GeneratedDemEdge, GeneratedDemEdgeRef};
use crate::{
    sparse_pauli_to_xz, word_count, ConcreteStabilizer, DemEvent, DemSamplerEdge, Mask, NpError,
    NpResult, Operation,
};

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

pub(super) fn generate_fallback_dem_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdge>> {
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
    Ok(assemble_generated_dem_edges_from_flat_flip_masks(
        &event_plan.fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

pub(super) fn generate_fallback_dem_edge_refs_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<GeneratedDemEdgeRef>> {
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
    Ok(assemble_dem_edge_refs_from_flat_flip_masks(
        event_plan.fault_events.len(),
        detector_flip_masks,
        observable_flip_masks,
    ))
}

pub(super) fn generate_fallback_sampling_edges_from_plan(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<Vec<DemSamplerEdge>> {
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
    Ok(assemble_sampling_edges_from_flat_flip_masks(
        &event_plan.fault_events,
        detector_flip_masks,
        observable_flip_masks,
    ))
}

fn propagate_fallback_state(
    n_qubits: usize,
    operations: &[Operation],
    measurement_plan: &DemMeasurementPlan,
    event_plan: &DemEventPlan,
) -> NpResult<DemFaultPropagationState> {
    let fault_events = &event_plan.fault_events;
    let fault_events_by_op = &event_plan.fault_events_by_op;
    let mut state = DemFaultPropagationState::new(
        n_qubits,
        fault_events.len(),
        measurement_plan.measurement_count,
    );
    for (op_index, operation) in operations.iter().enumerate() {
        apply_fault_propagation_operation(
            operation,
            op_index,
            fault_events,
            fault_events_by_op,
            measurement_plan,
            &mut state,
        )?;
    }
    Ok(state)
}

fn record_mask_measurement_flip(
    state: &mut DemFaultPropagationState,
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
    let mut value = frame_measurement_flip_mask(
        &state.x_frame,
        &state.z_frame,
        qubits,
        pauli,
        state.event_words,
    )?;
    xor_measurement_noise_events(&mut value, fault_events, fault_events_by_op, op_index)?;
    let range = flat_range(measurement_index, state.event_words);
    state.measurement_flip_words[range].copy_from_slice(&value.words);
    state.measurement_recorded[measurement_index] = true;
    Ok(())
}

fn apply_fault_propagation_operation(
    operation: &Operation,
    op_index: usize,
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    measurement_plan: &DemMeasurementPlan,
    state: &mut DemFaultPropagationState,
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
            apply_fault_events(fault_events, fault_events_by_op, op_index, state)?;
        }
        Operation::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = vec![*qubit];
            ensure_deterministic_dem_measurement(&state.reference, &qubits, basis, key.as_deref())?;
            if let Some(measurement_index) =
                optional_indexed_measurement_op(measurement_plan, op_index)
            {
                record_mask_measurement_flip(
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
            ensure_deterministic_dem_measurement(&state.reference, qubits, pauli, key.as_deref())?;
            if let Some(measurement_index) =
                optional_indexed_measurement_op(measurement_plan, op_index)
            {
                record_mask_measurement_flip(
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
            let qubits = vec![*qubit];
            if let Some(key) = key {
                ensure_deterministic_dem_measurement(&state.reference, &qubits, basis, Some(key))?;
                if let Some(measurement_index) =
                    optional_indexed_measurement_op(measurement_plan, op_index)
                {
                    record_mask_measurement_flip(
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
            state.reference.reset_prepare(*qubit, basis)?;
            state.x_frame[*qubit] = Mask::zero(state.event_words);
            state.z_frame[*qubit] = Mask::zero(state.event_words);
        }
        Operation::Detector { .. } | Operation::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn apply_fault_events(
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
    state: &mut DemFaultPropagationState,
) -> NpResult<()> {
    apply_fault_events_to_frames(
        fault_events,
        fault_events_by_op,
        op_index,
        &mut state.x_frame,
        &mut state.z_frame,
    )
}

fn apply_fault_events_to_frames(
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
    x_frame: &mut [Mask],
    z_frame: &mut [Mask],
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        if let DemEvent::Pauli(pauli) = &fault_events[*event_index].event {
            apply_fault_event_pauli_string(
                x_frame,
                z_frame,
                &fault_events[*event_index].qubits,
                pauli,
                *event_index,
            )?;
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
    fault_events: &[DemFaultEvent],
    fault_events_by_op: &[Vec<usize>],
    op_index: usize,
) -> NpResult<()> {
    for event_index in &fault_events_by_op[op_index] {
        match &fault_events[*event_index].event {
            DemEvent::Bool(true) => set_event_bit(value, *event_index),
            DemEvent::Bool(false) => {}
            DemEvent::Pauli(_) => {
                return Err(NpError::new("measurement noise event must be boolean"));
            }
        }
    }
    Ok(())
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

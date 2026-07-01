use super::bitset::{flat_range, xor_flat_frame_measurement_flip_into, xor_word_slices};
use super::measurement_plan::{IndexedDemDetector, IndexedDemObservable};
use crate::{NpError, NpResult};

pub(super) fn evaluate_indexed_detector_flip_masks(
    measurement_flip_words: &[u64],
    detectors: &[IndexedDemDetector],
    words: usize,
) -> NpResult<Vec<(i64, Vec<u64>)>> {
    let mut out = Vec::with_capacity(detectors.len());
    for detector in detectors {
        out.push((
            detector.id,
            measurement_index_flip_parity(
                measurement_flip_words,
                &detector.measurement_indices,
                words,
            )?,
        ));
    }
    Ok(out)
}

pub(super) fn evaluate_indexed_observable_flip_masks(
    measurement_flip_words: &[u64],
    x_frame: &[u64],
    z_frame: &[u64],
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
            xor_flat_frame_measurement_flip_into(
                &mut value,
                x_frame,
                z_frame,
                &observable.pauli_qubits,
                &observable.pauli,
                words,
            )?;
        }
        out.push((observable.id, value));
    }
    Ok(out)
}

pub(super) fn measurement_index_flip_parity(
    measurement_flip_words: &[u64],
    indices: &[usize],
    words: usize,
) -> NpResult<Vec<u64>> {
    let mut parity = vec![0; words];
    for index in indices {
        let range = flat_range(*index, words);
        let value = measurement_flip_words
            .get(range)
            .ok_or_else(|| NpError::new(format!("unknown measurement index {index}")))?;
        xor_word_slices(&mut parity, value);
    }
    Ok(parity)
}

use crate::{NpError, NpResult};

pub(super) fn flat_range(index: usize, words: usize) -> std::ops::Range<usize> {
    let start = index * words;
    start..start + words
}

pub(super) fn xor_word_slices(target: &mut [u64], source: &[u64]) {
    for (left, right) in target.iter_mut().zip(source) {
        *left ^= *right;
    }
}

pub(super) fn xor_within_flat_frame(frame: &mut [u64], target: usize, source: usize, words: usize) {
    let source = frame[flat_range(source, words)].to_vec();
    xor_word_slices(&mut frame[flat_range(target, words)], &source);
}

pub(super) fn xor_between_flat_frames(
    source: &[u64],
    target: &mut [u64],
    source_idx: usize,
    target_idx: usize,
    words: usize,
) {
    xor_word_slices(
        &mut target[flat_range(target_idx, words)],
        &source[flat_range(source_idx, words)],
    );
}

pub(super) fn swap_flat_rows(frame: &mut [u64], left: usize, right: usize, words: usize) {
    for word_index in 0..words {
        frame.swap(left * words + word_index, right * words + word_index);
    }
}

pub(super) fn swap_flat_rows_between_frames(
    left: &mut [u64],
    right: &mut [u64],
    index: usize,
    words: usize,
) {
    for word_index in 0..words {
        std::mem::swap(
            &mut left[index * words + word_index],
            &mut right[index * words + word_index],
        );
    }
}

pub(super) fn zero_flat_row(frame: &mut [u64], index: usize, words: usize) {
    frame[flat_range(index, words)].fill(0);
}

pub(super) fn xor_flat_frame_measurement_flip_into(
    out: &mut [u64],
    x_frame: &[u64],
    z_frame: &[u64],
    qubits: &[usize],
    pauli: &str,
    words: usize,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.as_bytes()) {
        match *local {
            b'I' => {}
            b'X' => xor_word_slices(out, &z_frame[flat_range(*qubit, words)]),
            b'Z' => xor_word_slices(out, &x_frame[flat_range(*qubit, words)]),
            b'Y' => {
                xor_word_slices(out, &x_frame[flat_range(*qubit, words)]);
                xor_word_slices(out, &z_frame[flat_range(*qubit, words)]);
            }
            _ => {
                return Err(NpError::new(format!(
                    "unsupported Pauli {:?}",
                    *local as char
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn set_flat_event_bit(frame: &mut [u64], row: usize, words: usize, event_index: usize) {
    let word_index = event_index / 64;
    if word_index >= words {
        return;
    }
    let bit_index = event_index % 64;
    let offset = row * words + word_index;
    if let Some(word) = frame.get_mut(offset) {
        *word |= 1u64 << bit_index;
    }
}

pub(super) fn set_event_bit_in_words(words: &mut [u64], event_index: usize) {
    let word_index = event_index / 64;
    let bit_index = event_index % 64;
    if let Some(word) = words.get_mut(word_index) {
        *word |= 1u64 << bit_index;
    }
}

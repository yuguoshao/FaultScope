use crate::{pauli_to_xz, word_count, Mask, NpError, NpResult, SmallRng};

pub const TWO_QUBIT_DEPOLARIZING_EVENTS: [(bool, bool, bool, bool); 15] = [
    (false, false, true, false),
    (false, false, true, true),
    (false, false, false, true),
    (true, false, false, false),
    (true, false, true, false),
    (true, false, true, true),
    (true, false, false, true),
    (true, true, false, false),
    (true, true, true, false),
    (true, true, true, true),
    (true, true, false, true),
    (false, true, false, false),
    (false, true, true, false),
    (false, true, true, true),
    (false, true, false, true),
];

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPauliEvent {
    pub x: Vec<bool>,
    pub z: Vec<bool>,
    pub weight: f64,
}

pub fn bernoulli_mask(rng: &mut SmallRng, shots: usize, rate: f64) -> Mask {
    if rate <= 0.0 {
        return Mask::zero(word_count(shots));
    }
    if rate >= 1.0 {
        return Mask::all(shots);
    }
    let mut mask = Mask::zero(word_count(shots));
    for_each_bernoulli_event(rng, shots, rate, |shot, _| set_shot_bit(&mut mask, shot));
    mask
}

pub fn for_each_bernoulli_event<F>(rng: &mut SmallRng, shots: usize, rate: f64, mut visit: F)
where
    F: FnMut(usize, &mut SmallRng),
{
    if rate <= 0.0 {
        return;
    }
    if rate >= 1.0 {
        for shot in 0..shots {
            visit(shot, rng);
        }
        return;
    }
    let log1mp = (-rate).ln_1p();
    let mut shot = 0usize;
    loop {
        let skip = (positive_unit_f64(rng).ln() / log1mp).floor() as usize;
        match shot.checked_add(skip) {
            Some(next) if next < shots => shot = next,
            _ => break,
        }
        visit(shot, rng);
        shot += 1;
        if shot >= shots {
            break;
        }
    }
}

pub fn positive_unit_f64(rng: &mut SmallRng) -> f64 {
    rng.next_f64().max(f64::MIN_POSITIVE)
}

pub fn random_bit_mask(rng: &mut SmallRng, words: usize, shots: usize) -> Mask {
    let mut mask = Mask::zero(words);
    for word in &mut mask.words {
        *word = rng.next_u64();
    }
    mask.clear_unused(shots);
    mask
}

pub fn set_shot_bit(mask: &mut Mask, shot: usize) {
    mask.words[shot / 64] |= 1u64 << (shot % 64);
}

pub fn compile_pauli_channel_events(
    qubits: &[usize],
    weights: &[(String, f64)],
) -> NpResult<Vec<CompiledPauliEvent>> {
    let mut events = Vec::new();
    for (event, weight) in weights {
        if *weight < 0.0 {
            return Err(NpError::new("PauliChannel weights must be non-negative"));
        }
        if *weight == 0.0 {
            continue;
        }
        if event.len() != qubits.len() {
            return Err(NpError::new(
                "PauliChannel event length does not match qubits",
            ));
        }
        let mut x = Vec::with_capacity(event.len());
        let mut z = Vec::with_capacity(event.len());
        for local in event.chars() {
            let (px, pz) = pauli_to_xz(local)?;
            x.push(px != 0);
            z.push(pz != 0);
        }
        events.push(CompiledPauliEvent {
            x,
            z,
            weight: *weight,
        });
    }
    Ok(events)
}

pub fn choose_weighted_event(
    events: &[CompiledPauliEvent],
    total: f64,
    rng: &mut SmallRng,
) -> usize {
    let mut threshold = rng.next_f64() * total;
    for (idx, event) in events.iter().enumerate() {
        if threshold < event.weight {
            return idx;
        }
        threshold -= event.weight;
    }
    events.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_mask_handles_extreme_rates() {
        let mut rng = SmallRng::new(1);

        assert_eq!(bernoulli_mask(&mut rng, 70, 0.0), Mask::zero(2));
        assert_eq!(bernoulli_mask(&mut rng, 70, 1.0), Mask::all(70));
    }

    #[test]
    fn compiles_pauli_channel_events() {
        let events = compile_pauli_channel_events(&[0, 1], &[("XI".to_string(), 2.0)]).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].x, vec![true, false]);
        assert_eq!(events[0].z, vec![false, false]);
        assert_eq!(events[0].weight, 2.0);
    }

    #[test]
    fn rejects_negative_channel_weights() {
        let err = compile_pauli_channel_events(&[0], &[("X".to_string(), -1.0)]).unwrap_err();

        assert!(err.message().contains("non-negative"));
    }
}

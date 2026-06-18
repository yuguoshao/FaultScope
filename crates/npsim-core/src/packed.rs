use std::collections::HashMap;

use crate::{
    bernoulli_mask, choose_weighted_event, compile_pauli_channel_events, for_each_bernoulli_event,
    pauli_to_xz, random_bit_mask, set_shot_bit, word_count, Circuit, HotspotEstimate,
    LogicalObservable, Mask, NoiseLocation, NoiseModel, NpError, NpResult, RunOperation, SmallRng,
    TWO_QUBIT_DEPOLARIZING_EVENTS,
};

/// Compiled forward bit-packed simulator.
///
/// This is the Rust core counterpart of the Python
/// `BatchForwardNoiseAwareSimulator` API. It owns the compiled runtime
/// operations and can run batches without depending on PyO3 or Python callback
/// objects. Custom decoder and loss callback integration remains in the Python
/// binding crate.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchForwardNoiseAwareSimulator {
    pub n_qubits: usize,
    pub runtime_operations: Vec<RunOperation>,
    pub observables: Vec<LogicalObservable>,
    pub noise_location_ids: Vec<String>,
    pub noise_locations: Vec<NoiseLocation>,
}

impl BatchForwardNoiseAwareSimulator {
    /// Compile a typed circuit into the bit-packed forward simulator.
    pub fn new(circuit: Circuit, observables: Vec<LogicalObservable>) -> NpResult<Self> {
        let compiled = crate::compile_runtime_operations(circuit.n_qubits, circuit.operations)?;
        Ok(Self {
            n_qubits: circuit.n_qubits,
            runtime_operations: compiled.runtime_operations,
            observables,
            noise_location_ids: compiled.noise_location_ids,
            noise_locations: compiled.noise_locations,
        })
    }

    /// Run a batch of forward trajectories.
    ///
    /// Set `record_events` when hotspot estimation needs per-location event
    /// masks. `seed=None` uses the core default deterministic seed.
    pub fn run_batch(
        &self,
        shots: usize,
        seed: Option<u64>,
        record_events: bool,
    ) -> NpResult<RuntimeState> {
        if shots == 0 {
            return Err(NpError::new("shots must be positive"));
        }
        run_packed_sample(
            self.n_qubits,
            &self.runtime_operations,
            &self.observables,
            &self.noise_location_ids,
            shots,
            seed,
            record_events,
        )
    }

    /// Estimate hotspot scores from an externally supplied loss mask.
    ///
    /// This method is useful when a decoder or loss function is implemented
    /// outside of the core crate. The returned estimate is fully aggregated in
    /// Rust.
    pub fn estimate_from_loss(
        &self,
        state: &RuntimeState,
        loss_mask: &Mask,
        baseline: Option<f64>,
        top_k: usize,
    ) -> HotspotEstimate {
        crate::compute_packed_estimate(&self.noise_locations, state, loss_mask, baseline, top_k)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeState {
    pub shots: usize,
    pub all_mask: Mask,
    pub x_frame: Vec<Mask>,
    pub z_frame: Vec<Mask>,
    pub measurements: HashMap<String, Mask>,
    pub detectors: HashMap<i64, Mask>,
    pub observables: HashMap<i64, Mask>,
    pub event_masks: HashMap<String, Mask>,
    pub record_events: bool,
}

impl RuntimeState {
    pub fn new(
        n_qubits: usize,
        shots: usize,
        noise_location_ids: &[String],
        record_events: bool,
    ) -> Self {
        let words = word_count(shots);
        let all_mask = Mask::all(shots);
        let mut event_masks = HashMap::new();
        if record_events {
            for location_id in noise_location_ids {
                event_masks.insert(location_id.clone(), Mask::zero(words));
            }
        }
        Self {
            shots,
            all_mask: all_mask.clone(),
            x_frame: vec![Mask::zero(words); n_qubits],
            z_frame: vec![Mask::zero(words); n_qubits],
            measurements: HashMap::new(),
            detectors: HashMap::new(),
            observables: HashMap::new(),
            event_masks,
            record_events,
        }
    }
}

pub fn run_packed_sample(
    n_qubits: usize,
    runtime_operations: &[RunOperation],
    observables: &[LogicalObservable],
    noise_location_ids: &[String],
    shots: usize,
    seed: Option<u64>,
    record_events: bool,
) -> NpResult<RuntimeState> {
    let mut rng = SmallRng::new(seed.unwrap_or(0x4d59_5df4_d0f3_3173));
    let mut state = RuntimeState::new(n_qubits, shots, noise_location_ids, record_events);
    let random_masks = (0..random_source_count(runtime_operations))
        .map(|_| random_bit_mask(&mut rng, state.all_mask.words.len(), state.shots))
        .collect::<Vec<_>>();
    for operation in runtime_operations {
        apply_operation(operation, &mut state, &random_masks, &mut rng)?;
    }
    evaluate_observables(&mut state, observables)?;
    Ok(state)
}

fn random_source_count(runtime_operations: &[RunOperation]) -> usize {
    runtime_operations
        .iter()
        .flat_map(|operation| match operation {
            RunOperation::Measure { ideal, .. } | RunOperation::Reset { ideal, .. } => {
                ideal.terms().iter().copied()
            }
            _ => [].iter().copied(),
        })
        .max()
        .map(|source| source + 1)
        .unwrap_or(0)
}

fn apply_operation(
    op: &RunOperation,
    state: &mut RuntimeState,
    random_masks: &[Mask],
    rng: &mut SmallRng,
) -> NpResult<()> {
    match op {
        RunOperation::H(q) => {
            let old_x = state.x_frame[*q].clone();
            state.x_frame[*q] = state.z_frame[*q].clone();
            state.z_frame[*q] = old_x;
        }
        RunOperation::S(q) | RunOperation::SDag(q) => {
            let x = state.x_frame[*q].clone();
            state.z_frame[*q].xor_assign(&x);
        }
        RunOperation::Cx(control, target) => {
            let x_control = state.x_frame[*control].clone();
            state.x_frame[*target].xor_assign(&x_control);
            let z_target = state.z_frame[*target].clone();
            state.z_frame[*control].xor_assign(&z_target);
        }
        RunOperation::Cz(left, right) => {
            let x_right = state.x_frame[*right].clone();
            let x_left = state.x_frame[*left].clone();
            state.z_frame[*left].xor_assign(&x_right);
            state.z_frame[*right].xor_assign(&x_left);
        }
        RunOperation::Swap(left, right) => {
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        RunOperation::Noise(location) => {
            sample_noise(location, state, rng)?;
        }
        RunOperation::Measure {
            qubits,
            key,
            pauli,
            ideal,
            noise,
        } => {
            let mut bit = ideal.eval(random_masks, state.all_mask.words.len(), &state.all_mask);
            let flip = frame_measurement_flip(state, qubits, pauli)?;
            bit.xor_assign(&flip);
            if let Some(location) = noise {
                let flip = sample_measurement_noise(location, state, rng)?;
                bit.xor_assign(&flip);
            }
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", state.measurements.len()));
            record_measurement(&mut state.measurements, &key, bit)?;
        }
        RunOperation::Reset {
            qubit,
            key,
            basis,
            ideal,
        } => {
            let outcome = ideal.eval(random_masks, state.all_mask.words.len(), &state.all_mask);
            if let Some(key) = key {
                let mut bit = outcome.clone();
                let flip = frame_measurement_flip(state, &[*qubit], basis)?;
                bit.xor_assign(&flip);
                record_measurement(&mut state.measurements, key, bit)?;
            }
            state.x_frame[*qubit] = Mask::zero(state.all_mask.words.len());
            state.z_frame[*qubit] = Mask::zero(state.all_mask.words.len());
        }
        RunOperation::Detector {
            detector_id,
            measurement_keys,
        } => {
            let id = if *detector_id >= 0 {
                *detector_id
            } else {
                state.detectors.len() as i64
            };
            let value = measurement_parity(&state.measurements, measurement_keys)?;
            state.detectors.insert(id, value);
        }
        RunOperation::ObservableInclude {
            observable_id,
            measurement_keys,
        } => {
            let value = measurement_parity(&state.measurements, measurement_keys)?;
            state
                .observables
                .entry(*observable_id)
                .and_modify(|existing| existing.xor_assign(&value))
                .or_insert(value);
        }
    }
    Ok(())
}

fn sample_noise(
    location: &NoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> NpResult<()> {
    match &location.model {
        NoiseModel::MeasurementBitFlip => Err(NpError::new(
            "MeasurementBitFlip must be attached to a measurement operation",
        )),
        NoiseModel::BernoulliPauli(pauli) => sample_fixed_pauli_noise(location, state, rng, pauli),
        NoiseModel::SingleQubitDepolarizing => {
            sample_single_qubit_depolarizing_noise(location, state, rng)
        }
        NoiseModel::TwoQubitDepolarizing => {
            sample_two_qubit_depolarizing_noise(location, state, rng)
        }
        NoiseModel::PauliChannel(weights) => {
            sample_pauli_channel_noise(location, state, rng, weights)
        }
    }
}

fn sample_measurement_noise(
    location: &NoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> NpResult<Mask> {
    if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
        return Err(NpError::new(
            "native measurement noise supports MeasurementBitFlip only",
        ));
    }
    let flip = bernoulli_mask(rng, state.shots, location.rate);
    if state.record_events {
        if let Some(mask) = state.event_masks.get_mut(&location.id) {
            mask.xor_assign(&flip);
        }
    }
    Ok(flip)
}

fn sample_fixed_pauli_noise(
    location: &NoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    pauli: &str,
) -> NpResult<()> {
    let event_mask = bernoulli_mask(rng, state.shots, location.rate);
    if state.record_events {
        record_location_event_mask(state, &location.id, &event_mask);
    }
    apply_masked_pauli_to_frame(state, &location.qubits, pauli, &event_mask)
}

fn sample_single_qubit_depolarizing_noise(
    location: &NoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> NpResult<()> {
    let qubit = one_qubit(&location.qubits, "SingleQubitDepolarizing")?;
    let mut event_mask = event_union_mask(state);
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, rng| {
        if let Some(mask) = &mut event_mask {
            set_shot_bit(mask, shot);
        }
        match rng.next_u64() % 3 {
            0 => xor_frame_shot(state, qubit, true, false, shot),
            1 => xor_frame_shot(state, qubit, true, true, shot),
            _ => xor_frame_shot(state, qubit, false, true, shot),
        }
    });
    if let Some(mask) = event_mask {
        record_location_event_mask(state, &location.id, &mask);
    }
    Ok(())
}

fn sample_two_qubit_depolarizing_noise(
    location: &NoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> NpResult<()> {
    let (left, right) = two_qubits(&location.qubits, "TwoQubitDepolarizing")?;
    let mut event_mask = event_union_mask(state);
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, rng| {
        if let Some(mask) = &mut event_mask {
            set_shot_bit(mask, shot);
        }
        let (left_x, left_z, right_x, right_z) = TWO_QUBIT_DEPOLARIZING_EVENTS
            [(rng.next_u64() % TWO_QUBIT_DEPOLARIZING_EVENTS.len() as u64) as usize];
        xor_frame_shot(state, left, left_x, left_z, shot);
        xor_frame_shot(state, right, right_x, right_z, shot);
    });
    if let Some(mask) = event_mask {
        record_location_event_mask(state, &location.id, &mask);
    }
    Ok(())
}

fn sample_pauli_channel_noise(
    location: &NoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    weights: &[(String, f64)],
) -> NpResult<()> {
    let events = compile_pauli_channel_events(&location.qubits, weights)?;
    let total: f64 = events.iter().map(|event| event.weight).sum();
    if total <= 0.0 {
        return Err(NpError::new(
            "PauliChannel weights must have positive total weight",
        ));
    }
    let mut event_mask = event_union_mask(state);
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, rng| {
        if let Some(mask) = &mut event_mask {
            set_shot_bit(mask, shot);
        }
        let event_index = choose_weighted_event(&events, total, rng);
        for (local_index, qubit) in location.qubits.iter().enumerate() {
            xor_frame_shot(
                state,
                *qubit,
                events[event_index].x[local_index],
                events[event_index].z[local_index],
                shot,
            );
        }
    });
    if let Some(mask) = event_mask {
        record_location_event_mask(state, &location.id, &mask);
    }
    Ok(())
}

fn event_union_mask(state: &RuntimeState) -> Option<Mask> {
    if state.record_events {
        Some(Mask::zero(state.all_mask.words.len()))
    } else {
        None
    }
}

fn record_location_event_mask(state: &mut RuntimeState, location_id: &str, event_mask: &Mask) {
    if state.record_events {
        if let Some(mask) = state.event_masks.get_mut(location_id) {
            mask.or_assign(event_mask);
        }
    }
}

fn xor_frame_shot(state: &mut RuntimeState, qubit: usize, x: bool, z: bool, shot: usize) {
    let word = shot / 64;
    let bit = 1u64 << (shot % 64);
    if x {
        state.x_frame[qubit].words[word] ^= bit;
    }
    if z {
        state.z_frame[qubit].words[word] ^= bit;
    }
}

fn apply_masked_pauli_to_frame(
    state: &mut RuntimeState,
    qubits: &[usize],
    pauli: &str,
    mask: &Mask,
) -> NpResult<()> {
    if mask.is_zero() {
        return Ok(());
    }
    if qubits.len() != pauli.len() {
        return Err(NpError::new("event Pauli length does not match qubits"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        if x != 0 {
            state.x_frame[*qubit].xor_assign(mask);
        }
        if z != 0 {
            state.z_frame[*qubit].xor_assign(mask);
        }
    }
    Ok(())
}

fn frame_measurement_flip(state: &RuntimeState, qubits: &[usize], pauli: &str) -> NpResult<Mask> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    let mut flip = Mask::zero(state.all_mask.words.len());
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        if z != 0 {
            flip.xor_assign(&state.x_frame[*qubit]);
        }
        if x != 0 {
            flip.xor_assign(&state.z_frame[*qubit]);
        }
    }
    Ok(flip)
}

fn record_measurement(
    measurements: &mut HashMap<String, Mask>,
    key: &str,
    bit: Mask,
) -> NpResult<()> {
    if measurements.contains_key(key) {
        return Err(NpError::new(format!("duplicate measurement key {key:?}")));
    }
    measurements.insert(key.to_string(), bit);
    Ok(())
}

fn measurement_parity(measurements: &HashMap<String, Mask>, keys: &[String]) -> NpResult<Mask> {
    let words = measurements
        .values()
        .next()
        .map(|mask| mask.words.len())
        .unwrap_or(1);
    let mut parity = Mask::zero(words);
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| NpError::new(format!("unknown measurement key {key:?}")))?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

fn evaluate_observables(
    state: &mut RuntimeState,
    observables: &[LogicalObservable],
) -> NpResult<()> {
    for observable in observables {
        let mut value = measurement_parity(&state.measurements, &observable.measurement_keys)?;
        if !observable.pauli.is_empty() {
            let flip = frame_measurement_flip(state, &observable.pauli_qubits, &observable.pauli)?;
            value.xor_assign(&flip);
        }
        state.observables.insert(observable.id, value);
    }
    Ok(())
}

fn one_qubit(qubits: &[usize], kind: &str) -> NpResult<usize> {
    if qubits.len() == 1 {
        Ok(qubits[0])
    } else {
        Err(NpError::new(format!("{kind} requires exactly one qubit")))
    }
}

fn two_qubits(qubits: &[usize], kind: &str) -> NpResult<(usize, usize)> {
    if qubits.len() == 2 {
        Ok((qubits[0], qubits[1]))
    } else {
        Err(NpError::new(format!("{kind} requires two qubits")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, Operation};

    #[test]
    fn packed_sample_records_measurement_and_detector() {
        let operations = vec![
            RunOperation::Measure {
                qubits: vec![0],
                pauli: "Z".to_string(),
                key: Some("m0".to_string()),
                ideal: Expr::constant(false),
                noise: None,
            },
            RunOperation::Detector {
                detector_id: 0,
                measurement_keys: vec!["m0".to_string()],
            },
        ];

        let state = run_packed_sample(1, &operations, &[], &[], 5, Some(1), true).unwrap();

        assert_eq!(state.measurements["m0"], Mask::zero(1));
        assert_eq!(state.detectors[&0], Mask::zero(1));
    }

    #[test]
    fn simulator_compiles_and_estimates_loss_mask() {
        let location = NoiseLocation {
            id: "x0".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 1.0,
            qubits: vec![0],
            tags: HashMap::new(),
        };
        let simulator = BatchForwardNoiseAwareSimulator::new(
            Circuit {
                n_qubits: 1,
                operations: vec![
                    Operation::Noise(location),
                    Operation::Measure {
                        qubit: 0,
                        key: Some("m0".to_string()),
                        basis: "Z".to_string(),
                        noise: None,
                    },
                ],
            },
            Vec::new(),
        )
        .unwrap();

        let state = simulator.run_batch(8, Some(1), true).unwrap();
        let estimate = simulator.estimate_from_loss(&state, &state.measurements["m0"], None, 1);

        assert_eq!(state.measurements["m0"], state.all_mask);
        assert_eq!(estimate.mean_loss, 1.0);
        assert_eq!(estimate.top_locations, vec!["x0"]);
    }

    #[test]
    fn simulator_rejects_duplicate_noise_location_ids() {
        let location = NoiseLocation {
            id: "dup".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.1,
            qubits: vec![0],
            tags: HashMap::new(),
        };

        let err = BatchForwardNoiseAwareSimulator::new(
            Circuit {
                n_qubits: 1,
                operations: vec![
                    Operation::Noise(location.clone()),
                    Operation::Noise(location),
                ],
            },
            Vec::new(),
        )
        .unwrap_err();

        assert!(err.message().contains("unique noise location ids"));
    }
}

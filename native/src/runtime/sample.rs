use crate::*;

pub(crate) fn run_packed_sample(
    sampler: &NativePackedSampler,
    shots: usize,
    seed: Option<u64>,
    record_events: bool,
) -> PyResult<RuntimeState> {
    let mut rng = SmallRng::new(seed.unwrap_or(0x4d59_5df4_d0f3_3173));
    let mut state = RuntimeState::new(
        sampler.n_qubits,
        shots,
        &sampler.noise_location_ids,
        record_events,
    );
    let random_masks = (0..sampler.random_source_count)
        .map(|_| random_bit_mask(&mut rng, state.all_mask.words.len(), state.shots))
        .collect::<Vec<_>>();
    for operation in &sampler.runtime_operations {
        apply_operation(operation, &mut state, &random_masks, &mut rng)?;
    }
    evaluate_observables(&mut state, &sampler.observables)?;
    Ok(state)
}

pub(crate) fn apply_operation(
    op: &RunOp,
    state: &mut RuntimeState,
    random_masks: &[Mask],
    rng: &mut SmallRng,
) -> PyResult<()> {
    match op {
        RunOp::H(q) => {
            std::mem::swap(&mut state.x_frame[*q], &mut state.z_frame[*q]);
        }
        RunOp::S(q) | RunOp::SDag(q) => {
            xor_between_frames(&state.x_frame, &mut state.z_frame, *q, *q);
        }
        RunOp::Cx(control, target) => {
            xor_within_frame(&mut state.x_frame, *target, *control);
            xor_within_frame(&mut state.z_frame, *control, *target);
        }
        RunOp::Cz(left, right) => {
            xor_between_frames(&state.x_frame, &mut state.z_frame, *left, *right);
            xor_between_frames(&state.x_frame, &mut state.z_frame, *right, *left);
        }
        RunOp::Swap(left, right) => {
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        RunOp::Noise(location) => {
            sample_noise(location, state, rng)?;
        }
        RunOp::Measure {
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
            record_measurement(state, &key, bit)?;
        }
        RunOp::Reset {
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
                record_measurement(state, key, bit)?;
            }
            state.x_frame[*qubit] = Mask::zero(state.all_mask.words.len());
            state.z_frame[*qubit] = Mask::zero(state.all_mask.words.len());
        }
        RunOp::Detector {
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
        RunOp::ObservableInclude {
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

pub(crate) fn sample_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<()> {
    match &location.model {
        NoiseModel::MeasurementBitFlip => Err(PyValueError::new_err(
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

pub(crate) fn sample_measurement_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<Mask> {
    if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
        return Err(PyValueError::new_err(
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

pub(crate) fn sample_fixed_pauli_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    pauli: &str,
) -> PyResult<()> {
    let event_mask = bernoulli_mask(rng, state.shots, location.rate);
    if state.record_events {
        record_location_event_mask(state, &location.id, &event_mask);
    }
    apply_masked_pauli_to_frame(state, &location.qubits, pauli, &event_mask)
}

pub(crate) fn sample_single_qubit_depolarizing_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<()> {
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

pub(crate) fn sample_two_qubit_depolarizing_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<()> {
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

pub(crate) fn sample_pauli_channel_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    weights: &[(String, f64)],
) -> PyResult<()> {
    let events = compile_pauli_channel_events(&location.qubits, weights)?;
    let total: f64 = events.iter().map(|event| event.weight).sum();
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "PauliChannel weights must have positive total weight",
        ));
    }
    let mut event_mask = event_union_mask(state);
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, rng| {
        if let Some(mask) = &mut event_mask {
            set_shot_bit(mask, shot);
        }
        let event_index = choose_weighted_event(&events, total, rng);
        apply_compiled_pauli_event(state, &location.qubits, &events[event_index], shot);
    });
    if let Some(mask) = event_mask {
        record_location_event_mask(state, &location.id, &mask);
    }
    Ok(())
}

pub(crate) fn bernoulli_mask(rng: &mut SmallRng, shots: usize, rate: f64) -> Mask {
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

pub(crate) const TWO_QUBIT_DEPOLARIZING_EVENTS: [(bool, bool, bool, bool); 15] = [
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

pub(crate) struct CompiledPauliEvent {
    pub(crate) x: Vec<bool>,
    pub(crate) z: Vec<bool>,
    pub(crate) weight: f64,
}

pub(crate) fn for_each_bernoulli_event<F>(rng: &mut SmallRng, shots: usize, rate: f64, mut visit: F)
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

pub(crate) fn positive_unit_f64(rng: &mut SmallRng) -> f64 {
    rng.next_f64().max(f64::MIN_POSITIVE)
}

pub(crate) fn event_union_mask(state: &RuntimeState) -> Option<Mask> {
    if state.record_events {
        Some(Mask::zero(state.all_mask.words.len()))
    } else {
        None
    }
}

pub(crate) fn record_location_event_mask(
    state: &mut RuntimeState,
    location_id: &str,
    event_mask: &Mask,
) {
    if state.record_events {
        if let Some(mask) = state.event_masks.get_mut(location_id) {
            mask.or_assign(event_mask);
        }
    }
}

pub(crate) fn xor_frame_shot(
    state: &mut RuntimeState,
    qubit: usize,
    x: bool,
    z: bool,
    shot: usize,
) {
    let word = shot / 64;
    let bit = 1u64 << (shot % 64);
    if x {
        state.x_frame[qubit].words[word] ^= bit;
    }
    if z {
        state.z_frame[qubit].words[word] ^= bit;
    }
}

pub(crate) fn compile_pauli_channel_events(
    qubits: &[usize],
    weights: &[(String, f64)],
) -> PyResult<Vec<CompiledPauliEvent>> {
    let mut events = Vec::new();
    for (event, weight) in weights {
        if *weight < 0.0 {
            return Err(PyValueError::new_err(
                "PauliChannel weights must be non-negative",
            ));
        }
        if *weight == 0.0 {
            continue;
        }
        if event.len() != qubits.len() {
            return Err(PyValueError::new_err(
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

pub(crate) fn choose_weighted_event(
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

pub(crate) fn apply_compiled_pauli_event(
    state: &mut RuntimeState,
    qubits: &[usize],
    event: &CompiledPauliEvent,
    shot: usize,
) {
    for (local_index, qubit) in qubits.iter().enumerate() {
        xor_frame_shot(
            state,
            *qubit,
            event.x[local_index],
            event.z[local_index],
            shot,
        );
    }
}

pub(crate) fn random_bit_mask(rng: &mut SmallRng, words: usize, shots: usize) -> Mask {
    let mut mask = Mask::zero(words);
    for word in &mut mask.words {
        *word = rng.next_u64();
    }
    mask.clear_unused(shots);
    mask
}

pub(crate) fn set_shot_bit(mask: &mut Mask, shot: usize) {
    mask.words[shot / 64] |= 1u64 << (shot % 64);
}

pub(crate) fn apply_masked_pauli_to_frame(
    state: &mut RuntimeState,
    qubits: &[usize],
    pauli: &str,
    mask: &Mask,
) -> PyResult<()> {
    if mask.is_zero() {
        return Ok(());
    }
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "event Pauli length does not match qubits",
        ));
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

pub(crate) fn frame_measurement_flip(
    state: &RuntimeState,
    qubits: &[usize],
    pauli: &str,
) -> PyResult<Mask> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "qubits and pauli must have the same length",
        ));
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

pub(crate) fn record_measurement(state: &mut RuntimeState, key: &str, bit: Mask) -> PyResult<()> {
    if state.measurements.contains_key(key) {
        return Err(PyValueError::new_err(format!(
            "duplicate measurement key {key:?}"
        )));
    }
    state.measurement_order.push(key.to_string());
    state.measurements.insert(key.to_string(), bit);
    Ok(())
}

pub(crate) fn xor_within_frame(frame: &mut [Mask], dest: usize, src: usize) {
    if dest == src {
        let source = frame[src].clone();
        frame[dest].xor_assign(&source);
        return;
    }
    if dest < src {
        let (left, right) = frame.split_at_mut(src);
        left[dest].xor_assign(&right[0]);
    } else {
        let (left, right) = frame.split_at_mut(dest);
        right[0].xor_assign(&left[src]);
    }
}

pub(crate) fn xor_between_frames(
    src_frame: &[Mask],
    dest_frame: &mut [Mask],
    dest: usize,
    src: usize,
) {
    dest_frame[dest].xor_assign(&src_frame[src]);
}

pub(crate) fn measurement_parity(
    measurements: &HashMap<String, Mask>,
    keys: &[String],
) -> PyResult<Mask> {
    let words = measurements
        .values()
        .next()
        .map(|mask| mask.words.len())
        .unwrap_or(1);
    let mut parity = Mask::zero(words);
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

pub(crate) fn evaluate_observables(
    state: &mut RuntimeState,
    observables: &[DemObservableSpec],
) -> PyResult<()> {
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

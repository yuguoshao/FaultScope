use std::collections::HashMap;
use std::sync::Arc;

use crate::pauli::pauli_to_xz;
use crate::{
    bernoulli_mask, choose_weighted_event, compile_pauli_channel_events, for_each_bernoulli_event,
    set_shot_bit, word_count, Circuit, HotspotEstimate, IndexedNoiseLocation, LocationCatalog,
    LogicalObservable, Mask, NoiseModel, NpError, NpResult, SmallRng,
    TWO_QUBIT_DEPOLARIZING_EVENTS,
};

/// Integer-indexed program used by the native forward sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerProgram {
    pub n_qubits: usize,
    pub operations: Vec<SamplerOperation>,
    pub observables: Vec<LogicalObservable>,
    pub compiled_observables: Vec<SamplerObservable>,
    pub measurement_keys: Vec<String>,
    pub(crate) noise_locations: Vec<IndexedNoiseLocation>,
    pub(crate) location_catalog: LocationCatalog,
    pub capacities: RuntimeCapacities,
    pub stored_operation_count: usize,
    pub logical_operation_count: usize,
    pub loop_kernel_count: usize,
    pub(crate) hotspot_layout_identity: Arc<()>,
}

impl SamplerProgram {
    pub fn noise_locations(&self) -> &[IndexedNoiseLocation] {
        &self.noise_locations
    }

    pub fn location_catalog(&self) -> &LocationCatalog {
        &self.location_catalog
    }

    fn validate_hotspot_state(&self, state: &RuntimeState) -> NpResult<()> {
        let state_layout = state.hotspot_layout_identity.as_ref().ok_or_else(|| {
            NpError::new("hotspot state is not bound to a compiled sampler program")
        })?;
        if !Arc::ptr_eq(state_layout, &self.hotspot_layout_identity) {
            return Err(NpError::new(
                "hotspot state layout does not match the current simulator",
            ));
        }
        if state.shots == 0 {
            return Err(NpError::new(
                "forward hotspot estimation requires shots to be positive",
            ));
        }
        if !state.record_events {
            return Err(NpError::new(
                "forward hotspot estimation requires recorded event masks",
            ));
        }
        Ok(())
    }

    /// Estimate hotspots from a state produced by this program or one of its clones.
    pub fn estimate_from_loss(
        &self,
        state: &RuntimeState,
        loss_mask: &Mask,
        baseline: Option<f64>,
        top_k: usize,
    ) -> NpResult<HotspotEstimate> {
        self.validate_hotspot_state(state)?;
        crate::hotspot::validate_loss_mask_width("forward hotspot", state.shots, loss_mask)?;
        Ok(crate::hotspot::compute_packed_estimate_trusted(
            &self.noise_locations,
            &self.location_catalog,
            state,
            loss_mask,
            baseline,
            top_k,
        ))
    }
}

/// Integer-indexed logical observable compiled against a sampler program.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerObservable {
    pub id: i64,
    pub measurement_ids: Vec<usize>,
    pub pauli_qubits: Vec<usize>,
    pub pauli: String,
}

/// Single-qubit Pauli measurement basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliBasis {
    X,
    Y,
    Z,
}

impl PauliBasis {
    pub(crate) fn parse(value: &str) -> NpResult<Self> {
        match value {
            "X" => Ok(Self::X),
            "Y" => Ok(Self::Y),
            "Z" => Ok(Self::Z),
            _ => Err(NpError::new(format!(
                "unsupported measurement basis {value:?}"
            ))),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }
}

/// Executable instruction in a compact sampler program.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplerOperation {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Noise(usize),
    MeasureSingle {
        qubit: usize,
        basis: PauliBasis,
        measurement_id: usize,
        ideal: crate::Expr,
        noise: Option<usize>,
    },
    MeasurePauli {
        qubits: Vec<usize>,
        pauli: String,
        measurement_id: usize,
        ideal: crate::Expr,
        noise: Option<usize>,
    },
    Reset {
        qubit: usize,
        measurement_id: Option<usize>,
        basis: PauliBasis,
        ideal: crate::Expr,
    },
    Detector {
        detector_id: i64,
        measurement_ids: Vec<usize>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_ids: Vec<usize>,
    },
}

/// Compiled forward bit-packed simulator.
///
/// This is the Rust core counterpart of the Python
/// `FaultScopeSimulator` API. It owns the canonical compact sampler program and
/// can run batches without depending on PyO3 or Python callback objects. Custom
/// decoder and loss callback integration remains in the Python binding crate.
#[derive(Debug, Clone, PartialEq)]
pub struct FaultScopeSimulator {
    pub program: SamplerProgram,
}

impl FaultScopeSimulator {
    /// Compile a typed circuit into the bit-packed forward simulator.
    pub fn new(circuit: Circuit, observables: Vec<LogicalObservable>) -> NpResult<Self> {
        Self::from_circuit_ref(&circuit, observables)
    }

    /// Compile a borrowed typed circuit without cloning its operation tree.
    pub fn from_circuit_ref(
        circuit: &Circuit,
        observables: Vec<LogicalObservable>,
    ) -> NpResult<Self> {
        let program =
            crate::compile_sampler_program_ref(circuit.n_qubits, &circuit.operations, observables)?;
        Ok(Self { program })
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
        run_sampler_program(&self.program, shots, seed, record_events)
    }

    /// Estimate hotspot scores from an externally supplied loss mask.
    ///
    /// This method is useful when a decoder or loss function is implemented
    /// outside of the core crate. The returned estimate is fully aggregated in
    /// Rust. The state must come from this compiled program (or one of its
    /// clones) and must have event recording enabled.
    pub fn estimate_from_loss(
        &self,
        state: &RuntimeState,
        loss_mask: &Mask,
        baseline: Option<f64>,
        top_k: usize,
    ) -> NpResult<HotspotEstimate> {
        self.program
            .estimate_from_loss(state, loss_mask, baseline, top_k)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeState {
    pub(crate) shots: usize,
    pub(crate) all_mask: Mask,
    pub x_frame: Vec<Mask>,
    pub z_frame: Vec<Mask>,
    pub measurements: Vec<Option<Mask>>,
    pub detectors: HashMap<i64, Mask>,
    pub observables: HashMap<i64, Mask>,
    pub(crate) event_masks: Vec<Mask>,
    pub(crate) record_events: bool,
    hotspot_layout_identity: Option<Arc<()>>,
}

impl RuntimeState {
    pub fn shots(&self) -> usize {
        self.shots
    }

    pub fn all_mask(&self) -> &Mask {
        &self.all_mask
    }

    pub fn event_masks(&self) -> &[Mask] {
        &self.event_masks
    }

    pub fn records_events(&self) -> bool {
        self.record_events
    }

    pub fn new(
        n_qubits: usize,
        shots: usize,
        measurement_count: usize,
        noise_location_count: usize,
        record_events: bool,
    ) -> Self {
        Self::new_with_capacities(
            n_qubits,
            shots,
            measurement_count,
            noise_location_count,
            record_events,
            0,
            0,
        )
    }

    fn new_with_capacities(
        n_qubits: usize,
        shots: usize,
        measurement_count: usize,
        noise_location_count: usize,
        record_events: bool,
        detector_capacity: usize,
        observable_capacity: usize,
    ) -> Self {
        let words = word_count(shots);
        let all_mask = Mask::all(shots);
        let event_masks = if record_events {
            vec![Mask::zero(words); noise_location_count]
        } else {
            Vec::new()
        };
        Self {
            shots,
            all_mask,
            x_frame: vec![Mask::zero(words); n_qubits],
            z_frame: vec![Mask::zero(words); n_qubits],
            measurements: vec![None; measurement_count],
            detectors: HashMap::with_capacity(detector_capacity),
            observables: HashMap::with_capacity(observable_capacity),
            event_masks,
            record_events,
            hotspot_layout_identity: None,
        }
    }
}

/// Execute a previously compiled compact sampler program.
pub fn run_sampler_program(
    program: &SamplerProgram,
    shots: usize,
    seed: Option<u64>,
    record_events: bool,
) -> NpResult<RuntimeState> {
    let mut rng = SmallRng::new(seed.unwrap_or(0x4d59_5df4_d0f3_3173));
    let mut state = RuntimeState::new_with_capacities(
        program.n_qubits,
        shots,
        program.measurement_keys.len(),
        program.noise_locations.len(),
        record_events,
        program.capacities.detectors,
        program
            .capacities
            .observables
            .max(program.compiled_observables.len()),
    );
    state.hotspot_layout_identity = Some(program.hotspot_layout_identity.clone());
    let random_masks = random_masks_flat(
        &mut rng,
        program.capacities.random_sources,
        state.all_mask.words.len(),
        state.shots,
    );
    for operation in &program.operations {
        apply_sampler_operation(operation, program, &mut state, &random_masks, &mut rng)?;
    }
    evaluate_sampler_observables(program, &mut state)?;
    Ok(state)
}

fn random_masks_flat(
    rng: &mut SmallRng,
    random_source_count: usize,
    mask_words: usize,
    shots: usize,
) -> Vec<u64> {
    let mut random_masks = vec![0u64; random_source_count * mask_words];
    if mask_words == 0 {
        return random_masks;
    }
    for mask in random_masks.chunks_exact_mut(mask_words) {
        for word in mask.iter_mut() {
            *word = rng.next_u64();
        }
        if shots % 64 != 0 {
            mask[mask_words - 1] &= (1u64 << (shots % 64)) - 1;
        }
    }
    random_masks
}

fn apply_sampler_operation(
    operation: &SamplerOperation,
    program: &SamplerProgram,
    state: &mut RuntimeState,
    random_masks: &[u64],
    rng: &mut SmallRng,
) -> NpResult<()> {
    let random_mask_words = state.all_mask.words.len();
    match operation {
        SamplerOperation::H(qubit) => {
            std::mem::swap(&mut state.x_frame[*qubit], &mut state.z_frame[*qubit]);
        }
        SamplerOperation::S(qubit) | SamplerOperation::SDag(qubit) => {
            state.z_frame[*qubit].xor_assign(&state.x_frame[*qubit]);
        }
        SamplerOperation::Cx(control, target) => {
            xor_mask_between(&mut state.x_frame, *control, *target);
            xor_mask_between(&mut state.z_frame, *target, *control);
        }
        SamplerOperation::Cz(left, right) => {
            if left != right {
                state.z_frame[*left].xor_assign(&state.x_frame[*right]);
                state.z_frame[*right].xor_assign(&state.x_frame[*left]);
            }
        }
        SamplerOperation::Swap(left, right) => {
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        SamplerOperation::Noise(noise_id) => {
            sample_noise(*noise_id, &program.noise_locations[*noise_id], state, rng)?;
        }
        SamplerOperation::MeasureSingle {
            qubit,
            basis,
            measurement_id,
            ideal,
            noise,
        } => {
            let mut bit = ideal.eval_flat(random_masks, random_mask_words, &state.all_mask);
            xor_single_basis_measurement_flip_into(&mut bit, state, *qubit, *basis);
            if let Some(noise_id) = noise {
                sample_measurement_noise_into(
                    *noise_id,
                    &program.noise_locations[*noise_id],
                    state,
                    rng,
                    &mut bit,
                )?;
            }
            record_indexed_measurement(&mut state.measurements, *measurement_id, bit)?;
        }
        SamplerOperation::MeasurePauli {
            qubits,
            pauli,
            measurement_id,
            ideal,
            noise,
        } => {
            let mut bit = ideal.eval_flat(random_masks, random_mask_words, &state.all_mask);
            xor_frame_measurement_flip_into(&mut bit, state, qubits, pauli)?;
            if let Some(noise_id) = noise {
                sample_measurement_noise_into(
                    *noise_id,
                    &program.noise_locations[*noise_id],
                    state,
                    rng,
                    &mut bit,
                )?;
            }
            record_indexed_measurement(&mut state.measurements, *measurement_id, bit)?;
        }
        SamplerOperation::Reset {
            qubit,
            measurement_id,
            basis,
            ideal,
        } => {
            if let Some(measurement_id) = measurement_id {
                let mut outcome = ideal.eval_flat(random_masks, random_mask_words, &state.all_mask);
                xor_single_basis_measurement_flip_into(&mut outcome, state, *qubit, *basis);
                record_indexed_measurement(&mut state.measurements, *measurement_id, outcome)?;
            }
            state.x_frame[*qubit].words.fill(0);
            state.z_frame[*qubit].words.fill(0);
        }
        SamplerOperation::Detector {
            detector_id,
            measurement_ids,
        } => {
            let value = indexed_measurement_parity(
                &state.measurements,
                measurement_ids,
                state.all_mask.words.len(),
            )?;
            state.detectors.insert(*detector_id, value);
        }
        SamplerOperation::ObservableInclude {
            observable_id,
            measurement_ids,
        } => {
            let value = indexed_measurement_parity(
                &state.measurements,
                measurement_ids,
                state.all_mask.words.len(),
            )?;
            state
                .observables
                .entry(*observable_id)
                .and_modify(|existing| existing.xor_assign(&value))
                .or_insert(value);
        }
    }
    Ok(())
}

fn xor_single_basis_measurement_flip_into(
    target: &mut Mask,
    state: &RuntimeState,
    qubit: usize,
    basis: PauliBasis,
) {
    match basis {
        PauliBasis::X => target.xor_assign(&state.z_frame[qubit]),
        PauliBasis::Y => {
            target.xor_assign(&state.x_frame[qubit]);
            target.xor_assign(&state.z_frame[qubit]);
        }
        PauliBasis::Z => target.xor_assign(&state.x_frame[qubit]),
    }
}

fn record_indexed_measurement(
    measurements: &mut [Option<Mask>],
    measurement_id: usize,
    value: Mask,
) -> NpResult<()> {
    if measurements[measurement_id].is_some() {
        return Err(NpError::new(format!(
            "duplicate internal measurement id {measurement_id}"
        )));
    }
    measurements[measurement_id] = Some(value);
    Ok(())
}

fn indexed_measurement_parity(
    measurements: &[Option<Mask>],
    measurement_ids: &[usize],
    words: usize,
) -> NpResult<Mask> {
    let mut parity = Mask::zero(words);
    for measurement_id in measurement_ids {
        let value = measurements[*measurement_id].as_ref().ok_or_else(|| {
            NpError::new(format!("unknown internal measurement id {measurement_id}"))
        })?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

fn evaluate_sampler_observables(
    program: &SamplerProgram,
    state: &mut RuntimeState,
) -> NpResult<()> {
    for observable in &program.compiled_observables {
        let mut value = indexed_measurement_parity(
            &state.measurements,
            &observable.measurement_ids,
            state.all_mask.words.len(),
        )?;
        if !observable.pauli.is_empty() {
            xor_frame_measurement_flip_into(
                &mut value,
                state,
                &observable.pauli_qubits,
                &observable.pauli,
            )?;
        }
        state.observables.insert(observable.id, value);
    }
    Ok(())
}

/// Precomputed capacities used to allocate packed sampler runtime state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeCapacities {
    pub random_sources: usize,
    pub measurements: usize,
    pub detectors: usize,
    pub observables: usize,
}

fn xor_mask_between(masks: &mut [Mask], source: usize, target: usize) {
    if source == target {
        masks[target].words.fill(0);
    } else if source < target {
        let (left, right) = masks.split_at_mut(target);
        right[0].xor_assign(&left[source]);
    } else {
        let (left, right) = masks.split_at_mut(source);
        left[target].xor_assign(&right[0]);
    }
}

fn sample_noise(
    noise_id: usize,
    location: &IndexedNoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> NpResult<()> {
    match &location.model {
        NoiseModel::MeasurementBitFlip => Err(NpError::new(
            "MeasurementBitFlip must be attached to a measurement operation",
        )),
        NoiseModel::BernoulliPauli(pauli) => {
            sample_fixed_pauli_noise(noise_id, location, state, rng, pauli)
        }
        NoiseModel::SingleQubitDepolarizing => {
            sample_single_qubit_depolarizing_noise(noise_id, location, state, rng)
        }
        NoiseModel::TwoQubitDepolarizing => {
            sample_two_qubit_depolarizing_noise(noise_id, location, state, rng)
        }
        NoiseModel::PauliChannel(weights) => {
            sample_pauli_channel_noise(noise_id, location, state, rng, weights)
        }
    }
}

fn sample_measurement_noise_into(
    noise_id: usize,
    location: &IndexedNoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    target: &mut Mask,
) -> NpResult<()> {
    if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
        return Err(NpError::new(
            "native measurement noise supports MeasurementBitFlip only",
        ));
    }
    let mut event_mask = state
        .record_events
        .then(|| state.event_masks.get_mut(noise_id))
        .flatten();
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, _| {
        let word = shot / 64;
        let bit = 1u64 << (shot % 64);
        target.words[word] ^= bit;
        if let Some(mask) = event_mask.as_deref_mut() {
            mask.words[word] ^= bit;
        }
    });
    Ok(())
}

fn sample_fixed_pauli_noise(
    noise_id: usize,
    location: &IndexedNoiseLocation,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    pauli: &str,
) -> NpResult<()> {
    let event_mask = bernoulli_mask(rng, state.shots, location.rate);
    if state.record_events {
        record_location_event_mask(state, noise_id, &event_mask);
    }
    apply_masked_pauli_to_frame(state, &location.qubits, pauli, &event_mask)
}

fn sample_single_qubit_depolarizing_noise(
    noise_id: usize,
    location: &IndexedNoiseLocation,
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
        record_location_event_mask(state, noise_id, &mask);
    }
    Ok(())
}

fn sample_two_qubit_depolarizing_noise(
    noise_id: usize,
    location: &IndexedNoiseLocation,
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
        record_location_event_mask(state, noise_id, &mask);
    }
    Ok(())
}

fn sample_pauli_channel_noise(
    noise_id: usize,
    location: &IndexedNoiseLocation,
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
        record_location_event_mask(state, noise_id, &mask);
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

fn record_location_event_mask(state: &mut RuntimeState, noise_id: usize, event_mask: &Mask) {
    if state.record_events {
        if let Some(mask) = state.event_masks.get_mut(noise_id) {
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

fn xor_frame_measurement_flip_into(
    target: &mut Mask,
    state: &RuntimeState,
    qubits: &[usize],
    pauli: &str,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and pauli must have the same length"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        if z != 0 {
            target.xor_assign(&state.x_frame[*qubit]);
        }
        if x != 0 {
            target.xor_assign(&state.z_frame[*qubit]);
        }
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
    use crate::labels::LocationCatalogBuilder;
    use crate::{Expr, NoiseLocation, Operation};

    fn patterned_mask(shots: usize, offset: usize) -> Mask {
        let mut mask = Mask::zero(word_count(shots));
        for shot in 0..shots {
            if (shot + offset) % 3 == 0 {
                set_shot_bit(&mut mask, shot);
            }
        }
        mask
    }

    fn empty_sampler_program(n_qubits: usize) -> SamplerProgram {
        SamplerProgram {
            n_qubits,
            operations: Vec::new(),
            observables: Vec::new(),
            compiled_observables: Vec::new(),
            measurement_keys: Vec::new(),
            noise_locations: Vec::new(),
            location_catalog: LocationCatalog::default(),
            capacities: RuntimeCapacities::default(),
            stored_operation_count: 0,
            logical_operation_count: 0,
            loop_kernel_count: 0,
            hotspot_layout_identity: Arc::new(()),
        }
    }

    fn apply_test_operation(
        operation: SamplerOperation,
        program: &SamplerProgram,
        state: &mut RuntimeState,
        rng: &mut SmallRng,
    ) {
        apply_sampler_operation(&operation, program, state, &[], rng).unwrap();
    }

    #[test]
    fn clifford_frame_updates_preserve_boundary_semantics() {
        for shots in [1, 63, 64, 65, 129] {
            let a = patterned_mask(shots, 0);
            let b = patterned_mask(shots, 1);
            let c = patterned_mask(shots, 2);
            let d = patterned_mask(shots, 3);
            let program = empty_sampler_program(2);
            let mut rng = SmallRng::new(7);

            let mut state = RuntimeState::new(2, shots, 0, 0, false);
            state.x_frame[0] = a.clone();
            state.z_frame[0] = b.clone();
            apply_test_operation(SamplerOperation::H(0), &program, &mut state, &mut rng);
            assert_eq!(state.x_frame[0], b);
            assert_eq!(state.z_frame[0], a);

            let mut state = RuntimeState::new(2, shots, 0, 0, false);
            state.x_frame[0] = a.clone();
            state.z_frame[0] = b.clone();
            let mut expected = b.clone();
            expected.xor_assign(&a);
            apply_test_operation(SamplerOperation::S(0), &program, &mut state, &mut rng);
            assert_eq!(state.z_frame[0], expected);
            state.z_frame[0] = b.clone();
            apply_test_operation(SamplerOperation::SDag(0), &program, &mut state, &mut rng);
            assert_eq!(state.z_frame[0], expected);

            let mut state = RuntimeState::new(2, shots, 0, 0, false);
            state.x_frame = vec![a.clone(), b.clone()];
            state.z_frame = vec![c.clone(), d.clone()];
            let mut expected_x_target = b.clone();
            expected_x_target.xor_assign(&a);
            let mut expected_z_control = c.clone();
            expected_z_control.xor_assign(&d);
            apply_test_operation(SamplerOperation::Cx(0, 1), &program, &mut state, &mut rng);
            assert_eq!(state.x_frame, vec![a.clone(), expected_x_target]);
            assert_eq!(state.z_frame, vec![expected_z_control, d.clone()]);

            let mut state = RuntimeState::new(2, shots, 0, 0, false);
            state.x_frame = vec![a.clone(), b.clone()];
            state.z_frame = vec![c.clone(), d.clone()];
            let mut expected_z_left = c.clone();
            expected_z_left.xor_assign(&b);
            let mut expected_z_right = d.clone();
            expected_z_right.xor_assign(&a);
            apply_test_operation(SamplerOperation::Cz(0, 1), &program, &mut state, &mut rng);
            assert_eq!(state.z_frame, vec![expected_z_left, expected_z_right]);

            let mut state = RuntimeState::new(2, shots, 0, 0, false);
            state.x_frame = vec![a.clone(), b.clone()];
            state.z_frame = vec![c.clone(), d.clone()];
            apply_test_operation(SamplerOperation::Swap(0, 1), &program, &mut state, &mut rng);
            assert_eq!(state.x_frame, vec![b.clone(), a.clone()]);
            assert_eq!(state.z_frame, vec![d.clone(), c.clone()]);

            let mut state = RuntimeState::new(1, shots, 1, 0, false);
            state.x_frame[0] = a.clone();
            state.z_frame[0] = b;
            let mut reset_program = empty_sampler_program(1);
            reset_program.measurement_keys.push("reset".to_string());
            apply_test_operation(
                SamplerOperation::Reset {
                    qubit: 0,
                    measurement_id: Some(0),
                    basis: PauliBasis::Z,
                    ideal: Expr::constant(false),
                },
                &reset_program,
                &mut state,
                &mut rng,
            );
            assert_eq!(state.measurements[0], Some(a));
            assert_eq!(state.x_frame[0], Mask::zero(word_count(shots)));
            assert_eq!(state.z_frame[0], Mask::zero(word_count(shots)));
        }
    }

    #[test]
    fn faultscope_simulator_owns_the_canonical_sampler_program() {
        let circuit = Circuit {
            n_qubits: 2,
            operations: vec![
                Operation::H(0),
                Operation::Cx(0, 1),
                Operation::Noise(NoiseLocation {
                    id: "n".to_string(),
                    model: NoiseModel::BernoulliPauli("X".to_string()),
                    rate: 0.2,
                    qubits: vec![0],
                    tags: HashMap::new(),
                }),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m0".to_string()),
                    basis: "Z".to_string(),
                    noise: Some(NoiseLocation {
                        id: "nm".to_string(),
                        model: NoiseModel::MeasurementBitFlip,
                        rate: 0.15,
                        qubits: vec![0],
                        tags: HashMap::new(),
                    }),
                },
                Operation::Reset {
                    qubit: 1,
                    key: Some("m1".to_string()),
                    basis: "X".to_string(),
                },
                Operation::Detector {
                    detector_id: Some(3),
                    measurement_keys: vec!["m0".to_string(), "m1".to_string()],
                    coords: Vec::new(),
                },
                Operation::ObservableInclude {
                    observable_id: 5,
                    measurement_keys: vec!["m0".to_string()],
                },
            ],
        };
        let observables = vec![LogicalObservable {
            id: 6,
            measurement_keys: vec!["m1".to_string()],
            pauli_qubits: vec![0],
            pauli: "X".to_string(),
        }];
        let public = FaultScopeSimulator::from_circuit_ref(&circuit, observables.clone()).unwrap();
        let compact =
            crate::compile_sampler_program_ref(circuit.n_qubits, &circuit.operations, observables)
                .unwrap();

        assert_eq!(compact, public.program);
        let public_state = public.run_batch(129, Some(9182), true).unwrap();
        let compact_state = run_sampler_program(&compact, 129, Some(9182), true).unwrap();
        assert_eq!(compact_state, public_state);
    }

    #[test]
    fn compact_sampler_preserves_unknown_measurement_error() {
        let circuit = Circuit {
            n_qubits: 1,
            operations: vec![Operation::Detector {
                detector_id: Some(0),
                measurement_keys: vec!["missing".to_string()],
                coords: Vec::new(),
            }],
        };
        let public = FaultScopeSimulator::from_circuit_ref(&circuit, Vec::new()).unwrap();
        let compact =
            crate::compile_sampler_program_ref(1, &circuit.operations, Vec::new()).unwrap();

        let public_error = public.run_batch(65, Some(1), false).unwrap_err();
        let compact_error = run_sampler_program(&compact, 65, Some(1), false).unwrap_err();
        assert_eq!(compact_error.message(), public_error.message());
    }

    #[test]
    fn faultscope_simulator_repeat_program_matches_direct_compile() {
        let circuit = Circuit {
            n_qubits: 2,
            operations: vec![Operation::Repeat {
                count: 8,
                body: vec![
                    Operation::H(0),
                    Operation::Cx(0, 1),
                    Operation::MeasureReset {
                        qubit: 0,
                        basis: "Z".to_string(),
                    },
                    Operation::MeasureReset {
                        qubit: 1,
                        basis: "X".to_string(),
                    },
                    Operation::DetectorRec {
                        detector_id: None,
                        lookbacks: vec![1, 2],
                        coords: Vec::new(),
                    },
                    Operation::ObservableIncludeRec {
                        observable_id: 0,
                        lookbacks: vec![2],
                    },
                ],
            }],
        };
        let public = FaultScopeSimulator::from_circuit_ref(&circuit, Vec::new()).unwrap();
        let compact =
            crate::compile_sampler_program_ref(2, &circuit.operations, Vec::new()).unwrap();

        assert_eq!(compact, public.program);
        let public_state = public.run_batch(130, Some(73), false).unwrap();
        let compact_state = run_sampler_program(&compact, 130, Some(73), false).unwrap();
        assert_eq!(compact_state, public_state);
    }

    #[test]
    fn in_place_measurement_noise_matches_mask_sampler() {
        for shots in [1, 63, 64, 65, 129] {
            let location = NoiseLocation {
                id: "measurement".to_string(),
                model: NoiseModel::MeasurementBitFlip,
                rate: 0.17,
                qubits: vec![0],
                tags: HashMap::new(),
            };
            let mut expected_rng = SmallRng::new(1234);
            let expected = bernoulli_mask(&mut expected_rng, shots, location.rate);
            let mut program = empty_sampler_program(1);
            program.measurement_keys.push("m".to_string());
            let mut catalog = LocationCatalogBuilder::with_capacity(1);
            let location_id = catalog.intern(location.id.clone(), location.tags.clone());
            program
                .noise_locations
                .push(IndexedNoiseLocation::from_input(location_id, location));
            program.location_catalog = catalog.finish();
            program.capacities.measurements = 1;
            program.operations.push(SamplerOperation::MeasureSingle {
                qubit: 0,
                basis: PauliBasis::Z,
                measurement_id: 0,
                ideal: Expr::constant(false),
                noise: Some(0),
            });
            let state = run_sampler_program(&program, shots, Some(1234), true).unwrap();
            assert_eq!(state.measurements[0].as_ref(), Some(&expected));
            assert_eq!(state.event_masks[0], expected);
        }
    }

    #[test]
    fn sampler_program_records_measurement_and_detector() {
        let mut program = empty_sampler_program(1);
        program.measurement_keys.push("m0".to_string());
        program.capacities.measurements = 1;
        program.capacities.detectors = 1;
        program.operations = vec![
            SamplerOperation::MeasureSingle {
                qubit: 0,
                basis: PauliBasis::Z,
                measurement_id: 0,
                ideal: Expr::constant(false),
                noise: None,
            },
            SamplerOperation::Detector {
                detector_id: 0,
                measurement_ids: vec![0],
            },
        ];

        let state = run_sampler_program(&program, 5, Some(1), true).unwrap();

        assert_eq!(state.measurements[0], Some(Mask::zero(1)));
        assert_eq!(state.detectors[&0], Mask::zero(1));
    }

    #[test]
    fn sampler_program_preserves_zero_shot_behavior() {
        let mut program = empty_sampler_program(1);
        program.measurement_keys.push("m0".to_string());
        program.capacities.random_sources = 1;
        program.capacities.measurements = 1;
        program.operations.push(SamplerOperation::MeasureSingle {
            qubit: 0,
            basis: PauliBasis::Z,
            measurement_id: 0,
            ideal: Expr::random(0),
            noise: None,
        });

        let state = run_sampler_program(&program, 0, Some(1), false).unwrap();

        assert_eq!(state.shots, 0);
        assert!(state.measurements[0].as_ref().unwrap().words.is_empty());
    }

    #[test]
    fn measurement_free_parity_and_frame_observables_preserve_batch_width() {
        let simulator = FaultScopeSimulator::new(
            Circuit {
                n_qubits: 1,
                operations: vec![
                    Operation::Noise(NoiseLocation {
                        id: "x0".to_string(),
                        model: NoiseModel::BernoulliPauli("X".to_string()),
                        rate: 1.0,
                        qubits: vec![0],
                        tags: HashMap::new(),
                    }),
                    Operation::Detector {
                        detector_id: Some(7),
                        measurement_keys: Vec::new(),
                        coords: Vec::new(),
                    },
                    Operation::ObservableInclude {
                        observable_id: 8,
                        measurement_keys: Vec::new(),
                    },
                ],
            },
            vec![LogicalObservable {
                id: 9,
                measurement_keys: Vec::new(),
                pauli_qubits: vec![0],
                pauli: "Z".to_string(),
            }],
        )
        .unwrap();

        for shots in [0, 1, 63, 64, 65, 129, 513] {
            let state = run_sampler_program(&simulator.program, shots, Some(1), true).unwrap();
            let words = word_count(shots);

            assert_eq!(state.all_mask.words.len(), words);
            assert_eq!(state.detectors[&7], Mask::zero(words));
            assert_eq!(state.observables[&8], Mask::zero(words));
            assert_eq!(state.observables[&9], state.all_mask);
            for mask in state
                .x_frame
                .iter()
                .chain(&state.z_frame)
                .chain(state.detectors.values())
                .chain(state.observables.values())
                .chain(&state.event_masks)
            {
                assert_eq!(mask.words.len(), words);
            }
        }
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
        let simulator = FaultScopeSimulator::new(
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
        let measurement = state.measurements[0].as_ref().unwrap();
        let estimate = simulator
            .estimate_from_loss(&state, measurement, None, 1)
            .unwrap();

        assert_eq!(measurement, &state.all_mask);
        assert_eq!(estimate.mean_loss, 1.0);
        assert_eq!(estimate.top_locations, vec!["x0"]);
    }

    #[test]
    fn simulator_hotspot_rejects_unrecorded_foreign_and_unbound_states() {
        let circuit = Circuit {
            n_qubits: 1,
            operations: vec![Operation::Noise(NoiseLocation {
                id: "x0".to_string(),
                model: NoiseModel::BernoulliPauli("X".to_string()),
                rate: 0.25,
                qubits: vec![0],
                tags: HashMap::new(),
            })],
        };
        let simulator = FaultScopeSimulator::new(circuit.clone(), Vec::new()).unwrap();
        let foreign = FaultScopeSimulator::new(circuit, Vec::new()).unwrap();
        let state = simulator.run_batch(8, Some(1), true).unwrap();
        let loss = state.event_masks[0].clone();

        let err = foreign
            .estimate_from_loss(&state, &loss, None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("layout does not match"));
        simulator
            .clone()
            .estimate_from_loss(&state, &loss, None, 1)
            .unwrap();

        let state = simulator.run_batch(8, Some(1), false).unwrap();
        let err = simulator
            .estimate_from_loss(&state, &Mask::zero(1), None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("recorded event masks"));

        let unbound = RuntimeState::new(1, 8, 0, 1, true);
        let err = simulator
            .estimate_from_loss(&unbound, &Mask::zero(1), None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("not bound"));
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

        let err = FaultScopeSimulator::new(
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

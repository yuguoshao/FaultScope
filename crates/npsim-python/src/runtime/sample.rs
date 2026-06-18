use crate::*;

pub(crate) fn run_packed_sample(
    sampler: &NativePackedSampler,
    shots: usize,
    seed: Option<u64>,
    record_events: bool,
) -> PyResult<RuntimeState> {
    sampler
        .simulator
        .run_batch(shots, seed, record_events)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

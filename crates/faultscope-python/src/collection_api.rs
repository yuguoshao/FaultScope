use crate::*;

#[pyfunction]
#[pyo3(signature = (sampler, max_shots, max_errors=None, batch_size=10000, seed=None, decoder=None))]
pub(crate) fn _collect_dem_logical_error_stats(
    py: Python<'_>,
    sampler: PyRef<'_, NativeDemSampler>,
    max_shots: usize,
    max_errors: Option<usize>,
    batch_size: usize,
    seed: Option<u64>,
    decoder: Option<&Bound<'_, PyAny>>,
) -> PyResult<(usize, usize, f64)> {
    if max_shots == 0 {
        return Err(PyValueError::new_err("max_shots must be positive"));
    }
    if batch_size == 0 {
        return Err(PyValueError::new_err("batch_size must be positive"));
    }
    let native_decoder = match decoder {
        Some(decoder) => Some(
            native_decoder_from_py(decoder)?
                .ok_or_else(|| PyTypeError::new_err("collection requires a native decoder"))?,
        ),
        None => None,
    };
    let options = faultscope_collection::api::DemLogicalCollectionOptions {
        max_shots,
        max_errors,
        batch_size,
        seed,
    };
    let simulator = &sampler.simulator;
    let stats = py
        .allow_threads(|| {
            faultscope_collection::api::collect_dem_logical_error_stats(
                simulator,
                options,
                native_decoder.as_deref(),
            )
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((stats.shots, stats.errors, stats.seconds))
}

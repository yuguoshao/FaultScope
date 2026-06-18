use crate::*;

pub(crate) type DemBatch = npsim_core::DemBatch;

pub(crate) type DemEstimate = npsim_core::DemHotspotEstimate;
pub(crate) type PackedEstimate = npsim_core::HotspotEstimate;
pub(crate) type DetectorGraphKey = npsim_core::DetectorGraphKey;

pub(crate) fn run_dem_batch(
    sampler: &NativeDemSampler,
    shots: usize,
    rng: &mut SmallRng,
    return_edge_events: bool,
) -> DemBatch {
    sampler
        .simulator
        .run_batch_with_rng(shots, rng, return_edge_events)
}

pub(crate) fn compute_packed_estimate(
    sampler: &NativePackedSampler,
    state: &RuntimeState,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> PackedEstimate {
    sampler
        .simulator
        .estimate_from_loss(state, loss_mask, baseline, top_k)
}

pub(crate) fn compute_dem_estimate(
    sampler: &NativeDemSampler,
    batch: &DemBatch,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> DemEstimate {
    sampler
        .simulator
        .estimate_from_loss(batch, loss_mask, baseline, top_k)
}

pub(crate) fn string_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<String, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn usize_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<usize, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(*key, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn i64_f64_map_to_py(py: Python<'_>, values: &HashMap<i64, f64>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(*key, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn tag_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<TagValue, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(tag_value_to_py(py, key)?, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn tag_value_to_py(py: Python<'_>, value: &TagValue) -> PyResult<PyObject> {
    match value {
        TagValue::None => Ok(py.None()),
        TagValue::Bool(value) => Ok(PyBool::new(py, *value).to_owned().into_any().unbind()),
        TagValue::Int(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
        TagValue::Float(bits) => Ok(PyFloat::new(py, f64::from_bits(*bits)).into_any().unbind()),
        TagValue::String(value) => Ok(PyString::new(py, value).into_any().unbind()),
    }
}

pub(crate) fn string_tag_map_to_py(
    py: Python<'_>,
    values: &HashMap<String, TagValue>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, tag_value_to_py(py, value)?)?;
    }
    Ok(dict.into())
}

pub(crate) fn graph_key_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<DetectorGraphKey, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        let detectors = PyTuple::new(py, key.detectors.iter().copied())?;
        let observables = PyTuple::new(py, key.observables.iter().copied())?;
        let graph_key = PyTuple::new(py, [detectors, observables])?;
        dict.set_item(graph_key, *value)?;
    }
    Ok(dict.into())
}

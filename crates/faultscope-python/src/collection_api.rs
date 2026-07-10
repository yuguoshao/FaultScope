use std::sync::Arc;

use crate::*;
use pyo3::exceptions::PyRuntimeError;

#[pyfunction]
#[pyo3(signature = (
    tasks,
    num_workers=1,
    seed=None,
    count_observable_error_combos=false,
    count_detection_events=false,
    custom_error_count_key=None,
    existing_stats=None,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _collect_dem_logical_error_stats_many(
    py: Python<'_>,
    tasks: &Bound<'_, PyAny>,
    num_workers: usize,
    seed: Option<u64>,
    count_observable_error_combos: bool,
    count_detection_events: bool,
    custom_error_count_key: Option<String>,
    existing_stats: Option<&Bound<'_, PyAny>>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Vec<PyObject>> {
    let rust_tasks = py_collection_tasks_to_rust(tasks)?;
    let existing = py_existing_stats_to_rust(existing_stats)?;
    let run_options = faultscope_collection::api::DemLogicalCollectionRunOptions {
        num_workers,
        seed,
        count_observable_error_combos,
        count_detection_events,
        custom_error_count_key,
    };

    let stats = if let Some(progress_callback) = progress_callback {
        py.allow_threads(move || {
            let progress = move |stats: &faultscope_collection::api::DemLogicalCollectionStats| {
                Python::with_gil(|py| {
                    let py_stats = collection_stats_to_py(py, stats).map_err(|err| {
                        faultscope_core::NpError::new(format!(
                            "collection progress callback failed: {err}"
                        ))
                    })?;
                    progress_callback
                        .bind(py)
                        .call1((py_stats,))
                        .map_err(|err| {
                            faultscope_core::NpError::new(format!(
                                "collection progress callback failed: {err}"
                            ))
                        })?;
                    Ok(())
                })
            };
            faultscope_collection::api::collect_dem_logical_error_tasks_with_progress(
                rust_tasks,
                run_options,
                existing,
                progress,
            )
        })
    } else {
        py.allow_threads(|| {
            faultscope_collection::api::collect_dem_logical_error_tasks(
                rust_tasks,
                run_options,
                existing,
            )
        })
    }
    .map_err(collection_error_to_py)?;

    stats
        .iter()
        .map(|stats| collection_stats_to_py(py, stats))
        .collect()
}

fn collection_error_to_py(err: faultscope_core::NpError) -> PyErr {
    let message = err.to_string();
    if message.contains("collection progress callback failed") {
        PyRuntimeError::new_err(message)
    } else {
        PyValueError::new_err(message)
    }
}

fn py_collection_tasks_to_rust(
    tasks: &Bound<'_, PyAny>,
) -> PyResult<Vec<faultscope_collection::api::DemLogicalCollectionTask>> {
    let iterator = PyIterator::from_object(tasks)?;
    let mut out = Vec::new();
    for item in iterator {
        let item = item?;
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("collection task entries must be dicts"))?;
        out.push(py_collection_task_to_rust(dict)?);
    }
    Ok(out)
}

fn py_collection_task_to_rust(
    dict: &Bound<'_, PyDict>,
) -> PyResult<faultscope_collection::api::DemLogicalCollectionTask> {
    let sampler_value = required_item(dict, "sampler")?;
    let sampler = sampler_value.extract::<PyRef<'_, NativeDemSampler>>()?;
    let sampler = Arc::new(sampler.simulator.clone());

    let decoder = optional_item(dict, "decoder")?;
    let decoder = match decoder {
        Some(decoder) if !decoder.is_none() => Some(
            native_decoder_from_py(&decoder)?
                .ok_or_else(|| PyTypeError::new_err("collection requires a native decoder"))?,
        ),
        _ => None,
    };
    let decoder_name = optional_string(dict, "decoder_name")?;

    Ok(faultscope_collection::api::DemLogicalCollectionTask {
        task_id: required_string(dict, "task_id")?,
        strong_id: required_string(dict, "strong_id")?,
        sampler,
        decoder,
        decoder_name,
        metadata_json: required_string(dict, "metadata_json")?,
        options: faultscope_collection::api::DemLogicalCollectionOptions {
            max_shots: required_item(dict, "max_shots")?.extract::<usize>()?,
            max_errors: optional_usize(dict, "max_errors")?,
            batch_size: required_item(dict, "batch_size")?.extract::<usize>()?,
            seed: optional_u64(dict, "seed")?,
            start_batch_size: optional_usize(dict, "start_batch_size")?,
            max_batch_size: optional_usize(dict, "max_batch_size")?,
            max_batch_seconds: optional_f64(dict, "max_batch_seconds")?,
        },
        postselection_mask: optional_bytes(dict, "postselection_mask")?,
        postselected_observables_mask: optional_bytes(dict, "postselected_observables_mask")?,
    })
}

fn py_existing_stats_to_rust(
    stats: Option<&Bound<'_, PyAny>>,
) -> PyResult<HashMap<String, faultscope_collection::api::DemLogicalCollectionStats>> {
    let Some(stats) = stats else {
        return Ok(HashMap::new());
    };
    if stats.is_none() {
        return Ok(HashMap::new());
    }
    let iterator = PyIterator::from_object(stats)?;
    let mut out = HashMap::<String, faultscope_collection::api::DemLogicalCollectionStats>::new();
    for item in iterator {
        let item = item?;
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("existing collection stats entries must be dicts"))?;
        let stats = py_collection_stats_to_rust(dict)?;
        if let Some(existing) = out.get_mut(&stats.strong_id) {
            existing
                .add_assign_checked(&stats)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        } else {
            out.insert(stats.strong_id.clone(), stats);
        }
    }
    Ok(out)
}

fn py_collection_stats_to_rust(
    dict: &Bound<'_, PyDict>,
) -> PyResult<faultscope_collection::api::DemLogicalCollectionStats> {
    let custom_counts = required_item(dict, "custom_counts")?;
    let custom_counts = custom_counts
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("custom_counts must be a dict"))?;
    let mut counts = HashMap::new();
    for (key, value) in custom_counts {
        counts.insert(key.extract::<String>()?, value.extract::<usize>()?);
    }
    Ok(faultscope_collection::api::DemLogicalCollectionStats {
        task_id: required_string(dict, "task_id")?,
        strong_id: required_string(dict, "strong_id")?,
        decoder: optional_string(dict, "decoder")?,
        metadata_json: required_string(dict, "metadata_json")?,
        shots: required_item(dict, "shots")?.extract::<usize>()?,
        errors: required_item(dict, "errors")?.extract::<usize>()?,
        discards: required_item(dict, "discards")?.extract::<usize>()?,
        seconds: required_item(dict, "seconds")?.extract::<f64>()?,
        custom_counts: counts,
    })
}

fn collection_stats_to_py(
    py: Python<'_>,
    stats: &faultscope_collection::api::DemLogicalCollectionStats,
) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("task_id", &stats.task_id)?;
    out.set_item("strong_id", &stats.strong_id)?;
    out.set_item("decoder", stats.decoder.as_deref())?;
    out.set_item("metadata_json", &stats.metadata_json)?;
    out.set_item("shots", stats.shots)?;
    out.set_item("errors", stats.errors)?;
    out.set_item("discards", stats.discards)?;
    out.set_item("seconds", stats.seconds)?;
    let counts = PyDict::new(py);
    for (key, value) in &stats.custom_counts {
        counts.set_item(key, value)?;
    }
    out.set_item("custom_counts", counts)?;
    Ok(out.into())
}

fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing collection task field {key}")))
}

fn optional_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    dict.get_item(key)
}

fn required_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    required_item(dict, key)?.extract::<String>()
}

fn optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    optional_item(dict, key)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()
}

fn optional_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    optional_item(dict, key)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<usize>())
        .transpose()
}

fn optional_u64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u64>> {
    optional_item(dict, key)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<u64>())
        .transpose()
}

fn optional_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    optional_item(dict, key)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<f64>())
        .transpose()
}

fn optional_bytes(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<u8>>> {
    let Some(value) = optional_item(dict, key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(bytes) = value.downcast::<PyBytes>() {
        return Ok(Some(bytes.as_bytes().to_vec()));
    }
    value.extract::<Vec<u8>>().map(Some)
}

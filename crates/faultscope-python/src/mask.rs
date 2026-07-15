use crate::*;
pub(crate) use faultscope_core::{Mask, RuntimeState};

pub(crate) fn mask_to_py(py: Python<'_>, mask: &Mask) -> PyResult<PyObject> {
    let int_type = py.import("builtins")?.getattr("int")?;
    let from_bytes = int_type.getattr("from_bytes")?;
    let byteorder = PyString::new(py, "little");
    mask_to_py_with_converter(py, mask, &from_bytes, &byteorder, &mut Vec::new())
}

pub(crate) fn mask_to_py_with_converter(
    py: Python<'_>,
    mask: &Mask,
    from_bytes: &Bound<'_, PyAny>,
    byteorder: &Bound<'_, PyString>,
    bytes: &mut Vec<u8>,
) -> PyResult<PyObject> {
    bytes.clear();
    bytes.reserve(mask.words.len() * 8);
    for word in &mask.words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    Ok(from_bytes
        .call1((PyBytes::new(py, bytes), byteorder))?
        .into())
}

pub(crate) fn py_int_to_mask(
    value: &Bound<'_, PyAny>,
    words: usize,
    shots: usize,
) -> PyResult<Mask> {
    let byte_len = words * 8;
    let bytes_any = value.call_method1("to_bytes", (byte_len, "little"))?;
    let bytes = bytes_any.downcast::<PyBytes>()?.as_bytes();
    let mut out = Mask::zero(words);
    for (word_index, chunk) in bytes.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        out.words[word_index] = u64::from_le_bytes(word);
    }
    out.clear_unused(shots);
    out.and_assign(&Mask::all(shots));
    Ok(out)
}

pub(crate) fn py_int_bit(value: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
    let shifted = value.call_method1("__rshift__", (shot,))?;
    shifted.call_method1("__and__", (1,))?.extract::<u8>()
}

pub(crate) fn mask_bit(mask: &Mask, shot: usize) -> PyResult<u8> {
    if shot >= mask.words.len() * 64 {
        return Err(PyValueError::new_err(format!(
            "shot index {shot} out of range"
        )));
    }
    Ok(((mask.words[shot / 64] >> (shot % 64)) & 1) as u8)
}

pub(crate) fn mask_vec_to_py(py: Python<'_>, values: &[Mask]) -> PyResult<PyObject> {
    let int_type = py.import("builtins")?.getattr("int")?;
    let from_bytes = int_type.getattr("from_bytes")?;
    let byteorder = PyString::new(py, "little");
    let mut bytes = Vec::new();
    let list = PyList::empty(py);
    for value in values {
        list.append(mask_to_py_with_converter(
            py,
            value,
            &from_bytes,
            &byteorder,
            &mut bytes,
        )?)?;
    }
    Ok(list.into())
}

pub(crate) fn measurement_masks_to_py(
    py: Python<'_>,
    program: &faultscope_core::SamplerProgram,
    values: &[Option<Mask>],
) -> PyResult<PyObject> {
    let int_type = py.import("builtins")?.getattr("int")?;
    let from_bytes = int_type.getattr("from_bytes")?;
    let byteorder = PyString::new(py, "little");
    let mut bytes = Vec::new();
    let dict = PyDict::new(py);
    for (measurement_id, value) in values.iter().enumerate() {
        let Some(value) = value else {
            continue;
        };
        let Some(key) = program.measurement_keys.get(measurement_id) else {
            continue;
        };
        dict.set_item(
            key,
            mask_to_py_with_converter(py, value, &from_bytes, &byteorder, &mut bytes)?,
        )?;
    }
    Ok(dict.into())
}

pub(crate) fn noise_event_masks_to_py(
    py: Python<'_>,
    program: &faultscope_core::SamplerProgram,
    values: &[Mask],
) -> PyResult<PyObject> {
    let int_type = py.import("builtins")?.getattr("int")?;
    let from_bytes = int_type.getattr("from_bytes")?;
    let byteorder = PyString::new(py, "little");
    let mut bytes = Vec::new();
    let dict = PyDict::new(py);
    for (noise_id, value) in values.iter().enumerate() {
        let Some(location) = program.noise_locations.get(noise_id) else {
            continue;
        };
        dict.set_item(
            program.location_catalog.label(location.location_id),
            mask_to_py_with_converter(py, value, &from_bytes, &byteorder, &mut bytes)?,
        )?;
    }
    Ok(dict.into())
}

pub(crate) fn int_map_to_py(py: Python<'_>, values: &HashMap<i64, Mask>) -> PyResult<PyObject> {
    let int_type = py.import("builtins")?.getattr("int")?;
    let from_bytes = int_type.getattr("from_bytes")?;
    let byteorder = PyString::new(py, "little");
    let mut bytes = Vec::new();
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(
            key,
            mask_to_py_with_converter(py, value, &from_bytes, &byteorder, &mut bytes)?,
        )?;
    }
    Ok(dict.into())
}

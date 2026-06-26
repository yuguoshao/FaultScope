use crate::*;
pub(crate) use faultscope_core::{Mask, RuntimeState};

pub(crate) fn mask_to_py(py: Python<'_>, mask: &Mask) -> PyResult<PyObject> {
    let mut bytes = Vec::with_capacity(mask.words.len() * 8);
    for word in &mask.words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let int_type = py.import("builtins")?.getattr("int")?;
    Ok(int_type
        .call_method1("from_bytes", (PyBytes::new(py, &bytes), "little"))?
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
    let list = PyList::empty(py);
    for value in values {
        list.append(mask_to_py(py, value)?)?;
    }
    Ok(list.into())
}

pub(crate) fn map_to_py(py: Python<'_>, values: &HashMap<String, Mask>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, mask_to_py(py, value)?)?;
    }
    Ok(dict.into())
}

pub(crate) fn int_map_to_py(py: Python<'_>, values: &HashMap<i64, Mask>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, mask_to_py(py, value)?)?;
    }
    Ok(dict.into())
}

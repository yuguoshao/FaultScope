use crate::*;

#[derive(Clone)]
pub(crate) struct Mask {
    pub(crate) words: Vec<u64>,
}

impl Mask {
    pub(crate) fn zero(words: usize) -> Self {
        Self {
            words: vec![0; words],
        }
    }

    pub(crate) fn all(shots: usize) -> Self {
        let words = word_count(shots);
        let mut mask = Self {
            words: vec![u64::MAX; words],
        };
        mask.clear_unused(shots);
        mask
    }

    pub(crate) fn xor_assign(&mut self, other: &Mask) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left ^= *right;
        }
    }

    pub(crate) fn or_assign(&mut self, other: &Mask) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left |= *right;
        }
    }

    pub(crate) fn and_assign(&mut self, other: &Mask) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left &= *right;
        }
    }

    pub(crate) fn bit_count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub(crate) fn and_count(&self, other: &Mask) -> usize {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(left, right)| (left & right).count_ones() as usize)
            .sum()
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    pub(crate) fn clear_unused(&mut self, shots: usize) {
        let extra = shots % 64;
        if extra != 0 {
            let keep = (1u64 << extra) - 1;
            if let Some(last) = self.words.last_mut() {
                *last &= keep;
            }
        }
    }
}

pub(crate) struct RuntimeState {
    pub(crate) shots: usize,
    pub(crate) all_mask: Mask,
    pub(crate) x_frame: Vec<Mask>,
    pub(crate) z_frame: Vec<Mask>,
    pub(crate) measurements: HashMap<String, Mask>,
    pub(crate) detectors: HashMap<i64, Mask>,
    pub(crate) observables: HashMap<i64, Mask>,
    pub(crate) event_masks: HashMap<String, Mask>,
    pub(crate) record_events: bool,
}

impl RuntimeState {
    pub(crate) fn new(
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

    pub(crate) fn to_py(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("shots", self.shots)?;
        out.set_item("all_mask", mask_to_py(py, &self.all_mask)?)?;
        out.set_item("x_frame", mask_vec_to_py(py, &self.x_frame)?)?;
        out.set_item("z_frame", mask_vec_to_py(py, &self.z_frame)?)?;
        out.set_item("measurements", map_to_py(py, &self.measurements)?)?;
        out.set_item("detectors", int_map_to_py(py, &self.detectors)?)?;
        out.set_item("observables", int_map_to_py(py, &self.observables)?)?;
        out.set_item("noise_event_masks", map_to_py(py, &self.event_masks)?)?;
        Ok(out.into())
    }
}

pub(crate) fn word_count(shots: usize) -> usize {
    (shots + 63) / 64
}

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

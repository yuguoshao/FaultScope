use crate::*;

const MIN_SCORE_RATE: f64 = 1e-12;
const TWO_QUBIT_EVENTS: [&str; 15] = [
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
];

#[pyclass(name = "BernoulliPauliNoise", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyBernoulliPauliNoise {
    pub(crate) pauli: String,
}

#[pymethods]
impl PyBernoulliPauliNoise {
    #[new]
    pub(crate) fn new(pauli: String) -> Self {
        Self { pauli }
    }

    #[getter]
    pub(crate) fn pauli(&self) -> &str {
        &self.pauli
    }

    pub(crate) fn sample(&self, rng: &Bound<'_, PyAny>, rate: f64) -> PyResult<String> {
        validate_rate(rate)?;
        let draw = rng.call_method0("random")?.extract::<f64>()?;
        if draw < rate {
            Ok(self.pauli.clone())
        } else {
            Ok("I".repeat(self.pauli.len()))
        }
    }

    pub(crate) fn score(&self, event: &Bound<'_, PyAny>, rate: f64) -> PyResult<f64> {
        let rate = score_rate(rate);
        let event = py_str(event)?;
        if event == "I".repeat(self.pauli.len()) {
            Ok(-1.0 / (1.0 - rate))
        } else {
            Ok(1.0 / rate)
        }
    }

    pub(crate) fn apply(
        &self,
        py: Python<'_>,
        event: &Bound<'_, PyAny>,
        state: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
        qubits: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        apply_pauli_event(py, event, state, frame, qubits, None)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!("BernoulliPauliNoise(pauli={:?})", self.pauli)
    }
}

#[pyclass(name = "PauliChannel", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyPauliChannel {
    pub(crate) weights: Vec<(String, f64)>,
}

#[pymethods]
impl PyPauliChannel {
    #[new]
    pub(crate) fn new(weights: &Bound<'_, PyAny>) -> PyResult<Self> {
        let weights = coerce_weights(weights)?;
        validate_pauli_channel(&weights)?;
        Ok(Self { weights })
    }

    #[getter]
    pub(crate) fn weights(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (pauli, weight) in &self.weights {
            dict.set_item(pauli, *weight)?;
        }
        Ok(dict.into())
    }

    #[getter]
    pub(crate) fn event_length(&self) -> usize {
        self.weights
            .first()
            .map(|(pauli, _)| pauli.len())
            .unwrap_or(0)
    }

    #[getter]
    pub(crate) fn total_weight(&self) -> f64 {
        self.weights.iter().map(|(_, weight)| *weight).sum()
    }

    pub(crate) fn sample(&self, rng: &Bound<'_, PyAny>, rate: f64) -> PyResult<String> {
        validate_rate(rate)?;
        if rng.call_method0("random")?.extract::<f64>()? >= rate {
            return Ok("I".repeat(self.event_length()));
        }

        let threshold = rng.call_method0("random")?.extract::<f64>()? * self.total_weight();
        let mut acc = 0.0;
        for (pauli, weight) in &self.weights {
            if *weight == 0.0 {
                continue;
            }
            acc += *weight;
            if threshold <= acc {
                return Ok(pauli.clone());
            }
        }
        Ok(self
            .weights
            .last()
            .map(|(pauli, _)| pauli.clone())
            .unwrap_or_default())
    }

    pub(crate) fn score(&self, event: &Bound<'_, PyAny>, rate: f64) -> PyResult<f64> {
        let rate = score_rate(rate);
        let event = py_str(event)?;
        if event == "I".repeat(self.event_length()) {
            Ok(-1.0 / (1.0 - rate))
        } else {
            Ok(1.0 / rate)
        }
    }

    pub(crate) fn apply(
        &self,
        py: Python<'_>,
        event: &Bound<'_, PyAny>,
        state: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
        qubits: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        apply_pauli_event(py, event, state, frame, qubits, None)
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let weights = self.weights(py)?.bind(py).repr()?.to_str()?.to_string();
        Ok(format!("PauliChannel(weights={weights})"))
    }
}

#[pyclass(
    name = "SingleQubitDepolarizing",
    module = "npsim._npsim_native",
    frozen
)]
pub(crate) struct PySingleQubitDepolarizing;

#[pymethods]
impl PySingleQubitDepolarizing {
    #[new]
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn sample(&self, rng: &Bound<'_, PyAny>, rate: f64) -> PyResult<String> {
        validate_rate(rate)?;
        if rng.call_method0("random")?.extract::<f64>()? >= rate {
            return Ok("I".to_string());
        }
        let index = rng.call_method1("randrange", (3,))?.extract::<usize>()?;
        Ok(["X", "Y", "Z"][index].to_string())
    }

    pub(crate) fn score(&self, event: &Bound<'_, PyAny>, rate: f64) -> PyResult<f64> {
        let rate = score_rate(rate);
        if py_str(event)? == "I" {
            Ok(-1.0 / (1.0 - rate))
        } else {
            Ok(1.0 / rate)
        }
    }

    pub(crate) fn apply(
        &self,
        py: Python<'_>,
        event: &Bound<'_, PyAny>,
        state: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
        qubits: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        apply_pauli_event(
            py,
            event,
            state,
            frame,
            qubits,
            Some((1, "single-qubit depolarizing noise requires one qubit")),
        )
    }

    pub(crate) fn __repr__(&self) -> &'static str {
        "SingleQubitDepolarizing()"
    }
}

#[pyclass(name = "TwoQubitDepolarizing", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyTwoQubitDepolarizing {
    events: Vec<String>,
}

#[pymethods]
impl PyTwoQubitDepolarizing {
    #[new]
    #[pyo3(signature = (_events=None))]
    pub(crate) fn new(_events: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let events = match _events {
            Some(events) if !events.is_none() => events.extract::<Vec<String>>()?,
            _ => TWO_QUBIT_EVENTS
                .iter()
                .map(|event| event.to_string())
                .collect(),
        };
        Ok(Self { events })
    }

    #[getter]
    pub(crate) fn _events(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.events.iter())?.into())
    }

    pub(crate) fn sample(&self, rng: &Bound<'_, PyAny>, rate: f64) -> PyResult<String> {
        validate_rate(rate)?;
        if rng.call_method0("random")?.extract::<f64>()? >= rate {
            return Ok("II".to_string());
        }
        let index = rng
            .call_method1("randrange", (self.events.len(),))?
            .extract::<usize>()?;
        Ok(self.events[index].clone())
    }

    pub(crate) fn score(&self, event: &Bound<'_, PyAny>, rate: f64) -> PyResult<f64> {
        let rate = score_rate(rate);
        if py_str(event)? == "II" {
            Ok(-1.0 / (1.0 - rate))
        } else {
            Ok(1.0 / rate)
        }
    }

    pub(crate) fn apply(
        &self,
        py: Python<'_>,
        event: &Bound<'_, PyAny>,
        state: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
        qubits: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        apply_pauli_event(
            py,
            event,
            state,
            frame,
            qubits,
            Some((2, "two-qubit depolarizing noise requires two qubits")),
        )
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let events = self._events(py)?.bind(py).repr()?.to_str()?.to_string();
        Ok(format!("TwoQubitDepolarizing(_events={events})"))
    }
}

#[pyclass(name = "MeasurementBitFlip", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyMeasurementBitFlip;

#[pymethods]
impl PyMeasurementBitFlip {
    #[new]
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn sample(&self, rng: &Bound<'_, PyAny>, rate: f64) -> PyResult<bool> {
        validate_rate(rate)?;
        Ok(rng.call_method0("random")?.extract::<f64>()? < rate)
    }

    pub(crate) fn score(&self, event: &Bound<'_, PyAny>, rate: f64) -> PyResult<f64> {
        let rate = score_rate(rate);
        if event.is_truthy()? {
            Ok(1.0 / rate)
        } else {
            Ok(-1.0 / (1.0 - rate))
        }
    }

    pub(crate) fn apply(
        &self,
        event: &Bound<'_, PyAny>,
        state: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
        qubits: &Bound<'_, PyAny>,
    ) {
        let _ = (event, state, frame, qubits);
    }

    pub(crate) fn apply_to_bit(&self, bit: i64, event: &Bound<'_, PyAny>) -> PyResult<i64> {
        Ok(bit ^ i64::from(event.is_truthy()?))
    }

    pub(crate) fn __repr__(&self) -> &'static str {
        "MeasurementBitFlip()"
    }
}

fn validate_rate(rate: f64) -> PyResult<()> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(PyValueError::new_err(format!(
            "noise rate must be in [0, 1], got {rate}"
        )));
    }
    Ok(())
}

fn score_rate(rate: f64) -> f64 {
    rate.clamp(MIN_SCORE_RATE, 1.0 - MIN_SCORE_RATE)
}

fn py_str(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(value.str()?.to_str()?.to_string())
}

fn coerce_weights(weights: &Bound<'_, PyAny>) -> PyResult<Vec<(String, f64)>> {
    let py = weights.py();
    let dict = PyDict::new(py);
    dict.call_method1("update", (weights,))?;
    let mut out = Vec::with_capacity(dict.len());
    for (pauli, weight) in dict.iter() {
        out.push((pauli.extract::<String>()?, weight.extract::<f64>()?));
    }
    Ok(out)
}

fn validate_pauli_channel(weights: &[(String, f64)]) -> PyResult<()> {
    if weights.is_empty() {
        return Err(PyValueError::new_err(
            "PauliChannel requires at least one non-identity event",
        ));
    }

    let mut total = 0.0;
    let mut length = None;
    for (pauli, weight) in weights {
        if let Some(bad) = pauli
            .chars()
            .find(|value| !matches!(value, 'I' | 'X' | 'Y' | 'Z'))
        {
            let _ = bad;
            return Err(PyValueError::new_err(format!(
                "unsupported Pauli string {pauli:?}"
            )));
        }
        if !pauli.is_empty() && pauli.chars().all(|value| value == 'I') {
            return Err(PyValueError::new_err(
                "identity should not appear in PauliChannel weights",
            ));
        }
        if *weight < 0.0 {
            return Err(PyValueError::new_err(
                "PauliChannel weights must be non-negative",
            ));
        }
        match length {
            None => length = Some(pauli.len()),
            Some(length) if pauli.len() != length => {
                return Err(PyValueError::new_err(
                    "all PauliChannel events must have the same length",
                ));
            }
            _ => {}
        }
        total += *weight;
    }

    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "PauliChannel weights must have positive total weight",
        ));
    }
    Ok(())
}

fn apply_pauli_event(
    py: Python<'_>,
    event: &Bound<'_, PyAny>,
    state: &Bound<'_, PyAny>,
    frame: &Bound<'_, PyAny>,
    qubits: &Bound<'_, PyAny>,
    required_qubits: Option<(usize, &'static str)>,
) -> PyResult<()> {
    let pauli = py_str(event)?;
    let qubits = qubits.extract::<Vec<usize>>()?;
    if let Some((required_len, message)) = required_qubits {
        if qubits.len() != required_len {
            return Err(PyValueError::new_err(message));
        }
    }
    if pauli.len() != qubits.len() {
        return Err(PyValueError::new_err(
            "event Pauli length does not match qubits",
        ));
    }
    let n_qubits = state.getattr("n_qubits")?.extract::<usize>()?;
    let (x, z) = npsim_core::sparse_pauli_to_xz(n_qubits, &qubits, &pauli)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    state.call_method1(
        "apply_pauli_string",
        (PyList::new(py, x)?, PyList::new(py, z)?),
    )?;
    frame.call_method1("apply_pauli_string", (PyTuple::new(py, qubits)?, pauli))?;
    Ok(())
}

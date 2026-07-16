use crate::*;

#[pyclass(name = "PauliFrame", module = "faultscope._native")]
#[derive(Clone)]
pub(crate) struct PyPauliFrame {
    pub(crate) frame: faultscope_core::PauliFrame,
}

#[pymethods]
impl PyPauliFrame {
    #[new]
    pub(crate) fn new(x: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x = u8_vector(x, "x")?;
        let z = u8_vector(z, "z")?;
        Ok(Self {
            frame: core_value_error(faultscope_core::PauliFrame::new(x, z))?,
        })
    }

    #[staticmethod]
    pub(crate) fn zero(n_qubits: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            frame: faultscope_core::PauliFrame::zero(nonnegative_usize(n_qubits, "n_qubits")?),
        })
    }

    #[getter]
    pub(crate) fn x(&self) -> Vec<usize> {
        self.frame
            .x_bits()
            .iter()
            .copied()
            .map(usize::from)
            .collect()
    }

    #[getter]
    pub(crate) fn z(&self) -> Vec<usize> {
        self.frame
            .z_bits()
            .iter()
            .copied()
            .map(usize::from)
            .collect()
    }

    #[getter]
    pub(crate) fn n_qubits(&self) -> usize {
        self.frame.n_qubits()
    }

    pub(crate) fn copy(&self) -> Self {
        self.clone()
    }

    pub(crate) fn apply_pauli(&mut self, qubit: &Bound<'_, PyAny>, pauli: &str) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.frame.apply_pauli(qubit, pauli))
    }

    pub(crate) fn apply_pauli_string(
        &mut self,
        qubits: &Bound<'_, PyAny>,
        paulis: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let qubits = usize_vector(qubits, "qubits")?;
        let pauli = pauli_sequence_to_string(paulis)?;
        core_value_error(self.frame.apply_pauli_string(&qubits, &pauli))
    }

    pub(crate) fn apply_h(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.frame.apply_h(qubit))
    }

    pub(crate) fn apply_s(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.frame.apply_s(qubit))
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.frame.apply_s_dag(qubit))
    }

    pub(crate) fn apply_cx(
        &mut self,
        control: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let control = nonnegative_usize(control, "control")?;
        let target = nonnegative_usize(target, "target")?;
        core_value_error(self.frame.apply_cx(control, target))
    }

    pub(crate) fn apply_cz(
        &mut self,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let left = nonnegative_usize(left, "left")?;
        let right = nonnegative_usize(right, "right")?;
        core_value_error(self.frame.apply_cz(left, right))
    }

    pub(crate) fn apply_swap(
        &mut self,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let left = nonnegative_usize(left, "left")?;
        let right = nonnegative_usize(right, "right")?;
        core_value_error(self.frame.apply_swap(left, right))
    }

    pub(crate) fn reset(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.frame.reset(qubit))
    }

    pub(crate) fn measurement_flip(
        &self,
        qubits: &Bound<'_, PyAny>,
        paulis: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        let qubits = usize_vector(qubits, "qubits")?;
        let pauli = pauli_sequence_to_string(paulis)?;
        core_value_error(self.frame.measurement_flip(&qubits, &pauli)).map(u8::from)
    }

    pub(crate) fn pauli_on(&self, qubits: &Bound<'_, PyAny>) -> PyResult<String> {
        let qubits = usize_vector(qubits, "qubits")?;
        core_value_error(self.frame.pauli_on(&qubits))
    }
}

#[pyclass(name = "StabilizerState", module = "faultscope._native")]
#[derive(Clone)]
pub(crate) struct PyStabilizerState {
    pub(crate) state: faultscope_core::ConcreteStabilizer,
}

#[pymethods]
impl PyStabilizerState {
    #[staticmethod]
    pub(crate) fn zero(n_qubits: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            state: faultscope_core::ConcreteStabilizer::zero(nonnegative_usize(
                n_qubits, "n_qubits",
            )?),
        })
    }

    #[getter]
    pub(crate) fn x(&self) -> Vec<Vec<usize>> {
        self.state
            .x_rows()
            .iter()
            .map(|row| row.iter().map(|value| usize::from(*value)).collect())
            .collect()
    }

    #[getter]
    pub(crate) fn z(&self) -> Vec<Vec<usize>> {
        self.state
            .z_rows()
            .iter()
            .map(|row| row.iter().map(|value| usize::from(*value)).collect())
            .collect()
    }

    #[getter]
    pub(crate) fn sign(&self) -> Vec<usize> {
        self.state
            .sign_bits()
            .iter()
            .copied()
            .map(usize::from)
            .collect()
    }

    #[getter]
    pub(crate) fn n_qubits(&self) -> usize {
        self.state.n_qubits()
    }

    pub(crate) fn copy(&self) -> Self {
        self.clone()
    }

    pub(crate) fn apply_h(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.state.apply_h(qubit))
    }

    pub(crate) fn apply_s(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.state.apply_s(qubit))
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.state.apply_s_dag(qubit))
    }

    pub(crate) fn apply_cx(
        &mut self,
        control: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let control = nonnegative_usize(control, "control")?;
        let target = nonnegative_usize(target, "target")?;
        core_value_error(self.state.apply_cx(control, target))
    }

    pub(crate) fn apply_cz(
        &mut self,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let left = nonnegative_usize(left, "left")?;
        let right = nonnegative_usize(right, "right")?;
        core_value_error(self.state.apply_cz(left, right))
    }

    pub(crate) fn apply_swap(
        &mut self,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let left = nonnegative_usize(left, "left")?;
        let right = nonnegative_usize(right, "right")?;
        core_value_error(self.state.apply_swap(left, right))
    }

    pub(crate) fn apply_pauli(&mut self, qubit: &Bound<'_, PyAny>, pauli: &str) -> PyResult<()> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        core_value_error(self.state.apply_pauli(qubit, pauli))
    }

    #[pyo3(signature = (x, z=None))]
    pub(crate) fn apply_pauli_string(
        &mut self,
        x: &Bound<'_, PyAny>,
        z: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let z = z.ok_or_else(|| PyValueError::new_err("x and z vectors are required"))?;
        let x = u8_vector(x, "x")?;
        let z = u8_vector(z, "z")?;
        core_value_error(self.state.apply_pauli_string(&x, &z))
    }

    pub(crate) fn measure_z(
        &mut self,
        qubit: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        let (x, z) = single_pauli_bits(self.state.n_qubits(), qubit, 'Z')?;
        self.measure_pauli_bits(&x, &z, rng)
    }

    pub(crate) fn measure_x(
        &mut self,
        qubit: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        let (x, z) = single_pauli_bits(self.state.n_qubits(), qubit, 'X')?;
        self.measure_pauli_bits(&x, &z, rng)
    }

    pub(crate) fn measure_y(
        &mut self,
        qubit: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        let (x, z) = single_pauli_bits(self.state.n_qubits(), qubit, 'Y')?;
        self.measure_pauli_bits(&x, &z, rng)
    }

    pub(crate) fn measure_pauli(
        &mut self,
        x: &Bound<'_, PyAny>,
        z: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        let x = u8_vector(x, "x")?;
        let z = u8_vector(z, "z")?;
        self.measure_pauli_bits(&x, &z, rng)
    }

    pub(crate) fn is_deterministic_pauli(
        &self,
        x: &Bound<'_, PyAny>,
        z: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let x = u8_vector(x, "x")?;
        let z = u8_vector(z, "z")?;
        core_value_error(self.state.is_deterministic_pauli(&x, &z))
    }

    pub(crate) fn deterministic_measurement_bit(
        &self,
        x: &Bound<'_, PyAny>,
        z: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        let x = u8_vector(x, "x")?;
        let z = u8_vector(z, "z")?;
        core_value_error(self.state.deterministic_measurement_bit(&x, &z)).map(u8::from)
    }

    pub(crate) fn reset_z(
        &mut self,
        qubit: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        self.reset_basis(qubit, rng, 'Z', "X")
    }

    pub(crate) fn reset_x(
        &mut self,
        qubit: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        self.reset_basis(qubit, rng, 'X', "Z")
    }

    pub(crate) fn reset_y(
        &mut self,
        qubit: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        self.reset_basis(qubit, rng, 'Y', "X")
    }
}

enum PyMeasurementError {
    Core(faultscope_core::NpError),
    Python(PyErr),
}

impl From<faultscope_core::NpError> for PyMeasurementError {
    fn from(error: faultscope_core::NpError) -> Self {
        Self::Core(error)
    }
}

impl From<PyErr> for PyMeasurementError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

impl PyMeasurementError {
    fn into_pyerr(self) -> PyErr {
        match self {
            Self::Core(error) => PyValueError::new_err(error.to_string()),
            Self::Python(error) => error,
        }
    }
}

impl PyStabilizerState {
    fn measure_pauli_bits(&mut self, x: &[u8], z: &[u8], rng: &Bound<'_, PyAny>) -> PyResult<u8> {
        let result: Result<bool, PyMeasurementError> = self.state.measure_pauli_with(x, z, || {
            let outcome = rng
                .call_method1("randrange", (2,))
                .and_then(|value| value.extract::<u8>())?;
            Ok(outcome != 0)
        });
        result.map(u8::from).map_err(PyMeasurementError::into_pyerr)
    }

    fn reset_basis(
        &mut self,
        qubit: &Bound<'_, PyAny>,
        rng: &Bound<'_, PyAny>,
        basis: char,
        correction: &str,
    ) -> PyResult<u8> {
        let qubit = nonnegative_usize(qubit, "qubit")?;
        let (x, z) = single_pauli_bits(self.state.n_qubits(), qubit, basis)?;
        let outcome = self.measure_pauli_bits(&x, &z, rng)?;
        if outcome != 0 {
            core_value_error(self.state.apply_pauli(qubit, correction))?;
        }
        Ok(outcome)
    }
}

fn core_value_error<T>(result: faultscope_core::NpResult<T>) -> PyResult<T> {
    result.map_err(|error| PyValueError::new_err(error.to_string()))
}

fn pauli_sequence_to_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(value) = value.extract::<String>() {
        return Ok(value);
    }
    let values = value
        .extract::<Vec<String>>()
        .map_err(|_| PyValueError::new_err("paulis must be a string or sequence of strings"))?;
    Ok(values.join(""))
}

fn u8_vector(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u8>> {
    let values = value
        .extract::<Vec<i64>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a sequence of integers")))?;
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            u8::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "{name} value at index {index} is outside the supported integer range"
                ))
            })
        })
        .collect()
}

fn nonnegative_usize(value: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    let value = value
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be an integer")))?;
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{name} must be non-negative")))
}

pub(crate) fn usize_vector(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<usize>> {
    let values = value
        .extract::<Vec<i64>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a sequence of integers")))?;
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            usize::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "{name} value at index {index} must be non-negative"
                ))
            })
        })
        .collect()
}

fn single_pauli_bits(n_qubits: usize, qubit: usize, pauli: char) -> PyResult<(Vec<u8>, Vec<u8>)> {
    if qubit >= n_qubits {
        return Err(PyValueError::new_err(format!(
            "Pauli target qubit {qubit} is out of range for {n_qubits} qubits"
        )));
    }
    let mut x = vec![0; n_qubits];
    let mut z = vec![0; n_qubits];
    let (x_bit, z_bit) = match pauli {
        'X' => (1, 0),
        'Y' => (1, 1),
        'Z' => (0, 1),
        _ => unreachable!("single_pauli_bits is called only with X, Y, or Z"),
    };
    x[qubit] = x_bit;
    z[qubit] = z_bit;
    Ok((x, z))
}

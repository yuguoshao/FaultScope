use crate::*;

#[pyclass(name = "PauliFrame", module = "faultscope._native")]
#[derive(Clone)]
pub(crate) struct PyPauliFrame {
    x: Vec<u8>,
    z: Vec<u8>,
}

#[pymethods]
impl PyPauliFrame {
    #[new]
    pub(crate) fn new(x: Vec<u8>, z: Vec<u8>) -> PyResult<Self> {
        if x.len() != z.len() {
            return Err(PyValueError::new_err("x and z must have the same length"));
        }
        Ok(Self { x, z })
    }

    #[staticmethod]
    pub(crate) fn zero(n_qubits: usize) -> Self {
        Self {
            x: vec![0; n_qubits],
            z: vec![0; n_qubits],
        }
    }

    #[getter]
    pub(crate) fn x(&self) -> Vec<usize> {
        self.x.iter().map(|value| usize::from(*value)).collect()
    }

    #[getter]
    pub(crate) fn z(&self) -> Vec<usize> {
        self.z.iter().map(|value| usize::from(*value)).collect()
    }

    #[getter]
    pub(crate) fn n_qubits(&self) -> usize {
        self.x.len()
    }

    pub(crate) fn copy(&self) -> Self {
        self.clone()
    }

    pub(crate) fn apply_pauli(&mut self, qubit: usize, pauli: &str) -> PyResult<()> {
        faultscope_core::frame_apply_pauli_string(&mut self.x, &mut self.z, &[qubit], pauli)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    pub(crate) fn apply_pauli_string(
        &mut self,
        qubits: Vec<usize>,
        paulis: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let pauli = pauli_sequence_to_string(paulis)?;
        faultscope_core::frame_apply_pauli_string(&mut self.x, &mut self.z, &qubits, &pauli)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    pub(crate) fn apply_h(&mut self, qubit: usize) -> PyResult<()> {
        validate_public_qubit(self.x.len(), qubit, "PauliFrame.apply_h")?;
        faultscope_core::frame_apply_h(&mut self.x, &mut self.z, qubit);
        Ok(())
    }

    pub(crate) fn apply_s(&mut self, qubit: usize) -> PyResult<()> {
        validate_public_qubit(self.x.len(), qubit, "PauliFrame.apply_s")?;
        faultscope_core::frame_apply_s(&mut self.x, &mut self.z, qubit);
        Ok(())
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: usize) -> PyResult<()> {
        validate_public_qubit(self.x.len(), qubit, "PauliFrame.apply_s_dag")?;
        faultscope_core::frame_apply_s(&mut self.x, &mut self.z, qubit);
        Ok(())
    }

    pub(crate) fn apply_cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        validate_public_pair(self.x.len(), control, target, "PauliFrame.apply_cx")?;
        faultscope_core::frame_apply_cx(&mut self.x, &mut self.z, control, target);
        Ok(())
    }

    pub(crate) fn apply_cz(&mut self, left: usize, right: usize) -> PyResult<()> {
        validate_public_pair(self.x.len(), left, right, "PauliFrame.apply_cz")?;
        faultscope_core::frame_apply_cz(&mut self.x, &mut self.z, left, right);
        Ok(())
    }

    pub(crate) fn apply_swap(&mut self, left: usize, right: usize) -> PyResult<()> {
        validate_public_pair(self.x.len(), left, right, "PauliFrame.apply_swap")?;
        faultscope_core::frame_apply_swap(&mut self.x, &mut self.z, left, right);
        Ok(())
    }

    pub(crate) fn reset(&mut self, qubit: usize) -> PyResult<()> {
        validate_public_qubit(self.x.len(), qubit, "PauliFrame.reset")?;
        self.x[qubit] = 0;
        self.z[qubit] = 0;
        Ok(())
    }

    pub(crate) fn measurement_flip(
        &self,
        qubits: Vec<usize>,
        paulis: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        let pauli = pauli_sequence_to_string(paulis)?;
        faultscope_core::frame_measurement_flip_bits(&self.x, &self.z, &qubits, &pauli)
            .map(u8::from)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    pub(crate) fn pauli_on(&self, qubits: Vec<usize>) -> PyResult<String> {
        for qubit in &qubits {
            validate_public_qubit(self.x.len(), *qubit, "PauliFrame.pauli_on")?;
        }
        Ok(qubits
            .iter()
            .map(|qubit| faultscope_core::xz_to_pauli(self.x[*qubit], self.z[*qubit]))
            .collect())
    }
}

#[pyclass(name = "StabilizerState", module = "faultscope._native")]
#[derive(Clone)]
pub(crate) struct PyStabilizerState {
    state: faultscope_core::ConcreteStabilizer,
}

#[pymethods]
impl PyStabilizerState {
    #[new]
    pub(crate) fn new(x: Vec<Vec<u8>>, z: Vec<Vec<u8>>, sign: Vec<u8>) -> PyResult<Self> {
        let sign = sign.into_iter().map(|value| value != 0).collect();
        let state = faultscope_core::ConcreteStabilizer::new(x, z, sign)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self { state })
    }

    #[staticmethod]
    pub(crate) fn zero(n_qubits: usize) -> Self {
        Self {
            state: faultscope_core::ConcreteStabilizer::zero(n_qubits),
        }
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

    pub(crate) fn apply_h(&mut self, qubit: usize) -> PyResult<()> {
        validate_public_qubit(self.state.n_qubits(), qubit, "StabilizerState.apply_h")?;
        self.state.apply_h(qubit);
        Ok(())
    }

    pub(crate) fn apply_s(&mut self, qubit: usize) -> PyResult<()> {
        validate_public_qubit(self.state.n_qubits(), qubit, "StabilizerState.apply_s")?;
        self.state.apply_s(qubit);
        Ok(())
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: usize) -> PyResult<()> {
        validate_public_qubit(self.state.n_qubits(), qubit, "StabilizerState.apply_s_dag")?;
        self.state.apply_s_dag(qubit);
        Ok(())
    }

    pub(crate) fn apply_cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        validate_public_pair(
            self.state.n_qubits(),
            control,
            target,
            "StabilizerState.apply_cx",
        )?;
        self.state.apply_cx(control, target);
        Ok(())
    }

    pub(crate) fn apply_cz(&mut self, left: usize, right: usize) -> PyResult<()> {
        validate_public_pair(
            self.state.n_qubits(),
            left,
            right,
            "StabilizerState.apply_cz",
        )?;
        self.state.apply_cz(left, right);
        Ok(())
    }

    pub(crate) fn apply_swap(&mut self, left: usize, right: usize) -> PyResult<()> {
        validate_public_pair(
            self.state.n_qubits(),
            left,
            right,
            "StabilizerState.apply_swap",
        )?;
        self.state.apply_swap(left, right);
        Ok(())
    }

    pub(crate) fn apply_pauli(&mut self, qubit: usize, pauli: &str) -> PyResult<()> {
        let (x, z) = faultscope_core::sparse_pauli_to_xz(self.state.n_qubits(), &[qubit], pauli)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.state.apply_pauli_string(&x, &z);
        Ok(())
    }

    #[pyo3(signature = (x, z=None))]
    pub(crate) fn apply_pauli_string(&mut self, x: Vec<u8>, z: Option<Vec<u8>>) -> PyResult<()> {
        let z = z.ok_or_else(|| PyValueError::new_err("x and z vectors are required"))?;
        self.state.apply_pauli_string(&x, &z);
        Ok(())
    }

    pub(crate) fn measure_z(&mut self, qubit: usize, rng: &Bound<'_, PyAny>) -> PyResult<u8> {
        let (x, z) = faultscope_core::sparse_pauli_to_xz(self.state.n_qubits(), &[qubit], "Z")
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.measure_pauli(x, z, rng)
    }

    pub(crate) fn measure_x(&mut self, qubit: usize, rng: &Bound<'_, PyAny>) -> PyResult<u8> {
        let (x, z) = faultscope_core::sparse_pauli_to_xz(self.state.n_qubits(), &[qubit], "X")
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.measure_pauli(x, z, rng)
    }

    pub(crate) fn measure_y(&mut self, qubit: usize, rng: &Bound<'_, PyAny>) -> PyResult<u8> {
        let (x, z) = faultscope_core::sparse_pauli_to_xz(self.state.n_qubits(), &[qubit], "Y")
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.measure_pauli(x, z, rng)
    }

    pub(crate) fn measure_pauli(
        &mut self,
        x: Vec<u8>,
        z: Vec<u8>,
        rng: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        if self.state.is_deterministic_pauli(&x, &z) {
            return self
                .state
                .deterministic_measurement_bit(&x, &z)
                .map(u8::from)
                .map_err(|err| PyValueError::new_err(err.to_string()));
        }
        let outcome = rng.call_method1("randrange", (2,))?.extract::<u8>()? != 0;
        self.state
            .measure_pauli_with_outcome(&x, &z, outcome)
            .map(u8::from)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    pub(crate) fn is_deterministic_pauli(&self, x: Vec<u8>, z: Vec<u8>) -> bool {
        self.state.is_deterministic_pauli(&x, &z)
    }

    pub(crate) fn deterministic_measurement_bit(&self, x: Vec<u8>, z: Vec<u8>) -> PyResult<u8> {
        if !self.state.is_deterministic_pauli(&x, &z) {
            return Err(PyValueError::new_err(
                "Pauli measurement is random for this stabilizer state",
            ));
        }
        self.state
            .deterministic_measurement_bit(&x, &z)
            .map(u8::from)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    pub(crate) fn reset_z(&mut self, qubit: usize, rng: &Bound<'_, PyAny>) -> PyResult<u8> {
        self.reset_basis(qubit, rng, "Z", "X")
    }

    pub(crate) fn reset_x(&mut self, qubit: usize, rng: &Bound<'_, PyAny>) -> PyResult<u8> {
        self.reset_basis(qubit, rng, "X", "Z")
    }

    pub(crate) fn reset_y(&mut self, qubit: usize, rng: &Bound<'_, PyAny>) -> PyResult<u8> {
        self.reset_basis(qubit, rng, "Y", "X")
    }
}

impl PyStabilizerState {
    fn reset_basis(
        &mut self,
        qubit: usize,
        rng: &Bound<'_, PyAny>,
        basis: &str,
        correction: &str,
    ) -> PyResult<u8> {
        let (x, z) = faultscope_core::sparse_pauli_to_xz(self.state.n_qubits(), &[qubit], basis)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let outcome = self.measure_pauli(x, z, rng)?;
        if outcome != 0 {
            self.apply_pauli(qubit, correction)?;
        }
        Ok(outcome)
    }
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

fn validate_public_qubit(n_qubits: usize, qubit: usize, context: &str) -> PyResult<()> {
    if qubit >= n_qubits {
        return Err(PyValueError::new_err(format!(
            "{context} targets qubit {qubit}, but only {n_qubits} qubits are available"
        )));
    }
    Ok(())
}

fn validate_public_pair(n_qubits: usize, left: usize, right: usize, context: &str) -> PyResult<()> {
    validate_public_qubit(n_qubits, left, context)?;
    validate_public_qubit(n_qubits, right, context)?;
    if left == right {
        return Err(PyValueError::new_err(format!(
            "{context} requires distinct qubit targets; duplicate qubit {left}"
        )));
    }
    Ok(())
}

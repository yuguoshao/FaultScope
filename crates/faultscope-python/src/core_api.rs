use crate::*;

/// Named stochastic noise source attached to circuit operations.
#[pyclass(name = "NoiseLocation", module = "faultscope._native", frozen)]
pub(crate) struct PyNoiseLocation {
    pub(crate) id: String,
    pub(crate) model: Py<PyAny>,
    pub(crate) rate: f64,
    pub(crate) qubits: Vec<usize>,
    pub(crate) tags: Py<PyAny>,
    pub(crate) core_tags: Option<HashMap<String, TagValue>>,
}

#[pymethods]
impl PyNoiseLocation {
    #[new]
    #[pyo3(signature = (id, model, rate, qubits, tags=None))]
    pub(crate) fn new(
        py: Python<'_>,
        id: String,
        model: Py<PyAny>,
        rate: f64,
        qubits: Vec<usize>,
        tags: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(PyValueError::new_err(format!(
                "noise rate must be in [0, 1], got {rate}"
            )));
        }
        let tags = mapping_to_dict(py, tags)?;
        let core_tags = parse_tags_mapping(tags.bind(py)).ok();
        Ok(Self {
            id,
            model,
            rate,
            qubits,
            tags,
            core_tags,
        })
    }

    #[getter]
    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    #[getter]
    pub(crate) fn model(&self, py: Python<'_>) -> PyObject {
        self.model.clone_ref(py)
    }

    #[getter]
    pub(crate) fn rate(&self) -> f64 {
        self.rate
    }

    #[getter]
    pub(crate) fn qubits(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize(py, &self.qubits)
    }

    #[getter]
    pub(crate) fn tags(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.tags.bind(py).call_method0("copy")?.unbind())
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let model_repr = self.model.bind(py).repr()?.to_str()?.to_string();
        let qubits_repr = tuple_usize(py, &self.qubits)?
            .bind(py)
            .repr()?
            .to_str()?
            .to_string();
        let tags_repr = self.tags.bind(py).repr()?.to_str()?.to_string();
        Ok(format!(
            "NoiseLocation(id={:?}, model={}, rate={}, qubits={}, tags={})",
            self.id, model_repr, self.rate, qubits_repr, tags_repr,
        ))
    }
}

/// Stabilizer-compatible circuit operation.
#[pyclass(name = "Operation", module = "faultscope._native", frozen)]
pub(crate) struct PyOperation {
    pub(crate) kind: String,
    pub(crate) qubits: Vec<usize>,
    pub(crate) key: Option<String>,
    pub(crate) basis: String,
    pub(crate) pauli: Option<String>,
    pub(crate) measurement_keys: Vec<String>,
    pub(crate) observable_id: Option<i64>,
    pub(crate) noise_location: Option<Py<PyAny>>,
    pub(crate) metadata: Py<PyAny>,
    pub(crate) repeat_count: Option<usize>,
    pub(crate) body: Vec<Py<PyAny>>,
    pub(crate) record_lookbacks: Vec<usize>,
    pub(crate) core_op: Option<faultscope_core::Operation>,
}

#[pymethods]
impl PyOperation {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        kind,
        qubits=None,
        key=None,
        basis=None,
        pauli=None,
        measurement_keys=None,
        observable_id=None,
        noise_location=None,
        metadata=None
    ))]
    pub(crate) fn new(
        py: Python<'_>,
        kind: String,
        qubits: Option<Vec<usize>>,
        key: Option<String>,
        basis: Option<String>,
        pauli: Option<String>,
        measurement_keys: Option<Vec<String>>,
        observable_id: Option<i64>,
        noise_location: Option<Py<PyAny>>,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind,
            qubits: qubits.unwrap_or_default(),
            key,
            basis: basis.unwrap_or_else(|| "Z".to_string()),
            pauli,
            measurement_keys: measurement_keys.unwrap_or_default(),
            observable_id,
            noise_location,
            metadata: mapping_to_dict(py, metadata)?,
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, **metadata))]
    pub(crate) fn h(
        py: Python<'_>,
        qubit: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::simple_gate(py, "h", vec![qubit], metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, **metadata))]
    pub(crate) fn s(
        py: Python<'_>,
        qubit: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::simple_gate(py, "s", vec![qubit], metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, **metadata))]
    pub(crate) fn s_dag(
        py: Python<'_>,
        qubit: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::simple_gate(py, "s_dag", vec![qubit], metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, **metadata))]
    pub(crate) fn x(
        py: Python<'_>,
        qubit: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::pauli_gate(py, vec![qubit], "X".to_string(), metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, **metadata))]
    pub(crate) fn y(
        py: Python<'_>,
        qubit: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::pauli_gate(py, vec![qubit], "Y".to_string(), metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, **metadata))]
    pub(crate) fn z(
        py: Python<'_>,
        qubit: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::pauli_gate(py, vec![qubit], "Z".to_string(), metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (control, target, **metadata))]
    pub(crate) fn cx(
        py: Python<'_>,
        control: usize,
        target: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::simple_gate(py, "cx", vec![control, target], metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (left, right, **metadata))]
    pub(crate) fn cz(
        py: Python<'_>,
        left: usize,
        right: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::simple_gate(py, "cz", vec![left, right], metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (left, right, **metadata))]
    pub(crate) fn swap(
        py: Python<'_>,
        left: usize,
        right: usize,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Self::simple_gate(py, "swap", vec![left, right], metadata)
    }

    #[staticmethod]
    #[pyo3(signature = (qubits, pauli, **metadata))]
    pub(crate) fn pauli_gate(
        py: Python<'_>,
        qubits: Vec<usize>,
        pauli: String,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: "pauli".to_string(),
            qubits: qubits.clone(),
            key: None,
            basis: "Z".to_string(),
            pauli: Some(pauli.clone()),
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: Some(faultscope_core::Operation::Pauli { qubits, pauli }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (location, **metadata))]
    pub(crate) fn noise(
        py: Python<'_>,
        location: Py<PyAny>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let core_location = cache_safe_native_noise_location(py, &location);
        let qubits = match &core_location {
            Some(location) => location.qubits.clone(),
            None => location
                .bind(py)
                .getattr("qubits")?
                .extract::<Vec<usize>>()?,
        };
        Ok(Self {
            kind: "noise".to_string(),
            qubits,
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: Some(location),
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: core_location.map(faultscope_core::Operation::Noise),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, *, key=None, basis="Z", noise=None, **metadata))]
    pub(crate) fn measure(
        py: Python<'_>,
        qubit: usize,
        key: Option<String>,
        basis: &str,
        noise: Option<Py<PyAny>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let core_op = cache_safe_optional_noise_location(py, &noise).map(|core_noise| {
            faultscope_core::Operation::Measure {
                qubit,
                key: key.clone(),
                basis: basis.to_uppercase(),
                noise: core_noise,
            }
        });
        Ok(Self {
            kind: "measure".to_string(),
            qubits: vec![qubit],
            key: key.clone(),
            basis: basis.to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: noise,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (qubits, pauli, *, key=None, noise=None, **metadata))]
    pub(crate) fn measure_pauli(
        py: Python<'_>,
        qubits: Vec<usize>,
        pauli: String,
        key: Option<String>,
        noise: Option<Py<PyAny>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let core_op = cache_safe_optional_noise_location(py, &noise).map(|core_noise| {
            faultscope_core::Operation::MeasurePauli {
                qubits: qubits.clone(),
                pauli: pauli.clone(),
                key: key.clone(),
                noise: core_noise,
            }
        });
        Ok(Self {
            kind: "measure_pauli".to_string(),
            qubits: qubits.clone(),
            key: key.clone(),
            basis: "Z".to_string(),
            pauli: Some(pauli.clone()),
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: noise,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, *, key=None, basis="Z", **metadata))]
    pub(crate) fn reset(
        py: Python<'_>,
        qubit: usize,
        key: Option<String>,
        basis: &str,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: "reset".to_string(),
            qubits: vec![qubit],
            key: key.clone(),
            basis: basis.to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: Some(faultscope_core::Operation::Reset {
                qubit,
                key,
                basis: basis.to_uppercase(),
            }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, *, basis="Z", **metadata))]
    pub(crate) fn measure_reset(
        py: Python<'_>,
        qubit: usize,
        basis: &str,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: "measure_reset".to_string(),
            qubits: vec![qubit],
            key: None,
            basis: basis.to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: Some(faultscope_core::Operation::MeasureReset {
                qubit,
                basis: basis.to_uppercase(),
            }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (measurement_keys, *, detector_id=None, coords=None, **metadata))]
    pub(crate) fn detector(
        py: Python<'_>,
        measurement_keys: Vec<String>,
        detector_id: Option<i64>,
        coords: Option<Vec<f64>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let metadata = kwargs_to_dict(py, metadata)?;
        match detector_id {
            Some(detector_id) => metadata.set_item("detector_id", detector_id)?,
            None => metadata.set_item("detector_id", py.None())?,
        }
        let coords = coords.unwrap_or_default();
        metadata.set_item("coords", PyTuple::new(py, &coords)?)?;
        Ok(Self {
            kind: "detector".to_string(),
            qubits: Vec::new(),
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: measurement_keys.clone(),
            observable_id: None,
            noise_location: None,
            metadata: metadata.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: Some(faultscope_core::Operation::Detector {
                detector_id,
                measurement_keys,
                coords,
            }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (observable_id, measurement_keys, **metadata))]
    pub(crate) fn observable_include(
        py: Python<'_>,
        observable_id: i64,
        measurement_keys: Vec<String>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: "observable_include".to_string(),
            qubits: Vec::new(),
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: measurement_keys.clone(),
            observable_id: Some(observable_id),
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: Some(faultscope_core::Operation::ObservableInclude {
                observable_id,
                measurement_keys,
            }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (**metadata))]
    pub(crate) fn tick(py: Python<'_>, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::structured(
            py,
            "tick",
            None,
            Vec::new(),
            Vec::new(),
            metadata,
            faultscope_core::Operation::Tick,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (offsets, **metadata))]
    pub(crate) fn shift_coords(
        py: Python<'_>,
        offsets: Vec<f64>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let metadata_dict = kwargs_to_dict(py, metadata)?;
        metadata_dict.set_item("offsets", PyTuple::new(py, &offsets)?)?;
        Ok(Self {
            kind: "shift_coords".to_string(),
            qubits: Vec::new(),
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: metadata_dict.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op: Some(faultscope_core::Operation::ShiftCoords(offsets)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (count, operations, **metadata))]
    pub(crate) fn repeat(
        py: Python<'_>,
        count: usize,
        operations: Vec<Py<PyAny>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if count == 0 {
            return Err(PyValueError::new_err("repeat count must be positive"));
        }
        let core_body = operations
            .iter()
            .map(|operation| {
                operation
                    .bind(py)
                    .extract::<PyRef<'_, PyOperation>>()
                    .ok()?
                    .core_op
                    .clone()
            })
            .collect::<Option<Vec<_>>>();
        Ok(Self {
            kind: "repeat".to_string(),
            qubits: Vec::new(),
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: Some(count),
            body: operations,
            record_lookbacks: Vec::new(),
            core_op: core_body.map(|body| faultscope_core::Operation::Repeat { count, body }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (lookbacks, *, detector_id=None, coords=None, **metadata))]
    pub(crate) fn detector_rec(
        py: Python<'_>,
        lookbacks: Vec<usize>,
        detector_id: Option<i64>,
        coords: Option<Vec<f64>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        validate_lookbacks(&lookbacks)?;
        let metadata_dict = kwargs_to_dict(py, metadata)?;
        metadata_dict.set_item("detector_id", detector_id)?;
        let coords = coords.unwrap_or_default();
        metadata_dict.set_item("coords", PyTuple::new(py, &coords)?)?;
        Ok(Self {
            kind: "detector_rec".to_string(),
            qubits: Vec::new(),
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: metadata_dict.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: lookbacks.clone(),
            core_op: Some(faultscope_core::Operation::DetectorRec {
                detector_id,
                lookbacks,
                coords,
            }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (observable_id, lookbacks, **metadata))]
    pub(crate) fn observable_include_rec(
        py: Python<'_>,
        observable_id: i64,
        lookbacks: Vec<usize>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        validate_lookbacks(&lookbacks)?;
        Ok(Self {
            kind: "observable_include_rec".to_string(),
            qubits: Vec::new(),
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: Some(observable_id),
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: lookbacks.clone(),
            core_op: Some(faultscope_core::Operation::ObservableIncludeRec {
                observable_id,
                lookbacks,
            }),
        })
    }

    #[getter]
    pub(crate) fn kind(&self) -> &str {
        &self.kind
    }

    #[getter]
    pub(crate) fn qubits(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize(py, &self.qubits)
    }

    #[getter]
    pub(crate) fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    #[getter]
    pub(crate) fn basis(&self) -> &str {
        &self.basis
    }

    #[getter]
    pub(crate) fn pauli(&self) -> Option<&str> {
        self.pauli.as_deref()
    }

    #[getter]
    pub(crate) fn measurement_keys(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_string(py, &self.measurement_keys)
    }

    #[getter]
    pub(crate) fn observable_id(&self) -> Option<i64> {
        self.observable_id
    }

    #[getter]
    pub(crate) fn noise_location(&self, py: Python<'_>) -> Option<PyObject> {
        self.noise_location
            .as_ref()
            .map(|location| location.clone_ref(py))
    }

    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyObject {
        self.metadata.clone_ref(py)
    }

    #[getter]
    pub(crate) fn repeat_count(&self) -> Option<usize> {
        self.repeat_count
    }

    #[getter]
    pub(crate) fn body(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.body.iter().map(|item| item.clone_ref(py)))?.into())
    }

    #[getter]
    pub(crate) fn record_lookbacks(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize(py, &self.record_lookbacks)
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let qubits_repr = tuple_usize(py, &self.qubits)?
            .bind(py)
            .repr()?
            .to_str()?
            .to_string();
        let measurement_keys_repr = tuple_string(py, &self.measurement_keys)?
            .bind(py)
            .repr()?
            .to_str()?
            .to_string();
        let noise_location_repr = self
            .noise_location
            .as_ref()
            .map(|location| {
                location
                    .bind(py)
                    .repr()
                    .and_then(|repr| Ok(repr.to_str()?.to_string()))
            })
            .transpose()?
            .unwrap_or_else(|| "None".to_string());
        let metadata_repr = self.metadata.bind(py).repr()?.to_str()?.to_string();
        Ok(format!(
            "Operation(kind={:?}, qubits={}, key={:?}, basis={:?}, pauli={:?}, measurement_keys={}, observable_id={:?}, noise_location={}, metadata={})",
            self.kind,
            qubits_repr,
            self.key,
            self.basis,
            self.pauli,
            measurement_keys_repr,
            self.observable_id,
            noise_location_repr,
            metadata_repr,
        ))
    }
}

impl PyOperation {
    #[allow(clippy::too_many_arguments)]
    fn structured(
        py: Python<'_>,
        kind: &str,
        repeat_count: Option<usize>,
        body: Vec<Py<PyAny>>,
        record_lookbacks: Vec<usize>,
        metadata: Option<&Bound<'_, PyDict>>,
        core_op: faultscope_core::Operation,
    ) -> PyResult<Self> {
        Ok(Self {
            kind: kind.to_string(),
            qubits: Vec::new(),
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count,
            body,
            record_lookbacks,
            core_op: Some(core_op),
        })
    }

    fn simple_gate(
        py: Python<'_>,
        kind: &str,
        qubits: Vec<usize>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let core_op = match kind {
            "h" => Some(faultscope_core::Operation::H(qubits[0])),
            "s" => Some(faultscope_core::Operation::S(qubits[0])),
            "s_dag" => Some(faultscope_core::Operation::SDag(qubits[0])),
            "cx" => Some(faultscope_core::Operation::Cx(qubits[0], qubits[1])),
            "cz" => Some(faultscope_core::Operation::Cz(qubits[0], qubits[1])),
            "swap" => Some(faultscope_core::Operation::Swap(qubits[0], qubits[1])),
            _ => None,
        };
        Ok(Self {
            kind: kind.to_string(),
            qubits,
            key: None,
            basis: "Z".to_string(),
            pauli: None,
            measurement_keys: Vec::new(),
            observable_id: None,
            noise_location: None,
            metadata: kwargs_to_dict(py, metadata)?.into(),
            repeat_count: None,
            body: Vec::new(),
            record_lookbacks: Vec::new(),
            core_op,
        })
    }
}

fn validate_lookbacks(lookbacks: &[usize]) -> PyResult<()> {
    if lookbacks.contains(&0) {
        return Err(PyValueError::new_err(
            "measurement record lookbacks must be positive",
        ));
    }
    Ok(())
}

/// Ordered stabilizer circuit consumed by FaultScope runtimes.
#[pyclass(name = "Circuit", module = "faultscope._native", frozen)]
pub(crate) struct PyCircuit {
    pub(crate) n_qubits: usize,
    pub(crate) operations: Vec<Py<PyAny>>,
    pub(crate) core_circuit: Option<faultscope_core::ValidatedDemCircuit>,
}

#[pymethods]
impl PyCircuit {
    #[new]
    #[pyo3(signature = (n_qubits, operations))]
    pub(crate) fn new(
        py: Python<'_>,
        n_qubits: usize,
        operations: Vec<Py<PyAny>>,
    ) -> PyResult<Self> {
        let core_operations = operations
            .iter()
            .map(|operation| {
                let operation = operation
                    .bind(py)
                    .extract::<PyRef<'_, PyOperation>>()
                    .ok()?;
                operation.core_op.clone()
            })
            .collect::<Option<Vec<_>>>();

        // Cache-safe native operations are cloned and validated exactly once.
        // Preserve lazy handling for non-cache-safe Python objects while still
        // rejecting any fully parseable invalid target at construction time;
        // execution entry points parse and validate those objects again.
        let core_circuit = if let Some(core_operations) = core_operations {
            let circuit = faultscope_core::ValidatedDemCircuit::new(faultscope_core::Circuit {
                n_qubits,
                operations: core_operations,
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
            Some(circuit)
        } else {
            let parsed_operations = operations
                .iter()
                .map(|operation| parse_operation_object(operation.bind(py)))
                .collect::<PyResult<Vec<_>>>();
            if let Ok(parsed_operations) = parsed_operations {
                faultscope_core::Circuit {
                    n_qubits,
                    operations: parsed_operations,
                }
                .validate()
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            None
        };
        if let Some(validated_circuit) = &core_circuit {
            let circuit = validated_circuit.circuit();
            let structured = circuit.operations.iter().any(|operation| {
                matches!(
                    operation,
                    faultscope_core::Operation::Tick
                        | faultscope_core::Operation::ShiftCoords(_)
                        | faultscope_core::Operation::Repeat { .. }
                        | faultscope_core::Operation::DetectorRec { .. }
                        | faultscope_core::Operation::ObservableIncludeRec { .. }
                        | faultscope_core::Operation::MeasureReset { .. }
                )
            });
            if !structured {
                // Preserve eager plan construction for flat native circuits so
                // DEM compile timing only measures generator construction.
                let _ = validated_circuit.event_plan();
            }
        }
        Ok(Self {
            n_qubits,
            operations,
            core_circuit,
        })
    }

    #[getter]
    pub(crate) fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    #[getter]
    pub(crate) fn operations(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = self
            .operations
            .iter()
            .map(|operation| operation.clone_ref(py));
        Ok(PyTuple::new(py, items)?.into())
    }

    pub(crate) fn noise_locations(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        collect_py_noise_locations(py, &self.operations, &mut Vec::new(), &out)?;
        Ok(out.into())
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Circuit(n_qubits={}, operations={})",
            self.n_qubits,
            self.operations(py)?.bind(py).repr()?,
        ))
    }
}

fn collect_py_noise_locations(
    py: Python<'_>,
    operations: &[Py<PyAny>],
    repeat_path: &mut Vec<usize>,
    out: &Bound<'_, PyDict>,
) -> PyResult<()> {
    for operation in operations {
        let operation_bound = operation.bind(py);
        let kind = operation_bound.getattr("kind")?.extract::<String>()?;
        if kind == "repeat" {
            let operation_ref = operation_bound.extract::<PyRef<'_, PyOperation>>()?;
            let count = operation_ref
                .repeat_count
                .ok_or_else(|| PyValueError::new_err("repeat requires repeat_count"))?;
            for iteration in 0..count {
                repeat_path.push(iteration);
                collect_py_noise_locations(py, &operation_ref.body, repeat_path, out)?;
                repeat_path.pop();
            }
            continue;
        }
        if kind != "noise" && kind != "measure" && kind != "measure_pauli" {
            continue;
        }
        let location = operation_bound.getattr("noise_location")?;
        if location.is_none() {
            continue;
        }
        let base_id = location.getattr("id")?.extract::<String>()?;
        let location_id = if repeat_path.is_empty() {
            base_id
        } else {
            format!(
                "{}@r{}",
                base_id,
                repeat_path
                    .iter()
                    .map(|index| format!("[{index}]"))
                    .collect::<String>()
            )
        };
        if repeat_path.is_empty() {
            out.set_item(location_id, location)?;
        } else if let Ok(native) = location.extract::<PyRef<'_, PyNoiseLocation>>() {
            let cloned = Py::new(
                py,
                PyNoiseLocation {
                    id: location_id.clone(),
                    model: native.model.clone_ref(py),
                    rate: native.rate,
                    qubits: native.qubits.clone(),
                    tags: native.tags.clone_ref(py),
                    core_tags: native.core_tags.clone(),
                },
            )?;
            out.set_item(location_id, cloned)?;
        } else {
            out.set_item(location_id, location)?;
        }
    }
    Ok(())
}

/// Detector parity declaration over measurement keys.
#[pyclass(name = "Detector", module = "faultscope._native", frozen)]
pub(crate) struct PyDetector {
    id: i64,
    measurement_keys: Vec<String>,
    coords: Vec<f64>,
}

#[pymethods]
impl PyDetector {
    #[new]
    #[pyo3(signature = (id, measurement_keys, coords=None))]
    pub(crate) fn new(id: i64, measurement_keys: Vec<String>, coords: Option<Vec<f64>>) -> Self {
        Self {
            id,
            measurement_keys,
            coords: coords.unwrap_or_default(),
        }
    }

    #[getter]
    pub(crate) fn id(&self) -> i64 {
        self.id
    }

    #[getter]
    pub(crate) fn measurement_keys(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_string(py, &self.measurement_keys)
    }

    #[getter]
    pub(crate) fn coords(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.coords.iter().copied())?.into())
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let measurement_keys = tuple_string(py, &self.measurement_keys)?
            .bind(py)
            .repr()?
            .to_str()?
            .to_string();
        let coords = PyTuple::new(py, self.coords.iter().copied())?
            .repr()?
            .to_str()?
            .to_string();
        Ok(format!(
            "Detector(id={}, measurement_keys={}, coords={})",
            self.id, measurement_keys, coords
        ))
    }
}

/// Logical observable declaration from measurements or a Pauli projection.
#[pyclass(name = "LogicalObservable", module = "faultscope._native", frozen)]
pub(crate) struct PyLogicalObservable {
    id: i64,
    measurement_keys: Vec<String>,
    pauli_qubits: Vec<usize>,
    pauli: String,
}

#[pymethods]
impl PyLogicalObservable {
    #[new]
    #[pyo3(signature = (id, measurement_keys=None, pauli_qubits=None, pauli=""))]
    pub(crate) fn new(
        id: i64,
        measurement_keys: Option<Vec<String>>,
        pauli_qubits: Option<Vec<usize>>,
        pauli: &str,
    ) -> PyResult<Self> {
        let pauli_qubits = pauli_qubits.unwrap_or_default();
        if pauli_qubits.is_empty() != pauli.is_empty() {
            return Err(PyValueError::new_err(
                "pauli_qubits and pauli must be supplied together",
            ));
        }
        if !pauli.is_empty() && pauli_qubits.len() != pauli.len() {
            return Err(PyValueError::new_err(
                "pauli_qubits and pauli must have the same length",
            ));
        }
        Ok(Self {
            id,
            measurement_keys: measurement_keys.unwrap_or_default(),
            pauli_qubits,
            pauli: pauli.to_string(),
        })
    }

    #[getter]
    pub(crate) fn id(&self) -> i64 {
        self.id
    }

    #[getter]
    pub(crate) fn measurement_keys(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_string(py, &self.measurement_keys)
    }

    #[getter]
    pub(crate) fn pauli_qubits(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize(py, &self.pauli_qubits)
    }

    #[getter]
    pub(crate) fn pauli(&self) -> &str {
        &self.pauli
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let measurement_keys = tuple_string(py, &self.measurement_keys)?
            .bind(py)
            .repr()?
            .to_str()?
            .to_string();
        let pauli_qubits = tuple_usize(py, &self.pauli_qubits)?
            .bind(py)
            .repr()?
            .to_str()?
            .to_string();
        Ok(format!(
            "LogicalObservable(id={}, measurement_keys={}, pauli_qubits={}, pauli={:?})",
            self.id, measurement_keys, pauli_qubits, self.pauli
        ))
    }
}

/// One probabilistic detector-error-model instruction.
#[pyclass(name = "DetectorErrorEdge", module = "faultscope._native", frozen)]
pub(crate) struct PyDetectorErrorEdge {
    probability: f64,
    detectors: Vec<i64>,
    observables: Vec<i64>,
    location_id: String,
    event: Py<PyAny>,
    tags: Py<PyAny>,
}

impl PyDetectorErrorEdge {
    pub(crate) fn from_core_parts(
        probability: f64,
        detectors: Vec<i64>,
        observables: Vec<i64>,
        location_id: String,
        event: Py<PyAny>,
        tags: Py<PyAny>,
    ) -> Self {
        Self {
            probability,
            detectors,
            observables,
            location_id,
            event,
            tags,
        }
    }
}

#[pymethods]
impl PyDetectorErrorEdge {
    #[new]
    #[pyo3(signature = (probability, detectors, observables, location_id, event, tags=None))]
    pub(crate) fn new(
        py: Python<'_>,
        probability: f64,
        detectors: Vec<i64>,
        observables: Vec<i64>,
        location_id: String,
        event: Py<PyAny>,
        tags: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(PyValueError::new_err(format!(
                "DEM edge probability must be in [0, 1], got {probability}"
            )));
        }
        Ok(Self {
            probability,
            detectors,
            observables,
            location_id,
            event,
            tags: mapping_to_dict(py, tags)?,
        })
    }

    #[getter]
    pub(crate) fn probability(&self) -> f64 {
        self.probability
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.detectors.iter().copied())?.into())
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.observables.iter().copied())?.into())
    }

    #[getter]
    pub(crate) fn location_id(&self) -> &str {
        &self.location_id
    }

    #[getter]
    pub(crate) fn event(&self, py: Python<'_>) -> PyObject {
        self.event.clone_ref(py)
    }

    #[getter]
    pub(crate) fn tags(&self, py: Python<'_>) -> PyObject {
        self.tags.clone_ref(py)
    }

    pub(crate) fn to_dem_line(&self, py: Python<'_>) -> PyResult<String> {
        let mut targets: Vec<String> = self
            .detectors
            .iter()
            .map(|detector_id| format!("D{detector_id}"))
            .collect();
        targets.extend(
            self.observables
                .iter()
                .map(|observable_id| format!("L{observable_id}")),
        );
        let probability = py
            .import("builtins")?
            .getattr("format")?
            .call1((self.probability, ".17g"))?
            .extract::<String>()?;
        if targets.is_empty() {
            Ok(format!("error({probability})"))
        } else {
            Ok(format!("error({probability}) {}", targets.join(" ")))
        }
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let detectors = PyTuple::new(py, self.detectors.iter().copied())?
            .repr()?
            .to_str()?
            .to_string();
        let observables = PyTuple::new(py, self.observables.iter().copied())?
            .repr()?
            .to_str()?
            .to_string();
        let event = self.event.bind(py).repr()?.to_str()?.to_string();
        let tags = self.tags.bind(py).repr()?.to_str()?.to_string();
        Ok(format!(
            "DetectorErrorEdge(probability={}, detectors={}, observables={}, location_id={:?}, event={}, tags={})",
            self.probability, detectors, observables, self.location_id, event, tags
        ))
    }
}

/// Typed detector error model with detector and observable declarations.
#[pyclass(name = "DetectorErrorModel", module = "faultscope._native", frozen)]
pub(crate) struct PyDetectorErrorModel {
    core_lazy_dem: Option<faultscope_core::LazyDetectorErrorModel>,
    core_graphlike_problem:
        std::sync::OnceLock<std::sync::Arc<faultscope_core::GraphlikeDecodingProblem>>,
    detectors: Vec<Py<PyAny>>,
    observables: Vec<Py<PyAny>>,
    edges: Vec<Py<PyAny>>,
}

type CoreGraphlikeDecomposition =
    HashMap<usize, Vec<faultscope_core::GraphlikeDecompositionComponent>>;

/// Sparse graphlike decomposition hints bound to one canonical DEM instance.
///
/// The canonical model remains the only sampling model.  The compiled problem
/// cached here is a decoder-only view whose component edges retain their
/// canonical parent edge index.
#[pyclass(
    name = "GraphlikeDecompositionHints",
    module = "faultscope._native",
    frozen
)]
pub(crate) struct PyGraphlikeDecompositionHints {
    dem: Py<PyDetectorErrorModel>,
    decomposition: CoreGraphlikeDecomposition,
    graphlike_problem: std::sync::Arc<faultscope_core::GraphlikeDecodingProblem>,
}

#[pymethods]
impl PyGraphlikeDecompositionHints {
    #[new]
    pub(crate) fn new(
        py: Python<'_>,
        dem: Py<PyDetectorErrorModel>,
        components_by_edge: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let decomposition = parse_graphlike_decomposition(py, components_by_edge)?;
        let graphlike_problem = {
            let dem_ref = dem.bind(py).borrow();
            let core_dem = dem_ref.to_core_dem(py)?;
            std::sync::Arc::new(
                core_dem
                    .compile_graphlike_problem_with_decomposition(&decomposition)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        };
        Ok(Self {
            dem,
            decomposition,
            graphlike_problem,
        })
    }

    #[getter]
    pub(crate) fn components_by_edge(&self, py: Python<'_>) -> PyResult<PyObject> {
        graphlike_decomposition_to_py(py, &self.decomposition)
    }

    pub(crate) fn __repr__(&self) -> String {
        let component_count = self.decomposition.values().map(Vec::len).sum::<usize>();
        format!(
            "GraphlikeDecompositionHints(hinted_edges={}, components={})",
            self.decomposition.len(),
            component_count,
        )
    }
}

/// One canonical detector error model plus optional decoder-only structure.
#[pyclass(
    name = "GeneratedDetectorErrorModel",
    module = "faultscope._native",
    frozen
)]
pub(crate) struct PyGeneratedDetectorErrorModel {
    dem: Py<PyDetectorErrorModel>,
    graphlike_hints: Option<Py<PyGraphlikeDecompositionHints>>,
}

#[pymethods]
impl PyGeneratedDetectorErrorModel {
    #[new]
    #[pyo3(signature = (dem, *, graphlike_hints=None))]
    pub(crate) fn new(
        py: Python<'_>,
        dem: Py<PyDetectorErrorModel>,
        graphlike_hints: Option<Py<PyGraphlikeDecompositionHints>>,
    ) -> PyResult<Self> {
        if let Some(hints) = &graphlike_hints {
            let hinted_dem = hints.bind(py).borrow().dem.as_ptr();
            if hinted_dem != dem.as_ptr() {
                return Err(PyValueError::new_err(
                    "graphlike_hints are bound to a different DetectorErrorModel instance",
                ));
            }
        }
        Ok(Self::from_parts(dem, graphlike_hints))
    }

    #[getter]
    pub(crate) fn dem(&self, py: Python<'_>) -> Py<PyDetectorErrorModel> {
        self.dem.clone_ref(py)
    }

    #[getter]
    pub(crate) fn graphlike_hints(
        &self,
        py: Python<'_>,
    ) -> Option<Py<PyGraphlikeDecompositionHints>> {
        self.graphlike_hints
            .as_ref()
            .map(|hints| hints.clone_ref(py))
    }

    pub(crate) fn compile_graphlike_problem(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyGraphlikeDecodingProblem> {
        if let Some(hints) = &self.graphlike_hints {
            let problem = std::sync::Arc::clone(&hints.bind(py).borrow().graphlike_problem);
            return Ok(PyGraphlikeDecodingProblem::new(problem));
        }
        self.dem
            .bind(py)
            .borrow()
            .compile_graphlike_problem(py, None)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "GeneratedDetectorErrorModel(has_graphlike_hints={})",
            self.graphlike_hints.is_some(),
        )
    }
}

impl PyGeneratedDetectorErrorModel {
    pub(crate) fn without_hints(dem: Py<PyDetectorErrorModel>) -> Self {
        Self::from_parts(dem, None)
    }

    fn from_parts(
        dem: Py<PyDetectorErrorModel>,
        graphlike_hints: Option<Py<PyGraphlikeDecompositionHints>>,
    ) -> Self {
        Self {
            dem,
            graphlike_hints,
        }
    }
}

struct PyDemEdgeView {
    location_id: String,
    event: Py<PyAny>,
    probability: f64,
    detectors: Vec<i64>,
    observables: Vec<i64>,
}

#[pymethods]
impl PyDetectorErrorModel {
    #[new]
    #[pyo3(signature = (detectors, observables, edges))]
    pub(crate) fn new(
        detectors: Vec<Py<PyAny>>,
        observables: Vec<Py<PyAny>>,
        edges: Vec<Py<PyAny>>,
    ) -> Self {
        Self {
            core_lazy_dem: None,
            core_graphlike_problem: std::sync::OnceLock::new(),
            detectors,
            observables,
            edges,
        }
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(dem) = &self.core_lazy_dem {
            let detectors = detectors_to_py_objects(py, &dem.detectors)?;
            return Ok(
                PyTuple::new(py, detectors.iter().map(|detector| detector.clone_ref(py)))?.into(),
            );
        }
        Ok(PyTuple::new(
            py,
            self.detectors.iter().map(|detector| detector.clone_ref(py)),
        )?
        .into())
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(dem) = &self.core_lazy_dem {
            let observables = observables_to_py_objects(py, &dem.observables)?;
            return Ok(PyTuple::new(
                py,
                observables
                    .iter()
                    .map(|observable| observable.clone_ref(py)),
            )?
            .into());
        }
        Ok(PyTuple::new(
            py,
            self.observables
                .iter()
                .map(|observable| observable.clone_ref(py)),
        )?
        .into())
    }

    #[getter]
    pub(crate) fn edges(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(dem) = &self.core_lazy_dem {
            let dem = dem.materialize();
            let edges = edges_to_py_objects(py, &dem.edges)?;
            return Ok(PyTuple::new(py, edges.iter().map(|edge| edge.clone_ref(py)))?.into());
        }
        Ok(PyTuple::new(py, self.edges.iter().map(|edge| edge.clone_ref(py)))?.into())
    }

    #[pyo3(signature = (*, include_detector_coords=true))]
    pub(crate) fn to_dem_text(
        &self,
        py: Python<'_>,
        include_detector_coords: bool,
    ) -> PyResult<String> {
        let mut lines = Vec::<String>::new();
        if include_detector_coords {
            if let Some(dem) = &self.core_lazy_dem {
                for detector in &dem.detectors {
                    lines.push(detector_dem_line(py, detector.id, &detector.coords)?);
                }
            } else {
                for detector in &self.detectors {
                    let detector = detector.bind(py);
                    let detector_id = detector.getattr("id")?.extract::<i64>()?;
                    let coords = detector.getattr("coords")?.extract::<Vec<f64>>()?;
                    lines.push(detector_dem_line(py, detector_id, &coords)?);
                }
            }
        }
        if let Some(dem) = &self.core_lazy_dem {
            let dem = dem.materialize();
            for edge in &dem.edges {
                lines.push(core_edge_dem_line(py, edge)?);
            }
        } else {
            for edge in &self.edges {
                lines.push(
                    edge.bind(py)
                        .call_method0("to_dem_line")?
                        .extract::<String>()?,
                );
            }
        }
        Ok(lines.join("\n"))
    }

    pub(crate) fn edges_by_location(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        if let Some(dem) = &self.core_lazy_dem {
            let dem = dem.materialize();
            let edges = edges_to_py_objects(py, &dem.edges)?;
            for (core_edge, edge) in dem.edges.iter().zip(edges.iter()) {
                let list = match out.get_item(core_edge.location_id.as_str())? {
                    Some(list) => list.downcast::<PyList>()?.clone(),
                    None => {
                        let list = PyList::empty(py);
                        out.set_item(core_edge.location_id.as_str(), &list)?;
                        list
                    }
                };
                list.append(edge.clone_ref(py))?;
            }
            return Ok(out.into());
        }
        for edge in &self.edges {
            let edge = edge.bind(py);
            let location_id = edge.getattr("location_id")?.extract::<String>()?;
            let list = match out.get_item(location_id.as_str())? {
                Some(list) => list.downcast::<PyList>()?.clone(),
                None => {
                    let list = PyList::empty(py);
                    out.set_item(location_id.as_str(), &list)?;
                    list
                }
            };
            list.append(edge)?;
        }
        Ok(out.into())
    }

    pub(crate) fn project_hotspots_to_edges(
        &self,
        py: Python<'_>,
        hotspots: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let edge_views = self.edge_views(py)?;
        let groups = dem_location_groups(&edge_views);
        let out = PyDict::new(py);
        for (location_id, edge_indices) in groups {
            let hotspot = mapping_get_f64(hotspots, location_id.as_str(), 0.0)?;
            if edge_indices.is_empty() {
                continue;
            }
            let total_probability = edge_indices
                .iter()
                .map(|edge_index| edge_views[*edge_index].probability)
                .sum::<f64>();
            for edge_index in &edge_indices {
                let edge = &edge_views[*edge_index];
                let weight = if total_probability > 0.0 {
                    edge.probability / total_probability
                } else {
                    1.0 / edge_indices.len() as f64
                };
                out.set_item(
                    tuple_location_event(py, &edge.location_id, edge.event.clone_ref(py))?,
                    hotspot * weight,
                )?;
            }
        }
        Ok(out.into())
    }

    pub(crate) fn project_result_to_detector_graph(
        &self,
        py: Python<'_>,
        result: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let sensitivities = result.getattr("sensitivities")?;
        self.project_sensitivities_to_detector_graph(py, &sensitivities)
    }

    pub(crate) fn project_sensitivities_to_detector_graph(
        &self,
        py: Python<'_>,
        sensitivities: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let edge_views = self.edge_views(py)?;
        let groups = dem_location_groups(&edge_views);
        let mut location_totals = HashMap::<String, f64>::new();
        for (location_id, edge_indices) in &groups {
            let total_probability = edge_indices
                .iter()
                .map(|edge_index| edge_views[*edge_index].probability)
                .sum::<f64>();
            location_totals.insert(location_id.clone(), total_probability);
        }

        let edge_hotspots = PyList::empty(py);
        let mut by_detector_edge = HashMap::<DetectorGraphKey, f64>::new();
        let mut signed_by_detector_edge = HashMap::<DetectorGraphKey, f64>::new();
        let mut by_detector = HashMap::<i64, f64>::new();
        let mut signed_by_detector = HashMap::<i64, f64>::new();
        let mut by_observable = HashMap::<i64, f64>::new();
        let mut signed_by_observable = HashMap::<i64, f64>::new();
        let mut by_location = HashMap::<String, f64>::new();
        let mut signed_by_location = HashMap::<String, f64>::new();

        for (edge_index, edge) in edge_views.iter().enumerate() {
            let location_sensitivity =
                mapping_get_f64(sensitivities, edge.location_id.as_str(), 0.0)?;
            let total_probability = *location_totals.get(&edge.location_id).unwrap_or(&0.0);
            let sibling_count = groups
                .get(&edge.location_id)
                .map(|edge_indices| edge_indices.len())
                .unwrap_or(0);
            let weight = if total_probability > 0.0 {
                edge.probability / total_probability
            } else if sibling_count > 0 {
                1.0 / sibling_count as f64
            } else {
                0.0
            };
            let edge_sensitivity = location_sensitivity * weight;
            let edge_hotspot = edge_sensitivity.abs();
            edge_hotspots.append(detector_graph_edge_hotspot_object(
                py,
                edge_index,
                edge.location_id.clone(),
                edge.event.clone_ref(py),
                edge.probability,
                PyTuple::new(py, edge.detectors.iter().copied())?.into(),
                PyTuple::new(py, edge.observables.iter().copied())?.into(),
                edge_sensitivity,
                edge_hotspot,
                weight,
            )?)?;

            if edge_hotspot == 0.0 {
                continue;
            }
            let graph_key = DetectorGraphKey {
                detectors: edge.detectors.clone(),
                observables: edge.observables.clone(),
            };
            add_f64(&mut by_detector_edge, graph_key.clone(), edge_hotspot);
            add_f64(&mut signed_by_detector_edge, graph_key, edge_sensitivity);
            add_f64(&mut by_location, edge.location_id.clone(), edge_hotspot);
            add_f64(
                &mut signed_by_location,
                edge.location_id.clone(),
                edge_sensitivity,
            );

            if !edge.detectors.is_empty() {
                let share = edge_hotspot / edge.detectors.len() as f64;
                let signed_share = edge_sensitivity / edge.detectors.len() as f64;
                for detector_id in &edge.detectors {
                    add_f64(&mut by_detector, *detector_id, share);
                    add_f64(&mut signed_by_detector, *detector_id, signed_share);
                }
            }

            if !edge.observables.is_empty() {
                let share = edge_hotspot / edge.observables.len() as f64;
                let signed_share = edge_sensitivity / edge.observables.len() as f64;
                for observable_id in &edge.observables {
                    add_f64(&mut by_observable, *observable_id, share);
                    add_f64(&mut signed_by_observable, *observable_id, signed_share);
                }
            }
        }

        detector_graph_hotspots_object(
            py,
            PyTuple::new(py, edge_hotspots.iter())?.into(),
            graph_key_f64_map_to_py(py, &by_detector_edge)?,
            graph_key_f64_map_to_py(py, &signed_by_detector_edge)?,
            i64_f64_map_to_py(py, &by_detector)?,
            i64_f64_map_to_py(py, &signed_by_detector)?,
            i64_f64_map_to_py(py, &by_observable)?,
            i64_f64_map_to_py(py, &signed_by_observable)?,
            string_f64_map_to_py(py, &by_location)?,
            string_f64_map_to_py(py, &signed_by_location)?,
        )
    }

    pub(crate) fn compile_indexed(&self, py: Python<'_>) -> PyResult<PyIndexedDem> {
        let dem = self.to_core_dem(py)?;
        let indexed = dem
            .compile_indexed()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyIndexedDem { indexed })
    }

    #[pyo3(signature = (*, decomposition=None))]
    pub(crate) fn compile_graphlike_problem(
        &self,
        py: Python<'_>,
        decomposition: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyGraphlikeDecodingProblem> {
        if let Some(decomposition) = decomposition {
            let decomposition = parse_graphlike_decomposition(py, &decomposition)?;
            let dem = self.to_core_dem(py)?;
            let problem = dem
                .compile_graphlike_problem_with_decomposition(&decomposition)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            return Ok(PyGraphlikeDecodingProblem::new(std::sync::Arc::new(
                problem,
            )));
        }

        let problem = if let Some(dem) = &self.core_lazy_dem {
            if let Some(problem) = self.core_graphlike_problem.get() {
                std::sync::Arc::clone(problem)
            } else {
                let compiled = std::sync::Arc::new(
                    dem.compile_graphlike_problem()
                        .map_err(|err| PyValueError::new_err(err.to_string()))?,
                );
                let _ = self
                    .core_graphlike_problem
                    .set(std::sync::Arc::clone(&compiled));
                std::sync::Arc::clone(self.core_graphlike_problem.get().unwrap_or(&compiled))
            }
        } else {
            let dem = self.to_core_dem(py)?;
            std::sync::Arc::new(
                dem.compile_graphlike_problem()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        };
        Ok(PyGraphlikeDecodingProblem::new(problem))
    }

    pub(crate) fn compile_binary_linear_problem(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyBinaryLinearDecodingProblem> {
        let dem = self.to_core_dem(py)?;
        let problem = dem
            .compile_binary_linear_problem()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyBinaryLinearDecodingProblem { problem })
    }

    pub(crate) fn is_graphlike(&self, py: Python<'_>) -> PyResult<bool> {
        if let Some(dem) = &self.core_lazy_dem {
            return Ok(dem.is_graphlike());
        }
        Ok(self.to_core_dem(py)?.is_graphlike())
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "DetectorErrorModel(detectors={}, observables={}, edges={})",
            self.detectors(py)?.bind(py).repr()?,
            self.observables(py)?.bind(py).repr()?,
            self.edges(py)?.bind(py).repr()?,
        ))
    }
}

impl PyDetectorErrorModel {
    pub(crate) fn from_core_lazy_dem(dem: faultscope_core::LazyDetectorErrorModel) -> Self {
        Self {
            core_lazy_dem: Some(dem),
            core_graphlike_problem: std::sync::OnceLock::new(),
            detectors: Vec::new(),
            observables: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub(crate) fn to_core_dem(
        &self,
        py: Python<'_>,
    ) -> PyResult<faultscope_core::DetectorErrorModel> {
        if let Some(dem) = &self.core_lazy_dem {
            return Ok(dem.materialize());
        }
        let detectors = parse_dem_detector_sequence(self.detectors(py)?.bind(py))?;
        let observables = parse_dem_observable_sequence(self.observables(py)?.bind(py))?;
        let edges = parse_dem_edge_sequence(self.edges(py)?.bind(py))?;
        Ok(faultscope_core::DetectorErrorModel {
            detectors,
            observables,
            edges: edges
                .into_iter()
                .map(|edge| faultscope_core::DetectorErrorEdge {
                    probability: edge.probability,
                    detectors: edge.detectors,
                    observables: edge.observables,
                    location_id: edge.location_id,
                    event: edge.event,
                    tags: edge.tags,
                })
                .collect(),
        })
    }

    fn edge_views(&self, py: Python<'_>) -> PyResult<Vec<PyDemEdgeView>> {
        if let Some(dem) = &self.core_lazy_dem {
            let dem = dem.materialize();
            return dem
                .edges
                .iter()
                .map(|edge| {
                    Ok(PyDemEdgeView {
                        location_id: edge.location_id.clone(),
                        event: dem_event_to_py(py, &edge.event)?,
                        probability: edge.probability,
                        detectors: edge.detectors.clone(),
                        observables: edge.observables.clone(),
                    })
                })
                .collect::<PyResult<Vec<_>>>();
        }
        let mut out = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            let edge = edge.bind(py);
            out.push(PyDemEdgeView {
                location_id: edge.getattr("location_id")?.extract::<String>()?,
                event: edge.getattr("event")?.into(),
                probability: edge.getattr("probability")?.extract::<f64>()?,
                detectors: edge.getattr("detectors")?.extract::<Vec<i64>>()?,
                observables: edge.getattr("observables")?.extract::<Vec<i64>>()?,
            });
        }
        Ok(out)
    }
}

fn detector_dem_line(py: Python<'_>, detector_id: i64, coords: &[f64]) -> PyResult<String> {
    if coords.is_empty() {
        Ok(format!("detector D{detector_id}"))
    } else {
        let coords = coords
            .iter()
            .map(|coord| py_format_float(py, *coord, ".17g"))
            .collect::<PyResult<Vec<_>>>()?
            .join(", ");
        Ok(format!("detector({coords}) D{detector_id}"))
    }
}

fn core_edge_dem_line(
    py: Python<'_>,
    edge: &faultscope_core::DetectorErrorEdge,
) -> PyResult<String> {
    let mut targets: Vec<String> = edge
        .detectors
        .iter()
        .map(|detector_id| format!("D{detector_id}"))
        .collect();
    targets.extend(
        edge.observables
            .iter()
            .map(|observable_id| format!("L{observable_id}")),
    );
    let probability = py_format_float(py, edge.probability, ".17g")?;
    if targets.is_empty() {
        Ok(format!("error({probability})"))
    } else {
        Ok(format!("error({probability}) {}", targets.join(" ")))
    }
}

fn mapping_to_dict(py: Python<'_>, value: Option<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    if let Some(value) = value {
        if !value.is_none() {
            dict.call_method1("update", (value,))?;
        }
    }
    Ok(dict.into())
}

fn parse_graphlike_decomposition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<CoreGraphlikeDecomposition> {
    let decomposition_dict = PyDict::new(py);
    decomposition_dict.call_method1("update", (value,))?;
    let decomposition =
        decomposition_dict.extract::<HashMap<usize, Vec<(Vec<i64>, Vec<i64>)>>>()?;
    Ok(decomposition
        .into_iter()
        .map(|(edge_index, components)| {
            (
                edge_index,
                components
                    .into_iter()
                    .map(|(detectors, observables)| {
                        faultscope_core::GraphlikeDecompositionComponent {
                            detectors,
                            observables,
                        }
                    })
                    .collect(),
            )
        })
        .collect())
}

fn graphlike_decomposition_to_py(
    py: Python<'_>,
    decomposition: &CoreGraphlikeDecomposition,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    let mut edge_indices = decomposition.keys().copied().collect::<Vec<_>>();
    edge_indices.sort_unstable();
    for edge_index in edge_indices {
        let mut component_objects = Vec::with_capacity(decomposition[&edge_index].len());
        for component in &decomposition[&edge_index] {
            let detectors = PyTuple::new(py, component.detectors.iter().copied())?.into_any();
            let observables = PyTuple::new(py, component.observables.iter().copied())?.into_any();
            component_objects.push(PyTuple::new(py, [detectors, observables])?);
        }
        result.set_item(edge_index, PyTuple::new(py, component_objects)?)?;
    }
    Ok(result.into())
}

fn kwargs_to_dict<'py>(
    py: Python<'py>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            dict.set_item(key, value)?;
        }
    }
    Ok(dict)
}

fn tuple_usize(py: Python<'_>, values: &[usize]) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, values.iter().copied())?.into())
}

fn tuple_string(py: Python<'_>, values: &[String]) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, values.iter())?.into())
}

fn py_format_float(py: Python<'_>, value: f64, spec: &str) -> PyResult<String> {
    py.import("builtins")?
        .getattr("format")?
        .call1((value, spec))?
        .extract::<String>()
}

fn mapping_get_f64(mapping: &Bound<'_, PyAny>, key: &str, default: f64) -> PyResult<f64> {
    mapping
        .call_method1("get", (key, default))?
        .extract::<f64>()
}

fn tuple_location_event(py: Python<'_>, location_id: &str, event: Py<PyAny>) -> PyResult<PyObject> {
    let location_id = PyString::new(py, location_id).into_any();
    Ok(PyTuple::new(py, [location_id, event.bind(py).clone()])?.into())
}

fn dem_location_groups(edges: &[PyDemEdgeView]) -> HashMap<String, Vec<usize>> {
    let mut groups = HashMap::<String, Vec<usize>>::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        groups
            .entry(edge.location_id.clone())
            .or_default()
            .push(edge_index);
    }
    groups
}

fn add_f64<K>(values: &mut HashMap<K, f64>, key: K, value: f64)
where
    K: std::hash::Hash + Eq,
{
    values
        .entry(key)
        .and_modify(|existing| *existing += value)
        .or_insert(value);
}

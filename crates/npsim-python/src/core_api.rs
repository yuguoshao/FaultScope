use crate::*;

#[pyclass(name = "NoiseLocation", module = "npsim._npsim_native", frozen)]
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
    pub(crate) fn tags(&self, py: Python<'_>) -> PyObject {
        self.tags.clone_ref(py)
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

#[pyclass(name = "Operation", module = "npsim._npsim_native", frozen)]
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
    pub(crate) core_op: Option<npsim_core::Operation>,
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
            core_op: Some(npsim_core::Operation::Pauli { qubits, pauli }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (location, **metadata))]
    pub(crate) fn noise(
        py: Python<'_>,
        location: Py<PyAny>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let core_location = parse_noise_location_object(location.bind(py)).ok();
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
            core_op: core_location.map(npsim_core::Operation::Noise),
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
        let core_op = optional_native_noise_location(py, &noise)
            .ok()
            .map(|core_noise| npsim_core::Operation::Measure {
                qubit,
                key: key.clone(),
                basis: basis.to_uppercase(),
                noise: core_noise,
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
        let core_op = optional_native_noise_location(py, &noise)
            .ok()
            .map(|core_noise| npsim_core::Operation::MeasurePauli {
                qubits: qubits.clone(),
                pauli: pauli.clone(),
                key: key.clone(),
                noise: core_noise,
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
            core_op: Some(npsim_core::Operation::Reset {
                qubit,
                key,
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
            core_op: Some(npsim_core::Operation::Detector {
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
            core_op: Some(npsim_core::Operation::ObservableInclude {
                observable_id,
                measurement_keys,
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
    fn simple_gate(
        py: Python<'_>,
        kind: &str,
        qubits: Vec<usize>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let core_op = match kind {
            "h" => Some(npsim_core::Operation::H(qubits[0])),
            "s" => Some(npsim_core::Operation::S(qubits[0])),
            "s_dag" => Some(npsim_core::Operation::SDag(qubits[0])),
            "cx" => Some(npsim_core::Operation::Cx(qubits[0], qubits[1])),
            "cz" => Some(npsim_core::Operation::Cz(qubits[0], qubits[1])),
            "swap" => Some(npsim_core::Operation::Swap(qubits[0], qubits[1])),
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
            core_op,
        })
    }
}

#[pyclass(name = "Circuit", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyCircuit {
    pub(crate) n_qubits: usize,
    pub(crate) operations: Vec<Py<PyAny>>,
    pub(crate) core_circuit: Option<std::sync::Arc<npsim_core::Circuit>>,
    pub(crate) core_event_plan: Option<std::sync::Arc<npsim_core::DemEventPlan>>,
}

#[pymethods]
impl PyCircuit {
    #[new]
    #[pyo3(signature = (n_qubits, operations))]
    pub(crate) fn new(py: Python<'_>, n_qubits: usize, operations: Vec<Py<PyAny>>) -> Self {
        let core_operations = operations
            .iter()
            .map(|operation| parse_operation_object(operation.bind(py)))
            .collect::<PyResult<Vec<_>>>()
            .ok();
        let core_circuit = core_operations.map(|operations| {
            std::sync::Arc::new(npsim_core::Circuit {
                n_qubits,
                operations,
            })
        });
        let core_event_plan = core_circuit.as_ref().and_then(|circuit| {
            npsim_core::collect_dem_event_plan(&circuit.operations)
                .ok()
                .map(std::sync::Arc::new)
        });
        Self {
            n_qubits,
            operations,
            core_circuit,
            core_event_plan,
        }
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
        for operation in &self.operations {
            let operation = operation.bind(py);
            let kind = operation.getattr("kind")?.extract::<String>()?;
            if kind != "noise" && kind != "measure" && kind != "measure_pauli" {
                continue;
            }
            let location = operation.getattr("noise_location")?;
            if location.is_none() {
                continue;
            }
            let location_id = location.getattr("id")?.extract::<String>()?;
            out.set_item(location_id, location)?;
        }
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

#[pyclass(name = "Detector", module = "npsim._npsim_native", frozen)]
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

#[pyclass(name = "LogicalObservable", module = "npsim._npsim_native", frozen)]
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

#[pyclass(name = "DetectorErrorEdge", module = "npsim._npsim_native", frozen)]
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

#[pyclass(name = "DetectorErrorModel", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyDetectorErrorModel {
    core_dem: Option<npsim_core::DetectorErrorModel>,
    core_lazy_dem: Option<npsim_core::LazyDetectorErrorModel>,
    detectors: Vec<Py<PyAny>>,
    observables: Vec<Py<PyAny>>,
    edges: Vec<Py<PyAny>>,
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
            core_dem: None,
            core_lazy_dem: None,
            detectors,
            observables,
            edges,
        }
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(dem) = &self.core_dem {
            let detectors = detectors_to_py_objects(py, &dem.detectors)?;
            return Ok(
                PyTuple::new(py, detectors.iter().map(|detector| detector.clone_ref(py)))?.into(),
            );
        }
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
        if let Some(dem) = &self.core_dem {
            let observables = observables_to_py_objects(py, &dem.observables)?;
            return Ok(PyTuple::new(
                py,
                observables
                    .iter()
                    .map(|observable| observable.clone_ref(py)),
            )?
            .into());
        }
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
        if let Some(dem) = &self.core_dem {
            let edges = edges_to_py_objects(py, &dem.edges)?;
            return Ok(PyTuple::new(py, edges.iter().map(|edge| edge.clone_ref(py)))?.into());
        }
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
            if let Some(dem) = &self.core_dem {
                for detector in &dem.detectors {
                    lines.push(detector_dem_line(py, detector.id, &detector.coords)?);
                }
            } else if let Some(dem) = &self.core_lazy_dem {
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
        if let Some(dem) = &self.core_dem {
            for edge in &dem.edges {
                lines.push(core_edge_dem_line(py, edge)?);
            }
        } else if let Some(dem) = &self.core_lazy_dem {
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
        if let Some(dem) = &self.core_dem {
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
    pub(crate) fn from_core_dem(dem: npsim_core::DetectorErrorModel) -> Self {
        Self {
            core_dem: Some(dem),
            core_lazy_dem: None,
            detectors: Vec::new(),
            observables: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub(crate) fn from_core_lazy_dem(dem: npsim_core::LazyDetectorErrorModel) -> Self {
        Self {
            core_dem: None,
            core_lazy_dem: Some(dem),
            detectors: Vec::new(),
            observables: Vec::new(),
            edges: Vec::new(),
        }
    }

    fn edge_views(&self, py: Python<'_>) -> PyResult<Vec<PyDemEdgeView>> {
        if let Some(dem) = &self.core_dem {
            return Ok(dem
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
                .collect::<PyResult<Vec<_>>>()?);
        }
        if let Some(dem) = &self.core_lazy_dem {
            let dem = dem.materialize();
            return Ok(dem
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
                .collect::<PyResult<Vec<_>>>()?);
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

fn core_edge_dem_line(py: Python<'_>, edge: &npsim_core::DetectorErrorEdge) -> PyResult<String> {
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

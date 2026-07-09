use crate::*;
use faultscope_core::DetectorErrorModelGenerator as CoreDetectorErrorModelGenerator;

#[pyclass(name = "DetectorErrorModelGenerator", module = "faultscope._native")]
pub(crate) struct PyDetectorErrorModelGenerator {
    py_circuit: Py<PyAny>,
    py_detectors: Py<PyAny>,
    py_observables: Py<PyAny>,
    generator: CoreDetectorErrorModelGenerator,
}

#[pymethods]
impl PyDetectorErrorModelGenerator {
    #[new]
    #[pyo3(signature = (circuit, *, detectors=None, observables=None))]
    pub(crate) fn new(
        py: Python<'_>,
        circuit: &Bound<'_, PyAny>,
        detectors: Option<&Bound<'_, PyAny>>,
        observables: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let core_circuit = parse_core_circuit_object(circuit)?;
        let detector_specs = optional_detector_specs(detectors)?;
        let observable_specs = optional_observable_specs(observables)?;
        let generator =
            CoreDetectorErrorModelGenerator::new(core_circuit, detector_specs, observable_specs)
                .map_err(np_error_to_py)?;
        let py_detectors = match detectors {
            Some(detectors) if !detectors.is_none() => py_tuple_from_sequence(py, detectors)?,
            _ => detectors_to_py_tuple(py, &generator.detectors)?,
        };
        let py_observables = match observables {
            Some(observables) if !observables.is_none() => py_tuple_from_sequence(py, observables)?,
            _ => observables_to_py_tuple(py, &generator.observables)?,
        };
        Ok(Self {
            py_circuit: circuit.clone().unbind(),
            py_detectors,
            py_observables,
            generator,
        })
    }

    #[getter]
    pub(crate) fn circuit(&self, py: Python<'_>) -> PyObject {
        self.py_circuit.clone_ref(py)
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyObject {
        self.py_detectors.clone_ref(py)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyObject {
        self.py_observables.clone_ref(py)
    }

    pub(crate) fn generate(&self, py: Python<'_>) -> PyResult<PyDetectorErrorModel> {
        let dem = self.generator.generate_lazy().map_err(np_error_to_py)?;
        detector_error_model_lazy_to_py(py, dem)
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "DetectorErrorModelGenerator(circuit={}, detectors={}, observables={})",
            self.py_circuit.bind(py).repr()?,
            self.py_detectors.bind(py).repr()?,
            self.py_observables.bind(py).repr()?,
        ))
    }
}

pub(crate) fn detector_error_model_to_py(
    _py: Python<'_>,
    dem: faultscope_core::DetectorErrorModel,
) -> PyResult<PyDetectorErrorModel> {
    Ok(PyDetectorErrorModel::from_core_dem(dem))
}

pub(crate) fn detector_error_model_lazy_to_py(
    _py: Python<'_>,
    dem: faultscope_core::LazyDetectorErrorModel,
) -> PyResult<PyDetectorErrorModel> {
    Ok(PyDetectorErrorModel::from_core_lazy_dem(dem))
}

pub(crate) fn dem_event_to_py(
    py: Python<'_>,
    event: &faultscope_core::DemEvent,
) -> PyResult<Py<PyAny>> {
    match event {
        faultscope_core::DemEvent::Pauli(pauli) => Ok(PyString::new(py, pauli).into_any().unbind()),
        faultscope_core::DemEvent::Bool(value) => {
            Ok(PyBool::new(py, *value).to_owned().into_any().unbind())
        }
    }
}

fn optional_detector_specs(
    detectors: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<DemDetectorSpec>>> {
    match detectors {
        Some(detectors) if !detectors.is_none() => {
            Ok(Some(parse_dem_detector_sequence(detectors)?))
        }
        _ => Ok(None),
    }
}

fn optional_observable_specs(
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<DemObservableSpec>>> {
    match observables {
        Some(observables) if !observables.is_none() => {
            Ok(Some(parse_dem_observable_sequence(observables)?))
        }
        _ => Ok(None),
    }
}

fn py_tuple_from_sequence(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("builtins")?
        .getattr("tuple")?
        .call1((value,))?
        .unbind())
}

fn detectors_to_py_tuple(
    py: Python<'_>,
    detectors: &[faultscope_core::Detector],
) -> PyResult<Py<PyAny>> {
    let items = detectors_to_py_objects(py, detectors)?;
    Ok(
        PyTuple::new(py, items.iter().map(|item| item.clone_ref(py)))?
            .into_any()
            .unbind(),
    )
}

fn observables_to_py_tuple(
    py: Python<'_>,
    observables: &[faultscope_core::LogicalObservable],
) -> PyResult<Py<PyAny>> {
    let items = observables_to_py_objects(py, observables)?;
    Ok(
        PyTuple::new(py, items.iter().map(|item| item.clone_ref(py)))?
            .into_any()
            .unbind(),
    )
}

pub(crate) fn detectors_to_py_objects(
    py: Python<'_>,
    detectors: &[faultscope_core::Detector],
) -> PyResult<Vec<Py<PyAny>>> {
    detectors
        .iter()
        .map(|detector| {
            Ok(Py::new(
                py,
                PyDetector::new(
                    detector.id,
                    detector.measurement_keys.clone(),
                    Some(detector.coords.clone()),
                ),
            )?
            .into_any())
        })
        .collect()
}

pub(crate) fn observables_to_py_objects(
    py: Python<'_>,
    observables: &[faultscope_core::LogicalObservable],
) -> PyResult<Vec<Py<PyAny>>> {
    observables
        .iter()
        .map(|observable| {
            Ok(Py::new(
                py,
                PyLogicalObservable::new(
                    observable.id,
                    Some(observable.measurement_keys.clone()),
                    Some(observable.pauli_qubits.clone()),
                    &observable.pauli,
                )?,
            )?
            .into_any())
        })
        .collect()
}

pub(crate) fn edges_to_py_objects(
    py: Python<'_>,
    edges: &[faultscope_core::DetectorErrorEdge],
) -> PyResult<Vec<Py<PyAny>>> {
    edges
        .iter()
        .map(|edge| {
            let tags = PyDict::new(py);
            for (key, value) in &edge.tags {
                tags.set_item(key, tag_value_to_py(py, value)?)?;
            }
            Ok(Py::new(
                py,
                PyDetectorErrorEdge::from_core_parts(
                    edge.probability,
                    edge.detectors.clone(),
                    edge.observables.clone(),
                    edge.location_id.clone(),
                    dem_event_to_py(py, &edge.event)?,
                    tags.into_any().unbind(),
                ),
            )?
            .into_any())
        })
        .collect()
}

fn np_error_to_py(err: faultscope_core::NpError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

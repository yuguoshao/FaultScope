use crate::*;

#[pyclass(name = "SampleBatch", module = "faultscope._native", frozen)]
pub(crate) struct PySampleBatch {
    shots: usize,
    all_mask: Py<PyAny>,
    x_frame: Py<PyAny>,
    z_frame: Py<PyAny>,
    measurements: Py<PyAny>,
    detectors: Py<PyAny>,
    observables: Py<PyAny>,
    noise_event_masks: Py<PyAny>,
}

#[pymethods]
impl PySampleBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        shots,
        all_mask,
        x_frame,
        z_frame,
        measurements,
        detectors,
        observables,
        noise_event_masks
    ))]
    pub(crate) fn new(
        py: Python<'_>,
        shots: usize,
        all_mask: Py<PyAny>,
        x_frame: &Bound<'_, PyAny>,
        z_frame: &Bound<'_, PyAny>,
        measurements: &Bound<'_, PyAny>,
        detectors: &Bound<'_, PyAny>,
        observables: &Bound<'_, PyAny>,
        noise_event_masks: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            shots,
            all_mask,
            x_frame: sequence_to_tuple(py, x_frame)?,
            z_frame: sequence_to_tuple(py, z_frame)?,
            measurements: mapping_to_dict_object(py, measurements)?,
            detectors: mapping_to_dict_object(py, detectors)?,
            observables: mapping_to_dict_object(py, observables)?,
            noise_event_masks: mapping_to_dict_object(py, noise_event_masks)?,
        })
    }

    #[getter]
    pub(crate) fn shots(&self) -> usize {
        self.shots
    }

    #[getter]
    pub(crate) fn all_mask(&self, py: Python<'_>) -> PyObject {
        self.all_mask.clone_ref(py)
    }

    #[getter]
    pub(crate) fn x_frame(&self, py: Python<'_>) -> PyObject {
        self.x_frame.clone_ref(py)
    }

    #[getter]
    pub(crate) fn z_frame(&self, py: Python<'_>) -> PyObject {
        self.z_frame.clone_ref(py)
    }

    #[getter]
    pub(crate) fn measurements(&self, py: Python<'_>) -> PyObject {
        self.measurements.clone_ref(py)
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyObject {
        self.detectors.clone_ref(py)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyObject {
        self.observables.clone_ref(py)
    }

    #[getter]
    pub(crate) fn noise_event_masks(&self, py: Python<'_>) -> PyObject {
        self.noise_event_masks.clone_ref(py)
    }

    pub(crate) fn bit(&self, mask: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
        py_int_bit(mask, shot)
    }

    pub(crate) fn measurement_bit(&self, py: Python<'_>, key: &str, shot: usize) -> PyResult<u8> {
        let mask = self.measurements.bind(py).get_item(key)?;
        py_int_bit(&mask, shot)
    }

    pub(crate) fn measurement_mask(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        Ok(self.measurements.bind(py).get_item(key)?.into())
    }

    pub(crate) fn detector_bit(
        &self,
        py: Python<'_>,
        detector_id: i64,
        shot: usize,
    ) -> PyResult<u8> {
        let mask = self.detectors.bind(py).get_item(detector_id)?;
        py_int_bit(&mask, shot)
    }

    pub(crate) fn observable_bit(
        &self,
        py: Python<'_>,
        observable_id: i64,
        shot: usize,
    ) -> PyResult<u8> {
        let mask = self.observables.bind(py).get_item(observable_id)?;
        py_int_bit(&mask, shot)
    }

    pub(crate) fn x_bit(&self, py: Python<'_>, qubit: usize, shot: usize) -> PyResult<u8> {
        let mask = self.x_frame.bind(py).get_item(qubit)?;
        py_int_bit(&mask, shot)
    }

    pub(crate) fn x_mask(&self, py: Python<'_>, qubit: usize) -> PyResult<PyObject> {
        Ok(self.x_frame.bind(py).get_item(qubit)?.into())
    }

    pub(crate) fn z_bit(&self, py: Python<'_>, qubit: usize, shot: usize) -> PyResult<u8> {
        let mask = self.z_frame.bind(py).get_item(qubit)?;
        py_int_bit(&mask, shot)
    }

    pub(crate) fn z_mask(&self, py: Python<'_>, qubit: usize) -> PyResult<PyObject> {
        Ok(self.z_frame.bind(py).get_item(qubit)?.into())
    }
}

#[pyclass(name = "DemSampleBatch", module = "faultscope._native", frozen)]
pub(crate) struct PyDemSampleBatch {
    shots: usize,
    all_mask: Py<PyAny>,
    detectors: Py<PyAny>,
    observables: Py<PyAny>,
    edge_event_masks: Py<PyAny>,
}

#[pymethods]
impl PyDemSampleBatch {
    #[new]
    #[pyo3(signature = (shots, all_mask, detectors, observables, edge_event_masks))]
    pub(crate) fn new(
        py: Python<'_>,
        shots: usize,
        all_mask: Py<PyAny>,
        detectors: &Bound<'_, PyAny>,
        observables: &Bound<'_, PyAny>,
        edge_event_masks: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            shots,
            all_mask,
            detectors: mapping_to_dict_object(py, detectors)?,
            observables: mapping_to_dict_object(py, observables)?,
            edge_event_masks: mapping_to_dict_object(py, edge_event_masks)?,
        })
    }

    #[getter]
    pub(crate) fn shots(&self) -> usize {
        self.shots
    }

    #[getter]
    pub(crate) fn all_mask(&self, py: Python<'_>) -> PyObject {
        self.all_mask.clone_ref(py)
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyObject {
        self.detectors.clone_ref(py)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyObject {
        self.observables.clone_ref(py)
    }

    #[getter]
    pub(crate) fn edge_event_masks(&self, py: Python<'_>) -> PyObject {
        self.edge_event_masks.clone_ref(py)
    }

    pub(crate) fn bit(&self, mask: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
        py_int_bit(mask, shot)
    }

    pub(crate) fn detector_bit(
        &self,
        py: Python<'_>,
        detector_id: i64,
        shot: usize,
    ) -> PyResult<u8> {
        let mask = self.detectors.bind(py).get_item(detector_id)?;
        py_int_bit(&mask, shot)
    }

    pub(crate) fn observable_bit(
        &self,
        py: Python<'_>,
        observable_id: i64,
        shot: usize,
    ) -> PyResult<u8> {
        let mask = self.observables.bind(py).get_item(observable_id)?;
        py_int_bit(&mask, shot)
    }

    pub(crate) fn edge_event_bit(
        &self,
        py: Python<'_>,
        edge_index: usize,
        shot: usize,
    ) -> PyResult<u8> {
        let mask = self.edge_event_masks.bind(py).get_item(edge_index)?;
        py_int_bit(&mask, shot)
    }
}

pub(crate) fn batch_trajectory_from_state(
    py: Python<'_>,
    state: &RuntimeState,
) -> PyResult<PySampleBatch> {
    Ok(PySampleBatch {
        shots: state.shots,
        all_mask: mask_to_py(py, &state.all_mask)?,
        x_frame: mask_vec_to_tuple_py(py, &state.x_frame)?,
        z_frame: mask_vec_to_tuple_py(py, &state.z_frame)?,
        measurements: map_to_py(py, &state.measurements)?,
        detectors: int_map_to_py(py, &state.detectors)?,
        observables: int_map_to_py(py, &state.observables)?,
        noise_event_masks: map_to_py(py, &state.event_masks)?,
    })
}

pub(crate) fn dem_batch_trajectory_from_batch(
    py: Python<'_>,
    batch: &DemBatch,
) -> PyResult<PyDemSampleBatch> {
    let edge_masks = PyDict::new(py);
    for (edge_index, mask) in batch.edge_event_masks.iter().enumerate() {
        edge_masks.set_item(edge_index, mask_to_py(py, mask)?)?;
    }
    Ok(PyDemSampleBatch {
        shots: batch.shots,
        all_mask: mask_to_py(py, &batch.all_mask)?,
        detectors: int_map_to_py(py, &batch.detectors)?,
        observables: int_map_to_py(py, &batch.observables)?,
        edge_event_masks: edge_masks.into(),
    })
}

fn mask_vec_to_tuple_py(py: Python<'_>, values: &[Mask]) -> PyResult<PyObject> {
    let items = values
        .iter()
        .map(|value| mask_to_py(py, value))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, items)?.into())
}

fn sequence_to_tuple(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(py
        .import("builtins")?
        .getattr("tuple")?
        .call1((value,))?
        .into())
}

fn mapping_to_dict_object(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.call_method1("update", (value,))?;
    Ok(dict.into())
}

#[pyclass(name = "FaultHotspot", module = "faultscope._native", frozen)]
pub(crate) struct PyFaultHotspot {
    location_id: String,
    sensitivity: f64,
    hotspot: f64,
    qubits: Py<PyAny>,
    tags: Py<PyAny>,
}

#[pymethods]
impl PyFaultHotspot {
    #[new]
    #[pyo3(signature = (location_id, sensitivity, hotspot, qubits, tags))]
    pub(crate) fn new(
        py: Python<'_>,
        location_id: String,
        sensitivity: f64,
        hotspot: f64,
        qubits: &Bound<'_, PyAny>,
        tags: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            location_id,
            sensitivity,
            hotspot,
            qubits: sequence_to_tuple(py, qubits)?,
            tags: mapping_to_dict_object(py, tags)?,
        })
    }

    #[getter]
    pub(crate) fn location_id(&self) -> &str {
        &self.location_id
    }

    #[getter]
    pub(crate) fn sensitivity(&self) -> f64 {
        self.sensitivity
    }

    #[getter]
    pub(crate) fn hotspot(&self) -> f64 {
        self.hotspot
    }

    #[getter]
    pub(crate) fn qubits(&self, py: Python<'_>) -> PyObject {
        self.qubits.clone_ref(py)
    }

    #[getter]
    pub(crate) fn tags(&self, py: Python<'_>) -> PyObject {
        self.tags.clone_ref(py)
    }
}

#[pyclass(name = "FailureEstimate", module = "faultscope._native", frozen)]
pub(crate) struct PyFailureEstimate {
    shots: usize,
    mean_loss: f64,
    baseline: f64,
    sensitivities: Py<PyAny>,
    hotspots: Py<PyAny>,
    by_qubit: Py<PyAny>,
    by_round: Py<PyAny>,
    by_gate: Py<PyAny>,
    by_operation: Py<PyAny>,
    locations: Py<PyAny>,
    losses: Py<PyAny>,
    top_hotspots_cache: Py<PyAny>,
}

#[pymethods]
impl PyFailureEstimate {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        shots,
        mean_loss,
        baseline,
        sensitivities,
        hotspots,
        by_qubit,
        by_round,
        by_gate,
        by_operation,
        locations,
        losses,
        top_hotspots_cache=None
    ))]
    pub(crate) fn new(
        py: Python<'_>,
        shots: usize,
        mean_loss: f64,
        baseline: f64,
        sensitivities: &Bound<'_, PyAny>,
        hotspots: &Bound<'_, PyAny>,
        by_qubit: &Bound<'_, PyAny>,
        by_round: &Bound<'_, PyAny>,
        by_gate: &Bound<'_, PyAny>,
        by_operation: &Bound<'_, PyAny>,
        locations: &Bound<'_, PyAny>,
        losses: &Bound<'_, PyAny>,
        top_hotspots_cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            shots,
            mean_loss,
            baseline,
            sensitivities: mapping_to_dict_object(py, sensitivities)?,
            hotspots: mapping_to_dict_object(py, hotspots)?,
            by_qubit: mapping_to_dict_object(py, by_qubit)?,
            by_round: mapping_to_dict_object(py, by_round)?,
            by_gate: mapping_to_dict_object(py, by_gate)?,
            by_operation: mapping_to_dict_object(py, by_operation)?,
            locations: mapping_to_dict_object(py, locations)?,
            losses: list_to_list_object(py, losses)?,
            top_hotspots_cache: optional_sequence_to_tuple(py, top_hotspots_cache)?,
        })
    }

    #[getter]
    pub(crate) fn shots(&self) -> usize {
        self.shots
    }

    #[getter]
    pub(crate) fn mean_loss(&self) -> f64 {
        self.mean_loss
    }

    #[getter]
    pub(crate) fn logical_failure_rate(&self) -> f64 {
        self.mean_loss
    }

    #[getter]
    pub(crate) fn baseline(&self) -> f64 {
        self.baseline
    }

    #[getter]
    pub(crate) fn sensitivities(&self, py: Python<'_>) -> PyObject {
        self.sensitivities.clone_ref(py)
    }

    #[getter]
    pub(crate) fn hotspots(&self, py: Python<'_>) -> PyObject {
        self.hotspots.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_qubit(&self, py: Python<'_>) -> PyObject {
        self.by_qubit.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_round(&self, py: Python<'_>) -> PyObject {
        self.by_round.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_gate(&self, py: Python<'_>) -> PyObject {
        self.by_gate.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_operation(&self, py: Python<'_>) -> PyObject {
        self.by_operation.clone_ref(py)
    }

    #[getter]
    pub(crate) fn locations(&self, py: Python<'_>) -> PyObject {
        self.locations.clone_ref(py)
    }

    #[getter]
    pub(crate) fn losses(&self, py: Python<'_>) -> PyObject {
        self.losses.clone_ref(py)
    }

    #[getter]
    pub(crate) fn top_hotspots_cache(&self, py: Python<'_>) -> PyObject {
        self.top_hotspots_cache.clone_ref(py)
    }

    #[pyo3(signature = (top_k=10))]
    pub(crate) fn top_hotspots(&self, py: Python<'_>, top_k: usize) -> PyResult<PyObject> {
        let cache = self.top_hotspots_cache.bind(py);
        let cache_len = cache.len()?;
        if cache_len > 0 && top_k <= cache_len {
            let out = PyList::empty(py);
            for item in cache.try_iter()?.take(top_k) {
                out.append(item?)?;
            }
            return Ok(out.into());
        }

        let mut rows: Vec<(f64, Py<PyFaultHotspot>)> = Vec::new();
        let hotspots = self.hotspots.bind(py).downcast::<PyDict>()?;
        for (location_id, hotspot) in hotspots.iter() {
            let location_id = location_id.extract::<String>()?;
            let hotspot = hotspot.extract::<f64>()?;
            let sensitivity = self
                .sensitivities
                .bind(py)
                .get_item(location_id.as_str())?
                .extract::<f64>()?;
            let location = self.locations.bind(py).get_item(location_id.as_str())?;
            let row = PyFaultHotspot {
                location_id,
                sensitivity,
                hotspot,
                qubits: location.getattr("qubits")?.into(),
                tags: location.getattr("tags")?.into(),
            };
            rows.push((hotspot, Py::new(py, row)?));
        }
        rows.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let out = PyList::empty(py);
        for (_, row) in rows.into_iter().take(top_k) {
            out.append(row)?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (top_k=10))]
    pub(crate) fn hotspot_table(&self, py: Python<'_>, top_k: usize) -> PyResult<String> {
        let rows = self.top_hotspots(py, top_k)?;
        let mut lines = vec!["location_id\tsensitivity\thotspot\tqubits\ttags".to_string()];
        for row in rows.bind(py).try_iter()? {
            let row = row?;
            let location_id = row.getattr("location_id")?.extract::<String>()?;
            let sensitivity = row.getattr("sensitivity")?.extract::<f64>()?;
            let hotspot = row.getattr("hotspot")?.extract::<f64>()?;
            let sensitivity = py
                .import("builtins")?
                .getattr("format")?
                .call1((sensitivity, ".6g"))?
                .extract::<String>()?;
            let hotspot = py
                .import("builtins")?
                .getattr("format")?
                .call1((hotspot, ".6g"))?
                .extract::<String>()?;
            let qubits = row.getattr("qubits")?.repr()?.to_str()?.to_string();
            let tags = py
                .import("builtins")?
                .getattr("dict")?
                .call1((row.getattr("tags")?,))?
                .repr()?
                .to_str()?
                .to_string();
            lines.push(format!(
                "{location_id}\t{sensitivity}\t{hotspot}\t{qubits}\t{tags}"
            ));
        }
        Ok(lines.join("\n"))
    }
}

#[pyclass(name = "DemLocationMetadata", module = "faultscope._native", frozen)]
pub(crate) struct PyDemLocationMetadata {
    id: String,
    tags: Py<PyAny>,
    qubits: Py<PyAny>,
}

#[pymethods]
impl PyDemLocationMetadata {
    #[new]
    #[pyo3(signature = (id, tags, qubits=None))]
    pub(crate) fn new(
        py: Python<'_>,
        id: String,
        tags: &Bound<'_, PyAny>,
        qubits: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            id,
            tags: mapping_to_dict_object(py, tags)?,
            qubits: optional_sequence_to_tuple(py, qubits)?,
        })
    }

    #[getter]
    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    #[getter]
    pub(crate) fn tags(&self, py: Python<'_>) -> PyObject {
        self.tags.clone_ref(py)
    }

    #[getter]
    pub(crate) fn qubits(&self, py: Python<'_>) -> PyObject {
        self.qubits.clone_ref(py)
    }
}

#[pyclass(name = "DemLocationHotspot", module = "faultscope._native", frozen)]
pub(crate) struct PyDemLocationHotspot {
    location_id: String,
    sensitivity: f64,
    hotspot: f64,
    qubits: Py<PyAny>,
    tags: Py<PyAny>,
}

#[pymethods]
impl PyDemLocationHotspot {
    #[new]
    #[pyo3(signature = (location_id, sensitivity, hotspot, qubits, tags))]
    pub(crate) fn new(
        py: Python<'_>,
        location_id: String,
        sensitivity: f64,
        hotspot: f64,
        qubits: &Bound<'_, PyAny>,
        tags: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            location_id,
            sensitivity,
            hotspot,
            qubits: sequence_to_tuple(py, qubits)?,
            tags: mapping_to_dict_object(py, tags)?,
        })
    }

    #[getter]
    pub(crate) fn location_id(&self) -> &str {
        &self.location_id
    }

    #[getter]
    pub(crate) fn sensitivity(&self) -> f64 {
        self.sensitivity
    }

    #[getter]
    pub(crate) fn hotspot(&self) -> f64 {
        self.hotspot
    }

    #[getter]
    pub(crate) fn qubits(&self, py: Python<'_>) -> PyObject {
        self.qubits.clone_ref(py)
    }

    #[getter]
    pub(crate) fn tags(&self, py: Python<'_>) -> PyObject {
        self.tags.clone_ref(py)
    }
}

#[pyclass(name = "DemEdgeHotspot", module = "faultscope._native", frozen)]
pub(crate) struct PyDemEdgeHotspot {
    edge_index: usize,
    location_id: String,
    event: Py<PyAny>,
    probability: f64,
    detectors: Py<PyAny>,
    observables: Py<PyAny>,
    sensitivity: f64,
    hotspot: f64,
}

#[pymethods]
impl PyDemEdgeHotspot {
    #[new]
    #[pyo3(signature = (
        edge_index,
        location_id,
        event,
        probability,
        detectors,
        observables,
        sensitivity,
        hotspot
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        py: Python<'_>,
        edge_index: usize,
        location_id: String,
        event: Py<PyAny>,
        probability: f64,
        detectors: &Bound<'_, PyAny>,
        observables: &Bound<'_, PyAny>,
        sensitivity: f64,
        hotspot: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            edge_index,
            location_id,
            event,
            probability,
            detectors: sequence_to_tuple(py, detectors)?,
            observables: sequence_to_tuple(py, observables)?,
            sensitivity,
            hotspot,
        })
    }

    #[getter]
    pub(crate) fn edge_index(&self) -> usize {
        self.edge_index
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
    pub(crate) fn probability(&self) -> f64 {
        self.probability
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyObject {
        self.detectors.clone_ref(py)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyObject {
        self.observables.clone_ref(py)
    }

    #[getter]
    pub(crate) fn sensitivity(&self) -> f64 {
        self.sensitivity
    }

    #[getter]
    pub(crate) fn hotspot(&self) -> f64 {
        self.hotspot
    }
}

#[pyclass(
    name = "DetectorGraphEdgeHotspot",
    module = "faultscope._native",
    frozen
)]
pub(crate) struct PyDetectorGraphEdgeHotspot {
    edge_index: usize,
    location_id: String,
    event: Py<PyAny>,
    probability: f64,
    detectors: Py<PyAny>,
    observables: Py<PyAny>,
    sensitivity: f64,
    hotspot: f64,
    weight: f64,
}

#[pymethods]
impl PyDetectorGraphEdgeHotspot {
    #[new]
    #[pyo3(signature = (
        edge_index,
        location_id,
        event,
        probability,
        detectors,
        observables,
        sensitivity,
        hotspot,
        weight
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        py: Python<'_>,
        edge_index: usize,
        location_id: String,
        event: Py<PyAny>,
        probability: f64,
        detectors: &Bound<'_, PyAny>,
        observables: &Bound<'_, PyAny>,
        sensitivity: f64,
        hotspot: f64,
        weight: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            edge_index,
            location_id,
            event,
            probability,
            detectors: sequence_to_tuple(py, detectors)?,
            observables: sequence_to_tuple(py, observables)?,
            sensitivity,
            hotspot,
            weight,
        })
    }

    #[getter]
    pub(crate) fn edge_index(&self) -> usize {
        self.edge_index
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
    pub(crate) fn probability(&self) -> f64 {
        self.probability
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyObject {
        self.detectors.clone_ref(py)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyObject {
        self.observables.clone_ref(py)
    }

    #[getter]
    pub(crate) fn sensitivity(&self) -> f64 {
        self.sensitivity
    }

    #[getter]
    pub(crate) fn hotspot(&self) -> f64 {
        self.hotspot
    }

    #[getter]
    pub(crate) fn weight(&self) -> f64 {
        self.weight
    }
}

#[pyclass(name = "DetectorGraphHotspots", module = "faultscope._native", frozen)]
pub(crate) struct PyDetectorGraphHotspots {
    edge_hotspots: Py<PyAny>,
    by_detector_edge: Py<PyAny>,
    signed_by_detector_edge: Py<PyAny>,
    by_detector: Py<PyAny>,
    signed_by_detector: Py<PyAny>,
    by_observable: Py<PyAny>,
    signed_by_observable: Py<PyAny>,
    by_location: Py<PyAny>,
    signed_by_location: Py<PyAny>,
}

#[pymethods]
impl PyDetectorGraphHotspots {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        edge_hotspots,
        by_detector_edge,
        signed_by_detector_edge,
        by_detector,
        signed_by_detector,
        by_observable,
        signed_by_observable,
        by_location,
        signed_by_location
    ))]
    pub(crate) fn new(
        py: Python<'_>,
        edge_hotspots: &Bound<'_, PyAny>,
        by_detector_edge: &Bound<'_, PyAny>,
        signed_by_detector_edge: &Bound<'_, PyAny>,
        by_detector: &Bound<'_, PyAny>,
        signed_by_detector: &Bound<'_, PyAny>,
        by_observable: &Bound<'_, PyAny>,
        signed_by_observable: &Bound<'_, PyAny>,
        by_location: &Bound<'_, PyAny>,
        signed_by_location: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            edge_hotspots: sequence_to_tuple(py, edge_hotspots)?,
            by_detector_edge: mapping_to_dict_object(py, by_detector_edge)?,
            signed_by_detector_edge: mapping_to_dict_object(py, signed_by_detector_edge)?,
            by_detector: mapping_to_dict_object(py, by_detector)?,
            signed_by_detector: mapping_to_dict_object(py, signed_by_detector)?,
            by_observable: mapping_to_dict_object(py, by_observable)?,
            signed_by_observable: mapping_to_dict_object(py, signed_by_observable)?,
            by_location: mapping_to_dict_object(py, by_location)?,
            signed_by_location: mapping_to_dict_object(py, signed_by_location)?,
        })
    }

    #[getter]
    pub(crate) fn edge_hotspots(&self, py: Python<'_>) -> PyObject {
        self.edge_hotspots.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_detector_edge(&self, py: Python<'_>) -> PyObject {
        self.by_detector_edge.clone_ref(py)
    }

    #[getter]
    pub(crate) fn signed_by_detector_edge(&self, py: Python<'_>) -> PyObject {
        self.signed_by_detector_edge.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_detector(&self, py: Python<'_>) -> PyObject {
        self.by_detector.clone_ref(py)
    }

    #[getter]
    pub(crate) fn signed_by_detector(&self, py: Python<'_>) -> PyObject {
        self.signed_by_detector.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_observable(&self, py: Python<'_>) -> PyObject {
        self.by_observable.clone_ref(py)
    }

    #[getter]
    pub(crate) fn signed_by_observable(&self, py: Python<'_>) -> PyObject {
        self.signed_by_observable.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_location(&self, py: Python<'_>) -> PyObject {
        self.by_location.clone_ref(py)
    }

    #[getter]
    pub(crate) fn signed_by_location(&self, py: Python<'_>) -> PyObject {
        self.signed_by_location.clone_ref(py)
    }

    #[pyo3(signature = (top_k=10))]
    pub(crate) fn top_edges(&self, py: Python<'_>, top_k: usize) -> PyResult<PyObject> {
        let mut rows: Vec<(f64, Py<PyAny>)> = Vec::new();
        for row in self.edge_hotspots.bind(py).try_iter()? {
            let row = row?;
            rows.push((row.getattr("hotspot")?.extract::<f64>()?, row.into()));
        }
        rows.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let out = PyList::empty(py);
        for (_, row) in rows.into_iter().take(top_k) {
            out.append(row)?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (top_k=10))]
    pub(crate) fn edge_table(&self, py: Python<'_>, top_k: usize) -> PyResult<String> {
        let rows = self.top_edges(py, top_k)?;
        let mut lines = vec![
            "edge_index\tlocation_id\tevent\tsensitivity\thotspot\tdetectors\tobservables"
                .to_string(),
        ];
        for row in rows.bind(py).try_iter()? {
            let row = row?;
            let edge_index = row.getattr("edge_index")?.extract::<usize>()?;
            let location_id = row.getattr("location_id")?.extract::<String>()?;
            let event = row.getattr("event")?.str()?.to_str()?.to_string();
            let sensitivity = row.getattr("sensitivity")?.extract::<f64>()?;
            let hotspot = row.getattr("hotspot")?.extract::<f64>()?;
            let sensitivity = py_format_float(py, sensitivity, ".6g")?;
            let hotspot = py_format_float(py, hotspot, ".6g")?;
            let detectors = row.getattr("detectors")?.repr()?.to_str()?.to_string();
            let observables = row.getattr("observables")?.repr()?.to_str()?.to_string();
            lines.push(format!(
                "{edge_index}\t{location_id}\t{event}\t{sensitivity}\t{hotspot}\t{detectors}\t{observables}"
            ));
        }
        Ok(lines.join("\n"))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn detector_graph_edge_hotspot_object(
    py: Python<'_>,
    edge_index: usize,
    location_id: String,
    event: Py<PyAny>,
    probability: f64,
    detectors: Py<PyAny>,
    observables: Py<PyAny>,
    sensitivity: f64,
    hotspot: f64,
    weight: f64,
) -> PyResult<PyObject> {
    Ok(Py::new(
        py,
        PyDetectorGraphEdgeHotspot {
            edge_index,
            location_id,
            event,
            probability,
            detectors,
            observables,
            sensitivity,
            hotspot,
            weight,
        },
    )?
    .into_any())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn detector_graph_hotspots_object(
    py: Python<'_>,
    edge_hotspots: Py<PyAny>,
    by_detector_edge: Py<PyAny>,
    signed_by_detector_edge: Py<PyAny>,
    by_detector: Py<PyAny>,
    signed_by_detector: Py<PyAny>,
    by_observable: Py<PyAny>,
    signed_by_observable: Py<PyAny>,
    by_location: Py<PyAny>,
    signed_by_location: Py<PyAny>,
) -> PyResult<PyObject> {
    Ok(Py::new(
        py,
        PyDetectorGraphHotspots {
            edge_hotspots,
            by_detector_edge,
            signed_by_detector_edge,
            by_detector,
            signed_by_detector,
            by_observable,
            signed_by_observable,
            by_location,
            signed_by_location,
        },
    )?
    .into_any())
}

#[pyclass(name = "DemHotspotEstimate", module = "faultscope._native", frozen)]
pub(crate) struct PyDemHotspotEstimate {
    dem: Py<PyAny>,
    shots: usize,
    mean_loss: f64,
    baseline: f64,
    edge_sensitivities: Py<PyAny>,
    edge_hotspots: Py<PyAny>,
    sensitivities: Py<PyAny>,
    hotspots: Py<PyAny>,
    by_detector: Py<PyAny>,
    by_round: Py<PyAny>,
    by_gate: Py<PyAny>,
    by_operation: Py<PyAny>,
    locations: Py<PyAny>,
    detector_graph_hotspots: Py<PyAny>,
    top_edges_cache: Py<PyAny>,
    top_hotspots_cache: Py<PyAny>,
}

#[pymethods]
impl PyDemHotspotEstimate {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        dem,
        shots,
        mean_loss,
        baseline,
        edge_sensitivities,
        edge_hotspots,
        sensitivities,
        hotspots,
        by_detector,
        by_round,
        by_gate,
        by_operation,
        locations,
        detector_graph_hotspots,
        top_edges_cache=None,
        top_hotspots_cache=None
    ))]
    pub(crate) fn new(
        py: Python<'_>,
        dem: Py<PyAny>,
        shots: usize,
        mean_loss: f64,
        baseline: f64,
        edge_sensitivities: &Bound<'_, PyAny>,
        edge_hotspots: &Bound<'_, PyAny>,
        sensitivities: &Bound<'_, PyAny>,
        hotspots: &Bound<'_, PyAny>,
        by_detector: &Bound<'_, PyAny>,
        by_round: &Bound<'_, PyAny>,
        by_gate: &Bound<'_, PyAny>,
        by_operation: &Bound<'_, PyAny>,
        locations: &Bound<'_, PyAny>,
        detector_graph_hotspots: Py<PyAny>,
        top_edges_cache: Option<&Bound<'_, PyAny>>,
        top_hotspots_cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            dem,
            shots,
            mean_loss,
            baseline,
            edge_sensitivities: mapping_to_dict_object(py, edge_sensitivities)?,
            edge_hotspots: mapping_to_dict_object(py, edge_hotspots)?,
            sensitivities: mapping_to_dict_object(py, sensitivities)?,
            hotspots: mapping_to_dict_object(py, hotspots)?,
            by_detector: mapping_to_dict_object(py, by_detector)?,
            by_round: mapping_to_dict_object(py, by_round)?,
            by_gate: mapping_to_dict_object(py, by_gate)?,
            by_operation: mapping_to_dict_object(py, by_operation)?,
            locations: mapping_to_dict_object(py, locations)?,
            detector_graph_hotspots,
            top_edges_cache: optional_sequence_to_tuple(py, top_edges_cache)?,
            top_hotspots_cache: optional_sequence_to_tuple(py, top_hotspots_cache)?,
        })
    }

    #[getter]
    pub(crate) fn dem(&self, py: Python<'_>) -> PyObject {
        self.dem.clone_ref(py)
    }

    #[getter]
    pub(crate) fn shots(&self) -> usize {
        self.shots
    }

    #[getter]
    pub(crate) fn mean_loss(&self) -> f64 {
        self.mean_loss
    }

    #[getter]
    pub(crate) fn logical_failure_rate(&self) -> f64 {
        self.mean_loss
    }

    #[getter]
    pub(crate) fn baseline(&self) -> f64 {
        self.baseline
    }

    #[getter]
    pub(crate) fn edge_sensitivities(&self, py: Python<'_>) -> PyObject {
        self.edge_sensitivities.clone_ref(py)
    }

    #[getter]
    pub(crate) fn edge_hotspots(&self, py: Python<'_>) -> PyObject {
        self.edge_hotspots.clone_ref(py)
    }

    #[getter]
    pub(crate) fn sensitivities(&self, py: Python<'_>) -> PyObject {
        self.sensitivities.clone_ref(py)
    }

    #[getter]
    pub(crate) fn hotspots(&self, py: Python<'_>) -> PyObject {
        self.hotspots.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_detector(&self, py: Python<'_>) -> PyObject {
        self.by_detector.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_round(&self, py: Python<'_>) -> PyObject {
        self.by_round.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_gate(&self, py: Python<'_>) -> PyObject {
        self.by_gate.clone_ref(py)
    }

    #[getter]
    pub(crate) fn by_operation(&self, py: Python<'_>) -> PyObject {
        self.by_operation.clone_ref(py)
    }

    #[getter]
    pub(crate) fn locations(&self, py: Python<'_>) -> PyObject {
        self.locations.clone_ref(py)
    }

    #[getter]
    pub(crate) fn detector_graph_hotspots(&self, py: Python<'_>) -> PyObject {
        self.detector_graph_hotspots.clone_ref(py)
    }

    #[getter]
    pub(crate) fn top_edges_cache(&self, py: Python<'_>) -> PyObject {
        self.top_edges_cache.clone_ref(py)
    }

    #[getter]
    pub(crate) fn top_hotspots_cache(&self, py: Python<'_>) -> PyObject {
        self.top_hotspots_cache.clone_ref(py)
    }

    #[pyo3(signature = (top_k=10))]
    pub(crate) fn top_edges(&self, py: Python<'_>, top_k: usize) -> PyResult<PyObject> {
        let cache = self.top_edges_cache.bind(py);
        let cache_len = cache.len()?;
        if cache_len > 0 && top_k <= cache_len {
            let out = PyList::empty(py);
            for item in cache.try_iter()?.take(top_k) {
                out.append(item?)?;
            }
            return Ok(out.into());
        }

        let mut rows: Vec<(f64, Py<PyDemEdgeHotspot>)> = Vec::new();
        let dem = self.dem.bind(py);
        if dem.is_none() {
            let graph_edges = self
                .detector_graph_hotspots
                .bind(py)
                .getattr("edge_hotspots")?;
            for item in graph_edges.try_iter()? {
                let item = item?;
                let hotspot = item.getattr("hotspot")?.extract::<f64>()?;
                let row = PyDemEdgeHotspot {
                    edge_index: item.getattr("edge_index")?.extract::<usize>()?,
                    location_id: item.getattr("location_id")?.extract::<String>()?,
                    event: item.getattr("event")?.into(),
                    probability: item.getattr("probability")?.extract::<f64>()?,
                    detectors: item.getattr("detectors")?.into(),
                    observables: item.getattr("observables")?.into(),
                    sensitivity: item.getattr("sensitivity")?.extract::<f64>()?,
                    hotspot,
                };
                rows.push((hotspot, Py::new(py, row)?));
            }
        } else {
            let edges = dem.getattr("edges")?;
            for (edge_index, edge) in edges.try_iter()?.enumerate() {
                let edge = edge?;
                let sensitivity = self
                    .edge_sensitivities
                    .bind(py)
                    .get_item(edge_index)?
                    .extract::<f64>()?;
                let hotspot = self
                    .edge_hotspots
                    .bind(py)
                    .get_item(edge_index)?
                    .extract::<f64>()?;
                let row = PyDemEdgeHotspot {
                    edge_index,
                    location_id: edge.getattr("location_id")?.extract::<String>()?,
                    event: edge.getattr("event")?.into(),
                    probability: edge.getattr("probability")?.extract::<f64>()?,
                    detectors: edge.getattr("detectors")?.into(),
                    observables: edge.getattr("observables")?.into(),
                    sensitivity,
                    hotspot,
                };
                rows.push((hotspot, Py::new(py, row)?));
            }
        }
        rows.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let out = PyList::empty(py);
        for (_, row) in rows.into_iter().take(top_k) {
            out.append(row)?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (top_k=10))]
    pub(crate) fn top_hotspots(&self, py: Python<'_>, top_k: usize) -> PyResult<PyObject> {
        let cache = self.top_hotspots_cache.bind(py);
        let cache_len = cache.len()?;
        if cache_len > 0 && top_k <= cache_len {
            let out = PyList::empty(py);
            for item in cache.try_iter()?.take(top_k) {
                out.append(item?)?;
            }
            return Ok(out.into());
        }

        let mut rows: Vec<(f64, Py<PyDemLocationHotspot>)> = Vec::new();
        let hotspots = self.hotspots.bind(py).downcast::<PyDict>()?;
        for (location_id, hotspot) in hotspots.iter() {
            let location_id = location_id.extract::<String>()?;
            let hotspot = hotspot.extract::<f64>()?;
            let sensitivity = self
                .sensitivities
                .bind(py)
                .get_item(location_id.as_str())?
                .extract::<f64>()?;
            let location = self.locations.bind(py).get_item(location_id.as_str())?;
            let row = PyDemLocationHotspot {
                location_id,
                sensitivity,
                hotspot,
                qubits: location.getattr("qubits")?.into(),
                tags: location.getattr("tags")?.into(),
            };
            rows.push((hotspot, Py::new(py, row)?));
        }
        rows.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let out = PyList::empty(py);
        for (_, row) in rows.into_iter().take(top_k) {
            out.append(row)?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (top_k=10))]
    pub(crate) fn hotspot_table(&self, py: Python<'_>, top_k: usize) -> PyResult<String> {
        let rows = self.top_hotspots(py, top_k)?;
        let mut lines = vec!["location_id\tsensitivity\thotspot\ttags".to_string()];
        for row in rows.bind(py).try_iter()? {
            let row = row?;
            let location_id = row.getattr("location_id")?.extract::<String>()?;
            let sensitivity = row.getattr("sensitivity")?.extract::<f64>()?;
            let hotspot = row.getattr("hotspot")?.extract::<f64>()?;
            let sensitivity = py
                .import("builtins")?
                .getattr("format")?
                .call1((sensitivity, ".6g"))?
                .extract::<String>()?;
            let hotspot = py
                .import("builtins")?
                .getattr("format")?
                .call1((hotspot, ".6g"))?
                .extract::<String>()?;
            let tags = py
                .import("builtins")?
                .getattr("dict")?
                .call1((row.getattr("tags")?,))?
                .repr()?
                .to_str()?
                .to_string();
            lines.push(format!("{location_id}\t{sensitivity}\t{hotspot}\t{tags}"));
        }
        Ok(lines.join("\n"))
    }
}

pub(crate) fn simulation_result_from_estimate(
    py: Python<'_>,
    estimate: &PackedEstimate,
    locations: &HashMap<String, Py<PyAny>>,
) -> PyResult<PyFailureEstimate> {
    let top_rows = PyList::empty(py);
    for location_id in &estimate.top_locations {
        let location = locations.get(location_id).ok_or_else(|| {
            PyValueError::new_err(format!(
                "native estimate references unknown noise location {location_id:?}"
            ))
        })?;
        top_rows.append(Py::new(
            py,
            PyFaultHotspot {
                location_id: location_id.clone(),
                sensitivity: *estimate.sensitivities.get(location_id).unwrap_or(&0.0),
                hotspot: *estimate.hotspots.get(location_id).unwrap_or(&0.0),
                qubits: location.bind(py).getattr("qubits")?.into(),
                tags: location.bind(py).getattr("tags")?.into(),
            },
        )?)?;
    }
    let locations_dict = PyDict::new(py);
    for (location_id, location) in locations {
        locations_dict.set_item(location_id, location.clone_ref(py))?;
    }
    Ok(PyFailureEstimate {
        shots: estimate.shots,
        mean_loss: estimate.mean_loss,
        baseline: estimate.baseline,
        sensitivities: string_f64_map_to_py(py, &estimate.sensitivities)?,
        hotspots: string_f64_map_to_py(py, &estimate.hotspots)?,
        by_qubit: usize_f64_map_to_py(py, &estimate.by_qubit)?,
        by_round: tag_f64_map_to_py(py, &estimate.by_round)?,
        by_gate: tag_f64_map_to_py(py, &estimate.by_gate)?,
        by_operation: tag_f64_map_to_py(py, &estimate.by_operation)?,
        locations: locations_dict.into(),
        losses: PyList::empty(py).into(),
        top_hotspots_cache: PyTuple::new(py, top_rows.iter())?.into(),
    })
}

pub(crate) fn dem_hotspot_result_from_estimate(
    py: Python<'_>,
    sampler: &NativeDemSampler,
    estimate: &DemEstimate,
) -> PyResult<PyDemHotspotEstimate> {
    let locations = PyDict::new(py);
    let mut location_ids = HashSet::<String>::new();
    for edge in &sampler.simulator.edges {
        if !location_ids.insert(edge.location_id.clone()) {
            continue;
        }
        locations.set_item(
            &edge.location_id,
            Py::new(
                py,
                PyDemLocationMetadata {
                    id: edge.location_id.clone(),
                    tags: string_tag_map_to_py(py, &edge.tags)?,
                    qubits: PyTuple::empty(py).into(),
                },
            )?,
        )?;
    }

    let top_edges = PyList::empty(py);
    for edge_index in &estimate.top_edges {
        top_edges.append(Py::new(
            py,
            dem_edge_hotspot_row(py, sampler, estimate, *edge_index)?,
        )?)?;
    }

    let top_hotspots = PyList::empty(py);
    for location_id in &estimate.top_locations {
        let location = locations.get_item(location_id.as_str())?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "native estimate references unknown DEM location {location_id:?}"
            ))
        })?;
        top_hotspots.append(Py::new(
            py,
            PyDemLocationHotspot {
                location_id: location_id.clone(),
                sensitivity: *estimate
                    .location_sensitivities
                    .get(location_id)
                    .unwrap_or(&0.0),
                hotspot: *estimate.location_hotspots.get(location_id).unwrap_or(&0.0),
                qubits: location.getattr("qubits")?.into(),
                tags: location.getattr("tags")?.into(),
            },
        )?)?;
    }

    Ok(PyDemHotspotEstimate {
        dem: sampler
            .py_dem
            .as_ref()
            .map(|py_dem| py_dem.clone_ref(py))
            .unwrap_or_else(|| py.None()),
        shots: estimate.shots,
        mean_loss: estimate.mean_loss,
        baseline: estimate.baseline,
        edge_sensitivities: vec_f64_map_to_py(py, &estimate.edge_sensitivities)?,
        edge_hotspots: vec_f64_map_to_py(py, &estimate.edge_hotspots)?,
        sensitivities: string_f64_map_to_py(py, &estimate.location_sensitivities)?,
        hotspots: string_f64_map_to_py(py, &estimate.location_hotspots)?,
        by_detector: i64_f64_map_to_py(py, &estimate.by_detector)?,
        by_round: tag_f64_map_to_py(py, &estimate.by_round)?,
        by_gate: tag_f64_map_to_py(py, &estimate.by_gate)?,
        by_operation: tag_f64_map_to_py(py, &estimate.by_operation)?,
        locations: locations.into(),
        detector_graph_hotspots: detector_graph_hotspots_to_py(py, sampler, estimate)?,
        top_edges_cache: PyTuple::new(py, top_edges.iter())?.into(),
        top_hotspots_cache: PyTuple::new(py, top_hotspots.iter())?.into(),
    })
}

fn vec_f64_map_to_py(py: Python<'_>, values: &[f64]) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (index, value) in values.iter().enumerate() {
        dict.set_item(index, *value)?;
    }
    Ok(dict.into())
}

fn dem_edge_hotspot_row(
    py: Python<'_>,
    sampler: &NativeDemSampler,
    estimate: &DemEstimate,
    edge_index: usize,
) -> PyResult<PyDemEdgeHotspot> {
    let edge = sampler
        .simulator
        .edges
        .get(edge_index)
        .ok_or_else(|| PyValueError::new_err(format!("unknown DEM edge index {edge_index}")))?;
    Ok(PyDemEdgeHotspot {
        edge_index,
        location_id: edge.location_id.clone(),
        event: dem_event_to_py(py, &edge.event)?,
        probability: edge.probability,
        detectors: PyTuple::new(py, edge.detectors.iter().copied())?.into(),
        observables: PyTuple::new(py, edge.observables.iter().copied())?.into(),
        sensitivity: estimate
            .edge_sensitivities
            .get(edge_index)
            .copied()
            .unwrap_or(0.0),
        hotspot: estimate
            .edge_hotspots
            .get(edge_index)
            .copied()
            .unwrap_or(0.0),
    })
}

fn detector_graph_hotspots_to_py(
    py: Python<'_>,
    sampler: &NativeDemSampler,
    estimate: &DemEstimate,
) -> PyResult<PyObject> {
    let edge_hotspots = PyList::empty(py);
    for edge_index in 0..estimate.edge_sensitivities.len() {
        let row = dem_edge_hotspot_row(py, sampler, estimate, edge_index)?;
        edge_hotspots.append(detector_graph_edge_hotspot_object(
            py,
            row.edge_index,
            row.location_id,
            row.event,
            row.probability,
            row.detectors,
            row.observables,
            row.sensitivity,
            row.hotspot,
            1.0_f64,
        )?)?;
    }

    detector_graph_hotspots_object(
        py,
        PyTuple::new(py, edge_hotspots.iter())?.into(),
        graph_key_f64_map_to_py(py, &estimate.detector_graph.by_detector_edge)?,
        graph_key_f64_map_to_py(py, &estimate.detector_graph.signed_by_detector_edge)?,
        i64_f64_map_to_py(py, &estimate.detector_graph.by_detector)?,
        i64_f64_map_to_py(py, &estimate.detector_graph.signed_by_detector)?,
        i64_f64_map_to_py(py, &estimate.detector_graph.by_observable)?,
        i64_f64_map_to_py(py, &estimate.detector_graph.signed_by_observable)?,
        string_f64_map_to_py(py, &estimate.detector_graph.by_location)?,
        string_f64_map_to_py(py, &estimate.detector_graph.signed_by_location)?,
    )
}

fn list_to_list_object(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    Ok(py
        .import("builtins")?
        .getattr("list")?
        .call1((value,))?
        .into())
}

fn optional_sequence_to_tuple(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    match value {
        Some(value) if !value.is_none() => sequence_to_tuple(py, value),
        _ => Ok(PyTuple::empty(py).into()),
    }
}

fn py_format_float(py: Python<'_>, value: f64, spec: &str) -> PyResult<String> {
    py.import("builtins")?
        .getattr("format")?
        .call1((value, spec))?
        .extract::<String>()
}

use crate::*;

#[pyclass]
pub(crate) struct NativePackedSampler {
    pub(crate) n_qubits: usize,
    pub(crate) runtime_operations: Vec<RunOp>,
    pub(crate) observables: Vec<DemObservableSpec>,
    pub(crate) noise_location_ids: Vec<String>,
    pub(crate) noise_locations: Vec<NoiseLocationSpec>,
    pub(crate) random_source_count: usize,
}

#[pyclass]
pub(crate) struct NativePackedBatch {
    pub(crate) state: RuntimeState,
}

#[pymethods]
impl NativePackedSampler {
    #[getter]
    pub(crate) fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    #[getter]
    pub(crate) fn operation_count(&self) -> usize {
        self.runtime_operations.len()
    }

    #[pyo3(signature = (shots, seed=None))]
    pub(crate) fn sample(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, true))?;
        state.to_py(py)
    }

    #[pyo3(signature = (shots, seed=None))]
    pub(crate) fn sample_measurements(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, false))?;
        map_to_py(py, &state.measurements)
    }

    #[pyo3(signature = (shots, seed=None))]
    pub(crate) fn sample_measurements_packed(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, false))?;
        state.measurements_to_packed_py(py)
    }

    #[pyo3(signature = (shots, seed=None))]
    pub(crate) fn run_native_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<NativePackedBatch> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, true))?;
        Ok(NativePackedBatch { state })
    }

    #[pyo3(signature = (batch, loss_mask, baseline=None, top_k=10))]
    pub(crate) fn estimate_hotspots(
        &self,
        py: Python<'_>,
        batch: PyRef<'_, NativePackedBatch>,
        loss_mask: &Bound<'_, PyAny>,
        baseline: Option<f64>,
        top_k: usize,
    ) -> PyResult<PyObject> {
        let loss_mask = py_int_to_mask(
            loss_mask,
            batch.state.all_mask.words.len(),
            batch.state.shots,
        )?;
        let estimate = compute_packed_estimate(self, &batch.state, &loss_mask, baseline, top_k);
        packed_estimate_to_py(py, &estimate)
    }
}

#[pymethods]
impl NativePackedBatch {
    #[getter]
    pub(crate) fn shots(&self) -> usize {
        self.state.shots
    }

    #[getter]
    pub(crate) fn all_mask(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_to_py(py, &self.state.all_mask)
    }

    #[getter]
    pub(crate) fn x_frame(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_vec_to_py(py, &self.state.x_frame)
    }

    #[getter]
    pub(crate) fn z_frame(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_vec_to_py(py, &self.state.z_frame)
    }

    #[getter]
    pub(crate) fn measurements(&self, py: Python<'_>) -> PyResult<PyObject> {
        map_to_py(py, &self.state.measurements)
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.state.detectors)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.state.observables)
    }

    #[getter]
    pub(crate) fn noise_event_masks(&self, py: Python<'_>) -> PyResult<PyObject> {
        map_to_py(py, &self.state.event_masks)
    }

    pub(crate) fn x_mask(&self, py: Python<'_>, qubit: usize) -> PyResult<PyObject> {
        let mask = self
            .state
            .x_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_to_py(py, mask)
    }

    pub(crate) fn z_mask(&self, py: Python<'_>, qubit: usize) -> PyResult<PyObject> {
        let mask = self
            .state
            .z_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_to_py(py, mask)
    }

    pub(crate) fn measurement_mask(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        let mask = self
            .state
            .measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        mask_to_py(py, mask)
    }

    pub(crate) fn bit(&self, mask: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
        py_int_bit(mask, shot)
    }

    pub(crate) fn measurement_bit(&self, key: &str, shot: usize) -> PyResult<u8> {
        let mask = self
            .state
            .measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        mask_bit(mask, shot)
    }

    pub(crate) fn detector_bit(&self, detector_id: i64, shot: usize) -> PyResult<u8> {
        let mask =
            self.state.detectors.get(&detector_id).ok_or_else(|| {
                PyValueError::new_err(format!("unknown detector id {detector_id}"))
            })?;
        mask_bit(mask, shot)
    }

    pub(crate) fn observable_bit(&self, observable_id: i64, shot: usize) -> PyResult<u8> {
        let mask = self.state.observables.get(&observable_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown observable id {observable_id}"))
        })?;
        mask_bit(mask, shot)
    }

    pub(crate) fn x_bit(&self, qubit: usize, shot: usize) -> PyResult<u8> {
        let mask = self
            .state
            .x_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_bit(mask, shot)
    }

    pub(crate) fn z_bit(&self, qubit: usize, shot: usize) -> PyResult<u8> {
        let mask = self
            .state
            .z_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_bit(mask, shot)
    }
}

#[derive(Clone)]
#[pyclass]
pub(crate) struct NativeDemSampler {
    pub(crate) detectors: Arc<Vec<i64>>,
    pub(crate) observables: Arc<Vec<i64>>,
    pub(crate) edges: Arc<Vec<DemEdgeSpec>>,
    pub(crate) location_groups: Arc<Vec<DemLocationGroup>>,
}

#[pyclass]
pub(crate) struct NativeDemGenerator {
    pub(crate) n_qubits: usize,
    pub(crate) operations: Vec<Op>,
    pub(crate) detectors: Vec<DemDetectorSpec>,
    pub(crate) observables: Vec<DemObservableSpec>,
    pub(crate) measurement_plan: DemMeasurementPlan,
    pub(crate) event_plan: DemEventPlan,
    pub(crate) sampling_sampler: NativeDemSampler,
    pub(crate) use_product_fast_path: bool,
}

#[pyclass]
pub(crate) struct NativeGeneratedDem {
    pub(crate) edges: Vec<GeneratedDemEdge>,
}

#[pyclass]
pub(crate) struct NativeDemBatch {
    pub(crate) batch: DemBatch,
}

#[pymethods]
impl NativeDemSampler {
    #[getter]
    pub(crate) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[pyo3(signature = (shots, seed=None, return_edge_events=true))]
    pub(crate) fn run_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        return_edge_events: bool,
    ) -> PyResult<PyObject> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let batch = py.allow_threads(|| {
            let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
            run_dem_batch(self, shots, &mut rng, return_edge_events)
        });
        dem_batch_to_py(py, &batch)
    }

    #[pyo3(signature = (shots, seed=None))]
    pub(crate) fn run_native_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<NativeDemBatch> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let batch = py.allow_threads(|| {
            let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
            run_dem_batch(self, shots, &mut rng, true)
        });
        Ok(NativeDemBatch { batch })
    }

    #[pyo3(signature = (shots, seed=None, baseline=None, top_k=10))]
    pub(crate) fn estimate_default(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        baseline: Option<f64>,
        top_k: usize,
    ) -> PyResult<PyObject> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let estimate = py.allow_threads(|| {
            let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
            let batch = run_dem_batch(self, shots, &mut rng, true);
            compute_dem_estimate(self, &batch, &batch.loss_mask, baseline, top_k)
        });
        dem_estimate_to_py(py, &estimate)
    }

    #[pyo3(signature = (batch, loss_mask, baseline=None, top_k=10))]
    pub(crate) fn estimate_hotspots(
        &self,
        py: Python<'_>,
        batch: PyRef<'_, NativeDemBatch>,
        loss_mask: &Bound<'_, PyAny>,
        baseline: Option<f64>,
        top_k: usize,
    ) -> PyResult<PyObject> {
        let loss_mask = py_int_to_mask(
            loss_mask,
            batch.batch.all_mask.words.len(),
            batch.batch.shots,
        )?;
        let estimate = compute_dem_estimate(self, &batch.batch, &loss_mask, baseline, top_k);
        dem_estimate_to_py(py, &estimate)
    }
}

#[pymethods]
impl NativeDemGenerator {
    #[pyo3(signature = ())]
    pub(crate) fn generate_native_dem(&self, py: Python<'_>) -> PyResult<NativeGeneratedDem> {
        let edges = py.allow_threads(|| self.generate_edges())?;
        Ok(NativeGeneratedDem { edges })
    }

    #[pyo3(signature = ())]
    pub(crate) fn generate_dem(&self, py: Python<'_>) -> PyResult<PyObject> {
        let edges = py.allow_threads(|| self.generate_edges())?;
        dem_edges_to_py(py, &edges)
    }

    #[pyo3(signature = ())]
    pub(crate) fn compile_sampler(&self, py: Python<'_>) -> PyResult<NativeDemSampler> {
        py.allow_threads(|| Ok(self.sampling_sampler.clone()))
    }

    #[pyo3(signature = ())]
    pub(crate) fn generate_and_compile_sampler(&self, py: Python<'_>) -> PyResult<PyObject> {
        let edges = py.allow_threads(|| self.generate_edges())?;
        let sampler = generated_edges_to_native_dem_sampler(
            &edges,
            &self.operations,
            &self.detectors,
            &self.observables,
            true,
        );
        let out = PyDict::new(py);
        out.set_item("edges", dem_edges_to_py(py, &edges)?)?;
        out.set_item("sampler", Py::new(py, sampler)?)?;
        Ok(out.into())
    }
}

impl NativeDemGenerator {
    pub(crate) fn generate_edges(&self) -> PyResult<Vec<GeneratedDemEdge>> {
        if self.use_product_fast_path {
            generate_indexed_product_dem_edges_from_plan(
                self.n_qubits,
                &self.operations,
                &self.measurement_plan,
                &self.event_plan,
            )
        } else {
            generate_dem_edges(
                self.n_qubits,
                &self.operations,
                &self.detectors,
                &self.observables,
            )
        }
    }
}

#[pymethods]
impl NativeGeneratedDem {
    #[getter]
    pub(crate) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub(crate) fn to_rows(&self, py: Python<'_>) -> PyResult<PyObject> {
        dem_edges_to_py(py, &self.edges)
    }
}

#[pymethods]
impl NativeDemBatch {
    #[getter]
    pub(crate) fn shots(&self) -> usize {
        self.batch.shots
    }

    #[getter]
    pub(crate) fn all_mask(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_to_py(py, &self.batch.all_mask)
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.batch.detectors)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.batch.observables)
    }

    #[getter]
    pub(crate) fn edge_event_masks(&self, py: Python<'_>) -> PyResult<PyObject> {
        let edge_masks = PyDict::new(py);
        for (edge_index, mask) in self.batch.edge_event_masks.iter().enumerate() {
            edge_masks.set_item(edge_index, mask_to_py(py, mask)?)?;
        }
        Ok(edge_masks.into())
    }

    pub(crate) fn bit(&self, mask: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
        py_int_bit(mask, shot)
    }

    pub(crate) fn detector_bit(&self, detector_id: i64, shot: usize) -> PyResult<u8> {
        let mask =
            self.batch.detectors.get(&detector_id).ok_or_else(|| {
                PyValueError::new_err(format!("unknown detector id {detector_id}"))
            })?;
        mask_bit(mask, shot)
    }

    pub(crate) fn observable_bit(&self, observable_id: i64, shot: usize) -> PyResult<u8> {
        let mask = self.batch.observables.get(&observable_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown observable id {observable_id}"))
        })?;
        mask_bit(mask, shot)
    }

    pub(crate) fn edge_event_bit(&self, edge_index: usize, shot: usize) -> PyResult<u8> {
        let mask =
            self.batch.edge_event_masks.get(edge_index).ok_or_else(|| {
                PyValueError::new_err(format!("unknown DEM edge index {edge_index}"))
            })?;
        mask_bit(mask, shot)
    }
}

#[pyfunction]
pub(crate) fn compile_sampler(spec: &Bound<'_, PyDict>) -> PyResult<NativePackedSampler> {
    let (n_qubits, operations) = parse_circuit_spec(spec)?;
    let observables = match spec.get_item("observables")? {
        Some(items) => parse_dem_observables(items.downcast::<PyList>()?)?,
        None => Vec::new(),
    };
    let compiled = compile_runtime_operations(n_qubits, operations)?;

    Ok(NativePackedSampler {
        n_qubits,
        runtime_operations: compiled.runtime_operations,
        observables,
        noise_location_ids: compiled.noise_location_ids,
        noise_locations: compiled.noise_locations,
        random_source_count: compiled.random_source_count,
    })
}

#[pyfunction]
pub(crate) fn operation_sequence_identity_key(
    py: Python<'_>,
    operations: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let len = if let Ok(items) = operations.downcast::<PyTuple>() {
        items.len()
    } else if let Ok(items) = operations.downcast::<PyList>() {
        items.len()
    } else {
        return Err(PyValueError::new_err(
            "operation sequence identity key requires a list or tuple",
        ));
    };
    let mut out = Vec::with_capacity(len * std::mem::size_of::<usize>());
    if let Ok(items) = operations.downcast::<PyTuple>() {
        for item in items.iter() {
            out.extend_from_slice(&(item.as_ptr() as usize).to_ne_bytes());
        }
    } else {
        let items = operations.downcast::<PyList>()?;
        for item in items.iter() {
            out.extend_from_slice(&(item.as_ptr() as usize).to_ne_bytes());
        }
    }
    Ok(PyBytes::new(py, &out).into())
}

#[pyfunction]
pub(crate) fn generate_dem(
    py: Python<'_>,
    spec: &Bound<'_, PyDict>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<PyObject> {
    let (n_qubits, operations) = parse_circuit_spec(spec)?;
    let detectors = parse_dem_detectors(detectors)?;
    let observables = parse_dem_observables(observables)?;
    let edges = generate_dem_edges(n_qubits, &operations, &detectors, &observables)?;
    dem_edges_to_py(py, &edges)
}

#[pyfunction]
pub(crate) fn generate_and_compile_dem_sampler(
    py: Python<'_>,
    spec: &Bound<'_, PyDict>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<PyObject> {
    let (n_qubits, operations) = parse_circuit_spec(spec)?;
    let detectors = parse_dem_detectors(detectors)?;
    let observables = parse_dem_observables(observables)?;
    let generated_edges = generate_dem_edges(n_qubits, &operations, &detectors, &observables)?;
    let edge_specs = generated_edges_to_dem_edge_specs(&generated_edges, &operations);
    let location_groups = build_dem_location_groups(&edge_specs);
    let sampler = NativeDemSampler {
        detectors: Arc::new(detectors.iter().map(|detector| detector.id).collect()),
        observables: Arc::new(observables.iter().map(|observable| observable.id).collect()),
        edges: Arc::new(edge_specs),
        location_groups: Arc::new(location_groups),
    };

    let out = PyDict::new(py);
    out.set_item("edges", dem_edges_to_py(py, &generated_edges)?)?;
    out.set_item("sampler", Py::new(py, sampler)?)?;
    Ok(out.into())
}

#[pyfunction]
pub(crate) fn compile_generated_dem_sampler(
    spec: &Bound<'_, PyDict>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<NativeDemSampler> {
    let (n_qubits, operations) = parse_circuit_spec(spec)?;
    let detectors = parse_dem_detectors(detectors)?;
    let observables = parse_dem_observables(observables)?;
    let edge_specs = if supports_product_reference_fast_path(&operations) {
        let measurement_plan = compile_dem_measurement_plan(&operations, &detectors, &observables)?;
        let event_plan = collect_dem_event_plan(&operations)?;
        generate_indexed_product_sampling_dem_edge_specs_from_plan(
            n_qubits,
            &operations,
            &measurement_plan,
            &event_plan,
        )?
    } else {
        generate_sampling_dem_edge_specs(n_qubits, &operations, &detectors, &observables)?
    };
    Ok(sampling_edge_specs_to_native_dem_sampler(
        edge_specs,
        &detectors,
        &observables,
    ))
}

#[pyfunction]
pub(crate) fn compile_generated_dem_sampler_from_circuit(
    circuit: &Bound<'_, PyAny>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<NativeDemSampler> {
    let (n_qubits, operations) = parse_circuit_object(circuit, false)?;
    let detectors = parse_dem_detectors(detectors)?;
    let observables = parse_dem_observables(observables)?;
    let edge_specs = if supports_product_reference_fast_path(&operations) {
        let measurement_plan = compile_dem_measurement_plan(&operations, &detectors, &observables)?;
        let event_plan = collect_dem_event_plan(&operations)?;
        generate_indexed_product_sampling_dem_edge_specs_from_plan(
            n_qubits,
            &operations,
            &measurement_plan,
            &event_plan,
        )?
    } else {
        generate_sampling_dem_edge_specs(n_qubits, &operations, &detectors, &observables)?
    };
    Ok(sampling_edge_specs_to_native_dem_sampler(
        edge_specs,
        &detectors,
        &observables,
    ))
}

#[pyfunction]
pub(crate) fn generate_native_dem_from_circuit(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<NativeGeneratedDem> {
    let (n_qubits, operations) = parse_circuit_object(circuit, true)?;
    let detector_specs = parse_dem_detectors(detectors)?;
    let observable_specs = parse_dem_observables(observables)?;
    let edges = py.allow_threads(|| {
        generate_dem_edges(n_qubits, &operations, &detector_specs, &observable_specs)
    })?;
    Ok(NativeGeneratedDem { edges })
}

#[pyfunction]
pub(crate) fn compile_dem_generator_from_circuit(
    circuit: &Bound<'_, PyAny>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<NativeDemGenerator> {
    let (n_qubits, operations) = parse_circuit_object(circuit, true)?;
    compile_dem_generator_from_parts(n_qubits, operations, detectors, observables)
}

#[pyfunction]
pub(crate) fn compile_dem_generator(
    spec: &Bound<'_, PyDict>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<NativeDemGenerator> {
    let (n_qubits, operations) = parse_circuit_spec(spec)?;
    compile_dem_generator_from_parts(n_qubits, operations, detectors, observables)
}

pub(crate) fn compile_dem_generator_from_parts(
    n_qubits: usize,
    operations: Vec<Op>,
    detectors: &Bound<'_, PyList>,
    observables: &Bound<'_, PyList>,
) -> PyResult<NativeDemGenerator> {
    let detectors = parse_dem_detectors(detectors)?;
    let observables = parse_dem_observables(observables)?;
    let measurement_plan = compile_dem_measurement_plan(&operations, &detectors, &observables)?;
    let event_plan = collect_dem_event_plan(&operations)?;
    let use_product_fast_path = supports_product_reference_fast_path(&operations);
    let sampling_edge_specs = if use_product_fast_path {
        generate_indexed_product_sampling_dem_edge_specs_from_plan(
            n_qubits,
            &operations,
            &measurement_plan,
            &event_plan,
        )?
    } else {
        generate_sampling_dem_edge_specs_from_plan(
            n_qubits,
            &operations,
            &detectors,
            &observables,
            &event_plan,
        )?
    };
    let sampling_sampler =
        sampling_edge_specs_to_native_dem_sampler(sampling_edge_specs, &detectors, &observables);
    Ok(NativeDemGenerator {
        n_qubits,
        operations,
        detectors,
        observables,
        measurement_plan,
        event_plan,
        sampling_sampler,
        use_product_fast_path,
    })
}

#[pyfunction]
pub(crate) fn compile_dem_sampler(spec: &Bound<'_, PyDict>) -> PyResult<NativeDemSampler> {
    let detectors = required(spec, "detectors")?
        .downcast::<PyList>()?
        .iter()
        .map(|item| {
            item.downcast::<PyDict>()?
                .get_item("id")?
                .ok_or_else(|| PyValueError::new_err("native DEM detector missing id"))?
                .extract::<i64>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let observables = required(spec, "observables")?
        .downcast::<PyList>()?
        .iter()
        .map(|item| {
            item.downcast::<PyDict>()?
                .get_item("id")?
                .ok_or_else(|| PyValueError::new_err("native DEM observable missing id"))?
                .extract::<i64>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let edge_items_any = required(spec, "edges")?;
    let edge_items = edge_items_any.downcast::<PyList>()?;
    let mut edges = Vec::with_capacity(edge_items.len());
    for item in edge_items.iter() {
        let dict = item.downcast::<PyDict>()?;
        let probability = required(dict, "probability")?.extract::<f64>()?;
        if !(0.0..=1.0).contains(&probability) {
            return Err(PyValueError::new_err(format!(
                "DEM edge probability must be in [0, 1], got {probability}"
            )));
        }
        edges.push(DemEdgeSpec {
            probability,
            detectors: required(dict, "detectors")?.extract::<Vec<i64>>()?,
            observables: required(dict, "observables")?.extract::<Vec<i64>>()?,
            location_id: required(dict, "location_id")?.extract::<String>()?,
            tags: parse_optional_tags(dict)?,
        });
    }
    let location_groups = build_dem_location_groups(&edges);
    Ok(NativeDemSampler {
        detectors: Arc::new(detectors),
        observables: Arc::new(observables),
        edges: Arc::new(edges),
        location_groups: Arc::new(location_groups),
    })
}

pub(crate) fn generated_edges_to_dem_edge_specs(
    edges: &[GeneratedDemEdge],
    operations: &[Op],
) -> Vec<DemEdgeSpec> {
    let tags_by_location = noise_location_tags_by_id(operations);
    edges
        .iter()
        .map(|edge| DemEdgeSpec {
            probability: edge.probability,
            detectors: edge.detectors.clone(),
            observables: edge.observables.clone(),
            location_id: edge.location_id.clone(),
            tags: tags_by_location
                .get(&edge.location_id)
                .cloned()
                .unwrap_or_default(),
        })
        .collect()
}

pub(crate) fn generated_edges_to_native_dem_sampler(
    edges: &[GeneratedDemEdge],
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
    include_tags: bool,
) -> NativeDemSampler {
    if !include_tags {
        return generated_edges_to_sampling_dem_sampler(edges, detectors, observables);
    }
    let edge_specs = generated_edges_to_dem_edge_specs(edges, operations);
    let location_groups = build_dem_location_groups(&edge_specs);
    NativeDemSampler {
        detectors: Arc::new(detectors.iter().map(|detector| detector.id).collect()),
        observables: Arc::new(observables.iter().map(|observable| observable.id).collect()),
        edges: Arc::new(edge_specs),
        location_groups: Arc::new(location_groups),
    }
}

pub(crate) fn generated_edges_to_sampling_dem_sampler(
    edges: &[GeneratedDemEdge],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> NativeDemSampler {
    let edge_specs: Vec<DemEdgeSpec> = edges
        .iter()
        .map(|edge| DemEdgeSpec {
            probability: edge.probability,
            detectors: edge.detectors.clone(),
            observables: edge.observables.clone(),
            location_id: String::new(),
            tags: HashMap::new(),
        })
        .collect();
    NativeDemSampler {
        detectors: Arc::new(detectors.iter().map(|detector| detector.id).collect()),
        observables: Arc::new(observables.iter().map(|observable| observable.id).collect()),
        edges: Arc::new(edge_specs),
        location_groups: Arc::new(Vec::new()),
    }
}

pub(crate) fn sampling_edge_specs_to_native_dem_sampler(
    edge_specs: Vec<DemEdgeSpec>,
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> NativeDemSampler {
    NativeDemSampler {
        detectors: Arc::new(detectors.iter().map(|detector| detector.id).collect()),
        observables: Arc::new(observables.iter().map(|observable| observable.id).collect()),
        edges: Arc::new(edge_specs),
        location_groups: Arc::new(Vec::new()),
    }
}

pub(crate) fn noise_location_tags_by_id(
    operations: &[Op],
) -> HashMap<String, HashMap<String, TagValue>> {
    let mut out = HashMap::new();
    for operation in operations {
        for location in operation.noise_locations() {
            out.insert(location.id.clone(), location.tags.clone());
        }
    }
    out
}

#[pymodule]
pub(crate) fn _npsim_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", NATIVE_KERNEL_VERSION)?;
    module.add_class::<NativePackedSampler>()?;
    module.add_class::<NativePackedBatch>()?;
    module.add_class::<NativeDemSampler>()?;
    module.add_class::<NativeDemGenerator>()?;
    module.add_class::<NativeGeneratedDem>()?;
    module.add_class::<NativeDemBatch>()?;
    module.add_function(wrap_pyfunction!(compile_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(operation_sequence_identity_key, module)?)?;
    module.add_function(wrap_pyfunction!(generate_dem, module)?)?;
    module.add_function(wrap_pyfunction!(generate_and_compile_dem_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(compile_generated_dem_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(
        compile_generated_dem_sampler_from_circuit,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(generate_native_dem_from_circuit, module)?)?;
    module.add_function(wrap_pyfunction!(
        compile_dem_generator_from_circuit,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(compile_dem_generator, module)?)?;
    module.add_function(wrap_pyfunction!(compile_dem_sampler, module)?)?;
    Ok(())
}

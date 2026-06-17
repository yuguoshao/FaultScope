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

#[pyclass]
pub(crate) struct NativeDemSampler {
    pub(crate) detectors: Vec<i64>,
    pub(crate) observables: Vec<i64>,
    pub(crate) edges: Vec<DemEdgeSpec>,
    pub(crate) location_groups: Vec<DemLocationGroup>,
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
        detectors,
        observables,
        edges,
        location_groups,
    })
}

#[pymodule]
pub(crate) fn _npsim_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", NATIVE_KERNEL_VERSION)?;
    module.add_class::<NativePackedSampler>()?;
    module.add_class::<NativePackedBatch>()?;
    module.add_class::<NativeDemSampler>()?;
    module.add_class::<NativeDemBatch>()?;
    module.add_function(wrap_pyfunction!(compile_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(generate_dem, module)?)?;
    module.add_function(wrap_pyfunction!(compile_dem_sampler, module)?)?;
    Ok(())
}

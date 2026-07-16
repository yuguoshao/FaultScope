use crate::*;
use faultscope_core::{
    DemHotspotEstimator as CoreDemHotspotEstimator,
    DetectorErrorModelGenerator as CoreDetectorErrorModelGenerator,
};

/// Low-level native packed circuit sampler.
#[pyclass]
pub(crate) struct NativePackedSampler {
    pub(crate) py_circuit: Py<PyAny>,
    pub(crate) py_observables: Py<PyAny>,
    pub(crate) program: std::sync::Arc<faultscope_core::SamplerProgram>,
    pub(crate) py_noise_locations: Vec<Py<PyAny>>,
}

/// Opaque native packed batch used by native callback paths.
#[pyclass]
pub(crate) struct NativePackedBatch {
    pub(crate) state: RuntimeState,
    pub(crate) program: std::sync::Arc<faultscope_core::SamplerProgram>,
}

fn indexed_measurement_mask<'a>(
    program: &faultscope_core::SamplerProgram,
    state: &'a RuntimeState,
    key: &str,
) -> PyResult<&'a Mask> {
    let measurement_id = program
        .measurement_keys
        .iter()
        .position(|candidate| candidate == key)
        .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
    state
        .measurements
        .get(measurement_id)
        .and_then(Option::as_ref)
        .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))
}

/// Bit-packed forward circuit simulator and hotspot estimator.
#[pyclass(name = "FaultScopeSimulator", module = "faultscope._native")]
pub(crate) struct PyFaultScopeSimulator {
    sampler: NativePackedSampler,
}

/// Circuit-backed detector-error-model simulator.
#[pyclass(name = "DemFaultScopeSimulator", module = "faultscope._native")]
pub(crate) struct PyDemFaultScopeSimulator {
    py_circuit: Py<PyAny>,
    sampler: NativeDemSampler,
}

#[pymethods]
impl NativePackedSampler {
    #[getter]
    pub(crate) fn circuit(&self, py: Python<'_>) -> PyObject {
        self.py_circuit.clone_ref(py)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyObject {
        self.py_observables.clone_ref(py)
    }

    #[getter]
    pub(crate) fn n_qubits(&self) -> usize {
        self.program.n_qubits
    }

    #[getter]
    pub(crate) fn operation_count(&self) -> usize {
        self.program.operations.len()
    }

    #[getter]
    pub(crate) fn stored_operation_count(&self) -> usize {
        self.program.stored_operation_count
    }

    #[getter]
    pub(crate) fn loop_kernel_count(&self) -> usize {
        self.program.loop_kernel_count
    }

    #[pyo3(signature = (shots, seed=None, rng=None))]
    pub(crate) fn sample(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        rng: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PySampleBatch> {
        reject_python_rng(rng, "native sampler accepts seed, not a Python rng")?;
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, true))?;
        validate_declared_observables(&state.observables, &self.program.observables)?;
        batch_trajectory_from_state(py, &state, &self.program)
    }

    #[pyo3(signature = (shots, seed=None, rng=None))]
    pub(crate) fn sample_measurements(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        rng: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        reject_python_rng(rng, "native sampler accepts seed, not a Python rng")?;
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, false))?;
        measurement_masks_to_py(py, &self.program, &state.measurements)
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
        Ok(NativePackedBatch {
            state,
            program: self.program.clone(),
        })
    }

    #[pyo3(signature = (
        shots,
        loss_mask_fn=None,
        decoder=None,
        correction_mask_fn=None,
        seed=None,
        baseline=None,
        top_k=10
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn estimate(
        &self,
        py: Python<'_>,
        shots: usize,
        loss_mask_fn: Option<&Bound<'_, PyAny>>,
        decoder: Option<&Bound<'_, PyAny>>,
        correction_mask_fn: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        baseline: Option<&Bound<'_, PyAny>>,
        top_k: usize,
    ) -> PyResult<PyFailureEstimate> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        if decoder.is_some() && correction_mask_fn.is_some() {
            return Err(PyValueError::new_err(
                "supply either decoder or correction_mask_fn, not both",
            ));
        }
        let baseline = native_baseline_value(baseline)?;
        let native_decoder = match decoder {
            Some(decoder) => native_decoder_from_py(decoder)?,
            None => None,
        };
        if let Some(native_decoder) = native_decoder.as_deref() {
            let canonical_observable_ids = self
                .program
                .observables
                .iter()
                .map(|observable| observable.id)
                .collect::<Vec<_>>();
            validate_native_decoder_observable_layout(native_decoder, &canonical_observable_ids)?;
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, true))?;
        if loss_mask_fn.is_none() && correction_mask_fn.is_none() {
            if decoder.is_none() {
                let corrections = faultscope_core::CorrectionMaskBatch::empty(state.shots);
                validate_declared_observables(&state.observables, &self.program.observables)?;
                let observable_ids = self
                    .program
                    .observables
                    .iter()
                    .map(|observable| observable.id)
                    .collect::<Vec<_>>();
                let loss_mask = faultscope_core::logical_residual_loss_mask_native(
                    &state.observables,
                    &corrections,
                    &observable_ids,
                    &state.all_mask,
                );
                let estimate = compute_packed_estimate(self, &state, &loss_mask, baseline, top_k);
                return simulation_result_from_estimate(
                    py,
                    &estimate,
                    &self.program,
                    &self.py_noise_locations,
                );
            }
            if decoder.is_some() {
                if let Some(native_decoder) = native_decoder {
                    let detector_ids = native_decoder.detector_ids().to_vec();
                    let mut worker = native_decoder
                        .create_worker()
                        .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    let detector_masks =
                        detector_mask_view_from_map(&state.detectors, &detector_ids, state.shots)?;
                    let view = faultscope_core::DetectorMaskBatchView::new(
                        &detector_ids,
                        &detector_masks,
                        state.shots,
                    )
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    let corrections = worker
                        .decode_batch_checked(view)
                        .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    validate_declared_observables(&state.observables, &self.program.observables)?;
                    let observable_ids = self
                        .program
                        .observables
                        .iter()
                        .map(|observable| observable.id)
                        .collect::<Vec<_>>();
                    let loss_mask = faultscope_core::logical_residual_loss_mask_native(
                        &state.observables,
                        &corrections,
                        &observable_ids,
                        &state.all_mask,
                    );
                    let estimate =
                        compute_packed_estimate(self, &state, &loss_mask, baseline, top_k);
                    return simulation_result_from_estimate(
                        py,
                        &estimate,
                        &self.program,
                        &self.py_noise_locations,
                    );
                }
            }
        }
        let batch = Py::new(
            py,
            NativePackedBatch {
                state,
                program: self.program.clone(),
            },
        )?;
        let corrections = forward_correction_masks(py, &batch, decoder, correction_mask_fn)?;
        let loss_mask = if let Some(loss_mask_fn) = loss_mask_fn {
            call_forward_loss_mask_fn(py, loss_mask_fn, &batch, &corrections)?
        } else {
            let batch_ref = batch.bind(py).borrow();
            forward_default_loss_mask(py, &batch_ref, &corrections, &self.program.observables)?
        };
        let batch_ref = batch.bind(py).borrow();
        let estimate = compute_packed_estimate(self, &batch_ref.state, &loss_mask, baseline, top_k);
        simulation_result_from_estimate(py, &estimate, &self.program, &self.py_noise_locations)
    }

    #[pyo3(signature = (batch, loss_mask, baseline=None, top_k=10))]
    pub(crate) fn estimate_hotspots(
        &self,
        py: Python<'_>,
        batch: PyRef<'_, NativePackedBatch>,
        loss_mask: &Bound<'_, PyAny>,
        baseline: Option<f64>,
        top_k: usize,
    ) -> PyResult<PyFailureEstimate> {
        let loss_mask = py_int_to_mask(
            loss_mask,
            batch.state.all_mask.words.len(),
            batch.state.shots,
        )?;
        let estimate = compute_packed_estimate(self, &batch.state, &loss_mask, baseline, top_k);
        simulation_result_from_estimate(py, &estimate, &self.program, &self.py_noise_locations)
    }
}

#[pymethods]
impl PyFaultScopeSimulator {
    #[new]
    #[pyo3(signature = (circuit, *, observables=None))]
    pub(crate) fn new(
        py: Python<'_>,
        circuit: &Bound<'_, PyAny>,
        observables: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            sampler: native_packed_sampler_from_circuit(py, circuit, observables)?,
        })
    }

    #[getter]
    pub(crate) fn circuit(&self, py: Python<'_>) -> PyObject {
        self.sampler.py_circuit.clone_ref(py)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyObject {
        self.sampler.py_observables.clone_ref(py)
    }

    #[getter]
    pub(crate) fn locations(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        for (noise_id, location) in self.sampler.py_noise_locations.iter().enumerate() {
            let compiled = &self.sampler.program.noise_locations[noise_id];
            out.set_item(
                self.sampler
                    .program
                    .location_catalog
                    .label(compiled.location_id),
                location.clone_ref(py),
            )?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (*, shots, rng=None, seed=None))]
    pub(crate) fn run_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        rng: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<PySampleBatch> {
        let seed = seed_from_optional_rng(rng, seed)?;
        self.sampler.sample(py, shots, seed, None)
    }

    #[pyo3(signature = (shots, seed=None, rng=None))]
    pub(crate) fn sample(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        rng: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PySampleBatch> {
        let seed = seed_from_optional_rng(rng, seed)?;
        self.sampler.sample(py, shots, seed, None)
    }

    #[pyo3(signature = (shots, seed=None, rng=None))]
    pub(crate) fn sample_measurements(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        rng: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let seed = seed_from_optional_rng(rng, seed)?;
        self.sampler.sample_measurements(py, shots, seed, None)
    }

    #[pyo3(signature = (
        *,
        shots,
        loss_mask_fn=None,
        decoder=None,
        correction_mask_fn=None,
        seed=None,
        baseline=None,
        top_k=10
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn estimate(
        &self,
        py: Python<'_>,
        shots: usize,
        loss_mask_fn: Option<&Bound<'_, PyAny>>,
        decoder: Option<&Bound<'_, PyAny>>,
        correction_mask_fn: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        baseline: Option<&Bound<'_, PyAny>>,
        top_k: usize,
    ) -> PyResult<PyFailureEstimate> {
        self.sampler.estimate(
            py,
            shots,
            loss_mask_fn,
            decoder,
            correction_mask_fn,
            seed,
            baseline,
            top_k,
        )
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "FaultScopeSimulator(circuit={}, observables={})",
            self.sampler.py_circuit.bind(py).repr()?,
            self.sampler.py_observables.bind(py).repr()?,
        ))
    }
}

#[pymethods]
impl PyDemFaultScopeSimulator {
    #[new]
    #[pyo3(signature = (circuit, *, detectors=None, observables=None, materialize_dem=true))]
    pub(crate) fn new(
        py: Python<'_>,
        circuit: &Bound<'_, PyAny>,
        detectors: Option<&Bound<'_, PyAny>>,
        observables: Option<&Bound<'_, PyAny>>,
        materialize_dem: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            py_circuit: circuit.clone().unbind(),
            sampler: native_dem_sampler_from_circuit(
                py,
                circuit,
                detectors,
                observables,
                materialize_dem,
            )?,
        })
    }

    #[getter]
    pub(crate) fn circuit(&self, py: Python<'_>) -> PyObject {
        self.py_circuit.clone_ref(py)
    }

    #[getter]
    pub(crate) fn dem(&self, py: Python<'_>) -> PyObject {
        self.sampler.dem(py)
    }

    #[getter]
    pub(crate) fn edge_count(&self) -> usize {
        self.sampler.edge_count
    }

    #[pyo3(signature = (*, shots, rng=None, seed=None, return_edge_events=true))]
    pub(crate) fn run_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        rng: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        return_edge_events: bool,
    ) -> PyResult<PyDemSampleBatch> {
        let seed = seed_from_optional_rng(rng, seed)?;
        self.sampler
            .run_batch(py, shots, seed, return_edge_events, None)
    }

    #[pyo3(signature = (shots, seed=None, rng=None))]
    pub(crate) fn sample(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        rng: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDemSampleBatch> {
        let seed = seed_from_optional_rng(rng, seed)?;
        self.sampler.run_batch(py, shots, seed, true, None)
    }

    #[pyo3(signature = (shots, seed=None))]
    pub(crate) fn run_native_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<NativeDemBatch> {
        self.sampler.run_native_batch(py, shots, seed)
    }

    #[pyo3(signature = (shots, seed=None, baseline=None, top_k=10))]
    pub(crate) fn estimate_default(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        baseline: Option<&Bound<'_, PyAny>>,
        top_k: usize,
    ) -> PyResult<PyDemHotspotEstimate> {
        self.sampler
            .estimate_default(py, shots, seed, baseline, top_k)
    }

    #[pyo3(signature = (
        *,
        shots,
        seed=None,
        decoder=None,
        correction_mask_fn=None,
        loss_mask_fn=None,
        baseline=None,
        top_k=10,
        aggregate_hotspots=true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn estimate(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        decoder: Option<&Bound<'_, PyAny>>,
        correction_mask_fn: Option<&Bound<'_, PyAny>>,
        loss_mask_fn: Option<&Bound<'_, PyAny>>,
        baseline: Option<&Bound<'_, PyAny>>,
        top_k: usize,
        aggregate_hotspots: bool,
    ) -> PyResult<PyDemHotspotEstimate> {
        self.sampler.estimate(
            py,
            shots,
            seed,
            decoder,
            correction_mask_fn,
            loss_mask_fn,
            baseline,
            top_k,
            aggregate_hotspots,
        )
    }

    #[pyo3(signature = (batch, loss_mask, baseline=None, top_k=10))]
    pub(crate) fn estimate_hotspots(
        &self,
        py: Python<'_>,
        batch: PyRef<'_, NativeDemBatch>,
        loss_mask: &Bound<'_, PyAny>,
        baseline: Option<f64>,
        top_k: usize,
    ) -> PyResult<PyDemHotspotEstimate> {
        self.sampler
            .estimate_hotspots(py, batch, loss_mask, baseline, top_k)
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "DemFaultScopeSimulator(circuit={}, edge_count={})",
            self.py_circuit.bind(py).repr()?,
            self.sampler.edge_count,
        ))
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
        measurement_masks_to_py(py, &self.program, &self.state.measurements)
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
        noise_event_masks_to_py(py, &self.program, &self.state.event_masks)
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
        let mask = indexed_measurement_mask(&self.program, &self.state, key)?;
        mask_to_py(py, mask)
    }

    /// Materializes only the requested measurement masks for Python decoders.
    pub(crate) fn measurement_masks(
        &self,
        py: Python<'_>,
        keys: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        if keys.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(
                "measurement keys must be an iterable of strings, not a string",
            ));
        }
        let int_type = py.import("builtins")?.getattr("int")?;
        let from_bytes = int_type.getattr("from_bytes")?;
        let byteorder = PyString::new(py, "little");
        let mut bytes = Vec::new();
        let selected = PyDict::new(py);
        for key in PyIterator::from_object(keys)? {
            let key = key?.extract::<String>().map_err(|_| {
                PyTypeError::new_err("measurement keys must be an iterable of strings")
            })?;
            let mask = indexed_measurement_mask(&self.program, &self.state, &key)?;
            selected.set_item(
                &key,
                mask_to_py_with_converter(py, mask, &from_bytes, &byteorder, &mut bytes)?,
            )?;
        }
        Ok(selected.into())
    }

    pub(crate) fn bit(&self, mask: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
        py_int_bit(mask, shot)
    }

    pub(crate) fn measurement_bit(&self, key: &str, shot: usize) -> PyResult<u8> {
        let mask = indexed_measurement_mask(&self.program, &self.state, key)?;
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

/// Low-level native detector-error-model sampler.
#[pyclass]
pub(crate) struct NativeDemSampler {
    pub(crate) observables: Vec<i64>,
    pub(crate) edge_count: usize,
    pub(crate) simulator: CoreDemHotspotEstimator,
    pub(crate) py_dem: Option<Py<PyAny>>,
}

/// Reusable native detector-error-model generator.
#[pyclass]
pub(crate) struct NativeDemGenerator {
    generator: CoreDetectorErrorModelGenerator,
}

/// Opaque native DEM batch used by native decoder paths.
#[pyclass]
pub(crate) struct NativeDemBatch {
    pub(crate) batch: DemBatch,
}

/// Samples a detector error model and estimates edge/location hotspots.
#[pyclass(name = "DemHotspotEstimator", module = "faultscope._native")]
pub(crate) struct PyDemHotspotEstimator {
    sampler: NativeDemSampler,
}

#[pymethods]
impl NativeDemSampler {
    #[getter]
    pub(crate) fn dem(&self, py: Python<'_>) -> PyObject {
        match &self.py_dem {
            Some(dem) => dem.clone_ref(py),
            None => py.None(),
        }
    }

    #[getter]
    pub(crate) fn edge_count(&self) -> usize {
        self.edge_count
    }

    #[pyo3(signature = (shots, seed=None, return_edge_events=true, rng=None))]
    pub(crate) fn run_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        return_edge_events: bool,
        rng: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDemSampleBatch> {
        reject_python_rng(rng, "native DEM sampler accepts seed, not a Python rng")?;
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let batch = py.allow_threads(|| {
            let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
            run_dem_batch(self, shots, &mut rng, return_edge_events)
        });
        dem_batch_trajectory_from_batch(py, &batch)
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
        baseline: Option<&Bound<'_, PyAny>>,
        top_k: usize,
    ) -> PyResult<PyDemHotspotEstimate> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        self.require_dem_metadata()?;
        let baseline = native_baseline_value(baseline)?;
        let estimate = py.allow_threads(|| {
            let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
            let batch = run_dem_batch(self, shots, &mut rng, true);
            compute_dem_estimate(self, &batch, &batch.loss_mask, baseline, top_k)
        });
        dem_hotspot_result_from_estimate(py, self, &estimate)
    }

    #[pyo3(signature = (
        shots,
        seed=None,
        decoder=None,
        correction_mask_fn=None,
        loss_mask_fn=None,
        baseline=None,
        top_k=10,
        aggregate_hotspots=true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn estimate(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        decoder: Option<&Bound<'_, PyAny>>,
        correction_mask_fn: Option<&Bound<'_, PyAny>>,
        loss_mask_fn: Option<&Bound<'_, PyAny>>,
        baseline: Option<&Bound<'_, PyAny>>,
        top_k: usize,
        aggregate_hotspots: bool,
    ) -> PyResult<PyDemHotspotEstimate> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        self.require_dem_metadata()?;
        if decoder.is_some() && correction_mask_fn.is_some() {
            return Err(PyValueError::new_err(
                "supply either decoder or correction_mask_fn, not both",
            ));
        }
        let baseline = native_baseline_value(baseline)?;
        let native_decoder = match decoder {
            Some(decoder) => native_decoder_from_py(decoder)?,
            None => None,
        };
        if let Some(native_decoder) = native_decoder.as_deref() {
            validate_native_decoder_observable_layout(
                native_decoder,
                self.simulator.observable_ids(),
            )?;
        }
        if correction_mask_fn.is_none() && loss_mask_fn.is_none() {
            if decoder.is_some() {
                if let Some(native_decoder) = native_decoder {
                    let detector_ids = native_decoder.detector_ids().to_vec();
                    let mut worker = native_decoder
                        .create_worker()
                        .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    let estimate = py.allow_threads(|| {
                        let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
                        if !aggregate_hotspots && worker.supports_detector_event_batch() {
                            let event_batch = self
                                .simulator
                                .run_detector_event_shot_batch_with_rng(
                                    shots,
                                    &mut rng,
                                    &detector_ids,
                                    &self.observables,
                                )
                                .map_err(|err| PyValueError::new_err(err.to_string()))?;
                            let detector_view = faultscope_core::DetectorEventShotBatchView::new(
                                &detector_ids,
                                &event_batch.offsets,
                                &event_batch.events,
                                event_batch.shots,
                            )
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                            let corrections = worker
                                .decode_detector_event_batch_checked(detector_view)
                                .map_err(|err| PyValueError::new_err(err.to_string()))?;
                            let mean_loss = packed_residual_mean_loss_from_rows(
                                &event_batch.observable_ids,
                                &event_batch.observable_data,
                                event_batch.observable_byte_count,
                                &corrections,
                                event_batch.shots,
                            )?;
                            return Ok::<DemEstimate, PyErr>(dem_estimate_from_mean_loss(
                                event_batch.shots,
                                mean_loss,
                                baseline,
                            ));
                        }
                        if !aggregate_hotspots && worker.supports_packed_batch() {
                            let packed_batch = self
                                .simulator
                                .run_packed_shot_batch_with_rng(
                                    shots,
                                    &mut rng,
                                    &detector_ids,
                                    &self.observables,
                                )
                                .map_err(|err| PyValueError::new_err(err.to_string()))?;
                            let detector_view = faultscope_core::PackedDetectorShotBatchView::new(
                                &detector_ids,
                                &packed_batch.detector_data,
                                packed_batch.shots,
                            )
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                            let corrections = worker
                                .decode_packed_batch_checked(detector_view)
                                .map_err(|err| PyValueError::new_err(err.to_string()))?;
                            let mean_loss = packed_residual_mean_loss_from_rows(
                                &packed_batch.observable_ids,
                                &packed_batch.observable_data,
                                packed_batch.observable_byte_count,
                                &corrections,
                                packed_batch.shots,
                            )?;
                            return Ok::<DemEstimate, PyErr>(dem_estimate_from_mean_loss(
                                packed_batch.shots,
                                mean_loss,
                                baseline,
                            ));
                        }
                        let batch = run_dem_batch(self, shots, &mut rng, aggregate_hotspots);
                        let detector_masks = detector_mask_view_from_map(
                            &batch.detectors,
                            &detector_ids,
                            batch.shots,
                        )?;
                        let view = faultscope_core::DetectorMaskBatchView::new(
                            &detector_ids,
                            &detector_masks,
                            batch.shots,
                        )
                        .map_err(|err| PyValueError::new_err(err.to_string()))?;
                        let corrections = worker
                            .decode_batch_checked(view)
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                        let loss_mask = faultscope_core::logical_residual_loss_mask_native(
                            &batch.observables,
                            &corrections,
                            &self.observables,
                            &batch.all_mask,
                        );
                        Ok::<DemEstimate, PyErr>(dem_estimate_from_loss_mask(
                            self,
                            &batch,
                            &loss_mask,
                            baseline,
                            top_k,
                            aggregate_hotspots,
                        ))
                    })?;
                    return dem_hotspot_result_from_estimate(py, self, &estimate);
                }
            } else {
                let estimate = py.allow_threads(|| {
                    let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
                    let batch = run_dem_batch(self, shots, &mut rng, aggregate_hotspots);
                    dem_estimate_from_loss_mask(
                        self,
                        &batch,
                        &batch.loss_mask,
                        baseline,
                        top_k,
                        aggregate_hotspots,
                    )
                });
                return dem_hotspot_result_from_estimate(py, self, &estimate);
            }
        }
        let batch = py.allow_threads(|| {
            let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
            run_dem_batch(self, shots, &mut rng, aggregate_hotspots)
        });
        let batch = Py::new(py, NativeDemBatch { batch })?;
        let corrections = dem_correction_masks(py, &batch, decoder, correction_mask_fn)?;
        let loss_mask = if let Some(loss_mask_fn) = loss_mask_fn {
            let value = loss_mask_fn.call1((batch.clone_ref(py), corrections.clone()))?;
            let batch_ref = batch.bind(py).borrow();
            py_value_to_mask(
                py,
                &value,
                batch_ref.batch.all_mask.words.len(),
                batch_ref.batch.shots,
            )?
        } else {
            let batch_ref = batch.bind(py).borrow();
            dem_default_loss_mask(py, &batch_ref, &corrections, &self.observables)?
        };
        let batch_ref = batch.bind(py).borrow();
        let estimate = dem_estimate_from_loss_mask(
            self,
            &batch_ref.batch,
            &loss_mask,
            baseline,
            top_k,
            aggregate_hotspots,
        );
        dem_hotspot_result_from_estimate(py, self, &estimate)
    }

    #[pyo3(signature = (batch, loss_mask, baseline=None, top_k=10))]
    pub(crate) fn estimate_hotspots(
        &self,
        py: Python<'_>,
        batch: PyRef<'_, NativeDemBatch>,
        loss_mask: &Bound<'_, PyAny>,
        baseline: Option<f64>,
        top_k: usize,
    ) -> PyResult<PyDemHotspotEstimate> {
        self.require_dem_metadata()?;
        let loss_mask = py_int_to_mask(
            loss_mask,
            batch.batch.all_mask.words.len(),
            batch.batch.shots,
        )?;
        let estimate = compute_dem_estimate(self, &batch.batch, &loss_mask, baseline, top_k);
        dem_hotspot_result_from_estimate(py, self, &estimate)
    }

    fn require_dem_metadata(&self) -> PyResult<()> {
        if self.py_dem.is_none() {
            return Err(PyValueError::new_err(
                "this native DEM sampler was compiled without Python DEM metadata",
            ));
        }
        Ok(())
    }
}

fn dem_estimate_from_mean_loss(shots: usize, mean_loss: f64, baseline: Option<f64>) -> DemEstimate {
    DemEstimate {
        shots,
        mean_loss,
        baseline: baseline.unwrap_or(mean_loss),
        edge_sensitivities: Vec::new(),
        edge_hotspots: Vec::new(),
        location_sensitivities: HashMap::new(),
        location_hotspots: HashMap::new(),
        by_detector: HashMap::new(),
        by_round: HashMap::new(),
        by_gate: HashMap::new(),
        by_operation: HashMap::new(),
        detector_graph: faultscope_core::DetectorGraphEstimate {
            by_detector_edge: HashMap::new(),
            signed_by_detector_edge: HashMap::new(),
            by_detector: HashMap::new(),
            signed_by_detector: HashMap::new(),
            by_observable: HashMap::new(),
            signed_by_observable: HashMap::new(),
            by_location: HashMap::new(),
            signed_by_location: HashMap::new(),
        },
        top_edges: Vec::new(),
        top_locations: Vec::new(),
    }
}

fn dem_estimate_from_loss_mask(
    sampler: &NativeDemSampler,
    batch: &DemBatch,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
    aggregate_hotspots: bool,
) -> DemEstimate {
    if aggregate_hotspots {
        return compute_dem_estimate(sampler, batch, loss_mask, baseline, top_k);
    }

    let mut clipped_loss = loss_mask.clone();
    clipped_loss.and_assign(&batch.all_mask);
    let loss_count = clipped_loss.bit_count();
    let mean_loss = loss_count as f64 / batch.shots as f64;
    dem_estimate_from_mean_loss(batch.shots, mean_loss, baseline)
}

fn packed_residual_mean_loss_from_rows(
    observable_ids: &[i64],
    observable_data: &[u8],
    observable_byte_count: usize,
    corrections: &faultscope_core::PackedObservableShotBatch,
    shots: usize,
) -> PyResult<f64> {
    let failures = faultscope_core::packed_residual_failure_count(
        observable_ids,
        observable_data,
        observable_byte_count,
        corrections,
        shots,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(failures as f64 / shots as f64)
}

#[pymethods]
impl NativeDemGenerator {
    pub(crate) fn generate_dem(&self, py: Python<'_>) -> PyResult<PyDetectorErrorModel> {
        let dem = py
            .allow_threads(|| self.generator.generate_lazy())
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        detector_error_model_lazy_to_py(py, dem)
    }

    pub(crate) fn generate(&self, py: Python<'_>) -> PyResult<PyDetectorErrorModel> {
        self.generate_dem(py)
    }

    #[pyo3(signature = (*, materialize_dem=true))]
    pub(crate) fn compile_sampler(
        &self,
        py: Python<'_>,
        materialize_dem: bool,
    ) -> PyResult<NativeDemSampler> {
        native_dem_sampler_from_core_generator(py, &self.generator, materialize_dem)
    }
}

#[pymethods]
impl PyDemHotspotEstimator {
    #[new]
    pub(crate) fn new(dem: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            sampler: native_dem_sampler_from_dem(dem)?,
        })
    }

    #[getter]
    pub(crate) fn dem(&self, py: Python<'_>) -> PyObject {
        self.sampler.dem(py)
    }

    #[pyo3(signature = (*, shots, rng=None, seed=None, return_edge_events=true))]
    pub(crate) fn run_batch(
        &self,
        py: Python<'_>,
        shots: usize,
        rng: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        return_edge_events: bool,
    ) -> PyResult<PyDemSampleBatch> {
        let seed = seed_from_optional_rng(rng, seed)?;
        self.sampler
            .run_batch(py, shots, seed, return_edge_events, None)
    }

    #[pyo3(signature = (
        *,
        shots,
        seed=None,
        decoder=None,
        correction_mask_fn=None,
        loss_mask_fn=None,
        baseline=None,
        top_k=10,
        aggregate_hotspots=true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn estimate(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
        decoder: Option<&Bound<'_, PyAny>>,
        correction_mask_fn: Option<&Bound<'_, PyAny>>,
        loss_mask_fn: Option<&Bound<'_, PyAny>>,
        baseline: Option<&Bound<'_, PyAny>>,
        top_k: usize,
        aggregate_hotspots: bool,
    ) -> PyResult<PyDemHotspotEstimate> {
        self.sampler.estimate(
            py,
            shots,
            seed,
            decoder,
            correction_mask_fn,
            loss_mask_fn,
            baseline,
            top_k,
            aggregate_hotspots,
        )
    }

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let dem_repr = match &self.sampler.py_dem {
            Some(dem) => dem.bind(py).repr()?.to_str()?.to_string(),
            None => "None".to_string(),
        };
        Ok(format!("DemHotspotEstimator(dem={})", dem_repr,))
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

fn reject_python_rng(rng: Option<&Bound<'_, PyAny>>, message: &str) -> PyResult<()> {
    if rng.is_some_and(|value| !value.is_none()) {
        return Err(PyValueError::new_err(message.to_string()));
    }
    Ok(())
}

fn seed_from_optional_rng(
    rng: Option<&Bound<'_, PyAny>>,
    seed: Option<u64>,
) -> PyResult<Option<u64>> {
    let Some(rng) = rng else {
        return Ok(seed);
    };
    if rng.is_none() {
        return Ok(seed);
    }
    if seed.is_some() {
        return Err(PyValueError::new_err("supply either seed or rng, not both"));
    }
    rng.call_method1("getrandbits", (64,))?
        .extract::<u64>()
        .map(Some)
}

fn native_baseline_value(baseline: Option<&Bound<'_, PyAny>>) -> PyResult<Option<f64>> {
    let Some(baseline) = baseline else {
        return Ok(None);
    };
    if baseline.is_none() {
        return Ok(None);
    }
    if let Ok(value) = baseline.extract::<String>() {
        if value == "mean" {
            return Ok(None);
        }
        return Err(PyValueError::new_err(
            "baseline must be 'mean' or a numeric value",
        ));
    }
    baseline
        .extract::<f64>()
        .map(Some)
        .map_err(|_| PyValueError::new_err("baseline must be 'mean' or a numeric value"))
}

fn py_value_to_mask(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    words: usize,
    shots: usize,
) -> PyResult<Mask> {
    let int_value = py.import("builtins")?.getattr("int")?.call1((value,))?;
    py_int_to_mask(&int_value, words, shots)
}

fn mapping_to_dict_bound<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.call_method1("update", (value,))?;
    Ok(dict)
}

fn forward_correction_masks<'py>(
    py: Python<'py>,
    batch: &Py<NativePackedBatch>,
    decoder: Option<&Bound<'py, PyAny>>,
    correction_mask_fn: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    if let Some(correction_mask_fn) = correction_mask_fn {
        let value = correction_mask_fn.call1((batch.clone_ref(py),))?;
        return mapping_to_dict_bound(py, &value);
    }
    let Some(decoder) = decoder else {
        return Ok(PyDict::new(py));
    };
    if !decoder.hasattr("decode_batch_masks")? {
        return Err(PyTypeError::new_err(
            "batch decoder must provide decode_batch_masks(batch)",
        ));
    }
    let value = decoder.call_method1("decode_batch_masks", (batch.clone_ref(py),))?;
    mapping_to_dict_bound(py, &value)
}

fn dem_correction_masks<'py>(
    py: Python<'py>,
    batch: &Py<NativeDemBatch>,
    decoder: Option<&Bound<'py, PyAny>>,
    correction_mask_fn: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    if let Some(correction_mask_fn) = correction_mask_fn {
        let value = correction_mask_fn.call1((batch.clone_ref(py),))?;
        return mapping_to_dict_bound(py, &value);
    }
    let Some(decoder) = decoder else {
        return Ok(PyDict::new(py));
    };
    if !decoder.hasattr("decode_batch_masks")? {
        return Err(PyTypeError::new_err(
            "DEM decoder must provide decode_batch_masks(batch)",
        ));
    }
    let value = decoder.call_method1("decode_batch_masks", (batch.clone_ref(py),))?;
    mapping_to_dict_bound(py, &value)
}

fn call_forward_loss_mask_fn(
    py: Python<'_>,
    loss_mask_fn: &Bound<'_, PyAny>,
    batch: &Py<NativePackedBatch>,
    corrections: &Bound<'_, PyDict>,
) -> PyResult<Mask> {
    let value = if accepts_positional_args(py, loss_mask_fn, 2)? {
        loss_mask_fn.call1((batch.clone_ref(py), corrections))?
    } else {
        loss_mask_fn.call1((batch.clone_ref(py),))?
    };
    let batch_ref = batch.bind(py).borrow();
    py_value_to_mask(
        py,
        &value,
        batch_ref.state.all_mask.words.len(),
        batch_ref.state.shots,
    )
}

fn accepts_positional_args(
    py: Python<'_>,
    fn_obj: &Bound<'_, PyAny>,
    count: usize,
) -> PyResult<bool> {
    let inspect = py.import("inspect")?;
    let signature = match inspect.call_method1("signature", (fn_obj,)) {
        Ok(signature) => signature,
        Err(_) => return Ok(false),
    };
    let values = signature.getattr("parameters")?.call_method0("values")?;
    let mut positional = 0;
    for parameter in PyIterator::from_object(&values)? {
        let parameter = parameter?;
        let kind_name = parameter
            .getattr("kind")?
            .getattr("name")?
            .extract::<String>()?;
        if kind_name == "VAR_POSITIONAL" {
            return Ok(true);
        }
        if kind_name == "POSITIONAL_ONLY" || kind_name == "POSITIONAL_OR_KEYWORD" {
            positional += 1;
        }
    }
    Ok(positional >= count)
}

fn forward_default_loss_mask(
    py: Python<'_>,
    batch: &NativePackedBatch,
    corrections: &Bound<'_, PyDict>,
    observables: &[DemObservableSpec],
) -> PyResult<Mask> {
    validate_declared_observables(&batch.state.observables, observables)?;
    logical_residual_loss_mask(
        py,
        int_map_to_py(py, &batch.state.observables)?,
        corrections,
        observables.iter().map(|observable| observable.id).collect(),
        mask_to_py(py, &batch.state.all_mask)?,
        batch.state.all_mask.words.len(),
        batch.state.shots,
    )
}

fn dem_default_loss_mask(
    py: Python<'_>,
    batch: &NativeDemBatch,
    corrections: &Bound<'_, PyDict>,
    observable_ids: &[i64],
) -> PyResult<Mask> {
    logical_residual_loss_mask(
        py,
        int_map_to_py(py, &batch.batch.observables)?,
        corrections,
        observable_ids.to_vec(),
        mask_to_py(py, &batch.batch.all_mask)?,
        batch.batch.all_mask.words.len(),
        batch.batch.shots,
    )
}

fn logical_residual_loss_mask(
    py: Python<'_>,
    observables: PyObject,
    corrections: &Bound<'_, PyDict>,
    observable_ids: Vec<i64>,
    all_mask: PyObject,
    words: usize,
    shots: usize,
) -> PyResult<Mask> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("observable_ids", observable_ids)?;
    kwargs.set_item("all_mask", all_mask)?;
    let value = py
        .import("faultscope.runtime.loss")?
        .getattr("logical_residual_loss_mask")?
        .call((observables, corrections), Some(&kwargs))?;
    py_value_to_mask(py, &value, words, shots)
}

fn validate_declared_observables(
    observed: &HashMap<i64, Mask>,
    declared: &[DemObservableSpec],
) -> PyResult<()> {
    let missing: Vec<i64> = declared
        .iter()
        .map(|observable| observable.id)
        .filter(|observable_id| !observed.contains_key(observable_id))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "native batch did not return declared logical observables: {}",
        missing
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

fn py_tuple_from_optional_sequence(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    match value {
        Some(value) if !value.is_none() => Ok(py
            .import("builtins")?
            .getattr("tuple")?
            .call1((value,))?
            .unbind()),
        _ => Ok(PyTuple::empty(py).into_any().unbind()),
    }
}

#[pyfunction]
#[pyo3(signature = (circuit, observables=None))]
pub(crate) fn compile_sampler(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<NativePackedSampler> {
    native_packed_sampler_from_circuit(py, circuit, observables)
}

fn native_packed_sampler_from_circuit(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<NativePackedSampler> {
    let py_observables = py_tuple_from_optional_sequence(py, observables)?;
    let observables = match observables {
        Some(items) if !items.is_none() => parse_dem_observable_sequence(items)?,
        _ => Vec::new(),
    };
    let native_circuit = circuit.extract::<PyRef<'_, PyCircuit>>().ok();
    let shared_core_circuit = native_circuit
        .as_ref()
        .and_then(|circuit| circuit.core_circuit.clone());
    let fallback_core_circuit;
    let core_circuit = match shared_core_circuit.as_deref() {
        Some(core_circuit) => core_circuit,
        None => {
            fallback_core_circuit = parse_core_circuit_object(circuit)?;
            &fallback_core_circuit
        }
    };
    let program = std::sync::Arc::new(
        faultscope_core::compile_sampler_program_ref(
            core_circuit.n_qubits,
            &core_circuit.operations,
            observables,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?,
    );
    let py_noise_locations = if program.noise_locations.is_empty() {
        Vec::new()
    } else {
        parse_py_noise_locations(circuit, &program)?
    };

    Ok(NativePackedSampler {
        py_circuit: circuit.clone().unbind(),
        py_observables,
        program,
        py_noise_locations,
    })
}

#[pyfunction]
pub(crate) fn generate_dem(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    detectors: &Bound<'_, PyAny>,
    observables: &Bound<'_, PyAny>,
) -> PyResult<PyDetectorErrorModel> {
    let generator =
        core_dem_generator_from_circuit(py, circuit, Some(detectors), Some(observables))?;
    let dem = generator
        .generate_lazy()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    detector_error_model_lazy_to_py(py, dem)
}

#[pyfunction]
#[pyo3(signature = (circuit, detectors=None, observables=None))]
pub(crate) fn compile_dem_generator(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    detectors: Option<&Bound<'_, PyAny>>,
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<NativeDemGenerator> {
    Ok(NativeDemGenerator {
        generator: core_dem_generator_from_circuit(py, circuit, detectors, observables)?,
    })
}

#[pyfunction]
#[pyo3(signature = (circuit, detectors=None, observables=None))]
pub(crate) fn generate_and_compile_dem_sampler(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    detectors: Option<&Bound<'_, PyAny>>,
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let sampler = native_dem_sampler_from_circuit(py, circuit, detectors, observables, true)?;
    let dem = sampler
        .py_dem
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("missing generated DEM metadata"))?
        .clone_ref(py);
    let out = PyDict::new(py);
    out.set_item("dem", dem)?;
    out.set_item("sampler", Py::new(py, sampler)?)?;
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature = (circuit, detectors=None, observables=None))]
pub(crate) fn compile_generated_dem_sampler(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    detectors: Option<&Bound<'_, PyAny>>,
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<NativeDemSampler> {
    native_dem_sampler_from_circuit(py, circuit, detectors, observables, false)
}

#[pyfunction]
pub(crate) fn compile_dem_sampler(dem: &Bound<'_, PyAny>) -> PyResult<NativeDemSampler> {
    native_dem_sampler_from_dem(dem)
}

fn native_dem_sampler_from_circuit(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    detectors: Option<&Bound<'_, PyAny>>,
    observables: Option<&Bound<'_, PyAny>>,
    materialize_dem: bool,
) -> PyResult<NativeDemSampler> {
    let generator = core_dem_generator_from_circuit(py, circuit, detectors, observables)?;
    native_dem_sampler_from_core_generator(py, &generator, materialize_dem)
}

pub(crate) fn core_dem_generator_from_circuit(
    _py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
    detectors: Option<&Bound<'_, PyAny>>,
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<CoreDetectorErrorModelGenerator> {
    if let (Some(event_plan), Some(core_circuit)) = (
        cached_core_event_plan(circuit),
        cached_core_circuit(circuit),
    ) {
        let detector_specs = match detectors {
            Some(items) if !items.is_none() => Some(parse_dem_detector_sequence(items)?),
            _ => None,
        };
        let observable_specs = match observables {
            Some(items) if !items.is_none() => Some(parse_dem_observable_sequence(items)?),
            _ => None,
        };
        let detector_specs = detector_specs.unwrap_or_else(|| event_plan.inferred_detectors());
        let observable_specs =
            observable_specs.unwrap_or_else(|| event_plan.inferred_observables());
        return CoreDetectorErrorModelGenerator::new_with_shared_event_plan(
            core_circuit,
            detector_specs,
            observable_specs,
            event_plan,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()));
    }

    let cached_event_plan = cached_core_event_plan(circuit);
    let core_circuit = parse_core_circuit_object(circuit)?;
    let detector_specs = match detectors {
        Some(items) if !items.is_none() => Some(parse_dem_detector_sequence(items)?),
        _ => None,
    };
    let observable_specs = match observables {
        Some(items) if !items.is_none() => Some(parse_dem_observable_sequence(items)?),
        _ => None,
    };
    match cached_event_plan {
        Some(event_plan) => CoreDetectorErrorModelGenerator::new_with_shared_event_plan_options(
            std::sync::Arc::new(core_circuit),
            detector_specs,
            observable_specs,
            event_plan,
        ),
        None => {
            CoreDetectorErrorModelGenerator::new(core_circuit, detector_specs, observable_specs)
        }
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn native_dem_sampler_from_core_generator(
    py: Python<'_>,
    generator: &CoreDetectorErrorModelGenerator,
    materialize_dem: bool,
) -> PyResult<NativeDemSampler> {
    let lazy_dem = generator
        .generate_lazy()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let (simulator, py_dem) = if materialize_dem {
        let simulator = lazy_dem.compile_hotspot_estimator();
        let py_dem = Py::new(py, detector_error_model_lazy_to_py(py, lazy_dem)?)?.into_any();
        (simulator, Some(py_dem))
    } else {
        (lazy_dem.into_sampling_estimator(), None)
    };
    let edge_count = simulator.edge_count();
    let observables = simulator.observable_ids().to_vec();
    Ok(NativeDemSampler {
        observables,
        edge_count,
        simulator,
        py_dem,
    })
}

fn native_dem_sampler_from_dem(dem: &Bound<'_, PyAny>) -> PyResult<NativeDemSampler> {
    if let Ok(native_dem) = dem.extract::<PyRef<'_, PyDetectorErrorModel>>() {
        let core_dem = native_dem.to_core_dem(dem.py())?;
        let detectors: Vec<i64> = core_dem
            .detectors
            .iter()
            .map(|detector| detector.id)
            .collect();
        let observables: Vec<i64> = core_dem
            .observables
            .iter()
            .map(|observable| observable.id)
            .collect();
        let edges: Vec<DemEdgeSpec> = core_dem
            .edges
            .into_iter()
            .map(|edge| DemEdgeSpec {
                probability: edge.probability,
                detectors: edge.detectors,
                observables: edge.observables,
                location_id: edge.location_id,
                event: edge.event,
                tags: edge.tags,
            })
            .collect();
        return native_dem_sampler_from_parts(
            detectors,
            observables,
            edges,
            Some(dem.clone().unbind()),
        );
    }
    let detector_specs =
        parse_dem_detector_sequence(&required_attr(dem, "detectors", "DetectorErrorModel")?)?;
    let observable_specs =
        parse_dem_observable_sequence(&required_attr(dem, "observables", "DetectorErrorModel")?)?;
    let detectors: Vec<i64> = detector_specs.iter().map(|detector| detector.id).collect();
    let observables: Vec<i64> = observable_specs
        .iter()
        .map(|observable| observable.id)
        .collect();
    let edges = parse_dem_edge_sequence(&required_attr(dem, "edges", "DetectorErrorModel")?)?;
    native_dem_sampler_from_parts(detectors, observables, edges, Some(dem.clone().unbind()))
}

fn native_dem_sampler_from_parts(
    detectors: Vec<i64>,
    observables: Vec<i64>,
    edges: Vec<DemEdgeSpec>,
    py_dem: Option<Py<PyAny>>,
) -> PyResult<NativeDemSampler> {
    let edge_count = edges.len();
    let simulator = if py_dem.is_some() {
        CoreDemHotspotEstimator::from_parts(detectors, observables, edges)
    } else {
        CoreDemHotspotEstimator::from_sampling_parts(detectors, observables, edges)
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let observables = simulator.observable_ids().to_vec();
    Ok(NativeDemSampler {
        observables,
        edge_count,
        simulator,
        py_dem,
    })
}

#[pymodule]
pub(crate) fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", NATIVE_KERNEL_VERSION)?;
    module.add(
        "NATIVE_DECODER_PLUGIN_ABI_VERSION",
        faultscope_core::NATIVE_DECODER_PLUGIN_ABI_VERSION,
    )?;
    module.add(
        "NATIVE_DECODER_PLUGIN_ABI",
        faultscope_core::NATIVE_DECODER_PLUGIN_ABI_NAME,
    )?;
    module.add(
        "NATIVE_DECODER_PLUGIN_CAPSULE_NAME",
        faultscope_core::NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
    )?;
    module.add(
        "NATIVE_DECODER_PLUGIN_CAPSULE_METHOD",
        faultscope_core::NATIVE_DECODER_PLUGIN_CAPSULE_METHOD,
    )?;
    module.add(
        "NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP",
        faultscope_core::NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
    )?;
    module.add_class::<PyBernoulliPauliNoise>()?;
    module.add_class::<PyPauliChannel>()?;
    module.add_class::<PySingleQubitDepolarizing>()?;
    module.add_class::<PyTwoQubitDepolarizing>()?;
    module.add_class::<PyMeasurementBitFlip>()?;
    module.add_class::<PyPauliFrame>()?;
    module.add_class::<PyStabilizerState>()?;
    module.add_class::<PyCircuit>()?;
    module.add_class::<PyOperation>()?;
    module.add_class::<PyNoiseLocation>()?;
    module.add_class::<PyDetector>()?;
    module.add_class::<PyLogicalObservable>()?;
    module.add_class::<PyDetectorErrorEdge>()?;
    module.add_class::<PyDetectorErrorModel>()?;
    module.add_class::<PyDetectorErrorModelGenerator>()?;
    module.add_class::<PyIndexedDemEdge>()?;
    module.add_class::<PyIndexedDem>()?;
    module.add_class::<PyGraphlikeEdge>()?;
    module.add_class::<PyGraphlikeDecodingProblem>()?;
    module.add_class::<PySparseBinaryMatrix>()?;
    module.add_class::<PyBinaryLinearDecodingProblem>()?;
    module.add_class::<PyNativeBatchDecoder>()?;
    module.add_class::<PyNativeCompositeDecoder>()?;
    module.add_class::<PyNativeNoCorrectionDecoder>()?;
    module.add_class::<PyNativeGraphlikeDetectorCopyDecoder>()?;
    #[cfg(feature = "decoder-fusion-blossom")]
    module.add_class::<PyNativeFusionBlossomDecoder>()?;
    module.add_class::<PyDetectorGraphEdgeHotspot>()?;
    module.add_class::<PyDetectorGraphHotspots>()?;
    module.add_class::<PySampleBatch>()?;
    module.add_class::<PyDemSampleBatch>()?;
    module.add_class::<PyFaultHotspot>()?;
    module.add_class::<PyFailureEstimate>()?;
    module.add_class::<PyDemLocationMetadata>()?;
    module.add_class::<PyDemLocationHotspot>()?;
    module.add_class::<PyDemEdgeHotspot>()?;
    module.add_class::<PyDemHotspotEstimate>()?;
    module.add_class::<PyFaultScopeSimulator>()?;
    module.add_class::<PyDemFaultScopeSimulator>()?;
    module.add_class::<NativePackedSampler>()?;
    module.add_class::<NativePackedBatch>()?;
    module.add_class::<NativeDemSampler>()?;
    module.add_class::<NativeDemGenerator>()?;
    module.add_class::<NativeDemBatch>()?;
    module.add_class::<PyDemHotspotEstimator>()?;
    module.add_function(wrap_pyfunction!(compile_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(generate_dem, module)?)?;
    module.add_function(wrap_pyfunction!(available_native_decoders, module)?)?;
    module.add_function(wrap_pyfunction!(compile_dem_generator, module)?)?;
    module.add_function(wrap_pyfunction!(generate_and_compile_dem_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(compile_generated_dem_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(compile_dem_sampler, module)?)?;
    module.add_function(wrap_pyfunction!(
        _collect_dem_logical_error_stats_many,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_collect_dem_hotspots_many, module)?)?;
    Ok(())
}

use faultscope_core::{
    log_likelihood_ratio, FaultScopeNativeCorrectionMaskBatchMutViewV1, FaultScopeNativeDecoderI64SliceV1,
    FaultScopeNativeDecoderStatusV1, FaultScopeNativeDecoderStringViewV1, FaultScopeNativeDecoderV1,
    FaultScopeNativeDetectorMaskBatchViewV1, FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeNativePackedObservableShotBatchMutViewV1, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE, NATIVE_DECODER_PLUGIN_STATUS_ERROR,
    NATIVE_DECODER_PLUGIN_STATUS_OK,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyTuple};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::mem;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::Mutex;

const BACKEND_NAME: &str = "pymatching";
const CAPSULE_NAME: &[u8] = b"faultscope.native_decoder_plugin.v1\0";
const NUM_DISTINCT_WEIGHTS: f64 = (1u64 << 24) as f64;
const MAX_USER_EDGE_WEIGHT: f64 = NUM_DISTINCT_WEIGHTS - 1.0;

#[repr(C)]
struct PymatchingShimEdge {
    left: usize,
    right: usize,
    is_boundary: u8,
    weight: i32,
    observables: *const usize,
    observable_count: usize,
}

#[repr(C)]
struct PymatchingShimMaskView {
    words: *const u64,
    word_count: usize,
}

#[repr(C)]
struct PymatchingShimMaskMutView {
    words: *mut u64,
    word_count: usize,
}

enum PymatchingShimDecoder {}

extern "C" {
    fn faultscope_pymatching_decoder_new(
        detector_count: usize,
        observable_count: usize,
        edges: *const PymatchingShimEdge,
        edge_count: usize,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> *mut PymatchingShimDecoder;

    fn faultscope_pymatching_decoder_free(decoder: *mut PymatchingShimDecoder);

    fn faultscope_pymatching_decoder_decode_batch(
        decoder: *mut PymatchingShimDecoder,
        detector_masks: *const PymatchingShimMaskView,
        detector_count: usize,
        observable_masks: *mut PymatchingShimMaskMutView,
        observable_count: usize,
        shots: usize,
        word_count: usize,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> c_int;

    fn faultscope_pymatching_decoder_decode_packed_batch(
        decoder: *mut PymatchingShimDecoder,
        detector_shots: *const u8,
        detector_count: usize,
        detector_byte_count: usize,
        observable_predictions: *mut u8,
        observable_count: usize,
        observable_byte_count: usize,
        shots: usize,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> c_int;
}

#[pyclass(name = "NativePyMatchingNativeDecoder")]
struct PyNativePyMatchingNativeDecoder {
    capsule: Py<PyAny>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    edge_count: usize,
    solver_edge_count: usize,
    build_summary: BuildSummary,
}

#[pymethods]
impl PyNativePyMatchingNativeDecoder {
    #[staticmethod]
    fn from_dem(py: Python<'_>, dem: &Bound<'_, PyAny>) -> PyResult<Self> {
        let builder = PyMatchingProblemBuilder::from_dem_object(dem)?;
        build_py_native_decoder(py, builder)
    }

    #[staticmethod]
    fn from_graphlike_problem(py: Python<'_>, problem: &Bound<'_, PyAny>) -> PyResult<Self> {
        let builder = PyMatchingProblemBuilder::from_graphlike_problem(problem)?;
        build_py_native_decoder(py, builder)
    }

    #[getter]
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    #[getter]
    fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.detector_ids.iter().copied())?.into())
    }

    #[getter]
    fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.observable_ids.iter().copied())?.into())
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.edge_count
    }

    #[getter]
    fn solver_edge_count(&self) -> usize {
        self.solver_edge_count
    }

    #[getter]
    fn build_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        build_summary_to_py(py, &self.build_summary)
    }

    fn decode_batch_words(
        &self,
        py: Python<'_>,
        shots: usize,
        detector_words: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let word_count = shots.div_ceil(64);
        let mut detector_word_buffers = Vec::with_capacity(self.detector_ids.len());
        for detector_id in &self.detector_ids {
            let words = detector_words
                .get_item(*detector_id)?
                .ok_or_else(|| PyValueError::new_err(format!("missing detector id {detector_id}")))?
                .extract::<Vec<u64>>()?;
            if words.len() != word_count {
                return Err(PyValueError::new_err(format!(
                    "detector {detector_id} has {} words; expected {word_count}",
                    words.len()
                )));
            }
            detector_word_buffers.push(words);
        }

        let mut output_buffers = vec![vec![0; word_count]; self.observable_ids.len()];
        let input_views = detector_word_buffers
            .iter()
            .map(|words| faultscope_core::FaultScopeNativeDecoderMaskViewV1 {
                words: words.as_ptr(),
                word_count: words.len(),
            })
            .collect::<Vec<_>>();
        let mut output_views = output_buffers
            .iter_mut()
            .map(|words| faultscope_core::FaultScopeNativeDecoderMaskMutViewV1 {
                words: words.as_mut_ptr(),
                word_count: words.len(),
            })
            .collect::<Vec<_>>();
        let input = FaultScopeNativeDetectorMaskBatchViewV1 {
            detector_ids: self.detector_ids.as_ptr(),
            detector_count: self.detector_ids.len(),
            masks: input_views.as_ptr(),
            shots,
            word_count,
        };
        let mut output = FaultScopeNativeCorrectionMaskBatchMutViewV1 {
            observable_ids: self.observable_ids.as_ptr(),
            observable_count: self.observable_ids.len(),
            masks: output_views.as_mut_ptr(),
            shots,
            word_count,
        };
        let descriptor = unsafe {
            &*(pyo3::ffi::PyCapsule_GetPointer(
                self.capsule.as_ptr(),
                CAPSULE_NAME.as_ptr().cast::<c_char>(),
            )
            .cast::<FaultScopeNativeDecoderV1>())
        };
        let status = unsafe {
            descriptor
                .decode_batch
                .expect("pymatching descriptor has decode callback")(
                descriptor.state,
                &input,
                &mut output,
            )
        };
        if status.code != NATIVE_DECODER_PLUGIN_STATUS_OK {
            return Err(PyValueError::new_err(format!(
                "native decoder plugin error: {}",
                string_view_to_string(status.message)
            )));
        }

        let out = PyDict::new(py);
        for (observable_id, words) in self.observable_ids.iter().zip(output_buffers) {
            out.set_item(*observable_id, words)?;
        }
        Ok(out.into())
    }

    fn __faultscope_native_decoder_capsule__(&self, py: Python<'_>) -> Py<PyAny> {
        self.capsule.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NativePyMatchingNativeDecoder(name={:?}, detector_ids={:?}, observable_ids={:?}, solver_edge_count={})",
            BACKEND_NAME, self.detector_ids, self.observable_ids, self.solver_edge_count
        )
    }
}

fn build_py_native_decoder(
    py: Python<'_>,
    builder: PyMatchingProblemBuilder,
) -> PyResult<PyNativePyMatchingNativeDecoder> {
    let detector_ids = builder.detector_ids.clone();
    let observable_ids = builder.observable_ids.clone();
    let built = builder.build()?;
    let native =
        PymatchingNativeDecoder::new(detector_ids.len(), observable_ids.len(), &built.edges)?;
    let build_summary = BuildSummary {
        dem_edge_count: built.dem_edge_count,
        solver_edge_count: built.edges.len(),
        merged_parallel_edge_count: built.merged_parallel_edge_count,
        edges: built.edges,
    };
    let state = Box::new(DecoderState {
        detector_ids: detector_ids.clone(),
        observable_ids: observable_ids.clone(),
        native: Mutex::new(native),
        last_error: Mutex::new(CString::new("").expect("empty CString")),
    });
    let descriptor = Box::new(FaultScopeNativeDecoderV1 {
        abi_version: NATIVE_DECODER_PLUGIN_ABI_VERSION,
        struct_size: mem::size_of::<FaultScopeNativeDecoderV1>(),
        flags: NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
        state: Box::into_raw(state).cast::<c_void>(),
        drop_state: Some(drop_state),
        name: Some(decoder_name),
        detector_ids: Some(decoder_detector_ids),
        observable_ids: Some(decoder_observable_ids),
        decode_batch: Some(decoder_decode_batch),
        decode_packed_batch: Some(decoder_decode_packed_batch),
        decode_detector_event_batch: None,
    });
    let capsule = unsafe { create_decoder_capsule(py, Box::into_raw(descriptor))? };
    Ok(PyNativePyMatchingNativeDecoder {
        capsule,
        detector_ids,
        observable_ids,
        edge_count: build_summary.dem_edge_count,
        solver_edge_count: build_summary.solver_edge_count,
        build_summary,
    })
}

struct DecoderState {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    native: Mutex<PymatchingNativeDecoder>,
    last_error: Mutex<CString>,
}

struct PymatchingNativeDecoder {
    ptr: *mut PymatchingShimDecoder,
}

unsafe impl Send for PymatchingNativeDecoder {}

impl PymatchingNativeDecoder {
    fn new(
        detector_count: usize,
        observable_count: usize,
        edges: &[BuiltPyMatchingEdge],
    ) -> PyResult<Self> {
        let observable_storage = edges
            .iter()
            .map(|edge| edge.fault_observables.clone())
            .collect::<Vec<_>>();
        let ffi_edges = edges
            .iter()
            .zip(&observable_storage)
            .map(|(edge, observables)| PymatchingShimEdge {
                left: edge.left,
                right: edge.right.unwrap_or(0),
                is_boundary: u8::from(edge.right.is_none()),
                weight: edge.scaled_weight,
                observables: observables.as_ptr(),
                observable_count: observables.len(),
            })
            .collect::<Vec<_>>();
        let mut error = ErrorBuffer::new();
        let ptr = unsafe {
            faultscope_pymatching_decoder_new(
                detector_count,
                observable_count,
                ffi_edges.as_ptr(),
                ffi_edges.len(),
                error.ptr(),
                error.capacity(),
            )
        };
        if ptr.is_null() {
            return Err(PyValueError::new_err(format!(
                "pymatching native build failed: {}",
                error.message()
            )));
        }
        Ok(Self { ptr })
    }

    fn decode_batch(
        &mut self,
        detector_masks: &[PymatchingShimMaskView],
        observable_masks: &mut [PymatchingShimMaskMutView],
        shots: usize,
        word_count: usize,
    ) -> Result<(), String> {
        let mut error = ErrorBuffer::new();
        let code = unsafe {
            faultscope_pymatching_decoder_decode_batch(
                self.ptr,
                detector_masks.as_ptr(),
                detector_masks.len(),
                observable_masks.as_mut_ptr(),
                observable_masks.len(),
                shots,
                word_count,
                error.ptr(),
                error.capacity(),
            )
        };
        if code != 0 {
            return Err(error.message());
        }
        Ok(())
    }

    fn decode_packed_batch(
        &mut self,
        detector_shots: &[u8],
        detector_count: usize,
        detector_byte_count: usize,
        observable_predictions: &mut [u8],
        observable_count: usize,
        observable_byte_count: usize,
        shots: usize,
    ) -> Result<(), String> {
        let mut error = ErrorBuffer::new();
        let code = unsafe {
            faultscope_pymatching_decoder_decode_packed_batch(
                self.ptr,
                detector_shots.as_ptr(),
                detector_count,
                detector_byte_count,
                observable_predictions.as_mut_ptr(),
                observable_count,
                observable_byte_count,
                shots,
                error.ptr(),
                error.capacity(),
            )
        };
        if code != 0 {
            return Err(error.message());
        }
        Ok(())
    }
}

impl Drop for PymatchingNativeDecoder {
    fn drop(&mut self) {
        unsafe {
            faultscope_pymatching_decoder_free(self.ptr);
        }
    }
}

struct ErrorBuffer {
    bytes: Vec<c_char>,
}

impl ErrorBuffer {
    fn new() -> Self {
        Self {
            bytes: vec![0; 4096],
        }
    }

    fn ptr(&mut self) -> *mut c_char {
        self.bytes.as_mut_ptr()
    }

    fn capacity(&self) -> usize {
        self.bytes.len()
    }

    fn message(&self) -> String {
        let bytes = self
            .bytes
            .iter()
            .map(|byte| *byte as u8)
            .take_while(|byte| *byte != 0)
            .collect::<Vec<_>>();
        if bytes.is_empty() {
            "<no message>".to_string()
        } else {
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }
}

struct PyMatchingProblemBuilder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    dem_edge_count: usize,
    groups: Vec<EdgeGroup>,
    group_by_endpoint: HashMap<(usize, Option<usize>), usize>,
}

#[derive(Clone)]
struct EdgeGroup {
    endpoint: (usize, Option<usize>),
    dem_edge_indices: Vec<usize>,
    fault_observables: Vec<usize>,
    probabilities: Vec<f64>,
}

#[derive(Clone)]
struct BuiltPyMatchingEdge {
    left: usize,
    right: Option<usize>,
    dem_edge_indices: Vec<usize>,
    fault_observables: Vec<usize>,
    probability: f64,
    weight: f64,
    scaled_weight: i32,
}

#[derive(Clone)]
struct BuildSummary {
    dem_edge_count: usize,
    solver_edge_count: usize,
    merged_parallel_edge_count: usize,
    edges: Vec<BuiltPyMatchingEdge>,
}

struct BuiltPyMatchingProblem {
    edges: Vec<BuiltPyMatchingEdge>,
    dem_edge_count: usize,
    merged_parallel_edge_count: usize,
}

impl PyMatchingProblemBuilder {
    fn from_dem_object(dem: &Bound<'_, PyAny>) -> PyResult<Self> {
        let detector_ids = extract_ids(dem.getattr("detectors")?, "detector")?;
        let observable_ids = extract_ids(dem.getattr("observables")?, "observable")?;
        let detector_index = detector_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect::<HashMap<_, _>>();
        let observable_index = observable_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect::<HashMap<_, _>>();
        let mut builder = Self {
            detector_ids,
            observable_ids,
            dem_edge_count: 0,
            groups: Vec::new(),
            group_by_endpoint: HashMap::new(),
        };
        let edges = dem.getattr("edges")?;
        for (fallback_index, edge) in PyIterator::from_object(&edges)?.enumerate() {
            let edge = edge?;
            let probability = edge.getattr("probability")?.extract::<f64>()?;
            let detector_ids = edge.getattr("detectors")?.extract::<Vec<i64>>()?;
            let observable_ids = edge.getattr("observables")?.extract::<Vec<i64>>()?;
            let dem_edge_index = edge
                .getattr("original_edge_index")
                .and_then(|value| value.extract::<usize>())
                .unwrap_or(fallback_index);
            let detectors = detector_ids
                .iter()
                .map(|detector_id| {
                    detector_index.get(detector_id).copied().ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "pymatching edge {dem_edge_index} references unknown detector id {detector_id}"
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let fault_observables = observable_ids
                .iter()
                .map(|observable_id| {
                    observable_index.get(observable_id).copied().ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "pymatching edge {dem_edge_index} references unknown observable id {observable_id}"
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            builder.push_edge(dem_edge_index, detectors, fault_observables, probability)?;
        }
        Ok(builder)
    }

    fn from_graphlike_problem(problem: &Bound<'_, PyAny>) -> PyResult<Self> {
        let detector_ids = problem.getattr("detector_ids")?.extract::<Vec<i64>>()?;
        let observable_ids = problem.getattr("observable_ids")?.extract::<Vec<i64>>()?;
        let mut builder = Self {
            detector_ids,
            observable_ids,
            dem_edge_count: 0,
            groups: Vec::new(),
            group_by_endpoint: HashMap::new(),
        };
        let edges = problem.getattr("edges")?;
        for edge in PyIterator::from_object(&edges)? {
            let edge = edge?;
            builder.push_edge(
                edge.getattr("dem_edge_index")?.extract::<usize>()?,
                edge.getattr("detectors")?.extract::<Vec<usize>>()?,
                edge.getattr("fault_observables")?.extract::<Vec<usize>>()?,
                edge.getattr("probability")?.extract::<f64>()?,
            )?;
        }
        Ok(builder)
    }

    fn push_edge(
        &mut self,
        dem_edge_index: usize,
        detectors: Vec<usize>,
        fault_observables: Vec<usize>,
        probability: f64,
    ) -> PyResult<()> {
        self.dem_edge_count += 1;
        validate_probability(probability, dem_edge_index)?;
        let fault_observables = canonical_fault_observables(fault_observables);
        for &detector_index in &detectors {
            if detector_index >= self.detector_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "pymatching edge {dem_edge_index} references detector index {detector_index} but only {} detectors exist",
                    self.detector_ids.len()
                )));
            }
        }
        for &observable_index in &fault_observables {
            if observable_index >= self.observable_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "pymatching edge {dem_edge_index} references observable index {observable_index} but only {} observables exist",
                    self.observable_ids.len()
                )));
            }
        }

        let endpoint = match detectors.as_slice() {
            [] => {
                if fault_observables.is_empty() {
                    return Ok(());
                }
                return Err(PyValueError::new_err(format!(
                    "pymatching edge {dem_edge_index} flips observables but has no detectors; pure logical edges are unsupported"
                )));
            }
            [detector] => (*detector, None),
            [left, right] => {
                if left == right {
                    return Err(PyValueError::new_err(format!(
                        "pymatching edge {dem_edge_index} has identical endpoints {left}"
                    )));
                }
                normalized_endpoint(*left, *right)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "pymatching edge {dem_edge_index} has {} detectors; expected one boundary detector or two graph detectors",
                    detectors.len()
                )));
            }
        };

        if let Some(group_index) = self.group_by_endpoint.get(&endpoint).copied() {
            let group = &mut self.groups[group_index];
            group.dem_edge_indices.push(dem_edge_index);
            group.probabilities.push(probability);
        } else {
            let group_index = self.groups.len();
            self.group_by_endpoint.insert(endpoint, group_index);
            self.groups.push(EdgeGroup {
                endpoint,
                dem_edge_indices: vec![dem_edge_index],
                fault_observables,
                probabilities: vec![probability],
            });
        }
        Ok(())
    }

    fn build(self) -> PyResult<BuiltPyMatchingProblem> {
        let mut edges = Vec::with_capacity(self.groups.len());
        for group in self.groups {
            let probability = odd_parity_probability(&group.probabilities);
            let weight = log_likelihood_ratio(probability);
            edges.push(BuiltPyMatchingEdge {
                left: group.endpoint.0,
                right: group.endpoint.1,
                dem_edge_indices: group.dem_edge_indices,
                fault_observables: group.fault_observables,
                probability,
                weight,
                scaled_weight: 0,
            });
        }
        scale_weights(&mut edges)?;
        let merged_parallel_edge_count = edges
            .iter()
            .map(|edge| edge.dem_edge_indices.len().saturating_sub(1))
            .sum();
        Ok(BuiltPyMatchingProblem {
            edges,
            dem_edge_count: self.dem_edge_count,
            merged_parallel_edge_count,
        })
    }
}

fn extract_ids(sequence: Bound<'_, PyAny>, kind: &str) -> PyResult<Vec<i64>> {
    PyIterator::from_object(&sequence)?
        .enumerate()
        .map(|(index, item)| {
            let item = item?;
            item.getattr("id")?.extract::<i64>().map_err(|err| {
                PyValueError::new_err(format!(
                    "pymatching {kind} at index {index} does not expose integer id: {err}"
                ))
            })
        })
        .collect()
}

fn scale_weights(edges: &mut [BuiltPyMatchingEdge]) -> PyResult<()> {
    let mut max_abs_weight = 0.0;
    let mut all_integral = true;
    for edge in edges.iter() {
        let abs = edge.weight.abs();
        if abs > max_abs_weight {
            max_abs_weight = abs;
        }
        if edge.weight.round() != edge.weight {
            all_integral = false;
        }
    }
    if max_abs_weight > MAX_USER_EDGE_WEIGHT {
        return Err(PyValueError::new_err(format!(
            "pymatching maximum absolute edge weight {max_abs_weight} exceeds {MAX_USER_EDGE_WEIGHT}"
        )));
    }
    let normalizing_constant = if all_integral || max_abs_weight == 0.0 {
        1.0
    } else {
        (NUM_DISTINCT_WEIGHTS - 1.0) / max_abs_weight
    };
    for edge in edges {
        let scaled = (edge.weight * normalizing_constant).round() * 2.0;
        if !scaled.is_finite() || scaled < i32::MIN as f64 || scaled > i32::MAX as f64 {
            return Err(PyValueError::new_err(format!(
                "pymatching edge {:?} scaled weight {scaled} exceeds i32 range",
                edge.dem_edge_indices
            )));
        }
        edge.scaled_weight = scaled as i32;
    }
    Ok(())
}

fn validate_probability(probability: f64, dem_edge_index: usize) -> PyResult<()> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(PyValueError::new_err(format!(
            "pymatching edge {dem_edge_index} has invalid probability {probability}; expected a finite value in [0, 1]"
        )));
    }
    Ok(())
}

fn odd_parity_probability(probabilities: &[f64]) -> f64 {
    let even_minus_odd = probabilities
        .iter()
        .fold(1.0, |acc, probability| acc * (1.0 - 2.0 * probability));
    ((1.0 - even_minus_odd) * 0.5).clamp(0.0, 1.0)
}

fn canonical_fault_observables(mut fault_observables: Vec<usize>) -> Vec<usize> {
    fault_observables.sort_unstable();
    let mut canonical = Vec::new();
    let mut index = 0;
    while index < fault_observables.len() {
        let observable = fault_observables[index];
        let mut count = 1;
        index += 1;
        while index < fault_observables.len() && fault_observables[index] == observable {
            count += 1;
            index += 1;
        }
        if count % 2 == 1 {
            canonical.push(observable);
        }
    }
    canonical
}

fn normalized_endpoint(left: usize, right: usize) -> (usize, Option<usize>) {
    if left < right {
        (left, Some(right))
    } else {
        (right, Some(left))
    }
}

unsafe extern "C" fn drop_state(state: *mut c_void) {
    if !state.is_null() {
        drop(Box::from_raw(state.cast::<DecoderState>()));
    }
}

unsafe extern "C" fn decoder_name(
    state: *const c_void,
    out: *mut FaultScopeNativeDecoderStringViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
    if out.is_null() {
        return static_error("pymatching name callback received null output pointer");
    }
    if state.is_null() {
        return static_error("pymatching name callback received null state pointer");
    }
    *out = FaultScopeNativeDecoderStringViewV1 {
        ptr: BACKEND_NAME.as_ptr().cast::<c_char>(),
        len: BACKEND_NAME.len(),
    };
    FaultScopeNativeDecoderStatusV1::ok()
}

unsafe extern "C" fn decoder_detector_ids(
    state: *const c_void,
    out: *mut FaultScopeNativeDecoderI64SliceV1,
) -> FaultScopeNativeDecoderStatusV1 {
    ids_callback(state, out, |state| &state.detector_ids)
}

unsafe extern "C" fn decoder_observable_ids(
    state: *const c_void,
    out: *mut FaultScopeNativeDecoderI64SliceV1,
) -> FaultScopeNativeDecoderStatusV1 {
    ids_callback(state, out, |state| &state.observable_ids)
}

unsafe fn ids_callback(
    state: *const c_void,
    out: *mut FaultScopeNativeDecoderI64SliceV1,
    ids: impl FnOnce(&DecoderState) -> &[i64],
) -> FaultScopeNativeDecoderStatusV1 {
    if out.is_null() {
        return static_error("pymatching ids callback received null output pointer");
    }
    if state.is_null() {
        return static_error("pymatching ids callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    let ids = ids(state);
    *out = FaultScopeNativeDecoderI64SliceV1 {
        ptr: ids.as_ptr(),
        len: ids.len(),
    };
    FaultScopeNativeDecoderStatusV1::ok()
}

unsafe extern "C" fn decoder_decode_batch(
    state: *mut c_void,
    input: *const FaultScopeNativeDetectorMaskBatchViewV1,
    output: *mut FaultScopeNativeCorrectionMaskBatchMutViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
    if state.is_null() {
        return static_error("pymatching decode callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        decoder_decode_batch_impl(state, input, output)
    }));
    match result {
        Ok(status) => status,
        Err(_) => state_error(state, "pymatching decoder panicked during decode"),
    }
}

unsafe extern "C" fn decoder_decode_packed_batch(
    state: *mut c_void,
    input: *const FaultScopeNativePackedDetectorShotBatchViewV1,
    output: *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
    if state.is_null() {
        return static_error("pymatching packed decode callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        decoder_decode_packed_batch_impl(state, input, output)
    }));
    match result {
        Ok(status) => status,
        Err(_) => state_error(state, "pymatching decoder panicked during packed decode"),
    }
}

unsafe fn decoder_decode_batch_impl(
    state: &DecoderState,
    input: *const FaultScopeNativeDetectorMaskBatchViewV1,
    output: *mut FaultScopeNativeCorrectionMaskBatchMutViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
    if input.is_null() {
        return state_error(
            state,
            "pymatching decode callback received null input pointer",
        );
    }
    if output.is_null() {
        return state_error(
            state,
            "pymatching decode callback received null output pointer",
        );
    }
    let input = &*input;
    let output = &mut *output;
    if input.detector_count != state.detector_ids.len() {
        return state_error(
            state,
            format!(
                "pymatching expected {} detector masks but received {}",
                state.detector_ids.len(),
                input.detector_count
            ),
        );
    }
    if output.observable_count != state.observable_ids.len() {
        return state_error(
            state,
            format!(
                "pymatching expected {} correction masks but received {}",
                state.observable_ids.len(),
                output.observable_count
            ),
        );
    }
    if input.word_count != output.word_count {
        return state_error(
            state,
            "pymatching input and output word counts do not match",
        );
    }
    if input.detector_count > 0 && input.masks.is_null() {
        return state_error(state, "pymatching input masks pointer is null");
    }
    if output.observable_count > 0 && output.masks.is_null() {
        return state_error(state, "pymatching output masks pointer is null");
    }
    if input.detector_count > 0 && input.detector_ids.is_null() {
        return state_error(state, "pymatching input detector ids pointer is null");
    }
    if output.observable_count > 0 && output.observable_ids.is_null() {
        return state_error(state, "pymatching output observable ids pointer is null");
    }

    let input_detector_ids = if input.detector_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(input.detector_ids, input.detector_count)
    };
    if input_detector_ids != state.detector_ids.as_slice() {
        return state_error(
            state,
            format!(
                "pymatching detector id order mismatch: expected {:?}, received {:?}",
                state.detector_ids, input_detector_ids
            ),
        );
    }
    let output_observable_ids = if output.observable_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(output.observable_ids, output.observable_count)
    };
    if output_observable_ids != state.observable_ids.as_slice() {
        return state_error(
            state,
            format!(
                "pymatching observable id order mismatch: expected {:?}, received {:?}",
                state.observable_ids, output_observable_ids
            ),
        );
    }
    let input_masks = if input.detector_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(input.masks, input.detector_count)
    };
    let output_masks = if output.observable_count == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(output.masks, output.observable_count)
    };
    for input_mask in input_masks {
        if input_mask.word_count != input.word_count {
            return state_error(
                state,
                "pymatching input mask word count does not match batch word count",
            );
        }
        if input.word_count > 0 && input_mask.words.is_null() {
            return state_error(state, "pymatching input mask words pointer is null");
        }
    }
    for output_mask in output_masks.iter() {
        if output_mask.word_count != output.word_count {
            return state_error(
                state,
                "pymatching output mask word count does not match batch word count",
            );
        }
        if output.word_count > 0 && output_mask.words.is_null() {
            return state_error(state, "pymatching output mask words pointer is null");
        }
    }

    let mut native = match state.native.lock() {
        Ok(native) => native,
        Err(_) => return state_error(state, "pymatching native decoder mutex poisoned"),
    };
    let detector_mask_views = input_masks
        .iter()
        .map(|mask| PymatchingShimMaskView {
            words: mask.words,
            word_count: mask.word_count,
        })
        .collect::<Vec<_>>();
    let mut observable_mask_views = output_masks
        .iter_mut()
        .map(|mask| PymatchingShimMaskMutView {
            words: mask.words,
            word_count: mask.word_count,
        })
        .collect::<Vec<_>>();
    if let Err(message) = native.decode_batch(
        &detector_mask_views,
        &mut observable_mask_views,
        input.shots,
        input.word_count,
    ) {
        return state_error(state, format!("pymatching solver error: {message}"));
    }
    FaultScopeNativeDecoderStatusV1::ok()
}

unsafe fn decoder_decode_packed_batch_impl(
    state: &DecoderState,
    input: *const FaultScopeNativePackedDetectorShotBatchViewV1,
    output: *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
    if input.is_null() {
        return state_error(
            state,
            "pymatching packed decode callback received null input pointer",
        );
    }
    if output.is_null() {
        return state_error(
            state,
            "pymatching packed decode callback received null output pointer",
        );
    }
    let input = &*input;
    let output = &mut *output;
    if input.detector_count != state.detector_ids.len() {
        return state_error(
            state,
            format!(
                "pymatching expected {} packed detector columns but received {}",
                state.detector_ids.len(),
                input.detector_count
            ),
        );
    }
    if output.observable_count != state.observable_ids.len() {
        return state_error(
            state,
            format!(
                "pymatching expected {} packed observable columns but received {}",
                state.observable_ids.len(),
                output.observable_count
            ),
        );
    }
    let expected_detector_bytes = state.detector_ids.len().div_ceil(8);
    if input.detector_byte_count != expected_detector_bytes {
        return state_error(
            state,
            format!(
                "pymatching packed detector byte count is {}; expected {expected_detector_bytes}",
                input.detector_byte_count
            ),
        );
    }
    let expected_observable_bytes = state.observable_ids.len().div_ceil(8);
    if output.observable_byte_count != expected_observable_bytes {
        return state_error(
            state,
            format!(
                "pymatching packed observable byte count is {}; expected {expected_observable_bytes}",
                output.observable_byte_count
            ),
        );
    }
    if input.detector_count > 0 && input.detector_ids.is_null() {
        return state_error(
            state,
            "pymatching packed input detector ids pointer is null",
        );
    }
    if output.observable_count > 0 && output.observable_ids.is_null() {
        return state_error(
            state,
            "pymatching packed output observable ids pointer is null",
        );
    }
    if input.shots > 0 && input.detector_byte_count > 0 && input.data.is_null() {
        return state_error(state, "pymatching packed input data pointer is null");
    }
    if output.shots > 0 && output.observable_byte_count > 0 && output.data.is_null() {
        return state_error(state, "pymatching packed output data pointer is null");
    }
    if input.shots != output.shots {
        return state_error(
            state,
            format!(
                "pymatching packed input has {} shots but output has {}",
                input.shots, output.shots
            ),
        );
    }

    let input_detector_ids = if input.detector_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(input.detector_ids, input.detector_count)
    };
    if input_detector_ids != state.detector_ids.as_slice() {
        return state_error(
            state,
            format!(
                "pymatching packed detector id order mismatch: expected {:?}, received {:?}",
                state.detector_ids, input_detector_ids
            ),
        );
    }
    let output_observable_ids = if output.observable_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(output.observable_ids, output.observable_count)
    };
    if output_observable_ids != state.observable_ids.as_slice() {
        return state_error(
            state,
            format!(
                "pymatching packed observable id order mismatch: expected {:?}, received {:?}",
                state.observable_ids, output_observable_ids
            ),
        );
    }

    let input_len = match input.shots.checked_mul(input.detector_byte_count) {
        Some(len) => len,
        None => return state_error(state, "pymatching packed input byte length overflow"),
    };
    let output_len = match output.shots.checked_mul(output.observable_byte_count) {
        Some(len) => len,
        None => return state_error(state, "pymatching packed output byte length overflow"),
    };
    let detector_shots = if input_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(input.data, input_len)
    };
    let observable_predictions = if output_len == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(output.data, output_len)
    };

    let mut native = match state.native.lock() {
        Ok(native) => native,
        Err(_) => return state_error(state, "pymatching native decoder mutex poisoned"),
    };
    if let Err(message) = native.decode_packed_batch(
        detector_shots,
        input.detector_count,
        input.detector_byte_count,
        observable_predictions,
        output.observable_count,
        output.observable_byte_count,
        input.shots,
    ) {
        return state_error(state, format!("pymatching packed solver error: {message}"));
    }
    FaultScopeNativeDecoderStatusV1::ok()
}

fn build_summary_to_py(py: Python<'_>, summary: &BuildSummary) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("dem_edge_count", summary.dem_edge_count)?;
    out.set_item("solver_edge_count", summary.solver_edge_count)?;
    out.set_item(
        "merged_parallel_edge_count",
        summary.merged_parallel_edge_count,
    )?;

    let mut edge_objects = Vec::with_capacity(summary.edges.len());
    for (solver_edge_index, edge) in summary.edges.iter().enumerate() {
        let edge_out = PyDict::new(py);
        edge_out.set_item("solver_edge_index", solver_edge_index)?;
        let endpoint = match edge.right {
            Some(right) => PyTuple::new(py, [edge.left as i64, right as i64])?,
            None => PyTuple::new(py, [edge.left as i64, -1])?,
        };
        edge_out.set_item("endpoint", endpoint)?;
        edge_out.set_item(
            "dem_edge_indices",
            PyTuple::new(py, edge.dem_edge_indices.iter().copied())?,
        )?;
        edge_out.set_item(
            "fault_observables",
            PyTuple::new(py, edge.fault_observables.iter().copied())?,
        )?;
        edge_out.set_item("probability", edge.probability)?;
        edge_out.set_item("weight", edge.weight)?;
        edge_out.set_item("scaled_weight", edge.scaled_weight)?;
        edge_objects.push(edge_out.into_any().unbind());
    }
    out.set_item(
        "edges",
        PyTuple::new(py, edge_objects.iter().map(|item| item.clone_ref(py)))?,
    )?;
    Ok(out.into())
}

unsafe fn create_decoder_capsule(
    py: Python<'_>,
    descriptor: *mut FaultScopeNativeDecoderV1,
) -> PyResult<Py<PyAny>> {
    let ptr = pyo3::ffi::PyCapsule_New(
        descriptor.cast::<c_void>(),
        CAPSULE_NAME.as_ptr().cast::<c_char>(),
        Some(capsule_destructor),
    );
    if ptr.is_null() {
        drop_descriptor(descriptor);
        return Err(PyErr::fetch(py));
    }
    Ok(Py::from_owned_ptr(py, ptr))
}

unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr().cast::<c_char>());
    if !pointer.is_null() {
        drop_descriptor(pointer.cast::<FaultScopeNativeDecoderV1>());
    }
}

unsafe fn drop_descriptor(descriptor: *mut FaultScopeNativeDecoderV1) {
    if descriptor.is_null() {
        return;
    }
    let descriptor = Box::from_raw(descriptor);
    if let Some(drop_state) = descriptor.drop_state {
        drop_state(descriptor.state);
    }
}

fn static_error(message: &'static str) -> FaultScopeNativeDecoderStatusV1 {
    FaultScopeNativeDecoderStatusV1 {
        code: NATIVE_DECODER_PLUGIN_STATUS_ERROR,
        message: FaultScopeNativeDecoderStringViewV1 {
            ptr: message.as_ptr().cast::<c_char>(),
            len: message.len(),
        },
    }
}

fn state_error(state: &DecoderState, message: impl Into<String>) -> FaultScopeNativeDecoderStatusV1 {
    let sanitized = message.into().replace('\0', "\\0");
    let mut last_error = state
        .last_error
        .lock()
        .expect("pymatching error mutex poisoned");
    *last_error = CString::new(sanitized).expect("NUL was sanitized");
    FaultScopeNativeDecoderStatusV1 {
        code: NATIVE_DECODER_PLUGIN_STATUS_ERROR,
        message: FaultScopeNativeDecoderStringViewV1 {
            ptr: last_error.as_ptr(),
            len: last_error.as_bytes().len(),
        },
    }
}

fn string_view_to_string(view: FaultScopeNativeDecoderStringViewV1) -> String {
    if view.len == 0 {
        return String::new();
    }
    if view.ptr.is_null() {
        return "<null error message>".to_string();
    }
    let bytes = unsafe { slice::from_raw_parts(view.ptr.cast::<u8>(), view.len) };
    String::from_utf8_lossy(bytes).into_owned()
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativePyMatchingNativeDecoder>()?;
    Ok(())
}

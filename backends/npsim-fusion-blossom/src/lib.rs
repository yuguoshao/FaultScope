use fusion_blossom::util::{SolverInitializer, SyndromePattern, VertexIndex, VertexNum, Weight};
use fusion_blossom::{detailed_matching, fusion_mwpm};
use npsim_core::{
    log_likelihood_ratio, NpsimNativeCorrectionMaskBatchMutViewV1, NpsimNativeDecoderI64SliceV1,
    NpsimNativeDecoderStatusV1, NpsimNativeDecoderStringViewV1, NpsimNativeDecoderV1,
    NpsimNativeDetectorMaskBatchViewV1, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE, NATIVE_DECODER_PLUGIN_STATUS_ERROR,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyTuple};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::mem;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::Mutex;

const BACKEND_NAME: &str = "fusion-blossom";
const CAPSULE_NAME: &[u8] = b"npsim.native_decoder_plugin.v1\0";
const DEFAULT_WEIGHT_SCALE: f64 = 1_000_000.0;

#[pyclass(name = "NativeFusionBlossomNativeDecoder")]
struct PyNativeFusionBlossomNativeDecoder {
    capsule: Py<PyAny>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    edge_count: usize,
    solver_vertex_count: usize,
    solver_edge_count: usize,
    boundary_vertex_count: usize,
    build_summary: BuildSummary,
}

#[pymethods]
impl PyNativeFusionBlossomNativeDecoder {
    #[staticmethod]
    #[pyo3(signature = (problem, *, weight_scale=DEFAULT_WEIGHT_SCALE))]
    fn from_graphlike_problem(
        py: Python<'_>,
        problem: &Bound<'_, PyAny>,
        weight_scale: f64,
    ) -> PyResult<Self> {
        validate_weight_scale(weight_scale)?;
        let builder = FusionBlossomProblemBuilder::from_graphlike_problem(problem, weight_scale)?;
        let detector_ids = builder.detector_ids.clone();
        let observable_ids = builder.observable_ids.clone();
        let built = builder.build()?;
        let build_summary = built.summary;
        let state = Box::new(DecoderState {
            detector_ids: detector_ids.clone(),
            observable_ids: observable_ids.clone(),
            initializer: built.initializer,
            edge_effects: built.edge_effects.clone(),
            edge_lookup: built.edge_lookup,
            last_error: Mutex::new(CString::new("").expect("empty CString")),
        });
        let descriptor = Box::new(NpsimNativeDecoderV1 {
            abi_version: NATIVE_DECODER_PLUGIN_ABI_VERSION,
            struct_size: mem::size_of::<NpsimNativeDecoderV1>(),
            flags: NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
            state: Box::into_raw(state).cast::<c_void>(),
            drop_state: Some(drop_state),
            name: Some(decoder_name),
            detector_ids: Some(decoder_detector_ids),
            observable_ids: Some(decoder_observable_ids),
            decode_batch: Some(decoder_decode_batch),
            decode_packed_batch: None,
        });
        let capsule = unsafe { create_decoder_capsule(py, Box::into_raw(descriptor))? };

        Ok(Self {
            capsule,
            detector_ids,
            observable_ids,
            edge_count: build_summary.dem_edge_count,
            solver_vertex_count: build_summary.solver_vertex_count,
            solver_edge_count: build_summary.solver_edge_count,
            boundary_vertex_count: build_summary.boundary_vertex_count,
            build_summary,
        })
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
    fn solver_vertex_count(&self) -> usize {
        self.solver_vertex_count
    }

    #[getter]
    fn solver_edge_count(&self) -> usize {
        self.solver_edge_count
    }

    #[getter]
    fn boundary_vertex_count(&self) -> usize {
        self.boundary_vertex_count
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
            .map(|words| npsim_core::NpsimNativeDecoderMaskViewV1 {
                words: words.as_ptr(),
                word_count: words.len(),
            })
            .collect::<Vec<_>>();
        let mut output_views = output_buffers
            .iter_mut()
            .map(|words| npsim_core::NpsimNativeDecoderMaskMutViewV1 {
                words: words.as_mut_ptr(),
                word_count: words.len(),
            })
            .collect::<Vec<_>>();
        let input = NpsimNativeDetectorMaskBatchViewV1 {
            detector_ids: self.detector_ids.as_ptr(),
            detector_count: self.detector_ids.len(),
            masks: input_views.as_ptr(),
            shots,
            word_count,
        };
        let mut output = NpsimNativeCorrectionMaskBatchMutViewV1 {
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
            .cast::<NpsimNativeDecoderV1>())
        };
        let status = unsafe {
            descriptor
                .decode_batch
                .expect("fusion-blossom descriptor has decode callback")(
                descriptor.state,
                &input,
                &mut output,
            )
        };
        if status.code != npsim_core::NATIVE_DECODER_PLUGIN_STATUS_OK {
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

    fn __npsim_native_decoder_capsule__(&self, py: Python<'_>) -> Py<PyAny> {
        self.capsule.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeFusionBlossomNativeDecoder(name={:?}, detector_ids={:?}, observable_ids={:?}, solver_vertex_count={}, solver_edge_count={})",
            BACKEND_NAME,
            self.detector_ids,
            self.observable_ids,
            self.solver_vertex_count,
            self.solver_edge_count
        )
    }
}

#[pyclass(name = "InvalidNativeDecoderCapsule")]
struct PyInvalidNativeDecoderCapsule {
    capsule: Py<PyAny>,
}

#[pymethods]
impl PyInvalidNativeDecoderCapsule {
    #[new]
    fn new(py: Python<'_>, kind: &str) -> PyResult<Self> {
        let state = Box::new(DecoderState {
            detector_ids: vec![0],
            observable_ids: vec![0],
            initializer: SolverInitializer::new(2, Vec::new(), Vec::new()),
            edge_effects: Vec::new(),
            edge_lookup: HashMap::new(),
            last_error: Mutex::new(CString::new("").expect("empty CString")),
        });
        let mut descriptor = NpsimNativeDecoderV1 {
            abi_version: NATIVE_DECODER_PLUGIN_ABI_VERSION,
            struct_size: mem::size_of::<NpsimNativeDecoderV1>(),
            flags: NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
            state: Box::into_raw(state).cast::<c_void>(),
            drop_state: Some(drop_state),
            name: Some(decoder_name),
            detector_ids: Some(decoder_detector_ids),
            observable_ids: Some(decoder_observable_ids),
            decode_batch: Some(decoder_decode_batch),
            decode_packed_batch: None,
        };
        match kind {
            "abi-mismatch" => descriptor.abi_version = NATIVE_DECODER_PLUGIN_ABI_VERSION + 1,
            "missing-callback" => descriptor.decode_batch = None,
            "not-thread-safe" => descriptor.flags = 0,
            "decode-error" => descriptor.decode_batch = Some(forced_error_decode_batch),
            _ => {
                unsafe {
                    drop_descriptor(Box::into_raw(Box::new(descriptor)));
                }
                return Err(PyValueError::new_err(format!(
                    "unknown invalid native decoder kind {kind:?}"
                )));
            }
        }
        let capsule = unsafe { create_decoder_capsule(py, Box::into_raw(Box::new(descriptor)))? };
        Ok(Self { capsule })
    }

    fn __npsim_native_decoder_capsule__(&self, py: Python<'_>) -> Py<PyAny> {
        self.capsule.clone_ref(py)
    }
}

struct DecoderState {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    initializer: SolverInitializer,
    edge_effects: Vec<SolverEdgeEffect>,
    edge_lookup: HashMap<(usize, usize), usize>,
    last_error: Mutex<CString>,
}

struct BuiltFusionBlossomProblem {
    initializer: SolverInitializer,
    edge_lookup: HashMap<(usize, usize), usize>,
    edge_effects: Vec<SolverEdgeEffect>,
    summary: BuildSummary,
}

struct FusionBlossomProblemBuilder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    weight_scale: f64,
    dem_edge_count: usize,
    boundary_edges: Vec<BoundaryEdge>,
    graph_groups: Vec<GraphEdgeGroup>,
    graph_group_by_endpoint: HashMap<(usize, usize), usize>,
}

#[derive(Debug, Clone)]
struct BoundaryEdge {
    detector: usize,
    dem_edge_index: usize,
    fault_observables: Vec<usize>,
    probability: f64,
    weight: f64,
}

#[derive(Debug, Clone)]
struct GraphEdgeGroup {
    endpoint: (usize, usize),
    dem_edge_indices: Vec<usize>,
    fault_observables: Vec<usize>,
    probabilities: Vec<f64>,
}

#[derive(Debug, Clone)]
struct SolverEdgeEffect {
    endpoint: (usize, usize),
    dem_edge_indices: Vec<usize>,
    fault_observables: Vec<usize>,
    probability: f64,
    weight: f64,
}

#[derive(Debug, Clone)]
struct BuildSummary {
    dem_edge_count: usize,
    solver_vertex_count: usize,
    solver_edge_count: usize,
    boundary_vertex_count: usize,
    merged_parallel_edge_count: usize,
    edges: Vec<SolverEdgeEffect>,
}

impl FusionBlossomProblemBuilder {
    fn from_graphlike_problem(problem: &Bound<'_, PyAny>, weight_scale: f64) -> PyResult<Self> {
        let detector_ids = problem.getattr("detector_ids")?.extract::<Vec<i64>>()?;
        let observable_ids = problem.getattr("observable_ids")?.extract::<Vec<i64>>()?;
        let mut builder = Self {
            detector_ids,
            observable_ids,
            weight_scale,
            dem_edge_count: 0,
            boundary_edges: Vec::new(),
            graph_groups: Vec::new(),
            graph_group_by_endpoint: HashMap::new(),
        };

        let edges = problem.getattr("edges")?;
        for edge in PyIterator::from_object(&edges)? {
            let edge = edge?;
            builder.push_edge(
                edge.getattr("dem_edge_index")?.extract::<usize>()?,
                edge.getattr("detectors")?.extract::<Vec<usize>>()?,
                edge.getattr("fault_observables")?.extract::<Vec<usize>>()?,
                edge.getattr("probability")?.extract::<f64>()?,
                edge.getattr("weight")?.extract::<f64>()?,
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
        weight: f64,
    ) -> PyResult<()> {
        self.dem_edge_count += 1;
        validate_probability(probability, dem_edge_index)?;
        let fault_observables = canonical_fault_observables(fault_observables);
        for &detector_index in &detectors {
            if detector_index >= self.detector_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "fusion-blossom edge {dem_edge_index} references detector index {detector_index} but only {} detectors exist",
                    self.detector_ids.len()
                )));
            }
        }
        for &observable_index in &fault_observables {
            if observable_index >= self.observable_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "fusion-blossom edge {dem_edge_index} references observable index {observable_index} but only {} observables exist",
                    self.observable_ids.len()
                )));
            }
        }

        match detectors.as_slice() {
            [detector] => {
                self.boundary_edges.push(BoundaryEdge {
                    detector: *detector,
                    dem_edge_index,
                    fault_observables,
                    probability,
                    weight,
                });
            }
            [left, right] => {
                if left == right {
                    return Err(PyValueError::new_err(format!(
                        "fusion-blossom edge {dem_edge_index} has identical endpoints {left}"
                    )));
                }
                let endpoint = normalized_endpoint(*left, *right);
                if let Some(group_index) = self.graph_group_by_endpoint.get(&endpoint).copied() {
                    let group = &mut self.graph_groups[group_index];
                    if group.fault_observables != fault_observables {
                        return Err(PyValueError::new_err(format!(
                            "fusion-blossom found ambiguous parallel graph endpoint {:?}: DEM edge {} fault_observables {:?} conflicts with DEM edge(s) {:?} fault_observables {:?}",
                            endpoint,
                            dem_edge_index,
                            fault_observables,
                            group.dem_edge_indices,
                            group.fault_observables
                        )));
                    }
                    group.dem_edge_indices.push(dem_edge_index);
                    group.probabilities.push(probability);
                } else {
                    let group_index = self.graph_groups.len();
                    self.graph_group_by_endpoint.insert(endpoint, group_index);
                    self.graph_groups.push(GraphEdgeGroup {
                        endpoint,
                        dem_edge_indices: vec![dem_edge_index],
                        fault_observables,
                        probabilities: vec![probability],
                    });
                }
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "fusion-blossom edge {dem_edge_index} has {} detectors; expected one boundary detector or two graph detectors",
                    detectors.len()
                )));
            }
        }
        Ok(())
    }

    fn build(self) -> PyResult<BuiltFusionBlossomProblem> {
        let detector_count = self.detector_ids.len();
        let boundary_vertex_count = self.boundary_edges.len();
        let solver_vertex_count = detector_count + boundary_vertex_count;
        if solver_vertex_count < 2 {
            return Err(PyValueError::new_err(
                "fusion-blossom requires at least two graph vertices",
            ));
        }

        let mut weighted_edges =
            Vec::with_capacity(boundary_vertex_count + self.graph_groups.len());
        let mut virtual_vertices = Vec::with_capacity(boundary_vertex_count);
        let mut edge_lookup = HashMap::new();
        let mut edge_effects = Vec::with_capacity(boundary_vertex_count + self.graph_groups.len());
        let mut merged_parallel_edge_count = 0;

        for (boundary_index, edge) in self.boundary_edges.iter().enumerate() {
            let virtual_vertex = detector_count + boundary_index;
            virtual_vertices.push(virtual_vertex as VertexIndex);
            let endpoint = normalized_endpoint(edge.detector, virtual_vertex);
            let edge_label = format!("DEM edge {}", edge.dem_edge_index);
            let scaled = scaled_weight(
                edge.weight,
                self.weight_scale,
                &edge_label,
                solver_vertex_count,
            )?;
            edge_lookup.insert(endpoint, weighted_edges.len());
            weighted_edges.push((endpoint.0 as VertexIndex, endpoint.1 as VertexIndex, scaled));
            edge_effects.push(SolverEdgeEffect {
                endpoint,
                dem_edge_indices: vec![edge.dem_edge_index],
                fault_observables: edge.fault_observables.clone(),
                probability: edge.probability,
                weight: edge.weight,
            });
        }

        for group in &self.graph_groups {
            if group.dem_edge_indices.len() > 1 {
                merged_parallel_edge_count += group.dem_edge_indices.len() - 1;
            }
            let probability = odd_parity_probability(&group.probabilities);
            let weight = log_likelihood_ratio(probability);
            let edge_label = format!("DEM edges {:?}", group.dem_edge_indices);
            let scaled =
                scaled_weight(weight, self.weight_scale, &edge_label, solver_vertex_count)?;
            edge_lookup.insert(group.endpoint, weighted_edges.len());
            weighted_edges.push((
                group.endpoint.0 as VertexIndex,
                group.endpoint.1 as VertexIndex,
                scaled,
            ));
            edge_effects.push(SolverEdgeEffect {
                endpoint: group.endpoint,
                dem_edge_indices: group.dem_edge_indices.clone(),
                fault_observables: group.fault_observables.clone(),
                probability,
                weight,
            });
        }

        let solver_edge_count = weighted_edges.len();
        let initializer = SolverInitializer::new(
            solver_vertex_count as VertexNum,
            weighted_edges,
            virtual_vertices,
        );
        let summary = BuildSummary {
            dem_edge_count: self.dem_edge_count,
            solver_vertex_count,
            solver_edge_count,
            boundary_vertex_count,
            merged_parallel_edge_count,
            edges: edge_effects.clone(),
        };
        Ok(BuiltFusionBlossomProblem {
            initializer,
            edge_lookup,
            edge_effects,
            summary,
        })
    }
}

fn validate_weight_scale(weight_scale: f64) -> PyResult<()> {
    if !weight_scale.is_finite() || weight_scale <= 0.0 {
        return Err(PyValueError::new_err(
            "fusion-blossom weight_scale must be a positive finite number",
        ));
    }
    Ok(())
}

fn validate_probability(probability: f64, dem_edge_index: usize) -> PyResult<()> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(PyValueError::new_err(format!(
            "fusion-blossom edge {dem_edge_index} has invalid probability {probability}; expected a finite value in [0, 1]"
        )));
    }
    Ok(())
}

fn scaled_weight(
    weight: f64,
    weight_scale: f64,
    edge_label: &str,
    vertex_count: usize,
) -> PyResult<Weight> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(PyValueError::new_err(format!(
            "fusion-blossom {edge_label} has non-finite or negative weight {weight}"
        )));
    }
    let scaled = (weight * weight_scale).round();
    let max_safe = (Weight::MAX as f64) / (vertex_count.max(1) as f64);
    let even_scaled = if (scaled as i128) % 2 == 0 {
        scaled
    } else {
        scaled + 1.0
    };
    if !even_scaled.is_finite() || even_scaled < 0.0 || even_scaled > max_safe {
        return Err(PyValueError::new_err(format!(
            "fusion-blossom {edge_label} scaled weight {even_scaled} exceeds safe maximum {max_safe}"
        )));
    }
    Ok(even_scaled as Weight)
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

fn normalized_endpoint(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

unsafe extern "C" fn forced_error_decode_batch(
    state: *mut c_void,
    _input: *const NpsimNativeDetectorMaskBatchViewV1,
    _output: *mut NpsimNativeCorrectionMaskBatchMutViewV1,
) -> NpsimNativeDecoderStatusV1 {
    if state.is_null() {
        return static_error("forced error decoder received null state pointer");
    }
    state_error(
        &*state.cast::<DecoderState>(),
        "forced native decoder decode failure",
    )
}

unsafe extern "C" fn drop_state(state: *mut c_void) {
    if !state.is_null() {
        drop(Box::from_raw(state.cast::<DecoderState>()));
    }
}

unsafe extern "C" fn decoder_name(
    state: *const c_void,
    out: *mut NpsimNativeDecoderStringViewV1,
) -> NpsimNativeDecoderStatusV1 {
    if out.is_null() {
        return static_error("fusion-blossom name callback received null output pointer");
    }
    if state.is_null() {
        return static_error("fusion-blossom name callback received null state pointer");
    }
    *out = NpsimNativeDecoderStringViewV1 {
        ptr: BACKEND_NAME.as_ptr().cast::<c_char>(),
        len: BACKEND_NAME.len(),
    };
    NpsimNativeDecoderStatusV1::ok()
}

unsafe extern "C" fn decoder_detector_ids(
    state: *const c_void,
    out: *mut NpsimNativeDecoderI64SliceV1,
) -> NpsimNativeDecoderStatusV1 {
    ids_callback(state, out, |state| &state.detector_ids)
}

unsafe extern "C" fn decoder_observable_ids(
    state: *const c_void,
    out: *mut NpsimNativeDecoderI64SliceV1,
) -> NpsimNativeDecoderStatusV1 {
    ids_callback(state, out, |state| &state.observable_ids)
}

unsafe fn ids_callback(
    state: *const c_void,
    out: *mut NpsimNativeDecoderI64SliceV1,
    ids: impl FnOnce(&DecoderState) -> &[i64],
) -> NpsimNativeDecoderStatusV1 {
    if out.is_null() {
        return static_error("fusion-blossom ids callback received null output pointer");
    }
    if state.is_null() {
        return static_error("fusion-blossom ids callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    let ids = ids(state);
    *out = NpsimNativeDecoderI64SliceV1 {
        ptr: ids.as_ptr(),
        len: ids.len(),
    };
    NpsimNativeDecoderStatusV1::ok()
}

unsafe extern "C" fn decoder_decode_batch(
    state: *mut c_void,
    input: *const NpsimNativeDetectorMaskBatchViewV1,
    output: *mut NpsimNativeCorrectionMaskBatchMutViewV1,
) -> NpsimNativeDecoderStatusV1 {
    if state.is_null() {
        return static_error("fusion-blossom decode callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    if input.is_null() {
        return state_error(
            state,
            "fusion-blossom decode callback received null input pointer",
        );
    }
    if output.is_null() {
        return state_error(
            state,
            "fusion-blossom decode callback received null output pointer",
        );
    }
    let input = &*input;
    let output = &mut *output;
    if input.detector_count != state.detector_ids.len() {
        return state_error(
            state,
            format!(
                "fusion-blossom expected {} detector masks but received {}",
                state.detector_ids.len(),
                input.detector_count
            ),
        );
    }
    if output.observable_count != state.observable_ids.len() {
        return state_error(
            state,
            format!(
                "fusion-blossom expected {} correction masks but received {}",
                state.observable_ids.len(),
                output.observable_count
            ),
        );
    }
    if input.word_count != output.word_count {
        return state_error(
            state,
            "fusion-blossom input and output word counts do not match",
        );
    }
    if input.detector_count > 0 && input.masks.is_null() {
        return state_error(state, "fusion-blossom input masks pointer is null");
    }
    if output.observable_count > 0 && output.masks.is_null() {
        return state_error(state, "fusion-blossom output masks pointer is null");
    }
    if input.detector_count > 0 && input.detector_ids.is_null() {
        return state_error(state, "fusion-blossom input detector ids pointer is null");
    }
    if output.observable_count > 0 && output.observable_ids.is_null() {
        return state_error(
            state,
            "fusion-blossom output observable ids pointer is null",
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
                "fusion-blossom detector id order mismatch: expected {:?}, received {:?}",
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
                "fusion-blossom observable id order mismatch: expected {:?}, received {:?}",
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
                "fusion-blossom input mask word count does not match batch word count",
            );
        }
        if input.word_count > 0 && input_mask.words.is_null() {
            return state_error(state, "fusion-blossom input mask words pointer is null");
        }
    }
    for output_mask in output_masks.iter() {
        if output_mask.word_count != output.word_count {
            return state_error(
                state,
                "fusion-blossom output mask word count does not match batch word count",
            );
        }
        if output.word_count > 0 && output_mask.words.is_null() {
            return state_error(state, "fusion-blossom output mask words pointer is null");
        }
    }
    let mut defect_vertices = Vec::with_capacity(state.detector_ids.len());
    for shot in 0..input.shots {
        defect_vertices.clear();
        for (detector_index, input_mask) in input_masks.iter().enumerate() {
            if read_bit(input_mask.words, shot) {
                defect_vertices.push(detector_index as VertexIndex);
            }
        }
        if defect_vertices.is_empty() {
            continue;
        }
        let syndrome = SyndromePattern::new(defect_vertices.clone(), Vec::new());
        let mwpm_result = match catch_unwind(AssertUnwindSafe(|| {
            fusion_mwpm(&state.initializer, &syndrome)
        })) {
            Ok(result) => result,
            Err(_) => {
                return state_error(state, "fusion-blossom solver panicked during MWPM decode");
            }
        };
        let details = match catch_unwind(AssertUnwindSafe(|| {
            detailed_matching(&state.initializer, &defect_vertices, &mwpm_result)
        })) {
            Ok(details) => details,
            Err(_) => {
                return state_error(
                    state,
                    "fusion-blossom solver panicked while recovering detailed matching",
                );
            }
        };
        for detail in details {
            let mut left = detail.a as usize;
            for (right, _weight) in detail.path {
                let right = right as usize;
                let endpoint = normalized_endpoint(left, right);
                let Some(edge_index) = state.edge_lookup.get(&endpoint).copied() else {
                    return state_error(
                        state,
                        format!(
                            "fusion-blossom detailed path used unknown graph endpoint {:?}",
                            endpoint
                        ),
                    );
                };
                for &observable_index in &state.edge_effects[edge_index].fault_observables {
                    let output_mask = &mut output_masks[observable_index];
                    xor_bit(output_mask.words, shot);
                }
                left = right;
            }
        }
    }
    NpsimNativeDecoderStatusV1::ok()
}

unsafe fn read_bit(words: *const u64, shot: usize) -> bool {
    let word = *words.add(shot / 64);
    ((word >> (shot % 64)) & 1) != 0
}

unsafe fn xor_bit(words: *mut u64, shot: usize) {
    let word = words.add(shot / 64);
    *word ^= 1u64 << (shot % 64);
}

fn build_summary_to_py(py: Python<'_>, summary: &BuildSummary) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("dem_edge_count", summary.dem_edge_count)?;
    out.set_item("solver_vertex_count", summary.solver_vertex_count)?;
    out.set_item("solver_edge_count", summary.solver_edge_count)?;
    out.set_item("boundary_vertex_count", summary.boundary_vertex_count)?;
    out.set_item(
        "merged_parallel_edge_count",
        summary.merged_parallel_edge_count,
    )?;

    let mut edge_objects = Vec::with_capacity(summary.edges.len());
    for (solver_edge_index, edge) in summary.edges.iter().enumerate() {
        let edge_out = PyDict::new(py);
        edge_out.set_item("solver_edge_index", solver_edge_index)?;
        edge_out.set_item(
            "endpoint",
            PyTuple::new(py, [edge.endpoint.0, edge.endpoint.1])?,
        )?;
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
    descriptor: *mut NpsimNativeDecoderV1,
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
        drop_descriptor(pointer.cast::<NpsimNativeDecoderV1>());
    }
}

unsafe fn drop_descriptor(descriptor: *mut NpsimNativeDecoderV1) {
    if descriptor.is_null() {
        return;
    }
    let descriptor = Box::from_raw(descriptor);
    if let Some(drop_state) = descriptor.drop_state {
        drop_state(descriptor.state);
    }
}

fn static_error(message: &'static str) -> NpsimNativeDecoderStatusV1 {
    NpsimNativeDecoderStatusV1 {
        code: NATIVE_DECODER_PLUGIN_STATUS_ERROR,
        message: NpsimNativeDecoderStringViewV1 {
            ptr: message.as_ptr().cast::<c_char>(),
            len: message.len(),
        },
    }
}

fn state_error(state: &DecoderState, message: impl Into<String>) -> NpsimNativeDecoderStatusV1 {
    let sanitized = message.into().replace('\0', "\\0");
    let mut last_error = state
        .last_error
        .lock()
        .expect("fusion-blossom error mutex poisoned");
    *last_error = CString::new(sanitized).expect("NUL was sanitized");
    NpsimNativeDecoderStatusV1 {
        code: NATIVE_DECODER_PLUGIN_STATUS_ERROR,
        message: NpsimNativeDecoderStringViewV1 {
            ptr: last_error.as_ptr(),
            len: last_error.as_bytes().len(),
        },
    }
}

fn string_view_to_string(view: NpsimNativeDecoderStringViewV1) -> String {
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
    module.add_class::<PyNativeFusionBlossomNativeDecoder>()?;
    module.add_class::<PyInvalidNativeDecoderCapsule>()?;
    Ok(())
}

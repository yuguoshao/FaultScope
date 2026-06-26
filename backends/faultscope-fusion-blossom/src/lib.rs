use fusion_blossom::complete_graph::CompleteGraph;
use fusion_blossom::dual_module::{DualNodeClass, DualNodePtr};
use fusion_blossom::mwpm_solver::{PrimalDualSolver, SolverSerial};
use fusion_blossom::pointers::RwLockPtr;
use fusion_blossom::primal_module::PerfectMatching;
use fusion_blossom::util::{
    EdgeIndex, SolverInitializer, SyndromePattern, VertexIndex, VertexNum, Weight,
};
use faultscope_core::{
    log_likelihood_ratio, FaultScopeNativeCorrectionMaskBatchMutViewV1, FaultScopeNativeDecoderI64SliceV1,
    FaultScopeNativeDecoderStatusV1, FaultScopeNativeDecoderStringViewV1, FaultScopeNativeDecoderV1,
    FaultScopeNativeDetectorMaskBatchViewV1, FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeNativePackedObservableShotBatchMutViewV1, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE, NATIVE_DECODER_PLUGIN_STATUS_ERROR,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyTuple};
use std::collections::HashMap;
use std::env;
use std::ffi::{c_void, CString};
use std::mem;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

const BACKEND_NAME: &str = "fusion-blossom";
const CAPSULE_NAME: &[u8] = b"faultscope.native_decoder_plugin.v1\0";
const DEFAULT_WEIGHT_SCALE: f64 = 10_000.0;

#[pyclass(name = "NativeFusionBlossomNativeDecoder")]
struct PyNativeFusionBlossomNativeDecoder {
    capsule: Py<PyAny>,
    detector_ids: Vec<i64>,
    detector_coords: Vec<Vec<f64>>,
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
        let detector_coords = builder.detector_coords.clone();
        let observable_ids = builder.observable_ids.clone();
        let built = builder.build()?;
        let build_summary = built.summary;
        let solver = SolverBackend::new(&built.initializer).map_err(PyValueError::new_err)?;
        let worker_states = build_worker_state_slots();
        let state = Box::new(DecoderState {
            detector_ids: detector_ids.clone(),
            observable_ids: observable_ids.clone(),
            detector_to_solver_vertices: built.detector_to_solver_vertices,
            is_virtual_vertex: built.is_virtual_vertex,
            path_resolver: Mutex::new(PathEffectResolver::new(&built.initializer)),
            initializer: built.initializer.clone(),
            solver: Mutex::new(solver),
            worker_states,
            edge_effects: built.edge_effects,
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
        });
        let capsule = unsafe { create_decoder_capsule(py, Box::into_raw(descriptor))? };

        Ok(Self {
            capsule,
            detector_ids,
            detector_coords,
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
    fn detector_coords(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = self
            .detector_coords
            .iter()
            .map(|coords| {
                PyTuple::new(py, coords.iter().copied()).map(|item| item.into_any().unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, items.iter().map(|item| item.clone_ref(py)))?.into())
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
                .expect("fusion-blossom descriptor has decode callback")(
                descriptor.state,
                &input,
                &mut output,
            )
        };
        if status.code != faultscope_core::NATIVE_DECODER_PLUGIN_STATUS_OK {
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
        let initializer = SolverInitializer::new(2, Vec::new(), Vec::new());
        let solver = SolverBackend::new(&initializer).map_err(PyValueError::new_err)?;
        let state = Box::new(DecoderState {
            detector_ids: vec![0],
            observable_ids: vec![0],
            detector_to_solver_vertices: vec![0],
            is_virtual_vertex: vec![false, true],
            path_resolver: Mutex::new(PathEffectResolver::new(&initializer)),
            initializer,
            solver: Mutex::new(solver),
            worker_states: Vec::new(),
            edge_effects: Vec::new(),
            last_error: Mutex::new(CString::new("").expect("empty CString")),
        });
        let mut descriptor = FaultScopeNativeDecoderV1 {
            abi_version: NATIVE_DECODER_PLUGIN_ABI_VERSION,
            struct_size: mem::size_of::<FaultScopeNativeDecoderV1>(),
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

    fn __faultscope_native_decoder_capsule__(&self, py: Python<'_>) -> Py<PyAny> {
        self.capsule.clone_ref(py)
    }
}

struct DecoderState {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    detector_to_solver_vertices: Vec<VertexIndex>,
    is_virtual_vertex: Vec<bool>,
    path_resolver: Mutex<PathEffectResolver>,
    initializer: SolverInitializer,
    solver: Mutex<SolverBackend>,
    worker_states: Vec<Mutex<Option<WorkerDecodeState>>>,
    edge_effects: Vec<SolverEdgeEffect>,
    last_error: Mutex<CString>,
}

struct WorkerDecodeState {
    solver: SolverBackend,
    path_resolver: PathEffectResolver,
}

enum SolverBackend {
    Serial(SolverSerial),
}

#[derive(Default)]
struct SolverProfileTimings {
    clear_time: Duration,
    solve_time: Duration,
    extract_time: Duration,
}

struct BuiltFusionBlossomProblem {
    initializer: SolverInitializer,
    detector_to_solver_vertices: Vec<VertexIndex>,
    is_virtual_vertex: Vec<bool>,
    edge_effects: Vec<SolverEdgeEffect>,
    summary: BuildSummary,
}

struct FusionBlossomProblemBuilder {
    detector_ids: Vec<i64>,
    detector_coords: Vec<Vec<f64>>,
    observable_ids: Vec<i64>,
    weight_scale: f64,
    dem_edge_count: usize,
    boundary_groups: Vec<BoundaryEdgeGroup>,
    boundary_group_by_key: HashMap<(usize, Vec<usize>), usize>,
    graph_groups: Vec<GraphEdgeGroup>,
    graph_group_by_endpoint: HashMap<(usize, usize), usize>,
}

#[derive(Debug, Clone)]
struct BoundaryEdgeGroup {
    detector: usize,
    dem_edge_indices: Vec<usize>,
    fault_observables: Vec<usize>,
    probabilities: Vec<f64>,
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
    packed_observable_flips: Vec<(usize, u8)>,
    observable_mask64: Option<u64>,
    probability: f64,
    weight: f64,
}

struct PathEffectResolver {
    complete_graph: CompleteGraph,
    vertex_pair_edges: HashMap<(VertexIndex, VertexIndex), usize>,
    path_cache: HashMap<(VertexIndex, VertexIndex), Vec<usize>>,
    observable_mask64_cache: HashMap<(VertexIndex, VertexIndex), u64>,
}

struct SolverVertexLayout {
    old_to_new: Vec<usize>,
    detector_to_solver_vertices: Vec<VertexIndex>,
    is_virtual_vertex: Vec<bool>,
}

#[derive(Debug, Clone)]
struct BuildSummary {
    dem_edge_count: usize,
    solver_vertex_count: usize,
    solver_edge_count: usize,
    boundary_vertex_count: usize,
    boundary_edge_group_count: usize,
    boundary_virtual_mode: &'static str,
    merged_parallel_edge_count: usize,
    edges: Vec<SolverEdgeEffect>,
}

impl SolverBackend {
    fn new(initializer: &SolverInitializer) -> Result<Self, String> {
        catch_unwind(AssertUnwindSafe(|| {
            Self::Serial(SolverSerial::new(initializer))
        }))
        .map_err(|_| "fusion-blossom serial solver panicked during construction".to_string())
    }

    fn solve_perfect_matching(&mut self, syndrome: &SyndromePattern) -> PerfectMatching {
        match self {
            Self::Serial(solver) => {
                solver.clear();
                solver.solve(syndrome);
                solver.perfect_matching()
            }
        }
    }

    fn solve_perfect_matching_profiled(
        &mut self,
        syndrome: &SyndromePattern,
        timings: &mut SolverProfileTimings,
    ) -> PerfectMatching {
        match self {
            Self::Serial(solver) => {
                let started = Instant::now();
                solver.clear();
                timings.clear_time += started.elapsed();
                let started = Instant::now();
                solver.solve(syndrome);
                timings.solve_time += started.elapsed();
                let started = Instant::now();
                let perfect_matching = solver.perfect_matching();
                timings.extract_time += started.elapsed();
                perfect_matching
            }
        }
    }
}

impl SolverVertexLayout {
    fn new(detector_count: usize, solver_vertex_count: usize) -> Self {
        let old_to_new = (0..solver_vertex_count).collect::<Vec<_>>();
        let detector_to_solver_vertices = (0..detector_count)
            .map(|index| index as VertexIndex)
            .collect::<Vec<_>>();
        let mut is_virtual_vertex = vec![false; solver_vertex_count];
        for value in is_virtual_vertex.iter_mut().skip(detector_count) {
            *value = true;
        }
        Self {
            old_to_new,
            detector_to_solver_vertices,
            is_virtual_vertex,
        }
    }
}

impl FusionBlossomProblemBuilder {
    fn from_graphlike_problem(problem: &Bound<'_, PyAny>, weight_scale: f64) -> PyResult<Self> {
        let detector_ids = problem.getattr("detector_ids")?.extract::<Vec<i64>>()?;
        let detector_coords = problem
            .getattr("detector_coords")
            .and_then(|coords| coords.extract::<Vec<Vec<f64>>>())
            .unwrap_or_else(|_| vec![Vec::new(); detector_ids.len()]);
        if detector_coords.len() != detector_ids.len() {
            return Err(PyValueError::new_err(format!(
                "fusion-blossom graphlike problem has {} detector coords entries but {} detector ids",
                detector_coords.len(),
                detector_ids.len()
            )));
        }
        let observable_ids = problem.getattr("observable_ids")?.extract::<Vec<i64>>()?;
        let mut builder = Self {
            detector_ids,
            detector_coords,
            observable_ids,
            weight_scale,
            dem_edge_count: 0,
            boundary_groups: Vec::new(),
            boundary_group_by_key: HashMap::new(),
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
        _weight: f64,
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
                let key = (*detector, fault_observables.clone());
                if let Some(group_index) = self.boundary_group_by_key.get(&key).copied() {
                    let group = &mut self.boundary_groups[group_index];
                    group.dem_edge_indices.push(dem_edge_index);
                    group.probabilities.push(probability);
                } else {
                    let group_index = self.boundary_groups.len();
                    self.boundary_group_by_key.insert(key, group_index);
                    self.boundary_groups.push(BoundaryEdgeGroup {
                        detector: *detector,
                        dem_edge_indices: vec![dem_edge_index],
                        fault_observables,
                        probabilities: vec![probability],
                    });
                }
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
        let boundary_edge_group_count = self.boundary_groups.len();
        let mut boundary_virtual_for_group = Vec::with_capacity(boundary_edge_group_count);
        let mut boundary_virtual_detectors = Vec::new();
        for (boundary_index, group) in self.boundary_groups.iter().enumerate() {
            boundary_virtual_for_group.push(boundary_index);
            boundary_virtual_detectors.push(group.detector);
        }
        let boundary_vertex_count = boundary_virtual_detectors.len();
        let solver_vertex_count = detector_count + boundary_vertex_count;
        if solver_vertex_count < 2 {
            return Err(PyValueError::new_err(
                "fusion-blossom requires at least two graph vertices",
            ));
        }

        let mut scaled_edges =
            Vec::with_capacity(boundary_edge_group_count + self.graph_groups.len());
        let mut virtual_vertices = Vec::with_capacity(boundary_vertex_count);
        let mut edge_effects =
            Vec::with_capacity(boundary_edge_group_count + self.graph_groups.len());
        let mut merged_parallel_edge_count = 0;
        let vertex_layout = SolverVertexLayout::new(detector_count, solver_vertex_count);
        for boundary_index in 0..boundary_vertex_count {
            let virtual_vertex = detector_count + boundary_index;
            virtual_vertices.push(vertex_layout.old_to_new[virtual_vertex] as VertexIndex);
        }

        for (group_index, group) in self.boundary_groups.iter().enumerate() {
            if group.dem_edge_indices.len() > 1 {
                merged_parallel_edge_count += group.dem_edge_indices.len() - 1;
            }
            let probability = odd_parity_probability(&group.probabilities);
            let weight = log_likelihood_ratio(probability);
            let boundary_index = boundary_virtual_for_group[group_index];
            let virtual_vertex = detector_count + boundary_index;
            let endpoint = normalized_endpoint(
                vertex_layout.old_to_new[group.detector],
                vertex_layout.old_to_new[virtual_vertex],
            );
            let edge_label = format!("DEM edges {:?}", group.dem_edge_indices);
            let scaled = scaled_weight(weight, self.weight_scale, &edge_label)?;
            scaled_edges.push((endpoint.0, endpoint.1, scaled));
            edge_effects.push(SolverEdgeEffect::new(
                endpoint,
                group.dem_edge_indices.clone(),
                group.fault_observables.clone(),
                probability,
                weight,
            ));
        }

        for group in &self.graph_groups {
            if group.dem_edge_indices.len() > 1 {
                merged_parallel_edge_count += group.dem_edge_indices.len() - 1;
            }
            let probability = odd_parity_probability(&group.probabilities);
            let weight = log_likelihood_ratio(probability);
            let edge_label = format!("DEM edges {:?}", group.dem_edge_indices);
            let scaled = scaled_weight(weight, self.weight_scale, &edge_label)?;
            let endpoint = normalized_endpoint(
                vertex_layout.old_to_new[group.endpoint.0],
                vertex_layout.old_to_new[group.endpoint.1],
            );
            scaled_edges.push((endpoint.0, endpoint.1, scaled));
            edge_effects.push(SolverEdgeEffect::new(
                endpoint,
                group.dem_edge_indices.clone(),
                group.fault_observables.clone(),
                probability,
                weight,
            ));
        }

        normalize_scaled_edges_by_gcd(&mut scaled_edges);
        let solver_edge_count = scaled_edges.len();
        validate_solver_index_bounds(solver_vertex_count, solver_edge_count)?;
        let weighted_edges = scaled_edges_to_solver_edges(scaled_edges, solver_vertex_count)?;
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
            boundary_edge_group_count,
            boundary_virtual_mode: "per-boundary-group",
            merged_parallel_edge_count,
            edges: edge_effects.clone(),
        };
        Ok(BuiltFusionBlossomProblem {
            initializer,
            detector_to_solver_vertices: vertex_layout.detector_to_solver_vertices,
            is_virtual_vertex: vertex_layout.is_virtual_vertex,
            edge_effects,
            summary,
        })
    }
}

impl SolverEdgeEffect {
    fn new(
        endpoint: (usize, usize),
        dem_edge_indices: Vec<usize>,
        fault_observables: Vec<usize>,
        probability: f64,
        weight: f64,
    ) -> Self {
        let packed_observable_flips = fault_observables
            .iter()
            .map(|observable_index| (observable_index >> 3, 1u8 << (observable_index & 7)))
            .collect();
        let observable_mask64 = observable_mask64(&fault_observables);
        Self {
            endpoint,
            dem_edge_indices,
            fault_observables,
            packed_observable_flips,
            observable_mask64,
            probability,
            weight,
        }
    }
}

impl PathEffectResolver {
    fn new(initializer: &SolverInitializer) -> Self {
        let mut vertex_pair_edges = HashMap::with_capacity(initializer.weighted_edges.len());
        for (edge_index, (left, right, _weight)) in initializer.weighted_edges.iter().enumerate() {
            vertex_pair_edges.insert(normalized_vertex_endpoint(*left, *right), edge_index);
        }
        Self {
            complete_graph: CompleteGraph::new(initializer.vertex_num, &initializer.weighted_edges),
            vertex_pair_edges,
            path_cache: HashMap::new(),
            observable_mask64_cache: HashMap::new(),
        }
    }

    fn path_edges(&mut self, left: VertexIndex, right: VertexIndex) -> Result<&[usize], String> {
        let key = normalized_vertex_endpoint(left, right);
        if self.path_cache.contains_key(&key) {
            return Ok(self
                .path_cache
                .get(&key)
                .expect("path cache contains key")
                .as_slice());
        }
        if let Some(edge_index) = self.vertex_pair_edges.get(&key).copied() {
            self.path_cache
                .entry(key)
                .or_insert_with(|| vec![edge_index]);
            return Ok(self
                .path_cache
                .get(&key)
                .expect("direct edge path was just populated")
                .as_slice());
        }
        self.ensure_path_edges(key, left, right)?;
        Ok(self
            .path_cache
            .get(&key)
            .expect("path cache was just populated")
            .as_slice())
    }

    fn path_observable_mask64(
        &mut self,
        left: VertexIndex,
        right: VertexIndex,
        edge_effects: &[SolverEdgeEffect],
    ) -> Result<u64, String> {
        let key = normalized_vertex_endpoint(left, right);
        if let Some(mask) = self.observable_mask64_cache.get(&key).copied() {
            return Ok(mask);
        }
        if let Some(edge_index) = self.vertex_pair_edges.get(&key).copied() {
            let Some(edge_effect) = edge_effects.get(edge_index) else {
                return Err(format!(
                    "fusion-blossom solver returned unknown edge index {edge_index}"
                ));
            };
            let Some(edge_mask) = edge_effect.observable_mask64 else {
                return Err(
                    "fusion-blossom observable mask64 path requested for more than 64 observables"
                        .to_string(),
                );
            };
            self.observable_mask64_cache.insert(key, edge_mask);
            return Ok(edge_mask);
        }
        if !self.observable_mask64_cache.contains_key(&key) {
            self.ensure_path_edges(key, left, right)?;
            let path_edges = self
                .path_cache
                .get(&key)
                .expect("path cache was just populated");
            let mut mask = 0u64;
            for &edge_index in path_edges {
                let Some(edge_effect) = edge_effects.get(edge_index) else {
                    return Err(format!(
                        "fusion-blossom solver returned unknown edge index {edge_index}"
                    ));
                };
                let Some(edge_mask) = edge_effect.observable_mask64 else {
                    return Err(
                        "fusion-blossom observable mask64 path requested for more than 64 observables"
                            .to_string(),
                    );
                };
                mask ^= edge_mask;
            }
            self.observable_mask64_cache.insert(key, mask);
        }
        Ok(*self
            .observable_mask64_cache
            .get(&key)
            .expect("observable mask64 cache was just populated"))
    }

    fn ensure_path_edges(
        &mut self,
        key: (VertexIndex, VertexIndex),
        left: VertexIndex,
        right: VertexIndex,
    ) -> Result<(), String> {
        if !self.path_cache.contains_key(&key) {
            let edges = self.compute_path_edges(left, right)?;
            self.path_cache.insert(key, edges);
        }
        Ok(())
    }

    fn compute_path_edges(
        &mut self,
        left: VertexIndex,
        right: VertexIndex,
    ) -> Result<Vec<usize>, String> {
        let (path, _) = self.complete_graph.get_path(left, right);
        let mut edges = Vec::with_capacity(path.len());
        let mut previous = left;
        for (vertex, _) in path {
            let edge_key = normalized_vertex_endpoint(previous, vertex);
            let Some(edge_index) = self.vertex_pair_edges.get(&edge_key).copied() else {
                return Err(format!(
                    "fusion-blossom path recovery found unknown solver edge {:?}",
                    edge_key
                ));
            };
            edges.push(edge_index);
            previous = vertex;
        }
        Ok(edges)
    }
}

impl WorkerDecodeState {
    fn new(initializer: &SolverInitializer) -> Result<Self, String> {
        Ok(Self {
            solver: SolverBackend::new(initializer)?,
            path_resolver: PathEffectResolver::new(initializer),
        })
    }
}

fn build_worker_state_slots() -> Vec<Mutex<Option<WorkerDecodeState>>> {
    let worker_count = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    (0..worker_count).map(|_| Mutex::new(None)).collect()
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

fn scaled_weight(weight: f64, weight_scale: f64, edge_label: &str) -> PyResult<i128> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(PyValueError::new_err(format!(
            "fusion-blossom {edge_label} has non-finite or negative weight {weight}"
        )));
    }
    let scaled = (weight * weight_scale).round();
    let max_supported = i128::MAX as f64;
    let even_scaled = if (scaled as i128) % 2 == 0 {
        scaled
    } else {
        scaled + 1.0
    };
    if !even_scaled.is_finite() || even_scaled < 0.0 || even_scaled > max_supported {
        return Err(PyValueError::new_err(format!(
            "fusion-blossom {edge_label} scaled weight {even_scaled} exceeds backend maximum {max_supported}"
        )));
    }
    Ok(even_scaled as i128)
}

fn normalize_scaled_edges_by_gcd(weighted_edges: &mut [(usize, usize, i128)]) {
    let mut common = 0u128;
    for &(_, _, weight) in weighted_edges.iter() {
        let magnitude = weight.unsigned_abs();
        if magnitude == 0 {
            continue;
        }
        let half_magnitude = magnitude / 2;
        if half_magnitude == 0 {
            return;
        }
        common = if common == 0 {
            half_magnitude
        } else {
            integer_gcd(common, half_magnitude)
        };
        if common == 1 {
            return;
        }
    }
    if common <= 1 {
        return;
    }
    let divisor = common as i128;
    for (_, _, weight) in weighted_edges {
        *weight /= divisor;
    }
}

fn scaled_edges_to_solver_edges(
    scaled_edges: Vec<(usize, usize, i128)>,
    vertex_count: usize,
) -> PyResult<Vec<(VertexIndex, VertexIndex, Weight)>> {
    let max_safe = (Weight::MAX as i128) / (vertex_count.max(1) as i128);
    let mut weighted_edges = Vec::with_capacity(scaled_edges.len());
    for (left, right, weight) in scaled_edges {
        if weight < 0 || weight > max_safe {
            return Err(PyValueError::new_err(format!(
                "fusion-blossom normalized scaled weight {weight} exceeds safe maximum {max_safe}"
            )));
        }
        weighted_edges.push((left as VertexIndex, right as VertexIndex, weight as Weight));
    }
    Ok(weighted_edges)
}

fn integer_gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn validate_solver_index_bounds(vertex_count: usize, edge_count: usize) -> PyResult<()> {
    if vertex_count > VertexIndex::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "fusion-blossom solver graph has {vertex_count} vertices, exceeding backend vertex index limit {}",
            VertexIndex::MAX
        )));
    }
    if edge_count > EdgeIndex::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "fusion-blossom solver graph has {edge_count} edges, exceeding backend edge index limit {}",
            EdgeIndex::MAX
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

fn observable_mask64(fault_observables: &[usize]) -> Option<u64> {
    let mut mask = 0u64;
    for &observable_index in fault_observables {
        if observable_index >= 64 {
            return None;
        }
        mask ^= 1u64 << observable_index;
    }
    Some(mask)
}

fn normalized_endpoint(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn normalized_vertex_endpoint(left: VertexIndex, right: VertexIndex) -> (VertexIndex, VertexIndex) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

unsafe extern "C" fn forced_error_decode_batch(
    state: *mut c_void,
    _input: *const FaultScopeNativeDetectorMaskBatchViewV1,
    _output: *mut FaultScopeNativeCorrectionMaskBatchMutViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
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
    out: *mut FaultScopeNativeDecoderStringViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
    if out.is_null() {
        return static_error("fusion-blossom name callback received null output pointer");
    }
    if state.is_null() {
        return static_error("fusion-blossom name callback received null state pointer");
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
        return static_error("fusion-blossom ids callback received null output pointer");
    }
    if state.is_null() {
        return static_error("fusion-blossom ids callback received null state pointer");
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
    let mut solver = match state.solver.lock() {
        Ok(solver) => solver,
        Err(_) => return state_error(state, "fusion-blossom solver mutex poisoned"),
    };
    let mut path_resolver = match state.path_resolver.lock() {
        Ok(path_resolver) => path_resolver,
        Err(_) => return state_error(state, "fusion-blossom path resolver mutex poisoned"),
    };
    let decode_result = catch_unwind(AssertUnwindSafe(|| {
        decode_detector_major_with_solver(
            &mut solver,
            &mut path_resolver,
            &state.edge_effects,
            input_masks,
            output_masks,
            input.shots,
            &state.detector_to_solver_vertices,
            &state.is_virtual_vertex,
        )
    }));
    match decode_result {
        Ok(Ok(())) => {}
        Ok(Err(message)) => return state_error(state, message),
        Err(_) => return state_error(state, "fusion-blossom solver panicked during MWPM decode"),
    }
    FaultScopeNativeDecoderStatusV1::ok()
}

fn decode_detector_major_with_solver(
    solver: &mut SolverBackend,
    path_resolver: &mut PathEffectResolver,
    edge_effects: &[SolverEdgeEffect],
    input_masks: &[faultscope_core::FaultScopeNativeDecoderMaskViewV1],
    output_masks: &mut [faultscope_core::FaultScopeNativeDecoderMaskMutViewV1],
    shots: usize,
    detector_to_solver_vertices: &[VertexIndex],
    is_virtual_vertex: &[bool],
) -> Result<(), String> {
    let mut syndrome = SyndromePattern::new(Vec::with_capacity(input_masks.len()), Vec::new());
    for shot in 0..shots {
        syndrome.defect_vertices.clear();
        for (detector_index, input_mask) in input_masks.iter().enumerate() {
            if unsafe { read_bit(input_mask.words, shot) } {
                syndrome
                    .defect_vertices
                    .push(detector_to_solver_vertices[detector_index]);
            }
        }
        if syndrome.defect_vertices.is_empty() {
            continue;
        }
        let word_index = shot / 64;
        let bit_mask = 1u64 << (shot % 64);
        let perfect_matching = solver.solve_perfect_matching(&syndrome);
        if output_masks.len() <= 64 {
            apply_matching_masks64(
                &perfect_matching,
                path_resolver,
                edge_effects,
                is_virtual_vertex,
                |observable_mask| {
                    let mut mask = observable_mask;
                    while mask != 0 {
                        let observable_index = mask.trailing_zeros() as usize;
                        unsafe {
                            *output_masks[observable_index].words.add(word_index) ^= bit_mask;
                        }
                        mask &= mask - 1;
                    }
                    Ok(())
                },
            )?;
        } else {
            apply_matching_paths(
                &perfect_matching,
                path_resolver,
                is_virtual_vertex,
                |edge_index| {
                    let Some(edge_effect) = edge_effects.get(edge_index) else {
                        return Err(format!(
                            "fusion-blossom solver returned unknown edge index {edge_index}"
                        ));
                    };
                    for &observable_index in &edge_effect.fault_observables {
                        unsafe {
                            *output_masks[observable_index].words.add(word_index) ^= bit_mask;
                        }
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(())
}

fn apply_matching_masks64<F>(
    perfect_matching: &PerfectMatching,
    path_resolver: &mut PathEffectResolver,
    edge_effects: &[SolverEdgeEffect],
    is_virtual_vertex: &[bool],
    mut apply_mask: F,
) -> Result<(), String>
where
    F: FnMut(u64) -> Result<(), String>,
{
    for (left, right) in &perfect_matching.peer_matchings {
        let left = defect_vertex(left)?;
        let right = defect_vertex(right)?;
        let mask = path_resolver.path_observable_mask64(left, right, edge_effects)?;
        apply_mask(mask)?;
    }
    for (left, right) in &perfect_matching.virtual_matchings {
        let left = defect_vertex(left)?;
        if !is_virtual_vertex
            .get(*right as usize)
            .copied()
            .unwrap_or(false)
        {
            return Err(format!(
                "fusion-blossom virtual matching target {right} is not a virtual vertex"
            ));
        }
        let mask = path_resolver.path_observable_mask64(left, *right, edge_effects)?;
        apply_mask(mask)?;
    }
    Ok(())
}

fn apply_matching_paths<F>(
    perfect_matching: &PerfectMatching,
    path_resolver: &mut PathEffectResolver,
    is_virtual_vertex: &[bool],
    mut apply_edge: F,
) -> Result<(), String>
where
    F: FnMut(usize) -> Result<(), String>,
{
    for (left, right) in &perfect_matching.peer_matchings {
        let left = defect_vertex(left)?;
        let right = defect_vertex(right)?;
        apply_path_edges(left, right, path_resolver, &mut apply_edge)?;
    }
    for (left, right) in &perfect_matching.virtual_matchings {
        let left = defect_vertex(left)?;
        if !is_virtual_vertex
            .get(*right as usize)
            .copied()
            .unwrap_or(false)
        {
            return Err(format!(
                "fusion-blossom virtual matching target {right} is not a virtual vertex"
            ));
        }
        apply_path_edges(left, *right, path_resolver, &mut apply_edge)?;
    }
    Ok(())
}

fn apply_path_edges<F>(
    left: VertexIndex,
    right: VertexIndex,
    path_resolver: &mut PathEffectResolver,
    apply_edge: &mut F,
) -> Result<(), String>
where
    F: FnMut(usize) -> Result<(), String>,
{
    for &edge_index in path_resolver.path_edges(left, right)? {
        apply_edge(edge_index)?;
    }
    Ok(())
}

fn defect_vertex(node: &DualNodePtr) -> Result<VertexIndex, String> {
    let node = node.read_recursive();
    match &node.class {
        DualNodeClass::DefectVertex { defect_index } => Ok(*defect_index),
        DualNodeClass::Blossom { .. } => {
            Err("fusion-blossom perfect matching contained an unexpanded blossom node".to_string())
        }
    }
}

unsafe extern "C" fn decoder_decode_packed_batch(
    state: *mut c_void,
    input: *const FaultScopeNativePackedDetectorShotBatchViewV1,
    output: *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
) -> FaultScopeNativeDecoderStatusV1 {
    if state.is_null() {
        return static_error("fusion-blossom packed decode callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    if input.is_null() {
        return state_error(
            state,
            "fusion-blossom packed decode callback received null input pointer",
        );
    }
    if output.is_null() {
        return state_error(
            state,
            "fusion-blossom packed decode callback received null output pointer",
        );
    }
    let input = &*input;
    let output = &mut *output;
    if input.detector_count != state.detector_ids.len() {
        return state_error(
            state,
            format!(
                "fusion-blossom expected {} packed detector columns but received {}",
                state.detector_ids.len(),
                input.detector_count
            ),
        );
    }
    if output.observable_count != state.observable_ids.len() {
        return state_error(
            state,
            format!(
                "fusion-blossom expected {} packed observable columns but received {}",
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
                "fusion-blossom packed detector byte count is {}; expected {expected_detector_bytes}",
                input.detector_byte_count
            ),
        );
    }
    let expected_observable_bytes = state.observable_ids.len().div_ceil(8);
    if output.observable_byte_count != expected_observable_bytes {
        return state_error(
            state,
            format!(
                "fusion-blossom packed observable byte count is {}; expected {expected_observable_bytes}",
                output.observable_byte_count
            ),
        );
    }
    if input.detector_count > 0 && input.detector_ids.is_null() {
        return state_error(
            state,
            "fusion-blossom packed input detector ids pointer is null",
        );
    }
    if output.observable_count > 0 && output.observable_ids.is_null() {
        return state_error(
            state,
            "fusion-blossom packed output observable ids pointer is null",
        );
    }
    if input.shots > 0 && input.detector_byte_count > 0 && input.data.is_null() {
        return state_error(state, "fusion-blossom packed input data pointer is null");
    }
    if output.shots > 0 && output.observable_byte_count > 0 && output.data.is_null() {
        return state_error(state, "fusion-blossom packed output data pointer is null");
    }
    if input.shots != output.shots {
        return state_error(
            state,
            format!(
                "fusion-blossom packed input has {} shots but output has {}",
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
                "fusion-blossom packed detector id order mismatch: expected {:?}, received {:?}",
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
                "fusion-blossom packed observable id order mismatch: expected {:?}, received {:?}",
                state.observable_ids, output_observable_ids
            ),
        );
    }

    let input_len = match input.shots.checked_mul(input.detector_byte_count) {
        Some(len) => len,
        None => return state_error(state, "fusion-blossom packed input byte length overflow"),
    };
    let output_len = match output.shots.checked_mul(output.observable_byte_count) {
        Some(len) => len,
        None => return state_error(state, "fusion-blossom packed output byte length overflow"),
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

    if input.shots == 0 || input.detector_count == 0 || output.observable_count == 0 {
        return FaultScopeNativeDecoderStatusV1::ok();
    }
    let worker_count = fusion_blossom_worker_count(input.shots, state.worker_states.len());
    let profile_enabled = env::var_os("NPSIM_FUSION_BLOSSOM_PROFILE").is_some();
    if worker_count <= 1 || input.shots < 1024 {
        if profile_enabled {
            eprintln!(
                "faultscope fusion-blossom scheduler: shots={} workers=1 block_rows={} blocks=1 detector_bytes={} observable_bytes={}",
                input.shots,
                input.shots,
                input.detector_byte_count,
                output.observable_byte_count,
            );
        }
        let mut solver = match state.solver.lock() {
            Ok(solver) => solver,
            Err(_) => return state_error(state, "fusion-blossom solver mutex poisoned"),
        };
        let mut path_resolver = match state.path_resolver.lock() {
            Ok(path_resolver) => path_resolver,
            Err(_) => return state_error(state, "fusion-blossom path resolver mutex poisoned"),
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            decode_packed_rows_with_solver(
                &mut solver,
                &mut path_resolver,
                &state.edge_effects,
                detector_shots,
                input.detector_count,
                input.detector_byte_count,
                observable_predictions,
                output.observable_byte_count,
                &state.detector_to_solver_vertices,
                &state.is_virtual_vertex,
                profile_enabled,
            )
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(message)) => return state_error(state, message),
            Err(_) => {
                return state_error(
                    state,
                    "fusion-blossom solver panicked during packed MWPM decode",
                );
            }
        }
        return FaultScopeNativeDecoderStatusV1::ok();
    }

    let edge_effects = &state.edge_effects;
    let shots = input.shots;
    let detector_count = input.detector_count;
    let detector_byte_count = input.detector_byte_count;
    let observable_byte_count = output.observable_byte_count;
    let detector_to_solver_vertices = &state.detector_to_solver_vertices;
    let is_virtual_vertex = &state.is_virtual_vertex;
    let initializer = &state.initializer;
    let worker_states = &state.worker_states;
    let block_rows = fusion_blossom_block_rows(shots, worker_count);
    let block_count = shots.div_ceil(block_rows);
    if profile_enabled {
        eprintln!(
            "faultscope fusion-blossom scheduler: shots={} workers={} block_rows={} blocks={} detector_bytes={} observable_bytes={}",
            shots,
            worker_count,
            block_rows,
            block_count,
            detector_byte_count,
            observable_byte_count,
        );
    }
    let next_block = AtomicUsize::new(0);
    let detector_ptr = detector_shots.as_ptr() as usize;
    let observable_ptr = observable_predictions.as_mut_ptr() as usize;
    let result = thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker_index in 0..worker_count {
            let next_block = &next_block;
            handles.push(scope.spawn(move || {
                let worker_slot = worker_states
                    .get(worker_index)
                    .ok_or_else(|| format!("fusion-blossom missing worker slot {worker_index}"))?;
                let mut worker = worker_slot
                    .lock()
                    .map_err(|_| "fusion-blossom worker state mutex poisoned".to_string())?;
                if worker.is_none() {
                    *worker = Some(WorkerDecodeState::new(initializer)?);
                }
                let worker = worker.as_mut().expect("worker state was just initialized");
                let worker_result = catch_unwind(AssertUnwindSafe(|| {
                    loop {
                        let block_index = next_block.fetch_add(1, Ordering::Relaxed);
                        if block_index >= block_count {
                            break;
                        }
                        let row_begin = block_index * block_rows;
                        let row_end = (row_begin + block_rows).min(shots);
                        let detector_begin = row_begin * detector_byte_count;
                        let detector_len = (row_end - row_begin) * detector_byte_count;
                        let observable_begin = row_begin * observable_byte_count;
                        let observable_len = (row_end - row_begin) * observable_byte_count;
                        let detector_chunk = unsafe {
                            slice::from_raw_parts(
                                (detector_ptr as *const u8).add(detector_begin),
                                detector_len,
                            )
                        };
                        let observable_chunk = unsafe {
                            slice::from_raw_parts_mut(
                                (observable_ptr as *mut u8).add(observable_begin),
                                observable_len,
                            )
                        };
                        decode_packed_rows_with_solver(
                            &mut worker.solver,
                            &mut worker.path_resolver,
                            edge_effects,
                            detector_chunk,
                            detector_count,
                            detector_byte_count,
                            observable_chunk,
                            observable_byte_count,
                            detector_to_solver_vertices,
                            is_virtual_vertex,
                            profile_enabled,
                        )?;
                    }
                    Ok(())
                }));
                match worker_result {
                    Ok(Ok(())) => {}
                    Ok(Err(message)) => return Err(message),
                    Err(_) => {
                        return Err(
                            "fusion-blossom solver panicked during packed MWPM decode".to_string()
                        );
                    }
                }
                Ok(())
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(message)) => return Err(message),
                Err(_) => return Err("fusion-blossom packed worker thread panicked".to_string()),
            }
        }
        Ok(())
    });
    if let Err(message) = result {
        return state_error(state, message);
    }
    FaultScopeNativeDecoderStatusV1::ok()
}

fn fusion_blossom_worker_count(shots: usize, worker_slot_count: usize) -> usize {
    let available_threads = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let configured_threads = env::var("NPSIM_FUSION_BLOSSOM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(available_threads);
    configured_threads
        .min(available_threads)
        .min(worker_slot_count.max(1))
        .min(shots)
}

fn fusion_blossom_block_rows(shots: usize, worker_count: usize) -> usize {
    if let Some(configured) = env::var("NPSIM_FUSION_BLOSSOM_BLOCK_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
    {
        return configured.min(shots.max(1));
    }
    let target_blocks = worker_count.saturating_mul(20).max(1);
    shots.div_ceil(target_blocks).clamp(128, 4096)
}

fn decode_packed_rows_with_solver(
    solver: &mut SolverBackend,
    path_resolver: &mut PathEffectResolver,
    edge_effects: &[SolverEdgeEffect],
    detector_shots: &[u8],
    detector_count: usize,
    detector_byte_count: usize,
    observable_predictions: &mut [u8],
    observable_byte_count: usize,
    detector_to_solver_vertices: &[VertexIndex],
    is_virtual_vertex: &[bool],
    profile_enabled: bool,
) -> Result<(), String> {
    if profile_enabled {
        return decode_packed_rows_with_solver_profiled(
            solver,
            path_resolver,
            edge_effects,
            detector_shots,
            detector_count,
            detector_byte_count,
            observable_predictions,
            observable_byte_count,
            detector_to_solver_vertices,
            is_virtual_vertex,
        );
    }
    let shots = detector_shots.len() / detector_byte_count;
    let mut syndrome = SyndromePattern::new(Vec::with_capacity(detector_count), Vec::new());
    for shot in 0..shots {
        syndrome.defect_vertices.clear();
        let detector_row_begin = shot * detector_byte_count;
        let detector_row_end = detector_row_begin + detector_byte_count;
        collect_packed_defects(
            &detector_shots[detector_row_begin..detector_row_end],
            detector_count,
            detector_to_solver_vertices,
            &mut syndrome.defect_vertices,
        );
        if syndrome.defect_vertices.is_empty() {
            continue;
        }
        let observable_row_begin = shot * observable_byte_count;
        let perfect_matching = solver.solve_perfect_matching(&syndrome);
        if observable_byte_count <= 8 {
            apply_matching_masks64(
                &perfect_matching,
                path_resolver,
                edge_effects,
                is_virtual_vertex,
                |observable_mask| {
                    if observable_mask == 0 {
                        return Ok(());
                    }
                    for byte_index in 0..observable_byte_count {
                        observable_predictions[observable_row_begin + byte_index] ^=
                            ((observable_mask >> (byte_index * 8)) & 0xFF) as u8;
                    }
                    Ok(())
                },
            )?;
        } else {
            apply_matching_paths(
                &perfect_matching,
                path_resolver,
                is_virtual_vertex,
                |edge_index| {
                    let Some(edge_effect) = edge_effects.get(edge_index) else {
                        return Err(format!(
                            "fusion-blossom solver returned unknown edge index {edge_index}"
                        ));
                    };
                    for &(observable_byte_offset, observable_bit_mask) in
                        &edge_effect.packed_observable_flips
                    {
                        observable_predictions[observable_row_begin + observable_byte_offset] ^=
                            observable_bit_mask;
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(())
}

fn decode_packed_rows_with_solver_profiled(
    solver: &mut SolverBackend,
    path_resolver: &mut PathEffectResolver,
    edge_effects: &[SolverEdgeEffect],
    detector_shots: &[u8],
    detector_count: usize,
    detector_byte_count: usize,
    observable_predictions: &mut [u8],
    observable_byte_count: usize,
    detector_to_solver_vertices: &[VertexIndex],
    is_virtual_vertex: &[bool],
) -> Result<(), String> {
    let shots = detector_shots.len() / detector_byte_count;
    let mut syndrome = SyndromePattern::new(Vec::with_capacity(detector_count), Vec::new());
    let mut nonzero_shots = 0usize;
    let mut defect_total = 0usize;
    let mut collect_time = Duration::ZERO;
    let mut apply_time = Duration::ZERO;
    let mut solver_timings = SolverProfileTimings::default();
    let total_started = Instant::now();
    for shot in 0..shots {
        syndrome.defect_vertices.clear();
        let detector_row_begin = shot * detector_byte_count;
        let detector_row_end = detector_row_begin + detector_byte_count;
        let started = Instant::now();
        collect_packed_defects(
            &detector_shots[detector_row_begin..detector_row_end],
            detector_count,
            detector_to_solver_vertices,
            &mut syndrome.defect_vertices,
        );
        collect_time += started.elapsed();
        if syndrome.defect_vertices.is_empty() {
            continue;
        }
        nonzero_shots += 1;
        defect_total += syndrome.defect_vertices.len();
        let perfect_matching =
            solver.solve_perfect_matching_profiled(&syndrome, &mut solver_timings);
        let observable_row_begin = shot * observable_byte_count;
        let started = Instant::now();
        if observable_byte_count <= 8 {
            apply_matching_masks64(
                &perfect_matching,
                path_resolver,
                edge_effects,
                is_virtual_vertex,
                |observable_mask| {
                    if observable_mask == 0 {
                        return Ok(());
                    }
                    for byte_index in 0..observable_byte_count {
                        observable_predictions[observable_row_begin + byte_index] ^=
                            ((observable_mask >> (byte_index * 8)) & 0xFF) as u8;
                    }
                    Ok(())
                },
            )?;
        } else {
            apply_matching_paths(
                &perfect_matching,
                path_resolver,
                is_virtual_vertex,
                |edge_index| {
                    let Some(edge_effect) = edge_effects.get(edge_index) else {
                        return Err(format!(
                            "fusion-blossom solver returned unknown edge index {edge_index}"
                        ));
                    };
                    for &(observable_byte_offset, observable_bit_mask) in
                        &edge_effect.packed_observable_flips
                    {
                        observable_predictions[observable_row_begin + observable_byte_offset] ^=
                            observable_bit_mask;
                    }
                    Ok(())
                },
            )?;
        }
        apply_time += started.elapsed();
    }
    let total_time = total_started.elapsed();
    eprintln!(
        "faultscope fusion-blossom profile: shots={} nonzero={} avg_defects={:.3} collect_s={:.6} clear_s={:.6} grow_s={:.6} extract_s={:.6} apply_s={:.6} total_s={:.6}",
        shots,
        nonzero_shots,
        if nonzero_shots == 0 {
            0.0
        } else {
            defect_total as f64 / nonzero_shots as f64
        },
        collect_time.as_secs_f64(),
        solver_timings.clear_time.as_secs_f64(),
        solver_timings.solve_time.as_secs_f64(),
        solver_timings.extract_time.as_secs_f64(),
        apply_time.as_secs_f64(),
        total_time.as_secs_f64(),
    );
    Ok(())
}

fn collect_packed_defects(
    row: &[u8],
    detector_count: usize,
    detector_to_solver_vertices: &[VertexIndex],
    defects: &mut Vec<VertexIndex>,
) {
    for (byte_index, byte) in row.iter().copied().enumerate() {
        let base_detector = byte_index * 8;
        if base_detector >= detector_count {
            break;
        }
        let valid_bits = (detector_count - base_detector).min(8);
        let valid_mask = if valid_bits == 8 {
            u8::MAX
        } else {
            (1u8 << valid_bits) - 1
        };
        let mut bits = byte & valid_mask;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            defects.push(detector_to_solver_vertices[base_detector + bit]);
            bits &= bits - 1;
        }
    }
}

unsafe fn read_bit(words: *const u64, shot: usize) -> bool {
    let word = *words.add(shot / 64);
    ((word >> (shot % 64)) & 1) != 0
}

fn build_summary_to_py(py: Python<'_>, summary: &BuildSummary) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("dem_edge_count", summary.dem_edge_count)?;
    out.set_item("solver_vertex_count", summary.solver_vertex_count)?;
    out.set_item("solver_edge_count", summary.solver_edge_count)?;
    out.set_item("boundary_vertex_count", summary.boundary_vertex_count)?;
    out.set_item(
        "boundary_edge_group_count",
        summary.boundary_edge_group_count,
    )?;
    out.set_item("boundary_virtual_mode", summary.boundary_virtual_mode)?;
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
        .expect("fusion-blossom error mutex poisoned");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_weighted_edges_without_creating_odd_weights() {
        let mut weighted_edges = vec![(0, 1, 12), (1, 2, 20), (2, 3, 0)];

        normalize_scaled_edges_by_gcd(&mut weighted_edges);

        assert_eq!(weighted_edges[0].2, 6);
        assert_eq!(weighted_edges[1].2, 10);
        assert_eq!(weighted_edges[2].2, 0);
        assert!(weighted_edges.iter().all(|(_, _, weight)| weight % 2 == 0));
    }

    #[test]
    fn skips_weight_normalization_when_evenness_would_be_lost() {
        let mut weighted_edges = vec![(0, 1, 2), (1, 2, 6)];

        normalize_scaled_edges_by_gcd(&mut weighted_edges);

        assert_eq!(weighted_edges[0].2, 2);
        assert_eq!(weighted_edges[1].2, 6);
    }

    #[test]
    fn validates_weight_bounds_after_gcd_normalization() {
        let vertex_count = 10;
        let max_safe = (Weight::MAX as i128) / (vertex_count as i128);
        let mut weighted_edges = vec![(0, 1, max_safe * 2), (1, 2, max_safe * 4)];

        normalize_scaled_edges_by_gcd(&mut weighted_edges);
        let converted = scaled_edges_to_solver_edges(weighted_edges, vertex_count)
            .expect("gcd-normalized weights should fit final solver bound");

        assert_eq!(converted[0].2, 2 as Weight);
        assert_eq!(converted[1].2, 4 as Weight);
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeFusionBlossomNativeDecoder>()?;
    module.add_class::<PyInvalidNativeDecoderCapsule>()?;
    Ok(())
}

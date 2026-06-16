use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple,
};
use rand::rngs::SmallRng as RandSmallRng;
use rand::{RngCore, SeedableRng};
use std::collections::{HashMap, HashSet};

const NATIVE_KERNEL_VERSION: &str = "0.1.0";

#[derive(Clone)]
enum NoiseModel {
    BernoulliPauli(String),
    MeasurementBitFlip,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
    PauliChannel(Vec<(String, f64)>),
}

#[derive(Clone)]
struct NoiseLocationSpec {
    id: String,
    model: NoiseModel,
    rate: f64,
    qubits: Vec<usize>,
    tags: HashMap<String, TagValue>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum TagValue {
    None,
    Bool(bool),
    Int(i64),
    Float(u64),
    String(String),
}

#[derive(Clone)]
enum DemEvent {
    Pauli(String),
    Bool(bool),
}

#[derive(Clone)]
struct DemDetectorSpec {
    id: i64,
    measurement_keys: Vec<String>,
}

#[derive(Clone)]
struct DemObservableSpec {
    id: i64,
    measurement_keys: Vec<String>,
    pauli_qubits: Vec<usize>,
    pauli: String,
}

#[derive(Clone)]
struct DemEdgeSpec {
    probability: f64,
    detectors: Vec<i64>,
    observables: Vec<i64>,
    location_id: String,
    tags: HashMap<String, TagValue>,
}

#[derive(Clone)]
struct DemLocationGroup {
    location_id: String,
    edge_indices: Vec<usize>,
    total_probability: f64,
}

struct GeneratedDemEdge {
    probability: f64,
    detectors: Vec<i64>,
    observables: Vec<i64>,
    location_id: String,
    event: DemEvent,
}

#[derive(Clone)]
struct NoiseOccurrence {
    op_index: usize,
    location: NoiseLocationSpec,
}

#[derive(Clone)]
struct DemRunRecord {
    measurements: HashMap<String, bool>,
    x_frame: Vec<u8>,
    z_frame: Vec<u8>,
}

#[derive(Clone)]
enum Op {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Pauli {
        qubits: Vec<usize>,
        pauli: String,
    },
    Noise(NoiseLocationSpec),
    Measure {
        qubit: usize,
        key: Option<String>,
        basis: String,
        noise: Option<NoiseLocationSpec>,
    },
    MeasurePauli {
        qubits: Vec<usize>,
        pauli: String,
        key: Option<String>,
        noise: Option<NoiseLocationSpec>,
    },
    Reset {
        qubit: usize,
        key: Option<String>,
        basis: String,
    },
    Detector {
        detector_id: i64,
        measurement_keys: Vec<String>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_keys: Vec<String>,
    },
}

#[derive(Clone)]
enum RunOp {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Noise(NoiseLocationSpec),
    Measure {
        qubits: Vec<usize>,
        pauli: String,
        key: Option<String>,
        ideal: Expr,
        noise: Option<NoiseLocationSpec>,
    },
    Reset {
        qubit: usize,
        key: Option<String>,
        basis: String,
        ideal: Expr,
    },
    Detector {
        detector_id: i64,
        measurement_keys: Vec<String>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_keys: Vec<String>,
    },
}

#[pyclass]
struct NativePackedSampler {
    n_qubits: usize,
    runtime_operations: Vec<RunOp>,
    noise_location_ids: Vec<String>,
    noise_locations: Vec<NoiseLocationSpec>,
    random_source_count: usize,
}

#[pyclass]
struct NativePackedBatch {
    state: RuntimeState,
}

#[pymethods]
impl NativePackedSampler {
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    #[getter]
    fn operation_count(&self) -> usize {
        self.runtime_operations.len()
    }

    #[pyo3(signature = (shots, seed=None))]
    fn sample(&self, py: Python<'_>, shots: usize, seed: Option<u64>) -> PyResult<PyObject> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let state = py.allow_threads(|| run_packed_sample(self, shots, seed, true))?;
        state.to_py(py)
    }

    #[pyo3(signature = (shots, seed=None))]
    fn sample_measurements(
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
    fn run_native_batch(
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
    fn estimate_hotspots(
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
    fn shots(&self) -> usize {
        self.state.shots
    }

    #[getter]
    fn all_mask(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_to_py(py, &self.state.all_mask)
    }

    #[getter]
    fn x_frame(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_vec_to_py(py, &self.state.x_frame)
    }

    #[getter]
    fn z_frame(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_vec_to_py(py, &self.state.z_frame)
    }

    #[getter]
    fn measurements(&self, py: Python<'_>) -> PyResult<PyObject> {
        map_to_py(py, &self.state.measurements)
    }

    #[getter]
    fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.state.detectors)
    }

    #[getter]
    fn observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.state.observables)
    }

    #[getter]
    fn noise_event_masks(&self, py: Python<'_>) -> PyResult<PyObject> {
        map_to_py(py, &self.state.event_masks)
    }

    fn x_mask(&self, py: Python<'_>, qubit: usize) -> PyResult<PyObject> {
        let mask = self
            .state
            .x_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_to_py(py, mask)
    }

    fn z_mask(&self, py: Python<'_>, qubit: usize) -> PyResult<PyObject> {
        let mask = self
            .state
            .z_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_to_py(py, mask)
    }

    fn measurement_mask(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        let mask = self
            .state
            .measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        mask_to_py(py, mask)
    }

    fn bit(&self, mask: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
        py_int_bit(mask, shot)
    }

    fn measurement_bit(&self, key: &str, shot: usize) -> PyResult<u8> {
        let mask = self
            .state
            .measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        mask_bit(mask, shot)
    }

    fn detector_bit(&self, detector_id: i64, shot: usize) -> PyResult<u8> {
        let mask =
            self.state.detectors.get(&detector_id).ok_or_else(|| {
                PyValueError::new_err(format!("unknown detector id {detector_id}"))
            })?;
        mask_bit(mask, shot)
    }

    fn observable_bit(&self, observable_id: i64, shot: usize) -> PyResult<u8> {
        let mask = self.state.observables.get(&observable_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown observable id {observable_id}"))
        })?;
        mask_bit(mask, shot)
    }

    fn x_bit(&self, qubit: usize, shot: usize) -> PyResult<u8> {
        let mask = self
            .state
            .x_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_bit(mask, shot)
    }

    fn z_bit(&self, qubit: usize, shot: usize) -> PyResult<u8> {
        let mask = self
            .state
            .z_frame
            .get(qubit)
            .ok_or_else(|| PyValueError::new_err(format!("unknown qubit {qubit}")))?;
        mask_bit(mask, shot)
    }
}

#[pyclass]
struct NativeDemSampler {
    detectors: Vec<i64>,
    observables: Vec<i64>,
    edges: Vec<DemEdgeSpec>,
    location_groups: Vec<DemLocationGroup>,
}

#[pyclass]
struct NativeDemBatch {
    batch: DemBatch,
}

#[pymethods]
impl NativeDemSampler {
    #[getter]
    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[pyo3(signature = (shots, seed=None))]
    fn run_batch(&self, py: Python<'_>, shots: usize, seed: Option<u64>) -> PyResult<PyObject> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let batch = py.allow_threads(|| {
            let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
            run_dem_batch(self, shots, &mut rng)
        });
        dem_batch_to_py(py, &batch)
    }

    #[pyo3(signature = (shots, seed=None))]
    fn run_native_batch(
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
            run_dem_batch(self, shots, &mut rng)
        });
        Ok(NativeDemBatch { batch })
    }

    #[pyo3(signature = (shots, seed=None, baseline=None, top_k=10))]
    fn estimate_default(
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
            let batch = run_dem_batch(self, shots, &mut rng);
            compute_dem_estimate(self, &batch, &batch.loss_mask, baseline, top_k)
        });
        dem_estimate_to_py(py, &estimate)
    }

    #[pyo3(signature = (batch, loss_mask, baseline=None, top_k=10))]
    fn estimate_hotspots(
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
    fn shots(&self) -> usize {
        self.batch.shots
    }

    #[getter]
    fn all_mask(&self, py: Python<'_>) -> PyResult<PyObject> {
        mask_to_py(py, &self.batch.all_mask)
    }

    #[getter]
    fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.batch.detectors)
    }

    #[getter]
    fn observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        int_map_to_py(py, &self.batch.observables)
    }

    #[getter]
    fn edge_event_masks(&self, py: Python<'_>) -> PyResult<PyObject> {
        let edge_masks = PyDict::new(py);
        for (edge_index, mask) in self.batch.edge_event_masks.iter().enumerate() {
            edge_masks.set_item(edge_index, mask_to_py(py, mask)?)?;
        }
        Ok(edge_masks.into())
    }

    fn bit(&self, mask: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
        py_int_bit(mask, shot)
    }

    fn detector_bit(&self, detector_id: i64, shot: usize) -> PyResult<u8> {
        let mask =
            self.batch.detectors.get(&detector_id).ok_or_else(|| {
                PyValueError::new_err(format!("unknown detector id {detector_id}"))
            })?;
        mask_bit(mask, shot)
    }

    fn observable_bit(&self, observable_id: i64, shot: usize) -> PyResult<u8> {
        let mask = self.batch.observables.get(&observable_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown observable id {observable_id}"))
        })?;
        mask_bit(mask, shot)
    }

    fn edge_event_bit(&self, edge_index: usize, shot: usize) -> PyResult<u8> {
        let mask =
            self.batch.edge_event_masks.get(edge_index).ok_or_else(|| {
                PyValueError::new_err(format!("unknown DEM edge index {edge_index}"))
            })?;
        mask_bit(mask, shot)
    }
}

#[pyfunction]
fn compile_sampler(spec: &Bound<'_, PyDict>) -> PyResult<NativePackedSampler> {
    let (n_qubits, operations) = parse_circuit_spec(spec)?;
    let compiled = compile_runtime_operations(n_qubits, operations)?;

    Ok(NativePackedSampler {
        n_qubits,
        runtime_operations: compiled.runtime_operations,
        noise_location_ids: compiled.noise_location_ids,
        noise_locations: compiled.noise_locations,
        random_source_count: compiled.random_source_count,
    })
}

#[pyfunction]
fn generate_dem(
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
fn compile_dem_sampler(spec: &Bound<'_, PyDict>) -> PyResult<NativeDemSampler> {
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
fn _npsim_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
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

impl Op {
    fn noise_locations(&self) -> Vec<&NoiseLocationSpec> {
        match self {
            Op::Noise(location) => vec![location],
            Op::Measure {
                noise: Some(location),
                ..
            } => vec![location],
            Op::MeasurePauli {
                noise: Some(location),
                ..
            } => vec![location],
            _ => Vec::new(),
        }
    }
}

struct CompiledCircuit {
    runtime_operations: Vec<RunOp>,
    noise_location_ids: Vec<String>,
    noise_locations: Vec<NoiseLocationSpec>,
    random_source_count: usize,
}

#[derive(Clone, Default)]
struct Expr {
    terms: Vec<usize>,
    constant: bool,
}

impl Expr {
    fn constant(value: bool) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
        }
    }

    fn random(source: usize) -> Self {
        Self {
            terms: vec![source],
            constant: false,
        }
    }

    fn xor_assign(&mut self, other: &Expr) {
        self.constant ^= other.constant;
        self.terms.extend_from_slice(&other.terms);
        self.terms.sort_unstable();
        let mut out = Vec::with_capacity(self.terms.len());
        let mut idx = 0;
        while idx < self.terms.len() {
            let value = self.terms[idx];
            let mut count = 1;
            idx += 1;
            while idx < self.terms.len() && self.terms[idx] == value {
                count += 1;
                idx += 1;
            }
            if count & 1 == 1 {
                out.push(value);
            }
        }
        self.terms = out;
    }

    fn toggle_constant(&mut self) {
        self.constant ^= true;
    }

    fn eval(&self, random_masks: &[Mask], words: usize, all_mask: &Mask) -> Mask {
        let mut out = Mask::zero(words);
        if self.constant {
            out.xor_assign(all_mask);
        }
        for source in &self.terms {
            out.xor_assign(&random_masks[*source]);
        }
        out
    }
}

struct SymbolicStabilizer {
    x: Vec<Vec<u8>>,
    z: Vec<Vec<u8>>,
    sign: Vec<Expr>,
    random_source_count: usize,
}

impl SymbolicStabilizer {
    fn zero(n_qubits: usize) -> Self {
        let x = vec![vec![0; n_qubits]; n_qubits];
        let mut z = vec![vec![0; n_qubits]; n_qubits];
        for qubit in 0..n_qubits {
            z[qubit][qubit] = 1;
        }
        Self {
            x,
            z,
            sign: vec![Expr::default(); n_qubits],
            random_source_count: 0,
        }
    }

    fn n_qubits(&self) -> usize {
        self.x.len()
    }

    fn apply_h(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row].toggle_constant();
            }
            self.x[row][qubit] = old_z;
            self.z[row][qubit] = old_x;
        }
    }

    fn apply_s(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row].toggle_constant();
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    fn apply_s_dag(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z == 0 {
                self.sign[row].toggle_constant();
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    fn apply_cx(&mut self, control: usize, target: usize) {
        for row in 0..self.n_qubits() {
            let x_c = self.x[row][control];
            let z_c = self.z[row][control];
            let x_t = self.x[row][target];
            let z_t = self.z[row][target];
            if (x_t & z_c & (x_c ^ z_t ^ 1)) != 0 {
                self.sign[row].toggle_constant();
            }
            self.x[row][target] ^= x_c;
            self.z[row][control] ^= z_t;
        }
    }

    fn apply_cz(&mut self, left: usize, right: usize) {
        self.apply_h(right);
        self.apply_cx(left, right);
        self.apply_h(right);
    }

    fn apply_swap(&mut self, left: usize, right: usize) {
        if left == right {
            return;
        }
        self.apply_cx(left, right);
        self.apply_cx(right, left);
        self.apply_cx(left, right);
    }

    fn apply_pauli_string(&mut self, x: &[u8], z: &[u8]) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row].toggle_constant();
            }
        }
    }

    fn apply_pauli_string_expr(&mut self, x: &[u8], z: &[u8], expr: &Expr) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row].xor_assign(expr);
            }
        }
    }

    fn measure_pauli_expr(&mut self, x: &[u8], z: &[u8]) -> PyResult<Expr> {
        let anti: Vec<usize> = (0..self.n_qubits())
            .filter(|row| symplectic_product(&self.x[*row], &self.z[*row], x, z) != 0)
            .collect();
        if anti.is_empty() {
            return self.deterministic_measurement_expr(x, z);
        }

        let source = self.random_source_count;
        self.random_source_count += 1;
        let outcome = Expr::random(source);
        let pivot = anti[0];
        let old_x = self.x[pivot].clone();
        let old_z = self.z[pivot].clone();
        let old_sign = self.sign[pivot].clone();

        for row in anti.into_iter().skip(1) {
            let (new_x, new_z, new_sign) = multiply_symbolic_rows(
                &self.x[row],
                &self.z[row],
                &self.sign[row],
                &old_x,
                &old_z,
                &old_sign,
            )?;
            self.x[row] = new_x;
            self.z[row] = new_z;
            self.sign[row] = new_sign;
        }

        self.x[pivot] = x.to_vec();
        self.z[pivot] = z.to_vec();
        self.sign[pivot] = outcome.clone();
        Ok(outcome)
    }

    fn deterministic_measurement_expr(&self, x: &[u8], z: &[u8]) -> PyResult<Expr> {
        let rows: Vec<Vec<u64>> = (0..self.n_qubits())
            .map(|row| support_to_words(&self.x[row], &self.z[row]))
            .collect();
        let target = support_to_words(x, z);
        let coeff = solve_row_span(&rows, &target, self.n_qubits()).ok_or_else(|| {
            PyValueError::new_err("commuting Pauli was not in the stabilizer span")
        })?;

        let mut selected_any = false;
        let mut acc_x = vec![0; self.n_qubits()];
        let mut acc_z = vec![0; self.n_qubits()];
        let mut acc_sign = Expr::default();
        for row in 0..self.n_qubits() {
            if !coeff_bit(&coeff, row) {
                continue;
            }
            if !selected_any {
                acc_x = self.x[row].clone();
                acc_z = self.z[row].clone();
                acc_sign = self.sign[row].clone();
                selected_any = true;
            } else {
                let (new_x, new_z, new_sign) = multiply_symbolic_rows(
                    &acc_x,
                    &acc_z,
                    &acc_sign,
                    &self.x[row],
                    &self.z[row],
                    &self.sign[row],
                )?;
                acc_x = new_x;
                acc_z = new_z;
                acc_sign = new_sign;
            }
        }
        if selected_any {
            Ok(acc_sign)
        } else {
            Ok(Expr::constant(false))
        }
    }
}

fn compile_runtime_operations(n_qubits: usize, operations: Vec<Op>) -> PyResult<CompiledCircuit> {
    let mut symbolic = SymbolicStabilizer::zero(n_qubits);
    let mut runtime_operations = Vec::new();
    let mut seen_noise_ids = HashSet::new();
    let mut noise_location_ids = Vec::new();
    let mut noise_locations = Vec::new();

    for operation in operations {
        for location in operation.noise_locations() {
            if !seen_noise_ids.insert(location.id.clone()) {
                return Err(PyValueError::new_err(format!(
                    "native sampler requires unique noise location ids; duplicate {:?}",
                    location.id
                )));
            }
            noise_location_ids.push(location.id.clone());
            noise_locations.push(location.clone());
        }

        match operation {
            Op::H(q) => {
                symbolic.apply_h(q);
                runtime_operations.push(RunOp::H(q));
            }
            Op::S(q) => {
                symbolic.apply_s(q);
                runtime_operations.push(RunOp::S(q));
            }
            Op::SDag(q) => {
                symbolic.apply_s_dag(q);
                runtime_operations.push(RunOp::SDag(q));
            }
            Op::Cx(c, t) => {
                symbolic.apply_cx(c, t);
                runtime_operations.push(RunOp::Cx(c, t));
            }
            Op::Cz(l, r) => {
                symbolic.apply_cz(l, r);
                runtime_operations.push(RunOp::Cz(l, r));
            }
            Op::Swap(l, r) => {
                symbolic.apply_swap(l, r);
                runtime_operations.push(RunOp::Swap(l, r));
            }
            Op::Pauli { qubits, pauli } => {
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &pauli)?;
                symbolic.apply_pauli_string(&x, &z);
            }
            Op::Noise(location) => runtime_operations.push(RunOp::Noise(location)),
            Op::Measure {
                qubit,
                key,
                basis,
                noise,
            } => {
                let qubits = vec![qubit];
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &basis)?;
                let ideal = symbolic.measure_pauli_expr(&x, &z)?;
                runtime_operations.push(RunOp::Measure {
                    qubits,
                    pauli: basis,
                    key,
                    ideal,
                    noise,
                });
            }
            Op::MeasurePauli {
                qubits,
                pauli,
                key,
                noise,
            } => {
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &pauli)?;
                let ideal = symbolic.measure_pauli_expr(&x, &z)?;
                runtime_operations.push(RunOp::Measure {
                    qubits,
                    pauli,
                    key,
                    ideal,
                    noise,
                });
            }
            Op::Reset { qubit, key, basis } => {
                let qubits = vec![qubit];
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &basis)?;
                let ideal = symbolic.measure_pauli_expr(&x, &z)?;
                let correction = match basis.as_str() {
                    "Z" => "X",
                    "X" => "Z",
                    "Y" => "X",
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported reset basis {basis:?}"
                        )))
                    }
                };
                let (cx, cz) = sparse_pauli_to_xz(n_qubits, &qubits, correction)?;
                symbolic.apply_pauli_string_expr(&cx, &cz, &ideal);
                runtime_operations.push(RunOp::Reset {
                    qubit,
                    key,
                    basis,
                    ideal,
                });
            }
            Op::Detector {
                detector_id,
                measurement_keys,
            } => runtime_operations.push(RunOp::Detector {
                detector_id,
                measurement_keys,
            }),
            Op::ObservableInclude {
                observable_id,
                measurement_keys,
            } => runtime_operations.push(RunOp::ObservableInclude {
                observable_id,
                measurement_keys,
            }),
        }
    }
    Ok(CompiledCircuit {
        runtime_operations,
        noise_location_ids,
        noise_locations,
        random_source_count: symbolic.random_source_count,
    })
}

fn parse_circuit_spec(spec: &Bound<'_, PyDict>) -> PyResult<(usize, Vec<Op>)> {
    let n_qubits = required(spec, "n_qubits")?.extract::<usize>()?;
    let operations_any = required(spec, "operations")?;
    let operations_seq = operations_any.downcast::<PyList>()?;
    let mut operations = Vec::with_capacity(operations_seq.len());
    for item in operations_seq.iter() {
        let dict = item.downcast::<PyDict>()?;
        operations.push(parse_operation(dict)?);
    }
    Ok((n_qubits, operations))
}

fn parse_operation(dict: &Bound<'_, PyDict>) -> PyResult<Op> {
    let kind = required(dict, "kind")?.extract::<String>()?;
    let qubits = optional_vec_usize(dict, "qubits")?.unwrap_or_default();
    match kind.as_str() {
        "h" => Ok(Op::H(one_qubit(&qubits, "h")?)),
        "s" => Ok(Op::S(one_qubit(&qubits, "s")?)),
        "s_dag" => Ok(Op::SDag(one_qubit(&qubits, "s_dag")?)),
        "cx" => {
            let (a, b) = two_qubits(&qubits, "cx")?;
            Ok(Op::Cx(a, b))
        }
        "cz" => {
            let (a, b) = two_qubits(&qubits, "cz")?;
            Ok(Op::Cz(a, b))
        }
        "swap" => {
            let (a, b) = two_qubits(&qubits, "swap")?;
            Ok(Op::Swap(a, b))
        }
        "pauli" => Ok(Op::Pauli {
            qubits,
            pauli: required(dict, "pauli")?.extract::<String>()?,
        }),
        "noise" => {
            let location = parse_noise_location(required(dict, "noise_location")?)?;
            Ok(Op::Noise(location))
        }
        "measure" => Ok(Op::Measure {
            qubit: one_qubit(&qubits, "measure")?,
            key: optional_string(dict, "key")?,
            basis: required(dict, "basis")?.extract::<String>()?.to_uppercase(),
            noise: optional_noise_location(dict)?,
        }),
        "measure_pauli" => Ok(Op::MeasurePauli {
            qubits,
            pauli: required(dict, "pauli")?.extract::<String>()?,
            key: optional_string(dict, "key")?,
            noise: optional_noise_location(dict)?,
        }),
        "reset" => Ok(Op::Reset {
            qubit: one_qubit(&qubits, "reset")?,
            key: optional_string(dict, "key")?,
            basis: required(dict, "basis")?.extract::<String>()?.to_uppercase(),
        }),
        "detector" => {
            let metadata_any = required(dict, "metadata")?;
            let metadata = metadata_any.downcast::<PyDict>()?;
            let detector_id = match metadata.get_item("detector_id")? {
                Some(value) if !value.is_none() => value.extract::<i64>()?,
                _ => -1,
            };
            Ok(Op::Detector {
                detector_id,
                measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
            })
        }
        "observable_include" => Ok(Op::ObservableInclude {
            observable_id: required(dict, "observable_id")?.extract::<i64>()?,
            measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unsupported native operation kind {kind:?}"
        ))),
    }
}

fn optional_noise_location(dict: &Bound<'_, PyDict>) -> PyResult<Option<NoiseLocationSpec>> {
    match dict.get_item("noise_location")? {
        Some(value) if !value.is_none() => Ok(Some(parse_noise_location(value)?)),
        _ => Ok(None),
    }
}

fn parse_noise_location(value: Bound<'_, PyAny>) -> PyResult<NoiseLocationSpec> {
    let dict = value.downcast::<PyDict>()?;
    let model_any = required(dict, "model")?;
    let model_dict = model_any.downcast::<PyDict>()?;
    let model_type = required(model_dict, "type")?.extract::<String>()?;
    let model = match model_type.as_str() {
        "bernoulli_pauli" => {
            NoiseModel::BernoulliPauli(required(model_dict, "pauli")?.extract::<String>()?)
        }
        "measurement_bit_flip" => NoiseModel::MeasurementBitFlip,
        "single_qubit_depolarizing" => NoiseModel::SingleQubitDepolarizing,
        "two_qubit_depolarizing" => NoiseModel::TwoQubitDepolarizing,
        "pauli_channel" => {
            let weights = required(model_dict, "weights")?.extract::<Vec<(String, f64)>>()?;
            NoiseModel::PauliChannel(weights)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported native noise model {model_type:?}"
            )))
        }
    };
    let rate = required(dict, "rate")?.extract::<f64>()?;
    if !(0.0..=1.0).contains(&rate) {
        return Err(PyValueError::new_err(format!(
            "noise rate must be in [0, 1], got {rate}"
        )));
    }
    Ok(NoiseLocationSpec {
        id: required(dict, "id")?.extract::<String>()?,
        model,
        rate,
        qubits: required(dict, "qubits")?.extract::<Vec<usize>>()?,
        tags: parse_optional_tags(dict)?,
    })
}

fn parse_optional_tags(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, TagValue>> {
    match dict.get_item("tags")? {
        Some(value) if !value.is_none() => {
            let tags = value.downcast::<PyDict>()?;
            let mut out = HashMap::new();
            for (key, value) in tags.iter() {
                out.insert(key.extract::<String>()?, parse_tag_value(value)?);
            }
            Ok(out)
        }
        _ => Ok(HashMap::new()),
    }
}

fn parse_tag_value(value: Bound<'_, PyAny>) -> PyResult<TagValue> {
    if value.is_none() {
        return Ok(TagValue::None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(TagValue::Bool(value.extract::<bool>()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(TagValue::String(value.extract::<String>()?));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(TagValue::Int(value.extract::<i64>()?));
    }
    if value.is_instance_of::<PyFloat>() {
        let value = value.extract::<f64>()?;
        if value.is_nan() {
            return Err(PyValueError::new_err("native tags do not support NaN"));
        }
        return Ok(TagValue::Float(value.to_bits()));
    }
    Err(PyValueError::new_err(
        "native tags support only str, int, float, bool, or None",
    ))
}

fn required<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("native sampler spec missing {key}")))
}

fn optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<String>()?)),
        _ => Ok(None),
    }
}

fn optional_vec_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<usize>>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<Vec<usize>>()?)),
        _ => Ok(None),
    }
}

fn one_qubit(qubits: &[usize], kind: &str) -> PyResult<usize> {
    if qubits.len() == 1 {
        Ok(qubits[0])
    } else {
        Err(PyValueError::new_err(format!(
            "{kind} requires exactly one qubit"
        )))
    }
}

fn two_qubits(qubits: &[usize], kind: &str) -> PyResult<(usize, usize)> {
    if qubits.len() == 2 {
        Ok((qubits[0], qubits[1]))
    } else {
        Err(PyValueError::new_err(format!("{kind} requires two qubits")))
    }
}

fn parse_dem_detectors(items: &Bound<'_, PyList>) -> PyResult<Vec<DemDetectorSpec>> {
    let mut out = Vec::with_capacity(items.len());
    let mut seen = HashSet::new();
    for item in items.iter() {
        let dict = item.downcast::<PyDict>()?;
        let id = required(dict, "id")?.extract::<i64>()?;
        if !seen.insert(id) {
            return Err(PyValueError::new_err("detector ids must be unique"));
        }
        out.push(DemDetectorSpec {
            id,
            measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
        });
    }
    Ok(out)
}

fn parse_dem_observables(items: &Bound<'_, PyList>) -> PyResult<Vec<DemObservableSpec>> {
    let mut out = Vec::with_capacity(items.len());
    let mut seen = HashSet::new();
    for item in items.iter() {
        let dict = item.downcast::<PyDict>()?;
        let id = required(dict, "id")?.extract::<i64>()?;
        if !seen.insert(id) {
            return Err(PyValueError::new_err(
                "logical observable ids must be unique",
            ));
        }
        let pauli_qubits = required(dict, "pauli_qubits")?.extract::<Vec<usize>>()?;
        let pauli = required(dict, "pauli")?.extract::<String>()?;
        if pauli_qubits.len() != pauli.len() {
            return Err(PyValueError::new_err(
                "pauli_qubits and pauli must have the same length",
            ));
        }
        out.push(DemObservableSpec {
            id,
            measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
            pauli_qubits,
            pauli,
        });
    }
    Ok(out)
}

fn generate_dem_edges(
    n_qubits: usize,
    operations: &[Op],
    detectors: &[DemDetectorSpec],
    observables: &[DemObservableSpec],
) -> PyResult<Vec<GeneratedDemEdge>> {
    let occurrences = collect_noise_occurrences(operations)?;
    let reference = run_dem_with_injection(n_qubits, operations, None, None)?;
    let reference_detectors = evaluate_dem_detectors(&reference, detectors)?;
    let reference_observables = evaluate_dem_observables(&reference, observables)?;
    let mut edges = Vec::new();

    for occurrence in occurrences {
        for (event, probability) in non_identity_events(&occurrence.location)? {
            let injected = run_dem_with_injection(
                n_qubits,
                operations,
                Some(occurrence.op_index),
                Some(&event),
            )?;
            let detector_flips = flipped_ids(
                &reference_detectors,
                &evaluate_dem_detectors(&injected, detectors)?,
            );
            let observable_flips = flipped_ids(
                &reference_observables,
                &evaluate_dem_observables(&injected, observables)?,
            );
            if detector_flips.is_empty() && observable_flips.is_empty() {
                continue;
            }
            edges.push(GeneratedDemEdge {
                probability,
                detectors: detector_flips,
                observables: observable_flips,
                location_id: occurrence.location.id.clone(),
                event,
            });
        }
    }
    Ok(edges)
}

fn collect_noise_occurrences(operations: &[Op]) -> PyResult<Vec<NoiseOccurrence>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (op_index, operation) in operations.iter().enumerate() {
        match operation {
            Op::Noise(location) => {
                if !seen.insert(location.id.clone()) {
                    return Err(PyValueError::new_err(format!(
                        "DetectorErrorModelGenerator requires unique noise location ids; duplicate id {:?}",
                        location.id
                    )));
                }
                out.push(NoiseOccurrence {
                    op_index,
                    location: location.clone(),
                });
            }
            Op::Measure {
                noise: Some(location),
                ..
            }
            | Op::MeasurePauli {
                noise: Some(location),
                ..
            } => {
                if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
                    return Err(PyValueError::new_err(
                        "DEM generation currently supports MeasurementBitFlip on measurement operations",
                    ));
                }
                if !seen.insert(location.id.clone()) {
                    return Err(PyValueError::new_err(format!(
                        "DetectorErrorModelGenerator requires unique noise location ids; duplicate id {:?}",
                        location.id
                    )));
                }
                out.push(NoiseOccurrence {
                    op_index,
                    location: location.clone(),
                });
            }
            _ => {}
        }
    }
    Ok(out)
}

fn non_identity_events(location: &NoiseLocationSpec) -> PyResult<Vec<(DemEvent, f64)>> {
    match &location.model {
        NoiseModel::BernoulliPauli(pauli) => {
            Ok(vec![(DemEvent::Pauli(pauli.clone()), location.rate)])
        }
        NoiseModel::MeasurementBitFlip => Ok(vec![(DemEvent::Bool(true), location.rate)]),
        NoiseModel::SingleQubitDepolarizing => Ok(["X", "Y", "Z"]
            .iter()
            .map(|event| (DemEvent::Pauli((*event).to_string()), location.rate / 3.0))
            .collect()),
        NoiseModel::TwoQubitDepolarizing => {
            let events = [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY",
                "ZZ",
            ];
            Ok(events
                .iter()
                .map(|event| {
                    (
                        DemEvent::Pauli((*event).to_string()),
                        location.rate / events.len() as f64,
                    )
                })
                .collect())
        }
        NoiseModel::PauliChannel(weights) => {
            let total: f64 = weights.iter().map(|(_, weight)| *weight).sum();
            if total <= 0.0 {
                return Err(PyValueError::new_err(
                    "PauliChannel weights must have positive total weight",
                ));
            }
            Ok(weights
                .iter()
                .filter(|(_, weight)| *weight > 0.0)
                .map(|(event, weight)| {
                    (
                        DemEvent::Pauli(event.clone()),
                        location.rate * *weight / total,
                    )
                })
                .collect())
        }
    }
}

fn run_dem_with_injection(
    n_qubits: usize,
    operations: &[Op],
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
) -> PyResult<DemRunRecord> {
    let mut state = ConcreteStabilizer::zero(n_qubits);
    let mut x_frame = vec![0; n_qubits];
    let mut z_frame = vec![0; n_qubits];
    let mut measurements = HashMap::new();
    for (op_index, operation) in operations.iter().enumerate() {
        apply_dem_operation(
            operation,
            op_index,
            injected_op_index,
            injected_event,
            &mut state,
            &mut x_frame,
            &mut z_frame,
            &mut measurements,
        )?;
    }
    Ok(DemRunRecord {
        measurements,
        x_frame,
        z_frame,
    })
}

fn apply_dem_operation(
    operation: &Op,
    op_index: usize,
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
    state: &mut ConcreteStabilizer,
    x_frame: &mut [u8],
    z_frame: &mut [u8],
    measurements: &mut HashMap<String, bool>,
) -> PyResult<()> {
    match operation {
        Op::H(q) => {
            state.apply_h(*q);
            frame_apply_h(x_frame, z_frame, *q);
        }
        Op::S(q) => {
            state.apply_s(*q);
            frame_apply_s(x_frame, z_frame, *q);
        }
        Op::SDag(q) => {
            state.apply_s_dag(*q);
            frame_apply_s(x_frame, z_frame, *q);
        }
        Op::Cx(control, target) => {
            state.apply_cx(*control, *target);
            frame_apply_cx(x_frame, z_frame, *control, *target);
        }
        Op::Cz(left, right) => {
            state.apply_cz(*left, *right);
            frame_apply_cz(x_frame, z_frame, *left, *right);
        }
        Op::Swap(left, right) => {
            state.apply_swap(*left, *right);
            frame_apply_swap(x_frame, z_frame, *left, *right);
        }
        Op::Pauli { qubits, pauli } => {
            let (x, z) = sparse_pauli_to_xz(state.n_qubits(), qubits, pauli)?;
            state.apply_pauli_string(&x, &z);
        }
        Op::Noise(location) => {
            if Some(op_index) == injected_op_index {
                let event = injected_event.ok_or_else(|| {
                    PyValueError::new_err("missing injected DEM event for noise operation")
                })?;
                apply_dem_noise_event(location, event, state, x_frame, z_frame)?;
            }
        }
        Op::Measure {
            qubit, key, basis, ..
        } => {
            let qubits = vec![*qubit];
            let bit = deterministic_dem_measurement(state, &qubits, basis, key.as_deref())?;
            let bit = maybe_flip_measurement_bit(bit, op_index, injected_op_index, injected_event)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", measurements.len()));
            record_dem_measurement(measurements, &key, bit)?;
        }
        Op::MeasurePauli {
            qubits, pauli, key, ..
        } => {
            let bit = deterministic_dem_measurement(state, qubits, pauli, key.as_deref())?;
            let bit = maybe_flip_measurement_bit(bit, op_index, injected_op_index, injected_event)?;
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", measurements.len()));
            record_dem_measurement(measurements, &key, bit)?;
        }
        Op::Reset { qubit, key, basis } => {
            let qubits = vec![*qubit];
            if let Some(key) = key {
                let bit = deterministic_dem_measurement(state, &qubits, basis, Some(key))?;
                record_dem_measurement(measurements, key, bit)?;
            }
            state.reset_prepare(*qubit, basis)?;
            x_frame[*qubit] = 0;
            z_frame[*qubit] = 0;
        }
        Op::Detector { .. } | Op::ObservableInclude { .. } => {}
    }
    Ok(())
}

fn deterministic_dem_measurement(
    state: &ConcreteStabilizer,
    qubits: &[usize],
    pauli: &str,
    key: Option<&str>,
) -> PyResult<bool> {
    let (x, z) = sparse_pauli_to_xz(state.n_qubits(), qubits, pauli)?;
    if !state.is_deterministic_pauli(&x, &z) {
        return Err(PyValueError::new_err(format!(
            "measurement {:?} is random in the ideal/single-error circuit",
            key.unwrap_or("measure")
        )));
    }
    state.deterministic_measurement_bit(&x, &z)
}

fn maybe_flip_measurement_bit(
    bit: bool,
    op_index: usize,
    injected_op_index: Option<usize>,
    injected_event: Option<&DemEvent>,
) -> PyResult<bool> {
    if Some(op_index) != injected_op_index {
        return Ok(bit);
    }
    match injected_event {
        Some(DemEvent::Bool(value)) => Ok(bit ^ *value),
        Some(DemEvent::Pauli(_)) => Err(PyValueError::new_err(
            "measurement injection requires a boolean event",
        )),
        None => Err(PyValueError::new_err(
            "missing injected DEM event for measurement operation",
        )),
    }
}

fn apply_dem_noise_event(
    location: &NoiseLocationSpec,
    event: &DemEvent,
    state: &mut ConcreteStabilizer,
    x_frame: &mut [u8],
    z_frame: &mut [u8],
) -> PyResult<()> {
    match event {
        DemEvent::Bool(_) => {
            if matches!(location.model, NoiseModel::MeasurementBitFlip) {
                Ok(())
            } else {
                Err(PyValueError::new_err(
                    "non-measurement DEM noise event must be a Pauli string",
                ))
            }
        }
        DemEvent::Pauli(pauli) => {
            let (x, z) = sparse_pauli_to_xz(state.n_qubits(), &location.qubits, pauli)?;
            state.apply_pauli_string(&x, &z);
            frame_apply_pauli_string(x_frame, z_frame, &location.qubits, pauli)
        }
    }
}

fn evaluate_dem_detectors(
    run: &DemRunRecord,
    detectors: &[DemDetectorSpec],
) -> PyResult<HashMap<i64, bool>> {
    let mut out = HashMap::new();
    for detector in detectors {
        out.insert(
            detector.id,
            dem_measurement_parity(&run.measurements, &detector.measurement_keys)?,
        );
    }
    Ok(out)
}

fn evaluate_dem_observables(
    run: &DemRunRecord,
    observables: &[DemObservableSpec],
) -> PyResult<HashMap<i64, bool>> {
    let mut out = HashMap::new();
    for observable in observables {
        let mut value = dem_measurement_parity(&run.measurements, &observable.measurement_keys)?;
        if !observable.pauli.is_empty() {
            value ^= frame_measurement_flip_bits(
                &run.x_frame,
                &run.z_frame,
                &observable.pauli_qubits,
                &observable.pauli,
            )?;
        }
        out.insert(observable.id, value);
    }
    Ok(out)
}

fn dem_measurement_parity(measurements: &HashMap<String, bool>, keys: &[String]) -> PyResult<bool> {
    let mut parity = false;
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        parity ^= *value;
    }
    Ok(parity)
}

fn record_dem_measurement(
    measurements: &mut HashMap<String, bool>,
    key: &str,
    bit: bool,
) -> PyResult<()> {
    if measurements.contains_key(key) {
        return Err(PyValueError::new_err(format!(
            "duplicate measurement key {key:?}"
        )));
    }
    measurements.insert(key.to_string(), bit);
    Ok(())
}

fn flipped_ids(reference: &HashMap<i64, bool>, injected: &HashMap<i64, bool>) -> Vec<i64> {
    let mut ids: Vec<i64> = reference.keys().copied().collect();
    ids.sort_unstable();
    ids.into_iter()
        .filter(|id| {
            reference.get(id).copied().unwrap_or(false) ^ injected.get(id).copied().unwrap_or(false)
        })
        .collect()
}

fn dem_edges_to_py(py: Python<'_>, edges: &[GeneratedDemEdge]) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for edge in edges {
        let dict = PyDict::new(py);
        dict.set_item("probability", edge.probability)?;
        dict.set_item("detectors", edge.detectors.clone())?;
        dict.set_item("observables", edge.observables.clone())?;
        dict.set_item("location_id", edge.location_id.clone())?;
        match &edge.event {
            DemEvent::Pauli(pauli) => dict.set_item("event", pauli)?,
            DemEvent::Bool(value) => dict.set_item("event", *value)?,
        }
        list.append(dict)?;
    }
    Ok(list.into())
}

#[derive(Clone)]
struct ConcreteStabilizer {
    x: Vec<Vec<u8>>,
    z: Vec<Vec<u8>>,
    sign: Vec<bool>,
}

impl ConcreteStabilizer {
    fn zero(n_qubits: usize) -> Self {
        let x = vec![vec![0; n_qubits]; n_qubits];
        let mut z = vec![vec![0; n_qubits]; n_qubits];
        for qubit in 0..n_qubits {
            z[qubit][qubit] = 1;
        }
        Self {
            x,
            z,
            sign: vec![false; n_qubits],
        }
    }

    fn n_qubits(&self) -> usize {
        self.x.len()
    }

    fn apply_h(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row] ^= true;
            }
            self.x[row][qubit] = old_z;
            self.z[row][qubit] = old_x;
        }
    }

    fn apply_s(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row] ^= true;
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    fn apply_s_dag(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z == 0 {
                self.sign[row] ^= true;
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    fn apply_cx(&mut self, control: usize, target: usize) {
        for row in 0..self.n_qubits() {
            let x_c = self.x[row][control];
            let z_c = self.z[row][control];
            let x_t = self.x[row][target];
            let z_t = self.z[row][target];
            if (x_t & z_c & (x_c ^ z_t ^ 1)) != 0 {
                self.sign[row] ^= true;
            }
            self.x[row][target] ^= x_c;
            self.z[row][control] ^= z_t;
        }
    }

    fn apply_cz(&mut self, left: usize, right: usize) {
        self.apply_h(right);
        self.apply_cx(left, right);
        self.apply_h(right);
    }

    fn apply_swap(&mut self, left: usize, right: usize) {
        if left == right {
            return;
        }
        self.apply_cx(left, right);
        self.apply_cx(right, left);
        self.apply_cx(left, right);
    }

    fn apply_pauli_string(&mut self, x: &[u8], z: &[u8]) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row] ^= true;
            }
        }
    }

    fn is_deterministic_pauli(&self, x: &[u8], z: &[u8]) -> bool {
        (0..self.n_qubits()).all(|row| symplectic_product(&self.x[row], &self.z[row], x, z) == 0)
    }

    fn deterministic_measurement_bit(&self, x: &[u8], z: &[u8]) -> PyResult<bool> {
        let rows: Vec<Vec<u64>> = (0..self.n_qubits())
            .map(|row| support_to_words(&self.x[row], &self.z[row]))
            .collect();
        let target = support_to_words(x, z);
        let coeff = solve_row_span(&rows, &target, self.n_qubits()).ok_or_else(|| {
            PyValueError::new_err("commuting Pauli was not in the stabilizer span")
        })?;

        let mut selected_any = false;
        let mut acc_x = vec![0; self.n_qubits()];
        let mut acc_z = vec![0; self.n_qubits()];
        let mut acc_sign = false;
        for row in 0..self.n_qubits() {
            if !coeff_bit(&coeff, row) {
                continue;
            }
            if !selected_any {
                acc_x = self.x[row].clone();
                acc_z = self.z[row].clone();
                acc_sign = self.sign[row];
                selected_any = true;
            } else {
                let (new_x, new_z, new_sign) = multiply_concrete_rows(
                    &acc_x,
                    &acc_z,
                    acc_sign,
                    &self.x[row],
                    &self.z[row],
                    self.sign[row],
                )?;
                acc_x = new_x;
                acc_z = new_z;
                acc_sign = new_sign;
            }
        }
        Ok(selected_any && acc_sign)
    }

    fn measure_pauli_with_outcome(&mut self, x: &[u8], z: &[u8], outcome: bool) -> PyResult<bool> {
        let anti: Vec<usize> = (0..self.n_qubits())
            .filter(|row| symplectic_product(&self.x[*row], &self.z[*row], x, z) != 0)
            .collect();
        if anti.is_empty() {
            return self.deterministic_measurement_bit(x, z);
        }

        let pivot = anti[0];
        let old_x = self.x[pivot].clone();
        let old_z = self.z[pivot].clone();
        let old_sign = self.sign[pivot];
        for row in anti.into_iter().skip(1) {
            let (new_x, new_z, new_sign) = multiply_concrete_rows(
                &self.x[row],
                &self.z[row],
                self.sign[row],
                &old_x,
                &old_z,
                old_sign,
            )?;
            self.x[row] = new_x;
            self.z[row] = new_z;
            self.sign[row] = new_sign;
        }
        self.x[pivot] = x.to_vec();
        self.z[pivot] = z.to_vec();
        self.sign[pivot] = outcome;
        Ok(outcome)
    }

    fn reset_prepare(&mut self, qubit: usize, basis: &str) -> PyResult<()> {
        let qubits = vec![qubit];
        let (x, z) = sparse_pauli_to_xz(self.n_qubits(), &qubits, basis)?;
        if self.is_deterministic_pauli(&x, &z) {
            let bit = self.deterministic_measurement_bit(&x, &z)?;
            if bit {
                let correction = match basis {
                    "Z" => "X",
                    "X" => "Z",
                    "Y" => "X",
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported reset basis {basis:?}"
                        )))
                    }
                };
                let (cx, cz) = sparse_pauli_to_xz(self.n_qubits(), &qubits, correction)?;
                self.apply_pauli_string(&cx, &cz);
            }
        } else {
            self.measure_pauli_with_outcome(&x, &z, false)?;
        }
        Ok(())
    }
}

fn multiply_concrete_rows(
    left_x: &[u8],
    left_z: &[u8],
    left_sign: bool,
    right_x: &[u8],
    right_z: &[u8],
    right_sign: bool,
) -> PyResult<(Vec<u8>, Vec<u8>, bool)> {
    let mut phase = 0u8;
    let mut out_x = Vec::with_capacity(left_x.len());
    let mut out_z = Vec::with_capacity(left_x.len());
    for (((lx, lz), rx), rz) in left_x.iter().zip(left_z).zip(right_x).zip(right_z) {
        let lp = xz_to_pauli(*lx, *lz);
        let rp = xz_to_pauli(*rx, *rz);
        let (local_phase, product) = pauli_product(lp, rp);
        phase = (phase + local_phase) & 3;
        let (px, pz) = pauli_to_xz(product)?;
        out_x.push(px);
        out_z.push(pz);
    }
    let mut sign = left_sign ^ right_sign;
    if phase == 2 {
        sign ^= true;
    } else if phase != 0 {
        return Err(PyValueError::new_err(
            "product of stabilizer rows produced a non-Hermitian phase",
        ));
    }
    Ok((out_x, out_z, sign))
}

fn frame_apply_h(x_frame: &mut [u8], z_frame: &mut [u8], qubit: usize) {
    std::mem::swap(&mut x_frame[qubit], &mut z_frame[qubit]);
}

fn frame_apply_s(x_frame: &mut [u8], z_frame: &mut [u8], qubit: usize) {
    z_frame[qubit] ^= x_frame[qubit];
}

fn frame_apply_cx(x_frame: &mut [u8], z_frame: &mut [u8], control: usize, target: usize) {
    x_frame[target] ^= x_frame[control];
    z_frame[control] ^= z_frame[target];
}

fn frame_apply_cz(x_frame: &mut [u8], z_frame: &mut [u8], left: usize, right: usize) {
    frame_apply_h(x_frame, z_frame, right);
    frame_apply_cx(x_frame, z_frame, left, right);
    frame_apply_h(x_frame, z_frame, right);
}

fn frame_apply_swap(x_frame: &mut [u8], z_frame: &mut [u8], left: usize, right: usize) {
    if left == right {
        return;
    }
    frame_apply_cx(x_frame, z_frame, left, right);
    frame_apply_cx(x_frame, z_frame, right, left);
    frame_apply_cx(x_frame, z_frame, left, right);
}

fn frame_apply_pauli_string(
    x_frame: &mut [u8],
    z_frame: &mut [u8],
    qubits: &[usize],
    pauli: &str,
) -> PyResult<()> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "event Pauli length does not match qubits",
        ));
    }
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        x_frame[*qubit] ^= x;
        z_frame[*qubit] ^= z;
    }
    Ok(())
}

fn frame_measurement_flip_bits(
    x_frame: &[u8],
    z_frame: &[u8],
    qubits: &[usize],
    pauli: &str,
) -> PyResult<bool> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "qubits and paulis must have the same length",
        ));
    }
    let mut x = vec![0; x_frame.len()];
    let mut z = vec![0; z_frame.len()];
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (px, pz) = pauli_to_xz(local)?;
        x[*qubit] ^= px;
        z[*qubit] ^= pz;
    }
    Ok(symplectic_product(x_frame, z_frame, &x, &z) != 0)
}

#[derive(Clone)]
struct Mask {
    words: Vec<u64>,
}

impl Mask {
    fn zero(words: usize) -> Self {
        Self {
            words: vec![0; words],
        }
    }

    fn all(shots: usize) -> Self {
        let words = word_count(shots);
        let mut mask = Self {
            words: vec![u64::MAX; words],
        };
        mask.clear_unused(shots);
        mask
    }

    fn xor_assign(&mut self, other: &Mask) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left ^= *right;
        }
    }

    fn or_assign(&mut self, other: &Mask) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left |= *right;
        }
    }

    fn and_assign(&mut self, other: &Mask) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left &= *right;
        }
    }

    fn bit_count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    fn and_count(&self, other: &Mask) -> usize {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(left, right)| (left & right).count_ones() as usize)
            .sum()
    }

    fn is_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    fn clear_unused(&mut self, shots: usize) {
        let extra = shots % 64;
        if extra != 0 {
            let keep = (1u64 << extra) - 1;
            if let Some(last) = self.words.last_mut() {
                *last &= keep;
            }
        }
    }
}

struct RuntimeState {
    shots: usize,
    all_mask: Mask,
    x_frame: Vec<Mask>,
    z_frame: Vec<Mask>,
    measurements: HashMap<String, Mask>,
    detectors: HashMap<i64, Mask>,
    observables: HashMap<i64, Mask>,
    event_masks: HashMap<String, Mask>,
    record_events: bool,
}

impl RuntimeState {
    fn new(
        n_qubits: usize,
        shots: usize,
        noise_location_ids: &[String],
        record_events: bool,
    ) -> Self {
        let words = word_count(shots);
        let all_mask = Mask::all(shots);
        let mut event_masks = HashMap::new();
        if record_events {
            for location_id in noise_location_ids {
                event_masks.insert(location_id.clone(), Mask::zero(words));
            }
        }
        Self {
            shots,
            all_mask: all_mask.clone(),
            x_frame: vec![Mask::zero(words); n_qubits],
            z_frame: vec![Mask::zero(words); n_qubits],
            measurements: HashMap::new(),
            detectors: HashMap::new(),
            observables: HashMap::new(),
            event_masks,
            record_events,
        }
    }

    fn to_py(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("shots", self.shots)?;
        out.set_item("all_mask", mask_to_py(py, &self.all_mask)?)?;
        out.set_item("x_frame", mask_vec_to_py(py, &self.x_frame)?)?;
        out.set_item("z_frame", mask_vec_to_py(py, &self.z_frame)?)?;
        out.set_item("measurements", map_to_py(py, &self.measurements)?)?;
        out.set_item("detectors", int_map_to_py(py, &self.detectors)?)?;
        out.set_item("observables", int_map_to_py(py, &self.observables)?)?;
        out.set_item("noise_event_masks", map_to_py(py, &self.event_masks)?)?;
        Ok(out.into())
    }
}

fn word_count(shots: usize) -> usize {
    (shots + 63) / 64
}

fn mask_to_py(py: Python<'_>, mask: &Mask) -> PyResult<PyObject> {
    let mut bytes = Vec::with_capacity(mask.words.len() * 8);
    for word in &mask.words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let int_type = py.import("builtins")?.getattr("int")?;
    Ok(int_type
        .call_method1("from_bytes", (PyBytes::new(py, &bytes), "little"))?
        .into())
}

fn py_int_to_mask(value: &Bound<'_, PyAny>, words: usize, shots: usize) -> PyResult<Mask> {
    let byte_len = words * 8;
    let bytes_any = value.call_method1("to_bytes", (byte_len, "little"))?;
    let bytes = bytes_any.downcast::<PyBytes>()?.as_bytes();
    let mut out = Mask::zero(words);
    for (word_index, chunk) in bytes.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        out.words[word_index] = u64::from_le_bytes(word);
    }
    out.clear_unused(shots);
    out.and_assign(&Mask::all(shots));
    Ok(out)
}

fn py_int_bit(value: &Bound<'_, PyAny>, shot: usize) -> PyResult<u8> {
    let shifted = value.call_method1("__rshift__", (shot,))?;
    shifted.call_method1("__and__", (1,))?.extract::<u8>()
}

fn mask_bit(mask: &Mask, shot: usize) -> PyResult<u8> {
    if shot >= mask.words.len() * 64 {
        return Err(PyValueError::new_err(format!(
            "shot index {shot} out of range"
        )));
    }
    Ok(((mask.words[shot / 64] >> (shot % 64)) & 1) as u8)
}

fn mask_vec_to_py(py: Python<'_>, values: &[Mask]) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for value in values {
        list.append(mask_to_py(py, value)?)?;
    }
    Ok(list.into())
}

fn map_to_py(py: Python<'_>, values: &HashMap<String, Mask>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, mask_to_py(py, value)?)?;
    }
    Ok(dict.into())
}

fn int_map_to_py(py: Python<'_>, values: &HashMap<i64, Mask>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, mask_to_py(py, value)?)?;
    }
    Ok(dict.into())
}

struct DemBatch {
    shots: usize,
    all_mask: Mask,
    detectors: HashMap<i64, Mask>,
    observables: HashMap<i64, Mask>,
    edge_event_masks: Vec<Mask>,
    loss_mask: Mask,
}

struct DemEstimate {
    shots: usize,
    mean_loss: f64,
    baseline: f64,
    edge_sensitivities: Vec<f64>,
    edge_hotspots: Vec<f64>,
    location_sensitivities: HashMap<String, f64>,
    location_hotspots: HashMap<String, f64>,
    by_detector: HashMap<i64, f64>,
    by_round: HashMap<TagValue, f64>,
    by_gate: HashMap<TagValue, f64>,
    by_operation: HashMap<TagValue, f64>,
    detector_graph: DetectorGraphEstimate,
    top_edges: Vec<usize>,
    top_locations: Vec<String>,
}

struct PackedEstimate {
    shots: usize,
    mean_loss: f64,
    baseline: f64,
    sensitivities: HashMap<String, f64>,
    hotspots: HashMap<String, f64>,
    by_qubit: HashMap<usize, f64>,
    by_round: HashMap<TagValue, f64>,
    by_gate: HashMap<TagValue, f64>,
    by_operation: HashMap<TagValue, f64>,
    top_locations: Vec<String>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DetectorGraphKey {
    detectors: Vec<i64>,
    observables: Vec<i64>,
}

struct DetectorGraphEstimate {
    by_detector_edge: HashMap<DetectorGraphKey, f64>,
    signed_by_detector_edge: HashMap<DetectorGraphKey, f64>,
    by_detector: HashMap<i64, f64>,
    signed_by_detector: HashMap<i64, f64>,
    by_observable: HashMap<i64, f64>,
    signed_by_observable: HashMap<i64, f64>,
    by_location: HashMap<String, f64>,
    signed_by_location: HashMap<String, f64>,
}

fn run_dem_batch(sampler: &NativeDemSampler, shots: usize, rng: &mut SmallRng) -> DemBatch {
    let words = word_count(shots);
    let all_mask = Mask::all(shots);
    let mut detectors = HashMap::new();
    for detector_id in &sampler.detectors {
        detectors.insert(*detector_id, Mask::zero(words));
    }
    let mut observables = HashMap::new();
    for observable_id in &sampler.observables {
        observables.insert(*observable_id, Mask::zero(words));
    }
    let mut edge_event_masks = Vec::with_capacity(sampler.edges.len());

    for edge in &sampler.edges {
        let event_mask = bernoulli_mask(rng, shots, edge.probability);
        if !event_mask.is_zero() {
            for detector_id in &edge.detectors {
                detectors
                    .entry(*detector_id)
                    .or_insert_with(|| Mask::zero(words))
                    .xor_assign(&event_mask);
            }
            for observable_id in &edge.observables {
                observables
                    .entry(*observable_id)
                    .or_insert_with(|| Mask::zero(words))
                    .xor_assign(&event_mask);
            }
        }
        edge_event_masks.push(event_mask);
    }

    let mut loss_mask = Mask::zero(words);
    for observable in observables.values() {
        loss_mask.or_assign(observable);
    }
    loss_mask.and_assign(&all_mask);

    DemBatch {
        shots,
        all_mask,
        detectors,
        observables,
        edge_event_masks,
        loss_mask,
    }
}

fn dem_batch_to_py(py: Python<'_>, batch: &DemBatch) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("shots", batch.shots)?;
    out.set_item("all_mask", mask_to_py(py, &batch.all_mask)?)?;
    out.set_item("detectors", int_map_to_py(py, &batch.detectors)?)?;
    out.set_item("observables", int_map_to_py(py, &batch.observables)?)?;
    let edge_masks = PyDict::new(py);
    for (edge_index, mask) in batch.edge_event_masks.iter().enumerate() {
        edge_masks.set_item(edge_index, mask_to_py(py, mask)?)?;
    }
    out.set_item("edge_event_masks", edge_masks)?;
    Ok(out.into())
}

fn compute_packed_estimate(
    sampler: &NativePackedSampler,
    state: &RuntimeState,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> PackedEstimate {
    let mut clipped_loss = loss_mask.clone();
    clipped_loss.and_assign(&state.all_mask);
    let loss_count = clipped_loss.bit_count();
    let mean_loss = loss_count as f64 / state.shots as f64;
    let baseline_value = baseline.unwrap_or(mean_loss);
    let mut sensitivities = HashMap::new();
    let mut hotspots = HashMap::new();
    let mut by_qubit = HashMap::new();

    for location in &sampler.noise_locations {
        let zero_mask;
        let event_mask = if let Some(mask) = state.event_masks.get(&location.id) {
            mask
        } else {
            zero_mask = Mask::zero(state.all_mask.words.len());
            &zero_mask
        };
        let event_count = event_mask.bit_count();
        let no_event_count = state.shots - event_count;
        let loss_event_count = clipped_loss.and_count(event_mask);
        let loss_no_event_count = loss_count - loss_event_count;
        let (event_score, no_event_score) = score_pair(location.rate);
        let sum_loss_score =
            loss_event_count as f64 * event_score + loss_no_event_count as f64 * no_event_score;
        let sum_score = event_count as f64 * event_score + no_event_count as f64 * no_event_score;
        let sensitivity = (sum_loss_score - baseline_value * sum_score) / state.shots as f64;
        let hotspot = sensitivity.abs();
        sensitivities.insert(location.id.clone(), sensitivity);
        hotspots.insert(location.id.clone(), hotspot);
        for qubit in &location.qubits {
            add_f64(&mut by_qubit, *qubit, hotspot);
        }
    }

    let by_round = aggregate_location_tag_hotspots(&sampler.noise_locations, &hotspots, "round");
    let by_gate = aggregate_location_tag_hotspots(&sampler.noise_locations, &hotspots, "gate");
    let by_operation =
        aggregate_location_tag_hotspots(&sampler.noise_locations, &hotspots, "operation");
    let top_locations = top_location_ids(&hotspots, top_k);

    PackedEstimate {
        shots: state.shots,
        mean_loss,
        baseline: baseline_value,
        sensitivities,
        hotspots,
        by_qubit,
        by_round,
        by_gate,
        by_operation,
        top_locations,
    }
}

fn compute_dem_estimate(
    sampler: &NativeDemSampler,
    batch: &DemBatch,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> DemEstimate {
    let mut clipped_loss = loss_mask.clone();
    clipped_loss.and_assign(&batch.all_mask);
    let loss_count = clipped_loss.bit_count();
    let mean_loss = loss_count as f64 / batch.shots as f64;
    let baseline_value = baseline.unwrap_or(mean_loss);
    let mut edge_sensitivities = Vec::with_capacity(sampler.edges.len());
    for (edge_index, edge) in sampler.edges.iter().enumerate() {
        let event_mask = &batch.edge_event_masks[edge_index];
        let event_count = event_mask.bit_count();
        let no_event_count = batch.shots - event_count;
        let loss_event_count = clipped_loss.and_count(event_mask);
        let loss_no_event_count = loss_count - loss_event_count;
        let (event_score, no_event_score) = score_pair(edge.probability);
        let sum_loss_score =
            loss_event_count as f64 * event_score + loss_no_event_count as f64 * no_event_score;
        let sum_score = event_count as f64 * event_score + no_event_count as f64 * no_event_score;
        edge_sensitivities.push((sum_loss_score - baseline_value * sum_score) / batch.shots as f64);
    }

    let location_sensitivities = aggregate_dem_location_sensitivities(sampler, &edge_sensitivities);
    let edge_hotspots = edge_sensitivities
        .iter()
        .map(|sensitivity| sensitivity.abs())
        .collect::<Vec<_>>();
    let location_hotspots = location_sensitivities
        .iter()
        .map(|(location_id, sensitivity)| (location_id.clone(), sensitivity.abs()))
        .collect::<HashMap<_, _>>();
    let by_detector = aggregate_dem_detector_hotspots(sampler, &edge_hotspots);
    let by_round = aggregate_dem_tag_hotspots(sampler, &location_hotspots, "round");
    let by_gate = aggregate_dem_tag_hotspots(sampler, &location_hotspots, "gate");
    let by_operation = aggregate_dem_tag_hotspots(sampler, &location_hotspots, "operation");
    let detector_graph = compute_detector_graph_estimate(sampler, &edge_sensitivities);
    let top_edges = top_edge_indices(&edge_hotspots, top_k);
    let top_locations = top_location_ids(&location_hotspots, top_k);

    DemEstimate {
        shots: batch.shots,
        mean_loss,
        baseline: baseline_value,
        edge_sensitivities,
        edge_hotspots,
        location_sensitivities,
        location_hotspots,
        by_detector,
        by_round,
        by_gate,
        by_operation,
        detector_graph,
        top_edges,
        top_locations,
    }
}

fn packed_estimate_to_py(py: Python<'_>, estimate: &PackedEstimate) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("shots", estimate.shots)?;
    out.set_item("mean_loss", estimate.mean_loss)?;
    out.set_item("baseline", estimate.baseline)?;
    out.set_item(
        "sensitivities",
        string_f64_map_to_py(py, &estimate.sensitivities)?,
    )?;
    out.set_item("hotspots", string_f64_map_to_py(py, &estimate.hotspots)?)?;
    out.set_item("by_qubit", usize_f64_map_to_py(py, &estimate.by_qubit)?)?;
    out.set_item("by_round", tag_f64_map_to_py(py, &estimate.by_round)?)?;
    out.set_item("by_gate", tag_f64_map_to_py(py, &estimate.by_gate)?)?;
    out.set_item(
        "by_operation",
        tag_f64_map_to_py(py, &estimate.by_operation)?,
    )?;
    out.set_item(
        "top_hotspots",
        packed_top_locations_to_py(
            py,
            &estimate.top_locations,
            &estimate.sensitivities,
            &estimate.hotspots,
        )?,
    )?;
    Ok(out.into())
}

fn dem_estimate_to_py(py: Python<'_>, estimate: &DemEstimate) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("shots", estimate.shots)?;
    out.set_item("mean_loss", estimate.mean_loss)?;
    out.set_item("baseline", estimate.baseline)?;
    let edge_dict = PyDict::new(py);
    for (edge_index, sensitivity) in estimate.edge_sensitivities.iter().enumerate() {
        edge_dict.set_item(edge_index, *sensitivity)?;
    }
    out.set_item("edge_sensitivities", edge_dict)?;
    let edge_hotspots = PyDict::new(py);
    for (edge_index, hotspot) in estimate.edge_hotspots.iter().enumerate() {
        edge_hotspots.set_item(edge_index, *hotspot)?;
    }
    out.set_item("edge_hotspots", edge_hotspots)?;
    out.set_item(
        "sensitivities",
        string_f64_map_to_py(py, &estimate.location_sensitivities)?,
    )?;
    out.set_item(
        "hotspots",
        string_f64_map_to_py(py, &estimate.location_hotspots)?,
    )?;
    out.set_item("by_detector", i64_f64_map_to_py(py, &estimate.by_detector)?)?;
    out.set_item("by_round", tag_f64_map_to_py(py, &estimate.by_round)?)?;
    out.set_item("by_gate", tag_f64_map_to_py(py, &estimate.by_gate)?)?;
    out.set_item(
        "by_operation",
        tag_f64_map_to_py(py, &estimate.by_operation)?,
    )?;
    out.set_item(
        "detector_graph_hotspots",
        detector_graph_to_py(
            py,
            &estimate.detector_graph,
            &estimate.edge_sensitivities,
            &estimate.edge_hotspots,
        )?,
    )?;
    out.set_item(
        "top_edges",
        dem_top_edges_to_py(
            py,
            &estimate.top_edges,
            &estimate.edge_sensitivities,
            &estimate.edge_hotspots,
        )?,
    )?;
    out.set_item(
        "top_hotspots",
        packed_top_locations_to_py(
            py,
            &estimate.top_locations,
            &estimate.location_sensitivities,
            &estimate.location_hotspots,
        )?,
    )?;
    Ok(out.into())
}

fn string_f64_map_to_py(py: Python<'_>, values: &HashMap<String, f64>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, *value)?;
    }
    Ok(dict.into())
}

fn usize_f64_map_to_py(py: Python<'_>, values: &HashMap<usize, f64>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(*key, *value)?;
    }
    Ok(dict.into())
}

fn i64_f64_map_to_py(py: Python<'_>, values: &HashMap<i64, f64>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(*key, *value)?;
    }
    Ok(dict.into())
}

fn tag_f64_map_to_py(py: Python<'_>, values: &HashMap<TagValue, f64>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(tag_value_to_py(py, key)?, *value)?;
    }
    Ok(dict.into())
}

fn tag_value_to_py(py: Python<'_>, value: &TagValue) -> PyResult<PyObject> {
    match value {
        TagValue::None => Ok(py.None()),
        TagValue::Bool(value) => Ok(value.into_py(py)),
        TagValue::Int(value) => Ok(value.into_py(py)),
        TagValue::Float(bits) => Ok(f64::from_bits(*bits).into_py(py)),
        TagValue::String(value) => Ok(value.into_py(py)),
    }
}

fn packed_top_locations_to_py(
    py: Python<'_>,
    location_ids: &[String],
    sensitivities: &HashMap<String, f64>,
    hotspots: &HashMap<String, f64>,
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for location_id in location_ids {
        let row = PyDict::new(py);
        row.set_item("location_id", location_id)?;
        row.set_item(
            "sensitivity",
            *sensitivities.get(location_id).unwrap_or(&0.0),
        )?;
        row.set_item("hotspot", *hotspots.get(location_id).unwrap_or(&0.0))?;
        list.append(row)?;
    }
    Ok(list.into())
}

fn dem_top_edges_to_py(
    py: Python<'_>,
    edge_indices: &[usize],
    sensitivities: &[f64],
    hotspots: &[f64],
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for edge_index in edge_indices {
        let row = PyDict::new(py);
        row.set_item("edge_index", *edge_index)?;
        row.set_item(
            "sensitivity",
            sensitivities.get(*edge_index).copied().unwrap_or(0.0),
        )?;
        row.set_item("hotspot", hotspots.get(*edge_index).copied().unwrap_or(0.0))?;
        list.append(row)?;
    }
    Ok(list.into())
}

fn detector_graph_to_py(
    py: Python<'_>,
    graph: &DetectorGraphEstimate,
    sensitivities: &[f64],
    hotspots: &[f64],
) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    let edge_rows = PyList::empty(py);
    for (edge_index, sensitivity) in sensitivities.iter().enumerate() {
        let row = PyDict::new(py);
        row.set_item("edge_index", edge_index)?;
        row.set_item("sensitivity", *sensitivity)?;
        row.set_item("hotspot", hotspots.get(edge_index).copied().unwrap_or(0.0))?;
        edge_rows.append(row)?;
    }
    out.set_item("edge_hotspots", edge_rows)?;
    out.set_item(
        "by_detector_edge",
        graph_key_f64_map_to_py(py, &graph.by_detector_edge)?,
    )?;
    out.set_item(
        "signed_by_detector_edge",
        graph_key_f64_map_to_py(py, &graph.signed_by_detector_edge)?,
    )?;
    out.set_item("by_detector", i64_f64_map_to_py(py, &graph.by_detector)?)?;
    out.set_item(
        "signed_by_detector",
        i64_f64_map_to_py(py, &graph.signed_by_detector)?,
    )?;
    out.set_item(
        "by_observable",
        i64_f64_map_to_py(py, &graph.by_observable)?,
    )?;
    out.set_item(
        "signed_by_observable",
        i64_f64_map_to_py(py, &graph.signed_by_observable)?,
    )?;
    out.set_item("by_location", string_f64_map_to_py(py, &graph.by_location)?)?;
    out.set_item(
        "signed_by_location",
        string_f64_map_to_py(py, &graph.signed_by_location)?,
    )?;
    Ok(out.into())
}

fn graph_key_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<DetectorGraphKey, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        let detectors = PyTuple::new(py, key.detectors.iter().copied())?;
        let observables = PyTuple::new(py, key.observables.iter().copied())?;
        let detectors_obj: PyObject = detectors.into_py(py);
        let observables_obj: PyObject = observables.into_py(py);
        let graph_key = PyTuple::new(py, [detectors_obj, observables_obj])?;
        dict.set_item(graph_key, *value)?;
    }
    Ok(dict.into())
}

fn aggregate_dem_location_sensitivities(
    sampler: &NativeDemSampler,
    edge_sensitivities: &[f64],
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for group in &sampler.location_groups {
        let mut value = 0.0;
        for edge_index in &group.edge_indices {
            let weight = if group.total_probability > 0.0 {
                sampler.edges[*edge_index].probability / group.total_probability
            } else {
                1.0 / group.edge_indices.len() as f64
            };
            value += edge_sensitivities[*edge_index] * weight;
        }
        out.insert(group.location_id.clone(), value);
    }
    out
}

fn build_dem_location_groups(edges: &[DemEdgeSpec]) -> Vec<DemLocationGroup> {
    let mut group_indices = HashMap::<String, usize>::new();
    let mut groups = Vec::<DemLocationGroup>::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        let group_index = if let Some(group_index) = group_indices.get(&edge.location_id) {
            *group_index
        } else {
            let group_index = groups.len();
            group_indices.insert(edge.location_id.clone(), group_index);
            groups.push(DemLocationGroup {
                location_id: edge.location_id.clone(),
                edge_indices: Vec::new(),
                total_probability: 0.0,
            });
            group_index
        };
        let group = &mut groups[group_index];
        group.edge_indices.push(edge_index);
        group.total_probability += edge.probability;
    }
    groups
}

fn aggregate_location_tag_hotspots(
    locations: &[NoiseLocationSpec],
    hotspots: &HashMap<String, f64>,
    tag: &str,
) -> HashMap<TagValue, f64> {
    let mut out = HashMap::new();
    for location in locations {
        let Some(tag_value) = location.tags.get(tag) else {
            continue;
        };
        let hotspot = *hotspots.get(&location.id).unwrap_or(&0.0);
        add_f64(&mut out, tag_value.clone(), hotspot);
    }
    out
}

fn aggregate_dem_tag_hotspots(
    sampler: &NativeDemSampler,
    hotspots: &HashMap<String, f64>,
    tag: &str,
) -> HashMap<TagValue, f64> {
    let mut by_location: HashMap<String, &HashMap<String, TagValue>> = HashMap::new();
    for edge in &sampler.edges {
        by_location
            .entry(edge.location_id.clone())
            .or_insert(&edge.tags);
    }
    let mut out = HashMap::new();
    for (location_id, tags) in by_location {
        let Some(tag_value) = tags.get(tag) else {
            continue;
        };
        let hotspot = *hotspots.get(&location_id).unwrap_or(&0.0);
        add_f64(&mut out, tag_value.clone(), hotspot);
    }
    out
}

fn aggregate_dem_detector_hotspots(
    sampler: &NativeDemSampler,
    edge_hotspots: &[f64],
) -> HashMap<i64, f64> {
    let mut out = HashMap::new();
    for (edge_index, edge) in sampler.edges.iter().enumerate() {
        let hotspot = edge_hotspots.get(edge_index).copied().unwrap_or(0.0);
        if hotspot == 0.0 || edge.detectors.is_empty() {
            continue;
        }
        let share = hotspot / edge.detectors.len() as f64;
        for detector_id in &edge.detectors {
            add_f64(&mut out, *detector_id, share);
        }
    }
    out
}

fn compute_detector_graph_estimate(
    sampler: &NativeDemSampler,
    edge_sensitivities: &[f64],
) -> DetectorGraphEstimate {
    let mut graph = DetectorGraphEstimate {
        by_detector_edge: HashMap::new(),
        signed_by_detector_edge: HashMap::new(),
        by_detector: HashMap::new(),
        signed_by_detector: HashMap::new(),
        by_observable: HashMap::new(),
        signed_by_observable: HashMap::new(),
        by_location: HashMap::new(),
        signed_by_location: HashMap::new(),
    };
    for (edge_index, edge) in sampler.edges.iter().enumerate() {
        let sensitivity = edge_sensitivities.get(edge_index).copied().unwrap_or(0.0);
        let hotspot = sensitivity.abs();
        if hotspot == 0.0 {
            continue;
        }
        let key = DetectorGraphKey {
            detectors: edge.detectors.clone(),
            observables: edge.observables.clone(),
        };
        add_f64(&mut graph.by_detector_edge, key.clone(), hotspot);
        add_f64(&mut graph.signed_by_detector_edge, key, sensitivity);
        add_f64(&mut graph.by_location, edge.location_id.clone(), hotspot);
        add_f64(
            &mut graph.signed_by_location,
            edge.location_id.clone(),
            sensitivity,
        );
        if !edge.detectors.is_empty() {
            let share = hotspot / edge.detectors.len() as f64;
            let signed_share = sensitivity / edge.detectors.len() as f64;
            for detector_id in &edge.detectors {
                add_f64(&mut graph.by_detector, *detector_id, share);
                add_f64(&mut graph.signed_by_detector, *detector_id, signed_share);
            }
        }
        if !edge.observables.is_empty() {
            let share = hotspot / edge.observables.len() as f64;
            let signed_share = sensitivity / edge.observables.len() as f64;
            for observable_id in &edge.observables {
                add_f64(&mut graph.by_observable, *observable_id, share);
                add_f64(
                    &mut graph.signed_by_observable,
                    *observable_id,
                    signed_share,
                );
            }
        }
    }
    graph
}

fn top_location_ids(hotspots: &HashMap<String, f64>, top_k: usize) -> Vec<String> {
    if top_k == 0 || hotspots.is_empty() {
        return Vec::new();
    }
    let limit = top_k.min(hotspots.len());
    let mut rows = hotspots
        .iter()
        .map(|(location_id, hotspot)| (location_id, *hotspot))
        .collect::<Vec<_>>();
    let compare = |left: &(&String, f64), right: &(&String, f64)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if limit < rows.len() {
        rows.select_nth_unstable_by(limit, compare);
    }
    rows.truncate(limit);
    rows.sort_by(compare);
    rows.into_iter()
        .map(|(location_id, _)| location_id.clone())
        .collect()
}

fn top_edge_indices(hotspots: &[f64], top_k: usize) -> Vec<usize> {
    if top_k == 0 || hotspots.is_empty() {
        return Vec::new();
    }
    let limit = top_k.min(hotspots.len());
    let mut rows = hotspots.iter().copied().enumerate().collect::<Vec<_>>();
    let compare = |left: &(usize, f64), right: &(usize, f64)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if limit < rows.len() {
        rows.select_nth_unstable_by(limit, compare);
    }
    rows.truncate(limit);
    rows.sort_by(compare);
    rows.into_iter().map(|(edge_index, _)| edge_index).collect()
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

fn score_pair(probability: f64) -> (f64, f64) {
    let p = probability.clamp(1e-12, 1.0 - 1e-12);
    (1.0 / p, -1.0 / (1.0 - p))
}

fn run_packed_sample(
    sampler: &NativePackedSampler,
    shots: usize,
    seed: Option<u64>,
    record_events: bool,
) -> PyResult<RuntimeState> {
    let mut rng = SmallRng::new(seed.unwrap_or(0x4d59_5df4_d0f3_3173));
    let mut state = RuntimeState::new(
        sampler.n_qubits,
        shots,
        &sampler.noise_location_ids,
        record_events,
    );
    let random_masks = (0..sampler.random_source_count)
        .map(|_| random_bit_mask(&mut rng, state.all_mask.words.len(), state.shots))
        .collect::<Vec<_>>();
    for operation in &sampler.runtime_operations {
        apply_operation(operation, &mut state, &random_masks, &mut rng)?;
    }
    Ok(state)
}

fn apply_operation(
    op: &RunOp,
    state: &mut RuntimeState,
    random_masks: &[Mask],
    rng: &mut SmallRng,
) -> PyResult<()> {
    match op {
        RunOp::H(q) => {
            let old_x = state.x_frame[*q].clone();
            state.x_frame[*q] = state.z_frame[*q].clone();
            state.z_frame[*q] = old_x;
        }
        RunOp::S(q) | RunOp::SDag(q) => {
            let x = state.x_frame[*q].clone();
            state.z_frame[*q].xor_assign(&x);
        }
        RunOp::Cx(control, target) => {
            let x_control = state.x_frame[*control].clone();
            state.x_frame[*target].xor_assign(&x_control);
            let z_target = state.z_frame[*target].clone();
            state.z_frame[*control].xor_assign(&z_target);
        }
        RunOp::Cz(left, right) => {
            let x_right = state.x_frame[*right].clone();
            let x_left = state.x_frame[*left].clone();
            state.z_frame[*left].xor_assign(&x_right);
            state.z_frame[*right].xor_assign(&x_left);
        }
        RunOp::Swap(left, right) => {
            state.x_frame.swap(*left, *right);
            state.z_frame.swap(*left, *right);
        }
        RunOp::Noise(location) => {
            sample_noise(location, state, rng)?;
        }
        RunOp::Measure {
            qubits,
            key,
            pauli,
            ideal,
            noise,
        } => {
            let mut bit = ideal.eval(random_masks, state.all_mask.words.len(), &state.all_mask);
            let flip = frame_measurement_flip(state, qubits, pauli)?;
            bit.xor_assign(&flip);
            if let Some(location) = noise {
                let flip = sample_measurement_noise(location, state, rng)?;
                bit.xor_assign(&flip);
            }
            let key = key
                .clone()
                .unwrap_or_else(|| format!("m{}", state.measurements.len()));
            record_measurement(&mut state.measurements, &key, bit)?;
        }
        RunOp::Reset {
            qubit,
            key,
            basis,
            ideal,
        } => {
            let outcome = ideal.eval(random_masks, state.all_mask.words.len(), &state.all_mask);
            if let Some(key) = key {
                let mut bit = outcome.clone();
                let flip = frame_measurement_flip(state, &[*qubit], basis)?;
                bit.xor_assign(&flip);
                record_measurement(&mut state.measurements, key, bit)?;
            }
            state.x_frame[*qubit] = Mask::zero(state.all_mask.words.len());
            state.z_frame[*qubit] = Mask::zero(state.all_mask.words.len());
        }
        RunOp::Detector {
            detector_id,
            measurement_keys,
        } => {
            let id = if *detector_id >= 0 {
                *detector_id
            } else {
                state.detectors.len() as i64
            };
            let value = measurement_parity(&state.measurements, measurement_keys)?;
            state.detectors.insert(id, value);
        }
        RunOp::ObservableInclude {
            observable_id,
            measurement_keys,
        } => {
            let value = measurement_parity(&state.measurements, measurement_keys)?;
            state
                .observables
                .entry(*observable_id)
                .and_modify(|existing| existing.xor_assign(&value))
                .or_insert(value);
        }
    }
    Ok(())
}

fn sample_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<()> {
    match &location.model {
        NoiseModel::MeasurementBitFlip => Err(PyValueError::new_err(
            "MeasurementBitFlip must be attached to a measurement operation",
        )),
        NoiseModel::BernoulliPauli(pauli) => sample_fixed_pauli_noise(location, state, rng, pauli),
        NoiseModel::SingleQubitDepolarizing => {
            sample_single_qubit_depolarizing_noise(location, state, rng)
        }
        NoiseModel::TwoQubitDepolarizing => {
            sample_two_qubit_depolarizing_noise(location, state, rng)
        }
        NoiseModel::PauliChannel(weights) => {
            sample_pauli_channel_noise(location, state, rng, weights)
        }
    }
}

fn sample_measurement_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<Mask> {
    if !matches!(location.model, NoiseModel::MeasurementBitFlip) {
        return Err(PyValueError::new_err(
            "native measurement noise supports MeasurementBitFlip only",
        ));
    }
    let flip = bernoulli_mask(rng, state.shots, location.rate);
    if state.record_events {
        if let Some(mask) = state.event_masks.get_mut(&location.id) {
            mask.xor_assign(&flip);
        }
    }
    Ok(flip)
}

fn sample_fixed_pauli_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    pauli: &str,
) -> PyResult<()> {
    let event_mask = bernoulli_mask(rng, state.shots, location.rate);
    if state.record_events {
        record_location_event_mask(state, &location.id, &event_mask);
    }
    apply_masked_pauli_to_frame(state, &location.qubits, pauli, &event_mask)
}

fn sample_single_qubit_depolarizing_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<()> {
    let qubit = one_qubit(&location.qubits, "SingleQubitDepolarizing")?;
    let mut event_mask = event_union_mask(state);
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, rng| {
        if let Some(mask) = &mut event_mask {
            set_shot_bit(mask, shot);
        }
        match rng.next_u64() % 3 {
            0 => xor_frame_shot(state, qubit, true, false, shot),
            1 => xor_frame_shot(state, qubit, true, true, shot),
            _ => xor_frame_shot(state, qubit, false, true, shot),
        }
    });
    if let Some(mask) = event_mask {
        record_location_event_mask(state, &location.id, &mask);
    }
    Ok(())
}

fn sample_two_qubit_depolarizing_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
) -> PyResult<()> {
    let (left, right) = two_qubits(&location.qubits, "TwoQubitDepolarizing")?;
    let mut event_mask = event_union_mask(state);
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, rng| {
        if let Some(mask) = &mut event_mask {
            set_shot_bit(mask, shot);
        }
        let (left_x, left_z, right_x, right_z) = TWO_QUBIT_DEPOLARIZING_EVENTS
            [(rng.next_u64() % TWO_QUBIT_DEPOLARIZING_EVENTS.len() as u64) as usize];
        xor_frame_shot(state, left, left_x, left_z, shot);
        xor_frame_shot(state, right, right_x, right_z, shot);
    });
    if let Some(mask) = event_mask {
        record_location_event_mask(state, &location.id, &mask);
    }
    Ok(())
}

fn sample_pauli_channel_noise(
    location: &NoiseLocationSpec,
    state: &mut RuntimeState,
    rng: &mut SmallRng,
    weights: &[(String, f64)],
) -> PyResult<()> {
    let events = compile_pauli_channel_events(&location.qubits, weights)?;
    let total: f64 = events.iter().map(|event| event.weight).sum();
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "PauliChannel weights must have positive total weight",
        ));
    }
    let mut event_mask = event_union_mask(state);
    for_each_bernoulli_event(rng, state.shots, location.rate, |shot, rng| {
        if let Some(mask) = &mut event_mask {
            set_shot_bit(mask, shot);
        }
        let event_index = choose_weighted_event(&events, total, rng);
        apply_compiled_pauli_event(state, &location.qubits, &events[event_index], shot);
    });
    if let Some(mask) = event_mask {
        record_location_event_mask(state, &location.id, &mask);
    }
    Ok(())
}

fn bernoulli_mask(rng: &mut SmallRng, shots: usize, rate: f64) -> Mask {
    if rate <= 0.0 {
        return Mask::zero(word_count(shots));
    }
    if rate >= 1.0 {
        return Mask::all(shots);
    }
    let mut mask = Mask::zero(word_count(shots));
    for_each_bernoulli_event(rng, shots, rate, |shot, _| set_shot_bit(&mut mask, shot));
    mask
}

const TWO_QUBIT_DEPOLARIZING_EVENTS: [(bool, bool, bool, bool); 15] = [
    (false, false, true, false),
    (false, false, true, true),
    (false, false, false, true),
    (true, false, false, false),
    (true, false, true, false),
    (true, false, true, true),
    (true, false, false, true),
    (true, true, false, false),
    (true, true, true, false),
    (true, true, true, true),
    (true, true, false, true),
    (false, true, false, false),
    (false, true, true, false),
    (false, true, true, true),
    (false, true, false, true),
];

struct CompiledPauliEvent {
    x: Vec<bool>,
    z: Vec<bool>,
    weight: f64,
}

fn for_each_bernoulli_event<F>(rng: &mut SmallRng, shots: usize, rate: f64, mut visit: F)
where
    F: FnMut(usize, &mut SmallRng),
{
    if rate <= 0.0 {
        return;
    }
    if rate >= 1.0 {
        for shot in 0..shots {
            visit(shot, rng);
        }
        return;
    }
    let log1mp = (-rate).ln_1p();
    let mut shot = 0usize;
    loop {
        let skip = (positive_unit_f64(rng).ln() / log1mp).floor() as usize;
        match shot.checked_add(skip) {
            Some(next) if next < shots => shot = next,
            _ => break,
        }
        visit(shot, rng);
        shot += 1;
        if shot >= shots {
            break;
        }
    }
}

fn positive_unit_f64(rng: &mut SmallRng) -> f64 {
    rng.next_f64().max(f64::MIN_POSITIVE)
}

fn event_union_mask(state: &RuntimeState) -> Option<Mask> {
    if state.record_events {
        Some(Mask::zero(state.all_mask.words.len()))
    } else {
        None
    }
}

fn record_location_event_mask(state: &mut RuntimeState, location_id: &str, event_mask: &Mask) {
    if state.record_events {
        if let Some(mask) = state.event_masks.get_mut(location_id) {
            mask.or_assign(event_mask);
        }
    }
}

fn xor_frame_shot(state: &mut RuntimeState, qubit: usize, x: bool, z: bool, shot: usize) {
    let word = shot / 64;
    let bit = 1u64 << (shot % 64);
    if x {
        state.x_frame[qubit].words[word] ^= bit;
    }
    if z {
        state.z_frame[qubit].words[word] ^= bit;
    }
}

fn compile_pauli_channel_events(
    qubits: &[usize],
    weights: &[(String, f64)],
) -> PyResult<Vec<CompiledPauliEvent>> {
    let mut events = Vec::new();
    for (event, weight) in weights {
        if *weight < 0.0 {
            return Err(PyValueError::new_err(
                "PauliChannel weights must be non-negative",
            ));
        }
        if *weight == 0.0 {
            continue;
        }
        if event.len() != qubits.len() {
            return Err(PyValueError::new_err(
                "PauliChannel event length does not match qubits",
            ));
        }
        let mut x = Vec::with_capacity(event.len());
        let mut z = Vec::with_capacity(event.len());
        for local in event.chars() {
            let (px, pz) = pauli_to_xz(local)?;
            x.push(px != 0);
            z.push(pz != 0);
        }
        events.push(CompiledPauliEvent {
            x,
            z,
            weight: *weight,
        });
    }
    Ok(events)
}

fn choose_weighted_event(events: &[CompiledPauliEvent], total: f64, rng: &mut SmallRng) -> usize {
    let mut threshold = rng.next_f64() * total;
    for (idx, event) in events.iter().enumerate() {
        if threshold < event.weight {
            return idx;
        }
        threshold -= event.weight;
    }
    events.len() - 1
}

fn apply_compiled_pauli_event(
    state: &mut RuntimeState,
    qubits: &[usize],
    event: &CompiledPauliEvent,
    shot: usize,
) {
    for (local_index, qubit) in qubits.iter().enumerate() {
        xor_frame_shot(
            state,
            *qubit,
            event.x[local_index],
            event.z[local_index],
            shot,
        );
    }
}

fn random_bit_mask(rng: &mut SmallRng, words: usize, shots: usize) -> Mask {
    let mut mask = Mask::zero(words);
    for word in &mut mask.words {
        *word = rng.next_u64();
    }
    mask.clear_unused(shots);
    mask
}

fn set_shot_bit(mask: &mut Mask, shot: usize) {
    mask.words[shot / 64] |= 1u64 << (shot % 64);
}

fn apply_masked_pauli_to_frame(
    state: &mut RuntimeState,
    qubits: &[usize],
    pauli: &str,
    mask: &Mask,
) -> PyResult<()> {
    if mask.is_zero() {
        return Ok(());
    }
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "event Pauli length does not match qubits",
        ));
    }
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        if x != 0 {
            state.x_frame[*qubit].xor_assign(mask);
        }
        if z != 0 {
            state.z_frame[*qubit].xor_assign(mask);
        }
    }
    Ok(())
}

fn frame_measurement_flip(state: &RuntimeState, qubits: &[usize], pauli: &str) -> PyResult<Mask> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "qubits and pauli must have the same length",
        ));
    }
    let mut flip = Mask::zero(state.all_mask.words.len());
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = pauli_to_xz(local)?;
        if z != 0 {
            flip.xor_assign(&state.x_frame[*qubit]);
        }
        if x != 0 {
            flip.xor_assign(&state.z_frame[*qubit]);
        }
    }
    Ok(flip)
}

fn record_measurement(
    measurements: &mut HashMap<String, Mask>,
    key: &str,
    bit: Mask,
) -> PyResult<()> {
    if measurements.contains_key(key) {
        return Err(PyValueError::new_err(format!(
            "duplicate measurement key {key:?}"
        )));
    }
    measurements.insert(key.to_string(), bit);
    Ok(())
}

fn measurement_parity(measurements: &HashMap<String, Mask>, keys: &[String]) -> PyResult<Mask> {
    let words = measurements
        .values()
        .next()
        .map(|mask| mask.words.len())
        .unwrap_or(1);
    let mut parity = Mask::zero(words);
    for key in keys {
        let value = measurements
            .get(key)
            .ok_or_else(|| PyValueError::new_err(format!("unknown measurement key {key:?}")))?;
        parity.xor_assign(value);
    }
    Ok(parity)
}

fn sparse_pauli_to_xz(
    n_qubits: usize,
    qubits: &[usize],
    pauli: &str,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    if qubits.len() != pauli.len() {
        return Err(PyValueError::new_err(
            "qubits and paulis must have the same length",
        ));
    }
    let mut x = vec![0; n_qubits];
    let mut z = vec![0; n_qubits];
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (px, pz) = pauli_to_xz(local)?;
        x[*qubit] ^= px;
        z[*qubit] ^= pz;
    }
    Ok((x, z))
}

fn pauli_to_xz(pauli: char) -> PyResult<(u8, u8)> {
    match pauli {
        'I' => Ok((0, 0)),
        'X' => Ok((1, 0)),
        'Y' => Ok((1, 1)),
        'Z' => Ok((0, 1)),
        _ => Err(PyValueError::new_err(format!(
            "unsupported Pauli {pauli:?}"
        ))),
    }
}

fn xz_to_pauli(x: u8, z: u8) -> char {
    match (x & 1, z & 1) {
        (0, 0) => 'I',
        (1, 0) => 'X',
        (1, 1) => 'Y',
        (0, 1) => 'Z',
        _ => unreachable!(),
    }
}

fn symplectic_product(x1: &[u8], z1: &[u8], x2: &[u8], z2: &[u8]) -> u8 {
    let mut acc = 0;
    for (((a_x, a_z), b_x), b_z) in x1.iter().zip(z1).zip(x2).zip(z2) {
        acc ^= (a_x & b_z) ^ (a_z & b_x);
    }
    acc & 1
}

fn multiply_symbolic_rows(
    left_x: &[u8],
    left_z: &[u8],
    left_sign: &Expr,
    right_x: &[u8],
    right_z: &[u8],
    right_sign: &Expr,
) -> PyResult<(Vec<u8>, Vec<u8>, Expr)> {
    let mut phase = 0u8;
    let mut out_x = Vec::with_capacity(left_x.len());
    let mut out_z = Vec::with_capacity(left_x.len());
    for (((lx, lz), rx), rz) in left_x.iter().zip(left_z).zip(right_x).zip(right_z) {
        let lp = xz_to_pauli(*lx, *lz);
        let rp = xz_to_pauli(*rx, *rz);
        let (local_phase, product) = pauli_product(lp, rp);
        phase = (phase + local_phase) & 3;
        let (px, pz) = pauli_to_xz(product)?;
        out_x.push(px);
        out_z.push(pz);
    }
    let mut out_sign = left_sign.clone();
    out_sign.xor_assign(right_sign);
    if phase == 2 {
        out_sign.toggle_constant();
    } else if phase != 0 {
        return Err(PyValueError::new_err(
            "product of stabilizer rows produced a non-Hermitian phase",
        ));
    }
    Ok((out_x, out_z, out_sign))
}

fn pauli_product(left: char, right: char) -> (u8, char) {
    match (left, right) {
        ('I', p) => (0, p),
        (p, 'I') => (0, p),
        ('X', 'X') | ('Y', 'Y') | ('Z', 'Z') => (0, 'I'),
        ('X', 'Y') => (1, 'Z'),
        ('Y', 'Z') => (1, 'X'),
        ('Z', 'X') => (1, 'Y'),
        ('Y', 'X') => (3, 'Z'),
        ('Z', 'Y') => (3, 'X'),
        ('X', 'Z') => (3, 'Y'),
        _ => unreachable!(),
    }
}

fn support_to_words(x: &[u8], z: &[u8]) -> Vec<u64> {
    let bits = x.len() * 2;
    let mut words = vec![0u64; (bits + 63) / 64];
    for (idx, bit) in x.iter().chain(z.iter()).enumerate() {
        if *bit != 0 {
            words[idx / 64] |= 1u64 << (idx % 64);
        }
    }
    words
}

fn solve_row_span(rows: &[Vec<u64>], target: &[u64], row_count: usize) -> Option<Vec<u64>> {
    let bit_count = rows.first().map(|row| row.len() * 64).unwrap_or(0);
    let coeff_words = (row_count + 63) / 64;
    let mut basis: Vec<Option<(Vec<u64>, Vec<u64>)>> = vec![None; bit_count];
    for (row_idx, row) in rows.iter().enumerate() {
        let mut vec = row.clone();
        let mut coeff = vec![0u64; coeff_words];
        coeff[row_idx / 64] |= 1u64 << (row_idx % 64);
        while let Some(pivot) = highest_bit(&vec) {
            if basis[pivot].is_none() {
                basis[pivot] = Some((vec, coeff));
                break;
            }
            let (basis_vec, basis_coeff) = basis[pivot].as_ref().unwrap();
            xor_words(&mut vec, basis_vec);
            xor_words(&mut coeff, basis_coeff);
        }
    }

    let mut vec = target.to_vec();
    let mut coeff = vec![0u64; coeff_words];
    while let Some(pivot) = highest_bit(&vec) {
        let (basis_vec, basis_coeff) = basis[pivot].as_ref()?;
        xor_words(&mut vec, basis_vec);
        xor_words(&mut coeff, basis_coeff);
    }
    Some(coeff)
}

fn highest_bit(words: &[u64]) -> Option<usize> {
    for (word_idx, word) in words.iter().enumerate().rev() {
        if *word != 0 {
            let local = 63 - word.leading_zeros() as usize;
            return Some(word_idx * 64 + local);
        }
    }
    None
}

fn xor_words(left: &mut [u64], right: &[u64]) {
    for (a, b) in left.iter_mut().zip(right) {
        *a ^= *b;
    }
}

fn coeff_bit(coeff: &[u64], idx: usize) -> bool {
    ((coeff[idx / 64] >> (idx % 64)) & 1) != 0
}

struct SmallRng {
    inner: RandSmallRng,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self {
            inner: RandSmallRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

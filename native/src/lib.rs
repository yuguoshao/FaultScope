use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule};
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
}

#[derive(Clone)]
enum Op {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Pauli { qubits: Vec<usize>, pauli: String },
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
    random_source_count: usize,
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
        let mut rng = SmallRng::new(seed.unwrap_or(0x4d59_5df4_d0f3_3173));
        let mut state = RuntimeState::new(
            self.n_qubits,
            shots,
            &self.noise_location_ids,
            true,
        );
        let random_masks = (0..self.random_source_count)
            .map(|_| random_bit_mask(&mut rng, state.all_mask.words.len(), state.shots))
            .collect::<Vec<_>>();
        for operation in &self.runtime_operations {
            apply_operation(operation, &mut state, &random_masks, &mut rng)?;
        }
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
        let mut rng = SmallRng::new(seed.unwrap_or(0x4d59_5df4_d0f3_3173));
        let mut state = RuntimeState::new(
            self.n_qubits,
            shots,
            &self.noise_location_ids,
            false,
        );
        let random_masks = (0..self.random_source_count)
            .map(|_| random_bit_mask(&mut rng, state.all_mask.words.len(), state.shots))
            .collect::<Vec<_>>();
        for operation in &self.runtime_operations {
            apply_operation(operation, &mut state, &random_masks, &mut rng)?;
        }
        map_to_py(py, &state.measurements)
    }
}

#[pyfunction]
fn compile_sampler(spec: &Bound<'_, PyDict>) -> PyResult<NativePackedSampler> {
    let n_qubits = required(spec, "n_qubits")?.extract::<usize>()?;
    let operations_any = required(spec, "operations")?;
    let operations_seq = operations_any.downcast::<PyList>()?;
    let mut operations = Vec::with_capacity(operations_seq.len());

    for item in operations_seq.iter() {
        let dict = item.downcast::<PyDict>()?;
        let op = parse_operation(dict)?;
        operations.push(op);
    }
    let compiled = compile_runtime_operations(n_qubits, operations)?;

    Ok(NativePackedSampler {
        n_qubits,
        runtime_operations: compiled.runtime_operations,
        noise_location_ids: compiled.noise_location_ids,
        random_source_count: compiled.random_source_count,
    })
}

#[pymodule]
fn _npsim_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", NATIVE_KERNEL_VERSION)?;
    module.add_class::<NativePackedSampler>()?;
    module.add_function(wrap_pyfunction!(compile_sampler, module)?)?;
    Ok(())
}

impl Op {
    fn noise_locations(&self) -> Vec<&NoiseLocationSpec> {
        match self {
            Op::Noise(location) => vec![location],
            Op::Measure { noise: Some(location), .. } => vec![location],
            Op::MeasurePauli { noise: Some(location), .. } => vec![location],
            _ => Vec::new(),
        }
    }
}

struct CompiledCircuit {
    runtime_operations: Vec<RunOp>,
    noise_location_ids: Vec<String>,
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

    for operation in operations {
        for location in operation.noise_locations() {
            if !seen_noise_ids.insert(location.id.clone()) {
                return Err(PyValueError::new_err(format!(
                    "native sampler requires unique noise location ids; duplicate {:?}",
                    location.id
                )));
            }
            noise_location_ids.push(location.id.clone());
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
        random_source_count: symbolic.random_source_count,
    })
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
    })
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
    let event_masks = sample_noise_event_masks(location, state.shots, state.all_mask.words.len(), rng)?;
    let mut error_mask = if state.record_events {
        Some(Mask::zero(state.all_mask.words.len()))
    } else {
        None
    };
    for (pauli, mask) in event_masks {
        if let Some(error_mask) = &mut error_mask {
            error_mask.xor_assign(&mask);
        }
        apply_masked_pauli_to_frame(state, &location.qubits, &pauli, &mask)?;
    }
    if let Some(error_mask) = error_mask {
        if let Some(mask) = state.event_masks.get_mut(&location.id) {
            mask.xor_assign(&error_mask);
        }
    }
    Ok(())
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

fn sample_noise_event_masks(
    location: &NoiseLocationSpec,
    shots: usize,
    words: usize,
    rng: &mut SmallRng,
) -> PyResult<Vec<(String, Mask)>> {
    match &location.model {
        NoiseModel::MeasurementBitFlip => Err(PyValueError::new_err(
            "MeasurementBitFlip must be attached to a measurement operation",
        )),
        NoiseModel::BernoulliPauli(pauli) => Ok(vec![(
            pauli.clone(),
            bernoulli_mask(rng, shots, location.rate),
        )]),
        NoiseModel::SingleQubitDepolarizing => Ok(sample_weighted_pauli_masks(
            &["X", "Y", "Z"],
            &[1.0, 1.0, 1.0],
            location.rate,
            shots,
            words,
            rng,
        )),
        NoiseModel::TwoQubitDepolarizing => {
            let events = [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI",
                "ZX", "ZY", "ZZ",
            ];
            Ok(sample_weighted_pauli_masks(
                &events,
                &[1.0; 15],
                location.rate,
                shots,
                words,
                rng,
            ))
        }
        NoiseModel::PauliChannel(weights) => {
            let events: Vec<&str> = weights.iter().map(|(event, _)| event.as_str()).collect();
            let values: Vec<f64> = weights.iter().map(|(_, weight)| *weight).collect();
            Ok(sample_weighted_pauli_masks(
                &events,
                &values,
                location.rate,
                shots,
                words,
                rng,
            ))
        }
    }
}

fn sample_weighted_pauli_masks(
    events: &[&str],
    weights: &[f64],
    rate: f64,
    shots: usize,
    words: usize,
    rng: &mut SmallRng,
) -> Vec<(String, Mask)> {
    let mut masks: Vec<(String, Mask)> = events
        .iter()
        .map(|event| ((*event).to_string(), Mask::zero(words)))
        .collect();
    if rate <= 0.0 {
        return masks;
    }
    let total: f64 = weights.iter().sum();
    for shot in 0..shots {
        if rng.next_f64() >= rate {
            continue;
        }
        let mut threshold = rng.next_f64() * total;
        let mut chosen = events.len() - 1;
        for (idx, weight) in weights.iter().enumerate() {
            if *weight == 0.0 {
                continue;
            }
            if threshold <= *weight {
                chosen = idx;
                break;
            }
            threshold -= *weight;
        }
        set_shot_bit(&mut masks[chosen].1, shot);
    }
    masks
}

fn bernoulli_mask(rng: &mut SmallRng, shots: usize, rate: f64) -> Mask {
    if rate <= 0.0 {
        return Mask::zero(word_count(shots));
    }
    if rate >= 1.0 {
        return Mask::all(shots);
    }
    let threshold = (rate * (u64::MAX as f64)) as u64;
    let mut mask = Mask::zero(word_count(shots));
    for shot in 0..shots {
        if rng.next_u64() <= threshold {
            set_shot_bit(&mut mask, shot);
        }
    }
    mask
}

fn random_bit_mask(rng: &mut SmallRng, words: usize, _all_words: usize) -> Mask {
    let mut mask = Mask::zero(words);
    for word in &mut mask.words {
        *word = rng.next_u64();
    }
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
        return Err(PyValueError::new_err("event Pauli length does not match qubits"));
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
        return Err(PyValueError::new_err("qubits and pauli must have the same length"));
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

fn measurement_parity(
    measurements: &HashMap<String, Mask>,
    keys: &[String],
) -> PyResult<Mask> {
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
        return Err(PyValueError::new_err("qubits and paulis must have the same length"));
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
        _ => Err(PyValueError::new_err(format!("unsupported Pauli {pauli:?}"))),
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
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(0xda94_2042_e4dd_58b5)
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

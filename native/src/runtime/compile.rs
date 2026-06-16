use crate::*;

pub(crate) struct CompiledCircuit {
    pub(crate) runtime_operations: Vec<RunOp>,
    pub(crate) noise_location_ids: Vec<String>,
    pub(crate) noise_locations: Vec<NoiseLocationSpec>,
    pub(crate) random_source_count: usize,
}

#[derive(Clone, Default)]
pub(crate) struct Expr {
    pub(crate) terms: Vec<usize>,
    pub(crate) constant: bool,
}

impl Expr {
    pub(crate) fn constant(value: bool) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
        }
    }

    pub(crate) fn random(source: usize) -> Self {
        Self {
            terms: vec![source],
            constant: false,
        }
    }

    pub(crate) fn xor_assign(&mut self, other: &Expr) {
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

    pub(crate) fn toggle_constant(&mut self) {
        self.constant ^= true;
    }

    pub(crate) fn eval(&self, random_masks: &[Mask], words: usize, all_mask: &Mask) -> Mask {
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

pub(crate) struct SymbolicStabilizer {
    pub(crate) x: Vec<Vec<u8>>,
    pub(crate) z: Vec<Vec<u8>>,
    pub(crate) sign: Vec<Expr>,
    pub(crate) random_source_count: usize,
}

impl SymbolicStabilizer {
    pub(crate) fn zero(n_qubits: usize) -> Self {
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

    pub(crate) fn n_qubits(&self) -> usize {
        self.x.len()
    }

    pub(crate) fn apply_h(&mut self, qubit: usize) {
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

    pub(crate) fn apply_s(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row].toggle_constant();
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z == 0 {
                self.sign[row].toggle_constant();
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    pub(crate) fn apply_cx(&mut self, control: usize, target: usize) {
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

    pub(crate) fn apply_cz(&mut self, left: usize, right: usize) {
        self.apply_h(right);
        self.apply_cx(left, right);
        self.apply_h(right);
    }

    pub(crate) fn apply_swap(&mut self, left: usize, right: usize) {
        if left == right {
            return;
        }
        self.apply_cx(left, right);
        self.apply_cx(right, left);
        self.apply_cx(left, right);
    }

    pub(crate) fn apply_pauli_string(&mut self, x: &[u8], z: &[u8]) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row].toggle_constant();
            }
        }
    }

    pub(crate) fn apply_pauli_string_expr(&mut self, x: &[u8], z: &[u8], expr: &Expr) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row].xor_assign(expr);
            }
        }
    }

    pub(crate) fn measure_pauli_expr(&mut self, x: &[u8], z: &[u8]) -> PyResult<Expr> {
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

    pub(crate) fn deterministic_measurement_expr(&self, x: &[u8], z: &[u8]) -> PyResult<Expr> {
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

pub(crate) fn compile_runtime_operations(
    n_qubits: usize,
    operations: Vec<Op>,
) -> PyResult<CompiledCircuit> {
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

use crate::*;

pub(crate) struct ConcreteStabilizer {
    pub(crate) x: Vec<Vec<u8>>,
    pub(crate) z: Vec<Vec<u8>>,
    pub(crate) sign: Vec<bool>,
}

impl ConcreteStabilizer {
    pub(crate) fn zero(n_qubits: usize) -> Self {
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

    pub(crate) fn n_qubits(&self) -> usize {
        self.x.len()
    }

    pub(crate) fn apply_h(&mut self, qubit: usize) {
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

    pub(crate) fn apply_s(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row] ^= true;
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z == 0 {
                self.sign[row] ^= true;
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
                self.sign[row] ^= true;
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
                self.sign[row] ^= true;
            }
        }
    }

    pub(crate) fn is_deterministic_pauli(&self, x: &[u8], z: &[u8]) -> bool {
        (0..self.n_qubits()).all(|row| symplectic_product(&self.x[row], &self.z[row], x, z) == 0)
    }

    pub(crate) fn deterministic_measurement_bit(&self, x: &[u8], z: &[u8]) -> PyResult<bool> {
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

    pub(crate) fn measure_pauli_with_outcome(
        &mut self,
        x: &[u8],
        z: &[u8],
        outcome: bool,
    ) -> PyResult<bool> {
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

    pub(crate) fn reset_prepare(&mut self, qubit: usize, basis: &str) -> PyResult<()> {
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

pub(crate) fn multiply_concrete_rows(
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

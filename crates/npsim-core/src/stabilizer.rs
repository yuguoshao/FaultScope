use crate::{
    coeff_bit, multiply_concrete_rows, multiply_symbolic_rows, sparse_pauli_to_xz,
    support_to_words, symplectic_product, Expr, NpError, NpResult,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConcreteStabilizer {
    x: Vec<Vec<u8>>,
    z: Vec<Vec<u8>>,
    sign: Vec<bool>,
}

impl ConcreteStabilizer {
    pub fn new(x: Vec<Vec<u8>>, z: Vec<Vec<u8>>, sign: Vec<bool>) -> NpResult<Self> {
        if x.len() != z.len() || x.len() != sign.len() {
            return Err(NpError::new("x, z, and sign must have the same length"));
        }
        let n_qubits = x.len();
        if x.iter().any(|row| row.len() != n_qubits) || z.iter().any(|row| row.len() != n_qubits) {
            return Err(NpError::new("stabilizer rows must be square"));
        }
        Ok(Self { x, z, sign })
    }

    pub fn zero(n_qubits: usize) -> Self {
        let x = vec![vec![0; n_qubits]; n_qubits];
        let mut z = vec![vec![0; n_qubits]; n_qubits];
        for (qubit, row) in z.iter_mut().enumerate().take(n_qubits) {
            row[qubit] = 1;
        }
        Self {
            x,
            z,
            sign: vec![false; n_qubits],
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.x.len()
    }

    pub fn x_rows(&self) -> &[Vec<u8>] {
        &self.x
    }

    pub fn z_rows(&self) -> &[Vec<u8>] {
        &self.z
    }

    pub fn sign_bits(&self) -> &[bool] {
        &self.sign
    }

    pub fn apply_h(&mut self, qubit: usize) {
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

    pub fn apply_s(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row] ^= true;
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    pub fn apply_s_dag(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z == 0 {
                self.sign[row] ^= true;
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    pub fn apply_cx(&mut self, control: usize, target: usize) {
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

    pub fn apply_cz(&mut self, left: usize, right: usize) {
        self.apply_h(right);
        self.apply_cx(left, right);
        self.apply_h(right);
    }

    pub fn apply_swap(&mut self, left: usize, right: usize) {
        if left == right {
            return;
        }
        self.apply_cx(left, right);
        self.apply_cx(right, left);
        self.apply_cx(left, right);
    }

    pub fn apply_pauli_string(&mut self, x: &[u8], z: &[u8]) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row] ^= true;
            }
        }
    }

    pub fn is_deterministic_pauli(&self, x: &[u8], z: &[u8]) -> bool {
        (0..self.n_qubits()).all(|row| symplectic_product(&self.x[row], &self.z[row], x, z) == 0)
    }

    pub fn deterministic_measurement_bit(&self, x: &[u8], z: &[u8]) -> NpResult<bool> {
        let rows: Vec<Vec<u64>> = (0..self.n_qubits())
            .map(|row| support_to_words(&self.x[row], &self.z[row]))
            .collect();
        let target = support_to_words(x, z);
        let coeff = crate::solve_row_span(&rows, &target, self.n_qubits())
            .ok_or_else(|| NpError::new("commuting Pauli was not in the stabilizer span"))?;

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

    pub fn measure_pauli_with_outcome(
        &mut self,
        x: &[u8],
        z: &[u8],
        outcome: bool,
    ) -> NpResult<bool> {
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

    pub fn reset_prepare(&mut self, qubit: usize, basis: &str) -> NpResult<()> {
        let qubits = vec![qubit];
        let (x, z) = sparse_pauli_to_xz(self.n_qubits(), &qubits, basis)?;
        if self.is_deterministic_pauli(&x, &z) {
            let bit = self.deterministic_measurement_bit(&x, &z)?;
            if bit {
                let correction = match basis {
                    "Z" => "X",
                    "X" => "Z",
                    "Y" => "X",
                    _ => return Err(NpError::new(format!("unsupported reset basis {basis:?}"))),
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

pub fn frame_apply_h(x_frame: &mut [u8], z_frame: &mut [u8], qubit: usize) {
    std::mem::swap(&mut x_frame[qubit], &mut z_frame[qubit]);
}

pub fn frame_apply_s(x_frame: &mut [u8], z_frame: &mut [u8], qubit: usize) {
    z_frame[qubit] ^= x_frame[qubit];
}

pub fn frame_apply_cx(x_frame: &mut [u8], z_frame: &mut [u8], control: usize, target: usize) {
    x_frame[target] ^= x_frame[control];
    z_frame[control] ^= z_frame[target];
}

pub fn frame_apply_cz(x_frame: &mut [u8], z_frame: &mut [u8], left: usize, right: usize) {
    frame_apply_h(x_frame, z_frame, right);
    frame_apply_cx(x_frame, z_frame, left, right);
    frame_apply_h(x_frame, z_frame, right);
}

pub fn frame_apply_swap(x_frame: &mut [u8], z_frame: &mut [u8], left: usize, right: usize) {
    if left == right {
        return;
    }
    frame_apply_cx(x_frame, z_frame, left, right);
    frame_apply_cx(x_frame, z_frame, right, left);
    frame_apply_cx(x_frame, z_frame, left, right);
}

pub fn frame_apply_pauli_string(
    x_frame: &mut [u8],
    z_frame: &mut [u8],
    qubits: &[usize],
    pauli: &str,
) -> NpResult<()> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("event Pauli length does not match qubits"));
    }
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (x, z) = crate::pauli_to_xz(local)?;
        x_frame[*qubit] ^= x;
        z_frame[*qubit] ^= z;
    }
    Ok(())
}

pub fn frame_measurement_flip_bits(
    x_frame: &[u8],
    z_frame: &[u8],
    qubits: &[usize],
    pauli: &str,
) -> NpResult<bool> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and paulis must have the same length"));
    }
    let mut x = vec![0; x_frame.len()];
    let mut z = vec![0; z_frame.len()];
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (px, pz) = crate::pauli_to_xz(local)?;
        x[*qubit] ^= px;
        z[*qubit] ^= pz;
    }
    Ok(symplectic_product(x_frame, z_frame, &x, &z) != 0)
}

pub struct SymbolicStabilizer {
    x: Vec<Vec<u8>>,
    z: Vec<Vec<u8>>,
    sign: Vec<Expr>,
    random_source_count: usize,
}

impl SymbolicStabilizer {
    pub fn zero(n_qubits: usize) -> Self {
        let x = vec![vec![0; n_qubits]; n_qubits];
        let mut z = vec![vec![0; n_qubits]; n_qubits];
        for (qubit, row) in z.iter_mut().enumerate().take(n_qubits) {
            row[qubit] = 1;
        }
        Self {
            x,
            z,
            sign: vec![Expr::default(); n_qubits],
            random_source_count: 0,
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.x.len()
    }

    pub fn random_source_count(&self) -> usize {
        self.random_source_count
    }

    pub fn apply_h(&mut self, qubit: usize) {
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

    pub fn apply_s(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z != 0 {
                self.sign[row].toggle_constant();
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    pub fn apply_s_dag(&mut self, qubit: usize) {
        for row in 0..self.n_qubits() {
            let old_x = self.x[row][qubit];
            let old_z = self.z[row][qubit];
            if old_x != 0 && old_z == 0 {
                self.sign[row].toggle_constant();
            }
            self.z[row][qubit] = old_z ^ old_x;
        }
    }

    pub fn apply_cx(&mut self, control: usize, target: usize) {
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

    pub fn apply_cz(&mut self, left: usize, right: usize) {
        self.apply_h(right);
        self.apply_cx(left, right);
        self.apply_h(right);
    }

    pub fn apply_swap(&mut self, left: usize, right: usize) {
        if left == right {
            return;
        }
        self.apply_cx(left, right);
        self.apply_cx(right, left);
        self.apply_cx(left, right);
    }

    pub fn apply_pauli_string(&mut self, x: &[u8], z: &[u8]) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row].toggle_constant();
            }
        }
    }

    pub fn apply_pauli_string_expr(&mut self, x: &[u8], z: &[u8], expr: &Expr) {
        for row in 0..self.n_qubits() {
            if symplectic_product(&self.x[row], &self.z[row], x, z) != 0 {
                self.sign[row].xor_assign(expr);
            }
        }
    }

    pub fn measure_pauli_expr(&mut self, x: &[u8], z: &[u8]) -> NpResult<Expr> {
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

    pub fn deterministic_measurement_expr(&self, x: &[u8], z: &[u8]) -> NpResult<Expr> {
        let rows: Vec<Vec<u64>> = (0..self.n_qubits())
            .map(|row| support_to_words(&self.x[row], &self.z[row]))
            .collect();
        let target = support_to_words(x, z);
        let coeff = crate::solve_row_span(&rows, &target, self.n_qubits())
            .ok_or_else(|| NpError::new("commuting Pauli was not in the stabilizer span"))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_prepares_zero_state() {
        let mut state = ConcreteStabilizer::zero(1);
        let (x, z) = sparse_pauli_to_xz(1, &[0], "Z").unwrap();

        state.reset_prepare(0, "Z").unwrap();

        assert!(state.is_deterministic_pauli(&x, &z));
        assert!(!state.deterministic_measurement_bit(&x, &z).unwrap());
    }

    #[test]
    fn frame_pauli_flips_measurement_when_anticommuting() {
        let mut x_frame = vec![1];
        let mut z_frame = vec![0];

        assert!(frame_measurement_flip_bits(&x_frame, &z_frame, &[0], "Z").unwrap());

        frame_apply_h(&mut x_frame, &mut z_frame, 0);

        assert!(!frame_measurement_flip_bits(&x_frame, &z_frame, &[0], "Z").unwrap());
    }

    #[test]
    fn symbolic_measurement_tracks_random_sources() {
        let mut state = SymbolicStabilizer::zero(1);
        let (x, z) = sparse_pauli_to_xz(1, &[0], "X").unwrap();

        let expr = state.measure_pauli_expr(&x, &z).unwrap();

        assert_eq!(expr.terms(), &[0]);
        assert_eq!(state.random_source_count(), 1);
    }
}

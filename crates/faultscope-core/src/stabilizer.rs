use crate::pauli::sparse_symplectic_product;
use crate::{
    coeff_bit, multiply_concrete_rows, sparse_pauli_to_xz, support_to_words, symplectic_product,
    Expr, NpError, NpResult,
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

    pub(crate) fn apply_sparse_pauli_string(
        &mut self,
        qubits: &[usize],
        pauli: &str,
    ) -> NpResult<()> {
        for row in 0..self.n_qubits() {
            if sparse_symplectic_product(&self.x[row], &self.z[row], qubits, pauli)? != 0 {
                self.sign[row] ^= true;
            }
        }
        Ok(())
    }

    pub fn is_deterministic_pauli(&self, x: &[u8], z: &[u8]) -> bool {
        (0..self.n_qubits()).all(|row| symplectic_product(&self.x[row], &self.z[row], x, z) == 0)
    }

    pub fn is_deterministic_sparse_pauli(&self, qubits: &[usize], pauli: &str) -> NpResult<bool> {
        let pauli_bytes = pauli.as_bytes();
        if qubits.len() != pauli_bytes.len() {
            return Err(NpError::new("qubits and paulis must have the same length"));
        }
        if pauli_bytes.iter().all(|local| *local == b'Z') {
            for row in 0..self.n_qubits() {
                let mut acc = 0;
                for qubit in qubits {
                    acc ^= self.x[row][*qubit];
                }
                if (acc & 1) != 0 {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if pauli_bytes.iter().all(|local| *local == b'X') {
            for row in 0..self.n_qubits() {
                let mut acc = 0;
                for qubit in qubits {
                    acc ^= self.z[row][*qubit];
                }
                if (acc & 1) != 0 {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if pauli_bytes.iter().all(|local| *local == b'Y') {
            for row in 0..self.n_qubits() {
                let mut acc = 0;
                for qubit in qubits {
                    acc ^= self.x[row][*qubit] ^ self.z[row][*qubit];
                }
                if (acc & 1) != 0 {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        for row in 0..self.n_qubits() {
            let mut acc = 0;
            for (qubit, local) in qubits.iter().zip(pauli_bytes) {
                match *local {
                    b'I' => {}
                    b'X' => acc ^= self.z[row][*qubit],
                    b'Z' => acc ^= self.x[row][*qubit],
                    b'Y' => acc ^= self.x[row][*qubit] ^ self.z[row][*qubit],
                    _ => {
                        return Err(NpError::new(format!(
                            "unsupported Pauli {:?}",
                            *local as char
                        )))
                    }
                }
            }
            if (acc & 1) != 0 {
                return Ok(false);
            }
        }
        Ok(true)
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

/// Compiler-only symbolic Aaronson-Gottesman tableau.
///
/// Rows ``0..n`` are destabilizers and rows ``n..2n`` are stabilizers. Pauli
/// supports are packed into u64 words. Keeping the dual destabilizer basis lets
/// deterministic measurements recover their stabilizer coefficients directly,
/// instead of rebuilding and eliminating a dense row span for every result.
pub(crate) struct SymbolicStabilizer {
    n_qubits: usize,
    x: Vec<Vec<u64>>,
    z: Vec<Vec<u64>>,
    sign: Vec<Expr>,
    random_source_count: usize,
}

impl SymbolicStabilizer {
    pub(crate) fn zero(n_qubits: usize) -> Self {
        let words = n_qubits.div_ceil(64);
        let mut x = vec![vec![0; words]; n_qubits * 2];
        let mut z = vec![vec![0; words]; n_qubits * 2];
        for qubit in 0..n_qubits {
            toggle_packed_bit(&mut x[qubit], qubit);
            toggle_packed_bit(&mut z[n_qubits + qubit], qubit);
        }
        Self {
            n_qubits,
            x,
            z,
            sign: vec![Expr::default(); n_qubits * 2],
            random_source_count: 0,
        }
    }

    pub(crate) fn random_source_count(&self) -> usize {
        self.random_source_count
    }

    pub(crate) fn apply_h(&mut self, qubit: usize) {
        for row in 0..self.x.len() {
            let old_x = packed_bit(&self.x[row], qubit);
            let old_z = packed_bit(&self.z[row], qubit);
            if old_x && old_z {
                self.sign[row].toggle_constant();
            }
            set_packed_bit(&mut self.x[row], qubit, old_z);
            set_packed_bit(&mut self.z[row], qubit, old_x);
        }
    }

    pub(crate) fn apply_s(&mut self, qubit: usize) {
        for row in 0..self.x.len() {
            let old_x = packed_bit(&self.x[row], qubit);
            let old_z = packed_bit(&self.z[row], qubit);
            if old_x && old_z {
                self.sign[row].toggle_constant();
            }
            set_packed_bit(&mut self.z[row], qubit, old_z ^ old_x);
        }
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: usize) {
        for row in 0..self.x.len() {
            let old_x = packed_bit(&self.x[row], qubit);
            let old_z = packed_bit(&self.z[row], qubit);
            if old_x && !old_z {
                self.sign[row].toggle_constant();
            }
            set_packed_bit(&mut self.z[row], qubit, old_z ^ old_x);
        }
    }

    pub(crate) fn apply_cx(&mut self, control: usize, target: usize) {
        for row in 0..self.x.len() {
            let x_c = packed_bit(&self.x[row], control);
            let z_c = packed_bit(&self.z[row], control);
            let x_t = packed_bit(&self.x[row], target);
            let z_t = packed_bit(&self.z[row], target);
            if x_t && z_c && !(x_c ^ z_t) {
                self.sign[row].toggle_constant();
            }
            set_packed_bit(&mut self.x[row], target, x_t ^ x_c);
            set_packed_bit(&mut self.z[row], control, z_c ^ z_t);
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

    pub(crate) fn apply_sparse_pauli_string(
        &mut self,
        qubits: &[usize],
        pauli: &str,
    ) -> NpResult<()> {
        let target = SparsePackedPauli::new(self.n_qubits, qubits, pauli)?;
        for row in 0..self.x.len() {
            if target.symplectic_product(&self.x[row], &self.z[row]) {
                self.sign[row].toggle_constant();
            }
        }
        Ok(())
    }

    pub(crate) fn apply_sparse_pauli_string_expr(
        &mut self,
        qubits: &[usize],
        pauli: &str,
        expr: &Expr,
    ) -> NpResult<()> {
        let target = SparsePackedPauli::new(self.n_qubits, qubits, pauli)?;
        for row in 0..self.x.len() {
            if target.symplectic_product(&self.x[row], &self.z[row]) {
                self.sign[row].xor_assign(expr);
            }
        }
        Ok(())
    }

    pub(crate) fn measure_sparse_pauli_expr(
        &mut self,
        qubits: &[usize],
        pauli: &str,
    ) -> NpResult<Expr> {
        let target = SparsePackedPauli::new(self.n_qubits, qubits, pauli)?;
        let pivot = (self.n_qubits..self.n_qubits * 2)
            .find(|row| target.symplectic_product(&self.x[*row], &self.z[*row]));
        match pivot {
            Some(pivot) => self.measure_random_sparse_pauli(pivot, &target),
            None => self.deterministic_sparse_measurement_expr(&target),
        }
    }

    fn measure_random_sparse_pauli(
        &mut self,
        pivot: usize,
        target: &SparsePackedPauli,
    ) -> NpResult<Expr> {
        let paired_destabilizer = pivot - self.n_qubits;
        let pivot_x = self.x[pivot].clone();
        let pivot_z = self.z[pivot].clone();
        let pivot_sign = self.sign[pivot].clone();
        for row in 0..self.x.len() {
            if row != pivot
                && row != paired_destabilizer
                && target.symplectic_product(&self.x[row], &self.z[row])
            {
                packed_rowsum(
                    &mut self.x[row],
                    &mut self.z[row],
                    &mut self.sign[row],
                    &pivot_x,
                    &pivot_z,
                    &pivot_sign,
                )?;
            }
        }
        self.finish_random_measurement(
            pivot,
            paired_destabilizer,
            pivot_x,
            pivot_z,
            pivot_sign,
            target.x.clone(),
            target.z.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_random_measurement(
        &mut self,
        pivot: usize,
        paired_destabilizer: usize,
        pivot_x: Vec<u64>,
        pivot_z: Vec<u64>,
        pivot_sign: Expr,
        target_x: Vec<u64>,
        target_z: Vec<u64>,
    ) -> NpResult<Expr> {
        self.x[paired_destabilizer] = pivot_x;
        self.z[paired_destabilizer] = pivot_z;
        self.sign[paired_destabilizer] = pivot_sign;

        let source = self.random_source_count;
        self.random_source_count += 1;
        let outcome = Expr::random(source);
        self.x[pivot] = target_x;
        self.z[pivot] = target_z;
        self.sign[pivot] = outcome.clone();
        Ok(outcome)
    }

    fn deterministic_sparse_measurement_expr(&self, target: &SparsePackedPauli) -> NpResult<Expr> {
        self.combine_stabilizers_selected_by(|destabilizer| {
            target.symplectic_product(&self.x[destabilizer], &self.z[destabilizer])
        })
    }

    fn combine_stabilizers_selected_by<F>(&self, mut selected: F) -> NpResult<Expr>
    where
        F: FnMut(usize) -> bool,
    {
        let words = self.n_qubits.div_ceil(64);
        let mut acc_x = vec![0; words];
        let mut acc_z = vec![0; words];
        let mut acc_sign = Expr::default();
        for destabilizer in 0..self.n_qubits {
            if selected(destabilizer) {
                let stabilizer = self.n_qubits + destabilizer;
                packed_rowsum(
                    &mut acc_x,
                    &mut acc_z,
                    &mut acc_sign,
                    &self.x[stabilizer],
                    &self.z[stabilizer],
                    &self.sign[stabilizer],
                )?;
            }
        }
        Ok(acc_sign)
    }
}

struct SparsePackedPauli {
    entries: Vec<(usize, bool, bool)>,
    x: Vec<u64>,
    z: Vec<u64>,
}

impl SparsePackedPauli {
    fn new(n_qubits: usize, qubits: &[usize], pauli: &str) -> NpResult<Self> {
        if qubits.len() != pauli.len() {
            return Err(NpError::new("qubits and paulis must have the same length"));
        }
        let words = n_qubits.div_ceil(64);
        let mut entries = Vec::with_capacity(qubits.len());
        let mut x = vec![0; words];
        let mut z = vec![0; words];
        for (qubit, local) in qubits.iter().zip(pauli.chars()) {
            if *qubit >= n_qubits {
                return Err(NpError::new(format!("qubit {qubit} is out of range")));
            }
            let (local_x, local_z) = crate::pauli_to_xz(local)?;
            let local_x = local_x != 0;
            let local_z = local_z != 0;
            entries.push((*qubit, local_x, local_z));
            if local_x {
                toggle_packed_bit(&mut x, *qubit);
            }
            if local_z {
                toggle_packed_bit(&mut z, *qubit);
            }
        }
        Ok(Self { entries, x, z })
    }

    fn symplectic_product(&self, row_x: &[u64], row_z: &[u64]) -> bool {
        self.entries.iter().fold(false, |acc, (qubit, x, z)| {
            acc ^ (packed_bit(row_x, *qubit) && *z) ^ (packed_bit(row_z, *qubit) && *x)
        })
    }
}

fn packed_bit(words: &[u64], qubit: usize) -> bool {
    ((words[qubit / 64] >> (qubit % 64)) & 1) != 0
}

fn set_packed_bit(words: &mut [u64], qubit: usize, value: bool) {
    let mask = 1u64 << (qubit % 64);
    if value {
        words[qubit / 64] |= mask;
    } else {
        words[qubit / 64] &= !mask;
    }
}

fn toggle_packed_bit(words: &mut [u64], qubit: usize) {
    words[qubit / 64] ^= 1u64 << (qubit % 64);
}

fn packed_rowsum(
    target_x: &mut [u64],
    target_z: &mut [u64],
    target_sign: &mut Expr,
    source_x: &[u64],
    source_z: &[u64],
    source_sign: &Expr,
) -> NpResult<()> {
    let mut phase = 0u32;
    for (((target_x_word, target_z_word), source_x_word), source_z_word) in target_x
        .iter_mut()
        .zip(target_z.iter_mut())
        .zip(source_x)
        .zip(source_z)
    {
        let tx = *target_x_word;
        let tz = *target_z_word;
        let sx = *source_x_word;
        let sz = *source_z_word;
        let plus = (tx & !tz & sx & sz).count_ones()
            + (tx & tz & !sx & sz).count_ones()
            + (!tx & tz & sx & !sz).count_ones();
        let minus = (tx & tz & sx & !sz).count_ones()
            + (!tx & tz & sx & sz).count_ones()
            + (tx & !tz & !sx & sz).count_ones();
        phase = (phase + plus + 3 * minus) & 3;
        *target_x_word = tx ^ sx;
        *target_z_word = tz ^ sz;
    }
    target_sign.xor_assign(source_sign);
    match phase {
        0 => Ok(()),
        2 => {
            target_sign.toggle_constant();
            Ok(())
        }
        _ => Err(NpError::new(
            "product of symbolic tableau rows produced a non-Hermitian phase",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DenseSymbolicReference {
        x: Vec<Vec<u8>>,
        z: Vec<Vec<u8>>,
        sign: Vec<Expr>,
        random_source_count: usize,
    }

    impl DenseSymbolicReference {
        fn zero(n_qubits: usize) -> Self {
            let x = vec![vec![0; n_qubits]; n_qubits];
            let mut z = vec![vec![0; n_qubits]; n_qubits];
            for (qubit, row) in z.iter_mut().enumerate() {
                row[qubit] = 1;
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
            self.apply_cx(left, right);
            self.apply_cx(right, left);
            self.apply_cx(left, right);
        }

        fn apply_sparse_pauli_expr(&mut self, qubits: &[usize], pauli: &str, expr: &Expr) {
            let (x, z) = sparse_pauli_to_xz(self.n_qubits(), qubits, pauli).unwrap();
            for row in 0..self.n_qubits() {
                if symplectic_product(&self.x[row], &self.z[row], &x, &z) != 0 {
                    self.sign[row].xor_assign(expr);
                }
            }
        }

        fn measure_sparse(&mut self, qubits: &[usize], pauli: &str) -> Expr {
            let (x, z) = sparse_pauli_to_xz(self.n_qubits(), qubits, pauli).unwrap();
            self.measure(&x, &z)
        }

        fn measure(&mut self, x: &[u8], z: &[u8]) -> Expr {
            let anti: Vec<usize> = (0..self.n_qubits())
                .filter(|row| symplectic_product(&self.x[*row], &self.z[*row], x, z) != 0)
                .collect();
            if anti.is_empty() {
                return self.deterministic_measurement(x, z);
            }

            let source = self.random_source_count;
            self.random_source_count += 1;
            let outcome = Expr::random(source);
            let pivot = anti[0];
            let old_x = self.x[pivot].clone();
            let old_z = self.z[pivot].clone();
            let old_sign = self.sign[pivot].clone();
            for row in anti.into_iter().skip(1) {
                let (new_x, new_z, new_sign) = crate::pauli::multiply_symbolic_rows(
                    &self.x[row],
                    &self.z[row],
                    &self.sign[row],
                    &old_x,
                    &old_z,
                    &old_sign,
                )
                .unwrap();
                self.x[row] = new_x;
                self.z[row] = new_z;
                self.sign[row] = new_sign;
            }
            self.x[pivot] = x.to_vec();
            self.z[pivot] = z.to_vec();
            self.sign[pivot] = outcome.clone();
            outcome
        }

        fn deterministic_measurement(&self, x: &[u8], z: &[u8]) -> Expr {
            let rows: Vec<Vec<u64>> = (0..self.n_qubits())
                .map(|row| support_to_words(&self.x[row], &self.z[row]))
                .collect();
            let target = support_to_words(x, z);
            let coeff = crate::solve_row_span(&rows, &target, self.n_qubits()).unwrap();
            let mut acc_x = vec![0; self.n_qubits()];
            let mut acc_z = vec![0; self.n_qubits()];
            let mut acc_sign = Expr::default();
            for row in 0..self.n_qubits() {
                if coeff_bit(&coeff, row) {
                    (acc_x, acc_z, acc_sign) = crate::pauli::multiply_symbolic_rows(
                        &acc_x,
                        &acc_z,
                        &acc_sign,
                        &self.x[row],
                        &self.z[row],
                        &self.sign[row],
                    )
                    .unwrap();
                }
            }
            acc_sign
        }
    }

    fn assert_matches_dense_reference(packed: &SymbolicStabilizer, dense: &DenseSymbolicReference) {
        assert_eq!(packed.n_qubits, dense.n_qubits());
        assert_eq!(packed.random_source_count, dense.random_source_count);
        for row in 0..dense.n_qubits() {
            let packed_row = packed.n_qubits + row;
            for qubit in 0..dense.n_qubits() {
                assert_eq!(
                    packed_bit(&packed.x[packed_row], qubit),
                    dense.x[row][qubit] != 0,
                    "x row {row}, qubit {qubit}"
                );
                assert_eq!(
                    packed_bit(&packed.z[packed_row], qubit),
                    dense.z[row][qubit] != 0,
                    "z row {row}, qubit {qubit}"
                );
            }
            assert_eq!(packed.sign[packed_row], dense.sign[row], "sign row {row}");
        }
    }

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
    fn concrete_sparse_pauli_path_matches_dense_path() {
        for (qubits, pauli) in [
            (vec![0], "X"),
            (vec![1], "Y"),
            (vec![2], "Z"),
            (vec![0, 2], "XZ"),
        ] {
            let mut sparse = ConcreteStabilizer::zero(3);
            let mut dense = ConcreteStabilizer::zero(3);
            sparse.apply_h(0);
            dense.apply_h(0);
            sparse.apply_cx(0, 1);
            dense.apply_cx(0, 1);
            let (x, z) = sparse_pauli_to_xz(3, &qubits, pauli).unwrap();

            sparse.apply_sparse_pauli_string(&qubits, pauli).unwrap();
            dense.apply_pauli_string(&x, &z);

            assert_eq!(sparse, dense);
        }
    }

    #[test]
    fn symbolic_measurement_tracks_random_sources() {
        let mut state = SymbolicStabilizer::zero(1);
        let expr = state.measure_sparse_pauli_expr(&[0], "X").unwrap();

        assert_eq!(expr.terms(), &[0]);
        assert_eq!(state.random_source_count(), 1);
    }

    #[test]
    fn packed_symbolic_tableau_matches_dense_reference_gate_by_gate() {
        let mut packed = SymbolicStabilizer::zero(3);
        let mut dense = DenseSymbolicReference::zero(3);

        packed.apply_h(0);
        dense.apply_h(0);
        assert_matches_dense_reference(&packed, &dense);
        packed.apply_s(0);
        dense.apply_s(0);
        assert_matches_dense_reference(&packed, &dense);
        packed.apply_s_dag(0);
        dense.apply_s_dag(0);
        assert_matches_dense_reference(&packed, &dense);
        packed.apply_cx(0, 1);
        dense.apply_cx(0, 1);
        assert_matches_dense_reference(&packed, &dense);
        packed.apply_cz(1, 2);
        dense.apply_cz(1, 2);
        assert_matches_dense_reference(&packed, &dense);
        packed.apply_swap(0, 2);
        dense.apply_swap(0, 2);
        assert_matches_dense_reference(&packed, &dense);

        for (qubit, pauli) in [(0, "X"), (1, "Y"), (2, "Z")] {
            packed
                .apply_sparse_pauli_string_expr(&[qubit], pauli, &Expr::constant(true))
                .unwrap();
            dense.apply_sparse_pauli_expr(&[qubit], pauli, &Expr::constant(true));
            assert_matches_dense_reference(&packed, &dense);
        }
    }

    #[test]
    fn packed_symbolic_measurements_and_reset_match_dense_reference() {
        let mut packed = SymbolicStabilizer::zero(3);
        let mut dense = DenseSymbolicReference::zero(3);
        packed.apply_h(0);
        dense.apply_h(0);

        let packed_random = packed.measure_sparse_pauli_expr(&[0], "Z").unwrap();
        let dense_random = dense.measure_sparse(&[0], "Z");
        assert_eq!(packed_random, dense_random);
        assert_matches_dense_reference(&packed, &dense);

        packed
            .apply_sparse_pauli_string_expr(&[0], "X", &packed_random)
            .unwrap();
        dense.apply_sparse_pauli_expr(&[0], "X", &dense_random);
        assert_matches_dense_reference(&packed, &dense);
        let packed_reset_result = packed.measure_sparse_pauli_expr(&[0], "Z").unwrap();
        let dense_reset_result = dense.measure_sparse(&[0], "Z");
        assert_eq!(packed_reset_result, dense_reset_result);
        assert_eq!(packed_reset_result, Expr::constant(false));

        packed.apply_h(1);
        dense.apply_h(1);
        let packed_mpp = packed.measure_sparse_pauli_expr(&[1, 2], "XX").unwrap();
        let dense_mpp = dense.measure_sparse(&[1, 2], "XX");
        assert_eq!(packed_mpp, dense_mpp);
        assert_matches_dense_reference(&packed, &dense);
        assert_eq!(
            packed.measure_sparse_pauli_expr(&[1, 2], "XX").unwrap(),
            dense.measure_sparse(&[1, 2], "XX")
        );
        assert_matches_dense_reference(&packed, &dense);
    }
}

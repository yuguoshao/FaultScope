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
        crate::model::validate_pauli_targets(self.n_qubits(), qubits, pauli, "stabilizer Pauli")?;
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
        crate::model::validate_pauli_targets(self.n_qubits(), qubits, pauli, "stabilizer Pauli")?;
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
    crate::model::validate_pauli_targets(x_frame.len(), qubits, pauli, "Pauli frame event")?;
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
    crate::model::validate_pauli_targets(x_frame.len(), qubits, pauli, "Pauli frame measurement")?;
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
#[derive(Clone)]
pub(crate) struct SymbolicStabilizer {
    n_qubits: usize,
    words: usize,
    row_words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    x_columns: Vec<u64>,
    z_columns: Vec<u64>,
    rows_valid: bool,
    sign: Vec<Expr>,
    random_source_count: usize,
    row_scratch: Vec<u64>,
    support_scratch_x: Vec<u64>,
    support_scratch_z: Vec<u64>,
}

impl SymbolicStabilizer {
    pub(crate) fn zero(n_qubits: usize) -> Self {
        let words = n_qubits.div_ceil(64);
        let mut x = vec![0; n_qubits * 2 * words];
        let mut z = vec![0; n_qubits * 2 * words];
        for qubit in 0..n_qubits {
            toggle_flat_row_bit(&mut x, words, qubit, qubit);
            toggle_flat_row_bit(&mut z, words, n_qubits + qubit, qubit);
        }
        let row_words = (n_qubits * 2).div_ceil(64);
        let mut x_columns = vec![0; n_qubits * row_words];
        let mut z_columns = vec![0; n_qubits * row_words];
        for qubit in 0..n_qubits {
            toggle_flat_column_bit(&mut x_columns, row_words, qubit, qubit);
            toggle_flat_column_bit(&mut z_columns, row_words, qubit, n_qubits + qubit);
        }
        Self {
            n_qubits,
            words,
            row_words,
            x,
            z,
            x_columns,
            z_columns,
            rows_valid: true,
            sign: vec![Expr::default(); n_qubits * 2],
            random_source_count: 0,
            row_scratch: vec![0; row_words],
            support_scratch_x: vec![0; words],
            support_scratch_z: vec![0; words],
        }
    }

    pub(crate) fn random_source_count(&self) -> usize {
        self.random_source_count
    }

    pub(crate) fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    fn row_start(&self, row: usize) -> usize {
        row * self.words
    }

    fn x_row(&self, row: usize) -> &[u64] {
        let start = self.row_start(row);
        &self.x[start..start + self.words]
    }

    fn z_row(&self, row: usize) -> &[u64] {
        let start = self.row_start(row);
        &self.z[start..start + self.words]
    }

    fn column_start(&self, qubit: usize) -> usize {
        qubit * self.row_words
    }

    #[cfg(test)]
    fn x_column(&self, qubit: usize) -> &[u64] {
        let start = self.column_start(qubit);
        &self.x_columns[start..start + self.row_words]
    }

    #[cfg(test)]
    fn z_column(&self, qubit: usize) -> &[u64] {
        let start = self.column_start(qubit);
        &self.z_columns[start..start + self.row_words]
    }

    fn synchronize_rows(&mut self) {
        if self.rows_valid {
            return;
        }
        self.x.fill(0);
        self.z.fill(0);
        for qubit_block in 0..self.words {
            let first_qubit = qubit_block * 64;
            let block_qubits = (self.n_qubits - first_qubit).min(64);
            for local_qubit in 0..block_qubits {
                let qubit = first_qubit + local_qubit;
                let column_start = self.column_start(qubit);
                for row_word in 0..self.row_words {
                    let mut x_rows = self.x_columns[column_start + row_word];
                    while x_rows != 0 {
                        let row_bit = x_rows.trailing_zeros() as usize;
                        let row = row_word * 64 + row_bit;
                        if row < self.n_qubits * 2 {
                            self.x[row * self.words + qubit_block] |= 1u64 << local_qubit;
                        }
                        x_rows &= x_rows - 1;
                    }
                    let mut z_rows = self.z_columns[column_start + row_word];
                    while z_rows != 0 {
                        let row_bit = z_rows.trailing_zeros() as usize;
                        let row = row_word * 64 + row_bit;
                        if row < self.n_qubits * 2 {
                            self.z[row * self.words + qubit_block] |= 1u64 << local_qubit;
                        }
                        z_rows &= z_rows - 1;
                    }
                }
            }
        }
        self.rows_valid = true;
    }

    pub(crate) fn affine_template(&self) -> Self {
        let mut template = self.clone();
        template.sign = (0..self.sign.len()).map(Expr::random).collect();
        template.random_source_count = self.sign.len();
        template
    }

    pub(crate) fn support_equals(&self, other: &Self) -> bool {
        self.n_qubits == other.n_qubits
            && self.x_columns == other.x_columns
            && self.z_columns == other.z_columns
    }

    pub(crate) fn signs(&self) -> &[Expr] {
        &self.sign
    }

    pub(crate) fn replace_affine_state(&mut self, signs: Vec<Expr>, random_source_count: usize) {
        self.sign = signs;
        self.random_source_count = random_source_count;
    }

    pub(crate) fn replace_with_affine_template(
        &mut self,
        mut template: Self,
        signs: Vec<Expr>,
        random_source_count: usize,
    ) {
        template.sign = signs;
        template.random_source_count = random_source_count;
        *self = template;
    }

    pub(crate) fn apply_h(&mut self, qubit: usize) {
        let start = self.column_start(qubit);
        for word_index in 0..self.row_words {
            let index = start + word_index;
            let old_x = self.x_columns[index];
            let old_z = self.z_columns[index];
            for_each_set_bit_in_word(word_index, old_x & old_z, |row| {
                self.sign[row].toggle_constant()
            });
            self.x_columns[index] = old_z;
            self.z_columns[index] = old_x;
        }
        self.rows_valid = false;
    }

    pub(crate) fn apply_s(&mut self, qubit: usize) {
        let start = self.column_start(qubit);
        for word_index in 0..self.row_words {
            let index = start + word_index;
            let x_rows = self.x_columns[index];
            let z_rows = self.z_columns[index];
            for_each_set_bit_in_word(word_index, x_rows & z_rows, |row| {
                self.sign[row].toggle_constant()
            });
            self.z_columns[index] = z_rows ^ x_rows;
        }
        self.rows_valid = false;
    }

    pub(crate) fn apply_s_dag(&mut self, qubit: usize) {
        let start = self.column_start(qubit);
        for word_index in 0..self.row_words {
            let index = start + word_index;
            let x_rows = self.x_columns[index];
            let z_rows = self.z_columns[index];
            for_each_set_bit_in_word(word_index, x_rows & !z_rows, |row| {
                self.sign[row].toggle_constant()
            });
            self.z_columns[index] = z_rows ^ x_rows;
        }
        self.rows_valid = false;
    }

    pub(crate) fn apply_cx(&mut self, control: usize, target: usize) {
        let control_start = self.column_start(control);
        let target_start = self.column_start(target);
        for word_index in 0..self.row_words {
            let control_index = control_start + word_index;
            let target_index = target_start + word_index;
            let x_c = self.x_columns[control_index];
            let z_c = self.z_columns[control_index];
            let x_t = self.x_columns[target_index];
            let z_t = self.z_columns[target_index];
            let sign_rows = x_t & z_c & !(x_c ^ z_t);
            for_each_set_bit_in_word(word_index, sign_rows, |row| {
                self.sign[row].toggle_constant()
            });
            if control == target {
                self.x_columns[target_index] = 0;
                self.z_columns[control_index] = 0;
            } else {
                self.x_columns[target_index] = x_t ^ x_c;
                self.z_columns[control_index] = z_c ^ z_t;
            }
        }
        self.rows_valid = false;
    }

    pub(crate) fn apply_cz(&mut self, left: usize, right: usize) {
        if left == right {
            self.apply_h(right);
            self.apply_cx(left, right);
            self.apply_h(right);
            return;
        }
        let left_start = self.column_start(left);
        let right_start = self.column_start(right);
        for word_index in 0..self.row_words {
            let left_index = left_start + word_index;
            let right_index = right_start + word_index;
            let x_l = self.x_columns[left_index];
            let z_l = self.z_columns[left_index];
            let x_r = self.x_columns[right_index];
            let z_r = self.z_columns[right_index];
            let sign_rows = x_l & x_r & !(z_l ^ z_r);
            for_each_set_bit_in_word(word_index, sign_rows, |row| {
                self.sign[row].toggle_constant()
            });
            self.z_columns[left_index] = z_l ^ x_r;
            self.z_columns[right_index] = z_r ^ x_l;
        }
        self.rows_valid = false;
    }

    pub(crate) fn apply_swap(&mut self, left: usize, right: usize) {
        if left == right {
            return;
        }
        let left_start = self.column_start(left);
        let right_start = self.column_start(right);
        for word_index in 0..self.row_words {
            self.x_columns
                .swap(left_start + word_index, right_start + word_index);
            self.z_columns
                .swap(left_start + word_index, right_start + word_index);
        }
        self.rows_valid = false;
    }

    /*
     * Gate updates above keep only the contiguous column-major tableau current.
     * Row-major support is rebuilt once, in 64-qubit blocks, before the next
     * measurement that needs row products.
     */

    pub(crate) fn apply_sparse_pauli_string(
        &mut self,
        qubits: &[usize],
        pauli: &str,
    ) -> NpResult<()> {
        let target = SparsePackedPauli::new(self.n_qubits, qubits, pauli)?;
        let rows = target.anticommuting_rows(&self.x_columns, &self.z_columns, self.row_words);
        self.toggle_sign_constants(&rows);
        Ok(())
    }

    pub(crate) fn apply_single_pauli_string_expr(
        &mut self,
        qubit: usize,
        pauli: &str,
        expr: &Expr,
    ) -> NpResult<()> {
        let (x, z) = single_pauli_bits(qubit, self.n_qubits, pauli)?;
        let column_start = self.column_start(qubit);
        for word_index in 0..self.row_words {
            let rows = (if x {
                self.z_columns[column_start + word_index]
            } else {
                0
            }) ^ (if z {
                self.x_columns[column_start + word_index]
            } else {
                0
            });
            for_each_set_bit_in_word(word_index, rows, |row| self.sign[row].xor_assign(expr));
        }
        Ok(())
    }

    pub(crate) fn measure_single_pauli_expr(
        &mut self,
        qubit: usize,
        pauli: &str,
    ) -> NpResult<Expr> {
        let (x, z) = single_pauli_bits(qubit, self.n_qubits, pauli)?;
        let mut anticommuting_rows = std::mem::take(&mut self.row_scratch);
        let column_start = self.column_start(qubit);
        for (word_index, rows) in anticommuting_rows.iter_mut().enumerate() {
            *rows = (if x {
                self.z_columns[column_start + word_index]
            } else {
                0
            }) ^ (if z {
                self.x_columns[column_start + word_index]
            } else {
                0
            });
        }
        self.synchronize_rows();
        let pivot = first_set_bit_from(&anticommuting_rows, self.n_qubits);
        let result = match pivot {
            Some(pivot) => {
                self.measure_random_single_pauli(pivot, qubit, x, z, &anticommuting_rows)
            }
            None => self.combine_stabilizers_selected_by(|destabilizer| {
                packed_bit(&anticommuting_rows, destabilizer)
            }),
        };
        self.row_scratch = anticommuting_rows;
        result
    }

    pub(crate) fn measure_sparse_pauli_expr(
        &mut self,
        qubits: &[usize],
        pauli: &str,
    ) -> NpResult<Expr> {
        let target = SparsePackedPauli::new(self.n_qubits, qubits, pauli)?;
        let anticommuting_rows =
            target.anticommuting_rows(&self.x_columns, &self.z_columns, self.row_words);
        self.synchronize_rows();
        let pivot = first_set_bit_from(&anticommuting_rows, self.n_qubits);
        match pivot {
            Some(pivot) => self.measure_random_sparse_pauli(pivot, target, &anticommuting_rows),
            None => self.combine_stabilizers_selected_by(|destabilizer| {
                packed_bit(&anticommuting_rows, destabilizer)
            }),
        }
    }

    fn measure_random_sparse_pauli(
        &mut self,
        pivot: usize,
        target: SparsePackedPauli,
        anticommuting_rows: &[u64],
    ) -> NpResult<Expr> {
        self.measure_random_pauli_with_support(
            pivot,
            anticommuting_rows,
            Some((target.x, target.z)),
            None,
        )
    }

    fn measure_random_single_pauli(
        &mut self,
        pivot: usize,
        qubit: usize,
        x: bool,
        z: bool,
        anticommuting_rows: &[u64],
    ) -> NpResult<Expr> {
        self.measure_random_pauli_with_support(pivot, anticommuting_rows, None, Some((qubit, x, z)))
    }

    fn measure_random_pauli_with_support(
        &mut self,
        pivot: usize,
        anticommuting_rows: &[u64],
        target: Option<(Vec<u64>, Vec<u64>)>,
        single_target: Option<(usize, bool, bool)>,
    ) -> NpResult<Expr> {
        let paired_destabilizer = pivot - self.n_qubits;
        let mut pivot_x = std::mem::take(&mut self.support_scratch_x);
        let mut pivot_z = std::mem::take(&mut self.support_scratch_z);
        pivot_x.clone_from_slice(self.x_row(pivot));
        pivot_z.clone_from_slice(self.z_row(pivot));
        let pivot_sign = self.sign[pivot].clone();
        let rowsum_result = for_each_set_bit_result(anticommuting_rows, |row| {
            if row != pivot && row != paired_destabilizer {
                self.rowsum_row_only(row, &pivot_x, &pivot_z, &pivot_sign)?;
            }
            Ok(())
        });
        if let Err(error) = rowsum_result {
            self.support_scratch_x = pivot_x;
            self.support_scratch_z = pivot_z;
            return Err(error);
        }
        self.apply_rowsum_columns_batch(
            anticommuting_rows,
            pivot,
            paired_destabilizer,
            &pivot_x,
            &pivot_z,
        );

        self.replace_row_support(paired_destabilizer, &pivot_x, &pivot_z);
        self.sign[paired_destabilizer] = pivot_sign;
        let source = self.random_source_count;
        self.random_source_count += 1;
        let outcome = Expr::random(source);
        if let Some((target_x, target_z)) = target {
            self.replace_row_support(pivot, &target_x, &target_z);
        } else if let Some((qubit, x, z)) = single_target {
            self.replace_row_support_single(pivot, qubit, x, z);
        }
        self.sign[pivot] = outcome.clone();
        self.support_scratch_x = pivot_x;
        self.support_scratch_z = pivot_z;
        Ok(outcome)
    }

    fn combine_stabilizers_selected_by<F>(&mut self, mut selected: F) -> NpResult<Expr>
    where
        F: FnMut(usize) -> bool,
    {
        let mut acc_x = std::mem::take(&mut self.support_scratch_x);
        let mut acc_z = std::mem::take(&mut self.support_scratch_z);
        acc_x.fill(0);
        acc_z.fill(0);
        let mut acc_sign = Expr::default();
        let mut result = Ok(());
        for destabilizer in 0..self.n_qubits {
            if selected(destabilizer) {
                let stabilizer = self.n_qubits + destabilizer;
                let start = self.row_start(stabilizer);
                result = packed_rowsum(
                    &mut acc_x,
                    &mut acc_z,
                    &mut acc_sign,
                    &self.x[start..start + self.words],
                    &self.z[start..start + self.words],
                    &self.sign[stabilizer],
                );
                if result.is_err() {
                    break;
                }
            }
        }
        self.support_scratch_x = acc_x;
        self.support_scratch_z = acc_z;
        result.map(|_| acc_sign)
    }

    fn toggle_sign_constants(&mut self, rows: &[u64]) {
        for_each_set_bit(rows, |row| self.sign[row].toggle_constant());
    }

    fn rowsum_row_only(
        &mut self,
        target: usize,
        source_x: &[u64],
        source_z: &[u64],
        source_sign: &Expr,
    ) -> NpResult<()> {
        let start = self.row_start(target);
        packed_rowsum(
            &mut self.x[start..start + self.words],
            &mut self.z[start..start + self.words],
            &mut self.sign[target],
            source_x,
            source_z,
            source_sign,
        )
    }

    fn apply_rowsum_columns_batch(
        &mut self,
        anticommuting_rows: &[u64],
        pivot: usize,
        paired_destabilizer: usize,
        source_x: &[u64],
        source_z: &[u64],
    ) {
        let update_word = |word_index: usize| {
            let mut rows = anticommuting_rows[word_index];
            if pivot / 64 == word_index {
                rows &= !(1u64 << (pivot % 64));
            }
            if paired_destabilizer / 64 == word_index {
                rows &= !(1u64 << (paired_destabilizer % 64));
            }
            rows
        };
        for_each_set_bit(source_x, |qubit| {
            let column_start = qubit * self.row_words;
            for word_index in 0..self.row_words {
                self.x_columns[column_start + word_index] ^= update_word(word_index);
            }
        });
        for_each_set_bit(source_z, |qubit| {
            let column_start = qubit * self.row_words;
            for word_index in 0..self.row_words {
                self.z_columns[column_start + word_index] ^= update_word(word_index);
            }
        });
    }

    fn replace_row_support(&mut self, row: usize, new_x: &[u64], new_z: &[u64]) {
        let start = self.row_start(row);
        for word_index in 0..self.words {
            let index = start + word_index;
            let changed_x = self.x[index] ^ new_x[word_index];
            let changed_z = self.z[index] ^ new_z[word_index];
            for_each_set_bit_in_word(word_index, changed_x, |qubit| {
                toggle_flat_column_bit(&mut self.x_columns, self.row_words, qubit, row)
            });
            for_each_set_bit_in_word(word_index, changed_z, |qubit| {
                toggle_flat_column_bit(&mut self.z_columns, self.row_words, qubit, row)
            });
            self.x[index] = new_x[word_index];
            self.z[index] = new_z[word_index];
        }
    }

    fn replace_row_support_single(&mut self, row: usize, qubit: usize, x: bool, z: bool) {
        let target_word = qubit / 64;
        let target_bit = 1u64 << (qubit % 64);
        let start = self.row_start(row);
        for word_index in 0..self.words {
            let new_x = if x && word_index == target_word {
                target_bit
            } else {
                0
            };
            let new_z = if z && word_index == target_word {
                target_bit
            } else {
                0
            };
            let index = start + word_index;
            let changed_x = self.x[index] ^ new_x;
            let changed_z = self.z[index] ^ new_z;
            for_each_set_bit_in_word(word_index, changed_x, |changed_qubit| {
                toggle_flat_column_bit(&mut self.x_columns, self.row_words, changed_qubit, row)
            });
            for_each_set_bit_in_word(word_index, changed_z, |changed_qubit| {
                toggle_flat_column_bit(&mut self.z_columns, self.row_words, changed_qubit, row)
            });
            self.x[index] = new_x;
            self.z[index] = new_z;
        }
    }
}

struct SparsePackedPauli {
    entries: Vec<(usize, bool, bool)>,
    x: Vec<u64>,
    z: Vec<u64>,
}

fn single_pauli_bits(qubit: usize, n_qubits: usize, pauli: &str) -> NpResult<(bool, bool)> {
    if qubit >= n_qubits {
        return Err(NpError::new(format!(
            "qubit {qubit} is out of range for {n_qubits} qubits"
        )));
    }
    match pauli.as_bytes() {
        [b'X'] => Ok((true, false)),
        [b'Y'] => Ok((true, true)),
        [b'Z'] => Ok((false, true)),
        _ => Err(NpError::new(format!(
            "unsupported single-qubit Pauli {pauli:?}"
        ))),
    }
}

fn toggle_flat_row_bit(matrix: &mut [u64], words: usize, row: usize, bit: usize) {
    matrix[row * words + bit / 64] ^= 1u64 << (bit % 64);
}

fn toggle_flat_column_bit(matrix: &mut [u64], row_words: usize, column: usize, row: usize) {
    matrix[column * row_words + row / 64] ^= 1u64 << (row % 64);
}

fn xor_words_in_place(target: &mut [u64], source: &[u64]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target ^= source;
    }
}

fn for_each_set_bit(words: &[u64], mut function: impl FnMut(usize)) {
    for (word_index, word) in words.iter().copied().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            function(word_index * 64 + bit);
            remaining &= remaining - 1;
        }
    }
}

fn for_each_set_bit_in_word(word_index: usize, mut word: u64, mut function: impl FnMut(usize)) {
    while word != 0 {
        let bit = word.trailing_zeros() as usize;
        function(word_index * 64 + bit);
        word &= word - 1;
    }
}

fn for_each_set_bit_result(
    words: &[u64],
    mut function: impl FnMut(usize) -> NpResult<()>,
) -> NpResult<()> {
    for (word_index, word) in words.iter().copied().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            function(word_index * 64 + bit)?;
            remaining &= remaining - 1;
        }
    }
    Ok(())
}

fn first_set_bit_from(words: &[u64], start: usize) -> Option<usize> {
    let mut word_index = start / 64;
    if word_index >= words.len() {
        return None;
    }
    let mut word = words[word_index] & (!0u64 << (start % 64));
    loop {
        if word != 0 {
            return Some(word_index * 64 + word.trailing_zeros() as usize);
        }
        word_index += 1;
        if word_index >= words.len() {
            return None;
        }
        word = words[word_index];
    }
}

impl SparsePackedPauli {
    fn new(n_qubits: usize, qubits: &[usize], pauli: &str) -> NpResult<Self> {
        crate::model::validate_pauli_targets(n_qubits, qubits, pauli, "stabilizer Pauli")?;
        let words = n_qubits.div_ceil(64);
        let mut entries = Vec::with_capacity(qubits.len());
        let mut x = vec![0; words];
        let mut z = vec![0; words];
        for (qubit, local) in qubits.iter().zip(pauli.chars()) {
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

    fn anticommuting_rows(
        &self,
        x_columns: &[u64],
        z_columns: &[u64],
        row_words: usize,
    ) -> Vec<u64> {
        let mut rows = vec![0; row_words];
        for (qubit, x, z) in &self.entries {
            let start = qubit * row_words;
            if *z {
                xor_words_in_place(&mut rows, &x_columns[start..start + row_words]);
            }
            if *x {
                xor_words_in_place(&mut rows, &z_columns[start..start + row_words]);
            }
        }
        rows
    }
}

fn packed_bit(words: &[u64], qubit: usize) -> bool {
    ((words[qubit / 64] >> (qubit % 64)) & 1) != 0
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
                    packed_bit(packed.x_column(qubit), packed_row),
                    dense.x[row][qubit] != 0,
                    "x row {row}, qubit {qubit}"
                );
                assert_eq!(
                    packed_bit(packed.z_column(qubit), packed_row),
                    dense.z[row][qubit] != 0,
                    "z row {row}, qubit {qubit}"
                );
            }
            assert_eq!(packed.sign[packed_row], dense.sign[row], "sign row {row}");
        }
    }

    fn assert_symbolic_states_equal(left: &SymbolicStabilizer, right: &SymbolicStabilizer) {
        assert_eq!(left.n_qubits, right.n_qubits);
        assert_eq!(left.x, right.x);
        assert_eq!(left.z, right.z);
        assert_eq!(left.x_columns, right.x_columns);
        assert_eq!(left.z_columns, right.z_columns);
        assert_eq!(left.rows_valid, right.rows_valid);
        assert_eq!(left.sign, right.sign);
        assert_eq!(left.random_source_count, right.random_source_count);
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
    fn single_qubit_measurement_fast_path_matches_sparse_path() {
        for prepared_with_h in [false, true] {
            for basis in ["X", "Y", "Z"] {
                let mut initial = SymbolicStabilizer::zero(2);
                if prepared_with_h {
                    initial.apply_h(0);
                    initial.apply_cx(0, 1);
                }
                let mut direct = initial.clone();
                let mut sparse = initial;

                let direct_ideal = direct.measure_single_pauli_expr(0, basis).unwrap();
                let sparse_ideal = sparse.measure_sparse_pauli_expr(&[0], basis).unwrap();

                assert_eq!(direct_ideal, sparse_ideal, "basis {basis}");
                assert_symbolic_states_equal(&direct, &sparse);
            }
        }
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
                .apply_single_pauli_string_expr(qubit, pauli, &Expr::constant(true))
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
            .apply_single_pauli_string_expr(0, "X", &packed_random)
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

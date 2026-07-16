#[cfg(test)]
use crate::Expr;
use crate::{NpError, NpResult};

pub fn sparse_pauli_to_xz(
    n_qubits: usize,
    qubits: &[usize],
    pauli: &str,
) -> NpResult<(Vec<u8>, Vec<u8>)> {
    crate::model::validate_pauli_targets(n_qubits, qubits, pauli, "sparse Pauli")?;
    let mut x = vec![0; n_qubits];
    let mut z = vec![0; n_qubits];
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (px, pz) = pauli_to_xz(local)?;
        x[*qubit] ^= px;
        z[*qubit] ^= pz;
    }
    Ok((x, z))
}

pub fn pauli_to_xz(pauli: char) -> NpResult<(u8, u8)> {
    match pauli {
        'I' => Ok((0, 0)),
        'X' => Ok((1, 0)),
        'Y' => Ok((1, 1)),
        'Z' => Ok((0, 1)),
        _ => Err(NpError::new(format!("unsupported Pauli {pauli:?}"))),
    }
}

pub fn xz_to_pauli(x: u8, z: u8) -> char {
    match (x & 1, z & 1) {
        (0, 0) => 'I',
        (1, 0) => 'X',
        (1, 1) => 'Y',
        (0, 1) => 'Z',
        _ => unreachable!(),
    }
}

pub fn symplectic_product(x1: &[u8], z1: &[u8], x2: &[u8], z2: &[u8]) -> u8 {
    let mut acc = 0;
    for (((a_x, a_z), b_x), b_z) in x1.iter().zip(z1).zip(x2).zip(z2) {
        acc ^= (a_x & b_z) ^ (a_z & b_x);
    }
    acc & 1
}

pub(crate) fn sparse_symplectic_product(
    row_x: &[u8],
    row_z: &[u8],
    qubits: &[usize],
    pauli: &str,
) -> NpResult<u8> {
    if qubits.len() != pauli.len() {
        return Err(NpError::new("qubits and paulis must have the same length"));
    }
    let mut acc = 0;
    for (qubit, local) in qubits.iter().zip(pauli.chars()) {
        let (target_x, target_z) = pauli_to_xz(local)?;
        acc ^= (row_x[*qubit] & target_z) ^ (row_z[*qubit] & target_x);
    }
    Ok(acc & 1)
}

pub fn multiply_concrete_rows(
    left_x: &[u8],
    left_z: &[u8],
    left_sign: bool,
    right_x: &[u8],
    right_z: &[u8],
    right_sign: bool,
) -> NpResult<(Vec<u8>, Vec<u8>, bool)> {
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
        return Err(NpError::new(
            "product of stabilizer rows produced a non-Hermitian phase",
        ));
    }
    Ok((out_x, out_z, sign))
}

#[cfg(test)]
pub(crate) fn multiply_symbolic_rows(
    left_x: &[u8],
    left_z: &[u8],
    left_sign: &Expr,
    right_x: &[u8],
    right_z: &[u8],
    right_sign: &Expr,
) -> NpResult<(Vec<u8>, Vec<u8>, Expr)> {
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
        return Err(NpError::new(
            "product of stabilizer rows produced a non-Hermitian phase",
        ));
    }
    Ok((out_x, out_z, out_sign))
}

pub fn pauli_product(left: char, right: char) -> (u8, char) {
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

pub fn support_to_words(x: &[u8], z: &[u8]) -> Vec<u64> {
    let bits = x.len() * 2;
    let mut words = vec![0u64; bits.div_ceil(64)];
    for (idx, bit) in x.iter().chain(z.iter()).enumerate() {
        if *bit != 0 {
            words[idx / 64] |= 1u64 << (idx % 64);
        }
    }
    words
}

pub fn solve_row_span(rows: &[Vec<u64>], target: &[u64], row_count: usize) -> Option<Vec<u64>> {
    let bit_count = rows.first().map(|row| row.len() * 64).unwrap_or(0);
    let coeff_words = row_count.div_ceil(64);
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

pub fn highest_bit(words: &[u64]) -> Option<usize> {
    for (word_idx, word) in words.iter().enumerate().rev() {
        if *word != 0 {
            let local = 63 - word.leading_zeros() as usize;
            return Some(word_idx * 64 + local);
        }
    }
    None
}

pub fn xor_words(left: &mut [u64], right: &[u64]) {
    for (a, b) in left.iter_mut().zip(right) {
        *a ^= *b;
    }
}

pub fn coeff_bit(coeff: &[u64], idx: usize) -> bool {
    ((coeff[idx / 64] >> (idx % 64)) & 1) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_sparse_pauli_to_xz() {
        let (x, z) = sparse_pauli_to_xz(3, &[0, 2], "XZ").unwrap();

        assert_eq!(x, vec![1, 0, 0]);
        assert_eq!(z, vec![0, 0, 1]);
    }

    #[test]
    fn rejects_unknown_pauli() {
        let err = pauli_to_xz('A').unwrap_err();

        assert!(err.message().contains("unsupported Pauli"));
    }

    #[test]
    fn solves_binary_row_span() {
        let rows = vec![vec![0b011], vec![0b110]];
        let target = vec![0b101];

        let coeff = solve_row_span(&rows, &target, 2).unwrap();

        assert!(coeff_bit(&coeff, 0));
        assert!(coeff_bit(&coeff, 1));
    }

    #[test]
    fn sparse_symplectic_product_matches_dense_product() {
        let row_x = vec![1, 0, 1, 1];
        let row_z = vec![0, 1, 1, 0];
        let qubits = vec![0, 2, 3];
        let pauli = "XYZ";
        let (target_x, target_z) = sparse_pauli_to_xz(4, &qubits, pauli).unwrap();

        assert_eq!(
            sparse_symplectic_product(&row_x, &row_z, &qubits, pauli).unwrap(),
            symplectic_product(&row_x, &row_z, &target_x, &target_z),
        );
    }
}

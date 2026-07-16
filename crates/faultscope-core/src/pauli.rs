#[cfg(test)]
use crate::Expr;
use crate::{NpError, NpResult};
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DensePauliRef<'a> {
    x: &'a [u8],
    z: &'a [u8],
}

impl<'a> DensePauliRef<'a> {
    pub(crate) fn new(x: &'a [u8], z: &'a [u8], n_qubits: usize) -> NpResult<Self> {
        if x.len() != n_qubits || z.len() != n_qubits {
            return Err(NpError::new(format!(
                "x and z vectors must each have length {n_qubits}, got x={} and z={}",
                x.len(),
                z.len()
            )));
        }
        validate_binary_vector(x, "x")?;
        validate_binary_vector(z, "z")?;
        Ok(Self { x, z })
    }

    pub(crate) fn from_validated(x: &'a [u8], z: &'a [u8]) -> Self {
        debug_assert_eq!(x.len(), z.len(), "Pauli row widths must match");
        debug_assert!(x.iter().chain(z).all(|bit| *bit <= 1));
        Self { x, z }
    }

    pub(crate) fn x(self) -> &'a [u8] {
        self.x
    }

    pub(crate) fn z(self) -> &'a [u8] {
        self.z
    }
}

fn validate_binary_vector(values: &[u8], name: &str) -> NpResult<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value > 1)
    {
        return Err(NpError::new(format!(
            "{name} vector must contain only 0 or 1, found {value} at index {index}"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SparsePauli<'a> {
    qubits: &'a [usize],
    pauli: &'a [u8],
}

impl<'a> SparsePauli<'a> {
    pub(crate) fn new<C>(
        n_qubits: usize,
        qubits: &'a [usize],
        pauli: &'a str,
        context: &C,
    ) -> NpResult<Self>
    where
        C: fmt::Display + ?Sized,
    {
        crate::model::validate_pauli_targets(n_qubits, qubits, pauli, context)?;
        Ok(Self {
            qubits,
            pauli: pauli.as_bytes(),
        })
    }

    pub(crate) fn len(self) -> usize {
        self.qubits.len()
    }

    pub(crate) fn entries(self) -> impl Iterator<Item = (usize, u8, u8)> + 'a {
        self.qubits
            .iter()
            .copied()
            .zip(self.pauli.iter().copied())
            .map(sparse_entry)
    }
}

fn sparse_entry((qubit, local): (usize, u8)) -> (usize, u8, u8) {
    let (x, z) = match local {
        b'I' => (0, 0),
        b'X' => (1, 0),
        b'Y' => (1, 1),
        b'Z' => (0, 1),
        _ => unreachable!("SparsePauli stores only validated Pauli bytes"),
    };
    (qubit, x, z)
}

#[cfg(test)]
pub(crate) fn sparse_pauli_to_xz(
    n_qubits: usize,
    qubits: &[usize],
    pauli: &str,
) -> NpResult<(Vec<u8>, Vec<u8>)> {
    let target = SparsePauli::new(n_qubits, qubits, pauli, "sparse Pauli")?;
    let mut x = vec![0; n_qubits];
    let mut z = vec![0; n_qubits];
    for (qubit, local_x, local_z) in target.entries() {
        x[qubit] = local_x;
        z[qubit] = local_z;
    }
    Ok((x, z))
}

pub(crate) fn pauli_to_xz(pauli: char) -> NpResult<(u8, u8)> {
    match pauli {
        'I' => Ok((0, 0)),
        'X' => Ok((1, 0)),
        'Y' => Ok((1, 1)),
        'Z' => Ok((0, 1)),
        _ => Err(NpError::new(format!("unsupported Pauli {pauli:?}"))),
    }
}

pub(crate) fn xz_to_pauli(x: u8, z: u8) -> char {
    debug_assert!(x <= 1 && z <= 1, "Pauli bits must be binary");
    match (x, z) {
        (0, 0) => 'I',
        (1, 0) => 'X',
        (1, 1) => 'Y',
        (0, 1) => 'Z',
        _ => unreachable!(),
    }
}

pub(crate) fn symplectic_product_bits(x1: &[u8], z1: &[u8], x2: &[u8], z2: &[u8]) -> u8 {
    debug_assert_eq!(x1.len(), z1.len(), "Pauli row widths must match");
    debug_assert_eq!(x1.len(), x2.len(), "Pauli row widths must match");
    debug_assert_eq!(x1.len(), z2.len(), "Pauli row widths must match");
    let mut acc = 0;
    for (((a_x, a_z), b_x), b_z) in x1.iter().zip(z1).zip(x2).zip(z2) {
        acc ^= (a_x & b_z) ^ (a_z & b_x);
    }
    acc & 1
}

pub(crate) fn sparse_symplectic_product_bits(
    row_x: &[u8],
    row_z: &[u8],
    target: SparsePauli<'_>,
) -> u8 {
    debug_assert_eq!(row_x.len(), row_z.len(), "Pauli row widths must match");
    let mut acc = 0;
    for (qubit, target_x, target_z) in target.entries() {
        acc ^= (row_x[qubit] & target_z) ^ (row_z[qubit] & target_x);
    }
    acc & 1
}

pub(crate) fn multiply_concrete_rows(
    left_x: &[u8],
    left_z: &[u8],
    left_sign: bool,
    right_x: &[u8],
    right_z: &[u8],
    right_sign: bool,
) -> NpResult<(Vec<u8>, Vec<u8>, bool)> {
    debug_assert_eq!(left_x.len(), left_z.len(), "Pauli row widths must match");
    debug_assert_eq!(left_x.len(), right_x.len(), "Pauli row widths must match");
    debug_assert_eq!(left_x.len(), right_z.len(), "Pauli row widths must match");
    let mut phase = 0u8;
    let mut out_x = Vec::with_capacity(left_x.len());
    let mut out_z = Vec::with_capacity(left_x.len());
    for (((lx, lz), rx), rz) in left_x.iter().zip(left_z).zip(right_x).zip(right_z) {
        let lp = xz_to_pauli(*lx, *lz);
        let rp = xz_to_pauli(*rx, *rz);
        let (local_phase, product) = pauli_product(lp, rp);
        phase = (phase + local_phase) & 3;
        let (px, pz) = pauli_to_xz(product).expect("Pauli products are valid");
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
    debug_assert_eq!(left_x.len(), left_z.len(), "Pauli row widths must match");
    debug_assert_eq!(left_x.len(), right_x.len(), "Pauli row widths must match");
    debug_assert_eq!(left_x.len(), right_z.len(), "Pauli row widths must match");
    let mut phase = 0u8;
    let mut out_x = Vec::with_capacity(left_x.len());
    let mut out_z = Vec::with_capacity(left_x.len());
    for (((lx, lz), rx), rz) in left_x.iter().zip(left_z).zip(right_x).zip(right_z) {
        let lp = xz_to_pauli(*lx, *lz);
        let rp = xz_to_pauli(*rx, *rz);
        let (local_phase, product) = pauli_product(lp, rp);
        phase = (phase + local_phase) & 3;
        let (px, pz) = pauli_to_xz(product).expect("Pauli products are valid");
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

pub(crate) fn pauli_product(left: char, right: char) -> (u8, char) {
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

pub(crate) fn support_to_words(x: &[u8], z: &[u8]) -> Vec<u64> {
    debug_assert_eq!(x.len(), z.len(), "Pauli row widths must match");
    debug_assert!(x.iter().chain(z).all(|bit| *bit <= 1));
    let bits = x.len() * 2;
    let mut words = vec![0u64; bits.div_ceil(64)];
    for (idx, bit) in x.iter().chain(z.iter()).enumerate() {
        if *bit != 0 {
            words[idx / 64] |= 1u64 << (idx % 64);
        }
    }
    words
}

pub(crate) fn solve_row_span(
    rows: &[Vec<u64>],
    target: &[u64],
    row_count: usize,
) -> Option<Vec<u64>> {
    debug_assert!(
        row_count >= rows.len(),
        "row_count must be at least the number of rows"
    );
    let word_count = rows.first().map(Vec::len).unwrap_or(target.len());
    debug_assert!(
        rows.iter().all(|row| row.len() == word_count) && target.len() == word_count,
        "row-span word counts must match"
    );
    let bit_count = word_count * 64;
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
    debug_assert_eq!(left.len(), right.len(), "word counts must match");
    for (a, b) in left.iter_mut().zip(right) {
        *a ^= *b;
    }
}

pub(crate) fn coeff_bit(coeff: &[u64], idx: usize) -> bool {
    debug_assert!(idx / 64 < coeff.len(), "coefficient index out of range");
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
        let sparse = SparsePauli::new(4, &qubits, pauli, "test Pauli").unwrap();

        assert_eq!(
            sparse_symplectic_product_bits(&row_x, &row_z, sparse),
            symplectic_product_bits(&row_x, &row_z, &target_x, &target_z),
        );
    }

    #[test]
    fn validates_dense_pauli_support() {
        DensePauliRef::new(&[], &[], 0).unwrap();
        DensePauliRef::new(&[0, 1], &[1, 0], 2).unwrap();

        let length_error = DensePauliRef::new(&[1], &[0], 2).unwrap_err();
        assert!(length_error.message().contains("each have length 2"));
        let binary_error = DensePauliRef::new(&[2], &[0], 1).unwrap_err();
        assert!(binary_error.message().contains("only 0 or 1"));
    }

    #[test]
    fn sparse_pauli_stores_validated_bits() {
        let target = SparsePauli::new(3, &[0, 2], "XY", "test Pauli").unwrap();
        assert_eq!(
            target.entries().collect::<Vec<_>>(),
            vec![(0, 1, 0), (2, 1, 1)]
        );

        assert!(SparsePauli::new(3, &[0, 0], "XY", "test Pauli").is_err());
        assert!(SparsePauli::new(3, &[3], "X", "test Pauli").is_err());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "word counts must match")]
    fn xor_words_debug_asserts_mismatched_widths() {
        xor_words(&mut [0], &[0, 1]);
    }
}

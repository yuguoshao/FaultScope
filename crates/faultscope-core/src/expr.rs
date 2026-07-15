use crate::Mask;
use smallvec::{smallvec, SmallVec};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Expr {
    terms: SmallVec<[usize; 4]>,
    constant: bool,
}

impl Expr {
    pub fn constant(value: bool) -> Self {
        Self {
            terms: SmallVec::new(),
            constant: value,
        }
    }

    pub fn random(source: usize) -> Self {
        Self {
            terms: smallvec![source],
            constant: false,
        }
    }

    pub fn terms(&self) -> &[usize] {
        &self.terms
    }

    pub fn constant_value(&self) -> bool {
        self.constant
    }

    pub fn xor_assign(&mut self, other: &Expr) {
        self.constant ^= other.constant;
        if other.terms.is_empty() {
            return;
        }
        if self.terms.is_empty() {
            self.terms.clone_from(&other.terms);
            return;
        }
        if self.terms.len() + other.terms.len() <= 4 {
            for term in other.terms.iter().copied() {
                self.toggle_term(term);
            }
            return;
        }
        let left = std::mem::take(&mut self.terms);
        let mut out = SmallVec::with_capacity(left.len() + other.terms.len());
        let mut left_idx = 0;
        let mut right_idx = 0;
        while left_idx < left.len() && right_idx < other.terms.len() {
            match left[left_idx].cmp(&other.terms[right_idx]) {
                std::cmp::Ordering::Less => {
                    out.push(left[left_idx]);
                    left_idx += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(other.terms[right_idx]);
                    right_idx += 1;
                }
                std::cmp::Ordering::Equal => {
                    left_idx += 1;
                    right_idx += 1;
                }
            }
        }
        out.extend_from_slice(&left[left_idx..]);
        out.extend_from_slice(&other.terms[right_idx..]);
        self.terms = out;
    }

    pub(crate) fn toggle_term(&mut self, term: usize) {
        match self.terms.binary_search(&term) {
            Ok(index) => {
                self.terms.remove(index);
            }
            Err(index) => self.terms.insert(index, term),
        }
    }

    pub fn toggle_constant(&mut self) {
        self.constant ^= true;
    }

    pub fn eval(&self, random_masks: &[Mask], words: usize, all_mask: &Mask) -> Mask {
        let mut out = Mask::zero(words);
        if self.constant {
            out.xor_assign(all_mask);
        }
        for source in &self.terms {
            out.xor_assign(&random_masks[*source]);
        }
        out
    }

    pub(crate) fn eval_flat(&self, random_masks: &[u64], words: usize, all_mask: &Mask) -> Mask {
        let mut out = Mask::zero(words);
        if self.constant {
            out.xor_assign(all_mask);
        }
        for source in &self.terms {
            let start = source * words;
            for (target, random) in out
                .words
                .iter_mut()
                .zip(&random_masks[start..start + words])
            {
                *target ^= random;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_assign_cancels_duplicate_terms() {
        let mut expr = Expr::random(2);
        expr.xor_assign(&Expr::random(1));
        expr.xor_assign(&Expr::random(2));

        assert_eq!(expr.terms(), &[1]);
        assert!(!expr.constant_value());
    }

    #[test]
    fn xor_assign_merges_sorted_terms_by_symmetric_difference() {
        let mut left = Expr {
            terms: vec![1, 3, 5, 8].into(),
            constant: true,
        };
        let right = Expr {
            terms: vec![0, 3, 4, 8, 9].into(),
            constant: true,
        };

        left.xor_assign(&right);

        assert_eq!(left.terms(), &[0, 1, 4, 5, 9]);
        assert!(!left.constant_value());
    }

    #[test]
    fn xor_assign_matches_randomized_symmetric_differences() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..128 {
            let mut left_terms = Vec::new();
            let mut right_terms = Vec::new();
            let mut expected = Vec::new();
            for term in 0..96 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let in_left = state >> 63 != 0;
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let in_right = state >> 63 != 0;
                if in_left {
                    left_terms.push(term);
                }
                if in_right {
                    right_terms.push(term);
                }
                if in_left ^ in_right {
                    expected.push(term);
                }
            }
            let mut left = Expr {
                terms: left_terms.into(),
                constant: false,
            };
            left.xor_assign(&Expr {
                terms: right_terms.into(),
                constant: true,
            });
            assert_eq!(left.terms(), expected);
            assert!(left.constant_value());
        }
    }

    #[test]
    fn eval_includes_constant_mask() {
        let expr = Expr::constant(true);
        let all = Mask { words: vec![0b101] };

        assert_eq!(expr.eval(&[], 1, &all), all);
    }
}

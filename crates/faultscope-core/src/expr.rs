use crate::Mask;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Expr {
    terms: Vec<usize>,
    constant: bool,
}

impl Expr {
    pub fn constant(value: bool) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
        }
    }

    pub fn random(source: usize) -> Self {
        Self {
            terms: vec![source],
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
    fn eval_includes_constant_mask() {
        let expr = Expr::constant(true);
        let all = Mask { words: vec![0b101] };

        assert_eq!(expr.eval(&[], 1, &all), all);
    }
}

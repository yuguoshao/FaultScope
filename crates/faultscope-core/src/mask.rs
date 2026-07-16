#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub words: Vec<u64>,
}

impl Mask {
    pub fn zero(words: usize) -> Self {
        Self {
            words: vec![0; words],
        }
    }

    pub fn all(shots: usize) -> Self {
        let words = word_count(shots);
        let mut mask = Self {
            words: vec![u64::MAX; words],
        };
        mask.clear_unused(shots);
        mask
    }

    pub fn xor_assign(&mut self, other: &Mask) {
        self.assert_same_width(other);
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left ^= *right;
        }
    }

    pub fn or_assign(&mut self, other: &Mask) {
        self.assert_same_width(other);
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left |= *right;
        }
    }

    pub fn and_assign(&mut self, other: &Mask) {
        self.assert_same_width(other);
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left &= *right;
        }
    }

    pub fn bit_count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn and_count(&self, other: &Mask) -> usize {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(left, right)| (left & right).count_ones() as usize)
            .sum()
    }

    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    pub fn clear_unused(&mut self, shots: usize) {
        let extra = shots % 64;
        if extra != 0 {
            let keep = (1u64 << extra) - 1;
            if let Some(last) = self.words.last_mut() {
                *last &= keep;
            }
        }
    }

    fn assert_same_width(&self, other: &Mask) {
        assert_eq!(
            self.words.len(),
            other.words.len(),
            "mask word counts must match"
        );
    }
}

pub fn word_count(shots: usize) -> usize {
    shots.div_ceil(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_clears_unused_bits() {
        let mask = Mask::all(65);

        assert_eq!(mask.words, vec![u64::MAX, 1]);
    }

    #[test]
    fn counts_and_intersections() {
        let left = Mask {
            words: vec![0b1011],
        };
        let right = Mask {
            words: vec![0b1100],
        };

        assert_eq!(left.bit_count(), 3);
        assert_eq!(left.and_count(&right), 1);
    }

    #[test]
    #[should_panic(expected = "mask word counts must match")]
    fn xor_assign_rejects_mismatched_widths() {
        Mask::zero(1).xor_assign(&Mask::zero(2));
    }

    #[test]
    #[should_panic(expected = "mask word counts must match")]
    fn or_assign_rejects_mismatched_widths() {
        Mask::zero(1).or_assign(&Mask::zero(2));
    }

    #[test]
    #[should_panic(expected = "mask word counts must match")]
    fn and_assign_rejects_mismatched_widths() {
        Mask::zero(1).and_assign(&Mask::zero(2));
    }
}

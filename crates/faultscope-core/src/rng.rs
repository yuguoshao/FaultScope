use rand::rngs::SmallRng as RandSmallRng;
use rand::{RngCore, SeedableRng};

pub struct SmallRng {
    inner: RandSmallRng,
}

impl SmallRng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: RandSmallRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    pub fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_is_reproducible() {
        let mut left = SmallRng::new(123);
        let mut right = SmallRng::new(123);

        assert_eq!(left.next_u64(), right.next_u64());
        assert_eq!(left.next_u64(), right.next_u64());
    }

    #[test]
    fn next_f64_is_in_unit_interval() {
        let mut rng = SmallRng::new(456);

        for _ in 0..100 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value));
        }
    }
}

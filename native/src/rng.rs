use crate::*;

pub(crate) struct SmallRng {
    pub(crate) inner: RandSmallRng,
}

impl SmallRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            inner: RandSmallRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15),
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

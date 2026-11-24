//! SplitMix64: a small, fast, splittable PRNG. Not cryptographic.

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, n) via rejection sampling (no modulo bias). Requires n > 0.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below: n must be > 0");
        let limit = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < limit {
                return x % n;
            }
        }
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        let i = self.below(xs.len() as u64);
        &xs[i as usize]
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        let n = xs.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.below((i + 1) as u64) as usize;
            xs.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed0_matches_reference() {
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
        assert_eq!(r.next_u64(), 0x6E789E6AA1B965F4);
        assert_eq!(r.next_u64(), 0x06C45D188009454F);
    }

    #[test]
    fn below_is_within_bounds() {
        let mut r = SplitMix64::new(42);
        for _ in 0..10_000 {
            let x = r.below(7);
            assert!(x < 7);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = SplitMix64::new(7);
        let mut xs: Vec<u32> = (0..20).collect();
        r.shuffle(&mut xs);
        let mut sorted = xs.clone();
        sorted.sort();
        assert_eq!(sorted, (0..20).collect::<Vec<u32>>());
    }

    #[test]
    fn pick_returns_an_element() {
        let mut r = SplitMix64::new(1);
        let xs = [10, 20, 30];
        for _ in 0..100 {
            let v = *r.pick(&xs);
            assert!(xs.contains(&v));
        }
    }
}

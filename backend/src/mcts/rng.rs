//! 재현성 모드 — 결정론적 RNG
//!
//! 전략:
//!   - 병렬 탐색: 각 스레드가 `seed ^ thread_id` 로 독립 RNG 초기화
//!   - 단일 스레드: `seed` 그대로 사용
//!   - seed=None: `thread_rng()` (비결정적)
//!
//! 사용법:
//!   let rng = MctsRng::new(config.seed, thread_id);
//!   let mv  = rng.choose(&legal_moves);

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// 단일 평가 호출용 경량 RNG 컨텍스트
pub enum MctsRng {
    Seeded(StdRng),
    Random,
}

impl MctsRng {
    /// seed=None → random, seed=Some(s) → s XOR thread_id
    pub fn new(seed: Option<u64>, thread_id: usize) -> Self {
        match seed {
            Some(s) => MctsRng::Seeded(StdRng::seed_from_u64(s ^ thread_id as u64)),
            None => MctsRng::Random,
        }
    }

    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        match self {
            MctsRng::Seeded(rng) => slice.choose(rng),
            MctsRng::Random => slice.choose(&mut rand::thread_rng()),
        }
    }

    pub fn gen_range_u64(&mut self, n: u64) -> u64 {
        match self {
            MctsRng::Seeded(rng) => rng.gen_range(0..n),
            MctsRng::Random => rand::thread_rng().gen_range(0..n),
        }
    }

    pub fn sample_weighted_index(&mut self, weights: &[f64]) -> Option<usize> {
        if weights.is_empty() {
            return None;
        }
        let total: f64 = weights.iter().copied().sum();
        if total < 1e-12 {
            return Some(self.gen_range_u64(weights.len() as u64) as usize);
        }
        let threshold = (self.gen_range_u64(1u64 << 53) as f64 / (1u64 << 53) as f64) * total;
        let mut cumulative = 0.0;
        for (idx, weight) in weights.iter().copied().enumerate() {
            cumulative += weight;
            if threshold <= cumulative {
                return Some(idx);
            }
        }
        Some(weights.len().saturating_sub(1))
    }
}

//! MCTS 공유 타입

use serde::{Deserialize, Serialize};

/// k(N) = max(1, floor(α · N^β))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PwConfig {
    pub alpha: f32,
    pub beta: f32,
}

impl PwConfig {
    pub fn new(alpha: f32, beta: f32) -> Self {
        PwConfig { alpha, beta }
    }
    pub fn default_gomoku() -> Self {
        PwConfig {
            alpha: 2.0,
            beta: 0.5,
        }
    }
    #[inline]
    pub fn k(&self, n: u32) -> usize {
        (self.alpha * (n as f32).powf(self.beta)).floor() as usize
    }
}

impl Default for PwConfig {
    fn default() -> Self {
        Self::default_gomoku()
    }
}

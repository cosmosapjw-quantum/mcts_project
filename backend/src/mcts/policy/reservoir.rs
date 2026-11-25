//! BQ++ Phase 6: Nested-reservoir live-set maintenance.
//!
//! Per the audit (§6.4), this is "nested-reservoir search" — borrowing
//! the live-set + threshold maintenance idea from Skilling 2006 nested
//! sampling — and explicitly NOT nested sampling for evidence
//! estimation. The reservoir keeps a set of candidate arms ranked by
//!
//!     Λ_a = U_a + ρ · KG_a + τ · log π̃_0(a)
//!
//! and prunes arms whose Λ falls below a quantile threshold. Pruned
//! arms enter a cooldown to prevent thrashing on borderline candidates.
//! Replenishment from unexplored / low-prior / high-uncertainty arms
//! happens via Gumbel sampling (Phase 3 primitive).
//!
//! The Python prototype at `prototype/bqpp_prototype/reservoir.py`
//! validates the math; this is the Rust port.

use smallvec::SmallVec;
use std::collections::HashMap;

/// Compute the lambda score for an arm.
///
/// Formula: `Λ_a = U_a + ρ · KG_a + τ · log π̃_0(a)`
///
/// Components:
/// - `U_a`: Empirical Bernstein upper CI (from Phase 2 PolicyCache).
///   The "this arm could plausibly become the best" signal.
/// - `KG_a`: Knowledge Gradient (from Phase 4). The "value of one
///   more pull" signal.
/// - `log π̃_0(a)`: Smoothed log prior. The "network's recommendation"
///   signal.
/// - `rho`, `tau`: relative weights. The audit recommends
///   `rho = 1.0` (certification phase) or `1.5` (exploration phase),
///   and `tau` = entropy temperature derived from budget + uncertainty.
#[inline]
pub fn lambda_score(upper_ci: f32, kg: f32, log_prior_smoothed: f32, rho: f32, tau: f32) -> f32 {
    upper_ci + rho * kg + tau * log_prior_smoothed
}

/// Compute the q-th quantile (linear interpolation between data points).
///
/// Same formula as Python's `numpy.quantile(method='linear')` and the
/// prototype's `quantile()`. Used for the bottom-quantile threshold for
/// live-set pruning.
pub fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: SmallVec<[f32; 32]> = SmallVec::from_slice(values);
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f32 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Live-set with quantile pruning + cooldown hysteresis.
///
/// `live`: currently-live edge-local positions, ordered by insertion.
/// `cooldown_until`: per-arm cooldown expiry iteration. An arm just
/// removed cannot re-enter until `current_iter >= cooldown_until[arm]`.
pub struct Reservoir {
    pub live: SmallVec<[u16; 32]>,
    pub cooldown_until: HashMap<u16, u32>,
    pub max_size: usize,
    pub cooldown_iters: u32,
}

impl Reservoir {
    pub fn new(max_size: usize, cooldown_iters: u32) -> Self {
        Self {
            live: SmallVec::new(),
            cooldown_until: HashMap::new(),
            max_size,
            cooldown_iters,
        }
    }

    /// Default per the audit: max_size=16, cooldown=200 (= 2 × default
    /// check_interval).
    pub fn default_for_budget(_budget: u32) -> Self {
        Self::new(16, 200)
    }

    /// Returns true iff `idx` is eligible to (re-)enter the live set.
    pub fn is_eligible(&self, idx: u16, current_iter: u32) -> bool {
        match self.cooldown_until.get(&idx) {
            Some(&until) => until <= current_iter,
            None => true,
        }
    }

    /// Add `idx` to the live set. Returns true iff added.
    /// Fails when: already live, max_size reached, or in cooldown.
    pub fn add(&mut self, idx: u16, current_iter: u32) -> bool {
        if self.live.contains(&idx) {
            return false;
        }
        if self.live.len() >= self.max_size {
            return false;
        }
        if !self.is_eligible(idx, current_iter) {
            return false;
        }
        self.live.push(idx);
        true
    }

    /// Remove `idx` from the live set and start its cooldown.
    /// Returns true iff removed.
    pub fn remove(&mut self, idx: u16, current_iter: u32) -> bool {
        let pos = self.live.iter().position(|&x| x == idx);
        match pos {
            Some(p) => {
                self.live.remove(p);
                self.cooldown_until
                    .insert(idx, current_iter.saturating_add(self.cooldown_iters));
                true
            }
            None => false,
        }
    }

    /// Remove all live arms with score below the q-th quantile.
    /// Returns the list of removed indices in removal order.
    ///
    /// `scores` must contain a value for every live index; missing
    /// entries are treated as `-inf` (always pruned). Strict <
    /// comparison: arms with score == quantile are kept.
    pub fn prune_below_quantile(
        &mut self,
        scores: &HashMap<u16, f32>,
        q: f32,
        current_iter: u32,
    ) -> SmallVec<[u16; 32]> {
        if self.live.is_empty() {
            return SmallVec::new();
        }
        // Compute threshold using only arms with explicit scores; arms
        // without a score entry are *automatically* pruned. This avoids
        // the degenerate case where a -inf in the live_scores collapses
        // the quantile to -inf and prevents any strict-less-than match.
        let scored_values: SmallVec<[f32; 32]> = self
            .live
            .iter()
            .filter_map(|i| scores.get(i).copied())
            .collect();
        let threshold = if scored_values.is_empty() {
            f32::NEG_INFINITY
        } else {
            quantile(&scored_values, q)
        };
        let to_remove: SmallVec<[u16; 32]> = self
            .live
            .iter()
            .copied()
            .filter(|i| match scores.get(i) {
                Some(&s) => s < threshold,
                None => true, // missing score ⇒ always prune
            })
            .collect();
        for idx in &to_remove {
            self.remove(*idx, current_iter);
        }
        to_remove
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 6: lambda_score = U + ρ·KG + τ·log π. Hand-derived.
    #[test]
    fn test_phase6_lambda_score_components() {
        // U=0.6, KG=0.05, log_prior=-1.0, ρ=1.0, τ=0.1
        // ⇒ 0.6 + 1.0·0.05 + 0.1·(−1.0) = 0.55
        let s = lambda_score(0.6, 0.05, -1.0, 1.0, 0.1);
        assert!((s - 0.55).abs() < 1e-6);
    }

    /// Phase 6: quantile of single value is that value.
    #[test]
    fn test_phase6_quantile_single_value() {
        assert_eq!(quantile(&[5.0], 0.25), 5.0);
    }

    /// Phase 6: quantile of two values, linear interpolation.
    #[test]
    fn test_phase6_quantile_two_values() {
        assert_eq!(quantile(&[0.0, 10.0], 0.5), 5.0);
    }

    /// Phase 6: quantile of [1, 2, 3, 4] at q=0.25.
    /// pos = 0.25 * 3 = 0.75. lo=0, hi=1. frac=0.75.
    /// sorted[0]·0.25 + sorted[1]·0.75 = 1.0·0.25 + 2.0·0.75 = 1.75.
    #[test]
    fn test_phase6_quantile_clean_25th() {
        let q = quantile(&[1.0, 2.0, 3.0, 4.0], 0.25);
        assert!((q - 1.75).abs() < 1e-6);
    }

    /// Phase 6: empty quantile returns 0.
    #[test]
    fn test_phase6_quantile_empty() {
        assert_eq!(quantile(&[], 0.5), 0.0);
    }

    /// Phase 6: reservoir add respects max_size.
    #[test]
    fn test_phase6_reservoir_add_respects_max_size() {
        let mut r = Reservoir::new(2, 10);
        assert!(r.add(0, 0));
        assert!(r.add(1, 0));
        // Full; cannot add a 3rd.
        assert!(!r.add(2, 0));
    }

    /// Phase 6: removal starts cooldown.
    #[test]
    fn test_phase6_reservoir_remove_starts_cooldown() {
        let mut r = Reservoir::new(4, 10);
        r.add(0, 0);
        r.remove(0, 5);
        // Cooldown until iter 15.
        assert!(!r.is_eligible(0, 10));
        assert!(!r.is_eligible(0, 14));
        assert!(r.is_eligible(0, 15));
    }

    /// Phase 6: quantile pruning removes bottom arms.
    #[test]
    fn test_phase6_reservoir_quantile_pruning() {
        let mut r = Reservoir::new(10, 5);
        for i in 0..4_u16 {
            r.add(i, 0);
        }
        let mut scores = HashMap::new();
        scores.insert(0, 0.9);
        scores.insert(1, 0.7);
        scores.insert(2, 0.5);
        scores.insert(3, 0.3);
        // q=0.25 of [0.3, 0.5, 0.7, 0.9] = 0.3 + 0.75 * 0.2 = 0.45.
        // Strict <: arm with score 0.3 is removed.
        let removed = r.prune_below_quantile(&scores, 0.25, 10);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0], 3);
        let mut live: Vec<u16> = r.live.to_vec();
        live.sort();
        assert_eq!(live, vec![0, 1, 2]);
    }

    /// Phase 6: anti-thrashing — just-removed arm cannot re-enter
    /// within cooldown_iters.
    #[test]
    fn test_phase6_reservoir_no_thrashing() {
        let mut r = Reservoir::new(4, 200);
        r.add(0, 0);
        r.remove(0, 100);
        // 50 iters later, still in cooldown.
        assert!(!r.add(0, 150));
        // After cooldown expires, can re-enter.
        assert!(r.add(0, 301));
    }

    /// Phase 6: empty reservoir prune_below_quantile is a no-op.
    #[test]
    fn test_phase6_reservoir_empty_prune_noop() {
        let mut r = Reservoir::new(10, 5);
        let scores = HashMap::new();
        let removed = r.prune_below_quantile(&scores, 0.25, 0);
        assert!(removed.is_empty());
    }

    /// Phase 6: arm without a score entry is treated as -inf and pruned.
    #[test]
    fn test_phase6_reservoir_missing_score_pruned() {
        let mut r = Reservoir::new(4, 5);
        r.add(0, 0);
        r.add(1, 0);
        let mut scores = HashMap::new();
        scores.insert(0, 0.9);
        // Arm 1 has no score ⇒ -inf ⇒ pruned at any q > 0.
        let removed = r.prune_below_quantile(&scores, 0.5, 10);
        assert!(removed.contains(&1));
    }
}

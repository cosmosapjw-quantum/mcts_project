//! BQ++ Phase 7: MENTS — Maximum Entropy Monte-Carlo Tree Search.
//!
//! Reference: Xiao, C., Mei, J., Müller, M., & Schuurmans, D. (2019).
//! "Maximum Entropy Monte-Carlo Planning." NeurIPS 2019.
//!
//! Soft-Bellman backups + entropy-regularized policy. Useful for
//! single-agent / non-zero-sum contexts where soft-Bellman backups
//! are appropriate. **NOT the default for AlphaZero zero-sum games**.
//!
//! This module ships the soft-Bellman primitives:
//! - `soft_value(q_values, tau)` — `V_soft(s) = τ · log Σ_a exp(Q_a / τ)`.
//! - `soft_policy(q_values, n_state, tau, epsilon)` — `π_soft(a | s)
//!   = (1 − λ_s) · softmax(Q / τ) + λ_s / K`, with `λ_s = ε · K /
//!   log(2 + N(s))`.
//! - `kl_visit_to_soft(visit_counts, soft_policy)` — `KL(π_visit ‖ π_soft)`,
//!   used as the convergence criterion.
//!
//! The MENTS halt rule: stop when `KL(π_visit ‖ π_soft) < kl_threshold`.

use smallvec::SmallVec;
use std::f32;

/// Soft-Bellman value V_soft(s) = τ · log Σ_a exp(Q_a / τ).
///
/// Numerically stable via log-sum-exp shift by max(Q):
///     V_soft = max(Q) + τ · log Σ_a exp((Q_a − max(Q)) / τ).
///
/// Returns the soft value. Edge cases:
///   τ ≤ 0           ⇒ behaves as max(Q) (no entropy regularization).
///   empty input     ⇒ −inf.
pub fn soft_value(q_values: &[f32], tau: f32) -> f32 {
    if q_values.is_empty() {
        return f32::NEG_INFINITY;
    }
    if tau <= 0.0 {
        return q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    }
    let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for &q in q_values {
        sum += ((q - max_q) / tau).exp();
    }
    max_q + tau * sum.ln()
}

/// Soft policy with exploration-floor smoothing.
///
/// Formula (Xiao et al. 2019, Algorithm 1):
///     λ_s = ε · K / log(2 + N(s))
///     π_soft(a | s) = (1 − λ_s) · softmax(Q_a / τ) + λ_s / K
///
/// where:
///   K = number of actions.
///   N(s) = total visit count at state s.
///   ε = exploration floor (default 0.1 per the paper).
///   τ = entropy temperature (default 0.01).
///
/// Returns the soft-policy probabilities, len() == q_values.len().
/// Numerically stable via the same log-sum-exp shift as `soft_value`.
pub fn soft_policy(q_values: &[f32], n_state: u32, tau: f32, epsilon: f32) -> SmallVec<[f32; 32]> {
    let k = q_values.len();
    let mut out: SmallVec<[f32; 32]> = SmallVec::new();
    if k == 0 {
        return out;
    }
    if tau <= 0.0 {
        // Degenerate: all mass on argmax.
        let argmax = (0..k)
            .max_by(|&a, &b| {
                q_values[a]
                    .partial_cmp(&q_values[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        for i in 0..k {
            out.push(if i == argmax { 1.0 } else { 0.0 });
        }
        return out;
    }
    let lambda_s = epsilon * (k as f32) / (2.0 + n_state as f32).ln();
    let lambda_s = lambda_s.clamp(0.0, 1.0);
    let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: SmallVec<[f32; 32]> = SmallVec::new();
    let mut sum = 0.0_f32;
    for &q in q_values {
        let e = ((q - max_q) / tau).exp();
        exps.push(e);
        sum += e;
    }
    let inv_k = 1.0 / k as f32;
    for &e in &exps {
        let softmax_a = if sum > 0.0 { e / sum } else { inv_k };
        let p = (1.0 - lambda_s) * softmax_a + lambda_s * inv_k;
        out.push(p);
    }
    out
}

/// KL divergence KL(π_visit ‖ π_soft).
///
/// Formula:
///     KL(p ‖ q) = Σ_a p_a · log(p_a / q_a)   (terms with p_a = 0 contribute 0).
///
/// Used as the MENTS convergence criterion: when this is below the
/// kl_threshold (default 1e-3), the search has converged.
///
/// Numerical stability: clamps q at 1e-12 to avoid log(0).
pub fn kl_visit_to_soft(visit_counts: &[u32], soft_policy: &[f32]) -> f32 {
    if visit_counts.len() != soft_policy.len() || visit_counts.is_empty() {
        return 0.0;
    }
    let n_total: u32 = visit_counts.iter().sum();
    if n_total == 0 {
        return 0.0;
    }
    let inv_n = 1.0 / n_total as f32;
    let mut kl = 0.0_f32;
    for (i, &n_a) in visit_counts.iter().enumerate() {
        if n_a == 0 {
            continue; // p_a = 0 ⇒ term is 0.
        }
        let p = n_a as f32 * inv_n;
        let q = soft_policy[i].max(1e-12);
        kl += p * (p / q).ln();
    }
    kl
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 7: V_soft = max(Q) at τ → 0.
    #[test]
    fn test_phase7_soft_value_at_zero_tau() {
        let q = [0.5_f32, 0.3, 0.7];
        let v = soft_value(&q, 0.0);
        assert!((v - 0.7).abs() < 1e-6);
    }

    /// Phase 7: V_soft = log_sum_exp(Q) at τ = 1.
    #[test]
    fn test_phase7_soft_value_at_unit_tau() {
        let q = [0.0_f32, 0.0, 0.0];
        let v = soft_value(&q, 1.0);
        // log(3 · exp(0)) = log(3) ≈ 1.0986
        assert!((v - 3.0_f32.ln()).abs() < 1e-5);
    }

    /// Phase 7: V_soft = max(Q) + small entropy bonus for nearly-tied Q.
    #[test]
    fn test_phase7_soft_value_entropy_bonus() {
        let q = [0.5_f32, 0.495];
        let v_max = 0.5_f32;
        // V_soft > 0.5 because of the entropy contribution.
        let v_soft = soft_value(&q, 0.01);
        assert!(v_soft > v_max);
    }

    /// Phase 7: empty input → -inf.
    #[test]
    fn test_phase7_soft_value_empty() {
        assert_eq!(soft_value(&[], 1.0), f32::NEG_INFINITY);
    }

    /// Phase 7: soft_policy at τ → 0 ⇒ delta on argmax.
    #[test]
    fn test_phase7_soft_policy_delta_at_zero_tau() {
        let q = [0.5_f32, 0.3, 0.7];
        let p = soft_policy(&q, 100, 0.0, 0.1);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], 0.0);
        assert_eq!(p[1], 0.0);
        assert_eq!(p[2], 1.0);
    }

    /// Phase 7: soft_policy sums to 1.
    #[test]
    fn test_phase7_soft_policy_sums_to_one() {
        let q = [0.5_f32, 0.3, 0.7, 0.1];
        let p = soft_policy(&q, 100, 0.01, 0.1);
        let total: f32 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    /// Phase 7: soft_policy with λ_s ≥ 1 (e.g. ε large) gives uniform.
    #[test]
    fn test_phase7_soft_policy_high_epsilon_is_near_uniform() {
        let q = [0.5_f32, 0.3, 0.7];
        // λ_s = 1.0 · 3 / log(102) ≈ 0.65 — still mixes; can't easily
        // hit pure uniform. Instead test that high epsilon brings the
        // distribution closer to uniform than the pure softmax would.
        let p_eps_high = soft_policy(&q, 100, 0.01, 1.0);
        let p_eps_low = soft_policy(&q, 100, 0.01, 0.0);
        // uniform = 1/3 ≈ 0.333 each
        let uniform_dev = |p: &[f32]| -> f32 { p.iter().map(|x| (x - 0.333).abs()).sum::<f32>() };
        assert!(
            uniform_dev(&p_eps_high) < uniform_dev(&p_eps_low),
            "high ε should be closer to uniform"
        );
    }

    /// Phase 7: KL convergence — Q = [0.5, 0.495], τ = 0.01.
    /// Cross-checked with Python prototype:
    ///     softmax(Q/0.01) = softmax([50, 49.5]) ≈ [0.622, 0.378]
    ///     after 50 visits with π_visit = [0.6, 0.4]
    ///     KL ≈ 0.6 · log(0.6/0.622) + 0.4 · log(0.4/0.378)
    ///        ≈ 0.6 · (-0.036) + 0.4 · 0.057
    ///        ≈ -0.0217 + 0.0228 ≈ 0.0011  (≈ 1e-3)
    /// At smaller ε (less mixing), the KL should be tiny and MENTS halts.
    #[test]
    fn test_phase7_kl_visit_to_soft_convergence() {
        let q = [0.5_f32, 0.495];
        let p_soft = soft_policy(&q, 100, 0.01, 0.0);
        let visits = [60_u32, 40_u32];
        let kl = kl_visit_to_soft(&visits, &p_soft);
        // KL is small (< 0.005 for this near-converged case).
        assert!(kl < 0.005, "kl = {kl}");
    }

    /// Phase 7: KL is 0 when visit and soft policies match exactly.
    #[test]
    fn test_phase7_kl_zero_on_match() {
        let p = [0.6_f32, 0.4];
        let visits = [60_u32, 40];
        let kl = kl_visit_to_soft(&visits, &p);
        assert!(kl.abs() < 1e-6, "kl = {kl}");
    }

    /// Phase 7: KL for divergent policies is positive.
    #[test]
    fn test_phase7_kl_positive_on_divergence() {
        let p_soft = [0.5_f32, 0.5];
        let visits = [90_u32, 10]; // visit policy = [0.9, 0.1]
        let kl = kl_visit_to_soft(&visits, &p_soft);
        // KL([0.9, 0.1] || [0.5, 0.5]) = 0.9 log(1.8) + 0.1 log(0.2)
        //                              ≈ 0.529 - 0.161 ≈ 0.368
        assert!((kl - 0.368).abs() < 0.01, "kl = {kl}");
    }

    /// Phase 7: KL handles zero-visit arms (term contributes 0).
    #[test]
    fn test_phase7_kl_zero_visit_arm() {
        let p_soft = [0.5_f32, 0.4, 0.1];
        let visits = [50_u32, 50, 0]; // visit policy = [0.5, 0.5, 0]
        let kl = kl_visit_to_soft(&visits, &p_soft);
        // Hand: 0.5 log(1.0) + 0.5 log(1.25) + 0 (third term)
        //     ≈ 0 + 0.5 · 0.223 ≈ 0.112
        assert!((kl - 0.112).abs() < 0.01, "kl = {kl}");
    }
}

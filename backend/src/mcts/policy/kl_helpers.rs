//! P06: KL bisection helpers shared by P08 (KLLUCBStop) and P09
//! (BayesianQuartz).
//!
//! All formulas follow Kaufmann & Kalyanakrishnan 2013 (KK13),
//! "Information Complexity in Bandit Subset Selection" Theorem 8,
//! with the standard Bernoulli KL inversion via bisection.

/// Bernoulli KL divergence d(p ‖ q) with both arguments clamped away
/// from {0, 1} for numerical stability.
#[inline]
pub fn bernoulli_kl(p: f32, q: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    let q = q.clamp(1e-6, 1.0 - 1e-6);
    p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln()
}

/// Solve for the upper KL-confidence bound `q ∈ [μ̂, 1)`:
///   `n · KL(μ̂, q) ≤ β`
/// via 32-iteration bisection. Returns `q` such that `n·KL(μ̂, q) ≈ β`.
///
/// 32 iterations gives ~1e-9 precision on a unit interval — well beyond
/// the f32 representable range. Reduce if you need to micro-optimize a
/// hot path; rare in practice since this fires only at the controller's
/// periodic boundary.
pub fn kl_upper(mu: f32, n: u32, beta: f32) -> f32 {
    if n == 0 {
        return 1.0 - 1e-6;
    }
    let (mut lo, mut hi) = (mu, 1.0_f32 - 1e-6);
    let n_f = n as f32;
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        if n_f * bernoulli_kl(mu, mid) > beta {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Solve for the lower KL-confidence bound `q ∈ [0, μ̂]`:
///   `n · KL(μ̂, q) ≤ β`
/// (mirror of `kl_upper`).
pub fn kl_lower(mu: f32, n: u32, beta: f32) -> f32 {
    if n == 0 {
        return 1e-6;
    }
    let (mut lo, mut hi) = (1e-6_f32, mu);
    let n_f = n as f32;
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        if n_f * bernoulli_kl(mu, mid) > beta {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// KK13 Theorem 8 stopping threshold:
///   β(t, δ) = log(k₁ · K · t^α / δ),  k₁ = 405.5, α = 1.1
///
/// with `K = n_children` (number of root candidates), `t = iteration`,
/// `δ = confidence level` (e.g. 0.05 for 95% PAC). Caller passes `t` and
/// `k` as f32s to avoid repeated casts in the hot path.
#[inline]
pub fn kl_lucb_beta(t: f32, k: f32, delta: f32) -> f32 {
    (405.5_f32 * k * t.powf(1.1) / delta).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bernoulli KL is non-negative and zero on the diagonal.
    #[test]
    fn bernoulli_kl_zero_on_diagonal() {
        for &p in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            assert!(bernoulli_kl(p, p).abs() < 1e-5, "p={p}");
        }
    }

    /// Bernoulli KL is non-negative away from the diagonal.
    #[test]
    fn bernoulli_kl_non_negative() {
        for &p in &[0.1, 0.5, 0.9] {
            for &q in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let kl = bernoulli_kl(p, q);
                assert!(kl >= -1e-5, "p={p}, q={q}, kl={kl}");
            }
        }
    }

    /// kl_upper inverts the KL via bisection: at the returned q,
    /// n · KL(μ̂, q) ≈ β within f32 tolerance.
    #[test]
    fn kl_upper_inverts_bisection() {
        let mu = 0.6_f32;
        let n = 100_u32;
        let beta = 5.0_f32;
        let q = kl_upper(mu, n, beta);
        let target = (n as f32) * bernoulli_kl(mu, q);
        assert!(
            (target - beta).abs() < 1e-2,
            "target={target}, beta={beta}, q={q}"
        );
        // Sanity: q > mu (the upper CI is above the mean).
        assert!(q > mu);
    }

    /// kl_lower inverts symmetrically.
    #[test]
    fn kl_lower_inverts_bisection() {
        let mu = 0.6_f32;
        let n = 100_u32;
        let beta = 5.0_f32;
        let q = kl_lower(mu, n, beta);
        let target = (n as f32) * bernoulli_kl(mu, q);
        assert!(
            (target - beta).abs() < 1e-2,
            "target={target}, beta={beta}, q={q}"
        );
        assert!(q < mu);
    }

    /// kl_lucb_beta matches the KK13 Theorem 8 form for a sanity case.
    /// β(t=151, K=3, δ=0.05) = log(405.5·3·151^1.1/0.05) ≈ 15.618.
    /// Hand re-derivation:
    ///   151^1.1 = 151·exp(0.1·ln(151)) ≈ 151·1.6515 ≈ 249.4
    ///   405.5·3·249.4/0.05 ≈ 6.068e6
    ///   ln(6.068e6) ≈ 15.618
    #[test]
    fn kl_lucb_beta_kk13_sanity() {
        let beta = kl_lucb_beta(151.0, 3.0, 0.05);
        assert!((beta - 15.618).abs() < 0.05, "beta={beta}");
    }
}

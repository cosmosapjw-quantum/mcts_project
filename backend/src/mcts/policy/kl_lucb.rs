//! P08: `KLLUCBStop` — pure-PUCT action selection with
//! Kaufmann-Kalyanakrishnan 2013 PAC best-arm-id stopping.
//!
//! Replaces the legacy `HaltMode::SimpleThreshold` (P_flip < 0.159)
//! with a formal δ-PAC certificate. The 0.159 threshold was a
//! heuristic 1-σ rule; KK13's GLR-style stopping rule has provable
//! sample complexity bounds and configurable confidence δ.
//!
//! Reference: Kaufmann & Kalyanakrishnan 2013, "Information Complexity
//! in Bandit Subset Selection", Theorem 8. With β(t,δ) =
//! log(k₁·K·t^α/δ), k₁=405.5, α=1.1, the stopping rule is:
//!
//!     L_b̂(t,δ) > U_c(t,δ)
//!
//! where b̂ = argmax_a μ̂_a, c = argmax_{a≠b̂} U_a(t,δ), and L/U are
//! KL-inversion CIs from `kl_helpers::kl_lower` / `kl_upper`.
//! Equivalent gap statistic: `gap_bits = N_b̂ · KL(μ̂_b̂, μ̂_c) − β(t,δ)`,
//! stop when positive.
//!
//! Q ↦ μ̂ mapping: Q is in `[-1, 1]` (two-player zero-sum); we map to
//! `[0, 1]` via `μ̂ = (Q + 1) / 2` for Bernoulli KL. Rank-preserving.

use parking_lot::Mutex;
use smallvec::SmallVec;

use super::kl_helpers::{bernoulli_kl, kl_lower, kl_lucb_beta, kl_upper};
use super::trait_def::{
    ControllerTelemetry, EdgeView, HaltDecision, ScoreAdjustment, SearchPolicy, SearchSnapshot,
};
use crate::mcts::quartz::HaltReason;

/// Cached PAC certificate fields. Updated at each `observe` call;
/// read in the hot path by `should_halt` (and exposed via
/// `telemetry`).
#[derive(Clone, Copy, Debug, Default)]
struct Cache {
    /// Positive ⇒ PAC certificate fires at confidence δ.
    gap_bits: f32,
    /// Empirical-best arm at last observe.
    best_idx: u16,
    /// Argmax of UCB across arms ≠ best.
    second_idx: u16,
    /// Most recent best mean (in [0, 1] mapped space) for telemetry.
    best_mu: f32,
    /// Most recent runner-up UCB — useful for diagnostics and the
    /// "how close did we come?" reading.
    second_ucb: f32,
}

pub struct KLLUCBStop {
    /// PAC confidence (default 0.05 ⇒ 95% confidence).
    pub delta: f32,
    /// Minimum visits per arm before that arm is eligible for the
    /// best-arm comparison. Without this, a single lucky rollout
    /// can trigger a spurious stop.
    pub min_pulls: u32,
    /// Minimum total root visits before any halt is allowed. Avoids
    /// fast-stopping on tiny budgets where the KK13 bound is loose.
    pub min_total: u32,
    /// Hard ceiling on root visits (mirrors max_visits semantics
    /// from the legacy controller).
    pub max_visits: u32,
    cached: Mutex<Cache>,
}

impl KLLUCBStop {
    pub fn new(delta: f32, min_pulls: u32, min_total: u32, max_visits: u32) -> Self {
        Self {
            delta,
            min_pulls,
            min_total,
            max_visits,
            cached: Mutex::new(Cache::default()),
        }
    }

    /// Default tuning per the design doc: δ=0.05, min_pulls=30,
    /// `min_total = clamp(budget/4, 20, 200)`, `max_visits = u32::MAX`.
    ///
    /// `min_total` scales down for toy budgets (≤800) so the policy
    /// has at least 75% of the budget to fire its PAC certificate;
    /// at production budgets (≥800) it caps at 200 per the design doc.
    /// `max_visits` is left at u32::MAX so the host controller's
    /// `BudgetExhausted` is the sole budget-cap halt — the policy's
    /// only halt path is the actual `gap_bits > 0` certificate.
    /// (BQ++ Phase 8c followup: was `min_total=200, max_visits=budget`
    /// which raced the controller for tie wins on toy budgets and
    /// gated out below-200 fires entirely.)
    pub fn default_for_budget(budget: u32) -> Self {
        let min_total = (budget / 4).clamp(20, 200);
        Self::new(0.05, 30, min_total, u32::MAX)
    }
}

/// Map Q ∈ [-1, 1] to μ̂ ∈ [0, 1] for Bernoulli KL operations.
/// Rank-preserving by construction.
#[inline]
fn q_to_mu(q: f32) -> f32 {
    0.5 * (q + 1.0)
}

impl SearchPolicy for KLLUCBStop {
    fn name(&self) -> &'static str {
        "kl_lucb_stop"
    }

    fn observe(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]) {
        if snap.root_visits < self.min_total || edges.len() < 2 {
            return;
        }
        let t = snap.iteration.max(1) as f32;
        let k = snap.n_children.max(2) as f32;
        let beta = kl_lucb_beta(t, k, self.delta);

        // Empirical best — argmax_a μ̂_a, restricted to arms with at
        // least `min_pulls` visits so a single rollout doesn't
        // dominate.
        let mut best_idx = u16::MAX;
        let mut best_mu = -1.0f32;
        for e in edges {
            if e.n < self.min_pulls {
                continue;
            }
            let mu = q_to_mu(e.q);
            if mu > best_mu {
                best_mu = mu;
                best_idx = e.idx;
            }
        }
        if best_idx == u16::MAX {
            return;
        }

        // Runner-up — argmax_{a≠b̂} U_a(t,δ). A1-a audit fix: this side
        // must NOT filter by `min_pulls`. KK13's certificate requires
        // the upper bound to dominate over ALL a≠b̂, not just the
        // well-sampled ones — an unvisited or barely-visited arm has
        // an upper bound near 1.0 (see `kl_upper`'s n=0 fast path) and
        // must be allowed to correctly block a premature stop. The
        // previous code applied `min_pulls` here too, silently
        // excluding under-sampled arms from the max-U comparison —
        // anti-conservative, and worst exactly at the low visit
        // budgets (8-64) this project targets, where most arms have
        // far fewer than `min_pulls` visits. `min_pulls` remains a
        // legitimate eligibility gate for the *best* arm above (an
        // unvetted single lucky rollout should not become b̂), but
        // must not gate the runner-up's upper bound.
        let mut second_ucb = -1.0f32;
        let mut second_idx = best_idx;
        let mut n_best = 0_u32;
        for e in edges {
            if e.idx == best_idx {
                n_best = e.n;
                continue;
            }
            let mu = q_to_mu(e.q);
            let u = kl_upper(mu, e.n, beta);
            if u > second_ucb {
                second_ucb = u;
                second_idx = e.idx;
            }
        }
        if second_idx == best_idx {
            // Defensive only: with `edges.len() >= 2` guaranteed by the
            // early return above and every a≠b̂ now unconditionally
            // considered, this is unreachable — kept as a safety net
            // against a future signature change, not live logic.
            return;
        }

        // Lower CI for the best arm.
        let l_best = kl_lower(best_mu, n_best, beta);
        let gap_bits = l_best - second_ucb;

        *self.cached.lock() = Cache {
            gap_bits,
            best_idx,
            second_idx,
            best_mu,
            second_ucb,
        };
    }

    fn score_adjustment(&self, _e: EdgeView<'_>) -> ScoreAdjustment {
        // Pure PUCT: no penalty, no refresh, no Fisher weighting.
        ScoreAdjustment::default()
    }

    fn should_halt(&self, snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) -> HaltDecision {
        if snap.root_visits >= self.max_visits {
            return HaltDecision::Stop(HaltReason::MaxVisits);
        }
        if snap.root_visits < self.min_total {
            return HaltDecision::Continue;
        }
        let cache = *self.cached.lock();
        if cache.gap_bits > 0.0 {
            HaltDecision::Stop(HaltReason::KLLUCBStop)
        } else {
            HaltDecision::Continue
        }
    }

    fn telemetry(&self) -> ControllerTelemetry {
        let cache = *self.cached.lock();
        ControllerTelemetry {
            schema_version: 1,
            policy_name: self.name().to_string(),
            halt_reason: None,
            gap_bits: cache.gap_bits,
            glr_z: 0.0,
            mean_sigma_a: 0.0,
            chi2: 0.0,
            chi2_dof: 0,
            bayes_voi: 0.0,
            eval_sigma: 0.0,
            iters_at_halt: 0,
            ..ControllerTelemetry::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(root_visits: u32, n_children: u16) -> SearchSnapshot {
        SearchSnapshot {
            root_visits,
            n_children,
            n_visible: n_children,
            elapsed_ms: 0,
            depth_max: 1,
            mean_q_root: 0.0,
            sigma_q_root: 0.3,
            sigma_eval: None,
            iteration: root_visits as u64,
            best_idx: 0,
            second_idx: 1,
        }
    }

    fn make_edge<'a>(
        idx: u16,
        n: u32,
        q: f32,
        snap: &'a SearchSnapshot,
        n_total: &'a u32,
    ) -> EdgeView<'a> {
        EdgeView {
            idx,
            n,
            n_virtual: 0,
            o_a: 0,
            q,
            q_sum: q * n as f32,
            m2: 0.0,
            prior: 1.0 / snap.n_children as f32,
            depth: 0,
            last_value: q,
            envar_partial: 0.0,
            root_total_n: n_total,
            stats: snap,
        }
    }

    /// P08: identity score adjustment — KLLUCBStop never modifies
    /// the prior or adds penalty. Pure PUCT for selection.
    #[test]
    fn test_p08_kl_lucb_stop_score_adjustment_is_identity() {
        let policy = KLLUCBStop::default_for_budget(10000);
        let snap = make_snap(0, 4);
        let n = 0_u32;
        let edge = make_edge(0, 5, 0.5, &snap, &n);
        let adj = policy.score_adjustment(edge);
        assert_eq!(adj.penalty, 0.0);
        assert_eq!(adj.fisher_alpha, 0.0);
        assert!(adj.q_override.is_none());
    }

    /// P08: do NOT halt when total visits are below min_total.
    #[test]
    fn test_p08_kl_lucb_stop_below_min_total_continues() {
        let policy = KLLUCBStop::new(0.05, 30, 200, 10000);
        let snap = make_snap(199, 3);
        let edges: Vec<EdgeView<'_>> = vec![];
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Continue
        ));
    }

    /// P08: halt at max_visits regardless of gap_bits state.
    /// Use direct constructor — `default_for_budget` was changed in
    /// the Phase 8c followup to set max_visits=u32::MAX so the host
    /// controller's BudgetExhausted is the sole budget-cap halt.
    #[test]
    fn test_p08_kl_lucb_stop_halt_at_max_visits() {
        let policy = KLLUCBStop::new(0.05, 30, 200, 800);
        let snap = make_snap(800, 3);
        let edges: Vec<EdgeView<'_>> = vec![];
        match policy.should_halt(&snap, &edges) {
            HaltDecision::Stop(HaltReason::MaxVisits) => {}
            other => panic!("expected Stop(MaxVisits), got {other:?}"),
        }
    }

    /// P08: with a tight gap (Q values close, modest visits), gap_bits
    /// stays negative ⇒ Continue. Hand-computed:
    ///   N=[100, 50, 1], Q=[0.6, 0.5, 0.4], mapped μ=[0.8, 0.75, 0.7].
    ///   t=151, K=3, δ=0.05 ⇒ β ≈ 15.618 (verified in P06's
    ///   kl_lucb_beta_kk13_sanity test).
    ///   Arm 0 (n=100) is the best (min_pulls=30 still gates the best
    ///   arm side). A1-a: the runner-up side no longer filters by
    ///   min_pulls, so arm 2 (n=1) now correctly participates in the
    ///   U-side race — and wins it, since 1 pull gives an upper bound
    ///   near 1.0 (see `kl_upper`'s tiny-n behavior). That's the
    ///   intended fix: an almost-unvisited arm must be able to block a
    ///   premature stop, not be silently excluded from the comparison.
    #[test]
    fn test_p08_kl_lucb_stop_tight_gap_does_not_halt() {
        let policy = KLLUCBStop::new(0.05, 30, 100, 10000);
        let snap = make_snap(151, 3);
        let n_total = 151_u32;
        let edges = vec![
            make_edge(0, 100, 0.6, &snap, &n_total),
            make_edge(1, 50, 0.5, &snap, &n_total),
            make_edge(2, 1, 0.4, &snap, &n_total),
        ];
        policy.observe(&snap, &edges);
        let cache = *policy.cached.lock();
        // Best is arm 0 (μ̂=0.8); runner-up is arm 2 — its n=1 upper
        // bound dominates arm 1's (n=50, better-sampled) bound.
        assert_eq!(cache.best_idx, 0);
        assert_eq!(cache.second_idx, 2);
        // Gap should be negative — an almost-unvisited arm's wide
        // bound easily blocks certification at this budget.
        assert!(
            cache.gap_bits < 0.0,
            "expected negative gap_bits, got {}",
            cache.gap_bits
        );
        // Continue.
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Continue
        ));
    }

    /// P08: with a wide separation (best arm strongly preferred and
    /// runner-up clearly worse), gap_bits goes positive ⇒
    /// Stop(KLLUCBStop). Q=[0.9, 0.0] mapped to μ̂=[0.95, 0.5] —
    /// at N=[10000, 500] the KK13 confidence bounds clearly separate.
    /// Only two (adequately-sampled) arms — see
    /// `test_p08_kl_lucb_stop_a1a_underpulled_arm_blocks_premature_stop`
    /// below for the case where a third, barely-visited candidate is
    /// live: A1-a intentionally makes that case block the stop.
    ///
    /// Hand sanity: β(10501, 3, 0.05) ≈ 20.3.
    /// L_best ≈ 0.937, U_second ≈ 0.641, gap ≈ 0.296 > 0.
    #[test]
    fn test_p08_kl_lucb_stop_wide_gap_halts() {
        let policy = KLLUCBStop::new(0.05, 30, 200, 100000);
        let snap = make_snap(10501, 3);
        let n_total = 10501_u32;
        let edges = vec![
            make_edge(0, 10000, 0.9, &snap, &n_total),
            make_edge(1, 500, 0.0, &snap, &n_total),
        ];
        policy.observe(&snap, &edges);
        let cache = *policy.cached.lock();
        assert_eq!(cache.best_idx, 0);
        assert!(
            cache.gap_bits > 0.0,
            "expected positive gap_bits, got {}",
            cache.gap_bits
        );
        match policy.should_halt(&snap, &edges) {
            HaltDecision::Stop(HaltReason::KLLUCBStop) => {}
            other => panic!("expected Stop(KLLUCBStop), got {other:?}"),
        }
    }

    /// A1-a regression: the whole point of removing `min_pulls` from
    /// the runner-up side. Same wide, well-resolved gap as
    /// `test_p08_kl_lucb_stop_wide_gap_halts` above, but with a third
    /// candidate live at only 1 pull. Before the fix, that arm was
    /// silently excluded from the runner-up race and the cert fired
    /// anyway (anti-conservative — the barely-sampled arm was never
    /// actually ruled out). After the fix, its near-1.0 upper bound
    /// correctly blocks the stop.
    #[test]
    fn test_p08_kl_lucb_stop_a1a_underpulled_arm_blocks_premature_stop() {
        let policy = KLLUCBStop::new(0.05, 30, 200, 100000);
        let snap = make_snap(10501, 3);
        let n_total = 10501_u32;
        let edges = vec![
            make_edge(0, 10000, 0.9, &snap, &n_total),
            make_edge(1, 500, 0.0, &snap, &n_total),
            make_edge(2, 1, -0.5, &snap, &n_total),
        ];
        policy.observe(&snap, &edges);
        let cache = *policy.cached.lock();
        assert_eq!(cache.best_idx, 0);
        assert_eq!(
            cache.second_idx, 2,
            "the under-sampled arm must win the runner-up slot on its wide bound"
        );
        assert!(
            cache.gap_bits < 0.0,
            "an unresolved 1-pull candidate must block certification, got gap_bits={}",
            cache.gap_bits
        );
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Continue
        ));
    }

    /// P08: min_pulls guard — the best-arm side still requires
    /// `min_pulls` (a single lucky rollout should not become b̂), so
    /// with only arm 0 adequately sampled, best_idx=0 is the sole
    /// candidate for b̂. A1-a: the runner-up side no longer applies
    /// `min_pulls`, so arms 1 and 2 (n=5 each, below min_pulls) DO
    /// participate in the U-side race and one of them wins it with a
    /// wide, barely-constrained bound — correctly keeping gap_bits
    /// negative (never positive), same qualitative Continue result as
    /// before the fix, but now via a real computed gap rather than
    /// the old early-return-to-default-0.0 path.
    #[test]
    fn test_p08_kl_lucb_stop_min_pulls_guard() {
        let policy = KLLUCBStop::new(0.05, 30, 100, 10000);
        let snap = make_snap(120, 3);
        let n_total = 120_u32;
        let edges = vec![
            make_edge(0, 100, 0.6, &snap, &n_total),
            make_edge(1, 5, 0.5, &snap, &n_total),
            make_edge(2, 5, 0.4, &snap, &n_total),
        ];
        policy.observe(&snap, &edges);
        let cache = *policy.cached.lock();
        assert_eq!(cache.best_idx, 0);
        assert_ne!(
            cache.second_idx, cache.best_idx,
            "runner-up must now resolve to a real arm even though both are under min_pulls"
        );
        assert!(cache.gap_bits <= 0.0);
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Continue
        ));
    }

    /// P08: telemetry exposes the cached gap_bits and stable name.
    #[test]
    fn test_p08_kl_lucb_stop_telemetry() {
        let policy = KLLUCBStop::default_for_budget(10000);
        let tel = policy.telemetry();
        assert_eq!(tel.schema_version, 1);
        assert_eq!(tel.policy_name, "kl_lucb_stop");
        // Pre-observe, cache is default ⇒ gap_bits=0.0.
        assert_eq!(tel.gap_bits, 0.0);

        // After observe with wide gap (matching the wide-gap test
        // above), telemetry reflects the cache.
        let snap = make_snap(10501, 3);
        let n_total = 10501_u32;
        let edges = vec![
            make_edge(0, 10000, 0.9, &snap, &n_total),
            make_edge(1, 500, 0.0, &snap, &n_total),
        ];
        policy.observe(&snap, &edges);
        let tel2 = policy.telemetry();
        assert!(tel2.gap_bits > 0.0);
    }

    /// P08: Q-mapping invariance. Shifting all Q values by a constant
    /// (which is meaningless in two-player zero-sum games — only
    /// relative ordering matters) does not change the cached
    /// best_idx / second_idx ranking. (The gap_bits magnitude does
    /// shift because the Bernoulli KL is not translation-invariant
    /// in mu, but the qualitative result holds.)
    #[test]
    fn test_p08_kl_lucb_stop_q_mapping_preserves_ranking() {
        let policy = KLLUCBStop::new(0.05, 30, 100, 100000);
        let snap = make_snap(151, 3);
        let n_total = 151_u32;

        // Original Q ordering
        let edges_a = vec![
            make_edge(0, 100, 0.6, &snap, &n_total),
            make_edge(1, 50, 0.5, &snap, &n_total),
        ];
        policy.observe(&snap, &edges_a);
        let cache_a = *policy.cached.lock();

        // Same gap pattern, but shifted lower
        let policy_b = KLLUCBStop::new(0.05, 30, 100, 100000);
        let edges_b = vec![
            make_edge(0, 100, 0.4, &snap, &n_total),
            make_edge(1, 50, 0.3, &snap, &n_total),
        ];
        policy_b.observe(&snap, &edges_b);
        let cache_b = *policy_b.cached.lock();

        // Ranking unchanged by the shift.
        assert_eq!(cache_a.best_idx, cache_b.best_idx);
        assert_eq!(cache_a.second_idx, cache_b.second_idx);
    }

    /// P08: SmallVec import sanity — make sure the dependency is wired.
    /// Also exercises the empty-edges early return path.
    #[test]
    fn test_p08_kl_lucb_stop_empty_edges_no_op() {
        let policy = KLLUCBStop::default_for_budget(10000);
        let snap = make_snap(500, 0);
        let edges: SmallVec<[EdgeView<'_>; 4]> = SmallVec::new();
        policy.observe(&snap, &edges[..]);
        let cache = *policy.cached.lock();
        assert_eq!(cache.gap_bits, 0.0);
    }
}

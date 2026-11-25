//! P06: search-policy module — unified `SearchPolicy` trait + types
//! + a `DefaultBridgePolicy` no-op fallback.
//!
//! P06 is **scaffolding only**: nothing in the engine consults the
//! trait yet. P07 lands `LegacyAlphaZero` + `LegacyQuartz` (the
//! bit-identical shim around the existing controller path); P08 adds
//! `KLLUCBStop`; P09 adds `BayesianQuartz`; P10 flips the default;
//! P11 adds the opt-in `MENTSEntropyRegularized`.
//!
//! Pre-P06 the controller surface was 7 PenaltyMode × 4 HaltMode × 3
//! CostMode × 2¹¹ booleans ≈ 229k combinations dispatched at
//! [src/mcts/select.rs:408](`select.rs`). Many of those combinations
//! interact unpredictably (e.g. SelfAdaptive penalty + Fixed VL = double
//! pessimism per `parallel.rs:26`). The trait collapses this surface
//! into ~5 named, individually-tunable policies where each policy is
//! a single named author with one set of hyperparameters.

pub mod cache;
pub mod gumbel_sh;
pub mod kg_stop;
pub mod kl_helpers;
pub mod kl_lucb;
pub mod legacy_az;
pub mod legacy_quartz;
pub mod ments;
pub mod reservoir;
pub mod tactical;
pub mod trait_def;

pub use cache::{EdgeRef, PolicyCache, PolicyCachePublisher};
pub use gumbel_sh::{gumbel_top_m, initial_bracket, sample_gumbel, SequentialHalvingBracket};
pub use kg_stop::{
    compute_kg_array, expected_improvement, kg_per_arm, should_halt_by_kg, KgCostSource, KgStop,
};
pub use kl_helpers::{bernoulli_kl, kl_lower, kl_lucb_beta, kl_upper};
pub use kl_lucb::KLLUCBStop;
pub use legacy_az::LegacyAlphaZero;
pub use legacy_quartz::LegacyQuartz;
pub use ments::{kl_visit_to_soft, soft_policy, soft_value};
pub use reservoir::{lambda_score, quantile, Reservoir};
pub use tactical::{tactical_sentinel, TacticalResult};
pub use trait_def::{
    BoxedPolicy, ControllerTelemetry, EdgeView, EffectivePrior, HaltDecision, ScoreAdjustment,
    SearchPolicy, SearchSnapshot,
};

use crate::mcts::quartz::HaltReason;

/// No-op fallback policy. Returns `ScoreAdjustment::default()`
/// (= identity / pure PUCT) and `HaltDecision::Continue`. Used as
/// the default when the engine isn't given an explicit policy yet —
/// which preserves the existing search behavior (the engine still
/// consults the legacy `quartz_policy_adjustment` and
/// `QuartzController::should_stop` paths).
///
/// Once P10 ships and `BayesianQuartz` becomes the engine's default,
/// this fallback continues to be useful for unit tests that need a
/// policy object without policy effects.
pub struct DefaultBridgePolicy;

impl SearchPolicy for DefaultBridgePolicy {
    fn name(&self) -> &'static str {
        "default_bridge"
    }
    fn observe(&self, _snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) {}
    fn score_adjustment(&self, _e: EdgeView<'_>) -> ScoreAdjustment {
        ScoreAdjustment::default()
    }
    fn should_halt(&self, _snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) -> HaltDecision {
        // The fallback never halts; the engine must use its own halt
        // mechanism. P07's LegacyAlphaZero is the first policy that
        // actually halts (on max_visits).
        HaltDecision::Continue
    }
    fn telemetry(&self) -> ControllerTelemetry {
        ControllerTelemetry {
            schema_version: 1,
            policy_name: "default_bridge".to_string(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P06: the default fallback is a no-op — score adj is all-zero,
    /// should_halt is Continue, telemetry has schema_version=1 and
    /// the canonical name. This pins the scaffolding before P07 wires
    /// in real policies.
    #[test]
    fn test_p06_default_bridge_no_op() {
        let policy: BoxedPolicy = Box::new(DefaultBridgePolicy);
        assert_eq!(policy.name(), "default_bridge");

        let snap = SearchSnapshot {
            root_visits: 100,
            n_children: 4,
            n_visible: 4,
            elapsed_ms: 10,
            depth_max: 5,
            mean_q_root: 0.0,
            sigma_q_root: 0.3,
            sigma_eval: None,
            iteration: 100,
            best_idx: 0,
            second_idx: 1,
        };
        let edges: Vec<EdgeView<'_>> = vec![];
        // observe must be infallible
        policy.observe(&snap, &edges);

        // score adjustment is the all-zero identity
        let n = 0_u32;
        let edge = EdgeView {
            idx: 0,
            n: 5,
            n_virtual: 0,
            o_a: 0,
            q: 0.0,
            q_sum: 0.0,
            m2: 0.0,
            prior: 0.25,
            depth: 0,
            last_value: 0.0,
            envar_partial: 0.0,
            root_total_n: &n,
            stats: &snap,
        };
        let adj = policy.score_adjustment(edge);
        assert_eq!(adj.penalty, 0.0);
        assert_eq!(adj.fisher_alpha, 0.0);
        assert!(adj.q_override.is_none());

        // halt decision is always Continue
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Continue
        ));

        // telemetry is canonical and serializable
        let tel = policy.telemetry();
        assert_eq!(tel.schema_version, 1);
        assert_eq!(tel.policy_name, "default_bridge");
        let _ = serde_json::to_string(&tel).expect("ControllerTelemetry must serialize");
    }

    /// P06: HaltDecision::Stop carries a HaltReason — verify the
    /// integration point with the P01 HaltReason enum.
    #[test]
    fn test_p06_halt_decision_carries_reason() {
        let d = HaltDecision::Stop(HaltReason::FixedBudget);
        match d {
            HaltDecision::Stop(reason) => assert_eq!(reason, HaltReason::FixedBudget),
            HaltDecision::Continue => panic!("expected Stop"),
        }
    }

    /// P06/A1-c: EdgeView::sigma_a applies the additive smoothing
    /// correctly. Denominator is `max(N-1,1)+λ₀` (matching the Python
    /// prototype's `belief.py::empirical_bayes_shrinkage`), which
    /// collapses N=0 and N=1 to the same effective count of
    /// `max(-1,1)+λ₀ = 1+λ₀ = 5`: σ_a = √((0 + 4·0.09)/5) ≈ 0.2683 —
    /// identical to the N=1 case below, and matching belief.py's own
    /// documented hand-derivation for N=0.
    #[test]
    fn test_p06_edge_view_sigma_a_smoothing_at_zero_visits() {
        let n_total = 0_u32;
        let snap = SearchSnapshot {
            root_visits: 0,
            n_children: 1,
            n_visible: 1,
            elapsed_ms: 0,
            depth_max: 0,
            mean_q_root: 0.0,
            sigma_q_root: 0.3,
            sigma_eval: None,
            iteration: 0,
            best_idx: 0,
            second_idx: 0,
        };
        let edge = EdgeView {
            idx: 0,
            n: 0,
            n_virtual: 0,
            o_a: 0,
            q: 0.0,
            q_sum: 0.0,
            m2: 0.0,
            prior: 1.0,
            depth: 0,
            last_value: 0.0,
            envar_partial: 0.0,
            root_total_n: &n_total,
            stats: &snap,
        };
        // λ₀=4, σ_root=0.3, M2=0, N=0 ⇒ n_eff=max(-1,1)+4=5
        // ⇒ σ_a = √((0 + 4·0.09)/5) = √0.072 ≈ 0.2683
        let sigma = edge.sigma_a(4.0);
        assert!((sigma - 0.2683).abs() < 1e-3, "sigma_a={sigma}");
    }

    /// P06: σ_a after one observation. Hand-computed:
    /// N=1, M2=0, σ_root=0.3, λ₀=4 ⇒ n_eff=max(0,1)+4=5
    /// ⇒ σ_a = √((0 + 4·0.09)/5) = √(0.072) ≈ 0.2683.
    #[test]
    fn test_p06_edge_view_sigma_a_smoothing_after_one_observation() {
        let n_total = 1_u32;
        let snap = SearchSnapshot {
            root_visits: 1,
            n_children: 1,
            n_visible: 1,
            elapsed_ms: 0,
            depth_max: 1,
            mean_q_root: 0.0,
            sigma_q_root: 0.3,
            sigma_eval: None,
            iteration: 1,
            best_idx: 0,
            second_idx: 0,
        };
        let edge = EdgeView {
            idx: 0,
            n: 1,
            n_virtual: 0,
            o_a: 0,
            q: 0.0,
            q_sum: 0.0,
            m2: 0.0,
            prior: 1.0,
            depth: 0,
            last_value: 0.0,
            envar_partial: 0.0,
            root_total_n: &n_total,
            stats: &snap,
        };
        // (0 + 4·0.09)/(1+4) = 0.072 → √ ≈ 0.2683
        let sigma = edge.sigma_a(4.0);
        assert!((sigma - 0.2683).abs() < 1e-3, "sigma_a={sigma}");
    }

    /// A1-c cross-language regression: pins `sigma_a` at N ∈ {0,1,2,10}
    /// against hand-derived values using the SAME denominator formula
    /// as `prototype/bqpp_prototype/belief.py::empirical_bayes_shrinkage`
    /// (`max(N-1,1)+λ₀`), with a mirrored Python test
    /// (`test_empirical_bayes_shrinkage_matches_rust_a1c_pinned_values`
    /// in `prototype/tests/test_belief.py`) asserting the identical
    /// values independently. Both must be updated together if the
    /// formula ever changes again.
    #[test]
    fn test_a1c_edge_view_sigma_a_matches_prototype_formula_across_n() {
        // lambda0=4, sigma_root=0.3 (prior_var=0.09), M2=0 throughout.
        // n_eff = max(n-1,1) + 4; sigma_a = sqrt((4*0.09)/n_eff).
        let cases: [(u32, f32); 4] = [
            (0, 0.2683),  // n_eff = max(-1,1)+4 = 5
            (1, 0.2683),  // n_eff = max(0,1)+4  = 5
            (2, 0.2683),  // n_eff = max(1,1)+4  = 5
            (10, 0.1664), // n_eff = max(9,1)+4 = 13
        ];
        for (n, expected) in cases {
            let n_total = n.max(1);
            let snap = SearchSnapshot {
                root_visits: n,
                n_children: 1,
                n_visible: 1,
                elapsed_ms: 0,
                depth_max: 0,
                mean_q_root: 0.0,
                sigma_q_root: 0.3,
                sigma_eval: None,
                iteration: n as u64,
                best_idx: 0,
                second_idx: 0,
            };
            let edge = EdgeView {
                idx: 0,
                n,
                n_virtual: 0,
                o_a: 0,
                q: 0.0,
                q_sum: 0.0,
                m2: 0.0,
                prior: 1.0,
                depth: 0,
                last_value: 0.0,
                envar_partial: 0.0,
                root_total_n: &n_total,
                stats: &snap,
            };
            let sigma = edge.sigma_a(4.0);
            assert!(
                (sigma - expected).abs() < 1e-3,
                "n={n}: expected sigma_a≈{expected}, got {sigma}"
            );
        }
    }
}

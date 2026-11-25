//! P07: `LegacyAlphaZero` — pure PUCT + fixed-budget halt.
//!
//! Reproduces the original AlphaZero / Silver et al. 2017 search shape:
//! no penalty, no refresh, no Fisher weighting, halt at exactly `budget`
//! root visits. Reference for the policy framework — confirms a clean
//! "do nothing" policy works through the trait without surprises.
//!
//! Migration: replaces `PenaltyMode::None × HaltMode::Fixed × any
//! CostMode × no flags`. Use this as the baseline for any ablation
//! that wants to isolate "does the policy help?" vs pure PUCT.

use super::trait_def::{
    ControllerTelemetry, EdgeView, HaltDecision, ScoreAdjustment, SearchPolicy, SearchSnapshot,
};
use crate::mcts::quartz::HaltReason;

/// Pure-PUCT, fixed-budget policy. ~30 LOC of "interesting" code; the
/// rest is the trait implementation boilerplate.
pub struct LegacyAlphaZero {
    pub budget: u32,
}

impl LegacyAlphaZero {
    /// Construct a fixed-budget AlphaZero policy. `budget` is the
    /// target root-visit count; halt fires the moment
    /// `root_visits >= budget`.
    pub fn new(budget: u32) -> Self {
        Self { budget }
    }
}

impl SearchPolicy for LegacyAlphaZero {
    fn name(&self) -> &'static str {
        "legacy_az"
    }

    fn observe(&self, _snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) {
        // No state. AlphaZero policy doesn't compute anything between
        // selections — vanilla PUCT reads N, Q, prior off the edge each
        // time and that's it.
    }

    fn score_adjustment(&self, _e: EdgeView<'_>) -> ScoreAdjustment {
        // Default = identity. Caller's PUCT formula sees
        // effective_prior=0 (zero prior override), penalty=0,
        // fisher_alpha=0, q_override=None. The PUCT entry point
        // interprets `effective_prior=0` as "no override" and reads
        // edge.prior directly. (See P10 wiring; not yet exercised.)
        ScoreAdjustment::default()
    }

    fn should_halt(&self, snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) -> HaltDecision {
        if snap.root_visits >= self.budget {
            HaltDecision::Stop(HaltReason::FixedBudget)
        } else {
            HaltDecision::Continue
        }
    }

    fn telemetry(&self) -> ControllerTelemetry {
        ControllerTelemetry {
            schema_version: 1,
            policy_name: self.name().to_string(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_snapshot(root_visits: u32) -> SearchSnapshot {
        SearchSnapshot {
            root_visits,
            n_children: 4,
            n_visible: 4,
            elapsed_ms: 0,
            depth_max: 0,
            mean_q_root: 0.0,
            sigma_q_root: 0.3,
            sigma_eval: None,
            iteration: root_visits as u64,
            best_idx: 0,
            second_idx: 1,
        }
    }

    /// P07: identity score adjustment. AlphaZero policy must not
    /// modify priors, must not add penalty.
    #[test]
    fn test_p07_legacy_az_score_adjustment_is_identity() {
        let policy = LegacyAlphaZero::new(800);
        let snap = build_snapshot(0);
        let n_total = 0_u32;
        let edge = EdgeView {
            idx: 0,
            n: 5,
            n_virtual: 0,
            o_a: 0,
            q: 0.5,
            q_sum: 2.5,
            m2: 0.1,
            prior: 0.25,
            depth: 0,
            last_value: 0.5,
            envar_partial: 0.0,
            root_total_n: &n_total,
            stats: &snap,
        };
        let adj = policy.score_adjustment(edge);
        assert_eq!(adj.effective_prior, 0.0);
        assert_eq!(adj.penalty, 0.0);
        assert_eq!(adj.fisher_alpha, 0.0);
        assert!(adj.q_override.is_none());
    }

    /// P07: fixed-budget halt fires at exactly budget visits.
    #[test]
    fn test_p07_legacy_az_halt_at_budget() {
        let policy = LegacyAlphaZero::new(100);
        let edges: Vec<EdgeView<'_>> = vec![];

        // Below budget: continue.
        let snap_below = build_snapshot(99);
        assert!(matches!(
            policy.should_halt(&snap_below, &edges),
            HaltDecision::Continue
        ));

        // At budget: stop with FixedBudget reason.
        let snap_at = build_snapshot(100);
        match policy.should_halt(&snap_at, &edges) {
            HaltDecision::Stop(HaltReason::FixedBudget) => {}
            other => panic!("expected Stop(FixedBudget), got {other:?}"),
        }

        // Above budget: also stop (defensive against missed checks).
        let snap_over = build_snapshot(150);
        assert!(matches!(
            policy.should_halt(&snap_over, &edges),
            HaltDecision::Stop(HaltReason::FixedBudget)
        ));
    }

    /// P07: telemetry has the canonical name and schema_version=1.
    #[test]
    fn test_p07_legacy_az_telemetry() {
        let policy = LegacyAlphaZero::new(800);
        let tel = policy.telemetry();
        assert_eq!(tel.schema_version, 1);
        assert_eq!(tel.policy_name, "legacy_az");
        assert_eq!(tel.gap_bits, 0.0);
        assert_eq!(tel.bayes_voi, 0.0);
    }
}

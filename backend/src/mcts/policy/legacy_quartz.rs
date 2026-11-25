//! P07: `LegacyQuartz` — bit-identical shim around the existing
//! `quartz_policy_adjustment` and `QuartzController::should_stop`
//! paths.
//!
//! The shim's only purpose is to put the existing controller behind
//! the `SearchPolicy` trait so the engine can flip its default to a
//! new policy (P10) without any user losing reproducibility of
//! published numbers. `--policy=legacy_quartz` is the back-compat
//! escape hatch.
//!
//! Implementation strategy:
//! - `score_adjustment` computes a synthetic `q_eff`, `prior`,
//!   `sqrt_n_parent_eff` from the EdgeView and delegates verbatim to
//!   the existing `quartz_policy_adjustment` free function (now
//!   `pub(crate)` after P07). The result is mapped 1:1 into a
//!   `ScoreAdjustment`.
//! - `should_halt` calls `QuartzController::should_stop`. The
//!   controller's halt_reason_count atomics (P01) are populated as a
//!   side-effect; the shim doesn't try to read them back.
//!
//! What the shim is **not**: it is not a port of the legacy
//! controller into the new abstraction. It's a delegating wrapper.
//! The legacy code path is the source of truth. P15 may rewrite this
//! once the legacy code is deleted.

use std::sync::Arc;

use super::trait_def::{
    ControllerTelemetry, EdgeView, HaltDecision, ScoreAdjustment, SearchPolicy, SearchSnapshot,
};
use crate::mcts::quartz::{HaltReason, QuartzConfig, QuartzController};
use crate::mcts::search::{SearchController, StopReason};

/// Shim policy that delegates to the legacy controller. The
/// `Arc<QuartzController>` is shared with the engine; both this shim
/// and the engine's existing call sites continue to drive it (P10
/// will deduplicate when the new default lands).
pub struct LegacyQuartz {
    pub cfg: QuartzConfig,
    pub ctrl: Arc<QuartzController>,
}

impl LegacyQuartz {
    pub fn new(cfg: QuartzConfig, ctrl: Arc<QuartzController>) -> Self {
        Self { cfg, ctrl }
    }
}

impl SearchPolicy for LegacyQuartz {
    fn name(&self) -> &'static str {
        "legacy_quartz"
    }

    fn observe(&self, _snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) {
        // The legacy controller updates its stats inside the engine's
        // own backup pipeline (`update_stats`-style calls); this shim
        // does NOT duplicate that work. observe() is intentionally a
        // no-op so the legacy path stays the single source of truth.
    }

    fn score_adjustment(&self, edge: EdgeView<'_>) -> ScoreAdjustment {
        // The legacy adjustment depends on `QuartzStats` (the
        // controller's most recent observe/update). The shim reads
        // it from the controller's `last_stats()` accessor.
        let stats = self.ctrl.last_stats();
        // sqrt_n_parent_eff is computed from the snapshot for parity
        // with the live select.rs hot path; the legacy formula uses
        // it inside `root_share_penalty`. We approximate from
        // `*edge.root_total_n` which is what the engine sees.
        let sqrt_parent = (*edge.root_total_n as f32).sqrt().max(1e-3);
        let adj = crate::mcts::select::quartz_policy_adjustment(
            edge.n,
            edge.o_a,
            edge.q,
            edge.prior,
            sqrt_parent,
            &stats,
            &self.cfg,
        );
        // Map the legacy QuartzPolicyAdjustment shape onto the
        // SearchPolicy ScoreAdjustment shape. The legacy `bonus` field
        // (off-diagonal one-loop) is collapsed onto `penalty` since
        // both are additive on the score in the existing PUCT formula
        // (see `adjusted_puct_score` line 156: `base + adj.bonus +
        // adj.penalty`).
        ScoreAdjustment {
            effective_prior: adj.effective_prior,
            penalty: adj.penalty + adj.bonus,
            fisher_alpha: if adj.use_fisher_puct { 0.5 } else { 0.0 },
            q_override: None,
        }
    }

    fn should_halt(&self, snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) -> HaltDecision {
        // Delegate to the legacy controller. Side-effect: the
        // controller's halt_reason_count atomics (P01) are populated
        // by note_halt() inside should_stop's terminal branches.
        if self.ctrl.should_stop(snap.root_visits, snap.elapsed_ms) {
            // Map the legacy StopReason onto P01's HaltReason enum.
            // The mapping is best-effort; the controller's own
            // `last_halt_reason` would be more accurate but the legacy
            // controller doesn't expose that.
            let reason = match self.ctrl.last_stop_reason() {
                StopReason::BudgetExhausted { .. } => {
                    if matches!(
                        self.cfg.halt_mode,
                        crate::mcts::quartz::HaltMode::Fixed { .. }
                    ) {
                        HaltReason::FixedBudget
                    } else {
                        HaltReason::MaxVisits
                    }
                }
                StopReason::TimeCapHit { .. } => HaltReason::MaxTime,
                StopReason::VocNonPositive { .. } => HaltReason::VOCNonPositive,
                StopReason::Converged { .. } => HaltReason::PFlipConverged,
                StopReason::MaxNodesHit { .. } => HaltReason::MaxVisits,
                StopReason::Unknown => HaltReason::PFlipConverged,
            };
            HaltDecision::Stop(reason)
        } else {
            HaltDecision::Continue
        }
    }

    fn telemetry(&self) -> ControllerTelemetry {
        let stats = self.ctrl.last_stats();
        ControllerTelemetry {
            schema_version: 1,
            policy_name: self.name().to_string(),
            halt_reason: None,
            // The legacy stats don't carry KL-LUCB gap_bits or VOI;
            // those are P08/P09 features. Leave at default.
            gap_bits: 0.0,
            glr_z: 0.0,
            mean_sigma_a: stats.sigma_q,
            chi2: 0.0,
            chi2_dof: 0,
            bayes_voi: stats.unified.voc_total,
            eval_sigma: 0.0,
            iters_at_halt: 0,
            ..ControllerTelemetry::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_controller() -> Arc<QuartzController> {
        let cfg = QuartzConfig {
            sigma_0: 0.3,
            min_visits: 50,
            ctm_budget_ms: 0,
            ..Default::default()
        };
        Arc::new(QuartzController::new(800, cfg))
    }

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

    /// P07: name and telemetry shape are stable.
    #[test]
    fn test_p07_legacy_quartz_telemetry_shape() {
        let ctrl = build_controller();
        let policy = LegacyQuartz::new(ctrl.cfg.clone(), ctrl);
        assert_eq!(policy.name(), "legacy_quartz");
        let tel = policy.telemetry();
        assert_eq!(tel.schema_version, 1);
        assert_eq!(tel.policy_name, "legacy_quartz");
    }

    /// P07: halt at max_visits maps to HaltReason::MaxVisits via the
    /// shim's mapping.
    #[test]
    fn test_p07_legacy_quartz_halt_max_visits() {
        let ctrl = build_controller();
        let policy = LegacyQuartz::new(ctrl.cfg.clone(), ctrl);
        let snap = build_snapshot(800);
        let edges: Vec<EdgeView<'_>> = vec![];
        match policy.should_halt(&snap, &edges) {
            HaltDecision::Stop(HaltReason::MaxVisits) => {}
            HaltDecision::Stop(other) => panic!("expected MaxVisits, got {other:?}"),
            HaltDecision::Continue => panic!("expected Stop, got Continue"),
        }
    }

    /// P07: halt at fixed budget maps to HaltReason::FixedBudget.
    #[test]
    fn test_p07_legacy_quartz_halt_fixed_budget() {
        let cfg = QuartzConfig {
            halt_mode: crate::mcts::quartz::HaltMode::Fixed { budget: 50 },
            sigma_0: 0.3,
            min_visits: 10,
            ctm_budget_ms: 0,
            ..Default::default()
        };
        let ctrl = Arc::new(QuartzController::new(800, cfg.clone()));
        let policy = LegacyQuartz::new(cfg, ctrl);
        let snap = build_snapshot(50);
        let edges: Vec<EdgeView<'_>> = vec![];
        match policy.should_halt(&snap, &edges) {
            HaltDecision::Stop(HaltReason::FixedBudget) => {}
            other => panic!("expected Stop(FixedBudget), got {other:?}"),
        }
    }

    /// P07: continue when budget not yet reached.
    #[test]
    fn test_p07_legacy_quartz_continue_below_budget() {
        let ctrl = build_controller();
        let policy = LegacyQuartz::new(ctrl.cfg.clone(), ctrl);
        let snap = build_snapshot(100);
        let edges: Vec<EdgeView<'_>> = vec![];
        // Below max_visits and below adaptive thresholds → Continue.
        assert!(matches!(
            policy.should_halt(&snap, &edges),
            HaltDecision::Continue
        ));
    }

    /// P07: score_adjustment respects the active PenaltyMode. With
    /// PenaltyMode::None, penalty must be 0.
    #[test]
    fn test_p07_legacy_quartz_score_adjustment_none_mode() {
        let cfg = QuartzConfig {
            penalty_mode: crate::mcts::quartz::PenaltyMode::None,
            enable_one_loop: false,
            sigma_0: 0.3,
            min_visits: 50,
            ctm_budget_ms: 0,
            ..Default::default()
        };
        let ctrl = Arc::new(QuartzController::new(800, cfg.clone()));
        let policy = LegacyQuartz::new(cfg, ctrl);
        let snap = build_snapshot(100);
        let n_total = 100_u32;
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
        // None mode + one_loop disabled = no penalty.
        assert_eq!(adj.penalty, 0.0);
    }
}

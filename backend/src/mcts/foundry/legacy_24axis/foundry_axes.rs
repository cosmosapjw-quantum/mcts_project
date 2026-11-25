//! QUARTZ idea-foundry axis skeletons (not wired into `policy/mod.rs`).
//!
//! The purpose of this file is to pin the Rust-side class/trait shapes for the
//! online and shadow axes documented under `docs/idea_foundry/`.  The methods
//! deliberately return conservative or shadow proposals; each lane receives a
//! separate promotion commit after its Phase-15 mechanism gate.

use std::collections::BTreeMap;
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::mcts::policy::{
    ControllerTelemetry, EdgeView, HaltDecision, ScoreAdjustment, SearchPolicy, SearchSnapshot,
};
use crate::mcts::quartz::HaltReason;

use super::foundry_contracts::{
    AxisMode, FoundryAxis, FoundryCache, FoundryRootView, MetaActionKind, MetaCost,
    MetaProposal,
};

fn edge_total_radius(edge: EdgeView<'_>) -> f32 {
    edge.sigma_a(4.0) + edge.stats.sigma_eval.unwrap_or(0.0)
}

fn best_edge(edges: &[EdgeView<'_>]) -> Option<EdgeView<'_>> {
    edges
        .iter()
        .copied()
        .filter(|edge| edge.n > 0)
        .max_by(|lhs, rhs| lhs.q.total_cmp(&rhs.q))
}

#[derive(Clone, Copy, Debug)]
pub struct StopCouncilAxis {
    pub min_visits: u32,
    pub max_wrong_risk: f32,
}

impl Default for StopCouncilAxis {
    fn default() -> Self {
        Self {
            min_visits: 16,
            max_wrong_risk: 0.05,
        }
    }
}

impl FoundryAxis for StopCouncilAxis {
    fn axis_id(&self) -> &'static str { "A01_stop_council" }
    fn mode(&self) -> AxisMode { AxisMode::Shadow }

    fn propose(&self, view: &FoundryRootView<'_>) -> Vec<MetaProposal> {
        if !view.fresh || view.snapshot.root_visits < self.min_visits {
            return Vec::new();
        }
        let Some(best) = best_edge(view.edges) else { return Vec::new(); };
        let risk = view
            .candidate_omission_risk
            .max(view.p_flip.unwrap_or(1.0))
            .max(1.0 - view.h1_stability.unwrap_or(0.0));
        if risk > self.max_wrong_risk { return Vec::new(); }
        vec![MetaProposal {
            axis_id: self.axis_id(),
            kind: MetaActionKind::Stop,
            target_edge_pos: vec![best.idx],
            amount: 0,
            expected_regret_reduction: 0.0,
            regret_reduction_lcb: 0.0,
            cost: MetaCost::default(),
            confidence: 1.0 - risk,
            activation_guard: "fresh + calibrated risk + arbiter no-positive-compute check",
            explanation: "STOP proposal only; the arbiter still checks all computation proposals.",
            telemetry: BTreeMap::from([("estimated_wrong_risk".to_string(), risk as f64)]),
        }]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StaticAnchorRpoAxis;
impl FoundryAxis for StaticAnchorRpoAxis {
    fn axis_id(&self) -> &'static str { "A02_static_anchor_rpo" }
    fn mode(&self) -> AxisMode { AxisMode::Shadow }
    fn propose(&self, _view: &FoundryRootView<'_>) -> Vec<MetaProposal> {
        vec![MetaProposal::shadow(
            self.axis_id(),
            "Compute a temporary root policy from the immutable network anchor; preserve hidden anchor mass.",
        )]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UncertaintyAxis;
impl FoundryAxis for UncertaintyAxis {
    fn axis_id(&self) -> &'static str { "A03_uncertainty_decomposition" }
    fn mode(&self) -> AxisMode { AxisMode::Shadow }
    fn propose(&self, view: &FoundryRootView<'_>) -> Vec<MetaProposal> {
        let mean_radius = if view.edges.is_empty() {
            0.0
        } else {
            view.edges.iter().copied().map(edge_total_radius).sum::<f32>() / view.edges.len() as f32
        };
        let mut proposal = MetaProposal::shadow(
            self.axis_id(),
            "Publish distinct uncertainty channels; never call Welford dispersion evaluator epistemic uncertainty.",
        );
        proposal.telemetry.insert("mean_available_radius".to_string(), mean_radius as f64);
        vec![proposal]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KgAllocationAxis { pub amount: u32 }
impl Default for KgAllocationAxis { fn default() -> Self { Self { amount: 8 } } }
impl FoundryAxis for KgAllocationAxis {
    fn axis_id(&self) -> &'static str { "A04_kg_voi_allocator" }
    fn mode(&self) -> AxisMode { AxisMode::Shadow }
    fn propose(&self, view: &FoundryRootView<'_>) -> Vec<MetaProposal> {
        let Some(best) = best_edge(view.edges) else { return Vec::new(); };
        let challenger = view.edges.iter().copied().filter(|edge| edge.idx != best.idx).max_by(|lhs, rhs| {
            let lhs_score = edge_total_radius(*lhs) / (lhs.n.max(1) as f32);
            let rhs_score = edge_total_radius(*rhs) / (rhs.n.max(1) as f32);
            lhs_score.total_cmp(&rhs_score)
        });
        let Some(challenger) = challenger else { return Vec::new(); };
        vec![MetaProposal {
            axis_id: self.axis_id(),
            kind: MetaActionKind::Sample,
            target_edge_pos: vec![challenger.idx],
            amount: self.amount,
            expected_regret_reduction: 0.0,
            regret_reduction_lcb: 0.0,
            cost: MetaCost { nn_evals: self.amount as f32, ..MetaCost::default() },
            confidence: 0.0,
            activation_guard: "allocation-only; Stage-7 low-budget KG stopping remains closed",
            explanation: "Resolve the most uncertain incumbent challenger per measured cost.",
            telemetry: BTreeMap::new(),
        }]
    }
}

macro_rules! shadow_axis {
    ($name:ident, $id:literal, $mode:expr, $message:literal) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
        impl FoundryAxis for $name {
            fn axis_id(&self) -> &'static str { $id }
            fn mode(&self) -> AxisMode { $mode }
            fn propose(&self, _view: &FoundryRootView<'_>) -> Vec<MetaProposal> {
                vec![MetaProposal::shadow(self.axis_id(), $message)]
            }
        }
    };
}

shadow_axis!(GumbelSequentialHalvingAxis, "A05_gumbel_sequential_halving", AxisMode::Shadow, "Persist a resumable root bracket; do not replace interior PUCT.");
shadow_axis!(ResidualEvidenceAxis, "A06_residual_evidence_widening", AxisMode::Shadow, "Bound hidden anchor-weighted posterior mass and emit WIDEN only above the truncation threshold.");
shadow_axis!(JlbRootAxis, "A07_jsd_locally_balanced_root", AxisMode::Shadow, "Use sibling JSD as geometry and regularized P/Q as the target density.");
shadow_axis!(DynamicLiveSetAxis, "A08_dynamic_live_set_particle", AxisMode::Shadow, "Allocate independent root particle groups with hibernation and resurrection.");
shadow_axis!(TacticalProofAxis, "A09_tactical_sentinel_proof", AxisMode::Shadow, "Run a bounded game-specific proof action before statistical elimination.");
shadow_axis!(PriorRefreshSpecialistAxis, "A10_prior_refresh_specialist", AxisMode::Shadow, "Preserve historical refresh only as a router-selected weak-evaluator/shift specialist.");
shadow_axis!(EntropyMarginRouterAxis, "A11_entropy_margin_regime_router", AxisMode::Shadow, "Replace the zero-fire H3 binary gate with a continuous entropy/margin change-point feature.");
shadow_axis!(ServiceCurveAxis, "A12_service_curve_scheduler", AxisMode::Shadow, "Load a hardware-specific service-curve artifact and propose batch/inflight/thread settings.");
shadow_axis!(PendingFlowAxis, "A13_pending_flow_wu_uct", AxisMode::Shadow, "Track incomplete simulations separately; pending counts never enter evidence or STOP certificates.");
shadow_axis!(SemanticPathLshAxis, "A14_semantic_path_lsh", AxisMode::Shadow, "Control whole-path semantic overlap only after edge-level contention is already managed.");
shadow_axis!(B13ReadoutAxis, "A15_b13_curvature_readout", AxisMode::Shadow, "Keep curvature as a post-hoc readout/target lane until an independent selection benefit is shown.");
shadow_axis!(SignedPathAxis, "A16_coherence_signed_path_shadow", AxisMode::AnalysisOnly, "Log a bounded signed disagreement feature and its classical decay; never call it a quantum amplitude.");
shadow_axis!(MentsAxis, "A22_ments_decaying_entropy", AxisMode::Shadow, "Expose root/shallow decaying entropy as an opt-in BTS/DENTS/MENTS comparator.");
shadow_axis!(GraphConsistencyAxis, "A23_graph_state_sharing_consistency", AxisMode::Shadow, "Share state/evaluation identity first; parent-edge visits and priors remain parent-specific.");

pub struct ShadowAxisPolicy<A: FoundryAxis> {
    axis: A,
    cache: ArcSwap<FoundryCache>,
}

impl<A: FoundryAxis> ShadowAxisPolicy<A> {
    pub fn new(axis: A) -> Self {
        Self { axis, cache: ArcSwap::from_pointee(FoundryCache::default()) }
    }
    pub fn proposals(&self) -> Arc<FoundryCache> { self.cache.load_full() }
}

impl<A: FoundryAxis + 'static> SearchPolicy for ShadowAxisPolicy<A> {
    fn name(&self) -> &'static str { self.axis.axis_id() }

    fn observe(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]) {
        let view = FoundryRootView::minimal(snap, edges);
        self.cache.store(Arc::new(FoundryCache {
            epoch: snap.iteration,
            root_visits: snap.root_visits,
            proposals: self.axis.propose(&view),
        }));
    }

    fn score_adjustment(&self, _edge: EdgeView<'_>) -> ScoreAdjustment { ScoreAdjustment::default() }

    fn should_halt(&self, snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) -> HaltDecision {
        if self.axis.mode() != AxisMode::Online {
            return HaltDecision::Continue;
        }
        let cache = self.cache.load();
        if cache.epoch != snap.iteration { return HaltDecision::Continue; }
        if cache.proposals.iter().any(|proposal| proposal.kind == MetaActionKind::Stop) {
            return HaltDecision::Stop(HaltReason::PolicyConverged);
        }
        HaltDecision::Continue
    }

    fn telemetry(&self) -> ControllerTelemetry {
        let cache = self.cache.load();
        ControllerTelemetry {
            schema_version: 1,
            policy_name: self.axis.axis_id().to_string(),
            iters_at_halt: cache.epoch,
            ..ControllerTelemetry::default()
        }
    }
}

//! QUARTZ idea-foundry runtime contracts (design skeleton; not wired by default).
//!
//! This module is intentionally not declared from `policy/mod.rs` yet.  A lane
//! must first pass its Python/Phase-15 shadow gate, then add the module export
//! and focused Rust tests in the promotion commit.

use std::collections::BTreeMap;

use crate::mcts::policy::{EdgeView, SearchSnapshot};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisMode {
    Shadow,
    Online,
    AnalysisOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetaActionKind {
    Sample,
    Challenge,
    Widen,
    Deepen,
    Prove,
    ReweightPolicy,
    MergeOrShare,
    SetBatch,
    SetInflight,
    SetThreads,
    Stop,
    ShadowOnly,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MetaCost {
    pub nn_evals: f32,
    pub cpu_ms: f32,
    pub gpu_ms: f32,
    pub energy_proxy: f32,
}

impl MetaCost {
    #[inline]
    pub fn provisional_scalar(self) -> f32 {
        self.nn_evals + self.cpu_ms + self.gpu_ms + self.energy_proxy
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct MetaProposal {
    pub axis_id: &'static str,
    pub kind: MetaActionKind,
    pub target_edge_pos: Vec<u16>,
    pub amount: u32,
    pub expected_regret_reduction: f32,
    pub regret_reduction_lcb: f32,
    pub cost: MetaCost,
    pub confidence: f32,
    pub activation_guard: &'static str,
    pub explanation: &'static str,
    pub telemetry: BTreeMap<String, f64>,
}

impl MetaProposal {
    #[inline]
    pub fn conservative_utility(&self) -> f32 {
        self.regret_reduction_lcb - self.cost.provisional_scalar()
    }

    pub fn shadow(axis_id: &'static str, explanation: &'static str) -> Self {
        Self {
            axis_id,
            kind: MetaActionKind::ShadowOnly,
            target_edge_pos: Vec::new(),
            amount: 0,
            expected_regret_reduction: 0.0,
            regret_reduction_lcb: 0.0,
            cost: MetaCost::default(),
            confidence: 0.0,
            activation_guard: "shadow-only",
            explanation,
            telemetry: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct FoundryRuntimeView {
    pub threads: u16,
    pub batch_size: u16,
    pub inflight: u16,
    pub queue_wait_ms: f32,
    pub eval_latency_ms: f32,
    pub edge_duplicate_rate: f32,
    pub semantic_path_overlap: f32,
    pub tt_wait_ms: f32,
}

pub struct FoundryRootView<'a> {
    pub snapshot: &'a SearchSnapshot,
    pub edges: &'a [EdgeView<'a>],
    pub runtime: FoundryRuntimeView,
    pub search_epoch: u64,
    pub fresh: bool,
    pub h1_stability: Option<f32>,
    pub p_flip: Option<f32>,
    pub top2_margin: f32,
    pub margin_slope: f32,
    pub entropy_slope: f32,
    pub candidate_omission_risk: f32,
}

impl<'a> FoundryRootView<'a> {
    pub fn minimal(snapshot: &'a SearchSnapshot, edges: &'a [EdgeView<'a>]) -> Self {
        Self {
            snapshot,
            edges,
            runtime: FoundryRuntimeView::default(),
            search_epoch: snapshot.iteration,
            fresh: true,
            h1_stability: None,
            p_flip: None,
            top2_margin: 0.0,
            margin_slope: 0.0,
            entropy_slope: 0.0,
            candidate_omission_risk: 1.0,
        }
    }

    #[inline]
    pub fn edge(&self, edge_pos: u16) -> Option<EdgeView<'a>> {
        self.edges.iter().copied().find(|edge| edge.idx == edge_pos)
    }
}

pub trait FoundryAxis: Send + Sync {
    fn axis_id(&self) -> &'static str;
    fn mode(&self) -> AxisMode;
    fn propose(&self, view: &FoundryRootView<'_>) -> Vec<MetaProposal>;
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct FoundryCache {
    pub epoch: u64,
    pub root_visits: u32,
    pub proposals: Vec<MetaProposal>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConservativeArbiter;

impl ConservativeArbiter {
    pub fn select<'a>(&self, proposals: &'a [MetaProposal], fresh: bool) -> Option<&'a MetaProposal> {
        proposals
            .iter()
            .filter(|proposal| fresh || proposal.kind != MetaActionKind::Stop)
            .filter(|proposal| {
                proposal.kind == MetaActionKind::Stop
                    || proposal.conservative_utility().is_sign_positive()
            })
            .max_by(|lhs, rhs| lhs.conservative_utility().total_cmp(&rhs.conservative_utility()))
    }
}

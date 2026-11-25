//! P06: types and trait for the unified `SearchPolicy` interface.
//!
//! This file is *scaffolding only* in P06: nothing in the search loop
//! consults the trait yet. P07 lands the LegacyQuartz shim that wraps
//! the existing `quartz_policy_adjustment` + `should_stop` path behind
//! the trait so behavior is bit-identical. P08–P09 add named policies
//! (KLLUCBStop, BayesianQuartz). P10 flips the default.
//!
//! Naming and field choices follow the design doc at
//! `~/.claude/plans/iridescent-giggling-bachman.md` §P06.
//!
//! Why this trait exists: the current 7×4×3×2¹¹ ≈ 229k-combination
//! mode surface (PenaltyMode × HaltMode × CostMode × 11 booleans)
//! makes "controller mode A vs B" comparisons confounded — changing
//! penalty mode silently changes halt iteration count, refresh
//! probability, and search-tree depth distribution. The trait
//! collapses this surface to a small set of named, composable
//! policies (LegacyAlphaZero, LegacyQuartz, KLLUCBStop,
//! BayesianQuartz, MENTSEntropyRegularized) where each policy has
//! one named author and one set of hyperparameters.

use crate::mcts::quartz::HaltReason;

/// Snapshot of root-side search state. Refreshed at the controller's
/// periodic boundary (every `check_interval` iterations) under whatever
/// synchronization the controller uses internally. Cheap to clone — POD
/// only, no Arc/Vec inside the struct itself. Per-edge data is borrowed
/// via `EdgeView`.
#[derive(Clone, Debug)]
pub struct SearchSnapshot {
    /// Total visits at the root since search began.
    pub root_visits: u32,
    /// Number of root children with at least one entry in the candidate
    /// list (materialized + non-materialized).
    pub n_children: u16,
    /// Number of root children currently materialized. Useful for
    /// progressive widening policies.
    pub n_visible: u16,
    /// Wall-clock elapsed since search start.
    pub elapsed_ms: u64,
    /// Maximum tree depth observed.
    pub depth_max: u16,
    /// Visit-weighted mean Q at root (the "value" reported in standard
    /// AlphaZero output).
    pub mean_q_root: f32,
    /// Visit-weighted standard deviation of root child Q-means.
    /// Substitutes for σ_Q in the legacy controller.
    pub sigma_q_root: f32,
    /// Optional ensemble-derived per-state evaluator uncertainty
    /// (see `Evaluator::evaluate_with_uncertainty`). When `None`, the
    /// policy must fall back to search-only uncertainty.
    pub sigma_eval: Option<f32>,
    /// Monotonic iteration counter; in parallel search this is NOT
    /// equal to root_visits because workers can be mid-rollout. Use
    /// this for time-decay calculations that need a clock independent
    /// of completion order.
    pub iteration: u64,
    /// Index of the empirical-best root child (argmax Q over visited
    /// children). Caller's choice whether to use Q or N as the rank
    /// statistic; the snapshot just records the controller's pick.
    pub best_idx: u16,
    /// Index of the empirical second-best root child. Used by halt
    /// rules that compare top two arms.
    pub second_idx: u16,
}

/// Per-edge view exposed read-only to a policy. All fields are computed
/// or cached on the edge struct already; this is just a borrow surface
/// so policies can be tested in isolation from the live MCTS engine.
///
/// `m2` is the Welford running sum of squared deviations; the policy
/// converts this to a posterior std-dev via `sigma_a()` with the
/// caller's choice of prior pseudo-count. f64 to avoid f32 drift past
/// N=10⁵ (verified empirically — f32 Welford diverges by ~5e-4 from
/// scipy.var at 10⁶ samples; f64 stays within 1e-9).
#[derive(Copy, Clone, Debug)]
pub struct EdgeView<'a> {
    pub idx: u16,
    /// Real visits to this edge.
    pub n: u32,
    /// Virtual-loss reservation count (parallel selection).
    pub n_virtual: u32,
    /// "Outside" / un-materialized neighbor visits (PUCT denominator).
    pub o_a: u32,
    /// Mean Q value seen at this edge.
    pub q: f32,
    /// Sum of values seen at this edge (Welford).
    pub q_sum: f32,
    /// Welford running sum of squared deviations from running mean.
    pub m2: f64,
    /// Raw policy prior π₀(a) from the network.
    pub prior: f32,
    pub depth: u16,
    /// Last leaf value backed up through this edge.
    pub last_value: f32,
    /// Per-edge contribution to root KL: q_T(a)·log(q_T(a)/π₀(a)).
    /// Pre-aggregated so policies can sum without recomputing logs.
    pub envar_partial: f32,
    pub root_total_n: &'a u32,
    pub stats: &'a SearchSnapshot,
}

impl<'a> EdgeView<'a> {
    /// Per-action posterior std deviation with empirical-Bayes
    /// variance shrinkage.
    ///
    /// Formula: σ_a² = (M2 + λ₀ · σ_root²) / (max(N − 1, 1) + λ₀)
    ///
    /// Reference: Welford 1962 (online variance) + standard
    /// Normal-inverse-Gamma conjugate posterior with prior pseudo-count
    /// `lambda0` and prior variance σ_root² (the visit-weighted variance
    /// of root Q-means, available as `stats.sigma_q_root²`). Returns
    /// at least 1e-3 to avoid downstream divisions by zero.
    ///
    /// A1-c audit fix: the denominator is `max(N-1, 1) + λ₀`, matching
    /// the validated Python prototype (`belief.py::empirical_bayes_shrinkage`,
    /// audit §6.1's canonical bias-corrected form — it aligns with the
    /// unbiased sample-variance denominator `N-1`, floored at 1 to
    /// handle N ∈ {0, 1}). This file previously used `N + λ₀`, which
    /// disagreed with the prototype by one effective count at every N
    /// (e.g. σ=0.300 vs 0.268 at N=0) despite the design doc's
    /// "bit-identical numerical behavior between Python prototype and
    /// Rust" claim. `max(N-1,1)` collapses N=0 and N=1 to the same
    /// effective count, matching `belief.py`'s documented hand-derived
    /// values for both.
    ///
    /// `lambda0=4` is the canonical weak-prior choice (matches α=2,
    /// β=2σ_root² in shape-rate parameterization). Tunable but rarely
    /// helpful to change.
    #[inline]
    pub fn sigma_a(&self, lambda0: f32) -> f32 {
        let n = (self.n as f32 - 1.0).max(1.0) + lambda0;
        let prior_var = self.stats.sigma_q_root.max(1e-4).powi(2);
        let m2 = self.m2 as f32 + lambda0 * prior_var;
        (m2 / n).sqrt().max(1e-3)
    }
}

/// Score-side adjustment a policy emits for each candidate edge during
/// selection. Drives PUCT: final_score = base_puct(effective_prior, q,
/// n) × fisher_alpha-aware factor + penalty (+ optional q_override).
///
/// Defaults to all-zero / identity ⇒ pure PUCT, no policy effect.
#[derive(Default, Copy, Clone, Debug)]
pub struct ScoreAdjustment {
    /// Override of the network prior. Set equal to `e.prior` for
    /// "no refresh" policies.
    pub effective_prior: f32,
    /// Additive penalty term on the score. Negative discourages
    /// revisits.
    pub penalty: f32,
    /// 0.0 ⇒ standard PUCT; 0.5 ⇒ Fisher-natural-gradient √π scaling.
    /// Reserved for ablation; default is 0.0 because empirical
    /// experiments showed √π hurts with weak priors.
    pub fisher_alpha: f32,
    /// Optional override of Q itself (for regularized-policy variants
    /// like Grill et al. 2020's closed-form solver). None ⇒ use the
    /// edge's own Q.
    pub q_override: Option<f32>,
}

/// Returned by `should_halt`; carries the reason for telemetry attribution.
#[derive(Copy, Clone, Debug)]
pub enum HaltDecision {
    Continue,
    Stop(HaltReason),
}

/// Returned by `refresh_prior`. Default behavior is "no refresh"
/// (use the raw prior); policies that mix prior with a posterior
/// signal return Some(EffectivePrior).
#[derive(Copy, Clone, Debug)]
pub struct EffectivePrior {
    pub raw: f32,
    pub posterior: f32,
    /// ρ ∈ [0, 1]; effective = (1-ρ)·raw + ρ·posterior in some
    /// reference space (linear, log, or visit-share — depends on
    /// policy).
    pub blend: f32,
}

/// One-shot summary of a policy's per-search state. Goes into the
/// JSON `controller_summary` block alongside the existing extended
/// telemetry from P01.
#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct ControllerTelemetry {
    pub schema_version: u8,
    pub policy_name: String,
    pub halt_reason: Option<String>,
    /// PAC certificate gap from KL-LUCB (Kaufmann-Kalyanakrishnan 2013):
    /// `gap_bits = N_b̂ · KL(μ̂_b̂, μ̂_c) − β(t,δ)`. Positive ⇒ stop allowed.
    pub gap_bits: f32,
    /// Generalized likelihood ratio z-statistic (Garivier-Kaufmann
    /// 2016 Track-and-Stop). Reserved for a future GLR upgrade.
    pub glr_z: f32,
    /// Mean σ_a across candidates (BayesianQuartz). Snapshot at the
    /// last `observe`.
    pub mean_sigma_a: f32,
    /// Pearson χ² goodness-of-fit of visit distribution vs network
    /// prior. Replaces the legacy ε-envariance Pinsker-bound test.
    pub chi2: f32,
    /// Degrees of freedom for the χ² test (= n_children - 1, capped
    /// at 1).
    pub chi2_dof: u32,
    /// Russo-Van Roy one-step value-of-information; max over
    /// candidates. Replaces the hand-built VOC = P_flip × σ_Δ − cost.
    pub bayes_voi: f32,
    /// Mean ensemble-derived evaluator uncertainty (BootstrapDQN-style),
    /// or 0.0 when the evaluator doesn't expose this.
    pub eval_sigma: f32,
    /// Iteration count at halt (0 if still running).
    pub iters_at_halt: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub foundry_mode: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub foundry_axis_ids: Vec<String>,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub metacontroller_decisions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub metacontroller_proposals: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub metacontroller_actions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub metacontroller_coordination_errors: u64,
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

/// The unified policy interface. All current PenaltyMode/HaltMode/
/// CostMode/flag combinations are absorbed into named impls of this
/// trait once P07–P11 land.
///
/// Hot-path contract: `score_adjustment` is called on every selection
/// step and must be branch-light — the policy's heavy computation
/// must be cached by `observe`, which fires at most every
/// `check_interval` iterations under whatever synchronization the
/// controller uses (typically a Mutex or RwLock).
///
/// `&self` everywhere (not `&mut self`) so multiple worker threads
/// can share one Arc<dyn SearchPolicy>. Internal mutability is the
/// implementor's responsibility (parking_lot::Mutex<Cache> is the
/// canonical choice).
pub trait SearchPolicy: Send + Sync {
    /// Stable name used as the JSON `policy_name` field. Pin it; do
    /// not rename without bumping the appropriate schema version.
    fn name(&self) -> &'static str;

    /// Refresh the policy's heavy state from the latest snapshot.
    /// Called at the controller's periodic boundary (every
    /// `check_interval` iterations); typical implementations use
    /// `parking_lot::Mutex<Cache>` to guard the write. `&self` so the
    /// engine can hold an `Arc<dyn SearchPolicy>` across workers.
    fn observe(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]);

    /// Per-selection-step adjustment. Hot path; must be O(1) reads
    /// from the cache populated by `observe`.
    fn score_adjustment(&self, edge: EdgeView<'_>) -> ScoreAdjustment;

    /// Halt decision based on the latest snapshot. Periodicity is
    /// enforced by the framework, not the policy.
    fn should_halt(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]) -> HaltDecision;

    /// Optional per-edge prior refresh. Default = no refresh.
    fn refresh_prior(&self, _e: EdgeView<'_>) -> Option<EffectivePrior> {
        None
    }

    /// One-shot telemetry snapshot for end-of-search aggregation.
    fn telemetry(&self) -> ControllerTelemetry;
}

/// Type alias used by the engine when stashing a boxed policy.
pub type BoxedPolicy = Box<dyn SearchPolicy>;

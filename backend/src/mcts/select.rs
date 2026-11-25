//! PUCT Selection (v0.7)
//! Standard PUCT + Fisher Metric PUCT (§4 Information Geometry)
//! + EFT-PUCT (§6.1.2 one-loop bonus + visit penalty)

use std::sync::atomic::{AtomicU64, Ordering};

use smallvec::SmallVec;

use crate::game::GameState;
use crate::mcts::expand::{materialize_edges_in_place, materialize_edges_in_place_best_effort};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::node::{atomic_f64_load, ArenaRef, MctsEdge, MctsNode, PathEdge};
use crate::mcts::quartz::{
    eft_action_bonus, fisher_prior_weight, one_loop_visit_penalty, penalty_mode_idx, QuartzConfig,
    QuartzStats, PENALTY_MODE_COUNT,
};
use crate::mcts::tt::TranspositionTable;

// ─────────────────────────────────────────────
// § PUCT Variants
// ─────────────────────────────────────────────

/// Standard AlphaZero PUCT (baseline)
#[inline]
pub fn puct_score(
    n_eff: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    n_parent_eff: u32,
    c_puct: f32,
) -> f32 {
    puct_score_with_parent_sqrt(
        n_eff,
        q_eff,
        prior,
        noise_adj,
        (n_parent_eff as f32).sqrt(),
        c_puct,
    )
}

#[inline]
fn puct_score_with_parent_sqrt(
    n_eff: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
) -> f32 {
    let p = (prior + noise_adj).max(0.0);
    let explore = c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32);
    q_eff + explore
}

#[inline]
fn blended_prior(base_prior: f32, alt_prior: f32, rho: f32) -> f32 {
    let rho = rho.clamp(0.0, 1.0);
    if rho <= 1e-6 {
        return base_prior.max(1e-8);
    }
    let log_p0 = base_prior.max(1e-8).ln();
    let log_p1 = alt_prior.max(1e-8).ln();
    ((1.0 - rho) * log_p0 + rho * log_p1).exp().max(1e-8)
}

#[inline]
fn root_share_penalty(n_raw: u32, sqrt_n_parent_eff: f32, hbar_eff: f32, cap: f32) -> f32 {
    if n_raw == 0 {
        return 0.0;
    }
    let n_parent_eff = (sqrt_n_parent_eff * sqrt_n_parent_eff).max(1.0);
    let h_cap = hbar_eff.min(cap);
    -(h_cap * n_raw as f32 / n_parent_eff)
}

/// PR-1B: Ablation-aware PUCT — respects penalty_mode and prior refresh settings.
/// SelfAdaptive mode: state-derived with fixed constants. All inputs from search observables.
#[inline]
pub fn ablation_puct_score(
    n_eff: u32,
    n_raw: u32,
    o_a: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    n_parent_eff: u32,
    c_puct: f32,
    stats: &QuartzStats,
    qcfg: &QuartzConfig,
) -> f32 {
    ablation_puct_score_with_parent_sqrt(
        n_eff,
        n_raw,
        o_a,
        q_eff,
        prior,
        noise_adj,
        (n_parent_eff as f32).sqrt(),
        c_puct,
        stats,
        qcfg,
    )
}

#[inline]
fn ablation_puct_score_with_parent_sqrt(
    n_eff: u32,
    n_raw: u32,
    o_a: u32,
    q_eff: f32,
    prior: f32,
    noise_adj: f32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
    stats: &QuartzStats,
    qcfg: &QuartzConfig,
) -> f32 {
    let adj = quartz_policy_adjustment(n_raw, o_a, q_eff, prior, sqrt_n_parent_eff, stats, qcfg);
    adjusted_puct_score(
        n_eff,
        q_eff,
        adj.effective_prior,
        noise_adj,
        sqrt_n_parent_eff,
        c_puct,
        adj,
    )
}

/// Internal score-side adjustment produced by `quartz_policy_adjustment`.
/// P07: now `pub(crate)` so the new `LegacyQuartz` policy can read it
/// when shimming the existing controller path through the SearchPolicy
/// trait. Field semantics unchanged from pre-P07.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuartzPolicyAdjustment {
    pub(crate) effective_prior: f32,
    pub(crate) penalty: f32,
    pub(crate) bonus: f32,
    pub(crate) use_fisher_puct: bool,
}

#[inline]
fn adjusted_puct_score(
    n_eff: u32,
    q_eff: f32,
    effective_prior: f32,
    noise_adj: f32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
    adj: QuartzPolicyAdjustment,
) -> f32 {
    let p = (effective_prior + noise_adj).max(0.0);
    let base = if adj.use_fisher_puct {
        let p_fish = fisher_prior_weight(p);
        q_eff + c_puct * p_fish * sqrt_n_parent_eff / (1.0 + n_eff as f32)
    } else {
        q_eff + c_puct * p * sqrt_n_parent_eff / (1.0 + n_eff as f32)
    };
    base + adj.bonus + adj.penalty
}

/// P07: now `pub(crate)` so `policy::legacy_quartz::LegacyQuartz` can
/// reuse the existing penalty/refresh dispatch verbatim. Behavior
/// unchanged from pre-P07.
#[inline]
pub(crate) fn quartz_policy_adjustment(
    n_raw: u32,
    o_a: u32,
    q_eff: f32,
    prior: f32,
    sqrt_n_parent_eff: f32,
    stats: &QuartzStats,
    qcfg: &QuartzConfig,
) -> QuartzPolicyAdjustment {
    use crate::mcts::quartz::PenaltyMode;

    // ═══════════════════════════════════════════
    // SelfAdaptive: zero config reads for penalty + refresh
    // All derived from: sigma_q, surprise_kl, n_visible, root_visits
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::SelfAdaptive {
        let k = (stats.n_visible as f32).max(2.0);
        let n_total = (stats.root_visits as f32).max(1.0);
        let sigma_q = stats.sigma_q.max(0.001);

        // ── Penalty: ν = σ_Q  →  -σ_Q / M_a ──
        let penalty = if n_raw > 0 || o_a > 0 {
            let m_a = 1.0 + n_raw as f32 + o_a as f32;
            -sigma_q / m_a
        } else {
            0.0
        };

        // ── Dynamic prior refresh: per-action Bayesian weight ──
        // α_a = N_a / (N_a + K)  — no global ρ, no hardcoded constants
        //   N_a=0 → α=0 (pure original prior)
        //   N_a=K → α=0.5 (balanced blend)
        //   N_a>>K → α→1 (fully visit-based)
        // τ = ln(1 + N_total/K)  — self-normalizing temperature
        let effective_prior = if n_raw > 0 {
            let n_a = n_raw as f32;
            let alpha_a = n_a / (n_a + k);
            let n_avg = n_total / k;
            let tau = (1.0 + n_avg).ln().max(0.1);
            let log_p0 = prior.max(1e-8).ln();
            let log_visit = (1.0 + n_a).ln();
            let log_pt = (1.0 - alpha_a) * log_p0 + alpha_a * log_visit / tau;
            log_pt.exp().max(1e-8)
        } else {
            prior
        };

        return QuartzPolicyAdjustment {
            effective_prior,
            penalty,
            bonus: 0.0,
            use_fisher_puct: false,
        };
    }

    // ═══════════════════════════════════════════
    // Canonical GatedRefresh
    // Root-only visit-share penalty + divergence-gated visit refresh.
    //
    // - penalty(a) = -min(ħ_eff, cap) * N_a / N_parent
    // - gate opens when prior_q_divergence exceeds ε_t
    // - refresh blends original prior with visit share at the root
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::GatedRefresh {
        let penalty = root_share_penalty(
            n_raw,
            sqrt_n_parent_eff,
            stats.hbar_eff,
            qcfg.hbar_penalty_cap,
        );

        let effective_prior = if n_raw > 0 && stats.root_visits > 0 {
            let gate_threshold = stats.epsilon_t.max(1e-6);
            let divergence = stats.prior_q_divergence.max(0.0);
            if divergence > gate_threshold {
                let rho_t = ((divergence - gate_threshold) / divergence.max(gate_threshold))
                    .clamp(0.0, 1.0);
                let visit_share = (n_raw as f32 / stats.root_visits as f32).max(1e-8);
                blended_prior(prior, visit_share, rho_t)
            } else {
                prior
            }
        } else {
            prior
        };

        return QuartzPolicyAdjustment {
            effective_prior,
            penalty,
            bonus: 0.0,
            use_fisher_puct: qcfg.enable_fisher_puct,
        };
    }

    // ═══════════════════════════════════════════
    // Historical GatedRefresh heuristic kept for direct ablation
    // against the canonical doc-aligned controller.
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::GatedRefreshLegacy {
        let penalty = crate::mcts::quartz::effective_penalty_v2(n_raw, o_a, qcfg.hbar_penalty_cap);

        let flip_thresh = 0.159_f32;
        let rho_max = 0.3_f32;
        let rho_t = rho_max * (stats.p_flip / flip_thresh).min(1.0);

        // P2 (audit_codex_20260425.md W3): honor config.prior_refresh_temp
        // verbatim, clamping only against zero-division (1e-6 floor). The
        // prior `if temp < 0.01 { 0.5 }` fallback silently snapped Optuna
        // sweeps near zero to the legacy literal 0.5, hiding the actual
        // effect of the parameter at small values.
        let tau = qcfg.prior_refresh_temp.max(1e-6);
        let effective_prior = if n_raw > 0 && rho_t > 1e-4 {
            let log_p0 = prior.max(1e-8).ln();
            let log_pt = (1.0 - rho_t) * log_p0 + rho_t * q_eff / tau;
            log_pt.exp().max(1e-8)
        } else {
            prior
        };

        return QuartzPolicyAdjustment {
            effective_prior,
            penalty,
            bonus: 0.0,
            use_fisher_puct: false,
        };
    }

    // ═══════════════════════════════════════════
    // Adaptive (H6): P_flip-mediated VF↔Q mixture + σ_Q-adaptive penalty
    //
    // P_flip high (uncertain) → Q-refresh corrects prior
    // P_flip low (converged) → VF-refresh concentrates search
    // penalty ν = max(cap, σ_Q) → captures SA's σ_Q amplification
    // ═══════════════════════════════════════════
    if qcfg.penalty_mode == PenaltyMode::PFlipMixture {
        let k = (stats.n_visible as f32).max(2.0);
        let n_total = (stats.root_visits as f32).max(1.0);
        let sigma_q = stats.sigma_q.max(0.001);

        // ── Penalty: ν = max(cap, σ_Q) ──
        let nu = qcfg.hbar_penalty_cap.max(sigma_q);
        let penalty = if n_raw > 0 || o_a > 0 {
            let m_a = 1.0 + n_raw as f32 + o_a as f32;
            -nu / m_a
        } else {
            0.0
        };

        // ── P_flip-mediated mixture ──
        let flip_thresh = 0.159_f32;
        let rho_max = 0.3_f32;
        let p_ratio = (stats.p_flip / flip_thresh).min(2.0); // cap at 2.0

        // Q-refresh: strong when P_flip high (uncertain)
        let rho_q = rho_max * p_ratio.min(1.0);
        // VF-refresh: strong when P_flip low (converged)
        let rho_vf = rho_max * (1.0 - p_ratio).max(0.0);

        // Q8 (audit_codex_20260428.md W'2): optional divergence gate. When
        // `pflip_mixture_divergence_gate` is enabled, the entire mixture
        // contribution is masked off until prior_q_divergence exceeds the
        // per-check epsilon_t threshold — mirroring GatedRefresh. Default
        // false preserves the published PFlipMixture math; flip the flag
        // explicitly to make `prior_q_divergence` an actual sweep axis for
        // this mode instead of the previously-documented no-op.
        let divergence_mask = if qcfg.pflip_mixture_divergence_gate {
            let gate_threshold = stats.epsilon_t.max(1e-6);
            let divergence = stats.prior_q_divergence.max(0.0);
            if divergence > gate_threshold {
                1.0_f32
            } else {
                0.0_f32
            }
        } else {
            1.0_f32
        };
        let rho_q = rho_q * divergence_mask;
        let rho_vf = rho_vf * divergence_mask;

        let effective_prior = if n_raw > 0 && (rho_q + rho_vf) > 1e-4 {
            let log_p0 = prior.max(1e-8).ln();
            // P2 (audit_codex_20260425.md W3): see GatedRefreshLegacy branch
            // above. Same fix — clamp to 1e-6 instead of snapping to 0.5.
            let tau_q = qcfg.prior_refresh_temp.max(1e-6);
            let q_signal = q_eff / tau_q;

            let n_a = n_raw as f32;
            let n_avg = n_total / k;
            let tau_vf = (1.0 + n_avg).ln().max(0.1);
            let vf_signal = (1.0 + n_a).ln() / tau_vf;

            let rho_total = rho_q + rho_vf;
            let log_pt = (1.0 - rho_total) * log_p0 + rho_q * q_signal + rho_vf * vf_signal;
            log_pt.exp().max(1e-8)
        } else {
            prior
        };

        return QuartzPolicyAdjustment {
            effective_prior,
            penalty,
            bonus: 0.0,
            use_fisher_puct: false,
        };
    }

    // ═══════════════════════════════════════════
    // Non-SelfAdaptive: config-driven modes (backward compatible)
    // ═══════════════════════════════════════════

    // Dynamic prior refresh (manual/K-adaptive modes)
    let effective_prior = if qcfg.prior_refresh_rate > 1e-6 && n_raw > 0 {
        let manual_mode = qcfg.prior_refresh_temp > 0.01;
        let rho_effective = if manual_mode {
            qcfg.prior_refresh_rate
        } else {
            let k_ref = 10.0_f32;
            let k = (stats.n_visible as f32).max(2.0);
            qcfg.prior_refresh_rate * (k_ref / k).min(1.0)
        };
        if rho_effective > 1e-4 {
            if manual_mode {
                let tau = qcfg.prior_refresh_temp;
                let log_p0 = prior.max(1e-8).ln();
                let log_pt = (1.0 - rho_effective) * log_p0 + rho_effective * q_eff / tau.max(0.01);
                log_pt.exp().max(1e-8)
            } else {
                let log_p0 = prior.max(1e-8).ln();
                let log_visit = (1.0 + n_raw as f32).ln();
                let tau_visit = 2.0_f32;
                let log_pt = (1.0 - rho_effective) * log_p0 + rho_effective * log_visit / tau_visit;
                log_pt.exp().max(1e-8)
            }
        } else {
            prior
        }
    } else {
        prior
    };

    // Off-diagonal bonus
    let bonus = if qcfg.enable_one_loop {
        eft_action_bonus(stats)
    } else {
        0.0
    };

    // Penalty dispatch
    let penalty = match qcfg.penalty_mode {
        PenaltyMode::Legacy => {
            if qcfg.enable_one_loop {
                one_loop_visit_penalty(n_raw, stats.hbar_eff, 1.0, qcfg.hbar_penalty_cap)
            } else {
                0.0
            }
        }
        PenaltyMode::EffectiveV2 => {
            crate::mcts::quartz::effective_penalty_v2(n_raw, o_a, qcfg.hbar_penalty_cap)
        }
        PenaltyMode::None => 0.0,
        PenaltyMode::SelfAdaptive => unreachable!(), // handled above
        PenaltyMode::GatedRefresh => unreachable!(), // handled above
        PenaltyMode::GatedRefreshLegacy => unreachable!(), // handled above
        PenaltyMode::PFlipMixture => unreachable!(), // handled above
    };

    QuartzPolicyAdjustment {
        effective_prior,
        penalty,
        bonus,
        use_fisher_puct: qcfg.enable_fisher_puct,
    }
}

// ─────────────────────────────────────────────
// § SelectResult
// ─────────────────────────────────────────────

pub struct SelectResult<G: GameState> {
    pub path: SmallVec<[PathEdge<G::Move>; 32]>,
    pub leaf: ArenaRef<MctsNode<G::Move>>,
    pub leaf_state: G,
    pub root_selection_trace: Option<RootSelectionTrace>,
}

pub struct SelectInPlaceResult<G: GameState> {
    pub path: SmallVec<[PathEdge<G::Move>; 32]>,
    pub leaf: ArenaRef<MctsNode<G::Move>>,
    pub root_selection_trace: Option<RootSelectionTrace>,
}

pub struct SelectScratch<G: GameState> {
    state: G,
    undos: Vec<G::Undo>,
}

impl<G: GameState> SelectScratch<G> {
    pub fn new(root_state: &G) -> Self {
        Self {
            state: root_state.clone(),
            undos: Vec::with_capacity(32),
        }
    }

    #[inline]
    pub fn state_mut(&mut self) -> &mut G {
        &mut self.state
    }

    #[inline]
    pub fn reset_to_root(&mut self) {
        while let Some(undo) = self.undos.pop() {
            self.state.undo_move(undo);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RootSelectionTrace {
    pub candidate_count: u32,
    pub effective_prior_l1: f32,
    pub penalty_abs: f32,
    pub refresh_activated: bool,
    /// P01: Active `PenaltyMode` index at this root select. Used for
    /// `controller_penalty_mode_counts` aggregation. Sentinel `u8::MAX`
    /// means "no quartz config / non-Quartz profile" — `record_root` must
    /// skip the histogram increment for sentinel values.
    pub penalty_mode_idx: u8,
    /// P01: True iff the active penalty mode's refresh branch was *eligible*
    /// to fire (i.e. all gating predicates that don't depend on the value
    /// of stats.prior_q_divergence were satisfied). Distinguishes "refresh
    /// could have fired but didn't" from "refresh never had a chance".
    pub refresh_eligible: bool,
}

impl Default for RootSelectionTrace {
    fn default() -> Self {
        // P01: sentinel `penalty_mode_idx = u8::MAX` for non-Quartz traces;
        // record_root's bounds check skips it instead of mis-attributing to
        // PenaltyMode::Legacy (idx 0).
        RootSelectionTrace {
            candidate_count: 0,
            effective_prior_l1: 0.0,
            penalty_abs: 0.0,
            refresh_activated: false,
            penalty_mode_idx: u8::MAX,
            refresh_eligible: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RootScoreDetail {
    score: f32,
    effective_prior_l1: f32,
    penalty_abs: f32,
    refresh_activated: bool,
    /// P01: see `RootSelectionTrace.penalty_mode_idx`. Sentinel u8::MAX
    /// for "no quartz".
    penalty_mode_idx: u8,
    /// P01: see `RootSelectionTrace.refresh_eligible`.
    refresh_eligible: bool,
}

impl Default for RootScoreDetail {
    fn default() -> Self {
        RootScoreDetail {
            score: 0.0,
            effective_prior_l1: 0.0,
            penalty_abs: 0.0,
            refresh_activated: false,
            penalty_mode_idx: u8::MAX,
            refresh_eligible: false,
        }
    }
}

impl RootScoreDetail {
    #[inline]
    fn into_trace(self, candidate_count: usize) -> RootSelectionTrace {
        RootSelectionTrace {
            candidate_count: candidate_count.min(u32::MAX as usize) as u32,
            effective_prior_l1: self.effective_prior_l1,
            penalty_abs: self.penalty_abs,
            refresh_activated: self.refresh_activated,
            penalty_mode_idx: self.penalty_mode_idx,
            refresh_eligible: self.refresh_eligible,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SelectionTelemetrySnapshot {
    pub root_selects: u64,
    pub refresh_selected_count: u64,
    pub selected_penalty_abs_sum: f64,
    pub selected_effective_prior_l1_sum: f64,
    pub selected_mean_candidate_count: f64,
    pub selected_max_candidate_count: u64,
    /// P01: per-`PenaltyMode` invocation count at root selects (indexed
    /// identically to `quartz::PENALTY_MODE_KEYS`).
    pub penalty_mode_invoke_count: [u64; PENALTY_MODE_COUNT],
    /// P01: # of root selects where refresh was *eligible* for the active
    /// penalty mode (all non-divergence gates satisfied).
    pub refresh_eligible_count: u64,
    /// P01: # of root selects where refresh actually fired (refresh_activated).
    /// Distinct from `refresh_selected_count` (which is the same thing today
    /// but kept as a separate semantic: see `record_root`).
    pub refresh_active_count: u64,
}

#[derive(Default)]
pub struct SelectionTelemetry {
    root_selects: AtomicU64,
    refresh_selected_count: AtomicU64,
    selected_penalty_abs_sum_micro: AtomicU64,
    selected_effective_prior_l1_sum_micro: AtomicU64,
    selected_candidate_count_sum: AtomicU64,
    selected_candidate_count_max: AtomicU64,
    /// P01: see snapshot fields. All `Relaxed`; final consistency at
    /// search-end barrier is sufficient (telemetry-only data, not load-bearing).
    penalty_mode_invoke_count: [AtomicU64; PENALTY_MODE_COUNT],
    refresh_eligible_count: AtomicU64,
    refresh_active_count: AtomicU64,
}

impl SelectionTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&self) {
        self.root_selects.store(0, Ordering::Relaxed);
        self.refresh_selected_count.store(0, Ordering::Relaxed);
        self.selected_penalty_abs_sum_micro
            .store(0, Ordering::Relaxed);
        self.selected_effective_prior_l1_sum_micro
            .store(0, Ordering::Relaxed);
        self.selected_candidate_count_sum
            .store(0, Ordering::Relaxed);
        self.selected_candidate_count_max
            .store(0, Ordering::Relaxed);
        // P01: reset the new counters too so per-search aggregation starts clean.
        for slot in self.penalty_mode_invoke_count.iter() {
            slot.store(0, Ordering::Relaxed);
        }
        self.refresh_eligible_count.store(0, Ordering::Relaxed);
        self.refresh_active_count.store(0, Ordering::Relaxed);
    }

    pub fn record_root(&self, trace: RootSelectionTrace) {
        self.root_selects.fetch_add(1, Ordering::Relaxed);
        if trace.refresh_activated {
            self.refresh_selected_count.fetch_add(1, Ordering::Relaxed);
        }
        self.selected_penalty_abs_sum_micro
            .fetch_add(float_to_micro(trace.penalty_abs), Ordering::Relaxed);
        self.selected_effective_prior_l1_sum_micro
            .fetch_add(float_to_micro(trace.effective_prior_l1), Ordering::Relaxed);
        self.selected_candidate_count_sum
            .fetch_add(trace.candidate_count as u64, Ordering::Relaxed);
        self.selected_candidate_count_max
            .fetch_max(trace.candidate_count as u64, Ordering::Relaxed);
        // P01: penalty-mode invocation histogram + refresh eligibility ratio.
        // The eligible/active ordering is "eligible-after-decision":
        // refresh_eligible=true is set only when the gate has been evaluated
        // and the non-divergence preconditions held, so a snapshot taken
        // mid-search will always see eligible >= active.
        if (trace.penalty_mode_idx as usize) < PENALTY_MODE_COUNT {
            self.penalty_mode_invoke_count[trace.penalty_mode_idx as usize]
                .fetch_add(1, Ordering::Relaxed);
        }
        if trace.refresh_eligible {
            self.refresh_eligible_count.fetch_add(1, Ordering::Relaxed);
            if trace.refresh_activated {
                self.refresh_active_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn snapshot(&self) -> SelectionTelemetrySnapshot {
        let root_selects = self.root_selects.load(Ordering::Relaxed);
        let candidate_sum = self.selected_candidate_count_sum.load(Ordering::Relaxed);
        let mut pm = [0u64; PENALTY_MODE_COUNT];
        for (i, slot) in self.penalty_mode_invoke_count.iter().enumerate() {
            pm[i] = slot.load(Ordering::Relaxed);
        }
        SelectionTelemetrySnapshot {
            root_selects,
            refresh_selected_count: self.refresh_selected_count.load(Ordering::Relaxed),
            selected_penalty_abs_sum: micro_to_float(
                self.selected_penalty_abs_sum_micro.load(Ordering::Relaxed),
            ),
            selected_effective_prior_l1_sum: micro_to_float(
                self.selected_effective_prior_l1_sum_micro
                    .load(Ordering::Relaxed),
            ),
            selected_mean_candidate_count: if root_selects > 0 {
                candidate_sum as f64 / root_selects as f64
            } else {
                0.0
            },
            selected_max_candidate_count: self.selected_candidate_count_max.load(Ordering::Relaxed),
            penalty_mode_invoke_count: pm,
            refresh_eligible_count: self.refresh_eligible_count.load(Ordering::Relaxed),
            refresh_active_count: self.refresh_active_count.load(Ordering::Relaxed),
        }
    }
}

#[inline]
fn float_to_micro(v: f32) -> u64 {
    if v.is_finite() {
        (v.max(0.0) as f64 * 1_000_000.0)
            .round()
            .clamp(0.0, u64::MAX as f64) as u64
    } else {
        0
    }
}

#[inline]
fn micro_to_float(v: u64) -> f64 {
    v as f64 / 1_000_000.0
}

#[derive(Clone, Copy, Debug)]
struct EdgeScoreSnapshot {
    n_raw: u32,
    o_a: u32,
    n_eff: u32,
    q_eff: f32,
    terminal_q: Option<f32>,
    prior: f32,
    noise_adj: f32,
    edge_sigma: Option<f32>,
}

#[inline]
fn selected_q_eff(
    snapshot: EdgeScoreSnapshot,
    parent_q: f32,
    parent_visits: u32,
    fpu_reduction: f32,
) -> f32 {
    if let Some(terminal_q) = snapshot.terminal_q {
        terminal_q
    } else if snapshot.n_raw == 0 && snapshot.o_a == 0 && parent_visits >= 4 && fpu_reduction > 1e-6
    {
        parent_q - fpu_reduction * (1.0 - snapshot.prior)
    } else {
        snapshot.q_eff
    }
}

/// P01: Returns true iff the active `PenaltyMode` *can* refresh the prior
/// at all (regardless of whether the runtime gate fires this iteration).
/// `Legacy`/`EffectiveV2`/`None` only refresh through the manual fall-through
/// branch in `quartz_policy_adjustment`, which is gated on
/// `qcfg.prior_refresh_rate > 1e-6`.
#[inline]
fn refresh_supported(qcfg: &QuartzConfig) -> bool {
    use crate::mcts::quartz::PenaltyMode;
    match qcfg.penalty_mode {
        PenaltyMode::SelfAdaptive
        | PenaltyMode::GatedRefresh
        | PenaltyMode::GatedRefreshLegacy
        | PenaltyMode::PFlipMixture => true,
        PenaltyMode::Legacy | PenaltyMode::EffectiveV2 | PenaltyMode::None => {
            qcfg.prior_refresh_rate > 1e-6
        }
    }
}

#[inline]
fn root_quartz_score_detail(
    snapshot: EdgeScoreSnapshot,
    q_eff: f32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
    stats: &QuartzStats,
    qcfg: &QuartzConfig,
) -> RootScoreDetail {
    let adj = quartz_policy_adjustment(
        snapshot.n_raw,
        snapshot.o_a,
        q_eff,
        snapshot.prior,
        sqrt_n_parent_eff,
        stats,
        qcfg,
    );
    let b1 = if stats.lambda_1loop > 1e-6 {
        crate::mcts::quartz::paper_b1loop_bonus(
            snapshot.edge_sigma.unwrap_or(0.0),
            stats.lambda_1loop,
        )
    } else {
        0.0
    };
    let score = adjusted_puct_score(
        snapshot.n_eff,
        q_eff,
        adj.effective_prior,
        snapshot.noise_adj,
        sqrt_n_parent_eff,
        c_puct,
        adj,
    ) + b1;
    let effective_prior_l1 = (adj.effective_prior - snapshot.prior).abs();
    let refresh_activated = effective_prior_l1 > 1e-6;
    // P01: eligibility = "this edge had a chance to refresh under the
    // current penalty mode". Excludes the trivial "n_raw==0 first-visit"
    // case where every refresh branch returns the unchanged prior.
    let refresh_eligible = snapshot.n_raw > 0 && refresh_supported(qcfg);
    RootScoreDetail {
        score,
        effective_prior_l1,
        penalty_abs: adj.penalty.abs(),
        refresh_activated,
        penalty_mode_idx: penalty_mode_idx(qcfg.penalty_mode),
        refresh_eligible,
    }
}

#[inline]
fn snapshot_edge<M>(
    edge: &MctsEdge<M>,
    noise_adj: f32,
    need_sigma: bool,
    exact_terminal_value: bool,
    vvalue_without_vvisit: bool,
) -> EdgeScoreSnapshot {
    let n_raw = edge.n.load(Ordering::Relaxed);
    let o_a = edge.virtual_losses.load(Ordering::Acquire).max(0) as u32;
    let n_eff = n_raw + o_a;
    let q_eff = {
        let has_virtual_value = o_a > 0 || vvalue_without_vvisit;
        if n_eff == 0 && !has_virtual_value {
            return EdgeScoreSnapshot {
                n_raw,
                o_a,
                n_eff,
                q_eff: 0.0,
                terminal_q: if exact_terminal_value {
                    edge.child.terminal_value.map(|v| -v)
                } else {
                    None
                },
                prior: edge.p,
                noise_adj,
                edge_sigma: None,
            };
        }
        let denom = n_eff.max(1) as f32;
        let w = if n_raw > 0 {
            atomic_f64_load(&edge.w) as f32
        } else {
            0.0
        };
        let vv = if has_virtual_value {
            atomic_f64_load(&edge.virtual_value) as f32
        } else {
            0.0
        };
        (w - vv) / denom
    };
    let edge_sigma = if need_sigma && n_raw >= 2 {
        let m2 = f64::from_bits(edge.m2.load(Ordering::Acquire));
        let var = (m2 / (n_raw - 1) as f64).max(0.0);
        Some(var.sqrt() as f32)
    } else {
        None
    };
    let terminal_q = if exact_terminal_value {
        edge.child.terminal_value.map(|v| -v)
    } else {
        None
    };

    EdgeScoreSnapshot {
        n_raw,
        o_a,
        n_eff,
        q_eff,
        terminal_q,
        prior: edge.p,
        noise_adj,
        edge_sigma,
    }
}

#[inline]
fn snapshot_edge_no_vl<M>(
    edge: &MctsEdge<M>,
    noise_adj: f32,
    need_sigma: bool,
    exact_terminal_value: bool,
) -> EdgeScoreSnapshot {
    let n_raw = edge.n.load(Ordering::Relaxed);
    let q_eff = if n_raw == 0 {
        0.0
    } else {
        atomic_f64_load(&edge.w) as f32 / n_raw as f32
    };
    let edge_sigma = if need_sigma && n_raw >= 2 {
        let m2 = f64::from_bits(edge.m2.load(Ordering::Acquire));
        let var = (m2 / (n_raw - 1) as f64).max(0.0);
        Some(var.sqrt() as f32)
    } else {
        None
    };
    let terminal_q = if exact_terminal_value {
        edge.child.terminal_value.map(|v| -v)
    } else {
        None
    };

    EdgeScoreSnapshot {
        n_raw,
        o_a: 0,
        n_eff: n_raw,
        q_eff,
        terminal_q,
        prior: edge.p,
        noise_adj,
        edge_sigma,
    }
}

#[inline]
fn score_snapshot(
    snapshot: EdgeScoreSnapshot,
    depth: usize,
    parent_q: f32,
    parent_visits: u32,
    sqrt_n_parent_eff: f32,
    c_puct: f32,
    fpu_reduction: f32,
    quartz: Option<(&QuartzStats, &QuartzConfig)>,
) -> f32 {
    let q_eff = selected_q_eff(snapshot, parent_q, parent_visits, fpu_reduction);

    match quartz {
        Some((stats, qcfg)) if depth == 0 => {
            // Full QUARTZ scoring at root
            root_quartz_score_detail(snapshot, q_eff, sqrt_n_parent_eff, c_puct, stats, qcfg).score
        }
        Some((stats, qcfg)) if !qcfg.root_only_shaping && depth <= 3 => {
            // Historical shallow-tree blend preserved only for controller ablations.
            let depth_weight = 1.0 / (1.0 + depth as f32);
            let base_puct = puct_score_with_parent_sqrt(
                snapshot.n_eff,
                q_eff,
                snapshot.prior,
                snapshot.noise_adj,
                sqrt_n_parent_eff,
                c_puct,
            );
            let full_quartz = ablation_puct_score_with_parent_sqrt(
                snapshot.n_eff,
                snapshot.n_raw,
                snapshot.o_a,
                q_eff,
                snapshot.prior,
                snapshot.noise_adj,
                sqrt_n_parent_eff,
                c_puct,
                stats,
                qcfg,
            );
            let blended = depth_weight * full_quartz + (1.0 - depth_weight) * base_puct;
            let b1 = if stats.lambda_1loop > 1e-6 {
                crate::mcts::quartz::paper_b1loop_bonus(
                    snapshot.edge_sigma.unwrap_or(0.0),
                    stats.lambda_1loop,
                ) * depth_weight
            } else {
                0.0
            };
            blended + b1
        }
        _ => puct_score_with_parent_sqrt(
            snapshot.n_eff,
            q_eff,
            snapshot.prior,
            snapshot.noise_adj,
            sqrt_n_parent_eff,
            c_puct,
        ),
    }
}

#[inline]
fn should_replace_best(
    best_score: f32,
    best_idx: usize,
    candidate_score: f32,
    candidate_idx: usize,
) -> bool {
    best_score
        .partial_cmp(&candidate_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(candidate_idx.cmp(&best_idx))
        == std::cmp::Ordering::Less
}

struct SelectCoreResult<M: Copy + Send + Sync + 'static> {
    path: SmallVec<[PathEdge<M>; 32]>,
    leaf: ArenaRef<MctsNode<M>>,
    root_selection_trace: Option<RootSelectionTrace>,
}

// ─────────────────────────────────────────────
// § select
// ─────────────────────────────────────────────

#[inline(always)]
fn select_core<G, F>(
    root: &ArenaRef<MctsNode<G::Move>>,
    cur_state: &mut G,
    c_puct: f32,
    root_prior_noise: Option<&[f32]>,
    pw: Option<&PwConfig>,
    max_depth: usize,
    tt: &TranspositionTable<G::Move>,
    // QUARTZ EFT-PUCT (None = standard PUCT)
    quartz: Option<(&QuartzStats, &QuartzConfig)>,
    exact_terminal_value: bool,
    fpu_reduction: f32,
    reserve_virtual_loss: bool,
    // Parallelism controller (1st-class runtime object)
    par_ctrl: &super::parallel::ParallelismController,
    mut apply_selected_move: F,
) -> SelectCoreResult<G::Move>
where
    G: GameState,
    F: FnMut(&mut G, G::Move),
{
    // Keep the common root-to-leaf path on stack. Most profiled selections are
    // under 20 ply; paths beyond 32 still spill to heap without changing API
    // semantics.
    let mut path: SmallVec<[PathEdge<G::Move>; 32]> = SmallVec::new();
    let mut cur_node = *root;
    let mut depth = 0usize;
    let mut root_selection_trace_record = None;
    let vvalue_without_vvisit =
        reserve_virtual_loss && par_ctrl.can_publish_vvalue_without_vvisit();

    loop {
        if cur_node.terminal_value.is_some()
            || !cur_node.is_expanded()
            || (max_depth > 0 && depth >= max_depth)
        {
            break;
        }

        let n_candidates = cur_node.candidate_count();
        if n_candidates == 0 {
            break;
        }

        let n_total = cur_node.n_total.load(Ordering::Acquire);
        let n_visible = match pw {
            Some(cfg) => {
                let base_k = cfg.k(n_total).max(1).min(n_candidates);
                // Depth-aware PW: reduce materialized edges at deeper nodes
                // Root (depth=0) gets full k; deeper nodes get progressively fewer
                if depth > 0 {
                    let depth_factor = 1.0 / (1.0 + 0.3 * depth as f32);
                    (base_k as f32 * depth_factor).ceil() as usize
                } else {
                    base_k
                }
            }
            None => n_candidates,
        };

        let n_mat = cur_node.materialized_count();
        if n_mat < n_visible {
            if reserve_virtual_loss {
                materialize_edges_in_place_best_effort(&cur_node, cur_state, n_visible, tt);
            } else {
                materialize_edges_in_place(&cur_node, cur_state, n_visible, tt);
            }
        }

        // Phase 7 C (2026-04-26): lock-free slab read. `read_edges()`
        // returns `&[MctsEdge<M>]` covering [0..edge_cursor]. Slice
        // lifetime is tied to `&cur_node` (slab lives in TT bucket
        // Bump, outlives this borrow).
        let edges_full = cur_node.read_edges();
        let n_edges = n_visible.min(edges_full.len());
        if n_edges == 0 {
            break;
        }
        let edges = &edges_full[..n_edges];

        let vl_sum: u32 = if reserve_virtual_loss {
            edges
                .iter()
                .map(|e| e.virtual_losses.load(Ordering::Acquire).max(0) as u32)
                .sum()
        } else {
            0
        };
        let n_parent_eff = n_total + vl_sum;

        let parent_visits = n_total;
        let parent_q = if parent_visits >= 4 && fpu_reduction > 1e-6 {
            cur_node.mean_q()
        } else {
            0.0
        };
        let sqrt_n_parent_eff = (n_parent_eff as f32).sqrt();
        let need_sigma = matches!(quartz, Some((stats, qcfg)) if (depth == 0 || (!qcfg.root_only_shaping && depth <= 3)) && stats.lambda_1loop > 1e-6);
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_has_pending = false;
        let mut best_root_detail = None;

        for (idx, edge) in edges.iter().enumerate() {
            let noise_adj = if depth == 0 {
                root_prior_noise
                    .and_then(|noise| noise.get(idx).copied())
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let snapshot = if reserve_virtual_loss {
                snapshot_edge(
                    edge,
                    noise_adj,
                    need_sigma,
                    exact_terminal_value,
                    vvalue_without_vvisit,
                )
            } else {
                snapshot_edge_no_vl(edge, noise_adj, need_sigma, exact_terminal_value)
            };
            let (score, root_detail) = if depth == 0 {
                if let Some((stats, qcfg)) = quartz {
                    let q_eff = selected_q_eff(snapshot, parent_q, parent_visits, fpu_reduction);
                    let detail = root_quartz_score_detail(
                        snapshot,
                        q_eff,
                        sqrt_n_parent_eff,
                        c_puct,
                        stats,
                        qcfg,
                    );
                    (detail.score, Some(detail))
                } else {
                    (
                        score_snapshot(
                            snapshot,
                            depth,
                            parent_q,
                            parent_visits,
                            sqrt_n_parent_eff,
                            c_puct,
                            fpu_reduction,
                            quartz,
                        ),
                        None,
                    )
                }
            } else {
                (
                    score_snapshot(
                        snapshot,
                        depth,
                        parent_q,
                        parent_visits,
                        sqrt_n_parent_eff,
                        c_puct,
                        fpu_reduction,
                        quartz,
                    ),
                    None,
                )
            };
            if idx == 0 || should_replace_best(best_score, best_idx, score, idx) {
                best_idx = idx;
                best_score = score;
                best_has_pending = snapshot.o_a > 0;
                best_root_detail = root_detail;
            }
        }

        if depth == 0 && best_root_detail.is_some() {
            root_selection_trace_record = best_root_detail.map(|detail| detail.into_trace(n_edges));
        }

        // Parallelism telemetry: record pending count at this node
        if reserve_virtual_loss {
            par_ctrl.telemetry.record_select(vl_sum);
        }

        // Detect duplicate path: if selected edge already has pending VL
        if reserve_virtual_loss && best_has_pending {
            par_ctrl.telemetry.record_dup_leaf();
        }

        let vl_split = if reserve_virtual_loss {
            let vl_split = par_ctrl.vl_at_depth(depth as u32);
            edges[best_idx].apply_vl(vl_split.vvisit, vl_split.vvalue);
            if vl_split.vvalue.abs() > 1e-9 {
                par_ctrl.telemetry.record_vvalue(vl_split.vvalue);
            }
            vl_split
        } else {
            super::parallel::VlSplit::ZERO
        };

        let best_mv = edges[best_idx].mv;
        let next_node = edges[best_idx].child;
        // Phase 7 C: no guard to drop — slab read is lock-free.

        // Hint the prefetcher to fetch the next node's body. The next loop
        // iteration's first reads (`terminal_value`, `is_expanded()`,
        // `candidate_count()`, `n_total.load()`) are all on this body; the
        // intervening apply_move_in_place + path.push (~50 cycles) hides
        // the L2/L3 fetch latency. Random-walk descent through the
        // bumpalo arena makes successive nodes mostly cold lines.
        // No semantic effect — `_mm_prefetch` is a non-faulting hint.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch::<_MM_HINT_T0>(ArenaRef::as_ptr(&next_node) as *const i8);
        }

        apply_selected_move(cur_state, best_mv);
        path.push(PathEdge {
            parent: cur_node,
            edge_idx: best_idx,
            applied_vl: (vl_split.vvisit, vl_split.vvalue),
        });
        cur_node = next_node;
        depth += 1;
    }

    SelectCoreResult {
        path,
        leaf: cur_node,
        root_selection_trace: root_selection_trace_record,
    }
}

#[inline(always)]
pub fn select<G: GameState>(
    root: &ArenaRef<MctsNode<G::Move>>,
    root_state: &G,
    c_puct: f32,
    root_prior_noise: Option<&[f32]>,
    pw: Option<&PwConfig>,
    max_depth: usize,
    tt: &TranspositionTable<G::Move>,
    // QUARTZ EFT-PUCT (None = standard PUCT)
    quartz: Option<(&QuartzStats, &QuartzConfig)>,
    exact_terminal_value: bool,
    fpu_reduction: f32,
    reserve_virtual_loss: bool,
    // Parallelism controller (1st-class runtime object)
    par_ctrl: &super::parallel::ParallelismController,
) -> SelectResult<G> {
    let mut leaf_state = root_state.clone();
    let result = select_core(
        root,
        &mut leaf_state,
        c_puct,
        root_prior_noise,
        pw,
        max_depth,
        tt,
        quartz,
        exact_terminal_value,
        fpu_reduction,
        reserve_virtual_loss,
        par_ctrl,
        |state, mv| state.apply_move_in_place_no_undo(mv),
    );

    SelectResult {
        path: result.path,
        leaf: result.leaf,
        leaf_state,
        root_selection_trace: result.root_selection_trace,
    }
}

#[inline(always)]
pub fn select_in_place<G: GameState>(
    root: &ArenaRef<MctsNode<G::Move>>,
    scratch: &mut SelectScratch<G>,
    c_puct: f32,
    root_prior_noise: Option<&[f32]>,
    pw: Option<&PwConfig>,
    max_depth: usize,
    tt: &TranspositionTable<G::Move>,
    quartz: Option<(&QuartzStats, &QuartzConfig)>,
    exact_terminal_value: bool,
    fpu_reduction: f32,
    reserve_virtual_loss: bool,
    par_ctrl: &super::parallel::ParallelismController,
) -> SelectInPlaceResult<G> {
    scratch.reset_to_root();
    let SelectScratch { state, undos } = scratch;
    let result = select_core(
        root,
        state,
        c_puct,
        root_prior_noise,
        pw,
        max_depth,
        tt,
        quartz,
        exact_terminal_value,
        fpu_reduction,
        reserve_virtual_loss,
        par_ctrl,
        |state, mv| {
            let undo = state.apply_move_in_place(mv);
            undos.push(undo);
        },
    );

    SelectInPlaceResult {
        path: result.path,
        leaf: result.leaf,
        root_selection_trace: result.root_selection_trace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::quartz::{PenaltyMode, QuartzConfig, QuartzStats};
    use std::hint::black_box;
    use std::time::Instant;

    fn linear_best_index(
        snapshots: &[EdgeScoreSnapshot],
        depth: usize,
        parent_q: f32,
        parent_visits: u32,
        sqrt_n_parent_eff: f32,
        c_puct: f32,
        fpu_reduction: f32,
        quartz: Option<(&QuartzStats, &QuartzConfig)>,
    ) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (idx, snapshot) in snapshots.iter().copied().enumerate() {
            let score = score_snapshot(
                snapshot,
                depth,
                parent_q,
                parent_visits,
                sqrt_n_parent_eff,
                c_puct,
                fpu_reduction,
                quartz,
            );
            if idx == 0 || should_replace_best(best_score, best_idx, score, idx) {
                best_idx = idx;
                best_score = score;
            }
        }
        best_idx
    }

    fn reference_best_index(
        snapshots: &[EdgeScoreSnapshot],
        depth: usize,
        parent_q: f32,
        parent_visits: u32,
        sqrt_n_parent_eff: f32,
        c_puct: f32,
        fpu_reduction: f32,
        quartz: Option<(&QuartzStats, &QuartzConfig)>,
    ) -> usize {
        snapshots
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                let sa = score_snapshot(
                    **a,
                    depth,
                    parent_q,
                    parent_visits,
                    sqrt_n_parent_eff,
                    c_puct,
                    fpu_reduction,
                    quartz,
                );
                let sb = score_snapshot(
                    **b,
                    depth,
                    parent_q,
                    parent_visits,
                    sqrt_n_parent_eff,
                    c_puct,
                    fpu_reduction,
                    quartz,
                );
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(j.cmp(i))
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn test_linear_scan_matches_reference_puct() {
        let snapshots = vec![
            EdgeScoreSnapshot {
                n_raw: 8,
                o_a: 0,
                n_eff: 8,
                q_eff: 0.42,
                terminal_q: None,
                prior: 0.30,
                noise_adj: 0.00,
                edge_sigma: None,
            },
            EdgeScoreSnapshot {
                n_raw: 2,
                o_a: 1,
                n_eff: 3,
                q_eff: 0.10,
                terminal_q: None,
                prior: 0.45,
                noise_adj: 0.05,
                edge_sigma: None,
            },
            EdgeScoreSnapshot {
                n_raw: 0,
                o_a: 0,
                n_eff: 0,
                q_eff: 0.00,
                terminal_q: None,
                prior: 0.25,
                noise_adj: 0.00,
                edge_sigma: None,
            },
        ];
        let sqrt_n_parent_eff = 4.0;
        let c_puct = 2.0;
        let parent_q = 0.35;

        let expected = reference_best_index(
            &snapshots,
            1,
            parent_q,
            8,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            None,
        );
        let best_idx = linear_best_index(
            &snapshots,
            1,
            parent_q,
            8,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            None,
        );
        assert_eq!(best_idx, expected);
    }

    #[test]
    fn test_linear_scan_matches_reference_quartz_root() {
        let snapshots = vec![
            EdgeScoreSnapshot {
                n_raw: 10,
                o_a: 1,
                n_eff: 11,
                q_eff: 0.30,
                terminal_q: None,
                prior: 0.34,
                noise_adj: 0.02,
                edge_sigma: Some(0.08),
            },
            EdgeScoreSnapshot {
                n_raw: 5,
                o_a: 0,
                n_eff: 5,
                q_eff: 0.28,
                terminal_q: None,
                prior: 0.33,
                noise_adj: 0.01,
                edge_sigma: Some(0.11),
            },
            EdgeScoreSnapshot {
                n_raw: 1,
                o_a: 0,
                n_eff: 1,
                q_eff: 0.15,
                terminal_q: None,
                prior: 0.33,
                noise_adj: 0.00,
                edge_sigma: None,
            },
        ];
        let sqrt_n_parent_eff = 6.0;
        let c_puct = 2.25;
        let stats = QuartzStats {
            lambda_1loop: 0.09,
            sigma_q: 0.12,
            p_flip: 0.04,
            root_visits: 36,
            n_visible: 3,
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            ..Default::default()
        };

        let expected = reference_best_index(
            &snapshots,
            0,
            0.22,
            12,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            Some((&stats, &qcfg)),
        );
        let best_idx = linear_best_index(
            &snapshots,
            0,
            0.22,
            12,
            sqrt_n_parent_eff,
            c_puct,
            0.0,
            Some((&stats, &qcfg)),
        );
        assert_eq!(best_idx, expected);
    }

    #[test]
    fn test_snapshot_edge_skips_vvalue_without_reservation_except_vvalue_only() {
        let child = crate::mcts::tt::leak_node::<u8>(7, None);
        let edge = MctsEdge::new(3u8, child, 0.5);
        edge.n.store(4, Ordering::Relaxed);
        edge.w.store(2.0_f64.to_bits(), Ordering::Relaxed);
        edge.virtual_value
            .store(1.0_f64.to_bits(), Ordering::Relaxed);

        let paired_mode = snapshot_edge(&edge, 0.0, false, false, false);
        assert_eq!(paired_mode.o_a, 0);
        assert_eq!(paired_mode.n_eff, 4);
        assert!((paired_mode.q_eff - 0.5).abs() < 1e-6);

        edge.n.store(0, Ordering::Relaxed);
        edge.w.store(0.0_f64.to_bits(), Ordering::Relaxed);
        let unvisited_paired_mode = snapshot_edge(&edge, 0.0, false, false, false);
        assert_eq!(unvisited_paired_mode.o_a, 0);
        assert_eq!(unvisited_paired_mode.n_eff, 0);
        assert_eq!(unvisited_paired_mode.q_eff, 0.0);

        edge.n.store(4, Ordering::Relaxed);
        edge.w.store(2.0_f64.to_bits(), Ordering::Relaxed);
        let vvalue_only_mode = snapshot_edge(&edge, 0.0, false, false, true);
        assert_eq!(vvalue_only_mode.o_a, 0);
        assert_eq!(vvalue_only_mode.n_eff, 4);
        assert!((vvalue_only_mode.q_eff - 0.25).abs() < 1e-6);

        edge.virtual_losses.store(1, Ordering::Relaxed);
        let live_reservation = snapshot_edge(&edge, 0.0, false, false, false);
        assert_eq!(live_reservation.o_a, 1);
        assert_eq!(live_reservation.n_eff, 5);
        assert!((live_reservation.q_eff - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_quartz_scoring_is_root_only() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 6,
            o_a: 0,
            n_eff: 6,
            q_eff: 0.22,
            terminal_q: None,
            prior: 0.45,
            noise_adj: 0.0,
            edge_sigma: Some(0.08),
        };
        let stats = QuartzStats {
            lambda_1loop: 0.11,
            sigma_q: 0.18,
            hbar_eff: 0.6,
            p_flip: 0.03,
            prior_q_divergence: 0.8,
            epsilon_t: 0.1,
            root_visits: 24,
            n_visible: 4,
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            ..Default::default()
        };

        let quartz_nonroot =
            score_snapshot(snapshot, 1, 0.20, 24, 5.0, 2.0, 0.0, Some((&stats, &qcfg)));
        let plain_nonroot = puct_score_with_parent_sqrt(
            snapshot.n_eff,
            snapshot.q_eff,
            snapshot.prior,
            snapshot.noise_adj,
            5.0,
            2.0,
        );
        assert!((quartz_nonroot - plain_nonroot).abs() < 1e-6);
    }

    #[test]
    fn test_gated_refresh_ignores_pflip_when_other_state_matches() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.10,
            terminal_q: None,
            prior: 0.70,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            hbar_penalty_cap: 0.3,
            ..Default::default()
        };
        let stats_a = QuartzStats {
            hbar_eff: 0.8,
            p_flip: 0.01,
            prior_q_divergence: 0.6,
            epsilon_t: 0.1,
            root_visits: 20,
            n_visible: 5,
            ..Default::default()
        };
        let stats_b = QuartzStats {
            p_flip: 0.95,
            ..stats_a.clone()
        };

        let score_a = score_snapshot(snapshot, 0, 0.0, 20, 5.0, 2.0, 0.0, Some((&stats_a, &qcfg)));
        let score_b = score_snapshot(snapshot, 0, 0.0, 20, 5.0, 2.0, 0.0, Some((&stats_b, &qcfg)));
        assert!((score_a - score_b).abs() < 1e-6);
    }

    #[test]
    fn test_gated_refresh_opens_on_prior_divergence() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.10,
            terminal_q: None,
            prior: 0.70,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            hbar_penalty_cap: 0.3,
            ..Default::default()
        };
        let gate_off = QuartzStats {
            hbar_eff: 0.8,
            prior_q_divergence: 0.05,
            epsilon_t: 0.10,
            root_visits: 20,
            n_visible: 5,
            ..Default::default()
        };
        let gate_on = QuartzStats {
            prior_q_divergence: 0.60,
            ..gate_off.clone()
        };

        let score_off = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&gate_off, &qcfg)),
        );
        let score_on = score_snapshot(snapshot, 0, 0.0, 20, 5.0, 2.0, 0.0, Some((&gate_on, &qcfg)));
        assert!(
            score_on < score_off,
            "visit-share refresh should pull a 0.70 prior toward the 0.20 visit share"
        );
    }

    #[test]
    fn test_q8_pflip_mixture_divergence_gate_default_off_is_noop() {
        // Q8 (audit_codex_20260428.md W'2): with the gate flag at its
        // default (false), divergence value must NOT change the score —
        // existing PFlipMixture math is preserved.
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.10,
            terminal_q: None,
            prior: 0.20,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::PFlipMixture,
            hbar_penalty_cap: 0.3,
            prior_refresh_temp: 0.5,
            pflip_mixture_divergence_gate: false,
            ..Default::default()
        };
        let stats_low_div = QuartzStats {
            p_flip: 0.10,
            prior_q_divergence: 0.0,
            epsilon_t: 0.10,
            root_visits: 20,
            n_visible: 5,
            sigma_q: 0.10,
            ..Default::default()
        };
        let stats_high_div = QuartzStats {
            prior_q_divergence: 0.80,
            ..stats_low_div.clone()
        };
        let s_low = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_low_div, &qcfg)),
        );
        let s_high = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_high_div, &qcfg)),
        );
        assert!(
            (s_low - s_high).abs() < 1e-5,
            "PFlipMixture must be divergence-insensitive when gate flag is off (s_low={s_low}, s_high={s_high})"
        );
    }

    #[test]
    fn test_q8_pflip_mixture_divergence_gate_on_masks_below_threshold() {
        // Q8: with the gate flag enabled, divergence below epsilon_t must
        // mask off the refresh-mixture contribution; setting divergence
        // above the threshold must re-enable it. The two scores must
        // therefore differ.
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.40,
            terminal_q: None,
            prior: 0.20,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::PFlipMixture,
            hbar_penalty_cap: 0.3,
            prior_refresh_temp: 0.5,
            pflip_mixture_divergence_gate: true,
            ..Default::default()
        };
        let stats_below = QuartzStats {
            p_flip: 0.10,
            prior_q_divergence: 0.05, // below epsilon_t
            epsilon_t: 0.10,
            root_visits: 20,
            n_visible: 5,
            sigma_q: 0.10,
            ..Default::default()
        };
        let stats_above = QuartzStats {
            prior_q_divergence: 0.80, // above epsilon_t
            ..stats_below.clone()
        };
        let s_below = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_below, &qcfg)),
        );
        let s_above = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_above, &qcfg)),
        );
        assert!(
            (s_below - s_above).abs() > 1e-5,
            "PFlipMixture with divergence gate must respond to divergence above epsilon_t (s_below={s_below}, s_above={s_above})"
        );
    }

    #[test]
    fn test_legacy_gated_refresh_depends_on_pflip() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 4,
            o_a: 0,
            n_eff: 4,
            q_eff: 0.10,
            terminal_q: None,
            prior: 0.20,
            noise_adj: 0.0,
            edge_sigma: None,
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefreshLegacy,
            hbar_penalty_cap: 0.3,
            ..Default::default()
        };
        let stats_low = QuartzStats {
            p_flip: 0.01,
            prior_q_divergence: 0.9,
            epsilon_t: 0.1,
            root_visits: 20,
            n_visible: 5,
            ..Default::default()
        };
        let stats_high = QuartzStats {
            p_flip: 0.15,
            ..stats_low.clone()
        };

        let score_low = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_low, &qcfg)),
        );
        let score_high = score_snapshot(
            snapshot,
            0,
            0.0,
            20,
            5.0,
            2.0,
            0.0,
            Some((&stats_high, &qcfg)),
        );
        assert!(score_high > score_low);
    }

    #[test]
    fn test_legacy_profile_can_shape_shallow_nonroot_nodes() {
        let snapshot = EdgeScoreSnapshot {
            n_raw: 6,
            o_a: 0,
            n_eff: 6,
            q_eff: 0.22,
            terminal_q: None,
            prior: 0.45,
            noise_adj: 0.0,
            edge_sigma: Some(0.08),
        };
        let stats = QuartzStats {
            lambda_1loop: 0.11,
            sigma_q: 0.18,
            hbar_eff: 0.6,
            p_flip: 0.10,
            prior_q_divergence: 0.8,
            epsilon_t: 0.1,
            root_visits: 24,
            n_visible: 4,
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefreshLegacy,
            root_only_shaping: false,
            ..Default::default()
        };

        let shallow_score =
            score_snapshot(snapshot, 1, 0.20, 24, 5.0, 2.0, 0.0, Some((&stats, &qcfg)));
        let plain_nonroot = puct_score_with_parent_sqrt(
            snapshot.n_eff,
            snapshot.q_eff,
            snapshot.prior,
            snapshot.noise_adj,
            5.0,
            2.0,
        );
        assert!((shallow_score - plain_nonroot).abs() > 1e-5);
    }

    #[test]
    fn test_fpu_improves_unvisited_child_score_without_affecting_visited_edge() {
        let snapshots = vec![
            EdgeScoreSnapshot {
                n_raw: 4,
                o_a: 0,
                n_eff: 4,
                q_eff: 0.05,
                terminal_q: None,
                prior: 0.30,
                noise_adj: 0.0,
                edge_sigma: None,
            },
            EdgeScoreSnapshot {
                n_raw: 0,
                o_a: 0,
                n_eff: 0,
                q_eff: 0.0,
                terminal_q: None,
                prior: 0.55,
                noise_adj: 0.0,
                edge_sigma: None,
            },
        ];

        let no_fpu = linear_best_index(&snapshots, 1, 0.40, 8, 4.0, 2.0, 0.0, None);
        let with_fpu = linear_best_index(&snapshots, 1, 0.40, 8, 4.0, 2.0, 0.20, None);
        assert_eq!(no_fpu, 1);
        assert_eq!(with_fpu, 1);

        let unvisited_no_fpu = score_snapshot(snapshots[1], 1, 0.40, 8, 4.0, 2.0, 0.0, None);
        let unvisited_with_fpu = score_snapshot(snapshots[1], 1, 0.40, 8, 4.0, 2.0, 0.20, None);
        let visited_no_fpu = score_snapshot(snapshots[0], 1, 0.40, 8, 4.0, 2.0, 0.0, None);
        let visited_with_fpu = score_snapshot(snapshots[0], 1, 0.40, 8, 4.0, 2.0, 0.20, None);
        assert!(unvisited_with_fpu > unvisited_no_fpu);
        assert!((visited_with_fpu - visited_no_fpu).abs() < 1e-6);
    }

    #[test]
    fn test_should_replace_best_keeps_lower_index_on_tie_and_nan() {
        assert!(!should_replace_best(1.0, 0, 1.0, 1));
        assert!(!should_replace_best(f32::NAN, 0, 2.0, 1));
        assert!(should_replace_best(0.5, 1, 0.6, 2));
    }

    #[test]
    #[ignore]
    fn bench_linear_scan_vs_reference() {
        let snapshots: Vec<_> = (0..128)
            .map(|idx| EdgeScoreSnapshot {
                n_raw: (idx * 7 % 19) as u32,
                o_a: (idx % 3) as u32,
                n_eff: (idx * 7 % 19) as u32 + (idx % 3) as u32,
                q_eff: ((idx as f32 * 0.137).sin() * 0.5).clamp(-1.0, 1.0),
                terminal_q: None,
                prior: 1.0 / 128.0 + (idx as f32 * 0.017).cos().abs() * 0.01,
                noise_adj: if idx < 32 {
                    (idx as f32 * 0.031).sin().abs() * 0.02
                } else {
                    0.0
                },
                edge_sigma: Some(0.05 + idx as f32 * 0.001),
            })
            .collect();
        let stats = QuartzStats {
            lambda_1loop: 0.09,
            sigma_q: 0.18,
            p_flip: 0.07,
            root_visits: 512,
            n_visible: snapshots.len(),
            ..Default::default()
        };
        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            ..Default::default()
        };
        let sqrt_n_parent_eff = (640.0f32).sqrt();
        let c_puct = 2.25;
        let reps = 200_000usize;

        let t_ref = Instant::now();
        let mut acc_ref = 0usize;
        for _ in 0..reps {
            acc_ref ^= black_box(reference_best_index(
                black_box(&snapshots),
                0,
                0.18,
                32,
                sqrt_n_parent_eff,
                c_puct,
                0.20,
                Some((&stats, &qcfg)),
            ));
        }
        let ref_ns = t_ref.elapsed().as_nanos();

        let t_new = Instant::now();
        let mut acc_new = 0usize;
        for _ in 0..reps {
            acc_new ^= black_box(linear_best_index(
                black_box(&snapshots),
                0,
                0.18,
                32,
                sqrt_n_parent_eff,
                c_puct,
                0.20,
                Some((&stats, &qcfg)),
            ));
        }
        let new_ns = t_new.elapsed().as_nanos();

        assert_eq!(acc_new, acc_ref);
        eprintln!(
            "[BENCH] select reference={}ns linear={}ns speedup={:.2}x",
            ref_ns,
            new_ns,
            ref_ns as f64 / new_ns.max(1) as f64
        );
    }
}

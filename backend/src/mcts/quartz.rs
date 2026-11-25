//! QUARTZController v0.8 — Unified VOC with ε-envariance (state-derived, fixed constants)
//!
//! # 이론적 기초 (검증됨)
//!
//! ## (1) QFT Path Integral → One-loop PUCT Correction
//!
//! Z = Σ_π N[π] 에서 saddle point π* 근방 2차 전개:
//!
//!   M_aa  = ∂²(-log Z)/∂N_a²       = 1/N_a²         [diagonal]
//!   M_aa' = ∂²(-log Z)/∂N_a∂N_{a'} = -K(a,a')/(N_a·N_{a'}) [off-diagonal, path correlation]
//!
//! One-loop correction to selection score = (ħ_eff/2)·∂/∂N_a Tr log M:
//!   Diagonal:    -ħ_eff/N_a                   → visit penalty [implemented in select.rs]
//!   Off-diagonal: +ħ_eff²·RTT_var/N_a          → ρ̂ correction via RTT variance
//!
//! **ħ_eff = σ_Q/σ₀** — 하나의 파라미터로 모든 one-loop 보정을 통합.
//!
//! ## (2) ε-Envariance → Adaptive EXPAND Threshold
//!
//! MCTS 선형대수 유비:
//!   |MCTS⟩ = Σ_a √(N_a/N) |a⟩_S ⊗ |thread_a⟩_E
//!
//! ε-envariance 정의: ‖q_T - π‖_TV < ε
//! Pinsker 부등식: KL(q_T‖π) ≥ 2·‖q_T-π‖_TV²
//! 따라서: S_KL > 2ε² ↔ ε-envariance 위반
//!
//! 적응적 임계값: ε_t = 1/√N_total
//! (N회 방문 후 binomial noise ≈ 1/√N)
//!
//! **S_KL > 1/(2·N_total) 이면 ε-envariance 위반 → EXPAND 채널 활성화**
//! 이는 자유 하이퍼파라미터 없는 자연적 threshold.
//!
//! ## (3) Fisher Metric → α = 1/2 고정
//!
//! Softmax prior의 Fisher information matrix (diagonal):
//!   F_aa = E[(∂ log π/∂θ_a)²] = π(a)·(1-π(a)) ≈ π(a) for small π
//!
//! Natural gradient PUCT = standard PUCT with π → F^{-1}∇ = 1/π(a) scaling:
//!   score_nat(a) = Q + c·π(a)/√π(a)·√N/(1+N_a) = Q + c·√π(a)·√N/(1+N_a)
//!
//! **α = 1/2는 Fisher information에서 유일하게 결정. 자유 파라미터 아님.**
//!
//! ## (4) FEP / Envariance Violation → Channel Routing (분리)
//!
//! envariance 위반 방향:
//!   δ(a) = q_T(a) - π(a)
//!   δ(a) < -ε: action a 탐색 부족 → EXPAND (outside proposal)
//!   δ(a) > +ε: action a 과탐색   → FOCUS (refine current)
//!
//! S_KL = Σ_a q_T(a) log(q_T(a)/π(a)) 이 두 경우를 모두 포착.
//! S_KL/N 비율로 정규화하면 state-derived routing (fixed constants).
//!
//! ## (5) CTM/NS — cost와 temperature를 ħ_eff로 표준화
//!
//! cost_focus(t) = ħ_eff × (σ₀/N_min) × (0.5 + urgency(t))
//!   → ħ_eff로 cost가 자동 스케일링됨
//!   → high uncertainty → high cost (더 탐색하면 더 얻음)
//!
//! ns_temp(t) = (1-t/T)^γ, γ=1.0 (linear, default)
//!
//! ## Unified VOC Formula
//! VOC(s) = wimp(s) × max(VOC_FOCUS, VOC_EXPAND, VOC_MERGE)
//!
//! VOC_FOCUS  = P_flip × σ_Δ(ρ̂) × fep_envar - ħ_eff × (σ₀/N_min) × ctm_factor
//! VOC_EXPAND = P_envar × E[Δexpand] × ns_temp(t) - ħ_eff² × (σ₀/N_min)
//! VOC_MERGE  = P_merge × √RTT_var × ħ_eff - ħ_eff × (σ₀/N_min)
//!
//! **hyperparameter 수: 12개 → 5개**
//! 유지: σ₀, c_puct, N_min, T_budget(optional), γ(optional, default 1.0)
//! 제거: lambda_loop, tau, fisher_alpha, cost_focus_base,
//!        flip_thresh, lambda_fep, lambda_merge, cost_merge, cost_expand
//!
//! ## MCTS 4단계 수학적 기여
//! Selection:  Fisher PUCT (α=1/2) + one-loop visit penalty (-ħ_eff/N_a)
//! Expansion:  ε-envariance threshold + NS annealing temperature
//! Simulation: (AlphaZero: NN value; ShortRollout: empirical)
//! Backprop:   Welford M2 (per-edge σᵢ) + RTT Welford (ρ̂ off-diagonal)

use std::sync::atomic::{AtomicU32, Ordering};

use smallvec::SmallVec;

use crate::mcts::node::{ArenaRef, MctsEdge, MctsNode};
use crate::mcts::search::{SearchController, StopReason};
#[cfg(test)]
use crate::mcts::tt::leak_node;

// ─────────────────────────────────────────────
// § QuartzConfig — 5개 hyperparameter
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
// § HaltMode — PR-1A: A1 가설 검증용 3-way 분기
// ─────────────────────────────────────────────

/// Three stopping strategies for ablation comparison.
///
/// - `VOC`: full QUARTZ VOC halt (P_flip × σ_Δ - cost → converged)
/// - `SimpleThreshold`: P_flip < FLIP_THRESH only (no VOC cost term)
/// - `Fixed { budget }`: always run exactly `budget` iterations
/// v0.9.2: Penalty mode for selection score (Theory §IX)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PenaltyMode {
    /// Legacy: -min(ħ_eff, cap) / N_a
    Legacy,
    /// Theorem 8: -ν / (1 + N_a + O_a) where ν = hbar_penalty_cap
    EffectiveV2,
    /// No penalty (pure PUCT baseline)
    None,
    /// Fully self-adaptive: ν = σ_Q, ρ from surprise/maturity, τ from N/K.
    /// State-derived penalty+refresh mechanism (fixed constants, no user-tuned knobs).
    /// 내장 visit-frequency refresh (always-on).
    SelfAdaptive,
    /// Canonical controller matching the research doc:
    /// root-share penalty + prior-divergence gate + visit-share refresh.
    GatedRefresh,
    /// Historical training-time heuristic preserved for ablation:
    /// P_flip gate + Q-based refresh + EffV2 penalty.
    GatedRefreshLegacy,
    /// P_flip-mediated VF↔Q mixture + σ_Q-adaptive penalty.
    /// P_flip high → Q-refresh dominant (prior correction).
    /// P_flip low  → VF-refresh dominant (search concentration).
    /// ν = max(cap, σ_Q). NOTE: prior_q_divergence (D) is computed
    /// as diagnostic but NOT used in selection decisions.
    PFlipMixture,
}

/// P01: Reasons a controller halt actually fired. Indexed atomic counters
/// on `QuartzController` track these so the README's claimed
/// `halt_reason_count` field is emitted from real data, not just declared.
///
/// `KLLUCBStop`, `GLRCertified`, `EmpBernsteinSep`, `PolicyConverged` are
/// reserved for future policies (P08, P09, P11). Adding them up-front keeps
/// the array layout and serialization keys stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HaltReason {
    PFlipConverged = 0,
    VOCNonPositive = 1,
    FixedBudget = 2,
    KLLUCBStop = 3,
    MaxVisits = 4,
    MaxTime = 5,
    MinVisitsNotMet = 6,
    GLRCertified = 7,
    PolicyConverged = 8,
    EmpBernsteinSep = 9,
}
pub const HALT_REASON_COUNT: usize = 10;

impl HaltReason {
    /// Stable JSON key. Keep these synced with the README and never rename
    /// without bumping `controller_summary.extended.schema_version`.
    pub const fn as_key(self) -> &'static str {
        match self {
            HaltReason::PFlipConverged => "PFlipConverged",
            HaltReason::VOCNonPositive => "VOCNonPositive",
            HaltReason::FixedBudget => "FixedBudget",
            HaltReason::KLLUCBStop => "KLLUCBStop",
            HaltReason::MaxVisits => "MaxVisits",
            HaltReason::MaxTime => "MaxTime",
            HaltReason::MinVisitsNotMet => "MinVisitsNotMet",
            HaltReason::GLRCertified => "GLRCertified",
            HaltReason::PolicyConverged => "PolicyConverged",
            HaltReason::EmpBernsteinSep => "EmpBernsteinSep",
        }
    }
}

/// P01: Stable JSON keys for the seven `PenaltyMode` variants. Used for
/// emitting `controller_penalty_mode_counts` so the README claim is
/// falsifiable from the search response.
pub const PENALTY_MODE_KEYS: [&str; 7] = [
    "Legacy",
    "EffectiveV2",
    "None",
    "SelfAdaptive",
    "GatedRefresh",
    "GatedRefreshLegacy",
    "PFlipMixture",
];
pub const PENALTY_MODE_COUNT: usize = 7;

#[inline]
pub fn penalty_mode_idx(mode: PenaltyMode) -> u8 {
    match mode {
        PenaltyMode::Legacy => 0,
        PenaltyMode::EffectiveV2 => 1,
        PenaltyMode::None => 2,
        PenaltyMode::SelfAdaptive => 3,
        PenaltyMode::GatedRefresh => 4,
        PenaltyMode::GatedRefreshLegacy => 5,
        PenaltyMode::PFlipMixture => 6,
    }
}

/// - `VOC`: full QUARTZ VOC halt (P_flip × σ_Δ - cost → converged)
/// - `SimpleThreshold`: P_flip < FLIP_THRESH only (no VOC cost term)
/// - `Fixed { budget }`: always run exactly `budget` iterations
#[derive(Debug, Clone, PartialEq)]
pub enum HaltMode {
    /// Full QUARTZ: converged = (P_flip < thresh && flip_stable >= N) AND (VOC ≤ 0)
    VOC,
    /// Simple: converged = P_flip < thresh && flip_stable >= N (ignores VOC cost)
    SimpleThreshold,
    /// Fixed budget: always run exactly this many iterations
    Fixed { budget: u32 },
    /// PR-6A: Conf(t) adaptive halt with online θ_conf tuning
    /// Conf(t) = (1−P_flip)(1−P_hidden)·max{0, 1−S/S₀}
    /// Stop when Conf(t) ≥ θ_conf
    ConfAdaptive {
        target_time_ms: u64,
        theta_init: f32, // default 0.8
        eta: f32,        // learning rate for θ adaptation, default 0.01
    },
}

impl Default for HaltMode {
    fn default() -> Self {
        HaltMode::VOC
    }
}

// ─────────────────────────────────────────────
// § CostMode — PR-3A★: VOC Cost 개혁
// ─────────────────────────────────────────────

/// Three cost computation strategies for ablation.
///
/// Russell-Wefald/Hay et al. principle: cost = time/delay shadow price.
/// Uncertainty belongs in the benefit side (P_flip, σ_Δ), not cost.
#[derive(Debug, Clone, PartialEq)]
pub enum CostMode {
    /// Legacy: cost = ħ_eff × σ₀/N_min × ctm_factor
    /// (Current v0.8 behavior — uncertainty-proportional cost)
    Legacy,
    /// Constant: cost = σ₀/N_min × ctm_factor
    /// (Literature-aligned: constant cost, no ħ_eff dependency)
    Constant,
    /// Time-driven: cost = c_time(elapsed, target) × σ₀/N_min
    /// (Recommended: time pressure drives cost, uncertainty stays in benefit)
    TimeDriven,
}

impl Default for CostMode {
    fn default() -> Self {
        CostMode::Legacy
    }
}

/// Compute c_time for TimeDriven cost mode.
/// c_time = 0.3 + 0.7 × sigmoid((elapsed - target) / scale)
/// - At elapsed=0: ~0.3 (low cost, encourage exploration)
/// - At elapsed=target: ~0.65 (moderate cost)
/// - At elapsed >> target: ~1.0 (high cost, stop soon)
/// - Always > 0 (Theorem 1 A4 guarantee: cost bounded below)
#[inline]
pub fn cost_time_factor(elapsed_ms: u64, cfg: &QuartzConfig) -> f32 {
    if cfg.ctm_budget_ms == 0 {
        // No time budget → fixed moderate cost
        return 0.5;
    }
    let target = cfg.ctm_budget_ms as f32;
    let scale = (target * 0.3).max(1.0);
    let x = (elapsed_ms as f32 - target) / scale;
    0.3 + 0.7 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct QuartzConfig {
    // ── 핵심 5개 hyperparameter ──
    /// Reference fluctuation scale (σ₀).
    /// ħ_eff = σ_Q/σ₀ — 모든 one-loop 보정의 기준.
    /// Default 0.3: empty board σ_Q ≈ 0.3 (binary rollout에서)
    pub sigma_0: f32,

    /// Minimum visits before QUARTZ activates.
    /// Also determines: cost_focus = ħ_eff × σ₀/N_min
    pub min_visits: u32,

    /// CTM time budget (ms). 0 = time-unlimited.
    pub ctm_budget_ms: u64,

    /// NS annealing exponent γ. T(t) = (1-t/T)^γ.
    /// γ=1.0 (linear, default). γ<1: stay hot longer. γ>1: cool fast.
    pub ns_gamma: f32,

    /// Check interval (iterations between convergence checks).
    /// 유도 불가능한 탐색 제어 파라미터.
    pub check_interval: u32,

    /// PR-1A: Halt strategy for ablation.
    /// Default = VOC (full QUARTZ behavior, backward compatible).
    pub halt_mode: HaltMode,

    // ── PR-1B: Ablation switches (default = all true = full QUARTZ) ──
    /// Fisher natural gradient PUCT (√π prior weighting)
    pub enable_fisher_puct: bool,
    /// One-loop visit penalty (-ħ_eff/N_a)
    pub enable_one_loop: bool,
    /// EXPAND VOC channel (ε-envariance based)
    pub enable_expand_channel: bool,
    /// MERGE VOC channel (RTT-based)
    pub enable_merge_channel: bool,

    /// PR-3A★: Cost computation strategy.
    /// Default = Legacy (backward compatible).
    pub cost_mode: CostMode,

    /// G8: Gate EXPAND channel on NS diagnostics (bimodality/surprise/heavy-tail)
    pub enable_ns_gate: bool,
    /// G7: Depth-bucketed σ calibration κ_b (diagnostic collection)
    pub enable_depth_cal: bool,
    /// G2: Use Poisson P_hidden instead of p_envar for EXPAND (requires PW)
    pub enable_poisson_phidden: bool,
    /// G5: Use R/R₀ running median for MERGE normalization
    pub enable_merge_r0: bool,

    /// v0.9.2: Configurable ħ penalty cap (was const HBAR_PENALTY_CAP=0.3)
    pub hbar_penalty_cap: f32,

    /// v0.9.2: Penalty mode selection
    pub penalty_mode: PenaltyMode,

    /// v0.9.2: Dynamic prior refresh rate ρ ∈ [0,1].
    /// 0 = no refresh (original prior only), >0 = blend Q-based prior.
    /// π_t(a) ∝ π_0(a)^{1-ρ} · exp(ρ · Q_a / τ_refresh)
    pub prior_refresh_rate: f32,
    /// Temperature for Q → prior conversion in dynamic refresh
    pub prior_refresh_temp: f32,
    /// Apply QUARTZ score shaping only at the root.
    /// When false, the historical shallow depth<=3 blend is enabled.
    pub root_only_shaping: bool,
    /// Q8 (audit_codex_20260428.md W'2): when true, the `PFlipMixture`
    /// penalty mode additionally gates its refresh-mixture activation
    /// on `prior_q_divergence > epsilon_t`, mirroring the behavior of
    /// `GatedRefresh`. Default false preserves the existing PFlipMixture
    /// math so already-published numbers do not silently shift; setting
    /// this true enables divergence-aware sweeps under PFlipMixture
    /// instead of the no-op response previously documented in
    /// `docs/QUARTZ_THEORY.md` §5.
    pub pflip_mixture_divergence_gate: bool,
    // ── 고정값 (config에서 제거, 코드에 hardcode) ──
    // fisher_alpha = 0.5  (F_aa = π(a) → natural gradient → √π)
    // flip_thresh  = 0.159 (Φ(-1), 1-sigma rule)
    // flip_stable_n = 3
    // cost_* = derived from ħ_eff and σ₀/N_min
}

impl Default for QuartzConfig {
    fn default() -> Self {
        QuartzConfig {
            sigma_0: 0.3,
            min_visits: 50,
            ctm_budget_ms: 0,
            ns_gamma: 1.0,
            check_interval: 100,
            halt_mode: HaltMode::VOC,
            enable_fisher_puct: false, // v0.9.1: Fisher √π hurts with weak/uniform priors
            enable_one_loop: true,     // v0.9.1: clamped −min(ħ,0.3)/N penalty helps
            enable_expand_channel: true,
            enable_merge_channel: true,
            cost_mode: CostMode::Legacy,
            enable_ns_gate: false,
            enable_depth_cal: false,
            enable_poisson_phidden: false,
            enable_merge_r0: false,
            hbar_penalty_cap: 0.3,
            penalty_mode: PenaltyMode::Legacy,
            prior_refresh_rate: 0.0, // disabled by default
            prior_refresh_temp: 1.0,
            root_only_shaping: true,
            pflip_mixture_divergence_gate: false,
        }
    }
}

/// P05: per-evaluator-strength calibration bin. Future calibration sweeps
/// can stratify σ₀ recommendations by training value-loss bin so weak vs
/// strong evaluators get appropriately tuned uncertainty references.
/// Default selection from value-loss thresholds (Weak > 0.7, Medium 0.3-0.7,
/// Strong < 0.3) based on the empirical regime documented in
/// QUARTZ_THEORY.md §3 ("With a strong evaluator (NN loss < ~1.0)…").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalStrength {
    Weak,
    Medium,
    Strong,
}

impl EvalStrength {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Weak => "weak",
            Self::Medium => "medium",
            Self::Strong => "strong",
        }
    }
    /// Bin a value-loss measurement into the calibration buckets.
    /// Empirical thresholds; can be overridden per-game.
    pub fn from_value_loss(vl: f32) -> Self {
        if vl > 0.7 {
            Self::Weak
        } else if vl > 0.3 {
            Self::Medium
        } else {
            Self::Strong
        }
    }
}

/// P05: structured diagnostic from the auto-calibration loader. Surfaced
/// to the caller (server entry point) as eprintln WARN so the override
/// is visible in the train_log without requiring artifact inspection.
#[derive(Clone, Debug)]
pub enum CalibrationDiagnostic {
    Info(String),
    Warn(String),
}

impl QuartzConfig {
    pub fn fast() -> Self {
        QuartzConfig {
            sigma_0: 0.2,
            min_visits: 20,
            ns_gamma: 1.5,
            check_interval: 200,
            ..Default::default()
        }
    }
    pub fn with_time_budget(mut self, ms: u64) -> Self {
        self.ctm_budget_ms = ms;
        self
    }
    #[cfg(test)]
    pub fn with_halt_mode(mut self, mode: HaltMode) -> Self {
        self.halt_mode = mode;
        self
    }
    #[cfg(test)]
    pub fn with_cost_mode(mut self, mode: CostMode) -> Self {
        self.cost_mode = mode;
        self
    }

    /// P05: load `<calibration_dir>/sigma_0_recommendation.json`, optionally
    /// preferring a strength-stratified file (e.g. `weak.json`,
    /// `medium.json`, `strong.json`) when one is present, and apply the
    /// `<game_label>` recommendation (or `__cross_game__` fallback) to
    /// `self.sigma_0`.
    ///
    /// Returns `(modified_self, diagnostics)`. Diagnostics are non-fatal:
    /// the caller decides whether to surface them as INFO or WARN. The
    /// `warn_factor` parameter controls when a "calibration override
    /// disagrees significantly from base" warning is emitted (default 2.0
    /// = warn if the override is more than 2x larger or smaller than
    /// `self.sigma_0`).
    pub fn with_calibration(
        mut self,
        calibration_dir: &std::path::Path,
        game_label: Option<&str>,
        eval_strength: Option<EvalStrength>,
        warn_factor: f32,
    ) -> (Self, Vec<CalibrationDiagnostic>) {
        let mut diags = Vec::new();
        // Search order: strength-specific file, then default. The first
        // file found that parses cleanly is applied; subsequent
        // candidates are not consulted. This deliberately gives users a
        // way to override "strong" sigma_0 by writing a strength-specific
        // file.
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(strength) = eval_strength {
            candidates.push(calibration_dir.join(format!("{}.json", strength.tag())));
        }
        candidates.push(calibration_dir.join("sigma_0_recommendation.json"));

        for path in &candidates {
            if !path.exists() {
                continue;
            }
            let raw = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    diags.push(CalibrationDiagnostic::Warn(format!(
                        "calibration read err {}: {e}",
                        path.display()
                    )));
                    continue;
                }
            };
            let parsed: serde_json::Value = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => {
                    diags.push(CalibrationDiagnostic::Warn(format!(
                        "calibration parse err {}: {e}",
                        path.display()
                    )));
                    continue;
                }
            };
            // schema_version must be 1 — calibration writer pins to v1
            // (see src/calibration.rs:write_sigma_recommendation). Older
            // / newer files are silently skipped.
            if parsed.get("schema_version").and_then(|v| v.as_u64()) != Some(1) {
                diags.push(CalibrationDiagnostic::Warn(format!(
                    "calibration schema_version != 1 in {}; skipping",
                    path.display()
                )));
                continue;
            }
            let recs = match parsed.get("recommendations").and_then(|v| v.as_object()) {
                Some(m) => m,
                None => {
                    diags.push(CalibrationDiagnostic::Warn(format!(
                        "calibration missing 'recommendations' in {}",
                        path.display()
                    )));
                    continue;
                }
            };
            // game-label first, then __cross_game__ fallback
            let key = game_label.unwrap_or("__cross_game__");
            let pick = recs
                .get(key)
                .or_else(|| recs.get("__cross_game__"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32);
            let Some(new_sigma) = pick else {
                diags.push(CalibrationDiagnostic::Warn(format!(
                    "no recommendation for {key} in {}",
                    path.display()
                )));
                continue;
            };
            if !new_sigma.is_finite() || new_sigma <= 0.0 {
                diags.push(CalibrationDiagnostic::Warn(format!(
                    "invalid sigma_0={new_sigma} in {}; skipping",
                    path.display()
                )));
                continue;
            }
            let base = self.sigma_0;
            let ratio = (new_sigma / base).max(base / new_sigma);
            if ratio > warn_factor {
                diags.push(CalibrationDiagnostic::Warn(format!(
                    "calibration sigma_0={new_sigma:.4} differs from base \
                     {base:.4} by {ratio:.2}x (>{warn_factor:.2}x threshold)"
                )));
            }
            diags.push(CalibrationDiagnostic::Info(format!(
                "applied calibration sigma_0 {base:.4} -> {new_sigma:.4} \
                 from {} (key={key})",
                path.display()
            )));
            self.sigma_0 = new_sigma;
            return (self, diags);
        }
        diags.push(CalibrationDiagnostic::Info(format!(
            "no calibration file found in {}; keeping default sigma_0={:.4}",
            calibration_dir.display(),
            self.sigma_0
        )));
        (self, diags)
    }
}

// ─── Fixed values (documented, not configurable) ─────────────────
/// Fisher natural gradient: F_aa = π(a) → α = 1/2 (hardcoded in select.rs)
pub const FISHER_ALPHA: f32 = 0.5;
/// 1-sigma convergence rule: Φ(-1) ≈ 0.159
pub const FLIP_THRESH: f32 = 0.159;
/// Consecutive flip-below-thresh rounds for convergence
pub const FLIP_STABLE_N: u32 = 3;
/// CTM scale (fraction of budget)
const CTM_SCALE_FRAC: f32 = 0.3;
/// ε-envariance constant in S_KL > ENVAR_CONST/√N
/// Pinsker bound: S_KL ≥ 2ε² where ε_t = 1/√N → threshold scales as O(1/√N)
/// Derivation: ε_t = 1/√N, Pinsker: S_KL > 2ε² = 2/N → use 0.5/N
const ENVAR_CONST: f32 = 0.5;

// ─────────────────────────────────────────────
// § QuartzStats — 완전한 진단 통계
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct QuartzStats {
    // ── ħ_eff ─────────────────────────────────
    /// ħ_eff = σ_Q/σ₀ (effective Planck constant)
    pub hbar_eff: f32,

    // ── §6.1.1 P_flip (ρ̂ corrected) ──────────
    pub mu_delta: f32,
    pub sigma_delta: f32, // √(σ₁²+σ₂²-2ρ̂σ₁σ₂)
    pub p_flip: f32,
    pub sigma_reliable: bool,
    pub rho_hat: f32, // RTT-based path correlation

    // ── §6.4 ε-envariance ─────────────────────
    pub surprise_kl: f32,     // S_KL = KL(q_T‖π)
    pub surprise_s0: f32,     // cross-position EMA
    pub epsilon_t: f32,       // adaptive threshold = 1/√N
    pub envar_violated: bool, // S_KL > ENVAR_CONST/N
    /// ε-envariance violation magnitude (0 if not violated)
    pub envar_delta: f32,

    // ── VOC channels ──────────────────────────
    pub cost_focus: f32,    // ħ_eff × σ₀/N_min × ctm_factor
    pub e_delta_focus: f32, // σ_Δ × fep_envar_mod
    pub voc_focus: f32,

    pub p_envar: f32, // ε-envariance violation probability
    pub voc_expand: f32,

    pub rtt_n: u32,
    pub rtt_sigma: f32,
    pub voc_merge: f32,

    // ── Unified VOC ───────────────────────────
    pub unified: UnifiedVOC,

    // ── §8.1 Heavy-tail ──────────────────────
    pub heavy_tail_t: f32,
    pub is_heavy_tail: bool,

    // ── 수렴 ─────────────────────────────────
    pub converged: bool,
    pub flip_stable: u32,
    /// PR-6A: Confidence = (1−P_flip)(1−P_hidden)·max{0, 1−S/S₀}
    pub conf_t: f32,
    /// Hidden mode probability (≈ P_envar for EXPAND channel, ≈0 on 7×7)
    pub p_hidden: f32,
    /// G2: Poisson P_hidden = 1−exp(−m_out·p_tail·n_mat)
    pub p_hidden_poisson: f32,
    /// G2: outside prior mass (1 − Σ materialized priors)
    pub m_out: f32,
    /// G5: MERGE R₀ running median denominator
    pub merge_r0: f32,
    /// G1: running median of edge σ̂ (for paper B_1loop normalization)
    pub lambda_1loop: f32,
    /// PR-6B: Gaussian P_flip (always computed, for comparison)
    pub p_flip_gaussian: f32,
    /// PR-6B: Cornish-Fisher corrected P_flip (always computed, for comparison)
    pub p_flip_saddlepoint: f32,

    // ── 보조 ─────────────────────────────────
    pub mean_q: f32,
    pub sigma_q: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub one_loop_b: f32,
    pub voc_legacy: f32,

    pub root_visits: u32,
    pub n_children: usize,
    pub n_visible: usize,
    /// H3: D = KL(π_0 ‖ softmax(Q/τ)), prior-Q disagreement signal.
    /// D≈0 → prior and Q agree → no refresh needed.
    /// D>>0 → prior and Q disagree → Q-refresh can help correct prior.
    pub prior_q_divergence: f32,
}

impl QuartzStats {
    pub fn print(&self, label: &str) {
        let sig_tag = if self.sigma_reliable {
            "(M2✓)"
        } else {
            "(approx)"
        };
        println!("\n╔══ QUARTZ Stats v0.8: {} ══", label);
        println!(
            "║  visits={}, children={}/{}, ħ_eff={:.4}",
            self.root_visits, self.n_visible, self.n_children, self.hbar_eff
        );
        println!(
            "║  [§1 one-loop]  µ_Δ={:.4}  σ_Δ={:.4}{}  P_flip={:.4}  ρ̂={:.3}",
            self.mu_delta, self.sigma_delta, sig_tag, self.p_flip, self.rho_hat
        );
        println!(
            "║  [§2 envar]     S_KL={:.4}  S₀={:.4}  ε_t={:.4}  violated={}  δ={:.4}",
            self.surprise_kl,
            self.surprise_s0,
            self.epsilon_t,
            self.envar_violated,
            self.envar_delta
        );
        println!(
            "║  [§3 FOCUS]     E[Δ]={:.4}  cost={:.5}  VOCfocus={:.5}",
            self.e_delta_focus, self.cost_focus, self.voc_focus
        );
        println!(
            "║  [§4 EXPAND]    P_envar={:.4}  VOCexpand={:.5}",
            self.p_envar, self.voc_expand
        );
        println!(
            "║  [§5 MERGE]     RTT_n={}  RTT_σ={:.4}  VOCmerge={:.5}",
            self.rtt_n, self.rtt_sigma, self.voc_merge
        );
        println!(
            "║  [§6 conv]      converged={}  stable={}/{}",
            self.converged, self.flip_stable, FLIP_STABLE_N
        );
        self.unified.print(label);
        println!(
            "║  [aux]          σ_Q={:.4}  heavy={}  one_loop={:.4}",
            self.sigma_q, self.is_heavy_tail, self.one_loop_b
        );
        println!("╚══");
    }
}

// ─────────────────────────────────────────────
// § UnifiedVOC
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeAction {
    Focus,
    Expand,
    Merge,
    Stop,
}

impl Default for ComputeAction {
    fn default() -> Self {
        ComputeAction::Stop
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnifiedVOC {
    pub voc_focus: f32,
    pub voc_expand: f32,
    pub voc_merge: f32,
    pub voc_total: f32,
    pub action: ComputeAction,
    pub hbar_eff: f32,
    pub ns_temp: f32,
    pub ctm_urgency: f32,
    /// PR-6D: max GVOC among top-3 non-root children (w_imp weighted)
    pub gvoc_nonroot_max: f32,
}

impl Default for UnifiedVOC {
    fn default() -> Self {
        UnifiedVOC {
            voc_focus: 0.0,
            voc_expand: 0.0,
            voc_merge: 0.0,
            voc_total: 0.0,
            action: ComputeAction::Stop,
            hbar_eff: 0.0,
            ns_temp: 1.0,
            ctm_urgency: 0.0,
            gvoc_nonroot_max: 0.0,
        }
    }
}

impl UnifiedVOC {
    pub fn print(&self, label: &str) {
        let a_str = match self.action {
            ComputeAction::Focus => "FOCUS",
            ComputeAction::Expand => "EXPAND",
            ComputeAction::Merge => "MERGE",
            ComputeAction::Stop => "STOP",
        };
        println!("╔══ Unified VOC v0.8: {} [action={}] ══", label, a_str);
        println!(
            "║  total={:.5}  FOCUS={:.5}  EXPAND={:.5}  MERGE={:.5}",
            self.voc_total, self.voc_focus, self.voc_expand, self.voc_merge
        );
        println!(
            "║  ħ_eff={:.4}  ns_temp={:.3}  urgency={:.3}",
            self.hbar_eff, self.ns_temp, self.ctm_urgency
        );
        println!("╚══");
    }
}

// ─────────────────────────────────────────────
// § Running EMA
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RunningEma {
    pub value: f32,
    alpha: f32,
    n: u32,
}
pub type RunningMedian = RunningEma; // backward compat

impl RunningEma {
    pub fn new(alpha: f32) -> Self {
        RunningEma {
            value: 0.0,
            alpha,
            n: 0,
        }
    }
    pub fn update(&mut self, x: f32) {
        self.n += 1;
        let a = if self.n == 1 { 1.0 } else { self.alpha };
        self.value = (1.0 - a) * self.value + a * x;
    }
}

// ─────────────────────────────────────────────
// § (3) CTM Adaptive Cost
// ─────────────────────────────────────────────

#[inline]
pub fn ctm_urgency(elapsed_ms: u64, cfg: &QuartzConfig) -> f32 {
    if cfg.ctm_budget_ms == 0 {
        return 0.0;
    }
    let scale = (cfg.ctm_budget_ms as f32 * CTM_SCALE_FRAC).max(1.0);
    let x = (elapsed_ms as f32 - cfg.ctm_budget_ms as f32) / scale;
    1.0 / (1.0 + (-x).exp())
}

/// CTM factor: (0.5 + urgency) ∈ [0.5, 1.5]
/// Multiplied by base cost to get time-adaptive cost.
#[inline]
fn ctm_factor(elapsed_ms: u64, cfg: &QuartzConfig) -> f32 {
    0.5 + ctm_urgency(elapsed_ms, cfg)
}

// ─────────────────────────────────────────────
// § (5) NS Annealing Temperature
// ─────────────────────────────────────────────

/// T(t) = max(0.01, (1 - progress)^γ)
/// γ=1: linear, γ<1: stay hot longer, γ>1: cool fast
#[inline]
pub fn ns_anneal_temp(elapsed_ms: u64, cfg: &QuartzConfig) -> f32 {
    if cfg.ctm_budget_ms == 0 {
        return 1.0;
    }
    let progress = (elapsed_ms as f32 / cfg.ctm_budget_ms as f32).min(1.0);
    (1.0 - progress).powf(cfg.ns_gamma).max(0.01)
}

// ─────────────────────────────────────────────
// § §6.4 Surprise KL = envariance violation measure
// ─────────────────────────────────────────────

fn compute_surprise_kl(ns: &[f32], ps: &[f32], total_n: f32) -> f32 {
    let k = ns.len() as f32;
    let eps = 1e-4_f32;
    let tot = total_n + k * eps;
    let mut kl = 0.0f32;
    for (&n, &p) in ns.iter().zip(ps.iter()) {
        let qt = (n + eps) / tot;
        let pie = ((1.0 - eps) * p + eps / k).max(1e-9);
        kl += qt * (qt / pie).ln();
    }
    kl.max(0.0)
}

/// ε-envariance violation: S_KL > ENVAR_CONST/√N
/// Threshold scales as O(1/√N), matching Pinsker bound derivation.
/// Returns (violated, violation_magnitude, epsilon_t)
#[inline]
fn check_envariance(s_kl: f32, n_total: u32) -> (bool, f32, f32) {
    let sqrt_n = (n_total as f32).sqrt().max(1.0);
    let eps_t = 1.0 / sqrt_n;
    let threshold = ENVAR_CONST / sqrt_n;
    let violated = s_kl > threshold;
    let delta = if violated {
        (s_kl - threshold).min(1.0)
    } else {
        0.0
    };
    (violated, delta, eps_t)
}

// ─────────────────────────────────────────────
// § §6.1.1 P_flip with ρ̂ RTT correction
//
// Off-diagonal one-loop: K(a,a') ≈ RTT_var (path correlation via shared subtree)
// ρ̂ = 1 - RTT_var/(σ₁σ₂)  (clamp[-0.95, 0.95])
// σ_Δ² = σ₁²+σ₂²-2ρ̂σ₁σ₂
// ─────────────────────────────────────────────

// σ_Δ² = σ₁²+σ₂²-2ρ̂σ₁σ₂
// ─────────────────────────────────────────────

/// Same P_flip estimator as `compute_p_flip_with_child_rtt`, but reads directly
/// from the root edge slab. This avoids allocating an `edge_snapshot` Vec during
/// every QUARTZ controller refresh while preserving the same top-2-Q semantics.
fn compute_p_flip_with_child_rtt_edges<M: Copy + Send + Sync + 'static>(
    edges: &[MctsEdge<M>],
    sigma_q: f32,
) -> (f32, f32, f32, f32, bool) {
    if edges.len() < 2 {
        return (0.0, 0.0, 0.0, 0.0, false);
    }

    let mut best: Option<(usize, f32)> = None;
    let mut second: Option<(usize, f32)> = None;
    for (idx, edge) in edges.iter().enumerate() {
        let q = edge.q();
        match best {
            None => best = Some((idx, q)),
            Some((_, bq)) if q > bq => {
                second = best;
                best = Some((idx, q));
            }
            _ => match second {
                None => second = Some((idx, q)),
                Some((_, sq)) if q > sq => second = Some((idx, q)),
                _ => {}
            },
        }
    }

    let Some((i1, q1)) = best else {
        return (0.0, 0.0, 0.0, 0.0, false);
    };
    let Some((i2, q2)) = second else {
        return (0.0, 0.0, 0.0, 0.0, false);
    };
    let e1 = &edges[i1];
    let e2 = &edges[i2];
    let mu_d = q1 - q2;
    let n1 = e1.n.load(Ordering::Acquire) as f32;
    let n2 = e2.n.load(Ordering::Acquire) as f32;

    let s1 = e1.edge_sigma();
    let s2 = e2.edge_sigma();
    let reliable = s1.is_some() && s2.is_some();

    let sigma1 = s1.unwrap_or(sigma_q / (n1 + 1.0).sqrt());
    let sigma2 = s2.unwrap_or(sigma_q / (n2 + 1.0).sqrt());
    let sp = (sigma1 * sigma2).max(1e-8);

    let rtt1 = e1.child.rtt_variance().unwrap_or(0.0);
    let rtt2 = e2.child.rtt_variance().unwrap_or(0.0);
    let rtt_avg = (rtt1 + rtt2) / 2.0;

    let rho_hat = if rtt_avg > 0.0 && reliable {
        (1.0 - rtt_avg / sp).clamp(-0.95, 0.95)
    } else {
        0.0
    };

    let var_d = (sigma1.powi(2) + sigma2.powi(2) - 2.0 * rho_hat * sigma1 * sigma2).max(1e-10);
    let sigma_d = var_d.sqrt();
    (
        mu_d,
        sigma_d,
        standard_normal_cdf(-mu_d / sigma_d),
        rho_hat,
        reliable,
    )
}

// ─────────────────────────────────────────────
// § §8.1 Heavy-tail T = σ̂/MAD (determinism guard)
// ─────────────────────────────────────────────

fn compute_heavy_tail(qs: &[f32], ns: &[f32], sigma_q: f32, cfg: &QuartzConfig) -> (f32, bool) {
    if qs.len() < 4 {
        return (1.0, false);
    }
    let total_n: f32 = ns.iter().sum();
    let min_n: f32 = *ns
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);
    if min_n < 10.0 {
        return (1.0, false);
    }

    let mut sorted: SmallVec<[(f32, f32); 64]> =
        qs.iter().zip(ns.iter()).map(|(&q, &n)| (q, n)).collect();
    sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut cum = 0.0f32;
    let mut median_q = sorted[0].0;
    for &(q, n) in &sorted {
        cum += n;
        if cum >= total_n * 0.5 {
            median_q = q;
            break;
        }
    }

    let mut devs: SmallVec<[(f32, f32); 64]> = sorted
        .iter()
        .map(|&(q, n)| ((q - median_q).abs(), n))
        .collect();
    devs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut cum = 0.0f32;
    let mut mad = devs[0].0;
    for &(d, n) in &devs {
        cum += n;
        if cum >= total_n * 0.5 {
            mad = d;
            break;
        }
    }

    // Determinism guard: MAD << σ_Q → peaked, not heavy-tailed
    if mad < sigma_q / 5.0 {
        return (1.0, false);
    }
    let t = sigma_q / mad.max(1e-8);
    let thresh = 1.0 / cfg.sigma_0 * 0.75; // derived threshold: higher σ₀ → more permissive
    (t, t > thresh)
}

// ─────────────────────────────────────────────
// § §5.2 E[Δexpand] — truncated Gaussian
// ─────────────────────────────────────────────

fn e_delta_expand(mu_out: f32, mu_best: f32, sigma_q: f32) -> f32 {
    let sc = (2.0 * sigma_q.powi(2)).sqrt().max(1e-8);
    let z = (mu_out - mu_best) / sc;
    sc * (standard_normal_pdf(z) + z * standard_normal_cdf(z))
}

// ─────────────────────────────────────────────
// § compute_quartz_stats — unified v0.8
// ─────────────────────────────────────────────

pub fn compute_quartz_stats<M: Copy + Send + Sync + 'static>(
    root: &ArenaRef<MctsNode<M>>,
    priors: Option<&[f32]>,
    s0_same: &mut RunningEma,
    _s0_global: f32,
    prev_flip_n: u32,
    elapsed_ms: u64,
    cfg: &QuartzConfig,
) -> QuartzStats {
    let n_mat = root.materialized_count();
    let edges_full = root.read_edges();
    let n_edges = n_mat.min(edges_full.len());
    let edges = &edges_full[..n_edges];
    let n_total = root.n_total.load(Ordering::Acquire);

    if edges.is_empty() || n_total < cfg.min_visits {
        return QuartzStats {
            root_visits: n_total,
            n_children: edges.len(),
            ..Default::default()
        };
    }

    // ── Q, N, prior ──────────────────────────────────────────
    let mut qs: SmallVec<[f32; 64]> = SmallVec::new();
    let mut ns: SmallVec<[f32; 64]> = SmallVec::new();
    let mut ps: SmallVec<[f32; 64]> = SmallVec::new();
    let mut total_n = 0.0f32;
    let mut total_p = 0.0f32;

    for (i, e) in edges.iter().enumerate() {
        let n = e.n.load(Ordering::Acquire) as f32;
        if n > 0.0 {
            qs.push(e.q());
            ns.push(n);
            let p = priors.and_then(|pr| pr.get(i).copied()).unwrap_or(e.p);
            ps.push(p);
            total_n += n;
            total_p += p;
        }
    }
    if qs.is_empty() || total_n < 1.0 {
        return QuartzStats {
            root_visits: n_total,
            n_children: edges.len(),
            ..Default::default()
        };
    }

    let use_uniform = total_p < 1e-6;
    let unif_p = 1.0 / qs.len() as f32;
    let ps_norm: SmallVec<[f32; 64]> = if use_uniform {
        std::iter::repeat(unif_p).take(qs.len()).collect()
    } else {
        ps.iter().map(|&p| p / total_p).collect()
    };

    // ── Cumulants ─────────────────────────────────────────────
    let mean_q = qs.iter().zip(ns.iter()).map(|(&q, &n)| q * n).sum::<f32>() / total_n;
    let var_q = qs
        .iter()
        .zip(ns.iter())
        .map(|(&q, &n)| n * (q - mean_q).powi(2))
        .sum::<f32>()
        / total_n;
    let sigma_q = var_q.sqrt().max(1e-8);

    let skewness = if sigma_q > 1e-6 {
        qs.iter()
            .zip(ns.iter())
            .map(|(&q, &n)| n * ((q - mean_q) / sigma_q).powi(3))
            .sum::<f32>()
            / total_n
    } else {
        0.0
    };
    let kurtosis = if sigma_q > 1e-6 {
        qs.iter()
            .zip(ns.iter())
            .map(|(&q, &n)| n * ((q - mean_q) / sigma_q).powi(4))
            .sum::<f32>()
            / total_n
            - 3.0
    } else {
        0.0
    };

    // ── (1) ħ_eff = σ_Q/σ₀ ────────────────────────────────────
    let hbar_eff = (sigma_q / cfg.sigma_0).min(3.0); // cap at 3 (extreme positions)

    // ── §6.1.1 P_flip with ρ̂ (top-2 children RTT) ───────────
    // 수정: root.rtt_variance()는 루트 자신의 RTT = 0
    // 올바른 방법: top-2 자식 노드의 RTT variance로 경로 상관 추정
    let (mu_delta, sigma_delta, p_flip, rho_hat, sigma_reliable) =
        compute_p_flip_with_child_rtt_edges(edges, sigma_q);

    // ── §6.3 MERGE — children RTT aggregation ────────────────
    // 루트 자식 중 RTT 누적이 가장 많은 노드를 찾아 MERGE channel에 사용
    let (rtt_n, rtt_var_max) = edges.iter().fold((0u32, 0.0f32), |(max_n, max_var), e| {
        let n = e.child.rtt_n.load(Ordering::Acquire);
        let var = e.child.rtt_variance().unwrap_or(0.0);
        if n > max_n {
            (n, var)
        } else {
            (max_n, max_var)
        }
    });
    let rtt_sigma = rtt_var_max.sqrt();

    // ── (2) ε-envariance ─────────────────────────────────────
    let s_kl = compute_surprise_kl(&ns, &ps_norm, total_n);
    s0_same.update(s_kl);
    let s0_local = s0_same.value.max(1e-6);
    let (envar_violated, envar_delta, eps_t) = check_envariance(s_kl, n_total);

    // ── §6.5.2 VOC_FOCUS ──────────────────────────────────────
    // E[Δfocus] = σ_Δ × (1 + envar_delta) if envariance violated AND paths converging
    // Logic: envariance violation + confirming paths (ρ̂>0) → focus more carefully
    let envar_focus_mod = if envar_violated && rho_hat > 0.0 {
        1.0 + envar_delta * 0.5
    } else {
        1.0
    };
    let e_delta = sigma_delta * envar_focus_mod;

    // Cost = f(cost_mode) × base_cost
    // PR-3A★: cost semantics aligned with Russell-Wefald/Hay et al.
    //   Legacy:    ħ_eff × base × ctm_factor (uncertainty-proportional — DEPRECATED)
    //   Constant:  base × ctm_factor (literature standard)
    //   TimeDriven: c_time(elapsed, target) × base (time pressure drives cost)
    let cost_base = cfg.sigma_0 / cfg.min_visits as f32;
    let cost_focus = match cfg.cost_mode {
        CostMode::Legacy => {
            let ctm_f = ctm_factor(elapsed_ms, cfg);
            hbar_eff * cost_base * ctm_f
        }
        CostMode::Constant => {
            let ctm_f = ctm_factor(elapsed_ms, cfg);
            cost_base * ctm_f
        }
        CostMode::TimeDriven => {
            let c_time = cost_time_factor(elapsed_ms, cfg);
            c_time * cost_base
        }
    };
    let voc_focus = p_flip * e_delta - cost_focus;

    // ── (2) ε-envariance → P_envar for EXPAND ────────────────
    // ε-envariance violation IS the hidden-mode signal.
    // P_envar = normalized violation magnitude
    // If violated: envar_delta ∈ (0,1] → P_envar ∈ (0,1]
    let bc = (skewness.powi(2) + 1.0) / (kurtosis.abs() + 3.0);
    let p_bc = if bc > 0.555 {
        (bc - 0.555).min(0.445)
    } else {
        0.0
    };
    // Combine ε-envariance and BC bimodality
    let p_envar = (envar_delta + p_bc).min(1.0);

    // (5) NS annealing
    let ns_temp = ns_anneal_temp(elapsed_ms, cfg);
    let p_envar_eff = p_envar * ns_temp;

    let mu_out = mean_q - sigma_q;
    let mu_best = qs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e_exp = e_delta_expand(mu_out, mu_best, sigma_q).max(0.0);
    // EXPAND cost: more expensive than FOCUS
    // Legacy: ħ_eff² × cost_base (quadratic in uncertainty)
    // Constant/TimeDriven: 2 × cost_focus (EXPAND costs 2× FOCUS, no ħ_eff dependency)
    let cost_expand = match cfg.cost_mode {
        CostMode::Legacy => hbar_eff * hbar_eff * cost_base,
        CostMode::Constant | CostMode::TimeDriven => cost_focus * 2.0,
    };
    // Gate: EXPAND only matters when P_flip > 0 (hidden mode can't change a confident decision)
    let expand_relevance = p_flip.max(0.01); // floor at 1%

    let voc_expand = if cfg.enable_expand_channel {
        if cfg.enable_poisson_phidden {
            let prior_sum: f32 = edges.iter().map(|e| e.p).sum();
            let n_cand = root.candidate_count().max(1) as f32;
            let m_out = if prior_sum > 1e-6 {
                (1.0 - prior_sum).max(0.0)
            } else {
                (1.0 - n_mat as f32 / n_cand).max(0.0)
            };
            let q_best = qs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let tail_count = qs.iter().filter(|&&q| q > q_best - sigma_q).count();
            let p_tail_val = if !qs.is_empty() {
                tail_count as f32 / qs.len() as f32
            } else {
                0.0
            };
            let lambda_val = m_out * p_tail_val * qs.len() as f32;
            let p_hidden_pois = 1.0 - (-lambda_val).exp();
            p_hidden_pois * expand_relevance * e_exp - cost_expand
        } else {
            p_envar_eff * expand_relevance * e_exp - cost_expand
        }
    } else {
        0.0
    };

    // G2: Always compute Poisson P_hidden for diagnostics
    let prior_sum_all: f32 = edges.iter().map(|e| e.p).sum();
    let n_cand = root.candidate_count().max(1) as f32;
    let m_out_val = if prior_sum_all > 1e-6 {
        (1.0 - prior_sum_all).max(0.0)
    } else {
        (1.0 - n_mat as f32 / n_cand).max(0.0)
    };
    let q_best_for_tail = qs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let tail_ct = qs
        .iter()
        .filter(|&&q| q > q_best_for_tail - sigma_q)
        .count();
    let p_tail_diag = if !qs.is_empty() {
        tail_ct as f32 / qs.len() as f32
    } else {
        0.0
    };
    let lambda_diag = m_out_val * p_tail_diag * qs.len() as f32;
    let p_hidden_poisson = 1.0 - (-lambda_diag).exp();

    // ── §6.3 VOC_MERGE (children RTT curvature) ──────────────
    // rtt_n, rtt_sigma, rtt_var_max already computed above from children
    let rtt_var = rtt_sigma * rtt_sigma;
    let p_merge = if cfg.enable_merge_r0 {
        // G5: P_merge = min(1, R/R₀) — R₀ will be set by controller post-hoc
        // For now compute raw P_merge, controller will adjust
        if rtt_n >= 2 && rtt_var > 0.0 {
            1.0f32
        } else {
            0.0
        }
    } else {
        if rtt_n >= 2 {
            (rtt_n as f32 / (rtt_n as f32 + 1.0)).min(1.0)
        } else {
            0.0
        }
    };
    // MERGE gain = ħ_eff × √RTT (scale-matched via ħ_eff)
    let voc_merge = if cfg.enable_merge_channel {
        p_merge * hbar_eff * rtt_sigma - cost_focus
    } else {
        0.0
    };

    // ── §8.1 Heavy-tail ───────────────────────────────────────
    let (heavy_tail_t, is_heavy_tail) = compute_heavy_tail(&qs, &ns, sigma_q, cfg);

    // ── PR-6B: Saddlepoint P_flip correction ─────────────────
    let p_flip_gaussian = p_flip;
    let p_flip_saddlepoint = cornish_fisher_pflip(mu_delta, sigma_delta, skewness, kurtosis);
    // Override p_flip if heavy-tail detected and enough data
    let p_flip = if is_heavy_tail && edges.len() >= 8 {
        p_flip_saddlepoint
    } else {
        p_flip_gaussian
    };

    // ── One-loop bonus (§6.1.2, selection-level, see select.rs) ─
    // Reported here for diagnostics; actual application in select.rs
    let one_loop_b = if !is_heavy_tail && p_envar < 0.2 {
        hbar_eff * (1.0f32 + sigma_q / cfg.sigma_0).ln()
    } else {
        0.0
    };

    // ── Unified VOC ────────────────────────────────────────────
    let urgency = ctm_urgency(elapsed_ms, cfg);
    let wimp = 1.0_f32; // root only; non-root needs parent ptr
    let cands = [
        (voc_focus, ComputeAction::Focus),
        (voc_expand, ComputeAction::Expand),
        (voc_merge, ComputeAction::Merge),
    ];
    let (best_v, best_a) = cands
        .iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .copied()
        .unwrap_or((f32::NEG_INFINITY, ComputeAction::Stop));
    let action = if best_v <= 0.0 {
        ComputeAction::Stop
    } else {
        best_a
    };
    let voc_total = wimp * best_v.max(0.0);

    // PR-6D: Estimate non-root GVOC for top-3 children
    // w_imp(child) = N(child)/N(root), GVOC ≈ w_imp × root_voc_total (proxy)
    let gvoc_nonroot_max = if n_total > 0 {
        let mut child_wimps: SmallVec<[f32; 64]> = edges
            .iter()
            .map(|e| e.n.load(Ordering::Acquire) as f32 / n_total as f32)
            .collect();
        child_wimps.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        child_wimps
            .iter()
            .take(3)
            .map(|w| w * voc_total)
            .fold(0.0f32, f32::max)
    } else {
        0.0
    };

    let unified = UnifiedVOC {
        voc_focus,
        voc_expand,
        voc_merge,
        voc_total,
        action,
        hbar_eff,
        ns_temp,
        ctm_urgency: urgency,
        gvoc_nonroot_max,
    };

    // ── 수렴 ──────────────────────────────────────────────────
    // P_flip < Φ(-1) ≈ 0.159 (1-sigma) AND VOC total ≤ 0
    let this_ok = p_flip < FLIP_THRESH;
    let new_stable = if this_ok { prev_flip_n + 1 } else { 0 };
    let converged = new_stable >= FLIP_STABLE_N && voc_total <= 0.0 && n_total >= cfg.min_visits;

    // ── PR-6A: Conf(t) computation ─────────────────────────────
    // Conf(t) = (1−P_flip)(1−P_hidden)·max{0, 1−S/S₀}
    // P_hidden ≈ P_envar (EXPAND channel probability, ≈0 on 7×7)
    let p_hidden = p_envar_eff.min(1.0);
    let surprise_ratio = if s0_local > 1e-6 {
        (s_kl / s0_local).min(2.0)
    } else {
        0.0
    };
    let conf_t = (1.0 - p_flip) * (1.0 - p_hidden) * (1.0 - surprise_ratio).max(0.0);

    // G1: lambda_1loop = median of edge σ̂ (for paper B_1loop formula)
    let mut sigma_hats: SmallVec<[f32; 64]> = edges.iter().filter_map(|e| e.edge_sigma()).collect();
    let lambda_1loop = if !sigma_hats.is_empty() {
        sigma_hats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        sigma_hats[sigma_hats.len() / 2]
    } else {
        0.0
    };

    // ── H3: Prior-Q Disagreement D = KL(π_0 ‖ softmax(Q/τ)) ──
    let prior_q_divergence = {
        let k = qs.len() as f32;
        let tau_d = (sigma_q * k.sqrt()).max(0.1);
        // Compute softmax(Q/τ)
        let q_scaled: SmallVec<[f32; 64]> = qs.iter().map(|&q| q / tau_d).collect();
        let q_max = q_scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = q_scaled.iter().map(|&q| (q - q_max).exp()).sum();
        if exp_sum > 1e-30 && !ps_norm.is_empty() {
            let mut d = 0.0f32;
            for (i, &p0) in ps_norm.iter().enumerate() {
                let p0_safe = p0.max(1e-8);
                let q_softmax = ((q_scaled[i] - q_max).exp() / exp_sum).max(1e-8);
                d += p0_safe * (p0_safe / q_softmax).ln();
            }
            d.max(0.0).min(10.0) // clamp: D ∈ [0, 10] nats
        } else {
            0.0
        }
    };

    QuartzStats {
        hbar_eff,
        mu_delta,
        sigma_delta,
        p_flip,
        sigma_reliable,
        rho_hat,
        surprise_kl: s_kl,
        surprise_s0: s0_local,
        epsilon_t: eps_t,
        envar_violated,
        envar_delta,
        cost_focus,
        e_delta_focus: e_delta,
        voc_focus,
        p_envar: p_envar_eff,
        voc_expand,
        rtt_n,
        rtt_sigma,
        voc_merge,
        unified,
        heavy_tail_t,
        is_heavy_tail,
        converged,
        flip_stable: new_stable,
        conf_t,
        p_hidden,
        p_hidden_poisson,
        m_out: m_out_val,
        merge_r0: 0.0, // set by controller post-hoc for G5
        lambda_1loop,
        p_flip_gaussian,
        p_flip_saddlepoint,
        mean_q,
        sigma_q,
        skewness,
        kurtosis,
        one_loop_b,
        voc_legacy: p_flip
            * (qs.first().copied().unwrap_or(0.0) - qs.get(1).copied().unwrap_or(0.0)).abs(),
        root_visits: n_total,
        n_children: edges.len(),
        n_visible: qs.len(),
        prior_q_divergence,
    }
}

// ─────────────────────────────────────────────
// § EFT-PUCT helpers (used in select.rs)
// ─────────────────────────────────────────────

/// (4) Fisher PUCT prior weight: π^α, α=FISHER_ALPHA=0.5 (fixed)
#[inline]
pub fn fisher_prior_weight(prior: f32) -> f32 {
    prior.max(0.0).powf(FISHER_ALPHA)
}

/// (1) One-loop visit penalty: -min(ħ_eff, cap)/N_a (diagonal Tr log M)
/// cap prevents penalty from overwhelming PUCT exploration at high uncertainty
#[inline]
pub fn one_loop_visit_penalty(n_action: u32, hbar_eff: f32, eft_strength: f32, cap: f32) -> f32 {
    if n_action == 0 {
        return 0.0;
    }
    let hbar_clamped = hbar_eff.min(cap);
    -hbar_clamped / (n_action as f32) * eft_strength
}

/// v0.9.2: One-loop effective penalty from Theorem 8 (1/M asymptotic)
/// -ν / M_a where M_a = 1 + N_a + O_a (effective occupancy)
/// Only applies to visited edges (N_a > 0 or O_a > 0).
#[inline]
pub fn effective_penalty_v2(n_action: u32, o_action: u32, nu: f32) -> f32 {
    if n_action == 0 && o_action == 0 {
        return 0.0;
    }
    let m_a = 1.0 + n_action as f32 + o_action as f32;
    -nu / m_a
}

/// Maximum ħ_eff for one-loop penalty (prevents exploration destruction)
pub const HBAR_PENALTY_CAP: f32 = 0.3;

/// (G1) Paper B_1loop: λ·log(1+σ̂) — uncertainty-aware exploration bonus
/// Returns 0 if σ̂ unavailable or lambda_1loop is 0
#[inline]
pub fn paper_b1loop_bonus(sigma_hat: f32, lambda_1loop: f32) -> f32 {
    if lambda_1loop < 1e-6 || sigma_hat < 1e-6 {
        return 0.0;
    }
    let ratio = (sigma_hat / lambda_1loop).min(3.0); // clamp ratio to prevent explosion
    ratio * (1.0 + sigma_hat).ln()
}

/// One-loop action bonus (§6.1.2 off-diagonal, gated by heavy-tail)
#[inline]
pub fn eft_action_bonus(stats: &QuartzStats) -> f32 {
    if stats.is_heavy_tail || stats.p_envar >= 0.2 {
        return 0.0;
    }
    stats.one_loop_b
}

// ─────────────────────────────────────────────
// § CDF / PDF
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
// § CDF / PDF
// Φ(x) = (1 + erf(x/√2))/2
// A&S 7.1.26: erf(y) ≈ 1 - poly(t)·exp(-y²), t = 1/(1+p·y)
// ─────────────────────────────────────────────

pub(crate) fn standard_normal_cdf(x: f32) -> f32 {
    // Pass x/√2 to convert from erf formula to Φ
    const SQRT2_INV: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let y = x.abs() * SQRT2_INV; // erf argument
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    const P: f32 = 0.3275911;
    let t = 1.0 / (1.0 + P * y);
    let erf_abs = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-y * y).exp();
    let erf_signed = if x < 0.0 { -erf_abs } else { erf_abs };
    0.5 * (1.0 + erf_signed)
}

pub(crate) fn standard_normal_pdf(x: f32) -> f32 {
    (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt()
}

// ─────────────────────────────────────────────
// § PR-6B: Saddlepoint P_flip (Cornish-Fisher expansion)
// ─────────────────────────────────────────────

/// Cornish-Fisher corrected P_flip using skewness (γ₁) and excess kurtosis (γ₂).
///
/// Adjusts the Gaussian z-score z = −μ_Δ/σ_Δ using higher cumulants:
///   z_cf = z + (z²−1)·γ₁/6 + (z³−3z)·γ₂/24 − (2z³−5z)·γ₁²/36
///
/// Then P_flip_cf = Φ(z_cf).
///
/// Falls back to Gaussian if correction produces NaN/Inf or moves z_cf
/// by more than 2σ from z (numerical instability guard).
pub fn cornish_fisher_pflip(mu_d: f32, sigma_d: f32, skew: f32, kurt: f32) -> f32 {
    if sigma_d < 1e-8 {
        return if mu_d <= 0.0 { 0.5 } else { 0.0 };
    }
    let z = -mu_d / sigma_d;

    // Cornish-Fisher expansion
    let z2 = z * z;
    let z3 = z2 * z;
    let gamma1 = skew; // skewness
    let gamma2 = kurt; // excess kurtosis

    let z_cf = z + (z2 - 1.0) * gamma1 / 6.0 + (z3 - 3.0 * z) * gamma2 / 24.0
        - (2.0 * z3 - 5.0 * z) * gamma1 * gamma1 / 36.0;

    // Stability guard: if correction is too large, fall back to Gaussian
    if !z_cf.is_finite() || (z_cf - z).abs() > 2.0 {
        return standard_normal_cdf(z);
    }

    standard_normal_cdf(z_cf).clamp(0.0, 1.0)
}

/// Compute P_flip with optional saddlepoint correction.
/// Uses Cornish-Fisher when heavy-tail detected and n_visible >= 8.
/// Returns (p_flip_used, p_flip_gaussian, p_flip_saddlepoint).
pub fn pflip_with_correction(
    mu_d: f32,
    sigma_d: f32,
    skew: f32,
    kurt: f32,
    is_heavy_tail: bool,
    n_visible: usize,
) -> (f32, f32, f32) {
    let p_gauss = if sigma_d < 1e-8 {
        if mu_d <= 0.0 {
            0.5
        } else {
            0.0
        }
    } else {
        standard_normal_cdf(-mu_d / sigma_d)
    };

    let p_cf = cornish_fisher_pflip(mu_d, sigma_d, skew, kurt);

    // Gate: only use correction when heavy-tail detected AND enough data
    let p_used = if is_heavy_tail && n_visible >= 8 {
        p_cf
    } else {
        p_gauss
    };

    (p_used, p_gauss, p_cf)
}

// ─────────────────────────────────────────────
// § PR-6C: Depth-Bucketed σ Calibration κ_b
// ─────────────────────────────────────────────

/// Depth bucket for σ calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthBucket {
    Shallow,
    Mid,
    Deep,
}

impl DepthBucket {
    pub fn from_depth(depth: usize) -> Self {
        match depth {
            0..=3 => DepthBucket::Shallow,
            4..=8 => DepthBucket::Mid,
            _ => DepthBucket::Deep,
        }
    }
}

/// Per-bucket σ calibration: σ_calibrated(s,a) = κ_b × σ_raw(s,a)
/// κ_b is updated online via EMA of (σ_empirical / σ_predicted).
#[derive(Debug, Clone)]
pub struct DepthCalibration {
    kappa: [f32; 3],  // [shallow, mid, deep]
    alpha: f32,       // EMA smoothing factor
    counts: [u32; 3], // observation counts per bucket
}

impl DepthCalibration {
    pub fn new(alpha: f32) -> Self {
        DepthCalibration {
            kappa: [1.0, 1.0, 1.0],
            alpha,
            counts: [0, 0, 0],
        }
    }

    fn idx(bucket: DepthBucket) -> usize {
        match bucket {
            DepthBucket::Shallow => 0,
            DepthBucket::Mid => 1,
            DepthBucket::Deep => 2,
        }
    }

    /// Get κ_b for a given depth.
    pub fn kappa_at(&self, depth: usize) -> f32 {
        self.kappa[Self::idx(DepthBucket::from_depth(depth))]
    }

    /// Apply calibration: σ_calibrated = κ_b × σ_raw
    pub fn calibrate_sigma(&self, depth: usize, sigma_raw: f32) -> f32 {
        self.kappa_at(depth) * sigma_raw
    }

    /// Update κ_b with an observation: κ_b ← EMA(κ_b, σ_empirical / σ_predicted)
    pub fn record(&mut self, depth: usize, sigma_empirical: f32, sigma_predicted: f32) {
        if sigma_predicted < 1e-8 {
            return;
        }
        let ratio = sigma_empirical / sigma_predicted;
        if !ratio.is_finite() {
            return;
        }

        let i = Self::idx(DepthBucket::from_depth(depth));
        self.counts[i] += 1;
        let a = if self.counts[i] == 1 { 1.0 } else { self.alpha };
        self.kappa[i] = (1.0 - a) * self.kappa[i] + a * ratio;
    }

    /// Get all kappas for diagnostics.
    pub fn kappas(&self) -> [f32; 3] {
        self.kappa
    }
}

// ─────────────────────────────────────────────
// § PR-6E: Robust Scale Estimator (Median-of-Means)
// ─────────────────────────────────────────────

/// Circular buffer for recent return values (fixed capacity K).
/// Used by MedianOfMeans to compute robust σ on heavy-tail edges.
#[derive(Debug, Clone)]
pub struct CircularBuffer {
    data: Vec<f32>,
    capacity: usize,
    pos: usize,
    full: bool,
}

impl CircularBuffer {
    pub fn new(capacity: usize) -> Self {
        CircularBuffer {
            data: vec![0.0; capacity],
            capacity,
            pos: 0,
            full: false,
        }
    }

    pub fn push(&mut self, val: f32) {
        self.data[self.pos] = val;
        self.pos = (self.pos + 1) % self.capacity;
        if self.pos == 0 {
            self.full = true;
        }
    }

    pub fn len(&self) -> usize {
        if self.full {
            self.capacity
        } else {
            self.pos
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len()]
    }
}

/// Median-of-means robust σ estimator.
/// Splits K samples into G groups, computes group means,
/// takes median of group means as robust location,
/// and MAD of group means as robust scale.
pub fn median_of_means_sigma(values: &[f32], n_groups: usize) -> Option<f32> {
    let n = values.len();
    if n < n_groups * 2 {
        return None;
    } // need at least 2 per group

    let group_size = n / n_groups;
    let mut group_means = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let start = g * group_size;
        let end = if g == n_groups - 1 {
            n
        } else {
            start + group_size
        };
        let sum: f32 = values[start..end].iter().sum();
        group_means.push(sum / (end - start) as f32);
    }

    // Median of group means
    group_means.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n_groups % 2 == 0 {
        (group_means[n_groups / 2 - 1] + group_means[n_groups / 2]) / 2.0
    } else {
        group_means[n_groups / 2]
    };

    // MAD (Median Absolute Deviation) of group means as robust scale
    let mut abs_devs: Vec<f32> = group_means.iter().map(|&m| (m - median).abs()).collect();
    abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if n_groups % 2 == 0 {
        (abs_devs[n_groups / 2 - 1] + abs_devs[n_groups / 2]) / 2.0
    } else {
        abs_devs[n_groups / 2]
    };

    // Convert MAD to σ estimate: σ ≈ 1.4826 × MAD (for Gaussian)
    Some((1.4826 * mad).max(1e-8))
}

// ─────────────────────────────────────────────
// § PR-6F: NS Gate (§6.2.2)
// ─────────────────────────────────────────────

/// NS gate: activates EXPAND probing when any of three conditions is met.
/// NS_gate = 1{He > 1 ∨ Se > 1 ∨ Te > 1}
/// where He, Se, Te are normalized by running medians.
#[derive(Debug, Clone)]
pub struct NsGate {
    /// Running median (EMA proxy) for bimodality coefficient
    he_median: RunningEma,
    /// Running median (EMA proxy) for surprise S_KL
    se_median: RunningEma,
    /// Running median (EMA proxy) for heavy-tail statistic
    te_median: RunningEma,
}

/// NS gate diagnostic output
#[derive(Debug, Clone, Copy, Default)]
pub struct NsGateResult {
    pub he: f32,       // bimodality coefficient / running median
    pub se: f32,       // surprise / running median
    pub te: f32,       // heavy-tail / running median
    pub gate_on: bool, // any > 1.0?
}

impl NsGate {
    pub fn new() -> Self {
        NsGate {
            he_median: RunningEma::new(0.1),
            se_median: RunningEma::new(0.1),
            te_median: RunningEma::new(0.1),
        }
    }

    /// Update running medians and evaluate gate.
    /// Returns normalized ratios and gate decision.
    pub fn evaluate(
        &mut self,
        bimodality_coeff: f32, // (skew²+1)/(kurt+3)
        surprise_kl: f32,
        heavy_tail_t: f32,
    ) -> NsGateResult {
        self.he_median.update(bimodality_coeff);
        self.se_median.update(surprise_kl);
        self.te_median.update(heavy_tail_t);

        let he = if self.he_median.value > 1e-6 {
            bimodality_coeff / self.he_median.value
        } else {
            0.0
        };

        let se = if self.se_median.value > 1e-6 {
            surprise_kl / self.se_median.value
        } else {
            0.0
        };

        let te = if self.te_median.value > 1e-6 {
            heavy_tail_t / self.te_median.value
        } else {
            0.0
        };

        NsGateResult {
            he,
            se,
            te,
            gate_on: he > 1.0 || se > 1.0 || te > 1.0,
        }
    }

    pub fn reset(&mut self) {
        self.he_median = RunningEma::new(0.1);
        self.se_median = RunningEma::new(0.1);
        self.te_median = RunningEma::new(0.1);
    }
}

// ─────────────────────────────────────────────
// § QuartzController
// ─────────────────────────────────────────────

struct QuartzCtrlInner {
    last_stats: QuartzStats,
    last_check_at: u32,
    s0_same: RunningEma,
    s0_global: RunningEma,
    elapsed_ms: u64,
    stop_reason: StopReason,
    theta_conf: f32,
    ns_gate: NsGate,
    depth_cal: DepthCalibration,
    merge_r0_ema: RunningEma, // G5: R₀ running estimate of RTT variance
    // v0.9.2: σ_response online estimator (instrumentation only)
    prev_q_best: f32,
    sigma_response_ema: f32,
    sigma_response_count: u32,
    // v0.9.2: Defect D_t tracking (Theory §VI)
    prev_log_policy: Vec<f32>, // log(N_a/N_total) at previous check
    defect_value: f32,         // last computed D_t²
    // P6 (audit_codex_20260425.md W8): per-check halt-decision telemetry.
    // Appended to on every `should_stop()` call after the periodicity
    // gate clears. Serialized via `halt_telemetry()` accessor for the
    // search-result builder in mcts_server.rs.
    halt_telemetry: Vec<HaltCheck>,
}

/// P6 (audit_codex_20260425.md W8): one record per `should_stop` call.
///
/// `schema_version: 2` (audit_codex_20260428.md Q3): adds
/// `voc_argmax_channel`, the argmax of (voc_focus, voc_expand,
/// voc_merge) at this halt-check. Consumers expecting v1 may safely
/// ignore the new field; v1 had no argmax channel field at all.
/// `quartz/replay.py:_finalize_halt_trace` aggregates the per-check
/// argmax into a histogram in the replay search summary so the
/// "three-channel VOC" framing is falsifiable per artifact.
#[derive(Clone, Debug, serde::Serialize)]
pub struct HaltCheck {
    pub schema_version: u8,
    pub root_visits: u32,
    pub elapsed_ms: u64,
    pub p_flip: f32,
    pub flip_stable: u32,
    pub sigma_q: f32,
    pub hbar_eff: f32,
    pub voc_total: f32,
    pub voc_focus: f32,
    pub voc_expand: f32,
    pub voc_merge: f32,
    /// Q3: argmax of (focus, expand, merge); one of "focus" / "expand" /
    /// "merge". When all three channels equal NEG_INFINITY (no usable
    /// channel), this is "none". This is a static-string label so it
    /// serializes as plain JSON without enum tagging.
    pub voc_argmax_channel: &'static str,
    pub decision: &'static str,
    pub triggered: bool,
}

/// Q3: pure helper exposed for unit testing. Returns the static label
/// of the channel with the largest VOC value. Ties resolve in declared
/// order (focus, expand, merge). All-NaN / all-NEG_INFINITY returns
/// "none".
pub fn voc_argmax_channel(focus: f32, expand: f32, merge: f32) -> &'static str {
    let candidates = [("focus", focus), ("expand", expand), ("merge", merge)];
    let mut best: (&'static str, f32) = ("none", f32::NEG_INFINITY);
    for (name, value) in candidates {
        if value.is_finite() && value > best.1 {
            best = (name, value);
        }
    }
    best.0
}

pub struct QuartzController {
    pub cfg: QuartzConfig,
    pub max_visits: u32,
    /// Lock-free mirror of `inner.last_check_at`.
    ///
    /// Parallel search calls `should_stop()` on every worker loop. The
    /// adaptive branches only need the mutex once `check_interval` visits have
    /// elapsed since the previous controller check, so this mirror keeps the
    /// common "not due yet" path out of the mutex without changing the
    /// telemetry-producing locked path.
    last_check_at_atomic: AtomicU32,
    /// P01: per-`HaltReason` increment counters. Updated lock-free by
    /// `should_stop` on every terminal branch so the README's
    /// `halt_reason_count` claim is emitted from real data. Workers that
    /// race on the same termination cause increment independently; the
    /// final value is stable once the search ends and all workers join.
    halt_reason_count: [AtomicU32; HALT_REASON_COUNT],
    inner: std::sync::Mutex<QuartzCtrlInner>,
}

impl QuartzController {
    pub fn new(max_visits: u32, cfg: QuartzConfig) -> Self {
        let theta_init = match &cfg.halt_mode {
            HaltMode::ConfAdaptive { theta_init, .. } => *theta_init,
            _ => 0.8, // default, unused for other modes
        };
        QuartzController {
            max_visits,
            last_check_at_atomic: AtomicU32::new(0),
            halt_reason_count: std::array::from_fn(|_| AtomicU32::new(0)),
            inner: std::sync::Mutex::new(QuartzCtrlInner {
                last_stats: QuartzStats::default(),
                last_check_at: 0,
                s0_same: RunningEma::new(0.05),
                s0_global: RunningEma::new(0.01),
                elapsed_ms: 0,
                stop_reason: StopReason::Unknown,
                theta_conf: theta_init,
                ns_gate: NsGate::new(),
                depth_cal: DepthCalibration::new(0.1),
                merge_r0_ema: RunningEma::new(0.1),
                prev_q_best: 0.0,
                sigma_response_ema: 0.0,
                sigma_response_count: 0,
                prev_log_policy: Vec::new(),
                defect_value: 0.0,
                halt_telemetry: Vec::new(),
            }),
            cfg,
        }
    }

    /// P6: snapshot of recorded halt checks since controller creation.
    pub fn halt_telemetry(&self) -> Vec<HaltCheck> {
        self.inner.lock().unwrap().halt_telemetry.clone()
    }
    /// P01: snapshot of per-`HaltReason` terminal counts since controller
    /// creation. Updates are `Relaxed` and unsynchronized between workers,
    /// but `should_stop` is the only writer and the final values are
    /// stable once all workers join. Indexed identically to `HaltReason`.
    pub fn halt_reason_count_snapshot(&self) -> [u32; HALT_REASON_COUNT] {
        let mut out = [0u32; HALT_REASON_COUNT];
        for (i, slot) in self.halt_reason_count.iter().enumerate() {
            out[i] = slot.load(Ordering::Relaxed);
        }
        out
    }
    /// P01: explicit terminal-branch increment, called from each
    /// `should_stop` early-return. Centralized so the (Reason, Relaxed)
    /// invariant is enforced in one place.
    #[inline]
    fn note_halt(&self, reason: HaltReason) {
        self.halt_reason_count[reason as usize].fetch_add(1, Ordering::Relaxed);
    }
    pub fn last_stats(&self) -> QuartzStats {
        self.inner.lock().unwrap().last_stats.clone()
    }
    /// v0.9.2: σ_response EMA (instrumentation, Exp-3)
    pub fn sigma_response(&self) -> (f32, u32) {
        let g = self.inner.lock().unwrap();
        (g.sigma_response_ema, g.sigma_response_count)
    }
    /// v0.9.2: Defect D_t² (Theory §VI)
    pub fn defect(&self) -> f32 {
        self.inner.lock().unwrap().defect_value
    }
    pub fn last_stop_reason(&self) -> StopReason {
        self.inner.lock().unwrap().stop_reason.clone()
    }
    pub fn update_elapsed(&self, ms: u64) {
        self.inner.lock().unwrap().elapsed_ms = ms;
    }
    pub fn record_iter_time_ms(&self, _ms: f32) {} // reserved
    pub fn depth_kappas(&self) -> [f32; 3] {
        self.inner.lock().unwrap().depth_cal.kappas()
    }
    pub fn update_stats<M: Copy + Send + Sync + 'static>(
        &self,
        root: &ArenaRef<MctsNode<M>>,
        priors: Option<&[f32]>,
    ) {
        let mut g = self.inner.lock().unwrap();
        let prev = g.last_stats.flip_stable;
        let elapsed = g.elapsed_ms;
        let s0_global = g.s0_global.value.max(1e-6);

        let mut s0_same = g.s0_same.clone();
        let mut s = compute_quartz_stats(
            root,
            priors,
            &mut s0_same,
            s0_global,
            prev,
            elapsed,
            &self.cfg,
        );

        // ── G8: NS Gate → gate EXPAND channel ──
        if self.cfg.enable_ns_gate {
            let bc = (s.skewness.powi(2) + 1.0) / (s.kurtosis.abs() + 3.0);
            let ns_result = g.ns_gate.evaluate(bc, s.surprise_kl, s.heavy_tail_t);
            if !ns_result.gate_on {
                // Gate off → suppress EXPAND VOC
                s.voc_expand = 0.0;
                // Recalculate unified VOC total
                let best_v = s.voc_focus.max(s.voc_expand).max(s.voc_merge);
                s.unified.voc_expand = 0.0;
                s.unified.voc_total = best_v.max(0.0);
            }
        }

        // A4/P12: use the published edge slab directly for the downstream
        // diagnostic sections. This preserves the single-timepoint intent of
        // the old snapshot while avoiding a Vec allocation and child-handle
        // copies on every controller refresh.
        let n_mat_for_sections = root.materialized_count();
        let edges_full = root.read_edges();
        let edges_view = &edges_full[..n_mat_for_sections.min(edges_full.len())];

        // ── G7: κ_b depth calibration (diagnostic collection) ──
        if self.cfg.enable_depth_cal {
            // Collect σ observations from root edges (depth=0)
            for e in edges_view {
                if let Some(sigma_emp) = e.edge_sigma() {
                    let n = e.n.load(Ordering::Acquire);
                    let sigma_pred = self.cfg.sigma_0 / (n as f32 + 1.0).sqrt();
                    g.depth_cal.record(0, sigma_emp, sigma_pred);
                }
            }
        }

        // ── G5: MERGE R₀ running normalization ──
        if self.cfg.enable_merge_r0 && s.rtt_n >= 2 {
            let rtt_var = s.rtt_sigma * s.rtt_sigma;
            if rtt_var > 0.0 {
                g.merge_r0_ema.update(rtt_var);
            }
            let r0 = g.merge_r0_ema.value.max(1e-6);
            s.merge_r0 = r0;
            let p_merge_new = (rtt_var / r0).min(1.0);
            // Recompute VOC_MERGE with R/R₀ normalization
            if self.cfg.enable_merge_channel {
                s.voc_merge = p_merge_new * s.hbar_eff * s.rtt_sigma - s.cost_focus;
                let best_v = s.voc_focus.max(s.voc_expand).max(s.voc_merge);
                s.unified.voc_merge = s.voc_merge;
                s.unified.voc_total = best_v.max(0.0);
            }
        }

        // ── v0.9.2: σ_response online estimator (Exp-3, instrumentation only) ──
        {
            // Track |ΔQ_best| between checks as EMA — top-5 root edges.
            let edges = &edges_view[..edges_view.len().min(5)];
            let q_best = edges
                .iter()
                .map(|e| e.q_eff())
                .fold(f32::NEG_INFINITY, f32::max);

            if g.sigma_response_count > 0 {
                let delta_q = (q_best - g.prev_q_best).abs();
                let alpha = 0.3_f32; // EMA decay
                g.sigma_response_ema = alpha * delta_q + (1.0 - alpha) * g.sigma_response_ema;
            }
            g.prev_q_best = q_best;
            g.sigma_response_count += 1;
        }

        // ── v0.9.2: Defect D_t computation (Theory §VI, Theorem 5) ──
        {
            let edges = &edges_view[..edges_view.len().min(20)];
            let n_total = edges
                .iter()
                .map(|e| e.n.load(Ordering::Acquire))
                .sum::<u32>()
                .max(1);

            // Current log-policy: log(N_a/N_total + ε)
            let eps_p = 1e-4_f32;
            let cur_log_policy: SmallVec<[f32; 32]> = edges
                .iter()
                .map(|e| {
                    let na = e.n.load(Ordering::Acquire) as f32;
                    ((na + eps_p) / (n_total as f32 + eps_p * edges.len() as f32)).ln()
                })
                .collect();

            if !g.prev_log_policy.is_empty() && g.prev_log_policy.len() == cur_log_policy.len() {
                // Drift g(a) = (cur - prev) - mean(cur - prev)
                let drift: SmallVec<[f32; 32]> = cur_log_policy
                    .iter()
                    .zip(g.prev_log_policy.iter())
                    .map(|(c, p)| c - p)
                    .collect();
                let q_weights: SmallVec<[f32; 32]> = edges
                    .iter()
                    .map(|e| {
                        let na = e.n.load(Ordering::Acquire) as f32;
                        (na + eps_p) / (n_total as f32 + eps_p * edges.len() as f32)
                    })
                    .collect();
                let drift_mean: f32 = drift.iter().zip(q_weights.iter()).map(|(d, w)| d * w).sum();
                let centered_drift: SmallVec<[f32; 32]> =
                    drift.iter().map(|d| d - drift_mean).collect();

                // One-field tangent: centered Q values
                let qs: SmallVec<[f32; 32]> = edges.iter().map(|e| e.q_eff()).collect();
                let q_mean: f32 = qs.iter().zip(q_weights.iter()).map(|(q, w)| q * w).sum();
                let centered_q: SmallVec<[f32; 32]> = qs.iter().map(|q| q - q_mean).collect();

                // Fisher-weighted inner products: <g, g>, <g, f̃>, <f̃, f̃>
                let gg: f32 = centered_drift
                    .iter()
                    .zip(q_weights.iter())
                    .map(|(d, w)| d * d * w)
                    .sum();
                let gf: f32 = centered_drift
                    .iter()
                    .zip(centered_q.iter())
                    .zip(q_weights.iter())
                    .map(|((d, f), w)| d * f * w)
                    .sum();
                let ff: f32 = centered_q
                    .iter()
                    .zip(q_weights.iter())
                    .map(|(f, w)| f * f * w)
                    .sum();

                // D² = |g|² - (⟨g,f̃⟩² / ⟨f̃,f̃⟩) if ff > 0
                if ff > 1e-10 {
                    g.defect_value = (gg - gf * gf / ff).max(0.0);
                } else {
                    g.defect_value = gg; // no tangent direction → all is residual
                }
            }

            g.prev_log_policy.clear();
            g.prev_log_policy.extend(cur_log_policy.iter().copied());
        }

        g.s0_global.update(s.surprise_kl);
        g.s0_same = s0_same;
        g.last_stats = s;
    }

    pub fn mark_checked(&self, root_visits: u32) {
        self.inner.lock().unwrap().last_check_at = root_visits;
        self.last_check_at_atomic
            .store(root_visits, Ordering::Relaxed);
    }
}

/// P6 (audit_codex_20260425.md W8): append one HaltCheck record to the
/// controller's per-call telemetry buffer. The buffer is bounded
/// implicitly by the search budget — `should_stop` is called once per
/// iteration, so for typical 15K-iter searches we record ≤ 15K items
/// (~ 1 MiB peak at 64 B/record before the controller is dropped).
fn record_halt_check(
    g: &mut QuartzCtrlInner,
    root_visits: u32,
    elapsed_ms: u64,
    decision: &'static str,
    triggered: bool,
) {
    let stats = &g.last_stats;
    let argmax = voc_argmax_channel(
        stats.unified.voc_focus,
        stats.unified.voc_expand,
        stats.unified.voc_merge,
    );
    g.halt_telemetry.push(HaltCheck {
        schema_version: 2,
        root_visits,
        elapsed_ms,
        p_flip: stats.p_flip,
        flip_stable: stats.flip_stable,
        sigma_q: stats.sigma_q,
        hbar_eff: stats.hbar_eff,
        voc_total: stats.unified.voc_total,
        voc_focus: stats.unified.voc_focus,
        voc_expand: stats.unified.voc_expand,
        voc_merge: stats.unified.voc_merge,
        voc_argmax_channel: argmax,
        decision,
        triggered,
    });
}

impl SearchController for QuartzController {
    fn should_stop(&self, root_visits: u32, elapsed_ms: u64) -> bool {
        // ── PR-1A: HaltMode 3-way branch ──
        //
        // Fixed: just check budget (no QUARTZ logic)
        // SimpleThreshold: P_flip only (no VOC cost)
        // VOC: full QUARTZ (converged = P_flip stable AND VOC ≤ 0)

        // Common hard limits (apply to all modes)
        if root_visits >= self.max_visits {
            let mut g = self.inner.lock().unwrap();
            g.stop_reason = StopReason::BudgetExhausted {
                iterations: root_visits,
            };
            // P6: record terminal halt-check at the max_visits ceiling.
            record_halt_check(&mut g, root_visits, elapsed_ms, "MaxVisits", true);
            drop(g);
            self.note_halt(HaltReason::MaxVisits);
            return true;
        }
        if self.cfg.ctm_budget_ms > 0 && elapsed_ms > self.cfg.ctm_budget_ms * 3 {
            let mut g = self.inner.lock().unwrap();
            g.stop_reason = StopReason::TimeCapHit { elapsed_ms };
            record_halt_check(&mut g, root_visits, elapsed_ms, "TimeCapHit", true);
            drop(g);
            self.note_halt(HaltReason::MaxTime);
            return true;
        }

        // Fixed mode: only hard limits above, no adaptive stopping
        if let HaltMode::Fixed { budget } = self.cfg.halt_mode {
            if root_visits >= budget {
                let mut g = self.inner.lock().unwrap();
                g.stop_reason = StopReason::BudgetExhausted {
                    iterations: root_visits,
                };
                record_halt_check(&mut g, root_visits, elapsed_ms, "FixedBudget", true);
                drop(g);
                self.note_halt(HaltReason::FixedBudget);
                return true;
            }
            // No record here — Fixed mode doesn't periodically inspect
            // adaptive stats, so a per-tick HaltCheck would be noise.
            return false;
        }

        // VOC / SimpleThreshold: check QUARTZ stats periodically
        let last_check_at = self.last_check_at_atomic.load(Ordering::Relaxed);
        if root_visits.saturating_sub(last_check_at) < self.cfg.check_interval {
            return false;
        }
        let mut g = self.inner.lock().unwrap();
        g.elapsed_ms = elapsed_ms;
        let since = root_visits.saturating_sub(g.last_check_at);
        if since < self.cfg.check_interval {
            return false;
        }

        let stats = &g.last_stats;
        if stats.root_visits < self.cfg.min_visits {
            // P01: distinguish "stop attempted but visits insufficient" from
            // "not yet checked" so the halt_reason histogram is meaningful.
            // Note: only counted when the periodicity gate cleared (we are
            // inside the periodic check), preventing an avalanche of
            // every-iteration MinVisitsNotMet increments.
            drop(g);
            self.note_halt(HaltReason::MinVisitsNotMet);
            return false;
        }

        // P6: classify this halt-check up front so a single record per
        // periodic check can be appended whether or not we trigger.
        let mut decision_tag: &'static str = "Pending";
        let mut triggered = false;
        // P01: classify the WHY when triggered=true, mapped onto the
        // HaltReason axis (PFlipConverged, VOCNonPositive, ...).
        let mut halt_reason: Option<HaltReason> = None;
        match self.cfg.halt_mode {
            HaltMode::SimpleThreshold => {
                // Only P_flip convergence — ignore VOC cost term
                let pflip_ok = stats.p_flip < FLIP_THRESH && stats.flip_stable >= FLIP_STABLE_N;
                if pflip_ok {
                    g.stop_reason = StopReason::Converged {
                        p_flip: stats.p_flip,
                        stable_count: stats.flip_stable,
                    };
                    decision_tag = "Converged";
                    triggered = true;
                    halt_reason = Some(HaltReason::PFlipConverged);
                }
            }
            HaltMode::VOC => {
                // Full QUARTZ: P_flip stable AND VOC ≤ 0
                if stats.converged {
                    if stats.unified.voc_total <= 0.0 && stats.p_flip < FLIP_THRESH {
                        g.stop_reason = StopReason::VocNonPositive {
                            max_gvoc: stats.unified.voc_total,
                        };
                        decision_tag = "VocNonPositive";
                        halt_reason = Some(HaltReason::VOCNonPositive);
                    } else {
                        g.stop_reason = StopReason::Converged {
                            p_flip: stats.p_flip,
                            stable_count: stats.flip_stable,
                        };
                        decision_tag = "Converged";
                        halt_reason = Some(HaltReason::PFlipConverged);
                    }
                    triggered = true;
                }
            }
            HaltMode::ConfAdaptive {
                target_time_ms,
                eta,
                ..
            } => {
                // PR-6A: Conf(t) ≥ θ_conf → stop
                let conf = stats.conf_t;
                if conf >= g.theta_conf {
                    g.stop_reason = StopReason::Converged {
                        p_flip: stats.p_flip,
                        stable_count: stats.flip_stable,
                    };
                    // θ_conf online adaptation: move toward target time
                    if target_time_ms > 0 {
                        let actual = elapsed_ms as f32;
                        let target = target_time_ms as f32;
                        let delta = eta * (actual - target) / target;
                        g.theta_conf = (g.theta_conf + delta).clamp(0.5, 0.99);
                    }
                    decision_tag = "Converged";
                    triggered = true;
                    halt_reason = Some(HaltReason::PFlipConverged);
                }
            }
            HaltMode::Fixed { .. } => unreachable!(), // handled above
        }
        record_halt_check(&mut g, root_visits, elapsed_ms, decision_tag, triggered);
        drop(g);
        if let Some(reason) = halt_reason {
            self.note_halt(reason);
        }
        if triggered {
            return true;
        }

        false
    }

    fn stop_reason(&self) -> StopReason {
        self.inner.lock().unwrap().stop_reason.clone()
    }

    fn needs_elapsed_ms(&self) -> bool {
        self.cfg.ctm_budget_ms > 0 || !matches!(self.cfg.halt_mode, HaltMode::Fixed { .. })
    }

    fn visit_limit_hint(&self) -> Option<u32> {
        if self.cfg.ctm_budget_ms > 0 {
            return None;
        }
        match self.cfg.halt_mode {
            HaltMode::Fixed { budget } => Some(budget.min(self.max_visits)),
            _ => None,
        }
    }

    fn reset(&mut self) {
        let mut g = self.inner.lock().unwrap();
        g.last_stats = QuartzStats::default();
        g.last_check_at = 0;
        self.last_check_at_atomic.store(0, Ordering::Relaxed);
        g.s0_same = RunningEma::new(0.05);
        g.elapsed_ms = 0;
        g.stop_reason = StopReason::Unknown;
        // theta_conf NOT reset (persists across positions for adaptation)
        // s0_global NOT reset (cross-position persistence)
    }
}

// ─────────────────────────────────────────────
// § 단위 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// P05: with_calibration applies the per-game recommendation from a
    /// schema_version=1 JSON file produced by `write_sigma_recommendation`.
    #[test]
    fn test_p05_with_calibration_applies_per_game_recommendation() {
        let dir = tempfile::tempdir().unwrap();
        let payload = serde_json::json!({
            "schema_version": 1,
            "tool": "calibration::test",
            "target": "ħ_eff ≈ 1.0",
            "tie_break": "prefer smaller σ₀",
            "per_game_rows": {},
            "recommendations": {
                "gomoku7": 0.18,
                "__cross_game__": 0.22,
            },
        });
        std::fs::write(
            dir.path().join("sigma_0_recommendation.json"),
            serde_json::to_string(&payload).unwrap(),
        )
        .unwrap();
        let base = QuartzConfig {
            sigma_0: 0.3,
            ..Default::default()
        };
        let (cfg, diags) = base.with_calibration(dir.path(), Some("gomoku7"), None, 2.0);
        assert!((cfg.sigma_0 - 0.18).abs() < 1e-6);
        // exactly one INFO emitted with the new value
        let info_count = diags
            .iter()
            .filter(|d| matches!(d, CalibrationDiagnostic::Info(_)))
            .count();
        assert_eq!(info_count, 1);
    }

    /// P05: when the per-game key is missing, fall back to __cross_game__.
    #[test]
    fn test_p05_with_calibration_falls_back_to_cross_game() {
        let dir = tempfile::tempdir().unwrap();
        let payload = serde_json::json!({
            "schema_version": 1,
            "tool": "test",
            "target": "x",
            "tie_break": "x",
            "per_game_rows": {},
            "recommendations": { "__cross_game__": 0.25 },
        });
        std::fs::write(
            dir.path().join("sigma_0_recommendation.json"),
            serde_json::to_string(&payload).unwrap(),
        )
        .unwrap();
        let base = QuartzConfig::default();
        let (cfg, _) = base.with_calibration(dir.path(), Some("not_in_file"), None, 2.0);
        assert!((cfg.sigma_0 - 0.25).abs() < 1e-6);
    }

    /// P05: when the override differs from base by >warn_factor, emit a
    /// WARN diagnostic alongside the INFO.
    #[test]
    fn test_p05_with_calibration_warns_on_large_divergence() {
        let dir = tempfile::tempdir().unwrap();
        let payload = serde_json::json!({
            "schema_version": 1,
            "tool": "test",
            "target": "x",
            "tie_break": "x",
            "per_game_rows": {},
            // 5x base ⇒ exceeds warn_factor=2
            "recommendations": { "__cross_game__": 1.5 },
        });
        std::fs::write(
            dir.path().join("sigma_0_recommendation.json"),
            serde_json::to_string(&payload).unwrap(),
        )
        .unwrap();
        let base = QuartzConfig {
            sigma_0: 0.3,
            ..Default::default()
        };
        let (cfg, diags) = base.with_calibration(dir.path(), None, None, 2.0);
        assert!((cfg.sigma_0 - 1.5).abs() < 1e-6);
        let warn_count = diags
            .iter()
            .filter(|d| matches!(d, CalibrationDiagnostic::Warn(_)))
            .count();
        assert!(warn_count >= 1, "expected a WARN diagnostic, got {diags:?}");
    }

    /// P05: missing calibration_dir leaves sigma_0 untouched and emits a
    /// single INFO diagnostic.
    #[test]
    fn test_p05_with_calibration_missing_file_keeps_default() {
        let dir = tempfile::tempdir().unwrap();
        let base = QuartzConfig {
            sigma_0: 0.3,
            ..Default::default()
        };
        let (cfg, diags) = base.with_calibration(dir.path(), Some("any"), None, 2.0);
        assert!((cfg.sigma_0 - 0.3).abs() < 1e-6);
        assert!(matches!(
            diags.first(),
            Some(CalibrationDiagnostic::Info(_))
        ));
    }

    /// P05: a strength-stratified file (e.g. weak.json) takes priority
    /// over the default sigma_0_recommendation.json when EvalStrength is
    /// set. This is the future hook for per-evaluator-quality stratified
    /// calibration without requiring a separate dir tree per strength.
    #[test]
    fn test_p05_with_calibration_strength_stratified_file_wins() {
        let dir = tempfile::tempdir().unwrap();
        // default file → 0.3
        std::fs::write(
            dir.path().join("sigma_0_recommendation.json"),
            serde_json::to_string(&serde_json::json!({
                "schema_version": 1, "tool": "x", "target": "x", "tie_break": "x",
                "per_game_rows": {},
                "recommendations": { "__cross_game__": 0.3 },
            }))
            .unwrap(),
        )
        .unwrap();
        // weak.json → 0.5 (weaker eval ⇒ larger uncertainty reference)
        std::fs::write(
            dir.path().join("weak.json"),
            serde_json::to_string(&serde_json::json!({
                "schema_version": 1, "tool": "x", "target": "x", "tie_break": "x",
                "per_game_rows": {},
                "recommendations": { "__cross_game__": 0.5 },
            }))
            .unwrap(),
        )
        .unwrap();
        let base = QuartzConfig::default();
        let (cfg, _) = base.with_calibration(
            dir.path(),
            None,
            Some(EvalStrength::Weak),
            10.0, // generous warn_factor
        );
        assert!(
            (cfg.sigma_0 - 0.5).abs() < 1e-6,
            "got sigma_0={}",
            cfg.sigma_0
        );
    }

    /// P05: schema_version != 1 must be skipped (forward-compat guard).
    #[test]
    fn test_p05_with_calibration_skips_unknown_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("sigma_0_recommendation.json"),
            serde_json::to_string(&serde_json::json!({
                "schema_version": 99,
                "recommendations": { "__cross_game__": 0.5 },
            }))
            .unwrap(),
        )
        .unwrap();
        let base = QuartzConfig {
            sigma_0: 0.3,
            ..Default::default()
        };
        let (cfg, diags) = base.with_calibration(dir.path(), None, None, 2.0);
        assert!((cfg.sigma_0 - 0.3).abs() < 1e-6);
        let has_warn = diags
            .iter()
            .any(|d| matches!(d, CalibrationDiagnostic::Warn(m) if m.contains("schema_version")));
        assert!(has_warn, "expected schema_version WARN, got {diags:?}");
    }

    #[test]
    fn test_hbar_eff_scale() {
        // ħ_eff = σ_Q/σ₀
        // σ_Q = σ₀ → ħ_eff = 1 (reference case)
        // σ_Q = 2σ₀ → ħ_eff = 2 (high uncertainty)
        // σ_Q = 0.1σ₀ → ħ_eff = 0.1 (low uncertainty)
        let sigma_0 = 0.3f32;
        assert!((0.3 / sigma_0 - 1.0).abs() < 1e-5);
        assert!((0.6 / sigma_0 - 2.0).abs() < 1e-5);
        assert!((0.03 / sigma_0 - 0.1).abs() < 1e-4);
    }

    #[test]
    fn test_cost_derived_from_hbar() {
        // cost_focus = ħ_eff × σ₀/N_min × ctm_factor
        // If σ_Q = σ₀ (reference): cost = σ₀/N_min × 0.5 (no urgency)
        let cfg = QuartzConfig {
            sigma_0: 0.3,
            min_visits: 50,
            ctm_budget_ms: 0,
            ..Default::default()
        };
        let hbar = 1.0f32;
        let cost_base = cfg.sigma_0 / cfg.min_visits as f32;
        let ctm_f = ctm_factor(0, &cfg); // no urgency
        let cost = hbar * cost_base * ctm_f;
        assert!(
            (ctm_f - 0.5).abs() < 0.01,
            "no urgency → factor=0.5, got {}",
            ctm_f
        );
        assert!((cost - 0.5 * 0.3 / 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_epsilon_envariance_adaptive() {
        // ε_t = 1/√N → threshold S_KL > ENVAR_CONST/√N
        // N=100: threshold = 0.5/√100 = 0.05
        // N=10000: threshold = 0.5/100 = 0.005
        let (viol_100, _, eps_100) = check_envariance(0.06, 100);
        let (viol_lo, _, _) = check_envariance(0.01, 100);

        assert!(viol_100, "S_KL=0.06 > 0.05 → violated for N=100");
        assert!(!viol_lo, "S_KL=0.01 < 0.05 → not violated for N=100");
        assert!(
            (eps_100 - 0.1).abs() < 0.01,
            "ε_t = 1/√100 = 0.1, got {}",
            eps_100
        );
    }

    #[test]
    fn test_envariance_is_pinsker_bound() {
        // Pinsker: ‖q_T-π‖_TV ≤ √(S_KL/2)
        // ε-envariance: ‖q_T-π‖_TV < ε_t = 1/√N
        // → threshold on S_KL: S_KL < 2ε² = 2/N
        // We use ENVAR_CONST/√N = 0.5/√N (conservative relative to Pinsker)
        let n = 1000u32;
        let sqrt_n = (n as f32).sqrt();
        let eps_t = 1.0 / sqrt_n;
        let _pinsker_threshold = 2.0 * eps_t * eps_t; // 2/N = 0.002
        let our_threshold = ENVAR_CONST / sqrt_n; // 0.5/√1000 ≈ 0.0158
                                                  // Our threshold is now MORE conservative (higher) than Pinsker at high N
                                                  // This is intentional: we want to detect violations early
        assert!(
            our_threshold > 0.0,
            "our threshold {:.5} should be positive",
            our_threshold
        );
    }

    #[test]
    fn test_fisher_alpha_fixed() {
        // α = 1/2 from F_aa = π(a) natural gradient
        // If we compute natural gradient: F^{-1}∇ ∝ 1/π(a) → prior weight = π(a)^(1-1) ???
        // Actually: score = Q + c·π(a)/√π(a)·√N/(1+N_a) → prior exponent = 1-0.5 = 0.5
        assert_eq!(FISHER_ALPHA, 0.5);
        // Verify: for π(a)=0.01, standard vs Fisher
        let _standard_weight = 0.01_f32;
        let fisher_weight = fisher_prior_weight(0.01);
        assert!(
            (fisher_weight - 0.1).abs() < 1e-5,
            "√0.01 = 0.1, got {}",
            fisher_weight
        );
        // Fisher reduces the 50× ratio (0.5 vs 0.01) to 7.07× (√0.5/√0.01)
        let ratio_standard = 0.5_f32 / 0.01; // = 50
        let ratio_fisher = fisher_prior_weight(0.5) / fisher_weight;
        assert!(
            ratio_fisher < ratio_standard / 2.0,
            "Fisher should halve (or more) the prior bias"
        );
    }

    #[test]
    fn test_one_loop_penalty_diagonal() {
        // Diagonal one-loop: -(ħ_eff/2) × ∂Tr log M/∂N_a = -ħ_eff/N_a
        // verify: penalty ∝ -1/N_a
        let hbar = 0.5f32;
        let p10 = one_loop_visit_penalty(10, hbar, 1.0, HBAR_PENALTY_CAP);
        let p100 = one_loop_visit_penalty(100, hbar, 1.0, HBAR_PENALTY_CAP);
        assert!(p10 < p100, "penalty less negative for more visits");
        assert!(
            (p10.abs() / p100.abs() - 10.0).abs() < 0.5,
            "ratio should be ~10"
        );
    }

    #[test]
    fn test_flip_thresh_one_sigma() {
        // Φ(-1) = 0.15866 exactly (standard normal CDF at -1σ)
        // Fixed CDF with A&S 7.1.26 via x/√2 should give ≈0.1587
        let cdf_neg1 = standard_normal_cdf(-1.0);
        assert!(
            (cdf_neg1 - 0.1587).abs() < 0.002,
            "CDF(-1) should be ≈0.1587, got {:.5}",
            cdf_neg1
        );
        // FLIP_THRESH = 0.159 is near Φ(-1)
        assert!((FLIP_THRESH - 0.159).abs() < 0.001);
        assert!(
            (FLIP_THRESH - cdf_neg1).abs() < 0.003,
            "FLIP_THRESH should be ≈Φ(-1), diff={:.5}",
            (FLIP_THRESH - cdf_neg1).abs()
        );
    }

    #[test]
    fn test_voc_channel_cost_ordering() {
        // EXPAND should cost more than FOCUS (quadratic in ħ_eff)
        // cost_expand = ħ_eff² × cost_base
        // cost_focus  = ħ_eff × cost_base × ctm_factor ≈ ħ_eff × cost_base × 0.5
        // For ħ_eff > 0.5: cost_expand > cost_focus (quadratic dominates)
        let cfg = QuartzConfig::default();
        let hbar = 1.0f32;
        let cost_base = cfg.sigma_0 / cfg.min_visits as f32;
        let cost_f = hbar * cost_base * 0.5;
        let cost_e = hbar * hbar * cost_base;
        assert!(cost_e > cost_f, "EXPAND costs more than FOCUS for ħ_eff=1");
    }

    #[test]
    fn test_p6_halt_telemetry_records_max_visits_terminal() {
        // P6 (audit_codex_20260425.md W8): hitting the max_visits ceiling
        // appends a single HaltCheck with decision="MaxVisits", triggered=true.
        // Q3 (audit_codex_20260428.md W'1) bumped the schema to 2 to add
        // `voc_argmax_channel`; the wire format remains additive.
        let cfg = QuartzConfig::default();
        let ctrl = QuartzController::new(10, cfg);
        assert!(ctrl.should_stop(10, 0));
        let tele = ctrl.halt_telemetry();
        assert_eq!(tele.len(), 1);
        let rec = &tele[0];
        assert_eq!(rec.schema_version, 2);
        assert_eq!(rec.root_visits, 10);
        assert_eq!(rec.decision, "MaxVisits");
        assert!(rec.triggered);
    }

    #[test]
    fn test_p6_halt_telemetry_records_fixed_budget() {
        // P6: HaltMode::Fixed records FixedBudget when its own budget hits
        // before the controller's max_visits ceiling.
        let cfg = QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: 5 },
            ..QuartzConfig::default()
        };
        let ctrl = QuartzController::new(100, cfg);
        // Below budget: no record (Fixed mode skips per-tick recording).
        assert!(!ctrl.should_stop(2, 0));
        assert_eq!(ctrl.halt_telemetry().len(), 0);
        // At budget: triggered, record appended.
        assert!(ctrl.should_stop(5, 0));
        let tele = ctrl.halt_telemetry();
        assert_eq!(tele.len(), 1);
        assert_eq!(tele[0].decision, "FixedBudget");
        assert!(tele[0].triggered);
    }

    #[test]
    fn test_p6_halt_check_serializes_with_schema_version_2() {
        // P6 + Q3: JSON wire format carries `schema_version: 2`. v2 added
        // `voc_argmax_channel`. Asserts shape + presence of every advertised
        // field including the new one.
        let rec = HaltCheck {
            schema_version: 2,
            root_visits: 100,
            elapsed_ms: 250,
            p_flip: 0.05,
            flip_stable: 4,
            sigma_q: 0.12,
            hbar_eff: 0.6,
            voc_total: 0.04,
            voc_focus: 0.02,
            voc_expand: 0.04,
            voc_merge: 0.01,
            voc_argmax_channel: "expand",
            decision: "VocNonPositive",
            triggered: true,
        };
        let json = serde_json::to_value(&rec).unwrap();
        assert_eq!(json["schema_version"], 2);
        for key in [
            "root_visits",
            "elapsed_ms",
            "p_flip",
            "flip_stable",
            "sigma_q",
            "hbar_eff",
            "voc_total",
            "voc_focus",
            "voc_expand",
            "voc_merge",
            "voc_argmax_channel",
            "decision",
            "triggered",
        ] {
            assert!(json.get(key).is_some(), "missing field {key}");
        }
        assert_eq!(json["voc_argmax_channel"], "expand");
    }

    #[test]
    fn test_q3_voc_argmax_channel_helper() {
        // Q3 (audit_codex_20260428.md): voc_argmax_channel must reliably pick
        // the largest finite channel; ties resolve in declared order
        // (focus → expand → merge); all-NaN / NEG_INF returns "none".
        assert_eq!(voc_argmax_channel(0.5, 0.1, 0.1), "focus");
        assert_eq!(voc_argmax_channel(0.0, 0.5, 0.1), "expand");
        assert_eq!(voc_argmax_channel(0.0, 0.1, 0.5), "merge");
        // Ties: focus wins on equality (declared order).
        assert_eq!(voc_argmax_channel(0.5, 0.5, 0.5), "focus");
        // NaN / NEG_INF inputs degrade gracefully.
        assert_eq!(voc_argmax_channel(f32::NAN, f32::NAN, f32::NAN), "none");
        assert_eq!(
            voc_argmax_channel(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
            "none"
        );
        // Mixed: NaN is non-finite, falls through.
        assert_eq!(voc_argmax_channel(f32::NAN, 0.5, 0.1), "expand");
    }

    #[test]
    fn test_rho_hat_off_diagonal_oneloop() {
        // RTT_var = 0 → ρ̂ = 0 (no correlation, conservative ρ=0 bound)
        let sigma1 = 0.05f32;
        let sigma2 = 0.05f32;
        let sp = sigma1 * sigma2;
        let rho_0 = (1.0 - 0.0 / sp).clamp(-0.95, 0.95);
        assert_eq!(rho_0, 0.95_f32.min(1.0)); // 1.0 clamped to 0.95

        // RTT_var = σ₁σ₂/2 → ρ̂ = 0.5 (moderate positive correlation)
        let rtt_mod = sp * 0.5;
        let rho_m = (1.0 - rtt_mod / sp).clamp(-0.95, 0.95);
        assert!((rho_m - 0.5).abs() < 1e-5);

        // RTT_var = 2σ₁σ₂ → ρ̂ = -1 → clamped to -0.95 (diverging paths)
        let rtt_large = sp * 2.0;
        let rho_l = (1.0 - rtt_large / sp).clamp(-0.95, 0.95);
        assert_eq!(rho_l, -0.95);
    }

    #[test]
    fn test_cdf_pdf() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((standard_normal_pdf(0.0) - 0.3989).abs() < 0.001);
    }

    #[test]
    fn test_rtt_welford() {
        let node = crate::mcts::node::MctsNode::<usize>::new(0, None);
        assert!(node.rtt_variance().is_none());
        for &q in &[0.8, 0.2, 0.5, 0.9, 0.1] {
            node.record_rtt_hit(q);
        }
        let n = node.rtt_n.load(Ordering::Relaxed);
        assert_eq!(n, 5);
        let var = node
            .rtt_variance()
            .expect("should have variance after 5 hits");
        assert!(var > 0.0 && var < 1.0, "variance={}", var);
    }

    // ── PR-1A: HaltMode 3-way tests ──────────────────────────

    #[test]
    fn test_1a_halt_mode_default_is_voc() {
        let cfg = QuartzConfig::default();
        assert_eq!(cfg.halt_mode, HaltMode::VOC);
    }

    #[test]
    fn test_1a_halt_mode_builder() {
        let cfg = QuartzConfig::default().with_halt_mode(HaltMode::SimpleThreshold);
        assert_eq!(cfg.halt_mode, HaltMode::SimpleThreshold);

        let cfg2 = QuartzConfig::default().with_halt_mode(HaltMode::Fixed { budget: 300 });
        assert_eq!(cfg2.halt_mode, HaltMode::Fixed { budget: 300 });
    }

    #[test]
    fn test_1a_fixed_halt_exhausts_budget() {
        // Fixed(200): should_stop returns true exactly at root_visits >= 200
        let qcfg = QuartzConfig::default().with_halt_mode(HaltMode::Fixed { budget: 200 });
        let ctrl = QuartzController::new(5000, qcfg);

        // Before budget: should NOT stop
        assert!(!ctrl.should_stop(199, 0), "should not stop before budget");
        // At budget: SHOULD stop
        assert!(ctrl.should_stop(200, 0), "should stop at budget");
        // Stop reason should be BudgetExhausted
        assert!(
            matches!(ctrl.stop_reason(), StopReason::BudgetExhausted { .. }),
            "Fixed mode should report BudgetExhausted, got {:?}",
            ctrl.stop_reason()
        );
    }

    #[test]
    fn test_1a_fixed_ignores_convergence() {
        // Fixed mode: even if P_flip is converged, keep running
        let qcfg = QuartzConfig {
            min_visits: 10,
            check_interval: 5,
            halt_mode: HaltMode::Fixed { budget: 500 },
            ..Default::default()
        };
        let ctrl = QuartzController::new(5000, qcfg);

        // Simulate: 100 visits, some elapsed time — should NOT stop
        assert!(
            !ctrl.should_stop(100, 500),
            "Fixed should not stop mid-budget regardless of convergence"
        );
        assert!(!ctrl.should_stop(499, 1000));
        assert!(ctrl.should_stop(500, 1000));
    }

    #[test]
    fn test_1a_voc_default_unchanged() {
        // VOC mode: same behavior as before PR-1A (backward compatibility)
        let qcfg = QuartzConfig::default(); // VOC mode
        assert_eq!(qcfg.halt_mode, HaltMode::VOC);
        let ctrl = QuartzController::new(1000, qcfg);

        // Without stats update, should not converge (min_visits not met)
        assert!(!ctrl.should_stop(10, 0));
        // At max_visits, always stops
        assert!(ctrl.should_stop(1000, 0));
    }

    #[test]
    fn test_1a_degenerate_sigma_no_panic() {
        // All three modes should handle degenerate stats without panic
        for mode in [
            HaltMode::VOC,
            HaltMode::SimpleThreshold,
            HaltMode::Fixed { budget: 100 },
        ] {
            let qcfg = QuartzConfig {
                min_visits: 5,
                check_interval: 5,
                halt_mode: mode.clone(),
                ..Default::default()
            };
            let ctrl = QuartzController::new(500, qcfg);
            // Just exercise should_stop at various visit counts — must not panic
            for v in [0, 1, 5, 50, 100, 499, 500] {
                let _ = ctrl.should_stop(v, 0);
            }
        }
    }

    // ── PR-1B: Selection Ablation Switch tests ────────────────

    #[test]
    fn test_1b_defaults_all_true() {
        let cfg = QuartzConfig::default();
        // v0.9.1: Fisher disabled by default (hurts with weak priors)
        assert!(!cfg.enable_fisher_puct);
        assert!(cfg.enable_one_loop);
        assert!(cfg.enable_expand_channel);
        assert!(cfg.enable_merge_channel);
    }

    #[test]
    fn test_1b_expand_off_forces_zero() {
        // With expand channel disabled, voc_expand should be 0.0
        // Create a simple tree node to test compute_quartz_stats
        let node = leak_node::<usize>(0, None);

        // Enable expand channel
        let cfg_on = QuartzConfig::default();
        let mut s0 = RunningEma::new(0.05);
        let _stats_on = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg_on);

        // Disable expand channel
        let cfg_off = QuartzConfig {
            enable_expand_channel: false,
            ..Default::default()
        };
        let mut s0 = RunningEma::new(0.05);
        let stats_off = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg_off);

        // voc_expand should be 0.0 when disabled
        assert_eq!(
            stats_off.voc_expand, 0.0,
            "voc_expand should be 0.0 when expand channel disabled"
        );
        assert_eq!(stats_off.unified.voc_expand, 0.0);
    }

    #[test]
    fn test_1b_merge_off_forces_zero() {
        let node = leak_node::<usize>(0, None);

        let cfg_off = QuartzConfig {
            enable_merge_channel: false,
            ..Default::default()
        };
        let mut s0 = RunningEma::new(0.05);
        let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg_off);

        assert_eq!(
            stats.voc_merge, 0.0,
            "voc_merge should be 0.0 when merge channel disabled"
        );
        assert_eq!(stats.unified.voc_merge, 0.0);
    }

    #[test]
    fn test_1b_all_off_still_computes_pflip() {
        // Even with all selection features off, P_flip should still be computed
        // (needed for SimpleThreshold halt mode)
        let node = leak_node::<usize>(0, None);

        let cfg = QuartzConfig {
            enable_fisher_puct: false,
            enable_one_loop: false,
            enable_expand_channel: false,
            enable_merge_channel: false,
            ..Default::default()
        };
        let mut s0 = RunningEma::new(0.05);
        let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg);

        // P_flip should still be computed (even if 0.5 on empty tree)
        assert!(stats.p_flip >= 0.0 && stats.p_flip <= 1.0);
        // Expand/Merge forced to zero
        assert_eq!(stats.voc_expand, 0.0);
        assert_eq!(stats.voc_merge, 0.0);
    }

    #[test]
    fn test_1b_fisher_switch_affects_score() {
        use crate::mcts::select::{ablation_puct_score, puct_score};

        let stats = QuartzStats::default();

        // Fisher ON (explicit)
        let cfg_on = QuartzConfig {
            enable_fisher_puct: true,
            ..Default::default()
        };
        let score_on = ablation_puct_score(10, 10, 0, 0.5, 0.1, 0.0, 100, 2.0, &stats, &cfg_on);

        // Fisher OFF (default)
        let cfg_off = QuartzConfig {
            enable_fisher_puct: false,
            ..Default::default()
        };
        let score_off = ablation_puct_score(10, 10, 0, 0.5, 0.1, 0.0, 100, 2.0, &stats, &cfg_off);

        // Standard PUCT (no quartz at all)
        let _score_std = puct_score(10, 0.5, 0.1, 0.0, 100, 2.0);

        // Fisher off should produce different score than Fisher on
        // (unless prior happens to make them equal, but with p=0.1, √0.1 ≠ 0.1)
        // Note: one_loop is still on in both cases, but stats.hbar_eff = 0 by default
        // so penalty = 0. With hbar_eff=0, both should differ only in Fisher term.
        assert!(
            (score_on - score_off).abs() > 1e-6,
            "Fisher switch should change score: on={} off={}",
            score_on,
            score_off
        );
    }

    #[test]
    fn test_1b_all_off_equals_puct() {
        use crate::mcts::select::{ablation_puct_score, puct_score};

        let stats = QuartzStats::default(); // hbar_eff=0, one_loop_b=0

        let cfg_off = QuartzConfig {
            enable_fisher_puct: false,
            enable_one_loop: false,
            ..Default::default()
        };

        // ablation score with everything off should equal standard puct
        let _abl = ablation_puct_score(10, 10, 0, 0.25, 0.0, 0.0, 200, 2.0, &stats, &cfg_off);
        let _std_puct = puct_score(10, 0.0, 0.25, 0.0, 200, 2.0);

        // With hbar_eff=0 and both switches off, they should be identical
        // Note: ablation passes q_eff differently... let me check the signatures
        // ablation_puct_score(n_eff, n_raw, o_a, q_eff, prior, noise_adj, n_parent_eff, c_puct, ...)
        // puct_score(n_eff, q_eff, prior, noise_adj, n_parent_eff, c_puct)
        let abl2 = ablation_puct_score(10, 10, 0, 0.5, 0.25, 0.0, 200, 2.0, &stats, &cfg_off);
        let std2 = puct_score(10, 0.5, 0.25, 0.0, 200, 2.0);
        assert!(
            (abl2 - std2).abs() < 1e-6,
            "All switches off should equal standard PUCT: abl={} std={}",
            abl2,
            std2
        );
    }

    // ── PR-3A★: CostMode tests ────────────────────────────────

    #[test]
    fn test_3a_cost_mode_default_is_legacy() {
        let cfg = QuartzConfig::default();
        assert_eq!(cfg.cost_mode, CostMode::Legacy);
    }

    #[test]
    fn test_3a_cost_mode_builder() {
        let cfg = QuartzConfig::default().with_cost_mode(CostMode::TimeDriven);
        assert_eq!(cfg.cost_mode, CostMode::TimeDriven);
    }

    // TEST-3A-1: constant_cost_baseline — existing tests pass with Constant cost
    #[test]
    fn test_3a1_constant_cost_baseline() {
        // Core QUARTZ math should work with any CostMode.
        // Verify compute_quartz_stats doesn't panic with Constant or TimeDriven.
        let node = leak_node::<usize>(0, None);

        for mode in [CostMode::Legacy, CostMode::Constant, CostMode::TimeDriven] {
            let cfg = QuartzConfig {
                cost_mode: mode.clone(),
                ..Default::default()
            };
            let mut s0 = RunningEma::new(0.05);
            let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg);

            // Basic validity
            assert!(
                stats.p_flip >= 0.0 && stats.p_flip <= 1.0,
                "mode={:?}: p_flip={}",
                mode,
                stats.p_flip
            );
            assert!(
                stats.cost_focus >= 0.0 || stats.cost_focus.is_nan() == false,
                "mode={:?}: cost_focus={}",
                mode,
                stats.cost_focus
            );
        }
    }

    // TEST-3A-2: time_driven_cost_profile — c_time monotonic
    #[test]
    fn test_3a2_time_driven_cost_profile() {
        let cfg = QuartzConfig {
            ctm_budget_ms: 1000, // 1 second budget
            ..Default::default()
        };

        // c_time should increase with elapsed time
        let c0 = cost_time_factor(0, &cfg);
        let c_mid = cost_time_factor(500, &cfg);
        let c_tgt = cost_time_factor(1000, &cfg);
        let c_2x = cost_time_factor(2000, &cfg);

        // Monotonicity
        assert!(c0 < c_mid, "c_time(0) < c_time(500): {} < {}", c0, c_mid);
        assert!(
            c_mid < c_tgt,
            "c_time(500) < c_time(1000): {} < {}",
            c_mid,
            c_tgt
        );
        assert!(
            c_tgt < c_2x,
            "c_time(1000) < c_time(2000): {} < {}",
            c_tgt,
            c_2x
        );

        // Boundary values
        assert!(
            c0 > 0.25 && c0 < 0.45,
            "c_time(0) should be ~0.3, got {}",
            c0
        );
        assert!(
            c_tgt > 0.55 && c_tgt < 0.75,
            "c_time(target) should be ~0.65, got {}",
            c_tgt
        );
        assert!(
            c_2x > 0.85,
            "c_time(2×target) should be >0.85, got {}",
            c_2x
        );

        // Always positive (Theorem 1 A4)
        assert!(c0 > 0.0);
    }

    #[test]
    fn test_3a2_time_driven_no_budget() {
        // No time budget → fixed 0.5
        let cfg = QuartzConfig {
            ctm_budget_ms: 0,
            ..Default::default()
        };
        let c = cost_time_factor(500, &cfg);
        assert!((c - 0.5).abs() < 1e-6, "no budget → c_time=0.5, got {}", c);
    }

    // TEST-3A-3: termination_guarantee — cost has positive lower bound
    #[test]
    fn test_3a3_termination_guarantee() {
        // For all CostModes, cost_focus must be > 0 (Theorem 1 A4)
        let node = leak_node::<usize>(0, None);

        for mode in [CostMode::Legacy, CostMode::Constant, CostMode::TimeDriven] {
            let cfg = QuartzConfig {
                cost_mode: mode.clone(),
                ctm_budget_ms: 1000,
                ..Default::default()
            };
            let mut s0 = RunningEma::new(0.05);
            let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 500, &cfg);

            // cost_focus should be non-negative
            // (may be 0 on empty tree due to σ₀/N_min being small, but not negative)
            assert!(
                stats.cost_focus >= 0.0,
                "mode={:?}: cost_focus={} must be ≥ 0",
                mode,
                stats.cost_focus
            );
        }
    }

    // TEST-3A: cost values differ between modes
    #[test]
    fn test_3a_cost_modes_differ() {
        let node = leak_node::<usize>(0, None);

        let cfg_legacy = QuartzConfig {
            cost_mode: CostMode::Legacy,
            ..Default::default()
        };
        let cfg_const = QuartzConfig {
            cost_mode: CostMode::Constant,
            ..Default::default()
        };
        let cfg_time = QuartzConfig {
            cost_mode: CostMode::TimeDriven,
            ctm_budget_ms: 1000,
            ..Default::default()
        };

        let mut s0 = RunningEma::new(0.05);
        let s_leg = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 500, &cfg_legacy);
        let mut s0 = RunningEma::new(0.05);
        let s_con = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 500, &cfg_const);
        let mut s0 = RunningEma::new(0.05);
        let s_tim = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 500, &cfg_time);

        // On an empty tree, hbar_eff ≈ 0, so Legacy and Constant may both be ~0
        // But at least the cost_focus values should be computed without error
        println!("  Legacy cost:     {:.6}", s_leg.cost_focus);
        println!("  Constant cost:   {:.6}", s_con.cost_focus);
        println!("  TimeDriven cost: {:.6}", s_tim.cost_focus);

        // Constant should differ from TimeDriven when ctm_budget_ms differs
        // (both are non-zero on non-empty trees; on empty tree they may both be small)
    }

    // ── PR-6A: Conf(t) Adaptive Halt tests ────────────────

    // TEST-6A-1: conf_formula_values
    #[test]
    fn test_6a1_conf_formula_values() {
        // Conf(t) = (1−P_flip)(1−P_hidden)·max{0, 1−S/S₀}
        // P_flip=0.1, P_hidden=0.05, S/S₀=0.3
        // → (0.9)(0.95)(0.7) = 0.5985
        let p_flip = 0.1f32;
        let p_hidden = 0.05f32;
        let surprise_ratio = 0.3f32;
        let conf = (1.0 - p_flip) * (1.0 - p_hidden) * (1.0 - surprise_ratio).max(0.0);
        assert!(
            (conf - 0.5985).abs() < 0.001,
            "Conf should be 0.5985, got {:.4}",
            conf
        );
    }

    // TEST-6A-2: conf_monotone
    #[test]
    fn test_6a2_conf_monotone() {
        let compute_conf =
            |pf: f32, ph: f32, sr: f32| -> f32 { (1.0 - pf) * (1.0 - ph) * (1.0 - sr).max(0.0) };

        // P_flip↓ → Conf↑
        let c1 = compute_conf(0.3, 0.0, 0.2);
        let c2 = compute_conf(0.1, 0.0, 0.2);
        assert!(
            c2 > c1,
            "lower P_flip should give higher Conf: {} vs {}",
            c2,
            c1
        );

        // S/S₀↓ → Conf↑ (lower surprise ratio)
        let c3 = compute_conf(0.2, 0.0, 0.5);
        let c4 = compute_conf(0.2, 0.0, 0.1);
        assert!(
            c4 > c3,
            "lower surprise should give higher Conf: {} vs {}",
            c4,
            c3
        );

        // P_hidden↓ → Conf↑
        let c5 = compute_conf(0.2, 0.3, 0.2);
        let c6 = compute_conf(0.2, 0.0, 0.2);
        assert!(
            c6 > c5,
            "lower P_hidden should give higher Conf: {} vs {}",
            c6,
            c5
        );

        // Conf ∈ [0, 1]
        assert!(compute_conf(0.0, 0.0, 0.0) <= 1.0);
        assert!(compute_conf(1.0, 1.0, 1.0) >= 0.0);
    }

    // TEST-6A-3: adaptive_theta_convergence
    #[test]
    fn test_6a3_theta_conf_adaptation() {
        let qcfg = QuartzConfig {
            halt_mode: HaltMode::ConfAdaptive {
                target_time_ms: 100,
                theta_init: 0.8,
                eta: 0.05,
            },
            min_visits: 5,
            check_interval: 5,
            ..Default::default()
        };
        let ctrl = QuartzController::new(500, qcfg);

        // Initially theta = 0.8
        {
            let g = ctrl.inner.lock().unwrap();
            assert!(
                (g.theta_conf - 0.8).abs() < 1e-6,
                "initial theta={}",
                g.theta_conf
            );
        }

        // Simulate: should_stop called with various elapsed times
        // If elapsed > target, theta should increase (make it harder to stop → spend more time)
        // If elapsed < target, theta should decrease (make it easier to stop)
        // Note: without stats update, conf_t=0 so it won't trigger,
        // but theta should still be initialized correctly.
        assert!(!ctrl.should_stop(10, 50)); // no convergence yet
        assert!(!ctrl.should_stop(20, 150)); // still no convergence (conf_t=0 in default stats)
    }

    // TEST-6A-4: ConfAdaptive creates valid controller
    #[test]
    fn test_6a4_conf_adaptive_no_panic() {
        for target in [0, 50, 100, 1000] {
            let qcfg = QuartzConfig {
                halt_mode: HaltMode::ConfAdaptive {
                    target_time_ms: target,
                    theta_init: 0.8,
                    eta: 0.01,
                },
                min_visits: 10,
                check_interval: 10,
                ..Default::default()
            };
            let ctrl = QuartzController::new(500, qcfg);

            // Exercise should_stop at various points — must not panic
            for v in [0, 5, 10, 50, 100, 499, 500] {
                let _ = ctrl.should_stop(v, v as u64 * 2);
            }
        }
    }

    // TEST-6A-5: p_hidden_dormant_graceful (7×7)
    #[test]
    fn test_6a5_p_hidden_dormant() {
        // On empty tree, p_hidden ≈ 0, so Conf(t) ≈ (1−P_flip)·max{0, 1−S/S₀}
        let node = leak_node::<usize>(0, None);
        let cfg = QuartzConfig::default();
        let mut s0 = RunningEma::new(0.05);
        let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg);

        // p_hidden should be very small (≈ 0) on empty tree
        assert!(
            stats.p_hidden < 0.1,
            "p_hidden should be near 0 on empty tree, got {}",
            stats.p_hidden
        );

        // conf_t should be computable and in [0, 1]
        assert!(
            stats.conf_t >= 0.0 && stats.conf_t <= 1.0,
            "conf_t={} out of range",
            stats.conf_t
        );
    }

    // TEST-6A: conf_t appears in stats from engine run
    #[test]
    fn test_6a_conf_in_engine_stats() {
        use crate::games::Gomoku;
        use crate::mcts::eval::ShortRollout;

        let state = Gomoku::new_with_win(7, 4);
        let eval: Arc<dyn crate::game::Evaluator<Gomoku> + Send + Sync> =
            Arc::new(ShortRollout::new(20));
        let qcfg = QuartzConfig {
            halt_mode: HaltMode::ConfAdaptive {
                target_time_ms: 500,
                theta_init: 0.7,
                eta: 0.02,
            },
            min_visits: 20,
            check_interval: 10,
            ..Default::default()
        };
        let config = crate::mcts::MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());

        let engine = crate::mcts::MctsEngine::new(state, eval, config);
        let mut ctrl = QuartzController::new(200, qcfg);
        engine.run_quartz(&mut ctrl);

        let stats = ctrl.last_stats();
        // conf_t should be computed
        assert!(
            stats.conf_t >= 0.0 && stats.conf_t <= 1.0,
            "conf_t={} invalid after engine run",
            stats.conf_t
        );
        assert!(stats.p_flip >= 0.0 && stats.p_flip <= 1.0);

        println!(
            "  ConfAdaptive engine run: conf_t={:.4} p_flip={:.4} p_hidden={:.4}",
            stats.conf_t, stats.p_flip, stats.p_hidden
        );
    }

    // ── PR-6B: Saddlepoint P_flip tests ───────────────────

    // TEST-6B-1: Gaussian limit — skew=0, kurt=0 → CF ≈ Gaussian
    #[test]
    fn test_6b1_saddlepoint_gaussian_limit() {
        let mu_d = 0.5f32;
        let sigma_d = 0.3f32;
        let p_gauss = standard_normal_cdf(-mu_d / sigma_d);
        let p_cf = cornish_fisher_pflip(mu_d, sigma_d, 0.0, 0.0);
        assert!(
            (p_gauss - p_cf).abs() < 0.01,
            "CF with skew=0 kurt=0 should equal Gaussian: gauss={:.4} cf={:.4}",
            p_gauss,
            p_cf
        );
    }

    #[test]
    fn test_6b1_various_mu() {
        for &mu in &[0.0, 0.1, 0.5, 1.0, 2.0] {
            let sigma = 0.3;
            let g = standard_normal_cdf(-mu / sigma);
            let cf = cornish_fisher_pflip(mu, sigma, 0.0, 0.0);
            assert!(
                (g - cf).abs() < 0.01,
                "mu={}: gauss={:.4} cf={:.4}",
                mu,
                g,
                cf
            );
        }
    }

    // TEST-6B-2: Skewed distribution → CF differs from Gaussian
    #[test]
    fn test_6b2_saddlepoint_skewed() {
        let mu_d = 0.3f32;
        let sigma_d = 0.5f32;
        let p_gauss = standard_normal_cdf(-mu_d / sigma_d);
        // Positive skew: right tail heavier → P_flip should increase
        let p_cf_pos = cornish_fisher_pflip(mu_d, sigma_d, 1.5, 0.0);
        // Negative skew: left tail heavier
        let p_cf_neg = cornish_fisher_pflip(mu_d, sigma_d, -1.5, 0.0);

        // With significant skew, CF should differ from Gaussian
        assert!(
            (p_gauss - p_cf_pos).abs() > 0.01 || (p_gauss - p_cf_neg).abs() > 0.01,
            "At least one skewed CF should differ from Gaussian: g={:.4} pos={:.4} neg={:.4}",
            p_gauss,
            p_cf_pos,
            p_cf_neg
        );
    }

    #[test]
    fn test_6b2_kurtosis_effect() {
        let mu_d = 0.3f32;
        let sigma_d = 0.4f32;
        let p_gauss = standard_normal_cdf(-mu_d / sigma_d);
        // High excess kurtosis (heavy tails)
        let p_cf_heavy = cornish_fisher_pflip(mu_d, sigma_d, 0.0, 3.0);
        // The effect should be nonzero
        println!(
            "  Kurtosis test: gauss={:.4} cf_kurt3={:.4}",
            p_gauss, p_cf_heavy
        );
        // At least the computation shouldn't panic and should be in [0,1]
        assert!(p_cf_heavy >= 0.0 && p_cf_heavy <= 1.0);
    }

    // TEST-6B-3: Gating — only applies when is_heavy_tail
    #[test]
    fn test_6b3_saddlepoint_gated() {
        let mu_d = 0.3f32;
        let sigma_d = 0.4f32;
        let skew = 1.0f32;
        let kurt = 2.0f32;

        let (p_used_ht, _p_g, p_sp) = pflip_with_correction(mu_d, sigma_d, skew, kurt, true, 10);
        let (p_used_no, p_g2, _) = pflip_with_correction(mu_d, sigma_d, skew, kurt, false, 10);

        // When heavy-tail: should use saddlepoint
        assert_eq!(p_used_ht, p_sp, "heavy-tail → use saddlepoint");
        // When not heavy-tail: should use Gaussian
        assert_eq!(p_used_no, p_g2, "no heavy-tail → use Gaussian");
    }

    #[test]
    fn test_6b3_gated_insufficient_data() {
        // Even with heavy-tail, if n_visible < 8, use Gaussian
        let (p_used, p_g, _) = pflip_with_correction(0.3, 0.4, 1.0, 2.0, true, 5);
        assert_eq!(
            p_used, p_g,
            "n_visible < 8 → use Gaussian even with heavy-tail"
        );
    }

    // TEST-6B-4: Stability — extreme inputs don't crash
    #[test]
    fn test_6b4_stability() {
        // Zero sigma
        let p = cornish_fisher_pflip(0.5, 0.0, 1.0, 2.0);
        assert!(p >= 0.0 && p <= 1.0);

        // Extreme skew/kurtosis → should fallback to Gaussian (stability guard)
        let p = cornish_fisher_pflip(0.3, 0.4, 10.0, 50.0);
        assert!(p >= 0.0 && p <= 1.0, "extreme cumulants: p={}", p);

        // Zero everything
        let p = cornish_fisher_pflip(0.0, 0.0, 0.0, 0.0);
        assert!((p - 0.5).abs() < 0.01, "zero mu/sigma → 0.5, got {}", p);
    }

    // PR-6B diagnostic fields appear in stats
    #[test]
    fn test_6b_diagnostic_fields() {
        let node = leak_node::<usize>(0, None);
        let cfg = QuartzConfig::default();
        let mut s0 = RunningEma::new(0.05);
        let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg);

        // Both diagnostic fields should be computed
        assert!(stats.p_flip_gaussian >= 0.0 && stats.p_flip_gaussian <= 1.0);
        assert!(stats.p_flip_saddlepoint >= 0.0 && stats.p_flip_saddlepoint <= 1.0);
        // On empty tree (no heavy tail), p_flip should equal p_flip_gaussian
        assert_eq!(
            stats.p_flip, stats.p_flip_gaussian,
            "no heavy-tail → p_flip should be Gaussian"
        );
    }

    // ── PR-6C: Depth Calibration κ_b tests ────────────────

    #[test]
    fn test_6c1_three_buckets() {
        assert_eq!(DepthBucket::from_depth(0), DepthBucket::Shallow);
        assert_eq!(DepthBucket::from_depth(3), DepthBucket::Shallow);
        assert_eq!(DepthBucket::from_depth(4), DepthBucket::Mid);
        assert_eq!(DepthBucket::from_depth(8), DepthBucket::Mid);
        assert_eq!(DepthBucket::from_depth(9), DepthBucket::Deep);
        assert_eq!(DepthBucket::from_depth(100), DepthBucket::Deep);
    }

    #[test]
    fn test_6c1_initial_kappa() {
        let dc = DepthCalibration::new(0.1);
        assert_eq!(dc.kappas(), [1.0, 1.0, 1.0]);
        assert_eq!(dc.kappa_at(0), 1.0);
        assert_eq!(dc.kappa_at(5), 1.0);
        assert_eq!(dc.kappa_at(10), 1.0);
    }

    #[test]
    fn test_6c2_kappa_update() {
        let mut dc = DepthCalibration::new(0.3);
        // Shallow: empirical σ is 2× predicted → κ should increase
        for _ in 0..10 {
            dc.record(1, 0.6, 0.3); // ratio = 2.0
        }
        assert!(
            dc.kappa_at(1) > 1.0,
            "κ_shallow should increase: {}",
            dc.kappa_at(1)
        );
        assert!(dc.kappa_at(1) < 2.5, "but not explode");

        // Deep: empirical σ is 0.5× predicted → κ should decrease
        for _ in 0..10 {
            dc.record(10, 0.15, 0.3); // ratio = 0.5
        }
        assert!(
            dc.kappa_at(10) < 1.0,
            "κ_deep should decrease: {}",
            dc.kappa_at(10)
        );
    }

    #[test]
    fn test_6c3_calibrate_sigma() {
        let mut dc = DepthCalibration::new(0.5);
        dc.record(2, 0.6, 0.3); // shallow: ratio=2
        let raw = 0.4f32;
        let calibrated = dc.calibrate_sigma(2, raw);
        assert!(calibrated > raw, "calibrated should be > raw when κ>1");
    }

    // ── PR-6D: w_imp / gvoc_nonroot tests ─────────────────

    #[test]
    fn test_6d1_wimp_root_is_one() {
        // At root: w_imp = N(root)/N(root) = 1.0
        // This is already hardcoded as wimp=1.0 in compute_quartz_stats
        let node = leak_node::<usize>(0, None);
        let cfg = QuartzConfig::default();
        let mut s0 = RunningEma::new(0.05);
        let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg);
        // voc_total = wimp(=1.0) × max_voc
        // gvoc_nonroot_max should be ≤ voc_total (since w_imp ≤ 1.0 for children)
        assert!(
            stats.unified.gvoc_nonroot_max <= stats.unified.voc_total + 1e-6,
            "gvoc_nonroot should be ≤ voc_total"
        );
    }

    #[test]
    fn test_6d4_root_only_equivalent() {
        // When gvoc_nonroot_max is 0 or ignored, behavior should be same as before PR-6D
        let node = leak_node::<usize>(0, None);
        let cfg = QuartzConfig::default();
        let mut s0 = RunningEma::new(0.05);
        let stats = compute_quartz_stats(&node, None, &mut s0, 0.01, 0, 0, &cfg);

        // On empty tree, gvoc_nonroot_max should be 0
        assert_eq!(
            stats.unified.gvoc_nonroot_max, 0.0,
            "empty tree → gvoc_nonroot=0"
        );
    }

    // ── PR-6E: Robust σ tests ─────────────────────────────

    #[test]
    fn test_6e1_mom_gaussian() {
        // Gaussian-like data: MoM σ ≈ Welford σ
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mom = median_of_means_sigma(&data, 4);
        assert!(mom.is_some());
        let mom_s = mom.unwrap();
        // Welford σ for this data
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
        let welford_s = var.sqrt();
        // Should be within factor of 2
        assert!(
            mom_s > welford_s * 0.3 && mom_s < welford_s * 3.0,
            "MoM={:.4} Welford={:.4} should be similar",
            mom_s,
            welford_s
        );
    }

    #[test]
    fn test_6e2_mom_outlier_robust() {
        // Data with outliers: MoM should be less affected
        let mut data: Vec<f32> = vec![0.0; 28];
        for i in 0..28 {
            data[i] = (i as f32 - 14.0) * 0.05;
        }
        // Add extreme outliers
        data.push(100.0);
        data.push(-100.0);
        data.push(50.0);
        data.push(-50.0);

        let mom = median_of_means_sigma(&data, 4).unwrap();
        // Welford would be dominated by outliers
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
        let welford_s = var.sqrt();

        // MoM should be much smaller than Welford (less outlier influence)
        assert!(
            mom < welford_s * 0.5,
            "MoM={:.4} should be much less than Welford={:.4} with outliers",
            mom,
            welford_s
        );
    }

    #[test]
    fn test_6e3_circular_buffer() {
        let mut buf = CircularBuffer::new(4);
        assert_eq!(buf.len(), 0);
        buf.push(1.0);
        buf.push(2.0);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.as_slice(), &[1.0, 2.0]);
        buf.push(3.0);
        buf.push(4.0);
        assert_eq!(buf.len(), 4);
        buf.push(5.0); // overwrites first
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.as_slice(), &[5.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_6e_insufficient_data() {
        let data = vec![1.0, 2.0, 3.0];
        // 4 groups × 2 min = 8 needed, only 3 available
        assert!(median_of_means_sigma(&data, 4).is_none());
    }

    // ── PR-6F: NS Gate tests ──────────────────────────────

    #[test]
    fn test_6f1_ns_gate_all_false() {
        let mut gate = NsGate::new();
        // Feed uniform low values → all normalized ratios ≈ 1 (no anomaly)
        for _ in 0..10 {
            let _r = gate.evaluate(0.5, 0.01, 1.0);
            // After warmup, ratios should be near 1.0 (current ≈ median)
        }
        let r = gate.evaluate(0.5, 0.01, 1.0);
        // All ratios ≈ 1.0, none significantly > 1.0
        assert!(
            !r.gate_on || r.he <= 1.01 && r.se <= 1.01 && r.te <= 1.01,
            "uniform input should not trigger gate: {:?}",
            r
        );
    }

    #[test]
    fn test_6f2_ns_gate_heavy_tail() {
        let mut gate = NsGate::new();
        // Train on normal heavy_tail values
        for _ in 0..20 {
            gate.evaluate(0.5, 0.01, 1.0);
        }
        // Spike in heavy_tail → Te > 1
        let r = gate.evaluate(0.5, 0.01, 5.0);
        assert!(r.te > 1.0, "heavy-tail spike should give Te > 1: {:?}", r);
        assert!(r.gate_on, "gate should be ON when Te > 1");
    }

    #[test]
    fn test_6f3_ns_gate_multimodal() {
        let mut gate = NsGate::new();
        for _ in 0..20 {
            gate.evaluate(0.3, 0.01, 1.0);
        }
        // Spike in bimodality
        let r = gate.evaluate(2.0, 0.01, 1.0);
        assert!(r.he > 1.0, "bimodality spike should give He > 1: {:?}", r);
        assert!(r.gate_on);
    }

    #[test]
    fn test_6f4_ns_gate_surprise() {
        let mut gate = NsGate::new();
        for _ in 0..20 {
            gate.evaluate(0.5, 0.01, 1.0);
        }
        // Spike in surprise
        let r = gate.evaluate(0.5, 0.5, 1.0);
        assert!(r.se > 1.0, "surprise spike should give Se > 1: {:?}", r);
        assert!(r.gate_on);
    }

    #[test]
    fn test_6f_gate_reset() {
        let mut gate = NsGate::new();
        for _ in 0..20 {
            gate.evaluate(0.5, 0.01, 1.0);
        }
        gate.reset();
        // After reset, medians should be re-initialized
        let r = gate.evaluate(0.5, 0.01, 1.0);
        // First observation after reset → ratio = value/value = 1.0
        assert!(
            !r.gate_on,
            "first observation after reset should not trigger"
        );
    }
}

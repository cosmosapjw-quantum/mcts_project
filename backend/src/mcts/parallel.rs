//! § ParallelismController — Adaptive Virtual Loss for tree-parallel MCTS
//!
//! Separated from the search-policy controller (QUARTZ) by design:
//! - QUARTZ controls **what** to search (penalty, refresh, halt)
//! - ParallelismController controls **how** to parallelise (VL magnitude)
//!
//! # Design: Split VL
//!
//! Traditional VL applies a single value: `n += 1, w -= 1`.
//! We split into two independently adaptive components:
//!
//! - **vvisit (λ_v)**: virtual visit inflation → reduces exploration bonus
//! - **vvalue (λ_q)**: virtual value penalty  → pessimises Q estimate
//!
//! # Hyperparameter-free derivation
//!
//! All parameters are derived from observable search state:
//!
//! - `λ_v = 1.0` (topological: 1 reservation per pending thread — constant)
//! - `λ_q = σ_Q × depth_decay` (scales with Q uncertainty — adaptive)
//! - `depth_decay = 1 / (1 + depth)` (root contention most expensive)
//!
//! # Evidence (ablation_vl.rs, gomoku7 × ShortRollout)
//!
//! - Adaptive VL default: +5-10%p agreement over Fixed VL at 4 threads
//! - Fixed VL=1 over-pessimises Q by ~10× (AvgVV=1.0 vs σ_Q-scaled=0.1)
//! - Fixed VL + SelfAdaptive QUARTZ penalty = destructive interaction (40%)
//! - Adaptive VL + SelfAdaptive = rescued to 50% (σ_Q auto-correction)
//! - VvisitOnly (reservation) is the primary mechanism; vvalue is refinement
//! - Adaptive advantage is largest at low budgets, converges at high budget

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ───────────────────────────────────────────
// § VL Mode
// ───────────────────────────────────────────

/// Virtual loss strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VlMode {
    /// Classic fixed VL=1 (baseline for ablation).
    Fixed,
    /// No virtual loss at all (serial-equivalent for ablation).
    Disabled,
    /// Adaptive: vvisit=1, vvalue=σ_Q/(1+depth).
    /// Hyperparameter-free — reads σ_Q from search state.
    Adaptive,
    /// Ablation: vvisit=1 only (reservation, no pessimism).
    VvisitOnly,
    /// Ablation: vvalue=σ_Q/(1+depth) only (pessimism, no reservation).
    VvalueOnly,
}

impl Default for VlMode {
    fn default() -> Self {
        VlMode::Adaptive
    }
}

// ───────────────────────────────────────────
// § Split VL values
// ───────────────────────────────────────────

/// Split virtual loss magnitudes for one select→backup cycle.
#[derive(Debug, Clone, Copy)]
pub struct VlSplit {
    /// Virtual visit: inflates N_eff, reduces PUCT exploration bonus.
    pub vvisit: f32,
    /// Virtual value: pessimises Q_eff, diverts other threads.
    pub vvalue: f32,
}

impl VlSplit {
    pub const FIXED: VlSplit = VlSplit {
        vvisit: 1.0,
        vvalue: 1.0,
    };
    pub const ZERO: VlSplit = VlSplit {
        vvisit: 0.0,
        vvalue: 0.0,
    };
}

// ───────────────────────────────────────────
// § Telemetry (parallel-specific)
// ───────────────────────────────────────────

/// Cache-line padded wrapper. B1: prevents false sharing of telemetry
/// atomics that are written/read by all rayon worker threads in run_par.
/// `Deref`/`DerefMut` keep call sites unchanged
/// (`self.field.fetch_add(...)` still works via auto-deref).
#[repr(C, align(64))]
struct CacheLined<T>(T);

impl<T> std::ops::Deref for CacheLined<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T> std::ops::DerefMut for CacheLined<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// Runtime parallel telemetry — accumulated across iterations.
/// B1: each atomic occupies its own 64-byte line via `CacheLined<T>`,
/// eliminating cross-core line bouncing on the rayon hot path.
pub struct ParallelTelemetry {
    total_selects: CacheLined<AtomicU32>,
    dup_leaf_count: CacheLined<AtomicU32>, // times a leaf was already pending
    max_pending: CacheLined<AtomicU32>,    // peak pending leaves across all nodes
    /// Sum of applied vvalues in micro-units.
    ///
    /// This is telemetry, not a search-state variable. Using integer
    /// accumulation turns the former f64 CAS loop into one fetch_add on the
    /// hot path while preserving sub-ppm reporting precision for dashboards.
    vvalue_sum_micro: CacheLined<AtomicU64>,
}

impl ParallelTelemetry {
    pub fn new() -> Self {
        ParallelTelemetry {
            total_selects: CacheLined(AtomicU32::new(0)),
            dup_leaf_count: CacheLined(AtomicU32::new(0)),
            max_pending: CacheLined(AtomicU32::new(0)),
            vvalue_sum_micro: CacheLined(AtomicU64::new(0)),
        }
    }

    #[inline(always)]
    pub fn record_select(&self, pending_at_node: u32) {
        self.total_selects.fetch_add(1, Ordering::Relaxed);
        if pending_at_node == 0 {
            return;
        }
        let mut prev = self.max_pending.load(Ordering::Relaxed);
        while pending_at_node > prev {
            match self.max_pending.compare_exchange_weak(
                prev,
                pending_at_node,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(cur) => prev = cur,
            }
        }
    }

    #[inline(always)]
    pub fn record_dup_leaf(&self) {
        self.dup_leaf_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_vvalue(&self, v: f32) {
        self.vvalue_sum_micro
            .fetch_add(float_to_micro(v), Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.total_selects.store(0, Ordering::Relaxed);
        self.dup_leaf_count.store(0, Ordering::Relaxed);
        self.max_pending.store(0, Ordering::Relaxed);
        self.vvalue_sum_micro.store(0, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dup_rate(&self) -> f32 {
        let total = self.total_selects.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.dup_leaf_count.load(Ordering::Relaxed) as f32 / total as f32
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            #[cfg(test)]
            total_selects: self.total_selects.load(Ordering::Relaxed),
            #[cfg(test)]
            dup_leaf_count: self.dup_leaf_count.load(Ordering::Relaxed),
            max_pending: self.max_pending.load(Ordering::Relaxed),
            avg_vvalue: {
                let t = self.total_selects.load(Ordering::Relaxed);
                if t == 0 {
                    0.0
                } else {
                    let s = micro_to_float(self.vvalue_sum_micro.load(Ordering::Relaxed));
                    (s / t as f64) as f32
                }
            },
            dup_rate: self.dup_rate(),
        }
    }
}

#[inline(always)]
fn float_to_micro(v: f32) -> u64 {
    if v.is_finite() {
        (v.max(0.0) as f64 * 1_000_000.0)
            .round()
            .clamp(0.0, u64::MAX as f64) as u64
    } else {
        0
    }
}

#[inline(always)]
fn micro_to_float(v: u64) -> f64 {
    v as f64 / 1_000_000.0
}

#[derive(Debug, Clone, Copy)]
pub struct TelemetrySnapshot {
    #[cfg(test)]
    pub total_selects: u32,
    #[cfg(test)]
    pub dup_leaf_count: u32,
    pub max_pending: u32,
    pub avg_vvalue: f32,
    pub dup_rate: f32,
}

// ───────────────────────────────────────────
// § Auto Thread Policy
// ───────────────────────────────────────────

/// Search-thread policy for opt-in automatic thread selection.
///
/// This is intentionally separate from `VlMode`: thread count controls CPU
/// scheduling, while VL controls duplicate-path avoidance inside shared-tree
/// MCTS. Keeping them separate preserves existing ablation semantics for
/// explicit `run_par(..., n_threads)` calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoThreadMode {
    /// Maximize raw node throughput when the visit budget can absorb worker
    /// overhead. This may tolerate higher duplicate-selection telemetry.
    Throughput,
    /// Conservative mode for search-quality or ablation runs. It caps tiny
    /// action spaces more aggressively to avoid excessive virtual-loss churn.
    Quality,
}

impl Default for AutoThreadMode {
    fn default() -> Self {
        AutoThreadMode::Throughput
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoThreadPolicy {
    pub mode: AutoThreadMode,
    /// Optional caller cap. `None` means the host's available parallelism.
    pub max_threads: Option<usize>,
}

impl AutoThreadPolicy {
    pub const fn throughput() -> Self {
        Self {
            mode: AutoThreadMode::Throughput,
            max_threads: None,
        }
    }

    pub const fn quality() -> Self {
        Self {
            mode: AutoThreadMode::Quality,
            max_threads: None,
        }
    }

    pub const fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.max_threads = Some(max_threads);
        self
    }
}

impl Default for AutoThreadPolicy {
    fn default() -> Self {
        AutoThreadPolicy::throughput()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoThreadReason {
    SingleHostThread,
    TrivialSearch,
    TinyBudget,
    BudgetLimited,
    QualityLowBranchingCap,
    ThroughputHostCap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoThreadInput {
    pub host_threads: usize,
    /// Remaining visit budget, not absolute root visits. `None` means the
    /// controller has no exact fixed-visit hint.
    pub remaining_visits: Option<u32>,
    pub root_legal_count: usize,
    pub pw_enabled: bool,
    pub reusable_select_scratch: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoThreadDecision {
    pub threads: usize,
    pub host_threads: usize,
    pub requested_cap: usize,
    pub remaining_visits: Option<u32>,
    pub root_legal_count: usize,
    pub reason: AutoThreadReason,
}

pub fn available_search_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

pub fn recommend_auto_threads(
    input: AutoThreadInput,
    policy: AutoThreadPolicy,
) -> AutoThreadDecision {
    let host_threads = input.host_threads.max(1);
    let requested_cap = policy
        .max_threads
        .unwrap_or(host_threads)
        .max(1)
        .min(host_threads);
    let legal = input.root_legal_count.max(1);
    let remaining = input.remaining_visits.map(|v| v as usize);

    let make = |threads: usize, reason: AutoThreadReason| AutoThreadDecision {
        threads: threads.max(1).min(requested_cap),
        host_threads,
        requested_cap,
        remaining_visits: input.remaining_visits,
        root_legal_count: input.root_legal_count,
        reason,
    };

    if requested_cap <= 1 {
        return make(1, AutoThreadReason::SingleHostThread);
    }
    if legal <= 1 || matches!(remaining, Some(0 | 1)) {
        return make(1, AutoThreadReason::TrivialSearch);
    }

    if let Some(visits) = remaining {
        if visits < 128 {
            return make(1, AutoThreadReason::TinyBudget);
        }
        if visits < 512 {
            return make(2, AutoThreadReason::TinyBudget);
        }
    }

    let visits_per_thread = match policy.mode {
        AutoThreadMode::Throughput => 64,
        AutoThreadMode::Quality => 384,
    };
    let mut threads = requested_cap;
    let mut reason = AutoThreadReason::ThroughputHostCap;
    if let Some(visits) = remaining {
        let budget_cap = (visits / visits_per_thread).max(1);
        if budget_cap < threads {
            threads = budget_cap;
            reason = AutoThreadReason::BudgetLimited;
        }
    }

    if policy.mode == AutoThreadMode::Quality {
        let low_branch_cap = if legal <= 16 {
            4
        } else if legal <= 32 {
            8
        } else if !input.pw_enabled && legal <= 64 {
            // Full materialization plus modest branching is the pattern that
            // produced the highest TT/materialization pressure in profiling.
            requested_cap.saturating_div(2).max(4)
        } else {
            requested_cap
        };
        if low_branch_cap < threads {
            threads = low_branch_cap;
            reason = AutoThreadReason::QualityLowBranchingCap;
        }

        if !input.reusable_select_scratch && legal <= 32 && threads > 8 {
            threads = 8;
            reason = AutoThreadReason::QualityLowBranchingCap;
        }
    }

    make(threads, reason)
}

// ───────────────────────────────────────────
// § ParallelismController
// ───────────────────────────────────────────

/// Adaptive virtual loss controller. Separated from QUARTZ.
///
/// Reads QUARTZ diagnostic output (σ_Q) in one-way coupling.
/// Computes per-depth (vvisit, vvalue) split for each select step.
///
/// Thread-safe: all mutable state uses atomics.
pub struct ParallelismController {
    mode: VlMode,
    n_threads: AtomicU32,
    /// σ_Q from QUARTZ (updated periodically). Stored as f32 bits in AtomicU32.
    sigma_q: AtomicU32,
    /// Root policy entropy from QUARTZ. Stored as f32 bits in AtomicU32.
    root_entropy: AtomicU32,
    /// Cached `(clamp(root_entropy, 0.5, 2.0) / 2.0)` used in adaptive VL.
    entropy_factor: AtomicU32,
    pub telemetry: ParallelTelemetry,
}

impl ParallelismController {
    pub fn new(mode: VlMode, n_threads: u32) -> Self {
        ParallelismController {
            mode,
            n_threads: AtomicU32::new(n_threads.max(1)),
            sigma_q: AtomicU32::new(0.3_f32.to_bits()),
            root_entropy: AtomicU32::new(1.0_f32.to_bits()),
            entropy_factor: AtomicU32::new(0.5_f32.to_bits()),
            telemetry: ParallelTelemetry::new(),
        }
    }

    /// Update thread count (e.g., when switching between single and parallel search).
    pub fn set_n_threads(&self, n: u32) {
        self.n_threads.store(n.max(1), Ordering::Relaxed);
    }

    /// Whether ordinary `iterate()` calls need virtual-loss reservations.
    ///
    /// Serial search has no competing worker observing pending edges, so VL
    /// only adds atomic traffic. Batched async selection forces reservation at
    /// its call site because multiple prepared leaves can be pending even on a
    /// single OS thread.
    #[inline(always)]
    pub fn should_reserve_virtual_loss(&self) -> bool {
        self.mode != VlMode::Disabled && self.n_threads.load(Ordering::Relaxed) > 1
    }

    /// Whether selection must read virtual-value pessimism even when an edge
    /// has no virtual-visit reservation. This is true only for the explicit
    /// `VvalueOnly` ablation; production Adaptive/Fixed modes publish vvalue
    /// together with vvisit, so `o_a == 0` is enough to skip that atomic load.
    #[inline(always)]
    pub fn can_publish_vvalue_without_vvisit(&self) -> bool {
        self.mode == VlMode::VvalueOnly
    }

    /// One-way read from QUARTZ: update σ_Q and root_entropy.
    /// Called by thread 0 periodically during run_par.
    pub fn update_from_search(&self, sigma_q: f32, root_entropy: f32) {
        let root_entropy = root_entropy.max(0.01);
        self.sigma_q
            .store(sigma_q.max(0.01).to_bits(), Ordering::Relaxed);
        self.root_entropy
            .store(root_entropy.to_bits(), Ordering::Relaxed);
        self.entropy_factor.store(
            (root_entropy.clamp(0.5, 2.0) / 2.0).to_bits(),
            Ordering::Relaxed,
        );
    }

    #[inline(always)]
    fn sigma_q_val(&self) -> f32 {
        f32::from_bits(self.sigma_q.load(Ordering::Relaxed))
    }

    #[cfg(test)]
    #[inline(always)]
    fn root_entropy_val(&self) -> f32 {
        f32::from_bits(self.root_entropy.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn entropy_factor_val(&self) -> f32 {
        f32::from_bits(self.entropy_factor.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn adaptive_vvalue_at_depth(&self, depth: u32) -> f32 {
        let depth_decay = 1.0 / (1.0 + depth as f32);
        let sigma = self.sigma_q_val();
        let entropy_factor = self.entropy_factor_val();
        let dr = self.telemetry.dup_rate();
        // B3: when dup_rate == 0 the contention amplifier is exactly 1.0
        // (1 + 0 * (1 + c) = 1 for any finite c). Skip the two extra atomic
        // loads (max_pending + n_threads) in that branch. dup_rate() returns
        // exact 0.0 iff dup_leaf_count == 0, which holds for every
        // single-thread search (record_dup_leaf only fires when another
        // worker is already pending on the leaf) and for the early phase of
        // multi-thread runs before any duplicate path materializes.
        let amplifier = if dr == 0.0 {
            1.0
        } else {
            let mp = self.telemetry.max_pending.load(Ordering::Relaxed) as f32;
            let contention = (mp / self.n_threads.load(Ordering::Relaxed).max(1) as f32).min(2.0);
            1.0 + dr * (1.0 + contention)
        };
        (sigma * depth_decay * entropy_factor * amplifier).max(0.01)
    }

    /// Compute split VL for a select step at given depth.
    ///
    /// State-derived control law with fixed constants:
    /// - vvisit = 1.0 (topological reservation)
    /// - vvalue = σ_Q × depth_decay × entropy_factor × contention_amplifier
    ///
    /// **Feedback loop** (2nd generation):
    /// - σ_Q: Q uncertainty from QUARTZ (one-way read)
    /// - depth_decay: 1/(1+d), root contention most expensive
    /// - entropy_factor: low entropy → less VL needed
    /// - contention_amplifier: f(dup_rate, max_pending, n_threads)
    ///   Combines duplicate frequency with contention severity.
    ///   High dup_rate + high pending → aggressive spread.
    ///   Low dup_rate + low pending → minimal overhead.
    #[inline(always)]
    pub fn vl_at_depth(&self, depth: u32) -> VlSplit {
        match self.mode {
            VlMode::Fixed => VlSplit::FIXED,
            VlMode::Disabled => VlSplit::ZERO,
            VlMode::Adaptive => VlSplit {
                vvisit: 1.0,
                vvalue: self.adaptive_vvalue_at_depth(depth),
            },
            VlMode::VvisitOnly => VlSplit {
                vvisit: 1.0,
                vvalue: 0.0,
            },
            VlMode::VvalueOnly => VlSplit {
                vvisit: 0.0,
                vvalue: self.adaptive_vvalue_at_depth(depth),
            },
        }
    }

    pub fn reset_for_search(&self) {
        self.telemetry.reset();
    }
}

// ───────────────────────────────────────────
// § Tests
// ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn reference_adaptive_vvalue(
        pc: &ParallelismController,
        depth: u32,
        include_vvisit: bool,
    ) -> VlSplit {
        let depth_decay = 1.0 / (1.0 + depth as f32);
        let sigma = pc.sigma_q_val();
        let ent = pc.root_entropy_val();
        let entropy_factor = (ent.min(2.0).max(0.5)) / 2.0;
        let dr = pc.telemetry.dup_rate();
        let mp = pc.telemetry.max_pending.load(Ordering::Relaxed) as f32;
        let contention = (mp / pc.n_threads.load(Ordering::Relaxed).max(1) as f32).min(2.0);
        let amplifier = 1.0 + dr * (1.0 + contention);
        let vvalue = (sigma * depth_decay * entropy_factor * amplifier).max(0.01);
        VlSplit {
            vvisit: if include_vvisit { 1.0 } else { 0.0 },
            vvalue,
        }
    }

    fn thread_input(visits: u32, legal: usize) -> AutoThreadInput {
        AutoThreadInput {
            host_threads: 24,
            remaining_visits: Some(visits),
            root_legal_count: legal,
            pw_enabled: true,
            reusable_select_scratch: false,
        }
    }

    #[test]
    fn test_auto_threads_throughput_uses_host_when_budget_allows() {
        let decision =
            recommend_auto_threads(thread_input(80_000, 82), AutoThreadPolicy::throughput());
        assert_eq!(decision.threads, 24);
        assert_eq!(decision.reason, AutoThreadReason::ThroughputHostCap);
    }

    #[test]
    fn test_auto_threads_caps_tiny_budgets() {
        let one = recommend_auto_threads(thread_input(100, 225), AutoThreadPolicy::throughput());
        let two = recommend_auto_threads(thread_input(256, 225), AutoThreadPolicy::throughput());
        assert_eq!(one.threads, 1);
        assert_eq!(one.reason, AutoThreadReason::TinyBudget);
        assert_eq!(two.threads, 2);
        assert_eq!(two.reason, AutoThreadReason::TinyBudget);
    }

    #[test]
    fn test_auto_threads_quality_caps_low_branching() {
        let decision =
            recommend_auto_threads(thread_input(500_000, 9), AutoThreadPolicy::quality());
        assert_eq!(decision.threads, 4);
        assert_eq!(decision.reason, AutoThreadReason::QualityLowBranchingCap);
    }

    #[test]
    fn test_auto_threads_respects_caller_cap() {
        let decision = recommend_auto_threads(
            thread_input(80_000, 82),
            AutoThreadPolicy::throughput().with_max_threads(8),
        );
        assert_eq!(decision.threads, 8);
        assert_eq!(decision.requested_cap, 8);
    }

    #[test]
    fn test_fixed_vl() {
        let pc = ParallelismController::new(VlMode::Fixed, 4);
        let vl = pc.vl_at_depth(0);
        assert_eq!(vl.vvisit, 1.0);
        assert_eq!(vl.vvalue, 1.0);
        let vl5 = pc.vl_at_depth(5);
        assert_eq!(vl5.vvisit, 1.0); // fixed doesn't decay
    }

    #[test]
    fn test_disabled_vl() {
        let pc = ParallelismController::new(VlMode::Disabled, 4);
        let vl = pc.vl_at_depth(0);
        assert_eq!(vl.vvisit, 0.0);
        assert_eq!(vl.vvalue, 0.0);
    }

    #[test]
    fn test_adaptive_depth_decay() {
        let pc = ParallelismController::new(VlMode::Adaptive, 4);
        pc.update_from_search(0.3, 1.5);

        let vl0 = pc.vl_at_depth(0);
        let vl3 = pc.vl_at_depth(3);
        let vl10 = pc.vl_at_depth(10);

        // vvisit always 1.0
        assert_eq!(vl0.vvisit, 1.0);
        assert_eq!(vl3.vvisit, 1.0);

        // vvalue decays with depth
        assert!(
            vl0.vvalue > vl3.vvalue,
            "depth 0 should have more VL than depth 3"
        );
        assert!(
            vl3.vvalue > vl10.vvalue,
            "depth 3 should have more VL than depth 10"
        );
    }

    #[test]
    fn test_adaptive_sigma_scaling() {
        let pc1 = ParallelismController::new(VlMode::Adaptive, 4);
        pc1.update_from_search(0.1, 1.5);
        let vl_low = pc1.vl_at_depth(0);

        let pc2 = ParallelismController::new(VlMode::Adaptive, 4);
        pc2.update_from_search(0.5, 1.5);
        let vl_high = pc2.vl_at_depth(0);

        assert!(
            vl_high.vvalue > vl_low.vvalue,
            "higher σ_Q should produce larger virtual value loss"
        );
    }

    #[test]
    fn test_entropy_modulation() {
        let pc_low = ParallelismController::new(VlMode::Adaptive, 4);
        pc_low.update_from_search(0.3, 0.2); // low entropy = dominant move
        let vl_low = pc_low.vl_at_depth(0);

        let pc_high = ParallelismController::new(VlMode::Adaptive, 4);
        pc_high.update_from_search(0.3, 2.0); // high entropy = uncertain
        let vl_high = pc_high.vl_at_depth(0);

        assert!(
            vl_high.vvalue > vl_low.vvalue,
            "higher entropy should produce larger VL (more contention risk)"
        );
    }

    #[test]
    fn test_telemetry() {
        let tel = ParallelTelemetry::new();
        tel.record_select(3);
        tel.record_select(5);
        tel.record_dup_leaf();
        let snap = tel.snapshot();
        assert_eq!(snap.total_selects, 2);
        assert_eq!(snap.dup_leaf_count, 1);
        assert_eq!(snap.max_pending, 5);
        assert!((snap.dup_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_telemetry_reset() {
        let tel = ParallelTelemetry::new();
        tel.record_select(4);
        tel.record_dup_leaf();
        tel.record_vvalue(0.25);
        tel.reset();

        let snap = tel.snapshot();
        assert_eq!(snap.total_selects, 0);
        assert_eq!(snap.dup_leaf_count, 0);
        assert_eq!(snap.max_pending, 0);
        assert_eq!(snap.avg_vvalue, 0.0);
        assert_eq!(snap.dup_rate, 0.0);
    }

    #[test]
    fn test_telemetry_vvalue_accumulates_under_concurrency() {
        use std::sync::Arc;
        use std::thread;

        let tel = Arc::new(ParallelTelemetry::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let tel = tel.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    tel.record_select(1);
                    tel.record_vvalue(0.25);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let snap = tel.snapshot();
        assert_eq!(snap.total_selects, 8000);
        assert!((snap.avg_vvalue - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_cached_entropy_factor_matches_reference_formula() {
        let pc = ParallelismController::new(VlMode::Adaptive, 4);
        pc.update_from_search(0.37, 1.73);
        pc.telemetry.record_select(3);
        pc.telemetry.record_dup_leaf();

        let expected = reference_adaptive_vvalue(&pc, 2, true);
        let actual = pc.vl_at_depth(2);
        assert_eq!(actual.vvisit, expected.vvisit);
        assert_eq!(actual.vvalue.to_bits(), expected.vvalue.to_bits());
    }

    #[test]
    #[ignore]
    fn bench_parallel_vl_hot_path() {
        let pc = ParallelismController::new(VlMode::Adaptive, 8);
        pc.update_from_search(0.41, 1.61);
        for pending in [1, 2, 4, 3, 5, 2, 6, 1] {
            pc.telemetry.record_select(pending);
        }
        for _ in 0..3 {
            pc.telemetry.record_dup_leaf();
        }

        let rounds = 2_000_000u32;

        let start = Instant::now();
        let mut ref_acc = 0.0f32;
        for i in 0..rounds {
            let vl = reference_adaptive_vvalue(&pc, i % 8, true);
            ref_acc += black_box(vl.vvalue);
        }
        let ref_ns = start.elapsed().as_nanos();

        let start = Instant::now();
        let mut opt_acc = 0.0f32;
        for i in 0..rounds {
            let vl = pc.vl_at_depth(i % 8);
            opt_acc += black_box(vl.vvalue);
        }
        let opt_ns = start.elapsed().as_nanos();

        eprintln!(
            "\nParallel VL hot path: reference={}ns optimized={}ns speedup={:.2}x acc=({:.3},{:.3})",
            ref_ns,
            opt_ns,
            ref_ns as f64 / opt_ns as f64,
            ref_acc,
            opt_acc
        );
    }
}

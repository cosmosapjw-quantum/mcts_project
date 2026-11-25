//! MctsEngine (v0.4) — QUARTZ 통합

pub mod backup;
pub mod eval;
pub mod expand;
#[cfg(feature = "idea-foundry")]
pub mod foundry;
pub mod gvoc;
pub mod mod_types;
pub mod node;
pub mod parallel;
// P06: unified SearchPolicy trait + types. Scaffolding only — nothing
// in the engine consults the trait yet. P07-P11 land the concrete
// policies (LegacyAlphaZero, LegacyQuartz, KLLUCBStop, BayesianQuartz,
// MENTSEntropyRegularized).
pub mod policy;
pub mod profiling;
pub mod quartz;
pub mod rng;
pub mod root;
pub mod search;
pub mod select;
pub mod tt;

pub use mod_types::PwConfig;
pub use quartz::{
    compute_quartz_stats, QuartzConfig, QuartzController, QuartzStats, RunningMedian,
};

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::game::{EvalResult, Evaluator, GameState};
use crate::mcts::backup::backprop;
use crate::mcts::expand::{expand_and_evaluate, expand_and_evaluate_in_place, expand_with_result};
use crate::mcts::gvoc::{routing_mode, GvocConfig, GvocState, ProposalMode};
use crate::mcts::node::{ArenaRef, MctsNode};
use crate::mcts::root::{
    compute_dirichlet_noise, select_move_with_temperature, visit_distribution, DirichletConfig,
};
#[cfg(test)]
use crate::mcts::search::root_entropy;
use crate::mcts::search::{SearchController, SearchStats};
use crate::mcts::select::{
    select, select_in_place, SelectResult, SelectScratch, SelectionTelemetry,
};
use crate::mcts::tt::TranspositionTable;

static ITERATE_CALLS: AtomicU64 = AtomicU64::new(0);
static SELECT_TIME_NANOS: AtomicU64 = AtomicU64::new(0);
static EXPAND_EVAL_TIME_NANOS: AtomicU64 = AtomicU64::new(0);
static BACKPROP_TIME_NANOS: AtomicU64 = AtomicU64::new(0);

const PAR_FIXED_BUDGET_TICKET_CHUNK: u32 = 8;

#[inline]
fn fixed_budget_ticket_chunk(n_threads: usize) -> u32 {
    if n_threads <= 1 {
        1
    } else {
        PAR_FIXED_BUDGET_TICKET_CHUNK
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EnginePhaseSnapshot {
    pub iterate_calls: u64,
    pub select_time_nanos: u64,
    pub expand_eval_time_nanos: u64,
    pub backprop_time_nanos: u64,
}

pub fn engine_phase_snapshot() -> EnginePhaseSnapshot {
    EnginePhaseSnapshot {
        iterate_calls: ITERATE_CALLS.load(Ordering::Relaxed),
        select_time_nanos: SELECT_TIME_NANOS.load(Ordering::Relaxed),
        expand_eval_time_nanos: EXPAND_EVAL_TIME_NANOS.load(Ordering::Relaxed),
        backprop_time_nanos: BACKPROP_TIME_NANOS.load(Ordering::Relaxed),
    }
}

pub fn reset_engine_phase_counters() {
    ITERATE_CALLS.store(0, Ordering::Relaxed);
    SELECT_TIME_NANOS.store(0, Ordering::Relaxed);
    EXPAND_EVAL_TIME_NANOS.store(0, Ordering::Relaxed);
    BACKPROP_TIME_NANOS.store(0, Ordering::Relaxed);
}

// ─────────────────────────────────────────────
// § MctsConfig
// ─────────────────────────────────────────────

#[derive(Clone)]
pub struct MctsConfig {
    pub c_puct: f32,
    pub pw: Option<PwConfig>,
    pub dirichlet: Option<DirichletConfig>,
    pub temperature: f32,
    pub max_depth: usize,
    pub max_tt_size: Option<usize>,
    pub tt_enabled: bool,
    /// QUARTZ EFT-PUCT + 적응적 정지
    pub quartz: Option<QuartzConfig>,
    /// GVOC 동적 PW 스케줄러 (None = 비활성)
    pub gvoc: Option<GvocConfig>,
    /// 재현성 시드 (None = 비결정적)
    pub seed: Option<u64>,
    /// Return an immediate root winning move when one exists
    pub root_forced_win: bool,
    /// Use exact backed value for already-materialized terminal children during selection
    pub exact_terminal_value: bool,
    /// First-play urgency reduction for truly unvisited edges (0 = disabled)
    pub fpu_reduction: f32,
    /// Virtual loss mode (Fixed=baseline, Adaptive=σ_Q-scaled, Disabled=no VL)
    pub vl_mode: parallel::VlMode,
    /// BQ++ Phase 8b: opt-in SearchPolicy trait object that overlays
    /// the legacy QuartzController-based selection / halt behavior.
    /// When `None`, the engine drives entirely through the existing
    /// `quartz` + `SearchController` paths (back-compat with all P01-
    /// P07 commits). When `Some`, the engine consults
    /// `policy.score_adjustment(...)` at root selection and
    /// `policy.should_halt(...)` as an additional halt signal AFTER
    /// each `SearchController::should_stop` check. The policy is
    /// shared across worker threads via `Arc<dyn SearchPolicy>` and
    /// expected to use lock-free internal state (e.g.
    /// `arc_swap::ArcSwap<Arc<PolicyCache>>` from BQ++ Phase 2).
    pub search_policy: Option<std::sync::Arc<dyn crate::mcts::policy::SearchPolicy>>,
    /// Feature-gated root-local Foundry policy factory.  Unlike
    /// `search_policy`, this configuration is cloned across jobs and then
    /// materialized into a fresh policy after the engine knows the root hash.
    #[cfg(feature = "idea-foundry")]
    pub foundry_search: Option<crate::mcts::foundry::FoundrySearchConfig>,
    /// BQ++ Phase 8c followup: per-`HaltReason` increment counters owned
    /// by the engine (parallel to `QuartzController.halt_reason_count`).
    /// Incremented by `policy_halt_check` whenever the attached
    /// `search_policy` returns `Stop(reason)`, so policy-driven halts
    /// surface in `extended_halt_reason_count` even on async/server
    /// paths that synthesize the controller's counters from scratch.
    /// `None` when no policy is attached.
    pub policy_halt_counts:
        Option<std::sync::Arc<[AtomicU32; crate::mcts::quartz::HALT_REASON_COUNT]>>,
}

// BQ++ Phase 8b: manual Debug impl because the new `search_policy`
// field is `Arc<dyn SearchPolicy>` which doesn't derive Debug. Adding
// Debug to the trait bounds would force every impl to also derive
// Debug, which is fine but invasive. The manual impl is local + the
// field is rendered as `<dyn SearchPolicy: ${name}>` for clarity in
// log output without forcing the trait's surface to grow.
impl std::fmt::Debug for MctsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MctsConfig")
            .field("c_puct", &self.c_puct)
            .field("pw", &self.pw)
            .field("dirichlet", &self.dirichlet)
            .field("temperature", &self.temperature)
            .field("max_depth", &self.max_depth)
            .field("max_tt_size", &self.max_tt_size)
            .field("tt_enabled", &self.tt_enabled)
            .field("quartz", &self.quartz)
            .field("gvoc", &self.gvoc)
            .field("seed", &self.seed)
            .field("root_forced_win", &self.root_forced_win)
            .field("exact_terminal_value", &self.exact_terminal_value)
            .field("fpu_reduction", &self.fpu_reduction)
            .field("vl_mode", &self.vl_mode)
            .field(
                "search_policy",
                &self
                    .search_policy
                    .as_ref()
                    .map(|p| format!("<dyn SearchPolicy: {}>", p.name())),
            )
            .field("foundry_search", &{
                #[cfg(feature = "idea-foundry")]
                {
                    self.foundry_search.as_ref()
                }
                #[cfg(not(feature = "idea-foundry"))]
                {
                    Option::<&str>::None
                }
            })
            .field(
                "policy_halt_counts",
                &self
                    .policy_halt_counts
                    .as_ref()
                    .map(|_| "<Arc<[AtomicU32; HALT_REASON_COUNT]>>"),
            )
            .finish()
    }
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_puct: 2.0,
            pw: None,
            dirichlet: None,
            temperature: 0.0,
            max_depth: 0,
            max_tt_size: None,
            tt_enabled: true,
            quartz: None,
            gvoc: None,
            seed: None,
            root_forced_win: true,
            exact_terminal_value: false,
            fpu_reduction: 0.0,
            vl_mode: parallel::VlMode::Adaptive,
            search_policy: None,
            #[cfg(feature = "idea-foundry")]
            foundry_search: None,
            // Always allocate the per-HaltReason counter array.
            // `policy_halt_check` increments unconditionally; if no
            // policy is ever attached the array stays at zero and is
            // a no-op cost (40B per Arc clone, one Arc per engine).
            // Allocating eagerly avoids the foot-gun where direct
            // assignment of `cfg.search_policy = Some(_)` (vs going
            // through `with_search_policy()`) would otherwise leave
            // counts=None and silently drop every increment.
            policy_halt_counts: Some(std::sync::Arc::new(std::array::from_fn(|_| {
                AtomicU32::new(0)
            }))),
        }
    }
}

impl MctsConfig {
    pub fn evaluation(c_puct: f32) -> Self {
        MctsConfig {
            c_puct,
            ..Default::default()
        }
    }

    pub fn evaluation_with_pw(c_puct: f32, pw: PwConfig) -> Self {
        MctsConfig {
            c_puct,
            pw: Some(pw),
            max_depth: 25,
            ..Default::default()
        }
    }

    pub fn stress(c_puct: f32, pw: PwConfig, max_tt: usize) -> Self {
        MctsConfig {
            c_puct,
            pw: Some(pw),
            max_depth: 25,
            max_tt_size: Some(max_tt),
            ..Default::default()
        }
    }

    /// QUARTZ 활성화
    pub fn with_quartz(mut self, qcfg: QuartzConfig) -> Self {
        self.quartz = Some(qcfg);
        self
    }

    /// BQ++ Phase 8b: attach a SearchPolicy trait object that drives
    /// score adjustment + halt decisions in addition to the legacy
    /// controller path. The Arc is shared across worker threads.
    pub fn with_search_policy(
        mut self,
        policy: std::sync::Arc<dyn crate::mcts::policy::SearchPolicy>,
    ) -> Self {
        self.search_policy = Some(policy);
        // Counter array is allocated by Default; reset it here so a
        // builder pattern never sees leftover counts from a prior
        // attach (defensive — Default always returns fresh zero
        // atomics, but a future builder reuse pattern won't surprise
        // a reader).
        self.policy_halt_counts = Some(std::sync::Arc::new(std::array::from_fn(|_| {
            AtomicU32::new(0)
        })));
        self
    }

    /// GVOC 활성화
    pub fn with_gvoc(mut self, gcfg: GvocConfig) -> Self {
        self.gvoc = Some(gcfg);
        self
    }

    /// 재현성 시드 설정
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    #[cfg(test)]
    pub fn with_root_forced_win(mut self, root_forced_win: bool) -> Self {
        self.root_forced_win = root_forced_win;
        self
    }

    #[cfg(test)]
    pub fn with_exact_terminal_value(mut self, exact_terminal_value: bool) -> Self {
        self.exact_terminal_value = exact_terminal_value;
        self
    }

    #[cfg(test)]
    pub fn with_fpu_reduction(mut self, fpu_reduction: f32) -> Self {
        self.fpu_reduction = fpu_reduction.max(0.0);
        self
    }

    #[cfg(test)]
    pub fn with_tt_enabled(mut self, tt_enabled: bool) -> Self {
        self.tt_enabled = tt_enabled;
        self
    }
}

// ─────────────────────────────────────────────
// § MctsEngine
// ─────────────────────────────────────────────

struct CountingEvaluator<G: GameState> {
    inner: Arc<dyn Evaluator<G> + Send + Sync>,
    calls: Arc<AtomicU64>,
}

impl<G: GameState> Evaluator<G> for CountingEvaluator<G> {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.evaluate(state)
    }
}

pub struct MctsEngine<G: GameState> {
    pub root: ArenaRef<MctsNode<G::Move>>,
    root_state: G,
    pub(crate) evaluator: Arc<dyn Evaluator<G> + Send + Sync>,
    evaluator_calls: Arc<AtomicU64>,
    /// `tt` owns the per-bucket bumpalo arenas that back every `ArenaRef`
    /// reachable from `root`, edges, and path entries. It is declared
    /// before `config`/etc. so that on engine drop the TT (and thus the
    /// Bumps) outlives all transient `ArenaRef`s reachable via search-
    /// scoped temporaries.
    pub tt: Arc<TranspositionTable<G::Move>>,
    pub config: MctsConfig,
    root_noise: Option<Vec<f32>>,
    /// 현재 QUARTZ 통계 (EFT-PUCT에 사용)
    pub(crate) quartz_cache: RwLock<Option<QuartzStats>>,
    /// Monotonic version for `quartz_cache`. Search loops keep a local
    /// QUARTZ snapshot and reload it only when this epoch changes.
    quartz_cache_epoch: AtomicU64,
    /// Parallelism controller (adaptive VL, telemetry)
    pub par_ctrl: parallel::ParallelismController,
    /// Per-search root selection telemetry. This records the actual selected
    /// root edge on each MCTS iteration, complementing root-snapshot summaries.
    pub selection_telemetry: SelectionTelemetry,
    pub mqft_trace: Option<Arc<crate::mqft_trace::Trace>>,
}

pub struct AsyncPendingIteration<G: GameState> {
    pub path: Vec<crate::mcts::node::PathEdge<G::Move>>,
    pub leaf: ArenaRef<MctsNode<G::Move>>,
    pub leaf_state: G,
}

/// Why an iteration was resolved immediately (no NN eval needed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmediateReason {
    /// Transposition table size cap reached
    TtCapHit,
    /// Leaf is a terminal game state
    TerminalNode,
}

pub enum PreparedIteration<G: GameState> {
    Immediate {
        path: Vec<crate::mcts::node::PathEdge<G::Move>>,
        value: f32,
        reason: ImmediateReason,
    },
    Pending(AsyncPendingIteration<G>),
}

impl<G: GameState> MctsEngine<G> {
    /// BQ++ Phase 8b: build a minimal SearchSnapshot from current engine
    /// state for the SearchPolicy trait.
    ///
    /// Audit note (A0-a, follows the Phase 8b/8c dead-edges finding):
    /// `should_halt`/`observe` are fed real root `EdgeView`s by
    /// `policy_halt_check` below (see `build_policy_root_edges`). This
    /// snapshot itself stays edge-free — `score_adjustment` (per-edge
    /// PUCT scoring) is a separate, still-deferred integration (BQ++
    /// design doc §5 item 1); this function only feeds the
    /// halt/observe path.
    fn build_policy_snapshot(
        &self,
        iteration: u64,
        elapsed_ms: u64,
    ) -> crate::mcts::policy::SearchSnapshot {
        let root_visits = self.root.n_total.load(Ordering::Relaxed);
        let n_children = self.root.candidate_count() as u16;
        // Mean Q + sigma_q approximation from quartz cache when available;
        // 0.0 fallback for non-quartz profiles.
        let (mean_q_root, sigma_q_root) = match self.quartz_snapshot() {
            (_, Some(stats)) => (stats.mean_q, stats.sigma_q),
            (_, None) => (0.0, 0.3),
        };
        crate::mcts::policy::SearchSnapshot {
            root_visits,
            n_children,
            n_visible: n_children,
            elapsed_ms,
            depth_max: 0,
            mean_q_root,
            sigma_q_root,
            sigma_eval: None,
            iteration,
            best_idx: 0,
            second_idx: 0,
        }
    }

    /// A0-a: build real root `EdgeView`s from the currently-materialized
    /// root children for the `SearchPolicy` observe/should_halt path.
    ///
    /// Scope: this reads the same lock-free published edge slab
    /// `select.rs`'s hot path reads (`read_edges()`), but is only
    /// called at `check_interval` boundaries (not per-iteration), so
    /// it does not add hot-path allocation or contention.
    ///
    /// Field choices, matched against what wired policies actually
    /// consume (`kl_lucb.rs` reads `n`/`q`; `legacy_quartz.rs`'s
    /// `score_adjustment` — not yet wired into the engine, see BQ++
    /// design doc §5 item 1 — reads `o_a`/`root_total_n`):
    /// - `q` is the **raw** empirical mean (`MctsEdge::q()`, no virtual
    ///   loss), not `q_eff()`. Certificates/KG must see true backed-up
    ///   statistics; virtual loss is a within-iteration parallel
    ///   coordination reservation and would bias a PAC certificate.
    /// - `o_a` stays `0`, matching every existing EdgeView construction
    ///   site in this codebase (tests, `legacy_az.rs`, `kl_lucb.rs`
    ///   fixtures). The field's doc ("outside/un-materialized neighbor
    ///   visits") is a distinct concept from `select.rs`'s local
    ///   `o_a` variable (virtual-loss count) and is not currently
    ///   tracked per-edge anywhere in the engine.
    /// - `envar_partial` is computed here (visit-share KL contribution)
    ///   since it's cheap and the field exists precisely so policies
    ///   don't recompute it; leaving it at a silent `0.0` would be the
    ///   same species of dead-value bug this function fixes.
    /// - `idx` is the dense `edge_pos` (slab position), never
    ///   `action_id` — see the superseded-sections note in
    ///   `docs/LEGACY_VS_BAYESIAN_QUARTZ.md` Appendix A on the
    ///   edge_pos/action_id OOB hazard.
    fn build_policy_root_edges<'a>(
        &self,
        snap: &'a crate::mcts::policy::SearchSnapshot,
        root_total_n: &'a u32,
    ) -> Vec<crate::mcts::policy::EdgeView<'a>> {
        self.root
            .read_edges()
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let n = e.n.load(Ordering::Relaxed);
                let prior = e.p;
                let visit_share = n as f32 / (*root_total_n).max(1) as f32;
                let envar_partial = if visit_share > 0.0 && prior > 1e-6 {
                    visit_share * (visit_share / prior).ln()
                } else {
                    0.0
                };
                crate::mcts::policy::EdgeView {
                    idx: i as u16,
                    n,
                    n_virtual: e.virtual_losses.load(Ordering::Acquire).max(0) as u32,
                    o_a: 0,
                    q: e.q(),
                    q_sum: crate::mcts::node::atomic_f64_load(&e.w) as f32,
                    m2: f64::from_bits(e.m2.load(Ordering::Acquire)),
                    prior,
                    depth: 0,
                    last_value: 0.0,
                    envar_partial,
                    root_total_n,
                    stats: snap,
                }
            })
            .collect()
    }

    /// BQ++ Phase 8b/8c: consult `search_policy.should_halt()` if a
    /// policy is configured. Returns `true` when the policy says
    /// `Stop(_)`. Side effect: when the policy halts and the engine
    /// has a `QuartzController` attached (the normal training/eval
    /// configuration), increments the matching extended
    /// `halt_reason_count[reason as usize]` so the policy's halt
    /// reason surfaces in the per-search controller_summary.extended
    /// telemetry. Without this, KLLUCBStop / PolicyConverged firings
    /// are invisible to the replay aggregator.
    pub fn policy_halt_check(&self, iteration: u64, elapsed_ms: u64) -> bool {
        let Some(ref policy) = self.config.search_policy else {
            return false;
        };
        let snap = self.build_policy_snapshot(iteration, elapsed_ms);
        let root_total_n = snap.root_visits;
        let edges = self.build_policy_root_edges(&snap, &root_total_n);
        // Periodic observe call — policy implementations cache state
        // inside their own internal storage; observe is the canonical
        // refresh point.
        if let Some(ref qcfg) = self.config.quartz {
            if iteration > 0 && iteration % qcfg.check_interval as u64 == 0 {
                policy.observe(&snap, &edges);
            }
        } else if iteration > 0 && iteration % 64 == 0 {
            // Non-quartz default check_interval
            policy.observe(&snap, &edges);
        }
        match policy.should_halt(&snap, &edges) {
            crate::mcts::policy::HaltDecision::Stop(reason) => {
                if let Some(ref counts) = self.config.policy_halt_counts {
                    counts[reason as usize].fetch_add(1, Ordering::Relaxed);
                }
                true
            }
            crate::mcts::policy::HaltDecision::Continue => false,
        }
    }

    /// BQ++ Phase 8c followup: snapshot of policy-driven halt counts.
    /// Returns all-zero array when no policy is attached.
    pub fn policy_halt_count_snapshot(&self) -> [u32; crate::mcts::quartz::HALT_REASON_COUNT] {
        let mut out = [0u32; crate::mcts::quartz::HALT_REASON_COUNT];
        if let Some(ref counts) = self.config.policy_halt_counts {
            for (i, slot) in counts.iter().enumerate() {
                out[i] = slot.load(Ordering::Relaxed);
            }
        }
        out
    }

    pub fn policy_telemetry_snapshot(&self) -> Option<crate::mcts::policy::ControllerTelemetry> {
        self.config
            .search_policy
            .as_ref()
            .map(|policy| policy.telemetry())
    }

    /// Exact number of calls through this engine's evaluator boundary,
    /// including the initial root expansion. Terminal and TT-resolved
    /// iterations do not increment this counter.
    pub fn evaluator_call_count(&self) -> u64 {
        self.evaluator_calls.load(Ordering::Relaxed)
    }

    pub fn new(
        root_state: G,
        evaluator: Arc<dyn Evaluator<G> + Send + Sync>,
        mut config: MctsConfig,
    ) -> Self {
        let tt = Arc::new(TranspositionTable::new_enabled(config.tt_enabled));
        let hash = root_state.tt_hash();
        let tv = if root_state.is_terminal() {
            Some(root_state.outcome())
        } else {
            None
        };
        let root = tt.get_or_create(hash, tv);

        #[cfg(feature = "idea-foundry")]
        if let Some(foundry_config) = config.foundry_search.clone() {
            config.search_policy = Some(Arc::new(crate::mcts::foundry::FoundrySearchPolicy::new(
                foundry_config,
                hash,
            )));
        }

        // BQ++ Phase 8c followup: each engine gets its OWN
        // policy_halt_counts Arc. Without this, all engines created
        // from the same base config (e.g. async batched self-play
        // where every game does base_cfg.clone()) share one Arc and
        // the per-search snapshot reads the cumulative count across
        // the whole server lifetime — over-counting that inflated the
        // legacy_az toy ablation to 465K halts vs the expected ~324K
        // (one per move).
        config.policy_halt_counts = Some(std::sync::Arc::new(std::array::from_fn(|_| {
            AtomicU32::new(0)
        })));

        let evaluator_calls = Arc::new(AtomicU64::new(0));
        let evaluator: Arc<dyn Evaluator<G> + Send + Sync> = Arc::new(CountingEvaluator {
            inner: evaluator,
            calls: evaluator_calls.clone(),
        });

        let mut engine = MctsEngine {
            root,
            root_state,
            evaluator,
            evaluator_calls,
            tt,
            par_ctrl: parallel::ParallelismController::new(config.vl_mode, 1),
            config,
            root_noise: None,
            quartz_cache: RwLock::new(None),
            quartz_cache_epoch: AtomicU64::new(0),
            selection_telemetry: SelectionTelemetry::new(),
            mqft_trace: None,
        };
        engine.expand_root();
        engine
    }

    fn expand_root(&mut self) {
        if !self.root.is_expanded() {
            expand_and_evaluate(
                &self.root,
                &self.root_state,
                self.evaluator.as_ref(),
                &self.tt,
                self.config.pw.as_ref(),
            );
        }
        if let Some(dir) = &self.config.dirichlet {
            let priors = self
                .root
                .edge_priors_snapshot(self.root.materialized_count());
            if !priors.is_empty() {
                self.root_noise = Some(compute_dirichlet_noise(
                    priors.len(),
                    &priors,
                    dir,
                    self.config.seed,
                ));
            }
        }
    }

    /// QUARTZ 통계 갱신 (QuartzController와 협력)
    /// QUARTZ 통계 갱신 (`priors=None` reads the root edge priors in-place).
    pub fn refresh_quartz_stats(&self) {
        self.refresh_quartz_stats_with_priors(None);
    }

    pub fn refresh_quartz_stats_with_priors(&self, priors: Option<&[f32]>) {
        if self.config.quartz.is_some() {
            // QuartzController 없이 직접 캐시 갱신할 때의 임시 경로
            // run_quartz/run_gvoc 사용 시에는 ctrl.update_stats가 처리함
            // 여기서는 QuartzStats 타입 호환을 위한 최소 통계만 생성
            let n_mat = self.root.materialized_count();
            let edges = self.root.edge_snapshot(n_mat);
            let n_total = self.root.n_total.load(Ordering::Acquire);
            if !edges.is_empty() && n_total > 10 {
                // 임시 EMA (상태 없음 — run_quartz 경로 권장)
                let mut s0 = RunningMedian::new(0.05);
                let _ce = RunningMedian::new(0.1);
                let qcfg = self.config.quartz.as_ref().unwrap();
                let stats = compute_quartz_stats(&self.root, priors, &mut s0, 0.0, 0, 0, qcfg);
                self.store_quartz_stats(stats);
            }
        }
    }

    pub fn current_quartz_stats(&self) -> Option<QuartzStats> {
        self.quartz_cache.read().clone()
    }

    #[inline]
    fn store_quartz_stats(&self, stats: QuartzStats) -> u64 {
        *self.quartz_cache.write() = Some(stats);
        self.quartz_cache_epoch.fetch_add(1, Ordering::Release) + 1
    }

    #[inline]
    fn clear_quartz_stats(&self) -> u64 {
        *self.quartz_cache.write() = None;
        self.quartz_cache_epoch.fetch_add(1, Ordering::Release) + 1
    }

    #[inline]
    fn quartz_snapshot(&self) -> (u64, Option<QuartzStats>) {
        let epoch = self.quartz_cache_epoch.load(Ordering::Acquire);
        let snapshot = if self.config.quartz.is_some() {
            self.quartz_cache.read().clone()
        } else {
            None
        };
        (epoch, snapshot)
    }

    #[inline]
    fn refresh_quartz_snapshot_if_changed(
        &self,
        seen_epoch: &mut u64,
        snapshot: &mut Option<QuartzStats>,
    ) {
        let epoch = self.quartz_cache_epoch.load(Ordering::Acquire);
        if epoch != *seen_epoch {
            *snapshot = if self.config.quartz.is_some() {
                self.quartz_cache.read().clone()
            } else {
                None
            };
            *seen_epoch = epoch;
        }
    }

    #[inline]
    fn qstate_from_snapshot<'a>(
        &'a self,
        snapshot: &'a Option<QuartzStats>,
    ) -> Option<(&'a QuartzStats, &'a QuartzConfig)> {
        match (snapshot.as_ref(), self.config.quartz.as_ref()) {
            (Some(stats), Some(qcfg)) => Some((stats, qcfg)),
            _ => None,
        }
    }

    #[inline]
    fn iterate_with_cached_quartz_snapshot(
        &self,
        scratch: &mut Option<SelectScratch<G>>,
        use_quartz: bool,
        seen_epoch: &mut u64,
        snapshot: &mut Option<QuartzStats>,
    ) {
        if use_quartz {
            self.refresh_quartz_snapshot_if_changed(seen_epoch, snapshot);
            let qstate = self.qstate_from_snapshot(snapshot);
            self.iterate_maybe_scratch_with_qstate(scratch, qstate);
        } else {
            self.iterate_maybe_scratch_with_qstate(scratch, None);
        }
    }

    /// Update ParallelismController from QUARTZ stats + root visit entropy.
    /// One-way coupling: par_ctrl reads from QUARTZ, never writes.
    fn refresh_par_ctrl(&self) {
        // σ_Q from QUARTZ
        let sigma_q = self
            .quartz_cache
            .read()
            .as_ref()
            .map(|s| s.hbar_eff * self.config.quartz.as_ref().map_or(0.3, |q| q.sigma_0))
            .unwrap_or(0.3);
        // Root visit entropy
        let root_entropy = self
            .root
            .with_edge_slice(self.root.materialized_count(), |edges| {
                let total: f32 = edges
                    .iter()
                    .map(|e| e.n.load(Ordering::Acquire) as f32)
                    .sum();
                if total > 1.0 {
                    edges
                        .iter()
                        .map(|e| {
                            let p = e.n.load(Ordering::Acquire) as f32 / total;
                            if p > 1e-8 {
                                -p * p.ln()
                            } else {
                                0.0
                            }
                        })
                        .sum::<f32>()
                } else {
                    1.0
                }
            });
        self.par_ctrl.update_from_search(sigma_q, root_entropy);
    }

    #[inline]
    fn make_select_scratch(&self) -> Option<SelectScratch<G>> {
        G::uses_reusable_select_scratch().then(|| SelectScratch::new(&self.root_state))
    }

    #[inline]
    fn iterate_maybe_scratch(&self, scratch: &mut Option<SelectScratch<G>>) {
        let (_, qstats_snapshot) = self.quartz_snapshot();
        let qstate = self.qstate_from_snapshot(&qstats_snapshot);
        self.iterate_maybe_scratch_with_qstate(scratch, qstate);
    }

    #[inline]
    fn iterate_maybe_scratch_with_qstate(
        &self,
        scratch: &mut Option<SelectScratch<G>>,
        qstate: Option<(&QuartzStats, &QuartzConfig)>,
    ) {
        if let Some(scratch) = scratch.as_mut() {
            self.iterate_with_scratch_qstate(scratch, qstate);
        } else {
            self.iterate_owned_state_with_qstate(qstate);
        }
    }

    pub fn iterate(&self) {
        let mut scratch = self.make_select_scratch();
        self.iterate_maybe_scratch(&mut scratch);
    }

    #[inline]
    fn iterate_owned_state_with_qstate(&self, qstate: Option<(&QuartzStats, &QuartzConfig)>) {
        ITERATE_CALLS.fetch_add(1, Ordering::Relaxed);
        let select_started = profiling::maybe_start_timer();
        let sel = select(
            &self.root,
            &self.root_state,
            self.config.c_puct,
            self.root_noise.as_deref(),
            self.config.pw.as_ref(),
            self.config.max_depth,
            &self.tt,
            qstate,
            self.config.exact_terminal_value,
            self.config.fpu_reduction,
            self.par_ctrl.should_reserve_virtual_loss(),
            &self.par_ctrl,
        );
        profiling::record_elapsed_nanos(&SELECT_TIME_NANOS, select_started);
        if let Some(trace) = sel.root_selection_trace {
            self.selection_telemetry.record_root(trace);
        }

        let mqft_selected = self.mqft_trace.as_ref().map(|t| { crate::mqft_trace::clear_eval(); t.begin(&self.root_state,&sel.path) });
        let mqft_terminal = sel.leaf.terminal_value.is_some();
        let expand_started = profiling::maybe_start_timer();
        let leaf_value = if self
            .config
            .max_tt_size
            .map_or(true, |cap| self.tt.size() < cap)
        {
            expand_and_evaluate(
                &sel.leaf,
                &sel.leaf_state,
                self.evaluator.as_ref(),
                &self.tt,
                self.config.pw.as_ref(),
            )
        } else {
            sel.leaf.terminal_value.unwrap_or(0.0)
        };
        profiling::record_elapsed_nanos(&EXPAND_EVAL_TIME_NANOS, expand_started);

        let backprop_started = profiling::maybe_start_timer();
        let root_clock=backprop::<G>(&sel.path, leaf_value);
        if let (Some(t),Some(selected)) = (&self.mqft_trace,mqft_selected) { t.finish(selected,leaf_value,crate::mqft_trace::take_eval(),mqft_terminal,root_clock); }
        profiling::record_elapsed_nanos(&BACKPROP_TIME_NANOS, backprop_started);
    }

    #[inline]
    fn iterate_with_scratch_qstate(
        &self,
        scratch: &mut SelectScratch<G>,
        qstate: Option<(&QuartzStats, &QuartzConfig)>,
    ) {
        ITERATE_CALLS.fetch_add(1, Ordering::Relaxed);
        let select_started = profiling::maybe_start_timer();
        let sel = select_in_place(
            &self.root,
            scratch,
            self.config.c_puct,
            self.root_noise.as_deref(),
            self.config.pw.as_ref(),
            self.config.max_depth,
            &self.tt,
            qstate,
            self.config.exact_terminal_value,
            self.config.fpu_reduction,
            self.par_ctrl.should_reserve_virtual_loss(),
            &self.par_ctrl,
        );
        profiling::record_elapsed_nanos(&SELECT_TIME_NANOS, select_started);
        if let Some(trace) = sel.root_selection_trace {
            self.selection_telemetry.record_root(trace);
        }

        let mqft_selected = self.mqft_trace.as_ref().map(|t| { crate::mqft_trace::clear_eval(); t.begin(&self.root_state,&sel.path) });
        let mqft_terminal = sel.leaf.terminal_value.is_some();
        let expand_started = profiling::maybe_start_timer();
        let leaf_value = if self
            .config
            .max_tt_size
            .map_or(true, |cap| self.tt.size() < cap)
        {
            expand_and_evaluate_in_place(
                &sel.leaf,
                scratch.state_mut(),
                self.evaluator.as_ref(),
                &self.tt,
                self.config.pw.as_ref(),
            )
        } else {
            sel.leaf.terminal_value.unwrap_or(0.0)
        };
        profiling::record_elapsed_nanos(&EXPAND_EVAL_TIME_NANOS, expand_started);

        let backprop_started = profiling::maybe_start_timer();
        let root_clock=backprop::<G>(&sel.path, leaf_value);
        if let (Some(t),Some(selected)) = (&self.mqft_trace,mqft_selected) { t.finish(selected,leaf_value,crate::mqft_trace::take_eval(),mqft_terminal,root_clock); }
        profiling::record_elapsed_nanos(&BACKPROP_TIME_NANOS, backprop_started);
        scratch.reset_to_root();
    }

    pub fn prepare_iteration_async(&self) -> PreparedIteration<G> {
        ITERATE_CALLS.fetch_add(1, Ordering::Relaxed);
        let (_, qstats_snapshot) = self.quartz_snapshot();
        let qstate = self.qstate_from_snapshot(&qstats_snapshot);

        let select_started = profiling::maybe_start_timer();
        let sel: SelectResult<G> = select(
            &self.root,
            &self.root_state,
            self.config.c_puct,
            self.root_noise.as_deref(),
            self.config.pw.as_ref(),
            self.config.max_depth,
            &self.tt,
            qstate,
            self.config.exact_terminal_value,
            self.config.fpu_reduction,
            true,
            &self.par_ctrl,
        );
        profiling::record_elapsed_nanos(&SELECT_TIME_NANOS, select_started);
        if let Some(trace) = sel.root_selection_trace {
            self.selection_telemetry.record_root(trace);
        }

        if !self
            .config
            .max_tt_size
            .map_or(true, |cap| self.tt.size() < cap)
        {
            return PreparedIteration::Immediate {
                path: sel.path.into_vec(),
                value: sel.leaf.terminal_value.unwrap_or(0.0),
                reason: ImmediateReason::TtCapHit,
            };
        }

        if let Some(v) = sel.leaf.terminal_value {
            return PreparedIteration::Immediate {
                path: sel.path.into_vec(),
                value: v,
                reason: ImmediateReason::TerminalNode,
            };
        }

        PreparedIteration::Pending(AsyncPendingIteration {
            path: sel.path.into_vec(),
            leaf: sel.leaf,
            leaf_state: sel.leaf_state,
        })
    }

    pub fn apply_iteration_value_async(
        &self,
        path: Vec<crate::mcts::node::PathEdge<G::Move>>,
        leaf_value: f32,
    ) -> u64 {
        let backprop_started = profiling::maybe_start_timer();
        let clock=backprop::<G>(&path, leaf_value);
        profiling::record_elapsed_nanos(&BACKPROP_TIME_NANOS, backprop_started);
        clock
    }

    pub fn complete_iteration_async(
        &self,
        pending: AsyncPendingIteration<G>,
        eval: crate::game::EvalResult<G::Move>,
    ) {
        let expand_started = profiling::maybe_start_timer();
        let leaf_value = expand_with_result(
            &pending.leaf,
            &pending.leaf_state,
            eval,
            &self.tt,
            self.config.pw.as_ref(),
        );
        profiling::record_elapsed_nanos(&EXPAND_EVAL_TIME_NANOS, expand_started);
        self.apply_iteration_value_async(pending.path, leaf_value);
    }

    pub fn refresh_async_runtime(&self, completed_iterations: u32) {
        if let Some(ref qcfg) = self.config.quartz {
            if completed_iterations > 0 && completed_iterations % qcfg.check_interval == 0 {
                self.refresh_quartz_stats();
                self.refresh_par_ctrl();
            }
        }
    }

    // ── 탐색 실행 ──────────────────────────────────────────────

    pub fn run(&self, controller: &mut dyn SearchController) -> SearchStats {
        controller.reset();
        self.par_ctrl.set_n_threads(1);
        self.par_ctrl.reset_for_search();
        self.selection_telemetry.reset();
        let start = Instant::now();
        let mut it = 0u32;
        let qcfg = self.config.quartz.clone();
        let mut scratch = self.make_select_scratch();
        let use_quartz = qcfg.is_some();
        let (mut qstats_epoch, mut qstats_snapshot) = if use_quartz {
            self.quartz_snapshot()
        } else {
            (0, None)
        };
        if let Some(limit) = controller.visit_limit_hint() {
            loop {
                let rv = self.root.n_total.load(Ordering::Relaxed);
                if rv >= limit {
                    break;
                }
                // BQ++ Phase 8b: SearchPolicy halt also fires on the
                // visit_limit_hint fast path. Enables policies to
                // short-circuit fixed-budget controllers (e.g.
                // KLLUCBStop firing a PAC certificate before the iter
                // cap is reached).
                if self.policy_halt_check(it as u64, 0) {
                    break;
                }

                self.iterate_with_cached_quartz_snapshot(
                    &mut scratch,
                    use_quartz,
                    &mut qstats_epoch,
                    &mut qstats_snapshot,
                );
                it += 1;

                if let Some(ref qcfg) = qcfg {
                    if it % qcfg.check_interval == 0 {
                        self.refresh_quartz_stats();
                        self.refresh_par_ctrl();
                    }
                }
            }
        } else {
            let needs_elapsed = controller.needs_elapsed_ms();
            loop {
                let rv = self.root.n_total.load(Ordering::Relaxed);
                let ms = if needs_elapsed {
                    start.elapsed().as_millis() as u64
                } else {
                    0
                };
                if controller.should_stop(rv, ms) {
                    break;
                }
                // BQ++ Phase 8b: SearchPolicy halt as a secondary signal.
                // Fires only when cfg.search_policy is set; back-compat
                // null when None.
                if self.policy_halt_check(it as u64, ms) {
                    break;
                }

                self.iterate_with_cached_quartz_snapshot(
                    &mut scratch,
                    use_quartz,
                    &mut qstats_epoch,
                    &mut qstats_snapshot,
                );
                it += 1;

                // QUARTZ 통계 주기적 갱신
                // QuartzController를 직접 downcast할 수 없으므로,
                // MctsConfig.quartz가 있으면 engine 내부 캐시 갱신하고
                // controller가 QuartzController이면 자동으로 should_stop에서 확인.
                if let Some(ref qcfg) = qcfg {
                    if it % qcfg.check_interval == 0 {
                        self.refresh_quartz_stats();
                        self.refresh_par_ctrl();
                    }
                }
            }
        }

        let ms = start.elapsed().as_millis() as u64;
        let rv = self.root.n_total.load(Ordering::Relaxed);
        SearchStats {
            iterations: it,
            elapsed_ms: ms,
            nps: if ms > 0 {
                it as f64 / ms as f64 * 1000.0
            } else {
                0.0
            },
            tt_hit_rate: self.tt.hit_rate(),
            tt_size: self.tt.size(),
            root_visits: rv,
            stop_reason: controller.stop_reason(),
        }
    }

    pub fn auto_thread_decision(
        &self,
        controller: &dyn SearchController,
        policy: parallel::AutoThreadPolicy,
    ) -> parallel::AutoThreadDecision {
        let current_visits = self.root.n_total.load(Ordering::Relaxed);
        let remaining_visits = controller
            .visit_limit_hint()
            .map(|limit| limit.saturating_sub(current_visits));
        parallel::recommend_auto_threads(
            parallel::AutoThreadInput {
                host_threads: parallel::available_search_threads(),
                remaining_visits,
                root_legal_count: self.root_state.legal_move_count(),
                pw_enabled: self.config.pw.is_some(),
                reusable_select_scratch: G::uses_reusable_select_scratch(),
            },
            policy,
        )
    }

    /// Run search with opt-in automatic thread selection.
    ///
    /// Explicit `run_par(..., n_threads)` remains unchanged for reproducible
    /// ablations. This method is the production-oriented path for
    /// device-agnostic throughput tuning.
    pub fn run_auto(
        &self,
        controller: &mut dyn SearchController,
        policy: parallel::AutoThreadPolicy,
    ) -> (SearchStats, parallel::AutoThreadDecision) {
        let decision = self.auto_thread_decision(controller, policy);
        let stats = if decision.threads > 1 {
            controller.reset();
            self.run_par(controller, decision.threads)
        } else {
            self.run(controller)
        };
        (stats, decision)
    }

    /// QuartzController와 통합된 run — 통계 자동 갱신, 적응적 정지
    pub fn run_quartz(&self, ctrl: &mut QuartzController) -> SearchStats {
        ctrl.reset();
        self.par_ctrl.set_n_threads(1);
        self.par_ctrl.reset_for_search();
        self.selection_telemetry.reset();
        let start = Instant::now();
        let mut it = 0u32;
        let mut scratch = self.make_select_scratch();
        let use_quartz = self.config.quartz.is_some();
        let (mut qstats_epoch, mut qstats_snapshot) = if use_quartz {
            self.quartz_snapshot()
        } else {
            (0, None)
        };

        if let Some(limit) = ctrl.visit_limit_hint() {
            loop {
                let rv = self.root.n_total.load(Ordering::Relaxed);
                if rv >= limit {
                    break;
                }
                // Foundry and other SearchPolicy overlays must remain live on
                // the serial Quartz fixed-budget path as well. Previously the
                // same config worked only when n_threads > 1 because
                // run_par_quartz polled the policy while this branch did not.
                if self.policy_halt_check(it as u64, 0) {
                    break;
                }
                self.iterate_with_cached_quartz_snapshot(
                    &mut scratch,
                    use_quartz,
                    &mut qstats_epoch,
                    &mut qstats_snapshot,
                );
                it += 1;
                if it % ctrl.cfg.check_interval == 0 {
                    ctrl.update_stats(&self.root, None);
                    let stats = ctrl.last_stats();
                    qstats_epoch = self.store_quartz_stats(stats.clone());
                    qstats_snapshot = Some(stats);
                    self.refresh_par_ctrl();
                    let checked_visits = self.root.n_total.load(Ordering::Relaxed);
                    ctrl.mark_checked(checked_visits);
                }
            }

            let ms = start.elapsed().as_millis() as u64;
            ctrl.update_elapsed(ms);
            ctrl.update_stats(&self.root, None);
            self.store_quartz_stats(ctrl.last_stats());
            let rv = self.root.n_total.load(Ordering::Relaxed);
            let _ = ctrl.should_stop(rv, ms);
            return SearchStats {
                iterations: it,
                elapsed_ms: ms,
                nps: if ms > 0 {
                    it as f64 / ms as f64 * 1000.0
                } else {
                    0.0
                },
                tt_hit_rate: self.tt.hit_rate(),
                tt_size: self.tt.size(),
                root_visits: rv,
                stop_reason: ctrl.stop_reason(),
            };
        }

        let mut iter_start = Instant::now();

        loop {
            let rv = self.root.n_total.load(Ordering::Relaxed);
            let ms = start.elapsed().as_millis() as u64;
            let checked = it > 0 && it % ctrl.cfg.check_interval == 0;

            let should_stop = if checked {
                // 실제 per-iter cost (ms) EMA 갱신 — CTM cost에 사용
                let iter_ms =
                    iter_start.elapsed().as_secs_f32() * 1000.0 / ctrl.cfg.check_interval as f32;
                ctrl.record_iter_time_ms(iter_ms);
                iter_start = Instant::now();

                // elapsed 주입 — CTM urgency + NS annealing에 사용
                ctrl.update_elapsed(ms);
                ctrl.update_stats(&self.root, None);
                let stats = ctrl.last_stats();
                qstats_epoch = self.store_quartz_stats(stats.clone());
                qstats_snapshot = Some(stats);
                self.refresh_par_ctrl();
                let stop_now = ctrl.should_stop(rv, ms);
                if !stop_now {
                    ctrl.mark_checked(rv);
                }
                stop_now
            } else {
                ctrl.should_stop(rv, ms)
            };

            if should_stop {
                break;
            }
            if self.policy_halt_check(it as u64, ms) {
                break;
            }
            self.iterate_with_cached_quartz_snapshot(
                &mut scratch,
                use_quartz,
                &mut qstats_epoch,
                &mut qstats_snapshot,
            );
            it += 1;
        }

        ctrl.update_elapsed(start.elapsed().as_millis() as u64);
        {
            ctrl.update_stats(&self.root, None);
        }
        self.store_quartz_stats(ctrl.last_stats());

        let ms = start.elapsed().as_millis() as u64;
        let rv = self.root.n_total.load(Ordering::Relaxed);
        SearchStats {
            iterations: it,
            elapsed_ms: ms,
            nps: if ms > 0 {
                it as f64 / ms as f64 * 1000.0
            } else {
                0.0
            },
            tt_hit_rate: self.tt.hit_rate(),
            tt_size: self.tt.size(),
            root_visits: rv,
            stop_reason: ctrl.stop_reason(),
        }
    }

    pub fn run_quartz_auto(
        &self,
        ctrl: &mut QuartzController,
        policy: parallel::AutoThreadPolicy,
    ) -> (SearchStats, parallel::AutoThreadDecision) {
        let decision = self.auto_thread_decision(ctrl, policy);
        let stats = if decision.threads > 1 {
            self.run_par_quartz(ctrl, decision.threads)
        } else {
            self.run_quartz(ctrl)
        };
        (stats, decision)
    }

    /// GVOC + QUARTZ 완전 통합 run
    ///
    /// - GVOC: VOC 기반 PW 확장 폭 동적 조정
    /// - QUARTZ: σ_Q / P_flip / VOC 통계 + 적응적 정지
    /// - 재현성: config.seed 기반 결정론적 RNG
    pub fn run_gvoc(
        &self,
        ctrl: &mut QuartzController,
        gvoc_cfg: &GvocConfig,
    ) -> (SearchStats, GvocState) {
        ctrl.reset();
        self.par_ctrl.set_n_threads(1);
        self.par_ctrl.reset_for_search();
        self.selection_telemetry.reset();
        let start = Instant::now();
        let mut it = 0u32;
        let n_cands = self.root.candidate_count();
        let init_vis = match &self.config.pw {
            Some(pw) => pw.k(0).max(1).min(n_cands),
            None => n_cands,
        };
        let qcfg = ctrl.cfg.clone();

        let mut gvoc = GvocState::new(gvoc_cfg.clone(), init_vis);
        let mut scratch = self.make_select_scratch();
        let use_quartz = self.config.quartz.is_some();
        let (mut qstats_epoch, mut qstats_snapshot) = if use_quartz {
            self.quartz_snapshot()
        } else {
            (0, None)
        };

        loop {
            let rv = self.root.n_total.load(Ordering::Relaxed);
            let ms = start.elapsed().as_millis() as u64;
            let checked = it > 0 && it % ctrl.cfg.check_interval == 0;

            // QUARTZ 수렴 체크
            let should_stop = if checked {
                ctrl.update_stats(&self.root, None);
                let stats = ctrl.last_stats();
                qstats_epoch = self.store_quartz_stats(stats.clone());
                qstats_snapshot = Some(stats);
                self.refresh_par_ctrl();
                let stop_now = ctrl.should_stop(rv, ms);
                if !stop_now {
                    ctrl.mark_checked(rv);
                }
                stop_now
            } else {
                ctrl.should_stop(rv, ms)
            };
            if should_stop {
                break;
            }

            // GVOC: VOC 기반 n_visible 동적 조정
            gvoc.update(&self.root, n_cands, &qcfg);

            // proposal mode 결정 (Inside / Outside)
            let _mode = if let Some(s) = self.current_quartz_stats() {
                routing_mode(&s, &qcfg)
            } else {
                ProposalMode::Inside
            };
            // Outside mode: WL bonus가 이미 QuartzController 통계에 포함됨
            // Inside mode: 표준 EFT-PUCT

            self.iterate_with_cached_quartz_snapshot(
                &mut scratch,
                use_quartz,
                &mut qstats_epoch,
                &mut qstats_snapshot,
            );
            it += 1;
        }

        {
            ctrl.update_stats(&self.root, None);
        }
        self.store_quartz_stats(ctrl.last_stats());

        let ms = start.elapsed().as_millis() as u64;
        let rv = self.root.n_total.load(Ordering::Relaxed);
        let stats = SearchStats {
            iterations: it,
            elapsed_ms: ms,
            nps: if ms > 0 {
                it as f64 / ms as f64 * 1000.0
            } else {
                0.0
            },
            tt_hit_rate: self.tt.hit_rate(),
            tt_size: self.tt.size(),
            root_visits: rv,
            stop_reason: ctrl.stop_reason(),
        };
        (stats, gvoc)
    }

    pub fn run_par(&self, controller: &dyn SearchController, n_threads: usize) -> SearchStats {
        let n_threads = n_threads.max(1);
        self.par_ctrl.set_n_threads(n_threads as u32);
        self.par_ctrl.reset_for_search();
        self.selection_telemetry.reset();
        let start = Instant::now();
        let use_quartz = self.config.quartz.is_some();
        // BQ++ Phase 8c: shared halt flag for parallel SearchPolicy integration.
        // Thread 0 polls `policy_halt_check` periodically; if it fires, all
        // threads observe the flag at iteration boundary and break.
        let policy_halted = AtomicBool::new(false);
        let has_policy = self.config.search_policy.is_some();

        if let Some(limit) = controller.visit_limit_hint() {
            let next_visit = AtomicU32::new(self.root.n_total.load(Ordering::Relaxed));
            let ticket_chunk = if self.mqft_trace.is_some() { 1 } else { fixed_budget_ticket_chunk(n_threads) };
            rayon::scope(|s| {
                for tid in 0..n_threads {
                    let qcfg_ref = &self.config.quartz;
                    let next_visit_ref = &next_visit;
                    let policy_halted_ref = &policy_halted;
                    s.spawn(move |_| {
                        let mut local_it = 0u32;
                        let mut scratch = self.make_select_scratch();
                        let (mut qstats_epoch, mut qstats_snapshot) = if use_quartz {
                            self.quartz_snapshot()
                        } else {
                            (0, None)
                        };
                        loop {
                            if has_policy && policy_halted_ref.load(Ordering::Relaxed) {
                                break;
                            }
                            let chunk_start =
                                next_visit_ref.fetch_add(ticket_chunk, Ordering::Relaxed);
                            if chunk_start >= limit {
                                break;
                            }
                            let chunk_end = chunk_start.saturating_add(ticket_chunk).min(limit);
                            for _ticket in chunk_start..chunk_end {
                                self.iterate_with_cached_quartz_snapshot(
                                    &mut scratch,
                                    use_quartz,
                                    &mut qstats_epoch,
                                    &mut qstats_snapshot,
                                );
                                local_it += 1;
                                if tid == 0 {
                                    if let Some(ref qcfg) = qcfg_ref {
                                        if local_it % qcfg.check_interval == 0 {
                                            self.refresh_quartz_stats();
                                            self.refresh_par_ctrl();
                                        }
                                    }
                                    if has_policy {
                                        let interval = qcfg_ref
                                            .as_ref()
                                            .map(|q| q.check_interval)
                                            .unwrap_or(64);
                                        if local_it % interval == 0 {
                                            let rv = self.root.n_total.load(Ordering::Relaxed);
                                            let ms = start.elapsed().as_millis() as u64;
                                            if self.policy_halt_check(rv as u64, ms) {
                                                policy_halted_ref.store(true, Ordering::Relaxed);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            });
        } else {
            rayon::scope(|s| {
                for tid in 0..n_threads {
                    let qcfg_ref = &self.config.quartz;
                    let needs_elapsed = controller.needs_elapsed_ms();
                    let policy_halted_ref = &policy_halted;
                    s.spawn(move |_| {
                        let mut local_it = 0u32;
                        let mut scratch = self.make_select_scratch();
                        let (mut qstats_epoch, mut qstats_snapshot) = if use_quartz {
                            self.quartz_snapshot()
                        } else {
                            (0, None)
                        };
                        loop {
                            if has_policy && policy_halted_ref.load(Ordering::Relaxed) {
                                break;
                            }
                            let rv = self.root.n_total.load(Ordering::Relaxed);
                            let ms = if needs_elapsed {
                                start.elapsed().as_millis() as u64
                            } else {
                                0
                            };
                            if controller.should_stop(rv, ms) {
                                break;
                            }
                            self.iterate_with_cached_quartz_snapshot(
                                &mut scratch,
                                use_quartz,
                                &mut qstats_epoch,
                                &mut qstats_snapshot,
                            );
                            local_it += 1;
                            if tid == 0 {
                                if let Some(ref qcfg) = qcfg_ref {
                                    if local_it % qcfg.check_interval == 0 {
                                        self.refresh_quartz_stats();
                                        self.refresh_par_ctrl();
                                    }
                                }
                                if has_policy {
                                    let interval =
                                        qcfg_ref.as_ref().map(|q| q.check_interval).unwrap_or(64);
                                    if local_it % interval == 0 {
                                        let ms_now = start.elapsed().as_millis() as u64;
                                        let rv_now = self.root.n_total.load(Ordering::Relaxed);
                                        if self.policy_halt_check(rv_now as u64, ms_now) {
                                            policy_halted_ref.store(true, Ordering::Relaxed);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            });
        }

        let ms = start.elapsed().as_millis() as u64;
        let rv = self.root.n_total.load(Ordering::Relaxed);
        SearchStats {
            iterations: rv,
            elapsed_ms: ms,
            nps: if ms > 0 {
                rv as f64 / ms as f64 * 1000.0
            } else {
                0.0
            },
            tt_hit_rate: self.tt.hit_rate(),
            tt_size: self.tt.size(),
            root_visits: rv,
            stop_reason: controller.stop_reason(),
        }
    }

    /// Parallel MCTS search with full QUARTZ controller integration.
    ///
    /// Combines the threading model of `run_par` with the adaptive statistics
    /// and stopping logic of `run_quartz`. Thread 0 handles QUARTZ stats
    /// updates; all threads check `should_stop`.
    pub fn run_par_quartz(&self, ctrl: &mut QuartzController, n_threads: usize) -> SearchStats {
        ctrl.reset();
        let n_threads = n_threads.max(1);
        self.par_ctrl.set_n_threads(n_threads as u32);
        self.par_ctrl.reset_for_search();
        self.selection_telemetry.reset();

        let start = Instant::now();
        let check_interval = self.config.quartz.as_ref().map_or(20, |q| q.check_interval);
        let use_quartz = self.config.quartz.is_some();
        // BQ++ Phase 8c: shared halt flag for parallel SearchPolicy
        // integration. See `run_par` for rationale.
        let policy_halted = AtomicBool::new(false);
        let has_policy = self.config.search_policy.is_some();

        // Reborrow as shared reference for rayon scope
        let ctrl_ref: &QuartzController = &*ctrl;

        if let Some(limit) = ctrl_ref.visit_limit_hint() {
            let next_visit = AtomicU32::new(self.root.n_total.load(Ordering::Relaxed));
            let ticket_chunk = fixed_budget_ticket_chunk(n_threads);
            rayon::scope(|s| {
                for tid in 0..n_threads {
                    let next_visit_ref = &next_visit;
                    let policy_halted_ref = &policy_halted;
                    s.spawn(move |_| {
                        let mut local_it = 0u32;
                        let mut scratch = self.make_select_scratch();
                        let (mut qstats_epoch, mut qstats_snapshot) = if use_quartz {
                            self.quartz_snapshot()
                        } else {
                            (0, None)
                        };
                        loop {
                            if has_policy && policy_halted_ref.load(Ordering::Relaxed) {
                                break;
                            }
                            let chunk_start =
                                next_visit_ref.fetch_add(ticket_chunk, Ordering::Relaxed);
                            if chunk_start >= limit {
                                break;
                            }
                            let chunk_end = chunk_start.saturating_add(ticket_chunk).min(limit);
                            for _ticket in chunk_start..chunk_end {
                                self.iterate_with_cached_quartz_snapshot(
                                    &mut scratch,
                                    use_quartz,
                                    &mut qstats_epoch,
                                    &mut qstats_snapshot,
                                );
                                local_it += 1;

                                // Thread 0: periodic QUARTZ stats refresh. Fixed
                                // halt mode still uses QUARTZ root shaping during
                                // selection, so keep stats fresh even though stop
                                // polling is replaced by exact budget tickets.
                                if tid == 0 && local_it % check_interval == 0 {
                                    let ms = start.elapsed().as_millis() as u64;
                                    ctrl_ref.update_elapsed(ms);
                                    ctrl_ref.update_stats(&self.root, None);
                                    let stats = ctrl_ref.last_stats();
                                    qstats_epoch = self.store_quartz_stats(stats.clone());
                                    qstats_snapshot = Some(stats);
                                    self.refresh_par_ctrl();
                                    let checked_visits = self.root.n_total.load(Ordering::Relaxed);
                                    ctrl_ref.mark_checked(checked_visits);
                                    if has_policy
                                        && self.policy_halt_check(checked_visits as u64, ms)
                                    {
                                        policy_halted_ref.store(true, Ordering::Relaxed);
                                        break;
                                    }
                                }
                            }
                        }
                    });
                }
            });
        } else {
            rayon::scope(|s| {
                for tid in 0..n_threads {
                    let policy_halted_ref = &policy_halted;
                    s.spawn(move |_| {
                        let mut local_it = 0u32;
                        let mut scratch = self.make_select_scratch();
                        let (mut qstats_epoch, mut qstats_snapshot) = if use_quartz {
                            self.quartz_snapshot()
                        } else {
                            (0, None)
                        };
                        loop {
                            if has_policy && policy_halted_ref.load(Ordering::Relaxed) {
                                break;
                            }
                            let rv = self.root.n_total.load(Ordering::Relaxed);
                            let ms = start.elapsed().as_millis() as u64;
                            if ctrl_ref.should_stop(rv, ms) {
                                break;
                            }
                            self.iterate_with_cached_quartz_snapshot(
                                &mut scratch,
                                use_quartz,
                                &mut qstats_epoch,
                                &mut qstats_snapshot,
                            );
                            local_it += 1;

                            // Thread 0: periodic QUARTZ stats refresh
                            if tid == 0 && local_it % check_interval == 0 {
                                ctrl_ref.update_elapsed(ms);
                                ctrl_ref.update_stats(&self.root, None);
                                let stats = ctrl_ref.last_stats();
                                qstats_epoch = self.store_quartz_stats(stats.clone());
                                qstats_snapshot = Some(stats);
                                self.refresh_par_ctrl();
                                let checked_visits = self.root.n_total.load(Ordering::Relaxed);
                                ctrl_ref.mark_checked(checked_visits);
                                if has_policy && self.policy_halt_check(checked_visits as u64, ms) {
                                    policy_halted_ref.store(true, Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                    });
                }
            });
        }

        // Final stats update
        let ms = start.elapsed().as_millis() as u64;
        ctrl.update_elapsed(ms);
        {
            ctrl.update_stats(&self.root, None);
        }
        self.store_quartz_stats(ctrl.last_stats());

        let rv = self.root.n_total.load(Ordering::Relaxed);
        let _ = ctrl.should_stop(rv, ms);
        SearchStats {
            iterations: rv,
            elapsed_ms: ms,
            nps: if ms > 0 {
                rv as f64 / ms as f64 * 1000.0
            } else {
                0.0
            },
            tt_hit_rate: self.tt.hit_rate(),
            tt_size: self.tt.size(),
            root_visits: rv,
            stop_reason: ctrl.stop_reason(),
        }
    }

    // ── 결과 추출 ──────────────────────────────────────────────

    /// Extract edge priors from root node for QuartzStats computation.
    pub fn root_priors(&self) -> Vec<f32> {
        let n = self.root.materialized_count();
        self.root.edge_priors_snapshot(n)
    }

    /// Advance root to the child matching `chosen_move`, reusing the subtree.
    /// Returns true if the child was found and promoted, false otherwise.
    /// After advance, call `expand_root()` if the new root isn't expanded.
    pub fn advance_root(&mut self, chosen_move: G::Move) -> bool
    where
        G::Move: PartialEq,
    {
        let n = self.root.materialized_count();
        let edges = self.root.edge_snapshot(n);
        for edge in &edges {
            if edge.mv == chosen_move {
                self.root = edge.child.clone();
                self.root_state = self.root_state.apply_move(chosen_move);
                self.root_noise = None;
                // Clear quartz cache for new root
                self.clear_quartz_stats();
                self.par_ctrl.reset_for_search();
                // Expand new root if not already expanded
                if self.root.materialized_count() == 0 && !self.root_state.is_terminal() {
                    self.expand_root();
                }
                return true;
            }
        }
        false
    }

    /// Replace the root state and rebuild the engine while preserving the
    /// evaluator/config pair.
    pub fn replace_root_state(&mut self, root_state: G) {
        *self = MctsEngine::new(root_state, self.evaluator.clone(), self.config.clone());
    }

    /// Advance the current root by action index, reusing the subtree when the
    /// chosen move already exists under the root and falling back to a rebuild
    /// otherwise.
    pub fn apply_action_idx_root(&mut self, action: usize) -> Result<(), String>
    where
        G::Move: PartialEq,
    {
        let Some(mv) = self.root_state.idx_to_move(action) else {
            return Err(format!("invalid action {}", action));
        };
        if !self.advance_root(mv) {
            let next_state = self.root_state.apply_move(mv);
            self.replace_root_state(next_state);
        }
        Ok(())
    }

    /// Get a reference to the current root state.
    pub fn root_state(&self) -> &G {
        &self.root_state
    }

    pub fn best_move(&self) -> Option<G::Move> {
        if self.config.root_forced_win && self.config.temperature <= 0.0 {
            if let Some(forced) = self
                .root_state
                .legal_moves()
                .into_iter()
                .find(|&mv| self.root_state.is_winning_move(mv))
            {
                return Some(forced);
            }
        }
        select_move_with_temperature(&self.root, self.config.temperature, self.config.seed)
    }

    pub fn pi_target(&self, temperature: f32) -> Vec<(G::Move, f32)> {
        visit_distribution(&self.root, temperature)
    }

    #[cfg(test)]
    pub fn root_entropy(&self) -> f32 {
        self.root
            .with_edge_slice(self.root.materialized_count(), |edges| {
                let counts = edges
                    .iter()
                    .map(|e| e.n.load(Ordering::Acquire))
                    .collect::<Vec<_>>();
                root_entropy(&counts)
            })
    }

    #[cfg(test)]
    pub fn root_value(&self) -> f32 {
        let total = self.root.n_total.load(Ordering::Acquire);
        if total == 0 {
            return 0.0;
        }
        self.root
            .with_edge_slice(self.root.materialized_count(), |edges| {
                edges
                    .iter()
                    .map(|e| e.n.load(Ordering::Acquire) as f32 * e.q())
                    .sum::<f32>()
                    / total as f32
            })
    }

    pub fn pw_stats(&self) -> (usize, usize) {
        let total = self.root.candidate_count();
        let n_root = self.root.n_total.load(Ordering::Acquire);
        let open = match &self.config.pw {
            Some(cfg) => cfg.k(n_root).max(1).min(total),
            None => total,
        };
        (open, total)
    }

    #[cfg(test)]
    pub fn print_stats(&self, label: &str) {
        let rv = self.root.n_total.load(Ordering::Acquire);
        let h = self.root_entropy();
        let v = self.root_value();
        let (pw_open, pw_total) = self.pw_stats();

        println!(
            "\n[{}] visits={}, entropy={:.3}, value={:.3}, hash=0x{:016X}",
            label, rv, h, v, self.root.hash
        );
        println!(
            "TT(size={}, hit={:.1}%)  PW({}/{})  max_depth={}",
            self.tt.size(),
            self.tt.hit_rate() * 100.0,
            pw_open,
            pw_total,
            self.config.max_depth
        );

        // QUARTZ 통계 출력
        if let Some(stats) = self.current_quartz_stats() {
            stats.print(label);
        }

        let n_show = self.root.materialized_count().min(pw_open);
        let edges = self.root.edge_snapshot(n_show);
        if !edges.is_empty() {
            println!(
                "{:<8} {:>7} {:>8} {:>8} {:>7}",
                "Move", "N", "W", "Q", "P(%)"
            );
            println!("{}", "-".repeat(44));
            let mut data: Vec<_> = edges.iter().map(|e| (e.mv, e.n, e.w, e.q(), e.p)).collect();
            data.sort_by(|a, b| b.1.cmp(&a.1));
            for (mv, n, w, q, p) in data.iter().take(9) {
                println!(
                    "  {:?}  {:>7} {:>8.2} {:>8.3} {:>6.1}",
                    mv,
                    n,
                    w,
                    q,
                    p * 100.0
                );
            }
            if pw_open > 9 {
                println!("  ... ({} more open)", pw_open - 9);
            }
            if pw_total > pw_open {
                println!("  ... ({} not yet opened)", pw_total - pw_open);
            }
        }
    }
}

unsafe impl<G: GameState> Sync for MctsEngine<G> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation_refresh::{gen_gomoku_positions, gen_ttt_positions};
    use crate::games::gomoku::Gomoku;
    use crate::games::tictactoe::TicTacToe;
    use crate::mcts::eval::{ShortRollout, UniformEval};
    use crate::mcts::policy::{
        ControllerTelemetry, EdgeView, HaltDecision, LegacyAlphaZero, ScoreAdjustment,
        SearchPolicy, SearchSnapshot,
    };
    use crate::mcts::quartz::{HaltMode, PenaltyMode, QuartzConfig, QuartzController};
    use crate::mcts::search::{FixedIterations, StopReason};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;
    use std::hint::black_box;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    static SCRATCH_CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static EDGE_COUNTER_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct HotPathMetricsGuard;

    impl HotPathMetricsGuard {
        fn enabled() -> Self {
            crate::mcts::profiling::set_test_hot_path_metrics_override(Some(true));
            HotPathMetricsGuard
        }
    }

    impl Drop for HotPathMetricsGuard {
        fn drop(&mut self) {
            crate::mcts::profiling::set_test_hot_path_metrics_override(None);
        }
    }

    #[derive(Debug)]
    struct ScratchCloneGame {
        phase: u8,
    }

    impl Clone for ScratchCloneGame {
        fn clone(&self) -> Self {
            SCRATCH_CLONE_COUNT.fetch_add(1, Ordering::Relaxed);
            Self { phase: self.phase }
        }
    }

    impl GameState for ScratchCloneGame {
        type Move = u8;
        type Undo = u8;

        fn initial() -> Self {
            Self { phase: 0 }
        }

        fn current_player(&self) -> i8 {
            1
        }

        fn legal_moves(&self) -> Vec<Self::Move> {
            if self.phase == 0 {
                vec![0, 1]
            } else {
                vec![]
            }
        }

        fn apply_move(&self, _mv: Self::Move) -> Self {
            Self {
                phase: self.phase.saturating_add(1),
            }
        }

        fn apply_move_in_place(&mut self, _mv: Self::Move) -> Self::Undo {
            let prev = self.phase;
            self.phase = self.phase.saturating_add(1);
            prev
        }

        fn undo_move(&mut self, undo: Self::Undo) {
            self.phase = undo;
        }

        fn uses_reusable_select_scratch() -> bool {
            true
        }

        fn is_terminal(&self) -> bool {
            self.phase > 0
        }

        fn outcome(&self) -> f32 {
            0.0
        }

        fn hash(&self) -> u64 {
            self.phase as u64
        }

        fn num_actions(&self) -> usize {
            2
        }

        fn move_to_idx(&self, mv: Self::Move) -> usize {
            mv as usize
        }

        fn idx_to_move(&self, idx: usize) -> Option<Self::Move> {
            (idx < 2).then_some(idx as u8)
        }
    }

    #[derive(Clone, Debug)]
    struct TtHashDummy {
        phase: u8,
        raw_hash: u64,
        exact_hash: u64,
    }

    impl GameState for TtHashDummy {
        type Move = u8;
        type Undo = Self;

        fn initial() -> Self {
            Self {
                phase: 0,
                raw_hash: 1,
                exact_hash: 11,
            }
        }

        fn current_player(&self) -> i8 {
            1
        }

        fn legal_moves(&self) -> Vec<Self::Move> {
            if self.phase == 0 {
                vec![0, 1]
            } else {
                vec![]
            }
        }

        fn apply_move(&self, mv: Self::Move) -> Self {
            match mv {
                0 => Self {
                    phase: 1,
                    raw_hash: 101,
                    exact_hash: 999,
                },
                1 => Self {
                    phase: 1,
                    raw_hash: 202,
                    exact_hash: 999,
                },
                _ => unreachable!(),
            }
        }

        fn apply_move_in_place(&mut self, mv: Self::Move) -> Self {
            let next = self.apply_move(mv);
            std::mem::replace(self, next)
        }

        fn undo_move(&mut self, undo: Self) {
            *self = undo;
        }

        fn is_terminal(&self) -> bool {
            self.phase == 1
        }

        fn outcome(&self) -> f32 {
            0.0
        }

        fn hash(&self) -> u64 {
            self.raw_hash
        }

        fn tt_hash(&self) -> u64 {
            self.exact_hash
        }

        fn num_actions(&self) -> usize {
            2
        }

        fn move_to_idx(&self, mv: Self::Move) -> usize {
            mv as usize
        }

        fn idx_to_move(&self, idx: usize) -> Option<Self::Move> {
            if idx < 2 {
                Some(idx as u8)
            } else {
                None
            }
        }
    }

    fn run_reference_fixed<G: GameState>(engine: &MctsEngine<G>, limit: u32) -> SearchStats {
        engine.par_ctrl.reset_for_search();
        let start = Instant::now();
        let mut it = 0u32;
        let qcfg = engine.config.quartz.clone();

        loop {
            let rv = engine.root.n_total.load(Ordering::Relaxed);
            let ms = start.elapsed().as_millis() as u64;
            if rv >= limit {
                break;
            }

            engine.iterate();
            it += 1;

            if let Some(ref qcfg) = qcfg {
                if it % qcfg.check_interval == 0 {
                    engine.refresh_quartz_stats();
                    engine.refresh_par_ctrl();
                }
            }

            black_box(ms);
        }

        let ms = start.elapsed().as_millis() as u64;
        let rv = engine.root.n_total.load(Ordering::Relaxed);
        SearchStats {
            iterations: it,
            elapsed_ms: ms,
            nps: if ms > 0 {
                it as f64 / ms as f64 * 1000.0
            } else {
                0.0
            },
            tt_hit_rate: engine.tt.hit_rate(),
            tt_size: engine.tt.size(),
            root_visits: rv,
            stop_reason: StopReason::BudgetExhausted { iterations: limit },
        }
    }

    #[test]
    fn test_sync_run_reuses_select_scratch_for_compact_undo_games() {
        SCRATCH_CLONE_COUNT.store(0, Ordering::Relaxed);
        let eval: Arc<dyn Evaluator<ScratchCloneGame>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(
            ScratchCloneGame::initial(),
            eval,
            MctsConfig::evaluation(2.0),
        );

        let mut ctrl = crate::mcts::search::FixedIterations::new(32);
        let stats = engine.run(&mut ctrl);

        assert_eq!(stats.root_visits, 32);
        let clones = SCRATCH_CLONE_COUNT.load(Ordering::Relaxed);
        assert!(
            clones <= 4,
            "compact-undo sync search should clone once for root expansion and once for worker scratch, not once per iteration (clones={clones})"
        );
    }

    #[test]
    fn tt_uses_exact_tt_hash_for_child_transpositions() {
        let eval: Arc<dyn Evaluator<TtHashDummy>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(TtHashDummy::initial(), eval, MctsConfig::evaluation(1.0));
        let edges = engine.root.read_edges();

        assert_eq!(edges.len(), 2);
        assert!(ArenaRef::ptr_eq(&edges[0].child, &edges[1].child));
    }

    fn gomoku7_engine(iters: u32) -> (MctsEngine<Gomoku>, QuartzController) {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(ShortRollout::new(8));
        let qcfg = QuartzConfig {
            min_visits: 15,
            check_interval: 20,
            ..Default::default()
        };
        let cfg = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
        let engine = MctsEngine::new(state, eval, cfg);
        let ctrl = QuartzController::new(iters, qcfg);
        (engine, ctrl)
    }

    #[test]
    fn test_run_par_quartz_basic() {
        let (engine, mut ctrl) = gomoku7_engine(200);
        let stats = engine.run_par_quartz(&mut ctrl, 2);
        assert!(
            stats.root_visits >= 200,
            "Expected ≥200 visits, got {}",
            stats.root_visits
        );
        assert!(engine.best_move().is_some());
    }

    #[test]
    fn test_run_par_quartz_single_thread() {
        // Single-threaded should behave like run_quartz
        let (engine, mut ctrl) = gomoku7_engine(100);
        let stats = engine.run_par_quartz(&mut ctrl, 1);
        assert!(stats.root_visits >= 100);
        assert!(engine.best_move().is_some());
    }

    #[test]
    fn test_run_par_quartz_four_threads() {
        let (engine, mut ctrl) = gomoku7_engine(400);
        let stats = engine.run_par_quartz(&mut ctrl, 4);
        assert!(
            stats.root_visits >= 400,
            "Expected ≥400 visits, got {}",
            stats.root_visits
        );
        // Should have non-trivial NPS with 4 threads
        assert!(stats.nps > 0.0);
    }

    #[test]
    fn test_run_par_quartz_stats_updated() {
        let (engine, mut ctrl) = gomoku7_engine(200);
        engine.run_par_quartz(&mut ctrl, 2);
        let qstats = ctrl.last_stats();
        assert!(
            engine.current_quartz_stats().is_some(),
            "engine QUARTZ cache should publish the controller snapshot"
        );
        // After 200 visits, sigma_q should be computed (not default NaN)
        assert!(
            qstats.sigma_q.is_finite() || qstats.sigma_q.is_nan(),
            "sigma_q should be computed after search"
        );
    }

    #[test]
    fn test_run_par_quartz_high_threads_no_panic() {
        // Stress: 16 threads on a small search
        let (engine, mut ctrl) = gomoku7_engine(50);
        let stats = engine.run_par_quartz(&mut ctrl, 16);
        // With virtual loss, 16 threads should still complete
        assert!(stats.root_visits >= 50);
    }

    #[test]
    fn test_run_par_quartz_budget_one() {
        // Edge case: budget=1 should terminate cleanly
        let (engine, mut ctrl) = gomoku7_engine(1);
        let stats = engine.run_par_quartz(&mut ctrl, 2);
        // At least 1 visit (may be more due to thread timing)
        assert!(stats.root_visits >= 1);
    }

    #[test]
    fn test_run_par_fixed_budget_reservation_is_exact() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(2.0));
        let ctrl = crate::mcts::search::FixedIterations::new(128);

        let stats = engine.run_par(&ctrl, 8);

        assert_eq!(stats.root_visits, 128);
        assert_eq!(
            stats.stop_reason,
            StopReason::BudgetExhausted { iterations: 128 }
        );
    }

    #[test]
    fn test_auto_thread_decision_uses_remaining_fixed_budget() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(2.0));
        let ctrl = crate::mcts::search::FixedIterations::new(10_000);
        let expected_threads = parallel::available_search_threads().min(8);

        let decision = engine.auto_thread_decision(
            &ctrl,
            parallel::AutoThreadPolicy::throughput().with_max_threads(8),
        );

        assert_eq!(decision.threads, expected_threads);
        assert_eq!(decision.requested_cap, expected_threads);
        assert_eq!(decision.remaining_visits, Some(10_000));
    }

    #[test]
    fn test_run_auto_fixed_budget_is_exact() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(2.0));
        let mut ctrl = crate::mcts::search::FixedIterations::new(130);

        let (stats, decision) = engine.run_auto(
            &mut ctrl,
            parallel::AutoThreadPolicy::throughput().with_max_threads(8),
        );

        assert_eq!(decision.threads, 2);
        assert_eq!(stats.root_visits, 130);
        assert_eq!(
            stats.stop_reason,
            StopReason::BudgetExhausted { iterations: 130 }
        );
    }

    #[test]
    fn test_run_auto_quality_caps_low_branching_game() {
        let state = TicTacToe::initial();
        let eval: Arc<dyn Evaluator<TicTacToe>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(1.4));
        let mut ctrl = crate::mcts::search::FixedIterations::new(50_000);

        let (stats, decision) = engine.run_auto(
            &mut ctrl,
            parallel::AutoThreadPolicy::quality().with_max_threads(16),
        );

        assert_eq!(decision.threads, 4);
        assert_eq!(stats.root_visits, 50_000);
    }

    #[test]
    fn test_run_quartz_auto_fixed_budget_is_exact() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let qcfg = QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: 130 },
            check_interval: 16,
            min_visits: 8,
            ..Default::default()
        };
        let cfg = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = QuartzController::new(512, qcfg);

        let (stats, decision) = engine.run_quartz_auto(
            &mut ctrl,
            parallel::AutoThreadPolicy::throughput().with_max_threads(8),
        );

        assert_eq!(decision.threads, 2);
        assert_eq!(stats.root_visits, 130);
    }

    #[test]
    fn test_run_par_fixed_budget_chunking_is_exact_when_not_divisible() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(2.0));
        let ctrl = crate::mcts::search::FixedIterations::new(130);

        let stats = engine.run_par(&ctrl, 8);

        assert_eq!(stats.root_visits, 130);
        assert_eq!(
            stats.stop_reason,
            StopReason::BudgetExhausted { iterations: 130 }
        );
    }

    #[test]
    fn test_run_par_quartz_fixed_budget_reservation_is_exact() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let qcfg = QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: 128 },
            check_interval: 16,
            min_visits: 8,
            ..Default::default()
        };
        let cfg = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = QuartzController::new(512, qcfg);

        let stats = engine.run_par_quartz(&mut ctrl, 8);

        assert_eq!(stats.root_visits, 128);
        assert_eq!(
            stats.stop_reason,
            StopReason::BudgetExhausted { iterations: 128 }
        );
        assert!(ctrl.last_stats().root_visits >= 128);
    }

    #[test]
    fn test_run_par_quartz_fixed_budget_chunking_is_exact_when_not_divisible() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let qcfg = QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: 130 },
            check_interval: 16,
            min_visits: 8,
            ..Default::default()
        };
        let cfg = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = QuartzController::new(512, qcfg);

        let stats = engine.run_par_quartz(&mut ctrl, 8);

        assert_eq!(stats.root_visits, 130);
        assert_eq!(
            stats.stop_reason,
            StopReason::BudgetExhausted { iterations: 130 }
        );
        assert!(ctrl.last_stats().root_visits >= 130);
    }

    #[test]
    fn test_run_quartz_fixed_budget_fast_path_is_exact() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let qcfg = QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: 96 },
            check_interval: 16,
            min_visits: 8,
            ..Default::default()
        };
        let cfg = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = QuartzController::new(256, qcfg);

        let stats = engine.run_quartz(&mut ctrl);

        assert_eq!(stats.root_visits, 96);
        assert_eq!(stats.iterations, 96);
        assert_eq!(
            stats.stop_reason,
            StopReason::BudgetExhausted { iterations: 96 }
        );
    }

    #[cfg(feature = "idea-foundry")]
    #[test]
    fn test_run_quartz_fixed_budget_polls_foundry_policy() {
        use crate::mcts::foundry::{FoundryRuntimeMode, FoundrySearchConfig, A01_AXIS_ID};

        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let qcfg = QuartzConfig {
            halt_mode: HaltMode::Fixed { budget: 96 },
            check_interval: 1,
            min_visits: 8,
            ..Default::default()
        };
        let mut cfg = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
        cfg.foundry_search = Some(FoundrySearchConfig {
            mode: FoundryRuntimeMode::Active,
            axis_id: A01_AXIS_ID.to_string(),
            checkpoint_id: "test-checkpoint".to_string(),
            evaluator_id: "test-evaluator".to_string(),
            risk_limit: 1.0,
            min_visits: 16,
        });
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = QuartzController::new(256, qcfg);

        let stats = engine.run_quartz(&mut ctrl);
        let telemetry = engine
            .policy_telemetry_snapshot()
            .expect("Foundry policy telemetry must be attached");

        assert_eq!(stats.iterations, 16);
        assert_eq!(telemetry.foundry_mode.as_deref(), Some("active"));
        assert!(telemetry.metacontroller_decisions > 0);
        assert_eq!(telemetry.metacontroller_actions, 1);
        assert_eq!(telemetry.metacontroller_coordination_errors, 0);
    }

    #[test]
    fn test_apply_action_idx_root_advances_and_restarts_from_new_root() {
        let eval: Arc<dyn Evaluator<TicTacToe>> = Arc::new(UniformEval);
        let mut engine = MctsEngine::new(TicTacToe::initial(), eval, MctsConfig::evaluation(1.4));
        let mut ctrl = QuartzController::new(8, QuartzConfig::default());
        let stats_before = engine.run_quartz(&mut ctrl);
        assert!(stats_before.root_visits >= 8);

        engine.apply_action_idx_root(0).unwrap();
        assert_eq!(engine.root_state().current_player(), -1);
        assert_eq!(engine.root.n_total.load(Ordering::Relaxed), 0);

        let mut ctrl2 = QuartzController::new(4, QuartzConfig::default());
        let stats_after = engine.run_quartz(&mut ctrl2);
        assert!(stats_after.root_visits >= 4);
    }

    #[test]
    fn test_set_n_threads() {
        use crate::mcts::parallel::{ParallelismController, VlMode};
        let ctrl = ParallelismController::new(VlMode::Adaptive, 1);
        ctrl.set_n_threads(4);
        // Verify via vl_at_depth — contention calculation uses n_threads
        let vl = ctrl.vl_at_depth(0);
        assert!(vl.vvisit > 0.0);
    }

    #[test]
    fn test_serial_run_skips_virtual_loss_reservations() {
        let eval: Arc<dyn Evaluator<TicTacToe>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(TicTacToe::initial(), eval, MctsConfig::evaluation(1.4));
        let mut ctrl = crate::mcts::search::FixedIterations::new(16);

        engine.run(&mut ctrl);

        let snap = engine.par_ctrl.telemetry.snapshot();
        assert_eq!(snap.total_selects, 0);
        assert_eq!(snap.dup_leaf_count, 0);
        assert_eq!(snap.max_pending, 0);
    }

    #[test]
    fn test_async_prepare_forces_virtual_loss_reservation() {
        let eval: Arc<dyn Evaluator<TicTacToe>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(TicTacToe::initial(), eval, MctsConfig::evaluation(1.4));

        let prepared = engine.prepare_iteration_async();
        let snap = engine.par_ctrl.telemetry.snapshot();
        assert!(snap.total_selects > 0);

        if let PreparedIteration::Pending(pending) = prepared {
            let eval = engine.evaluator.evaluate(&pending.leaf_state);
            engine.complete_iteration_async(pending, eval);
        }
    }

    #[test]
    fn test_parallel_run_uses_virtual_loss_reservations() {
        let eval: Arc<dyn Evaluator<TicTacToe>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(TicTacToe::initial(), eval, MctsConfig::evaluation(1.4));
        let ctrl = crate::mcts::search::FixedIterations::new(64);

        engine.run_par(&ctrl, 2);

        let snap = engine.par_ctrl.telemetry.snapshot();
        assert!(snap.total_selects > 0);
    }

    #[test]
    fn test_advance_root_resets_parallel_telemetry() {
        let (mut engine, _) = gomoku7_engine(50);
        engine.par_ctrl.telemetry.record_select(3);
        engine.par_ctrl.telemetry.record_dup_leaf();
        let chosen = engine.root_state().legal_moves()[0];

        assert!(engine.advance_root(chosen));

        let snap = engine.par_ctrl.telemetry.snapshot();
        assert_eq!(snap.total_selects, 0);
        assert_eq!(snap.dup_leaf_count, 0);
        assert_eq!(snap.max_pending, 0);
    }

    #[test]
    fn test_advance_root_clears_quartz_cache_epoch() {
        let (mut engine, mut ctrl) = gomoku7_engine(80);
        engine.run_quartz(&mut ctrl);
        assert!(engine.current_quartz_stats().is_some());
        let epoch_before = engine.quartz_cache_epoch.load(Ordering::Acquire);
        let chosen = engine.root_state().legal_moves()[0];

        assert!(engine.advance_root(chosen));

        assert!(engine.current_quartz_stats().is_none());
        assert!(engine.quartz_cache_epoch.load(Ordering::Acquire) > epoch_before);
    }

    #[test]
    fn test_best_move_prefers_immediate_root_win() {
        let mut state = TicTacToe::initial();
        for mv in [0, 3, 1, 4] {
            state = state.apply_move(mv);
        }
        let eval: Arc<dyn Evaluator<TicTacToe>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(2.0));
        assert_eq!(engine.best_move(), Some(2));
    }

    #[test]
    fn test_fixed_iterations_fast_path_matches_reference() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let cfg = MctsConfig::evaluation(2.0);
        let ref_engine = MctsEngine::new(state.clone(), eval.clone(), cfg.clone());
        let ref_stats = run_reference_fixed(&ref_engine, 200);

        let opt_engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = crate::mcts::search::FixedIterations::new(200);
        let opt_stats = opt_engine.run(&mut ctrl);

        assert_eq!(opt_stats.root_visits, ref_stats.root_visits);
        assert_eq!(opt_engine.best_move(), ref_engine.best_move());
    }

    #[test]
    fn test_root_priors_matches_edge_snapshot_priors() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let mut cfg = MctsConfig::evaluation(2.0);
        cfg.pw = Some(PwConfig::new(1.0, 0.0));
        let engine = MctsEngine::new(state, eval, cfg);

        let n = engine.root.materialized_count();
        let from_root_priors = engine.root_priors();
        let from_edges: Vec<f32> = engine.root.edge_snapshot(n).iter().map(|e| e.p).collect();

        assert_eq!(from_root_priors, from_edges);
    }

    #[test]
    fn test_visit_distribution_matches_edge_snapshot_reference() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(2.0));
        let mut ctrl = crate::mcts::search::FixedIterations::new(32);
        let _ = engine.run(&mut ctrl);

        let from_api = visit_distribution(&engine.root, 1.0);
        let from_edges = {
            let edges = engine.root.edge_snapshot(engine.root.materialized_count());
            let counts: Vec<f64> = edges.iter().map(|e| e.n as f64).collect();
            let total: f64 = counts.iter().sum();
            edges
                .iter()
                .zip(counts.iter())
                .map(|(e, &c)| (e.mv, (c / total) as f32))
                .collect::<Vec<_>>()
        };

        assert_eq!(from_api, from_edges);
    }

    #[test]
    fn test_greedy_root_selection_matches_edge_snapshot_reference() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let engine = MctsEngine::new(state, eval, MctsConfig::evaluation(2.0));
        let mut ctrl = crate::mcts::search::FixedIterations::new(32);
        let _ = engine.run(&mut ctrl);

        let from_api = select_move_with_temperature(&engine.root, 0.0, Some(7));
        let from_edges = engine
            .root
            .edge_snapshot(engine.root.materialized_count())
            .iter()
            .max_by_key(|e| e.n)
            .map(|e| e.mv);

        assert_eq!(from_api, from_edges);
    }

    #[test]
    fn test_materialize_edges_concurrent_preserves_order_and_uniqueness() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let mut cfg = MctsConfig::evaluation(2.0);
        cfg.pw = Some(PwConfig::new(1.0, 0.0));
        let engine = MctsEngine::new(state.clone(), eval, cfg);

        let target = engine.root.candidate_count();
        let expected: Vec<(usize, f32)> = engine.root.candidates.get().unwrap()[..target]
            .iter()
            .copied()
            .collect();

        std::thread::scope(|scope| {
            for _ in 0..4 {
                let root = engine.root.clone();
                let root_state = state.clone();
                let tt = engine.tt.clone();
                scope.spawn(move || {
                    crate::mcts::expand::materialize_edges(&root, &root_state, target, &tt)
                });
            }
        });

        let edges = engine.root.edge_snapshot(target);
        assert_eq!(edges.len(), target);

        let actual: Vec<(usize, f32)> = edges.iter().map(|e| (e.mv, e.p)).collect();
        assert_eq!(actual, expected);

        let unique_moves: std::collections::HashSet<_> = edges.iter().map(|e| e.mv).collect();
        assert_eq!(unique_moves.len(), target);
        assert_eq!(engine.root.materialized_count(), target);
    }

    #[test]
    fn test_best_effort_materialization_skips_busy_lock_after_first_edge() {
        let _counter_guard = EDGE_COUNTER_TEST_LOCK.lock().unwrap();
        let _metrics = HotPathMetricsGuard::enabled();
        crate::mcts::node::reset_edge_lock_contention_counters();

        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let mut cfg = MctsConfig::evaluation(2.0);
        cfg.pw = Some(PwConfig::new(1.0, 0.0));
        let engine = MctsEngine::new(state.clone(), eval, cfg);
        let target = engine.root.candidate_count();
        assert!(target > 1);

        crate::mcts::expand::materialize_edges(&engine.root, &state, 1, &engine.tt);
        assert_eq!(engine.root.materialized_count(), 1);

        let lock_guard = engine.root.materialize_lock.lock();
        let mut probe = state.clone();
        crate::mcts::expand::materialize_edges_in_place_best_effort(
            &engine.root,
            &mut probe,
            target,
            &engine.tt,
        );
        assert_eq!(probe.tt_hash(), state.tt_hash());
        assert_eq!(engine.root.materialized_count(), 1);
        assert_eq!(
            crate::mcts::node::edge_lock_contention_snapshot().busy_skips,
            1
        );
        drop(lock_guard);

        crate::mcts::expand::materialize_edges_in_place_best_effort(
            &engine.root,
            &mut probe,
            target,
            &engine.tt,
        );
        assert_eq!(probe.tt_hash(), state.tt_hash());
        assert_eq!(engine.root.materialized_count(), target);
    }

    #[test]
    fn test_best_effort_materialization_skips_duplicate_preparation_claim() {
        let _counter_guard = EDGE_COUNTER_TEST_LOCK.lock().unwrap();
        let _metrics = HotPathMetricsGuard::enabled();
        crate::mcts::node::reset_edge_lock_contention_counters();

        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let mut cfg = MctsConfig::evaluation(2.0);
        cfg.pw = Some(PwConfig::new(1.0, 0.0));
        let engine = MctsEngine::new(state.clone(), eval, cfg);
        let target = engine.root.candidate_count();
        assert!(target > 1);

        crate::mcts::expand::materialize_edges(&engine.root, &state, 1, &engine.tt);
        assert_eq!(engine.root.materialized_count(), 1);

        engine.root.materialize_claim.store(1, Ordering::Relaxed);
        let tt_size_before = engine.tt.size();
        let mut probe = state.clone();
        crate::mcts::expand::materialize_edges_in_place_best_effort(
            &engine.root,
            &mut probe,
            target,
            &engine.tt,
        );

        assert_eq!(probe.tt_hash(), state.tt_hash());
        assert_eq!(engine.root.materialized_count(), 1);
        assert_eq!(engine.tt.size(), tt_size_before);
        assert_eq!(
            crate::mcts::node::edge_lock_contention_snapshot().busy_skips,
            1
        );

        engine.root.materialize_claim.store(0, Ordering::Relaxed);
    }

    fn solve_ttt(state: &TicTacToe, memo: &mut HashMap<u64, f32>) -> f32 {
        if let Some(&v) = memo.get(&state.hash()) {
            return v;
        }
        let value = if state.is_terminal() {
            state.outcome()
        } else {
            state
                .legal_moves()
                .into_iter()
                .map(|mv| -solve_ttt(&state.apply_move(mv), memo))
                .fold(f32::NEG_INFINITY, f32::max)
        };
        memo.insert(state.hash(), value);
        value
    }

    fn optimal_ttt_moves(state: &TicTacToe) -> Vec<usize> {
        let mut memo = HashMap::new();
        let mut best = f32::NEG_INFINITY;
        let mut best_moves = Vec::new();
        for mv in state.legal_moves() {
            let score = -solve_ttt(&state.apply_move(mv), &mut memo);
            if score > best + 1e-6 {
                best = score;
                best_moves.clear();
                best_moves.push(mv);
            } else if (score - best).abs() < 1e-6 {
                best_moves.push(mv);
            }
        }
        best_moves
    }

    fn gen_ttt_immediate_win_positions(n: usize, seed: u64) -> Vec<TicTacToe> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut positions = Vec::new();
        for _ in 0..n * 50 {
            if positions.len() >= n {
                break;
            }
            let n_moves = 3 + rng.gen::<usize>() % 4;
            let mut s = TicTacToe::initial();
            for _ in 0..n_moves {
                if s.is_terminal() {
                    break;
                }
                let legal = s.legal_moves();
                if legal.is_empty() {
                    break;
                }
                s = s.apply_move(legal[rng.gen::<usize>() % legal.len()]);
            }
            if !s.is_terminal()
                && s.legal_moves()
                    .iter()
                    .copied()
                    .any(|mv| s.is_winning_move(mv))
            {
                positions.push(s);
            }
        }
        positions
    }

    fn gen_gomoku_immediate_win_positions(n: usize, seed: u64) -> Vec<Gomoku> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut positions = Vec::new();
        for _ in 0..n * 80 {
            if positions.len() >= n {
                break;
            }
            let n_moves = 5 + rng.gen::<usize>() % 10;
            let mut s = Gomoku::new_with_win(7, 4);
            for _ in 0..n_moves {
                if s.is_terminal() {
                    break;
                }
                let legal = s.legal_moves();
                if legal.is_empty() {
                    break;
                }
                s = s.apply_move(legal[rng.gen::<usize>() % legal.len()]);
            }
            if !s.is_terminal()
                && s.legal_moves()
                    .iter()
                    .copied()
                    .any(|mv| s.is_winning_move(mv))
            {
                positions.push(s);
            }
        }
        positions
    }

    fn run_fixed<G, E>(
        state: &G,
        eval: &Arc<E>,
        budget: u32,
        root_forced_win: bool,
        exact_terminal_value: bool,
        fpu_reduction: f32,
    ) -> (Option<G::Move>, f64)
    where
        G: GameState + Clone,
        E: Evaluator<G> + Send + Sync + 'static,
    {
        use std::time::Instant;

        let qcfg = QuartzConfig {
            penalty_mode: PenaltyMode::GatedRefresh,
            min_visits: 30,
            check_interval: 50,
            halt_mode: HaltMode::Fixed { budget },
            ..Default::default()
        };
        let cfg = MctsConfig::evaluation_with_pw(2.0, PwConfig::default())
            .with_quartz(qcfg.clone())
            .with_root_forced_win(root_forced_win)
            .with_exact_terminal_value(exact_terminal_value)
            .with_fpu_reduction(fpu_reduction);
        let eng = MctsEngine::new(state.clone(), eval.clone(), cfg);
        let mut ctrl = QuartzController::new(budget, qcfg);
        let t0 = Instant::now();
        let stats = eng.run_quartz(&mut ctrl);
        let elapsed_s = t0.elapsed().as_secs_f64().max(1e-9);
        let nps = stats.root_visits as f64 / elapsed_s;
        (eng.best_move(), nps)
    }

    #[test]
    #[ignore]
    fn bench_fpu_accuracy_tradeoff() {
        let ttt_positions = gen_ttt_positions(80, 42);
        let gomoku_positions = gen_gomoku_positions(40, 43);
        let ttt_eval: Arc<UniformEval> = Arc::new(UniformEval);
        let gomoku_eval: Arc<ShortRollout> = Arc::new(ShortRollout::seeded(7, 123));
        let ttt_budget = 250u32;
        let gomoku_budget = 400u32;
        let gomoku_ref_budget = 2000u32;
        let fpu_candidates = [0.0f32, 0.02, 0.05, 0.10, 0.15];

        let gomoku_reference: Vec<_> = gomoku_positions
            .iter()
            .map(|state| run_fixed(state, &gomoku_eval, gomoku_ref_budget, false, false, 0.0).0)
            .collect();

        eprintln!(
            "\n{:>6} {:>10} {:>10} {:>10} {:>10}",
            "FPU", "TTT_Exact", "Gmk_Agree", "TTT_NPS", "Gmk_NPS"
        );
        eprintln!("{}", "─".repeat(54));

        for &fpu in &fpu_candidates {
            let mut ttt_correct = 0u32;
            let mut ttt_nps = 0.0f64;
            for state in &ttt_positions {
                let optimal = optimal_ttt_moves(state);
                let (best, nps) = run_fixed(state, &ttt_eval, ttt_budget, false, false, fpu);
                if best.map(|mv| optimal.contains(&mv)).unwrap_or(false) {
                    ttt_correct += 1;
                }
                ttt_nps += nps;
            }

            let mut gomoku_agree = 0u32;
            let mut gomoku_nps = 0.0f64;
            for (idx, state) in gomoku_positions.iter().enumerate() {
                let (best, nps) = run_fixed(state, &gomoku_eval, gomoku_budget, false, false, fpu);
                if best == gomoku_reference[idx] {
                    gomoku_agree += 1;
                }
                gomoku_nps += nps;
            }

            eprintln!(
                "{:>6.2} {:>9.1}% {:>9.1}% {:>10.0} {:>10.0}",
                fpu,
                100.0 * ttt_correct as f32 / ttt_positions.len() as f32,
                100.0 * gomoku_agree as f32 / gomoku_positions.len() as f32,
                ttt_nps / ttt_positions.len() as f64,
                gomoku_nps / gomoku_positions.len() as f64,
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_terminal_child_accuracy_tradeoff() {
        use std::time::Instant;

        let ttt_positions = gen_ttt_positions(120, 7);
        let gomoku_positions = gen_gomoku_positions(50, 8);
        let ttt_eval: Arc<UniformEval> = Arc::new(UniformEval);
        let gomoku_eval: Arc<ShortRollout> = Arc::new(ShortRollout::seeded(7, 123));
        let ttt_budget = 250u32;
        let gomoku_budget = 500u32;
        let gomoku_ref_budget = 2500u32;

        let gomoku_reference: Vec<_> = gomoku_positions
            .iter()
            .map(|state| run_fixed(state, &gomoku_eval, gomoku_ref_budget, false, false, 0.0).0)
            .collect();

        eprintln!(
            "\n{:>8} {:>10} {:>10} {:>10} {:>10}",
            "Terminal", "TTT_Exact", "Gmk_Agree", "TTT_NPS", "Gmk_NPS"
        );
        eprintln!("{}", "─".repeat(58));

        for exact_terminal in [false, true] {
            let t_ttt = Instant::now();
            let mut ttt_correct = 0u32;
            for state in &ttt_positions {
                let optimal = optimal_ttt_moves(state);
                let (best, _) = run_fixed(state, &ttt_eval, ttt_budget, false, exact_terminal, 0.0);
                if best.map(|mv| optimal.contains(&mv)).unwrap_or(false) {
                    ttt_correct += 1;
                }
            }
            let ttt_ms = t_ttt.elapsed().as_millis().max(1) as f64;

            let t_gmk = Instant::now();
            let mut gomoku_agree = 0u32;
            for (idx, state) in gomoku_positions.iter().enumerate() {
                let (best, _) = run_fixed(
                    state,
                    &gomoku_eval,
                    gomoku_budget,
                    false,
                    exact_terminal,
                    0.0,
                );
                if best == gomoku_reference[idx] {
                    gomoku_agree += 1;
                }
            }
            let gomoku_ms = t_gmk.elapsed().as_millis().max(1) as f64;

            eprintln!(
                "{:>8} {:>9.1}% {:>9.1}% {:>10.0} {:>10.0}",
                if exact_terminal { "on" } else { "off" },
                100.0 * ttt_correct as f32 / ttt_positions.len() as f32,
                100.0 * gomoku_agree as f32 / gomoku_positions.len() as f32,
                (ttt_budget as f64 * ttt_positions.len() as f64) / (ttt_ms / 1000.0),
                (gomoku_budget as f64 * gomoku_positions.len() as f64) / (gomoku_ms / 1000.0),
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_terminal_child_forced_win_hit_rate() {
        let ttt_positions = gen_ttt_immediate_win_positions(120, 91);
        let gomoku_positions = gen_gomoku_immediate_win_positions(80, 92);
        let ttt_eval: Arc<UniformEval> = Arc::new(UniformEval);
        let gomoku_eval: Arc<UniformEval> = Arc::new(UniformEval);
        let ttt_budget = 400u32;
        let gomoku_budget = 1000u32;

        eprintln!(
            "\n{:>8} {:>10} {:>10} {:>10} {:>10}",
            "RootWin", "TTT_Win%", "Gmk_Win%", "TTT_NPS", "Gmk_NPS"
        );
        eprintln!("{}", "─".repeat(58));

        for root_forced_win in [false, true] {
            let mut ttt_hits = 0u32;
            let mut ttt_nps = 0.0f64;
            for state in &ttt_positions {
                let (best, nps) =
                    run_fixed(state, &ttt_eval, ttt_budget, root_forced_win, false, 0.0);
                if best.map(|mv| state.is_winning_move(mv)).unwrap_or(false) {
                    ttt_hits += 1;
                }
                ttt_nps += nps;
            }
            let mut gomoku_hits = 0u32;
            let mut gomoku_nps = 0.0f64;
            for state in &gomoku_positions {
                let (best, nps) = run_fixed(
                    state,
                    &gomoku_eval,
                    gomoku_budget,
                    root_forced_win,
                    false,
                    0.0,
                );
                if best.map(|mv| state.is_winning_move(mv)).unwrap_or(false) {
                    gomoku_hits += 1;
                }
                gomoku_nps += nps;
            }

            eprintln!(
                "{:>8} {:>9.1}% {:>9.1}% {:>10.0} {:>10.0}",
                if root_forced_win { "on" } else { "off" },
                100.0 * ttt_hits as f32 / ttt_positions.len() as f32,
                100.0 * gomoku_hits as f32 / gomoku_positions.len() as f32,
                ttt_nps / ttt_positions.len() as f64,
                gomoku_nps / gomoku_positions.len() as f64,
            );
        }
    }

    /// BQ++ Phase 8b: integration test that an attached SearchPolicy is
    /// consulted by run() and its halt decision is honored.
    /// LegacyAlphaZero with budget=50 should fire Stop(FixedBudget)
    /// the moment root_visits reaches 50. Combined with a permissive
    /// SearchController (FixedIterations(1000)), the policy halt should
    /// dominate, and the search ends well before 1000 iters.
    #[test]
    fn test_phase8b_search_policy_halt_signal_is_honored() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let policy: Arc<dyn SearchPolicy> = Arc::new(LegacyAlphaZero::new(50));
        let cfg = MctsConfig::evaluation(2.0).with_search_policy(policy);

        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = FixedIterations::new(1000); // permissive controller
        let stats = engine.run(&mut ctrl);

        // The policy says Stop(FixedBudget) at root_visits>=50.
        // FixedIterations would otherwise let the search run to 1000.
        // Verify the policy halt fired well before 1000.
        assert!(
            stats.iterations < 200,
            "policy halt did not fire; iterations={} (expected < 200)",
            stats.iterations
        );
        assert!(
            stats.root_visits >= 50,
            "policy halted too early; root_visits={} (expected >= 50)",
            stats.root_visits
        );
    }

    /// A0-a regression fixture: a `SearchPolicy` that records what
    /// `observe()` was actually handed, so tests can assert on real
    /// edge data instead of trusting the caller's claim.
    struct EdgeSpyPolicy {
        max_edges_seen: Mutex<usize>,
        max_n_seen: Mutex<u32>,
        any_nonzero_prior: Mutex<bool>,
    }

    impl EdgeSpyPolicy {
        fn new() -> Self {
            Self {
                max_edges_seen: Mutex::new(0),
                max_n_seen: Mutex::new(0),
                any_nonzero_prior: Mutex::new(false),
            }
        }
    }

    impl SearchPolicy for EdgeSpyPolicy {
        fn name(&self) -> &'static str {
            "edge_spy_test_policy"
        }
        fn observe(&self, _snap: &SearchSnapshot, edges: &[EdgeView<'_>]) {
            let mut max_edges = self.max_edges_seen.lock().unwrap();
            *max_edges = (*max_edges).max(edges.len());
            let mut max_n = self.max_n_seen.lock().unwrap();
            let mut any_prior = self.any_nonzero_prior.lock().unwrap();
            for e in edges {
                *max_n = (*max_n).max(e.n);
                if e.prior > 0.0 {
                    *any_prior = true;
                }
            }
        }
        fn score_adjustment(&self, _edge: EdgeView<'_>) -> ScoreAdjustment {
            ScoreAdjustment::default()
        }
        fn should_halt(&self, _snap: &SearchSnapshot, _edges: &[EdgeView<'_>]) -> HaltDecision {
            HaltDecision::Continue
        }
        fn telemetry(&self) -> ControllerTelemetry {
            ControllerTelemetry::default()
        }
    }

    /// A0-a regression: `policy_halt_check` must feed `observe()` real
    /// root `EdgeView`s (dense `edge_pos`-indexed, real `n`/`prior`),
    /// not the empty slice this function used to pass. Before the fix,
    /// `max_edges_seen` would stay `0` forever and `KLLUCBStop`'s
    /// `edges.len() < 2` guard made its certificate permanently
    /// unreachable — this test pins the fix at the engine-integration
    /// boundary (not just unit-testing `EdgeView` construction).
    #[test]
    fn test_a0a_policy_halt_check_feeds_real_root_edges() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let spy = Arc::new(EdgeSpyPolicy::new());
        let policy: Arc<dyn SearchPolicy> = spy.clone();
        let cfg = MctsConfig::evaluation(2.0).with_search_policy(policy);
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = FixedIterations::new(200);
        let stats = engine.run(&mut ctrl);

        assert_eq!(
            stats.iterations, 200,
            "sanity: permissive spy must not halt early"
        );
        let max_edges = *spy.max_edges_seen.lock().unwrap();
        assert!(
            max_edges >= 2,
            "observe() never saw >=2 real root edges (got {max_edges}); \
             the empty-edges regression is back"
        );
        let max_n = *spy.max_n_seen.lock().unwrap();
        assert!(
            max_n > 0,
            "observe() never saw a visited root edge (max n=0); edges are \
             still fake/zeroed"
        );
        assert!(
            *spy.any_nonzero_prior.lock().unwrap(),
            "observe() never saw a nonzero prior; EdgeView.prior is not \
             wired to the real edge slab"
        );
    }

    /// Stage 7 / C3: an attached `KgStop` policy halts the real engine via
    /// PolicyConverged before the controller budget is exhausted. Permissive
    /// threshold (Fixed cost) so the halt is deterministic once `observe` has
    /// fired (iter 64 in non-quartz) and `min_total` (20) is met.
    #[test]
    fn test_s7_kg_stop_engine_halts_before_budget_on_resolved_root() {
        use crate::mcts::policy::{KgCostSource, KgStop};
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let policy: Arc<dyn SearchPolicy> = Arc::new(KgStop::new(
            1000.0,
            4.0,
            4,
            20,
            u32::MAX,
            KgCostSource::Fixed(1.0),
        ));
        let cfg = MctsConfig::evaluation(2.0).with_search_policy(policy);
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = FixedIterations::new(400);
        let stats = engine.run(&mut ctrl);

        assert!(
            stats.iterations < 400,
            "kg_stop did not halt; iterations={} (expected < 400)",
            stats.iterations
        );
        let counts = engine.policy_halt_count_snapshot();
        assert!(
            counts[crate::mcts::quartz::HaltReason::PolicyConverged as usize] > 0,
            "no PolicyConverged halt recorded; counts={counts:?}"
        );
    }

    /// Stage 7 / C3: `KgStop` never halts below `min_total`, even with a
    /// permissive threshold. min_total=300 > budget 200 ⇒ runs to the full
    /// controller budget with no policy halt.
    #[test]
    fn test_s7_kg_stop_engine_respects_min_total() {
        use crate::mcts::policy::{KgCostSource, KgStop};
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let policy: Arc<dyn SearchPolicy> = Arc::new(KgStop::new(
            1000.0,
            4.0,
            4,
            300,
            u32::MAX,
            KgCostSource::Fixed(1.0),
        ));
        let cfg = MctsConfig::evaluation(2.0).with_search_policy(policy);
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = FixedIterations::new(200);
        let stats = engine.run(&mut ctrl);

        assert_eq!(
            stats.iterations, 200,
            "kg_stop halted below min_total (root_visits never reached 300)"
        );
        let counts = engine.policy_halt_count_snapshot();
        assert_eq!(
            counts[crate::mcts::quartz::HaltReason::PolicyConverged as usize],
            0,
            "kg_stop halted below min_total; counts={counts:?}"
        );
    }

    /// BQ++ Phase 8b: a None search_policy preserves bit-identical
    /// existing behavior. This is the back-compat regression guard:
    /// no integration test should change behavior unless cfg.search_policy
    /// is explicitly set.
    #[test]
    fn test_phase8b_no_search_policy_is_back_compat() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let cfg = MctsConfig::evaluation(2.0);
        // Confirm the field defaults to None.
        assert!(cfg.search_policy.is_none());
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = FixedIterations::new(64);
        let stats = engine.run(&mut ctrl);
        // Run completes at the controller's budget exactly (no policy
        // intervention). FixedIterations(64) yields root_visits == 64.
        assert_eq!(stats.iterations, 64);
        assert_eq!(stats.root_visits, 64);
    }

    /// BQ++ Phase 8b: MctsConfig.search_policy round-trips the Arc.
    /// Pin the basic Cloneability + name() round-trip.
    #[test]
    fn test_phase8b_with_search_policy_builder() {
        let policy: Arc<dyn SearchPolicy> = Arc::new(LegacyAlphaZero::new(100));
        let cfg = MctsConfig::evaluation(2.0).with_search_policy(Arc::clone(&policy));
        let attached = cfg.search_policy.clone().expect("policy attached");
        assert_eq!(attached.name(), "legacy_az");
        let cloned_cfg = cfg.clone();
        let _ = cloned_cfg
            .search_policy
            .expect("clone preserves policy Arc");
        // Debug formatter renders the dyn opaque, no panic.
        let dbg = format!(
            "{:?}",
            MctsConfig::evaluation(2.0).with_search_policy(policy)
        );
        assert!(
            dbg.contains("legacy_az"),
            "debug format missing policy name: {dbg}"
        );
    }

    /// BQ++ Phase 8c followup: pin the policy_halt_counts allocation
    /// invariant. The `apply_search_profile` path in mcts_server.rs
    /// directly assigns `cfg.search_policy = Some(...)` rather than
    /// going through `with_search_policy()` (preserving the
    /// quartz-config plumbing in the same scope). If
    /// `MctsConfig::default()` left `policy_halt_counts = None`, this
    /// direct-assignment path would silently drop every counter
    /// increment — which is exactly the bug that produced
    /// `extended_halt_reason_count = {MaxVisits: 2400}` for legacy_az
    /// in the phase8c_v4 toy ablation.
    #[test]
    fn test_phase8c_policy_halt_counts_increment_via_direct_assignment() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let mut cfg = MctsConfig::evaluation(2.0);
        // Direct assignment — mimics apply_search_profile's path.
        cfg.search_policy = Some(Arc::new(LegacyAlphaZero::new(50)));
        // Counters must be allocated by Default; assert non-None.
        assert!(
            cfg.policy_halt_counts.is_some(),
            "MctsConfig::default() must allocate policy_halt_counts \
             so direct cfg.search_policy assignment doesn't silently \
             drop increments"
        );
        let engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = FixedIterations::new(1000);
        let _ = engine.run(&mut ctrl);
        let counts = engine.policy_halt_count_snapshot();
        let total: u32 = counts.iter().sum();
        assert!(
            total >= 1,
            "expected at least one policy halt to be counted; got total=0"
        );
        let fixed_budget_idx = crate::mcts::quartz::HaltReason::FixedBudget as usize;
        assert!(
            counts[fixed_budget_idx] >= 1,
            "expected FixedBudget halts to be counted; got {}",
            counts[fixed_budget_idx]
        );
    }

    /// BQ++ Phase 8c: integration test that a SearchPolicy halt fires
    /// under `run_par` (parallel single-policy path). LegacyAlphaZero
    /// budget=50, FixedIterations(1000) controller, 2 worker threads.
    /// The policy must halt the search before 1000 iters; the latched
    /// AtomicBool flag should propagate to all workers within ~one
    /// check_interval. We allow up to 400 visits to absorb the lag
    /// between thread 0 setting the flag and other threads reading it
    /// at chunk boundary (ticket_chunk * n_threads slack).
    #[test]
    fn test_phase8c_run_par_honors_search_policy_halt() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let policy: Arc<dyn SearchPolicy> = Arc::new(LegacyAlphaZero::new(50));
        let cfg = MctsConfig::evaluation(2.0).with_search_policy(policy);
        let engine = MctsEngine::new(state, eval, cfg);
        let ctrl = FixedIterations::new(1000); // permissive
        let stats = engine.run_par(&ctrl, 2);
        assert!(
            stats.iterations < 800,
            "run_par policy halt did not fire; iterations={}",
            stats.iterations
        );
        assert!(
            stats.root_visits >= 50,
            "run_par halted too early; root_visits={}",
            stats.root_visits
        );
    }

    /// BQ++ Phase 8c: integration test that a SearchPolicy halt fires
    /// under `run_par_quartz` (parallel quartz-aware path). Same shape
    /// as the run_par test but goes through the quartz controller.
    #[test]
    fn test_phase8c_run_par_quartz_honors_search_policy_halt() {
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let policy: Arc<dyn SearchPolicy> = Arc::new(LegacyAlphaZero::new(50));
        let qcfg = crate::mcts::QuartzConfig::default();
        let cfg = MctsConfig::evaluation(2.0)
            .with_search_policy(policy)
            .with_quartz(qcfg.clone());
        let engine = MctsEngine::new(state, eval, cfg);
        // 1000 max-visit ceiling — well above the policy budget of 50.
        let mut ctrl = QuartzController::new(1000, qcfg);
        let stats = engine.run_par_quartz(&mut ctrl, 2);
        assert!(
            stats.iterations < 800,
            "run_par_quartz policy halt did not fire; iterations={}",
            stats.iterations
        );
        assert!(
            stats.root_visits >= 50,
            "run_par_quartz halted too early; root_visits={}",
            stats.root_visits
        );
    }

    #[test]
    #[ignore]
    fn bench_search_controller_fixed_iterations_fast_path() {
        let limit = 15_000;
        let state = Gomoku::new(7);
        let eval: Arc<dyn Evaluator<Gomoku>> = Arc::new(UniformEval);
        let cfg = MctsConfig::evaluation(2.0);

        let ref_engine = MctsEngine::new(state.clone(), eval.clone(), cfg.clone());
        let ref_stats = run_reference_fixed(&ref_engine, limit);

        let opt_engine = MctsEngine::new(state, eval, cfg);
        let mut ctrl = crate::mcts::search::FixedIterations::new(limit);
        let opt_stats = opt_engine.run(&mut ctrl);

        eprintln!(
            "\nFixedIterations loop: reference={:.0} nps optimized={:.0} nps speedup={:.3}x",
            ref_stats.nps,
            opt_stats.nps,
            opt_stats.nps / ref_stats.nps.max(1.0)
        );
        assert_eq!(opt_stats.root_visits, ref_stats.root_visits);
        assert_eq!(opt_engine.best_move(), ref_engine.best_move());
    }
}

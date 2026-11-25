//! Expansion — Compact candidates + Lazy materialization
//!
//! Phase 1: expand_and_evaluate(node, state, evaluator, tt, pw)
//!   - evaluator.evaluate(state) → policy + value
//!   - candidates 저장: (mv, prior) 쌍, prior 내림차순, OnceLock CAS
//!   - k_initial개만 즉시 materialize (apply_move + TT 등록)
//!
//! Phase 2: materialize_edges<G>(node, state, target, tt)
//!   - select.rs에서 호출: state AT this node 전달
//!   - candidates[current..target] 에 대해 apply_move → TT 등록
//!   - Mutex 내 append-only push, double-check pattern

use std::sync::atomic::Ordering;

use smallvec::SmallVec;

use crate::game::{EvalResult, Evaluator, GameState};
use crate::mcts::mod_types::PwConfig;
use crate::mcts::node::{ArenaRef, MctsEdge, MctsNode};
use crate::mcts::tt::{TranspositionTable, TtLookup};

#[derive(Clone, Copy)]
struct PendingEdge<M: Copy + Send + Sync + 'static> {
    mv: M,
    prior: f32,
    child: ArenaRef<MctsNode<M>>,
}

struct BestEffortMaterializeClaim<'a, M> {
    node: &'a MctsNode<M>,
    active: bool,
}

impl<'a, M> BestEffortMaterializeClaim<'a, M> {
    #[inline]
    fn none(node: &'a MctsNode<M>) -> Self {
        Self {
            node,
            active: false,
        }
    }

    #[inline]
    fn try_acquire(node: &'a MctsNode<M>) -> Option<Self> {
        node.materialize_claim
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| Self { node, active: true })
    }
}

impl<M> Drop for BestEffortMaterializeClaim<'_, M> {
    #[inline]
    fn drop(&mut self) {
        if self.active {
            self.node.materialize_claim.store(0, Ordering::Release);
        }
    }
}

// ─────────────────────────────────────────────
// § Phase 1: expand_and_evaluate
// ─────────────────────────────────────────────

#[inline]
pub fn expand_and_evaluate<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &G,
    evaluator: &dyn Evaluator<G>,
    tt: &TranspositionTable<G::Move>,
    pw: Option<&PwConfig>,
) -> f32 {
    if let Some(v) = node.terminal_value {
        return v;
    }

    let eval = evaluator.evaluate(state);
    expand_with_result(node, state, eval, tt, pw)
}

#[inline]
pub fn expand_with_result<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &G,
    eval: EvalResult<G::Move>,
    tt: &TranspositionTable<G::Move>,
    pw: Option<&PwConfig>,
) -> f32 {
    if let Some(v) = node.terminal_value {
        return v;
    }

    let (value, k_initial) = publish_candidates::<G>(node, eval, pw);
    materialize_edges(node, state, k_initial, tt);
    value
}

#[inline]
pub fn expand_and_evaluate_in_place<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &mut G,
    evaluator: &dyn Evaluator<G>,
    tt: &TranspositionTable<G::Move>,
    pw: Option<&PwConfig>,
) -> f32 {
    if let Some(v) = node.terminal_value {
        return v;
    }

    let eval = evaluator.evaluate(state);
    expand_with_result_in_place(node, state, eval, tt, pw)
}

#[inline]
pub fn expand_with_result_in_place<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &mut G,
    eval: EvalResult<G::Move>,
    tt: &TranspositionTable<G::Move>,
    pw: Option<&PwConfig>,
) -> f32 {
    if let Some(v) = node.terminal_value {
        return v;
    }

    let (value, k_initial) = publish_candidates::<G>(node, eval, pw);
    materialize_edges_in_place(node, state, k_initial, tt);
    value
}

#[inline]
fn publish_candidates<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    eval: EvalResult<G::Move>,
    pw: Option<&PwConfig>,
) -> (f32, usize) {
    let value = eval.value;
    // [OPT] Use policy directly as candidates — avoids redundant legal_moves() call
    // and eliminates O(n²) policy-to-legal lookup.
    // Evaluator.evaluate() returns policy containing only legal moves with priors.
    let mut candidates = eval.policy;
    if candidates.is_empty() {
        return (value, 0);
    }

    // Clamp negative priors to 0 and skip sorting when the evaluator already
    // returns descending priors. UniformEval is the dominant CPU profile path,
    // and its equal-prior policy would otherwise pay O(n log n) per expansion.
    let mut is_descending = true;
    let mut prev = f32::INFINITY;
    for entry in candidates.iter_mut() {
        let prior = entry.1.max(0.0);
        if prior > prev {
            is_descending = false;
        }
        entry.1 = prior;
        prev = prior;
    }
    if !is_descending || !G::can_skip_sorted_policy_resort() {
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    let n_candidates = candidates.len();

    // ── 2. CAS exactly-once ─────────────────────────────────────
    let _ = node.candidates.set(candidates.into_boxed_slice());

    // ── 3. k_initial 개 즉시 materialization ────────────────────
    let k_initial = match pw {
        Some(cfg) => cfg.k(0).max(1).min(n_candidates),
        None => n_candidates,
    };
    (value, k_initial)
}

// ─────────────────────────────────────────────
// § Phase 2: materialize_edges
// ─────────────────────────────────────────────

/// `target`개까지 edges materialization
/// state: 이 노드에 해당하는 게임 상태 (apply_move용)
///
/// Phase 7 C (2026-04-26): edge buffer is a raw slab in the TT bucket's
/// bumpalo Bump (not a `Vec`). Allocation is done via
/// `tt.allocate_edge_slab(...)` (takes the bucket write lock for the
/// one-shot Bump alloc, then releases it). Child hashes and TT nodes are
/// prepared before taking the per-node `materialize_lock`; the lock only
/// protects raw-slot writes and the Release-store of `edge_cursor`.
#[inline]
pub fn materialize_edges<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &G,
    target: usize,
    tt: &TranspositionTable<G::Move>,
) {
    if !needs_materialization::<G>(node, target) {
        return;
    }
    let mut probe = state.clone();
    materialize_edges_in_place_impl(node, &mut probe, target, tt, false);
}

#[inline]
pub fn materialize_edges_in_place<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &mut G,
    target: usize,
    tt: &TranspositionTable<G::Move>,
) {
    materialize_edges_in_place_impl(node, state, target, tt, false);
}

/// Parallel select may use the already-published prefix when another worker is
/// widening the same node. This preserves blocking publication for the first
/// visible edge while avoiding idle workers on later progressive-widening
/// increments.
#[inline]
pub fn materialize_edges_in_place_best_effort<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &mut G,
    target: usize,
    tt: &TranspositionTable<G::Move>,
) {
    materialize_edges_in_place_impl(node, state, target, tt, true);
}

#[inline]
fn needs_materialization<G: GameState>(node: &ArenaRef<MctsNode<G::Move>>, target: usize) -> bool {
    let candidates = match node.candidates.get() {
        Some(c) => c,
        None => return false,
    };
    let actual = target.min(candidates.len());

    let current = node.edge_cursor.load(Ordering::Acquire) as usize;
    current < actual
}

#[inline]
fn materialize_edges_in_place_impl<G: GameState>(
    node: &ArenaRef<MctsNode<G::Move>>,
    state: &mut G,
    target: usize,
    tt: &TranspositionTable<G::Move>,
    best_effort_after_first_edge: bool,
) {
    let candidates = match node.candidates.get() {
        Some(c) => c,
        None => return,
    };
    let actual = target.min(candidates.len());

    let current = node.edge_cursor.load(Ordering::Acquire) as usize;
    if current >= actual {
        return;
    }

    let _best_effort_claim = if best_effort_after_first_edge && current > 0 {
        match BestEffortMaterializeClaim::try_acquire(node) {
            Some(claim) => claim,
            None => {
                crate::mcts::node::record_edges_materialize_busy_skip();
                return;
            }
        }
    } else {
        BestEffortMaterializeClaim::none(node)
    };

    if best_effort_after_first_edge && current > 0 {
        match node.materialize_lock.try_lock() {
            Some(guard) => drop(guard),
            None => {
                crate::mcts::node::record_edges_materialize_busy_skip();
                return;
            }
        }
    }

    // Allocate / look up the slab. Idempotent: only the first caller takes the
    // bucket write lock; subsequent calls observe the published `edges_ptr` and
    // return its (ptr, cap).
    let (slab_ptr, slab_cap) = tt.allocate_edge_slab::<G::Move>(&**node, candidates.len());
    debug_assert!(actual <= slab_cap as usize);

    let planned_start = current;
    let pending = prepare_edge_slots(candidates, planned_start, actual, state, tt);

    let _mat_guard = {
        let lock_started = crate::mcts::profiling::maybe_start_timer();
        let guard = if best_effort_after_first_edge && current > 0 {
            match node.materialize_lock.try_lock() {
                Some(guard) => guard,
                None => {
                    crate::mcts::node::record_edges_materialize_busy_skip();
                    return;
                }
            }
        } else {
            node.materialize_lock.lock()
        };
        if let Some(t0) = lock_started {
            crate::mcts::node::record_edges_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        guard
    };

    let current = node.edge_cursor.load(Ordering::Relaxed) as usize;
    if current >= actual {
        return;
    }

    materialize_edge_slots(
        node,
        current.max(planned_start),
        actual,
        planned_start,
        slab_ptr,
        slab_cap,
        &pending,
    );
}

#[inline(always)]
fn prepare_edge_slots<G: GameState>(
    candidates: &[(G::Move, f32)],
    planned_start: usize,
    actual: usize,
    state: &mut G,
    tt: &TranspositionTable<G::Move>,
) -> SmallVec<[PendingEdge<G::Move>; 64]> {
    let n_pending = actual.saturating_sub(planned_start);
    let mut move_priors = SmallVec::<[(G::Move, f32); 64]>::with_capacity(n_pending);
    let mut lookups = SmallVec::<[TtLookup<G::Move>; 64]>::with_capacity(n_pending);

    for i in planned_start..actual {
        let (mv, prior) = candidates[i];
        let undo = state.apply_move_in_place(mv);
        let child_hash = state.tt_hash();
        // Hint the prefetcher to start fetching the TT bucket cache line
        // we're about to probe. The is_terminal + outcome + undo_move
        // sequence below gives ~30 cycles of latency hiding before
        // tt.get_or_create reads the same line under the read lock.
        // Profiling (artifacts/profiling_20260428) attributed 54.7 % of
        // all D1 read misses to TT::get_or_create on this workload.
        tt.prefetch(child_hash);
        let tv = if state.is_terminal() {
            Some(state.outcome())
        } else {
            None
        };
        state.undo_move(undo);

        move_priors.push((mv, prior));
        lookups.push(TtLookup::new(child_hash, tv));
    }

    tt.get_or_create_batch(&mut lookups);

    let mut pending = SmallVec::with_capacity(n_pending);
    for ((mv, prior), lookup) in move_priors.into_iter().zip(lookups.into_iter()) {
        let child = lookup
            .node
            .expect("TT batch lookup must resolve every child node");
        pending.push(PendingEdge { mv, prior, child });
    }
    pending
}

#[inline(always)]
fn materialize_edge_slots<M: Copy + Send + Sync + 'static>(
    node: &ArenaRef<MctsNode<M>>,
    current: usize,
    actual: usize,
    planned_start: usize,
    slab_ptr: *mut MctsEdge<M>,
    _slab_cap: u32,
    pending: &[PendingEdge<M>],
) {
    debug_assert!(current >= planned_start);
    debug_assert!(actual >= current);
    debug_assert!(actual - planned_start <= pending.len());
    for i in current..actual {
        let PendingEdge { mv, prior, child } = pending[i - planned_start];
        // SAFETY: we hold `materialize_lock`, so no other writer is
        // touching slot `i`. `i < slab_cap` was established by the
        // `actual <= slab_cap` debug_assert above. The slot has not
        // been initialized before (cursor < i+1 at this point), so
        // `std::ptr::write` does NOT drop any prior value — it just
        // writes the new edge.
        unsafe {
            std::ptr::write(slab_ptr.add(i), MctsEdge::new(mv, child, prior));
        }
    }
    // Single Release-store publishes every slot we just wrote.
    // Lock-free readers see either the old cursor (and old slots) or
    // the new cursor (and all slots through `actual`).
    node.edge_cursor.store(actual as u32, Ordering::Release);
}

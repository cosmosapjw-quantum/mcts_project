//! MCTS 노드 구조체 (v0.3.1 — Compact Candidates)
//!
//! 메모리 최적화 변경:
//!   candidates: Box<[(M, f32, u64, Option<f32>)]>  →  Box<[(M, f32)]>
//!   (28 bytes/entry → 12 bytes/entry, 57% 절약)
//!
//!   child_hash, terminal_value는 candidates에 저장하지 않고,
//!   materialize_edges(node, state, ...) 시점에 state.apply_move(mv)로 lazy 계산.
//!
//! 불변식 (Phase 7 C, 2026-04-26):
//!   1. candidates: OnceLock — CAS exactly-once
//!   2. edges_ptr: AtomicPtr<MctsEdge<M>> — null until first materialize_edges
//!      call, then points to a slab allocated in the TT bucket's bumpalo Bump.
//!      Slab capacity = candidates.len() (set once at expand time).
//!   3. edge_cursor: AtomicU32 — number of materialized edges. Acts as the
//!      Release-store of the slab writes; readers use Acquire load + raw
//!      slice construction (no lock).
//!   4. materialize_claim: AtomicU32 — best-effort parallel widening owner
//!      flag. After at least one edge is visible, parallel selectors skip
//!      duplicate edge preparation when another worker is already preparing
//!      this node's next widening step.
//!   5. materialize_lock: parking_lot::Mutex<()> — serializes concurrent
//!      materialize_edges calls on the same node.
//!
//! Pre-Phase-7 layout (kept for reference):
//!   edges: RwLock<Vec<MctsEdge<M>>> — append-only Vec under parking_lot
//!   RwLock. PUCT read path took an uncontended read lock per visit. Replaced
//!   to remove RwLock contention from the hot path AND to shed the per-search
//!   global-allocator traffic for the Vec backing buffer (~30 K nodes/search
//!   × 49 edges ≈ 1.5 M MctsEdge<M> allocations/search → now slab-allocated
//!   in the bucket Bump alongside the node bodies).

use parking_lot::Mutex as PlMutex;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

// ─────────────────────────────────────────────
// § ArenaRef<T> — bumpalo-arena-allocated node reference
// ─────────────────────────────────────────────
//
// Replaces Arc<MctsNode<M>> (Phase 3, 2026-04-25). Per the audit, 760 K
// `Arc<MctsNode>` allocations accounted for 86 % of remaining heap traffic
// and 51 % of all D1 read-misses on scenario A. Moving node bodies into a
// per-bucket `bumpalo::Bump` (see `mcts::tt::ArenaPool`) eliminates the
// global-allocator round-trip and packs node bodies contiguously per bucket.
//
// Safety invariant
//   The pointed-to `MctsNode` lives in a `bumpalo::Bump` that is owned by
//   the `MctsEngine` via `Arc<ArenaPool<M>>`. All `ArenaRef<MctsNode>`
//   instances in the program are reachable only from the engine's TT or
//   from `Vec<MctsEdge>` lists hanging off TT-stored nodes. The engine is
//   declared with `tt` BEFORE `pool`, so on `MctsEngine` drop the TT (and
//   its node references) drops first, then the pool frees the Bumps. No
//   dangling reference can escape the engine.
//
// Bumpalo guarantees: allocations never move; `Bump::alloc` returns
// addresses that remain valid until the Bump is dropped or `reset()` is
// called (we never call `reset` on engine-owned pools).
//
// Sync: `MctsNode<M>` is `Sync` (atomics + parking_lot::RwLock with
// inline contents). A shared raw pointer to a `Sync` type is itself `Sync`,
// so the unsafe impls below are sound.

/// A non-owning reference to an `MctsNode` allocated in an `ArenaPool`'s
/// bumpalo arena. Cheap to copy; does not refcount. The arena keeps the
/// pointee alive for as long as `Arc<ArenaPool<M>>` is held by the engine.
pub struct ArenaRef<T> {
    ptr: NonNull<T>,
    // `*const T` carries no lifetime obligation (so `T: 'static` is not
    // required at the type level). Send/Sync are hand-implemented below
    // with the `T: Sync` bound.
    _marker: PhantomData<*const T>,
}

impl<T> ArenaRef<T> {
    /// # Safety
    /// `ptr` must point to a `T` that lives in an arena whose lifetime
    /// outlives every clone of this `ArenaRef`. The engine's
    /// `Arc<ArenaPool>` Drop-order discipline (see `MctsEngine`) is the
    /// guarantor.
    #[inline]
    pub unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr()
    }

    #[inline]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.ptr.as_ptr() == b.ptr.as_ptr()
    }
}

impl<T> Clone for ArenaRef<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for ArenaRef<T> {}

impl<T> Deref for ArenaRef<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: see ArenaRef invariant above.
        unsafe { self.ptr.as_ref() }
    }
}

// SAFETY: ArenaRef is a shared raw pointer to a `Sync` payload. Sharing it
// across threads is sound iff the pointee is Sync, which the bound enforces.
unsafe impl<T: Sync> Send for ArenaRef<T> {}
unsafe impl<T: Sync> Sync for ArenaRef<T> {}

impl<T> std::fmt::Debug for ArenaRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArenaRef({:p})", self.ptr.as_ptr())
    }
}

// ─────────────────────────────────────────────
// § Atomic f64 헬퍼
// ─────────────────────────────────────────────

pub fn atomic_f64_add(a: &AtomicU64, delta: f64) {
    let mut cur = a.load(Ordering::Relaxed);
    let mut spins = 0u32;
    loop {
        let new = f64::from_bits(cur) + delta;
        match a.compare_exchange_weak(cur, new.to_bits(), Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return,
            Err(v) => {
                cur = v;
                spins += 1;
                if spins <= 3 {
                    core::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

pub fn atomic_f64_load(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Acquire))
}

static EDGES_LOCK_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);
static EDGES_LOCK_WAIT_MAX_NANOS: AtomicU64 = AtomicU64::new(0);
static EDGES_LOCK_CALLS: AtomicU64 = AtomicU64::new(0);
static EDGES_MATERIALIZE_BUSY_SKIPS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn record_edges_lock_wait(wait_nanos: u64) {
    if !crate::mcts::profiling::hot_path_metrics_enabled() {
        return;
    }
    EDGES_LOCK_CALLS.fetch_add(1, Ordering::Relaxed);
    EDGES_LOCK_WAIT_NANOS.fetch_add(wait_nanos, Ordering::Relaxed);
    let mut cur = EDGES_LOCK_WAIT_MAX_NANOS.load(Ordering::Relaxed);
    while wait_nanos > cur {
        match EDGES_LOCK_WAIT_MAX_NANOS.compare_exchange_weak(
            cur,
            wait_nanos,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => cur = next,
        }
    }
}

pub(crate) fn record_edges_materialize_busy_skip() {
    if !crate::mcts::profiling::hot_path_metrics_enabled() {
        return;
    }
    EDGES_MATERIALIZE_BUSY_SKIPS.fetch_add(1, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeLockContentionSnapshot {
    pub calls: u64,
    pub wait_nanos: u64,
    pub max_wait_nanos: u64,
    pub busy_skips: u64,
}

pub fn edge_lock_contention_snapshot() -> EdgeLockContentionSnapshot {
    EdgeLockContentionSnapshot {
        calls: EDGES_LOCK_CALLS.load(Ordering::Relaxed),
        wait_nanos: EDGES_LOCK_WAIT_NANOS.load(Ordering::Relaxed),
        max_wait_nanos: EDGES_LOCK_WAIT_MAX_NANOS.load(Ordering::Relaxed),
        busy_skips: EDGES_MATERIALIZE_BUSY_SKIPS.load(Ordering::Relaxed),
    }
}

pub fn reset_edge_lock_contention_counters() {
    EDGES_LOCK_CALLS.store(0, Ordering::Relaxed);
    EDGES_LOCK_WAIT_NANOS.store(0, Ordering::Relaxed);
    EDGES_LOCK_WAIT_MAX_NANOS.store(0, Ordering::Relaxed);
    EDGES_MATERIALIZE_BUSY_SKIPS.store(0, Ordering::Relaxed);
}

// ─────────────────────────────────────────────
// § MctsEdge
// ─────────────────────────────────────────────

pub struct MctsEdge<M> {
    pub mv: M,
    pub child: ArenaRef<MctsNode<M>>,
    pub p: f32,
    pub n: AtomicU32,
    pub w: AtomicU64, // f64 bits
    /// Welford M2 accumulator (f64 bits) — σᵢ² = M2/(N-1)
    pub m2: AtomicU64,
    /// Virtual visit count (integer, for PUCT N inflation)
    pub virtual_losses: AtomicI32,
    /// Virtual value penalty (f64 bits, for Q pessimism)
    pub virtual_value: AtomicU64,
}

impl<M: Copy + Send + Sync + 'static> MctsEdge<M> {
    pub fn new(mv: M, child: ArenaRef<MctsNode<M>>, prior: f32) -> Self {
        MctsEdge {
            mv,
            child,
            p: prior.max(0.0),
            n: AtomicU32::new(0),
            w: AtomicU64::new(0),
            m2: AtomicU64::new(0),
            virtual_losses: AtomicI32::new(0),
            virtual_value: AtomicU64::new(0),
        }
    }

    /// 논문 §6.1.1: σᵢ = √(M2/(N-1)), N≥2 시 유효
    /// N<2이면 fallback (σ_Q proxy 불필요 — 호출자가 처리)
    pub fn edge_sigma(&self) -> Option<f32> {
        let n = self.n.load(Ordering::Acquire);
        if n < 2 {
            return None;
        }
        let m2 = f64::from_bits(self.m2.load(Ordering::Acquire));
        let var = (m2 / (n - 1) as f64).max(0.0);
        Some(var.sqrt() as f32)
    }

    /// Q with split virtual loss: Q_eff = (W - Σvvalue) / (N + Σvvisit)
    #[inline]
    pub fn q_eff(&self) -> f32 {
        let n = self.n.load(Ordering::Acquire);
        let vl = self.virtual_losses.load(Ordering::Acquire).max(0) as u32;
        let n_eff = (n + vl).max(1) as f32;
        let w = atomic_f64_load(&self.w) as f32;
        let vv = atomic_f64_load(&self.virtual_value) as f32;
        (w - vv) / n_eff
    }

    /// Q — 실제 누적 (출력용, no VL)
    #[inline]
    pub fn q(&self) -> f32 {
        let n = self.n.load(Ordering::Acquire);
        if n == 0 {
            return 0.0;
        }
        atomic_f64_load(&self.w) as f32 / n as f32
    }

    /// Apply split virtual loss (called during select)
    #[inline]
    pub fn apply_vl(&self, vvisit: f32, vvalue: f32) {
        // Publish value pessimism before the virtual visit. Selection treats
        // the virtual visit as the reservation marker; this order prevents a
        // reader from seeing a published reservation without its paired
        // value penalty.
        if vvalue.abs() > 1e-9 {
            atomic_f64_add(&self.virtual_value, vvalue as f64);
        }
        // vvisit: round to nearest integer for atomic increment
        let vi = vvisit.round() as i32;
        if vi > 0 {
            self.virtual_losses.fetch_add(vi, Ordering::AcqRel);
        }
    }

    /// Remove split virtual loss (called during backup)
    #[inline]
    pub fn remove_vl(&self, vvisit: f32, vvalue: f32) {
        // Remove the paired value pessimism before clearing the reservation
        // marker. For modes with vvisit > 0, this maintains the invariant
        // that `virtual_losses == 0` implies no live paired virtual value.
        if vvalue.abs() > 1e-9 {
            atomic_f64_add(&self.virtual_value, -(vvalue as f64));
        }
        let vi = vvisit.round() as i32;
        if vi > 0 {
            self.virtual_losses.fetch_sub(vi, Ordering::AcqRel);
        }
    }

    pub fn add_w(&self, delta: f32) {
        atomic_f64_add(&self.w, delta as f64);
    }
}

#[derive(Clone)]
pub struct MctsEdgeSnapshot<M> {
    pub mv: M,
    pub child: ArenaRef<MctsNode<M>>,
    pub p: f32,
    pub n: u32,
    pub w: f64,
    pub m2: f64,
    pub virtual_losses: i32,
    pub virtual_value: f64,
}

impl<M: Copy + Send + Sync + 'static> MctsEdgeSnapshot<M> {
    #[inline]
    pub fn q(&self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        (self.w as f32) / self.n as f32
    }

    #[inline]
    pub fn q_eff(&self) -> f32 {
        let vl = self.virtual_losses.max(0) as u32;
        let n_eff = (self.n + vl).max(1) as f32;
        ((self.w - self.virtual_value) as f32) / n_eff
    }

    pub fn edge_sigma(&self) -> Option<f32> {
        if self.n < 2 {
            return None;
        }
        let var = (self.m2 / (self.n - 1) as f64).max(0.0);
        Some(var.sqrt() as f32)
    }
}

// ─────────────────────────────────────────────
// § PathEdge — backprop용 경로 기록 (with applied VL)
// ─────────────────────────────────────────────

pub struct PathEdge<M> {
    pub parent: ArenaRef<MctsNode<M>>,
    pub edge_idx: usize,
    /// VL that was applied during select — must be removed during backup
    pub applied_vl: (f32, f32), // (vvisit, vvalue)
}

// ─────────────────────────────────────────────
// § MctsNode
// ─────────────────────────────────────────────

pub struct MctsNode<M> {
    pub hash: u64,
    pub terminal_value: Option<f32>,

    /// CAS exactly-once: (move, prior) 쌍, prior 내림차순 정렬
    pub candidates: OnceLock<Box<[(M, f32)]>>,

    /// Phase 7 C edge buffer (2026-04-26): raw pointer to a slab of
    /// `MctsEdge<M>` allocated in the TT bucket's bumpalo Bump. Null
    /// until the first `materialize_edges` call publishes the slab.
    /// Slab capacity = `candidates.get().unwrap().len()` (fixed at
    /// expand time). Lifetime: tied to the bucket's Bump, which the TT
    /// owns; ArenaRef discipline guarantees the Bump outlives any
    /// `&MctsNode<M>` reachable from outside.
    pub edges_ptr: AtomicPtr<MctsEdge<M>>,

    /// Number of materialized edges (Phase 7 C). Acts as the Release-
    /// store synchronizer for the slab writes. Readers do
    /// `cursor.load(Acquire)` then `edges_ptr.load(Acquire)` and treat
    /// `[..cursor]` as fully-published.
    /// (Field name kept from the pre-Phase-7 `edge_cursor` for API
    /// compatibility with `materialized_count` / external callers.)
    pub edge_cursor: AtomicU32,

    /// Best-effort parallel materialization owner flag.
    ///
    /// This is intentionally separate from `materialize_lock`: it is acquired
    /// before expensive child-hash/TT preparation only on the non-blocking
    /// parallel widening path after at least one edge is already published.
    /// Blocking materialization ignores this flag and is still serialized by
    /// `materialize_lock`, preserving first-edge and serial semantics.
    pub materialize_claim: AtomicU32,

    /// Serializes concurrent `materialize_edges` calls on this node.
    /// Held only during slab fills; readers never block on it.
    pub materialize_lock: PlMutex<()>,

    pub n_total: AtomicU32,
    pub w_total: AtomicU64,

    // ── §6.3 MERGE channel: RTT holonomy residual curvature ──────
    // RTT(s) = Var[Q_γ(s)], γᵢ = 서로 다른 경로
    // TT hit 때마다 backed-up Q를 Welford online variance로 누적
    /// TT 히트 횟수 (서로 다른 경로 수)
    pub rtt_n: AtomicU32,
    /// Welford W: 누적 Q 합 (f64 bits)
    pub rtt_w: AtomicU64,
    /// Welford M2: 누적 분산 (f64 bits)
    pub rtt_m2: AtomicU64,
}

impl<M: Copy + Send + Sync + 'static> MctsNode<M> {
    /// Construct an `MctsNode` body in place. The TT layer allocates the
    /// resulting value into a `bumpalo::Bump` and returns an `ArenaRef`.
    /// Callers outside the engine (tests, isolated benchmarks) can wrap
    /// in `Box::leak` if they need an `ArenaRef` that outlives the call,
    /// or use `crate::mcts::tt::leak_node` for a shared helper.
    pub fn new(hash: u64, terminal_value: Option<f32>) -> Self {
        MctsNode {
            hash,
            terminal_value,
            candidates: OnceLock::new(),
            edges_ptr: AtomicPtr::new(std::ptr::null_mut()),
            edge_cursor: AtomicU32::new(0),
            materialize_claim: AtomicU32::new(0),
            materialize_lock: PlMutex::new(()),
            n_total: AtomicU32::new(0),
            w_total: AtomicU64::new(0),
            rtt_n: AtomicU32::new(0),
            rtt_w: AtomicU64::new(0),
            rtt_m2: AtomicU64::new(0),
        }
    }

    pub fn is_expanded(&self) -> bool {
        self.candidates.get().is_some()
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.get().map(|c| c.len()).unwrap_or(0)
    }

    pub fn materialized_count(&self) -> usize {
        self.edge_cursor.load(Ordering::Acquire) as usize
    }

    #[inline]
    pub fn mean_q(&self) -> f32 {
        let n = self.n_total.load(Ordering::Acquire);
        if n == 0 {
            return 0.0;
        }
        atomic_f64_load(&self.w_total) as f32 / n as f32
    }

    /// §5.3 GVOC: impact weight wimp(s) = N(s)/N(root)
    /// root가 알려진 경우 루트 N(root) 전달
    pub fn wimp(&self, root_n: u32) -> f32 {
        if root_n == 0 {
            return 1.0;
        }
        let my_n = self.n_total.load(Ordering::Acquire);
        (my_n as f32 / root_n as f32).min(1.0)
    }

    /// backup.rs에서 TT hit path에 대해 호출
    pub fn record_rtt_hit(&self, q_value: f32) {
        let q = q_value as f64;
        let n_old = self.rtt_n.fetch_add(1, Ordering::AcqRel) as f64;
        let n_new = n_old + 1.0;
        // Welford update
        let w_old = f64::from_bits(self.rtt_w.load(Ordering::Acquire));
        let w_new = w_old + q;
        atomic_f64_add(&self.rtt_w, q);
        // M2 += (q - mean_old)(q - mean_new)
        if n_old >= 1.0 {
            let mean_old = w_old / n_old;
            let mean_new = w_new / n_new;
            let delta_m2 = (q - mean_old) * (q - mean_new);
            atomic_f64_add(&self.rtt_m2, delta_m2);
        }
    }

    /// RTT variance = M2 / (n-1)  →  σ_RTT = √Var
    /// None if fewer than 2 TT hits
    pub fn rtt_variance(&self) -> Option<f32> {
        let n = self.rtt_n.load(Ordering::Acquire) as f64;
        if n < 2.0 {
            return None;
        }
        let m2 = f64::from_bits(self.rtt_m2.load(Ordering::Acquire));
        Some(((m2 / (n - 1.0)).max(0.0)) as f32)
    }

    /// Lock-free read of the materialized edges (Phase 7 C, 2026-04-26).
    ///
    /// Returns a slice covering `[0..edge_cursor]`. The slice's lifetime
    /// is tied to `&self`; the slab itself lives in the TT bucket's
    /// bumpalo Bump and outlives `&self` by the engine's ArenaPool
    /// discipline. Concurrent writers extend `edge_cursor` monotonically
    /// (under `materialize_lock`); a reader observing cursor = N
    /// happens-after every slot write at indices `[0..N)` via the
    /// Acquire/Release pair on `edge_cursor`.
    ///
    /// # Safety reasoning
    ///   - If `edge_cursor.load(Acquire)` returns 0, no slab access is
    ///     needed — return `&[]`.
    ///   - Otherwise the writer must have stored `edges_ptr` (Release)
    ///     BEFORE its first Release-store of `edge_cursor` (see
    ///     `TranspositionTable::allocate_edge_slab`). Our Acquire load
    ///     of `edge_cursor` synchronizes-with that prior Release-chain,
    ///     so the subsequent Acquire load of `edges_ptr` is guaranteed
    ///     to observe the published, non-null pointer.
    ///   - `from_raw_parts(ptr, len)` is sound: `ptr` is non-null and
    ///     properly aligned (allocated via `Layout::array::<MctsEdge<M>>`),
    ///     all `len` slots have been initialized, and the slab does not
    ///     move or shrink for the lifetime of `&self`.
    pub fn read_edges(&self) -> &[MctsEdge<M>] {
        let len = self.edge_cursor.load(Ordering::Acquire) as usize;
        if len == 0 {
            return &[];
        }
        let ptr = self.edges_ptr.load(Ordering::Acquire);
        debug_assert!(
            !ptr.is_null(),
            "edge_cursor > 0 implies edges_ptr published"
        );
        // SAFETY: see method-level reasoning above.
        unsafe { std::slice::from_raw_parts(ptr as *const MctsEdge<M>, len) }
    }

    /// Phase 7 C (2026-04-26): per-edge snapshot. Now backed by
    /// `read_edges()` (lock-free); the legacy lock-wait timer is dropped
    /// since there is no lock to wait on.
    pub fn edge_snapshot(&self, n_snap: usize) -> Vec<MctsEdgeSnapshot<M>> {
        let edges = self.read_edges();
        let n = n_snap.min(edges.len());
        edges[..n]
            .iter()
            .map(|edge| MctsEdgeSnapshot {
                mv: edge.mv,
                child: edge.child,
                p: edge.p,
                n: edge.n.load(Ordering::Acquire),
                w: atomic_f64_load(&edge.w),
                m2: atomic_f64_load(&edge.m2),
                virtual_losses: edge.virtual_losses.load(Ordering::Acquire),
                virtual_value: atomic_f64_load(&edge.virtual_value),
            })
            .collect()
    }

    /// Snapshot only the priors for the first `n_snap` edges. Phase 7 C:
    /// lock-free read via slab.
    pub fn edge_priors_snapshot(&self, n_snap: usize) -> Vec<f32> {
        let edges = self.read_edges();
        let n = n_snap.min(edges.len());
        edges[..n].iter().map(|edge| edge.p).collect()
    }

    /// Read-only access to the first `n_snap` edges. Phase 7 C: lock-free.
    pub fn with_edge_slice<R>(&self, n_snap: usize, f: impl FnOnce(&[MctsEdge<M>]) -> R) -> R {
        let edges = self.read_edges();
        let n = n_snap.min(edges.len());
        f(&edges[..n])
    }
}

// Phase 7 C-prep (2026-04-26): bounds-free helper + explicit `Drop` for
// `MctsNode`. The struct definition itself has no trait bounds on `M`
// (see `pub struct MctsNode<M>` above), so Rust's dropck rule requires
// the `Drop` impl to be equally unbounded. We split the helper into its
// own bounds-free inherent impl rather than adding it to the bounded
// inherent impl block above.
//
// With the current `RwLock<Vec<MctsEdge<M>>>` storage this `Drop` is
// structurally redundant — `Vec`'s own drop already calls `Drop` on
// every contained `MctsEdge<M>`. The impl is landed now so the next
// commit (Phase 7 C) only has to swap the body of
// `drop_edges_in_place` once `edges` becomes a raw-pointer slab
// allocated inside the bucket's `bumpalo::Bump` (bumpalo does NOT run
// `Drop` on its allocations).
//
// Drop order
//   `Drop::drop` runs BEFORE field auto-drop in declaration order. Our
//   Drop drains the edge buffer, then field auto-drop runs on (in
//   declaration order): `hash`, `terminal_value`, `candidates`,
//   `edges` (now empty), `edge_cursor`, `n_total`, `w_total`, `rtt_*`.
//   No double-drop: the Vec is already empty when its auto-drop runs.
//
// Interaction with `TtBucket::Drop`
//   `TtBucket::Drop` calls `std::ptr::drop_in_place` on each node body
//   stored in the bucket's Bump arena. That `drop_in_place` invokes
//   this `Drop` impl, then runs the field auto-drops, then returns.
//   The bucket's `Bump` then frees its raw chunks. The
//   ArenaPool/MctsEngine field-declaration discipline guarantees the
//   nodes are unreachable from anywhere else by the time we get here.
impl<M> MctsNode<M> {
    /// Drop every materialized edge in place (Phase 7 C, 2026-04-26).
    ///
    /// The slab lives in the TT bucket's bumpalo Bump, which does not
    /// run `Drop` on its allocations. Without this walk, each
    /// `MctsEdge<M>` would leak any heap-owned sub-fields it carries
    /// (currently atomics only — no boxed sub-fields — but future edge
    /// fields would silently leak).
    ///
    /// The slab MEMORY itself is not freed here: the bucket's Bump
    /// reclaims its raw chunks when `TtBucket::Drop` runs (after every
    /// node body in the bucket has been `drop_in_place`'d).
    ///
    /// # Safety
    ///   - We hold `&mut self`, so no other thread is racing on this
    ///     node. The engine's drop-order discipline (TT drops before
    ///     ArenaPool) guarantees any concurrent search threads have
    ///     joined by the time this runs.
    ///   - Each slot in `[0..len)` was initialized exactly once by a
    ///     prior `materialize_edges` call (writes serialized by
    ///     `materialize_lock`, published via the Release-store of
    ///     `edge_cursor`). `drop_in_place` runs `MctsEdge<M>::drop`
    ///     once per slot.
    ///   - After this, we zero `edge_cursor` so any (theoretically
    ///     impossible) post-drop reader observes empty.
    fn drop_edges_in_place(&mut self) {
        let len = *self.edge_cursor.get_mut() as usize;
        let ptr = *self.edges_ptr.get_mut();
        if len == 0 || ptr.is_null() {
            return;
        }
        // SAFETY: see method-level reasoning above.
        for i in 0..len {
            unsafe {
                std::ptr::drop_in_place(ptr.add(i));
            }
        }
        *self.edge_cursor.get_mut() = 0;
    }
}

impl<M> Drop for MctsNode<M> {
    fn drop(&mut self) {
        self.drop_edges_in_place();
    }
}

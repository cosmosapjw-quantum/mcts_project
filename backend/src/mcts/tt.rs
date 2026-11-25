//! TranspositionTable — lock-striped open-addressing
//!
//! Phase 7 F (2026-04-26): per-bucket storage migrated from
//! `HashMap<u64, ArenaRef<MctsNode<M>>>` to a fixed-size open-addressing
//! slot array (`Box<[TtSlot<M>]>`). The current production geometry keeps
//! the total slot budget at 256 K entries split across 256 lock stripes,
//! so each bucket owns 1024 slots (16 KiB / bucket).
//! Lookups linearly probe an 8-slot window starting from
//! `(hash >> BUCKET_INDEX_BITS) & SLOT_MASK`. Misses on a full window
//! evict the lowest-`n_total` slot
//! in the window (matches the pre-Phase-7 HashMap eviction policy
//! semantically — no body reclaim, the Bump retains the bytes until
//! `TtBucket::Drop`).
//!
//! Why: callgrind on the Phase 7 C bench shows
//! `TranspositionTable::get_or_create` at ~25 % Ir share. The HashMap
//! path traverses dozens of cycles per lookup (hash compute, bucket
//! redirection, entry walk). The open-addressing window touches at
//! most 8 16-byte slots = 2 cache lines, with a single comparison per
//! slot. Phase 7 plan target: get_or_create < 8 % Ir.
//!
//! 버킷 배분: low `BUCKET_INDEX_BITS` of the hash. Slot start within
//! bucket: the next bits above the bucket index. Different bits ensure
//! low-bucket-collision hashes still spread within a bucket.

use crate::mcts::node::{ArenaRef, MctsEdge, MctsNode};
use bumpalo::Bump;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

// ─────────────────────────────────────────────
// § 설정
// ─────────────────────────────────────────────

/// Total open-addressing slots kept constant across geometry sweeps.
/// 256 K slots x 16 B/slot = 4 MiB of slot metadata, before per-bucket
/// bump chunks.
const TOTAL_TT_SLOTS: usize = 256 * 1024;

/// 버킷 수 — 2의 거듭제곱이면 % 연산이 & 연산으로 최적화됨.
/// Phase 7 G (2026-05-03): a 512×512 geometry was benchmarked because
/// hotpath telemetry showed TT write-lock wait dominating read-lock wait
/// on parallel Gomoku15. It reduced TT wait, but did not produce a clear
/// non-instrumented throughput win. Production therefore keeps the
/// previous 256×1024 geometry and preserves the read/write split
/// telemetry for future larger-budget comparisons.
pub const NUM_BUCKETS: usize = 256;

/// Number of low hash bits consumed by the bucket index. Keep slot
/// indexing on the next bits so bucket-local probe starts do not reuse
/// the same entropy as the lock stripe.
const BUCKET_INDEX_BITS: u32 = NUM_BUCKETS.trailing_zeros();

/// 버킷당 최대 엔트리 수. 초과 시 가장 적게 방문된 노드를 제거.
/// Phase 7 G (2026-05-03): cap remains 1024 with 256 buckets after an
/// inconclusive 512×512 geometry comparison. This preserves the Phase 7
/// F total slot budget of 256 K entries and the measured per-bucket
/// locality profile. Cap is a power of two so `& SLOT_MASK` stays
/// branch-free.
pub(crate) const MAX_ENTRIES_PER_BUCKET: usize = TOTAL_TT_SLOTS / NUM_BUCKETS;

const _: () = assert!(NUM_BUCKETS.is_power_of_two());
const _: () = assert!(MAX_ENTRIES_PER_BUCKET.is_power_of_two());

/// Phase 7 F (2026-04-26): per-slot record for the open-addressing TT.
///
/// Vacancy invariant (Phase 7 F semantics, sealed in next commit):
///   - `hash == 0 && node.is_none()` ↔ vacant slot.
///   - Insertion always populates `node = Some(...)` and any non-zero
///     hash. The TT bucket index uses the full 64-bit `hash`, so a
///     genuine `hash = 0` value would be a Zobrist collision with the
///     sentinel — the engine's Zobrist setup guarantees the empty
///     position hash is non-zero on every game type currently shipped
///     (verified in `zobrist_tt_parallel_verify::v1_zobrist_collision_rate`).
///
/// Layout (M = Copy + Send + Sync + 'static):
///   - `hash: u64` — 8 bytes
///   - `node: Option<ArenaRef<MctsNode<M>>>` — 8 bytes (niche-opt over
///     `NonNull<MctsNode<M>>`, so `None` == null pointer)
/// Total: 16 bytes. With default alignment (8), four slots fit per
/// 64-byte L1 line; the 8-slot probe window covers two adjacent cache
/// lines.
#[derive(Clone, Copy)]
pub(crate) struct TtSlot<M: Copy + Send + Sync + 'static> {
    pub hash: u64,
    pub node: Option<ArenaRef<MctsNode<M>>>,
}

impl<M: Copy + Send + Sync + 'static> TtSlot<M> {
    /// Sentinel for an empty slot. Used to initialize the slot array
    /// when the F commit replaces `HashMap` with `Box<[TtSlot<M>]>` of
    /// length `MAX_ENTRIES_PER_BUCKET`.
    pub const VACANT: Self = Self {
        hash: 0,
        node: None,
    };

    /// True iff this slot has no published node. Equivalent to
    /// `self.node.is_none()`; the helper exists for symmetry with the
    /// open-addressing probe-window scan in F.
    #[inline]
    pub fn is_vacant(&self) -> bool {
        self.node.is_none()
    }
}

// ─────────────────────────────────────────────
// § TT Bucket (Phase 7 F open-addressing)
// ─────────────────────────────────────────────

/// Bitmask for slot index reduction. `MAX_ENTRIES_PER_BUCKET` is a
/// power of two so `& SLOT_MASK` replaces `%` (cheaper, branch-free).
const SLOT_MASK: usize = MAX_ENTRIES_PER_BUCKET - 1;

/// Linear-probe window. A probe walks at most `PROBE_WINDOW` slots
/// before declaring "no match" (read path) or evicting the
/// lowest-`n_total` slot (write path). At 16 B / slot the window
/// covers two 64-byte cache lines, which is a tight working set even
/// under heavy concurrent probing.
const PROBE_WINDOW: usize = 8;

struct TtBucket<M: Copy + Send + Sync + 'static> {
    /// Phase 7 F: fixed-size open-addressing slot array.
    /// `len() == MAX_ENTRIES_PER_BUCKET`. Each slot is `TtSlot::VACANT`
    /// at construction; insertion sets `(hash, Some(node))` and
    /// eviction overwrites in place (the displaced node body is left
    /// in the Bump until `TtBucket::Drop`).
    slots: Box<[TtSlot<M>]>,
    /// Bumpalo arena: all `MctsNode<M>` bodies AND per-node edge slabs
    /// (Phase 7 C, 2026-04-26) for this bucket are allocated here.
    /// Bumpalo guarantees pointer stability for the life of the Bump,
    /// so `ArenaRef`s and `edges_ptr`s into it remain valid until the
    /// containing TT is dropped.
    arena: Bump,
    // MQFT: own every node, including no-TT and evicted bodies, until search drop.
    owned: Vec<ArenaRef<MctsNode<M>>>,
}

#[repr(align(64))]
struct TtBucketStats {
    hits: AtomicU64,
    misses: AtomicU64,
    get_or_create_calls: AtomicU64,
    get_calls: AtomicU64,
}

impl TtBucketStats {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            get_or_create_calls: AtomicU64::new(0),
            get_calls: AtomicU64::new(0),
        }
    }
}

// SAFETY (Phase 6.2 → Phase 7 F): `TtBucket` contains a `Bump` (!Sync)
// AND a `Box<[TtSlot<M>]>` of plain (Copy, no interior mutability) records.
// Wrapping in `parking_lot::RwLock` requires `Sync`; soundness rests on
// the per-bucket access discipline:
//
//   - Read guards (`buckets[i].read()`) may concurrently load fields
//     from `bucket.slots[..]` (plain reads of Copy data — no race) and
//     MUST NOT touch `bucket.arena` (`&Bump` aliasing across readers
//     would be UB).
//   - Write guards (`buckets[i].write()`) hold exclusive access by
//     RwLock contract and may mutate either field freely.
//
// Phase 7 F adds one new caller: `allocate_edge_slab` (write guard,
// touches arena). The original `get_or_create` paths remain (read for
// probe, write on miss).
unsafe impl<M: Copy + Send + Sync + 'static> Sync for TtBucket<M> {}

impl<M: Copy + Send + Sync + 'static> TtBucket<M> {
    fn new() -> Self {
        // Pre-allocate the slot array filled with `VACANT`. Allocation
        // pattern: vec! macro → Box<[TtSlot<M>]> via into_boxed_slice.
        // The vec is stack-built then heap-shrunk; no separate raw
        // alloc helper needed.
        let slots = vec![TtSlot::<M>::VACANT; MAX_ENTRIES_PER_BUCKET].into_boxed_slice();
        TtBucket {
            slots,
            // Initial bumpalo chunk sized to amortize first-touch zero-fault
            // cost across many node bodies + edge slabs. Profiling
            // (artifacts/profiling_20260428) attributed ~20% of cumulative
            // CPU to the kernel page-fault path through bumpalo's grow
            // chunks — pre-sizing the first chunk to 64 KiB collapses the
            // initial doubling sequence (1 KiB → 2 → 4 → 8 → 16 → 32 → 64)
            // into a single allocation, cutting first-touch faults ~6×.
            arena: Bump::with_capacity(64 * 1024),
            owned: Vec::new(),
        }
    }
}

impl<M: Copy + Send + Sync + 'static> Drop for TtBucket<M> {
    fn drop(&mut self) {
        // Every allocated body is listed exactly once, not merely the current slots.
        // No readers remain when the last TT owner drops. Child references have no
        // destructor dereference; each node drops its own candidate box and edge slab.
        for node_ref in self.owned.drain(..) {
            let raw = ArenaRef::as_ptr(&node_ref) as *mut MctsNode<M>;
            unsafe { std::ptr::drop_in_place(raw); }
        }
    }
}

// ─────────────────────────────────────────────
// § Per-bucket bumpalo storage
// ─────────────────────────────────────────────
//
// Each `TtBucket` owns its own `bumpalo::Bump`. All `MctsNode<M>` bodies
// inserted via `get_or_create` are allocated in the bucket's Bump and
// returned as `ArenaRef<MctsNode<M>>`. Allocations are serialized
// through the existing per-bucket `Mutex<TtBucket>` (Bump is `!Sync` —
// the lock is the synchronization mechanism).
//
// Lifetime safety
//   The TT owns the Bumps. Edges/snapshots/path entries hold
//   `ArenaRef`s into those Bumps. The engine holds `Arc<TT>` and ensures
//   the TT outlives every `ArenaRef` reachable through the engine's
//   internal state. When the last `Arc<TT>` drops, both the map (which
//   contains `ArenaRef`s with no Drop impl) and the Bumps drop. After
//   that point no `ArenaRef` is reachable.
//
// Eviction note: removing an entry from the map does NOT free its
// Bump-allocated body (bumpalo never reclaims individual allocations).
// This is a deliberate trade — eviction is rare under scenario A and
// the per-bucket cap (`MAX_ENTRIES_PER_BUCKET`) bounds total bytes.

/// Test/standalone helper: allocate an `MctsNode` body on the heap and
/// leak it, returning an `ArenaRef`. Used by detached test setups in
/// `mcts/quartz.rs` and similar places that previously called
/// `Arc::new(MctsNode::new(...))` outside of an engine. Production hot
/// paths (engine select/expand/backup) MUST go through
/// `TranspositionTable::get_or_create`.
pub fn leak_node<M: Copy + Send + Sync + 'static>(
    hash: u64,
    terminal_value: Option<f32>,
) -> ArenaRef<MctsNode<M>> {
    let boxed = Box::new(MctsNode::new(hash, terminal_value));
    let raw = Box::into_raw(boxed);
    // SAFETY: `raw` is a valid heap pointer (non-null, aligned). It is
    // intentionally leaked; this helper is for test scaffolding only and
    // its leak is acceptable in those contexts.
    unsafe { ArenaRef::from_raw(NonNull::new_unchecked(raw)) }
}

// ─────────────────────────────────────────────
// § TranspositionTable
// ─────────────────────────────────────────────

pub struct TranspositionTable<M: Copy + Send + Sync + 'static> {
    enabled: bool,
    /// Per-bucket `RwLock<TtBucket>` (Phase 6.2, 2026-04-25). The hit path
    /// in `get_or_create` and `get` takes a read lock so multiple threads
    /// can probe the same bucket in parallel; the miss path of
    /// `get_or_create` upgrades to a write lock for the alloc + insert.
    /// See the safety note on `TtBucket`'s `unsafe Sync` impl above.
    buckets: Vec<RwLock<TtBucket<M>>>,
    /// Per-bucket counters avoid a single global atomic cache-line hotspot
    /// during parallel search. `hit_rate()` and `contention_snapshot()` fold
    /// these counters after the search, where O(NUM_BUCKETS) reads are cheap.
    stats: Box<[TtBucketStats]>,
    pub lock_wait_nanos: AtomicU64,
    pub max_lock_wait_nanos: AtomicU64,
    pub read_lock_wait_nanos: AtomicU64,
    pub max_read_lock_wait_nanos: AtomicU64,
    pub write_lock_wait_nanos: AtomicU64,
    pub max_write_lock_wait_nanos: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct TtContentionSnapshot {
    pub get_or_create_calls: u64,
    pub get_calls: u64,
    pub lock_wait_nanos: u64,
    pub max_lock_wait_nanos: u64,
    pub read_lock_wait_nanos: u64,
    pub max_read_lock_wait_nanos: u64,
    pub write_lock_wait_nanos: u64,
    pub max_write_lock_wait_nanos: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TtLookup<M: Copy + Send + Sync + 'static> {
    pub hash: u64,
    pub terminal_value: Option<f32>,
    pub node: Option<ArenaRef<MctsNode<M>>>,
}

impl<M: Copy + Send + Sync + 'static> TtLookup<M> {
    #[inline]
    pub fn new(hash: u64, terminal_value: Option<f32>) -> Self {
        Self {
            hash,
            terminal_value,
            node: None,
        }
    }
}

impl<M: Copy + Send + Sync + 'static> TranspositionTable<M> {
    pub fn new() -> Self {
        Self::new_enabled(true)
    }

    pub fn new_enabled(enabled: bool) -> Self {
        let buckets = (0..NUM_BUCKETS)
            .map(|_| RwLock::new(TtBucket::new()))
            .collect();
        let stats = (0..NUM_BUCKETS)
            .map(|_| TtBucketStats::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        TranspositionTable {
            enabled,
            buckets,
            stats,
            lock_wait_nanos: AtomicU64::new(0),
            max_lock_wait_nanos: AtomicU64::new(0),
            read_lock_wait_nanos: AtomicU64::new(0),
            max_read_lock_wait_nanos: AtomicU64::new(0),
            write_lock_wait_nanos: AtomicU64::new(0),
            max_write_lock_wait_nanos: AtomicU64::new(0),
        }
    }

    #[inline]
    fn bucket_idx(hash: u64) -> usize {
        (hash as usize) & (NUM_BUCKETS - 1)
    }

    /// Phase 7 F/G: slot start within a bucket. Uses the hash bits
    /// immediately above the bucket index so lock stripe selection and
    /// bucket-local probe starts draw from disjoint regions of the
    /// Zobrist key.
    #[inline]
    fn slot_start(hash: u64) -> usize {
        ((hash >> BUCKET_INDEX_BITS) as usize) & SLOT_MASK
    }

    #[inline(always)]
    fn probe_locked(
        bucket: &TtBucket<M>,
        hash: u64,
        slot_start: usize,
    ) -> Option<ArenaRef<MctsNode<M>>> {
        for offset in 0..PROBE_WINDOW {
            let i = (slot_start + offset) & SLOT_MASK;
            // SAFETY: `i & SLOT_MASK` is in [0, MAX_ENTRIES_PER_BUCKET);
            // `bucket.slots.len() == MAX_ENTRIES_PER_BUCKET` by construction.
            let slot = unsafe { bucket.slots.get_unchecked(i) };
            if slot.hash == hash {
                if let Some(node) = slot.node {
                    return Some(node);
                }
            }
            if slot.is_vacant() {
                break;
            }
        }
        None
    }

    fn insert_missing_locked(
        bucket: &mut TtBucket<M>,
        hash: u64,
        terminal_value: Option<f32>,
        slot_start: usize,
    ) -> ArenaRef<MctsNode<M>> {
        // Find insertion slot: first vacant in window, else evict the
        // lowest-`n_total` slot. Eviction does NOT drop_in_place the
        // displaced body — outstanding `ArenaRef`s elsewhere in the
        // tree may still reference it. The body is reclaimed at
        // bucket-Drop time only if it remains in a slot; otherwise it
        // lives unreferenced in the Bump (acceptable per the Phase 3
        // arena leak contract). This matches the pre-Phase-7 HashMap
        // eviction policy (which `bucket.map.remove(victim_hash)`
        // dropped the ArenaRef without touching the body).
        let mut insert_idx = slot_start;
        let mut min_n: u32 = u32::MAX;
        let mut min_idx = slot_start;
        let mut found_vacant = false;
        for offset in 0..PROBE_WINDOW {
            let i = (slot_start + offset) & SLOT_MASK;
            let slot = unsafe { bucket.slots.get_unchecked(i) };
            if slot.is_vacant() {
                insert_idx = i;
                found_vacant = true;
                break;
            }
            // Occupied — track lowest n_total candidate for eviction.
            // SAFETY: `slot.is_vacant()` would have returned true for
            // `node = None`; we are in the else branch, so node is Some.
            let n = unsafe { slot.node.unwrap_unchecked() }
                .n_total
                .load(Ordering::Relaxed);
            if n < min_n {
                min_n = n;
                min_idx = i;
            }
        }
        if !found_vacant {
            insert_idx = min_idx;
        }

        // Allocate the body in the bucket's Bump. Bumpalo's `alloc`
        // returns a `&mut T` whose lifetime is bound by `&self`, but the
        // allocation itself is stable until the Bump drops. We capture
        // the address as `NonNull<T>` and wrap in `ArenaRef`, whose
        // safety invariant is documented at its definition.
        let body = MctsNode::new(hash, terminal_value);
        let allocated: &mut MctsNode<M> = bucket.arena.alloc(body);
        let node = unsafe { ArenaRef::from_raw(NonNull::new_unchecked(allocated as *mut _)) };

        bucket.owned.push(node);
        // Publish the new entry. SAFETY: insert_idx <= SLOT_MASK <
        // bucket.slots.len(); we hold the write guard so no other
        // thread observes this slot mid-update.
        unsafe {
            *bucket.slots.get_unchecked_mut(insert_idx) = TtSlot {
                hash,
                node: Some(node),
            };
        }
        node
    }

    /// 해시에 해당하는 노드를 조회.
    /// 없으면 `terminal_value`로 새 노드를 생성하고 삽입.
    /// 있으면 기존 노드 반환 (always-replace 충돌 정책 → 먼저 들어온 것 우선).
    ///
    /// Phase 7 F (2026-04-26): open-addressing replacement for the
    /// pre-Phase-7 `HashMap` lookup. The fast path reads at most
    /// `PROBE_WINDOW` (= 8) 16-byte slots within a single bucket,
    /// terminating on either a hash match (HIT, return the slot's
    /// `ArenaRef`) or a vacant slot (MISS — insertion path).
    ///
    /// Lock ladder unchanged from Phase 6.2: read lock for probe,
    /// upgrade to write lock on miss. A second probe under write lock
    /// double-checks for racing writers before alloc + insert.
    pub fn get_or_create(&self, hash: u64, terminal_value: Option<f32>) -> ArenaRef<MctsNode<M>> {
        if !self.enabled {
            let mut bucket = self.buckets[Self::bucket_idx(hash)].write();
            let allocated = bucket.arena.alloc(MctsNode::new(hash,terminal_value));
            let node = unsafe { ArenaRef::from_raw(NonNull::new_unchecked(allocated as *mut _)) };
            bucket.owned.push(node);
            return node;
        }
        let bucket_idx = Self::bucket_idx(hash);
        let stats = &self.stats[bucket_idx];
        stats.get_or_create_calls.fetch_add(1, Ordering::Relaxed);
        let slot_start = Self::slot_start(hash);

        // Fast path: read lock + linear probe. Multiple threads can be
        // in this branch simultaneously on the same bucket.
        {
            let t0 = crate::mcts::profiling::maybe_start_timer();
            let bucket = self.buckets[bucket_idx].read();
            if let Some(t0) = t0 {
                self.record_read_lock_wait(t0.elapsed().as_nanos() as u64);
            }
            if let Some(node) = Self::probe_locked(&bucket, hash, slot_start) {
                stats.hits.fetch_add(1, Ordering::Relaxed);
                return node;
            }
        }

        // Slow path: write lock + re-probe + alloc + insert.
        let t1 = crate::mcts::profiling::maybe_start_timer();
        let mut bucket = self.buckets[bucket_idx].write();
        if let Some(t1) = t1 {
            self.record_write_lock_wait(t1.elapsed().as_nanos() as u64);
        }

        // Re-probe under write lock to catch racing inserts.
        if let Some(node) = Self::probe_locked(&bucket, hash, slot_start) {
            stats.hits.fetch_add(1, Ordering::Relaxed);
            return node;
        }

        stats.misses.fetch_add(1, Ordering::Relaxed);
        Self::insert_missing_locked(&mut bucket, hash, terminal_value, slot_start)
    }

    /// Resolve several child hashes while coalescing work by TT bucket.
    ///
    /// Small batches intentionally fall back to `get_or_create`: progressive
    /// widening often materializes only one edge, where the grouping scan would
    /// cost more than it saves. Larger batches preserve per-lookup hit/miss
    /// accounting while reducing repeated read/write lock acquisitions on the
    /// same bucket.
    pub fn get_or_create_batch(&self, lookups: &mut [TtLookup<M>]) {
        if lookups.is_empty() {
            return;
        }
        if !self.enabled {
            for lookup in lookups.iter_mut() {
                if lookup.node.is_none() {
                    lookup.node = Some(self.get_or_create(lookup.hash, lookup.terminal_value));
                }
            }
            return;
        }
        if lookups.len() < 4 {
            for lookup in lookups.iter_mut() {
                if lookup.node.is_none() {
                    lookup.node = Some(self.get_or_create(lookup.hash, lookup.terminal_value));
                }
            }
            return;
        }

        for i in 0..lookups.len() {
            if lookups[i].node.is_some() {
                continue;
            }
            let bucket_idx = Self::bucket_idx(lookups[i].hash);
            let stats = &self.stats[bucket_idx];
            let mut misses: SmallVec<[usize; 16]> = SmallVec::new();

            {
                let t0 = crate::mcts::profiling::maybe_start_timer();
                let bucket = self.buckets[bucket_idx].read();
                if let Some(t0) = t0 {
                    self.record_read_lock_wait(t0.elapsed().as_nanos() as u64);
                }

                for j in i..lookups.len() {
                    if lookups[j].node.is_some() || Self::bucket_idx(lookups[j].hash) != bucket_idx
                    {
                        continue;
                    }

                    stats.get_or_create_calls.fetch_add(1, Ordering::Relaxed);
                    let hash = lookups[j].hash;
                    let slot_start = Self::slot_start(hash);
                    if let Some(node) = Self::probe_locked(&bucket, hash, slot_start) {
                        stats.hits.fetch_add(1, Ordering::Relaxed);
                        lookups[j].node = Some(node);
                    } else {
                        misses.push(j);
                    }
                }
            }

            if misses.is_empty() {
                continue;
            }

            let t1 = crate::mcts::profiling::maybe_start_timer();
            let mut bucket = self.buckets[bucket_idx].write();
            if let Some(t1) = t1 {
                self.record_write_lock_wait(t1.elapsed().as_nanos() as u64);
            }

            for j in misses {
                if lookups[j].node.is_some() {
                    continue;
                }
                let hash = lookups[j].hash;
                let slot_start = Self::slot_start(hash);
                let node = if let Some(node) = Self::probe_locked(&bucket, hash, slot_start) {
                    stats.hits.fetch_add(1, Ordering::Relaxed);
                    node
                } else {
                    stats.misses.fetch_add(1, Ordering::Relaxed);
                    Self::insert_missing_locked(
                        &mut bucket,
                        hash,
                        lookups[j].terminal_value,
                        slot_start,
                    )
                };
                lookups[j].node = Some(node);
            }
        }
    }

    /// Phase 7 C (2026-04-26): allocate (or look up) the per-node edge
    /// slab in the bucket's bumpalo Bump.
    ///
    /// Idempotent: if `node.edges_ptr` is already non-null, returns the
    /// existing (ptr, cap) under an Acquire load — no lock taken. On
    /// first call, takes the bucket write lock, allocates a slab of
    /// `n_candidates` `MctsEdge<M>` slots in `bucket.arena`, and
    /// publishes the pointer via Release-store.
    ///
    /// # Memory ordering
    ///   - Writer: bucket-write-locked alloc, then `edges_ptr.store(Release)`,
    ///     then `edges_cap` is implicit (capacity = `n_candidates`,
    ///     known to all callers via `node.candidates`).
    ///   - Reader (in `materialize_edges` fill loop): observes the
    ///     non-null ptr via Acquire load, then writes to slots under
    ///     `materialize_lock`.
    ///   - Lock-free PUCT reader (`MctsNode::read_edges`): sees ptr
    ///     publication transitively via the Release-store of
    ///     `edge_cursor` at the end of the materialize fill loop.
    ///
    /// # Safety
    ///   The returned pointer is valid for `n_candidates` aligned
    ///   `MctsEdge<M>` slots and remains valid until the bucket's Bump
    ///   is dropped (i.e., until the engine's TT drops). All writes to
    ///   the slab MUST be serialized via `node.materialize_lock`.
    pub fn allocate_edge_slab<MM: Copy + Send + Sync + 'static>(
        &self,
        node: &MctsNode<MM>,
        n_candidates: usize,
    ) -> (*mut MctsEdge<MM>, u32) {
        debug_assert!(n_candidates > 0);
        // Fast path: slab already published.
        let existing = node.edges_ptr.load(Ordering::Acquire);
        if !existing.is_null() {
            // The cap was set once and is not stored on the node — it
            // is implicit in `node.candidates.get().unwrap().len()`.
            // Callers that need it should consult that. We return the
            // published pointer plus the *requested* `n_candidates`
            // verbatim; callers always pass `candidates.len()`, so the
            // cap is identical across calls.
            return (existing as *mut MctsEdge<MM>, n_candidates as u32);
        }

        let idx = Self::bucket_idx(node.hash);
        // Slab alloc requires bucket write lock (per the unsafe Sync
        // discipline on TtBucket: only write guards may touch arena).
        let t0 = crate::mcts::profiling::maybe_start_timer();
        let bucket = self.buckets[idx].write();
        if let Some(t0) = t0 {
            self.record_write_lock_wait(t0.elapsed().as_nanos() as u64);
        }

        // Double-check after acquiring the write lock.
        let existing = node.edges_ptr.load(Ordering::Relaxed);
        if !existing.is_null() {
            return (existing as *mut MctsEdge<MM>, n_candidates as u32);
        }

        let layout =
            std::alloc::Layout::array::<MctsEdge<MM>>(n_candidates).expect("MctsEdge slab layout");
        // SAFETY: `Bump::alloc_layout` returns a `NonNull<u8>` with the
        // requested layout. The pointer is valid until the Bump drops
        // (the TT owns the bucket; the bucket owns the Bump).
        let raw = bucket.arena.alloc_layout(layout);
        let slab_ptr = raw.as_ptr() as *mut MctsEdge<MM>;

        // Release-store publishes the pointer to lock-free readers.
        node.edges_ptr.store(slab_ptr, Ordering::Release);

        (slab_ptr, n_candidates as u32)
    }

    /// Architectural prefetch hint for the cache line that the next
    /// `get_or_create(hash, _)` will read on its read-lock fast path.
    ///
    /// Intended call pattern (callers race the L2/L3 fetch with other
    /// useful work between the hint and the actual access):
    /// ```ignore
    /// let h = state.hash();
    /// tt.prefetch(h);          // <-- issue here
    /// // ~30+ cycles of cheap work …
    /// let n = tt.get_or_create(h, tv);
    /// ```
    ///
    /// No semantic effect on any architecture — `_mm_prefetch` is
    /// non-faulting and does not perturb program-visible state. On
    /// non-x86_64 targets this fn is a no-op.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        if !self.enabled {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            let bucket_idx = Self::bucket_idx(hash);
            let slot_start = Self::slot_start(hash);
            // SAFETY:
            //   - `RwLock::data_ptr()` is valid for the life of the TT —
            //     the underlying `TtBucket<M>` is heap-pinned by the
            //     parking_lot RwLock and never moves.
            //   - The non-atomic read of `bucket.slots.as_ptr()` is
            //     race-free in practice: `slots` is a `Box<[TtSlot<M>]>`
            //     set once in `TtBucket::new` and never reassigned, so
            //     the pointer field's bit pattern is invariant after
            //     construction (any thread reading observes the same
            //     stable pointer).
            //   - `_mm_prefetch` is a hint that issues no architectural
            //     load and cannot fault, even if the address is
            //     pathological. Worst case it is silently dropped by the
            //     CPU.
            unsafe {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                let bucket_ptr = self.buckets[bucket_idx].data_ptr();
                let slot_ptr = (*bucket_ptr).slots.as_ptr().add(slot_start);
                _mm_prefetch::<_MM_HINT_T0>(slot_ptr as *const i8);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = hash;
        }
    }

    /// 조회만 (삽입 없음)
    /// Phase 7 F (2026-04-26): open-addressing probe — same probe
    /// window as `get_or_create` fast path.
    pub fn get(&self, hash: u64) -> Option<ArenaRef<MctsNode<M>>> {
        if !self.enabled {
            return None;
        }
        let bucket_idx = Self::bucket_idx(hash);
        self.stats[bucket_idx]
            .get_calls
            .fetch_add(1, Ordering::Relaxed);
        let slot_start = Self::slot_start(hash);
        let t0 = crate::mcts::profiling::maybe_start_timer();
        let bucket = self.buckets[bucket_idx].read();
        if let Some(t0) = t0 {
            self.record_read_lock_wait(t0.elapsed().as_nanos() as u64);
        }
        for offset in 0..PROBE_WINDOW {
            let i = (slot_start + offset) & SLOT_MASK;
            let slot = unsafe { bucket.slots.get_unchecked(i) };
            if slot.hash == hash {
                return slot.node;
            }
            if slot.is_vacant() {
                return None;
            }
        }
        None
    }

    pub fn allocated_nodes(&self) -> usize { self.buckets.iter().map(|b|b.read().owned.len()).sum() }

    pub fn size(&self) -> usize {
        if !self.enabled {
            return 0;
        }
        // Phase 7 F/G: linear count of occupied slots per bucket. With
        // TOTAL_TT_SLOTS = 256 K, this is only used for diagnostics,
        // not the hot path.
        self.buckets
            .iter()
            .map(|b| {
                let bucket = b.read();
                bucket.slots.iter().filter(|s| !s.is_vacant()).count()
            })
            .sum()
    }

    pub fn hit_rate(&self) -> f64 {
        let (h, m) = self
            .stats
            .iter()
            .fold((0_u64, 0_u64), |(hits, misses), bucket| {
                (
                    hits + bucket.hits.load(Ordering::Relaxed),
                    misses + bucket.misses.load(Ordering::Relaxed),
                )
            });
        let h = h as f64;
        let m = m as f64;
        if h + m > 0.0 {
            h / (h + m)
        } else {
            0.0
        }
    }

    fn record_read_lock_wait(&self, wait_nanos: u64) {
        self.record_lock_wait(wait_nanos);
        if !crate::mcts::profiling::hot_path_metrics_enabled() {
            return;
        }
        self.read_lock_wait_nanos
            .fetch_add(wait_nanos, Ordering::Relaxed);
        update_max_atomic(&self.max_read_lock_wait_nanos, wait_nanos);
    }

    fn record_write_lock_wait(&self, wait_nanos: u64) {
        self.record_lock_wait(wait_nanos);
        if !crate::mcts::profiling::hot_path_metrics_enabled() {
            return;
        }
        self.write_lock_wait_nanos
            .fetch_add(wait_nanos, Ordering::Relaxed);
        update_max_atomic(&self.max_write_lock_wait_nanos, wait_nanos);
    }

    fn record_lock_wait(&self, wait_nanos: u64) {
        if !crate::mcts::profiling::hot_path_metrics_enabled() {
            return;
        }
        self.lock_wait_nanos
            .fetch_add(wait_nanos, Ordering::Relaxed);
        update_max_atomic(&self.max_lock_wait_nanos, wait_nanos);
    }

    pub fn contention_snapshot(&self) -> TtContentionSnapshot {
        let (get_or_create_calls, get_calls) =
            self.stats
                .iter()
                .fold((0_u64, 0_u64), |(goc, get), bucket| {
                    (
                        goc + bucket.get_or_create_calls.load(Ordering::Relaxed),
                        get + bucket.get_calls.load(Ordering::Relaxed),
                    )
                });
        TtContentionSnapshot {
            get_or_create_calls,
            get_calls,
            lock_wait_nanos: self.lock_wait_nanos.load(Ordering::Relaxed),
            max_lock_wait_nanos: self.max_lock_wait_nanos.load(Ordering::Relaxed),
            read_lock_wait_nanos: self.read_lock_wait_nanos.load(Ordering::Relaxed),
            max_read_lock_wait_nanos: self.max_read_lock_wait_nanos.load(Ordering::Relaxed),
            write_lock_wait_nanos: self.write_lock_wait_nanos.load(Ordering::Relaxed),
            max_write_lock_wait_nanos: self.max_write_lock_wait_nanos.load(Ordering::Relaxed),
        }
    }
}

fn update_max_atomic(max: &AtomicU64, value: u64) {
    let mut prev = max.load(Ordering::Relaxed);
    while value > prev {
        match max.compare_exchange(prev, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(cur) => prev = cur,
        }
    }
}

impl<M: Copy + Send + Sync + 'static> Default for TranspositionTable<M> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{TranspositionTable, TtLookup, BUCKET_INDEX_BITS};
    use crate::mcts::node::ArenaRef;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn disabled_table_does_not_merge_or_store_nodes() {
        let tt = TranspositionTable::<usize>::new_enabled(false);
        let a = tt.get_or_create(123, None);
        let b = tt.get_or_create(123, None);

        assert_eq!(tt.size(), 0);
        assert!(tt.get(123).is_none());
        assert!(!ArenaRef::ptr_eq(&a, &b));
    }

    #[test]
    fn identity_hasher_preserves_distinct_u64_keys() {
        let tt = TranspositionTable::<usize>::new_enabled(true);
        let a = tt.get_or_create(0, None);
        let b = tt.get_or_create(u64::MAX, None);
        let c = tt.get_or_create(0x0102_0304_0506_0708, None);

        assert!(ArenaRef::ptr_eq(&a, &tt.get(0).unwrap()));
        assert!(ArenaRef::ptr_eq(&b, &tt.get(u64::MAX).unwrap()));
        assert!(ArenaRef::ptr_eq(
            &c,
            &tt.get(0x0102_0304_0506_0708).unwrap()
        ));
        assert!(!ArenaRef::ptr_eq(&a, &b));
        assert!(!ArenaRef::ptr_eq(&a, &c));
        assert!(!ArenaRef::ptr_eq(&b, &c));
    }

    #[test]
    fn bucket_and_slot_indices_use_disjoint_hash_bits() {
        let base = 0_u64;
        let bucket_bit = 1_u64;
        let slot_bit = 1_u64 << BUCKET_INDEX_BITS;

        assert_ne!(
            TranspositionTable::<usize>::bucket_idx(base),
            TranspositionTable::<usize>::bucket_idx(base ^ bucket_bit)
        );
        assert_eq!(
            TranspositionTable::<usize>::slot_start(base),
            TranspositionTable::<usize>::slot_start(base ^ bucket_bit)
        );

        assert_eq!(
            TranspositionTable::<usize>::bucket_idx(base),
            TranspositionTable::<usize>::bucket_idx(base ^ slot_bit)
        );
        assert_ne!(
            TranspositionTable::<usize>::slot_start(base),
            TranspositionTable::<usize>::slot_start(base ^ slot_bit)
        );
    }

    #[test]
    fn batch_get_or_create_matches_sequential_duplicate_semantics() {
        let tt = TranspositionTable::<usize>::new_enabled(true);
        let mut lookups = [
            TtLookup::new(0x101, None),
            TtLookup::new(0x202, None),
            TtLookup::new(0x101, Some(0.5)),
            TtLookup::new(0x303, Some(-1.0)),
        ];

        tt.get_or_create_batch(&mut lookups);

        let a = lookups[0].node.unwrap();
        let b = lookups[1].node.unwrap();
        let a_dup = lookups[2].node.unwrap();
        let c = lookups[3].node.unwrap();

        assert!(ArenaRef::ptr_eq(&a, &a_dup));
        assert!(!ArenaRef::ptr_eq(&a, &b));
        assert!(!ArenaRef::ptr_eq(&a, &c));
        assert_eq!(tt.size(), 3);
        assert!(ArenaRef::ptr_eq(&a, &tt.get(0x101).unwrap()));
        assert!(ArenaRef::ptr_eq(&b, &tt.get(0x202).unwrap()));
        assert!(ArenaRef::ptr_eq(&c, &tt.get(0x303).unwrap()));

        let contention = tt.contention_snapshot();
        assert_eq!(contention.get_or_create_calls, 4);
        assert!((tt.hit_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn batch_get_or_create_uses_existing_nodes() {
        let tt = TranspositionTable::<usize>::new_enabled(true);
        let existing = tt.get_or_create(0xfeed, None);
        let mut lookups = [
            TtLookup::new(0xfeed, Some(1.0)),
            TtLookup::new(0xbeef, None),
            TtLookup::new(0xfeed, None),
            TtLookup::new(0xbeef, Some(-1.0)),
        ];

        tt.get_or_create_batch(&mut lookups);

        assert!(ArenaRef::ptr_eq(&existing, &lookups[0].node.unwrap()));
        assert!(ArenaRef::ptr_eq(&existing, &lookups[2].node.unwrap()));
        assert!(ArenaRef::ptr_eq(
            &lookups[1].node.unwrap(),
            &lookups[3].node.unwrap()
        ));
        assert_eq!(tt.size(), 2);
    }

    #[test]
    fn concurrent_get_or_create_same_hash_publishes_one_node() {
        let tt = Arc::new(TranspositionTable::<usize>::new_enabled(true));
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let tt = Arc::clone(&tt);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                tt.get_or_create(0xfeed_beef_cafe_babe, None)
            }));
        }

        let first = handles.remove(0).join().unwrap();
        for handle in handles {
            let node = handle.join().unwrap();
            assert!(ArenaRef::ptr_eq(&first, &node));
        }
        assert!(ArenaRef::ptr_eq(
            &first,
            &tt.get(0xfeed_beef_cafe_babe).unwrap()
        ));
        assert_eq!(tt.size(), 1);
    }
}

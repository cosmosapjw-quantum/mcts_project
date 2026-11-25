//! BQ++ Phase 2: lock-free immutable PolicyCache + ArcSwap publish.
//!
//! Replaces the `parking_lot::Mutex<Cache>` pattern from P06. The
//! audit's §1.7 finding ("gap_bits monotonic" was wrong) and §1.8
//! ("edge-local index vs action_id") motivated this redesign:
//!
//! 1. **Stale cache must reject halts.** The cache carries
//!    `root_visits_at_observe` and `edge_version_hash`; the halt
//!    path checks these against the live snapshot and treats any
//!    drift as "no certificate."
//! 2. **All arrays indexed by `edge_pos: u32`** — a dense per-search
//!    edge index, NEVER by `action_id`. The index bug class from
//!    audit §1.8 is impossible by construction.
//! 3. **Hot path is mutex-free.** `score_adjustment` does an
//!    `arc_swap::Guard` read (pointer + atomic counter increment, no
//!    contention).
//!
//! The cache is published once per `observe` boundary. Multiple
//! workers calling `observe` concurrently are safe: the last writer
//! wins, but the cache shape is monotonic in N (P03's freshness
//! guard ensures stale-but-not-bad is the worst case).

use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "idea-foundry")]
use std::sync::Mutex;

use arc_swap::ArcSwap;
use smallvec::SmallVec;

use std::sync::Arc;

/// Stable BQ++ extended HaltReason variants used by Phase 4+ stop rules.
/// The codes mirror `crate::mcts::quartz::HaltReason` reserved variants;
/// see [`crate::mcts::quartz::HaltReason::as_key`] for the JSON keys.
pub use crate::mcts::quartz::HaltReason;

/// Per-search policy cache published by `observe()` and read by
/// `score_adjustment()` / `should_halt()` on every worker.
///
/// All `Vec<f32>` arrays have `len() == n_children` and are indexed
/// by edge-local position (0..n_children). The struct is `Send + Sync`
/// because every field is either a primitive or an `Option` of a
/// primitive; the `ArcSwap` wrapper provides the atomic publish.
#[derive(Clone, Debug)]
pub struct PolicyCache {
    /// Monotonically-increasing publish epoch; lets `should_halt`
    /// detect when a cache is from an older `observe` cycle.
    pub epoch: u64,
    /// Snapshot of `root_visits` at publish time. Halt rejects when
    /// `current_root_visits != cache.root_visits` (audit §1.7
    /// regression: stale cache must not be allowed to halt).
    pub root_visits: u32,
    /// Hash of `(edge_pos, action_id, n_visits)` triplets at publish
    /// time. If the engine has materialized new edges since publish,
    /// the hash mismatches and the cache is treated as stale.
    pub edge_version_hash: u64,

    // Edge-local indexed; len() == n_children at observe time.
    /// Effective prior for PUCT (may be refresh-blended).
    pub p_eff: SmallVec<[f32; 32]>,
    /// Optional Q-override (regularized-policy variants like Grill 2020).
    /// Empty when no override; otherwise `len() == n_children`.
    pub q_ctrl: SmallVec<[f32; 32]>,
    /// Additive PUCT penalty per arm.
    pub penalty: SmallVec<[f32; 32]>,
    /// Knowledge Gradient (or KG-bound) per arm. Populated by Phase 4.
    pub kg: SmallVec<[f32; 32]>,
    /// Empirical-Bernstein lower CI on [0, 1] scale.
    pub lower: SmallVec<[f32; 32]>,
    /// Empirical-Bernstein upper CI on [0, 1] scale.
    pub upper: SmallVec<[f32; 32]>,

    /// Edge-local position of empirical best arm.
    pub best_pos: u16,
    /// Signed certificate gap: `L_b - max_{a≠b} U_a`. Positive ⇒ EB
    /// certificate fires.
    pub cert_gap: f32,
    /// Maximum KG / cost-per-pull across challengers; populated by
    /// Phase 4.
    pub max_kg_per_ms: f32,
    /// χ²-style prior surprise statistic (NOT a p-value, per audit
    /// §1.6).
    pub prior_surprise: f32,
    /// If a tactical sentinel found a forced move (Phase 5), this
    /// carries the edge_pos. The halt path then returns
    /// `Stop(TacticalForced)` with the forced action.
    pub forced_move_pos: Option<u16>,
}

impl PolicyCache {
    /// Construct an empty initial cache. Used when the search hasn't
    /// yet hit `min_total` and no real cache has been published.
    pub fn empty() -> Self {
        Self {
            epoch: 0,
            root_visits: 0,
            edge_version_hash: 0,
            p_eff: SmallVec::new(),
            q_ctrl: SmallVec::new(),
            penalty: SmallVec::new(),
            kg: SmallVec::new(),
            lower: SmallVec::new(),
            upper: SmallVec::new(),
            best_pos: 0,
            cert_gap: f32::NEG_INFINITY,
            max_kg_per_ms: 0.0,
            prior_surprise: 0.0,
            forced_move_pos: None,
        }
    }

    /// Returns true iff the cache is stale relative to the given snapshot.
    ///
    /// A cache is stale when:
    /// - The current root_visits differs from the cached value, OR
    /// - The current edge_version_hash differs from the cached value.
    ///
    /// Stale caches must not halt (audit §1.7 regression). The
    /// `score_adjustment` path can still read stale caches as a
    /// "best-effort" hint — but the halt path treats stale as
    /// "Continue" regardless of cache.cert_gap.
    pub fn is_stale_for(&self, current_root_visits: u32, current_edge_hash: u64) -> bool {
        self.root_visits != current_root_visits || self.edge_version_hash != current_edge_hash
    }
}

/// Atomic publish of the immutable PolicyCache.
///
/// Hot path:
/// ```ignore
/// let cache = publisher.load();   // arc_swap::Guard, no lock
/// let p_eff = cache.p_eff[edge.edge_pos as usize];   // O(1) array read
/// ```
///
/// `observe()` builds a new `PolicyCache` and publishes via
/// `publisher.store(Arc::new(cache))`. Workers calling `load()`
/// concurrently see either the old or the new cache — never a torn
/// read.
pub struct PolicyCachePublisher {
    inner: ArcSwap<PolicyCache>,
    /// Monotone epoch counter for `observe` sequencing.
    next_epoch: AtomicU64,
    /// In Foundry builds, serialize all writers while readers remain
    /// lock-free.  Default production builds retain the original direct-store
    /// path.  An older writer may not reserve an epoch, pause, and overwrite a
    /// newer Foundry CAS publish when it resumes.
    #[cfg(feature = "idea-foundry")]
    writer_lock: Mutex<()>,
}

impl PolicyCachePublisher {
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::new(Arc::new(PolicyCache::empty())),
            next_epoch: AtomicU64::new(1),
            #[cfg(feature = "idea-foundry")]
            writer_lock: Mutex::new(()),
        }
    }

    /// Atomic load — hot path; never blocks.
    pub fn load(&self) -> arc_swap::Guard<Arc<PolicyCache>> {
        self.inner.load()
    }

    /// Atomic publish. The `epoch` field on `cache` will be overwritten
    /// with the next monotone counter value.
    pub fn store(&self, cache: PolicyCache) {
        #[cfg(feature = "idea-foundry")]
        {
            self.store_inner(cache, |_| {});
        }
        #[cfg(not(feature = "idea-foundry"))]
        {
            let mut cache = cache;
            cache.epoch = self.next_epoch.fetch_add(1, Ordering::AcqRel);
            self.inner.store(Arc::new(cache));
        }
    }

    #[cfg(feature = "idea-foundry")]
    fn store_inner<F>(&self, mut cache: PolicyCache, before_publish: F)
    where
        F: FnOnce(u64),
    {
        let _writer = self
            .writer_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let epoch = self.next_epoch.fetch_add(1, Ordering::AcqRel);
        cache.epoch = epoch;
        before_publish(epoch);
        self.inner.store(Arc::new(cache));
    }

    /// Deterministic test seam for the epoch-reserved/pre-publish interval.
    #[cfg(all(test, feature = "idea-foundry"))]
    pub(crate) fn store_with_before_publish_hook<F>(&self, cache: PolicyCache, hook: F)
    where
        F: FnOnce(u64),
    {
        self.store_inner(cache, hook);
    }

    /// Feature-gated compare-and-swap publish used by the Idea Foundry score
    /// vector bridge.  A concurrent `observe()` publish wins rather than being
    /// overwritten; the caller receives that newer cache and must revalidate
    /// its expected epoch and freshness before retrying.
    #[cfg(feature = "idea-foundry")]
    pub(crate) fn compare_and_store(
        &self,
        expected: &Arc<PolicyCache>,
        mut replacement: PolicyCache,
    ) -> Result<u64, Arc<PolicyCache>> {
        let _writer = self
            .writer_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.inner.load_full();
        if !Arc::ptr_eq(&current, expected) {
            return Err(current);
        }
        let replacement_epoch = self.next_epoch.fetch_add(1, Ordering::AcqRel);
        replacement.epoch = replacement_epoch;
        let replacement = Arc::new(replacement);
        let previous = self
            .inner
            .compare_and_swap(expected, Arc::clone(&replacement));
        if Arc::ptr_eq(&*previous, expected) {
            Ok(replacement_epoch)
        } else {
            Err(arc_swap::Guard::into_inner(previous))
        }
    }

    /// Convenience: snapshot the current cache contents (clones the
    /// `Arc`, no full copy).
    pub fn snapshot(&self) -> Arc<PolicyCache> {
        self.inner.load_full()
    }
}

impl Default for PolicyCachePublisher {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-edge view exposed to a SearchPolicy that uses the cache.
///
/// `edge_pos` is the dense per-search-edge index used to look up the
/// cache arrays. `action_id` is the game-engine-specific action
/// identifier (e.g. chess Move::to_index(), Go board position) that
/// the policy never indexes into per-edge arrays — it's a separate
/// communication channel back to the engine.
#[derive(Copy, Clone, Debug)]
pub struct EdgeRef {
    pub edge_pos: u32,
    pub action_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::thread;

    fn make_cache(epoch: u64, root_visits: u32, edge_hash: u64) -> PolicyCache {
        PolicyCache {
            epoch,
            root_visits,
            edge_version_hash: edge_hash,
            p_eff: SmallVec::from_slice(&[0.5, 0.3, 0.2]),
            q_ctrl: SmallVec::new(),
            penalty: SmallVec::from_slice(&[0.0; 3]),
            kg: SmallVec::from_slice(&[0.0; 3]),
            lower: SmallVec::from_slice(&[0.0; 3]),
            upper: SmallVec::from_slice(&[1.0; 3]),
            best_pos: 0,
            cert_gap: -1.0,
            max_kg_per_ms: 0.0,
            prior_surprise: 0.0,
            forced_move_pos: None,
        }
    }

    /// Phase 2: cache publish is visible to a concurrent reader.
    /// Writes-then-reads (rather than concurrent loops) — the
    /// arc_swap publish is store-release / load-acquire, so a
    /// post-publish read must see the new value.
    #[test]
    fn test_phase2_cache_publish_visible_to_concurrent_reader() {
        let publisher = Arc::new(PolicyCachePublisher::new());

        // Initially the cache is the empty default (root_visits = 0).
        assert_eq!(publisher.load().root_visits, 0);

        // Publish 100 caches, then verify the reader sees the latest.
        for i in 1..=100u32 {
            publisher.store(make_cache(0, i, i as u64));
        }

        // Spawn reader threads and verify they all see the latest publish.
        let mut handles = vec![];
        for _ in 0..4 {
            let p = Arc::clone(&publisher);
            handles.push(thread::spawn(move || p.load().root_visits));
        }
        for h in handles {
            let v = h.join().unwrap();
            assert_eq!(v, 100, "reader saw {v}, expected 100");
        }
    }

    /// Phase 2: stale cache (root_visits drift) is rejected by
    /// `is_stale_for`.
    #[test]
    fn test_phase2_cache_stale_root_visits_rejected() {
        let cache = make_cache(1, 100, 0xabcd);
        // Live snapshot has root_visits=200; cache is stale.
        assert!(cache.is_stale_for(200, 0xabcd));
    }

    /// Phase 2: stale cache (edge_version_hash drift) is rejected.
    #[test]
    fn test_phase2_cache_stale_edge_hash_rejected() {
        let cache = make_cache(1, 100, 0xabcd);
        // Same root_visits but different edge hash ⇒ stale.
        assert!(cache.is_stale_for(100, 0xbeef));
    }

    /// Phase 2: matching root_visits AND edge_hash ⇒ NOT stale.
    #[test]
    fn test_phase2_cache_fresh_when_both_match() {
        let cache = make_cache(1, 100, 0xabcd);
        assert!(!cache.is_stale_for(100, 0xabcd));
    }

    /// Phase 2: epoch is monotonically increasing across publishes.
    #[test]
    fn test_phase2_cache_epoch_monotone() {
        let publisher = PolicyCachePublisher::new();
        let mut last_epoch = 0u64;
        for i in 1..=10u32 {
            publisher.store(make_cache(0, i, i as u64));
            let g = publisher.load();
            assert!(
                g.epoch > last_epoch,
                "epoch={} last={}",
                g.epoch,
                last_epoch
            );
            last_epoch = g.epoch;
        }
    }

    /// Phase 2: empty cache fields are present so callers can index
    /// into them without checking for None.
    #[test]
    fn test_phase2_empty_cache_has_zero_arrays() {
        let cache = PolicyCache::empty();
        assert_eq!(cache.p_eff.len(), 0);
        assert_eq!(cache.kg.len(), 0);
        assert_eq!(cache.lower.len(), 0);
        assert_eq!(cache.upper.len(), 0);
        assert_eq!(cache.best_pos, 0);
        assert!(cache.cert_gap.is_finite() || cache.cert_gap == f32::NEG_INFINITY);
        assert!(cache.forced_move_pos.is_none());
    }

    /// Phase 2: edge_pos and action_id are independent. The audit's
    /// §1.8 bug class is constructively impossible — `EdgeRef` carries
    /// both fields and they never share storage.
    #[test]
    fn test_phase2_edge_pos_distinct_from_action_id() {
        let edge = EdgeRef {
            edge_pos: 1,    // dense per-search edge index (must be < n_children)
            action_id: 481, // sparse chess action id, much larger than n_children
        };
        // Cache lookup uses edge_pos (within array bounds).
        let cache = make_cache(0, 100, 0);
        let p = cache.p_eff[edge.edge_pos as usize];
        assert_eq!(p, 0.3); // make_cache constructs [0.5, 0.3, 0.2]
                            // action_id is forwarded back to the engine; never indexes the cache.
        assert_eq!(edge.action_id, 481);
        // If we incorrectly indexed by action_id, this would panic.
        assert!(edge.action_id as usize > cache.p_eff.len());
    }
}

//! Negamax Backpropagation + Welford online variance
//!
//! 매 backup step:
//!   1. Virtual Loss 환원
//!   2. N += 1, W += value  (negamax 부호 반전)
//!   3. M2 Welford update: M2 += (value - Q_prev)(value - Q_new)
//!      → edge_sigma() = √(M2/(N-1)) = per-edge std 근사 (아래 A3-b 참고)
//!   4. parent.n_total += 1
//!
//! MQFT RESTORE40: parent-node lock makes N/W/M2/S2 and virtual-loss
//! consumption coherent for selection. Atomic fields retain upstream ABI.
//! The exact second raw moment is stored separately and audited at barriers.

use std::sync::atomic::Ordering;

use crate::game::GameState;
use crate::mcts::node::{atomic_f64_add, atomic_f64_load, PathEdge};

pub fn backprop<G: GameState>(path: &[PathEdge<G::Move>], leaf_value: f32) -> u64 {
    let mut clock = u64::MAX;
    let mut value = leaf_value;

    for (depth, pe) in path.iter().enumerate().rev() {
        value = -value; // negamax: 부모 관점 부호 반전

        // Phase 7 C (2026-04-26): lock-free slab read.
        let guard = pe.parent.stats_lock.lock();
        let edges = pe.parent.read_edges();
        let e = &edges[pe.edge_idx];

        // Virtual Loss: remove the exact (vvisit, vvalue) applied during select
        let (vv, vq) = pe.applied_vl;
        e.remove_vl(vv, vq);

        // N 증가
        let n_old = e.n.fetch_add(1, Ordering::AcqRel) as f64;
        let n_new = n_old + 1.0;

        // All writes to this parent are serialized; W and N now match.
        let w_before = atomic_f64_load(&e.w) as f64;

        // W 갱신
        e.add_w(value);
        atomic_f64_add(&e.s2, (value as f64) * (value as f64));

        // Welford M2 (per-edge σᵢ for §6.1.1)
        // Use locally computed w_old/w_new to avoid reading e.w again
        // (another thread may have updated it between add_w and the read)
        if n_old >= 1.0 {
            let mu_old = w_before / n_old;
            let mu_new = (w_before + value as f64) / n_new;
            let delta_m2 = (value as f64 - mu_old) * (value as f64 - mu_new);
            atomic_f64_add(&e.m2, delta_m2);
        }

        atomic_f64_add(&pe.parent.w_total, value as f64);
        pe.parent.n_total.fetch_add(1, Ordering::AcqRel);
        if depth == 0 && pe.root_selection.is_some() {
            clock = pe.parent.chronology.fetch_add(1, Ordering::Relaxed);
        }
        let child = e.child;
        drop(guard);
        // Repeated edge visits are not necessarily transpositions. This is
        // retained upstream telemetry, never interpreted as distinct paths.
        if n_old >= 1.0 {
            child.record_rtt_hit(value);
        }
    }
    clock
}

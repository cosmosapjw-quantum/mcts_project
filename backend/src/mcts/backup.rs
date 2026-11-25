//! Negamax Backpropagation + Welford online variance
//!
//! 매 backup step:
//!   1. Virtual Loss 환원
//!   2. N += 1, W += value  (negamax 부호 반전)
//!   3. M2 Welford update: M2 += (value - Q_prev)(value - Q_new)
//!      → edge_sigma() = √(M2/(N-1)) = per-edge std 근사 (아래 A3-b 참고)
//!   4. parent.n_total += 1
//!
//! ## A3-b audit note: 병렬 백업 하 M2 갱신은 정확하지 않은 근사다
//!
//! `n.fetch_add`(atomic 예약)와 `w`의 load(`w_before`)는 개별로는 각각
//! atomic이지만, 그 둘의 조합(n_old ↔ w_before의 페어링)은 하나의 원자적
//! 연산이 아니다. 두 스레드가 같은 edge를 동시에 backup할 때, 이 스레드가
//! `n_old`를 예약한 직후·`w_before` load 이전에 다른 스레드가 자신의
//! `fetch_add` + `add_w`를 먼저 끝내버리면, `w_before`는 실제로는
//! `n_old`개가 아니라 `n_old+1`개(또는 그 이상) 값의 누적합을 담고 있을 수
//! 있다. 이 경우 `mu_old = w_before / n_old`는 "정확히 n_old개 값의
//! 평균"이라는 Welford 재귀식의 전제를 깨고, `delta_m2`가 실제로 관측된
//! 갱신 순서와 불일치하는 값이 될 수 있다.
//!
//! 결과적으로 `edge_sigma()`(및 이를 소비하는 `σ_Q`)는 heavy contention
//! 하에서 정확한 표본표준편차가 아니라 **잡음 섞인 근사**다. `σ_Q`는
//! `QuartzController`의 root penalty 계산과 `ParallelismController`의
//! adaptive split virtual-loss 크기(`vvalue = σ_Q × depth_decay × ...`,
//! parallel.rs) 양쪽에 공급된다 — 즉 이 근사 오차는 penalty 강도와 VL
//! 크기 모두에 전파된다.
//!
//! 완전한 수정(락 또는 CAS 루프로 {n,w,m2} 트리플을 원자적으로 갱신)은
//! 이 lock-free hot path의 설계 목표(§7 Production Hot-Path Contract,
//! docs/QUARTZ_THEORY.md)와 충돌하므로 이번 패스의 범위 밖이다. 여기서는
//! 근사임을 명시적으로 기록한다 — docs/CLAIM_LEDGER.md 참고.

use std::sync::atomic::Ordering;

use crate::game::GameState;
use crate::mcts::node::{atomic_f64_add, atomic_f64_load, PathEdge};

pub fn backprop<G: GameState>(path: &[PathEdge<G::Move>], leaf_value: f32) {
    let mut value = leaf_value;

    for pe in path.iter().rev() {
        value = -value; // negamax: 부모 관점 부호 반전

        // Phase 7 C (2026-04-26): lock-free slab read.
        let edges = pe.parent.read_edges();
        let e = &edges[pe.edge_idx];

        // Virtual Loss: remove the exact (vvisit, vvalue) applied during select
        let (vv, vq) = pe.applied_vl;
        e.remove_vl(vv, vq);

        // N 증가
        let n_old = e.n.fetch_add(1, Ordering::AcqRel) as f64;
        let n_new = n_old + 1.0;

        // Snapshot W before our update to compute correct Welford delta locally.
        // A3-b: (n_old, w_before) is NOT an atomic pair — see the module-level
        // doc comment above. Under concurrent backups on the same edge,
        // w_before can already include another thread's value, making
        // mu_old/delta_m2 below an approximation, not an exact Welford step.
        let w_before = atomic_f64_load(&e.w) as f64;

        // W 갱신
        e.add_w(value);

        // Welford M2 (per-edge σᵢ for §6.1.1)
        // Use locally computed w_old/w_new to avoid reading e.w again
        // (another thread may have updated it between add_w and the read)
        if n_old >= 1.0 {
            let mu_old = w_before / n_old;
            let mu_new = (w_before + value as f64) / n_new;
            let delta_m2 = (value as f64 - mu_old) * (value as f64 - mu_new);
            atomic_f64_add(&e.m2, delta_m2);
        }

        // §6.3 MERGE: 같은 엣지가 2번 이상 백업 = 여러 경로가 이 child에 수렴
        // n_old >= 1 (이번 backup 전 N이 1 이상) = 이전에도 방문됐음 = transposition
        if n_old >= 1.0 {
            e.child.record_rtt_hit(value);
        }
        // Phase 7 C: no guard to drop — slab read is lock-free.

        atomic_f64_add(&pe.parent.w_total, value as f64);
        pe.parent.n_total.fetch_add(1, Ordering::AcqRel);
    }
}

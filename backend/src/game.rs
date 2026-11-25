//! specs/010-game-api.md 구현
//!
//! GameState 트레이트: 게임-무관(game-agnostic) 인터페이스
//! Evaluator 트레이트: NN / random playout 양쪽을 동일 인터페이스로 사용
//!
//! 핵심 설계 원칙:
//!   - apply_move: pure function (입력 불변, 새 상태 반환)
//!   - outcome: negamax convention (현재 플레이어 관점, +1/-1/0)
//!   - current_player: +1 or -1

use std::fmt::Debug;
use std::hash::Hash;

// ─────────────────────────────────────────────
// § 1. GameState 트레이트
// ─────────────────────────────────────────────

#[allow(dead_code)]
pub trait GameState: Clone + Send + Sync + 'static {
    /// 이 게임의 착수(이동) 타입
    type Move: Copy + Eq + Hash + Send + Sync + Debug + 'static;

    /// Undo info returned by `apply_move_in_place` and consumed by `undo_move`.
    /// Games override this with a compact delta struct (Phase 6.1, 2026-04-25).
    /// The default — `Undo = Self` plus clone-and-replace bodies in
    /// `apply_move_in_place` / `undo_move` — preserves the legacy
    /// pure-function semantics without requiring every impl to migrate.
    type Undo: Send;

    /// 게임 초기 상태 생성
    fn initial() -> Self;

    /// 현재 차례 플레이어 (+1 or -1)
    fn current_player(&self) -> i8;

    /// 현재 상태에서 합법 착수 목록
    fn legal_moves(&self) -> Vec<Self::Move>;

    /// Uniform policy/value evaluation for the current legal moves. The
    /// default uses `legal_moves()`, while hot games can override this to
    /// fill the policy directly and avoid an intermediate legal-move Vec.
    fn uniform_eval(&self, value: f32) -> EvalResult<Self::Move> {
        EvalResult::uniform(&self.legal_moves(), value)
    }

    /// 착수를 적용해 새 상태 반환 (pure function, self 불변)
    fn apply_move(&self, mv: Self::Move) -> Self;

    /// Apply a move in place. The returned `Undo` is consumed by `undo_move`
    /// to restore the previous state. Hot paths (MCTS select descent,
    /// edge materialization) use this to avoid full state clones.
    fn apply_move_in_place(&mut self, mv: Self::Move) -> Self::Undo;

    /// Apply a move destructively when the caller will never need to undo it.
    /// Selection descent uses this path because it only advances toward the
    /// leaf state. Games with large `Undo` payloads can override this to avoid
    /// constructing reverse state that would be dropped immediately.
    fn apply_move_in_place_no_undo(&mut self, mv: Self::Move) {
        let undo = self.apply_move_in_place(mv);
        drop(undo);
    }

    /// Reverse a prior `apply_move_in_place`. After this call, the receiver
    /// must equal the state observed before the matched apply, including
    /// hash and any auxiliary fields.
    fn undo_move(&mut self, undo: Self::Undo);

    /// Whether MCTS should reuse one mutable state per worker during
    /// synchronous selection. Games should return true only when `Undo` is a
    /// compact delta. If `Undo = Self` or otherwise stores a full board/state
    /// snapshot, the clone-free select path is usually slower and more memory
    /// hungry than the default clone-and-descend path.
    fn uses_reusable_select_scratch() -> bool {
        false
    }

    /// Whether expansion may skip re-sorting an evaluator policy that is
    /// already non-increasing after prior clamping. Keep this false unless
    /// the resulting move-order semantics and performance have both been
    /// measured for the game.
    fn can_skip_sorted_policy_resort() -> bool {
        false
    }

    /// 게임 종료 여부
    fn is_terminal(&self) -> bool;

    /// 게임 결과 — negamax convention:
    ///   현재 플레이어(current_player) 관점으로 +1(승) / -1(패) / 0(무)
    ///   terminal이 아닌 상태에서는 호출하지 않는다.
    fn outcome(&self) -> f32;

    /// Zobrist 해시 (TT 키)
    fn hash(&self) -> u64;

    /// Exact TT key. Defaults to `hash()`, but history-sensitive games can
    /// override this to include rule state beyond the board hash.
    fn tt_hash(&self) -> u64 {
        self.hash()
    }

    /// 전체 행동 공간 크기 (NN 정책 출력 차원)
    fn num_actions(&self) -> usize;

    /// 착수 → 행동 인덱스
    fn move_to_idx(&self, mv: Self::Move) -> usize;

    /// 행동 인덱스 → 착수 (불합법이면 None)
    fn idx_to_move(&self, idx: usize) -> Option<Self::Move>;

    /// NN 입력용 feature plane (flat, channel-first)
    fn encode_planes(&self) -> Vec<f32> {
        let mut out = Vec::new();
        self.encode_planes_into(&mut out);
        out
    }

    /// Encode NN input into a caller-provided buffer so hot paths can reuse
    /// allocation capacity across requests.
    fn encode_planes_into(&self, out: &mut Vec<f32>) {
        out.clear();
    }

    /// Monotonic encoder revision for cache fingerprints. Override when a
    /// game's feature encoding changes incompatibly.
    fn eval_encoder_revision(&self) -> u32 {
        1
    }

    /// [OPT] Check if move wins for current player without full apply_move.
    /// Default: apply_move + check terminal (expensive).
    /// Override for O(1) win check (e.g., Gomoku line scan).
    fn is_winning_move(&self, mv: Self::Move) -> bool {
        let next = self.apply_move(mv);
        next.is_terminal() && next.outcome() != 0.0
    }

    /// [OPT] Pick a random legal move without Vec allocation.
    /// Default: legal_moves() + choose (allocates Vec).
    /// Override for O(n) board scan with no allocation.
    fn random_legal_move(&self, rand_idx: usize) -> Option<Self::Move> {
        let moves = self.legal_moves();
        if moves.is_empty() {
            None
        } else {
            Some(moves[rand_idx % moves.len()])
        }
    }

    /// [OPT] Count of legal moves without allocating Vec.
    fn legal_move_count(&self) -> usize {
        self.legal_moves().len()
    }

    /// Board state for selfplay recording (0/1/2 encoding: empty/black/white).
    /// Used by the Rust selfplay state machine to record game history.
    /// Default: encode_planes() flattened (games should override for compact format).
    fn board_state_record(&self) -> Vec<i64> {
        self.encode_planes()
            .iter()
            .map(|&v| {
                if v > 0.5 {
                    1
                } else if v < -0.5 {
                    2
                } else {
                    0
                }
            })
            .collect()
    }
}

#[inline]
pub fn tt_mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline]
pub fn tt_combine(seed: u64, value: u64) -> u64 {
    let mixed = tt_mix64(value.wrapping_add(0x9e37_79b9_7f4a_7c15));
    tt_mix64(seed ^ mixed.rotate_left(25) ^ 0x517c_c1b7_2722_0a95)
}

// ─────────────────────────────────────────────
// § 2. Evaluator 트레이트 및 결과 타입
// ─────────────────────────────────────────────

/// Leaf 평가 결과
#[derive(Debug, Clone)]
pub struct EvalResult<M> {
    /// (착수, 사전 확률) 쌍 — 합 ≈ 1.0, 합법수만 포함
    pub policy: Vec<(M, f32)>,

    /// 현재 플레이어 관점 가치: [-1.0, 1.0]
    pub value: f32,
}

impl<M: Copy + Eq + Hash + Debug> EvalResult<M> {
    /// 균등 사전 확률 + 지정 가치로 생성
    pub fn uniform(legal_moves: &[M], value: f32) -> Self {
        let n = legal_moves.len();
        let p = if n > 0 { 1.0 / n as f32 } else { 0.0 };
        EvalResult {
            policy: legal_moves.iter().map(|&m| (m, p)).collect(),
            value,
        }
    }
}

/// Leaf 평가기 트레이트
/// NN 추론, 랜덤 플레이아웃, 균등 사전 확률 등을 동일 인터페이스로 사용
pub trait Evaluator<G: GameState>: Send + Sync {
    fn evaluate(&self, state: &G) -> EvalResult<G::Move>;
}

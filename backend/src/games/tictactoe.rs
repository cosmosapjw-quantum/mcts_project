//! TicTacToe 3×3 — GameState 구현체
//!
//! 설계:
//!   - Board: [i8; 9]  (+1=X, -1=O, 0=Empty)
//!   - current_player: +1 or -1
//!   - outcome: negamax — 현재 플레이어 관점
//!   - Zobrist hash: 고정 시드, 증분 XOR

use crate::game::GameState;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::LazyLock;

// ─────────────────────────────────────────────
// § Zobrist 테이블 (thread-local, 프로그램 시작시 1회 초기화)
// ─────────────────────────────────────────────

struct ZobTable {
    piece: [[u64; 9]; 2], // [player_idx 0=X 1=O][square]
    side: u64,            // O 차례일 때 XOR
}

impl ZobTable {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut piece = [[0u64; 9]; 2];
        for p in 0..2 {
            for sq in 0..9 {
                piece[p][sq] = rng.gen();
            }
        }
        ZobTable {
            piece,
            side: rng.gen(),
        }
    }
    fn piece_hash(&self, player: i8, sq: usize) -> u64 {
        let pi = if player > 0 { 0 } else { 1 };
        self.piece[pi][sq]
    }
}

unsafe impl Sync for ZobTable {}

static ZOB: LazyLock<ZobTable> = LazyLock::new(|| ZobTable::new(0xDEAD_BEEF_CAFE_1234));

// ─────────────────────────────────────────────
// § TicTacToe 상태
// ─────────────────────────────────────────────

/// History depth for AlphaZero-style encoding (T=8 timesteps including current).
const TICTACTOE_HISTORY_LEN: usize = 8;
const TICTACTOE_HISTORY_MOVES: usize = TICTACTOE_HISTORY_LEN - 1;

#[derive(Clone, Debug, PartialEq)]
pub struct TicTacToe {
    board: [i8; 9],     // +1=X, -1=O, 0=empty
    current_player: i8, // +1 or -1
    hash: u64,          // Zobrist 증분 해시
    /// Recent move history for AlphaZero-style feature reconstruction.
    recent_moves: [u8; TICTACTOE_HISTORY_MOVES],
    recent_move_len: u8,
}

impl TicTacToe {
    pub fn from_board(board: &[i8], player: i8) -> Self {
        let mut state = TicTacToe {
            board: [0; 9],
            current_player: if player >= 0 { 1 } else { -1 },
            hash: 0,
            recent_moves: [0; TICTACTOE_HISTORY_MOVES],
            recent_move_len: 0,
        };
        for i in 0..9.min(board.len()) {
            state.board[i] = match board[i] {
                v if v > 0 => 1,
                v if v < 0 => -1,
                _ => 0,
            };
        }
        {
            let z = &*ZOB;
            for (sq, piece) in state.board.iter().copied().enumerate() {
                if piece != 0 {
                    state.hash ^= z.piece_hash(piece, sq);
                }
            }
            if state.current_player < 0 {
                state.hash ^= z.side;
            }
        }
        state
    }

    fn check_winner(&self) -> i8 {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];
        for [a, b, c] in &LINES {
            let v = self.board[*a];
            if v != 0 && v == self.board[*b] && v == self.board[*c] {
                return v; // +1 (X) 또는 -1 (O)
            }
        }
        0
    }

    #[inline]
    fn push_recent_move(&mut self, mv: usize) {
        let mv = mv as u8;
        let len = self.recent_move_len as usize;
        if len < TICTACTOE_HISTORY_MOVES {
            self.recent_moves[len] = mv;
            self.recent_move_len += 1;
            return;
        }
        self.recent_moves.copy_within(1..TICTACTOE_HISTORY_MOVES, 0);
        self.recent_moves[TICTACTOE_HISTORY_MOVES - 1] = mv;
    }
}

impl std::fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = |v: i8| match v {
            1 => 'X',
            -1 => 'O',
            _ => '.',
        };
        let b = &self.board;
        writeln!(f, " {} | {} | {}", s(b[0]), s(b[1]), s(b[2]))?;
        writeln!(f, "---+---+---")?;
        writeln!(f, " {} | {} | {}", s(b[3]), s(b[4]), s(b[5]))?;
        writeln!(f, "---+---+---")?;
        write!(f, " {} | {} | {}", s(b[6]), s(b[7]), s(b[8]))
    }
}

// ─────────────────────────────────────────────
// § GameState 구현
// ─────────────────────────────────────────────

impl GameState for TicTacToe {
    type Move = usize; // 0..8 (보드 위치)
    type Undo = Self;

    fn initial() -> Self {
        TicTacToe {
            board: [0; 9],
            current_player: 1,
            hash: 0,
            recent_moves: [0; TICTACTOE_HISTORY_MOVES],
            recent_move_len: 0,
        }
    }

    fn current_player(&self) -> i8 {
        self.current_player
    }

    fn legal_moves(&self) -> Vec<usize> {
        if self.is_terminal() {
            return vec![];
        }
        self.board
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn apply_move(&self, mv: usize) -> Self {
        let mut next = self.clone();

        // 보드 갱신
        next.board[mv] = self.current_player;
        // Zobrist 증분 갱신: 기물 배치 + 차례 전환
        next.hash ^= ZOB.piece_hash(self.current_player, mv);
        next.hash ^= ZOB.side;
        // 차례 전환
        next.current_player = -self.current_player;
        next.push_recent_move(mv);
        next
    }

    /// Phase 6.1: clone-based fallback. TicTacToe is small (~40 B); the
    /// clone is cheap, so the make-unmake hot-path optimization isn't
    /// worth the per-game implementation cost.
    fn apply_move_in_place(&mut self, mv: usize) -> Self {
        let next = self.apply_move(mv);
        std::mem::replace(self, next)
    }

    fn apply_move_in_place_no_undo(&mut self, mv: usize) {
        let next = self.apply_move(mv);
        *self = next;
    }

    fn undo_move(&mut self, undo: Self) {
        *self = undo;
    }

    fn is_terminal(&self) -> bool {
        self.check_winner() != 0 || self.board.iter().all(|&v| v != 0)
    }

    /// negamax convention: 현재 플레이어 관점
    /// X가 방금 이겼으면 current_player = O → outcome = -1 (O 패배)
    fn outcome(&self) -> f32 {
        let winner = self.check_winner();
        if winner == self.current_player {
            1.0
        } else if winner != 0 {
            -1.0
        } else {
            0.0
        } // draw
    }

    fn hash(&self) -> u64 {
        self.hash
    }

    fn num_actions(&self) -> usize {
        9
    }

    fn move_to_idx(&self, mv: usize) -> usize {
        mv
    }

    fn idx_to_move(&self, idx: usize) -> Option<usize> {
        if idx < 9 {
            Some(idx)
        } else {
            None
        }
    }

    /// NN 입력 feature planes (AlphaZero-style: [17, 3, 3])
    ///   planes 0-1:   현재(t=0) 내 돌, 상대 돌
    ///   planes 2-3:   t=1 (1수 전) 내 돌, 상대 돌
    ///   ...
    ///   planes 14-15: t=7 (7수 전) 내 돌, 상대 돌
    ///   plane 16:     현재 플레이어 색상 (X=1, O=0)
    fn encode_planes_into(&self, out: &mut Vec<f32>) {
        let n = 9;
        let total_planes = TICTACTOE_HISTORY_LEN * 2 + 1; // 17
        out.clear();
        out.resize(total_planes * n, 0.0);
        let cp = self.current_player;

        // t=0: current board
        for (i, &v) in self.board.iter().enumerate() {
            if v == cp {
                out[i] = 1.0;
            } else if v != 0 {
                out[n + i] = 1.0;
            }
        }

        // t=1..7: reconstruct prior boards by undoing recent moves.
        let mut hist_board = self.board;
        for (k, &mv) in self.recent_moves[..self.recent_move_len as usize]
            .iter()
            .rev()
            .enumerate()
        {
            let t = k + 1;
            hist_board[mv as usize] = 0;
            let base = t * 2 * n;
            for (i, &v) in hist_board.iter().enumerate() {
                if v == cp {
                    out[base + i] = 1.0;
                } else if v != 0 {
                    out[base + n + i] = 1.0;
                }
            }
        }

        // Color plane
        let color_val = if cp == 1 { 1.0 } else { 0.0 };
        let color_base = (total_planes - 1) * n;
        out[color_base..color_base + n].fill(color_val);
    }

    fn board_state_record(&self) -> Vec<i64> {
        self.board
            .iter()
            .map(|&v| match v {
                1 => 1,
                -1 => 2,
                _ => 0,
            })
            .collect()
    }
}

// ─────────────────────────────────────────────
// § 단위 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negamax_outcome() {
        //  X 승리: X(0) O(3) X(1) O(4) X(2)
        let mut s = TicTacToe::initial();
        for mv in [0, 3, 1, 4, 2] {
            s = s.apply_move(mv);
        }
        // X가 이겼음, current_player = O
        assert!(s.is_terminal());
        assert_eq!(s.current_player(), -1); // O 차례
        assert_eq!(s.outcome(), -1.0); // O 관점에서 패배
    }

    #[test]
    fn test_draw() {
        // 무승부: X O X / X X O / O X O  (시퀀스 0→1→2→6→4→8→3→5→7)
        let mut s = TicTacToe::initial();
        for mv in [0, 1, 2, 6, 4, 8, 3, 5, 7] {
            s = s.apply_move(mv);
        }
        // 보드가 가득 찼고 승자 없음
        assert!(s.is_terminal());
        assert_eq!(s.outcome(), 0.0);
    }

    #[test]
    fn test_hash_transposition() {
        // 두 가지 순서로 같은 보드 → 해시 동일
        let mut a = TicTacToe::initial();
        for mv in [0, 4, 2, 6] {
            a = a.apply_move(mv);
        }

        let mut b = TicTacToe::initial();
        for mv in [2, 6, 0, 4] {
            b = b.apply_move(mv);
        }

        assert_eq!(a.board, b.board);
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn test_apply_move_pure() {
        let s = TicTacToe::initial();
        let s2 = s.apply_move(4);
        assert_eq!(s.board, [0i8; 9]); // 원본 불변
        assert_eq!(s2.board[4], 1);
    }
}

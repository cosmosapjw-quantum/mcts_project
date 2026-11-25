//! Gomoku (오목) — GameState 구현체
//!
//! 규칙: 표준 5목 (5개 이상 일렬이면 승리)
//!   이후 Renju(렌주) 확장 시 overline/double-3/double-4 금수 추가 예정
//!
//! 지원 보드 크기: 9×9 / 13×13 / 15×15 (런타임 파라미터)
//!
//! Zobrist 해시:
//!   piece[player_idx][square], side — 최대 19×19=361 칸 대응 전역 테이블
//!
//! encode_planes (NN 입력):
//!   채널 0: 현재 플레이어 기물 위치 (1.0)
//!   채널 1: 상대 플레이어 기물 위치 (1.0)
//!   채널 2: 현재 플레이어 색상 (black=1.0, white=0.0)

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::fmt;
use std::sync::LazyLock;

use crate::game::{EvalResult, GameState};

// ─────────────────────────────────────────────
// § Zobrist 테이블 (최대 19×19)
// ─────────────────────────────────────────────

const MAX_SQ: usize = 19 * 19; // 361

struct GomokuZob {
    piece: [[u64; MAX_SQ]; 2], // [player_idx 0=black(+1) 1=white(-1)][square]
    side: u64,
}

impl GomokuZob {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut piece = [[0u64; MAX_SQ]; 2];
        for p in 0..2 {
            for sq in 0..MAX_SQ {
                piece[p][sq] = rng.gen();
            }
        }
        GomokuZob {
            piece,
            side: rng.gen(),
        }
    }
    fn piece_hash(&self, player: i8, sq: usize) -> u64 {
        let pi = if player > 0 { 0 } else { 1 };
        self.piece[pi][sq]
    }
}

unsafe impl Sync for GomokuZob {}

static GOB: LazyLock<GomokuZob> = LazyLock::new(|| GomokuZob::new(0xFEED_FACE_CAFE_BABE));

thread_local! {
    static GOMOKU_FEATURE_SCRATCH: RefCell<GomokuFeatureScratch> =
        RefCell::new(GomokuFeatureScratch::default());
}

struct GomokuFeatureScratch {
    touched: Vec<u16>,
    recent_epoch: [u16; MAX_SQ],
    recent_rank: [u8; MAX_SQ],
    color_plane_active: bool,
    current_epoch: u16,
}

impl Default for GomokuFeatureScratch {
    fn default() -> Self {
        Self {
            touched: Vec::with_capacity(MAX_SQ * GOMOKU_HISTORY_LEN),
            recent_epoch: [0; MAX_SQ],
            recent_rank: [0; MAX_SQ],
            color_plane_active: false,
            current_epoch: 0,
        }
    }
}

impl GomokuFeatureScratch {
    #[inline]
    fn next_recent_epoch(&mut self) -> u16 {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        if self.current_epoch == 0 {
            self.recent_epoch = [0; MAX_SQ];
            self.current_epoch = 1;
        }
        self.current_epoch
    }
}

// ─────────────────────────────────────────────
// § Gomoku 상태
// ─────────────────────────────────────────────

/// History depth for AlphaZero-style encoding (T=8 timesteps including current).
const GOMOKU_HISTORY_LEN: usize = 8;
const GOMOKU_HISTORY_MOVES: usize = GOMOKU_HISTORY_LEN - 1;

/// Reverse-info for `apply_move_in_place` (Phase 6.1, 2026-04-25). Compact
/// (~24 B) so the MCTS select descent can stack-allocate one per ply
/// instead of cloning the full ~1144 B `Gomoku` struct. The fields capture
/// only what changes during a single apply; everything else is restored by
/// inverting the deterministic mutation (XOR for Zobrist, decrement for
/// `move_count`, write-zero for the cell).
#[derive(Clone, Copy, Debug)]
pub struct GomokuUndo {
    pos: u16,
    prev_winner: i8,
    prev_recent_move_len: u8,
    prev_last_move_some: bool,
    prev_last_move_pos: u16,
    prev_recent_moves: [u16; GOMOKU_HISTORY_MOVES],
}

#[derive(Clone, Debug)]
pub struct Gomoku {
    pub size: usize,    // 보드 크기 (7/9/13/15)
    pub win_len: usize, // 승리 조건 (연속 몇 개: 4 or 5)
    // Fixed-size [i8; MAX_SQ=361] (was Vec<i8>) per the Apr-25 profile audit
    // Step 4 / P0-1. Eliminates the per-apply_move heap allocation that was
    // 61.6% of all heap allocations in the canonical benchmark. Only the
    // first `size * size` cells are valid game state; cells `size*size..MAX_SQ`
    // are always 0 (empty) and must NOT be iterated by any game-logic code —
    // every iterator over `self.board` clips with `[..self.size * self.size]`.
    // The `semantics_audit_*` tests at the bottom of this file pin that
    // invariant.
    board: [i8; MAX_SQ], // +1=black, -1=white, 0=empty  (row-major)
    occupied: [u16; MAX_SQ],
    current_player: i8, // +1=black, -1=white
    hash: u64,          // Zobrist 증분 해시
    move_count: u32,
    winner: i8, // 0=없음, +1=black 승, -1=white 승
    last_move: Option<usize>,
    /// Recent move history for AlphaZero-style feature reconstruction.
    /// We reconstruct historical boards on demand in encode_planes() to keep
    /// search-state clones cheap.
    recent_moves: [u16; GOMOKU_HISTORY_MOVES],
    recent_move_len: u8,
}

impl Gomoku {
    pub fn new(size: usize) -> Self {
        Self::new_with_win(size, 5)
    }

    /// 보드 크기와 승리 조건을 지정하여 생성
    /// e.g. Gomoku::new_with_win(7, 4) — 7×7 board, 4-in-a-row
    pub fn new_with_win(size: usize, win_len: usize) -> Self {
        assert!(size <= 19, "board size > 19 not supported");
        assert!(
            win_len >= 3 && win_len <= size,
            "win_len must be in [3, size]"
        );
        Gomoku {
            size,
            win_len,
            board: [0; MAX_SQ],
            occupied: [0; MAX_SQ],
            current_player: 1, // black 선수
            hash: 0,
            move_count: 0,
            winner: 0,
            last_move: None,
            recent_moves: [0; GOMOKU_HISTORY_MOVES],
            recent_move_len: 0,
        }
    }

    #[inline]
    fn push_recent_move(&mut self, mv: usize) {
        let mv = mv as u16;
        let len = self.recent_move_len as usize;
        if len < GOMOKU_HISTORY_MOVES {
            self.recent_moves[len] = mv;
            self.recent_move_len += 1;
            return;
        }
        self.recent_moves.copy_within(1..GOMOKU_HISTORY_MOVES, 0);
        self.recent_moves[GOMOKU_HISTORY_MOVES - 1] = mv;
    }

    /// JSON board (0/1/2 encoding) + player (1 or 2)로부터 상태 복원
    /// mcts_server용: 외부 프로토콜과 내부 엔진 사이의 변환
    pub fn from_board_12(size: usize, win_len: usize, board_12: &[i64], player_12: u8) -> Self {
        let mut g = Self::new_with_win(size, win_len);
        let mut mc = 0u32;
        for (i, &v) in board_12.iter().enumerate() {
            match v {
                1 => {
                    g.board[i] = 1;
                    g.occupied[mc as usize] = i as u16;
                    mc += 1;
                    g.hash ^= GOB.piece_hash(1, i);
                }
                2 => {
                    g.board[i] = -1;
                    g.occupied[mc as usize] = i as u16;
                    mc += 1;
                    g.hash ^= GOB.piece_hash(-1, i);
                }
                _ => {}
            }
        }
        g.move_count = mc;
        g.current_player = if player_12 == 1 { 1 } else { -1 };
        // side hash: black's turn = base, white's turn = xor side
        if g.current_player == -1 {
            g.hash ^= GOB.side;
        }
        // Check if already terminal (winner from last move). Bounded to
        // size*size — cells beyond that index are storage padding.
        let cells = g.size * g.size;
        for i in 0..cells {
            let v = g.board[i];
            if v != 0 && g.check_win_at(i, v) {
                g.winner = v;
                break;
            }
        }
        g
    }

    /// 내부 board를 0/1/2 encoding으로 변환 (JSON 응답용)
    #[cfg(test)]
    pub fn board_as_12(&self) -> Vec<i64> {
        let cells = self.size * self.size;
        let mut out = Vec::with_capacity(cells);
        for &v in &self.board[..cells] {
            out.push(match v {
                1 => 1,
                -1 => 2,
                _ => 0,
            });
        }
        out
    }

    /// 현재 보드 상태에서 pos에 player를 놓으면 승리하는지 체크 (착수 전 검사)
    pub fn wins_if(&self, pos: usize, player: i8) -> bool {
        if self.board[pos] != 0 {
            return false;
        }
        // [OPT] No clone — check_win_at only reads neighbors, and we count
        // the center cell as 1 implicitly. Neighbors already on the board.
        self.check_win_at_hypothetical(pos, player)
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.size + col
    }

    /// Check if placing `player` at `pos` would form win_len in a row.
    /// Does NOT require the piece to actually be on the board.
    ///
    /// Forwards to `check_win_lines`. Both `check_win_at` (post-placement
    /// invariant: `board[pos] == player`) and this fn (pre-placement) share
    /// the same body because neither reads `board[pos]` — the count is seeded
    /// at 1 representing the (real or hypothetical) piece, then walks outward.
    #[inline]
    fn check_win_at_hypothetical(&self, pos: usize, player: i8) -> bool {
        self.check_win_lines(pos, player)
    }

    /// Internal in-place mutator shared by `apply_move` (clone-then-mutate)
    /// and `apply_move_in_place` (Phase 6.1 hot-path entry point). Returns
    /// the `GomokuUndo` blob needed to reverse the mutation.
    #[inline]
    fn apply_move_mut_internal(&mut self, mv: usize) -> GomokuUndo {
        let player = self.current_player;
        let undo = GomokuUndo {
            pos: mv as u16,
            prev_winner: self.winner,
            prev_recent_move_len: self.recent_move_len,
            prev_last_move_some: self.last_move.is_some(),
            prev_last_move_pos: self.last_move.unwrap_or(0) as u16,
            prev_recent_moves: self.recent_moves,
        };

        // Zobrist 갱신
        self.hash ^= GOB.piece_hash(player, mv);
        self.hash ^= GOB.side;

        self.board[mv] = player;
        self.occupied[self.move_count as usize] = mv as u16;
        self.move_count += 1;
        self.last_move = Some(mv);
        self.push_recent_move(mv);

        // 승리 판정 (착수 위치에서만 검사)
        if self.check_win_at(mv, player) {
            self.winner = player;
        }

        self.current_player = -player;
        undo
    }

    /// pos에 player 기물을 놓았을 때 win_len목 달성 여부
    #[inline]
    fn check_win_at(&self, pos: usize, player: i8) -> bool {
        self.check_win_lines(pos, player)
    }

    /// Shared body for `check_win_at` (post-placement) and
    /// `check_win_at_hypothetical` (pre-placement). The count is seeded at 1
    /// for the piece at `pos` and the walks read only neighbouring cells —
    /// neither caller depends on `board[pos]`'s actual value.
    ///
    /// Algorithm: for each of the 4 line orientations, precompute the maximum
    /// number of valid steps in each polarity (forward / backward) from the
    /// board geometry, then walk via a flat board index with a single
    /// per-step compare. Eliminates the per-step (r, c) bounds check (4
    /// branches per step) plus the per-step `imul` for index reconstruction
    /// — the original loop's branch-miss share was 56.9 % of the program
    /// total per profiling (artifacts/profiling_20260428).
    #[inline]
    fn check_win_lines(&self, pos: usize, player: i8) -> bool {
        let sz = self.size;
        let row = pos / sz;
        let col = pos % sz;
        let target = self.win_len as usize;
        let board = &self.board[..];
        let pos_i = pos as isize;
        let szi = sz as isize;

        // For each direction (dr, dc):
        //   step       = dr * sz + dc                    (signed flat index delta)
        //   fwd_max    = furthest steps in (+dr, +dc) before edge
        //   bwd_max    = furthest steps in (-dr, -dc) before edge
        //   horizontal (0, 1):  step=+1,    fwd=sz-1-col,                   bwd=col
        //   vertical   (1, 0):  step=+sz,   fwd=sz-1-row,                   bwd=row
        //   diag SE-NW (1, 1):  step=+sz+1, fwd=min(sz-1-row, sz-1-col),    bwd=min(row, col)
        //   diag SW-NE (1,-1):  step=+sz-1, fwd=min(sz-1-row, col),         bwd=min(row, sz-1-col)
        let dirs: [(isize, usize, isize, usize); 4] = [
            (1, sz - 1 - col, -1, col),
            (szi, sz - 1 - row, -szi, row),
            (
                szi + 1,
                (sz - 1 - row).min(sz - 1 - col),
                -(szi + 1),
                row.min(col),
            ),
            (
                szi - 1,
                (sz - 1 - row).min(col),
                -(szi - 1),
                row.min(sz - 1 - col),
            ),
        ];

        // Flat-index walk. `max_steps` was derived from board geometry, so
        // every `idx` produced is in `[0, sz*sz)` — bounds-checked indexing
        // is safe and elided where the compiler can prove the invariant.
        #[inline(always)]
        fn walk(board: &[i8], start: isize, step: isize, max_steps: usize, player: i8) -> usize {
            let mut idx = start;
            let mut cnt = 0;
            for _ in 0..max_steps {
                idx += step;
                if board[idx as usize] != player {
                    break;
                }
                cnt += 1;
            }
            cnt
        }

        for &(fs, fmax, bs, bmax) in &dirs {
            let cnt =
                1 + walk(board, pos_i, fs, fmax, player) + walk(board, pos_i, bs, bmax, player);
            if cnt >= target {
                return true;
            }
        }
        false
    }
}

impl fmt::Display for Gomoku {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 좌표 헤더
        write!(f, "   ")?;
        for c in 0..self.size {
            write!(f, "{:2}", c)?;
        }
        writeln!(f)?;
        for r in 0..self.size {
            write!(f, "{:2} ", r)?;
            for c in 0..self.size {
                let sym = match self.board[self.idx(r, c)] {
                    1 => " X",
                    -1 => " O",
                    _ => " .",
                };
                write!(f, "{}", sym)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────
// § BQ++ Phase 5: tactical-sentinel accessors
// ─────────────────────────────────────────────

impl Gomoku {
    /// BQ++ Phase 5: dimension accessor for the tactical sentinel.
    /// Returns the board size (e.g. 7 for 7×7, 15 for 15×15).
    pub fn size_dim(&self) -> usize {
        self.size
    }

    /// BQ++ Phase 5: win-condition accessor for the tactical sentinel.
    /// Returns the number of consecutive stones required to win
    /// (4 for Gomoku 7×7 in this codebase, 5 for standard 15×15).
    pub fn win_len_dim(&self) -> usize {
        self.win_len
    }

    /// BQ++ Phase 5: current-player sign accessor (+1 black / −1 white).
    pub fn current_player_sign(&self) -> i8 {
        self.current_player
    }

    /// BQ++ Phase 5: per-cell occupancy check used by the tactical
    /// sentinel. Returns true iff `pos` is in-bounds AND the cell is
    /// empty (board[pos] == 0).
    pub fn cell_is_empty(&self, pos: usize) -> bool {
        if pos >= self.size * self.size {
            return false;
        }
        self.board[pos] == 0
    }

    /// BQ++ Phase 5: per-cell player check. Returns true iff `pos` is
    /// in-bounds AND the cell holds a stone of the given player.
    pub fn cell_is_player(&self, pos: usize, player: i8) -> bool {
        if pos >= self.size * self.size {
            return false;
        }
        self.board[pos] == player
    }
}

// ─────────────────────────────────────────────
// § GameState 구현
// ─────────────────────────────────────────────

impl GameState for Gomoku {
    type Move = usize; // flat index (0..size*size)
    type Undo = GomokuUndo;

    fn initial() -> Self {
        Gomoku::new(9)
    } // 기본 9×9

    fn current_player(&self) -> i8 {
        self.current_player
    }

    fn legal_moves(&self) -> Vec<usize> {
        if self.is_terminal() {
            return vec![];
        }
        let cells = self.size * self.size;
        let mut moves = Vec::with_capacity(cells - self.move_count as usize);
        for (i, &v) in self.board[..cells].iter().enumerate() {
            if v == 0 {
                moves.push(i);
            }
        }
        moves
    }

    fn uniform_eval(&self, value: f32) -> EvalResult<usize> {
        if self.is_terminal() {
            return EvalResult {
                policy: Vec::new(),
                value,
            };
        }
        let count = self.legal_move_count();
        let prior = if count > 0 { 1.0 / count as f32 } else { 0.0 };
        let cells = self.size * self.size;
        let mut policy = Vec::with_capacity(count);
        for (i, &v) in self.board[..cells].iter().enumerate() {
            if v == 0 {
                policy.push((i, prior));
            }
        }
        EvalResult { policy, value }
    }

    fn apply_move(&self, mv: usize) -> Self {
        let mut next = self.clone();
        // Phase 6.1: share the mutation logic with `apply_move_in_place`. The
        // returned undo is dropped here — `apply_move`'s contract is to
        // produce the next state without mutating `self`.
        let _undo = next.apply_move_mut_internal(mv);
        next
    }

    fn apply_move_in_place(&mut self, mv: usize) -> GomokuUndo {
        self.apply_move_mut_internal(mv)
    }

    fn apply_move_in_place_no_undo(&mut self, mv: usize) {
        let _ = self.apply_move_mut_internal(mv);
    }

    fn undo_move(&mut self, undo: GomokuUndo) {
        // current_player was flipped at the end of apply; the player who just
        // moved is the *new* opponent of the current side.
        let player = -self.current_player;
        let pos = undo.pos as usize;

        // Reverse the cell write and decrement move_count. occupied[move_count-1]
        // becomes garbage by convention (only [..move_count] is valid).
        self.board[pos] = 0;
        self.move_count -= 1;

        // Reverse Zobrist (XOR is involutive).
        self.hash ^= GOB.piece_hash(player, pos);
        self.hash ^= GOB.side;

        // Restore winner / recent-history fields from the undo blob.
        self.winner = undo.prev_winner;
        self.recent_move_len = undo.prev_recent_move_len;
        self.recent_moves = undo.prev_recent_moves;
        self.last_move = if undo.prev_last_move_some {
            Some(undo.prev_last_move_pos as usize)
        } else {
            None
        };

        self.current_player = player;
    }

    fn is_terminal(&self) -> bool {
        self.winner != 0 || self.move_count as usize >= self.size * self.size
    }

    /// negamax convention: current_player 관점
    /// 방금 상대가 이겼으면 current_player가 진 것 → -1.0
    fn outcome(&self) -> f32 {
        if self.winner == self.current_player {
            1.0
        }
        // 이 상태는 정상적으로 나타나지 않음
        else if self.winner != 0 {
            -1.0
        }
        // 상대(직전 착수자)가 이겼음
        else {
            0.0
        } // 무승부
    }

    fn hash(&self) -> u64 {
        self.hash
    }

    fn num_actions(&self) -> usize {
        self.size * self.size
    }

    fn move_to_idx(&self, mv: usize) -> usize {
        mv
    }

    fn idx_to_move(&self, idx: usize) -> Option<usize> {
        if idx < self.size * self.size {
            Some(idx)
        } else {
            None
        }
    }

    /// NN 입력 feature planes (AlphaZero-style: [17, H, W])
    ///   planes 0-1:   현재(t=0) 내 돌, 상대 돌
    ///   planes 2-3:   t=1 (1수 전) 내 돌, 상대 돌
    ///   ...
    ///   planes 14-15: t=7 (7수 전) 내 돌, 상대 돌
    ///   plane 16:     현재 플레이어 색상 (black=1, white=0)
    fn encode_planes_into(&self, out: &mut Vec<f32>) {
        let n = self.size * self.size;
        let total_planes = GOMOKU_HISTORY_LEN * 2 + 1; // 17
        let cp = self.current_player;
        GOMOKU_FEATURE_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let total = total_planes * n;
            let color_base = (total_planes - 1) * n;
            if out.len() != total {
                out.clear();
                out.resize(total, 0.0);
                scratch.touched.clear();
                scratch.color_plane_active = false;
            } else {
                let touched_len =
                    scratch.touched.len() + if scratch.color_plane_active { n } else { 0 };
                if touched_len * 2 >= total {
                    out.fill(0.0);
                    scratch.touched.clear();
                    scratch.color_plane_active = false;
                } else {
                    for &idx in &scratch.touched {
                        out[idx as usize] = 0.0;
                    }
                    scratch.touched.clear();
                    if scratch.color_plane_active {
                        out[color_base..color_base + n].fill(0.0);
                        scratch.color_plane_active = false;
                    }
                }
            }

            let recent_epoch = scratch.next_recent_epoch();
            for (rank, &mv) in self.recent_moves[..self.recent_move_len as usize]
                .iter()
                .rev()
                .enumerate()
            {
                let pos = mv as usize;
                scratch.recent_epoch[pos] = recent_epoch;
                scratch.recent_rank[pos] = (rank + 1) as u8;
            }

            for &pos in &self.occupied[..self.move_count as usize] {
                let i = pos as usize;
                let v = self.board[i];
                let plane_offset = if v == cp { 0 } else { n };
                let active_steps = match (scratch.recent_epoch[i] == recent_epoch)
                    .then_some(scratch.recent_rank[i])
                {
                    Some(rank) => rank as usize,
                    None => GOMOKU_HISTORY_LEN,
                };
                for t in 0..active_steps {
                    let idx = t * 2 * n + plane_offset + i;
                    out[idx] = 1.0;
                    scratch.touched.push(idx as u16);
                }
            }

            if cp == 1 {
                out[color_base..color_base + n].fill(1.0);
                scratch.color_plane_active = true;
            }
        });
    }

    fn board_state_record(&self) -> Vec<i64> {
        let cells = self.size * self.size;
        self.board[..cells]
            .iter()
            .map(|&v| match v {
                1 => 1,
                -1 => 2,
                _ => 0,
            })
            .collect()
    }

    /// [OPT] O(win_len) win check — no state clone, no hash update.
    fn is_winning_move(&self, mv: usize) -> bool {
        self.wins_if(mv, self.current_player)
    }

    /// [OPT] Random legal move without Vec allocation.
    /// Scans board for the (rand_idx % n_empty)-th empty cell.
    fn random_legal_move(&self, rand_idx: usize) -> Option<usize> {
        if self.is_terminal() {
            return None;
        }
        let n_empty = self.size * self.size - self.move_count as usize;
        if n_empty == 0 {
            return None;
        }
        let target = rand_idx % n_empty;
        let mut count = 0usize;
        let cells = self.size * self.size;
        for (i, &v) in self.board[..cells].iter().enumerate() {
            if v == 0 {
                if count == target {
                    return Some(i);
                }
                count += 1;
            }
        }
        None // unreachable if n_empty > 0
    }

    /// [OPT] Count empty cells without Vec.
    fn legal_move_count(&self) -> usize {
        if self.is_terminal() {
            return 0;
        }
        self.size * self.size - self.move_count as usize
    }
}

// ─────────────────────────────────────────────
// § 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_planes_reference(state: &Gomoku) -> Vec<f32> {
        let n = state.size * state.size;
        let total_planes = GOMOKU_HISTORY_LEN * 2 + 1;
        let mut out = vec![0.0; total_planes * n];
        let cp = state.current_player;
        let mut recent_rank = [0u8; MAX_SQ];
        for (rank, &mv) in state.recent_moves[..state.recent_move_len as usize]
            .iter()
            .rev()
            .enumerate()
        {
            recent_rank[mv as usize] = (rank + 1) as u8;
        }
        for (i, &v) in state.board[..n].iter().enumerate() {
            if v == 0 {
                continue;
            }
            let plane_offset = if v == cp { 0 } else { n };
            let active_steps = match recent_rank[i] {
                0 => GOMOKU_HISTORY_LEN,
                rank => rank as usize,
            };
            for t in 0..active_steps {
                out[t * 2 * n + plane_offset + i] = 1.0;
            }
        }
        if cp == 1 {
            let color_base = (total_planes - 1) * n;
            out[color_base..color_base + n].fill(1.0);
        }
        out
    }

    fn assert_occupied_matches_board(state: &Gomoku) {
        let occupied_len = state.move_count as usize;
        let board_len = state.size * state.size;
        let board_count = state.board[..board_len].iter().filter(|&&v| v != 0).count();
        assert_eq!(
            occupied_len, board_count,
            "occupied prefix length must match stone count"
        );

        let mut seen = [false; MAX_SQ];
        for &pos in &state.occupied[..occupied_len] {
            let pos = pos as usize;
            assert!(
                pos < board_len,
                "occupied position must stay in board bounds"
            );
            assert_ne!(state.board[pos], 0, "occupied entries must point to stones");
            assert!(!seen[pos], "occupied entries must not duplicate positions");
            seen[pos] = true;
        }
        for (idx, &cell) in state.board[..board_len].iter().enumerate() {
            assert_eq!(
                seen[idx],
                cell != 0,
                "occupied prefix must cover exactly all stones"
            );
        }
    }

    fn make_state(size: usize, moves: &[(usize, usize)]) -> Gomoku {
        // moves: (row, col) 쌍, 교대로 착수
        let mut s = Gomoku::new(size);
        for &(r, c) in moves {
            s = s.apply_move(r * size + c);
        }
        s
    }

    #[test]
    fn test_initial_state() {
        let s = Gomoku::new(9);
        assert_eq!(s.legal_moves().len(), 81);
        assert!(!s.is_terminal());
    }

    #[test]
    fn test_horizontal_win() {
        // Black: (0,0)(0,1)(0,2)(0,3)(0,4) — 5 in a row
        // White: (1,0)(1,1)(1,2)(1,3)
        let s = make_state(
            9,
            &[
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
            ],
        );
        assert!(s.is_terminal());
        assert_eq!(s.current_player(), -1); // white's turn (black just won)
        assert_eq!(s.outcome(), -1.0); // white perspective: lost
    }

    #[test]
    fn test_diagonal_win() {
        // Black: (0,0)(1,1)(2,2)(3,3)(4,4)
        let s = make_state(
            9,
            &[
                (0, 0),
                (0, 1),
                (1, 1),
                (0, 2),
                (2, 2),
                (0, 3),
                (3, 3),
                (0, 4),
                (4, 4),
            ],
        );
        assert!(s.is_terminal());
        assert_eq!(s.outcome(), -1.0);
    }

    #[test]
    fn test_no_win_four_in_row() {
        // Black: 4 in a row (not 5) — should not be terminal
        let s = make_state(9, &[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]);
        assert!(!s.is_terminal());
    }

    #[test]
    fn test_hash_transposition() {
        let size = 9;
        // Same board via different move orders
        let mut a = Gomoku::new(size);
        for mv in [0 * size + 0, 1 * size + 0, 0 * size + 1, 1 * size + 1] {
            a = a.apply_move(mv);
        }

        let mut b = Gomoku::new(size);
        for mv in [0 * size + 1, 1 * size + 1, 0 * size + 0, 1 * size + 0] {
            b = b.apply_move(mv);
        }

        assert_eq!(a.board, b.board);
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn test_apply_move_pure() {
        let s = Gomoku::new(9);
        let before_board = s.board.clone();
        let s2 = s.apply_move(40); // center of 9×9
        assert_eq!(s.board, before_board); // 원본 불변
        assert_eq!(s2.board[40], 1);
    }

    #[test]
    fn test_encode_planes() {
        let s = make_state(9, &[(4, 4), (0, 0)]);
        let planes = s.encode_planes();
        let n = 9 * 9;
        assert_eq!(planes.len(), 17 * n);
        // current_player = black (+1) after 2 moves
        assert_eq!(planes[4 * 9 + 4], 1.0); // plane 0: black's piece at (4,4)
        assert_eq!(planes[n + 0], 1.0); // plane 1: white's piece at (0,0)
    }

    #[test]
    fn test_encode_planes_matches_reference_and_occupied_consistency() {
        let states = [
            Gomoku::new(9),
            make_state(9, &[(4, 4), (0, 0), (4, 5), (0, 1), (4, 6)]),
            make_state_7x4(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]),
            make_state_15(&[(7, 7), (0, 0), (7, 8), (0, 1), (7, 9), (0, 2), (7, 10)]),
        ];

        for state in &states {
            assert_occupied_matches_board(state);
            let expected = encode_planes_reference(state);
            assert_eq!(state.encode_planes(), expected);
            let mut scratch = Vec::new();
            state.encode_planes_into(&mut scratch);
            assert_eq!(scratch, expected);
        }

        let board = vec![
            1, 0, 2, 0, 0, 0, 0, 0, 0, //
            0, 1, 0, 2, 0, 0, 0, 0, 0, //
            0, 0, 1, 0, 2, 0, 0, 0, 0, //
            0, 0, 0, 1, 0, 2, 0, 0, 0, //
            0, 0, 0, 0, 1, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let rebuilt = Gomoku::from_board_12(9, 5, &board, 2);
        assert_occupied_matches_board(&rebuilt);
        let expected = encode_planes_reference(&rebuilt);
        let mut scratch = Vec::new();
        rebuilt.encode_planes_into(&mut scratch);
        assert_eq!(scratch, expected);
    }

    // ── win_len=4 tests (7×7 server compatibility) ──

    fn make_state_7x4(moves: &[(usize, usize)]) -> Gomoku {
        let mut s = Gomoku::new_with_win(7, 4);
        for &(r, c) in moves {
            s = s.apply_move(r * 7 + c);
        }
        s
    }

    #[test]
    fn test_7x4_four_in_row_wins() {
        // Black: (0,0)(0,1)(0,2)(0,3) — 4 in a row on 7×7 w/ win_len=4
        // White: (1,0)(1,1)(1,2)
        let s = make_state_7x4(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]);
        assert!(s.is_terminal(), "4-in-row should win with win_len=4");
    }

    #[test]
    fn test_7x4_three_no_win() {
        // Black: 3 in a row — not enough for win_len=4
        let s = make_state_7x4(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]);
        assert!(!s.is_terminal(), "3-in-row should NOT win with win_len=4");
    }

    #[test]
    fn test_9x5_four_no_win() {
        // 9×9 standard: 4 in a row should NOT win (need 5)
        let s = make_state(9, &[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]);
        assert!(!s.is_terminal(), "4-in-row should NOT win on standard 9×9");
    }

    #[test]
    fn test_from_board_12() {
        // 7×7 board with 1=black, 2=white, 0=empty
        let mut board = vec![0i64; 49];
        board[0] = 1; // black at (0,0)
        board[1] = 1; // black at (0,1)
        board[7] = 2; // white at (1,0)
        board[8] = 2; // white at (1,1)
        let s = Gomoku::from_board_12(7, 4, &board, 1);
        assert_eq!(s.current_player(), 1); // black's turn
        assert_eq!(s.board[0], 1); // internal: +1
        assert_eq!(s.board[7], -1); // internal: -1
        assert!(!s.is_terminal());
        assert_eq!(s.move_count, 4);
    }

    #[test]
    fn test_wins_if() {
        // Black has 3 in a row at (0,0)(0,1)(0,2) on 7×7 w/ win_len=4
        let s = make_state_7x4(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]);
        // Black placing at (0,3) should win
        assert!(
            s.wins_if(0 * 7 + 3, 1),
            "black should win by placing at (0,3)"
        );
        // White placing at (0,3) should NOT win (only 1 white piece at row 1)
        assert!(!s.wins_if(0 * 7 + 3, -1));
    }

    // ── PR-4A: Gomoku 15×15 tests ──────────────────────

    fn make_state_15(moves: &[(usize, usize)]) -> Gomoku {
        let mut s = Gomoku::new(15); // 15×15, win_len=5 (standard)
        for &(r, c) in moves {
            s = s.apply_move(r * 15 + c);
        }
        s
    }

    // TEST-4A-1: legal_moves on 15×15
    #[test]
    fn test_4a1_gomoku15_legal_moves() {
        let s = Gomoku::new(15);
        assert_eq!(s.legal_moves().len(), 225, "15×15 = 225 cells");
        assert!(!s.is_terminal());
        assert_eq!(s.size, 15);
        assert_eq!(s.win_len, 5);
    }

    #[test]
    fn test_4a1_gomoku15_after_moves() {
        let s = make_state_15(&[(7, 7), (0, 0), (7, 8)]);
        assert_eq!(s.legal_moves().len(), 222); // 225 - 3
        assert_eq!(s.current_player(), -1); // white's turn (3 moves = black just played)
    }

    // TEST-4A-2: win detection all directions on 15×15
    #[test]
    fn test_4a2_gomoku15_horizontal_win() {
        // Black: (7,3)(7,4)(7,5)(7,6)(7,7) = 5 in a row horizontal
        let s = make_state_15(&[
            (7, 3),
            (0, 0),
            (7, 4),
            (0, 1),
            (7, 5),
            (0, 2),
            (7, 6),
            (0, 3),
            (7, 7),
        ]);
        assert!(s.is_terminal(), "5-in-row horizontal should win on 15×15");
    }

    #[test]
    fn test_4a2_gomoku15_vertical_win() {
        // Black: (3,7)(4,7)(5,7)(6,7)(7,7) = 5 in a row vertical
        let s = make_state_15(&[
            (3, 7),
            (0, 0),
            (4, 7),
            (0, 1),
            (5, 7),
            (0, 2),
            (6, 7),
            (0, 3),
            (7, 7),
        ]);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_4a2_gomoku15_diagonal_win() {
        // Black: (3,3)(4,4)(5,5)(6,6)(7,7) = diagonal
        let s = make_state_15(&[
            (3, 3),
            (0, 0),
            (4, 4),
            (0, 1),
            (5, 5),
            (0, 2),
            (6, 6),
            (0, 3),
            (7, 7),
        ]);
        assert!(s.is_terminal());
    }

    #[test]
    fn test_4a2_gomoku15_four_no_win() {
        // 4 in a row should NOT win (need 5) on 15×15
        let s = make_state_15(&[(7, 3), (0, 0), (7, 4), (0, 1), (7, 5), (0, 2), (7, 6)]);
        assert!(!s.is_terminal(), "4-in-row should NOT win on 15×15");
    }

    // TEST-4A-3: Rust MctsEngine integration with 15×15
    #[test]
    fn test_4a3_gomoku15_engine_integration() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::search::FixedIterations;
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let state = Gomoku::new(15);
        let eval: Arc<dyn crate::game::Evaluator<Gomoku> + Send + Sync> = Arc::new(UniformEval);
        let config = MctsConfig::evaluation(2.0);
        let engine = MctsEngine::new(state, eval, config);
        engine.run(&mut FixedIterations::new(100));

        let best = engine.best_move();
        assert!(best.is_some(), "should find a move on 15×15");
        let mv = best.unwrap();
        assert!(mv < 225, "move should be valid index");
    }

    // TEST-4A: 15×15 hash/transposition
    #[test]
    fn test_4a_gomoku15_hash() {
        let mut a = Gomoku::new(15);
        for mv in [7 * 15 + 7, 0, 7 * 15 + 8, 1] {
            a = a.apply_move(mv);
        }

        let mut b = Gomoku::new(15);
        for mv in [7 * 15 + 8, 1, 7 * 15 + 7, 0] {
            b = b.apply_move(mv);
        }

        assert_eq!(
            a.hash(),
            b.hash(),
            "same board via different order should have same hash"
        );
    }

    // ── Semantics audit (Apr-25 profile-audit Step 4 / P0-1) ──────────────
    //
    // These tests pin properties that must be preserved under any board
    // representation change (Vec<i8> → fixed-size array). They check that
    // every iteration over `self.board` is correctly bounded to the
    // game's `size * size` cells, not the underlying storage capacity.

    /// `legal_moves()` must return exactly `size*size - move_count` cells
    /// for every supported board size. If an internal iteration is
    /// unbounded, this assertion fails immediately.
    #[test]
    fn semantics_audit_legal_moves_bounded_to_size() {
        for size in [7usize, 9, 13, 15] {
            let s = Gomoku::new(size);
            let want = size * size;
            let got = s.legal_moves().len();
            assert_eq!(got, want, "legal_moves on empty {}x{}", size, size);

            let s2 = if size == 7 {
                make_state_7x4(&[(0, 0), (1, 1), (0, 1)])
            } else {
                let mut s = Gomoku::new(size);
                for &(r, c) in &[(0usize, 0), (1, 1), (0, 1)] {
                    s = s.apply_move(r * size + c);
                }
                s
            };
            assert_eq!(s2.legal_moves().len(), size * size - 3);
        }
    }

    /// `board_state_record()` returns exactly `size*size` entries (used by
    /// JSON export and replay snapshots). An unbounded iterator would emit
    /// `MAX_SQ` entries on smaller boards, breaking the contract.
    #[test]
    fn semantics_audit_board_state_record_bounded_to_size() {
        use crate::game::GameState;
        for size in [7usize, 9, 13, 15] {
            let s = Gomoku::new(size);
            assert_eq!(s.board_state_record().len(), size * size);
        }
    }

    /// `random_legal_move()` must only ever return indices `< size*size`.
    /// Since unfilled storage cells beyond `size*size` will read as 0 (empty)
    /// under any representation, an unbounded scan would happily return
    /// out-of-board indices. We sample 100 RNG seeds for each board size.
    #[test]
    fn semantics_audit_random_legal_move_inside_board() {
        use crate::game::GameState;
        for size in [7usize, 9, 13, 15] {
            let s = Gomoku::new(size);
            let bound = size * size;
            for seed in 0..100 {
                let mv = s.random_legal_move(seed * 7919).unwrap();
                assert!(
                    mv < bound,
                    "size={} seed={} move={} bound={}",
                    size,
                    seed,
                    mv,
                    bound
                );
            }
        }
    }

    /// Phase 6.1: random sequences of (apply_in_place, undo) leave board
    /// state and hash identical to the pre-apply state. Also asserts that
    /// `apply_move_in_place(mv)` produces the same post-state as
    /// `apply_move(mv)` so the make-unmake path is fully equivalent.
    #[test]
    fn semantics_audit_apply_in_place_undo_equivalence() {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        for &size in &[7usize, 9, 13, 15] {
            for seed in 0..16u64 {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(31 + size as u64));
                let mut s = Gomoku::new(size);
                for _ply in 0..8 {
                    if s.is_terminal() {
                        break;
                    }
                    let legal = s.legal_moves();
                    if legal.is_empty() {
                        break;
                    }
                    let mv = legal[rng.gen::<usize>() % legal.len()];

                    // 1) apply_move (clone) and apply_in_place must yield equal post-states.
                    let pure = s.apply_move(mv);
                    let mut mutated = s.clone();
                    let undo = mutated.apply_move_in_place(mv);
                    assert_eq!(pure.board, mutated.board);
                    assert_eq!(pure.occupied, mutated.occupied);
                    assert_eq!(pure.current_player, mutated.current_player);
                    assert_eq!(pure.hash(), mutated.hash());
                    assert_eq!(pure.move_count, mutated.move_count);
                    assert_eq!(pure.winner, mutated.winner);
                    assert_eq!(pure.last_move, mutated.last_move);
                    assert_eq!(pure.recent_move_len, mutated.recent_move_len);
                    assert_eq!(pure.recent_moves, mutated.recent_moves);

                    // 2) undo restores the pre-apply state exactly.
                    let pre_board = s.board;
                    let pre_occupied = s.occupied;
                    let pre_hash = s.hash();
                    let pre_player = s.current_player;
                    let pre_winner = s.winner;
                    let pre_last_move = s.last_move;
                    let pre_recent_len = s.recent_move_len;
                    let pre_recent_moves = s.recent_moves;
                    let pre_move_count = s.move_count;

                    mutated.undo_move(undo);
                    assert_eq!(mutated.board, pre_board);
                    // Only [..move_count] of `occupied` is semantically valid;
                    // the slot at index `move_count` may contain a stale value
                    // from before undo (this is by-design and is invariant
                    // with how `apply_move` writes occupied[move_count]).
                    assert_eq!(
                        mutated.occupied[..pre_move_count as usize],
                        pre_occupied[..pre_move_count as usize]
                    );
                    assert_eq!(mutated.hash(), pre_hash);
                    assert_eq!(mutated.current_player, pre_player);
                    assert_eq!(mutated.winner, pre_winner);
                    assert_eq!(mutated.last_move, pre_last_move);
                    assert_eq!(mutated.recent_move_len, pre_recent_len);
                    assert_eq!(mutated.recent_moves, pre_recent_moves);
                    assert_eq!(mutated.move_count, pre_move_count);

                    // Advance the outer state via the pure path.
                    s = pure;
                }
            }
        }
    }

    /// from_board_12 must check terminal state only over valid cells, not
    /// the underlying storage. We construct a non-terminal small-board
    /// state and confirm winner stays at 0.
    #[test]
    fn semantics_audit_from_board_12_no_phantom_winner() {
        let board: Vec<i64> = (0..49).map(|i| if i == 0 { 1 } else { 0 }).collect();
        let s = Gomoku::from_board_12(7, 4, &board, 2);
        assert_eq!(s.winner, 0);
        assert!(!s.is_terminal());
        assert_eq!(s.move_count, 1);
    }
}

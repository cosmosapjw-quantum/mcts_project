//! Gomoku 15×15 — Freestyle / Standard / Caro / Renju / Omok 통합 구현
//!
//! 핵심 설계:
//!   - [u8; 225] 고정 배열: Copy semantics, ~1.2KB bitwise copy
//!   - apply_move(&self, mv) -> Self: pure function (parallel MCTS safe)
//!   - GomokuVariant:
//!       Freestyle (>=5 wins),
//!       Standard (exactly 5 wins),
//!       Caro (blocked-both-ends 5 and exact 6 do not win),
//!       Omok (Korean omok preserved),
//!       Renju (exactly 5, double-three+double-four+overline forbidden)
//!   - 금수 판정: 착수점 중심 4방향 스캔 (O(1), 전체 보드 스캔 X)
//!   - 5목 완성 수는 금수 면제 (한국 오목 공식 규칙)
//!
//! 좌표계: flat index pos = row * 15 + col, 0-indexed
//! 플레이어: +1 = Black(first), -1 = White(second), negamax convention
//! board: 0 = empty, 1 = black, 2 = white

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::fmt;
use std::sync::LazyLock;

use crate::game::{EvalResult, GameState};

// ─────────────────────────────────────────────
// § Constants
// ─────────────────────────────────────────────

const N: usize = 15;
const N2: usize = N * N; // 225

/// 4 directions: horizontal, vertical, diagonal-DR, diagonal-UR
const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

// ─────────────────────────────────────────────
// § Zobrist 테이블
// ─────────────────────────────────────────────

struct Zob15 {
    piece: [[u64; N2]; 2], // [0=black, 1=white][square]
    side: u64,
}

impl Zob15 {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut piece = [[0u64; N2]; 2];
        for p in 0..2 {
            for sq in 0..N2 {
                piece[p][sq] = rng.gen();
            }
        }
        Zob15 {
            piece,
            side: rng.gen(),
        }
    }

    #[inline]
    fn piece_hash(&self, player: i8, sq: usize) -> u64 {
        let pi = if player > 0 { 0 } else { 1 };
        self.piece[pi][sq]
    }
}

unsafe impl Sync for Zob15 {}

static ZOB15: LazyLock<Zob15> = LazyLock::new(|| Zob15::new(0xA0B0_C015_DEAD_BEEF));

thread_local! {
    static GOMOKU15_FEATURE_SCRATCH: RefCell<Gomoku15FeatureScratch> =
        RefCell::new(Gomoku15FeatureScratch::default());
}

struct Gomoku15FeatureScratch {
    touched: Vec<u16>,
    recent_epoch: [u16; N2],
    recent_rank: [u8; N2],
    color_plane_active: bool,
    current_epoch: u16,
}

impl Default for Gomoku15FeatureScratch {
    fn default() -> Self {
        Self {
            touched: Vec::with_capacity(N2 * GOMOKU15_HISTORY_LEN),
            recent_epoch: [0; N2],
            recent_rank: [0; N2],
            color_plane_active: false,
            current_epoch: 0,
        }
    }
}

impl Gomoku15FeatureScratch {
    #[inline]
    fn next_recent_epoch(&mut self) -> u16 {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        if self.current_epoch == 0 {
            self.recent_epoch = [0; N2];
            self.current_epoch = 1;
        }
        self.current_epoch
    }
}

// ─────────────────────────────────────────────
// § GomokuVariant
// ─────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GomokuVariant {
    /// Gomocup freestyle: 5개 이상 일렬이면 승리 (양측 동일)
    Freestyle,
    /// Gomocup standard: 정확히 5만 승리, 장목은 승리 아님
    Standard,
    /// 한국 오목: Black에게 장목 금지 + 쌍삼 금지, 정확히 5만 승리
    Omok,
    /// 렌주: Black에게 장목 + 쌍삼 + 쌍사 금지, 정확히 5만 승리
    Renju,
    /// Gomocup caro: 양끝 막힌 5와 정확히 6은 승리 아님
    Caro,
}

// ─────────────────────────────────────────────
// § Gomoku15 상태
// ─────────────────────────────────────────────

/// History depth for AlphaZero-style encoding (T=8 timesteps including current).
const GOMOKU15_HISTORY_LEN: usize = 8;
const GOMOKU15_HISTORY_MOVES: usize = GOMOKU15_HISTORY_LEN - 1;

#[derive(Clone, Copy, Debug)]
pub struct Gomoku15Undo {
    pos: u16,
    prev_winner: i8,
    prev_last_mv: u16,
    prev_recent_move_len: u8,
    prev_recent_moves: [u16; GOMOKU15_HISTORY_MOVES],
    prev_near_black: [u64; 4],
}

#[derive(Clone, Debug)]
pub struct Gomoku15 {
    board: [u8; N2], // 0=empty, 1=black, 2=white
    occupied: [u16; N2],
    hash: u64,
    side: i8, // +1=black, -1=white
    moves: u16,
    winner: i8,   // 0=none, +1=black, -1=white
    last_mv: u16, // 0..224 or 0xFFFF=none
    pub rules: GomokuVariant,
    /// Bitmask: bit i set ⇔ cell i is within distance 4 (in any direction)
    /// of at least one black stone. Used to skip forbidden checks for far cells.
    near_black: [u64; 4], // ceil(225/64) = 4
    /// Recent move history for AlphaZero-style feature reconstruction.
    recent_moves: [u16; GOMOKU15_HISTORY_MOVES],
    recent_move_len: u8,
}

impl Gomoku15 {
    #[inline]
    fn move_limit_for_variant(variant: GomokuVariant) -> usize {
        match variant {
            GomokuVariant::Renju => 200,
            _ => N2,
        }
    }

    #[inline]
    fn move_limit(&self) -> usize {
        Self::move_limit_for_variant(self.rules)
    }

    pub fn new(variant: GomokuVariant) -> Self {
        Gomoku15 {
            board: [0u8; N2],
            occupied: [0; N2],
            hash: 0,
            side: 1, // black first
            moves: 0,
            winner: 0,
            last_mv: 0xFFFF,
            rules: variant,
            near_black: [0u64; 4],
            recent_moves: [0; GOMOKU15_HISTORY_MOVES],
            recent_move_len: 0,
        }
    }

    #[inline]
    fn push_recent_move(&mut self, mv: u16) {
        let len = self.recent_move_len as usize;
        if len < GOMOKU15_HISTORY_MOVES {
            self.recent_moves[len] = mv;
            self.recent_move_len += 1;
            return;
        }
        self.recent_moves.copy_within(1..GOMOKU15_HISTORY_MOVES, 0);
        self.recent_moves[GOMOKU15_HISTORY_MOVES - 1] = mv;
    }

    #[inline]
    fn apply_move_mut_internal(&mut self, mv: u16) -> Gomoku15Undo {
        let pos = mv as usize;
        let side = self.side;
        let val = Self::side_to_val(side);
        let undo = Gomoku15Undo {
            pos: mv,
            prev_winner: self.winner,
            prev_last_mv: self.last_mv,
            prev_recent_move_len: self.recent_move_len,
            prev_recent_moves: self.recent_moves,
            prev_near_black: self.near_black,
        };

        self.board[pos] = val;
        self.occupied[self.moves as usize] = mv;

        if val == 1 {
            self.nb_mark_around(pos);
        }

        {
            let z = &*ZOB15;
            self.hash ^= z.piece_hash(side, pos);
            self.hash ^= z.side;
        }

        if self.check_win_at(pos) {
            self.winner = side;
        }

        self.side = -side;
        self.moves += 1;
        self.last_mv = mv;
        self.push_recent_move(mv);
        undo
    }

    #[cfg(test)]
    pub fn standard() -> Self {
        Self::new(GomokuVariant::Standard)
    }
    pub fn freestyle() -> Self {
        Self::new(GomokuVariant::Freestyle)
    }
    #[cfg(test)]
    pub fn omok() -> Self {
        Self::new(GomokuVariant::Omok)
    }
    #[cfg(test)]
    pub fn renju() -> Self {
        Self::new(GomokuVariant::Renju)
    }
    #[cfg(test)]
    pub fn caro() -> Self {
        Self::new(GomokuVariant::Caro)
    }

    /// Reconstruct state from board array.
    /// board[i]: +1=black, -1=white, 0=empty. player: +1 or -1.
    pub fn from_board(board: &[i8], player: i8, variant: GomokuVariant) -> Self {
        let mut g = Self::new(variant);
        g.side = player;
        let mut move_count = 0u16;
        let mut last = 0xFFFFu16;
        for i in 0..N2.min(board.len()) {
            if board[i] == 1 {
                g.board[i] = 1;
                g.occupied[move_count as usize] = i as u16;
                move_count += 1;
                last = i as u16;
                g.nb_mark_around(i);
            } else if board[i] == -1 || board[i] == 2 {
                g.board[i] = 2;
                g.occupied[move_count as usize] = i as u16;
                move_count += 1;
                last = i as u16;
            }
        }
        g.moves = move_count;
        g.last_mv = last;
        // Recompute hash
        {
            let z = &*ZOB15;
            g.hash = 0;
            for i in 0..N2 {
                if g.board[i] == 1 {
                    g.hash ^= z.piece[0][i];
                } else if g.board[i] == 2 {
                    g.hash ^= z.piece[1][i];
                }
            }
            if g.side < 0 {
                g.hash ^= z.side;
            }
        }
        // Check winner
        for i in 0..N2 {
            if g.board[i] != 0 && g.check_win_at(i) {
                g.winner = if g.board[i] == 1 { 1 } else { -1 };
                break;
            }
        }
        g
    }

    #[inline]
    fn row(pos: usize) -> i32 {
        (pos / N) as i32
    }
    #[inline]
    fn col(pos: usize) -> i32 {
        (pos % N) as i32
    }
    #[inline]
    fn to_pos(r: i32, c: i32) -> usize {
        (r as usize) * N + c as usize
    }
    #[inline]
    fn in_bounds(r: i32, c: i32) -> bool {
        r >= 0 && r < N as i32 && c >= 0 && c < N as i32
    }

    /// side -> board value (1 for black, 2 for white)
    #[inline]
    fn side_to_val(side: i8) -> u8 {
        if side > 0 {
            1
        } else {
            2
        }
    }

    #[inline]
    fn needs_forbidden_filter(&self) -> bool {
        self.side > 0 && matches!(self.rules, GomokuVariant::Omok | GomokuVariant::Renju)
    }

    #[inline]
    fn filtered_black_move_is_legal(&self, pos: usize) -> bool {
        debug_assert!(self.side > 0);
        debug_assert_eq!(self.board[pos], 0);
        !self.nb_get(pos) || !self.compute_black_forbidden(pos)
    }

    // ── near_black bitmask ──

    #[inline]
    fn nb_get(&self, pos: usize) -> bool {
        let word = pos >> 6; // pos / 64
        let bit = pos & 63; // pos % 64
        (self.near_black[word] >> bit) & 1 != 0
    }

    #[inline]
    fn nb_set(&mut self, pos: usize) {
        let word = pos >> 6;
        let bit = pos & 63;
        self.near_black[word] |= 1u64 << bit;
    }

    /// Mark all cells within distance 4 of `pos` (in all 4+4 directions) as near-black.
    fn nb_mark_around(&mut self, pos: usize) {
        let r = Self::row(pos);
        let c = Self::col(pos);
        self.nb_set(pos); // the stone itself
        for &(dr, dc) in &DIRS {
            for sign in [-1i32, 1] {
                for dist in 1..=4i32 {
                    let nr = r + sign * dist * dr;
                    let nc = c + sign * dist * dc;
                    if !Self::in_bounds(nr, nc) {
                        break;
                    }
                    self.nb_set(Self::to_pos(nr, nc));
                }
            }
        }
    }

    // ── Win detection ──

    /// Count consecutive stones of `val` from (r,c) in direction (dr,dc), NOT including (r,c).
    #[inline]
    fn count_dir(&self, r: i32, c: i32, dr: i32, dc: i32, val: u8) -> i32 {
        let mut cnt = 0;
        let (mut nr, mut nc) = (r + dr, c + dc);
        while Self::in_bounds(nr, nc) && self.board[Self::to_pos(nr, nc)] == val {
            cnt += 1;
            nr += dr;
            nc += dc;
        }
        cnt
    }

    /// Total line length through pos in direction dir (including pos)
    fn line_len_at(&self, pos: usize, dir: usize, val: u8) -> i32 {
        let r = Self::row(pos);
        let c = Self::col(pos);
        let (dr, dc) = DIRS[dir];
        1 + self.count_dir(r, c, dr, dc, val) + self.count_dir(r, c, -dr, -dc, val)
    }

    /// Check if stone at pos (already placed) creates a winning line.
    fn check_win_at(&self, pos: usize) -> bool {
        let val = self.board[pos];
        if val == 0 {
            return false;
        }
        let is_black = val == 1;

        for dir in 0..4 {
            let len = self.line_len_at(pos, dir, val);
            match self.rules {
                GomokuVariant::Freestyle => {
                    if len >= 5 {
                        return true;
                    }
                }
                GomokuVariant::Standard => {
                    if len == 5 {
                        return true;
                    }
                }
                GomokuVariant::Omok | GomokuVariant::Renju => {
                    if is_black {
                        // Black wins with EXACTLY 5 (overline does NOT win)
                        if len == 5 {
                            return true;
                        }
                    } else {
                        // White wins with 5 or more
                        if len >= 5 {
                            return true;
                        }
                    }
                }
                GomokuVariant::Caro => {
                    if self.caro_line_wins(pos, dir, val, len) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Hypothetical: would placing `val` at `pos` form a winning line?
    /// Does NOT require the stone to be placed on the board.
    fn would_win(&self, pos: usize, val: u8) -> bool {
        let r = Self::row(pos);
        let c = Self::col(pos);
        let is_black = val == 1;

        for dir in 0..4 {
            let (dr, dc) = DIRS[dir];
            let len = 1 + self.count_dir(r, c, dr, dc, val) + self.count_dir(r, c, -dr, -dc, val);
            match self.rules {
                GomokuVariant::Freestyle => {
                    if len >= 5 {
                        return true;
                    }
                }
                GomokuVariant::Standard => {
                    if len == 5 {
                        return true;
                    }
                }
                GomokuVariant::Omok | GomokuVariant::Renju => {
                    if is_black {
                        if len == 5 {
                            return true;
                        }
                    } else {
                        if len >= 5 {
                            return true;
                        }
                    }
                }
                GomokuVariant::Caro => {
                    if self.caro_line_wins(pos, dir, val, len) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn caro_line_wins(&self, pos: usize, dir: usize, val: u8, len: i32) -> bool {
        if len != 5 {
            return false;
        }

        let r = Self::row(pos);
        let c = Self::col(pos);
        let (dr, dc) = DIRS[dir];
        let forward = self.count_dir(r, c, dr, dc, val);
        let backward = self.count_dir(r, c, -dr, -dc, val);

        let before_r = r - (backward + 1) * dr;
        let before_c = c - (backward + 1) * dc;
        let after_r = r + (forward + 1) * dr;
        let after_c = c + (forward + 1) * dc;

        let before_blocked = !Self::in_bounds(before_r, before_c)
            || self.board[Self::to_pos(before_r, before_c)] != 0;
        let after_blocked =
            !Self::in_bounds(after_r, after_c) || self.board[Self::to_pos(after_r, after_c)] != 0;

        !(before_blocked && after_blocked)
    }

    // ── Forbidden move detection ──

    /// Is placing at `pos` forbidden for the current player?
    pub fn is_forbidden(&self, pos: usize) -> bool {
        if self.side != 1 {
            return false;
        }
        self.compute_black_forbidden(pos)
    }

    /// Compute whether placing BLACK at `pos` would be forbidden.
    /// **Phase 1.5 최적화**: 4방향 × 9-cell 배열을 한 번만 읽고,
    /// would_win / is_overline / is_double_three / is_double_four 모두에 재사용.
    /// 셀당 보드 읽기: 100회 → 36회.
    fn compute_black_forbidden(&self, pos: usize) -> bool {
        match self.rules {
            GomokuVariant::Freestyle | GomokuVariant::Standard | GomokuVariant::Caro => {
                return false;
            }
            _ => {}
        }
        if self.board[pos] != 0 {
            return false;
        }
        if !self.nb_get(pos) {
            return false;
        }

        // ── 1. Read 4 × 9-cell arrays ONCE ──
        // v[dir][0..8], pos at index 4 (hypothetical black), OOB=3
        let r = Self::row(pos);
        let c = Self::col(pos);

        let mut v = [[3u8; 9]; 4];
        for dir in 0..4 {
            let (dr, dc) = DIRS[dir];
            for i in 0..9u8 {
                let off = i as i32 - 4;
                let nr = r + off * dr;
                let nc = c + off * dc;
                if Self::in_bounds(nr, nc) {
                    let p = Self::to_pos(nr, nc);
                    v[dir][i as usize] = if p == pos { 1 } else { self.board[p] };
                }
            }
        }

        // ── 2. Per-direction consecutive counts (from arrays, no board reads) ──
        let mut line_len = [1i32; 4];
        for dir in 0..4 {
            let mut fwd = 0i32;
            for i in 5..9 {
                if v[dir][i] == 1 {
                    fwd += 1;
                } else {
                    break;
                }
            }
            let mut bwd = 0i32;
            for i in (0..4).rev() {
                if v[dir][i] == 1 {
                    bwd += 1;
                } else {
                    break;
                }
            }
            line_len[dir] = 1 + fwd + bwd;
        }

        // ── 3. would_win: exactly 5 → NOT forbidden (winning move overrides) ──
        for dir in 0..4 {
            if line_len[dir] == 5 {
                return false;
            }
        }

        // ── 4. is_overline: 6+ → forbidden ──
        for dir in 0..4 {
            if line_len[dir] >= 6 {
                return true;
            }
        }

        // ── 5. is_double_three (uses pre-read arrays) ──
        let mut three_count = 0u32;
        for dir in 0..4 {
            let has = match self.rules {
                GomokuVariant::Omok => Self::check_omok_open_three(&v[dir]),
                GomokuVariant::Renju => Self::check_renju_active_three(&v[dir]),
                _ => false,
            };
            if has {
                three_count += 1;
                if three_count >= 2 {
                    return true;
                }
            }
        }

        // ── 6. is_double_four (Renju only, uses pre-read arrays) ──
        if self.rules == GomokuVariant::Renju {
            let mut four_count = 0u32;
            for dir in 0..4 {
                four_count += Self::count_fours_from_array(&v[dir]);
                if four_count >= 2 {
                    return true;
                }
            }
        }

        false
    }

    // ── Static pattern checks on pre-read 9-cell arrays ──

    /// Omok 활삼 check on 9-cell array (pos at index 4).
    /// Patterns: _BBB_ (5-cell) or _BB_B_ / _B_BB_ (6-cell).
    #[inline]
    fn check_omok_open_three(v: &[u8; 9]) -> bool {
        // Consecutive: _BBB_ → v[s]=0, v[s+1..s+3]=1, v[s+4]=0
        for s in 1..=3usize {
            if v[s] == 0 && v[s + 1] == 1 && v[s + 2] == 1 && v[s + 3] == 1 && v[s + 4] == 0 {
                if (s == 0 || v[s - 1] != 1) && (s + 5 >= 9 || v[s + 5] != 1) {
                    return true;
                }
            }
        }
        // Gap: _BB_B_ or _B_BB_
        for s in 0..=3usize {
            if s + 5 >= 9 {
                continue;
            }
            if v[s] != 0 || v[s + 5] != 0 {
                continue;
            }
            if v[s + 1] == 1 && v[s + 2] == 1 && v[s + 3] == 0 && v[s + 4] == 1 {
                if (4 == s + 1 || 4 == s + 2 || 4 == s + 4)
                    && (s == 0 || v[s - 1] != 1)
                    && (s + 6 >= 9 || v[s + 6] != 1)
                {
                    return true;
                }
            }
            if v[s + 1] == 1 && v[s + 2] == 0 && v[s + 3] == 1 && v[s + 4] == 1 {
                if (4 == s + 1 || 4 == s + 3 || 4 == s + 4)
                    && (s == 0 || v[s - 1] != 1)
                    && (s + 6 >= 9 || v[s + 6] != 1)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Renju 활삼 check on 9-cell array (pos at index 4).
    /// Broader definition + straight-four filter.
    #[inline]
    fn check_renju_active_three(v: &[u8; 9]) -> bool {
        for s in 0..=4usize {
            if s + 4 >= 9 {
                break;
            }
            if 4 < s || 4 > s + 4 {
                continue;
            }
            let w = &v[s..s + 5];
            let mut blacks = 0u8;
            let mut bad = false;
            for j in 0..5 {
                match w[j] {
                    1 => blacks += 1,
                    0 => {}
                    _ => {
                        bad = true;
                        break;
                    }
                }
            }
            if bad || blacks != 3 {
                continue;
            }
            let before = if s > 0 { v[s - 1] } else { 3 };
            let after = if s + 5 < 9 { v[s + 5] } else { 3 };
            if before == 1 || after == 1 {
                continue;
            }
            if before == 0 || after == 0 {
                return true;
            }
        }
        false
    }

    /// Count distinct fours from 9-cell array (pos at index 4).
    #[inline]
    fn count_fours_from_array(v: &[u8; 9]) -> u32 {
        let mut sigs: [u16; 3] = [0; 3];
        let mut n = 0u32;
        for s in 0..=4usize {
            if s + 4 >= 9 {
                break;
            }
            if 4 < s || 4 > s + 4 {
                continue;
            }
            let w = &v[s..s + 5];
            let mut blacks = 0u8;
            let mut empties = 0u8;
            let mut sig: u16 = 0;
            let mut ok = true;
            for j in 0..5 {
                match w[j] {
                    1 => {
                        blacks += 1;
                        sig |= 1 << (s + j);
                    }
                    0 => empties += 1,
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || blacks != 4 || empties != 1 {
                continue;
            }
            let before = if s > 0 { v[s - 1] } else { 3 };
            let after = if s + 5 < 9 { v[s + 5] } else { 3 };
            if before == 1 || after == 1 {
                continue;
            }
            let mut dup = false;
            for k in 0..n as usize {
                if sigs[k] == sig {
                    dup = true;
                    break;
                }
            }
            if !dup && (n as usize) < sigs.len() {
                sigs[n as usize] = sig;
                n += 1;
            }
        }
        n
    }

    /// Board as 0/1/2 encoding (for JSON protocol compatibility)
    #[cfg(test)]
    pub fn board_as_12(&self) -> &[u8; N2] {
        &self.board
    }
}

impl fmt::Display for Gomoku15 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "    ")?;
        for c in 0..N {
            write!(f, "{:2}", c)?;
        }
        writeln!(f)?;
        for r in 0..N {
            write!(f, "{:2}  ", r)?;
            for c in 0..N {
                let sym = match self.board[r * N + c] {
                    1 => " X",
                    2 => " O",
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
// § GameState 구현
// ─────────────────────────────────────────────

impl GameState for Gomoku15 {
    type Move = u16; // 0..224
    type Undo = Gomoku15Undo;

    fn initial() -> Self {
        Gomoku15::freestyle()
    }

    fn current_player(&self) -> i8 {
        self.side
    }

    fn legal_moves(&self) -> Vec<u16> {
        if self.is_terminal() {
            return vec![];
        }
        let need_filter = self.needs_forbidden_filter();
        let mut moves = Vec::with_capacity(N2 - self.moves as usize);

        if need_filter {
            for (i, &v) in self.board.iter().enumerate() {
                if v == 0 && self.filtered_black_move_is_legal(i) {
                    moves.push(i as u16);
                }
            }
        } else {
            for (i, &v) in self.board.iter().enumerate() {
                if v == 0 {
                    moves.push(i as u16);
                }
            }
        }
        moves
    }

    fn uniform_eval(&self, value: f32) -> EvalResult<u16> {
        if self.is_terminal() {
            return EvalResult {
                policy: Vec::new(),
                value,
            };
        }
        let count = self.legal_move_count();
        let prior = if count > 0 { 1.0 / count as f32 } else { 0.0 };
        let need_filter = self.needs_forbidden_filter();
        let mut policy = Vec::with_capacity(count);
        if need_filter {
            for (i, &v) in self.board.iter().enumerate() {
                if v == 0 && self.filtered_black_move_is_legal(i) {
                    policy.push((i as u16, prior));
                }
            }
        } else {
            for (i, &v) in self.board.iter().enumerate() {
                if v == 0 {
                    policy.push((i as u16, prior));
                }
            }
        }
        EvalResult { policy, value }
    }

    fn apply_move(&self, mv: u16) -> Self {
        let mut next = self.clone();
        let _undo = next.apply_move_mut_internal(mv);
        next
    }

    fn apply_move_in_place(&mut self, mv: u16) -> Gomoku15Undo {
        self.apply_move_mut_internal(mv)
    }

    fn apply_move_in_place_no_undo(&mut self, mv: u16) {
        let _ = self.apply_move_mut_internal(mv);
    }

    fn undo_move(&mut self, undo: Gomoku15Undo) {
        let moved_side = -self.side;
        let pos = undo.pos as usize;
        {
            let z = &*ZOB15;
            self.hash ^= z.piece_hash(moved_side, pos);
            self.hash ^= z.side;
        }
        self.side = moved_side;
        self.moves = self.moves.saturating_sub(1);
        self.board[pos] = 0;
        self.occupied[self.moves as usize] = 0;
        self.winner = undo.prev_winner;
        self.last_mv = undo.prev_last_mv;
        self.recent_move_len = undo.prev_recent_move_len;
        self.recent_moves = undo.prev_recent_moves;
        self.near_black = undo.prev_near_black;
    }

    fn uses_reusable_select_scratch() -> bool {
        true
    }

    fn can_skip_sorted_policy_resort() -> bool {
        true
    }

    fn is_terminal(&self) -> bool {
        self.winner != 0 || self.moves as usize >= self.move_limit()
    }

    /// negamax convention: current_player 관점
    fn outcome(&self) -> f32 {
        if self.winner == self.side {
            1.0
        } else if self.winner != 0 {
            -1.0
        } else {
            0.0
        }
    }

    fn hash(&self) -> u64 {
        self.hash
    }

    fn num_actions(&self) -> usize {
        N2
    }

    fn move_to_idx(&self, mv: u16) -> usize {
        mv as usize
    }

    fn idx_to_move(&self, idx: usize) -> Option<u16> {
        if idx < N2 {
            Some(idx as u16)
        } else {
            None
        }
    }

    /// NN 입력 feature planes (AlphaZero-style: [17, 15, 15])
    ///   planes 0-1:   현재(t=0) 내 돌, 상대 돌
    ///   planes 2-3:   t=1 (1수 전) 내 돌, 상대 돌
    ///   ...
    ///   planes 14-15: t=7 (7수 전) 내 돌, 상대 돌
    ///   plane 16:     현재 플레이어 색상 (black=1, white=0)
    fn encode_planes_into(&self, out: &mut Vec<f32>) {
        let total_planes = GOMOKU15_HISTORY_LEN * 2 + 1; // 17
        let my_val = Self::side_to_val(self.side);
        GOMOKU15_FEATURE_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let total = total_planes * N2;
            let color_base = (total_planes - 1) * N2;
            if out.len() != total {
                out.clear();
                out.resize(total, 0.0);
                scratch.touched.clear();
                scratch.color_plane_active = false;
            } else {
                let touched_len =
                    scratch.touched.len() + if scratch.color_plane_active { N2 } else { 0 };
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
                        out[color_base..color_base + N2].fill(0.0);
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

            for &pos in &self.occupied[..self.moves as usize] {
                let i = pos as usize;
                let v = self.board[i];
                let plane_offset = if v == my_val { 0 } else { N2 };
                let active_steps = match (scratch.recent_epoch[i] == recent_epoch)
                    .then_some(scratch.recent_rank[i])
                {
                    Some(rank) => rank as usize,
                    None => GOMOKU15_HISTORY_LEN,
                };
                for t in 0..active_steps {
                    let idx = t * 2 * N2 + plane_offset + i;
                    out[idx] = 1.0;
                    scratch.touched.push(idx as u16);
                }
            }

            if self.side > 0 {
                out[color_base..color_base + N2].fill(1.0);
                scratch.color_plane_active = true;
            }
        });
    }

    fn board_state_record(&self) -> Vec<i64> {
        self.board.iter().map(|&v| v as i64).collect()
    }

    /// [OPT] O(1) win check — no clone
    fn is_winning_move(&self, mv: u16) -> bool {
        let val = Self::side_to_val(self.side);
        self.would_win(mv as usize, val)
    }

    /// [OPT] Random legal move without Vec allocation
    fn random_legal_move(&self, rand_idx: usize) -> Option<u16> {
        if self.is_terminal() {
            return None;
        }
        let need_filter = self.needs_forbidden_filter();

        let n_empty = N2 - self.moves as usize;
        if n_empty == 0 {
            return None;
        }

        if !need_filter {
            // Standard or white's turn: just pick nth empty cell
            let target = rand_idx % n_empty;
            let mut count = 0usize;
            for (i, &v) in self.board.iter().enumerate() {
                if v == 0 {
                    if count == target {
                        return Some(i as u16);
                    }
                    count += 1;
                }
            }
        } else {
            // Omok/Renju black: need forbidden check. Two-pass:
            // First count legal, then pick.
            let legal_count = self
                .board
                .iter()
                .enumerate()
                .filter(|(i, &v)| v == 0 && self.filtered_black_move_is_legal(*i))
                .count();
            if legal_count == 0 {
                return None;
            }
            let target = rand_idx % legal_count;
            let mut count = 0usize;
            for (i, &v) in self.board.iter().enumerate() {
                if v == 0 && self.filtered_black_move_is_legal(i) {
                    if count == target {
                        return Some(i as u16);
                    }
                    count += 1;
                }
            }
        }
        None
    }

    fn legal_move_count(&self) -> usize {
        if self.is_terminal() {
            return 0;
        }
        let need_filter = self.needs_forbidden_filter();

        if !need_filter {
            N2 - self.moves as usize
        } else {
            let mut count = 0usize;
            for (i, &v) in self.board.iter().enumerate() {
                if v == 0 && self.filtered_black_move_is_legal(i) {
                    count += 1;
                }
            }
            count
        }
    }
}

// ─────────────────────────────────────────────
// § MctsConfig 프리셋
// ─────────────────────────────────────────────

use crate::mcts::gvoc::GvocConfig;
use crate::mcts::quartz::QuartzConfig;
use crate::mcts::{MctsConfig, PwConfig};

/// Gomoku 15×15 QUARTZ 프리셋.
///
/// v0.9.2 실험 결론 반영:
/// - PenaltyMode::Legacy, prior_refresh_rate: 0.0
/// - PW: α=2.0, β=0.5 → k(N) = 2√N
/// - GVOC: max_visible=64 (Omok 금수로 합법수 < 225)
/// - σ₀ = 0.3 (binary rollout 기준, NN 사용 시 재캘리브레이션 필요)
pub fn gomoku15_quartz(variant: GomokuVariant) -> MctsConfig {
    let gvoc_max = match variant {
        GomokuVariant::Omok | GomokuVariant::Renju => 64,
        GomokuVariant::Freestyle | GomokuVariant::Standard | GomokuVariant::Caro => 80,
    };

    MctsConfig::evaluation_with_pw(2.0, PwConfig::new(2.0, 0.5))
        .with_quartz(QuartzConfig {
            sigma_0: 0.3,
            min_visits: 30,
            check_interval: 50,
            ..Default::default()
        })
        .with_gvoc(GvocConfig {
            expand_thresh: 0.01,
            contract_thresh: 0.001,
            expand_delta: 2,
            max_visible: gvoc_max,
            min_visible: 1,
            score_interval: 50,
        })
}

/// Gomoku 15×15 QUARTZ 프리셋 — 시간 제한 모드.
pub fn gomoku15_quartz_timed(variant: GomokuVariant, budget_ms: u64) -> MctsConfig {
    let mut cfg = gomoku15_quartz(variant);
    if let Some(ref mut q) = cfg.quartz {
        q.ctm_budget_ms = budget_ms;
    }
    cfg
}

// ─────────────────────────────────────────────
// § 테스트
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::{Gomoku, TicTacToe};
    use std::hint::black_box;
    use std::time::Instant;

    fn bench_loops(default: usize) -> usize {
        std::env::var("GAME_BENCH_LOOPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(default)
    }

    // ── Helpers ──

    fn make(variant: GomokuVariant, moves: &[(usize, usize)]) -> Gomoku15 {
        let mut s = Gomoku15::new(variant);
        for &(r, c) in moves {
            s = s.apply_move((r * N + c) as u16);
        }
        s
    }

    fn std(moves: &[(usize, usize)]) -> Gomoku15 {
        make(GomokuVariant::Standard, moves)
    }

    fn free(moves: &[(usize, usize)]) -> Gomoku15 {
        make(GomokuVariant::Freestyle, moves)
    }

    fn omok(moves: &[(usize, usize)]) -> Gomoku15 {
        make(GomokuVariant::Omok, moves)
    }

    fn a(r: usize, c: usize) -> u16 {
        (r * N + c) as u16
    }

    fn encode_planes_reference(state: &Gomoku15) -> Vec<f32> {
        let total_planes = GOMOKU15_HISTORY_LEN * 2 + 1;
        let mut out = vec![0.0; total_planes * N2];
        let my_val = Gomoku15::side_to_val(state.side);
        let mut recent_rank = [0u8; N2];
        for (rank, &mv) in state.recent_moves[..state.recent_move_len as usize]
            .iter()
            .rev()
            .enumerate()
        {
            recent_rank[mv as usize] = (rank + 1) as u8;
        }
        for (i, &v) in state.board.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let plane_offset = if v == my_val { 0 } else { N2 };
            let active_steps = match recent_rank[i] {
                0 => GOMOKU15_HISTORY_LEN,
                rank => rank as usize,
            };
            for t in 0..active_steps {
                out[t * 2 * N2 + plane_offset + i] = 1.0;
            }
        }
        if state.side > 0 {
            let color_base = (total_planes - 1) * N2;
            out[color_base..color_base + N2].fill(1.0);
        }
        out
    }

    fn assert_occupied_matches_board(state: &Gomoku15) {
        let occupied_len = state.moves as usize;
        let board_count = state.board.iter().filter(|&&v| v != 0).count();
        assert_eq!(
            occupied_len, board_count,
            "occupied prefix length must match stone count"
        );

        let mut seen = [false; N2];
        for &pos in &state.occupied[..occupied_len] {
            let pos = pos as usize;
            assert_ne!(state.board[pos], 0, "occupied entries must point to stones");
            assert!(!seen[pos], "occupied entries must not duplicate positions");
            seen[pos] = true;
        }
        for (idx, &cell) in state.board.iter().enumerate() {
            assert_eq!(
                seen[idx],
                cell != 0,
                "occupied prefix must cover exactly all stones"
            );
        }
    }

    // ══════════════════════════════════════════
    // C++ adversarial_gomoku.cpp 포팅 (20개)
    // ══════════════════════════════════════════

    // TEST 1: Standard — exact 5 wins (horizontal)
    #[test]
    fn test_standard_five_horizontal() {
        // Black: (7,5)(7,6)(7,7)(7,8)(7,9)
        // White: (0,0)(0,1)(0,2)(0,3)
        let s = std(&[
            (7, 5),
            (0, 0),
            (7, 6),
            (0, 1),
            (7, 7),
            (0, 2),
            (7, 8),
            (0, 3),
            (7, 9),
        ]);
        assert!(s.is_terminal(), "Black should win with 5-in-a-row");
        assert_eq!(s.winner, 1, "Black wins");
    }

    // TEST 2: Freestyle — 6+ in a row also wins
    #[test]
    fn test_freestyle_overline_wins_legacy_path() {
        // Black places in row 7: cols 0,1,2,3,5 then fills gap at 4 -> 6-in-a-row
        let s = free(&[
            (7, 0),
            (0, 0),
            (7, 1),
            (1, 14),
            (7, 2),
            (2, 0),
            (7, 3),
            (3, 14),
            (7, 5),
            (4, 0),
            (7, 4),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, 1, "6-in-a-row should win in Freestyle");
    }

    // TEST 3: All 4 directions
    #[test]
    fn test_all_directions_vertical() {
        let s = std(&[
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
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_all_directions_diagonal_dr() {
        let s = std(&[
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
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_all_directions_diagonal_ur() {
        let s = std(&[
            (7, 3),
            (0, 0),
            (6, 4),
            (0, 1),
            (5, 5),
            (0, 2),
            (4, 6),
            (0, 3),
            (3, 7),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, 1);
    }

    // TEST 4: White wins
    #[test]
    fn test_white_wins() {
        // Black spreads out, White lines up
        let s = std(&[
            (0, 0),
            (7, 5),
            (2, 14),
            (7, 6),
            (4, 0),
            (7, 7),
            (6, 14),
            (7, 8),
            (14, 7),
            (7, 9),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, -1, "White should win");
    }

    // TEST 5: White 6+ also wins in Freestyle
    #[test]
    fn test_white_freestyle_overline_wins() {
        let s = free(&[
            (0, 0),
            (8, 0),
            (2, 14),
            (8, 1),
            (4, 0),
            (8, 2),
            (6, 14),
            (8, 3),
            (14, 7),
            (8, 5),
            (12, 12),
            (8, 4),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, -1);
    }

    // TEST 6: Omok — double-three forbidden
    #[test]
    fn test_omok_double_three_forbidden() {
        // Two threes intersecting at (7,7)
        // B: (7,6),(7,8) horizontal + (6,7),(8,7) vertical
        let s = omok(&[
            (7, 6),
            (0, 0),
            (7, 8),
            (0, 1),
            (6, 7),
            (0, 2),
            (8, 7),
            (0, 3),
        ]);
        // (7,7) should be forbidden for black
        assert!(
            s.is_forbidden(7 * N + 7),
            "Double-three at (7,7) should be forbidden"
        );
        let moves = s.legal_moves();
        assert!(
            !moves.contains(&a(7, 7)),
            "(7,7) should not be in legal moves"
        );
    }

    // TEST 7: Omok — single three OK
    #[test]
    fn test_omok_single_three_ok() {
        let s = omok(&[(7, 6), (0, 0), (7, 8), (0, 1)]);
        // (7,7) only forms one three — should be legal
        assert!(!s.is_forbidden(7 * N + 7));
        let moves = s.legal_moves();
        assert!(moves.contains(&a(7, 7)));
    }

    // TEST 8: Omok — gap-three double
    #[test]
    fn test_omok_gap_three_double() {
        // Horizontal gap three at (7,7): B at (7,5),(7,8) → _B(5)_B(7)B(8)_
        // Vertical gap three at (7,7): B at (5,7),(8,7) → _B(5)_B(7)B(8)_
        let s = omok(&[
            (7, 5),
            (0, 0),
            (7, 8),
            (0, 1),
            (5, 7),
            (0, 2),
            (8, 7),
            (0, 3),
        ]);
        assert!(
            s.is_forbidden(7 * N + 7),
            "Double gap-three at (7,7) should be forbidden"
        );
    }

    // TEST 9: Omok — overline forbidden for Black
    #[test]
    fn test_omok_overline_forbidden() {
        // B: row 7, cols 0,1,2,3,5 → gap at 4 → would make 6-in-a-row
        let s = omok(&[
            (7, 0),
            (0, 0),
            (7, 1),
            (1, 14),
            (7, 2),
            (2, 0),
            (7, 3),
            (3, 14),
            (7, 5),
            (4, 0),
        ]);
        let fill = a(7, 4);
        assert!(
            s.is_forbidden(fill as usize),
            "Omok overline should be forbidden"
        );
    }

    // TEST 10: Draw detection (not terminal at start)
    #[test]
    fn test_draw_not_terminal_at_start() {
        let s = Gomoku15::standard();
        assert!(!s.is_terminal());
        assert_eq!(s.outcome(), 0.0);
    }

    // TEST 11-13: Edge/corner wins
    #[test]
    fn test_edge_bottom() {
        let s = std(&[
            (14, 0),
            (0, 0),
            (14, 1),
            (0, 1),
            (14, 2),
            (0, 2),
            (14, 3),
            (0, 3),
            (14, 4),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_edge_right() {
        let s = std(&[
            (0, 14),
            (0, 0),
            (1, 14),
            (1, 0),
            (2, 14),
            (2, 0),
            (3, 14),
            (3, 0),
            (4, 14),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_corner_diagonal() {
        let s = std(&[
            (0, 0),
            (14, 0),
            (1, 1),
            (14, 1),
            (2, 2),
            (14, 2),
            (3, 3),
            (14, 3),
            (4, 4),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, 1);
    }

    // ══════════════════════════════════════════
    // 추가 테스트 (30+)
    // ══════════════════════════════════════════

    #[test]
    fn test_initial_state() {
        let s = Gomoku15::standard();
        assert_eq!(s.legal_moves().len(), 225);
        assert_eq!(s.current_player(), 1);
        assert!(!s.is_terminal());
    }

    #[test]
    fn test_apply_move_pure() {
        let s = Gomoku15::standard();
        let s2 = s.apply_move(112); // center
        assert_eq!(s.board[112], 0, "original should be unchanged");
        assert_eq!(s2.board[112], 1);
        assert_eq!(s2.current_player(), -1);
    }

    #[test]
    fn test_apply_move_in_place_roundtrip_matches_pure() {
        let mut s = Gomoku15::renju();
        for mv in [112u16, 0, 113, 1, 114, 2, 115] {
            let before = s.clone();
            let pure = s.apply_move(mv);
            let undo = s.apply_move_in_place(mv);
            assert_eq!(s.board, pure.board);
            assert_eq!(s.hash, pure.hash);
            assert_eq!(s.side, pure.side);
            assert_eq!(s.moves, pure.moves);
            assert_eq!(s.winner, pure.winner);
            assert_eq!(s.near_black, pure.near_black);
            assert_eq!(s.recent_moves, pure.recent_moves);
            s.undo_move(undo);
            assert_eq!(s.board, before.board);
            assert_eq!(s.hash, before.hash);
            assert_eq!(s.side, before.side);
            assert_eq!(s.moves, before.moves);
            assert_eq!(s.winner, before.winner);
            assert_eq!(s.last_mv, before.last_mv);
            assert_eq!(s.near_black, before.near_black);
            assert_eq!(s.recent_moves, before.recent_moves);
            s = pure;
        }
    }

    #[test]
    fn test_four_no_win() {
        let s = std(&[(7, 3), (0, 0), (7, 4), (0, 1), (7, 5), (0, 2), (7, 6)]);
        assert!(!s.is_terminal(), "4-in-a-row should NOT win");
    }

    #[test]
    fn test_hash_transposition() {
        let mut a = Gomoku15::standard();
        for mv in [7 * N + 7, 0, 7 * N + 8, 1] {
            a = a.apply_move(mv as u16);
        }

        let mut b = Gomoku15::standard();
        for mv in [7 * N + 8, 1, 7 * N + 7, 0] {
            b = b.apply_move(mv as u16);
        }

        assert_eq!(a.hash(), b.hash(), "transposition hash should match");
    }

    #[test]
    fn test_encode_planes() {
        let s = std(&[(7, 7), (0, 0)]); // B at center, W at corner
        let planes = s.encode_planes();
        assert_eq!(planes.len(), 17 * 225);
        // current player is now black (after 2 moves)
        assert_eq!(planes[7 * 15 + 7], 1.0); // plane 0: black's piece
        assert_eq!(planes[225 + 0], 1.0); // plane 1: white's piece
    }

    #[test]
    fn test_is_winning_move() {
        let s = std(&[
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
        ]);
        assert!(s.is_winning_move(a(0, 4)), "placing at (0,4) should win");
        assert!(!s.is_winning_move(a(2, 0)), "(2,0) should not win");
    }

    #[test]
    fn test_omok_white_no_restrictions() {
        // White can make 6+ in Omok (no restrictions for white)
        let s = omok(&[
            (14, 0),
            (8, 0),
            (14, 1),
            (8, 1),
            (14, 2),
            (8, 2),
            (14, 3),
            (8, 3),
            (14, 4),
            (8, 5),
            (14, 5),
            (8, 4),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, -1, "White's 6-in-a-row should win in Omok");
    }

    #[test]
    fn test_omok_exactly_five_wins() {
        // Black makes exactly 5 — should win even in Omok
        let s = omok(&[
            (7, 5),
            (0, 0),
            (7, 6),
            (0, 1),
            (7, 7),
            (0, 2),
            (7, 8),
            (0, 3),
            (7, 9),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_omok_winning_move_overrides_forbidden() {
        // If placing creates exactly 5, it should be legal even if it would also
        // create other patterns (like double-three). The 5-in-a-row takes precedence.
        // Set up: B has 4 in a row at (7,5)(7,6)(7,7)(7,8), placing at (7,9) makes 5
        let s = omok(&[
            (7, 5),
            (0, 0),
            (7, 6),
            (0, 1),
            (7, 7),
            (0, 2),
            (7, 8),
            (0, 3),
        ]);
        // (7,9) would complete exactly 5 — should NOT be forbidden
        assert!(!s.is_forbidden(7 * N + 9));
    }

    #[test]
    fn test_legal_move_count_matches() {
        let s = std(&[(7, 7), (0, 0), (7, 8)]);
        assert_eq!(s.legal_moves().len(), s.legal_move_count());
    }

    #[test]
    fn test_move_to_idx_roundtrip() {
        let s = Gomoku15::standard();
        for i in 0..225u16 {
            let idx = s.move_to_idx(i);
            let mv = s.idx_to_move(idx).unwrap();
            assert_eq!(mv, i);
        }
    }

    #[test]
    fn test_num_actions() {
        let s = Gomoku15::standard();
        assert_eq!(s.num_actions(), 225);
    }

    #[test]
    fn test_negamax_convention() {
        // After black wins, it's white's turn, outcome should be -1.0 (white lost)
        let s = std(&[
            (7, 5),
            (0, 0),
            (7, 6),
            (0, 1),
            (7, 7),
            (0, 2),
            (7, 8),
            (0, 3),
            (7, 9),
        ]);
        assert_eq!(s.current_player(), -1);
        assert_eq!(s.outcome(), -1.0, "white's perspective: lost");
    }

    #[test]
    fn test_variant_preserved_after_apply() {
        let s = Gomoku15::omok();
        let s2 = s.apply_move(0);
        assert_eq!(s2.rules, GomokuVariant::Omok);
    }

    // ── Omok: more edge cases ──

    #[test]
    fn test_omok_overline_not_forbidden_for_white() {
        // White should be able to make 6+ in Omok
        // Set up so it's white's turn and white has 5 stones in a row
        let s = omok(&[
            (14, 0),
            (8, 0),
            (14, 1),
            (8, 1),
            (14, 2),
            (8, 2),
            (14, 3),
            (8, 3),
            (14, 4),
            (8, 5),
        ]);
        // White to play at (8,4) would make 6-in-a-row — should be legal
        assert!(
            !s.is_forbidden(8 * N + 4),
            "White should have no forbidden moves"
        );
    }

    #[test]
    fn test_omok_three_not_open_if_blocked() {
        // If one end of a three is blocked by white, it's not an "open" three
        // B at (7,6),(7,7) with W at (7,5) → (7,8) would be _BB?_ but one end blocked
        let s = omok(&[
            (7, 6),
            (7, 5), // B at (7,6), W at (7,5) — blocks one end
            (7, 7),
            (0, 0), // B at (7,7)
            (6, 7),
            (0, 1), // B at (6,7) — vertical
            (8, 7),
            (0, 2), // B at (8,7) — vertical
        ]);
        // (7,8) creates horizontal _BB(8)_ but one end blocked by W
        // and vertical three. Only one open three (vertical), not double-three.
        // Actually: horizontal direction from (7,8): W(7,5) B(7,6) B(7,7) B(7,8) ?
        // That's not _BBB_ pattern since W is at (7,5). The window would be W,B,B,B,? — not open.
        // So only vertical is open three → single three → legal
        assert!(!s.is_forbidden(7 * N + 8));
    }

    // ── MCTS integration ──

    #[test]
    fn test_mcts_integration_standard() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::search::FixedIterations;
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let state = Gomoku15::standard();
        let eval: Arc<dyn crate::game::Evaluator<Gomoku15> + Send + Sync> = Arc::new(UniformEval);
        let config = MctsConfig::evaluation(2.0);
        let engine = MctsEngine::new(state, eval, config);
        engine.run(&mut FixedIterations::new(100));

        let best = engine.best_move();
        assert!(best.is_some());
        let mv = best.unwrap();
        assert!(mv < 225);
    }

    #[test]
    fn test_mcts_integration_omok() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::search::FixedIterations;
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let state = Gomoku15::omok();
        let eval: Arc<dyn crate::game::Evaluator<Gomoku15> + Send + Sync> = Arc::new(UniformEval);
        let config = MctsConfig::evaluation(2.0);
        let engine = MctsEngine::new(state, eval, config);
        engine.run(&mut FixedIterations::new(100));

        let best = engine.best_move();
        assert!(best.is_some());
    }

    #[test]
    fn test_quartz_integration() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::quartz::{QuartzConfig, QuartzController};
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let state = Gomoku15::standard();
        let eval: Arc<dyn crate::game::Evaluator<Gomoku15> + Send + Sync> = Arc::new(UniformEval);
        let qcfg = QuartzConfig::default();
        let config = MctsConfig::evaluation(2.0).with_quartz(qcfg.clone());
        let engine = MctsEngine::new(state, eval, config);

        let mut ctrl = QuartzController::new(200, qcfg);
        let stats = engine.run_quartz(&mut ctrl);

        assert!(stats.iterations > 0);
        assert!(engine.best_move().is_some());
    }

    #[test]
    fn test_gomoku15_quartz_preset() {
        use super::gomoku15_quartz;
        use crate::mcts::eval::UniformEval;
        use crate::mcts::quartz::QuartzController;
        use crate::mcts::MctsEngine;
        use std::sync::Arc;

        for variant in [
            GomokuVariant::Standard,
            GomokuVariant::Omok,
            GomokuVariant::Renju,
        ] {
            let state = Gomoku15::new(variant);
            let eval: Arc<dyn crate::game::Evaluator<Gomoku15> + Send + Sync> =
                Arc::new(UniformEval);
            let config = gomoku15_quartz(variant);
            let qcfg = config.quartz.clone().unwrap();
            let engine = MctsEngine::new(state, eval, config);

            let mut ctrl = QuartzController::new(100, qcfg);
            let stats = engine.run_quartz(&mut ctrl);
            assert!(stats.iterations > 0, "preset {:?} should run", variant);
        }
    }

    // ── Concurrent safety: Gomoku15 is Send + Sync ──
    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Gomoku15>();
    }

    // ══════════════════════════════════════════
    // RENJU 테스트 (렌주 = 장목 + 쌍삼 + 쌍사 금지)
    // ══════════════════════════════════════════

    fn renju(moves: &[(usize, usize)]) -> Gomoku15 {
        make(GomokuVariant::Renju, moves)
    }

    // ── Renju: 장목(overline) 금지 ──

    #[test]
    fn test_renju_overline_forbidden() {
        // B: row 7, cols 0,1,2,3,5 → gap at 4 → would make 6-in-a-row
        let s = renju(&[
            (7, 0),
            (0, 0),
            (7, 1),
            (1, 14),
            (7, 2),
            (2, 0),
            (7, 3),
            (3, 14),
            (7, 5),
            (4, 0),
        ]);
        assert!(
            s.is_forbidden(7 * N + 4),
            "Renju overline should be forbidden"
        );
    }

    #[test]
    fn test_renju_overline_not_forbidden_for_white() {
        // White has no restrictions in Renju
        let s = renju(&[
            (14, 0),
            (8, 0),
            (14, 1),
            (8, 1),
            (14, 2),
            (8, 2),
            (14, 3),
            (8, 3),
            (14, 4),
            (8, 5),
        ]);
        // White to play, (8,4) makes 6-in-a-row — should be legal
        assert!(!s.is_forbidden(8 * N + 4));
    }

    // ── Renju: 쌍삼(double-three) 금지 ──

    #[test]
    fn test_renju_double_three_forbidden() {
        // Same as Omok: intersecting horizontal + vertical threes at (7,7)
        let s = renju(&[
            (7, 6),
            (0, 0),
            (7, 8),
            (0, 1),
            (6, 7),
            (0, 2),
            (8, 7),
            (0, 3),
        ]);
        assert!(
            s.is_forbidden(7 * N + 7),
            "Renju double-three should be forbidden"
        );
    }

    #[test]
    fn test_renju_single_three_ok() {
        let s = renju(&[(7, 6), (0, 0), (7, 8), (0, 1)]);
        assert!(!s.is_forbidden(7 * N + 7), "Single three is OK in Renju");
    }

    // ── Renju: 쌍사(double-four) 금지 ──

    #[test]
    fn test_renju_double_four_straight_x_straight() {
        // Black has two groups of 3 meeting at pos, each becoming a straight four.
        //
        // Horizontal: B at (7,4),(7,5),(7,6) → placing (7,7) → BBBB with gap at (7,8)
        // Vertical:   B at (4,7),(5,7),(6,7) → placing (7,7) → BBBB with gap at (8,7)
        //
        // Both become straight fours → double-four → forbidden
        let s = renju(&[
            (7, 4),
            (0, 0),
            (7, 5),
            (0, 1),
            (7, 6),
            (0, 2), // horizontal: 3 in a row
            (4, 7),
            (0, 3),
            (5, 7),
            (0, 4),
            (6, 7),
            (0, 5), // vertical: 3 in a row
        ]);
        assert!(
            s.is_forbidden(7 * N + 7),
            "Renju double-four (straight×straight) should be forbidden"
        );
    }

    #[test]
    fn test_renju_double_four_broken() {
        // Black forms two broken fours crossing at pos.
        //
        // Horizontal broken four: B at (7,4),(7,5),(7,8) — placing (7,7) → BBx_BB pattern
        //   wait, let me think more carefully...
        //
        // Horizontal: B at (7,5),(7,6),(7,8) → placing (7,7) makes BB_B? No:
        //   (7,5)=B, (7,6)=B, (7,7)=B(new), (7,8)=B → that's actually 4 consecutive (BBBB)
        //   That's a straight four, not broken.
        //
        // For a true broken four: B at (7,4),(7,5),(7,8) → placing (7,7):
        //   (7,4)=B, (7,5)=B, (7,6)=empty, (7,7)=B(new), (7,8)=B → BB_BB
        //   5-cell window [4..8]: B,B,_,B,B → 4 blacks, 1 empty at (7,6) → broken four!
        //
        // Vertical broken four: B at (4,7),(5,7),(8,7) → placing (7,7):
        //   (4,7)=B, (5,7)=B, (6,7)=empty, (7,7)=B(new), (8,7)=B → BB_BB
        //   → broken four with gap at (6,7)
        let s = renju(&[
            (7, 4),
            (0, 0),
            (7, 5),
            (0, 1),
            (7, 8),
            (0, 2), // horizontal broken
            (4, 7),
            (0, 3),
            (5, 7),
            (0, 4),
            (8, 7),
            (0, 5), // vertical broken
        ]);
        assert!(
            s.is_forbidden(7 * N + 7),
            "Renju double-four (broken×broken) should be forbidden"
        );
    }

    #[test]
    fn test_renju_double_four_straight_x_broken() {
        // Horizontal straight four: B at (7,5),(7,6),(7,8) → (7,7) makes BBBB (consecutive 4)
        // Diagonal broken four: B at (5,5),(6,6),(9,9) → (7,7) makes B,B,B(new),_,_,B → not a four
        //
        // Better setup:
        // Horizontal: B at (7,4),(7,5),(7,6) → (7,7) makes BBBB_ = straight four
        // Vertical broken: B at (4,7),(5,7),(8,7) → (7,7) makes BB_BB with gap at (6,7) = broken four
        let s = renju(&[
            (7, 4),
            (0, 0),
            (7, 5),
            (0, 1),
            (7, 6),
            (0, 2), // horizontal straight
            (4, 7),
            (0, 3),
            (5, 7),
            (0, 4),
            (8, 7),
            (0, 5), // vertical broken four
        ]);
        assert!(
            s.is_forbidden(7 * N + 7),
            "Renju double-four (straight×broken) should be forbidden"
        );
    }

    #[test]
    fn test_renju_single_four_ok() {
        // Only one four in one direction → not forbidden
        // B at (7,4),(7,5),(7,6) → (7,7) makes straight four BBBB
        let s = renju(&[(7, 4), (0, 0), (7, 5), (0, 1), (7, 6), (0, 2)]);
        assert!(!s.is_forbidden(7 * N + 7), "Single four is OK in Renju");
    }

    #[test]
    fn test_renju_single_broken_four_ok() {
        // B at (7,4),(7,5),(7,8) → (7,7) makes BB_BB = one broken four → OK
        let s = renju(&[(7, 4), (0, 0), (7, 5), (0, 1), (7, 8), (0, 2)]);
        assert!(
            !s.is_forbidden(7 * N + 7),
            "Single broken four is OK in Renju"
        );
    }

    // ── Renju: 5목 완성이 금수를 면제 ──

    #[test]
    fn test_renju_five_overrides_double_four() {
        // Even if placing creates double-four, if it also makes exactly 5 → legal (winning move)
        // B at (7,3),(7,4),(7,5),(7,6) → (7,7) makes 5-in-a-row → winning, not forbidden
        // Even if vertical also makes a four.
        let s = renju(&[
            (7, 3),
            (0, 0),
            (7, 4),
            (0, 1),
            (7, 5),
            (0, 2),
            (7, 6),
            (0, 3),
            (5, 7),
            (0, 4),
            (6, 7),
            (0, 5), // vertical partial → (7,7) also makes vertical four
        ]);
        assert!(
            !s.is_forbidden(7 * N + 7),
            "5-in-a-row overrides double-four"
        );
    }

    #[test]
    fn test_renju_five_overrides_double_three() {
        // B at (7,3),(7,4),(7,5),(7,6) → (7,7) makes exactly 5
        // Vertical threes don't matter — 5-in-a-row takes precedence
        let s = renju(&[
            (7, 3),
            (0, 0),
            (7, 4),
            (0, 1),
            (7, 5),
            (0, 2),
            (7, 6),
            (0, 3),
            (6, 7),
            (0, 4),
            (8, 7),
            (0, 5), // vertical three setup
        ]);
        assert!(!s.is_forbidden(7 * N + 7));
    }

    #[test]
    fn test_renju_five_overrides_overline() {
        // If placing makes EXACTLY 5 (not 6), it's fine.
        // B at (7,3),(7,4),(7,5),(7,6) → (7,7) makes exactly 5 → legal!
        let s = renju(&[
            (7, 3),
            (0, 0),
            (7, 4),
            (0, 1),
            (7, 5),
            (0, 2),
            (7, 6),
            (0, 3),
        ]);
        assert!(
            !s.is_forbidden(7 * N + 7),
            "Exactly 5 should win, not be forbidden"
        );
        let s2 = s.apply_move(a(7, 7));
        assert!(s2.is_terminal());
        assert_eq!(s2.winner, 1);
    }

    // ── Renju: White 무제한 확인 ──

    #[test]
    fn test_renju_white_can_make_overline() {
        let s = renju(&[
            (14, 0),
            (8, 0),
            (14, 1),
            (8, 1),
            (14, 2),
            (8, 2),
            (14, 3),
            (8, 3),
            (14, 4),
            (8, 5),
            (14, 5),
            (8, 4),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, -1, "White 6+ should win in Renju");
    }

    #[test]
    fn test_renju_white_double_four_ok() {
        // White should have no forbidden moves at all
        // Set up white's turn with potential double-four
        let s = renju(&[
            (0, 0),
            (7, 4),
            (0, 1),
            (7, 5),
            (0, 2),
            (7, 6), // W horizontal 3
            (0, 3),
            (4, 7),
            (0, 4),
            (5, 7),
            (0, 5),
            (6, 7),   // W vertical 3
            (14, 14), // B plays somewhere far
        ]);
        // Now white plays (7,7) — would be double-four if white had restrictions
        assert!(
            !s.is_forbidden(7 * N + 7),
            "White has no forbidden moves in Renju"
        );
    }

    // ── Renju: Omok과의 차이 확인 ──

    // ── Omok vs Renju: 쌍삼 정의 차이 핵심 테스트 ──
    #[test]
    fn test_omok_vs_renju_double_three_with_white_blocking() {
        // 핵심 차이 케이스: 양쪽이 백돌로 막힌 삼
        //
        // 수평 방향: W(7,4) _(7,5) B(7,6) B(7,7)★ B(7,8) _(7,9) 가 아니라...
        // 정확히: W(7,4) 빈(7,5) 은 안 되고,
        //
        // 구체적 배치: 수평 _BBB_ 에서 양쪽 외부를 백으로 막기
        //   W(7,3)  _(7,4)  B(7,5)  B(7,6)  B(7,7)★  _(7,8)  W(7,9)
        //   5-cell window [4..8]: _(7,4) B B B _(7,8) → _BBB_
        //   외부: v[3]=W(7,3), v[9]=W(7,9)
        //   Omok: endpoint(7,4),(7,8) 빈칸 → 활삼 ○
        //   Renju: 사 확장 시 (7,3)=W, (7,9)=W → 양끝 막힘 → 활삼 ✗
        //
        // 수직: B(6,7) B(7,7)★ B(8,7) + 양쪽 열림 → 활삼 ○
        //   Omok: 수평 + 수직 = 쌍삼 → 금수
        //   Renju: 수직만 활삼 → 단삼 → 합법

        let s_omok = omok(&[
            (7, 5),
            (7, 3), // B(7,5), W(7,3) — white beyond left
            (7, 6),
            (7, 9), // B(7,6), W(7,9) — white beyond right
            (6, 7),
            (0, 0), // B(6,7) — vertical
            (8, 7),
            (0, 1), // B(8,7) — vertical
        ]);

        let s_renju = renju(&[
            (7, 5),
            (7, 3),
            (7, 6),
            (7, 9),
            (6, 7),
            (0, 0),
            (8, 7),
            (0, 1),
        ]);

        let pos = 7 * N + 7;

        assert!(
            s_omok.is_forbidden(pos),
            "Omok: W._BBB_.W 도 활삼으로 인정 → 수직과 합쳐 쌍삼 → 금수"
        );
        assert!(
            !s_renju.is_forbidden(pos),
            "Renju: W._BBB_.W 는 활사 불가능 → 수직 활삼 1개만 → 합법"
        );
    }

    #[test]
    fn test_renju_double_three_with_open_ends() {
        // 양쪽 외부가 빈칸이면 Renju에서도 활삼으로 인정 → 쌍삼 가능
        // 수평: _(7,4) B(7,5) B(7,6) B(7,7)★ _(7,8) — 외부도 빈칸
        // 수직: _(5,7) B(6,7) B(7,7)★ B(8,7) _(9,7)
        let s = renju(&[
            (7, 5),
            (0, 0),
            (7, 6),
            (0, 1), // horizontal
            (6, 7),
            (0, 2),
            (8, 7),
            (0, 3), // vertical
        ]);
        assert!(
            s.is_forbidden(7 * N + 7),
            "Renju: 양쪽 열린 활삼 2개 → 쌍삼 → 금수"
        );
    }

    // ── Renju vs Omok: 쌍사 차이 ──

    #[test]
    fn test_renju_vs_omok_double_four_difference() {
        // Double-four is forbidden in Renju but NOT in Omok
        // B forms two fours at (7,7):
        //   Horizontal: B at (7,4),(7,5),(7,6) → (7,7) = BBBB_
        //   Vertical:   B at (4,7),(5,7),(6,7) → (7,7) = BBBB|
        let moves: &[(usize, usize)] = &[
            (7, 4),
            (0, 0),
            (7, 5),
            (0, 1),
            (7, 6),
            (0, 2),
            (4, 7),
            (0, 3),
            (5, 7),
            (0, 4),
            (6, 7),
            (0, 5),
        ];
        let s_renju = make(GomokuVariant::Renju, moves);
        let s_omok = make(GomokuVariant::Omok, moves);

        assert!(
            s_renju.is_forbidden(7 * N + 7),
            "Double-four forbidden in Renju"
        );
        assert!(
            !s_omok.is_forbidden(7 * N + 7),
            "Double-four NOT forbidden in Omok"
        );
    }

    // ── Renju: MCTS 통합 ──

    #[test]
    fn test_mcts_integration_renju() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::search::FixedIterations;
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let state = Gomoku15::renju();
        let eval: Arc<dyn crate::game::Evaluator<Gomoku15> + Send + Sync> = Arc::new(UniformEval);
        let config = MctsConfig::evaluation(2.0);
        let engine = MctsEngine::new(state, eval, config);
        engine.run(&mut FixedIterations::new(100));

        let best = engine.best_move();
        assert!(best.is_some());
    }

    // ── Renju: 엣지 케이스 ──

    #[test]
    fn test_renju_four_at_edge_not_double() {
        // Four along the board edge — only one four, should be OK
        // B at (0,0),(0,1),(0,2) → (0,3) makes BBBB_ along top edge
        let s = renju(&[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]);
        assert!(!s.is_forbidden(0 * N + 3), "Single four at edge is OK");
    }

    #[test]
    fn test_renju_overline_blocks_four_count() {
        // If a "four" window has black extending beyond → overline → NOT counted as four
        // B at (7,3),(7,4),(7,5),(7,6),(7,8) → placing (7,7) makes BBBBB_B = overline
        // The 5-cell window [3..7] = BBBBB → 5 blacks, 0 empty → not a four
        // The 5-cell window [4..8] = BBBB_ → but v[3]=B means it extends to overline
        let s = renju(&[
            (7, 3),
            (0, 0),
            (7, 4),
            (0, 1),
            (7, 5),
            (0, 2),
            (7, 6),
            (0, 3),
            (7, 8),
            (0, 4),
            (4, 7),
            (0, 5),
            (5, 7),
            (0, 6),
            (6, 7),
            (0, 7), // vertical four setup
        ]);
        // (7,7) makes overline in horizontal → that horizontal "four" shouldn't count
        // Vertical makes BBBB → one four
        // So total fours = 1 (vertical only) → not double-four
        // But (7,7) IS overline → forbidden anyway (checked before double-four)
        assert!(s.is_forbidden(7 * N + 7), "Overline is still forbidden");
    }

    #[test]
    fn test_renju_diagonal_double_four() {
        // Double-four across two diagonals
        // Diagonal DR: B at (4,4),(5,5),(6,6) → (7,7) makes BBBB
        // Diagonal UR: B at (10,4),(9,5),(8,6) → (7,7) makes BBBB
        let s = renju(&[
            (4, 4),
            (0, 0),
            (5, 5),
            (0, 1),
            (6, 6),
            (0, 2), // diag DR
            (10, 4),
            (0, 3),
            (9, 5),
            (0, 4),
            (8, 6),
            (0, 5), // diag UR
        ]);
        assert!(
            s.is_forbidden(7 * N + 7),
            "Renju double-four on diagonals should be forbidden"
        );
    }

    #[test]
    fn test_renju_exactly_five_wins() {
        // Verify that exactly 5 wins in Renju (same as Omok)
        let s = renju(&[
            (7, 5),
            (0, 0),
            (7, 6),
            (0, 1),
            (7, 7),
            (0, 2),
            (7, 8),
            (0, 3),
            (7, 9),
        ]);
        assert!(s.is_terminal());
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_renju_variant_preserved() {
        let s = Gomoku15::renju();
        let s2 = s.apply_move(112);
        assert_eq!(s2.rules, GomokuVariant::Renju);
    }

    #[test]
    fn test_encode_planes_matches_reference_and_occupied_consistency_across_variants() {
        let states = [
            Gomoku15::freestyle(),
            std(&[(7, 5), (0, 0), (7, 6), (0, 1), (7, 7)]),
            free(&[(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8)]),
            omok(&[
                (7, 6),
                (0, 0),
                (7, 8),
                (0, 1),
                (6, 7),
                (0, 2),
                (8, 7),
                (0, 3),
            ]),
            renju(&[
                (7, 5),
                (0, 0),
                (7, 6),
                (0, 1),
                (7, 8),
                (0, 2),
                (6, 7),
                (0, 3),
            ]),
            make(
                GomokuVariant::Caro,
                &[(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8)],
            ),
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
            1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let rebuilt = Gomoku15::from_board(&board, -1, GomokuVariant::Omok);
        assert_occupied_matches_board(&rebuilt);
        let expected = encode_planes_reference(&rebuilt);
        let mut scratch = Vec::new();
        rebuilt.encode_planes_into(&mut scratch);
        assert_eq!(scratch, expected);
    }

    #[test]
    fn test_freestyle_overline_wins() {
        let s = make(
            GomokuVariant::Freestyle,
            &[
                (7, 0),
                (0, 0),
                (7, 1),
                (0, 1),
                (7, 2),
                (0, 2),
                (7, 3),
                (0, 3),
                (7, 4),
                (0, 4),
                (7, 5),
            ],
        );
        assert!(s.is_terminal(), "freestyle overline should win");
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_gomocup_standard_overline_does_not_win() {
        let mut board = vec![0i8; N2];
        for c in 0..=5 {
            board[7 * N + c] = 1;
        }
        let s = Gomoku15::from_board(&board, 1, GomokuVariant::Standard);
        assert!(
            !s.is_terminal(),
            "Gomocup standard should require exactly five"
        );
    }

    #[test]
    fn test_caro_blocked_five_is_not_win() {
        let s = make(
            GomokuVariant::Caro,
            &[
                (7, 5),
                (7, 4),
                (7, 6),
                (7, 10),
                (7, 7),
                (7, 11),
                (7, 8),
                (7, 12),
                (7, 9),
            ],
        );
        assert!(
            !s.is_terminal(),
            "blocked-both-ends five should not win in caro"
        );
    }

    #[test]
    fn test_caro_open_five_wins() {
        let s = make(
            GomokuVariant::Caro,
            &[
                (7, 5),
                (0, 0),
                (7, 6),
                (0, 1),
                (7, 7),
                (0, 2),
                (7, 8),
                (0, 3),
                (7, 9),
            ],
        );
        assert!(s.is_terminal(), "open-ended five should win in caro");
        assert_eq!(s.winner, 1);
    }

    #[test]
    fn test_caro_exact_six_is_not_win() {
        let mut board = vec![0i8; N2];
        for c in 0..=5 {
            board[7 * N + c] = 1;
        }
        let s = Gomoku15::from_board(&board, 1, GomokuVariant::Caro);
        assert!(!s.is_terminal(), "exactly six should not win in caro");
    }

    #[test]
    fn test_caro_seven_is_not_win() {
        let mut board = vec![0i8; N2];
        for c in 0..=6 {
            board[7 * N + c] = 1;
        }
        let s = Gomoku15::from_board(&board, 1, GomokuVariant::Caro);
        assert!(!s.is_terminal(), "seven-in-a-row should not win in caro");
    }

    #[test]
    fn test_omok_kr_variant_preserved() {
        let s = Gomoku15::omok();
        assert_eq!(s.rules, GomokuVariant::Omok);
    }

    #[test]
    fn test_renju_200_move_limit_is_draw() {
        let mut board = vec![0i8; N2];
        let mut placed = 0usize;
        for r in 0..N {
            for c in 0..N {
                if placed >= 200 {
                    break;
                }
                board[r * N + c] = if ((c / 2) + (r % 2)) % 2 == 0 { 1 } else { -1 };
                placed += 1;
            }
            if placed >= 200 {
                break;
            }
        }

        let s = Gomoku15::from_board(&board, 1, GomokuVariant::Renju);
        assert_eq!(s.moves as usize, 200);
        assert_eq!(s.winner, 0, "200-move cap test must not contain a winner");
        assert!(s.is_terminal(), "Renju should auto-draw at 200 moves");
        assert_eq!(s.outcome(), 0.0);
        assert!(
            s.legal_moves().is_empty(),
            "drawn position should not continue"
        );
    }

    fn bench_gomoku_state(mut state: Gomoku, label: &str) {
        for &mv in &[112usize, 113, 127, 128, 142, 143, 157, 158] {
            if state.idx_to_move(mv).is_some() && state.legal_moves().contains(&mv) {
                state = state.apply_move(mv);
            }
        }
        let loops = bench_loops(20_000);

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.clone());
        }
        let clone_ms = start.elapsed().as_secs_f64() * 1000.0;

        let legal = state.legal_moves();
        let mv = legal[legal.len() / 2];
        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.apply_move(mv));
        }
        let apply_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.encode_planes());
        }
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut scratch = Vec::new();
        let start = Instant::now();
        for _ in 0..loops {
            state.encode_planes_into(&mut scratch);
            black_box(&scratch);
        }
        let encode_reuse_ms = start.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
                "[bench] {label}: clone={clone_ms:.2}ms apply={apply_ms:.2}ms encode={encode_ms:.2}ms encode_reuse={encode_reuse_ms:.2}ms loops={loops}"
            );
    }

    fn bench_gomoku15_state(mut state: Gomoku15, label: &str) {
        for &mv in &[112u16, 113, 127, 128, 142, 143, 157, 158] {
            if state.legal_moves().contains(&mv) {
                state = state.apply_move(mv);
            }
        }
        let loops = bench_loops(20_000);

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.clone());
        }
        let clone_ms = start.elapsed().as_secs_f64() * 1000.0;

        let legal = state.legal_moves();
        let mv = legal[legal.len() / 2];
        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.apply_move(mv));
        }
        let apply_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.encode_planes());
        }
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut scratch = Vec::new();
        let start = Instant::now();
        for _ in 0..loops {
            state.encode_planes_into(&mut scratch);
            black_box(&scratch);
        }
        let encode_reuse_ms = start.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
                "[bench] {label}: clone={clone_ms:.2}ms apply={apply_ms:.2}ms encode={encode_ms:.2}ms encode_reuse={encode_reuse_ms:.2}ms loops={loops}"
            );
    }

    fn bench_ttt_state(mut state: TicTacToe, label: &str) {
        for &mv in &[4usize, 0, 8, 2] {
            state = state.apply_move(mv);
        }
        let loops = bench_loops(50_000);

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.clone());
        }
        let clone_ms = start.elapsed().as_secs_f64() * 1000.0;

        let legal = state.legal_moves();
        let mv = legal[0];
        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.apply_move(mv));
        }
        let apply_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.encode_planes());
        }
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut scratch = Vec::new();
        let start = Instant::now();
        for _ in 0..loops {
            state.encode_planes_into(&mut scratch);
            black_box(&scratch);
        }
        let encode_reuse_ms = start.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
                "[bench] {label}: clone={clone_ms:.2}ms apply={apply_ms:.2}ms encode={encode_ms:.2}ms encode_reuse={encode_reuse_ms:.2}ms loops={loops}"
            );
    }

    #[test]
    #[ignore]
    fn bench_gomoku_hotpaths() {
        bench_gomoku_state(Gomoku::new(15), "gomoku15-generic");
    }

    #[test]
    #[ignore]
    fn bench_gomoku15_hotpaths() {
        bench_gomoku15_state(Gomoku15::freestyle(), "gomoku15-freestyle");
    }

    #[test]
    #[ignore]
    fn bench_gomoku15_renju_hotpaths() {
        bench_gomoku15_state(Gomoku15::renju(), "gomoku15-renju");
    }

    #[test]
    #[ignore]
    fn bench_tictactoe_hotpaths() {
        bench_ttt_state(TicTacToe::initial(), "tictactoe");
    }
}

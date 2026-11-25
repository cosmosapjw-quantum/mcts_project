//! Go — Board game engine (9×9, 13×13, 19×19)
//!
//! 핵심 설계:
//!   - [u8; 361] board: 0=empty, 1=black, 2=white (max 19×19)
//!   - BFS liberty counting (simple, correct — optimize to union-find later)
//!   - Simple ko (ko_point) + ruleset-aware repetition handling
//!   - Tromp-Taylor scoring (Chinese rules, area counting)
//!   - Pass = action N*N, two consecutive passes → terminal
//!   - apply_move(&self, mv) -> Self: pure function
//!
//! 좌표계: pos = row * size + col, 0-indexed
//! 플레이어: Black=1, White=2 (board), +1/-1 (GameState convention)

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, LazyLock};

use crate::game::{tt_combine, tt_mix64, GameState};

// ═══════════════════════════════════════════════════════
// § Constants
// ═══════════════════════════════════════════════════════

const MAX_SZ: usize = 19;
const MAX_N2: usize = MAX_SZ * MAX_SZ; // 361
const GO_HISTORY_DIGEST_SEED: u64 = 0x4d31_aa72_61c5_f09b;
const GO_TT_HASH_SEED: u64 = 0x85b8_bf34_6f2d_91c3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoScoring {
    Area,
    Territory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoRuleset {
    Chinese,
    Japanese,
    Korean,
}

impl GoRuleset {
    pub fn scoring(self) -> GoScoring {
        match self {
            GoRuleset::Chinese => GoScoring::Area,
            GoRuleset::Japanese | GoRuleset::Korean => GoScoring::Territory,
        }
    }
}

// ═══════════════════════════════════════════════════════
// § Zobrist
// ═══════════════════════════════════════════════════════

struct GoZob {
    piece: [[u64; MAX_N2]; 2], // [0=black, 1=white][pos]
    side: u64,
}

impl GoZob {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut piece = [[0u64; MAX_N2]; 2];
        for c in 0..2 {
            for p in 0..MAX_N2 {
                piece[c][p] = rng.gen();
            }
        }
        GoZob {
            piece,
            side: rng.gen(),
        }
    }
}

// SAFETY: GoZob is read-only after init and contains only plain data.
unsafe impl Sync for GoZob {}

static GZOB: LazyLock<GoZob> = LazyLock::new(|| GoZob::new(0xBADC_AFE0_6060_1234));

thread_local! {
    static GO_SCRATCH: RefCell<GoScratch> = RefCell::new(GoScratch::default());
}

#[derive(Clone)]
struct GoScratch {
    epoch: u32,
    visited: [u32; MAX_N2],
    libs: [u32; MAX_N2],
    checked: [u32; MAX_N2],
    stack: [usize; MAX_N2],
    group_stones: [usize; MAX_N2],
}

impl Default for GoScratch {
    fn default() -> Self {
        Self {
            epoch: 1,
            visited: [0; MAX_N2],
            libs: [0; MAX_N2],
            checked: [0; MAX_N2],
            stack: [0; MAX_N2],
            group_stones: [0; MAX_N2],
        }
    }
}

impl GoScratch {
    #[inline]
    fn next_epoch(&mut self) -> u32 {
        let next = self.epoch.wrapping_add(1);
        if next == 0 {
            self.visited.fill(0);
            self.libs.fill(0);
            self.epoch = 1;
        } else {
            self.epoch = next;
        }
        self.epoch
    }
}

// ═══════════════════════════════════════════════════════
// § Board history ring buffer (zero-alloc)
// ═══════════════════════════════════════════════════════

const GO_HISTORY_LEN: usize = 8;
const GO_HISTORY_SLOTS: usize = GO_HISTORY_LEN - 1; // 7 past snapshots

/// Fixed-size ring buffer for board snapshots. Replaces Vec<Vec<u8>>.
/// Copy semantics — no heap allocation.
#[derive(Clone)]
struct BoardRing {
    boards: [[u8; MAX_N2]; GO_HISTORY_SLOTS],
    len: u8,  // number of valid entries (0..=7)
    head: u8, // index of most recent entry
}

#[derive(Clone)]
struct BoardRingUndo {
    slot: u8,
    prev_len: u8,
    prev_head: u8,
    prev_board: [u8; MAX_N2],
}

impl BoardRing {
    fn new() -> Self {
        BoardRing {
            boards: [[0u8; MAX_N2]; GO_HISTORY_SLOTS],
            len: 0,
            head: 0,
        }
    }

    /// Push a board snapshot. Oldest entry is overwritten when full.
    #[inline]
    fn push(&mut self, board: &[u8; MAX_N2], n2: usize) {
        let slot = if self.len == 0 {
            0
        } else {
            (self.head as usize + 1) % GO_HISTORY_SLOTS
        };
        self.boards[slot][..n2].copy_from_slice(&board[..n2]);
        self.head = slot as u8;
        if (self.len as usize) < GO_HISTORY_SLOTS {
            self.len += 1;
        }
    }

    #[inline]
    fn push_with_undo(&mut self, board: &[u8; MAX_N2], n2: usize) -> BoardRingUndo {
        let slot = if self.len == 0 {
            0
        } else {
            (self.head as usize + 1) % GO_HISTORY_SLOTS
        };
        let undo = BoardRingUndo {
            slot: slot as u8,
            prev_len: self.len,
            prev_head: self.head,
            prev_board: self.boards[slot],
        };
        self.boards[slot][..n2].copy_from_slice(&board[..n2]);
        self.head = slot as u8;
        if (self.len as usize) < GO_HISTORY_SLOTS {
            self.len += 1;
        }
        undo
    }

    #[inline]
    fn undo_push(&mut self, undo: BoardRingUndo) {
        self.boards[undo.slot as usize] = undo.prev_board;
        self.len = undo.prev_len;
        self.head = undo.prev_head;
    }

    /// Iterate past boards from most recent to oldest.
    #[inline]
    fn iter_rev(&self) -> BoardRingIter<'_> {
        BoardRingIter {
            ring: self,
            remaining: self.len as usize,
            cur: self.head as usize,
        }
    }
}

struct BoardRingIter<'a> {
    ring: &'a BoardRing,
    remaining: usize,
    cur: usize,
}

impl<'a> Iterator for BoardRingIter<'a> {
    type Item = &'a [u8; MAX_N2];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let board = &self.ring.boards[self.cur];
        self.remaining -= 1;
        self.cur = if self.cur == 0 {
            GO_HISTORY_SLOTS - 1
        } else {
            self.cur - 1
        };
        Some(board)
    }
}

// ═══════════════════════════════════════════════════════
// § Go feature scratch (sparse encode_planes)
// ═══════════════════════════════════════════════════════

struct GoFeatureScratch {
    touched: Vec<usize>,
    color_plane_filled: bool,
}

impl Default for GoFeatureScratch {
    fn default() -> Self {
        Self {
            touched: Vec::with_capacity(2048),
            color_plane_filled: false,
        }
    }
}

thread_local! {
    static GO_FEATURE_SCRATCH: RefCell<GoFeatureScratch> = RefCell::new(GoFeatureScratch::default());
}

// ═══════════════════════════════════════════════════════
// § Go state
// ═══════════════════════════════════════════════════════

#[derive(Clone)]
pub struct Go {
    board: [u8; MAX_N2], // 0=empty, 1=black, 2=white
    pub size: usize,
    side: u8,        // 1=black, 2=white
    ko_point: u16,   // forbidden recapture point, u16::MAX = none
    passes: u8,      // consecutive passes (0, 1, or 2)
    black_caps: u16, // black's captured stones (prisoners)
    white_caps: u16,
    komi: f32, // compensation for white (e.g., 7.5)
    ruleset: GoRuleset,
    scoring: GoScoring,
    hash: u64,
    move_count: u16,
    // Position history: board + side-to-move hash (Arc for cheap clone)
    hash_history: Arc<Vec<u64>>,
    history_digest: u64,
    allow_suicide: bool,
    cycle_terminal: bool,
    /// Past board snapshots for AlphaZero-style history encoding.
    board_ring: BoardRing,
}

#[derive(Clone)]
pub struct GoUndo {
    prev_side: u8,
    prev_ko_point: u16,
    prev_passes: u8,
    prev_black_caps: u16,
    prev_white_caps: u16,
    prev_hash: u64,
    prev_move_count: u16,
    prev_history_digest: u64,
    prev_cycle_terminal: bool,
    prev_hash_history: Arc<Vec<u64>>,
    board_ring_undo: BoardRingUndo,
    board_changes: SmallVec<[(u16, u8); 16]>,
}

impl Go {
    #[cfg(test)]
    pub fn new(size: usize, komi: f32) -> Self {
        Self::new_with_options(size, komi, GoRuleset::Chinese, GoScoring::Area, false)
    }

    #[inline]
    fn history_digest_for_hash(position_hash: u64) -> u64 {
        tt_mix64(GO_HISTORY_DIGEST_SEED ^ tt_mix64(position_hash))
    }

    #[inline]
    fn history_digest_insert(history_digest: u64, position_hash: u64) -> u64 {
        history_digest ^ tt_mix64(position_hash.wrapping_add(GO_HISTORY_DIGEST_SEED))
    }

    pub fn new_with_rules(size: usize, komi: f32, ruleset: GoRuleset) -> Self {
        Self::new_with_options(size, komi, ruleset, ruleset.scoring(), false)
    }

    pub fn new_with_options(
        size: usize,
        komi: f32,
        ruleset: GoRuleset,
        scoring: GoScoring,
        allow_suicide: bool,
    ) -> Self {
        assert!(size <= MAX_SZ && size >= 2);
        let mut g = Go {
            board: [0u8; MAX_N2],
            size,
            side: 1,
            ko_point: u16::MAX,
            passes: 0,
            black_caps: 0,
            white_caps: 0,
            komi,
            ruleset,
            scoring,
            hash: 0,
            move_count: 0,
            hash_history: Arc::new(Vec::with_capacity(size * size * 2)),
            history_digest: 0,
            allow_suicide,
            cycle_terminal: false,
            board_ring: BoardRing::new(),
        };
        Arc::make_mut(&mut g.hash_history).push(g.hash);
        g.history_digest = Self::history_digest_for_hash(g.hash);
        g
    }

    pub fn new_9x9() -> Self {
        Self::new_with_rules(9, 7.5, GoRuleset::Chinese)
    }
    #[cfg(test)]
    pub fn new_13x13() -> Self {
        Self::new_with_rules(13, 7.5, GoRuleset::Chinese)
    }
    #[cfg(test)]
    pub fn new_19x19() -> Self {
        Self::new_with_rules(19, 7.5, GoRuleset::Chinese)
    }

    /// Reconstruct state from board array (0=empty, 1=black, 2=white) + side to move.
    #[cfg(test)]
    pub fn from_board(size: usize, komi: f32, board: &[u8], side: u8) -> Self {
        Self::from_board_with_options(
            size,
            komi,
            board,
            side,
            GoRuleset::Chinese,
            GoScoring::Area,
            false,
            0,
            None,
            0,
            0,
        )
    }

    pub fn from_board_with_options(
        size: usize,
        komi: f32,
        board: &[u8],
        side: u8,
        ruleset: GoRuleset,
        scoring: GoScoring,
        allow_suicide: bool,
        passes: u8,
        ko_point: Option<u16>,
        black_caps: u16,
        white_caps: u16,
    ) -> Self {
        let mut g = Self::new_with_options(size, komi, ruleset, scoring, allow_suicide);
        let n2 = size * size;
        for i in 0..n2.min(board.len()) {
            g.board[i] = board[i];
        }
        g.side = if side == 2 { 2 } else { 1 };
        g.passes = passes.min(2);
        g.ko_point = ko_point.unwrap_or(u16::MAX);
        g.black_caps = black_caps;
        g.white_caps = white_caps;
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..n2 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
            if g.side == 2 {
                g.hash ^= z.side;
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Self::history_digest_for_hash(g.hash);
        g
    }

    fn n2(&self) -> usize {
        self.size * self.size
    }
    pub fn pass_action(&self) -> u16 {
        self.n2() as u16
    }

    fn repeats_position_hash(&self, next_hash: u64) -> bool {
        self.hash_history.iter().any(|&seen| seen == next_hash)
    }

    fn finalize_transition(&self, next: &mut Self) {
        let repeated = self.repeats_position_hash(next.hash);
        if matches!(self.ruleset, GoRuleset::Japanese | GoRuleset::Korean)
            && repeated
            && next.passes < 2
        {
            next.cycle_terminal = true;
        }
        if !repeated {
            next.history_digest = Self::history_digest_insert(self.history_digest, next.hash);
        }
        Arc::make_mut(&mut next.hash_history).push(next.hash);
    }

    fn finalize_transition_from_previous(
        &mut self,
        prev_hash_history: &Arc<Vec<u64>>,
        prev_history_digest: u64,
    ) {
        let repeated = prev_hash_history.iter().any(|&seen| seen == self.hash);
        if matches!(self.ruleset, GoRuleset::Japanese | GoRuleset::Korean)
            && repeated
            && self.passes < 2
        {
            self.cycle_terminal = true;
        }
        self.history_digest = if repeated {
            prev_history_digest
        } else {
            Self::history_digest_insert(prev_history_digest, self.hash)
        };
        Arc::make_mut(&mut self.hash_history).push(self.hash);
    }

    fn row(&self, pos: usize) -> usize {
        pos / self.size
    }
    fn col(&self, pos: usize) -> usize {
        pos % self.size
    }

    fn neighbors(&self, pos: usize) -> impl Iterator<Item = usize> + '_ {
        let r = self.row(pos);
        let c = self.col(pos);
        let sz = self.size;
        let mut nbrs = [usize::MAX; 4];
        let mut n = 0;
        if r > 0 {
            nbrs[n] = pos - sz;
            n += 1;
        }
        if r + 1 < sz {
            nbrs[n] = pos + sz;
            n += 1;
        }
        if c > 0 {
            nbrs[n] = pos - 1;
            n += 1;
        }
        if c + 1 < sz {
            nbrs[n] = pos + 1;
            n += 1;
        }
        (0..n).map(move |i| nbrs[i])
    }

    #[inline]
    fn neighbor_flags(&self, pos: usize, opp: u8) -> (bool, bool) {
        let r = pos / self.size;
        let c = pos - r * self.size;
        let sz = self.size;
        let board = &self.board;
        let mut has_empty = false;
        let mut has_opp = false;

        #[inline(always)]
        fn observe(cell: u8, opp: u8, has_empty: &mut bool, has_opp: &mut bool) {
            if cell == 0 {
                *has_empty = true;
            } else if cell == opp {
                *has_opp = true;
            }
        }

        if r > 0 {
            observe(board[pos - sz], opp, &mut has_empty, &mut has_opp);
        }
        if r + 1 < sz {
            observe(board[pos + sz], opp, &mut has_empty, &mut has_opp);
        }
        if c > 0 {
            observe(board[pos - 1], opp, &mut has_empty, &mut has_opp);
        }
        if c + 1 < sz {
            observe(board[pos + 1], opp, &mut has_empty, &mut has_opp);
        }
        (has_empty, has_opp)
    }

    fn opp(color: u8) -> u8 {
        3 - color
    }

    #[inline]
    fn record_board_change(
        changes: &mut SmallVec<[(u16, u8); 16]>,
        board: &[u8; MAX_N2],
        pos: usize,
    ) {
        changes.push((pos as u16, board[pos]));
    }

    #[inline]
    fn clone_core(&self, hash_history: Arc<Vec<u64>>, board_ring: BoardRing) -> Self {
        Go {
            board: self.board,
            size: self.size,
            side: self.side,
            ko_point: self.ko_point,
            passes: self.passes,
            black_caps: self.black_caps,
            white_caps: self.white_caps,
            komi: self.komi,
            ruleset: self.ruleset,
            scoring: self.scoring,
            hash: self.hash,
            move_count: self.move_count,
            hash_history,
            history_digest: self.history_digest,
            allow_suicide: self.allow_suicide,
            cycle_terminal: self.cycle_terminal,
            board_ring,
        }
    }

    // ── BFS liberty / group detection ──

    /// Count liberties of the group containing `pos`. Returns 0 if `pos` is empty.
    #[cfg(test)]
    fn count_liberties(&self, pos: usize) -> u32 {
        self.group_liberties_on_board(&self.board, pos)
    }

    fn group_liberties_with_scratch(
        &self,
        board: &[u8; MAX_N2],
        pos: usize,
        scratch: &mut GoScratch,
    ) -> u32 {
        let color = board[pos];
        if color == 0 {
            return 0;
        }
        let visit_epoch = scratch.next_epoch();
        let lib_epoch = scratch.next_epoch();
        let GoScratch {
            stack,
            visited,
            libs,
            ..
        } = scratch;
        let mut top = 0usize;
        stack[top] = pos;
        top += 1;
        visited[pos] = visit_epoch;
        let mut lib_count = 0u32;
        while top > 0 {
            top -= 1;
            let p = stack[top];
            for nb in self.neighbors(p) {
                if visited[nb] == visit_epoch {
                    continue;
                }
                if board[nb] == color {
                    visited[nb] = visit_epoch;
                    stack[top] = nb;
                    top += 1;
                } else if board[nb] == 0 && libs[nb] != lib_epoch {
                    libs[nb] = lib_epoch;
                    lib_count += 1;
                }
            }
        }
        lib_count
    }

    fn group_liberties_on_board(&self, board: &[u8; MAX_N2], pos: usize) -> u32 {
        GO_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            self.group_liberties_with_scratch(board, pos, &mut scratch)
        })
    }

    fn group_stones_and_liberties_with_scratch(
        &self,
        board: &[u8; MAX_N2],
        pos: usize,
        scratch: &mut GoScratch,
    ) -> (usize, u32) {
        let color = board[pos];
        if color == 0 {
            return (0, 0);
        }
        let visit_epoch = scratch.next_epoch();
        let lib_epoch = scratch.next_epoch();
        let GoScratch {
            stack,
            visited,
            libs,
            group_stones,
            ..
        } = scratch;
        let mut top = 0usize;
        stack[top] = pos;
        top += 1;
        visited[pos] = visit_epoch;
        let mut stone_count = 0usize;
        let mut lib_count = 0u32;
        while top > 0 {
            top -= 1;
            let p = stack[top];
            group_stones[stone_count] = p;
            stone_count += 1;
            for nb in self.neighbors(p) {
                if board[nb] == color && visited[nb] != visit_epoch {
                    visited[nb] = visit_epoch;
                    stack[top] = nb;
                    top += 1;
                } else if board[nb] == 0 && libs[nb] != lib_epoch {
                    libs[nb] = lib_epoch;
                    lib_count += 1;
                }
            }
        }
        (stone_count, lib_count)
    }

    fn group_stones_and_liberties_on_board(
        &self,
        board: &[u8; MAX_N2],
        pos: usize,
        stones_out: &mut [usize; MAX_N2],
    ) -> (usize, u32) {
        GO_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let (stone_count, lib_count) =
                self.group_stones_and_liberties_with_scratch(board, pos, &mut scratch);
            stones_out[..stone_count].copy_from_slice(&scratch.group_stones[..stone_count]);
            (stone_count, lib_count)
        })
    }

    /// Remove a group and return the number of stones removed.
    fn remove_group(&mut self, pos: usize) -> u16 {
        let mut stones = [0usize; MAX_N2];
        let (stone_count, _) =
            self.group_stones_and_liberties_on_board(&self.board, pos, &mut stones);
        let count = stone_count as u16;
        let z = &*GZOB;
        for &s in &stones[..stone_count] {
            let color = self.board[s];
            self.hash ^= z.piece[(color - 1) as usize][s];
            self.board[s] = 0;
        }
        count
    }

    // ── Legal move check ──

    pub fn is_legal(&self, pos: usize) -> bool {
        if self.cycle_terminal {
            return false;
        }
        if pos >= self.n2() {
            return false;
        }
        if self.board[pos] != 0 {
            return false;
        }
        if pos as u16 == self.ko_point {
            return false;
        }

        let color = self.side;
        let opp = Self::opp(color);

        let mut basic_legal = false;
        let mut has_empty_neighbor = false;
        let mut capture_possible = false;

        // Fast path 1: if ANY neighbor is empty → guaranteed ≥1 liberty
        for nb in self.neighbors(pos) {
            if self.board[nb] == 0 {
                has_empty_neighbor = true;
                basic_legal = true;
            }
        }

        // Merged GO_SCRATCH.with: check opponent captures + friendly groups in one call
        if !basic_legal {
            GO_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                // Phase 1: opponent neighbor with exactly 1 liberty → capture
                let checked_epoch = scratch.next_epoch();
                for nb in self.neighbors(pos) {
                    if self.board[nb] != opp || scratch.checked[nb] == checked_epoch {
                        continue;
                    }
                    let (stone_count, lib_count) =
                        self.group_stones_and_liberties_with_scratch(&self.board, nb, &mut scratch);
                    for idx in 0..stone_count {
                        let s = scratch.group_stones[idx];
                        scratch.checked[s] = checked_epoch;
                    }
                    if lib_count == 1 {
                        capture_possible = true;
                        basic_legal = true;
                        return;
                    }
                }
                // Phase 2: friendly neighbor's group with >1 liberty
                let checked_epoch2 = scratch.next_epoch();
                for nb in self.neighbors(pos) {
                    if self.board[nb] != color || scratch.checked[nb] == checked_epoch2 {
                        continue;
                    }
                    let (stone_count, lib_count) =
                        self.group_stones_and_liberties_with_scratch(&self.board, nb, &mut scratch);
                    for idx in 0..stone_count {
                        let s = scratch.group_stones[idx];
                        scratch.checked[s] = checked_epoch2;
                    }
                    if lib_count > 1 {
                        basic_legal = true;
                        return;
                    }
                }
            });
        }

        if !basic_legal {
            basic_legal = self.allow_suicide;
        }
        if !basic_legal {
            return false;
        }

        // Chinese superko: hash-only check (no full state construction)
        if self.ruleset == GoRuleset::Chinese {
            let next_hash = if has_empty_neighbor && !capture_possible {
                let z = &*GZOB;
                self.hash ^ z.piece[(color - 1) as usize][pos] ^ z.side
            } else {
                self.compute_hash_after_place(pos)
            };
            if self.repeats_position_hash(next_hash) {
                return false;
            }
        }
        true
    }

    /// Compute the Zobrist hash that would result from placing a stone at `pos`,
    /// performing captures, and switching sides — without constructing a full Go state.
    fn compute_hash_after_place(&self, pos: usize) -> u64 {
        let color = self.side;
        let opp = Self::opp(color);
        let z = &*GZOB;
        let mut hash = self.hash;
        let mut board = self.board; // stack copy

        board[pos] = color;
        hash ^= z.piece[(color - 1) as usize][pos];

        GO_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let checked_epoch = scratch.next_epoch();
            for nb in self.neighbors(pos) {
                if board[nb] != opp || scratch.checked[nb] == checked_epoch {
                    continue;
                }
                let (stone_count, lib_count) =
                    self.group_stones_and_liberties_with_scratch(&board, nb, &mut scratch);
                for idx in 0..stone_count {
                    let s = scratch.group_stones[idx];
                    scratch.checked[s] = checked_epoch;
                }
                if lib_count == 0 {
                    for idx in 0..stone_count {
                        let stone = scratch.group_stones[idx];
                        hash ^= z.piece[(opp - 1) as usize][stone];
                        board[stone] = 0;
                    }
                }
            }
            // Check self-capture (suicide)
            let libs = self.group_liberties_with_scratch(&board, pos, &mut scratch);
            if libs == 0 {
                let (sc, _) =
                    self.group_stones_and_liberties_with_scratch(&board, pos, &mut scratch);
                for idx in 0..sc {
                    let s = scratch.group_stones[idx];
                    hash ^= z.piece[(color - 1) as usize][s];
                }
            }
        });

        // Switch sides
        hash ^= z.side;
        hash
    }

    /// Internal: place stone and handle captures. Returns new state.
    /// Optimized: single GZOB access (static), single GO_SCRATCH.with(),
    /// single BFS for ko+suicide check.
    fn do_place(&self, pos: usize) -> Self {
        // Use dummy histories — caller (apply_move) sets them properly
        let mut next = self.clone_core(Arc::new(Vec::new()), BoardRing::new());
        let color = self.side;
        let opp = Self::opp(color);
        let z = &*GZOB;

        // Place stone
        next.board[pos] = color;
        next.hash ^= z.piece[(color - 1) as usize][pos];

        // All BFS work in one GO_SCRATCH.with() call
        GO_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();

            // -- Capture phase --
            let mut total_captured = 0u16;
            let mut single_cap = u16::MAX;
            let checked_epoch = scratch.next_epoch();
            for nb in self.neighbors(pos) {
                if next.board[nb] != opp || scratch.checked[nb] == checked_epoch {
                    continue;
                }
                let (stone_count, lib_count) =
                    next.group_stones_and_liberties_with_scratch(&next.board, nb, &mut scratch);
                for idx in 0..stone_count {
                    let s = scratch.group_stones[idx];
                    scratch.checked[s] = checked_epoch;
                }
                if lib_count == 0 {
                    total_captured += stone_count as u16;
                    if stone_count == 1 {
                        single_cap = scratch.group_stones[0] as u16;
                    }
                    for idx in 0..stone_count {
                        let stone = scratch.group_stones[idx];
                        next.hash ^= z.piece[(opp - 1) as usize][stone];
                        next.board[stone] = 0;
                    }
                }
            }

            // Record captures
            if color == 1 {
                next.black_caps += total_captured;
            } else {
                next.white_caps += total_captured;
            }

            // -- Single BFS for placed stone's liberties (ko + suicide) --
            let placed_libs = next.group_liberties_with_scratch(&next.board, pos, &mut scratch);

            // Ko check
            next.ko_point = u16::MAX;
            if total_captured == 1 && placed_libs == 1 {
                next.ko_point = single_cap;
            }

            // Suicide check
            if placed_libs == 0 {
                let (sc, _) =
                    next.group_stones_and_liberties_with_scratch(&next.board, pos, &mut scratch);
                let n = sc as u16;
                for idx in 0..sc {
                    let s = scratch.group_stones[idx];
                    let c = next.board[s];
                    next.hash ^= z.piece[(c - 1) as usize][s];
                    next.board[s] = 0;
                }
                if color == 1 {
                    next.white_caps += n;
                } else {
                    next.black_caps += n;
                }
            }
        });

        next
    }

    fn apply_move_mut_internal(&mut self, mv: u16) -> GoUndo {
        let n2 = self.n2();
        let prev_hash_history = Arc::clone(&self.hash_history);
        let prev_history_digest = self.history_digest;
        let board_ring_undo = self.board_ring.push_with_undo(&self.board, n2);
        let mut undo = GoUndo {
            prev_side: self.side,
            prev_ko_point: self.ko_point,
            prev_passes: self.passes,
            prev_black_caps: self.black_caps,
            prev_white_caps: self.white_caps,
            prev_hash: self.hash,
            prev_move_count: self.move_count,
            prev_history_digest,
            prev_cycle_terminal: self.cycle_terminal,
            prev_hash_history,
            board_ring_undo,
            board_changes: SmallVec::new(),
        };

        let z = &*GZOB;
        if mv == n2 as u16 {
            self.passes += 1;
            self.ko_point = u16::MAX;
            self.side = Self::opp(self.side);
            self.hash ^= z.side;
            self.move_count += 1;
            self.finalize_transition_from_previous(
                &undo.prev_hash_history,
                undo.prev_history_digest,
            );
            return undo;
        }

        let pos = mv as usize;
        let color = self.side;
        let opp = Self::opp(color);

        Self::record_board_change(&mut undo.board_changes, &self.board, pos);
        self.board[pos] = color;
        self.hash ^= z.piece[(color - 1) as usize][pos];

        let mut neighbors = [usize::MAX; 4];
        let mut neighbor_count = 0usize;
        for nb in self.neighbors(pos) {
            neighbors[neighbor_count] = nb;
            neighbor_count += 1;
        }

        GO_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();

            let mut total_captured = 0u16;
            let mut single_cap = u16::MAX;
            let checked_epoch = scratch.next_epoch();
            for &nb in &neighbors[..neighbor_count] {
                if self.board[nb] != opp || scratch.checked[nb] == checked_epoch {
                    continue;
                }
                let (stone_count, lib_count) =
                    self.group_stones_and_liberties_with_scratch(&self.board, nb, &mut scratch);
                for idx in 0..stone_count {
                    let s = scratch.group_stones[idx];
                    scratch.checked[s] = checked_epoch;
                }
                if lib_count == 0 {
                    total_captured += stone_count as u16;
                    if stone_count == 1 {
                        single_cap = scratch.group_stones[0] as u16;
                    }
                    for idx in 0..stone_count {
                        let stone = scratch.group_stones[idx];
                        Self::record_board_change(&mut undo.board_changes, &self.board, stone);
                        self.hash ^= z.piece[(opp - 1) as usize][stone];
                        self.board[stone] = 0;
                    }
                }
            }

            if color == 1 {
                self.black_caps += total_captured;
            } else {
                self.white_caps += total_captured;
            }

            let placed_libs = self.group_liberties_with_scratch(&self.board, pos, &mut scratch);
            self.ko_point = u16::MAX;
            if total_captured == 1 && placed_libs == 1 {
                self.ko_point = single_cap;
            }

            if placed_libs == 0 {
                let (sc, _) =
                    self.group_stones_and_liberties_with_scratch(&self.board, pos, &mut scratch);
                let n = sc as u16;
                for idx in 0..sc {
                    let stone = scratch.group_stones[idx];
                    let captured_color = self.board[stone];
                    Self::record_board_change(&mut undo.board_changes, &self.board, stone);
                    self.hash ^= z.piece[(captured_color - 1) as usize][stone];
                    self.board[stone] = 0;
                }
                if color == 1 {
                    self.white_caps += n;
                } else {
                    self.black_caps += n;
                }
            }
        });

        self.passes = 0;
        self.side = Self::opp(color);
        self.hash ^= z.side;
        self.move_count += 1;
        self.finalize_transition_from_previous(&undo.prev_hash_history, undo.prev_history_digest);
        undo
    }

    fn score_empty_regions_on_board(&self, board: &[u8; MAX_N2], black: &mut f32, white: &mut f32) {
        let n2 = self.n2();
        let mut visited = [false; MAX_N2];
        let mut stack = [0usize; MAX_N2];

        for pos in 0..n2 {
            if board[pos] != 0 || visited[pos] {
                continue;
            }
            let mut top = 0usize;
            let mut region_size = 0usize;
            let mut adj_black = false;
            let mut adj_white = false;
            stack[top] = pos;
            top += 1;
            visited[pos] = true;

            while top > 0 {
                top -= 1;
                let p = stack[top];
                region_size += 1;
                for nb in self.neighbors(p) {
                    match board[nb] {
                        1 => adj_black = true,
                        2 => adj_white = true,
                        0 if !visited[nb] => {
                            visited[nb] = true;
                            stack[top] = nb;
                            top += 1;
                        }
                        _ => {}
                    }
                }
            }

            let territory = region_size as f32;
            if adj_black && !adj_white {
                *black += territory;
            } else if adj_white && !adj_black {
                *white += territory;
            }
        }
    }

    // ── Scoring (Tromp-Taylor) ──

    /// Area scoring: stones on board + empty territory
    pub fn tromp_taylor_score(&self) -> (f32, f32) {
        let mut black = 0f32;
        let mut white = 0f32;
        let n2 = self.n2();

        for pos in 0..n2 {
            match self.board[pos] {
                1 => black += 1.0,
                2 => white += 1.0,
                _ => {}
            }
        }
        self.score_empty_regions_on_board(&self.board, &mut black, &mut white);

        (black, white + self.komi)
    }

    pub fn territory_score(&self) -> (f32, f32) {
        let (board, mut black, mut white) = self.territory_scoring_snapshot();
        self.score_empty_regions_on_board(&board, &mut black, &mut white);
        (black, white)
    }

    fn territory_scoring_snapshot(&self) -> ([u8; MAX_N2], f32, f32) {
        let mut board = self.board;
        let mut black = self.black_caps as f32;
        let mut white = self.white_caps as f32 + self.komi;
        if !matches!(self.ruleset, GoRuleset::Japanese | GoRuleset::Korean) {
            return (board, black, white);
        }

        loop {
            let (region_ids, region_owners, region_count) = self.classify_empty_regions(&board);
            let mut visited = [false; MAX_N2];
            let mut removed_any = false;
            let mut group = [0usize; MAX_N2];
            let mut stack = [0usize; MAX_N2];
            let mut adj_regions = [false; MAX_N2];

            for pos in 0..self.n2() {
                let color = board[pos];
                if color == 0 || visited[pos] {
                    continue;
                }
                let mut touches_opponent = false;
                let mut touches_edge = false;
                let mut group_count = 0usize;
                let mut top = 0usize;
                for rid in 0..region_count {
                    adj_regions[rid] = false;
                }
                stack[top] = pos;
                top += 1;
                visited[pos] = true;
                while top > 0 {
                    top -= 1;
                    let cur = stack[top];
                    group[group_count] = cur;
                    group_count += 1;
                    let row = self.row(cur);
                    let col = self.col(cur);
                    if row == 0 || row + 1 == self.size || col == 0 || col + 1 == self.size {
                        touches_edge = true;
                    }
                    for nb in self.neighbors(cur) {
                        if board[nb] == color && !visited[nb] {
                            visited[nb] = true;
                            stack[top] = nb;
                            top += 1;
                        } else if board[nb] == 0 {
                            let rid = region_ids[nb];
                            if rid != usize::MAX {
                                adj_regions[rid] = true;
                            }
                        } else if board[nb] == Self::opp(color) {
                            touches_opponent = true;
                        }
                    }
                }

                let mut eye_count = 0usize;
                let mut touches_neutral = false;
                for (rid, present) in adj_regions.iter().enumerate() {
                    if !present {
                        continue;
                    }
                    match region_owners[rid] {
                        owner if owner == color => eye_count += 1,
                        3 => touches_neutral = true,
                        _ => {}
                    }
                }

                if eye_count < 2 && !touches_neutral && touches_opponent && !touches_edge {
                    let removed = group_count as f32;
                    removed_any = true;
                    for &stone in &group[..group_count] {
                        board[stone] = 0;
                    }
                    if color == 1 {
                        white += removed;
                    } else {
                        black += removed;
                    }
                }
            }

            if !removed_any {
                break;
            }
        }

        (board, black, white)
    }

    fn classify_empty_regions(
        &self,
        board: &[u8; MAX_N2],
    ) -> ([usize; MAX_N2], [u8; MAX_N2], usize) {
        let n2 = self.n2();
        let mut region_ids = [usize::MAX; MAX_N2];
        let mut region_owners = [0u8; MAX_N2];
        let mut stack = [0usize; MAX_N2];
        let mut region_count = 0usize;

        for pos in 0..n2 {
            if board[pos] != 0 || region_ids[pos] != usize::MAX {
                continue;
            }
            let rid = region_count;
            let mut owners = [false; 3];
            let mut top = 0usize;
            stack[top] = pos;
            top += 1;
            region_ids[pos] = rid;
            while top > 0 {
                top -= 1;
                let cur = stack[top];
                for nb in self.neighbors(cur) {
                    match board[nb] {
                        0 if region_ids[nb] == usize::MAX => {
                            region_ids[nb] = rid;
                            stack[top] = nb;
                            top += 1;
                        }
                        1 | 2 => owners[board[nb] as usize] = true,
                        _ => {}
                    }
                }
            }
            let owner = match (owners[1], owners[2]) {
                (true, false) => 1,
                (false, true) => 2,
                _ => 3,
            };
            region_owners[rid] = owner;
            region_count += 1;
        }

        (region_ids, region_owners, region_count)
    }
}

// ═══════════════════════════════════════════════════════
// § Display
// ═══════════════════════════════════════════════════════

impl fmt::Display for Go {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in (0..self.size).rev() {
            write!(f, "{:2} ", r + 1)?;
            for c in 0..self.size {
                let ch = match self.board[r * self.size + c] {
                    1 => 'X',
                    2 => 'O',
                    _ => '.',
                };
                write!(f, "{} ", ch)?;
            }
            writeln!(f)?;
        }
        write!(f, "   ")?;
        for c in 0..self.size {
            write!(f, "{} ", (b'A' + c as u8) as char)?;
        }
        writeln!(f)
    }
}

impl fmt::Debug for Go {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Go({}x{}, side={}, caps=B{}W{}, ko={:?}, cycle={})",
            self.size,
            self.size,
            if self.side == 1 { "B" } else { "W" },
            self.black_caps,
            self.white_caps,
            if self.ko_point == u16::MAX {
                None
            } else {
                Some(self.ko_point)
            },
            self.cycle_terminal
        )
    }
}

// ═══════════════════════════════════════════════════════
// § GameState trait
// ═══════════════════════════════════════════════════════

impl GameState for Go {
    type Move = u16; // 0..N*N for board, N*N for pass
    type Undo = GoUndo;

    fn initial() -> Self {
        Go::new_9x9()
    }

    fn current_player(&self) -> i8 {
        if self.side == 1 {
            1
        } else {
            -1
        }
    }

    fn legal_moves(&self) -> Vec<u16> {
        if self.passes >= 2 || self.cycle_terminal {
            return vec![];
        }
        let n2 = self.n2();
        let mut moves = Vec::with_capacity(n2);
        let color = self.side;
        let opp = Self::opp(color);
        let z = &*GZOB;
        let color_idx = (color - 1) as usize;
        let chinese_superko = self.ruleset == GoRuleset::Chinese;
        for pos in 0..n2 {
            if self.board[pos] != 0 || pos as u16 == self.ko_point {
                continue;
            }

            let (has_empty_neighbor, has_adjacent_opp) = self.neighbor_flags(pos, opp);
            if has_empty_neighbor && !has_adjacent_opp {
                if chinese_superko {
                    let next_hash = self.hash ^ z.piece[color_idx][pos] ^ z.side;
                    if self.repeats_position_hash(next_hash) {
                        continue;
                    }
                }
                moves.push(pos as u16);
            } else if self.is_legal(pos) {
                moves.push(pos as u16);
            }
        }
        moves.push(self.pass_action()); // pass is always legal
        moves
    }

    fn apply_move(&self, mv: u16) -> Self {
        let n2 = self.n2() as u16;
        let z = &*GZOB;

        // Prepare board ring: push current board snapshot (zero-alloc)
        let mut ring = self.board_ring.clone();
        ring.push(&self.board, self.n2());

        // O(1) hash_history clone via Arc
        let hash_hist = Arc::clone(&self.hash_history);

        if mv == n2 {
            // Pass
            let mut next = self.clone_core(hash_hist, ring);
            next.passes += 1;
            next.ko_point = u16::MAX;
            next.side = Self::opp(self.side);
            next.hash ^= z.side;
            next.move_count += 1;
            self.finalize_transition(&mut next);
            return next;
        }

        let pos = mv as usize;
        let mut next = self.do_place(pos);
        next.hash_history = hash_hist;
        next.board_ring = ring;
        next.passes = 0;
        next.side = Self::opp(self.side);
        next.hash ^= z.side;
        next.move_count += 1;
        self.finalize_transition(&mut next);
        next
    }

    fn apply_move_in_place(&mut self, mv: u16) -> GoUndo {
        self.apply_move_mut_internal(mv)
    }

    fn apply_move_in_place_no_undo(&mut self, mv: u16) {
        let _ = self.apply_move_mut_internal(mv);
    }

    fn undo_move(&mut self, undo: GoUndo) {
        for &(pos, value) in undo.board_changes.iter().rev() {
            self.board[pos as usize] = value;
        }
        self.side = undo.prev_side;
        self.ko_point = undo.prev_ko_point;
        self.passes = undo.prev_passes;
        self.black_caps = undo.prev_black_caps;
        self.white_caps = undo.prev_white_caps;
        self.hash = undo.prev_hash;
        self.move_count = undo.prev_move_count;
        self.history_digest = undo.prev_history_digest;
        self.cycle_terminal = undo.prev_cycle_terminal;
        self.hash_history = undo.prev_hash_history;
        self.board_ring.undo_push(undo.board_ring_undo);
    }

    fn is_terminal(&self) -> bool {
        self.passes >= 2 || self.cycle_terminal
    }

    fn outcome(&self) -> f32 {
        if self.cycle_terminal {
            return 0.0;
        }
        let (bs, ws) = match self.scoring {
            GoScoring::Area => self.tromp_taylor_score(),
            GoScoring::Territory => self.territory_score(),
        };
        if bs > ws {
            if self.side == 1 {
                1.0
            } else {
                -1.0
            }
        } else if ws > bs {
            if self.side == 2 {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        }
    }

    fn hash(&self) -> u64 {
        self.hash
    }

    fn tt_hash(&self) -> u64 {
        let mut tt = tt_combine(GO_TT_HASH_SEED, self.hash);
        tt = tt_combine(tt, self.history_digest);
        tt = tt_combine(tt, self.size as u64);
        tt = tt_combine(tt, self.ko_point as u64);
        tt = tt_combine(tt, self.passes as u64);
        tt = tt_combine(tt, self.black_caps as u64);
        tt = tt_combine(tt, self.white_caps as u64);
        tt = tt_combine(tt, self.komi.to_bits() as u64);
        tt = tt_combine(tt, self.ruleset as u64);
        tt = tt_combine(tt, self.scoring as u64);
        tt = tt_combine(tt, self.allow_suicide as u64);
        tt = tt_combine(tt, self.cycle_terminal as u64);
        tt_mix64(tt)
    }

    fn num_actions(&self) -> usize {
        self.n2() + 1
    } // board + pass

    fn move_to_idx(&self, mv: u16) -> usize {
        mv as usize
    }

    fn idx_to_move(&self, idx: usize) -> Option<u16> {
        if idx <= self.n2() {
            Some(idx as u16)
        } else {
            None
        }
    }

    /// NN 입력 feature planes (AlphaZero-style: [17, N, N])
    ///   planes 0-1:   현재(t=0) 내 돌, 상대 돌
    ///   planes 2-3:   t=1 (1수 전) 내 돌, 상대 돌
    ///   ...
    ///   planes 14-15: t=7 (7수 전) 내 돌, 상대 돌
    ///   plane 16:     현재 플레이어 색상 (black=1, white=0)
    fn encode_planes_into(&self, out: &mut Vec<f32>) {
        let n2 = self.n2();
        let total_planes = GO_HISTORY_LEN * 2 + 1; // 17
        let total = total_planes * n2;
        let my = self.side;
        let opp = Go::opp(my);

        GO_FEATURE_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let color_base = (total_planes - 1) * n2;
            if out.len() != total {
                out.clear();
                out.resize(total, 0.0);
                scratch.touched.clear();
                scratch.color_plane_filled = false;
            } else {
                // Sparse reset: only zero indices we set last time
                for &idx in &scratch.touched {
                    out[idx] = 0.0;
                }
                scratch.touched.clear();
                if scratch.color_plane_filled {
                    out[color_base..color_base + n2].fill(0.0);
                    scratch.color_plane_filled = false;
                }
            }

            // t=0: current board
            for i in 0..n2 {
                if self.board[i] == my {
                    out[i] = 1.0;
                    scratch.touched.push(i);
                } else if self.board[i] == opp {
                    let idx = n2 + i;
                    out[idx] = 1.0;
                    scratch.touched.push(idx);
                }
            }

            // t=1..7: history from ring buffer
            for (k, hist_board) in self.board_ring.iter_rev().enumerate() {
                let t = k + 1;
                if t >= GO_HISTORY_LEN {
                    break;
                }
                let base = t * 2 * n2;
                for i in 0..n2 {
                    let v = hist_board[i];
                    if v == my {
                        let idx = base + i;
                        out[idx] = 1.0;
                        scratch.touched.push(idx);
                    } else if v == opp {
                        let idx = base + n2 + i;
                        out[idx] = 1.0;
                        scratch.touched.push(idx);
                    }
                }
            }

            // Color plane
            if self.side == 1 {
                out[color_base..color_base + n2].fill(1.0);
                scratch.color_plane_filled = true;
            }
        });
    }

    fn board_state_record(&self) -> Vec<i64> {
        let n2 = self.n2();
        self.board[..n2].iter().map(|&v| v as i64).collect()
    }

    /// O(n²) scan without Vec allocation — critical for ShortRollout performance.
    fn random_legal_move(&self, rand_idx: usize) -> Option<u16> {
        if self.passes >= 2 || self.cycle_terminal {
            return None;
        }
        let n2 = self.n2();

        // Count empty cells (potential moves)
        let mut empty_count = 0usize;
        for i in 0..n2 {
            if self.board[i] == 0 {
                empty_count += 1;
            }
        }

        if empty_count == 0 {
            return Some(self.pass_action());
        }

        // Try from random offset, wrap around
        let start = rand_idx % n2;
        for offset in 0..n2 {
            let pos = (start + offset) % n2;
            if self.board[pos] == 0 && self.is_legal(pos) {
                return Some(pos as u16);
            }
        }

        // No legal board move → pass
        Some(self.pass_action())
    }

    fn legal_move_count(&self) -> usize {
        if self.passes >= 2 || self.cycle_terminal {
            return 0;
        }
        let n2 = self.n2();
        let mut count = 1usize; // pass always legal
        for pos in 0..n2 {
            if self.is_legal(pos) {
                count += 1;
            }
        }
        count
    }
}

// ═══════════════════════════════════════════════════════
// § MctsConfig preset
// ═══════════════════════════════════════════════════════

use crate::mcts::gvoc::GvocConfig;
use crate::mcts::quartz::QuartzConfig;
use crate::mcts::{MctsConfig, PwConfig};

/// Go QUARTZ 프리셋.
/// - PW: α=2.0, β=0.5 (9×9 branching ~40-80)
/// - σ₀ = 0.3
/// - GVOC: max_visible scales with board size
pub fn go_quartz(size: usize) -> MctsConfig {
    let max_visible = match size {
        9 => 40,
        13 => 80,
        _ => 120,
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
            max_visible,
            min_visible: 1,
            score_interval: 50,
        })
}

// ═══════════════════════════════════════════════════════
// § GoFastRollout — Vec 할당 없는 고속 롤아웃 평가기
// ═══════════════════════════════════════════════════════

#[cfg(test)]
use crate::game::{EvalResult, Evaluator};

#[cfg(test)]
/// Go-specific evaluator: uniform policy + fast rollout using random_legal_move().
/// ShortRollout 대비 Vec 할당이 없어 Go 19×19에서 2-3× 빠름.
pub struct GoFastRollout {
    pub max_depth: usize,
}

#[cfg(test)]
impl GoFastRollout {
    pub fn new(max_depth: usize) -> Self {
        GoFastRollout { max_depth }
    }
}

#[cfg(test)]
impl Evaluator<Go> for GoFastRollout {
    fn evaluate(&self, state: &Go) -> EvalResult<u16> {
        let legal = state.legal_moves();
        if legal.is_empty() {
            return EvalResult::uniform(&[], state.outcome());
        }
        let p = 1.0 / legal.len() as f32;
        let policy: Vec<(u16, f32)> = legal.iter().map(|&m| (m, p)).collect();

        // Fast rollout using random_legal_move (no Vec allocation per step)
        let value = go_fast_playout(state, self.max_depth);
        EvalResult { policy, value }
    }
}

#[cfg(test)]
fn go_fast_playout(start: &Go, max_depth: usize) -> f32 {
    let mut state = start.clone();
    let root_player = start.current_player();
    let mut rng_state = start
        .hash()
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);

    for _ in 0..max_depth {
        if state.is_terminal() {
            let raw = state.outcome();
            let flip = if state.current_player() == root_player {
                1.0
            } else {
                -1.0
            };
            return raw * flip;
        }
        // Use random_legal_move: O(N²) scan, no Vec allocation
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let rand_idx = (rng_state >> 33) as usize;
        let mv = match state.random_legal_move(rand_idx) {
            Some(m) => m,
            None => break,
        };
        state = state.apply_move(mv);
    }
    0.0 // draw if max depth reached
}

// ═══════════════════════════════════════════════════════
// § Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    fn bench_loops(default: usize) -> usize {
        std::env::var("GAME_BENCH_LOOPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(default)
    }

    fn pos(x: usize, y: usize, sz: usize) -> u16 {
        (y * sz + x) as u16
    }
    fn p(x: usize, y: usize) -> u16 {
        pos(x, y, 9)
    }

    fn play_moves(g: &Go, moves: &[u16]) -> Go {
        let mut s = g.clone();
        for &mv in moves {
            s = s.apply_move(mv);
        }
        s
    }

    fn assert_same_state(a: &Go, b: &Go) {
        assert_eq!(a.board, b.board);
        assert_eq!(a.size, b.size);
        assert_eq!(a.side, b.side);
        assert_eq!(a.ko_point, b.ko_point);
        assert_eq!(a.passes, b.passes);
        assert_eq!(a.black_caps, b.black_caps);
        assert_eq!(a.white_caps, b.white_caps);
        assert_eq!(a.komi, b.komi);
        assert_eq!(a.ruleset, b.ruleset);
        assert_eq!(a.scoring, b.scoring);
        assert_eq!(a.hash, b.hash);
        assert_eq!(a.move_count, b.move_count);
        assert_eq!(a.hash_history.as_ref(), b.hash_history.as_ref());
        assert_eq!(a.history_digest, b.history_digest);
        assert_eq!(a.allow_suicide, b.allow_suicide);
        assert_eq!(a.cycle_terminal, b.cycle_terminal);
        assert_eq!(a.board_ring.len, b.board_ring.len);
        assert_eq!(a.board_ring.head, b.board_ring.head);
        assert_eq!(a.board_ring.boards, b.board_ring.boards);
        assert_eq!(a.tt_hash(), b.tt_hash());
    }

    // ── Basic ──

    #[test]
    fn test_basic_9x9() {
        let g = Go::new_9x9();
        assert_eq!(g.size, 9);
        assert_eq!(g.current_player(), 1);
        let moves = g.legal_moves();
        assert_eq!(moves.len(), 82, "81 board + 1 pass");
        assert!(moves.contains(&g.pass_action()));
        assert!(!g.is_terminal());
    }

    #[test]
    fn test_basic_19x19() {
        let g = Go::new_19x19();
        let moves = g.legal_moves();
        assert_eq!(moves.len(), 362, "361 + pass");
    }

    // ── Capture ──

    #[test]
    fn test_single_capture() {
        let g = Go::new_9x9();
        // Surround W(1,1): B at (0,1),(2,1),(1,0),(1,2)
        let g = play_moves(
            &g,
            &[
                p(0, 1), // B
                p(4, 4), // W far
                p(2, 1), // B
                p(1, 1), // W target
                p(1, 0), // B
                p(4, 5), // W far
                p(1, 2), // B captures W(1,1)
            ],
        );
        assert_eq!(g.board[p(1, 1) as usize], 0, "W(1,1) captured");
        assert_eq!(g.black_caps, 1);
    }

    #[test]
    fn test_multi_capture() {
        let g = Go::new_9x9();
        // W group at (1,0)+(2,0) on left edge
        let g = play_moves(
            &g,
            &[
                p(0, 0), // B
                p(1, 0), // W
                p(1, 1), // B
                p(2, 0), // W
                p(2, 1), // B
                p(4, 4), // W far
                p(3, 0), // B captures W group
            ],
        );
        assert_eq!(g.board[p(1, 0) as usize], 0, "W(1,0) captured");
        assert_eq!(g.board[p(2, 0) as usize], 0, "W(2,0) captured");
        assert_eq!(g.black_caps, 2);
    }

    // ── Ko ──

    #[test]
    fn test_ko_rule() {
        let g = Go::new_9x9();
        // Standard ko setup:
        // W stones at (row=1,col=0), (row=0,col=1), (row=2,col=1) surround (1,1)
        // B stones at (row=0,col=2), (row=2,col=2), (row=1,col=3) surround (1,2)
        // W at (1,2) has 1 liberty at (1,1)
        // B plays (1,1) → captures W(1,2) → B(1,1) has 1 liberty at (1,2) → ko!
        //
        // Board after setup (before B plays (1,1)):
        //   col: 0 1 2 3
        // row 2: . W B .
        // row 1: W . W B   ← W(1,2) surrounded by B(0,2),B(2,2),B(1,3) and liberty=(1,1)
        // row 0: . W B .
        //
        // pos = row*9 + col
        let _g = play_moves(
            &g,
            &[
                18, // B at (2,0) — far, just to give B a move
                1,  // W at (0,1) — left of ko center
                20, // B at (2,2) — surrounds W(1,2) from below
                9,  // W at (1,0) — above ko center... wait
            ],
        );
        // This is getting complicated with alternating moves. Let me use a FEN-like direct setup.
        // Instead, build the position move by move more carefully:

        let g = Go::new_9x9();
        // Need: W at pos 1(0,1), 9(1,0), 19(2,1) and W at 11(1,2)
        //       B at pos 20(2,2), 2(0,2), 28(1,3? no: row=3,col=1)
        // Actually let me just use y*9+x notation carefully
        // pos = row * 9 + col
        // I want a ko at intersection (row=1,col=1) and (row=1,col=2)
        // W group around (1,1): W at (0,1)=9, (2,1)=19, (1,0)=1
        // B group around (1,2): B at (0,2)=2, (2,2)=20, (1,3)=12
        // W at (1,2)=11 (to be captured)
        // B plays at (1,1)=10 → captures W(1,2)=11

        // Build by alternating. B=odd moves, W=even:
        let g = play_moves(
            &g,
            &[
                2,  // B at (0,2)  — surround (1,2)
                1,  // W at (0,1)  — surround (1,1)
                20, // B at (2,2)  — surround (1,2)
                19, // W at (2,1)  — surround (1,1)
                12, // B at (1,3)  — surround (1,2)
                11, // W at (1,2)  — target (1 liberty at (1,1))
            ],
        );

        // Verify setup: W(1,2)=11 has 1 liberty at (1,1)=10
        assert_eq!(g.count_liberties(11), 1, "W(1,2) should have 1 liberty");

        // Now W plays to give us more W stones around (1,1)
        // Actually, it's B's turn. B plays at (1,1)=10 → captures W(1,2)=11
        let _g = play_moves(
            &g,
            &[
                9, // B at (1,0) — also surrounds (1,1) but we need W there for ko
            ],
        );
        // Hmm, now it's W's turn. W needs to play at (1,0)? But B just played there.
        // Let me restart with a better approach.

        // Simplest ko: edge ko
        // W at corner (0,0), B at (0,1) and (1,0) surround it.
        // But that's just a capture, not a ko, because B at (0,1)+(1,0) has many liberties.

        // Proper ko requires BOTH sides have exactly 1 liberty at the ko point.
        // Setup (using direct board manipulation for clarity):
        let mut g = Go::new_9x9();
        // Directly place stones for the ko position
        // B: (0,2), (2,2), (1,3)  — surround (1,2)
        // W: (0,1), (2,1), (1,0)  — surround (1,1)
        // W: (1,2)                 — target stone
        // Then B plays (1,1) to capture W(1,2)
        g.board[2] = 1; // B at (0,2)
        g.board[20] = 1; // B at (2,2)
        g.board[12] = 1; // B at (1,3)
        g.board[9] = 2; // W at (1,0)
        g.board[1] = 2; // W at (0,1)
        g.board[19] = 2; // W at (2,1)
        g.board[11] = 2; // W at (1,2) — target
        g.side = 1; // B to play
                    // Recompute hash
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                if g.board[pos] == 1 {
                    g.hash ^= z.piece[0][pos];
                } else if g.board[pos] == 2 {
                    g.hash ^= z.piece[1][pos];
                }
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Go::history_digest_for_hash(g.hash);

        assert_eq!(
            g.count_liberties(11),
            1,
            "W(1,2) should have 1 lib at (1,1)"
        );

        // B plays (1,1)=10 → captures W(1,2)=11
        let g2 = g.apply_move(10);
        assert_eq!(g2.board[11], 0, "W(1,2) captured");
        assert_eq!(g2.black_caps, 1);
        // B(1,1) neighbors: W(0,1)=1, W(2,1)=19, W(1,0)=9, empty(1,2)=11 → 1 liberty
        assert_eq!(g2.count_liberties(10), 1, "B(1,1) should have 1 lib");
        assert_eq!(g2.ko_point, 11, "Ko point should be (1,2)=11");
        assert!(!g2.is_legal(11), "Ko point illegal");
    }

    #[test]
    fn test_ko_cleared_after_play_elsewhere() {
        // Use same ko setup, then play elsewhere
        let mut g = Go::new_9x9();
        g.board[2] = 1;
        g.board[20] = 1;
        g.board[12] = 1; // B surround (1,2)
        g.board[9] = 2;
        g.board[1] = 2;
        g.board[19] = 2; // W surround (1,1)
        g.board[11] = 2; // W target
        g.side = 1;
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Go::history_digest_for_hash(g.hash);

        let g = g.apply_move(10); // B captures at (1,1)
        assert_eq!(g.ko_point, 11);
        let g = g.apply_move(50); // W elsewhere
        assert_eq!(g.ko_point, u16::MAX, "Ko cleared after play elsewhere");
    }

    #[test]
    fn test_ko_cleared_on_pass() {
        let mut g = Go::new_9x9();
        g.board[2] = 1;
        g.board[20] = 1;
        g.board[12] = 1;
        g.board[9] = 2;
        g.board[1] = 2;
        g.board[19] = 2;
        g.board[11] = 2;
        g.side = 1;
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Go::history_digest_for_hash(g.hash);

        let g = g.apply_move(10); // B captures
        assert_eq!(g.ko_point, 11);
        let g = g.apply_move(g.pass_action()); // W pass
        assert_eq!(g.ko_point, u16::MAX, "Ko cleared on pass");
    }

    // ── Suicide ──

    #[test]
    fn test_suicide_forbidden() {
        let mut g = Go::new_9x9();
        g.allow_suicide = false;
        // W at (1,0) and (0,1): B playing (0,0) would be suicide
        let g = play_moves(
            &g,
            &[
                p(5, 5), // B far
                p(1, 0), // W
                p(5, 6), // B far
                p(0, 1), // W
            ],
        );
        assert!(!g.is_legal(p(0, 0) as usize), "Suicide forbidden");
    }

    #[test]
    fn test_capture_not_suicide() {
        let g = Go::new_9x9();
        // B(1,0), W(0,0) has 1 lib at (0,1). B plays (0,1) → captures W, not suicide
        let g = play_moves(
            &g,
            &[
                p(1, 0), // B
                p(0, 0), // W (1 lib at (0,1))
                p(0, 1), // B captures W(0,0)
            ],
        );
        assert_eq!(g.board[p(0, 0) as usize], 0, "W(0,0) captured");
    }

    // ── Pass ──

    #[test]
    fn test_double_pass_terminal() {
        let g = Go::new_9x9();
        let pass = g.pass_action();
        let g = play_moves(&g, &[pass, pass]);
        assert!(g.is_terminal(), "Double pass → terminal");
    }

    #[test]
    fn test_single_pass_not_terminal() {
        let g = Go::new_9x9();
        let g = g.apply_move(g.pass_action());
        assert!(!g.is_terminal());
    }

    // ── Scoring ──

    #[test]
    fn test_scoring_empty_board() {
        let g = Go::new_9x9();
        let pass = g.pass_action();
        let g = play_moves(&g, &[pass, pass]);
        let (bs, ws) = g.tromp_taylor_score();
        // Empty board: no stones, no territory (neutral) + komi
        assert_eq!(bs, 0.0);
        assert_eq!(ws, 7.5); // komi only
    }

    #[test]
    fn test_scoring_one_stone() {
        let g = Go::new_9x9();
        let pass = g.pass_action();
        let g = play_moves(&g, &[p(4, 4), pass, pass]);
        let (bs, ws) = g.tromp_taylor_score();
        // B has 1 stone + entire board territory (81), W has komi
        assert_eq!(bs, 81.0);
        assert_eq!(ws, 7.5);
    }

    // ── Edge ──

    #[test]
    fn test_corner_stone() {
        let g = Go::new_9x9();
        let g = g.apply_move(p(0, 0));
        assert_eq!(g.board[0], 1, "Corner stone placed");
        let libs = g.count_liberties(0);
        assert_eq!(libs, 2, "Corner has 2 liberties");
    }

    #[test]
    fn test_edge_stone() {
        let g = Go::new_9x9();
        let g = g.apply_move(p(0, 4));
        assert_eq!(
            g.count_liberties(p(0, 4) as usize),
            3,
            "Edge has 3 liberties"
        );
    }

    #[test]
    fn test_center_stone() {
        let g = Go::new_9x9();
        let g = g.apply_move(p(4, 4));
        assert_eq!(
            g.count_liberties(p(4, 4) as usize),
            4,
            "Center has 4 liberties"
        );
    }

    #[test]
    fn test_cant_play_occupied() {
        let g = Go::new_9x9();
        let g = g.apply_move(p(4, 4));
        assert!(!g.is_legal(p(4, 4) as usize));
    }

    // ── Board sizes ──

    #[test]
    fn test_board_13x13() {
        let g = Go::new_13x13();
        assert_eq!(g.legal_moves().len(), 170); // 169 + pass
    }

    // ── Pure function ──

    #[test]
    fn test_apply_move_pure() {
        let g = Go::new_9x9();
        let g2 = g.apply_move(p(4, 4));
        assert_eq!(g.board[p(4, 4) as usize], 0, "Original unchanged");
        assert_eq!(g2.board[p(4, 4) as usize], 1);
    }

    #[test]
    fn test_apply_move_in_place_roundtrip_matches_pure() {
        let mut s = Go::new_9x9();
        let opening = [p(4, 4), p(3, 4), p(4, 3), p(3, 3), s.pass_action()];
        for mv in opening {
            let before = s.clone();
            let pure = s.apply_move(mv);
            let undo = s.apply_move_in_place(mv);
            assert_same_state(&s, &pure);
            s.undo_move(undo);
            assert_same_state(&s, &before);
            s = pure;
        }
    }

    #[test]
    fn test_apply_move_in_place_roundtrip_with_capture() {
        let mut g = Go::new_9x9();
        g.board[0] = 2;
        g.board[9] = 2;
        g.board[18] = 2;
        g.board[27] = 2;
        g.board[1] = 1;
        g.board[10] = 1;
        g.board[19] = 1;
        g.board[28] = 1;
        g.board[37] = 1;
        g.side = 1;
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Go::history_digest_for_hash(g.hash);

        let before = g.clone();
        let pure = g.apply_move(36);
        let undo = g.apply_move_in_place(36);
        assert_same_state(&g, &pure);
        g.undo_move(undo);
        assert_same_state(&g, &before);
    }

    // ── Hash ──

    #[test]
    fn test_hash_changes() {
        let g = Go::new_9x9();
        let g2 = g.apply_move(p(4, 4));
        assert_ne!(g.hash(), g2.hash());
    }

    #[test]
    fn test_tt_hash_distinguishes_pass_count_on_same_board() {
        let g0 = Go::new_9x9();
        let g2 = g0.apply_move(g0.pass_action()).apply_move(g0.pass_action());

        assert_eq!(g0.hash(), g2.hash());
        assert_ne!(g0.tt_hash(), g2.tt_hash());
    }

    #[test]
    fn test_tt_hash_distinguishes_superko_history_on_same_board() {
        let g0 = Go::new_9x9();
        let mut g1 = g0.clone();
        let synthetic_seen = 0x1234_5678_9abc_def0;
        Arc::make_mut(&mut g1.hash_history).push(synthetic_seen);
        g1.history_digest = Go::history_digest_insert(g1.history_digest, synthetic_seen);

        assert_eq!(g0.hash(), g1.hash());
        assert_ne!(g0.tt_hash(), g1.tt_hash());
    }

    #[test]
    fn test_tt_hash_distinguishes_ko_point_on_same_board() {
        let g0 = Go::new_9x9();
        let mut g1 = g0.clone();
        g1.ko_point = 10;

        assert_eq!(g0.hash(), g1.hash());
        assert_ne!(g0.tt_hash(), g1.tt_hash());
    }

    // ── Send + Sync ──

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Go>();
    }

    // ── MCTS ──

    #[test]
    fn test_mcts_integration() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::search::FixedIterations;
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let state = Go::new_9x9();
        let eval: Arc<dyn crate::game::Evaluator<Go> + Send + Sync> = Arc::new(UniformEval);
        let config = MctsConfig::evaluation(2.0);
        let engine = MctsEngine::new(state, eval, config);
        engine.run(&mut FixedIterations::new(100));
        assert!(engine.best_move().is_some());
    }

    // ── Negamax ──

    #[test]
    fn test_negamax_convention() {
        let g = Go::new_9x9();
        let pass = g.pass_action();
        // B plays center, then both pass
        let g = play_moves(&g, &[p(4, 4), pass, pass]);
        assert!(g.is_terminal());
        // Black has 81 territory, White has 7.5 komi → B wins
        // Current player is Black (side alternates: B, W_pass, B_pass → side = B? Actually:
        // B plays p(4,4) → side=W, W passes → side=B, B passes → side=W
        // So current_player = -1 (White). Black won → White's perspective: -1
        assert!(
            g.outcome() < 0.0,
            "White's perspective: Black won → negative"
        );
    }

    // ══════════════════════════════════════
    // C++ adversarial_go.cpp 포팅 + 추가
    // ══════════════════════════════════════

    // ── Large group capture ──

    #[test]
    fn test_large_group_capture() {
        // Build a 4-stone W group on the edge and capture it
        let mut g = Go::new_9x9();
        // W stones along left edge: (0,0),(0,1),(0,2),(0,3)
        // B surrounds from col=1: (1,0),(1,1),(1,2),(1,3) and top/bottom
        g.board[0] = 2;
        g.board[9] = 2;
        g.board[18] = 2;
        g.board[27] = 2; // W col0, rows 0-3
        g.board[1] = 1;
        g.board[10] = 1;
        g.board[19] = 1;
        g.board[28] = 1; // B col1, rows 0-3
                         // W group has 1 liberty at (0,4)=36
        g.board[37] = 1; // B at (1,4) — doesn't block (0,4)
        g.side = 1;
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Go::history_digest_for_hash(g.hash);

        assert_eq!(
            g.count_liberties(0),
            1,
            "W group should have 1 liberty at (0,4)"
        );
        let g = g.apply_move(36); // B plays (0,4) → captures 4-stone W group
        assert_eq!(g.board[0], 0);
        assert_eq!(g.board[9], 0);
        assert_eq!(g.board[18], 0);
        assert_eq!(g.board[27], 0);
        assert_eq!(g.black_caps, 4);
    }

    // ── Self-atari (not suicide) ──

    #[test]
    fn test_self_atari_legal() {
        // Playing into self-atari (1 liberty remaining) is legal
        let mut g = Go::new_9x9();
        // B at (0,0). W at (1,0), (0,1). B plays (1,1)? No.
        // Simpler: B plays into a spot with only 1 liberty but doesn't capture → legal (not suicide)
        g.board[1] = 2; // W at (0,1)
        g.board[9] = 2; // W at (1,0)
        g.side = 1;
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Go::history_digest_for_hash(g.hash);

        // B at (0,0): neighbors (0,1)=W, (1,0)=W → 0 liberties → suicide → illegal
        assert!(!g.is_legal(0), "Suicide at corner should be illegal");

        // But if B has a friendly stone adjacent that provides a liberty:
        g.board[10] = 1; // B at (1,1)
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
        }
        // Now B at (1,0)=W, but (0,0)'s group won't connect to B(1,1) because (1,0) is W
        // So still suicide.
        assert!(!g.is_legal(0), "Still suicide — no friendly connection");
    }

    // ── Capture vs suicide precedence ──

    #[test]
    fn test_capture_trumps_suicide() {
        // B plays a move that looks like suicide but actually captures, making it legal
        let mut g = Go::new_9x9();
        // W at (0,0) with 1 liberty at (1,0). B at (0,1).
        g.board[0] = 2; // W at (0,0)
        g.board[1] = 1; // B at (0,1)
        g.side = 1;
        {
            let z = &*GZOB;
            g.hash = 0;
            for pos in 0..81 {
                match g.board[pos] {
                    1 => g.hash ^= z.piece[0][pos],
                    2 => g.hash ^= z.piece[1][pos],
                    _ => {}
                }
            }
        }
        g.hash_history = Arc::new(vec![g.hash]);
        g.history_digest = Go::history_digest_for_hash(g.hash);

        // W(0,0) has 1 liberty at (1,0)=9. B plays (1,0) → captures W(0,0)
        assert!(g.is_legal(9), "Capture at (1,0) should be legal");
        let g = g.apply_move(9);
        assert_eq!(g.board[0], 0, "W(0,0) captured");
        assert_eq!(g.black_caps, 1);
    }

    // ── Scoring with territory ──

    #[test]
    fn test_scoring_territory() {
        let mut g = Go::new_9x9();
        // B wall at col=3, W wall at col=5 → neutral col=4
        for r in 0..9 {
            g.board[r * 9 + 3] = 1;
        } // B wall at col 3
        for r in 0..9 {
            g.board[r * 9 + 5] = 2;
        } // W wall at col 5
        g.side = 1;
        let pass = g.pass_action();
        let g = play_moves(&g, &[pass, pass]);
        let (bs, ws) = g.tromp_taylor_score();
        // B: 9 stones + cols 0-2 territory (9*3=27) = 36
        // W: 9 stones + cols 6-8 territory (9*3=27) + komi 7.5 = 43.5
        // Col 4 = neutral (adjacent to both B and W)
        assert_eq!(bs, 36.0, "B: 9 stones + 27 territory");
        assert_eq!(ws, 36.0 + 7.5, "W: 9 stones + 27 territory + 7.5 komi");
    }

    #[test]
    fn test_territory_cleanup_removes_surrounded_one_eye_group() {
        let mut g = Go::new_with_rules(5, 6.5, GoRuleset::Japanese);
        for pos in [0usize, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24] {
            g.board[pos] = 1;
        }
        for pos in [6usize, 7, 8, 11, 13, 16, 17, 18] {
            g.board[pos] = 2;
        }
        let (bs, ws) = g.territory_score();
        assert_eq!(bs, 17.0);
        assert_eq!(ws, 6.5);
    }

    #[test]
    fn test_chinese_ruleset_rejects_repeated_position_hash() {
        let mut g = Go::new_with_rules(9, 7.5, GoRuleset::Chinese);
        let candidate = {
            let mut next = g.do_place(40);
            next.passes = 0;
            next.side = Go::opp(g.side);
            next.hash ^= GZOB.side;
            next.hash
        };
        Arc::make_mut(&mut g.hash_history).push(candidate);
        assert!(!g.is_legal(40));
    }

    #[test]
    fn test_korean_ruleset_marks_repetition_as_draw() {
        let mut g = Go::new_with_rules(9, 6.5, GoRuleset::Korean);
        let candidate = {
            let mut next = g.do_place(40);
            next.passes = 0;
            next.side = Go::opp(g.side);
            next.hash ^= GZOB.side;
            next.hash
        };
        Arc::make_mut(&mut g.hash_history).push(candidate);
        let g2 = g.apply_move(40);
        assert!(g2.is_terminal());
        assert!(g2.cycle_terminal);
        assert_eq!(g2.outcome(), 0.0);
    }

    // ── Move count ──

    #[test]
    fn test_move_count() {
        let g = Go::new_9x9();
        let g = play_moves(&g, &[p(0, 0), p(8, 8), p(4, 4)]);
        assert_eq!(g.move_count, 3);
    }

    // ── Occupied squares illegal ──

    #[test]
    fn test_all_occupied_illegal() {
        let g = Go::new_9x9();
        let g = g.apply_move(p(4, 4)); // B at center
        let g = g.apply_move(p(3, 3)); // W
                                       // Both occupied squares should be illegal
        assert!(!g.is_legal(p(4, 4) as usize));
        assert!(!g.is_legal(p(3, 3) as usize));
    }

    // ── Board size 19×19 ──

    #[test]
    fn test_19x19_play_and_capture() {
        let g = Go::new_19x19();
        let g = g.apply_move(0); // B corner (0,0)
        assert_eq!(g.board[0], 1);
        assert_eq!(g.count_liberties(0), 2); // corner = 2 libs
    }

    // ── QUARTZ preset ──

    #[test]
    fn test_go_quartz_preset() {
        use crate::mcts::eval::UniformEval;
        use crate::mcts::quartz::QuartzController;
        use crate::mcts::MctsEngine;
        use std::sync::Arc;

        for size in [9, 13] {
            let config = go_quartz(size);
            assert!(config.quartz.is_some());
            assert!(config.gvoc.is_some());
            let qcfg = config.quartz.clone().unwrap();
            let eval: Arc<UniformEval> = Arc::new(UniformEval);
            let engine = MctsEngine::new(Go::new(size, 7.5), eval, config);
            let mut ctrl = QuartzController::new(50, qcfg);
            let stats = engine.run_quartz(&mut ctrl);
            assert!(stats.iterations > 0, "{}x{} preset should run", size, size);
        }
    }

    // ── GoFastRollout ──

    #[test]
    fn test_fast_rollout_mcts() {
        use crate::mcts::search::FixedIterations;
        use crate::mcts::{MctsConfig, MctsEngine};
        use std::sync::Arc;

        let eval: Arc<GoFastRollout> = Arc::new(GoFastRollout::new(200));
        let config = MctsConfig::evaluation(2.0);
        let engine = MctsEngine::new(Go::new_9x9(), eval, config);
        engine.run(&mut FixedIterations::new(200));
        assert!(engine.best_move().is_some());
    }

    // ── Group liberty merging ──

    #[test]
    fn test_connected_group_liberties() {
        let g = Go::new_9x9();
        // Two adjacent B stones should share liberties
        let g = play_moves(&g, &[p(4, 4), p(0, 0), p(5, 4)]);
        // B group: (4,4)+(5,4) connected vertically
        // Combined liberties: (3,4),(4,3),(4,5),(5,3),(5,5),(6,4) = 6
        assert_eq!(g.count_liberties(p(4, 4) as usize), 6);
        assert_eq!(g.count_liberties(p(5, 4) as usize), 6); // same group
    }

    // ── Pass doesn't change board ──

    #[test]
    fn test_pass_preserves_board() {
        let g = Go::new_9x9();
        let g = g.apply_move(p(4, 4)); // B
        let board_before = g.board;
        let g = g.apply_move(g.pass_action()); // W pass
        assert_eq!(g.board, board_before);
        assert_eq!(g.side, 1); // B to play again
    }

    fn encode_planes_reference(state: &Go) -> Vec<f32> {
        let n2 = state.n2();
        let total_planes = GO_HISTORY_LEN * 2 + 1;
        let total = total_planes * n2;
        let my = state.side;
        let opp = Go::opp(my);
        let mut out = vec![0.0; total];

        for i in 0..n2 {
            if state.board[i] == my {
                out[i] = 1.0;
            } else if state.board[i] == opp {
                out[n2 + i] = 1.0;
            }
        }

        for (k, hist_board) in state.board_ring.iter_rev().enumerate() {
            let t = k + 1;
            if t >= GO_HISTORY_LEN {
                break;
            }
            let base = t * 2 * n2;
            for i in 0..n2 {
                let v = hist_board[i];
                if v == my {
                    out[base + i] = 1.0;
                } else if v == opp {
                    out[base + n2 + i] = 1.0;
                }
            }
        }

        if state.side == 1 {
            let color_base = (total_planes - 1) * n2;
            out[color_base..color_base + n2].fill(1.0);
        }
        out
    }

    #[test]
    fn test_encode_planes_matches_reference_and_board_ring_on_edge_cases() {
        let mut sequence = Vec::new();
        let mut snapshots = Vec::new();
        let mut g = Go::new_9x9();
        sequence.push(g.clone());
        for mv in [
            p(4, 4),
            p(0, 0),
            p(4, 5),
            p(0, 1),
            p(5, 4),
            p(1, 0),
            p(3, 4),
            p(1, 1),
            p(4, 3),
            g.pass_action(),
            p(2, 0),
        ] {
            snapshots.push(g.board);
            g = g.apply_move(mv);
            sequence.push(g.clone());
        }

        let mut scratch = Vec::new();
        for state in &sequence {
            let expected = encode_planes_reference(state);
            state.encode_planes_into(&mut scratch);
            assert_eq!(
                scratch, expected,
                "encode_planes_into drifted from reference on {:?}",
                state
            );
        }

        let latest = sequence.last().unwrap();
        for (k, hist_board) in latest.board_ring.iter_rev().enumerate() {
            let expected = snapshots[snapshots.len() - 1 - k];
            assert_eq!(
                &hist_board[..latest.n2()],
                &expected[..latest.n2()],
                "board ring drifted at history step {}",
                k + 1
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_go_hotpaths() {
        let mut state = Go::new_9x9();
        for &mv in &[
            p(4, 4),
            p(0, 0),
            p(4, 5),
            p(0, 1),
            p(5, 4),
            p(1, 0),
            p(3, 4),
            p(1, 1),
        ] {
            state = state.apply_move(mv);
        }
        let loops = bench_loops(5_000);

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.clone());
        }
        let clone_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        for _ in 0..loops {
            black_box(state.legal_moves());
        }
        let legal_ms = start.elapsed().as_secs_f64() * 1000.0;

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
            "[bench] go9: clone={clone_ms:.2}ms legal={legal_ms:.2}ms apply={apply_ms:.2}ms encode={encode_ms:.2}ms encode_reuse={encode_reuse_ms:.2}ms loops={loops}"
        );
    }
}

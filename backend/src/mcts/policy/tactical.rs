//! BQ++ Phase 5: Tactical sentinel — cheap CPU-only forced-move detection.
//!
//! Hidden-win recall is the primary failure mode of pure PAC stops:
//! a low-prior winning move never enters the candidate set, so the
//! certificate certifies the wrong best arm. This module detects
//! "this move forces an immediate win" or "this move is the sole
//! block against the opponent's immediate win" and exposes the
//! result as `TacticalResult::Forced(action_id)` for injection into
//! the candidate reservoir (`forced_move_pos` on `PolicyCache`).
//!
//! The sentinel is **conservative**: it returns `Some(action)` only
//! when the move is provably forcing (no false positives). False
//! negatives (a forcing move not detected — e.g. a mate-in-2, or a
//! block requiring 3+ ply lookahead) are acceptable since the main
//! search will eventually find them; false positives (mis-detecting
//! a non-forcing move as forced) would corrupt the search.
//!
//! ## A4-a audit fix: game-agnostic rewrite
//!
//! The original implementation was `gomoku_sentinel(&Gomoku)` —
//! Gomoku-typed board-pattern code (directional line scanning keyed
//! on `win_len`) living in mainline `src/mcts/policy/`, in direct
//! tension with the CLAIM_LEDGER FORBIDDEN row banning game-specific
//! rule/pattern injection in mainline. The promised `chess_sentinel`/
//! `go_sentinel` stubs were never actually implemented (doc-comment
//! only).
//!
//! `tactical_sentinel<G: GameState>` replaces all three with one
//! generic function using only the `GameState` trait — no board
//! geometry, no per-game pattern code:
//!
//! - **1-ply immediate win**: for each legal move, `is_winning_move`
//!   (a `GameState` trait method; games may override it for an O(1)
//!   check — Gomoku/Gomoku15 do — the generic default is
//!   apply_move + is_terminal).
//! - **2-ply forced block**: if no immediate win, check every legal
//!   move `mv`: after applying it, does the opponent have ANY
//!   immediate winning reply? If EXACTLY ONE move leaves the
//!   opponent without one, that move is the forced block (all others
//!   let the opponent win next turn). If two or more moves defuse
//!   the threat, it isn't uniquely forcing (no false positive). If
//!   no move defuses it, the loss is unavoidable and no single move
//!   is "the" block (also no false positive).
//!
//! This is a strictly more general mechanism than the old occupy-the-
//! winning-cell block (which implicitly assumed Gomoku's no-capture,
//! no-movement rules): it directly checks the RESULT — does the
//! opponent still have a winning reply — rather than assuming HOW
//! blocking works mechanically, so it is correct for capture/
//! movement-based games (chess, go) with zero game-specific code.
//!
//! Cost: O(L x L') `is_winning_move` calls plus L `legal_moves()`
//! allocations, where L/L' are legal-move counts before/after one
//! ply. Not currently on the hot path (this module is not yet wired
//! into the live engine — see BQ_PLUS_PLUS_DESIGN.md §5); revisit if
//! it is wired for a large-branching-factor game (e.g. Go).

use crate::game::GameState;

/// Result of a tactical-sentinel check.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TacticalResult {
    /// The sentinel found a provably forcing move; the engine should
    /// halt and play this move regardless of the search state.
    Forced(u32),
    /// No forcing move detected (the sentinel is conservative; this
    /// does NOT mean "no forced move exists").
    None,
}

impl TacticalResult {
    pub fn is_forced(&self) -> bool {
        matches!(self, TacticalResult::Forced(_))
    }
    pub fn action_id(&self) -> Option<u32> {
        match self {
            TacticalResult::Forced(a) => Some(*a),
            TacticalResult::None => None,
        }
    }
}

/// Game-agnostic tactical sentinel. See the module doc for the
/// 1-ply-win / 2-ply-forced-block algorithm and its conservativeness
/// argument.
pub fn tactical_sentinel<G: GameState>(state: &G) -> TacticalResult {
    if state.is_terminal() {
        return TacticalResult::None;
    }
    let legal = state.legal_moves();

    // 1-ply: immediate win for the current player.
    for &mv in &legal {
        if state.is_winning_move(mv) {
            return TacticalResult::Forced(state.move_to_idx(mv) as u32);
        }
    }

    // 2-ply: forced block. `mv` "defuses" the threat iff, after
    // playing it, the opponent has no immediate winning reply.
    let mut safe_move: Option<G::Move> = None;
    let mut safe_count = 0usize;
    for &mv in &legal {
        let next = state.apply_move(mv);
        let opponent_has_win = next
            .legal_moves()
            .iter()
            .any(|&opp_mv| next.is_winning_move(opp_mv));
        if !opponent_has_win {
            safe_count += 1;
            safe_move = Some(mv);
            if safe_count > 1 {
                break; // ambiguous; no need to keep counting
            }
        }
    }

    match (safe_count, safe_move) {
        (1, Some(mv)) => TacticalResult::Forced(state.move_to_idx(mv) as u32),
        _ => TacticalResult::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;
    use crate::games::Gomoku;

    /// Phase 5: empty 7×7 Gomoku (win_len=4) — no forcing move.
    #[test]
    fn test_phase5_gomoku_empty_board_no_forced() {
        let g = Gomoku::new_with_win(7, 4);
        let r = tactical_sentinel(&g);
        assert_eq!(r, TacticalResult::None);
    }

    /// Phase 5: 4-in-a-row already present in Gomoku 7×7 (win_len=4).
    /// Black has stones at (3, 0), (3, 1), (3, 2). Black to move.
    /// Placing at (3, 3) wins.
    #[test]
    fn test_phase5_gomoku_immediate_four_detected() {
        let mut g = Gomoku::new_with_win(7, 4);
        // Black plays (3, 0)
        let mv = 3 * 7 + 0;
        g.apply_move_in_place_no_undo(mv);
        // White plays (0, 0)
        g.apply_move_in_place_no_undo(0);
        // Black plays (3, 1)
        g.apply_move_in_place_no_undo(3 * 7 + 1);
        // White plays (0, 1)
        g.apply_move_in_place_no_undo(1);
        // Black plays (3, 2)
        g.apply_move_in_place_no_undo(3 * 7 + 2);
        // White plays (0, 2)
        g.apply_move_in_place_no_undo(2);
        // Now black to move; placing at (3, 3) completes 4-in-a-row.
        let r = tactical_sentinel(&g);
        assert_eq!(r, TacticalResult::Forced((3 * 7 + 3) as u32));
    }

    /// Phase 5: forced block — white is one move away from 4-in-a-row,
    /// black to move must block.
    #[test]
    fn test_phase5_gomoku_forced_block_detected() {
        let mut g = Gomoku::new_with_win(7, 4);
        // Black plays (0, 0)
        g.apply_move_in_place_no_undo(0);
        // White plays (3, 0)
        g.apply_move_in_place_no_undo(3 * 7);
        // Black plays (1, 0)
        g.apply_move_in_place_no_undo(7);
        // White plays (3, 1)
        g.apply_move_in_place_no_undo(3 * 7 + 1);
        // Black plays (2, 0)
        g.apply_move_in_place_no_undo(2 * 7);
        // White plays (3, 2)
        g.apply_move_in_place_no_undo(3 * 7 + 2);
        // Now black to move. White has 3-in-a-row at row 3, cols 0-2,
        // threatening (3,3). Black's own column-0 extension is capped
        // by white at (3,0), so black has no immediate win — exactly
        // one move (the block at (3,3)) leaves white without an
        // immediate winning reply.
        let r = tactical_sentinel(&g);
        assert!(r.is_forced(), "expected forced move, got {r:?}");
        assert_eq!(r, TacticalResult::Forced((3 * 7 + 3) as u32));
    }

    /// Phase 5: no forcing move when opponent has only 2 in a row.
    #[test]
    fn test_phase5_gomoku_two_in_row_not_forced() {
        let mut g = Gomoku::new_with_win(7, 4);
        // Black at (0, 0)
        g.apply_move_in_place_no_undo(0);
        // White at (3, 0)
        g.apply_move_in_place_no_undo(3 * 7);
        // Black at (1, 0)
        g.apply_move_in_place_no_undo(7);
        // White at (3, 1)
        g.apply_move_in_place_no_undo(3 * 7 + 1);
        // Now black to move. Both sides have 2-in-a-row. No forcing move yet.
        let r = tactical_sentinel(&g);
        assert_eq!(r, TacticalResult::None);
    }

    /// Phase 5: terminal state ⇒ no forcing move.
    #[test]
    fn test_phase5_gomoku_terminal_state_no_forced() {
        let mut g = Gomoku::new_with_win(7, 4);
        // Black wins by 4-in-a-row at row 3
        g.apply_move_in_place_no_undo(3 * 7); // black (3, 0)
        g.apply_move_in_place_no_undo(0); // white (0, 0)
        g.apply_move_in_place_no_undo(3 * 7 + 1); // black (3, 1)
        g.apply_move_in_place_no_undo(1); // white (0, 1)
        g.apply_move_in_place_no_undo(3 * 7 + 2); // black (3, 2)
        g.apply_move_in_place_no_undo(2); // white (0, 2)
        g.apply_move_in_place_no_undo(3 * 7 + 3); // black (3, 3) — wins
        assert!(g.is_terminal());
        let r = tactical_sentinel(&g);
        assert_eq!(r, TacticalResult::None);
    }

    /// Phase 5: TacticalResult API.
    #[test]
    fn test_phase5_tactical_result_api() {
        let r1 = TacticalResult::Forced(42);
        assert!(r1.is_forced());
        assert_eq!(r1.action_id(), Some(42));
        let r2 = TacticalResult::None;
        assert!(!r2.is_forced());
        assert_eq!(r2.action_id(), None);
    }

    /// Phase 5: priority — immediate win takes precedence over block.
    /// Construct a position where black has 3-in-a-row to win AND white
    /// also has 3-in-a-row threatening. Black should play the win, not
    /// the block.
    #[test]
    fn test_phase5_gomoku_immediate_win_priority_over_block() {
        let mut g = Gomoku::new_with_win(7, 4);
        // Black: (0, 0), (0, 1), (0, 2)  — needs (0, 3) to win
        g.apply_move_in_place_no_undo(0); // black (0, 0)
        g.apply_move_in_place_no_undo(6 * 7); // white (6, 0)
        g.apply_move_in_place_no_undo(1); // black (0, 1)
        g.apply_move_in_place_no_undo(6 * 7 + 1); // white (6, 1)
        g.apply_move_in_place_no_undo(2); // black (0, 2)
        g.apply_move_in_place_no_undo(6 * 7 + 2); // white (6, 2) — both sides have 3-in-a-row
                                                  // Now black to move. Both win positions are at (0, 3) and (6, 3).
        let r = tactical_sentinel(&g);
        // Black takes the win at (0, 3), not the block at (6, 3).
        assert_eq!(r, TacticalResult::Forced(3));
    }

    /// A4-a: the sentinel is generic over ANY GameState, not just
    /// Gomoku. TicTacToe (a much smaller board) exercises the same
    /// 1-ply-win / 2-ply-block logic through the identical function,
    /// with zero game-specific code in this module.
    #[test]
    fn test_a4a_tactical_sentinel_is_generic_over_tictactoe() {
        use crate::games::tictactoe::TicTacToe;

        let mut t = TicTacToe::initial();
        // X: (0,0), (0,1) — needs (0,2) to win a 3x3 row.
        t.apply_move_in_place_no_undo(0); // X (0,0)
        t.apply_move_in_place_no_undo(3); // O (1,0)
        t.apply_move_in_place_no_undo(1); // X (0,1)
        t.apply_move_in_place_no_undo(4); // O (1,1)
                                          // X to move; (0,2) completes the top row.
        let r = tactical_sentinel(&t);
        assert_eq!(r, TacticalResult::Forced(2));
    }

    /// A4-a regression: exactly the case the old occupy-the-cell
    /// block would have mishandled if ported naively — two DIFFERENT
    /// moves both defuse the opponent's threat (e.g. no threat exists
    /// yet), so the sentinel must not pick either one as "the" forced
    /// block.
    #[test]
    fn test_a4a_tactical_sentinel_no_false_positive_when_multiple_moves_are_safe() {
        let g = Gomoku::new_with_win(7, 4);
        // Empty board: every move is "safe" (no threat exists at all).
        let r = tactical_sentinel(&g);
        assert_eq!(r, TacticalResult::None);
    }
}

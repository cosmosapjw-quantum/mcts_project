// gomoku.h
#ifndef GOMOKU_H
#define GOMOKU_H

#include <vector>
#include <set>
#include <cstdint>
#include <utility>
#include <unordered_set>

#include "hash_specializations.h"

// Constants
const int BLACK = 1;
const int WHITE = 2;

class Gamestate {
public:
    // Constructor
    Gamestate(int board_size = 15, 
              bool use_renju = false, 
              bool use_omok = false, 
              int seed = 0, 
              bool use_pro_long_opening = false);
    
    // Copy constructor
    Gamestate(const Gamestate& other);
    
    // Assignment operator
    Gamestate& operator=(const Gamestate& other);
    
    // Public fields
    int board_size;
    int current_player;        // 1=BLACK, 2=WHITE
    std::vector<std::vector<uint64_t>> player_bitboards;  // shape=(2, num_words)
    int num_words;
    int action;                // last move made, or -1 if none
    
    bool use_renju;
    bool use_omok;
    bool use_pro_long_opening;
    int black_first_stone;
    
    // Cached valid moves to avoid recomputing
    mutable std::unordered_set<int> cached_valid_moves;
    mutable bool valid_moves_dirty; // Flag to indicate if cache needs refreshing
    
    // Cached terminal state status
    mutable int cached_winner;
    mutable bool winner_check_dirty;
    
    // Cached board representation for faster equality checks
    mutable uint64_t hash_signature;
    mutable bool hash_dirty;
    
    // Bitboard operations
    bool is_bit_set(int player_index, int action) const noexcept;
    void set_bit(int player_index, int action);
    void clear_bit(int player_index, int action) noexcept;
    
    std::pair<int, int> action_to_coords_pair(int action) const noexcept;
    int coords_to_action(int x, int y) const;
    int _count_total_stones() const noexcept;
    
    // Public interface
    bool is_terminal() const;
    int get_winner() const;
    bool board_equal(const Gamestate& other) const;
    std::vector<int> get_valid_moves() const;
    void make_move(int action, int player);
    void undo_move(int action);
    bool is_occupied(int action) const;
    bool is_stalemate() const;
    std::vector<std::vector<int>> get_board() const;
    
    // Fast path for checking if a move is valid (without computing all valid moves)
    bool is_move_valid(int action) const;
    
    // For MCTS or NN usage
    Gamestate apply_action(int action) const;
    std::vector<std::vector<std::vector<float>>> to_tensor() const;
    int get_action(const Gamestate& child_state) const;
    
    // Line counting
    bool is_five_in_a_row(int action, int player) const;
    
    // Forbidden move checks
    bool is_black_renju_forbidden(int action);
    bool is_black_omok_forbidden(int action);
    
    // Deep copy
    Gamestate copy() const;

    // Move history
    std::vector<int> move_history;
    
    // Get previous moves for a player
    std::vector<int> get_previous_moves(int player, int count = 7) const;
    
    // Compute and update hash signature
    uint64_t compute_hash_signature() const;
    
private:
    // Directions array
    int _dirs[8];
    
    // Line checking
    bool _check_line_for_five(int cell, int player, int p_idx) const noexcept;
    
    // Overline checks
    bool renju_is_overline(int action) const;
    bool _omok_is_overline(int action) const;
    
    // Renju forbidden checks
    bool renju_double_four_or_more(int action) const;
    bool renju_double_three_or_more(int action) const;
    int _renju_count_all_fours() const;
    int _renju_count_all_threes(int action) const;
    
    // Shape checks
    bool _renju_is_three_shape(const std::vector<std::pair<int, int>>& segment) const;
    bool _renju_is_four_shape(const std::vector<std::pair<int, int>>& segment) const;
    std::pair<bool, bool> _ends_are_open(const std::vector<std::pair<int, int>>& segment) const;
    bool _check_broken_four(const std::vector<std::pair<int, int>>& segment, bool front_open, bool back_open) const;
    bool _simple_is_4_contiguous(const std::vector<std::pair<int, int>>& segment) const;
    std::set<int> _positions_of_black(const std::vector<std::pair<int, int>>& segment) const;
    bool _try_unify_four_shape(std::set<std::pair<std::set<int>, int>>& found_fours, 
                               const std::set<int>& new_fs, int size) const;
    bool _try_unify_three_shape(std::set<std::set<int>>& found_threes, 
                                const std::set<int>& new_fs, int action) const;
    
    // Omok checks
    std::vector<std::set<int>> _get_open_three_patterns_globally() const;
    bool _are_patterns_connected(const std::set<int>& pattern1, const std::set<int>& pattern2) const;
    bool _omok_check_double_three_strict(int action) const;
    int _count_open_threes_globally() const;
    std::set<int> _check_open_three_5slice(const std::vector<std::pair<int, int>>& cells_5) const;
    
    // Helpers
    std::vector<std::pair<int, int>> _build_entire_line(int x0, int y0, int dx, int dy) const;
    bool _in_bounds(int x, int y) const;
    bool _is_pro_long_move_ok(int action, int stone_count) const;
    int _count_direction(int x0, int y0, int dx, int dy, int p_idx) const noexcept;

    // Enhanced double-three detection for Renju rules
    bool _is_allowed_double_three(int action) const;
    
    // Check if a three pattern can be made into a straight four
    bool _can_make_straight_four(const std::set<int>& three_pattern) const;
    
    // Count how many three patterns can be made into straight fours
    int _count_straight_four_capable_threes(const std::vector<std::set<int>>& three_patterns) const;
    
    // Recursively check if a double-three is allowed per section 9.3
    bool _is_double_three_allowed_recursive(const std::vector<std::set<int>>& three_patterns, 
                                          int depth = 0, int max_depth = 3) const;
    
    // Detect if a pattern is a straight four (open on both ends)
    bool _is_straight_four(const std::set<int>& pattern) const;
    
    // Find potential placements that would convert a three to a four
    std::vector<int> _find_three_to_four_placements(const std::set<int>& three_pattern) const;
    
    // Added missing function declarations
    bool _is_three_pattern(const std::vector<std::pair<int, int>>& segment, int action) const;
    bool _is_four_pattern(const std::vector<std::pair<int, int>>& segment) const;
    std::vector<std::set<int>> _get_three_patterns_for_action(int action) const;
    
    // Invalidate caches after state changes
    void _invalidate_caches();
    
    // Refresh valid moves cache
    void _refresh_valid_moves_cache() const;
    
    // Refresh winner cache
    void _refresh_winner_cache() const;
};

#endif // GOMOKU_H
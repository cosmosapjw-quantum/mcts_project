// gomoku.cpp
#include "gomoku.h"
#include <algorithm>
#include <random>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <string>
#include <iterator>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hash_specializations.h"

namespace py = pybind11;

// Constructor with initialization of caching fields
Gamestate::Gamestate(int board_size, bool use_renju, bool use_omok, int seed, bool use_pro_long_opening) 
    : board_size(board_size),
      current_player(BLACK),
      action(-1),
      use_renju(use_renju),
      use_omok(use_omok),
      use_pro_long_opening(use_pro_long_opening),
      black_first_stone(-1),
      valid_moves_dirty(true),
      cached_winner(0),
      winner_check_dirty(true),
      hash_signature(0),
      hash_dirty(true),
      move_history() {
    
    int total_cells = board_size * board_size;
    num_words = (total_cells + 63) / 64;
    
    // Initialize bitboards with zeros
    player_bitboards.resize(2, std::vector<uint64_t>(num_words, 0));
    
    // Directions for line scanning (dx,dy pairs)
    _dirs[0] = 0;   // dx=0
    _dirs[1] = 1;   // dy=1   (vertical)
    _dirs[2] = 1;   // dx=1
    _dirs[3] = 0;   // dy=0   (horizontal)
    _dirs[4] = 1;   // dx=1
    _dirs[5] = 1;   // dy=1   (diag-down)
    _dirs[6] = -1;  // dx=-1
    _dirs[7] = 1;   // dy=1   (diag-up)
    
    // Optional seed initialization
    if (seed != 0) {
        std::srand(seed);
    } else {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }
}

// Copy constructor with cache preservation
Gamestate::Gamestate(const Gamestate& other) 
    : board_size(other.board_size),
      current_player(other.current_player),
      player_bitboards(other.player_bitboards),
      num_words(other.num_words),
      action(other.action),
      use_renju(other.use_renju),
      use_omok(other.use_omok),
      use_pro_long_opening(other.use_pro_long_opening),
      black_first_stone(other.black_first_stone),
      cached_valid_moves(other.cached_valid_moves),
      valid_moves_dirty(other.valid_moves_dirty),
      cached_winner(other.cached_winner),
      winner_check_dirty(other.winner_check_dirty),
      hash_signature(other.hash_signature),
      hash_dirty(other.hash_dirty),
      move_history(other.move_history) {
    
    // Copy the directions array
    for (int i = 0; i < 8; i++) {
        _dirs[i] = other._dirs[i];
    }
}

// Assignment operator implementation
Gamestate& Gamestate::operator=(const Gamestate& other) {
    if (this != &other) {
        board_size = other.board_size;
        current_player = other.current_player;
        player_bitboards = other.player_bitboards;
        num_words = other.num_words;
        action = other.action;
        use_renju = other.use_renju;
        use_omok = other.use_omok;
        use_pro_long_opening = other.use_pro_long_opening;
        black_first_stone = other.black_first_stone;
        cached_valid_moves = other.cached_valid_moves;
        valid_moves_dirty = other.valid_moves_dirty;
        cached_winner = other.cached_winner;
        winner_check_dirty = other.winner_check_dirty;
        hash_signature = other.hash_signature;
        hash_dirty = other.hash_dirty;
        move_history = other.move_history;
        
        // Copy the directions array
        for (int i = 0; i < 8; i++) {
            _dirs[i] = other._dirs[i];
        }
    }
    return *this;
}

// Create a deep copy
Gamestate Gamestate::copy() const {
    return Gamestate(*this);
}

std::vector<int> Gamestate::get_previous_moves(int player, int count) const {
    std::vector<int> prev_moves(count, -1);  // Initialize with -1 (no move)
    
    int found = 0;
    // Iterate backward through move history
    for (int i = static_cast<int>(move_history.size()) - 1; i >= 0 && found < count; --i) {
        int move = move_history[i];
        // Determine which player made this move based on position in history
        int move_player = (move_history.size() - i) % 2 == 1 ? current_player : 3 - current_player;
        
        if (move_player == player) {
            prev_moves[found] = move;
            found++;
        }
    }
    
    return prev_moves;
}

// Optimized bitboard operations for fast move validation and terminal state detection

// Fast bit checking with bounds checking
bool Gamestate::is_bit_set(int player_index, int action) const noexcept {
    // Early bounds check to avoid out-of-bounds access
    if (player_index < 0 || player_index >= 2 || action < 0 || action >= board_size * board_size) {
        return false;
    }
    
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Additional bounds check for word_idx
    if (word_idx >= num_words) {
        return false;
    }
    
    // Use uint64_t mask for proper bit manipulation
    uint64_t mask = static_cast<uint64_t>(1) << bit_idx;
    return (player_bitboards[player_index][word_idx] & mask) != 0;
}

// Fast bit setting with inline optimization
inline void Gamestate::set_bit(int player_index, int action) {
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Use |= for optimal bit setting
    player_bitboards[player_index][word_idx] |= (static_cast<uint64_t>(1) << bit_idx);
    
    // Mark caches as dirty
    valid_moves_dirty = true;
    winner_check_dirty = true;
    hash_dirty = true;
}

// Fast bit clearing with inline optimization
inline void Gamestate::clear_bit(int player_index, int action) noexcept {
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    // Use &= with negated mask for optimal bit clearing
    player_bitboards[player_index][word_idx] &= ~(static_cast<uint64_t>(1) << bit_idx);
    
    // Mark caches as dirty
    valid_moves_dirty = true;
    winner_check_dirty = true;
    hash_dirty = true;
}

std::pair<int, int> Gamestate::action_to_coords_pair(int action) const noexcept {
    return {action / board_size, action % board_size};
}

int Gamestate::coords_to_action(int x, int y) const {
    return x * board_size + y;
}

// Optimized stone counting for bitboards
int Gamestate::_count_total_stones() const noexcept {
    int total = 0;
    
    for (int p = 0; p < 2; p++) {
        for (int w = 0; w < num_words; w++) {
            uint64_t chunk = player_bitboards[p][w];
            
            // Use __builtin_popcountll for fast bit counting if available
            #if defined(__GNUC__) || defined(__clang__)
                total += __builtin_popcountll(chunk);
            #else
                // Fallback to manual counting with Brian Kernighan's algorithm
                while (chunk != 0) {
                    chunk &= (chunk - 1);  // Clear lowest set bit
                    total++;
                }
            #endif
        }
    }
    
    return total;
}

// Main game functionality

// Refresh winner cache
void Gamestate::_refresh_winner_cache() const {
    // Check for a win by either player
    for (int p : {BLACK, WHITE}) {
        if (is_five_in_a_row(-1, p)) {
            cached_winner = p;
            winner_check_dirty = false;
            return;
        }
    }
    
    cached_winner = 0;
    winner_check_dirty = false;
}

// Optimized terminal state detection with caching
bool Gamestate::is_terminal() const {
    // Check winner cache first
    if (winner_check_dirty) {
        _refresh_winner_cache();
    }
    
    // If we have a winner, game is over
    if (cached_winner != 0) {
        return true;
    }
    
    // Check for stalemate (board full)
    return is_stalemate();
}

// Faster stalemate detection with bitboard operations
bool Gamestate::is_stalemate() const {
    // Use cached valid moves if available
    if (!valid_moves_dirty) {
        return cached_valid_moves.empty();
    }
    
    // Simple check: if board is full, it's a stalemate
    int stones = _count_total_stones();
    if (stones >= board_size * board_size) {
        return true;
    }
    
    // Otherwise, check if there are any valid moves
    _refresh_valid_moves_cache();
    return cached_valid_moves.empty();
}


// Optimized get_winner with caching
int Gamestate::get_winner() const {
    if (winner_check_dirty) {
        _refresh_winner_cache();
    }
    
    return cached_winner;
}

// Optimized get_valid_moves with smart caching
std::vector<int> Gamestate::get_valid_moves() const {
    // Use cache if available
    if (!valid_moves_dirty) {
        return std::vector<int>(cached_valid_moves.begin(), cached_valid_moves.end());
    }
    
    // Refresh cache
    _refresh_valid_moves_cache();
    
    // Return cached result
    return std::vector<int>(cached_valid_moves.begin(), cached_valid_moves.end());
}

// Refresh valid moves cache
void Gamestate::_refresh_valid_moves_cache() const {
    cached_valid_moves.clear();
    int total = board_size * board_size;
    int stone_count = _count_total_stones();
    Gamestate tmp(*this);
    
    // First identify occupied cells
    std::vector<bool> occupied(total, false);
    for (int a = 0; a < total; a++) {
        occupied[a] = is_bit_set(0, a) || is_bit_set(1, a);
    }
    
    // Check for valid moves based on rules
    if (use_renju && current_player == BLACK) {
        // Renju rules with forbidden move checks for Black
        for (int a = 0; a < total; a++) {
            if (!occupied[a]) {
                if (use_pro_long_opening) {
                    if (!_is_pro_long_move_ok(a, stone_count)) {
                        continue;
                    }
                }
                
                // Check forbidden moves (needs temporary state)
                tmp = copy();
                tmp.set_bit(0, a);
                if (!tmp.is_black_renju_forbidden(a)) {
                    cached_valid_moves.insert(a);
                }
            }
        }
    } else if (use_omok && current_player == BLACK) {
        // Omok rules with forbidden move checks for Black
        for (int a = 0; a < total; a++) {
            if (!occupied[a]) {
                if (use_pro_long_opening) {
                    if (!_is_pro_long_move_ok(a, stone_count)) {
                        continue;
                    }
                }
                
                // Check forbidden moves
                tmp = copy();
                tmp.set_bit(0, a);
                if (!tmp.is_black_omok_forbidden(a)) {
                    cached_valid_moves.insert(a);
                }
            }
        }
    } else {
        // Standard rules for White or no special rules
        for (int a = 0; a < total; a++) {
            if (!occupied[a]) {
                if (use_pro_long_opening && current_player == BLACK) {
                    if (!_is_pro_long_move_ok(a, stone_count)) {
                        continue;
                    }
                }
                
                cached_valid_moves.insert(a);
            }
        }
    }
    
    valid_moves_dirty = false;
}

// Fast path for checking if a move is valid without computing all valid moves
bool Gamestate::is_move_valid(int action) const {
    // Quick bounds check
    int total = board_size * board_size;
    if (action < 0 || action >= total) {
        return false;
    }
    
    // Check if already occupied
    if (is_occupied(action)) {
        return false;
    }
    
    // Check cached valid moves if available
    if (!valid_moves_dirty) {
        return cached_valid_moves.find(action) != cached_valid_moves.end();
    }
    
    // Special case for pro-long opening
    if (use_pro_long_opening && current_player == BLACK) {
        if (!_is_pro_long_move_ok(action, _count_total_stones())) {
            return false;
        }
    }
    
    // Check forbidden moves for Black
    if (current_player == BLACK) {
        if (use_renju) {
            Gamestate tmp = copy();
            tmp.set_bit(0, action);
            if (tmp.is_black_renju_forbidden(action)) {
                return false;
            }
        } else if (use_omok) {
            Gamestate tmp = copy();
            tmp.set_bit(0, action);
            if (tmp.is_black_omok_forbidden(action)) {
                return false;
            }
        }
    }
    
    return true;
}

// Optimized hash computation using Zobrist-inspired approach
uint64_t Gamestate::compute_hash_signature() const {
    if (!hash_dirty) {
        return hash_signature;
    }
    
    uint64_t hash = 0;
    const int cells = board_size * board_size;
    
    // Use large prime multipliers for better distribution
    const uint64_t black_prime = 73856093;
    const uint64_t white_prime = 19349663;
    
    // Process in chunks of 64 bits
    for (int w = 0; w < num_words; w++) {
        uint64_t black_word = player_bitboards[0][w];
        uint64_t white_word = player_bitboards[1][w];
        
        // Process all bits set to 1
        for (int b = 0; b < 64; b++) {
            uint64_t mask = static_cast<uint64_t>(1) << b;
            int action = w * 64 + b;
            
            if (action >= cells) break;
            
            if (black_word & mask) {
                hash ^= (static_cast<uint64_t>(action) * black_prime);
            } else if (white_word & mask) {
                hash ^= (static_cast<uint64_t>(action) * white_prime);
            }
        }
    }
    
    // Include current player in hash
    if (current_player == BLACK) {
        hash ^= 0xABCDEF;
    }
    
    // Update cached hash
    hash_signature = hash;
    hash_dirty = false;
    
    return hash_signature;
}

// Optimized board comparison using hash signatures
bool Gamestate::board_equal(const Gamestate& other) const {
    // Quick check for different board sizes
    if (board_size != other.board_size || current_player != other.current_player) {
        return false;
    }
    
    // Compare hash signatures if available
    if (!hash_dirty && !other.hash_dirty) {
        return hash_signature == other.hash_signature;
    }
    
    // Compare individual bitboards
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < num_words; j++) {
            if (player_bitboards[i][j] != other.player_bitboards[i][j]) {
                return false;
            }
        }
    }
    
    return true;
}

// Optimized move making with incremental updates
void Gamestate::make_move(int action, int player) {
    // Quick validation
    if (action < 0 || action >= board_size * board_size) {
        throw std::runtime_error("Move " + std::to_string(action) + " out of range.");
    }
    if (is_occupied(action)) {
        throw std::runtime_error("Cell " + std::to_string(action) + " is already occupied.");
    }
    
    // Rule validation if needed
    if (use_pro_long_opening && player == BLACK) {
        if (!_is_pro_long_move_ok(action, _count_total_stones())) {
            throw std::runtime_error("Pro-Long Opening constraint violated.");
        }
    }
    
    if (player == BLACK) {
        if (use_renju && is_black_renju_forbidden(action)) {
            throw std::runtime_error("Forbidden Move by Black (Renju).");
        } else if (use_omok && is_black_omok_forbidden(action)) {
            throw std::runtime_error("Forbidden Move by Black (Omok).");
        }
    }
    
    // Place the stone with bitboard operations
    set_bit(player - 1, action);
    this->action = action;
    
    // Update black's first stone if needed
    if (player == BLACK && black_first_stone < 0) {
        black_first_stone = action;
    }
    
    // Update player turn
    current_player = 3 - player;
    
    // Add to move history
    move_history.push_back(action);
    
    // Invalidate caches
    _invalidate_caches();
}

// Optimized undo_move with cache invalidation
void Gamestate::undo_move(int action) {
    int total = board_size * board_size;
    if (action < 0 || action >= total) {
        throw std::runtime_error("Undo " + std::to_string(action) + " out of range.");
    }

    int prev_player = 3 - current_player;
    int p_idx = prev_player - 1;

    if (!is_bit_set(p_idx, action)) {
        throw std::runtime_error("Undo error: Stone not found for last mover.");
    }

    // Remove the stone
    clear_bit(p_idx, action);
    this->action = -1;
    
    // Update player turn
    current_player = prev_player;

    if (prev_player == BLACK && black_first_stone == action) {
        black_first_stone = -1;
    }

    // Update move history
    if (!move_history.empty()) {
        move_history.pop_back();
    }

    // Invalidate caches
    _invalidate_caches();
}

// Cache invalidation helper
void Gamestate::_invalidate_caches() {
    valid_moves_dirty = true;
    winner_check_dirty = true;
    hash_dirty = true;
}

// Optimized is_occupied check using inline optimization
inline bool Gamestate::is_occupied(int action) const {
    // Use bitwise OR to check both players in one operation
    int word_idx = action / 64;
    int bit_idx = action % 64;
    
    if (word_idx >= num_words) {
        return true; // Out of bounds is considered occupied
    }
    
    uint64_t mask = static_cast<uint64_t>(1) << bit_idx;
    return ((player_bitboards[0][word_idx] | player_bitboards[1][word_idx]) & mask) != 0;
}

std::vector<std::vector<int>> Gamestate::get_board() const {
    std::vector<std::vector<int>> arr(board_size, std::vector<int>(board_size, 0));
    int total = board_size * board_size;
    
    for (int p_idx = 0; p_idx < 2; p_idx++) {
        for (int w = 0; w < num_words; w++) {
            uint64_t chunk = player_bitboards[p_idx][w];
            if (chunk == 0) {
                continue;
            }
            
            for (int b = 0; b < 64; b++) {
                if ((chunk & (static_cast<uint64_t>(1) << b)) != 0) {
                    int action = w * 64 + b;
                    if (action >= total) {
                        break;
                    }
                    int x = action / board_size;
                    int y = action % board_size;
                    arr[x][y] = (p_idx + 1);
                }
            }
        }
    }
    
    return arr;
}

// MCTS/NN support functions
Gamestate Gamestate::apply_action(int action) const {
    Gamestate new_state(*this);
    new_state.make_move(action, current_player);
    return new_state;
}

std::vector<std::vector<std::vector<float>>> Gamestate::to_tensor() const {
    std::vector<std::vector<std::vector<float>>> tensor(3, 
        std::vector<std::vector<float>>(board_size, 
            std::vector<float>(board_size, 0.0f)));
    
    int p_idx = current_player - 1;
    int opp_idx = 1 - p_idx;
    int total = board_size * board_size;
    
    for (int a = 0; a < total; a++) {
        int x = a / board_size;
        int y = a % board_size;
        
        if (is_bit_set(p_idx, a)) {
            tensor[0][x][y] = 1.0f;
        } else if (is_bit_set(opp_idx, a)) {
            tensor[1][x][y] = 1.0f;
        }
    }
    
    if (current_player == BLACK) {
        for (int i = 0; i < board_size; i++) {
            for (int j = 0; j < board_size; j++) {
                tensor[2][i][j] = 1.0f;
            }
        }
    }
    
    return tensor;
}

int Gamestate::get_action(const Gamestate& child_state) const {
    int total = board_size * board_size;
    for (int a = 0; a < total; a++) {
        if (is_occupied(a) != child_state.is_occupied(a)) {
            return a;
        }
    }
    return -1;
}

// Line checking functions
bool Gamestate::is_five_in_a_row(int action, int player) const {
    int p_idx = player - 1;
    int total = board_size * board_size;
    
    if (action == -1) {
        // Checking entire board
        for (int cell = 0; cell < total; cell++) {
            if (is_bit_set(p_idx, cell)) {
                if (_check_line_for_five(cell, player, p_idx)) {
                    return true;
                }
            }
        }
        return false;
    } else {
        // Check just the specific action
        if (!is_bit_set(p_idx, action)) {
            return false;
        }
        return _check_line_for_five(action, player, p_idx);
    }
}

bool Gamestate::_check_line_for_five(int cell, int player, int p_idx) const noexcept {
    if (!is_bit_set(p_idx, cell)) {
        return false;
    }
    
    int x = cell / board_size;
    int y = cell % board_size;
    
    for (int d = 0; d < 4; d++) {
        int dx = _dirs[2*d];
        int dy = _dirs[2*d + 1];
        int forward = _count_direction(x, y, dx, dy, p_idx);
        int backward = _count_direction(x, y, -dx, -dy, p_idx) - 1;
        int length = forward + backward;
        
        if (player == BLACK) {
            if (use_renju || use_omok) {
                if (length == 5) {
                    return true;
                }
            } else {
                if (length >= 5) {
                    return true;
                }
            }
        } else {
            if (use_renju) {
                if (length == 5) {
                    return true;
                }
            } else if (use_omok) {
                if (length >= 5) {
                    return true;
                }
            } else {
                if (length >= 5) {
                    return true;
                }
            }
        }
    }
    return false;
}

int Gamestate::_count_direction(int x0, int y0, int dx, int dy, int p_idx) const noexcept {
    int count = 0;
    int x = x0;
    int y = y0;
    int bs = board_size;
    int action;
    
    while (0 <= x && x < bs && 0 <= y && y < bs) {
        action = x * bs + y;
        if (is_bit_set(p_idx, action)) {
            count++;
            x += dx;
            y += dy;
        } else {
            break;
        }
    }
    
    return count;
}

// Forbidden move checks
bool Gamestate::is_black_renju_forbidden(int action) {
    set_bit(0, action);
    bool forbidden = false;
    
    if (renju_is_overline(action)) {
        forbidden = true;
    } else if (renju_double_four_or_more(action)) {
        forbidden = true;
    } else if (!_is_allowed_double_three(action)) {
        // _is_allowed_double_three returns true if the move is allowed (i.e. not double-three)
        // and false if the move would create a disallowed double-three.
        forbidden = true;
    }
    
    clear_bit(0, action);
    return forbidden;
}

bool Gamestate::is_black_omok_forbidden(int action) {
    set_bit(0, action);
    bool is_forbidden = false;
    
    if (_omok_is_overline(action)) {
        is_forbidden = true;
    } else if (_omok_check_double_three_strict(action)) {
        is_forbidden = true;
    }
    
    clear_bit(0, action);
    return is_forbidden;
}

// Overline checks
bool Gamestate::renju_is_overline(int action) const {
    int x0, y0, direction, dx, dy, nx, ny, count_line, bs = board_size;
    auto [x0_val, y0_val] = action_to_coords_pair(action);
    x0 = x0_val;
    y0 = y0_val;
    
    for (direction = 0; direction < 4; direction++) {
        dx = _dirs[2*direction];
        dy = _dirs[2*direction + 1];
        count_line = 1;
        
        nx = x0 + dx;
        ny = y0 + dy;
        while (0 <= nx && nx < bs && 0 <= ny && ny < bs) {
            if (is_bit_set(0, coords_to_action(nx, ny))) {
                count_line++;
                nx += dx;
                ny += dy;
            } else {
                break;
            }
        }
        
        nx = x0 - dx;
        ny = y0 - dy;
        while (0 <= nx && nx < bs && 0 <= ny && ny < bs) {
            if (is_bit_set(0, coords_to_action(nx, ny))) {
                count_line++;
                nx -= dx;
                ny -= dy;
            } else {
                break;
            }
        }
        
        if (count_line >= 6) {
            return true;
        }
    }
    return false;
}

bool Gamestate::_omok_is_overline(int action) const {
    return renju_is_overline(action);
}

// Renju forbidden checks
bool Gamestate::renju_double_four_or_more(int action) const {
    int c4 = _renju_count_all_fours();
    return (c4 >= 2);
}

// Improved renju_double_three_or_more function
bool Gamestate::renju_double_three_or_more(int action) const {
    Gamestate tmp = copy();
    tmp.set_bit(0, action);
    
    // Get the unified set of three patterns.
    std::vector<std::set<int>> three_patterns = tmp._get_three_patterns_for_action(action);
    
    tmp.clear_bit(0, action);
    
    // If 2 or more distinct three patterns exist, then it's a double-three.
    return (three_patterns.size() >= 2);
}

// Enhanced double-three detection
bool Gamestate::_is_allowed_double_three(int action) const {
    // Step 1: Get all three patterns that include this action
    std::vector<std::set<int>> three_patterns = _get_three_patterns_for_action(action);
    
    // If there's fewer than 2 three patterns, it's not a double-three
    if (three_patterns.size() < 2) {
        return true; // Not a double-three, so it's allowed
    }
    
    // Apply rule 9.3(a): Check how many threes can be made into straight fours
    int straight_four_capable_count = _count_straight_four_capable_threes(three_patterns);
    
    // If at most one of the threes can be made into a straight four, the double-three is allowed
    if (straight_four_capable_count <= 1) {
        return true;
    }
    
    // Apply rule 9.3(b): Recursive check for potential future double-threes
    return _is_double_three_allowed_recursive(three_patterns);
}

std::vector<std::set<int>> Gamestate::_get_three_patterns_for_action(int action) const {
    std::vector<std::set<int>> three_patterns;
    int bs = board_size;
    std::vector<std::pair<int, int>> directions = { {0, 1}, {1, 0}, {1, 1}, {-1, 1} };
    
    auto [x0, y0] = action_to_coords_pair(action);
    
    for (auto [dx, dy] : directions) {
        std::vector<std::pair<int, int>> line_cells;
        // Build a line of up to 7 cells centered on the action.
        for (int offset = -3; offset <= 3; offset++) {
            int nx = x0 + offset * dx;
            int ny = y0 + offset * dy;
            if (_in_bounds(nx, ny)) {
                line_cells.push_back({nx, ny});
            }
        }
        
        // Slide a 5-cell window over the line.
        for (size_t start = 0; start + 4 < line_cells.size(); start++) {
            std::vector<std::pair<int, int>> segment(line_cells.begin() + start, line_cells.begin() + start + 5);
            
            // Check if this segment forms a three pattern containing our action.
            if (_is_three_pattern(segment, action)) {
                std::set<int> pattern;
                for (auto [x, y] : segment) {
                    pattern.insert(coords_to_action(x, y));
                }
                
                // Unify: check if this pattern overlaps in at least 3 cells with any existing one.
                bool duplicate = false;
                for (const auto &existing : three_patterns) {
                    std::set<int> inter;
                    std::set_intersection(existing.begin(), existing.end(),
                                          pattern.begin(), pattern.end(),
                                          std::inserter(inter, inter.begin()));
                    if (inter.size() >= 3) {  // Overlap is significant; consider it the same three.
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    three_patterns.push_back(pattern);
                }
            }
        }
    }
    return three_patterns;
}

bool Gamestate::_is_three_pattern(const std::vector<std::pair<int, int>>& segment, int action) const {
    // A three pattern has exactly 3 black stones, the rest empty, and can form a four
    
    int black_count = 0;
    int white_count = 0;
    bool contains_action = false;
    
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            black_count++;
            if (a == action) {
                contains_action = true;
            }
        } else if (is_bit_set(1, a)) {
            white_count++;
        }
    }
    
    if (black_count != 3 || white_count > 0 || !contains_action) {
        return false;
    }
    
    // Check if this pattern can form a four by placing a stone in an empty spot
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            // Temporarily place a black stone here
            Gamestate tmp = copy();
            tmp.set_bit(0, a);
            
            // Check if this forms a four
            if (tmp._is_four_pattern(segment)) {
                return true;
            }
        }
    }
    
    return false;
}

bool Gamestate::_is_four_pattern(const std::vector<std::pair<int, int>>& segment) const {
    // A four pattern has exactly 4 black stones and can form a five
    
    int black_count = 0;
    int white_count = 0;
    
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            black_count++;
        } else if (is_bit_set(1, a)) {
            white_count++;
        }
    }
    
    if (black_count != 4 || white_count > 0) {
        return false;
    }
    
    // Check if there's at least one empty spot that would form a five
    for (auto [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            return true;
        }
    }
    
    return false;
}

bool Gamestate::_can_make_straight_four(const std::set<int>& three_pattern) const {
    // Get candidate placements that might convert the three into a four.
    std::vector<int> possible_placements = _find_three_to_four_placements(three_pattern);
    for (int placement : possible_placements) {
        // Create a temporary state with the candidate move.
        Gamestate tmp = copy();
        tmp.set_bit(0, placement);
        // Form a new pattern by adding the candidate.
        std::set<int> new_pattern = three_pattern;
        new_pattern.insert(placement);
        // Extract only the black stone positions from new_pattern.
        std::set<int> black_positions;
        for (int a : new_pattern) {
            if (tmp.is_bit_set(0, a))
                black_positions.insert(a);
        }
        // Only consider candidate patterns that yield exactly 4 black stones.
        if (black_positions.size() != 4)
            continue;
        // If the new pattern qualifies as a straight four and doesn't create an overline, count it.
        if (tmp._is_straight_four(new_pattern)) {
            if (!tmp.renju_is_overline(placement))
                return true;
        }
    }
    return false;
}

std::vector<int> Gamestate::_find_three_to_four_placements(const std::set<int>& three_pattern) const {
    std::vector<int> placements;
    
    // Convert pattern to coordinates for easier analysis
    std::vector<std::pair<int, int>> coords;
    for (int a : three_pattern) {
        coords.push_back(action_to_coords_pair(a));
    }
    
    // Sort coordinates to find the pattern direction
    bool is_horizontal = true;
    bool is_vertical = true;
    bool is_diag_down = true;
    bool is_diag_up = true;
    
    for (size_t i = 1; i < coords.size(); i++) {
        if (coords[i].second != coords[0].second) is_horizontal = false;
        if (coords[i].first != coords[0].first) is_vertical = false;
        if (coords[i].first - coords[0].first != coords[i].second - coords[0].second) is_diag_down = false;
        if (coords[i].first - coords[0].first != coords[0].second - coords[i].second) is_diag_up = false;
    }
    
    // Determine direction vector
    int dx = 0, dy = 0;
    if (is_horizontal) {
        dx = 0; dy = 1;
    } else if (is_vertical) {
        dx = 1; dy = 0;
    } else if (is_diag_down) {
        dx = 1; dy = 1;
    } else if (is_diag_up) {
        dx = 1; dy = -1;
    } else {
        // Not a straight line, shouldn't happen with valid three patterns
        return placements;
    }
    
    // Find min and max coordinates
    int min_x = coords[0].first, min_y = coords[0].second;
    int max_x = coords[0].first, max_y = coords[0].second;
    
    for (auto [x, y] : coords) {
        min_x = std::min<int>(min_x, x);
        min_y = std::min<int>(min_y, y);
        max_x = std::max<int>(max_x, x);
        max_y = std::max<int>(max_y, y);
    }
    
    // Check for empty spots that could complete a four
    // Need to check both within the pattern and at the ends
    
    // Check within the pattern
    for (int i = 0; i <= 4; i++) {
        int x = min_x + i * dx;
        int y = min_y + i * dy;
        
        if (!_in_bounds(x, y)) continue;
        
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a) && three_pattern.find(a) == three_pattern.end()) {
            placements.push_back(a);
        }
    }
    
    // Check beyond the ends
    int before_x = min_x - dx;
    int before_y = min_y - dy;
    int after_x = max_x + dx;
    int after_y = max_y + dy;
    
    if (_in_bounds(before_x, before_y)) {
        int a = coords_to_action(before_x, before_y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            placements.push_back(a);
        }
    }
    
    if (_in_bounds(after_x, after_y)) {
        int a = coords_to_action(after_x, after_y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            placements.push_back(a);
        }
    }
    
    return placements;
}

bool Gamestate::_is_straight_four(const std::set<int>& pattern) const {
    // Build the segment of coordinates corresponding to the pattern.
    std::vector<std::pair<int,int>> segment;
    for (int a : pattern) {
        segment.push_back(action_to_coords_pair(a));
    }
    // Sort the coordinates (assumes the segment lies along one line).
    std::sort(segment.begin(), segment.end(), [&](const std::pair<int,int>& p1, const std::pair<int,int>& p2) {
        if (p1.first == p2.first)
            return p1.second < p2.second;
        return p1.first < p2.first;
    });

    // Count black and white stones in the segment.
    int black_count = 0, white_count = 0;
    for (const auto &p : segment) {
        int a = coords_to_action(p.first, p.second);
        if (is_bit_set(0, a))
            ++black_count;
        else if (is_bit_set(1, a))
            ++white_count;
    }
    if (white_count > 0)
        return false;
    
    // Only consider a pattern with exactly 4 black stones as a four-shape.
    if (black_count == 4) {
        auto ends = _ends_are_open(segment); // returns {front_open, back_open}
        return (ends.first || ends.second);
    }
    return false;
}

int Gamestate::_count_straight_four_capable_threes(const std::vector<std::set<int>>& three_patterns) const {
    int count = 0;
    
    for (const auto& pattern : three_patterns) {
        if (_can_make_straight_four(pattern)) {
            count++;
        }
    }
    
    return count;
}

bool Gamestate::_is_double_three_allowed_recursive(const std::vector<std::set<int>>& three_patterns, 
                                                 int depth, int max_depth) const {
    // Avoid too deep recursion
    if (depth >= max_depth) {
        return false;
    }
    
    // Apply rule 9.3(a) again at this level
    int straight_four_capable_count = _count_straight_four_capable_threes(three_patterns);
    if (straight_four_capable_count <= 1) {
        return true;
    }
    
    // Apply rule 9.3(b): Check all possible future moves that would create a straight four
    for (const auto& pattern : three_patterns) {
        std::vector<int> placements = _find_three_to_four_placements(pattern);
        
        for (int placement : placements) {
            // Skip if already occupied
            if (is_bit_set(0, placement) || is_bit_set(1, placement)) {
                continue;
            }
            
            // Make temporary move
            Gamestate tmp = copy();
            tmp.set_bit(0, placement);
            
            // Check if this creates a new double-three
            std::vector<std::set<int>> new_three_patterns = tmp._get_three_patterns_for_action(placement);
            if (new_three_patterns.size() >= 2) {
                // Recursively check if this new double-three is allowed
                if (tmp._is_double_three_allowed_recursive(new_three_patterns, depth + 1, max_depth)) {
                    return true;
                }
            }
        }
    }
    
    // If we've checked all possibilities and found no allowed configuration
    return false;
}

// Renju shape detection methods
int Gamestate::_renju_count_all_fours() const {
    int bs = board_size;
    std::set<std::pair<std::set<int>, int>> found_fours;
    std::vector<std::pair<int, int>> directions = {{0,1}, {1,0}, {1,1}, {-1,1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx, dy] : directions) {
                std::vector<std::pair<int, int>> line_cells;
                int xx = x, yy = y;
                int step = 0;
                
                while (step < 7) {
                    if (!_in_bounds(xx, yy)) {
                        break;
                    }
                    line_cells.push_back({xx, yy});
                    xx += dx;
                    yy += dy;
                    step++;
                }
                
                for (int window_size : {5, 6, 7}) {
                    if (line_cells.size() < window_size) {
                        break;
                    }
                    
                    for (size_t start_idx = 0; start_idx <= line_cells.size() - window_size; start_idx++) {
                        std::vector<std::pair<int, int>> segment(
                            line_cells.begin() + start_idx,
                            line_cells.begin() + start_idx + window_size
                        );
                        
                        if (_renju_is_four_shape(segment)) {
                            std::set<int> black_positions = _positions_of_black(segment);
                            bool unified = _try_unify_four_shape(found_fours, black_positions, black_positions.size());
                            
                            if (!unified) {
                                found_fours.insert({black_positions, black_positions.size()});
                            }
                        }
                    }
                }
            }
        }
    }
    
    return found_fours.size();
}

int Gamestate::_renju_count_all_threes(int action) const {
    int bs = board_size;
    std::set<std::set<int>> found_threes;
    std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx, dy] : directions) {
                std::vector<std::pair<int, int>> line_cells;
                int xx = x, yy = y;
                int step = 0;
                
                while (step < 7) {
                    if (!_in_bounds(xx, yy)) {
                        break;
                    }
                    line_cells.push_back({xx, yy});
                    xx += dx;
                    yy += dy;
                    step++;
                }
                
                for (int window_size : {5, 6}) {
                    if (line_cells.size() < window_size) {
                        break;
                    }
                    
                    for (size_t start_idx = 0; start_idx <= line_cells.size() - window_size; start_idx++) {
                        std::vector<std::pair<int, int>> segment(
                            line_cells.begin() + start_idx,
                            line_cells.begin() + start_idx + window_size
                        );
                        
                        if (_renju_is_three_shape(segment)) {
                            std::set<int> black_positions = _positions_of_black(segment);
                            std::set<int> new_fs(black_positions);
                            
                            if (!_try_unify_three_shape(found_threes, new_fs, action)) {
                                found_threes.insert(new_fs);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return found_threes.size();
}

bool Gamestate::_renju_is_three_shape(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    int black_count = 0, white_count = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            black_count++;
        } else if (is_bit_set(1, a)) {
            white_count++;
        }
    }
    
    if (white_count > 0 || black_count < 2 || black_count >= 4) {
        return false;
    }
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            Gamestate tmp = copy();
            tmp.set_bit(0, a);
            
            if (tmp._renju_is_four_shape(segment)) {
                return true;
            }
        }
    }
    
    return false;
}

bool Gamestate::_renju_is_four_shape(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    int black_count = 0, white_count = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(1, a)) {
            white_count++;
        } else if (is_bit_set(0, a)) {
            black_count++;
        }
    }
    
    if (white_count > 0) {
        return false;
    }
    
    if (black_count < 3 || black_count > 4) {
        return false;
    }
    
    auto [front_open, back_open] = _ends_are_open(segment);
    
    if (black_count == 4) {
        return (front_open || back_open);
    } else {
        return _check_broken_four(segment, front_open, back_open);
    }
}

std::pair<bool, bool> Gamestate::_ends_are_open(const std::vector<std::pair<int, int>>& segment) const {
    int seg_len = segment.size();
    if (seg_len < 2) {
        return {false, false};
    }
    
    auto [x0, y0] = segment[0];
    auto [x1, y1] = segment[seg_len - 1];
    bool front_open = false, back_open = false;
    
    int dx = 0, dy = 0;
    if (seg_len >= 2) {
        auto [x2, y2] = segment[1];
        dx = x2 - x0;
        dy = y2 - y0;
    }
    
    int fx = x0 - dx;
    int fy = y0 - dy;
    if (_in_bounds(fx, fy)) {
        int af = coords_to_action(fx, fy);
        if (!is_bit_set(0, af) && !is_bit_set(1, af)) {
            front_open = true;
        }
    }
    
    int lx = x1 + dx;
    int ly = y1 + dy;
    if (_in_bounds(lx, ly)) {
        int ab = coords_to_action(lx, ly);
        if (!is_bit_set(0, ab) && !is_bit_set(1, ab)) {
            back_open = true;
        }
    }
    
    return {front_open, back_open};
}

bool Gamestate::_check_broken_four(const std::vector<std::pair<int, int>>& segment, bool front_open, bool back_open) const {
    if (!front_open && !back_open) {
        return false;
    }
    
    std::vector<std::pair<int, int>> empties;
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (!is_bit_set(0, a) && !is_bit_set(1, a)) {
            empties.push_back({x, y});
        }
    }
    
    if (empties.size() != 1) {
        return false;
    }
    
    auto [gapx, gapy] = empties[0];
    int gap_action = coords_to_action(gapx, gapy);
    
    Gamestate tmp = copy();
    tmp.set_bit(0, gap_action);
    bool is_now_4 = tmp._simple_is_4_contiguous(segment);
    
    return is_now_4;
}

bool Gamestate::_simple_is_4_contiguous(const std::vector<std::pair<int, int>>& segment) const {
    int consecutive = 0, best = 0;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            consecutive++;
            if (consecutive > best) {
                best = consecutive;
            }
        } else {
            consecutive = 0;
        }
    }
    
    return (best >= 4);
}

std::set<int> Gamestate::_positions_of_black(const std::vector<std::pair<int, int>>& segment) const {
    std::set<int> black_set;
    
    for (const auto& [x, y] : segment) {
        int a = coords_to_action(x, y);
        if (is_bit_set(0, a)) {
            black_set.insert(a);
        }
    }
    
    return black_set;
}

bool Gamestate::_try_unify_four_shape(std::set<std::pair<std::set<int>, int>>& found_fours, 
                                     const std::set<int>& new_fs, int size) const {
    for (const auto& [existing_fs, existing_size] : found_fours) {
        std::set<int> intersection;
        std::set_intersection(
            existing_fs.begin(), existing_fs.end(),
            new_fs.begin(), new_fs.end(),
            std::inserter(intersection, intersection.begin())
        );
        
        if (intersection.size() >= 3) {
            return true;
        }
    }
    
    return false;
}

bool Gamestate::_try_unify_three_shape(std::set<std::set<int>>& found_threes, 
                                      const std::set<int>& new_fs, int action) const {
    for (const auto& existing_fs : found_threes) {
        std::set<int> intersection;
        std::set_intersection(
            existing_fs.begin(), existing_fs.end(),
            new_fs.begin(), new_fs.end(),
            std::inserter(intersection, intersection.begin())
        );
        
        // Remove action from intersection
        intersection.erase(action);
        
        if (!intersection.empty()) {
            return true;
        }
    }
    
    return false;
}

// Omok specific methods
std::vector<std::set<int>> Gamestate::_get_open_three_patterns_globally() const {
    int bs = board_size;
    std::set<std::set<int>> found_threes;
    std::vector<std::pair<int, int>> directions = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    
    for (int x = 0; x < bs; x++) {
        for (int y = 0; y < bs; y++) {
            for (auto [dx0, dy0] : directions) {
                std::vector<std::pair<int, int>> cells_5;
                int step = 0;
                int cx = x, cy = y;
                
                while (step < 5) {
                    if (!_in_bounds(cx, cy)) {
                        break;
                    }
                    cells_5.push_back({cx, cy});
                    cx += dx0;
                    cy += dy0;
                    step++;
                }
                
                if (cells_5.size() == 5) {
                    std::set<int> triple = _check_open_three_5slice(cells_5);
                    if (!triple.empty()) {
                        bool skip = false;
                        std::vector<std::set<int>> to_remove;
                        
                        for (const auto& existing : found_threes) {
                            // Check if existing is a superset of triple
                            if (std::includes(existing.begin(), existing.end(), 
                                            triple.begin(), triple.end())) {
                                skip = true;
                                break;
                            }
                            
                            // Check if triple is a superset of existing
                            if (std::includes(triple.begin(), triple.end(), 
                                            existing.begin(), existing.end())) {
                                to_remove.push_back(existing);
                            }
                        }
                        
                        if (!skip) {
                            for (const auto& r : to_remove) {
                                found_threes.erase(r);
                            }
                            found_threes.insert(triple);
                        }
                    }
                }
            }
        }
    }
    
    return std::vector<std::set<int>>(found_threes.begin(), found_threes.end());
}

bool Gamestate::_are_patterns_connected(const std::set<int>& pattern1, const std::set<int>& pattern2) const {
    for (int cell1 : pattern1) {
        auto [ax, ay] = action_to_coords_pair(cell1);
        
        for (int cell2 : pattern2) {
            auto [bx, by] = action_to_coords_pair(cell2);
            
            if (abs(ax - bx) <= 1 && abs(ay - by) <= 1) {
                return true;
            }
        }
    }
    return false;
}

bool Gamestate::_omok_check_double_three_strict(int action) const {
    std::vector<std::set<int>> patterns = _get_open_three_patterns_globally();
    int n = patterns.size();
    
    if (n < 2) {
        return false;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (_are_patterns_connected(patterns[i], patterns[j])) {
                return true;
            }
        }
    }
    
    return false;
}

int Gamestate::_count_open_threes_globally() const {
    return _get_open_three_patterns_globally().size();
}

std::set<int> Gamestate::_check_open_three_5slice(const std::vector<std::pair<int, int>>& cells_5) const {
    if (cells_5.size() != 5) {
        return {};
    }
    
    int black_count = 0, white_count = 0, empty_count = 0;
    int arr[5] = {0}; // Represents the contents of cells_5: 0=empty, 1=black, -1=white
    
    for (int i = 0; i < 5; i++) {
        auto [xx, yy] = cells_5[i];
        int act = coords_to_action(xx, yy);
        
        if (is_bit_set(0, act)) {
            black_count++;
            arr[i] = 1;
        } else if (is_bit_set(1, act)) {
            white_count++;
            arr[i] = -1;
        } else {
            empty_count++;
        }
    }
    
    if (black_count != 3 || white_count != 0 || empty_count != 2) {
        return {};
    }
    
    if (arr[0] != 0 || arr[4] != 0) {
        return {};
    }
    
    bool has_triple = false, has_gap = false;
    
    if (arr[1] == 1 && arr[2] == 1 && arr[3] == 1) {
        has_triple = true;
    }
    
    if (arr[1] == 1 && arr[2] == 0 && arr[3] == 1) {
        has_gap = true;
    }
    
    if (!has_triple && !has_gap) {
        return {};
    }
    
    int dx = cells_5[1].first - cells_5[0].first;
    int dy = cells_5[1].second - cells_5[0].second;
    
    int left_x = cells_5[0].first - dx;
    int left_y = cells_5[0].second - dy;
    int right_x = cells_5[4].first + dx;
    int right_y = cells_5[4].second + dy;
    
    // Check if this is an "open" three (both ends must be empty)
    if (_in_bounds(left_x, left_y)) {
        int left_act = coords_to_action(left_x, left_y);
        if (is_bit_set(0, left_act)) {
            return {};
        }
    }
    
    if (_in_bounds(right_x, right_y)) {
        int right_act = coords_to_action(right_x, right_y);
        if (is_bit_set(0, right_act)) {
            return {};
        }
    }
    
    // Get the positions of the three black stones
    std::set<int> triple;
    for (int i = 0; i < 5; i++) {
        if (arr[i] == 1) {
            triple.insert(coords_to_action(cells_5[i].first, cells_5[i].second));
        }
    }
    
    return triple;
}

// Helper methods
std::vector<std::pair<int, int>> Gamestate::_build_entire_line(int x0, int y0, int dx, int dy) const {
    std::vector<std::pair<int, int>> backward_positions;
    std::vector<std::pair<int, int>> forward_positions;
    
    int bx = x0, by = y0;
    while (_in_bounds(bx, by)) {
        backward_positions.push_back({bx, by});
        bx -= dx;
        by -= dy;
    }
    
    std::reverse(backward_positions.begin(), backward_positions.end());
    
    int fx = x0 + dx, fy = y0 + dy;
    while (_in_bounds(fx, fy)) {
        forward_positions.push_back({fx, fy});
        fx += dx;
        fy += dy;
    }
    
    std::vector<std::pair<int, int>> result = backward_positions;
    result.insert(result.end(), forward_positions.begin(), forward_positions.end());
    return result;
}

bool Gamestate::_in_bounds(int x, int y) const {
    return (0 <= x && x < board_size) && (0 <= y && y < board_size);
}

bool Gamestate::_is_pro_long_move_ok(int action, int stone_count) const {
    int center = (board_size / 2) * board_size + (board_size / 2);
    
    if (stone_count == 0 || stone_count == 1) {
        return (action == center);
    } else if (stone_count == 2 || stone_count == 3) {
        if (black_first_stone < 0) {
            return false;
        }
        
        auto [x0, y0] = action_to_coords_pair(black_first_stone);
        auto [x1, y1] = action_to_coords_pair(action);
        int dist = abs(x1 - x0) + abs(y1 - y0);
        return (dist >= 4);
    }
    
    return true;
}

/**
 * Estimates the approximate memory usage of this Gamestate.
 * This helps track memory consumption during search.
 * 
 * @return Approximate memory usage in bytes
 */
size_t Gamestate::approximate_memory_usage() const {
    size_t base_size = sizeof(*this);

    // Calculate board memory
    size_t board_size = 0;
    for (const auto& row : board) {
        board_size += row.size() * sizeof(int);
    }

    // Calculate history memory
    size_t history_size = move_history.size() * sizeof(int);

    return base_size + board_size + history_size;
}

int Gamestate::get_move_count() const {
    return static_cast<int>(move_history.size());
}

PYBIND11_MODULE(gomoku, m) {
    m.doc() = "Pybind11 bindings for the Gomoku game logic";

    py::class_<Gamestate>(m, "Gamestate")
        .def(py::init<int, bool, bool, int, bool>(),
             py::arg("board_size") = 15,
             py::arg("use_renju") = false,
             py::arg("use_omok") = false,
             py::arg("seed") = 0,
             py::arg("use_pro_long_opening") = false,
             "Construct a new Gamestate with optional rule settings and board size")
        .def("copy", &Gamestate::copy, "Return a deep copy of the current game state")
        .def("make_move", &Gamestate::make_move, "Make a move at the specified action for the given player")
        .def("undo_move", &Gamestate::undo_move, "Undo the move at the specified action")
        .def("is_terminal", &Gamestate::is_terminal, "Check if the game is over (win or stalemate)")
        .def("get_winner", &Gamestate::get_winner, "Return the winner (1 for BLACK, 2 for WHITE, 0 if none)")
        .def("get_valid_moves", &Gamestate::get_valid_moves, "Return a list of valid moves")
        .def("get_board", &Gamestate::get_board, "Return a 2D vector representing the current board state")
        .def("apply_action", &Gamestate::apply_action, "Apply an action and return the new game state")
        .def("to_tensor", &Gamestate::to_tensor, "Convert the game state to a tensor for AI training")
        .def("get_action", &Gamestate::get_action, "Get the move that led from the current state to a child state")
        .def("is_five_in_a_row", &Gamestate::is_five_in_a_row, "Check if there is a five-in-a-row from the given cell")
        .def("get_move_count", &Gamestate::get_move_count, "Get the total number of moves made in the game")
        // Expose some key public fields so that Python can inspect them:
        .def_readwrite("board_size", &Gamestate::board_size)
        .def_readwrite("current_player", &Gamestate::current_player)
        .def_readwrite("action", &Gamestate::action)
        .def_readwrite("black_first_stone", &Gamestate::black_first_stone)
        .def_readwrite("use_renju", &Gamestate::use_renju)
        .def_readwrite("use_omok", &Gamestate::use_omok)
        .def_readwrite("use_pro_long_opening", &Gamestate::use_pro_long_opening)
        ;
}
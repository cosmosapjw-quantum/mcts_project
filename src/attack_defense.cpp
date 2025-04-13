#include "attack_defense.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hash_specializations.h"

namespace py = pybind11;

AttackDefenseModule::AttackDefenseModule(int board_size) 
    : board_size_(board_size) {
}

std::pair<std::vector<float>, std::vector<float>> AttackDefenseModule::compute_bonuses(
    const std::vector<std::vector<std::vector<int>>>& board_batch,
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    auto attack = compute_attack_bonus(board_batch, chosen_moves, player_batch);
    auto defense = compute_defense_bonus(board_batch, chosen_moves, player_batch);
    
    return {attack, defense};
}

std::vector<float> AttackDefenseModule::compute_attack_bonus(
    const std::vector<std::vector<std::vector<int>>>& board_batch, 
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    const size_t B = board_batch.size();
    std::vector<std::vector<std::vector<int>>> board_pre(board_batch);
    std::vector<bool> mask(B, false);
    
    // Check if moves are by the current player
    for (size_t i = 0; i < B; i++) {
        int action = chosen_moves[i];
        int row = action / board_size_;
        int col = action % board_size_;
        mask[i] = (board_pre[i][row][col] == player_batch[i]);
    }
    
    // Clear the moves to calculate "before" state
    for (size_t i = 0; i < B; i++) {
        if (mask[i]) {
            int action = chosen_moves[i];
            int row = action / board_size_;
            int col = action % board_size_;
            board_pre[i][row][col] = 0;
        }
    }
    
    // Calculate threats after the move
    auto threats_after = count_threats_for_color(board_batch, player_batch);
    
    // Calculate threats before the move
    auto threats_before = count_threats_for_color(board_pre, player_batch);
    
    // Calculate the difference (attack score)
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = threats_after[i] - threats_before[i];
    }
    
    return result;
}

std::vector<float> AttackDefenseModule::compute_defense_bonus(
    const std::vector<std::vector<std::vector<int>>>& board_batch, 
    const std::vector<int>& chosen_moves,
    const std::vector<int>& player_batch) {
    
    const size_t B = board_batch.size();
    std::vector<std::vector<std::vector<int>>> board_pre(board_batch);
    std::vector<bool> mask(B, false);
    
    // Check if moves are by the current player
    for (size_t i = 0; i < B; i++) {
        int action = chosen_moves[i];
        int row = action / board_size_;
        int col = action % board_size_;
        mask[i] = (board_pre[i][row][col] == player_batch[i]);
    }
    
    // Clear the moves to calculate "before" state
    for (size_t i = 0; i < B; i++) {
        if (mask[i]) {
            int action = chosen_moves[i];
            int row = action / board_size_;
            int col = action % board_size_;
            board_pre[i][row][col] = 0;
        }
    }
    
    // Calculate opponent IDs
    std::vector<int> opponent_batch(B);
    for (size_t i = 0; i < B; i++) {
        opponent_batch[i] = player_batch[i] == 1 ? 2 : 1;
    }
    
    // Calculate threats for opponent after the move
    auto threats_post = count_threats_for_color(board_batch, opponent_batch);
    
    // Calculate threats for opponent before the move
    auto threats_pre = count_threats_for_color(board_pre, opponent_batch);
    
    // Calculate the difference (defense score)
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = threats_pre[i] - threats_post[i];
    }
    
    return result;
}

std::vector<float> AttackDefenseModule::count_threats_for_color(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& opponent_ids) {
    
    auto open_three_hv = count_open_threats_horiz_vert(boards, opponent_ids, 5, 3);
    auto open_four_hv = count_open_threats_horiz_vert(boards, opponent_ids, 6, 4);
    auto diag_open_three = count_open_threats_diagonals(boards, opponent_ids, 5, 3);
    auto diag_open_four = count_open_threats_diagonals(boards, opponent_ids, 6, 4);
    
    const size_t B = boards.size();
    std::vector<float> result(B);
    
    for (size_t i = 0; i < B; i++) {
        result[i] = open_three_hv[i] + open_four_hv[i] + diag_open_three[i] + diag_open_four[i];
    }
    
    return result;
}

// Create a mask where 1 represents the player's stones
std::vector<std::vector<std::vector<float>>> AttackDefenseModule::create_mask(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& player_ids) {
    
    const size_t B = boards.size();
    const size_t H = boards[0].size();
    const size_t W = boards[0][0].size();
    
    std::vector<std::vector<std::vector<float>>> mask(B, 
        std::vector<std::vector<float>>(1, 
            std::vector<float>(H * W, 0.0f)));
    
    for (size_t b = 0; b < B; b++) {
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                if (boards[b][i][j] == player_ids[b]) {
                    mask[b][0][i * W + j] = 1.0f;
                }
            }
        }
    }
    
    return mask;
}

// Create a mask where 1 represents empty cells
std::vector<std::vector<std::vector<float>>> AttackDefenseModule::create_empty_mask(
    const std::vector<std::vector<std::vector<int>>>& boards) {
    
    const size_t B = boards.size();
    const size_t H = boards[0].size();
    const size_t W = boards[0][0].size();
    
    std::vector<std::vector<std::vector<float>>> mask(B, 
        std::vector<std::vector<float>>(1, 
            std::vector<float>(H * W, 0.0f)));
    
    for (size_t b = 0; b < B; b++) {
        for (size_t i = 0; i < H; i++) {
            for (size_t j = 0; j < W; j++) {
                if (boards[b][i][j] == 0) {
                    mask[b][0][i * W + j] = 1.0f;
                }
            }
        }
    }
    
    return mask;
}

// Transpose a mask (swap height and width dimensions)
std::vector<std::vector<std::vector<float>>> AttackDefenseModule::transpose(
    const std::vector<std::vector<std::vector<float>>>& mask) {
    
    const size_t B = mask.size();
    const size_t C = mask[0].size();
    // Fix the sqrt conversion issue by using a temp double first
    const double sqrtResult = std::sqrt(static_cast<double>(mask[0][0].size()));
    const size_t H = static_cast<size_t>(sqrtResult);
    const size_t W = H;
    
    std::vector<std::vector<std::vector<float>>> transposed(B,
        std::vector<std::vector<float>>(C,
            std::vector<float>(H * W, 0.0f)));
    
    for (size_t b = 0; b < B; b++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t i = 0; i < H; i++) {
                for (size_t j = 0; j < W; j++) {
                    transposed[b][c][j * H + i] = mask[b][c][i * W + j];
                }
            }
        }
    }
    
    return transposed;
}

std::vector<float> AttackDefenseModule::count_open_threats_horiz_vert(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& opponent_ids,
    int window_length,
    int required_sum) {
    
    // Create opponent mask and empty mask
    auto opp_mask = create_mask(boards, opponent_ids);
    auto empty_mask = create_empty_mask(boards);
    
    // Count patterns in horizontal direction
    auto horiz = count_1d_patterns(opp_mask, empty_mask, window_length, required_sum);
    
    // Transpose masks for vertical direction
    auto opp_mask_vert = transpose(opp_mask);
    auto empty_mask_vert = transpose(empty_mask);
    
    // Count patterns in vertical direction
    auto vert = count_1d_patterns(opp_mask_vert, empty_mask_vert, window_length, required_sum);
    
    const size_t B = boards.size();
    std::vector<float> result(B);
    
    for (size_t i = 0; i < B; i++) {
        result[i] = horiz[i] + vert[i];
    }
    
    return result;
}

std::vector<float> AttackDefenseModule::count_1d_patterns(
    const std::vector<std::vector<std::vector<float>>>& opp_mask,
    const std::vector<std::vector<std::vector<float>>>& empty_mask,
    int window_length,
    int required_sum) {
    
    const size_t B = opp_mask.size();
    // Fix the sqrt conversion issue by using a temp double first
    const double sqrtResult = std::sqrt(static_cast<double>(opp_mask[0][0].size()));
    const size_t H = static_cast<size_t>(sqrtResult);
    const size_t W = H;
    
    std::vector<float> perfect_counts(B, 0.0f);
    std::vector<float> broken_counts(B, 0.0f);
    
    // For each board in the batch
    for (size_t b = 0; b < B; b++) {
        // For each row
        for (size_t i = 0; i < H; i++) {
            // For each possible window start in the row
            for (size_t j = 0; j <= W - window_length; j++) {
                float opp_sum_full = 0.0f;
                float opp_sum_border = 0.0f;
                float empty_sum_border = 0.0f;
                
                // Sum opponents in the window
                for (int k = 0; k < window_length; k++) {
                    size_t idx = i * W + (j + k);
                    opp_sum_full += opp_mask[b][0][idx];
                    
                    // Sum border cells (first and last)
                    if (k == 0 || k == window_length - 1) {
                        opp_sum_border += opp_mask[b][0][idx];
                        empty_sum_border += empty_mask[b][0][idx];
                    }
                }
                
                // Calculate interior sum (exclude border cells)
                float opp_sum_interior = opp_sum_full - opp_sum_border;
                
                // Check for perfect pattern (required opponent stones in interior, both borders empty)
                if (opp_sum_interior == required_sum && empty_sum_border == 2.0f) {
                    perfect_counts[b] += 1.0f;
                }
                // Check for broken pattern (one less than required opponent stones, both borders empty)
                else if (opp_sum_interior == (required_sum - 1) && empty_sum_border == 2.0f) {
                    broken_counts[b] += 1.0f;
                }
            }
        }
    }
    
    // Sum perfect and broken counts
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = perfect_counts[i] + broken_counts[i];
    }
    
    return result;
}

std::vector<float> AttackDefenseModule::count_open_threats_diagonals(
    const std::vector<std::vector<std::vector<int>>>& boards,
    const std::vector<int>& opponent_ids,
    int window_length,
    int required_sum) {
    
    const size_t B = boards.size();
    const size_t H = boards[0].size();
    const size_t W = boards[0][0].size();
    
    // Create opponent and empty masks
    auto opp_mask = create_mask(boards, opponent_ids);
    auto empty_mask = create_empty_mask(boards);
    
    std::vector<float> diag_count_main(B, 0.0f);
    std::vector<float> diag_count_anti(B, 0.0f);
    
    // For each board in the batch
    for (size_t b = 0; b < B; b++) {
        // Main diagonals (top-left to bottom-right)
        for (size_t i = 0; i <= H - window_length; i++) {
            for (size_t j = 0; j <= W - window_length; j++) {
                float opp_sum_full = 0.0f;
                float opp_sum_border = 0.0f;
                float empty_sum_border = 0.0f;
                
                // Sum opponents in the diagonal window
                for (int k = 0; k < window_length; k++) {
                    size_t idx = (i + k) * W + (j + k);
                    opp_sum_full += opp_mask[b][0][idx];
                    
                    // Sum border cells (first and last)
                    if (k == 0 || k == window_length - 1) {
                        opp_sum_border += opp_mask[b][0][idx];
                        empty_sum_border += empty_mask[b][0][idx];
                    }
                }
                
                // Calculate interior sum (exclude border cells)
                float opp_sum_interior = opp_sum_full - opp_sum_border;
                
                // Check for perfect pattern
                if (opp_sum_interior == required_sum && empty_sum_border == 2.0f) {
                    diag_count_main[b] += 1.0f;
                }
                // Check for broken pattern
                else if (opp_sum_interior == (required_sum - 1) && empty_sum_border == 2.0f) {
                    diag_count_main[b] += 1.0f;
                }
            }
        }
        
        // Anti-diagonals (top-right to bottom-left)
        for (size_t i = 0; i <= H - window_length; i++) {
            for (size_t j = window_length - 1; j < W; j++) {
                float opp_sum_full = 0.0f;
                float opp_sum_border = 0.0f;
                float empty_sum_border = 0.0f;
                
                // Sum opponents in the diagonal window
                for (int k = 0; k < window_length; k++) {
                    size_t idx = (i + k) * W + (j - k);
                    opp_sum_full += opp_mask[b][0][idx];
                    
                    // Sum border cells (first and last)
                    if (k == 0 || k == window_length - 1) {
                        opp_sum_border += opp_mask[b][0][idx];
                        empty_sum_border += empty_mask[b][0][idx];
                    }
                }
                
                // Calculate interior sum (exclude border cells)
                float opp_sum_interior = opp_sum_full - opp_sum_border;
                
                // Check for perfect pattern
                if (opp_sum_interior == required_sum && empty_sum_border == 2.0f) {
                    diag_count_anti[b] += 1.0f;
                }
                // Check for broken pattern
                else if (opp_sum_interior == (required_sum - 1) && empty_sum_border == 2.0f) {
                    diag_count_anti[b] += 1.0f;
                }
            }
        }
    }
    
    // Sum main and anti-diagonal counts
    std::vector<float> result(B);
    for (size_t i = 0; i < B; i++) {
        result[i] = diag_count_main[i] + diag_count_anti[i];
    }
    
    return result;
}

PYBIND11_MODULE(attack_defense, m) {
    m.doc() = "Attack and Defense calculation module for Gomoku/Omok";
    
    py::class_<AttackDefenseModule>(m, "AttackDefenseModule")
        .def(py::init<int>(), py::arg("board_size"))
        .def("__call__", [](AttackDefenseModule& self, 
                           py::array_t<float> board_np, 
                           py::array_t<int64_t> moves_np,
                           py::array_t<int64_t> player_np) {
            auto board_buffer = board_np.request();
            auto moves_buffer = moves_np.request();
            auto player_buffer = player_np.request();
            
            // Get shape information - using appropriate types for pybind
            const auto batch_size = static_cast<size_t>(board_buffer.shape[0]);
            const auto board_height = static_cast<size_t>(board_buffer.shape[1]);
            const auto board_width = static_cast<size_t>(board_buffer.shape[2]);
            
            // Convert numpy arrays to C++ vectors
            std::vector<std::vector<std::vector<int>>> board_batch(batch_size, 
                std::vector<std::vector<int>>(board_height, 
                    std::vector<int>(board_width, 0)));
            
            std::vector<int> chosen_moves(batch_size, 0);
            std::vector<int> player_batch(batch_size, 0);
            
            // Copy data from numpy arrays
            float* board_ptr = static_cast<float*>(board_buffer.ptr);
            int64_t* moves_ptr = static_cast<int64_t*>(moves_buffer.ptr);
            int64_t* player_ptr = static_cast<int64_t*>(player_buffer.ptr);
            
            // Fill board data
            for (size_t b = 0; b < batch_size; b++) {
                for (size_t i = 0; i < board_height; i++) {
                    for (size_t j = 0; j < board_width; j++) {
                        board_batch[b][i][j] = static_cast<int>(board_ptr[b * board_height * board_width + i * board_width + j]);
                    }
                }
                
                // Copy moves and player IDs
                chosen_moves[b] = static_cast<int>(moves_ptr[b]);
                player_batch[b] = static_cast<int>(player_ptr[b]);
            }
            
            // Call the C++ implementation
            auto [attack_bonus, defense_bonus] = self.compute_bonuses(board_batch, chosen_moves, player_batch);
            
            // Convert back to numpy arrays - fix the shape parameter to avoid narrowing
            std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(attack_bonus.size())};
            py::array_t<float> attack_bonus_np(shape);
            py::array_t<float> defense_bonus_np(shape);
            
            auto attack_buffer = attack_bonus_np.request();
            auto defense_buffer = defense_bonus_np.request();
            
            float* attack_ptr = static_cast<float*>(attack_buffer.ptr);
            float* defense_ptr = static_cast<float*>(defense_buffer.ptr);
            
            for (size_t i = 0; i < attack_bonus.size(); i++) {
                attack_ptr[i] = attack_bonus[i];
                defense_ptr[i] = defense_bonus[i];
            }
            
            return py::make_tuple(attack_bonus_np, defense_bonus_np);
        });
}
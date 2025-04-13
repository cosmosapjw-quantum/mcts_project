#pragma once

#include <string>
#include <utility>
#include <vector>

#include "hash_specializations.h"

class AttackDefenseModule {
public:
    AttackDefenseModule(int board_size);
    
    // Calculate attack and defense bonuses
    std::pair<std::vector<float>, std::vector<float>> compute_bonuses(
        const std::vector<std::vector<std::vector<int>>>& board_batch,
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch);

private:
    int board_size_;
    
    // Internal implementations
    std::vector<float> compute_attack_bonus(
        const std::vector<std::vector<std::vector<int>>>& board_batch, 
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch);
    
    std::vector<float> compute_defense_bonus(
        const std::vector<std::vector<std::vector<int>>>& board_batch, 
        const std::vector<int>& chosen_moves,
        const std::vector<int>& player_batch);
    
    std::vector<float> count_threats_for_color(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& opponent_ids);
    
    std::vector<float> count_open_threats_horiz_vert(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& opponent_ids,
        int window_length,
        int required_sum);
    
    std::vector<float> count_open_threats_diagonals(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& opponent_ids,
        int window_length,
        int required_sum);
    
    std::vector<float> count_1d_patterns(
        const std::vector<std::vector<std::vector<float>>>& opp_mask,
        const std::vector<std::vector<std::vector<float>>>& empty_mask,
        int window_length,
        int required_sum);
    
    // Helper functions
    std::vector<std::vector<std::vector<float>>> create_mask(
        const std::vector<std::vector<std::vector<int>>>& boards,
        const std::vector<int>& player_ids);
    
    std::vector<std::vector<std::vector<float>>> create_empty_mask(
        const std::vector<std::vector<std::vector<int>>>& boards);
    
    std::vector<std::vector<std::vector<float>>> transpose(
        const std::vector<std::vector<std::vector<float>>>& mask);
};
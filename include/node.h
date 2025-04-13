// node.h
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include "gomoku.h"  // from your user-provided code

/**
 * MCTS Node that references a Gamestate. 
 * We store "moveFromParent" to know which action led to this child.
 */
class Node {
public:
    Node(const Gamestate& state, int moveFromParent=-1, float prior=0.0f);

    // Basic MCTS stats
    float get_q_value() const;
    int   get_visit_count() const;
    float get_prior() const;

    void update_stats(float value);

    // Expand children once
    void expand(const std::vector<int>& moves, const std::vector<float>& priors);

    bool is_leaf() const { return children_.empty(); }
    const Gamestate& get_state() const { return state_; }
    Node* get_parent() const { return parent_; }

    int get_move_from_parent() const { return move_from_parent_; }
    std::vector<Node*> get_children() const;

private:
    Gamestate state_;            // The game state at this node
    Node* parent_;               // pointer to parent
    float prior_;                // prior probability (P(s,a))

    std::atomic<float> total_value_;
    std::atomic<int>   visit_count_;

    int move_from_parent_;

    std::vector<std::unique_ptr<Node>> children_;
    mutable std::mutex expand_mutex_;
};

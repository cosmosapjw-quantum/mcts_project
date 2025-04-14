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
    // Add static counter for debugging
    static std::atomic<int> total_nodes_;
    
    // Update constructor to increment counter
    Node(const Gamestate& state, int moveFromParent=-1, float prior=0.0f) 
        : state_(state),
          parent_(nullptr),
          prior_(prior),
          total_value_(0.0f),
          visit_count_(0),
          move_from_parent_(moveFromParent)
    {
        total_nodes_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Update destructor to decrement counter
    ~Node() {
        total_nodes_.fetch_sub(1, std::memory_order_relaxed);
    }

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

    void limit_tree_depth(int current_depth, int max_depth) {
        if (current_depth >= max_depth) {
            // Truncate children at max depth
            std::lock_guard<std::mutex> lock(expand_mutex_);
            children_.clear();
            return;
        }
        
        // Recursively limit children depth
        std::vector<Node*> kids = get_children();
        for (Node* child : kids) {
            if (child) {
                child->limit_tree_depth(current_depth + 1, max_depth);
            }
        }
    }

    // Add these methods for virtual loss
    void add_virtual_loss();
    void remove_virtual_loss();

private:
    Gamestate state_;            // The game state at this node
    Node* parent_;               // pointer to parent
    float prior_;                // prior probability (P(s,a))

    std::atomic<float> total_value_;
    std::atomic<int>   visit_count_;

    int move_from_parent_;

    std::vector<std::unique_ptr<Node>> children_;
    mutable std::mutex expand_mutex_;
    // Instead of using raw parent pointer, let's track if node is owned by another node
    bool is_owned_ = false;

    // Add atomic counter for virtual losses
    std::atomic<int> virtual_losses_{0};
};

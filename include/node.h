// node.h
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include "gomoku.h"

/**
 * MCTS Node that references a Gamestate. 
 * We store "moveFromParent" to know which action led to this child.
 */
class Node {
public:
    static std::atomic<int> total_nodes_;
    
    Node(const Gamestate& state, int moveFromParent=-1, float prior=0.0f) 
        : state_(state),
          parent_(nullptr),
          prior_(prior),
          total_value_(0.0f),
          visit_count_(0),
          move_from_parent_(moveFromParent),
          virtual_losses_(0)
    {
        total_nodes_.fetch_add(1, std::memory_order_relaxed);
    }
    
    ~Node() {
        total_nodes_.fetch_sub(1, std::memory_order_relaxed);
    }

    float get_q_value() const;
    int get_visit_count() const;
    float get_prior() const;

    void update_stats(float value);
    void expand(const std::vector<int>& moves, const std::vector<float>& priors);

    bool is_leaf() const { return children_.empty(); }
    const Gamestate& get_state() const { return state_; }
    Node* get_parent() const { return parent_; }

    int get_move_from_parent() const { return move_from_parent_; }
    std::vector<Node*> get_children() const;

    void limit_tree_depth(int current_depth, int max_depth) {
        if (current_depth >= max_depth) {
            std::lock_guard<std::mutex> lock(expand_mutex_);
            children_.clear();
            return;
        }
        
        std::vector<Node*> kids = get_children();
        for (Node* child : kids) {
            if (child) {
                child->limit_tree_depth(current_depth + 1, max_depth);
            }
        }
    }

    void add_virtual_loss();
    void remove_virtual_loss();

    int get_virtual_losses() const { 
        return virtual_losses_.load(std::memory_order_acquire); 
    }
    
    // Access the expand mutex for thread-safe operations
    std::mutex& get_expand_mutex() const { return expand_mutex_; }

private:
    Gamestate state_;
    Node* parent_;
    float prior_;

    std::atomic<float> total_value_;
    std::atomic<int> visit_count_;

    int move_from_parent_;

    std::vector<std::unique_ptr<Node>> children_;
    mutable std::mutex expand_mutex_;
    
    std::atomic<int> virtual_losses_{0};
};
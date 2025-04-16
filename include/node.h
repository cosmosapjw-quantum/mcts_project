#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <shared_mutex>  // For reader-writer lock
#include <map>
#include <chrono>
#include <cmath>         // Include for std::pow
#include "gomoku.h"
#include "debug.h"

/**
 * Memory-efficient MCTS Node that references a Gamestate. 
 * Optimized for concurrent access and memory usage.
 */
class Node {
public:
    // Track overall node count for memory monitoring
    static std::atomic<int> total_nodes_;
    static std::atomic<size_t> total_memory_bytes_;
    
    // Static configuration for memory management
    static constexpr int MAX_CHILDREN_DEFAULT = 64;
    static constexpr float PRUNE_THRESHOLD = 0.01f;  // Prune nodes with < 1% visit probability
    
    // Memory management constants
    static constexpr int MAX_NODES_SOFT_LIMIT = 100000;  // Start pruning at 100K nodes
    static constexpr int MAX_NODES_HARD_LIMIT = 500000;  // Hard limit at 500K nodes
    
    Node(const Gamestate& state, int moveFromParent=-1, float prior=0.0f);
    ~Node();

    // Thread-safe getters for node statistics
    float get_q_value() const;
    int get_visit_count() const;
    float get_prior() const;
    bool is_leaf() const;
    const Gamestate& get_state() const;
    Node* get_parent() const;
    int get_move_from_parent() const;
    std::vector<Node*> get_children() const;

    // Update node statistics with improved thread safety
    void update_stats(float value);
    
    // Memory-efficient node expansion with adaptive strategies
    void expand(const std::vector<int>& moves, const std::vector<float>& priors);

    // Improved virtual loss handling
    void add_virtual_loss();
    void remove_virtual_loss();
    int get_virtual_losses() const;
    
    // Memory management and optimization
    static size_t get_memory_usage_kb();
    int get_tree_depth() const;
    std::map<std::string, int> collect_tree_stats() const;
    std::vector<int> get_principal_variation() const;
    int prune_low_visit_branches(float visit_threshold = PRUNE_THRESHOLD);
    int prune_tree(float visit_threshold = PRUNE_THRESHOLD);

private:
    Gamestate state_;
    Node* parent_;
    float prior_;

    std::atomic<float> total_value_;
    std::atomic<int> visit_count_;

    int move_from_parent_;
    bool is_fully_expanded_;

    std::vector<std::unique_ptr<Node>> children_;
    mutable std::shared_mutex expand_mutex_;  // Reader-writer lock for expansion
    mutable std::shared_mutex rw_mutex_;      // Reader-writer lock for stats
    
    std::atomic<int> virtual_losses_{0};
    
    // Track node creation time for age-based pruning if needed
    std::chrono::steady_clock::time_point creation_timestamp_;
    
    // Helper methods for different expansion strategies
    void expand_normal(const std::vector<int>& moves, const std::vector<float>& priors);
    void expand_with_pruning(const std::vector<int>& moves, const std::vector<float>& priors);
    void expand_limited(const std::vector<int>& moves, const std::vector<float>& priors, size_t max_children);

    bool can_add_children() const;

    // Progressive widening parameters
    static constexpr float PROGRESSIVE_WIDENING_C = 2.0f;
    static constexpr float PROGRESSIVE_WIDENING_ALPHA = 0.5f;
    
    // Helper method for progressive widening
    bool should_add_child_progressive_widening() const {
        int n = visit_count_.load(std::memory_order_acquire);
        int k = children_.size();
        
        if (n <= 0) return true;  // Always expand the first child
        
        float max_children = PROGRESSIVE_WIDENING_C * std::pow(static_cast<float>(n), PROGRESSIVE_WIDENING_ALPHA);
        return k < max_children;
    }
};

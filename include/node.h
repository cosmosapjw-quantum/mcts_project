// node.h - Memory-efficient optimized version
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <shared_mutex>  // For reader-writer lock
#include <map>
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
    
    // Static configuration for memory management
    static constexpr int MAX_CHILDREN_DEFAULT = 64;
    static constexpr float PRUNE_THRESHOLD = 0.01f;  // Prune nodes with < 1% visit probability
    
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

    // Thread-safe getters for node statistics
    float get_q_value() const {
        // Read-only operation - use shared lock
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        
        int vc = visit_count_.load(std::memory_order_acquire);
        int vl = virtual_losses_.load(std::memory_order_acquire);
        
        // If no real visits and no virtual losses, return prior * 0.5 as a default value
        if (vc == 0 && vl == 0) {
            return 0.0f;
        }
        
        float tv = total_value_.load(std::memory_order_acquire);
        
        // Apply virtual loss effect - each virtual loss is treated as a loss (-1)
        float virtual_loss_value = -1.0f * vl;
        
        // Return adjusted Q value
        return (tv + virtual_loss_value) / (float)(vc + vl);
    }

    int get_visit_count() const {
        return visit_count_.load(std::memory_order_acquire);
    }

    float get_prior() const {
        // Prior is immutable, no lock needed
        return prior_;
    }

    // Update node statistics with improved thread safety
    void update_stats(float value) {
        // Use exclusive lock for writing
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        
        visit_count_.fetch_add(1, std::memory_order_acq_rel);
        
        // For atomic<float>, we need to use compare_exchange_weak
        float current = total_value_.load(std::memory_order_acquire);
        float desired = current + value;
        while (!total_value_.compare_exchange_weak(current, desired,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
            desired = current + value;
        }
    }
    
    // Memory-efficient node expansion
    void expand(const std::vector<int>& moves, const std::vector<float>& priors) {
        // Use exclusive lock for writing
        std::unique_lock<std::shared_mutex> lock(expand_mutex_);
        
        // Already expanded - early return
        if (!children_.empty()) {
            return;
        }
        
        // Check if we're approaching memory limits and apply pruning strategies
        if (total_nodes_.load() > MAX_NODES_HARD_LIMIT) {
            // Hard memory limit reached - only expand a few most promising nodes
            expand_limited(moves, priors, 3);
        }
        else if (total_nodes_.load() > MAX_NODES_SOFT_LIMIT) {
            // Soft memory limit reached - apply pruning strategy
            expand_with_pruning(moves, priors);
        }
        else {
            // Normal expansion with reasonable limits
            expand_normal(moves, priors);
        }
    }

    bool is_leaf() const { 
        // Read-only operation - use shared lock
        std::shared_lock<std::shared_mutex> lock(expand_mutex_);
        return children_.empty(); 
    }
    
    const Gamestate& get_state() const { 
        // State is immutable after construction, no lock needed
        return state_; 
    }
    
    // Updated to return nullptr if no parent
    Node* get_parent() const { 
        // Parent pointer is immutable after expand, no lock needed
        return parent_; 
    }

    int get_move_from_parent() const { 
        // Move is immutable, no lock needed
        return move_from_parent_; 
    }
    
    // Thread-safe access to children with read lock
    std::vector<Node*> get_children() const {
        std::shared_lock<std::shared_mutex> lock(expand_mutex_);
        
        std::vector<Node*> result;
        result.reserve(children_.size());
        
        for (const auto& c : children_) {
            if (c) {  // Add null check to be defensive
                result.push_back(c.get());
            }
        }
        
        return result;
    }

    // Limit tree depth for memory efficiency
    void limit_tree_depth(int current_depth, int max_depth) {
        if (current_depth >= max_depth) {
            std::unique_lock<std::shared_mutex> lock(expand_mutex_);
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

    // Improved virtual loss handling with atomic operations
    void add_virtual_loss() {
        // Atomic increment, no lock needed
        MCTS_DEBUG("Adding virtual loss to node with move " << move_from_parent_);
        virtual_losses_.fetch_add(1, std::memory_order_acq_rel);
    }

    void remove_virtual_loss() {
        // Atomic decrement with floor check
        MCTS_DEBUG("Removing virtual loss from node with move " << move_from_parent_);
        int prev = virtual_losses_.fetch_sub(1, std::memory_order_acq_rel);
        
        // Ensure we don't go below zero (defensive programming)
        if (prev <= 0) {
            MCTS_DEBUG("WARNING: virtual_losses_ went negative, resetting to 0");
            virtual_losses_.store(0, std::memory_order_release);
        }
    }

    int get_virtual_losses() const { 
        return virtual_losses_.load(std::memory_order_acquire); 
    }
    
    // Memory usage statistics
    static size_t get_approx_memory_usage() {
        // Rough estimate of memory used by all nodes
        int node_count = total_nodes_.load(std::memory_order_acquire);
        constexpr size_t approx_node_size = 
            sizeof(Node) +     // Base class size
            15*15*sizeof(int) + // Typical gamestate board
            8 * sizeof(void*);  // Average overhead for children pointers
        
        return node_count * approx_node_size;
    }
    
    // Prune low-value branches to save memory
    int prune_low_visit_branches(float visit_threshold = 0.05f) {
        std::unique_lock<std::shared_mutex> lock(expand_mutex_);
        
        if (children_.empty()) {
            return 0;
        }
        
        int total_visits = visit_count_.load(std::memory_order_acquire);
        if (total_visits < 10) {
            // Don't prune nodes with too few visits
            return 0;
        }
        
        int pruned_count = 0;
        std::vector<std::unique_ptr<Node>> remaining_children;
        
        for (auto& child : children_) {
            if (!child) continue;
            
            float visit_ratio = static_cast<float>(child->get_visit_count()) / total_visits;
            
            if (visit_ratio >= visit_threshold) {
                remaining_children.push_back(std::move(child));
            } else {
                pruned_count++;
            }
        }
        
        if (pruned_count > 0) {
            MCTS_DEBUG("Pruned " << pruned_count << " low-visit branches");
            children_ = std::move(remaining_children);
        }
        
        return pruned_count;
    }

    // Add memory tracking functionality
    size_t get_memory_usage_kb();

    // Add tree statistics collection
    std::map<std::string, int> collect_tree_stats() const;

    // Helper method to identify the most critical nodes
    std::vector<Node*> get_critical_path() const;

    bool is_transposition(const Gamestate& state) const;
    int prune_tree(float visit_threshold);

private:
    Gamestate state_;
    Node* parent_;
    float prior_;

    std::atomic<float> total_value_;
    std::atomic<int> visit_count_;

    int move_from_parent_;

    std::vector<std::unique_ptr<Node>> children_;
    mutable std::shared_mutex expand_mutex_;  // Reader-writer lock for expansion
    mutable std::shared_mutex rw_mutex_;      // Reader-writer lock for stats
    
    std::atomic<int> virtual_losses_{0};
    
    // Memory management constants
    static constexpr int MAX_NODES_SOFT_LIMIT = 100000;
    static constexpr int MAX_NODES_HARD_LIMIT = 500000;
    
    // Helper method for normal expansion (no memory pressure)
    void expand_normal(const std::vector<int>& moves, const std::vector<float>& priors) {
        const size_t MAX_CHILDREN = MAX_CHILDREN_DEFAULT;
        size_t num_children = std::min(moves.size(), MAX_CHILDREN);
        
        children_.reserve(num_children);
        
        for (size_t i = 0; i < num_children; i++) {
            try {
                Gamestate childState = state_.copy();
                childState.make_move(moves[i], state_.current_player);
                
                auto child = std::make_unique<Node>(childState, moves[i], 
                    (i < priors.size()) ? priors[i] : 1.0f/num_children);
                    
                child->parent_ = this;
                children_.push_back(std::move(child));
            } catch (const std::exception& e) {
                // Log error and continue
                MCTS_DEBUG("Error creating child node: " << e.what());
            }
        }
        
        MCTS_DEBUG("Expanded node with " << children_.size() << " children (normal)");
    }
    
    // Helper method for expansion with pruning (soft memory pressure)
    void expand_with_pruning(const std::vector<int>& moves, const std::vector<float>& priors) {
        // Find total prior sum and calculate threshold
        float total_prior = 0.0f;
        for (size_t i = 0; i < priors.size(); i++) {
            total_prior += priors[i];
        }
        
        // Sort moves by prior probability
        std::vector<std::pair<int, float>> move_priors;
        move_priors.reserve(moves.size());
        
        for (size_t i = 0; i < moves.size() && i < priors.size(); i++) {
            move_priors.emplace_back(moves[i], priors[i]);
        }
        
        // Sort in descending order of prior probability
        std::sort(move_priors.begin(), move_priors.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Determine how many children to create
        size_t num_children = move_priors.size();
        float cum_prob = 0.0f;
        
        // Only expand nodes that contribute significantly to the total prior
        for (size_t i = 0; i < move_priors.size(); i++) {
            cum_prob += move_priors[i].second / total_prior;
            if (cum_prob > 0.95f || i >= MAX_CHILDREN_DEFAULT / 2) {
                num_children = i + 1;
                break;
            }
        }
        
        // Apply a minimum to avoid overly aggressive pruning
        num_children = std::max(num_children, size_t(3));
        
        // Create children for the most promising moves
        children_.reserve(num_children);
        
        for (size_t i = 0; i < num_children; i++) {
            try {
                Gamestate childState = state_.copy();
                childState.make_move(move_priors[i].first, state_.current_player);
                
                auto child = std::make_unique<Node>(childState, move_priors[i].first, move_priors[i].second);
                    
                child->parent_ = this;
                children_.push_back(std::move(child));
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error creating child node: " << e.what());
            }
        }
        
        MCTS_DEBUG("Expanded node with " << children_.size() << " children (pruned from " 
                  << moves.size() << " possible moves)");
    }
    
    // Helper method for limited expansion (hard memory pressure)
    void expand_limited(const std::vector<int>& moves, const std::vector<float>& priors, size_t max_children) {
        // Sort moves by prior probability
        std::vector<std::pair<int, float>> move_priors;
        move_priors.reserve(moves.size());
        
        for (size_t i = 0; i < moves.size() && i < priors.size(); i++) {
            move_priors.emplace_back(moves[i], priors[i]);
        }
        
        // Sort in descending order of prior probability
        std::sort(move_priors.begin(), move_priors.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Only expand the top few most promising moves
        size_t num_children = std::min(move_priors.size(), max_children);
        
        children_.reserve(num_children);
        
        for (size_t i = 0; i < num_children; i++) {
            try {
                Gamestate childState = state_.copy();
                childState.make_move(move_priors[i].first, state_.current_player);
                
                auto child = std::make_unique<Node>(childState, move_priors[i].first, move_priors[i].second);
                    
                child->parent_ = this;
                children_.push_back(std::move(child));
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error creating child node: " << e.what());
            }
        }
        
        MCTS_DEBUG("Expanded node with " << children_.size() << " children (hard limit, "
                  << "total nodes: " << total_nodes_.load() << ")");
    }
};

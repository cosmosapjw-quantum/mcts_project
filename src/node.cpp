// node.cpp - Memory-optimized implementation
#include "node.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <map>

#include "debug.h"

// Define the static member
std::atomic<int> Node::total_nodes_(0);
std::atomic<size_t> Node::total_memory_bytes_(0);

// Memory-efficient implementation of the constructor
Node::Node(const Gamestate& state, int moveFromParent, float prior) 
    : state_(state),
      parent_(nullptr),
      prior_(prior),
      total_value_(0.0f),
      visit_count_(0),
      move_from_parent_(moveFromParent),
      virtual_losses_(0),
      is_fully_expanded_(false),
      creation_timestamp_(std::chrono::steady_clock::now())
{
    // Increment counter with atomic operation
    total_nodes_.fetch_add(1, std::memory_order_relaxed);
    
    // Track memory usage (approximate)
    size_t this_node_size = sizeof(Node) + state.approximate_memory_usage();
    total_memory_bytes_.fetch_add(this_node_size, std::memory_order_relaxed);
}

// Destructor to properly clean up
Node::~Node() {
    // Decrement counter with atomic operation
    total_nodes_.fetch_sub(1, std::memory_order_relaxed);
    
    // Approximate memory freed
    size_t this_node_size = sizeof(Node) + state_.approximate_memory_usage();
    total_memory_bytes_.fetch_sub(this_node_size, std::memory_order_relaxed);
}

float Node::get_q_value() const {
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

int Node::get_visit_count() const {
    return visit_count_.load(std::memory_order_acquire);
}

float Node::get_prior() const {
    // Prior is immutable, no lock needed
    return prior_;
}

bool Node::is_leaf() const { 
    // Read-only operation - use shared lock
    std::shared_lock<std::shared_mutex> lock(expand_mutex_);
    return children_.empty(); 
}

const Gamestate& Node::get_state() const { 
    // State is immutable after construction, no lock needed
    return state_; 
}

Node* Node::get_parent() const { 
    // Parent pointer is immutable after expand, no lock needed
    return parent_; 
}

int Node::get_move_from_parent() const { 
    // Move is immutable, no lock needed
    return move_from_parent_; 
}

bool Node::can_add_children() const {
    return should_add_child_progressive_widening();
}

// Updated thread-safe expand method with smart policy refinement
void Node::expand(const std::vector<int>& moves, const std::vector<float>& priors) {
    // Use exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(expand_mutex_);
    
    // If already fully expanded, early return
    if (is_fully_expanded_) {
        return;
    }
    
    // Progressive widening: don't expand too many children too quickly
    if (!children_.empty() && !should_add_child_progressive_widening()) {
        return;
    }
    
    // Check if we're approaching memory limits and apply pruning strategies
    bool memory_constrained = total_nodes_.load() > MAX_NODES_SOFT_LIMIT;
    size_t memory_usage_mb = total_memory_bytes_.load() / (1024 * 1024);
    
    // Hard memory limit
    if (total_nodes_.load() > MAX_NODES_HARD_LIMIT || memory_usage_mb > 1000) {
        // Only expand a few most promising nodes
        MCTS_DEBUG("Memory limit reached (" << total_nodes_.load() << " nodes, " 
                 << memory_usage_mb << " MB), using limited expansion");
        expand_limited(moves, priors, 5);
        
        // Mark as fully expanded to avoid further expansion attempts
        is_fully_expanded_ = true;
        return;
    }
    else if (memory_constrained) {
        // Soft memory limit - apply pruning strategy
        MCTS_DEBUG("Memory constraint detected (" << total_nodes_.load() << " nodes, " 
                 << memory_usage_mb << " MB), using pruning strategy");
        expand_with_pruning(moves, priors);
        
        // Mark as fully expanded to avoid further expansion attempts
        is_fully_expanded_ = true;
        return;
    }
    
    // First expansion of node - use normal expansion
    if (children_.empty()) {
        expand_normal(moves, priors);
        
        // Not fully expanded for progressive widening
        is_fully_expanded_ = false;
    }
    // Progressive expansion - add one more child
    else {
        // Find the best move not yet expanded
        std::vector<std::pair<int, float>> unexpanded_moves;
        
        // Identify already expanded moves
        std::set<int> expanded_moves;
        for (const auto& child : children_) {
            if (child) {
                expanded_moves.insert(child->get_move_from_parent());
            }
        }
        
        // Find unexpanded moves with their priors
        for (size_t i = 0; i < moves.size() && i < priors.size(); i++) {
            if (expanded_moves.find(moves[i]) == expanded_moves.end()) {
                unexpanded_moves.emplace_back(moves[i], priors[i]);
            }
        }
        
        // If all moves are expanded, mark as fully expanded
        if (unexpanded_moves.empty()) {
            is_fully_expanded_ = true;
            return;
        }
        
        // Sort by prior probability
        std::sort(unexpanded_moves.begin(), unexpanded_moves.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Add the best unexpanded move
        try {
            int move = unexpanded_moves[0].first;
            float prior = unexpanded_moves[0].second;
            
            Gamestate childState = state_.copy();
            childState.make_move(move, state_.current_player);
            
            auto child = std::make_unique<Node>(childState, move, prior);
            child->parent_ = this;
            children_.push_back(std::move(child));
            
            MCTS_DEBUG("Progressive widening: added child with move " << move 
                      << ", prior " << prior
                      << ", now have " << children_.size() << " children");
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error creating child node: " << e.what());
        }
    }
}

// Thread-safe access to children
std::vector<Node*> Node::get_children() const {
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

// Improved virtual loss handling with atomic operations
void Node::add_virtual_loss() {
    // Atomic increment, no lock needed
    virtual_losses_.fetch_add(1, std::memory_order_acq_rel);
}

void Node::remove_virtual_loss() {
    // Atomic decrement with floor check
    int prev = virtual_losses_.fetch_sub(1, std::memory_order_acq_rel);
    
    // Ensure we don't go below zero (defensive programming)
    if (prev <= 0) {
        virtual_losses_.store(0, std::memory_order_release);
    }
}

int Node::get_virtual_losses() const { 
    return virtual_losses_.load(std::memory_order_acquire); 
}

// Update node statistics with improved thread safety
void Node::update_stats(float value) {
    // Use exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    
    visit_count_.fetch_add(1, std::memory_order_acq_rel);
    
    // For atomic<float>, we need to use compare_exchange_weak in a loop
    float current = total_value_.load(std::memory_order_acquire);
    float desired = current + value;
    while (!total_value_.compare_exchange_weak(current, desired,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        desired = current + value;
    }
}

// Memory usage tracking
size_t Node::get_memory_usage_kb() {
    return total_memory_bytes_.load(std::memory_order_acquire) / 1024;
}

// Tree depth analysis
int Node::get_tree_depth() const {
    int depth = 0;
    const Node* current = this;
    
    while (current) {
        depth++;
        
        // Get children
        auto children = current->get_children();
        if (children.empty()) break;
        
        // Find child with highest visits
        const Node* best_child = nullptr;
        int max_visits = -1;
        
        for (const Node* child : children) {
            if (!child) continue;
            
            int visits = child->get_visit_count();
            if (visits > max_visits) {
                max_visits = visits;
                best_child = child;
            }
        }
        
        current = best_child;
    }
    
    return depth;
}

// Tree statistics collection
std::map<std::string, int> Node::collect_tree_stats() const {
    std::map<std::string, int> stats;
    stats["total_nodes"] = total_nodes_.load(std::memory_order_acquire);
    stats["memory_kb"] = total_memory_bytes_.load(std::memory_order_acquire) / 1024;
    stats["depth"] = get_tree_depth();
    stats["max_visits"] = visit_count_.load(std::memory_order_acquire);
    
    // Calculate average branching factor
    int total_branches = 0;
    int node_count = 0;
    
    // Helper function to calculate branching factor
    std::function<void(const Node*)> calculate_branching = [&](const Node* node) {
        if (!node) return;
        
        auto children = node->get_children();
        int valid_children = 0;
        
        for (const Node* child : children) {
            if (child) {
                valid_children++;
                calculate_branching(child);
            }
        }
        
        if (valid_children > 0) {
            total_branches += valid_children;
            node_count++;
        }
    };
    
    calculate_branching(this);
    
    stats["branching_factor"] = node_count > 0 ? total_branches / node_count : 0;
    
    return stats;
}

// Return the primary variation (most visited path)
std::vector<int> Node::get_principal_variation() const {
    std::vector<int> pv;
    const Node* current = this;
    
    // Maximum PV length to avoid infinite loops
    const int MAX_PV_LENGTH = 30;
    
    while (current && pv.size() < MAX_PV_LENGTH) {
        auto children = current->get_children();
        if (children.empty()) break;
        
        // Find most visited child
        Node* best_child = nullptr;
        int max_visits = -1;
        
        for (Node* child : children) {
            if (!child) continue;
            
            int visits = child->get_visit_count();
            if (visits > max_visits) {
                max_visits = visits;
                best_child = child;
            }
        }
        
        if (!best_child) break;
        
        // Add move to PV
        pv.push_back(best_child->get_move_from_parent());
        
        // Move to next node
        current = best_child;
    }
    
    return pv;
}

// Memory-efficient tree pruning
int Node::prune_low_visit_branches(float visit_threshold) {
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

// Recursively prune the tree
int Node::prune_tree(float visit_threshold) {
    int pruned = 0;
    
    // First prune this node's low-visit children
    pruned += prune_low_visit_branches(visit_threshold);
    
    // Then recursively prune all children
    for (auto& child_ptr : children_) {
        if (child_ptr) {
            pruned += child_ptr->prune_tree(visit_threshold);
        }
    }
    
    return pruned;
}

// Private helper methods

// Normal expansion with reasonable limits
void Node::expand_normal(const std::vector<int>& moves, const std::vector<float>& priors) {
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

// Expansion with pruning for soft memory constraints
void Node::expand_with_pruning(const std::vector<int>& moves, const std::vector<float>& priors) {
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

// Limited expansion for hard memory constraints
void Node::expand_limited(const std::vector<int>& moves, const std::vector<float>& priors, size_t max_children) {
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
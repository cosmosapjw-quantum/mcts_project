// node.cpp - Optimized implementation
#include "node.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <map>

#include "debug.h"

// Define the static member
std::atomic<int> Node::total_nodes_(0);

// Add memory tracking functionality
size_t Node::get_memory_usage_kb() {
    size_t total_bytes = get_approx_memory_usage();
    return total_bytes / 1024;
}

// Add tree statistics collection
std::map<std::string, int> Node::collect_tree_stats() const {
    std::map<std::string, int> stats;
    stats["total_nodes"] = total_nodes_.load(std::memory_order_acquire);
    stats["depth"] = 0;
    stats["branching_factor"] = 0;
    stats["max_visits"] = visit_count_.load(std::memory_order_acquire);
    
    // Calculate average branching factor and maximum depth
    int total_branches = 0;
    int branch_count = 0;
    int max_depth = 0;
    
    // Helper function for recursive traversal
    std::function<void(const Node*, int)> traverse = [&](const Node* node, int depth) {
        if (!node) return;
        
        max_depth = std::max(max_depth, depth);
        
        auto children = node->get_children();
        if (!children.empty()) {
            total_branches += children.size();
            branch_count++;
        }
        
        stats["max_visits"] = std::max(stats["max_visits"], 
                                      node->get_visit_count());
        
        for (Node* child : children) {
            if (child) {
                traverse(child, depth + 1);
            }
        }
    };
    
    // Start traversal from this node
    traverse(this, 0);
    
    // Calculate average branching factor
    stats["depth"] = max_depth;
    stats["branching_factor"] = branch_count > 0 ? total_branches / branch_count : 0;
    
    return stats;
}

// Helper method to identify the most critical nodes
std::vector<Node*> Node::get_critical_path() const {
    std::vector<Node*> path;
    const Node* current = this;
    
    while (current) {
        path.push_back(const_cast<Node*>(current));
        
        // Get children
        auto children = current->get_children();
        if (children.empty()) break;
        
        // Find child with highest visit count
        Node* best_child = nullptr;
        int max_visits = -1;
        
        for (Node* child : children) {
            if (child) {
                int visits = child->get_visit_count();
                if (visits > max_visits) {
                    max_visits = visits;
                    best_child = child;
                }
            }
        }
        
        current = best_child;
    }
    
    return path;
}

// Add transposition awareness to avoid duplicate states
bool Node::is_transposition(const Gamestate& state) const {
    // In Gomoku, a transposition would be the same board state
    // This is a simplified check - for a full implementation, you'd need proper
    // game-specific transposition detection
    const auto& my_board = state_.get_board();
    const auto& other_board = state.get_board();
    
    // Simple board comparison
    if (my_board.size() != other_board.size()) return false;
    
    for (size_t i = 0; i < my_board.size(); ++i) {
        if (my_board[i].size() != other_board[i].size()) return false;
        
        for (size_t j = 0; j < my_board[i].size(); ++j) {
            if (my_board[i][j] != other_board[i][j]) return false;
        }
    }
    
    return true;
}

// Add memory-efficient tree pruning
int Node::prune_tree(float visit_threshold) {
    int pruned = 0;
    
    // First prune this node's children
    pruned += prune_low_visit_branches(visit_threshold);
    
    // Then recursively prune all children
    for (auto& child_ptr : children_) {
        if (child_ptr) {
            pruned += child_ptr->prune_tree(visit_threshold);
        }
    }
    
    return pruned;
}
// node.cpp
#include "node.h"
#include <iostream>
#include <numeric>

#include "debug.h"

std::atomic<int> Node::total_nodes_(0);

void Node::add_virtual_loss() {
    MCTS_DEBUG("Adding virtual loss to node with move " << move_from_parent_);
    virtual_losses_.fetch_add(1, std::memory_order_relaxed);
}

void Node::remove_virtual_loss() {
    MCTS_DEBUG("Removing virtual loss from node with move " << move_from_parent_);
    int prev = virtual_losses_.fetch_sub(1, std::memory_order_relaxed);
    
    // Ensure we don't go below zero (defensive programming)
    if (prev <= 0) {
        MCTS_DEBUG("WARNING: virtual_losses_ went negative, resetting to 0");
        virtual_losses_.store(0, std::memory_order_relaxed);
    }
}

float Node::get_q_value() const {
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
    return prior_;
}

void Node::update_stats(float value) {
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

void Node::expand(const std::vector<int>& moves, const std::vector<float>& priors) {
    std::lock_guard<std::mutex> lock(expand_mutex_);
    
    // Already expanded - early return
    if (!children_.empty()) {
        return;
    }
    
    // Check if we should limit nodes
    if (total_nodes_.load() > 4000) { // Approach limit
        // Create only a few children
        size_t max_to_create = std::min<size_t>(5, moves.size());
        
        children_.reserve(max_to_create);
        
        for (size_t i = 0; i < max_to_create && i < moves.size(); i++) {
            try {
                Gamestate childState = state_.copy();
                childState.make_move(moves[i], state_.current_player);
                
                auto child = std::make_unique<Node>(childState, moves[i], 
                    (i < priors.size()) ? priors[i] : 1.0f/max_to_create);
                    
                child->parent_ = this;
                children_.push_back(std::move(child));
            }
            catch (const std::exception& e) {
                // Log error and continue
                std::cerr << "Error creating child node: " << e.what() << std::endl;
            }
        }
        
        return;
    }
    
    // Normal expansion but with strict limits
    const size_t MAX_CHILDREN = 10; // Very strict limit
    size_t num_children = std::min(moves.size(), MAX_CHILDREN);
    
    children_.reserve(num_children);
    
    for (size_t i = 0; i < num_children && i < moves.size(); i++) {
        try {
            Gamestate childState = state_.copy();
            childState.make_move(moves[i], state_.current_player);
            
            auto child = std::make_unique<Node>(childState, moves[i], 
                (i < priors.size()) ? priors[i] : 1.0f/num_children);
                
            child->parent_ = this;
            children_.push_back(std::move(child));
        } catch (const std::exception& e) {
            // Log error and continue
            std::cerr << "Error creating child node: " << e.what() << std::endl;
        }
    }
}

std::vector<Node*> Node::get_children() const {
    std::lock_guard<std::mutex> lock(expand_mutex_);
    
    std::vector<Node*> result;
    result.reserve(children_.size());
    
    for (const auto& c : children_) {
        if (c) {  // Add null check to be defensive
            result.push_back(c.get());
        }
    }
    
    return result;
}
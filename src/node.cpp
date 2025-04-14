// node.cpp
#include "node.h"
#include <iostream>
#include <numeric>

std::atomic<int> Node::total_nodes_(0);

void Node::add_virtual_loss() {
    virtual_losses_.fetch_add(1, std::memory_order_relaxed);
}

void Node::remove_virtual_loss() {
    virtual_losses_.fetch_sub(1, std::memory_order_relaxed);
}

// Update get_q_value to account for virtual losses
float Node::get_q_value() const {
    int vc = visit_count_.load(std::memory_order_acquire);
    int vl = virtual_losses_.load(std::memory_order_acquire);
    if (vc == 0) return 0.f;
    
    float tv = total_value_.load(std::memory_order_acquire);
    
    // Apply virtual loss effect - each virtual loss is treated as a loss (-1)
    return (tv - vl) / (float)(vc + vl);
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
        std::cerr << "WARNING: Approaching node limit, limiting expansion" << std::endl;
        // Create only a few children
        size_t max_to_create = std::min<size_t>(5, moves.size());
        
        children_.reserve(max_to_create);
        
        for (size_t i = 0; i < max_to_create; i++) {
            Gamestate childState = state_.copy();
            childState.make_move(moves[i], state_.current_player);
            
            auto child = std::make_unique<Node>(childState, moves[i], 
                (i < priors.size()) ? priors[i] : 1.0f/max_to_create);
                
            child->parent_ = this;
            children_.push_back(std::move(child));
        }
        
        return;
    }
    
    // Normal expansion but with strict limits
    const size_t MAX_CHILDREN = 10; // Very strict limit
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
            std::cerr << "Error creating child node: " << e.what() << std::endl;
        }
    }
    
    // Log how many nodes we have
    if (total_nodes_.load() % 500 == 0) {
        std::cerr << "MEMORY: Total nodes: " << total_nodes_.load() << std::endl;
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
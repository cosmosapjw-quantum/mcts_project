// node.cpp
#include "node.h"

Node::Node(const Gamestate& state, int moveFromParent, float prior)
 : state_(state),
   parent_(nullptr),
   prior_(prior),
   total_value_(0.0f),
   visit_count_(0),
   move_from_parent_(moveFromParent)
{}

float Node::get_q_value() const {
    int vc = visit_count_.load(std::memory_order_acquire);
    if (vc == 0) return 0.f;
    float tv = total_value_.load(std::memory_order_acquire);
    return tv / (float)vc;
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
    if (!children_.empty()) {
        return; // already expanded
    }
    children_.reserve(moves.size());
    for (size_t i = 0; i < moves.size(); i++) {
        // Create a deep copy of the state
        Gamestate childState = state_.copy();
        // The user-provided code typically has something like `make_move(action, player)`.
        childState.make_move(moves[i], state_.current_player);
        // or if you prefer childState = state_.apply_action(moves[i]) 
        auto child = std::make_unique<Node>(childState, moves[i], priors[i]);
        child->parent_ = this;
        children_.push_back(std::move(child));
    }
}

std::vector<Node*> Node::get_children() const {
    std::lock_guard<std::mutex> lock(expand_mutex_);
    std::vector<Node*> result;
    result.reserve(children_.size());
    for (auto& c : children_) {
        result.push_back(c.get());
    }
    return result;
}
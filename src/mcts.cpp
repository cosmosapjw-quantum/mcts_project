// mcts.cpp
#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <numeric>

MCTS::MCTS(const MCTSConfig& config,
           std::shared_ptr<BatchingNNInterface> nn,
           int boardSize)
 : config_(config),
   nn_(nn),
   simulations_done_(0),
   attackDefense_(boardSize)
{}

int MCTS::select_move_with_temperature(float temperature) const {
    if (!root_) {
        return -1;
    }
    
    std::vector<Node*> children = root_->get_children();
    if (children.empty()) {
        return -1;
    }
    
    std::vector<float> distribution;
    std::vector<int> moves;
    
    for (Node* child : children) {
        if (!child) continue;
        
        moves.push_back(child->get_move_from_parent());
        
        float count = static_cast<float>(child->get_visit_count());
        if (temperature > 0) {
            distribution.push_back(std::pow(count, 1.0f / temperature));
        } else {
            distribution.push_back(count);
        }
    }
    
    if (temperature <= 0) {
        int best_idx = std::distance(distribution.begin(), 
                                   std::max_element(distribution.begin(), distribution.end()));
        return moves[best_idx];
    }
    
    float sum = std::accumulate(distribution.begin(), distribution.end(), 0.0f);
    if (sum > 0) {
        for (float& d : distribution) {
            d /= sum;
        }
    } else {
        for (float& d : distribution) {
            d = 1.0f / distribution.size();
        }
    }
    
    std::discrete_distribution<int> dist(distribution.begin(), distribution.end());
    int selected_idx = dist(rng_);
    
    return moves[selected_idx];
}

void MCTS::add_dirichlet_noise(std::vector<float>& priors) {
    if (priors.empty()) return;
    
    std::gamma_distribution<float> gamma_dist(dirichlet_alpha_, 1.0f);
    std::vector<float> noise(priors.size());
    float noise_sum = 0.0f;
    
    for (size_t i = 0; i < priors.size(); i++) {
        noise[i] = gamma_dist(rng_);
        noise_sum += noise[i];
    }
    
    if (noise_sum > 0) {
        for (float& n : noise) {
            n /= noise_sum;
        }
    }
    
    for (size_t i = 0; i < priors.size(); i++) {
        priors[i] = (1.0f - noise_weight_) * priors[i] + noise_weight_ * noise[i];
    }
    
    float sum = std::accumulate(priors.begin(), priors.end(), 0.0f);
    if (sum > 0) {
        for (float& p : priors) {
            p /= sum;
        }
    }
}

void MCTS::run_search(const Gamestate& rootState) {
    MCTS_DEBUG("Starting MCTS search");
    
    // Always use single-threaded search for now
    root_ = std::make_unique<Node>(rootState);
    simulations_done_ = 0;
    
    for (int i = 0; i < config_.num_simulations; i++) {
        try {
            Node* leaf = select_node(root_.get());
            if (!leaf) continue;
            
            if (leaf->get_state().is_terminal()) {
                float value = 0.0f;
                int winner = leaf->get_state().get_winner();
                int current_player = leaf->get_state().current_player;
                
                if (winner == current_player) {
                    value = 1.0f;
                } else if (winner == 0) {
                    value = 0.0f;
                } else {
                    value = -1.0f;
                }
                
                backup(leaf, value);
            } else {
                expand_and_evaluate(leaf);
            }
        } catch (const std::exception& e) {
            // Silent error handling
        }
    }
    
    MCTS_DEBUG("MCTS search completed");
}

void MCTS::run_parallel_search(const Gamestate& rootState) {
    MCTS_DEBUG("Starting parallel MCTS search");
    root_ = std::make_unique<Node>(rootState);
    simulations_done_ = 0;
    shutdown_flag_ = false;
    
    auto validMoves = rootState.get_valid_moves();
    MCTS_DEBUG("Root has " << validMoves.size() << " valid moves");
    
    if (!validMoves.empty()) {
        std::vector<float> priors(validMoves.size(), 1.0f / validMoves.size());
        MCTS_DEBUG("Adding Dirichlet noise to root node priors");
        add_dirichlet_noise(priors);
        root_->expand(validMoves, priors);
    }
    
    // Clear previous threads
    for (auto& t : threads_) {
        if (t.joinable()) t.detach();
    }
    threads_.clear();
    
    MCTS_DEBUG("Running " << config_.num_simulations << " parallel simulations");
    
    // Run simulations with a fixed number of parallel threads
    const int num_threads = std::min(16, config_.num_threads);
    
    // Create a pool of simulations
    std::vector<int> simulation_indices(config_.num_simulations);
    std::iota(simulation_indices.begin(), simulation_indices.end(), 0);
    
    // Distribute simulations to threads
    std::vector<std::thread> sim_threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * (simulation_indices.size() / num_threads);
        int end = (i == num_threads - 1) ? simulation_indices.size() : 
                 (i + 1) * (simulation_indices.size() / num_threads);
        
        if (start < end) {
            MCTS_DEBUG("Starting thread " << i << " with simulations " << start << " to " << end);
            sim_threads.emplace_back([this, start, end, &simulation_indices]() {
                for (int j = start; j < end; j++) {
                    if (shutdown_flag_) break;
                    run_single_simulation(simulation_indices[j]);
                }
            });
        }
    }
    
    // Wait up to 10 seconds for threads to complete
    auto start_time = std::chrono::steady_clock::now();
    const auto max_wait = std::chrono::seconds(10);
    
    while (simulations_done_.load() < config_.num_simulations) {
        auto current_time = std::chrono::steady_clock::now();
        if (current_time - start_time > max_wait) {
            MCTS_DEBUG("Search timeout reached, forcing termination");
            shutdown_flag_ = true;
            break;
        }
        
        // Check periodically
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        MCTS_DEBUG("Simulations completed: " << simulations_done_.load() << "/" << config_.num_simulations);
    }
    
    // Ensure we're definitely done
    shutdown_flag_ = true;
    
    // Detach all simulation threads - don't try to join
    for (auto& t : sim_threads) {
        if (t.joinable()) t.detach();
    }
    
    MCTS_DEBUG("Parallel search completed with " << simulations_done_.load() << " simulations");
}

void MCTS::queue_leaf_for_evaluation(Node* leaf) {
    if (!leaf) return;
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    LeafTask task;
    task.leaf = leaf;
    task.state = leaf->get_state().copy();
    task.chosen_move = leaf->get_move_from_parent();
    
    leaf_queue_.push(task);
    
    queue_cv_.notify_one();
}

void MCTS::leaf_evaluation_worker() {
    MCTS_DEBUG("Worker thread started");
    const int local_batch_size = 4;  // Use a fixed small batch size for reliability
    
    while (!search_done_) {  // Only continue if search is not done
        // Process at most one batch then check if we should exit
        std::vector<LeafTask> batch;
        batch.reserve(local_batch_size);
        
        // Get tasks from the queue - with minimal locking
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            // Check if should exit immediately after acquiring lock
            if (search_done_) break;
            
            // Take up to local_batch_size items from queue
            int count = 0;
            while (!leaf_queue_.empty() && count < local_batch_size) {
                batch.push_back(leaf_queue_.front());
                leaf_queue_.pop();
                count++;
            }
        }
        
        // Process the batch outside the lock if we got any items
        if (!batch.empty()) {
            try {
                for (const auto& task : batch) {
                    // Process each task individually for better responsiveness
                    if (search_done_) break;  // Exit early if search done
                    
                    // Individual processing of leaves
                    try {
                        evaluate_leaf(task.leaf, task.state, task.chosen_move);
                        simulations_done_.fetch_add(1, std::memory_order_relaxed);
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error evaluating leaf: " << e.what());
                    }
                }
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error in batch processing: " << e.what());
            }
            
            // Notify any waiting threads that we've processed some items
            queue_cv_.notify_all();
        } else {
            // No work - sleep a tiny bit to avoid spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    MCTS_DEBUG("Worker thread exiting");
}

void MCTS::evaluate_leaf(Node* leaf, const Gamestate& state, int chosen_move) {
    if (!leaf) return;
    
    std::vector<Node*> path;
    Node* current = leaf;
    while (current) {
        path.push_back(current);
        current = current->get_parent();
    }
    
    if (state.is_terminal()) {
        for (Node* node : path) {
            node->remove_virtual_loss();
        }
        return;
    }
    
    std::vector<std::vector<int>> board2D = state.get_board(); 
    std::vector<std::vector<std::vector<int>>> board_batch;
    board_batch.push_back(board2D);
    
    std::vector<int> chosen_moves = {chosen_move};
    std::vector<int> player_batch = {state.current_player};
    
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    float attack = attackVec[0];
    float defense = defenseVec[0];
    
    std::vector<float> policy;
    float value = 0.0f;
    nn_->request_inference(state, chosen_move, attack, defense, policy, value);
    
    auto validMoves = state.get_valid_moves();
    std::vector<float> validPolicies;
    validPolicies.reserve(validMoves.size());
    
    for (int move : validMoves) {
        if (move >= 0 && move < static_cast<int>(policy.size())) {
            validPolicies.push_back(policy[move]);
        } else {
            validPolicies.push_back(1.0f / validMoves.size());
        }
    }
    
    float sum = std::accumulate(validPolicies.begin(), validPolicies.end(), 0.0f);
    if (sum > 0) {
        for (auto& p : validPolicies) {
            p /= sum;
        }
    }
    
    leaf->expand(validMoves, validPolicies);
    
    backup(leaf, value);

    for (Node* node : path) {
        node->remove_virtual_loss();
    }
}

void MCTS::evaluate_leaf_batch(const std::vector<LeafTask>& batch) {
    if (batch.empty()) return;
    
    std::vector<std::vector<std::vector<int>>> board_batch;
    std::vector<int> chosen_moves;
    std::vector<int> player_batch;
    
    std::vector<Node*> leaves;
    std::vector<std::vector<float>> policies(batch.size());
    std::vector<float> values(batch.size(), 0.0f);
    
    for (const auto& task : batch) {
        if (!task.leaf) continue;
        
        leaves.push_back(task.leaf);
        
        std::vector<std::vector<int>> board2D = task.state.get_board();
        board_batch.push_back(board2D);
        
        chosen_moves.push_back(task.chosen_move);
        player_batch.push_back(task.state.current_player);
    }
    
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    for (size_t i = 0; i < leaves.size(); i++) {
        Node* leaf = leaves[i];
        const Gamestate& state = batch[i].state;
        int chosen_move = chosen_moves[i];
        float attack = attackVec[i];
        float defense = defenseVec[i];
        
        if (state.is_terminal()) continue;
        
        std::vector<float> policy;
        float value = 0.0f;
        nn_->request_inference(state, chosen_move, attack, defense, policy, value);
        
        policies[i] = policy;
        values[i] = value;
    }
    
    for (size_t i = 0; i < leaves.size(); i++) {
        Node* leaf = leaves[i];
        if (!leaf) continue;
        
        const Gamestate& state = batch[i].state;
        if (state.is_terminal()) continue;
        
        auto validMoves = state.get_valid_moves();
        std::vector<float> validPolicies;
        validPolicies.reserve(validMoves.size());
        
        const auto& policy = policies[i];
        for (int move : validMoves) {
            if (move >= 0 && move < static_cast<int>(policy.size())) {
                validPolicies.push_back(policy[move]);
            } else {
                validPolicies.push_back(1.0f / validMoves.size());
            }
        }
        
        float sum = std::accumulate(validPolicies.begin(), validPolicies.end(), 0.0f);
        if (sum > 0) {
            for (auto& p : validPolicies) {
                p /= sum;
            }
        }
        
        leaf->expand(validMoves, validPolicies);
        
        backup(leaf, values[i]);
    }
}

void MCTS::worker_thread() {
    while (true) {
        if (reached_node_limit()) {
            break;
        }
        
        int current_sim = simulations_done_.load(std::memory_order_acquire);
        if (current_sim >= config_.num_simulations) {
            break;
        }
        
        current_sim = simulations_done_.fetch_add(1, std::memory_order_acq_rel);
        if (current_sim >= config_.num_simulations) {
            break;
        }
        
        try {
            Node* root_ptr = root_.get();
            if (!root_ptr) {
                break;
            }
            
            Node* leaf = select_node(root_ptr);
            if (!leaf) {
                continue;
            }
            
            expand_and_evaluate(leaf);
            
        } catch (const std::exception& e) {
            // Silent error handling
        }
    }
}

void MCTS::run_single_simulation(int sim_index) {
    try {
        // Skip if we should terminate
        if (shutdown_flag_) return;
        
        if (sim_index % 10 == 0) {
            MCTS_DEBUG("Running simulation " << sim_index);
        }
        
        Node* leaf = select_node(root_.get());
        if (!leaf) return;
        
        if (leaf->get_state().is_terminal()) {
            float value = 0.0f;
            int winner = leaf->get_state().get_winner();
            int current_player = leaf->get_state().current_player;
            
            if (winner == current_player) {
                value = 1.0f;
            } else if (winner == 0) {
                value = 0.0f;
            } else {
                value = -1.0f;
            }
            
            backup(leaf, value);
        } else {
            // Directly evaluate the leaf - no worker queue
            int chosen_move = leaf->get_move_from_parent();
            if (chosen_move < 0) {
                auto valid_moves = leaf->get_state().get_valid_moves();
                if (!valid_moves.empty()) {
                    chosen_move = valid_moves[0];
                } else {
                    chosen_move = 0;
                }
            }
            
            std::vector<std::vector<int>> board2D = leaf->get_state().get_board();
            std::vector<std::vector<std::vector<int>>> board_batch = {board2D};
            std::vector<int> chosen_moves = {chosen_move};
            std::vector<int> player_batch = {leaf->get_state().current_player};
            
            auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
                board_batch, chosen_moves, player_batch);
                
            float attack = attackVec[0];
            float defense = defenseVec[0];
            
            std::vector<float> policy;
            float value = 0.0f;
            
            // Direct neural network evaluation
            nn_->request_inference(leaf->get_state(), chosen_move, attack, defense, policy, value);
            
            auto validMoves = leaf->get_state().get_valid_moves();
            std::vector<float> validPriors;
            validPriors.reserve(validMoves.size());
            
            for (int move : validMoves) {
                if (move >= 0 && move < static_cast<int>(policy.size())) {
                    validPriors.push_back(policy[move]);
                } else {
                    validPriors.push_back(1.0f / validMoves.size());
                }
            }
            
            float sum = std::accumulate(validPriors.begin(), validPriors.end(), 0.0f);
            if (sum > 0) {
                for (auto& p : validPriors) {
                    p /= sum;
                }
            }
            
            leaf->expand(validMoves, validPriors);
            backup(leaf, value);
        }
        
        // Increment the simulation counter
        simulations_done_.fetch_add(1, std::memory_order_relaxed);
        
    } catch (const std::exception& e) {
        MCTS_DEBUG("Error in simulation " << sim_index << ": " << e.what());
    }
}

void MCTS::run_thread_simulations(int start_idx, int end_idx) {
    for (int i = start_idx; i < end_idx; i++) {
        if (i >= config_.num_simulations) break;
        run_single_simulation(i);
    }
}

int MCTS::select_move() const {
    if (!root_) {
        return -1;
    }
    
    std::vector<Node*> children = root_->get_children();
    if (children.empty()) {
        return -1;
    }
    
    int bestMove = -1;
    Node* bestChild = nullptr;
    int maxVisits = -1;
    float bestValue = -2.0f;
    
    std::vector<std::pair<Node*, int>> sorted_children;
    for (Node* c : children) {
        if (!c) continue;
        sorted_children.push_back({c, c->get_visit_count()});
    }
    
    std::sort(sorted_children.begin(), sorted_children.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    if (!sorted_children.empty()) {
        Node* c = sorted_children[0].first;
        maxVisits = sorted_children[0].second;
        bestMove = c->get_move_from_parent();
        bestChild = c;
        bestValue = c->get_q_value();
    }
    
    return bestMove;
}

Node* MCTS::select_node(Node* root) const {
    if (!root) {
        return nullptr;
    }
    
    Node* current = root;
    std::vector<Node*> path;
    
    while (true) {
        path.push_back(current);
        
        if (current->get_state().is_terminal()) {
            break;
        }
        
        if (current->is_leaf()) {
            break;
        }
        
        std::vector<Node*> kids;
        try {
            kids = current->get_children();
        } catch (const std::exception& e) {
            break;
        }
        
        if (kids.empty()) {
            break;
        }
        
        float bestVal = -1e9;
        Node* bestChild = nullptr;
        
        for (Node* c : kids) {
            if (!c) continue;
            
            float sc;
            try {
                sc = uct_score(current, c);
            } catch (const std::exception& e) {
                continue;
            }
            
            if (sc > bestVal) {
                bestVal = sc;
                bestChild = c;
            }
        }
        
        if (!bestChild) {
            break;
        }
        
        current = bestChild;
    }
    
    for (Node* node : path) {
        node->add_virtual_loss();
    }
    
    return current;
}

void MCTS::expand_and_evaluate(Node* leaf) {
    Gamestate st = leaf->get_state();
    
    if (st.is_terminal()) {
        float r = 0.f;
        int winner = st.get_winner();
        if (winner == st.current_player) {
            r = 1.f;
        } else if (winner == 0) {
            r = 0.f;
        } else {
            r = -1.f;
        }
        backup(leaf, r);
        return;
    }

    int chosenMove = leaf->get_move_from_parent();
    if (chosenMove < 0) {
        std::vector<int> valid_moves = st.get_valid_moves();
        if (!valid_moves.empty()) {
            chosenMove = valid_moves[0];
        } else {
            chosenMove = 0;
        }
    }

    std::vector<std::vector<int>> board2D = st.get_board(); 
    std::vector<std::vector<std::vector<int>>> board_batch;
    board_batch.push_back(board2D);
    
    std::vector<int> chosen_moves;
    chosen_moves.push_back(chosenMove);
    
    std::vector<int> player_batch;
    player_batch.push_back(st.current_player);
    
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
        board_batch, chosen_moves, player_batch);

    float attack = attackVec[0];
    float defense = defenseVec[0];

    std::vector<float> policy;
    float value = 0.f;
    
    nn_->request_inference(st, chosenMove, attack, defense, policy, value);

    auto validMoves = st.get_valid_moves();
    std::vector<float> validPolicies;
    validPolicies.reserve(validMoves.size());
    
    for (int move : validMoves) {
        if (move >= 0 && move < static_cast<int>(policy.size())) {
            validPolicies.push_back(policy[move]);
        } else {
            validPolicies.push_back(1.0f / validMoves.size());
        }
    }
    
    float sum = std::accumulate(validPolicies.begin(), validPolicies.end(), 0.0f);
    if (sum > 0) {
        for (auto& p : validPolicies) {
            p /= sum;
        }
    }
    
    leaf->expand(validMoves, validPolicies);

    backup(leaf, value);
}

void MCTS::backup(Node* leaf, float value) {
    if (!leaf) {
        return;
    }
    
    Node* current = leaf;
    int leafPlayer = leaf->get_state().current_player;
    
    while (current) {
        int nodePlayer = current->get_state().current_player;
        float toAdd = (nodePlayer == leafPlayer) ? value : -value;
        current->update_stats(toAdd);
        current = current->get_parent();
    }
}

float MCTS::uct_score(const Node* parent, const Node* child) const {
    float Q = child->get_q_value();
    float P = child->get_prior();
    int parentVisits = parent->get_visit_count();
    int childVisits  = child->get_visit_count();
    float c = config_.c_puct;
    float U = c * P * std::sqrt((float)parentVisits + 1e-8f) / (1 + childVisits);
    return Q + U;
}
// mcts.cpp
#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

MCTS::MCTS(const MCTSConfig& config,
    std::shared_ptr<PythonNNProxy> nn,
    int boardSize)
: config_(config),
nn_(nn),
simulations_done_(0),
attackDefense_(boardSize),
rng_(std::random_device{}()) // Initialize rng_
{
// We'll create the leaf gatherer on demand in run_search
}

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

void MCTS::force_shutdown() {
    MCTS_DEBUG("FORCING EMERGENCY SHUTDOWN");
    
    // Set shutdown flag
    shutdown_flag_ = true;
    
    // Handle the leaf queue and fulfill all pending promises
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Count remaining leaves
        int remaining = leaf_queue_.size();
        MCTS_DEBUG("Processing " << remaining << " queued leaves");
        
        // Fulfill all promises in the queue with default values before clearing
        int processed = 0;
        while (!leaf_queue_.empty()) {
            try {
                auto& task = leaf_queue_.front();
                if (task.result_promise) {
                    auto valid_moves = task.state.get_valid_moves();
                    std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                    task.result_promise->set_value({default_policy, 0.0f});
                    processed++;
                }
                leaf_queue_.pop();
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error fulfilling promise during shutdown: " << e.what());
                leaf_queue_.pop();
            }
        }
        
        MCTS_DEBUG("Fulfilled " << processed << " promises during shutdown");
        
        // Signal any waiting threads
        queue_cv_.notify_all();
    }
    
    // Reset leaves in flight counter
    int in_flight = leaves_in_flight_.exchange(0, std::memory_order_acq_rel);
    if (in_flight < 0) {
        MCTS_DEBUG("WARNING: Negative leaves in flight count: " << in_flight);
    } else {
        MCTS_DEBUG("Reset " << in_flight << " leaves in flight");
    }
    
    // Detach all threads immediately instead of trying to join
    MCTS_DEBUG("Force detaching " << threads_.size() << " threads");
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.detach();
        }
    }
    threads_.clear();
    
    MCTS_DEBUG("Emergency shutdown complete");
}

void MCTS::run_search(const Gamestate& rootState) {
    MCTS_DEBUG("Starting MCTS search with semi-parallel approach");
    
    // Force cleanup of any previous search
    shutdown_flag_ = true;
    threads_.clear();
    
    // Reset search state
    root_ = std::make_unique<Node>(rootState);
    simulations_done_ = 0;
    shutdown_flag_ = false;
    leaves_in_flight_ = 0;
    
    MCTS_DEBUG("Initialized search with root state, player: " << rootState.current_player);
    
    // Initialize root with uniform priors
    std::vector<int> validMoves = rootState.get_valid_moves();
    if (!validMoves.empty()) {
        MCTS_DEBUG("Root has " << validMoves.size() << " valid moves");
        
        // Use uniform priors
        std::vector<float> uniformPriors(validMoves.size(), 1.0f / validMoves.size());
        
        // Add Dirichlet noise for exploration
        add_dirichlet_noise(uniformPriors);
        
        // Expand root with priors
        root_->expand(validMoves, uniformPriors);
        
        // Create a new leaf gatherer for this search
        int batch_size = std::max(1, config_.parallel_leaf_batch_size);
        leaf_gatherer_ = std::make_unique<LeafGatherer>(nn_, attackDefense_, batch_size);
        
        // Run semi-parallel search
        run_semi_parallel_search(config_.num_simulations);
        
        // Clean up leaf gatherer
        leaf_gatherer_.reset();
    }
    
    MCTS_DEBUG("MCTS search completed with " << simulations_done_.load() << " simulations");
}

void MCTS::run_semi_parallel_search(int num_simulations) {
    MCTS_DEBUG("Running semi-parallel search for " << num_simulations << " simulations");
    
    // Define max search time 
    const auto start_time = std::chrono::steady_clock::now();
    const int MAX_SEARCH_TIME_MS = 10000; // 10 seconds max
    
    // Track in-flight futures
    struct PendingEval {
        Node* leaf;
        std::future<std::pair<std::vector<float>, float>> future;
    };
    
    std::vector<PendingEval> pending_evals;
    
    int simulations_completed = 0;
    int max_pending = config_.num_threads * 2; // Allow some parallelism
    
    // For batch processing
    std::vector<std::tuple<std::string, int, float, float>> batch_inputs;
    std::vector<Node*> batch_leaves;
    const int MAX_BATCH_SIZE = config_.parallel_leaf_batch_size;
    
    while (simulations_completed < num_simulations) {
        // Check for timeout
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();
            
        if (elapsed_ms > MAX_SEARCH_TIME_MS) {
            MCTS_DEBUG("Search timeout reached after " << elapsed_ms << "ms");
            break;
        }
        
        // Process any completed futures first
        for (auto it = pending_evals.begin(); it != pending_evals.end();) {
            auto status = it->future.wait_for(std::chrono::milliseconds(0));
            
            if (status == std::future_status::ready) {
                try {
                    // Get the result
                    auto [policy, value] = it->future.get();
                    Node* leaf = it->leaf;
                    
                    if (leaf) {
                        // Expand with policy
                        auto valid_moves = leaf->get_state().get_valid_moves();
                        
                        if (!valid_moves.empty() && valid_moves.size() == policy.size()) {
                            leaf->expand(valid_moves, policy);
                            backup(leaf, value);
                            simulations_completed++;
                        } else {
                            // Remove virtual losses if we can't expand
                            Node* current = leaf;
                            while (current) {
                                current->remove_virtual_loss();
                                current = current->get_parent();
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error processing future: " << e.what());
                }
                
                // Remove from pending list
                it = pending_evals.erase(it);
            } else {
                ++it;
            }
        }
        
        // Process batch if needed (either max size reached or no more pending slots)
        if ((batch_inputs.size() >= MAX_BATCH_SIZE) || 
            (batch_inputs.size() > 0 && pending_evals.size() >= max_pending)) {
            
            MCTS_DEBUG("Processing batch of " << batch_inputs.size() << " leaves");
            
            // Request batch inference
            std::vector<NNOutput> results = nn_->batch_inference(batch_inputs);
            
            // Process results
            for (size_t i = 0; i < batch_leaves.size() && i < results.size(); i++) {
                Node* leaf = batch_leaves[i];
                
                try {
                    auto valid_moves = leaf->get_state().get_valid_moves();
                    
                    // Only expand if we have valid moves and the policy is the right size
                    if (!valid_moves.empty() && valid_moves.size() == results[i].policy.size()) {
                        leaf->expand(valid_moves, results[i].policy);
                        backup(leaf, results[i].value);
                        simulations_completed++;
                    } else if (!valid_moves.empty()) {
                        // Create uniform policy if sizes don't match
                        std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        leaf->expand(valid_moves, uniform_policy);
                        backup(leaf, results[i].value);
                        simulations_completed++;
                    } else {
                        // Remove virtual losses if we can't expand
                        Node* current = leaf;
                        while (current) {
                            current->remove_virtual_loss();
                            current = current->get_parent();
                        }
                    }
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error processing batch result: " << e.what());
                    
                    // Remove virtual losses
                    Node* current = leaf;
                    while (current) {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    }
                }
            }
            
            // Clear batch
            batch_inputs.clear();
            batch_leaves.clear();
        }
        
        // See if we can select more leaves
        while ((batch_inputs.size() < MAX_BATCH_SIZE) && 
               (pending_evals.size() < max_pending) && 
               (simulations_completed + pending_evals.size() + batch_inputs.size() < num_simulations)) {
            
            // Select a leaf node
            Node* leaf = select_node(root_.get());
            if (!leaf) {
                MCTS_DEBUG("Failed to select leaf, breaking");
                break;
            }
            
            // Handle terminal nodes immediately
            if (leaf->get_state().is_terminal()) {
                float value = 0.0f;
                int winner = leaf->get_state().get_winner();
                int current_player = leaf->get_state().current_player;
                
                if (winner == current_player) {
                    value = 1.0f;
                } else if (winner == 0) {
                    value = 0.0f;  // Draw
                } else {
                    value = -1.0f;  // Loss
                }
                
                backup(leaf, value);
                simulations_completed++;
                continue;
            }
            
            // Add to batch
            try {
                const Gamestate& state = leaf->get_state();
                int chosen_move = leaf->get_move_from_parent();
                
                if (chosen_move < 0) {
                    auto valid_moves = state.get_valid_moves();
                    if (!valid_moves.empty()) {
                        chosen_move = valid_moves[0];
                    } else {
                        chosen_move = 0;
                    }
                }
                
                // Calculate attack/defense
                std::vector<std::vector<std::vector<int>>> board_batch = {state.get_board()};
                std::vector<int> chosen_moves = {chosen_move};
                std::vector<int> player_batch = {state.current_player};
                
                auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
                    board_batch, chosen_moves, player_batch);
                
                float attack = attackVec.empty() ? 0.0f : attackVec[0];
                float defense = defenseVec.empty() ? 0.0f : defenseVec[0];
                
                // Create state string
                std::string state_str = nn_->create_state_string(state, chosen_move, attack, defense);
                
                // Add to batch
                batch_inputs.emplace_back(state_str, chosen_move, attack, defense);
                batch_leaves.push_back(leaf);
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error preparing leaf for batch: " << e.what());
                
                // Remove virtual losses
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
            }
        }
        
        // If we have nothing to do, sleep briefly
        if (pending_evals.empty() && batch_inputs.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Log progress periodically
        if (simulations_completed % 100 == 0 && simulations_completed > 0) {
            MCTS_DEBUG("Completed " << simulations_completed << "/" << num_simulations 
                      << " simulations, " << pending_evals.size() << " pending, "
                      << batch_inputs.size() << " in batch");
        }
    }
    
    // Process any remaining batch
    if (!batch_inputs.empty()) {
        MCTS_DEBUG("Processing remaining batch of " << batch_inputs.size() << " leaves");
        
        // Request batch inference
        std::vector<NNOutput> results = nn_->batch_inference(batch_inputs);
        
        // Process results
        for (size_t i = 0; i < batch_leaves.size() && i < results.size(); i++) {
            Node* leaf = batch_leaves[i];
            
            try {
                auto valid_moves = leaf->get_state().get_valid_moves();
                
                // Only expand if we have valid moves and the policy is the right size
                if (!valid_moves.empty() && valid_moves.size() == results[i].policy.size()) {
                    leaf->expand(valid_moves, results[i].policy);
                    backup(leaf, results[i].value);
                    simulations_completed++;
                } else if (!valid_moves.empty()) {
                    // Create uniform policy if sizes don't match
                    std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                    leaf->expand(valid_moves, uniform_policy);
                    backup(leaf, results[i].value);
                    simulations_completed++;
                } else {
                    // Remove virtual losses if we can't expand
                    Node* current = leaf;
                    while (current) {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    }
                }
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error processing batch result: " << e.what());
                
                // Remove virtual losses
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
            }
        }
    }
    
    // Set final count
    simulations_done_ = simulations_completed;
    
    // Wait for any remaining evaluations to complete (with timeout)
    auto wait_start = std::chrono::steady_clock::now();
    const int MAX_WAIT_MS = 1000; // 1 second timeout for cleanup
    
    while (!pending_evals.empty()) {
        auto now = std::chrono::steady_clock::now();
        auto wait_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - wait_start).count();
            
        if (wait_elapsed > MAX_WAIT_MS) {
            MCTS_DEBUG("Timeout waiting for remaining evaluations, abandoning " 
                      << pending_evals.size() << " pending evaluations");
            break;
        }
        
        // Try to process completed futures
        for (auto it = pending_evals.begin(); it != pending_evals.end();) {
            auto status = it->future.wait_for(std::chrono::milliseconds(0));
            
            if (status == std::future_status::ready) {
                try {
                    it->future.get(); // Just to clean up
                } catch (...) {
                    // Ignore errors
                }
                
                it = pending_evals.erase(it);
            } else {
                ++it;
            }
        }
        
        // Brief sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    MCTS_DEBUG("Semi-parallel search completed with " << simulations_completed << " simulations");
}

void MCTS::run_simple_search(int num_simulations) {
    MCTS_DEBUG("Running single-threaded search with neural network batch processing for " << num_simulations << " simulations");
    
    // Define max search time 
    const auto start_time = std::chrono::steady_clock::now();
    const int MAX_SEARCH_TIME_MS = 10000; // 10 seconds max
    
    // Define batch size
    const int BATCH_SIZE = std::min(16, config_.parallel_leaf_batch_size);
    MCTS_DEBUG("Using batch size: " << BATCH_SIZE);
    
    // We'll collect leaf nodes for batch processing
    struct BatchItem {
        Node* leaf;
        const Gamestate* state;
        int chosen_move;
    };
    
    int simulations_completed = 0;
    
    while (simulations_completed < num_simulations) {
        // Check for timeout
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();
            
        if (elapsed_ms > MAX_SEARCH_TIME_MS) {
            MCTS_DEBUG("Search timeout reached after " << elapsed_ms << "ms");
            break;
        }
        
        // Collect a batch of leaf nodes
        std::vector<BatchItem> batch;
        std::vector<Node*> terminal_nodes; // Handle separately
        
        // Fill the batch
        while (batch.size() < BATCH_SIZE && 
               batch.size() + terminal_nodes.size() + simulations_completed < num_simulations) {
            
            // Select a leaf node
            Node* leaf = select_node(root_.get());
            if (!leaf) {
                MCTS_DEBUG("Failed to select leaf, skipping");
                continue;
            }
            
            // Handle terminal nodes immediately
            if (leaf->get_state().is_terminal()) {
                terminal_nodes.push_back(leaf);
                continue;
            }
            
            // Add non-terminal nodes to the batch
            BatchItem item;
            item.leaf = leaf;
            item.state = &(leaf->get_state());
            item.chosen_move = leaf->get_move_from_parent();
            
            // Fix chosen_move if invalid
            if (item.chosen_move < 0) {
                auto valid_moves = item.state->get_valid_moves();
                if (!valid_moves.empty()) {
                    item.chosen_move = valid_moves[0];
                } else {
                    item.chosen_move = 0;
                }
            }
            
            batch.push_back(item);
            
            // Stop collecting if we have enough nodes
            if (batch.size() >= BATCH_SIZE) {
                break;
            }
        }
        
        // Process terminal nodes
        for (Node* leaf : terminal_nodes) {
            float value = 0.0f;
            int winner = leaf->get_state().get_winner();
            int current_player = leaf->get_state().current_player;
            
            if (winner == current_player) {
                value = 1.0f;
            } else if (winner == 0) {
                value = 0.0f;  // Draw
            } else {
                value = -1.0f;  // Loss
            }
            
            backup(leaf, value);
            simulations_completed++;
        }
        
        // Skip batch processing if no non-terminal nodes
        if (batch.empty()) {
            if (terminal_nodes.empty()) {
                // No nodes at all, probably at end of search
                break;
            }
            continue;
        }
        
        // Prepare batch data
        std::vector<std::vector<std::vector<int>>> board_batch;
        std::vector<int> chosen_moves;
        std::vector<int> player_batch;
        
        for (const auto& item : batch) {
            board_batch.push_back(item.state->get_board());
            chosen_moves.push_back(item.chosen_move);
            player_batch.push_back(item.state->current_player);
        }
        
        // Calculate attack/defense bonuses
        std::vector<float> attack_vec;
        std::vector<float> defense_vec;
        
        try {
            auto [a_vec, d_vec] = attackDefense_.compute_bonuses(
                board_batch, chosen_moves, player_batch);
            attack_vec = a_vec;
            defense_vec = d_vec;
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error computing attack/defense bonuses: " << e.what());
            // Create default values
            attack_vec.resize(batch.size(), 0.0f);
            defense_vec.resize(batch.size(), 0.0f);
        }
        
        // Prepare neural network input
        std::vector<std::tuple<std::string, int, float, float>> nn_inputs;
        
        for (size_t i = 0; i < batch.size(); i++) {
            float attack = (i < attack_vec.size()) ? attack_vec[i] : 0.0f;
            float defense = (i < defense_vec.size()) ? defense_vec[i] : 0.0f;
            
            std::string state_str = nn_->create_state_string(
                *(batch[i].state), chosen_moves[i], attack, defense);
                
            nn_inputs.emplace_back(state_str, chosen_moves[i], attack, defense);
        }
        
        // Call neural network for batch inference
        std::vector<NNOutput> results;
        
        try {
            auto nn_start = std::chrono::steady_clock::now();
            
            results = nn_->batch_inference(nn_inputs);
            
            auto nn_end = std::chrono::steady_clock::now();
            auto nn_duration = std::chrono::duration_cast<std::chrono::milliseconds>(nn_end - nn_start).count();
            
            MCTS_DEBUG("Neural network batch inference completed in " << nn_duration 
                       << "ms for " << batch.size() << " leaves");
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error in neural network batch inference: " << e.what());
            results.clear();
        }
        
        // Process results
        for (size_t i = 0; i < batch.size(); i++) {
            Node* leaf = batch[i].leaf;
            auto valid_moves = leaf->get_state().get_valid_moves();
            
            if (valid_moves.empty()) {
                // Remove virtual losses if we can't expand
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
                continue;
            }
            
            std::vector<float> valid_policy;
            float value = 0.0f;
            
            if (i < results.size()) {
                // Extract policy for valid moves
                const auto& policy = results[i].policy;
                valid_policy.reserve(valid_moves.size());
                
                for (int move : valid_moves) {
                    if (move >= 0 && move < static_cast<int>(policy.size())) {
                        valid_policy.push_back(policy[move]);
                    } else {
                        valid_policy.push_back(1.0f / valid_moves.size());
                    }
                }
                
                // Normalize the policy
                float sum = std::accumulate(valid_policy.begin(), valid_policy.end(), 0.0f);
                if (sum > 0) {
                    for (auto& p : valid_policy) {
                        p /= sum;
                    }
                } else {
                    // Uniform policy if sum is zero
                    for (auto& p : valid_policy) {
                        p = 1.0f / valid_policy.size();
                    }
                }
                
                // Get value
                value = results[i].value;
            } else {
                // Use uniform policy if neural network failed
                valid_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                value = 0.0f;
            }
            
            // Expand the node
            leaf->expand(valid_moves, valid_policy);
            
            // Backup the value
            backup(leaf, value);
            
            // Count this simulation
            simulations_completed++;
        }
        
        if (simulations_completed % 100 == 0 && simulations_completed > 0) {
            MCTS_DEBUG("Completed " << simulations_completed << "/" << num_simulations << " simulations");
        }
    }
    
    simulations_done_ = simulations_completed;
    MCTS_DEBUG("Search completed with " << simulations_completed << " simulations");
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

std::future<std::pair<std::vector<float>, float>> MCTS::queue_leaf_for_evaluation(Node* leaf) {
    // Check for shutdown first to avoid queuing when we're shutting down
    if (shutdown_flag_) {
        MCTS_DEBUG("Shutdown flag detected, skipping leaf evaluation");
        // Create promise with shared_ptr
        auto promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
        
        // Create uniform policy for the default value
        std::vector<float> default_policy;
        if (leaf && !leaf->get_state().is_terminal()) {
            auto valid_moves = leaf->get_state().get_valid_moves();
            default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
        }
        
        // Set default value and return
        promise->set_value({default_policy, 0.0f});
        return promise->get_future();
    }
    
    MCTS_DEBUG("Queueing leaf node for evaluation");
    
    // Create promise using shared_ptr
    auto promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
    auto future = promise->get_future();
    
    // Handle null leaf with default values
    if (!leaf) {
        MCTS_DEBUG("Null leaf provided, returning default values");
        std::vector<float> default_policy;
        promise->set_value({default_policy, 0.0f});
        return future;
    }
    
    // Check if leaf is terminal - handle directly without queueing
    if (leaf->get_state().is_terminal()) {
        MCTS_DEBUG("Terminal leaf node, calculating value directly");
        float value = 0.0f;
        int winner = leaf->get_state().get_winner();
        int current_player = leaf->get_state().current_player;
        
        if (winner == current_player) {
            value = 1.0f;
        } else if (winner == 0) {
            value = 0.0f;  // Draw
        } else {
            value = -1.0f;  // Loss
        }
        
        // Return an empty policy for terminal nodes
        std::vector<float> empty_policy;
        promise->set_value({empty_policy, value});
        return future;
    }
    
    // Check leaves in flight to avoid excessive queuing
    int current_leaves = leaves_in_flight_.load(std::memory_order_acquire);
    if (current_leaves > 1000) {
        MCTS_DEBUG("WARNING: Too many leaves in flight (" << current_leaves << "), using default values");
        
        // Create uniform policy for the default value
        auto valid_moves = leaf->get_state().get_valid_moves();
        std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
        
        // Set default value and return
        promise->set_value({default_policy, 0.0f});
        return future;
    }
    
    // Add the leaf to the evaluation queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Check shutdown flag again with the lock held
        if (shutdown_flag_) {
            MCTS_DEBUG("Shutdown detected during leaf queuing, using default values");
            
            // Create uniform policy for the default value
            auto valid_moves = leaf->get_state().get_valid_moves();
            std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
            
            // Set default value and return
            promise->set_value({default_policy, 0.0f});
            return future;
        }
        
        MCTS_DEBUG("Adding leaf to queue (current size: " << leaf_queue_.size() << ")");
        
        // Create task with the leaf's state
        LeafTask task;
        task.leaf = leaf;
        task.state = leaf->get_state().copy();  // Deep copy to avoid race conditions
        task.chosen_move = leaf->get_move_from_parent();
        task.result_promise = std::move(promise);  // Move the shared_ptr
        
        // Check queue size before adding
        if (leaf_queue_.size() >= 1000) {
            MCTS_DEBUG("WARNING: Queue size (" << leaf_queue_.size() << ") too large, flushing");
            
            // Fulfill all promises in the queue with default values before clearing
            while (!leaf_queue_.empty()) {
                try {
                    auto& task = leaf_queue_.front();
                    if (task.result_promise) {
                        auto valid_moves = task.state.get_valid_moves();
                        std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        task.result_promise->set_value({default_policy, 0.0f});
                    }
                    leaf_queue_.pop();
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error fulfilling promise during queue flush: " << e.what());
                    leaf_queue_.pop();
                }
            }
        }
        
        leaf_queue_.push(std::move(task));
        
        // Notify the evaluation thread
        queue_cv_.notify_one();
    }
    
    // Increment counter for leaves in flight - use relaxed ordering as we just care about rough count
    leaves_in_flight_.fetch_add(1, std::memory_order_relaxed);
    MCTS_DEBUG("Leaf queued, current leaves in flight: " << leaves_in_flight_.load());
    
    return future;
}

void MCTS::leaf_evaluation_thread() {
    MCTS_DEBUG("Leaf evaluation thread started");
    
    // Determine batch size - use config if provided, otherwise default to 8
    const int batch_size = config_.parallel_leaf_batch_size > 0 ? 
                          config_.parallel_leaf_batch_size : 8;
    
    MCTS_DEBUG("Using batch size: " << batch_size);
    
    // Track last activity time for stall detection
    auto last_activity_time = std::chrono::steady_clock::now();
    
    while (!shutdown_flag_) {
        // Check termination conditions first - this is crucial
        // If we've reached simulation limit, exit even if queue isn't empty
        if (simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations) {
            MCTS_DEBUG("Simulation limit reached in leaf evaluation thread, exiting");
            break;
        }
        
        // Collect a batch of leaves to evaluate
        std::vector<LeafTask> current_batch;
        current_batch.reserve(batch_size);
        
        // Critical section: get leaves from the queue
        bool got_tasks = false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for leaves or shutdown signal with a short timeout
            auto wait_status = queue_cv_.wait_for(lock, std::chrono::milliseconds(10),
                [this] { 
                    // Check both shutdown flag AND simulation count
                    return !leaf_queue_.empty() || shutdown_flag_ || 
                        simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations; 
                });
            
            // Check termination conditions after wait
            if (shutdown_flag_ || simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations) {
                MCTS_DEBUG("Termination condition detected in leaf evaluation thread after wait");
                break;
            }
            
            // Get leaves up to batch size if available
            int count = 0;
            while (!leaf_queue_.empty() && count < batch_size) {
                current_batch.push_back(std::move(leaf_queue_.front()));
                leaf_queue_.pop();
                count++;
                got_tasks = true;
            }
        }
        
        // Update activity timestamp if we got tasks
        if (got_tasks) {
            last_activity_time = std::chrono::steady_clock::now();
        }
        
        // Check for thread starvation (no activity for too long)
        auto current_time = std::chrono::steady_clock::now();
        auto inactivity_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_activity_time).count();
            
        // If no activity for 5 seconds and search isn't complete, report potential stall
        if (inactivity_duration > 5000 && 
            simulations_done_.load(std::memory_order_acquire) < config_.num_simulations) {
            
            MCTS_DEBUG("WARNING: Leaf evaluation thread inactive for " << inactivity_duration 
                      << "ms - possible stall detected");
            // Reset the timer to avoid spamming logs
            last_activity_time = current_time;
        }
        
        // If no leaves and not shutting down, just loop again with short sleep
        if (current_batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Process the batch
        MCTS_DEBUG("Processing batch of " << current_batch.size() << " leaves using neural network");
        
        try {
            // Prepare data for attack/defense module and neural network
            std::vector<std::vector<std::vector<int>>> board_batch;
            std::vector<int> chosen_moves;
            std::vector<int> player_batch;
            
            for (const auto& task : current_batch) {
                board_batch.push_back(task.state.get_board());
                chosen_moves.push_back(task.chosen_move);
                player_batch.push_back(task.state.current_player);
            }
            
            // Compute attack/defense bonuses
            MCTS_DEBUG("Computing attack/defense bonuses for batch");
            auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
                board_batch, chosen_moves, player_batch);
            
            // Prepare neural network inputs
            std::vector<std::tuple<std::string, int, float, float>> nn_inputs;
            for (size_t i = 0; i < current_batch.size(); i++) {
                std::string stateStr = nn_->create_state_string(
                    current_batch[i].state, 
                    chosen_moves[i],
                    attackVec[i], 
                    defenseVec[i]);
                
                nn_inputs.emplace_back(stateStr, chosen_moves[i], attackVec[i], defenseVec[i]);
            }
            
            // Call neural network for batch inference
            MCTS_DEBUG("Calling neural network for batch inference");
            std::vector<NNOutput> results;
            bool success = false;
            
            try {
                auto start_time = std::chrono::steady_clock::now();
                
                // Use BatchingNNInterface for batch inference
                results = nn_->batch_inference(nn_inputs);
                
                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                
                MCTS_DEBUG("Neural network batch inference completed in " << duration << "ms");
                success = !results.empty();
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error in neural network batch inference: " << e.what());
                success = false;
            }
            
            // Process results and fulfill promises
            MCTS_DEBUG("Processing neural network results");
            for (size_t i = 0; i < current_batch.size(); i++) {
                try {
                    // Check termination condition again for responsiveness
                    if (shutdown_flag_ || simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations) {
                        MCTS_DEBUG("Termination condition detected during result processing");
                        // Just use defaults for remaining items
                        for (size_t j = i; j < current_batch.size(); j++) {
                            auto valid_moves = current_batch[j].state.get_valid_moves();
                            std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                            // Using -> operator for shared_ptr
                            if (current_batch[j].result_promise) {
                                current_batch[j].result_promise->set_value({default_policy, 0.0f});
                            }
                            leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
                        }
                        break;
                    }
                    
                    if (success && i < results.size()) {
                        // Get valid moves for this state
                        auto valid_moves = current_batch[i].state.get_valid_moves();
                        
                        // Extract policy for valid moves
                        std::vector<float> valid_policy;
                        valid_policy.reserve(valid_moves.size());
                        
                        for (int move : valid_moves) {
                            if (move >= 0 && move < static_cast<int>(results[i].policy.size())) {
                                valid_policy.push_back(results[i].policy[move]);
                            } else {
                                valid_policy.push_back(1.0f / valid_moves.size());
                            }
                        }
                        
                        // Normalize policy
                        float sum = std::accumulate(valid_policy.begin(), valid_policy.end(), 0.0f);
                        if (sum > 0) {
                            for (auto& p : valid_policy) {
                                p /= sum;
                            }
                        } else {
                            // Uniform policy if sum is zero
                            for (auto& p : valid_policy) {
                                p = 1.0f / valid_policy.size();
                            }
                        }
                        
                        // Fulfill promise with policy and value - using -> operator for shared_ptr
                        if (current_batch[i].result_promise) {
                            current_batch[i].result_promise->set_value({valid_policy, results[i].value});
                        }
                    } else {
                        // Use default values on error
                        auto valid_moves = current_batch[i].state.get_valid_moves();
                        std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        
                        // Using -> operator for shared_ptr
                        if (current_batch[i].result_promise) {
                            current_batch[i].result_promise->set_value({default_policy, 0.0f});
                        }
                    }
                }
                catch (const std::exception& e) {
                    MCTS_DEBUG("Error processing neural network result for leaf " << i << ": " << e.what());
                    try {
                        // Ensure promise is fulfilled even on error
                        auto valid_moves = current_batch[i].state.get_valid_moves();
                        std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        
                        // Using -> operator for shared_ptr
                        if (current_batch[i].result_promise) {
                            current_batch[i].result_promise->set_value({default_policy, 0.0f});
                        }
                    }
                    catch (...) {
                        // Promise might already be fulfilled
                    }
                }
                
                // Decrement counter for leaves in flight
                leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error processing batch: " << e.what());
            
            // Handle each leaf with default values on error
            for (auto& task : current_batch) {
                try {
                    auto valid_moves = task.state.get_valid_moves();
                    std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                    
                    // Using -> operator for shared_ptr
                    if (task.result_promise) {
                        task.result_promise->set_value({default_policy, 0.0f});
                    }
                    leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
                }
                catch (...) {
                    // Promise might already be fulfilled
                }
            }
        }
    }
    
    // Handle any remaining leaves in the queue with default values
    MCTS_DEBUG("Leaf evaluation thread shutting down, handling remaining tasks");
    std::vector<LeafTask> remaining_tasks;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!leaf_queue_.empty()) {
            remaining_tasks.push_back(std::move(leaf_queue_.front()));
            leaf_queue_.pop();
        }
    }
    
    MCTS_DEBUG("Processing " << remaining_tasks.size() << " remaining tasks");
    for (auto& task : remaining_tasks) {
        try {
            auto valid_moves = task.state.get_valid_moves();
            std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
            
            // Using -> operator for shared_ptr
            if (task.result_promise) {
                task.result_promise->set_value({default_policy, 0.0f});
            }
            leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
        } catch (...) {
            // Promise might already be fulfilled
        }
    }
    
    MCTS_DEBUG("Leaf evaluation thread exiting");
}

void MCTS::process_leaf_batch(std::vector<LeafTask>& batch) {
    if (batch.empty()) {
        MCTS_DEBUG("Empty batch provided to process_leaf_batch, ignoring");
        return;
    }
    
    MCTS_DEBUG("Processing batch of " << batch.size() << " leaf nodes");
    
    // Prepare data for attack/defense module
    std::vector<std::vector<std::vector<int>>> board_batch;
    std::vector<int> chosen_moves;
    std::vector<int> player_batch;
    
    // Collect batch data
    for (const auto& task : batch) {
        board_batch.push_back(task.state.get_board());
        
        // Use a valid move if chosen_move is invalid
        int move = task.chosen_move;
        if (move < 0) {
            auto valid_moves = task.state.get_valid_moves();
            if (!valid_moves.empty()) {
                move = valid_moves[0];
            } else {
                move = 0; // Fallback
            }
        }
        chosen_moves.push_back(move);
        player_batch.push_back(task.state.current_player);
    }
    
    MCTS_DEBUG("Computing attack/defense bonuses for batch");
    // Compute attack/defense bonuses
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    MCTS_DEBUG("Preparing neural network inputs");
    // Prepare neural network inputs
    std::vector<std::tuple<std::string, int, float, float>> nn_inputs;
    for (size_t i = 0; i < batch.size(); i++) {
        // Create state string using the neural network interface
        std::string stateStr = nn_->create_state_string(batch[i].state, 
                                                      chosen_moves[i],
                                                      attackVec[i], 
                                                      defenseVec[i]);
        
        nn_inputs.emplace_back(stateStr, chosen_moves[i], attackVec[i], defenseVec[i]);
    }
    
    MCTS_DEBUG("Calling neural network for batch inference");
    // Call neural network with batch
    std::vector<NNOutput> results;
    bool success = false;
    
    try {
        // Request batch inference from the neural network
        auto start_time = std::chrono::steady_clock::now();
        
        // Process individually if neural network doesn't support batching
        for (size_t i = 0; i < nn_inputs.size(); i++) {
            NNOutput output;
            std::vector<float> policy;
            float value = 0.0f;
            
            const auto& [stateStr, move, attack, defense] = nn_inputs[i];
            nn_->request_inference(batch[i].state, move, attack, defense, policy, value);
            
            output.policy = std::move(policy);
            output.value = value;
            results.push_back(std::move(output));
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        MCTS_DEBUG("Neural network inference completed in " << duration << "ms");
        
        success = !results.empty();
    } catch (const std::exception& e) {
        MCTS_DEBUG("Error during neural network inference: " << e.what());
        success = false;
    }
    
    MCTS_DEBUG("Processing neural network results and fulfilling promises");
    // Process results and fulfill promises
    for (size_t i = 0; i < batch.size(); i++) {
        try {
            if (success && i < results.size()) {
                // Get policy and normalize for valid moves
                auto valid_moves = batch[i].state.get_valid_moves();
                std::vector<float> valid_policy;
                valid_policy.reserve(valid_moves.size());
                
                for (int move : valid_moves) {
                    if (move >= 0 && move < static_cast<int>(results[i].policy.size())) {
                        valid_policy.push_back(results[i].policy[move]);
                    } else {
                        valid_policy.push_back(1.0f / valid_moves.size());
                    }
                }
                
                // Normalize policy
                float sum = std::accumulate(valid_policy.begin(), valid_policy.end(), 0.0f);
                if (sum > 0) {
                    for (auto& p : valid_policy) {
                        p /= sum;
                    }
                } else {
                    // Uniform policy if sum is zero
                    for (auto& p : valid_policy) {
                        p = 1.0f / valid_policy.size();
                    }
                }
                
                MCTS_DEBUG("Setting result for leaf " << i << " with " << valid_policy.size() 
                          << " policy values and value " << results[i].value);
                
                // Set result in promise
                batch[i].result_promise->set_value({valid_policy, results[i].value});
            } else {
                // Use default values on error
                MCTS_DEBUG("Using default values for leaf " << i << " due to neural network error");
                std::vector<float> default_policy(batch[i].state.get_valid_moves().size(), 
                                               1.0f / batch[i].state.get_valid_moves().size());
                batch[i].result_promise->set_value({default_policy, 0.0f});
            }
        } catch (const std::exception& e) {
            // Ensure promise is always fulfilled even on error
            MCTS_DEBUG("Error setting promise value: " << e.what());
            try {
                std::vector<float> default_policy(batch[i].state.get_valid_moves().size(), 
                                               1.0f / batch[i].state.get_valid_moves().size());
                batch[i].result_promise->set_value({default_policy, 0.0f});
            } catch (...) {
                // Promise might already be satisfied
                MCTS_DEBUG("Failed to set default value in promise");
            }
        }
        
        // Decrement counter for leaves in flight
        leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    MCTS_DEBUG("Batch processing completed, remaining leaves in flight: " << leaves_in_flight_.load());
}

void MCTS::search_worker_thread() {
    MCTS_DEBUG("Search worker thread started");
    
    while (!shutdown_flag_) {
        // Check shutdown flag at the start of each iteration
        if (shutdown_flag_) {
            MCTS_DEBUG("Shutdown detected in search worker thread");
            break;
        }
        
        // Check if we've reached the simulation limit
        int current_simulations = simulations_done_.load(std::memory_order_acquire);
        if (current_simulations >= config_.num_simulations) {
            MCTS_DEBUG("Simulation limit reached in search worker");
            break;
        }
        
        try {
            // 1. Select a leaf node
            Node* leaf = nullptr;
            
            // Guard against null root
            if (root_.get() == nullptr) {
                MCTS_DEBUG("Root node is null, exiting search worker");
                break;
            }
            
            try {
                leaf = select_node(root_.get());
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error selecting leaf node: " << e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            if (!leaf) {
                MCTS_DEBUG("Failed to select leaf, sleeping briefly");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Check shutdown flag after selection
            if (shutdown_flag_) {
                MCTS_DEBUG("Shutdown detected after leaf selection");
                
                // Remove virtual losses
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
                
                break;
            }
            
            // 2. For terminal nodes, process immediately
            if (leaf->get_state().is_terminal()) {
                float value = 0.0f;
                int winner = leaf->get_state().get_winner();
                int current_player = leaf->get_state().current_player;
                
                if (winner == current_player) {
                    value = 1.0f;
                } else if (winner == 0) {
                    value = 0.0f; // Draw
                } else {
                    value = -1.0f; // Loss
                }
                
                // Backup value and increment simulation counter
                backup(leaf, value);
                simulations_done_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            
            // 3. Queue the leaf for neural network evaluation
            try {
                // Get valid moves before queuing
                auto valid_moves = leaf->get_state().get_valid_moves();
                
                // Handle empty valid moves edge case
                if (valid_moves.empty()) {
                    MCTS_DEBUG("Leaf has no valid moves, skipping evaluation");
                    
                    // Remove virtual losses
                    Node* current = leaf;
                    while (current) {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    }
                    
                    continue;
                }
                
                // Queue the leaf node for evaluation and get a future for the result
                std::future<std::pair<std::vector<float>, float>> future;
                
                try {
                    future = queue_leaf_for_evaluation(leaf);
                } 
                catch (const std::exception& e) {
                    MCTS_DEBUG("Error queueing leaf for evaluation: " << e.what());
                    
                    // Use uniform policy as fallback
                    std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                    leaf->expand(valid_moves, uniform_policy);
                    backup(leaf, 0.0f);
                    simulations_done_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                
                // Wait for the result with timeout
                auto status = future.wait_for(std::chrono::milliseconds(1000)); // 1 second timeout
                
                if (status == std::future_status::ready) {
                    // Get the policy and value from the future
                    std::pair<std::vector<float>, float> result;
                    
                    try {
                        result = future.get();
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("Error getting future result: " << e.what());
                        // Use uniform policy on error
                        std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        leaf->expand(valid_moves, uniform_policy);
                        backup(leaf, 0.0f);
                        simulations_done_.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    
                    auto [policy, value] = result;
                    
                    // Check if we have a valid policy
                    if (!policy.empty() && valid_moves.size() == policy.size()) {
                        // Expand with the policy from neural network
                        leaf->expand(valid_moves, policy);
                        
                        // Backup the value from neural network
                        backup(leaf, value);
                        
                        // Count this as a completed simulation
                        simulations_done_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        // Fallback to uniform policy if the sizes don't match
                        MCTS_DEBUG("Policy size mismatch: expected " << valid_moves.size() 
                                  << ", got " << policy.size());
                        std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        leaf->expand(valid_moves, uniform_policy);
                        backup(leaf, value);
                        simulations_done_.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    // Timeout occurred, use uniform policy as fallback
                    MCTS_DEBUG("Neural network evaluation timeout");
                    std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                    leaf->expand(valid_moves, uniform_policy);
                    backup(leaf, 0.0f);
                    simulations_done_.fetch_add(1, std::memory_order_relaxed);
                }
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error in neural network evaluation: " << e.what());
                
                try {
                    // Fallback to uniform policy on error
                    auto valid_moves = leaf->get_state().get_valid_moves();
                    if (!valid_moves.empty()) {
                        std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        leaf->expand(valid_moves, uniform_policy);
                        backup(leaf, 0.0f);
                        simulations_done_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        // Remove virtual losses if we can't expand
                        Node* current = leaf;
                        while (current) {
                            current->remove_virtual_loss();
                            current = current->get_parent();
                        }
                    }
                }
                catch (const std::exception& inner_e) {
                    MCTS_DEBUG("Error in fallback handling: " << inner_e.what());
                    // Just try to remove virtual losses
                    try {
                        Node* current = leaf;
                        while (current) {
                            current->remove_virtual_loss();
                            current = current->get_parent();
                        }
                    } catch (...) {}
                }
            }
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error in search thread: " << e.what());
            // Just continue with next iteration
        }
        
        // Brief sleep to prevent CPU hogging
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    MCTS_DEBUG("Search worker thread exiting");
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
        MCTS_DEBUG("select_move called with null root");
        return -1;
    }
    
    try {
        std::vector<Node*> children = root_->get_children();
        if (children.empty()) {
            MCTS_DEBUG("Root has no children");
            return -1;
        }
        
        // Create a list of (child, visit_count) pairs
        std::vector<std::pair<Node*, int>> visited_children;
        visited_children.reserve(children.size());
        
        MCTS_DEBUG("Selecting best move from " << children.size() << " children");
        
        // Collect all children with their visit counts
        for (Node* c : children) {
            if (!c) continue;
            
            // Safe get_visit_count
            int visit_count = 0;
            try {
                visit_count = c->get_visit_count();
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error getting visit count: " << e.what());
                continue;
            }
            
            // Safe get_move_from_parent
            int move = -1;
            try {
                move = c->get_move_from_parent();
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error getting move: " << e.what());
                continue;
            }
            
            // Only consider valid moves
            if (move >= 0) {
                visited_children.push_back({c, visit_count});
            }
        }
        
        if (visited_children.empty()) {
            MCTS_DEBUG("No valid children found");
            return -1;
        }
        
        // Sort by visit count
        std::sort(visited_children.begin(), visited_children.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Get the best child
        Node* bestChild = visited_children[0].first;
        int bestMove = -1;
        
        try {
            bestMove = bestChild->get_move_from_parent();
            int visits = bestChild->get_visit_count();
            float value = bestChild->get_q_value();
            
            MCTS_DEBUG("Selected best move: " << bestMove << " with " 
                      << visits << " visits and value " << value);
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error getting best move info: " << e.what());
            return -1;
        }
        
        return bestMove;
    }
    catch (const std::exception& e) {
        MCTS_DEBUG("Exception in select_move: " << e.what());
        return -1;
    }
}

Node* MCTS::select_node(Node* root) const {
    if (!root) {
        MCTS_DEBUG("select_node called with null root");
        return nullptr;
    }
    
    Node* current = root;
    std::vector<Node*> path;
    
    // Track the path from root to leaf with a maximum depth
    const int MAX_SEARCH_DEPTH = 1000; // Prevent infinite loops
    int depth = 0;
    
    while (current && depth < MAX_SEARCH_DEPTH) {
        try {
            path.push_back(current);
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error adding node to path: " << e.what());
            break;
        }
        
        // Check for terminal state
        try {
            if (current->get_state().is_terminal()) {
                MCTS_DEBUG("Found terminal state during selection");
                break;
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error checking terminal state: " << e.what());
            break;
        }
        
        // Check if this is a leaf node (no children yet)
        bool is_leaf = true;
        try {
            is_leaf = current->is_leaf();
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error checking leaf: " << e.what());
            break;
        }
        
        if (is_leaf) {
            MCTS_DEBUG("Found leaf node (unexpanded) during selection");
            break;
        }
        
        // Get children with proper error handling
        std::vector<Node*> children;
        try {
            children = current->get_children();
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error getting children: " << e.what());
            break;
        }
        
        if (children.empty()) {
            MCTS_DEBUG("Node has no children during selection");
            break;
        }
        
        // Find child with best UCT score
        float bestScore = -std::numeric_limits<float>::infinity();
        Node* bestChild = nullptr;
        
        for (Node* child : children) {
            if (!child) continue;
            
            float score = -std::numeric_limits<float>::infinity();
            try {
                score = uct_score(current, child);
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error calculating UCT score: " << e.what());
                continue;
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        
        if (!bestChild) {
            MCTS_DEBUG("Could not find best child during selection");
            break;
        }
        
        try {
            int visit_count = bestChild->get_visit_count();
            float q_val = bestChild->get_q_value();
            float prior = bestChild->get_prior();
            
            MCTS_DEBUG("Selected child with move " << bestChild->get_move_from_parent() 
                      << ", score: " << bestScore 
                      << ", visits: " << visit_count
                      << ", Q: " << q_val
                      << ", prior: " << prior);
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error logging selection: " << e.what());
            // Continue anyway
        }
        
        current = bestChild;
        depth++;
    }
    
    if (depth >= MAX_SEARCH_DEPTH) {
        MCTS_DEBUG("WARNING: Max search depth reached, possible loop in tree");
        return nullptr;
    }
    
    // Apply virtual loss to entire path for thread diversity
    for (Node* node : path) {
        if (node) {
            try {
                node->add_virtual_loss();
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error adding virtual loss: " << e.what());
                // Continue anyway
            }
        }
    }
    
    if (current) {
        try {
            MCTS_DEBUG("Selected path length: " << path.size() 
                      << ", returning leaf with move: " 
                      << current->get_move_from_parent());
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error logging selected path: " << e.what());
            // Continue anyway
        }
    }
    else {
        MCTS_DEBUG("WARNING: Returning null leaf from selection");
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
        MCTS_DEBUG("backup called with null leaf");
        return;
    }
    
    MCTS_DEBUG("Backing up value " << value << " from leaf with move " << leaf->get_move_from_parent());
    
    Node* current = leaf;
    int leafPlayer = leaf->get_state().current_player;
    
    // Use a maximum depth to prevent infinite loops
    const int MAX_BACKUP_DEPTH = 100;
    int depth = 0;
    
    while (current && depth < MAX_BACKUP_DEPTH) {
        int nodePlayer = current->get_state().current_player;
        
        // Flip the sign for opponent's turns
        float adjusted_value = (nodePlayer == leafPlayer) ? value : -value;
        
        // Update node statistics
        try {
            current->update_stats(adjusted_value);
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error updating node stats: " << e.what());
            break;
        }
        
        // Remove the virtual loss that was added during selection
        try {
            current->remove_virtual_loss();
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error removing virtual loss: " << e.what());
            // Continue anyway
        }
        
        // Store current parent before moving to it
        Node* parent = nullptr;
        try {
            parent = current->get_parent();
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error getting parent: " << e.what());
            break;
        }
        
        // Move to parent - with null check
        current = parent;
        depth++;
    }
    
    if (depth >= MAX_BACKUP_DEPTH) {
        MCTS_DEBUG("WARNING: Max backup depth reached, possible loop in tree");
    }
    
    MCTS_DEBUG("Backup complete");
}

float MCTS::uct_score(const Node* parent, const Node* child) const {
    // Null checks
    if (!parent || !child) {
        return -std::numeric_limits<float>::infinity();
    }

    try {
        float Q = child->get_q_value(); // This already accounts for virtual losses
        float P = child->get_prior();
        int parentVisits = parent->get_visit_count();
        int childVisits = child->get_visit_count();
        
        // Add virtual losses to the child visit count for exploration term
        int virtual_losses = child->get_virtual_losses();
        int effective_child_visits = childVisits + virtual_losses;
        
        // Ensure valid values
        if (parentVisits < 0) parentVisits = 0;
        if (effective_child_visits < 0) effective_child_visits = 0;
        
        float c = config_.c_puct;
        
        // Safety check to prevent square root of negative number (shouldn't happen, but just in case)
        float parentSqrt = std::sqrt(std::max(1e-8f, static_cast<float>(parentVisits)));
        
        // Calculate exploration term with numeric stability
        float U = c * P * parentSqrt / (1.0f + effective_child_visits);
        
        // Return combined score
        return Q + U;
    }
    catch (const std::exception& e) {
        MCTS_DEBUG("Error calculating UCT score: " << e.what());
        return -std::numeric_limits<float>::infinity();
    }
}

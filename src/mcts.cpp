// mcts.cpp
#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

MCTS::MCTS(const MCTSConfig& config,
           std::shared_ptr<BatchingNNInterface> nn,
           int boardSize)
 : config_(config),
   nn_(nn),
   simulations_done_(0),
   attackDefense_(boardSize),
   rng_(std::random_device{}()) // Initialize rng_
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
    MCTS_DEBUG("Starting MCTS search with simplified approach");
    
    // Ensure any previous search is cleaned up
    shutdown_flag_ = true;
    
    // Signal all threads to stop
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }
    
    // Wait briefly for threads to exit
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Join any existing threads with timeout
    for (auto& t : threads_) {
        if (t.joinable()) {
            try {
                auto thread_future = std::async(std::launch::async, [&t]() {
                    t.join();
                });
                
                // Wait with timeout
                if (thread_future.wait_for(std::chrono::milliseconds(500)) != std::future_status::ready) {
                    MCTS_DEBUG("Thread join timeout - thread will be detached");
                    t.detach();
                }
            } catch (...) {
                // If join fails, detach the thread
                t.detach();
            }
        }
    }
    threads_.clear();
    
    // Reset search state
    root_ = std::make_unique<Node>(rootState);
    simulations_done_ = 0;
    shutdown_flag_ = false;
    leaves_in_flight_ = 0;
    
    // Clear leaf evaluation queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::queue<LeafTask> empty;
        std::swap(leaf_queue_, empty);
    }
    
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
    }
    
    // Run simpler single-threaded search for now
    MCTS_DEBUG("Running simplified single-threaded search for " << config_.num_simulations << " simulations");
    
    for (int i = 0; i < config_.num_simulations; i++) {
        // Select a leaf node
        Node* leaf = select_node(root_.get());
        if (!leaf) continue;
        
        // Check if leaf is terminal
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
            // Expand with uniform policy
            auto valid_moves = leaf->get_state().get_valid_moves();
            if (!valid_moves.empty()) {
                std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                leaf->expand(valid_moves, uniform_policy);
                backup(leaf, 0.0f);
            } else {
                // Remove virtual losses
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
            }
        }
        
        simulations_done_++;
        
        if (i % 100 == 0) {
            MCTS_DEBUG("Completed " << i << " simulations");
        }
    }
    
    MCTS_DEBUG("MCTS search completed with " << simulations_done_.load() << " simulations");
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
    MCTS_DEBUG("Queueing leaf node for evaluation");
    
    // Create promise for the result
    std::promise<std::pair<std::vector<float>, float>> promise;
    std::future<std::pair<std::vector<float>, float>> future = promise.get_future();
    
    // Handle null leaf with default values
    if (!leaf) {
        MCTS_DEBUG("Null leaf provided, returning default values");
        std::vector<float> default_policy(config_.parallel_leaf_batch_size, 1.0f / config_.parallel_leaf_batch_size);
        promise.set_value({default_policy, 0.0f});
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
        promise.set_value({empty_policy, value});
        return future;
    }
    
    // Add the leaf to the evaluation queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        MCTS_DEBUG("Adding leaf to queue (current size: " << leaf_queue_.size() << ")");
        
        // Create task with the leaf's state
        LeafTask task;
        task.leaf = leaf;
        task.state = leaf->get_state().copy();  // Deep copy to avoid race conditions
        task.chosen_move = leaf->get_move_from_parent();
        task.result_promise = std::move(promise);
        
        leaf_queue_.push(std::move(task));
        
        // Notify the evaluation thread
        queue_cv_.notify_one();
    }
    
    // Increment counter for leaves in flight
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
    
    while (!shutdown_flag_) {
        // Collect a batch of leaves to evaluate
        std::vector<LeafTask> current_batch;
        current_batch.reserve(batch_size);
        
        // Critical section: get leaves from the queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for leaves or shutdown signal with a short timeout
            auto wait_status = queue_cv_.wait_for(lock, std::chrono::milliseconds(10),
                [this] { return !leaf_queue_.empty() || shutdown_flag_; });
            
            // Check shutdown flag again after wait
            if (shutdown_flag_) {
                MCTS_DEBUG("Shutdown detected in leaf evaluation thread, exiting");
                break;
            }
            
            // Get leaves up to batch size if available
            int count = 0;
            while (!leaf_queue_.empty() && count < batch_size) {
                current_batch.push_back(std::move(leaf_queue_.front()));
                leaf_queue_.pop();
                count++;
            }
        }
        
        // If no leaves and not shutting down, just loop again with short sleep
        if (current_batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Process the batch
        MCTS_DEBUG("Processing batch of " << current_batch.size() << " leaves");
        
        // Handle each leaf individually for better responsiveness
        for (auto& task : current_batch) {
            // Check shutdown flag before each leaf
            if (shutdown_flag_) {
                MCTS_DEBUG("Shutdown detected during batch processing, completing remaining tasks with defaults");
                break;
            }
            
            try {
                // Create default policy
                std::vector<float> default_policy(task.state.board_size * task.state.board_size, 
                                                1.0f / (task.state.board_size * task.state.board_size));
                
                // Just use defaults for now to avoid neural network issues
                task.result_promise.set_value({default_policy, 0.0f});
                
                // Decrement counter for leaves in flight
                leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
            } catch (...) {
                // Promise might already be fulfilled
                MCTS_DEBUG("Error setting promise value");
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
    
    for (auto& task : remaining_tasks) {
        try {
            std::vector<float> default_policy(task.state.board_size * task.state.board_size, 
                                            1.0f / (task.state.board_size * task.state.board_size));
            task.result_promise.set_value({default_policy, 0.0f});
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
                batch[i].result_promise.set_value({valid_policy, results[i].value});
            } else {
                // Use default values on error
                MCTS_DEBUG("Using default values for leaf " << i << " due to neural network error");
                std::vector<float> default_policy(batch[i].state.get_valid_moves().size(), 
                                               1.0f / batch[i].state.get_valid_moves().size());
                batch[i].result_promise.set_value({default_policy, 0.0f});
            }
        } catch (const std::exception& e) {
            // Ensure promise is always fulfilled even on error
            MCTS_DEBUG("Error setting promise value: " << e.what());
            try {
                std::vector<float> default_policy(batch[i].state.get_valid_moves().size(), 
                                               1.0f / batch[i].state.get_valid_moves().size());
                batch[i].result_promise.set_value({default_policy, 0.0f});
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
            Node* leaf = select_node(root_.get());
            
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
            
            // Special case - just use uniform random policy for now
            // This avoids potential hanging with neural network calls
            auto valid_moves = leaf->get_state().get_valid_moves();
            if (!valid_moves.empty()) {
                std::vector<float> uniform_policy(valid_moves.size(), 1.0f / valid_moves.size());
                
                // Expand with uniform policy
                leaf->expand(valid_moves, uniform_policy);
                
                // Use value of 0 (neutral)
                backup(leaf, 0.0f);
                
                // Count this as a completed simulation
                simulations_done_.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Remove virtual losses if we can't expand
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
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
        MCTS_DEBUG("select_node called with null root");
        return nullptr;
    }
    
    Node* current = root;
    std::vector<Node*> path;
    
    // Track the path from root to leaf
    while (true) {
        path.push_back(current);
        
        // Check for terminal state
        if (current->get_state().is_terminal()) {
            MCTS_DEBUG("Found terminal state during selection");
            break;
        }
        
        // Check if this is a leaf node (no children yet)
        if (current->is_leaf()) {
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
            
            float score;
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
        
        MCTS_DEBUG("Selected child with move " << bestChild->get_move_from_parent() 
                  << ", score: " << bestScore 
                  << ", visits: " << bestChild->get_visit_count()
                  << ", Q: " << bestChild->get_q_value()
                  << ", prior: " << bestChild->get_prior());
        
        current = bestChild;
    }
    
    // Apply virtual loss to entire path for thread diversity
    for (Node* node : path) {
        node->add_virtual_loss();
    }
    
    MCTS_DEBUG("Selected path length: " << path.size() 
              << ", returning leaf with move: " 
              << current->get_move_from_parent());
    
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
    
    while (current) {
        int nodePlayer = current->get_state().current_player;
        
        // Flip the sign for opponent's turns
        float adjusted_value = (nodePlayer == leafPlayer) ? value : -value;
        
        // Update node statistics
        current->update_stats(adjusted_value);
        
        // Remove the virtual loss that was added during selection
        current->remove_virtual_loss();
        
        // Move to parent
        current = current->get_parent();
    }
    
    MCTS_DEBUG("Backup complete");
}

float MCTS::uct_score(const Node* parent, const Node* child) const {
    float Q = child->get_q_value(); // This already accounts for virtual losses
    float P = child->get_prior();
    int parentVisits = parent->get_visit_count();
    int childVisits = child->get_visit_count();
    
    // Add virtual losses to the child visit count for exploration term
    int virtual_losses = child->get_virtual_losses();
    int effective_child_visits = childVisits + virtual_losses;
    
    float c = config_.c_puct;
    float U = c * P * std::sqrt((float)parentVisits + 1e-8f) / (1 + effective_child_visits);
    
    return Q + U;
}

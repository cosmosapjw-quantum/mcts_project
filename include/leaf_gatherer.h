#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include "gomoku.h"
#include "python_nn_proxy.h"
#include "nn_interface.h"
#include "attack_defense.h"
#include "node.h"
#include "debug.h"

/**
 * A dedicated thread for gathering leaf nodes and processing them in batches.
 * This provides a middle ground between full parallel search and single-threaded search.
 */
class LeafGatherer {
public:
    // Initialize with neural network interface and attack/defense module
    LeafGatherer(std::shared_ptr<PythonNNProxy> nn, 
                 AttackDefenseModule& attack_defense, // Remove const here
                 int batch_size = 16)
        : nn_(nn),
          attack_defense_(attack_defense),
          batch_size_(batch_size),
          shutdown_(false),
          total_processed_(0)
    {
        MCTS_DEBUG("Creating LeafGatherer with batch size " << batch_size);
        // Start the worker thread
        worker_thread_ = std::thread(&LeafGatherer::worker_function, this);
    }
    
    // Destructor - ensure clean shutdown
    ~LeafGatherer() {
        shutdown();
    }
    
    // Struct to hold leaf evaluation request
    struct LeafEvalRequest {
        Node* leaf;
        Gamestate state;
        int chosen_move;
        std::shared_ptr<std::promise<std::pair<std::vector<float>, float>>> result_promise;
    };
    
    // Queue a leaf for evaluation
    std::future<std::pair<std::vector<float>, float>> queue_leaf(Node* leaf) {
        // Check for shutdown
        if (shutdown_) {
            MCTS_DEBUG("LeafGatherer shutting down, returning default values");
            auto promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
            
            std::vector<float> default_policy;
            if (leaf && !leaf->get_state().is_terminal()) {
                auto valid_moves = leaf->get_state().get_valid_moves();
                default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
            }
            
            promise->set_value({default_policy, 0.0f});
            return promise->get_future();
        }
        
        // Handle terminal nodes directly
        if (leaf && leaf->get_state().is_terminal()) {
            auto promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
            
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
            
            promise->set_value({std::vector<float>(), value});
            return promise->get_future();
        }
        
        // Create a request
        LeafEvalRequest request;
        request.leaf = leaf;
        
        if (leaf) {
            request.state = leaf->get_state().copy();
            request.chosen_move = leaf->get_move_from_parent();
            
            // Fix chosen_move if invalid
            if (request.chosen_move < 0) {
                auto valid_moves = request.state.get_valid_moves();
                if (!valid_moves.empty()) {
                    request.chosen_move = valid_moves[0];
                } else {
                    request.chosen_move = 0;
                }
            }
        } else {
            // Default values for null leaf
            request.chosen_move = 0;
        }
        
        request.result_promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
        auto future = request.result_promise->get_future();
        
        // Add to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            // Check queue size
            if (request_queue_.size() >= 1000) {
                MCTS_DEBUG("Queue full, returning default values");
                std::vector<float> default_policy;
                if (leaf && !leaf->get_state().is_terminal()) {
                    auto valid_moves = leaf->get_state().get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                }
                
                request.result_promise->set_value({default_policy, 0.0f});
                return future;
            }
            
            request_queue_.push(std::move(request));
            queue_cv_.notify_one();
        }
        
        return future;
    }
    
    // Shutdown the worker thread
    void shutdown() {
        if (!shutdown_) {
            MCTS_DEBUG("Shutting down LeafGatherer");
            shutdown_ = true;
            
            // Notify worker thread
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                queue_cv_.notify_all();
            }
            
            // Wait for worker thread to finish
            if (worker_thread_.joinable()) {
                worker_thread_.join();
            }
            
            // Process any remaining items in queue
            process_remaining_queue();
            
            MCTS_DEBUG("LeafGatherer shutdown complete, processed " << total_processed_ << " leaves total");
        }
    }
    
    // Get total leaves processed
    int get_total_processed() const {
        return total_processed_;
    }

private:
    std::shared_ptr<PythonNNProxy> nn_;
    AttackDefenseModule& attack_defense_;  // Remove const here
    int batch_size_;
    std::atomic<bool> shutdown_;
    std::atomic<int> total_processed_;
    
    std::queue<LeafEvalRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::thread worker_thread_;
    
    // Process remaining items in queue after shutdown
    void process_remaining_queue() {
        std::queue<LeafEvalRequest> remaining;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            remaining.swap(request_queue_);
        }
        
        MCTS_DEBUG("Processing " << remaining.size() << " remaining requests after shutdown");
        
        while (!remaining.empty()) {
            auto& request = remaining.front();
            
            if (request.result_promise) {
                // Fulfill with default values
                std::vector<float> default_policy;
                
                if (!request.state.is_terminal()) {
                    auto valid_moves = request.state.get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                }
                
                request.result_promise->set_value({default_policy, 0.0f});
            }
            
            remaining.pop();
        }
    }
    
    // Worker thread function
    void worker_function() {
        MCTS_DEBUG("LeafGatherer worker thread started");
        
        while (!shutdown_) {
            std::vector<LeafEvalRequest> batch;
            
            // Wait for items to process
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                
                // Wait for items or shutdown
                queue_cv_.wait(lock, [this] {
                    return !request_queue_.empty() || shutdown_;
                });
                
                // Check shutdown again
                if (shutdown_ && request_queue_.empty()) {
                    break;
                }
                
                // Collect a batch
                int count = 0;
                while (!request_queue_.empty() && count < batch_size_) {
                    batch.push_back(std::move(request_queue_.front()));
                    request_queue_.pop();
                    count++;
                }
            }
            
            // Process the batch if not empty
            if (!batch.empty()) {
                process_batch(batch);
                
                // Update counter
                total_processed_ += batch.size();
                
                if (total_processed_ % 100 == 0) {
                    MCTS_DEBUG("LeafGatherer processed " << total_processed_ << " leaves total");
                }
            }
        }
        
        MCTS_DEBUG("LeafGatherer worker thread exiting");
    }
    
    // Process a batch of requests
    void process_batch(std::vector<LeafEvalRequest>& batch) {
        MCTS_DEBUG("Processing batch of " << batch.size() << " leaves");
        
        // Prepare data for attack/defense module
        std::vector<std::vector<std::vector<int>>> board_batch;
        std::vector<int> chosen_moves;
        std::vector<int> player_batch;
        
        for (const auto& request : batch) {
            board_batch.push_back(request.state.get_board());
            chosen_moves.push_back(request.chosen_move);
            player_batch.push_back(request.state.current_player);
        }
        
        // Calculate attack/defense bonuses
        std::vector<float> attack_vec;
        std::vector<float> defense_vec;
        
        try {
            auto [a_vec, d_vec] = attack_defense_.compute_bonuses(
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
                batch[i].state, chosen_moves[i], attack, defense);
                
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
        
        // Process results and fulfill promises
        for (size_t i = 0; i < batch.size(); i++) {
            try {
                if (!batch[i].result_promise) {
                    continue; // Skip if no promise (shouldn't happen)
                }
                
                auto valid_moves = batch[i].state.get_valid_moves();
                
                if (valid_moves.empty()) {
                    // Terminal state or no valid moves
                    batch[i].result_promise->set_value({std::vector<float>(), 0.0f});
                    continue;
                }
                
                std::vector<float> valid_policy;
                float value = 0.0f;
                
                if (i < results.size() && !results[i].policy.empty()) {
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
                }
                
                // Fulfill the promise
                batch[i].result_promise->set_value({valid_policy, value});
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error processing result for leaf " << i << ": " << e.what());
                
                // Use default values on error
                std::vector<float> default_policy;
                
                if (!batch[i].state.is_terminal()) {
                    auto valid_moves = batch[i].state.get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                }
                
                try {
                    if (batch[i].result_promise) {
                        batch[i].result_promise->set_value({default_policy, 0.0f});
                    }
                } catch (...) {
                    // Ignore errors when setting promise value
                }
            }
        }
    }
};
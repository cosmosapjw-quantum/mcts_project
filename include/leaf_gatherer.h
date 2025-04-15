// include/leaf_gatherer.h - Enhanced version with multiple worker threads

#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <chrono>
#include <functional>
#include "gomoku.h"
#include "python_nn_proxy.h"
#include "nn_interface.h"
#include "attack_defense.h"
#include "node.h"
#include "debug.h"

/**
 * A thread pool for gathering leaf nodes and processing them in batches.
 * Utilizes multiple worker threads to improve parallelism on multi-core CPUs.
 */
class LeafGatherer {
public:
    // Initialize with neural network interface, attack/defense module, and thread configuration
    LeafGatherer(std::shared_ptr<PythonNNProxy> nn, 
                 AttackDefenseModule& attack_defense,
                 int batch_size = 256,        // Default increased to match NN batch size
                 int num_workers = 4)         // Default worker threads
        : nn_(nn),
          attack_defense_(attack_defense),
          batch_size_(batch_size),
          shutdown_(false),
          total_processed_(0),
          active_workers_(0),
          max_workers_(num_workers)
    {
        MCTS_DEBUG("Creating LeafGatherer with batch size " << batch_size << " and " << num_workers << " workers");
        
        // Determine optimal number of workers if not specified
        if (num_workers <= 0) {
            // Auto-detect based on hardware
            unsigned int hw_threads = std::thread::hardware_concurrency();
            if (hw_threads == 0) hw_threads = 8; // Fallback if detection fails
            
            // For Ryzen 9 5900X with 24 threads, we want to leave some threads
            // for the main search process and Python NN inference
            max_workers_ = std::min<int>(hw_threads / 3, 8); // Use at most 1/3 of threads or 8
            MCTS_DEBUG("Auto-configured worker count to " << max_workers_ << " based on hardware");
        } else {
            max_workers_ = num_workers;
        }
        
        // Cap batch size to reasonable limits
        if (batch_size_ <= 0) {
            batch_size_ = 256; // Default batch size
        } else if (batch_size_ > 1024) {
            batch_size_ = 1024; // Maximum reasonable batch size
        }
        
        // Initialize worker threads
        startWorkers();
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
        
        // Timestamp for monitoring
        std::chrono::steady_clock::time_point submit_time;
        
        LeafEvalRequest() : leaf(nullptr), chosen_move(0) {
            submit_time = std::chrono::steady_clock::now();
        }
    };
    
    // Queue a leaf for evaluation with improved monitoring
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
            
            // Check queue size and monitor queue health
            const int MAX_QUEUE_SIZE = 2000;
            if (request_queue_.size() >= MAX_QUEUE_SIZE) {
                MCTS_DEBUG("Queue full (" << request_queue_.size() << " items), returning default values");
                std::vector<float> default_policy;
                if (leaf && !leaf->get_state().is_terminal()) {
                    auto valid_moves = leaf->get_state().get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                }
                
                request.result_promise->set_value({default_policy, 0.0f});
                
                // Log queue statistics to help diagnose issues
                MCTS_DEBUG("Queue health: active workers=" << active_workers_ 
                          << ", total processed=" << total_processed_);
                
                // Check if workers are stuck by measuring time since last activity
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_activity_time_).count();
                    
                MCTS_DEBUG("Time since last activity: " << elapsed << "ms");
                
                // If we have inactive workers and a full queue, try restarting workers
                if (active_workers_ < max_workers_ && elapsed > 5000) {
                    MCTS_DEBUG("Detected potential worker stall, restarting workers");
                    restartWorkers();
                }
                
                return future;
            }
            
            // Update submission time
            request.submit_time = std::chrono::steady_clock::now();
            
            request_queue_.push(std::move(request));
            queue_cv_.notify_one();
        }
        
        return future;
    }
    
    // Shutdown the worker threads with improved cleanup
    void shutdown() {
        MCTS_DEBUG("LeafGatherer shutdown initiated");
        
        // Set shutdown flag first (atomic operation)
        bool was_running = false;
        if (!shutdown_.compare_exchange_strong(was_running, true)) {
            MCTS_DEBUG("LeafGatherer already shutting down");
            return;  // Already shutting down
        }
        
        MCTS_DEBUG("LeafGatherer notifying all workers");
        
        // Notify all worker threads immediately
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_cv_.notify_all();
        }
        
        // Track worker join status
        std::vector<bool> joined(worker_threads_.size(), false);
        
        // Join worker threads with timeout to prevent deadlocks
        MCTS_DEBUG("Waiting for " << worker_threads_.size() << " workers to exit (with timeout)");
        
        const auto join_start = std::chrono::steady_clock::now();
        const int PER_THREAD_JOIN_TIMEOUT_MS = 100;  // 100ms per thread
        int joined_count = 0;
        
        for (size_t i = 0; i < worker_threads_.size(); ++i) {
            if (!worker_threads_[i].joinable()) {
                MCTS_DEBUG("Worker " << i << " not joinable, skipping");
                joined[i] = true;
                joined_count++;
                continue;
            }
            
            auto timeout_point = std::chrono::steady_clock::now() + 
                std::chrono::milliseconds(PER_THREAD_JOIN_TIMEOUT_MS);
                
            // Use C++11 join with timeout via interruption
            std::thread joiner([&, i]() {
                try {
                    worker_threads_[i].join();
                    joined[i] = true;
                    joined_count++;
                    MCTS_DEBUG("Worker " << i << " joined successfully");
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error joining worker " << i << ": " << e.what());
                }
            });
            
            // Wait for joiner with timeout
            if (joiner.joinable()) {
                auto status = std::chrono::steady_clock::now();
                if (status < timeout_point) {
                    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                        timeout_point - status).count();
                    std::this_thread::sleep_for(std::chrono::milliseconds(remaining));
                }
                
                // Detach joiner if it's still running
                if (joiner.joinable()) {
                    MCTS_DEBUG("Join timeout for worker " << i << ", detaching");
                    joiner.detach();
                }
            }
        }
        
        // Detach any threads that couldn't be joined to avoid resource leaks
        for (size_t i = 0; i < worker_threads_.size(); ++i) {
            if (!joined[i] && worker_threads_[i].joinable()) {
                MCTS_DEBUG("Worker " << i << " failed to join, detaching");
                worker_threads_[i].detach();
            }
        }
        
        // Clear worker threads vector
        worker_threads_.clear();
        
        // Process remaining items with timeout protection
        MCTS_DEBUG("Processing remaining items in queue");
        
        std::queue<LeafEvalRequest> remaining;
        
        // Get remaining items with lock to prevent race conditions
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            remaining.swap(request_queue_);
        }
        
        // Process a limited number of remaining items to avoid stalling forever
        const int MAX_ITEMS_TO_PROCESS = 100;
        int processed = 0;
        
        while (!remaining.empty() && processed < MAX_ITEMS_TO_PROCESS) {
            try {
                auto& request = remaining.front();
                
                if (request.result_promise) {
                    std::vector<float> default_policy;
                    
                    if (!request.state.is_terminal()) {
                        auto valid_moves = request.state.get_valid_moves();
                        default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                    }
                    
                    request.result_promise->set_value({default_policy, 0.0f});
                    processed++;
                }
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error fulfilling promise: " << e.what());
            }
            
            remaining.pop();
        }
        
        // Report if we couldn't process all items
        if (!remaining.empty()) {
            MCTS_DEBUG("WARNING: " << remaining.size() << " items left unprocessed after max limit");
            // Clear remaining items to avoid memory leaks
            std::queue<LeafEvalRequest>().swap(remaining);
        }
        
        // Reset active workers count
        active_workers_ = 0;
        
        // Calculate shutdown duration
        auto shutdown_end = std::chrono::steady_clock::now();
        auto shutdown_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            shutdown_end - join_start).count();
        
        MCTS_DEBUG("LeafGatherer shutdown completed in " << shutdown_duration << "ms, "
                  << "processed " << processed << " remaining items, "
                  << "joined " << joined_count << "/" << joined.size() << " threads");
    }
    
    // Get total leaves processed
    int get_total_processed() const {
        return total_processed_;
    }
    
    // Get current queue size
    int get_queue_size() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return request_queue_.size();
    }
    
    // Get current active worker count
    int get_active_workers() const {
        return active_workers_;
    }
    
    // Get statistics for diagnostics
    std::string get_stats() const {
        std::ostringstream oss;
        oss << "LeafGatherer stats: "
            << "processed=" << total_processed_
            << ", active_workers=" << active_workers_
            << ", max_workers=" << max_workers_
            << ", batch_size=" << batch_size_
            << ", queue_size=" << get_queue_size();
            
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_activity_time_).count();
        oss << ", time_since_activity=" << elapsed << "ms";
        
        return oss.str();
    }

    // Getter for batch size
    int get_batch_size() const {
        return batch_size_;
    }

private:
    std::shared_ptr<PythonNNProxy> nn_;
    AttackDefenseModule& attack_defense_;
    int batch_size_;
    std::atomic<bool> shutdown_;
    std::atomic<int> total_processed_;
    std::atomic<int> active_workers_;
    int max_workers_;
    
    std::queue<LeafEvalRequest> request_queue_;
    mutable std::mutex queue_mutex_; // Add 'mutable' to allow locking in const methods
    std::condition_variable queue_cv_;
    
    std::vector<std::thread> worker_threads_;
    
    // Timestamp of last successful batch processing
    std::chrono::steady_clock::time_point last_activity_time_;
    
    // Process remaining items in queue after shutdown
    void process_remaining_queue() {
        std::queue<LeafEvalRequest> remaining;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            remaining.swap(request_queue_);
        }
        
        MCTS_DEBUG("Processing " << remaining.size() << " remaining requests after shutdown");
        
        // Count processed promises
        int processed = 0;
        
        while (!remaining.empty()) {
            auto& request = remaining.front();
            
            try {
                if (request.result_promise) {
                    // Fulfill with default values
                    std::vector<float> default_policy;
                    
                    if (!request.state.is_terminal()) {
                        auto valid_moves = request.state.get_valid_moves();
                        default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                    }
                    
                    request.result_promise->set_value({default_policy, 0.0f});
                    processed++;
                }
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error fulfilling promise: " << e.what());
            }
            
            remaining.pop();
        }
        
        MCTS_DEBUG("Fulfilled " << processed << " promises during cleanup");
    }
    
    // Start worker threads
    void startWorkers() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Clear existing threads if any
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.detach(); // Detach instead of join to avoid blocking
            }
        }
        worker_threads_.clear();
        
        // Reset shutdown flag
        shutdown_ = false;
        
        // Initialize last activity time
        last_activity_time_ = std::chrono::steady_clock::now();
        
        // Start new worker threads
        worker_threads_.reserve(max_workers_);
        
        MCTS_DEBUG("Starting " << max_workers_ << " worker threads");
        
        for (int i = 0; i < max_workers_; i++) {
            worker_threads_.emplace_back(&LeafGatherer::worker_function, this, i);
        }
        
        active_workers_ = max_workers_;
    }
    
    // Restart workers if they appear to be stuck
    void restartWorkers() {
        MCTS_DEBUG("Restarting worker threads");
        
        // Detach existing threads
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.detach(); // Detach to avoid blocking
            }
        }
        worker_threads_.clear();
        
        // Reset active worker count
        active_workers_ = 0;
        
        // Start fresh workers
        startWorkers();
        
        MCTS_DEBUG("Worker threads restarted");
    }
    
    // Worker thread function with thread ID
    void worker_function(int thread_id) {
        MCTS_DEBUG("LeafGatherer worker " << thread_id << " started");
        
        // Store thread ID for debugging
        thread_local int my_thread_id = thread_id;
        
        // Report as active
        active_workers_++;
        
        // Use a try/catch block to handle all exceptions
        try {
            while (!shutdown_.load(std::memory_order_acquire)) {
                // Create a local batch vector that will be destroyed properly on thread exit
                std::vector<LeafEvalRequest> local_batch;
                
                // Use a timeout when waiting for work
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    
                    // Wait with short timeout to check shutdown flag frequently
                    auto wait_result = queue_cv_.wait_for(lock, std::chrono::milliseconds(50), 
                        [this] { 
                            return !request_queue_.empty() || shutdown_.load(std::memory_order_acquire); 
                        });
                    
                    // Check shutdown flag with lock held
                    if (shutdown_.load(std::memory_order_acquire)) {
                        MCTS_DEBUG("Worker " << my_thread_id << " shutdown detected during wait");
                        break;
                    }
                    
                    // If queue is empty after timeout, continue checking shutdown flag
                    if (request_queue_.empty()) {
                        continue;
                    }
                    
                    // Sanity check that queue size is valid
                    if (request_queue_.size() > 10000) {
                        MCTS_DEBUG("WARNING: Worker " << my_thread_id << " detected very large queue size: " 
                                  << request_queue_.size());
                    }
                    
                    // Get a reasonable batch size (1-64)
                    int current_batch_size = std::min(static_cast<int>(request_queue_.size()), 64);
                    current_batch_size = std::max(1, current_batch_size);
                    
                    // Move items to local batch (which will be properly destroyed on thread exit)
                    local_batch.reserve(current_batch_size);
                    for (int i = 0; i < current_batch_size && !request_queue_.empty(); ++i) {
                        local_batch.push_back(std::move(request_queue_.front()));
                        request_queue_.pop();
                    }
                }
                
                // Double-check shutdown flag before processing
                if (shutdown_.load(std::memory_order_acquire)) {
                    MCTS_DEBUG("Worker " << my_thread_id << " shutdown detected before processing batch");
                    
                    // Complete all promises to avoid memory leaks
                    for (auto& item : local_batch) {
                        try {
                            if (item.result_promise) {
                                std::vector<float> default_policy;
                                if (!item.state.is_terminal()) {
                                    auto valid_moves = item.state.get_valid_moves();
                                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                                }
                                item.result_promise->set_value({default_policy, 0.0f});
                            }
                        } catch (const std::exception& e) {
                            MCTS_DEBUG("Error fulfilling promise during shutdown: " << e.what());
                        }
                    }
                    
                    // Clear the batch explicitly
                    local_batch.clear();
                    break;
                }
                
                // Process the local batch
                if (!local_batch.empty()) {
                    try {
                        process_batch(local_batch, my_thread_id);
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("Worker " << my_thread_id << " exception during process_batch: " << e.what());
                        
                        // Complete all remaining promises
                        for (auto& item : local_batch) {
                            try {
                                if (item.result_promise) {
                                    std::vector<float> default_policy;
                                    if (!item.state.is_terminal()) {
                                        auto valid_moves = item.state.get_valid_moves();
                                        default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                                    }
                                    item.result_promise->set_value({default_policy, 0.0f});
                                }
                            } catch (...) {
                                // Ignore errors in cleanup
                            }
                        }
                    }
                    
                    // Clear the batch explicitly
                    local_batch.clear();
                }
                
                // Check for shutdown one more time before looping
                if (shutdown_.load(std::memory_order_acquire)) {
                    MCTS_DEBUG("Worker " << my_thread_id << " shutdown detected after processing");
                    break;
                }
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Worker " << my_thread_id << " terminated with exception: " << e.what());
        }
        catch (...) {
            MCTS_DEBUG("Worker " << my_thread_id << " terminated with unknown exception");
        }
        
        // Final shutdown check to avoid segfault from accessing freed resources
        if (shutdown_.load(std::memory_order_acquire)) {
            MCTS_DEBUG("Worker " << my_thread_id << " detected shutdown, exiting cleanly");
        } else {
            MCTS_DEBUG("Worker " << my_thread_id << " exiting unexpectedly while system is still running");
        }
        
        // Report as inactive before exiting
        active_workers_--;
        
        MCTS_DEBUG("LeafGatherer worker " << my_thread_id << " exiting");
    }
    
    // Process a batch of requests
    void process_batch(std::vector<LeafEvalRequest>& batch, int thread_id) {
        MCTS_DEBUG("Worker " << thread_id << " processing batch of " << batch.size() << " leaves");
        
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
            auto attack_defense_start = std::chrono::steady_clock::now();
            
            auto [a_vec, d_vec] = attack_defense_.compute_bonuses(
                board_batch, chosen_moves, player_batch);
            attack_vec = a_vec;
            defense_vec = d_vec;
            
            auto attack_defense_end = std::chrono::steady_clock::now();
            auto attack_defense_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                attack_defense_end - attack_defense_start).count();
                
            MCTS_DEBUG("Worker " << thread_id << " computed attack/defense bonuses in " 
                       << attack_defense_duration << "ms");
        } catch (const std::exception& e) {
            MCTS_DEBUG("Worker " << thread_id << " error computing attack/defense bonuses: " << e.what());
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
            
            MCTS_DEBUG("Worker " << thread_id << " neural network batch inference completed in " << nn_duration 
                       << "ms for " << batch.size() << " leaves");
        } catch (const std::exception& e) {
            MCTS_DEBUG("Worker " << thread_id << " error in neural network batch inference: " << e.what());
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
                MCTS_DEBUG("Worker " << thread_id << " error processing result for leaf " << i << ": " << e.what());
                
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
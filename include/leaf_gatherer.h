// include/leaf_gatherer.h - Enhanced version with more robust worker threads

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
#include <algorithm>
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
                int batch_size = 256,
                int num_workers = 1)  // Changed from 4 to 1
        : nn_(nn),
        attack_defense_(attack_defense),
        batch_size_(batch_size),
        shutdown_(false),
        total_processed_(0),
        active_workers_(0),
        max_workers_(num_workers),
        health_check_passed_(true)
    {
        MCTS_DEBUG("Creating LeafGatherer with batch size " << batch_size << " and " << num_workers << " workers");
        
        // Use 1 worker by default, regardless of hardware
        if (num_workers <= 0) {
            max_workers_ = 1;  // Changed to always use 1 worker
            MCTS_DEBUG("Auto-configured worker count to " << max_workers_);
        } else {
            max_workers_ = std::min(num_workers, 1);  // Cap at 1 worker
        }
        
        // Cap batch size to reasonable limits
        if (batch_size_ <= 0) {
            batch_size_ = 256; // Default batch size
        } else if (batch_size_ > 1024) {
            batch_size_ = 1024; // Maximum reasonable batch size
        }
        
        // Initialize last activity time
        last_activity_time_ = std::chrono::steady_clock::now();
        
        // Initialize worker threads
        startWorkers();
        
        // Start health monitoring thread
        startHealthMonitor();
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
        if (shutdown_.load(std::memory_order_acquire)) {
            MCTS_DEBUG("LeafGatherer shutting down, returning default values");
            auto promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
            
            // IMPROVED: Create appropriate default values based on state
            std::vector<float> default_policy;
            float default_value = 0.0f;
            
            // Handle terminal nodes specifically
            if (leaf && leaf->get_state().is_terminal()) {
                // For terminal nodes, return empty policy and appropriate value
                int winner = leaf->get_state().get_winner();
                int current_player = leaf->get_state().current_player;
                
                if (winner == current_player) {
                    default_value = 1.0f;
                } else if (winner == 0) {
                    default_value = 0.0f; // Draw
                } else {
                    default_value = -1.0f; // Loss
                }
            } else if (leaf && !leaf->get_state().is_terminal()) {
                // For non-terminal nodes, use uniform policy over valid moves
                auto valid_moves = leaf->get_state().get_valid_moves();
                default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
            }
            
            promise->set_value({default_policy, default_value});
            return promise->get_future();
        }
        
        // Handle terminal nodes directly (without queuing)
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
            try {
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
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error copying state: " << e.what());
                // Fall back to default values if state copy fails
                auto promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
                promise->set_value({std::vector<float>(), 0.0f});
                return promise->get_future();
            }
        } else {
            // Default values for null leaf
            request.chosen_move = 0;
        }
        
        request.result_promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
        auto future = request.result_promise->get_future();
        
        // Check if health check has been failing
        if (!health_check_passed_.load(std::memory_order_relaxed)) {
            MCTS_DEBUG("Health check failing, attempting worker recovery");
            restartWorkers();
            health_check_passed_.store(true, std::memory_order_relaxed);
        }
        
        // Add to queue with health check
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            // Check queue size and monitor queue health
            const int MAX_QUEUE_SIZE = 1000;
            if (request_queue_.size() >= MAX_QUEUE_SIZE) {
                MCTS_DEBUG("Queue full (" << request_queue_.size() << " items), returning default values");
                
                // IMPROVED: Create appropriate default values based on state
                std::vector<float> default_policy;
                float default_value = 0.0f;
                
                // Handle terminal nodes specifically
                if (leaf && leaf->get_state().is_terminal()) {
                    // For terminal nodes, return empty policy and appropriate value
                    int winner = leaf->get_state().get_winner();
                    int current_player = leaf->get_state().current_player;
                    
                    if (winner == current_player) {
                        default_value = 1.0f;
                    } else if (winner == 0) {
                        default_value = 0.0f; // Draw
                    } else {
                        default_value = -1.0f; // Loss
                    }
                } else if (leaf && !leaf->get_state().is_terminal()) {
                    // For non-terminal nodes, use uniform policy over valid moves
                    auto valid_moves = leaf->get_state().get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                }
                
                request.result_promise->set_value({default_policy, default_value});
                
                // Log queue statistics to help diagnose issues
                MCTS_DEBUG("Queue health: active workers=" << active_workers_ 
                          << ", total processed=" << total_processed_);
                
                // Check if workers are stuck by measuring time since last activity
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_activity_time_).count();
                    
                MCTS_DEBUG("Time since last activity: " << elapsed << "ms");
                
                // If we have inactive workers and a full queue, try restarting workers
                if (active_workers_.load(std::memory_order_relaxed) < max_workers_ && elapsed > 2000) {
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
        
        // Check if we need more workers
        if (active_workers_.load(std::memory_order_relaxed) < max_workers_ / 2) {
            MCTS_DEBUG("Active workers below half capacity, restarting workers");
            restartWorkers();
        }
        
        return future;
    }
    
    // Shutdown the worker threads with improved cleanup
    void shutdown() {
        MCTS_DEBUG("LeafGatherer shutdown initiated");
        
        // Set shutdown flag first (atomic operation)
        bool was_running = false;
        if (!shutdown_.compare_exchange_strong(was_running, true, std::memory_order_acq_rel)) {
            MCTS_DEBUG("LeafGatherer already shutting down");
            return;  // Already shutting down
        }
        
        // Stop health monitor with timeout
        if (health_thread_.joinable()) {
            auto health_join_start = std::chrono::steady_clock::now();
            const int HEALTH_JOIN_TIMEOUT_MS = 100;  // 100ms timeout
            
            std::thread([&]() {
                health_thread_.join();
            }).detach();
            
            // Wait briefly for health thread to join
            std::this_thread::sleep_for(std::chrono::milliseconds(HEALTH_JOIN_TIMEOUT_MS));
            
            // If still joinable after timeout, detach it
            if (health_thread_.joinable()) {
                MCTS_DEBUG("Health monitor thread didn't join within timeout, detaching");
                health_thread_.detach();
            } else {
                MCTS_DEBUG("Health monitor thread joined successfully");
            }
        }
        
        MCTS_DEBUG("LeafGatherer notifying all workers");
        
        // Notify all worker threads immediately
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_cv_.notify_all();
        }
        
        // IMPROVED: First, fulfill all promises with default values
        int cleared_requests = 0;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            std::vector<LeafEvalRequest> remaining_requests;
            while (!request_queue_.empty()) {
                remaining_requests.push_back(std::move(request_queue_.front()));
                request_queue_.pop();
            }
            
            // First clear the queue
            MCTS_DEBUG("Cleared " << remaining_requests.size() << " requests from queue");
            
            // Now fulfill promises without accessing the queue again
            for (auto& request : remaining_requests) {
                if (request.result_promise) {
                    try {
                        // IMPROVED: Handle terminal states properly
                        std::vector<float> default_policy;
                        float default_value = 0.0f;
                        
                        if (request.state.is_terminal()) {
                            // For terminal nodes, return empty policy and appropriate value
                            int winner = request.state.get_winner();
                            int current_player = request.state.current_player;
                            
                            if (winner == current_player) {
                                default_value = 1.0f;
                            } else if (winner == 0) {
                                default_value = 0.0f; // Draw
                            } else {
                                default_value = -1.0f; // Loss
                            }
                        } else {
                            // For non-terminal nodes, use uniform policy over valid moves
                            auto valid_moves = request.state.get_valid_moves();
                            default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                        }
                        
                        request.result_promise->set_value({default_policy, default_value});
                        cleared_requests++;
                        
                        // If this has a leaf node, clean up its virtual losses too
                        if (request.leaf) {
                            Node* current = request.leaf;
                            while (current) {
                                current->clear_all_virtual_losses();  // Use our improved method
                                current = current->get_parent();
                            }
                        }
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error setting promise value during shutdown: " << e.what());
                    } catch (...) {
                        MCTS_DEBUG("Unknown error setting promise value during shutdown");
                    }
                }
            }
        }
        
        if (cleared_requests > 0) {
            MCTS_DEBUG("Fulfilled " << cleared_requests << " promises during shutdown");
        }
        
        // Make a local copy of worker threads to avoid race conditions
        std::vector<std::thread> workers_to_join;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            workers_to_join = std::move(worker_threads_);
            worker_threads_.clear();
        }
        
        // Join worker threads with individual timeouts - CRITICAL CHANGE
        MCTS_DEBUG("Waiting for " << workers_to_join.size() << " workers to exit (with timeout)");
        
        const int JOIN_TIMEOUT_PER_THREAD_MS = 100;  // 100ms timeout per thread
        int joined_count = 0;
        
        for (size_t i = 0; i < workers_to_join.size(); ++i) {
            if (workers_to_join[i].joinable()) {
                // Create a detached thread to attempt joining with timeout
                std::thread([i, &workers_to_join, &joined_count]() {
                    try {
                        if (workers_to_join[i].joinable()) {
                            workers_to_join[i].join();
                            joined_count++;
                        }
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error joining worker thread: " << e.what());
                    } catch (...) {
                        MCTS_DEBUG("Unknown error joining worker thread");
                    }
                }).detach();
                
                // Wait for the joining to complete with timeout
                std::this_thread::sleep_for(std::chrono::milliseconds(JOIN_TIMEOUT_PER_THREAD_MS));
                
                // If the thread is still joinable, detach it
                if (workers_to_join[i].joinable()) {
                    try {
                        MCTS_DEBUG("Worker " << i << " failed to join within timeout, detaching");
                        workers_to_join[i].detach();
                    } catch (...) {
                        // Ignore any detach errors
                    }
                }
            }
        }
        
        // Reset active workers count
        active_workers_.store(0, std::memory_order_release);
        
        MCTS_DEBUG("LeafGatherer shutdown completed, joined " 
                  << joined_count << "/" << workers_to_join.size() << " threads");
    }
    
    // Get current queue size
    int get_queue_size() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return request_queue_.size();
    }
    
    // Get current active worker count
    int get_active_workers() const {
        return active_workers_.load(std::memory_order_acquire);
    }
    
    // Get statistics for diagnostics
    std::string get_stats() const {
        std::ostringstream oss;
        oss << "LeafGatherer stats: "
            << "processed=" << total_processed_.load(std::memory_order_acquire)
            << ", active_workers=" << active_workers_.load(std::memory_order_acquire)
            << ", max_workers=" << max_workers_
            << ", batch_size=" << batch_size_
            << ", queue_size=" << get_queue_size();
            
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_activity_time_).count();
        oss << ", time_since_activity=" << elapsed << "ms";
        
        // Add health check status
        oss << ", health_check=" << (health_check_passed_.load(std::memory_order_relaxed) ? "passed" : "failing");
        
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
    std::thread health_thread_;
    std::atomic<bool> health_check_passed_;
    
    // Timestamp of last successful batch processing
    std::chrono::steady_clock::time_point last_activity_time_;
    
    // Start health monitoring thread
    void startHealthMonitor() {
        health_thread_ = std::thread([this]() {
            MCTS_DEBUG("Health monitor thread started");
            
            while (!shutdown_.load(std::memory_order_acquire)) {
                // Sleep for a short time
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                
                // Skip if shutting down
                if (shutdown_.load(std::memory_order_acquire)) {
                    break;
                }
                
                // Check for worker health
                checkWorkerHealth();
            }
            
            MCTS_DEBUG("Health monitor thread exiting");
        });
    }
    
    // Check if workers are healthy
    void checkWorkerHealth() {
        // Check time since last activity
        auto now = std::chrono::steady_clock::now();
        auto inactivity_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_activity_time_).count();
            
        // If queue is not empty and no activity for some time, workers might be stuck
        int queue_size = get_queue_size();
        int active = active_workers_.load(std::memory_order_relaxed);
        
        if (queue_size > 0 && inactivity_ms > 3000 && active < max_workers_ / 2) {
            MCTS_DEBUG("Health check: Detected potential worker stall - " 
                    << "queue_size=" << queue_size 
                    << ", active_workers=" << active 
                    << ", inactivity=" << inactivity_ms << "ms");
                    
            health_check_passed_.store(false, std::memory_order_relaxed);
        } else {
            health_check_passed_.store(true, std::memory_order_relaxed);
        }
    }
    
    // Start worker threads
    void startWorkers() {
        // Use a double-lock pattern to ensure thread safety
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Additional atomic flag to track if workers are being started
        static std::atomic<bool> starting_workers(false);
        
        // If another thread is starting workers, wait briefly then return
        if (starting_workers.exchange(true)) {
            lock.unlock();
            MCTS_DEBUG("Another thread is already starting workers, skipping");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            starting_workers.store(false);
            return;
        }
        
        // Guard to ensure starting_workers is reset on function exit
        struct StartingWorkersGuard {
            StartingWorkersGuard() {}
            ~StartingWorkersGuard() { starting_workers.store(false); }
        } guard;
        
        // First, ensure all existing threads are properly terminated
        bool had_active_threads = false;
        
        // Set shutdown flag to terminate existing workers
        shutdown_.store(true, std::memory_order_release);
        
        // Notify all workers to check the flag
        queue_cv_.notify_all();
        
        // Release the lock while waiting for threads to exit
        lock.unlock();
        
        // Wait briefly for worker threads to notice shutdown flag
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Detach all threads - use a local copy to avoid race conditions
        std::vector<std::thread> threads_to_detach;
        {
            std::lock_guard<std::mutex> detach_lock(queue_mutex_);
            threads_to_detach = std::move(worker_threads_);
            worker_threads_.clear();
        }
        
        for (auto& thread : threads_to_detach) {
            if (thread.joinable()) {
                thread.detach(); // Detach to avoid blocking
                had_active_threads = true;
            }
        }
        
        if (had_active_threads) {
            MCTS_DEBUG("Detached " << threads_to_detach.size() << " existing worker threads");
            // Wait a bit longer after detaching threads
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // Re-acquire the lock for the rest of the setup
        lock.lock();
        
        // Reset active worker count
        active_workers_.store(0, std::memory_order_release);
        
        // Reset shutdown flag before starting new workers
        shutdown_.store(false, std::memory_order_release);
        
        // Initialize last activity time
        last_activity_time_ = std::chrono::steady_clock::now();
        
        // Start new worker threads - ensure vector is empty first
        worker_threads_.clear();
        worker_threads_.reserve(max_workers_);
        
        MCTS_DEBUG("Starting " << max_workers_ << " worker threads");
        
        for (int i = 0; i < max_workers_; i++) {
            worker_threads_.emplace_back(&LeafGatherer::worker_function, this, i);
        }
        
        // No need to update active_workers_ here, each thread will increment it
    }    
    
    // Restart workers if they appear to be stuck
    void restartWorkers() {
        MCTS_DEBUG("Restarting worker threads");
        
        // Simply call startWorkers which will properly handle cleanup and restart
        startWorkers();
        
        MCTS_DEBUG("Worker threads restarted");
    }

    // Worker thread function with thread ID
    void worker_function(int thread_id) {
        MCTS_DEBUG("LeafGatherer worker " << thread_id << " started");
        
        // Report as active
        active_workers_.fetch_add(1, std::memory_order_relaxed);
        
        // Set up thread-local timeout monitoring
        auto last_activity = std::chrono::steady_clock::now();
        const int ACTIVITY_TIMEOUT_MS = 10000;  // Increased to 10 seconds max inactivity time
        
        // Use a try/catch block to handle all exceptions
        try {
            // Small batch vector for batch processing
            std::vector<LeafEvalRequest> batch;
            batch.reserve(batch_size_);
            
            // Main worker loop
            while (!shutdown_.load(std::memory_order_acquire)) {
                // Periodically check if we've been inactive too long
                auto current_time = std::chrono::steady_clock::now();
                auto inactive_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - last_activity).count();
                    
                if (inactive_time > ACTIVITY_TIMEOUT_MS) {
                    MCTS_DEBUG("Worker " << thread_id << " inactive for " << inactive_time 
                              << "ms, self-terminating");
                    break;  // Break out of worker loop if inactive too long
                }
                
                // Clear batch for this iteration
                batch.clear();
                
                // Collect items for batch
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    
                    // Wait with timeout to check shutdown flag
                    auto wait_result = queue_cv_.wait_for(lock, std::chrono::milliseconds(100), 
                        [this] { 
                            return !request_queue_.empty() || shutdown_.load(std::memory_order_acquire); 
                        });
                    
                    // Check shutdown flag with lock held
                    if (shutdown_.load(std::memory_order_acquire)) {
                        MCTS_DEBUG("Worker " << thread_id << " shutdown detected during wait");
                        break;
                    }
                    
                    // If queue is empty after timeout, continue checking shutdown flag
                    if (request_queue_.empty()) {
                        last_activity = std::chrono::steady_clock::now();  // Update activity time
                        continue;
                    }
                    
                    // Determine optimal batch size based on queue contents
                    int queue_size = request_queue_.size();
                    int optimal_batch = std::min(std::min(queue_size, batch_size_), 256); // Increased max
                    
                    // Always process at least one item
                    optimal_batch = std::max(1, optimal_batch);
                    
                    // Collect batch with timeout monitoring
                    auto collect_start = std::chrono::steady_clock::now();
                    const int MAX_COLLECT_MS = 100;  // Increased time to collect batch
                    
                    for (int i = 0; i < optimal_batch && !request_queue_.empty(); ++i) {
                        batch.push_back(std::move(request_queue_.front()));
                        request_queue_.pop();
                        
                        // Periodically check if collection is taking too long
                        if (i % 32 == 0) {  // Check less frequently
                            auto now = std::chrono::steady_clock::now();
                            auto collect_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                now - collect_start).count();
                            if (collect_ms > MAX_COLLECT_MS) {
                                MCTS_DEBUG("Worker " << thread_id << " collection taking too long (" 
                                          << collect_ms << "ms), breaking early");
                                break;
                            }
                        }
                    }
                    
                    // Update activity time after collection
                    last_activity = std::chrono::steady_clock::now();
                }
                
                // Check shutdown flag before processing
                if (shutdown_.load(std::memory_order_acquire)) {
                    MCTS_DEBUG("Worker " << thread_id << " shutdown detected before processing batch");
                    
                    // Complete all promises with default values
                    for (auto& item : batch) {
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
                    
                    break;
                }
                
                // Process batch directly (no thread spawning)
                if (!batch.empty()) {
                    try {
                        // Start timing
                        auto batch_start = std::chrono::steady_clock::now();
                        
                        // Process the batch directly
                        process_batch(batch, thread_id);
                        
                        // Track timing
                        auto batch_end = std::chrono::steady_clock::now();
                        auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            batch_end - batch_start).count();
                            
                        if (batch.size() > 1) {
                            MCTS_DEBUG("Worker " << thread_id << " processed batch of " 
                                     << batch.size() << " items in " << batch_time << "ms");
                        }
                        
                        // Update activity time after processing
                        last_activity = std::chrono::steady_clock::now();
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("Worker " << thread_id << " exception during process_batch: " << e.what());
                        
                        // Complete all promises with default values
                        for (auto& item : batch) {
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
                                // Ignore errors in default value setting
                            }
                        }
                        
                        // Update activity time even after error
                        last_activity = std::chrono::steady_clock::now();
                    }
                }
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Worker " << thread_id << " terminated with exception: " << e.what());
        }
        catch (...) {
            MCTS_DEBUG("Worker " << thread_id << " terminated with unknown exception");
        }
        
        // Report as inactive before exiting
        active_workers_.fetch_sub(1, std::memory_order_relaxed);
        
        MCTS_DEBUG("LeafGatherer worker " << thread_id << " exiting");
    }
    
    // Process a batch of requests directly without spawning a thread
    void process_batch(const std::vector<LeafEvalRequest>& batch, int thread_id) {
        if (batch.empty()) {
            return;
        }
        
        // Log batch processing
        if (batch.size() > 1) {
            MCTS_DEBUG("Worker " << thread_id << " processing batch of " << batch.size() << " leaves");
        }
        
        // Prepare data for attack/defense module
        std::vector<std::vector<std::vector<int>>> board_batch;
        std::vector<int> chosen_moves;
        std::vector<int> player_batch;
        
        board_batch.reserve(batch.size());
        chosen_moves.reserve(batch.size());
        player_batch.reserve(batch.size());
        
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
                
            if (batch.size() > 4 && attack_defense_duration > 10) {
                MCTS_DEBUG("Worker " << thread_id << " computed attack/defense bonuses in " 
                        << attack_defense_duration << "ms");
            }
        } catch (const std::exception& e) {
            MCTS_DEBUG("Worker " << thread_id << " error computing attack/defense bonuses: " << e.what());
            // Create default values
            attack_vec.resize(batch.size(), 0.0f);
            defense_vec.resize(batch.size(), 0.0f);
        }
        
        // Prepare neural network input
        std::vector<std::tuple<std::string, int, float, float>> nn_inputs;
        nn_inputs.reserve(batch.size());
        
        for (size_t i = 0; i < batch.size(); i++) {
            float attack = (i < attack_vec.size()) ? attack_vec[i] : 0.0f;
            float defense = (i < defense_vec.size()) ? defense_vec[i] : 0.0f;
            
            std::string state_str;
            try {
                state_str = nn_->create_state_string(
                    batch[i].state, chosen_moves[i], attack, defense);
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error creating state string: " << e.what());
                // Create a minimal valid state string
                state_str = "Board:" + std::to_string(batch[i].state.board_size) + 
                        ";Player:" + std::to_string(batch[i].state.current_player) + 
                        ";State:";
            }
                
            nn_inputs.emplace_back(state_str, chosen_moves[i], attack, defense);
        }
        
        // Call neural network for batch inference
        std::vector<NNOutput> results;
        
        try {
            auto nn_start = std::chrono::steady_clock::now();
            
            results = nn_->batch_inference(nn_inputs);
            
            auto nn_end = std::chrono::steady_clock::now();
            auto nn_duration = std::chrono::duration_cast<std::chrono::milliseconds>(nn_end - nn_start).count();
            
            if (batch.size() > 4) {
                MCTS_DEBUG("Worker " << thread_id << " neural network batch inference completed in " << nn_duration 
                        << "ms for " << batch.size() << " leaves");
            }
        } catch (const std::exception& e) {
            MCTS_DEBUG("Worker " << thread_id << " error in neural network batch inference: " << e.what());
            results.clear();
        }
        
        // Update total processed count
        total_processed_.fetch_add(batch.size(), std::memory_order_relaxed);
        
        // Update last activity timestamp
        last_activity_time_ = std::chrono::steady_clock::now();
        
        // Process results and fulfill promises
        for (size_t i = 0; i < batch.size(); i++) {
            try {
                if (!batch[i].result_promise) {
                    continue; // Skip if no promise (shouldn't happen)
                }
                
                auto valid_moves = batch[i].state.get_valid_moves();
                
                if (valid_moves.empty() || batch[i].state.is_terminal()) {
                    // Terminal state or no valid moves
                    float value = 0.0f;
                    int winner = batch[i].state.get_winner();
                    int current_player = batch[i].state.current_player;
                    
                    if (winner == current_player) {
                        value = 1.0f;
                    } else if (winner == 0) {
                        value = 0.0f; // Draw
                    } else {
                        value = -1.0f; // Loss
                    }
                    
                    batch[i].result_promise->set_value({std::vector<float>(), value});
                    continue;
                }
                
                // Check if we have a valid result for this leaf
                std::vector<float> policy;
                float value = 0.0f;
                
                if (i < results.size() && !results[i].policy.empty()) {
                    // Extract policy for valid moves
                    const auto& full_policy = results[i].policy;
                    policy.reserve(valid_moves.size());
                    
                    // Policy might be for all possible moves, so extract just what we need
                    if (full_policy.size() == valid_moves.size()) {
                        // Policy is already aligned with valid moves
                        policy = full_policy;
                    } else {
                        // Extract policy for each valid move
                        for (int move : valid_moves) {
                            if (move >= 0 && move < static_cast<int>(full_policy.size())) {
                                policy.push_back(full_policy[move]);
                            } else {
                                policy.push_back(1.0f / valid_moves.size());
                            }
                        }
                    }
                    
                    // Normalize the policy
                    float sum = std::accumulate(policy.begin(), policy.end(), 0.0f);
                    if (sum > 0) {
                        for (auto& p : policy) {
                            p /= sum;
                        }
                    } else {
                        // Uniform policy if sum is zero
                        for (auto& p : policy) {
                            p = 1.0f / policy.size();
                        }
                    }
                    
                    // Get value
                    value = results[i].value;
                } else {
                    // Use center-biased policy if neural network failed
                    MCTS_DEBUG("Invalid result at index " << i << ", using center-biased policy");
                    
                    policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                    
                    // Apply slight bias toward center for better default play
                    if (policy.size() > 4) {
                        const float CENTER_BIAS = 1.2f;  // 20% boost for center moves
                        int board_size = batch[i].state.board_size;
                        float center_row = (board_size - 1) / 2.0f;
                        float center_col = (board_size - 1) / 2.0f;
                        float max_dist = std::sqrt(center_row * center_row + center_col * center_col);
                        
                        float sum = 0.0f;
                        for (size_t j = 0; j < valid_moves.size(); j++) {
                            int move = valid_moves[j];
                            int row = move / board_size;
                            int col = move % board_size;
                            
                            // Calculate distance from center (normalized to [0,1])
                            float dist = std::sqrt(std::pow(row - center_row, 2) + std::pow(col - center_col, 2));
                            float norm_dist = dist / max_dist;
                            
                            // Closer to center gets higher prior
                            policy[j] *= (1.0f + (CENTER_BIAS - 1.0f) * (1.0f - norm_dist));
                            sum += policy[j];
                        }
                        
                        // Renormalize
                        if (sum > 0) {
                            for (auto& p : policy) {
                                p /= sum;
                            }
                        }
                    }
                    
                    // Use neutral value
                    value = 0.0f;
                }
                
                // Fulfill the promise with policy and value
                batch[i].result_promise->set_value({policy, value});
                
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
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error setting default value: " << e.what());
                }
            }
        }
    }
};
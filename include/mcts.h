// mcts.h
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include <future>  // Added for std::promise/future
#include <random>  // Ensure this is included
#include "mcts_config.h"
#include "nn_interface.h"
#include "python_nn_proxy.h"
#include "leaf_gatherer.h"
#include "node.h"
#include "attack_defense.h"
#include "debug.h"

/**
 * MCTS class that uses multi-threading. 
 * For each leaf, we compute Attack/Defense for the move leading to that leaf,
 * then pass it (along with the gamestate) to the NN.
 */
class MCTS {
public:
    MCTS(const MCTSConfig& config,
        std::shared_ptr<PythonNNProxy> nn,
        int boardSize);
    ~MCTS();

    void run_search(const Gamestate& rootState);
    void run_parallel_search(const Gamestate& rootState);
    int select_move() const;

    float dirichlet_alpha_ = 0.03f;
    float noise_weight_ = 0.25f;

    void set_dirichlet_alpha(float alpha) { dirichlet_alpha_ = alpha; }
    void set_noise_weight(float weight) { noise_weight_ = weight; }

    void add_dirichlet_noise(std::vector<float>& priors);
    int select_move_with_temperature(float temperature = 1.0f) const;

    float get_dynamic_cpuct(int simulations_done, int total_simulations) const;
    float get_optimal_temperature(int move_num, int board_size) const;

    void set_shutdown_flag(bool flag) {
        shutdown_flag_ = flag;
    }
    
    std::shared_ptr<LeafGatherer> get_leaf_gatherer() {
        return std::shared_ptr<LeafGatherer>(leaf_gatherer_.release());
    }
    
    void clear_leaf_gatherer() {
        // Gracefully shutdown leaf gatherer
        if (leaf_gatherer_) {
            try {
                MCTS_DEBUG("MCTS explicitly shutting down leaf gatherer");
                leaf_gatherer_->shutdown();
                MCTS_DEBUG("Leaf gatherer shutdown complete, clearing reference");
                leaf_gatherer_.reset();
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error shutting down leaf gatherer: " << e.what());
                // Force reset anyway
                leaf_gatherer_.reset();
            }
        }
    }

    bool check_and_restart_leaf_gatherer() {
        if (!leaf_gatherer_) {
            MCTS_DEBUG("LeafGatherer not available, creating new one");
            create_or_reset_leaf_gatherer();
            return leaf_gatherer_ != nullptr;
        }
        
        // Check if gatherer appears stalled
        int queue_size = leaf_gatherer_->get_queue_size();
        int active_workers = leaf_gatherer_->get_active_workers();
        
        if (queue_size > 10 && active_workers == 0) {
            MCTS_DEBUG("LeafGatherer appears stuck (queue_size=" << queue_size 
                      << ", active_workers=" << active_workers << "), restarting");
            create_or_reset_leaf_gatherer();
            return leaf_gatherer_ != nullptr;
        }
        
        return true;
    }

    void force_clear() {
        // Set shutdown flag
        shutdown_flag_ = true;
        
        // Clear leaf gatherer first
        if (leaf_gatherer_) {
            try {
                leaf_gatherer_->shutdown();
            } catch (...) {
                // Ignore errors
            }
            leaf_gatherer_.reset();
        }
        
        // Clear root and free all nodes
        root_.reset();
        
        // Clear other resources
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            leaf_queue_ = std::queue<LeafTask>();  // Replace with empty queue
        }
        
        // Reset counters
        leaves_in_flight_ = 0;
        simulations_done_ = 0;
    }    

    void create_or_reset_leaf_gatherer();
    std::string get_leaf_gatherer_stats() const;

private:
    void worker_thread();
    void run_single_simulation(int sim_index);
    void run_thread_simulations(int start_idx, int end_idx);
    Node* select_node(Node* root) const;
    void expand_and_evaluate(Node* leaf);
    void backup(Node* leaf, float value);
    float uct_score(const Node* parent, const Node* child) const;

private:
    MCTSConfig config_;
    std::shared_ptr<PythonNNProxy> nn_;  // Changed from BatchingNNInterface
    std::unique_ptr<Node> root_;
    std::vector<std::thread> threads_;
    std::atomic<int> simulations_done_;

    AttackDefenseModule attackDefense_;
    
    // Add thread pool management
    std::mutex thread_pool_mutex_;
    std::atomic<bool> shutdown_flag_{false};

    static const int MAX_NODES = 5000;
    std::atomic<int> nodes_created_{0};
    bool reached_node_limit() const { return nodes_created_.load() >= MAX_NODES; }

    // New fields for leaf parallelization
    struct LeafTask {
        // Non-owning pointer - doesn't delete the Node
        Node* leaf;
        Gamestate state;
        int chosen_move;
        // Use shared_ptr for promise to ensure proper cleanup
        std::shared_ptr<std::promise<std::pair<std::vector<float>, float>>> result_promise;
    };
    
    std::queue<LeafTask> leaf_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<int> leaves_in_flight_{0};
    
    // New methods for leaf parallelization
    std::future<std::pair<std::vector<float>, float>> queue_leaf_for_evaluation(Node* leaf);
    void leaf_evaluation_thread();
    void process_leaf_batch(std::vector<LeafTask>& batch);
    void search_worker_thread();

    mutable std::mt19937 rng_; // Random number generator

    void force_shutdown();
    void run_simple_search(int num_simulations);

    // Add leaf gatherer for parallel evaluation
    std::unique_ptr<LeafGatherer> leaf_gatherer_;
    
    // Method for semi-parallel search
    void run_semi_parallel_search(int num_simulations);
    bool perform_tree_pruning();
    std::string get_tree_stats() const;
    void analyze_search_result();

    // helper method to MCTS class to create center-biased policies
    std::vector<float> create_center_biased_policy(const std::vector<int>& valid_moves, int board_size) const {
        std::vector<float> policy(valid_moves.size(), 1.0f / valid_moves.size());
        
        // Apply slight bias toward center for better initial play
        if (policy.size() > 4) {
            const float CENTER_BIAS = 1.2f;  // 20% boost for center moves
            float center_row = (board_size - 1) / 2.0f;
            float center_col = (board_size - 1) / 2.0f;
            float max_dist = sqrt(center_row * center_row + center_col * center_col);
            
            float sum = 0.0f;
            for (size_t i = 0; i < valid_moves.size(); i++) {
                int move = valid_moves[i];
                int row = move / board_size;
                int col = move % board_size;
                
                // Calculate distance from center (normalized to [0,1])
                float dist = sqrt(pow(row - center_row, 2) + pow(col - center_col, 2));
                float norm_dist = dist / max_dist;
                
                // Closer to center gets higher prior
                policy[i] *= (1.0f + (CENTER_BIAS - 1.0f) * (1.0f - norm_dist));
                sum += policy[i];
            }
            
            // Renormalize
            if (sum > 0) {
                for (auto& p : policy) {
                    p /= sum;
                }
            }
        }
        
        return policy;
    }

    struct ThreadWatchdog {
        std::atomic<int> last_simulation_count{0};
        std::chrono::steady_clock::time_point last_progress_time;
        int stall_recovery_attempts{0};
        
        ThreadWatchdog() : last_progress_time(std::chrono::steady_clock::now()) {}
        
        // Check if search is stalled and attempt recovery
        bool check_stalled(int current_sim_count, const std::function<void()>& recovery_action) {
            auto current_time = std::chrono::steady_clock::now();
            auto stall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_progress_time).count();
            
            // If we've made progress, update the timestamp
            if (current_sim_count > last_simulation_count) {
                last_progress_time = current_time;
                last_simulation_count = current_sim_count;
                return false;
            }
            
            // Check if we've been stalled for too long - reduced to 3 seconds
            if (stall_ms > 3000) {
                MCTS_DEBUG("Search appears stalled for " << stall_ms << "ms");
                
                // If we've already tried recovery multiple times, consider search failed
                if (stall_recovery_attempts >= 2) {
                    MCTS_DEBUG("Multiple recovery attempts failed, aborting search");
                    return true;
                }
                
                // Try the recovery action
                MCTS_DEBUG("Attempting stall recovery...");
                recovery_action();
                
                // Update recovery attempt counter and last progress time
                stall_recovery_attempts++;
                last_progress_time = std::chrono::steady_clock::now();
                
                // Give the recovery action time to work
                return false;
            }
            
            return false;
        }
    };
    ThreadWatchdog watchdog_;
};
// mcts.h
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include "mcts_config.h"
#include "nn_interface.h"
#include "node.h"
#include "attack_defense.h"

/**
 * MCTS class that uses multi-threading. 
 * For each leaf, we compute Attack/Defense for the move leading to that leaf,
 * then pass it (along with the gamestate) to the NN.
 */
class MCTS {
public:
    MCTS(const MCTSConfig& config,
         std::shared_ptr<BatchingNNInterface> nn,
         int boardSize); // pass board size so we can create AttackDefenseModule

    // Run MCTS from an initial Gamestate
    void run_search(const Gamestate& rootState);

    // New helper method for leaf parallelization
    void run_parallel_search(const Gamestate& rootState);

    // Pick best move from root
    int select_move() const;

    // Add for Dirichlet noise
    float dirichlet_alpha_ = 0.03f; // 0.03 is AlphaZero's value for Go; adjust as needed
    float noise_weight_ = 0.25f;    // 25% noise, 75% prior

    // Add setter methods for exploration parameters
    void set_dirichlet_alpha(float alpha) { dirichlet_alpha_ = alpha; }
    void set_noise_weight(float weight) { noise_weight_ = weight; }

    // Helper for Dirichlet noise
    void add_dirichlet_noise(std::vector<float>& priors);

    // Add temperature-based move selection
    int select_move_with_temperature(float temperature = 1.0f) const;

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
    std::shared_ptr<BatchingNNInterface> nn_;
    std::unique_ptr<Node> root_;
    std::vector<std::thread> threads_;
    std::atomic<int> simulations_done_;

    // We store an AttackDefenseModule that might need boardSize
    AttackDefenseModule attackDefense_;

    static const int MAX_NODES = 5000; // Limit total nodes in the tree
    std::atomic<int> nodes_created_{0};
    bool reached_node_limit() const { return nodes_created_.load() >= MAX_NODES; }

    // Queue for leaf evaluation tasks
    struct LeafTask {
        Node* leaf;
        Gamestate state;
        int chosen_move;
    };
    
    std::queue<LeafTask> leaf_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> search_done_{false};
    
    // Methods for leaf parallelization
    void queue_leaf_for_evaluation(Node* leaf);
    void leaf_evaluation_worker();
    void evaluate_leaf(Node* leaf, const Gamestate& state, int chosen_move);
    void evaluate_leaf_batch(const std::vector<LeafTask>& batch);
    
    // Random number generator for noise
    mutable std::mt19937 rng_;
};

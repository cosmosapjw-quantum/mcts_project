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
         int boardSize);

    void run_search(const Gamestate& rootState);
    void run_parallel_search(const Gamestate& rootState);
    int select_move() const;

    float dirichlet_alpha_ = 0.03f;
    float noise_weight_ = 0.25f;

    void set_dirichlet_alpha(float alpha) { dirichlet_alpha_ = alpha; }
    void set_noise_weight(float weight) { noise_weight_ = weight; }

    void add_dirichlet_noise(std::vector<float>& priors);
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

    AttackDefenseModule attackDefense_;

    static const int MAX_NODES = 5000;
    std::atomic<int> nodes_created_{0};
    bool reached_node_limit() const { return nodes_created_.load() >= MAX_NODES; }

    struct LeafTask {
        Node* leaf;
        Gamestate state;
        int chosen_move;
    };
    
    std::queue<LeafTask> leaf_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> search_done_{false};
    
    void queue_leaf_for_evaluation(Node* leaf);
    void leaf_evaluation_worker();
    void evaluate_leaf(Node* leaf, const Gamestate& state, int chosen_move);
    void evaluate_leaf_batch(const std::vector<LeafTask>& batch);
    
    mutable std::mt19937 rng_;
};
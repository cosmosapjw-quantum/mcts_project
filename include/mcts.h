// mcts.h
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <thread>
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

    // Pick best move from root
    int select_move() const;

private:
    void worker_thread();
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
};

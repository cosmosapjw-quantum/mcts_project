#pragma once

/**
 * Basic struct holding MCTS parameters.
 */
struct MCTSConfig {
    int num_simulations;          // total MCTS rollouts
    float c_puct;                 // exploration constant
    int parallel_leaf_batch_size; // not always used, but you can keep it
    int num_threads;              // how many CPU threads to run for MCTS
};

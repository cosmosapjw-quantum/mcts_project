// mcts.cpp
#include "mcts.h"
#include <cmath>
#include <algorithm>

MCTS::MCTS(const MCTSConfig& config,
           std::shared_ptr<BatchingNNInterface> nn,
           int boardSize)
 : config_(config),
   nn_(nn),
   simulations_done_(0),
   // create your AttackDefenseModule
   attackDefense_(boardSize)
{}

void MCTS::run_search(const Gamestate& rootState) {
    // Create a new root node
    root_ = std::make_unique<Node>(rootState);
    simulations_done_ = 0;

    threads_.clear();
    threads_.reserve(config_.num_threads);

    // Spawn threads
    for (int i = 0; i < config_.num_threads; i++) {
        threads_.emplace_back(&MCTS::worker_thread, this);
    }
    for (auto& t : threads_) {
        t.join();
    }
}

void MCTS::worker_thread() {
    while(true) {
        int simCount = simulations_done_.fetch_add(1);
        if (simCount >= config_.num_simulations) {
            break;
        }
        // selection
        Node* leaf = select_node(root_.get());
        // expand & evaluate
        expand_and_evaluate(leaf);
    }
}

int MCTS::select_move() const {
    auto children = root_->get_children();
    if (children.empty()) return -1;
    int bestMove = -1;
    int maxVisits = -1;
    for (auto c : children) {
        int vc = c->get_visit_count();
        if (vc > maxVisits) {
            maxVisits = vc;
            bestMove = c->get_move_from_parent();
        }
    }
    return bestMove;
}

Node* MCTS::select_node(Node* root) const {
    Node* current = root;
    while (!current->is_leaf() && !current->get_state().is_terminal()) {
        auto kids = current->get_children();
        float bestVal = -1e9;
        Node* bestChild = nullptr;
        for (auto c : kids) {
            float sc = uct_score(current, c);
            if (sc > bestVal) {
                bestVal = sc;
                bestChild = c;
            }
        }
        current = bestChild;
    }
    return current;
}

/**
 * 1) If leaf is terminal => backup final result
 * 2) Else compute AttackDefense for the "chosen move" (the move from the parent)
 * 3) Call NN => policy + value
 * 4) Expand
 * 5) Backup
 */
void MCTS::expand_and_evaluate(Node* leaf) {
    Gamestate st = leaf->get_state();
    if (st.is_terminal()) {
        float r = 0.f;
        int winner = st.get_winner();
        if (winner == st.current_player) {
            r = 1.f;
        } else if (winner == 0) {
            r = 0.f; // draw or no winner
        } else {
            r = -1.f;
        }
        backup(leaf, r);
        return;
    }

    // The move used to arrive at this leaf:
    int chosenMove = leaf->get_move_from_parent();
    if (chosenMove < 0) {
        // means we are at root. Let's define a dummy chosenMove=0
        chosenMove = 0;
    }

    // Attack/Defense: 
    // We'll do a single-board call, single move. 
    // If your real code is batch-based, adapt as needed.
    std::vector<std::vector<int>> board2D = st.get_board(); 
    std::vector<std::vector<std::vector<int>>> board_batch;
    board_batch.push_back(board2D); // length=1
    
    std::vector<int> chosen_moves;
    chosen_moves.push_back(chosenMove); // length=1
    
    std::vector<int> player_batch;
    player_batch.push_back(st.current_player); // length=1
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(board_batch, chosen_moves, player_batch);

    float attack = attackVec[0];
    float defense = defenseVec[0];

    // Call NN
    std::vector<float> policy;
    float value = 0.f;
    nn_->request_inference(st, chosenMove, attack, defense, policy, value);

    // Expand
    auto validMoves = st.get_valid_moves();
    // Suppose policy.size() == validMoves.size()
    leaf->expand(validMoves, policy);

    // Backup
    backup(leaf, value);
}

void MCTS::backup(Node* leaf, float value) {
    Node* current = leaf;
    // Usually, we say "leafPlayer" = the player who made the move to get this node, 
    // or "leaf->get_state().current_player" if you prefer. 
    // This can vary based on how you define the sign of value. 
    // We'll keep it simple:
    int leafPlayer = leaf->get_state().current_player;
    while (current) {
        int nodePlayer = current->get_state().current_player;
        float toAdd = (nodePlayer == leafPlayer) ? value : -value;
        current->update_stats(toAdd);
        current = current->get_parent();
    }
}

float MCTS::uct_score(const Node* parent, const Node* child) const {
    float Q = child->get_q_value();
    float P = child->get_prior();
    int parentVisits = parent->get_visit_count();
    int childVisits  = child->get_visit_count();
    float c = config_.c_puct;
    float U = c * P * std::sqrt((float)parentVisits + 1e-8f) / (1 + childVisits);
    return Q + U;
}

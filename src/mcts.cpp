// mcts.cpp
#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <numeric>

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

    // Single-threaded version
    for (int i = 0; i < config_.num_simulations; i++) {
        try {
            // Select
            Node* leaf = select_node(root_.get());
            if (!leaf) continue;
            
            // Expand and evaluate
            expand_and_evaluate(leaf);
        } catch (const std::exception& e) {
            std::cerr << "Error in simulation " << i << ": " << e.what() << std::endl;
        }
    }
    
    // Process any remaining batched requests
    nn_->flush_batch();
}

void MCTS::worker_thread() {
    while(true) {
        int simCount = simulations_done_.fetch_add(1, std::memory_order_acq_rel);
        if (simCount >= config_.num_simulations) {
            break;
        }
        
        try {
            // selection
            Node* leaf = select_node(root_.get());
            if (leaf == nullptr) {
                std::cerr << "Error: select_node returned nullptr" << std::endl;
                continue;
            }
            
            // expand & evaluate
            expand_and_evaluate(leaf);
        } catch (const std::exception& e) {
            std::cerr << "Exception in worker thread: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in worker thread" << std::endl;
        }
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
    if (!root) {
        std::cerr << "Error: select_node called with null root" << std::endl;
        return nullptr;
    }
    
    Node* current = root;
    while (!current->is_leaf() && !current->get_state().is_terminal()) {
        auto kids = current->get_children();
        if (kids.empty()) {
            break;  // No children, return current node
        }
        
        float bestVal = -1e9;
        Node* bestChild = nullptr;
        for (auto c : kids) {
            if (!c) continue;  // Skip null children
            
            float sc = uct_score(current, c);
            if (sc > bestVal) {
                bestVal = sc;
                bestChild = c;
            }
        }
        
        if (!bestChild) {
            break;  // No valid child found
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
        // means we are at root. Pick a default move
        std::vector<int> valid_moves = st.get_valid_moves();
        if (!valid_moves.empty()) {
            chosenMove = valid_moves[0]; // Use first valid move as default
        } else {
            chosenMove = 0; // Default if no valid moves (shouldn't happen)
        }
    }

    // Attack/Defense calculation
    std::vector<std::vector<int>> board2D = st.get_board(); 
    std::vector<std::vector<std::vector<int>>> board_batch;
    board_batch.push_back(board2D); // length=1
    
    std::vector<int> chosen_moves;
    chosen_moves.push_back(chosenMove); // length=1
    
    std::vector<int> player_batch;
    player_batch.push_back(st.current_player); // length=1
    
    // Get attack and defense bonuses from the module
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
        board_batch, chosen_moves, player_batch);

    float attack = attackVec[0];
    float defense = defenseVec[0];
    
    // Print the attack-defense values for debugging
    std::cerr << "Move: " << chosenMove << " Attack: " << attack 
              << " Defense: " << defense << std::endl;

    // Call NN
    std::vector<float> policy;
    float value = 0.f;
    nn_->request_inference(st, chosenMove, attack, defense, policy, value);

    // Expand
    auto validMoves = st.get_valid_moves();
    std::vector<float> validPolicies;
    validPolicies.reserve(validMoves.size());
    
    for (int move : validMoves) {
        if (move >= 0 && move < policy.size()) {
            // Apply attack-defense heuristics to policy
            float moveAttackBonus = 0.0f;
            float moveDefenseBonus = 0.0f;
            
            // For each valid move, we could compute its attack-defense bonus
            // and incorporate it into the policy, but that would require many
            // additional calls to the attack-defense module. For simplicity, 
            // we'll just use the raw policy for now and rely on the neural
            // network to learn the attack-defense patterns.
            
            validPolicies.push_back(policy[move]);
        } else {
            validPolicies.push_back(1.0f / validMoves.size());  // Default for out-of-bounds
        }
    }
    
    // Renormalize
    float sum = std::accumulate(validPolicies.begin(), validPolicies.end(), 0.0f);
    if (sum > 0) {
        for (auto& p : validPolicies) {
            p /= sum;
        }
    }
    
    leaf->expand(validMoves, validPolicies);

    // Backup
    backup(leaf, value);
}

void MCTS::backup(Node* leaf, float value) {
    if (!leaf) {
        std::cerr << "Error: backup called with null leaf" << std::endl;
        return;
    }
    
    Node* current = leaf;
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

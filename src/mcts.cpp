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

int MCTS::select_move_with_temperature(float temperature) const {
    if (!root_) {
        std::cerr << "MCTS: Root is null in select_move_with_temperature" << std::endl;
        return -1;
    }
    
    std::vector<Node*> children = root_->get_children();
    if (children.empty()) {
        std::cerr << "MCTS: No children in select_move_with_temperature" << std::endl;
        return -1;
    }
    
    // Compute visit count distribution
    std::vector<float> distribution;
    std::vector<int> moves;
    
    for (Node* child : children) {
        if (!child) continue;
        
        moves.push_back(child->get_move_from_parent());
        
        // Get visit count raised to 1/temperature
        float count = static_cast<float>(child->get_visit_count());
        if (temperature > 0) {
            distribution.push_back(std::pow(count, 1.0f / temperature));
        } else {
            // If temperature is 0 or negative, use max visits only
            distribution.push_back(count);
        }
    }
    
    // Temperature = 0 means deterministic choice (highest visits)
    if (temperature <= 0) {
        int best_idx = std::distance(distribution.begin(), 
                                   std::max_element(distribution.begin(), distribution.end()));
        return moves[best_idx];
    }
    
    // Normalize distribution
    float sum = std::accumulate(distribution.begin(), distribution.end(), 0.0f);
    if (sum > 0) {
        for (float& d : distribution) {
            d /= sum;
        }
    } else {
        // If all zeros, use uniform distribution
        for (float& d : distribution) {
            d = 1.0f / distribution.size();
        }
    }
    
    // Sample from distribution
    std::discrete_distribution<int> dist(distribution.begin(), distribution.end());
    int selected_idx = dist(rng_); // Ensure rng_ is not const
    
    return moves[selected_idx];
}

void MCTS::add_dirichlet_noise(std::vector<float>& priors) {
    if (priors.empty()) return;
    
    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma_dist(dirichlet_alpha_, 1.0f);
    std::vector<float> noise(priors.size());
    float noise_sum = 0.0f;
    
    for (size_t i = 0; i < priors.size(); i++) {
        noise[i] = gamma_dist(rng_);
        noise_sum += noise[i];
    }
    
    // Normalize noise
    if (noise_sum > 0) {
        for (float& n : noise) {
            n /= noise_sum;
        }
    }
    
    // Mix noise with prior
    for (size_t i = 0; i < priors.size(); i++) {
        priors[i] = (1.0f - noise_weight_) * priors[i] + noise_weight_ * noise[i];
    }
    
    // Renormalize the mixed prior
    float sum = std::accumulate(priors.begin(), priors.end(), 0.0f);
    if (sum > 0) {
        for (float& p : priors) {
            p /= sum;
        }
    }
}

void MCTS::run_search(const Gamestate& rootState) {
    // Clean up previous search tree completely
    threads_.clear();
    root_.reset();
    
    // Choose approach based on threads
    if (config_.num_threads > 1) {
        // Use leaf parallelization approach
        run_parallel_search(rootState);
    } else {
        // Single-threaded version
        root_ = std::make_unique<Node>(rootState);
        simulations_done_ = 0;
        
        std::cerr << "MCTS: Starting single-threaded search with " 
                  << config_.num_simulations << " simulations" << std::endl;
        
        for (int i = 0; i < config_.num_simulations; i++) {
            try {
                Node* leaf = select_node(root_.get());
                if (!leaf) continue;
                expand_and_evaluate(leaf);
                
                // Report progress periodically
                if (i % 20 == 0) {
                    std::cerr << "MCTS: Completed " << i << " simulations" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "MCTS: Error in simulation " << i << ": " << e.what() << std::endl;
            }
        }
        
        std::cerr << "MCTS: Single-threaded search completed" << std::endl;
    }
}

void MCTS::run_parallel_search(const Gamestate& rootState) {
    // Reset state
    root_ = std::make_unique<Node>(rootState);
    simulations_done_ = 0;
    search_done_ = false;

    // After creating the root node, expand it first
    auto validMoves = rootState.get_valid_moves();
    if (!validMoves.empty()) {
        // Create uniform priors initially
        std::vector<float> priors(validMoves.size(), 1.0f / validMoves.size());
        
        // Add Dirichlet noise to root node priors for exploration
        add_dirichlet_noise(priors);
        
        // Expand root with noisy priors
        root_->expand(validMoves, priors);
    }
    
    // Get parallelism parameters
    int num_threads = config_.num_threads;
    int num_simulations = config_.num_simulations;
    int leaf_batch_size = config_.parallel_leaf_batch_size > 0 ? 
                          config_.parallel_leaf_batch_size : 4;
    
    std::cerr << "MCTS: Starting parallel search with " << num_simulations 
              << " simulations using " << num_threads << " threads" << std::endl;
    
    // Clear leaf queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!leaf_queue_.empty()) {
            leaf_queue_.pop();
        }
    }
    
    // Start worker threads for leaf evaluation
    threads_.clear();
    for (int i = 0; i < num_threads; i++) {
        threads_.emplace_back(&MCTS::leaf_evaluation_worker, this);
    }
    
    // Track search progress
    int queued_leaves = 0;
    int nodes_expanded = 0;
    std::vector<Node*> leaf_buffer;
    leaf_buffer.reserve(leaf_batch_size * 2);
    
    // Main thread does the tree traversal (selection phase)
    auto start_time = std::chrono::steady_clock::now();
    
    for (int i = 0; i < num_simulations; i++) {
        try {
            // Select a leaf node
            Node* leaf = select_node(root_.get());
            
            if (!leaf) {
                continue;
            }
            
            // Handle terminal states directly in main thread
            if (leaf->get_state().is_terminal()) {
                float value = 0.0f;
                int winner = leaf->get_state().get_winner();
                int current_player = leaf->get_state().current_player;
                
                if (winner == current_player) {
                    value = 1.0f;
                } else if (winner == 0) {
                    value = 0.0f; // Draw
                } else {
                    value = -1.0f;
                }
                
                // Update stats
                backup(leaf, value);
                nodes_expanded++;
            } else {
                // Queue non-terminal leaves for parallel evaluation
                leaf_buffer.push_back(leaf);
                
                // Process leaf buffer when it reaches batch size
                if (leaf_buffer.size() >= leaf_batch_size) {
                    for (Node* l : leaf_buffer) {
                        queue_leaf_for_evaluation(l);
                        queued_leaves++;
                    }
                    leaf_buffer.clear();
                }
            }
            
            // Periodically report search progress
            if (i > 0 && i % 50 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                
                std::cerr << "MCTS: Completed " << i << "/" << num_simulations 
                          << " simulations, " << nodes_expanded << " nodes expanded, "
                          << queued_leaves << " leaves queued in " 
                          << elapsed << "ms" << std::endl;
            }
            
            // Avoid getting too far ahead of the worker threads
            if (i > 0 && i % 100 == 0) {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (leaf_queue_.size() > leaf_batch_size * num_threads * 2) {
                    // Too many pending leaves, wait for workers to catch up
                    std::cerr << "MCTS: Main thread waiting for workers to catch up..." << std::endl;
                    queue_cv_.wait(lock, [this, leaf_batch_size, num_threads]() {
                        return leaf_queue_.size() <= leaf_batch_size * num_threads;
                    });
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "MCTS: Error in simulation " << i << ": " << e.what() << std::endl;
        }
    }
    
    // Queue any remaining leaves
    for (Node* l : leaf_buffer) {
        queue_leaf_for_evaluation(l);
        queued_leaves++;
    }
    
    // Wait for all queued leaves to be evaluated
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() { return leaf_queue_.empty(); });
    }
    
    // Signal workers to exit
    search_done_ = true;
    queue_cv_.notify_all();
    
    // Wait for workers to finish
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // Report final statistics
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cerr << "MCTS: Parallel search completed in " << total_time << "ms" << std::endl;
    std::cerr << "MCTS: " << nodes_expanded << " nodes expanded directly, " 
              << queued_leaves << " leaves processed in parallel" << std::endl;
}

void MCTS::queue_leaf_for_evaluation(Node* leaf) {
    if (!leaf) return;
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Create a task with a copy of the state
    LeafTask task;
    task.leaf = leaf;
    task.state = leaf->get_state().copy(); // Safe copy
    task.chosen_move = leaf->get_move_from_parent();
    
    // Add to queue
    leaf_queue_.push(task);
    
    // Notify a worker
    queue_cv_.notify_one();
}

void MCTS::leaf_evaluation_worker() {
    std::cerr << "WORKER: Started leaf evaluation thread" << std::endl;
    
    // Create local batching structures
    const int local_batch_size = 4;  // Process 4 leaves at a time
    std::vector<LeafTask> batch;
    batch.reserve(local_batch_size);
    
    while (!search_done_) {
        // Clear batch
        batch.clear();
        
        // Get tasks from the queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for at least one task or search completion
            queue_cv_.wait(lock, [this]() { 
                return !leaf_queue_.empty() || search_done_; 
            });
            
            // Check if we should exit
            if (search_done_ && leaf_queue_.empty()) {
                break;
            }
            
            // Get up to batch_size tasks from the queue
            while (!leaf_queue_.empty() && batch.size() < local_batch_size) {
                batch.push_back(leaf_queue_.front());
                leaf_queue_.pop();
            }
        }
        
        // Process the batch outside the lock
        if (!batch.empty()) {
            try {
                // Process the entire batch at once
                evaluate_leaf_batch(batch);
            } catch (const std::exception& e) {
                std::cerr << "WORKER: Error evaluating batch: " << e.what() << std::endl;
            }
            
            // Notify that we've finished the batch
            queue_cv_.notify_all();
        }
    }
    
    std::cerr << "WORKER: Leaf evaluation thread finished" << std::endl;
}

void MCTS::evaluate_leaf(Node* leaf, const Gamestate& state, int chosen_move) {
    // Skip if null leaf
    if (!leaf) return;
    
    // Create a vector of nodes to remove virtual loss from
    std::vector<Node*> path;
    Node* current = leaf;
    while (current) {
        path.push_back(current);
        current = current->get_parent();
    }
    
    // Skip if terminal
    if (state.is_terminal()) {
        // Remove virtual loss from path
        for (Node* node : path) {
            node->remove_virtual_loss();
        }
        return;
    }
    
    // Get attack/defense bonuses
    std::vector<std::vector<int>> board2D = state.get_board(); 
    std::vector<std::vector<std::vector<int>>> board_batch;
    board_batch.push_back(board2D);
    
    std::vector<int> chosen_moves = {chosen_move};
    std::vector<int> player_batch = {state.current_player};
    
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    float attack = attackVec[0];
    float defense = defenseVec[0];
    
    // Get policy and value from NN
    std::vector<float> policy;
    float value = 0.0f;
    nn_->request_inference(state, chosen_move, attack, defense, policy, value);
    
    // Get valid moves
    auto validMoves = state.get_valid_moves();
    std::vector<float> validPolicies;
    validPolicies.reserve(validMoves.size());
    
    // Extract policy values for valid moves
    for (int move : validMoves) {
        if (move >= 0 && move < static_cast<int>(policy.size())) {
            validPolicies.push_back(policy[move]);
        } else {
            validPolicies.push_back(1.0f / validMoves.size());
        }
    }
    
    // Normalize policy
    float sum = std::accumulate(validPolicies.begin(), validPolicies.end(), 0.0f);
    if (sum > 0) {
        for (auto& p : validPolicies) {
            p /= sum;
        }
    }
    
    // Expand the leaf (this is now thread-safe as each leaf is only expanded once)
    leaf->expand(validMoves, validPolicies);
    
    // Backup the result
    backup(leaf, value);

    // Remove virtual loss from path
    for (Node* node : path) {
        node->remove_virtual_loss();
    }
}

void MCTS::evaluate_leaf_batch(const std::vector<LeafTask>& batch) {
    if (batch.empty()) return;
    
    // Prepare batch data for Attack/Defense module
    std::vector<std::vector<std::vector<int>>> board_batch;
    std::vector<int> chosen_moves;
    std::vector<int> player_batch;
    
    // Create individual vectors to track results
    std::vector<Node*> leaves;
    std::vector<std::vector<float>> policies(batch.size());
    std::vector<float> values(batch.size(), 0.0f);
    
    // Fill batch data
    for (const auto& task : batch) {
        if (!task.leaf) continue;
        
        // Store leaf for later lookup
        leaves.push_back(task.leaf);
        
        // Get board state
        std::vector<std::vector<int>> board2D = task.state.get_board();
        board_batch.push_back(board2D);
        
        // Store move and player
        chosen_moves.push_back(task.chosen_move);
        player_batch.push_back(task.state.current_player);
    }
    
    // Calculate attack/defense bonuses for the entire batch
    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    // Process each leaf with its corresponding attack/defense values
    for (size_t i = 0; i < leaves.size(); i++) {
        Node* leaf = leaves[i];
        const Gamestate& state = batch[i].state;
        int chosen_move = chosen_moves[i];
        float attack = attackVec[i];
        float defense = defenseVec[i];
        
        // Skip terminal states
        if (state.is_terminal()) continue;
        
        // Get policy and value from NN
        std::vector<float> policy;
        float value = 0.0f;
        nn_->request_inference(state, chosen_move, attack, defense, policy, value);
        
        // Store policy and value for this leaf
        policies[i] = policy;
        values[i] = value;
    }
    
    // Now expand each leaf and backup values
    for (size_t i = 0; i < leaves.size(); i++) {
        Node* leaf = leaves[i];
        if (!leaf) continue;
        
        const Gamestate& state = batch[i].state;
        if (state.is_terminal()) continue;
        
        // Get valid moves for this leaf
        auto validMoves = state.get_valid_moves();
        std::vector<float> validPolicies;
        validPolicies.reserve(validMoves.size());
        
        // Extract policy values for valid moves
        const auto& policy = policies[i];
        for (int move : validMoves) {
            if (move >= 0 && move < static_cast<int>(policy.size())) {
                validPolicies.push_back(policy[move]);
            } else {
                validPolicies.push_back(1.0f / validMoves.size());
            }
        }
        
        // Normalize policy
        float sum = std::accumulate(validPolicies.begin(), validPolicies.end(), 0.0f);
        if (sum > 0) {
            for (auto& p : validPolicies) {
                p /= sum;
            }
        }
        
        // Expand the leaf
        leaf->expand(validMoves, validPolicies);
        
        // Backup the result
        backup(leaf, values[i]);
    }
}

void MCTS::worker_thread() {
    std::cerr << "WORKER: Thread " << std::this_thread::get_id() << " started" << std::endl;
    
    // Process simulations until we reach the limit
    while (true) {
        // Check if we've reached node limit
        if (reached_node_limit()) {
            std::cerr << "WORKER: Thread " << std::this_thread::get_id() 
                      << " exiting due to node limit" << std::endl;
            break;
        }
        
        // Check if we've reached the simulation limit
        int current_sim = simulations_done_.load(std::memory_order_acquire);
        if (current_sim >= config_.num_simulations) {
            break;
        }
        
        // Increment the counter to claim this simulation
        current_sim = simulations_done_.fetch_add(1, std::memory_order_acq_rel);
        if (current_sim >= config_.num_simulations) {
            break; // Another thread might have incremented past the limit
        }
        
        try {
            // Get a local copy of the root pointer
            Node* root_ptr = root_.get();
            if (!root_ptr) {
                std::cerr << "WORKER: Root is null" << std::endl;
                break;
            }
            
            // Selection phase
            Node* leaf = select_node(root_ptr);
            if (!leaf) {
                std::cerr << "WORKER: Null leaf for simulation " << current_sim << std::endl;
                continue;
            }
            
            // Expansion phase
            expand_and_evaluate(leaf);
            
            // Report memory usage periodically
            if (current_sim % 5 == 0) {
                std::cerr << "MEMORY: Thread " << std::this_thread::get_id() 
                          << " after sim " << current_sim << ": "
                          << Node::total_nodes_.load() << " nodes" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "WORKER: Error in simulation " << current_sim 
                      << ": " << e.what() << std::endl;
        }
    }
    
    std::cerr << "WORKER: Thread " << std::this_thread::get_id() << " finished" << std::endl;
}

void MCTS::run_single_simulation(int sim_index) {
    try {
        std::cerr << "Running simulation " << sim_index << std::endl;
        
        // Get root
        Node* root_ptr = root_.get();
        if (!root_ptr) return;
        
        // Selection phase
        Node* leaf = select_node(root_ptr);
        if (!leaf) return;
        
        // Expansion and evaluation phase
        expand_and_evaluate(leaf);
        
        std::cerr << "Completed simulation " << sim_index << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in simulation " << sim_index << ": " << e.what() << std::endl;
    }
}

void MCTS::run_thread_simulations(int start_idx, int end_idx) {
    for (int i = start_idx; i < end_idx; i++) {
        if (i >= config_.num_simulations) break;
        run_single_simulation(i);
    }
}

int MCTS::select_move() const {
    if (!root_) {
        std::cerr << "MCTS: Root is null in select_move" << std::endl;
        return -1;
    }
    
    std::vector<Node*> children = root_->get_children();
    if (children.empty()) {
        std::cerr << "MCTS: No children in select_move" << std::endl;
        return -1;
    }
    
    // Find best child by visit count
    int bestMove = -1;
    Node* bestChild = nullptr;
    int maxVisits = -1;
    float bestValue = -2.0f;  // Outside the [-1, 1] range
    
    std::cerr << "MCTS: Move selection statistics:" << std::endl;
    
    // Sort children by visit count (descending)
    std::vector<std::pair<Node*, int>> sorted_children;
    for (Node* c : children) {
        if (!c) continue;
        sorted_children.push_back({c, c->get_visit_count()});
    }
    
    std::sort(sorted_children.begin(), sorted_children.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Print top moves
    const int TOP_MOVES = 5;
    for (size_t i = 0; i < std::min(sorted_children.size(), static_cast<size_t>(TOP_MOVES)); i++) {
        Node* c = sorted_children[i].first;
        int vc = sorted_children[i].second;
        float qval = c->get_q_value();
        int move = c->get_move_from_parent();
        int row = move / root_->get_state().board_size;
        int col = move % root_->get_state().board_size;
        
        std::cerr << "  Move " << row << "," << col 
                  << " (idx " << move << "): visits=" << vc
                  << ", Q=" << qval << std::endl;
        
        // Track best move
        if (vc > maxVisits) {
            maxVisits = vc;
            bestMove = move;
            bestChild = c;
            bestValue = qval;
        }
    }
    
    if (bestChild) {
        std::cerr << "MCTS: Selected move " << bestMove << " with " 
                  << maxVisits << " visits and Q-value " << bestValue << std::endl;
    }
    
    return bestMove;
}

Node* MCTS::select_node(Node* root) const {
    if (!root) {
        std::cerr << "ERROR: select_node called with null root" << std::endl;
        return nullptr;
    }
    
    Node* current = root;
    std::vector<Node*> path; // Track path for virtual loss
    
    while (true) {
        // Add current node to path for virtual loss
        path.push_back(current);
        
        // Check for terminal state first
        if (current->get_state().is_terminal()) {
            break;
        }
        
        // Check if leaf node (no children)
        if (current->is_leaf()) {
            break;
        }
        
        // Get children with proper locking
        std::vector<Node*> kids;
        try {
            kids = current->get_children();
        } catch (const std::exception& e) {
            std::cerr << "SELECT: Exception getting children: " << e.what() << std::endl;
            break;
        }
        
        // If no children, break
        if (kids.empty()) {
            break;
        }
        
        // Find best child according to UCT
        float bestVal = -1e9;
        Node* bestChild = nullptr;
        
        for (Node* c : kids) {
            if (!c) continue;  // Skip null children
            
            float sc;
            try {
                sc = uct_score(current, c);
            } catch (const std::exception& e) {
                continue;
            }
            
            if (sc > bestVal) {
                bestVal = sc;
                bestChild = c;
            }
        }
        
        // If we couldn't find a valid child, break
        if (!bestChild) {
            break;
        }
        
        // Move to the best child
        current = bestChild;
    }
    
    // Apply virtual loss to the entire path
    for (Node* node : path) {
        node->add_virtual_loss();
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
    std::cerr << "MCTS: [" << std::this_thread::get_id() << "] expand_and_evaluate start" << std::endl;
    
    Gamestate st = leaf->get_state();
    
    if (st.is_terminal()) {
        std::cerr << "MCTS: [" << std::this_thread::get_id() << "] Terminal state found" << std::endl;
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
    
    std::cerr << "Move: " << chosenMove << " Attack: " << attack 
              << " Defense: " << defense << std::endl;

    // Call NN
    std::vector<float> policy;
    float value = 0.f;
    
    std::cerr << "MCTS: [" << std::this_thread::get_id() << "] Calling NN interface" << std::endl;
    nn_->request_inference(st, chosenMove, attack, defense, policy, value);
    std::cerr << "MCTS: [" << std::this_thread::get_id() << "] NN interface returned with value " << value << std::endl;

    // Expand
    auto validMoves = st.get_valid_moves();
    std::vector<float> validPolicies;
    validPolicies.reserve(validMoves.size());
    
    for (int move : validMoves) {
        if (move >= 0 && move < static_cast<int>(policy.size())) {
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
    
    std::cerr << "MCTS: [" << std::this_thread::get_id() << "] Expanding node with " << validMoves.size() << " moves" << std::endl;
    leaf->expand(validMoves, validPolicies);
    std::cerr << "MCTS: [" << std::this_thread::get_id() << "] Node expanded" << std::endl;

    // Backup
    backup(leaf, value);
    std::cerr << "MCTS: [" << std::this_thread::get_id() << "] expand_and_evaluate complete" << std::endl;
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

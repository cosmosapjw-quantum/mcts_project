// mcts.cpp
#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

MCTS::MCTS(const MCTSConfig& config,
    std::shared_ptr<PythonNNProxy> nn,
    int boardSize)
: config_(config),
nn_(nn),
simulations_done_(0),
attackDefense_(boardSize),
rng_(std::random_device{}()) // Initialize rng_
{
// We'll create the leaf gatherer on demand in run_search
}

MCTS::~MCTS() {
    MCTS_DEBUG("MCTS destructor called");
    
    // Set shutdown flag immediately to stop any ongoing operations
    shutdown_flag_ = true;
    
    // First handle leaf gatherer shutdown with timeout
    if (leaf_gatherer_) {
        MCTS_DEBUG("Shutting down LeafGatherer");
        
        // Use a separate thread with timeout
        std::atomic<bool> gatherer_shutdown_complete{false};
        std::thread gatherer_thread([this, &gatherer_shutdown_complete]() {
            try {
                leaf_gatherer_->shutdown();
                gatherer_shutdown_complete.store(true, std::memory_order_release);
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error in leaf gatherer shutdown: " << e.what());
            }
        });
        
        // Wait for gatherer shutdown with short timeout
        auto shutdown_start = std::chrono::steady_clock::now();
        const int GATHERER_TIMEOUT_MS = 300; // 300ms max wait
        
        while (!gatherer_shutdown_complete.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - shutdown_start).count();
                
            if (elapsed_ms > GATHERER_TIMEOUT_MS) {
                MCTS_DEBUG("LeafGatherer shutdown timeout after " << elapsed_ms << "ms");
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Detach thread if still running (don't join)
        if (gatherer_thread.joinable()) {
            MCTS_DEBUG("Detaching leaf gatherer shutdown thread");
            gatherer_thread.detach();
        }
        
        // Reset pointer regardless
        leaf_gatherer_.reset();
    }
    
    // Process any pending leaf futures with default values
    MCTS_DEBUG("Processing pending leaves");
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Process any pending leaves with default values
        int cleared = 0;
        while (!leaf_queue_.empty()) {
            auto& task = leaf_queue_.front();
            if (task.result_promise) {
                try {
                    std::vector<float> default_policy;
                    float default_value = 0.0f;
                    
                    // Get valid moves for policy
                    auto valid_moves = task.state.get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                    
                    // Set default value
                    task.result_promise->set_value({default_policy, default_value});
                    cleared++;
                } catch (...) {
                    // Ignore errors in cleanup
                }
            }
            leaf_queue_.pop();
        }
        
        MCTS_DEBUG("Cleared " << cleared << " pending leaves");
    }
    
    // Immediately detach any running worker threads instead of joining
    MCTS_DEBUG("Detaching worker threads");
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.detach();
        }
    }
    threads_.clear();
    
    // Reset the root node with timeout
    MCTS_DEBUG("Clearing root node");
    try {
        // Use a separate thread with timeout for root reset
        std::atomic<bool> root_reset_complete{false};
        std::thread root_thread([this, &root_reset_complete]() {
            try {
                root_.reset();
                root_reset_complete.store(true, std::memory_order_release);
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error resetting root: " << e.what());
            }
        });
        
        // Wait for root reset with short timeout
        auto reset_start = std::chrono::steady_clock::now();
        const int ROOT_TIMEOUT_MS = 200; // 200ms max wait
        
        while (!root_reset_complete.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - reset_start).count();
                
            if (elapsed_ms > ROOT_TIMEOUT_MS) {
                MCTS_DEBUG("Root reset timeout after " << elapsed_ms << "ms");
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Detach thread if still running (don't join)
        if (root_thread.joinable()) {
            MCTS_DEBUG("Detaching root reset thread");
            root_thread.detach();
            
            // Force reset of the unique_ptr to ensure destructor completes
            root_ = nullptr;
        }
    } catch (const std::exception& e) {
        MCTS_DEBUG("Exception during root reset: " << e.what());
        
        // Force reset
        root_ = nullptr;
    }
    
    MCTS_DEBUG("MCTS resources cleared");
}

void MCTS::create_or_reset_leaf_gatherer() {
    MCTS_DEBUG("Creating/resetting LeafGatherer");
    
    try {
        // Get batch size from config, increase for better GPU utilization
        int batch_size = std::max(256, config_.parallel_leaf_batch_size);
        
        // For Ryzen 9 5900X (24 threads), use 4-6 worker threads
        // Fewer workers than before to reduce contention
        int num_workers = 4;
        
        // Adjust based on hardware if possible
        unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads > 0) {
            // Use approximately 1/6 of available threads for leaf gathering
            // This leaves threads for the main search and Python NN inference
            num_workers = std::max(2, static_cast<int>(hw_threads / 6));
            num_workers = std::min(num_workers, 6);  // Cap at 6 workers max
        }
        
        MCTS_DEBUG("LeafGatherer configuration: batch_size=" << batch_size 
                   << ", workers=" << num_workers);
        
        // If leaf gatherer exists, shut it down first
        if (leaf_gatherer_) {
            try {
                MCTS_DEBUG("Shutting down existing LeafGatherer");
                leaf_gatherer_->shutdown();
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error shutting down LeafGatherer: " << e.what());
            }
        }
        
        // Create new LeafGatherer
        leaf_gatherer_ = std::make_unique<LeafGatherer>(
            nn_, attackDefense_, batch_size, num_workers);
            
        MCTS_DEBUG("LeafGatherer created successfully");
    } catch (const std::exception& e) {
        MCTS_DEBUG("Error creating LeafGatherer: " << e.what());
        
        // Fall back to nullptr if creation fails
        leaf_gatherer_.reset();
    }
}

std::string MCTS::get_leaf_gatherer_stats() const {
    if (leaf_gatherer_) {
        return leaf_gatherer_->get_stats();
    }
    return "LeafGatherer not available";
}

int MCTS::select_move_with_temperature(float temperature) const {
    if (!root_) {
        MCTS_DEBUG("select_move_with_temperature called with null root");
        return -1;
    }
    
    std::vector<Node*> children = root_->get_children();
    if (children.empty()) {
        MCTS_DEBUG("Root has no children in select_move_with_temperature");
        return -1;
    }
    
    std::vector<float> distribution;
    std::vector<int> moves;
    
    // Filter out nodes with zero visits to avoid potential issues
    std::vector<std::pair<Node*, int>> valid_children;
    for (Node* child : children) {
        if (!child) continue;
        
        int visits = child->get_visit_count();
        if (visits > 0) { // Only consider nodes with at least one visit
            valid_children.emplace_back(child, visits);
        }
    }
    
    // If no children have visits, fall back to uniform selection
    if (valid_children.empty()) {
        MCTS_DEBUG("No children with visits, using prior-based selection");
        for (Node* child : children) {
            if (!child) continue;
            moves.push_back(child->get_move_from_parent());
            distribution.push_back(child->get_prior());  // Use prior instead of visit count
        }
    } else {
        // Sort children by visit count for logging
        std::sort(valid_children.begin(), valid_children.end(), 
                [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Calculate distribution based on temperature
        for (const auto& [child, visits] : valid_children) {
            moves.push_back(child->get_move_from_parent());
            
            if (temperature <= 0.01f) {
                // With zero temperature, select most visited move deterministically
                distribution.push_back(visits);
            } else if (temperature >= 100.0f) {
                // With very high temperature, use uniform distribution
                distribution.push_back(1.0f);
            } else {
                // Apply temperature scaling to visit counts
                distribution.push_back(std::pow(static_cast<float>(visits), 1.0f / temperature));
            }
        }
        
        // Log distribution for top moves
        if (valid_children.size() >= 2) {
            std::ostringstream oss;
            oss << "Move selection (temp=" << temperature << "): ";
            
            int top_moves = std::min(3, static_cast<int>(valid_children.size()));
            for (int i = 0; i < top_moves; i++) {
                Node* child = valid_children[i].first;
                int visits = valid_children[i].second;
                int move = child->get_move_from_parent();
                int x = move / root_->get_state().board_size;
                int y = move % root_->get_state().board_size;
                
                oss << "(" << x << "," << y << "):" << visits << " ";
            }
            MCTS_DEBUG(oss.str());
        }
    }
    
    // Handle special temperature cases
    if (temperature <= 0.01f && !distribution.empty()) {
        // Deterministic selection of best move
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
        // Fallback to uniform if sum is zero
        for (float& d : distribution) {
            d = 1.0f / distribution.size();
        }
    }
    
    // Sample from distribution
    std::discrete_distribution<int> dist(distribution.begin(), distribution.end());
    int selected_idx = dist(rng_);
    
    // Log the selected move
    if (selected_idx >= 0 && selected_idx < static_cast<int>(moves.size())) {
        int selected_move = moves[selected_idx];
        int x = selected_move / root_->get_state().board_size;
        int y = selected_move % root_->get_state().board_size;
        
        MCTS_DEBUG("Selected move (" << x << "," << y << ") with temperature " << temperature);
    }
    
    return selected_idx >= 0 && selected_idx < static_cast<int>(moves.size()) ? 
           moves[selected_idx] : -1;
}

void MCTS::add_dirichlet_noise(std::vector<float>& priors) {
    if (priors.empty()) return;
    
    std::gamma_distribution<float> gamma_dist(dirichlet_alpha_, 1.0f);
    std::vector<float> noise(priors.size());
    float noise_sum = 0.0f;
    
    for (size_t i = 0; i < priors.size(); i++) {
        noise[i] = gamma_dist(rng_);
        noise_sum += noise[i];
    }
    
    if (noise_sum > 0) {
        for (float& n : noise) {
            n /= noise_sum;
        }
    }
    
    for (size_t i = 0; i < priors.size(); i++) {
        priors[i] = (1.0f - noise_weight_) * priors[i] + noise_weight_ * noise[i];
    }
    
    float sum = std::accumulate(priors.begin(), priors.end(), 0.0f);
    if (sum > 0) {
        for (float& p : priors) {
            p /= sum;
        }
    }
}

/**
 * Get a dynamically adjusted exploration parameter (c_puct) based on search progress.
 * 
 * This method returns a c_puct value that starts high for better exploration
 * early in the search and gradually decreases for better exploitation later.
 * 
 * @param simulations_done Current number of completed simulations
 * @param total_simulations Target number of simulations
 * @return Adjusted c_puct value
 */
float MCTS::get_dynamic_cpuct(int simulations_done, int total_simulations) const {
    // Base c_puct from config
    float base_cpuct = config_.c_puct;
    
    // Ensure simulations_done is in valid range
    simulations_done = std::max(0, std::min(simulations_done, total_simulations));
    
    // Calculate progress ratio (0.0 to 1.0)
    float progress = static_cast<float>(simulations_done) / std::max(1, total_simulations);
    
    // Early stage: use higher c_puct for exploration (up to 20% more)
    // Late stage: use lower c_puct for exploitation (down to 50% less)
    float scaling_factor;
    if (progress < 0.3f) {
        // Early stage: 1.0 -> 1.2
        scaling_factor = 1.0f + 0.2f * (1.0f - progress / 0.3f);
    } else if (progress > 0.7f) {
        // Late stage: 1.0 -> 0.5
        scaling_factor = 1.0f - 0.5f * ((progress - 0.7f) / 0.3f);
    } else {
        // Middle stage: constant base value
        scaling_factor = 1.0f;
    }
    
    // Apply scaling to base value
    float adjusted_cpuct = base_cpuct * scaling_factor;
    
    // Further adjust based on tree size for large trees
    int node_count = Node::total_nodes_.load(std::memory_order_acquire);
    if (node_count > 10000) {
        // Gradually reduce c_puct for very large trees to focus on exploitation
        float tree_scaling = std::max(0.7f, 1.0f - 0.3f * (node_count - 10000) / 90000.0f);
        adjusted_cpuct *= tree_scaling;
    }
    
    return adjusted_cpuct;
}

/**
 * Get optimal temperature for move selection based on game phase and search confidence.
 * 
 * Higher temperature -> more exploration
 * Lower temperature -> more exploitation
 * 
 * @param move_num Current move number
 * @param board_size Size of the board
 * @return Appropriate temperature for move selection
 */
float MCTS::get_optimal_temperature(int move_num, int board_size) const {
    // Default temperature
    float temp = 1.0f;
    
    // Calculate game phase (0.0 to 1.0)
    float max_moves = board_size * board_size;
    float game_progress = std::min(1.0f, move_num / max_moves);
    
    // Early game: high temperature (1.0)
    // Mid game: medium temperature (0.5)
    // Late game: low temperature (0.1)
    if (game_progress < 0.2f) {
        // Early game: encourage exploration
        temp = 1.0f;
    } else if (game_progress < 0.7f) {
        // Mid game: gradual decrease
        float mid_progress = (game_progress - 0.2f) / 0.5f;  // 0.0 to 1.0 in mid game
        temp = 1.0f - 0.9f * mid_progress;  // 1.0 down to 0.1
    } else {
        // Late game: exploit best moves
        temp = 0.1f;
    }
    
    // Further adjust based on search confidence
    if (root_) {
        auto children = root_->get_children();
        if (children.size() >= 2) {
            // Sort by visit count
            std::vector<std::pair<Node*, int>> sorted_children;
            for (Node* child : children) {
                if (child) {
                    sorted_children.emplace_back(child, child->get_visit_count());
                }
            }
            
            std::sort(sorted_children.begin(), sorted_children.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            if (sorted_children.size() >= 2) {
                float top_visits = sorted_children[0].second;
                float second_visits = sorted_children[1].second;
                
                // Calculate confidence as ratio between top and second-best
                float confidence = 0.0f;
                if (top_visits + second_visits > 0) {
                    confidence = top_visits / (top_visits + second_visits);
                }
                
                // If high confidence (> 80%), reduce temperature further
                if (confidence > 0.8f) {
                    temp *= 0.5f;  // Cut temperature in half for high confidence
                }
                // If low confidence (< 60%), increase temperature
                else if (confidence < 0.6f) {
                    temp *= 1.5f;  // Increase temperature for low confidence
                }
            }
        }
    }
    
    // Clamp to reasonable range
    temp = std::max(0.1f, std::min(temp, 2.0f));
    
    return temp;
}

void MCTS::force_shutdown() {
    MCTS_DEBUG("FORCING EMERGENCY SHUTDOWN");
    
    // Set shutdown flag
    shutdown_flag_ = true;
    
    // Handle the leaf queue and fulfill all pending promises
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Count remaining leaves
        int remaining = leaf_queue_.size();
        MCTS_DEBUG("Processing " << remaining << " queued leaves");
        
        // Fulfill all promises in the queue with default values before clearing
        int processed = 0;
        while (!leaf_queue_.empty()) {
            try {
                auto& task = leaf_queue_.front();
                if (task.result_promise) {
                    auto valid_moves = task.state.get_valid_moves();
                    std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                    task.result_promise->set_value({default_policy, 0.0f});
                    processed++;
                }
                leaf_queue_.pop();
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error fulfilling promise during shutdown: " << e.what());
                leaf_queue_.pop();
            }
        }
        
        MCTS_DEBUG("Fulfilled " << processed << " promises during shutdown");
        
        // Signal any waiting threads
        queue_cv_.notify_all();
    }
    
    // Reset leaves in flight counter
    int in_flight = leaves_in_flight_.exchange(0, std::memory_order_acq_rel);
    if (in_flight < 0) {
        MCTS_DEBUG("WARNING: Negative leaves in flight count: " << in_flight);
    } else {
        MCTS_DEBUG("Reset " << in_flight << " leaves in flight");
    }
    
    // Detach all threads immediately instead of trying to join
    MCTS_DEBUG("Force detaching " << threads_.size() << " threads");
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.detach();
        }
    }
    threads_.clear();
    
    MCTS_DEBUG("Emergency shutdown complete");
}

void MCTS::run_search(const Gamestate& rootState) {
    MCTS_DEBUG("Starting MCTS search with semi-parallel approach");
    
    // Record start time for performance tracking
    auto search_start_time = std::chrono::steady_clock::now();
    
    // Force cleanup of any previous search
    shutdown_flag_ = true;
    for (auto& t : threads_) {
        if (t.joinable()) {
            try {
                t.join();
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error joining thread: " << e.what());
                t.detach(); // Detach if join fails
            }
        }
    }
    threads_.clear();
    
    // ADDED: Ensure LeafGatherer is properly shutdown before starting a new search
    if (leaf_gatherer_) {
        try {
            MCTS_DEBUG("Shutting down existing LeafGatherer before new search");
            leaf_gatherer_->shutdown();
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error shutting down LeafGatherer: " << e.what());
        }
        leaf_gatherer_.reset();
    }
    
    // Reset search state
    MCTS_DEBUG("Initializing new search with root state");
    root_ = std::make_unique<Node>(rootState);
    simulations_done_ = 0;
    shutdown_flag_ = false;
    leaves_in_flight_ = 0;
    
    MCTS_DEBUG("Initialized search with root state, player: " << rootState.current_player);
    
    // Initialize root with uniform priors
    std::vector<int> validMoves = rootState.get_valid_moves();
    if (validMoves.empty()) {
        MCTS_DEBUG("No valid moves from root state, search aborted");
        return;
    }
    
    MCTS_DEBUG("Root has " << validMoves.size() << " valid moves");
    
    // Use uniform priors
    std::vector<float> uniformPriors(validMoves.size(), 1.0f / validMoves.size());
    
    // Add Dirichlet noise for exploration
    MCTS_DEBUG("Adding Dirichlet noise with alpha=" << dirichlet_alpha_ 
               << ", noise_weight=" << noise_weight_);
    add_dirichlet_noise(uniformPriors);
    
    // Expand root with priors
    try {
        root_->expand(validMoves, uniformPriors);
    } catch (const std::exception& e) {
        MCTS_DEBUG("Error expanding root: " << e.what());
        return;
    }
    
    // Create or check LeafGatherer
    create_or_reset_leaf_gatherer();
    if (!leaf_gatherer_) {
        MCTS_DEBUG("Failed to create LeafGatherer, continuing without it");
    }
    
    // Configure search parameters
    int num_simulations = config_.num_simulations;
    if (num_simulations <= 0) {
        // Set a reasonable default if not specified
        num_simulations = 800; 
        MCTS_DEBUG("Invalid simulation count, using default: " << num_simulations);
    } else {
        MCTS_DEBUG("Running search with " << num_simulations << " simulations");
    }
    
    // Set up performance monitoring
    int tree_size_before = Node::total_nodes_.load();
    
    // Run the actual search with exception protection
    try {
        // ADDED: Use a scoped guard to ensure shutdown flag is always set
        struct ShutdownGuard {
            std::atomic<bool>& flag;
            ShutdownGuard(std::atomic<bool>& f) : flag(f) { 
                flag.store(false, std::memory_order_release); 
            }
            ~ShutdownGuard() { 
                flag.store(true, std::memory_order_release); 
            }
        };
        
        // Create guard to ensure shutdown flag is set on exit
        ShutdownGuard guard(shutdown_flag_);
        
        run_semi_parallel_search(num_simulations);
    } catch (const std::exception& e) {
        MCTS_DEBUG("Error during search: " << e.what());
    }
    
    // ADDED: Ensure shutdown flag is set after search
    shutdown_flag_.store(true, std::memory_order_release);
    
    // ADDED: Ensure LeafGatherer is properly shutdown after search
    if (leaf_gatherer_) {
        try {
            MCTS_DEBUG("Shutting down LeafGatherer after search");
            leaf_gatherer_->shutdown();
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error shutting down LeafGatherer: " << e.what());
        }
        leaf_gatherer_.reset();
    }
    
    // Gather performance statistics
    auto search_end_time = std::chrono::steady_clock::now();
    auto search_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end_time - search_start_time).count();
    
    int simulations_completed = simulations_done_.load();
    int tree_size_after = Node::total_nodes_.load();
    int tree_growth = tree_size_after - tree_size_before;
    
    float simulations_per_second = 0.0f;
    if (search_duration_ms > 0) {
        simulations_per_second = simulations_completed * 1000.0f / search_duration_ms;
    }
    
    // Report search results
    MCTS_DEBUG("MCTS search completed with " << simulations_completed << " simulations "
              << "in " << search_duration_ms << "ms");
    MCTS_DEBUG("Performance: " << simulations_per_second << " simulations/second, "
              << "tree size: " << tree_size_after << " nodes (+" << tree_growth << ")");
    
    // Analyze the search result
    try {
        analyze_search_result();
    } catch (const std::exception& e) {
        MCTS_DEBUG("Error analyzing search result: " << e.what());
    }
    
    try {
        // Only call force_clear if this is the last search
        // (we don't want to destroy the tree if more searches will follow)
        if (simulations_completed >= num_simulations) {
            MCTS_DEBUG("Final cleanup of MCTS resources");
            force_clear();
        }
    } catch (const std::exception& e) {
        MCTS_DEBUG("Error in final cleanup: " << e.what());
    }
}

// Perform periodic tree pruning to conserve memory
bool MCTS::perform_tree_pruning() {
    if (!root_) {
        return false;
    }
    
    // Get current memory usage
    size_t memory_kb = Node::get_memory_usage_kb();
    int num_nodes = Node::total_nodes_.load();
    
    // Only prune if we're using significant memory or have lots of nodes
    if (memory_kb < 50 * 1024 && num_nodes < 50000) {  // Less than 50MB and 50K nodes
        return false;
    }
    
    MCTS_DEBUG("Starting tree pruning, current memory: " << memory_kb / 1024 << "MB, nodes: " << num_nodes);
    
    // Calculate pruning threshold based on memory pressure
    float pruning_threshold = 0.01f;  // Default 1%
    
    if (memory_kb > 500 * 1024) {  // > 500MB
        pruning_threshold = 0.05f;  // More aggressive 5% 
    } else if (memory_kb > 200 * 1024) {  // > 200MB
        pruning_threshold = 0.03f;  // Medium 3%
    }
    
    // Perform pruning
    int pruned = root_->prune_tree(pruning_threshold);
    
    // Get new stats
    size_t new_memory_kb = Node::get_memory_usage_kb();
    int new_num_nodes = Node::total_nodes_.load();
    
    MCTS_DEBUG("Tree pruning complete, pruned " << pruned << " nodes");
    MCTS_DEBUG("Memory after pruning: " << new_memory_kb / 1024 << "MB, nodes: " << new_num_nodes);
    MCTS_DEBUG("Memory reduction: " << (memory_kb - new_memory_kb) / 1024 << "MB");
    
    return pruned > 0;
}

// Add tree statistics analysis
std::string MCTS::get_tree_stats() const {
    if (!root_) {
        return "Tree not initialized";
    }
    
    std::ostringstream oss;
    auto stats = root_->collect_tree_stats();
    
    oss << "Tree statistics:" << std::endl;
    oss << "  Total nodes: " << stats["total_nodes"] << std::endl;
    oss << "  Memory usage: " << stats["memory_kb"] / 1024 << " MB" << std::endl;
    oss << "  Tree depth: " << stats["depth"] << std::endl;
    oss << "  Avg branching factor: " << stats["branching_factor"] << std::endl;
    oss << "  Max visit count: " << stats["max_visits"] << std::endl;
    
    // Get principal variation
    std::vector<int> pv = root_->get_principal_variation();
    
    if (!pv.empty()) {
        oss << "  Principal variation:";
        
        for (int move : pv) {
            int x = move / root_->get_state().board_size;
            int y = move % root_->get_state().board_size;
            oss << " (" << x << "," << y << ")";
        }
        
        oss << std::endl;
    }
    
    return oss.str();
}

// analyze_search_result method to include memory information
void MCTS::analyze_search_result() {
    if (!root_) {
        MCTS_DEBUG("No root node available for analysis");
        return;
    }
    
    auto children = root_->get_children();
    if (children.empty()) {
        MCTS_DEBUG("Root has no children to analyze");
        return;
    }
    
    // Sort children by visit count
    std::vector<std::pair<Node*, int>> sorted_children;
    for (Node* child : children) {
        if (child) {
            sorted_children.emplace_back(child, child->get_visit_count());
        }
    }
    
    std::sort(sorted_children.begin(), sorted_children.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Report top moves
    const int TOP_MOVES_TO_REPORT = 5;
    int moves_to_report = std::min(static_cast<int>(sorted_children.size()), TOP_MOVES_TO_REPORT);
    
    MCTS_DEBUG("Top " << moves_to_report << " moves from root:");
    
    for (int i = 0; i < moves_to_report; i++) {
        Node* child = sorted_children[i].first;
        int visits = sorted_children[i].second;
        float value = child->get_q_value();
        float prior = child->get_prior();
        
        int move = child->get_move_from_parent();
        int x = move / root_->get_state().board_size;
        int y = move % root_->get_state().board_size;

        float denom = std::max(1, simulations_done_.load());
        float posterior = 0.0f;
        if (denom > 0) {
            posterior = float(visits) / denom;
        } else {
            posterior = 0.0f;
        }
        
        MCTS_DEBUG("  Move " << i+1 << ": (" << x << "," << y << "), "
                  << visits << " visits, value=" << value 
                  << ", prior=" << prior
                  << ", posterior=" << posterior);
    }
    
    // Calculate search confidence and stability
    if (sorted_children.size() >= 2) {
        float top_visits = sorted_children[0].second;
        float second_visits = sorted_children[1].second;
        
        float confidence = 0.0f;
        if (top_visits + second_visits > 0) {
            confidence = top_visits / (top_visits + second_visits);
        }
        
        MCTS_DEBUG("Search confidence: " << confidence * 100.0f << "%");
    }
    
    // Log memory usage
    int total_nodes = Node::total_nodes_.load();
    size_t memory_kb = Node::get_memory_usage_kb();
    
    MCTS_DEBUG("Memory usage: " << memory_kb / 1024 << " MB (" << total_nodes << " nodes)");
}

void MCTS::run_semi_parallel_search(int num_simulations) {
    MCTS_DEBUG("Running semi-parallel search for " << num_simulations << " simulations");
    
    // Dynamic exploration parameters
    float base_cpuct = config_.c_puct;
    bool use_dynamic_cpuct = true;  // Enable dynamic exploration parameter
    
    // Store the original c_puct for restoration after search
    float original_cpuct = config_.c_puct;

    // Define search parameters with adaptive timeout based on simulation count
    const auto start_time = std::chrono::steady_clock::now();
    // Increased and adaptive max search time - more simulations get more time
    const int BASE_SEARCH_TIME_MS = 5000;  // 5 seconds base (from 5s fixed)
    const int MAX_SEARCH_TIME_MS = std::min(30000, BASE_SEARCH_TIME_MS + 
                                           num_simulations * 50);  // 50ms per simulation
    
    MCTS_DEBUG("Using adaptive max search time of " << MAX_SEARCH_TIME_MS << "ms");
    
    // Track search statistics
    std::atomic<int> batch_count{0};
    std::atomic<int> leaf_count{0};
    std::atomic<int> eval_failures{0};
    std::atomic<int> eval_timeouts{0};
    
    // Track health of search
    auto last_progress_time = std::chrono::steady_clock::now();
    int last_progress_count = 0;
    int stall_recovery_attempts = 0;
    
    // For thread safety, use a thread-safe queue for pending evaluations
    struct PendingEval {
        // Use raw pointer for leaf - this is a non-owning reference
        Node* leaf = nullptr;
        
        // Use shared_ptr for the promise to ensure proper cleanup
        std::shared_ptr<std::promise<std::pair<std::vector<float>, float>>> promise_ptr;
        
        // Store the future separately
        std::future<std::pair<std::vector<float>, float>> future;
        
        // Timestamp for timing
        std::chrono::steady_clock::time_point submit_time;
        
        // Default constructor
        PendingEval() : leaf(nullptr), promise_ptr(nullptr) {
            submit_time = std::chrono::steady_clock::now();
        }
        
        // Constructor with leaf and future
        PendingEval(Node* l, std::future<std::pair<std::vector<float>, float>> f) 
            : leaf(l), promise_ptr(nullptr), future(std::move(f)) {
            submit_time = std::chrono::steady_clock::now();
        }
        
        // Constructor with leaf and promise
        PendingEval(Node* l, std::shared_ptr<std::promise<std::pair<std::vector<float>, float>>> p) 
            : leaf(l), promise_ptr(p) {
            if (p) {
                future = p->get_future();
            }
            submit_time = std::chrono::steady_clock::now();
        }
        
        // Move constructor
        PendingEval(PendingEval&& other) noexcept
            : leaf(other.leaf), 
              promise_ptr(std::move(other.promise_ptr)),
              future(std::move(other.future)),
              submit_time(other.submit_time) {
            // Clear the source's leaf to prevent double cleanup
            other.leaf = nullptr;
        }
        
        // Move assignment
        PendingEval& operator=(PendingEval&& other) noexcept {
            if (this != &other) {
                leaf = other.leaf;
                promise_ptr = std::move(other.promise_ptr);
                future = std::move(other.future);
                submit_time = other.submit_time;
                
                // Clear the source's leaf to prevent double cleanup
                other.leaf = nullptr;
            }
            return *this;
        }
        
        // Disable copy operations
        PendingEval(const PendingEval&) = delete;
        PendingEval& operator=(const PendingEval&) = delete;
        
        // Destructor - just for safety, doesn't need to do anything special
        ~PendingEval() {
            // We're careful not to delete leaf - it's not owned by this struct
        }
    };
    
    // Use a mutex-protected vector instead of raw vector
    std::vector<PendingEval> pending_evals;
    std::mutex pending_mutex;
    
    // For batch processing
    std::vector<std::tuple<std::string, int, float, float>> batch_inputs;
    std::vector<Node*> batch_leaves;
    
    // Use the batch size from the enhanced LeafGatherer or config
    const int MAX_BATCH_SIZE = leaf_gatherer_ ? 
        leaf_gatherer_->get_batch_size() : 
        std::max(8, config_.parallel_leaf_batch_size);
    
    MCTS_DEBUG("Using batch size: " << MAX_BATCH_SIZE);
    
    // Helper function to create a center-biased policy
    auto create_center_biased_policy = [](const std::vector<int>& valid_moves, int board_size) -> std::vector<float> {
        std::vector<float> policy(valid_moves.size(), 1.0f / valid_moves.size());
        
        // Apply slight bias toward center for better initial play
        if (policy.size() > 4) {
            const float CENTER_BIAS = 1.2f;  // 20% boost for center moves
            float center_row = (board_size - 1) / 2.0f;
            float center_col = (board_size - 1) / 2.0f;
            float max_dist = std::sqrt(center_row * center_row + center_col * center_col);
            
            float sum = 0.0f;
            for (size_t i = 0; i < valid_moves.size(); i++) {
                int move = valid_moves[i];
                int row = move / board_size;
                int col = move % board_size;
                
                // Calculate distance from center (normalized to [0,1])
                float dist = std::sqrt(std::pow(row - center_row, 2) + std::pow(col - center_col, 2));
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
    };
    
    // Helper to process completed futures with timeout protection
    auto process_completed_futures = [&]() {
        // Make a local copy of pending evals
        std::vector<PendingEval> local_pending;
        std::vector<size_t> completed_indices;
        std::vector<size_t> timeout_indices;
        
        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            
            if (pending_evals.empty()) {
                return 0;
            }
            
            // Find completed and timed out futures
            for (size_t i = 0; i < pending_evals.size(); i++) {
                auto current_time = std::chrono::steady_clock::now();
                auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - pending_evals[i].submit_time).count();
                    
                auto status = pending_evals[i].future.wait_for(std::chrono::milliseconds(0));
                
                if (status == std::future_status::ready) {
                    completed_indices.push_back(i);
                }
                
                // Use adaptive timeout based on current load
                // - More pending evals = longer timeout
                // - More time since start = longer timeout (avoid timeouts at end of search)
                int dynamic_timeout = 
                    std::min(3000, 1000 + static_cast<int>(pending_evals.size()) * 50);
                
                // Consider a future timed out if waiting over the dynamic timeout
                if (wait_time > dynamic_timeout) {
                    timeout_indices.push_back(i);
                    eval_timeouts++;
                }
            }
            
            // If no completed or timed out futures, return early
            if (completed_indices.empty() && timeout_indices.empty()) {
                return 0;
            }
            
            // Copy all indices (completed and timed out) to process
            std::vector<size_t> all_indices = completed_indices;
            all_indices.insert(all_indices.end(), timeout_indices.begin(), timeout_indices.end());
            
            // Sort indices in descending order to safely erase
            std::sort(all_indices.begin(), all_indices.end(), std::greater<size_t>());
            
            // Extract futures to process (in reverse order to avoid shifting indices)
            for (auto idx : all_indices) {
                local_pending.push_back(std::move(pending_evals[idx]));
                pending_evals.erase(pending_evals.begin() + idx);
            }
        }
        
        // Process copied futures without holding the lock
        int processed_count = 0;
        
        for (auto& eval : local_pending) {
            try {
                Node* leaf = eval.leaf;
                if (!leaf) {
                    continue;
                }
                
                // Measure time in queue
                auto now = std::chrono::steady_clock::now();
                auto queue_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - eval.submit_time).count();
                    
                if (queue_ms > 1000) {
                    MCTS_DEBUG("Leaf evaluation took " << queue_ms << "ms");
                }
                
                // Get valid moves here before future.get() which might throw
                auto valid_moves = leaf->get_state().get_valid_moves();
                
                if (valid_moves.empty()) {
                    MCTS_DEBUG("Leaf has no valid moves, can't expand");
                    // Remove virtual losses
                    Node* current = leaf;
                    while (current) {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    }
                    continue;
                }
                
                // ADDED: Check if the node is already being expanded or fully expanded
                if (leaf->is_being_expanded() || !leaf->is_leaf()) {
                    MCTS_DEBUG("Node is already being expanded or is no longer a leaf, skipping");
                    
                    // Remove virtual losses
                    Node* current = leaf;
                    while (current) {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    }
                    continue;
                }
                
                // ADDED: Try to mark for expansion - skip if another thread is already expanding
                if (!leaf->mark_for_expansion()) {
                    MCTS_DEBUG("Cannot mark node for expansion, another thread is handling it");
                    
                    // Remove virtual losses
                    Node* current = leaf;
                    while (current) {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    }
                    continue;
                }
                
                // Try to get result with timeout protection
                std::vector<float> policy;
                float value = 0.0f;
                bool use_default = false;
                
                // Use a longer final timeout for pending futures
                auto wait_status = eval.future.wait_for(std::chrono::milliseconds(50));
                if (wait_status == std::future_status::ready) {
                    try {
                        // Get the result
                        auto [p, v] = eval.future.get();
                        policy = std::move(p);
                        value = v;
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("Error getting future result: " << e.what());
                        use_default = true;
                        eval_failures++;
                    }
                }
                else {
                    use_default = true;
                    MCTS_DEBUG("Future not ready after final wait, using default values");
                    eval_timeouts++;
                }
                
                // Use center-biased policy for better defaults
                if (use_default || policy.empty() || policy.size() != valid_moves.size()) {
                    // Create center-biased policy instead of completely uniform
                    policy = create_center_biased_policy(valid_moves, leaf->get_state().board_size);
                }
                
                // Double-check node is still a leaf before expanding
                if (!leaf->is_leaf()) {
                    MCTS_DEBUG("Node is no longer a leaf after future result, skipping expansion");
                    leaf->clear_expansion_flag();
                    
                    // Remove virtual losses
                    Node* current = leaf;
                    while (current) {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    }
                    continue;
                }
                
                // Expand the node with policy
                try {
                    leaf->expand(valid_moves, policy);
                    
                    // Backup the value
                    backup(leaf, value);
                    processed_count++;
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error expanding node: " << e.what());
                    
                    // Remove virtual losses on error
                    Node* current = leaf;
                    while (current) {
                        try {
                            current->remove_virtual_loss();
                            current = current->get_parent();
                        } catch (...) {
                            break;
                        }
                    }
                }
                
                // Clear expansion flag
                leaf->clear_expansion_flag();
                
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error processing future: " << e.what());
                eval_failures++;
                
                // Try to remove virtual losses from the leaf's path
                try {
                    if (eval.leaf) {
                        // Clear expansion flag if set
                        eval.leaf->clear_expansion_flag();
                        
                        Node* current = eval.leaf;
                        while (current) {
                            current->remove_virtual_loss();
                            current = current->get_parent();
                        }
                    }
                } catch (...) {
                    // Ignore errors in cleanup
                }
            }
        }
        
        return processed_count;
    };
    
    // Process a batch directly
    auto process_batch = [&]() {
        if (batch_inputs.empty()) {
            return 0;
        }
        
        MCTS_DEBUG("Processing batch of " << batch_inputs.size() << " leaves directly");
        batch_count++;
        
        // Request batch inference with timeout protection
        std::vector<NNOutput> results;
        bool inference_completed = false;
        
        // Use a separate thread with timeout for batch inference
        std::atomic<bool> inference_done{false};
        std::thread inference_thread([&]() {
            try {
                results = nn_->batch_inference(batch_inputs);
                inference_done.store(true, std::memory_order_release);
                inference_completed = true;
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error in batch inference thread: " << e.what());
            }
        });
        
        // Wait for inference with timeout
        const int INFERENCE_TIMEOUT_MS = 500; // 500ms max wait time
        auto inference_start = std::chrono::steady_clock::now();
        
        while (!inference_done.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - inference_start).count();
            
            if (elapsed_ms > INFERENCE_TIMEOUT_MS) {
                MCTS_DEBUG("Batch inference timeout after " << elapsed_ms << "ms");
                break;
            }
            
            // Brief sleep to avoid tight loop
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        // Handle thread cleanup
        if (inference_thread.joinable()) {
            if (inference_done.load(std::memory_order_acquire)) {
                inference_thread.join();
            } else {
                MCTS_DEBUG("Detaching inference thread after timeout");
                inference_thread.detach();
                
                // Create default results if timed out
                if (!inference_completed) {
                    // Create default outputs
                    MCTS_DEBUG("Using default values after inference timeout");
                    results.clear();
                }
            }
        }
        
        // CRITICAL FIX: Make local copies of the batch vectors to prevent use-after-free
        auto local_batch_leaves = batch_leaves;  // Create local copy
        auto local_batch_inputs = batch_inputs;  // Create local copy
        
        // Clear global batches immediately to prevent reuse
        batch_inputs.clear();
        batch_leaves.clear();
        
        // Process results - even if inference failed we'll use default policies
        int processed_count = 0;
        
        for (size_t i = 0; i < local_batch_leaves.size(); i++) {
            Node* leaf = local_batch_leaves[i];
            if (!leaf) continue;
            
            try {
                // Double-check node hasn't changed state
                if (!leaf->is_leaf()) {
                    MCTS_DEBUG("Node at index " << i << " is no longer a leaf, skipping expansion");
                    
                    // Still need to clean up virtual losses
                    leaf->clear_all_virtual_losses();
                    processed_count++;
                    continue;
                }
                
                // Verify node isn't being expanded by another thread
                if (leaf->is_being_expanded()) {
                    MCTS_DEBUG("Node at index " << i << " is already being expanded, skipping");
                    
                    // Clean up virtual losses
                    leaf->clear_all_virtual_losses();
                    processed_count++;
                    continue;
                }
                
                // Try to mark for expansion
                if (!leaf->mark_for_expansion()) {
                    MCTS_DEBUG("Cannot mark node at index " << i << " for expansion, skipping");
                    
                    // Clean up virtual losses
                    leaf->clear_all_virtual_losses();
                    processed_count++;
                    continue;
                }
                
                // Get valid moves with safety check
                auto valid_moves = leaf->get_state().get_valid_moves();
                if (valid_moves.empty()) {
                    MCTS_DEBUG("Node at index " << i << " has no valid moves, skipping");
                    leaf->clear_expansion_flag();
                    leaf->clear_all_virtual_losses();
                    processed_count++;
                    continue;
                }
                
                // Prepare policy and value
                std::vector<float> policy;
                float value = 0.0f;
                
                // Check if we have a result from neural network
                if (i < results.size() && !results[i].policy.empty()) {
                    // Make sure policy size matches valid moves
                    if (results[i].policy.size() == valid_moves.size()) {
                        policy = results[i].policy;
                        value = results[i].value;
                    } else {
                        MCTS_DEBUG("Policy size mismatch at index " << i 
                                 << ": got " << results[i].policy.size() 
                                 << ", expected " << valid_moves.size());
                        
                        // Create center-biased policy
                        policy = create_center_biased_policy(valid_moves, leaf->get_state().board_size);
                    }
                } else {
                    MCTS_DEBUG("Invalid result at index " << i << ", using center-biased policy");
                    policy = create_center_biased_policy(valid_moves, leaf->get_state().board_size);
                }
                
                // Expand the node - double-check it's still a leaf first
                if (!leaf->is_leaf()) {
                    MCTS_DEBUG("Node is no longer a leaf (race condition), skipping expansion");
                    leaf->clear_expansion_flag();
                    leaf->clear_all_virtual_losses();
                    processed_count++;
                    continue;
                }
                
                try {
                    // Expand with policy
                    leaf->expand(valid_moves, policy);
                    
                    // Backup the value
                    backup(leaf, value);
                    processed_count++;
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error expanding node at index " << i << ": " << e.what());
                    
                    // Try best-effort backup anyway
                    try {
                        backup(leaf, value);
                    } catch (...) {
                        // Ignore errors in backup
                    }
                    
                    processed_count++; // Count as processed
                }
                
                // Clear expansion flag regardless of success/failure
                leaf->clear_expansion_flag();
                
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error processing batch result for leaf " << i << ": " << e.what());
                eval_failures++;
                
                // Clean up for safety
                try {
                    if (leaf) {
                        leaf->clear_expansion_flag();
                        leaf->clear_all_virtual_losses();
                    }
                } catch (...) {
                    // Ignore cleanup errors
                }
            }
        }
        
        return processed_count;
    };
    
    // Helper to check for search timeout
    auto check_timeout = [&]() -> bool {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();
            
        if (elapsed_ms > MAX_SEARCH_TIME_MS) {
            MCTS_DEBUG("Search timeout reached after " << elapsed_ms << "ms");
            return true;
        }
        
        return false;
    };
    
    // Helper to check for search stall
    auto check_stalled = [&]() -> bool {
        return watchdog_.check_stalled(simulations_done_.load(), [&]() {
            // Recovery actions
            MCTS_DEBUG("Stall recovery action: restarting leaf gatherer");
            create_or_reset_leaf_gatherer();
            
            // Clear any pending evaluations - they might be stuck
            {
                std::lock_guard<std::mutex> lock(pending_mutex);
                
                int cleared = 0;
                for (auto& eval : pending_evals) {
                    if (eval.leaf) {
                        eval.leaf->clear_all_virtual_losses();
                        cleared++;
                    }
                }
                
                MCTS_DEBUG("Cleared virtual losses for " << cleared << " pending evaluations");
                pending_evals.clear();
            }
            
            // Process any existing batch to prevent stale data
            if (!batch_inputs.empty()) {
                MCTS_DEBUG("Processing potentially stale batch of " << batch_inputs.size() << " items");
                process_batch();
            }
        });
    };
    
    // Initialize progress tracking
    last_progress_count = simulations_done_.load(std::memory_order_relaxed);
    
    // Calculate optimal parallelism based on hardware
    int max_pending = std::max(8, config_.num_threads * 2);
    
    // On high-thread CPUs, allow more parallelism
    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads >= 16) {
        max_pending = std::min(static_cast<int>(hw_threads * 1.5), 32);
        MCTS_DEBUG("Detected high thread count CPU (" << hw_threads 
                   << "), increasing parallel evals to " << max_pending);
    }
    
    int simulations_completed = simulations_done_.load(std::memory_order_relaxed);
    int target_simulations = simulations_completed + num_simulations;

    // Check if we should restart the leaf gatherer
    if (simulations_completed % 50 == 0) {
        check_and_restart_leaf_gatherer();
    }

    // Reserve space for batch processing
    batch_inputs.reserve(MAX_BATCH_SIZE);
    batch_leaves.reserve(MAX_BATCH_SIZE);
    
    MCTS_DEBUG("Starting search with target of " << target_simulations << " simulations");
    
    // Track when to perform memory pruning
    auto last_pruning_time = std::chrono::steady_clock::now();
    const int PRUNING_INTERVAL_MS = 1000;  // Prune at most once per second
    bool pruning_enabled = true;   // Enable or disable pruning

    MCTS_DEBUG("Starting search with dynamic exploration: "
        << "base_cpuct=" << base_cpuct
        << ", target=" << target_simulations << " simulations");

    // Main search loop
    while (simulations_completed < target_simulations) {

        // Update exploration parameter based on search progress
        if (use_dynamic_cpuct) {
            config_.c_puct = get_dynamic_cpuct(simulations_completed, target_simulations);
        }

        // Check for timeout or stall
        if (check_timeout() || check_stalled()) {
            break;
        }
        
        // Process any completed futures
        int processed = process_completed_futures();
        if (processed > 0) {
            simulations_completed += processed;
            simulations_done_.store(simulations_completed, std::memory_order_relaxed);
            
            // Log progress periodically
            if (simulations_completed % 10 == 0 || processed > 5) {
                MCTS_DEBUG("Completed " << simulations_completed << "/" << target_simulations
                        << " simulations, " << eval_failures << " failures, "
                        << eval_timeouts << " timeouts");
                
                // Log leaf gatherer stats if available
                if (leaf_gatherer_) {
                    MCTS_DEBUG(leaf_gatherer_->get_stats());
                }
            }
            
            // If we've reached the target, break early
            if (simulations_completed >= target_simulations) {
                break;
            }
        }
        
        // Periodically check if tree pruning is needed
        if (pruning_enabled) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_since_pruning = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_pruning_time).count();
                
            if (elapsed_since_pruning > PRUNING_INTERVAL_MS) {
                // Perform memory-based pruning if needed
                bool pruned = perform_tree_pruning();
                
                if (pruned) {
                    MCTS_DEBUG("Performed tree pruning during search");
                }
                
                // Update timestamp regardless of whether pruning occurred
                last_pruning_time = std::chrono::steady_clock::now();
            }
        }
        
        // Check if we should process batch now
        bool should_process_batch = false;
        
        // Process batch if full
        if (batch_inputs.size() >= MAX_BATCH_SIZE) {
            should_process_batch = true;
        }
        
        // Process batch if we have some items and too many pending evals
        int pending_size = 0;
        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending_size = pending_evals.size();
        }
        
        if (batch_inputs.size() > 0 && pending_size >= max_pending) {
            should_process_batch = true;
        }
        
        // Process batch if it's been waiting too long
        if (batch_inputs.size() > 0) {
            static auto last_batch_time = std::chrono::steady_clock::now();
            auto current_time = std::chrono::steady_clock::now();
            auto batch_wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_batch_time).count();
                
            if (batch_wait_ms > 50) {  // Process at least every 50ms
                should_process_batch = true;
                last_batch_time = current_time;
            }
        }
        
        if (should_process_batch) {
            int batch_processed = process_batch();
            if (batch_processed > 0) {
                simulations_completed += batch_processed;
                simulations_done_.store(simulations_completed, std::memory_order_relaxed);
            }
        }
        
        // Determine how many new leaves to select
        int slots_available = max_pending - pending_size;
        int batch_slots = MAX_BATCH_SIZE - batch_inputs.size();
        int leaves_to_select = std::min(
            std::min(slots_available, batch_slots),
            target_simulations - simulations_completed
        );
        
        // Don't try to select too many at once
        leaves_to_select = std::min(leaves_to_select, 8);
        
        // If we can't select more leaves right now, do a quick check for completed evals
        if (leaves_to_select <= 0) {
            // Small sleep to avoid tight loop
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }
        
        // Select new leaves
        bool any_selection_successful = false;
        for (int i = 0; i < leaves_to_select; i++) {
            // Safety check - avoid infinite loop if select_node keeps returning nullptr
            if (!any_selection_successful && i > 0) {
                MCTS_DEBUG("Multiple selection failures, taking a short break");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Select a leaf node
            Node* leaf = select_node(root_.get());
            if (!leaf) {
                MCTS_DEBUG("Failed to select leaf, skipping");
                continue;
            }
            
            any_selection_successful = true;
            leaf_count++;
            
            // Track leaf selection to prevent infinite loops
            static std::unordered_map<Node*, int> selection_count;
            int& times_selected = selection_count[leaf];
            times_selected++;

        // If the same leaf is selected multiple times in a row, something is wrong
        if (times_selected > 3) {
            MCTS_DEBUG("WARNING: Same leaf selected " << times_selected << " times, attempting forced expansion");
            
            // ADDED: Only proceed if we can safely mark this node for expansion
            if (!leaf->mark_for_expansion()) {
                MCTS_DEBUG("Node already being expanded by another thread, skipping forced expansion");
                
                // ADDED: We still need to remove virtual losses for this path 
                // to prevent future selection of this node
                Node* current = leaf;
                while (current) {
                    try {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    } catch (...) {
                        break;  // Stop on error
                    }
                }
                
                // Reset selection count to avoid repeated warnings
                selection_count[leaf] = 0;
                
                // CRITICAL: Add an artificial visit to discourage immediate re-selection
                try {
                    leaf->update_stats(0.0f); // Neutral value, just to increment visit count
                } catch(...) {
                    // Ignore errors
                }
                
                // Skip this iteration and try with a new node next time
                continue;
            }
            
            // Force immediate evaluation instead of queuing
            try {
                // CRITICAL FIX: Add a timeout for forced expansion
                bool expansion_success = false;
                
                // Use a separate thread with timeout to handle expansion
                std::atomic<bool> expansion_done{false};
                std::thread expansion_thread([&]() {
                    try {
                        expand_and_evaluate(leaf);
                        expansion_done.store(true, std::memory_order_release);
                        expansion_success = true;
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error in expansion thread: " << e.what());
                    }
                });
                
                // Wait for expansion with a reasonable timeout
                const int EXPANSION_TIMEOUT_MS = 500; // 500ms timeout
                auto start_time = std::chrono::steady_clock::now();
                
                while (!expansion_done.load(std::memory_order_acquire)) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - start_time).count();
                    
                    if (elapsed_ms > EXPANSION_TIMEOUT_MS) {
                        MCTS_DEBUG("Expansion timeout after " << elapsed_ms << "ms");
                        break;
                    }
                    
                    // Brief sleep to avoid tight loop
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                
                // If thread is still running, detach it
                if (expansion_thread.joinable()) {
                    if (expansion_done.load(std::memory_order_acquire)) {
                        expansion_thread.join();
                    } else {
                        MCTS_DEBUG("Detaching expansion thread after timeout");
                        expansion_thread.detach();
                    }
                }
                
                // Only increment simulation counter on success
                if (expansion_success) {
                    simulations_completed++;
                    simulations_done_.store(simulations_completed, std::memory_order_relaxed);
                }
                
                // Reset selection count after expansion attempt
                selection_count[leaf] = 0;
                
                // ADDED: Clear expansion flag regardless of success/failure
                leaf->clear_expansion_flag();
                
                // CRITICAL: If expansion failed or timed out, make sure virtual losses are removed
                if (!expansion_success) {
                    MCTS_DEBUG("Removing virtual losses after failed expansion");
                    leaf->clear_all_virtual_losses();
                    
                    // Add a visit to prevent immediate reselection
                    try {
                        leaf->update_stats(0.0f);
                    } catch (...) {
                        // Ignore errors
                    }
                }
                
                continue;
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error in forced expansion: " << e.what());
                
                // ADDED: Clear expansion flag on error
                leaf->clear_expansion_flag();
                
                // ADDED: Always remove virtual losses on error to prevent repeated selection
                Node* current = leaf;
                while (current) {
                    try {
                        current->remove_virtual_loss();
                        current = current->get_parent();
                    } catch (...) {
                        break;  // Stop on error
                    }
                }
                
                // CRITICAL: Add an artificial visit to discourage immediate re-selection
                try {
                    leaf->update_stats(0.0f); // Neutral value, just to increment visit count
                } catch(...) {
                    // Ignore errors
                }
                
                // Reset selection count to avoid repeated warnings
                selection_count[leaf] = 0;
            }
        }
            
            // Handle terminal nodes immediately
            if (leaf->get_state().is_terminal()) {
                float value = 0.0f;
                int winner = leaf->get_state().get_winner();
                int current_player = leaf->get_state().current_player;
                
                if (winner == current_player) {
                    value = 1.0f;
                } else if (winner == 0) {
                    value = 0.0f;  // Draw
                } else {
                    value = -1.0f;  // Loss
                }
                
                backup(leaf, value);
                simulations_completed++;
                simulations_done_.store(simulations_completed, std::memory_order_relaxed);
                continue;
            }
            
            // Add to batch or use LeafGatherer
            try {
                const Gamestate& state = leaf->get_state();
                int chosen_move = leaf->get_move_from_parent();
                
                if (chosen_move < 0) {
                    auto valid_moves = state.get_valid_moves();
                    if (!valid_moves.empty()) {
                        chosen_move = valid_moves[0];
                    } else {
                        chosen_move = 0;
                    }
                }
                
                // Check if LeafGatherer is operational first
                bool use_leaf_gatherer = leaf_gatherer_ && 
                    leaf_gatherer_->get_active_workers() > 0 &&
                    pending_size < max_pending / 2;  // Only use gatherer if we have capacity
                
                if (use_leaf_gatherer) {
                    // Use LeafGatherer for evaluation
                    try {
                        auto future = leaf_gatherer_->queue_leaf(leaf);
                        
                        // Store in pending evaluations with timeout tracking
                        {
                            std::lock_guard<std::mutex> lock(pending_mutex);
                            pending_evals.emplace_back(leaf, std::move(future));
                        }
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("Error queuing leaf in gatherer: " << e.what() << ", falling back to direct batch");
                        use_leaf_gatherer = false;  // Fall through to direct batch
                    }
                }
                
                // If we're not using the leaf gatherer, add to direct batch
                if (!use_leaf_gatherer) {
                    // Calculate attack/defense for direct batch processing
                    std::vector<std::vector<std::vector<int>>> board_batch = {state.get_board()};
                    std::vector<int> chosen_moves = {chosen_move};
                    std::vector<int> player_batch = {state.current_player};
                    
                    auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
                        board_batch, chosen_moves, player_batch);
                        
                    float attack = attackVec.empty() ? 0.0f : attackVec[0];
                    float defense = defenseVec.empty() ? 0.0f : defenseVec[0];
                    
                    // Create state string
                    std::string state_str = nn_->create_state_string(state, chosen_move, attack, defense);
                    
                    // Add to batch
                    batch_inputs.emplace_back(state_str, chosen_move, attack, defense);
                    batch_leaves.push_back(leaf);
                }
                
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error preparing leaf for evaluation: " << e.what());
                eval_failures++;
                
                // Remove virtual losses
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
            }
        }
    }
    
    // Process any remaining batch
    if (!batch_inputs.empty()) {
        MCTS_DEBUG("Processing remaining batch of " << batch_inputs.size() << " leaves");
        int batch_processed = process_batch();
        if (batch_processed > 0) {
            simulations_completed += batch_processed;
            simulations_done_.store(simulations_completed, std::memory_order_relaxed);
        }
    }
    
    // Wait for pending evaluations with timeout
    auto wait_start = std::chrono::steady_clock::now();
    const int MAX_WAIT_MS = 1000; // 1 second timeout for cleanup (reduced from 2000ms)
    
    MCTS_DEBUG("Waiting for remaining evaluations to complete");
    
    // SAFER APPROACH: Make a local snapshot of pending evaluations for cleanup
    std::vector<Node*> nodes_to_cleanup;
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        // Only collect the leaf nodes for cleanup without touching futures
        for (const auto& eval : pending_evals) {
            if (eval.leaf) {
                nodes_to_cleanup.push_back(eval.leaf);
            }
        }
        
        // Log the counts but don't touch the originals yet
        MCTS_DEBUG("Found " << pending_evals.size() << " pending evaluations with " 
                   << nodes_to_cleanup.size() << " valid leaf nodes");
    }
    
    // First, clean up the leaf nodes without touching the futures
    for (Node* leaf : nodes_to_cleanup) {
        try {
            // Only cleanup nodes that still exist
            if (leaf) {
                // Clear expansion flag if set
                leaf->clear_expansion_flag();
                
                // Clear all virtual losses atomically
                leaf->clear_all_virtual_losses();
                
                // Also clean up parent path
                Node* current = leaf->get_parent();
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
            }
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error cleaning up leaf node: " << e.what());
        }
    }
    
    // Now safely clear the pending evaluations container without accessing futures
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        // Just clear the container - don't try to access the futures
        pending_evals.clear();
        MCTS_DEBUG("Cleared pending evaluations container");
    }
    
    // Don't wait in a loop - just check the elapsed time once
    auto now = std::chrono::steady_clock::now();
    auto wait_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - wait_start).count();
        
    MCTS_DEBUG("Cleanup completed in " << wait_elapsed << "ms");
    
    // Calculate final statistics
    auto search_end = std::chrono::steady_clock::now();
    auto search_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end - start_time).count();
    
    auto leaves_per_sec = leaf_count * 1000.0 / std::max(1, static_cast<int>(search_duration_ms));
    auto sims_per_sec = simulations_completed * 1000.0 / std::max(1, static_cast<int>(search_duration_ms));
    
    MCTS_DEBUG("Semi-parallel search completed with " << simulations_completed << " simulations");
    MCTS_DEBUG("Performance: " << search_duration_ms << "ms total, "
              << leaves_per_sec << " leaves/sec, "
              << sims_per_sec << " simulations/sec, "
              << batch_count << " batches, "
              << eval_failures << " failures, "
              << eval_timeouts << " timeouts");    
}

std::future<std::pair<std::vector<float>, float>> MCTS::queue_leaf_for_evaluation(Node* leaf) {
    // CRITICAL FIX: Use shared_ptr for the promise to ensure proper cleanup
    auto promise_ptr = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
    auto future = promise_ptr->get_future();
    
    // Prepare the task
    LeafTask task;
    task.leaf = leaf;
    
    if (leaf) {
        try {
            task.state = leaf->get_state().copy();
            task.chosen_move = leaf->get_move_from_parent();
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error copying state in queue_leaf: " << e.what());
            
            // Fulfill the promise with default values on error
            try {
                std::vector<float> default_policy;
                auto valid_moves = leaf->get_state().get_valid_moves();
                default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                promise_ptr->set_value({default_policy, 0.0f});
            } catch (...) {
                // Ignore errors in setting default value
            }
            
            return future;
        }
    }
    
    // IMPORTANT: Store shared_ptr to promise, not the promise itself
    task.result_promise = promise_ptr;
    
    // Add to queue with safety checks
    {
        // CRITICAL FIX: Check shutdown flag before adding to queue
        if (shutdown_flag_.load(std::memory_order_acquire)) {
            MCTS_DEBUG("Shutdown flag set, not queueing leaf evaluation");
            
            // Fulfill the promise with default values
            try {
                std::vector<float> default_policy;
                if (leaf) {
                    auto valid_moves = leaf->get_state().get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                }
                promise_ptr->set_value({default_policy, 0.0f});
            } catch (...) {
                // Ignore errors in setting default value
            }
            
            return future;
        }
    
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Check if the queue is too large
        const int MAX_QUEUE_SIZE = 1000;
        if (leaf_queue_.size() >= MAX_QUEUE_SIZE) {
            MCTS_DEBUG("Leaf queue full, not adding new task");
            
            // Fulfill the promise with default values
            try {
                std::vector<float> default_policy;
                if (leaf) {
                    auto valid_moves = leaf->get_state().get_valid_moves();
                    default_policy.resize(valid_moves.size(), 1.0f / valid_moves.size());
                }
                promise_ptr->set_value({default_policy, 0.0f});
            } catch (...) {
                // Ignore errors in setting default value
            }
            
            return future;
        }
        
        // Add to queue and increment counter
        leaf_queue_.push(std::move(task));
        leaves_in_flight_.fetch_add(1, std::memory_order_release);
        
        // Notify one worker thread
        queue_cv_.notify_one();
    }
    
    return future;
}

void MCTS::leaf_evaluation_thread() {
    MCTS_DEBUG("Leaf evaluation thread started");
    
    // Determine batch size - use config if provided, otherwise default to 8
    const int batch_size = config_.parallel_leaf_batch_size > 0 ? 
                          config_.parallel_leaf_batch_size : 8;
    
    MCTS_DEBUG("Using batch size: " << batch_size);
    
    // Track last activity time for stall detection
    auto last_activity_time = std::chrono::steady_clock::now();
    
    while (!shutdown_flag_) {
        // Check termination conditions first - this is crucial
        // If we've reached simulation limit, exit even if queue isn't empty
        if (simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations) {
            MCTS_DEBUG("Simulation limit reached in leaf evaluation thread, exiting");
            break;
        }
        
        // Collect a batch of leaves to evaluate
        std::vector<LeafTask> current_batch;
        current_batch.reserve(batch_size);
        
        // Critical section: get leaves from the queue
        bool got_tasks = false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for leaves or shutdown signal with a short timeout
            auto wait_status = queue_cv_.wait_for(lock, std::chrono::milliseconds(10),
                [this] { 
                    // Check both shutdown flag AND simulation count
                    return !leaf_queue_.empty() || shutdown_flag_ || 
                        simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations; 
                });
            
            // Check termination conditions after wait
            if (shutdown_flag_ || simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations) {
                MCTS_DEBUG("Termination condition detected in leaf evaluation thread after wait");
                break;
            }
            
            // Get leaves up to batch size if available
            int count = 0;
            while (!leaf_queue_.empty() && count < batch_size) {
                current_batch.push_back(std::move(leaf_queue_.front()));
                leaf_queue_.pop();
                count++;
                got_tasks = true;
            }
        }
        
        // Update activity timestamp if we got tasks
        if (got_tasks) {
            last_activity_time = std::chrono::steady_clock::now();
        }
        
        // Check for thread starvation (no activity for too long)
        auto current_time = std::chrono::steady_clock::now();
        auto inactivity_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_activity_time).count();
            
        // If no activity for 5 seconds and search isn't complete, report potential stall
        if (inactivity_duration > 5000 && 
            simulations_done_.load(std::memory_order_acquire) < config_.num_simulations) {
            
            MCTS_DEBUG("WARNING: Leaf evaluation thread inactive for " << inactivity_duration 
                      << "ms - possible stall detected");
            // Reset the timer to avoid spamming logs
            last_activity_time = current_time;
        }
        
        // If no leaves and not shutting down, just loop again with short sleep
        if (current_batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Process the batch
        MCTS_DEBUG("Processing batch of " << current_batch.size() << " leaves using neural network");
        
        try {
            // Prepare data for attack/defense module and neural network
            std::vector<std::vector<std::vector<int>>> board_batch;
            std::vector<int> chosen_moves;
            std::vector<int> player_batch;
            
            for (const auto& task : current_batch) {
                board_batch.push_back(task.state.get_board());
                chosen_moves.push_back(task.chosen_move);
                player_batch.push_back(task.state.current_player);
            }
            
            // Compute attack/defense bonuses
            MCTS_DEBUG("Computing attack/defense bonuses for batch");
            auto [attackVec, defenseVec] = attackDefense_.compute_bonuses(
                board_batch, chosen_moves, player_batch);
            
            // Prepare neural network inputs
            std::vector<std::tuple<std::string, int, float, float>> nn_inputs;
            for (size_t i = 0; i < current_batch.size(); i++) {
                std::string stateStr = nn_->create_state_string(
                    current_batch[i].state, 
                    chosen_moves[i],
                    attackVec[i], 
                    defenseVec[i]);
                
                nn_inputs.emplace_back(stateStr, chosen_moves[i], attackVec[i], defenseVec[i]);
            }
            
            // Call neural network for batch inference
            MCTS_DEBUG("Calling neural network for batch inference");
            std::vector<NNOutput> results;
            bool success = false;
            
            try {
                auto start_time = std::chrono::steady_clock::now();
                
                // Use BatchingNNInterface for batch inference
                results = nn_->batch_inference(nn_inputs);
                
                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                
                MCTS_DEBUG("Neural network batch inference completed in " << duration << "ms");
                success = !results.empty();
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error in neural network batch inference: " << e.what());
                success = false;
            }
            
            // Process results and fulfill promises
            MCTS_DEBUG("Processing neural network results");
            for (size_t i = 0; i < current_batch.size(); i++) {
                try {
                    // Check termination condition again for responsiveness
                    if (shutdown_flag_ || simulations_done_.load(std::memory_order_acquire) >= config_.num_simulations) {
                        MCTS_DEBUG("Termination condition detected during result processing");
                        // Just use defaults for remaining items
                        for (size_t j = i; j < current_batch.size(); j++) {
                            auto valid_moves = current_batch[j].state.get_valid_moves();
                            std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                            // Using -> operator for shared_ptr
                            if (current_batch[j].result_promise) {
                                current_batch[j].result_promise->set_value({default_policy, 0.0f});
                            }
                            leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
                        }
                        break;
                    }
                    
                    if (success && i < results.size()) {
                        // Get valid moves for this state
                        auto valid_moves = current_batch[i].state.get_valid_moves();
                        
                        // Extract policy for valid moves
                        std::vector<float> valid_policy;
                        valid_policy.reserve(valid_moves.size());
                        
                        for (int move : valid_moves) {
                            if (move >= 0 && move < static_cast<int>(results[i].policy.size())) {
                                valid_policy.push_back(results[i].policy[move]);
                            } else {
                                valid_policy.push_back(1.0f / valid_moves.size());
                            }
                        }
                        
                        // Normalize policy
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
                        
                        // Fulfill promise with policy and value - using -> operator for shared_ptr
                        if (current_batch[i].result_promise) {
                            current_batch[i].result_promise->set_value({valid_policy, results[i].value});
                        }
                    } else {
                        // Use default values on error
                        auto valid_moves = current_batch[i].state.get_valid_moves();
                        std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        
                        // Using -> operator for shared_ptr
                        if (current_batch[i].result_promise) {
                            current_batch[i].result_promise->set_value({default_policy, 0.0f});
                        }
                    }
                }
                catch (const std::exception& e) {
                    MCTS_DEBUG("Error processing neural network result for leaf " << i << ": " << e.what());
                    try {
                        // Ensure promise is fulfilled even on error
                        auto valid_moves = current_batch[i].state.get_valid_moves();
                        std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                        
                        // Using -> operator for shared_ptr
                        if (current_batch[i].result_promise) {
                            current_batch[i].result_promise->set_value({default_policy, 0.0f});
                        }
                    }
                    catch (...) {
                        // Promise might already be fulfilled
                    }
                }
                
                // Decrement counter for leaves in flight
                leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error processing batch: " << e.what());
            
            // Handle each leaf with default values on error
            for (auto& task : current_batch) {
                try {
                    auto valid_moves = task.state.get_valid_moves();
                    std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
                    
                    // Using -> operator for shared_ptr
                    if (task.result_promise) {
                        task.result_promise->set_value({default_policy, 0.0f});
                    }
                    leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
                }
                catch (...) {
                    // Promise might already be fulfilled
                }
            }
        }
    }
    
    // Handle any remaining leaves in the queue with default values
    MCTS_DEBUG("Leaf evaluation thread shutting down, handling remaining tasks");
    std::vector<LeafTask> remaining_tasks;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!leaf_queue_.empty()) {
            remaining_tasks.push_back(std::move(leaf_queue_.front()));
            leaf_queue_.pop();
        }
    }
    
    MCTS_DEBUG("Processing " << remaining_tasks.size() << " remaining tasks");
    for (auto& task : remaining_tasks) {
        try {
            auto valid_moves = task.state.get_valid_moves();
            std::vector<float> default_policy(valid_moves.size(), 1.0f / valid_moves.size());
            
            // Using -> operator for shared_ptr
            if (task.result_promise) {
                task.result_promise->set_value({default_policy, 0.0f});
            }
            leaves_in_flight_.fetch_sub(1, std::memory_order_relaxed);
        } catch (...) {
            // Promise might already be fulfilled
        }
    }
    
    MCTS_DEBUG("Leaf evaluation thread exiting");
}

int MCTS::select_move() const {
    if (!root_) {
        MCTS_DEBUG("select_move called with null root");
        return -1;
    }
    
    try {
        // Get optimal temperature for current game state
        int move_count = 0;
        int board_size = root_->get_state().board_size;
        
        // Estimate move count from position
        const Gamestate& state = root_->get_state();
        move_count = state.get_move_count();
        
        // Get optimal temperature
        float temp = get_optimal_temperature(move_count, board_size);
        
        MCTS_DEBUG("Using optimal temperature " << temp 
                  << " for move " << move_count
                  << " (game progress: " << (float)move_count/(board_size*board_size) << ")");
        
        // Select move with appropriate temperature
        return select_move_with_temperature(temp);
    }
    catch (const std::exception& e) {
        MCTS_DEBUG("Exception in select_move: " << e.what());
        return -1;
    }
}

Node* MCTS::select_node(Node* root) const {
    if (!root) {
        MCTS_DEBUG("select_node called with null root");
        return nullptr;
    }
    
    Node* current = root;
    std::vector<Node*> path;
    
    // Track the path from root to leaf with a maximum depth
    const int MAX_SEARCH_DEPTH = 200; // Reduced for safety
    int depth = 0;
    
    // CRITICAL FIX: Track already selected unvisited nodes to prevent loops
    std::set<Node*> selected_this_iteration;
    
    while (current && depth < MAX_SEARCH_DEPTH) {
        try {
            path.push_back(current);
            
            // Add to our set of selected nodes
            selected_this_iteration.insert(current);
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error adding node to path: " << e.what());
            break;
        }
        
        // Check for terminal state
        bool is_terminal = false;
        try {
            is_terminal = current->get_state().is_terminal();
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error checking terminal state: " << e.what());
            break;
        }
        
        if (is_terminal) {
            break;
        }
        
        // Check if this is a leaf node (no children yet)
        bool is_leaf = true;
        try {
            is_leaf = current->is_leaf();
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error checking leaf: " << e.what());
            break;
        }
        
        if (is_leaf) {
            break;
        }
        
        // Get children with proper error handling
        std::vector<Node*> children;
        try {
            children = current->get_children();
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error getting children: " << e.what());
            break;
        }
        
        if (children.empty()) {
            MCTS_DEBUG("Node has no children during selection");
            break;
        }
        
        // First, filter out null children and nodes already selected this iteration
        std::vector<Node*> valid_children;
        valid_children.reserve(children.size());
        
        for (Node* child : children) {
            if (child && selected_this_iteration.find(child) == selected_this_iteration.end()) {
                // Extra safety check: verify child state is accessible
                try {
                    const Gamestate& child_state = child->get_state();
                    int board_size = child_state.board_size;
                    
                    // Basic validation
                    if (board_size > 0 && board_size <= 25) {
                        valid_children.push_back(child);
                    }
                } catch (...) {
                    // Skip invalid children
                }
            }
        }
        
        // If no valid children, break to avoid loops
        if (valid_children.empty()) {
            MCTS_DEBUG("No more valid children to select (preventing loop), returning current node");
            break;
        }
        
        // Look for unvisited nodes - but track if we've already selected them before
        std::vector<Node*> unvisited_children;
        for (Node* child : valid_children) {
            try {
                // MODIFIED: Consider both visit count AND virtual losses
                int visit_count = child->get_visit_count();
                int virtual_losses = child->get_virtual_losses();
                
                // Only consider truly unvisited nodes with no virtual losses
                if (visit_count == 0 && virtual_losses == 0) {
                    unvisited_children.push_back(child);
                }
                // ADDED: For nodes with only virtual losses but no real visits,
                // don't add them to unvisited_children to prevent the issues observed in logs
            } catch (...) {
                // Skip on error
            }
        }
        
        // If we have unvisited children, prioritize them
        if (!unvisited_children.empty()) {
            // Select unvisited child with highest prior
            Node* best_unvisited = nullptr;
            float best_prior = -std::numeric_limits<float>::infinity();
            
            for (Node* child : unvisited_children) {
                try {
                    float prior = child->get_prior();
                    if (prior > best_prior) {
                        best_prior = prior;
                        best_unvisited = child;
                    }
                } catch (...) {
                    // Skip on error
                }
            }
            
            // If found, use this child
            if (best_unvisited) {
                // ADDED: Double-check it's actually safe to select
                int vl = 0;
                try {
                    vl = best_unvisited->get_virtual_losses();
                } catch (...) {
                    // Default to 0 if error
                }
                
                // Only use if still has no virtual losses (could have changed since we checked)
                if (vl > 0) {
                    MCTS_DEBUG("Unvisited child now has virtual losses, skipping");
                } else {
                    // Log selection of unvisited node
                    if (depth < 1) {
                        try {
                            MCTS_DEBUG("Selected unvisited child with move " << best_unvisited->get_move_from_parent() 
                                    << ", prior: " << best_unvisited->get_prior());
                        } catch (...) {
                            // Ignore logging errors
                        }
                    }
                    
                    current = best_unvisited;
                    depth++;
                    continue;
                }
            }
        }
        
        // If we reach here, we need to select from visited nodes
        float bestScore = -std::numeric_limits<float>::infinity();
        Node* bestChild = nullptr;
        
        // Calculate UCT scores for all valid children
        for (Node* child : valid_children) {
            float score = -std::numeric_limits<float>::infinity();
            try {
                score = uct_score(current, child);
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error calculating UCT score: " << e.what());
                continue;
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        
        if (!bestChild) {
            MCTS_DEBUG("Could not find best child during selection");
            // As a fallback, use first valid child
            if (!valid_children.empty()) {
                bestChild = valid_children[0];
                MCTS_DEBUG("Using fallback child");
            } else {
                break;
            }
        }
        
        // Log selected child
        if (depth < 1) {
            try {
                int visit_count = bestChild->get_visit_count();
                float q_val = bestChild->get_q_value();
                float prior = bestChild->get_prior();
                
                MCTS_DEBUG("Selected child with move " << bestChild->get_move_from_parent() 
                          << ", score: " << bestScore 
                          << ", visits: " << visit_count
                          << ", Q: " << q_val
                          << ", prior: " << prior);
            }
            catch (...) {
                // Ignore logging errors
            }
        }
        
        current = bestChild;
        depth++;
    }
    
    if (depth >= MAX_SEARCH_DEPTH) {
        MCTS_DEBUG("WARNING: Max search depth reached, possible loop in tree");
        return nullptr;
    }
    
    // Apply virtual loss to entire path
    for (Node* node : path) {
        if (node) {
            try {
                node->add_virtual_loss();
            }
            catch (...) {
                // Ignore virtual loss errors
            }
        }
    }
    
    return current;
}

void MCTS::expand_and_evaluate(Node* leaf) {
    if (!leaf) {
        MCTS_DEBUG("expand_and_evaluate called with null leaf");
        return;
    }
    
    // ADDED: Node should already be marked for expansion by the caller,
    // but double-check to be safe
    bool was_already_marked = leaf->is_being_expanded();
    if (!was_already_marked) {
        MCTS_DEBUG("Node was not marked for expansion, marking now");
        if (!leaf->mark_for_expansion()) {
            MCTS_DEBUG("Could not mark node for expansion, already being expanded");
            return;
        }
    }
    
    // Use a scoped guard to ensure we always clear the expansion flag on exit
    struct ExpansionGuard {
        Node* node;
        bool should_clear;
        
        ExpansionGuard(Node* n, bool clear) : node(n), should_clear(clear) {}
        
        ~ExpansionGuard() {
            if (node && should_clear) {
                try {
                    node->clear_expansion_flag();
                } catch (...) {
                    // Ignore errors in cleanup
                }
            }
        }
    };
    
    // Create the guard
    ExpansionGuard guard(leaf, !was_already_marked);
    
    try {
        Gamestate st = leaf->get_state();
        
        if (st.is_terminal()) {
            float r = 0.f;
            int winner = st.get_winner();
            if (winner == st.current_player) {
                r = 1.f;
            } else if (winner == 0) {
                r = 0.f;
            } else {
                r = -1.f;
            }
            backup(leaf, r);
            return;
        }
    
        int chosenMove = leaf->get_move_from_parent();
        if (chosenMove < 0) {
            std::vector<int> valid_moves = st.get_valid_moves();
            if (!valid_moves.empty()) {
                chosenMove = valid_moves[0];
            } else {
                chosenMove = 0;
            }
        }
    
        std::vector<std::vector<int>> board2D = st.get_board(); 
        std::vector<std::vector<std::vector<int>>> board_batch;
        board_batch.push_back(board2D);
        
        std::vector<int> chosen_moves;
        chosen_moves.push_back(chosenMove);
        
        std::vector<int> player_batch;
        player_batch.push_back(st.current_player);
        
        // Compute attack/defense bonuses safely
        std::vector<float> attackVec;
        std::vector<float> defenseVec;
        try {
            auto [a_vec, d_vec] = attackDefense_.compute_bonuses(
                board_batch, chosen_moves, player_batch);
            attackVec = a_vec;
            defenseVec = d_vec;
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error computing attack/defense bonuses: " << e.what());
            // Use defaults on error
            attackVec = {0.0f};
            defenseVec = {0.0f};
        }
    
        float attack = attackVec.empty() ? 0.0f : attackVec[0];
        float defense = defenseVec.empty() ? 0.0f : defenseVec[0];
    
        std::vector<float> policy;
        float value = 0.f;
        
        // Neural network evaluation with safety
        try {
            nn_->request_inference(st, chosenMove, attack, defense, policy, value);
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error in neural network inference: " << e.what());
            // Use default values on error
            auto validMoves = st.get_valid_moves();
            policy.resize(validMoves.size(), 1.0f / validMoves.size());
            value = 0.0f;
        }
    
        auto validMoves = st.get_valid_moves();
        std::vector<float> validPolicies;
        validPolicies.reserve(validMoves.size());
        
        // Process policy with safety checks
        if (policy.size() == validMoves.size()) {
            // Policy is already aligned with valid moves
            validPolicies = policy;
        } else {
            // Need to extract or create policy for valid moves
            for (int move : validMoves) {
                if (move >= 0 && move < static_cast<int>(policy.size())) {
                    validPolicies.push_back(policy[move]);
                } else {
                    validPolicies.push_back(1.0f / validMoves.size());
                }
            }
        }
        
        // Normalize policy
        float sum = 0.0f;
        for (float p : validPolicies) {
            sum += p;
        }
        
        if (sum > 0) {
            for (auto& p : validPolicies) {
                p /= sum;
            }
        } else {
            // Uniform policy if sum is zero
            for (auto& p : validPolicies) {
                p = 1.0f / validPolicies.size();
            }
        }
        
        // Expand node and mark as visited
        try {
            // ADDED: Check again if the node is still a leaf
            if (!leaf->is_leaf()) {
                MCTS_DEBUG("Node is no longer a leaf, skipping expansion");
                
                // Just backup the value
                backup(leaf, value);
                return;
            }
            
            leaf->expand(validMoves, validPolicies);
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error expanding leaf: " << e.what());
            // Try fallback expansion with uniform policy
            std::vector<float> uniform(validMoves.size(), 1.0f / validMoves.size());
            
            // ADDED: Check again if the node is still a leaf
            if (leaf->is_leaf()) {
                leaf->expand(validMoves, uniform);
            } else {
                MCTS_DEBUG("Node is no longer a leaf, skipping fallback expansion");
            }
        }
    
        // Backup value
        backup(leaf, value);
    } catch (const std::exception& e) {
        MCTS_DEBUG("Exception in expand_and_evaluate: " << e.what());
        
        // Emergency recovery - try to ensure node is marked as visited
        try {
            auto validMoves = leaf->get_state().get_valid_moves();
            if (!validMoves.empty() && leaf->is_leaf()) {
                std::vector<float> uniform(validMoves.size(), 1.0f / validMoves.size());
                leaf->expand(validMoves, uniform);
                leaf->update_stats(0.0f); // Mark as visited with neutral value
            }
        } catch (...) {
            // Last-resort fallback - at least remove virtual losses
            try {
                Node* current = leaf;
                while (current) {
                    current->remove_virtual_loss();
                    current = current->get_parent();
                }
            } catch (...) {}
        }
    }
    
    // Expansion flag will be cleared by the ExpansionGuard destructor
}

void MCTS::backup(Node* leaf, float value) {
    if (!leaf) {
        MCTS_DEBUG("backup called with null leaf");
        return;
    }
    
    // MCTS_DEBUG("Backing up value " << value << " from leaf with move " << leaf->get_move_from_parent());
    
    Node* current = leaf;
    int leafPlayer = leaf->get_state().current_player;
    
    // Use a maximum depth to prevent infinite loops
    const int MAX_BACKUP_DEPTH = 100;
    int depth = 0;
    
    // Keep track of the path for error reporting
    std::vector<int> move_path;
    
    while (current && depth < MAX_BACKUP_DEPTH) {
        // Record the path for error reporting
        try {
            move_path.push_back(current->get_move_from_parent());
        } catch (...) {
            // Ignore errors in move recording
        }
        
        int nodePlayer = 0;
        try {
            nodePlayer = current->get_state().current_player;
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error getting node player: " << e.what());
            break;
        }
        
        // Flip the sign for opponent's turns
        float adjusted_value = 0.0f;
        
        // Protect against invalid player values
        if (nodePlayer == 1 || nodePlayer == 2) {
            adjusted_value = (nodePlayer == leafPlayer) ? value : -value;
        } else {
            MCTS_DEBUG("Warning: Invalid player value: " << nodePlayer);
            adjusted_value = value;  // Default to original value
        }
        
        // Update node statistics
        try {
            current->update_stats(adjusted_value);
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error updating node stats: " << e.what());
            break;
        }
        
        // Remove the virtual loss that was added during selection
        try {
            current->remove_virtual_loss();
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error removing virtual loss: " << e.what());
            // Continue anyway
        }
        
        // Store current parent before moving to it
        Node* parent = nullptr;
        try {
            parent = current->get_parent();
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error getting parent: " << e.what());
            break;
        }
        
        // Move to parent - with null check
        current = parent;
        depth++;
    }
    
    if (depth >= MAX_BACKUP_DEPTH) {
        MCTS_DEBUG("WARNING: Max backup depth reached, possible loop in tree");
        
        // Print the path that caused the loop
        std::string path_str = "";
        for (int move : move_path) {
            path_str += std::to_string(move) + " -> ";
        }
        MCTS_DEBUG("Backup path: " << path_str);
    }
}

float MCTS::uct_score(const Node* parent, const Node* child) const {
    // Enhanced null checks with detailed logging
    if (!parent) {
        MCTS_DEBUG("uct_score: Parent node is null");
        return -std::numeric_limits<float>::infinity();
    }
    
    if (!child) {
        MCTS_DEBUG("uct_score: Child node is null");
        return -std::numeric_limits<float>::infinity();
    }

    // Use a try-catch block to handle any exceptions
    try {
        // Get child statistics safely
        int childVisits = 0;
        int virtual_losses = 0;
        
        try {
            childVisits = child->get_visit_count();
            virtual_losses = child->get_virtual_losses();
        } catch (...) {
            MCTS_DEBUG("uct_score: Exception getting child stats, defaulting to 0");
            childVisits = 0;
            virtual_losses = 0;
        }
        
        // CRITICAL FIX: For nodes with zero actual visits but with virtual losses,
        // they should be treated as *visited* nodes with negative results, not as
        // unvisited nodes with high exploration bonus
        if (childVisits == 0 && virtual_losses > 0) {
            // Use a strongly negative score to discourage further selection
            // but don't use -infinity to allow other nodes to be selected
            return -0.5f;
        }
        
        // Zero-visit case: use exploration-only score with some randomness to break ties
        if (childVisits == 0 && virtual_losses == 0) {
            float P = 0.0f;
            try {
                P = child->get_prior();
            } catch (...) {
                MCTS_DEBUG("uct_score: Exception getting prior, using default");
                P = 0.01f; // Small default prior
            }
            
            int parentVisits = 1;
            try {
                parentVisits = std::max(1, parent->get_visit_count());
            } catch (...) {
                MCTS_DEBUG("uct_score: Exception getting parent visits, using default");
            }
            
            // Use a safer exploration formula for unvisited nodes
            float exploration = config_.c_puct * P * std::sqrt(parentVisits) / 1.0f;
            
            // Add small random factor to break ties (but keep it deterministic based on move)
            int move = child->get_move_from_parent();
            float tie_breaker = 0.0001f * ((move * 123) % 1000) / 1000.0f;
            
            return exploration + tie_breaker;
        }
        
        // For visited nodes, use standard formula with safety checks
        float Q = 0.0f;
        try {
            Q = child->get_q_value();
            // Clamp Q to reasonable bounds to prevent extreme values
            Q = std::max(std::min(Q, 1.0f), -1.0f);
        } catch (...) {
            MCTS_DEBUG("uct_score: Exception getting Q value");
            Q = 0.0f; // Neutral score on error
        }
        
        float P = 0.0f;
        try {
            P = child->get_prior();
            // Ensure prior is positive and reasonable
            P = std::max(0.001f, std::min(P, 1.0f));
        } catch (...) {
            MCTS_DEBUG("uct_score: Exception getting prior");
            P = 0.01f; // Small default prior
        }
        
        int parentVisits = 0;
        try {
            parentVisits = parent->get_visit_count();
            // Ensure parent visits is positive
            parentVisits = std::max(1, parentVisits);
        } catch (...) {
            MCTS_DEBUG("uct_score: Exception getting parent visits");
            parentVisits = 1; // Default to 1 on error
        }
        
        // CRITICAL FIX: Count virtual losses for BOTH visit count AND Q-value calculation
        // Note: The Q-value already accounts for virtual losses (in get_q_value), but
        // they need to be considered consistently for the visit count used in the exploration term
        int effective_child_visits = childVisits + virtual_losses;
        effective_child_visits = std::max(1, effective_child_visits); // Ensure it's at least 1
        
        // Base PUCT constant
        float c_base = config_.c_puct;
        
        // Calculate exploration term with numeric stability 
        float parentSqrt = std::sqrt(static_cast<float>(parentVisits));
        
        // Ensure the exploration term is reasonable
        float U = c_base * P * parentSqrt / static_cast<float>(effective_child_visits);
        
        // Final safety check for NaN or infinity
        float final_score = Q + U;
        if (std::isnan(final_score) || std::isinf(final_score)) {
            MCTS_DEBUG("uct_score: Invalid final score: " << final_score 
                     << " (Q=" << Q << ", U=" << U << ")");
            return 0.0f; // Default to neutral score
        }
        
        // Return combined score (Q-value + exploration bonus)
        return final_score;
    }
    catch (const std::exception& e) {
        MCTS_DEBUG("Error calculating UCT score: " << e.what());
        return 0.0f; // Return neutral value on exception
    }
    catch (...) {
        MCTS_DEBUG("Unknown error calculating UCT score");
        return 0.0f; // Return neutral value on exception
    }
}

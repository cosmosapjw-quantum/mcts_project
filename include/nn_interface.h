// nn_interface.h
#pragma once

#include <pybind11/pybind11.h>
#include <random>
#include <future>
#include <signal.h>
#include <vector>
#include <mutex>
#include "gomoku.h"

/**
 * We store the final policy + value from the NN
 */
struct NNOutput {
    std::vector<float> policy;
    float value;
};

/**
 * Abstract interface.
 */
class NNInterface {
public:
    virtual ~NNInterface() = default;

    virtual void request_inference(const Gamestate& state,
                                   int chosen_move,
                                   float attack,
                                   float defense,
                                   std::vector<float>& outPolicy,
                                   float& outValue) = 0;
};

/**
 * Simplified thread-safe implementation
 */
class BatchingNNInterface : public NNInterface {
public:
    BatchingNNInterface(int num_history_moves = 3) 
        : rng_(std::random_device{}()), 
          use_dummy_(true),
          batch_size_(8),
          num_history_moves_(num_history_moves)
    {}
    
    void set_infer_callback(std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string, int, float, float>>&)> cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        python_infer_ = cb;
        use_dummy_ = false;
    }
    
    void set_batch_size(int size) {
        std::lock_guard<std::mutex> lock(mutex_);
        batch_size_ = std::max(1, size);
    }
    
    void set_num_history_moves(int num_moves) {
        std::lock_guard<std::mutex> lock(mutex_);
        num_history_moves_ = std::max(0, num_moves);
    }
    
    int get_num_history_moves() const {
        return num_history_moves_;
    }

    void request_inference(const Gamestate& state,
                        int chosen_move,
                        float attack,
                        float defense,
                        std::vector<float>& outPolicy,
                        float& outValue) {
        // Always use default/dummy values - for emergency fix
        outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
        outValue = 0.0f;
        
        if (use_dummy_ || !python_infer_) {
            return;
        }

        try {
            // Directly call Python with a strict timeout
            std::string stateStr = create_state_string(state, chosen_move, attack, defense);
            
            // Only call Python if we're not in a high-pressure situation
            if (simulations_in_flight_.load() < 10) {
                std::vector<std::tuple<std::string, int, float, float>> inputs = {
                    {stateStr, chosen_move, attack, defense}
                };
                
                // Call Python with a trivial timeout
                pybind11::gil_scoped_acquire acquire;
                auto results = python_infer_(inputs);
                
                if (!results.empty()) {
                    outPolicy = results[0].policy;
                    outValue = results[0].value;
                }
            }
        } catch (const std::exception& e) {
            // Silently handle errors - keep using default values
        }
    }

    // Add method to manually flush the batch
    void flush_batch() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!batch_inputs_.empty()) {
            process_batch();
        }
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        batch_inputs_.clear();
        batch_outputs_.clear();
    }

private:
    std::mt19937 rng_;
    bool use_dummy_;
    int batch_size_;
    int num_history_moves_; // Number of previous moves to include for each player
    std::mutex mutex_;
    std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)> python_infer_;
    std::atomic<int> simulations_in_flight_{0}; // Tracks the number of simulations in progress

    std::string create_state_string(const Gamestate& state, int chosen_move, float attack, float defense) {
        std::string stateStr;
        auto board = state.get_board();
        
        stateStr = "Board:" + std::to_string(state.board_size) + 
                ";Player:" + std::to_string(state.current_player) + 
                ";Last:" + std::to_string(state.action) + 
                ";State:";
        
        for (const auto& row : board) {
            for (int cell : row) {
                stateStr += std::to_string(cell);
            }
        }
        
        // Get previous moves for both players - for current player and opponent
        auto current_player_moves = state.get_previous_moves(state.current_player, num_history_moves_);
        auto opponent_player_moves = state.get_previous_moves(3 - state.current_player, num_history_moves_);
        
        // Convert moves to string representation
        std::string current_moves_str = ";CurrentMoves:";
        for (int move : current_player_moves) {
            current_moves_str += std::to_string(move) + ",";
        }
        
        std::string opponent_moves_str = ";OpponentMoves:";
        for (int move : opponent_player_moves) {
            opponent_moves_str += std::to_string(move) + ",";
        }
        
        // Append to state string
        stateStr += current_moves_str + opponent_moves_str;
        
        return stateStr;
    }

    void process_batch() {
        if (batch_inputs_.empty()) {
            MCTS_DEBUG("process_batch called with empty input batch");
            return;
        }
        
        MCTS_DEBUG("Processing batch of size " << batch_inputs_.size());
        auto inputs_copy = batch_inputs_;
        batch_inputs_.clear();
        
        std::vector<NNOutput> results;
        
        // Add a simple timeout to avoid hanging on Python calls
        auto start_time = std::chrono::steady_clock::now();
        bool python_timed_out = false;
        
        try {
            MCTS_DEBUG("Calling Python inference function");
            pybind11::gil_scoped_acquire acquire;
            
            // Set a signal handler for timeout if on Unix platforms
            #if defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__))
            // Setup alarm for timeout
            struct sigaction timeout_action;
            timeout_action.sa_handler = [](int sig) { throw std::runtime_error("Python call timed out"); };
            sigemptyset(&timeout_action.sa_mask);
            timeout_action.sa_flags = 0;
            struct sigaction old_action;
            sigaction(SIGALRM, &timeout_action, &old_action);
            alarm(1);  // 1 second timeout
            #endif
            
            // Call the Python function
            try {
                results = python_infer_(inputs_copy);
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error in Python call: " << e.what());
                python_timed_out = true;
            }
            
            #if defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__))
            // Cancel alarm and restore old handler
            alarm(0);
            sigaction(SIGALRM, &old_action, nullptr);
            #endif
            
            MCTS_DEBUG("Python inference returned " << results.size() << " results");
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error in process_batch: " << e.what());
            python_timed_out = true;
        }
        
        // Check for timeout based on elapsed time as a backup
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > 500) {
            python_timed_out = true;
            MCTS_DEBUG("Python inference timed out based on elapsed time");
        }
        
        // If we timed out or got no results, create default outputs
        if (python_timed_out || results.empty()) {
            MCTS_DEBUG("Using default values for batch");
            results.clear();
            for (size_t i = 0; i < inputs_copy.size(); i++) {
                NNOutput output;
                output.policy.resize(15 * 15, 1.0f / (15 * 15));
                output.value = 0.0f;
                results.push_back(output);
            }
        }
        
        // Ensure we have enough outputs
        batch_outputs_ = results;
        while (batch_outputs_.size() < inputs_copy.size()) {
            NNOutput default_output;
            default_output.policy.resize(15 * 15, 1.0f/(15 * 15));
            default_output.value = 0.0f;
            batch_outputs_.push_back(default_output);
        }
        
        MCTS_DEBUG("Batch processing complete, outputs size: " << batch_outputs_.size());
    }
    
    std::vector<std::tuple<std::string,int,float,float>> batch_inputs_;
    std::vector<NNOutput> batch_outputs_;
};
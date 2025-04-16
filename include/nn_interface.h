// nn_interface.h
#pragma once

#include <pybind11/pybind11.h>
#include <random>
#include <future>
#include <signal.h>
#include <vector>
#include <mutex>
#include <chrono>
#include "gomoku.h"
#include "debug.h"

// Add namespace alias at the top, after the includes
namespace py = pybind11;

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
 * Enhanced thread-safe implementation with batching support
 */
class BatchingNNInterface : public NNInterface {
public:
    BatchingNNInterface(int num_history_moves = 3) 
        : rng_(std::random_device{}()), 
          use_dummy_(true),
          batch_size_(8),
          num_history_moves_(num_history_moves),
          inference_count_(0),
          total_inference_time_ms_(0)
    {
        MCTS_DEBUG("BatchingNNInterface created with batch_size=" << batch_size_ 
                  << ", num_history_moves=" << num_history_moves_);
    }
    
    void set_infer_callback(std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string, int, float, float>>&)> cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        MCTS_DEBUG("Setting inference callback function");
        python_infer_ = cb;
        use_dummy_ = false;
    }
    
    void set_batch_size(int size) {
        std::lock_guard<std::mutex> lock(mutex_);
        MCTS_DEBUG("Setting batch size to " << size);
        batch_size_ = std::max(1, size);
    }
    
    void set_num_history_moves(int num_moves) {
        std::lock_guard<std::mutex> lock(mutex_);
        MCTS_DEBUG("Setting history moves to " << num_moves);
        num_history_moves_ = std::max(0, num_moves);
    }
    
    int get_num_history_moves() const {
        return num_history_moves_;
    }
    
    int get_batch_size() const {
        return batch_size_;
    }
    
    // Single inference request (maintaining compatibility)
    void request_inference(const Gamestate& state,
                        int chosen_move,
                        float attack,
                        float defense,
                        std::vector<float>& outPolicy,
                        float& outValue) override {
        MCTS_DEBUG("Requesting single inference");
        
        // Create default/dummy values as fallback
        outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
        outValue = 0.0f;
        
        if (use_dummy_ || !python_infer_) {
            MCTS_DEBUG("Using dummy values (no callback set)");
            return;
        }

        try {
            // Prepare batch with single input
            std::string stateStr = create_state_string(state, chosen_move, attack, defense);
            std::vector<std::tuple<std::string, int, float, float>> inputs = {
                {stateStr, chosen_move, attack, defense}
            };
            
            // Use batch inference method with timeout protection
            auto start_time = std::chrono::steady_clock::now();
            
            std::vector<NNOutput> results = batch_inference_internal(inputs);
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            // Track inference statistics
            inference_count_++;
            total_inference_time_ms_ += duration_ms;
            
            MCTS_DEBUG("Single inference completed in " << duration_ms << "ms");
            
            if (!results.empty()) {
                outPolicy = results[0].policy;
                outValue = results[0].value;
                MCTS_DEBUG("Received policy size: " << outPolicy.size() << ", value: " << outValue);
            } else {
                MCTS_DEBUG("Empty results from batch inference, using defaults");
            }
        } catch (const std::exception& e) {
            MCTS_DEBUG("Error in request_inference: " << e.what());
            // Keep using default values
        }
    }
    
    // New batch inference method for leaf parallelization
    std::vector<NNOutput> batch_inference(const std::vector<std::tuple<std::string, int, float, float>>& inputs) {
        MCTS_DEBUG("Batch inference requested with " << inputs.size() << " inputs");
        
        if (inputs.empty()) {
            MCTS_DEBUG("Empty batch, returning no results");
            return {};
        }
        
        if (use_dummy_) {
            MCTS_DEBUG("Using dummy values for batch inference (no callback set)");
            return create_default_outputs(inputs);
        }
        
        // Track timing
        auto start_time = std::chrono::steady_clock::now();
        
        // Call the Python inference function with proper GIL handling
        std::vector<NNOutput> results;
        
        try {
            // Acquire the GIL explicitly before calling into Python
            py::gil_scoped_acquire gil;
            
            MCTS_DEBUG("Acquired GIL, calling Python inference function");
            
            try {
                // Call the Python function and store results directly
                results = python_infer_(inputs);
                
                // Process the results if needed
                MCTS_DEBUG("Successfully received " << results.size() << " results");
            }
            catch (const pybind11::error_already_set& e) {
                MCTS_DEBUG("Python error during inference: " << e.what());
                results.clear();  // Ensure we return default values
            }
            
            // GIL is automatically released when gil goes out of scope
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error in batch inference: " << e.what());
            results.clear();
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Track inference statistics
        inference_count_++;
        total_inference_time_ms_ += duration_ms;
        
        MCTS_DEBUG("Batch inference completed in " << duration_ms << "ms");
        
        // If results are empty or mismatched, use defaults
        if (results.empty() || results.size() != inputs.size()) {
            MCTS_DEBUG("Empty or mismatched results, using defaults");
            return create_default_outputs(inputs);
        }
        
        return results;
    }
    
    // Make state string creation public for leaf parallelization
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
        
        // Append attack/defense values
        std::string bonus_str = ";Attack:" + std::to_string(attack) + 
                               ";Defense:" + std::to_string(defense);
        
        // Append to state string
        stateStr += current_moves_str + opponent_moves_str + bonus_str;
        
        return stateStr;
    }

private:
    std::mt19937 rng_;
    bool use_dummy_;
    int batch_size_;
    int num_history_moves_; // Number of previous moves to include for each player
    std::mutex mutex_;
    std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)> python_infer_;
    
    // Statistics tracking
    std::atomic<int> inference_count_;
    std::atomic<int64_t> total_inference_time_ms_;
    
    // Internal batch data
    std::vector<std::tuple<std::string,int,float,float>> batch_inputs_;
    std::vector<NNOutput> batch_outputs_;
    
    // Internal batch inference implementation with GIL safety and timeout
    std::vector<NNOutput> batch_inference_internal(const std::vector<std::tuple<std::string, int, float, float>>& inputs) {
        if (inputs.empty()) {
            MCTS_DEBUG("Empty inputs to batch_inference_internal");
            return {};
        }
        
        if (use_dummy_ || !python_infer_) {
            MCTS_DEBUG("Using dummy values for batch inference (no callback set)");
            return create_default_outputs(inputs);
        }
        
        // Prepare for Python call
        std::vector<NNOutput> results;
        
        // Track timing outside of inner scopes
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            MCTS_DEBUG("Calling Python inference with " << inputs.size() << " inputs");
            
            // CRITICAL GIL FIX: We need to ensure Python has the GIL before calling
            // Acquire the GIL explicitly before the function call
            pybind11::gil_scoped_acquire gil;
            
            // Call the inference function now that we have the GIL
            MCTS_DEBUG("GIL acquired, making Python callback");
            try {
                // Call the Python function and store results directly
                results = python_infer_(inputs);
                
                // Process the results if needed
                MCTS_DEBUG("Successfully received " << results.size() << " results");
            }
            catch (const pybind11::error_already_set& e) {
                MCTS_DEBUG("Python error during inference: " << e.what());
                results.clear();  // Ensure we return default values
            }
            
            // GIL is automatically released when gil goes out of scope
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error in batch_inference_internal: " << e.what());
            return create_default_outputs(inputs);
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        MCTS_DEBUG("Batch inference completed in " << duration_ms << "ms");
        
        // Track inference statistics
        inference_count_++;
        total_inference_time_ms_ += duration_ms;
        
        // Verify results
        if (results.empty() || results.size() != inputs.size()) {
            MCTS_DEBUG("Result size mismatch: got " << results.size() << ", expected " << inputs.size());
            return create_default_outputs(inputs);
        }
        
        return results;
    }
    
    // Add this helper method to the class for creating default outputs
    std::vector<NNOutput> create_default_outputs(const std::vector<std::tuple<std::string, int, float, float>>& inputs) const {
        std::vector<NNOutput> defaults;
        defaults.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            NNOutput output;
            const auto& [stateStr, move, attack, defense] = input;
            
            // Try to extract board size from state string
            int bs = 15; // Default
            size_t pos = stateStr.find("Board:");
            if (pos != std::string::npos) {
                size_t end = stateStr.find(';', pos);
                if (end != std::string::npos) {
                    std::string bs_str = stateStr.substr(pos + 6, end - pos - 6);
                    try {
                        bs = std::stoi(bs_str);
                    } catch (...) {
                        // Keep default
                    }
                }
            }
            
            output.policy.resize(bs * bs, 1.0f/(bs * bs));
            output.value = 0.0f;
            defaults.push_back(output);
        }
        return defaults;
    }
};
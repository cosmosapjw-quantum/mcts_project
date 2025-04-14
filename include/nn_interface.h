// nn_interface.h - Simplified thread-safe version
#pragma once

#include <pybind11/pybind11.h>
#include <random>
#include <vector>
#include <iostream>
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
// nn_interface.h
class BatchingNNInterface : public NNInterface {
public:
    BatchingNNInterface() 
        : rng_(std::random_device{}()), 
          use_dummy_(true),
          batch_size_(8)  // Default batch size
    {}
    
    void set_infer_callback(std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string, int, float, float>>&)> cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        python_infer_ = cb;
        use_dummy_ = false;  // When a callback is set, use it instead of dummy
    }
    
    void set_batch_size(int size) {
        std::lock_guard<std::mutex> lock(mutex_);
        batch_size_ = std::max(1, size);
    }

    void request_inference(const Gamestate& state,
                        int chosen_move,
                        float attack,
                        float defense,
                        std::vector<float>& outPolicy,
                        float& outValue) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // If no Python function is set or dummy mode is enabled, use random values
        if (use_dummy_ || !python_infer_) {
            // Generate uniform random policy
            outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
            outValue = 0.0f;
            return;
        }
        
        // Create a detailed state string with the full board state
        std::string stateStr;
        auto board = state.get_board();
        
        // Start with basic info
        stateStr = "Board:" + std::to_string(state.board_size) + 
                ";Player:" + std::to_string(state.current_player) + 
                ";Last:" + std::to_string(state.action) + 
                ";State:";
        
        // Add the full board state in a compact format
        for (const auto& row : board) {
            for (int cell : row) {
                stateStr += std::to_string(cell);
            }
        }
        
        // Add to batch queue
        batch_inputs_.push_back({stateStr, chosen_move, attack, defense});
        size_t request_idx = batch_outputs_.size();
        batch_outputs_.push_back({});
        
        // Process batch if full or force immediate processing for now
        // (In a more sophisticated implementation, we could delay processing)
        bool should_process = (batch_inputs_.size() >= batch_size_);
        
        if (should_process) {
            process_batch();
        } else {
            // Process immediately for simplicity
            process_batch();
        }
        
        // Copy result
        if (request_idx < batch_outputs_.size()) {
            const auto& result = batch_outputs_[request_idx];
            outPolicy = result.policy;
            outValue = result.value;
        } else {
            // Fallback values
            outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
            outValue = 0.0f;
        }
    }
    
    // No need for flush_batch with leaf parallelization
    void flush_batch() {}
    
private:
    std::mt19937 rng_;
    bool use_dummy_;
    int batch_size_;
    std::mutex mutex_;
    std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)> python_infer_;

    void process_batch() {
        if (batch_inputs_.empty()) return;
        
        // Copy inputs to temporary vector
        auto inputs_copy = batch_inputs_;
        
        // Clear inputs for next batch
        batch_inputs_.clear();
        
        // Release lock during Python call
        mutex_.unlock();
        
        // Call Python with GIL
        std::vector<NNOutput> results;
        try {
            pybind11::gil_scoped_acquire acquire;
            results = python_infer_(inputs_copy);
        } catch (const std::exception& e) {
            std::cerr << "NN_INTERFACE: Python error: " << e.what() << std::endl;
        }
        
        // Reacquire lock
        mutex_.lock();
        
        // Store results
        batch_outputs_ = results;
        
        // If sizes don't match, fill with defaults
        while (batch_outputs_.size() < inputs_copy.size()) {
            NNOutput default_output;
            default_output.policy.resize(15 * 15, 1.0f/(15 * 15));
            default_output.value = 0.0f;
            batch_outputs_.push_back(default_output);
        }
    }
    
    // Batch processing members
    std::vector<std::tuple<std::string,int,float,float>> batch_inputs_;
    std::vector<NNOutput> batch_outputs_;
};
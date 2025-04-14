// nn_interface.h
#pragma once

#include <pybind11/pybind11.h>
#include <random>
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
    BatchingNNInterface() 
        : rng_(std::random_device{}()), 
          use_dummy_(true),
          batch_size_(8)
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

    void request_inference(const Gamestate& state,
                        int chosen_move,
                        float attack,
                        float defense,
                        std::vector<float>& outPolicy,
                        float& outValue) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (use_dummy_ || !python_infer_) {
            outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
            outValue = 0.0f;
            return;
        }
        
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
        
        batch_inputs_.push_back({stateStr, chosen_move, attack, defense});
        size_t request_idx = batch_outputs_.size();
        batch_outputs_.push_back({});
        
        bool should_process = (batch_inputs_.size() >= batch_size_);
        
        if (should_process) {
            process_batch();
        } else {
            process_batch();
        }
        
        if (request_idx < batch_outputs_.size()) {
            const auto& result = batch_outputs_[request_idx];
            outPolicy = result.policy;
            outValue = result.value;
        } else {
            outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
            outValue = 0.0f;
        }
    }
    
    void flush_batch() {}
    
private:
    std::mt19937 rng_;
    bool use_dummy_;
    int batch_size_;
    std::mutex mutex_;
    std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)> python_infer_;

    void process_batch() {
        if (batch_inputs_.empty()) return;
        
        auto inputs_copy = batch_inputs_;
        batch_inputs_.clear();
        
        mutex_.unlock();
        
        std::vector<NNOutput> results;
        try {
            pybind11::gil_scoped_acquire acquire;
            results = python_infer_(inputs_copy);
        } catch (const std::exception& e) {
            // Silent error handling
        }
        
        mutex_.lock();
        
        batch_outputs_ = results;
        
        while (batch_outputs_.size() < inputs_copy.size()) {
            NNOutput default_output;
            default_output.policy.resize(15 * 15, 1.0f/(15 * 15));
            default_output.value = 0.0f;
            batch_outputs_.push_back(default_output);
        }
    }
    
    std::vector<std::tuple<std::string,int,float,float>> batch_inputs_;
    std::vector<NNOutput> batch_outputs_;
};
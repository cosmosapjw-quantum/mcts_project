// nn_interface.h
#pragma once

#include <pybind11/pybind11.h>
#include <random>
#include <vector>
#include <iostream>
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
 * Simplified dummy implementation that doesn't call into Python
 */
class BatchingNNInterface : public NNInterface {
public:
    BatchingNNInterface() 
        : rng_(std::random_device{}()), 
          use_dummy_(true),
          batch_size_(8)  // Default batch size
    {}
    
    void set_infer_callback(std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string, int, float, float>>&)> cb) {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        python_infer_ = cb;
        use_dummy_ = false;  // When a callback is set, use it instead of dummy
    }
    
    void set_batch_size(int size) {
        batch_size_ = std::max(1, size);
    }

    void request_inference(const Gamestate& state,
                                            int chosen_move,
                                            float attack,
                                            float defense,
                                            std::vector<float>& outPolicy,
                                            float& outValue) {
        // If no Python function is set or dummy mode is enabled, use random values
        if (use_dummy_ || !python_infer_) {
            std::cerr << "Using dummy NN with random policy/value" << std::endl;
            
            // Generate uniform random policy
            outPolicy.resize(state.board_size * state.board_size);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            // Fill with random values
            for (size_t i = 0; i < outPolicy.size(); i++) {
                outPolicy[i] = dist(rng_);
            }
            
            // Normalize
            float sum = 0.0f;
            for (float p : outPolicy) {
                sum += p;
            }
            if (sum > 0) {
                for (float& p : outPolicy) {
                    p /= sum;
                }
            } else {
                // Uniform if sum is 0
                float val = 1.0f / outPolicy.size();
                for (float& p : outPolicy) {
                    p = val;
                }
            }
            
            // Random value between -1 and 1
            std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
            outValue = val_dist(rng_);
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
        
        // Check if we should process immediately or batch
        if (batch_size_ <= 1) {
            // Process single request
            std::vector<std::tuple<std::string,int,float,float>> inputVec;
            inputVec.push_back({stateStr, chosen_move, attack, defense});
            
            pybind11::gil_scoped_acquire acquire;
            auto results = python_infer_(inputVec);
            
            if (!results.empty()) {
                outPolicy = results[0].policy;
                outValue = results[0].value;
            } else {
                // Fallback
                outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
                outValue = 0.0f;
            }
        } else {
            // Create a batch request
            std::lock_guard<std::mutex> lock(batch_mutex_);
            
            // Add to pending batch
            BatchRequest req;
            req.state_str = stateStr;
            req.chosen_move = chosen_move;
            req.attack = attack;
            req.defense = defense;
            req.out_policy = &outPolicy;
            req.out_value = &outValue;
            
            batch_requests_.push_back(req);
            
            // Process batch if we've reached the batch size
            if (batch_requests_.size() >= batch_size_) {
                process_batch();
            } else {
                // Just use a default value for now
                outPolicy.resize(state.board_size * state.board_size, 1.0f/(state.board_size * state.board_size));
                outValue = 0.0f;
            }
        }
    }
    
    // Should be called after MCTS search completes to process any remaining batched requests
    void flush_batch() {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        if (!batch_requests_.empty()) {
            process_batch();
        }
    }
    
private:
    struct BatchRequest {
        std::string state_str;
        int chosen_move;
        float attack;
        float defense;
        std::vector<float>* out_policy;
        float* out_value;
    };
    
    void process_batch() {
        if (batch_requests_.empty()) return;
        
        std::vector<std::tuple<std::string,int,float,float>> inputVec;
        for (const auto& req : batch_requests_) {
            inputVec.push_back({req.state_str, req.chosen_move, req.attack, req.defense});
        }
        
        std::cerr << "Processing batch of " << batch_requests_.size() << " requests" << std::endl;
        
        pybind11::gil_scoped_acquire acquire;
        auto results = python_infer_(inputVec);
        
        // Update the output policy and value for each request
        for (size_t i = 0; i < batch_requests_.size() && i < results.size(); i++) {
            *(batch_requests_[i].out_policy) = results[i].policy;
            *(batch_requests_[i].out_value) = results[i].value;
        }
        
        batch_requests_.clear();
    }
    
    std::mt19937 rng_;
    bool use_dummy_;
    int batch_size_;
    std::mutex batch_mutex_;
    std::vector<BatchRequest> batch_requests_;
    std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)> python_infer_;
};
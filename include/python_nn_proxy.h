// include/python_nn_proxy.h
#pragma once

#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <mutex>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nn_interface.h"
#include "gomoku.h"
#include "debug.h"

namespace py = pybind11;

/**
 * A thread-safe proxy for the Python neural network.
 * Uses a dedicated Python thread and message queues for communication.
 */
class PythonNNProxy : public NNInterface {
public:
    PythonNNProxy(int num_history_moves = 3) 
        : num_history_moves_(num_history_moves),
          next_request_id_(0),
          is_initialized_(false),
          inference_count_(0),
          total_inference_time_ms_(0)
    {
        MCTS_DEBUG("Creating PythonNNProxy");
    }
    
    ~PythonNNProxy() {
        MCTS_DEBUG("Destroying PythonNNProxy");
        shutdown();
    }
    
    /**
     * Initialize the proxy by importing the Python module and setting up the queues.
     * 
     * @param model The PyTorch model object
     * @param batch_size Maximum batch size for inference
     * @param device Device to run on ("cuda" or "cpu")
     * @return True if initialization was successful
     */
    bool initialize(py::object model, int batch_size = 16, const std::string& device = "cuda") {
        MCTS_DEBUG("Initializing PythonNNProxy with batch_size=" << batch_size << ", device=" << device);
        
        try {
            // Acquire the GIL
            py::gil_scoped_acquire gil;
            
            // Import the nn_proxy module
            py::module nn_proxy = py::module::import("nn_proxy");
            
            // Initialize the neural network
            nn_proxy.attr("initialize_neural_network")(model, batch_size, device);
            
            // Store the module for later use
            nn_proxy_module_ = nn_proxy;
            
            // Mark as initialized
            is_initialized_ = true;
            
            MCTS_DEBUG("PythonNNProxy initialization successful");
            return true;
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error initializing PythonNNProxy: " << e.what());
            return false;
        }
    }
    
    /**
     * Shutdown the proxy and the Python neural network thread.
     */
    void shutdown() {
        if (!is_initialized_) {
            return;
        }
        
        MCTS_DEBUG("Shutting down PythonNNProxy");
        
        try {
            // Acquire the GIL
            py::gil_scoped_acquire gil;
            
            // Call the shutdown function
            if (!nn_proxy_module_.is_none()) {
                nn_proxy_module_.attr("shutdown")();
            }
            
            // Clear the module reference
            nn_proxy_module_ = py::none();
            
            // Mark as uninitialized
            is_initialized_ = false;
            
            MCTS_DEBUG("PythonNNProxy shutdown complete");
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error shutting down PythonNNProxy: " << e.what());
        }
    }
    
    /**
     * Request inference for a single state.
     * 
     * @param state The game state
     * @param chosen_move The chosen move
     * @param attack The attack score
     * @param defense The defense score
     * @param outPolicy Output parameter for the policy vector
     * @param outValue Output parameter for the value
     */
    void request_inference(const Gamestate& state,
                           int chosen_move,
                           float attack,
                           float defense,
                           std::vector<float>& outPolicy,
                           float& outValue) override {
        MCTS_DEBUG("Single inference request");
        
        // Fill with default values in case of error
        outPolicy.resize(state.board_size * state.board_size, 1.0f / (state.board_size * state.board_size));
        outValue = 0.0f;
        
        if (!is_initialized_) {
            MCTS_DEBUG("PythonNNProxy not initialized");
            return;
        }
        
        // Start timing
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            // Create state string
            std::string state_str = create_state_string(state, chosen_move, attack, defense);
            
            // Generate a request ID
            int request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
            
            // Use the batch inference method with a single input
            std::vector<std::tuple<std::string, int, float, float>> inputs = {
                std::make_tuple(state_str, chosen_move, attack, defense)
            };
            
            auto results = inference_batch(inputs, request_id);
            
            if (!results.empty()) {
                outPolicy = results[0].policy;
                outValue = results[0].value;
            }
            
            // Record timing
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            inference_count_++;
            total_inference_time_ms_ += duration;
            
            MCTS_DEBUG("Single inference completed in " << duration << "ms");
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error in request_inference: " << e.what());
            // Default values will be used
        }
    }
    
    /**
     * Request batch inference for multiple states.
     * 
     * @param inputs Vector of (state_str, chosen_move, attack, defense) tuples
     * @return Vector of NNOutput objects containing policy and value
     */
    std::vector<NNOutput> batch_inference(const std::vector<std::tuple<std::string, int, float, float>>& inputs) {
        MCTS_DEBUG("Batch inference request with " << inputs.size() << " inputs");
        
        if (!is_initialized_) {
            MCTS_DEBUG("PythonNNProxy not initialized");
            return create_default_outputs(inputs);
        }
        
        if (inputs.empty()) {
            MCTS_DEBUG("Empty inputs");
            return {};
        }
        
        // Start timing
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            // Generate a base request ID
            int base_request_id = next_request_id_.fetch_add(static_cast<int>(inputs.size()), std::memory_order_relaxed);
            
            // Call the batch inference method
            auto results = inference_batch(inputs, base_request_id);
            
            // Record timing
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            inference_count_++;
            total_inference_time_ms_ += duration;
            
            MCTS_DEBUG("Batch inference completed in " << duration << "ms");
            
            return results;
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error in batch_inference: " << e.what());
            return create_default_outputs(inputs);
        }
    }
    
    /**
     * Set the number of history moves to include in the state string.
     * 
     * @param num_moves The number of history moves to include
     */
    void set_num_history_moves(int num_moves) {
        num_history_moves_ = num_moves;
    }
    
    /**
     * Get the number of history moves included in the state string.
     * 
     * @return The number of history moves
     */
    int get_num_history_moves() const {
        return num_history_moves_;
    }
    
    /**
     * Create a state string for the neural network.
     * 
     * @param state The game state
     * @param chosen_move The chosen move
     * @param attack The attack score
     * @param defense The defense score
     * @return The state string
     */
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
    
    /**
     * Get statistics about the neural network proxy.
     * 
     * @return A string containing statistics
     */
    std::string get_stats() const {
        std::string stats = "PythonNNProxy stats:\n";
        stats += "  Initialized: " + std::string(is_initialized_ ? "yes" : "no") + "\n";
        stats += "  Inference count: " + std::to_string(inference_count_) + "\n";
        
        if (inference_count_ > 0) {
            double avg_time = static_cast<double>(total_inference_time_ms_) / inference_count_;
            stats += "  Average inference time: " + std::to_string(avg_time) + " ms\n";
        }
        
        // Try to get queue information from Python
        if (is_initialized_) {
            try {
                // Acquire the GIL
                py::gil_scoped_acquire gil;
                
                // Get queue information
                std::string req_info = nn_proxy_module_.attr("get_request_info")().cast<std::string>();
                std::string resp_info = nn_proxy_module_.attr("get_response_info")().cast<std::string>();
                
                stats += "  " + req_info + "\n";
                stats += "  " + resp_info + "\n";
            }
            catch (...) {
                stats += "  Error getting queue information\n";
            }
        }
        
        return stats;
    }

private:
    int num_history_moves_;
    std::atomic<int> next_request_id_;
    bool is_initialized_;
    py::object nn_proxy_module_;
    
    // Statistics
    std::atomic<int> inference_count_;
    std::atomic<int64_t> total_inference_time_ms_;
    
    // Inference timeout
    static constexpr int INFERENCE_TIMEOUT_MS = 2000;
    
    /**
     * Create default outputs for the given inputs.
     * 
     * @param inputs Vector of (state_str, chosen_move, attack, defense) tuples
     * @return Vector of NNOutput objects with default values
     */
    std::vector<NNOutput> create_default_outputs(const std::vector<std::tuple<std::string, int, float, float>>& inputs) const {
        std::vector<NNOutput> outputs;
        outputs.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            const auto& state_str = std::get<0>(input);
            
            // Try to extract board size from state string
            int bs = 15; // Default
            size_t pos = state_str.find("Board:");
            if (pos != std::string::npos) {
                size_t end = state_str.find(';', pos);
                if (end != std::string::npos) {
                    std::string bs_str = state_str.substr(pos + 6, end - pos - 6);
                    try {
                        bs = std::stoi(bs_str);
                    } catch (...) {
                        // Keep default
                    }
                }
            }
            
            NNOutput output;
            output.policy.resize(bs * bs, 1.0f / (bs * bs));
            output.value = 0.0f;
            outputs.push_back(output);
        }
        
        return outputs;
    }
    
    /**
     * Send a batch of inference requests to the Python neural network thread.
     * 
     * @param inputs Vector of (state_str, chosen_move, attack, defense) tuples
     * @param base_request_id The base request ID to use
     * @return Vector of NNOutput objects containing policy and value
     */
    std::vector<NNOutput> inference_batch(const std::vector<std::tuple<std::string, int, float, float>>& inputs, int base_request_id) {
        MCTS_DEBUG("Processing batch of " << inputs.size() << " inputs");
        
        if (!is_initialized_) {
            MCTS_DEBUG("PythonNNProxy not initialized");
            return create_default_outputs(inputs);
        }
        
        // Prepare results vector
        std::vector<NNOutput> results(inputs.size());
        std::vector<bool> received(inputs.size(), false);
        
        // Map request IDs to indices
        std::map<int, int> request_map;
        
        try {
            // Acquire the GIL
            py::gil_scoped_acquire gil;
            
            // Get the queues
            py::object request_queue = nn_proxy_module_.attr("request_queue");
            py::object response_queue = nn_proxy_module_.attr("response_queue");
            
            // Send requests
            for (size_t i = 0; i < inputs.size(); i++) {
                int request_id = base_request_id + static_cast<int>(i);
                request_map[request_id] = static_cast<int>(i);
                
                const auto& [state_str, chosen_move, attack, defense] = inputs[i];
                
                // Create Python tuple for the request
                py::tuple py_request = py::make_tuple(
                    request_id,
                    state_str,
                    chosen_move,
                    attack,
                    defense
                );
                
                // Add to request queue
                request_queue.attr("put")(py_request);
            }
            
            // Wait for responses with timeout
            auto start_time = std::chrono::steady_clock::now();
            bool all_received = false;
            
            while (!all_received) {
                // Check timeout
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
                              (current_time - start_time).count();
                              
                if (elapsed > INFERENCE_TIMEOUT_MS) {
                    MCTS_DEBUG("Timeout waiting for inference results after " << elapsed << "ms");
                    break;
                }
                
                // Acquire the GIL for processing responses
                py::gil_scoped_acquire new_gil;
                
                // Check for responses
                int batch_responses = 0;
                bool queue_empty = false;
                
                while (!queue_empty && batch_responses < 10) {  // Process up to 10 responses at a time
                    try {
                        // Check if the queue is empty
                        if (response_queue.attr("empty")().cast<bool>()) {
                            queue_empty = true;
                            break;
                        }
                        
                        // Get a response without blocking
                        py::object py_response = response_queue.attr("get")(py::arg("block") = false);
                        
                        // Extract request ID and result
                        py::tuple py_tuple = py_response.cast<py::tuple>();
                        int request_id = py_tuple[0].cast<int>();
                        py::tuple py_result = py_tuple[1].cast<py::tuple>();
                        
                        // Find the corresponding request
                        auto it = request_map.find(request_id);
                        if (it != request_map.end()) {
                            int index = it->second;
                            
                            // Extract policy and value
                            py::list py_policy = py_result[0].cast<py::list>();
                            float value = py_result[1].cast<float>();
                            
                            // Convert policy to vector
                            std::vector<float> policy;
                            policy.reserve(py_policy.size());
                            for (auto p : py_policy) {
                                policy.push_back(p.cast<float>());
                            }
                            
                            // Store the result
                            results[index].policy = std::move(policy);
                            results[index].value = value;
                            received[index] = true;
                            
                            // Mark the task as done
                            response_queue.attr("task_done")();
                            
                            batch_responses++;
                        }
                    }
                    catch (const py::error_already_set& e) {
                        // If we get an empty exception, the queue is empty
                        if (std::string(e.what()).find("Empty") != std::string::npos) {
                            queue_empty = true;
                            break;
                        } else {
                            // Other Python error
                            MCTS_DEBUG("Python error processing response: " << e.what());
                        }
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("Error processing response: " << e.what());
                    }
                }
                
                // Check if all responses have been received
                all_received = std::all_of(received.begin(), received.end(), [](bool v){ return v; });
                
                if (all_received) {
                    break;
                }
                
                // Brief sleep to avoid tight loop
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            // Fill missing results with defaults
            for (size_t i = 0; i < inputs.size(); i++) {
                if (!received[i]) {
                    MCTS_DEBUG("Missing result for input " << i);
                    
                    // Extract board size from state string
                    const auto& state_str = std::get<0>(inputs[i]);
                    int bs = 15; // Default
                    size_t pos = state_str.find("Board:");
                    if (pos != std::string::npos) {
                        size_t end = state_str.find(';', pos);
                        if (end != std::string::npos) {
                            std::string bs_str = state_str.substr(pos + 6, end - pos - 6);
                            try {
                                bs = std::stoi(bs_str);
                            } catch (...) {
                                // Keep default
                            }
                        }
                    }
                    
                    // Create default values
                    results[i].policy.resize(bs * bs, 1.0f / (bs * bs));
                    results[i].value = 0.0f;
                }
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error in inference_batch: " << e.what());
            return create_default_outputs(inputs);
        }
        
        return results;
    }
};
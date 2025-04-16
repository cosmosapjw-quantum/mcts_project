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
     * Shutdown the proxy and the Python neural network thread with deadlock protection.
     */
    void shutdown() {
        MCTS_DEBUG("PythonNNProxy shutdown started");
        
        if (!is_initialized_) {
            MCTS_DEBUG("PythonNNProxy not initialized, nothing to shutdown");
            return;
        }
        
        // First, mark as uninitialized to prevent new requests
        is_initialized_ = false;
        
        // Wait briefly for any ongoing operations to complete
        MCTS_DEBUG("Waiting briefly for ongoing operations to complete...");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // We need a mutex here to prevent race conditions between shutdown paths
        static std::mutex shutdown_mutex;
        
        // Use try_lock with timeout instead of lock_guard to prevent deadlock
        bool got_lock = false;
        {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            while (std::chrono::steady_clock::now() < deadline) {
                if (shutdown_mutex.try_lock()) {
                    got_lock = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        if (!got_lock) {
            MCTS_DEBUG("Failed to acquire shutdown mutex after timeout, continuing with forced cleanup");
            // Continue without the lock as this is emergency shutdown
        } else {
            // We'll unlock manually at the end if we got the lock
        }
        
        // Try to shutdown Python in a separate thread with timeout
        std::atomic<bool> python_shutdown_complete{false};
        
        // Create a local copy of the Python module reference
        py::object local_module;
        try {
            if (!nn_proxy_module_.is_none()) {
                local_module = nn_proxy_module_;
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error accessing Python module: " << e.what());
        }
        
        // Clear the class reference BEFORE trying Python cleanup
        nn_proxy_module_ = py::none();
        
        // Only attempt Python shutdown if we have a valid module
        if (!local_module.is_none()) {
            // Use a separate thread for Python shutdown to avoid GIL deadlock
            std::thread shutdown_thread([&python_shutdown_complete, local_module]() {
                try {
                    // Try to acquire GIL and call shutdown
                    py::gil_scoped_acquire gil;
                    MCTS_DEBUG("GIL acquired for Python shutdown");
                    
                    try {
                        // Call Python shutdown function
                        MCTS_DEBUG("Calling Python shutdown function");
                        local_module.attr("shutdown")();
                        MCTS_DEBUG("Python shutdown function completed successfully");
                    }
                    catch (const py::error_already_set& e) {
                        MCTS_DEBUG("Python error in shutdown: " << e.what());
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("C++ error in shutdown: " << e.what());
                    }
                }
                catch (const std::exception& e) {
                    MCTS_DEBUG("Error acquiring GIL for shutdown: " << e.what());
                }
                
                // Mark as complete regardless of errors
                python_shutdown_complete.store(true);
            });
            
            // Wait for Python shutdown with timeout
            {
                auto start_time = std::chrono::steady_clock::now();
                auto timeout = std::chrono::milliseconds(1000); // 1 second timeout
                
                while (!python_shutdown_complete.load()) {
                    auto current_time = std::chrono::steady_clock::now();
                    auto elapsed = current_time - start_time;
                    
                    if (elapsed > timeout) {
                        MCTS_DEBUG("Python shutdown thread timed out after 1000ms, detaching thread");
                        shutdown_thread.detach();  // Detach the thread and continue
                        break;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                
                // Join thread if it completed in time
                if (python_shutdown_complete.load() && shutdown_thread.joinable()) {
                    shutdown_thread.join();
                }
            }
        }
        
        // Clear local reference to enable garbage collection
        local_module = py::none();
        
        // Reset statistics and state
        inference_count_ = 0;
        total_inference_time_ms_ = 0;
        
        // Release the mutex if we got it
        if (got_lock) {
            shutdown_mutex.unlock();
        }
        
        MCTS_DEBUG("PythonNNProxy shutdown complete");
    }

    // Helper method for thread joining with timeout (C++11 compatible)
    template <typename Rep, typename Period>
    void join_for(std::thread& t, const std::chrono::duration<Rep, Period>& timeout) {
        auto start = std::chrono::steady_clock::now();
        while (t.joinable()) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = now - start;
            
            if (elapsed >= timeout) {
                return;  // Timeout reached
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
        // Check if we're in shutdown or not initialized
        if (!is_initialized_) {
            MCTS_DEBUG("Batch inference requested during shutdown or uninitialized state, returning default values");
            return create_default_outputs(inputs);
        }
        
        MCTS_DEBUG("Batch inference requested with " << inputs.size() << " inputs");
        
        if (inputs.empty()) {
            MCTS_DEBUG("Empty batch, returning no results");
            return {};
        }
        
        if (nn_proxy_module_.is_none()) {
            MCTS_DEBUG("Neural network not initialized, returning default values");
            return create_default_outputs(inputs);
        }
        
        // Track timing
        auto start_time = std::chrono::steady_clock::now();
        
        // Create default results vector first - will be returned on any error
        std::vector<NNOutput> results = create_default_outputs(inputs);
        
        try {
            // Generate a base request ID
            int base_request_id = next_request_id_.fetch_add(static_cast<int>(inputs.size()), std::memory_order_relaxed);
            
            // Map request IDs to indices
            std::map<int, int> request_map;
            for (size_t i = 0; i < inputs.size(); i++) {
                request_map[base_request_id + static_cast<int>(i)] = static_cast<int>(i);
            }
            
            MCTS_DEBUG("Prepared batch with " << inputs.size() << " inputs, base request ID: " << base_request_id);
            
            // First phase: Send requests with minimal GIL holding
            bool request_send_success = false;
            {
                // Safety check for initialization before acquiring GIL
                if (!is_initialized_) {
                    MCTS_DEBUG("Detected uninitialized state before sending requests, returning default values");
                    return results;
                }
                
                // Explicitly acquire the GIL for Python operations
                py::gil_scoped_acquire gil;
                
                try {
                    // Check again if Python module is still valid
                    if (nn_proxy_module_.is_none()) {
                        MCTS_DEBUG("Python module is None, cannot send requests");
                        return results;
                    }
                    
                    // Get the queue - with extra safety checks
                    py::object request_queue;
                    try {
                        request_queue = nn_proxy_module_.attr("request_queue");
                        if (request_queue.is_none()) {
                            MCTS_DEBUG("Request queue is None");
                            return results;
                        }
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error getting request queue: " << e.what());
                        return results;
                    }
                    
                    // Attempt to send up to 5 requests with retries
                    int sent_count = 0;
                    int retry_count = 0;
                    const int MAX_RETRIES = 3;
                    const int MAX_TO_SEND = std::min(5, static_cast<int>(inputs.size()));
                    
                    while (sent_count < MAX_TO_SEND && retry_count < MAX_RETRIES) {
                        try {
                            for (int i = sent_count; i < MAX_TO_SEND; i++) {
                                int request_id = base_request_id + i;
                                const auto& [state_str, chosen_move, attack, defense] = inputs[i];
                                
                                // Create Python tuple for the request
                                py::tuple py_request = py::make_tuple(
                                    request_id,
                                    state_str,
                                    chosen_move,
                                    attack,
                                    defense
                                );
                                
                                // Try to add to request queue with very short timeout
                                request_queue.attr("put")(py_request, py::arg("block") = true, py::arg("timeout") = 0.01);
                                sent_count++;
                            }
                            break; // Successful sending, exit retry loop
                        } catch (const py::error_already_set& e) {
                            // Queue full or timeout
                            if (std::string(e.what()).find("Full") != std::string::npos) {
                                MCTS_DEBUG("Queue full, sent " << sent_count << "/" << MAX_TO_SEND << " requests");
                                retry_count++;
                                
                                // Short sleep before retry - need to temporarily release GIL
                                // IMPORTANT: Don't manually release GIL, create a nested scope instead
                                {
                                    // Temporarily release GIL
                                    py::gil_scoped_release release;
                                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                                }
                                // GIL is automatically re-acquired when the above scope ends
                            } else {
                                MCTS_DEBUG("Python error during request sending: " << e.what());
                                break;
                            }
                        } catch (const std::exception& e) {
                            MCTS_DEBUG("Error sending request: " << e.what());
                            break;
                        }
                    }
                    
                    request_send_success = (sent_count > 0);
                    MCTS_DEBUG("Sent " << sent_count << "/" << MAX_TO_SEND << " requests to Python queue");
                }
                catch (const py::error_already_set& e) {
                    MCTS_DEBUG("Python error during request sending: " << e.what());
                }
                catch (const std::exception& e) {
                    MCTS_DEBUG("C++ error during request sending: " << e.what());
                }
                
                // GIL automatically released when exiting scope
            }
            
            // If we couldn't send any requests, return default values
            if (!request_send_success) {
                MCTS_DEBUG("Failed to send any requests, returning default values");
                return results;
            }
            
            // Second phase: Wait for responses with timeout - using multiple short GIL acquisitions
            auto wait_start = std::chrono::steady_clock::now();
            
            // Calculate adaptive timeout based on batch size and device
            const int BASE_TIMEOUT_MS = 100;       // Reduced base timeout
            const int MS_PER_ITEM = 10;            // Reduced per-item timeout
            const int MAX_TIMEOUT_MS = 1000;       // Maximum 1 second timeout
            
            int timeout_ms = std::min(BASE_TIMEOUT_MS + static_cast<int>(inputs.size() * MS_PER_ITEM), MAX_TIMEOUT_MS);
            
            MCTS_DEBUG("Waiting for responses with timeout of " << timeout_ms << "ms");
            
            // Wait loop with timeout using multiple short GIL acquisitions
            std::vector<bool> received(inputs.size(), false);
            int received_count = 0;
            const int MAX_RESPONSE_BATCH = 5;  // Process at most 5 responses at once
            
            while (received_count < static_cast<int>(inputs.size())) {
                // Check timeout first
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - wait_start).count();
                    
                if (elapsed_ms > timeout_ms || !is_initialized_) {
                    if (elapsed_ms > timeout_ms) {
                        MCTS_DEBUG("Timeout waiting for inference results after " << elapsed_ms << "ms");
                    } else {
                        MCTS_DEBUG("Uninitialized state detected while waiting for responses");
                    }
                    break;
                }
                
                // Brief sleep to avoid tight loop
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                
                // Acquire GIL for a short period to check for responses
                {
                    py::gil_scoped_acquire gil;
                    
                    try {
                        // Check if Python module is still valid
                        if (nn_proxy_module_.is_none()) {
                            MCTS_DEBUG("Python module is None during response check");
                            break;
                        }
                        
                        // Get response queue with error handling
                        py::object response_queue;
                        try {
                            response_queue = nn_proxy_module_.attr("response_queue");
                            if (response_queue.is_none()) {
                                MCTS_DEBUG("Response queue is None");
                                break;
                            }
                        } catch (const std::exception& e) {
                            MCTS_DEBUG("Error getting response queue: " << e.what());
                            break;
                        }
                        
                        // Check if the queue is empty
                        bool queue_empty = false;
                        try {
                            queue_empty = response_queue.attr("empty")().cast<bool>();
                        } catch (const std::exception& e) {
                            MCTS_DEBUG("Error checking if queue is empty: " << e.what());
                            break;
                        }
                        
                        if (queue_empty) {
                            continue;  // No responses yet
                        }
                        
                        // Process a limited number of responses to keep GIL acquisition short
                        int responses_processed = 0;
                        
                        while (!response_queue.attr("empty")().cast<bool>() && 
                               responses_processed < MAX_RESPONSE_BATCH) {
                            
                            try {
                                // Get a response without blocking
                                py::object py_response = response_queue.attr("get")(py::arg("block") = false);
                                
                                // Mark task as done immediately to avoid queue blocking
                                response_queue.attr("task_done")();
                                
                                // Process the response
                                py::tuple py_tuple;
                                try {
                                    py_tuple = py_response.cast<py::tuple>();
                                } catch (const std::exception& e) {
                                    MCTS_DEBUG("Error casting response to tuple: " << e.what());
                                    continue;
                                }
                                
                                if (py_tuple.size() < 2) {
                                    MCTS_DEBUG("Invalid response tuple size: " << py_tuple.size());
                                    continue;
                                }
                                
                                // Extract request ID
                                int request_id;
                                try {
                                    request_id = py_tuple[0].cast<int>();
                                } catch (const std::exception& e) {
                                    MCTS_DEBUG("Error extracting request ID: " << e.what());
                                    continue;
                                }
                                
                                // Find the corresponding request
                                auto it = request_map.find(request_id);
                                if (it == request_map.end()) {
                                    continue;  // Not one of our requests
                                }
                                
                                int index = it->second;
                                if (index < 0 || index >= static_cast<int>(results.size())) {
                                    MCTS_DEBUG("Invalid response index: " << index);
                                    continue;
                                }
                                
                                // Extract result
                                py::tuple py_result;
                                try {
                                    py_result = py_tuple[1].cast<py::tuple>();
                                } catch (const std::exception& e) {
                                    MCTS_DEBUG("Error extracting result tuple: " << e.what());
                                    continue;
                                }
                                
                                if (py_result.size() < 2) {
                                    MCTS_DEBUG("Invalid result tuple size: " << py_result.size());
                                    continue;
                                }
                                
                                // Extract policy and value with error handling
                                std::vector<float> policy;
                                float value = 0.0f;
                                
                                try {
                                    // Extract policy
                                    py::list py_policy = py_result[0].cast<py::list>();
                                    policy.reserve(py_policy.size());
                                    for (auto p : py_policy) {
                                        policy.push_back(p.cast<float>());
                                    }
                                    
                                    // Extract value
                                    value = py_result[1].cast<float>();
                                } catch (const std::exception& e) {
                                    MCTS_DEBUG("Error extracting policy/value: " << e.what());
                                    continue;
                                }
                                
                                // Store the result
                                results[index].policy = std::move(policy);
                                results[index].value = value;
                                
                                if (!received[index]) {
                                    received[index] = true;
                                    received_count++;
                                }
                                
                                responses_processed++;
                            } catch (const py::error_already_set& e) {
                                // If Queue.Empty exception, break the loop
                                if (std::string(e.what()).find("Empty") != std::string::npos) {
                                    break;
                                }
                                MCTS_DEBUG("Python error processing responses: " << e.what());
                                break;
                            } catch (const std::exception& e) {
                                MCTS_DEBUG("Error processing response: " << e.what());
                                break;
                            }
                        }
                    }
                    catch (const py::error_already_set& e) {
                        MCTS_DEBUG("Python error checking responses: " << e.what());
                    }
                    catch (const std::exception& e) {
                        MCTS_DEBUG("Error checking responses: " << e.what());
                    }
                }
                
                // If we received all responses, break early
                if (received_count >= static_cast<int>(inputs.size())) {
                    MCTS_DEBUG("Received all " << inputs.size() << " responses");
                    break;
                }
            }
            
            // Record timing
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            
            inference_count_++;
            total_inference_time_ms_ += duration;
            
            int missing = static_cast<int>(inputs.size()) - received_count;
            if (missing > 0) {
                MCTS_DEBUG("Batch inference completed in " << duration << "ms, missing " << missing << " responses");
            } else {
                MCTS_DEBUG("Batch inference completed in " << duration << "ms with all responses received");
            }
            
            return results;
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error in batch_inference: " << e.what());
            return results;  // Return the pre-filled default results
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
    static constexpr int INFERENCE_TIMEOUT_MS = 500;  // Reduced from 2000ms to 500ms
    
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
            int bs = extract_board_size(state_str);
            
            // Create default output with uniform policy
            NNOutput output;
            output.policy.resize(bs * bs, 1.0f / (bs * bs));
            output.value = 0.0f;
            outputs.push_back(output);
        }
        
        return outputs;
    }
    
    // Helper method to extract board size from state string
    int extract_board_size(const std::string& state_str) const {
        int bs = 15; // Default
        size_t pos = state_str.find("Board:");
        if (pos != std::string::npos) {
            size_t end = state_str.find(';', pos);
            if (end != std::string::npos) {
                std::string bs_str = state_str.substr(pos + 6, end - pos - 6);
                try {
                    bs = std::stoi(bs_str);
                    // Sanity check - board size should be reasonable
                    if (bs < 3 || bs > 25) {
                        MCTS_DEBUG("Invalid board size: " << bs << ", using default 15");
                        bs = 15;
                    }
                } catch (...) {
                    // Keep default
                    MCTS_DEBUG("Failed to parse board size, using default 15");
                }
            }
        }
        return bs;
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
                if (elapsed % 10 == 0) {  // Only sleep every 10ms of elapsed time
                    std::this_thread::sleep_for(std::chrono::microseconds(100));  // Use 0.1ms instead of 1ms
                }
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

private:
    bool use_dummy_ = false; // Indicates whether to use dummy values
};
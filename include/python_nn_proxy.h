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

// Static mutex and shutdown flag to prevent deadlocks
extern std::recursive_mutex batch_inference_mutex;
extern std::atomic<bool> shutdown_in_progress;

/**
 * A thread-safe proxy for the Python neural network.
 * Uses a dedicated Python thread and message queues for communication.
 */
// Complete fixed PythonNNProxy class to prevent segmentation faults

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
        // Don't acquire GIL or access Python during construction
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
        
        // First, ensure we're not already initialized
        if (is_initialized_) {
            MCTS_DEBUG("PythonNNProxy already initialized, shutting down first");
            shutdown();
            
            // Brief wait to ensure cleanup is complete
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        try {
            // Acquire the GIL
            py::gil_scoped_acquire gil;
            
            // First check if pybind11 is still active
            if (PyErr_Occurred()) {
                MCTS_DEBUG("Python error detected during initialization");
                PyErr_Print();
                PyErr_Clear();
            }
            
            // Import the nn_proxy module
            try {
                py::module nn_proxy = py::module::import("nn_proxy");
                
                // Initialize the neural network - make a copy of the model to ensure lifetime
                py::object model_copy = model;
                nn_proxy.attr("initialize_neural_network")(model_copy, batch_size, device);
                
                // Store the module for later use
                nn_proxy_module_ = nn_proxy;
                
                // Mark as initialized
                is_initialized_ = true;
                
                MCTS_DEBUG("PythonNNProxy initialization successful");
                return true;
            }
            catch (const py::error_already_set& e) {
                MCTS_DEBUG("Python error importing nn_proxy: " << e.what());
                return false;
            }
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
        
        // First mark shutdown in progress to prevent new operations
        shutdown_in_progress.store(true, std::memory_order_release);
        
        if (!is_initialized_) {
            MCTS_DEBUG("PythonNNProxy not initialized, nothing to shutdown");
            return;
        }
        
        // First, mark as uninitialized to prevent new requests
        is_initialized_ = false;
        
        // Unlock the batch inference mutex if locked by this thread
        {
            if (batch_inference_mutex.try_lock()) {
                MCTS_DEBUG("Unlocking batch inference mutex during shutdown");
                batch_inference_mutex.unlock();
            } else {
                MCTS_DEBUG("Batch inference mutex already unlocked or owned by another thread");
            }
        }
        
        // Wait briefly for any ongoing operations to complete
        MCTS_DEBUG("Waiting briefly for ongoing operations to complete...");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Make a defensive copy of the Python module reference
        py::object local_module;
        try {
            py::gil_scoped_acquire gil;
            
            if (!nn_proxy_module_.is_none()) {
                local_module = nn_proxy_module_;
                
                // Clear the class reference BEFORE trying Python shutdown
                nn_proxy_module_ = py::none();
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error accessing Python module: " << e.what());
            
            // Force reset module reference on error
            try {
                nn_proxy_module_ = py::none();
            } catch (...) {
                // Ignore errors
            }
        }
        
        // Only attempt Python shutdown if we have a valid module
        if (!local_module.is_none()) {
            // Use a separate thread for Python shutdown to avoid GIL deadlock
            std::thread shutdown_thread([local_module]() {
                try {
                    // Try to acquire GIL and call shutdown
                    py::gil_scoped_acquire gil;
                    
                    // First check if Python is still active
                    if (PyErr_Occurred()) {
                        MCTS_DEBUG("Python error detected during shutdown");
                        PyErr_Print();
                        PyErr_Clear();
                    }
                    
                    try {
                        // Call Python shutdown function
                        MCTS_DEBUG("Calling Python shutdown function");
                        if (py::hasattr(local_module, "shutdown")) {
                            local_module.attr("shutdown")();
                            MCTS_DEBUG("Python shutdown function completed successfully");
                        } else {
                            MCTS_DEBUG("Python module has no shutdown function");
                        }
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
            });
            
            // Don't wait - detach the thread immediately
            MCTS_DEBUG("Detaching Python shutdown thread to avoid deadlocks");
            shutdown_thread.detach();
        }
        
        // Reset statistics and state
        inference_count_ = 0;
        total_inference_time_ms_ = 0;
        
        // Wait a moment for any detached threads to progress
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        MCTS_DEBUG("PythonNNProxy shutdown complete");
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
        
        if (!is_initialized_ || shutdown_in_progress.load(std::memory_order_acquire)) {
            MCTS_DEBUG("PythonNNProxy not initialized or shutting down");
            return;
        }
        
        // Start timing
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            // Create state string
            std::string state_str = create_state_string(state, chosen_move, attack, defense);
            
            // Generate a request ID
            int request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
            
            // Use the batch inference method with a single input for consistency
            std::vector<std::tuple<std::string, int, float, float>> inputs = {
                std::make_tuple(state_str, chosen_move, attack, defense)
            };
            
            // CRITICAL: Use batch_inference with consistent GIL management
            auto results = batch_inference(inputs);
            
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
        if (!is_initialized_ || shutdown_in_progress.load(std::memory_order_acquire)) {
            MCTS_DEBUG("Batch inference requested during shutdown or uninitialized state, returning default values");
            return create_default_outputs(inputs);
        }
        
        MCTS_DEBUG("Batch inference requested with " << inputs.size() << " inputs");
        
        if (inputs.empty()) {
            MCTS_DEBUG("Empty batch, returning no results");
            return {};
        }
        
        // Track timing
        auto start_time = std::chrono::steady_clock::now();
        
        // Create default results vector first - will be returned on any error
        std::vector<NNOutput> results = create_default_outputs(inputs);
        
        // CRITICAL FIX: Set up emergency timeout monitoring
        std::atomic<bool> batch_inference_completed{false};
        std::thread timeout_monitor([&batch_inference_completed, start_time]() {
            const int ABSOLUTE_MAX_TIMEOUT_MS = 1000; // 1 second absolute maximum
            
            while (!batch_inference_completed.load(std::memory_order_acquire)) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count();
                    
                if (elapsed_ms > ABSOLUTE_MAX_TIMEOUT_MS) {
                    MCTS_DEBUG("EMERGENCY: batch_inference absolute timeout after " << elapsed_ms << "ms");
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
        
        // Make sure the thread is detached for safety
        timeout_monitor.detach();
        
        // CRITICAL FIX: First check Python responsiveness WITHOUT acquiring the batch mutex
        // This prevents deadlock where another thread holds GIL but waits for mutex
        bool python_responsive = false;
        {
            try {
                // Quick check if Python is responsive - with short timeout
                py::gil_scoped_acquire gil;
                
                // First check if the module itself is valid
                if (nn_proxy_module_.is_none()) {
                    MCTS_DEBUG("Python module is None, not responsive");
                    is_initialized_ = false;  // Mark as uninitialized 
                    batch_inference_completed.store(true, std::memory_order_release);
                    return results;
                }
                
                // CRITICAL FIX: Also verify the required queues exist and are valid
                bool queues_valid = false;
                try {
                    py::object req_queue = nn_proxy_module_.attr("request_queue");
                    py::object resp_queue = nn_proxy_module_.attr("response_queue");
                    
                    if (req_queue.is_none() || resp_queue.is_none()) {
                        MCTS_DEBUG("Python queues are None, marking as uninitialized");
                        is_initialized_ = false;
                        batch_inference_completed.store(true, std::memory_order_release);
                        return results;
                    }
                    
                    // Check if the queue methods are callable
                    if (py::hasattr(req_queue, "put") && py::hasattr(resp_queue, "get")) {
                        queues_valid = true;
                    } else {
                        MCTS_DEBUG("Python queues missing required methods");
                        is_initialized_ = false;
                        batch_inference_completed.store(true, std::memory_order_release);
                        return results;
                    }
                } catch (const py::error_already_set& e) {
                    MCTS_DEBUG("Python error checking queues: " << e.what());
                    is_initialized_ = false;
                    batch_inference_completed.store(true, std::memory_order_release);
                    return results;
                } catch (const std::exception& e) {
                    MCTS_DEBUG("Error checking Python queues: " << e.what());
                    is_initialized_ = false;
                    batch_inference_completed.store(true, std::memory_order_release);
                    return results;
                }
                
                // Module and queues are valid
                if (queues_valid) {
                    python_responsive = true;
                    MCTS_DEBUG("Python module is responsive with valid queues");
                }
            }
            catch (const std::exception& e) {
                MCTS_DEBUG("Error checking Python responsiveness: " << e.what());
                is_initialized_ = false;  // Mark as uninitialized on any error
                batch_inference_completed.store(true, std::memory_order_release);
                return results;
            }
        }
        
        if (!python_responsive) {
            MCTS_DEBUG("Python appears unresponsive, returning default values");
            batch_inference_completed.store(true, std::memory_order_release);
            return results;
        }
        
        // Prepare request IDs and mapping outside the mutex
        int base_request_id = next_request_id_.fetch_add(static_cast<int>(inputs.size()), std::memory_order_relaxed);
        std::map<int, int> request_map;
        for (size_t i = 0; i < inputs.size(); i++) {
            request_map[base_request_id + static_cast<int>(i)] = static_cast<int>(i);
        }
        
        // CRITICAL: Use a smaller scope for the mutex lock and add timeout
        bool acquired_mutex = false;
        {
            // Try to acquire the mutex with a timeout
            std::unique_lock<std::recursive_mutex> lock(batch_inference_mutex, std::defer_lock);
            
            // First attempt with try_lock
            acquired_mutex = lock.try_lock();
            
            // If that fails, try again with a short timeout
            if (!acquired_mutex && !shutdown_in_progress.load(std::memory_order_acquire)) {
                auto lock_start = std::chrono::steady_clock::now();
                const int LOCK_TIMEOUT_MS = 200; // 200ms timeout
                
                while (!acquired_mutex && 
                      !shutdown_in_progress.load(std::memory_order_acquire) &&
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - lock_start).count() < LOCK_TIMEOUT_MS) {
                    
                    // Short sleep between attempts
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    acquired_mutex = lock.try_lock();
                }
            }
            
            // Check if we failed to acquire the mutex
            if (!acquired_mutex) {
                MCTS_DEBUG("Failed to acquire batch mutex after timeout, returning default values");
                batch_inference_completed.store(true, std::memory_order_release);
                return results;
            }
            
            // Mutex is now locked if we get here, and will be automatically unlocked when lock goes out of scope
            MCTS_DEBUG("Batch mutex acquired for request ID: " << base_request_id);
        } // Mutex is released here
        
        // Send requests with timeout protection - without holding mutex
        MCTS_DEBUG("Sending requests to Python queue");
        bool requests_sent = false;
        int sent_count = 0;
        
        {
            // Use a separate thread with timeout to send requests
            std::atomic<bool> sending_complete{false};
            std::atomic<int> local_sent_count{0};
            std::atomic<bool> sending_failed{false};
            
            std::thread sending_thread([&]() {
                try {
                    // Acquire GIL for Python operations
                    py::gil_scoped_acquire gil;
                    
                    // First check again if we're in shutdown
                    if (shutdown_in_progress.load(std::memory_order_acquire) || !is_initialized_) {
                        MCTS_DEBUG("Shutdown or uninitialized state detected before sending requests");
                        sending_failed.store(true, std::memory_order_release);
                        return;
                    }
                    
                    // Get the request queue
                    py::object request_queue;
                    try {
                        request_queue = nn_proxy_module_.attr("request_queue");
                        
                        // Double-check it's not None
                        if (request_queue.is_none()) {
                            MCTS_DEBUG("Request queue is None");
                            sending_failed.store(true, std::memory_order_release);
                            is_initialized_ = false;  // Mark as uninitialized
                            return;
                        }
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error getting request queue: " << e.what());
                        sending_failed.store(true, std::memory_order_release);
                        is_initialized_ = false;  // Mark as uninitialized
                        return;
                    }
                    
                    // Send requests with a simple retry approach
                    const int MAX_RETRIES = 2;
                    int local_count = 0;
                    
                    for (size_t i = 0; i < inputs.size(); i++) {
                        int retry_count = 0;
                        while (retry_count < MAX_RETRIES) {
                            try {
                                int request_id = base_request_id + static_cast<int>(i);
                                const auto& [state_str, chosen_move, attack, defense] = inputs[i];
                                
                                // Create Python tuple for the request
                                py::tuple py_request = py::make_tuple(
                                    request_id,
                                    state_str,
                                    chosen_move,
                                    attack,
                                    defense
                                );
                                
                                // Use a short timeout to prevent hanging
                                request_queue.attr("put")(py_request, py::arg("block") = true, py::arg("timeout") = 0.1);
                                local_count++;
                                break;  // Successfully sent
                            }
                            catch (const py::error_already_set& e) {
                                // Check for shutdown in middle of loop
                                if (shutdown_in_progress.load(std::memory_order_acquire)) {
                                    MCTS_DEBUG("Shutdown detected during request sending");
                                    sending_failed.store(true, std::memory_order_release);
                                    return;
                                }
                                
                                // Handle "Full" exception with retry
                                retry_count++;
                                if (retry_count >= MAX_RETRIES) {
                                    MCTS_DEBUG("Failed to send request " << i << " after " << MAX_RETRIES << " retries");
                                }
                                else {
                                    // Very brief sleep between retries
                                    py::gil_scoped_release release;
                                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                                }
                            }
                            catch (const std::exception& e) {
                                MCTS_DEBUG("Error sending request " << i << ": " << e.what());
                                break;  // Don't retry on other errors
                            }
                        }
                        
                        // Update atomic counter for timeout thread to see
                        local_sent_count.store(local_count, std::memory_order_release);
                        
                        // Check for shutdown periodically
                        if (i % 8 == 0 && 
                            (shutdown_in_progress.load(std::memory_order_acquire) || !is_initialized_)) {
                            MCTS_DEBUG("Shutdown detected during request batch");
                            break;
                        }
                    }
                    
                    // GIL is released automatically at the end of this scope
                }
                catch (const py::error_already_set& e) {
                    MCTS_DEBUG("Python error during request sending: " << e.what());
                    sending_failed.store(true, std::memory_order_release);
                    is_initialized_ = false;  // Mark as uninitialized
                }
                catch (const std::exception& e) {
                    MCTS_DEBUG("Error during request sending: " << e.what());
                    sending_failed.store(true, std::memory_order_release);
                }
                
                // Mark as complete
                sending_complete.store(true, std::memory_order_release);
            });
            
            // Wait for the sending thread with timeout
            const int SEND_TIMEOUT_MS = 300; // 300ms timeout
            auto send_start = std::chrono::steady_clock::now();
            
            while (!sending_complete.load(std::memory_order_acquire)) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - send_start).count();
                    
                if (elapsed_ms > SEND_TIMEOUT_MS) {
                    MCTS_DEBUG("Request sending timeout after " << elapsed_ms << "ms");
                    break;
                }
                
                // Update sent count from atomic
                sent_count = local_sent_count.load(std::memory_order_acquire);
                
                // Periodic check for shutdown
                if (shutdown_in_progress.load(std::memory_order_acquire) || !is_initialized_) {
                    MCTS_DEBUG("Shutdown detected while waiting for request sending");
                    break;
                }
                
                // Brief sleep
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Detach the thread if still running (don't join)
            if (sending_thread.joinable()) {
                sending_thread.detach();
            }
            
            // Final update of sent count
            sent_count = local_sent_count.load(std::memory_order_acquire);
            
            // Check if sending was successful
            requests_sent = (sent_count > 0) && !sending_failed.load(std::memory_order_acquire);
        }
        
        MCTS_DEBUG("Sent " << sent_count << "/" << inputs.size() << " requests to Python queue");
        
        if (!requests_sent) {
            MCTS_DEBUG("Failed to send any requests or sending failed, returning default values");
            batch_inference_completed.store(true, std::memory_order_release);
            return results;
        }
        
        // Wait for responses with strict timeout
        MCTS_DEBUG("Waiting for responses from Python");
        int received_count = 0;
        
        // Reduced timeouts to avoid long stalls
        const int BASE_TIMEOUT_MS = 200;
        const int MS_PER_ITEM = 20;
        const int MAX_TIMEOUT_MS = 800;
        
        int timeout_ms = std::min(BASE_TIMEOUT_MS + static_cast<int>(inputs.size() * MS_PER_ITEM), MAX_TIMEOUT_MS);
        MCTS_DEBUG("Using timeout of " << timeout_ms << "ms for response waiting");
        
        auto wait_start = std::chrono::steady_clock::now();
        
        // Use polling with frequent GIL release/reacquisition
        while (received_count < static_cast<int>(inputs.size())) {
            // Check total timeout
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - wait_start).count();
                
            if (elapsed_ms > timeout_ms || shutdown_in_progress.load(std::memory_order_acquire) || !is_initialized_) {
                MCTS_DEBUG("Timeout waiting for responses after " << elapsed_ms 
                          << "ms, received " << received_count << "/" << inputs.size());
                break;
            }
            
            // ADDED: Periodically check module validity during long waits
            if (elapsed_ms > 200 && elapsed_ms % 100 < 10) {  // Check roughly every 100ms after 200ms
                if (!ensure_module_valid()) {
                    MCTS_DEBUG("Module became invalid during response waiting");
                    break;
                }
            }
            
            // Short sleep outside the GIL
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            
            // Process available responses
            int responses_this_iteration = 0;
            const int MAX_RESPONSES_PER_ITERATION = 10;
            
            {
                // Acquire GIL for Python operations
                py::gil_scoped_acquire gil;
                
                try {
                    // Check if shutdown was requested
                    if (shutdown_in_progress.load(std::memory_order_acquire)) {
                        MCTS_DEBUG("Shutdown requested during response waiting");
                        break;
                    }
                    
                    // Check if Python module is still valid
                    if (nn_proxy_module_.is_none()) {
                        MCTS_DEBUG("Python module became None during response waiting");
                        is_initialized_ = false;
                        break;
                    }
                    
                    // Check if response queue is accessible
                    py::object response_queue;
                    try {
                        response_queue = nn_proxy_module_.attr("response_queue");
                        
                        // Double-check it's not None
                        if (response_queue.is_none()) {
                            MCTS_DEBUG("Response queue is None during waiting");
                            is_initialized_ = false;
                            break;
                        }
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error accessing response queue: " << e.what());
                        is_initialized_ = false;
                        break;
                    }
                    
                    // Check if queue is empty
                    bool queue_empty = false;
                    try {
                        queue_empty = response_queue.attr("empty")().cast<bool>();
                    } catch (const std::exception& e) {
                        MCTS_DEBUG("Error checking if queue is empty: " << e.what());
                        is_initialized_ = false;
                        break;
                    }
                    
                    if (queue_empty) {
                        continue; // Skip to next iteration with sleep
                    }
                    
                    // Process available responses up to the limit
                    for (int i = 0; i < MAX_RESPONSES_PER_ITERATION; i++) {
                        try {
                            // Check again if queue is empty
                            if (response_queue.attr("empty")().cast<bool>()) {
                                break; // No more responses
                            }
                            
                            // Get response without blocking
                            py::object py_response = response_queue.attr("get")(py::arg("block") = false);
                            
                            // Mark task as done
                            response_queue.attr("task_done")();
                            
                            // Process the response
                            py::tuple py_tuple = py_response.cast<py::tuple>();
                            if (py_tuple.size() < 2) {
                                MCTS_DEBUG("Invalid response tuple size: " << py_tuple.size());
                                continue;
                            }
                            
                            int request_id = py_tuple[0].cast<int>();
                            py::tuple py_result = py_tuple[1].cast<py::tuple>();
                            
                            // Check if this request belongs to our batch
                            auto it = request_map.find(request_id);
                            if (it == request_map.end()) {
                                MCTS_DEBUG("Received response for unknown request ID: " << request_id);
                                continue;
                            }
                            
                            int index = it->second;
                            if (index < 0 || index >= static_cast<int>(results.size())) {
                                MCTS_DEBUG("Invalid result index: " << index);
                                continue;
                            }
                            
                            // Extract policy and value
                            if (py_result.size() >= 2) {
                                // Extract policy
                                py::list py_policy = py_result[0].cast<py::list>();
                                results[index].policy.clear();
                                results[index].policy.reserve(py_policy.size());
                                
                                for (auto p : py_policy) {
                                    results[index].policy.push_back(p.cast<float>());
                                }
                                
                                // Extract value
                                results[index].value = py_result[1].cast<float>();
                                
                                received_count++;
                                responses_this_iteration++;
                            }
                            else {
                                MCTS_DEBUG("Invalid result tuple size: " << py_result.size());
                            }
                        }
                        catch (const py::error_already_set& e) {
                            // Queue.Empty exception is expected
                            if (std::string(e.what()).find("Empty") != std::string::npos) {
                                break;
                            }
                            MCTS_DEBUG("Python error processing response: " << e.what());
                            break;
                        }
                        catch (const std::exception& e) {
                            MCTS_DEBUG("Error processing response: " << e.what());
                            break;
                        }
                    }
                    
                    // Log progress for large batches
                    if (responses_this_iteration > 0 && inputs.size() > 8) {
                        MCTS_DEBUG("Processed " << responses_this_iteration << " responses, total " 
                                  << received_count << "/" << inputs.size());
                    }
                }
                catch (const py::error_already_set& e) {
                    MCTS_DEBUG("Python error during response processing: " << e.what());
                    is_initialized_ = false;
                    break;
                }
                catch (const std::exception& e) {
                    MCTS_DEBUG("Error during response processing: " << e.what());
                    is_initialized_ = false;
                    break;
                }
                
                // GIL is automatically released at end of scope
            }
            
            // If we've received all responses, we're done
            if (received_count >= static_cast<int>(inputs.size())) {
                MCTS_DEBUG("Received all " << inputs.size() << " responses");
                break;
            }
        }
        
        // Log final status
        if (received_count < static_cast<int>(inputs.size())) {
            MCTS_DEBUG("Incomplete response set: received " << received_count << "/" << inputs.size());
        }
        
        // Record timing
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        inference_count_++;
        total_inference_time_ms_ += duration;
        
        // Set completion flag for timeout monitor
        batch_inference_completed.store(true, std::memory_order_release);
        
        // Check for absolute timeout
        auto total_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
            
        if (total_elapsed_ms > 1000) {  // Same as ABSOLUTE_MAX_TIMEOUT_MS
            MCTS_DEBUG("batch_inference took " << total_elapsed_ms << "ms (exceeded emergency timeout)");
        }
        
        MCTS_DEBUG("Batch inference completed in " << duration << "ms with " 
                  << results.size() << " results (complete: " << (received_count == static_cast<int>(inputs.size())) << ")");
        
        return results;
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
        if (is_initialized_ && !shutdown_in_progress.load(std::memory_order_acquire)) {
            try {
                // Try to acquire the GIL with timeout
                py::gil_scoped_acquire gil;
                
                // Check if module is valid
                if (!nn_proxy_module_.is_none() && 
                    py::hasattr(nn_proxy_module_, "get_request_info") && 
                    py::hasattr(nn_proxy_module_, "get_response_info")) {
                    
                    // Get queue information
                    std::string req_info = nn_proxy_module_.attr("get_request_info")().cast<std::string>();
                    std::string resp_info = nn_proxy_module_.attr("get_response_info")().cast<std::string>();
                    
                    stats += "  " + req_info + "\n";
                    stats += "  " + resp_info + "\n";
                } else {
                    stats += "  Module not available for queue information\n";
                }
            }
            catch (...) {
                stats += "  Error getting queue information\n";
            }
        } else {
            stats += "  Python module not initialized or shutting down\n";
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
    
    // Added helper method to check module validity
    bool ensure_module_valid() {
        try {
            // Quick check with GIL acquisition
            py::gil_scoped_acquire gil;
            
            // Check if module is None
            if (nn_proxy_module_.is_none()) {
                MCTS_DEBUG("Neural network module is None");
                is_initialized_ = false;
                return false;
            }
            
            // Check if queues exist and are valid
            try {
                py::object req_queue = nn_proxy_module_.attr("request_queue");
                py::object resp_queue = nn_proxy_module_.attr("response_queue");
                
                if (req_queue.is_none() || resp_queue.is_none()) {
                    MCTS_DEBUG("Python queues are None, marking module as invalid");
                    is_initialized_ = false;
                    return false;
                }
            } catch (const py::error_already_set& e) {
                MCTS_DEBUG("Python error checking queues: " << e.what());
                is_initialized_ = false;
                return false;
            } catch (const std::exception& e) {
                MCTS_DEBUG("Error checking Python queues: " << e.what());
                is_initialized_ = false;
                return false;
            }
            
            // Everything looks valid
            return true;
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error checking module validity: " << e.what());
            is_initialized_ = false;
            return false;
        }
    }
    
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
};
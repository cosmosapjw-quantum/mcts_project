// nn_interface.cpp
#include "nn_interface.h"
#include <iostream>
#include <pybind11/pybind11.h>


BatchingNNInterface::BatchingNNInterface()
 : stop_(false)
{
    worker_ = std::thread(&BatchingNNInterface::worker_loop, this);
}

BatchingNNInterface::~BatchingNNInterface() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void BatchingNNInterface::set_infer_callback(std::function<
    std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)
> cb) {
    python_infer_ = cb;
}

void BatchingNNInterface::request_inference(const Gamestate& state,
                                            int chosen_move,
                                            float attack,
                                            float defense,
                                            std::vector<float>& outPolicy,
                                            float& outValue)
{
    Request req;
    req.state       = state;
    req.chosen_move = chosen_move;
    req.attack      = attack;
    req.defense     = defense;
    req.outPolicy   = &outPolicy;
    req.outValue    = &outValue;

    std::promise<void> pr;
    req.done = std::move(pr);
    auto fut = req.done.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        requests_.push(std::move(req));
    }
    cv_.notify_one();

    fut.wait();
}

void BatchingNNInterface::worker_loop() {
    while(true) {
        std::vector<Request> batch;
        {
            // Release GIL while waiting on condition variable
            pybind11::gil_scoped_release release_gil;
            
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]{ return stop_ || !requests_.empty(); });
            
            if (stop_ && requests_.empty()) {
                return;
            }
            
            // Move all pending requests to our local batch
            while(!requests_.empty()) {
                batch.push_back(std::move(requests_.front()));
                requests_.pop();
            }
        } // lock released here
        
        std::vector<std::tuple<std::string,int,float,float>> inputVec;
        inputVec.reserve(batch.size());
        
        try {
            // Prepare batch input data
            for (auto& r : batch) {
                // Better string representation of board state
                std::string stateStr;
                try {
                    auto board = r.state.get_board();
                    stateStr = "Board:" + std::to_string(r.state.board_size) + 
                              ";Player:" + std::to_string(r.state.current_player) + 
                              ";Last:" + std::to_string(r.state.action);
                } catch (const std::exception& e) {
                    stateStr = "Error getting board: " + std::string(e.what());
                }
                inputVec.push_back({stateStr, r.chosen_move, r.attack, r.defense});
            }
            
            std::vector<NNOutput> results;
            
            // Only acquire GIL when calling Python
            if (python_infer_) {
                pybind11::gil_scoped_acquire acquire_gil;
                try {
                    results = python_infer_(inputVec);
                } catch (const pybind11::error_already_set& e) {
                    std::cerr << "Python error in inference: " << e.what() << std::endl;
                    // Create fallback results on error
                    results.resize(batch.size(), {{0.5f, 0.5f}, 0.0f});
                } catch (const std::exception& e) {
                    std::cerr << "C++ error in inference: " << e.what() << std::endl;
                    results.resize(batch.size(), {{0.5f, 0.5f}, 0.0f});
                }
            } else {
                // Fallback if no Python function is set
                results.resize(batch.size(), {{0.5f, 0.5f}, 0.0f});
            }
            
            // Validate results size matches batch size
            if (results.size() != batch.size()) {
                std::cerr << "Warning: Python callback returned " << results.size() 
                          << " results, but expected " << batch.size() << std::endl;
                // Resize to match (either truncate or add defaults)
                if (results.size() < batch.size()) {
                    results.resize(batch.size(), {{0.5f, 0.5f}, 0.0f});
                }
            }
            
            // Return results to waiting threads
            for (size_t i = 0; i < batch.size(); i++) {
                try {
                    if (i < results.size()) {
                        *batch[i].outPolicy = results[i].policy;
                        *batch[i].outValue = results[i].value;
                    }
                    batch[i].done.set_value();
                } catch (const std::exception& e) {
                    std::cerr << "Error setting result: " << e.what() << std::endl;
                    // Still fulfill promise to avoid deadlock
                    batch[i].done.set_value();
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Unhandled error in worker thread: " << e.what() << std::endl;
            // Make sure to set all remaining promises to avoid deadlock
            for (auto& r : batch) {
                try {
                    r.done.set_value();
                } catch (...) {
                    // Last-resort error handling - nothing more we can do
                }
            }
        }
    }
}

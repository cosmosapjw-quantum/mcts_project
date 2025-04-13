// nn_interface.cpp
#include "nn_interface.h"
#include <iostream>

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
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]{ return stop_ || !requests_.empty();});
            if (stop_ && requests_.empty()) {
                return;
            }
            while(!requests_.empty()) {
                batch.push_back(std::move(requests_.front()));
                requests_.pop();
            }
        }

        // Build input for python: (stringifiedGamestate, chosenMove, attack, defense)
        std::vector<std::tuple<std::string,int,float,float>> inputVec;
        inputVec.reserve(batch.size());
        for (auto& r : batch) {
            // We'll use the user-provided "board_equal" or "to_tensor" or 
            // simply "to_string" from Gamestate. Suppose we do:
            std::string stateStr = "Gamestate board: " + std::to_string(r.state.board_size) + 
                                   " etc... " + std::to_string(r.state.current_player);
            // If your real gomoku code has a method to generate a string, do that:
            // e.g. `auto s = r.state.to_string();`
            inputVec.push_back({stateStr, r.chosen_move, r.attack, r.defense});
        }

        std::vector<NNOutput> results;
        if (python_infer_) {
            results = python_infer_(inputVec);
        } else {
            // fallback
            results.resize(batch.size(), {{0.5f, 0.5f}, 0.f});
        }

        // Return results
        for (size_t i = 0; i < batch.size(); i++) {
            *batch[i].outPolicy = results[i].policy;
            *batch[i].outValue  = results[i].value;
            batch[i].done.set_value();
        }
    }
}

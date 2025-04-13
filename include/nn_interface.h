// nn_interface.h
#pragma once

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>
#include <functional>
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
 * Implementation with a background worker that does big batch inference.
 * We pass a python callback that accepts a vector of (stateString, chosenMove, attack, defense).
 */
class BatchingNNInterface : public NNInterface {
public:
    BatchingNNInterface();
    ~BatchingNNInterface();

    void request_inference(const Gamestate& state,
                           int chosen_move,
                           float attack,
                           float defense,
                           std::vector<float>& outPolicy,
                           float& outValue) override;

    void set_infer_callback(std::function<
        std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)
    > cb);

private:
    void worker_loop();

    struct Request {
        Gamestate state;
        int chosen_move;
        float attack;
        float defense;

        std::vector<float>* outPolicy;
        float* outValue;

        std::promise<void> done;
    };

    bool stop_;
    std::thread worker_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::queue<Request> requests_;

    // The Python callback: receives a list of (boardString, move, attack, defense) => returns [NNOutput...]
    std::function<std::vector<NNOutput>(const std::vector<std::tuple<std::string,int,float,float>> &)> python_infer_;
};

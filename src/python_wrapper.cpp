// python_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gomoku.h"
#include "attack_defense.h"
#include "mcts_config.h"
#include "mcts.h"
#include "nn_interface.h"

namespace py = pybind11;

/**
 * We'll wrap everything in a single MCTSWrapper class 
 * that Python can instantiate. 
 */
class MCTSWrapper {
public:
    MCTSWrapper(const MCTSConfig& cfg,
                int boardSize,    // pass from Python
                bool use_renju,
                bool use_omok,
                int seed,
                bool use_pro_long_opening)
    {
        // Create a "blank" Gamestate with desired size and rules
        // per your gomoku constructor:
        Gamestate st(boardSize, use_renju, use_omok, seed, use_pro_long_opening);

        nn_ = std::make_shared<BatchingNNInterface>();
        mcts_ = std::make_unique<MCTS>(cfg, nn_, boardSize);

        config_ = cfg;
        rootState_ = st;
    }

    // Let Python set the GPU inference function
    void set_infer_function(py::function pyFn) {
        nn_->set_infer_callback(
          [pyFn](const std::vector<std::tuple<std::string,int,float,float>>& inputData)
          -> std::vector<NNOutput>
          {
             py::gil_scoped_acquire gil;
             // Build a python list of tuples
             py::list pyList;
             for (auto &item : inputData) {
                 auto [s, move, att, def] = item;
                 pyList.append(py::make_tuple(s, move, att, def));
             }
             // Call python, expecting list of (policyList, value)
             py::object pyResult = pyFn(pyList);

             std::vector<NNOutput> outputs;
             outputs.reserve(inputData.size());
             for (auto r : pyResult) {
                 auto tup = r.cast<std::pair<std::vector<float>, float>>();
                 NNOutput out;
                 out.policy = tup.first;
                 out.value  = tup.second;
                 outputs.push_back(std::move(out));
             }
             return outputs;
          }
        );
    }

    // Run MCTS from the stored rootState
    void run_search() {
        mcts_->run_search(rootState_);
    }

    int best_move() const {
        return mcts_->select_move();
    }

    // If you want to apply the best move to rootState and continue searching, you can:
    void apply_best_move() {
        int mv = mcts_->select_move();
        if (mv >= 0) {
            rootState_.make_move(mv, rootState_.current_player);
        }
    }

    // You might want direct access to the root Gamestate
    // to see if it's terminal, or show the board, etc.
    bool is_terminal() const {
        return rootState_.is_terminal();
    }

    int get_winner() const {
        return rootState_.get_winner();
    }

private:
    MCTSConfig config_;
    std::shared_ptr<BatchingNNInterface> nn_;
    std::unique_ptr<MCTS> mcts_;

    Gamestate rootState_;
};


PYBIND11_MODULE(mcts_py, m) {
    py::class_<MCTSConfig>(m, "MCTSConfig")
       .def(py::init<>())
       .def_readwrite("num_simulations", &MCTSConfig::num_simulations)
       .def_readwrite("c_puct", &MCTSConfig::c_puct)
       .def_readwrite("parallel_leaf_batch_size", &MCTSConfig::parallel_leaf_batch_size)
       .def_readwrite("num_threads", &MCTSConfig::num_threads);

    py::class_<MCTSWrapper>(m, "MCTSWrapper")
       .def(py::init<const MCTSConfig&,int,bool,bool,int,bool>(),
            py::arg("config"),
            py::arg("boardSize"),
            py::arg("use_renju")=false,
            py::arg("use_omok")=false,
            py::arg("seed")=0,
            py::arg("use_pro_long_opening")=false
       )
       .def("set_infer_function", &MCTSWrapper::set_infer_function)
       .def("run_search", &MCTSWrapper::run_search)
       .def("best_move", &MCTSWrapper::best_move)
       .def("apply_best_move", &MCTSWrapper::apply_best_move)
       .def("is_terminal", &MCTSWrapper::is_terminal)
       .def("get_winner", &MCTSWrapper::get_winner);
}

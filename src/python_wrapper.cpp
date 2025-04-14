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

    std::shared_ptr<BatchingNNInterface> _get_nn_interface() {
        return nn_;
    }

    void set_batch_size(int size) {
        nn_->set_batch_size(size);
    }

    // Let Python set the GPU inference function
    void set_infer_function(py::function pyFn) {
        // This is now a no-op as we're using a dummy implementation
        // We keep the method for API compatibility
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

    // Add temperature-based move selection
    int best_move_with_temperature(float temperature = 1.0f) const {
        return mcts_->select_move_with_temperature(temperature);
    }
    
    // Apply the move selected with temperature
    void apply_best_move_with_temperature(float temperature = 1.0f) {
        int mv = mcts_->select_move_with_temperature(temperature);
        if (mv >= 0) {
            rootState_.make_move(mv, rootState_.current_player);
        }
    }
    
    // Set Dirichlet noise parameters
    void set_exploration_parameters(float dirichlet_alpha, float noise_weight) {
        mcts_->set_dirichlet_alpha(dirichlet_alpha);
        mcts_->set_noise_weight(noise_weight);
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
       .def("get_winner", &MCTSWrapper::get_winner)
       .def("set_batch_size", &MCTSWrapper::set_batch_size)
       .def("best_move_with_temperature", &MCTSWrapper::best_move_with_temperature,
            py::arg("temperature") = 1.0f)
        .def("apply_best_move_with_temperature", &MCTSWrapper::apply_best_move_with_temperature,
            py::arg("temperature") = 1.0f)
        .def("set_exploration_parameters", &MCTSWrapper::set_exploration_parameters,
            py::arg("dirichlet_alpha") = 0.03f, py::arg("noise_weight") = 0.25f);
}

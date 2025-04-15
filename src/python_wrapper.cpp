// python_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gomoku.h"
#include "attack_defense.h"
#include "mcts_config.h"
#include "mcts.h"
#include "nn_interface.h"
#include <iostream>

namespace py = pybind11;

/**
 * We'll wrap everything in a single MCTSWrapper class 
 * that Python can instantiate. 
 */
class MCTSWrapper {
public:
MCTSWrapper(const MCTSConfig& cfg,
            int boardSize,
            bool use_renju,
            bool use_omok,
            int seed,
            bool use_pro_long_opening)
{
    Gamestate st(boardSize, use_renju, use_omok, seed, use_pro_long_opening);

    nn_ = std::make_shared<BatchingNNInterface>();
    
    // Use the original config but limit max threads to avoid excessive GIL contention
    MCTSConfig adjusted_cfg = cfg;
    
    // Get available CPU cores from the system
    unsigned int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 4; // Fallback if detection fails
    
    // Reserve one thread for Python and one for the main thread
    max_threads = std::max(1u, max_threads - 2);
    
    // Ensure we don't exceed the configuration
    adjusted_cfg.num_threads = std::min(static_cast<int>(max_threads), cfg.num_threads);
    
    // Always use at least 2 threads (one for search, one for evaluation)
    adjusted_cfg.num_threads = std::max(2, adjusted_cfg.num_threads);
    
    MCTS_DEBUG("Creating MCTS with " << adjusted_cfg.num_threads << " threads "
              << "(requested: " << cfg.num_threads << ", system max: " << max_threads << ")");
    
    mcts_ = std::make_unique<MCTS>(adjusted_cfg, nn_, boardSize);

    config_ = adjusted_cfg;
    rootState_ = st;
}

    void set_batch_size(int size) {
        nn_->set_batch_size(size);
    }

    void set_infer_function(py::function pyFn) {
        MCTS_DEBUG("Setting Python inference function");
        if (!nn_) {
            MCTS_DEBUG("Error: NN interface is null");
            return;
        }
        
        // Create a C++ function that wraps the Python function with proper GIL handling
        auto batch_inference_wrapper = [pyFn](const std::vector<std::tuple<std::string, int, float, float>>& inputs) 
            -> std::vector<NNOutput> {
            
            if (inputs.empty()) {
                MCTS_DEBUG("Empty inputs provided to batch_inference_wrapper");
                return {};
            }
            
            MCTS_DEBUG("Batch inference wrapper called with " << inputs.size() << " inputs");
            
            std::vector<NNOutput> results;
            
            try {
                // Convert C++ inputs to Python
                py::list py_inputs;
                
                // Build the input list - THIS MUST HAPPEN BEFORE GIL ACQUISITION
                for (const auto& input : inputs) {
                    py_inputs.append(py::make_tuple(
                        std::get<0>(input),  // state string
                        std::get<1>(input),  // chosen move
                        std::get<2>(input),  // attack
                        std::get<3>(input)   // defense
                    ));
                }
                
                MCTS_DEBUG("Acquiring GIL for Python function call");
                
                // Acquire the GIL explicitly before calling into Python
                py::gil_scoped_acquire acquire;
                
                // Call the Python function with a timeout
                MCTS_DEBUG("Calling Python function");
                
                // Call Python function
                py::object py_results;
                try {
                    py_results = pyFn(py_inputs);
                    MCTS_DEBUG("Python function returned successfully");
                }
                catch (const py::error_already_set& e) {
                    MCTS_DEBUG("Python error during function call: " << e.what());
                    // Return empty results - we'll handle defaults later
                    throw;
                }
                
                // Convert Python results back to C++
                try {
                    if (!py_results || py_results.is_none()) {
                        MCTS_DEBUG("Python function returned None");
                        return {};
                    }
                    
                    for (auto item : py_results) {
                        NNOutput output;
                        
                        // Extract policy and value from the Python tuple
                        py::tuple result_tuple = item.cast<py::tuple>();
                        py::list policy_list = result_tuple[0].cast<py::list>();
                        float value = result_tuple[1].cast<float>();
                        
                        // Convert policy list to vector
                        output.policy.reserve(policy_list.size());
                        for (auto p : policy_list) {
                            output.policy.push_back(p.cast<float>());
                        }
                        output.value = value;
                        results.push_back(std::move(output));
                    }
                    
                    MCTS_DEBUG("Converted " << results.size() << " results from Python");
                }
                catch (const std::exception& e) {
                    MCTS_DEBUG("Error converting Python results: " << e.what());
                    // Return empty results
                    return {};
                }
                
                // GIL is automatically released when acquire goes out of scope
            } 
            catch (const py::error_already_set& e) {
                MCTS_DEBUG("Python error: " << e.what());
                // Return empty results - BatchingNNInterface will handle defaults
            } 
            catch (const std::exception& e) {
                MCTS_DEBUG("C++ error in Python callback: " << e.what());
                // Return empty results - BatchingNNInterface will handle defaults
            }
            
            return results;
        };
        
        // Set the inference callback on the NN interface
        MCTS_DEBUG("Setting inference callback on NN interface");
        nn_->set_infer_callback(batch_inference_wrapper);
        MCTS_DEBUG("Python inference function set successfully");
    }

    void run_search() {
        MCTS_DEBUG("MCTSWrapper::run_search called");
        
        // Reset the NN interface before each search
        if (nn_) {
            nn_->reset();
        }
        
        // Run the search
        mcts_->run_search(rootState_);
        
        MCTS_DEBUG("MCTSWrapper::run_search completed");
    }

    int best_move() const {
        return mcts_->select_move();
    }

    void apply_best_move() {
        int mv = mcts_->select_move();
        if (mv >= 0) {
            rootState_.make_move(mv, rootState_.current_player);
        }
    }

    bool is_terminal() const {
        return rootState_.is_terminal();
    }

    int get_winner() const {
        return rootState_.get_winner();
    }

    int best_move_with_temperature(float temperature = 1.0f) const {
        return mcts_->select_move_with_temperature(temperature);
    }
    
    void apply_best_move_with_temperature(float temperature = 1.0f) {
        int mv = mcts_->select_move_with_temperature(temperature);
        if (mv >= 0) {
            rootState_.make_move(mv, rootState_.current_player);
        }
    }
    
    void set_exploration_parameters(float dirichlet_alpha, float noise_weight) {
        mcts_->set_dirichlet_alpha(dirichlet_alpha);
        mcts_->set_noise_weight(noise_weight);
    }

    void set_num_history_moves(int num_moves) {
        nn_->set_num_history_moves(num_moves);
    }
    
    int get_num_history_moves() const {
        return nn_->get_num_history_moves();
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
       .def("set_num_history_moves", &MCTSWrapper::set_num_history_moves)
       .def("get_num_history_moves", &MCTSWrapper::get_num_history_moves)
       .def("best_move_with_temperature", &MCTSWrapper::best_move_with_temperature,
            py::arg("temperature") = 1.0f)
        .def("apply_best_move_with_temperature", &MCTSWrapper::apply_best_move_with_temperature,
            py::arg("temperature") = 1.0f)
        .def("set_exploration_parameters", &MCTSWrapper::set_exploration_parameters,
            py::arg("dirichlet_alpha") = 0.03f, py::arg("noise_weight") = 0.25f);
}
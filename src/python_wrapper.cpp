// python_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gomoku.h"
#include "attack_defense.h"
#include "mcts_config.h"
#include "mcts.h"
#include "python_nn_proxy.h"
#include "nn_interface.h"
#include <iostream>
#include <csignal>

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
        try {
            // Initialize board state
            Gamestate st(boardSize, use_renju, use_omok, seed, use_pro_long_opening);
            
            // Create neural network proxy
            nn_ = std::make_shared<PythonNNProxy>();
            
            // Use a limited number of threads - start conservatively
            MCTSConfig adjusted_cfg = cfg;
            
            // Limit threads to a reasonable number (2-8) based on total available
            unsigned int max_system_threads = std::thread::hardware_concurrency();
            if (max_system_threads == 0) max_system_threads = 4; // Fallback
            
            // Use at most half of available threads, minimum 2, maximum 8
            int suggested_threads = std::max(2, static_cast<int>(max_system_threads / 2));
            suggested_threads = std::min(suggested_threads, 8);
            
            // Cap at user-requested thread count
            adjusted_cfg.num_threads = std::min(suggested_threads, cfg.num_threads);
            
            // Ensure reasonable batch size
            if (adjusted_cfg.parallel_leaf_batch_size <= 0) {
                adjusted_cfg.parallel_leaf_batch_size = 16;
            } else {
                // Cap at a reasonable maximum
                adjusted_cfg.parallel_leaf_batch_size = std::min(adjusted_cfg.parallel_leaf_batch_size, 64);
            }
            
            MCTS_DEBUG("Creating MCTS with semi-parallel mode: " << adjusted_cfg.num_threads 
                    << " threads, batch size " << adjusted_cfg.parallel_leaf_batch_size);
            
            // Create MCTS engine
            mcts_ = std::make_unique<MCTS>(adjusted_cfg, nn_, boardSize);
            
            // Store configuration
            config_ = adjusted_cfg;
            rootState_ = st;
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Exception in MCTSWrapper constructor: " << e.what());
            throw;
        }
    }

    ~MCTSWrapper() {
        MCTS_DEBUG("MCTSWrapper destructor called");
        
        // Explicit shutdown in the correct order with timeouts
        try {
            // Set global shutdown flag to stop any ongoing operations
            global_shutdown_requested.store(true, std::memory_order_release);
            
            // Also set the Python shutdown flag if available
            if (nn_) {
                // Cast to PythonNNProxy to access shutdown flag
                auto python_nn = dynamic_cast<PythonNNProxy*>(nn_.get());
                if (python_nn) {
                    // Trigger the shutdown flag in Python proxy
                    shutdown_in_progress.store(true, std::memory_order_release);
                }
            }
            
            // First, set flags to stop any ongoing search
            if (mcts_) {
                MCTS_DEBUG("Setting MCTS shutdown flag");
                mcts_->set_shutdown_flag(true);
                
                // Clear the leaf gatherer first with very short timeout
                if (mcts_->get_leaf_gatherer()) {
                    MCTS_DEBUG("Shutting down leaf gatherer");
                    
                    // Use a thread with timeout to shutdown the leaf gatherer
                    std::atomic<bool> leaf_shutdown_complete{false};
                    std::thread leaf_shutdown_thread([this, &leaf_shutdown_complete](){
                        try {
                            mcts_->clear_leaf_gatherer();
                            leaf_shutdown_complete.store(true, std::memory_order_release);
                        } catch (...) {
                            // Ignore errors
                        }
                    });
                    
                    // Very short wait - mostly fire and forget
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    
                    // Detach thread regardless - don't join
                    if (leaf_shutdown_thread.joinable()) {
                        MCTS_DEBUG("Detaching leaf gatherer shutdown thread");
                        leaf_shutdown_thread.detach();
                    }
                }
            }
            
            // Clear MCTS with fire-and-forget approach
            MCTS_DEBUG("Clearing MCTS engine");
            std::thread mcts_thread([this](){
                try {
                    mcts_.reset();
                } catch (...) {
                    // Ignore errors
                }
            });
            
            // Don't wait - just detach
            MCTS_DEBUG("Detaching MCTS reset thread to avoid deadlocks");
            mcts_thread.detach();
            
            // Force the pointer to null to ensure destructor continues
            mcts_ = nullptr;
            
            // Finally, shutdown neural network with fire-and-forget approach
            MCTS_DEBUG("Shutting down neural network interface");
            std::thread nn_thread([this](){
                try {
                    nn_.reset();
                } catch (...) {
                    // Ignore errors
                }
            });
            
            // Don't wait - just detach
            MCTS_DEBUG("Detaching neural network shutdown thread to avoid deadlocks");
            nn_thread.detach();
            
            // Force the pointer to null to ensure destructor continues
            nn_ = nullptr;
            
            MCTS_DEBUG("MCTSWrapper shutdown complete");
        } 
        catch (...) {
            MCTS_DEBUG("Exception in MCTSWrapper destructor");
            
            // Force cleanup to avoid memory leaks
            try {
                mcts_ = nullptr;
                nn_ = nullptr;
            } catch (...) {
                // Last resort - ignore errors
            }
        }
    }

    void set_infer_function(py::object model) {
        MCTS_DEBUG("Setting neural network model");
        
        if (!nn_) {
            MCTS_DEBUG("Error: NN proxy is null");
            return;
        }
        
        try {
            // First, make sure any previous neural network is properly shut down
            auto python_nn = dynamic_cast<PythonNNProxy*>(nn_.get());
            if (python_nn) {
                // Explicitly shut down before setting a new model
                MCTS_DEBUG("Shutting down previous neural network model");
                python_nn->shutdown();
            }
            
            // Wait a brief moment to ensure cleanup is complete
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Initialize the neural network proxy with the model
            bool success = nn_->initialize(model, config_.parallel_leaf_batch_size);
            
            if (success) {
                MCTS_DEBUG("Neural network model set successfully");
            } else {
                MCTS_DEBUG("Failed to initialize neural network proxy");
            }
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error setting neural network model: " << e.what());
        }
    }

    void reset_neural_network() {
        MCTS_DEBUG("Resetting neural network");
        
        try {
            // Create a new neural network proxy
            auto new_nn = std::make_shared<PythonNNProxy>();
            
            // Preserve any settings from the old proxy
            if (nn_) {
                auto old_nn = dynamic_cast<PythonNNProxy*>(nn_.get());
                if (old_nn) {
                    new_nn->set_num_history_moves(old_nn->get_num_history_moves());
                }
            }
            
            // Shut down the old proxy
            if (nn_) {
                MCTS_DEBUG("Shutting down old neural network");
                auto old_nn = dynamic_cast<PythonNNProxy*>(nn_.get());
                if (old_nn) {
                    old_nn->shutdown();
                }
            }
            
            // Wait a brief moment to ensure cleanup is complete
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Replace with the new proxy
            nn_ = new_nn;
            MCTS_DEBUG("Neural network reset complete");
        }
        catch (const std::exception& e) {
            MCTS_DEBUG("Error resetting neural network: " << e.what());
        }
    }

    void set_batch_size(int size) {
        if (!nn_) {
            MCTS_DEBUG("Error: NN proxy is null");
            return;
        }
        
        int capped_size = std::min(std::max(1, size), 64);  // Between 1 and 64
        MCTS_DEBUG("Setting batch size to " << capped_size);
        
        // Also update the MCTS config
        config_.parallel_leaf_batch_size = capped_size;
    }

    void run_search() {
        MCTS_DEBUG("MCTSWrapper::run_search called");
        
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
        if (!nn_) {
            MCTS_DEBUG("Error: NN proxy is null");
            return;
        }
        
        MCTS_DEBUG("Setting history moves to " << num_moves);
        nn_->set_num_history_moves(num_moves);
    }
    
    int get_num_history_moves() const {
        return nn_->get_num_history_moves();
    }

    std::string get_stats() const {
        std::string stats = "MCTSWrapper stats:\n";
        
        // Add configuration information
        stats += "  Threads: " + std::to_string(config_.num_threads) + "\n";
        stats += "  Batch size: " + std::to_string(config_.parallel_leaf_batch_size) + "\n";
        stats += "  C_puct: " + std::to_string(config_.c_puct) + "\n";
        
        // Add board state information
        stats += "  Board size: " + std::to_string(rootState_.board_size) + "\n";
        stats += "  Current player: " + std::to_string(rootState_.current_player) + "\n";
        
        // Add neural network stats if available
        if (nn_) {
            stats += nn_->get_stats();
        }
        
        return stats;
    }

    // Signal handler for graceful termination
    static std::atomic<bool> global_shutdown_requested;
    static void signal_handler(int sig) {
        MCTS_DEBUG("Received signal " << sig << ", initiating graceful shutdown");
        global_shutdown_requested.store(true, std::memory_order_release);
    }

private:
    MCTSConfig config_;
    std::shared_ptr<PythonNNProxy> nn_;  // Changed from BatchingNNInterface
    std::unique_ptr<MCTS> mcts_;

    Gamestate rootState_;
};

// Define and initialize the static member
std::atomic<bool> MCTSWrapper::global_shutdown_requested{false};

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
       // Changed from set_infer_function to accept a model object directly
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
            py::arg("dirichlet_alpha") = 0.03f, py::arg("noise_weight") = 0.25f)
        .def("reset_neural_network", &MCTSWrapper::reset_neural_network)
       .def("get_stats", &MCTSWrapper::get_stats);  // Add stats method binding

    // Register signal handlers for graceful termination
    m.def("register_signal_handlers", []() {
        std::signal(SIGINT, MCTSWrapper::signal_handler);  // Qualify with MCTSWrapper::
        std::signal(SIGTERM, MCTSWrapper::signal_handler); // Qualify with MCTSWrapper::
        MCTS_DEBUG("Signal handlers registered for graceful shutdown");
    });

    // Add a shutdown check function
    m.def("check_shutdown_requested", []() {
        return MCTSWrapper::global_shutdown_requested.load(std::memory_order_acquire); // Qualify with MCTSWrapper::
    });
}
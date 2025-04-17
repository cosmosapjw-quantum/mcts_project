<<<<<<< HEAD
#include "python_nn_proxy.h"

// Define the static variables that were declared in the header
std::recursive_mutex batch_inference_mutex;
std::atomic<bool> shutdown_in_progress{false};
=======
// python_nn_proxy.cpp
#include "python_nn_proxy.h"

// Define the global variables
std::recursive_mutex batch_inference_mutex;
std::atomic<bool> shutdown_in_progress{false};

// No other implementation needed - 
// all methods are defined in the header
>>>>>>> 576f610 (darw bug (incomplete, working))

#include "python_nn_proxy.h"

// Define the static variables that were declared in the header
std::recursive_mutex batch_inference_mutex;
std::atomic<bool> shutdown_in_progress{false};
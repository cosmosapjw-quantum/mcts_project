#define DEBUG_MCTS 1  // Set to 0 to disable debug prints

#if DEBUG_MCTS
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#define MCTS_DEBUG(msg) do { \
    auto now = std::chrono::system_clock::now(); \
    auto time = std::chrono::system_clock::to_time_t(now); \
    std::tm tm = *std::localtime(&time); \
    std::ostringstream oss; \
    oss << "[C++ " << std::put_time(&tm, "%H:%M:%S") << "] " << msg << std::endl; \
    std::cout << oss.str() << std::flush; \
} while(0)
#else
#define MCTS_DEBUG(msg)
#endif
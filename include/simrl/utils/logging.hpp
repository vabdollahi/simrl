#pragma once

#include <array>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace simrl
{

// Constant for timestamp buffer size
constexpr std::size_t kTimestampBufSize = 20;

// Get current timestamp for log messages
inline auto current_timestamp() -> std::string
{
    std::time_t now = std::time(nullptr);
    std::tm *time_info = std::localtime(&now);
    std::array<char, kTimestampBufSize> buf{};
    std::strftime(buf.data(), buf.size(), "%Y-%m-%d %H:%M:%S", time_info);
    return {buf.data()};
}

// Central log function
inline auto log(const std::string &level, const std::string &message) -> void
{
    std::cerr << "[" << current_timestamp() << "] [" << level << "] " << message << "\n";
}

// Use cuda_check() in templates, headers, and utility functions
#ifdef USE_CUDA
inline auto cuda_check(cudaError_t err, const char *msg = nullptr) -> void
{
    if (err != cudaSuccess)
    {
        std::ostringstream oss;
        if (msg)
        {
            oss << msg << ": ";
        }
        oss << cudaGetErrorString(err);
        log("ERROR", oss.str());
        throw std::runtime_error(oss.str());
    }
}
#endif

} // namespace simrl

// -----------------------------
// Logging Macros
// -----------------------------

// CUDA error check macro
// Use SIMRL_CHECK() by default in implementation files (.cpp/.cu)
#ifdef USE_CUDA
#define SIMRL_CHECK(expr)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err__ = (expr);                                                                \
        if (err__ != cudaSuccess)                                                                  \
        {                                                                                          \
            std::ostringstream oss__;                                                              \
            oss__ << "CUDA Error in " << #expr << " at " << __FILE__ << ":" << __LINE__ << ": "    \
                  << cudaGetErrorString(err__);                                                    \
            simrl::log("ERROR", oss__.str());                                                      \
            throw std::runtime_error(oss__.str());                                                 \
        }                                                                                          \
    } while (0)
#else
#define SIMRL_CHECK(expr) (expr)
#endif

// Assertion macro
#define SIMRL_ASSERT(cond, msg)                                                                    \
    do                                                                                             \
    {                                                                                              \
        if (!(cond))                                                                               \
        {                                                                                          \
            std::ostringstream oss__;                                                              \
            oss__ << "Assertion failed: (" << (msg) << ") at " << __FILE__ << ":" << __LINE__;     \
            simrl::log("ERROR", oss__.str());                                                      \
            throw std::runtime_error(oss__.str());                                                 \
        }                                                                                          \
    } while (0)

// Log level helpers
#define SIMRL_INFO(msg) simrl::log("INFO", msg)
#define SIMRL_WARN(msg) simrl::log("WARN", msg)
#define SIMRL_ERROR(msg) simrl::log("ERROR", msg)

cmake_minimum_required(VERSION 3.12)
project(SimRL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Include directories
include_directories(include)

# ---------------------------
# Core Library
# ---------------------------
add_library(simrl STATIC
    src/tensor.cpp
    src/autograd.cpp
    # Add more core source files here as needed
)

target_include_directories(simrl PUBLIC include)

# ---------------------------
# CLI Executable (Optional)
# ---------------------------
option(BUILD_SIMRL_CLI "Build the simrl_cli executable" ON)

if(BUILD_SIMRL_CLI AND EXISTS "${CMAKE_SOURCE_DIR}/src/simrl_cli.cpp")
    add_executable(simrl_cli src/simrl_cli.cpp)
    target_link_libraries(simrl_cli PRIVATE simrl)
endif()

# ---------------------------
# Testing
# ---------------------------
enable_testing()
add_subdirectory(tests)

# ---------------------------
# Python Bindings (Future)
# ---------------------------
# find_package(pybind11 REQUIRED)
# add_subdirectory(python)

cmake_minimum_required(VERSION 3.12)
project(SimRL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Optional flags
option(USE_CUDA "Enable CUDA support" OFF)
option(BUILD_SIMRL_CLI "Build the simrl_cli executable" ON)

# Include directories
include_directories(include)

# Core library
add_library(simrl STATIC
    src/tensor.cpp
    src/autograd.cpp
    # Add more core source files here
)
target_include_directories(simrl PUBLIC include)

# CUDA configuration (no test binaries here)
if(USE_CUDA)
  if(APPLE)
    message(WARNING "CUDA requested, but macOS does not support CUDA. Skipping CUDA setup.")
    target_compile_definitions(simrl PUBLIC USE_CUDA)
  else()
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 86)
    target_compile_definitions(simrl PUBLIC USE_CUDA)
    target_link_libraries(simrl PUBLIC CUDA::cudart)
    message(STATUS "CUDA support enabled")
  endif()
else()
  message(STATUS "CUDA support disabled")
endif()

# CLI executable
if(BUILD_SIMRL_CLI AND EXISTS "${CMAKE_SOURCE_DIR}/src/simrl_cli.cpp")
  add_executable(simrl_cli src/simrl_cli.cpp)
  target_link_libraries(simrl_cli PRIVATE simrl)
endif()

# Unit tests
enable_testing()
add_subdirectory(tests)

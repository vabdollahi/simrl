cmake_minimum_required(VERSION 3.12)
project(simrl)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(simrl src/tensor.cpp src/autograd.cpp)

enable_testing()
add_subdirectory(tests)

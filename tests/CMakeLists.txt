enable_testing()

# Catch2 setup (assumes installed or provided via CMake FetchContent later)
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE simrl)
add_test(NAME test_tensor COMMAND test_tensor)

# If Python bindings are added later:
# find_package(pybind11 REQUIRED)
# add_subdirectory(python)

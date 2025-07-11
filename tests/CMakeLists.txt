# CPU tests
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE simrl)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_tensor_utils test_tensor_utils.cpp)
target_link_libraries(test_tensor_utils PRIVATE simrl)
add_test(NAME test_tensor_utils COMMAND test_tensor_utils)

# GPU tests (only if CUDA enabled and not macOS)
if(USE_CUDA AND NOT APPLE)
    add_executable(test_tensor_cuda test_tensor_cuda.cu)
    target_link_libraries(test_tensor_cuda PRIVATE simrl)
    add_test(NAME test_tensor_cuda COMMAND test_tensor_cuda)
    set_tests_properties(test_tensor_cuda PROPERTIES LABELS "gpu")

    add_executable(test_tensor_utils_cuda test_tensor_utils_cuda.cu)
    target_link_libraries(test_tensor_utils_cuda PRIVATE simrl)
    add_test(NAME test_tensor_utils_cuda COMMAND test_tensor_utils_cuda)
    set_tests_properties(test_tensor_utils_cuda PROPERTIES LABELS "gpu")
endif()

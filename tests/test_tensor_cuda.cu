// tests/test_tensor_cuda.cu
#include "simrl/tensor.hpp"
#include <cassert>
#include <cuda_runtime.h>

__global__ void check_tensor_kernel(const size_t* shape, size_t* result) {
    if (threadIdx.x == 0) {
        // Check shape is {2, 3, 4}
        result[0] = (shape[0] == 2);
        result[1] = (shape[1] == 3);
        result[2] = (shape[2] == 4);
    }
}

auto main() -> int {
    using namespace simrl;

    Tensor tensor({2, 3, 4}, DType::Float32, DeviceType::CUDA);

    assert(tensor.dtype() == DType::Float32);
    assert(tensor.device() == DeviceType::CUDA);

    // Copy shape to GPU
    size_t host_shape[3] = {tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]};
    size_t *device_shape, *device_result;
    cudaMalloc(&device_shape, 3 * sizeof(size_t));
    cudaMalloc(&device_result, 3 * sizeof(size_t));
    cudaMemcpy(device_shape, host_shape, 3 * sizeof(size_t), cudaMemcpyHostToDevice);

    check_tensor_kernel<<<1, 1>>>(device_shape, device_result);

    size_t host_result[3];
    cudaMemcpy(host_result, device_result, 3 * sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaFree(device_shape);
    cudaFree(device_result);

    assert(host_result[0] == 1);
    assert(host_result[1] == 1);
    assert(host_result[2] == 1);

    return 0;
}

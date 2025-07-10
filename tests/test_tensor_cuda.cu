#include "simrl/tensor.hpp"
#include "simrl/utils/logging.hpp"

#include <cuda_runtime.h>
#include <cassert>

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

    try {
        SIMRL_INFO("Creating tensor on CUDA device...");
        Tensor tensor({2, 3, 4}, DType::Float32, DeviceType::CUDA);

        SIMRL_ASSERT(tensor.dtype() == DType::Float32, "Tensor dtype mismatch");
        SIMRL_ASSERT(tensor.device() == DeviceType::CUDA, "Tensor device mismatch");

        // Copy shape to GPU
        size_t host_shape[3] = {tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]};
        size_t *device_shape = nullptr, *device_result = nullptr;

        SIMRL_CHECK(cudaMalloc(&device_shape, 3 * sizeof(size_t)));
        SIMRL_CHECK(cudaMalloc(&device_result, 3 * sizeof(size_t)));
        SIMRL_CHECK(cudaMemcpy(device_shape, host_shape, 3 * sizeof(size_t), cudaMemcpyHostToDevice));

        SIMRL_INFO("Launching CUDA kernel...");
        check_tensor_kernel<<<1, 1>>>(device_shape, device_result);
        SIMRL_CHECK(cudaGetLastError());
        SIMRL_CHECK(cudaDeviceSynchronize());

        size_t host_result[3] = {};
        SIMRL_CHECK(cudaMemcpy(host_result, device_result, 3 * sizeof(size_t), cudaMemcpyDeviceToHost));

        SIMRL_CHECK(cudaFree(device_shape));
        SIMRL_CHECK(cudaFree(device_result));

        SIMRL_ASSERT(host_result[0] == 1, "Shape[0] check failed");
        SIMRL_ASSERT(host_result[1] == 1, "Shape[1] check failed");
        SIMRL_ASSERT(host_result[2] == 1, "Shape[2] check failed");

        SIMRL_INFO("CUDA tensor test passed successfully.");
    } catch (const std::exception &e) {
        SIMRL_ERROR(e.what());
        return 1;
    }

    return 0;
}
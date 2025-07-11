#include "simrl/tensor.hpp"
#include "simrl/utils/logging.hpp"
#include <cuda_runtime.h>

__global__ void validate_zero_kernel(float* data, size_t numel, bool* result) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < numel) {
        if (data[i] != 0.0f) *result = false;
    }
}

auto main() -> int {
    using namespace simrl;

    try {
        SIMRL_INFO("Creating CUDA tensor...");
        Tensor t({4, 4, 2}, DType::Float32, DeviceType::CUDA);
        t.zero();

        bool* dev_result;
        bool host_result = true;
        cudaMalloc(&dev_result, sizeof(bool));
        cudaMemcpy(dev_result, &host_result, sizeof(bool), cudaMemcpyHostToDevice);

        validate_zero_kernel<<<1, 32>>>(reinterpret_cast<float*>(t.data()), 32, dev_result);
        cudaMemcpy(&host_result, dev_result, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(dev_result);

        SIMRL_ASSERT(host_result, "CUDA tensor zero() failed");

        SIMRL_INFO("✅ All CUDA tensor utility tests passed.");
    } catch (const std::exception& e) {
        SIMRL_ERROR(e.what());
        return 1;
    }

    return 0;
}

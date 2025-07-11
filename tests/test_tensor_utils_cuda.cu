#include "simrl/tensor.hpp"
#include "simrl/utils/logging.hpp"
#include <cuda_runtime.h>

using namespace simrl;

constexpr size_t DIM_0 = 6;
constexpr size_t DIM_1 = 4;
constexpr size_t NUM_ELEMENTS = DIM_0 * DIM_1;

__global__ void validate_zero_kernel(float* data, size_t numel, bool* result) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < numel && data[i] != 0.0f) {
        *result = false;
    }
}

void test_zero_cuda() {
    SIMRL_INFO("Testing zero() on CUDA tensor...");
    Tensor tensor({DIM_0, DIM_1}, DType::Float32, DeviceType::CUDA);
    tensor.zero();

    bool* device_result;
    bool host_result = true;
    SIMRL_CHECK(cudaMalloc(&device_result, sizeof(bool)));
    SIMRL_CHECK(cudaMemcpy(device_result, &host_result, sizeof(bool), cudaMemcpyHostToDevice));

    validate_zero_kernel<<<(NUM_ELEMENTS + 31) / 32, 32>>>(
        tensor.as<float>(), NUM_ELEMENTS, device_result
    );
    SIMRL_CHECK(cudaGetLastError());  // Check for kernel launch errors

    SIMRL_CHECK(cudaMemcpy(&host_result, device_result, sizeof(bool), cudaMemcpyDeviceToHost));
    SIMRL_CHECK(cudaFree(device_result));

    SIMRL_ASSERT(host_result, "CUDA zero() failed");
}

void test_copy_from_cuda() {
    SIMRL_INFO("Testing copy_from() on CUDA tensor...");
    Tensor src({DIM_0, DIM_1}, DType::Float32, DeviceType::CUDA);
    Tensor dst({DIM_0, DIM_1}, DType::Float32, DeviceType::CUDA);
    src.zero();
    dst.copy_from(src);

    bool* device_result;
    bool host_result = true;
    SIMRL_CHECK(cudaMalloc(&device_result, sizeof(bool)));
    SIMRL_CHECK(cudaMemcpy(device_result, &host_result, sizeof(bool), cudaMemcpyHostToDevice));

    validate_zero_kernel<<<(NUM_ELEMENTS + 31) / 32, 32>>>(
        dst.as<float>(), NUM_ELEMENTS, device_result
    );
    SIMRL_CHECK(cudaGetLastError());

    SIMRL_CHECK(cudaMemcpy(&host_result, device_result, sizeof(bool), cudaMemcpyDeviceToHost));
    SIMRL_CHECK(cudaFree(device_result));

    SIMRL_ASSERT(host_result, "CUDA copy_from() failed");
}

// test tensor clone functionality
void test_clone_cuda() {
    SIMRL_INFO("Testing clone() on CUDA tensor...");
    Tensor original({DIM_0, DIM_1}, DType::Float32, DeviceType::CUDA);
    original.zero();
    Tensor clone = original.clone();

    SIMRL_ASSERT(clone.is_cuda(), "Clone should be on CUDA device");
    SIMRL_ASSERT(clone.shape() == original.shape(), "Clone shape mismatch");
    SIMRL_ASSERT(clone.dtype() == original.dtype(), "Clone dtype mismatch");
    SIMRL_ASSERT(clone.device() == original.device(), "Clone device mismatch");

    bool* device_result;
    bool host_result = true;
    
    SIMRL_CHECK(cudaMalloc(&device_result, sizeof(bool)));
    SIMRL_CHECK(cudaMemcpy(device_result, &host_result, sizeof(bool), cudaMemcpyHostToDevice));
    validate_zero_kernel<<<(NUM_ELEMENTS + 31) / 32, 32>>>(
        clone.as<float>(), NUM_ELEMENTS, device_result
    );
    SIMRL_CHECK(cudaGetLastError());
    SIMRL_CHECK(cudaMemcpy(&host_result, device_result, sizeof(bool), cudaMemcpyDeviceToHost));
    SIMRL_CHECK(cudaFree(device_result));

    SIMRL_ASSERT(host_result, "CUDA clone() failed");
}

auto main() -> int {
    try {
        test_zero_cuda();
        test_copy_from_cuda();
        SIMRL_INFO("✅ All CUDA tensor utility tests passed.");
    } catch (const std::exception& e) {
        SIMRL_ERROR(e.what());
        return 1;
    }
    return 0;
}

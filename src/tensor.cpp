#include "simrl/tensor.hpp"
#include "simrl/utils/logging.hpp"

#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace simrl
{

namespace
{
[[nodiscard]] auto dtype_size(DType dtype) -> size_t
{
    switch (dtype)
    {
    case DType::Float32:
    case DType::Int32:
        return 4;
    case DType::Bool:
        return 1;
    default:
        SIMRL_ERROR("Unsupported DType enum value");
        throw std::runtime_error("Unsupported DType");
    }
}
} // namespace

Tensor::Tensor(const std::vector<size_t> &shape, DType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device)
{
    SIMRL_ASSERT(!shape.empty(), "Tensor shape must be non-empty");

    compute_stride();
    numel_ = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<>());
    allocate();
}

Tensor::~Tensor()
{
    if (data_ != nullptr)
    {
        if (device_ == DeviceType::CPU)
        {
            free(data_);
        }
#ifdef USE_CUDA
        else if (device_ == DeviceType::CUDA)
        {
            SIMRL_CHECK(cudaFree(data_));
        }
#endif
    }
}

void Tensor::compute_stride()
{
    stride_.resize(shape_.size());
    size_t acc = 1;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape_.size()) - 1; i >= 0; --i)
    {
        stride_[i] = acc;
        acc *= shape_[i];
    }
}

void Tensor::allocate()
{
    size_t total_bytes = numel_ * dtype_size(dtype_);

    if (device_ == DeviceType::CPU)
    {
        data_ = malloc(total_bytes);
        SIMRL_ASSERT(data_ != nullptr, "CPU malloc failed");
    }
#ifdef USE_CUDA
    else if (device_ == DeviceType::CUDA)
    {
        SIMRL_CHECK(cudaMalloc(&data_, total_bytes));
    }
#endif
    else
    {
        SIMRL_ASSERT(false, "Unknown device type in allocate()");
    }
}

void Tensor::zero()
{
    size_t total_bytes = numel_ * dtype_size(dtype_);
    if (device_ == DeviceType::CPU)
    {
        std::memset(data_, 0, total_bytes);
    }
#ifdef USE_CUDA
    else if (device_ == DeviceType::CUDA)
    {
        SIMRL_CHECK(cudaMemset(data_, 0, total_bytes));
    }
#endif
    else
    {
        SIMRL_ASSERT(false, "Unknown device type in zero()");
    }
}

void Tensor::copy_from(const Tensor &other)
{
    SIMRL_ASSERT(shape_ == other.shape_, "Tensor shape mismatch in copy_from()");
    SIMRL_ASSERT(dtype_ == other.dtype_, "Tensor dtype mismatch in copy_from()");
    SIMRL_ASSERT(device_ == other.device_, "Tensor device mismatch in copy_from()");

    size_t total_bytes = numel_ * dtype_size(dtype_);

    if (device_ == DeviceType::CPU)
    {
        std::memcpy(data_, other.data_, total_bytes);
    }
#ifdef USE_CUDA
    else if (device_ == DeviceType::CUDA)
    {
        SIMRL_CHECK(cudaMemcpy(data_, other.data_, total_bytes, cudaMemcpyDeviceToDevice));
    }
#endif
    else
    {
        SIMRL_ASSERT(false, "Unknown device type in copy_from()");
    }
}

} // namespace simrl

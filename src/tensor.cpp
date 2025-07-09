#include "simrl/tensor.hpp"
#include <cstdint>
#include <cstring> // for memcpy
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
        throw std::runtime_error("Unsupported DType");
    }
}
} // namespace

Tensor::Tensor(const std::vector<size_t> &shape, DType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device)
{
    if (shape.empty())
    {
        throw std::invalid_argument("Shape must be non-empty");
    }
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
            cudaFree(data_);
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
        if (data_ == nullptr)
        {
            throw std::runtime_error("CPU malloc failed");
        }
    }
#ifdef USE_CUDA
    else if (device_ == DeviceType::CUDA)
    {
        cudaError_t err = cudaMalloc(&data_, total_bytes);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }
#endif
    else
    {
        throw std::runtime_error("Unknown device type");
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
        cudaMemset(data_, 0, total_bytes);
    }
#endif
    else
    {
        throw std::runtime_error("Unknown device type in zero()");
    }
}

void Tensor::copy_from(const Tensor &other)
{
    if (shape_ != other.shape_ || dtype_ != other.dtype_ || device_ != other.device_)
    {
        throw std::runtime_error("Tensor copy_from: incompatible tensors");
    }
    size_t total_bytes = numel_ * dtype_size(dtype_);

    if (device_ == DeviceType::CPU)
    {
        std::memcpy(data_, other.data_, total_bytes);
    }
#ifdef USE_CUDA
    else if (device_ == DeviceType::CUDA)
    {
        cudaMemcpy(data_, other.data_, total_bytes, cudaMemcpyDeviceToDevice);
    }
#endif
    else
    {
        throw std::runtime_error("Unknown device type in copy_from()");
    }
}

} // namespace simrl

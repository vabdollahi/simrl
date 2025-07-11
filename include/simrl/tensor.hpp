#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace simrl {

enum class DeviceType : std::uint8_t { CPU, CUDA };

enum class DType : std::uint8_t { Float32, Int32, Bool };

class Tensor {
  public:
    Tensor(const std::vector<size_t> &shape, DType dtype = DType::Float32,
           DeviceType device = DeviceType::CPU);
    ~Tensor();

    [[nodiscard]] auto shape() const -> const std::vector<size_t> & { return shape_; }
    [[nodiscard]] auto stride() const -> const std::vector<size_t> & { return stride_; }
    [[nodiscard]] auto dtype() const -> DType { return dtype_; }
    [[nodiscard]] auto device() const -> DeviceType { return device_; }
    [[nodiscard]] auto data() const -> void * { return data_; }

    void zero();
    void copy_from(const Tensor &other);
    void reshape(const std::vector<size_t> &new_shape);
    [[nodiscard]] auto clone() const -> Tensor;

    template <typename T> [[nodiscard]] auto as() -> T * {
        if constexpr (std::is_same_v<T, float>) {
            if (dtype_ != DType::Float32) {
                throw std::runtime_error("Tensor dtype is not Float32");
            }
        } else if constexpr (std::is_same_v<T, int>) {
            if (dtype_ != DType::Int32) {
                throw std::runtime_error("Tensor dtype is not Int32");
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            if (dtype_ != DType::Bool) {
                throw std::runtime_error("Tensor dtype is not Bool");
            }
        } else {
            throw std::runtime_error("Unsupported type in Tensor::as<T>()");
        }
        return static_cast<T *>(data_);
    }

  private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    DType dtype_;
    DeviceType device_;
    void *data_ = nullptr;
    size_t numel_;

    void allocate();
    void compute_stride();
};

} // namespace simrl

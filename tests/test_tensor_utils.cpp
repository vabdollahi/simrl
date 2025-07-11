#include "simrl/tensor.hpp"
#include "simrl/utils/logging.hpp"
#include <cstring>

using namespace simrl;

constexpr size_t DIM_0 = 6;
constexpr size_t DIM_1 = 4;
constexpr size_t NUM_ELEMENTS = DIM_0 * DIM_1;

void test_reshape() {
    SIMRL_INFO("Testing reshape...");
    Tensor tensor({2, 3, 4});
    tensor.reshape({DIM_0, DIM_1});
    const auto &new_shape = tensor.shape();
    SIMRL_ASSERT(new_shape.size() == 2 && new_shape[0] == DIM_0 && new_shape[1] == DIM_1,
                 "Reshape failed: incorrect shape");
}

void test_zero() {
    SIMRL_INFO("Testing zero...");
    Tensor tensor({DIM_0, DIM_1});
    tensor.zero();
    const auto *data_ptr = tensor.as<float>();
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        SIMRL_ASSERT(data_ptr[i] == 0.0F, "Zero failed at index " + std::to_string(i));
    }
}

void test_copy_from() {
    SIMRL_INFO("Testing copy_from...");
    Tensor src({DIM_0, DIM_1});
    Tensor dst({DIM_0, DIM_1});
    src.zero();
    dst.copy_from(src);
    const auto *dst_ptr = dst.as<float>();
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        SIMRL_ASSERT(dst_ptr[i] == 0.0F, "Copy_from failed at index " + std::to_string(i));
    }
}

void test_as_casting() {
    SIMRL_INFO("Testing as<T>() casting...");
    Tensor tensor({DIM_0, DIM_1});
    tensor.zero();
    auto *float_ptr = tensor.as<float>();
    SIMRL_ASSERT(float_ptr != nullptr, "as<float>() should return valid pointer");

    SIMRL_INFO("Testing as<T>() error handling...");
    try {
        [[maybe_unused]] auto *unused = tensor.as<int>();
        SIMRL_ASSERT(false, "as<int>() should have thrown on Float32 tensor");
    } catch (const std::exception &) {
        SIMRL_INFO("Caught expected as<int>() failure");
    }
}

void test_invalid_constructor() {
    SIMRL_INFO("Testing invalid constructor...");
    try {
        Tensor invalid({});
        SIMRL_ASSERT(false, "Tensor({}) should throw");
    } catch (const std::exception &) {
        SIMRL_INFO("Caught expected constructor failure");
    }
}

void test_copy_shape_mismatch() {
    SIMRL_INFO("Testing mismatched copy_from...");
    try {
        Tensor tensor_a({DIM_0, DIM_1});
        Tensor tensor_b({DIM_0, DIM_1 + 1});
        tensor_a.copy_from(tensor_b);
        SIMRL_ASSERT(false, "copy_from with mismatched shape should throw");
    } catch (const std::exception &) {
        SIMRL_INFO("Caught expected copy_from shape mismatch");
    }
}

void test_clone() {
    SIMRL_INFO("Testing clone...");
    Tensor original({DIM_0, DIM_1});
    original.zero();
    Tensor cloned = original.clone();
    SIMRL_ASSERT(cloned.shape() == original.shape(), "Clone shape mismatch");
    const auto *cloned_data = cloned.as<float>();
    const auto *original_data = original.as<float>();
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        SIMRL_ASSERT(cloned_data[i] == original_data[i], "Clone data mismatch at index " + std::to_string(i));
    }
}

auto main() -> int {
    try {
        test_reshape();
        test_zero();
        test_copy_from();
        test_as_casting();
        test_invalid_constructor();
        test_copy_shape_mismatch();

        SIMRL_INFO("✅ All CPU tensor utility tests passed.");
    } catch (const std::exception &e) {
        SIMRL_ERROR(e.what());
        return 1;
    }

    return 0;
}

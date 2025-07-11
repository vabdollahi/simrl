#include "simrl/tensor.hpp"
#include "simrl/utils/logging.hpp"
#include <cstring>

auto main() -> int {
    using namespace simrl;

    constexpr size_t DIM_0 = 6;
    constexpr size_t DIM_1 = 4;
    constexpr size_t NUM_ELEMENTS = DIM_0 * DIM_1;

    try {
        SIMRL_INFO("Testing reshape...");
        Tensor tensor({2, 3, 4});
        tensor.reshape({DIM_0, DIM_1});
        const auto& new_shape = tensor.shape();
        SIMRL_ASSERT(
            new_shape.size() == 2 && new_shape[0] == DIM_0 && new_shape[1] == DIM_1,
            "Reshape failed: incorrect shape");

        SIMRL_INFO("Testing zero...");
        tensor.zero();
        auto* data_ptr = reinterpret_cast<float*>(tensor.data());
        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            SIMRL_ASSERT(data_ptr[i] == 0.0F, "Zero failed at index " + std::to_string(i));
        }

        SIMRL_INFO("Testing copy_from...");
        Tensor src({DIM_0, DIM_1});
        src.zero();
        tensor.copy_from(src);
        auto* dst_ptr = reinterpret_cast<float*>(tensor.data());
        for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
            SIMRL_ASSERT(dst_ptr[i] == 0.0F, "Copy_from failed at index " + std::to_string(i));
        }

        SIMRL_INFO("✅ All CPU tensor utility tests passed.");
    } catch (const std::exception& e) {
        SIMRL_ERROR(e.what());
        return 1;
    }

    return 0;
}

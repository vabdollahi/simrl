#include "simrl/tensor.hpp"
#include "simrl/utils/logging.hpp"

auto main() -> int {
    using namespace simrl;

    try {
        SIMRL_INFO("Creating default tensor (CPU, Float32) with shape {2, 3, 4}...");
        Tensor tensor({2, 3, 4});

        // Check shape
        const auto &shape = tensor.shape();
        SIMRL_ASSERT(shape.size() == 3, "Shape size should be 3");
        SIMRL_ASSERT(shape[0] == 2, "Shape[0] should be 2");
        SIMRL_ASSERT(shape[1] == 3, "Shape[1] should be 3");
        SIMRL_ASSERT(shape[2] == 4, "Shape[2] should be 4");

        // Check default dtype and device
        SIMRL_ASSERT(tensor.dtype() == DType::Float32, "Default dtype should be Float32");
        SIMRL_ASSERT(tensor.device() == DeviceType::CPU, "Default device should be CPU");

        SIMRL_INFO("CPU tensor test passed.");
    } catch (const std::exception &e) {
        SIMRL_ERROR(e.what());
        return 1;
    }

    return 0;
}

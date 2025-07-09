#include "simrl/tensor.hpp"
#include <cassert>

auto main() -> int
{
    using namespace simrl;

    Tensor tensor({2, 3, 4});

    // Check shape
    const auto &shape = tensor.shape();
    assert(shape.size() == 3);
    assert(shape[0] == 2);
    assert(shape[1] == 3);
    assert(shape[2] == 4);

    // Check default dtype and device
    assert(tensor.dtype() == DType::Float32);
    assert(tensor.device() == DeviceType::CPU);

    return 0;
}

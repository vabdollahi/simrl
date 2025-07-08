# SimRL

**SimRL** is a CUDA-native deep learning and reinforcement learning framework for robotics and industrial process control. It features modular tensor/autograd support, GPU-accelerated reinforcement learning loops, pluggable simulation backends, and deployment tooling for edge devices.

---

## 📦 Requirements

### Build Tools
- CMake ≥ 3.12
- GCC/G++ ≥ 9 or Clang
- CUDA Toolkit ≥ 12.0
- Git

### Optional Tools
- Pybind11 (for Python bindings, later)
- Docker (for reproducible builds)
- Google Test or Catch2 (for testing)
- MLflow / Weights & Biases (for MLOps)

---

## ⚙️ Build & Test Instructions

Clone and configure the project:

```bash
git clone https://github.com/yourname/simrl.git
cd simrl
mkdir build && cd build
cmake ..
```

Build the project (this will also build the tests):

```bash
cmake --build .
```

Run the test suite:

```bash
ctest --output-on-failure
```

> 💡 Make sure your test targets are properly defined in `CMakeLists.txt` using `add_executable()` and `add_test()`.

---

## 🧱 Structure

- `src/`: Core C++/CUDA source
- `include/`: Public headers
- `tests/`: Unit and integration tests
- `examples/`: Sample training code
- `docker/`: Dockerfiles and container configs
- `scripts/`: Helper scripts
- `docs/`: Technical documentation

---

## 🔄 License

Apache License 2.0

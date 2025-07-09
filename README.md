# SimRL

**SimRL** is a CUDA-native deep learning and reinforcement learning framework for robotics and industrial process control. It features modular tensor/autograd support, GPU-accelerated reinforcement learning loops, pluggable simulation backends, and deployment tooling for edge devices.

---

## 📦 Requirements

### 🔧 Build Tools
- CMake ≥ 3.12
- GCC/G++ ≥ 9 or Clang (macOS supported)
- CUDA Toolkit ≥ 12.0
- Git

### 🧪 Optional Tools
- Catch2 (unit testing)
- pybind11 (Python bindings, planned)
- Docker (for reproducible builds)
- MLflow / Weights & Biases (for MLOps integration)

---

## ⚙️ Build & Test

Clone and configure the project:

```bash
git clone https://github.com/yourname/simrl.git
cd simrl
mkdir -p build && cd build
cmake ..
```

Build the project and tests:

```bash
cmake --build .
```

Run the test suite:

```bash
ctest --output-on-failure
```

---

## 🚀 Run the CLI

```bash
./build/simrl_cli
```

The CLI will be expanded to support training, evaluation, and config loading in future phases.

---

## 🧹 Linting

Lint and format your code using:

```bash
./scripts/lint.sh
```

- Uses `clang-format` and `clang-tidy`
- Automatically installs tools on macOS via Homebrew

---

## 🧱 Project Structure

```
simrl/
├── src/                # Core C++/CUDA source (tensor, autograd, RL, CLI)
├── include/            # Public headers
├── tests/              # Unit and integration tests (Catch2)
├── scripts/            # Linting, build, and dev tools
├── examples/           # Training examples (future)
├── docker/             # Docker and container config (future)
├── docs/               # Technical docs and setup guides
├── CMakeLists.txt      # CMake build config
└── README.md
```

---

## 💡 Roadmap Highlights

- [x] CLI executable for training/testing
- [x] Unit tests with Catch2
- [x] Clang-based linting and CI
- [ ] Core tensor & autograd engine (CUDA-backed)
- [ ] PPO and rollout worker (GPU)
- [ ] Simulator integration (Isaac Sim, ROS 2)
- [ ] Distributed & edge deployment (Jetson Orin)

---

## 📘 Doxygen Documentation

This project uses [Doxygen](https://www.doxygen.nl/) to generate C++ API documentation from annotated source code.

### 🔧 Setup Instructions

1. **Install Doxygen** (macOS):
   ```bash
   brew install doxygen
   ```

2. **Generate documentation**:
   ```bash
   doxygen Doxyfile
   ```

3. **View the docs**:
   Open `docs/doxygen/html/index.html` in a browser, or find it [online](https://vabdollahi.github.io/simrl/).

---

## 🔄 License

Apache License 2.0

# Architecture Overview: SimRL

This document outlines the modular architecture of SimRL, including the core components, extensibility strategy, and execution flow.

---

## 🧱 Core Components

### 1. `libtensor`
- CUDA-based tensor engine
- Shape/stride-aware indexing
- Fused kernels for efficiency

### 2. `autograd`
- Reverse-mode autodiff
- Operation tape and gradient dispatch

### 3. `env`
- EnvInterface for simulations
- Backends for CartPole, Isaac Sim, FMUs

### 4. `rl`
- PPO, SAC training loops
- Rollout workers on GPU

### 5. `quant`
- TensorRT inference export
- INT8 and FP16 optimization

### 6. `deploy`
- Inference runtime for Jetson/IPC
- Safety hooks and real-time I/O

---

## 🔌 Plugin Interfaces

| Interface         | Purpose                         |
|------------------|----------------------------------|
| ModelInterface    | Extend/customize neural nets    |
| EnvInterface      | Wrap new simulations/sensors    |
| TrainerInterface  | Add new RL algorithms           |

---

## 🔁 Execution Flow

1. Launch CLI or Python interface
2. Load config, env, model
3. Run training or deployment loop
4. Log metrics and export results

---

## 🔧 Extensibility
- Add new environments via `env_interface.hpp`
- Plug in PyTorch models with ONNX or native C++
- Extend training algorithms under `trainer_interface.hpp`

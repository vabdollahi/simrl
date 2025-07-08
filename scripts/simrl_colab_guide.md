# 🚀 Steps to Run SimRL on Colab

📁 **Make sure you've uploaded the project directory (e.g., `simrl.zip`) to Colab or Google Drive.**
To compress the project (exclude the build directory):
```
zip -r simrl.zip simrl -x "simrl/build/*"
```
---

## 🔧 1. Unzip Project

```bash
!unzip "./simrl.zip" -d /content
%cd simrl
```

---

## ⚙️ 2. Install Build Tools (if needed)

Colab already includes:

- `gcc`, `g++`
- `cmake`
- `nvcc` (CUDA 11.8)

But if anything is missing:

```bash
!sudo apt-get update
!sudo apt-get install -y build-essential cmake
```

---

## 🧱 3. Build the Project

```bash
!mkdir -p build
!cd build && cmake .. && make
```

---

## 🧪 4. Run the Tests

```bash
!cd build && ctest --output-on-failure
```

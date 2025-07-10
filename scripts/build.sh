#!/bin/bash
set -e

# Check for argument
if [ "$1" == "cuda" ]; then
    echo "🚀 Building with CUDA support..."
    CMAKE_FLAGS="-DUSE_CUDA=ON"
else
    echo "⚙️ Building for CPU only..."
    CMAKE_FLAGS=""
fi

echo "🧹 Cleaning old build directory..."
rm -rf build

echo "📁 Creating new build directory..."
mkdir -p build && cd build

echo "⚙️ Configuring project with CMake... $CMAKE_FLAGS"
cmake .. $CMAKE_FLAGS

echo "🔨 Building the project..."
cmake --build .

echo "✅ Build completed successfully!"

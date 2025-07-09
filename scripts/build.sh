#!/bin/bash
set -e

echo "🧹 Cleaning old build directory..."
rm -rf build

echo "📁 Creating new build directory..."
mkdir -p build && cd build

echo "⚙️ Configuring project with CMake..."
cmake ..

echo "🔨 Building the project..."
cmake --build .

echo "✅ Build completed successfully!"
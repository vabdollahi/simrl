#!/bin/bash
set -e

if [ "$1" == "cuda" ]; then
  echo "⚙️ Rebuilding with CUDA support..."
  rm -rf build
  mkdir -p build && cd build
  cmake .. -DUSE_CUDA=ON
  cmake --build .
  echo "🧪 Running GPU tests..."
  ctest -L gpu --output-on-failure
else
  echo "🧪 Running CPU tests..."
  cd build
  ctest --output-on-failure
fi

#!/bin/bash
set -e

# Default flags
VERBOSE_FLAG=""
MODE="cpu"

# Parse arguments
for arg in "$@"; do
  case $arg in
    cuda|--cuda)
      MODE="cuda"
      ;;
    -v|--verbose)
      VERBOSE_FLAG="-V"
      ;;
    *)
      echo "❌ Unknown argument: $arg"
      echo "Usage: ./test.sh [--cuda] [--verbose]"
      exit 1
      ;;
  esac
done

if [ "$MODE" == "cuda" ]; then
  echo "⚙️ Rebuilding with CUDA support..."
  rm -rf build
  mkdir -p build && cd build
  cmake .. -DUSE_CUDA=ON
  cmake --build .
  echo "🧪 Running GPU tests..."
  ctest -L gpu --output-on-failure $VERBOSE_FLAG
else
  echo "🧪 Running CPU tests..."
  cd build
  ctest --output-on-failure $VERBOSE_FLAG
fi

#!/bin/bash
set -e

echo "🧪 Running tests..."
cd build
ctest --output-on-failure
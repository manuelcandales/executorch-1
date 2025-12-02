#!/bin/bash
# Build and run the SDPA unit test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out"

echo "Building op_sdpa_test..."
cd "$BUILD_DIR"
cmake --build . --target op_sdpa_test

echo ""
echo "Running op_sdpa_test..."
./backends/apple/metal/op_sdpa_test


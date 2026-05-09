#!/usr/bin/env bash
# Build llama-cpp-python from source with ROCm/HIP support.
# Adjust CMAKE_HIP_ARCHITECTURES to match your GPU (gfx1201 = RX 9070 XT).
set -euo pipefail

ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx1201}"

echo "Building llama-cpp-python for HIP architecture: $ARCH"

CMAKE_ARGS="-DGGML_HIPBLAS=on -DCMAKE_HIP_ARCHITECTURES=${ARCH}" \
FORCE_CMAKE=1 \
pip install llama-cpp-python \
  --no-binary llama-cpp-python \
  --upgrade \
  "$@"

echo "Done."

#!/usr/bin/env bash
# Build llama-cpp-python from source with ROCm/HIP support.
# Adjust CMAKE_HIP_ARCHITECTURES to match your GPU (gfx1201 = RX 9070 XT).
set -euo pipefail

ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx1201}"
ROCM_ROOT="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"

echo "Building llama-cpp-python for HIP architecture: $ARCH"
echo "ROCm root: $ROCM_ROOT"

CMAKE_ARGS="-DGGML_HIP=on -DCMAKE_HIP_ARCHITECTURES=${ARCH} -DCMAKE_PREFIX_PATH=${ROCM_ROOT}" \
HIP_PATH="${ROCM_ROOT}" \
FORCE_CMAKE=1 \
pip install llama-cpp-python \
  --no-binary llama-cpp-python \
  --force-reinstall \
  --no-cache-dir \
  "$@"

echo "Done."

#!/usr/bin/env bash
# Run the Go2 RGB/JEPA vector-memory controller through ROCm-capable Python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TINYQUAD_JEPA_PYTHON:-/home/andrewknowles/TinyQuadJEPA/bin/python}"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm-7.1.1}"

if [[ -d "$ROCM_PREFIX" ]]; then
  export ROCM_PATH="${ROCM_PATH:-$ROCM_PREFIX}"
  if [[ -d "$ROCM_PREFIX/lib/llvm/bin" ]]; then
    export PATH="$ROCM_PREFIX/lib/llvm/bin:$PATH"
  fi
  if [[ -d "$ROCM_PREFIX/bin" ]]; then
    export PATH="$ROCM_PREFIX/bin:$PATH"
  fi
fi

export HIP_VISIBLE_DEVICES="${LEWM_HIP_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-0}}"
if [[ -n "${LEWM_HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  export HSA_OVERRIDE_GFX_VERSION="$LEWM_HSA_OVERRIDE_GFX_VERSION"
else
  unset HSA_OVERRIDE_GFX_VERSION
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "TinyQuadJEPA ROCm Python not found at: $PYTHON_BIN" >&2
  exit 2
fi

cd "$ROOT"
exec "$PYTHON_BIN" scripts/train_go2_rgb_jepa_vector_memory_controller.py "$@"

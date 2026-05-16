#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GENESIS_ROCM_PYTHON:-$ROOT/.generated/venvs/genesis_rocm/bin/python}"
EXAMPLES_DIR="${GENESIS_GO2_EXAMPLES_DIR:-$ROOT/.generated/upstream_genesis/locomotion}"

ROCM_PREFIX="${ROCM_PREFIX:-}"
if [[ -z "$ROCM_PREFIX" ]]; then
  if [[ -d /opt/rocm ]]; then
    ROCM_PREFIX="/opt/rocm"
  else
    ROCM_PREFIX="$(find /opt -maxdepth 1 -type d -name 'rocm-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
fi

if [[ -n "$ROCM_PREFIX" ]]; then
  export ROCM_PATH="${ROCM_PATH:-$ROCM_PREFIX}"
  if [[ -d "$ROCM_PREFIX/lib/llvm/bin" ]]; then
    export PATH="$ROCM_PREFIX/lib/llvm/bin:$PATH"
  elif [[ -d "$ROCM_PREFIX/llvm/bin" ]]; then
    export PATH="$ROCM_PREFIX/llvm/bin:$PATH"
  fi
  if [[ -d "$ROCM_PREFIX/bin" ]]; then
    export PATH="$ROCM_PREFIX/bin:$PATH"
  fi
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lewm_mplconfig}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ROCm Genesis Python not found at: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -f "$EXAMPLES_DIR/go2_train.py" || ! -f "$EXAMPLES_DIR/go2_env.py" ]]; then
  GENESIS_GO2_EXAMPLES_DIR="$EXAMPLES_DIR" "$ROOT/scripts/fetch_genesis_go2_locomotion_examples.sh"
fi

if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  export HSA_OVERRIDE_GFX_VERSION
  echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION (caller-supplied)"
else
  echo "HSA_OVERRIDE_GFX_VERSION unset (correct for RDNA4 / gfx1201)"
fi

echo "ROCM_PATH=${ROCM_PATH:-unset}"
echo "ld.lld=$(command -v ld.lld || echo missing)"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"

cd "$EXAMPLES_DIR"
"$PYTHON_BIN" go2_train.py \
  -e "${GENESIS_GO2_TRAIN_SMOKE_EXP_NAME:-codex-smoke}" \
  -B "${GENESIS_GO2_TRAIN_SMOKE_N_ENVS:-4}" \
  --max_iterations "${GENESIS_GO2_TRAIN_SMOKE_ITERS:-1}" \
  --seed "${GENESIS_GO2_TRAIN_SMOKE_SEED:-5}"

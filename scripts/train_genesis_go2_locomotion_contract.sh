#!/usr/bin/env bash
# Train a Genesis-native Go2 PPO policy on the LeWM trainable velocity
# primitives, rather than the upstream fixed-forward command recipe.
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
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ROCm Genesis Python not found at: $PYTHON_BIN" >&2
  echo "Run scripts/setup_genesis_rocm_training.sh first." >&2
  exit 2
fi

if [[ ! -f "$EXAMPLES_DIR/go2_train.py" || ! -f "$EXAMPLES_DIR/go2_env.py" ]]; then
  GENESIS_GO2_EXAMPLES_DIR="$EXAMPLES_DIR" "$ROOT/scripts/fetch_genesis_go2_locomotion_examples.sh"
fi

EXP_NAME="${LEWM_GO2_EXP_NAME:-lewm-go2-contract-$(date -u +%Y%m%dT%H%M%SZ)}"
NUM_ENVS="${LEWM_GO2_NUM_ENVS:-4096}"
MAX_ITERS="${LEWM_GO2_MAX_ITERS:-501}"
SEED="${LEWM_GO2_SEED:-11}"
SAVE_INTERVAL="${LEWM_GO2_SAVE_INTERVAL:-100}"
COMMAND_JITTER_STD="${LEWM_GO2_COMMAND_JITTER_STD:-0.0}"
TRACKING_ANG_VEL_SCALE="${LEWM_GO2_TRACKING_ANG_VEL_SCALE:-0.8}"

if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  export HSA_OVERRIDE_GFX_VERSION
  echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION (caller-supplied)"
else
  echo "HSA_OVERRIDE_GFX_VERSION unset (correct for RDNA4 / gfx1201)"
fi

echo "ROCM_PATH=${ROCM_PATH:-unset}"
echo "ld.lld=$(command -v ld.lld || echo missing)"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"

exec "$PYTHON_BIN" "$ROOT/scripts/train_genesis_go2_locomotion_contract.py" \
  --repo-root "$ROOT" \
  --examples-dir "$EXAMPLES_DIR" \
  --exp-name "$EXP_NAME" \
  --num-envs "$NUM_ENVS" \
  --max-iterations "$MAX_ITERS" \
  --seed "$SEED" \
  --save-interval "$SAVE_INTERVAL" \
  --command-jitter-std "$COMMAND_JITTER_STD" \
  --tracking-ang-vel-scale "$TRACKING_ANG_VEL_SCALE"

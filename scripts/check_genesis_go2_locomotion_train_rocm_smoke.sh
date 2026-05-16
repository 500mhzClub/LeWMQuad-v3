#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GENESIS_ROCM_PYTHON:-$ROOT/.generated/venvs/genesis_rocm/bin/python}"
EXAMPLES_DIR="${GENESIS_GO2_EXAMPLES_DIR:-$ROOT/.generated/upstream_genesis/locomotion}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ROCm Genesis Python not found at: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -f "$EXAMPLES_DIR/go2_train.py" || ! -f "$EXAMPLES_DIR/go2_env.py" ]]; then
  GENESIS_GO2_EXAMPLES_DIR="$EXAMPLES_DIR" "$ROOT/scripts/fetch_genesis_go2_locomotion_examples.sh"
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

cd "$EXAMPLES_DIR"
"$PYTHON_BIN" go2_train.py \
  -e "${GENESIS_GO2_TRAIN_SMOKE_EXP_NAME:-codex-smoke}" \
  -B "${GENESIS_GO2_TRAIN_SMOKE_N_ENVS:-4}" \
  --max_iterations "${GENESIS_GO2_TRAIN_SMOKE_ITERS:-1}" \
  --seed "${GENESIS_GO2_TRAIN_SMOKE_SEED:-5}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GENESIS_ROCM_PYTHON:-$ROOT/.generated/venvs/genesis_rocm/bin/python}"
EXAMPLES_DIR="${GENESIS_GO2_EXAMPLES_DIR:-$ROOT/.generated/upstream_genesis/locomotion}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  cat >&2 <<EOF
ROCm Genesis Python not found at:
  $PYTHON_BIN

Create it with:
  python3 -m venv --system-site-packages .generated/venvs/genesis_rocm
  .generated/venvs/genesis_rocm/bin/python -m pip install --upgrade pip wheel setuptools
  .generated/venvs/genesis_rocm/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
  .generated/venvs/genesis_rocm/bin/python -m pip install tensordict tensorboard GitPython onnx onnxscript
  .generated/venvs/genesis_rocm/bin/python -m pip install --no-deps 'rsl-rl-lib>=5.0.0'
EOF
  exit 2
fi

if [[ ! -f "$EXAMPLES_DIR/go2_train.py" || ! -f "$EXAMPLES_DIR/go2_env.py" ]]; then
  GENESIS_GO2_EXAMPLES_DIR="$EXAMPLES_DIR" "$ROOT/scripts/fetch_genesis_go2_locomotion_examples.sh"
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export GS_BACKEND="${GS_BACKEND:-amdgpu}"

"$PYTHON_BIN" "$ROOT/scripts/check_genesis_go2_locomotion_env_smoke.py" \
  --examples-dir "$EXAMPLES_DIR" \
  --backend "$GS_BACKEND" \
  --n-envs "${GENESIS_GO2_LOCO_SMOKE_N_ENVS:-4}" \
  --steps "${GENESIS_GO2_LOCO_SMOKE_STEPS:-3}" \
  "$@"

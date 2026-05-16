#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GENESIS_ROCM_PYTHON:-$ROOT/.generated/venvs/genesis_rocm/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  cat >&2 <<EOF
ROCm Genesis Python not found at:
  $PYTHON_BIN

Create it with:
  python3 -m venv --system-site-packages .generated/venvs/genesis_rocm
  .generated/venvs/genesis_rocm/bin/python -m pip install --upgrade pip wheel setuptools
  .generated/venvs/genesis_rocm/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
EOF
  exit 2
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export GS_BACKEND="${GS_BACKEND:-amdgpu}"

"$PYTHON_BIN" "$ROOT/scripts/check_genesis_go2_backend_smoke.py" \
  --backend "$GS_BACKEND" \
  --n-envs "${GENESIS_GO2_SMOKE_N_ENVS:-4}" \
  --steps "${GENESIS_GO2_SMOKE_STEPS:-10}" \
  "$@"

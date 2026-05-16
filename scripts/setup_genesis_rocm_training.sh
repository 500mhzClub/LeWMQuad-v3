#!/usr/bin/env bash
# Idempotent setup for the Genesis ROCm training venv on a fresh clone.
#
# Creates .generated/venvs/genesis_rocm with a ROCm PyTorch + Genesis + rsl-rl
# stack, then fetches the upstream Genesis Go2 locomotion examples into
# .generated/upstream_genesis/locomotion. Safe to re-run; existing installs
# are left in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${GENESIS_ROCM_VENV:-$ROOT/.generated/venvs/genesis_rocm}"
PYTHON_BIN="$VENV_DIR/bin/python"
TORCH_INDEX_URL="${GENESIS_ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
TORCH_PKGS=(torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1)
GENESIS_PKG="${GENESIS_PKG_SPEC:-genesis-world==0.4.6}"
RSL_RL_PKG="${GENESIS_RSL_RL_SPEC:-rsl-rl-lib>=5.0.0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating ROCm venv at $VENV_DIR"
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools

if ! "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if 'rocm' in torch.__version__ else 1)" 2>/dev/null; then
  echo "Installing ROCm PyTorch from $TORCH_INDEX_URL"
  "$PYTHON_BIN" -m pip install "${TORCH_PKGS[@]}" --index-url "$TORCH_INDEX_URL"
fi

"$PYTHON_BIN" -m pip install "$GENESIS_PKG"
"$PYTHON_BIN" -m pip install tensordict tensorboard GitPython onnx onnxscript
"$PYTHON_BIN" -m pip install --no-deps "$RSL_RL_PKG"

EXAMPLES_DIR="${GENESIS_GO2_EXAMPLES_DIR:-$ROOT/.generated/upstream_genesis/locomotion}"
if [[ ! -f "$EXAMPLES_DIR/go2_train.py" ]]; then
  GENESIS_GO2_EXAMPLES_DIR="$EXAMPLES_DIR" "$ROOT/scripts/fetch_genesis_go2_locomotion_examples.sh"
fi

echo
echo "Genesis ROCm training environment ready."
echo "  venv:     $VENV_DIR"
echo "  examples: $EXAMPLES_DIR"
echo
echo "Next: run a quick wiring check, then kick off training."
echo "  scripts/check_genesis_go2_locomotion_train_rocm_smoke.sh"
echo "  scripts/train_genesis_go2_locomotion.sh"

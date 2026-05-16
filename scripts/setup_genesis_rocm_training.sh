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
  # No --system-site-packages: previous iterations leaked stale system copies
  # of coverage (pre-7.6, breaking numba) and matplotlib (numpy-1.x ABI vs.
  # venv numpy 2.x) into the venv. Cleaner to fully isolate.
  python3 -m venv "$VENV_DIR"
fi

# Hard prerequisite: ld.lld is needed by Genesis's quadrants/Taichi AMDGPU JIT
# to link HSACO kernel modules. ROCm bundles it under /opt/rocm/llvm/bin/; the
# system package on Debian/Ubuntu is `lld`.
if ! command -v ld.lld >/dev/null 2>&1; then
  cat >&2 <<'EOF'
WARNING: ld.lld not found on PATH.
Genesis's AMDGPU JIT (quadrants/Taichi) needs it to compile HSACO kernels.
Install one of:
  sudo apt install lld
  export PATH=/opt/rocm/llvm/bin:$PATH    # if ROCm bundles it locally
EOF
fi

"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools

if ! "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if 'rocm' in torch.__version__ else 1)" 2>/dev/null; then
  echo "Installing ROCm PyTorch from $TORCH_INDEX_URL"
  "$PYTHON_BIN" -m pip install "${TORCH_PKGS[@]}" --index-url "$TORCH_INDEX_URL"
fi

"$PYTHON_BIN" -m pip install "$GENESIS_PKG"
# matplotlib must come from PyPI against the venv's numpy 2.x, not the system
# apt copy that was built against numpy 1.x. Genesis imports matplotlib via
# its recorders/plotters module at top level, so a numpy ABI mismatch crashes
# `import genesis` before training starts.
"$PYTHON_BIN" -m pip install matplotlib
"$PYTHON_BIN" -m pip install tensordict tensorboard GitPython onnx onnxscript
# Numba (pulled in by Genesis) imports `coverage.types.Tracer` (added in
# coverage 7.6) at module-load time whenever coverage is importable.
"$PYTHON_BIN" -m pip install 'coverage>=7.6'
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

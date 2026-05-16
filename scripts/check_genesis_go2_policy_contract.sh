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
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ROCm Genesis Python not found at: $PYTHON_BIN" >&2
  exit 2
fi

cd "$EXAMPLES_DIR"
exec "$PYTHON_BIN" "$ROOT/scripts/check_genesis_go2_policy_contract.py" \
  --repo-root "$ROOT" \
  --examples-dir "$EXAMPLES_DIR" \
  "$@"

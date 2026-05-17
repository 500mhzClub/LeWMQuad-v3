#!/usr/bin/env bash
# Render a planned raw_rollout replay with Genesis.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

ROOT="$(lewm_repo_root)"
PYTHON_BIN="${GENESIS_ROCM_PYTHON:-$ROOT/.generated/venvs/genesis_rocm/bin/python}"

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

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.generated/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.generated/mplconfig}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Genesis Python not found at: $PYTHON_BIN" >&2
  echo "Run scripts/setup_genesis_rocm_training.sh first." >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/lewm_genesis:${PYTHONPATH:-}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/render_replay_genesis.py" "$@"

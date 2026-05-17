#!/usr/bin/env bash
# Plan render-replay jobs for converted raw_rollout directories.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$ROOT/lewm_genesis:${PYTHONPATH:-}"
exec python3 "$SCRIPT_DIR/plan_bulk_render_replay.py" "$@"

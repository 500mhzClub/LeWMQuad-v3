#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/convert_smoke_bag_to_raw_rollout.sh <bag-dir> [--out <out-dir>] [--quality-profile smoke|pilot|training|raw_pilot|raw_training]" >&2
  exit 2
fi

cd "$REPO_ROOT"
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

exec python3 "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.py" "$@"

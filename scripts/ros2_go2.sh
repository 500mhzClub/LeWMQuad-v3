#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"

if [[ $# -eq 0 ]]; then
  echo "Usage: scripts/ros2_go2.sh <command> [args...]" >&2
  echo "Example: scripts/ros2_go2.sh ros2 control list_controllers" >&2
  exit 2
fi

cd "$REPO_ROOT"
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

exec "$@"


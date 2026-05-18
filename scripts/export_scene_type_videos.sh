#!/usr/bin/env bash
# Export visual QA videos from rendered Genesis replay outputs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

ROOT="$(lewm_repo_root)"
cd "$ROOT"

exec python3 "$SCRIPT_DIR/export_scene_type_videos.py" "$@"

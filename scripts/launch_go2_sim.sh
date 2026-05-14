#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"

cd "$REPO_ROOT"

lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

# Gazebo GUI and RViz on Pop!_OS/Wayland currently need Qt to use XWayland.
# COSMIC exports QT_QPA_PLATFORM=wayland;xcb by default, which still lets Qt try
# Wayland first and can trigger OGRE/GLX window creation failures. Force xcb for
# this launcher unless the caller deliberately opts out.
if [[ "${LEWM_GO2_KEEP_QT_PLATFORM:-0}" != "1" && -n "${WAYLAND_DISPLAY:-}" && -n "${DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

if [[ $# -eq 0 ]]; then
  set -- rviz:=false gui:=false
fi

exec ros2 launch lewm_go2_bringup go2_sim.launch.py "$@"

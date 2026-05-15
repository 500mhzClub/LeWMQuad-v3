#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
SMOKE_WORLD_SEED=""
SMOKE_WORLD_FAMILY="open_obstacle_field"

cd "$REPO_ROOT"

lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-world)
      SMOKE_WORLD_SEED="${2:-0}"
      shift 2
      ;;
    --smoke-world-family)
      SMOKE_WORLD_FAMILY="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

# Gazebo GUI and RViz on Pop!_OS/Wayland currently need Qt to use XWayland.
# COSMIC exports QT_QPA_PLATFORM=wayland;xcb by default, which still lets Qt try
# Wayland first and can trigger OGRE/GLX window creation failures. Force xcb for
# this launcher unless the caller deliberately opts out.
if [[ "${LEWM_GO2_KEEP_QT_PLATFORM:-0}" != "1" && -n "${WAYLAND_DISPLAY:-}" && -n "${DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

if [[ ${#PASSTHROUGH_ARGS[@]} -eq 0 ]]; then
  PASSTHROUGH_ARGS=(rviz:=false gui:=false)
fi

if [[ -n "$SMOKE_WORLD_SEED" ]]; then
  world_dir="$REPO_ROOT/.generated/worlds/${SMOKE_WORLD_FAMILY}_${SMOKE_WORLD_SEED}"
  "$SCRIPT_DIR/generate_smoke_world.sh" \
    --seed "$SMOKE_WORLD_SEED" \
    --family "$SMOKE_WORLD_FAMILY" \
    --out "$world_dir" >/dev/null
  PASSTHROUGH_ARGS+=("world:=$world_dir/world.sdf")
fi

exec ros2 launch lewm_go2_bringup go2_sim.launch.py "${PASSTHROUGH_ARGS[@]}"

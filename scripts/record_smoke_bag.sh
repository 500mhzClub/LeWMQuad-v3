#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
OUT_DIR="$REPO_ROOT/.generated/bags/go2_bringup_smoke_$(date +%Y%m%d_%H%M%S)"
DURATION_SECS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --duration)
      DURATION_SECS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"
mkdir -p "$(dirname "$OUT_DIR")"

topics=(
  /clock
  /tf
  /joint_states
  /imu/data
  /odom
  /gazebo/odom
  /rgb_image
  /lewm/go2/camera_info
  /cmd_vel
  /lewm/go2/command_block
  /lewm/go2/executed_command_block
  /lewm/go2/base_state
  /lewm/go2/foot_contacts
  /lewm/go2/mode
  /lewm/go2/reset_event
  /lewm/episode_info
)

echo "Recording smoke bag to $OUT_DIR for ${DURATION_SECS}s..."
set +e
timeout --signal=INT --kill-after=5s "${DURATION_SECS}s" \
  ros2 bag record -s mcap -o "$OUT_DIR" "${topics[@]}"
record_status=$?
set -e

if [[ "$record_status" -ne 0 && "$record_status" -ne 124 && "$record_status" -ne 130 ]]; then
  echo "ros2 bag record failed with exit code $record_status" >&2
  exit "$record_status"
fi

echo "Smoke bag written to $OUT_DIR"

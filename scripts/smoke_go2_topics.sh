#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
DRIVE=0
WAIT_SECS="${WAIT_SECS:-45}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drive)
      DRIVE=1
      shift
      ;;
    --wait)
      WAIT_SECS="$2"
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

required_topics=(
  /clock
  /tf
  /joint_states
  /imu/data
  /odom
  /rgb_image
  /cmd_vel
)

topic_file="$(mktemp)"
trap 'rm -f "$topic_file"' EXIT

deadline=$((SECONDS + WAIT_SECS))
while true; do
  ros2 topic list > "$topic_file"
  missing_now=0
  for topic_name in "${required_topics[@]}"; do
    if ! grep -qx "$topic_name" "$topic_file"; then
      missing_now=1
      break
    fi
  done

  if [[ "$missing_now" == "0" || "$SECONDS" -ge "$deadline" ]]; then
    break
  fi

  sleep 1
done

missing=0
for topic_name in "${required_topics[@]}"; do
  if grep -qx "$topic_name" "$topic_file"; then
    echo "ok: $topic_name"
  else
    echo "missing: $topic_name" >&2
    missing=1
  fi
done

if [[ "$missing" != "0" ]]; then
  echo "Smoke check failed: required topics are missing." >&2
  exit 1
fi

if [[ "$DRIVE" == "1" ]]; then
  echo "Publishing a short forward /cmd_vel command..."
  timeout --signal=INT --kill-after=2s 3s ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.15, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
    -r 5 >/dev/null || true
fi

echo "Smoke topics are present."

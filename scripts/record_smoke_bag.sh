#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
OUT_DIR="$REPO_ROOT/.generated/bags/go2_bringup_smoke_$(date +%Y%m%d_%H%M%S)"
DURATION_SECS=20
QOS_OVERRIDES="$REPO_ROOT/config/rosbag_record_qos_overrides.yaml"
MAX_CACHE_BYTES=$((256 * 1024 * 1024))
RECORD_PROFILE="vision"

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
    --qos-overrides)
      QOS_OVERRIDES="$2"
      shift 2
      ;;
    --no-qos-overrides)
      QOS_OVERRIDES=""
      shift
      ;;
    --max-cache-bytes)
      MAX_CACHE_BYTES="$2"
      shift 2
      ;;
    --profile)
      RECORD_PROFILE="$2"
      shift 2
      ;;
    --raw-only)
      RECORD_PROFILE="raw"
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/record_smoke_bag.sh [options]

Options:
  --out PATH             Bag output directory.
  --duration N           Recording duration in seconds (default 20).
  --profile vision|raw   vision records /rgb_image; raw omits RGB payloads.
  --raw-only             Alias for --profile raw.
  --qos-overrides PATH   QoS override YAML.
  --no-qos-overrides     Use rosbag2 default subscription QoS.
  --max-cache-bytes N    rosbag2 cache size in bytes.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RECORD_PROFILE" != "vision" && "$RECORD_PROFILE" != "raw" ]]; then
  echo "Unknown profile: $RECORD_PROFILE (expected vision or raw)" >&2
  exit 2
fi

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

if [[ "$RECORD_PROFILE" == "vision" ]]; then
  topics+=(
    /rgb_image
  )
fi

record_args=(-s mcap -o "$OUT_DIR" --max-cache-size "$MAX_CACHE_BYTES")
if [[ -n "$QOS_OVERRIDES" ]]; then
  if [[ ! -f "$QOS_OVERRIDES" ]]; then
    echo "QoS overrides file not found: $QOS_OVERRIDES" >&2
    exit 1
  fi
  record_args+=(--qos-profile-overrides-path "$QOS_OVERRIDES")
  echo "Using QoS overrides: $QOS_OVERRIDES"
else
  echo "Recording without QoS overrides (default subscription depth applies)."
fi

echo "Recording smoke bag to $OUT_DIR for ${DURATION_SECS}s (profile=${RECORD_PROFILE}, cache=${MAX_CACHE_BYTES}B)..."
set +e
timeout --signal=INT --kill-after=5s "${DURATION_SECS}s" \
  ros2 bag record "${record_args[@]}" "${topics[@]}"
record_status=$?
set -e

if [[ "$record_status" -ne 0 && "$record_status" -ne 124 && "$record_status" -ne 130 ]]; then
  echo "ros2 bag record failed with exit code $record_status" >&2
  exit "$record_status"
fi

echo "Smoke bag written to $OUT_DIR"

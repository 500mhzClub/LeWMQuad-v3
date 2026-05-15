#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
OUT_FILE=""
RESET_BEFORE=1
WAIT_SECS=90

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_FILE="$2"
      shift 2
      ;;
    --no-reset)
      RESET_BEFORE=0
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

if [[ -z "$OUT_FILE" ]]; then
  OUT_FILE="$REPO_ROOT/.generated/audits/primitive_motion_$(date +%Y%m%d_%H%M%S).json"
fi

mkdir -p "$(dirname "$OUT_FILE")"
cd "$REPO_ROOT"
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

deadline=$((SECONDS + WAIT_SECS))
while true; do
  controller_state="$(timeout 3s ros2 control list_controllers 2>/dev/null || true)"
  if printf '%s\n' "$controller_state" \
    | grep -Eq '^joint_group_effort_controller[[:space:]]+joint_trajectory_controller/JointTrajectoryController[[:space:]]+active'; then
    break
  fi
  if [[ "$SECONDS" -ge "$deadline" ]]; then
    echo "Timed out waiting for joint_group_effort_controller to become active." >&2
    exit 1
  fi
  sleep 1
done

if [[ "$RESET_BEFORE" == "1" ]]; then
  ros2 service call /lewm/go2/reset lewm_go2_control/srv/ResetEpisode \
    "{scene_id: 0, reason: 'primitive_motion_audit', spawn_pose_world: {position: {x: 0.0, y: 0.0, z: 0.375}, orientation: {w: 1.0}}, use_spawn_pose: true}" >/dev/null
  sleep 2
fi

response="$(ros2 service call /lewm/go2/run_feature_check \
  lewm_go2_control/srv/RunFeatureCheck \
  "{check_name: 'primitive_motion', include_optional: false}")"

printf '%s\n' "$response"
RESPONSE="$response" python3 - "$OUT_FILE" <<'PY'
import ast
import os
import re
import sys

out_file = sys.argv[1]
text = os.environ["RESPONSE"]
match = re.search(r"report_json='((?:[^'\\]|\\.)*)'", text, re.S)
if not match:
    raise SystemExit("could not find report_json in service response")
report_json = ast.literal_eval("'" + match.group(1) + "'")
with open(out_file, "w", encoding="utf-8") as stream:
    stream.write(report_json)
    stream.write("\n")
print(f"wrote {out_file}")
PY

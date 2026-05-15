#!/usr/bin/env bash
# End-to-end acceptance gate for the Go2 contract path.
#
# Stages (each fails the run if it fails; teardown always runs):
#   1. build_workspace        scripts/build_go2_sim.sh --skip-rosdep        (skippable)
#   2. scene_corpus_smoke     scripts/generate_scene_corpus.sh --name ...   (skippable)
#   3. launch_sim             scripts/launch_go2_sim.sh in a process group   (skippable)
#   4. await_topics           wait for required ROS topics to appear
#   5. feature_check_all      /lewm/go2/run_feature_check check_name=all
#   6. primitive_audit        scripts/audit_go2_primitives.sh
#   7. smoke_bag              scripts/record_smoke_bag.sh + /cmd_vel drive
#   8. bag_conversion         scripts/convert_smoke_bag_to_raw_rollout.sh
#
# Writes a single summary.json under the run directory and exits non-zero
# if any non-skipped stage failed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"

SKIP_BUILD=0
SKIP_SCENE_CORPUS=0
SKIP_LAUNCH=0
SKIP_BAG=0
KEEP_RUNNING=0
RUN_NAME="acceptance_$(date +%Y%m%d_%H%M%S)"
OUT_DIR=""
BAG_DURATION=20
TOPIC_TIMEOUT_S=90
FEATURE_CHECK_TIMEOUT_S=60
PRIMITIVE_AUDIT_WAIT_S=120
SCENE_CORPUS_NAME="acceptance"
PLAN_SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) RUN_NAME="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-scene-corpus) SKIP_SCENE_CORPUS=1; shift ;;
    --skip-launch) SKIP_LAUNCH=1; shift ;;
    --skip-bag) SKIP_BAG=1; shift ;;
    --keep-running) KEEP_RUNNING=1; shift ;;
    --bag-duration) BAG_DURATION="$2"; shift 2 ;;
    --topic-timeout) TOPIC_TIMEOUT_S="$2"; shift 2 ;;
    --scene-corpus-name) SCENE_CORPUS_NAME="$2"; shift 2 ;;
    --plan-seed) PLAN_SEED="$2"; shift 2 ;;
    -h|--help)
      cat <<'USAGE'
Usage: acceptance_go2_contract.sh [options]

Stages (each fails the run if it fails; teardown always runs):
  build_workspace        scripts/build_go2_sim.sh --skip-rosdep         (--skip-build)
  scene_corpus_smoke     scripts/generate_scene_corpus.sh --smoke       (--skip-scene-corpus)
  launch_sim             scripts/launch_go2_sim.sh in a process group   (--skip-launch)
  await_topics           wait for required ROS topics
  feature_check_all      /lewm/go2/run_feature_check check_name=all
  primitive_audit        scripts/audit_go2_primitives.sh
  smoke_bag              scripts/record_smoke_bag.sh + /cmd_vel drive   (--skip-bag)
  bag_conversion         scripts/convert_smoke_bag_to_raw_rollout.sh    (--skip-bag)

Options:
  --name STR             Run name; output goes to .generated/acceptance/<name>/.
  --out PATH             Absolute output directory (overrides --name location).
  --skip-build           Skip workspace build.
  --skip-scene-corpus    Skip scene corpus generation.
  --skip-launch          Assume a sim is already running.
  --skip-bag             Skip bag recording and conversion.
  --keep-running         Do not stop the simulator on exit.
  --bag-duration N       Smoke bag duration in seconds (default 20).
  --topic-timeout N      Seconds to wait for required topics (default 90).
  --scene-corpus-name S  Corpus subdirectory name (default 'acceptance').
  --plan-seed N          Corpus plan seed (default 0).

Writes a single summary.json under the run directory and exits non-zero
if any non-skipped stage failed.
USAGE
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/.generated/acceptance/$RUN_NAME"
fi
mkdir -p "$OUT_DIR"
STAGES_DIR="$OUT_DIR/stages"
mkdir -p "$STAGES_DIR"

LAUNCH_LOG="$OUT_DIR/launch.log"
LAUNCH_PGID=""
DRIVE_PID=""
SCRIPT_START_EPOCH=$(date +%s)

# Mirror everything to a transcript.
exec > >(tee "$OUT_DIR/acceptance.log") 2>&1

echo "==> Acceptance run: $RUN_NAME"
echo "    output: $OUT_DIR"
echo "    skip_build=$SKIP_BUILD skip_scene_corpus=$SKIP_SCENE_CORPUS skip_launch=$SKIP_LAUNCH skip_bag=$SKIP_BAG keep_running=$KEEP_RUNNING"

cd "$REPO_ROOT"

# ---- helpers --------------------------------------------------------------

declare -a STAGES_ORDER=()

run_stage() {
  # run_stage <name> <cmd...>
  local name="$1"; shift
  local stage_log="$STAGES_DIR/$name.log"
  local stage_status="$STAGES_DIR/$name.status"
  local started=$(date +%s)
  STAGES_ORDER+=("$name")
  echo
  echo "==> stage: $name"
  echo "    cmd: $*"
  local rc=0
  if "$@" > >(tee "$stage_log") 2>&1; then
    rc=0
  else
    rc=$?
  fi
  local ended=$(date +%s)
  local duration=$((ended - started))
  printf '{"name": "%s", "rc": %d, "duration_s": %d, "log": "%s"}\n' \
    "$name" "$rc" "$duration" "stages/$name.log" > "$stage_status"
  if [[ "$rc" -ne 0 ]]; then
    echo "==> stage failed: $name (rc=$rc)"
  else
    echo "==> stage ok: $name (${duration}s)"
  fi
  return "$rc"
}

mark_stage_skipped() {
  local name="$1"
  local reason="$2"
  STAGES_ORDER+=("$name")
  printf '{"name": "%s", "rc": null, "duration_s": 0, "log": null, "skipped": "%s"}\n' \
    "$name" "$reason" > "$STAGES_DIR/$name.status"
  echo
  echo "==> stage skipped: $name ($reason)"
}

# ---- teardown -------------------------------------------------------------

stop_drive() {
  if [[ -n "${DRIVE_PID:-}" ]] && kill -0 "$DRIVE_PID" 2>/dev/null; then
    kill -INT "$DRIVE_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "$DRIVE_PID" 2>/dev/null || true
  fi
  DRIVE_PID=""
}

stop_launch() {
  if [[ -z "${LAUNCH_PGID:-}" ]]; then
    return 0
  fi
  if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo "==> --keep-running set; leaving simulator process group $LAUNCH_PGID alive"
    return 0
  fi
  echo "==> stopping simulator (pgid=$LAUNCH_PGID)"
  kill -INT "-$LAUNCH_PGID" 2>/dev/null || true
  for _ in $(seq 1 15); do
    if ! kill -0 "-$LAUNCH_PGID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "-$LAUNCH_PGID" 2>/dev/null; then
    echo "    SIGINT did not exit; sending SIGTERM"
    kill -TERM "-$LAUNCH_PGID" 2>/dev/null || true
    sleep 3
  fi
  if kill -0 "-$LAUNCH_PGID" 2>/dev/null; then
    echo "    sending SIGKILL"
    kill -KILL "-$LAUNCH_PGID" 2>/dev/null || true
  fi
  LAUNCH_PGID=""
}

write_summary_and_exit() {
  local final_rc="$1"
  stop_drive
  stop_launch
  STAGES_ORDER_STR="${STAGES_ORDER[*]:-}" \
  RUN_NAME="$RUN_NAME" OUT_DIR="$OUT_DIR" STAGES_DIR="$STAGES_DIR" \
  SCRIPT_START_EPOCH="$SCRIPT_START_EPOCH" FINAL_RC="$final_rc" \
  python3 - <<'PY'
import json
import os
import time
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
stages_dir = Path(os.environ["STAGES_DIR"])
order = os.environ.get("STAGES_ORDER_STR", "").split()
final_rc = int(os.environ["FINAL_RC"])
started = int(os.environ["SCRIPT_START_EPOCH"])
now = int(time.time())

stages = []
any_failed = False
for name in order:
    path = stages_dir / f"{name}.status"
    if not path.exists():
        stages.append({"name": name, "rc": None, "missing": True})
        any_failed = True
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages.append(payload)
    if payload.get("skipped"):
        continue
    if payload.get("rc") not in (0, None):
        any_failed = True

summary = {
    "run_name": os.environ["RUN_NAME"],
    "started_epoch_s": started,
    "ended_epoch_s": now,
    "duration_s": now - started,
    "final_rc": final_rc,
    "pass": (final_rc == 0) and (not any_failed),
    "stages": stages,
}
summary_path = out_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(f"summary: {summary_path}  pass={summary['pass']}  rc={final_rc}")
PY
  exit "$final_rc"
}

trap 'write_summary_and_exit "${EXIT_RC:-1}"' EXIT
trap 'EXIT_RC=130; exit 130' INT
trap 'EXIT_RC=143; exit 143' TERM

set +e

# ---- stage: build ---------------------------------------------------------

if [[ "$SKIP_BUILD" == "1" ]]; then
  mark_stage_skipped build_workspace "--skip-build"
else
  run_stage build_workspace "$SCRIPT_DIR/build_go2_sim.sh" --skip-rosdep
  if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi
fi

# ---- stage: scene corpus smoke -------------------------------------------

if [[ "$SKIP_SCENE_CORPUS" == "1" ]]; then
  mark_stage_skipped scene_corpus_smoke "--skip-scene-corpus"
else
  run_stage scene_corpus_smoke "$SCRIPT_DIR/generate_scene_corpus.sh" \
    --smoke --name "$SCENE_CORPUS_NAME" --plan-seed "$PLAN_SEED"
  if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi
fi

# Past this point we need the workspace sourced for all ros2 commands.
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

# ---- stage: launch sim ----------------------------------------------------

if [[ "$SKIP_LAUNCH" == "1" ]]; then
  mark_stage_skipped launch_sim "--skip-launch"
else
  STAGES_ORDER+=("launch_sim")
  echo
  echo "==> stage: launch_sim (background)"
  setsid "$SCRIPT_DIR/launch_go2_sim.sh" rviz:=false gui:=false \
      > "$LAUNCH_LOG" 2>&1 &
  LAUNCH_PID=$!
  # Give the new session a moment to assign its pgid.
  sleep 1
  LAUNCH_PGID="$(ps -o pgid= -p "$LAUNCH_PID" 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "$LAUNCH_PGID" ]] || [[ "$LAUNCH_PGID" == "0" ]]; then
    printf '{"name": "launch_sim", "rc": 1, "duration_s": 0, "log": "launch.log", "error": "failed to spawn"}\n' \
      > "$STAGES_DIR/launch_sim.status"
    echo "    failed to spawn launch process"
    EXIT_RC=1
    exit 1
  fi
  printf '{"name": "launch_sim", "rc": 0, "duration_s": 0, "log": "launch.log", "pgid": %s}\n' \
    "$LAUNCH_PGID" > "$STAGES_DIR/launch_sim.status"
  echo "    launch pid=$LAUNCH_PID pgid=$LAUNCH_PGID log=$LAUNCH_LOG"
fi

# ---- stage: await topics --------------------------------------------------

await_topics() {
  local deadline=$((SECONDS + TOPIC_TIMEOUT_S))
  local required=(/clock /tf /joint_states /imu/data /odom /rgb_image /cmd_vel)
  while true; do
    local listing
    listing="$(ros2 topic list 2>/dev/null || true)"
    local missing=0
    for t in "${required[@]}"; do
      if ! printf '%s\n' "$listing" | grep -qx "$t"; then
        missing=1
        break
      fi
    done
    if [[ "$missing" == "0" ]]; then
      echo "all required topics present"
      return 0
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "timed out waiting for required topics; got:" >&2
      printf '%s\n' "$listing" >&2
      return 1
    fi
    sleep 2
  done
}
run_stage await_topics await_topics
if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi

# ---- stage: feature_check_all --------------------------------------------

feature_check_all() {
  local response
  response="$(timeout "${FEATURE_CHECK_TIMEOUT_S}s" ros2 service call \
    /lewm/go2/run_feature_check \
    lewm_go2_control/srv/RunFeatureCheck \
    "{check_name: 'all', include_optional: false}" 2>&1)"
  local rc=$?
  printf '%s\n' "$response"
  if [[ "$rc" -ne 0 ]]; then return "$rc"; fi
  RESPONSE="$response" python3 - <<'PY'
import ast
import json
import os
import re

text = os.environ["RESPONSE"]
match = re.search(r"report_json='((?:[^'\\]|\\.)*)'", text, re.S)
if not match:
    raise SystemExit("could not find report_json in service response")
report = json.loads(ast.literal_eval("'" + match.group(1) + "'"))
checks = report.get("checks", {})
failing = sorted(
    name for name, payload in checks.items() if not payload.get("success", False)
)
print(f"feature_check.success = {report.get('success')}  failing_checks = {failing}")
if not report.get("success"):
    raise SystemExit(1)
PY
}
run_stage feature_check_all feature_check_all
if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi

# ---- stage: primitive_audit ----------------------------------------------

PRIMITIVE_AUDIT_OUT="$OUT_DIR/primitive_motion_audit.json"
run_stage primitive_audit "$SCRIPT_DIR/audit_go2_primitives.sh" \
  --out "$PRIMITIVE_AUDIT_OUT" --wait "$PRIMITIVE_AUDIT_WAIT_S"
if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi

# Confirm the audit has no failures.
audit_pass() {
  python3 - "$PRIMITIVE_AUDIT_OUT" <<'PY'
import json
import sys
path = sys.argv[1]
report = json.loads(open(path, encoding="utf-8").read())
node = report.get("checks", {}).get("primitive_motion", report)
results = node.get("results", {})
if isinstance(results, list):
    # Defensive: older shapes returned a list of {name, ...} dicts.
    failures = [r for r in results if not r.get("success", r.get("pass", True))]
    failing_names = [r.get("name") for r in failures]
else:
    failures = {name: r for name, r in results.items() if not r.get("success", True)}
    failing_names = sorted(failures)
print(f"audit primitives evaluated={len(results)} failed={len(failing_names)}")
if failing_names:
    for name in failing_names:
        detail = failures[name] if isinstance(failures, dict) else None
        print(f"  fail: {name}: {detail}")
    raise SystemExit(1)
PY
}
run_stage primitive_audit_pass audit_pass
if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi

# ---- stage: smoke bag + conversion ---------------------------------------

if [[ "$SKIP_BAG" == "1" ]]; then
  mark_stage_skipped smoke_bag "--skip-bag"
  mark_stage_skipped bag_conversion "--skip-bag"
else
  BAG_DIR="$OUT_DIR/smoke_bag"
  # Drive the robot for the duration of the bag so the recording has motion.
  drive_robot() {
    local rate_hz=5
    local cycles=$((BAG_DURATION * rate_hz))
    for _ in $(seq 1 "$cycles"); do
      ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: 0.18, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.05}}" \
        >/dev/null 2>&1 || true
      sleep $(awk -v r="$rate_hz" 'BEGIN { printf "%.3f", 1.0 / r }')
    done
  }
  drive_robot &
  DRIVE_PID=$!
  run_stage smoke_bag "$SCRIPT_DIR/record_smoke_bag.sh" \
    --out "$BAG_DIR" --duration "$BAG_DURATION"
  stop_drive
  if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi

  RAW_ROLLOUT_DIR="$OUT_DIR/smoke_bag_raw_rollout"
  run_stage bag_conversion "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" \
    "$BAG_DIR" --out "$RAW_ROLLOUT_DIR"
  if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi

  # Verify the conversion captured the contract topics.
  conversion_contract() {
    python3 - "$RAW_ROLLOUT_DIR/summary.json" <<'PY'
import json
import sys
path = sys.argv[1]
summary = json.loads(open(path, encoding="utf-8").read())
counts = summary.get("topic_counts", {})
required = [
    "/lewm/go2/command_block",
    "/lewm/go2/executed_command_block",
    "/lewm/go2/reset_event",
    "/cmd_vel",
    "/gazebo/odom",
]
missing = [t for t in required if counts.get(t, 0) <= 0]
print(f"raw_rollout topic counts (subset): "
      + ", ".join(f"{t}={counts.get(t, 0)}" for t in required))
if missing:
    print("missing contract topics:", missing)
    raise SystemExit(1)
PY
  }
  run_stage bag_conversion_contract conversion_contract
  if [[ $? -ne 0 ]]; then EXIT_RC=1; exit 1; fi
fi

set -e
EXIT_RC=0
exit 0

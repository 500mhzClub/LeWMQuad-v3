#!/usr/bin/env bash
# Measure Gazebo-only data-capture throughput from an already running simulator.
#
# The benchmark drives through /lewm/go2/command_block, records the standard
# smoke topics, converts the bag to compact raw_rollout JSONL, and writes a
# throughput.json report next to the bag.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
RUN_NAME="gazebo_throughput_$(date +%Y%m%d_%H%M%S)"
OUT_DIR=""
DURATION_SECS=60
TOPIC_TIMEOUT_S=30
COMMAND_WARMUP_S=3
MACHINE_ROLE="${LEWM_BENCHMARK_MACHINE_ROLE:-unspecified}"
CAPTURE_PROFILE="vision"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      RUN_NAME="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --duration)
      DURATION_SECS="$2"
      shift 2
      ;;
    --topic-timeout)
      TOPIC_TIMEOUT_S="$2"
      shift 2
      ;;
    --machine-role)
      MACHINE_ROLE="$2"
      shift 2
      ;;
    --capture-profile)
      CAPTURE_PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: benchmark_gazebo_throughput.sh [options]

Options:
  --name STR           Run name; output goes to .generated/benchmarks/<name>/.
  --out PATH           Absolute output directory.
  --duration N         Bag duration in seconds (default 60).
  --topic-timeout N    Seconds to wait for required topics (default 30).
  --machine-role STR   Context label, e.g. dev_laptop or production_desktop.
  --capture-profile STR vision records RGB in rosbag; raw omits RGB payloads.

Requires a simulator already running through scripts/launch_go2_sim.sh.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$CAPTURE_PROFILE" != "vision" && "$CAPTURE_PROFILE" != "raw" ]]; then
  echo "Unknown capture profile: $CAPTURE_PROFILE (expected vision or raw)" >&2
  exit 2
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/.generated/benchmarks/$RUN_NAME"
fi

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"
lewm_source_jazzy_underlay
lewm_source_workspace_overlay "$REPO_ROOT"

LOG_PATH="$OUT_DIR/record.log"
BAG_DIR="$OUT_DIR/bag"
RAW_DIR="$OUT_DIR/raw_rollout"
REPORT_PATH="$OUT_DIR/throughput.json"
DRIVE_PID=""

cleanup() {
  if [[ -n "${DRIVE_PID:-}" ]] && kill -0 "$DRIVE_PID" 2>/dev/null; then
    kill -INT "$DRIVE_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$DRIVE_PID" 2>/dev/null; then
        wait "$DRIVE_PID" 2>/dev/null || true
        DRIVE_PID=""
        return 0
      fi
      sleep 0.1
    done
    kill -KILL "$DRIVE_PID" 2>/dev/null || true
    wait "$DRIVE_PID" 2>/dev/null || true
  fi
  DRIVE_PID=""
}
trap cleanup EXIT

await_topics() {
  local deadline=$((SECONDS + TOPIC_TIMEOUT_S))
  local required=(/clock /cmd_vel /lewm/go2/command_block /lewm/go2/executed_command_block)
  if [[ "$CAPTURE_PROFILE" == "vision" ]]; then
    required+=(/rgb_image)
  fi
  while true; do
    local listing
    listing="$(ros2 topic list 2>/dev/null || true)"
    local missing=()
    for topic in "${required[@]}"; do
      if ! printf '%s\n' "$listing" | grep -qx "$topic"; then
        missing+=("$topic")
      fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
      return 0
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "Timed out waiting for topics: ${missing[*]}" >&2
      printf '%s\n' "$listing" >&2
      return 1
    fi
    sleep 1
  done
}

drive_command_blocks() {
  local cycles="${1:-$((DURATION_SECS - COMMAND_WARMUP_S - 3))}"
  if [[ "$cycles" -lt 1 ]]; then cycles=1; fi
  local base_sequence=$(( ($(date +%s) % 1000000) * 1000 ))
  local primitive
  for i in $(seq 1 "$cycles"); do
    if (( i % 4 == 0 )); then
      primitive="arc_right"
    else
      primitive="forward_medium"
    fi
    ros2 topic pub --once --wait-matching-subscriptions 2 --max-wait-time-secs 5 \
      /lewm/go2/command_block lewm_go2_control/msg/CommandBlock \
      "{sequence_id: $((base_sequence + i)), block_size: 5, command_dt_s: 0.1, primitive_name: '$primitive'}" \
      >/dev/null 2>&1 || true
    sleep 1
  done
}

wait_for_bag_subscriptions() {
  local deadline=$((SECONDS + 20))
  while true; do
    if [[ -f "$LOG_PATH" ]] && grep -q "All requested topics are subscribed" "$LOG_PATH"; then
      return 0
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "Timed out waiting for rosbag subscriptions." >&2
      return 1
    fi
    sleep 0.5
  done
}

echo "Waiting for Gazebo/LeWM topics..."
await_topics

echo "Recording benchmark bag to $BAG_DIR for ${DURATION_SECS}s (capture_profile=${CAPTURE_PROFILE})..."
"$SCRIPT_DIR/record_smoke_bag.sh" --out "$BAG_DIR" --duration "$DURATION_SECS" \
  --profile "$CAPTURE_PROFILE" \
  > >(tee "$LOG_PATH") 2>&1 &
RECORD_PID=$!
RECORD_START_S=$SECONDS

wait_for_bag_subscriptions
ros2 service call /lewm/go2/reset lewm_go2_control/srv/ResetEpisode \
  "{scene_id: 0, reason: 'throughput_benchmark', use_spawn_pose: false}" \
  >/dev/null
sleep "$COMMAND_WARMUP_S"
ELAPSED_S=$((SECONDS - RECORD_START_S))
DRIVE_CYCLES=$((DURATION_SECS - ELAPSED_S - 3))
if [[ "$DRIVE_CYCLES" -lt 1 ]]; then DRIVE_CYCLES=1; fi
drive_command_blocks "$DRIVE_CYCLES" &
DRIVE_PID=$!

wait "$RECORD_PID"
cleanup

echo "Converting benchmark bag..."
CONVERT_START_S=$SECONDS
QUALITY_PROFILE="pilot"
if [[ "$CAPTURE_PROFILE" == "raw" ]]; then
  QUALITY_PROFILE="raw_pilot"
fi
"$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" \
  "$BAG_DIR" --out "$RAW_DIR" --quality-profile "$QUALITY_PROFILE" --no-strict
CONVERT_DURATION_S=$((SECONDS - CONVERT_START_S))

python3 - "$RAW_DIR/summary.json" "$BAG_DIR" "$LOG_PATH" "$REPORT_PATH" \
  "$MACHINE_ROLE" "$DURATION_SECS" "$COMMAND_WARMUP_S" "$CONVERT_DURATION_S" "$CAPTURE_PROFILE" <<'PY'
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

summary_path = Path(sys.argv[1])
bag_dir = Path(sys.argv[2])
log_path = Path(sys.argv[3])
report_path = Path(sys.argv[4])
machine_role = sys.argv[5]
requested_duration_s = float(sys.argv[6])
command_warmup_s = float(sys.argv[7])
conversion_duration_s = float(sys.argv[8])
capture_profile = sys.argv[9]

summary = json.loads(summary_path.read_text(encoding="utf-8"))
messages_path = Path(summary["messages_jsonl"])

first_clock: float | None = None
last_clock: float | None = None
rgb_header_times: list[float] = []

def stamp_s(payload: dict[str, Any]) -> float | None:
    stamp = payload.get("header", {}).get("stamp")
    if not isinstance(stamp, dict):
        return None
    return float(stamp.get("sec", 0)) + float(stamp.get("nanosec", 0)) / 1e9

with messages_path.open(encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        topic = record.get("topic")
        payload = record.get("payload", {})
        if topic == "/clock":
            clock = payload.get("clock", {})
            t = float(clock.get("sec", 0)) + float(clock.get("nanosec", 0)) / 1e9
            if first_clock is None:
                first_clock = t
            last_clock = t
        elif topic == "/rgb_image":
            t = stamp_s(payload)
            if t is not None:
                rgb_header_times.append(t)

wall_duration_s = float(summary.get("duration_s") or 0.0)
sim_duration_s = (
    None if first_clock is None or last_clock is None else max(0.0, last_clock - first_clock)
)
counts = summary.get("topic_counts", {})
bag_bytes = sum(path.stat().st_size for path in bag_dir.glob("*.mcap"))

log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
lost_or_drop_lines = [
    line for line in log_text.splitlines()
    if "lost" in line.lower() or "drop" in line.lower()
]

def command_output(argv: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            argv,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip() or None
    except Exception:
        return None

def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None

def mem_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None

host = {
    "machine_role": machine_role,
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "kernel": platform.release(),
    "cpu_count": os.cpu_count(),
    "cpu_model": cpu_model(),
    "mem_total_bytes": mem_total_bytes(),
    "gpu_lspci": command_output(["bash", "-lc", "command -v lspci >/dev/null && lspci | grep -Ei 'vga|3d|display'"]),
}

report = {
    "schema": "lewm_gazebo_throughput_v0",
    "benchmark_context": {
        "machine_role": machine_role,
        "capture_profile": capture_profile,
        "requested_duration_s": requested_duration_s,
        "command_warmup_s": command_warmup_s,
        "production_throughput_authority": machine_role == "production_desktop",
    },
    "host": host,
    "summary_json": str(summary_path),
    "bag_dir": str(bag_dir),
    "wall_duration_s": wall_duration_s,
    "sim_duration_s": sim_duration_s,
    "real_time_factor": (
        None if not wall_duration_s or sim_duration_s is None else sim_duration_s / wall_duration_s
    ),
    "bag_bytes": bag_bytes,
    "bag_gb_per_min": (
        None if not wall_duration_s else (bag_bytes / 1e9) / (wall_duration_s / 60.0)
    ),
    "message_count": summary.get("message_count", 0),
    "message_rate_hz": (
        None if not wall_duration_s else float(summary.get("message_count", 0)) / wall_duration_s
    ),
    "conversion_duration_s": conversion_duration_s,
    "conversion_message_rate_hz": (
        None
        if not conversion_duration_s
        else float(summary.get("message_count", 0)) / conversion_duration_s
    ),
    "rgb_frame_count": counts.get("/rgb_image", 0),
    "rgb_fps_wall": (
        None if not wall_duration_s else float(counts.get("/rgb_image", 0)) / wall_duration_s
    ),
    "rgb_fps_sim": (
        None
        if not sim_duration_s
        else float(counts.get("/rgb_image", 0)) / sim_duration_s
    ),
    "command_block_count": counts.get("/lewm/go2/command_block", 0),
    "executed_command_block_count": counts.get("/lewm/go2/executed_command_block", 0),
    "contract_audit": summary.get("contract_audit", {}),
    "data_quality_audit": summary.get("data_quality_audit", {}),
    "topic_audit": summary.get("topic_audit", {}),
    "lost_or_drop_line_count": len(lost_or_drop_lines),
    "lost_or_drop_lines": lost_or_drop_lines[:20],
    "camera_valid_audit_supported": False,
    "render_invalid_frame_rate": None,
    "topic_counts": counts,
}

report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {report_path}")
print(
    "throughput:"
    f" capture_profile={report['benchmark_context']['capture_profile']}"
    f" rtf={report['real_time_factor']}"
    f" rgb_fps_wall={report['rgb_fps_wall']}"
    f" bag_gb_per_min={report['bag_gb_per_min']}"
    f" conversion_s={report['conversion_duration_s']}"
    f" contract_pass={report['contract_audit'].get('pass')}"
    f" data_quality_pass={report['data_quality_audit'].get('pass')}"
)
PY

echo "Benchmark report: $REPORT_PATH"

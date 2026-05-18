#!/usr/bin/env bash
# Render short review clips for each collection-policy/data mode.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

ROOT="$(lewm_repo_root)"
cd "$ROOT"

CORPUS_DIR="${CORPUS_DIR:-$ROOT/.generated/scene_corpus/acceptance}"
RUN_NAME="${RUN_NAME:-data_mode_review_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/.generated/data_mode_review/$RUN_NAME}"
BACKEND_RENDER="${BACKEND_RENDER:-amdgpu}"
N_ENVS="${N_ENVS:-4}"
N_BLOCKS="${N_BLOCKS:-48}"
RECOVERY_BLOCKS="${RECOVERY_BLOCKS:-80}"
ROUTE_SOLVE_BLOCKS="${ROUTE_SOLVE_BLOCKS:-240}"
MAX_FRAMES="${MAX_FRAMES:-180}"
ROUTE_SOLVE_MAX_FRAMES="${ROUTE_SOLVE_MAX_FRAMES:-700}"
LONG_ROUTE_FIXED_SPAWN="${LONG_ROUTE_FIXED_SPAWN:-1}"
FPS="${FPS:-10}"
ENV_INDEX="${ENV_INDEX:-0}"
RENDER_TOP_DOWN="${RENDER_TOP_DOWN:-1}"
INCLUDE_ROUTE_SOLVE="${INCLUDE_ROUTE_SOLVE:-0}"
# When set to 1, skip every other mode and render only long_route_teacher.
# Useful for iterating on the route-teacher planner without re-rendering
# the six unaffected modes each time.
ONLY_LONG_ROUTE="${ONLY_LONG_ROUTE:-0}"

PHYSICS_ROOT="$OUT_ROOT/physics"
RAW_ROOT="$OUT_ROOT/raw"
PLAN_ROOT="$OUT_ROOT/plans"
RENDER_ROOT="$OUT_ROOT/rendered"
RENDER_TOP_ROOT="$OUT_ROOT/rendered_top"
VIDEO_ROOT="$OUT_ROOT/videos"
VIDEO_TOP_ROOT="$OUT_ROOT/videos_top"
VIDEO_BY_MODE="$OUT_ROOT/videos_by_mode"
LOG_ROOT="$OUT_ROOT/logs"

mkdir -p "$PHYSICS_ROOT" "$RAW_ROOT" "$PLAN_ROOT" "$RENDER_ROOT" \
  "$RENDER_TOP_ROOT" "$VIDEO_ROOT" "$VIDEO_TOP_ROOT" "$VIDEO_BY_MODE" "$LOG_ROOT"

if [[ ! -d "$CORPUS_DIR" ]]; then
  echo "scene corpus not found: $CORPUS_DIR" >&2
  exit 2
fi

run_logged() {
  local label="$1"
  shift
  local logfile="$LOG_ROOT/${label}.log"
  echo "[$label] $*" >&2
  if ! "$@" >"$logfile" 2>&1; then
    echo "[$label] failed; see $logfile" >&2
    tail -n 40 "$logfile" >&2 || true
    return 1
  fi
}

run_logged_nonfatal() {
  local label="$1"
  shift
  local logfile="$LOG_ROOT/${label}.log"
  echo "[$label] $*" >&2
  if ! "$@" >"$logfile" 2>&1; then
    echo "[$label] non-fatal nonzero exit; see $logfile" >&2
    tail -n 8 "$logfile" >&2 || true
  fi
  return 0
}

run_mode() {
  local mode="$1"
  local family="$2"
  local blocks="$3"
  local policy="${4:-$mode}"
  local max_frames="${5:-$MAX_FRAMES}"
  local fixed_spawn="${6:-0}"
  local recovery_interlock_clearance="${7:-}"

  local physics_out="$PHYSICS_ROOT/$mode"
  local raw_parent="$RAW_ROOT/$mode"
  local plan_out="$PLAN_ROOT/$mode"
  local render_out="$RENDER_ROOT/$mode"
  local render_top_out="$RENDER_TOP_ROOT/$mode"
  local video_out="$VIDEO_ROOT/$mode"
  local video_top_out="$VIDEO_TOP_ROOT/$mode"
  local mode_video_dir="$VIDEO_BY_MODE/$mode"
  local rollout_extra=()

  if [[ "$fixed_spawn" == "1" ]]; then
    rollout_extra+=(--fixed-spawn)
  fi
  if [[ -n "$recovery_interlock_clearance" ]]; then
    rollout_extra+=(--recovery-interlock-clearance-m "$recovery_interlock_clearance")
  fi

  mkdir -p "$raw_parent" "$plan_out" "$video_out" "$video_top_out" "$mode_video_dir"

  run_logged "physics_${mode}" \
    bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" \
      --scene-corpus "$CORPUS_DIR" \
      --split train \
      --family "$family" \
      --scene-limit 1 \
      --n-envs "$N_ENVS" \
      --n-blocks "$blocks" \
      --backend cpu \
      --no-rgb \
      --collector-mix "${policy}=1.0" \
      --log-progress-every-blocks 0 \
      "${rollout_extra[@]}" \
      --out "$physics_out"

  local scene_dir
  scene_dir="$(find "$physics_out" -maxdepth 1 -mindepth 1 -type d -name "${family}_*" | sort | head -n 1)"
  if [[ -z "$scene_dir" ]]; then
    echo "no scene output found for $mode under $physics_out" >&2
    exit 3
  fi

  local scene_name
  scene_name="$(basename "$scene_dir")"
  local raw_out="$raw_parent/$scene_name"
  run_logged "convert_${mode}" \
    bash "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" \
      "$scene_dir" \
      --out "$raw_out" \
      --quality-profile raw_pilot

  run_logged "plan_${mode}" \
    bash "$SCRIPT_DIR/plan_bulk_render_replay.sh" \
      "$raw_out" \
      --out-root "$plan_out" \
      --camera-hz "$FPS"

  local plan_json
  plan_json="$(find "$plan_out" -maxdepth 3 -name render_replay_plan.json | sort | head -n 1)"
  if [[ -z "$plan_json" ]]; then
    echo "no render plan produced for $mode under $plan_out" >&2
    exit 4
  fi

  run_logged_nonfatal "render_${mode}" \
    bash "$SCRIPT_DIR/render_replay_genesis.sh" \
      "$plan_json" \
      --scene-corpus "$CORPUS_DIR" \
      --backend "$BACKEND_RENDER" \
      --camera-mode replay \
      --env-index "$ENV_INDEX" \
      --max-frames "$max_frames" \
      --rgb-format png \
      --out "$render_out" \
      --overwrite

  run_logged "video_${mode}" \
    bash "$SCRIPT_DIR/export_scene_type_videos.sh" \
      --rendered-root "$render_out" \
      --scene-corpus "$CORPUS_DIR" \
      --out "$video_out" \
      --per-family 1 \
      --env-index "$ENV_INDEX" \
      --fps "$FPS" \
      --max-frames "$max_frames" \
      --format mp4 \
      --include-invalid \
      --overwrite

  local mp4
  mp4="$(find "$video_out" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 1)"
  if [[ -z "$mp4" ]]; then
    echo "no mp4 produced for $mode under $video_out" >&2
    exit 5
  fi
  cp "$mp4" "$mode_video_dir/${mode}__ego_all_frames.mp4"
  cp "$render_out/summary.json" "$mode_video_dir/${mode}__render_summary.json"

  if [[ "$RENDER_TOP_DOWN" != "0" ]]; then
    run_logged_nonfatal "render_top_${mode}" \
      bash "$SCRIPT_DIR/render_replay_genesis.sh" \
        "$plan_json" \
        --scene-corpus "$CORPUS_DIR" \
        --backend "$BACKEND_RENDER" \
        --camera-mode overview \
        --env-index "$ENV_INDEX" \
        --max-frames "$max_frames" \
        --rgb-format png \
        --out "$render_top_out" \
        --overwrite

    run_logged "video_top_${mode}" \
      bash "$SCRIPT_DIR/export_scene_type_videos.sh" \
        --rendered-root "$render_top_out" \
        --scene-corpus "$CORPUS_DIR" \
        --out "$video_top_out" \
        --per-family 1 \
        --env-index "$ENV_INDEX" \
        --fps "$FPS" \
        --max-frames "$max_frames" \
        --format mp4 \
        --include-invalid \
        --overwrite

    local top_mp4
    top_mp4="$(find "$video_top_out" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 1)"
    if [[ -z "$top_mp4" ]]; then
      echo "no top-down mp4 produced for $mode under $video_top_out" >&2
      exit 6
    fi
    cp "$top_mp4" "$mode_video_dir/${mode}__top_down.mp4"
    cp "$render_top_out/summary.json" "$mode_video_dir/${mode}__top_down_render_summary.json"
  fi
}

echo "=== data mode review suite ==="
echo "corpus: $CORPUS_DIR"
echo "out:    $OUT_ROOT"
echo "render backend: $BACKEND_RENDER"

if [[ "$ONLY_LONG_ROUTE" == "0" ]]; then
  run_mode route_teacher small_enclosed_maze "$N_BLOCKS"
  run_mode frontier small_enclosed_maze "$N_BLOCKS"
  run_mode primitive_curriculum open_obstacle_field "$N_BLOCKS"
  run_mode ou_noise open_obstacle_field "$N_BLOCKS"
  run_mode recovery small_enclosed_maze "$RECOVERY_BLOCKS"
  run_mode loop_revisit loop_alias_stress "$N_BLOCKS"
fi
if [[ "$ONLY_LONG_ROUTE" == "1" || "$INCLUDE_ROUTE_SOLVE" != "0" ]]; then
  # Disable the rollout-level recovery interlock for long_route_teacher:
  # the route teacher's own motion-gated stuck FSM handles wedge cases
  # in-band, and the rollout interlock's 16-block recovery handoff
  # otherwise steals state from the teacher every few seconds. Profiling
  # showed avg cells visited / 240 blocks jumped from 9.2 → 17.2 once
  # the two FSMs stopped fighting.
  run_mode long_route_teacher large_enclosed_maze \
    "$ROUTE_SOLVE_BLOCKS" route_teacher "$ROUTE_SOLVE_MAX_FRAMES" "$LONG_ROUTE_FIXED_SPAWN" 0
fi

python3 - "$OUT_ROOT" "$VIDEO_BY_MODE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
video_by_mode = Path(sys.argv[2])
videos = []
for video in sorted(video_by_mode.glob("*/*.mp4")):
    mode = video.parent.name
    camera_mode = "overview" if video.name.endswith("__top_down.mp4") else "replay"
    summary_name = (
        f"{mode}__top_down_render_summary.json"
        if camera_mode == "overview"
        else f"{mode}__render_summary.json"
    )
    summary = video.with_name(summary_name)
    payload = {}
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
    videos.append(
        {
            "mode": mode,
            "camera_mode": camera_mode,
            "video": str(video),
            "render_summary": str(summary) if summary.is_file() else None,
            "frame_count": payload.get("frame_count"),
            "invalid_frame_count": payload.get("invalid_frame_count"),
            "low_info_frame_count": payload.get("low_info_frame_count"),
            "low_info_allowed_frame_count": payload.get("low_info_allowed_frame_count"),
        }
    )
manifest = {
    "schema": "lewm_data_mode_review_suite_v0",
    "out_root": str(out_root),
    "videos_by_mode": str(video_by_mode),
    "videos": videos,
}
path = out_root / "manifest.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"manifest": str(path), "video_count": len(videos)}, sort_keys=True))
PY

echo "videos_by_mode: $VIDEO_BY_MODE"
echo "manifest: $OUT_ROOT/manifest.json"

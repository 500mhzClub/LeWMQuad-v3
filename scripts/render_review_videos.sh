#!/usr/bin/env bash
# Render a small set of overview-camera review videos for manual validation.
#
# For each family: roll out the first N scenes (1 env, route_teacher),
# convert to raw_rollout, render-replay with the static top-down overview
# camera, and stitch the frames into an mp4. Output mp4s land flat under
# --out so they are easy to scp.
#
# Usage:
#   scripts/render_review_videos.sh --scene-corpus PATH --out PATH \
#       [--scenes-per-family 2] [--n-blocks 600] [--backend cpu] \
#       [--families "fam1 fam2 ..."]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"
REPO_ROOT="$(lewm_repo_root)"
cd "$REPO_ROOT"

CORPUS=""
OUT=""
SCENES_PER_FAMILY=2
N_BLOCKS=600
BACKEND="cpu"
FPS=15
CAMERA_HZ=10
SPLIT="train"
FAMILIES="open_obstacle_field rough_local_dynamics loop_alias_stress local_composite_motifs small_enclosed_maze medium_enclosed_maze visual_sensor_stress large_enclosed_maze"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-corpus) CORPUS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --scenes-per-family) SCENES_PER_FAMILY="$2"; shift 2 ;;
    --n-blocks) N_BLOCKS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --families) FAMILIES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$CORPUS" || -z "$OUT" ]] && { echo "need --scene-corpus and --out" >&2; exit 2; }
mkdir -p "$OUT"
WORK="$OUT/work"
mkdir -p "$WORK"

manifest_count=0
for FAMILY in $FAMILIES; do
  echo "=== family: $FAMILY ==="
  roll="$WORK/rollout/$FAMILY"
  bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" \
    --scene-corpus "$CORPUS" \
    --split "$SPLIT" --family "$FAMILY" --scene-limit "$SCENES_PER_FAMILY" \
    --n-envs 1 --n-blocks "$N_BLOCKS" --backend "$BACKEND" --no-rgb \
    --collector-mix route_teacher=1.0 \
    --recovery-interlock-clearance-m 0 \
    --log-progress-every-blocks 0 \
    --out "$roll" >"$WORK/${FAMILY}_rollout.log" 2>&1 || {
      echo "  rollout FAILED for $FAMILY (see $WORK/${FAMILY}_rollout.log)"; continue; }

  for scene_dir in "$roll"/*/; do
    scene=$(basename "$scene_dir")
    [[ -f "$scene_dir/summary.json" ]] || continue
    raw="$WORK/raw/$FAMILY/$scene"
    bash "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" \
      "$scene_dir" --out "$raw" --quality-profile raw_pilot >/dev/null 2>&1 || {
        echo "  convert FAILED $scene"; continue; }

    plandir="$WORK/plan/$FAMILY/$scene"
    bash "$SCRIPT_DIR/plan_bulk_render_replay.sh" \
      --raw-root "$(dirname "$raw")" --out-root "$plandir" \
      --backend "$BACKEND" --camera-hz "$CAMERA_HZ" >/dev/null 2>&1 || true
    plan=$(find "$plandir" -name render_replay_plan.json | grep "/${scene}/" -m1 || \
           find "$plandir" -name render_replay_plan.json -path "*${scene}*" | head -1)
    [[ -z "$plan" ]] && plan=$(find "$plandir" -name render_replay_plan.json | head -1)
    [[ -z "$plan" ]] && { echo "  no plan for $scene"; continue; }

    rdir="$WORK/render/$FAMILY/$scene"
    bash "$SCRIPT_DIR/render_replay_genesis.sh" "$plan" \
      --scene-corpus "$CORPUS" --backend "$BACKEND" \
      --camera-mode overview --rgb-format png --no-depth \
      --out "$rdir" >"$WORK/${FAMILY}_${scene}_render.log" 2>&1 || {
        echo "  render FAILED $scene (see log)"; continue; }

    # Stitch env-00 frames into an mp4.
    mp4="$OUT/${FAMILY}__${scene}.mp4"
    if ffmpeg -y -framerate "$FPS" -pattern_type glob \
        -i "$rdir/rgb/frame_*_env_00.png" \
        -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        "$mp4" >/dev/null 2>&1; then
      frames=$(ls "$rdir"/rgb/frame_*_env_00.png 2>/dev/null | wc -l)
      echo "  wrote $mp4 ($frames frames)"
      manifest_count=$((manifest_count+1))
    else
      echo "  ffmpeg FAILED $scene"
    fi
  done
done

echo
echo "=== rendered $manifest_count videos under $OUT ==="
ls -la "$OUT"/*.mp4 2>/dev/null || echo "(no mp4s produced)"

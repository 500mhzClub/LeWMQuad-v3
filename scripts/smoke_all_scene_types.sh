#!/usr/bin/env bash
# Full smoke suite for every registered scene family.
#
# For each family in the target scene corpus, this script runs:
#   1. CPU Genesis physics rollout (writer enabled, no inline RGB) for one scene
#   2. raw_rollout conversion with --quality-profile raw_training
#   3. derived label pass (privileged per-step labels)
#   4. render replay plan
#   5. render replay in egocentric mode (AMDGPU)
#   6. render replay in overview (top-down) mode (AMDGPU)
#
# The audit assertion suite is then run against the produced artifacts.
# Per-family egocentric and top-down videos are written for manual review.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"
REPO_ROOT="$(lewm_repo_root)"
cd "$REPO_ROOT"

CORPUS_NAME="${CORPUS_NAME:-smoke_all_families}"
CORPUS_DIR="$REPO_ROOT/.generated/scene_corpus/$CORPUS_NAME"
RUN_NAME="${RUN_NAME:-smoke_all_scene_types_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="$REPO_ROOT/.generated/smoke_suite/$RUN_NAME"
PHYSICS_ROOT="$OUT_ROOT/physics"
RAW_ROOT="$OUT_ROOT/raw"
LABELS_ROOT="$OUT_ROOT/derived_labels"
PLAN_ROOT_EGO="$OUT_ROOT/plans_ego"
PLAN_ROOT_TOP="$OUT_ROOT/plans_top"
RENDER_ROOT_EGO="$OUT_ROOT/rendered_ego"
RENDER_ROOT_TOP="$OUT_ROOT/rendered_top"
VIDEO_ROOT="$OUT_ROOT/videos"
LOG_ROOT="$OUT_ROOT/logs"

N_ENVS="${N_ENVS:-8}"
N_BLOCKS="${N_BLOCKS:-100}"
REPLAY_ENV_INDEX="${REPLAY_ENV_INDEX:-0}"
RENDER_MAX_FRAMES="${RENDER_MAX_FRAMES:-300}"

mkdir -p "$OUT_ROOT" "$PHYSICS_ROOT" "$RAW_ROOT" "$LABELS_ROOT" \
  "$PLAN_ROOT_EGO" "$PLAN_ROOT_TOP" "$RENDER_ROOT_EGO" "$RENDER_ROOT_TOP" \
  "$VIDEO_ROOT" "$LOG_ROOT"

if [[ ! -d "$CORPUS_DIR" ]]; then
  echo "ERROR: scene corpus not found: $CORPUS_DIR" >&2
  echo "Run scripts/generate_scene_corpus.sh --name $CORPUS_NAME first." >&2
  exit 2
fi

mapfile -t FAMILIES < <(ls "$CORPUS_DIR/train" 2>/dev/null | sort)
if [[ ${#FAMILIES[@]} -eq 0 ]]; then
  echo "ERROR: no families under $CORPUS_DIR/train" >&2
  exit 2
fi

echo "=== smoke suite ==="
echo "  corpus:    $CORPUS_DIR"
echo "  out:       $OUT_ROOT"
echo "  families:  ${FAMILIES[*]}"
echo "  n_envs:    $N_ENVS"
echo "  n_blocks:  $N_BLOCKS"

run_logged() {
  local label="$1"
  shift
  local logfile="$LOG_ROOT/${label}.log"
  echo "[$label] $*" >&2
  if ! "$@" >"$logfile" 2>&1; then
    echo "[$label] FAILED -- see $logfile" >&2
    tail -n 40 "$logfile" >&2 || true
    return 1
  fi
}

# Like run_logged but never propagates the failure; logs continue. Used for
# render-replay stages where the renderer's exit==2 (any invalid frame) should
# surface in the assertion report rather than abort the per-family loop.
run_logged_nonfatal() {
  local label="$1"
  shift
  local logfile="$LOG_ROOT/${label}.log"
  echo "[$label] $*" >&2
  if ! "$@" >"$logfile" 2>&1; then
    echo "[$label] NON-FATAL: exit non-zero -- see $logfile" >&2
    tail -n 5 "$logfile" >&2 || true
  fi
  return 0
}

for FAMILY in "${FAMILIES[@]}"; do
  echo "=== family: $FAMILY ==="

  PHYS_OUT="$PHYSICS_ROOT/$FAMILY"
  run_logged "physics_${FAMILY}" \
    bash "$SCRIPT_DIR/genesis_bulk_rollout.sh" \
      --scene-corpus "$CORPUS_DIR" \
      --split train \
      --family "$FAMILY" \
      --scene-limit 1 \
      --n-envs "$N_ENVS" \
      --n-blocks "$N_BLOCKS" \
      --backend cpu \
      --no-rgb \
      --out "$PHYS_OUT"

  # Find the produced scene directory
  SCENE_DIR=$(find "$PHYS_OUT" -maxdepth 1 -mindepth 1 -type d -name "${FAMILY}_*" | head -n 1)
  if [[ -z "$SCENE_DIR" ]]; then
    echo "ERROR: no scene output under $PHYS_OUT" >&2
    exit 3
  fi
  SCENE_NAME=$(basename "$SCENE_DIR")

  RAW_OUT="$RAW_ROOT/$FAMILY/$SCENE_NAME"
  mkdir -p "$RAW_ROOT/$FAMILY"
  run_logged "convert_${FAMILY}" \
    bash "$SCRIPT_DIR/convert_smoke_bag_to_raw_rollout.sh" \
      "$SCENE_DIR" \
      --out "$RAW_OUT" \
      --quality-profile raw_training

  LABELS_OUT="$LABELS_ROOT/$FAMILY/$SCENE_NAME"
  mkdir -p "$LABELS_ROOT/$FAMILY"
  SCENE_MANIFEST="$CORPUS_DIR/train/$FAMILY/$SCENE_NAME/manifest.json"
  if [[ ! -f "$SCENE_MANIFEST" ]]; then
    echo "ERROR: scene manifest missing at $SCENE_MANIFEST" >&2
    exit 5
  fi
  run_logged "labels_${FAMILY}" \
    python3 "$SCRIPT_DIR/derive_raw_rollout_labels.py" \
      "$RAW_OUT" \
      --scene-manifest "$SCENE_MANIFEST" \
      --out "$LABELS_OUT"

  PLAN_EGO_OUT="$PLAN_ROOT_EGO/$FAMILY"
  mkdir -p "$PLAN_EGO_OUT"
  run_logged "plan_ego_${FAMILY}" \
    bash "$SCRIPT_DIR/plan_bulk_render_replay.sh" \
      "$RAW_OUT" \
      --out-root "$PLAN_EGO_OUT" \
      --camera-hz 10
  PLAN_EGO_JSON=$(find "$PLAN_EGO_OUT" -maxdepth 3 -name render_replay_plan.json | head -n 1)
  if [[ -z "$PLAN_EGO_JSON" ]]; then
    echo "ERROR: no ego plan produced for $FAMILY" >&2
    exit 4
  fi

  PLAN_TOP_OUT="$PLAN_ROOT_TOP/$FAMILY"
  mkdir -p "$PLAN_TOP_OUT"
  run_logged "plan_top_${FAMILY}" \
    bash "$SCRIPT_DIR/plan_bulk_render_replay.sh" \
      "$RAW_OUT" \
      --out-root "$PLAN_TOP_OUT" \
      --camera-hz 10
  PLAN_TOP_JSON=$(find "$PLAN_TOP_OUT" -maxdepth 3 -name render_replay_plan.json | head -n 1)

  RENDER_EGO_OUT="$RENDER_ROOT_EGO/$FAMILY"
  run_logged_nonfatal "render_ego_${FAMILY}" \
    bash "$SCRIPT_DIR/render_replay_genesis.sh" \
      "$PLAN_EGO_JSON" \
      --scene-corpus "$CORPUS_DIR" \
      --backend amdgpu \
      --camera-mode replay \
      --env-index "$REPLAY_ENV_INDEX" \
      --max-frames "$RENDER_MAX_FRAMES" \
      --rgb-format png \
      --out "$RENDER_EGO_OUT" \
      --overwrite

  RENDER_TOP_OUT="$RENDER_ROOT_TOP/$FAMILY"
  run_logged_nonfatal "render_top_${FAMILY}" \
    bash "$SCRIPT_DIR/render_replay_genesis.sh" \
      "$PLAN_TOP_JSON" \
      --scene-corpus "$CORPUS_DIR" \
      --backend amdgpu \
      --camera-mode overview \
      --env-index "$REPLAY_ENV_INDEX" \
      --max-frames "$RENDER_MAX_FRAMES" \
      --rgb-format png \
      --out "$RENDER_TOP_OUT" \
      --overwrite
done

echo "=== compose ego + top-down videos ==="
run_logged "videos_ego" \
  bash "$SCRIPT_DIR/export_scene_type_videos.sh" \
    --rendered-root "$RENDER_ROOT_EGO" \
    --scene-corpus "$CORPUS_DIR" \
    --out "$VIDEO_ROOT/ego" \
    --per-family 1 \
    --fps 10 \
    --max-frames "$RENDER_MAX_FRAMES" \
    --format mp4 \
    --overwrite
run_logged "videos_top" \
  bash "$SCRIPT_DIR/export_scene_type_videos.sh" \
    --rendered-root "$RENDER_ROOT_TOP" \
    --scene-corpus "$CORPUS_DIR" \
    --out "$VIDEO_ROOT/top" \
    --per-family 1 \
    --fps 10 \
    --max-frames "$RENDER_MAX_FRAMES" \
    --format mp4 \
    --overwrite

echo "=== assertion suite ==="
set +e
python3 "$SCRIPT_DIR/assert_smoke_suite.py" \
  --suite-root "$OUT_ROOT" \
  --corpus "$CORPUS_DIR" \
  --n-envs "$N_ENVS" \
  --n-blocks "$N_BLOCKS"
ASSERT_EXIT=$?
set -e

echo "videos: $VIDEO_ROOT"
echo "report: $OUT_ROOT/assertion_report.json"
if [[ $ASSERT_EXIT -eq 0 ]]; then
  echo "=== smoke suite OK ==="
else
  echo "=== smoke suite FAILED (assertion exit $ASSERT_EXIT) ==="
fi
exit $ASSERT_EXIT

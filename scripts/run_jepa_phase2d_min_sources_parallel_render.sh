#!/usr/bin/env bash
# Render Phase 2D minimum-source counterfactual plans with scene-level parallelism.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LEWM_PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"

PLAN_BASE="$ROOT/.generated/jepa_counterfactual/phase2d_min_sources"
RENDER_BASE="/tmp/lewm_phase2d_min_sources_render_20260614"
SCENE_CORPUS="$ROOT/.generated/scene_corpus/minimum_tex_20260520T211541Z"
READINESS_OUTPUT="$ROOT/.generated/jepa_counterfactual/phase2d_stage8_render_readiness_after_full_render.json"
JOBS=16
SPLITS=(validation test_id test_hard)
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan-base) PLAN_BASE="$2"; shift 2 ;;
    --render-base) RENDER_BASE="$2"; shift 2 ;;
    --scene-corpus) SCENE_CORPUS="$2"; shift 2 ;;
    --readiness-output) READINESS_OUTPUT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --splits)
      IFS=',' read -r -a SPLITS <<< "$2"
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be at least 1" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at: $PYTHON_BIN" >&2
  exit 2
fi

plan_root_for_split() {
  case "$1" in
    train) printf '%s/train_plans' "$PLAN_BASE" ;;
    validation) printf '%s/validation_plans' "$PLAN_BASE" ;;
    test_id) printf '%s/test_id_plans' "$PLAN_BASE" ;;
    test_hard) printf '%s/test_hard_plans' "$PLAN_BASE" ;;
    *) echo "Unknown split: $1" >&2; return 2 ;;
  esac
}

render_root_for_split() {
  case "$1" in
    train) printf '%s/train_render' "$RENDER_BASE" ;;
    validation) printf '%s/validation_render' "$RENDER_BASE" ;;
    test_id) printf '%s/test_id_render' "$RENDER_BASE" ;;
    test_hard) printf '%s/test_hard_render' "$RENDER_BASE" ;;
    *) echo "Unknown split: $1" >&2; return 2 ;;
  esac
}

export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export GENESIS_ROCM_PYTHON="${GENESIS_ROCM_PYTHON:-$PYTHON_BIN}"
export PYTHONPATH="$ROOT:$ROOT/lewm_worlds:${PYTHONPATH:-}"

mkdir -p "$RENDER_BASE" "$(dirname "$READINESS_OUTPUT")"

for SPLIT in "${SPLITS[@]}"; do
  PLAN_ROOT="$(plan_root_for_split "$SPLIT")"
  RENDER_ROOT="$(render_root_for_split "$SPLIT")"
  COMMAND=(
    "$PYTHON_BIN" "$ROOT/scripts/render_jepa_counterfactual_plan_root_parallel.py"
    --jobs "$JOBS"
    --plan-root "$PLAN_ROOT"
    --output-root "$RENDER_ROOT"
    --scene-corpus "$SCENE_CORPUS"
    --backend vulkan
    --camera-mode replay
    --replay-env-mode single
    --rgb-format png
    --store-resolution training
  )
  printf 'SPLIT=%s CMD:' "$SPLIT"
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "${COMMAND[@]}"
  fi
done

READINESS_COMMAND=(
  "$PYTHON_BIN" "$ROOT/scripts/check_jepa_phase2d_render_readiness.py"
  --plan-root "train=$(plan_root_for_split train)"
  --plan-root "validation=$(plan_root_for_split validation)"
  --plan-root "test_id=$(plan_root_for_split test_id)"
  --plan-root "test_hard=$(plan_root_for_split test_hard)"
  --render-root "train=$(render_root_for_split train)"
  --render-root "validation=$(render_root_for_split validation)"
  --render-root "test_id=$(render_root_for_split test_id)"
  --render-root "test_hard=$(render_root_for_split test_hard)"
  --output "$READINESS_OUTPUT"
)
printf 'READINESS CMD:'
printf ' %q' "${READINESS_COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" != "1" ]]; then
  "${READINESS_COMMAND[@]}"
fi

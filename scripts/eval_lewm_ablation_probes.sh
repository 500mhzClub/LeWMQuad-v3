#!/usr/bin/env bash
# Run the cheap A2/A3 readouts used to screen scaled SIGReg/source ablations.
# Use --full-a3 for the slower reachability+history probe after a cell is worth
# promoting beyond screening.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LEWM_PYTHON:-${GENESIS_ROCM_PYTHON:-}}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.generated/venvs/genesis_rocm/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.generated/venvs/genesis_rocm/bin/python"
  else
    PYTHON_BIN="$ROOT/.generated/venvs/genesis_render_vulkan/bin/python"
  fi
fi

CHECKPOINT=""
OUTPUT_DIR=""
ROLLOUT_ROOT="$ROOT/.generated/datagen_full/rollout"
RENDER_ROOT="$ROOT/.generated/datagen_full/render_textured_v03"
MANIFEST_CORPUS="$ROOT/.generated/scene_corpus/minimum_tex_20260520T211541Z"
DEVICE="cpu"
SPLIT="test_id"
A2_SCENES_PER_FAMILY=2
A2_FRAMES_PER_SCENE=120
A2_MAX_PAIRS_PER_SCENE=20000
A3_EVAL_SCENES_PER_FAMILY=2
A3_EVAL_FRAMES_PER_SCENE=160
FULL_A3=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --rollout-root)
      ROLLOUT_ROOT="$2"
      shift 2
      ;;
    --render-root)
      RENDER_ROOT="$2"
      shift 2
      ;;
    --manifest-corpus)
      MANIFEST_CORPUS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --a2-scenes-per-family)
      A2_SCENES_PER_FAMILY="$2"
      shift 2
      ;;
    --a2-frames-per-scene)
      A2_FRAMES_PER_SCENE="$2"
      shift 2
      ;;
    --a2-max-pairs-per-scene)
      A2_MAX_PAIRS_PER_SCENE="$2"
      shift 2
      ;;
    --a3-eval-scenes-per-family)
      A3_EVAL_SCENES_PER_FAMILY="$2"
      shift 2
      ;;
    --a3-eval-frames-per-scene)
      A3_EVAL_FRAMES_PER_SCENE="$2"
      shift 2
      ;;
    --full-a3)
      FULL_A3=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "Error: --checkpoint is required." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(dirname "$CHECKPOINT")/probes"
fi

mkdir -p "$OUTPUT_DIR"
BASE="$(basename "$CHECKPOINT" .pt)"

"$PYTHON_BIN" "$ROOT/scripts/probe_lewm_latent_aliasing.py" \
  --checkpoint "$CHECKPOINT" \
  --rollout-root "$ROLLOUT_ROOT" \
  --render-root "$RENDER_ROOT" \
  --manifest-corpus "$MANIFEST_CORPUS" \
  --split "$SPLIT" \
  --scenes-per-family "$A2_SCENES_PER_FAMILY" \
  --frames-per-scene "$A2_FRAMES_PER_SCENE" \
  --max-pairs-per-scene "$A2_MAX_PAIRS_PER_SCENE" \
  --device "$DEVICE" \
  --output "$OUTPUT_DIR/latent_aliasing_${BASE}_${SPLIT}.json"

A3_MODE_ARGS=(--retrieval-only)
A3_NAME="retrieval_a3"
if [[ "$FULL_A3" == "1" ]]; then
  A3_MODE_ARGS=()
  A3_NAME="reachability_a3"
fi

"$PYTHON_BIN" "$ROOT/scripts/probe_lewm_reachability_a3.py" \
  --checkpoint "$CHECKPOINT" \
  --rollout-root "$ROLLOUT_ROOT" \
  --render-root "$RENDER_ROOT" \
  --manifest-corpus "$MANIFEST_CORPUS" \
  --eval-split "$SPLIT" \
  --eval-scenes-per-family "$A3_EVAL_SCENES_PER_FAMILY" \
  --eval-frames-per-scene "$A3_EVAL_FRAMES_PER_SCENE" \
  --device "$DEVICE" \
  "${A3_MODE_ARGS[@]}" \
  --output "$OUTPUT_DIR/${A3_NAME}_${BASE}_${SPLIT}.json"

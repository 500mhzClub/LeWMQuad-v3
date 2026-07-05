#!/usr/bin/env bash
# Run the first matched Phase 2B pooled/spatial learnability factorial.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LEWM_PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"

TRAIN_DATA=""
EVAL_DATA=""
OUTPUT_ROOT="$ROOT/.generated/jepa_counterfactual/phase2b_bounded_factorial"
EPOCHS=8
BATCH_SIZE=8
LATENT_DIM=48
ENCODER_DEPTH=2
ENCODER_HEADS=3
ENCODER_MLP_RATIO=2
PRED_LAYERS=2
PRED_HEADS=4
PRED_DIM_HEAD=12
PRED_MLP_DIM=96
SIGREG_PROJECTIONS=64
SIGREG_KNOTS=9
SEED=20260614
DEVICE=cpu
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --eval-data) EVAL_DATA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TRAIN_DATA" || -z "$EVAL_DATA" ]]; then
  echo "--train-data and --eval-data are required" >&2
  exit 2
fi

COMMON=(
  --train-data "$TRAIN_DATA"
  --eval-data "$EVAL_DATA"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --latent-dim "$LATENT_DIM"
  --encoder-depth "$ENCODER_DEPTH"
  --encoder-heads "$ENCODER_HEADS"
  --encoder-mlp-ratio "$ENCODER_MLP_RATIO"
  --pred-layers "$PRED_LAYERS"
  --pred-heads "$PRED_HEADS"
  --pred-dim-head "$PRED_DIM_HEAD"
  --pred-mlp-dim "$PRED_MLP_DIM"
  --sigreg-projections "$SIGREG_PROJECTIONS"
  --sigreg-knots "$SIGREG_KNOTS"
  --seed "$SEED"
  --device "$DEVICE"
)

mkdir -p "$OUTPUT_ROOT"
for CELL in pooled spatial_var spatial_no_var; do
  if [[ "$CELL" == "pooled" ]]; then
    COMMAND=(
      "$PYTHON_BIN" "$ROOT/scripts/train_jepa_pooled_lewm_control.py"
      "${COMMON[@]}"
      --output "$OUTPUT_ROOT/$CELL.pt"
    )
  else
    VARIANCE_LAMBDA=1.0
    if [[ "$CELL" == "spatial_no_var" ]]; then
      VARIANCE_LAMBDA=0.0
    fi
    COMMAND=(
      "$PYTHON_BIN" "$ROOT/scripts/train_jepa_spatial_lewm.py"
      "${COMMON[@]}"
      --spatial-variance-lambda "$VARIANCE_LAMBDA"
      --output "$OUTPUT_ROOT/$CELL.pt"
    )
  fi
  printf 'CELL=%s CMD:' "$CELL"
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    PYTHONPATH="$ROOT:$ROOT/lewm_worlds" "${COMMAND[@]}"
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  PYTHONPATH="$ROOT:$ROOT/lewm_worlds" "$PYTHON_BIN" \
    "$ROOT/scripts/analyze_jepa_phase2b_factorial.py" \
    --input-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/analysis.json"
fi

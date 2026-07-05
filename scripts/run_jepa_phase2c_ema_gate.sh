#!/usr/bin/env bash
# Run the bounded Phase 2C stop-gradient EMA target-encoder gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LEWM_PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"

TRAIN_DATA=""
EVAL_DATA=""
OUTPUT_ROOT="$ROOT/.generated/jepa_counterfactual/phase2c_ema_gate"
PHASE2B_ROOT="$ROOT/.generated/jepa_counterfactual/phase2b_bounded_factorial"
EPOCHS=8
BATCH_SIZE=8
SEED=20260614
DEVICE=cpu
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --eval-data) EVAL_DATA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --phase2b-root) PHASE2B_ROOT="$2"; shift 2 ;;
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

mkdir -p "$OUTPUT_ROOT"
COMMAND=(
  "$PYTHON_BIN" "$ROOT/scripts/train_jepa_spatial_lewm.py"
  --train-data "$TRAIN_DATA"
  --eval-data "$EVAL_DATA"
  --output "$OUTPUT_ROOT/spatial_ema_var.pt"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --latent-dim 48
  --encoder-depth 2
  --encoder-heads 3
  --encoder-mlp-ratio 2
  --pred-layers 2
  --pred-heads 4
  --pred-dim-head 12
  --pred-mlp-dim 96
  --sigreg-projections 64
  --sigreg-knots 9
  --spatial-variance-lambda 1.0
  --target-ema-momentum 0.99
  --seed "$SEED"
  --device "$DEVICE"
)
printf 'CELL=spatial_ema_var CMD:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

PYTHONPATH="$ROOT:$ROOT/lewm_worlds" "${COMMAND[@]}"
PYTHONPATH="$ROOT:$ROOT/lewm_worlds" "$PYTHON_BIN" \
  "$ROOT/scripts/analyze_jepa_phase2c_ema_gate.py" \
  --phase2b-root "$PHASE2B_ROOT" \
  --phase2c-report "$OUTPUT_ROOT/spatial_ema_var.json" \
  --output "$OUTPUT_ROOT/analysis.json"

#!/usr/bin/env bash
# Launch the staged scaled ablation for SIGReg strength and source sampling.
#
# Default cells:
#   uniform:0.09        scaled control
#   uniform:0.03/0.01   lambda-only dose response
#   exploratory:0.09    sampling-only arm
#   exploratory:0.03/0.01 combined arms
#
# Pass --dry-run to print commands without training.
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

DATA_ROOT=""
RENDER_ROOT="$ROOT/.generated/datagen_full/render_textured_v03"
CHECKPOINT_ROOT="$ROOT/models/checkpoints_textured_v03_scaled_ablation_20260604"
EPOCHS=3
BATCH_SIZE=128
MAX_SESSIONS=300
SOURCE_ALLOW="ou_noise,primitive_curriculum"
SOURCE_CAP="auto"
CELLS="uniform:0.09 uniform:0.03 uniform:0.01 exploratory:0.09 exploratory:0.03 exploratory:0.01"
SEEDS="0"
EVAL_EVERY_EPOCHS=1
EVAL_HOLDOUT_FRACTION=0.02
EVAL_MAX_BATCHES=32
EVAL_NUM_WORKERS=2
NUM_WORKERS=6
SAVE_EVERY_BATCHES=0
RUN_PROBES=1
DRY_RUN=0
ALLOW_MATERIAL_COLOR_RENDER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --render-root)
      RENDER_ROOT="$2"
      shift 2
      ;;
    --checkpoint-root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-sessions)
      MAX_SESSIONS="$2"
      shift 2
      ;;
    --source-allow)
      SOURCE_ALLOW="$2"
      shift 2
      ;;
    --source-cap)
      SOURCE_CAP="$2"
      shift 2
      ;;
    --cells)
      CELLS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --eval-every-epochs)
      EVAL_EVERY_EPOCHS="$2"
      shift 2
      ;;
    --eval-holdout-fraction)
      EVAL_HOLDOUT_FRACTION="$2"
      shift 2
      ;;
    --eval-max-batches)
      EVAL_MAX_BATCHES="$2"
      shift 2
      ;;
    --eval-num-workers)
      EVAL_NUM_WORKERS="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --save-every-batches)
      SAVE_EVERY_BATCHES="$2"
      shift 2
      ;;
    --skip-probes)
      RUN_PROBES=0
      shift
      ;;
    --allow-material-color-render)
      ALLOW_MATERIAL_COLOR_RENDER=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  echo "Error: --data-root is required." >&2
  exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CHECKPOINT_ROOT"

if [[ "$SOURCE_CAP" == "auto" ]]; then
  MIX_JSON="$CHECKPOINT_ROOT/source_mix_maxsessions${MAX_SESSIONS}.json"
  MIX_CMD=("$PYTHON_BIN" "$ROOT/scripts/inspect_lewm_source_mix.py" \
    --data-root "$DATA_ROOT" \
    --render-root "$RENDER_ROOT" \
    --max-sessions "$MAX_SESSIONS" \
    --eval-holdout-fraction "$EVAL_HOLDOUT_FRACTION" \
    --source-allow "$SOURCE_ALLOW" \
    --output "$MIX_JSON")
  if [[ "$ALLOW_MATERIAL_COLOR_RENDER" == "1" ]]; then
    MIX_CMD+=(--allow-material-color-render)
  fi
  echo "=== Inspecting source mix for size-matched cap ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "${MIX_CMD[@]}"
    printf '\n'
    SOURCE_CAP=0
  else
    "${MIX_CMD[@]}"
    SOURCE_CAP="$(jq -r '.allowed_windows' "$MIX_JSON")"
    if [[ "$SOURCE_CAP" == "0" ]]; then
      echo "Error: exploratory source cap auto-resolved to zero." >&2
      exit 3
    fi
    echo "Using --source-cap $SOURCE_CAP for size-matched uniform controls."
  fi
fi

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
CELLS_NORMALIZED="${CELLS//,/ }"
read -r -a CELL_LIST <<< "$CELLS_NORMALIZED"

for SEED in "${SEED_LIST[@]}"; do
  for CELL in "${CELL_LIST[@]}"; do
    ARM="${CELL%%:*}"
    LAMBDA="${CELL##*:}"
    LAMBDA_TAG="${LAMBDA/./p}"
    OUT_DIR="$CHECKPOINT_ROOT/seed${SEED}/${ARM}_sig${LAMBDA_TAG}_maxsess${MAX_SESSIONS}"
    TRAIN_CMD=("$PYTHON_BIN" "$ROOT/scripts/train_lewm.py" \
      --data-root "$DATA_ROOT" \
      --render-root "$RENDER_ROOT" \
      --out-dir "$OUT_DIR" \
      --resume "$OUT_DIR" \
      --max-seq-len 4 \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --max-sessions "$MAX_SESSIONS" \
      --shuffle-seed "$SEED" \
      --sigreg-lambda "$LAMBDA" \
      --eval-every-epochs "$EVAL_EVERY_EPOCHS" \
      --eval-holdout-fraction "$EVAL_HOLDOUT_FRACTION" \
      --eval-max-batches "$EVAL_MAX_BATCHES" \
      --eval-num-workers "$EVAL_NUM_WORKERS" \
      --num-workers "$NUM_WORKERS" \
      --save-every-batches "$SAVE_EVERY_BATCHES")

    if [[ "$SOURCE_CAP" != "0" ]]; then
      TRAIN_CMD+=(--source-cap "$SOURCE_CAP")
    fi
    if [[ "$ARM" == "exploratory" ]]; then
      TRAIN_CMD+=(--source-allow "$SOURCE_ALLOW")
    elif [[ "$ARM" != "uniform" ]]; then
      echo "Unknown cell arm '$ARM' in '$CELL'; expected uniform or exploratory." >&2
      exit 2
    fi
    if [[ "$ALLOW_MATERIAL_COLOR_RENDER" == "1" ]]; then
      TRAIN_CMD+=(--allow-material-color-render)
    fi

    echo "=== Training seed=$SEED arm=$ARM sigreg_lambda=$LAMBDA ==="
    printf 'CMD:'
    printf ' %q' "${TRAIN_CMD[@]}"
    printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then
      "${TRAIN_CMD[@]}"
      if [[ "$RUN_PROBES" == "1" ]]; then
        LATEST_CKPT="$(find "$OUT_DIR" -regextype posix-extended -maxdepth 1 -type f -regex ".*/lewm_seq4_e[0-9]+\\.pt" -print | sort -V | tail -n 1)"
        if [[ -z "$LATEST_CKPT" ]]; then
          echo "Error: no completed checkpoint found under $OUT_DIR." >&2
          exit 4
        fi
        bash "$ROOT/scripts/eval_lewm_ablation_probes.sh" \
          --checkpoint "$LATEST_CKPT" \
          --output-dir "$OUT_DIR/probes" \
          --device cpu
      fi
    fi
  done
done

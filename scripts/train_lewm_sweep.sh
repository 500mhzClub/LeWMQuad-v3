#!/usr/bin/env bash
# Run the LeWM sequence-length ablation sweep {4, 8, 16}.
# Usage: bash scripts/train_lewm_sweep.sh --data-root .generated/datagen_full \
#   --render-root .generated/datagen_full/render_textured \
#   --checkpoint-root models/checkpoints_textured --eval-max-batches 32
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GENESIS_ROCM_PYTHON:-$ROOT/.generated/venvs/genesis_rocm/bin/python}"

DATA_ROOT=""
CHECKPOINT_ROOT="$ROOT/models/checkpoints"
EPOCHS=10
BATCH_SIZE=128
MAX_SESSIONS=""
RENDER_ROOT=""
ALLOW_MATERIAL_COLOR_RENDER=0
ALLOW_LEGACY_RESUME=0
EVAL_EVERY_EPOCHS=""
EVAL_HOLDOUT_FRACTION=""
EVAL_MAX_BATCHES=""
EVAL_NUM_WORKERS=""
SAVE_EVERY_BATCHES=""
SHUFFLE_SEED=""
SKIP_ROLLOUT_GATES=0
ROLLOUT_GATE_HORIZONS="1,2,3,5,8,10,16,20"
ROLLOUT_GATE_MAX_BATCHES=8
ROLLOUT_GATE_BATCH_SIZE=32
ROLLOUT_GATE_NUM_WORKERS=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root)
      DATA_ROOT="$2"
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
    --render-root)
      RENDER_ROOT="$2"
      shift 2
      ;;
    --allow-material-color-render)
      ALLOW_MATERIAL_COLOR_RENDER=1
      shift
      ;;
    --allow-legacy-resume)
      ALLOW_LEGACY_RESUME=1
      shift
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
    --save-every-batches)
      SAVE_EVERY_BATCHES="$2"
      shift 2
      ;;
    --shuffle-seed)
      SHUFFLE_SEED="$2"
      shift 2
      ;;
    --skip-rollout-gates)
      SKIP_ROLLOUT_GATES=1
      shift
      ;;
    --rollout-gate-horizons)
      ROLLOUT_GATE_HORIZONS="$2"
      shift 2
      ;;
    --rollout-gate-max-batches)
      ROLLOUT_GATE_MAX_BATCHES="$2"
      shift 2
      ;;
    --rollout-gate-batch-size)
      ROLLOUT_GATE_BATCH_SIZE="$2"
      shift 2
      ;;
    --rollout-gate-num-workers)
      ROLLOUT_GATE_NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  echo "Error: --data-root is required."
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

for SL in 4 8 16; do
  echo "=== Starting Sweep: seq_len=$SL ==="
  OUT_DIR="$CHECKPOINT_ROOT/sweep_seq$SL"
  if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT/$OUT_DIR"
  fi
  CMD=("$PYTHON_BIN" "$ROOT/scripts/train_lewm.py" \
    --data-root "$DATA_ROOT" \
    --max-seq-len "$SL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --out-dir "$OUT_DIR" \
    --resume "$OUT_DIR")
  
  if [[ -n "$MAX_SESSIONS" ]]; then
    CMD+=(--max-sessions "$MAX_SESSIONS")
  fi
  if [[ -n "$RENDER_ROOT" ]]; then
    CMD+=(--render-root "$RENDER_ROOT")
  fi
  if [[ "$ALLOW_MATERIAL_COLOR_RENDER" == "1" ]]; then
    CMD+=(--allow-material-color-render)
  fi
  if [[ "$ALLOW_LEGACY_RESUME" == "1" ]]; then
    CMD+=(--allow-legacy-resume)
  fi
  if [[ -n "$EVAL_EVERY_EPOCHS" ]]; then
    CMD+=(--eval-every-epochs "$EVAL_EVERY_EPOCHS")
  fi
  if [[ -n "$EVAL_HOLDOUT_FRACTION" ]]; then
    CMD+=(--eval-holdout-fraction "$EVAL_HOLDOUT_FRACTION")
  fi
  if [[ -n "$EVAL_MAX_BATCHES" ]]; then
    CMD+=(--eval-max-batches "$EVAL_MAX_BATCHES")
  fi
  if [[ -n "$EVAL_NUM_WORKERS" ]]; then
    CMD+=(--eval-num-workers "$EVAL_NUM_WORKERS")
  fi
  if [[ -n "$SAVE_EVERY_BATCHES" ]]; then
    CMD+=(--save-every-batches "$SAVE_EVERY_BATCHES")
  fi
  if [[ -n "$SHUFFLE_SEED" ]]; then
    CMD+=(--shuffle-seed "$SHUFFLE_SEED")
  fi

  "${CMD[@]}"

  if [[ "$SL" != "16" && "$SKIP_ROLLOUT_GATES" != "1" ]]; then
    LATEST_CKPT="$(find "$OUT_DIR" -regextype posix-extended -maxdepth 1 -type f -regex ".*/lewm_seq${SL}_e[0-9]+\\.pt" -print | sort -V | tail -n 1)"
    if [[ -z "$LATEST_CKPT" ]]; then
      echo "Error: no checkpoint found for seq_len=$SL under $OUT_DIR."
      exit 2
    fi

    CKPT_BASENAME="$(basename "$LATEST_CKPT" .pt)"
    GATE_REPORT="$OUT_DIR/planning_gate_${CKPT_BASENAME}.json"
    GATE_MARKER="$OUT_DIR/planning_gate_${CKPT_BASENAME}.approved"
    PROBE_CMD=("$PYTHON_BIN" "$ROOT/scripts/probe_lewm_rollout_horizons.py" \
      --checkpoint "$LATEST_CKPT" \
      --data-root "$DATA_ROOT" \
      --output "$GATE_REPORT" \
      --horizons "$ROLLOUT_GATE_HORIZONS" \
      --max-batches "$ROLLOUT_GATE_MAX_BATCHES" \
      --batch-size "$ROLLOUT_GATE_BATCH_SIZE" \
      --num-workers "$ROLLOUT_GATE_NUM_WORKERS")

    if [[ -n "$RENDER_ROOT" ]]; then
      PROBE_CMD+=(--render-root "$RENDER_ROOT")
    fi
    if [[ "$ALLOW_MATERIAL_COLOR_RENDER" == "1" ]]; then
      PROBE_CMD+=(--allow-material-color-render)
    fi
    if [[ -n "$EVAL_HOLDOUT_FRACTION" ]]; then
      PROBE_CMD+=(--holdout-fraction "$EVAL_HOLDOUT_FRACTION")
    fi

    echo "=== Running planning-readiness gate for seq_len=$SL ==="
    "${PROBE_CMD[@]}"

    if [[ ! -f "$GATE_MARKER" ]]; then
      echo
      echo "Planning-readiness gate is closed after seq_len=$SL."
      echo "Review: $GATE_REPORT"
      echo "To approve this exact checkpoint after review:"
      echo "  touch \"$GATE_MARKER\""
      echo "Then rerun this sweep command; completed training resumes without repeating epochs."
      exit 3
    fi
    echo "Planning-readiness gate approved for $LATEST_CKPT"
  fi
done

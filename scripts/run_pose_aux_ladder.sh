#!/usr/bin/env bash
# Pose-aux geometry ladder driver (Path 2C). Trains the controlled cells from the
# FIXED e3 source, each into its own out-dir, sequentially on the GPU. The
# companion watch_finetune_evals.sh (FT_DIRS=...) benchmarks every epoch
# checkpoint each cell produces: rollout decomposition, MPC/action-sensitivity,
# nav, and the pose-geometry contracts.
#
# Cells (docs/lewm_pose_aux_experiment_design_2026-06-06.md):
#   F0          frozen encoder, head learns both contracts -> frozen-head ceiling
#   C0          encoder updates, no geometry loss          -> continuation/drift control
#   C0/posthoc  fresh frozen head on C0's encoder          -> standardized drift geometry
#   C1          encoder updates, encoded-pose loss         -> can metric structure be injected?
#   C2          encoder updates, encoded + predicted loss  -> the deployed planning contract
#
# Screening weights are the 0.1x base-encoder-grad point from the measured
# gradient scale (models/pose_aux_proxy_20260606/gradient_scale_actual.json).
# Objective fields (seq_len/stride/sigreg/rollout) are pinned to the e3 recipe;
# rollout warmup is 0 because the source is already past warmup. All cells share
# torch-seed + shuffle-seed so head init and batch order are controlled.
set -u
ROOT=/home/andrewknowles/Workspace/LeWMQuad-v3
PY="$ROOT/.generated/venvs/genesis_render_vulkan/bin/python"
cd "$ROOT" || exit 1

SRC=models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/lewm_seq11_e3.pt
BASE="${BASE:-models/checkpoints_pose_aux_ladder_20260606}"   # override per run to avoid clobbering
RR=.generated/datagen_full
RT=.generated/datagen_full/render_textured_v03

LAM_ENC=0.097          # encoded-pose lambda  (10% of base encoder grad)
LAM_PRED=0.071         # predicted-pose lambda (10% of base encoder grad)
SESS="${SESS:-300}"    # design: 300 first, then repeat the winner at 1000
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-0}"

SENTINEL="$BASE/.ladder_running"
mkdir -p "$BASE"; : > "$SENTINEL"
trap 'rm -f "$SENTINEL"' EXIT

# $1 = out-dir, $2 = init-from checkpoint, remaining args appended verbatim.
cell () {
  local out="$1" init="$2"; shift 2
  mkdir -p "$out"
  echo "=== $(date) START $out (init $(basename "$init")) ==="
  "$PY" scripts/train_lewm.py \
    --data-root "$RR" --render-root "$RT" --allow-material-color-render \
    --init-from "$init" --out-dir "$out" \
    --max-seq-len 11 --stride 5 --batch-size 64 \
    --max-sessions "$SESS" --epochs "$EPOCHS" \
    --shuffle-seed 0 --torch-seed "$SEED" \
    --sigreg-lambda 0.09 --rollout-lambda 0.25 --rollout-horizon 10 \
    --rollout-gamma 0.9 --rollout-warmup-epochs 0 \
    --eval-max-batches 32 --device cuda "$@"
  echo "=== $(date) END $out exit=$? ==="
}

# F0 — frozen-encoder ceiling (both contracts trained on the head only)
cell "$BASE/F0" "$SRC" --freeze-model --pose-label-source actual \
  --pose-aux-lambda "$LAM_ENC" --pose-aux-predicted-lambda "$LAM_PRED"

# C0 — pure continuation control (no geometry objective; no pose head)
cell "$BASE/C0" "$SRC" --pose-aux-lambda 0 --pose-aux-predicted-lambda 0

# C0/posthoc — standardized fresh frozen head on C0's drifted encoder
cell "$BASE/C0/posthoc" "$BASE/C0/lewm_seq11_e0.pt" --freeze-model \
  --pose-label-source actual \
  --pose-aux-lambda "$LAM_ENC" --pose-aux-predicted-lambda "$LAM_PRED"

# C1 — inject encoded metric structure into the encoder
cell "$BASE/C1" "$SRC" --pose-label-source actual \
  --pose-aux-lambda "$LAM_ENC" --pose-aux-predicted-lambda 0

# C2 — full deployed planning contract (encoded + predicted-to-goal)
cell "$BASE/C2" "$SRC" --pose-label-source actual \
  --pose-aux-lambda "$LAM_ENC" --pose-aux-predicted-lambda "$LAM_PRED"

echo "=== $(date) LADDER COMPLETE ==="

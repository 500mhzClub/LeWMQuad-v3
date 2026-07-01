#!/usr/bin/env bash
# Train the Phase 3A old->action05 router on broad validation seeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TINYQUAD_JEPA_PYTHON:-/home/andrewknowles/TinyQuadJEPA/bin/python}"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm-7.1.1}"

if [[ -d "$ROCM_PREFIX" ]]; then
  export ROCM_PATH="${ROCM_PATH:-$ROCM_PREFIX}"
  if [[ -d "$ROCM_PREFIX/lib/llvm/bin" ]]; then
    export PATH="$ROCM_PREFIX/lib/llvm/bin:$PATH"
  fi
  if [[ -d "$ROCM_PREFIX/bin" ]]; then
    export PATH="$ROCM_PREFIX/bin:$PATH"
  fi
fi

export HIP_VISIBLE_DEVICES="${LEWM_HIP_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-0}}"
unset HSA_OVERRIDE_GFX_VERSION

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "TinyQuadJEPA ROCm Python not found at: $PYTHON_BIN" >&2
  exit 2
fi

TRAIN_SEEDS=(${PHASE3A_ROUTER_TRAIN_SEEDS:-20260701 20260709 20260717 20260723 20260731 20260735 20260739 20260743})
VALIDATION_SEED="${PHASE3A_ROUTER_VALIDATION_SEED:-20260747}"
OUTPUT="${PHASE3A_ROUTER_OUTPUT:-models/checkpoints/phase3a_explore_claim/phase3a_v5_value_map_router_old_to_action05_traceoutcome_sameoutcome_neg_train8_val47_pw050_256.pt}"
STEPS="${PHASE3A_ROUTER_STEPS:-256}"
POS_WEIGHT="${PHASE3A_ROUTER_POSITIVE_WEIGHT:-0.5}"
THRESHOLD="${PHASE3A_ROUTER_THRESHOLD:-0.5}"
DEVICE="${PHASE3A_ROUTER_DEVICE:-cuda}"
SAME_OUTCOME_NEGATIVE="${PHASE3A_ROUTER_SAME_OUTCOME_NEGATIVE:-1}"
LABEL_SOURCE="${PHASE3A_ROUTER_LABEL_SOURCE:-trace_outcome}"
PLANNER_HEAD="${PHASE3A_ROUTER_PRIMARY_HEAD:-models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_planner_expandedmap_frontiertrace_selected_allstates_multiseed35_1536.pt}"
FALLBACK_HEAD="${PHASE3A_ROUTER_FALLBACK_HEAD:-models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_planner_lookahead_h9_b32_action05_lr1e4_repeat2_768.pt}"

cd "$ROOT"

args=(
  --train-data .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/validation_phase3a_positive_control.jsonl
  --validation-data ".generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed${VALIDATION_SEED}/validation_phase3a_positive_control.jsonl"
  --base-checkpoint models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt
  --latent-map-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_broad_multiseed_0109_1723_31_train_2048.pt
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_broad_map_broad_seed20260701_2048.pt
  --latent-value-field-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
  --latent-value-extractor-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_extractor_1024.pt
  --train-router-only
  --router-label-source "$LABEL_SOURCE"
  --router-trace-pre-memory-marker-only
)

if [[ "$LABEL_SOURCE" == "trace_action_preference" ]]; then
  args+=(
    --dagger-rollout-value-map-planner-head "$PLANNER_HEAD"
    --dagger-rollout-value-map-fallback-head "$FALLBACK_HEAD"
  )
fi

if [[ "$LABEL_SOURCE" == "trace_outcome" && "$SAME_OUTCOME_NEGATIVE" != "0" ]]; then
  args+=(--router-trace-same-outcome-negative)
fi

for seed in "${TRAIN_SEEDS[@]}"; do
  d=".generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed${seed}"
  args+=(
    --trace-memory-data "${d}/phase3a_v5_expandedmap_frontiertrace_allstates_multiseed35_t098_trace.json"
    --trace-memory-source-data "${d}/validation_phase3a_positive_control.jsonl"
  )
  if [[ "$LABEL_SOURCE" == "trace_outcome" ]]; then
    args+=(
      --router-trace-primary-report "${d}/phase3a_v5_expandedmap_frontiertrace_allstates_multiseed35_t098_report.json"
      --router-trace-fallback-report "${d}/phase3a_v5_lookahead_action05_lr1e4_repeat2_t098_max68_report.json"
    )
  fi
done

vd=".generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed${VALIDATION_SEED}"
args+=(
  --router-validation-trace-memory-data "${vd}/phase3a_v5_expandedmap_frontiertrace_allstates_multiseed35_t098_trace.json"
  --router-validation-trace-memory-source-data "${vd}/validation_phase3a_positive_control.jsonl"
  --max-validation-episodes 64
  --optimization-steps "$STEPS"
  --batch-size 128
  --lr 0.001
  --weight-decay 0.0001
  --router-positive-weight "$POS_WEIGHT"
  --router-threshold "$THRESHOLD"
  --output "$OUTPUT"
  --device "$DEVICE"
  --log-every 64
)

if [[ "$LABEL_SOURCE" == "trace_outcome" ]]; then
  args+=(
    --router-validation-trace-primary-report "${vd}/phase3a_v5_expandedmap_frontiertrace_allstates_multiseed35_t098_report.json"
    --router-validation-trace-fallback-report "${vd}/phase3a_v5_lookahead_action05_lr1e4_repeat2_t098_max68_report.json"
  )
fi

exec "$PYTHON_BIN" scripts/train_jepa_phase3a_value_map_planner.py "${args[@]}"

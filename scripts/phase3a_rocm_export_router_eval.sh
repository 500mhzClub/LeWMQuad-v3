#!/usr/bin/env bash
# Strict Phase 3A broad-seed evaluation for an old->fallback value-map router.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <broad-seed> <output-stem> <router-head> <router-threshold>" >&2
  exit 2
fi

SEED="$1"
STEM="$2"
ROUTER_HEAD="$3"
ROUTER_THRESHOLD="$4"
SCENE_SEED="$((SEED + 1000003))"
DIR=".generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed${SEED}"

exec scripts/phase3a_rocm_export_closed_loop.sh \
  --validation-data "${DIR}/validation_phase3a_positive_control.jsonl" \
  --checkpoint models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt \
  --output "${DIR}/${STEM}.mp4" \
  --report-output "${DIR}/${STEM}_report.json" \
  --seed "$SCENE_SEED" \
  --score-source latent_recurrent_learned_value_map_planner \
  --latent-map-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_broad_multiseed_0109_1723_31_train_2048.pt \
  --latent-memory-updater models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_broad_map_broad_seed20260701_2048.pt \
  --latent-value-field-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt \
  --latent-value-extractor-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_extractor_1024.pt \
  --latent-value-action-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_action_synth8192_smooth005_4096.pt \
  --latent-value-map-planner-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_planner_expandedmap_frontiertrace_selected_allstates_multiseed35_1536.pt \
  --latent-value-map-fallback-head models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_planner_lookahead_h9_b32_action05_lr1e4_repeat2_768.pt \
  --latent-value-map-fallback-after-step 999 \
  --latent-value-map-router-head "$ROUTER_HEAD" \
  --latent-value-map-router-threshold "$ROUTER_THRESHOLD" \
  --latent-soft-value-router-head models/checkpoints/phase3a_explore_claim/phase3a_v5_soft_value_router_outcome_balanced_seed20260701_train128_1536.pt \
  --latent-soft-value-router-threshold 0.4 \
  --latent-soft-value-router-mode latent_marker_seen \
  --latent-pre-marker-action-correction-head models/checkpoints/phase3a_explore_claim/phase3a_v5_action_correction_pre_marker_local_oracle_seed20260709_trace61_all_2048.pt \
  --latent-pre-marker-action-correction-threshold 0.9942 \
  --latent-pre-marker-action-correction-initial-threshold 0.99 \
  --latent-pre-marker-action-correction-initial-max-step 0 \
  --latent-pre-marker-action-correction-max-step 3 \
  --latent-map-marker-threshold 0.5 \
  --latent-memory-marker-threshold 0.9 \
  --latent-memory-blocked-threshold 0.5 \
  --latent-memory-free-threshold 0.5 \
  --latent-value-target-threshold 0.5 \
  --latent-value-target-top-k 16 \
  --latent-value-extractor-threshold 0.5 \
  --latent-value-sparse-target-top-k 1 \
  --latent-value-map-marker-action-return \
  --latent-value-map-turn-oscillation-breaker \
  --latent-value-map-state-loop-breaker \
  --exact-online-memory-size 31 \
  --max-episodes 64 \
  --max-steps 68 \
  --skip-video

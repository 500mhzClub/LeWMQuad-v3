#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/.generated/go2_memory_closed_loop/generalized_learned_local_suite_20260629}"
POLICY="${POLICY:-$OUT_DIR/generalized_learned_local_gru.pt}"
POLICY_REPORT="${POLICY_REPORT:-$OUT_DIR/generalized_learned_local_gru_report.json}"
RESULT_PREFIX="${RESULT_PREFIX:-heldout}"
TEACHER_MODE="${TEACHER_MODE:-kinematic}"
EVAL_MODE="${EVAL_MODE:-kinematic}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
HELDOUT_SPLIT="${HELDOUT_SPLIT:-test_id}"
DEVICE="${DEVICE:-cpu}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
MAX_TICKS="${MAX_TICKS:-560}"
TEACHER_MAX_TICKS="${TEACHER_MAX_TICKS:-700}"
EVAL_MAX_TICKS="${EVAL_MAX_TICKS:-$MAX_TICKS}"
EPOCHS="${EPOCHS:-220}"
MODEL_TYPE="${MODEL_TYPE:-gru}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
EMBED_DIM="${EMBED_DIM:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.20}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
TRAIN_DROPOUT="${TRAIN_DROPOUT:-0.05}"
TRAIN_LABEL_SMOOTHING="${TRAIN_LABEL_SMOOTHING:-0.03}"
FORCE="${FORCE:-0}"
FILTER_TEACHERS="${FILTER_TEACHERS:-1}"
MIN_TRAIN_TEACHERS="${MIN_TRAIN_TEACHERS:-2}"
STOP_AFTER_MIN_TRAIN_TEACHERS="${STOP_AFTER_MIN_TRAIN_TEACHERS:-0}"
MIN_TEACHER_CLAIMS="${MIN_TEACHER_CLAIMS:-4}"
REQUIRE_TEACHER_SUCCESS="${REQUIRE_TEACHER_SUCCESS:-1}"
REQUIRE_TEACHER_ALL_BEACONS="${REQUIRE_TEACHER_ALL_BEACONS:-1}"
MAX_TEACHER_CLAIM_DISTANCE_M="${MAX_TEACHER_CLAIM_DISTANCE_M:-}"
MAX_TEACHER_CONTACT_LIKE_STALLS="${MAX_TEACHER_CONTACT_LIKE_STALLS:-0}"
MAX_TEACHER_HARD_STALLS="${MAX_TEACHER_HARD_STALLS:-0}"
MAX_TEACHER_BODY_VIOLATIONS="${MAX_TEACHER_BODY_VIOLATIONS:-0}"
POLICY_STATES="${POLICY_STATES:-EXPLORE,SEEK,SERVO}"
TRAIN_INCLUDE_STATES="${TRAIN_INCLUDE_STATES:-$POLICY_STATES}"
EVAL_POLICY_STATES="${EVAL_POLICY_STATES:-$POLICY_STATES}"
WEAK_MEMORY_SEEK_AREA_LOGIT="${WEAK_MEMORY_SEEK_AREA_LOGIT:-1.2}"
WEAK_MEMORY_SEEK_STALL_STREAK="${WEAK_MEMORY_SEEK_STALL_STREAK:-2}"
WEAK_MEMORY_SEEK_EXPLORE_COOLDOWN_TICKS="${WEAK_MEMORY_SEEK_EXPLORE_COOLDOWN_TICKS:-36}"
BODY_CLEARANCE_TARGET_AREA_LOGIT="${BODY_CLEARANCE_TARGET_AREA_LOGIT:-1.2}"
SUCCESS_DIST_M="${SUCCESS_DIST_M:-1.2}"
CLAIM_NEAR_AREA_LOGIT="${CLAIM_NEAR_AREA_LOGIT:-2.6}"
CLAIM_NEAR_AREA_LOGIT_BY_COLOR="${CLAIM_NEAR_AREA_LOGIT_BY_COLOR:-yellow:2.45,green:2.45}"
CLAIM_NEAR_BEARING="${CLAIM_NEAR_BEARING:-0.32}"
CLAIM_NEAR_MIN_SEEN_TICKS="${CLAIM_NEAR_MIN_SEEN_TICKS:-8}"
POLICY_TRANSLATION_PRESSURE_AFTER="${POLICY_TRANSLATION_PRESSURE_AFTER:-8}"
POLICY_TRANSLATION_PRESSURE_MAX_BLOCKED_PROB="${POLICY_TRANSLATION_PRESSURE_MAX_BLOCKED_PROB:-1.01}"
POLICY_TRANSLATION_PRESSURE_MIN_PROGRESS_M="${POLICY_TRANSLATION_PRESSURE_MIN_PROGRESS_M:-0.02}"
POLICY_TRANSLATION_PRESSURE_PRIMITIVES="${POLICY_TRANSLATION_PRESSURE_PRIMITIVES:-forward_medium,arc_left,arc_right,forward_fast,backward}"
LEARNED_LOCAL_POLICY_OUTCOME_RERANK="${LEARNED_LOCAL_POLICY_OUTCOME_RERANK:-1}"
LEARNED_LOCAL_POLICY_RERANK_TOP_K="${LEARNED_LOCAL_POLICY_RERANK_TOP_K:-5}"
LEARNED_LOCAL_POLICY_RERANK_POLICY_WEIGHT="${LEARNED_LOCAL_POLICY_RERANK_POLICY_WEIGHT:-0.2}"
LEARNED_LOCAL_POLICY_RERANK_BLOCKED_WEIGHT="${LEARNED_LOCAL_POLICY_RERANK_BLOCKED_WEIGHT:-3.0}"
LEARNED_LOCAL_POLICY_RERANK_CLEARANCE_WEIGHT="${LEARNED_LOCAL_POLICY_RERANK_CLEARANCE_WEIGHT:-0.5}"
LEARNED_LOCAL_POLICY_RERANK_PROGRESS_WEIGHT="${LEARNED_LOCAL_POLICY_RERANK_PROGRESS_WEIGHT:-1.0}"
LEARNED_LOCAL_POLICY_RERANK_HARD_BLOCKED_PENALTY="${LEARNED_LOCAL_POLICY_RERANK_HARD_BLOCKED_PENALTY:-2.0}"
LEARNED_LOCAL_POLICY_RERANK_BEARING_TURN_THRESHOLD="${LEARNED_LOCAL_POLICY_RERANK_BEARING_TURN_THRESHOLD:-0.4}"
LEARNED_LOCAL_POLICY_RERANK_BEARING_TURN_BONUS="${LEARNED_LOCAL_POLICY_RERANK_BEARING_TURN_BONUS:-0.4}"
TARGET_COLORS="${TARGET_COLORS:-red,yellow,blue,green}"
TEACHER_TARGET_SWITCH_POLICY="${TEACHER_TARGET_SWITCH_POLICY:-visible_priority}"
EVAL_TARGET_SWITCH_POLICY="${EVAL_TARGET_SWITCH_POLICY:-visible_priority}"
TARGET_SWITCH_CONF="${TARGET_SWITCH_CONF:-0.8}"
APPEND_STATE_FEATURES="${APPEND_STATE_FEATURES:-1}"
APPEND_ONLINE_MAP_FEATURES="${APPEND_ONLINE_MAP_FEATURES:-1}"
LEARNED_LOCAL_ONLINE_MAP_SIZE="${LEARNED_LOCAL_ONLINE_MAP_SIZE:-21}"
LEARNED_LOCAL_ONLINE_MAP_CELL_M="${LEARNED_LOCAL_ONLINE_MAP_CELL_M:-0.45}"
EXPLORE_CLEAR_VISITED_ON_CLAIM="${EXPLORE_CLEAR_VISITED_ON_CLAIM:-1}"
LEARNED_LOCAL_ONLINE_MAP_HARD_GUARD_BLOCKS="${LEARNED_LOCAL_ONLINE_MAP_HARD_GUARD_BLOCKS:-0}"
LEARNED_LOCAL_ONLINE_MAP_ROUTE_REPLAY_GUARD_OVERRIDE="${LEARNED_LOCAL_ONLINE_MAP_ROUTE_REPLAY_GUARD_OVERRIDE:-0}"
LEARNED_LOCAL_ONLINE_MAP_LOW_PROGRESS_BLOCK_M="${LEARNED_LOCAL_ONLINE_MAP_LOW_PROGRESS_BLOCK_M:-0.0}"
POLICY_FRONTIER_PRESSURE_AFTER="${POLICY_FRONTIER_PRESSURE_AFTER:-0}"
POLICY_FRONTIER_PRESSURE_MAX_BLOCKED_PROB="${POLICY_FRONTIER_PRESSURE_MAX_BLOCKED_PROB:-1.01}"
POLICY_FRONTIER_PRESSURE_MIN_PROGRESS_M="${POLICY_FRONTIER_PRESSURE_MIN_PROGRESS_M:-0.02}"
POLICY_FRONTIER_PRESSURE_MIN_ROUTE_CELLS="${POLICY_FRONTIER_PRESSURE_MIN_ROUTE_CELLS:-0}"
POLICY_FRONTIER_PRESSURE_GUARD_BLOCKED_PENALTY="${POLICY_FRONTIER_PRESSURE_GUARD_BLOCKED_PENALTY:-1.0}"
POLICY_FRONTIER_PRESSURE_NONROUTE_BACKWARD_CLAIM_ESCAPE="${POLICY_FRONTIER_PRESSURE_NONROUTE_BACKWARD_CLAIM_ESCAPE:-0}"
POLICY_FRONTIER_PRESSURE_PREFER_UNGUARDED="${POLICY_FRONTIER_PRESSURE_PREFER_UNGUARDED:-0}"
POLICY_ONLINE_MAP_NOVELTY_WEIGHT="${POLICY_ONLINE_MAP_NOVELTY_WEIGHT:-0.0}"
POLICY_ONLINE_MAP_CLAIM_REPULSION_WEIGHT="${POLICY_ONLINE_MAP_CLAIM_REPULSION_WEIGHT:-0.0}"
POLICY_ONLINE_MAP_FRONTIER_ROUTE_WEIGHT="${POLICY_ONLINE_MAP_FRONTIER_ROUTE_WEIGHT:-0.0}"
TRAIN_SCENES="${TRAIN_SCENES:-medium_enclosed_maze_01732aabc542,medium_enclosed_maze_000c67a65968,medium_enclosed_maze_04f670cb21f8,medium_enclosed_maze_df939d2d7b68}"
HELDOUT_SCENES="${HELDOUT_SCENES:-medium_enclosed_maze_1efe5a925da9,medium_enclosed_maze_95c97a9b5b3a}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-1.0}"

mkdir -p "$OUT_DIR"

IFS=',' read -r -a TRAIN_SCENE_ARRAY <<< "$TRAIN_SCENES"
IFS=',' read -r -a HELDOUT_SCENE_ARRAY <<< "$HELDOUT_SCENES"

FEATURE_COLLECTION_ARGS=(--learned-local-clock-features)
TRAIN_FEATURE_ARGS=(--append-clock-features)
if [[ "$APPEND_STATE_FEATURES" == "1" ]]; then
  FEATURE_COLLECTION_ARGS+=(--learned-local-state-features)
  TRAIN_FEATURE_ARGS+=(--append-state-features)
fi
if [[ "$APPEND_ONLINE_MAP_FEATURES" == "1" ]]; then
  FEATURE_COLLECTION_ARGS+=(
    --learned-local-online-map-features
    --learned-local-online-map-size "$LEARNED_LOCAL_ONLINE_MAP_SIZE"
    --learned-local-online-map-cell-m "$LEARNED_LOCAL_ONLINE_MAP_CELL_M"
  )
  TRAIN_FEATURE_ARGS+=(
    --append-online-map-features
    --online-map-size "$LEARNED_LOCAL_ONLINE_MAP_SIZE"
    --online-map-cell-m "$LEARNED_LOCAL_ONLINE_MAP_CELL_M"
  )
fi

ONLINE_MAP_RUNTIME_ARGS=()
if [[ "$EXPLORE_CLEAR_VISITED_ON_CLAIM" == "1" || "$EXPLORE_CLEAR_VISITED_ON_CLAIM" == "true" ]]; then
  ONLINE_MAP_RUNTIME_ARGS+=(--explore-clear-visited-on-claim)
fi
if [[ "$LEARNED_LOCAL_ONLINE_MAP_ROUTE_REPLAY_GUARD_OVERRIDE" == "1" || "$LEARNED_LOCAL_ONLINE_MAP_ROUTE_REPLAY_GUARD_OVERRIDE" == "true" ]]; then
  ONLINE_MAP_RUNTIME_ARGS+=(--learned-local-online-map-route-replay-guard-override)
fi
if [[ "$LEARNED_LOCAL_ONLINE_MAP_HARD_GUARD_BLOCKS" == "1" || "$LEARNED_LOCAL_ONLINE_MAP_HARD_GUARD_BLOCKS" == "true" ]]; then
  ONLINE_MAP_RUNTIME_ARGS+=(--learned-local-online-map-hard-guard-blocks)
fi
if [[ "$LEARNED_LOCAL_ONLINE_MAP_LOW_PROGRESS_BLOCK_M" != "0" && "$LEARNED_LOCAL_ONLINE_MAP_LOW_PROGRESS_BLOCK_M" != "0.0" ]]; then
  ONLINE_MAP_RUNTIME_ARGS+=(--learned-local-online-map-low-progress-block-m "$LEARNED_LOCAL_ONLINE_MAP_LOW_PROGRESS_BLOCK_M")
fi

FRONTIER_PRESSURE_ARGS=()
if [[ "$POLICY_FRONTIER_PRESSURE_AFTER" != "0" ]]; then
  FRONTIER_PRESSURE_ARGS=(
    --learned-local-policy-frontier-pressure-after "$POLICY_FRONTIER_PRESSURE_AFTER"
    --learned-local-policy-frontier-pressure-max-blocked-prob "$POLICY_FRONTIER_PRESSURE_MAX_BLOCKED_PROB"
    --learned-local-policy-frontier-pressure-min-progress-m "$POLICY_FRONTIER_PRESSURE_MIN_PROGRESS_M"
    --learned-local-policy-frontier-pressure-min-route-cells "$POLICY_FRONTIER_PRESSURE_MIN_ROUTE_CELLS"
    --learned-local-policy-frontier-pressure-guard-blocked-penalty "$POLICY_FRONTIER_PRESSURE_GUARD_BLOCKED_PENALTY"
  )
  if [[ "$POLICY_FRONTIER_PRESSURE_NONROUTE_BACKWARD_CLAIM_ESCAPE" == "1" || "$POLICY_FRONTIER_PRESSURE_NONROUTE_BACKWARD_CLAIM_ESCAPE" == "true" ]]; then
    FRONTIER_PRESSURE_ARGS+=(--learned-local-policy-frontier-pressure-nonroute-backward-claim-escape)
  fi
  if [[ "$POLICY_FRONTIER_PRESSURE_PREFER_UNGUARDED" == "1" || "$POLICY_FRONTIER_PRESSURE_PREFER_UNGUARDED" == "true" ]]; then
    FRONTIER_PRESSURE_ARGS+=(--learned-local-policy-frontier-pressure-prefer-unguarded)
  fi
fi

COMMON_ARGS=(
  --policy memory
  --demo-mode explore
  --target-color all
  --target-colors "$TARGET_COLORS"
  --controller "$ROOT/.generated/go2_hidden_target_memory/go2_rgb_jepa_strict_exact_valuenorm_gate_neg6_pair8_nonforward_eval_seed20260825_h512.pt"
  --frozen-jepa-checkpoint "$ROOT/.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt"
  --primitive-outcome-checkpoint "$ROOT/.generated/go2_wallaware_learned/primitive_outcome_jepa_mixed_progress08.pt"
  --wall-aware-planner
  --wall-decision-source learned_action
  --success-dist-m "$SUCCESS_DIST_M"
  --mask-area-threshold 0.01
  --claim-min-seen-ticks 18
  --claim-near-area-logit "$CLAIM_NEAR_AREA_LOGIT"
  --claim-near-area-logit-by-color "$CLAIM_NEAR_AREA_LOGIT_BY_COLOR"
  --claim-near-bearing "$CLAIM_NEAR_BEARING"
  --claim-near-min-seen-ticks "$CLAIM_NEAR_MIN_SEEN_TICKS"
  --multi-target-switch-conf "$TARGET_SWITCH_CONF"
  --wall-body-probe-margin-m 0.0
  --wall-predicted-blocked-waypoint-replan
  --wall-predicted-blocked-waypoint-streak 2
  --primitive-outcome-forward-progress-floor 0.04
  --primitive-outcome-progress-floor-min-blocked-prob 0.05
  --primitive-outcome-progress-floor-force-below 0.01
  --primitive-outcome-forward-progress-penalty 0.9
  --primitive-outcome-progress-floor-prefer-yaw
  --primitive-outcome-preserve-turn-requests
  --primitive-outcome-preserve-turn-states EXPLORE
  --body-clearance-target-servo
  --body-clearance-target-area-logit "$BODY_CLEARANCE_TARGET_AREA_LOGIT"
  --body-clearance-target-bearing 0.16
  --body-clearance-target-forward-primitive forward_medium
  --body-clearance-latch-ticks 8
  --body-clearance-learned-min-area-logit "$BODY_CLEARANCE_TARGET_AREA_LOGIT"
)

TEACHER_ARGS=(
  --multi-target-switch-policy "$TEACHER_TARGET_SWITCH_POLICY"
  --weak-memory-seek-area-logit "$WEAK_MEMORY_SEEK_AREA_LOGIT"
  --weak-memory-seek-stall-streak "$WEAK_MEMORY_SEEK_STALL_STREAK"
  --weak-memory-seek-explore-cooldown-ticks "$WEAK_MEMORY_SEEK_EXPLORE_COOLDOWN_TICKS"
  --explore-reset-on-claim
  "${ONLINE_MAP_RUNTIME_ARGS[@]}"
  --wall-stall-block-waypoint
  --primitive-outcome-blocked-hard-veto
  --primitive-outcome-blocked-hard-veto-selected-primitives arc_left,arc_right
  --primitive-outcome-blocked-hard-veto-max-abs-bearing 0.18
  --body-clearance-target-forward-primitive forward_slow
  --explore-standoff-route
  --explore-standoff-route-until-area-logit 1.2
  --explore-standoff-lookahead-m 0.55
  --explore-standoff-path-spacing-m 0.2
  --explore-standoff-clearance-weight 12.0
  --explore-standoff-clearance-target-m 0.3
  --explore-standoff-candidates 16
  --explore-standoff-arrival-m 0.45
  --explore-standoff-allow-arcs
  --explore-standoff-arc-max-bearing 0.35
  --learned-local-policy-states "$POLICY_STATES"
  "${FEATURE_COLLECTION_ARGS[@]}"
)

DATASETS=()
SELECTED_TRAIN_SCENE_ARRAY=()
REJECTED_TRAIN_SCENE_ARRAY=()
for SCENE_ID in "${TRAIN_SCENE_ARRAY[@]}"; do
  DATASET="$OUT_DIR/teacher_${SCENE_ID}_clock_dataset.npz"
  RESULT="$OUT_DIR/teacher_${SCENE_ID}_result.json"
  QUALITY="$OUT_DIR/teacher_${SCENE_ID}_quality.json"
  if [[ "$FORCE" == "1" || ! -s "$DATASET" || ! -s "$RESULT" ]]; then
    "$PYTHON" "$ROOT/scripts/benchmark_go2_memory_closed_loop.py" \
      --mode "$TEACHER_MODE" \
      --policy-device "$POLICY_DEVICE" \
      --device "$DEVICE" \
      --split "$TRAIN_SPLIT" \
      --scene-id "$SCENE_ID" \
      --max-ticks "$TEACHER_MAX_TICKS" \
      "${COMMON_ARGS[@]}" \
      "${TEACHER_ARGS[@]}" \
      --learned-local-dataset-output "$DATASET" \
      --output "$RESULT"
  fi
  TEACHER_CHECK_ARGS=(
    --result "$RESULT"
    --dataset "$DATASET"
    --output "$QUALITY"
    --min-claims "$MIN_TEACHER_CLAIMS"
    --max-contact-like-stalls "$MAX_TEACHER_CONTACT_LIKE_STALLS"
    --max-hard-stalls "$MAX_TEACHER_HARD_STALLS"
    --max-body-violations "$MAX_TEACHER_BODY_VIOLATIONS"
    --forbid-pose-topology-features
  )
  if [[ "$REQUIRE_TEACHER_SUCCESS" == "1" ]]; then
    TEACHER_CHECK_ARGS+=(--require-success)
  fi
  if [[ "$REQUIRE_TEACHER_ALL_BEACONS" == "1" ]]; then
    TEACHER_CHECK_ARGS+=(--require-all-beacons)
  fi
  if [[ -n "$MAX_TEACHER_CLAIM_DISTANCE_M" ]]; then
    TEACHER_CHECK_ARGS+=(--max-claim-distance-m "$MAX_TEACHER_CLAIM_DISTANCE_M")
  fi
  if "$PYTHON" "$ROOT/scripts/check_go2_teacher_dataset.py" "${TEACHER_CHECK_ARGS[@]}"; then
    DATASETS+=("$DATASET")
    SELECTED_TRAIN_SCENE_ARRAY+=("$SCENE_ID")
  else
    REJECTED_TRAIN_SCENE_ARRAY+=("$SCENE_ID")
    if [[ "$FILTER_TEACHERS" != "1" ]]; then
      DATASETS+=("$DATASET")
      SELECTED_TRAIN_SCENE_ARRAY+=("$SCENE_ID")
    fi
  fi
  if [[ "$STOP_AFTER_MIN_TRAIN_TEACHERS" == "1" && ${#DATASETS[@]} -ge $MIN_TRAIN_TEACHERS ]]; then
    echo "accepted teacher datasets: ${#DATASETS[@]} >= MIN_TRAIN_TEACHERS=$MIN_TRAIN_TEACHERS; stopping collection early"
    break
  fi
done
SELECTED_TRAIN_SCENES="$(IFS=','; echo "${SELECTED_TRAIN_SCENE_ARRAY[*]}")"
if (( ${#DATASETS[@]} < MIN_TRAIN_TEACHERS )); then
  echo "accepted teacher datasets: ${#DATASETS[@]} < MIN_TRAIN_TEACHERS=$MIN_TRAIN_TEACHERS" >&2
  echo "selected scenes: $SELECTED_TRAIN_SCENES" >&2
  echo "rejected scenes: ${REJECTED_TRAIN_SCENE_ARRAY[*]}" >&2
  exit 1
fi

if [[ "$FORCE" == "1" || ! -s "$POLICY" || ! -s "$POLICY_REPORT" ]]; then
  "$PYTHON" "$ROOT/scripts/train_go2_closed_loop_learned_local_policy.py" \
    "${DATASETS[@]}" \
    --output "$POLICY" \
    --report-output "$POLICY_REPORT" \
    --model-type "$MODEL_TYPE" \
    --include-states "$TRAIN_INCLUDE_STATES" \
    "${TRAIN_FEATURE_ARGS[@]}" \
    --hidden-dim "$HIDDEN_DIM" \
    --embed-dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --class-weight-power "$CLASS_WEIGHT_POWER" \
    --dropout "${TRAIN_DROPOUT:-0.05}" \
    --label-smoothing "${TRAIN_LABEL_SMOOTHING:-0.03}" \
    --device "$TRAIN_DEVICE"
fi

RESULTS=()
PER_RESULT_CHECK_FAILED=0
LEARNED_LOCAL_RERANK_ARGS=()
if [[ "$LEARNED_LOCAL_POLICY_OUTCOME_RERANK" == "1" ]]; then
  LEARNED_LOCAL_RERANK_ARGS=(
    --learned-local-policy-outcome-rerank
    --learned-local-policy-rerank-top-k "$LEARNED_LOCAL_POLICY_RERANK_TOP_K"
    --learned-local-policy-rerank-policy-weight "$LEARNED_LOCAL_POLICY_RERANK_POLICY_WEIGHT"
    --learned-local-policy-rerank-blocked-weight "$LEARNED_LOCAL_POLICY_RERANK_BLOCKED_WEIGHT"
    --learned-local-policy-rerank-clearance-weight "$LEARNED_LOCAL_POLICY_RERANK_CLEARANCE_WEIGHT"
    --learned-local-policy-rerank-progress-weight "$LEARNED_LOCAL_POLICY_RERANK_PROGRESS_WEIGHT"
    --learned-local-policy-rerank-hard-blocked-penalty "$LEARNED_LOCAL_POLICY_RERANK_HARD_BLOCKED_PENALTY"
    --learned-local-policy-rerank-bearing-turn-threshold "$LEARNED_LOCAL_POLICY_RERANK_BEARING_TURN_THRESHOLD"
    --learned-local-policy-rerank-bearing-turn-bonus "$LEARNED_LOCAL_POLICY_RERANK_BEARING_TURN_BONUS"
  )
fi
for SCENE_ID in "${HELDOUT_SCENE_ARRAY[@]}"; do
  RESULT="$OUT_DIR/${RESULT_PREFIX}_${SCENE_ID}_result.json"
  RESULTS+=("$RESULT")
  if [[ "$FORCE" == "1" || ! -s "$RESULT" ]]; then
    "$PYTHON" "$ROOT/scripts/benchmark_go2_memory_closed_loop.py" \
      --generalized-runtime-contract \
      --mode "$EVAL_MODE" \
      --policy-device "$POLICY_DEVICE" \
      --device "$DEVICE" \
      --split "$HELDOUT_SPLIT" \
      --scene-id "$SCENE_ID" \
      --max-ticks "$EVAL_MAX_TICKS" \
      "${COMMON_ARGS[@]}" \
      --weak-memory-seek-area-logit "$WEAK_MEMORY_SEEK_AREA_LOGIT" \
      --weak-memory-seek-stall-streak "$WEAK_MEMORY_SEEK_STALL_STREAK" \
      --weak-memory-seek-explore-cooldown-ticks "$WEAK_MEMORY_SEEK_EXPLORE_COOLDOWN_TICKS" \
      --multi-target-switch-policy "$EVAL_TARGET_SWITCH_POLICY" \
      --explore-goal-policy learned_policy \
      --learned-local-policy-checkpoint "$POLICY" \
      --learned-local-policy-states "$EVAL_POLICY_STATES" \
      "${FEATURE_COLLECTION_ARGS[@]}" \
      "${LEARNED_LOCAL_RERANK_ARGS[@]}" \
      --learned-local-policy-translation-pressure-after "$POLICY_TRANSLATION_PRESSURE_AFTER" \
      --learned-local-policy-translation-pressure-max-blocked-prob "$POLICY_TRANSLATION_PRESSURE_MAX_BLOCKED_PROB" \
      --learned-local-policy-translation-pressure-min-progress-m "$POLICY_TRANSLATION_PRESSURE_MIN_PROGRESS_M" \
      --learned-local-policy-translation-pressure-primitives "$POLICY_TRANSLATION_PRESSURE_PRIMITIVES" \
      "${FRONTIER_PRESSURE_ARGS[@]}" \
      --learned-local-policy-online-map-novelty-weight "$POLICY_ONLINE_MAP_NOVELTY_WEIGHT" \
      --learned-local-policy-online-map-claim-repulsion-weight "$POLICY_ONLINE_MAP_CLAIM_REPULSION_WEIGHT" \
      --learned-local-policy-online-map-frontier-route-weight "$POLICY_ONLINE_MAP_FRONTIER_ROUTE_WEIGHT" \
      --output "$RESULT"
  fi
  if ! "$PYTHON" "$ROOT/scripts/check_go2_fully_learned_demo.py" \
      --result "$RESULT" \
      --max-ticks "$EVAL_MAX_TICKS" \
      --max-body-contact-events 0 \
      --require-generalized-runtime-contract \
      --forbid-route-memory \
      --forbid-pose-topology-features \
      --train-scenes "$SELECTED_TRAIN_SCENES"; then
    PER_RESULT_CHECK_FAILED=1
  fi
done

SUITE_CHECK_FAILED=0
if ! "$PYTHON" "$ROOT/scripts/check_go2_generalized_suite.py" \
    --results "${RESULTS[@]}" \
    --policy-report "$POLICY_REPORT" \
    --train-scenes "$SELECTED_TRAIN_SCENES" \
    --heldout-scenes "$HELDOUT_SCENES" \
    --min-success-rate "$MIN_SUCCESS_RATE" \
    --max-ticks "$EVAL_MAX_TICKS" \
    --output "$OUT_DIR/suite_report.json"; then
  SUITE_CHECK_FAILED=1
fi

if [[ "$PER_RESULT_CHECK_FAILED" != "0" || "$SUITE_CHECK_FAILED" != "0" ]]; then
  exit 1
fi

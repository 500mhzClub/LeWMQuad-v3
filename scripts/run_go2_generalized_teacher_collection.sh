#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"
SCENE_CORPUS="${SCENE_CORPUS:-$ROOT/.generated/scene_corpus/minimum_20260520T080420Z}"
FAMILY="${FAMILY:-medium_enclosed_maze}"
OUT_DIR="${OUT_DIR:-$ROOT/.generated/go2_memory_closed_loop/generalized_learned_local_teacher_current_features}"
ROUND_NAME="${ROUND_NAME:-teacher_current_features}"
MODE="${MODE:-kinematic}"
SPLIT="${SPLIT:-train}"
DEVICE="${DEVICE:-cpu}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
MAX_TICKS="${MAX_TICKS:-700}"
FORCE="${FORCE:-0}"

TARGET_COLORS="${TARGET_COLORS:-red,yellow,blue,green}"
TARGET_SWITCH_POLICY="${TARGET_SWITCH_POLICY:-visible_priority}"
TARGET_SWITCH_CONF="${TARGET_SWITCH_CONF:-0.8}"
SUCCESS_DIST_M="${SUCCESS_DIST_M:-1.2}"
CLAIM_NEAR_AREA_LOGIT="${CLAIM_NEAR_AREA_LOGIT:-2.6}"
CLAIM_NEAR_AREA_LOGIT_BY_COLOR="${CLAIM_NEAR_AREA_LOGIT_BY_COLOR:-yellow:2.45,green:2.45}"
CLAIM_NEAR_BEARING="${CLAIM_NEAR_BEARING:-0.32}"
CLAIM_NEAR_MIN_SEEN_TICKS="${CLAIM_NEAR_MIN_SEEN_TICKS:-8}"
BODY_CLEARANCE_TARGET_AREA_LOGIT="${BODY_CLEARANCE_TARGET_AREA_LOGIT:-1.2}"

PRIMITIVE_OUTCOME_CHECKPOINT="${PRIMITIVE_OUTCOME_CHECKPOINT:-$ROOT/.generated/go2_wallaware_learned/primitive_outcome_jepa_broad_train24_block080_v95.pt}"
PRIMITIVE_CLEARANCE_CHECKPOINT="${PRIMITIVE_CLEARANCE_CHECKPOINT:-$ROOT/.generated/go2_wallaware_learned/primitive_body_clearance_jepa_v106_train32val32_obstacle_margin002_afterstart_h192.pt}"
PRIMITIVE_CLEARANCE_THRESHOLD="${PRIMITIVE_CLEARANCE_THRESHOLD:-0.78}"
BODY_CLEARANCE_HARD_VETO_PROB="${BODY_CLEARANCE_HARD_VETO_PROB:-0.78}"
BODY_CLEARANCE_HARD_VETO_SELECTED_PRIMITIVES="${BODY_CLEARANCE_HARD_VETO_SELECTED_PRIMITIVES:-forward_medium,arc_left,arc_right}"
BODY_CLEARANCE_TARGET_SERVO="${BODY_CLEARANCE_TARGET_SERVO:-1}"
BODY_CLEARANCE_TARGET_FORWARD_PRIMITIVE="${BODY_CLEARANCE_TARGET_FORWARD_PRIMITIVE:-forward_slow}"

LEARNED_LOCAL_ONLINE_MAP_SIZE="${LEARNED_LOCAL_ONLINE_MAP_SIZE:-21}"
LEARNED_LOCAL_ONLINE_MAP_CELL_M="${LEARNED_LOCAL_ONLINE_MAP_CELL_M:-0.45}"
LEARNED_LOCAL_POLICY_STATES="${LEARNED_LOCAL_POLICY_STATES:-EXPLORE}"
EXPLORE_CLEAR_VISITED_ON_CLAIM="${EXPLORE_CLEAR_VISITED_ON_CLAIM:-1}"

ORACLE_STANDOFF_LOOKAHEAD_M="${ORACLE_STANDOFF_LOOKAHEAD_M:-0.55}"
ORACLE_STANDOFF_PATH_SPACING_M="${ORACLE_STANDOFF_PATH_SPACING_M:-0.20}"
ORACLE_STANDOFF_CLEARANCE_WEIGHT="${ORACLE_STANDOFF_CLEARANCE_WEIGHT:-12.0}"
ORACLE_STANDOFF_CLEARANCE_TARGET_M="${ORACLE_STANDOFF_CLEARANCE_TARGET_M:-0.30}"
ORACLE_STANDOFF_CANDIDATES="${ORACLE_STANDOFF_CANDIDATES:-16}"
ORACLE_STANDOFF_ARRIVAL_M="${ORACLE_STANDOFF_ARRIVAL_M:-0.45}"
ORACLE_STANDOFF_ARC_MAX_BEARING="${ORACLE_STANDOFF_ARC_MAX_BEARING:-0.35}"
ORACLE_STANDOFF_ROUTE_UNTIL_AREA_LOGIT="${ORACLE_STANDOFF_ROUTE_UNTIL_AREA_LOGIT:-1.2}"

MIN_TEACHER_CLAIMS="${MIN_TEACHER_CLAIMS:-4}"
REQUIRE_TEACHER_SUCCESS="${REQUIRE_TEACHER_SUCCESS:-1}"
REQUIRE_TEACHER_ALL_BEACONS="${REQUIRE_TEACHER_ALL_BEACONS:-1}"
MAX_TEACHER_CLAIM_DISTANCE_M="${MAX_TEACHER_CLAIM_DISTANCE_M:-}"
MAX_TEACHER_CONTACT_LIKE_STALLS="${MAX_TEACHER_CONTACT_LIKE_STALLS:-0}"
MAX_TEACHER_HARD_STALLS="${MAX_TEACHER_HARD_STALLS:-0}"
MAX_TEACHER_BODY_VIOLATIONS="${MAX_TEACHER_BODY_VIOLATIONS:-0}"
MIN_TEACHER_EXAMPLES="${MIN_TEACHER_EXAMPLES:-1}"
STOP_AFTER_ACCEPTED="${STOP_AFTER_ACCEPTED:-0}"
MIN_ACCEPTED="${MIN_ACCEPTED:-1}"

TRAIN_SCENES="${TRAIN_SCENES:-medium_enclosed_maze_5ae240e5e391,medium_enclosed_maze_42ec4c74ee43,medium_enclosed_maze_0100925f9754,medium_enclosed_maze_03fa030348c7,medium_enclosed_maze_abb9ac953e00,medium_enclosed_maze_000c67a65968,medium_enclosed_maze_595b349fbbf7,medium_enclosed_maze_df939d2d7b68,medium_enclosed_maze_62c21394102b,medium_enclosed_maze_01732aabc542,medium_enclosed_maze_04f670cb21f8}"

mkdir -p "$OUT_DIR"
IFS=',' read -r -a TRAIN_SCENE_ARRAY <<< "$TRAIN_SCENES"

ACCEPTED_DATASETS_LIST="$OUT_DIR/${ROUND_NAME}_accepted_train_paths.txt"
ACCEPTED_SCENES_LIST="$OUT_DIR/${ROUND_NAME}_accepted_scenes.txt"
REJECTED_SCENES_LIST="$OUT_DIR/${ROUND_NAME}_rejected_scenes.txt"
: > "$ACCEPTED_DATASETS_LIST"
: > "$ACCEPTED_SCENES_LIST"
: > "$REJECTED_SCENES_LIST"

PRIMITIVE_CLEARANCE_ARGS=()
if [[ -n "$PRIMITIVE_CLEARANCE_CHECKPOINT" && "$PRIMITIVE_CLEARANCE_CHECKPOINT" != "none" ]]; then
  PRIMITIVE_CLEARANCE_ARGS=(
    --primitive-clearance-checkpoint "$PRIMITIVE_CLEARANCE_CHECKPOINT"
    --primitive-clearance-threshold "$PRIMITIVE_CLEARANCE_THRESHOLD"
    --body-clearance-hard-veto-prob "$BODY_CLEARANCE_HARD_VETO_PROB"
    --body-clearance-hard-veto-selected-primitives "$BODY_CLEARANCE_HARD_VETO_SELECTED_PRIMITIVES"
  )
fi

BODY_CLEARANCE_TARGET_ARGS=()
if [[ "$BODY_CLEARANCE_TARGET_SERVO" == "1" || "$BODY_CLEARANCE_TARGET_SERVO" == "true" ]]; then
  BODY_CLEARANCE_TARGET_ARGS=(
    --body-clearance-target-servo
    --body-clearance-target-area-logit "$BODY_CLEARANCE_TARGET_AREA_LOGIT"
    --body-clearance-target-bearing 0.16
    --body-clearance-target-forward-primitive "$BODY_CLEARANCE_TARGET_FORWARD_PRIMITIVE"
    --body-clearance-latch-ticks 8
    --body-clearance-learned-min-area-logit "$BODY_CLEARANCE_TARGET_AREA_LOGIT"
  )
fi

COMMON_ARGS=(
  --policy memory
  --demo-mode explore
  --target-color all
  --target-colors "$TARGET_COLORS"
  --controller "$ROOT/.generated/go2_hidden_target_memory/go2_rgb_jepa_strict_exact_valuenorm_gate_neg6_pair8_nonforward_eval_seed20260825_h512.pt"
  --frozen-jepa-checkpoint "$ROOT/.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt"
  --primitive-outcome-checkpoint "$PRIMITIVE_OUTCOME_CHECKPOINT"
  "${PRIMITIVE_CLEARANCE_ARGS[@]}"
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
  --multi-target-switch-policy "$TARGET_SWITCH_POLICY"
  --weak-memory-seek-area-logit 1.2
  --weak-memory-seek-stall-streak 2
  --weak-memory-seek-explore-cooldown-ticks 36
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
  "${BODY_CLEARANCE_TARGET_ARGS[@]}"
  --explore-reset-on-claim
  --wall-stall-block-waypoint
  --primitive-outcome-blocked-hard-veto
  --primitive-outcome-blocked-hard-veto-selected-primitives arc_left,arc_right
  --primitive-outcome-blocked-hard-veto-max-abs-bearing 0.18
  --explore-standoff-route
  --explore-standoff-route-until-area-logit "$ORACLE_STANDOFF_ROUTE_UNTIL_AREA_LOGIT"
  --explore-standoff-lookahead-m "$ORACLE_STANDOFF_LOOKAHEAD_M"
  --explore-standoff-path-spacing-m "$ORACLE_STANDOFF_PATH_SPACING_M"
  --explore-standoff-clearance-weight "$ORACLE_STANDOFF_CLEARANCE_WEIGHT"
  --explore-standoff-clearance-target-m "$ORACLE_STANDOFF_CLEARANCE_TARGET_M"
  --explore-standoff-candidates "$ORACLE_STANDOFF_CANDIDATES"
  --explore-standoff-arrival-m "$ORACLE_STANDOFF_ARRIVAL_M"
  --explore-standoff-allow-arcs
  --explore-standoff-arc-max-bearing "$ORACLE_STANDOFF_ARC_MAX_BEARING"
  --learned-local-policy-states "$LEARNED_LOCAL_POLICY_STATES"
  --learned-local-clock-features
  --learned-local-state-features
  --learned-local-online-map-features
  --learned-local-online-map-size "$LEARNED_LOCAL_ONLINE_MAP_SIZE"
  --learned-local-online-map-cell-m "$LEARNED_LOCAL_ONLINE_MAP_CELL_M"
)

if [[ "$EXPLORE_CLEAR_VISITED_ON_CLAIM" == "1" || "$EXPLORE_CLEAR_VISITED_ON_CLAIM" == "true" ]]; then
  COMMON_ARGS+=(--explore-clear-visited-on-claim)
fi

accepted_count=0
for SCENE_ID in "${TRAIN_SCENE_ARRAY[@]}"; do
  DATASET="$OUT_DIR/${ROUND_NAME}_${SCENE_ID}_clock_dataset.npz"
  RESULT="$OUT_DIR/${ROUND_NAME}_${SCENE_ID}_result.json"
  QUALITY="$OUT_DIR/${ROUND_NAME}_${SCENE_ID}_quality.json"

  if [[ "$FORCE" == "1" || ! -s "$DATASET" || ! -s "$RESULT" ]]; then
    "$PYTHON" "$ROOT/scripts/benchmark_go2_memory_closed_loop.py" \
      --scene-corpus "$SCENE_CORPUS" \
      --family "$FAMILY" \
      --mode "$MODE" \
      --policy-device "$POLICY_DEVICE" \
      --device "$DEVICE" \
      --split "$SPLIT" \
      --scene-id "$SCENE_ID" \
      --max-ticks "$MAX_TICKS" \
      "${COMMON_ARGS[@]}" \
      --learned-local-dataset-output "$DATASET" \
      --output "$RESULT"
  fi

  TEACHER_CHECK_ARGS=(
    --result "$RESULT"
    --scene-manifest "$SCENE_CORPUS/$SPLIT/$FAMILY/$SCENE_ID/manifest.json"
    --dataset "$DATASET"
    --output "$QUALITY"
    --min-claims "$MIN_TEACHER_CLAIMS"
    --max-contact-like-stalls "$MAX_TEACHER_CONTACT_LIKE_STALLS"
    --max-hard-stalls "$MAX_TEACHER_HARD_STALLS"
    --max-body-violations "$MAX_TEACHER_BODY_VIOLATIONS"
    --min-examples "$MIN_TEACHER_EXAMPLES"
    --forbid-pose-topology-features
  )
  if [[ "$REQUIRE_TEACHER_SUCCESS" == "1" || "$REQUIRE_TEACHER_SUCCESS" == "true" ]]; then
    TEACHER_CHECK_ARGS+=(--require-success)
  fi
  if [[ "$REQUIRE_TEACHER_ALL_BEACONS" == "1" || "$REQUIRE_TEACHER_ALL_BEACONS" == "true" ]]; then
    TEACHER_CHECK_ARGS+=(--require-all-beacons)
  fi
  if [[ -n "$MAX_TEACHER_CLAIM_DISTANCE_M" ]]; then
    TEACHER_CHECK_ARGS+=(--max-claim-distance-m "$MAX_TEACHER_CLAIM_DISTANCE_M")
  fi

  if "$PYTHON" "$ROOT/scripts/check_go2_teacher_dataset.py" "${TEACHER_CHECK_ARGS[@]}"; then
    printf "%s\n" "$DATASET" >> "$ACCEPTED_DATASETS_LIST"
    printf "%s\n" "$SCENE_ID" >> "$ACCEPTED_SCENES_LIST"
    accepted_count=$((accepted_count + 1))
  else
    printf "%s\n" "$SCENE_ID" >> "$REJECTED_SCENES_LIST"
  fi

  if [[ "$STOP_AFTER_ACCEPTED" == "1" && "$accepted_count" -ge "$MIN_ACCEPTED" ]]; then
    break
  fi
done

if [[ "$accepted_count" -lt "$MIN_ACCEPTED" ]]; then
  echo "accepted teacher datasets: $accepted_count < MIN_ACCEPTED=$MIN_ACCEPTED" >&2
  exit 1
fi

echo "accepted teacher datasets: $accepted_count"
echo "accepted dataset list: $ACCEPTED_DATASETS_LIST"

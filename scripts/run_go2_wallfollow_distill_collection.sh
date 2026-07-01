#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/.generated/go2_memory_closed_loop/generalized_learned_local_suite_v65_wallfollow_distill_20260629}"
MODE="${MODE:-kinematic}"
SPLIT="${SPLIT:-train}"
DEVICE="${DEVICE:-cpu}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
MAX_TICKS="${MAX_TICKS:-700}"
TARGET_COLORS="${TARGET_COLORS:-red,yellow,blue,green}"
TARGET_SWITCH_POLICY="${TARGET_SWITCH_POLICY:-visible_priority}"
TARGET_SWITCH_CONF="${TARGET_SWITCH_CONF:-0.8}"
LEARNED_LOCAL_ONLINE_MAP_SIZE="${LEARNED_LOCAL_ONLINE_MAP_SIZE:-21}"
LEARNED_LOCAL_ONLINE_MAP_CELL_M="${LEARNED_LOCAL_ONLINE_MAP_CELL_M:-0.45}"
WALL_FOLLOW_SAFE_RISK="${WALL_FOLLOW_SAFE_RISK:-0.75}"
WALL_FOLLOW_PROGRESS_FLOOR="${WALL_FOLLOW_PROGRESS_FLOOR:-0.01}"
WALL_FOLLOW_TURN_PRESSURE_AFTER="${WALL_FOLLOW_TURN_PRESSURE_AFTER:-3}"
LABEL_SOURCE="${LABEL_SOURCE:-final_primitive}"
ROUND_NAME="${ROUND_NAME:-wallfollow_teacher}"
TRAIN_SCENES="${TRAIN_SCENES:-medium_enclosed_maze_000c67a65968,medium_enclosed_maze_0100925f9754,medium_enclosed_maze_01732aabc542,medium_enclosed_maze_5ae240e5e391,medium_enclosed_maze_62c21394102b,medium_enclosed_maze_b8c906fc9e8e}"

mkdir -p "$OUT_DIR"
IFS=',' read -r -a TRAIN_SCENE_ARRAY <<< "$TRAIN_SCENES"

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
  --success-dist-m 1.2
  --mask-area-threshold 0.01
  --claim-min-seen-ticks 18
  --claim-near-area-logit 4.0
  --claim-near-area-logit-by-color yellow:2.45,green:2.45
  --claim-near-bearing 0.32
  --claim-near-min-seen-ticks 8
  --multi-target-switch-policy "$TARGET_SWITCH_POLICY"
  --multi-target-switch-conf "$TARGET_SWITCH_CONF"
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
  --primitive-outcome-preserve-turn-until-first-claim
  --primitive-outcome-blocked-hard-veto
  --primitive-outcome-blocked-hard-veto-after-first-claim
  --body-clearance-target-servo
  --body-clearance-target-area-logit 1.2
  --body-clearance-target-bearing 0.16
  --body-clearance-target-forward-primitive forward_medium
  --body-clearance-latch-ticks 8
  --body-clearance-learned-min-area-logit 1.2
  --explore-reset-on-claim
  --explore-clear-visited-on-claim
  --explore-goal-policy learned_wall_follow
  --learned-wall-follow-safe-risk "$WALL_FOLLOW_SAFE_RISK"
  --learned-wall-follow-progress-floor "$WALL_FOLLOW_PROGRESS_FLOOR"
  --learned-wall-follow-turn-pressure-after "$WALL_FOLLOW_TURN_PRESSURE_AFTER"
  --learned-local-policy-states EXPLORE
  --learned-local-clock-features
  --learned-local-state-features
  --learned-local-online-map-features
  --learned-local-online-map-size "$LEARNED_LOCAL_ONLINE_MAP_SIZE"
  --learned-local-online-map-cell-m "$LEARNED_LOCAL_ONLINE_MAP_CELL_M"
)

for SCENE_ID in "${TRAIN_SCENE_ARRAY[@]}"; do
  RESULT="$OUT_DIR/${ROUND_NAME}_${SCENE_ID}_result.json"
  RAW_DATASET="$OUT_DIR/${ROUND_NAME}_${SCENE_ID}_raw_dataset.npz"
  RELABELED_DATASET="$OUT_DIR/${ROUND_NAME}_${SCENE_ID}_${LABEL_SOURCE}_labels.npz"
  echo "collect $SCENE_ID -> $RAW_DATASET"
  "$PYTHON" "$ROOT/scripts/benchmark_go2_memory_closed_loop.py" \
    --mode "$MODE" \
    --policy-device "$POLICY_DEVICE" \
    --device "$DEVICE" \
    --split "$SPLIT" \
    --scene-id "$SCENE_ID" \
    --max-ticks "$MAX_TICKS" \
    "${COMMON_ARGS[@]}" \
    --learned-local-dataset-output "$RAW_DATASET" \
    --output "$RESULT"
  echo "relabel $SCENE_ID -> $RELABELED_DATASET"
  "$PYTHON" "$ROOT/scripts/relabel_go2_learned_local_dataset_from_result.py" \
    --dataset "$RAW_DATASET" \
    --result "$RESULT" \
    --label-source "$LABEL_SOURCE" \
    --output "$RELABELED_DATASET"
done

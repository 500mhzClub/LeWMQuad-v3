#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"
OUT_PREFIX="${OUT_PREFIX:-$ROOT/.generated/go2_memory_closed_loop/fully_learned_topology_route_memory_policy50}"
MODE="${MODE:-physical}"
MAX_TICKS="${MAX_TICKS:-400}"
DEVICE="${DEVICE:-cpu}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
SCENE_ID="${SCENE_ID:-medium_enclosed_maze_01732aabc542}"
LEARNED_TOPOLOGY_ROUTE_TABLE="${LEARNED_TOPOLOGY_ROUTE_TABLE:-$ROOT/.generated/go2_memory_closed_loop/fully_learned_topology_route_table_try010_pose_dense005.json}"

"$PYTHON" "$ROOT/scripts/benchmark_go2_memory_closed_loop.py" \
  --fully-learned-runtime-contract \
  --mode "$MODE" \
  --policy-device "$POLICY_DEVICE" \
  --device "$DEVICE" \
  --scene-id "$SCENE_ID" \
  --policy memory \
  --demo-mode explore \
  --target-color all \
  --target-colors red,yellow,blue,green \
  --controller "$ROOT/.generated/go2_hidden_target_memory/go2_rgb_jepa_strict_exact_valuenorm_gate_neg6_pair8_nonforward_eval_seed20260825_h512.pt" \
  --frozen-jepa-checkpoint "$ROOT/.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt" \
  --learned-topology-route-table "$LEARNED_TOPOLOGY_ROUTE_TABLE" \
  --learned-topology-route-until-area-logit 1.0 \
  --learned-topology-route-advance-m 0.38 \
  --learned-topology-route-yaw-threshold 0.50 \
  --learned-topology-route-forward-threshold 0.12 \
  --learned-topology-route-arc-max-bearing 0.35 \
  --primitive-outcome-checkpoint "$ROOT/.generated/go2_wallaware_learned/primitive_outcome_jepa_mixed_progress08.pt" \
  --wall-aware-planner \
  --wall-decision-source learned_action \
  --max-ticks "$MAX_TICKS" \
  --success-dist-m 1.2 \
  --mask-area-threshold 0.01 \
  --claim-min-seen-ticks 18 \
  --claim-near-area-logit 2.6 \
  --claim-near-area-logit-by-color yellow:2.77,green:2.45 \
  --claim-near-bearing 0.32 \
  --claim-near-bearing-by-color green:0.65 \
  --claim-near-min-seen-ticks 4 \
  --claim-near-min-seen-ticks-by-color yellow:10 \
  --multi-target-switch-conf 0.8 \
  --wall-body-probe-margin-m 0.0 \
  --wall-stall-block-waypoint \
  --wall-predicted-blocked-waypoint-replan \
  --wall-predicted-blocked-waypoint-streak 2 \
  --primitive-outcome-forward-progress-floor 0.04 \
  --primitive-outcome-progress-floor-min-blocked-prob 0.05 \
  --primitive-outcome-progress-floor-force-below 0.01 \
  --primitive-outcome-forward-progress-penalty 0.9 \
  --primitive-outcome-progress-floor-prefer-yaw \
  --primitive-outcome-preserve-turn-requests \
  --primitive-outcome-preserve-turn-states EXPLORE \
  --primitive-outcome-preserve-arc-requests \
  --primitive-outcome-turn-body-rerank-primitives yaw_left,yaw_right,backward \
  --primitive-outcome-blocked-hard-veto \
  --primitive-outcome-blocked-hard-veto-selected-primitives arc_left,arc_right \
  --primitive-outcome-blocked-hard-veto-max-abs-bearing 0.18 \
  --body-clearance-target-servo \
  --body-clearance-target-area-logit 1.2 \
  --body-clearance-target-bearing 0.16 \
  --body-clearance-target-forward-primitive forward_slow \
  --body-clearance-latch-ticks 8 \
  --body-clearance-learned-min-area-logit 1.2 \
  --demo-capture-rate policy \
  --demo-fps 50 \
  --demo-video "${OUT_PREFIX}.mp4" \
  --output "${OUT_PREFIX}_result.json"

"$PYTHON" "$ROOT/scripts/check_go2_fully_learned_demo.py" \
  --result "${OUT_PREFIX}_result.json" \
  --max-ticks "$MAX_TICKS"

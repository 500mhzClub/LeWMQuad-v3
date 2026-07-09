#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.generated/venvs/genesis_render_vulkan/bin/python}"
OUT_PREFIX="${OUT_PREFIX:-$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/attempt397_full_from_spawn_localpolicy_strict}"
MODE="${MODE:-physical}"
MAX_TICKS="${MAX_TICKS:-1000}"
DEVICE="${DEVICE:-cpu}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
SCENE_ID="${SCENE_ID:-medium_enclosed_maze_01732aabc542}"
RENDER="${RENDER:-1}"
REVIEW_UI="${REVIEW_UI:-1}"
RENDER_PROGRESS_EVERY="${RENDER_PROGRESS_EVERY:-25}"
CHECK_RESULT="${CHECK_RESULT:-1}"
RUNTIME_CONTRACT="${RUNTIME_CONTRACT:-1}"
DEBUG_FORCE_PRIMITIVE_SCRIPT="${DEBUG_FORCE_PRIMITIVE_SCRIPT:-}"

CONTROLLER="${CONTROLLER:-$ROOT/.generated/go2_hidden_target_memory/go2_rgb_jepa_strict_exact_valuenorm_gate_neg6_pair8_nonforward_eval_seed20260825_h512.pt}"
FROZEN_JEPA_CHECKPOINT="${FROZEN_JEPA_CHECKPOINT:-$ROOT/.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt}"
PRIMITIVE_OUTCOME_CHECKPOINT="${PRIMITIVE_OUTCOME_CHECKPOINT:-$ROOT/.generated/go2_wallaware_learned/primitive_outcome_jepa_v235_geoencv4_collisionprogress004_h224.pt}"
GEOMETRIC_JEPA_CHECKPOINT="${GEOMETRIC_JEPA_CHECKPOINT:-$ROOT/.generated/go2_wallaware_learned/go2_jepa_geometric_encoder_v4_medium41_crossfam_lat192_img128.pt}"
LEARNED_LOCAL_POLICY_CHECKPOINT="${LEARNED_LOCAL_POLICY_CHECKPOINT:-$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/learned_local_01732_mapcnn_v3_balanced.pt}"
LEARNED_TARGET_SCHEDULER_ENABLED="${LEARNED_TARGET_SCHEDULER_ENABLED:-1}"
LEARNED_TARGET_SCHEDULER_CHECKPOINT="${LEARNED_TARGET_SCHEDULER_CHECKPOINT:-$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/target_scheduler_01732_v4_keepblue.pt}"
CLAIM_SUCCESS_MODEL_CHECKPOINT="${CLAIM_SUCCESS_MODEL_CHECKPOINT:-$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/claim_success_01732_hardneg_v6_colorcal_blue098_yellow098.pt}"
LEARNED_LOCAL_TARGET_POLICY_CHECKPOINTS="${LEARNED_LOCAL_TARGET_POLICY_CHECKPOINTS:-red=$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/learned_local_01732_red_v2_current_goodprefix.pt,yellow=$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/learned_local_01732_yellow_v82_v80_slicegeom.pt,green=$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/learned_local_01732_green_v6_dagger023.pt,blue=$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/learned_local_01732_blue_v8_contact_m06.pt}"
LEARNED_LOCAL_POST_CLAIM_POLICY_CHECKPOINT="${LEARNED_LOCAL_POST_CLAIM_POLICY_CHECKPOINT:-$ROOT/.generated/go2_memory_closed_loop/learned_physical_01732/learned_local_01732_yellow_v52_gru_fullcontinuous_dagger149.pt}"
PRIMITIVE_CLEARANCE_CHECKPOINT="${PRIMITIVE_CLEARANCE_CHECKPOINT:-$ROOT/.generated/go2_wallaware_learned/primitive_body_clearance_jepa_v234_geoencv4_obstacle_margin002_afterstart_h192.pt}"
PRIMITIVE_AUX_CLEARANCE_CHECKPOINT="${PRIMITIVE_AUX_CLEARANCE_CHECKPOINT:-$ROOT/.generated/go2_wallaware_learned/primitive_body_clearance_jepa_v302_mixed01732shoulder_obstacle_margin002_afterstart_h192.pt}"
CURRENT_BODY_RISK_CHECKPOINT="${CURRENT_BODY_RISK_CHECKPOINT:-$ROOT/.generated/go2_wallaware_learned/shoulder_switch_jepa_01732_v1_memenc.pt}"
CURRENT_BODY_RISK_THRESHOLD="${CURRENT_BODY_RISK_THRESHOLD:-0.5}"
CURRENT_BODY_RISK_MIN_AREA_LOGIT="${CURRENT_BODY_RISK_MIN_AREA_LOGIT:-}"
CURRENT_BODY_RISK_MIN_CLAIMED_COUNT="${CURRENT_BODY_RISK_MIN_CLAIMED_COUNT:-3}"
CURRENT_BODY_RISK_RECOVERY_BLOCKS="${CURRENT_BODY_RISK_RECOVERY_BLOCKS:-0}"
CURRENT_BODY_RISK_RECOVERY_SELECTED_PROB_FLOOR="${CURRENT_BODY_RISK_RECOVERY_SELECTED_PROB_FLOOR:-}"
CURRENT_BODY_RISK_RECOVERY_SELECTED_PRIMITIVES="${CURRENT_BODY_RISK_RECOVERY_SELECTED_PRIMITIVES:-}"
CURRENT_BODY_RISK_PRESERVE_YAW="${CURRENT_BODY_RISK_PRESERVE_YAW:-0}"
CURRENT_BODY_RISK_PRESERVE_YAW_THRESHOLD="${CURRENT_BODY_RISK_PRESERVE_YAW_THRESHOLD:-}"
CURRENT_BODY_RISK_PRESERVE_YAW_MAX_CLEARANCE_PROB="${CURRENT_BODY_RISK_PRESERVE_YAW_MAX_CLEARANCE_PROB:-}"
PRIMITIVE_AUX_CLEARANCE_SWITCH_THRESHOLD="${PRIMITIVE_AUX_CLEARANCE_SWITCH_THRESHOLD:-$CURRENT_BODY_RISK_THRESHOLD}"
PRIMITIVE_AUX_CLEARANCE_SWITCH_MIN_CLAIMED_COUNT="${PRIMITIVE_AUX_CLEARANCE_SWITCH_MIN_CLAIMED_COUNT:-3}"
PRIMITIVE_AUX_CLEARANCE_SWITCH_LATCH_TICKS="${PRIMITIVE_AUX_CLEARANCE_SWITCH_LATCH_TICKS:-40}"
BODY_CLEARANCE_VETO_MIN_CLAIMED_COUNT="${BODY_CLEARANCE_VETO_MIN_CLAIMED_COUNT:-0}"
BODY_CLEARANCE_HARD_VETO_PROB="${BODY_CLEARANCE_HARD_VETO_PROB:-0.65}"
BODY_CLEARANCE_HARD_VETO_MARGIN="${BODY_CLEARANCE_HARD_VETO_MARGIN:-0.03}"
BODY_CLEARANCE_HARD_VETO_REPLACEMENT_CAP="${BODY_CLEARANCE_HARD_VETO_REPLACEMENT_CAP:-0.90}"
BODY_CLEARANCE_YAW_CONTACT_VETO_PROB="${BODY_CLEARANCE_YAW_CONTACT_VETO_PROB:-0.90}"
BODY_CLEARANCE_YAW_DIRECTION_VETO_PROB="${BODY_CLEARANCE_YAW_DIRECTION_VETO_PROB:-1.01}"
BODY_CLEARANCE_AUX_VETO_PROB="${BODY_CLEARANCE_AUX_VETO_PROB:-1.01}"
BODY_CLEARANCE_NEAR_YAW_PROB_WEIGHT="${BODY_CLEARANCE_NEAR_YAW_PROB_WEIGHT:-1.0}"
BODY_CLEARANCE_TARGET_AREA_HARD_VETO_PROB="${BODY_CLEARANCE_TARGET_AREA_HARD_VETO_PROB:-0.18}"
BODY_CLEARANCE_TARGET_AREA_HARD_VETO_MIN_AREA_LOGIT="${BODY_CLEARANCE_TARGET_AREA_HARD_VETO_MIN_AREA_LOGIT:-2.8}"
BODY_CLEARANCE_GEOMETRY_VETO_MIN_CLEARANCE_M="${BODY_CLEARANCE_GEOMETRY_VETO_MIN_CLEARANCE_M:-}"
BODY_CLEARANCE_GEOMETRY_VETO_FEASIBLE_THRESHOLD="${BODY_CLEARANCE_GEOMETRY_VETO_FEASIBLE_THRESHOLD:-1.0}"
BODY_CLEARANCE_GEOMETRY_VETO_STATES="${BODY_CLEARANCE_GEOMETRY_VETO_STATES:-EXPLORE}"
BODY_CLEARANCE_GEOMETRY_VETO_MIN_CLAIMED_COUNT="${BODY_CLEARANCE_GEOMETRY_VETO_MIN_CLAIMED_COUNT:-0}"
BODY_CLEARANCE_GEOMETRY_VETO_TARGET_COLORS="${BODY_CLEARANCE_GEOMETRY_VETO_TARGET_COLORS:-}"
BODY_CLEARANCE_GEOMETRY_VETO_SELECTED_PRIMITIVES="${BODY_CLEARANCE_GEOMETRY_VETO_SELECTED_PRIMITIVES:-forward_slow,forward_medium,forward_fast,arc_left,arc_right,yaw_left,yaw_right,backward}"
BODY_CLEARANCE_GEOMETRY_VETO_REPLACEMENTS="${BODY_CLEARANCE_GEOMETRY_VETO_REPLACEMENTS:-forward_slow,arc_left,arc_right,yaw_left,yaw_right,backward,hold}"
BODY_CLEARANCE_GEOMETRY_VETO_BLOCKED_FALLBACK_PRIMITIVES="${BODY_CLEARANCE_GEOMETRY_VETO_BLOCKED_FALLBACK_PRIMITIVES:-}"
BODY_CLEARANCE_GEOMETRY_VETO_OVERRIDE_REPLACEMENTS="${BODY_CLEARANCE_GEOMETRY_VETO_OVERRIDE_REPLACEMENTS:-}"
BODY_CLEARANCE_GEOMETRY_VETO_OVERRIDE_MIN_CLAIMED_COUNT="${BODY_CLEARANCE_GEOMETRY_VETO_OVERRIDE_MIN_CLAIMED_COUNT:-0}"
BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_GUARD_DISABLED="${BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_GUARD_DISABLED:-0}"
BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_FORCE_SINGLE_CANDIDATE="${BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_FORCE_SINGLE_CANDIDATE:-0}"
WALL_GUARD_STATES="${WALL_GUARD_STATES:-EXPLORE}"
WALL_GUARD_POST_CLAIM_STATES="${WALL_GUARD_POST_CLAIM_STATES:-SERVO}"
WALL_GUARD_POST_CLAIM_MIN_CLAIMS="${WALL_GUARD_POST_CLAIM_MIN_CLAIMS:-0}"
CURRENT_BODY_RISK_CLEARANCE_RERANK_THRESHOLD="${CURRENT_BODY_RISK_CLEARANCE_RERANK_THRESHOLD:-0.95}"
CURRENT_BODY_RISK_CLEARANCE_RERANK_SELECTED_PROB_FLOOR="${CURRENT_BODY_RISK_CLEARANCE_RERANK_SELECTED_PROB_FLOOR:-0.12}"
CURRENT_BODY_RISK_CLEARANCE_RERANK_SELECTED_PRIMITIVES="${CURRENT_BODY_RISK_CLEARANCE_RERANK_SELECTED_PRIMITIVES:-}"
CURRENT_BODY_RISK_CLEARANCE_RERANK_PRIMITIVES="${CURRENT_BODY_RISK_CLEARANCE_RERANK_PRIMITIVES:-backward}"
PROPRIO_CONTACT_DETECTOR_CHECKPOINT="${PROPRIO_CONTACT_DETECTOR_CHECKPOINT:-}"
PROPRIO_CONTACT_ESCAPE_THRESHOLD="${PROPRIO_CONTACT_ESCAPE_THRESHOLD:-0.7}"
PROPRIO_CONTACT_ESCAPE_STREAK="${PROPRIO_CONTACT_ESCAPE_STREAK:-2}"
PROPRIO_CONTACT_ESCAPE_BLOCKS="${PROPRIO_CONTACT_ESCAPE_BLOCKS:-0}"
PROPRIO_CONTACT_ESCAPE_COOLDOWN_TICKS="${PROPRIO_CONTACT_ESCAPE_COOLDOWN_TICKS:-12}"
PROPRIO_CONTACT_ESCAPE_STATES="${PROPRIO_CONTACT_ESCAPE_STATES:-EXPLORE,SEEK,SERVO}"
PROPRIO_CONTACT_MAP_BLOCKS="${PROPRIO_CONTACT_MAP_BLOCKS:-0}"
HISTORY_RISK_CHECKPOINT="${HISTORY_RISK_CHECKPOINT:-}"
HISTORY_RISK_VETO_THRESHOLD="${HISTORY_RISK_VETO_THRESHOLD:-1.01}"
HISTORY_RISK_VETO_PRIMITIVES="${HISTORY_RISK_VETO_PRIMITIVES:-forward_slow,forward_medium,forward_fast,arc_left,arc_right,yaw_left,yaw_right}"
HISTORY_RISK_REPLACEMENTS="${HISTORY_RISK_REPLACEMENTS:-backward,yaw_left,yaw_right,hold}"
HISTORY_RISK_REPLACEMENT_CAP="${HISTORY_RISK_REPLACEMENT_CAP:-0.9}"
HISTORY_RISK_STATES="${HISTORY_RISK_STATES:-EXPLORE,SEEK,SERVO}"
HISTORY_RISK_WEDGE_ESCAPE_BLOCKS="${HISTORY_RISK_WEDGE_ESCAPE_BLOCKS:-2}"
HISTORY_RISK_WEDGE_ESCAPE_COOLDOWN_TICKS="${HISTORY_RISK_WEDGE_ESCAPE_COOLDOWN_TICKS:-6}"
HISTORY_RISK_FUSE_OUTCOMES="${HISTORY_RISK_FUSE_OUTCOMES:-0}"
HISTORY_RISK_FUSE_WEIGHT="${HISTORY_RISK_FUSE_WEIGHT:-1.0}"
HISTORY_RISK_CORRIDOR_COMMIT="${HISTORY_RISK_CORRIDOR_COMMIT:-0}"
HISTORY_RISK_CORRIDOR_YAW_MIN="${HISTORY_RISK_CORRIDOR_YAW_MIN:-0.7}"
HISTORY_RISK_CORRIDOR_FORWARD_MAX="${HISTORY_RISK_CORRIDOR_FORWARD_MAX:-0.3}"
HISTORY_RISK_CORRIDOR_MAX_RUN="${HISTORY_RISK_CORRIDOR_MAX_RUN:-6}"
HISTORY_RISK_CORRIDOR_STATES="${HISTORY_RISK_CORRIDOR_STATES:-EXPLORE,SEEK}"
HISTORY_RISK_RELAX_MIN_CLAIMS="${HISTORY_RISK_RELAX_MIN_CLAIMS:--1}"
HISTORY_RISK_RELAXED_VETO_THRESHOLD="${HISTORY_RISK_RELAXED_VETO_THRESHOLD:-0.97}"
HISTORY_RISK_RELAXED_FUSE_WEIGHT="${HISTORY_RISK_RELAXED_FUSE_WEIGHT:-0.4}"

BENCHMARK_EXTRA_ARGS=()
if [[ -n "$HISTORY_RISK_CHECKPOINT" ]]; then
  BENCHMARK_EXTRA_ARGS+=(
    --history-risk-checkpoint "$HISTORY_RISK_CHECKPOINT"
    --history-risk-veto-threshold "$HISTORY_RISK_VETO_THRESHOLD"
    --history-risk-veto-primitives "$HISTORY_RISK_VETO_PRIMITIVES"
    --history-risk-replacements "$HISTORY_RISK_REPLACEMENTS"
    --history-risk-replacement-cap "$HISTORY_RISK_REPLACEMENT_CAP"
    --history-risk-states "$HISTORY_RISK_STATES"
    --history-risk-wedge-escape-blocks "$HISTORY_RISK_WEDGE_ESCAPE_BLOCKS"
    --history-risk-wedge-escape-cooldown-ticks "$HISTORY_RISK_WEDGE_ESCAPE_COOLDOWN_TICKS"
    --history-risk-fuse-weight "$HISTORY_RISK_FUSE_WEIGHT"
    --history-risk-relax-min-claims "$HISTORY_RISK_RELAX_MIN_CLAIMS"
    --history-risk-relaxed-veto-threshold "$HISTORY_RISK_RELAXED_VETO_THRESHOLD"
    --history-risk-relaxed-fuse-weight "$HISTORY_RISK_RELAXED_FUSE_WEIGHT"
  )
  if [[ "$HISTORY_RISK_FUSE_OUTCOMES" == "1" || "$HISTORY_RISK_FUSE_OUTCOMES" == "true" ]]; then
    BENCHMARK_EXTRA_ARGS+=(--history-risk-fuse-outcomes)
  fi
  if [[ "$HISTORY_RISK_CORRIDOR_COMMIT" == "1" || "$HISTORY_RISK_CORRIDOR_COMMIT" == "true" ]]; then
    BENCHMARK_EXTRA_ARGS+=(
      --history-risk-corridor-commit
      --history-risk-corridor-yaw-min "$HISTORY_RISK_CORRIDOR_YAW_MIN"
      --history-risk-corridor-forward-max "$HISTORY_RISK_CORRIDOR_FORWARD_MAX"
      --history-risk-corridor-max-run "$HISTORY_RISK_CORRIDOR_MAX_RUN"
      --history-risk-corridor-states "$HISTORY_RISK_CORRIDOR_STATES"
    )
  fi
fi
if [[ -n "$PROPRIO_CONTACT_DETECTOR_CHECKPOINT" ]]; then
  BENCHMARK_EXTRA_ARGS+=(
    --proprio-contact-detector-checkpoint "$PROPRIO_CONTACT_DETECTOR_CHECKPOINT"
    --proprio-contact-escape-threshold "$PROPRIO_CONTACT_ESCAPE_THRESHOLD"
    --proprio-contact-escape-streak "$PROPRIO_CONTACT_ESCAPE_STREAK"
    --proprio-contact-escape-blocks "$PROPRIO_CONTACT_ESCAPE_BLOCKS"
    --proprio-contact-escape-cooldown-ticks "$PROPRIO_CONTACT_ESCAPE_COOLDOWN_TICKS"
    --proprio-contact-escape-states "$PROPRIO_CONTACT_ESCAPE_STATES"
  )
  if [[ "$PROPRIO_CONTACT_MAP_BLOCKS" == "1" || "$PROPRIO_CONTACT_MAP_BLOCKS" == "true" ]]; then
    BENCHMARK_EXTRA_ARGS+=(--proprio-contact-map-blocks)
  fi
fi
if [[ -n "$CURRENT_BODY_RISK_MIN_AREA_LOGIT" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--current-body-risk-min-area-logit "$CURRENT_BODY_RISK_MIN_AREA_LOGIT")
fi
if [[ -n "$CURRENT_BODY_RISK_RECOVERY_SELECTED_PROB_FLOOR" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--current-body-risk-recovery-selected-prob-floor "$CURRENT_BODY_RISK_RECOVERY_SELECTED_PROB_FLOOR")
fi
if [[ -n "$CURRENT_BODY_RISK_RECOVERY_SELECTED_PRIMITIVES" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--current-body-risk-recovery-selected-primitives "$CURRENT_BODY_RISK_RECOVERY_SELECTED_PRIMITIVES")
fi
if [[ "$CURRENT_BODY_RISK_PRESERVE_YAW" == "1" || "$CURRENT_BODY_RISK_PRESERVE_YAW" == "true" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--current-body-risk-preserve-yaw)
fi
if [[ -n "$CURRENT_BODY_RISK_PRESERVE_YAW_THRESHOLD" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--current-body-risk-preserve-yaw-threshold "$CURRENT_BODY_RISK_PRESERVE_YAW_THRESHOLD")
fi
if [[ -n "$CURRENT_BODY_RISK_PRESERVE_YAW_MAX_CLEARANCE_PROB" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--current-body-risk-preserve-yaw-max-clearance-prob "$CURRENT_BODY_RISK_PRESERVE_YAW_MAX_CLEARANCE_PROB")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_M:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-m "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_M")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_M_BY_PRIMITIVE:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-m-by-primitive "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_M_BY_PRIMITIVE")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_STREAK:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-min-streak "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_STREAK")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_COOLDOWN_TICKS:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-cooldown-ticks "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_COOLDOWN_TICKS")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_CLAIMED_COUNT:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-min-claimed-count "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_CLAIMED_COUNT")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_STATES:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-states "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_STATES")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_TARGET_COLORS:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-target-colors "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_TARGET_COLORS")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_PRIMITIVES:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-primitives "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_PRIMITIVES")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_REPLACEMENTS:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-replacements "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_REPLACEMENTS")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_REPLACEMENT_CAP:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-replacement-cap "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_REPLACEMENT_CAP")
fi
if [[ "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_REQUIRE_REPLACEMENT_UNDER_CAP:-0}" == "1" || "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_REQUIRE_REPLACEMENT_UNDER_CAP:-0}" == "true" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-require-replacement-under-cap)
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_AREA_LOGIT:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-min-area-logit "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_AREA_LOGIT")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_AREA_STATES:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-min-area-states "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_AREA_STATES")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_PROJECTED_CLEARANCE_M:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-min-projected-clearance-m "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_PROJECTED_CLEARANCE_M")
fi
if [[ -n "${BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_PROJECTED_IMPROVEMENT_M:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--body-clearance-current-contact-escape-min-projected-improvement-m "$BODY_CLEARANCE_CURRENT_CONTACT_ESCAPE_MIN_PROJECTED_IMPROVEMENT_M")
fi
if [[ -n "$BODY_CLEARANCE_GEOMETRY_VETO_MIN_CLEARANCE_M" ]]; then
  BENCHMARK_EXTRA_ARGS+=(
    --body-clearance-geometry-veto-min-clearance-m "$BODY_CLEARANCE_GEOMETRY_VETO_MIN_CLEARANCE_M"
    --body-clearance-geometry-veto-feasible-threshold "$BODY_CLEARANCE_GEOMETRY_VETO_FEASIBLE_THRESHOLD"
    --body-clearance-geometry-veto-states "$BODY_CLEARANCE_GEOMETRY_VETO_STATES"
    --body-clearance-geometry-veto-min-claimed-count "$BODY_CLEARANCE_GEOMETRY_VETO_MIN_CLAIMED_COUNT"
    --body-clearance-geometry-veto-selected-primitives "$BODY_CLEARANCE_GEOMETRY_VETO_SELECTED_PRIMITIVES"
    --body-clearance-geometry-veto-replacements "$BODY_CLEARANCE_GEOMETRY_VETO_REPLACEMENTS"
    --body-clearance-geometry-veto-override-min-claimed-count "$BODY_CLEARANCE_GEOMETRY_VETO_OVERRIDE_MIN_CLAIMED_COUNT"
  )
  if [[ -n "$BODY_CLEARANCE_GEOMETRY_VETO_TARGET_COLORS" ]]; then
    BENCHMARK_EXTRA_ARGS+=(
      --body-clearance-geometry-veto-target-colors "$BODY_CLEARANCE_GEOMETRY_VETO_TARGET_COLORS"
    )
  fi
  if [[ -n "$BODY_CLEARANCE_GEOMETRY_VETO_OVERRIDE_REPLACEMENTS" ]]; then
    BENCHMARK_EXTRA_ARGS+=(
      --body-clearance-geometry-veto-override-replacements "$BODY_CLEARANCE_GEOMETRY_VETO_OVERRIDE_REPLACEMENTS"
    )
  fi
  if [[ -n "$BODY_CLEARANCE_GEOMETRY_VETO_BLOCKED_FALLBACK_PRIMITIVES" ]]; then
    BENCHMARK_EXTRA_ARGS+=(
      --body-clearance-geometry-veto-blocked-fallback-primitives "$BODY_CLEARANCE_GEOMETRY_VETO_BLOCKED_FALLBACK_PRIMITIVES"
    )
  fi
  if [[ "$BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_GUARD_DISABLED" == "1" || "$BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_GUARD_DISABLED" == "true" ]]; then
    BENCHMARK_EXTRA_ARGS+=(--body-clearance-geometry-veto-allow-guard-disabled)
  fi
  if [[ "$BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_FORCE_SINGLE_CANDIDATE" == "1" || "$BODY_CLEARANCE_GEOMETRY_VETO_ALLOW_FORCE_SINGLE_CANDIDATE" == "true" ]]; then
    BENCHMARK_EXTRA_ARGS+=(--body-clearance-geometry-veto-allow-force-single-candidate)
  fi
fi
if [[ -n "${SLICE_START_RESULT:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-start-result "$SLICE_START_RESULT")
fi
if [[ -n "${SLICE_START_TICK:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-start-tick "$SLICE_START_TICK")
fi
if [[ -n "${SLICE_ACTIVE_TARGET_COLOR:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-active-target-color "$SLICE_ACTIVE_TARGET_COLOR")
fi
if [[ -n "${SLICE_PRECLAIMED_COLORS:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-preclaimed-colors "$SLICE_PRECLAIMED_COLORS")
fi
if [[ -n "${SLICE_FEATURE_MAX_TICKS:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-feature-max-ticks "$SLICE_FEATURE_MAX_TICKS")
fi
if [[ "${SLICE_PRELOAD_ONLINE_MAP:-0}" == "1" || "${SLICE_PRELOAD_ONLINE_MAP:-0}" == "true" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-preload-online-map)
fi
if [[ -n "${SLICE_SNAPSHOT_OUTPUT:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-snapshot-output "$SLICE_SNAPSHOT_OUTPUT")
fi
if [[ -n "${SLICE_SNAPSHOT_AFTER_CLAIMS:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-snapshot-after-claims "$SLICE_SNAPSHOT_AFTER_CLAIMS")
fi
if [[ -n "${SLICE_SNAPSHOT_AT_TICK:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-snapshot-at-tick "$SLICE_SNAPSHOT_AT_TICK")
fi
if [[ "${SLICE_SNAPSHOT_EXIT:-0}" == "1" || "${SLICE_SNAPSHOT_EXIT:-0}" == "true" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-snapshot-exit)
fi
if [[ -n "${SLICE_SNAPSHOT_INPUT:-}" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--slice-snapshot-input "$SLICE_SNAPSHOT_INPUT")
fi
if [[ -n "$DEBUG_FORCE_PRIMITIVE_SCRIPT" ]]; then
  BENCHMARK_EXTRA_ARGS+=(--debug-force-primitive-script "$DEBUG_FORCE_PRIMITIVE_SCRIPT")
fi
if [[ -n "${EXTRA_BENCHMARK_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  BENCHMARK_EXTRA_ARGS+=($EXTRA_BENCHMARK_ARGS)
fi

TARGET_SCHEDULER_ARGS=()
if [[ "$LEARNED_TARGET_SCHEDULER_ENABLED" == "1" || "$LEARNED_TARGET_SCHEDULER_ENABLED" == "true" ]]; then
  TARGET_SCHEDULER_ARGS+=(
    --learned-target-scheduler-checkpoint "$LEARNED_TARGET_SCHEDULER_CHECKPOINT"
    --learned-target-scheduler-log-scores
  )
fi

CONTRACT_ARGS=()
if [[ "$RUNTIME_CONTRACT" == "1" || "$RUNTIME_CONTRACT" == "true" ]]; then
  CONTRACT_ARGS+=(--generalized-runtime-contract --fully-learned-runtime-contract)
fi

"$PYTHON" "$ROOT/scripts/benchmark_go2_memory_closed_loop.py" \
  "${CONTRACT_ARGS[@]}" \
  --mode "$MODE" \
  --policy-device "$POLICY_DEVICE" \
  --device "$DEVICE" \
  --split "${SPLIT:-train}" \
  --scene-id "$SCENE_ID" \
  --seed 1 \
  --policy memory \
  --demo-mode explore \
  --target-color all \
  --target-colors red,yellow,blue,green \
  --controller "$CONTROLLER" \
  --frozen-jepa-checkpoint "$FROZEN_JEPA_CHECKPOINT" \
  --primitive-outcome-checkpoint "$PRIMITIVE_OUTCOME_CHECKPOINT" \
  --primitive-clearance-threshold 1.01 \
  --body-clearance-learned-prob-floor 0.35 \
  --body-clearance-learned-prob-weight 0.2 \
  --body-clearance-saturated-veto-prob 1.01 \
  --body-clearance-saturated-veto-spread 0.01 \
  --body-clearance-saturated-veto-primitives yaw_left,yaw_right,backward,hold \
  --body-clearance-saturated-veto-selected-primitives arc_left,arc_right \
  --body-clearance-yaw-direction-veto-margin 0.05 \
  --body-clearance-risk-escape-threshold 1.01 \
  --body-clearance-risk-escape-blocks 0 \
  --body-clearance-risk-escape-cooldown-ticks 0 \
  --body-clearance-risk-escape-states EXPLORE,SEEK,SERVO \
  --wall-aware-planner \
  --wall-decision-source learned_action \
  --wall-guard-states "$WALL_GUARD_STATES" \
  --wall-guard-post-claim-states "$WALL_GUARD_POST_CLAIM_STATES" \
  --wall-guard-post-claim-min-claims "$WALL_GUARD_POST_CLAIM_MIN_CLAIMS" \
  --mask-area-threshold 0.01 \
  --claim-min-seen-ticks 18 \
  --claim-near-area-logit 999.0 \
  --claim-near-area-logit-by-color red:999.0,yellow:999.0,blue:999.0,green:999.0 \
  --claim-near-bearing 0.32 \
  --claim-near-min-seen-ticks 8 \
  --multi-target-switch-conf 0.3 \
  --multi-target-switch-area-logit 0.0 \
  --wall-body-probe-margin-m 0.0 \
  --wall-predicted-blocked-waypoint-replan \
  --wall-predicted-blocked-waypoint-streak 2 \
  --primitive-outcome-preserve-backward-requests \
  --primitive-outcome-preserve-backward-clearance-margin 0.1 \
  --primitive-outcome-blocked-hard-veto \
  --primitive-outcome-blocked-hard-veto-primitives yaw_left,yaw_right \
  --primitive-outcome-blocked-hard-veto-selected-primitives forward_medium,forward_fast,arc_left,arc_right \
  --primitive-outcome-preserve-turn-requests \
  --primitive-outcome-preserve-turn-states EXPLORE,SEEK,SERVO \
  --primitive-outcome-progress-floor-min-blocked-prob 0.05 \
  --primitive-outcome-progress-floor-force-below 0.01 \
  --primitive-outcome-forward-progress-penalty 0.9 \
  --primitive-outcome-progress-floor-prefer-yaw \
  --body-clearance-target-servo \
  --body-clearance-target-area-logit 1.2 \
  --body-clearance-target-bearing 0.16 \
  --body-clearance-target-forward-primitive forward_medium \
  --body-clearance-latch-ticks 8 \
  --body-clearance-learned-min-area-logit 1.2 \
  --weak-memory-seek-area-logit 0.0 \
  --weak-memory-seek-stall-streak 4 \
  --weak-memory-seek-explore-cooldown-ticks 35 \
  --multi-target-switch-policy fixed \
  --explore-reset-on-claim \
  --explore-goal-policy "${EXPLORE_GOAL_POLICY:-learned_policy}" \
  --learned-local-policy-checkpoint "$LEARNED_LOCAL_POLICY_CHECKPOINT" \
  --learned-local-post-claim-policy-outcome-rerank off \
  --learned-local-target-policy-state-checkpoints "" \
  --learned-local-target-policy-outcome-rerank off \
  --learned-local-policy-states EXPLORE,SEEK,SERVO \
  --learned-local-clock-features \
  --learned-local-state-features \
  --learned-local-online-map-features \
  --learned-local-online-map-size 21 \
  --learned-local-online-map-cell-m 0.45 \
  --learned-local-policy-outcome-rerank \
  --learned-local-policy-rerank-top-k 5 \
  --learned-local-policy-rerank-policy-weight 0.2 \
  --learned-local-policy-rerank-blocked-weight 3.0 \
  --learned-local-policy-rerank-clearance-weight 0.1 \
  --learned-local-policy-rerank-progress-weight 1.0 \
  --learned-local-policy-rerank-hard-blocked-penalty 2.0 \
  --learned-local-policy-rerank-bearing-turn-threshold 0.4 \
  --learned-local-policy-rerank-bearing-turn-bonus 0.4 \
  --learned-local-post-claim-policy-rerank-policy-weight 0.25 \
  --learned-local-policy-translation-pressure-after 0 \
  --learned-local-policy-translation-pressure-max-blocked-prob 1.01 \
  --learned-local-policy-translation-pressure-min-progress-m 0.02 \
  --learned-local-policy-translation-pressure-primitives forward_medium,arc_left,arc_right \
  --learned-local-policy-translation-pressure-states EXPLORE \
  --learned-local-policy-frontier-pressure-after 0 \
  --learned-local-policy-frontier-pressure-max-blocked-prob 1.01 \
  --learned-local-policy-frontier-pressure-min-progress-m 0.02 \
  --learned-local-policy-frontier-pressure-min-route-cells 1 \
  --learned-local-policy-frontier-pressure-guard-blocked-penalty 1.2 \
  --learned-local-policy-frontier-pressure-prefer-unguarded \
  --learned-local-policy-frontier-pressure-map-blocked-backward-claim-escape \
  --learned-local-policy-frontier-pressure-guarded-retry-after-noops 3 \
  --learned-local-policy-frontier-pressure-combined-blocked-retry-after-noops 3 \
  --learned-local-policy-frontier-pressure-commit \
  --learned-local-policy-frontier-pressure-guard-recovery-rerank-on-commit \
  --learned-local-policy-frontier-pressure-guard-recovery-primitives yaw_left,yaw_right,backward \
  --learned-local-policy-frontier-pressure-states EXPLORE \
  --learned-local-policy-online-map-novelty-weight 0.0 \
  --learned-local-policy-online-map-claim-repulsion-weight 0.0 \
  --learned-local-policy-online-map-frontier-route-weight 0.0 \
  --learned-local-online-map-route-replay-guard-override \
  --learned-local-online-map-hard-guard-blocks \
  --weak-memory-seek-colors red,yellow,blue,green \
  --weak-memory-seek-yaw-loop-streak 6 \
  --weak-memory-seek-yaw-loop-max-displacement-m 0.012 \
  --primitive-outcome-frozen-jepa-checkpoint "$GEOMETRIC_JEPA_CHECKPOINT" \
  --primitive-clearance-frozen-jepa-checkpoint "$GEOMETRIC_JEPA_CHECKPOINT" \
  --family medium_enclosed_maze \
  --target-pursuit-stale-ticks 80 \
  --target-pursuit-stale-window-ticks 240 \
  --target-pursuit-stale-suppress-color-ticks 260 \
  --target-pursuit-stale-explore-cooldown-ticks 80 \
  --target-pursuit-stale-states SEEK,SERVO \
  --log-color-readouts \
  --multi-target-stale-seen-switch-after-frontier-noops 80 \
  --multi-target-stale-seen-switch-max-age-ticks 240 \
  --multi-target-opportunistic-claim-min-visible-ticks 8 \
  --primitive-outcome-forward-progress-floor-states EXPLORE \
  --learned-local-policy-post-claim-states EXPLORE,SEEK,SERVO \
  --success-dist-m 1.2 \
  --claim-success-proxy-area-logit-by-color "" \
  "${TARGET_SCHEDULER_ARGS[@]}" \
  --explore-route-waypoints "" \
  --claim-success-model-positive-trigger \
  --claim-success-model-trigger-min-seen-ticks 12 \
  --claim-area-logit 999.0 \
  --claim-success-proxy-bearing-by-color "" \
  --claim-success-model-threshold 0.95 \
  --primitive-outcome-threshold 0.99 \
  --primitive-outcome-forward-progress-floor 0.0 \
  --primitive-outcome-preserve-straight-states EXPLORE \
  --wall-escape-blocks 0 \
  --wall-turn-escape-blocks 0 \
  --wall-turn-loop-streak 0 \
  --wall-stall-streak 9999 \
  --wall-stall-penalty-score 0.0 \
  --wall-stall-penalty-ticks 0 \
  --learned-local-online-map-low-progress-block-m 0.0 \
  --claim-success-model-checkpoint "$CLAIM_SUCCESS_MODEL_CHECKPOINT" \
  --claim-success-model-threshold-by-color blue:0.996,yellow:0.995 \
  --learned-local-target-policy-checkpoints "$LEARNED_LOCAL_TARGET_POLICY_CHECKPOINTS" \
  --learned-local-post-claim-policy-checkpoint "$LEARNED_LOCAL_POST_CLAIM_POLICY_CHECKPOINT" \
  --learned-local-post-claim-policy-min-claims 3 \
  --body-clearance-hard-veto-selected-primitives forward_medium,arc_left,arc_right,yaw_left,yaw_right \
  --body-clearance-hard-veto-primitives backward,yaw_left,yaw_right,arc_left,arc_right \
  --primitive-clearance-checkpoint "$PRIMITIVE_CLEARANCE_CHECKPOINT" \
  --primitive-aux-clearance-checkpoint "$PRIMITIVE_AUX_CLEARANCE_CHECKPOINT" \
  --primitive-aux-clearance-frozen-jepa-checkpoint "$GEOMETRIC_JEPA_CHECKPOINT" \
  --current-body-risk-checkpoint "$CURRENT_BODY_RISK_CHECKPOINT" \
  --current-body-risk-threshold "$CURRENT_BODY_RISK_THRESHOLD" \
  --current-body-risk-min-claimed-count "$CURRENT_BODY_RISK_MIN_CLAIMED_COUNT" \
  --current-body-risk-recovery-blocks "$CURRENT_BODY_RISK_RECOVERY_BLOCKS" \
  --primitive-aux-clearance-switch-current-body-risk \
  --primitive-aux-clearance-switch-threshold "$PRIMITIVE_AUX_CLEARANCE_SWITCH_THRESHOLD" \
  --primitive-aux-clearance-switch-min-claimed-count "$PRIMITIVE_AUX_CLEARANCE_SWITCH_MIN_CLAIMED_COUNT" \
  --primitive-aux-clearance-switch-latch-ticks "$PRIMITIVE_AUX_CLEARANCE_SWITCH_LATCH_TICKS" \
  --body-clearance-hard-veto-margin "$BODY_CLEARANCE_HARD_VETO_MARGIN" \
  --body-clearance-hard-veto-replacement-cap "$BODY_CLEARANCE_HARD_VETO_REPLACEMENT_CAP" \
  --body-clearance-yaw-contact-veto-prob "$BODY_CLEARANCE_YAW_CONTACT_VETO_PROB" \
  --body-clearance-yaw-direction-veto-prob "$BODY_CLEARANCE_YAW_DIRECTION_VETO_PROB" \
  --body-clearance-aux-veto-prob "$BODY_CLEARANCE_AUX_VETO_PROB" \
  --body-clearance-veto-min-claimed-count "$BODY_CLEARANCE_VETO_MIN_CLAIMED_COUNT" \
  --learned-local-dataset-label-source executed \
  --body-clearance-hard-veto-prob "$BODY_CLEARANCE_HARD_VETO_PROB" \
  --body-clearance-near-yaw-prob-weight "$BODY_CLEARANCE_NEAR_YAW_PROB_WEIGHT" \
  --body-clearance-aux-switch-hard-veto-primitives backward,yaw_left,yaw_right,arc_left,arc_right,hold \
  --body-clearance-aux-switch-enable \
  --body-clearance-aux-switch-ignore-min-area \
  --body-clearance-yaw-always \
  --current-body-risk-clearance-rerank-threshold "$CURRENT_BODY_RISK_CLEARANCE_RERANK_THRESHOLD" \
  --current-body-risk-clearance-rerank-selected-prob-floor "$CURRENT_BODY_RISK_CLEARANCE_RERANK_SELECTED_PROB_FLOOR" \
  --current-body-risk-clearance-rerank-selected-primitives "$CURRENT_BODY_RISK_CLEARANCE_RERANK_SELECTED_PRIMITIVES" \
  --current-body-risk-clearance-rerank-primitives "$CURRENT_BODY_RISK_CLEARANCE_RERANK_PRIMITIVES" \
  --body-clearance-target-area-hard-veto-prob "$BODY_CLEARANCE_TARGET_AREA_HARD_VETO_PROB" \
  --body-clearance-target-area-hard-veto-min-area-logit "$BODY_CLEARANCE_TARGET_AREA_HARD_VETO_MIN_AREA_LOGIT" \
  --current-body-risk-clearance-rerank \
  "${BENCHMARK_EXTRA_ARGS[@]}" \
  --max-ticks "$MAX_TICKS" \
  --output "${OUT_PREFIX}_result.json"

CHECK_ARGS=(
  --result "${OUT_PREFIX}_result.json"
  --max-ticks "$MAX_TICKS"
  --require-scene-id "$SCENE_ID"
  --require-physical-mode
  --require-generalized-runtime-contract
  --require-learned-local-policy-runtime
  --forbid-route-memory
  --max-body-contact-events 0
  --max-body-violations 0
  --max-contact-like-stalls 0
  --max-hard-stalls 0
)

if [[ "$CHECK_RESULT" == "1" || "$CHECK_RESULT" == "true" ]]; then
  "$PYTHON" "$ROOT/scripts/check_go2_fully_learned_demo.py" "${CHECK_ARGS[@]}"
fi

if [[ "$RENDER" == "1" || "$RENDER" == "true" ]]; then
  RENDER_ARGS=()
  if [[ "$REVIEW_UI" == "1" || "$REVIEW_UI" == "true" ]]; then
    RENDER_ARGS+=(--review-ui)
  fi
  "$PYTHON" "$ROOT/scripts/render_go2_closed_loop_result_replay.py" \
    --result "${OUT_PREFIX}_result.json" \
    --demo-video "${OUT_PREFIX}.mp4" \
    --report "${OUT_PREFIX}_render_report.json" \
    --replay-mode physical \
    --capture-rate policy \
    --demo-fps 50 \
    --progress-every "$RENDER_PROGRESS_EVERY" \
    "${RENDER_ARGS[@]}"

  if [[ "$CHECK_RESULT" == "1" || "$CHECK_RESULT" == "true" ]]; then
    "$PYTHON" "$ROOT/scripts/check_go2_fully_learned_demo.py" \
      "${CHECK_ARGS[@]}" \
      --render-report "${OUT_PREFIX}_render_report.json" \
      --require-locomotion-policy-render
  fi
fi

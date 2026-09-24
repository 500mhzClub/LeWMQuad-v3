"""Prospective pure contract for physical graph-edge handoff qualification.

``PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1`` is a development-only,
scene-disjoint physical qualification of the interface between an oracle graph
edge, a body-frame local waypoint target, the frozen current-visual route
ranker, and a fixed twelve-candidate short-prefix action bank.  This module is
declarative: importing it performs no file I/O, simulation, rendering,
encoding, inference, checkpoint loading, or training.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lewm.safety import occluded_goal_topological_belief_v2_contract as V2


class PhysicalGraphEdgeHandoffContractError(ValueError):
    """Raised when prospective physical-handoff authority drifts."""


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
DEVELOPMENT_ONLY = True
SOURCE_PARENT_COMMIT = "3a784118b461d693d5dbf07035b3f85ee1553598"
SOURCE_BASELINE_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
CONTRACT_FREEZE_COMMIT_SUBJECT = "Freeze physical graph edge handoff qualification"
RESULT_COMMIT_SUBJECT = "Evaluate physical graph edge handoff qualification"
OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1"
)

STATE_COUNT = 64
ROLE_IDS = ("DEVELOPMENT", "DEVELOPMENT_HELDOUT")
ROLE_COUNTS = {"DEVELOPMENT": 48, "DEVELOPMENT_HELDOUT": 16}
FAMILY_IDS = (
    "STRAIGHT_PASSAGE",
    "TURNING_JUNCTION",
    "OFFSET_OPENING",
    "ROOM_OR_LOOP_EXIT",
)
STATES_PER_FAMILY = 16
FAMILY_ROLE_COUNTS = {
    family: {"DEVELOPMENT": 12, "DEVELOPMENT_HELDOUT": 4}
    for family in FAMILY_IDS
}
TURNING_JUNCTION_SIDE_COUNTS = {"LEFT": 8, "RIGHT": 8}
SPLIT_ASSIGNMENT_AUTHORITY = {
    "key_projection": (
        "canonical compact JSON without LF of {experiment_id,state_id,family,"
        "candidate_spec_id,canonical_spec_sha256}"
    ),
    "assignment_sha256": "SHA-256(key_projection)",
    "family_order": "ascending (assignment_sha256,state_id)",
    "roles": "family ranks 0..11 DEVELOPMENT; 12..15 DEVELOPMENT_HELDOUT",
    "complete_population_projection": (
        "state_id-sorted rows {state_id,family,candidate_spec_id,assignment_sha256}"
    ),
}

IDENTITY_DOMAIN = "pgehq-v1"
SCENE_ID_PREFIX = "pgehq-v1-scene-"
STATE_ID_PREFIX = "pgehq-v1-state-"
SNAPSHOT_ID_PREFIX = "pgehq-v1-snapshot-"
# A disjoint reserved stream; runtime must additionally compare it to every
# frozen predecessor numeric seed authority before constructing a scene.
PROCEDURAL_SEED_BASE = 5787502096473067520
PROSPECTIVE_POOL_PER_FAMILY = 64
PROSPECTIVE_POOL_COUNT = PROSPECTIVE_POOL_PER_FAMILY * len(FAMILY_IDS)
TEACHER_TRACE_COUNT = PROSPECTIVE_POOL_COUNT
TEACHER_QUALIFICATION_COMPONENT_IDS = (
    "goal_reachable",
    "teacher_trace_contact_free",
    "teacher_left_source_region",
    "teacher_crossed_directed_port",
    "teacher_positive_route_progress",
    "teacher_no_competing_port",
    "teacher_normal_positive",
    "teacher_within_lateral_bounds",
    "teacher_dwell_satisfied",
    "directed_port_defined",
    "current_rgb_valid",
    "graph_edge_physically_executable",
)
TEACHER_QUALIFICATION_REJECTION_RULE = (
    "selected rows use null; qualified nonselected rows use HASH_ORDER_NOT_SELECTED; "
    "unqualified rows use semicolon-joined lexicographically sorted false component IDs"
)
PROCEDURAL_SEED_VALUES = tuple(
    range(PROCEDURAL_SEED_BASE, PROCEDURAL_SEED_BASE + PROSPECTIVE_POOL_COUNT)
)
CONSTRUCTED_SET_CAVEAT = (
    "This is a constructed physical graph-edge handoff qualification set, not "
    "an estimate of natural graph-edge or controller failure prevalence."
)
GEOMETRY_AUTHORITY = {
    "passage_width_m": {"NARROW": 0.75, "WIDE": 1.20},
    "port_distance_m": {"NEAR": 0.22, "FAR": 0.32},
    "spawn_lateral_offset_m": (-0.04, 0.04),
    "spawn_yaw_offset_rad": (-0.12, 0.12),
    "hash_jitter_limit_m": 0.01,
    "source_chamber_selected_frame": {
        "rear_u_m": -1.0,
        "lateral_half_extent_m": 1.30,
        "front_u_is_robot_to_port_distance": True,
    },
    "relative_variant_limits": {
        "port_distance_delta_m": 0.01,
        "spawn_lateral_delta_m": 0.01,
        "spawn_yaw_delta_rad": 0.02,
        "corridor_length_delta_m": 0.08,
        "opening_lateral_delta_m": 0.03,
    },
    "direction_counts": {"STRAIGHT": 24, "LEFT": 20, "RIGHT": 20},
    "family_direction_counts": {
        "STRAIGHT_PASSAGE": {"STRAIGHT": 16},
        "TURNING_JUNCTION": {"LEFT": 8, "RIGHT": 8},
        "OFFSET_OPENING": {"LEFT": 8, "RIGHT": 8},
        "ROOM_OR_LOOP_EXIT": {"STRAIGHT": 8, "LEFT": 4, "RIGHT": 4},
    },
    "shared_visual_material": {
        "floor_rgb": [0.50, 0.50, 0.50],
        "wall_rgb": [0.35, 0.35, 0.35],
        "target_marker_visible": False,
        "family_specific_texture_palette_or_marker": False,
    },
    "camera": {
        "platform_manifest_status": "frozen_platform_manifest_physical_camera",
        "native_resolution_wh": [640, 480],
        "persisted_resolution_hw": [168, 224],
        "relative_position_body_m": [0.326, 0.0, 0.043],
        "relative_rpy_body_rad": [0.0, 0.0, 0.0],
        "horizontal_fov_deg": 78.323,
        "near_m": 0.05,
        "far_m": 200.0,
        "preprocess_to_persisted": (
            "render native 640x480 RGB8 from the frozen platform extrinsics/intrinsics, "
            "then deterministic area resize to width 224 and height 168 without crop"
        ),
    },
    "teacher_only_scan": {
        "prospective_candidates_per_family": PROSPECTIVE_POOL_PER_FAMILY,
        "strata_per_family": 16,
        "variants_per_stratum": 4,
        "selection": (
            "within each family and each of 16 frozen balance strata, choose the "
            "teacher-qualified candidate with the smallest SHA256 of its canonical spec; "
            "fail closed if a stratum has no qualified candidate"
        ),
        "information_allowed": (
            "teacher-only contact-free trace from the prospective decision state, exact "
            "geometry, directed-port existence, reachability, and current RGB validity; "
            "the two-trial serialized-reset fixture is applied only after the 64-state "
            "teacher-qualified panel is frozen and is not a selection input"
        ),
        "information_forbidden": (
            "V-JEPA latent",
            "ranker score",
            "candidate-bank fanout outcome",
            "heldout outcome",
        ),
        "geometry_patch_after_outcome": False,
    },
}

CANDIDATE_IDS = (
    "straight_fast",
    "straight_medium",
    "straight_slow",
    "arc_left",
    "arc_right",
    "turn_left",
    "turn_right",
    "turn_left_then_go",
    "turn_right_then_go",
    "go_then_turn_left",
    "reverse_then_turn",
    "hold",
)
CANDIDATE_PRIMITIVES = {
    "hold": (0.0, 0.0, 0.0),
    "forward_slow": (0.20, 0.0, 0.0),
    "forward_medium": (0.25, 0.0, 0.0),
    "forward_fast": (0.30, 0.0, 0.0),
    "backward": (-0.20, 0.0, 0.0),
    "yaw_left": (0.0, 0.0, 0.45),
    "yaw_right": (0.0, 0.0, -0.45),
    "arc_left": (0.20, 0.0, 0.45),
    "arc_right": (0.20, 0.0, -0.45),
}
CANDIDATE_BANK = (
    ("straight_fast", ("forward_fast",) * 4),
    ("straight_medium", ("forward_medium",) * 4),
    ("straight_slow", ("forward_slow",) * 4),
    ("arc_left", ("arc_left",) * 4),
    ("arc_right", ("arc_right",) * 4),
    ("turn_left", ("yaw_left",) * 4),
    ("turn_right", ("yaw_right",) * 4),
    ("turn_left_then_go", ("yaw_left", "yaw_left", "forward_medium", "forward_medium")),
    ("turn_right_then_go", ("yaw_right", "yaw_right", "forward_medium", "forward_medium")),
    ("go_then_turn_left", ("forward_medium", "forward_medium", "yaw_left", "yaw_left")),
    ("reverse_then_turn", ("backward", "backward", "yaw_left", "yaw_left")),
    ("hold", ("hold",) * 4),
)
COMMAND_TICKS_PER_BLOCK = 5
EXECUTED_BLOCK_COUNT = 3
CANDIDATE_COUNT = len(CANDIDATE_IDS)
BRANCH_COUNT = STATE_COUNT * CANDIDATE_COUNT
HORIZON_TICKS = {"H1": 5, "H2": 10, "H3": 15}
PRIMARY_HORIZON = "H3"
POLICY_STEPS_PER_COMMAND_TICK = 5
PHYSICS_STEPS_PER_POLICY_STEP = 10
PHYSICS_STEPS_PER_COMMAND_TICK = (
    POLICY_STEPS_PER_COMMAND_TICK * PHYSICS_STEPS_PER_POLICY_STEP
)
PHYSICS_STEPS_PER_BRANCH = (
    HORIZON_TICKS[PRIMARY_HORIZON] * PHYSICS_STEPS_PER_COMMAND_TICK
)
PORT_DWELL_COMMAND_TICKS = 2
PORT_DWELL_PHYSICS_SAMPLES = (
    PORT_DWELL_COMMAND_TICKS * PHYSICS_STEPS_PER_COMMAND_TICK
)
RESET_FIXTURE_TRACE_COUNT = STATE_COUNT * 2
RESET_FIXTURE_COMMAND = CANDIDATE_PRIMITIVES["forward_slow"]
RESET_FIXTURE_COMMAND_TICKS = HORIZON_TICKS[PRIMARY_HORIZON]
RESET_FIXTURE_PHYSICS_SAMPLES = (
    RESET_FIXTURE_COMMAND_TICKS * PHYSICS_STEPS_PER_COMMAND_TICK
)

TARGET_IDS = (
    "TARGET_NODE_CENTRE",
    "DIRECTED_EDGE_PORT",
    "ROUTE_LOOKAHEAD",
)
ROUTE_LOOKAHEAD_DISTANCE_M = 0.50
TARGET_DEFINITIONS = {
    "TARGET_NODE_CENTRE": (
        "destination-node centre transformed from frozen world coordinates into "
        "the exact reset robot body frame"
    ),
    "DIRECTED_EDGE_PORT": (
        "first contact-free teacher-trace crossing of the source-node boundary "
        "toward the registered directed edge, transformed into reset body frame"
    ),
    "ROUTE_LOOKAHEAD": (
        "point 0.50 m after the directed port along the registered teacher route, "
        "clipped to the remaining registered route length, transformed into reset body frame"
    ),
}
TARGET_FEATURE_ORDER = (
    "dx_m",
    "dy_m",
    "distance_m",
    "relative_heading_rad",
    "relative_heading_sin",
    "relative_heading_cos",
    "route_intent_dx",
    "route_intent_dy",
)
TARGET_TANGENT_AUDIT_FIELDS = (
    "target_tangent_heading_rad",
    "target_tangent_heading_sin",
    "target_tangent_heading_cos",
)
DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC = (
    {"metric": "selected_correct_edge_execution_rate", "direction": "maximize"},
    {"metric": "correct_edge_top3_rate", "direction": "maximize"},
    {"metric": "normalized_port_regret", "direction": "minimize"},
    {"metric": "mean_selected_port_progress_m", "direction": "maximize"},
    {"metric": "mean_target_transform_error_m", "direction": "minimize"},
    {"metric": "target_id", "direction": "fixed TARGET_IDS order"},
)

HELDOUT_CONDITION_IDS = (
    "DETERMINISTIC_KINEMATICS",
    "FROZEN_CURRENT_VISUAL_RANKER",
    "ORACLE_BEST_ADMISSIBLE_CANDIDATE",
    "TEACHER_TRACE",
)
REPEAT_BRANCH_IDS = (
    "FROZEN_CURRENT_VISUAL_RANKER_SELECTED",
    "ORACLE_BEST_ADMISSIBLE_CANDIDATE_SELECTED",
)
REPEATS_PER_BRANCH = 2
HELDOUT_SCORE_ROW_COUNT = ROLE_COUNTS["DEVELOPMENT_HELDOUT"] * len(
    HELDOUT_CONDITION_IDS
)
REPEATED_EXECUTION_ROW_COUNT = (
    ROLE_COUNTS["DEVELOPMENT_HELDOUT"]
    * len(REPEAT_BRANCH_IDS)
    * REPEATS_PER_BRANCH
)
CANDIDATE_TRACE_COUNT = (
    RESET_FIXTURE_TRACE_COUNT + BRANCH_COUNT + REPEATED_EXECUTION_ROW_COUNT
)
TRACE_INDEX_RANGES = {
    "reset_fixture": [0, RESET_FIXTURE_TRACE_COUNT],
    "candidate_fanout": [
        RESET_FIXTURE_TRACE_COUNT,
        RESET_FIXTURE_TRACE_COUNT + BRANCH_COUNT,
    ],
    "repeated_execution": [
        RESET_FIXTURE_TRACE_COUNT + BRANCH_COUNT,
        CANDIDATE_TRACE_COUNT,
    ],
}

# All equality is exact unless one of these frozen numerical tolerances is
# explicitly named.  Tolerances cover floating-point SE(2) reconstruction and
# repeat-state comparison only; no metric/gate tolerance exists.
NUMERICAL_TOLERANCES = {
    "se2_position_m": 1.0e-6,
    "se2_heading_rad": 1.0e-6,
    "route_arclength_m": 1.0e-6,
    "inverse_transform_position_m": 1.0e-6,
    "inverse_transform_heading_rad": 1.0e-6,
    "repeat_endpoint_position_m": 1.0e-4,
    "repeat_endpoint_heading_rad": 1.0e-4,
    "repeat_base_twist": 1.0e-6,
    "repeat_joint_position_rad": 1.0e-6,
    "repeat_joint_velocity_rad_s": 1.0e-6,
    "reset_base_position_m": 1.0e-6,
    "reset_base_quaternion_component": 1.0e-7,
    "reset_base_twist": 1.0e-6,
    "reset_joint_position_rad": 1.0e-6,
    "reset_joint_velocity_rad_s": 1.0e-6,
    "probability_or_rate": 0.0,
    "ranking_score_tie": 0.0,
    "trace_summary_float": 1.0e-9,
    "division_floor": 1.0e-12,
}
RESET_TRACE_PAIR_COMPARISON_AUTHORITY = {
    "physics_samples_per_trial": RESET_FIXTURE_PHYSICS_SAMPLES,
    "completion_termination_reason": "H3_COMPLETE",
    "samplewise_tolerances": {
        "base_pose_world_position_xyz": NUMERICAL_TOLERANCES["reset_base_position_m"],
        "base_pose_world_quaternion_xyzw": NUMERICAL_TOLERANCES["reset_base_quaternion_component"],
        "base_twist_world": NUMERICAL_TOLERANCES["reset_base_twist"],
        "joint_position": NUMERICAL_TOLERANCES["reset_joint_position_rad"],
        "joint_velocity": NUMERICAL_TOLERANCES["reset_joint_velocity_rad_s"],
    },
    "exact_members": (
        "timestamp_s", "requested_command", "post_slew_applied_command",
        "physics_contact", "source_region_member", "correct_edge_region_member",
        "wrong_edge_region_member", "target_region_member",
    ),
    "endpoint_position_tolerance_m": NUMERICAL_TOLERANCES["repeat_endpoint_position_m"],
    "endpoint_heading_tolerance_rad": NUMERICAL_TOLERANCES["repeat_endpoint_heading_rad"],
    "termination_reason_exact": True,
    "stuck_exact": True,
    "source": "independent direct comparison of paired candidate_traces.npz slices",
}
PHYSICAL_OUTCOME_AUTHORITY = {
    "port_progress_m": (
        "Euclidean reset-body distance from H0 to the directed port minus the minimum "
        "Euclidean distance to that port across all 750 physics samples"
    ),
    "lateral_error_m": (
        "absolute signed cross-track distance from the H3 base position to the directed "
        "port tangent line"
    ),
    "angular_error_rad": (
        "absolute wrapped H3 base-yaw difference from the directed port tangent heading"
    ),
    "entered_correct_edge": (
        "within H3 interpolate the first selected transverse-port crossing; directed-normal "
        "dot displacement >0; crossing lies within opening lateral bounds; no competing port "
        "crossing occurs first; remain beyond the port for at least 100 consecutive physics "
        "samples (two command ticks) unless the target-node region is entered earlier"
    ),
    "entered_wrong_edge": (
        "interpolate the first crossing of any registered competing transverse port; "
        "require positive competing-port outward-normal displacement and crossing within "
        "that port's lateral bounds; it must occur before the selected-port crossing"
    ),
    "no_edge": "neither entered_correct_edge nor entered_wrong_edge",
    "stuck": {
        "command_activity_threshold": 0.05,
        "h3_translation_threshold_m": 0.02,
        "h3_heading_threshold_rad": 0.05,
        "formula": (
            "non-hold branch with maximum requested planar-speed/yaw magnitude >0.05, "
            "H3 translation <0.02 m, and absolute H3 heading change <0.05 rad"
        ),
    },
    "successor_viable": {
        "finite_solver_state": True,
        "minimum_base_height_m": 0.20,
        "maximum_absolute_roll_rad": 0.70,
        "maximum_absolute_pitch_rad": 0.70,
        "disallowed_contact_absent_at_h3": True,
    },
    "oracle_admissible": (
        "no disallowed physics-rate contact, finite successor, successor_viable, and not stuck"
    ),
    "positive_port_progress": "port_progress_m > 0 with no tolerance",
}
COMMAND_TRACKING_AUTHORITY = {
    "command_ticks": HORIZON_TICKS[PRIMARY_HORIZON],
    "physics_samples_per_command_tick": PHYSICS_STEPS_PER_COMMAND_TICK,
    "discard_initial_physics_samples_per_command_tick": 20,
    "averaged_physics_samples_per_command_tick": 30,
    "samples": (
        "for each of 15 command ticks, discard the first 20 of 50 physics samples "
        "as a fixed 0.04 s transient; average body-frame vx, vy, and yaw rate over "
        "the remaining 30 samples"
    ),
    "active_command_threshold": 0.05,
    "vx_mae_maximum_mps": 0.15,
    "vy_absolute_mean_maximum_mps": 0.10,
    "yaw_rate_mae_maximum_rad_s": 0.30,
    "commanded_sign_agreement_minimum": 0.80,
    "pass": (
        "all three MAE/absolute-mean limits pass and sign agreement over active vx/yaw "
        "components is at least 0.80"
    ),
}

METRIC_FORMULAS = {
    "candidate_score_ranking": (
        "descending finite score; exact score ties use lower candidate_index"
    ),
    "correct_edge_candidate": (
        "oracle_admissible and entered_correct_edge and successor_viable and "
        "positive_port_progress and not physics_contact and not stuck"
    ),
    "coverage_rate": (
        "heldout states with at least one correct_edge_candidate / 16"
    ),
    "correct_edge_top1_rate": (
        "rows whose selected candidate is a correct_edge_candidate / evaluated rows"
    ),
    "correct_edge_top3_rate": (
        "rows with any correct_edge_candidate in the first three ranked candidates / evaluated rows"
    ),
    "correct_edge_mrr": (
        "mean reciprocal one-based rank of the first correct_edge_candidate; zero when absent"
    ),
    "pairwise_correct_edge_ordering": (
        "all correct-edge-candidate versus non-correct-candidate score pairs; one credit "
        "when the correct score is higher, half credit for an exact tie, zero otherwise; "
        "zero when no such pair exists; TEACHER_TRACE is not applicable"
    ),
    "selected_correct_edge_execution_rate": (
        "rows whose physically executed selected branch entered the registered directed edge / evaluated rows"
    ),
    "normalized_port_regret": (
        "per state: (best admissible port progress - selected port progress) / "
        "max(best admissible port progress - minimum admissible port progress, 1e-12), "
        "clipped to [0,1]; zero for a degenerate admissible range; mean over rows"
    ),
    "repeatability_rate": (
        "repeat rows with byte-identical applied-command sequence, identical correct-edge "
        "label, contact and stuck outcomes, and endpoint within frozen position/heading "
        "tolerances / 64"
    ),
    "teacher_execution_rate": (
        "heldout teacher traces that are contact-free, cross the registered directed port "
        "in source-to-target direction, leave the source, make positive registered-route "
        "progress, and enter no competing port / 16; target-node entry is descriptive only"
    ),
    "teacher_route_progress_m": (
        "initial base-xy Euclidean distance to the selected opening midpoint minus the "
        "minimum distance over all retained teacher physics samples; positive iff >0"
    ),
    "family_correct_execution": (
        "number of heldout selected-ranker rows with correct-edge execution in each family"
    ),
}

HANDOFF_GATE = {
    "teacher_correct_execution_count": 16,
    "coverage_rate_minimum": 0.90,
    "ranker_correct_edge_top1_rate_minimum": 0.70,
    "ranker_correct_edge_top3_rate_minimum": 0.90,
    "ranker_selected_correct_edge_execution_rate_minimum": 0.75,
    "ranker_normalized_port_regret_maximum": 0.25,
    "repeatability_rate_minimum": 0.95,
    "minimum_correct_execution_per_family": 1,
    "command_tracking_pass_required": True,
}

# Materiality is deliberately coarse and prospective.  A target is not called
# better due only to a sub-threshold numerical fluctuation.
TARGET_MATERIALITY = {
    "selected_correct_edge_execution_rate_improvement": 0.10,
    "correct_edge_top3_rate_improvement": 0.10,
    "normalized_port_regret_reduction": 0.10,
    "selected_port_progress_m_improvement": 0.10,
    "rule": (
        "material if execution improves by >=0.10, or top3 improves by >=0.10, "
        "or regret falls by >=0.10, or selected progress improves by >=0.10 m"
    ),
}

PRIMARY_CLASSIFICATIONS = (
    "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL",
    "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT",
    "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO",
    "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO",
    "LOW_LEVEL_PREFIX_EXECUTION_NO_GO",
    "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO",
)
SECONDARY_CLASSIFICATIONS = (
    "NODE_CENTRE_TARGET_OUT_OF_DISTRIBUTION",
    "DIRECTED_PORT_TARGET_SIGNAL",
    "ROUTE_LOOKAHEAD_TARGET_SIGNAL",
    "CANDIDATE_BANK_DIRECTIONAL_GAP",
    "LEFT_RIGHT_ASYMMETRY",
    "PORT_DISTANCE_LIMITATION",
    "RANKER_TARGET_DISTRIBUTION_SHIFT",
    "CONTROLLER_TRACKING_LIMITATION",
)
SECONDARY_PREDICATES = {
    "NODE_CENTRE_TARGET_OUT_OF_DISTRIBUTION": (
        "one or more TARGET_NODE_CENTRE rows lies outside every exact interval in "
        "ORIGINAL_RANKER_TRAINING_SUPPORT; report outside fraction"
    ),
    "DIRECTED_PORT_TARGET_SIGNAL": (
        "DIRECTED_EDGE_PORT materially outperforms TARGET_NODE_CENTRE under TARGET_MATERIALITY"
    ),
    "ROUTE_LOOKAHEAD_TARGET_SIGNAL": (
        "ROUTE_LOOKAHEAD materially outperforms TARGET_NODE_CENTRE under TARGET_MATERIALITY"
    ),
    "CANDIDATE_BANK_DIRECTIONAL_GAP": (
        "any of STRAIGHT, LEFT, RIGHT heldout correct-edge candidate coverage is <0.90"
    ),
    "LEFT_RIGHT_ASYMMETRY": (
        "absolute LEFT minus RIGHT heldout ranker execution-rate or coverage-rate gap is >=0.20"
    ),
    "PORT_DISTANCE_LIMITATION": (
        "FAR heldout ranker execution or coverage is at least 0.20 below NEAR"
    ),
    "RANKER_TARGET_DISTRIBUTION_SHIFT": (
        "the selected target has any row outside original ranker support and frozen-ranker "
        "execution is below oracle execution"
    ),
    "CONTROLLER_TRACKING_LIMITATION": (
        "repeatability <0.95 or registered command-tracking predicate fails"
    ),
}
COMPONENT_PRECEDENCE = (
    "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT",
    "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO",
    "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO",
    "LOW_LEVEL_PREFIX_EXECUTION_NO_GO",
)
CLASSIFICATION_PRECEDENCE = (
    "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO",
    "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT",
    "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL",
    "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO",
    "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO",
    "LOW_LEVEL_PREFIX_EXECUTION_NO_GO",
)
NEXT_DECISION_BY_CLASSIFICATION = {
    "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL": (
        "OCCLUDED_GOAL_MEMORY_PHYSICAL_INTEGRATION_V1"
    ),
    "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT": (
        "OCCLUDED_GOAL_MEMORY_PHYSICAL_INTEGRATION_V1"
    ),
    "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO": (
        "GRAPH_EDGE_LOCAL_ACTION_BANK_SUCCESSOR_V1"
    ),
    "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO": (
        "GRAPH_EDGE_LOCAL_WAYPOINT_RANKER_V1"
    ),
    "LOW_LEVEL_PREFIX_EXECUTION_NO_GO": (
        "GRAPH_EDGE_PREFIX_EXECUTION_INTERFACE_V1"
    ),
    "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO": (
        "ADDRESS_EARLIEST_FAILING_PHYSICAL_INTERFACE_FIRST"
    ),
}
COMPOSITE_NEXT_BY_EARLIEST_COMPONENT = {
    "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT": "OCCLUDED_GOAL_MEMORY_PHYSICAL_INTEGRATION_V1",
    "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO": "GRAPH_EDGE_LOCAL_ACTION_BANK_SUCCESSOR_V1",
    "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO": "GRAPH_EDGE_LOCAL_WAYPOINT_RANKER_V1",
    "LOW_LEVEL_PREFIX_EXECUTION_NO_GO": "GRAPH_EDGE_PREFIX_EXECUTION_INTERFACE_V1",
}

VJEPA_ENCODER_BINDING = copy.deepcopy(V2.VJEPA_ENCODER_BINDING)
CURRENT_VISUAL_RANKER_BINDING = copy.deepcopy(V2.STAGE_B_CURRENT_VISUAL_BINDING)
assert VJEPA_ENCODER_BINDING["checkpoint_sha256"] == (
    "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
)
assert CURRENT_VISUAL_RANKER_BINDING["checkpoint_sha256"] == (
    "e1a2a58ff527b4d2bc210f6b1c8fd1d00d87873691db64e8c3ccb2257fa6c127"
)

V2_CONTEXT_BINDING = {
    "experiment_id": V2.EXPERIMENT_ID,
    "source_freeze_commit": "de2e320617bdf47d78e5c142bc3bd3e1faf80d80",
    "result_commit": SOURCE_PARENT_COMMIT,
    "result_classifications": {
        "stage_a": "TOPOLOGICAL_MAP_SUFFICIENT",
        "stage_b": "LOCAL_EXECUTION_INTERFACE_NO_GO",
    },
    "stage_b_baseline": {
        "map_graph_edge_correct": [256, 256],
        "oracle_graph_edge_correct": [256, 256],
        "map_local_port_correct": [13, 256],
        "oracle_local_port_correct": [13, 256],
        "map_goals_reached": [0, 16],
        "oracle_goals_reached": [0, 16],
    },
    "bindings": {
        "panel_manifest.json": {
            "bytes": 38219,
            "sha256": "e348a83ec605256d93f6b6aeb09837919e927b61f95c46a60edf0bb958afe844",
        },
        "graph_manifest.json": {
            "bytes": 31169157,
            "sha256": "34902c7b25cf1a68076931da85811b9f2a89ff89c3622c06a28f8bd9b13aee09",
        },
        "query_ledger.jsonl": {
            "bytes": 15307920,
            "sha256": "b96576986416edcbadb053aedbf0bbd07c7e16226a1ce1c09cebd5ffe9feb3df",
        },
        "stage_b_trace.jsonl": {
            "bytes": 105604758,
            "sha256": "215662301b415dc8e5d6938f8a8a1d10aa83b8f0fc5f52465fc6a9c71e0e63e1",
        },
        "stage_b_metrics.json": {
            "bytes": 51952,
            "sha256": "30b15465ad25072f6f4cff98b30b3b3335ac97634ce66436427f3e598660952d",
        },
        "result.json": {
            "bytes": 223871,
            "sha256": "0114bbae2ff0f6e0c9e7338e988b82a610848cf96fc948369a898bf55f367192",
        },
        "result.md": {
            "bytes": 6333,
            "sha256": "f58dcd1776c8757971ec9c168b51cb31ce4317c6f05c5b3e5f3fb8da9c6047c0",
        },
        "file_hashes.json": {
            "bytes": 3199,
            "sha256": "b1b3780abda794b4b2250ce226fb0f0f538d2f36353b2043d6bec7f544f28f9c",
        },
    },
            "scientific_interpretation": (
        "current-frame nearest-node retrieval was insufficient; accumulated persistent "
        "V-JEPA observation likelihood supported MAP place belief, but the V2 Stage-B "
        "local handoff used symbolic graph-port outcomes and is not physical candidate-bank evidence"
    ),
    "runtime_reuse_authorized": False,
    "source_and_frozen_model_design_authority_only": True,
}

PRIOR_SCENE_EXCLUSION_AUTHORITY = {
    "inherited_authority": copy.deepcopy(V2.PRIOR_PANEL_EXCLUSION_AUTHORITY),
    "v2_panel_binding": {
        "path": str(V2.OUTPUT_ROOT / "panel_manifest.json"),
        **copy.deepcopy(V2_CONTEXT_BINDING["bindings"]["panel_manifest.json"]),
    },
    "broad_prior_plus_v2_projection": {
        "scene_identity_count": 1391,
        "scene_identity_canonical_json_sha256": "32716a7c25791b67885e0660ed948ed4b3c5fa61495dda712fcb4af23c0e3b35",
        "scene_identity_sha256_count": 1583,
        "scene_identity_sha256_canonical_json_sha256": "ba6b05288b523baeb46a90e3fa9eaa4104957d3a511c0ccb1da615c1a55b2b33",
        "episode_state_or_graph_identity_count": 492,
        "episode_state_or_graph_identity_canonical_json_sha256": "1862748709d3aa9b7df15a23eff3d070cf1ae158b1c84dded475e7286189afb8",
        "numeric_seed_count": 284,
        "numeric_seed_canonical_json_sha256": "644f6e66db0343aacf58418337d23beafd52181c633078602e62696b09ff86e5",
        "textual_path_geometry_or_source_identity_count": 131860,
        "textual_path_geometry_or_source_identity_canonical_json_sha256": "1c486aa5039d8f2f725680db517e1b95b0d3814137a2b511ae6dbaca3c0cf177",
        "structured_waypoint_or_sequence_path_count": 40,
        "structured_waypoint_or_sequence_path_canonical_json_sha256": "a9d63c5c7a8aae8c93c27c5d08c5e306779dccd790f8c2173df3a72729f77cf3",
    },
    "new_identity_domain": IDENTITY_DOMAIN,
    "new_scene_prefix": SCENE_ID_PREFIX,
    "new_state_prefix": STATE_ID_PREFIX,
    "new_seed_minimum": PROCEDURAL_SEED_VALUES[0],
    "new_seed_maximum": PROCEDURAL_SEED_VALUES[-1],
    "required_comparisons": (
        "scene_id exact set intersection",
        "SHA256(scene_id) against predecessor scene hashes",
        "state/episode identity exact set intersection",
        "numeric seed exact set intersection",
        "geometry/source path exact set intersection where registered",
        "structured waypoint, route, or sequence path exact set intersection",
    ),
    "all_overlap_counts_required_zero": True,
}

CANONICAL_ENCODING_AUTHORITY = {
    "pixel_identity": (
        "SHA-256(canonical compact JSON {shape,dtype,layout} without LF || NUL || "
        "C-contiguous RGB bytes)"
    ),
    "pixel_order": "unique pixel_sha256 lexicographic",
    "encoder": copy.deepcopy(VJEPA_ENCODER_BINDING),
    "external_encoder_source": {
        "repository_path": (
            "/home/andrewknowles/.cache/vjepa2-"
            + VJEPA_ENCODER_BINDING["external_repository_commit"]
        ),
        "commit": VJEPA_ENCODER_BINDING["external_repository_commit"],
        "ordinary_directory_not_symlink": True,
        "worktree_clean_including_untracked": True,
        "verification": "git rev-parse HEAD plus git status --porcelain=v1 --untracked-files=all",
    },
    "preprocessing": {
        "input_shape": [168, 224, 3],
        "input_dtype": "uint8",
        "input_layout": "C_CONTIGUOUS_RGB",
        "transport": (
            "inside a fresh TemporaryDirectory, write canonical-frame.png with "
            "PIL.Image.fromarray(C-contiguous RGB uint8, mode='RGB').save(..., format='PNG'); "
            "call VJepa21Arm.preprocess(str(path)); preprocess_array is not an available API"
        ),
        "temporary_png_retained": False,
        "output_type": "torch.Tensor",
        "output_shape": [3, 384, 512],
        "output_dtype": "float32",
        "finite_required": True,
        "output_projection": "detach, CPU, C-contiguous before canonical array hashing",
        "preprocessed_tensor_sha256_persisted_per_canonical_pixel": True,
    },
    "batch_size": 1,
    "raw_token_shape": [768, 1024],
    "raw_token_dtype": "float16",
    "descriptor_shape": [768, 1024],
    "descriptor_dtype": "float32",
    "spatial_grid": [24, 32],
    "duplicate_pixel_policy": "encode once and map every occurrence to the canonical row",
    "tolerance": 0,
}

RANKER_INPUT_AUTHORITY = {
    "checkpoint": copy.deepcopy(CURRENT_VISUAL_RANKER_BINDING),
    "scores": "all twelve candidates in exact CANDIDATE_IDS order",
    "target_features_persisted": list(TARGET_FEATURE_ORDER),
    "ranker_goal_projection": (
        "[dx_m,dy_m,sin(atan2(dy_m,dx_m)),cos(atan2(dy_m,dx_m))] in the checkpoint's "
        "frozen four-dimensional goal slice; relative_heading is the target bearing. "
        "Target/port tangent heading, distance, heading radians, and route-intent are "
        "audit evidence and do not add checkpoint inputs"
    ),
    "visual": "one canonical current-state V-JEPA spatial token grid",
    "candidate_plan": (
        "exact requested blocks and deterministic post-slew applied command plan "
        "for the first three five-tick blocks"
    ),
    "state_features": (
        "snapshot previous applied command and frozen control history only"
    ),
    "frozen_feature_widths": {"base_input": 131, "goal_action_query": 66},
    "frozen_input_builder_source_sha256": CURRENT_VISUAL_RANKER_BINDING["model_source_sha256"],
    "deterministic_kinematics_score": (
        "start target distance minus deterministic H3 endpoint target distance, "
        "computed from the frozen requested/post-slew plan without physics outcomes"
    ),
    "forbidden_inputs": (
        "candidate physical outcome",
        "contact",
        "stuck",
        "successor viability",
        "correct edge label",
        "teacher trace",
        "oracle admissibility",
    ),
    "tie_break": "lower candidate_index",
}
HELDOUT_SELECTOR_AUTHORITY = {
    "DETERMINISTIC_KINEMATICS": {
        "integration": (
            "SE(2) forward Euler at 0.10 s for each of 15 exact post-slew command ticks: "
            "x+=dt*vx*cos(yaw), y+=dt*vx*sin(yaw), yaw=wrap(yaw+dt*yaw_rate); vy=0"
        ),
        "score": "start target distance minus deterministic H3 endpoint target distance",
        "tie_break": "lower candidate_index",
        "physics_outcomes_used": False,
    },
    "FROZEN_CURRENT_VISUAL_RANKER": copy.deepcopy(RANKER_INPUT_AUTHORITY),
    "ORACLE_BEST_ADMISSIBLE_CANDIDATE": {
        "lexicographic": (
            "eligible correct-edge candidate first",
            "maximize port_progress_m",
            "minimize lateral_error_m",
            "minimize angular_error_rad",
            "lower candidate_index",
        ),
        "tie_tolerance": 0,
    },
    "TEACHER_TRACE": {
        "candidate_bank_member": False,
        "candidate_scores": None,
        "candidate_ranking": None,
        "selected_candidate_index": None,
        "physical_outcome": "bound teacher trace only",
    },
}
ORIGINAL_RANKER_TRAINING_SUPPORT = {
    "panel_path": str(
        Path("/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/")
        / "non_greedy_local_subgoal_jepa_planning_v1/panel_manifest.json"
    ),
    "panel_bytes": 249155,
    "panel_sha256": "e464f8df0e651bde541bece687d1b61b8baf4daa246188e9e16dca622d0bfde3",
    "role": "FIT",
    "state_count": 64,
    "derivation": (
        "transform final-goal minus start position by inverse start yaw for every FIT "
        "state in the frozen ranker panel"
    ),
    "body_dx_m": [0.0, 0.0],
    "body_dy_m": [-2.66, -2.66],
    "distance_m": [2.66, 2.66],
    "relative_heading_rad": [-math.pi / 2.0, -math.pi / 2.0],
    "relative_heading_sin": [-1.0, -1.0],
    "relative_heading_cos": [0.0, 0.0],
    "support_tolerance": 1.0e-9,
}
RANKER_CONTRACT_DIFFERENCE_INVENTORY = (
    "original goal role was a shared final-goal body vector [dx=0,dy=-2.66,heading=-pi/2]; qualification roles are local node-centre, directed-port, or route-lookahead targets",
    "original previous applied command and every control-history value were zero; qualification uses values captured in each exact physical snapshot",
    "original current RGB was produced by the analytic neutral 224x168 renderer at horizontal FOV 92 degrees; qualification RGB is a scene-disjoint Genesis capture using the platform mount [0.326,0,0.043], zero RPY, FOV 78.323 degrees, native 640x480, then deterministic area resize to 224x168",
    "V-JEPA preprocessing, 24x32 token order, and aligned 768x1024 current-token representation remain frozen",
    "the exact 12-candidate order, requested plan, post-slew plan, nominal endpoint features, 131-dimensional base input order, and 66-dimensional query input order remain frozen",
    "new graph-edge target semantics and physical-image domain are distribution shifts and are not asserted equivalent to the original ranker training support",
    "route-intent fields are persisted for audit only and are not appended to the frozen checkpoint input",
)

RESET_RUNTIME_BINDING = {
    "source_path": "scripts/run_go2_oracle_branch_pilot_v1.py",
    "source_sha256": "19cf29132d4d9c3c8a6f3630bc243caa62a27e5f3616a5598fa487775ef10955",
    "capture_function": "capture_branch_state",
    "restore_function": "restore_branch_state",
    "restore_api_scope": (
        "reviewed private capture_branch_state/restore_branch_state authority; the public "
        "RolloutRunner API has no snapshot/restore fallback and must not be substituted"
    ),
    "post_restore_state_sha256_semantics": (
        "descriptive digest of the immediate numeric state projection; it is not the "
        "serialized snapshot byte digest and reset pass/fail is determined by direct raw "
        "trace comparison under RESET_TRACE_PAIR_COMPARISON_AUTHORITY"
    ),
    "runner_extension_required": (
        "capture and restore torch CPU RNG plus every CUDA/ROCm device RNG state and "
        "device count; the historical helper's CPU-only torch RNG is insufficient"
    ),
    "runtime_compatibility": {
        "genesis_version": "0.4.6",
        "solver_runtime_package": "quadrants",
        "solver_runtime_version": "0.6.2",
        "historical_optional_module_absent": "gstaichi",
        "historical_collect_solver_fields_compatible": False,
        "compatibility_source_path": (
            "scripts/run_physical_graph_edge_handoff_qualification_v1.py"
        ),
        "compatibility_function": "_collect_solver_fields_compat",
        "compatibility_algorithm": (
            "the historical stable sorted __dict__ walk at depth four over scene and "
            "active-solver roots, preserving the same two static-geometry markers; accept "
            "quadrants.Field and quadrants.Ndarray values only when to_numpy/from_numpy "
            "are available, and require a nonempty unique field inventory"
        ),
        "replaced_historical_function_only": "collect_solver_fields",
        "retained_pilot_functions": (
            "capture_branch_state", "restore_branch_state", "dump_branch_state",
            "load_branch_state", "assert_branch_boundary",
        ),
        "required_resolution": (
            "use the tracked experiment runner's explicit Quadrants-compatible deterministic "
            "solver-field walker for solver capture/restore inventory while retaining the "
            "reviewed capture_branch_state/restore_branch_state state-domain contract"
        ),
        "outcome_use_before_resolution": False,
    },
    "required_state_domains": (
        "Genesis solver state and scene step index",
        "policy last action and controller/harness arrays and objects",
        "previous applied command and command/control history",
        "Python, NumPy, runner, spawn, and torch RNG state",
        "torch CPU RNG plus every CUDA/ROCm device RNG state and device count",
        "episode counters, reset counters, goal, identity, and canonical boundary",
    ),
    "graph_free_spawn_reset_compatibility": {
        "source_path": "scripts/run_physical_graph_edge_handoff_qualification_v1.py",
        "function": "_reset_robot_to_fixed_spawn_compat",
        "replaced_function_only": "RolloutRunner._reset_robot_to_spawn",
        "reason": (
            "the production route collector requires a planning grid, while the frozen "
            "physical qualification ScenePack deliberately has no routing manifest"
        ),
        "preconditions": "exactly one nonrandom environment and the frozen ScenePack spawn",
        "initialization": (
            "set the exact pack xyz/quaternion and production policy reset stance, zero joint "
            "velocities, call policy.reset([0]), and zero _last_executed plus block, tip, and "
            "recovery counters; do not construct a collector, planning grid, or manifest"
        ),
        "settling": {
            "function": "begin_and_settle",
            "command": "hold",
            "command_ticks": HORIZON_TICKS[PRIMARY_HORIZON],
            "policy_steps_per_command_tick": POLICY_STEPS_PER_COMMAND_TICK,
            "physics_steps_per_policy_step": PHYSICS_STEPS_PER_POLICY_STEP,
            "physics_samples": PHYSICS_STEPS_PER_BRANCH,
        },
        "outcome_dependent_tuning": False,
    },
}
CONTACT_AUTHORITY = {
    "ontology_path": "lewm/safety/contact_hazard_ontology_v1.py",
    "ontology_sha256": "69550fe787e84331560013678abed3aab58f573719b7198c53bfef786fb6204b",
    "sampling": "after every 2 ms scene.step() physics step",
    "api": "robot.get_contacts(exclude_self_contact=False) or safe equivalent manifold API",
    "robot_environment_pairs_only": True,
    "exclude_robot_self_contact": True,
    "exclude_permitted_calf_or_foot_ground_support": True,
    "force_threshold_n": 1.0e-3,
    "classification": "contact_hazard_ontology_v1.is_disallowed_contact",
    "forbidden_api": "get_links_net_contact_force",
    "scientific_role": "descriptive simulated contact proxy; not a deployment-safety label",
}
TEACHER_CONTROLLER_AUTHORITY = {
    "controller": "deterministic pure pursuit over registered teacher_route_polyline_world",
    "command_tick_s": 0.10,
    "lookahead_distance_m": 0.18,
    "waypoint_advance_radius_m": 0.08,
    "linear_gain": 0.25,
    "minimum_forward_command_mps": 0.08,
    "maximum_forward_command_mps": 0.25,
    "yaw_gain": 1.50,
    "maximum_absolute_yaw_rate_rad_s": 0.45,
    "turn_in_place_heading_error_rad": 1.20,
    "maximum_command_ticks": 80,
    "formula": (
        "choose the point 0.18 m ahead along the registered polyline; wrapped heading "
        "error e; yaw=clip(1.5*e,-0.45,0.45); vx=0 when |e|>=1.20 else "
        "clip(0.25*cos(e),0.08,0.25); vy=0; advance a vertex within 0.08 m"
    ),
    "success_stop": (
        "stop after source leave, selected-port crossing, strictly positive route progress, "
        "no competing-port entry, no disallowed contact, and 100 consecutive physics "
        "samples beyond the port unless the target-node region is entered earlier"
    ),
    "target_node_entry": "descriptive_only",
    "outcome_dependent_tuning": False,
}

NPZ_PAYLOAD_AUTHORITY = {
    "state_snapshots.npz": {
        "snapshot_payload_bytes": {
            "descr": "|u1", "shape": ["B"], "hash_mode": "offset_slices",
            "offsets_member": "snapshot_offsets",
            "slice_digest_domain": "sha256_of_exact_serialized_snapshot_bytes",
        },
        "snapshot_offsets": {"descr": "<i8", "shape": [STATE_COUNT + 1], "hash_mode": "whole"},
        "base_pose_world": {"descr": "<f8", "shape": [STATE_COUNT, 7], "hash_mode": "rows_axis0"},
        "base_twist_world": {"descr": "<f8", "shape": [STATE_COUNT, 6], "hash_mode": "rows_axis0"},
        "joint_position": {"descr": "<f8", "shape": [STATE_COUNT, 12], "hash_mode": "rows_axis0"},
        "joint_velocity": {"descr": "<f8", "shape": [STATE_COUNT, 12], "hash_mode": "rows_axis0"},
        "controller_observation": {"descr": "<f8", "shape": [STATE_COUNT, 45], "hash_mode": "rows_axis0"},
        "policy_last_action": {"descr": "<f8", "shape": [STATE_COUNT, 12], "hash_mode": "rows_axis0"},
        "previous_policy_action": {"descr": "<f8", "shape": [STATE_COUNT, 12], "hash_mode": "rows_axis0"},
        "previous_applied_command": {"descr": "<f8", "shape": [STATE_COUNT, 3], "hash_mode": "rows_axis0"},
        "command_history": {"descr": "<f8", "shape": [STATE_COUNT, 15, 3], "hash_mode": "rows_axis0"},
        "control_history": {"descr": "<f8", "shape": [STATE_COUNT, 15, 2], "hash_mode": "rows_axis0"},
        "low_level_policy_state": {"descr": "<f8", "shape": [STATE_COUNT, 12], "hash_mode": "rows_axis0"},
        "camera_world_transform": {"descr": "<f8", "shape": [STATE_COUNT, 4, 4], "hash_mode": "rows_axis0"},
    },
    "teacher_traces.npz": {
        "trace_offsets": {"descr": "<i8", "shape": [TEACHER_TRACE_COUNT + 1], "hash_mode": "whole"},
        "timestamp_s": {"descr": "<f8", "shape": ["T"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "base_pose_world": {"descr": "<f8", "shape": ["T", 7], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "base_twist_world": {"descr": "<f8", "shape": ["T", 6], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "joint_position": {"descr": "<f8", "shape": ["T", 12], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "joint_velocity": {"descr": "<f8", "shape": ["T", 12], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "applied_command": {"descr": "<f8", "shape": ["T", 3], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "requested_command": {"descr": "<f8", "shape": ["T", 3], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "physics_contact": {"descr": "|u1", "shape": ["T"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "source_region_member": {"descr": "|u1", "shape": ["T"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "edge_region_member": {"descr": "|u1", "shape": ["T"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "target_region_member": {"descr": "|u1", "shape": ["T"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    },
    "rgb_observations.npz": {
        "rgb": {"descr": "|u1", "shape": [STATE_COUNT, 168, 224, 3], "hash_mode": "rows_axis0"},
    },
    "canonical_latents.npz": {
        "raw_tokens": {"descr": "<f2", "shape": ["U", 768, 1024], "hash_mode": "rows_axis0"},
        "spatial_descriptors": {"descr": "<f4", "shape": ["U", 768, 1024], "hash_mode": "rows_axis0"},
    },
    "candidate_traces.npz": {
        "trace_offsets": {"descr": "<i8", "shape": [CANDIDATE_TRACE_COUNT + 1], "hash_mode": "whole"},
        "timestamp_s": {"descr": "<f8", "shape": ["P"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "base_pose_world": {"descr": "<f8", "shape": ["P", 7], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "base_twist_world": {"descr": "<f8", "shape": ["P", 6], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "joint_position": {"descr": "<f8", "shape": ["P", 12], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "joint_velocity": {"descr": "<f8", "shape": ["P", 12], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "requested_command": {"descr": "<f8", "shape": ["P", 3], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "post_slew_applied_command": {"descr": "<f8", "shape": ["P", 3], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "physics_contact": {"descr": "|u1", "shape": ["P"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "source_region_member": {"descr": "|u1", "shape": ["P"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "correct_edge_region_member": {"descr": "|u1", "shape": ["P"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "wrong_edge_region_member": {"descr": "|u1", "shape": ["P"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
        "target_region_member": {"descr": "|u1", "shape": ["P"], "hash_mode": "offset_slices", "offsets_member": "trace_offsets"},
    },
}
_DESCR_TO_DIGEST_DTYPE = {
    "|u1": "uint8",
    "<i8": "int64",
    "<f8": "float64",
    "<f4": "float32",
    "<f2": "float16",
}
for _members in NPZ_PAYLOAD_AUTHORITY.values():
    for _member in _members.values():
        _member["digest_dtype"] = _DESCR_TO_DIGEST_DTYPE[_member["descr"]]

RUNTIME_OUTPUT_PATHS = {
    "contract": "contract.json",
    "panel_manifest": "panel_manifest.json",
    "split_manifest": "split_manifest.json",
    "graph_manifest": "graph_manifest.json",
    "state_snapshot_index": "state_snapshot_index.json",
    "state_snapshots": "state_snapshots.npz",
    "teacher_trace_index": "teacher_trace_index.json",
    "teacher_traces": "teacher_traces.npz",
    "edge_port_index": "edge_port_index.json",
    "waypoint_contracts": "waypoint_contracts.json",
    "pixel_index": "pixel_index.json",
    "rgb_observations": "rgb_observations.npz",
    "latent_index": "latent_index.json",
    "canonical_latents": "canonical_latents.npz",
    "candidate_traces": "candidate_traces.npz",
    "candidate_fanout": "candidate_fanout.jsonl",
    "development_target_selection": "development_target_selection.json",
    "heldout_ranker_scores": "heldout_ranker_scores.jsonl",
    "repeated_execution": "repeated_execution.jsonl",
    "metrics": "metrics.json",
    "result": "result.json",
    "report": "result.md",
    "file_hashes": "file_hashes.json",
}
OUTPUT_LEAF_COUNT = len(RUNTIME_OUTPUT_PATHS)

SCIENTIFIC_PROHIBITIONS = {
    "model_training": 0,
    "ranker_training": 0,
    "predictor_training": 0,
    "predictor_inference": 0,
    "safety_model_training": 0,
    "v1_runtime_artifact_reuse": 0,
    "v2_runtime_artifact_reuse": 0,
    "online_graph_construction": 0,
    "topological_memory_execution": 0,
    "novelty_execution": 0,
    "beacon_discovery": 0,
    "closed_loop_maze_navigation": 0,
    "deployment_safety_claim": 0,
}
SAFETY_WORKSTREAM_STATUS = "REQUIREMENTS_ACQUISITION_REQUIRED"
CLAIMS = {
    "development_only": True,
    "permitted": "physical graph-edge handoff qualification for a fixed candidate bank",
    "does_not_establish": (
        "deployment safety",
        "learned contact avoidance",
        "physical Go2 safety",
        "online topology construction",
        "persistent memory",
        "hidden beacon discovery",
        "complete maze navigation",
    ),
}

DIRECT_RUNTIME_POLICY = {
    "interpreters": {
        "physical_collection": ".generated/venvs/genesis_rocm_0_4_6_v1/bin/python",
        "encoding_and_ranker": "/home/andrewknowles/TinyQuadJEPA/bin/python",
        "independent_reducer": "/usr/bin/python3",
    },
    "required_environment_before_simulator_creation": {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    },
    "persist_runtime_versions": (
        "python", "torch", "genesis", "quadrants", "device"
    ),
    "ordinary_foreground_processes": True,
    "exact_reset_fixture_before_fanout": True,
    "exact_reset_trials_per_state": 2,
    "singleton_encoder_batch_size": 1,
    "frozen_ranker_inference_only": True,
    "development_target_selection_before_heldout_open": True,
    "heldout_target_replacement_forbidden": True,
    "stage_order": (
        "verify broad scene exclusions before simulator creation",
        "execute and persist teacher-only qualification traces for all 256 prospective candidates",
        "freeze one hash-first eligible state per family balance stratum",
        "capture the selected 64 exact decision snapshots and pass two independent serialized-restore fixtures without replacement",
        "assign 12 DEVELOPMENT and 4 DEVELOPMENT_HELDOUT per family by frozen hash",
        "singleton-encode current RGB",
        "execute DEVELOPMENT candidate fanout and freeze target selection",
        "open DEVELOPMENT_HELDOUT candidate fanout and ranker outcomes",
        "execute the two exact-reset repeats for ranker-selected and oracle branches",
        "independently reduce metrics and publish",
    ),
    "role_on_prospective_candidate_spec": False,
    "ordinary_json_jsonl_npz": True,
    "independent_pure_reducer": True,
    "custom_audit_hooks": False,
    "custom_startup_or_forensic_framework": False,
}

PHYSICAL_RUNTIME_CORE_FIELDS = (
    "stage_id", "python_executable", "python_version", "torch_version",
    "torch_hip_version", "genesis_version", "quadrants_version",
    "visible_device_count", "device", "backend",
    "deterministic_environment", "fake_runtime",
)
PHYSICAL_RUNTIME_ENVIRONMENT_FIELDS = PHYSICAL_RUNTIME_CORE_FIELDS + (
    "runtime_core_sha256", "qualification_runtime_sha256s",
    "selected_snapshot_runtime_sha256s",
)
VISUAL_RUNTIME_ENVIRONMENT_FIELDS = (
    "stage_id", "python_executable", "python_version", "torch_version",
    "torch_hip_version", "visible_device_count", "device", "device_name",
    "device_capability", "backend", "fake_runtime", "model_role",
    "checkpoint_sha256", "model_source_path", "model_source_sha256",
    "external_repository_path", "external_repository_commit",
    "external_worktree_clean",
)
RUNTIME_ENVIRONMENT_AUTHORITY = {
    "digest_domain": "SHA-256 of canonical compact JSON projection without trailing LF",
    "physical": {
        "fields": list(PHYSICAL_RUNTIME_ENVIRONMENT_FIELDS),
        "core_fields": list(PHYSICAL_RUNTIME_CORE_FIELDS),
        "stage_id": "PHYSICAL_TEACHER_AND_RESET_COLLECTION",
        "real_python_executable": (
            "/home/andrewknowles/Workspace/LeWMQuad-v3/"
            ".generated/venvs/genesis_rocm_0_4_6_v1/bin/python"
        ),
        "python_version": "3.12.3",
        "torch_version": "2.12.0+rocm7.2",
        "torch_hip_version": "7.2.53211",
        "genesis_version": "0.4.6",
        "quadrants_version": "0.6.2",
        "visible_device_count": 2,
        "device": "cpu",
        "backend": "cpu",
        "qualification_runtime_count": TEACHER_TRACE_COUNT,
        "selected_snapshot_runtime_count": STATE_COUNT,
        "all_320_shard_core_digests_must_equal": True,
    },
    "encoder": {
        "fields": list(VISUAL_RUNTIME_ENVIRONMENT_FIELDS),
        "stage_id": "VJEPA_SINGLETON_ENCODING",
        "real_python_executable": DIRECT_RUNTIME_POLICY["interpreters"]["encoding_and_ranker"],
        "python_version": "3.12.3",
        "torch_version": "2.10.0.dev20250926+rocm6.3",
        "torch_hip_version": "6.3.42131-fa1d09cbd",
        "visible_device_count": 2,
        "device": "cuda:0",
        "device_name": "AMD Radeon AI PRO R9700",
        "device_capability": [12, 0],
        "backend": "torch_cuda_bfloat16_inference",
        "model_role": "VJEPA_ENCODER",
        "checkpoint_sha256": VJEPA_ENCODER_BINDING["checkpoint_sha256"],
        "model_source_path": VJEPA_ENCODER_BINDING["helper_path"],
        "model_source_sha256": VJEPA_ENCODER_BINDING["helper_sha256"],
        "external_repository_path": CANONICAL_ENCODING_AUTHORITY["external_encoder_source"]["repository_path"],
        "external_repository_commit": VJEPA_ENCODER_BINDING["external_repository_commit"],
        "external_worktree_clean": True,
    },
    "ranker": {
        "fields": list(VISUAL_RUNTIME_ENVIRONMENT_FIELDS),
        "stage_id": "FROZEN_CURRENT_VISUAL_RANKER_INFERENCE",
        "real_python_executable": DIRECT_RUNTIME_POLICY["interpreters"]["encoding_and_ranker"],
        "python_version": "3.12.3",
        "torch_version": "2.10.0.dev20250926+rocm6.3",
        "torch_hip_version": "6.3.42131-fa1d09cbd",
        "visible_device_count": 2,
        "device": "cpu",
        "device_name": None,
        "device_capability": None,
        "backend": "torch_cpu_inference",
        "model_role": "CURRENT_VISUAL_RANKER",
        "checkpoint_sha256": CURRENT_VISUAL_RANKER_BINDING["checkpoint_sha256"],
        "model_source_path": CURRENT_VISUAL_RANKER_BINDING["model_source_path"],
        "model_source_sha256": CURRENT_VISUAL_RANKER_BINDING["model_source_sha256"],
        "external_repository_path": None,
        "external_repository_commit": None,
        "external_worktree_clean": None,
    },
    "official_projections": {
        "physical": "panel_manifest.json.physical_runtime_environment",
        "physical_fanout_cross_binding": (
            "every candidate_fanout.jsonl and repeated_execution.jsonl row carries "
            "panel_manifest.physical_runtime_environment.runtime_core_sha256"
        ),
        "encoder": "latent_index.json.encoder_runtime_environment",
        "ranker": "development_target_selection.json.ranker_runtime_environment",
        "heldout_ranker_cross_binding": (
            "every heldout_ranker_scores.jsonl row carries the canonical SHA-256 "
            "of development_target_selection.ranker_runtime_environment"
        ),
    },
}
RUNTIME_CONTRACT_FIELDS = {
    "schema", "experiment_id", "status", "source_freeze_commit",
    "source_parent_commit", "source_baseline_commit", "v2_contract_freeze_commit",
    "scientific_contract", "predecessor_result_binding",
    "external_artifact_bindings", "runtime_policy", "content_digest",
}
PREDECESSOR_RESULT_BINDING = {
    "role": "v2_scientific_context_only",
    "path": str(V2.OUTPUT_ROOT / "result.json"),
    "bytes": V2_CONTEXT_BINDING["bindings"]["result.json"]["bytes"],
    "sha256": V2_CONTEXT_BINDING["bindings"]["result.json"]["sha256"],
    "kind": "frozen_result_identity_no_runtime_reuse",
}
EXTERNAL_ARTIFACT_BINDINGS = (
    {
        "role": "go2_platform_manifest",
        "path": "/home/andrewknowles/Workspace/LeWMQuad-v3/config/go2_platform_manifest.yaml",
        "bytes": 4613,
        "sha256": "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189",
        "kind": "frozen_physical_runtime_authority",
    },
    {
        "role": "go2_primitive_registry",
        "path": "/home/andrewknowles/Workspace/LeWMQuad-v3/config/go2_primitive_registry.yaml",
        "bytes": 2454,
        "sha256": "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8",
        "kind": "frozen_physical_runtime_authority",
    },
    {
        "role": "go2_ppo_policy_checkpoint",
        "path": "/home/andrewknowles/Workspace/LeWMQuad-v3/models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt",
        "bytes": 4547691,
        "sha256": "e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff",
        "kind": "frozen_low_level_controller_checkpoint_read_only",
    },
    {
        "role": "go2_ppo_policy_configuration",
        "path": "/home/andrewknowles/Workspace/LeWMQuad-v3/models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl",
        "bytes": 2409,
        "sha256": "bc3e68c18252475199e57b30c8ac49d813e3c784a3983e0e8b1a762490dde24f",
        "kind": "frozen_low_level_controller_configuration_read_only",
    },
    {
        "role": "genesis_go2_urdf",
        "path": "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
        "bytes": 24170,
        "sha256": "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4",
        "kind": "frozen_physical_robot_asset",
    },
    {
        "role": "vjepa_encoder_checkpoint",
        "path": VJEPA_ENCODER_BINDING["checkpoint_path"],
        "bytes": VJEPA_ENCODER_BINDING["checkpoint_size_bytes"],
        "sha256": VJEPA_ENCODER_BINDING["checkpoint_sha256"],
        "kind": "frozen_checkpoint_read_only",
    },
    {
        "role": "current_visual_ranker_checkpoint",
        "path": CURRENT_VISUAL_RANKER_BINDING["checkpoint_path"],
        "bytes": CURRENT_VISUAL_RANKER_BINDING["checkpoint_size_bytes"],
        "sha256": CURRENT_VISUAL_RANKER_BINDING["checkpoint_sha256"],
        "kind": "frozen_checkpoint_read_only",
    },
)
LOW_LEVEL_CONTROLLER_AUTHORITY = {
    "adapter_source_path": "lewm_genesis/lewm_genesis/rollout.py",
    "adapter_source_sha256": "06501bbbdd1e071a3a91e765d77bd19da5f2c311c35d75df4631c452beea034a",
    "adapter_class": "GenesisGo2PPOPolicy",
    "device": "cpu",
    "evaluation_mode": True,
    "torch_no_grad": True,
    "simulate_action_latency": True,
    "action_scale": 0.25,
    "policy_joint_order": (
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ),
    "command_order": ("vx_body_mps", "vy_body_mps", "yaw_rate_radps"),
    "training": False,
    "controller_observation_45_order": (
        "body angular velocity xyz scaled by obs_scales.ang_vel",
        "projected gravity xyz",
        "body command vx,vy,yaw scaled by lin_vel,lin_vel,ang_vel",
        "12 policy-order joint-position errors scaled by obs_scales.dof_pos",
        "12 policy-order joint velocities scaled by obs_scales.dof_vel",
        "12 previous policy actions",
    ),
    "snapshot_numeric_state": {
        "controller_observation": "the exact float64 projection of the 45-vector built immediately before the final snapshot policy act",
        "previous_policy_action": "the exact 12-vector _last_actions before that final policy act",
        "policy_last_action": "the exact 12-vector _last_actions after that final policy act",
        "command_history": "the latest 15 actually applied body commands [vx,vy,yaw], oldest to newest",
        "control_history": "the latest 15 actually applied [vx,yaw] pairs in frozen ranker order, oldest to newest",
        "low_level_policy_state": "the exact latency-applied 12-vector used for joint targets at snapshot time",
    },
    "rollout_foot_contact_source": "zero",
    "foot_contact_scope": (
        "the PPO 45-vector does not contain foot contact; the rollout collector's legacy "
        "foot-contact proxy is frozen to zero to avoid get_links_net_contact_force. "
        "Scientific contact outcomes are separately sampled with the safe manifold API"
    ),
}

TRACKED_SOURCE_PATHS = (
    "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_contract_2026-09-01.json",
    "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_fixture_2026-09-01.json",
    "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_output_schema_2026-09-01.json",
    "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_preregistration_2026-09-01.md",
    "docs/lewm_go2_physical_graph_edge_handoff_qualification_v1_source_closure_2026-09-01.json",
    "lewm/safety/physical_graph_edge_handoff_qualification_v1_contract.py",
    "lewm/safety/physical_graph_edge_handoff_qualification_v1_metrics.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v1_contract.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v1_metrics.py",
    "lewm/tests/test_run_physical_graph_edge_handoff_qualification_v1.py",
    "lewm/tests/test_evaluate_physical_graph_edge_handoff_qualification_v1.py",
    "scripts/run_physical_graph_edge_handoff_qualification_v1.py",
    "scripts/evaluate_physical_graph_edge_handoff_qualification_v1.py",
)
SOURCE_DEPENDENCY_PATHS = (
    "lewm/safety/occluded_goal_topological_belief_v1_contract.py",
    "lewm/safety/occluded_goal_topological_belief_v2_contract.py",
    "lewm/safety/occluded_goal_topological_belief_metrics_v2.py",
    "lewm/safety/occluded_goal_topological_belief_metrics_v1.py",
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "scripts/run_go2_oracle_branch_pilot_v1.py",
    "scripts/run_occluded_goal_topological_belief_v1.py",
    "lewm/safety/contact_hazard_ontology_v1.py",
    "scripts/evaluate_occluded_goal_topological_belief_v2.py",
    "scripts/evaluate_occluded_goal_topological_belief_v1.py",
    "lewm_genesis/lewm_genesis/__init__.py",
    "lewm_genesis/lewm_genesis/batch_renderer.py",
    "lewm_genesis/lewm_genesis/camera_safety.py",
    "lewm_genesis/lewm_genesis/collectors/__init__.py",
    "lewm_genesis/lewm_genesis/collectors/base.py",
    "lewm_genesis/lewm_genesis/collectors/frontier.py",
    "lewm_genesis/lewm_genesis/collectors/ou_noise.py",
    "lewm_genesis/lewm_genesis/collectors/primitive_curriculum.py",
    "lewm_genesis/lewm_genesis/collectors/recovery.py",
    "lewm_genesis/lewm_genesis/collectors/route_teacher.py",
    "lewm_genesis/lewm_genesis/go2_adapter.py",
    "lewm_genesis/lewm_genesis/lewm_contract.py",
    "lewm_genesis/lewm_genesis/parity_checks.py",
    "lewm_genesis/lewm_genesis/render_replay.py",
    "lewm_genesis/lewm_genesis/rollout.py",
    "lewm_genesis/lewm_genesis/ros_msg_adapter.py",
    "lewm_genesis/lewm_genesis/scene_builder.py",
    "lewm_genesis/lewm_genesis/scene_loader.py",
    "lewm_genesis/lewm_genesis/textures.py",
    "lewm_worlds/lewm_worlds/__init__.py",
    "lewm_worlds/lewm_worlds/corpus.py",
    "lewm_worlds/lewm_worlds/exporters/__init__.py",
    "lewm_worlds/lewm_worlds/exporters/to_gazebo_sdf.py",
    "lewm_worlds/lewm_worlds/exporters/to_genesis.py",
    "lewm_worlds/lewm_worlds/families.py",
    "lewm_worlds/lewm_worlds/labels/__init__.py",
    "lewm_worlds/lewm_worlds/labels/derived.py",
    "lewm_worlds/lewm_worlds/labels/topology.py",
    "lewm_worlds/lewm_worlds/manifest.py",
    "lewm_worlds/lewm_worlds/planning_grid.py",
    "lewm_worlds/lewm_worlds/randomization.py",
    "lewm_worlds/lewm_worlds/scene_graph.py",
    "lewm_worlds/lewm_worlds/scene_validation.py",
    "lewm_worlds/lewm_worlds/splits.py",
    "lewm/models/__init__.py",
    "lewm/models/direct_egocentric_bev_state_jepa_v1.py",
    "lewm/models/encoders.py",
    "lewm/models/lewm.py",
    "lewm/models/phase2d_spatial_lewm.py",
    "lewm/models/predictor.py",
    "lewm/models/primitive_affordance.py",
    "lewm/models/sigreg.py",
    "lewm/models/source_action_utility.py",
    "lewm/models/spatial_lewm.py",
    "lewm/models/spatial_predictor.py",
    "lewm/__init__.py",
    "lewm/safety/__init__.py",
)
# Generated freeze documents bind this executable closure; they cannot be
# members of their own source-hash projection.  The five docs remain in the
# exact tracked-change allow-list above, while closure covers the eight code /
# test paths and every imported scientific/runtime dependency.
SOURCE_CLOSURE_PATHS = TRACKED_SOURCE_PATHS[5:] + SOURCE_DEPENDENCY_PATHS


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON with one trailing LF."""

    def ready(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): ready(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (tuple, list)):
            return [ready(part) for part in item]
        if isinstance(item, Path):
            return str(item)
        return item

    return (
        json.dumps(
            ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Attach SHA-256 and return the exact JSON-native document projection."""

    row = copy.deepcopy(dict(value))
    if "content_digest" in row:
        raise PhysicalGraphEdgeHandoffContractError(
            "content_digest must be absent before attachment"
        )
    row["content_digest"] = hashlib.sha256(canonical_json_bytes(row)[:-1]).hexdigest()
    # Public documents cross process/file boundaries.  Normalise tuple-valued
    # frozen constants now so reducers and persisted JSON rebuild byte-for-byte
    # without accepting Python-only container types.
    return json.loads(canonical_json_bytes(row))


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalGraphEdgeHandoffContractError("document must be a mapping")
    row = copy.deepcopy(dict(value))
    digest = row.pop("content_digest", None)
    if not isinstance(digest, str) or digest != hashlib.sha256(
        canonical_json_bytes(row)[:-1]
    ).hexdigest():
        raise PhysicalGraphEdgeHandoffContractError("content_digest mismatch")
    row["content_digest"] = digest
    return row


def _route_geometry(
    *, family: str, direction: str, width_m: float, port_distance_m: float,
    translation_xy: tuple[float, float], spawn_lateral_m: float,
    spawn_yaw_rad: float, variant_port_distance_delta_m: float,
    variant_spawn_lateral_delta_m: float, variant_spawn_yaw_delta_rad: float,
    variant_corridor_length_delta_m: float, variant_opening_lateral_delta_m: float,
) -> dict[str, Any]:
    """Build a non-overlapping chamber/port recipe in the selected-edge frame."""

    angle_by_direction = {"STRAIGHT": 0.0, "LEFT": 0.65, "RIGHT": -0.65}
    selected_angle = angle_by_direction[direction]
    tx, ty = translation_xy
    c, s = math.cos(selected_angle), math.sin(selected_angle)
    tangent = (-s, c)
    front_u = port_distance_m + variant_port_distance_delta_m
    back_u, half_v = -1.0, 1.30
    corridor_length = 1.20 + variant_corridor_length_delta_m
    opening_base = (
        (0.25 if direction == "LEFT" else -0.25)
        if family == "OFFSET_OPENING" else 0.0
    )
    opening_v = opening_base + variant_opening_lateral_delta_m

    def world(u: float, v: float) -> list[float]:
        return [tx + c * u - s * v, ty + s * u + c * v]

    def segment_on_front(v0: float, v1: float) -> list[list[float]]:
        return [world(front_u, v0), world(front_u, v1)]

    def wall_between(label: str, a: list[float], b: list[float]) -> dict[str, Any]:
        dx, dy = b[0] - a[0], b[1] - a[1]
        return {
            "wall_id": label,
            "centre_xyz": [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, 0.30],
            "size_xyz": [math.hypot(dx, dy), 0.08, 0.60],
            "yaw_rad": math.atan2(dy, dx),
            "material_id": "NEUTRAL_WALL",
        }

    def corridor_polygon(centre: list[float], normal: tuple[float, float], width: float) -> list[list[float]]:
        nx, ny = -normal[1], normal[0]
        end = [centre[0] + normal[0] * corridor_length, centre[1] + normal[1] * corridor_length]
        half = width / 2.0
        return [
            [centre[0] + nx * half, centre[1] + ny * half],
            [end[0] + nx * half, end[1] + ny * half],
            [end[0] - nx * half, end[1] - ny * half],
            [centre[0] - nx * half, centre[1] - ny * half],
        ]

    selected_port = world(front_u, opening_v)
    selected_normal = (c, s)
    selected_opening = segment_on_front(opening_v - width_m / 2.0, opening_v + width_m / 2.0)
    selected_end = [
        selected_port[0] + selected_normal[0] * corridor_length,
        selected_port[1] + selected_normal[1] * corridor_length,
    ]
    source_boundary = [
        world(back_u, -half_v), world(front_u, -half_v),
        world(front_u, half_v), world(back_u, half_v),
    ]
    spawn = world(0.0, spawn_lateral_m + variant_spawn_lateral_delta_m) + [
        _angle_value(spawn_yaw_rad + variant_spawn_yaw_delta_rad)
    ]

    # Competing ports occupy distinct rectangle boundaries, never the selected
    # front boundary.  This makes first-port identity geometric rather than a
    # broad overlapping-region label.
    competitor_sides = {
        "STRAIGHT_PASSAGE": (),
        "TURNING_JUNCTION": ("BACK", "SIDE"),
        "OFFSET_OPENING": ("BACK",),
        "ROOM_OR_LOOP_EXIT": ("BACK", "TOP", "BOTTOM"),
    }[family]
    competing_edges: list[dict[str, Any]] = []
    competing_segments: dict[str, list[list[float]]] = {}
    for index, side in enumerate(competitor_sides):
        if side == "BACK":
            centre = world(back_u, 0.0); normal = (-c, -s)
            segment = [world(back_u, width_m / 2.0), world(back_u, -width_m / 2.0)]
        elif side == "TOP":
            centre_u = (back_u + front_u) / 2.0
            centre = world(centre_u, half_v); normal = tangent
            segment = [world(centre_u - width_m / 2.0, half_v), world(centre_u + width_m / 2.0, half_v)]
        elif side == "BOTTOM":
            centre_u = (back_u + front_u) / 2.0
            centre = world(centre_u, -half_v); normal = (-tangent[0], -tangent[1])
            segment = [world(centre_u + width_m / 2.0, -half_v), world(centre_u - width_m / 2.0, -half_v)]
        else:  # TURNING_JUNCTION's second branch is the body-opposite side.
            side_sign = -1.0 if direction == "LEFT" else 1.0
            centre_u = (back_u + front_u) / 2.0
            centre = world(centre_u, side_sign * half_v)
            normal = (side_sign * tangent[0], side_sign * tangent[1])
            segment = [world(centre_u - width_m / 2.0, side_sign * half_v), world(centre_u + width_m / 2.0, side_sign * half_v)]
        edge_id = f"competing-edge-{index}"
        competing_segments[side] = segment
        competing_edges.append({
            "edge_id": edge_id,
            "boundary_side": side,
            "opening_segment_world": segment,
            "opening_normal_world": [normal[0], normal[1]],
            "opening_width_m": width_m,
            "edge_region_polygon_world": corridor_polygon(centre, normal, width_m),
        })

    # Close each chamber boundary outside its declared port intervals.
    walls: list[dict[str, Any]] = []
    front_low, front_high = opening_v - width_m / 2.0, opening_v + width_m / 2.0
    if front_low > -half_v:
        walls.append(wall_between("front-low", world(front_u, -half_v), world(front_u, front_low)))
    if front_high < half_v:
        walls.append(wall_between("front-high", world(front_u, front_high), world(front_u, half_v)))
    boundary_port_side = set(competitor_sides)
    if "BACK" not in boundary_port_side:
        walls.append(wall_between("back", world(back_u, half_v), world(back_u, -half_v)))
    else:
        walls.extend([
            wall_between("back-upper", world(back_u, half_v), world(back_u, width_m / 2.0)),
            wall_between("back-lower", world(back_u, -width_m / 2.0), world(back_u, -half_v)),
        ])
    for side, v, label in (("TOP", half_v, "top"), ("BOTTOM", -half_v, "bottom")):
        if side not in boundary_port_side and not ("SIDE" in boundary_port_side and ((v < 0) == (direction == "LEFT"))):
            walls.append(wall_between(label, world(back_u, v), world(front_u, v)))
        else:
            centre_u = (back_u + front_u) / 2.0
            walls.extend([
                wall_between(f"{label}-rear", world(back_u, v), world(centre_u - width_m / 2.0, v)),
                wall_between(f"{label}-front", world(centre_u + width_m / 2.0, v), world(front_u, v)),
            ])
    # Corridor walls extend outward from the selected transverse opening.
    for side_sign, label in ((-1.0, "selected-corridor-low"), (1.0, "selected-corridor-high")):
        start = world(front_u, opening_v + side_sign * width_m / 2.0)
        end = [start[0] + selected_normal[0] * corridor_length, start[1] + selected_normal[1] * corridor_length]
        walls.append(wall_between(label, start, end))
    if family == "ROOM_OR_LOOP_EXIT":
        # A rear return partition distinguishes the chamber/loop family from a
        # simple junction while retaining the same neutral material.
        walls.append(wall_between("loop-return-partition", world(back_u + 0.22, -0.45), world(back_u + 0.22, 0.45)))

    target_half = 0.25
    target_boundary = [
        [selected_end[0] - target_half, selected_end[1] - target_half],
        [selected_end[0] + target_half, selected_end[1] - target_half],
        [selected_end[0] + target_half, selected_end[1] + target_half],
        [selected_end[0] - target_half, selected_end[1] + target_half],
    ]
    port_vector = [selected_port[0] - spawn[0], selected_port[1] - spawn[1]]
    bearing_body = _angle_value(math.atan2(port_vector[1], port_vector[0]) - spawn[2])
    return {
        "floor": {"centre_xyz": [tx, ty, -0.025], "size_xyz": [8.0, 8.0, 0.05], "material_id": "NEUTRAL_FLOOR"},
        "wall_boxes": walls,
        "source_node": {"node_id": "source", "centre_world": world((back_u + front_u) / 2.0, 0.0), "boundary_polygon_world": source_boundary},
        "target_node": {"node_id": "target", "centre_world": selected_end, "boundary_polygon_world": target_boundary},
        "selected_directed_edge": {
            "edge_id": "selected-edge", "source_node_id": "source", "target_node_id": "target",
            "route_direction": direction, "opening_segment_world": selected_opening,
            "opening_normal_world": [selected_normal[0], selected_normal[1]],
            "opening_width_m": width_m,
            "edge_region_polygon_world": corridor_polygon(selected_port, selected_normal, width_m),
        },
        "competing_directed_edges": competing_edges,
        "teacher_route_polyline_world": [[spawn[0], spawn[1]], selected_port, selected_end],
        "spawn_se2_world": spawn,
        "camera_recipe": copy.deepcopy(GEOMETRY_AUTHORITY["camera"]),
        "visual_material": copy.deepcopy(GEOMETRY_AUTHORITY["shared_visual_material"]),
        "nominal_port_hint_world": selected_port,
        "actual_port_rule": "derive from actual first teacher source-boundary crossing through selected transverse opening",
        "geometry_validity_witness": {
            "robot_start_inside_source": back_u < 0.0 < front_u and abs(spawn_lateral_m + variant_spawn_lateral_delta_m) < half_v,
            "selected_port_endpoints_on_source_boundary": True,
            "selected_and_competing_port_segments_are_on_distinct_boundary_sides": True,
            "selected_corridor_outside_source_except_boundary": True,
            "spawn_footprint_proxy_wall_clearance_m": min(front_u, -back_u, half_v - abs(spawn_lateral_m + variant_spawn_lateral_delta_m)),
            "body_frame_selected_port_bearing_rad": bearing_body,
        },
        "family_geometry_witness": {
            "STRAIGHT_PASSAGE": "single front passage with no competing source port",
            "TURNING_JUNCTION": "front selected branch plus distinct back and side source-boundary exits",
            "OFFSET_OPENING": "laterally offset front opening plus a distinct back exit",
            "ROOM_OR_LOOP_EXIT": "four-exit chamber with a neutral rear loop-return partition",
        }[family],
    }


def _angle_value(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def build_candidate_specs() -> list[dict[str, Any]]:
    """Return the 256 role-free, teacher-only prospective geometry specs."""

    specs: list[dict[str, Any]] = []
    width_ids = ("NARROW", "WIDE")
    distance_ids = ("NEAR", "FAR")
    lateral_values = tuple(GEOMETRY_AUTHORITY["spawn_lateral_offset_m"])
    yaw_values = tuple(GEOMETRY_AUTHORITY["spawn_yaw_offset_rad"])
    for family_index, family in enumerate(FAMILY_IDS):
        for stratum_index in range(16):
            width_id = width_ids[(stratum_index // 8) % 2]
            distance_id = distance_ids[(stratum_index // 4) % 2]
            lateral = lateral_values[(stratum_index // 2) % 2]
            yaw = yaw_values[stratum_index % 2]
            if family == "STRAIGHT_PASSAGE":
                direction = "STRAIGHT"
            elif family in {"TURNING_JUNCTION", "OFFSET_OPENING"}:
                direction = "LEFT" if stratum_index < 8 else "RIGHT"
            elif stratum_index < 8:
                direction = "STRAIGHT"
            elif stratum_index < 12:
                direction = "LEFT"
            else:
                direction = "RIGHT"
            for variant_index in range(4):
                pool_rank = stratum_index * 4 + variant_index
                global_rank = family_index * PROSPECTIVE_POOL_PER_FAMILY + pool_rank
                seed = PROCEDURAL_SEED_BASE + global_rank
                identity = f"{family}:{stratum_index}:{variant_index}:{seed}"
                digest = hashlib.sha256(identity.encode("utf-8")).digest()
                unit_x = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
                unit_y = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
                unit_port = int.from_bytes(digest[16:20], "big") / float(2**32 - 1)
                unit_lateral = int.from_bytes(digest[20:24], "big") / float(2**32 - 1)
                unit_yaw = int.from_bytes(digest[24:28], "big") / float(2**32 - 1)
                unit_length = int.from_bytes(digest[28:30], "big") / float(2**16 - 1)
                unit_opening = int.from_bytes(digest[30:32], "big") / float(2**16 - 1)
                jitter_limit = GEOMETRY_AUTHORITY["hash_jitter_limit_m"]
                variant_adjustments = {
                    "port_distance_delta_m": (2.0 * unit_port - 1.0) * 0.01,
                    "spawn_lateral_delta_m": (2.0 * unit_lateral - 1.0) * 0.01,
                    "spawn_yaw_delta_rad": (2.0 * unit_yaw - 1.0) * 0.02,
                    "corridor_length_delta_m": (2.0 * unit_length - 1.0) * 0.08,
                    "opening_lateral_delta_m": (2.0 * unit_opening - 1.0) * 0.03,
                }
                spec = {
                        "candidate_spec_id": f"pgehq-v1-spec-{family_index:02d}-{pool_rank:02d}",
                        "scene_id": f"{SCENE_ID_PREFIX}{family_index:02d}-{pool_rank:02d}",
                        "state_id": f"{STATE_ID_PREFIX}{family_index:02d}-{pool_rank:02d}",
                        "episode_id": f"pgehq-v1-episode-{family_index:02d}-{pool_rank:02d}",
                        "graph_id": f"pgehq-v1-graph-{family_index:02d}-{pool_rank:02d}",
                        "family": family,
                        "stratum_index": stratum_index,
                        "variant_index": variant_index,
                        "route_direction": direction,
                        "passage_width_id": width_id,
                        "passage_width_m": GEOMETRY_AUTHORITY["passage_width_m"][width_id],
                        "port_distance_id": distance_id,
                        "port_distance_m": GEOMETRY_AUTHORITY["port_distance_m"][distance_id],
                        "spawn_lateral_offset_m": lateral,
                        "spawn_yaw_offset_rad": yaw,
                        "geometry_jitter_xy_m": [
                            (2.0 * unit_x - 1.0) * jitter_limit,
                            (2.0 * unit_y - 1.0) * jitter_limit,
                        ],
                        "variant_adjustments": variant_adjustments,
                        "procedural_seed": seed,
                        "role": None,
                    }
                spec["geometry"] = _route_geometry(
                    family=family,
                    direction=direction,
                    width_m=spec["passage_width_m"],
                    port_distance_m=spec["port_distance_m"],
                    translation_xy=tuple(spec["geometry_jitter_xy_m"]),
                    spawn_lateral_m=lateral,
                    spawn_yaw_rad=yaw,
                    variant_port_distance_delta_m=variant_adjustments["port_distance_delta_m"],
                    variant_spawn_lateral_delta_m=variant_adjustments["spawn_lateral_delta_m"],
                    variant_spawn_yaw_delta_rad=variant_adjustments["spawn_yaw_delta_rad"],
                    variant_corridor_length_delta_m=variant_adjustments["corridor_length_delta_m"],
                    variant_opening_lateral_delta_m=variant_adjustments["opening_lateral_delta_m"],
                )
                spec["canonical_spec_sha256"] = hashlib.sha256(
                    canonical_json_bytes(spec)[:-1]
                ).hexdigest()
                specs.append(spec)
    return specs


def build_prospective_pool_specs() -> list[dict[str, Any]]:
    """Public runner-facing alias for the exact role-free physical pool."""

    return build_candidate_specs()


def build_contract() -> dict[str, Any]:
    return attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "development_only": DEVELOPMENT_ONLY,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "commit_subjects": {
                "freeze": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "result": RESULT_COMMIT_SUBJECT,
            },
            "panel": {
                "state_count": STATE_COUNT,
                "roles": copy.deepcopy(ROLE_COUNTS),
                "families": list(FAMILY_IDS),
                "states_per_family": STATES_PER_FAMILY,
                "family_role_counts": copy.deepcopy(FAMILY_ROLE_COUNTS),
                "turning_junction_side_counts": copy.deepcopy(
                    TURNING_JUNCTION_SIDE_COUNTS
                ),
                "identity_domain": IDENTITY_DOMAIN,
                "scene_id_prefix": SCENE_ID_PREFIX,
                "state_id_prefix": STATE_ID_PREFIX,
                "procedural_seed_values": list(PROCEDURAL_SEED_VALUES),
                "prospective_pool_count": PROSPECTIVE_POOL_COUNT,
                "prospective_pool_per_family": PROSPECTIVE_POOL_PER_FAMILY,
                "teacher_trace_count": TEACHER_TRACE_COUNT,
                "teacher_qualification_component_ids": list(
                    TEACHER_QUALIFICATION_COMPONENT_IDS
                ),
                "teacher_qualification_rejection_rule": (
                    TEACHER_QUALIFICATION_REJECTION_RULE
                ),
                "candidate_specs_sha256": hashlib.sha256(
                    canonical_json_bytes(build_candidate_specs())[:-1]
                ).hexdigest(),
                "constructed_set_caveat": CONSTRUCTED_SET_CAVEAT,
                "role_assignment": (
                    "freeze the complete eligible set, then deterministic hash order; "
                    "no replacement after any encoding, ranking, or physical fanout outcome"
                ),
                "split_assignment_authority": copy.deepcopy(
                    SPLIT_ASSIGNMENT_AUTHORITY
                ),
            },
            "prior_scene_exclusion_authority": copy.deepcopy(
                PRIOR_SCENE_EXCLUSION_AUTHORITY
            ),
            "v2_context_binding": copy.deepcopy(V2_CONTEXT_BINDING),
            "frozen_models": {
                "vjepa_encoder": copy.deepcopy(VJEPA_ENCODER_BINDING),
                "current_visual_ranker": copy.deepcopy(
                    CURRENT_VISUAL_RANKER_BINDING
                ),
                "low_level_controller": copy.deepcopy(
                    LOW_LEVEL_CONTROLLER_AUTHORITY
                ),
            },
            "candidate_bank": {
                "candidate_ids": list(CANDIDATE_IDS),
                "primitives": {
                    key: list(value) for key, value in CANDIDATE_PRIMITIVES.items()
                },
                "bank": [[name, list(blocks)] for name, blocks in CANDIDATE_BANK],
                "command_ticks_per_block": COMMAND_TICKS_PER_BLOCK,
                "executed_block_count": EXECUTED_BLOCK_COUNT,
                "candidate_count": CANDIDATE_COUNT,
                "branch_count": BRANCH_COUNT,
                "horizon_ticks": copy.deepcopy(HORIZON_TICKS),
                "primary_horizon": PRIMARY_HORIZON,
                "policy_steps_per_command_tick": POLICY_STEPS_PER_COMMAND_TICK,
                "physics_steps_per_policy_step": PHYSICS_STEPS_PER_POLICY_STEP,
                "physics_steps_per_command_tick": PHYSICS_STEPS_PER_COMMAND_TICK,
                "physics_steps_per_branch": PHYSICS_STEPS_PER_BRANCH,
                "port_dwell_command_ticks": PORT_DWELL_COMMAND_TICKS,
                "port_dwell_physics_samples": PORT_DWELL_PHYSICS_SAMPLES,
                "requested_and_post_slew_commands_persisted": True,
            },
            "snapshot_and_teacher_authority": {
                "exact_reset_trials_per_state": 2,
                "reset_definition": (
                    "restore the exact serialized solver, controller, command-history, "
                    "and RNG snapshot independently before each trial; clone or parallel "
                    "state equality is not a reset"
                ),
                "fixture_validation_before_fanout": True,
                "teacher_contact_free": True,
                "teacher_success": (
                    "contact-free; leaves the source; crosses the selected directed port; "
                    "positive route progress; no competing-port entry"
                ),
                "directed_port_definition": (
                    "first source-boundary crossing into the registered directed edge"
                ),
                "complete_physics_rate_trace_persisted": True,
                "reset_runtime_binding": copy.deepcopy(RESET_RUNTIME_BINDING),
                "reset_fixture_command": list(RESET_FIXTURE_COMMAND),
                "reset_fixture_command_ticks": RESET_FIXTURE_COMMAND_TICKS,
                "reset_fixture_physics_samples": RESET_FIXTURE_PHYSICS_SAMPLES,
                "reset_fixture_trace_count": RESET_FIXTURE_TRACE_COUNT,
                "reset_trace_pair_comparison_authority": copy.deepcopy(
                    RESET_TRACE_PAIR_COMPARISON_AUTHORITY
                ),
                "candidate_trace_count": CANDIDATE_TRACE_COUNT,
                "trace_index_ranges": copy.deepcopy(TRACE_INDEX_RANGES),
                "contact_authority": copy.deepcopy(CONTACT_AUTHORITY),
                "teacher_controller_authority": copy.deepcopy(
                    TEACHER_CONTROLLER_AUTHORITY
                ),
            },
            "targets": {
                "ids": list(TARGET_IDS),
                "definitions": copy.deepcopy(TARGET_DEFINITIONS),
                "feature_order": list(TARGET_FEATURE_ORDER),
                "target_tangent_audit_fields": list(TARGET_TANGENT_AUDIT_FIELDS),
                "route_lookahead_distance_m": ROUTE_LOOKAHEAD_DISTANCE_M,
                "development_selection_lexicographic": copy.deepcopy(
                    DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC
                ),
                "selected_before_heldout_open": True,
            },
            "heldout": {
                "conditions": list(HELDOUT_CONDITION_IDS),
                "score_row_count": HELDOUT_SCORE_ROW_COUNT,
                "repeat_branch_ids": list(REPEAT_BRANCH_IDS),
                "repeats_per_branch": REPEATS_PER_BRANCH,
                "repeat_row_count": REPEATED_EXECUTION_ROW_COUNT,
                "selector_authority": copy.deepcopy(HELDOUT_SELECTOR_AUTHORITY),
            },
            "canonical_encoding": copy.deepcopy(CANONICAL_ENCODING_AUTHORITY),
            "ranker_input_authority": copy.deepcopy(RANKER_INPUT_AUTHORITY),
            "original_ranker_training_support": copy.deepcopy(
                ORIGINAL_RANKER_TRAINING_SUPPORT
            ),
            "ranker_contract_difference_inventory": list(
                RANKER_CONTRACT_DIFFERENCE_INVENTORY
            ),
            "npz_payload_authority": copy.deepcopy(NPZ_PAYLOAD_AUTHORITY),
            "metric_formulas": copy.deepcopy(METRIC_FORMULAS),
            "physical_outcome_authority": copy.deepcopy(PHYSICAL_OUTCOME_AUTHORITY),
            "command_tracking_authority": copy.deepcopy(COMMAND_TRACKING_AUTHORITY),
            "numerical_tolerances": copy.deepcopy(NUMERICAL_TOLERANCES),
            "handoff_gate": copy.deepcopy(HANDOFF_GATE),
            "target_materiality": copy.deepcopy(TARGET_MATERIALITY),
            "primary_classifications": list(PRIMARY_CLASSIFICATIONS),
            "secondary_classifications": list(SECONDARY_CLASSIFICATIONS),
            "secondary_predicates": copy.deepcopy(SECONDARY_PREDICATES),
            "classification_precedence": list(CLASSIFICATION_PRECEDENCE),
            "component_precedence": list(COMPONENT_PRECEDENCE),
            "next_decisions": copy.deepcopy(NEXT_DECISION_BY_CLASSIFICATION),
            "composite_next_by_earliest_component": copy.deepcopy(
                COMPOSITE_NEXT_BY_EARLIEST_COMPONENT
            ),
            "claims": copy.deepcopy(CLAIMS),
            "safety_workstream": SAFETY_WORKSTREAM_STATUS,
            "prohibitions": copy.deepcopy(SCIENTIFIC_PROHIBITIONS),
            "runtime_policy": copy.deepcopy(DIRECT_RUNTIME_POLICY),
            "runtime_environment_authority": copy.deepcopy(
                RUNTIME_ENVIRONMENT_AUTHORITY
            ),
            "output": {
                "root": str(OUTPUT_ROOT),
                "runtime_paths": copy.deepcopy(RUNTIME_OUTPUT_PATHS),
                "leaf_count": OUTPUT_LEAF_COUNT,
                "receipt_self_digests": False,
            },
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "source_dependency_paths": list(SOURCE_DEPENDENCY_PATHS),
        }
    )


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_digest(value)
    expected = build_contract()
    # JSON persistence normalises tuple-valued frozen constants to lists.  The
    # canonical byte domain is the contract authority, so compare in that
    # domain instead of relying on Python container types.
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffContractError("contract value drift")
    return copy.deepcopy(dict(value))


def build_runtime_contract(source_freeze_commit: str) -> dict[str, Any]:
    """Bind the prospective contract to a descendant source-freeze commit."""

    if (
        not isinstance(source_freeze_commit, str)
        or len(source_freeze_commit) != 40
        or any(char not in "0123456789abcdef" for char in source_freeze_commit)
    ):
        raise PhysicalGraphEdgeHandoffContractError("invalid source_freeze_commit")
    return attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.runtime_contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "FROZEN_BEFORE_PHYSICAL_COLLECTION",
            "source_freeze_commit": source_freeze_commit,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "v2_contract_freeze_commit": V2_CONTEXT_BINDING["source_freeze_commit"],
            "scientific_contract": build_contract(),
            "predecessor_result_binding": copy.deepcopy(PREDECESSOR_RESULT_BINDING),
            "external_artifact_bindings": [
                copy.deepcopy(row) for row in EXTERNAL_ARTIFACT_BINDINGS
            ],
            "runtime_policy": copy.deepcopy(DIRECT_RUNTIME_POLICY),
        }
    )


def validate_runtime_contract(
    value: Mapping[str, Any], *, source_freeze_commit: str | None = None
) -> dict[str, Any]:
    row = validate_content_digest(value)
    if set(row) != RUNTIME_CONTRACT_FIELDS:
        raise PhysicalGraphEdgeHandoffContractError("runtime contract field drift")
    observed = row["source_freeze_commit"]
    if source_freeze_commit is not None and observed != source_freeze_commit:
        raise PhysicalGraphEdgeHandoffContractError("source freeze commit drift")
    expected = build_runtime_contract(observed)
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffContractError("runtime contract value drift")
    return copy.deepcopy(row)


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "PhysicalGraphEdgeHandoffContractError",
    "attach_content_digest",
    "build_contract",
    "build_candidate_specs",
    "build_prospective_pool_specs",
    "build_runtime_contract",
    "canonical_json_bytes",
    "validate_content_digest",
    "validate_contract",
    "validate_runtime_contract",
]

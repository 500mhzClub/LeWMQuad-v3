"""Pure contract and matched rankers for non-greedy local-subgoal planning V1.

This module is deliberately source-only.  Importing it performs no filesystem
discovery, model construction, checkpoint loading, simulator work, or device
initialisation.  The three registered rankers are instantiated only by an
explicit call to :func:`build_matched_rankers`.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:  # The pure authority/reducer path must not require the model runtime.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in an isolated subprocess.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    class _TorchUnavailableNN:
        Module = object

    nn = _TorchUnavailableNN()  # type: ignore[assignment]


EXPERIMENT_ID = "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
SOURCE_PARENT_COMMIT = "507577dcb62044fe449c833eef38dcf883fa71c2"
CONTRACT_FREEZE_COMMIT_SUBJECT = (
    "Freeze non-greedy local subgoal JEPA planning experiment"
)
RESULT_COMMIT_SUBJECT = "Evaluate non-greedy local subgoal JEPA planning experiment"
RUNNER_INTERPRETER = "/home/andrewknowles/TinyQuadJEPA/bin/python"
EXECUTION_RUNTIME_AUTHORITY = {
    "runner_interpreter": RUNNER_INTERPRETER,
    "python_version": "3.12.3",
    "required_thread_environment": {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    },
    "foreground_process_per_major_stage": 1,
    "custom_python_audit_hook": False,
    "launcher_child_role_hierarchy": False,
}

STATE_COUNT = 96
CANDIDATE_COUNT = 12
FAMILY_IDS = (
    "WALL_DETOUR",
    "U_ESCAPE",
    "DEAD_END_LURE",
    "OFFSET_PASSAGE",
)
STATES_PER_FAMILY = 24
SPLIT_ROLE_IDS = ("FIT", "CALIBRATION", "DEVELOPMENT_HELDOUT")
SPLIT_STATE_COUNTS = {"FIT": 64, "CALIBRATION": 16, "DEVELOPMENT_HELDOUT": 16}
SPLIT_FAMILY_STATE_COUNTS = {
    "FIT": 16,
    "CALIBRATION": 4,
    "DEVELOPMENT_HELDOUT": 4,
}
HORIZON_IDS = ("H1", "H2", "H3")
HORIZON_COUNT = 3
PANEL_SELECTION_SEED = 2_026_083_100
SCENE_HASH_NAMESPACE = f"{EXPERIMENT_ID}/SCENE_ORDER_V1"
PANEL_CANDIDATE_BLOCK_SIZE = 48
PANEL_MAXIMUM_CANDIDATES_PER_FAMILY = 1_536
PANEL_CONTINUATION_RULE = (
    "scan complete blocks of 48 deterministic candidate scenes for one family "
    "until the hash-first 24 eligible scenes also meet the frozen family-level "
    "panel adequacy contribution (at least 15 weak-or-negative direct-progress "
    "oracle winners and at least one useful initially-away admissible route); "
    "scan no more than 1536 candidates per family; then hash-order the complete "
    "eligible population from every scanned block"
)

PANEL_GEOMETRY_AUTHORITY = {
    "world_half_extent_m": 3.0,
    "robot_radius_m": 0.22,
    "occupancy_grid_resolution_m": 0.05,
    "occupancy_geodesic": (
        "exact_8_neighbor_with_no_diagonal_corner_cutting_and_"
        "edge_cost_aware_shortest_path_descent"
    ),
    "command_tick_seconds": 0.1,
    "physics_step_seconds": 0.002,
    "horizon_seconds": {"H1": 0.5, "H2": 1.0, "H3": 1.5},
    "slew_limits_per_command_tick": {
        "delta_vx_max_mps": 0.25,
        "delta_vy_max_mps": 0.0,
        "delta_yaw_rate_max_radps": 0.35,
    },
    "renderer": {
        "obstacles_visible": True,
        "width_pixels": 224,
        "height_pixels": 168,
        "horizontal_fov_degrees": 92.0,
        "goal_marker_visible": False,
        "wall_material": "uniform_neutral_geometry_only",
    },
}

NONVISUAL_COUNTERBALANCE_AUTHORITY = {
    "shared_start_pose_xy_yaw": [-0.48, 0.0, math.pi / 2.0],
    "shared_goal_xy": [2.18, 0.0],
    "family_and_side_are_not_model_inputs": True,
    "registered_nonvisual_feature_rule": (
        "for the fixed twelve-candidate bank, every selected state must produce "
        "byte-identical base features, byte-identical goal/action queries, and "
        "byte-identical deterministic direct-progress anchors; obstacle geometry, "
        "contact, viability, route family, geodesic outcomes, and candidate labels "
        "are forbidden from those inputs"
    ),
    "required_unique_base_feature_signatures": 1,
    "required_unique_query_feature_signatures": 1,
    "required_unique_anchor_signatures": 1,
}

COMPLETION_RADIUS_M = 0.25
STUCK_ENDPOINT_DISPLACEMENT_THRESHOLD_M = 0.015
STUCK_APPLIED_VX_THRESHOLD_MPS = 0.05
STUCK_APPLIED_ABS_YAW_RATE_THRESHOLD_RADPS = 0.10
DEAD_END_GEODESIC_NO_PROGRESS_TOLERANCE_M = 1.0e-9
DEAD_END_EUCLIDEAN_CLOSER_THRESHOLD_M = 1.0e-6
DESCRIPTIVE_OUTCOME_AUTHORITY = {
    "role": "descriptive_only_never_training_or_selection_targets",
    "completion": {
        "formula": "endpoint Euclidean goal distance <= completion_radius_m",
        "completion_radius_m": COMPLETION_RADIUS_M,
    },
    "stuck": {
        "formula": (
            "endpoint displacement < displacement_threshold_m and "
            "(absolute applied vx > applied_abs_vx_threshold_mps or absolute applied yaw rate "
            "> applied_abs_yaw_rate_threshold_radps)"
        ),
        "displacement_threshold_m": STUCK_ENDPOINT_DISPLACEMENT_THRESHOLD_M,
        "applied_abs_vx_threshold_mps": STUCK_APPLIED_VX_THRESHOLD_MPS,
        "applied_abs_yaw_rate_threshold_radps": (
            STUCK_APPLIED_ABS_YAW_RATE_THRESHOLD_RADPS
        ),
    },
    "dead_end": {
        "formula": (
            "endpoint geodesic distance >= start geodesic distance - "
            "geodesic_no_progress_tolerance_m while start Euclidean distance - "
            "endpoint Euclidean distance > euclidean_closer_threshold_m"
        ),
        "geodesic_no_progress_tolerance_m": (
            DEAD_END_GEODESIC_NO_PROGRESS_TOLERANCE_M
        ),
        "euclidean_closer_threshold_m": DEAD_END_EUCLIDEAN_CLOSER_THRESHOLD_M,
    },
}

PRIMITIVES = {
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
    (
        "turn_left_then_go",
        ("yaw_left", "yaw_left", "forward_medium", "forward_medium"),
    ),
    (
        "turn_right_then_go",
        ("yaw_right", "yaw_right", "forward_medium", "forward_medium"),
    ),
    (
        "go_then_turn_left",
        ("forward_medium", "forward_medium", "yaw_left", "yaw_left"),
    ),
    ("reverse_then_turn", ("backward", "backward", "yaw_left", "yaw_left")),
    ("hold", ("hold",) * 4),
)
BLOCK_COUNT = 4
TICKS_PER_BLOCK = 5

GEODESIC_TARGET_AUTHORITY = {
    "target": "exact H3 occupancy-grid geodesic progress toward the final route goal",
    "geodesic_progress_m": "start_geodesic_distance_m - h3_geodesic_distance_m",
    "euclidean_progress_role": "descriptive_only_never_training_or_selection_target",
    "admissibility": (
        "prospectively frozen deterministic obstacle-visible planar geometry, "
        "2 ms collision stepping, and geodesic occupancy oracle"
    ),
    "completion_role": "descriptive_only",
    "target_tie_break": (
        "larger geodesic progress, lower remaining geodesic, lower "
        "heading error to the next shortest-path segment, lower candidate index"
    ),
    "persisted_numeric_quantization_decimal_places": 9,
    "minimum_geometrically_plausible_route_alternatives": 2,
    "non_greedy_requirement": (
        "at least one oracle route begins with a local subgoal whose Euclidean "
        "distance to the final goal is strictly greater than the start distance"
    ),
    "direct_goal_line_blocked_required": True,
    "topological_memory": False,
    "beacon_layer": False,
}

RECOVERY_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "plan_aware_monotone_jepa_cost_development_recovery_v1"
)
RECOVERY_FILE_BINDINGS = {
    "recovered_development_result.json": {
        "bytes": 6_470_922,
        "sha256": "510661e5f7da88db7a7c17f06e328cfeb9c9e41e596ea761bf0d10642d0c0b1a",
        "content_digest": "9af7cf6ff751074e13b14bfee5a774f2c7c13f6fbc98b65fb4bd34e155be5f50",
        "authority_use": "parsed_recovery_decision_only",
    },
    "result.md": {
        "bytes": 7_457,
        "sha256": "56387c2b7c2cd648a3bca32d98efbaaf3ad548242a9752a4a3bf9e13436533d8",
        "content_digest": None,
        "authority_use": "hash_bound_not_scientific_input",
    },
    "row_evidence_index.json": {
        "bytes": 944_137,
        "sha256": "86fd98ac5ffdd7acf04f31d68586fcf688927ef282f5959698a26d6c22ae3201",
        "content_digest": "845ddd81c9c243d1fd3b6d69f1e5e42ac654b1e409967dd3954f2973c6c5fc4c",
        "authority_use": "hash_bound_not_new_scientific_input",
    },
    "regeneration_receipt.json": {
        "bytes": 13_754,
        "sha256": "4e3576242263bfeebd1533b053bd23cf6d40e060d524782f50ad902bce27f5d8",
        "content_digest": "5b3a79f0a577313db60a622323161a5ea05902590254c458c8953f13368b3547",
        "authority_use": "parsed_recovery_custody_only",
    },
    "file_hashes.json": {
        "bytes": 1_139,
        "sha256": "77de5e94ed96ebb33e624b2fee8df256aca1d958a72f19a520869770588b0226",
        "content_digest": "7b5f1e1a989d4d75ba1815820170b84539633565a83a050c6bf13ea069ccf651",
        "authority_use": "parsed_inventory_authority_only",
    },
}
RECOVERY_DECISION_AUTHORITY = {
    "status": "DEVELOPMENT_SCIENTIFIC_PAYLOAD_RECOVERED",
    "primary_classification": "KINEMATIC_BASELINE_DOMINANT",
    "preserved_classifications": [
        "DEVELOPMENT_SCIENTIFIC_PAYLOAD_RECOVERED",
        "KINEMATIC_BASELINE_DOMINANT",
        "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
        "TWO_STEP_PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED",
        "RAW_LATENT_GOAL_COST_NO_GO",
        "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
        "REQUIREMENTS_ACQUISITION_REQUIRED",
        "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED",
        "SIMULATED_CONTACT_PROXY_SCOPE_ONLY",
        "REPLANNING_INTERFACE_UNRESOLVED",
        "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
    ],
    "support_dispositions": {
        "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE": False,
        "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS": False,
    },
    "interpretation": [
        "Candidate specific information in true future V-JEPA trajectories affected route ranking.",
        "Deranging the candidate future trajectories materially reduced progress and worsened regret.",
        "Despite using candidate specific latent information, the full latent ranker was worse than deterministic kinematics.",
        "Zeroing the latent branch improved the trained model.",
        "R1 and RR produced almost identical selected route outcomes.",
        "PR improved relative to P1, but no registered proprioceptive contribution trigger passed.",
    ],
    "benchmark_conclusion": (
        "The previous local waypoint benchmark was largely solvable from immediate "
        "action kinematics and was not an adequate task on which to demonstrate "
        "incremental world model value."
    ),
    "exact_preserved_prior_development_status": {
        "classification_scope": {
            "authority": "JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1",
            "incremental_not_supported_scope": (
                "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS_NOT_SUPPORTED "
                "applies to the predecessor raw-cost result only"
            ),
            "scope": "predecessor raw token-wise goal-cosine planning-cost assay",
        },
        "classifications": [
            "RAW_LATENT_GOAL_COST_NO_GO",
            "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
            "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS_NOT_SUPPORTED",
        ],
        "findings": [
            "rollout training improves direct counterfactual future fidelity at H1–H4.",
            "rollout training improves selected action-specific retrieval metrics, strongest at H3–H4.",
            "predicted latents retain partial occupancy information but remain substantially below true-target occupancy.",
            "the registered proprioception interaction was broadly null.",
            "planning utility was not tested by the predictor qualification assay.",
            "Raw token-wise cosine distance between future V-JEPA latents and the virtual goal-view latent did not provide useful route ordering, even with true-future latents and oracle viability.",
            "The virtual goal renderer contained the floor plane but not the maze walls or landmarks. That result was therefore not a valid visual wall-avoidance test.",
            "deterministic kinematics was substantially stronger",
        ],
    },
    "exact_safety_workstream": {
        "classifications": [
            "REQUIREMENTS_ACQUISITION_REQUIRED",
            "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED",
            "SIMULATED_CONTACT_PROXY_SCOPE_ONLY",
            "REPLANNING_INTERFACE_UNRESOLVED",
            "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        ],
        "planning_result_resolves_deployment_safety": False,
    },
    "next_experiment": EXPERIMENT_ID,
    "caveat": (
        "This is a post hoc development recovery from an outcome observed failed "
        "publication attempt. It is suitable for design decisions but not an "
        "independent confirmatory claim."
    ),
    "confirmation_policy": (
        "OBSERVED_PANEL_EXPLORATORY_ONLY; "
        "NEW_SCENE_DISJOINT_PANEL_AND_ONE_EXPLORATORY_SEED_REQUIRED_BEFORE_"
        "CONFIRMATORY_REPLICATION"
    ),
    "artifacts_reused_for_new_science": 0,
    "failed_archive_dereference_authorized": False,
}

ENCODER_BINDING = {
    "path": "/home/andrewknowles/.cache/vjepa2_1_vitl_dist_vitG_384.pt",
    "sha256": "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
    "bytes": 5_151_198_524,
    "model": "vjepa2_1_vitl_384",
    "frozen": True,
}
PREDICTOR_BINDINGS = {
    "R1_RGB_ONE_STEP": {
        "path": (
            "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
            "seed_2026080901/seed_2026080901_rgb_one_step_epoch21.pt"
        ),
        "sha256": "20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a",
        "bytes": 206_534_551,
        "use_proprio": False,
        "rollout": False,
    },
    "RR_RGB_ROLLOUT": {
        "path": (
            "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
            "seed_2026080901/seed_2026080901_rgb_rollout_epoch21.pt"
        ),
        "sha256": "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4",
        "bytes": 206_534_551,
        "use_proprio": False,
        "rollout": True,
    },
}
PREDICTOR_INPUT_AUTHORITY = {
    "context_shape": [3, 768, 1024],
    "action_blocks_shape": [3, 5, 2],
    "flattened_action_block_shape": [3, 10],
    "control_history_shape": [3, 5, 2],
    "horizons": list(HORIZON_IDS),
    "weights_dtype": "float32",
    "autocast_dtype": "bfloat16",
    "output_persistence_dtype": "float16",
    "device": "cuda:0 (PyTorch CUDA compatibility API over frozen ROCm runtime)",
    "training_or_checkpoint_mutation": False,
    "fresh_state_history_convention": {
        "context": "three exact repetitions of the one CURRENT RGB/latent",
        "control": "exact zeros with shape [12,3,5,2]",
        "actions": "post-slew active vx/yaw blocks for H1-H3",
        "status": "declared distribution-shift limitation",
    },
}

NO_LATENT_NON_GREEDY_RANKER = "NO_LATENT_NON_GREEDY_RANKER"
CURRENT_VISUAL_REACTIVE_RANKER = "CURRENT_VISUAL_REACTIVE_RANKER"
TRUE_FUTURE_JEPA_TRAJECTORY_RANKER = "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER"
MODEL_IDS = (
    NO_LATENT_NON_GREEDY_RANKER,
    CURRENT_VISUAL_REACTIVE_RANKER,
    TRUE_FUTURE_JEPA_TRAJECTORY_RANKER,
)
STAGE_A_SOURCE_BY_MODEL = {
    NO_LATENT_NON_GREEDY_RANKER: "NO_LATENT",
    CURRENT_VISUAL_REACTIVE_RANKER: "CURRENT_VISUAL",
    TRUE_FUTURE_JEPA_TRAJECTORY_RANKER: "TRUE_FUTURE",
}
STAGE_B_SOURCE_IDS = ("R1", "RR")
DEVELOPMENT_HELDOUT_STATE_IDS = tuple(
    f"ngls-{family_index:02d}-{family_rank:02d}"
    for family_index in range(len(FAMILY_IDS))
    for family_rank in range(20, 24)
)
STAGE_A_MODEL_SOURCE_PAIRS = (
    ("DETERMINISTIC_KINEMATICS", "KINEMATIC"),
    (NO_LATENT_NON_GREEDY_RANKER, "NO_LATENT"),
    (CURRENT_VISUAL_REACTIVE_RANKER, "CURRENT_VISUAL"),
    (TRUE_FUTURE_JEPA_TRAJECTORY_RANKER, "TRUE_FUTURE"),
    ("FUTURE_TRAJECTORY_DERANGEMENT", "TRUE_FUTURE_DERANGED_CANDIDATE"),
    ("FUTURE_TIME_ORDER_DERANGEMENT", "TRUE_FUTURE_DERANGED_TIME"),
)
STAGE_B_MODEL_SOURCE_PAIRS = (
    (TRUE_FUTURE_JEPA_TRAJECTORY_RANKER, "R1"),
    (TRUE_FUTURE_JEPA_TRAJECTORY_RANKER, "RR"),
)
STAGE_A_SCORE_ROW_COUNT = len(DEVELOPMENT_HELDOUT_STATE_IDS) * CANDIDATE_COUNT * len(
    STAGE_A_MODEL_SOURCE_PAIRS
)
STAGE_B_SCORE_ROW_COUNT = len(DEVELOPMENT_HELDOUT_STATE_IDS) * CANDIDATE_COUNT * len(
    STAGE_B_MODEL_SOURCE_PAIRS
)

MODEL_SEED = 2_026_083_101
TOKEN_GRID_SHAPE = (24, 32)
TOKENS_PER_FRAME = 768
TOKEN_DIM = 1024
LATENT_WIDTH = 64
BASE_FEATURE_DIM = 131
QUERY_FEATURE_DIM = 66
BASE_FEATURE_SLICES = {
    "relative_goal_position_and_heading_sin_cos": (0, 4),
    "requested_action_plan_3x5x3": (4, 49),
    "applied_action_plan_3x5x3": (49, 94),
    "previous_command_vx_vy_yaw": (94, 97),
    "observed_control_history_3x5x2": (97, 127),
    "direct_progress": (127, 128),
    "endpoint_kinematics": (128, 131),
}
QUERY_FEATURE_SLICES = {
    "relative_goal_position_and_heading_sin_cos": (0, 4),
    "applied_active_vx_yaw_3x5x2": (4, 34),
    "previous_active_vx_yaw": (34, 36),
    "observed_control_history_3x5x2": (36, 66),
}
TIMEPOINT_IDS = ("CURRENT", "H1", "H2", "H3")
TIMEPOINT_COUNT = 4
RESIDUAL_INPUT_DIM = BASE_FEATURE_DIM + TIMEPOINT_COUNT * LATENT_WIDTH
RESIDUAL_HIDDEN_DIMS = (256, 128)
PARAMETER_CAP_EXCLUSIVE = 500_000
REGISTERED_PARAMETER_COUNT = 204_289

TRAINING_POLICY = {
    "seed": MODEL_SEED,
    "optimizer": "AdamW",
    "optimizer_arguments": {
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "differentiable": False,
    },
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "epochs": 60,
    "checkpoint_policy": "final_only",
    "loss_weights": {"pairwise": 1.0, "listwise": 0.5, "residual": 1.0e-3},
    "development_heldout_model_scores_open_after_final_checkpoints": True,
    "three_models_trained_independently_from_byte_identical_initialization": True,
}

FAMILY_PERFORMANCE_FLOOR = {
    "population": "ORACLE_VIABILITY_ADMISSIBLE",
    "pairwise_accuracy_minimum_each_family": 0.50,
    "oracle_progress_fraction_minimum_each_family": 0.50,
    "score_not_completely_collapsed_each_family": True,
    "formula": (
        "every one of the four families has defined pairwise accuracy >= 0.50, "
        "defined oracle-progress fraction >= 0.50, and at least one state with "
        "non-tied admissible scores"
    ),
}

STAGE_A_GATE = {
    "absolute": {
        "pairwise_accuracy_minimum": 0.75,
        "spearman_minimum": 0.60,
        "normalized_regret_maximum": 0.25,
        "best_route_top3_minimum": 0.75,
        "oracle_progress_fraction_minimum": 0.80,
        "no_family_complete_collapse": True,
        "family_performance_floor": True,
    },
    "incremental_over_each_of": [
        "KINEMATIC_BASELINE",
        CURRENT_VISUAL_REACTIVE_RANKER,
    ],
    "incremental_minimum_criteria": 2,
    "incremental_criteria": {
        "pairwise_accuracy_gain_minimum": 0.05,
        "normalized_regret_reduction_minimum": 0.05,
        "oracle_progress_fraction_gain_minimum": 0.10,
    },
    "candidate_future_derangement_material_if_any": {
        "pairwise_accuracy_drop_minimum": 0.05,
        "normalized_regret_increase_minimum": 0.05,
        "oracle_progress_fraction_drop_minimum": 0.10,
    },
}
STAGE_B_GATE = {
    "rr_absolute": {
        "pairwise_accuracy_minimum": 0.70,
        "normalized_regret_maximum": 0.30,
        "best_route_top3_minimum": 0.75,
        "oracle_progress_fraction_minimum": 0.75,
        "true_selected_progress_fraction_minimum": 0.85,
        "no_family_complete_collapse": True,
        "family_performance_floor": True,
    },
    "rr_over_r1": {
        "pairwise_accuracy_gain_minimum": 0.03,
        "and_one_of": {
            "normalized_regret_reduction_minimum": 0.03,
            "oracle_progress_fraction_gain_minimum": 0.05,
        },
    },
}

PRIMARY_CLASSIFICATIONS = (
    "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_SIGNAL",
    "NON_GREEDY_REACTIVE_OR_KINEMATIC_BASELINE_DOMINANT",
    "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_NO_SIGNAL",
)
STAGE_B_CLASSIFICATIONS = (
    "NON_GREEDY_TWO_STEP_JEPA_PLANNING_SIGNAL",
    "NON_GREEDY_TRUE_FUTURE_SIGNAL_PREDICTOR_NO_GO",
    "NON_GREEDY_JEPA_PLANNING_SIGNAL_NO_ROLLOUT_ADVANTAGE",
)
NEXT_DECISION_BY_CLASSIFICATION = {
    "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_SIGNAL": "RUN_CONDITIONAL_STAGE_B",
    "NON_GREEDY_REACTIVE_OR_KINEMATIC_BASELINE_DOMINANT": (
        "move the JEPA evaluation to topological or longer-horizon subgoal "
        "selection, rather than training another local route ranker."
    ),
    "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_NO_SIGNAL": (
        "compare the frozen V-JEPA representation against a strong dense spatial "
        "representation such as DINO features or an explicit learned spatial state "
        "before further predictor work."
    ),
    "NON_GREEDY_TWO_STEP_JEPA_PLANNING_SIGNAL": (
        "ORACLE_ADMISSIBLE_NON_GREEDY_CLOSED_LOOP_JEPA_MPC_V1"
    ),
    "NON_GREEDY_TRUE_FUTURE_SIGNAL_PREDICTOR_NO_GO": (
        "PLAN_AWARE_PREDICTOR_TRAINING_NON_GREEDY_V1"
    ),
    "NON_GREEDY_JEPA_PLANNING_SIGNAL_NO_ROLLOUT_ADVANTAGE": (
        "ORACLE_ADMISSIBLE_ONE_STEP_JEPA_MPC_V1"
    ),
}

CLAIM_AUTHORITY = {
    "concerns": [
        "simulated non-greedy local subgoal selection",
        "candidate route ordering",
        "oracle admissibility",
        "development planning suitability",
    ],
    "does_not_establish": [
        "deployment safety",
        "learned contact avoidance",
        "physical Go2 safety",
        "hidden beacon discovery",
        "topological localisation",
        "persistent memory",
        "complete maze navigation",
    ],
    "positive_wording": "JEPA route selection under oracle admissibility.",
    "prohibited_wording": "JEPA safety.",
    "deployment_safety_claim": False,
}
PROHIBITIONS = {
    "rerun_old_experiment": True,
    "same_48_state_route_cost_panel": True,
    "v1_or_v2_startup": True,
    "forensic_or_diagnostic_pass": True,
    "custom_audit_hooks": True,
    "deployment_safety_changes": True,
    "safety_model_changes": True,
    "topological_memory": True,
    "beacon_or_novelty_discovery": True,
    "full_maze_solver": True,
    "closed_loop_control": True,
    "predictor_training": True,
    "alter_safety_workstream": True,
    "recovery_artifact_reuse_for_new_science": True,
    "failed_archive_or_v2_attempt_root_reads": True,
}
STOP_POINTS = (
    "BENCHMARK_CONSTRUCTION",
    "PANEL_ADEQUACY",
    "STAGE_A",
    "CONDITIONAL_STAGE_B",
    "FINAL_CLASSIFICATION",
    "NEXT_EXPERIMENT_SPECIFICATION",
)

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "non_greedy_local_subgoal_jepa_planning_v1"
)
RUNTIME_OUTPUT_PATHS = {
    "contract": "contract.json",
    "panel_manifest": "panel_manifest.json",
    "split_manifest": "split_manifest.json",
    "branch_ledger": "branch_ledger.jsonl",
    "route_labels": "route_labels.jsonl",
    "latent_index": "latent_index.json",
    "predictor_index": "predictor_index.json",
    "training_ledger": "training_ledger.jsonl",
    "no_latent_checkpoint": (
        "checkpoints/no_latent_non_greedy_ranker_final_epoch_060.pt"
    ),
    "current_visual_checkpoint": (
        "checkpoints/current_visual_reactive_ranker_final_epoch_060.pt"
    ),
    "true_future_checkpoint": (
        "checkpoints/true_future_jepa_trajectory_ranker_final_epoch_060.pt"
    ),
    "stage_a_scores": "stage_a_scores.jsonl",
    "conditional_stage_b_scores": "conditional_stage_b_scores.jsonl",
    "metrics": "metrics.json",
    "result": "result.json",
    "independent_regeneration_receipt": "independent_regeneration_receipt.json",
    "report": "result.md",
    "file_hashes": "file_hashes.json",
}
OUTPUT_AUTHORITY = {
    "runtime_paths": copy.deepcopy(RUNTIME_OUTPUT_PATHS),
    "independent_reducer_reconstructs": "metrics.json",
    "report_publication_is_scientific_identity": False,
    "persistence": (
        "ordinary atomic file writes inside one fresh official output root plus "
        "a complete SHA-256 file manifest; no custom finaliser"
    ),
}

TRACKED_SOURCE_PATHS = (
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_contract_2026-08-31.json",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_fixture_2026-08-31.json",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_output_schema_2026-08-31.json",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_preregistration_2026-08-31.md",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_source_closure_2026-08-31.json",
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_metrics_v1.py",
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "lewm/tests/test_evaluate_non_greedy_local_subgoal_jepa_planning_v1.py",
    "lewm/tests/test_non_greedy_local_subgoal_jepa_planning_metrics_v1.py",
    "lewm/tests/test_non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "lewm/tests/test_run_non_greedy_local_subgoal_jepa_planning_v1.py",
    "scripts/evaluate_non_greedy_local_subgoal_jepa_planning_v1.py",
    "scripts/run_non_greedy_local_subgoal_jepa_planning_v1.py",
)
TRACKED_RESULT_PATHS = (
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.json",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.md",
)


class NonGreedyContractError(ValueError):
    """Raised when a pure contract/model input violates the frozen authority."""


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by every pure authority receipt."""

    def ready(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): ready(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float):
            if not math.isfinite(item):
                raise NonGreedyContractError("canonical JSON forbids non-finite floats")
            return item
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise NonGreedyContractError(
            f"canonical JSON cannot encode {type(item).__name__}"
        )

    return (
        json.dumps(ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    output.pop("content_digest", None)
    output["content_digest"] = hashlib.sha256(canonical_json_bytes(output)[:-1]).hexdigest()
    return output


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    declared = output.pop("content_digest", None)
    if not isinstance(declared, str) or len(declared) != 64:
        raise NonGreedyContractError("missing or malformed content_digest")
    observed = hashlib.sha256(canonical_json_bytes(output)[:-1]).hexdigest()
    if observed != declared:
        raise NonGreedyContractError("content_digest mismatch")
    return copy.deepcopy(dict(value))


def candidate_bank_digest() -> str:
    value = {
        "bank": [[name, list(sequence)] for name, sequence in CANDIDATE_BANK],
        "primitives": {key: list(value) for key, value in PRIMITIVES.items()},
        "blocks": BLOCK_COUNT,
        "ticks": TICKS_PER_BLOCK,
    }
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def deterministic_scene_hash(
    *, family: str, scene_id: str, scene_manifest_sha256: str
) -> str:
    if family not in FAMILY_IDS:
        raise NonGreedyContractError(f"unknown family {family!r}")
    if not isinstance(scene_id, str) or not scene_id:
        raise NonGreedyContractError("scene_id must be nonempty")
    if (
        not isinstance(scene_manifest_sha256, str)
        or len(scene_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in scene_manifest_sha256)
    ):
        raise NonGreedyContractError("scene manifest SHA-256 is malformed")
    preimage = "\0".join(
        (
            SCENE_HASH_NAMESPACE,
            str(PANEL_SELECTION_SEED),
            family,
            scene_id,
            scene_manifest_sha256,
        )
    ).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def deterministic_split_manifest(
    eligible_scene_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select and split 24 scenes/family after hashing the complete population."""

    rows = [copy.deepcopy(dict(row)) for row in eligible_scene_rows]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    for row in rows:
        family = row.get("family")
        scene_id = row.get("scene_id")
        manifest_digest = row.get("scene_manifest_sha256")
        if not isinstance(family, str) or not isinstance(scene_id, str):
            raise NonGreedyContractError("eligible scene lacks family/scene_id")
        identity = (family, scene_id)
        if identity in identities:
            raise NonGreedyContractError("duplicate eligible scene identity")
        identities.add(identity)
        row["selection_hash"] = deterministic_scene_hash(
            family=family,
            scene_id=scene_id,
            scene_manifest_sha256=str(manifest_digest),
        )
        by_family[family].append(row)
    if set(by_family) != set(FAMILY_IDS):
        raise NonGreedyContractError("eligible population does not cover four families")

    selected: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILY_IDS):
        population = sorted(
            by_family[family],
            key=lambda row: (str(row["selection_hash"]), str(row["scene_id"])),
        )
        if len(population) < STATES_PER_FAMILY:
            raise NonGreedyContractError(
                f"{family}: fewer than {STATES_PER_FAMILY} complete eligible scenes"
            )
        for family_rank, row in enumerate(population[:STATES_PER_FAMILY]):
            if family_rank < 16:
                role = "FIT"
            elif family_rank < 20:
                role = "CALIBRATION"
            else:
                role = "DEVELOPMENT_HELDOUT"
            selected.append(
                {
                    **row,
                    "family_population_count": len(population),
                    "family_hash_rank": family_rank,
                    "state_id": f"ngls-{family_index:02d}-{family_rank:02d}",
                    "split_role": role,
                }
            )
    role_counts = {
        role: sum(row["split_role"] == role for row in selected)
        for role in SPLIT_ROLE_IDS
    }
    family_counts = {
        family: sum(row["family"] == family for row in selected)
        for family in FAMILY_IDS
    }
    if role_counts != SPLIT_STATE_COUNTS or any(
        count != STATES_PER_FAMILY for count in family_counts.values()
    ):
        raise NonGreedyContractError("deterministic split cardinality drift")
    return attach_content_digest(
        {
            "schema": "non_greedy_local_subgoal_split_manifest_v1",
            "experiment_id": EXPERIMENT_ID,
            "complete_eligible_population_count": len(rows),
            "selection_rule": (
                "per family sort complete eligible population by frozen hash then "
                "scene_id; select first 24; first 16 fit, next 4 calibration, "
                "final 4 DEVELOPMENT_HELDOUT"
            ),
            "role_counts": role_counts,
            "family_counts": family_counts,
            "states": selected,
        }
    )


def _seed_for_parameter(seed: int, name: str) -> int:
    payload = f"{EXPERIMENT_ID}\0{int(seed)}\0{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _initialise_named_parameters(module: nn.Module, seed: int) -> None:
    """Name-keyed initialisation makes every shape-compatible tensor identical."""

    with torch.no_grad():
        for name, parameter in sorted(module.named_parameters()):
            if name.endswith("token_layer_norm.weight"):
                parameter.fill_(1.0)
            elif name.endswith("bias"):
                parameter.zero_()
            elif parameter.ndim >= 2:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(_seed_for_parameter(seed, name))
                fan_in = int(parameter.shape[-1])
                bound = 1.0 / math.sqrt(fan_in)
                value = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                value.uniform_(-bound, bound, generator=generator)
                parameter.copy_(value.to(parameter.device))
            else:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(_seed_for_parameter(seed, name))
                value = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                value.normal_(mean=0.0, std=0.02, generator=generator)
                parameter.copy_(value.to(parameter.device))


@dataclass(frozen=True)
class RankerOutput:
    score: torch.Tensor
    kinematic_anchor: torch.Tensor
    residual: torch.Tensor
    timepoint_summaries: torch.Tensor


class MatchedNonGreedyRanker(nn.Module):
    """One of three byte-matched non-greedy route ranker conditions."""

    def __init__(self, model_id: str, *, seed: int = MODEL_SEED) -> None:
        if torch is None or F is None:
            raise RuntimeError("ranker construction requires the registered PyTorch runtime")
        super().__init__()
        if model_id not in MODEL_IDS:
            raise NonGreedyContractError(f"unknown model_id {model_id!r}")
        self.model_id = model_id
        self.seed = int(seed)
        self.token_layer_norm = nn.LayerNorm(TOKEN_DIM)
        self.shared_token_projection = nn.Linear(TOKEN_DIM, LATENT_WIDTH)
        self.query_projection = nn.Linear(QUERY_FEATURE_DIM, LATENT_WIDTH)
        self.residual_mlp = nn.Sequential(
            nn.Linear(RESIDUAL_INPUT_DIM, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.register_buffer(
            "fixed_absence_summaries",
            torch.zeros(TIMEPOINT_COUNT, LATENT_WIDTH),
            persistent=True,
        )
        _initialise_named_parameters(self, self.seed)
        if parameter_count(self) != REGISTERED_PARAMETER_COUNT:
            raise RuntimeError("registered ranker parameter count drift")

    @staticmethod
    def _flat_tokens(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise NonGreedyContractError(f"{name} must be a floating tensor")
        if value.shape[-3:] == (*TOKEN_GRID_SHAPE, TOKEN_DIM):
            return value.reshape(*value.shape[:-3], TOKENS_PER_FRAME, TOKEN_DIM)
        if value.shape[-2:] == (TOKENS_PER_FRAME, TOKEN_DIM):
            return value
        raise NonGreedyContractError(
            f"{name} must end in [24,32,1024] or [768,1024]"
        )

    def _pool(
        self, tokens: torch.Tensor, query_features: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        flat = self._flat_tokens(tokens, name=name)
        query_leading = query_features.shape[:-1]
        if flat.shape[:-2] == torch.Size() and query_leading:
            flat = flat.reshape(
                *((1,) * len(query_leading)), TOKENS_PER_FRAME, TOKEN_DIM
            ).expand(*query_leading, TOKENS_PER_FRAME, TOKEN_DIM)
        if flat.shape[:-2] != query_leading:
            raise NonGreedyContractError(f"{name} and query leading shapes differ")
        projected = self.shared_token_projection(self.token_layer_norm(flat))
        query = self.query_projection(query_features)
        attention = torch.softmax(
            torch.einsum("...d,...nd->...n", query, projected)
            / math.sqrt(LATENT_WIDTH),
            dim=-1,
        )
        return torch.einsum("...n,...nd->...d", attention, projected)

    def forward(
        self,
        *,
        base_features: torch.Tensor,
        query_features: torch.Tensor,
        kinematic_anchor: torch.Tensor,
        current_tokens: torch.Tensor | None = None,
        future_tokens: torch.Tensor | None = None,
    ) -> RankerOutput:
        if (
            not isinstance(base_features, torch.Tensor)
            or base_features.shape[-1:] != (BASE_FEATURE_DIM,)
            or not base_features.is_floating_point()
        ):
            raise NonGreedyContractError("base_features must end in 131 floats")
        leading = base_features.shape[:-1]
        if (
            not isinstance(query_features, torch.Tensor)
            or query_features.shape != (*leading, QUERY_FEATURE_DIM)
            or not query_features.is_floating_point()
        ):
            raise NonGreedyContractError("query_features must align and end in 66 floats")
        if not isinstance(kinematic_anchor, torch.Tensor) or kinematic_anchor.shape != leading:
            raise NonGreedyContractError("kinematic_anchor must match leading shape")
        if any(
            value.dtype != base_features.dtype or value.device != base_features.device
            for value in (query_features, kinematic_anchor)
        ):
            raise NonGreedyContractError("ranker numeric inputs must share dtype/device")

        absence = self.fixed_absence_summaries.to(
            dtype=base_features.dtype, device=base_features.device
        ).reshape(*((1,) * len(leading)), TIMEPOINT_COUNT, LATENT_WIDTH)
        absence = absence.expand(*leading, TIMEPOINT_COUNT, LATENT_WIDTH)
        if self.model_id == NO_LATENT_NON_GREEDY_RANKER:
            if current_tokens is not None or future_tokens is not None:
                raise NonGreedyContractError("no-latent condition forbids token inputs")
            summaries = absence
        else:
            if current_tokens is None:
                raise NonGreedyContractError("visual conditions require current tokens")
            current = self._pool(current_tokens, query_features, name="current_tokens")
            if self.model_id == CURRENT_VISUAL_REACTIVE_RANKER:
                if future_tokens is not None:
                    raise NonGreedyContractError(
                        "current-visual condition requires fixed H1-H3 absence"
                    )
                summaries = torch.cat([current.unsqueeze(-2), absence[..., 1:, :]], dim=-2)
            else:
                if future_tokens is None:
                    raise NonGreedyContractError("true-future condition requires H1-H3")
                if future_tokens.shape[-4:] == (
                    HORIZON_COUNT,
                    *TOKEN_GRID_SHAPE,
                    TOKEN_DIM,
                ):
                    future = future_tokens.reshape(
                        *future_tokens.shape[:-4],
                        HORIZON_COUNT,
                        TOKENS_PER_FRAME,
                        TOKEN_DIM,
                    )
                elif future_tokens.shape[-3:] == (
                    HORIZON_COUNT,
                    TOKENS_PER_FRAME,
                    TOKEN_DIM,
                ):
                    future = future_tokens
                else:
                    raise NonGreedyContractError(
                        "future_tokens must end in [3,24,32,1024] or [3,768,1024]"
                    )
                if future.shape[:-3] != leading:
                    raise NonGreedyContractError("future token leading shape drift")
                pooled = [
                    self._pool(future[..., index, :, :], query_features, name=HORIZON_IDS[index])
                    for index in range(HORIZON_COUNT)
                ]
                summaries = torch.stack([current, *pooled], dim=-2)

        residual_input = torch.cat(
            [base_features, summaries.reshape(*leading, TIMEPOINT_COUNT * LATENT_WIDTH)],
            dim=-1,
        )
        residual = self.residual_mlp(residual_input).squeeze(-1)
        score = kinematic_anchor + residual
        return RankerOutput(
            score=score,
            kinematic_anchor=kinematic_anchor,
            residual=residual,
            timepoint_summaries=summaries,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_matched_rankers(
    seed: int = MODEL_SEED,
) -> dict[str, MatchedNonGreedyRanker]:
    """Construct all three models without perturbing the caller's RNG stream."""

    if torch is None:
        raise RuntimeError("ranker construction requires the registered PyTorch runtime")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        output = {model_id: MatchedNonGreedyRanker(model_id, seed=seed) for model_id in MODEL_IDS}
    assert_matched_initialization(output)
    return output


def assert_matched_initialization(
    models: Mapping[str, MatchedNonGreedyRanker],
) -> None:
    if set(models) != set(MODEL_IDS):
        raise NonGreedyContractError("matched ranker set is incomplete")
    reference = models[MODEL_IDS[0]].state_dict()
    for model_id in MODEL_IDS[1:]:
        observed = models[model_id].state_dict()
        if reference.keys() != observed.keys() or any(
            not torch.equal(reference[name], observed[name]) for name in reference
        ):
            raise NonGreedyContractError(
                f"{model_id}: initial state is not byte-identical and shape-compatible"
            )


def matched_ranker_loss(
    *,
    scores: torch.Tensor,
    target_utilities: torch.Tensor,
    state_indices: torch.Tensor,
    residuals: torch.Tensor,
    tie_tolerance: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Registered pairwise + listwise + residual loss, grouped by state."""

    if torch is None or F is None:
        raise RuntimeError("ranker loss requires the registered PyTorch runtime")
    if any(value.ndim != 1 for value in (scores, target_utilities, state_indices, residuals)):
        raise NonGreedyContractError("loss inputs must be one-dimensional")
    if not (scores.shape == target_utilities.shape == state_indices.shape == residuals.shape):
        raise NonGreedyContractError("loss inputs must align one-to-one")
    if not scores.numel() or not scores.is_floating_point() or not target_utilities.is_floating_point():
        raise NonGreedyContractError("loss requires nonempty floating scores/targets")
    if not math.isfinite(float(tie_tolerance)) or tie_tolerance < 0:
        raise NonGreedyContractError("tie tolerance must be finite and non-negative")

    pair_terms: list[torch.Tensor] = []
    list_terms: list[torch.Tensor] = []
    for state_id in torch.unique(state_indices, sorted=True):
        mask = state_indices == state_id
        state_scores = scores[mask]
        state_targets = target_utilities[mask]
        if state_scores.numel() < 2:
            continue
        target_probability = torch.softmax(state_targets, dim=0)
        list_terms.append(-(target_probability * torch.log_softmax(state_scores, dim=0)).sum())
        for left in range(state_scores.numel()):
            for right in range(left + 1, state_scores.numel()):
                delta = state_targets[left] - state_targets[right]
                if bool(torch.abs(delta).detach().cpu() <= tie_tolerance):
                    continue
                direction = torch.sign(delta)
                pair_terms.append(
                    F.softplus(-direction * (state_scores[left] - state_scores[right]))
                )
    zero = scores.sum() * 0.0
    pairwise = torch.stack(pair_terms).mean() if pair_terms else zero
    listwise = torch.stack(list_terms).mean() if list_terms else zero
    residual = residuals.square().mean()
    weights = TRAINING_POLICY["loss_weights"]
    total = (
        float(weights["pairwise"]) * pairwise
        + float(weights["listwise"]) * listwise
        + float(weights["residual"]) * residual
    )
    return {
        "total": total,
        "pairwise": pairwise,
        "listwise": listwise,
        "residual": residual,
        "pair_count": torch.tensor(len(pair_terms), device=scores.device),
        "state_count": torch.tensor(len(list_terms), device=scores.device),
    }


def model_contract() -> dict[str, Any]:
    return {
        "model_ids": list(MODEL_IDS),
        "seed": MODEL_SEED,
        "token_grid_shape": list(TOKEN_GRID_SHAPE),
        "tokens_per_frame": TOKENS_PER_FRAME,
        "token_dim": TOKEN_DIM,
        "latent_width": LATENT_WIDTH,
        "timepoints": list(TIMEPOINT_IDS),
        "base_feature_dim": BASE_FEATURE_DIM,
        "base_feature_slices": {
            key: list(value) for key, value in BASE_FEATURE_SLICES.items()
        },
        "query_feature_dim": QUERY_FEATURE_DIM,
        "query_feature_slices": {
            key: list(value) for key, value in QUERY_FEATURE_SLICES.items()
        },
        "predecessor_route_role_feature_used": False,
        "shared_single_query_attention": True,
        "token_transform": "LayerNorm(1024) then shared Linear(1024,64)",
        "reactive_future_summary": "fixed all-zero nontrainable H1-H3 summaries",
        "residual_mlp": [RESIDUAL_INPUT_DIM, 256, 128, 1],
        "score": "S = kinematic_anchor + learned_residual",
        "parameters_each": REGISTERED_PARAMETER_COUNT,
        "parameter_cap_exclusive": PARAMETER_CAP_EXCLUSIVE,
        "initialization": "byte-identical name-keyed tensors across all three models",
    }


def build_contract() -> dict[str, Any]:
    return attach_content_digest(
        {
            "schema": "non_greedy_local_subgoal_jepa_planning_v1.contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "commit_subjects": {
                "freeze": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "result": RESULT_COMMIT_SUBJECT,
            },
            "execution_runtime": copy.deepcopy(EXECUTION_RUNTIME_AUTHORITY),
            "panel": {
                "state_count": STATE_COUNT,
                "families": list(FAMILY_IDS),
                "states_per_family": STATES_PER_FAMILY,
                "split_state_counts": copy.deepcopy(SPLIT_STATE_COUNTS),
                "split_family_state_counts": copy.deepcopy(SPLIT_FAMILY_STATE_COUNTS),
                "selection_seed": PANEL_SELECTION_SEED,
                "candidate_block_size": PANEL_CANDIDATE_BLOCK_SIZE,
                "maximum_candidates_per_family": (
                    PANEL_MAXIMUM_CANDIDATES_PER_FAMILY
                ),
                "continuation_rule": PANEL_CONTINUATION_RULE,
                "selection_rule": (
                    "hash-order the complete eligible population independently per "
                    "family; choose first 24; split 16/4/4"
                ),
                "new_scene_disjoint_from_recovery_panel": True,
            },
            "panel_geometry": copy.deepcopy(PANEL_GEOMETRY_AUTHORITY),
            "nonvisual_counterbalance": copy.deepcopy(
                NONVISUAL_COUNTERBALANCE_AUTHORITY
            ),
            "candidate_bank": {
                "count": CANDIDATE_COUNT,
                "horizons": list(HORIZON_IDS),
                "blocks": BLOCK_COUNT,
                "ticks_per_block": TICKS_PER_BLOCK,
                "primitives": {key: list(value) for key, value in PRIMITIVES.items()},
                "bank": [[name, list(sequence)] for name, sequence in CANDIDATE_BANK],
                "digest": candidate_bank_digest(),
            },
            "geodesic_target": copy.deepcopy(GEODESIC_TARGET_AUTHORITY),
            "descriptive_outcomes": copy.deepcopy(DESCRIPTIVE_OUTCOME_AUTHORITY),
            "recovery_authority": {
                "root": str(RECOVERY_ROOT),
                "files": copy.deepcopy(RECOVERY_FILE_BINDINGS),
                "decision": copy.deepcopy(RECOVERY_DECISION_AUTHORITY),
            },
            "frozen_encoder": copy.deepcopy(ENCODER_BINDING),
            "frozen_predictors": copy.deepcopy(PREDICTOR_BINDINGS),
            "predictor_input": copy.deepcopy(PREDICTOR_INPUT_AUTHORITY),
            "model": model_contract(),
            "score_ledgers": {
                "stage_a": {
                    "split": "DEVELOPMENT_HELDOUT",
                    "state_count": len(DEVELOPMENT_HELDOUT_STATE_IDS),
                    "candidate_count": CANDIDATE_COUNT,
                    "model_source_pairs": [
                        {"model_id": model_id, "source_id": source_id}
                        for model_id, source_id in STAGE_A_MODEL_SOURCE_PAIRS
                    ],
                    "row_count": STAGE_A_SCORE_ROW_COUNT,
                },
                "conditional_stage_b": {
                    "split": "DEVELOPMENT_HELDOUT",
                    "state_count": len(DEVELOPMENT_HELDOUT_STATE_IDS),
                    "candidate_count": CANDIDATE_COUNT,
                    "model_source_pairs": [
                        {"model_id": model_id, "source_id": source_id}
                        for model_id, source_id in STAGE_B_MODEL_SOURCE_PAIRS
                    ],
                    "row_count": STAGE_B_SCORE_ROW_COUNT,
                    "all_or_absent": True,
                },
            },
            "training": copy.deepcopy(TRAINING_POLICY),
            "gates": {
                "stage_a": copy.deepcopy(STAGE_A_GATE),
                "stage_b": copy.deepcopy(STAGE_B_GATE),
                "family_performance_floor": copy.deepcopy(FAMILY_PERFORMANCE_FLOOR),
            },
            "classifications": {
                "stage_a": list(PRIMARY_CLASSIFICATIONS),
                "stage_b": list(STAGE_B_CLASSIFICATIONS),
                "next_decision": copy.deepcopy(NEXT_DECISION_BY_CLASSIFICATION),
            },
            "claims": copy.deepcopy(CLAIM_AUTHORITY),
            "prohibitions": copy.deepcopy(PROHIBITIONS),
            "stop_points": list(STOP_POINTS),
            "output": {
                "root": str(OUTPUT_ROOT),
                **copy.deepcopy(OUTPUT_AUTHORITY),
            },
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "tracked_result_paths": list(TRACKED_RESULT_PATHS),
        }
    )


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_digest(value)
    expected = build_contract()
    if dict(value) != expected:
        raise NonGreedyContractError("contract value drift")
    return copy.deepcopy(expected)


def contract_bytes() -> bytes:
    return canonical_json_bytes(build_contract())


__all__ = [
    "BASE_FEATURE_DIM",
    "CANDIDATE_BANK",
    "CANDIDATE_COUNT",
    "CLAIM_AUTHORITY",
    "COMPLETION_RADIUS_M",
    "CONTRACT_FREEZE_COMMIT_SUBJECT",
    "CURRENT_VISUAL_REACTIVE_RANKER",
    "DEAD_END_EUCLIDEAN_CLOSER_THRESHOLD_M",
    "DEAD_END_GEODESIC_NO_PROGRESS_TOLERANCE_M",
    "DEVELOPMENT_HELDOUT_STATE_IDS",
    "DESCRIPTIVE_OUTCOME_AUTHORITY",
    "ENCODER_BINDING",
    "EXPERIMENT_ID",
    "FAMILY_IDS",
    "FAMILY_PERFORMANCE_FLOOR",
    "GEODESIC_TARGET_AUTHORITY",
    "HORIZON_IDS",
    "LATENT_WIDTH",
    "MODEL_IDS",
    "MODEL_SEED",
    "MatchedNonGreedyRanker",
    "NEXT_DECISION_BY_CLASSIFICATION",
    "NO_LATENT_NON_GREEDY_RANKER",
    "NONVISUAL_COUNTERBALANCE_AUTHORITY",
    "NonGreedyContractError",
    "OUTPUT_ROOT",
    "PANEL_CANDIDATE_BLOCK_SIZE",
    "PANEL_CONTINUATION_RULE",
    "PANEL_GEOMETRY_AUTHORITY",
    "PANEL_MAXIMUM_CANDIDATES_PER_FAMILY",
    "PREDICTOR_BINDINGS",
    "PRIMARY_CLASSIFICATIONS",
    "PROHIBITIONS",
    "QUERY_FEATURE_DIM",
    "RECOVERY_DECISION_AUTHORITY",
    "RECOVERY_FILE_BINDINGS",
    "RECOVERY_ROOT",
    "REGISTERED_PARAMETER_COUNT",
    "RESULT_COMMIT_SUBJECT",
    "RUNTIME_OUTPUT_PATHS",
    "SPLIT_ROLE_IDS",
    "SPLIT_STATE_COUNTS",
    "STAGE_A_GATE",
    "STAGE_A_MODEL_SOURCE_PAIRS",
    "STAGE_A_SCORE_ROW_COUNT",
    "STAGE_A_SOURCE_BY_MODEL",
    "STAGE_B_CLASSIFICATIONS",
    "STAGE_B_GATE",
    "STAGE_B_MODEL_SOURCE_PAIRS",
    "STAGE_B_SCORE_ROW_COUNT",
    "STAGE_B_SOURCE_IDS",
    "STATE_COUNT",
    "STUCK_APPLIED_ABS_YAW_RATE_THRESHOLD_RADPS",
    "STUCK_APPLIED_VX_THRESHOLD_MPS",
    "STUCK_ENDPOINT_DISPLACEMENT_THRESHOLD_M",
    "TOKEN_DIM",
    "TOKEN_GRID_SHAPE",
    "TOKENS_PER_FRAME",
    "TRACKED_RESULT_PATHS",
    "TRACKED_SOURCE_PATHS",
    "TRAINING_POLICY",
    "TRUE_FUTURE_JEPA_TRAJECTORY_RANKER",
    "attach_content_digest",
    "assert_matched_initialization",
    "build_contract",
    "build_matched_rankers",
    "candidate_bank_digest",
    "canonical_json_bytes",
    "contract_bytes",
    "deterministic_scene_hash",
    "deterministic_split_manifest",
    "matched_ranker_loss",
    "model_contract",
    "parameter_count",
    "validate_content_digest",
    "validate_contract",
]

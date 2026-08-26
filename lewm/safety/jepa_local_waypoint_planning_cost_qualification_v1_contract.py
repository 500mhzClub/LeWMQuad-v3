"""Prospective, payload-free contract for JEPA local-waypoint cost qualification.

Importing this module does not open a checkpoint, latent cache, route-outcome
ledger, result table, simulator, GPU, or scientific output directory.  It only
defines the outcome-independent experiment contract, output schema, small pure
fixtures, and canonical receipt helpers.

The experiment is development-only.  It asks whether a frozen JEPA latent
distance is a useful *planning cost* on an already-frozen Route-Intent V2 panel.
Oracle contact/viability populations are evaluation strata, not learned safety
claims.  Nothing here authorises training, deployment-safety scope reduction,
G2 access, memory, or experimental candidate-selecting closed-loop execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1"
CONTRACT_SCHEMA_VERSION = "jepa_local_waypoint_planning_cost_qualification_v1.contract.v1"
OUTPUT_SCHEMA_VERSION = "jepa_local_waypoint_planning_cost_qualification_v1.output.v1"
FIXTURE_SCHEMA_VERSION = "jepa_local_waypoint_planning_cost_qualification_v1.fixture.v1"
SOURCE_CLOSURE_SCHEMA_VERSION = (
    "jepa_local_waypoint_planning_cost_qualification_v1.source_closure.v1"
)
GOAL_VIEW_AMENDMENT_SCHEMA_VERSION = (
    "jepa_local_waypoint_planning_cost_qualification_v1.goal_view_amendment.v1"
)

STARTING_HEAD = "b29eae1929725a4cc26a35d95662b545daee4553"
STAGE_A_FREEZE_COMMIT = "e9e8c41a327ddbe51c38fa04f05ae1d30266720b"
SEED = 2026080901
BOOTSTRAP_REPLICATES = 10_000

SOURCE_IDS = (
    "TRUE_FUTURE",
    "ONE_STEP_PREDICTED",
    "TWO_STEP_PREDICTED",
)
POPULATION_IDS = (
    "ALL_CANDIDATES",
    "ORACLE_CONTACT_FREE",
    "ORACLE_VIABILITY_ADMISSIBLE",
)
COMPARATOR_IDS = (
    "KINEMATIC_ROUTE_BASELINE",
    "RANDOM",
    "TRUE_FUTURE_LATENT_COST",
    "ONE_STEP_PREDICTED_LATENT_COST",
    "TWO_STEP_PREDICTED_LATENT_COST",
)
PAIRED_COMPARISON_IDS = (
    "TWO_STEP_MINUS_ONE_STEP",
    "TWO_STEP_MINUS_KINEMATIC",
    "TWO_STEP_MINUS_TRUE_FUTURE",
    "TRUE_FUTURE_MINUS_KINEMATIC",
)
PRIMARY_CLASSIFICATIONS = (
    "TWO_STEP_JEPA_PLANNING_COST_SIGNAL",
    "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO",
    "RAW_LATENT_GOAL_COST_NO_GO",
    "KINEMATIC_BASELINE_DOMINANT",
)
SECONDARY_CLASSIFICATIONS = (
    "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS",
)
TRUE_FUTURE_GATE_CLASSIFICATIONS = (
    "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL",
    "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
)
NEXT_EXPERIMENT_IDS = (
    "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1",
    "PLAN_AWARE_MONOTONE_JEPA_COST_V1",
    "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
)
REQUIREMENTS_CLASSIFICATIONS = (
    "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED",
    "SIMULATED_CONTACT_PROXY_SCOPE_ONLY",
    "DEPLOYMENT_MATERIAL_HAZARD_SCOPE_UNRESOLVED",
    "PERSON_AND_FRAGILE_ASSET_HAZARDS_NOT_REPRESENTED",
    "RECOVERABILITY_REQUIREMENTS_UNRESOLVED",
    "MISSION_PROGRESS_REQUIREMENTS_PRESENT",
    "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO",
    "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO",
    "SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO",
    "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
    "REPLANNING_INTERFACE_UNRESOLVED",
)
REQUIREMENTS_STATEMENT = (
    "Deployment hard-contact requirements, consequences and recovery criteria remain "
    "unresolved. No further deployment-safety scope reduction, sensor qualification "
    "or learned hard-safety model is authorised."
)
NEXT_REQUIREMENTS_DECISION = "REQUIREMENTS_ACQUISITION_REQUIRED"

FAMILY_IDS = (
    "large_enclosed_maze",
    "medium_enclosed_maze",
    "small_enclosed_maze",
    "loop_alias_stress",
)
HARD_FAMILY_IDS = (
    "large_enclosed_maze",
    "loop_alias_stress",
)
HORIZON_IDS = ("H1", "H2", "H3")
PROHIBITION_COUNTER_IDS = (
    "training_steps",
    "fresh_states",
    "replacement_states",
    "replacement_candidates",
    "checkpoint_mutations",
    "checkpoint_selection_events",
    "untouched_g2_accesses",
    "stage_b_runs",
    "memory_system_runs",
    "experimental_candidate_selecting_navigation_runs",
    "routing_or_beacon_capture_runs",
)

ONE_STEP_CHECKPOINT_SHA256 = (
    "20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a"
)
TWO_STEP_CHECKPOINT_SHA256 = (
    "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4"
)
ENCODER_SHA256 = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "jepa_local_waypoint_planning_cost_qualification_v1"
)
TRACKED_PREREGISTRATION_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "preregistration_2026-08-26.md"
)
TRACKED_CONTRACT_RECEIPT_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "contract_2026-08-26.json"
)
TRACKED_OUTPUT_SCHEMA_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "output_schema_2026-08-26.json"
)
TRACKED_FIXTURE_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "fixture_2026-08-26.json"
)
TRACKED_GOAL_VIEW_AMENDMENT_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "goal_view_amendment_2026-08-26.json"
)
TRACKED_SOURCE_CLOSURE_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "source_closure_2026-08-26.json"
)
TRACKED_RESULT_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "result_2026-08-26.json"
)
TRACKED_REPORT_PATH = Path(
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
    "result_2026-08-26.md"
)
ENTRYPOINT_PATH = Path(
    "scripts/evaluate_jepa_local_waypoint_planning_cost_qualification_v1.py"
)


ORIGINAL_FREEZE_COMMIT = "184e192f35740a2b300a771097c2c5c8d68ce4f9"
GOAL_VIEW_RENDER_SEMANTICS = (
    "VIRTUAL_COUNTERFACTUAL_GOAL_VIEW_NOT_A_PHYSICALLY_EXECUTABLE_SENSOR_POSE"
)
GOAL_CELL_BLOCK_CLASSIFICATIONS = (
    "UNBLOCKED",
    "BEACON_ENDPOINT",
    "LOW_CLEARANCE_TRANSIT_BLOCKED",
)
GOAL_CELL_CLASSIFICATION_COUNTS: dict[str, int] = {
    "states": 48,
    "endpoint_reachable": 48,
    "nav_blocked": 14,
    "beacon_endpoint": 13,
    "low_clearance_transit_blocked": 1,
    "unblocked": 34,
}
GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS: dict[str, Any] = {
    "valid_node_ids": 48,
    "consecutive_route_edges_traversable": 48,
    "endpoint_reachable": 48,
    "nav_blocked_diagnostic_only": True,
    "path1_position_substitutions": 0,
    "state_drops_or_alternate_goals": 0,
    "fresh_execution_only": True,
    "pass": True,
}
GOAL_CELL_BLOCKED_STATE_IDS: dict[str, list[str]] = {
    "beacon_endpoint": [
        "purpose-13",
        "purpose-14",
        "purpose-18",
        "purpose-34",
        "purpose-36",
        "purpose-38",
        "purpose-39",
        "purpose-40",
        "purpose-41",
        "purpose-43",
        "purpose-45",
        "purpose-46",
        "purpose-47",
    ],
    "low_clearance_transit_blocked": ["purpose-44"],
}
GOAL_VIEW_STATIC_VALIDATION_SUCCESS: dict[str, Any] = {
    "states": 48,
    "valid_node_ids": 48,
    "consecutive_route_edges_traversable": 48,
    "endpoint_reachable": 48,
    "nav_blocked": 14,
    "beacon_endpoint": 13,
    "low_clearance_transit_blocked": 1,
    "unblocked": 34,
    "blocked_state_ids": copy.deepcopy(GOAL_CELL_BLOCKED_STATE_IDS),
    "optional_manifest_waypoint_xy_present": 22,
    "optional_manifest_waypoint_xy_exact_matches": 22,
    "path1_position_substitutions": 0,
    "state_drops_or_alternate_goals": 0,
    "route_outcome_rows_read": 0,
    "pass": True,
}

FAILED_GOAL_VIEW_ATTEMPT_ARCHIVE = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    ".jepa_local_waypoint_planning_cost_qualification_v1."
    "failed-1787772621443348948-337116"
)
FAILED_GOAL_VIEW_ATTEMPT_INVENTORY: dict[str, Any] = {
    "record_fields": ["path", "sha256", "bytes"],
    "record_order": "ascending archive-relative POSIX path",
    "aggregate_algorithm": (
        "SHA-256 of compact canonical JSON bytes of the ordered record array, with "
        "object keys sorted and without terminal LF"
    ),
    "record_count": 41,
    "total_bytes": 1365155,
    "canonical_records_bytes": 5594,
    "aggregate_sha256": (
        "6442c13416dc1badd811ac4b19a0fd802fad4708c3698c7dcfada44d552469bb"
    ),
}


FOUNDATIONAL_PACKAGE_CLOSURE_POLICY: dict[str, Any] = {
    "classification": "INTERPRETER_VERSION_ROOT_IMPORT_RESOLUTION_BOUND",
    "packages": ["torch", "numpy", "scipy", "pillow", "pyyaml"],
    "bound_fields": [
        "distribution",
        "version",
        "import_name",
        "package_root",
        "find_spec origin",
        "live module __file__",
        "submodule search locations",
    ],
    "import_resolution_rule": (
        "importlib.util.find_spec origin, every submodule search location and the "
        "live imported module __file__ must resolve inside the prospectively frozen "
        "package root; any PYTHONPATH or other shadow import fails closed"
    ),
    "record_or_recursive_file_byte_closure": False,
    "residual_limitation": (
        "same-version foundational package mutation is not byte-closed"
    ),
    "scope_boundary": (
        "do not recursively close additional transitive or system libraries; Genesis, "
        "rsl_rl and tensordict remain separately RECORD-byte-closed"
    ),
}


INTERPRETER_BINARY_BINDING: dict[str, Any] = {
    "path": "/usr/bin/python3.12",
    "sha256": "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118",
    "bytes": 8020928,
}


CPU_FOUNDATIONAL_PACKAGE_BINDINGS: dict[str, dict[str, str]] = {
    "torch": {
        "distribution": "torch",
        "version": "2.12.0",
        "import_name": "torch",
        "package_root": (
            "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
            "genesis_render_vulkan/lib/python3.12/site-packages/torch"
        ),
    },
    "numpy": {
        "distribution": "numpy",
        "version": "2.4.6",
        "import_name": "numpy",
        "package_root": (
            "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
            "genesis_render_vulkan/lib/python3.12/site-packages/numpy"
        ),
    },
    "scipy": {
        "distribution": "scipy",
        "version": "1.17.1",
        "import_name": "scipy",
        "package_root": (
            "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
            "genesis_render_vulkan/lib/python3.12/site-packages/scipy"
        ),
    },
    "pillow": {
        "distribution": "Pillow",
        "version": "11.3.0",
        "import_name": "PIL",
        "package_root": (
            "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
            "genesis_render_vulkan/lib/python3.12/site-packages/PIL"
        ),
    },
    "pyyaml": {
        "distribution": "PyYAML",
        "version": "6.0.3",
        "import_name": "yaml",
        "package_root": (
            "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
            "genesis_render_vulkan/lib/python3.12/site-packages/yaml"
        ),
    },
}


GPU_FOUNDATIONAL_PACKAGE_BINDINGS: dict[str, dict[str, str]] = {
    "torch": {
        "distribution": "torch",
        "version": "2.10.0.dev20250926+rocm6.3",
        "import_name": "torch",
        "package_root": "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages/torch",
    },
    "numpy": {
        "distribution": "numpy",
        "version": "2.4.2",
        "import_name": "numpy",
        "package_root": "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages/numpy",
    },
    "scipy": {
        "distribution": "scipy",
        "version": "1.18.0",
        "import_name": "scipy",
        "package_root": "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages/scipy",
    },
    "pillow": {
        "distribution": "Pillow",
        "version": "12.1.0",
        "import_name": "PIL",
        "package_root": "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages/PIL",
    },
    "pyyaml": {
        "distribution": "PyYAML",
        "version": "6.0.3",
        "import_name": "yaml",
        "package_root": "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages/yaml",
    },
}


DENSE_ROUTE_REPLAY_INPUT_BINDINGS: dict[str, Any] = {
    "record_count": 48,
    "total_bytes": 10155856,
    "canonical_records_bytes": 10397,
    "canonical_sorted_path_sha_bytes_aggregate_sha256": (
        "729271c8d8535d2f02433de03306ea8d113c654ecfb9f0671e6bd4e67f11183f"
    ),
    "record_order": "frozen state-manifest order (purpose-0 through purpose-47)",
    "record_fields": ["state_id", "path", "sha256", "bytes"],
    "path_rule": (
        ".generated/dense_temporal_true_future_safety_observability_v1/"
        "dense_replay/{state_id}.json"
    ),
    "aggregate_algorithm": (
        "SHA-256 of compact canonical JSON bytes of the frozen numeric manifest-order "
        "record array, with object keys sorted and without terminal LF; sorted in the "
        "field name refers to JSON object keys, never lexicographic row reordering"
    ),
    "outcome_fields_parsed_before_freeze": [],
    "validation_order": "path, bytes and SHA-256 must match before JSON parse",
}

SCENE_INPUT_BYTE_INVENTORY_BINDING: dict[str, Any] = {
    "record_count": 96,
    "states": 48,
    "total_bytes": 5361073,
    "canonical_records_bytes": 30690,
    "canonical_sorted_path_sha_bytes_aggregate_sha256": (
        "87f989d37b23f974c0c98924b65f676ebc23dcc24c6a3cfc70b087e339a09c4a"
    ),
    "record_order": (
        "frozen state-manifest order; within each state manifest then genesis_scene"
    ),
    "record_fields": ["state_id", "scene_id", "kind", "path", "sha256", "bytes"],
    "kind_ids": ["manifest", "genesis_scene"],
    "aggregate_algorithm": (
        "SHA-256 of compact canonical JSON bytes of the frozen state-manifest-order "
        "record array, with object keys sorted and without terminal LF; rows are never "
        "lexicographically reordered"
    ),
    "outcome_fields_parsed_before_freeze": [],
    "validation_order": "path, bytes and SHA-256 must match before JSON parse",
}


CPU_RUNTIME_INPUT_BINDINGS: dict[str, Any] = {
    "platform_manifest": {
        "path": "config/go2_platform_manifest.yaml",
        "sha256": "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189",
        "bytes": 4613,
    },
    "primitive_registry": {
        "path": "config/go2_primitive_registry.yaml",
        "sha256": "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8",
        "bytes": 2454,
    },
    "policy_artifacts": {
        "model": {
            "path": "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt",
            "sha256": "e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff",
            "bytes": 4547691,
            "tensor_open_before_contract_freeze": False,
        },
        "configuration": {
            "path": "models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl",
            "sha256": "bc3e68c18252475199e57b30c8ac49d813e3c784a3983e0e8b1a762490dde24f",
            "bytes": 2409,
            "deserialised_before_contract_freeze": False,
        },
    },
    "genesis_builtin_urdf": {
        "interpreter_relative_path": (
            ".generated/venvs/genesis_render_vulkan/lib/python3.12/site-packages/"
            "genesis/assets/urdf/go2/urdf/go2.urdf"
        ),
        "resolved_path": (
            "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
            "genesis_render_vulkan/lib/python3.12/site-packages/genesis/assets/"
            "urdf/go2/urdf/go2.urdf"
        ),
        "sha256": "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4",
        "bytes": 24170,
        "referenced_meshes": [
            {"name": "base.dae", "relative_path": "../dae/base.dae", "sha256": "e52bebbb5c7ff1f6aedee620c9b5b349f6e965360ecc63a9bb743eeff2d0be64", "bytes": 10727399},
            {"name": "calf.dae", "relative_path": "../dae/calf.dae", "sha256": "29872fa1a435b92bc2b395fdfa7f0aee26101db3c494681896bd4ef24ee7883c", "bytes": 1125517},
            {"name": "calf_mirror.dae", "relative_path": "../dae/calf_mirror.dae", "sha256": "79c66be7ed06760bd5001c067a1fb0ac517f639d8b1a5f89be471903b3f359e1", "bytes": 1125457},
            {"name": "foot.dae", "relative_path": "../dae/foot.dae", "sha256": "ffbd95fd641866ce9bd277aa249b92dafc9148d70131fa2c52709a35ce5b710b", "bytes": 506988},
            {"name": "hip.dae", "relative_path": "../dae/hip.dae", "sha256": "8735e10617afe252cef0844c1d139d15bc6d4c653660d742fea9fe05c2704d8f", "bytes": 4666421},
            {"name": "thigh.dae", "relative_path": "../dae/thigh.dae", "sha256": "a622381ae308897ff143b8bbf02f075457c342b22984205040424a5a983246fa", "bytes": 3884136},
            {"name": "thigh_mirror.dae", "relative_path": "../dae/thigh_mirror.dae", "sha256": "c7bae0b0565c2aa4b7cc15a538d795e9153e0f10c821cc18429e92693172bdc6", "bytes": 3896575},
        ],
        "mesh_reference_validation": (
            "parse the URDF and require its unique mesh filename set equals exactly "
            "the seven relative_path values before resolving and hashing every file"
        ),
    },
    "cpu_packages": {
        "genesis": {
            "distribution": "genesis-world",
            "version": "0.3.14",
            "package_root": (
                "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
                "genesis_render_vulkan/lib/python3.12/site-packages/genesis"
            ),
            "record_closure": {
                "record_path": (
                    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/"
                    "venvs/genesis_render_vulkan/lib/python3.12/site-packages/"
                    "genesis_world-0.3.14.dist-info/RECORD"
                ),
                "record_sha256": "f7d1ed9c57a0d2521b235ae71b0e0639d78d3f05f2253acc0cc3d0d9c34ab6a3",
                "record_bytes": 90641,
                "record_entries": 963,
                "declared_hash_entries": 739,
                "present_files": 852,
                "absent_unhashed_files": 111,
                "absent_unhashed_path_list_sha256": "937f05ba5c229d1e4b2d8904647b441f1585493e2bf5d00ad8868c9a94cb2353",
                "present_file_bytes": 222210609,
                "present_file_aggregate_sha256": "90a31491713deb9e3210ca57a1a52ed426d6a3542f6c36434477f9297844bd4a",
            },
        },
        "rsl_rl": {
            "distribution": "rsl-rl-lib",
            "version": "5.4.1",
            "package_root": (
                "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
                "genesis_render_vulkan/lib/python3.12/site-packages/rsl_rl"
            ),
            "record_closure": {
                "record_path": (
                    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/"
                    "venvs/genesis_render_vulkan/lib/python3.12/site-packages/"
                    "rsl_rl_lib-5.4.1.dist-info/RECORD"
                ),
                "record_sha256": "8ed2f98c70445a864a3f4bd6601a09f8056adfca1cb2e467ada29f61d53899d7",
                "record_bytes": 6983,
                "record_entries": 83,
                "declared_hash_entries": 52,
                "present_files": 53,
                "absent_unhashed_files": 30,
                "absent_unhashed_path_list_sha256": "d0746836d1b1b64e92b7273830d6be7a652fcb40825230cdd9b4013fc49993d8",
                "present_file_bytes": 289353,
                "present_file_aggregate_sha256": "d491f769cef8ac76b699eef4dea31912758c02d43610d6641e657b835888a65a",
            },
        },
        "tensordict": {
            "distribution": "tensordict",
            "version": "0.13.0",
            "package_root": (
                "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
                "genesis_render_vulkan/lib/python3.12/site-packages/tensordict"
            ),
            "record_closure": {
                "record_path": (
                    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/"
                    "venvs/genesis_render_vulkan/lib/python3.12/site-packages/"
                    "tensordict-0.13.0.dist-info/RECORD"
                ),
                "record_sha256": "c995c99883cc39d43ca3652f32a4c7d247140f8fa2fe58c1cb1ef47afea8039b",
                "record_bytes": 8531,
                "record_entries": 118,
                "declared_hash_entries": 65,
                "present_files": 66,
                "absent_unhashed_files": 52,
                "absent_unhashed_path_list_sha256": "c941478b79a4621d50119a04bd8bfb620a828db0be20edcd809033aeaac8bbbf",
                "present_file_bytes": 2552117,
                "present_file_aggregate_sha256": "eaf403adbd5a66c5bbb67dfd27b4896a5f1454c2ab81bf320cd6da2458938e30",
            },
        },
    },
    "foundational_packages": copy.deepcopy(CPU_FOUNDATIONAL_PACKAGE_BINDINGS),
    "foundational_package_closure_policy": copy.deepcopy(
        FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
    ),
    "package_record_closure_algorithm": {
        "base": "RECORD dist-info parent (site-packages)",
        "declared_hash_validation": (
            "all hash-bearing RECORD rows must exist and match the declared "
            "URL-safe-base64 digest and declared size"
        ),
        "unhashed_rows": (
            "hash every present unhashed row directly; absent unhashed paths are "
            "allowed only when their exact sorted path-list digest matches"
        ),
        "present_aggregate": (
            "sort records by RECORD path; canonical compact sorted-key JSON list "
            "of {path,sha256,bytes}; SHA-256 without terminal LF"
        ),
        "bytecode_writes": "forbidden with PYTHONDONTWRITEBYTECODE=1",
    },
    "scene_inventory": {
        "prefreeze_byte_inventory_binding": copy.deepcopy(
            SCENE_INPUT_BYTE_INVENTORY_BINDING
        ),
        "states": 48,
        "unique_scene_directories": 48,
        "files_per_scene": ["manifest.json", "genesis_scene.json"],
        "required_records": 48,
        "validation": (
            "for each frozen state hash both files and record path/SHA-256/bytes; "
            "parse the canonical manifest with lewm_worlds.manifest, recompute "
            "manifest_sha256, require it equals both files' declared manifest_sha256, "
            "and require identical state/scene identity; no outcome field is read"
        ),
    },
    "textures": {
        "environment_variable": "LEWM_TEXTURE_ROOT",
        "environment_variable_required_state": "ABSENT",
        "resolved_root": "assets/textures",
        "selection": (
            "for every scene persist exact select_scene_textures output from visual_seed "
            "and scene_id; only floor is reached by the frozen renderer, while wall and "
            "obstacle selections are explicitly non-executed"
        ),
        "records": [
            {"path": "assets/textures/floor/Concrete034.jpg", "sha256": "8ceb9186d990b31fce785e06fdc6974c160f1d27c43f6160266f164deba713aa", "bytes": 269501},
            {"path": "assets/textures/floor/PavingStones131.jpg", "sha256": "98644aad83a6d570c1b66e48ecad9211a73c320f5a53021a46d98bf453f041db", "bytes": 1804238},
            {"path": "assets/textures/floor/Tiles093.jpg", "sha256": "a86ba7e3fb8dacdda45fbe132fdae14bf6a1af9cc53b6fba2dfd434b5367ca0f", "bytes": 1059920},
            {"path": "assets/textures/floor/WoodFloor043.jpg", "sha256": "307c33f1fc5e05f47bfc513a1cca2b6f4f2b179d74d873dd874a24938889cd98", "bytes": 649557},
            {"path": "assets/textures/obstacle/Cardboard004.jpg", "sha256": "21236a873daeff5b25c772d38e785aee5d876bf75d037c5cec6d6c863bb7b462", "bytes": 1156430},
            {"path": "assets/textures/obstacle/Concrete036.jpg", "sha256": "8b618898cadda99188f3d667e4a09c2108f07bbc800cc1c0a4f300653b221adb", "bytes": 1158249},
            {"path": "assets/textures/obstacle/Metal055A.jpg", "sha256": "79938ccc3c75aaab3a2b1ed31068a8db3634d512286bb8c3aa6f6205cd59ca3a", "bytes": 813574},
            {"path": "assets/textures/obstacle/Wood067.jpg", "sha256": "91a758b06ee054a80735da449434da8b3b264c272e8c3f2763195551ef351076", "bytes": 1507799},
            {"path": "assets/textures/wall/Bricks097.jpg", "sha256": "218f7343091f1c0d9044fd642247d23dd322087271df1b45c8f51fdff91042dd", "bytes": 831880},
            {"path": "assets/textures/wall/Concrete045.jpg", "sha256": "408cd7f05194421e1884e40a087f7dd6e3ffa374f3176bbc5e443ac820796b88", "bytes": 878168},
            {"path": "assets/textures/wall/PaintedPlaster017.jpg", "sha256": "ed8a69a6437de9d41d2cf380f7e6a216232fc4754f256c3405985c857547ffef", "bytes": 378542},
            {"path": "assets/textures/wall/Plaster001.jpg", "sha256": "d3fe43282859ab7293ad41f2fa3c29f02df077e5bc326cf90a65cc588a4ac5a4", "bytes": 828790},
        ],
    },
    "box_obj_cache": {
        "path": ".generated/box_meshes",
        "execution_status": "NOT_REACHED_BY_FROZEN_HISTORICAL_RENDERER",
        "scientific_runtime_input": False,
        "files_opened_or_used": 0,
        "validation_required_for_execution": False,
        "historical_scene_derived_nonexecuted_records": 176,
        "reason": (
            "the frozen caller supplies genesis_scene objects while the historical "
            "builder reads walls/obstacles/landmarks, so add_box and cached_box_obj "
            "are never reached"
        ),
        "regeneration_during_scientific_materialisation": False,
    },
}

HISTORICAL_RENDERER_LIMITATIONS: dict[str, Any] = {
    "effective_scene_geometry": "FLOOR_PLANE_ONLY",
    "input_schema": "genesis_scene.json with structural geometry under objects",
    "builder_schema": (
        "scripts.render_replay_v03.build_scene reads walls, obstacles and landmarks"
    ),
    "structural_walls_obstacles_landmarks_rendered": False,
    "current_true_future_byte_compatibility_preserved": True,
    "explicit_wall_visual_reasoning_claim": False,
    "interpretation": "HISTORICAL_RENDERER_LATENT_ROUTE_RANKING_ONLY",
}

RECONSTRUCTION_PREFIX_CUSTODY: dict[str, Any] = {
    "classification": "FROZEN_STATE_RECONSTRUCTION_REPLAY",
    "driver": "production collector scheduler/RouteTeacher plus frozen PPO",
    "states": 48,
    "blocks_per_state": 40,
    "total_blocks": 1920,
    "physics_frames_per_block": 250,
    "total_physics_frames": 480000,
    "snapshot_reproductions": 48,
    "frozen_state_reconstruction_route_teacher_ppo_blocks": 1920,
    "frozen_state_reconstruction_states": 48,
    "purpose": (
        "deterministic reconstruction of the frozen post-block-40 branch-ledger snapshot"
    ),
    "experimental_candidate_selection": False,
    "jepa_cost_actions_executed": 0,
    "navigation_qualification": False,
}

CONTROLLER_EXECUTION_CUSTODY: dict[str, Any] = {
    "classification": "FROZEN_CONTROLLER_REPLAY_AND_FIXED_ORACLE_FANOUT_ONLY",
    "frozen_ppo_controller_total_blocks": 6240,
    "frozen_ppo_controller_total_physics_frames": 1560000,
    "reconstruction_route_teacher_ppo_blocks": 1920,
    "reconstruction_route_teacher_ppo_physics_frames": 480000,
    "fixed_oracle_fanout_ppo_blocks": 4320,
    "fixed_oracle_fanout_ppo_physics_frames": 1080000,
    "experimental_candidate_selecting_jepa_mpc_navigation_planner_executions": 0,
    "candidate_or_state_selection_changes": 0,
    "navigation_system_training_or_qualification": False,
    "reporting_rule": (
        "disclose both positive frozen-controller uses and state that no navigation "
        "system was trained, evaluated or run as the experimental candidate-selecting "
        "planner; never claim that no controller or navigation code executed"
    ),
}

EXECUTION_WATCHDOGS: dict[str, Any] = {
    "cpu_worker_environment": {
        "MALLOC_ARENA_MAX": "1",
        "workers": 32,
        "dynamic_worker_fallback": False,
        "purpose": "allocator fragmentation mitigation only",
        "scientific_semantics_change": False,
    },
    "cpu_worker_no_progress_timeout_s": 3600,
    "cpu_materialization_global_timeout_s": 10800,
    "gpu_preflight_timeout_s": 600,
    "gpu_materialization_timeout_s": 10800,
    "gpu_check_timeout_s": 600,
    "timeout_action": (
        "terminate then kill all experiment children, archive the untouched hidden "
        "attempt with a failure receipt, and leave the canonical output absent"
    ),
    "automatic_retries": 0,
    "partial_phase_or_shard_resume": False,
}

CPU_WATCHDOG_STATUS_SUCCESS: dict[str, Any] = {
    "no_progress_timeout_s": 3600,
    "global_timeout_s": 10800,
    "workers": 32,
    "completed_workers": 48,
    "no_progress_timeout_breaches": 0,
    "global_timeout_breaches": 0,
    "terminate_signals": 0,
    "kill_signals": 0,
    "automatic_retries": 0,
    "resume_used": False,
    "pass": True,
}

GPU_WATCHDOG_STATUS_SUCCESS: dict[str, Any] = {
    "phase": "GPU_MATERIALIZATION",
    "timeout_s": 10800,
    "timed_out": False,
    "automatic_retries": 0,
    "resume_used": False,
    "pass": True,
}

EXECUTION_WATCHDOG_STATUS_SUCCESS: dict[str, Any] = {
    "configuration": copy.deepcopy(EXECUTION_WATCHDOGS),
    "gpu_preflight": {
        "phase": "GPU_PREFLIGHT",
        "timeout_s": 600,
        "timed_out": False,
        "pass": True,
    },
    "cpu_materialization": copy.deepcopy(CPU_WATCHDOG_STATUS_SUCCESS),
    "gpu_materialization": copy.deepcopy(GPU_WATCHDOG_STATUS_SUCCESS),
    "terminal_check_excluded_from_embedded_result": True,
    "automatic_retries": 0,
    "resume_used": False,
    "pass": True,
}


class ContractError(ValueError):
    """Raised when a frozen contract or receipt fails closed."""


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{location} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{location} contains a non-string JSON key")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ContractError(f"{location} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON bytes, without a trailing newline."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying the digest of the object without that field."""

    payload = copy.deepcopy(dict(value))
    payload.pop("content_digest", None)
    payload["content_digest"] = canonical_json_sha256(payload)
    return payload


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    declared = payload.pop("content_digest", None)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ContractError("content_digest is missing or malformed")
    if canonical_json_sha256(payload) != declared:
        raise ContractError("content_digest mismatch")
    return copy.deepcopy(dict(value))


def _goal_view_execution_amendment_core() -> dict[str, Any]:
    return {
        "schema": GOAL_VIEW_AMENDMENT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "date": "2026-08-26",
        "status": "PROSPECTIVE_BEFORE_FRESH_REEXECUTION",
        "original_freeze": {
            "commit": ORIGINAL_FREEZE_COMMIT,
            "contract": {
                "path": str(TRACKED_CONTRACT_RECEIPT_PATH),
                "sha256": "719544317108e9ae8f93972187a6da9feac4b7f659b3aa03bcc5dcfd76043f42",
                "bytes": 59563,
                "digest_field": "contract_sha256",
                "content_digest": "88b1dc179dc1c7dc7f74cff1c89e1631a81b78aad2a1bcf72398d51d6bd405ae",
            },
            "output_schema": {
                "path": str(TRACKED_OUTPUT_SCHEMA_PATH),
                "sha256": "e83027a9847bcc364c1e2cf69150e1cfe169a71cdba25fdf88e0d44ab6fa0d28",
                "bytes": 34798,
                "digest_field": "output_schema_sha256",
                "content_digest": "2b2d85bfa21ec5b2d6f136a9e1ba5b798cf39c30736c1033a37276c4ba03b59f",
            },
            "fixture": {
                "path": str(TRACKED_FIXTURE_PATH),
                "sha256": "90c1d9320377ed88dc94cfdea8de3ce4c7c5ca139c8e0ed992397c0b2230038d",
                "bytes": 7280,
                "digest_field": "content_digest",
                "content_digest": "5e644abc0b538ad857543bcadc995f274b53d3230498d790b4c40f59cb85928a",
            },
            "source_closure": {
                "path": str(TRACKED_SOURCE_CLOSURE_PATH),
                "sha256": "e72f8545ca5a408ded829cd0503692ee57fcd04a4a0e954aebe9b8691b43b9bc",
                "bytes": 12927,
                "digest_field": "content_digest",
                "content_digest": "2e79c122098b4805e88804aa2cadff9a939515dcd962a050273f8093c59fed6e",
                "rows": 82,
            },
        },
        "failed_attempt": {
            "archive_path": str(FAILED_GOAL_VIEW_ATTEMPT_ARCHIVE),
            "failure_receipt": {
                "path": "receipts/failure.json",
                "sha256": "63aad5c1d1a2e8902088cff647465be7c17ef56b4d12705b42ee0135cdd925a0",
                "bytes": 43448,
                "content_digest": "fb1b810fff46781243c0d0d36e4d3520789dd045f4f27fa6f353f43da51dbe64",
                "phase": "MATERIALIZATION",
                "error_type": "QualificationError",
                "terminal_error": (
                    "purpose-18: frozen path[2] goal cell is not free and reachable"
                ),
                "partial_artifacts_reusable": False,
                "nothing_running": True,
            },
            "archive_inventory": copy.deepcopy(FAILED_GOAL_VIEW_ATTEMPT_INVENTORY),
            "artifact_summary": {
                "worker_logs": 32,
                "partial_context_rgb_files": 3,
                "partial_context_rgb_state_ids": ["purpose-18"],
                "preexecution_or_environment_receipts": 3,
                "cpu_runtime_input_inventory_receipts": 1,
                "failure_or_running_marker_receipts": 2,
            },
            "terminal_absences": [
                "materialization/context_reconstruction_index.json",
                "materialization/dense_route_replay_input_index.json",
                "materialization/oracle_admissibility_fanout_index.json",
                "goal_views/index.json",
                "latents/tensor_index.json",
                "receipts/gpu_inference.json",
                "evidence/candidate_evidence.jsonl.gz",
                "evidence/selection_evidence.jsonl.gz",
                "evidence/paired_effect_evidence.jsonl.gz",
                "aggregates/metrics.json",
                "receipts/persistence.json",
                "result.json",
                "report.md",
            ],
            "scientific_result_published": False,
            "aggregate_metrics_or_gates_computed": False,
            "gpu_predictor_inference_executed": False,
            "canonical_output_root_absent": True,
            "scientific_phase_or_shard_reuse": False,
        },
        "static_diagnosis": {
            "sources": [
                "frozen state manifest",
                "frozen scene manifests",
                "SceneGraph endpoint semantics",
                "purpose-built waypoint collector source",
                "Route-Intent V2 input source",
            ],
            "route_outcome_rows_read_or_used": 0,
            "checkpoint_tensors_opened": 0,
            "predictor_inference_calls": 0,
            "finding": (
                "the frozen local waypoint is exactly cell_center(waypoint_path_cells[2]); "
                "a SceneGraph nav-blocked cell can be a reachable route endpoint"
            ),
            "goal_cell_classification_counts": copy.deepcopy(
                GOAL_CELL_CLASSIFICATION_COUNTS
            ),
            "blocked_state_ids": copy.deepcopy(GOAL_CELL_BLOCKED_STATE_IDS),
            "all_optional_manifest_waypoint_xy_fields_match_path2": True,
            "optional_manifest_waypoint_xy_fields_present": 22,
        },
        "amended_goal_pose_semantics": {
            "position_world_xy": "exact frozen SceneGraph cell_center(waypoint_path_cells[2])",
            "position_world_z": "exact frozen snapshot base z",
            "yaw": "atan2 from cell_center(path[0]) to cell_center(path[1])",
            "roll_rad": 0.0,
            "pitch_rad": 0.0,
            "candidate_independent": True,
            "goal_render_semantics": GOAL_VIEW_RENDER_SEMANTICS,
            "endpoint_reachability_rule": (
                "SceneGraph.bfs_distance(path[0], path[2], "
                "transit_blocked=nav_blocked_cells) is not None; a blocked goal may be "
                "reached as an endpoint but is never asserted free or transit-safe"
            ),
            "block_classification_precedence": (
                "BEACON_ENDPOINT when path[2] is a beacon cell; otherwise "
                "LOW_CLEARANCE_TRANSIT_BLOCKED when path[2] is nav-blocked; otherwise UNBLOCKED"
            ),
            "block_classification_ids": list(GOAL_CELL_BLOCK_CLASSIFICATIONS),
            "physical_executability_claim": False,
            "robot_reachability_or_stopping_claim": False,
            "physical_sensor_pose_claim": False,
            "counterfactual_render_only": True,
            "historical_floor_plane_only_renderer_limitation_preserved": True,
        },
        "prohibitions": {
            "path1_position_substitution": True,
            "alternate_standoff_or_goal_search": True,
            "state_drop_or_replacement": True,
            "candidate_role_label_or_route_outcome_change": True,
            "outcome_use_for_amendment": True,
            "partial_attempt_reuse": True,
            "training_or_checkpoint_change": True,
            "untouched_g2_or_stage_b_access": True,
        },
        "execution_lifecycle": {
            "commit_strategy": (
                "retain the original freeze commit as an ancestor because it binds the "
                "archived failed attempt, then create one separate prospective "
                "goal-view contract-correction commit before fresh preflight"
            ),
            "original_freeze_commit_remains_ancestor": True,
            "new_correction_commit_is_future_source_authority": True,
            "result_commit_message_remains": (
                "Evaluate JEPA local waypoint planning cost qualification"
            ),
            "new_hidden_attempt_namespace": True,
            "canonical_output_root_must_be_absent": True,
            "prior_state_phase_or_shard_reuse": False,
            "automatic_retry": False,
            "fresh_preflight_required": True,
            "fresh_complete_execution_required": True,
        },
    }


def build_goal_view_execution_amendment() -> dict[str, Any]:
    return attach_content_digest(_goal_view_execution_amendment_core())


def goal_view_execution_amendment_receipt_bytes() -> bytes:
    return canonical_json_bytes(build_goal_view_execution_amendment()) + b"\n"


GOAL_VIEW_EXECUTION_AMENDMENT = build_goal_view_execution_amendment()
GOAL_VIEW_EXECUTION_AMENDMENT_BINDING: dict[str, Any] = {
    "path": str(TRACKED_GOAL_VIEW_AMENDMENT_PATH),
    "sha256": hashlib.sha256(goal_view_execution_amendment_receipt_bytes()).hexdigest(),
    "bytes": len(goal_view_execution_amendment_receipt_bytes()),
    "content_digest": GOAL_VIEW_EXECUTION_AMENDMENT["content_digest"],
}


def _self_digest(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(key, None)
    payload[key] = canonical_json_sha256(payload)
    return payload


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def tokenwise_cosine_cost(
    candidate_tokens: Sequence[Sequence[float]],
    goal_tokens: Sequence[Sequence[float]],
    *,
    expected_tokens: int = 768,
    expected_width: int = 1024,
) -> float:
    """Exact reference reduction for the frozen raw latent goal cost.

    This small payload-free reference mirrors the frozen reduction: values are
    round-tripped through IEEE FP16 storage, loaded as FP32, layer-normalised
    across the last dimension with epsilon 1e-5, then independently L2-
    normalised with epsilon 1e-12.  Matching token dot products are formed in
    FP32 and their ``1-cosine`` values are averaged with a float64 accumulator.
    This dependency-free helper is used only for small outcome-free contract
    fixtures.  Scientific rows, aggregates, and terminal replay use the one
    source-closed NumPy reducer frozen in
    ``jepa_local_waypoint_planning_cost_metrics_v1`` so floating-point reduction
    order is never ambiguous.  The GPU process only encodes, predicts, and
    persists FP16 tensors.
    """

    if len(candidate_tokens) != expected_tokens or len(goal_tokens) != expected_tokens:
        raise ContractError("token count does not match the frozen 768-token contract")
    costs: list[float] = []
    for token_index, (candidate, goal) in enumerate(zip(candidate_tokens, goal_tokens)):
        if len(candidate) != expected_width or len(goal) != expected_width:
            raise ContractError(
                f"token {token_index} width does not match the frozen 1024-wide contract"
            )
        candidate32 = [_fp32_from_persisted_fp16(value) for value in candidate]
        goal32 = [_fp32_from_persisted_fp16(value) for value in goal]
        if not all(math.isfinite(value) for value in candidate32 + goal32):
            raise ContractError(f"token {token_index} contains non-finite values")
        candidate_ln = _layer_norm_fp32(candidate32)
        goal_ln = _layer_norm_fp32(goal32)
        candidate_unit = _l2_normalise_fp32(candidate_ln)
        goal_unit = _l2_normalise_fp32(goal_ln)
        cosine = _fp32(
            math.fsum(_fp32(a * b) for a, b in zip(candidate_unit, goal_unit))
        )
        cosine = min(1.0, max(-1.0, cosine))
        if not math.isfinite(cosine):
            raise ContractError(f"token {token_index} cosine is non-finite")
        costs.append(1.0 - cosine)
    return math.fsum(costs) / float(expected_tokens)


def _fp32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _fp32_from_persisted_fp16(value: float) -> float:
    half = struct.unpack("<e", struct.pack("<e", float(value)))[0]
    return _fp32(half)


def _layer_norm_fp32(token: Sequence[float]) -> list[float]:
    width = len(token)
    mean = _fp32(math.fsum(float(value) for value in token) / float(width))
    centred = [_fp32(float(value) - mean) for value in token]
    variance = _fp32(
        math.fsum(_fp32(value * value) for value in centred) / float(width)
    )
    inverse_std = _fp32(1.0 / math.sqrt(_fp32(variance + 1e-5)))
    return [_fp32(value * inverse_std) for value in centred]


def _l2_normalise_fp32(token: Sequence[float]) -> list[float]:
    norm_sq = _fp32(math.fsum(_fp32(value * value) for value in token))
    denominator = max(_fp32(math.sqrt(max(norm_sq, 0.0))), 1e-12)
    return [_fp32(float(value) / denominator) for value in token]


def route_heading_yaw(
    first_path_cell_center_xy: Sequence[float],
    second_path_cell_center_xy: Sequence[float],
) -> float:
    """Frozen goal-view yaw from path cell 0 to path cell 1."""

    if len(first_path_cell_center_xy) != 2 or len(second_path_cell_center_xy) != 2:
        raise ContractError("path-cell centres must each contain exactly x and y")
    x0, y0 = (float(value) for value in first_path_cell_center_xy)
    x1, y1 = (float(value) for value in second_path_cell_center_xy)
    values = (x0, y0, x1, y1)
    if not all(math.isfinite(value) for value in values):
        raise ContractError("path-cell centres contain a non-finite value")
    dx, dy = x1 - x0, y1 - y0
    if dx == 0.0 and dy == 0.0:
        raise ContractError("path cells 0 and 1 have identical centres")
    return math.atan2(dy, dx)


def family_complete_collapse(family_metric: Mapping[str, Any]) -> bool:
    """Apply the prospectively frozen family-collapse definition."""

    evaluable_states = int(family_metric.get("evaluable_nonabstaining_states", 0))
    ordered_pairs = int(family_metric.get("ordered_pairs", 0))
    if evaluable_states < 1 or ordered_pairs < 1:
        return True
    pairwise = float(family_metric["pairwise_accuracy"])
    top3 = float(family_metric["best_route_top3_rate"])
    progress = float(family_metric["selected_route_progress_sum_m"])
    if not all(math.isfinite(value) for value in (pairwise, top3, progress)):
        raise ContractError("family route signal contains a non-finite value")
    return not (pairwise > 0.5 or top3 > 0.0 or progress > 0.0)


def no_family_complete_collapse(
    family_metrics: Mapping[str, Mapping[str, Any]],
) -> bool:
    if set(family_metrics) != set(FAMILY_IDS):
        raise ContractError("family metrics must contain exactly the four frozen families")
    return all(not family_complete_collapse(family_metrics[name]) for name in FAMILY_IDS)


def derive_incremental_route_value_secondary(
    *,
    progress_delta_m: float,
    progress_bootstrap_ci95_lower_m: float,
    normalized_regret_improvement: float,
    regret_bootstrap_ci95_lower: float,
    hard_family_progress_ratio_delta: float,
    hard_family_bootstrap_ci95_lower: float,
    contact_selection_increase: int,
    nonviable_selection_increase: int,
    hard_family_collapse: bool,
) -> bool:
    """Outcome-independent rule for the sole optional secondary class."""

    numeric = (
        progress_delta_m,
        progress_bootstrap_ci95_lower_m,
        normalized_regret_improvement,
        regret_bootstrap_ci95_lower,
        hard_family_progress_ratio_delta,
        hard_family_bootstrap_ci95_lower,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ContractError("paired materiality input is non-finite")
    progress_trigger = (
        float(progress_delta_m) >= 0.05
        and float(progress_bootstrap_ci95_lower_m) > 0.0
    )
    regret_trigger = (
        float(normalized_regret_improvement) >= 0.05
        and float(regret_bootstrap_ci95_lower) > 0.0
    )
    hard_family_trigger = (
        float(hard_family_progress_ratio_delta) >= 0.10
        and float(hard_family_bootstrap_ci95_lower) > 0.0
    )
    safety_preserved = (
        int(contact_selection_increase) <= 0
        and int(nonviable_selection_increase) <= 0
        and not bool(hard_family_collapse)
    )
    return safety_preserved and (progress_trigger or regret_trigger or hard_family_trigger)


def derive_primary_classification(
    *,
    true_future_gate_passed: bool,
    two_step_gate_passed: bool,
    kinematic_baseline_materially_superior: bool,
    jepa_incremental_route_value: bool,
) -> str:
    """Exact outcome-independent primary-classification precedence."""

    if not true_future_gate_passed:
        return "RAW_LATENT_GOAL_COST_NO_GO"
    if (
        two_step_gate_passed
        and kinematic_baseline_materially_superior
        and not jepa_incremental_route_value
    ):
        return "KINEMATIC_BASELINE_DOMINANT"
    if two_step_gate_passed:
        return "TWO_STEP_JEPA_PLANNING_COST_SIGNAL"
    return "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO"


STATIC_FILE_BINDINGS: dict[str, dict[str, Any]] = {
    "platform_manifest": {
        "path": "config/go2_platform_manifest.yaml",
        "sha256": "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189",
        "bytes": 4613,
    },
    "renderer": {
        "path": "lewm/oracle/go2_textured_v03_renderer.py",
        "sha256": "392439be92c128f639c8c9682627530b34660168229c24c9944d847372524aba",
    },
    "encoder_driver": {
        "path": "scripts/dev_frozen_dense_representation_encoders_v1.py",
        "sha256": "c5bb12ddc4711071dbdbac8c2ad6cc4b7528dd8ceb263b752fd539bd954aa9e2",
    },
    "predictor_contract": {
        "path": "lewm/oracle/go2_rgb_control_history_four_step_autoregressive_v1_contract.py",
        "sha256": "84ddc01a65f15a4e15bbd0fefd2cf13d91e9d072c11948476fb7570a55e05fe2",
    },
    "predictor_implementation": {
        "path": "scripts/dev_proprio_predictor_v1.py",
        "sha256": "04e3b140727f3c3c661416940cca40bcdc3925d943c3c90628516e35da43ada0",
    },
    "action_slew": {
        "path": "scripts/dev_action_slew_reconstruction_v1.py",
        "sha256": "17075cc10bdfc637a630da1b495f064156be9d481b6be631f50fd1e370b9203e",
    },
    "two_step_rollout": {
        "path": "scripts/run_dev_v03_two_step_rollout_v1.py",
        "sha256": "03c2621c9d11d5741a5587ae982b20de140182f0e68d5c33027486cd48b47879",
    },
    "temporal_action_jepa": {
        "path": "scripts/run_dev_v03_temporal_action_jepa_v1.py",
        "sha256": "06e92ba7301ef710a68c4e16645f38951ccc2d10b07d97bb7dc1fdbda948c949",
    },
    "panel_collector": {
        "path": "scripts/collect_safe_local_waypoint_purpose_built_v1.py",
        "sha256": "0f9f71247c97abbe2f464c32167164661731e4e920c652ed0a168888d4f52e88",
    },
    "true_future_materializer": {
        "path": "scripts/materialize_dense_route_intent_true_future_v1.py",
        "sha256": "4b98abdac2d3a329f6a9f135794145d1362977895b8adb97fcd84fa24f5a3f00",
    },
    "route_intent_encoder": {
        "path": "scripts/encode_safe_local_waypoint_route_intent_v2.py",
        "sha256": "48eed8def0566338fe5dce5a91c03b346869bdef9752a313994d62c44db5d708",
    },
    "route_replay": {
        "path": "scripts/replay_safe_local_waypoint_route_intent_v2.py",
        "sha256": "40365ffe3b785aab4dcd255694952a32392f558a919621477f7213ba5c6e3c69",
    },
    "kinematic_route_baseline": {
        "path": "scripts/run_kinematic_route_with_runtime_safety_guard_v1.py",
        "sha256": "b8361d5ef9debdfc1b50b327940984affa46720eafaf771d2c51d878cc0cf027",
    },
    "prior_rollout_result": {
        "path": "docs/lewm_go2_v03_horizon_rollout_result_2026-08-09.md",
        "sha256": "8d9bfb69a97af903b99dce0e71c89f1895b5e54c46fee010c0bcb23a5c0f951c",
    },
    "predictor_claims_matrix": {
        "path": "docs/lewm_counterfactual_predictor_claims_matrix_2026-08-18.md",
        "sha256": "57e230bb066ac5217a03b1e75c5a7c91d7161b2dafa6105ba808e4d20219b7d7",
    },
    "contact_hazard_ontology": {
        "path": "lewm/safety/contact_hazard_ontology_v1.py",
        "sha256": "69550fe787e84331560013678abed3aab58f573719b7198c53bfef786fb6204b",
    },
    "control_commitment_viability": {
        "path": "lewm/safety/control_commitment_horizon_and_viability_v1.py",
        "sha256": "a9a0ec8874c962c2c28858c7fa7e5839ec3e2da7d9b3c5d0ee4d138dbf22e688",
    },
    "one_tick_viability": {
        "path": "lewm/safety/one_tick_viability_constrained_mpc_v1.py",
        "sha256": "e02fd1608ffca9d5282584b89ee8753e73f311ba3bf67d7c8e0f36daa3e2cf51",
    },
    "oracle_branch_pilot_v1": {
        "path": "scripts/run_go2_oracle_branch_pilot_v1.py",
        "sha256": "19cf29132d4d9c3c8a6f3630bc243caa62a27e5f3616a5598fa487775ef10955",
    },
    "oracle_branch_pilot_v1_2": {
        "path": "scripts/run_go2_oracle_branch_pilot_v1_2.py",
        "sha256": "03307d6718471b6a4358d0952ff24cb1dd67a4a08ba0ea3258ed2e5376e0c889",
    },
    "branch_oracle_v1_2": {
        "path": "lewm/oracle/go2_branch_oracle_v1_2.py",
        "sha256": "6d7a6b20bcfb5da112ff10e95a7d3573ebf07884e7b4e58315a733254d6f4fc2",
    },
    "contact_hazard_instrumentation": {
        "path": "scripts/instrument_contact_hazard_ontology_v1.py",
        "sha256": "9f68151337d92ba121217f9004930767d7e400b53a96d60d1d74c21187a318d9",
    },
    "primitive_registry": {
        "path": "config/go2_primitive_registry.yaml",
        "sha256": "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8",
    },
    "predictor_inference_precision_authority": {
        "path": "scripts/benchmark_one_tick_observation_prediction_control_loop_v1.py",
        "sha256": "b216355da52ecd2339dc1ff6475eb2229cb4ef5f54c49b25f91cbcd5123c6d81",
    },
    "temporal_sequence_authority": {
        "path": "scripts/build_dev_v03_temporal_sequences_v1.py",
        "sha256": "b99ff1c9c05ce08d5af3574040a0f29aadc94ccb0f2462d510503991d4c15c75",
    },
    "requirements_review": {
        "path": "docs/lewm_protected_contact_scope_requirements_review_v1.md",
        "sha256": "042e9487ea94d8cf7974ec008eabe7a38dd43e3ff4ab06a016ac919f06120814",
    },
    "requirements_traceability": {
        "path": "docs/lewm_protected_contact_scope_traceability_matrix_v1.md",
        "sha256": "b3adf20a3cc506e79bbbdbd7a24d086e2286034fd8e82baca69e6802a89037c7",
    },
    "requirements_decision": {
        "path": "docs/lewm_protected_contact_scope_decision_memo_v1.md",
        "sha256": "a1a8540413ba67dd7870dd0aae9b2ce41842553179a2d6278dc2a68b6432dc61",
    },
    "requirements_assurance": {
        "path": "docs/lewm_protected_contact_scope_assurance_fragment_v1.md",
        "sha256": "4e9739c6b9c42a0a50a824358624091d45bdb7f9e0d1a8792920cb9e0ba487aa",
    },
    "protected_scope_authoritative_result": {
        "path": "docs/lewm_protected_contact_scope_requirements_review_v1_result.json",
        "sha256": "c348d5e2d265a118922ae138c549d8a6e48e4e900a45b70d919de4aa530e4027",
        "content_digest": "148a1757f4b8d55291ab38010a2dd0701e4606d61cd4c656de78cff571dac948",
    },
}


def _contract_core() -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "date": "2026-08-26",
        "starting_head": STARTING_HEAD,
        "status": "PROSPECTIVE_NOT_EXECUTED",
        "mode": "EVALUATION_FIRST_SINGLE_SEED_DEVELOPMENT",
        "scientific_question": (
            "Can frozen true-future or frozen one-/two-step JEPA latent states, "
            "ranked only by an untuned goal-view cosine cost, recover useful local "
            "waypoint planning order on the frozen Route-Intent V2 panel?"
        ),
        "claims_boundary": {
            "development_only": True,
            "one_seed": SEED,
            "generalization_claim": False,
            "closed_loop_claim": False,
            "deployment_safety_claim": False,
            "oracle_populations_are_evaluation_strata_only": True,
            "true_future_is_an_observability_upper_bound_not_a_predictable_input": True,
            "requirements_statement": REQUIREMENTS_STATEMENT,
            "historical_renderer_limitations": copy.deepcopy(
                HISTORICAL_RENDERER_LIMITATIONS
            ),
            "goal_view_render_semantics": GOAL_VIEW_RENDER_SEMANTICS,
            "goal_view_physical_executability_or_reachability_claim": False,
        },
        "goal_view_execution_amendment": {
            "receipt_binding": copy.deepcopy(GOAL_VIEW_EXECUTION_AMENDMENT_BINDING),
            "original_freeze_commit": ORIGINAL_FREEZE_COMMIT,
            "failed_attempt_archive": str(FAILED_GOAL_VIEW_ATTEMPT_ARCHIVE),
            "goal_pose_semantics": copy.deepcopy(
                GOAL_VIEW_EXECUTION_AMENDMENT["amended_goal_pose_semantics"]
            ),
            "goal_cell_classification_counts": copy.deepcopy(
                GOAL_CELL_CLASSIFICATION_COUNTS
            ),
            "goal_cell_classification_validation_success": copy.deepcopy(
                GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS
            ),
            "preworker_static_validation_success": copy.deepcopy(
                GOAL_VIEW_STATIC_VALIDATION_SUCCESS
            ),
            "blocked_state_ids": copy.deepcopy(GOAL_CELL_BLOCKED_STATE_IDS),
            "fresh_execution_only": True,
            "prior_phase_or_shard_reuse": False,
        },
        "preexecution_custody": {
            "outcome_barrier": (
                "No route-outcome table, row ledger, result metric, checkpoint tensor, "
                "or predictor inference may be used to derive or tune this contract."
            ),
            "status": "ACCIDENTAL_EXPOSURES_DISCLOSED_AND_EXCLUDED",
            "authorized_outcome_fields_for_contract_derivation": [],
            "all_final_rules_authority": "user literals and static metadata only",
            "accidental_exposures": [
                {
                    "actor": "final_contract_audit",
                    "command": (
                        "sed -n 1,180p docs/"
                        "lewm_safe_local_waypoint_planner_route_intent_v2_result_2026-08-20.md"
                    ),
                    "path": (
                        "docs/lewm_safe_local_waypoint_planner_route_intent_v2_"
                        "result_2026-08-20.md"
                    ),
                    "exposure": "detailed held-out-state outcome table",
                    "values_retained_or_used": False,
                    "disposition": "EXCLUDED_FROM_ALL_CONTRACT_GATE_GOAL_AND_MATERIALITY_DECISIONS",
                },
                {
                    "actor": "final_contract_audit",
                    "command_pattern": (
                        "rg -n PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED|"
                        "REQUIREMENTS_ACQUISITION_REQUIRED|SIMULATED_CONTACT_PROXY_SCOPE_ONLY|"
                        "secondary_classification docs/lewm_protected_contact_scope_* "
                        "lewm/safety scripts"
                    ),
                    "path": "docs/lewm_protected_contact_scope_requirements_review_v1_result.json",
                    "exposure": "truncated Stage-A diagnostic outcome values from a one-line JSON match",
                    "values_retained_or_used": False,
                    "disposition": "EXCLUDED_FROM_THIS_EXPERIMENT_AND_ITS_RULES",
                },
                {
                    "actor": "root",
                    "command_pattern": (
                        "rg -n commit|prefix|H3|execute|candidate bank|replan "
                        "docs/lewm_safe_local_waypoint*"
                    ),
                    "path": (
                        "docs/lewm_safe_local_waypoint_planner_route_intent_v2_"
                        "result_2026-08-20.md"
                    ),
                    "exposure": "one aggregate result line containing unsafe/safe H3 branch/state counts",
                    "values_retained_or_used": False,
                    "rules_already_fixed_independently": [
                        "token cost",
                        "goal view",
                        "paired materiality",
                    ],
                    "disposition": "EXCLUDED_FROM_ALL_CONTRACT_AND_GATE_DECISIONS",
                },
            ],
            "required_preexecution_assertions": {
                "tracked_contract_and_schema_frozen": True,
                "source_closure_frozen": True,
                "fixture_gate_passed": True,
                "canonical_output_root_fresh": True,
                "checkpoint_tensors_opened_before_freeze": 0,
                "predictor_inference_before_freeze": 0,
                "outcome_values_used_for_contract_derivation": 0,
                "untouched_g2_reads": 0,
                "training_steps": 0,
                "goal_view_execution_amendment_bound": True,
                "archived_failed_attempt_reuse": 0,
            },
        },
        "frozen_panel": {
            "name": "SAFE_LOCAL_WAYPOINT_ROUTE_INTENT_V2",
            "state_manifest": {
                "path": ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json",
                "sha256": "da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37",
            },
            "split": {
                "path": ".generated/safe_local_waypoint_purpose_built_v1/split.json",
                "sha256": "ebef7db828a4c754432375818fd6b1eff0731cc3bc546ff2b69667b03abe56a8",
                "fit_states": 32,
                "calibration_states": 8,
                "heldout_states": 8,
                "policy": "8/2/2 states per family in frozen manifest order",
            },
            "branch_ledger": {
                "path": ".generated/safe_local_waypoint_purpose_built_v1/branch_labels.jsonl",
                "sha256": "9b25b227c3e4de11e68e4abee454c4251399fafb468458a4e0d65f89bc6cdf7c",
                "rows": 576,
                "open_before_freeze": False,
            },
            "route_intent_labels": {
                "path": ".generated/safe_local_waypoint_route_intent_v2/route_intent_labels.jsonl",
                "sha256": "e8d33671502f717426836ec9a1039d445558b81e636ae105a81e2113151a8b69",
                "rows": 576,
                "open_before_freeze": False,
            },
            "data_audit": {
                "path": ".generated/safe_local_waypoint_route_intent_v2/data_audit.json",
                "sha256": "73381d7dc834813286b52f571a6b5d3370d04582fb5bf27beeee9447e5e4fd92",
                "rows": 576,
                "states": 48,
                "distance_tie_m": 0.03,
                "heading_tie_deg": 5.0,
            },
            "states": 48,
            "candidates_per_state": 12,
            "candidate_rows": 576,
            "horizons": [1, 2, 3],
            "families": list(FAMILY_IDS),
            "states_frozen_before_branching": True,
            "identity_role_candidate_and_outcome_changes": "forbidden",
            "fresh_panel": "forbidden",
        },
        "cpu_runtime_inputs": copy.deepcopy(CPU_RUNTIME_INPUT_BINDINGS),
        "latent_bindings": {
            "true_future_target_index": {
                "path": ".generated/safe_local_waypoint_route_intent_v2/target_latent_index.json",
                "sha256": "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874",
                "entries": 1728,
                "states": 48,
                "horizons": [1, 2, 3],
                "token_shape": [768, 1024],
                "storage_dtype": "float16",
            },
            "dense_true_future_index": {
                "path": ".generated/dense_temporal_true_future_safety_observability_v1/token_index.json",
                "sha256": "3cf2d42f52525ce8291f76ee5af0bd58ef0928d62a8f879efb40a9ac6530cd15",
                "content_digest": "7c24306dd1082940f948e47584ac525e451717258255585c04f82956736571f0",
                "occurrences": 8688,
                "unique_frames": 7154,
                "token_shape": [768, 1024],
                "current_view_authority": {
                    "current_occurrences": 48,
                    "unique_state_ids": 48,
                    "rgb_authority": "occurrence rgb_sha256 for kind=current",
                    "token_authority": (
                        "records entry selected by current occurrence rgb_sha256, with "
                        "token_path and token_sha256"
                    ),
                },
            },
            "dense_evidence_receipt": {
                "path": ".generated/dense_temporal_true_future_safety_observability_v1/evidence_receipt.json",
                "sha256": "a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1",
                "content_digest": "d5e506293e419ea6534eb5f3eb695e2f675cb5671f5270d4575aa762cee77028",
                "h3_tick_count": 15,
                "boundaries": [5, 10, 15],
            },
            "dense_route_replay_outcome_authority": {
                "directory": (
                    ".generated/dense_temporal_true_future_safety_observability_v1/"
                    "dense_replay"
                ),
                "evidence_receipt": {
                    "path": (
                        ".generated/dense_temporal_true_future_safety_observability_v1/"
                        "evidence_receipt.json"
                    ),
                    "sha256": (
                        "a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1"
                    ),
                    "bytes": 1484,
                },
                "state_files": 48,
                "prefreeze_byte_inventory_binding": copy.deepcopy(
                    DENSE_ROUTE_REPLAY_INPUT_BINDINGS
                ),
                "state_file_name": "<frozen state_id>.json",
                "state_schema": "dense_route_intent_true_future_state_v1",
                "status": "PASS",
                "branches_per_state": 12,
                "h3_tick_count": 15,
                "horizon_tick_boundaries": [5, 10, 15],
                "required_pre_reduction_validation": (
                    "hash and byte-bind all 48 exact state files; validate each canonical "
                    "content_digest after removing only content_digest; require state identity, "
                    "schema/status, 12 branches, H3 tick count 15 and boundaries [5,10,15]"
                ),
                "use": (
                    "frozen H1/H2/H3 realised route and descriptive stuck fields only; "
                    "descriptive contact remains bound to the exact contact-event authority; "
                    "never contract, threshold, population or candidate selection"
                ),
                "outcome_values_open_before_freeze": False,
            },
            "encoder": {
                "checkpoint_sha256": ENCODER_SHA256,
                "architecture": "frozen V-JEPA 2.1 ViT-L final representation",
                "constructor": "vjepa2_1_vit_large_384",
                "source_repository": {
                    "path": (
                        "/home/andrewknowles/.cache/"
                        "vjepa2-204698b45b3712590f06245fbfba32d3be539812"
                    ),
                    "git_commit": "204698b45b3712590f06245fbfba32d3be539812",
                    "worktree_clean_required": True,
                    "backbones_path": "src/hub/backbones.py",
                    "backbones_sha256": (
                        "391cdde1e9a1da47cb8094bbea5fbbe8acac0135b27e82f1a6ab19c0b39cc692"
                    ),
                    "backbones_bytes": 10164,
                },
                "output_token_shape": [768, 1024],
                "target_and_context_normalisation": "T.normalise token-axis layer normalisation",
                "preprocessing": (
                    "RGB rows 28:196, bicubic 512x384, ImageNet normalisation, "
                    "frozen ViT-L final tokens"
                ),
                "preprocessing_digest": "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5",
            },
        },
        "predictor_bindings": {
            "seed": SEED,
            "one_step": {
                "path": (
                    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
                    "seed_2026080901/seed_2026080901_rgb_one_step_epoch21.pt"
                ),
                "sha256": ONE_STEP_CHECKPOINT_SHA256,
                "bytes": 206534551,
                "tensor_open_before_freeze": False,
            },
            "two_step": {
                "path": (
                    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
                    "seed_2026080901/seed_2026080901_rgb_rollout_epoch21.pt"
                ),
                "sha256": TWO_STEP_CHECKPOINT_SHA256,
                "bytes": 206534551,
                "tensor_open_before_freeze": False,
            },
            "architecture": {
                "context_shape": [3, 768, 1024],
                "action_block_shape": [5, 2],
                "flattened_action_dimension": 10,
                "control_history_shape": [3, 5, 2],
                "active_channels": ["vx_body_mps", "yaw_rate_radps"],
                "vy": "inert in frozen corpus and forbidden in evaluation",
                "stale_15d_source_comment": "non-authoritative; executable ACTION_DIM=10",
                "width": 384,
                "depth": 6,
                "heads": 6,
                "use_proprio": False,
            },
            "input_reconstruction": {
                "context_offsets_source_frames": [-480, -240, 0],
                "source_frames_per_timestep": 48,
                "context_offsets_command_ticks": [-10, -5, 0],
                "context_offsets_elapsed_s": [-1.0, -0.5, 0.0],
                "command_rate_hz": 10,
                "frame_domain_disambiguation": (
                    "these are historical renderer/source frames and 10 Hz command "
                    "timesteps, never the 250 two-millisecond physics frames in one "
                    "new viability contact block"
                ),
                "warmup_block_boundaries": [38, 39, 40],
                "observed_frames": (
                    "deterministically replay and render immediately after warmup blocks "
                    "38, 39 and 40; no duplicated or fabricated frame"
                ),
                "observed_control_history": {
                    "training_semantics": "applied[k-1] for each observed image time k",
                    "contiguous_command_span": (
                        "block37 tick5; all five ticks of blocks38 and 39; and "
                        "block40 ticks1-4 (15 commands total)"
                    ),
                    "slot_0": ["block37_tick5", "block38_tick1", "block38_tick2", "block38_tick3", "block38_tick4"],
                    "slot_1": ["block38_tick5", "block39_tick1", "block39_tick2", "block39_tick3", "block39_tick4"],
                    "slot_2": ["block39_tick5", "block40_tick1", "block40_tick2", "block40_tick3", "block40_tick4"],
                    "required_validation": (
                        "assert exact command indices, timestamps and [3,5,2] reshape; "
                        "using blocks38/39/40 directly is forbidden"
                    ),
                },
                "snapshot_digest": (
                    "the captured snapshot immediately after block 40 must equal the "
                    "unique frozen branch-ledger snapshot digest shared by all 12 rows "
                    "for the state and the qualified replay digest"
                ),
                "snapshot_authority": (
                    "unique per-state branch-ledger snapshot digest; require 12/12 row "
                    "agreement and equality to the current qualified replay for all 48 states"
                ),
                "manifest_snapshot_digest": (
                    "descriptive audit only because it is present for only 22/48 states "
                    "and has 12 historical mismatches; never a fail gate or state exclusion"
                ),
                "state_failure": (
                    "any missing frame/control block or authoritative snapshot mismatch "
                    "fails the qualification run before predictor inference; no state is dropped"
                ),
                "current_view_reproduction": (
                    "newly rendered post-block40/current RGB must match the existing "
                    "48-state current occurrence file SHA exactly; the newly encoded "
                    "C-order FP16 raw tensor payload at encoder batch size 16 must match "
                    "the existing raw token payload SHA exactly even when the new file "
                    "uses an NPY container; zero tolerance and no substitution"
                ),
                "current_tensor_use": (
                    "the newly encoded, equality-validated current tensor is both context "
                    "slot 3 and CURRENT cost; equality makes old/new reuse immaterial"
                ),
                "duplication_or_fabrication": "forbidden",
            },
            "action_and_control_semantics": {
                "requested_and_applied_authority": (
                    "persist each frozen branch-ledger requested [3,5,3] tape and "
                    "applied/post_slew [3,5,3] tape separately; require exact equality "
                    "for all 576 rows and never conflate requested with applied"
                ),
                "candidate_actions": (
                    "A1, A2 and A3 are raw 10-D tick-major flattened post-slew "
                    "vx/yaw blocks, each exactly reshaped to [5,2]"
                ),
                "initial_control_history": (
                    "the observed [3,5,2] history is normalised with frozen "
                    "control_mean/control_std"
                ),
                "initial_context_tokens": (
                    "reload persisted FP16 context grids as float32, apply T.normalise, "
                    "then use the checkpoint's frozen half/autocast compute path"
                ),
                "autoregressive_append": (
                    "P.unroll appends control_slot_from_action(action_blocks[step-1]) "
                    "RAW, without normalisation, to the already-normalised control window"
                ),
                "mixed_scale_disclosure": (
                    "This surprising mixed-scale behaviour is the frozen checkpoint "
                    "evaluation path and is reproduced exactly, not corrected."
                ),
                "normalisation_stats": {
                    "path": (
                        "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
                        "proprio_v1/proprio_norm_stats.json"
                    ),
                    "file_sha256": "9380b4c6d9b59099e43bba9898e1417c273f88075d1ed122401cbb3272e18f94",
                    "stats_sha256": "f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab",
                    "fields": ["control_mean", "control_std"],
                },
            },
            "rollout": {
                "one_step": "three observed context frames produce H1",
                "two_step": (
                    "H1 is appended to the sliding three-frame window and the frozen "
                    "two-step rollout checkpoint produces H2 without teacher forcing"
                ),
                "future_observation_or_proprioception": "forbidden",
                "prediction_normalisation": "T.normalise after every predicted step",
                "inference_before_contract_freeze": "forbidden",
                "precision": (
                    "frozen FP32 checkpoint weights under BF16 autocast exactly as bound "
                    "by scripts/benchmark_one_tick_observation_prediction_control_loop_v1.py"
                ),
            },
            "inference_custody": {
                "model_config_exact_match": True,
                "state_dict_load": "strict=True; missing or unexpected key fails closed",
                "model_mode": "eval plus torch.inference_mode",
                "requires_grad": False,
                "optimizer": "absent",
                "training_steps": 0,
                "parameter_state_digest": "before inference must equal after inference per model",
                "parameter_state_digest_algorithm": {
                    "namespace_bytes": (
                        "JEPA_LOCAL_WAYPOINT_PARAMETER_STATE_V1 followed by NUL"
                    ),
                    "coverage": "complete state_dict: parameters and buffers",
                    "order": "ascending state_dict UTF-8 key",
                    "per_tensor_bytes": (
                        "uint64-BE key length + key; uint64-BE dtype-string length + "
                        "dtype; uint64-BE ndim; each dimension int64-BE; uint64-BE "
                        "payload length; contiguous CPU C-order tensor bytes"
                    ),
                    "hash": "SHA-256",
                    "models": ["encoder", "one_step", "two_step"],
                },
                "persist_call_counts": [
                    "encoder context/current/goal calls",
                    "one-step predictor calls by state/candidate/horizon",
                    "two-step predictor calls by state/candidate/horizon",
                ],
                "future_input_fields": [],
                "autocast": "BF16 on cuda:0 in the frozen TinyQuadJEPA environment",
                "gpu_environment_revalidation": (
                    "the materialize child must recompute the five foundational package "
                    "roots, versions and import resolutions, require exact equality to "
                    "the persisted gpu_environment receipt and bind that receipt in the "
                    "gpu_inference receipt before opening checkpoints"
                ),
                "batching_and_order": {
                    "encoder": {
                        "batch_size": 16,
                        "record_order": (
                            "ascending RGB SHA-256, then kind, numeric state identity and "
                            "context slot; this matches the existing target-encoder "
                            "RGB-identity-first ordering"
                        ),
                        "records": 192,
                        "batches": 12,
                        "current": "alias of context slot 2; not re-encoded",
                        "true_future": "existing bound tokens; not re-encoded",
                    },
                    "predictor": {
                        "batch_size": 12,
                        "batch_unit": "all candidate indices 0..11 for exactly one state",
                        "state_order": "numeric state-id order",
                        "horizon_order": ["H1", "H2", "H3"],
                        "checkpoint_order": ["ONE_STEP_PREDICTED", "TWO_STEP_PREDICTED"],
                        "state_batches_per_checkpoint": 48,
                        "unroll_calls_per_checkpoint": 48,
                        "model_forward_calls_per_checkpoint": 144,
                    },
                    "dynamic_oom_batch_fallback": "forbidden; fail closed",
                },
            },
        },
        "prior_predictor_findings": {
            "authority": [
                copy.deepcopy(STATIC_FILE_BINDINGS["prior_rollout_result"]),
                copy.deepcopy(STATIC_FILE_BINDINGS["predictor_claims_matrix"]),
            ],
            "preserved_aggregate_findings_only": [
                "rollout improves direct future fidelity at H1 through H4",
                "rollout improves selected action-specific metrics, strongest at H3 and H4",
                "H1 action retrieval remains weak and imprecise",
                "planning utility was not previously evaluated",
                "the eight-seed experiment is not rerun in this qualification",
            ],
            "detailed_outcomes_copied_into_contract": False,
            "inference_or_reanalysis_before_freeze": False,
        },
        "goal_view": {
            "count": "exactly one candidate-independent goal view per frozen state",
            "position_world": {
                "x_y": (
                    "deterministic frozen SceneGraph cell_center of "
                    "waypoint_path_cells[2]"
                ),
                "z": "snapshot base z",
            },
            "orientation_world_rpy_rad": {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": (
                    "atan2(cell_center(waypoint_path_cells[1]).y - "
                    "cell_center(waypoint_path_cells[0]).y, "
                    "cell_center(waypoint_path_cells[1]).x - "
                    "cell_center(waypoint_path_cells[0]).x)"
                ),
            },
            "route_heading_authority": (
                "frozen scene graph cell_center and frozen waypoint_path_cells[0:2]"
            ),
            "body_goal_coordinates": (
                "from the frozen current start pose, persist body-frame delta x/y and "
                "sin/cos of wrapped goal-yaw minus start-yaw, matching existing g_t"
            ),
            "manifest_waypoint_fields": (
                "when waypoint_xy or waypoint_body_xy is present, require exact equality "
                "to the deterministic reconstruction; absence is allowed and does not "
                "exclude a state because only 22/48 manifests carry these fields"
            ),
            "preconditions": [
                "waypoint_path_cells has length at least three",
                "any present manifest waypoint_xy equals the exact SceneGraph cell centre of path[2]",
                (
                    "path[2] is endpoint-reachable from path[0] under frozen SceneGraph "
                    "nav-blocked endpoint semantics; it need not be free or transit-safe"
                ),
                "snapshot start/base z is finite",
                "render and encoded tokens are finite",
            ],
            "goal_pose_semantics": copy.deepcopy(
                GOAL_VIEW_EXECUTION_AMENDMENT["amended_goal_pose_semantics"]
            ),
            "goal_cell_classification_counts": copy.deepcopy(
                GOAL_CELL_CLASSIFICATION_COUNTS
            ),
            "blocked_state_ids": copy.deepcopy(GOAL_CELL_BLOCKED_STATE_IDS),
            "source_semantic_validation": {
                "path2_position_substitution": "forbidden",
                "path1_position_substitution": "forbidden",
                "state_drop_or_alternate_goal_search": "forbidden",
                "endpoint_reachability_required": True,
                "nav_blocked_is_diagnostic_not_failure": True,
                "all_48_endpoint_reachable": True,
            },
            "renderer": {
                "source": "lewm/oracle/go2_textured_v03_renderer.py",
                "source_sha256": STATIC_FILE_BINDINGS["renderer"]["sha256"],
                "image_shape": [224, 224, 3],
                "horizontal_fov_deg": 78.323,
                "near_m": 0.05,
                "far_m": 200.0,
                "mount": "frozen nominal textured-v03 camera mount",
                "jitter_retraction_resize": "none",
                "robot_visibility": "no-robot static view; explicitly not an embodied self-view",
                "goal_render_semantics": GOAL_VIEW_RENDER_SEMANTICS,
                "physical_executability_claim": False,
                "effective_scene_geometry": "FLOOR_PLANE_ONLY",
                "structural_geometry_omission": (
                    "the exact historical caller passes genesis_scene.json, whose "
                    "structures are in objects; build_scene reads only walls, obstacles "
                    "and landmarks, so no wall, obstacle or landmark entity is added"
                ),
                "compatibility_rule": (
                    "preserve this historical defect because current RGB/token bytes and "
                    "frozen true-future targets were generated by the same path"
                ),
                "explicit_wall_visual_reasoning_claim": False,
            },
            "encoder_and_preprocess": "exactly the frozen context/target encoder path",
            "candidate_dependent_inputs": [],
            "unrenderable_or_unencodable_state": (
                "fail the whole qualification before candidate inference; do not drop, "
                "replace or substitute a state"
            ),
            "persist": [
                "pose and source fields",
                "render SHA-256",
                "token SHA-256",
                "renderer/preprocess/encoder bindings",
            ],
        },
        "cost": {
            "name": "UNTUNED_TOKEN_MEAN_GOAL_COSINE_DISTANCE",
            "formula": (
                "C_h = mean_{t=1..768}(1 - dot(l2(candidate_h[t]), "
                "l2(goal[t])))"
            ),
            "primary_horizon": "H3",
            "diagnostic_horizons": ["CURRENT", "H1", "H2", "H3"],
            "token_alignment": "same spatial-token index; no matching, pooling or search",
            "token_count": 768,
            "token_width": 1024,
            "canonical_input_conversion": "persisted float16 tokens loaded as float32",
            "layer_normalisation": (
                "scripts.run_dev_v03_temporal_action_jepa_v1.T.normalise: "
                "torch.nn.functional.layer_norm over the last 1024 dimensions, "
                "elementwise affine absent, default eps=1e-5"
            ),
            "per_token_normalisation": (
                "torch L2 normalise over the last dimension with eps=1e-12 in float32"
            ),
            "dot_products": (
                "768 matching token-index float32 dot products, numerically clipped "
                "to the mathematical cosine range [-1,1] before 1-cosine"
            ),
            "aggregate": "unweighted arithmetic mean of all 768 (1-cosine) values in float64",
            "canonical_reducer": (
                "lewm.safety.jepa_local_waypoint_planning_cost_metrics_v1."
                "tokenwise_normalized_cosine_mean_cost NumPy CPU implementation over "
                "persisted FP16 grids; this exact executable is authoritative for "
                "every row, aggregate and terminal replay"
            ),
            "contract_reference_helper": (
                "tokenwise_cosine_cost is outcome-free fixture scaffolding only and is "
                "never an exact scientific scalar authority"
            ),
            "gpu_role": "encode, predict and persist only; never authoritative cost reduction",
            "torch_vs_canonical_numpy_fixture_absolute_tolerance": 1e-6,
            "nonfinite_token_or_cost": "fail the state before ranking",
            "weights": "none; no learned or tuned weighting",
            "monotonic_diagnostics": {
                "sequence": ["CURRENT", "H1", "H2", "H3"],
                "progress_delta": "C_previous - C_next",
                "monotone_toward_goal": "C_CURRENT >= C_H1 >= C_H2 >= C_H3",
                "reported": [
                    "three adjacent deltas",
                    "violation count",
                    "strict-improvement count",
                    "end-to-end C_CURRENT - C_H3",
                    "per-candidate current-to-H3 nonincrease and strict-decrease Booleans",
                    "aggregate current-to-H3 nonincrease and strict-decrease fractions",
                    "all-step monotonic-trajectory fraction reported separately",
                ],
                "gate": False,
            },
        },
        "sources": list(SOURCE_IDS),
        "source_semantics": {
            "TRUE_FUTURE": "frozen actual H1/H2/H3 target tokens",
            "ONE_STEP_PREDICTED": (
                "rgb_one_step epoch21 weights evaluated with the identical frozen "
                "P.unroll(..., max_h=3) autoregressive path from the three observed "
                "contexts, initial control history and H1-H3 action blocks"
            ),
            "TWO_STEP_PREDICTED": (
                "rgb_rollout epoch21 weights evaluated with that same exact frozen "
                "P.unroll(..., max_h=3) autoregressive path"
            ),
            "predicted_source_common_rule": (
                "both sources feed each prior prediction back into the sliding context; "
                "neither receives an independent true context, future observation or "
                "future latent at H2 or H3"
            ),
        },
        "populations": {
            "ids": list(POPULATION_IDS),
            "ALL_CANDIDATES": "all twelve frozen candidate identities per evaluable state",
            "ORACLE_CONTACT_FREE": (
                "only candidates with no frozen disallowed robot-environment contact "
                "during the committed first block/H1 (five 100 ms ticks; 250 physics frames)"
            ),
            "ORACLE_VIABILITY_ADMISSIBLE": (
                "only candidates admitted by the frozen exact two-ply oracle viability "
                "authority; no learned safety filter"
            ),
            "nesting_assertion": (
                "ORACLE_VIABILITY_ADMISSIBLE subset of ORACLE_CONTACT_FREE subset of "
                "ALL_CANDIDATES; any violation fails closed"
            ),
            "empty_population": "abstain and report; never borrow a candidate",
            "gate_population": "ORACLE_VIABILITY_ADMISSIBLE",
        },
        "comparators": list(COMPARATOR_IDS),
        "paired_comparison_ids": list(PAIRED_COMPARISON_IDS),
        "comparator_semantics": {
            "KINEMATIC_ROUTE_BASELINE": (
                "existing 0.1 s Euler integration of all 15 commands in the first three "
                "post-slew blocks, input shape exactly [3,5,3] with vx,vy,yaw_rate "
                "and the full vy formula retained although frozen vy is zero; "
                "derive nominal p_d and p_theta only; route_order "
                "takes maximum p_d, retains candidates within 0.03 m, then maximum p_theta, "
                "then ascending candidate index; no 5 degree kinematic heading deadband "
                "and no invented nominal completion threshold"
            ),
            "RANDOM": (
                "ascending SHA-256 digest of the exact RANDOM_V1 byte encoding; no "
                "global RNG state, then ascending candidate_index on digest ties"
            ),
            "latent_costs": "ascending H3 cost; deterministic candidate-index final tie-break",
            "oracle_best": (
                "frozen realised H3 route order within the evaluated population using "
                "distance, registered tie margin, heading, then candidate index"
            ),
            "random_v1": {
                "namespace_utf8": (
                    "JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1/RANDOM_V1"
                ),
                "bytes": (
                    "namespace UTF-8 + NUL + seed unsigned uint64 big-endian + NUL + "
                    "state_id UTF-8 + NUL + candidate_index unsigned uint32 big-endian"
                ),
                "order": "ascending 32-byte SHA-256 digest, then ascending candidate_index",
                "population": "hash-rank only candidates in the current population",
            },
        },
        "ranking_metrics": {
            "required": [
                "Spearman",
                "Kendall",
                "pairwise accuracy",
                "best-route top-1",
                "best-route top-3",
                "MRR",
                "mean best-route rank",
                "cost spread",
                "cost tie count and rate",
            ],
            "selection": [
                "selected candidate identity",
                "selected contact",
                "selected oracle-nonviable",
                "selected stuck",
                "selected distance progress",
                "selected heading improvement",
                "selected combined route utility",
                "normalised regret",
                "completion",
                "abstention",
            ],
            "aggregation": {
                "levels": ["all", "per-state", "per-family", "per-role"],
                "spearman": "arithmetic mean of finite per-state coefficients",
                "kendall_tau_b": "arithmetic mean of finite per-state coefficients",
                "pairwise_accuracy": (
                    "sum per-state ordered-pair correct credit divided by sum per-state "
                    "ordered-pair denominators"
                ),
                "family": "apply the same reductions to states within each frozen family",
                "persistence": (
                    "persist every per-state coefficient, credit and valid denominator "
                    "needed for exact aggregate reproduction"
                ),
            },
            "horizon_diagnostics": (
                "for H1, H2 and H3 separately compare that horizon's latent cost/order "
                "only with realised p_d, p_theta, completion and descriptive contact/stuck "
                "at the same horizon; H3 aliases remain the sole primary ranking authority"
            ),
            "paired_comparisons": {
                "ids": list(PAIRED_COMPARISON_IDS),
                "role": "heldout",
                "population": "ORACLE_VIABILITY_ADMISSIBLE",
                "per_state_rows_required": True,
                "effects": [
                    "selected progress",
                    "normalised regret",
                    "pairwise accuracy",
                    "best-route rank",
                    "H1 contact selection",
                    "nonviable-successor selection",
                ],
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "unit": "state",
                "interval": "descriptive percentile 95% CI",
                "generalisation_or_hypothesis_test": False,
                "state_order": "frozen state-manifest order within the reported role",
                "namespace": (
                    "JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1/"
                    "STATE_BOOTSTRAP_V1"
                ),
                "resampling": (
                    "for replicate r and draw d, SHA256(namespace UTF-8 + NUL + seed "
                    "uint64-BE + NUL + comparison_id UTF-8 + NUL + r uint32-BE + d "
                    "uint32-BE); first uint64-BE modulo N selects a state with replacement"
                ),
                "percentile_method": (
                    "sorted replicate values, Type-7 linear quantile at p=.025 and .975: "
                    "h=(B-1)*p and interpolate floor(h) to ceil(h)"
                ),
            },
            "route_outcome_authority": {
                "horizon": "H3",
                "ordered_tuple": [
                    "realised completed (true before false)",
                    "realised p_d with 0.03 m indifference margin",
                    "realised p_theta_rad with 5 degree indifference margin",
                    "ascending candidate_index only as deterministic total-order tie-break",
                ],
                "safety_fields_in_route_preference": [],
                "note": (
                    "contact, successor viability and stuck only define/report populations "
                    "and selected outcomes; they never enter route preference or utility"
                ),
            },
            "rank_ties": {
                "oracle_unordered_pair": (
                    "exclude when completion ties, abs(p_d difference)<=0.03 m and "
                    "abs(p_theta_rad difference)<=5 degrees"
                ),
                "predicted_cost_tie_abs_lte": 1e-12,
                "predicted_cost_tie_pairwise_credit": 0.5,
                "selection_final_tie_break": "ascending frozen candidate_index",
                "spearman": "scipy.stats.spearmanr(-cost, continuous realised p_d), midranks",
                "kendall": "scipy.stats.kendalltau(-cost, continuous realised p_d), tau-b",
                "constant_or_undefined": "null and any applicable gate fails",
            },
            "combined_route_utility": {
                "kind": "weight-free within-state margin-Borda fraction",
                "formula": "(wins + 0.5*unordered_or_tied_pairs)/(N-1)",
                "pair_authority": "completion, 0.03 m p_d, 5 degree p_theta_rad",
                "persist_authority_tuple": True,
            },
            "top_rank_metrics": (
                "oracle-best is first in the realised total order within the current "
                "population; top1, top3, MRR and mean rank refer to that candidate"
            ),
            "normalised_regret": (
                "per state: (best p_d - selected p_d)/(max p_d - min p_d) within the "
                "population; if the range is <=1e-12 use zero only when values are equal, "
                "otherwise fail; aggregate mean over nonabstaining states"
            ),
            "selected_progress_gate_ratio": (
                "sum selected realised p_d / max(abs(sum oracle-best realised p_d),1e-9); signed"
            ),
        },
        "gates": {
            "true_future": {
                "source": "TRUE_FUTURE",
                "population": "ORACLE_VIABILITY_ADMISSIBLE",
                "pairwise_accuracy_gte": 0.70,
                "spearman_gte": 0.60,
                "normalised_regret_lte": 0.25,
                "best_route_top3_gte": 0.75,
                "selected_progress_fraction_of_oracle_best_gte": 0.80,
                "no_family_complete_collapse": True,
                "classification_if_pass": "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL",
                "classification_if_fail": "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
            },
            "two_step_predicted": {
                "prerequisite_true_future_gate": True,
                "source": "TWO_STEP_PREDICTED",
                "population": "ORACLE_VIABILITY_ADMISSIBLE",
                "pairwise_accuracy_gte": 0.65,
                "normalised_regret_lte": 0.30,
                "selected_progress_fraction_of_true_future_selected_gte": 0.75,
                "all_candidates_contact_selections_lte_one_step": True,
                "all_candidates_nonviable_selections_lte_one_step": True,
                "pairwise_accuracy_gt_one_step": True,
                "at_least_one_of_regret_or_selected_progress_improves_vs_one_step": True,
                "no_family_complete_collapse": True,
                "classification_if_pass": "TWO_STEP_JEPA_PLANNING_COST_SIGNAL",
                "classification_if_fail": None,
            },
            "predicted_base_preservation_screens": {
                "sources": ["ONE_STEP_PREDICTED", "TWO_STEP_PREDICTED"],
                "population": "ORACLE_VIABILITY_ADMISSIBLE",
                "pairwise_accuracy_gte": 0.65,
                "normalised_regret_lte": 0.30,
                "selected_progress_fraction_of_true_future_selected_gte": 0.75,
                "no_family_complete_collapse": True,
                "purpose": (
                    "report absolute preservation separately for both predicted sources; "
                    "these screens do not replace the complete two-step improvement and "
                    "safety gate"
                ),
            },
            "family_complete_collapse": {
                "families": list(FAMILY_IDS),
                "required_evaluable_nonabstaining_states_gte": 1,
                "required_ordered_pairs_gte": 1,
                "positive_route_signals_any": [
                    "pairwise_accuracy > 0.5",
                    "best_route_top3_rate > 0",
                    "selected_route_progress_sum_m > 0",
                ],
                "collapse": (
                    "missing required support or none of the three positive route "
                    "signals; no-family-collapse requires all four families to pass"
                ),
            },
            "weakening": "forbidden",
        },
        "paired_materiality": {
            "population": "ORACLE_VIABILITY_ADMISSIBLE",
            "comparison": "TWO_STEP_PREDICTED_LATENT_COST minus KINEMATIC_ROUTE_BASELINE",
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "unit": "state",
                "algorithm": "ranking_metrics.bootstrap hash resampling and Type-7 quantiles",
            },
            "triggers_any": {
                "selected_progress": {
                    "two_step_minus_kinematic_m_gte": 0.05,
                    "paired_ci95_lower_gt": 0.0,
                },
                "normalised_regret": {
                    "kinematic_minus_two_step_gte": 0.05,
                    "paired_ci95_lower_gt": 0.0,
                },
                "hard_family_selected_progress_ratio": {
                    "families": list(HARD_FAMILY_IDS),
                    "aggregation": "equal-family mean",
                    "two_step_minus_kinematic_gte": 0.10,
                    "paired_state_bootstrap_ci95_lower_gt": 0.0,
                },
            },
            "required_for_every_trigger": {
                "contact_selection_increase_lte": 0,
                "nonviable_successor_selection_increase_lte": 0,
                "neither_hard_family_collapsed": True,
            },
            "secondary_if_passed": "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS",
            "kinematic_material_superiority": {
                "progress": (
                    "kinematic minus two-step selected progress >=0.05 m and paired "
                    "bootstrap CI95 lower >0"
                ),
                "regret": (
                    "two-step minus kinematic normalised regret >=0.05 and paired "
                    "bootstrap CI95 lower >0"
                ),
                "hard_family": (
                    "equal-family mean kinematic minus two-step selected-progress ratio "
                    ">=0.10 and paired bootstrap CI95 lower >0"
                ),
                "any_trigger": True,
                "required": (
                    "kinematic has no H1-contact or nonviable-successor selection increase "
                    "versus two-step and neither hard family collapses"
                ),
            },
        },
        "classification": {
            "primary_exactly_one": list(PRIMARY_CLASSIFICATIONS),
            "precedence": [
                {
                    "if": "true_future_gate fails",
                    "then": "RAW_LATENT_GOAL_COST_NO_GO",
                },
                {
                    "if": (
                        "true_future_gate passes and two_step_gate passes and kinematic "
                        "baseline is materially superior and JEPA incremental secondary is false"
                    ),
                    "then": "KINEMATIC_BASELINE_DOMINANT",
                },
                {
                    "if": "true_future_gate passes and two_step_gate passes",
                    "then": "TWO_STEP_JEPA_PLANNING_COST_SIGNAL",
                },
                {
                    "if": "true_future_gate passes and complete two_step predictor gate fails",
                    "then": "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO",
                    "note": "one-step and two-step absolute metrics remain separately reported",
                },
            ],
            "secondary_allowed": list(SECONDARY_CLASSIFICATIONS),
            "required_descriptive_flags": {
                "both_predicted_base_screens_failed": (
                    "one_step base screen false AND two_step base screen false"
                ),
                "ONE_STEP_BASE_SCREEN_ONLY": (
                    "one_step base screen true AND two_step base screen false; diagnostic "
                    "flag only, never a primary or secondary classification"
                ),
                "two_step_base_screen_passed_but_full_gate_failed": (
                    "two_step base screen true AND complete Section-14 two-step gate false"
                ),
            },
            "predictor_no_go_wording": (
                "The required complete two-step predictor-planning qualification failed; "
                "do not state that both absolute predicted screens failed unless "
                "both_predicted_base_screens_failed is true."
            ),
            "kinematic_material_superiority": (
                "same signed paired-materiality magnitudes and state-bootstrap rule, "
                "with direction reversed in favour of KINEMATIC_ROUTE_BASELINE, no "
                "contact/nonviable increase, and no family collapse"
            ),
        },
        "next_decision": {
            "allowed_ids": list(NEXT_EXPERIMENT_IDS),
            "selection": {
                "TWO_STEP_JEPA_PLANNING_COST_SIGNAL": "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1",
                "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO": "PLAN_AWARE_MONOTONE_JEPA_COST_V1",
                "RAW_LATENT_GOAL_COST_NO_GO": "PLAN_AWARE_MONOTONE_JEPA_COST_V1",
                "KINEMATIC_BASELINE_DOMINANT": "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
            },
            "requirements_decision": NEXT_REQUIREMENTS_DECISION,
            "conditional_specifications": {
                "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1": {
                    "candidate_bank": "unchanged fixed deployable bank",
                    "loop": "execute a short committed prefix, reobserve, and replan",
                    "safety_population": "exact oracle admissibility only",
                    "report": [
                        "immediate contact",
                        "successor viability",
                        "route progress",
                        "abstention",
                    ],
                    "continuous_optimisation": (
                        "CEM or MPPI may be considered only in a later separately frozen "
                        "experiment, never in this qualification"
                    ),
                },
                "PLAN_AWARE_MONOTONE_JEPA_COST_V1": {
                    "target": (
                        "route-only local-waypoint and future-trajectory pairwise/listwise "
                        "progress with rollout consistency"
                    ),
                    "forbidden_targets": [
                        "safety",
                        "completion",
                        "aggregate utility",
                        "material contact",
                    ],
                },
                "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1": {
                    "target": "local goal may temporarily move away from the direct route",
                    "safety_population": "exact oracle admissibility",
                    "topological_memory": "forbidden",
                },
            },
            "execution_authorised_here": False,
        },
        "requirements_custody": {
            "stage_a_freeze_commit": STAGE_A_FREEZE_COMMIT,
            "authoritative_result": {
                **copy.deepcopy(STATIC_FILE_BINDINGS["protected_scope_authoritative_result"]),
                "result_commit": STARTING_HEAD,
            },
            "classifications": list(REQUIREMENTS_CLASSIFICATIONS),
            "statement": REQUIREMENTS_STATEMENT,
            "next_decision": NEXT_REQUIREMENTS_DECISION,
            "scope_or_label_change": False,
            "stage_b": "NOT_RUN_NOT_AUTHORISED",
            "preserved_conclusions": [
                "the current simulated protected-contact proxy remains unchanged",
                "single-origin and up-to-three-origin range coverage are no-go under the full scope",
                "the sensor-coverage micro-viability path is no-go",
                "the platform stopping-mode parity and replanning interface remain unresolved",
                "no learned persistent memory or end-to-end maze solver is qualified",
                "local-waypoint JEPA planning utility has not previously been evaluated",
            ],
        },
        "storage": {
            "workspace_filesystem_minimum_free_gb": 20,
            "output_filesystem_minimum_free_gb": 50,
            "temporary_storage_ceiling_gb": 20,
            "final_storage_ceiling_gb": 12,
            "output_root": str(OUTPUT_ROOT),
            "output_root_filesystem": "independent high-capacity ext4; verify with df -hT",
            "large_cache_on_workspace": False,
            "streaming": True,
            "full_latent_persistence": (
                "persist raw predicted float16 latents plus goal and context latents, "
                "together with complete row evidence and aggregates"
            ),
            "preflight": "fail before inference if free-space or predicted-size gates fail",
        },
        "execution": {
            "entrypoint": str(ENTRYPOINT_PATH),
            "sequence": [
                "freeze contract, schema, fixtures and source closure",
                "run preexecution and environment checks",
                "reconstruct every state context and candidate input fail-closed",
                "render and encode one candidate-independent goal view per state",
                "open bound checkpoints only after all prior gates pass",
                "stream single-seed inference and row evidence",
                "derive aggregates, paired bootstrap, gates and classification from rows",
                "validate row-to-aggregate reproduction and publish atomically",
            ],
            "random_model_seeds": [SEED],
            "retries_or_seed_search": "forbidden",
            "heldout_selection_or_tuning": "forbidden",
            "reconstruction_prefix_custody": copy.deepcopy(
                RECONSTRUCTION_PREFIX_CUSTODY
            ),
            "controller_execution_custody": copy.deepcopy(
                CONTROLLER_EXECUTION_CUSTODY
            ),
            "execution_watchdogs": copy.deepcopy(EXECUTION_WATCHDOGS),
            "roles": {
                "primary_gate": "heldout (8 frozen states)",
                "descriptive": ["fit", "calibration", "heldout", "all_48"],
                "threshold_or_model_calibration": "none",
                "heldout_independence": (
                    "observed development-only and non-independent; no generalisation claim"
                ),
            },
            "environments": {
                "cpu_replay_render_oracle": {
                    "interpreter": ".generated/venvs/genesis_render_vulkan/bin/python",
                    "interpreter_binary_binding": copy.deepcopy(
                        INTERPRETER_BINARY_BINDING
                    ),
                    "python": "3.12.3",
                    "genesis": "0.3.14",
                    "numpy": "2.4.6",
                    "scipy": "1.17.1",
                    "torch": "2.12.0+cu130",
                    "pillow": "11.3.0",
                    "pyyaml": "6.0.3",
                    "foundational_packages": copy.deepcopy(
                        CPU_FOUNDATIONAL_PACKAGE_BINDINGS
                    ),
                    "cuda_available": False,
                    "workers": "exactly os.cpu_count(); expected 32 at preflight",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                "encoder_predictor": {
                    "interpreter": "/home/andrewknowles/TinyQuadJEPA/bin/python",
                    "interpreter_binary_binding": copy.deepcopy(
                        INTERPRETER_BINARY_BINDING
                    ),
                    "python": "3.12.3",
                    "torch": "2.10.0.dev20250926+rocm6.3",
                    "numpy": "2.4.2",
                    "scipy": "1.18.0",
                    "pillow": "12.1.0",
                    "pyyaml": "6.0.3",
                    "foundational_packages": copy.deepcopy(
                        GPU_FOUNDATIONAL_PACKAGE_BINDINGS
                    ),
                    "device": "cuda:0 AMD Radeon AI PRO R9700",
                },
                "process_separation": (
                    "Genesis replay/render/oracle only in the CPU environment; encoder "
                    "and predictor inference only in TinyQuadJEPA; no simulator import "
                    "or execution in the inference environment"
                ),
                "exact_interpreter_and_import_receipts_required": True,
                "foundational_package_closure_policy": copy.deepcopy(
                    FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
                ),
            },
            "commit_messages": [
                "Freeze JEPA local waypoint planning cost qualification",
                "Evaluate JEPA local waypoint planning cost qualification",
            ],
        },
        "candidate_and_viability_semantics": {
            "decision_block": "five 100 ms ticks; commit one block and then replan",
            "cost_horizons": "H1/H2/H3 cover the first one/two/three plan blocks",
            "immediate_contact": (
                "any frozen disallowed robot-environment contact at any 2 ms physics "
                "step during the first block; existing exclusions unchanged"
            ),
            "successor": "actual deterministic state after the first block",
            "unique_first_block_primitives": [
                "hold",
                "forward_slow",
                "forward_medium",
                "forward_fast",
                "backward",
                "yaw_left",
                "yaw_right",
                "arc_left",
                "arc_right",
            ],
            "macro_candidates": 12,
            "same_first_primitive_successor": "exact shared identity",
            "viability_rule": (
                "candidate is admissible iff its current prefix is contact-free and its "
                "actual successor has at least one contact-free next primitive"
            ),
            "row_identities": {
                "successor_safe_action_count": "integer in [0,9]",
                "successor_viable": "successor_safe_action_count > 0",
                "oracle_viability_admissible": (
                    "not immediate_contact_h1 AND successor_viable"
                ),
                "selected_nonviable_successor": (
                    "not successor_viable; never shorthand for not oracle_viability_admissible"
                ),
            },
            "unconditional_materialisation": {
                "current_blocks_per_state": 9,
                "successor_blocks_per_state": 81,
                "states": 48,
                "blocks": 4320,
                "physics_frames": 1080000,
                "current_contact_bitset_shape": [9, 250],
                "successor_contact_bitset_shape": [9, 9, 250],
                "include_successors_of_contacting_prefixes": True,
            },
            "raw_continuation_snapshot_rule": {
                "capture_mode": "RAW_CONTINUATION_AFTER_CURRENT_H1",
                "purpose": (
                    "evaluation-only definition of all 81 next-action outcomes; it is "
                    "not a deployable or canonical replanning boundary"
                ),
                "capture": (
                    "after every current H1, capture the complete Genesis solver, "
                    "controller, harness, RNG and counter state using the frozen snapshot "
                    "digest identity without invoking V1 assert_canonical_boundary"
                ),
                "permitted_terminal_flags": ["fall", "tipped", "out_of_bounds"],
                "nan": "always rejected fail-closed",
                "production_reset_checks": "suppressed for current and successor fanout",
                "restore_count_per_current_primitive": 9,
                "persist_terminal_flags": True,
                "all_current_primitives": 9,
                "all_states": 48,
            },
            "persist": [
                "2 ms current and successor contact bitsets",
                "first event and link attribution",
                "exact snapshot/action mappings",
                "12-macro-candidate to first-primitive map",
                "nine raw-continuation snapshot digests and terminal flags per state",
            ],
            "cross_validation": (
                "independent existing contact-event artifact validates each mapped "
                "12-candidate H1 physics-rate contact against rerun 9-current-row contact"
            ),
            "existing_h1_contact_authority": {
                "path": (
                    ".generated/contact_hazard_ontology_and_instrumentation_v1/"
                    "raw_contact_event_index.json"
                ),
                "sha256": "1eac5be90b48e88cac7aa8db7f3ce3bd6655e1404f7ba6591ac90afa9e1f0d4d",
                "bytes": 347250,
                "schema": "lewm_contact_hazard_raw_contact_event_index_v1",
                "content_digest": "b0897c9fbc1e739495a0b1184ede11639d6932e6a5fa37761efc246f8b45d610",
                "states": 48,
                "branches": 576,
                "physics_steps_per_branch": 750,
                "physics_dt_s": 0.002,
                "passing_states": 48,
                "mismatching_states": 0,
                "raw_point_cloud_required": False,
            },
            "replacement_state_or_transition": "forbidden",
        },
        "prohibitions": {
            "training_steps": 0,
            "checkpoint_mutation": False,
            "checkpoint_selection": False,
            "fresh_panel": False,
            "state_role_candidate_or_label_change": False,
            "contact_scope_change": False,
            "stage_b": False,
            "untouched_g2_access": False,
            "jepa_encoder_or_predictor_weight_inspection_before_freeze": False,
            "memory": False,
            "online_memory": False,
            "experimental_candidate_selecting_navigation": False,
            "experimental_candidate_selecting_navigation_scope": (
                "no experimental candidate-selecting JEPA, MPC or learned navigation; "
                "the explicitly bound 40-block production route-teacher/PPO prefix is "
                "reconstruction custody only and is not a navigation qualification"
            ),
            "routing_or_beacon_capture": False,
            "deployment_claim": False,
        },
        "tracked_paths": {
            "preregistration": str(TRACKED_PREREGISTRATION_PATH),
            "contract": str(TRACKED_CONTRACT_RECEIPT_PATH),
            "output_schema": str(TRACKED_OUTPUT_SCHEMA_PATH),
            "fixture": str(TRACKED_FIXTURE_PATH),
            "goal_view_amendment": str(TRACKED_GOAL_VIEW_AMENDMENT_PATH),
            "source_closure": str(TRACKED_SOURCE_CLOSURE_PATH),
            "result": str(TRACKED_RESULT_PATH),
            "report": str(TRACKED_REPORT_PATH),
        },
    }


_PHASE_RECEIPT_REQUIRED_KEYS = [
    "schema",
    "experiment_id",
    "source_freeze_commit",
    "contract_sha256",
    "output_schema_sha256",
]


_OUTPUT_FILES = {
    "preexecution_receipt": {
        "path": "receipts/preexecution.json",
        "schema": "jepa_local_waypoint_planning_cost_preexecution_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "head",
            "contract",
            "output_schema",
            "fixture",
            "source_closure",
            "goal_view_execution_amendment_binding",
            "goal_view_static_validation",
            "preexecution_custody",
            "panel_bindings",
            "cpu_runtime_input_inventory_binding",
            "checkpoint_file_bindings_without_tensor_open",
            "environment",
            "storage",
            "canonical_output_fresh",
            "canonical_output_root",
            "hidden_attempt_root",
            "execution_watchdog_config",
            "cpu_worker_environment",
            "prohibition_counters",
            "pass",
            "content_digest",
        ],
        "preexecution_custody_required_keys": [
            "contract_disclosure",
            "live_validation",
        ],
        "preexecution_contract_disclosure_exact": copy.deepcopy(
            _contract_core()["preexecution_custody"]
        ),
        "preexecution_live_validation_required_keys": [
            "outcome_barrier_lifted_only_after_freeze_commit",
            "outcome_rows_read",
            "checkpoint_tensors_opened",
            "predictor_inference_calls",
            "fixture_checks_passed",
            "fixture_pass",
        ],
        "preexecution_live_validation_exact": {
            "outcome_barrier_lifted_only_after_freeze_commit": True,
            "outcome_rows_read": 0,
            "checkpoint_tensors_opened": 0,
            "predictor_inference_calls": 0,
            "fixture_pass": True,
        },
        "execution_watchdog_config_exact": copy.deepcopy(EXECUTION_WATCHDOGS),
        "goal_view_execution_amendment_binding_exact": copy.deepcopy(
            GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
        ),
        "goal_view_static_validation_exact": copy.deepcopy(
            GOAL_VIEW_STATIC_VALIDATION_SUCCESS
        ),
        "cpu_worker_environment_exact": copy.deepcopy(
            EXECUTION_WATCHDOGS["cpu_worker_environment"]
        ),
    },
    "environment_receipt": {
        "path": "receipts/environment.json",
        "schema": "jepa_local_waypoint_planning_cost_environment_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "python",
            "torch",
            "device",
            "interpreter_bindings",
            "package_inventory",
            "import_closure",
            "process_separation",
            "source_hashes",
            "checkpoint_file_hashes",
            "checkpoint_tensor_open_count",
            "smoke",
            "gpu_environment_receipt_binding",
            "cpu_runtime_input_inventory_binding",
            "foundational_package_closure_policy",
            "historical_renderer_limitations",
            "content_digest",
        ],
        "interpreter_bindings_exact": {
            "cpu": copy.deepcopy(INTERPRETER_BINARY_BINDING),
            "gpu": copy.deepcopy(INTERPRETER_BINARY_BINDING),
        },
    },
    "gpu_environment_receipt": {
        "path": "receipts/gpu_environment.json",
        "schema": "jepa_local_waypoint_planning_cost_gpu_environment_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "python",
            "executable",
            "interpreter_binding",
            "torch",
            "device",
            "packages",
            "foundational_package_roots",
            "foundational_package_closure_policy",
            "import_closure",
            "import_source_bindings",
            "genesis_imported",
            "checkpoint_file_hashes",
            "encoder_source_repository_binding",
            "checkpoint_tensor_open_count",
            "predictor_inference_calls",
            "pass",
            "content_digest",
        ],
        "interpreter_binding_required_keys": ["path", "sha256", "bytes"],
        "interpreter_binding_exact": copy.deepcopy(INTERPRETER_BINARY_BINDING),
        "device_required_keys": [
            "type",
            "index",
            "name",
            "total_memory_bytes",
            "hip",
        ],
        "package_ids": ["numpy", "scipy", "pillow", "pyyaml"],
        "foundational_package_ids": [
            "torch",
            "numpy",
            "scipy",
            "pillow",
            "pyyaml",
        ],
        "foundational_package_root_required_keys": [
            "distribution",
            "version",
            "import_name",
            "package_root",
            "import_resolution",
        ],
        "import_resolution_required_keys": [
            "import_name",
            "find_spec_origin",
            "submodule_search_locations",
            "live_module_file",
            "expected_package_root",
            "resolved_inside_frozen_package_root",
            "pass",
        ],
        "import_closure_exact": [
            "torch",
            "numpy",
            "scipy",
            "PIL",
            "yaml",
            "scripts.dev_frozen_dense_representation_encoders_v1",
            "scripts.dev_proprio_predictor_v1",
            "scripts.run_dev_v03_temporal_action_jepa_v1",
            "scripts.build_dev_v03_proprio_action_manifest_v1",
            "scripts.dev_action_slew_reconstruction_v1",
            "scripts.dev_checkpoint_v1",
            "lewm.safety.jepa_local_waypoint_planning_cost_qualification_v1_contract",
        ],
        "preinference_exact": {
            "genesis_imported": False,
            "checkpoint_tensor_open_count": 0,
            "predictor_inference_calls": 0,
            "pass": True,
        },
    },
    "gpu_inference_receipt": {
        "path": "receipts/gpu_inference.json",
        "schema": "jepa_local_waypoint_planning_cost_gpu_inference_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "interpreter_entrypoint",
            "interpreter",
            "environment",
            "device",
            "autocast",
            "model_config",
            "checkpoint_bindings",
            "strict_state_dict_load",
            "eval_mode",
            "inference_mode",
            "requires_grad_all_false",
            "optimizer_absent",
            "training_steps",
            "parameter_state_digest_before",
            "parameter_state_digest_after",
            "parameter_state_unchanged",
            "encoder_call_counts",
            "predictor_call_counts_by_source",
            "batch_manifest",
            "batch_order_validation",
            "source_closure_binding",
            "gpu_environment_receipt_binding",
            "gpu_environment_revalidation",
            "gpu_watchdog_status",
            "encoder_source_repository_binding",
            "preprocessing_digest",
            "dynamic_oom_batch_fallback_used",
            "future_input_fields",
            "latent_tensor_index_binding",
            "pass",
            "content_digest",
        ],
        "interpreter_entrypoint_exact": "/home/andrewknowles/TinyQuadJEPA/bin/python",
        "interpreter_binding_exact": copy.deepcopy(INTERPRETER_BINARY_BINDING),
        "gpu_environment_revalidation_exact": {
            "interpreter_exact": True,
            "versions_exact": True,
            "device_exact": True,
            "foundational_roots_exact": True,
            "precheckpoint": True,
            "pass": True,
        },
        "gpu_watchdog_status_exact": copy.deepcopy(GPU_WATCHDOG_STATUS_SUCCESS),
    },
    "cpu_runtime_input_inventory": {
        "path": "materialization/cpu_runtime_input_inventory.json",
        "schema": "jepa_local_waypoint_cpu_runtime_input_inventory_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "platform_manifest",
            "primitive_registry",
            "policy_artifacts",
            "genesis_builtin_urdf",
            "prefreeze_scene_byte_inventory_binding",
            "prefreeze_scene_byte_inventory_records",
            "prefreeze_scene_byte_inventory_validation",
            "scene_records",
            "texture_root",
            "texture_records",
            "texture_selection_validation",
            "box_obj_cache",
            "package_roots",
            "foundational_package_roots",
            "foundational_package_closure_policy",
            "package_record_closure_algorithm",
            "historical_renderer_limitations",
            "renderer_structure_schema_audit",
            "historical_renderer_limitation_validation",
            "counts",
            "outcome_fields_read",
            "pass",
            "content_digest",
        ],
        "scene_record_required_keys": [
            "state_id",
            "scene_id",
            "scene_dir",
            "manifest_path",
            "manifest_sha256",
            "manifest_bytes",
            "declared_manifest_sha256",
            "recomputed_manifest_sha256",
            "genesis_scene_path",
            "genesis_scene_sha256",
            "genesis_scene_bytes",
            "genesis_scene_declared_manifest_sha256",
            "identity_and_internal_digest_consistent",
            "genesis_scene_top_level_keys",
            "historical_builder_structural_keys_present",
            "genesis_scene_object_records",
            "historical_floor_only_renderer",
            "texture_selection_by_category",
            "texture_categories_reached_by_renderer",
        ],
        "prefreeze_scene_byte_inventory_record_required_keys": [
            "state_id",
            "scene_id",
            "kind",
            "path",
            "sha256",
            "bytes",
        ],
        "prefreeze_scene_byte_inventory_validation_required_keys": [
            "record_count",
            "states",
            "total_bytes",
            "canonical_sorted_path_sha_bytes_aggregate_sha256",
            "validated_before_json_parse",
            "outcome_fields_parsed_before_validation",
            "pass",
        ],
        "texture_record_required_keys": ["path", "sha256", "bytes"],
        "package_ids": ["genesis", "rsl_rl", "tensordict"],
        "foundational_package_ids": [
            "torch",
            "numpy",
            "scipy",
            "pillow",
            "pyyaml",
        ],
        "package_root_required_keys": [
            "distribution",
            "version",
            "package_root",
            "record_closure",
            "import_resolution",
        ],
        "package_record_closure_required_keys": [
            "record_path",
            "record_sha256",
            "record_bytes",
            "record_entries",
            "declared_hash_entries",
            "present_files",
            "absent_unhashed_files",
            "absent_unhashed_path_list_sha256",
            "present_file_bytes",
            "present_file_aggregate_sha256",
        ],
        "foundational_package_root_required_keys": [
            "distribution",
            "version",
            "import_name",
            "package_root",
            "import_resolution",
        ],
        "import_resolution_required_keys": [
            "import_name",
            "find_spec_origin",
            "submodule_search_locations",
            "live_module_file",
            "expected_package_root",
            "resolved_inside_frozen_package_root",
            "pass",
        ],
        "genesis_urdf_required_keys": [
            "interpreter_relative_path",
            "resolved_path",
            "sha256",
            "bytes",
            "referenced_meshes",
            "mesh_reference_validation",
        ],
        "genesis_mesh_record_required_keys": [
            "name",
            "relative_path",
            "path",
            "sha256",
            "bytes",
        ],
        "counts_exact": {
            "states": 48,
            "scene_records": 48,
            "manifest_files": 48,
            "genesis_scene_files": 48,
            "textures": 12,
            "genesis_referenced_meshes": 7,
        },
        "box_obj_cache_exact": {
            "execution_status": "NOT_REACHED_BY_FROZEN_HISTORICAL_RENDERER",
            "scientific_runtime_input": False,
            "files_opened_or_used": 0,
            "validation_required_for_execution": False,
        },
        "renderer_structure_schema_audit_exact": {
            "scene_records": 48,
            "genesis_scene_records_with_objects": 48,
            "genesis_scene_records_with_nonempty_objects": 48,
            "total_genesis_scene_objects": 3067,
            "genesis_scene_records_with_walls": 0,
            "genesis_scene_records_with_obstacles": 0,
            "genesis_scene_records_with_landmarks": 0,
            "effective_scene_geometry": "FLOOR_PLANE_ONLY",
        },
        "outcome_fields_read_exact": [],
        "lifecycle": (
            "write after freeze and before CPU materialisation; revalidate immediately "
            "before and after CPU materialisation and during terminal check"
        ),
    },
    "context_reconstruction_index": {
        "path": "materialization/context_reconstruction_index.json",
        "schema": "jepa_local_waypoint_context_reconstruction_index_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "cpu_runtime_input_inventory_binding",
            "states",
            "records",
            "source_frame_offsets",
            "command_tick_offsets",
            "elapsed_offsets_s",
            "warmup_block_boundaries",
            "reconstruction_prefix_custody",
            "branch_snapshot_authority_validation",
            "manifest_snapshot_descriptive_audit",
            "current_rgb_authority_reproduction",
            "current_token_authority_reproduction",
            "control_history_index_timestamp_validation",
            "requested_applied_action_authority_validation",
            "failed_state_ids",
            "content_digest",
        ],
        "record_required_keys": [
            "state_id",
            "family",
            "role",
            "branch_snapshot_digest",
            "replay_snapshot_digest",
            "manifest_snapshot_digest_or_null",
            "manifest_snapshot_digest_match_or_null",
            "context_rgb_paths",
            "context_rgb_sha256s",
            "context_source_frame_indices",
            "context_command_tick_indices",
            "context_elapsed_s",
            "control_command_indices",
            "control_timestamps",
            "control_history_raw_3x5x2",
            "control_history_normalized_3x5x2",
            "action_blocks_raw_3x5x2_by_candidate",
            "action_blocks_raw_3x10_by_candidate",
            "requested_action_blocks_raw_3x5x3_by_candidate",
            "applied_action_blocks_raw_3x5x3_by_candidate",
            "requested_applied_action_authority_validation",
            "pass",
        ],
        "reconstruction_prefix_custody_exact": copy.deepcopy(
            RECONSTRUCTION_PREFIX_CUSTODY
        ),
    },
    "dense_route_replay_input_index": {
        "path": "materialization/dense_route_replay_input_index.json",
        "schema": "jepa_local_waypoint_dense_route_replay_input_index_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "evidence_receipt_binding",
            "prefreeze_byte_inventory_binding",
            "prefreeze_byte_inventory_records",
            "prefreeze_byte_inventory_validation",
            "states",
            "records",
            "cardinality_validation",
            "self_digest_validation",
            "failed_state_ids",
            "pass",
            "content_digest",
        ],
        "record_required_keys": [
            "state_id",
            "path",
            "sha256",
            "bytes",
            "schema",
            "status",
            "content_digest",
            "self_digest_valid",
            "branches",
            "h3_tick_count",
            "horizon_tick_boundaries",
            "pass",
        ],
        "prefreeze_byte_inventory_record_required_keys": [
            "state_id",
            "path",
            "sha256",
            "bytes",
        ],
        "prefreeze_byte_inventory_validation_required_keys": [
            "record_count",
            "total_bytes",
            "canonical_sorted_path_sha_bytes_aggregate_sha256",
            "validated_before_json_parse",
            "outcome_fields_parsed_before_validation",
            "pass",
        ],
        "cardinality": {
            "states": 48,
            "branches_per_state": 12,
            "h3_tick_count": 15,
            "horizon_tick_boundaries": [5, 10, 15],
        },
        "evidence_receipt": {
            "path": (
                ".generated/dense_temporal_true_future_safety_observability_v1/"
                "evidence_receipt.json"
            ),
            "sha256": "a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1",
            "bytes": 1484,
        },
    },
    "oracle_admissibility_fanout_index": {
        "path": "materialization/oracle_admissibility_fanout_index.json",
        "schema": "jepa_local_waypoint_oracle_admissibility_fanout_index_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "cpu_runtime_input_inventory_binding",
            "states",
            "current_blocks",
            "successor_blocks",
            "physics_frames",
            "records",
            "macro_candidate_to_first_primitive",
            "existing_h1_contact_cross_validation",
            "raw_continuation_policy_validation",
            "cpu_watchdog_status",
            "failed_state_ids",
            "content_digest",
        ],
        "record_required_keys": [
            "state_id",
            "family",
            "role",
            "branch_snapshot_digest",
            "shard_path",
            "shard_sha256",
            "shard_bytes",
            "array_manifest",
            "current_contact_bitset_shape",
            "successor_contact_bitset_shape",
            "first_contact_step",
            "first_contact_link",
            "current_action_mapping",
            "successor_action_mapping",
            "successor_safe_action_count",
            "successor_viable",
            "oracle_viability_admissible",
            "raw_continuation_snapshots",
            "h1_contact_cross_validation",
            "pass",
        ],
        "array_requirements": {
            "current_contact_bitset": {"shape": [9, 250], "dtype": "bool"},
            "successor_contact_bitset": {"shape": [9, 9, 250], "dtype": "bool"},
            "all_current_and_successor_rows_materialized": True,
        },
        "raw_continuation_snapshot_required_keys": [
            "current_primitive_index",
            "current_primitive_id",
            "capture_mode",
            "snapshot_digest",
            "terminal_flags",
            "consecutive_tipped_blocks",
            "production_reset_checks_suppressed",
            "evaluation_only",
            "restored_for_successors",
        ],
        "cpu_watchdog_status_exact": copy.deepcopy(CPU_WATCHDOG_STATUS_SUCCESS),
        "raw_continuation_snapshots_per_state": 9,
    },
    "latent_tensor_index": {
        "path": "latents/tensor_index.json",
        "schema": "jepa_local_waypoint_latent_tensor_index_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "records",
            "counts_by_kind",
            "total_records",
            "total_bytes",
            "unique_tensor_payload_count",
            "external_true_future_index_binding",
            "external_current_index_binding",
            "encoder_binding",
            "checkpoint_bindings",
            "batch_manifest",
            "shape_dtype_validation",
            "failed_records",
            "content_digest",
        ],
        "record_required_keys": [
            "kind",
            "state_id",
            "family",
            "role",
            "candidate_index_or_null",
            "horizon_or_null",
            "source_or_null",
            "path",
            "sha256",
            "bytes",
            "shape",
            "dtype",
            "external_existing_artifact",
        ],
        "kind_ids": [
            "CONTEXT",
            "CURRENT",
            "GOAL",
            "TRUE_FUTURE",
            "ONE_STEP_PREDICTED",
            "TWO_STEP_PREDICTED",
        ],
        "expected_counts": {
            "CONTEXT": 144,
            "CURRENT": 48,
            "GOAL": 48,
            "TRUE_FUTURE": 1728,
            "ONE_STEP_PREDICTED": 1728,
            "TWO_STEP_PREDICTED": 1728,
            "total": 5424,
        },
        "current_alias_rule": (
            "each CURRENT logical record is an exact path/SHA/bytes alias of that "
            "state's context slot at elapsed 0 s (post block 40); do not duplicate bytes"
        ),
        "unique_tensor_payload_count": (
            "derive and persist from unique (path,sha256) pairs; not predeclared because "
            "existing true-future identities may share a frozen frame"
        ),
        "tensor_contract": {"shape": [768, 1024], "dtype": "float16"},
    },
    "goal_view_index": {
        "path": "goal_views/index.json",
        "schema": "jepa_local_waypoint_goal_view_index_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "states",
            "records",
            "goal_view_execution_amendment_binding",
            "goal_pose_semantics",
            "goal_cell_classification_counts",
            "goal_cell_classification_validation",
            "renderer_sha256",
            "encoder_checkpoint_sha256",
            "candidate_independence_validation",
            "failed_state_ids",
            "content_digest",
        ],
        "record_required_keys": [
            "state_id",
            "family",
            "role",
            "waypoint_world_xy",
            "goal_body_dx_dy_sin_dyaw_cos_dyaw",
            "snapshot_base_z",
            "waypoint_path_cells",
            "path_cell_centers_world_xy",
            "goal_cell_preconditions",
            "goal_cell_endpoint_reachable",
            "goal_cell_nav_blocked",
            "goal_cell_block_classification",
            "goal_cell_is_beacon_endpoint",
            "goal_cell_is_low_clearance_transit_blocked",
            "goal_render_semantics",
            "goal_pose_world_xyz_rpy",
            "branch_snapshot_digest_expected",
            "snapshot_digest_observed",
            "manifest_waypoint_xy_or_null",
            "manifest_waypoint_body_xy_or_null",
            "manifest_field_equality_audit",
            "rgb_path",
            "rgb_sha256",
            "token_path",
            "token_sha256",
            "token_shape",
            "pass",
        ],
        "goal_view_execution_amendment_binding_exact": copy.deepcopy(
            GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
        ),
        "goal_pose_semantics_exact": copy.deepcopy(
            GOAL_VIEW_EXECUTION_AMENDMENT["amended_goal_pose_semantics"]
        ),
        "goal_cell_classification_ids": list(GOAL_CELL_BLOCK_CLASSIFICATIONS),
        "goal_cell_classification_counts_exact": copy.deepcopy(
            GOAL_CELL_CLASSIFICATION_COUNTS
        ),
        "goal_cell_classification_validation_exact": copy.deepcopy(
            GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS
        ),
        "goal_cell_precondition_required_keys": [
            "path_cells",
            "path_cell_centers_world_xy",
            "goal_cell",
            "goal_path_cell_ids_valid",
            "goal_path_consecutive_edge_pairs",
            "goal_path_consecutive_edges_traversable",
            "goal_cell_endpoint_reachable",
            "goal_cell_endpoint_bfs_hops",
            "goal_cell_nav_blocked",
            "goal_cell_block_classification",
            "goal_cell_is_beacon_endpoint",
            "goal_cell_is_low_clearance_transit_blocked",
            "goal_render_semantics",
            "pass",
        ],
        "goal_render_semantics_exact": GOAL_VIEW_RENDER_SEMANTICS,
        "path1_position_substitution": "forbidden",
        "state_drop_or_alternate_goal_search": "forbidden",
    },
    "candidate_evidence": {
        "path": "evidence/candidate_evidence.jsonl.gz",
        "schema": "jepa_local_waypoint_candidate_evidence_v1",
        "required_keys": [
            "schema",
            "state_id",
            "family",
            "role",
            "candidate_index",
            "candidate_identity",
            "source",
            "population_membership",
            "context_frame_identities",
            "context_frame_sha256s",
            "context_offsets_source_frames",
            "context_offsets_command_ticks",
            "context_offsets_elapsed_s",
            "context_control_history_raw",
            "context_control_history_normalized",
            "action_blocks_raw_3x5x2",
            "action_blocks_raw_3x10",
            "requested_action_blocks_raw_3x5x3",
            "applied_action_blocks_raw_3x5x3",
            "requested_applied_action_authority_validation",
            "snapshot_digest",
            "goal_view_sha256",
            "goal_token_sha256",
            "candidate_token_sha256_by_horizon",
            "cost_current",
            "cost_h1",
            "cost_h2",
            "cost_h3",
            "monotonic_diagnostics",
            "realised_route_fields_by_horizon",
            "oracle_route_fields_h3_primary",
            "dense_route_replay_state_record_ref",
            "immediate_contact_h1",
            "descriptive_contact_h2",
            "descriptive_contact_h3",
            "contact_free_h1",
            "successor_safe_action_count",
            "successor_viable",
            "oracle_viability_admissible",
            "successor_nonviable",
            "stuck",
            "completed",
            "current_contact_bitset_shard_ref",
            "successor_contact_bitset_shard_ref",
            "input_contract_validation",
        ],
        "realised_route_horizon_ids": ["H1", "H2", "H3"],
        "realised_route_horizon_required_keys": [
            "p_d_m",
            "p_theta_rad",
            "completed",
            "descriptive_contact",
            "stuck",
        ],
    },
    "selection_evidence": {
        "path": "evidence/selection_evidence.jsonl.gz",
        "schema": "jepa_local_waypoint_selection_evidence_v1",
        "required_keys": [
            "schema",
            "state_id",
            "family",
            "role",
            "source_or_comparator",
            "population",
            "eligible_candidate_indices",
            "ranked_candidate_indices",
            "selected_candidate_index",
            "oracle_best_candidate_index",
            "abstained",
            "selected_immediate_contact_h1",
            "selected_descriptive_contact_h2",
            "selected_descriptive_contact_h3",
            "selected_nonviable_successor",
            "selected_stuck",
            "selected_progress_m",
            "selected_heading_improvement_rad",
            "selected_combined_utility",
            "oracle_best_progress_m",
            "normalised_regret",
            "best_route_rank",
            "completed",
            "cost_spread",
            "cost_tie_pair_count",
            "cost_tie_pair_rate",
            "ordered_pair_count",
            "pairwise_denominator_ordered_pairs",
            "pairwise_correct_credit",
            "best_route_reciprocal_rank",
        ],
    },
    "paired_effect_evidence": {
        "path": "evidence/paired_effect_evidence.jsonl.gz",
        "schema": "jepa_local_waypoint_paired_effect_evidence_v1",
        "required_keys": [
            "schema",
            "comparison_id",
            "state_id",
            "family",
            "role",
            "population",
            "left_source_or_comparator",
            "right_source_or_comparator",
            "left_selected_candidate_index",
            "right_selected_candidate_index",
            "left_selected_progress_m",
            "right_selected_progress_m",
            "selected_progress_effect_m",
            "left_normalised_regret",
            "right_normalised_regret",
            "normalised_regret_improvement",
            "left_pairwise_accuracy",
            "right_pairwise_accuracy",
            "pairwise_accuracy_effect",
            "left_best_route_rank",
            "right_best_route_rank",
            "best_route_rank_improvement",
            "left_selected_immediate_contact_h1",
            "right_selected_immediate_contact_h1",
            "contact_selection_delta",
            "left_selected_nonviable_successor",
            "right_selected_nonviable_successor",
            "nonviable_selection_delta",
        ],
        "comparison_ids": list(PAIRED_COMPARISON_IDS),
        "population": "ORACLE_VIABILITY_ADMISSIBLE",
        "role": "heldout",
        "expected_rows": 4 * 8,
        "effect_conventions": {
            "selected_progress_effect_m": "left minus right",
            "normalised_regret_improvement": "right minus left; positive favours left",
            "pairwise_accuracy_effect": "left minus right",
            "best_route_rank_improvement": "right minus left; positive favours left",
            "contact_selection_delta": "left Boolean-as-integer minus right",
            "nonviable_selection_delta": "left Boolean-as-integer minus right",
        },
    },
    "aggregate_metrics": {
        "path": "aggregates/metrics.json",
        "schema": "jepa_local_waypoint_planning_cost_metrics_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "by_source_population",
            "by_source_population_family",
            "per_state",
            "per_role",
            "paired_comparisons",
            "bootstrap",
            "family_collapse",
            "latent_progress_diagnostics",
            "all_candidate_tendency_diagnostics",
            "gates",
            "classification",
            "row_reproduction",
            "content_digest",
        ],
        "classification_required_keys": [
            "schema",
            "primary_classification",
            "secondary_classifications",
            "true_future_gate_classification",
            "two_step_gate_passed",
            "two_step_gate_signal_or_null",
            "predicted_base_screens",
            "diagnostic_flags",
            "precedence",
            "next_experiment",
        ],
        "classification_predicted_base_screen_keys": [
            "ONE_STEP_PREDICTED",
            "TWO_STEP_PREDICTED",
        ],
        "classification_diagnostic_flag_keys": [
            "both_predicted_base_screens_failed",
            "ONE_STEP_BASE_SCREEN_ONLY",
            "two_step_base_screen_passed_but_full_gate_failed",
        ],
        "source_or_comparator_ids": list(COMPARATOR_IDS),
        "population_ids": list(POPULATION_IDS),
        "role_ids": ["fit", "calibration", "heldout"],
        "family_ids": list(FAMILY_IDS),
        "metric_group_required_keys": [
            "states",
            "candidates",
            "ordered_pairs",
            "spearman",
            "kendall",
            "pairwise_accuracy",
            "best_route_top1",
            "best_route_top3",
            "mrr",
            "mean_best_route_rank",
            "cost_spread",
            "cost_tie_pair_count",
            "cost_tie_pair_rate",
            "pairwise_correct_credit",
            "selected_identity_counts",
            "selected_immediate_contacts_h1",
            "selected_descriptive_contacts_h2",
            "selected_descriptive_contacts_h3",
            "selected_nonviable_successors",
            "selected_stuck",
            "selected_progress_sum_m",
            "selected_progress_mean_m",
            "oracle_best_progress_sum_m",
            "selected_progress_fraction_of_oracle_best",
            "selected_heading_improvement_rad",
            "selected_combined_utility",
            "normalised_regret",
            "completion",
            "abstention",
        ],
    },
    "result": {
        "path": "result.json",
        "schema": "jepa_local_waypoint_planning_cost_result_v1",
        "required_keys": [
            "schema",
            "experiment_id",
            "head",
            "source_freeze_commit",
            "contract_sha256",
            "output_schema_sha256",
            "fixture_sha256",
            "source_closure_sha256",
            "goal_view_execution_amendment_binding",
            "seed",
            "materialisation_counts",
            "goal_view_counts",
            "goal_pose_semantics",
            "goal_cell_classification_counts",
            "goal_cell_classification_validation",
            "gpu_inference_custody",
            "gpu_environment_receipt_binding",
            "dense_route_replay_input_index_binding",
            "cpu_runtime_input_inventory_binding",
            "metrics",
            "gates",
            "true_future_gate_classification",
            "two_step_gate_passed",
            "two_step_gate_signal_or_null",
            "predicted_base_screens",
            "diagnostic_flags",
            "paired_materiality",
            "primary_classification",
            "secondary_classifications",
            "next_experiment",
            "requirements_custody",
            "historical_renderer_limitations",
            "reconstruction_prefix_custody",
            "controller_execution_custody",
            "execution_watchdog_status",
            "runtime_s",
            "storage",
            "prohibition_counters",
            "row_reproduction",
            "nothing_running",
            "result_content_sha256",
        ],
        "materialisation_count_required_keys": [
            "states",
            "candidates",
            "reconstruction_prefix_blocks",
            "reconstruction_physics_frames",
            "oracle_fanout_blocks",
            "oracle_fanout_physics_frames",
            "total_simulator_blocks",
            "total_simulator_physics_frames",
            "snapshot_reproductions",
            "current_blocks",
            "successor_blocks",
            "total_oracle_blocks",
            "physics_frames",
            "latent_tensors",
            "predicted_tensors",
            "candidate_evidence_rows",
            "selection_evidence_rows",
            "paired_effect_evidence_rows",
        ],
        "goal_view_count_required_keys": ["states", "views", "failed"],
        "goal_view_execution_amendment_binding_exact": copy.deepcopy(
            GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
        ),
        "goal_pose_semantics_exact": copy.deepcopy(
            GOAL_VIEW_EXECUTION_AMENDMENT["amended_goal_pose_semantics"]
        ),
        "goal_cell_classification_counts_exact": copy.deepcopy(
            GOAL_CELL_CLASSIFICATION_COUNTS
        ),
        "goal_cell_classification_validation_exact": copy.deepcopy(
            GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS
        ),
        "historical_renderer_limitation_required_keys": [
            "effective_scene_geometry",
            "input_schema",
            "builder_schema",
            "structural_walls_obstacles_landmarks_rendered",
            "current_true_future_byte_compatibility_preserved",
            "explicit_wall_visual_reasoning_claim",
            "interpretation",
        ],
        "historical_renderer_limitations_exact": copy.deepcopy(
            HISTORICAL_RENDERER_LIMITATIONS
        ),
        "reconstruction_prefix_custody_exact": copy.deepcopy(
            RECONSTRUCTION_PREFIX_CUSTODY
        ),
        "controller_execution_custody_exact": copy.deepcopy(
            CONTROLLER_EXECUTION_CUSTODY
        ),
        "execution_watchdog_status_exact": copy.deepcopy(
            EXECUTION_WATCHDOG_STATUS_SUCCESS
        ),
        "runtime_required_keys": [
            "preflight",
            "cpu_materialization",
            "gpu_materialization",
            "evaluation_reduction",
            "total",
        ],
        "storage_required_keys": [
            "files",
            "bytes",
            "gb_decimal",
            "final_ceiling_bytes",
            "within_final_ceiling",
            "peak_rss_bytes",
            "peak_vram_bytes",
        ],
    },
    "report": {
        "path": "report.md",
        "required_sections": [
            "Bindings and custody",
            "Goal-view and predictor input reconstruction",
            "Goal-view amendment and virtual-pose limitation",
            "Historical renderer limitation",
            "Controller execution custody",
            "Population and materialisation counts",
            "True-future and predicted ranking metrics",
            "Per-family results and collapse audit",
            "Selected route outcomes",
            "Paired comparisons and descriptive bootstrap",
            "Gates and classifications",
            "Requirements boundary and next decision",
            "Runtime, storage and prohibitions",
        ],
    },
    "persistence_receipt": {
        "path": "receipts/persistence.json",
        "schema": "jepa_local_waypoint_planning_cost_persistence_v1",
        "required_keys": [
            *_PHASE_RECEIPT_REQUIRED_KEYS,
            "artifact_manifest",
            "artifact_manifest_exclusions",
            "artifact_count",
            "total_bytes",
            "row_counts",
            "row_reproduction",
            "context_reconstruction_index_binding",
            "goal_view_execution_amendment_binding",
            "dense_route_replay_input_index_binding",
            "cpu_runtime_input_inventory_binding",
            "oracle_admissibility_fanout_index_binding",
            "latent_tensor_index_binding",
            "gpu_inference_receipt_binding",
            "gpu_environment_receipt_binding",
            "external_input_index_bindings",
            "all_shard_hash_shape_dtype_byte_validation",
            "tensor_to_cost_row_reproduction",
            "cost_row_to_aggregate_reproduction",
            "contract_schema_validation",
            "execution_watchdog_status",
            "prohibition_counters",
            "nothing_running",
            "content_digest",
        ],
        "artifact_manifest_exclusions_exact": [
            "receipts/persistence.json",
            "result.json",
            "receipts/RUNNING.json",
        ],
        "execution_watchdog_status_exact": copy.deepcopy(
            EXECUTION_WATCHDOG_STATUS_SUCCESS
        ),
        "goal_view_execution_amendment_binding_exact": copy.deepcopy(
            GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
        ),
        "report_integrity": (
            "report.md is included in artifact_manifest with exact path, SHA-256 and bytes"
        ),
    },
}


def _output_schema_core() -> dict[str, Any]:
    return {
        "schema": OUTPUT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "output_root": str(OUTPUT_ROOT),
        "canonical_json": "UTF-8, sorted keys, compact separators, one trailing LF",
        "row_ledgers": "gzip JSONL; one canonical compact JSON object plus LF per row",
        "files": copy.deepcopy(_OUTPUT_FILES),
        "candidate_evidence_cardinality": {
            "unit": "one row per evaluable state x candidate x latent source",
            "maximum_expected": 48 * 12 * 3,
            "all_roles_persisted": True,
        },
        "selection_evidence_cardinality": {
            "unit": "one row per evaluable state x comparator/source x population",
            "maximum_expected": 48 * 5 * 3,
            "all_roles_persisted": True,
        },
        "paired_effect_evidence_cardinality": {
            "unit": (
                "one row per heldout state x four frozen comparisons under "
                "ORACLE_VIABILITY_ADMISSIBLE"
            ),
            "expected": 4 * 8,
            "comparison_ids": list(PAIRED_COMPARISON_IDS),
        },
        "aggregate_reproduction": {
            "tensor_to_cost_row": {
                "reload_every_bound_context_current_goal_true_future_and_predicted_fp16_grid": True,
                "reapply_canonical_numpy_cpu_fp32_layernorm_and_token_cosine": True,
                "predictor_inference": False,
                "checkpoint_open": False,
                "require_exact_scalar_row_equality": True,
            },
            "cost_row_to_aggregate": {
                "requires_predictor_inference": False,
                "requires_checkpoint_open": False,
                "requires_tensor_open": False,
                "require_exact_metrics_gates_classification_equality": True,
            },
            "source_rows": [
                "evidence/candidate_evidence.jsonl.gz",
                "evidence/selection_evidence.jsonl.gz",
                "evidence/paired_effect_evidence.jsonl.gz",
            ],
            "external_bound_indices": [
                ".generated/safe_local_waypoint_route_intent_v2/target_latent_index.json",
                ".generated/dense_temporal_true_future_safety_observability_v1/token_index.json",
                "materialization/dense_route_replay_input_index.json",
            ],
        },
        "atomic_publication": {
            "temporary_namespace": "sibling hidden run-specific directory on output ext4",
            "publish": "atomic rename only after complete validation",
            "partial_run": (
                "archive the complete untouched hidden attempt namespace with a "
                "self-digesting failure receipt; a later attempt starts from a new empty "
                "hidden namespace and reuses no scientific phase or shard; partial phase "
                "evidence is never resumed or copied into the canonical output"
            ),
            "bound_prior_failed_attempt": {
                "original_freeze_commit": ORIGINAL_FREEZE_COMMIT,
                "archive_path": str(FAILED_GOAL_VIEW_ATTEMPT_ARCHIVE),
                "inventory": copy.deepcopy(FAILED_GOAL_VIEW_ATTEMPT_INVENTORY),
                "reuse": False,
            },
        },
        "storage_ceilings_gb": {"temporary": 20, "final": 12},
        "prohibition_counter_ids": list(PROHIBITION_COUNTER_IDS),
        "raw_tensor_policy": (
            "Persist all predicted, goal and context latents as float16 with hashes and "
            "indices; scalar rows must still reproduce aggregates without inference."
        ),
        "forbidden_paths_or_payloads": [
            "untouched G2",
            "new route panel",
            "modified checkpoint",
            "trained model",
            "memory or experimental candidate-selecting navigation output",
        ],
    }


def build_contract() -> dict[str, Any]:
    return _self_digest(_contract_core(), "contract_sha256")


def build_output_schema() -> dict[str, Any]:
    return _self_digest(_output_schema_core(), "output_schema_sha256")


CONTRACT = build_contract()
CONTRACT_SHA256 = CONTRACT["contract_sha256"]
OUTPUT_SCHEMA = build_output_schema()
OUTPUT_SCHEMA_SHA256 = OUTPUT_SCHEMA["output_schema_sha256"]


def contract_receipt() -> dict[str, Any]:
    return build_contract()


def contract_receipt_bytes() -> bytes:
    return canonical_json_bytes(build_contract()) + b"\n"


def output_schema_receipt_bytes() -> bytes:
    return canonical_json_bytes(build_output_schema()) + b"\n"


CONTRACT_RECEIPT_SHA256 = hashlib.sha256(contract_receipt_bytes()).hexdigest()
OUTPUT_SCHEMA_RECEIPT_SHA256 = hashlib.sha256(output_schema_receipt_bytes()).hexdigest()


def _run_contract_fixture_checks() -> dict[str, bool]:
    """Execute the small outcome-free reference checks embedded in the fixture."""

    import numpy as np

    from lewm.safety import jepa_local_waypoint_planning_cost_metrics_v1 as metrics

    checks: dict[str, bool] = {}
    cost_cases = (
        (
            "token_identical",
            [[1.0, -1.0, 0.0]],
            [[1.0, -1.0, 0.0]],
            5.960464477539063e-08,
            1e-15,
        ),
        (
            "token_orthogonal",
            [[1.0, -1.0, 0.0]],
            [[1.0, 1.0, -2.0]],
            1.0,
            2e-7,
        ),
        (
            "token_opposite",
            [[1.0, -1.0, 0.0]],
            [[-1.0, 1.0, 0.0]],
            2.0,
            1e-15,
        ),
        (
            "tokenwise_not_global",
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, -3.0]],
            1.0000000298023224,
            1e-15,
        ),
        ("zero_norm_floor", [[0.0, 0.0]], [[1.0, -1.0]], 1.0, 1e-12),
    )
    for name, candidate, goal, expected, tolerance in cost_cases:
        observed = metrics.tokenwise_normalized_cosine_mean_cost(
            np.asarray(candidate, dtype=np.float16),
            np.asarray(goal, dtype=np.float16),
        )
        checks[name] = abs(observed - expected) <= tolerance
    checks["goal_heading_east"] = route_heading_yaw([0.0, 0.0], [1.0, 0.0]) == 0.0
    checks["goal_heading_north"] = abs(
        route_heading_yaw([0.0, 0.0], [0.0, 1.0]) - math.pi / 2.0
    ) <= 1e-15
    try:
        route_heading_yaw([1.0, 1.0], [1.0, 1.0])
    except ContractError:
        checks["identical_goal_cells_fail_closed"] = True
    else:
        checks["identical_goal_cells_fail_closed"] = False
    try:
        metrics.tokenwise_normalized_cosine_mean_cost(
            np.asarray([[float("nan"), 0.0]], dtype=np.float16),
            np.asarray([[1.0, -1.0]], dtype=np.float16),
        )
    except metrics.PlanningCostMetricsError:
        checks["nonfinite_token_fail_closed"] = True
    else:
        checks["nonfinite_token_fail_closed"] = False
    try:
        metrics.tokenwise_normalized_cosine_mean_cost(
            np.asarray([[1.0, 0.0]], dtype=np.float16),
            np.asarray([[1.0, -1.0], [0.0, 1.0]], dtype=np.float16),
        )
    except metrics.PlanningCostMetricsError:
        checks["wrong_token_shape_fail_closed"] = True
    else:
        checks["wrong_token_shape_fail_closed"] = False
    classification_cases = (
        ((False, False, False, False), "RAW_LATENT_GOAL_COST_NO_GO"),
        ((True, True, True, False), "KINEMATIC_BASELINE_DOMINANT"),
        ((True, True, False, True), "TWO_STEP_JEPA_PLANNING_COST_SIGNAL"),
        (
            (True, False, False, False),
            "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO",
        ),
    )
    checks["classification_precedence"] = all(
        derive_primary_classification(
            true_future_gate_passed=inputs[0],
            two_step_gate_passed=inputs[1],
            kinematic_baseline_materially_superior=inputs[2],
            jepa_incremental_route_value=inputs[3],
        )
        == expected
        for inputs, expected in classification_cases
    )
    checks["family_collapse_chance_only"] = family_complete_collapse(
        {
            "evaluable_nonabstaining_states": 1,
            "ordered_pairs": 1,
            "pairwise_accuracy": 0.5,
            "best_route_top3_rate": 0.0,
            "selected_route_progress_sum_m": 0.0,
        }
    )
    counts = GOAL_CELL_CLASSIFICATION_COUNTS
    checks["goal_cell_classification_partition"] = (
        counts["nav_blocked"]
        == counts["beacon_endpoint"] + counts["low_clearance_transit_blocked"]
        and counts["states"] == counts["nav_blocked"] + counts["unblocked"]
        and counts["endpoint_reachable"] == counts["states"] == 48
    )
    checks["goal_cell_blocked_identity_partition"] = (
        len(GOAL_CELL_BLOCKED_STATE_IDS["beacon_endpoint"]) == 13
        and len(GOAL_CELL_BLOCKED_STATE_IDS["low_clearance_transit_blocked"]) == 1
        and len(
            set(GOAL_CELL_BLOCKED_STATE_IDS["beacon_endpoint"])
            | set(GOAL_CELL_BLOCKED_STATE_IDS["low_clearance_transit_blocked"])
        )
        == 14
    )
    checks["goal_view_amendment_is_prospective_and_no_reuse"] = (
        GOAL_VIEW_EXECUTION_AMENDMENT["status"]
        == "PROSPECTIVE_BEFORE_FRESH_REEXECUTION"
        and GOAL_VIEW_EXECUTION_AMENDMENT["failed_attempt"][
            "scientific_phase_or_shard_reuse"
        ]
        is False
        and GOAL_VIEW_EXECUTION_AMENDMENT["static_diagnosis"][
            "route_outcome_rows_read_or_used"
        ]
        == 0
    )
    artifact_summary = GOAL_VIEW_EXECUTION_AMENDMENT["failed_attempt"][
        "artifact_summary"
    ]
    checks["failed_attempt_artifact_summary_matches_inventory"] = (
        artifact_summary["worker_logs"]
        + artifact_summary["partial_context_rgb_files"]
        + artifact_summary["preexecution_or_environment_receipts"]
        + artifact_summary["cpu_runtime_input_inventory_receipts"]
        + artifact_summary["failure_or_running_marker_receipts"]
        == FAILED_GOAL_VIEW_ATTEMPT_INVENTORY["record_count"]
        == 41
    )
    checks["goal_view_preworker_static_validation"] = (
        GOAL_VIEW_STATIC_VALIDATION_SUCCESS["pass"] is True
        and GOAL_VIEW_STATIC_VALIDATION_SUCCESS["route_outcome_rows_read"] == 0
        and GOAL_VIEW_STATIC_VALIDATION_SUCCESS["blocked_state_ids"]
        == GOAL_CELL_BLOCKED_STATE_IDS
    )
    checks["cpu_worker_allocator_mitigation"] = (
        EXECUTION_WATCHDOGS["cpu_worker_environment"]["MALLOC_ARENA_MAX"] == "1"
        and EXECUTION_WATCHDOGS["cpu_worker_environment"]["workers"] == 32
        and EXECUTION_WATCHDOGS["cpu_worker_environment"][
            "scientific_semantics_change"
        ]
        is False
    )
    checks["canonical_byte_regeneration"] = canonical_json_bytes(
        {"z": 2, "a": [True, None, 1.25]}
    ) == b'{"a":[true,null,1.25],"z":2}'
    return checks


def build_fixture_receipt() -> dict[str, Any]:
    """Build deterministic, outcome-free fixture declarations and expectations."""

    core = {
        "schema": FIXTURE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "outcome_rows_read": 0,
        "checkpoint_tensors_opened": 0,
        "predictor_inference_calls": 0,
        "fixtures": {
            "token_cosine": {
                "canonical_reducer": (
                    "lewm.safety.jepa_local_waypoint_planning_cost_metrics_v1."
                    "tokenwise_normalized_cosine_mean_cost"
                ),
                "input_dtype": "float16",
                "identical": {
                    "candidate": [[1.0, -1.0, 0.0]],
                    "goal": [[1.0, -1.0, 0.0]],
                    "cost": 5.960464477539063e-08,
                },
                "orthogonal_after_layer_norm": {
                    "candidate": [[1.0, -1.0, 0.0]],
                    "goal": [[1.0, 1.0, -2.0]],
                    "cost": 1.0,
                },
                "opposite": {
                    "candidate": [[1.0, -1.0, 0.0]],
                    "goal": [[-1.0, 1.0, 0.0]],
                    "cost": 2.0,
                },
                "tokenwise_not_global": {
                    "candidate": [[1.0, 0.0], [0.0, 2.0]],
                    "goal": [[1.0, 0.0], [0.0, -3.0]],
                    "cost": 1.0000000298023224,
                },
                "zero_norm_after_layer_norm": {"cost": 1.0, "l2_floor": 1e-12},
                "nonfinite": "FAIL_CLOSED",
                "wrong_shape": "FAIL_CLOSED",
                "torch_vs_canonical_numpy": {
                    "synthetic_grids": "identical, orthogonal, opposite, zero and mixed",
                    "absolute_tolerance": 1e-6,
                    "canonical_on_disagreement": (
                        "source-closed NumPy metrics reducer; contract helper is fixture-only"
                    ),
                },
            },
            "goal_heading": {
                "east": {"centres": [[0.0, 0.0], [1.0, 0.0]], "yaw": 0.0},
                "north": {"centres": [[0.0, 0.0], [0.0, 1.0]], "yaw": math.pi / 2.0},
                "identical_cells": "FAIL_CLOSED",
                "path2_world_xy_and_body_coordinates": "PASS_REQUIRED",
                "optional_manifest_waypoint_absent": "PASS_REQUIRED",
                "present_manifest_waypoint_mismatch": "FAIL_CLOSED",
                "path2_nav_blocked_but_endpoint_reachable": "PASS_WITH_BLOCK_CLASSIFICATION",
                "path2_endpoint_unreachable": "FAIL_CLOSED",
                "path1_position_substitution": "FAIL_CLOSED",
                "goal_render_semantics": GOAL_VIEW_RENDER_SEMANTICS,
                "goal_cell_classification_counts": copy.deepcopy(
                    GOAL_CELL_CLASSIFICATION_COUNTS
                ),
                "preworker_static_validation": copy.deepcopy(
                    GOAL_VIEW_STATIC_VALIDATION_SUCCESS
                ),
                "candidate_independence": "BYTE_IDENTICAL_ACROSS_12_CANDIDATES",
            },
            "predictor_input": {
                "observed_context_boundaries": [38, 39, 40],
                "context_offsets_source_frames": [-480, -240, 0],
                "context_offsets_command_ticks": [-10, -5, 0],
                "context_offsets_elapsed_s": [-1.0, -0.5, 0.0],
                "action_dimension": 10,
                "control_history_shape": [3, 5, 2],
                "initial_control_normalised": True,
                "autoregressive_appended_control_raw": True,
                "duplicated_or_fabricated_context": "FAIL_CLOSED",
                "snapshot_mismatch": "FAIL_CLOSED",
                "branch_ledger_snapshot_12_row_agreement": "PASS_REQUIRED",
                "manifest_snapshot_absence_or_historical_mismatch": "DESCRIPTIVE_ONLY",
                "mixed_scale_control_append": "RAW_APPEND_REQUIRED",
                "one_step_and_two_step_identical_unroll_api": "PASS_REQUIRED",
                "future_context_or_latent_injection": "FAIL_CLOSED",
                "vy_nonzero": "FAIL_CLOSED",
            },
            "population": {
                "proper_nesting": "PASS",
                "viability_outside_contact_free": "FAIL_CLOSED",
                "empty_population": "ABSTAIN",
                "h1_free_h3_contacting_candidate": "IN_ORACLE_CONTACT_FREE",
                "h1_contacting_h3_free_candidate": "OUT_OF_ORACLE_CONTACT_FREE",
                "one_safe_successor_of_9": "VIABILITY_ADMISSIBLE",
                "zero_safe_successors_of_9": "NOT_VIABILITY_ADMISSIBLE",
                "all_9x9_rows_materialized_even_after_contact": "PASS_REQUIRED",
            },
            "route_order": {
                "completion_precedes_progress": "PASS_REQUIRED",
                "distance_beyond_0_03_precedes_heading": "PASS_REQUIRED",
                "distance_within_0_03_uses_5deg_heading_margin": "PASS_REQUIRED",
                "oracle_unordered_pair_excluded": "PASS_REQUIRED",
                "cost_tie_within_1e_12_half_credit": 0.5,
                "selection_tie_lowest_candidate_index": "PASS_REQUIRED",
                "margin_borda": "PASS_REQUIRED",
                "zero_progress_range_equal_scores_regret": 0.0,
                "zero_progress_range_unequal_scores": "FAIL_CLOSED",
            },
            "kinematic": {
                "input_shape": [3, 5, 3],
                "commands_integrated": 15,
                "command_dt_s": 0.1,
                "vy_formula_retained": True,
                "nominal_completion_threshold": None,
                "heading_deadband_deg": None,
                "distance_near_set_m": 0.03,
            },
            "random": {
                "exact_byte_rule": "PASS_REQUIRED",
                "same_input_byte_identical": True,
                "population_subsetting_after_hash": True,
            },
            "family_collapse": {
                "missing_state": True,
                "missing_pair": True,
                "chance_pairwise_only": True,
                "positive_pairwise": False,
                "positive_top3": False,
                "positive_progress": False,
            },
            "classification": [
                {"inputs": [False, False, False, False], "expected": "RAW_LATENT_GOAL_COST_NO_GO"},
                {"inputs": [True, True, True, False], "expected": "KINEMATIC_BASELINE_DOMINANT"},
                {"inputs": [True, True, False, True], "expected": "TWO_STEP_JEPA_PLANNING_COST_SIGNAL"},
                {"inputs": [True, False, False, False], "expected": "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO"},
            ],
            "gate_classifications": {
                "true_future_pass": "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL",
                "true_future_fail": "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
                "two_step_pass": "TWO_STEP_JEPA_PLANNING_COST_SIGNAL",
                "two_step_fail": None,
                "one_step_base_only_is_diagnostic_not_classification": True,
            },
            "paired_materiality": {
                "below_magnitude": False,
                "zero_ci_lower": False,
                "progress_trigger": True,
                "regret_trigger": True,
                "hard_family_trigger": True,
                "safety_increase": False,
                "family_collapse": False,
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED,
                "unit": "state",
                "byte_identical_regeneration": True,
                "hash_resample_plan": "PASS_REQUIRED",
                "type7_percentiles": "PASS_REQUIRED",
                "global_numpy_rng_inert": True,
            },
            "gpu_custody": {
                "strict_load": "PASS_REQUIRED",
                "parameter_digest_before_equals_after": "PASS_REQUIRED",
                "requires_grad": False,
                "optimizer": None,
                "training_steps": 0,
                "future_input_fields": [],
            },
            "dense_route_replay_custody": {
                "state_files": 48,
                "branches_per_state": 12,
                "ticks_per_branch": 15,
                "boundaries": [5, 10, 15],
                "every_file_hash_and_self_digest": "PASS_REQUIRED",
                "prefreeze_byte_inventory_binding": copy.deepcopy(
                    DENSE_ROUTE_REPLAY_INPUT_BINDINGS
                ),
                "byte_inventory_validated_before_json_parse": "PASS_REQUIRED",
                "missing_duplicate_or_drifting_state": "FAIL_CLOSED",
            },
            "paired_effect_ledger": {
                "comparison_ids": list(PAIRED_COMPARISON_IDS),
                "heldout_states": 8,
                "rows": 32,
                "effect_conventions_exact": "PASS_REQUIRED",
                "missing_duplicate_or_wrong_population_row": "FAIL_CLOSED",
            },
            "atomic_publication": {
                "canonical_root_present_before_terminal_pass": False,
                "same_filesystem_atomic_rename": "PASS_REQUIRED",
                "failed_attempt_reuse": False,
                "bound_failed_attempt_archive": str(FAILED_GOAL_VIEW_ATTEMPT_ARCHIVE),
                "fresh_hidden_attempt_after_goal_view_amendment": "PASS_REQUIRED",
            },
            "cpu_worker_allocator": {
                "environment": copy.deepcopy(
                    EXECUTION_WATCHDOGS["cpu_worker_environment"]
                ),
                "worker_count_unchanged": 32,
                "fallback": "FORBIDDEN",
            },
            "tensor_index": {
                "logical_records": 5424,
                "unique_payloads": "derive from unique path/SHA pairs",
                "current_aliases_context_slot_2": True,
                "tampered_sha_shape_dtype_or_bytes": "FAIL_CLOSED",
                "pure_tensor_to_cost_row_reproduction": "PASS_REQUIRED",
            },
            "row_reproduction": {
                "aggregate_from_scalar_rows_without_inference": "PASS_REQUIRED",
                "tampered_row": "FAIL_CLOSED",
                "missing_row": "FAIL_CLOSED",
            },
        },
        "executed_checks": _run_contract_fixture_checks(),
    }
    core["pass"] = all(core["executed_checks"].values())
    return attach_content_digest(core)


def fixture_receipt_bytes() -> bytes:
    return canonical_json_bytes(build_fixture_receipt()) + b"\n"


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if canonical_json_bytes(value) != canonical_json_bytes(build_contract()):
        raise ContractError("contract receipt differs from the prospective contract")
    return copy.deepcopy(dict(value))


def validate_output_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    if canonical_json_bytes(value) != canonical_json_bytes(build_output_schema()):
        raise ContractError("output schema differs from the prospective schema")
    return copy.deepcopy(dict(value))


def validate_fixture_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_digest(value)
    if canonical_json_bytes(value) != canonical_json_bytes(build_fixture_receipt()):
        raise ContractError("fixture receipt differs from the prospective fixtures")
    checks = value.get("executed_checks")
    if (
        value.get("pass") is not True
        or not isinstance(checks, Mapping)
        or not checks
        or not all(check is True for check in checks.values())
    ):
        raise ContractError("fixture receipt did not pass every executed check")
    return copy.deepcopy(dict(value))


def validate_goal_view_execution_amendment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_digest(value)
    if canonical_json_bytes(value) != canonical_json_bytes(
        build_goal_view_execution_amendment()
    ):
        raise ContractError("goal-view execution amendment differs from its authority")
    if (
        value.get("status") != "PROSPECTIVE_BEFORE_FRESH_REEXECUTION"
        or value.get("static_diagnosis", {}).get("route_outcome_rows_read_or_used") != 0
        or value.get("failed_attempt", {}).get("scientific_phase_or_shard_reuse")
        is not False
    ):
        raise ContractError("goal-view execution amendment violates prospective custody")
    return copy.deepcopy(dict(value))


def _write_immutable(path: Path, payload: bytes, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"refusing to overwrite non-identical {label}: {path}")
        return path
    path.write_bytes(payload)
    return path


def write_contract(path: str | Path = TRACKED_CONTRACT_RECEIPT_PATH) -> Path:
    return _write_immutable(Path(path), contract_receipt_bytes(), "contract receipt")


def write_output_schema(path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH) -> Path:
    return _write_immutable(Path(path), output_schema_receipt_bytes(), "output schema")


def write_fixture_receipt(path: str | Path = TRACKED_FIXTURE_PATH) -> Path:
    return _write_immutable(Path(path), fixture_receipt_bytes(), "fixture receipt")


def write_goal_view_execution_amendment(
    path: str | Path = TRACKED_GOAL_VIEW_AMENDMENT_PATH,
) -> Path:
    return _write_immutable(
        Path(path),
        goal_view_execution_amendment_receipt_bytes(),
        "goal-view execution amendment",
    )


def _load_exact(path: Path, expected: bytes, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read {label}: {path}") from exc
    if raw != expected:
        raise ContractError(f"{label} bytes do not match the frozen canonical receipt")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def load_and_validate_contract(
    path: str | Path = TRACKED_CONTRACT_RECEIPT_PATH,
) -> dict[str, Any]:
    return validate_contract(_load_exact(Path(path), contract_receipt_bytes(), "contract"))


def load_and_validate_output_schema(
    path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH,
) -> dict[str, Any]:
    return validate_output_schema(
        _load_exact(Path(path), output_schema_receipt_bytes(), "output schema")
    )


def load_and_validate_fixture_receipt(
    path: str | Path = TRACKED_FIXTURE_PATH,
) -> dict[str, Any]:
    return validate_fixture_receipt(
        _load_exact(Path(path), fixture_receipt_bytes(), "fixture receipt")
    )


def load_and_validate_goal_view_execution_amendment(
    path: str | Path = TRACKED_GOAL_VIEW_AMENDMENT_PATH,
) -> dict[str, Any]:
    return validate_goal_view_execution_amendment(
        _load_exact(
            Path(path),
            goal_view_execution_amendment_receipt_bytes(),
            "goal-view execution amendment",
        )
    )


SOURCE_CLOSURE_DEFAULT_PATHS = (
    "lewm/__init__.py",
    "lewm/safety/__init__.py",
    "lewm/oracle/__init__.py",
    "lewm/safety/jepa_local_waypoint_planning_cost_qualification_v1_contract.py",
    "lewm/tests/test_jepa_local_waypoint_planning_cost_qualification_v1_contract.py",
    str(TRACKED_PREREGISTRATION_PATH),
    str(TRACKED_CONTRACT_RECEIPT_PATH),
    str(TRACKED_OUTPUT_SCHEMA_PATH),
    str(TRACKED_FIXTURE_PATH),
    str(TRACKED_GOAL_VIEW_AMENDMENT_PATH),
    str(ENTRYPOINT_PATH),
    "scripts/run_jepa_local_waypoint_planning_cost_inference_v1.py",
    "lewm/safety/jepa_local_waypoint_planning_cost_metrics_v1.py",
    "lewm/tests/test_evaluate_jepa_local_waypoint_planning_cost_qualification_v1.py",
    "lewm/tests/test_run_jepa_local_waypoint_planning_cost_inference_v1.py",
    "lewm/tests/test_jepa_local_waypoint_planning_cost_metrics_v1.py",
    "lewm/oracle/go2_textured_v03_renderer.py",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "lewm/oracle/go2_rgb_control_history_four_step_autoregressive_v1_contract.py",
    "scripts/dev_proprio_predictor_v1.py",
    "scripts/build_dev_v03_proprio_action_manifest_v1.py",
    "scripts/dev_action_slew_reconstruction_v1.py",
    "scripts/dev_checkpoint_v1.py",
    "scripts/run_dev_v03_two_step_rollout_v1.py",
    "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    "scripts/collect_safe_local_waypoint_purpose_built_v1.py",
    "scripts/materialize_dense_route_intent_true_future_v1.py",
    "scripts/encode_safe_local_waypoint_route_intent_v2.py",
    "scripts/replay_safe_local_waypoint_route_intent_v2.py",
    "scripts/run_kinematic_route_with_runtime_safety_guard_v1.py",
    "scripts/run_go2_oracle_branch_pilot_v1.py",
    "scripts/run_go2_oracle_branch_pilot_v1_2.py",
    "scripts/instrument_contact_hazard_ontology_v1.py",
    "scripts/benchmark_one_tick_observation_prediction_control_loop_v1.py",
    "scripts/build_dev_v03_temporal_sequences_v1.py",
    "scripts/render_replay_v03.py",
    "lewm/oracle/go2_branch_oracle_v1_2.py",
    "lewm/safety/contact_hazard_ontology_v1.py",
    "lewm/safety/control_commitment_horizon_and_viability_v1.py",
    "lewm/safety/one_tick_viability_constrained_mpc_v1.py",
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
    "config/go2_platform_manifest.yaml",
    "config/go2_primitive_registry.yaml",
    "docs/lewm_go2_v03_horizon_rollout_result_2026-08-09.md",
    "docs/lewm_counterfactual_predictor_claims_matrix_2026-08-18.md",
    "docs/lewm_protected_contact_scope_requirements_review_v1.md",
    "docs/lewm_protected_contact_scope_traceability_matrix_v1.md",
    "docs/lewm_protected_contact_scope_decision_memo_v1.md",
    "docs/lewm_protected_contact_scope_assurance_fragment_v1.md",
    "docs/lewm_protected_contact_scope_requirements_review_v1_result.json",
)


def build_source_closure(
    repo_root: str | Path,
    *,
    additional_paths: Iterable[str | Path] = (),
    require_complete: bool = True,
) -> dict[str, Any]:
    """Hash the explicit source closure; never traverse outcome or cache roots."""

    root = Path(repo_root).resolve()
    relative_paths = [Path(value) for value in SOURCE_CLOSURE_DEFAULT_PATHS]
    relative_paths.extend(Path(value) for value in additional_paths)
    if len({str(path) for path in relative_paths}) != len(relative_paths):
        raise ContractError("source closure contains a duplicate path")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in relative_paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"source-closure path must be repository-relative: {relative}")
        absolute = root / relative
        if not absolute.is_file():
            missing.append(str(relative))
            continue
        sha256, size = _sha256_file(absolute)
        rows.append({"path": str(relative), "sha256": sha256, "bytes": size})
    if require_complete and missing:
        raise ContractError(f"source closure is incomplete: {missing!r}")
    core = {
        "schema": SOURCE_CLOSURE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "starting_head": STARTING_HEAD,
        "rows": rows,
        "row_count": len(rows),
        "missing_paths": missing,
        "complete": not missing,
        "outcome_or_result_payloads_parsed": [],
        "custody_only_result_files_hashed_without_parsing": [
            "docs/lewm_protected_contact_scope_requirements_review_v1_result.json"
        ],
        "generated_cache_paths_traversed": [],
    }
    return attach_content_digest(core)


def source_closure_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    validate_content_digest(value)
    return canonical_json_bytes(value) + b"\n"


def write_source_closure(
    value: Mapping[str, Any],
    path: str | Path = TRACKED_SOURCE_CLOSURE_PATH,
) -> Path:
    if value.get("complete") is not True or value.get("missing_paths") != []:
        raise ContractError("refusing to freeze an incomplete source closure")
    return _write_immutable(
        Path(path), source_closure_receipt_bytes(value), "source-closure receipt"
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "COMPARATOR_IDS",
    "CONTRACT",
    "CONTRACT_RECEIPT_SHA256",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_SHA256",
    "ContractError",
    "CONTROLLER_EXECUTION_CUSTODY",
    "CPU_WATCHDOG_STATUS_SUCCESS",
    "CPU_FOUNDATIONAL_PACKAGE_BINDINGS",
    "CPU_RUNTIME_INPUT_BINDINGS",
    "DENSE_ROUTE_REPLAY_INPUT_BINDINGS",
    "ENTRYPOINT_PATH",
    "EXECUTION_WATCHDOGS",
    "EXECUTION_WATCHDOG_STATUS_SUCCESS",
    "EXPERIMENT_ID",
    "FAMILY_IDS",
    "FOUNDATIONAL_PACKAGE_CLOSURE_POLICY",
    "GPU_FOUNDATIONAL_PACKAGE_BINDINGS",
    "GPU_WATCHDOG_STATUS_SUCCESS",
    "GOAL_CELL_BLOCK_CLASSIFICATIONS",
    "GOAL_CELL_BLOCKED_STATE_IDS",
    "GOAL_CELL_CLASSIFICATION_COUNTS",
    "GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS",
    "GOAL_VIEW_AMENDMENT_SCHEMA_VERSION",
    "GOAL_VIEW_EXECUTION_AMENDMENT",
    "GOAL_VIEW_EXECUTION_AMENDMENT_BINDING",
    "GOAL_VIEW_RENDER_SEMANTICS",
    "GOAL_VIEW_STATIC_VALIDATION_SUCCESS",
    "FAILED_GOAL_VIEW_ATTEMPT_ARCHIVE",
    "FAILED_GOAL_VIEW_ATTEMPT_INVENTORY",
    "HARD_FAMILY_IDS",
    "HISTORICAL_RENDERER_LIMITATIONS",
    "INTERPRETER_BINARY_BINDING",
    "NEXT_EXPERIMENT_IDS",
    "ONE_STEP_CHECKPOINT_SHA256",
    "OUTPUT_ROOT",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_RECEIPT_SHA256",
    "OUTPUT_SCHEMA_SHA256",
    "ORIGINAL_FREEZE_COMMIT",
    "PAIRED_COMPARISON_IDS",
    "POPULATION_IDS",
    "PRIMARY_CLASSIFICATIONS",
    "PROHIBITION_COUNTER_IDS",
    "RECONSTRUCTION_PREFIX_CUSTODY",
    "REQUIREMENTS_CLASSIFICATIONS",
    "REQUIREMENTS_STATEMENT",
    "SCENE_INPUT_BYTE_INVENTORY_BINDING",
    "SECONDARY_CLASSIFICATIONS",
    "SEED",
    "SOURCE_CLOSURE_DEFAULT_PATHS",
    "SOURCE_IDS",
    "STATIC_FILE_BINDINGS",
    "TRACKED_CONTRACT_RECEIPT_PATH",
    "TRACKED_FIXTURE_PATH",
    "TRACKED_GOAL_VIEW_AMENDMENT_PATH",
    "TRACKED_OUTPUT_SCHEMA_PATH",
    "TRACKED_PREREGISTRATION_PATH",
    "TRACKED_REPORT_PATH",
    "TRACKED_RESULT_PATH",
    "TRACKED_SOURCE_CLOSURE_PATH",
    "TRUE_FUTURE_GATE_CLASSIFICATIONS",
    "TWO_STEP_CHECKPOINT_SHA256",
    "attach_content_digest",
    "build_contract",
    "build_fixture_receipt",
    "build_goal_view_execution_amendment",
    "build_output_schema",
    "build_source_closure",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "contract_receipt",
    "contract_receipt_bytes",
    "derive_incremental_route_value_secondary",
    "derive_primary_classification",
    "family_complete_collapse",
    "fixture_receipt_bytes",
    "goal_view_execution_amendment_receipt_bytes",
    "load_and_validate_contract",
    "load_and_validate_fixture_receipt",
    "load_and_validate_goal_view_execution_amendment",
    "load_and_validate_output_schema",
    "no_family_complete_collapse",
    "output_schema_receipt_bytes",
    "route_heading_yaw",
    "source_closure_receipt_bytes",
    "tokenwise_cosine_cost",
    "validate_content_digest",
    "validate_contract",
    "validate_fixture_receipt",
    "validate_goal_view_execution_amendment",
    "validate_output_schema",
    "write_contract",
    "write_fixture_receipt",
    "write_goal_view_execution_amendment",
    "write_output_schema",
    "write_source_closure",
]

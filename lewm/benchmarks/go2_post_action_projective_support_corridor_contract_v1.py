"""Source-only contract for the post-action projective-support JEPA probe.

Importing this module opens no generated artifact and imports no tensor, image,
simulator, or accelerator library.  It centralizes only the identities and
limits frozen by preregistration commit ``8a52adb``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping


ROOT: Final = Path(__file__).resolve().parents[2]
SCHEMA_PREFIX: Final = (
    "lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1"
)
EXPERIMENT_ID: Final = "rgb_post_action_projective_support_corridor_joint_jepa_v1"

PREREGISTRATION_RELATIVE_PATH: Final = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "preregistration_2026-07-28.md"
)
PREREGISTRATION_COMMIT: Final = "8a52adb77d30cb98a6dd086037e6f7c296d76d63"
PREREGISTRATION_FILE_SHA256: Final = (
    "fe39daa2ff2f19624d67910d60a6da640f6351a6b5a6135db7b877bbc784e045"
)
PREREGISTRATION_BYTE_COUNT: Final = 29_487

ACTION_VOCABULARY: Final = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
HOLD_ACTION_INDEX: Final = 6
NON_HOLD_ACTION_INDICES: Final = tuple(
    index for index in range(len(ACTION_VOCABULARY)) if index != HOLD_ACTION_INDEX
)
ROLE_ORDER: Final = ("train", "probability_calibration", "checkpoint_selection")
ROLE_COUNTS: Final = {
    "train": {"states": 4_262, "action_rows": 38_358, "scenes": 72},
    "probability_calibration": {"states": 415, "action_rows": 3_735, "scenes": 8},
    "checkpoint_selection": {"states": 495, "action_rows": 4_455, "scenes": 8},
}
TOTAL_STATES: Final = 5_172
TOTAL_ACTION_ROWS: Final = 46_548
STATION_COUNT: Final = 11
TOTAL_STATION_LABELS: Final = 512_028
SCENE_FAMILIES: Final = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)

INITIALIZATION_SEED: Final = 20260712
SCHEDULE_SEED: Final = 20260713
SCHEDULE_PREFIX_SHA256: Final = (
    "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528"
)
EXPERIMENT_SEED: Final = 20260728
BOOTSTRAP_SEED: Final = 20260728
MAXIMUM_ATTEMPTS: Final = 1
MAXIMUM_UPDATES: Final = 1_000
MAXIMUM_PRESENTATIONS: Final = 16_000
MICROBATCH_SIZE: Final = 4
MICROBATCHES_PER_UPDATE: Final = 4
EFFECTIVE_BATCH_SIZE: Final = 16
TARGET_EMA_MOMENTUM: Final = 0.996

RUNTIME_INTERPRETER_PATH: Final = (
    "/home/andrewknowles/.local/share/"
    "lewmquad-v12-runtime-torch291-rocm64/bin/python"
)
RUNTIME_SYS_PREFIX: Final = (
    "/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64"
)

RAW_ROOT_RELATIVE_PATH: Final = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
RAW_MANIFEST_RELATIVE_PATH: Final = f"{RAW_ROOT_RELATIVE_PATH}/manifest.json"
RAW_AUDIT_RELATIVE_PATH: Final = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1.audit_v13.json"
)
RAW_PAIRS_RELATIVE_PATH: Final = f"{RAW_ROOT_RELATIVE_PATH}/pairs.jsonl"
RAW_ENDPOINTS_RELATIVE_PATH: Final = f"{RAW_ROOT_RELATIVE_PATH}/endpoints.jsonl"
N320_GATE_RELATIVE_PATH: Final = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n320_compute_scaled_v1/gate.json"
)
N320_CHECKPOINT_RELATIVE_PATH: Final = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n320_compute_scaled_v1/checkpoint.pt"
)
SCHEDULE_RELATIVE_PATH: Final = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "matched_training_v4/schedule.json"
)

RUNTIME_BINDINGS: Final = {
    RAW_MANIFEST_RELATIVE_PATH: {
        "path": RAW_MANIFEST_RELATIVE_PATH,
        "file_sha256": "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360",
        "content_sha256": "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a",
        "byte_count": 311_598,
    },
    RAW_AUDIT_RELATIVE_PATH: {
        "path": RAW_AUDIT_RELATIVE_PATH,
        "file_sha256": "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76",
        "content_sha256": "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca",
        "byte_count": 26_975,
    },
    RAW_PAIRS_RELATIVE_PATH: {
        "path": RAW_PAIRS_RELATIVE_PATH,
        "file_sha256": "5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d",
    },
    RAW_ENDPOINTS_RELATIVE_PATH: {
        "path": RAW_ENDPOINTS_RELATIVE_PATH,
        "file_sha256": "34e47ddcc40ad8c1f092c73193d16773cf4dedae05e7f4f684abb385cc2c0d01",
    },
    N320_GATE_RELATIVE_PATH: {
        "path": N320_GATE_RELATIVE_PATH,
        "file_sha256": "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6",
        "content_sha256": "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b",
        "byte_count": 7_960,
    },
    N320_CHECKPOINT_RELATIVE_PATH: {
        "path": N320_CHECKPOINT_RELATIVE_PATH,
        "file_sha256": "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0",
        "content_sha256": "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b",
        "byte_count": 13_777_100,
    },
    SCHEDULE_RELATIVE_PATH: {
        "path": SCHEDULE_RELATIVE_PATH,
        "file_sha256": "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270",
        "content_sha256": "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15",
        "byte_count": 607_373,
    },
}

GEOMETRY_BINDINGS: Final = {
    "geometry_contract": {
        "path": "config/go2_generalization_geometry_v2.json",
        "file_sha256": "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52",
        "content_sha256": "e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca",
    },
    "directional_policy": {
        "path": (
            "config/go2_geometry_v2_artifacts/"
            "go2_directional_footprint_policy_v1_"
            "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc.json"
        ),
        "file_sha256": "750d8afe47ee3edd5988cdea443f19703efad7a3266218932671b9fdfbe43828",
        "content_sha256": "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc",
    },
    "directional_geometry": {
        "path": "lewm/planning/oriented_footprint.py",
        "file_sha256": "5831379e52eb0eaa1c2cf8d195b6d46b29ad8b66dbadc98f51629f22bc656b37",
    },
    "primitive_registry": {
        "path": "config/go2_primitive_registry.yaml",
        "file_sha256": "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8",
    },
}

LABEL_ROOT_RELATIVE_PATH: Final = (
    ".generated/go2_post_action_projective_support_labels_v1"
)
LABEL_MANIFEST_RELATIVE_PATH: Final = f"{LABEL_ROOT_RELATIVE_PATH}/manifest.json"
LABEL_ROLE_RELATIVE_PATHS: Final = {
    role: f"{LABEL_ROOT_RELATIVE_PATH}/{role}.jsonl" for role in ROLE_ORDER
}
OUTPUT_ROOT_RELATIVE_PATH: Final = (
    ".generated/go2_rgb_post_action_projective_support_corridor_joint_jepa_v1/"
    "attempt_v1"
)

REMOTE_POSE_SHA256: Final = (
    "df96a4d23e9f2a297467c7384e54e9d7f8eac64609e937392f0db51e3c87abc3"
)
FULL_MASK_SHA256: Final = (
    "63648c9c157d032db943b4dea5168879c287c847101606c56c97688f06e69da4"
)
PERSISTENCE_MASK_STACK_SHA256: Final = (
    "983577015f2822bbf60d89cd633baa9958afd624410e1a3e4390422647e59e34"
)

SEMANTIC_THRESHOLDS: Final = {
    "balanced_accuracy_minimum": 0.80,
    "free_recall_minimum": 0.85,
    "occupied_recall_minimum": 0.70,
    "unknown_recall_minimum": 0.90,
    "rough_occupied_recall_minimum": 0.65,
}
CORRIDOR_THRESHOLDS: Final = {
    "safe_precision_minimum": 0.99,
    "unsafe_recall_minimum": 0.95,
    "safe_recall_minimum": 0.90,
    "normalized_utility_minimum": 0.90,
    "nonempty_prefix_fraction_minimum": 0.90,
}

LABEL_ROW_SCHEMA: Final = "lewm_go2_post_action_projective_support_label_row_v1"
LABEL_MANIFEST_SCHEMA: Final = "lewm_go2_post_action_projective_support_labels_v1"
SOURCE_MANIFEST_SCHEMA: Final = f"{SCHEMA_PREFIX}_source_manifest_v1"
SOURCE_REVIEW_SCHEMA: Final = f"{SCHEMA_PREFIX}_source_review_v1"
EXECUTION_BINDING_SCHEMA: Final = f"{SCHEMA_PREFIX}_execution_binding_v1"
RESULT_SCHEMA: Final = f"{SCHEMA_PREFIX}_result_v1"
FAILURE_SCHEMA: Final = f"{SCHEMA_PREFIX}_failure_v1"
COMPLETION_SCHEMA: Final = f"{SCHEMA_PREFIX}_completion_v1"

DOWNSTREAM_DENIALS: Final = {
    "checkpoint_qualified": False,
    "checkpoint_read_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_resume_second_seed_or_second_attempt_authorized": False,
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    if "content_sha256" in core:
        raise ValueError("content_sha256 is derived")
    result = dict(core)
    result["content_sha256"] = canonical_json_sha256(core)
    return result


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{name} is not canonical newline-terminated JSON")
    content = value.get("content_sha256")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if content != canonical_json_sha256(core):
        raise ValueError(f"{name} content hash changed")
    return value


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def science_contract() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "actions": list(ACTION_VOCABULARY),
        "roles": {role: dict(ROLE_COUNTS[role]) for role in ROLE_ORDER},
        "updates": MAXIMUM_UPDATES,
        "presentations": MAXIMUM_PRESENTATIONS,
        "microbatch_size": MICROBATCH_SIZE,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "seeds": {
            "initialization": INITIALIZATION_SEED,
            "schedule": SCHEDULE_SEED,
            "experiment": EXPERIMENT_SEED,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "schedule_prefix_sha256": SCHEDULE_PREFIX_SHA256,
        "objective": "S+P+Q+R jointly from update 1",
        "remote_pose_sha256": REMOTE_POSE_SHA256,
        "full_mask_sha256": FULL_MASK_SHA256,
        "persistence_mask_stack_sha256": PERSISTENCE_MASK_STACK_SHA256,
        "semantic_thresholds": dict(SEMANTIC_THRESHOLDS),
        "corridor_thresholds": dict(CORRIDOR_THRESHOLDS),
        "one_attempt_no_retry_or_resume": True,
    }


def validate_static_contract() -> None:
    if (
        len(ACTION_VOCABULARY) != 9
        or ACTION_VOCABULARY[HOLD_ACTION_INDEX] != "hold"
        or sum(item["states"] for item in ROLE_COUNTS.values()) != TOTAL_STATES
        or sum(item["action_rows"] for item in ROLE_COUNTS.values())
        != TOTAL_ACTION_ROWS
        or TOTAL_ACTION_ROWS * STATION_COUNT != TOTAL_STATION_LABELS
        or MICROBATCH_SIZE * MICROBATCHES_PER_UPDATE != EFFECTIVE_BATCH_SIZE
        or MAXIMUM_UPDATES * EFFECTIVE_BATCH_SIZE != MAXIMUM_PRESENTATIONS
        or not math.isclose(TARGET_EMA_MOMENTUM, 0.996)
    ):
        raise RuntimeError("frozen projective-support contract is inconsistent")


validate_static_contract()


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_json_bytes",
    "canonical_json_sha256",
    "parse_canonical_json",
    "preregistration_binding",
    "science_contract",
    "validate_static_contract",
    "with_content_sha256",
]

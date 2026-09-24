"""Pure frozen contract for Shared JEPA V5 full training V2.

This module is standard-library only. It contains no execution capability,
payload reader, model import, backend hook, or mutable authority registry.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/coordinator_v2_qa"

V1_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_2026-07-13.md"
)
V1_AMENDMENT_SHA256 = (
    "b21d01d062543cc7b7f3f5281f66ac40df76726c678a9364f7a4e451b035a4a7"
)
V1_AUTHOR_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_author_"
    "handoff_2026-07-13.md"
)
V1_AUTHOR_HANDOFF_SHA256 = (
    "fa0a497fad2f17a5d0919e1160b6040cbe13740315cfc180418d99dbf494d6bc"
)
V1_INDEPENDENT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_shared_jepa_v5_full_training_execution_amendment_v1_"
    "independent_review.py"
)
V1_INDEPENDENT_TEST_SHA256 = (
    "b2959ea11cff80091a9f94c61dde14750726332001326c0fa30bd186418c6b38"
)
V1_INDEPENDENT_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v1_"
    "independent_review_2026-07-13.md"
)
V1_INDEPENDENT_REVIEW_SHA256 = (
    "2cd1bf56edd213041496c67238dcf540f2f4a1b72e9abae529e327b4e22c125c"
)
V1_BLOCK_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v1_"
    "independent_review_block_2026-07-13.json"
)
V1_BLOCK_SHA256 = (
    "c3debd1ee4394e8916b8bfeb7d9237c44f3152e0fd36c27cdf84819c3e356273"
)
V2_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v2_2026-07-13.md"
)
V2_AMENDMENT_SHA256 = (
    "b521d2885b5dca1a72838282fbb8e193a21ec0f2db0e0a5950074506fba1f66d"
)
V2_INDEPENDENT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_shared_jepa_v5_full_training_execution_amendment_v2_"
    "independent_review.py"
)
V2_INDEPENDENT_TEST_SHA256 = (
    "734a140f2b073e02970cb81897fd5edbb7beb28e56a60ba08f774df43f920e0b"
)
V2_INDEPENDENT_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v2_"
    "independent_review_2026-07-13.md"
)
V2_INDEPENDENT_REVIEW_SHA256 = (
    "f4b22ef6061a54b08b2e2afa5f0e56ecbfa20a5a364f5eda0395d71722182dae"
)
V2_PASS_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_execution_amendment_v2_"
    "independent_review_pass_2026-07-13.json"
)
V2_PASS_SHA256 = (
    "6a53a3c9d72da6499714883676f49a62d0c3ba61c2d2ccde741f1654e6f089d4"
)

FROZEN_GOVERNING_DESIGN_BINDINGS = {
    "docs/lewm_go2_v5_joint_training_execution_gap_audit_2026-07-13.md": (
        "b4bc71e6cc2728fdbc5c1a3822d4be130b9c2ccac3bb8cf2a9baece6bc497f6a"
    ),
    "docs/lewm_go2_shared_jepa_g2_g3_implementation_plan_2026-07-11.md": (
        "54ad8c08546c46c8989a84e497b54b83366526f8f5ed6faed6364880fa1a702a"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_preregistration_"
    "2026-07-13.md": (
        "07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_source_inventory_"
    "amendment_2026-07-13.md": (
        "39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_gate_2026-07-12.md": (
        "49887b8b39ba16e490f6171ac0efe239456e1d27081312a71800ca33c247f874"
    ),
}

MODEL_RELATIVE_PATH = "lewm/models/shared_observable_camera_ray_jepa_v5.py"
MODEL_SHA256 = "b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9"
MODEL_TEST_RELATIVE_PATH = "lewm/tests/test_shared_observable_camera_ray_jepa_v5.py"
MODEL_TEST_SHA256 = "848aa8be369b89c973a4da916f9c7abeff47eca12aceb4304cf612ed4d53227b"
OUTPUT_LOSS_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_output_loss_correction_candidate_2026-07-13.md"
)
OUTPUT_LOSS_REVIEW_SHA256 = (
    "83dcd8f8702656c25f4584295827d0c82cf1db113abe2de4a417e7b528abff1f"
)

REVIEWED_LIFECYCLE_BINDINGS = {
    "scripts/go2_shared_jepa_v5_one_shot.py": (
        "62a19f3028e9152120af990528752431b996f56b4bc9b62db32eba47ae235a1f"
    ),
    "scripts/go2_shared_jepa_v5_launcher.py": (
        "7f273649fa6c8b4256c552359927fc20bb59d1bfbd5b47194a3f5a941c5b8958"
    ),
    "scripts/run_go2_shared_jepa_v5_gate.py": (
        "37402f0f75a7a4f475539e269e77aeae072ce80b0af0bcb4147e2ec1b33ff57a"
    ),
    "scripts/finalize_go2_shared_jepa_v5_gate.py": (
        "f0426201f5344d0eb1d43e183e4755ac8fd7aecdc9af6e5b7c19076af3f5dc34"
    ),
    "scripts/publish_go2_shared_jepa_v5_checkpoint.py": (
        "4e045365dadb28bd37cdbb49808bef7528d4e5cb0c3e77ff5aae678559174fab"
    ),
}
LIFECYCLE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_staged_lifecycle_independent_review_"
    "2026-07-13.md"
)
LIFECYCLE_REVIEW_SHA256 = (
    "bcb587c5bd7ea08063cbbf1c8d5a4a99b29c24fdfc490469aae4bff6dbe98abc"
)

EXACT_EXECUTION_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v2_exact_execution_manifest_"
    "2026-07-13.json"
)
PREFLIGHT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "full_training_v2_preflight"
)
EXACT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v2"
)
CANONICAL_PREFLIGHT_ROOT = ROOT / PREFLIGHT_ROOT_RELATIVE_PATH
CANONICAL_EXACT_ROOT = ROOT / EXACT_ROOT_RELATIVE_PATH
PREFLIGHT_RECEIPT_RELATIVE_PATH = (
    f"{PREFLIGHT_ROOT_RELATIVE_PATH}/gpu_smoke_receipt.json"
)
PREFLIGHT_COMPLETED_RELATIVE_PATH = f"{PREFLIGHT_ROOT_RELATIVE_PATH}/completed.json"
PREFLIGHT_INDEPENDENT_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v2_preflight_independent_review_"
    "2026-07-13.json"
)

RAW_SUPERVISION_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
RAW_SUPERVISION_MANIFEST_RELATIVE_PATH = (
    f"{RAW_SUPERVISION_ROOT_RELATIVE_PATH}/manifest.json"
)
RAW_SUPERVISION_AUDIT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1.audit.json"
)
RAW_SUPERVISION_BUILDER_RELATIVE_PATH = (
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py"
)
RAW_SUPERVISION_AUDITOR_RELATIVE_PATH = (
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py"
)
RAW_SUPERVISION_MANIFEST_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_dataset_v1"
)
RAW_SUPERVISION_AUDIT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_v1"

V4_TWO_SEED_LADDER_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2/"
    "gates/two_seed.json"
)
V4_PRIMARY_CHECKPOINT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2/attempts/"
    "seed_20260710/n320/checkpoint.pt"
)

PAIRED_NAVIGATION_MANIFEST_RELATIVE_PATH = (
    ".generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/"
    "dataset_manifest.json"
)
PAIRED_NAVIGATION_MANIFEST_FILE_SHA256 = (
    "ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180"
)
PAIRED_NAVIGATION_ROW_INDEX_FILE_SHA256 = (
    "187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac"
)
PAIRED_NAVIGATION_ROLE_ASSIGNMENT_SHA256 = (
    "016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02"
)
PAIRED_NAVIGATION_G2_SCENE_SET_COMMITMENT = (
    "0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402"
)
PAIRED_NAVIGATION_SOURCE_INDEX_FILE_SHA256 = (
    "11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c"
)

POLICY_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_full_training_v2_policy.py"
)
PREFLIGHT_EXECUTOR_RELATIVE_PATH = (
    "scripts/preflight_go2_shared_jepa_v5_full_training_v2.py"
)
PREFLIGHT_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_shared_jepa_v5_full_training_v2_preflight.py"
)
EXACT_EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_shared_jepa_v5_full_training_v2.py"
)
EXACT_TRAINER_RELATIVE_PATH = (
    "scripts/train_go2_shared_jepa_v5_full_training_v2.py"
)
EXACT_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_shared_jepa_v5_full_training_v2.py"
)
IMPLEMENTATION_SOURCE_PATHS = (
    POLICY_RELATIVE_PATH,
    PREFLIGHT_EXECUTOR_RELATIVE_PATH,
    PREFLIGHT_VERIFIER_RELATIVE_PATH,
    EXACT_EXECUTOR_RELATIVE_PATH,
    EXACT_TRAINER_RELATIVE_PATH,
    EXACT_VERIFIER_RELATIVE_PATH,
)
IMPLEMENTATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v2_implementation_"
    "independent_review_2026-07-13.json"
)

EXECUTION_MANIFEST_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_manifest_v1"
IMPLEMENTATION_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v2_implementation_review_v1"
)
PREFLIGHT_RESERVATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v2_preflight_reservation_v1"
)
PREFLIGHT_RECEIPT_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v2_gpu_smoke_receipt_v1"
)
PREFLIGHT_COMPLETION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v2_preflight_completion_v1"
)
EXACT_RESERVATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v2_reservation_v1"
)
EXACT_COMPLETION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v2_completion_v1"
)
EXACT_FAILURE_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_failure_v1"
ACCESS_LEDGER_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_access_ledger_v1"
SCHEDULE_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_schedule_v1"
SOURCE_REVIEW_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_source_review_v1"
INPUT_BINDINGS_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_input_bindings_v1"
INITIALIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_initialization_v1"
TRAINING_RECORD_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_training_record_v1"
SELECTION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_selection_v1"
CALIBRATION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v2_calibration_v1"
DIAGNOSTIC_ABLATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v2_selection_role_ablation_diagnostic_v1"
)

FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
SCOPES = ("aggregate", *FAMILIES)
ROLE_COUNTS = {
    "train": {
        "scenes": 72,
        "pairs": 4262,
        "endpoint_instances": 8524,
        "unique_endpoints": 7777,
    },
    "checkpoint_selection": {
        "scenes": 8,
        "pairs": 495,
        "endpoint_instances": 990,
        "unique_endpoints": 924,
    },
    "probability_calibration": {
        "scenes": 8,
        "pairs": 415,
        "endpoint_instances": 830,
        "unique_endpoints": 759,
    },
}
DEVELOPMENT_ROLES = tuple(ROLE_COUNTS)
FORBIDDEN_ROLES = (
    "g2_evaluation",
    "g3",
    "heldout",
    "sealed",
    "runtime",
    "navigation",
    "hardware",
    "production",
    "promotion",
)
PROVENANCE_ROLES = (
    "source_closure",
    "execution_manifest",
    "implementation_review",
    "preflight_receipt",
    "raw_supervision_manifest",
    "raw_supervision_audit",
    "v4_two_seed_ladder",
    "v4_primary_checkpoint",
)
ARMS = ("promoted_jepa", "matched_no_jepa")
TRAIN_PAIR_COUNT = 4262
SCHEDULE_SEED = 20260713
INITIALIZATION_SEED = 20260712
PRIMARY_V4_SEED = 20260710
PRESENTATION_COUNT = 128000
UPDATE_COUNT = 8000
EFFECTIVE_BATCH_SIZE = 16
MICROBATCH_SIZE = 4
ACCUMULATION_STEPS = 4
CHECKPOINT_UPDATES = tuple(range(1000, 8001, 1000))
MODEL_IMAGE_SIZE = 112
MODEL_SOURCE_SHAPE = (128, 128)
MODEL_PIXEL_RAY_SHAPE = (84, 112)
MODEL_BEV_SHAPE = (64, 64)

OPTIMIZER_CONTRACT = {
    "name": "AdamW",
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 1e-4,
    "amsgrad": False,
    "updates": UPDATE_COUNT,
    "microbatch_size": MICROBATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "gradient_clip_norm": 1.0,
    "precision": "float32",
    "autocast": False,
    "ema_updates_per_optimizer_step": 1,
}
DEVICE_CONTRACT = {
    "device": "cuda:0",
    "device_name": "AMD Radeon AI PRO R9700",
    "minimum_total_memory_bytes": 32 * 1024**3,
    "hip_visible_devices": "0",
    "rocr_visible_devices": "0",
    "hsa_override_gfx_version_absent": True,
    "raphael_igpu_forbidden": True,
    "multi_gpu_forbidden": True,
}
JOINT_LOSS_CONTRACT = {
    "promoted_jepa": {
        "established_jepa_total_weight": 1.0,
        "current_v4_weight": 0.5,
        "next_v4_weight": 0.5,
        "v4_components": {
            "ordered_first_hit_nll": 0.25,
            "target_bin_offset_smooth_l1": 0.25,
            "ground_clear_distance_state_balanced_bce": 0.25,
            "derived_raster_hierarchical_bce": 0.25,
        },
    },
    "matched_no_jepa": {
        "established_jepa_total_weight": 0.0,
        "current_v4_weight": 0.5,
        "next_v4_weight": 0.5,
        "same_forward_and_diagnostics": True,
    },
}

REQUIRED_BINDING_NAMES = (
    "development_raw_supervision_manifest_file_sha256",
    "development_raw_supervision_manifest_content_sha256",
    "development_raw_supervision_builder_source_sha256",
    "development_raw_supervision_auditor_source_sha256",
    "development_raw_supervision_audit_file_sha256",
    "development_raw_supervision_audit_content_sha256",
    "v4_two_seed_ladder_pass_file_sha256",
    "v4_two_seed_ladder_pass_content_sha256",
    "v4_primary_seed_20260710_n320_checkpoint_file_sha256",
    "preflight_completed_file_sha256",
    "preflight_receipt_file_sha256",
    "preflight_independent_review_file_sha256",
    "implementation_policy_source_sha256",
    "preflight_executor_source_sha256",
    "preflight_verifier_source_sha256",
    "exact_executor_source_sha256",
    "exact_trainer_source_sha256",
    "exact_verifier_source_sha256",
    "implementation_independent_review_file_sha256",
)

PREFLIGHT_INVENTORY = (
    "reservation.json",
    "source_closure.json",
    "access_ledger.json",
    "gpu_smoke_receipt.json",
    "completed.json",
)
EXACT_INVENTORY = (
    "reservation.json",
    "source_review.json",
    "input_bindings.json",
    "preflight_receipt_binding.json",
    "schedule.json",
    "initialization.json",
    "arms/promoted_jepa/training_trace.jsonl",
    *(f"arms/promoted_jepa/checkpoints/update_{u}.pt" for u in CHECKPOINT_UPDATES),
    "arms/promoted_jepa/checkpoint_metrics.json",
    "arms/matched_no_jepa/training_trace.jsonl",
    *(f"arms/matched_no_jepa/checkpoints/update_{u}.pt" for u in CHECKPOINT_UPDATES),
    "arms/matched_no_jepa/matched_update_metrics.json",
    "selection.json",
    "calibration/promoted_jepa.json",
    "calibration/matched_no_jepa.json",
    "selection_role_ablation_diagnostic.json",
    "qualified_checkpoint.pt",
    "access_ledger.json",
    "training_record.json",
    "completed.json",
)

PHYSICAL_LOWER_THRESHOLDS = {
    "pixel_first_hit_balanced_accuracy": 0.95,
    "ground_clear_balanced_accuracy": 0.95,
    "derived_raster_balanced_accuracy": 0.95,
    "wrong_rgb_pixel_balanced_accuracy_drop": 0.12,
    "wrong_rgb_depth_median_error_increase_m": 0.12,
    "wrong_rgb_depth_p95_error_increase_m": 0.20,
    "wrong_rgb_ground_balanced_accuracy_drop": 0.12,
    "wrong_rgb_raster_nll_increase": 0.12,
    "wrong_rgb_raster_balanced_accuracy_drop": 0.12,
}
PHYSICAL_UPPER_THRESHOLDS = {
    "depth_median_error_m": 0.10,
    "depth_p95_error_m": 0.25,
    "derived_raster_nll": 0.15,
}
JEPA_LOWER_THRESHOLDS = {
    "target_cross_sample_std_mean": 0.05,
    "target_cross_sample_effective_rank": 4.0,
    "wrong_action_advantage_over_target_change": 0.10,
}
CALIBRATION_FREE_MIN_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
CALIBRATION_OCCUPIED_MAX_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35)
CALIBRATION_UNKNOWN_MAX_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35)
CALIBRATION_OCCUPIED_DETECTION_MIN_GRID = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def content_value(core: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(core)
    return {**copied, "content_sha256": canonical_json_sha256(copied)}


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite constant {value}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key}")
            result[key] = value
        return result

    value = json.loads(
        raw.decode("ascii"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical JSON plus newline")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content hash changed")
    return value


def reviewed_source_bindings() -> dict[str, str]:
    return {
        V1_AMENDMENT_RELATIVE_PATH: V1_AMENDMENT_SHA256,
        V1_AUTHOR_HANDOFF_RELATIVE_PATH: V1_AUTHOR_HANDOFF_SHA256,
        V1_INDEPENDENT_TEST_RELATIVE_PATH: V1_INDEPENDENT_TEST_SHA256,
        V1_INDEPENDENT_REVIEW_RELATIVE_PATH: V1_INDEPENDENT_REVIEW_SHA256,
        V1_BLOCK_RELATIVE_PATH: V1_BLOCK_SHA256,
        V2_AMENDMENT_RELATIVE_PATH: V2_AMENDMENT_SHA256,
        V2_INDEPENDENT_TEST_RELATIVE_PATH: V2_INDEPENDENT_TEST_SHA256,
        V2_INDEPENDENT_REVIEW_RELATIVE_PATH: V2_INDEPENDENT_REVIEW_SHA256,
        V2_PASS_RELATIVE_PATH: V2_PASS_SHA256,
        MODEL_RELATIVE_PATH: MODEL_SHA256,
        MODEL_TEST_RELATIVE_PATH: MODEL_TEST_SHA256,
        OUTPUT_LOSS_REVIEW_RELATIVE_PATH: OUTPUT_LOSS_REVIEW_SHA256,
        LIFECYCLE_REVIEW_RELATIVE_PATH: LIFECYCLE_REVIEW_SHA256,
        **FROZEN_GOVERNING_DESIGN_BINDINGS,
        **REVIEWED_LIFECYCLE_BINDINGS,
    }


def expected_implementation_review_core(
    *,
    reviewer: str,
    source_bindings: Mapping[str, str],
) -> dict[str, Any]:
    if (
        not isinstance(reviewer, str)
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
    ):
        raise PermissionError("implementation review must be by a different agent")
    if set(source_bindings) != set(IMPLEMENTATION_SOURCE_PATHS) or any(
        not is_sha256(value) for value in source_bindings.values()
    ):
        raise ValueError("implementation review source closure changed")
    return {
        "schema": IMPLEMENTATION_REVIEW_SCHEMA,
        "status": "different_agent_implementation_review_passed",
        "implementation_author": IMPLEMENTATION_AUTHOR,
        "reviewer": reviewer,
        "reviewed_sources": dict(source_bindings),
        "frozen_design_bindings": {
            V1_AMENDMENT_RELATIVE_PATH: V1_AMENDMENT_SHA256,
            V2_AMENDMENT_RELATIVE_PATH: V2_AMENDMENT_SHA256,
            V2_INDEPENDENT_REVIEW_RELATIVE_PATH: V2_INDEPENDENT_REVIEW_SHA256,
            V2_PASS_RELATIVE_PATH: V2_PASS_SHA256,
        },
        "frozen_parent_closure": reviewed_source_bindings(),
        "reviewed_model_binding": {MODEL_RELATIVE_PATH: MODEL_SHA256},
        "payload_free_preflight_authorized": True,
        "exact_execution_authorized": False,
        "dataset_or_checkpoint_access_authorized": False,
        "g2_or_heldout_authorized": False,
        "production_or_promotion_authorized": False,
    }


def validate_implementation_review(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("implementation review must be a mapping")
    copied = dict(value)
    declared = copied.pop("content_sha256", None)
    sources = copied.get("reviewed_sources")
    reviewer = copied.get("reviewer")
    if not isinstance(sources, Mapping) or not isinstance(reviewer, str):
        raise ValueError("implementation review bindings are missing")
    expected = expected_implementation_review_core(
        reviewer=reviewer,
        source_bindings=sources,
    )
    if (
        copied != expected
        or not is_sha256(declared)
        or canonical_json_sha256(copied) != declared
    ):
        raise PermissionError("implementation review contract changed")
    return {**copied, "content_sha256": declared}


def execution_manifest_core(
    *,
    required_bindings: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    bindings = {name: None for name in REQUIRED_BINDING_NAMES}
    if required_bindings is not None:
        if set(required_bindings) != set(REQUIRED_BINDING_NAMES):
            raise ValueError("exact manifest required-binding names changed")
        bindings.update(dict(required_bindings))
    unresolved = sorted(name for name, value in bindings.items() if value is None)
    for name, value in bindings.items():
        if value is not None and not is_sha256(value):
            raise ValueError(f"exact manifest binding is malformed: {name}")
    ready = not unresolved
    return {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "status": "ready_for_exact_reservation" if ready else "blocked_required_bindings_unset",
        "reviewed_design_and_model_bindings": reviewed_source_bindings(),
        "required_exact_bindings": bindings,
        "unresolved_required_bindings": unresolved,
        "live_navigation_readiness_hash_authoritative": False,
        "non_authoritative_status_context": {
            "path": "docs/lewm_go2_navigation_work_readiness_goal_2026-07-13.md",
            "hash_excluded": True,
        },
        "preflight_root": PREFLIGHT_ROOT_RELATIVE_PATH,
        "exact_root": EXACT_ROOT_RELATIVE_PATH,
        "preflight_and_exact_processes_distinct": True,
        "exact_reservation_before_torch_model_or_payload": True,
        "exact_execution_authorized": ready,
        "retry_authorized": False,
        "g2_authorized": False,
        "heldout_authorized": False,
        "runtime_navigation_hardware_authorized": False,
        "production_or_promotion_authorized": False,
    }


def validate_execution_manifest(
    value: Mapping[str, Any],
    *,
    require_ready: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("exact execution manifest must be a mapping")
    copied = dict(value)
    declared = copied.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(copied) != declared:
        raise ValueError("exact execution manifest content hash changed")
    bindings = copied.get("required_exact_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("exact execution manifest bindings are missing")
    expected = execution_manifest_core(required_bindings=bindings)
    if copied != expected:
        raise PermissionError("exact execution manifest contract changed")
    if require_ready and not expected["exact_execution_authorized"]:
        unresolved = ", ".join(expected["unresolved_required_bindings"])
        raise PermissionError(
            "exact execution is blocked before reservation and payload; unset: "
            + unresolved
        )
    return {**copied, "content_sha256": declared}


def learning_rate(update: int) -> float:
    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= 8000:
        raise ValueError("update must lie in [1,8000]")
    if update <= 400:
        return 1e-6 + (1e-4 - 1e-6) * (update - 1) / 399
    return 1e-5 + 0.5 * (1e-4 - 1e-5) * (
        1.0 + math.cos(math.pi * (update - 400) / 7600)
    )


def validate_exact_schedule_indices(indices: Sequence[int]) -> tuple[int, ...]:
    if len(indices) != PRESENTATION_COUNT:
        raise ValueError("exact train schedule must contain 128000 presentations")
    normalized: list[int] = []
    expected_cycle = set(range(TRAIN_PAIR_COUNT))
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("schedule indices must be integers")
        if not 0 <= value < TRAIN_PAIR_COUNT:
            raise ValueError("schedule index escaped the train role")
        normalized.append(value)
    complete_cycles, remainder = divmod(PRESENTATION_COUNT, TRAIN_PAIR_COUNT)
    for cycle in range(complete_cycles):
        start = cycle * TRAIN_PAIR_COUNT
        if set(normalized[start : start + TRAIN_PAIR_COUNT]) != expected_cycle:
            raise ValueError("schedule complete cycle is not a train-role permutation")
    if len(set(normalized[-remainder:])) != remainder:
        raise ValueError("schedule partial cycle repeats a pair")
    return tuple(normalized)


def schedule_commitment(
    indices: Sequence[int],
    ordered_pair_ids: Sequence[str],
) -> dict[str, Any]:
    normalized = validate_exact_schedule_indices(indices)
    if (
        len(ordered_pair_ids) != TRAIN_PAIR_COUNT
        or len(set(ordered_pair_ids)) != TRAIN_PAIR_COUNT
        or any(not isinstance(value, str) or not value for value in ordered_pair_ids)
    ):
        raise ValueError("ordered train-pair identities changed")
    presentations = [ordered_pair_ids[index] for index in normalized]
    update_hashes = [
        canonical_json_sha256(
            presentations[offset : offset + EFFECTIVE_BATCH_SIZE]
        )
        for offset in range(0, PRESENTATION_COUNT, EFFECTIVE_BATCH_SIZE)
    ]
    core = {
        "schema": SCHEDULE_SCHEMA,
        "seed": SCHEDULE_SEED,
        "train_pair_count": TRAIN_PAIR_COUNT,
        "presentation_count": PRESENTATION_COUNT,
        "update_count": UPDATE_COUNT,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "ordered_pair_ids_sha256": canonical_json_sha256(list(ordered_pair_ids)),
        "indices_sha256": canonical_json_sha256(list(normalized)),
        "presentation_pair_ids_sha256": canonical_json_sha256(presentations),
        "per_update_pair_ids_sha256": canonical_json_sha256(update_hashes),
    }
    return content_value(core)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _lower_margin(value: object, threshold: float, *, name: str) -> float:
    return (_finite(value, name=name) - threshold) / max(abs(threshold), 1e-12)


def _upper_margin(value: object, threshold: float, *, name: str) -> float:
    return (threshold - _finite(value, name=name)) / max(abs(threshold), 1e-12)


def evaluate_checkpoint_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(scope, Mapping) or set(scope) != {"physical", "jepa"}:
        raise ValueError("checkpoint scope must contain physical and jepa metrics")
    physical = scope["physical"]
    jepa = scope["jepa"]
    if not isinstance(physical, Mapping) or not isinstance(jepa, Mapping):
        raise TypeError("checkpoint metrics must be mappings")
    margins: list[float] = []
    for name, threshold in PHYSICAL_LOWER_THRESHOLDS.items():
        margins.append(_lower_margin(physical.get(name), threshold, name=name))
    for name, threshold in PHYSICAL_UPPER_THRESHOLDS.items():
        margins.append(_upper_margin(physical.get(name), threshold, name=name))
    distance_groups = physical.get("distance_group_balanced_accuracy")
    if not isinstance(distance_groups, Sequence) or isinstance(distance_groups, (str, bytes)) or not distance_groups:
        raise ValueError("distance-group metrics are empty")
    margins.extend(
        _lower_margin(value, 0.92, name="distance_group_balanced_accuracy")
        for value in distance_groups
    )
    recalls = physical.get("present_class_recall")
    if not isinstance(recalls, Mapping) or not recalls or not set(recalls) <= {
        "UNKNOWN",
        "FREE",
        "OCCUPIED",
    }:
        raise ValueError("present class recalls are malformed")
    margins.extend(
        _lower_margin(value, 0.95, name=f"{name}_recall")
        for name, value in sorted(recalls.items())
    )

    prediction_cells = _finite(
        jepa.get("prediction_valid_cell_count"),
        name="prediction_valid_cell_count",
    )
    target_change = _finite(
        jepa.get("warped_persistence_target_change"),
        name="warped_persistence_target_change",
    )
    ratio = _finite(
        jepa.get("prediction_to_warped_persistence_ratio"),
        name="prediction_to_warped_persistence_ratio",
    )
    jepa_margins = [
        prediction_cells,
        _lower_margin(
            jepa.get("target_cross_sample_std_mean"),
            JEPA_LOWER_THRESHOLDS["target_cross_sample_std_mean"],
            name="target_cross_sample_std_mean",
        ),
        _lower_margin(
            jepa.get("target_cross_sample_effective_rank"),
            JEPA_LOWER_THRESHOLDS["target_cross_sample_effective_rank"],
            name="target_cross_sample_effective_rank",
        ),
        target_change - 1e-4,
        1.0 - ratio,
        _lower_margin(
            jepa.get("wrong_action_advantage_over_target_change"),
            JEPA_LOWER_THRESHOLDS["wrong_action_advantage_over_target_change"],
            name="wrong_action_advantage_over_target_change",
        ),
        _finite(
            jepa.get("wrong_commanded_delta_advantage_over_target_change"),
            name="wrong_commanded_delta_advantage_over_target_change",
        ),
        _finite(
            jepa.get("wrong_action_prediction_sensitivity"),
            name="wrong_action_prediction_sensitivity",
        ),
        _finite(
            jepa.get("wrong_commanded_delta_prediction_sensitivity"),
            name="wrong_commanded_delta_prediction_sensitivity",
        ),
    ]
    eligible = (
        all(value >= 0.0 for value in margins)
        and prediction_cells > 0.0
        and jepa_margins[1] >= 0.0
        and jepa_margins[2] >= 0.0
        and target_change > 1e-4
        and ratio < 1.0
        and jepa_margins[5] >= 0.0
        and all(value > 0.0 for value in jepa_margins[6:])
    )
    return {
        "eligible": eligible,
        "physical_margins": margins,
        "jepa_margins": jepa_margins,
    }


def evaluate_checkpoint_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise TypeError("checkpoint candidate must be a mapping")
    update = candidate.get("update")
    if update not in CHECKPOINT_UPDATES:
        raise ValueError("checkpoint candidate update changed")
    scopes = candidate.get("scopes")
    if not isinstance(scopes, Mapping) or tuple(scopes) != SCOPES:
        raise ValueError("checkpoint candidate scope order changed")
    evaluated = {name: evaluate_checkpoint_scope(scopes[name]) for name in SCOPES}
    physical = [
        margin
        for scope in SCOPES
        for margin in evaluated[scope]["physical_margins"]
    ]
    jepa = [
        margin
        for scope in SCOPES
        for margin in evaluated[scope]["jepa_margins"]
    ]
    aggregate_v4_loss = _finite(
        candidate.get("aggregate_complete_v4_loss"),
        name="aggregate_complete_v4_loss",
    )
    aggregate_ratio = _finite(
        candidate.get("aggregate_prediction_to_persistence_ratio"),
        name="aggregate_prediction_to_persistence_ratio",
    )
    eligible = all(value["eligible"] for value in evaluated.values())
    rank = (
        min(physical),
        min(jepa),
        sum(physical) / len(physical),
        sum(jepa) / len(jepa),
        -aggregate_v4_loss,
        -aggregate_ratio,
        -int(update),
    )
    return {
        "update": update,
        "eligible": eligible,
        "rank": rank,
        "scope_evaluations": evaluated,
    }


def select_promoted_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [candidate.get("update") for candidate in candidates] != list(CHECKPOINT_UPDATES):
        raise ValueError("promoted checkpoint candidates are incomplete or reordered")
    evaluated = [evaluate_checkpoint_candidate(candidate) for candidate in candidates]
    eligible = [candidate for candidate in evaluated if candidate["eligible"]]
    if not eligible:
        raise ValueError("no eligible promoted checkpoint exists")
    selected = max(eligible, key=lambda candidate: candidate["rank"])
    return {
        "selected_update": selected["update"],
        "selected_rank": list(selected["rank"]),
        "eligible_updates": [candidate["update"] for candidate in eligible],
        "candidate_evaluations_sha256": canonical_json_sha256(evaluated),
    }


def centered_vector_scaling_parameters(
    log_scales: Sequence[float],
    biases: Sequence[float],
) -> dict[str, list[float]]:
    if len(log_scales) != 3 or len(biases) != 3:
        raise ValueError("vector calibration requires three scales and biases")
    clamped = [max(-3.0, min(3.0, _finite(value, name="log_scale"))) for value in log_scales]
    raw_biases = [_finite(value, name="bias") for value in biases]
    mean_bias = sum(raw_biases) / 3.0
    return {
        "log_scales": clamped,
        "scales": [math.exp(value) for value in clamped],
        "centered_biases": [value - mean_bias for value in raw_biases],
    }


def threshold_grid() -> tuple[tuple[float, float, float, float], ...]:
    result: list[tuple[float, float, float, float]] = []
    for free_min in CALIBRATION_FREE_MIN_GRID:
        for occupied_max in CALIBRATION_OCCUPIED_MAX_GRID:
            for unknown_max in CALIBRATION_UNKNOWN_MAX_GRID:
                for occupied_detection_min in CALIBRATION_OCCUPIED_DETECTION_MIN_GRID:
                    if occupied_max >= occupied_detection_min:
                        continue
                    result.append(
                        (
                            free_min,
                            occupied_max,
                            unknown_max,
                            occupied_detection_min,
                        )
                    )
    return tuple(result)


def select_calibration_threshold(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = {
        canonical_json_sha256(list(values)): values for values in threshold_grid()
    }
    if set(reports) != set(expected_keys):
        raise ValueError("calibration threshold grid reports changed")
    best: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    for key, values in expected_keys.items():
        report = reports[key]
        required = {
            "admitted_free_count",
            "admitted_free_true_free_count",
            "useful_free_count",
            "useful_free_admitted_count",
            "obstacle_within_2m_count",
            "obstacle_within_2m_excluded_count",
            "obstacle_within_2m_detected_count",
        }
        if not isinstance(report, Mapping) or set(report) != required:
            raise ValueError("calibration threshold report fields changed")
        counts: dict[str, int] = {}
        for name in sorted(required):
            number = _finite(report[name], name=name)
            integer = int(number)
            if number != integer:
                raise ValueError(f"calibration count is not integral: {name}")
            counts[name] = integer
        if any(value < 0 for value in counts.values()):
            raise ValueError("calibration counts must be nonnegative")
        admitted = counts["admitted_free_count"]
        useful = counts["useful_free_count"]
        obstacles = counts["obstacle_within_2m_count"]
        if admitted <= 0 or useful <= 0 or obstacles <= 0:
            continue
        precision = counts["admitted_free_true_free_count"] / admitted
        useful_recall = counts["useful_free_admitted_count"] / useful
        exclusion_recall = counts["obstacle_within_2m_excluded_count"] / obstacles
        detection_recall = counts["obstacle_within_2m_detected_count"] / obstacles
        passed = precision >= 0.99 and exclusion_recall >= 0.95 and detection_recall >= 0.95
        rank = (useful_recall, precision, detection_recall, values[3], -values[0])
        if passed and (best_rank is None or rank > best_rank):
            best_rank = rank
            best = {
                "free_probability_minimum": values[0],
                "occupied_probability_maximum": values[1],
                "unknown_probability_maximum": values[2],
                "occupied_detection_minimum": values[3],
                "useful_free_recall": useful_recall,
                "admitted_free_precision": precision,
                "obstacle_exclusion_recall_within_2m": exclusion_recall,
                "obstacle_detection_recall_within_2m": detection_recall,
                "rank": list(rank),
            }
    if best is None:
        raise ValueError("no calibration threshold tuple passes")
    return best


def selection_role_ablation_contract() -> dict[str, Any]:
    return {
        "population_role": "checkpoint_selection",
        "interpretation": "matched_development_diagnostic_only",
        "causal_generalization_claim_authorized": False,
        "qualification_or_selection_effect": "none",
        "ablation_checkpoint_substitution_authorized": False,
        "retry_or_intervention_authorized": False,
    }


def append_access_event(
    events: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    arm: str | None,
    role: str,
    operation: str,
    relative_path: str,
    expected_sha256: str,
    observed_sha256: str,
    byte_count: int,
    process_identity: str,
) -> dict[str, Any]:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PermissionError("access-ledger path escaped")
    if role in FORBIDDEN_ROLES:
        raise PermissionError(f"forbidden role open: {role}")
    if role not in (*DEVELOPMENT_ROLES, *PROVENANCE_ROLES):
        raise PermissionError("access-ledger role changed")
    if arm is not None and arm not in ARMS:
        raise PermissionError("access-ledger arm changed")
    if not is_sha256(expected_sha256) or not is_sha256(observed_sha256):
        raise ValueError("access-ledger hash is malformed")
    if expected_sha256 != observed_sha256:
        raise PermissionError("access-ledger observed hash differs from binding")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("access-ledger byte count is malformed")
    prior = events[-1].get("event_sha256") if events else "0" * 64
    if not is_sha256(prior):
        raise ValueError("access-ledger prior event is malformed")
    core = {
        "schema": ACCESS_LEDGER_SCHEMA,
        "sequence": len(events),
        "stage": stage,
        "arm": arm,
        "role": role,
        "operation": operation,
        "relative_path": str(path),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "byte_count": byte_count,
        "process_identity": process_identity,
        "prior_event_sha256": prior,
    }
    return {**core, "event_sha256": canonical_json_sha256(core)}


def validate_access_ledger(
    events: Sequence[Mapping[str, Any]],
    *,
    require_completion_rehash: bool = False,
) -> dict[str, Any]:
    prior = "0" * 64
    counts: dict[str, int] = {}
    opened_inputs: set[tuple[Any, ...]] = set()
    completion_rehashes: set[tuple[Any, ...]] = set()
    allowed_stage_roles = {
        "preflight_source_closure": {"source_closure"},
        "exact_source_closure": {
            "source_closure",
            "execution_manifest",
            "implementation_review",
        },
        "exact_input": {
            "preflight_receipt",
            "raw_supervision_manifest",
            "raw_supervision_audit",
            "v4_two_seed_ladder",
            "v4_primary_checkpoint",
        },
        "gradient": {"train"},
        "selection": {"checkpoint_selection"},
        "diagnostic": {"checkpoint_selection"},
        "calibration": {"probability_calibration"},
        "completion_rehash": set((*DEVELOPMENT_ROLES, *PROVENANCE_ROLES)),
    }
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TypeError("access-ledger event must be a mapping")
        core = dict(event)
        declared = core.pop("event_sha256", None)
        if (
            core.get("schema") != ACCESS_LEDGER_SCHEMA
            or core.get("sequence") != index
            or core.get("prior_event_sha256") != prior
            or not is_sha256(declared)
            or canonical_json_sha256(core) != declared
        ):
            raise PermissionError("access-ledger chain changed")
        role = core.get("role")
        stage = core.get("stage")
        if role in FORBIDDEN_ROLES:
            raise PermissionError("forbidden role appears in access ledger")
        if stage not in allowed_stage_roles or role not in allowed_stage_roles[stage]:
            raise PermissionError("access-ledger stage/role boundary changed")
        arm = core.get("arm")
        if role in PROVENANCE_ROLES and arm is not None:
            raise PermissionError("provenance open was assigned to a training arm")
        if role in DEVELOPMENT_ROLES and arm not in ARMS:
            raise PermissionError("development payload open lacks an exact arm")
        if not isinstance(core.get("operation"), str) or not core["operation"]:
            raise ValueError("access-ledger operation is malformed")
        if core.get("expected_sha256") != core.get("observed_sha256"):
            raise PermissionError("access-ledger expected/observed hash changed")
        input_identity = (
            arm,
            role,
            core.get("relative_path"),
            core.get("expected_sha256"),
            core.get("observed_sha256"),
            core.get("byte_count"),
        )
        if stage == "completion_rehash":
            if input_identity in completion_rehashes:
                raise PermissionError("access-ledger completion rehash is duplicated")
            completion_rehashes.add(input_identity)
        else:
            opened_inputs.add(input_identity)
        counts[str(role)] = counts.get(str(role), 0) + 1
        prior = str(declared)
    if require_completion_rehash and completion_rehashes != opened_inputs:
        raise PermissionError("access-ledger completion rehash closure changed")
    if not require_completion_rehash and completion_rehashes:
        raise PermissionError("unexpected access-ledger completion rehash")
    return {
        "event_count": len(events),
        "terminal_event_sha256": prior,
        "role_event_counts": counts,
        "forbidden_open_count": 0,
        "unique_input_count": len(opened_inputs),
        "completion_rehash_event_count": len(completion_rehashes),
    }


def artifact_binding(
    relative_path: str,
    raw: bytes,
    *,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PermissionError("artifact binding path escaped")
    result: dict[str, Any] = {
        "path": str(path),
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if content_sha256 is not None:
        if not is_sha256(content_sha256):
            raise ValueError("artifact content hash is malformed")
        result["content_sha256"] = content_sha256
    return result


def verify_fixed_source_hashes(read_bytes: Any) -> dict[str, str]:
    """Rehash reviewed source using a fixed reader supplied by an entry script."""

    result: dict[str, str] = {}
    for relative, expected in reviewed_source_bindings().items():
        raw = read_bytes(relative)
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected:
            raise PermissionError(f"reviewed source changed: {relative}")
        result[relative] = observed
    return result


__all__ = [
    "ACCESS_LEDGER_SCHEMA",
    "ACCUMULATION_STEPS",
    "ARMS",
    "CANONICAL_EXACT_ROOT",
    "CANONICAL_PREFLIGHT_ROOT",
    "CHECKPOINT_UPDATES",
    "DEVELOPMENT_ROLES",
    "DEVICE_CONTRACT",
    "EFFECTIVE_BATCH_SIZE",
    "EXACT_EXECUTION_MANIFEST_RELATIVE_PATH",
    "EXACT_INVENTORY",
    "EXACT_ROOT_RELATIVE_PATH",
    "FAMILIES",
    "INITIALIZATION_SEED",
    "IMPLEMENTATION_REVIEW_RELATIVE_PATH",
    "IMPLEMENTATION_SOURCE_PATHS",
    "INITIALIZATION_SCHEMA",
    "INPUT_BINDINGS_SCHEMA",
    "JOINT_LOSS_CONTRACT",
    "MICROBATCH_SIZE",
    "OPTIMIZER_CONTRACT",
    "PREFLIGHT_INVENTORY",
    "PREFLIGHT_ROOT_RELATIVE_PATH",
    "PRESENTATION_COUNT",
    "PRIMARY_V4_SEED",
    "PROVENANCE_ROLES",
    "RAW_SUPERVISION_AUDIT_RELATIVE_PATH",
    "RAW_SUPERVISION_AUDIT_SCHEMA",
    "RAW_SUPERVISION_AUDITOR_RELATIVE_PATH",
    "RAW_SUPERVISION_BUILDER_RELATIVE_PATH",
    "RAW_SUPERVISION_MANIFEST_RELATIVE_PATH",
    "RAW_SUPERVISION_MANIFEST_SCHEMA",
    "RAW_SUPERVISION_ROOT_RELATIVE_PATH",
    "REQUIRED_BINDING_NAMES",
    "ROLE_COUNTS",
    "SCHEDULE_SEED",
    "SELECTION_SCHEMA",
    "SCOPES",
    "SOURCE_REVIEW_SCHEMA",
    "TRAIN_PAIR_COUNT",
    "TRAINING_RECORD_SCHEMA",
    "UPDATE_COUNT",
    "V4_PRIMARY_CHECKPOINT_RELATIVE_PATH",
    "V4_TWO_SEED_LADDER_RELATIVE_PATH",
    "append_access_event",
    "artifact_binding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "centered_vector_scaling_parameters",
    "content_value",
    "evaluate_checkpoint_candidate",
    "evaluate_checkpoint_scope",
    "execution_manifest_core",
    "expected_implementation_review_core",
    "is_sha256",
    "learning_rate",
    "parse_canonical_json",
    "reviewed_source_bindings",
    "schedule_commitment",
    "select_calibration_threshold",
    "select_promoted_checkpoint",
    "selection_role_ablation_contract",
    "threshold_grid",
    "validate_access_ledger",
    "validate_exact_schedule_indices",
    "validate_execution_manifest",
    "validate_implementation_review",
    "verify_fixed_source_hashes",
]

"""Lean contract for the one Shared-V5 matched development training attempt.

This module is deliberately standard-library only.  It freezes the useful
science from Full Training V4 without importing that historical lifecycle.  A
separate runner owns reservation, payload access, training and publication.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/lean_shared_v5_impl"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_matched_training_v1"

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py"
)
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_matched_training_v1.py"
MODEL_RELATIVE_PATH = "lewm/models/shared_observable_camera_ray_jepa_v5.py"
LOSS_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v4_loss.py"
)
METRICS_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py"
)
EVIDENCE_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py"
)
EGOMOTION_RELATIVE_PATH = "lewm/models/egomotion_bev_jepa.py"
ENCODERS_RELATIVE_PATH = "lewm/models/encoders.py"
EVIDENCE_MODEL_RELATIVE_PATH = "lewm/models/observable_camera_ray_evidence_v4.py"
EVIDENCE_TRAINING_RELATIVE_PATH = (
    "lewm/models/observable_camera_ray_evidence_v4_training.py"
)
HIERARCHICAL_FIRST_HIT_RELATIVE_PATH = (
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py"
)
GATE_ALIGNED_NLL_RELATIVE_PATH = (
    "lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py"
)
DIRECT_SEMANTIC_DEPENDENCY_PATHS = (
    EVIDENCE_CONTRACT_RELATIVE_PATH,
    ENCODERS_RELATIVE_PATH,
    EGOMOTION_RELATIVE_PATH,
    EVIDENCE_MODEL_RELATIVE_PATH,
    EVIDENCE_TRAINING_RELATIVE_PATH,
    GATE_ALIGNED_NLL_RELATIVE_PATH,
    HIERARCHICAL_FIRST_HIT_RELATIVE_PATH,
)
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    LOSS_RELATIVE_PATH,
    METRICS_RELATIVE_PATH,
    *DIRECT_SEMANTIC_DEPENDENCY_PATHS,
)
MODEL_FILE_SHA256 = (
    "b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9"
)
LOSS_FILE_SHA256 = (
    "8422c253c3eca3b34dd42b4f823dab4ac67f0e90fb2cff8eeaa67a1310b3c53a"
)
EGOMOTION_FILE_SHA256 = (
    "c4006e9804182b077399229d43bc8c9be64b5af12c81fff4076d5a78e6ef359b"
)

REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v1_"
    "independent_review_2026-07-14.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_matched_training_v1_"
    "execution_authorization_2026-07-14.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "matched_training_v1"
)
RAW_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
RAW_MANIFEST_RELATIVE_PATH = f"{RAW_ROOT_RELATIVE_PATH}/manifest.json"
RAW_AUDIT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1.audit_v13.json"
)
CAMERA_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1"
)
CAMERA_GATE_RELATIVE_PATH = f"{CAMERA_ROOT_RELATIVE_PATH}/gate.json"
CAMERA_CHECKPOINT_RELATIVE_PATH = (
    f"{CAMERA_ROOT_RELATIVE_PATH}/checkpoint.pt"
)

RAW_MANIFEST_FILE_SHA256 = (
    "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360"
)
RAW_MANIFEST_CONTENT_SHA256 = (
    "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a"
)
RAW_AUDIT_FILE_SHA256 = (
    "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76"
)
RAW_AUDIT_CONTENT_SHA256 = (
    "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca"
)
RAW_SAMPLE_RESULTS_SHA256 = (
    "a051b9a0a10f14413105f2f1cc3c36ad10a43ec20071f0577efcc99fc321d356"
)
RAW_ORDERED_PAIR_SHA256 = (
    "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
)
RAW_ORDERED_ENDPOINT_SHA256 = (
    "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
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
    "train": {"pairs": 4262, "unique_endpoints": 7777, "scenes": 72},
    "checkpoint_selection": {
        "pairs": 495,
        "unique_endpoints": 924,
        "scenes": 8,
    },
    "probability_calibration": {
        "pairs": 415,
        "unique_endpoints": 759,
        "scenes": 8,
    },
}
ROLES = tuple(ROLE_COUNTS)
RAW_ARRAY_LAYOUT = (
    {"path": "camera_origin_body_m.f4", "dtype": "<f4", "trailing_shape": [3]},
    {
        "path": "camera_basis_body_fru.f4",
        "dtype": "<f4",
        "trailing_shape": [3, 3],
    },
    {"path": "ground_plane_z_body_m.f4", "dtype": "<f4", "trailing_shape": []},
    {
        "path": "ground_support_in_frustum.u1",
        "dtype": "|u1",
        "trailing_shape": [128, 128, 5],
    },
    {
        "path": "ground_support_clear_to_target.u1",
        "dtype": "|u1",
        "trailing_shape": [128, 128, 5],
    },
    {"path": "pixel_hit_mask.u1", "dtype": "|u1", "trailing_shape": [84, 112]},
    {
        "path": "pixel_first_hit_distance_m.f4",
        "dtype": "<f4",
        "trailing_shape": [84, 112],
    },
    {"path": "raster_labels.u1", "dtype": "|u1", "trailing_shape": [64, 64]},
)

ARMS = ("promoted_jepa", "matched_no_jepa")
INITIALIZATION_SEED = 20260712
SCHEDULE_SEED = 20260713
TRAIN_PAIR_COUNT = 4262
PRESENTATION_COUNT = 128_000
UPDATE_COUNT = 8_000
MICROBATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = 16
CHECKPOINT_UPDATES = tuple(range(1000, UPDATE_COUNT + 1, 1000))
CAMERA_TERMS = (
    "hierarchical_first_hit_nll",
    "target_bin_offset_smooth_l1",
    "ground_clear_distance_state_balanced_bce",
    "derived_raster_hierarchical_bce",
    "derived_raster_cell_nll",
)
OPTIMIZER_CONTRACT = {
    "name": "AdamW",
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 1e-4,
    "amsgrad": False,
    "precision": "float32",
    "autocast": False,
    "gradient_clip_norm": 1.0,
    "microbatch_size": MICROBATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "updates": UPDATE_COUNT,
    "ema_updates_per_optimizer_step": 1,
}
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
CONFLICTING_ACCELERATOR_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
)

REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SCHEDULE_SCHEMA = f"{SCHEMA_PREFIX}_schedule_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_training_snapshot_v1"
SELECTION_SCHEMA = f"{SCHEMA_PREFIX}_selection_v1"
CALIBRATION_SCHEMA = f"{SCHEMA_PREFIX}_calibration_v1"
PRE_G2_CHECKPOINT_SCHEMA = f"{SCHEMA_PREFIX}_pre_g2_candidate_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
VERIFICATION_SCHEMA = f"{SCHEMA_PREFIX}_isolated_verification_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

REVIEW_AUTHORITY = {
    "execution_authorized": False,
    "dataset_use_authorized": False,
    "rgb_decode_authorized": False,
    "camera_checkpoint_use_authorized": False,
    "training_authorized": False,
    "selection_authorized": False,
    "calibration_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "retry_authorized": False,
}
EXECUTION_AUTHORITY = {
    "one_exact_development_attempt_authorized": True,
    "dataset_use_for_fixed_roles_authorized": True,
    "rgb_decode_for_fixed_roles_authorized": True,
    "camera_checkpoint_migration_authorized": True,
    "matched_training_authorized": True,
    "discrete_gpu0_training_authorized": True,
    "promoted_selection_authorized": True,
    "matched_selected_update_diagnostic_authorized": True,
    "promoted_calibration_authorized": True,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "runtime_authorized": False,
    "hardware_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_authorized": False,
}
PRE_G2_DENIALS = {
    "g2_attempted": False,
    "g2_gate_receipt": None,
    "post_g2_qualified": False,
    "runtime_ready": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "hardware_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_authorized": False,
}

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
CALIBRATION_DETECTION_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


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
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    if type(core) is not dict or "content_sha256" in core:
        raise TypeError("self-hashed core must be a plain dict without its hash")
    return {**core, "content_sha256": canonical_json_sha256(core)}


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{name} must be one canonical JSON line")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not ASCII JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{name} is not canonical")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} self hash changed")
    return value


def safe_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} escaped its root")
    return value


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    safe_relative_path(path, name="artifact path")
    if not is_sha256(content_sha256):
        raise ValueError("artifact content hash is malformed")
    return {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
    }


def validate_binding(value: object, *, path: str | None = None) -> dict[str, Any]:
    fields = {"path", "file_sha256", "content_sha256", "byte_count"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("artifact binding fields changed")
    safe_relative_path(value["path"], name="artifact path")
    if (
        (path is not None and value["path"] != path)
        or not is_sha256(value["file_sha256"])
        or not is_sha256(value["content_sha256"])
        or type(value["byte_count"]) is not int
        or value["byte_count"] <= 0
    ):
        raise ValueError("artifact binding changed")
    return dict(value)


def science_contract() -> dict[str, Any]:
    return {
        "model": {
            "class": "SharedObservableCameraRayJepaV5",
            "source_path": MODEL_RELATIVE_PATH,
            "source_file_sha256": MODEL_FILE_SHA256,
        },
        "corrected_loss": {
            "source_path": LOSS_RELATIVE_PATH,
            "source_file_sha256": LOSS_FILE_SHA256,
            "camera_terms": list(CAMERA_TERMS),
            "each_camera_term_weight": 0.25,
            "current_next_weights": [0.5, 0.5],
            "real_microbatch_size": MICROBATCH_SIZE,
            "microbatch_scalar_weights": [0.25] * ACCUMULATION_STEPS,
            "synthetic_b16_pooling": False,
        },
        "arms": {
            "order": list(ARMS),
            "identical_initial_state": True,
            "identical_schedule": True,
            "identical_optimizer": True,
            "identical_forward_and_diagnostics": True,
            "sole_backward_difference": {
                "promoted_jepa": "established_jepa_plus_camera",
                "matched_no_jepa": "camera_only",
            },
            "matched_selection_effect": "none",
            "matched_calibration_effect": "none",
        },
        "initialization_seed": INITIALIZATION_SEED,
        "schedule_seed": SCHEDULE_SEED,
        "train_pair_count": TRAIN_PAIR_COUNT,
        "presentation_count": PRESENTATION_COUNT,
        "update_count": UPDATE_COUNT,
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "optimizer": dict(OPTIMIZER_CONTRACT),
        "selection": {
            "role": "checkpoint_selection",
            "promoted_only": True,
            "scopes": list(SCOPES),
            "evaluation_order": "family_then_endpoint_identity",
            "evaluation_batch_size": MICROBATCH_SIZE,
            "wrong_rgb_mapping": "cyclic_plus_one_within_family",
        },
        "calibration": {
            "role": "probability_calibration",
            "promoted_only": True,
            "scopes": list(SCOPES),
            "method": "six_parameter_centered_vector_scaling",
            "evaluation_order": "family_then_endpoint_identity",
        },
        "candidate": {
            "schema": PRE_G2_CHECKPOINT_SCHEMA,
            "full_evaluation_state_required": True,
            "required_training_state_prefixes": [
                "target_encoder.",
                "target_bev_decoder.",
                "predictor.",
            ],
            "deployment_state_also_required": True,
            **PRE_G2_DENIALS,
        },
        "maximum_attempts": 1,
        "retry_authorized": False,
    }


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    bindings = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }
    if (
        bindings[MODEL_RELATIVE_PATH] != MODEL_FILE_SHA256
        or bindings[LOSS_RELATIVE_PATH] != LOSS_FILE_SHA256
        or bindings[EGOMOTION_RELATIVE_PATH] != EGOMOTION_FILE_SHA256
    ):
        raise PermissionError(
            "stable Shared-V5 model, corrected loss or egomotion base changed"
        )
    return bindings


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "implementation_author",
        "reviewer",
        "reviewed_sources",
        "science_contract",
        "source_only",
        "findings",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("independent review fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(value["reviewer"]) is not str
        or not value["reviewer"].startswith("/root/")
        or value["reviewer"] == IMPLEMENTATION_AUTHOR
        or type(value["reviewed_sources"]) is not dict
        or value["reviewed_sources"] != dict(expected_sources)
        or value["science_contract"] != science_contract()
        or value["source_only"] is not True
        or type(value["findings"]) is not list
        or value["findings"]
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("independent review did not authorize these bytes")
    return dict(value)


def _expected_raw_authority() -> dict[str, Any]:
    return {
        "source_raw_grant_remains_false": True,
        "narrow_grant_created_by_this_authorization": True,
        "allowed_roles": list(ROLES),
        "allowed_operations": [
            "development_rgb_decode",
            "training",
            "promoted_selection",
            "matched_selected_update_diagnostic",
            "promoted_calibration",
        ],
        "g2_navigation_heldout_or_production_use": False,
    }


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "authorizer",
        "independent_review",
        "raw",
        "camera",
        "experiment",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    raw = value.get("raw")
    camera = value.get("camera")
    core = dict(value)
    declared = core.pop("content_sha256")
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "authorized_one_exact_development_attempt"
        or type(value["authorizer"]) is not str
        or not value["authorizer"].startswith("/root")
        or value["independent_review"] != dict(review_binding)
        or type(raw) is not dict
        or set(raw) != {"root", "manifest", "audit", "role_counts", "grant"}
        or raw["root"] != RAW_ROOT_RELATIVE_PATH
        or validate_binding(raw["manifest"], path=RAW_MANIFEST_RELATIVE_PATH)[
            "file_sha256"
        ]
        != RAW_MANIFEST_FILE_SHA256
        or raw["manifest"]["content_sha256"] != RAW_MANIFEST_CONTENT_SHA256
        or validate_binding(raw["audit"], path=RAW_AUDIT_RELATIVE_PATH)[
            "file_sha256"
        ]
        != RAW_AUDIT_FILE_SHA256
        or raw["audit"]["content_sha256"] != RAW_AUDIT_CONTENT_SHA256
        or raw["role_counts"] != ROLE_COUNTS
        or raw["grant"] != _expected_raw_authority()
        or type(camera) is not dict
        or set(camera)
        != {
            "root",
            "gate",
            "checkpoint",
            "seed",
            "fit_size",
            "updates",
            "gate_must_pass_all_checks",
        }
        or camera["root"] != CAMERA_ROOT_RELATIVE_PATH
        or validate_binding(camera["gate"], path=CAMERA_GATE_RELATIVE_PATH)[
            "path"
        ]
        != CAMERA_GATE_RELATIVE_PATH
        or validate_binding(
            camera["checkpoint"], path=CAMERA_CHECKPOINT_RELATIVE_PATH
        )["path"]
        != CAMERA_CHECKPOINT_RELATIVE_PATH
        or camera["seed"] != 20260710
        or camera["fit_size"] != 320
        or camera["updates"] != 40_000
        or camera["gate_must_pass_all_checks"] != 26
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


def validate_raw_manifest(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "evidence_schema",
        "raster_schema",
        "roles",
        "pair_counts",
        "endpoint_instance_count",
        "unique_endpoint_counts",
        "scene_shard_count",
        "ordered_pair_sha256",
        "ordered_endpoint_sha256",
        "pair_index",
        "endpoint_index",
        "array_layout",
        "shards",
        "files",
        "input_provenance",
        "access_ledger",
        "independent_audit_precommit",
        "parallel_contract",
        "publication",
        "licenses",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise PermissionError("Raw V13 manifest fields changed")
    pair_counts = {role: ROLE_COUNTS[role]["pairs"] for role in ROLES}
    endpoint_counts = {role: ROLE_COUNTS[role]["unique_endpoints"] for role in ROLES}
    core = dict(value)
    declared = core.pop("content_sha256")
    if (
        declared != RAW_MANIFEST_CONTENT_SHA256
        or canonical_json_sha256(core) != declared
        or value["status"] != "complete_pending_independent_audit"
        or value["roles"] != list(ROLES)
        or value["pair_counts"] != pair_counts
        or value["unique_endpoint_counts"] != endpoint_counts
        or value["endpoint_instance_count"] != 10_344
        or value["scene_shard_count"] != 88
        or value["ordered_pair_sha256"] != RAW_ORDERED_PAIR_SHA256
        or value["ordered_endpoint_sha256"] != RAW_ORDERED_ENDPOINT_SHA256
        or value["array_layout"] != list(RAW_ARRAY_LAYOUT)
        or value["pair_index"]
        != {
            "path": "pairs.jsonl",
            "row_count": 5172,
            "file_sha256": value["pair_index"].get("file_sha256")
            if type(value["pair_index"]) is dict
            else None,
        }
        or not is_sha256(value["pair_index"]["file_sha256"])
        or value["endpoint_index"]
        != {
            "path": "endpoints.jsonl",
            "row_count": 9460,
            "file_sha256": value["endpoint_index"].get("file_sha256")
            if type(value["endpoint_index"]) is dict
            else None,
        }
        or not is_sha256(value["endpoint_index"]["file_sha256"])
        or type(value["shards"]) is not list
        or len(value["shards"]) != 88
        or type(value["files"]) is not list
        or not value["files"]
        or type(value["licenses"]) is not dict
        or any(item is not False for item in value["licenses"].values())
    ):
        raise PermissionError("Raw V13 manifest contract changed")
    file_paths: set[str] = set()
    for item in value["files"]:
        if (
            type(item) is not dict
            or set(item) != {"path", "byte_count", "file_sha256"}
            or safe_relative_path(item["path"], name="raw inventory path") in file_paths
            or type(item["byte_count"]) is not int
            or item["byte_count"] <= 0
            or not is_sha256(item["file_sha256"])
        ):
            raise PermissionError("Raw V13 file inventory changed")
        file_paths.add(item["path"])
    if not {"pairs.jsonl", "endpoints.jsonl"} <= file_paths:
        raise PermissionError("Raw V13 indexes are absent from inventory")
    return dict(value)


def validate_raw_audit(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise PermissionError("Raw V13 audit is not a plain dict")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    denials = (
        "dataset_use_authorized",
        "rgb_decode_authorized",
        "training_authorized",
        "selection_authorized",
        "calibration_authorized",
        "g2_authorized",
        "heldout_authorized",
        "navigation_authorized",
        "runtime_authorized",
        "hardware_authorized",
        "production_authorized",
        "promotion_authorized",
        "deployment_authorized",
        "retry_authorized",
    )
    if (
        declared != RAW_AUDIT_CONTENT_SHA256
        or canonical_json_sha256(core) != declared
        or value.get("verdict") != "PASS"
        or value.get("dataset_manifest_file_sha256") != RAW_MANIFEST_FILE_SHA256
        or value.get("dataset_manifest_content_sha256")
        != RAW_MANIFEST_CONTENT_SHA256
        or value.get("pair_count") != 5172
        or value.get("unique_endpoint_count") != 9460
        or value.get("scene_shard_count") != 88
        or value.get("sample_count") != 24
        or value.get("sample_results_sha256") != RAW_SAMPLE_RESULTS_SHA256
        or type(value.get("sample_results")) is not list
        or len(value["sample_results"]) != 24
        or any(value.get(name) is not False for name in denials)
    ):
        raise PermissionError("Raw V13 PASS or downstream denials changed")
    coverage = {
        (item.get("dataset_role"), item.get("family"))
        for item in value["sample_results"]
        if type(item) is dict and item.get("passes") is True
    }
    if coverage != {(role, family) for role in ROLES for family in FAMILIES}:
        raise PermissionError("Raw V13 role/family audit coverage changed")
    return dict(value)


def learning_rate(update: int) -> float:
    if type(update) is not int or not 1 <= update <= UPDATE_COUNT:
        raise ValueError("update must lie in [1,8000]")
    if update <= 400:
        return 1e-6 + (1e-4 - 1e-6) * (update - 1) / 399
    return 1e-5 + 0.5 * (1e-4 - 1e-5) * (
        1.0 + math.cos(math.pi * (update - 400) / 7600)
    )


def validate_schedule_indices(indices: Sequence[int]) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)) or len(indices) != PRESENTATION_COUNT:
        raise ValueError("schedule must contain exactly 128000 presentations")
    result: list[int] = []
    complete, remainder = divmod(PRESENTATION_COUNT, TRAIN_PAIR_COUNT)
    expected = set(range(TRAIN_PAIR_COUNT))
    for value in indices:
        if type(value) is not int or not 0 <= value < TRAIN_PAIR_COUNT:
            raise ValueError("schedule escaped the train role")
        result.append(value)
    for cycle in range(complete):
        start = cycle * TRAIN_PAIR_COUNT
        if set(result[start : start + TRAIN_PAIR_COUNT]) != expected:
            raise ValueError("complete schedule cycle is not a permutation")
    if len(set(result[-remainder:])) != remainder:
        raise ValueError("partial schedule cycle repeats a pair")
    return tuple(result)


def schedule_core(
    indices: Sequence[int],
    ordered_pair_ids: Sequence[str],
) -> dict[str, Any]:
    normalized = validate_schedule_indices(indices)
    if (
        len(ordered_pair_ids) != TRAIN_PAIR_COUNT
        or len(set(ordered_pair_ids)) != TRAIN_PAIR_COUNT
        or any(not is_sha256(item) for item in ordered_pair_ids)
    ):
        raise ValueError("ordered train-pair identities changed")
    presented = [ordered_pair_ids[index] for index in normalized]
    update_hashes = [
        canonical_json_sha256(presented[start : start + EFFECTIVE_BATCH_SIZE])
        for start in range(0, PRESENTATION_COUNT, EFFECTIVE_BATCH_SIZE)
    ]
    return {
        "schema": SCHEDULE_SCHEMA,
        "seed": SCHEDULE_SEED,
        "train_pair_count": TRAIN_PAIR_COUNT,
        "presentation_count": PRESENTATION_COUNT,
        "update_count": UPDATE_COUNT,
        "microbatch_size": MICROBATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "ordered_pair_ids_sha256": canonical_json_sha256(list(ordered_pair_ids)),
        "indices_sha256": canonical_json_sha256(list(normalized)),
        "presentation_pair_ids_sha256": canonical_json_sha256(presented),
        "per_update_pair_ids_sha256": canonical_json_sha256(update_hashes),
    }


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _lower(value: object, threshold: float, *, name: str) -> float:
    return (_finite(value, name=name) - threshold) / max(threshold, 1e-12)


def _upper(value: object, threshold: float, *, name: str) -> float:
    return (threshold - _finite(value, name=name)) / max(threshold, 1e-12)


def evaluate_checkpoint_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    if type(scope) is not dict or set(scope) != {"physical", "jepa"}:
        raise ValueError("selection scope fields changed")
    physical = scope["physical"]
    jepa = scope["jepa"]
    if type(physical) is not dict or type(jepa) is not dict:
        raise TypeError("selection metrics must be plain dicts")
    physical_margins = [
        _lower(physical.get(name), threshold, name=name)
        for name, threshold in PHYSICAL_LOWER_THRESHOLDS.items()
    ] + [
        _upper(physical.get(name), threshold, name=name)
        for name, threshold in PHYSICAL_UPPER_THRESHOLDS.items()
    ]
    distance = physical.get("distance_group_balanced_accuracy")
    recalls = physical.get("present_class_recall")
    if (
        not isinstance(distance, Sequence)
        or isinstance(distance, (str, bytes))
        or not distance
        or type(recalls) is not dict
        or not recalls
    ):
        raise ValueError("selection physical groups are empty")
    physical_margins.extend(_lower(item, 0.92, name="distance group") for item in distance)
    physical_margins.extend(
        _lower(item, 0.95, name=f"{name} recall")
        for name, item in sorted(recalls.items())
    )
    cells = _finite(jepa.get("prediction_valid_cell_count"), name="valid cells")
    target_change = _finite(
        jepa.get("warped_persistence_target_change"), name="target change"
    )
    ratio = _finite(
        jepa.get("prediction_to_warped_persistence_ratio"), name="prediction ratio"
    )
    jepa_margins = [
        cells,
        _lower(
            jepa.get("target_cross_sample_std_mean"),
            JEPA_LOWER_THRESHOLDS["target_cross_sample_std_mean"],
            name="target std",
        ),
        _lower(
            jepa.get("target_cross_sample_effective_rank"),
            JEPA_LOWER_THRESHOLDS["target_cross_sample_effective_rank"],
            name="target rank",
        ),
        target_change - 1e-4,
        1.0 - ratio,
        _lower(
            jepa.get("wrong_action_advantage_over_target_change"),
            JEPA_LOWER_THRESHOLDS["wrong_action_advantage_over_target_change"],
            name="wrong action advantage",
        ),
        _finite(
            jepa.get("wrong_commanded_delta_advantage_over_target_change"),
            name="wrong delta advantage",
        ),
        _finite(
            jepa.get("wrong_action_prediction_sensitivity"),
            name="wrong action sensitivity",
        ),
        _finite(
            jepa.get("wrong_commanded_delta_prediction_sensitivity"),
            name="wrong delta sensitivity",
        ),
    ]
    eligible = (
        all(item >= 0.0 for item in physical_margins)
        and cells > 0.0
        and all(item >= 0.0 for item in jepa_margins[1:6])
        and all(item > 0.0 for item in jepa_margins[6:])
    )
    return {
        "eligible": eligible,
        "physical_margins": physical_margins,
        "jepa_margins": jepa_margins,
    }


def evaluate_checkpoint_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if type(candidate) is not dict or candidate.get("update") not in CHECKPOINT_UPDATES:
        raise ValueError("selection checkpoint update changed")
    scopes = candidate.get("scopes")
    if type(scopes) is not dict or tuple(scopes) != SCOPES:
        raise ValueError("selection scope order changed")
    evaluated = {scope: evaluate_checkpoint_scope(scopes[scope]) for scope in SCOPES}
    physical = [
        item
        for scope in SCOPES
        for item in evaluated[scope]["physical_margins"]
    ]
    jepa = [
        item for scope in SCOPES for item in evaluated[scope]["jepa_margins"]
    ]
    update = int(candidate["update"])
    rank = (
        min(physical),
        min(jepa),
        sum(physical) / len(physical),
        sum(jepa) / len(jepa),
        -_finite(candidate.get("aggregate_complete_v4_loss"), name="V4 loss"),
        -_finite(
            candidate.get("aggregate_prediction_to_persistence_ratio"),
            name="prediction ratio",
        ),
        -update,
    )
    return {
        "update": update,
        "eligible": all(value["eligible"] for value in evaluated.values()),
        "rank": list(rank),
        "scope_evaluations": evaluated,
    }


def select_promoted_checkpoint(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [item.get("update") for item in candidates] != list(CHECKPOINT_UPDATES):
        raise ValueError("promoted snapshots are incomplete or reordered")
    evaluated = [evaluate_checkpoint_candidate(item) for item in candidates]
    eligible = [item for item in evaluated if item["eligible"]]
    if not eligible:
        raise ValueError("no promoted checkpoint passes all nine development scopes")
    selected = max(eligible, key=lambda item: tuple(item["rank"]))
    return {
        "schema": SELECTION_SCHEMA,
        "role": "checkpoint_selection",
        "selected_arm": "promoted_jepa",
        "selected_update": selected["update"],
        "selected_rank": selected["rank"],
        "eligible_updates": [item["update"] for item in eligible],
        "candidate_evaluations_sha256": canonical_json_sha256(evaluated),
        "matched_no_jepa_influenced_selection": False,
        "calibration_influenced_selection": False,
    }


def threshold_grid() -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        (free, occupied, unknown, detection)
        for free in CALIBRATION_FREE_MIN_GRID
        for occupied in CALIBRATION_OCCUPIED_MAX_GRID
        for unknown in CALIBRATION_UNKNOWN_MAX_GRID
        for detection in CALIBRATION_DETECTION_GRID
        if occupied < detection
    )


def select_calibration_threshold(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {canonical_json_sha256(list(item)): item for item in threshold_grid()}
    if type(reports) is not dict or set(reports) != set(expected):
        raise ValueError("calibration grid changed")
    best: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    names = {
        "admitted_free_count",
        "admitted_free_true_free_count",
        "useful_free_count",
        "useful_free_admitted_count",
        "obstacle_within_2m_count",
        "obstacle_within_2m_excluded_count",
        "obstacle_within_2m_detected_count",
    }
    for key, values in expected.items():
        report = reports[key]
        if type(report) is not dict or set(report) != names:
            raise ValueError("calibration count fields changed")
        counts = {}
        for name in names:
            number = _finite(report[name], name=name)
            if number != int(number) or number < 0:
                raise ValueError("calibration count is not nonnegative integral")
            counts[name] = int(number)
        admitted = counts["admitted_free_count"]
        useful = counts["useful_free_count"]
        obstacles = counts["obstacle_within_2m_count"]
        if min(admitted, useful, obstacles) <= 0:
            continue
        precision = counts["admitted_free_true_free_count"] / admitted
        recall = counts["useful_free_admitted_count"] / useful
        exclusion = counts["obstacle_within_2m_excluded_count"] / obstacles
        detection = counts["obstacle_within_2m_detected_count"] / obstacles
        rank = (recall, precision, detection, values[3], -values[0])
        if (
            precision >= 0.99
            and recall >= 0.90
            and exclusion >= 0.95
            and detection >= 0.95
            and (best_rank is None or rank > best_rank)
        ):
            best_rank = rank
            best = {
                "free_probability_minimum": values[0],
                "occupied_probability_maximum": values[1],
                "unknown_probability_maximum": values[2],
                "occupied_detection_minimum": values[3],
                "useful_free_recall": recall,
                "admitted_free_precision": precision,
                "obstacle_exclusion_recall_within_2m": exclusion,
                "obstacle_detection_recall_within_2m": detection,
                "rank": list(rank),
            }
    if best is None:
        raise ValueError("no fixed calibration threshold passes")
    return best


def pre_g2_candidate_metadata(
    *,
    model_config: Mapping[str, Any],
    evaluation_state_manifest: Sequence[Mapping[str, Any]],
    evaluation_state_sha256: str,
    deployment_state_manifest: Sequence[Mapping[str, Any]],
    deployment_state_sha256: str,
    selection: Mapping[str, Any],
    calibration: Mapping[str, Any],
    primitive_vocabulary: Sequence[str],
    commanded_delta_table: Sequence[Sequence[float]],
    training_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_names = [item.get("name") for item in evaluation_state_manifest]
    deployment_names = [item.get("name") for item in deployment_state_manifest]
    required_prefixes = ("target_encoder.", "target_bev_decoder.", "predictor.")
    if (
        type(model_config) is not dict
        or not model_config
        or not is_sha256(evaluation_state_sha256)
        or not is_sha256(deployment_state_sha256)
        or any(not any(str(name).startswith(prefix) for name in evaluation_names) for prefix in required_prefixes)
        or any(
            not str(name).startswith(("encoder.", "bev_decoder.", "evidence_head."))
            for name in deployment_names
        )
        or any(name not in evaluation_names for name in deployment_names)
        or type(selection) is not dict
        or selection.get("selected_arm") != "promoted_jepa"
        or type(calibration) is not dict
        or calibration.get("arm") != "promoted_jepa"
        or len(primitive_vocabulary) != 9
        or len(commanded_delta_table) != 9
        or type(training_snapshot) is not dict
    ):
        raise ValueError("pre-G2 candidate state or provenance is incomplete")
    return {
        "schema": PRE_G2_CHECKPOINT_SCHEMA,
        "lifecycle_stage": "development_selected_and_calibrated_pending_g2",
        "checkpoint_kind": "pre_g2_candidate",
        "model_config": dict(model_config),
        "evaluation_state_manifest": list(evaluation_state_manifest),
        "evaluation_state_sha256": evaluation_state_sha256,
        "deployment_state_manifest": list(deployment_state_manifest),
        "deployment_state_sha256": deployment_state_sha256,
        "required_evaluation_state_prefixes": list(required_prefixes),
        "selection": dict(selection),
        "calibration": dict(calibration),
        "primitive_vocabulary": list(primitive_vocabulary),
        "commanded_delta_table": [list(row) for row in commanded_delta_table],
        "training_snapshot": dict(training_snapshot),
        "development_only": True,
        **PRE_G2_DENIALS,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "current_source_bindings",
    "evaluate_checkpoint_candidate",
    "is_sha256",
    "learning_rate",
    "parse_canonical_json",
    "pre_g2_candidate_metadata",
    "safe_relative_path",
    "schedule_core",
    "science_contract",
    "select_calibration_threshold",
    "select_promoted_checkpoint",
    "threshold_grid",
    "validate_authorization",
    "validate_binding",
    "validate_raw_audit",
    "validate_raw_manifest",
    "validate_review",
    "validate_schedule_indices",
    "with_content_sha256",
]

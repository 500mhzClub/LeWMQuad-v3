"""Torch-free execution contract for the dynamic-Cartesian N32 diagnostic.

This module performs no file I/O.  It owns deterministic identities and pure
decisions shared by the GPU runner and the validation-only finalizer.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


EXECUTION_BINDING_SHA256 = (
    "42687e80a16fb424be47d49782699bbc3ed549d7826a0ce6e78e92aa37188e1e"
)
PREOUTPUT_AMENDMENT_SHA256 = (
    "b1c226c71c60258382401146e4c7591e5491b94297cdf59395e10556857bed98"
)
ATTEMPT_CONTROL_AMENDMENT_SHA256 = (
    "1ecdfe8474e8be086db579515794d5411f00b2561a955939a41188e73156c875"
)
RESULT_SCHEMA = "lewm_go2_dynamic_cartesian_n32_result_v1"
SMOKE_RESULT_SCHEMA = "lewm_go2_dynamic_cartesian_n32_smoke_result_v1"
IMPLEMENTATION_MANIFEST_SCHEMA = (
    "lewm_go2_dynamic_cartesian_n32_implementation_manifest_v1"
)
STAGE_SCHEMA = "lewm_go2_dynamic_cartesian_n32_stage_v1"
FIT_GATE_SCHEMA = "lewm_go2_dynamic_cartesian_n32_fit_gate_v1"
TERMINAL_FIT_SCHEMA = "lewm_go2_dynamic_cartesian_n32_terminal_fit_gate_v1"
HOLDOUT_CHECK_SCHEMA = "lewm_go2_dynamic_cartesian_n32_holdout_checks_v1"
SEED_DECISION_SCHEMA = "lewm_go2_dynamic_cartesian_n32_seed_decision_v1"
SEED_PAIR_SCHEMA = "lewm_go2_dynamic_cartesian_n32_seed_pair_decision_v1"
PANEL_REPORT_SCHEMA = "lewm_go2_dynamic_cartesian_n32_panel_report_v1"
REFERENCE_SCHEMA = "lewm_go2_categorical_radial_n32_patch7_reference_v1"
ACCESS_LEDGER_SCHEMA = "lewm_go2_dynamic_cartesian_n32_access_ledger_v1"
ATTEMPT_MARKER_SCHEMA = "lewm_go2_dynamic_cartesian_n32_attempt_marker_v1"

EXPECTED_SEEDS = (20260710, 20260711)
FRAME_COUNT = 320
BATCH_SIZE = 4
EVALUATION_INTERVAL = 100
GRADIENT_CLIP = 1.0
CONDITIONS = (
    "correct_rgb",
    "role_global_shuffled_rgb",
    "same_scene_wrong_view_rgb",
)
CLASS_NAMES = ("unknown", "free", "occupied")
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
HOLDOUT_PANELS = ("same_scene_holdout", "cross_scene_holdout")
BRANCH_CONFIGS = {
    "production_faithful": {
        "updates": 2000,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
    },
    "ceiling_optimizer": {
        "updates": 5000,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
    },
}
MODEL_CONFIG = {
    "image_size": 112,
    "patch_size": 7,
    "encoder_dim": 192,
    "encoder_depth": 6,
    "encoder_heads": 6,
    "encoder_mlp_ratio": 4,
    "bev_dim": 64,
    "bev_size": [64, 64],
    "forward_range_m": [-0.95, 5.35],
    "left_range_m": [-3.15, 3.15],
    "action_dim": 9,
    "bev_attention_heads": 4,
    "bev_lift_type": "dynamic_projective_cell_square_attention_v1",
    "projective_horizontal_fov_deg": 78.323,
    "projective_vertical_fov_deg": 62.8370386364,
    "projective_camera_xyz_body_m": [0.326, 0.0, 0.043],
    "projective_camera_rpy_body_rad": [0.0, 0.0, 0.0],
    "projective_near_m": 0.05,
    "projective_vertical_anchor_z_body_m": [-0.333, -0.133, 0.067, 0.267, 0.467],
    "projective_output_cell_size_m": 0.1,
    "projective_attention_sigma_tokens": 2.0,
    "projective_attention_bias_floor": -6.0,
    "predictor_hidden_dim": 128,
    "target_ema_momentum": 0.996,
    "jepa_weight": 0.0,
    "occupancy_weight": 2.0,
    "equivariance_weight": 0.0,
    "action_contrast_weight": 0.0,
    "action_margin_fraction": 0.1,
    "variance_weight": 0.0,
    "variance_target_std": 0.5,
}
OPTIMIZER_CONFIGS = {
    branch: {
        "name": "AdamW",
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "gradient_clip": GRADIENT_CLIP,
        "constant_learning_rate": True,
    }
    for branch, config in BRANCH_CONFIGS.items()
}
OBJECTIVE_CONTRACT = {
    "entrypoint": "occupancy_logits",
    "loss": "direct_equal_capacity_hierarchical",
    "occupancy_weight_stored_but_not_multiplied": 2.0,
    "jepa_weight": 0.0,
    "equivariance_weight": 0.0,
    "action_contrast_weight": 0.0,
    "variance_weight": 0.0,
}
PREPROCESSING_CONTRACT = {
    "input_rgb_size": [112, 112],
    "resize": "PIL.Image.Resampling.BILINEAR",
    "rgb_scale": "uint8_div_255_float32",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "model_dtype": "float32",
    "metric_accumulator_dtype": "float64",
    "autocast_amp_compile_quantization": False,
}
CONTROL_CONTRACT = {
    "conditions": list(CONDITIONS),
    "target_batch_size": 4,
    "combined_model_batch_size": 12,
    "wrong_rgb_uses_target_attitude": True,
    "one_role_global_permutation_per_complete_panel": True,
    "one_same_scene_permutation_per_complete_panel": True,
    "family_slice_uses_target_family": True,
}
PROJECTIVE_QUERY_SUPPORT = {
    "schema": "lewm_projective_query_support_v1",
    "lift_type": "dynamic_projective_cell_square_attention_v1",
    "support_geometry": "output_cell_center_plus_four_corners_v1",
    "support_frame": "base_forward_left_offsets_from_output_cell_center",
    "output_cell_size_m": 0.1,
    "output_cell_half_extent_m": 0.05,
    "horizontal_offsets_body_m": [
        [0.0, 0.0],
        [-0.05, -0.05],
        [-0.05, 0.05],
        [0.05, -0.05],
        [0.05, 0.05],
    ],
    "support_point_count": 5,
    "uses_body_footprint": False,
    "attention_bias_aggregation": "minimum_normalized_image_token_distance_over_output_cell_support_and_vertical_anchors_v1",
    "query_visibility_aggregation": "any_output_cell_support_and_vertical_anchor_visible_v1",
    "physical_aggregation_contract": {
        "schema": "lewm_observable_physical_aggregation_v1",
        "contract_sha256": "db288979e7c389df2c4ca846f3309e395bcb6ec7bcf40cb8db6a3107f7e9f717",
        "output_cell_size_m": 0.1,
    },
    "contract_sha256": "316e2e48071ec0aea5541482c3f758ef0b3c27a167b7377103c43ecabbb04899",
}
EVENT_FIELDS = (
    "image_requests",
    "target_requests",
    "attitude_requests",
    "image_decode_events",
    "label_shard_npz_open_events",
    "model_calls",
    "model_output_frames",
    "model_attitude_frames",
)
PATCH7_FINAL_STATE_SHA256 = (
    "fba4e91b333d57a813fb94edb13b215064d03da2830aae9d0ae4b34685cd38c1"
)
PATCH7_REFERENCE_SHA256 = (
    "705f8c26ba4fa4f2d6838bb4ba04265c3137621384877de9653f369401b64d97"
)
REFERENCE_MACRO_ASSERTIONS = {
    "same_scene_holdout": {
        "hierarchical_nll": 0.3219876256599372,
        "far_free_recall": 0.46708481911812594,
    },
    "cross_scene_holdout": {
        "hierarchical_nll": 0.4054638461731662,
        "far_free_recall": 0.4665871805991353,
    },
}
REFERENCE_OUTPUT_MACROS = {
    "same_scene_holdout": {
        "hierarchical_nll": 0.3219876256599372,
        "far_free_recall": 0.46708481911812594,
    },
    "cross_scene_holdout": {
        "hierarchical_nll": 0.40546384617316633,
        "far_free_recall": 0.4665871805991353,
    },
}
PANEL_JOIN_CONTRACT = {
    "all_train_role": True,
    "frame_count": 960,
    "global_rows_sha256": "a3071ae9e1bac7fd81ed4af58291bee58923ef37da1b7db33e6c5c0f3b32915d",
    "panel_global_rows_sha256": {
        "cross_scene_holdout": "2fc332fbe7aba044c3ea163684d8ffe1b321a7fa7486ccfd1531e186ff53f37e",
        "fit": "b3e91a1b991017274c5485274c35aa0987e858e2dd017e93758e0ba361202f06",
        "same_scene_holdout": "4b698e6ccf15e4b863b7d518fe503fdbb0fb86422c951149155797336910dd45",
    },
    "row_identities_sha256": "701ebbd7f545c29f21f5f9f81d67a40a306b7708b580091d49e56161389458e6",
    "transition_count": 480,
}
CONTROL_PERMUTATION_SHA256 = {
    20260710: {
        "fit": {
            "role_global_shuffle": "bd2a686ea4353f7b2585fcf564059f7846f9f3b7fdbd101b86e31b531d25c013",
            "same_scene_wrong_view": "e83f1232f4bee96737361d7eb7428ea8df234ec722b2b012b4584937ef55cfdb",
        },
        "same_scene_holdout": {
            "role_global_shuffle": "3ca78afe861a2b9d251ced5061ad927feeb0362c384be4f38e3ed3cc96edc3fb",
            "same_scene_wrong_view": "4c559217966cf42e1781b2259bdca7c15c3bba17374e2adf27c46314e69e6fd9",
        },
        "cross_scene_holdout": {
            "role_global_shuffle": "b2d10444c977992d73c20310b683e7436bd66e0276862e3292ed1e94c34c0a8f",
            "same_scene_wrong_view": "18cdb830dc2d7f58c07dfa72e10ec5b84c170560226bebfca37d75e4a2bce9a8",
        },
    },
    20260711: {
        "fit": {
            "role_global_shuffle": "b6567109fc79aca96b7c7e0389d34d7b81075ebcdf6180b050824b09c9c1542d",
            "same_scene_wrong_view": "22b42ce8a380fa509aa86b54b431328d5520e60f4d81b533435b4e0c72551ce4",
        },
        "same_scene_holdout": {
            "role_global_shuffle": "94be2e7eba5a76c2dd74cf71690a94f96c1193c2a90c4c162fa22dff8e16ab45",
            "same_scene_wrong_view": "c38122c7fbc567768cd88c102301167caf7ea2bba916755f65a7905af1fd247a",
        },
        "cross_scene_holdout": {
            "role_global_shuffle": "9250d7acc35bb8744b424461793cefdd8eb7f0ff0a8436c6d3acc13f4e69c66d",
            "same_scene_wrong_view": "37f157a159797c5c266576cd2ea4b50bebabfac7531bdd54a7cdcc1b223cf8e5",
        },
    },
}

REPOSITORY_ROOT = "/home/andrewknowles/Workspace/LeWMQuad-v3"
IMPLEMENTATION_SOURCE_PATHS = {
    "attempt_control_amendment": f"{REPOSITORY_ROOT}/docs/lewm_go2_dynamic_cartesian_n32_v1_attempt_control_amendment_2026-07-11.md",
    "binding": f"{REPOSITORY_ROOT}/docs/lewm_go2_dynamic_cartesian_n32_v1_binding_2026-07-11.md",
    "benchmarks_package": f"{REPOSITORY_ROOT}/lewm/benchmarks/__init__.py",
    "categorical_n32_metrics": f"{REPOSITORY_ROOT}/lewm/benchmarks/go2_categorical_radial_n32.py",
    "counterfactual": f"{REPOSITORY_ROOT}/lewm/benchmarks/counterfactual.py",
    "datasets_package": f"{REPOSITORY_ROOT}/lewm/datasets/__init__.py",
    "dynamic_geometry": f"{REPOSITORY_ROOT}/lewm/benchmarks/go2_dynamic_cell_square_projection.py",
    "encoder": f"{REPOSITORY_ROOT}/lewm/models/encoders.py",
    "finalizer": f"{REPOSITORY_ROOT}/scripts/finalize_go2_dynamic_cartesian_n32.py",
    "finalizer_test": f"{REPOSITORY_ROOT}/lewm/tests/test_finalize_go2_dynamic_cartesian_n32.py",
    "lewm_package": f"{REPOSITORY_ROOT}/lewm/__init__.py",
    "manifest_preparer": f"{REPOSITORY_ROOT}/scripts/prepare_go2_dynamic_cartesian_n32_implementation.py",
    "manifest_preparer_test": f"{REPOSITORY_ROOT}/lewm/tests/test_prepare_go2_dynamic_cartesian_n32_implementation.py",
    "model": f"{REPOSITORY_ROOT}/lewm/models/egomotion_bev_jepa.py",
    "models_package": f"{REPOSITORY_ROOT}/lewm/models/__init__.py",
    "models_lewm": f"{REPOSITORY_ROOT}/lewm/models/lewm.py",
    "phase2d_spatial_lewm": f"{REPOSITORY_ROOT}/lewm/models/phase2d_spatial_lewm.py",
    "parity_report": f"{REPOSITORY_ROOT}/docs/lewm_go2_dynamic_cartesian_fit_panel_parity_result_2026-07-11.md",
    "parity_runner": f"{REPOSITORY_ROOT}/scripts/audit_go2_dynamic_cartesian_fit_panel_parity.py",
    "physical_metrics": f"{REPOSITORY_ROOT}/lewm/benchmarks/go2_physical_micro_overfit.py",
    "physical_spatial_metrics": f"{REPOSITORY_ROOT}/lewm/benchmarks/go2_physical_spatial_grounding.py",
    "preoutput_amendment": f"{REPOSITORY_ROOT}/docs/lewm_go2_dynamic_cartesian_n32_v1_preoutput_amendment_2026-07-11.md",
    "predictor": f"{REPOSITORY_ROOT}/lewm/models/predictor.py",
    "primitive_affordance": f"{REPOSITORY_ROOT}/lewm/models/primitive_affordance.py",
    "pure_contract": f"{REPOSITORY_ROOT}/lewm/benchmarks/go2_dynamic_cartesian_n32.py",
    "pure_contract_test": f"{REPOSITORY_ROOT}/lewm/tests/test_go2_dynamic_cartesian_n32.py",
    "runner": f"{REPOSITORY_ROOT}/scripts/run_go2_dynamic_cartesian_n32.py",
    "runner_test": f"{REPOSITORY_ROOT}/lewm/tests/test_run_go2_dynamic_cartesian_n32.py",
    "sigreg": f"{REPOSITORY_ROOT}/lewm/models/sigreg.py",
    "sidecar_library": f"{REPOSITORY_ROOT}/lewm/datasets/go2_attitude_sidecar.py",
    "source_action_utility": f"{REPOSITORY_ROOT}/lewm/models/source_action_utility.py",
    "spatial_lewm": f"{REPOSITORY_ROOT}/lewm/models/spatial_lewm.py",
    "spatial_predictor": f"{REPOSITORY_ROOT}/lewm/models/spatial_predictor.py",
}
ATTEMPT_MARKER_PATHS = {
    seed: f"{REPOSITORY_ROOT}/.generated/go2_dynamic_cartesian_n32/v1/seed_{seed}_attempt.json"
    for seed in EXPECTED_SEEDS
}
INPUT_BINDINGS = {
    "physical_dataset_manifest": {
        "path": f"{REPOSITORY_ROOT}/.generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json",
        "sha256": "ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180",
    },
    "panel": {
        "path": f"{REPOSITORY_ROOT}/.generated/go2_physical_micro_overfit/patch7_v1/panel.json",
        "sha256": "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c",
        "content_sha256": "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f",
        "rows_sha256": {
            "fit": "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d",
            "same_scene_holdout": "d32713086c042d20f94825aa362c27a07bef6fd0e0cce0aa5846bb67bf8dc465",
            "cross_scene_holdout": "3565f7f7844f3aeee28b0433aa6dc77d553a9ebb831cf9af20b6d392c5416817",
        },
    },
    "sidecar_manifest": {
        "path": f"{REPOSITORY_ROOT}/.generated/go2_attitude_sidecar/dynamic_cartesian_v1/manifest.json",
        "sha256": "6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529",
        "content_sha256": "6f1ef7d9ac0c55a42182c3e2c75909f00ab37fffa460aadb549d5cd60d278c1a",
    },
    "sidecar_train": {
        "path": f"{REPOSITORY_ROOT}/.generated/go2_attitude_sidecar/dynamic_cartesian_v1/train.jsonl",
        "sha256": "6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6",
        "content_sha256": "137f1286e85fbd3e4b45d1c9fb0337255ac735508d6ead57cd816e5134725fa2",
        "row_count": 4262,
    },
    "static_patch7_comparator": {
        "path": f"{REPOSITORY_ROOT}/.generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json",
        "sha256": "6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c",
        "content_sha256": "32d848d3df68e670ddb4cc24436981f62a1aa5562b89e6d6719ecb113f66b749",
        "final_state_sha256": PATCH7_FINAL_STATE_SHA256,
        "extracted_reference_sha256": PATCH7_REFERENCE_SHA256,
    },
    "fit_projection_parity": {
        "path": f"{REPOSITORY_ROOT}/.generated/go2_dynamic_cartesian_fit_panel_parity/v1/result.json",
        "sha256": "72d21aaf5e923126dd3a5022b0ea9775340877a00f40aa22845b244886fde70b",
        "content_sha256": "3729a3fcd61b523d744c476da89fb2f638593145055b52bc96035bb30c3f3cea",
    },
}
RESOURCE_POLICY = {
    "device": "cuda:0",
    "device_name_contains": "R9700",
    "hip_visible_devices": "0",
    "hsa_override_gfx_version_unset": True,
    "igpu": "forbidden",
    "minimum_device_memory_bytes": 16 * 1024**3,
    "source_workers": 6,
    "native_threads_per_worker": 1,
    "thread_environment": {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    },
    "query_chunking": False,
    "batch_size_frames": BATCH_SIZE,
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def validate_seed(seed: object) -> int:
    if type(seed) is not int or seed not in EXPECTED_SEEDS:
        raise ValueError("N32 seed must be exactly 20260710 or 20260711")
    return seed


def authoritative_runner_invocation(
    *,
    seed: int,
    implementation_manifest_file_sha256: str,
    primary_file_sha256: str | None = None,
    primary_attempt_marker_file_sha256: str | None = None,
) -> list[str]:
    seed = validate_seed(seed)
    if not _is_sha256(implementation_manifest_file_sha256):
        raise ValueError("authoritative invocation manifest SHA-256 is malformed")
    primary = (primary_file_sha256, primary_attempt_marker_file_sha256)
    if seed == EXPECTED_SEEDS[0] and any(item is not None for item in primary):
        raise ValueError("first authoritative invocation rejects primary evidence")
    if seed == EXPECTED_SEEDS[1] and not all(_is_sha256(item) for item in primary):
        raise ValueError("replication invocation requires exact primary evidence")
    invocation = [
        COMMAND_CONTRACT["runner"],
        "--output",
        COMMAND_CONTRACT["canonical_outputs"][str(seed)],
        "--implementation-manifest",
        (
            f"{REPOSITORY_ROOT}/docs/"
            "lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json"
        ),
        "--expected-implementation-manifest-sha256",
        implementation_manifest_file_sha256,
        "--device",
        "cuda:0",
        "--seed",
        str(seed),
    ]
    if seed == EXPECTED_SEEDS[1]:
        invocation.extend(
            [
                "--seed-20260710-result",
                COMMAND_CONTRACT["canonical_outputs"][str(EXPECTED_SEEDS[0])],
                "--expected-seed-20260710-sha256",
                str(primary_file_sha256),
                "--seed-20260710-attempt-marker",
                ATTEMPT_MARKER_PATHS[EXPECTED_SEEDS[0]],
                "--expected-seed-20260710-attempt-marker-sha256",
                str(primary_attempt_marker_file_sha256),
            ]
        )
    return invocation


def deterministic_minibatch_schedule(
    *, seed: int, branch: str, updates: int | None = None
) -> list[list[int]]:
    """Return the frozen pure hash-ranked frame schedule.

    Each epoch is an independent permutation.  The rank key includes the seed,
    branch-independent namespace, epoch and frame, so the ceiling schedule has
    the faithful schedule as an exact prefix for a given seed.
    """

    seed = validate_seed(seed)
    if branch not in BRANCH_CONFIGS:
        raise ValueError("unknown N32 optimizer branch")
    expected_updates = int(BRANCH_CONFIGS[branch]["updates"])
    if updates is None:
        updates = expected_updates
    if type(updates) is not int or updates <= 0 or updates > expected_updates:
        raise ValueError("N32 updates are outside the frozen branch budget")
    schedule: list[list[int]] = []
    epoch = 0
    while len(schedule) < updates:
        order = sorted(
            range(FRAME_COUNT),
            key=lambda frame: hashlib.sha256(
                f"dynamic-cartesian-n32-v1\0{seed}\0{epoch}\0{frame}".encode()
            ).digest(),
        )
        for offset in range(0, FRAME_COUNT, BATCH_SIZE):
            schedule.append(order[offset : offset + BATCH_SIZE])
            if len(schedule) == updates:
                return schedule
        epoch += 1
    return schedule


SCHEDULE_SHA256 = {
    (seed, branch): canonical_json_sha256(
        deterministic_minibatch_schedule(seed=seed, branch=branch)
    )
    for seed in EXPECTED_SEEDS
    for branch in BRANCH_CONFIGS
}
SMOKE_SCHEDULE_SHA256 = {
    (seed, branch): canonical_json_sha256(
        deterministic_minibatch_schedule(seed=seed, branch=branch, updates=3)
    )
    for seed in EXPECTED_SEEDS
    for branch in BRANCH_CONFIGS
}
SCHEDULE_CONTRACT = {
    "algorithm": "sha256_ranked_epoch_permutation_v1",
    "namespace": "dynamic-cartesian-n32-v1",
    "frame_count": FRAME_COUNT,
    "batch_size": BATCH_SIZE,
    "authoritative_sha256": {
        f"{seed}:{branch}": SCHEDULE_SHA256[(seed, branch)]
        for seed in EXPECTED_SEEDS
        for branch in BRANCH_CONFIGS
    },
    "smoke_updates": 3,
    "smoke_sha256": {
        f"{seed}:{branch}": SMOKE_SCHEDULE_SHA256[(seed, branch)]
        for seed in EXPECTED_SEEDS
        for branch in BRANCH_CONFIGS
    },
}
COMMAND_CONTRACT = {
    "environment_prefix": [
        "env",
        "-u",
        "HSA_OVERRIDE_GFX_VERSION",
        "HIP_VISIBLE_DEVICES=0",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "/home/andrewknowles/TinyQuadJEPA/bin/python",
    ],
    "runner": f"{REPOSITORY_ROOT}/scripts/run_go2_dynamic_cartesian_n32.py",
    "authoritative_argument_order": [
        "--output",
        "--implementation-manifest",
        "--expected-implementation-manifest-sha256",
        "--device",
        "--seed",
        "seed-20260711-primary-arguments-if-required",
    ],
    "implementation_manifest_preparation": {
        "script": f"{REPOSITORY_ROOT}/scripts/prepare_go2_dynamic_cartesian_n32_implementation.py",
        "output": f"{REPOSITORY_ROOT}/docs/lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json",
        "device": "cuda:0",
        "registered_seed_order": list(EXPECTED_SEEDS),
        "runs_frozen_tests_before_model_construction": True,
        "derives_state_hashes_without_forward_or_model_output": True,
        "immutable_no_replace_publication": True,
    },
    "registered_seed_order": list(EXPECTED_SEEDS),
    "both_seeds_always_run_once": True,
    "canonical_outputs": {
        str(seed): f"{REPOSITORY_ROOT}/.generated/go2_dynamic_cartesian_n32/v1/seed_{seed}_result.json"
        for seed in EXPECTED_SEEDS
    },
    "canonical_attempt_markers": {
        str(seed): ATTEMPT_MARKER_PATHS[seed] for seed in EXPECTED_SEEDS
    },
    "seed_20260710_rejects_primary_arguments": True,
    "seed_20260711_requires_primary_path_and_external_file_sha256": True,
    "seed_20260711_requires_primary_attempt_marker_and_external_file_sha256": True,
    "immutable_no_replace_publication": True,
    "attempt_marker_created_before_payload_access": True,
    "attempt_marker_immutable_no_replace": True,
    "attempt_marker_consumes_seed_after_crash": True,
    "non_authoritative_smoke_creates_attempt_marker": False,
}
IMPLEMENTATION_TEST_COMMAND = (
    "env PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3:"
    "/home/andrewknowles/Workspace/LeWMQuad-v3/lewm_worlds:"
    "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages "
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "NUMEXPR_NUM_THREADS=1 /usr/bin/pytest -q "
    "lewm/tests/test_go2_dynamic_cartesian_n32.py "
    "lewm/tests/test_run_go2_dynamic_cartesian_n32.py "
    "lewm/tests/test_finalize_go2_dynamic_cartesian_n32.py"
    " lewm/tests/test_prepare_go2_dynamic_cartesian_n32_implementation.py"
)
IMPLEMENTATION_TEST_PASSED = 79

DISTANCE_BIN_NAMES = (
    "0.0_to_0.5",
    "0.5_to_1.0",
    "1.0_to_2.0",
    "2.0_to_3.0",
    "3.0_plus",
)
UNKNOWN_KNOWN_WEIGHTS = (0.13853880763053894, 1.8614611625671387)
FREE_OCCUPIED_WEIGHTS = (0.21841949224472046, 1.7815805673599243)
METRIC_FIELDS = {
    "raw_joint_nll",
    "raw_joint_accuracy",
    "raw_hierarchical_balanced_nll",
    "raw_unknown_known_weighted_nll",
    "raw_known_free_occupied_weighted_nll",
    "raw_known_free_occupied_nll",
    "raw_known_free_occupied_accuracy",
    "cell_count",
    "known_cell_count",
    "joint_confusion",
    "unknown_known_confusion",
    "free_occupied_confusion",
    "unknown_known_balanced_accuracy",
    "free_occupied_balanced_accuracy",
    "class_recall",
    "class_precision",
    "free_average_precision",
    "occupied_average_precision",
    "posterior_quantiles_by_truth_class",
    "distance_free_recall",
    "distance_free_support",
}


def validate_minibatch_schedule(
    schedule: object, *, seed: int, branch: str
) -> str:
    seed = validate_seed(seed)
    if branch not in BRANCH_CONFIGS:
        raise ValueError("unknown N32 optimizer branch")
    updates = int(BRANCH_CONFIGS[branch]["updates"])
    if not isinstance(schedule, list) or len(schedule) != updates:
        raise ValueError("N32 minibatch schedule has the wrong length")
    for batch in schedule:
        if (
            not isinstance(batch, list)
            or len(batch) != BATCH_SIZE
            or any(type(index) is not int for index in batch)
            or len(set(batch)) != BATCH_SIZE
            or any(index < 0 or index >= FRAME_COUNT for index in batch)
        ):
            raise ValueError("N32 minibatch schedule contains an invalid batch")
    for start in range(0, updates - 79, 80):
        epoch = [index for batch in schedule[start : start + 80] for index in batch]
        if sorted(epoch) != list(range(FRAME_COUNT)):
            raise ValueError("N32 minibatch schedule contains a non-permutation epoch")
    digest = canonical_json_sha256(schedule)
    if digest != SCHEDULE_SHA256[(seed, branch)]:
        raise ValueError("N32 exact seeded minibatch schedule drift")
    return digest


def _required_metric(metrics: Mapping[str, Any], *path: str) -> float:
    value: Any = metrics
    for name in path:
        if not isinstance(value, Mapping) or name not in value:
            raise ValueError(f"metrics lack {'/'.join(path)}")
        value = value[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics contain invalid {'/'.join(path)}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metrics contain invalid {'/'.join(path)}")
    return result


def _same_float(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative JSON integer")
    return value


def _metric_number(
    value: object,
    name: str,
    *,
    required: bool,
    upper: float | None = None,
) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required by its support")
        return None
    if not required:
        raise ValueError(f"{name} must be null when its support is zero")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (
        upper is not None and result > upper
    ):
        suffix = " in [0, 1]" if upper == 1.0 else " and nonnegative"
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _confusion_matrix(value: object, size: int, name: str) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must be an exact {size}x{size} matrix")
    matrix: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{name} must be an exact {size}x{size} matrix")
        matrix.append(
            [_nonnegative_int(item, f"{name} entry") for item in row]
        )
    return matrix


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _balanced_accuracy(matrix: Sequence[Sequence[int]]) -> float | None:
    recalls = [
        float(row[index]) / sum(row)
        for index, row in enumerate(matrix)
        if sum(row) > 0
    ]
    return None if not recalls else sum(recalls) / len(recalls)


def _require_expected_float(
    observed: float | None, expected: float | None, name: str
) -> None:
    if observed is None or expected is None:
        if observed is not expected:
            raise ValueError(f"{name} null/support arithmetic mismatch")
    elif not _same_float(observed, expected):
        raise ValueError(f"{name} arithmetic mismatch")


def _validate_metric_record(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != METRIC_FIELDS:
        raise ValueError(f"{name} metric schema mismatch")
    cell_count = _nonnegative_int(value["cell_count"], f"{name}/cell_count")
    known_count = _nonnegative_int(
        value["known_cell_count"], f"{name}/known_cell_count"
    )
    joint = _confusion_matrix(value["joint_confusion"], 3, f"{name}/joint")
    unknown_known = _confusion_matrix(
        value["unknown_known_confusion"], 2, f"{name}/unknown_known"
    )
    free_occupied = _confusion_matrix(
        value["free_occupied_confusion"], 2, f"{name}/free_occupied"
    )
    truth_support = [sum(row) for row in joint]
    if sum(truth_support) != cell_count or sum(truth_support[1:]) != known_count:
        raise ValueError(f"{name} joint confusion/count arithmetic mismatch")
    if [sum(row) for row in unknown_known] != [truth_support[0], known_count]:
        raise ValueError(f"{name} unknown/known truth support mismatch")
    if [sum(row) for row in free_occupied] != truth_support[1:]:
        raise ValueError(f"{name} free/occupied truth support mismatch")

    joint_nll = _metric_number(
        value["raw_joint_nll"], f"{name}/raw_joint_nll", required=cell_count > 0
    )
    del joint_nll
    joint_accuracy = _metric_number(
        value["raw_joint_accuracy"],
        f"{name}/raw_joint_accuracy",
        required=cell_count > 0,
        upper=1.0,
    )
    _require_expected_float(
        joint_accuracy,
        _ratio(sum(joint[index][index] for index in range(3)), cell_count),
        f"{name}/raw_joint_accuracy",
    )
    uk_nll = _metric_number(
        value["raw_unknown_known_weighted_nll"],
        f"{name}/raw_unknown_known_weighted_nll",
        required=cell_count > 0,
    )
    fo_weighted_nll = _metric_number(
        value["raw_known_free_occupied_weighted_nll"],
        f"{name}/raw_known_free_occupied_weighted_nll",
        required=known_count > 0,
    )
    hierarchical_nll = _metric_number(
        value["raw_hierarchical_balanced_nll"],
        f"{name}/raw_hierarchical_balanced_nll",
        required=cell_count > 0 and known_count > 0,
    )
    _require_expected_float(
        hierarchical_nll,
        None
        if uk_nll is None or fo_weighted_nll is None
        else 0.5 * uk_nll + 0.5 * fo_weighted_nll,
        f"{name}/raw_hierarchical_balanced_nll",
    )
    _metric_number(
        value["raw_known_free_occupied_nll"],
        f"{name}/raw_known_free_occupied_nll",
        required=known_count > 0,
    )
    known_accuracy = _metric_number(
        value["raw_known_free_occupied_accuracy"],
        f"{name}/raw_known_free_occupied_accuracy",
        required=known_count > 0,
        upper=1.0,
    )
    _require_expected_float(
        known_accuracy,
        _ratio(free_occupied[0][0] + free_occupied[1][1], known_count),
        f"{name}/raw_known_free_occupied_accuracy",
    )
    uk_balanced = _metric_number(
        value["unknown_known_balanced_accuracy"],
        f"{name}/unknown_known_balanced_accuracy",
        required=cell_count > 0,
        upper=1.0,
    )
    fo_balanced = _metric_number(
        value["free_occupied_balanced_accuracy"],
        f"{name}/free_occupied_balanced_accuracy",
        required=known_count > 0,
        upper=1.0,
    )
    _require_expected_float(
        uk_balanced, _balanced_accuracy(unknown_known), f"{name}/UK balanced accuracy"
    )
    _require_expected_float(
        fo_balanced, _balanced_accuracy(free_occupied), f"{name}/FO balanced accuracy"
    )

    for field, denominator in (
        ("class_recall", lambda index: truth_support[index]),
        ("class_precision", lambda index: sum(row[index] for row in joint)),
    ):
        record = value[field]
        if not isinstance(record, Mapping) or set(record) != set(CLASS_NAMES):
            raise ValueError(f"{name}/{field} schema mismatch")
        for index, class_name in enumerate(CLASS_NAMES):
            support = denominator(index)
            observed = _metric_number(
                record[class_name],
                f"{name}/{field}/{class_name}",
                required=support > 0,
                upper=1.0,
            )
            _require_expected_float(
                observed,
                _ratio(joint[index][index], support),
                f"{name}/{field}/{class_name}",
            )

    for field, positive in (
        ("free_average_precision", truth_support[1]),
        ("occupied_average_precision", truth_support[2]),
    ):
        _metric_number(
            value[field],
            f"{name}/{field}",
            required=positive > 0 and cell_count - positive > 0,
            upper=1.0,
        )

    posterior = value["posterior_quantiles_by_truth_class"]
    if not isinstance(posterior, Mapping) or set(posterior) != set(CLASS_NAMES):
        raise ValueError(f"{name}/posterior quantile truth schema mismatch")
    for truth_index, truth_name in enumerate(CLASS_NAMES):
        predicted_record = posterior[truth_name]
        if not isinstance(predicted_record, Mapping) or set(predicted_record) != set(
            CLASS_NAMES
        ):
            raise ValueError(f"{name}/posterior quantile prediction schema mismatch")
        for predicted_name in CLASS_NAMES:
            quantiles = predicted_record[predicted_name]
            if truth_support[truth_index] == 0:
                if quantiles is not None:
                    raise ValueError(f"{name}/posterior quantiles require truth support")
                continue
            if not isinstance(quantiles, Mapping) or set(quantiles) != {
                "p05",
                "p50",
                "p95",
            }:
                raise ValueError(f"{name}/posterior quantile schema mismatch")
            ordered = [
                _metric_number(
                    quantiles[key],
                    f"{name}/posterior/{truth_name}/{predicted_name}/{key}",
                    required=True,
                    upper=1.0,
                )
                for key in ("p05", "p50", "p95")
            ]
            if not ordered[0] <= ordered[1] <= ordered[2]:
                raise ValueError(f"{name}/posterior quantiles are not monotonic")

    distance_support = value["distance_free_support"]
    distance_recall = value["distance_free_recall"]
    if not isinstance(distance_support, Mapping) or set(distance_support) != set(
        DISTANCE_BIN_NAMES
    ) or not isinstance(distance_recall, Mapping) or set(distance_recall) != set(
        DISTANCE_BIN_NAMES
    ):
        raise ValueError(f"{name}/distance-bin schema mismatch")
    support_total = 0
    for bin_name in DISTANCE_BIN_NAMES:
        support = _nonnegative_int(
            distance_support[bin_name], f"{name}/distance support/{bin_name}"
        )
        support_total += support
        recall = _metric_number(
            distance_recall[bin_name],
            f"{name}/distance recall/{bin_name}",
            required=support > 0,
            upper=1.0,
        )
        if recall is not None and not _same_float(
            recall * support, float(round(recall * support))
        ):
            raise ValueError(f"{name}/distance recall is not an integer ratio")
    if support_total != truth_support[1]:
        raise ValueError(f"{name}/distance free support does not cover truth-free cells")
    return value


def _conditions(
    report: Mapping[str, Any], *, name: str = "report"
) -> Mapping[str, Mapping[str, Any]]:
    conditions = report.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
        raise ValueError("N32 report conditions are incomplete")
    if not all(isinstance(conditions[name], Mapping) for name in CONDITIONS):
        raise ValueError("N32 report condition metrics are malformed")
    for condition in CONDITIONS:
        _validate_metric_record(
            conditions[condition], name=f"{name}/{condition}"
        )
    return conditions


def _sum_family_matrix(
    metrics: Sequence[Mapping[str, Any]], field: str, size: int
) -> list[list[int]]:
    return [
        [sum(int(value[field][row][column]) for value in metrics) for column in range(size)]
        for row in range(size)
    ]


def _weighted_family_value(
    metrics: Sequence[Mapping[str, Any]],
    field: str,
    denominator,
) -> float | None:
    weighted_sum = 0.0
    total = 0.0
    for value in metrics:
        weight = float(denominator(value))
        if weight == 0.0:
            if value[field] is not None:
                raise ValueError(f"family {field} exists without support")
            continue
        weighted_sum += float(value[field]) * weight
        total += weight
    return None if total == 0.0 else weighted_sum / total


def _validate_family_aggregation(report: Mapping[str, Any], *, name: str) -> None:
    aggregate_conditions = report["conditions"]
    families = report["families"]
    for condition in CONDITIONS:
        aggregate = aggregate_conditions[condition]
        family_metrics = [families[family]["conditions"][condition] for family in FAMILIES]
        if aggregate["cell_count"] != sum(
            value["cell_count"] for value in family_metrics
        ) or aggregate["known_cell_count"] != sum(
            value["known_cell_count"] for value in family_metrics
        ):
            raise ValueError(f"{name}/{condition} family count aggregation mismatch")
        for field, size in (
            ("joint_confusion", 3),
            ("unknown_known_confusion", 2),
            ("free_occupied_confusion", 2),
        ):
            if aggregate[field] != _sum_family_matrix(family_metrics, field, size):
                raise ValueError(
                    f"{name}/{condition} family {field} aggregation mismatch"
                )
        expected_distance_support = {
            bin_name: sum(
                value["distance_free_support"][bin_name] for value in family_metrics
            )
            for bin_name in DISTANCE_BIN_NAMES
        }
        if aggregate["distance_free_support"] != expected_distance_support:
            raise ValueError(
                f"{name}/{condition} family distance-support aggregation mismatch"
            )
        denominators = {
            "raw_joint_nll": lambda value: value["cell_count"],
            "raw_unknown_known_weighted_nll": lambda value: (
                sum(value["joint_confusion"][0]) * UNKNOWN_KNOWN_WEIGHTS[0]
                + value["known_cell_count"] * UNKNOWN_KNOWN_WEIGHTS[1]
            ),
            "raw_known_free_occupied_weighted_nll": lambda value: (
                sum(value["joint_confusion"][1]) * FREE_OCCUPIED_WEIGHTS[0]
                + sum(value["joint_confusion"][2]) * FREE_OCCUPIED_WEIGHTS[1]
            ),
            "raw_known_free_occupied_nll": lambda value: value[
                "known_cell_count"
            ],
        }
        for field, denominator in denominators.items():
            _require_expected_float(
                None if aggregate[field] is None else float(aggregate[field]),
                _weighted_family_value(family_metrics, field, denominator),
                f"{name}/{condition}/family aggregate/{field}",
            )
        for bin_name in DISTANCE_BIN_NAMES:
            expected_recall = (
                None
                if expected_distance_support[bin_name] == 0
                else sum(
                    float(value["distance_free_recall"][bin_name])
                    * value["distance_free_support"][bin_name]
                    for value in family_metrics
                    if value["distance_free_support"][bin_name] > 0
                )
                / expected_distance_support[bin_name]
            )
            _require_expected_float(
                (
                    None
                    if aggregate["distance_free_recall"][bin_name] is None
                    else float(aggregate["distance_free_recall"][bin_name])
                ),
                expected_recall,
                f"{name}/{condition}/family aggregate/distance/{bin_name}",
            )


def _gate_for_conditions(conditions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    correct = conditions["correct_rgb"]
    correct_nll = _required_metric(correct, "raw_hierarchical_balanced_nll")
    role_delta = _required_metric(
        conditions["role_global_shuffled_rgb"],
        "raw_hierarchical_balanced_nll",
    ) - correct_nll
    same_delta = _required_metric(
        conditions["same_scene_wrong_view_rgb"],
        "raw_hierarchical_balanced_nll",
    ) - correct_nll
    checks = {
        "raw_hierarchical_balanced_nll_le_0_03": correct_nll <= 0.03,
        "unknown_known_balanced_accuracy_ge_0_99": _required_metric(
            correct, "unknown_known_balanced_accuracy"
        ) >= 0.99,
        "free_occupied_balanced_accuracy_ge_0_99": _required_metric(
            correct, "free_occupied_balanced_accuracy"
        ) >= 0.99,
        **{
            f"{name}_recall_ge_0_98": _required_metric(
                correct, "class_recall", name
            ) >= 0.98
            for name in CLASS_NAMES
        },
        **{
            f"{name}_free_recall_ge_0_95": _required_metric(
                correct, "distance_free_recall", name
            ) >= 0.95
            for name in ("1.0_to_2.0", "2.0_to_3.0", "3.0_plus")
        },
        "cross_scene_shuffled_minus_correct_nll_ge_0_25": role_delta >= 0.25,
        "same_scene_wrong_view_minus_correct_nll_ge_0_25": same_delta >= 0.25,
    }
    return {
        "schema": "lewm_go2_physical_micro_overfit_fit_gate_v1",
        "passes": all(checks.values()),
        "checks": checks,
        "cross_scene_shuffled_minus_correct_raw_hierarchical_balanced_nll": role_delta,
        "same_scene_wrong_view_minus_correct_raw_hierarchical_balanced_nll": same_delta,
    }


def fit_panel_gate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _gate_for_conditions(_conditions(report))
    families = report.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError("N32 fit report must contain exactly five families")
    family_gates = {
        family: _gate_for_conditions(_conditions(families[family]))
        for family in FAMILIES
    }
    return {
        "schema": FIT_GATE_SCHEMA,
        "aggregate": aggregate,
        "families": family_gates,
        "family_order": list(FAMILIES),
        "requires_aggregate_and_all_families": True,
        "passes": bool(aggregate["passes"]) and all(
            bool(family_gates[family]["passes"]) for family in FAMILIES
        ),
    }


def terminal_fit_gate_summary(
    curve: Sequence[Mapping[str, Any]],
    max_steps: int,
    eval_interval: int = EVALUATION_INTERVAL,
) -> dict[str, Any]:
    if max_steps <= 0 or eval_interval <= 0 or max_steps % eval_interval:
        raise ValueError("N32 evaluation cadence does not divide its budget")
    expected = list(range(eval_interval, max_steps + 1, eval_interval))
    if [point.get("step") for point in curve] != expected or len(expected) < 3:
        raise ValueError("N32 fit curve cadence or fixed budget is incomplete")
    passes = []
    for point in curve:
        report = point.get("fit_panel", point.get("fit"))
        if not isinstance(report, Mapping):
            raise ValueError("N32 curve point lacks a fit panel")
        passes.append(bool(fit_panel_gate_report(report)["passes"]))
    first_single = next((step for step, passed in zip(expected, passes) if passed), None)
    run = 0
    first_three = None
    for step, passed in zip(expected, passes):
        run = run + 1 if passed else 0
        if run == 3 and first_three is None:
            first_three = step
    return {
        "schema": TERMINAL_FIT_SCHEMA,
        "maximum_steps": max_steps,
        "evaluation_interval": eval_interval,
        "evaluation_steps": expected,
        "evaluation_passes": passes,
        "terminal_evaluation_steps": expected[-3:],
        "terminal_evaluation_passes": passes[-3:],
        "requires_exact_final_three": True,
        "first_single_fit_gate_step": first_single,
        "first_three_consecutive_fit_gate_step": first_three,
        "passes": all(passes[-3:]),
    }


def _family_correct_metrics(panel: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    families = panel.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError("N32 panel must contain exactly five families")
    result = {}
    for family in FAMILIES:
        conditions = families[family].get("conditions") if isinstance(
            families[family], Mapping
        ) else None
        metrics = conditions.get("correct_rgb") if isinstance(conditions, Mapping) else None
        if not isinstance(metrics, Mapping):
            raise ValueError(f"N32 family lacks correct RGB metrics: {family}")
        result[family] = metrics
    return result


def _ordered_mean(values: Sequence[float]) -> float:
    if len(values) != len(FAMILIES):
        raise ValueError("N32 macro requires exactly five family values")
    return float(sum(float(value) for value in values) / len(FAMILIES))


def extract_faithful_patch7_family_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        value.get("schema") != "lewm_go2_physical_micro_overfit_result_v1"
        or declared != INPUT_BINDINGS["static_patch7_comparator"]["content_sha256"]
        or canonical_json_sha256(core) != declared
    ):
        raise ValueError("static patch7 comparator content mismatch")
    try:
        stage = value["stages"]["production_faithful"]["patch7_16x16"]
    except (KeyError, TypeError) as exc:
        raise ValueError("patch7 result lacks its faithful comparator") from exc
    if stage.get("final_state_sha256") != PATCH7_FINAL_STATE_SHA256:
        raise ValueError("faithful patch7 final-state SHA-256 mismatch")
    final_panels = stage.get("final_panels")
    if not isinstance(final_panels, Mapping):
        raise ValueError("faithful patch7 result lacks final panels")
    panels: dict[str, Any] = {}
    macros = {}
    for panel_name in HOLDOUT_PANELS:
        panel = final_panels.get(panel_name)
        if not isinstance(panel, Mapping):
            raise ValueError(f"faithful patch7 lacks {panel_name}")
        metrics = _family_correct_metrics(panel)
        nll = _ordered_mean([
            _required_metric(metrics[family], "raw_hierarchical_balanced_nll")
            for family in FAMILIES
        ])
        far = _ordered_mean([
            _required_metric(metrics[family], "distance_free_recall", "3.0_plus")
            for family in FAMILIES
        ])
        asserted = REFERENCE_MACRO_ASSERTIONS[panel_name]
        if not math.isclose(nll, asserted["hierarchical_nll"], rel_tol=0.0, abs_tol=5e-16) or not math.isclose(
            far, asserted["far_free_recall"], rel_tol=0.0, abs_tol=5e-16
        ):
            raise ValueError(f"faithful patch7 {panel_name} macro assertion failed")
        panels[panel_name] = panel
        macros[panel_name] = dict(REFERENCE_OUTPUT_MACROS[panel_name])
    support = value.get("post_selection_support_audit")
    if not isinstance(support, Mapping) or set(support) != {"fit", *HOLDOUT_PANELS}:
        raise ValueError("patch7 post-selection support audit is incomplete")
    result = {
        "schema": REFERENCE_SCHEMA,
        "source_stage": "production_faithful",
        "source_arm": "patch7_16x16",
        "final_state_sha256": PATCH7_FINAL_STATE_SHA256,
        "panels": panels,
        "macro_assertions": macros,
    }
    observed_reference_sha256 = canonical_json_sha256(result)
    if observed_reference_sha256 != PATCH7_REFERENCE_SHA256:
        raise ValueError(
            "extracted static patch7 reference identity mismatch: "
            f"{observed_reference_sha256}"
        )
    return result


def strict_patch7_holdout_checks(
    candidate_panel: Mapping[str, Any], reference_panel: Mapping[str, Any]
) -> dict[str, Any]:
    panel_name = candidate_panel.get("panel")
    if panel_name not in HOLDOUT_PANELS or reference_panel.get("panel", panel_name) != panel_name:
        raise ValueError("holdout panel name is invalid or mismatched")
    candidate = _family_correct_metrics(candidate_panel)
    reference = _family_correct_metrics(reference_panel)
    per_family = {}
    for family in FAMILIES:
        candidate_nll = _required_metric(candidate[family], "raw_hierarchical_balanced_nll")
        reference_nll = _required_metric(reference[family], "raw_hierarchical_balanced_nll")
        candidate_far = _required_metric(candidate[family], "distance_free_recall", "3.0_plus")
        reference_far = _required_metric(reference[family], "distance_free_recall", "3.0_plus")
        deltas = {
            name: _required_metric(candidate[family], "class_recall", name)
            - _required_metric(reference[family], "class_recall", name)
            for name in CLASS_NAMES
        }
        per_family[family] = {
            "candidate_hierarchical_nll": candidate_nll,
            "reference_hierarchical_nll": reference_nll,
            "candidate_far_free_recall": candidate_far,
            "reference_far_free_recall": reference_far,
            "candidate_minus_reference_class_recall": deltas,
            "strictly_lower_nll_and_higher_far_free": candidate_nll < reference_nll and candidate_far > reference_far,
        }
    candidate_macro = _ordered_mean([per_family[f]["candidate_hierarchical_nll"] for f in FAMILIES])
    reference_macro = _ordered_mean([per_family[f]["reference_hierarchical_nll"] for f in FAMILIES])
    ratio = None if reference_macro <= 0.0 else candidate_macro / reference_macro
    far_delta = _ordered_mean([
        per_family[f]["candidate_far_free_recall"] - per_family[f]["reference_far_free_recall"]
        for f in FAMILIES
    ])
    class_macro = {
        name: _ordered_mean([per_family[f]["candidate_minus_reference_class_recall"][name] for f in FAMILIES])
        for name in CLASS_NAMES
    }
    favorable_count = sum(bool(per_family[f]["strictly_lower_nll_and_higher_far_free"]) for f in FAMILIES)
    requirement = 5 if panel_name == "cross_scene_holdout" else 4
    checks = {
        "equal_weight_family_macro_nll_ratio_le_0_80": ratio is not None and ratio <= 0.80,
        "equal_weight_family_macro_far_free_delta_ge_0_10": far_delta >= 0.10,
        "every_macro_class_recall_delta_ge_neg_0_01": min(class_macro.values()) >= -0.01,
        "no_family_class_recall_delta_lt_neg_0_01": min(
            per_family[f]["candidate_minus_reference_class_recall"][name]
            for f in FAMILIES for name in CLASS_NAMES
        ) >= -0.01,
        f"strict_family_nll_and_far_improvement_ge_{requirement}_of_5": favorable_count >= requirement,
    }
    return {
        "schema": HOLDOUT_CHECK_SCHEMA,
        "panel": panel_name,
        "passes": all(checks.values()),
        "checks": checks,
        "family_order": list(FAMILIES),
        "family_macro_weighting": "equal_weight_across_five_families",
        "macro": {
            "candidate_hierarchical_nll": candidate_macro,
            "reference_hierarchical_nll": reference_macro,
            "candidate_minus_reference_far_free_recall": far_delta,
            "candidate_minus_reference_class_recall": class_macro,
        },
        "candidate_to_reference_macro_hierarchical_nll_ratio": ratio,
        "strictly_favorable_family_count": favorable_count,
        "strictly_favorable_family_requirement": requirement,
        "ties_count_as_failure": True,
        "per_family": per_family,
    }


categorical_holdout_checks = strict_patch7_holdout_checks


def _terminal_pass(stage: Mapping[str, Any], name: str) -> bool:
    terminal = stage.get("terminal_fit_gate", stage)
    if not isinstance(terminal, Mapping):
        raise ValueError(f"{name} lacks a terminal fit gate")
    return _strict_bool(terminal.get("passes"), f"{name} terminal passes")


def branch_access_decision(
    faithful: Mapping[str, Any], ceiling: Mapping[str, Any] | None
) -> dict[str, Any]:
    faithful_pass = _terminal_pass(faithful, "production_faithful")
    if faithful_pass and ceiling is not None:
        raise ValueError("ceiling is forbidden after a faithful fit pass")
    if not faithful_pass and ceiling is None:
        raise ValueError("ceiling is mandatory after a faithful fit failure")
    ceiling_pass = None if ceiling is None else _terminal_pass(ceiling, "ceiling_optimizer")
    qualifying = (
        "production_faithful" if faithful_pass else "ceiling_optimizer" if ceiling_pass else None
    )
    return {
        "ceiling_invoked": ceiling is not None,
        "qualifying_branch": qualifying,
        "holdouts_authorized": qualifying is not None,
    }


def per_seed_decision(
    faithful: Mapping[str, Any],
    ceiling: Mapping[str, Any] | None,
    holdouts: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    access = branch_access_decision(faithful, ceiling)
    if not access["holdouts_authorized"]:
        if holdouts not in (None, {}):
            raise ValueError("holdouts are forbidden when both fit branches fail")
        holdout_passes = None
    else:
        if not isinstance(holdouts, Mapping) or set(holdouts) != set(HOLDOUT_PANELS):
            raise ValueError("both holdouts are mandatory after a fit pass")
        holdout_passes = {}
        for panel in HOLDOUT_PANELS:
            record = holdouts[panel]
            if not isinstance(record, Mapping):
                raise ValueError(f"malformed holdout decision: {panel}")
            holdout_passes[panel] = _strict_bool(record.get("passes"), f"{panel} passes")
    favorable = holdout_passes is not None and all(holdout_passes.values())
    return {
        "schema": SEED_DECISION_SCHEMA,
        **access,
        "holdout_passes": holdout_passes,
        "classification": (
            "favorable"
            if favorable
            else "fit_pass_holdout_gate_failed"
            if access["holdouts_authorized"]
            else "fit_gate_failed"
        ),
        "favorable": favorable,
        "aggregation_eligible": True,
        "shared_jepa_construction_licensed": False,
        "g2_licensed": False,
        "runtime_licensed": False,
    }


def _validate_content_hash(value: Mapping[str, Any]) -> None:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError("canonical result content SHA-256 mismatch")


def validate_implementation_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("implementation manifest must be a JSON object")
    _validate_content_hash(value)
    required = {
        "schema",
        "binding",
        "preoutput_amendment",
        "attempt_control_amendment",
        "sources",
        "tests",
        "inputs",
        "resource_policy",
        "model_config",
        "objective",
        "preprocessing",
        "controls",
        "projective_query_support",
        "model_initial_state_sha256",
        "model_state_contract_sha256",
        "schedules",
        "commands",
        "content_sha256",
    }
    if set(value) != required or value.get("schema") != IMPLEMENTATION_MANIFEST_SCHEMA:
        raise ValueError("implementation manifest schema fields mismatch")
    if value.get("binding") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["binding"],
        "sha256": EXECUTION_BINDING_SHA256,
    }:
        raise ValueError("implementation manifest binding mismatch")
    if value.get("preoutput_amendment") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["preoutput_amendment"],
        "sha256": PREOUTPUT_AMENDMENT_SHA256,
    }:
        raise ValueError("implementation manifest amendment mismatch")
    if value.get("attempt_control_amendment") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["attempt_control_amendment"],
        "sha256": ATTEMPT_CONTROL_AMENDMENT_SHA256,
    }:
        raise ValueError("implementation manifest attempt-control amendment mismatch")
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "entries", "entry_count", "source_map_sha256"
    }:
        raise ValueError("implementation manifest source-map fields mismatch")
    entries = sources.get("entries")
    if not isinstance(entries, list) or sources.get("entry_count") != len(
        IMPLEMENTATION_SOURCE_PATHS
    ) or len(entries) != len(IMPLEMENTATION_SOURCE_PATHS):
        raise ValueError("implementation manifest source-map count mismatch")
    expected_roles = sorted(IMPLEMENTATION_SOURCE_PATHS)
    normalized = []
    for raw, role in zip(entries, expected_roles, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {"role", "path", "sha256"}:
            raise ValueError("implementation manifest source entry is malformed")
        entry = dict(raw)
        if entry["role"] != role or entry["path"] != IMPLEMENTATION_SOURCE_PATHS[role] or not _is_sha256(entry["sha256"]):
            raise ValueError("implementation manifest source role/path/hash mismatch")
        normalized.append(entry)
    if sources.get("source_map_sha256") != canonical_json_sha256(normalized):
        raise ValueError("implementation manifest source-map SHA-256 mismatch")
    tests = value.get("tests")
    if not isinstance(tests, Mapping) or set(tests) != {
        "command", "passed", "all_passed"
    } or tests.get("command") != IMPLEMENTATION_TEST_COMMAND or tests.get(
        "passed"
    ) != IMPLEMENTATION_TEST_PASSED or tests.get("all_passed") is not True:
        raise ValueError("implementation manifest tests mismatch")
    if value.get("inputs") != INPUT_BINDINGS:
        raise ValueError("implementation manifest input bindings mismatch")
    if value.get("resource_policy") != RESOURCE_POLICY:
        raise ValueError("implementation manifest resource policy mismatch")
    if value.get("model_config") != MODEL_CONFIG or value.get("objective") != OBJECTIVE_CONTRACT:
        raise ValueError("implementation manifest model/objective mismatch")
    if value.get("preprocessing") != PREPROCESSING_CONTRACT or value.get("controls") != CONTROL_CONTRACT:
        raise ValueError("implementation manifest preprocessing/control mismatch")
    if value.get("projective_query_support") != PROJECTIVE_QUERY_SUPPORT:
        raise ValueError("implementation manifest projective query support mismatch")
    for field in ("model_initial_state_sha256", "model_state_contract_sha256"):
        hashes = value.get(field)
        if not isinstance(hashes, Mapping) or set(hashes) != {
            str(seed) for seed in EXPECTED_SEEDS
        } or not all(_is_sha256(digest) for digest in hashes.values()):
            raise ValueError(f"implementation manifest {field} mismatch")
    if value.get("schedules") != SCHEDULE_CONTRACT:
        raise ValueError("implementation manifest schedule commitments mismatch")
    if value.get("commands") != COMMAND_CONTRACT:
        raise ValueError("implementation manifest command contract mismatch")
    return dict(value)


def _validate_utc_timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("+00:00"):
        raise ValueError(f"{name} must be an explicit UTC ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an explicit UTC ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"{name} must be an explicit UTC ISO-8601 timestamp")
    return value


def validate_attempt_marker(
    value: Mapping[str, Any],
    expected_seed: int,
    implementation_manifest: Mapping[str, Any],
    implementation_manifest_file_sha256: str,
    *,
    primary_result: Mapping[str, Any] | None = None,
    primary_file_sha256: str | None = None,
    primary_attempt_marker: Mapping[str, Any] | None = None,
    primary_attempt_marker_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable pre-payload receipt for one authoritative seed."""

    expected_seed = validate_seed(expected_seed)
    manifest = validate_implementation_manifest(implementation_manifest)
    if not _is_sha256(implementation_manifest_file_sha256):
        raise ValueError("external implementation-manifest file SHA-256 is malformed")
    if not isinstance(value, Mapping):
        raise ValueError("N32 attempt marker must be a JSON object")
    _validate_content_hash(value)
    fields = {
        "schema",
        "authoritative",
        "seed",
        "created_at_utc",
        "invocation",
        "invocation_sha256",
        "canonical_result_path",
        "canonical_attempt_marker_path",
        "contract",
        "preoutput_amendment",
        "attempt_control_amendment",
        "implementation_manifest",
        "seed_20260710_result",
        "seed_20260710_attempt_marker",
        "attempt_consumed",
        "retry_permitted",
        "payload_access_started",
        "content_sha256",
    }
    if (
        set(value) != fields
        or value.get("schema") != ATTEMPT_MARKER_SCHEMA
        or value.get("authoritative") is not True
        or value.get("seed") != expected_seed
        or value.get("attempt_consumed") is not True
        or value.get("retry_permitted") is not False
        or value.get("payload_access_started") is not False
        or value.get("canonical_result_path")
        != COMMAND_CONTRACT["canonical_outputs"][str(expected_seed)]
        or value.get("canonical_attempt_marker_path")
        != ATTEMPT_MARKER_PATHS[expected_seed]
    ):
        raise ValueError("authoritative N32 attempt-marker identity mismatch")
    _validate_utc_timestamp(value.get("created_at_utc"), "attempt-marker created_at_utc")
    invocation = value.get("invocation")
    expected_invocation = authoritative_runner_invocation(
        seed=expected_seed,
        implementation_manifest_file_sha256=implementation_manifest_file_sha256,
        primary_file_sha256=primary_file_sha256,
        primary_attempt_marker_file_sha256=primary_attempt_marker_file_sha256,
    )
    if (
        not isinstance(invocation, list)
        or invocation != expected_invocation
        or value.get("invocation_sha256") != canonical_json_sha256(invocation)
    ):
        raise ValueError("authoritative N32 attempt-marker invocation mismatch")
    if value.get("contract") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["binding"],
        "sha256": EXECUTION_BINDING_SHA256,
    } or value.get("preoutput_amendment") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["preoutput_amendment"],
        "sha256": PREOUTPUT_AMENDMENT_SHA256,
    } or value.get("attempt_control_amendment") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["attempt_control_amendment"],
        "sha256": ATTEMPT_CONTROL_AMENDMENT_SHA256,
    }:
        raise ValueError("authoritative N32 attempt-marker binding/amendment mismatch")
    if value.get("implementation_manifest") != {
        "path": (
            f"{REPOSITORY_ROOT}/docs/"
            "lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json"
        ),
        "sha256": implementation_manifest_file_sha256,
        "content_sha256": manifest["content_sha256"],
    }:
        raise ValueError("authoritative N32 attempt-marker implementation mismatch")

    primary_fields = (
        primary_result,
        primary_file_sha256,
        primary_attempt_marker,
        primary_attempt_marker_file_sha256,
    )
    if expected_seed == EXPECTED_SEEDS[0]:
        if any(item is not None for item in primary_fields) or value.get(
            "seed_20260710_result"
        ) is not None or value.get("seed_20260710_attempt_marker") is not None:
            raise ValueError("seed 20260710 attempt marker rejects primary evidence")
    else:
        if (
            not isinstance(primary_result, Mapping)
            or not _is_sha256(primary_file_sha256)
            or not isinstance(primary_attempt_marker, Mapping)
            or not _is_sha256(primary_attempt_marker_file_sha256)
        ):
            raise ValueError("seed 20260711 attempt marker requires primary evidence")
        _validate_content_hash(primary_result)
        _validate_content_hash(primary_attempt_marker)
        if primary_result.get("seed") != EXPECTED_SEEDS[0] or primary_attempt_marker.get(
            "seed"
        ) != EXPECTED_SEEDS[0]:
            raise ValueError("seed 20260711 attempt marker primary seed mismatch")
        if value.get("seed_20260710_result") != {
            "path": COMMAND_CONTRACT["canonical_outputs"][str(EXPECTED_SEEDS[0])],
            "sha256": primary_file_sha256,
            "content_sha256": primary_result["content_sha256"],
        } or value.get("seed_20260710_attempt_marker") != {
            "path": ATTEMPT_MARKER_PATHS[EXPECTED_SEEDS[0]],
            "sha256": primary_attempt_marker_file_sha256,
            "content_sha256": primary_attempt_marker["content_sha256"],
        }:
            raise ValueError("seed 20260711 attempt-marker primary binding mismatch")
    return dict(value)


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be a finite JSON number")
    return result


def _validate_controls(value: object, *, seed: int, panel: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "role_global_shuffle", "same_scene_wrong_view", "wrong_rgb_uses_target_attitude"
    } or value.get("wrong_rgb_uses_target_attitude") is not True:
        raise ValueError(f"{panel} wrong-view control contract mismatch")
    role = value["role_global_shuffle"]
    if not isinstance(role, Mapping) or set(role) != {
        "schema", "seed", "namespace", "record_count", "permutation_sha256",
        "same_image_pairs", "same_scene_pairs", "same_transition_pairs",
    } or role.get("schema") != "lewm_go2_micro_overfit_shuffle_v1" or role.get(
        "seed"
    ) != seed or role.get("namespace") != panel or role.get("record_count") != FRAME_COUNT or not _is_sha256(
        role.get("permutation_sha256")
    ) or any(role.get(name) != 0 for name in (
        "same_image_pairs", "same_scene_pairs", "same_transition_pairs"
    )):
        raise ValueError(f"{panel} role-global control mismatch")
    same = value["same_scene_wrong_view"]
    if not isinstance(same, Mapping) or set(same) != {
        "schema", "seed", "namespace", "record_count", "permutation_sha256",
        "same_image_pairs", "same_transition_pairs", "different_scene_pairs", "scenes",
    } or same.get("schema") != "lewm_go2_micro_overfit_same_scene_wrong_view_v1" or same.get(
        "seed"
    ) != seed or same.get("namespace") != panel or same.get("record_count") != FRAME_COUNT or not _is_sha256(
        same.get("permutation_sha256")
    ) or any(same.get(name) != 0 for name in (
        "same_image_pairs", "same_transition_pairs", "different_scene_pairs"
    )) or not isinstance(same.get("scenes"), Mapping) or not same["scenes"]:
        raise ValueError(f"{panel} same-scene control mismatch")
    frame_total = transition_total = 0
    for scene_id, record in same["scenes"].items():
        if not isinstance(scene_id, str) or not scene_id or not isinstance(record, Mapping) or set(record) != {
            "frame_count", "transition_count", "rotation"
        } or any(type(record[name]) is not int or record[name] <= 0 for name in record):
            raise ValueError(f"{panel} same-scene control scene record mismatch")
        if record["frame_count"] != 2 * record["transition_count"] or record["rotation"] > record["transition_count"]:
            raise ValueError(f"{panel} same-scene control scene arithmetic mismatch")
        frame_total += record["frame_count"]
        transition_total += record["transition_count"]
    if frame_total != FRAME_COUNT or transition_total != FRAME_COUNT // 2:
        raise ValueError(f"{panel} same-scene control totals mismatch")
    expected_permutations = CONTROL_PERMUTATION_SHA256[seed][panel]
    if role.get("permutation_sha256") != expected_permutations[
        "role_global_shuffle"
    ] or same.get("permutation_sha256") != expected_permutations[
        "same_scene_wrong_view"
    ]:
        raise ValueError(f"{panel} exact control permutation identity mismatch")


def validate_panel_report(
    value: object, *, seed: int, panel: str, require_fit_gate: bool
) -> Mapping[str, Any]:
    expected_fields = {
        "schema", "panel", "frame_count", "target_batch_size",
        "combined_model_batch_size", "model_call_dtype", "metric_accumulator_dtype",
        "wrong_rgb_uses_target_attitude", "conditions", "families", "controls",
    } | ({"fit_gate"} if require_fit_gate else set())
    if not isinstance(value, Mapping) or set(value) != expected_fields or value.get(
        "schema"
    ) != PANEL_REPORT_SCHEMA or value.get("panel") != panel or value.get(
        "frame_count"
    ) != FRAME_COUNT or value.get("target_batch_size") != BATCH_SIZE or value.get(
        "combined_model_batch_size"
    ) != 12 or value.get("model_call_dtype") != "float32" or value.get(
        "metric_accumulator_dtype"
    ) != "float64" or value.get("wrong_rgb_uses_target_attitude") is not True:
        raise ValueError(f"{panel} panel report contract mismatch")
    _conditions(value, name=panel)
    families = value.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError(f"{panel} family report mismatch")
    for family in FAMILIES:
        if not isinstance(families[family], Mapping):
            raise ValueError(f"{panel} family report mismatch: {family}")
        _conditions(families[family], name=f"{panel}/{family}")
    _validate_family_aggregation(value, name=panel)
    _validate_controls(value.get("controls"), seed=seed, panel=panel)
    if require_fit_gate and value.get("fit_gate") != fit_panel_gate_report(value):
        raise ValueError("stored fit-panel gate differs from raw metrics")
    return value


def _expected_stage_access(updates: int, evaluations: int) -> tuple[dict[str, int], dict[str, int]]:
    training = {
        "image_requests": updates * BATCH_SIZE,
        "target_requests": updates * BATCH_SIZE,
        "attitude_requests": updates * BATCH_SIZE,
        "image_decode_events": 0,
        "label_shard_npz_open_events": 0,
        "model_calls": updates,
        "model_output_frames": updates * BATCH_SIZE,
        "model_attitude_frames": updates * BATCH_SIZE,
    }
    evaluation = {
        "image_requests": evaluations * FRAME_COUNT * len(CONDITIONS),
        "target_requests": evaluations * FRAME_COUNT,
        "attitude_requests": evaluations * FRAME_COUNT,
        "image_decode_events": 0,
        "label_shard_npz_open_events": 0,
        "model_calls": evaluations * (FRAME_COUNT // BATCH_SIZE),
        "model_output_frames": evaluations * FRAME_COUNT * len(CONDITIONS),
        "model_attitude_frames": evaluations * FRAME_COUNT * len(CONDITIONS),
    }
    return training, evaluation


def _validate_event_record(value: object, expected: Mapping[str, int], *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(EVENT_FIELDS):
        raise ValueError(f"{name} access fields mismatch")
    for field in EVENT_FIELDS:
        if type(value[field]) is not int or value[field] != expected[field]:
            raise ValueError(f"{name} access does not reconcile")


def validate_stage(
    value: object,
    *,
    seed: int,
    branch: str,
    implementation_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = {
        "schema", "stage", "config", "maximum_steps", "completed_steps",
        "batch_size", "evaluation_interval", "optimizer", "objective",
        "fixed_update_budget_consumed", "one_direct_forward_backward_per_update",
        "gradient_accumulation_or_microbatching", "initial_state_sha256",
        "final_state_sha256", "exact_initial_state_restart_verified",
        "minibatch_indices", "minibatch_indices_sha256", "learning_curve",
        "terminal_fit_gate", "training_access", "fit_evaluation_access",
        "holdouts_evaluated",
    }
    config = BRANCH_CONFIGS[branch]
    updates = config["updates"]
    if not isinstance(value, Mapping) or set(value) != fields or value.get(
        "schema"
    ) != STAGE_SCHEMA or value.get("stage") != branch or value.get("config") != config or value.get(
        "maximum_steps"
    ) != updates or value.get("completed_steps") != updates or value.get(
        "batch_size"
    ) != BATCH_SIZE or value.get("evaluation_interval") != EVALUATION_INTERVAL or value.get(
        "optimizer"
    ) != OPTIMIZER_CONFIGS[branch] or value.get("objective") != OBJECTIVE_CONTRACT or value.get(
        "fixed_update_budget_consumed"
    ) is not True or value.get("one_direct_forward_backward_per_update") is not True or value.get(
        "gradient_accumulation_or_microbatching"
    ) is not False or value.get("exact_initial_state_restart_verified") is not True or type(
        value.get("holdouts_evaluated")
    ) is not bool:
        raise ValueError(f"{branch} exact stage budget/objective contract mismatch")
    expected_initial = implementation_manifest["model_initial_state_sha256"][str(seed)]
    if value.get("initial_state_sha256") != expected_initial or not _is_sha256(value.get("final_state_sha256")):
        raise ValueError(f"{branch} initial/final state identity mismatch")
    schedule_sha = validate_minibatch_schedule(value.get("minibatch_indices"), seed=seed, branch=branch)
    if value.get("minibatch_indices_sha256") != schedule_sha:
        raise ValueError(f"{branch} schedule content hash mismatch")
    curve = value.get("learning_curve")
    if not isinstance(curve, list) or len(curve) != updates // EVALUATION_INTERVAL:
        raise ValueError(f"{branch} complete learning curve is missing")
    for point in curve:
        if not isinstance(point, Mapping) or set(point) != {
            "step", "batch_loss", "gradient_norm_before_clip", "fit_panel"
        }:
            raise ValueError(f"{branch} learning curve point fields mismatch")
        _finite_number(point["batch_loss"], f"{branch} batch loss", minimum=0.0)
        _finite_number(point["gradient_norm_before_clip"], f"{branch} gradient norm", minimum=0.0)
        validate_panel_report(point["fit_panel"], seed=seed, panel="fit", require_fit_gate=True)
    terminal = terminal_fit_gate_summary(curve, updates, EVALUATION_INTERVAL)
    if value.get("terminal_fit_gate") != terminal:
        raise ValueError(f"{branch} terminal fit summary differs from the raw curve")
    training, evaluation = _expected_stage_access(updates, len(curve))
    _validate_event_record(value.get("training_access"), training, name=f"{branch} training")
    _validate_event_record(value.get("fit_evaluation_access"), evaluation, name=f"{branch} fit evaluation")
    return value


def validate_access_ledger(
    value: object,
    *,
    stages: Mapping[str, Mapping[str, Any] | None],
    qualifying_branch: str | None,
    implementation_manifest: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any]:
    fields = {
        "schema", "panels", "fit_dataset_totals", "sidecar", "dataset_roles",
        "holdout_payloads_opened_only_after_terminal_fit_pass",
        "wrong_rgb_target_attitude_frames", "non_train_image_opens",
        "non_train_label_shard_opens", "non_train_model_outputs",
        "controlled_metadata_reads",
    }
    if not isinstance(value, Mapping) or set(value) != fields or value.get(
        "schema"
    ) != ACCESS_LEDGER_SCHEMA or value.get(
        "holdout_payloads_opened_only_after_terminal_fit_pass"
    ) is not True or any(value.get(name) != 0 for name in (
        "non_train_image_opens", "non_train_label_shard_opens", "non_train_model_outputs"
    )):
        raise ValueError("N32 access ledger fields or forbidden counters mismatch")
    invoked = [stage for stage in stages.values() if stage is not None]
    summed = {field: 0 for field in EVENT_FIELDS}
    for stage in invoked:
        for source in (stage["training_access"], stage["fit_evaluation_access"]):
            for field in EVENT_FIELDS:
                summed[field] += source[field]
    fit_totals = dict(summed)
    fit_totals["image_decode_events"] = FRAME_COUNT
    fit_totals["label_shard_npz_open_events"] = 20
    fit_totals["model_calls"] = 0
    fit_totals["model_output_frames"] = 0
    fit_totals["model_attitude_frames"] = 0
    _validate_event_record(value.get("fit_dataset_totals"), fit_totals, name="fit dataset totals")
    panels = value.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != {"fit", *HOLDOUT_PANELS}:
        raise ValueError("N32 panel access ledger is incomplete")
    fit = panels["fit"]
    if not isinstance(fit, Mapping) or set(fit) != {
        "authorized", "artifact_hash_passes", "image_hash_byte_open_events",
        "shard_hash_byte_open_events", "dataset_access",
    } or fit.get("authorized") is not True or fit.get("artifact_hash_passes") != 2 or fit.get(
        "image_hash_byte_open_events"
    ) != 640 or fit.get("shard_hash_byte_open_events") != 40:
        raise ValueError("fit panel access contract mismatch")
    _validate_event_record(fit.get("dataset_access"), fit_totals, name="fit panel")
    holdouts_authorized = qualifying_branch is not None
    for panel, shard_count in (("same_scene_holdout", 20), ("cross_scene_holdout", 25)):
        record = panels[panel]
        if not isinstance(record, Mapping) or set(record) != {
            "authorized", "authorized_by_branch", "artifact_hash_passes",
            "image_hash_byte_open_events", "shard_hash_byte_open_events",
            "dataset_access", "one_shot_evaluation",
        } or record.get("authorized") is not holdouts_authorized or record.get(
            "authorized_by_branch"
        ) != qualifying_branch or record.get("one_shot_evaluation") is not holdouts_authorized:
            raise ValueError(f"{panel} access authorization mismatch")
        if holdouts_authorized:
            if record.get("artifact_hash_passes") != 2 or record.get(
                "image_hash_byte_open_events"
            ) != 640 or record.get("shard_hash_byte_open_events") != 2 * shard_count:
                raise ValueError(f"{panel} artifact access mismatch")
            expected = {
                "image_requests": 960,
                "target_requests": 320,
                "attitude_requests": 320,
                "image_decode_events": 320,
                "label_shard_npz_open_events": shard_count,
                "model_calls": 80,
                "model_output_frames": 960,
                "model_attitude_frames": 960,
            }
        else:
            if any(record.get(name) != 0 for name in (
                "artifact_hash_passes", "image_hash_byte_open_events", "shard_hash_byte_open_events"
            )):
                raise ValueError(f"{panel} unauthorized artifact opens exist")
            expected = {field: 0 for field in EVENT_FIELDS}
        _validate_event_record(record.get("dataset_access"), expected, name=panel)
    sidecar = value.get("sidecar")
    if sidecar != {
        "manifest_byte_opens": 2,
        "train_role_byte_opens": 2,
        "checkpoint_selection_role_byte_opens": 0,
        "probability_calibration_role_byte_opens": 0,
        "g2_evaluation_role_byte_opens": 0,
    }:
        raise ValueError("sidecar role access mismatch")
    metadata = value.get("controlled_metadata_reads")
    expected_source_opens = {
        entry["role"]: 2 for entry in implementation_manifest["sources"]["entries"]
    }
    if metadata != {
        "implementation_manifest_byte_opens": 2,
        "source_byte_opens": expected_source_opens,
        "input_byte_opens": {name: 2 for name in INPUT_BINDINGS},
        "seed_20260710_result_byte_opens": 0 if seed == EXPECTED_SEEDS[0] else 2,
        "authoritative_attempt_marker_byte_opens": 2,
        "seed_20260710_attempt_marker_byte_opens": (
            0 if seed == EXPECTED_SEEDS[0] else 2
        ),
    }:
        raise ValueError("controlled metadata/evidence read ledger mismatch")
    roles = value.get("dataset_roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "train", "checkpoint_selection", "probability_calibration", "g2_evaluation"
    } or roles.get("train") != {
        "panel_transition_rows_joined": 480,
        "model_outputs": summed["model_output_frames"] + (1920 if holdouts_authorized else 0),
    }:
        raise ValueError("train-role access reconciliation mismatch")
    zero_role = {"image_byte_opens": 0, "label_shard_byte_opens": 0, "model_outputs": 0}
    if any(roles[role] != zero_role for role in (
        "checkpoint_selection", "probability_calibration", "g2_evaluation"
    )):
        raise ValueError("forbidden non-train role access exists")
    expected_wrong = sum(stage["fit_evaluation_access"]["model_attitude_frames"] * 2 // 3 for stage in invoked) + (
        1280 if holdouts_authorized else 0
    )
    if value.get("wrong_rgb_target_attitude_frames") != expected_wrong:
        raise ValueError("wrong-RGB target-attitude access does not reconcile")
    return value


def validate_authoritative_result(
    value: Mapping[str, Any],
    expected_seed: int,
    implementation_manifest: Mapping[str, Any],
    implementation_manifest_file_sha256: str,
    attempt_marker: Mapping[str, Any],
    attempt_marker_file_sha256: str,
    *,
    primary_result: Mapping[str, Any] | None = None,
    primary_file_sha256: str | None = None,
    primary_attempt_marker: Mapping[str, Any] | None = None,
    primary_attempt_marker_file_sha256: str | None = None,
) -> dict[str, Any]:
    expected_seed = validate_seed(expected_seed)
    manifest = validate_implementation_manifest(implementation_manifest)
    if not _is_sha256(implementation_manifest_file_sha256):
        raise ValueError("external implementation-manifest file SHA-256 is malformed")
    if not isinstance(value, Mapping):
        raise ValueError("N32 result must be a JSON object")
    _validate_content_hash(value)
    primary_validated = None
    if expected_seed == EXPECTED_SEEDS[1]:
        if (
            not isinstance(primary_result, Mapping)
            or not _is_sha256(primary_file_sha256)
            or not isinstance(primary_attempt_marker, Mapping)
            or not _is_sha256(primary_attempt_marker_file_sha256)
        ):
            raise ValueError(
                "seed 20260711 requires the immutable first result, attempt marker, "
                "and external file SHA-256 values"
            )
        primary_validated = validate_authoritative_result(
            primary_result,
            EXPECTED_SEEDS[0],
            manifest,
            implementation_manifest_file_sha256,
            primary_attempt_marker,
            primary_attempt_marker_file_sha256,
        )
    marker = validate_attempt_marker(
        attempt_marker,
        expected_seed,
        manifest,
        implementation_manifest_file_sha256,
        primary_result=primary_validated,
        primary_file_sha256=primary_file_sha256,
        primary_attempt_marker=primary_attempt_marker,
        primary_attempt_marker_file_sha256=primary_attempt_marker_file_sha256,
    )
    if not _is_sha256(attempt_marker_file_sha256):
        raise ValueError("external attempt-marker file SHA-256 is malformed")
    fields = {
        "schema", "authoritative", "aggregation_eligible", "promotion_eligible",
        "seed", "created_at_utc", "completed_at_utc", "invocation", "execution",
        "contract", "preoutput_amendment", "attempt_control_amendment",
        "attempt_marker", "implementation_manifest",
        "implementation_manifest_content_sha256", "inputs", "source_hashes", "git",
        "model_config", "model", "preprocessing", "objective",
        "projective_query_support", "panel_join", "projection_parity", "stages",
        "qualifying_branch", "patch7_reference", "holdouts", "holdout_checks",
        "decision", "artifact_verification", "access_ledger", "publication",
        "shared_jepa_construction_licensed", "g2_licensed", "runtime_licensed",
        "content_sha256",
    }
    if (
        set(value) != fields
        or value.get("schema") != RESULT_SCHEMA
        or value.get("authoritative") is not True
        or value.get("aggregation_eligible") is not True
        or value.get("promotion_eligible") is not False
        or value.get("seed") != expected_seed
        or value.get("shared_jepa_construction_licensed") is not False
        or value.get("g2_licensed") is not False
        or value.get("runtime_licensed") is not False
    ):
        raise ValueError("authoritative N32 result identity mismatch")
    created = _validate_utc_timestamp(value.get("created_at_utc"), "result created_at_utc")
    completed = _validate_utc_timestamp(
        value.get("completed_at_utc"), "result completed_at_utc"
    )
    if (
        datetime.fromisoformat(completed) < datetime.fromisoformat(created)
        or value.get("invocation") != marker["invocation"]
        or created != marker["created_at_utc"]
    ):
        raise ValueError("authoritative N32 timestamps/invocation mismatch")
    if value.get("contract") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["binding"], "sha256": EXECUTION_BINDING_SHA256
    } or value.get("preoutput_amendment") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["preoutput_amendment"], "sha256": PREOUTPUT_AMENDMENT_SHA256
    } or value.get("attempt_control_amendment") != {
        "path": IMPLEMENTATION_SOURCE_PATHS["attempt_control_amendment"],
        "sha256": ATTEMPT_CONTROL_AMENDMENT_SHA256,
    }:
        raise ValueError("authoritative N32 binding/amendment mismatch")
    if value.get("attempt_marker") != {
        "path": ATTEMPT_MARKER_PATHS[expected_seed],
        "sha256": attempt_marker_file_sha256,
        "content_sha256": marker["content_sha256"],
    }:
        raise ValueError("authoritative N32 attempt-marker binding mismatch")
    implementation = value.get("implementation_manifest")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "path", "sha256", "content_sha256"
    } or implementation.get("path") != f"{REPOSITORY_ROOT}/docs/lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json" or implementation.get(
        "sha256"
    ) != implementation_manifest_file_sha256 or implementation.get("content_sha256") != manifest["content_sha256"] or value.get(
        "implementation_manifest_content_sha256"
    ) != manifest["content_sha256"]:
        raise ValueError("authoritative N32 implementation manifest mismatch")
    if value.get("model_config") != MODEL_CONFIG or value.get("preprocessing") != PREPROCESSING_CONTRACT or value.get(
        "objective"
    ) != OBJECTIVE_CONTRACT or value.get("projective_query_support") != PROJECTIVE_QUERY_SUPPORT:
        raise ValueError("authoritative N32 model/objective/preprocessing/support mismatch")
    source_hashes = {
        entry["role"]: entry["sha256"] for entry in manifest["sources"]["entries"]
    }
    if value.get("source_hashes") != source_hashes:
        raise ValueError("authoritative N32 source hashes mismatch")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {*INPUT_BINDINGS, "seed_20260710_result"}:
        raise ValueError("authoritative N32 input fields mismatch")
    for name, record in INPUT_BINDINGS.items():
        if inputs.get(name) != record:
            raise ValueError(f"authoritative N32 bound input mismatch: {name}")
    if expected_seed == EXPECTED_SEEDS[0]:
        if any(
            item is not None
            for item in (
                primary_result,
                primary_file_sha256,
                primary_attempt_marker,
                primary_attempt_marker_file_sha256,
            )
        ) or inputs["seed_20260710_result"] is not None:
            raise ValueError("seed 20260710 rejects primary-result authorization")
    else:
        assert primary_validated is not None
        expected_primary = {
            "path": COMMAND_CONTRACT["canonical_outputs"][str(EXPECTED_SEEDS[0])],
            "sha256": primary_file_sha256,
            "content_sha256": primary_validated["content_sha256"],
        }
        if inputs["seed_20260710_result"] != expected_primary:
            raise ValueError("seed 20260711 primary-result binding mismatch")
    execution = value.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {
        "device", "determinism", "batch_size_frames",
        "evaluation_combined_model_batch_size", "evaluation_interval", "branches",
        "source_workers", "native_threads_per_worker",
        "fp32_no_autocast_amp_compile_quantization_or_query_chunking",
    } or execution.get("batch_size_frames") != BATCH_SIZE or execution.get(
        "evaluation_combined_model_batch_size"
    ) != 12 or execution.get("evaluation_interval") != EVALUATION_INTERVAL or execution.get(
        "branches"
    ) != BRANCH_CONFIGS or execution.get("source_workers") != 6 or execution.get(
        "native_threads_per_worker"
    ) != 1 or execution.get("fp32_no_autocast_amp_compile_quantization_or_query_chunking") is not True:
        raise ValueError("authoritative N32 execution contract mismatch")
    device = execution["device"]
    if not isinstance(device, Mapping) or set(device) != {
        "device", "device_name", "total_memory_bytes", "hip_visible_devices",
        "hsa_override_gfx_version_unset", "raphael_rejected",
    } or device.get("device") != "cuda:0" or "r9700" not in str(device.get("device_name", "")).lower().replace(" ", "") or type(
        device.get("total_memory_bytes")
    ) is not int or device["total_memory_bytes"] < RESOURCE_POLICY["minimum_device_memory_bytes"] or device.get(
        "hip_visible_devices"
    ) != "0" or device.get("hsa_override_gfx_version_unset") is not True or device.get("raphael_rejected") is not True:
        raise ValueError("authoritative N32 discrete-device contract mismatch")
    determinism = execution["determinism"]
    if not isinstance(determinism, Mapping) or set(determinism) != {
        "seed", "torch_deterministic_algorithms", "warn_only",
        "cudnn_benchmark", "cudnn_deterministic",
    } or determinism.get("seed") != expected_seed or determinism.get(
        "torch_deterministic_algorithms"
    ) is not True or determinism.get("warn_only") is not False or determinism.get(
        "cudnn_benchmark"
    ) is not False or determinism.get("cudnn_deterministic") is not True:
        raise ValueError("authoritative N32 determinism contract mismatch")
    model = value.get("model")
    if not isinstance(model, Mapping) or set(model) != {
        "class", "entrypoint", "initialization", "all_invoked_branches_restart_same_initial_state",
        "n32_weights_are_not_checkpointed_or_promotable",
    } or model.get("class") != "EgomotionBevJepa" or model.get("entrypoint") != "occupancy_logits" or model.get(
        "all_invoked_branches_restart_same_initial_state"
    ) is not True or model.get("n32_weights_are_not_checkpointed_or_promotable") is not True:
        raise ValueError("authoritative N32 model record mismatch")
    initialization = model.get("initialization")
    if not isinstance(initialization, Mapping) or set(initialization) != {
        "initial_state_sha256", "state_contract", "state_contract_sha256",
        "parameter_count", "trainable_parameter_count",
    } or initialization.get("initial_state_sha256") != manifest[
        "model_initial_state_sha256"
    ][str(expected_seed)] or initialization.get("state_contract_sha256") != manifest[
        "model_state_contract_sha256"
    ][str(expected_seed)] or canonical_json_sha256(initialization.get("state_contract")) != initialization.get(
        "state_contract_sha256"
    ) or type(initialization.get("parameter_count")) is not int or initialization["parameter_count"] <= 0 or type(
        initialization.get("trainable_parameter_count")
    ) is not int or not 0 < initialization["trainable_parameter_count"] <= initialization["parameter_count"]:
        raise ValueError("authoritative N32 initialization record mismatch")
    state_contract = initialization["state_contract"]
    if not isinstance(state_contract, Mapping) or set(state_contract) != {"entry_count", "entries"} or not isinstance(
        state_contract.get("entries"), list
    ) or state_contract.get("entry_count") != len(state_contract["entries"]):
        raise ValueError("authoritative N32 state contract mismatch")
    parameter_count = trainable_count = 0
    for entry in state_contract["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "dtype", "shape", "requires_grad"} or not isinstance(
            entry["name"], str
        ) or not entry["name"] or not isinstance(entry["dtype"], str) or not isinstance(entry["shape"], list) or any(
            type(size) is not int or size < 0 for size in entry["shape"]
        ) or (
            entry["requires_grad"] is not None
            and type(entry["requires_grad"]) is not bool
        ):
            raise ValueError("authoritative N32 state-contract entry mismatch")
        if type(entry["requires_grad"]) is bool:
            elements = math.prod(entry["shape"])
            parameter_count += elements
            trainable_count += elements if entry["requires_grad"] else 0
    if parameter_count != initialization["parameter_count"] or trainable_count != initialization["trainable_parameter_count"]:
        raise ValueError("authoritative N32 parameter counts do not match state contract")
    stages = value.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(BRANCH_CONFIGS):
        raise ValueError("authoritative N32 stages are incomplete")
    faithful = stages["production_faithful"]
    ceiling = stages["ceiling_optimizer"]
    if not isinstance(faithful, Mapping):
        raise ValueError("authoritative N32 faithful stage is missing")
    validate_stage(faithful, seed=expected_seed, branch="production_faithful", implementation_manifest=manifest)
    faithful_pass = bool(faithful["terminal_fit_gate"]["passes"])
    if faithful_pass:
        if ceiling is not None:
            raise ValueError("ceiling stage is forbidden after faithful fit pass")
    else:
        validate_stage(ceiling, seed=expected_seed, branch="ceiling_optimizer", implementation_manifest=manifest)
        if ceiling["minibatch_indices"][: len(faithful["minibatch_indices"])] != faithful["minibatch_indices"]:
            raise ValueError("ceiling schedule does not share the faithful prefix")
    access_decision = branch_access_decision(faithful, ceiling)
    if value.get("qualifying_branch") != access_decision["qualifying_branch"]:
        raise ValueError("stored qualifying branch mismatch")
    reference = value.get("patch7_reference")
    if not isinstance(reference, Mapping) or canonical_json_sha256(reference) != PATCH7_REFERENCE_SHA256:
        raise ValueError("stored static patch7 reference mismatch")
    holdouts = value.get("holdouts")
    stored_checks = value.get("holdout_checks")
    recomputed_checks = None
    if access_decision["holdouts_authorized"]:
        if not isinstance(holdouts, Mapping) or set(holdouts) != set(HOLDOUT_PANELS) or not isinstance(
            stored_checks, Mapping
        ) or set(stored_checks) != set(HOLDOUT_PANELS):
            raise ValueError("authorized N32 holdout reports/checks are incomplete")
        recomputed_checks = {}
        for panel in HOLDOUT_PANELS:
            validate_panel_report(holdouts[panel], seed=expected_seed, panel=panel, require_fit_gate=False)
            recomputed_checks[panel] = strict_patch7_holdout_checks(
                holdouts[panel], reference["panels"][panel]
            )
        if stored_checks != recomputed_checks:
            raise ValueError("stored holdout checks differ from raw reports/reference")
    elif holdouts is not None or stored_checks is not None:
        raise ValueError("holdout reports are forbidden after fit failure")
    recomputed = per_seed_decision(faithful, ceiling, recomputed_checks)
    if value.get("decision") != recomputed:
        raise ValueError("authoritative N32 stored decision mismatch")
    for branch, stage in stages.items():
        if stage is not None and stage["holdouts_evaluated"] is not (
            branch == access_decision["qualifying_branch"]
        ):
            raise ValueError("stage holdout-evaluation flag mismatch")
    validate_access_ledger(
        value.get("access_ledger"),
        stages=stages,
        qualifying_branch=access_decision["qualifying_branch"],
        implementation_manifest=manifest,
        seed=expected_seed,
    )
    join = value.get("panel_join")
    if join != PANEL_JOIN_CONTRACT:
        raise ValueError("authoritative N32 panel/attitude join mismatch")
    if value.get("projection_parity") != {
        "content_sha256": INPUT_BINDINGS["fit_projection_parity"]["content_sha256"],
        "frame_count": 320,
        "mismatched_cells": 0,
    }:
        raise ValueError("authoritative N32 projection parity mismatch")
    if value.get("artifact_verification") != {
        "fit_verified_before_first_payload_access": True,
        "fit_verified_after_last_model_access": True,
        "holdouts_verified_only_after_terminal_fit_pass": True,
        "holdouts_evaluated_once": access_decision["holdouts_authorized"],
    }:
        raise ValueError("authoritative N32 artifact verification mismatch")
    if value.get("publication") != {
        "mode": "private_staging_hardlink_noreplace",
        "canonical_output": COMMAND_CONTRACT["canonical_outputs"][str(expected_seed)],
    }:
        raise ValueError("authoritative N32 publication contract mismatch")
    git = value.get("git")
    if not isinstance(git, Mapping) or set(git) != {"start", "end"}:
        raise ValueError("authoritative N32 git record mismatch")
    for moment in ("start", "end"):
        snapshot = git[moment]
        if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "head", "status_short", "tracked_dirty_diff_sha256", "tracked_dirty_diff_bytes"
        } or not isinstance(snapshot["head"], str) or not _is_sha256(snapshot["tracked_dirty_diff_sha256"]) or type(
            snapshot["tracked_dirty_diff_bytes"]
        ) is not int or snapshot["tracked_dirty_diff_bytes"] < 0:
            raise ValueError("authoritative N32 git snapshot mismatch")
    if git["start"]["head"] != git["end"]["head"] or git["start"]["tracked_dirty_diff_sha256"] != git["end"]["tracked_dirty_diff_sha256"]:
        raise ValueError("authoritative N32 tracked sources changed during execution")
    return dict(value)


validate_result_contract = validate_authoritative_result


def validate_seed_pair(
    primary: Mapping[str, Any],
    replication: Mapping[str, Any],
    implementation_manifest: Mapping[str, Any],
    implementation_manifest_file_sha256: str,
    primary_file_sha256: str,
    primary_attempt_marker: Mapping[str, Any],
    primary_attempt_marker_file_sha256: str,
    replication_attempt_marker: Mapping[str, Any],
    replication_attempt_marker_file_sha256: str,
) -> dict[str, Any]:
    manifest = validate_implementation_manifest(implementation_manifest)
    primary_validated = validate_authoritative_result(
        primary,
        EXPECTED_SEEDS[0],
        manifest,
        implementation_manifest_file_sha256,
        primary_attempt_marker,
        primary_attempt_marker_file_sha256,
    )
    replication_validated = validate_authoritative_result(
        replication,
        EXPECTED_SEEDS[1],
        manifest,
        implementation_manifest_file_sha256,
        replication_attempt_marker,
        replication_attempt_marker_file_sha256,
        primary_result=primary_validated,
        primary_file_sha256=primary_file_sha256,
        primary_attempt_marker=primary_attempt_marker,
        primary_attempt_marker_file_sha256=primary_attempt_marker_file_sha256,
    )
    favorable = primary_validated["decision"]["favorable"] is True and replication_validated[
        "decision"
    ]["favorable"] is True
    return {
        "schema": SEED_PAIR_SCHEMA,
        "seeds": list(EXPECTED_SEEDS),
        "both_registered_seeds_completed": True,
        "both_favorable": favorable,
        "shared_jepa_construction_licensed": favorable,
        "g2_licensed": False,
        "runtime_licensed": False,
    }


__all__ = [
    "ATTEMPT_CONTROL_AMENDMENT_SHA256", "ATTEMPT_MARKER_PATHS",
    "ATTEMPT_MARKER_SCHEMA", "BATCH_SIZE", "BRANCH_CONFIGS", "CLASS_NAMES", "CONDITIONS",
    "EVALUATION_INTERVAL", "EXECUTION_BINDING_SHA256", "EXPECTED_SEEDS",
    "FAMILIES", "FIT_GATE_SCHEMA", "FRAME_COUNT", "GRADIENT_CLIP",
    "HOLDOUT_CHECK_SCHEMA", "HOLDOUT_PANELS", "IMPLEMENTATION_MANIFEST_SCHEMA",
    "MODEL_CONFIG", "PREOUTPUT_AMENDMENT_SHA256", "RESULT_SCHEMA",
    "SCHEDULE_SHA256", "SEED_DECISION_SCHEMA", "SEED_PAIR_SCHEMA",
    "SMOKE_RESULT_SCHEMA", "STAGE_SCHEMA", "TERMINAL_FIT_SCHEMA",
    "branch_access_decision", "canonical_json_bytes", "canonical_json_sha256",
    "categorical_holdout_checks", "deterministic_minibatch_schedule",
    "extract_faithful_patch7_family_reference", "fit_panel_gate_report",
    "per_seed_decision", "strict_patch7_holdout_checks",
    "terminal_fit_gate_summary", "validate_authoritative_result",
    "validate_attempt_marker", "validate_implementation_manifest", "validate_minibatch_schedule",
    "validate_result_contract", "validate_seed", "validate_seed_pair",
]

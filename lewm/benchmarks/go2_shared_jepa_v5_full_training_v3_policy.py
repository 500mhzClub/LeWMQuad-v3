"""Pure frozen contract for Shared JEPA V5 full training V3.

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
IMPLEMENTATION_AUTHOR = "/root/full_training_v3"

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

V3_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v3_successor_amendment_"
    "2026-07-14.md"
)
V3_AMENDMENT_SHA256 = (
    "93737e1556fc3b523408e0fd01ed632ec8571acb30978ae1f17e1dd653e40278"
)
V3_TOPOLOGY_CORRECTION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v3_camera_ladder_topology_"
    "correction_amendment_2026-07-14.md"
)
V3_TOPOLOGY_CORRECTION_SHA256 = (
    "49e06b84da81141e59a3a9c4623abc82901320804732c864c8ecd66c51c768a0"
)

FROZEN_V2_IMPLEMENTATION_BINDINGS = {
    "lewm/benchmarks/go2_shared_jepa_v5_full_training_v2_policy.py": (
        "e0c3409ce104d954e40aa73ae5bd5b79ec3daa77564e90c6be183c2fbc19f680"
    ),
    "scripts/preflight_go2_shared_jepa_v5_full_training_v2.py": (
        "fbc6d63394625d2c3ccc79821d9a07b507fdfb95e02ee1768ed6325857531eff"
    ),
    "scripts/verify_go2_shared_jepa_v5_full_training_v2_preflight.py": (
        "1453a6a6134c25cad21d41f44628e4cc8e1e041ae8994d570413ebb1101e09e3"
    ),
    "scripts/execute_go2_shared_jepa_v5_full_training_v2.py": (
        "698fb92f2f854365f2d0bfbf6f034b1c3f04704a8d6227fceff7c3ed275fc271"
    ),
    "scripts/train_go2_shared_jepa_v5_full_training_v2.py": (
        "bdd8e4b1c24e855f3e3ff535a195f2c370c4ffdadc48eb9e83b214b53362f23b"
    ),
    "scripts/verify_go2_shared_jepa_v5_full_training_v2.py": (
        "d8950c8bf23b0bd5494c7c864f2f2543d533b0bc07af3f70287291227c872543"
    ),
    "docs/lewm_go2_shared_jepa_v5_full_training_v2_implementation_"
    "independent_review_2026-07-13.json": (
        "2ce422c2821491f936af9b47a5898f90969723338195d7f2069902357297132a"
    ),
}

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
HIERARCHICAL_FIRST_HIT_RELATIVE_PATH = (
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py"
)
HIERARCHICAL_FIRST_HIT_SHA256 = (
    "52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd"
)
GATE_ALIGNED_RASTER_NLL_RELATIVE_PATH = (
    "lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py"
)
GATE_ALIGNED_RASTER_NLL_SHA256 = (
    "735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662"
)
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
    "docs/lewm_go2_shared_jepa_v5_full_training_v3_exact_execution_manifest_"
    "2026-07-14.json"
)
PREFLIGHT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "full_training_v3_preflight"
)
EXACT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v3"
)
CANONICAL_PREFLIGHT_ROOT = ROOT / PREFLIGHT_ROOT_RELATIVE_PATH
CANONICAL_EXACT_ROOT = ROOT / EXACT_ROOT_RELATIVE_PATH
PREFLIGHT_RECEIPT_RELATIVE_PATH = (
    f"{PREFLIGHT_ROOT_RELATIVE_PATH}/gpu_smoke_receipt.json"
)
PREFLIGHT_COMPLETED_RELATIVE_PATH = f"{PREFLIGHT_ROOT_RELATIVE_PATH}/completed.json"
PREFLIGHT_INDEPENDENT_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v3_preflight_independent_review_"
    "2026-07-14.json"
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
    "development_raw_supervision_v1.audit_v13.json"
)
RAW_SUPERVISION_BUILDER_RELATIVE_PATH = (
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py"
)
RAW_SUPERVISION_AUDITOR_RELATIVE_PATH = (
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v13.py"
)
RAW_BUILDER_V9_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_"
    "independent_review_2026-07-13.json"
)
RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_independent_"
    "review_2026-07-14.json"
)
RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v13_authorization_"
    "2026-07-14.json"
)
RAW_AUDITOR_V13_FINGERPRINT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v13_authorization_"
    "fingerprint_2026-07-14.json"
)
RAW_SUPERVISION_MANIFEST_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_dataset_v1"
)
RAW_SUPERVISION_AUDIT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_v13"

RAW_V9_MANIFEST_FILE_SHA256 = (
    "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360"
)
RAW_V9_MANIFEST_CONTENT_SHA256 = (
    "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a"
)
RAW_V13_PASS_FILE_SHA256 = (
    "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76"
)
RAW_V13_PASS_CONTENT_SHA256 = (
    "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca"
)
RAW_V13_SAMPLE_RESULTS_SHA256 = (
    "a051b9a0a10f14413105f2f1cc3c36ad10a43ec20071f0577efcc99fc321d356"
)

RAW_CHAIN_SOURCE_BINDINGS = {
    RAW_SUPERVISION_BUILDER_RELATIVE_PATH: (
        "2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v9.py": (
        "f239a4ef7c067a71f991b30e14bd5c8632c31be3173780fc25b3d9801fff79ee"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py": (
        "541d1957df0a3da18c2b529cd2d7ca721d7e657c8ebcced2a37931d502cab7bc"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_"
    "author_handoff_2026-07-13.md": (
        "b6cdf34fa933214e1bb603681f4638f2226e093dad42705445fd8084d6442efd"
    ),
    RAW_BUILDER_V9_REVIEW_RELATIVE_PATH: (
        "c39eb2787c37f8cab064de75355b3af56971ef98209d329e4789eb383c1dc60f"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_self_"
    "resolving_cli_successor_amendment_2026-07-14.md": (
        "094072a8289e69a894310a1a327327ee92e7af5e448c39a8d2f6c9e0b3c008ed"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_authorization_"
    "2026-07-14.json": (
        "6b5f317119a00308390b8a32f1057f34455313eb80ec190aa9d8d27052a81575"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_authorization_"
    "fingerprint_2026-07-14.json": (
        "662e6c2f6386b8822b3bd968a4faf0bf3e2e222ff4aac9df8a99cc680c254327"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_launch_failure_"
    "2026-07-14.json": (
        "cc6313b1d6e56022204ba82dc57efc6b7cc85a715f078cd865883b61cee88eb3"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v12.py": (
        "f435406c7ff8d42a549cd678a65584bc88ac49f96b590247b811c6bb4b934943"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v12.py": (
        "45f93534b02afe99722144509fc9b7dde72e735daa8bed1dc58951d3c0bb8471"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v12.py": (
        "dbefb4dc455b45873e14256d5fa647e22fcf1eff1a43ba249e7b9fe7f5ed5dd7"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_author_handoff_"
    "2026-07-14.md": (
        "d1955fde4106cf54f1adb75fcbd84abb00b24597384c5ad05c51abf73b22e4ef"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_independent_"
    "review_2026-07-14.json": (
        "d7ae190f1971befbc26ae2e7b6a36955a614bf9c94f85860a4d4d26922d91d30"
    ),
    RAW_SUPERVISION_AUDITOR_RELATIVE_PATH: (
        "fddc678187f082a0a245ff5868ca5d944cba4adc2703d3b97088d57451deb4b7"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v13.py": (
        "c7b2018f9296d92ab0abf3745a8afa5108a7404496fa382a7f75bd3b7307ba4b"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v13.py": (
        "7fb40f59be369ec35852cc10604a2bd8c0a08f083d19403ef1eb7b9c759d4c7e"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_author_handoff_"
    "2026-07-14.md": (
        "da34de8e05e8ae03072e8fa5c211b53d39cf74fb0ff103298ebf6eaae701a79d"
    ),
    RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH: (
        "f3705d1a300204a3e4f7e52b31fae5401b56bbe8de018972ebe66f046c9b2343"
    ),
    RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH: (
        "8a12c5f8d6c6e64a418052cf01177dd25049d6d373f7e87cd52c5d2a5b2bf587"
    ),
    RAW_AUDITOR_V13_FINGERPRINT_RELATIVE_PATH: (
        "882bf8877b12874998ad0f4d179d89ebe8d7db048ffdf3ddc03d4ea38ea5b846"
    ),
}

RAW_DATASET_USE_GRANT = {
    "grant": "v3_exact_development_roles_only",
    "roles": ["train", "checkpoint_selection", "probability_calibration"],
    "requires_exact_reservation": True,
    "requires_raw_v13_chain": True,
    "dataset_use_authorized": True,
    "training_selection_calibration_authorized": True,
    "rgb_outside_bound_raw_leaves_authorized": False,
    "g2_or_heldout_authorized": False,
    "runtime_navigation_hardware_authorized": False,
    "production_promotion_deployment_authorized": False,
    "retry_authorized": False,
}

CAMERA_V13_SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_"
    "v13_independent_review_2026-07-14.json"
)
CAMERA_V13_N5_GATE_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n5_gate_aligned_raster_nll_v13/gates/seed_20260710_n5.json"
)
CAMERA_V13_LADDER_PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_gate_aligned_raster_nll_v13_"
    "two_seed_ladder_preregistration_2026-07-14.md"
)
CAMERA_V13_LADDER_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_gate_aligned_raster_nll_v13_"
    "two_seed_ladder_independent_review_2026-07-14.json"
)
V4_TWO_SEED_LADDER_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "gate_aligned_raster_nll_v13_ladder_v1/"
    "gates/two_seed.json"
)
V4_PRIMARY_CHECKPOINT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "gate_aligned_raster_nll_v13_ladder_v1/attempts/"
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
    "lewm/benchmarks/go2_shared_jepa_v5_full_training_v3_policy.py"
)
LOSS_ADAPTER_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v3_loss.py"
)
PREFLIGHT_EXECUTOR_RELATIVE_PATH = (
    "scripts/preflight_go2_shared_jepa_v5_full_training_v3.py"
)
PREFLIGHT_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_shared_jepa_v5_full_training_v3_preflight.py"
)
EXACT_EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_shared_jepa_v5_full_training_v3.py"
)
EXACT_TRAINER_RELATIVE_PATH = (
    "scripts/train_go2_shared_jepa_v5_full_training_v3.py"
)
EXACT_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_shared_jepa_v5_full_training_v3.py"
)
IMPLEMENTATION_SOURCE_PATHS = (
    POLICY_RELATIVE_PATH,
    LOSS_ADAPTER_RELATIVE_PATH,
    PREFLIGHT_EXECUTOR_RELATIVE_PATH,
    PREFLIGHT_VERIFIER_RELATIVE_PATH,
    EXACT_EXECUTOR_RELATIVE_PATH,
    EXACT_TRAINER_RELATIVE_PATH,
    EXACT_VERIFIER_RELATIVE_PATH,
)
IMPLEMENTATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v3_implementation_"
    "independent_review_2026-07-14.json"
)

EXECUTION_MANIFEST_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_manifest_v1"
IMPLEMENTATION_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_implementation_review_v1"
)
PREFLIGHT_RESERVATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_preflight_reservation_v1"
)
PREFLIGHT_RECEIPT_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_gpu_smoke_receipt_v1"
)
PREFLIGHT_COMPLETION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_preflight_completion_v1"
)
EXACT_RESERVATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_reservation_v1"
)
EXACT_COMPLETION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_completion_v1"
)
EXACT_FAILURE_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_failure_v1"
ACCESS_LEDGER_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_access_ledger_v1"
SCHEDULE_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_schedule_v1"
SOURCE_REVIEW_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_source_review_v1"
INPUT_BINDINGS_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_input_bindings_v1"
INITIALIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_initialization_v1"
TRAINING_RECORD_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_training_record_v1"
SELECTION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_selection_v1"
CALIBRATION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v3_calibration_v1"
DIAGNOSTIC_ABLATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_selection_role_ablation_diagnostic_v1"
)
PRE_G2_CANDIDATE_CHECKPOINT_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v3_pre_g2_candidate_checkpoint_v1"
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
            "hierarchical_first_hit_nll": 0.25,
            "target_bin_offset_smooth_l1": 0.25,
            "ground_clear_distance_state_balanced_bce": 0.25,
            "derived_raster_hierarchical_bce": 0.25,
            "derived_raster_cell_nll": 0.25,
        },
        "camera_model_config_weight": 1.0,
        "current_and_next_computed_separately_at_batch_size": 4,
        "current_next_scalar_average": [0.5, 0.5],
        "microbatch_scalar_average": [0.25, 0.25, 0.25, 0.25],
        "synthetic_b16_nonlinear_pooling_authorized": False,
    },
    "matched_no_jepa": {
        "established_jepa_total_weight": 0.0,
        "current_v4_weight": 0.5,
        "next_v4_weight": 0.5,
        "same_forward_and_diagnostics": True,
        "camera_model_config_weight": 1.0,
    },
}

REQUIRED_BINDING_NAMES = (
    "development_raw_supervision_manifest_file_sha256",
    "development_raw_supervision_manifest_content_sha256",
    "development_raw_supervision_builder_source_sha256",
    "development_raw_supervision_auditor_source_sha256",
    "development_raw_supervision_audit_file_sha256",
    "development_raw_supervision_audit_content_sha256",
    "camera_v13_source_review_file_sha256",
    "camera_v13_source_review_content_sha256",
    "camera_v13_n5_gate_pass_file_sha256",
    "camera_v13_n5_gate_pass_content_sha256",
    "camera_v13_ladder_preregistration_file_sha256",
    "camera_v13_ladder_independent_review_file_sha256",
    "v4_two_seed_ladder_pass_file_sha256",
    "v4_two_seed_ladder_pass_content_sha256",
    "v4_primary_seed_20260710_n320_checkpoint_file_sha256",
    "preflight_completed_file_sha256",
    "preflight_receipt_file_sha256",
    "preflight_independent_review_file_sha256",
    "implementation_policy_source_sha256",
    "loss_adapter_source_sha256",
    "preflight_executor_source_sha256",
    "preflight_verifier_source_sha256",
    "exact_executor_source_sha256",
    "exact_trainer_source_sha256",
    "exact_verifier_source_sha256",
    "implementation_independent_review_file_sha256",
)

FROZEN_RESOLVED_BINDINGS = {
    "development_raw_supervision_manifest_file_sha256": (
        RAW_V9_MANIFEST_FILE_SHA256
    ),
    "development_raw_supervision_manifest_content_sha256": (
        RAW_V9_MANIFEST_CONTENT_SHA256
    ),
    "development_raw_supervision_builder_source_sha256": (
        RAW_CHAIN_SOURCE_BINDINGS[RAW_SUPERVISION_BUILDER_RELATIVE_PATH]
    ),
    "development_raw_supervision_auditor_source_sha256": (
        RAW_CHAIN_SOURCE_BINDINGS[RAW_SUPERVISION_AUDITOR_RELATIVE_PATH]
    ),
    "development_raw_supervision_audit_file_sha256": RAW_V13_PASS_FILE_SHA256,
    "development_raw_supervision_audit_content_sha256": (
        RAW_V13_PASS_CONTENT_SHA256
    ),
}

CAMERA_V13_UNRESOLVED_BINDING_NAMES = (
    "camera_v13_source_review_file_sha256",
    "camera_v13_source_review_content_sha256",
    "camera_v13_n5_gate_pass_file_sha256",
    "camera_v13_n5_gate_pass_content_sha256",
    "camera_v13_ladder_preregistration_file_sha256",
    "camera_v13_ladder_independent_review_file_sha256",
    "v4_two_seed_ladder_pass_file_sha256",
    "v4_two_seed_ladder_pass_content_sha256",
    "v4_primary_seed_20260710_n320_checkpoint_file_sha256",
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
    "pre_g2_candidate_checkpoint.pt",
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
        V3_AMENDMENT_RELATIVE_PATH: V3_AMENDMENT_SHA256,
        V3_TOPOLOGY_CORRECTION_RELATIVE_PATH: V3_TOPOLOGY_CORRECTION_SHA256,
        MODEL_RELATIVE_PATH: MODEL_SHA256,
        HIERARCHICAL_FIRST_HIT_RELATIVE_PATH: HIERARCHICAL_FIRST_HIT_SHA256,
        GATE_ALIGNED_RASTER_NLL_RELATIVE_PATH: GATE_ALIGNED_RASTER_NLL_SHA256,
        MODEL_TEST_RELATIVE_PATH: MODEL_TEST_SHA256,
        OUTPUT_LOSS_REVIEW_RELATIVE_PATH: OUTPUT_LOSS_REVIEW_SHA256,
        LIFECYCLE_REVIEW_RELATIVE_PATH: LIFECYCLE_REVIEW_SHA256,
        **FROZEN_GOVERNING_DESIGN_BINDINGS,
        **FROZEN_V2_IMPLEMENTATION_BINDINGS,
        **RAW_CHAIN_SOURCE_BINDINGS,
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
        or reviewer
        in {
            "/root",
            IMPLEMENTATION_AUTHOR,
            "/root/coordinator_v2_qa",
            "/root/raw_v11_builder_auditor_diff",
            "/root/raw_v13_source_review",
            "/root/camera_v12_gate_aligned_implementer",
        }
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
            V3_AMENDMENT_RELATIVE_PATH: V3_AMENDMENT_SHA256,
            V3_TOPOLOGY_CORRECTION_RELATIVE_PATH: (
                V3_TOPOLOGY_CORRECTION_SHA256
            ),
        },
        "frozen_parent_closure": reviewed_source_bindings(),
        "reviewed_model_bindings": {
            MODEL_RELATIVE_PATH: MODEL_SHA256,
            HIERARCHICAL_FIRST_HIT_RELATIVE_PATH: (
                HIERARCHICAL_FIRST_HIT_SHA256
            ),
            GATE_ALIGNED_RASTER_NLL_RELATIVE_PATH: (
                GATE_ALIGNED_RASTER_NLL_SHA256
            ),
        },
        "raw_v13_dataset_use_grant": RAW_DATASET_USE_GRANT,
        "camera_v13_future_bindings_must_remain_unresolved": list(
            CAMERA_V13_UNRESOLVED_BINDING_NAMES
        ),
        "camera_ladder_existing_attempt_count": 1,
        "camera_ladder_future_attempt_count": 7,
        "seed_20260710_n5_reexecution_authorized": False,
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
    bindings = {
        name: FROZEN_RESOLVED_BINDINGS.get(name)
        for name in REQUIRED_BINDING_NAMES
    }
    if required_bindings is not None:
        if set(required_bindings) != set(REQUIRED_BINDING_NAMES):
            raise ValueError("exact manifest required-binding names changed")
        bindings.update(dict(required_bindings))
    if any(
        bindings[name] != expected
        for name, expected in FROZEN_RESOLVED_BINDINGS.items()
    ):
        raise PermissionError("terminal Raw V13 binding changed")
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
        "terminal_raw_v13_bindings": FROZEN_RESOLVED_BINDINGS,
        "raw_v13_dataset_use_grant": RAW_DATASET_USE_GRANT,
        "dataset_use_authorized_for_exact_attempt": ready,
        "camera_v13_future_binding_names": list(
            CAMERA_V13_UNRESOLVED_BINDING_NAMES
        ),
        "camera_ladder_topology": {
            "existing_seed_20260710_n5_attempt_count": 1,
            "future_attempt_count": 7,
            "future_attempts": [
                {"seed": 20260710, "fit_size": 16},
                {"seed": 20260710, "fit_size": 32},
                {"seed": 20260710, "fit_size": 320},
                {"seed": 20260711, "fit_size": 5},
                {"seed": 20260711, "fit_size": 16},
                {"seed": 20260711, "fit_size": 32},
                {"seed": 20260711, "fit_size": 320},
            ],
            "seed_20260710_n5_reexecution_authorized": False,
            "aggregate_rung_count": 8,
            "warm_start_authorized": False,
            "only_migratable_rung": {"seed": 20260710, "fit_size": 320},
        },
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


def average_current_next_b4_scalars(current: object, next_: object) -> float:
    """Freeze the arithmetic that follows two separately computed B=4 losses."""

    return 0.5 * _finite(current, name="current B4 scalar") + 0.5 * _finite(
        next_, name="next B4 scalar"
    )


def average_four_microbatch_scalars(values: Sequence[object]) -> float:
    """Reject synthetic-B16 pooling and average exactly four complete scalars."""

    if isinstance(values, (str, bytes)) or len(values) != ACCUMULATION_STEPS:
        raise ValueError("one update requires exactly four B4 scalar losses")
    normalized = [
        _finite(value, name=f"microbatch scalar {index}")
        for index, value in enumerate(values)
    ]
    return sum(0.25 * value for value in normalized)


def pre_g2_candidate_checkpoint_core(
    *,
    model_config: Mapping[str, Any],
    deployment_state_sha256: str,
    selection: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact development candidate metadata allowed before G2."""

    if not isinstance(model_config, Mapping) or not model_config:
        raise ValueError("pre-G2 model config is missing")
    if not is_sha256(deployment_state_sha256):
        raise ValueError("pre-G2 deployment-state hash is malformed")
    if not isinstance(selection, Mapping) or not isinstance(calibration, Mapping):
        raise TypeError("pre-G2 selection and calibration must be mappings")
    return {
        "schema": PRE_G2_CANDIDATE_CHECKPOINT_SCHEMA,
        "lifecycle_stage": (
            "development_selected_and_calibrated_pending_independent_"
            "exact_reconstruction_and_g2"
        ),
        "checkpoint_kind": "pre_g2_candidate",
        "model_config": dict(model_config),
        "deployment_state_sha256": deployment_state_sha256,
        "selection": dict(selection),
        "calibration": dict(calibration),
        "development_only": True,
        "independent_exact_reconstruction_required": True,
        "g2_attempted": False,
        "g2_gate_receipt": None,
        "post_g2_qualified": False,
        "runtime_ready": False,
        "heldout_authorized": False,
        "runtime_navigation_hardware_authorized": False,
        "production_promotion_deployment_authorized": False,
    }


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
    "CAMERA_V13_LADDER_PREREGISTRATION_RELATIVE_PATH",
    "CAMERA_V13_LADDER_REVIEW_RELATIVE_PATH",
    "CAMERA_V13_N5_GATE_RELATIVE_PATH",
    "CAMERA_V13_SOURCE_REVIEW_RELATIVE_PATH",
    "CAMERA_V13_UNRESOLVED_BINDING_NAMES",
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
    "LOSS_ADAPTER_RELATIVE_PATH",
    "INITIALIZATION_SCHEMA",
    "INPUT_BINDINGS_SCHEMA",
    "JOINT_LOSS_CONTRACT",
    "MICROBATCH_SIZE",
    "OPTIMIZER_CONTRACT",
    "PREFLIGHT_INVENTORY",
    "PREFLIGHT_ROOT_RELATIVE_PATH",
    "PRESENTATION_COUNT",
    "PRE_G2_CANDIDATE_CHECKPOINT_SCHEMA",
    "PRIMARY_V4_SEED",
    "PROVENANCE_ROLES",
    "RAW_SUPERVISION_AUDIT_RELATIVE_PATH",
    "RAW_SUPERVISION_AUDIT_SCHEMA",
    "RAW_SUPERVISION_AUDITOR_RELATIVE_PATH",
    "RAW_SUPERVISION_BUILDER_RELATIVE_PATH",
    "RAW_SUPERVISION_MANIFEST_RELATIVE_PATH",
    "RAW_SUPERVISION_MANIFEST_SCHEMA",
    "RAW_SUPERVISION_ROOT_RELATIVE_PATH",
    "RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH",
    "RAW_AUDITOR_V13_FINGERPRINT_RELATIVE_PATH",
    "RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH",
    "RAW_BUILDER_V9_REVIEW_RELATIVE_PATH",
    "RAW_CHAIN_SOURCE_BINDINGS",
    "RAW_DATASET_USE_GRANT",
    "RAW_V13_PASS_CONTENT_SHA256",
    "RAW_V13_PASS_FILE_SHA256",
    "RAW_V13_SAMPLE_RESULTS_SHA256",
    "RAW_V9_MANIFEST_CONTENT_SHA256",
    "RAW_V9_MANIFEST_FILE_SHA256",
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
    "average_current_next_b4_scalars",
    "average_four_microbatch_scalars",
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
    "pre_g2_candidate_checkpoint_core",
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

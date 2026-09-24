"""Pure frozen contract for Shared JEPA V5 full training V4.

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
IMPLEMENTATION_AUTHOR = "/root/full_training_v4_implementer"

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
V3_IMPLEMENTATION_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v3_implementation_author_"
    "handoff_2026-07-14.md"
)
V3_IMPLEMENTATION_HANDOFF_SHA256 = (
    "f7b273f3941fee2be86121eb24db6544c86a2669dce429570864d5c4b3f0e4d3"
)
V3_ARCHIVAL_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v3_implementation_"
    "independent_review_2026-07-14.json"
)
V3_ARCHIVAL_REVIEW_SHA256 = (
    "d885948a4fd82214b200b8cd122b44bd137bb6a5cf9af8aa42ce5bf916e6a4bb"
)
V3_ARCHIVAL_REVIEW_CONTENT_SHA256 = (
    "4cc0b3983b5451c13326669d0d6d0534119e3638ea910610a7db45f0ce9c37c9"
)
V4_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_successor_amendment_"
    "2026-07-14.md"
)
V4_AMENDMENT_SHA256 = (
    "5d475c0dc15d8a53fee5828492914b7473a299e3a6a5c6de1a738e2d3aebcda9"
)
CAMERA_V14_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_"
    "v14_review_open_order_successor_amendment_2026-07-14.md"
)
CAMERA_V14_AMENDMENT_SHA256 = (
    "39e9f840ede8f245d850b7eaaedf0a007fb5f083923629850ced11c8055cd1f6"
)
CAMERA_V13_TERMINAL_BLOCK_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_"
    "v13_independent_review_2026-07-14.json"
)
CAMERA_V13_TERMINAL_BLOCK_SHA256 = (
    "55ade66e943e3de1328fc63f536239ae3605f7edd6e8b7aae5a9b09bb33bdc3e"
)
CAMERA_V13_TERMINAL_BLOCK_CONTENT_SHA256 = (
    "3125e0ca414d8baf3979cecea0464eee0830738345cf37706420d7d44b335330"
)

FROZEN_V3_IMPLEMENTATION_BINDINGS = {
    "lewm/benchmarks/go2_shared_jepa_v5_full_training_v3_policy.py": (
        "53dac9784ad64e083424f304d1078e7c626e0fb824f45a54e60b6a2ab6fa64d0"
    ),
    "lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v3_loss.py": (
        "c04ab06ea6cbeb069e62915197e6d761dc6c9d9751278fcd16a982191a30b926"
    ),
    "scripts/preflight_go2_shared_jepa_v5_full_training_v3.py": (
        "ee8aa87b7f1663b22fd683d3fabfa5ffa5ce571e64fb97db92cfe4a95700062d"
    ),
    "scripts/verify_go2_shared_jepa_v5_full_training_v3_preflight.py": (
        "d9b4434fd4de9bda608f0cc9f6b634d4a194ab95c04e4df9ecb2071b8dace101"
    ),
    "scripts/execute_go2_shared_jepa_v5_full_training_v3.py": (
        "88b3435337ac3d9a756429c8ea4c67d6211192d6d5ab6e9a18fc61eb67d85d1d"
    ),
    "scripts/train_go2_shared_jepa_v5_full_training_v3.py": (
        "d2045622d847b5c07710e98c29a315332b851d3633817476576730c4caf6ba39"
    ),
    "scripts/verify_go2_shared_jepa_v5_full_training_v3.py": (
        "b85c064a4e2cd437ae82cb63f9ae6f0504bad8ce5606e2fded0215219901ce36"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_full_training_v3_implementation.py": (
        "95ec27e78b902bdcc66b4b3eb8663bd8c8a382249ca5c651cae8d58491532850"
    ),
}

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
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_exact_execution_manifest_"
    "2026-07-14.json"
)
PREFLIGHT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "full_training_v4_preflight"
)
EXACT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v4"
)
CANONICAL_PREFLIGHT_ROOT = ROOT / PREFLIGHT_ROOT_RELATIVE_PATH
CANONICAL_EXACT_ROOT = ROOT / EXACT_ROOT_RELATIVE_PATH
PREFLIGHT_RECEIPT_RELATIVE_PATH = (
    f"{PREFLIGHT_ROOT_RELATIVE_PATH}/gpu_smoke_receipt.json"
)
PREFLIGHT_COMPLETED_RELATIVE_PATH = f"{PREFLIGHT_ROOT_RELATIVE_PATH}/completed.json"
PREFLIGHT_INDEPENDENT_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_preflight_independent_review_"
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

RAW_V13_AUTHORIZATION_SOURCE_ROLE_PATHS = (
    (
        "amendment",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_"
        "self_resolving_cli_successor_amendment_2026-07-14.md",
    ),
    (
        "v12_audit_authorization",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_"
        "authorization_2026-07-14.json",
    ),
    (
        "v12_authorization_witness",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_"
        "authorization_fingerprint_2026-07-14.json",
    ),
    (
        "v12_launch_failure",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_launch_"
        "failure_2026-07-14.json",
    ),
    ("v12_auditor_source", "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v12.py"),
    ("v12_auditor_cli", "scripts/audit_go2_shared_jepa_v5_raw_supervision_v12.py"),
    ("v12_auditor_test", "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v12.py"),
    (
        "v12_auditor_handoff",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_author_"
        "handoff_2026-07-14.md",
    ),
    (
        "v12_auditor_review",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_"
        "independent_review_2026-07-14.json",
    ),
    ("auditor_source", RAW_SUPERVISION_AUDITOR_RELATIVE_PATH),
    ("auditor_cli", "scripts/audit_go2_shared_jepa_v5_raw_supervision_v13.py"),
    ("auditor_test", "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v13.py"),
    (
        "auditor_handoff",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_author_"
        "handoff_2026-07-14.md",
    ),
    ("auditor_review", RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH),
)
RAW_V13_AUTHORIZATION_SOURCE_ROWS = tuple(
    {"role": role, "path": path, "sha256": RAW_CHAIN_SOURCE_BINDINGS[path]}
    for role, path in RAW_V13_AUTHORIZATION_SOURCE_ROLE_PATHS
)
RAW_V13_AUTHORIZATION_SOURCE_MAP_SHA256 = (
    "88f748865ff132bc7afd6fe85def14d7f3180ce86b1304ce195c7220f75b8996"
)

RAW_DATASET_USE_GRANT = {
    "grant": "v4_final_exact_attempt_development_roles_only",
    "roles": ["train", "checkpoint_selection", "probability_calibration"],
    "requires_exact_reservation": True,
    "requires_raw_v13_chain": True,
    "requires_v14_ladder_evidence": True,
    "requires_final_exact_execution_authorization": True,
    "dataset_use_authorized": True,
    "training_selection_calibration_authorized": True,
    "rgb_outside_bound_raw_leaves_authorized": False,
    "g2_or_heldout_authorized": False,
    "runtime_navigation_hardware_authorized": False,
    "production_promotion_deployment_authorized": False,
    "retry_authorized": False,
}

CAMERA_V14_SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_"
    "v14_independent_review_2026-07-14.json"
)
CAMERA_V14_N5_GATE_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n5_gate_aligned_raster_nll_v14/gates/seed_20260710_n5.json"
)
CAMERA_V14_OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n5_gate_aligned_raster_nll_v14"
)
CAMERA_V14_LADDER_PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_gate_aligned_raster_nll_v14_"
    "two_seed_ladder_preregistration_2026-07-14.md"
)
CAMERA_V14_LADDER_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_gate_aligned_raster_nll_v14_"
    "two_seed_ladder_independent_review_2026-07-14.json"
)
CAMERA_V14_TWO_SEED_LADDER_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "gate_aligned_raster_nll_v14_ladder_v1/"
    "gates/two_seed.json"
)
CAMERA_V14_LADDER_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "gate_aligned_raster_nll_v14_ladder_v1"
)
CAMERA_V14_PRIMARY_CHECKPOINT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "gate_aligned_raster_nll_v14_ladder_v1/attempts/"
    "seed_20260710/n320/checkpoint.pt"
)
CAMERA_V14_SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_"
    "source_review_v1"
)
CAMERA_V14_GATE_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v14_"
    "gate_v1"
)
CAMERA_V14_PRODUCTION_SOURCE_PATHS = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v14.py",
    "scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v14.py",
    "scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v14.py",
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v14.py",
)
CAMERA_LADDER_ORDER = (
    (20260710, 5),
    (20260710, 16),
    (20260710, 32),
    (20260710, 320),
    (20260711, 5),
    (20260711, 16),
    (20260711, 32),
    (20260711, 320),
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
    "lewm/benchmarks/go2_shared_jepa_v5_full_training_v4_policy.py"
)
LOSS_ADAPTER_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v4_loss.py"
)
PREFLIGHT_EXECUTOR_RELATIVE_PATH = (
    "scripts/preflight_go2_shared_jepa_v5_full_training_v4.py"
)
PREFLIGHT_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_shared_jepa_v5_full_training_v4_preflight.py"
)
EXACT_EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_shared_jepa_v5_full_training_v4.py"
)
EXACT_TRAINER_RELATIVE_PATH = (
    "scripts/train_go2_shared_jepa_v5_full_training_v4.py"
)
EXACT_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_shared_jepa_v5_full_training_v4.py"
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
AUTHOR_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_shared_jepa_v5_full_training_v4_implementation.py"
)
IMPLEMENTATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_implementation_"
    "independent_review_2026-07-14.json"
)
IMPLEMENTATION_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_implementation_author_"
    "handoff_2026-07-14.md"
)
EXACT_BINDING_PREFLIGHT_AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_exact_binding_preflight_"
    "authorization_2026-07-14.json"
)
EXACT_BINDING_PREFLIGHT_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_exact_binding_preflight_"
    "authorization_independent_review_2026-07-14.json"
)
FINAL_EXACT_EXECUTION_AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_final_exact_execution_"
    "authorization_2026-07-14.json"
)
FINAL_EXACT_EXECUTION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_full_training_v4_final_exact_execution_"
    "authorization_independent_review_2026-07-14.json"
)

EXECUTION_MANIFEST_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_exact_execution_manifest_v1"
)
IMPLEMENTATION_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_source_only_implementation_"
    "review_v1"
)
EXACT_BINDING_PREFLIGHT_AUTHORIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_exact_binding_preflight_"
    "authorization_v1"
)
FINAL_EXACT_EXECUTION_AUTHORIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_final_exact_execution_"
    "authorization_v1"
)
PREFLIGHT_RESERVATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_preflight_reservation_v1"
)
PREFLIGHT_RECEIPT_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_gpu_smoke_receipt_v1"
)
PREFLIGHT_COMPLETION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_preflight_completion_v1"
)
EXACT_RESERVATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_reservation_v1"
)
EXACT_COMPLETION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_completion_v1"
)
EXACT_FAILURE_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_failure_v1"
ACCESS_LEDGER_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_access_ledger_v1"
SCHEDULE_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_schedule_v1"
SOURCE_REVIEW_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_source_review_v1"
INPUT_BINDINGS_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_input_bindings_v1"
INITIALIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_initialization_v1"
TRAINING_RECORD_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_training_record_v1"
SELECTION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_selection_v1"
CALIBRATION_SCHEMA = "lewm_go2_shared_jepa_v5_full_training_v4_calibration_v1"
DIAGNOSTIC_ABLATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_selection_role_ablation_diagnostic_v1"
)
PRE_G2_CANDIDATE_CHECKPOINT_SCHEMA = (
    "lewm_go2_shared_jepa_v5_full_training_v4_pre_g2_candidate_checkpoint_v1"
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
RAW_TOTAL_COUNTS = {
    "pairs": 5172,
    "endpoint_references": 10344,
    "unique_endpoints": 9460,
    "scene_shards": 88,
}
RAW_ORDERED_PAIR_SHA256 = (
    "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
)
RAW_ORDERED_ENDPOINT_SHA256 = (
    "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
)
RAW_ARRAY_LAYOUT = (
    {"path": "camera_origin_body_m.f4", "dtype": "<f4", "trailing_shape": [3]},
    {"path": "camera_basis_body_fru.f4", "dtype": "<f4", "trailing_shape": [3, 3]},
    {"path": "ground_plane_z_body_m.f4", "dtype": "<f4", "trailing_shape": []},
    {"path": "ground_support_in_frustum.u1", "dtype": "|u1", "trailing_shape": [128, 128, 5]},
    {"path": "ground_support_clear_to_target.u1", "dtype": "|u1", "trailing_shape": [128, 128, 5]},
    {"path": "pixel_hit_mask.u1", "dtype": "|u1", "trailing_shape": [84, 112]},
    {"path": "pixel_first_hit_distance_m.f4", "dtype": "<f4", "trailing_shape": [84, 112]},
    {"path": "raster_labels.u1", "dtype": "|u1", "trailing_shape": [64, 64]},
)
RAW_MANIFEST_FIELDS = frozenset(
    {
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
)
RAW_DOWNSTREAM_AUTHORITY_FIELDS = (
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
RAW_REPORT_FIELDS = frozenset(
    {
        "schema",
        "verdict",
        "dataset_manifest_file_sha256",
        "dataset_manifest_content_sha256",
        "pair_count",
        "unique_endpoint_count",
        "scene_shard_count",
        "sample_count",
        "sample_results",
        "sample_results_sha256",
        "observed_population",
        "strict_integer_cardinalities",
        "unaliased_descriptor_bound_dataset_leaves",
        "full_byte_inventory_revalidated",
        "pair_endpoint_joins_reconstructed",
        "all_stored_evidence_and_rasters_recomputed",
        "sample_original_geometry_recomputed",
        "source_file_count",
        "source_inventory_before_after_sha256",
        "source_payload_opens",
        "authorization_v13",
        "frozen_v9_terminal_artifacts",
        "frozen_v10_terminal_artifacts",
        "frozen_v11_terminal_artifacts",
        "frozen_v12_terminal_artifacts",
        "closed_publication_transaction_v13",
        "content_sha256",
        *RAW_DOWNSTREAM_AUTHORITY_FIELDS,
    }
)
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
    "final_exact_authorization",
    "implementation_review",
    "preflight_receipt",
    "raw_supervision_manifest",
    "raw_supervision_audit",
    "camera_v14_two_seed_ladder",
    "camera_v14_primary_checkpoint",
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

BLOCKED_FUTURE_BINDING_NAMES = (
    "v4_implementation_handoff_file_sha256",
    "v4_implementation_review_file_sha256",
    "v4_implementation_review_content_sha256",
    "camera_v14_source_review_file_sha256",
    "camera_v14_source_review_content_sha256",
    "camera_v14_reviewed_source_bindings",
    "camera_v14_n5_gate_file_sha256",
    "camera_v14_n5_gate_content_sha256",
    "camera_v14_n5_completion_file_sha256",
    "camera_v14_n5_completion_content_sha256",
    "camera_v14_n5_checkpoint_file_sha256",
    "camera_ladder_preregistration_file_sha256",
    "camera_ladder_preregistration_review_file_sha256",
    "camera_ladder_preregistration_review_content_sha256",
    "camera_ladder_rows",
    "camera_ladder_aggregate_gate_file_sha256",
    "camera_ladder_aggregate_gate_content_sha256",
    "camera_ladder_independent_review_file_sha256",
    "camera_ladder_independent_review_content_sha256",
    "camera_primary_seed_20260710_n320_checkpoint_file_sha256",
    "v4_exact_binding_preflight_authorization_file_sha256",
    "v4_exact_binding_preflight_authorization_content_sha256",
    "v4_exact_binding_preflight_review_file_sha256",
    "v4_exact_binding_preflight_review_content_sha256",
    "v4_preflight_receipt_file_sha256",
    "v4_preflight_receipt_content_sha256",
    "v4_preflight_verification_file_sha256",
    "v4_preflight_verification_content_sha256",
    "v4_final_exact_execution_authorization_file_sha256",
    "v4_final_exact_execution_authorization_content_sha256",
    "v4_final_exact_execution_review_file_sha256",
    "v4_final_exact_execution_review_content_sha256",
)
SOURCE_REVIEW_AUTHORITY = {
    "payload_free_preflight_execution_authorized": False,
    "exact_execution_authorized": False,
    "dataset_or_checkpoint_access_authorized": False,
    "gpu_or_accelerator_authorized": False,
    "shared_jepa_training_authorized": False,
    "selection_authorized": False,
    "calibration_authorized": False,
    "g2_authorized": False,
    "heldout_authorized": False,
    "navigation_authorized": False,
    "runtime_authorized": False,
    "hardware_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_authorized": False,
}
MANIFEST_AUTHORITY = {
    "dataset_use_authorized": False,
    **SOURCE_REVIEW_AUTHORITY,
}
PREFLIGHT_AUTHORITY = {
    "payload_free_preflight_execution_authorized": True,
    "exact_execution_authorized": False,
    "dataset_or_checkpoint_access_authorized": False,
    "gpu_or_accelerator_training_authorized": False,
    "shared_jepa_training_authorized": False,
    "selection_authorized": False,
    "calibration_authorized": False,
    "g2_authorized": False,
    "heldout_authorized": False,
    "navigation_authorized": False,
    "runtime_authorized": False,
    "hardware_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_authorized": False,
}
RAW_V13_TERMINAL_BINDINGS = {
    "builder_v9_source_file_sha256": RAW_CHAIN_SOURCE_BINDINGS[
        RAW_SUPERVISION_BUILDER_RELATIVE_PATH
    ],
    "auditor_v13_source_file_sha256": RAW_CHAIN_SOURCE_BINDINGS[
        RAW_SUPERVISION_AUDITOR_RELATIVE_PATH
    ],
    "dataset_manifest_file_sha256": RAW_V9_MANIFEST_FILE_SHA256,
    "dataset_manifest_content_sha256": RAW_V9_MANIFEST_CONTENT_SHA256,
    "terminal_report_file_sha256": RAW_V13_PASS_FILE_SHA256,
    "terminal_report_content_sha256": RAW_V13_PASS_CONTENT_SHA256,
    "sample_results_sha256": RAW_V13_SAMPLE_RESULTS_SHA256,
}
FINAL_REQUIRED_BINDING_NAMES = (
    "development_raw_supervision_manifest_file_sha256",
    "development_raw_supervision_manifest_content_sha256",
    "development_raw_supervision_builder_source_sha256",
    "development_raw_supervision_auditor_source_sha256",
    "development_raw_supervision_audit_file_sha256",
    "development_raw_supervision_audit_content_sha256",
    "camera_v14_source_review_file_sha256",
    "camera_v14_source_review_content_sha256",
    "camera_v14_n5_gate_pass_file_sha256",
    "camera_v14_n5_gate_pass_content_sha256",
    "camera_v14_ladder_preregistration_file_sha256",
    "camera_v14_ladder_independent_review_file_sha256",
    "camera_v14_two_seed_ladder_pass_file_sha256",
    "camera_v14_two_seed_ladder_pass_content_sha256",
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
FINAL_FROZEN_RAW_BINDINGS = {
    "development_raw_supervision_manifest_file_sha256": RAW_V9_MANIFEST_FILE_SHA256,
    "development_raw_supervision_manifest_content_sha256": RAW_V9_MANIFEST_CONTENT_SHA256,
    "development_raw_supervision_builder_source_sha256": RAW_CHAIN_SOURCE_BINDINGS[RAW_SUPERVISION_BUILDER_RELATIVE_PATH],
    "development_raw_supervision_auditor_source_sha256": RAW_CHAIN_SOURCE_BINDINGS[RAW_SUPERVISION_AUDITOR_RELATIVE_PATH],
    "development_raw_supervision_audit_file_sha256": RAW_V13_PASS_FILE_SHA256,
    "development_raw_supervision_audit_content_sha256": RAW_V13_PASS_CONTENT_SHA256,
}
FINAL_EXACT_AUTHORITY = {
    "payload_free_preflight_execution_authorized": False,
    "exact_execution_authorized": True,
    "dataset_or_checkpoint_access_authorized": True,
    "gpu_or_accelerator_training_authorized": True,
    "shared_jepa_training_authorized": True,
    "selection_authorized": True,
    "calibration_authorized": True,
    "g2_authorized": False,
    "heldout_authorized": False,
    "navigation_authorized": False,
    "runtime_authorized": False,
    "hardware_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
    "retry_authorized": False,
}

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
        V3_IMPLEMENTATION_HANDOFF_RELATIVE_PATH: V3_IMPLEMENTATION_HANDOFF_SHA256,
        V3_ARCHIVAL_REVIEW_RELATIVE_PATH: V3_ARCHIVAL_REVIEW_SHA256,
        V4_AMENDMENT_RELATIVE_PATH: V4_AMENDMENT_SHA256,
        CAMERA_V14_AMENDMENT_RELATIVE_PATH: CAMERA_V14_AMENDMENT_SHA256,
        CAMERA_V13_TERMINAL_BLOCK_RELATIVE_PATH: CAMERA_V13_TERMINAL_BLOCK_SHA256,
        MODEL_RELATIVE_PATH: MODEL_SHA256,
        HIERARCHICAL_FIRST_HIT_RELATIVE_PATH: HIERARCHICAL_FIRST_HIT_SHA256,
        GATE_ALIGNED_RASTER_NLL_RELATIVE_PATH: GATE_ALIGNED_RASTER_NLL_SHA256,
        MODEL_TEST_RELATIVE_PATH: MODEL_TEST_SHA256,
        OUTPUT_LOSS_REVIEW_RELATIVE_PATH: OUTPUT_LOSS_REVIEW_SHA256,
        LIFECYCLE_REVIEW_RELATIVE_PATH: LIFECYCLE_REVIEW_SHA256,
        **FROZEN_GOVERNING_DESIGN_BINDINGS,
        **FROZEN_V2_IMPLEMENTATION_BINDINGS,
        **FROZEN_V3_IMPLEMENTATION_BINDINGS,
        **RAW_CHAIN_SOURCE_BINDINGS,
        **REVIEWED_LIFECYCLE_BINDINGS,
    }


def expected_implementation_review_core(
    *,
    reviewer: str,
    source_bindings: Mapping[str, str],
    author_test_file_sha256: str,
    handoff_file_sha256: str,
    blocked_manifest_file_sha256: str,
    blocked_manifest_content_sha256: str,
) -> dict[str, Any]:
    excluded = {
        "/root",
        IMPLEMENTATION_AUTHOR,
        "/root/full_training_v4_contract",
        "/root/full_training_v3",
        "/root/full_training_v3_independent_review",
        "/root/raw_v11_builder_auditor_diff",
        "/root/raw_v13_source_review",
        "/root/camera_v12_gate_aligned_implementer",
        "/root/camera_v13_independent_review",
    }
    if type(reviewer) is not str or not reviewer.startswith("/root/") or reviewer in excluded:
        raise PermissionError("implementation review must be by a different eligible agent")
    if type(source_bindings) is not dict or set(source_bindings) != set(IMPLEMENTATION_SOURCE_PATHS):
        raise ValueError("implementation review production closure changed")
    if any(type(path) is not str or not is_sha256(value) for path, value in source_bindings.items()):
        raise ValueError("implementation review production hash changed")
    for value in (
        author_test_file_sha256,
        handoff_file_sha256,
        blocked_manifest_file_sha256,
        blocked_manifest_content_sha256,
    ):
        if not is_sha256(value):
            raise ValueError("implementation review proof binding is malformed")
    return {
        "schema": IMPLEMENTATION_REVIEW_SCHEMA,
        "status": "different_agent_source_review_passed_no_execution_authority",
        "implementation_author": IMPLEMENTATION_AUTHOR,
        "reviewer": reviewer,
        "reviewed_production_sources": dict(source_bindings),
        "author_test": {
            "path": AUTHOR_TEST_RELATIVE_PATH,
            "file_sha256": author_test_file_sha256,
        },
        "author_handoff": {
            "path": IMPLEMENTATION_HANDOFF_RELATIVE_PATH,
            "file_sha256": handoff_file_sha256,
        },
        "blocked_manifest": {
            "path": EXACT_EXECUTION_MANIFEST_RELATIVE_PATH,
            "file_sha256": blocked_manifest_file_sha256,
            "content_sha256": blocked_manifest_content_sha256,
        },
        "governing_amendment": {
            "path": V4_AMENDMENT_RELATIVE_PATH,
            "file_sha256": V4_AMENDMENT_SHA256,
        },
        "frozen_parent_closure": reviewed_source_bindings(),
        "raw_v13_terminal_bindings": RAW_V13_TERMINAL_BINDINGS,
        "raw_v13_dataset_use_grant_present_but_inactive": RAW_DATASET_USE_GRANT,
        "camera_dependency": {
            "version": 14,
            "v13_terminal_status": "blocked_changed_digest_zero_open_contract_unsatisfied",
            "ordered_ladder": [list(item) for item in CAMERA_LADDER_ORDER],
            "preexisting_attempt_count": 1,
            "additional_attempt_count": 7,
            "seed_20260710_n5_reexecution_authorized": False,
            "only_migratable_row": 3,
        },
        "blocked_future_bindings": list(BLOCKED_FUTURE_BINDING_NAMES),
        "authority": SOURCE_REVIEW_AUTHORITY,
    }


def validate_implementation_review(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("implementation review must be a plain dictionary")
    copied = dict(value)
    declared = copied.pop("content_sha256", None)
    sources = copied.get("reviewed_production_sources")
    reviewer = copied.get("reviewer")
    author_test = copied.get("author_test")
    handoff = copied.get("author_handoff")
    manifest = copied.get("blocked_manifest")
    if (
        type(sources) is not dict
        or type(reviewer) is not str
        or type(author_test) is not dict
        or type(handoff) is not dict
        or type(manifest) is not dict
    ):
        raise ValueError("implementation review bindings are missing")
    expected = expected_implementation_review_core(
        reviewer=reviewer,
        source_bindings=sources,
        author_test_file_sha256=author_test.get("file_sha256"),
        handoff_file_sha256=handoff.get("file_sha256"),
        blocked_manifest_file_sha256=manifest.get("file_sha256"),
        blocked_manifest_content_sha256=manifest.get("content_sha256"),
    )
    if (
        copied != expected
        or not is_sha256(declared)
        or canonical_json_sha256(copied) != declared
    ):
        raise PermissionError("implementation review contract changed")
    return {**copied, "content_sha256": declared}


def execution_manifest_core(
) -> dict[str, Any]:
    bindings = {name: None for name in BLOCKED_FUTURE_BINDING_NAMES}
    return {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "status": "blocked_required_bindings_unset",
        "governing_amendment": {
            "path": V4_AMENDMENT_RELATIVE_PATH,
            "file_sha256": V4_AMENDMENT_SHA256,
        },
        "stable_source_bindings": reviewed_source_bindings(),
        "terminal_raw_v13_bindings": RAW_V13_TERMINAL_BINDINGS,
        "future_bindings": bindings,
        "unresolved_future_bindings": list(BLOCKED_FUTURE_BINDING_NAMES),
        "camera_v14_ordered_ladder": [list(item) for item in CAMERA_LADDER_ORDER],
        "camera_v13_evidence_accepted": False,
        "source_review_can_authorize_preflight": False,
        "preflight_root": PREFLIGHT_ROOT_RELATIVE_PATH,
        "exact_root": EXACT_ROOT_RELATIVE_PATH,
        "preflight_and_exact_processes_distinct": True,
        "exact_reservation_before_torch_model_or_payload": True,
        **MANIFEST_AUTHORITY,
    }


def validate_execution_manifest(
    value: Mapping[str, Any],
    *,
    require_ready: bool,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("exact execution manifest must be a plain dictionary")
    copied = dict(value)
    declared = copied.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(copied) != declared:
        raise ValueError("exact execution manifest content hash changed")
    expected = execution_manifest_core()
    if copied != expected:
        raise PermissionError("exact execution manifest contract changed")
    if require_ready:
        unresolved = ", ".join(expected["unresolved_future_bindings"])
        raise PermissionError(
            "blocked source-time manifest cannot authorize reservation or payload; unset: "
            + unresolved
        )
    return {**copied, "content_sha256": declared}


def _plain_dict(value: object, *, fields: set[str] | frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{name} must be a plain dictionary with the exact key set")
    return value


def _canonical_relative_path_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty plain string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise PermissionError(f"{name} must be an exact canonical relative path")
    return value


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative plain integer")
    return value


def validate_raw_v13_source_chain(
    *,
    builder_review: object,
    auditor_review: object,
    authorization: object,
    fingerprint: object,
) -> dict[str, Any]:
    """Validate the complete source-only Raw V13 authorization chain."""

    builder = _plain_dict(
        builder_review,
        fields={"authority", "candidate", "content_sha256", "implementation_author", "reviewer", "schema", "verdict"},
        name="Raw V9 Builder review",
    )
    builder_candidate = [
        {"role": "builder_source", "path": RAW_SUPERVISION_BUILDER_RELATIVE_PATH, "sha256": RAW_CHAIN_SOURCE_BINDINGS[RAW_SUPERVISION_BUILDER_RELATIVE_PATH]},
        {"role": "builder_cli", "path": "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v9.py", "sha256": RAW_CHAIN_SOURCE_BINDINGS["scripts/build_go2_shared_jepa_v5_development_raw_supervision_v9.py"]},
        {"role": "builder_test", "path": "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py", "sha256": RAW_CHAIN_SOURCE_BINDINGS["lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py"]},
        {"role": "builder_handoff", "path": "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_author_handoff_2026-07-13.md", "sha256": RAW_CHAIN_SOURCE_BINDINGS["docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_author_handoff_2026-07-13.md"]},
    ]
    builder_false = {
        "calibration_authorized",
        "dataset_use_authorized",
        "exact_audit_authorized",
        "exact_build_authorized",
        "g2_authorized",
        "hardware_authorized",
        "heldout_authorized",
        "navigation_authorized",
        "production_authorized",
        "promotion_authorized",
        "retry_authorized",
        "runtime_authorized",
        "selection_authorized",
        "training_authorized",
    }
    if (
        builder["schema"] != "lewm_go2_shared_jepa_v5_raw_supervision_builder_v9_independent_review_v1"
        or builder["verdict"] != "PASS"
        or builder["implementation_author"]
        != "/root/raw_v7_successor_author/auditor_v7_author"
        or builder["reviewer"] != "/root/raw_v8_auditor_reviewer"
        or builder["content_sha256"] != "49d8024ae48211cc4fc7d7c2fb674c7ddc7adb38abccace1eb8c6bbc4f10b0df"
        or canonical_json_sha256(
            {name: item for name, item in builder.items() if name != "content_sha256"}
        )
        != builder["content_sha256"]
        or builder["candidate"] != builder_candidate
        or builder["authority"] != {"builder_source_approved": True, **{field: False for field in builder_false}}
    ):
        raise PermissionError("Raw V9 Builder review semantics changed")

    auditor = _plain_dict(
        auditor_review,
        fields={"authority", "candidate", "content_sha256", "implementation_author", "reviewer", "schema", "verdict"},
        name="Raw V13 Auditor review",
    )
    auditor_candidate = [
        {"role": "auditor_source", "path": RAW_SUPERVISION_AUDITOR_RELATIVE_PATH, "sha256": RAW_CHAIN_SOURCE_BINDINGS[RAW_SUPERVISION_AUDITOR_RELATIVE_PATH]},
        {"role": "auditor_cli", "path": "scripts/audit_go2_shared_jepa_v5_raw_supervision_v13.py", "sha256": RAW_CHAIN_SOURCE_BINDINGS["scripts/audit_go2_shared_jepa_v5_raw_supervision_v13.py"]},
        {"role": "auditor_test", "path": "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v13.py", "sha256": RAW_CHAIN_SOURCE_BINDINGS["lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v13.py"]},
        {"role": "auditor_handoff", "path": "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_author_handoff_2026-07-14.md", "sha256": RAW_CHAIN_SOURCE_BINDINGS["docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_author_handoff_2026-07-14.md"]},
    ]
    auditor_false = {
        "calibration_authorized", "dataset_use_authorized", "deployment_authorized",
        "exact_audit_v10_authorized", "exact_audit_v11_authorized", "exact_audit_v12_authorized",
        "exact_audit_v13_authorized", "exact_audit_v9_authorized", "exact_build_authorized",
        "exact_rebuild_authorized", "g2_authorized", "hardware_authorized", "heldout_authorized",
        "navigation_authorized", "production_authorized", "promotion_authorized", "retry_authorized",
        "rgb_decode_authorized", "runtime_authorized", "selection_authorized", "training_authorized",
    }
    if (
        auditor["schema"] != "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_independent_review_v1"
        or auditor["verdict"] != "PASS"
        or auditor["implementation_author"] != "/root/raw_v11_builder_auditor_diff"
        or auditor["reviewer"] != "/root/raw_v13_source_review"
        or auditor["content_sha256"] != "a338b5af5b437d1a046c3e3e95e761cca30a65c7c67d7736f7f0b17f1ec8639c"
        or canonical_json_sha256(
            {name: item for name, item in auditor.items() if name != "content_sha256"}
        )
        != auditor["content_sha256"]
        or auditor["candidate"] != auditor_candidate
        or auditor["authority"] != {"auditor_source_approved": True, **{field: False for field in auditor_false}}
    ):
        raise PermissionError("Raw V13 Auditor review semantics changed")

    authorization_false = {
        "exact_audit_v9_authorized", "exact_audit_v10_authorized", "exact_audit_v11_authorized",
        "exact_audit_v12_authorized", "exact_build_authorized", "exact_rebuild_authorized",
        "retry_authorized", "rgb_decode_authorized", "dataset_use_authorized", "training_authorized",
        "selection_authorized", "calibration_authorized", "g2_authorized", "heldout_authorized",
        "runtime_authorized", "navigation_authorized", "hardware_authorized", "production_authorized",
        "promotion_authorized", "deployment_authorized",
    }
    authorization_fields = {
        "schema", "exact_audit_v13_authorized", "input_dataset_path", "success_report_path",
        "failure_report_path", "v12_audit_authorization_file_sha256",
        "v12_audit_authorization_content_sha256", "v12_audit_authorization_source_map_sha256",
        "v12_authorization_witness_file_sha256", "v12_authorization_witness_content_sha256",
        "v12_launch_failure_file_sha256", "v12_launch_failure_content_sha256",
        "v9_dataset_manifest_file_sha256", "v9_dataset_manifest_content_sha256",
        "auditor_review", "source_map", "content_sha256", *authorization_false,
    }
    auth = _plain_dict(authorization, fields=authorization_fields, name="Raw V13 authorization")
    expected_review_binding = {
        "schema": "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v13",
        "review_schema": "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_independent_review_v1",
        "verdict": "PASS",
        "reviewer": auditor["reviewer"],
        "implementation_author": auditor["implementation_author"],
        "path": RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH,
        "file_sha256": RAW_CHAIN_SOURCE_BINDINGS[RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH],
        "content_sha256": auditor["content_sha256"],
        "candidate": auditor_candidate,
    }
    if (
        auth["schema"] != "lewm_go2_shared_jepa_v5_raw_supervision_audit_authorization_v13"
        or auth["content_sha256"] != "4b179c33de00399652f4f915285ca99a4d47cfa95d31878d1f91ca7e8fd9d0e8"
        or canonical_json_sha256(
            {name: item for name, item in auth.items() if name != "content_sha256"}
        )
        != auth["content_sha256"]
        or auth["exact_audit_v13_authorized"] is not True
        or any(auth[field] is not False for field in authorization_false)
        or auth["input_dataset_path"] != RAW_SUPERVISION_ROOT_RELATIVE_PATH
        or auth["success_report_path"] != RAW_SUPERVISION_AUDIT_RELATIVE_PATH
        or auth["failure_report_path"] != RAW_SUPERVISION_ROOT_RELATIVE_PATH + ".audit_v13.failed.json"
        or auth["v9_dataset_manifest_file_sha256"] != RAW_V9_MANIFEST_FILE_SHA256
        or auth["v9_dataset_manifest_content_sha256"] != RAW_V9_MANIFEST_CONTENT_SHA256
        or auth["v12_audit_authorization_file_sha256"]
        != "6b5f317119a00308390b8a32f1057f34455313eb80ec190aa9d8d27052a81575"
        or auth["v12_audit_authorization_content_sha256"]
        != "8db4611a321309a76a0dd81e3af0148fce788422e2008ccaef1039e3c5ae493a"
        or auth["v12_audit_authorization_source_map_sha256"]
        != "1fc3374101fca166fe74b34b779cf995ec46a12fbb609f5de3bc5a428d225bc2"
        or auth["v12_authorization_witness_file_sha256"]
        != "662e6c2f6386b8822b3bd968a4faf0bf3e2e222ff4aac9df8a99cc680c254327"
        or auth["v12_authorization_witness_content_sha256"]
        != "4845826d1caeedc58d01b580a8681a71730eb0ba17205bde36d3673c9052741b"
        or auth["v12_launch_failure_file_sha256"]
        != "cc6313b1d6e56022204ba82dc57efc6b7cc85a715f078cd865883b61cee88eb3"
        or auth["v12_launch_failure_content_sha256"]
        != "b9775ef4705d7505931b64c7ceaad57fb8d18da72429bb877245fb534197b2ee"
        or auth["auditor_review"] != expected_review_binding
        or auth["source_map"] != list(RAW_V13_AUTHORIZATION_SOURCE_ROWS)
        or canonical_json_sha256(auth["source_map"]) != RAW_V13_AUTHORIZATION_SOURCE_MAP_SHA256
    ):
        raise PermissionError("Raw V13 authorization semantics changed")

    witness = _plain_dict(
        fingerprint,
        fields={"authority", "authorization", "content_sha256", "implementation_author", "publisher", "reviewer", "schema", "verification", "witness"},
        name="Raw V13 authorization fingerprint",
    )
    if (
        witness["schema"] != "lewm_go2_shared_jepa_v5_raw_supervision_audit_v13_authorization_fingerprint_v1"
        or witness["content_sha256"] != "f5c8f84478929b0ee4753bf7f4531ebaebc88eeb6ba2cc9d9718945f44107e2c"
        or canonical_json_sha256(
            {name: item for name, item in witness.items() if name != "content_sha256"}
        )
        != witness["content_sha256"]
        or witness["implementation_author"] != "/root/raw_v11_builder_auditor_diff"
        or witness["publisher"] != "/root"
        or witness["reviewer"] != "/root/raw_v13_source_review"
        or witness["witness"]
        != "/root/camera_v10_later_rung_plan/v11_adapter_design"
        or witness["authority"] != {field: False for field in auditor_false}
        or witness["authorization"] != {
            "byte_count": 6114,
            "content_sha256": auth["content_sha256"],
            "file_sha256": RAW_CHAIN_SOURCE_BINDINGS[RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH],
            "path": RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH,
            "source_map_sha256": RAW_V13_AUTHORIZATION_SOURCE_MAP_SHA256,
        }
        or witness["verification"]
        != {
            "authority_booleans_exact": True,
            "authorization_actual_matches_independent_expected_bytes": True,
            "authorization_opened_after_expectation_constructed": True,
            "candidate_bytes_authored_or_reviewed_by_witness": False,
            "candidate_or_authorization_edited": False,
            "canonical_dataset_opened": False,
            "canonical_one_line_terminal_newline": True,
            "downstream_authority_granted": False,
            "exact_14_role_order_paths_hashes": True,
            "exact_audit_run": False,
            "expectation_source": (
                "frozen_v13_source_review_and_v12_authorization_witness_"
                "launch_chain"
            ),
            "generated_mapped_target_opened": False,
            "gpu_used": False,
            "phase_one_zero_target_opens": True,
            "review_binding_exact": True,
            "rgb_opened": False,
            "transitive_parent_commitments_exact": True,
        }
    ):
        raise PermissionError("Raw V13 authorization fingerprint semantics changed")
    return {"builder_review": builder, "auditor_review": auditor, "authorization": auth, "fingerprint": witness}


def validate_raw_v13_manifest(value: object) -> dict[str, Any]:
    manifest = _plain_dict(value, fields=RAW_MANIFEST_FIELDS, name="Raw V13 dataset manifest")
    expected_pairs = {role: ROLE_COUNTS[role]["pairs"] for role in DEVELOPMENT_ROLES}
    expected_unique = {role: ROLE_COUNTS[role]["unique_endpoints"] for role in DEVELOPMENT_ROLES}
    if (
        manifest["schema"] != RAW_SUPERVISION_MANIFEST_SCHEMA
        or manifest["status"] != "complete_pending_independent_audit"
        or manifest["content_sha256"] != RAW_V9_MANIFEST_CONTENT_SHA256
        or manifest["evidence_schema"] != "lewm_go2_observable_camera_ray_evidence_v4"
        or manifest["raster_schema"] != "lewm_go2_observable_camera_ray_raster_v4"
        or manifest["roles"] != list(DEVELOPMENT_ROLES)
        or manifest["pair_counts"] != expected_pairs
        or manifest["endpoint_instance_count"] != RAW_TOTAL_COUNTS["endpoint_references"]
        or manifest["unique_endpoint_counts"] != expected_unique
        or manifest["scene_shard_count"] != RAW_TOTAL_COUNTS["scene_shards"]
        or manifest["ordered_pair_sha256"] != RAW_ORDERED_PAIR_SHA256
        or manifest["ordered_endpoint_sha256"] != RAW_ORDERED_ENDPOINT_SHA256
        or manifest["array_layout"] != list(RAW_ARRAY_LAYOUT)
    ):
        raise PermissionError("Raw V13 dataset manifest commitments changed")
    pair_index = _plain_dict(
        manifest["pair_index"],
        fields={"path", "row_count", "file_sha256"},
        name="Raw V13 pair index binding",
    )
    endpoint_index = _plain_dict(
        manifest["endpoint_index"],
        fields={"path", "row_count", "file_sha256"},
        name="Raw V13 endpoint index binding",
    )
    if (
        pair_index["path"] != "pairs.jsonl"
        or pair_index["row_count"] != RAW_TOTAL_COUNTS["pairs"]
        or not is_sha256(pair_index["file_sha256"])
        or endpoint_index["path"] != "endpoints.jsonl"
        or endpoint_index["row_count"] != RAW_TOTAL_COUNTS["unique_endpoints"]
        or not is_sha256(endpoint_index["file_sha256"])
    ):
        raise PermissionError("Raw V13 pair/endpoint index population changed")
    if manifest["parallel_contract"] != {
        "worker_start_method": "spawn",
        "maximum_workers": 6,
        "native_threads_per_worker": 1,
        "gpu_visible_to_workers": False,
        "merge_order": "role_then_scene_then_endpoint_identity",
        "worker_count_does_not_change_artifact_bytes": True,
    }:
        raise PermissionError("Raw V13 parallel construction contract changed")
    if manifest["publication"] != {
        "staging": "private_sibling_directory_mode_0700",
        "commit": "single_renameat2_RENAME_NOREPLACE",
        "manifest_self_inventory": "canonical_content_sha256",
        "file_inventory": "every_regular_file_except_manifest_self",
    }:
        raise PermissionError("Raw V13 publication contract changed")
    manifest_license_fields = {
        "independent_audit_passed",
        "dataset_use_authorized",
        "rgb_decode_authorized",
        "training_authorized",
        "selection_authorized",
        "calibration_authorized",
        "g2_authorized",
        "heldout_authorized",
        "runtime_authorized",
        "hardware_authorized",
        "production_authorized",
        "promotion_authorized",
    }
    if (
        type(manifest["licenses"]) is not dict
        or set(manifest["licenses"]) != manifest_license_fields
        or any(item is not False for item in manifest["licenses"].values())
    ):
        raise PermissionError("Raw V13 unaudited manifest authority changed")
    precommit = _plain_dict(
        manifest["independent_audit_precommit"],
        fields={"scheme", "one_endpoint_per_observed_role_family", "expected_exact_record_count", "records", "records_sha256"},
        name="Raw V13 sample precommit",
    )
    if (
        precommit["scheme"]
        != "minimum_sha256_role_nul_family_nul_endpoint_identity_v1"
        or precommit["one_endpoint_per_observed_role_family"] is not True
        or precommit["expected_exact_record_count"] != 24
        or type(precommit["records"]) is not list
        or len(precommit["records"]) != 24
    ):
        raise PermissionError("Raw V13 sample precommit count changed")
    sample_pairs = []
    for index, row in enumerate(precommit["records"]):
        record = _plain_dict(row, fields={"dataset_role", "family", "endpoint_identity_sha256", "selection_sha256"}, name=f"Raw V13 sample precommit row {index}")
        if record["dataset_role"] not in DEVELOPMENT_ROLES or record["family"] not in FAMILIES or not is_sha256(record["endpoint_identity_sha256"]) or not is_sha256(record["selection_sha256"]):
            raise PermissionError("Raw V13 sample precommit row changed")
        sample_pairs.append((record["dataset_role"], record["family"]))
    if sample_pairs != [
        (role, family)
        for role in DEVELOPMENT_ROLES
        for family in FAMILIES
    ]:
        raise PermissionError("Raw V13 samples do not cover every role/family")
    if canonical_json_sha256(precommit["records"]) != precommit["records_sha256"]:
        raise PermissionError("Raw V13 sample-precommit hash changed")
    if type(manifest["shards"]) is not list or len(manifest["shards"]) != 88:
        raise PermissionError("Raw V13 scene-shard count changed")
    shard_counts = {role: 0 for role in DEVELOPMENT_ROLES}
    shard_endpoint_counts = {role: 0 for role in DEVELOPMENT_ROLES}
    shard_families = {role: set() for role in DEVELOPMENT_ROLES}
    scenes: set[str] = set()
    previous_scene = ""
    for index, row in enumerate(manifest["shards"]):
        shard = _plain_dict(row, fields={"path", "dataset_role", "family", "scene_id", "endpoint_count", "content_sha256"}, name=f"Raw V13 shard {index}")
        role, family, scene = shard["dataset_role"], shard["family"], shard["scene_id"]
        expected_path = (
            f"shards/{hashlib.sha256(str(scene).encode('utf-8')).hexdigest()[:16]}"
            "/shard.json"
        )
        if role not in DEVELOPMENT_ROLES or family not in FAMILIES or type(scene) is not str or scene <= previous_scene or scene in scenes or shard["path"] != expected_path or not is_sha256(shard["content_sha256"]):
            raise PermissionError("Raw V13 shard identity changed")
        endpoint_count = _exact_nonnegative_int(
            shard["endpoint_count"],
            name="Raw V13 shard endpoint count",
        )
        if endpoint_count == 0:
            raise PermissionError("Raw V13 shard endpoint population is empty")
        previous_scene = scene
        scenes.add(scene)
        shard_counts[role] += 1
        shard_endpoint_counts[role] += endpoint_count
        shard_families[role].add(family)
    if (
        shard_counts
        != {role: ROLE_COUNTS[role]["scenes"] for role in DEVELOPMENT_ROLES}
        or shard_endpoint_counts != expected_unique
        or any(values != set(FAMILIES) for values in shard_families.values())
    ):
        raise PermissionError("Raw V13 per-role shard/family population changed")
    return manifest


def validate_raw_v13_terminal_report(value: object) -> dict[str, Any]:
    report = _plain_dict(value, fields=RAW_REPORT_FIELDS, name="Raw V13 terminal report")
    if (
        report["schema"] != RAW_SUPERVISION_AUDIT_SCHEMA
        or report["verdict"] != "PASS"
        or report["content_sha256"] != RAW_V13_PASS_CONTENT_SHA256
        or report["dataset_manifest_file_sha256"] != RAW_V9_MANIFEST_FILE_SHA256
        or report["dataset_manifest_content_sha256"] != RAW_V9_MANIFEST_CONTENT_SHA256
        or report["pair_count"] != RAW_TOTAL_COUNTS["pairs"]
        or report["unique_endpoint_count"] != RAW_TOTAL_COUNTS["unique_endpoints"]
        or report["scene_shard_count"] != RAW_TOTAL_COUNTS["scene_shards"]
        or report["sample_count"] != 24
        or report["sample_results_sha256"] != RAW_V13_SAMPLE_RESULTS_SHA256
        or report["source_file_count"] != 354
        or any(report[field] is not False for field in RAW_DOWNSTREAM_AUTHORITY_FIELDS)
    ):
        raise PermissionError("Raw V13 terminal identity, count, or authority changed")
    proof_fields = (
        "strict_integer_cardinalities",
        "unaliased_descriptor_bound_dataset_leaves",
        "full_byte_inventory_revalidated",
        "pair_endpoint_joins_reconstructed",
        "all_stored_evidence_and_rasters_recomputed",
        "sample_original_geometry_recomputed",
    )
    if (
        any(report[field] is not True for field in proof_fields)
        or not is_sha256(report["source_inventory_before_after_sha256"])
    ):
        raise PermissionError("Raw V13 terminal proof semantics changed")
    population = _plain_dict(
        report["observed_population"],
        fields={"pair_counts", "pair_count", "endpoint_reference_counts", "endpoint_reference_count", "unique_endpoint_counts", "unique_endpoint_count", "role_count", "family_counts", "scene_shard_count"},
        name="Raw V13 observed population",
    )
    expected_pairs = {role: ROLE_COUNTS[role]["pairs"] for role in DEVELOPMENT_ROLES}
    expected_references = {role: ROLE_COUNTS[role]["endpoint_instances"] for role in DEVELOPMENT_ROLES}
    expected_unique = {role: ROLE_COUNTS[role]["unique_endpoints"] for role in DEVELOPMENT_ROLES}
    if population != {
        "pair_counts": expected_pairs,
        "pair_count": 5172,
        "endpoint_reference_counts": expected_references,
        "endpoint_reference_count": 10344,
        "unique_endpoint_counts": expected_unique,
        "unique_endpoint_count": 9460,
        "role_count": 3,
        "family_counts": {role: 8 for role in DEVELOPMENT_ROLES},
        "scene_shard_count": 88,
    }:
        raise PermissionError("Raw V13 complete population changed")
    if type(report["sample_results"]) is not list or len(report["sample_results"]) != 24:
        raise PermissionError("Raw V13 sample results count changed")
    observed_role_families = set()
    for index, row in enumerate(report["sample_results"]):
        sample = _plain_dict(row, fields={"dataset_role", "family", "endpoint_identity_sha256", "selection_sha256", "array_byte_sha256", "array_byte_sha256_set", "passes"}, name=f"Raw V13 sample result {index}")
        if sample["passes"] is not True or not is_sha256(sample["endpoint_identity_sha256"]) or not is_sha256(sample["selection_sha256"]) or type(sample["array_byte_sha256"]) is not list or len(sample["array_byte_sha256"]) != 8 or not all(is_sha256(item) for item in sample["array_byte_sha256"]) or canonical_json_sha256(sample["array_byte_sha256"]) != sample["array_byte_sha256_set"]:
            raise PermissionError("Raw V13 sample result changed")
        observed_role_families.add((sample["dataset_role"], sample["family"]))
    if observed_role_families != {(role, family) for role in DEVELOPMENT_ROLES for family in FAMILIES}:
        raise PermissionError("Raw V13 sample PASS coverage changed")
    opens = _plain_dict(report["source_payload_opens"], fields={"complete_inventory_hash_passes", "permitted_source_files_per_pass", "sample_endpoint_count", "rgb_byte_opens", "rgb_decodes", "label_shard_payload_opens", "g2_payload_opens", "checkpoint_model_runtime_heldout_hardware_production_opens"}, name="Raw V13 source payload opens")
    if opens != {"complete_inventory_hash_passes": 2, "permitted_source_files_per_pass": 354, "sample_endpoint_count": 24, "rgb_byte_opens": 0, "rgb_decodes": 0, "label_shard_payload_opens": 0, "g2_payload_opens": 0, "checkpoint_model_runtime_heldout_hardware_production_opens": 0}:
        raise PermissionError("Raw V13 source/provenance counts changed")
    if report["authorization_v13"] != {
        "file_sha256": RAW_CHAIN_SOURCE_BINDINGS[
            RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH
        ],
        "content_sha256": "4b179c33de00399652f4f915285ca99a4d47cfa95d31878d1f91ca7e8fd9d0e8",
        "source_map_sha256": RAW_V13_AUTHORIZATION_SOURCE_MAP_SHA256,
        "phase_one_zero_target_opens": True,
        "phase_two_fixed_target_count": 14,
        "transitive_v12_target_count": 25,
        "machine_pass_reviews_parsed": 2,
    }:
        raise PermissionError("Raw V13 authorization population changed")
    expected_terminal_artifacts = {
        "frozen_v9_terminal_artifacts": {
            "dataset_manifest_file_sha256": RAW_V9_MANIFEST_FILE_SHA256,
            "dataset_manifest_content_sha256": RAW_V9_MANIFEST_CONTENT_SHA256,
            "terminal_failure_file_sha256": "863630579e6d8f8ac222ff7ce5ba04ff3e7901885b606dcb6bcfd7a07fe7722f",
            "terminal_failure_content_sha256": "aaf342f7df88796e0d03259e964ed51e42ebd1faecb33bbfe9ea9cfd0d5e2c72",
            "terminal_failure_retry_authorized": False,
            "success_report_absent": True,
        },
        "frozen_v10_terminal_artifacts": {
            "audit_authorization_file_sha256": "146e0bbf029d28fdf883bfc357b1ddbbce955f86bda00508c6091cb01db4800a",
            "audit_authorization_content_sha256": "8bab96369a5633cb82266fef6ec54964a3c25f27dc0877fde550721f3b6af981",
            "audit_authorization_source_map_sha256": "1b6ffd40b72c7d02dba24d2035ac3442af361b8b804e7df4273f5e73d1cda79b",
            "terminal_failure_file_sha256": "2c391550df540d233ded11bfcf1531dbbb29663a51918fb60e7d8cf4146d0996",
            "terminal_failure_content_sha256": "66370ec52ae06bef81ab75a47cce481830067b88ec2d579ed41a4a58a7cecc83",
            "terminal_failure_retry_authorized": False,
            "success_report_absent": True,
        },
        "frozen_v11_terminal_artifacts": {
            "auditor_block_file_sha256": "169494633f8b9bd50ceac40436e6ef1b168624b8a8c487fedb2033a9c137f3db",
            "auditor_block_content_sha256": "f45610c0db743bfd6ec655bd7d9c3f1e1f3578a3b57e7b55f7d2fcf029d76a94",
            "auditor_block_verdict": "BLOCK",
            "exact_audit_authorized": False,
            "success_report_absent": True,
        },
        "frozen_v12_terminal_artifacts": {
            "audit_authorization_file_sha256": "6b5f317119a00308390b8a32f1057f34455313eb80ec190aa9d8d27052a81575",
            "audit_authorization_content_sha256": "8db4611a321309a76a0dd81e3af0148fce788422e2008ccaef1039e3c5ae493a",
            "audit_authorization_source_map_sha256": "1fc3374101fca166fe74b34b779cf995ec46a12fbb609f5de3bc5a428d225bc2",
            "authorization_witness_file_sha256": "662e6c2f6386b8822b3bd968a4faf0bf3e2e222ff4aac9df8a99cc680c254327",
            "authorization_witness_content_sha256": "4845826d1caeedc58d01b580a8681a71730eb0ba17205bde36d3673c9052741b",
            "launch_failure_file_sha256": "cc6313b1d6e56022204ba82dc57efc6b7cc85a715f078cd865883b61cee88eb3",
            "launch_failure_content_sha256": "b9775ef4705d7505931b64c7ceaad57fb8d18da72429bb877245fb534197b2ee",
            "launch_failure_terminal": True,
            "success_report_absent": True,
            "failure_report_absent": True,
        },
    }
    if any(
        report[name] != expected
        for name, expected in expected_terminal_artifacts.items()
    ):
        raise PermissionError("Raw V13 predecessor terminal chain changed")
    if report["closed_publication_transaction_v13"] != {
        "source_and_candidate_watches_continuous_through_rename": True,
        "retained_source_dataset_and_candidate_descriptors": True,
        "publication_and_source_ancestor_chains_watched": True,
        "single_renameat2_RENAME_NOREPLACE": True,
        "exact_owned_rename_event_sequence": True,
        "post_rename_inventory_and_quiescence": True,
    }:
        raise PermissionError("Raw V13 closed publication proof changed")
    return report


CAMERA_LADDER_ROW_FIELDS = frozenset(
    {
        "schema", "row_index", "seed", "fit_size", "origin", "attempt_identity",
        "reservation", "output_root", "completion", "production_source_bindings",
        "source_review", "gate", "checkpoint", "rung_review", "fresh_initialization",
        "warm_start_used", "retry_performed", "reexecution_performed",
        "predecessor_checkpoint_opened", "predecessor_checkpoint_copied",
        "predecessor_checkpoint_loaded", "launched_by_ladder", "migratable",
    }
)
CAMERA_LADDER_AGGREGATE_FIELDS = frozenset(
    {
        "ordered_rung_count",
        "preexisting_seed_20260710_n5_count",
        "additional_attempt_count",
        "seed_20260710_n5_reexecuted",
        "all_rungs_fresh_initialization",
        "warm_start_used",
        "retry_performed",
        "both_seed_ladders_pass",
        "rows_sha256",
        "gate_file_sha256",
        "gate_content_sha256",
        "independent_review_file_sha256",
        "independent_review_content_sha256",
    }
)


def validate_camera_v14_ladder_rows(
    rows: object,
    *,
    reviewed_source_bindings: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Require the exact ordered eight-row V14 evidence table."""

    if type(reviewed_source_bindings) is not dict or tuple(reviewed_source_bindings) != CAMERA_V14_PRODUCTION_SOURCE_PATHS or not all(is_sha256(value) for value in reviewed_source_bindings.values()):
        raise ValueError("Camera V14 reviewed production-source closure changed")
    if type(rows) is not list or len(rows) != len(CAMERA_LADDER_ORDER):
        raise ValueError("Camera V14 ladder must have exactly eight ordered rows")
    expected_sources = [
        {"role": role, "path": path, "file_sha256": reviewed_source_bindings[path]}
        for role, path in zip(("policy", "trainer", "verifier", "executor"), CAMERA_V14_PRODUCTION_SOURCE_PATHS)
    ]
    identities: set[str] = set()
    checkpoint_paths: set[str] = set()
    rung_specific_paths: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, (raw, expected_pair) in enumerate(zip(rows, CAMERA_LADDER_ORDER)):
        row = _plain_dict(raw, fields=CAMERA_LADDER_ROW_FIELDS, name=f"Camera V14 ladder row {index}")
        seed, fit_size = expected_pair
        if row["schema"] != "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_raster_nll_v14_ladder_row_v1" or row["row_index"] != index or row["seed"] != seed or row["fit_size"] != fit_size:
            raise PermissionError("Camera V14 ladder order or row identity changed")
        if type(row["attempt_identity"]) is not str or not row["attempt_identity"] or row["attempt_identity"] in identities:
            raise PermissionError("Camera V14 attempt identity is missing or repeated")
        identities.add(row["attempt_identity"])
        if row["origin"] != ("preexisting_v14_n5_evidence_only" if index == 0 else "new_ladder_attempt"):
            raise PermissionError("Camera V14 rung origin changed")
        expected_output_root = (
            CAMERA_V14_OUTPUT_ROOT_RELATIVE_PATH
            if index == 0
            else (
                f"{CAMERA_V14_LADDER_ROOT_RELATIVE_PATH}/attempts/"
                f"seed_{seed}/n{fit_size}"
            )
        )
        expected_attempt_root = (
            f"{CAMERA_V14_OUTPUT_ROOT_RELATIVE_PATH}/attempts/"
            f"seed_{seed}/n{fit_size}"
            if index == 0
            else expected_output_root
        )
        expected_paths = {
            "reservation": f"{expected_attempt_root}/reservation.json",
            "completion": f"{expected_attempt_root}/completed.json",
            "gate": (
                CAMERA_V14_N5_GATE_RELATIVE_PATH
                if index == 0
                else (
                    f"{CAMERA_V14_LADDER_ROOT_RELATIVE_PATH}/gates/"
                    f"seed_{seed}_n{fit_size}.json"
                )
            ),
            "checkpoint": f"{expected_attempt_root}/checkpoint.pt",
        }
        for name in ("reservation", "completion"):
            binding = _plain_dict(row[name], fields={"path", "file_sha256", "content_sha256"}, name=f"Camera V14 row {index} {name}")
            _canonical_relative_path_string(binding["path"], name=f"Camera V14 row {index} {name} path")
            if binding["path"] != expected_paths[name] or not is_sha256(binding["file_sha256"]) or not is_sha256(binding["content_sha256"]):
                raise PermissionError("Camera V14 rung reservation/completion binding changed")
            if binding["path"] in rung_specific_paths:
                raise PermissionError("Camera V14 rung-specific path repeated")
            rung_specific_paths.add(binding["path"])
        _canonical_relative_path_string(row["output_root"], name=f"Camera V14 row {index} output root")
        if row["output_root"] != expected_output_root:
            raise PermissionError("Camera V14 rung output root changed")
        if row["output_root"] in rung_specific_paths:
            raise PermissionError("Camera V14 rung-specific output root repeated")
        rung_specific_paths.add(row["output_root"])
        if row["production_source_bindings"] != expected_sources:
            raise PermissionError("Camera V14 rung production closure changed")
        source_review = _plain_dict(row["source_review"], fields={"path", "file_sha256", "content_sha256", "reviewer", "schema", "status", "verdict"}, name=f"Camera V14 row {index} source review")
        gate = _plain_dict(row["gate"], fields={"path", "file_sha256", "content_sha256", "schema", "gate_schema_sha256", "status", "passes"}, name=f"Camera V14 row {index} gate")
        checkpoint = _plain_dict(row["checkpoint"], fields={"path", "file_sha256", "schema", "initialization_identity", "emitted_by_attempt_identity"}, name=f"Camera V14 row {index} checkpoint")
        rung_review = _plain_dict(row["rung_review"], fields={"path", "file_sha256", "content_sha256", "reviewer", "schema", "verdict", "production_source_bindings_sha256", "source_review_file_sha256", "gate_file_sha256", "completion_file_sha256", "checkpoint_file_sha256"}, name=f"Camera V14 row {index} rung review")
        for name, binding in (("source review", source_review), ("gate", gate), ("checkpoint", checkpoint), ("rung review", rung_review)):
            _canonical_relative_path_string(binding["path"], name=f"Camera V14 row {index} {name} path")
        digest_values = [source_review["file_sha256"], source_review["content_sha256"], gate["file_sha256"], gate["content_sha256"], gate["gate_schema_sha256"], checkpoint["file_sha256"], rung_review["file_sha256"], rung_review["content_sha256"], rung_review["production_source_bindings_sha256"]]
        if (
            not all(is_sha256(value) for value in digest_values)
            or source_review["path"] != CAMERA_V14_SOURCE_REVIEW_RELATIVE_PATH
            or source_review["schema"] != CAMERA_V14_SOURCE_REVIEW_SCHEMA
            or type(source_review["reviewer"]) is not str
            or not source_review["reviewer"].startswith("/root/")
            or source_review["reviewer"] in {"/root", "/root/camera_v12_gate_aligned_implementer"}
            or type(rung_review["reviewer"]) is not str
            or not rung_review["reviewer"].startswith("/root/")
            or rung_review["reviewer"] in {"/root", "/root/camera_v12_gate_aligned_implementer"}
            or source_review["status"] != "different_agent_source_review_passed"
            or source_review["verdict"] != "PASS"
            or gate["path"] != expected_paths["gate"]
            or gate["schema"] != CAMERA_V14_GATE_SCHEMA
            or gate["status"] != "passed"
            or gate["passes"] is not True
            or checkpoint["path"] != expected_paths["checkpoint"]
            or type(checkpoint["schema"]) is not str
            or not checkpoint["schema"]
            or not is_sha256(checkpoint["initialization_identity"])
            or type(rung_review["schema"]) is not str
            or not rung_review["schema"]
            or rung_review["verdict"] != "PASS"
        ):
            raise PermissionError("Camera V14 rung review/gate verdict changed")
        if checkpoint["emitted_by_attempt_identity"] != row["attempt_identity"] or rung_review["source_review_file_sha256"] != source_review["file_sha256"] or rung_review["gate_file_sha256"] != gate["file_sha256"] or rung_review["completion_file_sha256"] != row["completion"]["file_sha256"] or rung_review["checkpoint_file_sha256"] != checkpoint["file_sha256"] or rung_review["production_source_bindings_sha256"] != canonical_json_sha256(expected_sources):
            raise PermissionError("Camera V14 per-rung evidence closure changed")
        if type(checkpoint["path"]) is not str or checkpoint["path"] in checkpoint_paths:
            raise PermissionError("Camera V14 checkpoint identity repeated")
        checkpoint_paths.add(checkpoint["path"])
        for binding in (gate, checkpoint, rung_review):
            if binding["path"] in rung_specific_paths:
                raise PermissionError("Camera V14 rung-specific evidence path repeated")
            rung_specific_paths.add(binding["path"])
        expected_bool = {
            "fresh_initialization": True,
            "warm_start_used": False,
            "retry_performed": False,
            "reexecution_performed": False,
            "predecessor_checkpoint_opened": False,
            "predecessor_checkpoint_copied": False,
            "predecessor_checkpoint_loaded": False,
            "launched_by_ladder": index != 0,
            "migratable": index == 3,
        }
        if any(row[name] is not expected for name, expected in expected_bool.items()):
            raise PermissionError("Camera V14 rung lifecycle or migration boundary changed")
        result.append(row)
    return result


def validate_camera_v14_ladder_aggregate(
    value: object,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = _plain_dict(
        value,
        fields=CAMERA_LADDER_AGGREGATE_FIELDS,
        name="Camera V14 ladder aggregate",
    )
    expected = {
        "ordered_rung_count": 8,
        "preexisting_seed_20260710_n5_count": 1,
        "additional_attempt_count": 7,
        "seed_20260710_n5_reexecuted": False,
        "all_rungs_fresh_initialization": True,
        "warm_start_used": False,
        "retry_performed": False,
        "both_seed_ladders_pass": True,
    }
    if (
        any(aggregate[name] != expected_value for name, expected_value in expected.items())
        or aggregate["rows_sha256"] != canonical_json_sha256(list(rows))
        or not all(
            is_sha256(aggregate[name])
            for name in (
                "gate_file_sha256",
                "gate_content_sha256",
                "independent_review_file_sha256",
                "independent_review_content_sha256",
            )
        )
    ):
        raise PermissionError("Camera V14 aggregate evidence changed")
    return aggregate


def validate_exact_binding_preflight_authorization(value: object) -> dict[str, Any]:
    """Validate the later additive authority; source reviews never reach here."""

    fields = {
        "schema", "status", "authorizer", "reviewer", "governing_amendment",
        "blocked_manifest", "implementation_review", "reviewed_sources",
        "raw_v13_terminal_bindings", "camera_v14_source_bindings",
        "camera_ladder_rows", "camera_ladder_aggregate",
        "primary_migratable_checkpoint", "authority", "content_sha256",
    }
    record = _plain_dict(value, fields=fields, name="V4 exact-binding preflight authorization")
    if (
        record["schema"] != EXACT_BINDING_PREFLIGHT_AUTHORIZATION_SCHEMA
        or record["status"] != "reviewed_authorized_for_one_payload_free_preflight"
        or type(record["authorizer"]) is not str
        or type(record["reviewer"]) is not str
        or not record["reviewer"].startswith("/root/")
        or record["reviewer"] in {"/root", IMPLEMENTATION_AUTHOR, record["authorizer"]}
        or record["governing_amendment"] != {"path": V4_AMENDMENT_RELATIVE_PATH, "file_sha256": V4_AMENDMENT_SHA256}
        or record["raw_v13_terminal_bindings"] != RAW_V13_TERMINAL_BINDINGS
        or record["authority"] != PREFLIGHT_AUTHORITY
    ):
        raise PermissionError("V4 exact-binding preflight authority changed")
    blocked = _plain_dict(record["blocked_manifest"], fields={"path", "file_sha256", "content_sha256"}, name="blocked-manifest binding")
    review = _plain_dict(record["implementation_review"], fields={"path", "file_sha256", "content_sha256"}, name="implementation-review binding")
    if blocked["path"] != EXACT_EXECUTION_MANIFEST_RELATIVE_PATH or review["path"] != IMPLEMENTATION_REVIEW_RELATIVE_PATH or not all(is_sha256(item) for item in (blocked["file_sha256"], blocked["content_sha256"], review["file_sha256"], review["content_sha256"])):
        raise PermissionError("V4 source assurance binding changed")
    sources = record["reviewed_sources"]
    if type(sources) is not dict or set(sources) != set(IMPLEMENTATION_SOURCE_PATHS) or not all(is_sha256(item) for item in sources.values()):
        raise PermissionError("V4 reviewed source closure changed")
    camera_sources = record["camera_v14_source_bindings"]
    rows = validate_camera_v14_ladder_rows(record["camera_ladder_rows"], reviewed_source_bindings=camera_sources)
    aggregate = validate_camera_v14_ladder_aggregate(
        record["camera_ladder_aggregate"],
        rows=rows,
    )
    primary = _plain_dict(record["primary_migratable_checkpoint"], fields={"row_index", "seed", "fit_size", "path", "file_sha256", "schema", "rung_review_file_sha256"}, name="Camera V14 primary checkpoint")
    row_three = rows[3]
    if primary != {
        "row_index": 3,
        "seed": 20260710,
        "fit_size": 320,
        "path": row_three["checkpoint"]["path"],
        "file_sha256": row_three["checkpoint"]["file_sha256"],
        "schema": row_three["checkpoint"]["schema"],
        "rung_review_file_sha256": row_three["rung_review"]["file_sha256"],
    }:
        raise PermissionError("Camera V14 migratable checkpoint changed")
    copied = dict(record)
    declared = copied.pop("content_sha256")
    if not is_sha256(declared) or canonical_json_sha256(copied) != declared:
        raise PermissionError("V4 exact-binding preflight authorization content changed")
    return record


def validate_final_exact_execution_authorization(value: object) -> dict[str, Any]:
    """Validate the only record allowed to authorize the exact V4 reservation."""

    fields = {
        "schema", "status", "authorizer", "reviewer", "governing_amendment",
        "blocked_manifest", "implementation_review", "preflight_authorization",
        "preflight_evidence", "reviewed_sources", "required_exact_bindings",
        "raw_v13_terminal_bindings", "camera_v14_source_bindings",
        "camera_ladder_rows", "camera_ladder_aggregate",
        "primary_migratable_checkpoint", "raw_v13_dataset_use_grant",
        "authority", "content_sha256",
    }
    record = _plain_dict(value, fields=fields, name="V4 final exact execution authorization")
    if (
        record["schema"] != FINAL_EXACT_EXECUTION_AUTHORIZATION_SCHEMA
        or record["status"] != "reviewed_authorized_for_one_exact_matched_training_attempt"
        or type(record["authorizer"]) is not str
        or type(record["reviewer"]) is not str
        or not record["reviewer"].startswith("/root/")
        or record["reviewer"] in {"/root", IMPLEMENTATION_AUTHOR, record["authorizer"]}
        or record["governing_amendment"] != {"path": V4_AMENDMENT_RELATIVE_PATH, "file_sha256": V4_AMENDMENT_SHA256}
        or record["raw_v13_terminal_bindings"] != RAW_V13_TERMINAL_BINDINGS
        or record["raw_v13_dataset_use_grant"] != RAW_DATASET_USE_GRANT
        or record["authority"] != FINAL_EXACT_AUTHORITY
    ):
        raise PermissionError("V4 final exact execution authority changed")
    for name, expected_path in (
        ("blocked_manifest", EXACT_EXECUTION_MANIFEST_RELATIVE_PATH),
        ("implementation_review", IMPLEMENTATION_REVIEW_RELATIVE_PATH),
        ("preflight_authorization", EXACT_BINDING_PREFLIGHT_AUTHORIZATION_RELATIVE_PATH),
    ):
        binding = _plain_dict(record[name], fields={"path", "file_sha256", "content_sha256"}, name=f"V4 {name} binding")
        if binding["path"] != expected_path or not is_sha256(binding["file_sha256"]) or not is_sha256(binding["content_sha256"]):
            raise PermissionError(f"V4 {name} binding changed")
    preflight = _plain_dict(
        record["preflight_evidence"],
        fields={"completion_file_sha256", "completion_content_sha256", "receipt_file_sha256", "receipt_content_sha256", "verification_file_sha256", "verification_content_sha256", "independently_verified_pass", "payload_open_count"},
        name="V4 preflight evidence",
    )
    if preflight["independently_verified_pass"] is not True or preflight["payload_open_count"] != 0 or not all(is_sha256(value) for key, value in preflight.items() if key.endswith("sha256")):
        raise PermissionError("V4 preflight evidence changed")
    sources = record["reviewed_sources"]
    if type(sources) is not dict or set(sources) != set(IMPLEMENTATION_SOURCE_PATHS) or not all(is_sha256(value) for value in sources.values()):
        raise PermissionError("V4 reviewed source closure changed")
    bindings = record["required_exact_bindings"]
    if type(bindings) is not dict or set(bindings) != set(FINAL_REQUIRED_BINDING_NAMES) or not all(is_sha256(value) for value in bindings.values()):
        raise PermissionError("V4 required exact binding set changed")
    if any(bindings[name] != expected for name, expected in FINAL_FROZEN_RAW_BINDINGS.items()):
        raise PermissionError("V4 frozen Raw V13 binding changed")
    source_binding_names = {
        POLICY_RELATIVE_PATH: "implementation_policy_source_sha256",
        LOSS_ADAPTER_RELATIVE_PATH: "loss_adapter_source_sha256",
        PREFLIGHT_EXECUTOR_RELATIVE_PATH: "preflight_executor_source_sha256",
        PREFLIGHT_VERIFIER_RELATIVE_PATH: "preflight_verifier_source_sha256",
        EXACT_EXECUTOR_RELATIVE_PATH: "exact_executor_source_sha256",
        EXACT_TRAINER_RELATIVE_PATH: "exact_trainer_source_sha256",
        EXACT_VERIFIER_RELATIVE_PATH: "exact_verifier_source_sha256",
    }
    if any(sources[path] != bindings[name] for path, name in source_binding_names.items()):
        raise PermissionError("V4 source hash and exact binding disagree")
    if bindings["preflight_completed_file_sha256"] != preflight["completion_file_sha256"] or bindings["preflight_receipt_file_sha256"] != preflight["receipt_file_sha256"] or bindings["preflight_independent_review_file_sha256"] != preflight["verification_file_sha256"]:
        raise PermissionError("V4 preflight hash and exact binding disagree")
    camera_sources = record["camera_v14_source_bindings"]
    rows = validate_camera_v14_ladder_rows(record["camera_ladder_rows"], reviewed_source_bindings=camera_sources)
    aggregate = validate_camera_v14_ladder_aggregate(
        record["camera_ladder_aggregate"],
        rows=rows,
    )
    if bindings["camera_v14_two_seed_ladder_pass_file_sha256"] != aggregate["gate_file_sha256"] or bindings["camera_v14_two_seed_ladder_pass_content_sha256"] != aggregate["gate_content_sha256"] or bindings["camera_v14_ladder_independent_review_file_sha256"] != aggregate["independent_review_file_sha256"]:
        raise PermissionError("V4 Camera aggregate exact binding changed")
    primary = record["primary_migratable_checkpoint"]
    if type(primary) is not dict or primary.get("row_index") != 3 or primary.get("file_sha256") != rows[3]["checkpoint"]["file_sha256"] or bindings["v4_primary_seed_20260710_n320_checkpoint_file_sha256"] != rows[3]["checkpoint"]["file_sha256"] or bindings["camera_v14_source_review_file_sha256"] != rows[0]["source_review"]["file_sha256"] or bindings["camera_v14_source_review_content_sha256"] != rows[0]["source_review"]["content_sha256"] or bindings["camera_v14_n5_gate_pass_file_sha256"] != rows[0]["gate"]["file_sha256"] or bindings["camera_v14_n5_gate_pass_content_sha256"] != rows[0]["gate"]["content_sha256"] or bindings["implementation_independent_review_file_sha256"] != record["implementation_review"]["file_sha256"]:
        raise PermissionError("V4 primary migratable checkpoint changed")
    copied = dict(record)
    declared = copied.pop("content_sha256")
    if not is_sha256(declared) or canonical_json_sha256(copied) != declared:
        raise PermissionError("V4 final exact authorization content changed")
    return record


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
        "navigation_authorized": False,
        "hardware_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
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
            "final_exact_authorization",
            "implementation_review",
        },
        "exact_input": {
            "preflight_receipt",
            "raw_supervision_manifest",
            "raw_supervision_audit",
            "camera_v14_two_seed_ladder",
            "camera_v14_primary_checkpoint",
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
    "AUTHOR_TEST_RELATIVE_PATH",
    "BLOCKED_FUTURE_BINDING_NAMES",
    "CANONICAL_EXACT_ROOT",
    "CANONICAL_PREFLIGHT_ROOT",
    "CAMERA_LADDER_AGGREGATE_FIELDS",
    "CAMERA_LADDER_ORDER",
    "CAMERA_LADDER_ROW_FIELDS",
    "CAMERA_V13_TERMINAL_BLOCK_RELATIVE_PATH",
    "CAMERA_V13_TERMINAL_BLOCK_SHA256",
    "CAMERA_V14_AMENDMENT_RELATIVE_PATH",
    "CAMERA_V14_AMENDMENT_SHA256",
    "CAMERA_V14_LADDER_PREREGISTRATION_RELATIVE_PATH",
    "CAMERA_V14_LADDER_REVIEW_RELATIVE_PATH",
    "CAMERA_V14_N5_GATE_RELATIVE_PATH",
    "CAMERA_V14_PRIMARY_CHECKPOINT_RELATIVE_PATH",
    "CAMERA_V14_PRODUCTION_SOURCE_PATHS",
    "CAMERA_V14_SOURCE_REVIEW_RELATIVE_PATH",
    "CAMERA_V14_TWO_SEED_LADDER_RELATIVE_PATH",
    "CHECKPOINT_UPDATES",
    "DEVELOPMENT_ROLES",
    "DEVICE_CONTRACT",
    "EFFECTIVE_BATCH_SIZE",
    "EXACT_BINDING_PREFLIGHT_AUTHORIZATION_RELATIVE_PATH",
    "EXACT_BINDING_PREFLIGHT_AUTHORIZATION_SCHEMA",
    "EXACT_EXECUTION_MANIFEST_RELATIVE_PATH",
    "EXACT_INVENTORY",
    "EXACT_ROOT_RELATIVE_PATH",
    "FAMILIES",
    "FINAL_EXACT_AUTHORITY",
    "FINAL_EXACT_EXECUTION_AUTHORIZATION_RELATIVE_PATH",
    "FINAL_EXACT_EXECUTION_AUTHORIZATION_SCHEMA",
    "FINAL_FROZEN_RAW_BINDINGS",
    "FINAL_REQUIRED_BINDING_NAMES",
    "INITIALIZATION_SEED",
    "IMPLEMENTATION_REVIEW_RELATIVE_PATH",
    "IMPLEMENTATION_HANDOFF_RELATIVE_PATH",
    "IMPLEMENTATION_SOURCE_PATHS",
    "IMPLEMENTATION_AUTHOR",
    "LOSS_ADAPTER_RELATIVE_PATH",
    "INITIALIZATION_SCHEMA",
    "INPUT_BINDINGS_SCHEMA",
    "JOINT_LOSS_CONTRACT",
    "MANIFEST_AUTHORITY",
    "MICROBATCH_SIZE",
    "OPTIMIZER_CONTRACT",
    "PREFLIGHT_AUTHORITY",
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
    "RAW_DOWNSTREAM_AUTHORITY_FIELDS",
    "RAW_MANIFEST_FIELDS",
    "RAW_ORDERED_ENDPOINT_SHA256",
    "RAW_ORDERED_PAIR_SHA256",
    "RAW_REPORT_FIELDS",
    "RAW_V13_PASS_CONTENT_SHA256",
    "RAW_V13_PASS_FILE_SHA256",
    "RAW_V13_SAMPLE_RESULTS_SHA256",
    "RAW_V9_MANIFEST_CONTENT_SHA256",
    "RAW_V9_MANIFEST_FILE_SHA256",
    "RAW_V13_TERMINAL_BINDINGS",
    "ROLE_COUNTS",
    "SCHEDULE_SEED",
    "SELECTION_SCHEMA",
    "SCOPES",
    "SOURCE_REVIEW_AUTHORITY",
    "SOURCE_REVIEW_SCHEMA",
    "TRAIN_PAIR_COUNT",
    "TRAINING_RECORD_SCHEMA",
    "UPDATE_COUNT",
    "V4_AMENDMENT_RELATIVE_PATH",
    "V4_AMENDMENT_SHA256",
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
    "validate_exact_binding_preflight_authorization",
    "validate_final_exact_execution_authorization",
    "validate_implementation_review",
    "validate_camera_v14_ladder_aggregate",
    "validate_camera_v14_ladder_rows",
    "validate_raw_v13_manifest",
    "validate_raw_v13_source_chain",
    "validate_raw_v13_terminal_report",
    "verify_fixed_source_hashes",
]

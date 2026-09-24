"""Static contract for the authority-free V6 N5 full-panel executor.

V6 deliberately has no execution token, authority object, capability, issuer,
registry, or mutable lifecycle state.  The only production entry point is the
isolated, canonical-path-only executor bound in ``SUCCESSOR_SOURCE_PATHS``.
Importing this module is stdlib-only and opens no data or output.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/camera_v5_independent"

POLICY_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py"
)
EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py"
)
SUCCESSOR_SOURCE_PATHS = (POLICY_RELATIVE_PATH, EXECUTOR_RELATIVE_PATH)

SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_"
    "independent_review_2026-07-13.json"
)
CANONICAL_SOURCE_REVIEW_PATH = ROOT / SOURCE_REVIEW_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_"
    "successor_preregistration_2026-07-13.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "0ad13e3897c70f90df6705538f4d86262ec53d3e096618a69563acdf63567c01"
)
TRIGGER_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_"
    "structural_trigger_amendment_2026-07-13.md"
)
TRIGGER_AMENDMENT_FILE_SHA256 = (
    "1e08aac0ace734d2cbcce9e965b10a7031a94764dd7b47114d38e33944990262"
)
TERMINAL_INVALIDATION_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "prepublication_structural_invalidation_2026-07-13.json"
)
TERMINAL_INVALIDATION_FILE_SHA256 = (
    "1744a50badd6c9f5c1ef4c8c3cbd05f8c0fc8acff4fbbf066e40e1f7de24f560"
)
TERMINAL_INVALIDATION_CONTENT_SHA256 = (
    "7bdaae6ebb13b7d90290dfe07f5d48f403d29cad977f4a56c9ac7b8cfbcb8602"
)
RECOVERY_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_"
    "lifecycle_recovery_amendment_2026-07-13.md"
)
RECOVERY_AMENDMENT_FILE_SHA256 = (
    "1fa4279c604b1a8be825e082a367a5404381154fe1784394e43aee35924caa90"
)

V5_POLICY_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py"
)
V5_POLICY_FILE_SHA256 = (
    "cc28934be4fe1109feae3a31803e9e09502e968591268f80fc7124ba0a63f2c1"
)
V5_EXECUTOR_RELATIVE_PATH = (
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py"
)
V5_EXECUTOR_FILE_SHA256 = (
    "5dcc77a7434b64d3ae759b563b16db95e909bec9d1751dacc7657f6a740ac2e1"
)
V5_SYNTHETIC_RELATIVE_PATH = (
    "lewm/tests/n5_full_panel_v5_synthetic_execution.py"
)
V5_SYNTHETIC_FILE_SHA256 = (
    "7601341cd92beb1a9a6738d2534e6f654a4058fe7d84b07547ac75f674fef608"
)
V5_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py"
)
V5_TEST_FILE_SHA256 = (
    "80f51db295cad4d2a8494d1c61a1f605dac12cf558b5137d0eeee15611d88264"
)
V5_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v5_"
    "implementation_handoff_2026-07-13.md"
)
V5_HANDOFF_FILE_SHA256 = (
    "df3d58eff6b582a113beb9d558c3e210f7a22acd38763f55037ae86609dc8b5c"
)
V5_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v5_"
    "independent_review_2026-07-13.md"
)
V5_REVIEW_FILE_SHA256 = (
    "d07407b5a21be6d44f214f902e16cf0d88ba8c6875a9fea5b09633e44f37ba59"
)
V5_REVIEW_RECORD_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v5_"
    "independent_review_2026-07-13.json"
)
V5_REVIEW_RECORD_FILE_SHA256 = (
    "81345d133e53da1911d2561c6eaab74c341645fbf45dbefdf89bf730fed36cb0"
)
V5_REVIEW_RECORD_CONTENT_SHA256 = (
    "441b0854fc50eda49b4124bd40d5e4beedaedfa41a4e99e1231b3ee81fa0d11d"
)
V5_RESERVATION_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/"
    "seed_20260710/n5/reservation.json"
)
V5_RESERVATION_FILE_SHA256 = (
    "f8062f2ed2bdb1589ca806fb9331ce7f1ec0675d4466e96c0a78530080ea501a"
)
V5_RESERVATION_CONTENT_SHA256 = (
    "1427a5524cbc7e72ac24d78c221775bab3c943d36967b88df6e780743faafc15"
)
V5_RESERVATION_BYTE_COUNT = 4532
V5_FAILURE_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/"
    "seed_20260710/n5/failed.json"
)
V5_FAILURE_FILE_SHA256 = (
    "7ead760085f5365ac83ebfc8875910cbc076437fa972d48d008aa3b2127e50af"
)
V5_FAILURE_CONTENT_SHA256 = (
    "84cfa81aa2db9fa7cd7233e314e7d3da50b4fc23af863ab38e9ab948ac51358b"
)
V5_FAILURE_BYTE_COUNT = 802

V1_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "independent_review_2026-07-13.md"
)
V1_REVIEW_FILE_SHA256 = (
    "11479b03ff9eac24dd5541d38faeda480739c8d17de7b2b658759e306ace2d5e"
)
V1_BLOCK_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "independent_review_block_2026-07-13.json"
)
V1_BLOCK_FILE_SHA256 = (
    "ccd8d97988d2ce165722703fbfcf813758ee42a5408e02d26bf7db38d8ea506e"
)
V1_BLOCK_CONTENT_SHA256 = (
    "99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7"
)
V1_EXPLOIT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "independent_review.py"
)
V1_EXPLOIT_TEST_FILE_SHA256 = (
    "387147a8dd6fe1a20184284a05c18df73419ca91c21054eb378e79a8194d5b3b"
)
V1_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "implementation_handoff_2026-07-13.md"
)
V1_HANDOFF_FILE_SHA256 = (
    "8f4735a3ecd20a8c19bd729fdaf71ceb60a3a884de717423e8f84ef6ef2745f7"
)

V2_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_"
    "independent_review_2026-07-13.md"
)
V2_REVIEW_FILE_SHA256 = (
    "24953fc64da151a6ff1f4ad89e5465e1caae300223556702e0f5c8430d47ee04"
)
V2_BLOCK_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_"
    "independent_review_block_2026-07-13.json"
)
V2_BLOCK_FILE_SHA256 = (
    "ddca89e467e4cc30e52bacf57b28c040465e712843fde465f472f3cc8b38fc73"
)
V2_BLOCK_CONTENT_SHA256 = (
    "c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a"
)
V2_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_"
    "implementation_handoff_2026-07-13.md"
)
V2_HANDOFF_FILE_SHA256 = (
    "3056b00f7b5f224c0507f07505c005f4f5ea2171fb97e6f78585cf7f0460bb61"
)
V2_EXPLOIT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_"
    "independent_review.py"
)
V2_EXPLOIT_TEST_FILE_SHA256 = (
    "a53c5e5d351784ff2a4824231998194e15040597897411c91e7727ec73a95e69"
)

V3_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_"
    "implementation_handoff_2026-07-13.md"
)
V3_HANDOFF_FILE_SHA256 = (
    "c97b3f761955fb6d73469c53632c27388626ae75b010c317fe64b860f76bf8db"
)
V3_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_"
    "independent_review_2026-07-13.md"
)
V3_REVIEW_FILE_SHA256 = (
    "d28eadce56668b0cf793806bb98e7c793eb9d874b7ca818d4d9b3c3205fe53e7"
)
V3_BLOCK_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_"
    "independent_review_block_2026-07-13.json"
)
V3_BLOCK_FILE_SHA256 = (
    "d1f859aea2a80f090c3ee09df5194f5b4bcfca22865f323de543f3b216b3e168"
)
V3_BLOCK_CONTENT_SHA256 = (
    "d84152d611631364e4c52114a753c36fdabd1cf69d5508d4cb25b5b93dd67f2f"
)
RETAINED_V3_ARTIFACT_BINDINGS = (
    (
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py",
        "b0f5929aadfaeb9a10f2211db21297c7c01d10305e094a249e5ad8f27b8f46d3",
    ),
    (
        "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py",
        "8a8bec79bbbfdd2554e0625afc3d423ea9ec8e56baf1134f70d334efe357af66",
    ),
    (
        "lewm/tests/n5_full_panel_v3_synthetic_execution.py",
        "83af899f8479f6a3e98530da5af2c58b2b0fd25b48e29954ef77db08e5bf5c91",
    ),
    (
        "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py",
        "730513d7607b02539b58cde883600a28e6d0e3592333a16d5df67ac3e092beee",
    ),
    (
        "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_independent_review.py",
        "b7d3669135f22311e13c840e04c4ec2ed583365fc77f7fce6c5c0ecc4e512395",
    ),
)

V4_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "implementation_handoff_2026-07-13.md"
)
V4_HANDOFF_FILE_SHA256 = (
    "4e0aa7e2efa266feb774a4b095cbddca105cfd046aac7a0da7f942f1b2b6925e"
)
V4_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "independent_review_2026-07-13.md"
)
V4_REVIEW_FILE_SHA256 = (
    "7edeff73d6022a4086706907b03084ff080c9ad1d52ae91e8659fc6ecdc6b18c"
)
V4_BLOCK_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "independent_review_block_2026-07-13.json"
)
V4_BLOCK_FILE_SHA256 = (
    "d2224049a4ee2b793737802d06d91757c17d20b0457c1624517467638173c507"
)
V4_BLOCK_CONTENT_SHA256 = (
    "0c34ec6931c8850a949498ca1b38f16548db76bc4d6e1e47994c6514898ff091"
)
V4_EXPLOIT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "independent_review.py"
)
V4_EXPLOIT_TEST_FILE_SHA256 = (
    "2942b23215f506fa9893013d377f5bb4ce4b2327083a1806be4746bfdae56e9f"
)
RETAINED_V4_ARTIFACT_BINDINGS = (
    (
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py",
        "ff291b94b1546ae9ccf0b85de5f96b87edce4ad5b7992ca16bbbf13dcd1d4485",
    ),
    (
        "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py",
        "19cbdc5692911b31b3b44883b0cfefcc81daa4afc16250b89c1317dd9b66afe4",
    ),
    (
        "lewm/tests/n5_full_panel_v4_synthetic_execution.py",
        "01e49c303d0e2c8e76e7ecbdbd2d0cf159948a5f36a4dc6248d0e014d9c69fb5",
    ),
    (
        "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py",
        "299fd18b88a869916a916adc4e8848235e955447e9a1f245aeaeec6e7ee69688",
    ),
)

RETAINED_V1_SOURCE_BINDINGS = (
    ("lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py", "875edc86efbe25d246b24c2ef2467cc7956b1b3bb90e6d8d1e03e4a9c5b11d88"),
    ("scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py", "3cb9ff782a15bc97dd3cca2cc25705e006d6af19a7dbef6d27dee893d9b570c8"),
    ("scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py", "48ac856c080906a8d73d5a9b97d1dcf7fe21f5bc99217cce669c43b9c091acca"),
    ("scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py", "00c62cec39e1eb05bf23a96a9153aa8ff350235c2e5c6662f6148934ab9d85b0"),
    ("scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py", "1d4471381a6c3b29f0b077e44e3126f956281ff105d4e38aa8e0f6ba18675b8b"),
)
RETAINED_V2_SOURCE_BINDINGS = (
    ("lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py", "096b597b0e84a6822fd8fcdd8221da27e95757aaa2c05ca148afad6e23ad60d2"),
    ("scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py", "03311bb48da80b912c2576844adf5cd488c1b9a0818268d2252902d860436591"),
    ("scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py", "357369b652c489ab99937c06afaed0ec4cf66aa1f46017f74f5dac46da93d3aa"),
    ("scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py", "cab757839c3d784cb5760f30c2bde6163311bfbf87df1620c9c0f77ff69b624b"),
    ("scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py", "a5dc625b8b270913df56d8b5044c263ba3fdbd1ef6cb3e6f62e084a5335ee323"),
)
FROZEN_SOURCE_BINDINGS = (
    ("scripts/launch_go2_observable_camera_ray_fit_v4_v2.py", "65c58e36cb97d155a58ec1cbc93a1f2f42a75e62f049b5d8e874481a435a614b"),
    ("scripts/train_go2_observable_camera_ray_fit_v4_v2.py", "c9d22fb38acdf5fd3099271661dc65bb9cea989426a3b6021ad28649d6dd74d3"),
    ("lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py", "aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad"),
    ("lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py", "6a0e40f9dcb496831553dc5bbc6d1efcdf6d82676d6f18aa20e417f8de4fa6a0"),
    ("lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py", "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85"),
    ("lewm/models/observable_camera_ray_evidence_v4.py", "6238f7fb2b9c0c5201c9d7ebb5343ceef72fa97b423dddb466465b6c594cc882"),
    ("lewm/models/observable_camera_ray_evidence_v4_training.py", "c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed"),
    ("docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json", "4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a"),
    ("scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py", "4624dd761901808c72b37eb256b360e3db61c9b8f61337879547ed38836a3eed"),
    ("docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v2_partition_amendment_2026-07-12.md", "1e65f8884b1b8e0ad2219ddad54f79f9fabae514bfcaa048b29c8113b076ac1f"),
)

DATASET_MANIFEST_RELATIVE_PATH = ".generated/go2_observable_camera_ray_fit_v4/v1/manifest.json"
DATASET_MANIFEST_FILE_SHA256 = "2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85"
DATASET_MANIFEST_CONTENT_SHA256 = "9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812"
AUDIT_RECEIPT_RELATIVE_PATH = ".generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json"
AUDIT_RECEIPT_FILE_SHA256 = "2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c"
AUDIT_RECEIPT_CONTENT_SHA256 = "a922114b7e42552043a487bae527c35fb511804d4e8683c5a3f64a2bf499cf76"
TRAINER_AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json"
TRAINER_AUTHORIZATION_FILE_SHA256 = "d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802"
TRAINER_AUTHORIZATION_CONTENT_SHA256 = "18a285e80252d41de7daadba918a00223d8770b71c533f74807e0ace5444ac1e"
TRAINER_REVIEW_RELATIVE_PATH = "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json"
TRAINER_REVIEW_FILE_SHA256 = "c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea"
TRAINER_REVIEW_CONTENT_SHA256 = "ab55270986268c5a326eeb6ba191cd9a0531112b1b742812d2cbd549f67158be"
RGB_RECEIPT_CONTENT_SHA256 = "d763d7ae294e4e5a9e5f2352672913bc06411388d92abe1fb0f5090dfc41d5c3"
SUBSET_CONTENT_SHA256 = "3595dff9d24dbb44f3e73086fce3be4ec53eb8659684738defa8591c4a375f15"
TARGET_PARTITION_CONTENT_SHA256 = "ac9d6e1c91ca58c1182fa5e05d3189a6dc319013c3dc07e2f229f88c55cca429"

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v6"
)
CANONICAL_OUTPUT_ROOT = ROOT / OUTPUT_ROOT_RELATIVE_PATH
CANONICAL_ATTEMPT_PATH = CANONICAL_OUTPUT_ROOT / "attempts/seed_20260710/n5"
CANONICAL_METRIC_RECEIPT_PATH = (
    CANONICAL_OUTPUT_ROOT / "metric_verifications/seed_20260710_n5.json"
)
CANONICAL_GATE_PATH = CANONICAL_OUTPUT_ROOT / "gates/seed_20260710_n5.json"

SCHEDULE_ALGORITHM = "torch_cpu_generator_manual_seed_then_concatenated_randperm_cycles_take_steps_times_batch_v1"
EXPECTED_SCHEDULE_SHA256 = "62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634"
LOSS_COMPONENTS = (
    "ordered_first_hit_nll",
    "target_bin_offset_smooth_l1",
    "ground_clear_distance_state_balanced_bce",
    "derived_raster_hierarchical_bce",
)
LOSS_ABSOLUTE_TOLERANCE = 1e-9
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

SOURCE_REVIEW_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_source_review_v1"
RESERVATION_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_reservation_v1"
RESULT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_result_v1"
COMPLETION_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_completion_v1"
FAILURE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_failure_v1"
METRIC_RECEIPT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_metric_verification_v1"
GATE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_gate_v1"


def frozen_source_bindings() -> dict[str, str]:
    return dict(FROZEN_SOURCE_BINDINGS)


def retained_v1_source_bindings() -> dict[str, str]:
    return dict(RETAINED_V1_SOURCE_BINDINGS)


def retained_v2_source_bindings() -> dict[str, str]:
    return dict(RETAINED_V2_SOURCE_BINDINGS)


def authority_bindings() -> dict[str, Any]:
    return {
        "preregistration": {"path": PREREGISTRATION_RELATIVE_PATH, "file_sha256": PREREGISTRATION_FILE_SHA256},
        "structural_trigger_amendment": {"path": TRIGGER_AMENDMENT_RELATIVE_PATH, "file_sha256": TRIGGER_AMENDMENT_FILE_SHA256},
        "terminal_invalidation": {
            "path": TERMINAL_INVALIDATION_RELATIVE_PATH,
            "file_sha256": TERMINAL_INVALIDATION_FILE_SHA256,
            "content_sha256": TERMINAL_INVALIDATION_CONTENT_SHA256,
        },
        "lifecycle_recovery_amendment": {
            "path": RECOVERY_AMENDMENT_RELATIVE_PATH,
            "file_sha256": RECOVERY_AMENDMENT_FILE_SHA256,
        },
        "v5_terminal_reservation": {
            "path": V5_RESERVATION_RELATIVE_PATH,
            "file_sha256": V5_RESERVATION_FILE_SHA256,
            "content_sha256": V5_RESERVATION_CONTENT_SHA256,
            "byte_count": V5_RESERVATION_BYTE_COUNT,
        },
        "v5_terminal_failure": {
            "path": V5_FAILURE_RELATIVE_PATH,
            "file_sha256": V5_FAILURE_FILE_SHA256,
            "content_sha256": V5_FAILURE_CONTENT_SHA256,
            "byte_count": V5_FAILURE_BYTE_COUNT,
        },
    }


def experiment_contract() -> dict[str, Any]:
    return {
        "seed": 20260710,
        "fit_size": 5,
        "fresh_model_initialization": True,
        "model_class": "ObservableCameraRayEvidenceV4Model",
        "optimizer": "AdamW",
        "optimizer_updates": 400,
        "training_batch_size": 5,
        "frame_exposures": 2000,
        "evaluation_batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "precision": "float32",
        "autocast": False,
        "gradient_clip_norm": 1.0,
        "loss_weights": {name: 0.25 for name in LOSS_COMPONENTS},
        "schedule_algorithm": SCHEDULE_ALGORITHM,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "checkpoint_selection": "final_update_only",
        "evaluation_controls": ["matched_rgb", "wrong_rgb_with_target_calibration"],
        "device": "cuda:0",
        "device_name": "AMD Radeon AI PRO R9700",
        "raphael_igpu_forbidden": True,
        "rgb_worker_count_max": 5,
        "native_threads_per_process": 1,
        "attempt_count": 1,
        "output_path": str(CANONICAL_ATTEMPT_PATH),
    }


def licenses() -> dict[str, bool]:
    return {
        "authorizes_one_fresh_n5_full_panel_infrastructure_replacement": True,
        "authorizes_metric_verification_only_checkpoint_use": True,
        "authorizes_stage_finalization": True,
        "authorizes_retry": False,
        "authorizes_scientific_retry": False,
        "authorizes_v5_numeric_payload_read": False,
        "authorizes_n16_execution": False,
        "authorizes_second_seed": False,
        "authorizes_later_v6_training": False,
        "authorizes_g2": False,
        "authorizes_holdout": False,
        "authorizes_selection": False,
        "authorizes_calibration_change": False,
        "authorizes_runtime": False,
        "authorizes_hardware": False,
        "authorizes_production": False,
        "authorizes_promotion": False,
    }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, *, name: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _stable_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _relative_parts(relative: str | Path, *, name: str) -> tuple[str, ...]:
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(
        component in {"", ".", ".."} for component in path.parts
    ):
        raise PermissionError(f"{name} path is not a strict relative entry")
    return tuple(path.parts)


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise PermissionError("component-wise no-follow directory opens are unavailable")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _file_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise PermissionError("no-follow file opens are unavailable")
    return (
        os.O_RDONLY
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _lstat_at(parent_fd: int, component: str, *, name: str) -> os.stat_result:
    try:
        return os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise PermissionError(f"{name} component changed") from error


def _revalidate_directory_chain(
    anchor_fd: int,
    anchor_fingerprint: tuple[int, ...],
    entries: Sequence[tuple[int, str, int, tuple[int, ...]]],
    *,
    name: str,
) -> None:
    if _stable_fingerprint(os.fstat(anchor_fd)) != anchor_fingerprint:
        raise RuntimeError(f"{name} filesystem root changed")
    for parent_fd, component, child_fd, fingerprint in entries:
        entry = _lstat_at(parent_fd, component, name=name)
        opened = os.fstat(child_fd)
        if (
            stat.S_ISLNK(entry.st_mode)
            or not stat.S_ISDIR(entry.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _stable_fingerprint(entry) != fingerprint
            or _stable_fingerprint(opened) != fingerprint
        ):
            raise RuntimeError(f"{name} directory component changed")


def _revalidate_leaf(
    parent_fd: int,
    leaf_name: str,
    fingerprint: tuple[int, ...],
    *,
    name: str,
) -> None:
    entry = _lstat_at(parent_fd, leaf_name, name=name)
    if (
        stat.S_ISLNK(entry.st_mode)
        or not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or _stable_fingerprint(entry) != fingerprint
    ):
        raise RuntimeError(f"{name} leaf entry changed")


def read_regular_bytes_at(root: Path, relative: str | Path, *, name: str) -> bytes:
    """Read a leaf through one filesystem-root-anchored descriptor chain."""

    root = Path(root)
    parts = _relative_parts(relative, name=name)
    if not root.is_absolute():
        raise PermissionError(f"{name} root is not absolute")
    try:
        root_before = root.stat(follow_symlinks=False)
        root_resolved = root.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise PermissionError(f"{name} root does not exist canonically") from error
    if (
        root_resolved != root
        or root.is_symlink()
        or not stat.S_ISDIR(root_before.st_mode)
    ):
        raise PermissionError(f"{name} root is not a canonical real directory")
    root_fingerprint = _stable_fingerprint(root_before)

    descriptors: list[int] = []
    entries: list[tuple[int, str, int, tuple[int, ...]]] = []
    leaf_descriptor = -1
    try:
        filesystem_root = Path(root.anchor)
        anchor_before = filesystem_root.stat(follow_symlinks=False)
        anchor_fingerprint = _stable_fingerprint(anchor_before)
        anchor_fd = os.open(filesystem_root, _directory_flags())
        descriptors.append(anchor_fd)
        opened_anchor = os.fstat(anchor_fd)
        if (
            not stat.S_ISDIR(opened_anchor.st_mode)
            or _stable_fingerprint(opened_anchor) != anchor_fingerprint
        ):
            raise PermissionError(f"{name} filesystem root changed during open")

        parent_fd = anchor_fd
        for component in root.parts[1:]:
            before = _lstat_at(parent_fd, component, name=name)
            fingerprint = _stable_fingerprint(before)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise PermissionError(f"{name} root component is not a real directory")
            child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
            descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or _stable_fingerprint(opened) != fingerprint
            ):
                raise PermissionError(f"{name} root component changed during open")
            entries.append((parent_fd, component, child_fd, fingerprint))
            parent_fd = child_fd

        root_fd = parent_fd
        if _stable_fingerprint(os.fstat(root_fd)) != root_fingerprint:
            raise PermissionError(f"{name} root fingerprint changed during open")
        for component in parts[:-1]:
            before = _lstat_at(parent_fd, component, name=name)
            fingerprint = _stable_fingerprint(before)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise PermissionError(f"{name} parent is not a real directory")
            child_fd = os.open(
                component,
                _directory_flags(),
                dir_fd=parent_fd,
            )
            descriptors.append(child_fd)
            metadata = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or _stable_fingerprint(metadata) != fingerprint
            ):
                raise PermissionError(f"{name} parent changed during open")
            entries.append((parent_fd, component, child_fd, fingerprint))
            parent_fd = child_fd

        leaf_name = parts[-1]
        leaf_before = _lstat_at(parent_fd, leaf_name, name=name)
        leaf_fingerprint = _stable_fingerprint(leaf_before)
        if (
            stat.S_ISLNK(leaf_before.st_mode)
            or not stat.S_ISREG(leaf_before.st_mode)
            or leaf_before.st_nlink != 1
        ):
            raise PermissionError(f"{name} is not a singly-linked regular file")
        leaf_descriptor = os.open(
            leaf_name,
            _file_flags(),
            dir_fd=parent_fd,
        )
        opened_leaf = os.fstat(leaf_descriptor)
        if (
            not stat.S_ISREG(opened_leaf.st_mode)
            or opened_leaf.st_nlink != 1
            or _stable_fingerprint(opened_leaf) != leaf_fingerprint
        ):
            raise PermissionError(f"{name} leaf changed during open")
        _revalidate_directory_chain(
            anchor_fd,
            anchor_fingerprint,
            entries,
            name=name,
        )
        _revalidate_leaf(
            parent_fd,
            leaf_name,
            leaf_fingerprint,
            name=name,
        )
        chunks: list[bytes] = []
        while chunk := os.read(leaf_descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(leaf_descriptor)
        if _stable_fingerprint(after) != leaf_fingerprint:
            raise RuntimeError(f"{name} changed while read")
        _revalidate_directory_chain(
            anchor_fd,
            anchor_fingerprint,
            entries,
            name=name,
        )
        _revalidate_leaf(
            parent_fd,
            leaf_name,
            leaf_fingerprint,
            name=name,
        )
        return b"".join(chunks)
    finally:
        if leaf_descriptor >= 0:
            os.close(leaf_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_regular_bytes(path: Path, *, name: str) -> bytes:
    path = Path(path)
    try:
        relative = path.relative_to(ROOT)
    except ValueError as error:
        raise PermissionError(f"{name} is outside the canonical repository") from error
    if ROOT / relative != path:
        raise PermissionError(f"{name} does not name a canonical repository entry")
    return read_regular_bytes_at(ROOT, relative, name=name)


def read_hashed_bytes(path: Path, expected_sha256: str, *, name: str) -> bytes:
    if not is_sha256(expected_sha256):
        raise ValueError(f"{name} caller SHA-256 is malformed")
    raw = read_regular_bytes(path, name=name)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} file SHA-256 changed")
    return raw


def load_hashed_json(path: Path, expected_sha256: str, *, name: str, require_canonical: bool = True) -> tuple[dict[str, Any], bytes]:
    raw = read_hashed_bytes(path, expected_sha256, name=name)
    value = parse_json(raw, name=name)
    if require_canonical and raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical JSON plus newline")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content SHA-256 changed")
    return value, raw


def _hash_file(relative: str, expected: str, *, name: str) -> bytes:
    return read_hashed_bytes(ROOT / relative, expected, name=name)


def _validate_block(relative: str, file_sha: str, content_sha: str, *, name: str) -> None:
    value, _ = load_hashed_json(
        ROOT / relative,
        file_sha,
        name=name,
        require_canonical=False,
    )
    if (
        value.get("content_sha256") != content_sha
        or not str(value.get("status", "")).startswith("blocked_")
    ):
        raise PermissionError(f"{name} binding changed")


def _validate_v5_terminal_incident() -> dict[str, Any]:
    review, _ = load_hashed_json(
        ROOT / V5_REVIEW_RECORD_RELATIVE_PATH,
        V5_REVIEW_RECORD_FILE_SHA256,
        name="V5 independent PASS review",
    )
    if (
        review.get("content_sha256") != V5_REVIEW_RECORD_CONTENT_SHA256
        or review.get("status")
        != "different_agent_review_passed_authority_free_exact_full_panel_v5"
        or review.get("exact_attempt_authorized") is not True
    ):
        raise PermissionError("V5 independent PASS review changed")

    reservation, reservation_raw = load_hashed_json(
        ROOT / V5_RESERVATION_RELATIVE_PATH,
        V5_RESERVATION_FILE_SHA256,
        name="V5 consumed reservation receipt",
    )
    failure, failure_raw = load_hashed_json(
        ROOT / V5_FAILURE_RELATIVE_PATH,
        V5_FAILURE_FILE_SHA256,
        name="V5 terminal failure receipt",
    )
    expected_reservation_binding = {
        "path": "reservation.json",
        "file_sha256": V5_RESERVATION_FILE_SHA256,
        "byte_count": V5_RESERVATION_BYTE_COUNT,
        "content_sha256": V5_RESERVATION_CONTENT_SHA256,
    }
    if (
        len(reservation_raw) != V5_RESERVATION_BYTE_COUNT
        or reservation.get("content_sha256") != V5_RESERVATION_CONTENT_SHA256
        or reservation.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v5_reservation_v1"
        or reservation.get("status") != "reserved"
        or reservation.get("maximum_attempts") != 1
        or reservation.get("licenses", {}).get("retry_authorized") is not False
        or len(failure_raw) != V5_FAILURE_BYTE_COUNT
        or failure.get("content_sha256") != V5_FAILURE_CONTENT_SHA256
        or failure.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v5_failure_v1"
        or failure.get("status") != "failed"
        or failure.get("failure_stage") != "training"
        or failure.get("failure")
        != {"class": "permission", "code": "scope_or_authorization_failure"}
        or failure.get("reservation") != expected_reservation_binding
        or failure.get("artifact_cleanup") != []
        or failure.get("partial_artifacts_removed") is not True
        or failure.get("retry_authorized") is not False
        or not isinstance(failure.get("licenses"), Mapping)
        or any(value is not False for value in failure["licenses"].values())
    ):
        raise PermissionError("V5 terminal lifecycle evidence changed")
    attempt = (ROOT / V5_RESERVATION_RELATIVE_PATH).parent
    if sorted(child.name for child in attempt.iterdir()) != [
        "failed.json",
        "reservation.json",
    ]:
        raise PermissionError("V5 terminal attempt gained a numeric payload")
    return {
        "source_review_content_sha256": V5_REVIEW_RECORD_CONTENT_SHA256,
        "reservation_content_sha256": V5_RESERVATION_CONTENT_SHA256,
        "failure_content_sha256": V5_FAILURE_CONTENT_SHA256,
        "numeric_payload_survived": False,
        "retry_authorized": False,
    }


def preflight_static_authority() -> dict[str, Any]:
    """Rehash every frozen parent without constructing execution authority."""

    _hash_file(PREREGISTRATION_RELATIVE_PATH, PREREGISTRATION_FILE_SHA256, name="N5 preregistration")
    _hash_file(TRIGGER_AMENDMENT_RELATIVE_PATH, TRIGGER_AMENDMENT_FILE_SHA256, name="N5 trigger amendment")
    _hash_file(
        RECOVERY_AMENDMENT_RELATIVE_PATH,
        RECOVERY_AMENDMENT_FILE_SHA256,
        name="N5 V6 lifecycle recovery amendment",
    )
    invalidation, _ = load_hashed_json(
        ROOT / TERMINAL_INVALIDATION_RELATIVE_PATH,
        TERMINAL_INVALIDATION_FILE_SHA256,
        name="N5 terminal structural invalidation",
        require_canonical=False,
    )
    authority = invalidation.get("authority")
    primary = invalidation.get("primary_structural_invalidation")
    if (
        invalidation.get("content_sha256") != TERMINAL_INVALIDATION_CONTENT_SHA256
        or invalidation.get("status") != "terminal_prepublication_structural_invalidation"
        or invalidation.get("scope", {}).get("decision") != "immutable_n5_is_structurally_invalid_for_canonical_finalization"
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
        or not isinstance(primary, Mapping)
        or primary.get("full_immutable_result_validation") != {"passed": False, "exception": "ValueError: V4 matched evaluation losses are inconsistent"}
    ):
        raise PermissionError("N5 structural-trigger authority changed")
    for relative, digest in FROZEN_SOURCE_BINDINGS:
        _hash_file(relative, digest, name=f"frozen dependency {relative}")
    for relative, digest in RETAINED_V1_SOURCE_BINDINGS:
        _hash_file(relative, digest, name=f"retained V1 source {relative}")
    for relative, digest in RETAINED_V2_SOURCE_BINDINGS:
        _hash_file(relative, digest, name=f"retained V2 source {relative}")
    for relative, digest in RETAINED_V3_ARTIFACT_BINDINGS:
        _hash_file(relative, digest, name=f"retained V3 artifact {relative}")
    for relative, digest in RETAINED_V4_ARTIFACT_BINDINGS:
        _hash_file(relative, digest, name=f"retained V4 artifact {relative}")
    for relative, digest in (
        (V5_POLICY_RELATIVE_PATH, V5_POLICY_FILE_SHA256),
        (V5_EXECUTOR_RELATIVE_PATH, V5_EXECUTOR_FILE_SHA256),
        (V5_SYNTHETIC_RELATIVE_PATH, V5_SYNTHETIC_FILE_SHA256),
        (V5_TEST_RELATIVE_PATH, V5_TEST_FILE_SHA256),
        (V5_HANDOFF_RELATIVE_PATH, V5_HANDOFF_FILE_SHA256),
        (V5_REVIEW_RELATIVE_PATH, V5_REVIEW_FILE_SHA256),
    ):
        _hash_file(relative, digest, name=f"retained V5 artifact {relative}")
    _hash_file(V1_REVIEW_RELATIVE_PATH, V1_REVIEW_FILE_SHA256, name="V1 BLOCK review")
    _validate_block(V1_BLOCK_RELATIVE_PATH, V1_BLOCK_FILE_SHA256, V1_BLOCK_CONTENT_SHA256, name="V1 BLOCK JSON")
    _hash_file(V1_EXPLOIT_TEST_RELATIVE_PATH, V1_EXPLOIT_TEST_FILE_SHA256, name="V1 exploit tests")
    _hash_file(V1_HANDOFF_RELATIVE_PATH, V1_HANDOFF_FILE_SHA256, name="V1 handoff")
    _hash_file(V2_REVIEW_RELATIVE_PATH, V2_REVIEW_FILE_SHA256, name="V2 BLOCK review")
    _validate_block(V2_BLOCK_RELATIVE_PATH, V2_BLOCK_FILE_SHA256, V2_BLOCK_CONTENT_SHA256, name="V2 BLOCK JSON")
    _hash_file(V2_EXPLOIT_TEST_RELATIVE_PATH, V2_EXPLOIT_TEST_FILE_SHA256, name="V2 exploit tests")
    _hash_file(V2_HANDOFF_RELATIVE_PATH, V2_HANDOFF_FILE_SHA256, name="V2 handoff")
    _hash_file(V3_HANDOFF_RELATIVE_PATH, V3_HANDOFF_FILE_SHA256, name="V3 handoff")
    _hash_file(V3_REVIEW_RELATIVE_PATH, V3_REVIEW_FILE_SHA256, name="V3 BLOCK review")
    _validate_block(
        V3_BLOCK_RELATIVE_PATH,
        V3_BLOCK_FILE_SHA256,
        V3_BLOCK_CONTENT_SHA256,
        name="V3 BLOCK JSON",
    )
    _hash_file(V4_HANDOFF_RELATIVE_PATH, V4_HANDOFF_FILE_SHA256, name="V4 handoff")
    _hash_file(V4_REVIEW_RELATIVE_PATH, V4_REVIEW_FILE_SHA256, name="V4 BLOCK review")
    _hash_file(
        V4_EXPLOIT_TEST_RELATIVE_PATH,
        V4_EXPLOIT_TEST_FILE_SHA256,
        name="V4 exploit tests",
    )
    _validate_block(
        V4_BLOCK_RELATIVE_PATH,
        V4_BLOCK_FILE_SHA256,
        V4_BLOCK_CONTENT_SHA256,
        name="V4 BLOCK JSON",
    )
    v5_terminal = _validate_v5_terminal_incident()
    return {
        "authority_bindings": authority_bindings(),
        "terminal_invalidation_status": invalidation["status"],
        "frozen_source_bindings": frozen_source_bindings(),
        "v1_block_content_sha256": V1_BLOCK_CONTENT_SHA256,
        "v2_block_content_sha256": V2_BLOCK_CONTENT_SHA256,
        "v3_block_content_sha256": V3_BLOCK_CONTENT_SHA256,
        "v4_block_content_sha256": V4_BLOCK_CONTENT_SHA256,
        "v5_terminal": v5_terminal,
    }


def expected_source_review_core(*, reviewer: str, successor_sources: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    return {
        "schema": SOURCE_REVIEW_SCHEMA,
        "status": "different_agent_review_passed_n5_full_panel_v6_lifecycle_recovery",
        "implementation_author": IMPLEMENTATION_AUTHOR,
        "reviewer": reviewer,
        "review_completed": True,
        "source_closure_approved": True,
        "exact_attempt_authorized": True,
        "infrastructure_replacement_authorized": True,
        "scientific_retry_authorized": False,
        "v5_numeric_payload_inspected": False,
        "successor_sources": dict(successor_sources),
        "retained_v1_sources": retained_v1_source_bindings(),
        "retained_v2_sources": retained_v2_source_bindings(),
        "v1_block_evidence": {"review_file_sha256": V1_REVIEW_FILE_SHA256, "block_file_sha256": V1_BLOCK_FILE_SHA256, "block_content_sha256": V1_BLOCK_CONTENT_SHA256, "exploit_test_file_sha256": V1_EXPLOIT_TEST_FILE_SHA256, "handoff_file_sha256": V1_HANDOFF_FILE_SHA256},
        "v2_block_evidence": {"review_file_sha256": V2_REVIEW_FILE_SHA256, "block_file_sha256": V2_BLOCK_FILE_SHA256, "block_content_sha256": V2_BLOCK_CONTENT_SHA256, "exploit_test_file_sha256": V2_EXPLOIT_TEST_FILE_SHA256, "handoff_file_sha256": V2_HANDOFF_FILE_SHA256},
        "v3_block_evidence": {
            "review_file_sha256": V3_REVIEW_FILE_SHA256,
            "block_file_sha256": V3_BLOCK_FILE_SHA256,
            "block_content_sha256": V3_BLOCK_CONTENT_SHA256,
            "handoff_file_sha256": V3_HANDOFF_FILE_SHA256,
            "retained_artifacts": dict(RETAINED_V3_ARTIFACT_BINDINGS),
        },
        "v4_block_evidence": {
            "review_file_sha256": V4_REVIEW_FILE_SHA256,
            "block_file_sha256": V4_BLOCK_FILE_SHA256,
            "block_content_sha256": V4_BLOCK_CONTENT_SHA256,
            "exploit_test_file_sha256": V4_EXPLOIT_TEST_FILE_SHA256,
            "handoff_file_sha256": V4_HANDOFF_FILE_SHA256,
            "retained_artifacts": dict(RETAINED_V4_ARTIFACT_BINDINGS),
        },
        "v5_terminal_evidence": {
            "source_review": {
                "path": V5_REVIEW_RECORD_RELATIVE_PATH,
                "file_sha256": V5_REVIEW_RECORD_FILE_SHA256,
                "content_sha256": V5_REVIEW_RECORD_CONTENT_SHA256,
            },
            "reservation": {
                "path": V5_RESERVATION_RELATIVE_PATH,
                "file_sha256": V5_RESERVATION_FILE_SHA256,
                "content_sha256": V5_RESERVATION_CONTENT_SHA256,
                "byte_count": V5_RESERVATION_BYTE_COUNT,
            },
            "failure": {
                "path": V5_FAILURE_RELATIVE_PATH,
                "file_sha256": V5_FAILURE_FILE_SHA256,
                "content_sha256": V5_FAILURE_CONTENT_SHA256,
                "byte_count": V5_FAILURE_BYTE_COUNT,
            },
            "numeric_payload_survived": False,
            "numeric_payload_inspected": False,
            "retry_authorized": False,
        },
        "execution_contract": {
            "caller_held_authority": False,
            "caller_held_capability": False,
            "mutable_lifecycle_registry": False,
            "single_isolated_end_to_end_operation": True,
            "production_path_injection": False,
            "filesystem_claim_is_single_use": True,
            "synthetic_test_executor_is_separate_and_production_ineligible": True,
            "source_and_rgb_rehashed_before_publication": True,
            "importable_partial_stage": False,
            "stage_values_constructed_inside_script_entry": True,
            "component_nofollow_source_walk": True,
            "filesystem_root_anchored_source_walk": True,
            "post_training_failure_terminalization": True,
            "shared_ancestor_child_churn_tolerated": True,
            "shared_ancestor_identity_and_security_bound": True,
            "exclusive_subtree_full_metadata_bound": True,
            "ancestor_alias_and_identity_replacement_rejected": True,
        },
        "reservation_contract": {
            "unique_private_staging": True,
            "process_death_safe_locking": True,
            "recovery_rehashes_complete_staging": True,
            "inode_owned_claim": True,
            "parent_fsync_immediately_after_rename": True,
            "terminal_attempt_and_parent_fsync": True,
            "claimed_directory_descriptor_retained_end_to_end": True,
            "canonical_claim_parent_chain_retained_end_to_end": True,
            "owned_derived_partials_removed_before_failure_terminalization": True,
            "new_exclusive_output_namespace": True,
            "unowned_exclusive_subtree_mutation_rejected": True,
        },
        "frozen_source_bindings": frozen_source_bindings(),
        "authority_bindings": authority_bindings(),
        "experiment": experiment_contract(),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "licenses": licenses(),
    }


def preflight_source_review(path: Path, file_sha256: str) -> tuple[dict[str, Any], bytes]:
    canonical = CANONICAL_SOURCE_REVIEW_PATH
    supplied = Path(path)
    if supplied != canonical:
        raise PermissionError("N5 full-panel V6 source review path is not canonical")
    review, raw = load_hashed_json(
        canonical,
        file_sha256,
        name="N5 full-panel V6 different-agent source review",
    )
    reviewer = review.get("reviewer")
    sources = review.get("successor_sources")
    if not isinstance(reviewer, str) or not reviewer.startswith("/root/") or reviewer == IMPLEMENTATION_AUTHOR or not isinstance(sources, Mapping) or set(sources) != set(SUCCESSOR_SOURCE_PATHS):
        raise PermissionError("N5 full-panel V6 review is not by a different agent")
    checked: dict[str, dict[str, str]] = {}
    for relative in SUCCESSOR_SOURCE_PATHS:
        binding = sources.get(relative)
        if not isinstance(binding, Mapping) or binding.get("path") != relative or not is_sha256(binding.get("file_sha256")):
            raise PermissionError("N5 full-panel V6 source binding changed")
        source_raw = read_regular_bytes(ROOT / relative, name=f"V6 source {relative}")
        if hashlib.sha256(source_raw).hexdigest() != binding["file_sha256"]:
            raise PermissionError(f"N5 full-panel V6 source changed: {relative}")
        checked[relative] = dict(binding)
    expected = expected_source_review_core(reviewer=reviewer, successor_sources=checked)
    core = dict(review)
    declared = core.pop("content_sha256", None)
    if core != expected or canonical_json_sha256(core) != declared:
        raise PermissionError("N5 full-panel V6 source review contract changed")
    return review, raw


def source_review_binding(review: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    if not is_sha256(file_sha256) or not is_sha256(review.get("content_sha256")):
        raise ValueError("N5 full-panel V6 review binding is malformed")
    return {"path": SOURCE_REVIEW_RELATIVE_PATH, "file_sha256": file_sha256, "content_sha256": str(review["content_sha256"])}


def artifact_binding(relative_path: str, raw: bytes, *, content_sha256: str | None = None) -> dict[str, Any]:
    binding: dict[str, Any] = {"path": relative_path, "file_sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    if content_sha256 is not None:
        binding["content_sha256"] = content_sha256
    return binding


def _shadow_review(review: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(review)
    value["path"] = "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review_2026-07-13.json"
    return value


def _shadow_experiment() -> dict[str, Any]:
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )

    return copy.deepcopy(retained.EXPERIMENT)


def _shadow_authority_bindings() -> dict[str, Any]:
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )

    return copy.deepcopy(retained.AUTHORITY_BINDINGS)


def validate_evaluation_structure(value: object) -> dict[str, Any]:
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained
    return retained.validate_evaluation_structure(value)


def validate_reservation_structure(reservation: Mapping[str, Any], *, expected_source_review: Mapping[str, str]) -> dict[str, Any]:
    core = dict(reservation)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError("N5 full-panel V6 reservation content SHA-256 changed")
    recovery = reservation.get("preclaim_recovery")
    if not isinstance(recovery, list) or any(not isinstance(item, Mapping) for item in recovery):
        raise ValueError("N5 full-panel V6 recovery ledger is malformed")
    shadow = copy.deepcopy(dict(reservation))
    shadow.pop("preclaim_recovery")
    shadow["schema"] = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_reservation_v1"
    shadow["source_review"] = _shadow_review(shadow["source_review"])
    shadow["scope"] = "one_exclusive_fresh_full_panel_attempt"
    shadow["experiment"] = _shadow_experiment()
    shadow["authority_bindings"] = _shadow_authority_bindings()
    shadow_core = dict(shadow)
    shadow_core.pop("content_sha256")
    shadow["content_sha256"] = canonical_json_sha256(shadow_core)
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained
    retained.validate_reservation_structure(shadow, expected_source_review=_shadow_review(expected_source_review))
    if (
        reservation.get("schema") != RESERVATION_SCHEMA
        or reservation.get("scope")
        != "one_exclusive_fresh_infrastructure_replacement_attempt"
        or reservation.get("maximum_attempts") != 1
        or reservation.get("experiment") != experiment_contract()
        or reservation.get("authority_bindings") != authority_bindings()
        or reservation.get("source_review") != dict(expected_source_review)
        or reservation.get("licenses", {}).get("retry_authorized") is not False
    ):
        raise ValueError("N5 full-panel V6 reservation contract changed")
    return dict(reservation)


def validate_result_structure(result: Mapping[str, Any], *, expected_source_review: Mapping[str, str] | None = None) -> dict[str, Any]:
    core = dict(result)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError("N5 full-panel V6 result content SHA-256 changed")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("experiment") != experiment_contract()
        or result.get("authority_bindings") != authority_bindings()
        or (
            expected_source_review is not None
            and result.get("source_review") != dict(expected_source_review)
        )
    ):
        raise ValueError("N5 full-panel V6 result recovery binding changed")
    shadow = copy.deepcopy(dict(result))
    shadow["schema"] = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_result_v1"
    shadow["source_review"] = _shadow_review(shadow["source_review"])
    shadow["experiment"] = _shadow_experiment()
    shadow["authority_bindings"] = _shadow_authority_bindings()
    shadow["attempt"]["scope"] = "one_exclusive_fresh_full_panel_attempt"
    result_licenses = shadow.get("licenses")
    if isinstance(result_licenses, dict) and "v6_training_authorized" in result_licenses:
        result_licenses["v5_training_authorized"] = result_licenses.pop(
            "v6_training_authorized"
        )
    shadow_core = dict(shadow)
    shadow_core.pop("content_sha256")
    shadow["content_sha256"] = canonical_json_sha256(shadow_core)
    shadow_expected = None if expected_source_review is None else _shadow_review(expected_source_review)
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained
    retained.validate_result_structure(shadow, expected_source_review=shadow_expected)
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("N5 full-panel V6 result schema changed")
    return dict(result)


def parse_bound_path(value: str) -> tuple[Path, str]:
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("artifact binding must be PATH:SHA256")
    path_text, digest = value.rsplit(":", 1)
    if not path_text or not is_sha256(digest):
        raise ValueError("artifact binding is malformed")
    return Path(path_text).resolve(strict=True), digest


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "artifact_binding", "authority_bindings", "canonical_json_bytes",
    "canonical_json_sha256", "experiment_contract", "frozen_source_bindings",
    "is_sha256", "licenses", "load_hashed_json", "parse_bound_path",
    "parse_json", "preflight_source_review", "preflight_static_authority",
    "read_hashed_bytes", "read_regular_bytes", "read_regular_bytes_at",
    "retained_v1_source_bindings",
    "retained_v2_source_bindings", "source_review_binding",
    "validate_evaluation_structure", "validate_reservation_structure",
    "validate_result_structure",
]

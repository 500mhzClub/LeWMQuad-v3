"""Prospective full-bank scorer-fit corpus V2 design authority.

The six-of-twelve allocation is an immutable exact infeasibility.  This
module supersedes only that partial-subset layer and freezes the prospective
replacement in which every selected state receives candidates 0--11 once.
It contains no simulator, solver, branch, frame, latent, scorer, predictor,
checkpoint, or outcome reader.

Historical scientific authorities remain immutable.  Chained preselection
corrections may reopen only their exact predecessor corrections and the
issued classification/design for custody validation; they cannot replace any
payload.  Every authority is installed with ``O_EXCL``, file and directory
``fsync``, and read-only permissions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from fnmatch import fnmatchcase
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from lewm.oracle import (
    go2_small_completion_global_execution_amendment_v1 as GLOBAL_AUTHORITY,
)


ROOT = Path(__file__).resolve().parents[2]

MASK_CLASSIFICATION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_rotation_mask_classification")
MASK_CLASSIFICATION_STATUS = (
    "PASS_ALL_OLD_ROTATION_CONDITIONS_PARTIAL_SUBSET_ALLOCATION_ONLY")
MASK_CLASSIFICATION_SELF_KEY = "rotation_mask_classification_digest"
DESIGN_SCHEMA = "go2_scorer_fit_corpus_v2_design_amendment"
DESIGN_STATUS = "ISSUED_PROSPECTIVE_FULL_TWELVE_CANDIDATE_BANK"
DESIGN_SELF_KEY = "scorer_fit_corpus_v2_design_digest"
SOURCE_CORRECTION_V1_SCHEMA = (
    "go2_scorer_fit_corpus_v2_preselection_source_correction_v1")
SOURCE_CORRECTION_V1_STATUS = (
    "ISSUED_PRESELECTION_SOURCE_CORRECTION_AFTER_REGISTERED_"
    "DEVELOPMENT_MANIFEST_ALIAS_FAILURE")
SOURCE_CORRECTION_SELF_KEY = (
    "scorer_fit_corpus_v2_source_correction_digest")
SOURCE_CORRECTION_V2_SCHEMA = (
    "go2_scorer_fit_corpus_v2_preselection_source_correction_v2")
SOURCE_CORRECTION_V2_STATUS = (
    "ISSUED_CHAINED_PRESELECTION_SOURCE_CORRECTION_AFTER_MANAGED_"
    "FINAL_EVAL_ABSENCE_PIN_FAILURE")
SOURCE_CORRECTION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_preselection_structural_validation_"
    "correction_v1")
SOURCE_CORRECTION_STATUS = (
    "ISSUED_CHAINED_PRESELECTION_STRUCTURAL_VALIDATION_CORRECTION_"
    "AFTER_SIGNED_BODY_CLEARANCE_DOMAIN_FAILURE")
MANIFEST_REPLAY_CORRECTION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_post_install_manifest_replay_correction_v1")
MANIFEST_REPLAY_CORRECTION_STATUS = (
    "ISSUED_POST_INSTALL_MANIFEST_REPLAY_CANONICALIZATION_CORRECTION")
MANIFEST_REPLAY_CORRECTION_SELF_KEY = (
    "scorer_fit_corpus_v2_manifest_replay_correction_digest")
ENCODER_IMPORT_CORRECTION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_post_smoke_encoder_import_"
    "compatibility_correction_v1")
ENCODER_IMPORT_CORRECTION_STATUS = (
    "ISSUED_POST_SMOKE_PRE_LATENT_ENCODER_IMPORT_COMPATIBILITY_CORRECTION")
ENCODER_IMPORT_CORRECTION_SELF_KEY = (
    "scorer_fit_corpus_v2_encoder_import_correction_digest")
ENCODER_COMPUTE_DTYPE_CORRECTION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_post_import_pre_latent_encoder_compute_"
    "dtype_correction_v1")
ENCODER_COMPUTE_DTYPE_CORRECTION_STATUS = (
    "ISSUED_POST_IMPORT_PRE_LATENT_ENCODER_COMPUTE_DTYPE_CORRECTION")
ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
    "scorer_fit_corpus_v2_encoder_compute_dtype_correction_digest")
ENCODER_PATH_PROJECTION_CORRECTION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_post_base_smoke_logical_path_projection_"
    "correction_v1")
ENCODER_PATH_PROJECTION_CORRECTION_STATUS = (
    "ISSUED_POST_BASE_SMOKE_PRE_ZERO_NEW_LOGICAL_PATH_PROJECTION_CORRECTION")
ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY = (
    "scorer_fit_corpus_v2_encoder_path_projection_correction_digest")
BRANCH_REDRIVE_PROJECTION_CORRECTION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_post_partial_corpus_branch_redrive_"
    "projection_correction_v1")
BRANCH_REDRIVE_PROJECTION_CORRECTION_STATUS = (
    "ISSUED_POST_PARTIAL_CORPUS_BRANCH_REDRIVE_PROJECTION_CORRECTION")
BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY = (
    "scorer_fit_corpus_v2_branch_redrive_projection_correction_digest")
FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SCHEMA = (
    "go2_scorer_fit_corpus_v2_single_shard_regeneration_prepared_v1")
FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STATUS = (
    "PREPARED_EXACT_ONCE_SINGLE_REGISTERED_SMOKE_SHARD_REGENERATION")
FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY = (
    "single_shard_regeneration_prepared_digest")
FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SCHEMA = (
    "go2_scorer_fit_corpus_v2_single_shard_regeneration_complete_v1")
FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STATUS = (
    "COMPLETE_EXACT_ONCE_SINGLE_REGISTERED_SMOKE_SHARD_REGENERATION")
FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY = (
    "single_shard_regeneration_complete_digest")
FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_CONTRACT_SCHEMA = (
    "go2_scorer_fit_corpus_v2_single_shard_regeneration_transaction_"
    "contract_v1")
FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_CONTRACT_STATUS = (
    "PROSPECTIVE_PRE_OUTCOME_EXACT_ONCE_SINGLE_SHARD_REGENERATION_"
    "TRANSACTION")

BRANCH_GENERATED_ROOT_RELATIVE_PATH = Path(
    ".generated/go2_branch_corpus_v1_2")
UTILITY_V2_ROOT_RELATIVE_PATH = Path(
    ".generated/go2_utility_scorer_fit_corpus_v2")
MANAGED_GENERATED_ROOTS = (
    BRANCH_GENERATED_ROOT_RELATIVE_PATH,
    Path(".generated/go2_utility_scorer_v1_2"),
    UTILITY_V2_ROOT_RELATIVE_PATH,
)
SCORER_FIT_RELATIVE_PATH = BRANCH_GENERATED_ROOT_RELATIVE_PATH / "scorer_fit"
MASK_CLASSIFICATION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_rotation_mask_classification.json"
)
DESIGN_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_design_amendment.json"
)
SOURCE_CORRECTION_V1_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_preselection_source_correction_v1.json"
)
SOURCE_CORRECTION_V2_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_preselection_source_correction_v2.json"
)
SOURCE_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_preselection_structural_validation_"
    "correction_v1.json"
)
MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_post_install_manifest_replay_correction_v1.json"
)
ENCODER_IMPORT_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_post_smoke_encoder_import_correction_v1.json"
)
ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_post_import_encoder_compute_dtype_correction_v1.json"
)
ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_post_base_smoke_path_projection_correction_v1.json"
)
ENCODER_PATH_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    ".scorer_fit_corpus_v2_post_base_smoke_path_projection_correction_v1."
    "json.staged"
)
BRANCH_REDRIVE_PROJECTION_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_post_partial_corpus_branch_redrive_projection_"
    "correction_v1.json"
)
BRANCH_REDRIVE_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    ".scorer_fit_corpus_v2_post_partial_corpus_branch_redrive_projection_"
    "correction_v1.json.staged"
)
FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH / "single_shard_regeneration_transaction_v1")
FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH = (
    FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH /
    "prepared.json")
FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STAGED_RELATIVE_PATH = (
    FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH /
    "prepared.json.staged")
FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH = (
    FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH /
    "complete.json")
FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STAGED_RELATIVE_PATH = (
    FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH /
    "complete.json.staged")
FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_RELATIVE_PATH = (
    FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH /
    "candidate_0_horizon_original.f16")

ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT = (
    "76bc465cb33ef94d535b433c83660d94335bee00")
SOURCE_CORRECTION_V1_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/encode_go2_branch_corpus_v1_2.py",
    "scripts/train_go2_utility_scorer_v1_2.py",
    "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT = (
    "6e59f26f3d4729772cac19245c96dc6451da2447")
IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST = (
    "0529e634d6b31c22395028f26a2330008fa0483887d5666758de8ad83f756c53")
SOURCE_CORRECTION_V2_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT = (
    "68bcec1a60a5354e0ae5125e5dc8478413ab94b5")
IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST = (
    "abc728d6b9f540befa184e58effa0fc9b21c018b584a91aba5ad7707ecaa948e")
SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT = (
    "66c6c314dbca04cbf702aa7ea2d45d6096e3945d")
IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST = (
    "5206c741b85c138dfaf6747df3f6852ac1446e99539b169d80d2fc744e8a6c35")
IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_BINDING = {
    "path": str(SOURCE_CORRECTION_RELATIVE_PATH),
    "schema": SOURCE_CORRECTION_SCHEMA,
    "self_digest_key": SOURCE_CORRECTION_SELF_KEY,
    "self_digest": IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST,
    "raw_sha256": (
        "62c57488c154562eae7d3fda68a31f4e4d917213302bf6429273d82e5e71b036"),
    "byte_count": 123_868,
    "source_repository_commit":
        IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT,
}
MANIFEST_REPLAY_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT = (
    "72b0d771b748e777a9da47fca88a9d6cfb62d0ef")
ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_PATHS = (
    "lewm/tests/test_go2_scorer_fit_corpus_v2_design.py",
    "lewm/tests/test_go2_scorer_fit_corpus_v2_scorer_contract.py",
    "lewm/tests/test_run_go2_scorer_fit_full_bank_v2.py",
    "lewm/tests/test_dev_frozen_dense_representation_encoders_v1.py",
)
ENCODER_IMPORT_CORRECTION_DEV_ENCODER_HISTORICAL_BINDING = {
    "path": "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "role": "frozen_target_encoder_import_route",
    "exists": True,
    "byte_count": 12_741,
    "sha256": (
        "9fa9780376416cc0181d2e37980d5af8a1bb632dec82ec802e53e33991f74efb"),
}
ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT = (
    "e41d0ee9935cc22fcd940dd9d988f41137f35baf")
ENCODER_COMPUTE_DTYPE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/encode_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_PATHS = (
    "lewm/tests/test_go2_scorer_fit_corpus_v2_design.py",
    "lewm/tests/test_go2_scorer_fit_corpus_v2_scorer_contract.py",
    "lewm/tests/test_encode_go2_branch_corpus_v1_2.py",
    "lewm/tests/test_run_go2_scorer_fit_full_bank_v2.py",
)
IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = (
    "1d782743cbec903e4bfde5fb349c12b0122bb5a2b39a35f3552dd673e47e8bf0")
IMMUTABLE_ENCODER_IMPORT_CORRECTION_BINDING = {
    "path": str(ENCODER_IMPORT_CORRECTION_RELATIVE_PATH),
    "schema": ENCODER_IMPORT_CORRECTION_SCHEMA,
    "self_digest_key": ENCODER_IMPORT_CORRECTION_SELF_KEY,
    "self_digest": IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST,
    "raw_sha256": (
        "23a4830dc74031b92350d197031ba3de6e7ca53061aced3da63589bec6524078"),
    "byte_count": 189_622,
    "source_repository_commit":
        ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
}
ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING = {
    "path": "scripts/encode_go2_branch_corpus_v1_2.py",
    "role": "failed_full_bank_v2_bfloat16_encoder_route",
    "exists": True,
    "byte_count": 101_507,
    "sha256": (
        "2751e09562251004f9d1e478ee060f6746ded62a09a7ff5b929ed8f6a5e1cd6c"),
}
ENCODER_COMPUTE_DTYPE_UNCHANGED_DEV_ENCODER_BINDING = {
    "path": "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "role": "frozen_target_encoder_import_route",
    "exists": True,
    "byte_count": 15_564,
    "sha256": (
        "c5bb12ddc4711071dbdbac8c2ad6cc4b7528dd8ceb263b752fd539bd954aa9e2"),
}
ENCODER_COMPUTE_DTYPE_STAGE_A_FP32_SOURCE_BINDING = {
    "path": "scripts/encode_go2_counterfactual_fidelity_stage_a_v1_2.py",
    "role": "frozen_stage_a_true_latent_fp32_encoder_route",
    "exists": True,
    "byte_count": 27_974,
    "sha256": (
        "5f67ac9bb879467d2698265189603f4b4f3a84daa6595b4fecd1e3be03ba9d35"),
}
ENCODER_COMPUTE_DTYPE_UPSTREAM_ROPE_SOURCE_BINDING = {
    "repository_commit": "204698b45b3712590f06245fbfba32d3be539812",
    "path": "app/vjepa_2_1/models/utils/modules.py",
    "role": "frozen_upstream_rope_sdpa_failure_source",
    "byte_count": 16_963,
    "sha256": (
        "64be6a87bd9f18d385f4e44186db3347d1665e18a1f0511d51d3b305531562e2"),
    "first_scaled_dot_product_attention_line": 287,
}
ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT = (
    "dc4f59aec63d8cac8b12deaa5adaa78daeabb33b")
ENCODER_PATH_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/encode_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_PATHS = (
    "lewm/tests/test_go2_scorer_fit_corpus_v2_design.py",
    "lewm/tests/test_go2_scorer_fit_corpus_v2_scorer_contract.py",
    "lewm/tests/test_encode_go2_branch_corpus_v1_2.py",
    "lewm/tests/test_run_go2_scorer_fit_full_bank_v2.py",
)
BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT = (
    "a207c0730e46207638dc18da68c1f520bb24b73e")
BRANCH_REDRIVE_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)
IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST = (
    "597ae5c7237749dbf5d6b7813ffd2d4441b7244b4f241e57d6fbd5b7349203b8")
IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_BINDING = {
    "path": str(ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH),
    "schema": ENCODER_PATH_PROJECTION_CORRECTION_SCHEMA,
    "self_digest_key": ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY,
    "self_digest": IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST,
    "raw_sha256": (
        "c91ddecfd270c0d099352ad16af3a8a0280e6d7a5fa1d41dbbb876413f62ee90"),
    "byte_count": 293_121,
    "source_repository_commit":
        BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
}
IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST = (
    "5427a631eda8d79c5b4cae9cbb486830f6354e9f0a90e2fa4c416b4bd15cbb86")
IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_BINDING = {
    "path": str(ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH),
    "schema": ENCODER_COMPUTE_DTYPE_CORRECTION_SCHEMA,
    "self_digest_key": ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY,
    "self_digest": IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST,
    "raw_sha256": (
        "4d039bb99c047cc090a94afc8c0370ded3edd804b3b04b81f83424ecc0848473"),
    "byte_count": 228_110,
    "source_repository_commit":
        ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
}
ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING = {
    "path": "scripts/encode_go2_branch_corpus_v1_2.py",
    "role": "failed_physical_to_repository_root_path_projection_route",
    "exists": True,
    "byte_count": 106_281,
    "sha256": (
        "1846d96888c9660896b89cb221e2effc7110112672ae1c38f2db3ff60e9009f8"),
}
IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_SMOKE_BINDING = {
    "path": str(SCORER_FIT_RELATIVE_PATH / "smoke_encoding_receipt_v2.json"),
    "schema": "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1",
    "self_digest_key": "smoke_receipt_digest",
    "self_digest": (
        "d4b845f06f6915f734511812457601134bb5e1d42f7531915105b05f4442a34c"),
    "raw_sha256": (
        "61eb248f41afebf220d6c573281080f64abcc0315bc433cdc9bf277785192c7d"),
    "byte_count": 7_678,
}
IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_INDEX_BINDING = {
    "path": str(SCORER_FIT_RELATIVE_PATH / "latents_index_v2.json"),
    "schema": "go2_scorer_fit_corpus_v2_latents_index_v1",
    "self_digest_key": "latents_index_digest",
    "self_digest": (
        "4135319ced308eaadbbcf1e0966ed3409d9f79e752026eab6baa07ee8e64cc68"),
    "raw_sha256": (
        "8852c8e40e455c1c265cb7ba91c8d7000e6d7df6061344dc6d9025c4d5ccabae"),
    "byte_count": 14_715,
}
IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_SUMMARY_BINDING = {
    "path": str(
        SCORER_FIT_RELATIVE_PATH / "encoding_invocation_summary_v2.json"),
    "schema": "go2_scorer_fit_corpus_v2_encoding_invocation_summary_v1",
    "raw_sha256": (
        "d5b28fb926f935cab4eafa712c563b4ae55743821e783314a4d2e61cf02c9104"),
    "byte_count": 522,
}
IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_SHARD_INVENTORY = (
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/context/"
            "c9bf42df529b75ebaf7e9053be059ee7ad639e690f501ba6b8750c968a37634e.f16")),
        "sha256": "1979fe99fada521a70334afbae5f6856dc62d1651ff9fdd3f5a1d3023fd57273",
        "byte_count": 4_718_592, "shape": [3, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "589e6cece3045e5002e8a258884796f76781b9d7aad150030c123722397e36b5.f16")),
        "sha256": "ffbfd64f34396a6ea96575ebdb117e16989542656ae1e22c449e9525d04ccb31",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "091910bef3cc5d0faa7182a45ff539537a992391e3fc5cca8ef071e3d94c2e6d.f16")),
        "sha256": "1fb08f53ac15880efe98a77e95f41543fbb8a1cb2c6a49c1ae6a37316e56713b",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "877bb7904f228deab46f15b194fa59e409dae43c7fd400e10ad6ba342e877073.f16")),
        "sha256": "a7e98b2ba3811c5b65c50502992a243aa5f1588af8686b43576fad9da068689f",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "8bec6e8d63d6baba98225b8014d37fbdf835d0bb0d8cc27ff75f94522985d80e.f16")),
        "sha256": "f792edf4ba1b652d72141203fd1931cd44de960f98c891161508e5c5a432b339",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "64304ded8767d895b0b4c5e4d699723528453179912dde97c2a6d83e467882dd.f16")),
        "sha256": "8573bbcdc2842d78e7f3e4913e0f97e67893683e89d4bcbede45e9bcf15a50cc",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "b1a2d43db8c20aec5badb08ade0a5b6af2cb8645dff0463d52502188d9972acb.f16")),
        "sha256": "a11156a739c4311db462c08fe18e2382ea4cb6f059d9c6bf51a058bca712b40a",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "5522e6c91a93b924105df80582c1e452f52387cd8b0f0ee65277e4111fafe114.f16")),
        "sha256": "90bb6b4b85e68b5715ee1f4bae0884070e5605e249f90400e98e59c18cb139e1",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "2231c264dae87c4ef543e3bc64a8605889bed32079427f813d114fdcecc3b933.f16")),
        "sha256": "3a5d346d44f641586e0877dc76cc90f4edba61b2d3a979d13c55879eba92925b",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "5bd95ba3985d1929ef40cd6c94c581d3d875ad9ea0f365efbfcde3d5b44642e5.f16")),
        "sha256": "ccd9d558c8168cf752a6c106cfbc24351073abdc7a7fb15583e297ad5a8e5aab",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "ec0fd3db4e44e0658f1295f2fab1a9b7dcecebeab64ea5063fda5508590bfca3.f16")),
        "sha256": "6d1b66ee0300013ee8a23f8d0667f43544900ab153f6fbac18131a6ad32f1ca9",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "b48177c5a83a63b6ad7b59e5f91d527622f6549e4a03191ccd170a5e0c7a51c7.f16")),
        "sha256": "0594e0e9d1263a7f7c190d09536908b1ca1bf4a3f80e69071fd55fd59165b009",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
    {
        "path": str(SCORER_FIT_RELATIVE_PATH / (
            "latents_v2/horizon/"
            "26cb63511286b1024b2ac37923abdfddbb64af3a5c6cc7c9a21ea58a0256cd30.f16")),
        "sha256": "64ed8aba7190c911644296f5015dbb3668aaf255295e0f7150095ba5d904e3bf",
        "byte_count": 6_291_456, "shape": [4, 768, 1024],
    },
)
IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_ARTIFACT_BUNDLE = {
    "schema": "go2_scorer_fit_corpus_v2_path_projection_failure_base_bundle_v1",
    "latent_index_binding": copy.deepcopy(
        IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_INDEX_BINDING),
    "encoding_invocation_summary_binding": copy.deepcopy(
        IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_SUMMARY_BINDING),
    "base_smoke_receipt_binding": copy.deepcopy(
        IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_SMOKE_BINDING),
    "latent_shard_inventory": copy.deepcopy(list(
        IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_SHARD_INVENTORY)),
    "context_latent_shard_count": 1,
    "horizon_latent_shard_count": 12,
    "total_latent_shard_count": 13,
    "total_latent_storage_bytes": 80_216_064,
}

BRANCH_REDRIVE_FAILURE_STATE_ID = (
    "scorer_fit-large_enclosed_maze-completion_enriched-00")
BRANCH_REDRIVE_FAILURE_STATE_IDENTITY_DIGEST = (
    "451efe30e767df2f0d2d5cb8cc0b7813c36b4f66aff5fb6c3a9fc3a8284dd015")
BRANCH_REDRIVE_FAILURE_REASON = (
    "redrive_previous_applied_command_snapshot_task_status_"
    "completion_full_bank_l_max_eligible_mismatch")
IMMUTABLE_BRANCH_REDRIVE_PARTIAL_CORPUS_BINDING = {
    "schema": "go2_scorer_fit_corpus_v2_partial_corpus_failure_boundary_v1",
    "corpus_receipt": {
        "path": str(SCORER_FIT_RELATIVE_PATH / "corpus_receipt_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_full_bank_completion_receipt_v1",
        "raw_sha256": (
            "d3326d80c1f31f8914cf6ca9cd65fd0ad761e7718886001df0b9339b021b4c27"),
        "byte_count": 15_939,
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING",
        "complete": False,
        "state_count": 120,
        "completed_states": 10,
        "expected_branches": 1_440,
        "attempted_branches": 120,
        "valid_branches": 120,
        "invalid_branches": 0,
        "corpus_digest": (
            "d3bf3622550d8ac59cdda55470c751ab7cd0e8aacf18ce7dfd8e2446f1f62928"),
        "branch_rows_sha256": (
            "b9cadbb5feb4925e9924ca791d1210477336fadb468bd91b31bf41a0ae909e3a"),
        "state_manifest_digest": (
            "db79efce49d949522832d920b23a38292a491dc9e6fb2cbf2b8e0a5176fb062e"),
        "full_bank_assignment_manifest_digest": (
            "a91d6d211f5b07270df5a66262ce4ba218e8a3925ae5f8aba196b8c10f4959f4"),
    },
    "branch_rows_ledger": {
        "path": str(SCORER_FIT_RELATIVE_PATH / "branch_rows_v2.jsonl"),
        "raw_sha256": (
            "b9cadbb5feb4925e9924ca791d1210477336fadb468bd91b31bf41a0ae909e3a"),
        "byte_count": 2_713_203,
        "content_parsed_for_correction": False,
        "outcome_or_label_value_read_for_correction": False,
    },
}

_BRANCH_REDRIVE_INVALID_RECEIPT_ROWS = (
    (0, "86ccf44fe9809ecc3079d25a8d131756bf2fa6dcccb27a42b8f9a22cca095b35",
     "78b3b3ef857b5717b3bfcf03427995cf717bd473220118c8351ced88cc62f5a3",
     "51bff755021280624fe442e63f24c5fbcd4fafc128663d7a3aca6ca2bf561d7f",
     "253237427519f299d1d90610876e430eb37acfe59313d62c0993c35a10816dab", 5070),
    (1, "6a9bad62a304a0ec96935b8a6be31184359a47fda26fe6a9bee3088559a50dc1",
     "5ad2adb75982f45c261cef1ad1a9200ca571ec327d3632b4da337b6bd85f592a",
     "26c5f1a6a65d692475c5b1c93deebca8a955aa54de7db55a6dba31c97bfb8453",
     "8ebc9e8a8d42fc7cc16a9bde8a1e0af90a6868013116d01c1c678353b32884e8", 5081),
    (2, "0e469929be64af8737162813ae56c5a65e339b2491224a4d5368b0421047f958",
     "6ed27d16f2b8dc109dbadd0439f97bed4b77b8c504eb6d54bdf9e937f9613ad5",
     "dca59f6520d3901cdb36732963e928de441085b8cea05dba7fac23e9fdf958d5",
     "f586b26b3a44355d4dc7c439d9b41e0ca3cbf55997da940c47b6660746281249", 5071),
    (3, "086a69c8db3528844b69764bf9d34c06357adfd73aa5c843df988c99750fd505",
     "7590d19f2e3dbdc21472cc485b4379e49c21bd6350fa6174f75e52ea800504a1",
     "dc09b1e4361e9bac502d19ec4cbc4a937c155f39593c5acd5ab99b93f6539491",
     "5ddb375e385aa75ffb119e700eb4a5e513176f257aad002bc350ebd708008ded", 5060),
    (4, "516b507462a9c25fb300fa044665f0e2537b515ed8b11585a9a2541ad0b194bb",
     "f7e6262ac893c6efc7cf6ab12de10906c197051f6449fbe2b8e23590c562827d",
     "02d8a63dde493ae3f9c10272aec086300a3849d54b83ecf0a9cef70e624d9c15",
     "4502202ce26525172abc9b3e9636b3494a78e14fc09eb4650fe6e69d0b9220f6", 5065),
    (5, "f5f6111035f44540c0d9c1832e9231551203207cd0643109123ccdbe6d0df4fd",
     "fb6b655a4f076bd56b2c5d61e77f2856fd033b3a4ed216d53170734fbde81a52",
     "7ae4f950b11b5661897445860c80d762cb9a9b4f009e22051a90a573a9a82239",
     "e6396de712f3f81539683807636b61a7a830d3528291db5e2fdb3da06fe9d814", 5061),
    (6, "e0725abb6d4a52b8312415d6ffcc67fae356c3ca71448e895c3830702209c01c",
     "c8b0db54ebf47108bc0e7bc3ba1e2572ddbb95eeb619fea907644853c6396bfe",
     "c4bdb50e3a77e04de2182461d9be33790f5ca0c6cb7eb784ef291ca468188432",
     "a18244daedbd23b17e2f28d5b17ef81098d35e427770d4b5c1ca44eec388c0f8", 5066),
    (7, "d8d0a55ee5fdd3b7fbe851fe494b9ee67a770bba61ecaa09434dd1cfb1d5b1a1",
     "655b1a942b5ddb8953908d01fc10c71df45ec632bbee0b8f6a970d0532f482db",
     "feb25788c043ff39caa47e822facc0ff0b9645fa62a23801655732a3d5ef0595",
     "a8da7b8d3ac3f1556e691713a4ae7e3d2c83bc2517bcfc7ddc7b1a22dab2aba4", 5071),
    (8, "74874a5e55839fb387c0971990292111995203c82d9af06d3653ba2e6af390a1",
     "ce107f834fd86fb5bb9649bec586c1ed653e2abc8acd34ae3d8ad90b224d1e9a",
     "49f88a710774624d0e9381fcaaa425bd943de7435e82001c863df070028725e1",
     "62a7f4cf9ff581c932c2b261411c25369d8267cb16bf53f695395856033bad58", 5074),
    (9, "3b830816e8edb314448d95941df2d1c4508e0abe785af29247c3f068209b3223",
     "e3a9b4763faa2ffce2d062042539d8b5ced37c6f81fc582a1070ed7fae27e606",
     "38eb5410a6df1702063d26ebdcde45c4f9a2fbff4dd013dcb2127b54b29207f1",
     "4d8800ecea7cfed771c4aef28a5ad59fd75e5e23c17b654efaaa831e41a048de", 5071),
    (10, "7f31e5bc5937147255375ea69e89108a2936d6bfe53497cb33e17f7b98b5027b",
     "c3b2824f5fb0bac79d9196506a10e3503128ae3e636d1a1d4b31d202bd35a38a",
     "b827b2d3c59dbe6c4466be24dbd3cdfcac7721fca7a37cd9d6852deb43f6c81b",
     "6b7c8128af251413d472a5aa7d6a9ff7933c37dba05194e004f03bc4a15f06e9", 5058),
    (11, "38f333b84c458a1a441e8a3e2c795a76a51be8dfb973a7db6c72b36cc4e135ca",
     "bb756fc9cb9c2ea69db5aef20c24a7430dbc9845eb2a968f10c235cd72d65b0f",
     "afde9a40d0f43fb52dabc8cf39f7eddc12ae12424739e2947ca64eaf0593303b",
     "1513f4e5fca909a7e291f336626082d0e77281ecd0ffe5fb8c6f9536d4471766", 5035),
)
IMMUTABLE_BRANCH_REDRIVE_INVALID_ATTEMPT_RECEIPT_BINDINGS = tuple({
    "path": str(SCORER_FIT_RELATIVE_PATH / "invalid_attempts_v2" /
                "redrive_records" /
                f"{branch}.{BRANCH_REDRIVE_FAILURE_REASON}.invalid.json"),
    "schema": "go2_scorer_fit_corpus_v2_full_bank_branch_row_v1",
    "raw_sha256": raw_sha256,
    "byte_count": byte_count,
    "state_id": BRANCH_REDRIVE_FAILURE_STATE_ID,
    "state_index": 10,
    "state_identity_digest": BRANCH_REDRIVE_FAILURE_STATE_IDENTITY_DIGEST,
    "candidate_index": candidate_index,
    "branch_identity_digest": branch,
    "assignment_identity_digest": assignment,
    "branch_row_digest": row_digest,
    "invalid_reason": BRANCH_REDRIVE_FAILURE_REASON,
} for candidate_index, branch, assignment, row_digest, raw_sha256, byte_count
   in _BRANCH_REDRIVE_INVALID_RECEIPT_ROWS)

IMMUTABLE_BRANCH_REDRIVE_COMPLETED_SMOKE_BUNDLE = {
    "schema": "go2_scorer_fit_corpus_v2_completed_smoke_boundary_v1",
    "encoder_path_projection_correction_digest":
        IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST,
    "complete_transaction_receipt": {
        "path": str(FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH),
        "schema": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SCHEMA,
        "self_digest_key": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY,
        "self_digest": (
            "881bdc5aad33f2227c8b642b15ada1bd007e2e769db1bd1a743cede0f639914e"),
        "raw_sha256": (
            "36e0761e5dc75cc55fb9b179384005b9853f1d0c97c7de655ea2d628920717f3"),
        "byte_count": 7_345,
    },
    "final_smoke_receipt": {
        "path": str(SCORER_FIT_RELATIVE_PATH / "smoke_encoding_receipt_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1",
        "self_digest_key": "smoke_receipt_digest",
        "self_digest": (
            "4bd3cbeceabb63dfd5d04f896c99672c59d6fd9d17a289c3e9f263f08477b4c6"),
        "raw_sha256": (
            "2977638b5b3d93eb0e2fa43da0bc166993d5a536954a17e622d47b7d5b62590b"),
        "byte_count": 8_207,
    },
    "smoke_latent_index": {
        "path": str(SCORER_FIT_RELATIVE_PATH / "latents_index_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_latents_index_v1",
        "self_digest_key": "latents_index_digest",
        "self_digest": (
            "78018ba53bdd1d42a7c748b0f76744c579fa127d631bd5328986d1b5328f3dad"),
        "raw_sha256": (
            "072c3139ff329662b93a4874fc36daeefd7a4aa5658f1feea9b8ab01fc77db85"),
        "byte_count": 14_830,
        "complete": False,
        "context_record_count": 1,
        "horizon_record_count": 12,
    },
}

ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT = {
    "schema": FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_CONTRACT_SCHEMA,
    "status": FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_CONTRACT_STATUS,
    "complete": True,
    "transaction_version": 1,
    "transaction_directory_relative_path": str(
        FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH),
    "designated_target_selection": {
        "producer_projection": (
            "load_and_validate_full_bank_v2_encoding_smoke_for_consumption"),
        "record_kind": "horizon_latent",
        "candidate_index": 0,
        "selection_rule": "MINIMUM_INTEGER_CANDIDATE_INDEX",
        "required_shape": [4, 768, 1024],
        "required_path_parent": str(
            SCORER_FIT_RELATIVE_PATH / "latents_v2/horizon"),
        "candidate_outcome_or_label_used": False,
    },
    "prepared_receipt_contract": {
        "schema": FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SCHEMA,
        "status": FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STATUS,
        "self_digest_key": FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY,
        "relative_path": str(
            FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH),
        "staged_relative_path": str(
            FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STAGED_RELATIVE_PATH),
        "complete_value": False,
        "required_before_target_move": True,
    },
    "backup_contract": {
        "relative_path": str(
            FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_RELATIVE_PATH),
        "same_filesystem_as_active_target_required": True,
        "must_be_absent_before_atomic_move": True,
        "atomic_move_not_copy_or_unlink": True,
        "atomic_move_primitive": "RENAME_NOREPLACE",
        "no_overwrite_allowed": True,
        "absence_precheck_alone_is_not_no_overwrite": True,
        "retained_backup_exact_reopen_before_durability_fsync_required": True,
        "retained_backup_file_fsync_after_move_required": True,
        "destination_directory_fsync_before_source_directory_required": True,
        "source_directory_fsync_after_destination_directory_required": True,
        "moved_resume_must_reestablish_backup_file_destination_directory_and_"
        "source_directory_durability_before_regeneration": True,
        "device_id_inode_mode_and_link_count_must_match_prepared_target": True,
        "retained_after_complete": True,
        "registered_as_active_latent": False,
    },
    "complete_receipt_contract": {
        "schema": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SCHEMA,
        "status": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STATUS,
        "self_digest_key": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY,
        "relative_path": str(
            FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH),
        "staged_relative_path": str(
            FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STAGED_RELATIVE_PATH),
        "complete_value": True,
        "required_before_pass_smoke_publication": True,
    },
    "non_target_custody_contract": {
        "required_row_count": 12,
        "row_keys": [
            "path", "sha256", "byte_count", "shape", "device_id",
            "inode", "mode_octal", "link_count", "user_id", "group_id",
            "access_time_ns", "modification_time_ns",
            "metadata_change_time_ns",
        ],
        "sha256_read_mode": "O_NOATIME_O_NOFOLLOW",
        "pretransaction_and_precomplete_canonical_digest_must_match": True,
        "target_candidate_index_zero_excluded": True,
    },
    "immutable_receipt_publication": {
        "direct_write_to_final_path_allowed": False,
        "same_directory_staged_file_required": True,
        "staged_file_create_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW"],
        "staged_file_fsync_required": True,
        "staged_file_mode_octal_before_publication": "0444",
        "atomic_no_overwrite_publication": "LINK_STAGED_TO_FINAL",
        "final_path_must_be_absent_before_publication": True,
        "staged_path_unlink_after_publication_required": True,
        "parent_directory_fsync_immediately_after_final_link_required": True,
        "parent_directory_fsync_immediately_after_staged_unlink_required": True,
        "exact_read_only_reopen_required": True,
        "idempotent_exact_reopen_required": True,
        "partial_or_nonexact_staged_file_recovery": {
            "prepared_staged_rebuild_allowed_only_when": (
                "PREPARED_FINAL_ABSENT_AND_TARGET_EXACT_AND_BACKUP_ABSENT_"
                "AND_COMPLETE_FINAL_ABSENT"),
            "complete_staged_rebuild_allowed_only_when": (
                "PREPARED_FINAL_EXACT_AND_TARGET_EXACT_AND_BACKUP_EXACT_"
                "AND_COMPLETE_FINAL_ABSENT"),
            "staged_file_only_unlink_and_parent_fsync_required": True,
            "active_target_or_backup_mutation_during_staged_rebuild_allowed":
                False,
            "final_receipt_unlink_or_overwrite_allowed": False,
            "all_other_partial_or_nonexact_staged_states": "FAIL_CLOSED",
        },
        "exact_final_and_exact_staged_link_recovery": {
            "both_paths_regular_non_symlink_read_only_required": True,
            "both_paths_exact_expected_bytes_required": True,
            "same_device_and_inode_hard_link_proof_required": True,
            "parent_fsync_before_staged_unlink_required": True,
            "staged_file_only_unlink_then_parent_fsync_required": True,
            "parent_fsync_after_staged_unlink_required": True,
            "final_receipt_target_or_backup_mutation_allowed": False,
        },
        "mutable_phase_receipt_allowed": False,
    },
    "authorised_mutation": {
        "transaction_count": 1,
        "active_context_latent_target_count": 0,
        "active_horizon_latent_target_count": 1,
        "non_target_registered_latent_count": 12,
        "target_atomic_move_count": 1,
        "target_regeneration_count": 1,
        "destructive_unlink_of_active_target_allowed": False,
        "branch_row_or_frame_write_allowed": False,
        "non_target_latent_write_allowed": False,
        "latent_value_read_for_transaction_control_allowed": False,
        "outcome_or_label_read_for_transaction_control_allowed": False,
        "restored_target_sha256_byte_count_shape_must_equal_prepared": True,
        "restored_target_device_id_must_equal_prepared": True,
        "restored_target_inode_must_differ_from_prepared": True,
        "restored_target_mode_and_link_count_must_equal_prepared": True,
        "restored_target_exact_reopen_and_file_fsync_before_complete_required":
            True,
        "restored_target_parent_directory_fsync_before_complete_required": True,
        "all_non_target_shard_bindings_must_remain_unchanged": True,
        "all_non_target_shard_bytes_device_inode_mode_link_size_and_times_"
        "must_remain_unchanged": True,
        "registered_stable_artifact_inventory_must_be_recomputed_live_before_"
        "complete": True,
        "prepared_lineage_must_equal_live_zero_new_manifest_assignment_corpus_"
        "branch_smoke_contract_and_corrections_before_every_precomplete_"
        "mutation": True,
        "complete_lineage_must_equal_prepared_and_original_protocol_pass_"
        "lineage_before_downstream_acceptance": True,
    },
    "resume_state_machine": [
        {
            "prepared": False, "target": "EXACT", "backup": False,
            "complete_receipt": False, "action": "CREATE_PREPARED",
        },
        {
            "prepared": True, "target": "EXACT", "backup": False,
            "complete_receipt": False, "action": "ATOMIC_MOVE_ONCE",
        },
        {
            "prepared": True, "target": "ABSENT", "backup": "EXACT",
            "complete_receipt": False, "action": "REGENERATE_TARGET_ONCE",
        },
        {
            "prepared": True, "target": "EXACT", "backup": "EXACT",
            "complete_receipt": False,
            "action": "VALIDATE_AND_CREATE_COMPLETE_WITHOUT_SECOND_MOVE",
        },
        {
            "prepared": True, "target": "EXACT", "backup": "EXACT",
            "complete_receipt": True,
            "action": "PUBLISH_OR_REOPEN_COMPLETE_BOUND_SMOKE_ONLY",
        },
    ],
    "unlisted_resume_state_action": "FAIL_CLOSED",
    "optional_validation_projection": {
        "required_fields": [
            "transaction_state", "prepared_present",
            "prepared_receipt_digest", "target_state", "backup_state",
            "complete_present", "complete_receipt_digest",
            "pass_smoke_state", "next_action",
            "prepared_staged_state", "complete_staged_state",
            "target_exact", "backup_exact",
            "target_backup_custody_exact",
            "regenerated_target_custody_exact",
            "encoder_path_projection_correction_digest",
            "single_shard_regeneration_transaction_contract_digest",
            "candidate_outcomes_used_for_selection",
            "final_200_state_corpus_generated",
        ],
        "transaction_states": [
            "UNSTARTED", "PREPARED_MOVE_PENDING",
            "MOVED_REGENERATION_PENDING", "RESTORED_COMPLETE_PENDING",
            "COMPLETE_SMOKE_PUBLICATION_PENDING", "COMPLETE",
        ],
        "target_states": ["NOT_APPLICABLE", "ABSENT", "EXACT"],
        "backup_states": ["ABSENT", "EXACT"],
        "staged_receipt_states": ["ABSENT", "EXACT", "PARTIAL_REGULAR"],
        "partial_or_nonexact_staged_receipt_recovery": {
            "prepared_allowed_only_in_unstarted_state": True,
            "complete_allowed_only_in_restored_complete_pending_state": True,
            "all_other_states": "FAIL_CLOSED",
        },
        "complete_staged_receipt_allowed_only_during_restored_complete_"
        "publication_or_exact_final_hardlink_cleanup": True,
        "pass_smoke_states": [
            "ABSENT_OR_PRETRANSACTION", "EXACT_BOUND_PROTOCOL_PASS",
            "VALID_REFRESHED_PASS_WITH_EXACT_PROTOCOL_PASS_ARCHIVE",
        ],
        "accepted_state_matrix": [
            {
                "transaction_state": "UNSTARTED", "prepared_present": False,
                "target_state": ["NOT_APPLICABLE", "EXACT"],
                "backup_state": "ABSENT", "complete_present": False,
                "pass_smoke_state": "ABSENT_OR_PRETRANSACTION",
                "next_action": (
                    "RUN_OR_RESUME_BASE_AND_ZERO_NEW_BEFORE_PREPARED"),
            },
            {
                "transaction_state": "PREPARED_MOVE_PENDING",
                "prepared_present": True, "target_state": "EXACT",
                "backup_state": "ABSENT", "complete_present": False,
                "pass_smoke_state": "ABSENT_OR_PRETRANSACTION",
                "next_action": "ATOMIC_MOVE_ONCE",
            },
            {
                "transaction_state": "MOVED_REGENERATION_PENDING",
                "prepared_present": True, "target_state": "ABSENT",
                "backup_state": "EXACT", "complete_present": False,
                "pass_smoke_state": "ABSENT_OR_PRETRANSACTION",
                "next_action": "RUN_REGENERATION_ENCODER_ONCE",
            },
            {
                "transaction_state": "RESTORED_COMPLETE_PENDING",
                "prepared_present": True, "target_state": "EXACT",
                "backup_state": "EXACT", "complete_present": False,
                "pass_smoke_state": "ABSENT_OR_PRETRANSACTION",
                "next_action": (
                    "CREATE_COMPLETE_WITHOUT_SECOND_MOVE_OR_REGENERATION"),
            },
            {
                "transaction_state": "COMPLETE_SMOKE_PUBLICATION_PENDING",
                "prepared_present": True, "target_state": "EXACT",
                "backup_state": "EXACT", "complete_present": True,
                "pass_smoke_state": "ABSENT_OR_PRETRANSACTION",
                "next_action": "PUBLISH_COMPLETE_BOUND_PASS_SMOKE_ONLY",
            },
            {
                "transaction_state": "COMPLETE", "prepared_present": True,
                "target_state": "EXACT", "backup_state": "EXACT",
                "complete_present": True,
                "pass_smoke_state": [
                    "EXACT_BOUND_PROTOCOL_PASS",
                    "VALID_REFRESHED_PASS_WITH_EXACT_PROTOCOL_PASS_ARCHIVE",
                ],
                "next_action": "NO_TRANSACTION_MUTATION",
            },
        ],
        "unlisted_or_invalid_artifact_combination": "FAIL_CLOSED",
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
    },
    "pass_smoke_publication": {
        "complete_receipt_durable_before_pass_smoke": True,
        "complete_receipt_binds_exact_final_smoke_bytes": True,
        "complete_binding_role": "ORIGINAL_SMOKE_PROTOCOL_PASS_BYTES",
        "original_protocol_pass_must_be_parsed_self_validated_and_cross_bound_"
        "to_prepared_and_complete_lineage": True,
        "original_protocol_pass_is_stable_historical_witness_after_full_"
        "corpus_receipts_advance": True,
        "complete_binding_initial_publication_path": str(
            SCORER_FIT_RELATIVE_PATH / "smoke_encoding_receipt_v2.json"),
        "pass_smoke_binds_prepared_receipt_transaction_id": True,
        "original_protocol_pass_omits_complete_digest_to_avoid_cyclic_"
        "self_binding": True,
        "pass_smoke_without_valid_complete_receipt_allowed": False,
        "complete_with_old_or_absent_smoke_action": (
            "PUBLISH_ONLY_THE_COMPLETE_BOUND_EXACT_SMOKE_BYTES"),
        "exact_successor_active_replay_requires_file_fsync_and_parent_"
        "directory_fsync_before_acceptance": True,
        "second_target_move_or_regeneration_after_complete_allowed": False,
        "later_complete_full_index_smoke_refresh_allowed": True,
        "original_protocol_pass_exact_archive_required_before_refresh": True,
        "original_protocol_pass_archive_directory": str(
            SCORER_FIT_RELATIVE_PATH / "superseded_receipts_v2"),
        "original_protocol_pass_archive_filename_rule": (
            "smoke_encoding_receipt_v2.<RAW_SHA256_FIRST_16_HEX>.json"),
        "refreshed_smoke_must_bind_prepared_and_complete_digests": True,
        "refreshed_smoke_must_bind_complete_full_latent_index": True,
        "refreshed_smoke_current_corpus_and_branch_smoke_may_advance_from_"
        "prepared_partial_smoke_lineage": True,
        "refreshed_smoke_state_assignment_scorer_and_correction_lineage_must_"
        "equal_prepared": True,
        "refreshed_smoke_must_be_fully_replayed_against_current_index_and_"
        "current_lineage": True,
        "full_consumer_must_validate_refreshed_smoke_and_exact_archived_"
        "protocol_pass": True,
        "active_full_index_digest_equality_may_be_relaxed": False,
    },
    "downstream_gate": {
        "full_corpus_generation_requires_complete_receipt_and_bound_pass_smoke":
            True,
        "training_requires_complete_receipt_and_bound_pass_smoke": True,
        "final_200_state_evaluation_authorised": False,
    },
}
IMMUTABLE_MANIFEST_REPLAY_CORRECTION_DIGEST = (
    "b35a46b02a51bb030d8777d6b081ec76445d1ebb081be9220e97f22519cbbe7c")
IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING = {
    "path": str(
        UTILITY_V2_ROOT_RELATIVE_PATH /
        "scorer_fit_corpus_v2_scorer_contract.json"),
    "schema": "go2_scorer_fit_corpus_v2_scorer_contract_artifact_v1",
    "self_digest_key": "contract_artifact_digest",
    "self_digest": (
        "4455fd397ce7665f02725924a64ab87b1e0e9a3506d9ba64edbcc9b4daa1e121"),
    "raw_sha256": (
        "6e6ae29b3dd38b50e6d87259dcb269df7fdbb0139152245cafb776a5776b1a3c"),
    "byte_count": 22_021,
    "source_repository_commit":
        ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
    "embedded_contract_schema":
        "go2_scorer_fit_corpus_v2_scorer_contract_v1",
    "embedded_contract_self_digest_key":
        "scorer_fit_corpus_v2_scorer_contract_digest",
    "embedded_contract_self_digest": (
        "8fc0edae875cba6487ff1a1a771f96b0157da1474ac00de4186ecdb41b66d5df"),
}
IMMUTABLE_ENCODER_IMPORT_FAILURE_BRANCH_SMOKE_BINDING = {
    "path": str(SCORER_FIT_RELATIVE_PATH / "smoke_branch_receipt_v2.json"),
    "schema": "go2_scorer_fit_corpus_v2_full_bank_branch_smoke_receipt_v1",
    "self_digest_key": "smoke_branch_receipt_digest",
    "self_digest": (
        "4b593daabbfc52ac4df9e389270c0cfed62d8634c2e95c6e1bf1e176d77592a3"),
    "raw_sha256": (
        "23957fbc97fa7789797ee5da0bf15e9723edffb545c248fbeb119692fc6cfeaa"),
    "byte_count": 8_176,
}
IMMUTABLE_ENCODER_IMPORT_FAILURE_CORPUS_RECEIPT_BINDING = {
    "path": str(SCORER_FIT_RELATIVE_PATH / "corpus_receipt_v2.json"),
    "schema": "go2_scorer_fit_corpus_v2_full_bank_completion_receipt_v1",
    "self_digest_key": "corpus_digest",
    "self_digest": (
        "c5da0ad188d99e513d505273f92dcbb1c2062c267d80ead5eae65760707e098a"),
    "raw_sha256": (
        "13d3711242b5d8bc5e81e71b771e90a3778bc40af3b7b3889d09b02703a00417"),
    "byte_count": 7_934,
}
INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS = (
    {
        "role": "small_completion_selection",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "full_bank_small_completion_selection_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_small_completion_selection_v1",
        "self_digest_key": "full_bank_small_completion_selection_digest",
        "self_digest": (
            "e2d889defc8ceb47742deabf3c79ccadc3b28bce3387b54370153599f739a529"),
        "raw_sha256": (
            "52b658b33b4d7325a7db3e928627f30cd41370bfc5882b844e94a3624017a31f"),
        "byte_count": 98_185,
        "mode": "0444",
    },
    {
        "role": "preoutcome_state_revalidation",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "full_bank_preoutcome_state_revalidation_v2.json"),
        "schema": (
            "go2_scorer_fit_corpus_v2_preoutcome_state_revalidation_v1"),
        "self_digest_key": "full_bank_preoutcome_state_revalidation_digest",
        "self_digest": (
            "d4bfde28459e7fe4f8c449177b1a14329e3e6dd6300fbfda92e1efdeeb86f25c"),
        "raw_sha256": (
            "9b274a958e1385d7cf453f939e19a5bd177ac39a1a4f425df616d7b3ccdfb71c"),
        "byte_count": 571_342,
        "mode": "0444",
    },
    {
        "role": "small_family_state_shard",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "state_shard_small_enclosed_maze_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_small_family_state_shard_v1",
        "self_digest_key": "state_shard_digest",
        "self_digest": (
            "35b558010ee71f3f61fdd0c56fcce622d61dab932a7bc770b4a616b14b1ab9d4"),
        "raw_sha256": (
            "fc4b7efd89a36d289b5b163f2be68f8b66d66c1bc3a770f1ed4018d88eda8710"),
        "byte_count": 30_389,
        "mode": "0444",
    },
    {
        "role": "assignment_manifest",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "full_bank_assignment_manifest_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_assignment_manifest_v1",
        "self_digest_key": "full_bank_assignment_manifest_digest",
        "self_digest": (
            "a91d6d211f5b07270df5a66262ce4ba218e8a3925ae5f8aba196b8c10f4959f4"),
        "raw_sha256": (
            "60171432ebe381ad31d41fee9d90549cbf4244333a7748fd2ec92b2bbea5d188"),
        "byte_count": 1_729_052,
        "mode": "0444",
    },
    {
        "role": "state_manifest",
        "path": str(SCORER_FIT_RELATIVE_PATH / "state_manifest_v2.json"),
        "schema": "go2_scorer_fit_corpus_v2_identity_manifest_v1",
        "self_digest_key": "state_manifest_digest",
        "self_digest": (
            "db79efce49d949522832d920b23a38292a491dc9e6fb2cbf2b8e0a5176fb062e"),
        "raw_sha256": (
            "35006fc0153ca087293aee29e7364282d773f02dbbcb7f781a135a8945f521ea"),
        "byte_count": 496_449,
        "mode": "0444",
    },
)

TERMINAL_SOURCE_REPOSITORY_COMMIT = (
    "e1bdbe7adc15d0aa85f69ffb9e97fa198eb152c5")
ACTIVE_GLOBAL_AMENDMENT_DIGEST = (
    "f4cbc2e5e7baa1c4cebda3dfde0e5f9744aa50a7d1d1c5c53704338a1bb6f822")
GLOBAL_EXACT_MODEL_DIGEST = (
    "57770fe998cb7d7cb952b122077fe7dd8daeadce5f70fa3150c6f8641e5162b4")
EXACT_INFEASIBILITY_DIGEST = (
    "eb9d347fee6d4b498cf02a2c0af51483304d9489aa98050a8212e99c737135dc")
TERMINAL_RECEIPT_DIGEST = (
    "d1a17c289e3993f6cad30cc3ff1725246075881c292e930e776d8bcd7fb215d4")
GLOBAL_MODEL_PLAN_DIGEST = (
    "c638e18ce376ed1cfa263c042ed9e89428c0ac5283b7f0c05429ed63baa1d082")
MODEL_EXECUTION_PLAN_DIGEST = (
    "c50375e73e3396561808619b3f882367f8befcdb88b99708364014cfb49c0d98")

ACTIVE_GLOBAL_AMENDMENT_BINDING = {
    "path": str(
        SCORER_FIT_RELATIVE_PATH /
        "small_completion_global_exact_preplan_integration_correction_v1.json"),
    "schema": GLOBAL_AUTHORITY.PREPLAN_INTEGRATION_CORRECTION_SCHEMA,
    "self_digest_key": GLOBAL_AUTHORITY.AMENDMENT_SELF_KEY,
    "self_digest": ACTIVE_GLOBAL_AMENDMENT_DIGEST,
    "raw_sha256": (
        "1c70ff5db95388dcc5dc5af004e7b0912091380bf7d81a98efa459cc6433f576"),
    "byte_count": 282_516,
    "source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
}
GLOBAL_MODEL_PLAN_BINDING = {
    "path": str(
        SCORER_FIT_RELATIVE_PATH /
        "small_completion_global_exact_model_plan_v1.json"),
    "schema": "go2_small_completion_global_exact_runner_plan_v1",
    "self_digest_key": "global_exact_model_plan_digest",
    "self_digest": GLOBAL_MODEL_PLAN_DIGEST,
    "raw_sha256": (
        "9d5fed3d851cfa7944dbccc53eaec4265b784042e590f471be4d1726c46ac9dd"),
    "byte_count": 15_420,
    "source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
}
GLOBAL_TERMINAL_INFEASIBILITY_BINDING = {
    "path": str(
        SCORER_FIT_RELATIVE_PATH /
        "small_completion_global_exact_terminal_infeasibility_v1.json"),
    "schema": "go2_small_completion_global_exact_runner_terminal_v1",
    "self_digest_key": "global_exact_terminal_result_digest",
    "self_digest": TERMINAL_RECEIPT_DIGEST,
    "raw_sha256": (
        "83724fa4ae370efd610d50be21a7424d8f7ab445c7679efbf5ff6af698684efb"),
    "byte_count": 5_108,
    "source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
}

CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9")
ACTIVE_SELECTOR_DIGEST = (
    "8c1d9f5ff1430fda6d9d80512afdba3070c78301befa57604aafcad9cb5c880b")
ORACLE_V1_2_DIGEST = (
    "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4")
SIX_OF_TWELVE_SUPERSESSION = (
    "SUPERSEDED_PRE_OUTCOME_SIX_OF_TWELVE_ALLOCATION_EXACTLY_INFEASIBLE")

FAMILIES = (
    "large_enclosed_maze", "local_composite_motifs", "loop_alias_stress",
    "medium_enclosed_maze", "open_obstacle_field", "rough_local_dynamics",
    "small_enclosed_maze", "visual_sensor_stress",
)
STRATA = ("general", "safety_enriched", "completion_enriched")
SPLIT_ROLES = ("fit", "calibration")
CANDIDATE_INDICES = tuple(range(12))
STATE_COUNT = 120
ASSIGNMENT_COUNT = 1_440

FULL_BANK_COUNT_CONTRACT: dict[str, Any] = {
    "schema": "go2_scorer_fit_corpus_v2_full_bank_count_contract",
    "state_count": STATE_COUNT,
    "family_count": 8,
    "states_per_family": 15,
    "strata": list(STRATA),
    "states_per_family_stratum": 5,
    "fit_state_count": 96,
    "calibration_state_count": 24,
    "fit_per_family_stratum": 4,
    "calibration_per_family_stratum": 1,
    "candidate_indices": list(CANDIDATE_INDICES),
    "candidate_count": 12,
    "candidates_per_state": 12,
    "assignments_total": ASSIGNMENT_COUNT,
    "per_candidate": {
        "overall": 120,
        "fit": 96,
        "calibration": 24,
        "per_stratum": 40,
        "per_family": 15,
        "fit_per_family": 12,
        "calibration_per_family": 3,
        "per_family_stratum": 5,
    },
    "unordered_candidate_pair_cooccurrence": 120,
    "candidate_by_family_distributions_identical": True,
    "candidate_by_stratum_distributions_identical": True,
    "candidate_by_split_distributions_identical": True,
    "candidate_by_goal_type_distributions_identical": True,
}

COMPLETION_ORDER_DOMAIN = (
    "LEWM_GO2_SCORER_FIT_CORPUS_V2_FULL_BANK_SMALL_COMPLETION_ORDER_V1")
COMPLETION_ORDERING_CONTRACT: dict[str, Any] = {
    "schema": "go2_scorer_fit_corpus_v2_small_completion_order_contract",
    "domain_separation_utf8": COMPLETION_ORDER_DOMAIN,
    "active_selector_contract_digest": ACTIVE_SELECTOR_DIGEST,
    "preimage": (
        "domain_separation_utf8 || 0x00 || active_selector_contract_digest || "
        "0x00 || compact_canonical_complete_structural_state_identity_json || "
        "0x00 || compact_canonical_designated_goal_identity_json"
    ),
    "hash": "SHA-256",
    "order": (
        "ascending SHA-256, then ascending complete structural identity JSON "
        "bytes as the final structural tie-break"
    ),
    "selection": {
        "calibration": (
            "first calibration-eligible full-bank-valid identity in order"),
        "fit": (
            "first four distinct fit-eligible full-bank-valid identities in "
            "the same order, excluding the calibration scene"),
        "failure": "fewer than one calibration and four fit scenes",
    },
    "candidate_outcome_or_downstream_metric_fields": [],
    "optimisation_or_solver": False,
}

PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS: dict[str, Any] = {
    "selection_digest": (
        "c20b4feceb865b25fb24e5534be5f84d14a5795d069ca2b0c14cd3f23d8ca9dd"),
    "scorer_fit_allocation_design_digest": (
        "a587b1de264dfb54176aa231e5183ae4b7b4229bbf65c02d62438f86af5e7116"),
    "candidate_allocator_contract_digest": (
        "bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e"),
    "candidate_allocation_amendment_digest": (
        "4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc"),
    "pre_identity_allocation_validation_digest": (
        "46efa42e3bdcad6df6cdcd4e404c2e8a796a9a331109a433cfbfffcfa18bf60d"),
    "invalid_scorer_identity_exclusion_digest": (
        "6d644c34b822fb5fb8e30906875047d1677aa730c2db584470cabdbe8bf6abc3"),
    "state_selector_amendment_digest": ACTIVE_SELECTOR_DIGEST,
    "state_selector_feasibility_receipt_digest": (
        "0e2013f40f506da6485bb5e2fe5a3108595243aeb9141a6437f8cac023642482"),
    "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
    "clean_source_launch_receipt_digest": (
        "1656fbd691a63a63338bcd2d4707275ac67fbcfdc90c8a2afe5504876a543e3e"),
    "source_repository_commit": (
        "d9d129e2bbea8519f7ed3186f3cfb3c661baba04"),
    "clean_source_binding_digest": (
        "563dfbe7f8a08b4cc059ca2f3f495f201fcf9693529ed783d79bb3dcb37663ee"),
    "bound_implementations_digest": (
        "4b5c70bcd35af13d284c55cb045af7c0b6d592be6e8e1f78a3a9b4293c0ac5f7"),
    "scorer_contract_artifact_digest": (
        "ed038cc3b773705c5ae1f4a1bc21d420ed43f0c6d7a408a5c911e250960ef78f"),
    "mixed_precontract_disposition_receipt_digest": (
        "fef1a98980bc41d63434367f518ff2876dbcf93afbea52ff8f555300d3220604"),
    "progress_contract_digest": (
        "840328d918f446bad1a5855e72f13f8937fc9a42eafd87818bf8cd94305e2c3d"),
    "safety_contract_digest": (
        "5cf4572be2490c1b6f748abc704fff3a3c15fb1ea8dc060e49314e2bbaf01e0f"),
    "oracle_v1_2_digest": ORACLE_V1_2_DIGEST,
    "scorer_contract_v1_2_digest": (
        "f268763ed9365205cd0b0001c4527afbf5e5d948846dbb891225a48acb74113a"),
    "boundary_digest": (
        "1faae05f843e6f02f0f354c63ab3bcad9404111140146b1355d025da3d0c7a92"),
    "render_contract_digest": (
        "2faa22e3b10a2c4199bdabdbc0ed0e1ff9c7c4ac48bb489daeb0fd70d5b65c17"),
    "preprocess_contract_digest": (
        "2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9"),
    "textured_v03_renderer_contract_digest": (
        "df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b"),
    "preprocessing_digest": (
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"),
    "target_encoder_digest": (
        "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"),
    "target_encoder_checkpoint_sha256": (
        "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"),
    "genesis_backend": "cpu",
}

FROZEN_PREDICTOR_QUALIFICATION: dict[str, Any] = {
    "commit": "ee47b47e7964c16360f265c4cfbe7f8181d16402",
    "stage_a_identity_digest": (
        "ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a"),
    "branch_corpus_digest": (
        "f84eb3271f1a3b7052bbf2e84240453e84772b0a530e60ec47f723a44e2e10e9"),
    "target_latent_index_digest": (
        "861285ec9c8fc6c92c6f3a31cade0f031172bf6818d76d1899634a60c7e5c291"),
    "direct_fidelity_and_retrieval_result_digest": (
        "3b5c500b4b1326056ce18c6276d7842f4230faec36f8f29cc65945f54527bbcb"),
    "occupancy_result_digest": (
        "09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6"),
    "occupancy_gate_digest": (
        "4bf9a92144fa728d953c9dffebb235c9b476ded59d7462a107fe2e6ade0894e4"),
    "result_report": {
        "path": (
            "docs/lewm_go2_counterfactual_predictor_qualification_v1_2_"
            "result_2026-08-11.md"),
        "raw_sha256": (
            "14a4276b1caee817a7097eb78f187c9e38b6d4c7eb70a16f88ec32ccd223894b"),
        "byte_count": 28_437,
    },
    "modified_or_rerun": False,
}

PRIOR_PREOUTCOME_FAILURE_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "role": "candidate_allocation_infeasibility",
        "path": (
            "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "allocation_failure_2026-08-11.json"),
        "schema": "go2_shared_utility_scorer_v1_2_preoutcome_failure_receipt",
        "semantic_status": "BLOCKED_PRE_OUTCOME_FROZEN_CONTRACT_INCONSISTENCY",
        "self_digest_canonicalization": "COMPACT_CANONICAL_JSON",
        "self_digest_key": "failure_receipt_digest",
        "self_digest": (
            "550c52f9a3ff04f8a564f6f28e75e9d36fc8bc0f73da4795b95dedc3ad2e3cab"),
        "raw_sha256": (
            "3e224158d43a4e75fc7a60436feaeb00cd538a5fabfae5a92983f7ede612df99"),
        "byte_count": 8_034,
    },
    {
        "role": "state_selection_contract_infeasibility",
        "path": (
            "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "state_selection_failure_2026-08-11.json"),
        "schema": (
            "lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "state_selection_failure_v1"),
        "semantic_status": "BLOCKED_PRE_OUTCOME_SCIENTIFIC_CONTRACT_INFEASIBILITY",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": "failure_report_digest",
        "self_digest": (
            "47c2bcc7cfaf79b328cd5a1cf2823554f2553fc419e020559fde1351df2ca75f"),
        "raw_sha256": (
            "26e372229af4bcde8242062b0b9f9d9c5ba85bdb073537812e9395a608c2455d"),
        "byte_count": 8_586,
    },
    {
        "role": "selector_feasibility_amendment_v2_failure_report",
        "path": (
            "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "selector_feasibility_failure_2026-08-12.json"),
        "schema": (
            "lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "selector_feasibility_failure_v1"),
        "semantic_status": "TERMINAL_PRE_OUTCOME_SELECTOR_FEASIBILITY_FAILURE",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": "failure_report_digest",
        "self_digest": (
            "81637cdf3889dc0856ea97aee9a644f182855ef49c4e466eee3f8aed4134a0b8"),
        "raw_sha256": (
            "db2e025fe71164943bb214a602dfdeb249d629de2420aa758bb97717c8974b49"),
        "byte_count": 28_618,
    },
    {
        "role": "selector_feasibility_v1_failed_census",
        "path": str(
            SCORER_FIT_RELATIVE_PATH / "state_selector_feasibility_receipt.json"),
        "schema": "go2_scorer_fit_state_selector_feasibility_receipt_v1",
        "semantic_status": "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": "state_selector_feasibility_receipt_digest",
        "self_digest": (
            "2310c3d1b138b605fda483b39cbd4775479cbcc502a4e3707e7a8670457f54d7"),
        "raw_sha256": (
            "28e852792b5de24b2d008c5bb3f95521da668927e555deb9eb3c508bb6b0e59f"),
        "byte_count": 1_194_515,
    },
    {
        "role": "preserved_state_precontract_revalidation_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "preserved_state_precontract_revalidation_reachability_v2.json"),
        "schema": (
            "go2_scorer_fit_preserved_state_precontract_revalidation_"
            "reachability_v2"),
        "semantic_status": "FAIL_PRECONTRACT_IDENTITY_REVALIDATION",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": (
            "preserved_state_precontract_revalidation_receipt_digest"),
        "self_digest": (
            "0316e7f9b8462670eabe76da5fefc003274b4d08355373d14e1100cd6165e8e3"),
        "raw_sha256": (
            "2e49ed6e47caa98ed30ce45bddccd07cd3fad4c1950e5c6ff3fe051d0307de25"),
        "byte_count": 334_260,
    },
)

IMMUTABLE_V1_V2_FAILURE_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "role": "v1_immutable_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_prefix_benchmark_v1.json"),
        "schema": "go2_parallel_small_completion_search_v1_benchmark_receipt",
        "self_digest_key": "benchmark_receipt_digest",
        "self_digest": GLOBAL_AUTHORITY.V1_FAILURE_RECEIPT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V1_FAILURE_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V1_FAILURE_BYTE_COUNT,
    },
    {
        "role": "v2_contract",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_prefix_benchmark_v2_contract.json"),
        "schema": "go2_parallel_small_completion_benchmark_v2_contract",
        "self_digest_key": "benchmark_v2_contract_digest",
        "self_digest": GLOBAL_AUTHORITY.V2_CONTRACT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V2_CONTRACT_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V2_CONTRACT_BYTE_COUNT,
    },
    {
        "role": "v2_benchmark_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_prefix_benchmark_v2.json"),
        "schema": "go2_parallel_small_completion_search_v2_benchmark_receipt",
        "self_digest_key": "benchmark_receipt_digest",
        "self_digest": GLOBAL_AUTHORITY.V2_BENCHMARK_RECEIPT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V2_BENCHMARK_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V2_BENCHMARK_BYTE_COUNT,
    },
    {
        "role": "v2_terminal_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_terminal_failure_v2.json"),
        "schema": "go2_parallel_small_completion_benchmark_v2_terminal_failure",
        "self_digest_key": "terminal_failure_receipt_digest",
        "self_digest": GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_RECEIPT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_BYTE_COUNT,
    },
)

SOURCE_SPECS = (
    ("lewm/oracle/go2_scorer_fit_corpus_v2_design.py", "v2_design_authority"),
    ("lewm/oracle/go2_small_completion_global_execution_amendment_v1.py",
     "immutable_exact_infeasibility_lineage_authority"),
    ("lewm/oracle/go2_candidate_allocation_v1_2.py",
     "superseded_partial_subset_allocator_lineage"),
    ("lewm/oracle/go2_scorer_state_selector_amendment_v2.py",
     "active_outcome_free_state_selector"),
    ("lewm/oracle/go2_branch_oracle_v1_2.py", "frozen_oracle_v1_2"),
    ("scripts/dev_action_slew_reconstruction_v1.py",
     "frozen_full_bank_completion_reachability_slew"),
    ("lewm/oracle/go2_scorer_contract_v1_2.py", "frozen_scorer_science"),
    ("lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
     "full_bank_v2_successor_scorer_contract"),
    ("scripts/build_go2_branch_corpus_v1_2.py", "v2_identity_and_branch_builder"),
    ("scripts/encode_go2_branch_corpus_v1_2.py", "frozen_target_encoder_route"),
    ("scripts/train_go2_utility_scorer_v1_2.py", "frozen_training_route"),
    ("scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
     "frozen_development_transfer_route"),
    ("scripts/run_go2_scorer_fit_full_bank_v2.py", "v2_execution_runner"),
)
EXPECTED_SOURCE_PATHS = tuple(path for path, _role in SOURCE_SPECS)

# This post-smoke closure is deliberately separate from ``SOURCE_SPECS``.
# The latter is embedded in the already-issued design/correction lineage and
# must remain byte-for-byte interpretable with its historical 13-row shape.
ENCODER_IMPORT_CORRECTION_DEV_ENCODER_SOURCE_SPEC = (
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "frozen_target_encoder_import_route",
)
ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_SPECS = tuple(
    (path, "focused_encoder_import_compatibility_test")
    for path in ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_PATHS)
ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_SPECS = tuple(
    (path, "focused_encoder_compute_dtype_correction_test")
    for path in ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_PATHS)
ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_SPECS = tuple(
    (path, "focused_encoder_path_projection_correction_test")
    for path in ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_PATHS)

V2_PREOUTCOME_ARTIFACT_PATHS = (
    SCORER_FIT_RELATIVE_PATH / "state_manifest_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_assignment_manifest_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_preoutcome_state_revalidation_v2.json",
    SCORER_FIT_RELATIVE_PATH / "state_shard_small_enclosed_maze_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_small_completion_selection_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_preoutcome_feasibility_failure_v2.json",
)
V2_SUCCESSOR_CONTRACT_PATH = (
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_fit_corpus_v2_scorer_contract.json")
V2_ALWAYS_ABSENT_PATHS = (
    BRANCH_GENERATED_ROOT_RELATIVE_PATH / "final_eval/state_manifest.json",
)
V2_RUNTIME_OUTPUT_PATHS = (
    SCORER_FIT_RELATIVE_PATH / "branch_rows_v2.jsonl",
    SCORER_FIT_RELATIVE_PATH / "corpus_receipt_v2.json",
    SCORER_FIT_RELATIVE_PATH / "smoke_branch_receipt_v2.json",
    SCORER_FIT_RELATIVE_PATH / "latents_index_v2.json",
    SCORER_FIT_RELATIVE_PATH / "smoke_encoding_receipt_v2.json",
    SCORER_FIT_RELATIVE_PATH / "encoding_invocation_summary_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "label_distributions_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "completion_degeneracy_failure_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "no_latent_baseline_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "no_latent_baseline_v2.receipt.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_package_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_package_receipt_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "failed_scorer_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "qualification_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/development_transfer_spec_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/result_v2.json",
)
# Compatibility aggregate for callers that need the complete pre-design
# absence surface.  Phase-aware code should use
# :func:`audit_v2_runtime_outputs_absent`.
V2_FUTURE_OUTPUT_PATHS = (
    *V2_PREOUTCOME_ARTIFACT_PATHS,
    *V2_ALWAYS_ABSENT_PATHS,
    V2_SUCCESSOR_CONTRACT_PATH,
    *V2_RUNTIME_OUTPUT_PATHS,
)
V2_FUTURE_OUTPUT_DIRECTORIES = (
    SCORER_FIT_RELATIVE_PATH / "row_records_v2",
    SCORER_FIT_RELATIVE_PATH / "frames_v2",
    SCORER_FIT_RELATIVE_PATH / "latents_v2/context",
    SCORER_FIT_RELATIVE_PATH / "latents_v2/horizon",
    SCORER_FIT_RELATIVE_PATH / "invalid_attempts_v2",
    SCORER_FIT_RELATIVE_PATH / "superseded_receipts_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH / "initialisations_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH / "training_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/score_shards",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/invalid_attempts",
)
V2_FUTURE_OUTPUT_GLOBS = (
    SCORER_FIT_RELATIVE_PATH / "branch_summary_v2_*.json",
)

ENCODER_IMPORT_CORRECTION_REQUIRED_ABSENT_PATHS = (
    *V2_ALWAYS_ABSENT_PATHS,
    SCORER_FIT_RELATIVE_PATH / "latents_index_v2.json",
    SCORER_FIT_RELATIVE_PATH / "smoke_encoding_receipt_v2.json",
    SCORER_FIT_RELATIVE_PATH / "encoding_invocation_summary_v2.json",
    *tuple(path for path in V2_RUNTIME_OUTPUT_PATHS
           if UTILITY_V2_ROOT_RELATIVE_PATH == path
           or UTILITY_V2_ROOT_RELATIVE_PATH in path.parents),
)
ENCODER_IMPORT_CORRECTION_REQUIRED_ABSENT_DIRECTORIES = (
    SCORER_FIT_RELATIVE_PATH / "latents_v2/context",
    SCORER_FIT_RELATIVE_PATH / "latents_v2/horizon",
    UTILITY_V2_ROOT_RELATIVE_PATH / "initialisations_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH / "training_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/score_shards",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/invalid_attempts",
)
ENCODER_COMPUTE_DTYPE_CORRECTION_REQUIRED_ABSENT_PATHS = (
    *ENCODER_IMPORT_CORRECTION_REQUIRED_ABSENT_PATHS,
)
ENCODER_COMPUTE_DTYPE_CORRECTION_REQUIRED_ABSENT_DIRECTORIES = (
    SCORER_FIT_RELATIVE_PATH / "latents_v2",
    *ENCODER_IMPORT_CORRECTION_REQUIRED_ABSENT_DIRECTORIES,
)
ENCODER_PATH_PROJECTION_CORRECTION_REQUIRED_ABSENT_PATHS = (
    *V2_ALWAYS_ABSENT_PATHS,
    *tuple(path for path in V2_RUNTIME_OUTPUT_PATHS
           if UTILITY_V2_ROOT_RELATIVE_PATH == path
           or UTILITY_V2_ROOT_RELATIVE_PATH in path.parents),
)
ENCODER_PATH_PROJECTION_CORRECTION_REQUIRED_ABSENT_DIRECTORIES = (
    UTILITY_V2_ROOT_RELATIVE_PATH / "initialisations_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH / "training_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/score_shards",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/invalid_attempts",
)
ENCODER_PATH_PROJECTION_TRANSACTION_REQUIRED_ABSENT_PATHS = (
    FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH,
    FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STAGED_RELATIVE_PATH,
    FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH,
    FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STAGED_RELATIVE_PATH,
    FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_RELATIVE_PATH,
)
ENCODER_PATH_PROJECTION_TRANSACTION_REQUIRED_ABSENT_DIRECTORIES = (
    FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH,
)
BRANCH_REDRIVE_PROJECTION_CORRECTION_REQUIRED_ABSENT_PATHS = (
    *V2_ALWAYS_ABSENT_PATHS,
    UTILITY_V2_ROOT_RELATIVE_PATH / "label_distributions_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "completion_degeneracy_failure_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "no_latent_baseline_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "no_latent_baseline_v2.receipt.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_package_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_package_receipt_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "failed_scorer_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "qualification_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/development_transfer_spec_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/result_v2.json",
)
BRANCH_REDRIVE_PROJECTION_CORRECTION_REQUIRED_ABSENT_DIRECTORIES = (
    UTILITY_V2_ROOT_RELATIVE_PATH / "initialisations_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH / "training_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/score_shards",
    UTILITY_V2_ROOT_RELATIVE_PATH /
        "counterfactual_development_transfer_v2/invalid_attempts",
)

_ROTATION_IDS = tuple(
    row["constraint_id"] for row in GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY
    if row["rotation_subset_identity_reference"]
)
EXPECTED_ROTATION_CONSTRAINT_IDS = (
    "ROTATION_BLOCK_CATALOGUE_12_OF_6",
    "ROTATION_BLOCK_FORWARD_TURNING_REVERSE_SEMANTICS",
    "EXACTLY_ONE_ROTATION_PER_STATE",
    "FIT_FAMILY_STRATUM_CANDIDATE_EXACT_2",
    "CALIBRATION_STRATUM_CANDIDATE_EXACT_4",
    "CALIBRATION_FAMILY_CANDIDATE_1_TO_2",
    "GOAL_TYPE_CANDIDATE_FLOOR_CEILING",
    "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
    "ALL_40_COMPLETION_MASKS_PASS",
    "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
    "FIRST_PASSING_SCENE_COMBINATION",
    "GLOBAL_CANDIDATE_EXACT_60",
    "FIT_CALIBRATION_CANDIDATE_EXACT_48_12",
    "STRATUM_CANDIDATE_EXACT_20",
    "FAMILY_CANDIDATE_BALANCE_7_8",
    "FAMILY_STRATUM_CANDIDATE_BALANCE_2_3",
    "REVERSE_CANDIDATE_IN_60_DISTINCT_STATES",
    "ASSIGNMENT_BLOCK_CONTINGENCY_AND_PAIRWISE_INTEGRITY",
)
COUPLING_INVENTORY_DIGEST = (
    "6a195d65d826a883057aa6ab5c535c48aefdc6457dc7ee427b41b12a4e1f7dce")
ROTATION_CONSTRAINT_INVENTORY_DIGEST = (
    "81cc9ff06fdca5359c9dab060625599298257026246c75c312761b16f2dddad3")

_ALLOCATION_BALANCE_IDS = frozenset({
    "ROTATION_BLOCK_CATALOGUE_12_OF_6",
    "ROTATION_BLOCK_FORWARD_TURNING_REVERSE_SEMANTICS",
    "EXACTLY_ONE_ROTATION_PER_STATE",
    "FIT_FAMILY_STRATUM_CANDIDATE_EXACT_2",
    "CALIBRATION_STRATUM_CANDIDATE_EXACT_4",
    "CALIBRATION_FAMILY_CANDIDATE_1_TO_2",
    "GOAL_TYPE_CANDIDATE_FLOOR_CEILING",
    "GLOBAL_CANDIDATE_EXACT_60",
    "FIT_CALIBRATION_CANDIDATE_EXACT_48_12",
    "STRATUM_CANDIDATE_EXACT_20",
    "FAMILY_CANDIDATE_BALANCE_7_8",
    "FAMILY_STRATUM_CANDIDATE_BALANCE_2_3",
    "REVERSE_CANDIDATE_IN_60_DISTINCT_STATES",
    "ASSIGNMENT_BLOCK_CONTINGENCY_AND_PAIRWISE_INTEGRITY",
})
_SUBSET_REACHABILITY_IDS = frozenset({
    "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
    "ALL_40_COMPLETION_MASKS_PASS",
})
_EXECUTION_POLICY_IDS = frozenset({
    "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
    "FIRST_PASSING_SCENE_COMBINATION",
})

_HEX = frozenset("0123456789abcdef")

PREDECESSOR_VALIDATION_PROJECTION: dict[str, Any] = {
    "schema": "go2_scorer_fit_corpus_v2_predecessor_validation",
    "active_global_amendment_exact_raw_self_schema_source_validated": True,
    "global_exact_model_plan_exact_raw_self_schema_source_validated": True,
    "terminal_infeasibility_exact_raw_self_schema_source_validated": True,
    "terminal_nested_model_and_infeasibility_digests_validated": True,
    "immutable_v1_v2_failure_count_validated": 4,
    "prior_preoutcome_failure_count_validated": len(
        PRIOR_PREOUTCOME_FAILURE_BINDINGS),
    "frozen_predictor_qualification_report_raw_binding_validated": True,
    "historical_generated_artifacts_reopened_for_lineage_custody": True,
    "historical_generated_artifacts_used_to_classify_rotation_conditions": False,
    "historical_generated_artifacts_used_to_select_states": False,
    "solve_or_solver_invocation": False,
    "candidate_outcome_or_branch_label_read": False,
    "frame_latent_scorer_metric_or_predictor_score_read": False,
}

ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE = {
    "scorer_fit_corpus_v2_design_digest": (
        "9640642c064a9f4b161addbc5feaa96529551ea2e794f077d2ca3a162f00062e"),
    "preselection_source_correction_digest":
        IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST,
    "manifest_replay_correction_digest":
        IMMUTABLE_MANIFEST_REPLAY_CORRECTION_DIGEST,
    "successor_scorer_contract_digest":
        IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING[
            "embedded_contract_self_digest"],
    "successor_scorer_contract_artifact_digest":
        IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING["self_digest"],
    "state_manifest_digest": (
        "db79efce49d949522832d920b23a38292a491dc9e6fb2cbf2b8e0a5176fb062e"),
    "assignment_manifest_digest": (
        "a91d6d211f5b07270df5a66262ce4ba218e8a3925ae5f8aba196b8c10f4959f4"),
    "scientific_scorer_contract_v1_2_digest": (
        "f268763ed9365205cd0b0001c4527afbf5e5d948846dbb891225a48acb74113a"),
    "target_encoder_digest": (
        "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"),
    "target_encoder_checkpoint_sha256": (
        "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"),
    "preprocess_contract_digest": (
        "2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9"),
    "preprocessing_identity_sha256": (
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"),
    "oracle_v1_2_digest": ORACLE_V1_2_DIGEST,
    "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
}

ENCODER_IMPORT_FAILURE_BOUNDARY = {
    "status": "IMMUTABLE_INFRASTRUCTURE_FAILURE_MISSING_TIMM",
    "historical_source_repository_commit":
        ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
    "runner_stage": "smoke_encoding",
    "encoder_command": [
        ".generated/venvs/world_model_rocm_7_2_1_v1/bin/python",
        "scripts/encode_go2_branch_corpus_v1_2.py",
        "--pool", "scorer_fit", "--corpus-design", "full-bank-v2", "--smoke",
    ],
    "exception_type": "ModuleNotFoundError",
    "exception_message": "No module named 'timm'",
    "failure_import_chain": [
        "src.hub.backbones",
        "app.vjepa_2_1.models.predictor",
        "app.vjepa_2_1.models.utils.modules",
        "timm.models.layers.drop_path",
    ],
    "branch_smoke_completed": True,
    "branch_record_count": 12,
    "rendered_horizon_frame_count": 48,
    "candidate_indices": list(CANDIDATE_INDICES),
    "branch_outcomes_exist": True,
    "branch_outcome_or_label_value_consumed_for_correction": False,
    "branch_frames_opened_by_correction": False,
    "encoder_smoke_entered": True,
    "target_encoder_identity_sha256_streamed": True,
    "checkpoint_file_read_only_for_sha256_identity_verification": True,
    "checkpoint_torch_load_or_tensor_deserialization_started": False,
    "encoder_or_predictor_model_constructed": False,
    "encoder_or_predictor_weights_loaded": False,
    "latent_shard_written": False,
    "latent_index_written": False,
    "smoke_encoding_receipt_written": False,
    "scorer_training_started": False,
    "development_transfer_started": False,
    "predictor_checkpoint_opened": False,
    "final_200_state_corpus_generated": False,
    "scientific_qualification_verdict_reached": False,
}

ENCODER_COMPUTE_DTYPE_CORRECTION_PRESERVED_SCIENCE = {
    **ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE,
    "encoder_import_correction_digest":
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST,
    "target_encoder_compute_dtype": "float32",
    "target_latent_storage_dtype": "float16",
    "stage_a_fp32_source_sha256":
        ENCODER_COMPUTE_DTYPE_STAGE_A_FP32_SOURCE_BINDING["sha256"],
}

ENCODER_COMPUTE_DTYPE_FAILURE_BOUNDARY = {
    "status": "IMMUTABLE_INFRASTRUCTURE_FAILURE_ROCM_SDPA_QKV_DTYPE_MISMATCH",
    "historical_source_repository_commit":
        ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
    "runner_stage": "smoke_encoding",
    "encoder_command": [
        ".generated/venvs/world_model_rocm_7_2_1_v1/bin/python",
        "scripts/encode_go2_branch_corpus_v1_2.py",
        "--pool", "scorer_fit", "--corpus-design", "full-bank-v2", "--smoke",
    ],
    "exception_type": "RuntimeError",
    "exception_message": (
        "Expected query, key, and value to have the same dtype, but got "
        "query.dtype: float key.dtype: float and value.dtype: c10::BFloat16 "
        "instead."),
    "failure_operation": "torch.nn.functional.scaled_dot_product_attention",
    "failure_source_binding": copy.deepcopy(
        ENCODER_COMPUTE_DTYPE_UPSTREAM_ROPE_SOURCE_BINDING),
    "branch_smoke_completed": True,
    "branch_smoke_zero_new_resume_completed": True,
    "branch_record_count": 12,
    "rendered_horizon_frame_count": 48,
    "candidate_indices": list(CANDIDATE_INDICES),
    "branch_outcomes_exist": True,
    "branch_outcome_or_label_value_consumed_for_correction": False,
    "registered_smoke_frames_verified_by_failed_encoder": True,
    "branch_frame_value_opened_by_correction_issuer": False,
    "missing_context_shard_count_at_failure": 1,
    "missing_horizon_shard_count_at_failure": 12,
    "device": "cuda:0",
    "failed_selected_encoder_compute_dtype": "bfloat16",
    "target_encoder_constructor_completed": True,
    "checkpoint_torch_load_map_location_cpu_completed": True,
    "strict_encoder_state_dict_load_completed": True,
    "encoder_to_cuda_bfloat16_completed": True,
    "encoder_eval_and_requires_grad_false_completed": True,
    "first_context_batch_preprocessing_completed": True,
    "first_encoder_forward_entered": True,
    "rope_query_dtype_at_failure": "float32",
    "rope_key_dtype_at_failure": "float32",
    "attention_value_dtype_at_failure": "bfloat16",
    "encode_paths_returned": False,
    "atomic_f16_reached": False,
    "context_latent_shard_written": False,
    "horizon_latent_shard_written": False,
    "latent_index_written": False,
    "encoding_invocation_summary_written": False,
    "smoke_encoding_receipt_written": False,
    "scorer_training_started": False,
    "development_transfer_started": False,
    "predictor_checkpoint_opened": False,
    "final_200_state_corpus_generated": False,
    "scientific_qualification_verdict_reached": False,
}

ENCODER_PATH_PROJECTION_CORRECTION_PRESERVED_SCIENCE = {
    **ENCODER_COMPUTE_DTYPE_CORRECTION_PRESERVED_SCIENCE,
    "encoder_compute_dtype_correction_digest":
        IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST,
    "base_smoke_context_latent_count": 1,
    "base_smoke_horizon_latent_count": 12,
    "base_smoke_total_latent_shard_count": 13,
    "base_smoke_receipt_digest":
        IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_SMOKE_BINDING["self_digest"],
    "single_registered_shard_regeneration_smoke_requirement_unchanged": True,
    "transaction_control_uses_candidate_identity_not_outcomes": True,
    "transaction_control_deserializes_no_latent_values": True,
}

ENCODER_PATH_PROJECTION_PREISSUE_SINGLE_SHARD_TRANSACTION_AUDIT = {
    "status": "PREISSUE_SOURCE_ONLY_INTERRUPTION_SAFETY_AUDIT",
    "finding_severity": "P1",
    "finding": (
        "DIRECT_UNLINK_AND_ARCHIVE_FIRST_PASS_SMOKE_PUBLICATION_COULD_"
        "AUTHORISE_A_SECOND_DELIBERATE_DELETION_AFTER_INTERRUPTION"),
    "historical_source_repository_commit":
        ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
    "historical_runner_function": "_delete_registered_smoke_shard",
    "historical_runner_operation": "PATH_UNLINK_THEN_PARENT_DIRECTORY_FSYNC",
    "historical_encoder_source_binding": copy.deepcopy(
        ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING),
    "historical_encoder_function": (
        "main_full_bank_v2_smoke_receipt_publication_blocks"),
    "historical_encoder_publication_source_lines": [2043, 2131],
    "historical_encoder_publication_operation": (
        "OS_REPLACE_ACTIVE_SMOKE_TO_ARCHIVE_THEN_ATOMIC_JSON_SUCCESSOR"),
    "historical_encoder_crash_window": (
        "AFTER_SUCCESSFUL_CANDIDATE_0_REGENERATION_AND_ACTIVE_SMOKE_ARCHIVE_"
        "BEFORE_SUCCESSOR_SMOKE_INSTALL"),
    "active_completion_proof_absent_in_historical_crash_window": True,
    "resume_could_authorise_second_deliberate_target_deletion": True,
    "second_deliberate_target_deletion_observed": False,
    "discovered_after_runtime_failure_by_source_audit": True,
    "observed_as_a_runtime_failure": False,
    "path_projection_correction_artifact_issued_when_discovered": False,
    "zero_new_resume_started_when_discovered": False,
    "single_shard_transaction_started_when_discovered": False,
    "prepared_receipt_existed_when_discovered": False,
    "prepared_staged_receipt_existed_when_discovered": False,
    "backup_existed_when_discovered": False,
    "complete_receipt_existed_when_discovered": False,
    "complete_staged_receipt_existed_when_discovered": False,
    "branch_outcome_or_label_value_read_for_audit": False,
    "latent_value_deserialized_for_audit": False,
    "scientific_requirement_changed": False,
    "required_hardening": (
        "IMMUTABLE_PREPARED_THEN_ATOMIC_MOVE_TO_RETAINED_EXACT_BACKUP_"
        "THEN_IMMUTABLE_COMPLETE_BEFORE_PASS_SMOKE"),
}

ENCODER_PATH_PROJECTION_FAILURE_BOUNDARY = {
    "status": "IMMUTABLE_INFRASTRUCTURE_FAILURE_LOGICAL_PATH_PROJECTION",
    "historical_source_repository_commit":
        ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
    "runner_stage": "post_base_smoke_read_only_validation",
    "failing_function": (
        "load_and_validate_full_bank_v2_encoding_smoke_for_consumption"),
    "failing_projection_name": "registered_smoke_shard_inventory",
    "failing_expression": "_resolve_frame(...).relative_to(ROOT)",
    "failing_source_line_at_historical_commit": 1271,
    "exception_type": "ValueError",
    "exception_message_prefix": (
        "'/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2/"
        "active/go2_branch_corpus_v1_2/scorer_fit/latents_v2/context/"
        "c9bf42df529b75ebaf7e9053be059ee7ad639e690f501ba6b8750c968a37634e."
        "f16' is not in the subpath of '/home/andrewknowles/Workspace/"
        "LeWMQuad-v3'"),
    "exception_message_suffix_claimed": False,
    "exception_physical_path_prefix": (
        "/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2/"
        "active/go2_branch_corpus_v1_2/scorer_fit/latents_v2/context/"),
    "exception_repository_root": (
        "/home/andrewknowles/Workspace/LeWMQuad-v3"),
    "managed_generated_root_is_logical_symlink": True,
    "resolved_shard_is_outside_repository_root_lexically": True,
    "base_smoke_encoder_execution_completed": True,
    "base_smoke_end_to_end_pass": True,
    "base_smoke_protocol_complete": False,
    "valid_context_latent_shard_count": 1,
    "valid_horizon_latent_shard_count": 12,
    "valid_total_latent_shard_count": 13,
    "latent_index_written": True,
    "encoding_invocation_summary_written": True,
    "base_smoke_receipt_written": True,
    "read_only_validator_entered": True,
    "latent_shard_bytes_hashed_by_validator": True,
    "latent_values_deserialized_by_validator": False,
    "branch_rows_and_oracle_records_loaded_by_validator": True,
    "branch_outcome_or_label_value_used_for_correction": False,
    "first_logical_path_inventory_construction_entered": True,
    "validator_projection_returned": False,
    "validator_write_attempted": False,
    "base_smoke_artifact_changed_by_validator": False,
    "zero_new_resume_started": False,
    "single_shard_deletion_started": False,
    "single_shard_regeneration_started": False,
    "single_shard_transaction_prepared_receipt_written": False,
    "single_shard_transaction_prepared_staged_receipt_written": False,
    "single_shard_transaction_backup_created": False,
    "single_shard_transaction_complete_receipt_written": False,
    "single_shard_transaction_complete_staged_receipt_written": False,
    "pass_smoke_published": False,
    "full_corpus_branch_generation_started": False,
    "full_corpus_latent_encoding_started": False,
    "scorer_training_started": False,
    "development_transfer_started": False,
    "predictor_checkpoint_opened": False,
    "final_200_state_corpus_generated": False,
    "scientific_qualification_verdict_reached": False,
}

BRANCH_REDRIVE_PROJECTION_FAILURE_BOUNDARY = {
    "status": "IMMUTABLE_SOURCE_FAILURE_BRANCH_REDRIVE_PROJECTION",
    "historical_source_repository_commit":
        BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
    "runner_stage": "full_branch_corpus",
    "failed_state_id": BRANCH_REDRIVE_FAILURE_STATE_ID,
    "failed_state_index": 10,
    "failed_state_identity_digest":
        BRANCH_REDRIVE_FAILURE_STATE_IDENTITY_DIGEST,
    "completed_preceding_state_count": 10,
    "completed_preceding_valid_branch_count": 120,
    "failing_function": "_redrive_mismatch",
    "failing_evidence_function": "full_bank_completion_reachability_evidence",
    "active_manifest_field_copied_into_structural_evidence":
        "candidate_indices",
    "structural_evidence_forbidden_field": "candidate_indices",
    "deterministic_exception_type": "RuntimeError",
    "broad_exception_handler_overwrote_prior_comparison_truth_values": True,
    "reported_invalid_reason": BRANCH_REDRIVE_FAILURE_REASON,
    "reported_reason_proves_previous_command_mismatch": False,
    "reported_reason_proves_snapshot_task_status_mismatch": False,
    "reported_reason_proves_full_bank_l_max_ineligibility": False,
    "base_structural_comparisons_not_named_in_reason_passed": True,
    "candidate_branch_execution_started_for_failed_state": False,
    "candidate_outcome_or_label_produced_for_failed_state": False,
    "invalid_attempt_receipts_are_metadata_only_placeholders": True,
    "scientific_or_feasibility_failure_established": False,
}

BRANCH_REDRIVE_PROJECTION_CORRECTION_PRESERVED_SCIENCE = {
    "immutable_encoder_path_projection_correction_digest":
        IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST,
    "state_manifest_digest": (
        "db79efce49d949522832d920b23a38292a491dc9e6fb2cbf2b8e0a5176fb062e"),
    "assignment_manifest_digest": (
        "a91d6d211f5b07270df5a66262ce4ba218e8a3925ae5f8aba196b8c10f4959f4"),
    "successor_scorer_contract_digest": (
        "8fc0edae875cba6487ff1a1a771f96b0157da1474ac00de4186ecdb41b66d5df"),
    "successor_scorer_contract_artifact_digest": (
        "4455fd397ce7665f02725924a64ab87b1e0e9a3506d9ba64edbcc9b4daa1e121"),
    "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
    "candidate_indices": list(CANDIDATE_INDICES),
    "oracle_v1_2_digest": ORACLE_V1_2_DIGEST,
    "state_identity_or_manifest_replacement_authorised": False,
    "candidate_bank_or_oracle_change_authorised": False,
    "branch_plan_render_preprocess_encoder_or_scorer_change_authorised": False,
    "completed_valid_branch_regeneration_authorised": False,
    "resume_scope": "MISSING_REGISTERED_ASSIGNMENTS_ONLY",
    "unchanged_source_retry_authorised": False,
    "invalid_attempt_receipts_preserved": True,
    "partial_valid_branch_rows_preserved": True,
    "candidate_outcome_or_label_value_read_for_correction": False,
    "candidate_outcome_or_label_value_used_for_selection": False,
    "frame_or_latent_value_read_for_correction": False,
    "preoutcome_feasibility_terminal_authorised": False,
    "scientific_failure_terminal_authorised_by_observed_reason": False,
}


class ScorerFitCorpusV2DesignError(RuntimeError):
    """The prospective design, classification, or custody binding changed."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScorerFitCorpusV2DesignError(
            "value is not compact canonical JSON") from exc


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True,
            allow_nan=False,
        ) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScorerFitCorpusV2DesignError(
            "value is not canonical JSON") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def builder_default_canonical_digest(value: Any) -> str:
    """Match the five installed builder payloads' historical self digest."""

    try:
        raw = json.dumps(
            value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScorerFitCorpusV2DesignError(
            "value is not builder-default canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _without(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _is_hex(value: Any, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(character in _HEX for character in value))


def _validate_source_bindings(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(SOURCE_SPECS):
        raise ScorerFitCorpusV2DesignError("source binding coverage changed")
    result: list[dict[str, Any]] = []
    for (path, role), raw in zip(SOURCE_SPECS, rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ScorerFitCorpusV2DesignError("source binding is malformed")
        row = dict(raw)
        if (set(row) != {"path", "role", "byte_count", "sha256"}
                or row.get("path") != path or row.get("role") != role
                or isinstance(row.get("byte_count"), bool)
                or not isinstance(row.get("byte_count"), int)
                or row["byte_count"] <= 0
                or not _is_hex(row.get("sha256"), 64)):
            raise ScorerFitCorpusV2DesignError("source binding changed")
        result.append(row)
    return result


def _changed_source_paths(
        historical: Sequence[Mapping[str, Any]],
        corrected: Sequence[Mapping[str, Any]],
        ) -> list[str]:
    before = {str(row["path"]): dict(row) for row in historical}
    after = {str(row["path"]): dict(row) for row in corrected}
    if set(before) != set(after):
        raise ScorerFitCorpusV2DesignError(
            "source-correction closure path set changed")
    return sorted(
        path for path in before
        if (before[path]["sha256"], before[path]["byte_count"])
        != (after[path]["sha256"], after[path]["byte_count"]))


def _validate_transition_endpoint(
        value: Any, *, path: str, role: str, current: bool,
        ) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScorerFitCorpusV2DesignError("source transition endpoint is malformed")
    row = dict(value)
    if set(row) != {"path", "role", "exists", "byte_count", "sha256"}:
        raise ScorerFitCorpusV2DesignError("source transition endpoint is not closed")
    if row.get("path") != path or row.get("role") != role:
        raise ScorerFitCorpusV2DesignError("source transition identity changed")
    exists = row.get("exists")
    if not isinstance(exists, bool) or (current and not exists):
        raise ScorerFitCorpusV2DesignError("current correction source is absent")
    if exists:
        if (isinstance(row.get("byte_count"), bool)
                or not isinstance(row.get("byte_count"), int)
                or row["byte_count"] <= 0
                or not _is_hex(row.get("sha256"), 64)):
            raise ScorerFitCorpusV2DesignError("source transition binding changed")
    elif row.get("byte_count") != 0 or row.get("sha256") is not None:
        raise ScorerFitCorpusV2DesignError("absent source transition is malformed")
    return row


def _validate_source_transition(
        value: Any, *, path: str, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
            "path", "role", "historical", "current"}:
        raise ScorerFitCorpusV2DesignError("source transition is not closed")
    transition = copy.deepcopy(dict(value))
    if transition.get("path") != path or transition.get("role") != role:
        raise ScorerFitCorpusV2DesignError("source transition path or role changed")
    historical = _validate_transition_endpoint(
        transition["historical"], path=path, role=role, current=False)
    current = _validate_transition_endpoint(
        transition["current"], path=path, role=role, current=True)
    if historical == current:
        raise ScorerFitCorpusV2DesignError("registered source transition did not change")
    transition["historical"] = historical
    transition["current"] = current
    return transition


def _validate_dev_encoder_source_transition(value: Any) -> dict[str, Any]:
    path, role = ENCODER_IMPORT_CORRECTION_DEV_ENCODER_SOURCE_SPEC
    transition = _validate_source_transition(value, path=path, role=role)
    if transition["historical"] != (
            ENCODER_IMPORT_CORRECTION_DEV_ENCODER_HISTORICAL_BINDING):
        raise ScorerFitCorpusV2DesignError(
            "historical frozen target-encoder source binding changed")
    return transition


def _validate_focused_test_source_transitions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(
            ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_SPECS):
        raise ScorerFitCorpusV2DesignError("focused test transition coverage changed")
    transitions = [
        _validate_source_transition(row, path=path, role=role)
        for (path, role), row in zip(
            ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_SPECS, value, strict=True)
    ]
    if [row["path"] for row in transitions] != list(
            ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_PATHS):
        raise ScorerFitCorpusV2DesignError("focused test transition order changed")
    return transitions


def _expected_absence_rows(*, phase: str) -> list[dict[str, Any]]:
    if phase not in {"design", "successor_contract", "post_contract_pre_branch"}:
        raise ScorerFitCorpusV2DesignError("unknown V2 absence-audit phase")
    exact_paths = list(V2_ALWAYS_ABSENT_PATHS)
    exact_paths.extend(V2_RUNTIME_OUTPUT_PATHS)
    if phase == "design":
        exact_paths.extend(V2_PREOUTCOME_ARTIFACT_PATHS)
    if phase in {"design", "successor_contract"}:
        exact_paths.append(V2_SUCCESSOR_CONTRACT_PATH)
    exact_paths.extend(Path(path) for path in
                       GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_EXACT_FILE_PATHS)
    directory_paths = list(V2_FUTURE_OUTPUT_DIRECTORIES)
    directory_paths.extend(Path(path) for path in
                           GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_DIRECTORY_ROOTS)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, paths in (("file", exact_paths), ("directory", directory_paths)):
        for relative in paths:
            key = str(relative)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"path": key, "expected_kind": kind, "exists": False})
    patterns = list(V2_FUTURE_OUTPUT_GLOBS)
    patterns.extend(Path(path) for path in
                    GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_GLOB_PATTERNS)
    for pattern in patterns:
        rows.append({
            "path": str(pattern), "expected_kind": "glob", "exists": False,
        })
    return rows


def _validate_absence_projection(value: Any, *, phase: str) -> list[dict[str, Any]]:
    expected = _expected_absence_rows(phase=phase)
    if not isinstance(value, list) or value != expected:
        raise ScorerFitCorpusV2DesignError(
            "source-correction output-absence projection changed")
    return copy.deepcopy(expected)


def _predecessor_lineage() -> dict[str, Any]:
    return {
        "schema": "go2_scorer_fit_corpus_v2_exact_infeasibility_lineage",
        "terminal_source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
        "active_global_amendment": copy.deepcopy(ACTIVE_GLOBAL_AMENDMENT_BINDING),
        "global_model_plan": copy.deepcopy(GLOBAL_MODEL_PLAN_BINDING),
        "terminal_infeasibility": copy.deepcopy(
            GLOBAL_TERMINAL_INFEASIBILITY_BINDING),
        "active_global_amendment_digest": ACTIVE_GLOBAL_AMENDMENT_DIGEST,
        "global_exact_model_digest": GLOBAL_EXACT_MODEL_DIGEST,
        "exact_infeasibility_digest": EXACT_INFEASIBILITY_DIGEST,
        "terminal_receipt_digest": TERMINAL_RECEIPT_DIGEST,
        "model_execution_plan_digest": MODEL_EXECUTION_PLAN_DIGEST,
        "terminal_status": "INFEASIBLE",
        "six_of_twelve_design_closed_exactly_infeasible": True,
        "six_of_twelve_model_retried_or_reinterpreted": False,
        "candidate_outcomes_consumed_at_proof": False,
        "branch_labels_read_at_proof": False,
        "frames_or_latents_created_at_proof": False,
        "scorer_or_predictor_accessed_at_proof": False,
        "scientific_conditions_relaxed": False,
        "immutable_v1_failure": {
            "status": GLOBAL_AUTHORITY.V1_FAILURE_STATUS,
            "receipt_digest": GLOBAL_AUTHORITY.V1_FAILURE_RECEIPT_DIGEST,
        },
        "immutable_v2_failure": {
            "source_repository_commit": GLOBAL_AUTHORITY.V2_SOURCE_REPOSITORY_COMMIT,
            "contract_digest": GLOBAL_AUTHORITY.V2_CONTRACT_DIGEST,
            "benchmark_receipt_digest": (
                GLOBAL_AUTHORITY.V2_BENCHMARK_RECEIPT_DIGEST),
            "terminal_failure_receipt_digest": (
                GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_RECEIPT_DIGEST),
            "backend_disposition": GLOBAL_AUTHORITY.V2_BACKEND_DISPOSITION,
        },
        "immutable_v1_v2_failure_bindings": copy.deepcopy(
            list(IMMUTABLE_V1_V2_FAILURE_BINDINGS)),
        "prior_preoutcome_failure_bindings": copy.deepcopy(
            list(PRIOR_PREOUTCOME_FAILURE_BINDINGS)),
        "all_invalid_attempt_and_preoutcome_lineage_transitively_bound_by": (
            ACTIVE_GLOBAL_AMENDMENT_DIGEST),
        "frozen_predictor_qualification": copy.deepcopy(
            FROZEN_PREDICTOR_QUALIFICATION),
    }


def _validate_predecessor_lineage(value: Any) -> dict[str, Any]:
    expected = _predecessor_lineage()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ScorerFitCorpusV2DesignError(
            "immutable exact-infeasibility lineage changed")
    return expected


def _validate_predecessor_validation_projection(value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or dict(value) != PREDECESSOR_VALIDATION_PROJECTION):
        raise ScorerFitCorpusV2DesignError(
            "historical predecessor validation projection changed")
    return copy.deepcopy(PREDECESSOR_VALIDATION_PROJECTION)


def _classification_reason(constraint_id: str) -> tuple[str, str]:
    if constraint_id in _ALLOCATION_BALANCE_IDS:
        return (
            "RETIRED_PARTIAL_SUBSET_DECISION_OR_BALANCE",
            "The condition exists only because six candidates were selected "
            "from twelve; assigning all twelve makes it either inapplicable "
            "or an algebraic identity under the V2 count contract.",
        )
    if constraint_id in _SUBSET_REACHABILITY_IDS:
        return (
            "REPLACED_BY_FULL_BANK_L_MAX_STATE_REVALIDATION",
            "The old mask compares completion enrichment with L_max of one "
            "allocated six-candidate subset.  It does not establish branch "
            "executability.  V2 recomputes the unchanged selector predicate "
            "with L_max over candidate indices 0--11.",
        )
    if constraint_id in _EXECUTION_POLICY_IDS:
        return (
            "RETIRED_NONSCIENTIFIC_ALLOCATION_EXECUTION_POLICY",
            "The condition canonicalizes or searches scene/rotation choices; "
            "V2 has no rotation choice or scene/rotation optimization.",
        )
    raise ScorerFitCorpusV2DesignError(
        f"unclassified old rotation condition: {constraint_id}")


def build_rotation_mask_classification(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        predecessor_validation: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the closed source-only classification; perform no file access."""

    if not _is_hex(source_repository_commit, 40):
        raise ScorerFitCorpusV2DesignError("source commit is malformed")
    sources = _validate_source_bindings(list(source_bindings))
    predecessor_projection = _validate_predecessor_validation_projection(
        predecessor_validation)
    if (_ROTATION_IDS != EXPECTED_ROTATION_CONSTRAINT_IDS
            or canonical_digest(list(GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY))
            != COUPLING_INVENTORY_DIGEST
            or canonical_digest([
                row for row in GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY
                if row["rotation_subset_identity_reference"]
            ]) != ROTATION_CONSTRAINT_INVENTORY_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "immutable coupling constraint inventory changed")
    rows: list[dict[str, Any]] = []
    by_id = {row["constraint_id"]: row
             for row in GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY}
    for constraint_id in EXPECTED_ROTATION_CONSTRAINT_IDS:
        disposition, rationale = _classification_reason(constraint_id)
        old = by_id[constraint_id]
        rows.append({
            "constraint_id": constraint_id,
            "classification": "PARTIAL_SUBSET_ALLOCATION_ONLY",
            "v2_disposition": disposition,
            "origin": old["origin"],
            "source_symbols": list(old["source_symbols"]),
            "old_exact_rule": old["exact_rule"],
            "old_scientific_status": old["scientific_status"],
            "rationale": rationale,
        })
    payload: dict[str, Any] = {
        "schema": MASK_CLASSIFICATION_SCHEMA,
        "status": MASK_CLASSIFICATION_STATUS,
        "source_repository_commit": source_repository_commit,
        "source_bindings": sources,
        "source_binding_set_digest": canonical_digest(sources),
        "coupling_report_binding": copy.deepcopy(
            GLOBAL_AUTHORITY.ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING),
        "coupling_inventory_digest": COUPLING_INVENTORY_DIGEST,
        "rotation_constraint_inventory_digest": (
            ROTATION_CONSTRAINT_INVENTORY_DIGEST),
        "predecessor_validation": predecessor_projection,
        "classification_values": [
            "PARTIAL_SUBSET_ALLOCATION_ONLY",
            "TRUE_BRANCH_EXECUTION_REQUIREMENT",
        ],
        "conditions": rows,
        "counts": {
            "old_rotation_related_condition_count": len(rows),
            "partial_subset_allocation_only_count": len(rows),
            "true_branch_execution_requirement_count": 0,
        },
        "true_branch_execution_test": {
            "qualifying_reasons": [
                "candidate branch cannot execute from the state",
                "required planning-time input is unavailable",
                "canonical snapshot contract is violated",
                "required oracle or latent record cannot be produced",
            ],
            "matching_old_rotation_condition_ids": [],
            "all_twelve_candidates_use_the_same_executor": True,
            "candidate_indices_accepted_by_executor": list(CANDIDATE_INDICES),
        },
        "retained_non_rotation_completion_requirements": {
            "finite_graph_reachability": True,
            "absolute_body_bearing_deg_max": 75.0,
            "snapshot_goal_claimed": False,
            "snapshot_task_completed": False,
            "snapshot_terminated": False,
            "snapshot_truncated": False,
            "canonical_snapshot_boundary": True,
            "all_oracle_and_scorer_inputs_available": True,
            "completion_radius_m": 0.75,
            "horizon_blocks": 4,
            "horizon_ticks": 20,
            "full_bank_l_max_candidate_indices": list(CANDIDATE_INDICES),
            "branch_execution_used_for_revalidation": False,
        },
        "outcome_access": {
            "candidate_outcome_read": False,
            "branch_label_read": False,
            "frame_or_latent_read": False,
            "scorer_metric_or_predictor_score_read": False,
            "historical_receipts_reopened_for_exact_lineage_custody": True,
            "historical_receipts_used_for_classification": False,
        },
    }
    payload[MASK_CLASSIFICATION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_rotation_mask_classification(
        payload: Mapping[str, Any], *, validate_live_source: bool = False,
        root: Path = ROOT) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ScorerFitCorpusV2DesignError("mask classification is not a mapping")
    value = copy.deepcopy(dict(payload))
    if (value.get("schema") != MASK_CLASSIFICATION_SCHEMA
            or value.get("status") != MASK_CLASSIFICATION_STATUS
            or value.get(MASK_CLASSIFICATION_SELF_KEY)
            != canonical_digest(_without(value, MASK_CLASSIFICATION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError("mask classification binding changed")
    expected = build_rotation_mask_classification(
        source_repository_commit=str(value.get("source_repository_commit", "")),
        source_bindings=value.get("source_bindings", []),
        predecessor_validation=value.get("predecessor_validation", {}))
    if value != expected:
        raise ScorerFitCorpusV2DesignError("mask classification is not exact")
    if validate_live_source:
        commit, sources = clean_source_authority(root=root)
        if commit != value["source_repository_commit"] or sources != value["source_bindings"]:
            raise ScorerFitCorpusV2DesignError(
                "live source differs from mask classification")
    return value


def rotation_mask_classification_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    value = validate_rotation_mask_classification(payload)
    if raw != _pretty_json_bytes(value):
        raise ScorerFitCorpusV2DesignError(
            "mask classification raw bytes changed")
    return {
        "path": str(MASK_CLASSIFICATION_RELATIVE_PATH),
        "schema": MASK_CLASSIFICATION_SCHEMA,
        "self_digest_key": MASK_CLASSIFICATION_SELF_KEY,
        "self_digest": value[MASK_CLASSIFICATION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": value["source_repository_commit"],
    }


def _validate_classification_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScorerFitCorpusV2DesignError(
            "rotation-mask classification binding is malformed")
    row = dict(value)
    required = {
        "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
        "byte_count", "source_repository_commit",
    }
    if (set(row) != required
            or row.get("path") != str(MASK_CLASSIFICATION_RELATIVE_PATH)
            or row.get("schema") != MASK_CLASSIFICATION_SCHEMA
            or row.get("self_digest_key") != MASK_CLASSIFICATION_SELF_KEY
            or not _is_hex(row.get("self_digest"), 64)
            or not _is_hex(row.get("raw_sha256"), 64)
            or isinstance(row.get("byte_count"), bool)
            or not isinstance(row.get("byte_count"), int)
            or row["byte_count"] <= 0
            or not _is_hex(row.get("source_repository_commit"), 40)):
        raise ScorerFitCorpusV2DesignError(
            "rotation-mask classification binding changed")
    return row


def build_design_amendment(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        rotation_mask_classification_binding: Mapping[str, Any],
        predecessor_validation: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the prospective V2 corpus design without reading any artifact."""

    if not _is_hex(source_repository_commit, 40):
        raise ScorerFitCorpusV2DesignError("source commit is malformed")
    sources = _validate_source_bindings(list(source_bindings))
    classification = _validate_classification_binding(
        rotation_mask_classification_binding)
    predecessor_projection = _validate_predecessor_validation_projection(
        predecessor_validation)
    if classification["source_repository_commit"] != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "classification and design source commits differ")
    lineage = _predecessor_lineage()
    payload: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "status": DESIGN_STATUS,
        "corpus_design_version": "scorer_fit_corpus_v2_full_bank_v1",
        "source_repository_commit": source_repository_commit,
        "source_bindings": sources,
        "source_binding_set_digest": canonical_digest(sources),
        "rotation_mask_classification": classification,
        "predecessor_validation": predecessor_projection,
        "preoutcome_lineage": lineage,
        "preoutcome_lineage_digest": canonical_digest(lineage),
        "preserved_scientific_contract_bindings": copy.deepcopy(
            PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS),
        "preserved_scientific_contract_bindings_digest": canonical_digest(
            PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS),
        "scientific_binding_dispositions": {
            "lineage_only_superseded_partial_allocation_bindings": [
                "scorer_fit_allocation_design_digest",
                "candidate_allocator_contract_digest",
                "candidate_allocation_amendment_digest",
                "pre_identity_allocation_validation_digest",
            ],
            "all_other_bound_selector_exclusion_oracle_render_preprocess_"
            "encoder_scorer_fields": "PRESERVED_ACTIVE_UNCHANGED",
            "lineage_binding_deleted_or_reinterpreted": False,
        },
        "supersession": {
            "status": SIX_OF_TWELVE_SUPERSESSION,
            "superseded_only": [
                "six candidates per state", "candidate rotations",
                "candidate exact-60 appearances", "scene-to-rotation assignment",
                "rotation eligibility used for subset allocation",
                "residual candidate-frequency MILPs",
                "partial-subset candidate co-occurrence balancing",
                "external or global scene/rotation optimization",
            ],
            "selector_superseded": False,
            "identity_exclusions_superseded": False,
            "oracle_superseded": False,
            "render_preprocess_encoder_superseded": False,
            "scorer_architecture_or_qualification_superseded": False,
        },
        "active_full_bank_rule": {
            "candidate_indices": list(CANDIDATE_INDICES),
            "every_selected_state_receives_every_candidate_exactly_once": True,
            "per_state_subset_decision": False,
            "per_state_rotation_decision": False,
            "candidate_assignment_optimization": False,
            "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        },
        "count_contract": copy.deepcopy(FULL_BANK_COUNT_CONTRACT),
        "count_contract_digest": canonical_digest(FULL_BANK_COUNT_CONTRACT),
        "state_design": {
            "families": list(FAMILIES),
            "strata": list(STRATA),
            "states_total": 120,
            "states_per_family": 15,
            "states_per_family_stratum": 5,
            "fit_states": 96,
            "calibration_states": 24,
            "fit_per_family_stratum": 4,
            "calibration_per_family_stratum": 1,
            "scene_disjoint_fit_calibration": True,
            "fixed_outcome_free_state_count": 115,
            "unresolved_small_completion_state_count": 5,
            "eligible_small_completion_scene_count": 17,
        },
        "small_completion_selection": copy.deepcopy(
            COMPLETION_ORDERING_CONTRACT),
        "small_completion_selection_digest": canonical_digest(
            COMPLETION_ORDERING_CONTRACT),
        "completion_full_bank_revalidation": {
            "formula": "max(d0 - 0.75 m, 0) <= L_max",
            "l_max_scope": "maximum over candidate indices 0--11",
            "actual_previous_applied_command": True,
            "frozen_slew_limiter": True,
            "exact_twenty_tick_plans": True,
            "branch_execution": False,
            "realised_outcome": False,
        },
        "preoutcome_feasibility_terminal": {
            "path": str(
                SCORER_FIT_RELATIVE_PATH /
                "full_bank_preoutcome_feasibility_failure_v2.json"),
            "condition": (
                "fewer than one calibration-eligible or four distinct fit-"
                "eligible full-bank-valid small completion scenes"),
            "mutually_exclusive_with_success_manifests": True,
            "candidate_outcomes_consumed": False,
            "automatic_selector_or_candidate_bank_revision": False,
        },
        "preserved_nonallocation_science": {
            "oracle_v1_2_digest": ORACLE_V1_2_DIGEST,
            "horizon_seconds": 2.0,
            "horizon_blocks": 4,
            "snapshot_time_goal_binding": True,
            "textured_v03_rendering": True,
            "crop_preprocessing_target_encoder": True,
            "scorer_architecture_training_and_qualification": True,
            "no_latent_baseline": True,
            "final_200_state_corpus_authorized_in_this_pass": False,
            "future_final_evaluation_rule": (
                "when separately authorised, final_eval selection excludes all "
                "120 scenes in the active scorer-fit V2 manifest"),
            "preexisting_reserved_final_evaluation_scene_set": False,
            "final_evaluation_manifest_absent_at_issue": True,
        },
        "selection_field_policy": {
            "candidate_outcome": False,
            "branch_label": False,
            "frame": False,
            "latent": False,
            "scorer_metric": False,
            "predictor_output": False,
            "historical_receipts_reopened_for_exact_lineage_custody": True,
            "historical_receipts_used_for_selection": False,
        },
        "issuance_boundary": {
            "source_tree_clean_and_committed": True,
            "classification_issued_first": True,
            "v1_and_v2_failures_immutable": True,
            "six_of_twelve_exact_infeasibility_immutable": True,
            "v2_state_selection_started": False,
            "v2_branch_execution_started": False,
            "milp_or_cp_sat_run": False,
            "performance_benchmark_run": False,
            "candidate_outcome_or_downstream_metric_used": False,
        },
    }
    payload[DESIGN_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_design_amendment(
        payload: Mapping[str, Any], *, validate_live_source: bool = False,
        root: Path = ROOT) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ScorerFitCorpusV2DesignError("design amendment is not a mapping")
    value = copy.deepcopy(dict(payload))
    if (value.get("schema") != DESIGN_SCHEMA
            or value.get("status") != DESIGN_STATUS
            or value.get(DESIGN_SELF_KEY)
            != canonical_digest(_without(value, DESIGN_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError("design amendment binding changed")
    expected = build_design_amendment(
        source_repository_commit=str(value.get("source_repository_commit", "")),
        source_bindings=value.get("source_bindings", []),
        rotation_mask_classification_binding=value.get(
            "rotation_mask_classification", {}),
        predecessor_validation=value.get("predecessor_validation", {}))
    if value != expected:
        raise ScorerFitCorpusV2DesignError("design amendment is not exact")
    _validate_predecessor_lineage(value["preoutcome_lineage"])
    if validate_live_source:
        commit, sources = clean_source_authority(root=root)
        if commit != value["source_repository_commit"] or sources != value["source_bindings"]:
            raise ScorerFitCorpusV2DesignError(
                "live source differs from design amendment")
    return value


def design_amendment_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    value = validate_design_amendment(payload)
    if raw != _pretty_json_bytes(value):
        raise ScorerFitCorpusV2DesignError("design amendment raw bytes changed")
    return {
        "path": str(DESIGN_RELATIVE_PATH),
        "schema": DESIGN_SCHEMA,
        "self_digest_key": DESIGN_SELF_KEY,
        "self_digest": value[DESIGN_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": value["source_repository_commit"],
    }


_ISSUED_DESIGN_AUTHORITY_KEYS = frozenset({
    "rotation_mask_classification_payload",
    "rotation_mask_classification_binding",
    "design_amendment_payload",
    "design_amendment_binding",
})

PRESELECTION_ALIAS_FAILURE_BOUNDARY_V1 = {
    "status": (
        "IMMUTABLE_PRESELECTION_FAILURE_REGISTERED_DEVELOPMENT_STAGE_A_"
        "MANIFEST_ALIAS_GUARD"),
    "issued_source_repository_commit":
        ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
    "classification_issued": True,
    "design_amendment_issued": True,
    "freeze_manifests_stage_entered": True,
    "preoutcome_input_loader_entered": True,
    "active_design_authority_validated": True,
    "predecessor_fixed_state_count_validated": 115,
    "eligible_small_completion_scene_count_validated": 17,
    "historical_preserved_rotation_evidence_loaded": True,
    "historical_rotation_evidence_used_as_active_mask": False,
    "factorial_exclusion_setup_loaded": True,
    "invalid_identity_index_setup_loaded": True,
    "failure_site": (
        "INITIAL_REGISTERED_EXCLUSION_MANIFEST_PATH_GUARD"),
    "failed_registered_path": (
        ".generated/go2_counterfactual_fidelity_v1_2/"
        "stage_a_identity_manifest.json"),
    "oracle_v1_1_identity_manifest_json_read": False,
    "oracle_v1_2_identity_manifest_json_read": False,
    "development_stage_a_identity_manifest_json_read": False,
    "exclusion_authority_returned": False,
    "candidate_revalidation_started": False,
    "completion_ordering_started": False,
    "small_completion_selection_started": False,
    "preoutcome_manifest_or_selection_artifact_issued": False,
    "solver_or_optimisation_invoked": False,
    "branch_execution_started": False,
    "candidate_outcome_or_branch_label_read": False,
    "frame_or_latent_created_or_read": False,
    "scorer_metric_or_predictor_output_read": False,
    "successor_scorer_contract_issued": False,
    "final_200_state_corpus_generated": False,
    "nothing_running": True,
}

_SOURCE_CORRECTION_V1_KEYS = frozenset({
    "schema", "status", "complete", "source_correction_version",
    "source_repository_commit", "source_bindings",
    "source_binding_set_digest", "historical_source_repository_commit",
    "immutable_issued_design_authority",
    "immutable_issued_design_authority_digest",
    "preserved_scientific_design_digest",
    "preserved_rotation_mask_classification_digest",
    "runtime_outputs_absent_at_issue",
    "runtime_outputs_absent_at_issue_digest",
    "preselection_alias_failure_boundary",
    "preselection_alias_failure_boundary_digest",
    "source_correction", "source_correction_material_digest",
    "issuance_boundary", SOURCE_CORRECTION_SELF_KEY,
})


def validate_immutable_issued_design_authority(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable issued payloads and their exact canonical bytes."""

    if not isinstance(value, Mapping) or set(value) != _ISSUED_DESIGN_AUTHORITY_KEYS:
        raise ScorerFitCorpusV2DesignError(
            "immutable issued V2 design authority is not closed")
    authority = copy.deepcopy(dict(value))
    classification = validate_rotation_mask_classification(
        authority["rotation_mask_classification_payload"],
        validate_live_source=False)
    design = validate_design_amendment(
        authority["design_amendment_payload"], validate_live_source=False)
    classification_binding = rotation_mask_classification_artifact_binding(
        classification, _pretty_json_bytes(classification))
    design_binding = design_amendment_artifact_binding(
        design, _pretty_json_bytes(design))
    if (authority["rotation_mask_classification_binding"]
            != classification_binding
            or authority["design_amendment_binding"] != design_binding
            or classification["source_repository_commit"]
            != ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT
            or design["source_repository_commit"]
            != ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT
            or design["rotation_mask_classification"]
            != classification_binding):
        raise ScorerFitCorpusV2DesignError(
            "immutable issued V2 design payload or artifact binding changed")
    return authority


def build_preselection_source_correction_v1(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_issued_design_authority: Mapping[str, Any],
        runtime_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the orthogonal source-only correction without generated access."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    issued = validate_immutable_issued_design_authority(
        immutable_issued_design_authority)
    classification = issued["rotation_mask_classification_payload"]
    design = issued["design_amendment_payload"]
    if classification["source_bindings"] != design["source_bindings"]:
        raise ScorerFitCorpusV2DesignError(
            "issued classification/design source closures differ")
    changed = _changed_source_paths(
        classification["source_bindings"], current_sources)
    if changed != sorted(SOURCE_CORRECTION_V1_ALLOWED_CHANGED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "preselection correction changed an unauthorised source path")
    absence = _validate_absence_projection(
        list(runtime_outputs_absent_at_issue), phase="design")
    failure = copy.deepcopy(PRESELECTION_ALIAS_FAILURE_BOUNDARY_V1)
    correction = {
        "status": "SOURCE_ONLY_REGISTERED_DEVELOPMENT_MANIFEST_ALIAS_CORRECTION",
        "defect": (
            "REGISTERED_DEVELOPMENT_STAGE_A_MANIFEST_ALIAS_FAILED_THE_"
            "INITIAL_EXCLUSION_PATH_GUARD"),
        "correction": (
            "RESOLVE_AND_VALIDATE_THE_REGISTERED_DEVELOPMENT_MANIFEST_"
            "THROUGH_ITS_EXACT_IMMUTABLE_ALIAS_TARGET"),
        "historical_source_repository_commit":
            ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        "successor_source_repository_commit": source_repository_commit,
        "allowed_changed_source_paths": list(
            SOURCE_CORRECTION_V1_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_source_binding_set_digest": classification[
            "source_binding_set_digest"],
        "successor_source_binding_set_digest": canonical_digest(
            current_sources),
        "old_classification_or_design_overwritten": False,
        "old_classification_or_design_reissued": False,
        "old_scientific_design_digest_preserved": True,
        "scene_state_or_candidate_pool_changed": False,
        "candidate_bank_or_frequency_changed": False,
        "selector_exclusion_rule_or_quota_changed": False,
        "oracle_render_preprocess_or_target_encoder_changed": False,
        "scorer_architecture_training_or_qualification_changed": False,
        "scientific_contract_changed": False,
        "candidate_outcome_or_downstream_metric_used": False,
    }
    payload: dict[str, Any] = {
        "schema": SOURCE_CORRECTION_V1_SCHEMA,
        "status": SOURCE_CORRECTION_V1_STATUS,
        "complete": True,
        "source_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        "immutable_issued_design_authority": copy.deepcopy(issued),
        "immutable_issued_design_authority_digest": canonical_digest(issued),
        "preserved_scientific_design_digest": design[DESIGN_SELF_KEY],
        "preserved_rotation_mask_classification_digest": classification[
            MASK_CLASSIFICATION_SELF_KEY],
        "runtime_outputs_absent_at_issue": absence,
        "runtime_outputs_absent_at_issue_digest": canonical_digest(absence),
        "preselection_alias_failure_boundary": failure,
        "preselection_alias_failure_boundary_digest": canonical_digest(failure),
        "source_correction": correction,
        "source_correction_material_digest": canonical_digest(correction),
        "issuance_boundary": {
            "source_tree_clean_and_committed": True,
            "classification_and_design_preserved_immutable": True,
            "failure_preserved_truthfully": True,
            "double_runtime_output_absence_audit_required": True,
            "preselection_only": True,
            "selection_or_manifest_issued": False,
            "branch_execution_started": False,
            "candidate_outcomes_consumed": False,
            "solver_or_optimisation_invoked": False,
        },
    }
    if set(payload) != _SOURCE_CORRECTION_V1_KEYS - {SOURCE_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction construction surface changed")
    payload[SOURCE_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_preselection_source_correction_v1(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    """Validate the correction and immutable old scientific authorities."""

    if not isinstance(payload, Mapping) or set(payload) != _SOURCE_CORRECTION_V1_KEYS:
        raise ScorerFitCorpusV2DesignError(
            "preselection source correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != SOURCE_CORRECTION_V1_SCHEMA
            or correction.get("status") != SOURCE_CORRECTION_V1_STATUS
            or correction.get("complete") is not True
            or correction.get("source_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction version changed")
    expected = build_preselection_source_correction_v1(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_issued_design_authority=correction.get(
            "immutable_issued_design_authority", {}),
        runtime_outputs_absent_at_issue=correction.get(
            "runtime_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(SOURCE_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, SOURCE_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or _load_issued_design_authority_for_source_correction(
                    root=root)
                != correction["immutable_issued_design_authority"]):
            raise ScorerFitCorpusV2DesignError(
                "live source or immutable issued authority differs from correction")
        if require_runtime_outputs_absent:
            observed = audit_v2_runtime_outputs_absent(
                root=root, phase="design")
            if observed != correction["runtime_outputs_absent_at_issue"]:
                raise ScorerFitCorpusV2DesignError(
                    "runtime-output absence changed after correction issuance")
    return correction


def preselection_source_correction_v1_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_preselection_source_correction_v1(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction raw bytes changed")
    return {
        "path": str(SOURCE_CORRECTION_V1_RELATIVE_PATH),
        "schema": SOURCE_CORRECTION_V1_SCHEMA,
        "self_digest_key": SOURCE_CORRECTION_SELF_KEY,
        "self_digest": correction[SOURCE_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


_IMMUTABLE_SOURCE_CORRECTION_V1_KEYS = frozenset({"payload", "binding"})

PRESELECTION_ALIAS_FAILURE_BOUNDARY_V2 = {
    "status": (
        "IMMUTABLE_PRESELECTION_FAILURE_MANAGED_FINAL_EVAL_ABSENCE_PIN_GUARD"),
    "immutable_source_correction_v1_issued": True,
    "immutable_source_correction_v1_digest":
        IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST,
    "immutable_source_correction_v1_source_repository_commit":
        IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT,
    "freeze_manifests_stage_reentered": True,
    "preoutcome_input_loader_entered": True,
    "active_corrected_design_authority_validated": True,
    "predecessor_fixed_state_count_validated": 115,
    "eligible_small_completion_scene_count_validated": 17,
    "historical_preserved_rotation_evidence_loaded": True,
    "historical_rotation_evidence_used_as_active_mask": False,
    "factorial_exclusion_setup_loaded": True,
    "invalid_identity_index_setup_loaded": True,
    "oracle_v1_1_identity_manifest_json_read_and_validated": True,
    "oracle_v1_2_identity_manifest_json_read_and_validated": True,
    "development_stage_a_identity_manifest_json_read_and_validated": True,
    "registered_development_manifest_alias_resolved_and_validated": True,
    "failure_site": "PROSPECTIVE_FINAL_EVAL_MANIFEST_ABSENCE_PATH_GUARD",
    "failure_cause": "OUT_ROOT_IS_A_REGISTERED_GENERATED_ROOT_SYMLINK",
    "failed_logical_path": (
        ".generated/go2_branch_corpus_v1_2/final_eval/state_manifest.json"),
    "final_eval_manifest_content_read": False,
    "prospective_final_eval_absence_verdict_returned": False,
    "exclusion_authority_returned": False,
    "candidate_revalidation_started": False,
    "completion_ordering_started": False,
    "small_completion_selection_started": False,
    "preoutcome_manifest_or_selection_artifact_issued": False,
    "solver_or_optimisation_invoked": False,
    "branch_execution_started": False,
    "candidate_outcome_or_branch_label_read": False,
    "frame_or_latent_created_or_read": False,
    "scorer_metric_or_predictor_output_read": False,
    "successor_scorer_contract_issued": False,
    "final_200_state_corpus_generated": False,
    "nothing_running": True,
}

_SOURCE_CORRECTION_V2_KEYS = frozenset({
    "schema", "status", "complete", "source_correction_version",
    "source_repository_commit", "source_bindings",
    "source_binding_set_digest", "historical_source_repository_commit",
    "immutable_preselection_source_correction_v1",
    "immutable_preselection_source_correction_v1_digest",
    "preserved_scientific_design_digest",
    "preserved_rotation_mask_classification_digest",
    "runtime_outputs_absent_at_issue",
    "runtime_outputs_absent_at_issue_digest",
    "preselection_alias_failure_boundary",
    "preselection_alias_failure_boundary_digest",
    "source_correction", "source_correction_material_digest",
    "issuance_boundary", SOURCE_CORRECTION_SELF_KEY,
})


def validate_immutable_preselection_source_correction_v1(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete immutable V1 correction and exact raw binding."""

    if (not isinstance(value, Mapping)
            or set(value) != _IMMUTABLE_SOURCE_CORRECTION_V1_KEYS
            or not isinstance(value.get("payload"), Mapping)):
        raise ScorerFitCorpusV2DesignError(
            "immutable preselection source correction V1 is not closed")
    authority = copy.deepcopy(dict(value))
    payload = validate_preselection_source_correction_v1(
        authority["payload"], validate_live_authorities=False)
    expected_binding = preselection_source_correction_v1_artifact_binding(
        payload, _pretty_json_bytes(payload))
    if (authority.get("binding") != expected_binding
            or payload.get(SOURCE_CORRECTION_SELF_KEY)
            != IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST
            or payload.get("source_repository_commit")
            != IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT
            or expected_binding.get("self_digest")
            != IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "immutable preselection source correction V1 changed")
    return authority


def build_preselection_source_correction_v2(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_preselection_source_correction_v1: Mapping[str, Any],
        runtime_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build chained correction V2 without reading an artifact or outcome."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    immutable_v1 = validate_immutable_preselection_source_correction_v1(
        immutable_preselection_source_correction_v1)
    v1 = immutable_v1["payload"]
    changed = _changed_source_paths(v1["source_bindings"], current_sources)
    if changed != sorted(SOURCE_CORRECTION_V2_ALLOWED_CHANGED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "preselection correction V2 changed an unauthorised source path")
    absence = _validate_absence_projection(
        list(runtime_outputs_absent_at_issue), phase="design")
    issued = v1["immutable_issued_design_authority"]
    classification = issued["rotation_mask_classification_payload"]
    design = issued["design_amendment_payload"]
    failure = copy.deepcopy(PRESELECTION_ALIAS_FAILURE_BOUNDARY_V2)
    correction = {
        "status": "SOURCE_ONLY_MANAGED_FINAL_EVAL_ABSENCE_PIN_CORRECTION",
        "defect": (
            "PROSPECTIVE_FINAL_EVAL_ABSENCE_CHECK_USED_AN_ORDINARY_PATH_"
            "GUARD_ON_REGISTERED_SYMLINKED_OUT_ROOT"),
        "correction": (
            "PIN_THE_EXACT_ABSENT_FINAL_EVAL_MANIFEST_LEAF_THROUGH_THE_"
            "REGISTERED_MANAGED_OUT_ROOT_WITHOUT_READING_FINAL_EVAL_DATA"),
        "historical_source_repository_commit":
            IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT,
        "successor_source_repository_commit": source_repository_commit,
        "allowed_changed_source_paths": list(
            SOURCE_CORRECTION_V2_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_source_binding_set_digest": v1[
            "source_binding_set_digest"],
        "successor_source_binding_set_digest": canonical_digest(
            current_sources),
        "immutable_v1_correction_overwritten_or_reissued": False,
        "old_classification_or_design_overwritten_or_reissued": False,
        "old_scientific_design_digest_preserved": True,
        "managed_out_root_identity_may_change": False,
        "final_eval_manifest_may_exist_at_correction_issue": False,
        "final_eval_manifest_content_read": False,
        "scene_state_or_candidate_pool_changed": False,
        "candidate_bank_or_frequency_changed": False,
        "selector_exclusion_rule_or_quota_changed": False,
        "oracle_render_preprocess_or_target_encoder_changed": False,
        "scorer_architecture_training_or_qualification_changed": False,
        "scientific_contract_changed": False,
        "candidate_outcome_or_downstream_metric_used": False,
    }
    payload: dict[str, Any] = {
        "schema": SOURCE_CORRECTION_V2_SCHEMA,
        "status": SOURCE_CORRECTION_V2_STATUS,
        "complete": True,
        "source_correction_version": 2,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT,
        "immutable_preselection_source_correction_v1": copy.deepcopy(
            immutable_v1),
        "immutable_preselection_source_correction_v1_digest":
            IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST,
        "preserved_scientific_design_digest": design[DESIGN_SELF_KEY],
        "preserved_rotation_mask_classification_digest": classification[
            MASK_CLASSIFICATION_SELF_KEY],
        "runtime_outputs_absent_at_issue": absence,
        "runtime_outputs_absent_at_issue_digest": canonical_digest(absence),
        "preselection_alias_failure_boundary": failure,
        "preselection_alias_failure_boundary_digest": canonical_digest(failure),
        "source_correction": correction,
        "source_correction_material_digest": canonical_digest(correction),
        "issuance_boundary": {
            "immutable_source_correction_v1_preserved": True,
            "source_tree_clean_and_committed": True,
            "failure_preserved_truthfully": True,
            "double_runtime_output_absence_audit_required": True,
            "preselection_only": True,
            "selection_or_manifest_issued": False,
            "branch_execution_started": False,
            "candidate_outcomes_consumed": False,
            "solver_or_optimisation_invoked": False,
        },
    }
    if set(payload) != _SOURCE_CORRECTION_V2_KEYS - {SOURCE_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 construction surface changed")
    payload[SOURCE_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_preselection_source_correction_v2(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _SOURCE_CORRECTION_V2_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "preselection source correction V2 is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != SOURCE_CORRECTION_V2_SCHEMA
            or correction.get("status") != SOURCE_CORRECTION_V2_STATUS
            or correction.get("complete") is not True
            or correction.get("source_correction_version") != 2):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 version changed")
    expected = build_preselection_source_correction_v2(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_preselection_source_correction_v1=correction.get(
            "immutable_preselection_source_correction_v1", {}),
        runtime_outputs_absent_at_issue=correction.get(
            "runtime_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(SOURCE_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, SOURCE_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or _load_immutable_preselection_source_correction_v1(
                    root=root)
                != correction[
                    "immutable_preselection_source_correction_v1"]):
            raise ScorerFitCorpusV2DesignError(
                "live source or immutable V1 differs from correction V2")
        if require_runtime_outputs_absent:
            observed = audit_v2_runtime_outputs_absent(
                root=root, phase="design")
            if observed != correction["runtime_outputs_absent_at_issue"]:
                raise ScorerFitCorpusV2DesignError(
                    "runtime-output absence changed after correction V2 issuance")
    return correction


def preselection_source_correction_v2_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_preselection_source_correction_v2(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 raw bytes changed")
    return {
        "path": str(SOURCE_CORRECTION_V2_RELATIVE_PATH),
        "schema": SOURCE_CORRECTION_V2_SCHEMA,
        "self_digest_key": SOURCE_CORRECTION_SELF_KEY,
        "self_digest": correction[SOURCE_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


_IMMUTABLE_SOURCE_CORRECTION_V2_KEYS = frozenset({"payload", "binding"})

PRESELECTION_STRUCTURAL_VALIDATION_FAILURE_BOUNDARY = {
    "status": (
        "IMMUTABLE_PRESELECTION_FAILURE_SIGNED_BODY_CLEARANCE_REJECTED"),
    "immutable_source_correction_v2_issued": True,
    "immutable_source_correction_v2_digest":
        IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST,
    "immutable_source_correction_v2_source_repository_commit":
        IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT,
    "freeze_manifests_stage_reentered": True,
    "preoutcome_input_loader_entered": True,
    "active_corrected_design_authority_validated": True,
    "predecessor_fixed_state_count_validated": 115,
    "eligible_small_completion_scene_count_validated": 17,
    "historical_preserved_rotation_evidence_loaded": True,
    "historical_rotation_evidence_used_as_active_mask": False,
    "exclusion_authority_returned": True,
    "eligible_small_completion_candidate_revalidation_count": 17,
    "eligible_small_completion_candidate_revalidation_complete": True,
    "deterministic_five_scene_selection_computed_in_memory": True,
    "deterministic_five_scene_selection_artifact_written": False,
    "complete_120_state_revalidation_started": True,
    "complete_120_state_revalidation_completed": False,
    "failure_site": "FIRST_COMPLETE_120_STATE_STRUCTURAL_REVALIDATION",
    "failure_field": "body_clearance_m",
    "first_rejected_value_relation": "body_clearance_m < 0.0",
    "first_rejected_value_was_finite": True,
    "first_rejected_value_was_valid_under_frozen_scientific_rules": True,
    "exact_exception": (
        "RuntimeError: full-bank structural field body_clearance_m is malformed"),
    "defective_validator_requirement": "body_clearance_m >= 0.0",
    "frozen_safety_rule": "body_clearance_m <= 0.10",
    "preoutcome_manifest_or_selection_artifact_issued": False,
    "solver_or_optimisation_invoked": False,
    "branch_execution_started": False,
    "candidate_outcome_or_branch_label_read": False,
    "frame_or_latent_created_or_read": False,
    "scorer_metric_or_predictor_output_read": False,
    "successor_scorer_contract_issued": False,
    "final_200_state_corpus_generated": False,
    "nothing_running": True,
}

POST_FIX_PRODUCTION_BUNDLE_DRY_RUN = {
    "status": "PASS_ACTUAL_PRODUCTION_BUNDLE_IN_MEMORY_ONLY",
    "production_builder_invoked": True,
    "production_replay_validator_invoked": True,
    "verify_scene_files": True,
    "state_count": 120,
    "candidate_count_per_state": 12,
    "assignment_count": 1_440,
    "immutable_source_authority_for_dirty_source_equality": {
        "source_repository_commit":
            IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT,
        "source_correction_digest": IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST,
        "authority": (
            "INSTALLED_SOURCE_CORRECTION_V2_RECORDED_SOURCE_BINDINGS_ONLY"),
    },
    "live_dirty_source_treated_as_issued_authority": False,
    "live_clean_source_equality_check_substituted_for_diagnostic": True,
    "scientific_constraint_validator_bypassed": False,
    "payload_or_digest_validator_bypassed": False,
    "generated_artifact_validator_bypassed": False,
    "bundle_built_in_memory": True,
    "bundle_replay_validated_in_memory": True,
    "payload_digests": {
        "full_bank_small_completion_selection_v2":
            "6c2acabaa4a69e8eea4d98ba71abf8a1d27587a8ac22536631716620d10b652a",
        "full_bank_preoutcome_state_revalidation_v2":
            "1d7ce9a49a672e72adb8558a472b5e7f1ecaa625261a4c4e52bf77128ee44d58",
        "state_shard_small_enclosed_maze_v2":
            "c413d0c834be06b66859eaeefdd3dfd75ee9f3887c9f1e31a53b7947b8ece937",
        "state_manifest_v2":
            "a85be133e696d733e4a8eff2e4420fa3b9807a26b76b264500ed36c314fbf408",
        "full_bank_assignment_manifest_v2":
            "46401ff9ea7f32cf5460e17d7c21854af650e2bf0c98db512f28a047f1490233",
    },
    "all_five_payloads_replay_valid": True,
    "generated_artifact_written": False,
    "branch_execution_started": False,
    "candidate_outcome_or_branch_label_read": False,
    "solver_or_optimisation_invoked": False,
}

_SOURCE_CORRECTION_KEYS = frozenset({
    "schema", "status", "complete",
    "structural_validation_correction_version",
    "source_repository_commit", "source_bindings",
    "source_binding_set_digest", "historical_source_repository_commit",
    "immutable_preselection_source_correction_v2",
    "immutable_preselection_source_correction_v2_digest",
    "transitive_immutable_preselection_source_correction_v1_digest",
    "preserved_scientific_design_digest",
    "preserved_rotation_mask_classification_digest",
    "runtime_outputs_absent_at_issue",
    "runtime_outputs_absent_at_issue_digest",
    "preselection_structural_validation_failure_boundary",
    "preselection_structural_validation_failure_boundary_digest",
    "post_fix_production_bundle_dry_run",
    "post_fix_production_bundle_dry_run_digest",
    "structural_validation_correction",
    "structural_validation_correction_material_digest",
    "issuance_boundary", SOURCE_CORRECTION_SELF_KEY,
})


def validate_immutable_preselection_source_correction_v2(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate complete immutable V2 correction and its exact raw binding."""

    if (not isinstance(value, Mapping)
            or set(value) != _IMMUTABLE_SOURCE_CORRECTION_V2_KEYS
            or not isinstance(value.get("payload"), Mapping)):
        raise ScorerFitCorpusV2DesignError(
            "immutable preselection source correction V2 is not closed")
    authority = copy.deepcopy(dict(value))
    payload = validate_preselection_source_correction_v2(
        authority["payload"], validate_live_authorities=False)
    expected_binding = preselection_source_correction_v2_artifact_binding(
        payload, _pretty_json_bytes(payload))
    nested_v1 = validate_immutable_preselection_source_correction_v1(
        payload["immutable_preselection_source_correction_v1"])
    if (authority.get("binding") != expected_binding
            or payload.get(SOURCE_CORRECTION_SELF_KEY)
            != IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST
            or payload.get("source_repository_commit")
            != IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT
            or expected_binding.get("self_digest")
            != IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST
            or nested_v1["payload"].get(SOURCE_CORRECTION_SELF_KEY)
            != IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "immutable preselection source correction V2 changed")
    return authority


def build_preselection_source_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_preselection_source_correction_v2: Mapping[str, Any],
        runtime_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the final source-only structural-validation correction."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "structural-validation correction commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    immutable_v2 = validate_immutable_preselection_source_correction_v2(
        immutable_preselection_source_correction_v2)
    v2 = immutable_v2["payload"]
    changed = _changed_source_paths(v2["source_bindings"], current_sources)
    if changed != sorted(SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "structural-validation correction changed an unauthorised source path")
    absence = _validate_absence_projection(
        list(runtime_outputs_absent_at_issue), phase="design")
    immutable_v1 = validate_immutable_preselection_source_correction_v1(
        v2["immutable_preselection_source_correction_v1"])
    issued = immutable_v1["payload"]["immutable_issued_design_authority"]
    classification = issued["rotation_mask_classification_payload"]
    design = issued["design_amendment_payload"]
    failure = copy.deepcopy(
        PRESELECTION_STRUCTURAL_VALIDATION_FAILURE_BOUNDARY)
    dry_run = copy.deepcopy(POST_FIX_PRODUCTION_BUNDLE_DRY_RUN)
    correction_material = {
        "status": "SOURCE_ONLY_SIGNED_BODY_CLEARANCE_DOMAIN_CORRECTION",
        "defect": (
            "STRUCTURAL_REVALIDATION_INCORRECTLY_REQUIRED_NONNEGATIVE_"
            "BODY_CLEARANCE_M"),
        "correction": (
            "ACCEPT_EVERY_FINITE_SIGNED_BODY_CLEARANCE_M_WHILE_RETAINING_"
            "FINITE_NONNEGATIVE_CLEARANCE_M_AND_THE_UNCHANGED_SAFETY_UPPER_"
            "BOUND"),
        "historical_source_repository_commit":
            IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT,
        "successor_source_repository_commit": source_repository_commit,
        "allowed_changed_source_paths": list(
            SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_source_binding_set_digest": v2[
            "source_binding_set_digest"],
        "successor_source_binding_set_digest": canonical_digest(
            current_sources),
        "body_clearance_m_domain": "FINITE_SIGNED_REAL",
        "clearance_m_domain": "FINITE_REAL_GTE_0",
        "safety_enriched_body_clearance_upper_bound_m": 0.10,
        "immutable_v2_correction_overwritten_or_reissued": False,
        "immutable_v1_correction_overwritten_or_reissued": False,
        "old_classification_or_design_overwritten_or_reissued": False,
        "old_scientific_design_digest_preserved": True,
        "scene_state_or_candidate_pool_changed": False,
        "candidate_bank_or_frequency_changed": False,
        "selector_exclusion_rule_or_quota_changed": False,
        "oracle_render_preprocess_or_target_encoder_changed": False,
        "scorer_architecture_training_or_qualification_changed": False,
        "scientific_contract_changed": False,
        "candidate_outcome_or_downstream_metric_used": False,
    }
    payload: dict[str, Any] = {
        "schema": SOURCE_CORRECTION_SCHEMA,
        "status": SOURCE_CORRECTION_STATUS,
        "complete": True,
        "structural_validation_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT,
        "immutable_preselection_source_correction_v2": copy.deepcopy(
            immutable_v2),
        "immutable_preselection_source_correction_v2_digest":
            IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST,
        "transitive_immutable_preselection_source_correction_v1_digest":
            IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST,
        "preserved_scientific_design_digest": design[DESIGN_SELF_KEY],
        "preserved_rotation_mask_classification_digest": classification[
            MASK_CLASSIFICATION_SELF_KEY],
        "runtime_outputs_absent_at_issue": absence,
        "runtime_outputs_absent_at_issue_digest": canonical_digest(absence),
        "preselection_structural_validation_failure_boundary": failure,
        "preselection_structural_validation_failure_boundary_digest":
            canonical_digest(failure),
        "post_fix_production_bundle_dry_run": dry_run,
        "post_fix_production_bundle_dry_run_digest": canonical_digest(dry_run),
        "structural_validation_correction": correction_material,
        "structural_validation_correction_material_digest": canonical_digest(
            correction_material),
        "issuance_boundary": {
            "immutable_source_correction_v2_preserved": True,
            "immutable_source_correction_v1_preserved_transitively": True,
            "source_tree_clean_and_committed": True,
            "failure_preserved_truthfully": True,
            "post_fix_production_bundle_dry_run_recorded": True,
            "double_runtime_output_absence_audit_required": True,
            "preselection_only": True,
            "selection_or_manifest_issued": False,
            "branch_execution_started": False,
            "candidate_outcomes_consumed": False,
            "solver_or_optimisation_invoked": False,
        },
    }
    if set(payload) != _SOURCE_CORRECTION_KEYS - {SOURCE_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "structural-validation correction construction surface changed")
    payload[SOURCE_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_preselection_source_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _SOURCE_CORRECTION_KEYS:
        raise ScorerFitCorpusV2DesignError(
            "preselection structural-validation correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != SOURCE_CORRECTION_SCHEMA
            or correction.get("status") != SOURCE_CORRECTION_STATUS
            or correction.get("complete") is not True
            or correction.get("structural_validation_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "preselection structural-validation correction version changed")
    expected = build_preselection_source_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_preselection_source_correction_v2=correction.get(
            "immutable_preselection_source_correction_v2", {}),
        runtime_outputs_absent_at_issue=correction.get(
            "runtime_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(SOURCE_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, SOURCE_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "preselection structural-validation correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or _load_immutable_preselection_source_correction_v2(
                    root=root)
                != correction[
                    "immutable_preselection_source_correction_v2"]):
            raise ScorerFitCorpusV2DesignError(
                "live source or immutable correction V2 differs from active "
                "structural-validation correction")
        if require_runtime_outputs_absent:
            observed = audit_v2_runtime_outputs_absent(
                root=root, phase="design")
            if observed != correction["runtime_outputs_absent_at_issue"]:
                raise ScorerFitCorpusV2DesignError(
                    "runtime-output absence changed after structural-validation "
                    "correction issuance")
    return correction


def preselection_source_correction_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_preselection_source_correction(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "preselection structural-validation correction raw bytes changed")
    return {
        "path": str(SOURCE_CORRECTION_RELATIVE_PATH),
        "schema": SOURCE_CORRECTION_SCHEMA,
        "self_digest_key": SOURCE_CORRECTION_SELF_KEY,
        "self_digest": correction[SOURCE_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


_IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_KEYS = frozenset({
    "payload", "binding",
})

MANIFEST_REPLAY_FAILURE_BOUNDARY = {
    "status": (
        "IMMUTABLE_POST_INSTALL_REPLAY_FAILURE_SELF_DIGEST_"
        "CANONICALIZATION_MISMATCH"),
    "active_preselection_correction_issued": True,
    "active_preselection_correction_digest":
        IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST,
    "active_preselection_correction_source_repository_commit":
        IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT,
    "freeze_manifests_stage_entered": True,
    "all_five_preoutcome_artifacts_installed": True,
    "installed_artifact_order": [
        "small_completion_selection",
        "preoutcome_state_revalidation",
        "small_family_state_shard",
        "assignment_manifest",
        "state_manifest",
    ],
    "state_manifest_installed_last_as_terminal_marker": True,
    "all_installed_artifact_modes": "0444",
    "post_install_replay_entered": True,
    "first_replay_role": "small_completion_selection",
    "first_replay_payload_equal_to_solve_free_expected_payload": True,
    "failure_site": "FIRST_POST_INSTALL_ARTIFACT_BINDING",
    "failed_artifact_path": str(
        SCORER_FIT_RELATIVE_PATH /
        "full_bank_small_completion_selection_v2.json"),
    "failure_cause": (
        "GENERIC_ARTIFACT_BINDING_APPLIED_PARALLEL_SEARCH_COMPACT_"
        "CANONICALIZATION_TO_A_BUILDER_DEFAULT_CANONICAL_SELF_DIGEST"),
    "exact_exception": (
        "RuntimeError: full_bank_small_completion_selection_v2.json "
        "parallel self digest mismatch"),
    "post_install_replay_completed": False,
    "installed_artifact_overwritten_deleted_or_reissued": False,
    "preoutcome_feasibility_failure_issued": False,
    "successor_scorer_contract_issued": False,
    "branch_execution_started": False,
    "candidate_outcome_or_branch_label_read": False,
    "frame_or_latent_created_or_read": False,
    "scorer_metric_or_predictor_output_read": False,
    "solver_or_optimisation_invoked": False,
    "final_200_state_corpus_generated": False,
    "nothing_running": True,
}

_MANIFEST_REPLAY_CORRECTION_KEYS = frozenset({
    "schema", "status", "complete", "manifest_replay_correction_version",
    "source_repository_commit", "source_bindings",
    "source_binding_set_digest", "historical_source_repository_commit",
    "immutable_active_preselection_source_correction",
    "immutable_active_preselection_source_correction_digest",
    "preserved_scientific_design_digest",
    "preserved_scientific_manifest_lineage_digest",
    "installed_preoutcome_artifact_bindings",
    "installed_preoutcome_artifact_binding_set_digest",
    "successor_and_runtime_outputs_absent_at_issue",
    "successor_and_runtime_outputs_absent_at_issue_digest",
    "manifest_replay_failure_boundary",
    "manifest_replay_failure_boundary_digest",
    "manifest_replay_correction", "manifest_replay_correction_material_digest",
    "issuance_boundary", MANIFEST_REPLAY_CORRECTION_SELF_KEY,
})


def validate_immutable_active_preselection_source_correction(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact correction 5206 as immutable manifest lineage."""

    if (not isinstance(value, Mapping)
            or set(value)
            != _IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_KEYS
            or not isinstance(value.get("payload"), Mapping)):
        raise ScorerFitCorpusV2DesignError(
            "immutable active preselection source correction is not closed")
    authority = copy.deepcopy(dict(value))
    payload = validate_preselection_source_correction(
        authority["payload"], validate_live_authorities=False)
    expected_binding = preselection_source_correction_artifact_binding(
        payload, _pretty_json_bytes(payload))
    if (authority.get("binding") != expected_binding
            or expected_binding
            != IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_BINDING
            or payload.get(SOURCE_CORRECTION_SELF_KEY)
            != IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST
            or payload.get("source_repository_commit")
            != IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "immutable active preselection source correction changed")
    return authority


def _validate_installed_binding_set(
        rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    installed = copy.deepcopy(list(rows))
    expected = copy.deepcopy(list(
        INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS))
    if installed != expected:
        raise ScorerFitCorpusV2DesignError(
            "installed full-bank V2 preoutcome artifact bindings changed")
    return installed


def validate_installed_full_bank_v2_preoutcome_artifacts(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    """Reopen only the five exact installed, outcome-free manifest artifacts."""

    bindings = _validate_installed_binding_set(
        INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS)
    if [row["role"] for row in bindings] != [
            "small_completion_selection",
            "preoutcome_state_revalidation",
            "small_family_state_shard",
            "assignment_manifest",
            "state_manifest"]:
        raise ScorerFitCorpusV2DesignError(
            "preoutcome artifact installation order changed")
    for binding in bindings:
        path = _pin_generated(
            root, binding["path"],
            label=f"installed {binding['role']} artifact")
        if (not path.is_file() or path.is_symlink()
                or stat.S_IMODE(path.stat().st_mode) != 0o444):
            raise ScorerFitCorpusV2DesignError(
                f"installed {binding['role']} artifact custody changed")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScorerFitCorpusV2DesignError(
                f"installed {binding['role']} artifact is invalid JSON") from exc
        self_key = binding["self_digest_key"]
        if (not isinstance(payload, dict)
                or len(raw) != binding["byte_count"]
                or hashlib.sha256(raw).hexdigest() != binding["raw_sha256"]
                or payload.get("schema") != binding["schema"]
                or payload.get(self_key) != binding["self_digest"]
                or builder_default_canonical_digest(
                    _without(payload, self_key))
                != binding["self_digest"]
                or payload.get(SOURCE_CORRECTION_SELF_KEY)
                != IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST):
            raise ScorerFitCorpusV2DesignError(
                f"installed {binding['role']} artifact changed")
        role = binding["role"]
        role_counts_valid = False
        if role == "small_completion_selection":
            role_counts_valid = (
                payload.get("ordered_candidate_count") == 17
                and isinstance(payload.get("selected_scene_ids"), list)
                and len(payload["selected_scene_ids"]) == 5
                and len(set(payload["selected_scene_ids"])) == 5
                and payload.get("branch_data_consumed") is False
                and payload.get("scientific_outcomes_accessed") is False
                and payload.get("downstream_metric_used") is False
                and payload.get("optimisation_or_solver_used") is False)
        elif role == "preoutcome_state_revalidation":
            role_counts_valid = (
                payload.get("fixed_state_count") == 115
                and payload.get("selected_small_completion_state_count") == 5
                and payload.get("revalidated_state_count") == 120
                and payload.get("completion_state_count") == 40
                and payload.get("full_bank_candidate_indices")
                == list(CANDIDATE_INDICES)
                and payload.get("branch_data_created") is False
                and payload.get("frames_or_latents_accessed") is False
                and payload.get("scientific_outcomes_accessed") is False
                and payload.get("scorer_or_predictor_accessed") is False
                and payload.get(
                    "true_branch_execution_requirement_count") == 0)
        elif role == "small_family_state_shard":
            role_counts_valid = (
                isinstance(payload.get("states"), list)
                and len(payload["states"]) == 15
                and payload.get("branch_data_created") is False
                and payload.get("scientific_outcomes_accessed") is False
                and payload.get("solver_or_optimisation_used") is False)
        elif role == "assignment_manifest":
            role_counts_valid = (
                payload.get("state_count") == 120
                and payload.get("assignment_count") == 1_440
                and payload.get("candidate_indices")
                == list(CANDIDATE_INDICES)
                and payload.get("branch_execution_used") is False)
        elif role == "state_manifest":
            role_counts_valid = (
                isinstance(payload.get("states"), list)
                and len(payload["states"]) == 120
                and payload.get("attempted_branch_count_registered") == 1_440
                and payload.get("candidate_indices_per_state")
                == list(CANDIDATE_INDICES)
                and payload.get("branch_data_created") is False
                and payload.get("frames_or_latents_accessed") is False
                and payload.get("scientific_outcomes_accessed") is False
                and payload.get("scorer_or_predictor_accessed") is False)
        if (payload.get("complete") is not True
                or payload.get("candidate_outcomes_consumed") is not False
                or not role_counts_valid):
            raise ScorerFitCorpusV2DesignError(
                f"installed {role} scientific count or outcome boundary changed")
    failure_path = _pin_generated(
        root,
        SCORER_FIT_RELATIVE_PATH /
        "full_bank_preoutcome_feasibility_failure_v2.json",
        label="preoutcome feasibility failure absence")
    if failure_path.exists() or failure_path.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "preoutcome feasibility failure coexists with installed manifests")
    return bindings


def build_manifest_replay_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_active_preselection_source_correction: Mapping[str, Any],
        installed_preoutcome_artifact_bindings: Sequence[Mapping[str, Any]],
        successor_and_runtime_outputs_absent_at_issue:
            Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the operational replay correction without changing manifests."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    immutable = validate_immutable_active_preselection_source_correction(
        immutable_active_preselection_source_correction)
    active = immutable["payload"]
    changed = _changed_source_paths(active["source_bindings"], current_sources)
    if changed != sorted(
            MANIFEST_REPLAY_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction changed an unauthorised source path")
    installed = _validate_installed_binding_set(
        installed_preoutcome_artifact_bindings)
    absence = _validate_absence_projection(
        list(successor_and_runtime_outputs_absent_at_issue),
        phase="successor_contract")
    failure = copy.deepcopy(MANIFEST_REPLAY_FAILURE_BOUNDARY)
    if (failure.get("active_preselection_correction_digest")
            != IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST
            or failure.get(
                "active_preselection_correction_source_repository_commit")
            != IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT
            or failure.get("installed_artifact_order")
            != [row["role"] for row in installed]):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay failure boundary lineage changed")
    correction_material = {
        "status": "SOURCE_ONLY_POST_INSTALL_REPLAY_BINDING_CORRECTION",
        "defect": (
            "FULL_BANK_V2_REPLAY_CALLED_GENERIC_COMPACT_SELF_DIGEST_"
            "VALIDATION_FOR_DEFAULT_CANONICAL_BUILDER_PAYLOADS"),
        "correction": (
            "VALIDATE_ONLY_THE_FIVE_FULL_BANK_V2_SELF_DIGESTS_WITH_THE_"
            "BUILDER_DEFAULT_JSON_DUMPS_SORT_KEYS_CANONICALIZATION"),
        "historical_source_repository_commit":
            IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT,
        "successor_source_repository_commit": source_repository_commit,
        "allowed_changed_source_paths": list(
            MANIFEST_REPLAY_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_source_binding_set_digest": active[
            "source_binding_set_digest"],
        "successor_source_binding_set_digest": canonical_digest(
            current_sources),
        "full_bank_v2_self_digest_canonicalization":
            "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "generic_parallel_search_binding_changed": False,
        "installed_manifest_payload_or_digest_changed": False,
        "installed_manifest_overwritten_deleted_or_reissued": False,
        "scientific_manifest_lineage_digest_preserved":
            IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST,
        "scene_state_assignment_or_candidate_changed": False,
        "selector_exclusion_rule_or_quota_changed": False,
        "scientific_oracle_scorer_or_qualification_criteria_changed": False,
        "candidate_outcome_or_downstream_metric_used": False,
    }
    payload: dict[str, Any] = {
        "schema": MANIFEST_REPLAY_CORRECTION_SCHEMA,
        "status": MANIFEST_REPLAY_CORRECTION_STATUS,
        "complete": True,
        "manifest_replay_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT,
        "immutable_active_preselection_source_correction": copy.deepcopy(
            immutable),
        "immutable_active_preselection_source_correction_digest":
            IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST,
        "preserved_scientific_design_digest": active[
            "preserved_scientific_design_digest"],
        "preserved_scientific_manifest_lineage_digest":
            IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST,
        "installed_preoutcome_artifact_bindings": installed,
        "installed_preoutcome_artifact_binding_set_digest": canonical_digest(
            installed),
        "successor_and_runtime_outputs_absent_at_issue": absence,
        "successor_and_runtime_outputs_absent_at_issue_digest":
            canonical_digest(absence),
        "manifest_replay_failure_boundary": failure,
        "manifest_replay_failure_boundary_digest": canonical_digest(failure),
        "manifest_replay_correction": correction_material,
        "manifest_replay_correction_material_digest": canonical_digest(
            correction_material),
        "issuance_boundary": {
            "active_preselection_source_correction_preserved": True,
            "all_five_installed_artifacts_preserved_exact": True,
            "source_tree_clean_and_committed": True,
            "failure_preserved_truthfully": True,
            "double_successor_and_runtime_absence_audit_required": True,
            "operational_replay_only": True,
            "manifest_lineage_replaced": False,
            "installed_artifact_written_or_rewritten": False,
            "successor_scorer_contract_issued": False,
            "branch_execution_started": False,
            "candidate_outcomes_consumed": False,
            "solver_or_optimisation_invoked": False,
        },
    }
    if (set(payload)
            != _MANIFEST_REPLAY_CORRECTION_KEYS
            - {MANIFEST_REPLAY_CORRECTION_SELF_KEY}):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction construction surface changed")
    payload[MANIFEST_REPLAY_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_manifest_replay_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_successor_and_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _MANIFEST_REPLAY_CORRECTION_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != MANIFEST_REPLAY_CORRECTION_SCHEMA
            or correction.get("status") != MANIFEST_REPLAY_CORRECTION_STATUS
            or correction.get("complete") is not True
            or correction.get("manifest_replay_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction version changed")
    expected = build_manifest_replay_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_active_preselection_source_correction=correction.get(
            "immutable_active_preselection_source_correction", {}),
        installed_preoutcome_artifact_bindings=correction.get(
            "installed_preoutcome_artifact_bindings", []),
        successor_and_runtime_outputs_absent_at_issue=correction.get(
            "successor_and_runtime_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(MANIFEST_REPLAY_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, MANIFEST_REPLAY_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or _load_immutable_active_preselection_source_correction(
                    root=root)
                != correction[
                    "immutable_active_preselection_source_correction"]
                or validate_installed_full_bank_v2_preoutcome_artifacts(
                    root=root)
                != correction["installed_preoutcome_artifact_bindings"]):
            raise ScorerFitCorpusV2DesignError(
                "live source, manifest lineage, or installed artifacts differ "
                "from manifest-replay correction")
        if require_successor_and_runtime_outputs_absent:
            observed = audit_v2_runtime_outputs_absent(
                root=root, phase="successor_contract")
            if observed != correction[
                    "successor_and_runtime_outputs_absent_at_issue"]:
                raise ScorerFitCorpusV2DesignError(
                    "successor/runtime absence changed after replay correction")
    return correction


def manifest_replay_correction_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_manifest_replay_correction(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction raw bytes changed")
    return {
        "path": str(MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH),
        "schema": MANIFEST_REPLAY_CORRECTION_SCHEMA,
        "self_digest_key": MANIFEST_REPLAY_CORRECTION_SELF_KEY,
        "self_digest": correction[MANIFEST_REPLAY_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


_IMMUTABLE_MANIFEST_REPLAY_CORRECTION_KEYS = frozenset({"payload", "binding"})


def validate_immutable_manifest_replay_correction(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact 72b0 replay correction without live-source equality."""

    if (not isinstance(value, Mapping)
            or set(value) != _IMMUTABLE_MANIFEST_REPLAY_CORRECTION_KEYS
            or not isinstance(value.get("payload"), Mapping)
            or not isinstance(value.get("binding"), Mapping)):
        raise ScorerFitCorpusV2DesignError(
            "immutable manifest-replay correction is not closed")
    immutable = copy.deepcopy(dict(value))
    payload = validate_manifest_replay_correction(
        immutable["payload"], validate_live_authorities=False)
    expected_binding = manifest_replay_correction_artifact_binding(
        payload, _pretty_json_bytes(payload))
    if (immutable["binding"] != expected_binding
            or payload.get(MANIFEST_REPLAY_CORRECTION_SELF_KEY)
            != IMMUTABLE_MANIFEST_REPLAY_CORRECTION_DIGEST
            or payload.get("source_repository_commit")
            != ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "immutable manifest-replay correction changed")
    immutable["payload"] = payload
    immutable["binding"] = expected_binding
    return immutable


def _validate_immutable_successor_scorer_contract_binding(
        value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or dict(value) != IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING):
        raise ScorerFitCorpusV2DesignError(
            "immutable successor scorer contract binding changed")
    return copy.deepcopy(IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING)


def _validate_failure_runtime_binding(
        value: Any, *, expected: Mapping[str, Any], label: str,
        ) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise ScorerFitCorpusV2DesignError(f"immutable {label} binding changed")
    return copy.deepcopy(dict(expected))


def _expected_encoder_import_correction_absence_rows() -> list[dict[str, Any]]:
    rows = [
        {"path": str(path), "expected_kind": "file", "exists": False}
        for path in ENCODER_IMPORT_CORRECTION_REQUIRED_ABSENT_PATHS
    ]
    rows.extend({
        "path": str(path), "expected_kind": "directory", "exists": False,
    } for path in ENCODER_IMPORT_CORRECTION_REQUIRED_ABSENT_DIRECTORIES)
    return rows


def _validate_encoder_import_correction_absence(value: Any) -> list[dict[str, Any]]:
    expected = _expected_encoder_import_correction_absence_rows()
    if not isinstance(value, list) or value != expected:
        raise ScorerFitCorpusV2DesignError(
            "post-smoke pre-latent absence projection changed")
    return copy.deepcopy(expected)


def _expected_encoder_compute_dtype_correction_absence_rows(
        ) -> list[dict[str, Any]]:
    rows = [
        {"path": str(path), "expected_kind": "file", "exists": False}
        for path in ENCODER_COMPUTE_DTYPE_CORRECTION_REQUIRED_ABSENT_PATHS
    ]
    rows.extend({
        "path": str(path), "expected_kind": "directory", "exists": False,
    } for path in ENCODER_COMPUTE_DTYPE_CORRECTION_REQUIRED_ABSENT_DIRECTORIES)
    return rows


def _validate_encoder_compute_dtype_correction_absence(
        value: Any) -> list[dict[str, Any]]:
    expected = _expected_encoder_compute_dtype_correction_absence_rows()
    if not isinstance(value, list) or value != expected:
        raise ScorerFitCorpusV2DesignError(
            "post-import pre-latent dtype absence projection changed")
    return copy.deepcopy(expected)


def _validate_encoder_compute_dtype_focused_test_transitions(
        value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(
            ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_SPECS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype focused test transition coverage changed")
    transitions = [
        _validate_source_transition(row, path=path, role=role)
        for (path, role), row in zip(
            ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_SPECS,
            value, strict=True)
    ]
    if [row["path"] for row in transitions] != list(
            ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype focused test transition order changed")
    return transitions


def _expected_encoder_path_projection_correction_absence_rows(
        ) -> list[dict[str, Any]]:
    rows = [
        {"path": str(path), "expected_kind": "file", "exists": False}
        for path in ENCODER_PATH_PROJECTION_CORRECTION_REQUIRED_ABSENT_PATHS
    ]
    rows.extend({
        "path": str(path), "expected_kind": "directory", "exists": False,
    } for path in ENCODER_PATH_PROJECTION_CORRECTION_REQUIRED_ABSENT_DIRECTORIES)
    return rows


def _validate_encoder_path_projection_correction_absence(
        value: Any) -> list[dict[str, Any]]:
    expected = _expected_encoder_path_projection_correction_absence_rows()
    if not isinstance(value, list) or value != expected:
        raise ScorerFitCorpusV2DesignError(
            "post-base-smoke path-projection absence projection changed")
    return copy.deepcopy(expected)


def _expected_encoder_path_projection_transaction_absence_rows(
        ) -> list[dict[str, Any]]:
    rows = [
        {"path": str(path), "expected_kind": "file", "exists": False}
        for path in ENCODER_PATH_PROJECTION_TRANSACTION_REQUIRED_ABSENT_PATHS
    ]
    rows.extend({
        "path": str(path), "expected_kind": "directory", "exists": False,
    } for path in
        ENCODER_PATH_PROJECTION_TRANSACTION_REQUIRED_ABSENT_DIRECTORIES)
    return rows


def _validate_encoder_path_projection_transaction_absence(
        value: Any) -> list[dict[str, Any]]:
    expected = _expected_encoder_path_projection_transaction_absence_rows()
    if not isinstance(value, list) or value != expected:
        raise ScorerFitCorpusV2DesignError(
            "pre-transaction path-projection absence projection changed")
    return copy.deepcopy(expected)


def _expected_branch_redrive_projection_correction_absence_rows(
        ) -> list[dict[str, Any]]:
    rows = [
        {"path": str(path), "expected_kind": "file", "exists": False}
        for path in BRANCH_REDRIVE_PROJECTION_CORRECTION_REQUIRED_ABSENT_PATHS
    ]
    rows.extend({
        "path": str(path), "expected_kind": "directory", "exists": False,
    } for path in
        BRANCH_REDRIVE_PROJECTION_CORRECTION_REQUIRED_ABSENT_DIRECTORIES)
    return rows


def _validate_branch_redrive_projection_correction_absence(
        value: Any) -> list[dict[str, Any]]:
    expected = _expected_branch_redrive_projection_correction_absence_rows()
    if not isinstance(value, list) or value != expected:
        raise ScorerFitCorpusV2DesignError(
            "post-partial-corpus downstream absence projection changed")
    return copy.deepcopy(expected)


def _validate_encoder_path_projection_focused_test_transitions(
        value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(
            ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_SPECS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection focused test transition coverage changed")
    transitions = [
        _validate_source_transition(row, path=path, role=role)
        for (path, role), row in zip(
            ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_SPECS,
            value, strict=True)
    ]
    if [row["path"] for row in transitions] != list(
            ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection focused test transition order changed")
    return transitions


_FULL_BANK_V2_SMOKE_REGENERATION_LINEAGE_KEYS = frozenset({
    "scorer_fit_corpus_v2_scorer_contract_digest",
    "scorer_fit_corpus_v2_scorer_contract_artifact_digest",
    "state_manifest_digest", "full_bank_assignment_manifest_digest",
    "corpus_digest", "branch_smoke_receipt_digest",
    "encoder_compute_dtype_correction_digest",
    "encoder_path_projection_correction_digest",
})
_FULL_BANK_V2_SMOKE_REGENERATION_TARGET_KEYS = frozenset({
    "path", "candidate_index", "sha256", "byte_count", "shape",
    "device_id", "inode", "mode_octal", "link_count",
})
_FULL_BANK_V2_SMOKE_REGENERATION_PRE_EVIDENCE_KEYS = frozenset({
    "latent_index_digest", "encoding_smoke_receipt_digest",
    "registered_smoke_shard_inventory_digest",
    "registered_smoke_non_target_shard_inventory_digest",
    "registered_smoke_non_target_shard_custody_inventory_digest",
    "registered_smoke_stable_artifact_inventory_digest",
    "zero_new_resume_verified",
})
_FULL_BANK_V2_SMOKE_REGENERATION_POST_EVIDENCE_KEYS = frozenset({
    "latent_index_digest", "encoding_smoke_receipt_digest",
    "registered_smoke_shard_inventory_digest",
    "registered_smoke_non_target_shard_custody_inventory_digest",
    "registered_smoke_stable_artifact_inventory_digest",
    "encoder_invocation_new_context_shards",
    "encoder_invocation_new_horizon_shards", "target_restored_exact",
    "non_target_shards_unchanged", "complete_before_pass_smoke",
})
_FULL_BANK_V2_SMOKE_REGENERATION_ARTIFACT_BINDING_KEYS = frozenset({
    "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
    "byte_count",
})
_FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_BINDING_KEYS = frozenset({
    *_FULL_BANK_V2_SMOKE_REGENERATION_ARTIFACT_BINDING_KEYS,
    "designated_target_digest", "prepared_lineage_digest",
    "pretransaction_registered_smoke_shard_inventory_digest",
    "pretransaction_registered_smoke_non_target_shard_inventory_digest",
    "pretransaction_registered_smoke_non_target_shard_custody_inventory_"
    "digest",
    "pretransaction_registered_smoke_stable_artifact_inventory_digest",
})
_FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_BINDING_KEYS = frozenset({
    "path", "sha256", "byte_count", "shape", "device_id", "inode",
    "mode_octal", "link_count",
})
_FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_KEYS = frozenset({
    "schema", "status", "complete", "transaction_version",
    "single_shard_regeneration_transaction_contract_digest", "lineage",
    "designated_target", "expected_backup_binding",
    "pretransaction_evidence", "receipt_publication_contract",
    "candidate_outcome_or_label_used", "latent_value_deserialized",
    "target_move_started_before_prepared_publication",
    FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY,
})
_FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_KEYS = frozenset({
    "schema", "status", "complete", "transaction_version",
    "single_shard_regeneration_transaction_contract_digest",
    "prepared_receipt_binding", "lineage", "designated_target",
    "retained_backup_binding", "regenerated_target_binding",
    "non_target_shard_inventory_digest", "posttransaction_evidence",
    "final_smoke_receipt_binding", "receipt_publication_contract",
    "candidate_outcome_or_label_used", "latent_value_deserialized",
    "second_target_move_or_regeneration_performed",
    FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY,
})


def _validate_full_bank_v2_smoke_regeneration_lineage(
        value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) != _FULL_BANK_V2_SMOKE_REGENERATION_LINEAGE_KEYS
            or not all(_is_hex(value.get(key), 64) for key in value)):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration lineage changed")
    lineage = copy.deepcopy(dict(value))
    if (lineage["encoder_compute_dtype_correction_digest"]
            != IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST
            or lineage[
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest"]
            != IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING["self_digest"]
            or lineage["scorer_fit_corpus_v2_scorer_contract_digest"]
            != IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING[
                "embedded_contract_self_digest"]):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration immutable lineage changed")
    return lineage


def _validate_full_bank_v2_smoke_regeneration_target(
        value: Any) -> dict[str, Any]:
    mode = value.get("mode_octal") if isinstance(value, Mapping) else None
    if (not isinstance(value, Mapping)
            or set(value) != _FULL_BANK_V2_SMOKE_REGENERATION_TARGET_KEYS
            or value.get("candidate_index") != 0
            or not _is_hex(value.get("sha256"), 64)
            or isinstance(value.get("byte_count"), bool)
            or value.get("byte_count") != 6_291_456
            or value.get("shape") != [4, 768, 1024]
            or any(isinstance(value.get(key), bool)
                   or not isinstance(value.get(key), int)
                   or value[key] <= 0 for key in (
                       "device_id", "inode", "link_count"))
            or value.get("link_count") != 1
            or not isinstance(mode, str) or len(mode) != 4
            or mode[0] != "0" or any(character not in "01234567"
                                      for character in mode)):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration target changed")
    path = Path(str(value.get("path", "")))
    if (path.is_absolute()
            or path.parent != SCORER_FIT_RELATIVE_PATH / "latents_v2/horizon"
            or path.suffix != ".f16"):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration target path changed")
    return copy.deepcopy(dict(value))


def _validate_full_bank_v2_smoke_regeneration_pre_evidence(
        value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) !=
            _FULL_BANK_V2_SMOKE_REGENERATION_PRE_EVIDENCE_KEYS
            or value.get("zero_new_resume_verified") is not True
            or not all(_is_hex(value.get(key), 64) for key in value
                       if key != "zero_new_resume_verified")):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration pretransaction evidence changed")
    return copy.deepcopy(dict(value))


def _validate_full_bank_v2_smoke_regeneration_post_evidence(
        value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) !=
            _FULL_BANK_V2_SMOKE_REGENERATION_POST_EVIDENCE_KEYS
            or not all(_is_hex(value.get(key), 64) for key in (
                "latent_index_digest", "encoding_smoke_receipt_digest",
                "registered_smoke_shard_inventory_digest",
                "registered_smoke_non_target_shard_custody_inventory_digest",
                "registered_smoke_stable_artifact_inventory_digest"))
            or value.get("encoder_invocation_new_context_shards") != 0
            or value.get("encoder_invocation_new_horizon_shards") not in {0, 1}
            or value.get("target_restored_exact") is not True
            or value.get("non_target_shards_unchanged") is not True
            or value.get("complete_before_pass_smoke") is not True):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration posttransaction evidence changed")
    return copy.deepcopy(dict(value))


def _validate_full_bank_v2_smoke_regeneration_binding(
        value: Any, *, expected_path: Path, expected_schema: str,
        expected_self_key: str, label: str) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) !=
            _FULL_BANK_V2_SMOKE_REGENERATION_ARTIFACT_BINDING_KEYS
            or value.get("path") != str(expected_path)
            or value.get("schema") != expected_schema
            or value.get("self_digest_key") != expected_self_key
            or not _is_hex(value.get("self_digest"), 64)
            or not _is_hex(value.get("raw_sha256"), 64)
            or isinstance(value.get("byte_count"), bool)
            or not isinstance(value.get("byte_count"), int)
            or value["byte_count"] <= 0):
        raise ScorerFitCorpusV2DesignError(f"{label} binding changed")
    return copy.deepcopy(dict(value))


def _validate_full_bank_v2_smoke_regeneration_prepared_binding(
        value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) !=
            _FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_BINDING_KEYS
            or value.get("path") != str(
                FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH)
            or value.get("schema")
            != FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SCHEMA
            or value.get("self_digest_key")
            != FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY
            or not all(_is_hex(value.get(key), 64) for key in (
                "self_digest", "raw_sha256", "designated_target_digest",
                "prepared_lineage_digest",
                "pretransaction_registered_smoke_shard_inventory_digest",
                "pretransaction_registered_smoke_non_target_shard_inventory_"
                "digest",
                "pretransaction_registered_smoke_non_target_shard_custody_"
                "inventory_digest",
                "pretransaction_registered_smoke_stable_artifact_inventory_"
                "digest"))
            or isinstance(value.get("byte_count"), bool)
            or not isinstance(value.get("byte_count"), int)
            or value["byte_count"] <= 0):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration PREPARED binding changed")
    return copy.deepcopy(dict(value))


def build_full_bank_v2_smoke_regeneration_prepared_receipt(
        *, lineage: Mapping[str, Any], designated_target: Mapping[str, Any],
        pretransaction_evidence: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the immutable intent receipt before the target can move."""

    closed_lineage = _validate_full_bank_v2_smoke_regeneration_lineage(lineage)
    target = _validate_full_bank_v2_smoke_regeneration_target(
        designated_target)
    evidence = _validate_full_bank_v2_smoke_regeneration_pre_evidence(
        pretransaction_evidence)
    contract = ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT
    expected_backup = {
        "path": str(FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_RELATIVE_PATH),
        "sha256": target["sha256"],
        "byte_count": target["byte_count"],
        "shape": target["shape"],
        "device_id": target["device_id"],
        "inode": target["inode"],
        "mode_octal": target["mode_octal"],
        "link_count": target["link_count"],
    }
    payload: dict[str, Any] = {
        "schema": FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SCHEMA,
        "status": FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STATUS,
        "complete": False,
        "transaction_version": 1,
        "single_shard_regeneration_transaction_contract_digest":
            canonical_digest(contract),
        "lineage": closed_lineage,
        "designated_target": target,
        "expected_backup_binding": expected_backup,
        "pretransaction_evidence": evidence,
        "receipt_publication_contract": copy.deepcopy(
            contract["immutable_receipt_publication"]),
        "candidate_outcome_or_label_used": False,
        "latent_value_deserialized": False,
        "target_move_started_before_prepared_publication": False,
    }
    if set(payload) != _FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_KEYS - {
            FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration PREPARED construction changed")
    payload[FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY] = (
        canonical_digest(payload))
    return payload


def validate_full_bank_v2_smoke_regeneration_prepared_receipt(
        payload: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration PREPARED receipt is not closed")
    receipt = copy.deepcopy(dict(payload))
    expected = build_full_bank_v2_smoke_regeneration_prepared_receipt(
        lineage=receipt.get("lineage", {}),
        designated_target=receipt.get("designated_target", {}),
        pretransaction_evidence=receipt.get("pretransaction_evidence", {}),
    )
    if receipt != expected:
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration PREPARED receipt changed")
    return receipt


def full_bank_v2_smoke_regeneration_prepared_receipt_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    receipt = validate_full_bank_v2_smoke_regeneration_prepared_receipt(
        payload)
    if raw != _pretty_json_bytes(receipt):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration PREPARED raw bytes changed")
    return {
        "path": str(FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH),
        "schema": FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SCHEMA,
        "self_digest_key": FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY,
        "self_digest": receipt[
            FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "designated_target_digest": canonical_digest(
            receipt["designated_target"]),
        "prepared_lineage_digest": canonical_digest(receipt["lineage"]),
        "pretransaction_registered_smoke_shard_inventory_digest":
            receipt["pretransaction_evidence"][
                "registered_smoke_shard_inventory_digest"],
        "pretransaction_registered_smoke_non_target_shard_inventory_digest":
            receipt["pretransaction_evidence"][
                "registered_smoke_non_target_shard_inventory_digest"],
        "pretransaction_registered_smoke_non_target_shard_custody_inventory_"
        "digest": receipt["pretransaction_evidence"][
            "registered_smoke_non_target_shard_custody_inventory_digest"],
        "pretransaction_registered_smoke_stable_artifact_inventory_digest":
            receipt["pretransaction_evidence"][
                "registered_smoke_stable_artifact_inventory_digest"],
    }


def build_full_bank_v2_smoke_regeneration_complete_receipt(
        *, prepared_receipt_binding: Mapping[str, Any],
        lineage: Mapping[str, Any], designated_target: Mapping[str, Any],
        retained_backup_binding: Mapping[str, Any],
        regenerated_target_binding: Mapping[str, Any],
        non_target_shard_inventory_digest: str,
        posttransaction_evidence: Mapping[str, Any],
        final_smoke_receipt_binding: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build completion evidence before the PASS smoke can be published."""

    prepared = _validate_full_bank_v2_smoke_regeneration_prepared_binding(
        prepared_receipt_binding)
    closed_lineage = _validate_full_bank_v2_smoke_regeneration_lineage(lineage)
    if prepared.get("prepared_lineage_digest") != canonical_digest(
            closed_lineage):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE changed PREPARED lineage")
    target = _validate_full_bank_v2_smoke_regeneration_target(
        designated_target)
    if prepared["designated_target_digest"] != canonical_digest(target):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE target changed from PREPARED")
    if (not isinstance(retained_backup_binding, Mapping)
            or set(retained_backup_binding) !=
            _FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_BINDING_KEYS
            or retained_backup_binding.get("path") != str(
                FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_RELATIVE_PATH)
            or retained_backup_binding.get("sha256") != target["sha256"]
            or retained_backup_binding.get("byte_count") != target["byte_count"]
            or retained_backup_binding.get("shape") != target["shape"]
            or retained_backup_binding.get("device_id") != target["device_id"]
            or retained_backup_binding.get("inode") != target["inode"]
            or retained_backup_binding.get("mode_octal")
            != target["mode_octal"]
            or retained_backup_binding.get("link_count")
            != target["link_count"]):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration retained backup changed")
    regenerated = _validate_full_bank_v2_smoke_regeneration_target(
        regenerated_target_binding)
    semantic_keys = {"path", "candidate_index", "sha256", "byte_count", "shape"}
    if (any(regenerated[key] != target[key] for key in semantic_keys)
            or regenerated["device_id"] != target["device_id"]
            or regenerated["inode"] == target["inode"]
            or regenerated["mode_octal"] != target["mode_octal"]
            or regenerated["link_count"] != target["link_count"]):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration did not restore the exact target")
    if not _is_hex(non_target_shard_inventory_digest, 64):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration non-target inventory changed")
    if non_target_shard_inventory_digest != prepared[
            "pretransaction_registered_smoke_non_target_shard_inventory_digest"]:
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration changed a non-target shard")
    post = _validate_full_bank_v2_smoke_regeneration_post_evidence(
        posttransaction_evidence)
    if (post["registered_smoke_shard_inventory_digest"] != prepared[
            "pretransaction_registered_smoke_shard_inventory_digest"]
            or post[
                "registered_smoke_non_target_shard_custody_inventory_digest"]
            != prepared[
                "pretransaction_registered_smoke_non_target_shard_custody_"
                "inventory_digest"]
            or post["registered_smoke_stable_artifact_inventory_digest"]
            != prepared[
                "pretransaction_registered_smoke_stable_artifact_inventory_"
                "digest"]):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration changed a registered stable artifact")
    smoke_binding = _validate_full_bank_v2_smoke_regeneration_binding(
        final_smoke_receipt_binding,
        expected_path=SCORER_FIT_RELATIVE_PATH /
            "smoke_encoding_receipt_v2.json",
        expected_schema=(
            "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1"),
        expected_self_key="smoke_receipt_digest",
        label="single-shard regeneration final smoke")
    if (post["encoding_smoke_receipt_digest"]
            != smoke_binding["self_digest"]):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE binds contradictory smoke "
            "digests")
    contract = ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT
    payload: dict[str, Any] = {
        "schema": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SCHEMA,
        "status": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STATUS,
        "complete": True,
        "transaction_version": 1,
        "single_shard_regeneration_transaction_contract_digest":
            canonical_digest(contract),
        "prepared_receipt_binding": prepared,
        "lineage": closed_lineage,
        "designated_target": target,
        "retained_backup_binding": copy.deepcopy(
            dict(retained_backup_binding)),
        "regenerated_target_binding": regenerated,
        "non_target_shard_inventory_digest":
            non_target_shard_inventory_digest,
        "posttransaction_evidence": post,
        "final_smoke_receipt_binding": smoke_binding,
        "receipt_publication_contract": copy.deepcopy(
            contract["immutable_receipt_publication"]),
        "candidate_outcome_or_label_used": False,
        "latent_value_deserialized": False,
        "second_target_move_or_regeneration_performed": False,
    }
    if set(payload) != _FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_KEYS - {
            FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE construction changed")
    payload[FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY] = (
        canonical_digest(payload))
    return payload


def validate_full_bank_v2_smoke_regeneration_complete_receipt(
        payload: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE receipt is not closed")
    receipt = copy.deepcopy(dict(payload))
    expected = build_full_bank_v2_smoke_regeneration_complete_receipt(
        prepared_receipt_binding=receipt.get("prepared_receipt_binding", {}),
        lineage=receipt.get("lineage", {}),
        designated_target=receipt.get("designated_target", {}),
        retained_backup_binding=receipt.get("retained_backup_binding", {}),
        regenerated_target_binding=receipt.get(
            "regenerated_target_binding", {}),
        non_target_shard_inventory_digest=str(receipt.get(
            "non_target_shard_inventory_digest", "")),
        posttransaction_evidence=receipt.get("posttransaction_evidence", {}),
        final_smoke_receipt_binding=receipt.get(
            "final_smoke_receipt_binding", {}),
    )
    if receipt != expected:
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE receipt changed")
    return receipt


def full_bank_v2_smoke_regeneration_complete_receipt_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    receipt = validate_full_bank_v2_smoke_regeneration_complete_receipt(
        payload)
    if raw != _pretty_json_bytes(receipt):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE raw bytes changed")
    return {
        "path": str(FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH),
        "schema": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SCHEMA,
        "self_digest_key": FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY,
        "self_digest": receipt[
            FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def validate_encoder_path_projection_single_shard_regeneration_transaction_contract(
        value: Mapping[str, Any]) -> dict[str, Any]:
    expected = (
        ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration transaction contract changed")
    return copy.deepcopy(expected)


def load_full_bank_v2_smoke_regeneration_prepared_receipt(
        *, root: Path = ROOT) -> dict[str, Any]:
    path = _pin_generated(
        root, FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH,
        label="single-shard regeneration PREPARED receipt")
    if (not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration PREPARED receipt mode changed")
    payload, raw = _load_json(
        path, label="single-shard regeneration PREPARED receipt")
    receipt = validate_full_bank_v2_smoke_regeneration_prepared_receipt(
        payload)
    full_bank_v2_smoke_regeneration_prepared_receipt_artifact_binding(
        receipt, raw)
    return receipt


def load_full_bank_v2_smoke_regeneration_complete_receipt(
        *, root: Path = ROOT) -> dict[str, Any]:
    path = _pin_generated(
        root, FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH,
        label="single-shard regeneration COMPLETE receipt")
    if (not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "single-shard regeneration COMPLETE receipt mode changed")
    payload, raw = _load_json(
        path, label="single-shard regeneration COMPLETE receipt")
    receipt = validate_full_bank_v2_smoke_regeneration_complete_receipt(
        payload)
    full_bank_v2_smoke_regeneration_complete_receipt_artifact_binding(
        receipt, raw)
    return receipt


_ENCODER_IMPORT_CORRECTION_KEYS = frozenset({
    "schema", "status", "complete", "encoder_import_correction_version",
    "source_repository_commit", "source_bindings", "source_binding_set_digest",
    "historical_source_repository_commit",
    "immutable_manifest_replay_correction",
    "immutable_manifest_replay_correction_digest",
    "immutable_successor_scorer_contract_binding",
    "immutable_branch_smoke_binding",
    "immutable_partial_corpus_receipt_binding",
    "dev_encoder_source_transition", "focused_test_source_transitions",
    "production_source_transition", "production_source_transition_digest",
    "focused_test_source_transition_digest", "preserved_scientific_contract",
    "preserved_scientific_contract_digest",
    "prelatent_outputs_absent_at_issue",
    "prelatent_outputs_absent_at_issue_digest",
    "encoder_import_failure_boundary", "encoder_import_failure_boundary_digest",
    "encoder_import_correction", "encoder_import_correction_material_digest",
    "issuance_boundary", ENCODER_IMPORT_CORRECTION_SELF_KEY,
})


def build_encoder_import_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_manifest_replay_correction: Mapping[str, Any],
        immutable_successor_scorer_contract_binding: Mapping[str, Any],
        dev_encoder_source_transition: Mapping[str, Any],
        focused_test_source_transitions: Sequence[Mapping[str, Any]],
        branch_smoke_binding: Mapping[str, Any],
        branch_corpus_binding: Mapping[str, Any],
        prelatent_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the post-smoke import shim authority without reading outcomes."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction source commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    immutable_replay = validate_immutable_manifest_replay_correction(
        immutable_manifest_replay_correction)
    historical_sources = immutable_replay["payload"]["source_bindings"]
    changed_base = _changed_source_paths(historical_sources, current_sources)
    expected_base = sorted(set(
        ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS
    ).intersection(EXPECTED_SOURCE_PATHS))
    if changed_base != expected_base:
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction changed an unauthorised bound source path")
    dev_transition = _validate_dev_encoder_source_transition(
        dev_encoder_source_transition)
    test_transitions = _validate_focused_test_source_transitions(
        list(focused_test_source_transitions))
    observed_production = sorted(changed_base + [dev_transition["path"]])
    if observed_production != sorted(
            ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import production source transition coverage changed")
    successor_contract = _validate_immutable_successor_scorer_contract_binding(
        immutable_successor_scorer_contract_binding)
    smoke_binding = _validate_failure_runtime_binding(
        branch_smoke_binding,
        expected=IMMUTABLE_ENCODER_IMPORT_FAILURE_BRANCH_SMOKE_BINDING,
        label="branch smoke")
    corpus_binding = _validate_failure_runtime_binding(
        branch_corpus_binding,
        expected=IMMUTABLE_ENCODER_IMPORT_FAILURE_CORPUS_RECEIPT_BINDING,
        label="partial corpus receipt")
    absence = _validate_encoder_import_correction_absence(
        list(prelatent_outputs_absent_at_issue))
    failure = copy.deepcopy(ENCODER_IMPORT_FAILURE_BOUNDARY)
    science = copy.deepcopy(ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE)
    production_transition = {
        "allowed_changed_source_paths": list(
            ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": observed_production,
        "historical_bound_source_paths": [
            row["path"] for row in historical_sources],
        "current_bound_source_paths": [row["path"] for row in current_sources],
        "base_closure_changed_source_paths": changed_base,
        "dev_encoder_source_transition": copy.deepcopy(dev_transition),
        "extra_production_source_path_changed": False,
    }
    correction_material = {
        "status": "SOURCE_ONLY_ENCODER_IMPORT_COMPATIBILITY_CORRECTION",
        "defect": (
            "BOUND_ROCM_ENVIRONMENT_LACKED_UNPINNED_TIMM_IMPORT_DEPENDENCY_"
            "REQUIRED_ONLY_FOR_VJEPA_DROP_PATH_SYMBOL"),
        "correction": (
            "SCOPED_IN_PROCESS_TIMM_MODELS_LAYERS_DROP_PATH_COMPATIBILITY_SHIM_"
            "AROUND_VJEPA_BACKBONES_IMPORT_AND_CONSTRUCTOR_ONLY"),
        "shim_symbols": ["drop_path", "DropPath"],
        "shim_formula": "TIMM_PER_SAMPLE_STOCHASTIC_DEPTH_FORMULA",
        "inference_semantics": (
            "EVAL_MODE_TRAINING_FALSE_RETURNS_INPUT_EXACTLY"),
        "sys_modules_scope_restored_on_exit": True,
        "checkpoint_sha256_identity_verification_changed": False,
        "checkpoint_torch_load_or_state_dict_changed": False,
        "target_encoder_constructor_architecture_or_weights_changed": False,
        "preprocessing_normalisation_or_target_encoding_changed": False,
        "corpus_manifest_branch_or_label_changed": False,
        "scorer_architecture_training_budget_or_qualification_changed": False,
        "runtime_venv_package_installed_or_mutated": False,
        "branch_outcome_or_label_value_used": False,
        "resume_scope": "VALID_REGISTERED_SMOKE_BRANCHES_THEN_MISSING_LATENTS_ONLY",
    }
    payload: dict[str, Any] = {
        "schema": ENCODER_IMPORT_CORRECTION_SCHEMA,
        "status": ENCODER_IMPORT_CORRECTION_STATUS,
        "complete": True,
        "encoder_import_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
        "immutable_manifest_replay_correction": copy.deepcopy(immutable_replay),
        "immutable_manifest_replay_correction_digest":
            IMMUTABLE_MANIFEST_REPLAY_CORRECTION_DIGEST,
        "immutable_successor_scorer_contract_binding": successor_contract,
        "immutable_branch_smoke_binding": smoke_binding,
        "immutable_partial_corpus_receipt_binding": corpus_binding,
        "dev_encoder_source_transition": dev_transition,
        "focused_test_source_transitions": test_transitions,
        "production_source_transition": production_transition,
        "production_source_transition_digest": canonical_digest(
            production_transition),
        "focused_test_source_transition_digest": canonical_digest(
            test_transitions),
        "preserved_scientific_contract": science,
        "preserved_scientific_contract_digest": canonical_digest(science),
        "prelatent_outputs_absent_at_issue": absence,
        "prelatent_outputs_absent_at_issue_digest": canonical_digest(absence),
        "encoder_import_failure_boundary": failure,
        "encoder_import_failure_boundary_digest": canonical_digest(failure),
        "encoder_import_correction": correction_material,
        "encoder_import_correction_material_digest": canonical_digest(
            correction_material),
        "issuance_boundary": {
            "immutable_design_source_and_replay_corrections_preserved": True,
            "immutable_successor_scorer_contract_preserved": True,
            "source_tree_clean_and_committed": True,
            "exact_branch_smoke_and_partial_corpus_metadata_validated": True,
            "branch_outcome_or_label_value_read_for_correction": False,
            "frame_latent_weight_or_predictor_artifact_read_for_correction": False,
            "double_prelatent_absence_audit_required": True,
            "failure_time_smoke_bindings_may_later_be_refreshed_by_full_corpus": True,
            "later_consumption_requires_failure_time_receipts_live": False,
            "scientific_contract_or_manifest_reissued": False,
            "branch_regenerated_based_on_label": False,
            "final_200_state_corpus_authorised": False,
        },
    }
    if set(payload) != _ENCODER_IMPORT_CORRECTION_KEYS - {
            ENCODER_IMPORT_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction construction surface changed")
    payload[ENCODER_IMPORT_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_encoder_import_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _ENCODER_IMPORT_CORRECTION_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != ENCODER_IMPORT_CORRECTION_SCHEMA
            or correction.get("status") != ENCODER_IMPORT_CORRECTION_STATUS
            or correction.get("complete") is not True
            or correction.get("encoder_import_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction version changed")
    expected = build_encoder_import_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_manifest_replay_correction=correction.get(
            "immutable_manifest_replay_correction", {}),
        immutable_successor_scorer_contract_binding=correction.get(
            "immutable_successor_scorer_contract_binding", {}),
        dev_encoder_source_transition=correction.get(
            "dev_encoder_source_transition", {}),
        focused_test_source_transitions=correction.get(
            "focused_test_source_transitions", []),
        branch_smoke_binding=correction.get("immutable_branch_smoke_binding", {}),
        branch_corpus_binding=correction.get(
            "immutable_partial_corpus_receipt_binding", {}),
        prelatent_outputs_absent_at_issue=correction.get(
            "prelatent_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(ENCODER_IMPORT_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, ENCODER_IMPORT_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        dev_transition = _dev_encoder_source_transition(root=root)
        test_transitions = _focused_test_source_transitions(root=root)
        immutable_replay = _load_immutable_manifest_replay_correction(root=root)
        successor = _load_immutable_successor_scorer_contract_binding(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or dev_transition != correction["dev_encoder_source_transition"]
                or test_transitions
                != correction["focused_test_source_transitions"]
                or immutable_replay
                != correction["immutable_manifest_replay_correction"]
                or successor
                != correction["immutable_successor_scorer_contract_binding"]):
            raise ScorerFitCorpusV2DesignError(
                "live source or immutable correction lineage changed")
        if require_failure_boundary_live:
            smoke, corpus = _validate_live_encoder_import_failure_receipts(root=root)
            absence = audit_encoder_import_correction_prelatent_absence(root=root)
            if (smoke != correction["immutable_branch_smoke_binding"]
                    or corpus
                    != correction["immutable_partial_corpus_receipt_binding"]
                    or absence != correction["prelatent_outputs_absent_at_issue"]):
                raise ScorerFitCorpusV2DesignError(
                    "encoder-import failure-time boundary changed before issue")
    return correction


def encoder_import_correction_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_encoder_import_correction(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction raw bytes changed")
    return {
        "path": str(ENCODER_IMPORT_CORRECTION_RELATIVE_PATH),
        "schema": ENCODER_IMPORT_CORRECTION_SCHEMA,
        "self_digest_key": ENCODER_IMPORT_CORRECTION_SELF_KEY,
        "self_digest": correction[ENCODER_IMPORT_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


_IMMUTABLE_ENCODER_IMPORT_CORRECTION_KEYS = frozenset({"payload", "binding"})


def validate_immutable_encoder_import_correction(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact e41d import correction without live-source equality."""

    if (not isinstance(value, Mapping)
            or set(value) != _IMMUTABLE_ENCODER_IMPORT_CORRECTION_KEYS
            or not isinstance(value.get("payload"), Mapping)
            or not isinstance(value.get("binding"), Mapping)):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-import correction is not closed")
    immutable = copy.deepcopy(dict(value))
    payload = validate_encoder_import_correction(
        immutable["payload"], validate_live_authorities=False)
    expected_binding = encoder_import_correction_artifact_binding(
        payload, _pretty_json_bytes(payload))
    if (immutable["binding"] != expected_binding
            or expected_binding != IMMUTABLE_ENCODER_IMPORT_CORRECTION_BINDING
            or payload.get(ENCODER_IMPORT_CORRECTION_SELF_KEY)
            != IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST
            or payload.get("source_repository_commit")
            != ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-import correction changed")
    immutable["payload"] = payload
    immutable["binding"] = expected_binding
    return immutable


_ENCODER_COMPUTE_DTYPE_CORRECTION_KEYS = frozenset({
    "schema", "status", "complete", "encoder_compute_dtype_correction_version",
    "source_repository_commit", "source_bindings", "source_binding_set_digest",
    "historical_source_repository_commit",
    "immutable_encoder_import_correction",
    "immutable_encoder_import_correction_digest",
    "immutable_successor_scorer_contract_binding",
    "immutable_branch_smoke_binding", "immutable_partial_corpus_receipt_binding",
    "failed_encoder_source_binding", "unchanged_dev_encoder_source_binding",
    "unchanged_stage_a_fp32_source_binding", "upstream_rope_source_binding",
    "focused_test_source_transitions", "production_source_transition",
    "production_source_transition_digest", "focused_test_source_transition_digest",
    "preserved_scientific_contract", "preserved_scientific_contract_digest",
    "prelatent_outputs_absent_at_issue",
    "prelatent_outputs_absent_at_issue_digest",
    "encoder_compute_dtype_failure_boundary",
    "encoder_compute_dtype_failure_boundary_digest",
    "encoder_compute_dtype_correction",
    "encoder_compute_dtype_correction_material_digest", "issuance_boundary",
    ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY,
})


def build_encoder_compute_dtype_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_encoder_import_correction: Mapping[str, Any],
        immutable_successor_scorer_contract_binding: Mapping[str, Any],
        focused_test_source_transitions: Sequence[Mapping[str, Any]],
        branch_smoke_binding: Mapping[str, Any],
        branch_corpus_binding: Mapping[str, Any],
        failed_encoder_source_binding: Mapping[str, Any],
        unchanged_dev_encoder_source_binding: Mapping[str, Any],
        unchanged_stage_a_fp32_source_binding: Mapping[str, Any],
        upstream_rope_source_binding: Mapping[str, Any],
        prelatent_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the chained FP32 fidelity correction without reading outcomes."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction source commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    immutable_import = validate_immutable_encoder_import_correction(
        immutable_encoder_import_correction)
    import_payload = immutable_import["payload"]
    historical_sources = import_payload["source_bindings"]
    changed = _changed_source_paths(historical_sources, current_sources)
    expected_changed = sorted(
        ENCODER_COMPUTE_DTYPE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    if changed != expected_changed:
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction changed an unauthorised source path")
    successor = _validate_immutable_successor_scorer_contract_binding(
        immutable_successor_scorer_contract_binding)
    if import_payload["immutable_successor_scorer_contract_binding"] != successor:
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction scorer-contract chain changed")
    smoke = _validate_failure_runtime_binding(
        branch_smoke_binding,
        expected=IMMUTABLE_ENCODER_IMPORT_FAILURE_BRANCH_SMOKE_BINDING,
        label="dtype-failure branch smoke")
    corpus = _validate_failure_runtime_binding(
        branch_corpus_binding,
        expected=IMMUTABLE_ENCODER_IMPORT_FAILURE_CORPUS_RECEIPT_BINDING,
        label="dtype-failure partial corpus receipt")
    if dict(failed_encoder_source_binding) != (
            ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING):
        raise ScorerFitCorpusV2DesignError(
            "failed encoder source binding changed")
    historical_encoder = next(
        row for row in historical_sources
        if row["path"] == ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING[
            "path"])
    if (historical_encoder["byte_count"]
            != ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING["byte_count"]
            or historical_encoder["sha256"]
            != ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING["sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "failed encoder source disagrees with immutable source closure")
    if dict(unchanged_dev_encoder_source_binding) != (
            ENCODER_COMPUTE_DTYPE_UNCHANGED_DEV_ENCODER_BINDING):
        raise ScorerFitCorpusV2DesignError(
            "unchanged target-encoder source binding changed")
    if dict(unchanged_stage_a_fp32_source_binding) != (
            ENCODER_COMPUTE_DTYPE_STAGE_A_FP32_SOURCE_BINDING):
        raise ScorerFitCorpusV2DesignError(
            "frozen Stage-A FP32 source binding changed")
    if dict(upstream_rope_source_binding) != (
            ENCODER_COMPUTE_DTYPE_UPSTREAM_ROPE_SOURCE_BINDING):
        raise ScorerFitCorpusV2DesignError(
            "frozen upstream RoPE source binding changed")
    tests = _validate_encoder_compute_dtype_focused_test_transitions(
        list(focused_test_source_transitions))
    absence = _validate_encoder_compute_dtype_correction_absence(
        list(prelatent_outputs_absent_at_issue))
    production_transition = {
        "allowed_changed_source_paths": list(
            ENCODER_COMPUTE_DTYPE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_bound_source_paths": [
            row["path"] for row in historical_sources],
        "current_bound_source_paths": [row["path"] for row in current_sources],
        "failed_encoder_source_binding": copy.deepcopy(
            ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING),
        "unchanged_dev_encoder_source_binding": copy.deepcopy(
            ENCODER_COMPUTE_DTYPE_UNCHANGED_DEV_ENCODER_BINDING),
        "extra_production_source_path_changed": False,
    }
    science = copy.deepcopy(ENCODER_COMPUTE_DTYPE_CORRECTION_PRESERVED_SCIENCE)
    failure = copy.deepcopy(ENCODER_COMPUTE_DTYPE_FAILURE_BOUNDARY)
    if (science.get("encoder_import_correction_digest")
            != IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST
            or failure.get("historical_source_repository_commit")
            != ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype immutable science or failure chain changed")
    correction_material = {
        "status": "SOURCE_ONLY_ENCODER_COMPUTE_DTYPE_FIDELITY_CORRECTION",
        "defect": (
            "FULL_BANK_V2_ROUTE_SELECTED_EXPLICIT_BFLOAT16_MODEL_AND_INPUT_"
            "COMPUTE_INSTEAD_OF_FROZEN_STAGE_A_FLOAT32_COMPUTE"),
        "correction": (
            "RESTORE_FULL_BANK_V2_TARGET_ENCODER_COMPUTE_DTYPE_TO_FLOAT32_"
            "IDENTICAL_TO_FROZEN_STAGE_A"),
        "failed_compute_dtype": "bfloat16",
        "corrected_compute_dtype": "float32",
        "latent_storage_dtype": "float16",
        "automatic_mixed_precision_or_autocast_enabled": False,
        "preprocessing_changed": False,
        "target_normalisation_changed": False,
        "target_encoder_architecture_changed": False,
        "target_encoder_checkpoint_changed": False,
        "target_encoder_output_layer_changed": False,
        "scientific_target_encoder_contract_changed": False,
        "runtime_compute_dtype_restored_to_frozen_stage_a": True,
        "strict_state_dict_loading_changed": False,
        "latent_shape_token_order_or_storage_dtype_changed": False,
        "corpus_manifest_branch_frame_or_label_changed": False,
        "scorer_architecture_training_budget_or_qualification_changed": False,
        "runtime_venv_package_installed_or_mutated": False,
        "branch_outcome_or_label_value_used": False,
        "resume_scope": "VALID_REGISTERED_SMOKE_BRANCHES_THEN_MISSING_LATENTS_ONLY",
    }
    payload: dict[str, Any] = {
        "schema": ENCODER_COMPUTE_DTYPE_CORRECTION_SCHEMA,
        "status": ENCODER_COMPUTE_DTYPE_CORRECTION_STATUS,
        "complete": True,
        "encoder_compute_dtype_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
        "immutable_encoder_import_correction": copy.deepcopy(immutable_import),
        "immutable_encoder_import_correction_digest":
            IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST,
        "immutable_successor_scorer_contract_binding": successor,
        "immutable_branch_smoke_binding": smoke,
        "immutable_partial_corpus_receipt_binding": corpus,
        "failed_encoder_source_binding": copy.deepcopy(
            ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING),
        "unchanged_dev_encoder_source_binding": copy.deepcopy(
            ENCODER_COMPUTE_DTYPE_UNCHANGED_DEV_ENCODER_BINDING),
        "unchanged_stage_a_fp32_source_binding": copy.deepcopy(
            ENCODER_COMPUTE_DTYPE_STAGE_A_FP32_SOURCE_BINDING),
        "upstream_rope_source_binding": copy.deepcopy(
            ENCODER_COMPUTE_DTYPE_UPSTREAM_ROPE_SOURCE_BINDING),
        "focused_test_source_transitions": tests,
        "production_source_transition": production_transition,
        "production_source_transition_digest": canonical_digest(
            production_transition),
        "focused_test_source_transition_digest": canonical_digest(tests),
        "preserved_scientific_contract": science,
        "preserved_scientific_contract_digest": canonical_digest(science),
        "prelatent_outputs_absent_at_issue": absence,
        "prelatent_outputs_absent_at_issue_digest": canonical_digest(absence),
        "encoder_compute_dtype_failure_boundary": failure,
        "encoder_compute_dtype_failure_boundary_digest": canonical_digest(failure),
        "encoder_compute_dtype_correction": correction_material,
        "encoder_compute_dtype_correction_material_digest": canonical_digest(
            correction_material),
        "issuance_boundary": {
            "immutable_encoder_import_correction_preserved": True,
            "immutable_successor_scorer_contract_preserved": True,
            "state_and_assignment_manifests_preserved": True,
            "valid_smoke_branches_and_frames_preserved": True,
            "source_tree_clean_and_committed": True,
            "exact_failure_time_branch_receipt_metadata_validated": True,
            "branch_outcome_or_label_value_read_for_correction": False,
            "frame_latent_weight_or_predictor_artifact_read_for_correction": False,
            "double_prelatent_absence_audit_required": True,
            "failure_time_receipts_may_later_be_refreshed_by_full_corpus": True,
            "later_consumption_requires_failure_time_receipts_live": False,
            "scientific_contract_manifest_or_encoder_checkpoint_reissued": False,
            "branch_regenerated_based_on_label": False,
            "final_200_state_corpus_authorised": False,
        },
    }
    if set(payload) != _ENCODER_COMPUTE_DTYPE_CORRECTION_KEYS - {
            ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction construction surface changed")
    payload[ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_encoder_compute_dtype_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _ENCODER_COMPUTE_DTYPE_CORRECTION_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != ENCODER_COMPUTE_DTYPE_CORRECTION_SCHEMA
            or correction.get("status") != ENCODER_COMPUTE_DTYPE_CORRECTION_STATUS
            or correction.get("complete") is not True
            or correction.get("encoder_compute_dtype_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction version changed")
    expected = build_encoder_compute_dtype_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_encoder_import_correction=correction.get(
            "immutable_encoder_import_correction", {}),
        immutable_successor_scorer_contract_binding=correction.get(
            "immutable_successor_scorer_contract_binding", {}),
        focused_test_source_transitions=correction.get(
            "focused_test_source_transitions", []),
        branch_smoke_binding=correction.get("immutable_branch_smoke_binding", {}),
        branch_corpus_binding=correction.get(
            "immutable_partial_corpus_receipt_binding", {}),
        failed_encoder_source_binding=correction.get(
            "failed_encoder_source_binding", {}),
        unchanged_dev_encoder_source_binding=correction.get(
            "unchanged_dev_encoder_source_binding", {}),
        unchanged_stage_a_fp32_source_binding=correction.get(
            "unchanged_stage_a_fp32_source_binding", {}),
        upstream_rope_source_binding=correction.get(
            "upstream_rope_source_binding", {}),
        prelatent_outputs_absent_at_issue=correction.get(
            "prelatent_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        immutable_import = _load_immutable_encoder_import_correction(root=root)
        successor = _load_immutable_successor_scorer_contract_binding(root=root)
        tests = _encoder_compute_dtype_focused_test_source_transitions(root=root)
        failed, dev, stage_a, upstream = (
            _validate_live_encoder_compute_dtype_source_evidence(root=root))
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or immutable_import
                != correction["immutable_encoder_import_correction"]
                or successor
                != correction["immutable_successor_scorer_contract_binding"]
                or tests != correction["focused_test_source_transitions"]
                or failed != correction["failed_encoder_source_binding"]
                or dev != correction["unchanged_dev_encoder_source_binding"]
                or stage_a
                != correction["unchanged_stage_a_fp32_source_binding"]
                or upstream != correction["upstream_rope_source_binding"]):
            raise ScorerFitCorpusV2DesignError(
                "live source or immutable dtype-correction lineage changed")
        if require_failure_boundary_live:
            smoke, corpus = _validate_live_encoder_import_failure_receipts(root=root)
            absence = audit_encoder_compute_dtype_correction_prelatent_absence(
                root=root)
            if (smoke != correction["immutable_branch_smoke_binding"]
                    or corpus
                    != correction["immutable_partial_corpus_receipt_binding"]
                    or absence != correction["prelatent_outputs_absent_at_issue"]):
                raise ScorerFitCorpusV2DesignError(
                    "encoder-compute-dtype failure boundary changed before issue")
    return correction


def encoder_compute_dtype_correction_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_encoder_compute_dtype_correction(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction raw bytes changed")
    return {
        "path": str(ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH),
        "schema": ENCODER_COMPUTE_DTYPE_CORRECTION_SCHEMA,
        "self_digest_key": ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY,
        "self_digest": correction[ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


_IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_KEYS = frozenset({
    "payload", "binding",
})


def validate_immutable_encoder_compute_dtype_correction(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact dc4f dtype correction without live-source equality."""

    if (not isinstance(value, Mapping)
            or set(value) != _IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_KEYS
            or not isinstance(value.get("payload"), Mapping)
            or not isinstance(value.get("binding"), Mapping)):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-compute-dtype correction is not closed")
    immutable = copy.deepcopy(dict(value))
    payload = validate_encoder_compute_dtype_correction(
        immutable["payload"], validate_live_authorities=False)
    expected_binding = encoder_compute_dtype_correction_artifact_binding(
        payload, _pretty_json_bytes(payload))
    if (immutable["binding"] != expected_binding
            or expected_binding
            != IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_BINDING
            or payload.get(ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY)
            != IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST
            or payload.get("source_repository_commit")
            != ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-compute-dtype correction changed")
    immutable["payload"] = payload
    immutable["binding"] = expected_binding
    return immutable


_ENCODER_PATH_PROJECTION_CORRECTION_KEYS = frozenset({
    "schema", "status", "complete", "encoder_path_projection_correction_version",
    "source_repository_commit", "source_bindings", "source_binding_set_digest",
    "historical_source_repository_commit",
    "immutable_encoder_compute_dtype_correction",
    "immutable_encoder_compute_dtype_correction_digest",
    "immutable_successor_scorer_contract_binding",
    "failed_encoder_source_binding", "immutable_base_smoke_artifact_bundle",
    "base_smoke_artifact_bundle_digest", "focused_test_source_transitions",
    "production_source_transition", "production_source_transition_digest",
    "focused_test_source_transition_digest", "preserved_scientific_contract",
    "preserved_scientific_contract_digest", "downstream_outputs_absent_at_issue",
    "downstream_outputs_absent_at_issue_digest",
    "single_shard_regeneration_transaction_contract",
    "single_shard_regeneration_transaction_contract_digest",
    "preissue_single_shard_regeneration_transaction_audit",
    "preissue_single_shard_regeneration_transaction_audit_digest",
    "single_shard_regeneration_transaction_artifacts_absent_at_issue",
    "single_shard_regeneration_transaction_artifacts_absent_at_issue_digest",
    "encoder_path_projection_failure_boundary",
    "encoder_path_projection_failure_boundary_digest",
    "encoder_path_projection_correction",
    "encoder_path_projection_correction_material_digest", "issuance_boundary",
    ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY,
})


def build_encoder_path_projection_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_encoder_compute_dtype_correction: Mapping[str, Any],
        immutable_successor_scorer_contract_binding: Mapping[str, Any],
        focused_test_source_transitions: Sequence[Mapping[str, Any]],
        failed_encoder_source_binding: Mapping[str, Any],
        base_smoke_artifact_bundle: Mapping[str, Any],
        downstream_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        single_shard_regeneration_transaction_artifacts_absent_at_issue:
            Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the post-base-smoke logical-path correction without outcomes."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction source commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    immutable_dtype = validate_immutable_encoder_compute_dtype_correction(
        immutable_encoder_compute_dtype_correction)
    dtype_payload = immutable_dtype["payload"]
    historical_sources = dtype_payload["source_bindings"]
    changed = _changed_source_paths(historical_sources, current_sources)
    expected_changed = sorted(
        ENCODER_PATH_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    if changed != expected_changed:
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction changed an unauthorised source path")
    successor = _validate_immutable_successor_scorer_contract_binding(
        immutable_successor_scorer_contract_binding)
    if dtype_payload["immutable_successor_scorer_contract_binding"] != successor:
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection scorer-contract chain changed")
    if dict(failed_encoder_source_binding) != (
            ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING):
        raise ScorerFitCorpusV2DesignError(
            "path-projection failure source binding changed")
    historical_encoder = next(
        row for row in historical_sources
        if row["path"] == ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING[
            "path"])
    if (historical_encoder["byte_count"]
            != ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING["byte_count"]
            or historical_encoder["sha256"]
            != ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING["sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "path-projection failure source disagrees with immutable closure")
    if (not isinstance(base_smoke_artifact_bundle, Mapping)
            or dict(base_smoke_artifact_bundle)
            != IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_ARTIFACT_BUNDLE):
        raise ScorerFitCorpusV2DesignError(
            "path-projection base-smoke artifact bundle changed")
    base_bundle = copy.deepcopy(
        IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_ARTIFACT_BUNDLE)
    tests = _validate_encoder_path_projection_focused_test_transitions(
        list(focused_test_source_transitions))
    absence = _validate_encoder_path_projection_correction_absence(
        list(downstream_outputs_absent_at_issue))
    transaction_absence = _validate_encoder_path_projection_transaction_absence(
        list(single_shard_regeneration_transaction_artifacts_absent_at_issue))
    production_transition = {
        "allowed_changed_source_paths": list(
            ENCODER_PATH_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_bound_source_paths": [
            row["path"] for row in historical_sources],
        "current_bound_source_paths": [row["path"] for row in current_sources],
        "failed_encoder_source_binding": copy.deepcopy(
            ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING),
        "extra_production_source_path_changed": False,
    }
    science = copy.deepcopy(ENCODER_PATH_PROJECTION_CORRECTION_PRESERVED_SCIENCE)
    failure = copy.deepcopy(ENCODER_PATH_PROJECTION_FAILURE_BOUNDARY)
    transaction_contract = copy.deepcopy(
        ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT)
    transaction_audit = copy.deepcopy(
        ENCODER_PATH_PROJECTION_PREISSUE_SINGLE_SHARD_TRANSACTION_AUDIT)
    if (science.get("encoder_compute_dtype_correction_digest")
            != IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST
            or failure.get("historical_source_repository_commit")
            != ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT
            or transaction_contract.get("complete") is not True
            or transaction_audit.get(
                "path_projection_correction_artifact_issued_when_discovered")
            is not False):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection immutable science or failure chain changed")
    correction_material = {
        "status": (
            "SOURCE_ONLY_LOGICAL_PATH_PROJECTION_CORRECTION_WITH_"
            "PREISSUE_INTERRUPTION_SAFETY_HARDENING"),
        "defect": (
            "RESOLVED_MANAGED_GENERATED_PATH_WAS_PROJECTED_RELATIVE_TO_"
            "LEXICAL_REPOSITORY_ROOT"),
        "correction": (
            "PROJECT_REGISTERED_ARTIFACTS_THROUGH_THE_MANAGED_LOGICAL_"
            "SCORER_FIT_ROOT"),
        "filesystem_alias_or_artifact_location_changed_during_issue_or_"
        "path_digest_migration": False,
        "branch_row_frame_or_latent_shard_changed_during_issue_or_path_"
        "digest_migration": False,
        "latent_values_deserialized_or_reencoded_during_correction_issue":
            False,
        "preprocessing_changed": False,
        "target_normalisation_changed": False,
        "target_encoder_architecture_checkpoint_or_output_layer_changed": False,
        "latent_shape_token_order_or_storage_dtype_changed": False,
        "oracle_label_scorer_contract_or_manifest_changed": False,
        "runtime_compute_dtype_changed": False,
        "path_projection_defect_is_read_only_validator_projection": True,
        "authorised_path_digest_metadata_transition_count": 1,
        "authorised_path_digest_metadata_transition": (
            "ADD_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST_TO_INDEX_AND_"
            "SMOKE_WITHOUT_LATENT_SHARD_WRITE"),
        "encoding_invocation_summary_is_operational_and_may_refresh": True,
        "path_digest_metadata_transition_requires_all_13_shard_bindings_"
        "unchanged": True,
        "path_digest_metadata_current_current_recovery_requires_exact_file_"
        "reopen_fsync_and_parent_directory_fsync": True,
        "prospective_interruption_safety_implementation_of_existing_smoke_"
        "requirement": True,
        "authorised_single_shard_regeneration_transaction_count": 1,
        "authorised_transaction_target_candidate_index": 0,
        "authorised_transaction_context_target_count": 0,
        "authorised_transaction_horizon_target_count": 1,
        "authorised_transaction_non_target_latent_write_count": 0,
        "direct_active_target_unlink_authorised": False,
        "prepared_receipt_before_atomic_move_required": True,
        "retained_exact_backup_required": True,
        "complete_receipt_before_pass_smoke_publication_required": True,
        "immutable_prepared_and_complete_receipts_required": True,
        "mutable_transaction_phase_receipt_authorised": False,
        "transaction_changes_science_labels_or_latent_values": False,
        "resume_scope_after_metadata_transition": (
            "ZERO_NEW_THEN_PREPARED_ATOMIC_MOVE_REGENERATION_COMPLETE_"
            "TRANSACTION_THEN_FULL_MISSING_CORPUS"),
    }
    payload: dict[str, Any] = {
        "schema": ENCODER_PATH_PROJECTION_CORRECTION_SCHEMA,
        "status": ENCODER_PATH_PROJECTION_CORRECTION_STATUS,
        "complete": True,
        "encoder_path_projection_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
        "immutable_encoder_compute_dtype_correction": copy.deepcopy(
            immutable_dtype),
        "immutable_encoder_compute_dtype_correction_digest":
            IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST,
        "immutable_successor_scorer_contract_binding": successor,
        "failed_encoder_source_binding": copy.deepcopy(
            ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING),
        "immutable_base_smoke_artifact_bundle": base_bundle,
        "base_smoke_artifact_bundle_digest": canonical_digest(base_bundle),
        "focused_test_source_transitions": tests,
        "production_source_transition": production_transition,
        "production_source_transition_digest": canonical_digest(
            production_transition),
        "focused_test_source_transition_digest": canonical_digest(tests),
        "preserved_scientific_contract": science,
        "preserved_scientific_contract_digest": canonical_digest(science),
        "downstream_outputs_absent_at_issue": absence,
        "downstream_outputs_absent_at_issue_digest": canonical_digest(absence),
        "single_shard_regeneration_transaction_contract":
            transaction_contract,
        "single_shard_regeneration_transaction_contract_digest":
            canonical_digest(transaction_contract),
        "preissue_single_shard_regeneration_transaction_audit":
            transaction_audit,
        "preissue_single_shard_regeneration_transaction_audit_digest":
            canonical_digest(transaction_audit),
        "single_shard_regeneration_transaction_artifacts_absent_at_issue":
            transaction_absence,
        "single_shard_regeneration_transaction_artifacts_absent_at_issue_digest":
            canonical_digest(transaction_absence),
        "encoder_path_projection_failure_boundary": failure,
        "encoder_path_projection_failure_boundary_digest": canonical_digest(failure),
        "encoder_path_projection_correction": correction_material,
        "encoder_path_projection_correction_material_digest": canonical_digest(
            correction_material),
        "issuance_boundary": {
            "immutable_dtype_and_import_correction_chain_preserved": True,
            "immutable_successor_scorer_contract_preserved": True,
            "exact_base_index_summary_smoke_and_13_shards_validated": True,
            "source_tree_clean_and_committed": True,
            "branch_outcome_or_label_value_read_for_correction": False,
            "latent_value_deserialized_for_correction": False,
            "latent_shard_bytes_hashed_for_identity_only": True,
            "latent_shard_hashing_must_use_noatime_reads": True,
            "double_base_bundle_downstream_and_transaction_absence_validation_"
            "required": True,
            "later_consumption_requires_pretransition_metadata_live": False,
            "path_digest_metadata_transition_authorised": True,
            "branch_or_latent_shard_write_authorised_during_issue": False,
            "single_shard_transaction_control_receipt_write_authorised_during_"
            "issue": False,
            "zero_new_or_single_shard_transaction_started_at_issue": False,
            "direct_write_to_final_correction_path_allowed": False,
            "correction_staged_relative_path": str(
                ENCODER_PATH_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH),
            "correction_staged_file_fsync_before_publication_required": True,
            "correction_atomic_no_overwrite_publication": (
                "LINK_STAGED_TO_FINAL"),
            "parent_fsync_immediately_after_final_link_required": True,
            "parent_fsync_immediately_after_staged_unlink_required": True,
            "exact_final_and_staged_hardlink_cleanup_is_resumable": True,
            "final_200_state_corpus_authorised": False,
        },
    }
    if set(payload) != _ENCODER_PATH_PROJECTION_CORRECTION_KEYS - {
            ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction construction surface changed")
    payload[ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_encoder_path_projection_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _ENCODER_PATH_PROJECTION_CORRECTION_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != ENCODER_PATH_PROJECTION_CORRECTION_SCHEMA
            or correction.get("status") != ENCODER_PATH_PROJECTION_CORRECTION_STATUS
            or correction.get("complete") is not True
            or correction.get("encoder_path_projection_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction version changed")
    expected = build_encoder_path_projection_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_encoder_compute_dtype_correction=correction.get(
            "immutable_encoder_compute_dtype_correction", {}),
        immutable_successor_scorer_contract_binding=correction.get(
            "immutable_successor_scorer_contract_binding", {}),
        focused_test_source_transitions=correction.get(
            "focused_test_source_transitions", []),
        failed_encoder_source_binding=correction.get(
            "failed_encoder_source_binding", {}),
        base_smoke_artifact_bundle=correction.get(
            "immutable_base_smoke_artifact_bundle", {}),
        downstream_outputs_absent_at_issue=correction.get(
            "downstream_outputs_absent_at_issue", []),
        single_shard_regeneration_transaction_artifacts_absent_at_issue=
            correction.get(
                "single_shard_regeneration_transaction_artifacts_absent_at_issue",
                []),
    )
    if (correction != expected
            or correction.get(ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        immutable_dtype = _load_immutable_encoder_compute_dtype_correction(root=root)
        successor = _load_immutable_successor_scorer_contract_binding(root=root)
        tests = _encoder_path_projection_focused_test_source_transitions(root=root)
        failed = _validate_live_encoder_path_projection_failure_source(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or immutable_dtype
                != correction["immutable_encoder_compute_dtype_correction"]
                or successor
                != correction["immutable_successor_scorer_contract_binding"]
                or tests != correction["focused_test_source_transitions"]
                or failed != correction["failed_encoder_source_binding"]):
            raise ScorerFitCorpusV2DesignError(
                "live source or immutable path-projection lineage changed")
        if require_failure_boundary_live:
            bundle = _validate_live_encoder_path_projection_base_bundle(root=root)
            absence = audit_encoder_path_projection_correction_downstream_absence(
                root=root)
            transaction_absence = (
                audit_encoder_path_projection_transaction_artifacts_absent(
                    root=root))
            if (bundle != correction["immutable_base_smoke_artifact_bundle"]
                    or absence
                    != correction["downstream_outputs_absent_at_issue"]
                    or transaction_absence != correction[
                        "single_shard_regeneration_transaction_artifacts_"
                        "absent_at_issue"]):
                raise ScorerFitCorpusV2DesignError(
                    "path-projection failure boundary changed before issue")
    return correction


def encoder_path_projection_correction_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_encoder_path_projection_correction(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction raw bytes changed")
    return {
        "path": str(ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH),
        "schema": ENCODER_PATH_PROJECTION_CORRECTION_SCHEMA,
        "self_digest_key": ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY,
        "self_digest": correction[ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


def validate_immutable_encoder_path_projection_correction(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact predecessor correction without live-source replay."""

    if not isinstance(value, Mapping) or set(value) != {"payload", "binding"}:
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-path-projection correction is not closed")
    immutable = copy.deepcopy(dict(value))
    payload = validate_encoder_path_projection_correction(
        immutable.get("payload", {}), validate_live_authorities=False)
    raw = _pretty_json_bytes(payload)
    expected_binding = encoder_path_projection_correction_artifact_binding(
        payload, raw)
    if (immutable.get("binding") != expected_binding
            or expected_binding !=
            IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_BINDING
            or payload.get(ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY)
            != IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST
            or payload.get("source_repository_commit")
            != BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-path-projection correction changed")
    immutable["payload"] = payload
    immutable["binding"] = expected_binding
    return immutable


_BRANCH_REDRIVE_PROJECTION_CORRECTION_KEYS = frozenset({
    "schema", "status", "complete",
    "branch_redrive_projection_correction_version",
    "source_repository_commit", "source_bindings", "source_binding_set_digest",
    "historical_source_repository_commit",
    "immutable_encoder_path_projection_correction",
    "immutable_encoder_path_projection_correction_digest",
    "production_source_transition", "production_source_transition_digest",
    "partial_corpus_failure_boundary", "partial_corpus_failure_boundary_digest",
    "invalid_attempt_receipt_bindings", "invalid_attempt_receipt_binding_digest",
    "completed_smoke_boundary", "completed_smoke_boundary_digest",
    "branch_redrive_projection_failure_boundary",
    "branch_redrive_projection_failure_boundary_digest",
    "preserved_scientific_contract", "preserved_scientific_contract_digest",
    "downstream_outputs_absent_at_issue",
    "downstream_outputs_absent_at_issue_digest",
    "branch_redrive_projection_correction",
    "branch_redrive_projection_correction_material_digest",
    "issuance_boundary", BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY,
})


def build_branch_redrive_projection_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_encoder_path_projection_correction: Mapping[str, Any],
        partial_corpus_failure_boundary: Mapping[str, Any],
        invalid_attempt_receipt_bindings: Sequence[Mapping[str, Any]],
        completed_smoke_boundary: Mapping[str, Any],
        downstream_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the source-only post-partial-corpus redrive correction."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction source commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    immutable_path = validate_immutable_encoder_path_projection_correction(
        immutable_encoder_path_projection_correction)
    path_payload = immutable_path["payload"]
    if (path_payload.get("source_repository_commit")
            != BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive historical source commit changed")
    changed = _changed_source_paths(
        path_payload["source_bindings"], current_sources)
    expected_changed = sorted(
        BRANCH_REDRIVE_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    if changed != expected_changed:
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction changed an unauthorised source path")
    partial = copy.deepcopy(dict(partial_corpus_failure_boundary))
    invalid = copy.deepcopy(list(invalid_attempt_receipt_bindings))
    smoke = copy.deepcopy(dict(completed_smoke_boundary))
    if partial != IMMUTABLE_BRANCH_REDRIVE_PARTIAL_CORPUS_BINDING:
        raise ScorerFitCorpusV2DesignError(
            "partial branch-corpus failure boundary changed")
    if invalid != list(IMMUTABLE_BRANCH_REDRIVE_INVALID_ATTEMPT_RECEIPT_BINDINGS):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive invalid-attempt receipt inventory changed")
    if smoke != IMMUTABLE_BRANCH_REDRIVE_COMPLETED_SMOKE_BUNDLE:
        raise ScorerFitCorpusV2DesignError(
            "completed smoke boundary changed")
    absence = _validate_branch_redrive_projection_correction_absence(
        list(downstream_outputs_absent_at_issue))
    production_transition = {
        "allowed_changed_source_paths": list(
            BRANCH_REDRIVE_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_bound_source_paths": [
            row["path"] for row in path_payload["source_bindings"]],
        "current_bound_source_paths": [row["path"] for row in current_sources],
        "extra_production_source_path_changed": False,
    }
    failure = copy.deepcopy(BRANCH_REDRIVE_PROJECTION_FAILURE_BOUNDARY)
    science = copy.deepcopy(BRANCH_REDRIVE_PROJECTION_CORRECTION_PRESERVED_SCIENCE)
    correction_material = {
        "status": "SOURCE_ONLY_BRANCH_REDRIVE_STRUCTURAL_PROJECTION_CORRECTION",
        "defect": (
            "ACTIVE_MANIFEST_CANDIDATE_INDICES_ENTERED_OUTCOME_FREE_"
            "STRUCTURAL_EVIDENCE_AND_BROAD_CATCH_RELABELLED_EXCEPTION"),
        "correction": (
            "PROJECT_ONLY_REGISTERED_STRUCTURAL_STATE_FIELDS_BEFORE_FULL_BANK_"
            "L_MAX_REVALIDATION_AND_PRESERVE_INDIVIDUAL_COMPARISON_TRUTH"),
        "scientific_state_or_assignment_identity_changed": False,
        "candidate_or_oracle_changed": False,
        "branch_outcome_or_label_value_read_for_correction": False,
        "branch_outcome_or_label_value_used_for_selection": False,
        "branch_rows_ledger_hashed_without_content_parse": True,
        "valid_completed_branch_write_authorised": False,
        "failed_state_candidate_branch_execution_observed": False,
        "invalid_attempt_receipts_preserved": True,
        "state_replacement_authorised": False,
        "unchanged_source_retry_authorised": False,
        "corrected_resume_authorised": True,
        "corrected_resume_scope": "MISSING_REGISTERED_ASSIGNMENTS_ONLY",
        "scientific_or_feasibility_terminal_issued": False,
    }
    payload: dict[str, Any] = {
        "schema": BRANCH_REDRIVE_PROJECTION_CORRECTION_SCHEMA,
        "status": BRANCH_REDRIVE_PROJECTION_CORRECTION_STATUS,
        "complete": True,
        "branch_redrive_projection_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
        "immutable_encoder_path_projection_correction": immutable_path,
        "immutable_encoder_path_projection_correction_digest":
            IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST,
        "production_source_transition": production_transition,
        "production_source_transition_digest": canonical_digest(
            production_transition),
        "partial_corpus_failure_boundary": partial,
        "partial_corpus_failure_boundary_digest": canonical_digest(partial),
        "invalid_attempt_receipt_bindings": invalid,
        "invalid_attempt_receipt_binding_digest": canonical_digest(invalid),
        "completed_smoke_boundary": smoke,
        "completed_smoke_boundary_digest": canonical_digest(smoke),
        "branch_redrive_projection_failure_boundary": failure,
        "branch_redrive_projection_failure_boundary_digest": canonical_digest(
            failure),
        "preserved_scientific_contract": science,
        "preserved_scientific_contract_digest": canonical_digest(science),
        "downstream_outputs_absent_at_issue": absence,
        "downstream_outputs_absent_at_issue_digest": canonical_digest(absence),
        "branch_redrive_projection_correction": correction_material,
        "branch_redrive_projection_correction_material_digest": canonical_digest(
            correction_material),
        "issuance_boundary": {
            "source_tree_clean_and_committed": True,
            "immutable_path_projection_correction_reopened": True,
            "partial_corpus_metadata_validated_twice_before_publication": True,
            "branch_rows_bytes_hashed_without_jsonl_parse": True,
            "invalid_attempt_metadata_receipts_validated_twice": True,
            "completed_smoke_metadata_validated_twice": True,
            "later_latent_training_qualification_and_development_absence_"
            "validated_twice": True,
            "later_consumption_requires_failure_time_partial_corpus_live": False,
            "later_consumption_requires_invalid_attempt_receipts_live": False,
            "direct_write_to_final_correction_path_allowed": False,
            "correction_staged_relative_path": str(
                BRANCH_REDRIVE_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH),
            "correction_staged_file_create_uses_o_excl": True,
            "correction_staged_file_fsync_before_publication_required": True,
            "correction_staged_mode_before_publication": "0444",
            "correction_atomic_no_overwrite_publication": "LINK_STAGED_TO_FINAL",
            "parent_directory_fsync_after_link_and_staged_unlink_required": True,
            "idempotent_exact_reopen_required": True,
            "final_200_state_corpus_authorised": False,
        },
    }
    if set(payload) != _BRANCH_REDRIVE_PROJECTION_CORRECTION_KEYS - {
            BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction construction surface changed")
    payload[BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY] = canonical_digest(
        payload)
    return payload


def validate_branch_redrive_projection_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    if (not isinstance(payload, Mapping)
            or set(payload) != _BRANCH_REDRIVE_PROJECTION_CORRECTION_KEYS):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != BRANCH_REDRIVE_PROJECTION_CORRECTION_SCHEMA
            or correction.get("status")
            != BRANCH_REDRIVE_PROJECTION_CORRECTION_STATUS
            or correction.get("complete") is not True
            or correction.get(
                "branch_redrive_projection_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction version changed")
    expected = build_branch_redrive_projection_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_encoder_path_projection_correction=correction.get(
            "immutable_encoder_path_projection_correction", {}),
        partial_corpus_failure_boundary=correction.get(
            "partial_corpus_failure_boundary", {}),
        invalid_attempt_receipt_bindings=correction.get(
            "invalid_attempt_receipt_bindings", []),
        completed_smoke_boundary=correction.get(
            "completed_smoke_boundary", {}),
        downstream_outputs_absent_at_issue=correction.get(
            "downstream_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]):
            raise ScorerFitCorpusV2DesignError(
                "live source differs from branch-redrive correction")
        if require_failure_boundary_live:
            partial, invalid, smoke = (
                _validate_live_branch_redrive_failure_boundary(root=root))
            absence = audit_branch_redrive_projection_correction_downstream_absence(
                root=root)
            if (partial != correction["partial_corpus_failure_boundary"]
                    or invalid != correction["invalid_attempt_receipt_bindings"]
                    or smoke != correction["completed_smoke_boundary"]
                    or absence != correction["downstream_outputs_absent_at_issue"]):
                raise ScorerFitCorpusV2DesignError(
                    "branch-redrive live failure boundary changed before issue")
    return correction


def branch_redrive_projection_correction_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_branch_redrive_projection_correction(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction raw bytes changed")
    return {
        "path": str(BRANCH_REDRIVE_PROJECTION_CORRECTION_RELATIVE_PATH),
        "schema": BRANCH_REDRIVE_PROJECTION_CORRECTION_SCHEMA,
        "self_digest_key": BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY,
        "self_digest": correction[
            BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


def completion_order_material(
        complete_structural_state_identity: Mapping[str, Any],
        designated_goal_identity: Mapping[str, Any], *,
        active_selector_digest: str = ACTIVE_SELECTOR_DIGEST,
        ) -> dict[str, Any]:
    """Return the exact outcome-free material used to order one identity."""

    if (not isinstance(complete_structural_state_identity, Mapping)
            or not isinstance(designated_goal_identity, Mapping)
            or not complete_structural_state_identity
            or not designated_goal_identity
            or active_selector_digest != ACTIVE_SELECTOR_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "completion ordering identity or selector binding changed")
    # Canonicalization also rejects NaN, infinity and non-JSON objects.
    structural = copy.deepcopy(dict(complete_structural_state_identity))
    goal = copy.deepcopy(dict(designated_goal_identity))
    _json_bytes(structural)
    _json_bytes(goal)
    return {
        "domain_separation_utf8": COMPLETION_ORDER_DOMAIN,
        "active_selector_contract_digest": active_selector_digest,
        "complete_structural_state_identity": structural,
        "designated_goal_identity": goal,
    }


def completion_order_key(
        complete_structural_state_identity: Mapping[str, Any],
        designated_goal_identity: Mapping[str, Any], *,
        active_selector_digest: str = ACTIVE_SELECTOR_DIGEST,
        ) -> tuple[str, bytes]:
    """Return SHA-256 primary order and structural bytes collision tie-break."""

    material = completion_order_material(
        complete_structural_state_identity, designated_goal_identity,
        active_selector_digest=active_selector_digest)
    structural = _json_bytes({
        "complete_structural_state_identity": material[
            "complete_structural_state_identity"],
        "designated_goal_identity": material["designated_goal_identity"],
    })
    preimage = (
        COMPLETION_ORDER_DOMAIN.encode("utf-8") + b"\x00"
        + active_selector_digest.encode("ascii") + b"\x00"
        + _json_bytes(material["complete_structural_state_identity"]) + b"\x00"
        + _json_bytes(material["designated_goal_identity"])
    )
    return hashlib.sha256(preimage).hexdigest(), structural


def _pin_relative(root: Path, relative: str | Path, *, label: str) -> Path:
    repository = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or any(
            part == "sealed" or part == "sealed_test.json"
            or part.startswith("sealed_") for part in rel.parts):
        raise ScorerFitCorpusV2DesignError(f"{label} path is inaccessible")
    cursor = repository
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink() and rel != SCORER_FIT_RELATIVE_PATH:
            # Exact generated-root aliases are handled by _pin_generated.
            raise ScorerFitCorpusV2DesignError(f"{label} path is symlinked")
    return repository / rel


def _pin_generated(root: Path, relative: str | Path, *, label: str) -> Path:
    repository = Path(root).resolve()
    rel = Path(relative)
    matches: list[tuple[Path, Path]] = []
    for managed_root in MANAGED_GENERATED_ROOTS:
        try:
            matches.append((managed_root, rel.relative_to(managed_root)))
        except ValueError:
            continue
    if len(matches) != 1:
        raise ScorerFitCorpusV2DesignError(
            f"{label} escaped the managed generated roots")
    managed_root, suffix = matches[0]
    if not suffix.parts or any(
            part == "sealed" or part == "sealed_test.json"
            or part.startswith("sealed_") or part == ".."
            for part in suffix.parts):
        raise ScorerFitCorpusV2DesignError(f"{label} path is inaccessible")
    logical = repository / managed_root
    if logical.is_symlink():
        target = logical.readlink()
        target = target if target.is_absolute() else logical.parent / target
        if target.name != logical.name:
            raise ScorerFitCorpusV2DesignError(
                "managed generated-root alias changed")
        base = target.resolve(strict=True)
    else:
        base = logical.resolve(strict=True) if logical.exists() else logical
    cursor = base
    for part in suffix.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ScorerFitCorpusV2DesignError(f"{label} path is symlinked")
    return base.joinpath(*suffix.parts)


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScorerFitCorpusV2DesignError(
            f"git source-custody check failed: {' '.join(args)}") from exc


def clean_source_authority(*, root: Path = ROOT) -> tuple[str, list[dict[str, Any]]]:
    """Bind the exact clean implementation closure without tree-wide export."""

    repository = Path(root).resolve()
    commit = _git("rev-parse", "HEAD", root=repository).decode().strip()
    if not _is_hex(commit, 40):
        raise ScorerFitCorpusV2DesignError("source commit is malformed")
    if _git("status", "--porcelain=v1", "--untracked-files=all", root=repository):
        raise ScorerFitCorpusV2DesignError(
            "V2 authority issuance requires a clean source commit")
    tracked = _git(
        "ls-files", "--error-unmatch", "--", *EXPECTED_SOURCE_PATHS,
        root=repository,
    ).decode().splitlines()
    if len(tracked) != len(EXPECTED_SOURCE_PATHS) or set(tracked) != set(
            EXPECTED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "V2 source closure is not fully tracked")
    rows: list[dict[str, Any]] = []
    for relative, role in SOURCE_SPECS:
        path = _pin_relative(repository, relative, label="V2 source")
        if not path.is_file() or path.is_symlink():
            raise ScorerFitCorpusV2DesignError(
                f"V2 source unavailable: {relative}")
        raw = path.read_bytes()
        if _git("show", f"HEAD:{relative}", root=repository) != raw:
            raise ScorerFitCorpusV2DesignError(
                f"V2 source differs from clean commit: {relative}")
        rows.append({
            "path": relative, "role": role, "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return commit, rows


def _transition_endpoint_from_raw(
        *, path: str, role: str, raw: bytes | None) -> dict[str, Any]:
    if raw is None:
        return {
            "path": path, "role": role, "exists": False,
            "byte_count": 0, "sha256": None,
        }
    return {
        "path": path, "role": role, "exists": True,
        "byte_count": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _historical_source_blob(
        *, root: Path, commit: str, relative: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=root, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode == 0:
        return completed.stdout
    # Distinguish a genuinely absent focused test from an invalid commit.
    resolved = _git("rev-parse", f"{commit}^{{commit}}", root=root).decode().strip()
    if resolved != commit:
        raise ScorerFitCorpusV2DesignError(
            "encoder-import historical source commit changed")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative}"], cwd=root,
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if exists.returncode == 0:
        raise ScorerFitCorpusV2DesignError(
            f"cannot read historical correction source: {relative}")
    return None


def _current_source_endpoint(
        *, root: Path, path: str, role: str) -> dict[str, Any]:
    repository = Path(root).resolve()
    source = _pin_relative(repository, path, label="encoder-import correction source")
    if not source.is_file() or source.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            f"encoder-import correction source unavailable: {path}")
    raw = source.read_bytes()
    if _git("show", f"HEAD:{path}", root=repository) != raw:
        raise ScorerFitCorpusV2DesignError(
            f"encoder-import correction source differs from clean HEAD: {path}")
    return _transition_endpoint_from_raw(path=path, role=role, raw=raw)


def _dev_encoder_source_transition(*, root: Path = ROOT) -> dict[str, Any]:
    path, role = ENCODER_IMPORT_CORRECTION_DEV_ENCODER_SOURCE_SPEC
    transition = {
        "path": path,
        "role": role,
        "historical": copy.deepcopy(
            ENCODER_IMPORT_CORRECTION_DEV_ENCODER_HISTORICAL_BINDING),
        "current": _current_source_endpoint(root=root, path=path, role=role),
    }
    return _validate_dev_encoder_source_transition(transition)


def _focused_test_source_transitions(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    repository = Path(root).resolve()
    transitions: list[dict[str, Any]] = []
    for path, role in ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_SPECS:
        transitions.append({
            "path": path,
            "role": role,
            "historical": _transition_endpoint_from_raw(
                path=path, role=role,
                raw=_historical_source_blob(
                    root=repository,
                    commit=ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
                    relative=path)),
            "current": _current_source_endpoint(
                root=repository, path=path, role=role),
        })
    return _validate_focused_test_source_transitions(transitions)


def _encoder_compute_dtype_focused_test_source_transitions(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    repository = Path(root).resolve()
    transitions: list[dict[str, Any]] = []
    for path, role in ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_SPECS:
        transitions.append({
            "path": path,
            "role": role,
            "historical": _transition_endpoint_from_raw(
                path=path, role=role,
                raw=_historical_source_blob(
                    root=repository,
                    commit=(
                        ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT),
                    relative=path)),
            "current": _current_source_endpoint(
                root=repository, path=path, role=role),
        })
    return _validate_encoder_compute_dtype_focused_test_transitions(transitions)


def _encoder_path_projection_focused_test_source_transitions(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    repository = Path(root).resolve()
    transitions: list[dict[str, Any]] = []
    for path, role in ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_SPECS:
        transitions.append({
            "path": path,
            "role": role,
            "historical": _transition_endpoint_from_raw(
                path=path, role=role,
                raw=_historical_source_blob(
                    root=repository,
                    commit=(
                        ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT),
                    relative=path)),
            "current": _current_source_endpoint(
                root=repository, path=path, role=role),
        })
    return _validate_encoder_path_projection_focused_test_transitions(
        transitions)


def _validate_live_encoder_path_projection_failure_source(
        *, root: Path = ROOT) -> dict[str, Any]:
    expected = ENCODER_PATH_PROJECTION_FAILURE_ENCODER_SOURCE_BINDING
    raw = _historical_source_blob(
        root=Path(root).resolve(),
        commit=ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
        relative=expected["path"])
    observed = _transition_endpoint_from_raw(
        path=expected["path"], role=expected["role"], raw=raw)
    if observed != expected:
        raise ScorerFitCorpusV2DesignError(
            "historical path-projection failure source binding changed")
    return copy.deepcopy(expected)


def _validate_live_encoder_compute_dtype_source_evidence(
        *, root: Path = ROOT,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repository = Path(root).resolve()
    failed_expected = ENCODER_COMPUTE_DTYPE_FAILURE_ENCODER_SOURCE_BINDING
    failed_raw = _historical_source_blob(
        root=repository,
        commit=ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT,
        relative=failed_expected["path"])
    failed = _transition_endpoint_from_raw(
        path=failed_expected["path"], role=failed_expected["role"],
        raw=failed_raw)
    if failed != failed_expected:
        raise ScorerFitCorpusV2DesignError(
            "historical failed encoder route binding changed")

    dev_expected = ENCODER_COMPUTE_DTYPE_UNCHANGED_DEV_ENCODER_BINDING
    dev = _current_source_endpoint(
        root=repository, path=dev_expected["path"], role=dev_expected["role"])
    stage_expected = ENCODER_COMPUTE_DTYPE_STAGE_A_FP32_SOURCE_BINDING
    stage_a = _current_source_endpoint(
        root=repository, path=stage_expected["path"], role=stage_expected["role"])
    if dev != dev_expected or stage_a != stage_expected:
        raise ScorerFitCorpusV2DesignError(
            "unchanged target-encoder or Stage-A FP32 source changed")

    upstream_expected = ENCODER_COMPUTE_DTYPE_UPSTREAM_ROPE_SOURCE_BINDING
    upstream_root = Path.home() / (
        ".cache/vjepa2-" + upstream_expected["repository_commit"])
    upstream_path = upstream_root / upstream_expected["path"]
    if (not upstream_path.is_file() or upstream_path.is_symlink()
            or _git("rev-parse", "HEAD", root=upstream_root).decode().strip()
            != upstream_expected["repository_commit"]):
        raise ScorerFitCorpusV2DesignError(
            "frozen upstream RoPE source repository changed")
    upstream_raw = upstream_path.read_bytes()
    if (len(upstream_raw) != upstream_expected["byte_count"]
            or hashlib.sha256(upstream_raw).hexdigest()
            != upstream_expected["sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "frozen upstream RoPE source binding changed")
    return (
        copy.deepcopy(failed_expected), copy.deepcopy(dev_expected),
        copy.deepcopy(stage_expected), copy.deepcopy(upstream_expected),
    )


def _load_immutable_manifest_replay_correction(
        *, root: Path = ROOT) -> dict[str, Any]:
    payload = load_manifest_replay_correction(
        root=root, validate_live_authorities=False)
    path = _pin_generated(
        root, MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH,
        label="immutable manifest-replay correction")
    _value, raw = _load_json(path, label="immutable manifest-replay correction")
    return validate_immutable_manifest_replay_correction({
        "payload": payload,
        "binding": manifest_replay_correction_artifact_binding(payload, raw),
    })


def _load_immutable_encoder_import_correction(
        *, root: Path = ROOT) -> dict[str, Any]:
    expected = IMMUTABLE_ENCODER_IMPORT_CORRECTION_BINDING
    path = _pin_generated(
        root, expected["path"], label="immutable encoder-import correction")
    if (not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-import correction mode changed")
    payload, raw = _load_json(path, label="immutable encoder-import correction")
    if (len(raw) != expected["byte_count"]
            or hashlib.sha256(raw).hexdigest() != expected["raw_sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-import correction raw binding changed")
    return validate_immutable_encoder_import_correction({
        "payload": payload,
        "binding": encoder_import_correction_artifact_binding(payload, raw),
    })


def _load_immutable_encoder_compute_dtype_correction(
        *, root: Path = ROOT) -> dict[str, Any]:
    expected = IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_BINDING
    path = _pin_generated(
        root, expected["path"],
        label="immutable encoder-compute-dtype correction")
    if (not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-compute-dtype correction mode changed")
    payload, raw = _load_json(
        path, label="immutable encoder-compute-dtype correction")
    if (len(raw) != expected["byte_count"]
            or hashlib.sha256(raw).hexdigest() != expected["raw_sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-compute-dtype correction raw binding changed")
    return validate_immutable_encoder_compute_dtype_correction({
        "payload": payload,
        "binding": encoder_compute_dtype_correction_artifact_binding(
            payload, raw),
    })


def _load_immutable_encoder_path_projection_correction(
        *, root: Path = ROOT) -> dict[str, Any]:
    expected = IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_BINDING
    path = _pin_generated(
        root, expected["path"],
        label="immutable encoder-path-projection correction")
    if (not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-path-projection correction mode changed")
    payload, raw = _load_json(
        path, label="immutable encoder-path-projection correction")
    if (len(raw) != expected["byte_count"]
            or hashlib.sha256(raw).hexdigest() != expected["raw_sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "immutable encoder-path-projection correction raw binding changed")
    return validate_immutable_encoder_path_projection_correction({
        "payload": payload,
        "binding": encoder_path_projection_correction_artifact_binding(
            payload, raw),
    })


def _load_immutable_successor_scorer_contract_binding(
        *, root: Path = ROOT) -> dict[str, Any]:
    expected = IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING
    path = _pin_generated(
        root, expected["path"], label="immutable successor scorer contract")
    if (not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "immutable successor scorer contract mode changed")
    payload, raw = _load_json(path, label="immutable successor scorer contract")
    contract = payload.get("contract")
    artifact_key = expected["self_digest_key"]
    contract_key = expected["embedded_contract_self_digest_key"]
    if (len(raw) != expected["byte_count"]
            or hashlib.sha256(raw).hexdigest() != expected["raw_sha256"]
            or payload.get("schema") != expected["schema"]
            or payload.get(artifact_key) != expected["self_digest"]
            or canonical_digest(_without(payload, artifact_key))
            != expected["self_digest"]
            or payload.get("source_repository_commit")
            != expected["source_repository_commit"]
            or payload.get("complete") is not True
            or not isinstance(contract, Mapping)
            or contract.get("schema") != expected["embedded_contract_schema"]
            or contract.get(contract_key)
            != expected["embedded_contract_self_digest"]
            or canonical_digest(_without(contract, contract_key))
            != expected["embedded_contract_self_digest"]
            or contract.get("source_binding", {}).get(
                "source_repository_commit")
            != expected["source_repository_commit"]
            or payload.get("branch_execution_started") is not False
            or payload.get("candidate_outcomes_consumed") is not False
            or payload.get("scorer_training_started") is not False
            or payload.get("predictor_checkpoints_opened") is not False):
        raise ScorerFitCorpusV2DesignError(
            "immutable successor scorer contract changed")
    return _validate_immutable_successor_scorer_contract_binding(expected)


def _load_exact_runtime_metadata_binding(
        binding: Mapping[str, Any], *, root: Path, label: str,
        ) -> dict[str, Any]:
    path = _pin_generated(root, binding["path"], label=label)
    if not path.is_file() or path.is_symlink():
        raise ScorerFitCorpusV2DesignError(f"{label} is unavailable")
    payload, raw = _load_json(path, label=label)
    if (len(raw) != binding["byte_count"]
            or hashlib.sha256(raw).hexdigest() != binding["raw_sha256"]
            or payload.get("schema") != binding["schema"]
            or payload.get(binding["self_digest_key"])
            != binding["self_digest"]):
        raise ScorerFitCorpusV2DesignError(f"{label} binding changed")
    return payload


def _validate_live_encoder_import_failure_receipts(
        *, root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    smoke_binding = IMMUTABLE_ENCODER_IMPORT_FAILURE_BRANCH_SMOKE_BINDING
    corpus_binding = IMMUTABLE_ENCODER_IMPORT_FAILURE_CORPUS_RECEIPT_BINDING
    smoke = _load_exact_runtime_metadata_binding(
        smoke_binding, root=root, label="encoder-import failure branch smoke")
    corpus = _load_exact_runtime_metadata_binding(
        corpus_binding, root=root, label="encoder-import failure corpus receipt")
    smoke_key = smoke_binding["self_digest_key"]
    corpus_payload = corpus.get("corpus_digest_payload")
    if (builder_default_canonical_digest(_without(smoke, smoke_key))
            != smoke_binding["self_digest"]
            or smoke.get("status") != "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
            or smoke.get("pass") is not True
            or smoke.get("candidate_indices") != list(CANDIDATE_INDICES)
            or smoke.get("branch_count") != 12
            or smoke.get("rendered_horizon_frame_count") != 48
            or smoke.get("state_manifest_digest")
            != ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE[
                "state_manifest_digest"]
            or smoke.get("full_bank_assignment_manifest_digest")
            != ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE[
                "assignment_manifest_digest"]
            or smoke.get("scorer_fit_corpus_v2_scorer_contract_digest")
            != ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE[
                "successor_scorer_contract_digest"]
            or smoke.get(
                "scorer_fit_corpus_v2_scorer_contract_artifact_digest")
            != ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE[
                "successor_scorer_contract_artifact_digest"]
            or not isinstance(corpus_payload, Mapping)
            or builder_default_canonical_digest(corpus_payload)
            != corpus_binding["self_digest"]
            or smoke.get("corpus_digest") != corpus_binding["self_digest"]
            or corpus.get("status") != "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
            or corpus.get("complete") is not False
            or corpus.get("state_count") != 120
            or corpus.get("completed_states") != 1
            or corpus.get("expected_branches") != 1_440
            or corpus.get("attempted_branches") != 12
            or corpus.get("valid_branches") != 12
            or corpus.get("invalid_branches") != 0
            or corpus.get("state_manifest_digest")
            != ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE[
                "state_manifest_digest"]
            or corpus.get("full_bank_assignment_manifest_digest")
            != ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE[
                "assignment_manifest_digest"]):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import failure receipt metadata changed")
    forbidden = {"progress", "safety", "completion", "utility", "labels"}
    if forbidden.intersection(smoke) or forbidden.intersection(corpus):
        raise ScorerFitCorpusV2DesignError(
            "outcome-bearing field appeared in correction receipt metadata")
    return copy.deepcopy(smoke_binding), copy.deepcopy(corpus_binding)


def _sha256_regular_file(path: Path, *, label: str) -> tuple[str, int]:
    if not path.is_file() or path.is_symlink():
        raise ScorerFitCorpusV2DesignError(f"{label} is unavailable")
    noatime = getattr(os, "O_NOATIME", None)
    if noatime is None:
        raise ScorerFitCorpusV2DesignError(
            f"{label} cannot be hashed without changing atime")
    flags = (os.O_RDONLY | noatime | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ScorerFitCorpusV2DesignError(
            f"{label} cannot be opened without changing atime") from exc
    digest = hashlib.sha256()
    byte_count = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ScorerFitCorpusV2DesignError(
                f"{label} changed during no-atime open")
        while True:
            block = os.read(descriptor, 8 << 20)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), byte_count


def _validate_live_branch_redrive_partial_corpus(
        *, root: Path = ROOT) -> dict[str, Any]:
    expected = IMMUTABLE_BRANCH_REDRIVE_PARTIAL_CORPUS_BINDING
    receipt_binding = expected["corpus_receipt"]
    receipt_path = _pin_generated(
        root, receipt_binding["path"],
        label="branch-redrive partial corpus receipt")
    receipt, raw = _load_json(
        receipt_path, label="branch-redrive partial corpus receipt")
    projection_keys = {
        "schema", "status", "complete", "state_count", "completed_states",
        "expected_branches", "attempted_branches", "valid_branches",
        "invalid_branches", "corpus_digest", "branch_rows_sha256",
        "state_manifest_digest", "full_bank_assignment_manifest_digest",
    }
    projection = {key: receipt.get(key) for key in projection_keys}
    expected_projection = {
        key: value for key, value in receipt_binding.items()
        if key in projection_keys}
    if (len(raw) != receipt_binding["byte_count"]
            or hashlib.sha256(raw).hexdigest()
            != receipt_binding["raw_sha256"]
            or projection != expected_projection
            or {"progress", "safety", "completion", "utility", "labels"}
            .intersection(receipt)):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive partial corpus metadata changed")
    ledger_binding = expected["branch_rows_ledger"]
    ledger_path = _pin_generated(
        root, ledger_binding["path"],
        label="branch-redrive partial branch-rows ledger")
    digest, byte_count = _sha256_regular_file(
        ledger_path, label="branch-redrive partial branch-rows ledger")
    if (digest != ledger_binding["raw_sha256"]
            or byte_count != ledger_binding["byte_count"]
            or digest != receipt_binding["branch_rows_sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive partial branch-rows raw binding changed")
    return copy.deepcopy(expected)


def _validate_live_branch_redrive_invalid_attempt_receipts(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    expected = list(IMMUTABLE_BRANCH_REDRIVE_INVALID_ATTEMPT_RECEIPT_BINDINGS)
    expected_paths = {row["path"] for row in expected}
    directory = _pin_generated(
        root,
        SCORER_FIT_RELATIVE_PATH / "invalid_attempts_v2" /
        "redrive_records" / "__inventory_probe__",
        label="branch-redrive invalid-attempt inventory").parent
    if not directory.is_dir() or directory.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive invalid-attempt directory changed")
    observed_paths = {
        str(SCORER_FIT_RELATIVE_PATH / "invalid_attempts_v2" /
            "redrive_records" / path.name)
        for path in directory.iterdir()
    }
    if observed_paths != expected_paths:
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive invalid-attempt inventory coverage changed")
    null_fields = (
        "requested", "post_slew", "action_blocks", "action_context_blocks",
        "previous_applied_command", "proprio", "control", "snapshot_digest",
        "start_geodesic_m", "final_geodesic_m", "progress",
        "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
        "safety", "completion", "utility", "min_clearance_m",
        "evaluation_points", "truncated_at_block",
    )
    for binding in expected:
        path = _pin_generated(
            root, binding["path"], label="branch-redrive invalid attempt")
        digest, byte_count = _sha256_regular_file(
            path, label="branch-redrive invalid attempt")
        payload, _raw = _load_json(path, label="branch-redrive invalid attempt")
        safe_projection = {key: payload.get(key) for key in (
            "schema", "state_id", "state_index", "state_identity_digest",
            "candidate_index", "branch_identity_digest",
            "assignment_identity_digest", "branch_row_digest", "invalid_reason",
        )}
        expected_projection = {key: binding[key] for key in safe_projection}
        if (digest != binding["raw_sha256"]
                or byte_count != binding["byte_count"]
                or safe_projection != expected_projection
                or payload.get("status") != "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
                or payload.get("record_complete") is not True
                or payload.get("valid") is not False
                or payload.get("blocks_completed") != 0
                or payload.get("storage_bytes") != 0
                or payload.get("context_frames") != []
                or payload.get("horizon_frames") != []
                or payload.get("context_paths") != []
                or payload.get("horizon_paths") != []
                or any(payload.get(key) is not None for key in null_fields)):
            raise ScorerFitCorpusV2DesignError(
                "branch-redrive invalid-attempt metadata changed")
    if sorted(row["candidate_index"] for row in expected) != list(
            CANDIDATE_INDICES):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive invalid-attempt candidate coverage changed")
    return copy.deepcopy(expected)


def _validate_live_branch_redrive_completed_smoke(
        *, root: Path = ROOT) -> dict[str, Any]:
    expected = IMMUTABLE_BRANCH_REDRIVE_COMPLETED_SMOKE_BUNDLE
    complete_binding = expected["complete_transaction_receipt"]
    complete_path = _pin_generated(
        root, complete_binding["path"],
        label="branch-redrive completed smoke transaction")
    complete, complete_raw = _load_json(
        complete_path, label="branch-redrive completed smoke transaction")
    if (len(complete_raw) != complete_binding["byte_count"]
            or hashlib.sha256(complete_raw).hexdigest()
            != complete_binding["raw_sha256"]
            or complete.get(complete_binding["self_digest_key"])
            != complete_binding["self_digest"]
            or validate_full_bank_v2_smoke_regeneration_complete_receipt(
                complete) != complete):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive completed transaction binding changed")
    smoke_binding = expected["final_smoke_receipt"]
    smoke = _load_exact_path_projection_metadata(
        smoke_binding, root=root,
        label="branch-redrive completed smoke receipt")
    if (smoke.get("pass") is not True
            or smoke.get("smoke_protocol_complete") is not True
            or smoke.get("zero_new_resume_verified") is not True
            or smoke.get("single_shard_deletion_regeneration_verified") is not True
            or smoke.get("single_shard_regeneration_transaction_complete")
            is not True
            or smoke.get("single_shard_regeneration_target_atomic_move_count") != 1
            or smoke.get("single_shard_regeneration_target_regeneration_count") != 1
            or smoke.get("encoder_path_projection_correction_digest")
            != IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST
            or smoke.get("branch_count") != 12
            or smoke.get("rendered_horizon_frame_count") != 48
            or smoke.get("true_latent_trajectory_count") != 12):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive completed smoke metadata changed")
    index_binding = expected["smoke_latent_index"]
    index = _load_exact_path_projection_metadata(
        index_binding, root=root,
        label="branch-redrive smoke-only latent index")
    if (index.get("complete") is not False
            or len(index.get("context_records", []))
            != index_binding["context_record_count"]
            or len(index.get("horizon_records", []))
            != index_binding["horizon_record_count"]
            or index.get("encoder_path_projection_correction_digest")
            != IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive smoke-only latent metadata changed")
    return copy.deepcopy(expected)


def _validate_live_branch_redrive_failure_boundary(
        *, root: Path = ROOT,
        ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read only exact failure metadata; never parse branch-row outcomes."""

    return (
        _validate_live_branch_redrive_partial_corpus(root=root),
        _validate_live_branch_redrive_invalid_attempt_receipts(root=root),
        _validate_live_branch_redrive_completed_smoke(root=root),
    )


def _load_exact_path_projection_metadata(
        binding: Mapping[str, Any], *, root: Path, label: str,
        ) -> dict[str, Any]:
    path = _pin_generated(root, binding["path"], label=label)
    payload, raw = _load_json(path, label=label)
    if (len(raw) != binding["byte_count"]
            or hashlib.sha256(raw).hexdigest() != binding["raw_sha256"]
            or payload.get("schema") != binding["schema"]):
        raise ScorerFitCorpusV2DesignError(f"{label} raw binding changed")
    self_key = binding.get("self_digest_key")
    if self_key is not None and (
            payload.get(self_key) != binding.get("self_digest")
            or hashlib.sha256(json.dumps(
                _without(payload, self_key), sort_keys=True,
                ensure_ascii=False).encode("utf-8")).hexdigest()
            != binding.get("self_digest")):
        raise ScorerFitCorpusV2DesignError(f"{label} self binding changed")
    return payload


def _validate_live_encoder_path_projection_base_bundle(
        *, root: Path = ROOT) -> dict[str, Any]:
    bundle = IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_ARTIFACT_BUNDLE
    index = _load_exact_path_projection_metadata(
        bundle["latent_index_binding"], root=root,
        label="path-projection failure latent index")
    summary = _load_exact_path_projection_metadata(
        bundle["encoding_invocation_summary_binding"], root=root,
        label="path-projection failure encoding summary")
    smoke = _load_exact_path_projection_metadata(
        bundle["base_smoke_receipt_binding"], root=root,
        label="path-projection failure base smoke")
    if any("encoder_path_projection_correction_digest" in payload
           for payload in (index, summary, smoke)):
        raise ScorerFitCorpusV2DesignError(
            "path-projection digest predates its correction authority")
    if (index.get("complete") is not False
            or index.get("encoder_compute_dtype") != "float32"
            or index.get("encoder_compute_dtype_correction_digest")
            != IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST
            or index.get("context_shape") != [1, 3, 768, 1024]
            or index.get("horizon_shape") != [12, 4, 768, 1024]
            or len(index.get("context_records", [])) != 1
            or len(index.get("horizon_records", [])) != 12
            or summary.get("smoke") is not True
            or summary.get("new_context_shards") != 1
            or summary.get("new_horizon_shards") != 12
            or summary.get("new_shards") != 13
            or summary.get("resume_only_verified") is not False
            or summary.get("latents_index_digest")
            != bundle["latent_index_binding"]["self_digest"]
            or smoke.get("base_end_to_end_pass") is not True
            or smoke.get("pass") is not False
            or smoke.get("zero_new_resume_verified") is not False
            or smoke.get("single_shard_deletion_regeneration_verified")
            is not False
            or smoke.get("smoke_protocol_complete") is not False
            or smoke.get("true_latent_trajectory_count") != 12
            or smoke.get("encoder_compute_dtype") != "float32"
            or smoke.get("encoder_compute_dtype_correction_digest")
            != IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST
            or smoke.get("latent_index_digest")
            != bundle["latent_index_binding"]["self_digest"]):
        raise ScorerFitCorpusV2DesignError(
            "path-projection base-smoke metadata changed")
    indexed_inventory = []
    for record in list(index["context_records"]) + list(
            index["horizon_records"]):
        indexed_inventory.append({
            "path": str(SCORER_FIT_RELATIVE_PATH / str(record["path"])),
            "sha256": record["sha256"],
            "byte_count": record["byte_count"],
            "shape": record["shape"],
        })
    expected_inventory = bundle["latent_shard_inventory"]
    if sorted(indexed_inventory, key=lambda row: row["path"]) != sorted(
            expected_inventory, key=lambda row: row["path"]):
        raise ScorerFitCorpusV2DesignError(
            "path-projection latent index inventory changed")
    for record in expected_inventory:
        path = _pin_generated(
            root, record["path"], label="path-projection base latent shard")
        digest, byte_count = _sha256_regular_file(
            path, label="path-projection base latent shard")
        if digest != record["sha256"] or byte_count != record["byte_count"]:
            raise ScorerFitCorpusV2DesignError(
                "path-projection base latent shard binding changed")
    if (sum(row["byte_count"] for row in expected_inventory)
            != bundle["total_latent_storage_bytes"]):
        raise ScorerFitCorpusV2DesignError(
            "path-projection base latent storage count changed")
    return copy.deepcopy(bundle)


def audit_encoder_import_correction_prelatent_absence(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    rows = _expected_encoder_import_correction_absence_rows()
    for row in rows:
        path = _pin_generated(root, row["path"], label="pre-latent correction absence")
        if path.exists() or path.is_symlink():
            raise ScorerFitCorpusV2DesignError(
                f"pre-latent output predates encoder correction: {row['path']}")
    return rows


def audit_encoder_compute_dtype_correction_prelatent_absence(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    rows = _expected_encoder_compute_dtype_correction_absence_rows()
    for row in rows:
        path = _pin_generated(
            root, row["path"], label="compute-dtype correction absence")
        if path.exists() or path.is_symlink():
            raise ScorerFitCorpusV2DesignError(
                f"pre-latent output predates dtype correction: {row['path']}")
    return rows


def audit_encoder_path_projection_correction_downstream_absence(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    rows = _expected_encoder_path_projection_correction_absence_rows()
    for row in rows:
        path = _pin_generated(
            root, row["path"], label="path-projection downstream absence")
        if path.exists() or path.is_symlink():
            raise ScorerFitCorpusV2DesignError(
                f"downstream output predates path correction: {row['path']}")
    return rows


def audit_encoder_path_projection_transaction_artifacts_absent(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    """Require no PREPARED, backup, or COMPLETE state before issuance."""

    rows = _expected_encoder_path_projection_transaction_absence_rows()
    for row in rows:
        path = _pin_generated(
            root, row["path"], label="path-projection transaction absence")
        if path.exists() or path.is_symlink():
            raise ScorerFitCorpusV2DesignError(
                "single-shard transaction predates path correction: "
                f"{row['path']}")
    return rows


def audit_branch_redrive_projection_correction_downstream_absence(
        *, root: Path = ROOT) -> list[dict[str, Any]]:
    """Require no post-branch-corpus latent, training, or transfer output."""

    rows = _expected_branch_redrive_projection_correction_absence_rows()
    for row in rows:
        path = _pin_generated(
            root, row["path"],
            label="branch-redrive correction downstream absence")
        if path.exists() or path.is_symlink():
            raise ScorerFitCorpusV2DesignError(
                "downstream output predates branch-redrive correction: "
                f"{row['path']}")
    return rows


def audit_v2_runtime_outputs_absent(
        *, root: Path = ROOT, phase: str = "design",
        ) -> list[dict[str, Any]]:
    """Check the exact phase-appropriate V2 output-absence surface.

    ``design`` additionally requires all preoutcome V2 artifacts and the
    successor contract to be absent. ``successor_contract`` permits the five
    issued preoutcome artifacts but still requires the contract itself to be
    absent. ``post_contract_pre_branch`` permits both while retaining every
    branch/frame/latent/training/qualification/development absence gate.
    """

    if phase not in {"design", "successor_contract", "post_contract_pre_branch"}:
        raise ScorerFitCorpusV2DesignError("unknown V2 absence-audit phase")
    exact_paths = list(V2_ALWAYS_ABSENT_PATHS)
    exact_paths.extend(V2_RUNTIME_OUTPUT_PATHS)
    if phase == "design":
        exact_paths.extend(V2_PREOUTCOME_ARTIFACT_PATHS)
    if phase in {"design", "successor_contract"}:
        exact_paths.append(V2_SUCCESSOR_CONTRACT_PATH)
    exact_paths.extend(Path(path) for path in
                       GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_EXACT_FILE_PATHS)
    directory_paths = list(V2_FUTURE_OUTPUT_DIRECTORIES)
    directory_paths.extend(Path(path) for path in
                           GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_DIRECTORY_ROOTS)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, paths in (("file", exact_paths), ("directory", directory_paths)):
        for relative in paths:
            key = str(relative)
            if key in seen:
                continue
            seen.add(key)
            path = _pin_generated(root, relative, label="V2 outcome absence")
            if path.exists() or path.is_symlink():
                raise ScorerFitCorpusV2DesignError(
                    f"V2 outcome-bearing path predates design: {relative}")
            rows.append({"path": key, "expected_kind": kind, "exists": False})
    patterns = list(V2_FUTURE_OUTPUT_GLOBS)
    patterns.extend(Path(path) for path in
                    GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_GLOB_PATTERNS)
    for pattern_raw in patterns:
        pattern = Path(pattern_raw)
        parent = _pin_generated(
            root, pattern.parent / "__v2_absence_probe__",
            label="V2 outcome glob absence").parent
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise ScorerFitCorpusV2DesignError(
                    "V2 outcome glob parent changed")
            if any(fnmatchcase(entry.name, pattern.name) for entry in parent.iterdir()):
                raise ScorerFitCorpusV2DesignError(
                    f"V2 outcome-bearing glob predates design: {pattern}")
        rows.append({
            "path": str(pattern), "expected_kind": "glob", "exists": False,
        })
    return rows


def audit_v2_outcome_outputs_absent(*, root: Path = ROOT) -> list[dict[str, Any]]:
    """Compatibility name for the complete pre-design absence audit."""

    return audit_v2_runtime_outputs_absent(root=root, phase="design")


def _exclusive_json(path: Path, value: Mapping[str, Any], *, label: str) -> bytes:
    raw = _pretty_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise ScorerFitCorpusV2DesignError(
            f"cannot exclusively create {label}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Never unlink a possibly durable authority after exclusive creation.
        raise
    return raw


def _fsync_parent_directory(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_json_atomic_no_overwrite(
        path: Path, staged_path: Path, value: Mapping[str, Any], *,
        label: str, recover_nonexact_staged: bool,
        ) -> bytes:
    """Durably link a complete staged authority into an absent final path."""

    if path.parent != staged_path.parent or path == staged_path:
        raise ScorerFitCorpusV2DesignError(
            f"{label} staged publication escaped its parent")
    raw = _pretty_json_bytes(value)

    final_present = path.exists() or path.is_symlink()
    staged_present = staged_path.exists() or staged_path.is_symlink()
    if final_present:
        if (path.is_symlink() or not path.is_file()
                or stat.S_IMODE(path.stat().st_mode) != 0o444
                or path.read_bytes() != raw):
            raise ScorerFitCorpusV2DesignError(
                f"{label} immutable final collision")
        if staged_present:
            if (staged_path.is_symlink() or not staged_path.is_file()
                    or stat.S_IMODE(staged_path.stat().st_mode) != 0o444
                    or staged_path.read_bytes() != raw
                    or staged_path.stat().st_dev != path.stat().st_dev
                    or staged_path.stat().st_ino != path.stat().st_ino):
                raise ScorerFitCorpusV2DesignError(
                    f"{label} staged cleanup state changed")
            # The previous process may have stopped immediately after link(2)
            # and before the first directory fsync.  Make the final name
            # durable before removing the only other name for these bytes.
            _fsync_parent_directory(path)
            staged_path.unlink()
            _fsync_parent_directory(path)
        return raw

    if staged_present:
        staged_exact = bool(
            not staged_path.is_symlink() and staged_path.is_file()
            and stat.S_IMODE(staged_path.stat().st_mode) == 0o444
            and staged_path.read_bytes() == raw)
        if not staged_exact:
            if not recover_nonexact_staged or staged_path.is_symlink():
                raise ScorerFitCorpusV2DesignError(
                    f"{label} staged bytes are partial or nonexact")
            staged_path.unlink()
            _fsync_parent_directory(path)
            staged_present = False
    if not staged_present:
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
        try:
            descriptor = os.open(staged_path, flags, 0o600)
        except OSError as exc:
            raise ScorerFitCorpusV2DesignError(
                f"cannot stage {label}") from exc
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise ScorerFitCorpusV2DesignError(
                        f"cannot fully stage {label}")
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    staged_stat = staged_path.stat()
    if (staged_path.is_symlink() or not staged_path.is_file()
            or stat.S_IMODE(staged_stat.st_mode) != 0o444
            or staged_path.read_bytes() != raw):
        raise ScorerFitCorpusV2DesignError(
            f"{label} staged publication changed")
    # A staged file may have survived an interruption after its original
    # fsync.  Reopen and fsync the exact bytes before linking so replay does
    # not rely on the durability of the interrupted process.
    staged_descriptor = os.open(
        staged_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        reopened = os.fstat(staged_descriptor)
        if (reopened.st_dev != staged_stat.st_dev
                or reopened.st_ino != staged_stat.st_ino
                or stat.S_IMODE(reopened.st_mode) != 0o444
                or not stat.S_ISREG(reopened.st_mode)):
            raise ScorerFitCorpusV2DesignError(
                f"{label} staged publication changed before fsync")
        os.fsync(staged_descriptor)
    finally:
        os.close(staged_descriptor)
    try:
        os.link(staged_path, path, follow_symlinks=False)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ScorerFitCorpusV2DesignError(
            f"cannot atomically publish {label}") from exc
    _fsync_parent_directory(path)
    final_stat = path.stat()
    if (path.is_symlink() or not path.is_file()
            or final_stat.st_dev != staged_stat.st_dev
            or final_stat.st_ino != staged_stat.st_ino
            or stat.S_IMODE(final_stat.st_mode) != 0o444
            or path.read_bytes() != raw):
        raise ScorerFitCorpusV2DesignError(
            f"{label} atomic no-overwrite publication changed")
    staged_path.unlink()
    _fsync_parent_directory(path)
    return raw


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ScorerFitCorpusV2DesignError(f"{label} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScorerFitCorpusV2DesignError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ScorerFitCorpusV2DesignError(f"{label} is not an object")
    return value, raw


def _artifact_path_from_binding(
        binding: Mapping[str, Any], *, root: Path, label: str) -> Path:
    relative = Path(str(binding.get("path", "")))
    if any(root_path == relative or root_path in relative.parents
           for root_path in MANAGED_GENERATED_ROOTS):
        return _pin_generated(root, relative, label=label)
    return _pin_relative(root, relative, label=label)


def _validate_exact_json_binding(
        binding: Mapping[str, Any], *, root: Path, label: str,
        ) -> dict[str, Any]:
    path = _artifact_path_from_binding(binding, root=root, label=label)
    payload, raw = _load_json(path, label=label)
    self_key = str(binding.get("self_digest_key", ""))
    canonicalization = binding.get(
        "self_digest_canonicalization", "COMPACT_CANONICAL_JSON")
    digest_payload = _without(payload, self_key)
    if canonicalization == "COMPACT_CANONICAL_JSON":
        recomputed_self_digest = canonical_digest(digest_payload)
    elif canonicalization == "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS":
        recomputed_self_digest = hashlib.sha256(json.dumps(
            digest_payload, sort_keys=True, ensure_ascii=True,
            allow_nan=False).encode("utf-8")).hexdigest()
    else:
        raise ScorerFitCorpusV2DesignError(
            f"unknown historical self-digest convention: {label}")
    if (len(raw) != binding.get("byte_count")
            or hashlib.sha256(raw).hexdigest() != binding.get("raw_sha256")
            or payload.get("schema") != binding.get("schema")
            or ("semantic_status" in binding
                and payload.get("status") != binding.get("semantic_status"))
            or payload.get(self_key) != binding.get("self_digest")
            or recomputed_self_digest != binding.get("self_digest")):
        raise ScorerFitCorpusV2DesignError(
            f"immutable historical artifact changed: {label}")
    return payload


def validate_historical_predecessor_artifacts(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen only the exact frozen lineage paths and validate them solve-free.

    This is custody validation for issuance.  None of the reopened JSON is
    supplied to the mask classifier or state selector.
    """

    active = _validate_exact_json_binding(
        ACTIVE_GLOBAL_AMENDMENT_BINDING, root=root,
        label="active global exact amendment")
    try:
        GLOBAL_AUTHORITY.validate_preplan_integration_correction(
            active, root=root, validate_live_authorities=False,
            require_runtime_outputs_absent=False)
    except Exception as exc:
        raise ScorerFitCorpusV2DesignError(
            "active global exact amendment historical validation failed") from exc
    if (active.get("source_repository_commit")
            != TERMINAL_SOURCE_REPOSITORY_COMMIT
            or active.get("scientific_contract_bindings")
            != PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS
            or active.get("preoutcome_input_bindings", {}).get(
                "candidate_outcomes_consumed") is not False):
        raise ScorerFitCorpusV2DesignError(
            "active global exact amendment scientific binding changed")

    plan = _validate_exact_json_binding(
        GLOBAL_MODEL_PLAN_BINDING, root=root, label="global exact model plan")
    nested_plan = plan.get("model_execution_plan")
    if (plan.get("status") != "FROZEN_PRE_SOLVE"
            or plan.get("source_repository_commit")
            != TERMINAL_SOURCE_REPOSITORY_COMMIT
            or plan.get("execution_amendment_digest")
            != ACTIVE_GLOBAL_AMENDMENT_DIGEST
            or plan.get("model_execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or not isinstance(nested_plan, Mapping)
            or nested_plan.get("model_digest") != GLOBAL_EXACT_MODEL_DIGEST
            or nested_plan.get("execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or nested_plan.get("candidate_outcomes_consumed") is not False
            or nested_plan.get("external_combination_enumeration") is not False
            or nested_plan.get("performance_gate") is not None
            or canonical_digest(_without(
                nested_plan, "execution_plan_digest"))
            != MODEL_EXECUTION_PLAN_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "global exact model plan semantic binding changed")

    terminal = _validate_exact_json_binding(
        GLOBAL_TERMINAL_INFEASIBILITY_BINDING, root=root,
        label="global exact terminal infeasibility")
    result = terminal.get("model_execution_result")
    exact = result.get("exact_model_result") if isinstance(result, Mapping) else None
    if (terminal.get("status") != "INFEASIBLE"
            or terminal.get("execution_amendment_digest")
            != ACTIVE_GLOBAL_AMENDMENT_DIGEST
            or terminal.get("global_exact_model_plan_digest")
            != GLOBAL_MODEL_PLAN_DIGEST
            or terminal.get("model_execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or terminal.get("exact_infeasibility_proved") is not True
            or terminal.get("scientific_conditions_relaxed") is not False
            or terminal.get("candidate_outcomes_consumed") is not False
            or terminal.get("branch_labels_read") is not False
            or terminal.get("scorer_or_predictor_accessed") is not False
            or not isinstance(result, Mapping)
            or result.get("model_digest") != GLOBAL_EXACT_MODEL_DIGEST
            or result.get("execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or result.get("status") != "EXACT_GLOBAL_ALLOCATION_INFEASIBLE"
            or result.get("candidate_outcomes_consumed") is not False
            or result.get("materialized_allocation") is not None
            or result.get("execution_result_digest")
            != terminal.get("model_execution_result_digest")
            or canonical_digest(_without(result, "execution_result_digest"))
            != result.get("execution_result_digest")
            or not isinstance(exact, Mapping)
            or exact.get("model_digest") != GLOBAL_EXACT_MODEL_DIGEST
            or exact.get("infeasibility_digest") != EXACT_INFEASIBILITY_DIGEST
            or exact.get("candidate_outcomes_consumed") is not False
            or canonical_digest(_without(exact, "infeasibility_digest"))
            != EXACT_INFEASIBILITY_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "global exact terminal infeasibility semantic binding changed")
    try:
        from lewm.oracle import go2_small_completion_global_exact_model_v1 as model
        model.validate_solver_runtime_identity_record(
            nested_plan.get("solver_runtime_identity", {}))
        model.validate_solver_runtime_identity_record(
            exact.get("solver_runtime_identity", {}))
    except Exception as exc:
        raise ScorerFitCorpusV2DesignError(
            "frozen solve-free runtime identity changed") from exc

    try:
        predecessor_lineage = GLOBAL_AUTHORITY.load_predecessor_lineage(root=root)
    except Exception as exc:
        raise ScorerFitCorpusV2DesignError(
            "immutable V1/V2 failure validation failed") from exc
    if (len(predecessor_lineage) != len(IMMUTABLE_V1_V2_FAILURE_BINDINGS)
            or {row["role"] for row in IMMUTABLE_V1_V2_FAILURE_BINDINGS}
            != set(predecessor_lineage)):
        raise ScorerFitCorpusV2DesignError(
            "immutable V1/V2 lineage coverage changed")

    prior_payloads: dict[str, dict[str, Any]] = {}
    for binding in PRIOR_PREOUTCOME_FAILURE_BINDINGS:
        prior_payloads[str(binding["role"])] = _validate_exact_json_binding(
            binding, root=root, label=str(binding["role"]))
    frozen_predictor = prior_payloads[
        "candidate_allocation_infeasibility"].get(
            "frozen_predictor_qualification")
    expected_predictor_projection = {
        key: value for key, value in FROZEN_PREDICTOR_QUALIFICATION.items()
        if key != "result_report"
    }
    if frozen_predictor != expected_predictor_projection:
        raise ScorerFitCorpusV2DesignError(
            "frozen predictor qualification lineage changed")

    report_binding = FROZEN_PREDICTOR_QUALIFICATION["result_report"]
    report_path = _artifact_path_from_binding(
        report_binding, root=root, label="predictor qualification report")
    if not report_path.is_file() or report_path.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "predictor qualification report is unavailable")
    report_raw = report_path.read_bytes()
    if (len(report_raw) != report_binding["byte_count"]
            or hashlib.sha256(report_raw).hexdigest()
            != report_binding["raw_sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "predictor qualification report binding changed")
    return copy.deepcopy(PREDECESSOR_VALIDATION_PROJECTION)


def load_rotation_mask_classification(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_source: bool = True) -> dict[str, Any]:
    expected = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "mask classification logical path changed")
    payload, raw = _load_json(expected, label="mask classification")
    value = validate_rotation_mask_classification(
        payload, validate_live_source=validate_live_source, root=root)
    rotation_mask_classification_artifact_binding(value, raw)
    return value


def issue_rotation_mask_classification(
        path: Path | None = None, *, root: Path = ROOT) -> dict[str, Any]:
    expected = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "mask classification logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "mask classification parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_rotation_mask_classification(root=root)
    commit, sources = clean_source_authority(root=root)
    predecessor = validate_historical_predecessor_artifacts(root=root)
    before = audit_v2_outcome_outputs_absent(root=root)
    value = build_rotation_mask_classification(
        source_repository_commit=commit, source_bindings=sources,
        predecessor_validation=predecessor)
    if (before != audit_v2_outcome_outputs_absent(root=root)
            or predecessor
            != validate_historical_predecessor_artifacts(root=root)):
        raise ScorerFitCorpusV2DesignError(
            "V2 predecessor or outcome absence changed before classification install")
    _exclusive_json(expected, value, label="mask classification")
    return load_rotation_mask_classification(root=root)


def load_design_amendment(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_source: bool = True) -> dict[str, Any]:
    expected = _pin_generated(root, DESIGN_RELATIVE_PATH, label="design amendment")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError("design amendment logical path changed")
    payload, raw = _load_json(expected, label="design amendment")
    value = validate_design_amendment(
        payload, validate_live_source=validate_live_source, root=root)
    design_amendment_artifact_binding(value, raw)
    classification = load_rotation_mask_classification(
        root=root, validate_live_source=validate_live_source)
    classification_path = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    _classification_payload, classification_raw = _load_json(
        classification_path, label="mask classification")
    if value["rotation_mask_classification"] != (
            rotation_mask_classification_artifact_binding(
                classification, classification_raw)):
        raise ScorerFitCorpusV2DesignError(
            "design does not bind the installed classification")
    return value


def issue_design_amendment(
        path: Path | None = None, *, root: Path = ROOT) -> dict[str, Any]:
    expected = _pin_generated(root, DESIGN_RELATIVE_PATH, label="design amendment")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError("design amendment logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError("design amendment parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_design_amendment(root=root)
    classification = load_rotation_mask_classification(root=root)
    classification_path = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    _classification_payload, classification_raw = _load_json(
        classification_path, label="mask classification")
    classification_binding = rotation_mask_classification_artifact_binding(
        classification, classification_raw)
    commit, sources = clean_source_authority(root=root)
    predecessor = validate_historical_predecessor_artifacts(root=root)
    before = audit_v2_outcome_outputs_absent(root=root)
    value = build_design_amendment(
        source_repository_commit=commit, source_bindings=sources,
        rotation_mask_classification_binding=classification_binding,
        predecessor_validation=predecessor)
    if (before != audit_v2_outcome_outputs_absent(root=root)
            or predecessor
            != validate_historical_predecessor_artifacts(root=root)):
        raise ScorerFitCorpusV2DesignError(
            "V2 predecessor or outcome absence changed before design install")
    _exclusive_json(expected, value, label="design amendment")
    return load_design_amendment(root=root)


def _load_issued_design_authority_for_source_correction(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen only the two issued artifacts with historical source replay off."""

    classification = load_rotation_mask_classification(
        root=root, validate_live_source=False)
    classification_path = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH,
        label="immutable issued mask classification")
    _payload, classification_raw = _load_json(
        classification_path, label="immutable issued mask classification")
    design = load_design_amendment(root=root, validate_live_source=False)
    design_path = _pin_generated(
        root, DESIGN_RELATIVE_PATH, label="immutable issued design amendment")
    _payload, design_raw = _load_json(
        design_path, label="immutable issued design amendment")
    return validate_immutable_issued_design_authority({
        "rotation_mask_classification_payload": classification,
        "rotation_mask_classification_binding":
            rotation_mask_classification_artifact_binding(
                classification, classification_raw),
        "design_amendment_payload": design,
        "design_amendment_binding": design_amendment_artifact_binding(
            design, design_raw),
    })


def load_preselection_source_correction_v1(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = False,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    expected = _pin_generated(
        root, SOURCE_CORRECTION_V1_RELATIVE_PATH,
        label="preselection source correction V1")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction logical path changed")
    payload, raw = _load_json(
        expected, label="preselection source correction V1")
    correction = validate_preselection_source_correction_v1(
        payload, root=root,
        validate_live_authorities=validate_live_authorities,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    preselection_source_correction_v1_artifact_binding(correction, raw)
    return correction


def issue_preselection_source_correction_v1(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Replay immutable V1 only; chained-V2 source cannot reissue it."""

    expected = _pin_generated(
        root, SOURCE_CORRECTION_V1_RELATIVE_PATH,
        label="preselection source correction V1")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_preselection_source_correction_v1(
            root=root, validate_live_authorities=False)
    del source_repository_commit
    raise ScorerFitCorpusV2DesignError(
        "immutable preselection source correction V1 cannot be reissued")


def _load_immutable_preselection_source_correction_v1(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen V1 as historical custody without replaying its live source."""

    path = _pin_generated(
        root, SOURCE_CORRECTION_V1_RELATIVE_PATH,
        label="immutable preselection source correction V1")
    payload, raw = _load_json(
        path, label="immutable preselection source correction V1")
    binding = preselection_source_correction_v1_artifact_binding(payload, raw)
    return validate_immutable_preselection_source_correction_v1({
        "payload": payload,
        "binding": binding,
    })


def load_preselection_source_correction_v2(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = False,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    expected = _pin_generated(
        root, SOURCE_CORRECTION_V2_RELATIVE_PATH,
        label="preselection source correction V2")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 logical path changed")
    payload, raw = _load_json(
        expected, label="preselection source correction V2")
    correction = validate_preselection_source_correction_v2(
        payload, root=root,
        validate_live_authorities=validate_live_authorities,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    preselection_source_correction_v2_artifact_binding(correction, raw)
    return correction


def issue_preselection_source_correction_v2(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Replay immutable V2 only; the final correction cannot reissue it."""

    expected = _pin_generated(
        root, SOURCE_CORRECTION_V2_RELATIVE_PATH,
        label="preselection source correction V2")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction V2 parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_preselection_source_correction_v2(
            root=root, validate_live_authorities=False)
    del source_repository_commit
    raise ScorerFitCorpusV2DesignError(
        "immutable preselection source correction V2 cannot be reissued")


def _load_immutable_preselection_source_correction_v2(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen V2 as historical custody without replaying its live source."""

    path = _pin_generated(
        root, SOURCE_CORRECTION_V2_RELATIVE_PATH,
        label="immutable preselection source correction V2")
    payload, raw = _load_json(
        path, label="immutable preselection source correction V2")
    binding = preselection_source_correction_v2_artifact_binding(payload, raw)
    return validate_immutable_preselection_source_correction_v2({
        "payload": payload,
        "binding": binding,
    })


def load_preselection_source_correction(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    expected = _pin_generated(
        root, SOURCE_CORRECTION_RELATIVE_PATH,
        label="preselection structural-validation correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection structural-validation correction logical path changed")
    payload, raw = _load_json(
        expected, label="preselection structural-validation correction")
    correction = validate_preselection_source_correction(
        payload, root=root,
        validate_live_authorities=validate_live_authorities,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    preselection_source_correction_artifact_binding(correction, raw)
    return correction


def issue_preselection_source_correction(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Replay immutable correction 5206 only; never reissue it."""

    expected = _pin_generated(
        root, SOURCE_CORRECTION_RELATIVE_PATH,
        label="preselection structural-validation correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection structural-validation correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "preselection structural-validation correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_preselection_source_correction(
            root=root, validate_live_authorities=False)
    del source_repository_commit
    raise ScorerFitCorpusV2DesignError(
        "immutable active preselection source correction cannot be reissued")


def _load_immutable_active_preselection_source_correction(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen exact correction 5206 without replaying its historical source."""

    path = _pin_generated(
        root, SOURCE_CORRECTION_RELATIVE_PATH,
        label="immutable active preselection source correction")
    if (not path.is_file() or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "immutable active preselection source correction mode changed")
    payload, raw = _load_json(
        path, label="immutable active preselection source correction")
    binding = preselection_source_correction_artifact_binding(payload, raw)
    return validate_immutable_active_preselection_source_correction({
        "payload": payload,
        "binding": binding,
    })


def load_manifest_replay_correction(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_successor_and_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    expected = _pin_generated(
        root, MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH,
        label="post-install manifest-replay correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction logical path changed")
    if (not expected.is_file() or expected.is_symlink()
            or stat.S_IMODE(expected.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction mode changed")
    payload, raw = _load_json(
        expected, label="post-install manifest-replay correction")
    correction = validate_manifest_replay_correction(
        payload, root=root,
        validate_live_authorities=validate_live_authorities,
        require_successor_and_runtime_outputs_absent=
            require_successor_and_runtime_outputs_absent)
    manifest_replay_correction_artifact_binding(correction, raw)
    return correction


def issue_manifest_replay_correction(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Install the replay correction after exact five-file custody checks."""

    expected = _pin_generated(
        root, MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH,
        label="post-install manifest-replay correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "manifest-replay correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_manifest_replay_correction(root=root)

    commit, sources = clean_source_authority(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "requested manifest-replay correction commit is not live")
    immutable = _load_immutable_active_preselection_source_correction(root=root)
    first_installed = validate_installed_full_bank_v2_preoutcome_artifacts(
        root=root)
    first_absence = audit_v2_runtime_outputs_absent(
        root=root, phase="successor_contract")
    correction = build_manifest_replay_correction(
        source_repository_commit=commit,
        source_bindings=sources,
        immutable_active_preselection_source_correction=immutable,
        installed_preoutcome_artifact_bindings=first_installed,
        successor_and_runtime_outputs_absent_at_issue=first_absence)
    second_installed = validate_installed_full_bank_v2_preoutcome_artifacts(
        root=root)
    second_absence = audit_v2_runtime_outputs_absent(
        root=root, phase="successor_contract")
    second_commit, second_sources = clean_source_authority(root=root)
    if (first_installed != second_installed
            or first_absence != second_absence
            or second_absence
            != correction["successor_and_runtime_outputs_absent_at_issue"]
            or (commit, sources) != (second_commit, second_sources)
            or immutable
            != _load_immutable_active_preselection_source_correction(
                root=root)):
        raise ScorerFitCorpusV2DesignError(
            "source, 5206 lineage, installed manifests, or absence changed "
            "before replay-correction install")
    _exclusive_json(
        expected, correction, label="post-install manifest-replay correction")
    return load_manifest_replay_correction(root=root)


def load_encoder_import_correction_for_consumption(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    """Load the immutable correction; later corpus refreshes remain admissible."""

    expected = _pin_generated(
        root, ENCODER_IMPORT_CORRECTION_RELATIVE_PATH,
        label="post-smoke encoder-import correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction logical path changed")
    if (not expected.is_file() or expected.is_symlink()
            or stat.S_IMODE(expected.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction mode changed")
    payload, raw = _load_json(expected, label="post-smoke encoder-import correction")
    correction = validate_encoder_import_correction(
        payload, root=root, validate_live_authorities=validate_live_authorities,
        require_failure_boundary_live=require_failure_boundary_live)
    encoder_import_correction_artifact_binding(correction, raw)
    return correction


def issue_encoder_import_correction(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Issue the one post-smoke shim authority before any latent is written."""

    expected = _pin_generated(
        root, ENCODER_IMPORT_CORRECTION_RELATIVE_PATH,
        label="post-smoke encoder-import correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "encoder-import correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_encoder_import_correction_for_consumption(root=root)

    commit, sources = clean_source_authority(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "requested encoder-import correction commit is not live")
    immutable_replay = _load_immutable_manifest_replay_correction(root=root)
    successor = _load_immutable_successor_scorer_contract_binding(root=root)
    dev_transition = _dev_encoder_source_transition(root=root)
    test_transitions = _focused_test_source_transitions(root=root)
    smoke_binding, corpus_binding = (
        _validate_live_encoder_import_failure_receipts(root=root))
    first_absence = audit_encoder_import_correction_prelatent_absence(root=root)
    correction = build_encoder_import_correction(
        source_repository_commit=commit,
        source_bindings=sources,
        immutable_manifest_replay_correction=immutable_replay,
        immutable_successor_scorer_contract_binding=successor,
        dev_encoder_source_transition=dev_transition,
        focused_test_source_transitions=test_transitions,
        branch_smoke_binding=smoke_binding,
        branch_corpus_binding=corpus_binding,
        prelatent_outputs_absent_at_issue=first_absence,
    )

    second_commit, second_sources = clean_source_authority(root=root)
    second_replay = _load_immutable_manifest_replay_correction(root=root)
    second_successor = _load_immutable_successor_scorer_contract_binding(root=root)
    second_dev = _dev_encoder_source_transition(root=root)
    second_tests = _focused_test_source_transitions(root=root)
    second_smoke, second_corpus = (
        _validate_live_encoder_import_failure_receipts(root=root))
    second_absence = audit_encoder_import_correction_prelatent_absence(root=root)
    if ((commit, sources) != (second_commit, second_sources)
            or immutable_replay != second_replay
            or successor != second_successor
            or dev_transition != second_dev
            or test_transitions != second_tests
            or (smoke_binding, corpus_binding) != (second_smoke, second_corpus)
            or first_absence != second_absence):
        raise ScorerFitCorpusV2DesignError(
            "source, lineage, smoke metadata, or pre-latent absence changed "
            "before encoder-import correction install")
    _exclusive_json(
        expected, correction, label="post-smoke encoder-import correction")
    return load_encoder_import_correction_for_consumption(
        root=root, require_failure_boundary_live=True)


def load_encoder_compute_dtype_correction_for_consumption(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    """Load the chained FP32 correction without pinning refreshed receipts."""

    expected = _pin_generated(
        root, ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH,
        label="encoder-compute-dtype correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction logical path changed")
    if (not expected.is_file() or expected.is_symlink()
            or stat.S_IMODE(expected.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction mode changed")
    payload, raw = _load_json(
        expected, label="encoder-compute-dtype correction")
    correction = validate_encoder_compute_dtype_correction(
        payload, root=root, validate_live_authorities=validate_live_authorities,
        require_failure_boundary_live=require_failure_boundary_live)
    encoder_compute_dtype_correction_artifact_binding(correction, raw)
    return correction


def issue_encoder_compute_dtype_correction(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Issue the one chained FP32 correction before any latent is written."""

    expected = _pin_generated(
        root, ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH,
        label="encoder-compute-dtype correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "encoder-compute-dtype correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_encoder_compute_dtype_correction_for_consumption(root=root)

    commit, sources = clean_source_authority(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "requested encoder-compute-dtype correction commit is not live")
    immutable_import = _load_immutable_encoder_import_correction(root=root)
    successor = _load_immutable_successor_scorer_contract_binding(root=root)
    tests = _encoder_compute_dtype_focused_test_source_transitions(root=root)
    failed, dev, stage_a, upstream = (
        _validate_live_encoder_compute_dtype_source_evidence(root=root))
    smoke, corpus = _validate_live_encoder_import_failure_receipts(root=root)
    first_absence = audit_encoder_compute_dtype_correction_prelatent_absence(
        root=root)
    correction = build_encoder_compute_dtype_correction(
        source_repository_commit=commit,
        source_bindings=sources,
        immutable_encoder_import_correction=immutable_import,
        immutable_successor_scorer_contract_binding=successor,
        focused_test_source_transitions=tests,
        branch_smoke_binding=smoke,
        branch_corpus_binding=corpus,
        failed_encoder_source_binding=failed,
        unchanged_dev_encoder_source_binding=dev,
        unchanged_stage_a_fp32_source_binding=stage_a,
        upstream_rope_source_binding=upstream,
        prelatent_outputs_absent_at_issue=first_absence,
    )

    second_commit, second_sources = clean_source_authority(root=root)
    second_import = _load_immutable_encoder_import_correction(root=root)
    second_successor = _load_immutable_successor_scorer_contract_binding(root=root)
    second_tests = _encoder_compute_dtype_focused_test_source_transitions(root=root)
    second_failed, second_dev, second_stage_a, second_upstream = (
        _validate_live_encoder_compute_dtype_source_evidence(root=root))
    second_smoke, second_corpus = (
        _validate_live_encoder_import_failure_receipts(root=root))
    second_absence = audit_encoder_compute_dtype_correction_prelatent_absence(
        root=root)
    if ((commit, sources) != (second_commit, second_sources)
            or immutable_import != second_import
            or successor != second_successor
            or tests != second_tests
            or (failed, dev, stage_a, upstream)
            != (second_failed, second_dev, second_stage_a, second_upstream)
            or (smoke, corpus) != (second_smoke, second_corpus)
            or first_absence != second_absence):
        raise ScorerFitCorpusV2DesignError(
            "source, chained lineage, failure metadata, or absence changed "
            "before encoder-compute-dtype correction install")
    _exclusive_json(
        expected, correction, label="encoder-compute-dtype correction")
    return load_encoder_compute_dtype_correction_for_consumption(
        root=root, require_failure_boundary_live=True)


def load_encoder_path_projection_correction_for_consumption(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    """Load the logical-path correction after its one metadata transition."""

    expected = _pin_generated(
        root, ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH,
        label="encoder-path-projection correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction logical path changed")
    if (not expected.is_file() or expected.is_symlink()
            or stat.S_IMODE(expected.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction mode changed")
    payload, raw = _load_json(
        expected, label="encoder-path-projection correction")
    correction = validate_encoder_path_projection_correction(
        payload, root=root, validate_live_authorities=validate_live_authorities,
        require_failure_boundary_live=require_failure_boundary_live)
    encoder_path_projection_correction_artifact_binding(correction, raw)
    return correction


def issue_encoder_path_projection_correction(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Issue the logical-path correction before zero-new or shard deletion."""

    expected = _pin_generated(
        root, ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH,
        label="encoder-path-projection correction")
    staged = _pin_generated(
        root, ENCODER_PATH_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH,
        label="encoder-path-projection correction staged publication")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "encoder-path-projection correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        correction = load_encoder_path_projection_correction_for_consumption(
            root=root)
        _exclusive_json_atomic_no_overwrite(
            expected, staged, correction,
            label="encoder-path-projection correction",
            recover_nonexact_staged=False)
        return correction

    commit, sources = clean_source_authority(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "requested encoder-path-projection correction commit is not live")
    immutable_dtype = _load_immutable_encoder_compute_dtype_correction(root=root)
    successor = _load_immutable_successor_scorer_contract_binding(root=root)
    tests = _encoder_path_projection_focused_test_source_transitions(root=root)
    failed = _validate_live_encoder_path_projection_failure_source(root=root)
    first_bundle = _validate_live_encoder_path_projection_base_bundle(root=root)
    first_absence = audit_encoder_path_projection_correction_downstream_absence(
        root=root)
    first_transaction_absence = (
        audit_encoder_path_projection_transaction_artifacts_absent(root=root))
    correction = build_encoder_path_projection_correction(
        source_repository_commit=commit,
        source_bindings=sources,
        immutable_encoder_compute_dtype_correction=immutable_dtype,
        immutable_successor_scorer_contract_binding=successor,
        focused_test_source_transitions=tests,
        failed_encoder_source_binding=failed,
        base_smoke_artifact_bundle=first_bundle,
        downstream_outputs_absent_at_issue=first_absence,
        single_shard_regeneration_transaction_artifacts_absent_at_issue=
            first_transaction_absence,
    )

    second_commit, second_sources = clean_source_authority(root=root)
    second_dtype = _load_immutable_encoder_compute_dtype_correction(root=root)
    second_successor = _load_immutable_successor_scorer_contract_binding(root=root)
    second_tests = _encoder_path_projection_focused_test_source_transitions(root=root)
    second_failed = _validate_live_encoder_path_projection_failure_source(root=root)
    second_bundle = _validate_live_encoder_path_projection_base_bundle(root=root)
    second_absence = audit_encoder_path_projection_correction_downstream_absence(
        root=root)
    second_transaction_absence = (
        audit_encoder_path_projection_transaction_artifacts_absent(root=root))
    if ((commit, sources) != (second_commit, second_sources)
            or immutable_dtype != second_dtype
            or successor != second_successor
            or tests != second_tests
            or failed != second_failed
            or first_bundle != second_bundle
            or first_absence != second_absence
            or first_transaction_absence != second_transaction_absence):
        raise ScorerFitCorpusV2DesignError(
            "source, chained lineage, base smoke, or absence changed before "
            "encoder-path-projection correction install")
    _exclusive_json_atomic_no_overwrite(
        expected, staged, correction,
        label="encoder-path-projection correction",
        recover_nonexact_staged=True)
    return load_encoder_path_projection_correction_for_consumption(
        root=root, require_failure_boundary_live=True)


def load_branch_redrive_projection_correction_for_consumption(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_failure_boundary_live: bool = False,
        ) -> dict[str, Any]:
    """Load the correction without reopening mutable partial-corpus bytes."""

    expected = _pin_generated(
        root, BRANCH_REDRIVE_PROJECTION_CORRECTION_RELATIVE_PATH,
        label="branch-redrive projection correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction logical path changed")
    if (not expected.is_file() or expected.is_symlink()
            or stat.S_IMODE(expected.stat().st_mode) != 0o444):
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction mode changed")
    payload, raw = _load_json(
        expected, label="branch-redrive projection correction")
    correction = validate_branch_redrive_projection_correction(
        payload, root=root, validate_live_authorities=validate_live_authorities,
        require_failure_boundary_live=require_failure_boundary_live)
    branch_redrive_projection_correction_artifact_binding(correction, raw)
    return correction


def issue_branch_redrive_projection_correction(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Issue the one source-only correction before corrected missing-row resume."""

    expected = _pin_generated(
        root, BRANCH_REDRIVE_PROJECTION_CORRECTION_RELATIVE_PATH,
        label="branch-redrive projection correction")
    staged = _pin_generated(
        root, BRANCH_REDRIVE_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH,
        label="branch-redrive projection correction staged publication")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "branch-redrive correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        correction = (
            load_branch_redrive_projection_correction_for_consumption(
                root=root))
        _exclusive_json_atomic_no_overwrite(
            expected, staged, correction,
            label="branch-redrive projection correction",
            recover_nonexact_staged=False)
        return correction

    commit, sources = clean_source_authority(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "requested branch-redrive correction commit is not live")
    immutable_path = _load_immutable_encoder_path_projection_correction(
        root=root)
    first_partial, first_invalid, first_smoke = (
        _validate_live_branch_redrive_failure_boundary(root=root))
    first_absence = (
        audit_branch_redrive_projection_correction_downstream_absence(
            root=root))
    correction = build_branch_redrive_projection_correction(
        source_repository_commit=commit,
        source_bindings=sources,
        immutable_encoder_path_projection_correction=immutable_path,
        partial_corpus_failure_boundary=first_partial,
        invalid_attempt_receipt_bindings=first_invalid,
        completed_smoke_boundary=first_smoke,
        downstream_outputs_absent_at_issue=first_absence,
    )

    second_commit, second_sources = clean_source_authority(root=root)
    second_path = _load_immutable_encoder_path_projection_correction(root=root)
    second_partial, second_invalid, second_smoke = (
        _validate_live_branch_redrive_failure_boundary(root=root))
    second_absence = (
        audit_branch_redrive_projection_correction_downstream_absence(
            root=root))
    if ((commit, sources) != (second_commit, second_sources)
            or immutable_path != second_path
            or first_partial != second_partial
            or first_invalid != second_invalid
            or first_smoke != second_smoke
            or first_absence != second_absence):
        raise ScorerFitCorpusV2DesignError(
            "source, chained correction, partial metadata, smoke, or absence "
            "changed before branch-redrive correction install")
    _exclusive_json_atomic_no_overwrite(
        expected, staged, correction,
        label="branch-redrive projection correction",
        recover_nonexact_staged=True)
    # Failure-time partial files may now advance.  Reopen only immutable
    # correction bytes and the current clean source authority.
    return load_branch_redrive_projection_correction_for_consumption(root=root)


def load_active_design_authority(*, root: Path = ROOT) -> dict[str, Any]:
    """Return immutable science plus the mandatory corrected source authority."""

    redrive_correction = (
        load_branch_redrive_projection_correction_for_consumption(root=root))
    immutable_path_correction = (
        validate_immutable_encoder_path_projection_correction(
            redrive_correction[
                "immutable_encoder_path_projection_correction"]))
    path_correction = immutable_path_correction["payload"]
    immutable_dtype_correction = (
        validate_immutable_encoder_compute_dtype_correction(
            path_correction["immutable_encoder_compute_dtype_correction"]))
    dtype_correction = immutable_dtype_correction["payload"]
    immutable_encoder_correction = validate_immutable_encoder_import_correction(
        dtype_correction["immutable_encoder_import_correction"])
    encoder_correction = immutable_encoder_correction["payload"]
    immutable_replay = validate_immutable_manifest_replay_correction(
        encoder_correction["immutable_manifest_replay_correction"])
    replay_correction = immutable_replay["payload"]
    immutable_active = validate_immutable_active_preselection_source_correction(
        replay_correction[
            "immutable_active_preselection_source_correction"])
    correction = immutable_active["payload"]
    immutable_v2 = validate_immutable_preselection_source_correction_v2(
        correction["immutable_preselection_source_correction_v2"])
    immutable_v1 = validate_immutable_preselection_source_correction_v1(
        immutable_v2["payload"][
            "immutable_preselection_source_correction_v1"])
    immutable = immutable_v1["payload"]["immutable_issued_design_authority"]
    classification = immutable["rotation_mask_classification_payload"]
    classification_binding = immutable[
        "rotation_mask_classification_binding"]
    design = immutable["design_amendment_payload"]
    design_binding = immutable["design_amendment_binding"]
    correction_binding = immutable_active["binding"]
    replay_path = _pin_generated(
        root, MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH,
        label="post-install manifest-replay correction")
    _payload, replay_raw = _load_json(
        replay_path, label="post-install manifest-replay correction")
    return {
        "rotation_mask_classification": classification,
        "rotation_mask_classification_binding": classification_binding,
        "design_amendment": design,
        "design_amendment_binding": design_binding,
        "source_correction": correction,
        "source_correction_binding": correction_binding,
        "source_correction_digest": correction[SOURCE_CORRECTION_SELF_KEY],
        "manifest_replay_correction": replay_correction,
        "manifest_replay_correction_binding":
            manifest_replay_correction_artifact_binding(
                replay_correction, replay_raw),
        "manifest_replay_correction_digest": replay_correction[
            MANIFEST_REPLAY_CORRECTION_SELF_KEY],
        "encoder_import_correction": encoder_correction,
        "encoder_import_correction_binding": immutable_encoder_correction[
            "binding"],
        "encoder_import_correction_digest": encoder_correction[
            ENCODER_IMPORT_CORRECTION_SELF_KEY],
        "encoder_compute_dtype_correction": dtype_correction,
        "encoder_compute_dtype_correction_binding": immutable_dtype_correction[
            "binding"],
        "encoder_compute_dtype_correction_digest": dtype_correction[
            ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY],
        "encoder_path_projection_correction": path_correction,
        "encoder_path_projection_correction_binding": immutable_path_correction[
            "binding"],
        "encoder_path_projection_correction_digest": path_correction[
            ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY],
        "branch_redrive_projection_correction": redrive_correction,
        "branch_redrive_projection_correction_binding":
            branch_redrive_projection_correction_artifact_binding(
                redrive_correction, _pretty_json_bytes(redrive_correction)),
        "branch_redrive_projection_correction_digest": redrive_correction[
            BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY],
        "single_shard_regeneration_transaction_contract": copy.deepcopy(
            path_correction[
                "single_shard_regeneration_transaction_contract"]),
        "single_shard_regeneration_transaction_contract_digest":
            path_correction[
                "single_shard_regeneration_transaction_contract_digest"],
        "preissue_single_shard_regeneration_transaction_audit": copy.deepcopy(
            path_correction[
                "preissue_single_shard_regeneration_transaction_audit"]),
        "preissue_single_shard_regeneration_transaction_audit_digest":
            path_correction[
                "preissue_single_shard_regeneration_transaction_audit_digest"],
        "manifest_replay_source_repository_commit": replay_correction[
            "source_repository_commit"],
        "encoder_import_source_repository_commit": encoder_correction[
            "source_repository_commit"],
        "encoder_compute_dtype_source_repository_commit": dtype_correction[
            "source_repository_commit"],
        "encoder_path_projection_source_repository_commit": path_correction[
            "source_repository_commit"],
        "active_source_repository_commit": redrive_correction[
            "source_repository_commit"],
        "active_selector_digest": ACTIVE_SELECTOR_DIGEST,
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "candidate_indices": list(CANDIDATE_INDICES),
        "count_contract": copy.deepcopy(FULL_BANK_COUNT_CONTRACT),
        "completion_ordering_contract": copy.deepcopy(
            COMPLETION_ORDERING_CONTRACT),
        "preserved_scientific_contract_bindings": copy.deepcopy(
            PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS),
        "candidate_outcomes_consumed": False,
    }


__all__ = [
    "ACTIVE_GLOBAL_AMENDMENT_BINDING", "ACTIVE_GLOBAL_AMENDMENT_DIGEST",
    "ACTIVE_SELECTOR_DIGEST", "ASSIGNMENT_COUNT", "CANDIDATE_BANK_DIGEST",
    "CANDIDATE_INDICES", "COMPLETION_ORDER_DOMAIN",
    "COMPLETION_ORDERING_CONTRACT", "DESIGN_RELATIVE_PATH", "DESIGN_SCHEMA",
    "DESIGN_SELF_KEY", "DESIGN_STATUS", "EXACT_INFEASIBILITY_DIGEST",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_PRESERVED_SCIENCE",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_RELATIVE_PATH",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_SCHEMA",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY",
    "BRANCH_REDRIVE_PROJECTION_CORRECTION_STATUS",
    "BRANCH_REDRIVE_PROJECTION_FAILURE_BOUNDARY",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_FOCUSED_TEST_PATHS",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_PRESERVED_SCIENCE",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_SCHEMA",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY",
    "ENCODER_COMPUTE_DTYPE_CORRECTION_STATUS",
    "ENCODER_COMPUTE_DTYPE_FAILURE_BOUNDARY",
    "ENCODER_PATH_PROJECTION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "ENCODER_PATH_PROJECTION_CORRECTION_FOCUSED_TEST_PATHS",
    "ENCODER_PATH_PROJECTION_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
    "ENCODER_PATH_PROJECTION_CORRECTION_PRESERVED_SCIENCE",
    "ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH",
    "ENCODER_PATH_PROJECTION_CORRECTION_STAGED_RELATIVE_PATH",
    "ENCODER_PATH_PROJECTION_CORRECTION_SCHEMA",
    "ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY",
    "ENCODER_PATH_PROJECTION_CORRECTION_STATUS",
    "ENCODER_PATH_PROJECTION_FAILURE_BOUNDARY",
    "ENCODER_PATH_PROJECTION_PREISSUE_SINGLE_SHARD_TRANSACTION_AUDIT",
    "ENCODER_PATH_PROJECTION_SINGLE_SHARD_REGENERATION_TRANSACTION_CONTRACT",
    "ENCODER_PATH_PROJECTION_TRANSACTION_REQUIRED_ABSENT_DIRECTORIES",
    "ENCODER_PATH_PROJECTION_TRANSACTION_REQUIRED_ABSENT_PATHS",
    "ENCODER_IMPORT_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "ENCODER_IMPORT_CORRECTION_FOCUSED_TEST_PATHS",
    "ENCODER_IMPORT_CORRECTION_HISTORICAL_SOURCE_REPOSITORY_COMMIT",
    "ENCODER_IMPORT_CORRECTION_PRESERVED_SCIENCE",
    "ENCODER_IMPORT_CORRECTION_RELATIVE_PATH",
    "ENCODER_IMPORT_CORRECTION_SCHEMA", "ENCODER_IMPORT_CORRECTION_SELF_KEY",
    "ENCODER_IMPORT_CORRECTION_STATUS", "ENCODER_IMPORT_FAILURE_BOUNDARY",
    "EXPECTED_ROTATION_CONSTRAINT_IDS", "EXPECTED_SOURCE_PATHS", "FAMILIES",
    "FROZEN_PREDICTOR_QUALIFICATION", "FULL_BANK_COUNT_CONTRACT",
    "FULL_BANK_V2_SMOKE_REGENERATION_BACKUP_RELATIVE_PATH",
    "FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_RELATIVE_PATH",
    "FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SCHEMA",
    "FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_SELF_KEY",
    "FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STAGED_RELATIVE_PATH",
    "FULL_BANK_V2_SMOKE_REGENERATION_COMPLETE_STATUS",
    "FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_RELATIVE_PATH",
    "FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SCHEMA",
    "FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_SELF_KEY",
    "FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STAGED_RELATIVE_PATH",
    "FULL_BANK_V2_SMOKE_REGENERATION_PREPARED_STATUS",
    "FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_CONTRACT_SCHEMA",
    "FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_CONTRACT_STATUS",
    "FULL_BANK_V2_SMOKE_REGENERATION_TRANSACTION_DIRECTORY_RELATIVE_PATH",
    "GLOBAL_EXACT_MODEL_DIGEST", "GLOBAL_MODEL_PLAN_BINDING",
    "GLOBAL_TERMINAL_INFEASIBILITY_BINDING", "MASK_CLASSIFICATION_RELATIVE_PATH",
    "IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_BINDING",
    "IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST",
    "IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_REPOSITORY_COMMIT",
    "IMMUTABLE_ENCODER_IMPORT_CORRECTION_BINDING",
    "IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST",
    "IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_BINDING",
    "IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST",
    "IMMUTABLE_ENCODER_PATH_PROJECTION_BASE_ARTIFACT_BUNDLE",
    "IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_BINDING",
    "IMMUTABLE_ENCODER_PATH_PROJECTION_CORRECTION_DIGEST",
    "IMMUTABLE_BRANCH_REDRIVE_COMPLETED_SMOKE_BUNDLE",
    "IMMUTABLE_BRANCH_REDRIVE_INVALID_ATTEMPT_RECEIPT_BINDINGS",
    "IMMUTABLE_BRANCH_REDRIVE_PARTIAL_CORPUS_BINDING",
    "INSTALLED_FULL_BANK_V2_PREOUTCOME_ARTIFACT_BINDINGS",
    "ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT",
    "IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST",
    "IMMUTABLE_SOURCE_CORRECTION_V1_SOURCE_REPOSITORY_COMMIT",
    "IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST",
    "IMMUTABLE_SOURCE_CORRECTION_V2_SOURCE_REPOSITORY_COMMIT",
    "IMMUTABLE_SUCCESSOR_SCORER_CONTRACT_BINDING",
    "MASK_CLASSIFICATION_SCHEMA", "MASK_CLASSIFICATION_SELF_KEY",
    "MASK_CLASSIFICATION_STATUS", "ORACLE_V1_2_DIGEST",
    "MANIFEST_REPLAY_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "MANIFEST_REPLAY_CORRECTION_RELATIVE_PATH",
    "MANIFEST_REPLAY_CORRECTION_SCHEMA", "MANIFEST_REPLAY_CORRECTION_SELF_KEY",
    "MANIFEST_REPLAY_CORRECTION_STATUS", "MANIFEST_REPLAY_FAILURE_BOUNDARY",
    "PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS", "PRIOR_PREOUTCOME_FAILURE_BINDINGS",
    "POST_FIX_PRODUCTION_BUNDLE_DRY_RUN", "PREDECESSOR_VALIDATION_PROJECTION",
    "PRESELECTION_ALIAS_FAILURE_BOUNDARY_V1",
    "PRESELECTION_ALIAS_FAILURE_BOUNDARY_V2",
    "PRESELECTION_STRUCTURAL_VALIDATION_FAILURE_BOUNDARY",
    "SIX_OF_TWELVE_SUPERSESSION", "SOURCE_SPECS", "SPLIT_ROLES", "STATE_COUNT",
    "SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "SOURCE_CORRECTION_RELATIVE_PATH", "SOURCE_CORRECTION_SCHEMA",
    "SOURCE_CORRECTION_SELF_KEY", "SOURCE_CORRECTION_STATUS",
    "SOURCE_CORRECTION_V1_ALLOWED_CHANGED_SOURCE_PATHS",
    "SOURCE_CORRECTION_V1_RELATIVE_PATH", "SOURCE_CORRECTION_V1_SCHEMA",
    "SOURCE_CORRECTION_V1_STATUS",
    "SOURCE_CORRECTION_V2_ALLOWED_CHANGED_SOURCE_PATHS",
    "SOURCE_CORRECTION_V2_RELATIVE_PATH", "SOURCE_CORRECTION_V2_SCHEMA",
    "SOURCE_CORRECTION_V2_STATUS",
    "STRATA", "ScorerFitCorpusV2DesignError", "TERMINAL_RECEIPT_DIGEST",
    "TERMINAL_SOURCE_REPOSITORY_COMMIT", "V2_FUTURE_OUTPUT_PATHS",
    "V2_PREOUTCOME_ARTIFACT_PATHS", "V2_RUNTIME_OUTPUT_PATHS",
    "V2_SUCCESSOR_CONTRACT_PATH", "audit_v2_outcome_outputs_absent",
    "audit_branch_redrive_projection_correction_downstream_absence",
    "audit_encoder_compute_dtype_correction_prelatent_absence",
    "audit_encoder_path_projection_correction_downstream_absence",
    "audit_encoder_path_projection_transaction_artifacts_absent",
    "audit_encoder_import_correction_prelatent_absence",
    "audit_v2_runtime_outputs_absent", "build_design_amendment",
    "build_encoder_compute_dtype_correction", "build_encoder_import_correction",
    "build_encoder_path_projection_correction",
    "build_branch_redrive_projection_correction",
    "build_full_bank_v2_smoke_regeneration_complete_receipt",
    "build_full_bank_v2_smoke_regeneration_prepared_receipt",
    "build_manifest_replay_correction", "builder_default_canonical_digest",
    "build_preselection_source_correction",
    "build_preselection_source_correction_v1",
    "build_preselection_source_correction_v2",
    "build_rotation_mask_classification", "canonical_digest",
    "clean_source_authority", "completion_order_key", "completion_order_material",
    "design_amendment_artifact_binding",
    "encoder_compute_dtype_correction_artifact_binding",
    "encoder_path_projection_correction_artifact_binding",
    "branch_redrive_projection_correction_artifact_binding",
    "full_bank_v2_smoke_regeneration_complete_receipt_artifact_binding",
    "full_bank_v2_smoke_regeneration_prepared_receipt_artifact_binding",
    "encoder_import_correction_artifact_binding", "issue_design_amendment",
    "issue_encoder_compute_dtype_correction",
    "issue_encoder_path_projection_correction",
    "issue_branch_redrive_projection_correction",
    "issue_encoder_import_correction",
    "issue_manifest_replay_correction",
    "issue_preselection_source_correction",
    "issue_preselection_source_correction_v1",
    "issue_preselection_source_correction_v2",
    "issue_rotation_mask_classification", "load_active_design_authority",
    "load_design_amendment",
    "load_encoder_compute_dtype_correction_for_consumption",
    "load_encoder_path_projection_correction_for_consumption",
    "load_branch_redrive_projection_correction_for_consumption",
    "load_full_bank_v2_smoke_regeneration_complete_receipt",
    "load_full_bank_v2_smoke_regeneration_prepared_receipt",
    "load_encoder_import_correction_for_consumption",
    "load_manifest_replay_correction",
    "load_preselection_source_correction",
    "load_preselection_source_correction_v1",
    "load_preselection_source_correction_v2",
    "load_rotation_mask_classification",
    "manifest_replay_correction_artifact_binding",
    "preselection_source_correction_artifact_binding",
    "preselection_source_correction_v1_artifact_binding",
    "preselection_source_correction_v2_artifact_binding",
    "rotation_mask_classification_artifact_binding", "validate_design_amendment",
    "validate_immutable_active_preselection_source_correction",
    "validate_immutable_manifest_replay_correction",
    "validate_immutable_issued_design_authority",
    "validate_immutable_preselection_source_correction_v1",
    "validate_immutable_preselection_source_correction_v2",
    "validate_historical_predecessor_artifacts",
    "validate_installed_full_bank_v2_preoutcome_artifacts",
    "validate_encoder_compute_dtype_correction",
    "validate_encoder_path_projection_correction",
    "validate_branch_redrive_projection_correction",
    "validate_encoder_path_projection_single_shard_regeneration_transaction_contract",
    "validate_full_bank_v2_smoke_regeneration_complete_receipt",
    "validate_full_bank_v2_smoke_regeneration_prepared_receipt",
    "validate_encoder_import_correction",
    "validate_immutable_encoder_import_correction",
    "validate_immutable_encoder_compute_dtype_correction",
    "validate_immutable_encoder_path_projection_correction",
    "validate_manifest_replay_correction",
    "validate_preselection_source_correction",
    "validate_preselection_source_correction_v1",
    "validate_preselection_source_correction_v2",
    "validate_rotation_mask_classification",
]

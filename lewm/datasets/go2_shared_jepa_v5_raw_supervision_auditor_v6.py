"""Standalone V6 auditor for the Shared-JEPA V5 raw-supervision dataset.

The module owns its filesystem, JSON, join, population, array, replay, and
publication engine. It retains no legacy auditor or builder module and exposes
one fixed exact entry only after dual V6 source-review authorization.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import select
import secrets
import stat
import struct
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "lewm_worlds") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    EVIDENCE_SCHEMA,
    RASTER_SCHEMA,
    ObservableCameraRayEvidenceV4,
    rasterize_observable_camera_ray_evidence_v4,
)
from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan_v5 import (
    DATASET_MANIFEST_FILE_SHA256,
    DATASET_ROWS_FILE_SHA256,
    DEVELOPMENT_ROLES,
    SOURCE_INDEX_FILE_SHA256,
    SOURCE_INDEX_RELATIVE_PATH,
    SOURCE_INVENTORY_SHA256,
    DevelopmentRawSupervisionPlan,
    DevelopmentSourceInventory,
)
from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan import (
    DATASET_MANIFEST_RELATIVE_PATH,
    DATASET_ROWS_RELATIVE_PATH,
)


ROOT = _REPOSITORY_ROOT
CANONICAL_DATASET = (
    ROOT
    / ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
CANONICAL_AUDIT_REPORT = CANONICAL_DATASET.with_name(
    CANONICAL_DATASET.name + ".audit_v6.json"
)
CANONICAL_AUDIT_FAILURE = CANONICAL_DATASET.with_name(
    CANONICAL_DATASET.name + ".audit_v6.failed.json"
)
DATASET_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_dataset_v1"
SHARD_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_shard_v1"
ENDPOINT_INDEX_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_endpoint_index_v1"
AUDIT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_v6"
AUDIT_FAILURE_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_failure_v6"
AUTHORIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v6"
)
REVIEW_BINDING_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v6"
)
BUILDER_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_builder_v6_independent_review_v1"
)
AUDITOR_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v6_independent_review_v1"
)
BUILDER_IMPLEMENTATION_AUTHOR = "/root/raw_builder_arch"
AUDITOR_IMPLEMENTATION_AUTHOR = "/root/raw_auditor_author"
MAX_WORKERS = 6
FROZEN_BUILDER_V6_ROLE_SHA256 = {
    "builder_source": (
        "88c36063e257d9d163317abb15d7854f3da783e0ec15537da4c3d62b113740d7"
    ),
    "builder_cli": (
        "089aca4882f4f574be7972914c12c05acabf1cd898bea6f59422bf07b94f828d"
    ),
    "builder_test": (
        "acf5ca8cdd829d1c3c4ef44dbc4fe7e5d2f05a7dc7ec01662b60d9f27ececdd0"
    ),
    "builder_handoff": (
        "d2cf130a9e2c902776327f6bd71a1b1f363a4dcfde6df0e2aba15edc3957e80b"
    ),
}
EXPECTED_SAMPLE_COUNT = 24
FROZEN_PAIR_COUNTS = {
    "train": 4262,
    "checkpoint_selection": 495,
    "probability_calibration": 415,
}
FROZEN_UNIQUE_ENDPOINT_COUNTS = {
    "train": 7777,
    "checkpoint_selection": 924,
    "probability_calibration": 759,
}
FROZEN_ENDPOINT_REFERENCE_COUNT = 10344
FROZEN_UNIQUE_ENDPOINT_COUNT = 9460
FROZEN_PAIR_COUNT = 5172
FROZEN_SCENE_SHARD_COUNT = 88
FROZEN_FAMILY_COUNT_PER_ROLE = 8

ACCESS_LEDGER_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_access_ledger_v1"
GEOMETRY_CONTRACT_PATH = ROOT / "config/go2_generalization_geometry_v2.json"
GEOMETRY_CONTRACT_FILE_SHA256 = (
    "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52"
)
GEOMETRY_CONTRACT_CONTENT_SHA256 = (
    "e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca"
)
RENDER_AUDIT_PATH = ROOT / ".generated/go2_render_selected_v04/audit_report.json"
RENDER_AUDIT_FILE_SHA256 = (
    "9a045dff82fb82adbbb89d10cb4dc0063297805038b000e5f6cd53816e995a9a"
)
RENDER_AUDIT_CONTENT_SHA256 = (
    "c9280ed4cab9ff54f7d8684835b8448886209a8cc50eba3588519c34572a6358"
)
BUILD_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_2026-07-13.json"
)
EXACT_PLAN_CONTENT_SHA256 = (
    "8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3"
)
EXACT_ORDERED_PAIR_SHA256 = (
    "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
)
EXACT_ORDERED_ENDPOINT_SHA256 = (
    "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
)

FROZEN_PARENT_FILE_SHA256 = {
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_preregistration_2026-07-13.md": (
        "07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_source_inventory_amendment_2026-07-13.md": (
        "39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v5.py": (
        "67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_author_handoff_2026-07-13.md": (
        "b362d26372f01e670a477dda5e7abb5e55370cc1d8d89052545afa229e7bba66"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5_independent_qa.py": (
        "8a50bcf5275d243f06b92264e017f355fd54faaca8f8e73aab1e3cc45dc51298"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_independent_review_2026-07-13.md": (
        "7d7344e423492a3cf36d1cd50ca09e6c7eb6eba17c25861c840531465aaf7706"
    ),
}
FROZEN_V6_PREDECESSOR_SHA256 = {
    **FROZEN_PARENT_FILE_SHA256,
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v6_"
    "authorization_successor_amendment_2026-07-13.md": (
        "09ced36b2eab16585c759e65f7eda844f76006b93de013e5f7057fb9a8e7a137"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v5_"
    "authorization_successor_amendment_2026-07-13.md": (
        "fe6a29a27eb0284ce84fcba409b530c6351befad18ee9d655f5f2e9b337d9e91"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v5.py": (
        "8d85635a85d5a6a3575602a89f37a01f97acf03bd0059a8ae452b21ed4cddce2"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v5.py": (
        "3116c2a5b429cf0fbed0674de91b0569d6ecf6e10c26cd6064a3bb0349e78019"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5.py": (
        "6b49d5d5847e22cea413a7b72da34d5fbf221f876b89bfdf899804024c9d05d6"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_"
    "author_handoff_2026-07-13.md": (
        "a8037613cca9c3879eb2dc8f9df847097a9053326ff973f01a79b3299aec9d26"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5_"
    "independent_qa.py": (
        "fc0ba7af24aeacf975a4b75855e830e9691475391979068385d9d256e8a66812"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_"
    "independent_review_2026-07-13.json": (
        "2687d43da0eb69c39b964ce72f5065fecceb5c2d28652589371d257711702307"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v5.py": (
        "6df29a2faea62191db3b48a93ce114adc23265458a2bb2986fa1a4c5ca732855"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v5.py": (
        "3f2b99ffbf3ab55f6d57c7686f95650f1086394739148978ab618e1b6d8e9b27"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v4_"
    "authorization_successor_amendment_2026-07-13.md": (
        "a535ee8de9a6002f5548f3c3894548ddb42cd9d077eccbb9ca922a41611ced83"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v4.py": (
        "e46f42db3b5ed50581ed916d459e05f2dd9b73dcbdd906ea5d1991b7b61893e0"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v4.py": (
        "db14bb159b39204e7576b71f3b93409e13b9f28c5cb0d2e87a627557471c0901"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4.py": (
        "80ca9d1d35b83fd29027ab297ac662c406dcdd15f68ac5aced9cc7419fef61c0"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_"
    "author_handoff_2026-07-13.md": (
        "575ae2a596901ba90253e57a9bd5f0e64dd5d07f6c5f8e4872cfefbf6fb93bdb"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4_"
    "independent_qa.py": (
        "116b81f65c6c6eb23ed8aba58e9fa2b62a0e0177c4c5e2a0c821c2d0aa8268e2"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_"
    "independent_review_2026-07-13.json": (
        "4c91d7ce09c97fea657ae279183c02f45da7911dbbd6178c5d311e938f602dc4"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v4.py": (
        "d030122e24b7ab2d6da96dff7b88b4bec6ff028da2767e24d480069165654e0d"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v3_"
    "authorization_successor_amendment_2026-07-13.md": (
        "501062e2eba625cf4d7ab28810f2a629652c327c770366c07f3b788f3f6f8b2b"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_"
    "structural_invalidation_2026-07-13.md": (
        "db86ea8bb72478b0f032068151a3c492660444b1fad21b33c700b658de33e213"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_"
    "author_handoff_2026-07-13.md": (
        "a3b66f150320aa790c2a9aa3c8aa0f437824cc619de12349448155559642fe23"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v3.py": (
        "423164701e735c17dca10449434d4d96692180ee148d2a222c9af9b357a83043"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v3.py": (
        "f1258680802be18ad77ca4cf0fa1aacef5e941d9aca40fa68a6d7d8105892445"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v3.py": (
        "4e111e961ed3e8a7250f6c0cfbff4033c5cb6487c67cbbb9d65d389081e9fd19"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v3.py": (
        "3f5154b8c48125146944c740d8cf2b8d7859543ed04e4a2513ddf06a4108c88f"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v3.py": (
        "5bf1e8114706596e0281787e849340ed2f87f4064e7efebfd338153cfaec7ad2"
    ),
    "lewm/tests/go2_shared_jepa_v5_raw_supervision_builder_v3_test_support.py": (
        "df1f92d116f185398ec8b752a24240d94d4a42da0756501ee58853756313145e"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v2.py": (
        "0ae5ddd836802ced1fcf7524b67970247dccace6787fd0acc7268cbae4d3e71c"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v2.py": (
        "c11396874677c3cd3d0ef76353ea7de1449ef610d35f0b4256530a4f62b1d303"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2.py": (
        "6755044af535dc0c2de93f0f5bd79b01b140da33bc8ff2ec5b003ef592b50339"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_"
    "author_handoff_2026-07-13.md": (
        "7f278c5c24a8e9d89c6b0e3ecb9252acd0edec5729bd9fdde5d72231848bc04f"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2_"
    "independent_review.py": (
        "2c34fec949ea43e03b3f7f3c97b8d8ddba0aad1c9192dfd8b00d3f646dd03d43"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_"
    "independent_review_2026-07-13.md": (
        "e42a5876c2b9f564085b3f8e98eeb607f7c15a24e75b5534da79619db1f7ccad"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_"
    "independent_review_2026-07-13.json": (
        "726e03fdc6242ca3074f0b861dbc49565469212e566338fcd3d2756ced886e4a"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v2.py": (
        "d57aacd4849ea3e79468618b73925418ad2035d47de636dc991afda777314b2a"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v2.py": (
        "4502ac44a451841af18e9f9eb545ef961bc81324ea84ce713e434c434e000ae9"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v2.py": (
        "45d60db1f1a7385b7941f8f52e01a923f056bb3f52cc85b7fec4097d54fa9399"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v2_"
    "author_handoff_2026-07-13.md": (
        "6a338b7c15c1fe23ab3680e80c4a30781369e29eebb33331e7ccff723cd4b7ab"
    ),
}
# Builder V6 records this complete closure in the dataset provenance.
FROZEN_PARENT_FILE_SHA256 = dict(FROZEN_V6_PREDECESSOR_SHA256)
REVIEWED_V4_SOURCE_SHA256 = {
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py": (
        "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85"
    ),
    "scripts/build_go2_observable_camera_ray_fit_v4.py": (
        "4efb0517130df39a1953539755d82289b16e89b314bba5713d6d9d944acf1d16"
    ),
    "scripts/audit_go2_n32_camera_frustum_observability.py": (
        "f7e3a3e60937caabbe003ff41af6aec44248df137b0a53c383364272152f3079"
    ),
    "lewm/benchmarks/go2_dynamic_cell_square_projection.py": (
        "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
    ),
    "lewm_worlds/lewm_worlds/manifest.py": (
        "5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888"
    ),
    "lewm/benchmarks/go2_n32_camera_frustum_observability.py": (
        "ab97c34a8a07a93d6b49b5adb0b1a82bc66d38be206baab362b7b1f1b59f3cc3"
    ),
    "lewm/datasets/go2_paired_navigation.py": (
        "14df0cf59ab7554431b1be2ef91e3ab7229200be94bb9afa88127e3ea53c2c08"
    ),
    "lewm/planning/geometry_contract.py": (
        "6873a9550399a5decc90e4a31b2945e54074bdb56855a035924f49b4511c813b"
    ),
    "lewm_worlds/lewm_worlds/planning_grid.py": (
        "e6f7e26d584dfd7923493803fc95a75135122b37a1f95cb51f9267b284649510"
    ),
}

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ACCELERATOR_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

ARRAY_LAYOUT = (
    ("camera_origin_body_m.f4", "<f4", (3,)),
    ("camera_basis_body_fru.f4", "<f4", (3, 3)),
    ("ground_plane_z_body_m.f4", "<f4", ()),
    ("ground_support_in_frustum.u1", "|u1", (128, 128, 5)),
    ("ground_support_clear_to_target.u1", "|u1", (128, 128, 5)),
    ("pixel_hit_mask.u1", "|u1", (84, 112)),
    ("pixel_first_hit_distance_m.f4", "<f4", (84, 112)),
    ("raster_labels.u1", "|u1", (64, 64)),
)

SOURCE_ROLE_PATHS: tuple[tuple[str, str], ...] = (
    (
        "builder_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v6.py",
    ),
    (
        "builder_cli",
        "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v6.py",
    ),
    (
        "builder_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py",
    ),
    (
        "builder_handoff",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
        "author_handoff_2026-07-13.md",
    ),
    (
        "builder_review",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
        "independent_review_2026-07-13.json",
    ),
    (
        "auditor_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v6.py",
    ),
    (
        "auditor_cli",
        "scripts/audit_go2_shared_jepa_v5_raw_supervision_v6.py",
    ),
    (
        "auditor_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v6.py",
    ),
    (
        "auditor_review",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v6_"
        "independent_review_2026-07-13.json",
    ),
)
SOURCE_ROLES = tuple(role for role, _path in SOURCE_ROLE_PATHS)
SOURCE_PATH_BY_ROLE = dict(SOURCE_ROLE_PATHS)
BUILDER_CANDIDATE_ROLES = (
    "builder_source",
    "builder_cli",
    "builder_test",
    "builder_handoff",
)
AUDITOR_CANDIDATE_ROLES = (
    "auditor_source",
    "auditor_cli",
    "auditor_test",
)
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "exact_build_authorized_after_independent_reviews",
        "builder_review",
        "auditor_review",
        "source_map",
        "content_sha256",
    }
)
SOURCE_ENTRY_FIELDS = frozenset({"role", "path", "sha256"})
REVIEW_BINDING_FIELDS = frozenset(
    {
        "schema",
        "review_schema",
        "verdict",
        "reviewer",
        "implementation_author",
        "path",
        "file_sha256",
        "content_sha256",
        "candidate",
    }
)
REVIEW_RECORD_FIELDS = frozenset(
    {
        "schema",
        "verdict",
        "reviewer",
        "implementation_author",
        "candidate",
        "authority",
        "content_sha256",
    }
)
REVIEW_AUTHORITY_FALSE_FIELDS = (
    "exact_build_authorized",
    "exact_audit_authorized",
    "dataset_use_authorized",
    "training_authorized",
    "selection_authorized",
    "calibration_authorized",
    "g2_authorized",
    "heldout_authorized",
    "runtime_authorized",
    "navigation_authorized",
    "hardware_authorized",
    "production_authorized",
    "promotion_authorized",
)

MANIFEST_FIELDS = frozenset(
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
SHARD_FIELDS = frozenset(
    {
        "schema",
        "dataset_role",
        "family",
        "scene_id",
        "scene_id_sha256",
        "endpoint_count",
        "ordered_endpoint_identity_sha256",
        "ordered_evidence_sha256",
        "ordered_raster_sha256",
        "files",
        "content_sha256",
    }
)
SHARD_INDEX_FIELDS = frozenset(
    {
        "schema",
        "dataset_role",
        "family",
        "scene_id",
        "endpoint_identity_sha256",
        "plan_endpoint_content_sha256",
        "shard_row",
        "image_path_metadata_only",
        "image_sha256_commitment_only",
        "evidence_content_sha256",
        "raster_content_sha256",
        "content_sha256",
    }
)
TOP_ENDPOINT_FIELDS = frozenset((*SHARD_INDEX_FIELDS, "scene_shard"))
ROOT_FILE_FIELDS = frozenset({"path", "byte_count", "file_sha256"})
SHARD_FILE_FIELDS = frozenset(
    {"path", "byte_count", "file_sha256", "dtype", "shape"}
)

FALSE_LICENSE_FIELDS = frozenset(
    {
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
)
FORBIDDEN_LEDGER_FRAGMENTS = (
    "rgb_byte_open",
    "rgb_decode",
    "label_shard_payload_open",
    "g2_payload_open",
    "g2_geometry",
    "g2_label",
    "g2_rgb",
    "checkpoint",
    "model_output",
    "runtime",
    "navigation_result",
    "heldout",
    "sealed",
    "hardware",
    "production",
)
EXACT_ACCESS_LEDGER_KEYS = frozenset(
    {
        "schema",
        "measurement_scope",
        "metadata_plan_first_pass",
        "metadata_source_inventory_first_pass",
        "metadata_plan_second_pass",
        "metadata_source_inventory_second_pass",
        "development_scene_workers",
        "unique_endpoint_raycasts",
        "pair_endpoint_references",
        "source_frames_jsonl_records_scanned",
        "source_frames_selected_records",
        "source_frames_byte_opens",
        "source_scene_manifest_byte_opens",
        "render_plan_byte_opens",
        "render_summary_byte_opens",
        "geometry_contract_byte_opens",
        "render_audit_byte_opens",
        "source_payload_first_pass_file_count",
        "source_payload_second_pass_file_count",
        "source_payload_total_byte_opens",
        "g2_source_index_rows_read_for_exclusion",
        "g2_sidecar_byte_opens",
        "g2_source_payload_opens",
        "g2_label_payload_opens",
        "g2_rgb_byte_opens",
        "rgb_byte_opens",
        "rgb_decodes",
        "parent_label_shard_payload_opens",
        "checkpoint_or_model_output_opens",
        "runtime_or_navigation_result_opens",
        "heldout_or_sealed_opens",
        "hardware_or_production_opens",
        "writes_outside_output_or_failure_namespace",
        "denied_or_unexpected_accesses",
    }
)
EXACT_INPUT_PROVENANCE_FIELDS = frozenset(
    {
        "authorization_file_sha256",
        "authorization_content_sha256",
        "authorization_source_map_sha256",
        "frozen_parent_file_sha256",
        "reviewed_v4_source_sha256",
        "metadata_plan_content_sha256",
        "metadata_ordered_pair_sha256",
        "metadata_ordered_endpoint_sha256",
        "source_inventory_sha256",
        "source_payload_inventory",
        "source_payload_inventory_sha256",
        "geometry_contract_file_sha256",
        "geometry_contract_content_sha256",
        "render_audit_file_sha256",
        "render_audit_content_sha256",
    }
)


class RawSupervisionAuditError(RuntimeError):
    """Raised when the immutable raw-supervision artifact is not exact."""


@dataclass(frozen=True)
class StoredEndpointEvidence:
    endpoint_identity_sha256: str
    arrays: tuple[np.ndarray, ...]
    evidence_content_sha256: str
    raster_content_sha256: str


@dataclass(frozen=True)
class AuditInputs:
    plan: DevelopmentRawSupervisionPlan
    inventory: DevelopmentSourceInventory


@dataclass(frozen=True)
class SourceBindingV6:
    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ReviewBindingV6:
    kind: str
    review_schema: str
    verdict: str
    reviewer: str
    implementation_author: str
    path: str
    file_sha256: str
    content_sha256: str
    candidate: tuple[SourceBindingV6, ...]


@dataclass(frozen=True)
class PhaseOneAuthorizationV6:
    authorization_file_sha256: str
    authorization_content_sha256: str
    source_map_sha256: str
    canonical_payload: bytes
    sources: tuple[SourceBindingV6, ...]
    builder_review: ReviewBindingV6
    auditor_review: ReviewBindingV6


@dataclass(frozen=True)
class AcceptedAuthorizationV6:
    authorization_file_sha256: str
    authorization_content_sha256: str
    source_map_sha256: str
    sources: tuple[SourceBindingV6, ...]


Fingerprint = tuple[int, int, int, int, int, int, int]


def _fingerprint(metadata: os.stat_result) -> Fingerprint:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_open_flags() -> int:
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise RawSupervisionAuditError(
            "descriptor-relative no-follow directories are unavailable"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


@dataclass
class _RetainedDirectoryChain:
    absolute_path: Path
    descriptors: list[int]
    entries: list[tuple[int, str, int, Fingerprint]]
    anchor_fingerprint: Fingerprint

    @property
    def directory_fd(self) -> int:
        return self.descriptors[-1]

    def validate(self, *, allow_final_metadata_change: bool = False) -> None:
        if _fingerprint(os.fstat(self.descriptors[0])) != self.anchor_fingerprint:
            raise RawSupervisionAuditError("filesystem root changed during publication")
        final_index = len(self.entries) - 1
        for index, (parent_fd, component, child_fd, expected) in enumerate(self.entries):
            try:
                named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                opened = os.fstat(child_fd)
            except (FileNotFoundError, NotADirectoryError, OSError) as error:
                raise RawSupervisionAuditError(
                    "audit output directory chain changed"
                ) from error
            named_fingerprint = _fingerprint(named)
            opened_fingerprint = _fingerprint(opened)
            # Publishing a leaf legitimately changes only its immediate
            # directory's size/timestamps.  Keep identity, type, mode, and
            # link count stable there; retain complete fingerprints above it.
            expected_matches = (
                named_fingerprint[:4] == expected[:4]
                and opened_fingerprint[:4] == expected[:4]
                if index == final_index and allow_final_metadata_change
                else named_fingerprint == expected
                and opened_fingerprint == expected
            )
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not expected_matches
            ):
                raise RawSupervisionAuditError("audit output directory chain changed")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()

    def __enter__(self) -> "_RetainedDirectoryChain":
        self.validate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _open_retained_directory_chain(path: Path) -> _RetainedDirectoryChain:
    absolute = Path(path).absolute()
    if not absolute.is_absolute() or not absolute.anchor:
        raise RawSupervisionAuditError("audit output parent must be absolute")
    descriptors: list[int] = []
    entries: list[tuple[int, str, int, Fingerprint]] = []
    try:
        filesystem_root = Path(absolute.anchor)
        anchor_before = filesystem_root.stat(follow_symlinks=False)
        anchor_fd = os.open(filesystem_root, _directory_open_flags())
        descriptors.append(anchor_fd)
        anchor_fingerprint = _fingerprint(anchor_before)
        if _fingerprint(os.fstat(anchor_fd)) != anchor_fingerprint:
            raise RawSupervisionAuditError("filesystem root changed during open")
        parent_fd = anchor_fd
        for component in absolute.parts[1:]:
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                raise PermissionError("audit output parent contains an alias")
            expected = _fingerprint(named)
            child_fd = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
            descriptors.append(child_fd)
            if _fingerprint(os.fstat(child_fd)) != expected:
                raise RawSupervisionAuditError(
                    "audit output parent changed during descriptor open"
                )
            entries.append((parent_fd, component, child_fd, expected))
            parent_fd = child_fd
        chain = _RetainedDirectoryChain(
            absolute_path=absolute,
            descriptors=descriptors,
            entries=entries,
            anchor_fingerprint=anchor_fingerprint,
        )
        chain.validate()
        return chain
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _lstat_optional_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _read_absolute_bound_payload(
    path: Path,
    expected_sha256: str,
    *,
    repository_root: Path,
    name: str,
) -> bytes:
    """Read one allowlisted absolute file through a retained no-follow chain."""

    lexical = Path(path)
    root = Path(repository_root).absolute()
    if not lexical.is_absolute() or lexical != Path(os.path.normpath(str(lexical))):
        raise PermissionError(f"{name} path must be canonical and absolute")
    try:
        lexical.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{name} escapes the repository") from error
    if not _is_sha256(expected_sha256):
        raise RawSupervisionAuditError(f"{name} SHA-256 is malformed")
    with _open_retained_directory_chain(lexical.parent) as chain:
        leaf_name = lexical.name
        before = os.stat(leaf_name, dir_fd=chain.directory_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
        ):
            raise PermissionError(f"{name} must be an unaliased regular file")
        expected_fingerprint = _fingerprint(before)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(leaf_name, flags, dir_fd=chain.directory_fd)
        try:
            if _fingerprint(os.fstat(descriptor)) != expected_fingerprint:
                raise RawSupervisionAuditError(f"{name} changed during open")
            chain.validate()
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            chain.validate()
            named_after = os.stat(
                leaf_name,
                dir_fd=chain.directory_fd,
                follow_symlinks=False,
            )
            if (
                _fingerprint(named_after) != expected_fingerprint
                or _fingerprint(os.fstat(descriptor)) != expected_fingerprint
            ):
                raise RawSupervisionAuditError(f"{name} changed while read")
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    if _sha256_bytes(payload) != expected_sha256:
        raise RawSupervisionAuditError(f"{name} file SHA-256 changed")
    return payload


def _read_fixed_manifest_payload() -> tuple[bytes, str]:
    """Descriptor-read the fixed manifest after V6 authority acceptance."""

    with _open_retained_directory_chain(CANONICAL_DATASET) as chain:
        name = "manifest.json"
        before = os.stat(name, dir_fd=chain.directory_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
        ):
            raise PermissionError(
                "dataset manifest must be an unaliased regular file"
            )
        expected = _fingerprint(before)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=chain.directory_fd,
        )
        try:
            if _fingerprint(os.fstat(descriptor)) != expected:
                raise RawSupervisionAuditError(
                    "dataset manifest changed during open"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            chain.validate()
            named_after = os.stat(
                name,
                dir_fd=chain.directory_fd,
                follow_symlinks=False,
            )
            if (
                _fingerprint(named_after) != expected
                or _fingerprint(os.fstat(descriptor)) != expected
            ):
                raise RawSupervisionAuditError(
                    "dataset manifest changed while read"
                )
        finally:
            os.close(descriptor)
    payload = b"".join(chunks)
    return payload, _sha256_bytes(payload)


def _rename_noreplace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        1,
    ) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination_name)
        raise OSError(number, os.strerror(number), destination_name)


class _FailureReceiptPublisher:
    """Publish only a non-authoritative terminal failure receipt."""

    def __init__(self) -> None:
        self._chain = _open_retained_directory_chain(
            CANONICAL_AUDIT_FAILURE.parent
        )
        self._counter = 0

    @property
    def parent_fd(self) -> int:
        return self._chain.directory_fd

    def __enter__(self) -> "_FailureReceiptPublisher":
        self._chain.validate(allow_final_metadata_change=True)
        return self

    def __exit__(self, *_args: object) -> None:
        self._chain.close()

    def require_absent(self) -> None:
        self._chain.validate(allow_final_metadata_change=True)
        for name in (
            CANONICAL_AUDIT_REPORT.name,
            CANONICAL_AUDIT_FAILURE.name,
        ):
            if _lstat_optional_at(self.parent_fd, name) is not None:
                raise FileExistsError(f"immutable audit leaf already exists: {name}")
        self._chain.validate(allow_final_metadata_change=True)

    def publish(self, value: Mapping[str, Any]) -> None:
        name = CANONICAL_AUDIT_FAILURE.name
        if (
            value.get("schema") != AUDIT_FAILURE_SCHEMA
            or value.get("status") != "terminal_failed_no_dataset_authority"
        ):
            raise PermissionError("only a terminal V6 failure receipt may publish")
        payload = canonical_json_bytes(value) + b"\n"
        self._counter += 1
        temporary = f".{name}.owned.{os.getpid()}.{self._counter}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=self.parent_fd)
        owned = _fingerprint(os.fstat(descriptor))
        owned_identity = owned[:2]
        renamed = False
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("audit publication write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            self._chain.validate(allow_final_metadata_change=True)
            if _lstat_optional_at(self.parent_fd, name) is not None:
                raise FileExistsError(f"immutable audit leaf already exists: {name}")
            _rename_noreplace_at(self.parent_fd, temporary, name)
            renamed = True
            os.fsync(self.parent_fd)
            published = os.stat(name, dir_fd=self.parent_fd, follow_symlinks=False)
            if _fingerprint(published)[:2] != owned_identity:
                raise RawSupervisionAuditError("published audit leaf identity changed")
            try:
                self._chain.validate(allow_final_metadata_change=True)
            except BaseException:
                current = _lstat_optional_at(self.parent_fd, name)
                if current is not None and _fingerprint(current)[:2] == owned_identity:
                    os.unlink(name, dir_fd=self.parent_fd)
                    os.fsync(self.parent_fd)
                raise
        finally:
            os.close(descriptor)
            if not renamed:
                current = _lstat_optional_at(self.parent_fd, temporary)
                if current is not None and _fingerprint(current)[:2] == owned_identity:
                    os.unlink(temporary, dir_fd=self.parent_fd)
                    os.fsync(self.parent_fd)


TransactionFingerprint = tuple[int, int, int, int, int, int, int]


def _transaction_fingerprint(metadata: os.stat_result) -> TransactionFingerprint:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _TransactionLeaf:
    path: Path
    descriptor: int
    fingerprint: TransactionFingerprint
    sha256: str
    byte_count: int
    namespace: str


@dataclass(frozen=True)
class _TransactionDirectory:
    path: Path
    descriptor: int
    fingerprint: TransactionFingerprint
    publication_parent: bool


@dataclass(frozen=True)
class _WatchBinding:
    descriptor: int
    path: Path
    roles: frozenset[str]


@dataclass(frozen=True)
class _InotifyEvent:
    watch_descriptor: int
    mask: int
    cookie: int
    name: str


@dataclass(frozen=True)
class _AuditPublicationContextV6:
    authorization: AcceptedAuthorizationV6
    manifest: Mapping[str, Any]
    manifest_file_sha256: str
    hashed_sources: tuple[Mapping[str, Any], ...]
    parent_contracts: tuple[Mapping[str, Any], ...]


_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ONLYDIR = 0x01000000
_IN_DONT_FOLLOW = 0x02000000
_IN_EXCL_UNLINK = 0x04000000
_IN_MASK_CREATE = 0x10000000
_IN_ISDIR = 0x40000000
_IN_EVENT_BITS = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_Q_OVERFLOW
    | _IN_IGNORED
    | _IN_ISDIR
)
_IN_DIRECTORY_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_ONLYDIR
    | _IN_DONT_FOLLOW
    | _IN_EXCL_UNLINK
    | _IN_MASK_CREATE
)
_IN_LEAF_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_DONT_FOLLOW
    | _IN_EXCL_UNLINK
    | _IN_MASK_CREATE
)
_INOTIFY_HEADER = struct.Struct("iIII")


def _inotify_init() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    initializer = getattr(libc, "inotify_init1", None)
    if initializer is None:
        raise OSError(errno.ENOSYS, "Linux inotify_init1 is required")
    initializer.argtypes = (ctypes.c_int,)
    initializer.restype = ctypes.c_int
    descriptor = int(
        initializer(getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    )
    if descriptor < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return descriptor


def _inotify_add(descriptor: int, path: Path, mask: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    add_watch = getattr(libc, "inotify_add_watch", None)
    if add_watch is None:
        raise OSError(errno.ENOSYS, "Linux inotify_add_watch is required")
    add_watch.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
    add_watch.restype = ctypes.c_int
    watch_descriptor = int(
        add_watch(int(descriptor), os.fsencode(path), ctypes.c_uint32(mask))
    )
    if watch_descriptor < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(path))
    return watch_descriptor


def _open_transaction_directory(
    path: Path, *, publication_parent: bool
) -> _TransactionDirectory:
    canonical = Path(path).absolute()
    before = canonical.stat(follow_symlinks=False)
    if (
        canonical != path
        or canonical.resolve(strict=True) != canonical
        or canonical.is_symlink()
        or not stat.S_ISDIR(before.st_mode)
    ):
        raise PermissionError(f"transaction directory is noncanonical: {path}")
    descriptor = os.open(canonical, _directory_open_flags())
    fingerprint = _transaction_fingerprint(before)
    try:
        if (
            _transaction_fingerprint(os.fstat(descriptor)) != fingerprint
            or _transaction_fingerprint(canonical.stat(follow_symlinks=False))
            != fingerprint
        ):
            raise RawSupervisionAuditError(
                f"transaction directory changed during open: {canonical}"
            )
        return _TransactionDirectory(
            path=canonical,
            descriptor=descriptor,
            fingerprint=fingerprint,
            publication_parent=publication_parent,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _open_transaction_leaf(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    namespace: str,
) -> _TransactionLeaf:
    canonical = Path(path).absolute()
    before = canonical.stat(follow_symlinks=False)
    if (
        canonical != path
        or canonical.resolve(strict=True) != canonical
        or canonical.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or not _is_sha256(expected_sha256)
    ):
        raise PermissionError(f"transaction leaf is not canonical: {path}")
    descriptor = os.open(
        canonical,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    fingerprint = _transaction_fingerprint(before)
    try:
        named = canonical.stat(follow_symlinks=False)
        opened = os.fstat(descriptor)
        actual_sha256 = _sha256_fd(descriptor)
        if (
            _transaction_fingerprint(named) != fingerprint
            or _transaction_fingerprint(opened) != fingerprint
            or int(named.st_nlink) != 1
            or int(opened.st_nlink) != 1
            or actual_sha256 != expected_sha256
            or (expected_bytes is not None and int(opened.st_size) != expected_bytes)
        ):
            raise RawSupervisionAuditError(
                f"transaction leaf changed during baseline: {canonical}"
            )
        return _TransactionLeaf(
            path=canonical,
            descriptor=descriptor,
            fingerprint=fingerprint,
            sha256=actual_sha256,
            byte_count=int(opened.st_size),
            namespace=namespace,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _canonical_relative_file(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise RawSupervisionAuditError(f"{name} path is noncanonical")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RawSupervisionAuditError(f"{name} path is noncanonical")
    return value


def _dataset_transaction_inventory(
    manifest: Mapping[str, Any], manifest_file_sha256: str
) -> tuple[dict[Path, tuple[str, int | None]], set[Path]]:
    bindings: dict[Path, tuple[str, int | None]] = {
        (CANONICAL_DATASET / "manifest.json").absolute(): (
            manifest_file_sha256,
            None,
        )
    }
    expected_files = {"manifest.json"}
    files = manifest.get("files")
    if type(files) is not list:
        raise RawSupervisionAuditError("dataset manifest files changed")
    for record in files:
        if type(record) is not dict:
            raise RawSupervisionAuditError("dataset file record changed")
        relative = _canonical_relative_file(
            record.get("path"), name="dataset transaction"
        )
        digest = record.get("file_sha256")
        byte_count = record.get("byte_count")
        if not _is_sha256(digest) or type(byte_count) is not int or byte_count < 0:
            raise RawSupervisionAuditError("dataset transaction binding changed")
        expected_files.add(relative)
        bindings[(CANONICAL_DATASET / relative).absolute()] = (
            str(digest),
            int(byte_count),
        )
    expected_directories = {CANONICAL_DATASET.absolute()}
    for relative in expected_files:
        parent = (CANONICAL_DATASET / relative).parent.absolute()
        while parent != CANONICAL_DATASET.parent.absolute():
            expected_directories.add(parent)
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories = {CANONICAL_DATASET.absolute()}
    for path in sorted(CANONICAL_DATASET.rglob("*"), key=str):
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise RawSupervisionAuditError("dataset transaction found a symlink")
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(str(path.relative_to(CANONICAL_DATASET)))
        elif stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(path.absolute())
        else:
            raise RawSupervisionAuditError(
                "dataset transaction found a special file"
            )
    if observed_files != expected_files or observed_directories != expected_directories:
        raise RawSupervisionAuditError("dataset transaction inventory changed")
    return bindings, expected_directories


def _audit_transaction_bindings(
    context: _AuditPublicationContextV6,
) -> tuple[dict[Path, tuple[str, int | None]], set[Path]]:
    bindings, dataset_directories = _dataset_transaction_inventory(
        context.manifest, context.manifest_file_sha256
    )

    def bind(path: Path, digest: str, byte_count: int | None = None) -> None:
        canonical = Path(path).absolute()
        if canonical != path or not _is_sha256(digest):
            raise RawSupervisionAuditError("audit transaction binding changed")
        try:
            canonical.relative_to(ROOT)
        except ValueError as error:
            raise PermissionError("audit transaction source escapes repository") from error
        value = (digest, byte_count)
        previous = bindings.get(canonical)
        if previous is not None and previous != value:
            if previous[0] != digest or (
                previous[1] is not None
                and byte_count is not None
                and previous[1] != byte_count
            ):
                raise RawSupervisionAuditError(
                    f"conflicting audit transaction binding: {canonical}"
                )
            value = (digest, previous[1] if previous[1] is not None else byte_count)
        bindings[canonical] = value

    authorization = context.authorization
    bind(BUILD_AUTHORIZATION_PATH, authorization.authorization_file_sha256)
    for source in authorization.sources:
        bind((ROOT / source.path).absolute(), source.sha256)
    for relative, digest in {
        **FROZEN_V6_PREDECESSOR_SHA256,
        **REVIEWED_V4_SOURCE_SHA256,
    }.items():
        bind((ROOT / relative).absolute(), digest)
    bind(
        (ROOT / DATASET_MANIFEST_RELATIVE_PATH).absolute(),
        DATASET_MANIFEST_FILE_SHA256,
    )
    bind(
        (ROOT / DATASET_ROWS_RELATIVE_PATH).absolute(),
        DATASET_ROWS_FILE_SHA256,
    )
    bind(
        (ROOT / SOURCE_INDEX_RELATIVE_PATH).absolute(),
        SOURCE_INDEX_FILE_SHA256,
    )
    for record in (*context.hashed_sources, *context.parent_contracts):
        bind(
            Path(str(record["path"])),
            str(record["file_sha256"]),
            int(record["byte_count"]),
        )
    return dict(sorted(bindings.items(), key=lambda item: str(item[0]))), dataset_directories


class _ClosedAuditPublicationTransaction:
    """Continuously bind every input and the owned report through rename."""

    def __init__(
        self,
        *,
        context: _AuditPublicationContextV6,
        retained: _RetainedDirectoryChain,
        candidate_name: str,
        candidate_descriptor: int,
        candidate_fingerprint: TransactionFingerprint,
        candidate_sha256: str,
    ) -> None:
        self._context = context
        self._retained = retained
        self._candidate_name = candidate_name
        self._candidate_descriptor = candidate_descriptor
        self._candidate_fingerprint = candidate_fingerprint
        self._candidate_sha256 = candidate_sha256
        self._destination_name = CANONICAL_AUDIT_REPORT.name
        self._directories: list[_TransactionDirectory] = []
        self._leaves: list[_TransactionLeaf] = []
        self._watch_by_descriptor: dict[int, _WatchBinding] = {}
        self._watch_by_path: dict[Path, _WatchBinding] = {}
        self._poison_reason: str | None = None
        self._renamed = False
        self._closed = False
        self._inotify_fd = _inotify_init()
        try:
            source_bindings, dataset_directories = _audit_transaction_bindings(
                context
            )
            directories = set(dataset_directories)
            directories.update(path.parent for path in source_bindings)
            # Watching each canonical ancestor closes the V6 builder's
            # post-rename ancestor-replacement gap.
            for path in tuple(directories) + (retained.absolute_path,):
                directories.add(path)
                directories.update(path.parents)
            self._bind_directories(directories)
            self._bind_source_leaves(source_bindings)
            self._bind_candidate()
            self._require_no_events("transaction baseline")
            self._validate_bound_inventory(after_rename=False)
            self._require_no_events("transaction baseline validation")
        except BaseException:
            self.close()
            raise

    @property
    def renamed(self) -> bool:
        return self._renamed

    def _poison(self, reason: str) -> None:
        if self._poison_reason is None:
            self._poison_reason = reason
        raise RawSupervisionAuditError(
            f"closed audit transaction poisoned: {self._poison_reason}"
        )

    def _add_watch(self, path: Path, *, directory: bool, role: str) -> None:
        existing = self._watch_by_path.get(path)
        if existing is not None:
            merged = _WatchBinding(
                descriptor=existing.descriptor,
                path=path,
                roles=frozenset((*existing.roles, role)),
            )
            self._watch_by_path[path] = merged
            self._watch_by_descriptor[merged.descriptor] = merged
            return
        descriptor = _inotify_add(
            self._inotify_fd,
            path,
            _IN_DIRECTORY_MASK if directory else _IN_LEAF_MASK,
        )
        if descriptor in self._watch_by_descriptor:
            self._poison("inotify watch descriptor was reused")
        binding = _WatchBinding(descriptor, path, frozenset({role}))
        self._watch_by_path[path] = binding
        self._watch_by_descriptor[descriptor] = binding

    def _bind_directories(self, paths: set[Path]) -> None:
        inode_paths: dict[tuple[int, int], Path] = {}
        for path in sorted(paths, key=str):
            directory = _open_transaction_directory(
                path,
                publication_parent=path == self._retained.absolute_path,
            )
            identity = directory.fingerprint[:2]
            previous = inode_paths.get(identity)
            if previous is not None and previous != path:
                os.close(directory.descriptor)
                self._poison("distinct transaction directories alias one inode")
            inode_paths[identity] = path
            self._directories.append(directory)
            self._add_watch(
                path,
                directory=True,
                role=(
                    "publication_parent"
                    if directory.publication_parent
                    else "source_or_ancestor_directory"
                ),
            )

    def _bind_source_leaves(
        self, bindings: Mapping[Path, tuple[str, int | None]]
    ) -> None:
        inode_paths: dict[tuple[int, int], Path] = {}
        for path, (digest, byte_count) in bindings.items():
            leaf = _open_transaction_leaf(
                path,
                expected_sha256=digest,
                expected_bytes=byte_count,
                namespace="source",
            )
            identity = leaf.fingerprint[:2]
            previous = inode_paths.get(identity)
            if previous is not None and previous != path:
                os.close(leaf.descriptor)
                self._poison("distinct transaction leaves alias one inode")
            inode_paths[identity] = path
            self._leaves.append(leaf)
            self._add_watch(path, directory=False, role="source_leaf")

    def _bind_candidate(self) -> None:
        named = os.stat(
            self._candidate_name,
            dir_fd=self._retained.directory_fd,
            follow_symlinks=False,
        )
        opened = os.fstat(self._candidate_descriptor)
        if (
            _transaction_fingerprint(named) != self._candidate_fingerprint
            or _transaction_fingerprint(opened) != self._candidate_fingerprint
            or not stat.S_ISREG(named.st_mode)
            or int(named.st_nlink) != 1
            or _sha256_fd(self._candidate_descriptor) != self._candidate_sha256
        ):
            raise RawSupervisionAuditError("owned audit candidate changed")
        self._add_watch(
            self._retained.absolute_path / self._candidate_name,
            directory=False,
            role="audit_candidate",
        )

    def _read_events(self, *, wait_milliseconds: int = 0) -> list[_InotifyEvent]:
        poller = select.poll()
        poller.register(
            self._inotify_fd, select.POLLIN | select.POLLERR | select.POLLHUP
        )
        try:
            ready = poller.poll(wait_milliseconds)
        except OSError:
            self._poison("inotify polling failed")
        events: list[_InotifyEvent] = []
        while ready:
            if any(fd != self._inotify_fd for fd, _mask in ready):
                self._poison("poll returned an unknown inotify descriptor")
            if any(
                mask
                & (select.POLLERR | select.POLLHUP | getattr(select, "POLLNVAL", 0))
                for _fd, mask in ready
            ):
                self._poison("inotify descriptor reported error or hangup")
            try:
                payload = os.read(self._inotify_fd, 1024 * 1024)
            except BlockingIOError:
                self._poison("inotify readiness produced no event")
            except OSError:
                self._poison("inotify event read failed")
            if not payload:
                self._poison("inotify descriptor reached unexpected EOF")
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < _INOTIFY_HEADER.size:
                    self._poison("truncated inotify event header")
                descriptor, mask, cookie, name_length = _INOTIFY_HEADER.unpack_from(
                    payload, offset
                )
                offset += _INOTIFY_HEADER.size
                if name_length % 4 or name_length > len(payload) - offset:
                    self._poison("truncated inotify event name")
                raw_name = payload[offset : offset + name_length]
                offset += name_length
                if name_length:
                    nul = raw_name.find(b"\0")
                    if nul <= 0 or any(raw_name[nul:]):
                        self._poison("malformed inotify event name")
                    try:
                        name = raw_name[:nul].decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        self._poison("undecodable inotify event name")
                    if (
                        name in {".", ".."}
                        or "/" in name
                        or "\\" in name
                        or Path(name).name != name
                    ):
                        self._poison("noncanonical inotify event name")
                else:
                    name = ""
                if mask & _IN_Q_OVERFLOW:
                    self._poison("inotify queue overflow")
                if descriptor not in self._watch_by_descriptor:
                    self._poison("unknown inotify watch descriptor")
                if mask & ~_IN_EVENT_BITS:
                    self._poison("unknown inotify event mask")
                if mask & (_IN_IGNORED | _IN_UNMOUNT):
                    self._poison("inotify watch was lost")
                events.append(_InotifyEvent(descriptor, mask, cookie, name))
            try:
                ready = poller.poll(0)
            except OSError:
                self._poison("inotify polling failed")
        return events

    def _require_no_events(self, phase: str) -> None:
        if self._read_events():
            self._poison(f"filesystem mutation observed during {phase}")

    def _candidate_path(self, *, after_rename: bool) -> Path:
        return self._retained.absolute_path / (
            self._destination_name if after_rename else self._candidate_name
        )

    def _validate_bound_inventory(self, *, after_rename: bool) -> None:
        _dataset_transaction_inventory(
            self._context.manifest, self._context.manifest_file_sha256
        )
        for directory in self._directories:
            try:
                named = directory.path.stat(follow_symlinks=False)
                opened = os.fstat(directory.descriptor)
            except OSError:
                self._poison(f"bound directory became unavailable: {directory.path}")
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or _transaction_fingerprint(named)
                != _transaction_fingerprint(opened)
                or (
                    not (after_rename and directory.publication_parent)
                    and _transaction_fingerprint(opened) != directory.fingerprint
                )
            ):
                self._poison(f"bound directory changed: {directory.path}")
        for leaf in self._leaves:
            try:
                named = leaf.path.stat(follow_symlinks=False)
                opened = os.fstat(leaf.descriptor)
            except OSError:
                self._poison(f"bound source became unavailable: {leaf.path}")
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or int(named.st_nlink) != 1
                or int(opened.st_nlink) != 1
                or _transaction_fingerprint(named) != leaf.fingerprint
                or _transaction_fingerprint(opened) != leaf.fingerprint
                or _sha256_fd(leaf.descriptor) != leaf.sha256
            ):
                self._poison(f"bound source changed: {leaf.path}")
        candidate_path = self._candidate_path(after_rename=after_rename)
        try:
            named_candidate = candidate_path.stat(follow_symlinks=False)
            opened_candidate = os.fstat(self._candidate_descriptor)
        except OSError:
            self._poison("owned audit candidate became unavailable")
        if (
            _transaction_fingerprint(named_candidate)
            != self._candidate_fingerprint
            or _transaction_fingerprint(opened_candidate)
            != self._candidate_fingerprint
            or _sha256_fd(self._candidate_descriptor) != self._candidate_sha256
        ):
            self._poison("owned audit candidate changed")

    def validate_before_rename(self) -> None:
        if self._renamed or self._closed or self._poison_reason is not None:
            self._poison("invalid pre-rename transaction state")
        self._validate_bound_inventory(after_rename=False)
        self._retained.validate(allow_final_metadata_change=True)
        self._require_no_events("final source and candidate validation")

    def rename_owned(self) -> None:
        if self._renamed or self._closed or self._poison_reason is not None:
            self._poison("invalid rename transaction state")
        try:
            _rename_noreplace_at(
                self._retained.directory_fd,
                self._candidate_name,
                self._destination_name,
            )
        except OSError:
            self._poison("atomic owned audit rename failed")
        self._renamed = True
        events = self._read_events(wait_milliseconds=100)
        parent = self._watch_by_path[self._retained.absolute_path]
        candidate = self._watch_by_path[
            self._retained.absolute_path / self._candidate_name
        ]
        if len(events) != 3:
            self._poison("owned audit rename emitted an unexpected event count")
        moved_from, moved_to, self_move = events
        if (
            moved_from.watch_descriptor != parent.descriptor
            or moved_from.mask != _IN_MOVED_FROM
            or moved_from.name != self._candidate_name
            or moved_from.cookie == 0
            or moved_to.watch_descriptor != parent.descriptor
            or moved_to.mask != _IN_MOVED_TO
            or moved_to.name != self._destination_name
            or moved_to.cookie != moved_from.cookie
            or self_move.watch_descriptor != candidate.descriptor
            or self_move.mask != _IN_MOVE_SELF
            or self_move.cookie != 0
            or self_move.name
        ):
            self._poison("owned audit rename event sequence changed")

    def validate_after_rename(self) -> None:
        if not self._renamed or self._closed or self._poison_reason is not None:
            self._poison("invalid post-rename transaction state")
        self._validate_bound_inventory(after_rename=True)
        self._retained.validate(allow_final_metadata_change=True)
        self._require_no_events("post-rename validation")

    def require_final_quiet(self) -> None:
        if not self._renamed or self._closed or self._poison_reason is not None:
            self._poison("invalid final transaction state")
        self._validate_bound_inventory(after_rename=True)
        self._retained.validate(allow_final_metadata_change=True)
        self._require_no_events("post-rename parent fsync")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for leaf in reversed(self._leaves):
            os.close(leaf.descriptor)
        for directory in reversed(self._directories):
            os.close(directory.descriptor)
        os.close(self._inotify_fd)


def _stage_owned_audit_candidate(
    retained: _RetainedDirectoryChain,
    result: Mapping[str, Any],
) -> tuple[str, int, TransactionFingerprint, str]:
    payload = canonical_json_bytes(result) + b"\n"
    name = f".{CANONICAL_AUDIT_REPORT.name}.owned.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=retained.directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("audit candidate write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=retained.directory_fd, follow_symlinks=False)
        fingerprint = _transaction_fingerprint(opened)
        if (
            _transaction_fingerprint(named) != fingerprint
            or not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_size) != len(payload)
        ):
            raise RawSupervisionAuditError("owned audit candidate changed while staged")
        retained.validate(allow_final_metadata_change=True)
        os.fsync(retained.directory_fd)
        return name, descriptor, fingerprint, _sha256_bytes(payload)
    except BaseException:
        fingerprint = _transaction_fingerprint(os.fstat(descriptor))
        current = _lstat_optional_at(retained.directory_fd, name)
        if current is not None and _transaction_fingerprint(current)[:2] == fingerprint[:2]:
            os.unlink(name, dir_fd=retained.directory_fd)
            os.fsync(retained.directory_fd)
        os.close(descriptor)
        raise


def _cleanup_owned_audit_candidate(
    retained: _RetainedDirectoryChain,
    *,
    candidate_name: str,
    candidate_descriptor: int,
    renamed: bool,
) -> bool:
    name = CANONICAL_AUDIT_REPORT.name if renamed else candidate_name
    try:
        opened = _transaction_fingerprint(os.fstat(candidate_descriptor))
        current = _lstat_optional_at(retained.directory_fd, name)
    except OSError:
        return False
    if current is None or _transaction_fingerprint(current)[:2] != opened[:2]:
        return False
    os.unlink(name, dir_fd=retained.directory_fd)
    os.fsync(retained.directory_fd)
    return True


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RawSupervisionAuditError("value is not strict JSON") from error


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RawSupervisionAuditError(f"{name} must be an exact integer")
    return value


def _strict_workers(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_WORKERS:
        raise ValueError(f"workers must be an exact integer in [1,{MAX_WORKERS}]")
    return value


_INTEGER_FIELD_NAMES = frozenset(
    {
        "byte_count",
        "development_scene_workers",
        "denied_or_unexpected_accesses",
        "endpoint_count",
        "endpoint_instance_count",
        "env_index",
        "episode_step",
        "expected_exact_record_count",
        "frame_index",
        "g2_label_payload_opens",
        "g2_rgb_byte_opens",
        "g2_sidecar_byte_opens",
        "g2_source_index_rows_read_for_exclusion",
        "g2_source_payload_opens",
        "geometry_contract_byte_opens",
        "global_row",
        "hardware_or_production_opens",
        "heldout_or_sealed_opens",
        "label_shard_row",
        "maximum_workers",
        "native_threads_per_worker",
        "pair_endpoint_references",
        "parent_label_shard_payload_opens",
        "render_audit_byte_opens",
        "render_plan_byte_opens",
        "render_summary_byte_opens",
        "reset_count",
        "rgb_byte_opens",
        "rgb_decodes",
        "row_count",
        "runtime_or_navigation_result_opens",
        "scene_shard_count",
        "shard_row",
        "source_frames_byte_opens",
        "source_frames_jsonl_records_scanned",
        "source_frames_selected_records",
        "source_payload_first_pass_file_count",
        "source_payload_second_pass_file_count",
        "source_payload_total_byte_opens",
        "source_scene_manifest_byte_opens",
        "timestamp_ns",
        "unique_endpoint_raycasts",
        "writes_outside_output_or_failure_namespace",
    }
)


def _strict_shape(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise RawSupervisionAuditError(f"{name} must be an exact shape list")
    return tuple(
        _strict_int(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )


def _validate_integer_fields(value: object, *, name: str) -> None:
    if type(value) is dict:
        for key, child in value.items():
            child_name = f"{name}.{key}"
            if key in {"shape", "trailing_shape"}:
                _strict_shape(child, name=child_name)
            elif key in _INTEGER_FIELD_NAMES:
                _strict_int(child, name=child_name)
            else:
                _validate_integer_fields(child, name=child_name)
    elif type(value) is list:
        for index, child in enumerate(value):
            _validate_integer_fields(child, name=f"{name}[{index}]")


def _validate_access_ledger_integers(value: object, *, name: str) -> None:
    if type(value) is not dict:
        raise RawSupervisionAuditError(f"{name} must be an exact object")
    for key, child in value.items():
        child_name = f"{name}.{key}"
        if type(child) is dict:
            _validate_access_ledger_integers(child, name=child_name)
        elif isinstance(child, (bool, int, float)):
            _strict_int(child, name=child_name)


def _object_pairs(name: str):
    def decode(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RawSupervisionAuditError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    return decode


def _decode_json(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RawSupervisionAuditError(f"{name} contains nonfinite {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_object_pairs(name),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawSupervisionAuditError(f"{name} is invalid JSON") from error
    if type(value) is not dict:
        raise RawSupervisionAuditError(f"{name} must be an exact object")
    return value


def _validate_content_hash(value: Mapping[str, Any], *, name: str) -> None:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise RawSupervisionAuditError(f"{name} content SHA-256 changed")


def _parse_canonical_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise RawSupervisionAuditError(
            f"{name} must be nonempty newline-terminated JSONL"
        )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(), start=1):
        row = _decode_json(line, name=f"{name}:{index}")
        if canonical_json_bytes(row) != line:
            raise RawSupervisionAuditError(f"{name}:{index} is not canonical JSON")
        _validate_content_hash(row, name=f"{name}:{index}")
        _validate_integer_fields(row, name=f"{name}:{index}")
        rows.append(row)
    return rows


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_real_directory(path: Path, *, name: str) -> Path:
    lexical = Path(path).absolute()
    try:
        metadata = lexical.stat(follow_symlinks=False)
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RawSupervisionAuditError(f"{name} is absent") from error
    if (
        lexical != resolved
        or lexical.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise PermissionError(f"{name} must be a canonical real directory")
    return lexical


def _canonical_relative_path(value: object, *, name: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise RawSupervisionAuditError(f"{name} must be a nonempty path")
    path = Path(value)
    if (
        path.is_absolute()
        or str(path) != value
        or os.path.normpath(value) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PermissionError(f"{name} must be canonical and relative")
    return path


def _resolve_regular_file(root: Path, relative: object, *, name: str) -> Path:
    rel = _canonical_relative_path(relative, name=name)
    current = root
    for component in rel.parts[:-1]:
        candidate = current / component
        metadata = candidate.stat(follow_symlinks=False)
        if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(f"{name} crosses a non-directory or alias")
        current = candidate
    path = current / rel.parts[-1]
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or int(metadata.st_nlink) != 1
    ):
        raise PermissionError(f"{name} must be an unaliased regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{name} escapes the dataset") from error
    return path


def _read_bound_file(
    root: Path,
    relative: object,
    *,
    expected_bytes: int,
    expected_sha256: str,
    name: str,
) -> bytes:
    path = _resolve_regular_file(root, relative, name=name)
    payload = _read_absolute_bound_payload(
        path.absolute(),
        expected_sha256,
        repository_root=root,
        name=name,
    )
    if len(payload) != expected_bytes:
        raise RawSupervisionAuditError(f"{name} bytes changed")
    return payload


def _tree_file_inventory(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        metadata = path.lstat()
        relative = str(path.relative_to(root))
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"dataset tree contains symlink {relative}")
        if stat.S_ISREG(metadata.st_mode):
            if int(metadata.st_nlink) != 1:
                raise PermissionError(
                    f"dataset tree contains hard-link alias {relative}"
                )
            result[relative] = path
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(f"dataset tree contains special entry {relative}")
    return result


def _validate_root_file_inventory(
    root: Path,
    records: object,
) -> dict[str, Mapping[str, Any]]:
    if type(records) is not list:
        raise RawSupervisionAuditError("manifest files must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if type(record) is not dict or set(record) != ROOT_FILE_FIELDS:
            raise RawSupervisionAuditError(f"manifest file {index} fields changed")
        relative = str(_canonical_relative_path(record["path"], name="manifest file"))
        if relative == "manifest.json" or relative in indexed:
            raise RawSupervisionAuditError("manifest file inventory repeats/self-includes")
        _strict_int(record.get("byte_count"), name=f"{relative}.byte_count")
        if not _is_sha256(record.get("file_sha256")):
            raise RawSupervisionAuditError(f"{relative} SHA-256 is malformed")
        indexed[relative] = record
    if list(indexed) != sorted(indexed):
        raise RawSupervisionAuditError("manifest file inventory is not ordered")
    observed = _tree_file_inventory(root)
    if set(observed) != set(indexed) | {"manifest.json"}:
        raise RawSupervisionAuditError("manifest and filesystem inventories differ")
    for relative, record in indexed.items():
        payload = _read_bound_file(
            root,
            relative,
            expected_bytes=int(record["byte_count"]),
            expected_sha256=str(record["file_sha256"]),
            name=f"dataset file {relative}",
        )
        del payload
    return indexed


def _validate_access_boundary(ledger: object) -> None:
    if type(ledger) is not dict or not ledger:
        raise RawSupervisionAuditError("access ledger must be a nonempty object")
    for name, value in ledger.items():
        if any(fragment in str(name).lower() for fragment in FORBIDDEN_LEDGER_FRAGMENTS):
            if type(value) is not int or value != 0:
                raise PermissionError(f"forbidden access ledger field is nonzero: {name}")


def _validate_manifest_constants(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != MANIFEST_FIELDS:
        raise RawSupervisionAuditError("dataset manifest fields changed")
    _validate_content_hash(manifest, name="dataset manifest")
    if (
        manifest.get("schema") != DATASET_SCHEMA
        or manifest.get("status") != "complete_pending_independent_audit"
        or manifest.get("evidence_schema") != EVIDENCE_SCHEMA
        or manifest.get("raster_schema") != RASTER_SCHEMA
        or manifest.get("roles") != list(DEVELOPMENT_ROLES)
    ):
        raise RawSupervisionAuditError("dataset manifest identity changed")
    expected_layout = [
        {"path": path, "dtype": np.dtype(dtype).str, "trailing_shape": list(shape)}
        for path, dtype, shape in ARRAY_LAYOUT
    ]
    if manifest.get("array_layout") != expected_layout:
        raise RawSupervisionAuditError("dataset array layout changed")
    licenses = manifest.get("licenses")
    if (
        type(licenses) is not dict
        or set(licenses) != FALSE_LICENSE_FIELDS
        or any(value is not False for value in licenses.values())
    ):
        raise PermissionError("unaudited dataset grants authority")
    if manifest.get("parallel_contract") != {
        "worker_start_method": "spawn",
        "maximum_workers": 6,
        "native_threads_per_worker": 1,
        "gpu_visible_to_workers": False,
        "merge_order": "role_then_scene_then_endpoint_identity",
        "worker_count_does_not_change_artifact_bytes": True,
    }:
        raise RawSupervisionAuditError("parallel construction contract changed")
    if manifest.get("publication") != {
        "staging": "private_sibling_directory_mode_0700",
        "commit": "single_renameat2_RENAME_NOREPLACE",
        "manifest_self_inventory": "canonical_content_sha256",
        "file_inventory": "every_regular_file_except_manifest_self",
    }:
        raise RawSupervisionAuditError("publication contract changed")
    _validate_access_boundary(manifest.get("access_ledger"))
def _parse_manifest(
    root: Path,
    *,
    expected_manifest_file_sha256: str,
) -> dict[str, Any]:
    if not _is_sha256(expected_manifest_file_sha256):
        raise RawSupervisionAuditError("expected manifest SHA-256 is malformed")
    path = _resolve_regular_file(root, "manifest.json", name="dataset manifest")
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_manifest_file_sha256:
        raise RawSupervisionAuditError("dataset manifest file SHA-256 changed")
    if not payload.endswith(b"\n"):
        raise RawSupervisionAuditError("dataset manifest lacks terminal newline")
    manifest = _decode_json(payload, name="dataset manifest")
    if canonical_json_bytes(manifest) + b"\n" != payload:
        raise RawSupervisionAuditError("dataset manifest is not canonical")
    _validate_integer_fields(manifest, name="dataset manifest")
    _validate_access_ledger_integers(
        manifest.get("access_ledger"),
        name="dataset manifest.access_ledger",
    )
    _validate_manifest_constants(manifest)
    return manifest


def _records_by_hash(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    name: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        digest = record.get(field)
        if not _is_sha256(digest) or digest in result:
            raise RawSupervisionAuditError(f"{name} hashes are malformed or repeated")
        result[str(digest)] = record
    return result


def _validate_pair_and_endpoint_indexes(
    root: Path,
    manifest: Mapping[str, Any],
    inputs: AuditInputs,
    file_records: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    pair_record = manifest.get("pair_index")
    endpoint_record = manifest.get("endpoint_index")
    for name, record, expected_path in (
        ("pair index", pair_record, "pairs.jsonl"),
        ("endpoint index", endpoint_record, "endpoints.jsonl"),
    ):
        if type(record) is not dict or set(record) != {"path", "row_count", "file_sha256"}:
            raise RawSupervisionAuditError(f"{name} manifest record changed")
        if record.get("path") != expected_path or record.get("file_sha256") != file_records[expected_path]["file_sha256"]:
            raise RawSupervisionAuditError(f"{name} file binding changed")
    pair_payload = _read_bound_file(
        root,
        "pairs.jsonl",
        expected_bytes=int(file_records["pairs.jsonl"]["byte_count"]),
        expected_sha256=str(file_records["pairs.jsonl"]["file_sha256"]),
        name="pair index",
    )
    endpoint_payload = _read_bound_file(
        root,
        "endpoints.jsonl",
        expected_bytes=int(file_records["endpoints.jsonl"]["byte_count"]),
        expected_sha256=str(file_records["endpoints.jsonl"]["file_sha256"]),
        name="endpoint index",
    )
    pairs = _parse_canonical_jsonl(pair_payload, name="pair index")
    endpoints = _parse_canonical_jsonl(endpoint_payload, name="endpoint index")
    if int(pair_record["row_count"]) != len(pairs) or int(endpoint_record["row_count"]) != len(endpoints):
        raise RawSupervisionAuditError("index row counts changed")
    expected_pairs = [dict(item) for item in inputs.plan.pairs]
    if pairs != expected_pairs:
        raise RawSupervisionAuditError("published pair index differs from metadata V5")
    if canonical_json_sha256([item["content_sha256"] for item in pairs]) != manifest.get("ordered_pair_sha256"):
        raise RawSupervisionAuditError("ordered pair hash changed")
    plan_endpoints = _records_by_hash(
        inputs.plan.endpoints,
        field="identity_sha256",
        name="metadata-plan endpoints",
    )
    seen: set[str] = set()
    previous_key: tuple[int, str] | None = None
    for index, row in enumerate(endpoints):
        if set(row) != TOP_ENDPOINT_FIELDS:
            raise RawSupervisionAuditError(f"endpoint index row {index} fields changed")
        digest = str(row.get("endpoint_identity_sha256", ""))
        planned = plan_endpoints.get(digest)
        if planned is None or digest in seen:
            raise RawSupervisionAuditError("endpoint index is extra, missing, or repeated")
        seen.add(digest)
        identity = planned.get("identity")
        if not isinstance(identity, Mapping):
            raise RawSupervisionAuditError("planned endpoint identity is absent")
        key = (DEVELOPMENT_ROLES.index(str(row["dataset_role"])), digest)
        if previous_key is not None and key <= previous_key:
            raise RawSupervisionAuditError("endpoint index order changed")
        previous_key = key
        if (
            row["schema"] != ENDPOINT_INDEX_SCHEMA
            or row["dataset_role"] != identity.get("dataset_role")
            or row["scene_id"] != identity.get("scene_id")
            or row["plan_endpoint_content_sha256"] != planned.get("content_sha256")
            or row["image_path_metadata_only"] != planned.get("image_path_metadata_only")
            or row["image_sha256_commitment_only"] != identity.get("image_sha256")
            or not _is_sha256(row.get("evidence_content_sha256"))
            or not _is_sha256(row.get("raster_content_sha256"))
        ):
            raise RawSupervisionAuditError("endpoint index disagrees with metadata V5")
        _strict_int(row.get("shard_row"), name="endpoint shard_row")
        expected_shard = f"shards/{hashlib.sha256(str(row['scene_id']).encode()).hexdigest()[:16]}/shard.json"
        if row.get("scene_shard") != expected_shard:
            raise RawSupervisionAuditError("endpoint scene-shard binding changed")
    if seen != set(plan_endpoints):
        raise RawSupervisionAuditError("endpoint index does not cover metadata V5")
    if canonical_json_sha256([item["content_sha256"] for item in endpoints]) != manifest.get("ordered_endpoint_sha256"):
        raise RawSupervisionAuditError("ordered endpoint hash changed")
    endpoint_by_hash = {str(item["endpoint_identity_sha256"]): item for item in endpoints}
    uses = Counter()
    for pair in pairs:
        for side in ("current", "next"):
            digest = str(pair[f"{side}_endpoint_sha256"])
            endpoint = endpoint_by_hash.get(digest)
            if endpoint is None:
                raise RawSupervisionAuditError("pair references an absent endpoint")
            if (
                endpoint["dataset_role"] != pair["dataset_role"]
                or endpoint["scene_id"] != pair["scene_id"]
                or endpoint["family"] != pair["family"]
            ):
                raise RawSupervisionAuditError("pair crossed endpoint role/scene/family")
            uses[digest] += 1
    if set(uses) != set(endpoint_by_hash):
        raise RawSupervisionAuditError("endpoint index contains an orphan")
    return pairs, endpoints, plan_endpoints


def _derive_population(
    manifest: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive every population from audited rows before frozen comparisons."""

    pair_counts: Counter[str] = Counter()
    reference_counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    pair_families: dict[str, set[str]] = defaultdict(set)
    endpoint_families: dict[str, set[str]] = defaultdict(set)
    endpoint_by_hash: dict[str, Mapping[str, Any]] = {}
    for index, endpoint in enumerate(endpoints):
        digest = endpoint.get("endpoint_identity_sha256")
        if not _is_sha256(digest) or digest in endpoint_by_hash:
            raise RawSupervisionAuditError(
                "endpoint population repeats or is malformed"
            )
        role = endpoint.get("dataset_role")
        family = endpoint.get("family")
        if type(role) is not str or type(family) is not str:
            raise RawSupervisionAuditError(
                f"endpoint {index} role/family changed"
            )
        endpoint_by_hash[str(digest)] = endpoint
        unique_counts[role] += 1
        endpoint_families[role].add(family)

    referenced: set[str] = set()
    for index, pair in enumerate(pairs):
        role = pair.get("dataset_role")
        family = pair.get("family")
        if type(role) is not str or type(family) is not str:
            raise RawSupervisionAuditError(f"pair {index} role/family changed")
        pair_counts[role] += 1
        pair_families[role].add(family)
        for side in ("current", "next"):
            digest = pair.get(f"{side}_endpoint_sha256")
            endpoint = endpoint_by_hash.get(str(digest))
            if endpoint is None:
                raise RawSupervisionAuditError("pair references an absent endpoint")
            if (
                endpoint.get("dataset_role") != role
                or endpoint.get("family") != family
                or endpoint.get("scene_id") != pair.get("scene_id")
            ):
                raise RawSupervisionAuditError(
                    "pair reference crossed role/family/scene"
                )
            reference_counts[role] += 1
            referenced.add(str(digest))
    if referenced != set(endpoint_by_hash):
        raise RawSupervisionAuditError("endpoint population contains an orphan")

    shard_records = manifest.get("shards")
    if type(shard_records) is not list:
        raise RawSupervisionAuditError("manifest shards must be an exact list")
    shard_families: dict[str, set[str]] = defaultdict(set)
    shard_scenes: set[str] = set()
    endpoints_per_scene = Counter(str(row.get("scene_id")) for row in endpoints)
    endpoint_scene_contract: dict[str, tuple[str, str]] = {}
    for endpoint in endpoints:
        scene = str(endpoint.get("scene_id"))
        contract = (
            str(endpoint.get("dataset_role")),
            str(endpoint.get("family")),
        )
        previous = endpoint_scene_contract.setdefault(scene, contract)
        if previous != contract:
            raise RawSupervisionAuditError(
                "endpoint scene crosses role/family populations"
            )
    for index, record in enumerate(shard_records):
        if type(record) is not dict:
            raise RawSupervisionAuditError(f"manifest shard {index} changed")
        role = record.get("dataset_role")
        family = record.get("family")
        scene = record.get("scene_id")
        if type(role) is not str or type(family) is not str or type(scene) is not str:
            raise RawSupervisionAuditError(
                "manifest shard role/family/scene changed"
            )
        if scene in shard_scenes:
            raise RawSupervisionAuditError("manifest shard scene repeats")
        shard_scenes.add(scene)
        shard_families[role].add(family)
        if endpoint_scene_contract.get(scene) != (role, family):
            raise RawSupervisionAuditError(
                "shard scene crosses endpoint role/family populations"
            )
        if _strict_int(
            record.get("endpoint_count"),
            name=f"manifest shard {index}.endpoint_count",
        ) != endpoints_per_scene[scene]:
            raise RawSupervisionAuditError(
                "shard count disagrees with endpoint rows"
            )
    if shard_scenes != set(endpoints_per_scene):
        raise RawSupervisionAuditError("shard scenes do not cover endpoint rows")

    roles = tuple(manifest.get("roles", ()))
    actual_roles = set(pair_counts) | set(unique_counts) | set(reference_counts)
    if (
        list(roles) != list(DEVELOPMENT_ROLES)
        or not actual_roles
        or not actual_roles <= set(roles)
    ):
        raise RawSupervisionAuditError("actual role population changed")
    for role in roles:
        if not (
            pair_families[role]
            == endpoint_families[role]
            == shard_families[role]
        ):
            raise RawSupervisionAuditError(
                f"actual family population disagrees in role {role}"
            )

    observed_pairs = {role: pair_counts[role] for role in roles}
    observed_unique = {role: unique_counts[role] for role in roles}
    observed_references = {role: reference_counts[role] for role in roles}
    if manifest.get("pair_counts") != observed_pairs:
        raise RawSupervisionAuditError(
            "declared pair counts differ from audited rows"
        )
    if manifest.get("unique_endpoint_counts") != observed_unique:
        raise RawSupervisionAuditError(
            "declared unique-endpoint counts differ from audited rows"
        )
    reference_total = sum(observed_references.values())
    if manifest.get("endpoint_instance_count") != reference_total:
        raise RawSupervisionAuditError(
            "declared endpoint-reference count differs from audited rows"
        )
    if manifest.get("scene_shard_count") != len(shard_records):
        raise RawSupervisionAuditError(
            "declared scene-shard count differs from audited rows"
        )
    return {
        "pair_counts": observed_pairs,
        "pair_count": len(pairs),
        "endpoint_reference_counts": observed_references,
        "endpoint_reference_count": reference_total,
        "unique_endpoint_counts": observed_unique,
        "unique_endpoint_count": len(endpoints),
        "role_count": len(actual_roles),
        "family_counts": {
            role: len(pair_families[role]) for role in roles
        },
        "scene_shard_count": len(shard_records),
    }


def _validate_frozen_population(population: Mapping[str, Any]) -> None:
    expected = {
        "pair_counts": FROZEN_PAIR_COUNTS,
        "pair_count": FROZEN_PAIR_COUNT,
        "endpoint_reference_count": FROZEN_ENDPOINT_REFERENCE_COUNT,
        "unique_endpoint_counts": FROZEN_UNIQUE_ENDPOINT_COUNTS,
        "unique_endpoint_count": FROZEN_UNIQUE_ENDPOINT_COUNT,
        "role_count": len(DEVELOPMENT_ROLES),
        "family_counts": {
            role: FROZEN_FAMILY_COUNT_PER_ROLE for role in DEVELOPMENT_ROLES
        },
        "scene_shard_count": FROZEN_SCENE_SHARD_COUNT,
    }
    for field, value in expected.items():
        if population.get(field) != value:
            raise RawSupervisionAuditError(
                f"frozen actual population changed at {field}"
            )


def _sample_records(endpoint_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[(str(row["dataset_role"]), str(row["family"]))].append(row)
    records: list[dict[str, Any]] = []
    for role, family in sorted(grouped):
        scored = [
            (
                hashlib.sha256(
                    role.encode("utf-8")
                    + b"\0"
                    + family.encode("utf-8")
                    + b"\0"
                    + str(row["endpoint_identity_sha256"]).encode("ascii")
                ).hexdigest(),
                row,
            )
            for row in grouped[(role, family)]
        ]
        score, chosen = min(scored, key=lambda item: item[0])
        records.append(
            {
                "dataset_role": role,
                "family": family,
                "endpoint_identity_sha256": chosen["endpoint_identity_sha256"],
                "selection_sha256": score,
            }
        )
    return records


def _validate_sample_precommit(
    manifest: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = _sample_records(endpoints)
    expected = {
        "scheme": "minimum_sha256_role_nul_family_nul_endpoint_identity_v1",
        "one_endpoint_per_observed_role_family": True,
        "expected_exact_record_count": 24,
        "records": records,
        "records_sha256": canonical_json_sha256(records),
    }
    if manifest.get("independent_audit_precommit") != expected:
        raise RawSupervisionAuditError("independent audit sample precommit changed")
    role_groups = Counter(str(item["dataset_role"]) for item in records)
    if len(records) != EXPECTED_SAMPLE_COUNT or role_groups != Counter(
        {role: 8 for role in DEVELOPMENT_ROLES}
    ):
        raise RawSupervisionAuditError(
            "audit sample is not eight families in each development role"
        )
    return records


def _read_array(
    directory: Path,
    record: Mapping[str, Any],
    *,
    endpoint_count: int,
    trailing_shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    if set(record) != SHARD_FILE_FIELDS:
        raise RawSupervisionAuditError("shard array file record changed")
    expected_shape = (endpoint_count, *trailing_shape)
    if record.get("dtype") != np.dtype(dtype).str or record.get("shape") != list(expected_shape):
        raise RawSupervisionAuditError("shard array dtype/shape changed")
    payload = _read_bound_file(
        directory,
        record["path"],
        expected_bytes=int(record["byte_count"]),
        expected_sha256=str(record["file_sha256"]),
        name=f"shard array {record['path']}",
    )
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if len(payload) != expected_bytes:
        raise RawSupervisionAuditError("shard array byte count disagrees with shape")
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(expected_shape)


def _stored_arrays_from_evidence(evidence: Any, raster: Any) -> tuple[np.ndarray, ...]:
    return (
        np.ascontiguousarray(evidence.camera_origin_body_m, dtype="<f4"),
        np.ascontiguousarray(evidence.camera_basis_body_fru, dtype="<f4"),
        np.asarray(evidence.ground_plane_z_body_m, dtype="<f4"),
        np.ascontiguousarray(evidence.ground_support_in_frustum, dtype=np.uint8),
        np.ascontiguousarray(evidence.ground_support_clear_to_target, dtype=np.uint8),
        np.ascontiguousarray(evidence.pixel_hit_mask, dtype=np.uint8),
        np.ascontiguousarray(evidence.pixel_first_hit_distance_m, dtype="<f4"),
        np.ascontiguousarray(raster.output_labels, dtype=np.uint8),
    )


def _validate_shards(
    root: Path,
    manifest: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    file_records: Mapping[str, Mapping[str, Any]],
    sample_hashes: set[str],
) -> dict[str, StoredEndpointEvidence]:
    shard_records = manifest.get("shards")
    if type(shard_records) is not list:
        raise RawSupervisionAuditError("manifest shards must be a list")
    endpoint_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for endpoint in endpoints:
        endpoint_by_shard[str(endpoint["scene_shard"])].append(endpoint)
    seen_endpoints: set[str] = set()
    samples: dict[str, StoredEndpointEvidence] = {}
    seen_directories: set[str] = set()
    previous_scene = ""
    for shard_record in shard_records:
        if type(shard_record) is not dict or set(shard_record) != {
            "path", "dataset_role", "family", "scene_id", "endpoint_count", "content_sha256"
        }:
            raise RawSupervisionAuditError("manifest shard record fields changed")
        scene_id = str(shard_record["scene_id"])
        if scene_id <= previous_scene:
            raise RawSupervisionAuditError("manifest shard order changed")
        previous_scene = scene_id
        directory_name = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:16]
        relative = f"shards/{directory_name}/shard.json"
        if shard_record["path"] != relative or directory_name in seen_directories:
            raise RawSupervisionAuditError("scene shard path collided or changed")
        seen_directories.add(directory_name)
        root_file = file_records.get(relative)
        if root_file is None:
            raise RawSupervisionAuditError("scene shard is absent from file inventory")
        payload = _read_bound_file(
            root,
            relative,
            expected_bytes=int(root_file["byte_count"]),
            expected_sha256=str(root_file["file_sha256"]),
            name=f"scene shard {scene_id}",
        )
        shard = _decode_json(payload, name=f"scene shard {scene_id}")
        if canonical_json_bytes(shard) + b"\n" != payload or set(shard) != SHARD_FIELDS:
            raise RawSupervisionAuditError("scene shard is noncanonical or fields changed")
        _validate_content_hash(shard, name=f"scene shard {scene_id}")
        if (
            shard["schema"] != SHARD_SCHEMA
            or shard["dataset_role"] != shard_record["dataset_role"]
            or shard["family"] != shard_record["family"]
            or shard["scene_id"] != scene_id
            or shard["scene_id_sha256"] != hashlib.sha256(scene_id.encode()).hexdigest()
            or shard["content_sha256"] != shard_record["content_sha256"]
        ):
            raise RawSupervisionAuditError("scene shard identity changed")
        count = _strict_int(shard["endpoint_count"], name="shard endpoint_count", minimum=1)
        if count != shard_record["endpoint_count"]:
            raise RawSupervisionAuditError("shard endpoint count changed")
        local_records = shard.get("files")
        if type(local_records) is not list:
            raise RawSupervisionAuditError("shard files must be a list")
        local_by_name: dict[str, Mapping[str, Any]] = {}
        for record in local_records:
            if type(record) is not dict or set(record) != SHARD_FILE_FIELDS:
                raise RawSupervisionAuditError("shard file record fields changed")
            name = str(_canonical_relative_path(record["path"], name="shard file"))
            if len(Path(name).parts) != 1 or name in local_by_name:
                raise RawSupervisionAuditError("shard file path repeats or is nested")
            local_by_name[name] = record
            root_relative = f"shards/{directory_name}/{name}"
            root_record = file_records.get(root_relative)
            if root_record is None or any(
                root_record[field] != record[field]
                for field in ("byte_count", "file_sha256")
            ):
                raise RawSupervisionAuditError("root/shard file inventories disagree")
        expected_names = {name for name, _dtype, _shape in ARRAY_LAYOUT} | {"index.jsonl"}
        if set(local_by_name) != expected_names:
            raise RawSupervisionAuditError("shard file inventory changed")
        directory = root / "shards" / directory_name
        arrays = {
            name: _read_array(
                directory,
                local_by_name[name],
                endpoint_count=count,
                trailing_shape=shape,
                dtype=dtype,
            )
            for name, dtype, shape in ARRAY_LAYOUT
        }
        for boolean_name in (
            "ground_support_in_frustum.u1",
            "ground_support_clear_to_target.u1",
            "pixel_hit_mask.u1",
        ):
            if not np.isin(arrays[boolean_name], (0, 1)).all():
                raise RawSupervisionAuditError("boolean evidence array contains another value")
        index_record = local_by_name["index.jsonl"]
        if index_record.get("dtype") != "canonical_jsonl" or index_record.get("shape") != [count]:
            raise RawSupervisionAuditError("shard index dtype/shape changed")
        index_payload = _read_bound_file(
            directory,
            "index.jsonl",
            expected_bytes=int(index_record["byte_count"]),
            expected_sha256=str(index_record["file_sha256"]),
            name=f"shard index {scene_id}",
        )
        rows = _parse_canonical_jsonl(index_payload, name=f"shard index {scene_id}")
        if len(rows) != count:
            raise RawSupervisionAuditError("shard index count changed")
        top_rows = endpoint_by_shard.get(relative, [])
        top_by_hash = {str(item["endpoint_identity_sha256"]): item for item in top_rows}
        if len(top_by_hash) != len(top_rows) or len(top_rows) != count:
            raise RawSupervisionAuditError("top endpoint/shard counts disagree")
        endpoint_hashes: list[str] = []
        evidence_hashes: list[str] = []
        raster_hashes: list[str] = []
        for row_index, row in enumerate(rows):
            if set(row) != SHARD_INDEX_FIELDS or row.get("shard_row") != row_index:
                raise RawSupervisionAuditError("shard index row fields/order changed")
            digest = str(row.get("endpoint_identity_sha256", ""))
            top = top_by_hash.get(digest)
            if top is None or digest in seen_endpoints:
                raise RawSupervisionAuditError("shard endpoint is absent or repeated")
            seen_endpoints.add(digest)
            expected_top = dict(row)
            expected_top.pop("content_sha256")
            expected_top["scene_shard"] = relative
            expected_top["content_sha256"] = canonical_json_sha256(expected_top)
            if top != expected_top:
                raise RawSupervisionAuditError("top endpoint differs from shard index")
            if (
                row["dataset_role"] != shard["dataset_role"]
                or row["family"] != shard["family"]
                or row["scene_id"] != scene_id
            ):
                raise RawSupervisionAuditError("shard index crossed role/family/scene")
            evidence = ObservableCameraRayEvidenceV4(
                camera_origin_body_m=arrays["camera_origin_body_m.f4"][row_index],
                camera_basis_body_fru=arrays["camera_basis_body_fru.f4"][row_index],
                ground_plane_z_body_m=float(arrays["ground_plane_z_body_m.f4"][row_index]),
                ground_support_in_frustum=arrays["ground_support_in_frustum.u1"][row_index].astype(bool),
                ground_support_clear_to_target=arrays["ground_support_clear_to_target.u1"][row_index].astype(bool),
                pixel_hit_mask=arrays["pixel_hit_mask.u1"][row_index].astype(bool),
                pixel_first_hit_distance_m=arrays["pixel_first_hit_distance_m.f4"][row_index],
            )
            raster = rasterize_observable_camera_ray_evidence_v4(evidence)
            if (
                evidence.content_sha256() != row["evidence_content_sha256"]
                or raster.content_sha256() != row["raster_content_sha256"]
                or not np.array_equal(
                    raster.output_labels,
                    arrays["raster_labels.u1"][row_index],
                )
            ):
                raise RawSupervisionAuditError("stored V4 evidence/raster changed")
            endpoint_hashes.append(digest)
            evidence_hashes.append(evidence.content_sha256())
            raster_hashes.append(raster.content_sha256())
            if digest in sample_hashes:
                samples[digest] = StoredEndpointEvidence(
                    endpoint_identity_sha256=digest,
                    arrays=tuple(
                        np.array(arrays[name][row_index], copy=True)
                        for name, _dtype, _shape in ARRAY_LAYOUT
                    ),
                    evidence_content_sha256=evidence.content_sha256(),
                    raster_content_sha256=raster.content_sha256(),
                )
        if (
            canonical_json_sha256(endpoint_hashes) != shard["ordered_endpoint_identity_sha256"]
            or canonical_json_sha256(evidence_hashes) != shard["ordered_evidence_sha256"]
            or canonical_json_sha256(raster_hashes) != shard["ordered_raster_sha256"]
        ):
            raise RawSupervisionAuditError("shard ordered hashes changed")
    if seen_endpoints != {str(item["endpoint_identity_sha256"]) for item in endpoints}:
        raise RawSupervisionAuditError("shards do not cover the endpoint index")
    if set(samples) != sample_hashes:
        raise RawSupervisionAuditError("not every precommitted sample is present")
    return samples


def _validate_one_shard_task(
    task: tuple[
        str,
        Mapping[str, Any],
        Sequence[Mapping[str, Any]],
        Mapping[str, Mapping[str, Any]],
        set[str],
    ],
) -> dict[str, StoredEndpointEvidence]:
    _set_worker_environment()
    authorization_sha256 = task[0]
    _require_exact_authority(authorization_sha256)
    _authorization, shard_record, endpoints, file_records, sample_hashes = task
    return _validate_shards(
        CANONICAL_DATASET,
        {"shards": [shard_record]},
        endpoints,
        file_records,
        sample_hashes,
    )


def _validate_shards_parallel(
    manifest: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    file_records: Mapping[str, Mapping[str, Any]],
    sample_hashes: set[str],
    *,
    authorization_sha256: str,
    workers: int,
) -> dict[str, StoredEndpointEvidence]:
    worker_count = _strict_workers(workers)
    _require_exact_authority(authorization_sha256)
    if worker_count == 1:
        return _validate_shards(
            CANONICAL_DATASET,
            manifest,
            endpoints,
            file_records,
            sample_hashes,
        )
    endpoint_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for endpoint in endpoints:
        endpoint_by_shard[str(endpoint["scene_shard"])].append(endpoint)
    tasks = []
    for shard_record in manifest["shards"]:
        path = str(shard_record["path"])
        directory_prefix = str(Path(path).parent) + "/"
        scoped_files = {
            name: record
            for name, record in file_records.items()
            if name == path or name.startswith(directory_prefix)
        }
        scoped_endpoints = endpoint_by_shard.get(path, [])
        scoped_sample = {
            str(item["endpoint_identity_sha256"])
            for item in scoped_endpoints
            if item["endpoint_identity_sha256"] in sample_hashes
        }
        tasks.append(
            (
                authorization_sha256,
                shard_record,
                scoped_endpoints,
                scoped_files,
                scoped_sample,
            )
        )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_exact_worker,
        initargs=(authorization_sha256,),
    ) as executor:
        partials = list(executor.map(_validate_one_shard_task, tasks))
    samples: dict[str, StoredEndpointEvidence] = {}
    for partial in partials:
        if set(samples) & set(partial):
            raise RawSupervisionAuditError("parallel shard audit repeated a sample")
        samples.update(partial)
    if set(samples) != sample_hashes:
        raise RawSupervisionAuditError("parallel shard audit missed a sample")
    return samples


def _compare_source_replay(
    sample: Sequence[Mapping[str, Any]],
    observed: Mapping[str, StoredEndpointEvidence],
    recomputed: Mapping[str, tuple[np.ndarray, ...]],
) -> list[dict[str, Any]]:
    wanted = {str(item["endpoint_identity_sha256"]) for item in sample}
    if set(recomputed) != wanted:
        raise RawSupervisionAuditError("source replay returned another endpoint set")
    results: list[dict[str, Any]] = []
    for record in sample:
        digest = str(record["endpoint_identity_sha256"])
        actual = observed[digest]
        replay = recomputed[digest]
        if len(replay) != len(ARRAY_LAYOUT):
            raise RawSupervisionAuditError("source replay array count changed")
        byte_hashes: list[str] = []
        for position, ((name, dtype, shape), expected, received) in enumerate(
            zip(ARRAY_LAYOUT, actual.arrays, replay)
        ):
            raw = np.asarray(received, dtype=np.dtype(dtype))
            normalized = (
                raw.reshape(())
                if shape == ()
                else np.ascontiguousarray(raw, dtype=np.dtype(dtype))
            )
            if normalized.shape != shape:
                raise RawSupervisionAuditError(
                    f"source replay {digest}:{name} shape changed at {position}"
                )
            if normalized.tobytes(order="C") != np.ascontiguousarray(expected).tobytes(order="C"):
                raise RawSupervisionAuditError(
                    f"source replay {digest}:{name} differs byte-for-byte"
                )
            byte_hashes.append(_sha256_bytes(normalized.tobytes(order="C")))
        results.append(
            {
                **dict(record),
                "array_byte_sha256": byte_hashes,
                "array_byte_sha256_set": canonical_json_sha256(byte_hashes),
                "passes": True,
            }
        )
    return results


def _audit_fixed_dataset(
    *,
    authorization_sha256: str,
    expected_manifest_file_sha256: str,
    inputs: AuditInputs,
    workers: int,
) -> dict[str, Any]:
    """Run the owned audit engine against the one fixed dataset."""

    _require_exact_authority(authorization_sha256)
    if not isinstance(inputs, AuditInputs):
        raise TypeError("inputs must be AuditInputs")
    worker_count = _strict_workers(workers)
    root = _require_real_directory(CANONICAL_DATASET, name="dataset root")
    manifest = _parse_manifest(
        root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    file_records = _validate_root_file_inventory(root, manifest.get("files"))
    pairs, endpoints, plan_endpoints = _validate_pair_and_endpoint_indexes(
        root, manifest, inputs, file_records
    )
    population = _derive_population(manifest, pairs, endpoints)
    _validate_frozen_population(population)
    sample = _validate_sample_precommit(manifest, endpoints)
    sample_hashes = {str(item["endpoint_identity_sha256"]) for item in sample}
    observed = _validate_shards_parallel(
        manifest,
        endpoints,
        file_records,
        sample_hashes,
        authorization_sha256=authorization_sha256,
        workers=worker_count,
    )
    recomputed = _exact_sample_recomputer(
        sample,
        plan_endpoints,
        inputs,
        worker_count,
        authorization_sha256=authorization_sha256,
    )
    sample_results = _compare_source_replay(sample, observed, recomputed)
    _validate_root_file_inventory(root, manifest.get("files"))
    _read_absolute_bound_payload(
        (root / "manifest.json").absolute(),
        expected_manifest_file_sha256,
        repository_root=root,
        name="dataset manifest final revalidation",
    )
    core = {
        "schema": AUDIT_SCHEMA,
        "verdict": "PASS",
        "dataset_manifest_file_sha256": expected_manifest_file_sha256,
        "dataset_manifest_content_sha256": manifest["content_sha256"],
        "pair_count": len(inputs.plan.pairs),
        "unique_endpoint_count": len(endpoints),
        "scene_shard_count": len(manifest["shards"]),
        "sample_count": len(sample_results),
        "sample_results": sample_results,
        "sample_results_sha256": canonical_json_sha256(sample_results),
        "observed_population": population,
        "strict_integer_cardinalities": True,
        "unaliased_descriptor_bound_dataset_leaves": True,
        "full_byte_inventory_revalidated": True,
        "pair_endpoint_joins_reconstructed": True,
        "all_stored_evidence_and_rasters_recomputed": True,
        "sample_original_geometry_recomputed": True,
        "dataset_use_authorized": False,
        "training_authorized": False,
        "g2_authorized": False,
        "production_authorized": False,
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _set_worker_environment() -> None:
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    for name in ACCELERATOR_ENVIRONMENT:
        os.environ[name] = ""


def _initialize_exact_worker(authorization_sha256: str) -> None:
    """Authorize a spawned process before it deserializes a task payload."""

    _set_worker_environment()
    _require_exact_authority(authorization_sha256)


def _source_file_records(inventory: DevelopmentSourceInventory) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in inventory.records:
        records.extend(
            (
                {"scene_id": scene["scene_id"], "kind": "frames", **scene["frames"]},
                {"scene_id": scene["scene_id"], "kind": "scene_manifest", **scene["scene_manifest"]},
                {"scene_id": scene["scene_id"], "kind": "render_plan", **scene["render_plan"]},
                {"scene_id": scene["scene_id"], "kind": "render_summary", **scene["render_summary"]},
            )
        )
    return records


def _hash_source_file(
    task: tuple[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _set_worker_environment()
    authorization_sha256, record = task
    _require_exact_authority(authorization_sha256)
    path = Path(str(record["path"]))
    expected = str(record.get("sha256", record.get("file_sha256", "")))
    payload = _read_absolute_bound_payload(
        path,
        expected,
        repository_root=ROOT,
        name=f"allowed {record['kind']} source",
    )
    result = {
        "scene_id": record["scene_id"],
        "kind": record["kind"],
        "path": str(path),
        "byte_count": len(payload),
        "file_sha256": _sha256_bytes(payload),
    }
    if record["kind"] == "frames":
        if not payload or not payload.endswith(b"\n") or b"\n\n" in payload:
            raise RawSupervisionAuditError("allowed frames source is malformed JSONL")
        result["jsonl_record_count"] = len(payload.splitlines())
    return result


def _hash_complete_source_inventory(
    inventory: DevelopmentSourceInventory,
    *,
    authorization_sha256: str,
    workers: int,
) -> list[dict[str, Any]]:
    worker_count = _strict_workers(workers)
    _require_exact_authority(authorization_sha256)
    records = _source_file_records(inventory)
    tasks = [(authorization_sha256, record) for record in records]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_exact_worker,
        initargs=(authorization_sha256,),
    ) as executor:
        results = list(executor.map(_hash_source_file, tasks))
    results.sort(key=lambda item: (item["scene_id"], item["kind"]))
    if len(results) != 352:
        raise RawSupervisionAuditError("allowed development source inventory is not 352 files")
    return results


def _parent_contract_receipts(
    authorization_sha256: str,
) -> list[dict[str, Any]]:
    _require_exact_authority(authorization_sha256)
    from scripts import audit_go2_n32_camera_frustum_observability as source_v4

    geometry_raw = _read_absolute_bound_payload(
        GEOMETRY_CONTRACT_PATH,
        GEOMETRY_CONTRACT_FILE_SHA256,
        repository_root=ROOT,
        name="physical geometry contract",
    )
    render_raw = _read_absolute_bound_payload(
        RENDER_AUDIT_PATH,
        RENDER_AUDIT_FILE_SHA256,
        repository_root=ROOT,
        name="render audit contract",
    )
    geometry = _decode_json(geometry_raw, name="physical geometry contract")
    render = _decode_json(render_raw, name="render audit contract")
    if source_v4._geometry_semantic_sha256(geometry) != GEOMETRY_CONTRACT_CONTENT_SHA256:
        raise RawSupervisionAuditError("physical geometry semantic hash changed")
    source_v4._geometry_flags(geometry)
    source_v4._validate_render_audit_contract(
        render, expected_content_sha256=RENDER_AUDIT_CONTENT_SHA256
    )
    return [
        {
            "path": str(GEOMETRY_CONTRACT_PATH),
            "file_sha256": GEOMETRY_CONTRACT_FILE_SHA256,
            "byte_count": len(geometry_raw),
            "purpose": "geometry_contract",
        },
        {
            "path": str(RENDER_AUDIT_PATH),
            "file_sha256": RENDER_AUDIT_FILE_SHA256,
            "byte_count": len(render_raw),
            "purpose": "render_audit",
        },
    ]


def _builder_source_receipts(
    hashed_sources: Sequence[Mapping[str, Any]],
    parent_contracts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    purpose_by_kind = {
        "frames": "source_frames_jsonl",
        "scene_manifest": "source_scene_manifest",
        "render_plan": "render_plan",
        "render_summary": "render_summary",
    }
    receipts = [dict(item) for item in parent_contracts]
    for item in hashed_sources:
        receipts.append(
            {
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "byte_count": item["byte_count"],
                "purpose": purpose_by_kind[str(item["kind"])],
                "scene_id": item["scene_id"],
            }
        )
    receipts.sort(
        key=lambda item: (
            str(item["path"]),
            str(item["purpose"]),
            str(item.get("scene_id", "")),
        )
    )
    if len(receipts) != 354:
        raise RawSupervisionAuditError("builder source receipt population changed")
    return receipts


def _validate_exact_access_ledger(
    value: object,
    *,
    inputs: AuditInputs,
    frames_scanned: int,
) -> None:
    if type(value) is not dict or set(value) != EXACT_ACCESS_LEDGER_KEYS:
        raise RawSupervisionAuditError("exact builder access-ledger fields changed")
    expected_scalars = {
        "schema": ACCESS_LEDGER_SCHEMA,
        "measurement_scope": (
            "controlled_data_opens_excluding_import_and_reviewed_source_hash_reads"
        ),
        "development_scene_workers": 88,
        "unique_endpoint_raycasts": 9460,
        "pair_endpoint_references": 10344,
        "source_frames_jsonl_records_scanned": int(frames_scanned),
        "source_frames_selected_records": 9460,
        "source_frames_byte_opens": 176,
        "source_scene_manifest_byte_opens": 176,
        "render_plan_byte_opens": 176,
        "render_summary_byte_opens": 176,
        "geometry_contract_byte_opens": 2,
        "render_audit_byte_opens": 2,
        "source_payload_first_pass_file_count": 354,
        "source_payload_second_pass_file_count": 354,
        "source_payload_total_byte_opens": 708,
        "g2_source_index_rows_read_for_exclusion": 8,
        "g2_sidecar_byte_opens": 0,
        "g2_source_payload_opens": 0,
        "g2_label_payload_opens": 0,
        "g2_rgb_byte_opens": 0,
        "rgb_byte_opens": 0,
        "rgb_decodes": 0,
        "parent_label_shard_payload_opens": 0,
        "checkpoint_or_model_output_opens": 0,
        "runtime_or_navigation_result_opens": 0,
        "heldout_or_sealed_opens": 0,
        "hardware_or_production_opens": 0,
        "writes_outside_output_or_failure_namespace": 0,
        "denied_or_unexpected_accesses": 0,
    }
    for name, expected in expected_scalars.items():
        if value.get(name) != expected:
            raise RawSupervisionAuditError(f"exact access ledger changed at {name}")
    if (
        value["metadata_plan_first_pass"] != inputs.plan.value["access_ledger"]
        or value["metadata_plan_second_pass"] != inputs.plan.value["access_ledger"]
        or value["metadata_source_inventory_first_pass"]
        != inputs.inventory.access_ledger
        or value["metadata_source_inventory_second_pass"]
        != inputs.inventory.access_ledger
    ):
        raise RawSupervisionAuditError("exact metadata access receipts changed")


def _validate_frozen_source_map(mapping: object, expected: Mapping[str, str], *, name: str) -> None:
    if mapping != dict(expected):
        raise RawSupervisionAuditError(f"{name} provenance map changed")
    for relative, digest in expected.items():
        payload = _read_absolute_bound_payload(
            ROOT / relative,
            digest,
            repository_root=ROOT,
            name=f"{name} {relative}",
        )
        del payload


def _validate_exact_input_provenance(
    value: object,
    *,
    inputs: AuditInputs,
    receipts: Sequence[Mapping[str, Any]],
    authorization: AcceptedAuthorizationV6,
) -> None:
    if type(value) is not dict or set(value) != EXACT_INPUT_PROVENANCE_FIELDS:
        raise RawSupervisionAuditError("exact input-provenance fields changed")
    if (
        value["authorization_file_sha256"]
        != authorization.authorization_file_sha256
        or value["authorization_content_sha256"]
        != authorization.authorization_content_sha256
        or value["authorization_source_map_sha256"]
        != authorization.source_map_sha256
        or value["metadata_plan_content_sha256"] != EXACT_PLAN_CONTENT_SHA256
        or value["metadata_plan_content_sha256"] != inputs.plan.value["content_sha256"]
        or value["metadata_ordered_pair_sha256"] != EXACT_ORDERED_PAIR_SHA256
        or value["metadata_ordered_pair_sha256"] != inputs.plan.value["ordered_pair_sha256"]
        or value["metadata_ordered_endpoint_sha256"] != EXACT_ORDERED_ENDPOINT_SHA256
        or value["metadata_ordered_endpoint_sha256"]
        != inputs.plan.value["ordered_endpoint_sha256"]
        or value["source_inventory_sha256"] != dict(inputs.inventory.hashes)
        or value["source_inventory_sha256"] != dict(SOURCE_INVENTORY_SHA256)
    ):
        raise RawSupervisionAuditError("exact metadata/source provenance changed")
    _validate_frozen_source_map(
        value["frozen_parent_file_sha256"],
        FROZEN_PARENT_FILE_SHA256,
        name="frozen parent",
    )
    _validate_frozen_source_map(
        value["reviewed_v4_source_sha256"],
        REVIEWED_V4_SOURCE_SHA256,
        name="reviewed V4 source",
    )
    if (
        value["geometry_contract_file_sha256"] != GEOMETRY_CONTRACT_FILE_SHA256
        or value["geometry_contract_content_sha256"]
        != GEOMETRY_CONTRACT_CONTENT_SHA256
        or value["render_audit_file_sha256"] != RENDER_AUDIT_FILE_SHA256
        or value["render_audit_content_sha256"] != RENDER_AUDIT_CONTENT_SHA256
        or value["source_payload_inventory"] != list(receipts)
        or value["source_payload_inventory_sha256"]
        != canonical_json_sha256(list(receipts))
    ):
        raise RawSupervisionAuditError("exact source payload provenance changed")


def _strict_canonical_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    value = _decode_json(raw, name=name)
    if raw != canonical_json_bytes(value) + b"\n":
        raise RawSupervisionAuditError(f"{name} is not canonical JSON")
    return value


def _authorization_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise RawSupervisionAuditError(
            "authorization source path is noncanonical"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RawSupervisionAuditError(
            "authorization source path is noncanonical"
        )
    return value


def _source_binding(entry: object) -> SourceBindingV6:
    if type(entry) is not dict or set(entry) != SOURCE_ENTRY_FIELDS:
        raise RawSupervisionAuditError(
            "authorization source entry is malformed"
        )
    role = entry["role"]
    digest = entry["sha256"]
    if type(role) is not str or not role or not _is_sha256(digest):
        raise RawSupervisionAuditError(
            "authorization source entry is malformed"
        )
    return SourceBindingV6(
        role=role,
        path=_authorization_relative_path(entry["path"]),
        sha256=digest,
    )


def _candidate_bindings(
    value: object,
    *,
    roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV6],
) -> tuple[SourceBindingV6, ...]:
    if type(value) is not list or len(value) != len(roles):
        raise RawSupervisionAuditError(
            "authorization review candidate changed"
        )
    candidate = tuple(_source_binding(entry) for entry in value)
    if tuple(item.role for item in candidate) != tuple(roles):
        raise RawSupervisionAuditError(
            "authorization review candidate roles changed"
        )
    if candidate != tuple(source_by_role[role] for role in roles):
        raise RawSupervisionAuditError(
            "authorization review candidate binding changed"
        )
    return candidate


def _review_binding(
    value: object,
    *,
    kind: str,
    review_role: str,
    review_schema: str,
    implementation_author: str,
    candidate_roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV6],
) -> ReviewBindingV6:
    if type(value) is not dict or set(value) != REVIEW_BINDING_FIELDS:
        raise RawSupervisionAuditError(
            f"authorization {kind} review binding is malformed"
        )
    if (
        value["schema"] != REVIEW_BINDING_SCHEMA
        or value["review_schema"] != review_schema
        or value["verdict"] != "PASS"
    ):
        raise PermissionError(
            f"authorization {kind} review is not a bound PASS"
        )
    reviewer = value["reviewer"]
    author = value["implementation_author"]
    if (
        type(reviewer) is not str
        or not reviewer
        or type(author) is not str
        or author != implementation_author
        or reviewer == author
    ):
        raise PermissionError(
            f"authorization {kind} review lacks a distinct reviewer"
        )
    review_source = source_by_role[review_role]
    path = _authorization_relative_path(value["path"])
    if (
        path != review_source.path
        or value["file_sha256"] != review_source.sha256
        or not _is_sha256(value["content_sha256"])
    ):
        raise RawSupervisionAuditError(
            f"authorization {kind} review file binding changed"
        )
    candidate = _candidate_bindings(
        value["candidate"],
        roles=candidate_roles,
        source_by_role=source_by_role,
    )
    return ReviewBindingV6(
        kind=kind,
        review_schema=review_schema,
        verdict="PASS",
        reviewer=reviewer,
        implementation_author=author,
        path=path,
        file_sha256=review_source.sha256,
        content_sha256=value["content_sha256"],
        candidate=candidate,
    )


def _validate_authorization_phase_one(
    payload: Mapping[str, Any],
    *,
    authorization_file_sha256: str,
) -> PhaseOneAuthorizationV6:
    """Validate complete V6 authority without opening a mapped target."""

    if type(payload) is not dict or set(payload) != AUTHORIZATION_FIELDS:
        raise RawSupervisionAuditError(
            "build authorization object fields changed"
        )
    if not _is_sha256(authorization_file_sha256):
        raise PermissionError("build authorization file hash is not frozen")
    if (
        payload["schema"] != AUTHORIZATION_SCHEMA
        or payload["exact_build_authorized_after_independent_reviews"] is not True
    ):
        raise PermissionError("raw-supervision exact audit is not authorized")
    core = dict(payload)
    declared = core.pop("content_sha256")
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise RawSupervisionAuditError(
            "build authorization content hash changed"
        )
    source_map = payload["source_map"]
    if type(source_map) is not list or len(source_map) != 9:
        raise RawSupervisionAuditError(
            "build authorization source map is incomplete"
        )
    sources = tuple(_source_binding(entry) for entry in source_map)
    roles = tuple(item.role for item in sources)
    paths = tuple(item.path for item in sources)
    if len(set(roles)) != 9 or len(set(paths)) != 9:
        raise RawSupervisionAuditError(
            "authorization source map repeats a role/path"
        )
    if roles != SOURCE_ROLES:
        raise RawSupervisionAuditError(
            "authorization source roles/order changed"
        )
    if paths != tuple(path for _role, path in SOURCE_ROLE_PATHS):
        raise RawSupervisionAuditError(
            "authorization role-to-path mapping changed"
        )
    source_by_role = {item.role: item for item in sources}
    if any(
        source_by_role[role].sha256 != digest
        for role, digest in FROZEN_BUILDER_V6_ROLE_SHA256.items()
    ):
        raise PermissionError(
            "authorization does not bind the frozen Builder V5 candidate"
        )
    builder_review = _review_binding(
        payload["builder_review"],
        kind="builder",
        review_role="builder_review",
        review_schema=BUILDER_REVIEW_SCHEMA,
        implementation_author=BUILDER_IMPLEMENTATION_AUTHOR,
        candidate_roles=BUILDER_CANDIDATE_ROLES,
        source_by_role=source_by_role,
    )
    auditor_review = _review_binding(
        payload["auditor_review"],
        kind="auditor",
        review_role="auditor_review",
        review_schema=AUDITOR_REVIEW_SCHEMA,
        implementation_author=AUDITOR_IMPLEMENTATION_AUTHOR,
        candidate_roles=AUDITOR_CANDIDATE_ROLES,
        source_by_role=source_by_role,
    )
    if builder_review.implementation_author == auditor_review.implementation_author:
        raise PermissionError(
            "builder and auditor implementations are not independent"
        )
    if builder_review.reviewer == auditor_review.reviewer:
        raise PermissionError("builder and auditor reviews are not independent")
    normalized = json.loads(canonical_json_bytes(payload))
    return PhaseOneAuthorizationV6(
        authorization_file_sha256=authorization_file_sha256,
        authorization_content_sha256=declared,
        source_map_sha256=canonical_json_sha256(source_map),
        canonical_payload=canonical_json_bytes(normalized),
        sources=sources,
        builder_review=builder_review,
        auditor_review=auditor_review,
    )


def _expected_review_authority(kind: str) -> dict[str, bool]:
    return {
        f"{kind}_source_approved": True,
        **{field: False for field in REVIEW_AUTHORITY_FALSE_FIELDS},
    }


def _review_candidate_value(
    candidate: Sequence[SourceBindingV6],
) -> list[dict[str, str]]:
    return [
        {"role": item.role, "path": item.path, "sha256": item.sha256}
        for item in candidate
    ]


def _validate_review_record(raw: bytes, binding: ReviewBindingV6) -> None:
    review = _strict_canonical_json_object(
        raw, name=f"{binding.kind} independent review"
    )
    if set(review) != REVIEW_RECORD_FIELDS:
        raise RawSupervisionAuditError(
            f"{binding.kind} independent review fields changed"
        )
    core = dict(review)
    declared = core.pop("content_sha256")
    if (
        not _is_sha256(declared)
        or canonical_json_sha256(core) != declared
        or declared != binding.content_sha256
    ):
        raise RawSupervisionAuditError(
            f"{binding.kind} independent review content hash changed"
        )
    if (
        review["schema"] != binding.review_schema
        or review["verdict"] != "PASS"
        or review["reviewer"] != binding.reviewer
        or review["implementation_author"] != binding.implementation_author
        or review["candidate"] != _review_candidate_value(binding.candidate)
        or review["authority"] != _expected_review_authority(binding.kind)
    ):
        raise PermissionError(
            f"{binding.kind} independent review PASS binding changed"
        )


def _validate_authorization_phase_two(
    phase_one: PhaseOneAuthorizationV6,
) -> AcceptedAuthorizationV6:
    """Rehash only the fixed V5 closure after structural acceptance."""

    if type(phase_one) is not PhaseOneAuthorizationV6:
        raise TypeError("phase two requires a completed V6 phase-one capsule")
    embedded = _strict_canonical_json_object(
        phase_one.canonical_payload + b"\n",
        name="phase-one authorization capsule",
    )
    revalidated = _validate_authorization_phase_one(
        embedded,
        authorization_file_sha256=phase_one.authorization_file_sha256,
    )
    if revalidated != phase_one:
        raise PermissionError("phase-one authorization capsule was fabricated")
    payload_by_role: dict[str, bytes] = {}
    for source in phase_one.sources:
        payload_by_role[source.role] = _read_absolute_bound_payload(
            (ROOT / source.path).absolute(),
            source.sha256,
            repository_root=ROOT,
            name=f"authorized source {source.role}",
        )
    _validate_review_record(
        payload_by_role["builder_review"], phase_one.builder_review
    )
    _validate_review_record(
        payload_by_role["auditor_review"], phase_one.auditor_review
    )
    for relative, digest in {
        **FROZEN_V6_PREDECESSOR_SHA256,
        **REVIEWED_V4_SOURCE_SHA256,
    }.items():
        _read_absolute_bound_payload(
            (ROOT / relative).absolute(),
            digest,
            repository_root=ROOT,
            name=f"frozen V6 authority source {relative}",
        )
    return AcceptedAuthorizationV6(
        authorization_file_sha256=phase_one.authorization_file_sha256,
        authorization_content_sha256=phase_one.authorization_content_sha256,
        source_map_sha256=phase_one.source_map_sha256,
        sources=phase_one.sources,
    )


def _require_exact_authority(
    authorization_sha256: str,
) -> AcceptedAuthorizationV6:
    if not _is_sha256(authorization_sha256):
        raise PermissionError("build authorization file hash is not frozen")
    if not BUILD_AUTHORIZATION_PATH.is_file():
        raise PermissionError(
            "reviewed raw-supervision build authorization is absent"
        )
    raw = _read_absolute_bound_payload(
        BUILD_AUTHORIZATION_PATH,
        authorization_sha256,
        repository_root=ROOT,
        name="build authorization",
    )
    payload = _strict_canonical_json_object(raw, name="build authorization")
    phase_one = _validate_authorization_phase_one(
        payload,
        authorization_file_sha256=authorization_sha256,
    )
    return _validate_authorization_phase_two(phase_one)


def _validate_exact_manifest_bindings(
    manifest: Mapping[str, Any],
    *,
    inputs: AuditInputs,
    hashed_sources: Sequence[Mapping[str, Any]],
    parent_contracts: Sequence[Mapping[str, Any]],
    authorization: AcceptedAuthorizationV6,
) -> None:
    frames_scanned = sum(
        int(item.get("jsonl_record_count", 0))
        for item in hashed_sources
        if item["kind"] == "frames"
    )
    receipts = _builder_source_receipts(hashed_sources, parent_contracts)
    _validate_exact_access_ledger(
        manifest.get("access_ledger"),
        inputs=inputs,
        frames_scanned=frames_scanned,
    )
    _validate_exact_input_provenance(
        manifest.get("input_provenance"),
        inputs=inputs,
        receipts=receipts,
        authorization=authorization,
    )


def _read_exact_source_json(
    path: str,
    expected_sha256: str,
    *,
    name: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    _require_exact_authority(authorization_sha256)
    payload = _read_absolute_bound_payload(
        Path(path), expected_sha256, repository_root=ROOT, name=name
    )
    return _decode_json(payload, name=name)


def _source_record_for_endpoint(
    endpoint_digest: str,
    endpoint: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    for pair in pairs:
        for side_rank, side in enumerate(("current", "next")):
            if pair.get(f"{side}_endpoint_sha256") == endpoint_digest:
                candidates.append((int(pair["global_row"]), side, pair))
    if not candidates:
        raise RawSupervisionAuditError("sample endpoint has no pair occurrence")
    _global, side, pair = min(candidates, key=lambda item: (item[0], item[1]))
    identity = endpoint["identity"]
    return {
        "family": pair["family"],
        "scene_id": identity["scene_id"],
        "global_row": pair["global_row"],
        "side": side,
        "image_path_metadata_only": endpoint["image_path_metadata_only"],
        "image_sha256": identity["image_sha256"],
        "label_shard_sha256": pair["label_shard_sha256"],
        "label_row": pair["label_shard_row"],
        "frame_index": identity["frame_index"],
        "env_index": identity["env_index"],
        "timestamp_ns": identity["timestamp_ns"],
        "episode_id": identity["episode_id"],
        "reset_count": pair["reset_count"],
        "episode_step": identity["episode_step"],
    }


def _summary_source_entry(summary: Mapping[str, Any], name: str) -> tuple[str, str]:
    source = summary.get("source")
    entry = source.get(name) if isinstance(source, Mapping) else None
    if type(entry) is not dict or set(entry) != {"path", "sha256"}:
        raise RawSupervisionAuditError(f"render summary source.{name} changed")
    if not _is_sha256(entry.get("sha256")):
        raise RawSupervisionAuditError(f"render summary source.{name} hash changed")
    return str(entry["path"]), str(entry["sha256"])


def _source_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _install_reviewed_source_semantics(authorization_sha256: str) -> None:
    _require_exact_authority(authorization_sha256)
    from scripts import audit_go2_n32_camera_frustum_observability as source_v4

    if source_v4._SEMANTICS_LOADED:
        return
    from lewm.benchmarks import go2_n32_camera_frustum_observability as core
    from lewm.datasets import go2_paired_navigation as paired_navigation
    from lewm_worlds import manifest as manifest_semantics
    from lewm_worlds import planning_grid as planning_semantics

    source_v4._install_semantic_modules(
        core,
        paired_navigation,
        manifest_semantics,
        planning_semantics,
    )


def _validate_sample_render_contract(
    render_plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    scene_manifest: Any,
    source_record: Mapping[str, Any],
    *,
    authorization_sha256: str,
) -> tuple[Any, ...]:
    _require_exact_authority(authorization_sha256)
    _install_reviewed_source_semantics(authorization_sha256)
    from scripts import audit_go2_n32_camera_frustum_observability as source_v4

    camera = render_plan.get("camera")
    expected_camera_fields = {
        "native_resolution",
        "training_resolution",
        "fov_axis",
        "fov_deg",
        "near_m",
        "far_m",
        "encoding",
        "mount_body",
    }
    if (
        not isinstance(camera, Mapping)
        or set(camera) != expected_camera_fields
        or camera.get("fov_axis") != "horizontal"
        or not math.isclose(
            float(camera.get("fov_deg", math.nan)),
            78.323,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(camera.get("near_m", math.nan)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or _source_path(render_plan.get("frames_jsonl"))
        != Path(str(source_record["frames"]["path"]))
    ):
        raise RawSupervisionAuditError("sample render-plan camera/source changed")
    projection = summary.get("camera_projection")
    expected_projection_fields = {
        "model",
        "renderer_fov_axis",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "near_m",
        "far_m",
        "runtime_rectification_required",
    }
    expected_vertical = math.degrees(
        2.0
        * math.atan(
            math.tan(math.radians(float(camera["fov_deg"])) * 0.5)
            * (168.0 / 224.0)
        )
    )
    if (
        not isinstance(projection, Mapping)
        or set(projection) != expected_projection_fields
        or projection.get("model") != "pinhole"
        or projection.get("renderer_fov_axis") != "vertical"
        or projection.get("runtime_rectification_required") is not False
        or summary.get("resolution_wh") != [224, 168]
        or not math.isclose(
            float(projection.get("horizontal_fov_deg", math.nan)),
            float(camera["fov_deg"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(projection.get("vertical_fov_deg", math.nan)),
            expected_vertical,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(projection.get("near_m", math.nan)),
            float(camera["near_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(projection.get("far_m", math.nan)),
            float(camera["far_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RawSupervisionAuditError("sample plan/summary projection changed")
    source_v4._validate_raw_scene_object_records(scene_manifest.to_dict())
    rendered_boxes = tuple(source_v4._rendered_boxes(scene_manifest))
    object_records = source_v4.labels_v3._render_object_records(scene_manifest)
    object_ids = sorted(str(item["object_id"]) for item in object_records)
    parity = summary.get("object_parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("schema") != "lewm_render_object_parity_v1"
        or parity.get("rendered_groups")
        != ["wall", "obstacle", "landmark", "distractor"]
        or parity.get("rendered_object_count") != len(object_records)
        or parity.get("rendered_object_ids") != object_ids
        or parity.get("rendered_object_ids_sha256")
        != source_v4.canonical_json_sha256(object_ids)
        or parity.get("rendered_object_records_sha256")
        != source_v4.canonical_json_sha256(object_records)
        or parity.get("collision_distractors_rendered") is not True
        or parity.get("full_box_roll_pitch_yaw_rendered") is not True
    ):
        raise RawSupervisionAuditError("sample full-RPY render parity changed")
    return rendered_boxes


def _find_source_frame(
    payload: bytes,
    record: Mapping[str, Any],
    *,
    plan_camera_mount: Mapping[str, Any],
    authorization_sha256: str,
) -> Mapping[str, Any]:
    _require_exact_authority(authorization_sha256)
    _install_reviewed_source_semantics(authorization_sha256)
    from scripts import audit_go2_n32_camera_frustum_observability as source_v4

    matches: list[Mapping[str, Any]] = []
    if not payload or not payload.endswith(b"\n"):
        raise RawSupervisionAuditError("source frames JSONL is not newline terminated")
    for line_number, line in enumerate(payload.splitlines(), start=1):
        frame = _decode_json(line, name=f"source frames:{line_number}")
        if (
            frame.get("frame_index") == record["frame_index"]
            and frame.get("env_index") == record["env_index"]
            and frame.get("timestamp_ns") == record["timestamp_ns"]
        ):
            matches.append(frame)
    if len(matches) != 1:
        raise RawSupervisionAuditError("sample source frame did not match exactly once")
    return source_v4._extract_source_frame(
        matches[0], record, plan_camera_mount_body=plan_camera_mount
    )


def _recompute_one_exact_sample(
    *,
    endpoint_digest: str,
    endpoint: Mapping[str, Any],
    pair_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
    authorization_sha256: str,
) -> tuple[np.ndarray, ...]:
    _require_exact_authority(authorization_sha256)
    _install_reviewed_source_semantics(authorization_sha256)
    from lewm.benchmarks import go2_dynamic_cell_square_projection as dynamic_projection
    from lewm_worlds import manifest as manifest_semantics
    from scripts import audit_go2_n32_camera_frustum_observability as source_v4
    from scripts import build_go2_observable_camera_ray_fit_v4 as raycast_v4

    frames_info = source_record["frames"]
    manifest_info = source_record["scene_manifest"]
    plan_info = source_record["render_plan"]
    summary_info = source_record["render_summary"]
    frames_payload = _read_absolute_bound_payload(
        Path(str(frames_info["path"])),
        str(frames_info["sha256"]),
        repository_root=ROOT,
        name="sample frames file",
    )
    manifest_payload = _read_exact_source_json(
        str(manifest_info["path"]),
        str(manifest_info["file_sha256"]),
        name="sample scene manifest",
        authorization_sha256=authorization_sha256,
    )
    render_plan = _read_exact_source_json(
        str(plan_info["path"]),
        str(plan_info["sha256"]),
        name="sample render plan",
        authorization_sha256=authorization_sha256,
    )
    summary = _read_exact_source_json(
        str(summary_info["path"]),
        str(summary_info["sha256"]),
        name="sample render summary",
        authorization_sha256=authorization_sha256,
    )
    scene_id = str(source_record["scene_id"])
    if (
        render_plan.get("schema") != "lewm_render_replay_plan_v0"
        or render_plan.get("scene_id") != scene_id
        or summary.get("schema") != "lewm_rendered_vision_v04"
        or summary.get("scene_id") != scene_id
        or summary.get("family") != source_record["family"]
        or summary.get("render_status") != "complete"
    ):
        raise RawSupervisionAuditError("sample plan/summary scene identity changed")
    source_section = summary.get("source")
    if not isinstance(source_section, Mapping) or set(source_section) != {
        "plan", "frames_jsonl", "scene_manifest", "renderer_source"
    }:
        raise RawSupervisionAuditError("sample render-summary source inventory changed")
    for summary_name, inventory_name in (
        ("frames_jsonl", "frames"),
        ("scene_manifest", "scene_manifest"),
        ("plan", "render_plan"),
    ):
        path, digest = _summary_source_entry(summary, summary_name)
        inventory_entry = source_record[inventory_name]
        expected_digest = inventory_entry.get("sha256", inventory_entry.get("file_sha256"))
        summary_path = Path(path)
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        if (
            summary_path != Path(str(inventory_entry["path"]))
            or digest != expected_digest
        ):
            raise RawSupervisionAuditError("render summary source inventory changed")
    camera = render_plan.get("camera")
    if not isinstance(camera, Mapping):
        raise RawSupervisionAuditError("render plan camera is absent")
    plan_mount = source_v4._camera_mount_record(
        camera.get("mount_body"), label="sample render plan camera.mount_body"
    )
    source_v4._validate_summary_records(
        summary,
        [pair_record],
        summary_path=Path(str(summary_info["path"])),
    )
    frame = _find_source_frame(
        frames_payload,
        pair_record,
        plan_camera_mount=plan_mount,
        authorization_sha256=authorization_sha256,
    )
    sidecar_quaternion, stored_yaw = raycast_v4._validated_sidecar_source_attitude(
        frame, endpoint
    )
    scene_manifest = manifest_semantics.parse_scene_manifest_dict(manifest_payload)
    if (
        scene_manifest.scene_id != scene_id
        or scene_manifest.family != source_record["family"]
        or manifest_semantics.manifest_sha256(scene_manifest)
        != manifest_info["content_sha256"]
    ):
        raise RawSupervisionAuditError("sample scene manifest semantic hash changed")
    raw_rendered_boxes = _validate_sample_render_contract(
        render_plan,
        summary,
        scene_manifest,
        source_record,
        authorization_sha256=authorization_sha256,
    )
    position = frame["base_pose_world"]["position"]
    base_position = tuple(float(position[axis]) for axis in ("x", "y", "z"))
    composed = dynamic_projection.compose_yaw_aligned_camera(
        sidecar_quaternion, stored_yaw
    )
    basis = raycast_v4._normalized_camera_basis_fru(composed)
    rendered_boxes = tuple(
        raycast_v4._box_in_yaw_body(
            box,
            base_position_world=base_position,
            stored_yaw_rad=stored_yaw,
        )
        for box in raw_rendered_boxes
    )
    frame_input = raycast_v4.FrameBuildInputV4(
        frame_key={
            "dataset_role": endpoint["identity"]["dataset_role"],
            "family": source_record["family"],
            "scene_id": scene_id,
            "endpoint_identity_sha256": endpoint_digest,
        },
        camera_origin_body_m=tuple(composed.origin_xyz),
        camera_basis_body_fru=basis,
        ground_plane_z_body_m=-base_position[2],
        rendered_boxes_body=rendered_boxes,
        image_path_metadata_only=str(endpoint["image_path_metadata_only"]),
        image_sha256=str(endpoint["identity"]["image_sha256"]),
        sidecar_row_identity_sha256=str(pair_record["sidecar_row_identity_sha256"]),
    )
    evidence = raycast_v4.build_frame_evidence_v4(frame_input)
    raster = raycast_v4.rasterize_observable_camera_ray_evidence_v4(evidence)
    return _stored_arrays_from_evidence(evidence, raster)


def _recompute_exact_sample_task(
    task: tuple[
        str,
        str,
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
    ],
) -> tuple[str, tuple[np.ndarray, ...]]:
    _set_worker_environment()
    authorization_sha256 = task[0]
    _require_exact_authority(authorization_sha256)
    _authorization, digest, endpoint, pair_record, source_record = task
    return digest, _recompute_one_exact_sample(
        endpoint_digest=digest,
        endpoint=endpoint,
        pair_record=pair_record,
        source_record=source_record,
        authorization_sha256=authorization_sha256,
    )


def _exact_sample_recomputer(
    sample: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]],
    inputs: AuditInputs,
    workers: int,
    *,
    authorization_sha256: str,
) -> Mapping[str, tuple[np.ndarray, ...]]:
    worker_count = _strict_workers(workers)
    _require_exact_authority(authorization_sha256)
    source_by_scene = {str(item["scene_id"]): item for item in inputs.inventory.records}
    tasks: list[
        tuple[
            str,
            str,
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ]
    ] = []
    for sample_record in sample:
        digest = str(sample_record["endpoint_identity_sha256"])
        endpoint = endpoints[digest]
        pair_record = _source_record_for_endpoint(digest, endpoint, inputs.plan.pairs)
        source = source_by_scene.get(str(endpoint["identity"]["scene_id"]))
        if source is None:
            raise RawSupervisionAuditError("sample scene is absent from source inventory")
        if source["role"] != endpoint["identity"]["dataset_role"] or source["family"] != pair_record["family"]:
            raise RawSupervisionAuditError("sample source crossed role/family")
        tasks.append(
            (authorization_sha256, digest, endpoint, pair_record, source)
        )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_exact_worker,
        initargs=(authorization_sha256,),
    ) as executor:
        replayed = list(executor.map(_recompute_exact_sample_task, tasks))
    result = dict(replayed)
    if len(result) != len(tasks):
        raise RawSupervisionAuditError("exact sample replay repeated an endpoint")
    return result


@dataclass(frozen=True)
class _PreparedAuditV6:
    result: Mapping[str, Any]
    authorization: AcceptedAuthorizationV6
    manifest: Mapping[str, Any]
    manifest_file_sha256: str
    plan: DevelopmentRawSupervisionPlan
    inventory: DevelopmentSourceInventory
    hashed_sources: tuple[Mapping[str, Any], ...]
    parent_contracts: tuple[Mapping[str, Any], ...]


def _prepare_authorized_audit_v6(
    *,
    authorization_sha256: str,
    workers: int,
) -> _PreparedAuditV6:
    """Complete the audit and prepare its report before the final pass."""

    worker_count = _strict_workers(workers)
    authorization = _require_exact_authority(authorization_sha256)
    manifest_payload, manifest_file_sha256 = _read_fixed_manifest_payload()
    if not manifest_payload.endswith(b"\n"):
        raise RawSupervisionAuditError(
            "dataset manifest lacks terminal newline"
        )
    manifest = _parse_manifest(
        CANONICAL_DATASET,
        expected_manifest_file_sha256=manifest_file_sha256,
    )

    from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan_v5 import (
        load_frozen_development_metadata,
        load_frozen_development_source_inventory,
    )

    plan = load_frozen_development_metadata(ROOT)
    inventory = load_frozen_development_source_inventory(ROOT, plan)
    before = _hash_complete_source_inventory(
        inventory,
        authorization_sha256=authorization_sha256,
        workers=worker_count,
    )
    parent_contracts_before = _parent_contract_receipts(authorization_sha256)
    inputs = AuditInputs(plan=plan, inventory=inventory)
    _validate_exact_manifest_bindings(
        manifest,
        inputs=inputs,
        hashed_sources=before,
        parent_contracts=parent_contracts_before,
        authorization=authorization,
    )
    result = _audit_fixed_dataset(
        authorization_sha256=authorization_sha256,
        expected_manifest_file_sha256=manifest_file_sha256,
        inputs=inputs,
        workers=worker_count,
    )
    core = dict(result)
    core.pop("content_sha256")
    core["source_file_count"] = len(before) + len(parent_contracts_before)
    complete_receipts = _builder_source_receipts(before, parent_contracts_before)
    core["source_inventory_before_after_sha256"] = canonical_json_sha256(
        complete_receipts
    )
    core["source_payload_opens"] = {
        "complete_inventory_hash_passes": 2,
        "permitted_source_files_per_pass": len(complete_receipts),
        "sample_endpoint_count": EXPECTED_SAMPLE_COUNT,
        "rgb_byte_opens": 0,
        "rgb_decodes": 0,
        "label_shard_payload_opens": 0,
        "g2_payload_opens": 0,
        "checkpoint_model_runtime_heldout_hardware_production_opens": 0,
    }
    core["authorization_v6"] = {
        "file_sha256": authorization.authorization_file_sha256,
        "content_sha256": authorization.authorization_content_sha256,
        "source_map_sha256": authorization.source_map_sha256,
        "phase_one_zero_target_opens": True,
        "phase_two_fixed_target_count": len(SOURCE_ROLE_PATHS),
        "machine_pass_reviews_parsed": 2,
    }
    core["closed_publication_transaction_v6"] = {
        "source_and_candidate_watches_continuous_through_rename": True,
        "retained_source_dataset_and_candidate_descriptors": True,
        "publication_and_source_ancestor_chains_watched": True,
        "single_renameat2_RENAME_NOREPLACE": True,
        "exact_owned_rename_event_sequence": True,
        "post_rename_inventory_and_quiescence": True,
    }
    complete = {**core, "content_sha256": canonical_json_sha256(core)}
    return _PreparedAuditV6(
        result=complete,
        authorization=authorization,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        plan=plan,
        inventory=inventory,
        hashed_sources=tuple(before),
        parent_contracts=tuple(parent_contracts_before),
    )


def _final_revalidate_authorized_audit_v6(
    prepared: _PreparedAuditV6,
    *,
    authorization_sha256: str,
    workers: int,
) -> None:
    """Run the one complete final pass while the transaction stays live."""

    if type(prepared) is not _PreparedAuditV6:
        raise TypeError("final audit pass requires a prepared V6 audit")
    _require_exact_authority(authorization_sha256)
    after = _hash_complete_source_inventory(
        prepared.inventory,
        authorization_sha256=authorization_sha256,
        workers=_strict_workers(workers),
    )
    parent_contracts_after = _parent_contract_receipts(authorization_sha256)
    if (
        after != list(prepared.hashed_sources)
        or parent_contracts_after != list(prepared.parent_contracts)
    ):
        raise RawSupervisionAuditError(
            "development source inventory changed during audit"
        )
    from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan_v5 import (
        load_frozen_development_metadata,
        load_frozen_development_source_inventory,
    )

    final_plan = load_frozen_development_metadata(ROOT)
    final_inventory = load_frozen_development_source_inventory(ROOT, final_plan)
    if (
        final_plan.value != prepared.plan.value
        or final_plan.pairs != prepared.plan.pairs
        or final_plan.endpoints != prepared.plan.endpoints
        or final_inventory.records != prepared.inventory.records
        or final_inventory.hashes != prepared.inventory.hashes
    ):
        raise RawSupervisionAuditError("metadata V5 changed during exact audit")
    payload, digest = _read_fixed_manifest_payload()
    if digest != prepared.manifest_file_sha256 or not payload.endswith(b"\n"):
        raise RawSupervisionAuditError("dataset manifest changed during final pass")
    _validate_root_file_inventory(CANONICAL_DATASET, prepared.manifest.get("files"))


def _publish_terminal_audit_failure(
    *, authorization_sha256: str, error: BaseException
) -> None:
    failure_core = {
        "schema": AUDIT_FAILURE_SCHEMA,
        "status": "terminal_failed_no_dataset_authority",
        "authorization_file_sha256": authorization_sha256,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "canonical_dataset_present": CANONICAL_DATASET.exists(),
        "audit_report_present": False,
        "dataset_use_authorized": False,
        "training_authorized": False,
        "g2_authorized": False,
        "production_authorized": False,
        "retry_authorized": False,
    }
    failure = {
        **failure_core,
        "content_sha256": canonical_json_sha256(failure_core),
    }
    with _FailureReceiptPublisher() as publisher:
        publisher.require_absent()
        publisher.publish(failure)


def execute_exact_audit_v6(
    *,
    authorization_sha256: str,
    workers: int,
) -> dict[str, Any]:
    """Authorize, audit, and publish the one fixed immutable report."""

    worker_count = _strict_workers(workers)
    _require_exact_authority(authorization_sha256)
    retained = _open_retained_directory_chain(CANONICAL_AUDIT_REPORT.parent)
    candidate_name = ""
    candidate_descriptor: int | None = None
    transaction: _ClosedAuditPublicationTransaction | None = None
    try:
        for name in (CANONICAL_AUDIT_REPORT.name, CANONICAL_AUDIT_FAILURE.name):
            if _lstat_optional_at(retained.directory_fd, name) is not None:
                raise FileExistsError(f"immutable audit leaf already exists: {name}")
        prepared = _prepare_authorized_audit_v6(
            authorization_sha256=authorization_sha256,
            workers=worker_count,
        )
        (
            candidate_name,
            candidate_descriptor,
            candidate_fingerprint,
            candidate_sha256,
        ) = _stage_owned_audit_candidate(retained, prepared.result)
        transaction = _ClosedAuditPublicationTransaction(
            context=_AuditPublicationContextV6(
                authorization=prepared.authorization,
                manifest=prepared.manifest,
                manifest_file_sha256=prepared.manifest_file_sha256,
                hashed_sources=prepared.hashed_sources,
                parent_contracts=prepared.parent_contracts,
            ),
            retained=retained,
            candidate_name=candidate_name,
            candidate_descriptor=candidate_descriptor,
            candidate_fingerprint=candidate_fingerprint,
            candidate_sha256=candidate_sha256,
        )
        _final_revalidate_authorized_audit_v6(
            prepared,
            authorization_sha256=authorization_sha256,
            workers=worker_count,
        )
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.validate(allow_final_metadata_change=True)
        transaction.validate_after_rename()
        os.fsync(retained.directory_fd)
        transaction.require_final_quiet()
        result = dict(prepared.result)
    except BaseException as error:
        renamed = transaction.renamed if transaction is not None else False
        if transaction is not None:
            transaction.close()
        if candidate_descriptor is not None:
            try:
                _cleanup_owned_audit_candidate(
                    retained,
                    candidate_name=candidate_name,
                    candidate_descriptor=candidate_descriptor,
                    renamed=renamed,
                )
            except BaseException:
                pass
            os.close(candidate_descriptor)
        retained.close()
        try:
            _publish_terminal_audit_failure(
                authorization_sha256=authorization_sha256, error=error
            )
        except BaseException:
            pass
        raise
    if transaction is not None:
        transaction.close()
    if candidate_descriptor is not None:
        os.close(candidate_descriptor)
    retained.close()
    return result


__all__ = [
    "ARRAY_LAYOUT",
    "AUDIT_SCHEMA",
    "CANONICAL_DATASET",
    "CANONICAL_AUDIT_FAILURE",
    "CANONICAL_AUDIT_REPORT",
    "DATASET_SCHEMA",
    "ENDPOINT_INDEX_SCHEMA",
    "MAX_WORKERS",
    "RawSupervisionAuditError",
    "SHARD_SCHEMA",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "execute_exact_audit_v6",
]

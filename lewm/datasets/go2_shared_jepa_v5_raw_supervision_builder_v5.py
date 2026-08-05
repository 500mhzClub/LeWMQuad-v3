"""Standalone V5 raw-supervision builder for Shared-JEPA V5 development.

The reviewed V1 construction engine is owned directly by this module. Exact
mode uses a closed two-phase V5 authority, fixed worker functions, and native
exact-source readers. No legacy builder module or exact entry is imported.
RGB and legacy label payloads are never opened by this module.
"""
from __future__ import annotations

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
import secrets
import shutil
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import RASTER_SCHEMA
from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan_v5 import (
    DEVELOPMENT_ROLES,
    SOURCE_INVENTORY_SHA256,
    DevelopmentRawSupervisionPlan,
    DevelopmentSourceInventory,
)
from scripts.build_go2_observable_camera_ray_fit_v4 import (
    EVIDENCE_SCHEMA,
    FrameBuildInputV4,
    _box_in_yaw_body,
    _normalized_camera_basis_fru,
    _validated_sidecar_source_attitude,
    build_frame_evidence_v4,
    rasterize_observable_camera_ray_evidence_v4,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_OUTPUT = (
    ROOT
    / ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
FAILURE_RECEIPT = CANONICAL_OUTPUT.with_name(
    CANONICAL_OUTPUT.name + ".failed.json"
)
AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_2026-07-13.json"
)

DATASET_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_dataset_v1"
SHARD_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_shard_v1"
ENDPOINT_INDEX_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_endpoint_index_v1"
FAILURE_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_build_failure_v1"
AUTHORIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v5"
REVIEW_BINDING_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v5"
)
BUILDER_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_builder_v5_independent_review_v1"
)
AUDITOR_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v5_independent_review_v1"
)
BUILDER_IMPLEMENTATION_AUTHOR = "/root/raw_builder_arch"
AUDITOR_IMPLEMENTATION_AUTHOR = "/root/raw_auditor_author"

MAX_WORKERS = 6
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

PREREGISTRATION_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_preregistration_2026-07-13.md"
)
SOURCE_INVENTORY_AMENDMENT_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_source_inventory_amendment_2026-07-13.md"
)
METADATA_V5_HANDOFF_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_author_handoff_2026-07-13.md"
)
METADATA_V5_QA_PATH = (
    ROOT
    / "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5_independent_qa.py"
)
METADATA_V5_REVIEW_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_independent_review_2026-07-13.md"
)
V5_SUCCESSOR_AMENDMENT_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v5_"
    "authorization_successor_amendment_2026-07-13.md"
)

FROZEN_PARENT_HASHES = {
    str(PREREGISTRATION_PATH.relative_to(ROOT)): (
        "07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb"
    ),
    str(SOURCE_INVENTORY_AMENDMENT_PATH.relative_to(ROOT)): (
        "39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v5.py": (
        "67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921"
    ),
    str(METADATA_V5_HANDOFF_PATH.relative_to(ROOT)): (
        "b362d26372f01e670a477dda5e7abb5e55370cc1d8d89052545afa229e7bba66"
    ),
    str(METADATA_V5_QA_PATH.relative_to(ROOT)): (
        "8a50bcf5275d243f06b92264e017f355fd54faaca8f8e73aab1e3cc45dc51298"
    ),
    str(METADATA_V5_REVIEW_PATH.relative_to(ROOT)): (
        "7d7344e423492a3cf36d1cd50ca09e6c7eb6eba17c25861c840531465aaf7706"
    ),
    str(V5_SUCCESSOR_AMENDMENT_PATH.relative_to(ROOT)): (
        "fe6a29a27eb0284ce84fcba409b530c6351befad18ee9d655f5f2e9b337d9e91"
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

REVIEWED_V4_SOURCES = {
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

AUTHORIZED_ROLE_PATHS: tuple[tuple[str, str], ...] = (
    (
        "builder_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v5.py",
    ),
    (
        "builder_cli",
        "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v5.py",
    ),
    (
        "builder_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5.py",
    ),
    (
        "builder_handoff",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_"
        "author_handoff_2026-07-13.md",
    ),
    (
        "builder_review",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_"
        "independent_review_2026-07-13.json",
    ),
    (
        "auditor_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v5.py",
    ),
    (
        "auditor_cli",
        "scripts/audit_go2_shared_jepa_v5_raw_supervision_v5.py",
    ),
    (
        "auditor_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v5.py",
    ),
    (
        "auditor_review",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v5_"
        "independent_review_2026-07-13.json",
    ),
)
ROLE_PATH_BY_NAME = dict(AUTHORIZED_ROLE_PATHS)
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

ARRAY_LAYOUT = (
    ("camera_origin_body_m.f4", "<f4", (3,)),
    ("camera_basis_body_fru.f4", "<f4", (3, 3)),
    ("ground_plane_z_body_m.f4", "<f4", ()),
    ("ground_support_in_frustum.u1", "u1", (128, 128, 5)),
    ("ground_support_clear_to_target.u1", "u1", (128, 128, 5)),
    ("pixel_hit_mask.u1", "u1", (84, 112)),
    ("pixel_first_hit_distance_m.f4", "<f4", (84, 112)),
    ("raster_labels.u1", "u1", (64, 64)),
)


class RawSupervisionBuildError(RuntimeError):
    """Raised when an exact or synthetic build violates the frozen contract."""


@dataclass(frozen=True)
class SourceBindingV5:
    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ReviewBindingV5:
    kind: str
    review_schema: str
    verdict: str
    reviewer: str
    implementation_author: str
    path: str
    file_sha256: str
    content_sha256: str
    candidate: tuple[SourceBindingV5, ...]


@dataclass(frozen=True)
class PhaseOneAuthorizationV5:
    authorization_file_sha256: str
    authorization_content_sha256: str
    source_map_sha256: str
    canonical_payload: bytes
    sources: tuple[SourceBindingV5, ...]
    builder_review: ReviewBindingV5
    auditor_review: ReviewBindingV5


@dataclass(frozen=True)
class AcceptedAuthorizationV5:
    authorization_file_sha256: str
    authorization_content_sha256: str
    source_map_sha256: str


@dataclass(frozen=True)
class ExactPrepublicationContextV5:
    plan: DevelopmentRawSupervisionPlan
    inventory: DevelopmentSourceInventory
    source_records: tuple[Mapping[str, Any], ...]
    authorization_sha256: str
    workers: int


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(canonical_json_bytes(core))
    return {**value, "content_sha256": canonical_json_sha256(value)}


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_workers(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_WORKERS:
        raise ValueError(f"workers must be an exact integer in [1,{MAX_WORKERS}]")
    return value


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RawSupervisionBuildError(f"nonregular output file: {path}")
    return {
        "path": str(path.relative_to(root)),
        "byte_count": int(metadata.st_size),
        "file_sha256": _sha256_file(path),
    }


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, canonical_json_bytes(payload) + b"\n")


def _write_bytes_exclusive_at(parent_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
        dir_fd=parent_fd,
    )
    metadata = os.fstat(descriptor)
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (int(named.st_dev), int(named.st_ino)) == identity:
                os.unlink(name, dir_fd=parent_fd)
        finally:
            os.close(descriptor)
        raise
    os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


Fingerprint = tuple[int, int, int, int, int, int, int]
OpenChainRow = tuple[int, str, int, Fingerprint]


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


def _validate_open_chain(
    *,
    filesystem_root: Path,
    anchor_fd: int,
    anchor_fingerprint: Fingerprint,
    directory_rows: Sequence[OpenChainRow],
    leaf_parent_fd: int,
    leaf_name: str,
    leaf_fd: int,
    leaf_fingerprint: Fingerprint,
) -> None:
    anchor_named = filesystem_root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(anchor_named.st_mode)
        or _fingerprint(anchor_named) != anchor_fingerprint
        or _fingerprint(os.fstat(anchor_fd)) != anchor_fingerprint
    ):
        raise RawSupervisionBuildError("filesystem root changed during bound read")
    for parent_fd, component, child_fd, original in directory_rows:
        named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(child_fd)
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _fingerprint(named) != original
            or _fingerprint(opened) != original
        ):
            raise RawSupervisionBuildError(
                f"bound directory component changed: {component!r}"
            )
    named_leaf = os.stat(leaf_name, dir_fd=leaf_parent_fd, follow_symlinks=False)
    opened_leaf = os.fstat(leaf_fd)
    if (
        stat.S_ISLNK(named_leaf.st_mode)
        or not stat.S_ISREG(named_leaf.st_mode)
        or not stat.S_ISREG(opened_leaf.st_mode)
        or int(named_leaf.st_nlink) != 1
        or int(opened_leaf.st_nlink) != 1
        or _fingerprint(named_leaf) != leaf_fingerprint
        or _fingerprint(opened_leaf) != leaf_fingerprint
    ):
        raise RawSupervisionBuildError("bound regular file changed during read")


def _read_bound_regular_file(
    *,
    repository_root: Path,
    path: Path,
    expected_sha256: str,
) -> bytes:
    """Read one allowlisted file through a no-follow, fingerprinted FD chain."""

    root = Path(repository_root).absolute()
    lexical = Path(path).absolute()
    if (
        root.resolve(strict=True) != root
        or root.is_symlink()
        or lexical != path
        or not _is_sha256(expected_sha256)
    ):
        raise PermissionError("bound read root/path/hash is not canonical")
    try:
        lexical.relative_to(root)
    except ValueError as error:
        raise PermissionError("bound read path escapes the repository") from error
    parts = lexical.parts
    if len(parts) < 2 or parts[0] != lexical.anchor:
        raise PermissionError("bound read path has no filesystem anchor")

    directory_fds: list[int] = []
    directory_rows: list[OpenChainRow] = []
    leaf_fd: int | None = None
    filesystem_root = Path(lexical.anchor)
    try:
        anchor_before = filesystem_root.stat(follow_symlinks=False)
        anchor_fingerprint = _fingerprint(anchor_before)
        anchor_fd = os.open(
            filesystem_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        directory_fds.append(anchor_fd)
        parent_fd = anchor_fd
        for component in parts[1:-1]:
            before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise PermissionError("bound read encountered a non-directory ancestor")
            original = _fingerprint(before)
            child_fd = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            directory_fds.append(child_fd)
            if _fingerprint(os.fstat(child_fd)) != original:
                raise RawSupervisionBuildError(
                    f"bound directory changed during open: {component!r}"
                )
            directory_rows.append((parent_fd, component, child_fd, original))
            parent_fd = child_fd
        leaf_name = parts[-1]
        leaf_before = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
        leaf_fingerprint = _fingerprint(leaf_before)
        if (
            stat.S_ISLNK(leaf_before.st_mode)
            or not stat.S_ISREG(leaf_before.st_mode)
            or int(leaf_before.st_nlink) != 1
        ):
            raise PermissionError("bound read leaf is not a single-link regular file")
        leaf_fd = os.open(
            leaf_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        if _fingerprint(os.fstat(leaf_fd)) != leaf_fingerprint:
            raise RawSupervisionBuildError("bound leaf changed during open")
        validation = {
            "filesystem_root": filesystem_root,
            "anchor_fd": anchor_fd,
            "anchor_fingerprint": anchor_fingerprint,
            "directory_rows": directory_rows,
            "leaf_parent_fd": parent_fd,
            "leaf_name": leaf_name,
            "leaf_fd": leaf_fd,
            "leaf_fingerprint": leaf_fingerprint,
        }
        _validate_open_chain(**validation)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(leaf_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        _validate_open_chain(**validation)
        payload = b"".join(chunks)
    finally:
        if leaf_fd is not None:
            os.close(leaf_fd)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)
    if _sha256_bytes(payload) != expected_sha256:
        raise RawSupervisionBuildError(f"bound file SHA-256 changed: {lexical}")
    return payload


def _libc_renameat2(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
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
    rename_noreplace = 1
    result = renameat2(
        int(source_fd),
        os.fsencode(source_name),
        int(destination_fd),
        os.fsencode(destination_name),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number, os.strerror(error_number), destination_name
            )
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Test helper for an atomic no-replace rename in one canonical parent."""

    if source.parent != destination.parent:
        raise ValueError("no-replace directory rename requires one parent")
    parent_fd = os.open(
        source.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _libc_renameat2(parent_fd, source.name, parent_fd, destination.name)
    finally:
        os.close(parent_fd)


@dataclass
class _RetainedPublicationParent:
    path: Path
    anchor_path: Path
    anchor_fd: int
    anchor_fingerprint: Fingerprint
    stable_rows: tuple[OpenChainRow, ...]
    parent_entry_parent_fd: int
    parent_entry_name: str
    parent_fd: int
    parent_identity: tuple[int, int]
    expected_parent_fingerprint: Fingerprint
    owned_fds: tuple[int, ...]

    def validate(self) -> None:
        anchor_named = self.anchor_path.stat(follow_symlinks=False)
        if (
            _fingerprint(anchor_named) != self.anchor_fingerprint
            or _fingerprint(os.fstat(self.anchor_fd)) != self.anchor_fingerprint
        ):
            raise RawSupervisionBuildError("publication filesystem root changed")
        for parent_fd, component, child_fd, original in self.stable_rows:
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or _fingerprint(named) != original
                or _fingerprint(os.fstat(child_fd)) != original
            ):
                raise RawSupervisionBuildError(
                    f"publication ancestor changed: {component!r}"
                )
        named_parent = os.stat(
            self.parent_entry_name,
            dir_fd=self.parent_entry_parent_fd,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(named_parent.st_mode)
            or not stat.S_ISDIR(named_parent.st_mode)
            or (int(named_parent.st_dev), int(named_parent.st_ino))
            != self.parent_identity
            or _fingerprint(named_parent) != self.expected_parent_fingerprint
            or _fingerprint(os.fstat(self.parent_fd))
            != self.expected_parent_fingerprint
        ):
            raise RawSupervisionBuildError("publication parent changed")

    def refresh_after_owned_mutation(self) -> None:
        named = os.stat(
            self.parent_entry_name,
            dir_fd=self.parent_entry_parent_fd,
            follow_symlinks=False,
        )
        opened = os.fstat(self.parent_fd)
        if (
            (int(named.st_dev), int(named.st_ino)) != self.parent_identity
            or (int(opened.st_dev), int(opened.st_ino)) != self.parent_identity
            or _fingerprint(named) != _fingerprint(opened)
        ):
            raise RawSupervisionBuildError("publication parent alias changed")
        self.expected_parent_fingerprint = _fingerprint(opened)
        self.validate()

    def close(self) -> None:
        for descriptor in reversed(self.owned_fds):
            os.close(descriptor)


def _open_publication_parent(path: Path) -> _RetainedPublicationParent:
    canonical = Path(path).absolute()
    if canonical.resolve(strict=True) != canonical or canonical.is_symlink():
        raise PermissionError("publication parent must be a canonical real directory")
    anchor_path = Path(canonical.anchor)
    anchor_before = anchor_path.stat(follow_symlinks=False)
    anchor_fd = os.open(
        anchor_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    owned_fds = [anchor_fd]
    rows: list[OpenChainRow] = []
    parent_fd = anchor_fd
    try:
        for component in canonical.parts[1:]:
            before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise PermissionError("publication path contains a non-directory")
            original = _fingerprint(before)
            child_fd = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            owned_fds.append(child_fd)
            if _fingerprint(os.fstat(child_fd)) != original:
                raise RawSupervisionBuildError("publication directory changed on open")
            rows.append((parent_fd, component, child_fd, original))
            parent_fd = child_fd
        if not rows:
            raise PermissionError("filesystem root cannot be a publication parent")
        last_parent_fd, last_name, final_fd, final_fingerprint = rows[-1]
        retained = _RetainedPublicationParent(
            path=canonical,
            anchor_path=anchor_path,
            anchor_fd=anchor_fd,
            anchor_fingerprint=_fingerprint(anchor_before),
            stable_rows=tuple(rows[:-1]),
            parent_entry_parent_fd=last_parent_fd,
            parent_entry_name=last_name,
            parent_fd=final_fd,
            parent_identity=(final_fingerprint[0], final_fingerprint[1]),
            expected_parent_fingerprint=final_fingerprint,
            owned_fds=tuple(owned_fds),
        )
        retained.validate()
        return retained
    except BaseException:
        for descriptor in reversed(owned_fds):
            os.close(descriptor)
        raise


def _named_directory_identity(parent_fd: int, name: str) -> tuple[int, int]:
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RawSupervisionBuildError("owned publication leaf is not a directory")
    return (int(metadata.st_dev), int(metadata.st_ino))


def _cleanup_owned_directory(
    retained: _RetainedPublicationParent,
    name: str,
    identity: tuple[int, int] | None,
) -> bool:
    if identity is None:
        return False
    try:
        current = _named_directory_identity(retained.parent_fd, name)
    except FileNotFoundError:
        return False
    if current != identity:
        return False
    shutil.rmtree(name, dir_fd=retained.parent_fd)
    retained.refresh_after_owned_mutation()
    return True


@dataclass(frozen=True)
class PreparedEndpointV5:
    plan_endpoint: Mapping[str, Any]
    family: str
    frame: FrameBuildInputV4

    def __post_init__(self) -> None:
        identity = self.plan_endpoint.get("identity")
        if not isinstance(identity, Mapping) or not self.family:
            raise ValueError("prepared endpoint is malformed")
        if self.plan_endpoint.get("identity_sha256") != canonical_json_sha256(identity):
            raise ValueError("prepared endpoint identity hash changed")
        core = dict(self.plan_endpoint)
        declared = core.pop("content_sha256", None)
        if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
            raise ValueError("prepared plan endpoint content hash changed")


@dataclass(frozen=True)
class PreparedSceneJobV5:
    scene_id: str
    role: str
    family: str
    endpoints: tuple[PreparedEndpointV5, ...]

    def __post_init__(self) -> None:
        if (
            not self.scene_id
            or self.role not in DEVELOPMENT_ROLES
            or not self.family
            or not self.endpoints
        ):
            raise ValueError("prepared scene job is malformed")
        for endpoint in self.endpoints:
            identity = endpoint.plan_endpoint["identity"]
            if (
                identity.get("scene_id") != self.scene_id
                or identity.get("dataset_role") != self.role
                or endpoint.family != self.family
            ):
                raise ValueError("prepared endpoint crossed its scene/role/family")


def _endpoint_arrays(
    evidence: Any,
    raster: Any,
) -> tuple[np.ndarray, ...]:
    return (
        np.ascontiguousarray(evidence.camera_origin_body_m, dtype="<f4"),
        np.ascontiguousarray(evidence.camera_basis_body_fru, dtype="<f4"),
        np.asarray(evidence.ground_plane_z_body_m, dtype="<f4"),
        np.ascontiguousarray(evidence.ground_support_in_frustum, dtype=np.uint8),
        np.ascontiguousarray(
            evidence.ground_support_clear_to_target, dtype=np.uint8
        ),
        np.ascontiguousarray(evidence.pixel_hit_mask, dtype=np.uint8),
        np.ascontiguousarray(evidence.pixel_first_hit_distance_m, dtype="<f4"),
        np.ascontiguousarray(raster.output_labels, dtype=np.uint8),
    )


def _worker_environment() -> None:
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    for name in ACCELERATOR_ENVIRONMENT:
        os.environ[name] = ""


def _initialize_exact_worker(authorization_sha256: str) -> None:
    """Authorize the spawned process before it receives a task payload."""

    _worker_environment()
    _require_exact_authority(authorization_sha256)


def _write_prepared_scene_job(
    job: PreparedSceneJobV5,
    staging_root: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    _worker_environment()
    _require_exact_authority(authorization_sha256)
    staging = Path(staging_root)
    scene_digest = hashlib.sha256(job.scene_id.encode("utf-8")).hexdigest()
    directory_name = scene_digest[:16]
    directory = staging / "shards" / directory_name
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    ordered = tuple(
        sorted(
            job.endpoints,
            key=lambda item: str(item.plan_endpoint["identity_sha256"]),
        )
    )
    evidences = [build_frame_evidence_v4(item.frame) for item in ordered]
    rasters = [
        rasterize_observable_camera_ray_evidence_v4(evidence)
        for evidence in evidences
    ]
    arrays = [
        _endpoint_arrays(evidence, raster)
        for evidence, raster in zip(evidences, rasters)
    ]
    files: list[dict[str, Any]] = []
    for position, (filename, dtype, trailing_shape) in enumerate(ARRAY_LAYOUT):
        values = np.stack([item[position] for item in arrays], axis=0).astype(
            np.dtype(dtype), copy=False
        )
        expected_shape = (len(ordered), *trailing_shape)
        if values.shape != expected_shape:
            raise RawSupervisionBuildError(
                f"{filename} shape {values.shape} != {expected_shape}"
            )
        path = directory / filename
        _write_bytes_exclusive(path, values.tobytes(order="C"))
        files.append(
            {
                **_file_record(path, root=directory),
                "dtype": np.dtype(dtype).str,
                "shape": list(expected_shape),
            }
        )

    index_rows: list[dict[str, Any]] = []
    for row_index, (item, evidence, raster) in enumerate(
        zip(ordered, evidences, rasters)
    ):
        endpoint = item.plan_endpoint
        identity = endpoint["identity"]
        core = {
            "schema": ENDPOINT_INDEX_SCHEMA,
            "dataset_role": job.role,
            "family": job.family,
            "scene_id": job.scene_id,
            "endpoint_identity_sha256": endpoint["identity_sha256"],
            "plan_endpoint_content_sha256": endpoint["content_sha256"],
            "shard_row": row_index,
            "image_path_metadata_only": endpoint["image_path_metadata_only"],
            "image_sha256_commitment_only": identity["image_sha256"],
            "evidence_content_sha256": evidence.content_sha256(),
            "raster_content_sha256": raster.content_sha256(),
        }
        index_rows.append(_with_content_sha256(core))
    index_payload = b"".join(
        canonical_json_bytes(item) + b"\n" for item in index_rows
    )
    index_path = directory / "index.jsonl"
    _write_bytes_exclusive(index_path, index_payload)
    files.append(
        {
            **_file_record(index_path, root=directory),
            "dtype": "canonical_jsonl",
            "shape": [len(index_rows)],
        }
    )
    shard = _with_content_sha256(
        {
            "schema": SHARD_SCHEMA,
            "dataset_role": job.role,
            "family": job.family,
            "scene_id": job.scene_id,
            "scene_id_sha256": scene_digest,
            "endpoint_count": len(index_rows),
            "ordered_endpoint_identity_sha256": canonical_json_sha256(
                [item["endpoint_identity_sha256"] for item in index_rows]
            ),
            "ordered_evidence_sha256": canonical_json_sha256(
                [item["evidence_content_sha256"] for item in index_rows]
            ),
            "ordered_raster_sha256": canonical_json_sha256(
                [item["raster_content_sha256"] for item in index_rows]
            ),
            "files": sorted(files, key=lambda item: item["path"]),
        }
    )
    shard_path = directory / "shard.json"
    _write_json_exclusive(shard_path, shard)
    _fsync_directory(directory)
    return {
        "directory_name": directory_name,
        "directory_path": str(directory),
        "shard": shard,
        "index_rows": index_rows,
    }


def _validate_jobs_and_pairs(
    jobs: Sequence[PreparedSceneJobV5],
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[PreparedSceneJobV5, ...], tuple[Mapping[str, Any], ...]]:
    ordered_jobs = tuple(sorted(jobs, key=lambda item: item.scene_id))
    if not ordered_jobs or len({item.scene_id for item in ordered_jobs}) != len(
        ordered_jobs
    ):
        raise ValueError("scene jobs must be nonempty and unique")
    prefixes = [hashlib.sha256(item.scene_id.encode("utf-8")).hexdigest()[:16] for item in ordered_jobs]
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("scene shard digest prefixes collide")
    endpoints: dict[str, PreparedEndpointV5] = {}
    for job in ordered_jobs:
        for endpoint in job.endpoints:
            digest = str(endpoint.plan_endpoint["identity_sha256"])
            if digest in endpoints:
                raise ValueError("one endpoint was scheduled more than once")
            endpoints[digest] = endpoint
    ordered_pairs = tuple(
        sorted(
            pairs,
            key=lambda item: (
                DEVELOPMENT_ROLES.index(str(item["dataset_role"])),
                int(item["global_row"]),
            ),
        )
    )
    if not ordered_pairs:
        raise ValueError("pair index is empty")
    uses: dict[str, int] = {digest: 0 for digest in endpoints}
    pair_hashes: set[str] = set()
    for pair in ordered_pairs:
        digest = str(pair.get("content_sha256", ""))
        pair_core = dict(pair)
        pair_core.pop("content_sha256", None)
        if (
            not _is_sha256(digest)
            or canonical_json_sha256(pair_core) != digest
            or digest in pair_hashes
        ):
            raise ValueError("pair content hashes are malformed or repeated")
        pair_hashes.add(digest)
        role = pair.get("dataset_role")
        scene_id = pair.get("scene_id")
        family = pair.get("family")
        for side in ("current", "next"):
            endpoint_digest = str(pair.get(f"{side}_endpoint_sha256", ""))
            endpoint = endpoints.get(endpoint_digest)
            if endpoint is None:
                raise ValueError("pair references an absent endpoint")
            identity = endpoint.plan_endpoint["identity"]
            if (
                identity.get("dataset_role") != role
                or identity.get("scene_id") != scene_id
                or endpoint.family != family
            ):
                raise ValueError("pair/endpoint role, scene, or family changed")
            uses[endpoint_digest] += 1
    if any(count == 0 for count in uses.values()):
        raise ValueError("endpoint index contains an orphan")
    return ordered_jobs, ordered_pairs


def _precommitted_audit_sample(
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in endpoint_rows:
        grouped.setdefault(
            (str(row["dataset_role"]), str(row["family"])), []
        ).append(row)
    records = []
    for (role, family), candidates in sorted(grouped.items()):
        chosen = min(
            candidates,
            key=lambda item: hashlib.sha256(
                role.encode("utf-8")
                + b"\0"
                + family.encode("utf-8")
                + b"\0"
                + str(item["endpoint_identity_sha256"]).encode("ascii")
            ).hexdigest(),
        )
        records.append(
            {
                "dataset_role": role,
                "family": family,
                "endpoint_identity_sha256": chosen["endpoint_identity_sha256"],
                "selection_sha256": hashlib.sha256(
                    role.encode("utf-8")
                    + b"\0"
                    + family.encode("utf-8")
                    + b"\0"
                    + str(chosen["endpoint_identity_sha256"]).encode("ascii")
                ).hexdigest(),
            }
        )
    return {
        "scheme": "minimum_sha256_role_nul_family_nul_endpoint_identity_v1",
        "one_endpoint_per_observed_role_family": True,
        "expected_exact_record_count": 24,
        "records": records,
        "records_sha256": canonical_json_sha256(records),
    }


def _validate_staging_inventory(
    staging: Path,
    expected_files: Sequence[Mapping[str, Any]],
    *,
    manifest_present: bool,
) -> None:
    observed_files: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for path in sorted(staging.rglob("*"), key=str):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RawSupervisionBuildError("staging tree contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            observed_files[str(path.relative_to(staging))] = path
        elif stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(str(path.relative_to(staging)))
        else:
            raise RawSupervisionBuildError("staging tree contains a special file")
    expected = {str(item["path"]): item for item in expected_files}
    names = set(expected) | ({"manifest.json"} if manifest_present else set())
    if set(observed_files) != names:
        raise RawSupervisionBuildError("staging file inventory changed")
    expected_directories = {"shards"}
    for name in expected:
        parent = Path(name).parent
        while str(parent) not in {"", "."}:
            expected_directories.add(str(parent))
            parent = parent.parent
    if observed_directories != expected_directories:
        raise RawSupervisionBuildError("staging directory inventory changed")
    for name, record in expected.items():
        path = observed_files[name]
        if (
            path.stat().st_size != int(record["byte_count"])
            or _sha256_file(path) != record["file_sha256"]
        ):
            raise RawSupervisionBuildError(f"staging file changed: {name}")


def _build_exact_prepared_dataset_v5(
    jobs: Sequence[PreparedSceneJobV5],
    pairs: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    input_provenance: Mapping[str, Any],
    access_ledger: Mapping[str, Any],
    prepublication_context: ExactPrepublicationContextV5,
) -> dict[str, Any]:
    """Build the one authorized exact artifact from prepared frames."""

    workers = _strict_workers(workers)
    if (
        type(prepublication_context) is not ExactPrepublicationContextV5
        or prepublication_context.workers != workers
    ):
        raise RawSupervisionBuildError("exact prepublication context changed")
    _require_exact_authority(prepublication_context.authorization_sha256)
    ordered_jobs, ordered_pairs = _validate_jobs_and_pairs(jobs, pairs)
    destination = CANONICAL_OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    retained = _open_publication_parent(destination.parent)
    retained.validate()
    try:
        os.stat(destination.name, dir_fd=retained.parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        retained.close()
        raise FileExistsError(
            f"immutable raw-supervision dataset exists: {destination}"
        )
    staging_name = f".{destination.name}.staging.{secrets.token_hex(16)}"
    staging_identity: tuple[int, int] | None = None
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=retained.parent_fd)
        retained.refresh_after_owned_mutation()
        staging_identity = _named_directory_identity(
            retained.parent_fd, staging_name
        )
        staging = retained.path / staging_name
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=retained.parent_fd,
        )
        if (
            int(os.fstat(staging_fd).st_dev),
            int(os.fstat(staging_fd).st_ino),
        ) != staging_identity:
            raise RawSupervisionBuildError("staging changed during retained open")
    except BaseException:
        _cleanup_owned_directory(retained, staging_name, staging_identity)
        retained.close()
        raise
    published = False
    try:
        (staging / "shards").mkdir(mode=0o700, exist_ok=False)
        previous = {
            name: os.environ.get(name)
            for name in (*THREAD_ENVIRONMENT, *ACCELERATOR_ENVIRONMENT)
        }
        try:
            _worker_environment()
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_exact_worker,
                initargs=(prepublication_context.authorization_sha256,),
            ) as executor:
                futures = [
                    executor.submit(
                        _write_prepared_scene_job,
                        job,
                        str(staging),
                        prepublication_context.authorization_sha256,
                    )
                    for job in ordered_jobs
                ]
                results = [future.result() for future in futures]
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        results.sort(key=lambda item: item["shard"]["scene_id"])

        endpoint_rows: list[dict[str, Any]] = []
        shard_records: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        for result in results:
            relative_directory = f"shards/{result['directory_name']}"
            shard_path = Path(result["directory_path"])
            for row in result["index_rows"]:
                endpoint_rows.append(
                    _with_content_sha256(
                        {
                            **{key: value for key, value in row.items() if key != "content_sha256"},
                            "scene_shard": f"{relative_directory}/shard.json",
                        }
                    )
                )
            for path in sorted(shard_path.iterdir(), key=lambda item: item.name):
                files.append(_file_record(path, root=staging))
            shard_records.append(
                {
                    "path": f"{relative_directory}/shard.json",
                    "dataset_role": result["shard"]["dataset_role"],
                    "family": result["shard"]["family"],
                    "scene_id": result["shard"]["scene_id"],
                    "endpoint_count": result["shard"]["endpoint_count"],
                    "content_sha256": result["shard"]["content_sha256"],
                }
            )
        endpoint_rows.sort(
            key=lambda item: (
                DEVELOPMENT_ROLES.index(item["dataset_role"]),
                item["endpoint_identity_sha256"],
            )
        )
        pair_payload = b"".join(
            canonical_json_bytes(pair) + b"\n" for pair in ordered_pairs
        )
        endpoint_payload = b"".join(
            canonical_json_bytes(endpoint) + b"\n" for endpoint in endpoint_rows
        )
        _write_bytes_exclusive(staging / "pairs.jsonl", pair_payload)
        _write_bytes_exclusive(staging / "endpoints.jsonl", endpoint_payload)
        files.extend(
            (
                _file_record(staging / "pairs.jsonl", root=staging),
                _file_record(staging / "endpoints.jsonl", root=staging),
            )
        )
        files.sort(key=lambda item: item["path"])
        retained.validate()
        if _named_directory_identity(retained.parent_fd, staging_name) != staging_identity:
            raise RawSupervisionBuildError("owned staging directory was replaced")
        sample = _precommitted_audit_sample(endpoint_rows)
        if len(sample["records"]) != 24:
            raise RawSupervisionBuildError(
                "exact development audit sample is not 24 role/family endpoints"
            )
        role_pair_counts = {
            role: sum(pair["dataset_role"] == role for pair in ordered_pairs)
            for role in DEVELOPMENT_ROLES
        }
        role_endpoint_counts = {
            role: sum(row["dataset_role"] == role for row in endpoint_rows)
            for role in DEVELOPMENT_ROLES
        }
        manifest = _with_content_sha256(
            {
                "schema": DATASET_SCHEMA,
                "status": "complete_pending_independent_audit",
                "evidence_schema": EVIDENCE_SCHEMA,
                "raster_schema": RASTER_SCHEMA,
                "roles": list(DEVELOPMENT_ROLES),
                "pair_counts": role_pair_counts,
                "endpoint_instance_count": 2 * len(ordered_pairs),
                "unique_endpoint_counts": role_endpoint_counts,
                "scene_shard_count": len(shard_records),
                "ordered_pair_sha256": canonical_json_sha256(
                    [pair["content_sha256"] for pair in ordered_pairs]
                ),
                "ordered_endpoint_sha256": canonical_json_sha256(
                    [row["content_sha256"] for row in endpoint_rows]
                ),
                "pair_index": {
                    "path": "pairs.jsonl",
                    "row_count": len(ordered_pairs),
                    "file_sha256": _sha256_bytes(pair_payload),
                },
                "endpoint_index": {
                    "path": "endpoints.jsonl",
                    "row_count": len(endpoint_rows),
                    "file_sha256": _sha256_bytes(endpoint_payload),
                },
                "array_layout": [
                    {
                        "path": name,
                        "dtype": np.dtype(dtype).str,
                        "trailing_shape": list(shape),
                    }
                    for name, dtype, shape in ARRAY_LAYOUT
                ],
                "shards": shard_records,
                "files": files,
                "input_provenance": json.loads(
                    canonical_json_bytes(input_provenance)
                ),
                "access_ledger": json.loads(canonical_json_bytes(access_ledger)),
                "independent_audit_precommit": sample,
                "parallel_contract": {
                    "worker_start_method": "spawn",
                    "maximum_workers": MAX_WORKERS,
                    "native_threads_per_worker": 1,
                    "gpu_visible_to_workers": False,
                    "merge_order": "role_then_scene_then_endpoint_identity",
                    "worker_count_does_not_change_artifact_bytes": True,
                },
                "publication": {
                    "staging": "private_sibling_directory_mode_0700",
                    "commit": "single_renameat2_RENAME_NOREPLACE",
                    "manifest_self_inventory": "canonical_content_sha256",
                    "file_inventory": "every_regular_file_except_manifest_self",
                },
                "licenses": {
                    "independent_audit_passed": False,
                    "dataset_use_authorized": False,
                    "rgb_decode_authorized": False,
                    "training_authorized": False,
                    "selection_authorized": False,
                    "calibration_authorized": False,
                    "g2_authorized": False,
                    "heldout_authorized": False,
                    "runtime_authorized": False,
                    "hardware_authorized": False,
                    "production_authorized": False,
                    "promotion_authorized": False,
                },
            }
        )
        _validate_staging_inventory(staging, files, manifest_present=False)
        _write_json_exclusive(staging / "manifest.json", manifest)
        _validate_staging_inventory(staging, files, manifest_present=True)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        os.fsync(staging_fd)
        retained.validate()
        if _named_directory_identity(retained.parent_fd, staging_name) != staging_identity:
            raise RawSupervisionBuildError("owned staging directory changed before commit")
        os.fsync(retained.parent_fd)
        _revalidate_exact_before_publication(prepublication_context)
        retained.validate()
        if _named_directory_identity(retained.parent_fd, staging_name) != staging_identity:
            raise RawSupervisionBuildError(
                "owned staging directory changed after source revalidation"
            )
        _libc_renameat2(
            retained.parent_fd,
            staging_name,
            retained.parent_fd,
            destination.name,
        )
        published = True
        retained.refresh_after_owned_mutation()
        if _named_directory_identity(retained.parent_fd, destination.name) != staging_identity:
            raise RawSupervisionBuildError("published directory identity changed")
        os.fsync(retained.parent_fd)
        return manifest
    except BaseException:
        _cleanup_owned_directory(
            retained,
            destination.name if published else staging_name,
            staging_identity,
        )
        raise
    finally:
        if not published:
            _cleanup_owned_directory(retained, staging_name, staging_identity)
        os.close(staging_fd)
        retained.close()


ACCESS_LEDGER_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_access_ledger_v1"
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


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RawSupervisionBuildError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawSupervisionBuildError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RawSupervisionBuildError(f"{name} must be a JSON object")
    return value


def _source_path_from_summary(value: object) -> Path:
    path = Path(str(value))
    candidate = path if path.is_absolute() else ROOT / path
    return candidate.absolute()


def _pair_endpoint_contexts(
    plan: DevelopmentRawSupervisionPlan,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    endpoints = {
        str(item["identity_sha256"]): item for item in plan.endpoints
    }
    if len(endpoints) != len(plan.endpoints):
        raise RawSupervisionBuildError("metadata plan repeats an endpoint identity")
    for pair in plan.pairs:
        for side in ("current", "next"):
            digest = str(pair[f"{side}_endpoint_sha256"])
            endpoint = endpoints.get(digest)
            if endpoint is None:
                raise RawSupervisionBuildError("metadata pair references an absent endpoint")
            identity = endpoint["identity"]
            candidate = {
                "scene_id": identity["scene_id"],
                "family": pair["family"],
                "episode_id": identity["episode_id"],
                "reset_count": pair["reset_count"],
                "episode_step": identity["episode_step"],
                "frame_index": identity["frame_index"],
                "env_index": identity["env_index"],
                "timestamp_ns": identity["timestamp_ns"],
                "image_sha256": identity["image_sha256"],
                "image_path_metadata_only": endpoint["image_path_metadata_only"],
            }
            previous = contexts.setdefault(digest, candidate)
            if previous != candidate:
                raise RawSupervisionBuildError(
                    "one endpoint has conflicting pair/source join metadata"
                )
    if set(contexts) != set(endpoints):
        raise RawSupervisionBuildError("metadata plan contains an orphan endpoint")
    return contexts


def _read_exact_source(
    *,
    path: Path,
    expected_sha256: str,
    authorization_sha256: str,
) -> bytes:
    _require_exact_authority(authorization_sha256)
    return _read_bound_regular_file(
        repository_root=ROOT,
        path=path,
        expected_sha256=expected_sha256,
    )


def _validate_summary_source_binding(
    summary: Mapping[str, Any],
    source_record: Mapping[str, Any],
) -> None:
    source = summary.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "plan",
        "frames_jsonl",
        "scene_manifest",
        "renderer_source",
    }:
        raise RawSupervisionBuildError("render summary source inventory is absent")
    expected = {
        "frames_jsonl": (
            source_record["frames"]["path"],
            source_record["frames"]["sha256"],
        ),
        "scene_manifest": (
            source_record["scene_manifest"]["path"],
            source_record["scene_manifest"]["file_sha256"],
        ),
        "plan": (
            source_record["render_plan"]["path"],
            source_record["render_plan"]["sha256"],
        ),
    }
    for name, (path, digest) in expected.items():
        entry = source.get(name)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256"}
            or _source_path_from_summary(entry["path"]) != Path(str(path))
            or entry.get("sha256") != digest
        ):
            raise RawSupervisionBuildError(
                f"render summary source.{name} differs from inventory"
            )


def _validate_render_object_parity(
    source_v4: Any,
    summary: Mapping[str, Any],
    manifest: Any,
) -> tuple[Any, ...]:
    rendered_boxes = tuple(source_v4._rendered_boxes(manifest))
    records = source_v4.labels_v3._render_object_records(manifest)
    object_ids = sorted(str(item["object_id"]) for item in records)
    parity = summary.get("object_parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("schema") != "lewm_render_object_parity_v1"
        or parity.get("rendered_groups")
        != ["wall", "obstacle", "landmark", "distractor"]
        or parity.get("rendered_object_count") != len(records)
        or parity.get("rendered_object_ids") != object_ids
        or parity.get("rendered_object_ids_sha256")
        != source_v4.canonical_json_sha256(object_ids)
        or parity.get("rendered_object_records_sha256")
        != source_v4.canonical_json_sha256(records)
        or parity.get("collision_distractors_rendered") is not True
        or parity.get("full_box_roll_pitch_yaw_rendered") is not True
    ):
        raise RawSupervisionBuildError("rendered full-RPY object parity changed")
    return rendered_boxes


def _reviewed_v4_source_semantics() -> tuple[Any, Any]:
    from lewm.benchmarks import go2_n32_camera_frustum_observability as core
    from lewm.datasets import go2_paired_navigation as label_semantics
    from lewm_worlds import manifest as manifest_semantics
    from lewm_worlds import planning_grid as planning_semantics
    from scripts import audit_go2_n32_camera_frustum_observability as source_v4

    if not source_v4._SEMANTICS_LOADED:
        source_v4._install_semantic_modules(
            core,
            label_semantics,
            manifest_semantics,
            planning_semantics,
        )
    return source_v4, manifest_semantics


def _load_exact_scene_job(
    source_record: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    authorization_sha256: str,
) -> dict[str, Any]:
    """Worker-local exact source opener and source-to-V4-frame conversion."""

    _worker_environment()
    _require_exact_authority(authorization_sha256)
    scene_id = str(source_record["scene_id"])
    role = str(source_record["role"])
    family = str(source_record["family"])
    bindings = (
        ("frames", "source_frames_jsonl", "sha256"),
        ("scene_manifest", "source_scene_manifest", "file_sha256"),
        ("render_plan", "render_plan", "sha256"),
        ("render_summary", "render_summary", "sha256"),
    )
    payloads: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    for key, purpose, hash_key in bindings:
        record = source_record[key]
        path = Path(str(record["path"]))
        digest = str(record[hash_key])
        payload = _read_exact_source(
            path=path,
            expected_sha256=digest,
            authorization_sha256=authorization_sha256,
        )
        payloads[key] = payload
        receipts.append(
            {
                "path": str(path),
                "file_sha256": digest,
                "byte_count": len(payload),
                "purpose": purpose,
                "scene_id": scene_id,
            }
        )

    from lewm.benchmarks import go2_dynamic_cell_square_projection as dynamic_projection

    source_v4, manifest_semantics = _reviewed_v4_source_semantics()

    summary = _strict_json_bytes(payloads["render_summary"], name="render summary")
    render_plan = _strict_json_bytes(payloads["render_plan"], name="render plan")
    manifest_payload = _strict_json_bytes(
        payloads["scene_manifest"], name="scene manifest"
    )
    for endpoint in endpoints:
        identity = endpoint.get("identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("scene_id") != scene_id
            or identity.get("dataset_role") != role
            or endpoint.get("frames_jsonl_sha256")
            != source_record["frames"]["sha256"]
            or endpoint.get("scene_manifest_sha256")
            != source_record["scene_manifest"]["content_sha256"]
        ):
            raise RawSupervisionBuildError(
                "planned endpoint crossed its frozen source binding"
            )
    selected_records = [dict(contexts[str(item["identity_sha256"])]) for item in endpoints]
    source_v4._validate_summary_records(
        summary,
        selected_records,
        summary_path=Path(str(source_record["render_summary"]["path"])),
    )
    _validate_summary_source_binding(summary, source_record)
    if (
        render_plan.get("schema") != "lewm_render_replay_plan_v0"
        or render_plan.get("scene_id") != scene_id
    ):
        raise RawSupervisionBuildError("render plan scene/schema changed")
    camera_record = render_plan.get("camera")
    if not isinstance(camera_record, Mapping):
        raise RawSupervisionBuildError("render plan camera contract is absent")
    plan_mount = source_v4._camera_mount_record(
        camera_record.get("mount_body"), label="render plan camera.mount_body"
    )
    if (
        set(camera_record)
        != {
            "native_resolution",
            "training_resolution",
            "fov_axis",
            "fov_deg",
            "near_m",
            "far_m",
            "encoding",
            "mount_body",
        }
        or camera_record.get("fov_axis") != "horizontal"
        or not math.isclose(
            float(camera_record.get("fov_deg", math.nan)),
            78.323,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(camera_record.get("near_m", math.nan)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or _source_path_from_summary(render_plan.get("frames_jsonl"))
        != Path(str(source_record["frames"]["path"]))
    ):
        raise RawSupervisionBuildError("render plan camera/source binding changed")
    projection = summary.get("camera_projection")
    expected_projection_keys = {
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
            math.tan(math.radians(float(camera_record["fov_deg"])) * 0.5)
            * (168.0 / 224.0)
        )
    )
    if (
        not isinstance(projection, Mapping)
        or set(projection) != expected_projection_keys
        or projection.get("model") != "pinhole"
        or projection.get("renderer_fov_axis") != "vertical"
        or projection.get("runtime_rectification_required") is not False
        or summary.get("resolution_wh") != [224, 168]
        or not math.isclose(
            float(projection.get("horizontal_fov_deg", math.nan)),
            float(camera_record["fov_deg"]),
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
            float(camera_record["near_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(projection.get("far_m", math.nan)),
            float(camera_record["far_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RawSupervisionBuildError("render plan/summary projection changed")
    source_v4._validate_raw_scene_object_records(manifest_payload)
    manifest = source_v4.parse_scene_manifest_dict(manifest_payload)
    if (
        manifest.scene_id != scene_id
        or manifest.family != family
        or manifest_semantics.manifest_sha256(manifest)
        != str(source_record["scene_manifest"]["content_sha256"])
    ):
        raise RawSupervisionBuildError("source scene manifest identity changed")
    raw_boxes = _validate_render_object_parity(source_v4, summary, manifest)

    wanted: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    endpoint_by_digest = {str(item["identity_sha256"]): item for item in endpoints}
    for digest, endpoint in endpoint_by_digest.items():
        identity = endpoint["identity"]
        key = (
            int(identity["frame_index"]),
            int(identity["env_index"]),
            int(identity["timestamp_ns"]),
        )
        if key in wanted:
            raise RawSupervisionBuildError("two endpoint identities share one source key")
        wanted[key] = endpoint
    frames_payload = payloads["frames"]
    if not frames_payload.endswith(b"\n"):
        raise RawSupervisionBuildError("source frames JSONL lacks terminal newline")
    selected_frames: dict[str, Mapping[str, Any]] = {}
    scanned = 0
    for line_number, line in enumerate(frames_payload.splitlines(), start=1):
        if not line:
            raise RawSupervisionBuildError("source frames JSONL contains a blank row")
        frame = _strict_json_bytes(line, name=f"source frame line {line_number}")
        scanned += 1
        raw_key = tuple(frame.get(name) for name in ("frame_index", "env_index", "timestamp_ns"))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_key):
            raise RawSupervisionBuildError("source frame key is not strict integer metadata")
        endpoint = wanted.get(tuple(map(int, raw_key)))
        if endpoint is None:
            continue
        digest = str(endpoint["identity_sha256"])
        if digest in selected_frames:
            raise RawSupervisionBuildError("source frame matched an endpoint twice")
        selected_frames[digest] = frame
    if set(selected_frames) != set(endpoint_by_digest):
        raise RawSupervisionBuildError("source frames did not match every endpoint once")

    prepared: list[PreparedEndpointV5] = []
    for digest in sorted(endpoint_by_digest):
        endpoint = endpoint_by_digest[digest]
        context = contexts[digest]
        extracted = source_v4._extract_source_frame(
            selected_frames[digest],
            context,
            plan_camera_mount_body=plan_mount,
        )
        position = extracted["base_pose_world"]["position"]
        base_position = tuple(float(position[axis]) for axis in ("x", "y", "z"))
        quaternion, stored_yaw = _validated_sidecar_source_attitude(
            extracted, endpoint
        )
        camera = dynamic_projection.compose_yaw_aligned_camera(quaternion, stored_yaw)
        rendered_boxes = tuple(
            _box_in_yaw_body(
                box,
                base_position_world=base_position,
                stored_yaw_rad=stored_yaw,
            )
            for box in raw_boxes
        )
        frame = FrameBuildInputV4(
            frame_key={"endpoint_identity_sha256": digest},
            camera_origin_body_m=tuple(camera.origin_xyz),
            camera_basis_body_fru=_normalized_camera_basis_fru(camera),
            ground_plane_z_body_m=-base_position[2],
            rendered_boxes_body=rendered_boxes,
            image_path_metadata_only=str(endpoint["image_path_metadata_only"]),
            image_sha256=str(endpoint["identity"]["image_sha256"]),
            sidecar_row_identity_sha256=str(endpoint["content_sha256"]),
        )
        prepared.append(
            PreparedEndpointV5(
                plan_endpoint=endpoint,
                family=family,
                frame=frame,
            )
        )
    return {
        "job": PreparedSceneJobV5(
            scene_id=scene_id,
            role=role,
            family=family,
            endpoints=tuple(prepared),
        ),
        "source_receipts": receipts,
        "source_frames_jsonl_records_scanned": scanned,
        "source_frames_selected_records": len(prepared),
    }


def _load_parent_contracts(authorization_sha256: str) -> tuple[dict[str, Any], ...]:
    _require_exact_authority(authorization_sha256)
    from scripts import audit_go2_n32_camera_frustum_observability as source_v4

    geometry_raw = _read_exact_source(
        path=GEOMETRY_CONTRACT_PATH,
        expected_sha256=GEOMETRY_CONTRACT_FILE_SHA256,
        authorization_sha256=authorization_sha256,
    )
    render_raw = _read_exact_source(
        path=RENDER_AUDIT_PATH,
        expected_sha256=RENDER_AUDIT_FILE_SHA256,
        authorization_sha256=authorization_sha256,
    )
    geometry = _strict_json_bytes(geometry_raw, name="geometry contract")
    render = _strict_json_bytes(render_raw, name="render audit")
    if source_v4._geometry_semantic_sha256(geometry) != GEOMETRY_CONTRACT_CONTENT_SHA256:
        raise RawSupervisionBuildError("geometry contract semantic hash changed")
    source_v4._geometry_flags(geometry)
    source_v4._validate_render_audit_contract(
        render, expected_content_sha256=RENDER_AUDIT_CONTENT_SHA256
    )
    return (
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
    )


def _validate_exact_plan_result(
    plan: DevelopmentRawSupervisionPlan,
    inventory: DevelopmentSourceInventory,
) -> None:
    if (
        plan.value.get("content_sha256")
        != "8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3"
        or plan.value.get("ordered_pair_sha256")
        != "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
        or plan.value.get("ordered_endpoint_sha256")
        != "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
        or len(plan.pairs) != 5172
        or len(plan.endpoints) != 9460
        or len(inventory.records) != 88
        or dict(inventory.hashes) != dict(SOURCE_INVENTORY_SHA256)
        or any(bool(value) for value in plan.value.get("licenses", {}).values())
    ):
        raise RawSupervisionBuildError("exact metadata-plan result changed")


def _run_exact_scene_load_pool(
    argument_rows: Sequence[tuple[Any, ...]],
    *,
    workers: int,
    authorization_sha256: str,
) -> list[Any]:
    workers = _strict_workers(workers)
    _require_exact_authority(authorization_sha256)
    previous = {
        name: os.environ.get(name)
        for name in (*THREAD_ENVIRONMENT, *ACCELERATOR_ENVIRONMENT)
    }
    try:
        _worker_environment()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_exact_worker,
            initargs=(authorization_sha256,),
        ) as executor:
            futures = [
                executor.submit(_load_exact_scene_job, *arguments)
                for arguments in argument_rows
            ]
            return [future.result() for future in futures]
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_exact_source_revalidation_pool(
    argument_rows: Sequence[tuple[Any, ...]],
    *,
    workers: int,
    authorization_sha256: str,
) -> list[tuple[str, ...]]:
    workers = _strict_workers(workers)
    _require_exact_authority(authorization_sha256)
    previous = {
        name: os.environ.get(name)
        for name in (*THREAD_ENVIRONMENT, *ACCELERATOR_ENVIRONMENT)
    }
    try:
        _worker_environment()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_exact_worker,
            initargs=(authorization_sha256,),
        ) as executor:
            futures = [
                executor.submit(_revalidate_exact_scene_sources, *arguments)
                for arguments in argument_rows
            ]
            return [future.result() for future in futures]
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _revalidate_exact_scene_sources(
    source_record: Mapping[str, Any],
    authorization_sha256: str,
) -> tuple[str, ...]:
    _worker_environment()
    _require_exact_authority(authorization_sha256)
    observed: list[str] = []
    for key, hash_key in (
        ("frames", "sha256"),
        ("scene_manifest", "file_sha256"),
        ("render_plan", "sha256"),
        ("render_summary", "sha256"),
    ):
        record = source_record[key]
        path = Path(str(record["path"]))
        _read_bound_regular_file(
            repository_root=ROOT,
            path=path,
            expected_sha256=str(record[hash_key]),
        )
        observed.append(str(path))
    return tuple(observed)


def _revalidate_exact_before_publication(
    context: ExactPrepublicationContextV5,
) -> None:
    if type(context) is not ExactPrepublicationContextV5:
        raise RawSupervisionBuildError("exact prepublication context changed")
    workers = _strict_workers(context.workers)
    _require_exact_authority(context.authorization_sha256)
    from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan_v5 import (
        load_frozen_development_metadata,
        load_frozen_development_source_inventory,
    )

    second_plan = load_frozen_development_metadata(ROOT)
    second_inventory = load_frozen_development_source_inventory(
        ROOT, second_plan
    )
    _validate_exact_plan_result(second_plan, second_inventory)
    if (
        canonical_json_bytes(second_plan.value)
        != canonical_json_bytes(context.plan.value)
        or canonical_json_bytes(second_plan.pairs)
        != canonical_json_bytes(context.plan.pairs)
        or canonical_json_bytes(second_plan.endpoints)
        != canonical_json_bytes(context.plan.endpoints)
        or canonical_json_bytes(second_inventory.records)
        != canonical_json_bytes(context.inventory.records)
    ):
        raise RawSupervisionBuildError("parent metadata changed before publication")
    _load_parent_contracts(context.authorization_sha256)
    revalidated = _run_exact_source_revalidation_pool(
        [
            (source_record, context.authorization_sha256)
            for source_record in context.source_records
        ],
        workers=workers,
        authorization_sha256=context.authorization_sha256,
    )
    if len(revalidated) != 88 or sum(map(len, revalidated)) != 352:
        raise RawSupervisionBuildError("source revalidation inventory changed")


def _ensure_exact_output_container() -> None:
    container = CANONICAL_OUTPUT.parent
    if container.exists():
        if container.resolve(strict=True) != container or container.is_symlink():
            raise PermissionError("exact output container is not canonical")
        return
    retained = _open_publication_parent(container.parent)
    try:
        retained.validate()
        try:
            os.mkdir(container.name, mode=0o755, dir_fd=retained.parent_fd)
        except FileExistsError:
            pass
        retained.refresh_after_owned_mutation()
        metadata = os.stat(
            container.name, dir_fd=retained.parent_fd, follow_symlinks=False
        )
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("exact output container was replaced")
    finally:
        retained.close()


def _exact_access_ledger(
    *,
    plan: DevelopmentRawSupervisionPlan,
    inventory: DevelopmentSourceInventory,
    frames_scanned: int,
) -> dict[str, Any]:
    ledger = {
        "schema": ACCESS_LEDGER_SCHEMA,
        "measurement_scope": (
            "controlled_data_opens_excluding_import_and_reviewed_source_hash_reads"
        ),
        "metadata_plan_first_pass": plan.value["access_ledger"],
        "metadata_source_inventory_first_pass": inventory.access_ledger,
        "metadata_plan_second_pass": plan.value["access_ledger"],
        "metadata_source_inventory_second_pass": inventory.access_ledger,
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
    if set(ledger) != EXACT_ACCESS_LEDGER_KEYS:
        raise AssertionError("exact access ledger schema changed")
    return ledger


def _strict_canonical_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    duplicate: str | None = None

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result and duplicate is None:
                duplicate = key
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawSupervisionBuildError(f"{name} is invalid JSON") from error
    if duplicate is not None:
        raise RawSupervisionBuildError(f"{name} has duplicate key {duplicate!r}")
    if type(value) is not dict:
        raise RawSupervisionBuildError(f"{name} is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise RawSupervisionBuildError(f"{name} is not canonical JSON")
    return value


def _canonical_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise RawSupervisionBuildError("authorization source path is noncanonical")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RawSupervisionBuildError("authorization source path is noncanonical")
    return value


def _source_binding(entry: object) -> SourceBindingV5:
    if type(entry) is not dict or set(entry) != SOURCE_ENTRY_FIELDS:
        raise RawSupervisionBuildError("authorization source entry is malformed")
    role = entry["role"]
    if type(role) is not str or not role:
        raise RawSupervisionBuildError("authorization source entry is malformed")
    path = _canonical_relative_path(entry["path"])
    digest = entry["sha256"]
    if not _is_sha256(digest):
        raise RawSupervisionBuildError("authorization source entry is malformed")
    return SourceBindingV5(role=role, path=path, sha256=digest)


def _candidate_bindings(
    value: object,
    *,
    roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV5],
) -> tuple[SourceBindingV5, ...]:
    if type(value) is not list or len(value) != len(roles):
        raise RawSupervisionBuildError("authorization review candidate changed")
    candidate = tuple(_source_binding(entry) for entry in value)
    if tuple(item.role for item in candidate) != tuple(roles):
        raise RawSupervisionBuildError("authorization review candidate roles changed")
    if candidate != tuple(source_by_role[role] for role in roles):
        raise RawSupervisionBuildError("authorization review candidate binding changed")
    return candidate


def _review_binding(
    value: object,
    *,
    kind: str,
    review_role: str,
    review_schema: str,
    implementation_author: str,
    candidate_roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV5],
) -> ReviewBindingV5:
    if type(value) is not dict or set(value) != REVIEW_BINDING_FIELDS:
        raise RawSupervisionBuildError(f"authorization {kind} review binding is malformed")
    if (
        value["schema"] != REVIEW_BINDING_SCHEMA
        or value["review_schema"] != review_schema
        or value["verdict"] != "PASS"
    ):
        raise PermissionError(f"authorization {kind} review is not a bound PASS")
    reviewer = value["reviewer"]
    author = value["implementation_author"]
    if (
        type(reviewer) is not str
        or not reviewer
        or type(author) is not str
        or not author
        or reviewer == author
    ):
        raise PermissionError(f"authorization {kind} review lacks a distinct reviewer")
    if author != implementation_author:
        raise PermissionError(f"authorization {kind} implementation author changed")
    review_source = source_by_role[review_role]
    path = _canonical_relative_path(value["path"])
    if (
        path != review_source.path
        or value["file_sha256"] != review_source.sha256
        or not _is_sha256(value["content_sha256"])
    ):
        raise RawSupervisionBuildError(f"authorization {kind} review file binding changed")
    candidate = _candidate_bindings(
        value["candidate"], roles=candidate_roles, source_by_role=source_by_role
    )
    return ReviewBindingV5(
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
) -> PhaseOneAuthorizationV5:
    """Validate the complete V5 authority without opening a target file."""

    if type(payload) is not dict or set(payload) != AUTHORIZATION_FIELDS:
        raise RawSupervisionBuildError("build authorization object fields changed")
    if not _is_sha256(authorization_file_sha256):
        raise PermissionError("build authorization file hash is not frozen")
    if (
        payload["schema"] != AUTHORIZATION_SCHEMA
        or payload["exact_build_authorized_after_independent_reviews"] is not True
    ):
        raise PermissionError("raw-supervision exact build is not authorized")
    core = dict(payload)
    declared = core.pop("content_sha256")
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise RawSupervisionBuildError("build authorization content hash changed")
    if (
        len(AUTHORIZED_ROLE_PATHS) != 9
        or len({role for role, _path in AUTHORIZED_ROLE_PATHS}) != 9
        or len({path for _role, path in AUTHORIZED_ROLE_PATHS}) != 9
    ):
        raise AssertionError("V5 authorization policy is not nine unique rows")
    source_map = payload["source_map"]
    if type(source_map) is not list:
        raise RawSupervisionBuildError("build authorization source map is absent")
    sources = tuple(_source_binding(entry) for entry in source_map)
    observed_roles = tuple(item.role for item in sources)
    observed_paths = tuple(item.path for item in sources)
    if len(set(observed_roles)) != len(observed_roles):
        raise RawSupervisionBuildError("authorization source roles are duplicated")
    if len(set(observed_paths)) != len(observed_paths):
        raise RawSupervisionBuildError("authorization source paths are duplicated")
    if observed_roles != tuple(role for role, _path in AUTHORIZED_ROLE_PATHS):
        raise RawSupervisionBuildError("authorization source roles changed")
    if observed_paths != tuple(path for _role, path in AUTHORIZED_ROLE_PATHS):
        raise RawSupervisionBuildError("authorization role-to-path mapping changed")
    source_by_role = {item.role: item for item in sources}
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
        raise PermissionError("builder and auditor implementations are not independent")
    if builder_review.reviewer == auditor_review.reviewer:
        raise PermissionError("builder and auditor reviews are not independent")
    normalized = json.loads(canonical_json_bytes(payload))
    return PhaseOneAuthorizationV5(
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
    candidate: Sequence[SourceBindingV5],
) -> list[dict[str, str]]:
    return [
        {"role": item.role, "path": item.path, "sha256": item.sha256}
        for item in candidate
    ]


def _validate_review_record(raw: bytes, binding: ReviewBindingV5) -> None:
    review = _strict_canonical_json_object(
        raw, name=f"{binding.kind} independent review"
    )
    if set(review) != REVIEW_RECORD_FIELDS:
        raise RawSupervisionBuildError(
            f"{binding.kind} independent review fields changed"
        )
    core = dict(review)
    declared = core.pop("content_sha256")
    if (
        not _is_sha256(declared)
        or canonical_json_sha256(core) != declared
        or declared != binding.content_sha256
    ):
        raise RawSupervisionBuildError(
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
    phase_one: PhaseOneAuthorizationV5,
) -> AcceptedAuthorizationV5:
    """Rehash the fixed V5 closure after a complete structural phase."""

    if type(phase_one) is not PhaseOneAuthorizationV5:
        raise TypeError("phase two requires a completed V5 phase-one capsule")
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
        payload_by_role[source.role] = _read_bound_regular_file(
            repository_root=ROOT,
            path=(ROOT / source.path).absolute(),
            expected_sha256=source.sha256,
        )
    _validate_review_record(
        payload_by_role["builder_review"], phase_one.builder_review
    )
    _validate_review_record(
        payload_by_role["auditor_review"], phase_one.auditor_review
    )
    for relative, expected in {
        **FROZEN_PARENT_HASHES,
        **REVIEWED_V4_SOURCES,
    }.items():
        _read_bound_regular_file(
            repository_root=ROOT,
            path=(ROOT / relative).absolute(),
            expected_sha256=expected,
        )
    return AcceptedAuthorizationV5(
        authorization_file_sha256=phase_one.authorization_file_sha256,
        authorization_content_sha256=phase_one.authorization_content_sha256,
        source_map_sha256=phase_one.source_map_sha256,
    )


def _require_exact_authority(
    authorization_sha256: str,
) -> AcceptedAuthorizationV5:
    if not _is_sha256(authorization_sha256):
        raise PermissionError("build authorization file hash is not frozen")
    if not AUTHORIZATION_PATH.is_file():
        raise PermissionError("reviewed raw-supervision build authorization is absent")
    raw = _read_bound_regular_file(
        repository_root=ROOT,
        path=AUTHORIZATION_PATH,
        expected_sha256=authorization_sha256,
    )
    payload = _strict_canonical_json_object(raw, name="build authorization")
    phase_one = _validate_authorization_phase_one(
        payload, authorization_file_sha256=authorization_sha256
    )
    return _validate_authorization_phase_two(phase_one)


def _write_failure_receipt(
    *,
    authorization_sha256: str,
    error: BaseException,
) -> None:
    FAILURE_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    receipt = _with_content_sha256(
        {
            "schema": FAILURE_SCHEMA,
            "status": "terminal_failed_no_dataset_authority",
            "authorization_file_sha256": authorization_sha256,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "canonical_output_present": CANONICAL_OUTPUT.exists(),
            "retry_authorized": False,
        }
    )
    payload = canonical_json_bytes(receipt) + b"\n"
    retained = _open_publication_parent(FAILURE_RECEIPT.parent)
    try:
        retained.validate()
        try:
            os.stat(
                FAILURE_RECEIPT.name,
                dir_fd=retained.parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            return
        _write_bytes_exclusive_at(retained.parent_fd, FAILURE_RECEIPT.name, payload)
        retained.refresh_after_owned_mutation()
        _read_bound_regular_file(
            repository_root=ROOT if FAILURE_RECEIPT.is_relative_to(ROOT) else FAILURE_RECEIPT.parent,
            path=FAILURE_RECEIPT,
            expected_sha256=_sha256_bytes(payload),
        )
        os.fsync(retained.parent_fd)
    finally:
        retained.close()


def execute_exact_build_v5(*, authorization_sha256: str, workers: int) -> dict[str, Any]:
    """Build all development labels after, and only after, dual source review."""

    authority_verified = False
    try:
        workers = _strict_workers(workers)
        authority = _require_exact_authority(authorization_sha256)
        authority_verified = True
        _ensure_exact_output_container()
        if FAILURE_RECEIPT.exists():
            raise PermissionError("a prior exact build failure is terminal")
        if CANONICAL_OUTPUT.exists():
            raise FileExistsError("the immutable exact raw-supervision dataset exists")

        from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan_v5 import (
            load_frozen_development_metadata,
            load_frozen_development_source_inventory,
        )

        plan = load_frozen_development_metadata(ROOT)
        inventory = load_frozen_development_source_inventory(ROOT, plan)
        _validate_exact_plan_result(plan, inventory)
        contexts = _pair_endpoint_contexts(plan)
        endpoints_by_scene: dict[str, list[Mapping[str, Any]]] = {}
        for endpoint in plan.endpoints:
            scene_id = str(endpoint["identity"]["scene_id"])
            endpoints_by_scene.setdefault(scene_id, []).append(endpoint)
        source_by_scene = {
            str(record["scene_id"]): record for record in inventory.records
        }
        if set(source_by_scene) != set(endpoints_by_scene):
            raise RawSupervisionBuildError("source inventory and endpoint scenes differ")

        parent_contract_receipts = list(_load_parent_contracts(authorization_sha256))
        load_arguments = [
            (
                source_by_scene[scene_id],
                tuple(endpoints_by_scene[scene_id]),
                {
                    str(endpoint["identity_sha256"]): contexts[
                        str(endpoint["identity_sha256"])
                    ]
                    for endpoint in endpoints_by_scene[scene_id]
                },
                authorization_sha256,
            )
            for scene_id in sorted(source_by_scene)
        ]
        loaded = _run_exact_scene_load_pool(
            load_arguments,
            workers=workers,
            authorization_sha256=authorization_sha256,
        )
        jobs = tuple(item["job"] for item in loaded)
        if len(jobs) != 88 or sum(len(job.endpoints) for job in jobs) != 9460:
            raise RawSupervisionBuildError("exact scene jobs changed population")
        source_receipts = parent_contract_receipts + [
            receipt for item in loaded for receipt in item["source_receipts"]
        ]
        if len(source_receipts) != 354:
            raise RawSupervisionBuildError("exact source payload inventory changed")
        source_receipts.sort(
            key=lambda item: (
                str(item["path"]),
                str(item["purpose"]),
                str(item.get("scene_id", "")),
            )
        )
        frames_scanned = sum(
            int(item["source_frames_jsonl_records_scanned"]) for item in loaded
        )
        access_ledger = _exact_access_ledger(
            plan=plan,
            inventory=inventory,
            frames_scanned=frames_scanned,
        )
        input_provenance = {
            "authorization_file_sha256": authority.authorization_file_sha256,
            "authorization_content_sha256": authority.authorization_content_sha256,
            "authorization_source_map_sha256": authority.source_map_sha256,
            "frozen_parent_file_sha256": dict(FROZEN_PARENT_HASHES),
            "reviewed_v4_source_sha256": dict(REVIEWED_V4_SOURCES),
            "metadata_plan_content_sha256": plan.value["content_sha256"],
            "metadata_ordered_pair_sha256": plan.value["ordered_pair_sha256"],
            "metadata_ordered_endpoint_sha256": plan.value[
                "ordered_endpoint_sha256"
            ],
            "source_inventory_sha256": dict(inventory.hashes),
            "source_payload_inventory": source_receipts,
            "source_payload_inventory_sha256": canonical_json_sha256(
                source_receipts
            ),
            "geometry_contract_file_sha256": GEOMETRY_CONTRACT_FILE_SHA256,
            "geometry_contract_content_sha256": GEOMETRY_CONTRACT_CONTENT_SHA256,
            "render_audit_file_sha256": RENDER_AUDIT_FILE_SHA256,
            "render_audit_content_sha256": RENDER_AUDIT_CONTENT_SHA256,
        }

        prepublication_context = ExactPrepublicationContextV5(
            plan=plan,
            inventory=inventory,
            source_records=tuple(
                source_by_scene[scene_id] for scene_id in sorted(source_by_scene)
            ),
            authorization_sha256=authorization_sha256,
            workers=workers,
        )
        return _build_exact_prepared_dataset_v5(
            jobs,
            plan.pairs,
            workers=workers,
            input_provenance=input_provenance,
            access_ledger=access_ledger,
            prepublication_context=prepublication_context,
        )
    except BaseException as error:
        if (
            authority_verified
            and not FAILURE_RECEIPT.exists()
        ):
            _write_failure_receipt(
                authorization_sha256=authorization_sha256,
                error=error,
            )
        raise


__all__ = [
    "ACCELERATOR_ENVIRONMENT",
    "ARRAY_LAYOUT",
    "AUTHORIZATION_PATH",
    "CANONICAL_OUTPUT",
    "DATASET_SCHEMA",
    "FAILURE_RECEIPT",
    "MAX_WORKERS",
    "PreparedEndpointV5",
    "PreparedSceneJobV5",
    "RawSupervisionBuildError",
    "THREAD_ENVIRONMENT",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "execute_exact_build_v5",
]

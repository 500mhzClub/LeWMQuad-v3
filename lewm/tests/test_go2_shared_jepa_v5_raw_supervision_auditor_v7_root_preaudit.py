"""Source-only cross-binding probes prepared for the distinct Auditor reviewer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v7 as auditor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v7 as builder


ROOT = Path(__file__).resolve().parents[2]
BUILDER_REVIEW_PATH = (
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_"
    "independent_review_2026-07-13.json"
)
BUILDER_REVIEW_FILE_SHA256 = (
    "85d1a111e10eaac865a80cebd97e771b39eaa47f6ebcf6ffe6716ed445a1ff46"
)
AUDITOR_CANDIDATE_SHA256 = {
    "auditor_source": "3550917e36d1401f8ad9c895afcf591b3226b2e0c5a09f4ad427d0b04bb1490e",
    "auditor_cli": "9940d35e4e33b628bf64c4947cb1f92a68e1413e20e63fd0b9080728a64f949e",
    "auditor_test": "6d123d39014fd9c3dc7b34d113e665861536010d79117a3004cb8ee1484e894f",
}
ROLE_PATH_BY_NAME = dict(auditor.SOURCE_ROLE_PATHS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_map_and_records() -> tuple[list[dict[str, str]], bytes, bytes]:
    builder_review_raw = (ROOT / BUILDER_REVIEW_PATH).read_bytes()
    builder_review = json.loads(builder_review_raw)
    auditor_candidate = [
        {
            "role": role,
            "path": ROLE_PATH_BY_NAME[role],
            "sha256": AUDITOR_CANDIDATE_SHA256[role],
        }
        for role in auditor.AUDITOR_CANDIDATE_ROLES
    ]
    auditor_core = {
        "schema": auditor.AUDITOR_REVIEW_SCHEMA,
        "verdict": "PASS",
        "reviewer": "/synthetic/distinct_auditor_v7_reviewer",
        "implementation_author": auditor.AUDITOR_IMPLEMENTATION_AUTHOR,
        "candidate": auditor_candidate,
        "authority": auditor._expected_review_authority("auditor"),
    }
    auditor_review = {
        **auditor_core,
        "content_sha256": auditor.canonical_json_sha256(auditor_core),
    }
    auditor_review_raw = auditor.canonical_json_bytes(auditor_review) + b"\n"
    hashes = {
        **auditor.FROZEN_BUILDER_V7_ROLE_SHA256,
        "builder_review": BUILDER_REVIEW_FILE_SHA256,
        **AUDITOR_CANDIDATE_SHA256,
        "auditor_review": hashlib.sha256(auditor_review_raw).hexdigest(),
    }
    source_map = [
        {"role": role, "path": path, "sha256": hashes[role]}
        for role, path in auditor.SOURCE_ROLE_PATHS
    ]
    assert builder_review["reviewer"] == "/root"
    assert builder_review["content_sha256"] == (
        "24ffe7b0c8fdba7d0e60636b865d1bf01d443e5527aa8c3db3e8eca0170e6202"
    )
    return source_map, builder_review_raw, auditor_review_raw


def _binding(
    review: dict[str, object],
    *,
    role: str,
    file_sha256: str,
) -> dict[str, object]:
    return {
        "schema": auditor.REVIEW_BINDING_SCHEMA,
        "review_schema": review["schema"],
        "verdict": "PASS",
        "reviewer": review["reviewer"],
        "implementation_author": review["implementation_author"],
        "path": ROLE_PATH_BY_NAME[role],
        "file_sha256": file_sha256,
        "content_sha256": review["content_sha256"],
        "candidate": review["candidate"],
    }


def test_auditor_v7_candidate_and_predecessor_maps_rehash_exactly() -> None:
    for role, digest in AUDITOR_CANDIDATE_SHA256.items():
        assert _sha256(ROOT / ROLE_PATH_BY_NAME[role]) == digest
    assert _sha256(ROOT / BUILDER_REVIEW_PATH) == BUILDER_REVIEW_FILE_SHA256
    assert auditor.FROZEN_V7_PREDECESSOR_SHA256 == builder.FROZEN_PARENT_HASHES
    assert len(auditor.FROZEN_V7_PREDECESSOR_SHA256) == 55
    assert all(
        _sha256(ROOT / path) == digest
        for path, digest in auditor.FROZEN_V7_PREDECESSOR_SHA256.items()
    )
    assert all(
        _sha256(ROOT / path) == digest
        for path, digest in auditor.REVIEWED_V4_SOURCE_SHA256.items()
    )


def test_builder_and_auditor_accept_one_identical_dual_review_structure() -> None:
    source_map, builder_raw, auditor_raw = _source_map_and_records()
    builder_review = json.loads(builder_raw)
    auditor_review = json.loads(auditor_raw)
    core = {
        "schema": auditor.AUTHORIZATION_SCHEMA,
        "exact_build_authorized_after_independent_reviews": True,
        "builder_review": _binding(
            builder_review,
            role="builder_review",
            file_sha256=BUILDER_REVIEW_FILE_SHA256,
        ),
        "auditor_review": _binding(
            auditor_review,
            role="auditor_review",
            file_sha256=hashlib.sha256(auditor_raw).hexdigest(),
        ),
        "source_map": source_map,
    }
    authorization = {
        **core,
        "content_sha256": auditor.canonical_json_sha256(core),
    }
    from_builder = builder._validate_authorization_phase_one(
        authorization,
        authorization_file_sha256="a" * 64,
    )
    from_auditor = auditor._validate_authorization_phase_one(
        authorization,
        authorization_file_sha256="a" * 64,
    )
    assert from_builder.authorization_content_sha256 == from_auditor.authorization_content_sha256
    assert from_builder.source_map_sha256 == from_auditor.source_map_sha256
    assert from_builder.builder_review.reviewer == "/root"
    assert from_auditor.auditor_review.reviewer == "/synthetic/distinct_auditor_v7_reviewer"
    assert from_builder.builder_review.reviewer != from_builder.auditor_review.reviewer
    builder._validate_review_record(builder_raw, from_builder.builder_review)
    builder._validate_review_record(auditor_raw, from_builder.auditor_review)
    auditor._validate_review_record(builder_raw, from_auditor.builder_review)
    auditor._validate_review_record(auditor_raw, from_auditor.auditor_review)

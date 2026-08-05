"""Production-ineligible synthetic authority support for Builder V6 tests only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest.mock import patch

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v6 as builder


PRODUCTION_ELIGIBLE = False
Reader = Callable[..., bytes]


def _source(role: str, path: str, digest: str) -> dict[str, str]:
    return {"role": role, "path": path, "sha256": digest}


def _hashed(core: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(core))
    value.pop("content_sha256", None)
    return {**value, "content_sha256": builder.canonical_json_sha256(value)}


def _review_raw(
    *,
    kind: str,
    schema: str,
    reviewer: str,
    author: str,
    candidate: list[dict[str, str]],
) -> tuple[bytes, dict[str, Any]]:
    value = _hashed(
        {
            "schema": schema,
            "verdict": "PASS",
            "reviewer": reviewer,
            "implementation_author": author,
            "candidate": candidate,
            "authority": builder._expected_review_authority(kind),
        }
    )
    return builder.canonical_json_bytes(value) + b"\n", value


def valid_authorization() -> tuple[
    dict[str, Any], dict[str, bytes], dict[str, str]
]:
    """Return a structurally valid, non-production synthetic authority closure."""

    digest_by_role = {
        role: hashlib.sha256(f"synthetic:{role}".encode("ascii")).hexdigest()
        for role, _path in builder.AUTHORIZED_ROLE_PATHS
    }
    source_by_role = {
        role: _source(role, path, digest_by_role[role])
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    builder_candidate = [
        source_by_role[role] for role in builder.BUILDER_CANDIDATE_ROLES
    ]
    auditor_candidate = [
        source_by_role[role] for role in builder.AUDITOR_CANDIDATE_ROLES
    ]
    builder_raw, builder_review = _review_raw(
        kind="builder",
        schema=builder.BUILDER_REVIEW_SCHEMA,
        reviewer="/synthetic/builder_reviewer",
        author=builder.BUILDER_IMPLEMENTATION_AUTHOR,
        candidate=builder_candidate,
    )
    auditor_raw, auditor_review = _review_raw(
        kind="auditor",
        schema=builder.AUDITOR_REVIEW_SCHEMA,
        reviewer="/synthetic/auditor_reviewer",
        author=builder.AUDITOR_IMPLEMENTATION_AUTHOR,
        candidate=auditor_candidate,
    )
    digest_by_role["builder_review"] = hashlib.sha256(builder_raw).hexdigest()
    digest_by_role["auditor_review"] = hashlib.sha256(auditor_raw).hexdigest()
    source_by_role = {
        role: _source(role, path, digest_by_role[role])
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    builder_candidate = [
        source_by_role[role] for role in builder.BUILDER_CANDIDATE_ROLES
    ]
    auditor_candidate = [
        source_by_role[role] for role in builder.AUDITOR_CANDIDATE_ROLES
    ]

    def binding(
        *,
        review_role: str,
        schema: str,
        review: Mapping[str, Any],
        candidate: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "schema": builder.REVIEW_BINDING_SCHEMA,
            "review_schema": schema,
            "verdict": "PASS",
            "reviewer": review["reviewer"],
            "implementation_author": review["implementation_author"],
            "path": builder.ROLE_PATH_BY_NAME[review_role],
            "file_sha256": digest_by_role[review_role],
            "content_sha256": review["content_sha256"],
            "candidate": deepcopy(candidate),
        }

    authorization = _hashed(
        {
            "schema": builder.AUTHORIZATION_SCHEMA,
            "exact_build_authorized_after_independent_reviews": True,
            "builder_review": binding(
                review_role="builder_review",
                schema=builder.BUILDER_REVIEW_SCHEMA,
                review=builder_review,
                candidate=builder_candidate,
            ),
            "auditor_review": binding(
                review_role="auditor_review",
                schema=builder.AUDITOR_REVIEW_SCHEMA,
                review=auditor_review,
                candidate=auditor_candidate,
            ),
            "source_map": [
                deepcopy(source_by_role[role])
                for role, _path in builder.AUTHORIZED_ROLE_PATHS
            ],
        }
    )
    raw_by_role = {
        role: (
            builder_raw
            if role == "builder_review"
            else auditor_raw
            if role == "auditor_review"
            else f"synthetic reviewed source {role}\n".encode("ascii")
        )
        for role, _path in builder.AUTHORIZED_ROLE_PATHS
    }
    return authorization, raw_by_role, digest_by_role


def validate_phase_two_for_tests(
    phase_one: builder.PhaseOneAuthorizationV6,
    *,
    repository_root: Path,
    reader: Reader,
) -> tuple[builder.AcceptedAuthorizationV6, tuple[str, ...]]:
    """Mirror phase two with injected bytes, outside all production modules."""

    if PRODUCTION_ELIGIBLE:
        raise AssertionError("synthetic authority helper became production eligible")
    if type(phase_one) is not builder.PhaseOneAuthorizationV6:
        raise TypeError("test phase two requires a completed V6 phase-one capsule")
    embedded = builder._strict_canonical_json_object(
        phase_one.canonical_payload + b"\n",
        name="phase-one authorization capsule",
    )
    revalidated = builder._validate_authorization_phase_one(
        embedded,
        authorization_file_sha256=phase_one.authorization_file_sha256,
    )
    if revalidated != phase_one:
        raise PermissionError("phase-one authorization capsule was fabricated")
    payload_by_role: dict[str, bytes] = {}
    opened: list[str] = []
    for source in phase_one.sources:
        payload_by_role[source.role] = reader(
            repository_root=repository_root,
            path=(repository_root / source.path).absolute(),
            expected_sha256=source.sha256,
        )
        opened.append(source.role)
    builder._validate_review_record(
        payload_by_role["builder_review"], phase_one.builder_review
    )
    builder._validate_review_record(
        payload_by_role["auditor_review"], phase_one.auditor_review
    )
    return (
        builder.AcceptedAuthorizationV6(
            authorization_file_sha256=phase_one.authorization_file_sha256,
            authorization_content_sha256=phase_one.authorization_content_sha256,
            source_map_sha256=phase_one.source_map_sha256,
        ),
        tuple(opened),
    )


def write_prepared_scene_job_for_tests(
    job: builder.PreparedSceneJobV6,
    staging_root: Path,
) -> dict[str, Any]:
    """Run one fixed construction worker with synthetic test-only authority."""

    if PRODUCTION_ELIGIBLE:
        raise AssertionError("synthetic construction helper became production eligible")
    receipt = builder.AcceptedAuthorizationV6("a" * 64, "b" * 64, "c" * 64)
    with patch.object(builder, "_require_exact_authority", return_value=receipt):
        return builder._write_prepared_scene_job(
            job,
            str(staging_root),
            "a" * 64,
        )

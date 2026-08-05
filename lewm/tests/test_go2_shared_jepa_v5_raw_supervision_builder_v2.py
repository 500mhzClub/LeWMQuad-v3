"""Source-free tests for the raw-supervision builder V2 authority boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v1 as v1
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v2 as builder
from lewm.tests import test_go2_shared_jepa_v5_raw_supervision_builder_v1 as v1_tests


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V1 = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py": (
        "3bc1559776e2f8471bb6a7a1ddd8808b1f1224687dedf280fd2300820afe25ec"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v1.py": (
        "df5fd60b50ba852d44fd6fe0034c7e763fc08030875488be3850e774906ceeb3"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v1.py": (
        "15767446ba45851a7f5774560db8e8f6f87d831a51fde7585acffa028f3ba2e4"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v1_"
    "author_handoff_2026-07-13.md": (
        "9d9aee5f636069d8beef2362bcc43b9be0063207d9ffe17d9045f99e3c30d28c"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v1_"
    "independent_review.py": (
        "306b02e9ebaec7eb4a0649e65bff203582a9dba99a43d708c9adfd962d332104"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashed(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "content_sha256": builder.canonical_json_sha256(core)}


def _source(role: str, path: str, digest: str) -> dict[str, str]:
    return {"role": role, "path": path, "sha256": digest}


def _review_raw(
    *,
    kind: str,
    schema: str,
    reviewer: str,
    author: str,
    candidate: list[dict[str, str]],
) -> tuple[bytes, dict[str, Any]]:
    core = {
        "schema": schema,
        "verdict": "PASS",
        "reviewer": reviewer,
        "implementation_author": author,
        "candidate": candidate,
        "authority": builder._expected_review_authority(kind),
    }
    value = _hashed(core)
    return builder.canonical_json_bytes(value) + b"\n", value


def _valid_authorization() -> tuple[
    dict[str, Any], dict[str, bytes], dict[str, str]
]:
    digest_by_role = {
        role: hashlib.sha256(f"source:{role}".encode("ascii")).hexdigest()
        for role, _path in builder.AUTHORIZED_ROLE_PATHS
    }
    candidate_by_role = {
        role: _source(role, path, digest_by_role[role])
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    builder_candidate = [
        candidate_by_role[role] for role in builder.BUILDER_CANDIDATE_ROLES
    ]
    auditor_candidate = [
        candidate_by_role[role] for role in builder.AUDITOR_CANDIDATE_ROLES
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
    source_map = [
        _source(role, path, digest_by_role[role])
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    ]

    def binding(
        *,
        kind: str,
        schema: str,
        review_role: str,
        review: dict[str, Any],
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
            "candidate": candidate,
        }

    core = {
        "schema": builder.AUTHORIZATION_SCHEMA,
        "exact_build_authorized_after_independent_reviews": True,
        "builder_review": binding(
            kind="builder",
            schema=builder.BUILDER_REVIEW_SCHEMA,
            review_role="builder_review",
            review=builder_review,
            candidate=builder_candidate,
        ),
        "auditor_review": binding(
            kind="auditor",
            schema=builder.AUDITOR_REVIEW_SCHEMA,
            review_role="auditor_review",
            review=auditor_review,
            candidate=auditor_candidate,
        ),
        "source_map": source_map,
    }
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
    return _hashed(core), raw_by_role, digest_by_role


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(value)
    value.pop("content_sha256", None)
    return _hashed(value)


def _forbid_all_openers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("opened")
        raise AssertionError("metadata, source-map, or referenced-source opener reached")

    monkeypatch.setattr(builder, "_read_bound_regular_file", forbidden)
    monkeypatch.setattr(v1.plan_v5, "load_frozen_development_metadata", forbidden)
    monkeypatch.setattr(
        v1.plan_v5, "load_frozen_development_source_inventory", forbidden
    )
    monkeypatch.setattr(v1, "_read_exact_source", forbidden)
    return calls


def test_frozen_v1_and_block_reproducer_are_unchanged() -> None:
    assert {relative: _sha(ROOT / relative) for relative in FROZEN_V1} == FROZEN_V1


def test_absent_authorization_rejects_before_every_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _forbid_all_openers(monkeypatch)
    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", tmp_path / "absent.json")
    with pytest.raises(PermissionError, match="authorization is absent"):
        builder.execute_exact_build_v2(authorization_sha256="0" * 64, workers=1)
    assert calls == []


def test_malformed_authorization_json_opens_only_the_fixed_authority_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path = tmp_path / "authorization.json"
    authority_path.write_bytes(b"{not json}\n")
    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", authority_path)
    opened: list[Path] = []

    def fixed_authority_reader(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        del repository_root, expected_sha256
        opened.append(path)
        if path != authority_path:
            raise AssertionError("a non-authority target was opened")
        return b"{not json}\n"

    monkeypatch.setattr(builder, "_read_bound_regular_file", fixed_authority_reader)
    metadata_calls: list[str] = []
    monkeypatch.setattr(
        v1.plan_v5,
        "load_frozen_development_metadata",
        lambda *_args, **_kwargs: metadata_calls.append("metadata"),
    )
    with pytest.raises(builder.RawSupervisionBuildError, match="invalid JSON"):
        builder._require_exact_authority("1" * 64)
    assert opened == [authority_path]
    assert metadata_calls == []


def test_duplicate_authorization_json_key_opens_no_reviewed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path = tmp_path / "authorization.json"
    authority_path.write_bytes(b'{"schema":"one","schema":"two"}\n')
    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", authority_path)
    opened: list[Path] = []

    def authority_only(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        del repository_root, expected_sha256
        opened.append(path)
        if path != authority_path:
            raise AssertionError("a reviewed target was opened")
        return authority_path.read_bytes()

    monkeypatch.setattr(builder, "_read_bound_regular_file", authority_only)
    with pytest.raises(builder.RawSupervisionBuildError, match="duplicate key"):
        builder._require_exact_authority("2" * 64)
    assert opened == [authority_path]


Mutation = Callable[[dict[str, Any]], None]


def _duplicate_role(value: dict[str, Any]) -> None:
    value["source_map"][-1]["role"] = value["source_map"][0]["role"]


def _duplicate_path(value: dict[str, Any]) -> None:
    value["source_map"][-1]["path"] = value["source_map"][0]["path"]


def _missing_role(value: dict[str, Any]) -> None:
    value["source_map"].pop()


def _extra_role(value: dict[str, Any]) -> None:
    value["source_map"].append(
        _source("extra_role", "review/extra.json", "a" * 64)
    )


def _wrong_role(value: dict[str, Any]) -> None:
    value["source_map"][0]["role"] = "wrong_builder_source"


def _wrong_path(value: dict[str, Any]) -> None:
    value["source_map"][0]["path"] = "lewm/datasets/wrong_builder.py"


def _noncanonical_path(value: dict[str, Any]) -> None:
    value["source_map"][0]["path"] = "lewm/datasets/../wrong_builder.py"


def _wrong_binding(value: dict[str, Any]) -> None:
    value["builder_review"]["candidate"][0]["sha256"] = "b" * 64


def _wrong_review_path_binding(value: dict[str, Any]) -> None:
    value["builder_review"]["path"] = "docs/wrong_review.json"


def _malformed_source_entry(value: dict[str, Any]) -> None:
    value["source_map"][0]["extra"] = False


def _extra_top_level_field(value: dict[str, Any]) -> None:
    value["unexpected"] = False


def _source_map_not_a_list(value: dict[str, Any]) -> None:
    value["source_map"] = {"rows": value["source_map"]}


def _reordered_roles(value: dict[str, Any]) -> None:
    value["source_map"][0], value["source_map"][1] = (
        value["source_map"][1],
        value["source_map"][0],
    )


def _wrong_implementation_author(value: dict[str, Any]) -> None:
    value["builder_review"]["implementation_author"] = "/synthetic/wrong_author"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_duplicate_role, "roles are duplicated"),
        (_duplicate_path, "paths are duplicated"),
        (_missing_role, "source roles changed"),
        (_extra_role, "source roles changed"),
        (_wrong_role, "source roles changed"),
        (_wrong_path, "role-to-path mapping changed"),
        (_noncanonical_path, "path is noncanonical"),
        (_wrong_binding, "candidate binding changed"),
        (_wrong_review_path_binding, "review file binding changed"),
        (_malformed_source_entry, "source entry is malformed"),
        (_extra_top_level_field, "object fields changed"),
        (_source_map_not_a_list, "source map is absent"),
        (_reordered_roles, "source roles changed"),
        (_wrong_implementation_author, "implementation author changed"),
    ],
)
def test_phase_one_adversarial_authorities_reach_zero_openers(
    mutation: Mutation,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _raw_by_role, _digests = _valid_authorization()
    mutation(authority)
    authority = _rehash(authority)
    calls = _forbid_all_openers(monkeypatch)
    with pytest.raises((PermissionError, builder.RawSupervisionBuildError), match=match):
        builder._validate_authorization_payload(
            authority, authorization_file_sha256="c" * 64
        )
    assert calls == []


def test_wrong_authorization_content_hash_reaches_zero_openers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _raw_by_role, _digests = _valid_authorization()
    authority["content_sha256"] = "0" * 64
    calls = _forbid_all_openers(monkeypatch)
    with pytest.raises(builder.RawSupervisionBuildError, match="content hash changed"):
        builder._validate_authorization_payload(
            authority, authorization_file_sha256="d" * 64
        )
    assert calls == []


def test_v1_block_reproducer_passes_against_v2_with_zero_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = {
        "schema": builder.AUTHORIZATION_SCHEMA,
        "exact_build_authorized_after_independent_reviews": True,
        "builder_review": {},
        "auditor_review": {},
        "source_map": [
            {
                "role": "builder_source",
                "path": "arbitrary/referenced_frames.jsonl",
                "sha256": "1" * 64,
            }
        ],
    }
    opened: list[Path] = []

    def record_open(**kwargs: Any) -> bytes:
        opened.append(Path(kwargs["path"]))
        return b"arbitrary referenced payload\n"

    monkeypatch.setattr(builder, "_read_bound_regular_file", record_open)
    with pytest.raises(builder.RawSupervisionBuildError, match="source roles changed"):
        builder._validate_authorization_payload(
            _hashed(core), authorization_file_sha256="2" * 64
        )
    assert opened == []


def test_valid_phase_two_opens_exactly_nine_fixed_targets_after_phase_one(
    tmp_path: Path,
) -> None:
    authority, raw_by_role, digests = _valid_authorization()
    phase_one = builder._validate_authorization_phase_one(
        authority, authorization_file_sha256="e" * 64
    )
    role_by_path = {
        path: role for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    opened: list[str] = []

    def synthetic_reader(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        assert repository_root == tmp_path
        relative = path.relative_to(tmp_path).as_posix()
        role = role_by_path[relative]
        assert expected_sha256 == digests[role]
        opened.append(role)
        return raw_by_role[role]

    result = builder._validate_authorization_phase_two(
        phase_one,
        repository_root=tmp_path,
        reader=synthetic_reader,
        rehash_frozen_parents=False,
    )
    assert result == authority
    assert opened == [role for role, _path in builder.AUTHORIZED_ROLE_PATHS]


def test_phase_two_rejects_duplicate_review_json_keys(
    tmp_path: Path,
) -> None:
    authority, raw_by_role, digests = _valid_authorization()
    phase_one = builder._validate_authorization_phase_one(
        authority, authorization_file_sha256="f" * 64
    )
    role_by_path = {
        path: role for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    original = raw_by_role["builder_review"].decode("utf-8")
    raw_by_role["builder_review"] = original.replace(
        '"verdict":"PASS"', '"verdict":"PASS","verdict":"PASS"'
    ).encode("utf-8")

    def synthetic_reader(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        del expected_sha256
        role = role_by_path[path.relative_to(repository_root).as_posix()]
        return raw_by_role[role]

    with pytest.raises(builder.RawSupervisionBuildError, match="duplicate key"):
        builder._validate_authorization_phase_two(
            phase_one,
            repository_root=tmp_path,
            reader=synthetic_reader,
            rehash_frozen_parents=False,
        )


def test_phase_two_rejects_a_fabricated_phase_capsule_before_any_read(
    tmp_path: Path,
) -> None:
    authority, _raw_by_role, _digests = _valid_authorization()
    valid = builder._validate_authorization_phase_one(
        authority, authorization_file_sha256="a" * 64
    )
    first = replace(valid.sources[0], path="arbitrary/referenced_frames.jsonl")
    fabricated = replace(valid, sources=(first, *valid.sources[1:]))
    opened: list[str] = []

    def forbidden(**_kwargs: Any) -> bytes:
        opened.append("opened")
        raise AssertionError("fabricated capsule reached a reader")

    with pytest.raises(PermissionError, match="capsule was fabricated"):
        builder._validate_authorization_phase_two(
            fabricated,
            repository_root=tmp_path,
            reader=forbidden,
            rehash_frozen_parents=False,
        )
    assert opened == []


def test_synthetic_construction_is_the_frozen_v1_engine(tmp_path: Path) -> None:
    jobs, pairs = v1_tests._synthetic_inputs(2)
    v1_output = tmp_path / "v1"
    v2_output = tmp_path / "v2"
    first = v1.build_prepared_dataset_v1(
        jobs,
        pairs,
        output_directory=v1_output,
        workers=1,
        input_provenance={"schema": "synthetic"},
        access_ledger={"schema": "synthetic"},
    )
    second = builder.build_prepared_dataset_v1(
        jobs,
        pairs,
        output_directory=v2_output,
        workers=1,
        input_provenance={"schema": "synthetic"},
        access_ledger={"schema": "synthetic"},
    )
    assert first == second
    assert v1_tests._file_hashes(v1_output) == v1_tests._file_hashes(v2_output)
    assert builder.ARRAY_LAYOUT == v1.ARRAY_LAYOUT
    assert builder.MAX_WORKERS == 6


@pytest.mark.parametrize("workers", [0, 7])
def test_worker_bound_rejects_before_authority_or_metadata_open(
    workers: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _forbid_all_openers(monkeypatch)
    with pytest.raises(ValueError, match="workers"):
        builder.execute_exact_build_v2(
            authorization_sha256="0" * 64, workers=workers
        )
    assert calls == []


def test_every_v1_bridge_rejects_fabricated_state_before_any_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _forbid_all_openers(monkeypatch)
    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", tmp_path / "absent.json")
    assert "_WORKER_AUTHORIZATION" not in vars(builder)
    assert "__getattr__" not in vars(builder)
    assert "execute_exact_build_v1" not in vars(builder)
    bridge_calls = (
        lambda: builder._call_v1_load_exact_scene_job({}, (), {}, "3" * 64),
        lambda: builder._load_exact_scene_job_v2({}, (), {}, "3" * 64),
        lambda: builder._call_v1_revalidate_exact_scene_sources({}, "3" * 64),
        lambda: builder._revalidate_exact_scene_sources_v2({}, "3" * 64),
        lambda: builder._call_v1_load_parent_contracts("3" * 64),
        lambda: builder._load_parent_contracts_v2("3" * 64),
    )
    for call in bridge_calls:
        with pytest.raises(PermissionError, match="authorization is absent"):
            call()
    assert calls == []

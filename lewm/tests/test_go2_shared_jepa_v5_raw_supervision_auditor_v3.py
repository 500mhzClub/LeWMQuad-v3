"""Synthetic and adversarial tests for raw-supervision auditor V3.

The canonical dataset, source inventory, RGB, G2, and accelerator payloads are
never opened.  V1 artifacts are treated as frozen predecessor evidence.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v1 as v1
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v2 as auditor_v2
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v3 as auditor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v3 as builder_v3
from lewm.tests import test_go2_shared_jepa_v5_raw_supervision_auditor_v1 as v1_tests
from lewm.tests import (
    test_go2_shared_jepa_v5_raw_supervision_auditor_v1_independent_review as v1_review,
)
from scripts import audit_go2_shared_jepa_v5_raw_supervision_v3 as audit_cli


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V1 = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py": (
        "854d433084af4bda7dca1e39bed69bc76e9904546111e9289cbb4066660c798c"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v1.py": (
        "246a8de16a9645a0af8f0cf69e6241b16d68588d54ee9f8eb8b087519a9b908d"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v1.py": (
        "6dfe991e3f5abc7a5a7405ad1a9ad74382d05ba27e1beb5e6d087aed41351557"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v1_author_handoff_2026-07-13.md": (
        "7d693902bf4517bb19a87b6769af0c272403ba553daccb6e03d9cef88eec279d"
    ),
}
FROZEN_V1_BLOCK = {
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v1_independent_review.py": (
        "9684b14c3a87825a1b0d9f4f5bfd17c98c67f92c198818fc441aec0d8b6776fc"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v1_independent_review_2026-07-13.md": (
        "a61b64e337f5f6e9341db97665ea7a01818d9a74916f77d31cd9721453abdca8"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v1_independent_review_block_2026-07-13.json": (
        "c427b927f863e587c25403ac00b9f06170844b5a936b492e9c213696bf378f5b"
    ),
}
FROZEN_AUDITOR_V2 = {
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
V3_AMENDMENT = {
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v3_"
    "authorization_successor_amendment_2026-07-13.md": (
        "501062e2eba625cf4d7ab28810f2a629652c327c770366c07f3b788f3f6f8b2b"
    )
}
EXPECTED_V3_ROLE_PATHS = (
    (
        "builder_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v3.py",
    ),
    (
        "builder_cli",
        "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v3.py",
    ),
    (
        "builder_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v3.py",
    ),
    (
        "builder_handoff",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v3_"
        "author_handoff_2026-07-13.md",
    ),
    (
        "builder_review",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v3_"
        "independent_review_2026-07-13.json",
    ),
    (
        "auditor_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v3.py",
    ),
    (
        "auditor_cli",
        "scripts/audit_go2_shared_jepa_v5_raw_supervision_v3.py",
    ),
    (
        "auditor_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v3.py",
    ),
    (
        "auditor_review",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_"
        "independent_review_2026-07-13.json",
    ),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_hash(core: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(auditor.canonical_json_bytes(core))
    return {**normalized, "content_sha256": auditor.canonical_json_sha256(normalized)}


def _write_canonical(path: Path, value: Mapping[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = auditor.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return raw


def _rehashed_manifest(root: Path, mutate: Any) -> str:
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    mutate(manifest)
    return v1_review._write_manifest(root, manifest)


def _rebind_manifest_file(
    manifest: dict[str, Any], root: Path, relative: str
) -> None:
    raw = (root / relative).read_bytes()
    record = next(row for row in manifest["files"] if row["path"] == relative)
    record["byte_count"] = len(raw)
    record["file_sha256"] = hashlib.sha256(raw).hexdigest()


def _synthetic_fixture(tmp_path: Path):
    return v1_tests._synthetic_fixture(tmp_path)


def _authority_fixture(
    tmp_path: Path,
    *,
    duplicate_auditor_review_key: bool = False,
) -> tuple[Path, Path, dict[str, str], tuple[auditor.SourceBindingV3, ...]]:
    repository = tmp_path / "repository"
    source_rows: list[dict[str, str]] = []
    source_bytes: dict[str, bytes] = {}
    for role, relative in auditor.SOURCE_ROLE_PATHS:
        if role in {"builder_review", "auditor_review"}:
            continue
        payload = f"synthetic reviewed source: {role}\n".encode("ascii")
        source_bytes[role] = payload
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    provisional = {
        role: {
            "role": role,
            "path": relative,
            "sha256": hashlib.sha256(source_bytes[role]).hexdigest(),
        }
        for role, relative in auditor.SOURCE_ROLE_PATHS
        if role not in {"builder_review", "auditor_review"}
    }

    review_specs = (
        (
            "builder",
            "builder_review",
            auditor.BUILDER_REVIEW_SCHEMA,
            "builder-reviewer",
            auditor.BUILDER_IMPLEMENTATION_AUTHOR,
            auditor.BUILDER_CANDIDATE_ROLES,
        ),
        (
            "auditor",
            "auditor_review",
            auditor.AUDITOR_REVIEW_SCHEMA,
            "auditor-reviewer",
            auditor.AUDITOR_IMPLEMENTATION_AUTHOR,
            auditor.AUDITOR_CANDIDATE_ROLES,
        ),
    )
    review_content: dict[str, str] = {}
    review_raw: dict[str, bytes] = {}
    for kind, role, schema, reviewer, author, candidate_roles in review_specs:
        candidate = [provisional[name] for name in candidate_roles]
        record = _with_hash(
            {
                "schema": schema,
                "verdict": "PASS",
                "reviewer": reviewer,
                "implementation_author": author,
                "candidate": candidate,
                "authority": auditor._expected_review_authority_v3(kind),
            }
        )
        raw = auditor.canonical_json_bytes(record) + b"\n"
        if role == "auditor_review" and duplicate_auditor_review_key:
            raw = b'{"verdict":"PASS",' + raw[1:]
        relative = auditor.SOURCE_PATH_BY_ROLE[role]
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        review_raw[role] = raw
        review_content[role] = record["content_sha256"]

    for role, relative in auditor.SOURCE_ROLE_PATHS:
        raw = review_raw[role] if role in review_raw else source_bytes[role]
        source_rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    by_role = {row["role"]: row for row in source_rows}

    def binding(
        kind: str,
        review_role: str,
        schema: str,
        reviewer: str,
        author: str,
        candidate_roles: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "schema": auditor.REVIEW_BINDING_SCHEMA,
            "review_schema": schema,
            "verdict": "PASS",
            "reviewer": reviewer,
            "implementation_author": author,
            "path": by_role[review_role]["path"],
            "file_sha256": by_role[review_role]["sha256"],
            "content_sha256": review_content[review_role],
            "candidate": [by_role[role] for role in candidate_roles],
        }

    authorization = _with_hash(
        {
            "schema": auditor.AUTHORIZATION_SCHEMA,
            "exact_build_authorized_after_independent_reviews": True,
            "builder_review": binding(
                "builder",
                "builder_review",
                auditor.BUILDER_REVIEW_SCHEMA,
                "builder-reviewer",
                auditor.BUILDER_IMPLEMENTATION_AUTHOR,
                auditor.BUILDER_CANDIDATE_ROLES,
            ),
            "auditor_review": binding(
                "auditor",
                "auditor_review",
                auditor.AUDITOR_REVIEW_SCHEMA,
                "auditor-reviewer",
                auditor.AUDITOR_IMPLEMENTATION_AUTHOR,
                auditor.AUDITOR_CANDIDATE_ROLES,
            ),
            "source_map": source_rows,
        }
    )
    authorization_path = repository / "docs/authorization.json"
    raw = _write_canonical(authorization_path, authorization)
    provenance = {
        "authorization_file_sha256": hashlib.sha256(raw).hexdigest(),
        "authorization_content_sha256": authorization["content_sha256"],
        "authorization_source_map_sha256": auditor.canonical_json_sha256(
            source_rows
        ),
    }
    sources = tuple(auditor._source_binding_v3(row) for row in source_rows)
    return repository, authorization_path, provenance, sources


def test_v1_candidate_and_block_artifacts_remain_frozen() -> None:
    expected = {**FROZEN_V1, **FROZEN_V1_BLOCK}
    assert {relative: _sha(ROOT / relative) for relative in expected} == expected
    block = json.loads(
        (
            ROOT
            / "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v1_"
            "independent_review_block_2026-07-13.json"
        ).read_text(encoding="ascii")
    )
    assert block["verdict"] == "BLOCK"
    assert block["content_sha256"] == (
        "4a8235ed6368f665cc17420dd93a810c7bc7b13963ac7ce79c278cf3bb8a6915"
    )


def test_v2_base_and_v3_amendment_remain_frozen() -> None:
    expected = {**FROZEN_AUDITOR_V2, **V3_AMENDMENT}
    assert {relative: _sha(ROOT / relative) for relative in expected} == expected


def test_contract_matches_amendment_and_builder_v3_without_production_import() -> None:
    assert auditor.AUTHORIZATION_SCHEMA == (
        "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v3"
    )
    assert auditor.REVIEW_BINDING_SCHEMA == (
        "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v3"
    )
    assert auditor.BUILDER_REVIEW_SCHEMA == (
        "lewm_go2_shared_jepa_v5_raw_supervision_builder_v3_independent_review_v1"
    )
    assert auditor.AUDITOR_REVIEW_SCHEMA == (
        "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_independent_review_v1"
    )
    assert auditor.SOURCE_ROLE_PATHS == EXPECTED_V3_ROLE_PATHS
    assert auditor.AUTHORIZATION_SCHEMA == builder_v3.AUTHORIZATION_SCHEMA
    assert auditor.REVIEW_BINDING_SCHEMA == builder_v3.REVIEW_BINDING_SCHEMA
    assert auditor.BUILDER_REVIEW_SCHEMA == builder_v3.BUILDER_REVIEW_SCHEMA
    assert auditor.AUDITOR_REVIEW_SCHEMA == builder_v3.AUDITOR_REVIEW_SCHEMA
    assert (
        auditor.BUILDER_IMPLEMENTATION_AUTHOR
        == builder_v3.BUILDER_IMPLEMENTATION_AUTHOR
    )
    assert (
        auditor.AUDITOR_IMPLEMENTATION_AUTHOR
        == builder_v3.AUDITOR_IMPLEMENTATION_AUTHOR
    )
    assert auditor.SOURCE_ROLE_PATHS == builder_v3.AUTHORIZED_ROLE_PATHS
    assert auditor.AUTHORIZATION_FIELDS == builder_v3.AUTHORIZATION_FIELDS
    assert auditor.REVIEW_BINDING_FIELDS == builder_v3.REVIEW_BINDING_FIELDS
    assert auditor.REVIEW_RECORD_FIELDS == builder_v3.REVIEW_RECORD_FIELDS
    assert auditor.BUILD_AUTHORIZATION_PATH == builder_v3.AUTHORIZATION_PATH

    tree = ast.parse(
        (ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v3.py")
        .read_text(encoding="ascii")
    )
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert all("raw_supervision_builder" not in name for name in imports)


def test_synthetic_callback_path_passes_and_has_no_exact_switch(tmp_path: Path) -> None:
    root, digest, inputs, replay = _synthetic_fixture(tmp_path)
    assert "exact" not in inspect.signature(auditor.audit_dataset_v3).parameters
    assert "sample_recomputer" not in inspect.signature(
        auditor.audit_exact_dataset_v3
    ).parameters
    result = auditor.audit_dataset_v3(
        root,
        expected_manifest_file_sha256=digest,
        inputs=inputs,
        sample_recomputer=lambda *_args: replay,
        workers=1,
    )
    assert result["verdict"] == "PASS"
    assert result["schema"] == auditor.SYNTHETIC_AUDIT_SCHEMA
    assert result["audit_scope"] == "synthetic_non_authoritative_callback"
    assert result["observed_population"]["pair_count"] == 1


def test_callback_path_cannot_be_promoted_to_exact(tmp_path: Path) -> None:
    root, _digest, inputs, replay = _synthetic_fixture(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    manifest["pair_counts"] = dict(auditor.FROZEN_PAIR_COUNTS)
    manifest["unique_endpoint_counts"] = dict(
        auditor.FROZEN_UNIQUE_ENDPOINT_COUNTS
    )
    manifest["endpoint_instance_count"] = auditor.FROZEN_ENDPOINT_REFERENCE_COUNT
    manifest["scene_shard_count"] = auditor.FROZEN_SCENE_SHARD_COUNT
    digest = v1_review._write_manifest(root, manifest)
    callback_called = False

    def callback(*_args: object):
        nonlocal callback_called
        callback_called = True
        return replay

    with pytest.raises(TypeError, match="exact"):
        auditor.audit_dataset_v3(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=callback,
            workers=1,
            exact=True,  # type: ignore[call-arg]
        )
    assert callback_called is False
    with pytest.raises(auditor.RawSupervisionAuditError, match="declared pair counts"):
        auditor._preflight_dataset_v3(
            root,
            expected_manifest_file_sha256=digest,
            exact=True,
        )


@pytest.mark.parametrize("target", ("manifest", "array"))
def test_every_dataset_leaf_rejects_hard_link_alias(
    tmp_path: Path, target: str
) -> None:
    root, digest, inputs, replay = _synthetic_fixture(tmp_path)
    path = (
        root / "manifest.json"
        if target == "manifest"
        else next((root / "shards").glob("*/camera_origin_body_m.f4"))
    )
    os.link(path, tmp_path / f"external-{target}.alias")
    with pytest.raises(PermissionError, match="alias|link"):
        auditor.audit_dataset_v3(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


@pytest.mark.parametrize(
    ("field_path", "value"),
    (
        (("pair_index", "row_count"), 1.0),
        (("endpoint_index", "row_count"), True),
        (("shards", 0, "endpoint_count"), 1.0),
        (("scene_shard_count",), True),
        (("array_layout", 0, "trailing_shape", 0), 3.0),
        (("files", 0, "byte_count"), False),
        (("access_ledger", "rgb_byte_opens"), 0.0),
    ),
)
def test_manifest_cardinalities_reject_bool_and_float(
    tmp_path: Path, field_path: tuple[Any, ...], value: object
) -> None:
    root, _digest, inputs, replay = _synthetic_fixture(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        target: Any = manifest
        for component in field_path[:-1]:
            target = target[component]
        target[field_path[-1]] = value

    digest = _rehashed_manifest(root, mutate)
    with pytest.raises(auditor.RawSupervisionAuditError, match="integer|count|shape"):
        auditor.audit_dataset_v3(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_pair_row_integer_rejects_float_before_plan_equality(tmp_path: Path) -> None:
    root, _digest, inputs, replay = _synthetic_fixture(tmp_path)
    pair_path = root / "pairs.jsonl"
    row = json.loads(pair_path.read_text(encoding="ascii"))
    row["global_row"] = 0.0
    core = dict(row)
    core.pop("content_sha256")
    row["content_sha256"] = auditor.canonical_json_sha256(core)
    pair_path.write_bytes(auditor.canonical_json_bytes(row) + b"\n")
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    _rebind_manifest_file(manifest, root, "pairs.jsonl")
    manifest["pair_index"]["file_sha256"] = hashlib.sha256(
        pair_path.read_bytes()
    ).hexdigest()
    manifest["ordered_pair_sha256"] = auditor.canonical_json_sha256(
        [row["content_sha256"]]
    )
    digest = v1_review._write_manifest(root, manifest)
    with pytest.raises(auditor.RawSupervisionAuditError, match="exact integer"):
        auditor.audit_dataset_v3(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_shard_local_shape_rejects_float(tmp_path: Path) -> None:
    root, _digest, inputs, replay = _synthetic_fixture(tmp_path)
    shard_path = next((root / "shards").glob("*/shard.json"))
    shard = json.loads(shard_path.read_text(encoding="ascii"))
    array_record = next(row for row in shard["files"] if row["path"].endswith(".f4"))
    array_record["shape"][0] = 1.0
    shard = _with_hash({key: value for key, value in shard.items() if key != "content_sha256"})
    _write_canonical(shard_path, shard)
    relative = str(shard_path.relative_to(root))
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    _rebind_manifest_file(manifest, root, relative)
    shard_record = next(row for row in manifest["shards"] if row["path"] == relative)
    shard_record["content_sha256"] = shard["content_sha256"]
    digest = v1_review._write_manifest(root, manifest)
    with pytest.raises(auditor.RawSupervisionAuditError, match="shape.*integer"):
        auditor.audit_dataset_v3(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_shard_index_row_rejects_boolean(tmp_path: Path) -> None:
    root, _digest, inputs, replay = _synthetic_fixture(tmp_path)
    index_path = next((root / "shards").glob("*/index.jsonl"))
    row = json.loads(index_path.read_text(encoding="ascii"))
    row["shard_row"] = False
    core = dict(row)
    core.pop("content_sha256")
    row["content_sha256"] = auditor.canonical_json_sha256(core)
    index_path.write_bytes(auditor.canonical_json_bytes(row) + b"\n")
    shard_path = index_path.parent / "shard.json"
    shard = json.loads(shard_path.read_text(encoding="ascii"))
    local = next(record for record in shard["files"] if record["path"] == "index.jsonl")
    local["byte_count"] = len(index_path.read_bytes())
    local["file_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    shard = _with_hash({key: value for key, value in shard.items() if key != "content_sha256"})
    _write_canonical(shard_path, shard)
    index_relative = str(index_path.relative_to(root))
    shard_relative = str(shard_path.relative_to(root))
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    _rebind_manifest_file(manifest, root, index_relative)
    _rebind_manifest_file(manifest, root, shard_relative)
    shard_record = next(
        record for record in manifest["shards"] if record["path"] == shard_relative
    )
    shard_record["content_sha256"] = shard["content_sha256"]
    digest = v1_review._write_manifest(root, manifest)
    with pytest.raises(auditor.RawSupervisionAuditError, match="exact integer"):
        auditor.audit_dataset_v3(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_workers_are_strict_integers_and_cli_paths_are_fixed(tmp_path: Path) -> None:
    for workers in (False, 1.0, 0, 7):
        with pytest.raises(ValueError, match="workers"):
            auditor.audit_exact_dataset_v3(
                auditor.ROOT,
                auditor.CANONICAL_DATASET,
                expected_manifest_file_sha256="a" * 64,
                expected_authorization_file_sha256="b" * 64,
                workers=workers,  # type: ignore[arg-type]
            )
    parsed = audit_cli._parse_args(
        [
            "--manifest-sha256",
            "a" * 64,
            "--authorization-sha256",
            "b" * 64,
            "--workers",
            "6",
        ]
    )
    assert parsed.workers == 6
    assert parsed.authorization_sha256 == "b" * 64
    for option in ("--dataset", "--output", "--report", "--repo-root"):
        with pytest.raises(SystemExit):
            audit_cli._parse_args(
                [
                    "--manifest-sha256",
                    "a" * 64,
                    "--authorization-sha256",
                    "b" * 64,
                    option,
                    str(tmp_path),
                ]
            )


@pytest.mark.parametrize("legacy_component", ("paths", "reviews"))
def test_v2_paths_and_reviews_cannot_authorize_v3_before_target_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_component: str,
) -> None:
    repository, _authorization_path, provenance, _sources = _authority_fixture(
        tmp_path
    )
    payload = v1._decode_json(
        (repository / "docs/authorization.json").read_bytes(),
        name="synthetic V3 authorization",
    )
    changed = json.loads(json.dumps(payload))
    changed.pop("content_sha256")
    if legacy_component == "paths":
        v2_paths = dict(auditor_v2.SOURCE_ROLE_PATHS)
        for row in changed["source_map"]:
            row["path"] = v2_paths[row["role"]]
        for kind, review_role in (
            ("builder_review", "builder_review"),
            ("auditor_review", "auditor_review"),
        ):
            changed[kind]["path"] = v2_paths[review_role]
            for row in changed[kind]["candidate"]:
                row["path"] = v2_paths[row["role"]]
    else:
        changed["builder_review"]["schema"] = auditor_v2.REVIEW_BINDING_SCHEMA
        changed["builder_review"]["review_schema"] = (
            auditor_v2.BUILDER_REVIEW_SCHEMA
        )
        changed["auditor_review"]["schema"] = auditor_v2.REVIEW_BINDING_SCHEMA
        changed["auditor_review"]["review_schema"] = (
            auditor_v2.AUDITOR_REVIEW_SCHEMA
        )
    legacy = _with_hash(changed)
    opened: list[Path] = []

    def forbidden(path: Path, *_args: object, **_kwargs: object) -> bytes:
        opened.append(Path(path))
        raise AssertionError("legacy V2 authority reached a target opener")

    monkeypatch.setattr(v1, "_read_absolute_bound_payload", forbidden)
    with pytest.raises(
        (PermissionError, auditor.RawSupervisionAuditError),
        match="mapping changed|not a bound PASS",
    ):
        auditor._validate_authorization_phase_one_v3(
            legacy,
            authorization_file_sha256=provenance["authorization_file_sha256"],
            authorization_content_sha256=legacy["content_sha256"],
            authorization_source_map_sha256=auditor.canonical_json_sha256(
                legacy["source_map"]
            ),
        )
    assert opened == []


def test_phase_one_is_zero_open_and_phase_two_opens_exactly_nine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _authorization_path, provenance, _sources = _authority_fixture(
        tmp_path
    )
    raw = (repository / "docs/authorization.json").read_bytes()
    payload = v1._decode_json(raw, name="synthetic authorization")
    phase_one = auditor._validate_authorization_phase_one_v3(
        payload,
        authorization_file_sha256=provenance["authorization_file_sha256"],
        authorization_content_sha256=provenance["authorization_content_sha256"],
        authorization_source_map_sha256=provenance[
            "authorization_source_map_sha256"
        ],
    )
    opened: list[Path] = []
    original = v1._read_absolute_bound_payload

    def record(path: Path, *args: object, **kwargs: object) -> bytes:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(v1, "_read_absolute_bound_payload", record)
    auditor._validate_authorization_phase_two_v3(
        phase_one, repository_root=repository
    )
    assert opened == [repository / path for _role, path in auditor.SOURCE_ROLE_PATHS]


def test_fabricated_phase_capsule_rejects_before_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _authorization_path, provenance, _sources = _authority_fixture(
        tmp_path
    )
    payload = v1._decode_json(
        (repository / "docs/authorization.json").read_bytes(),
        name="synthetic authorization",
    )
    phase_one = auditor._validate_authorization_phase_one_v3(
        payload,
        authorization_file_sha256=provenance["authorization_file_sha256"],
        authorization_content_sha256=provenance["authorization_content_sha256"],
        authorization_source_map_sha256=provenance[
            "authorization_source_map_sha256"
        ],
    )
    changed = replace(
        phase_one.sources[0], path="arbitrary/referenced_frames.jsonl"
    )
    fabricated = replace(phase_one, sources=(changed, *phase_one.sources[1:]))
    opened: list[Path] = []

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        opened.append(Path("opened"))
        raise AssertionError("fabricated capsule reached target reader")

    monkeypatch.setattr(v1, "_read_absolute_bound_payload", forbidden)
    with pytest.raises(PermissionError, match="capsule was fabricated"):
        auditor._validate_authorization_phase_two_v3(
            fabricated, repository_root=repository
        )
    assert opened == []


@pytest.mark.parametrize("authority_kind", ("absent", "malformed"))
def test_exact_api_rejects_authority_before_dataset_or_metadata_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_kind: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    authorization_path = repository / "docs/authorization.json"
    authorization_path.parent.mkdir()
    if authority_kind == "malformed":
        invalid = _with_hash(
            {
                "schema": auditor.AUTHORIZATION_SCHEMA,
                "exact_build_authorized_after_independent_reviews": True,
                "builder_review": {},
                "auditor_review": {},
                "source_map": [],
            }
        )
        raw = _write_canonical(authorization_path, invalid)
        expected_authorization_sha256 = hashlib.sha256(raw).hexdigest()
    else:
        expected_authorization_sha256 = "a" * 64
    dataset = repository / "canonical_dataset"
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def call(*_args: object, **_kwargs: object) -> Any:
            forbidden_calls.append(name)
            raise AssertionError(f"bad authority reached {name}")

        return call

    monkeypatch.setattr(auditor, "ROOT", repository)
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(auditor, "_read_manifest_bound_v3", forbidden("manifest"))
    monkeypatch.setattr(
        auditor.plan_v5,
        "load_frozen_development_metadata",
        forbidden("metadata"),
    )
    monkeypatch.setattr(
        auditor.plan_v5,
        "load_frozen_development_source_inventory",
        forbidden("source inventory"),
    )
    monkeypatch.setattr(
        v1, "_hash_complete_source_inventory", forbidden("development source")
    )
    with pytest.raises((PermissionError, auditor.RawSupervisionAuditError, FileNotFoundError)):
        auditor.audit_exact_dataset_v3(
            repository,
            dataset,
            expected_manifest_file_sha256="b" * 64,
            expected_authorization_file_sha256=expected_authorization_sha256,
            workers=1,
        )
    assert forbidden_calls == []


def test_incomplete_authority_opens_only_fixed_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    selected = repository / "arbitrary/referenced_frames.jsonl"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"caller-selected payload\n")
    source_map = [
        {
            "role": "builder_source",
            "path": str(selected.relative_to(repository)),
            "sha256": _sha(selected),
        }
    ]
    authorization = _with_hash(
        {
            "schema": auditor.AUTHORIZATION_SCHEMA,
            "exact_build_authorized_after_independent_reviews": True,
            "builder_review": {},
            "auditor_review": {},
            "source_map": source_map,
        }
    )
    authorization_path = repository / "docs/authorization.json"
    raw = _write_canonical(authorization_path, authorization)
    provenance = {
        "authorization_file_sha256": hashlib.sha256(raw).hexdigest(),
        "authorization_content_sha256": authorization["content_sha256"],
        "authorization_source_map_sha256": auditor.canonical_json_sha256(
            source_map
        ),
    }
    opened: list[Path] = []
    original = v1._read_absolute_bound_payload

    def record(path: Path, *args: object, **kwargs: object) -> bytes:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(auditor, "ROOT", repository)
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(v1, "_read_absolute_bound_payload", record)
    with pytest.raises(auditor.RawSupervisionAuditError, match="incomplete"):
        auditor._validate_exact_authorization_v3(provenance)
    assert opened == [authorization_path]


def test_machine_review_duplicate_json_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, authorization_path, provenance, _sources = _authority_fixture(
        tmp_path, duplicate_auditor_review_key=True
    )
    monkeypatch.setattr(auditor, "ROOT", repository)
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", authorization_path)
    with pytest.raises(auditor.RawSupervisionAuditError, match="duplicate JSON key"):
        auditor._validate_exact_authorization_v3(provenance)


def test_report_leaf_names_and_schema_are_additive_v3() -> None:
    assert auditor.CANONICAL_AUDIT_REPORT.name.endswith(".audit_v3.json")
    assert auditor.CANONICAL_AUDIT_FAILURE.name.endswith(".audit_v3.failed.json")
    assert auditor.AUDIT_SCHEMA == (
        "lewm_go2_shared_jepa_v5_raw_supervision_audit_v3"
    )
    assert auditor.SYNTHETIC_AUDIT_SCHEMA != auditor.AUDIT_SCHEMA


def test_v3_exact_failure_is_terminal_and_retry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "development_raw_supervision_v1"
    report = tmp_path / "development_raw_supervision_v1.audit_v3.json"
    failure = tmp_path / "development_raw_supervision_v1.audit_v3.failed.json"
    calls = 0

    def fail(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise auditor.RawSupervisionAuditError("synthetic terminal failure")

    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_REPORT", report)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_FAILURE", failure)
    monkeypatch.setattr(auditor, "audit_exact_dataset_v3", fail)
    with pytest.raises(auditor.RawSupervisionAuditError, match="terminal failure"):
        auditor.execute_exact_audit_v3(
            expected_manifest_file_sha256="a" * 64,
            expected_authorization_file_sha256="b" * 64,
            workers=1,
        )
    receipt = json.loads(failure.read_text(encoding="ascii"))
    assert receipt["schema"] == auditor.AUDIT_FAILURE_SCHEMA
    assert receipt["retry_authorized"] is False
    assert not report.exists()
    with pytest.raises(FileExistsError, match="immutable audit leaf"):
        auditor.execute_exact_audit_v3(
            expected_manifest_file_sha256="a" * 64,
            expected_authorization_file_sha256="b" * 64,
            workers=1,
        )
    assert calls == 1

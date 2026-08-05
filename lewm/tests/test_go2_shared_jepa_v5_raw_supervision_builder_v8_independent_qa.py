"""Independent CPU-only adversarial review for Raw Supervision Builder V8."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v8 as builder


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/raw_v8_builder_reviewer"
AMENDMENTS = {
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v8_"
    "terminal_quiet_successor_amendment_2026-07-13.md": (
        "054de82d8648cd6be7edff01b82d549ec916700ebffad51698d4c2041edc6c88"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_v8_existing_agent_identity_"
    "rebinding_amendment_2026-07-13.md": (
        "392745c80ca2c6e7a103cca4a55c3614cd2c988de9a379fba950b0087df41698"
    ),
}
CANDIDATE = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v8.py": (
        "f45533354c8b45b88f8eadb2126ec5eaf96fe1f57c21a9bfcd95a8855bfaaa35"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v8.py": (
        "f6471f1fa0ca7a13976f752a41ee9ddacbc76636e4d5fb0eee1ebf75bdaee72d"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v8.py": (
        "fc1f0cf3fc18bdbd1393be6a514bc04459f943f39b438ced78ebee30e7c57d9a"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v8_"
    "author_handoff_2026-07-13.md": (
        "9f4898e3620ac87c9a0145be103c4fdf397f727fe37d9f6ca306a0f50916156b"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _hashed(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(core))
    payload.pop("content_sha256", None)
    payload["content_sha256"] = builder.canonical_json_sha256(payload)
    return payload


def _source(role: str, path: str, digest: str) -> dict[str, str]:
    return {"role": role, "path": path, "sha256": digest}


def _synthetic_authority() -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a complete synthetic closure without production paths or payloads."""

    raw_by_role = {
        role: f"independent synthetic {role}\n".encode("ascii")
        for role, _path in builder.AUTHORIZED_ROLE_PATHS
        if role not in {"builder_review", "auditor_review"}
    }
    digest_by_role = {
        role: hashlib.sha256(raw).hexdigest() for role, raw in raw_by_role.items()
    }
    provisional = {
        role: _source(role, path, digest_by_role.get(role, "0" * 64))
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    }

    review_value_by_kind: dict[str, dict[str, Any]] = {}
    for kind, schema, reviewer, author, roles in (
        (
            "builder",
            builder.BUILDER_REVIEW_SCHEMA,
            REVIEWER,
            builder.BUILDER_IMPLEMENTATION_AUTHOR,
            builder.BUILDER_CANDIDATE_ROLES,
        ),
        (
            "auditor",
            builder.AUDITOR_REVIEW_SCHEMA,
            "/root/raw_v8_auditor_reviewer_synthetic",
            builder.AUDITOR_IMPLEMENTATION_AUTHOR,
            builder.AUDITOR_CANDIDATE_ROLES,
        ),
    ):
        review = _hashed(
            {
                "schema": schema,
                "verdict": "PASS",
                "reviewer": reviewer,
                "implementation_author": author,
                "candidate": [deepcopy(provisional[role]) for role in roles],
                "authority": builder._expected_review_authority(kind),
            }
        )
        review_role = f"{kind}_review"
        raw = builder.canonical_json_bytes(review) + b"\n"
        raw_by_role[review_role] = raw
        digest_by_role[review_role] = hashlib.sha256(raw).hexdigest()
        review_value_by_kind[kind] = review

    source_by_role = {
        role: _source(role, path, digest_by_role[role])
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    }

    def binding(kind: str, roles: tuple[str, ...]) -> dict[str, Any]:
        review_role = f"{kind}_review"
        review = review_value_by_kind[kind]
        schema = (
            builder.BUILDER_REVIEW_SCHEMA
            if kind == "builder"
            else builder.AUDITOR_REVIEW_SCHEMA
        )
        return {
            "schema": builder.REVIEW_BINDING_SCHEMA,
            "review_schema": schema,
            "verdict": "PASS",
            "reviewer": review["reviewer"],
            "implementation_author": review["implementation_author"],
            "path": builder.ROLE_PATH_BY_NAME[review_role],
            "file_sha256": digest_by_role[review_role],
            "content_sha256": review["content_sha256"],
            "candidate": [deepcopy(source_by_role[role]) for role in roles],
        }

    authority = _hashed(
        {
            "schema": builder.AUTHORIZATION_SCHEMA,
            "exact_build_authorized_after_independent_reviews": True,
            "builder_review": binding("builder", builder.BUILDER_CANDIDATE_ROLES),
            "auditor_review": binding("auditor", builder.AUDITOR_CANDIDATE_ROLES),
            "source_map": [
                deepcopy(source_by_role[role])
                for role, _path in builder.AUTHORIZED_ROLE_PATHS
            ],
        }
    )
    return authority, raw_by_role


def _top_level_key(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _normalized_module(path: Path, *, from_v7: bool) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    if from_v7:
        source = source.replace("V7", "V8").replace("v7", "v8")
    return ast.parse(source)


def _open_transaction(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, dict[str, Path]]:
    source_parent = root / "sources"
    publication_parent = root / "publication"
    source_parent.mkdir(parents=True)
    publication_parent.mkdir()
    source = source_parent / "source.bin"
    source.write_bytes(b"independent frozen source\n")
    staging_name = ".dataset.independent-review-staging"
    staging = publication_parent / staging_name
    staging.mkdir(mode=0o700)
    payload = staging / "payload.bin"
    payload.write_bytes(b"independent staged payload\n")
    manifest = builder._with_content_sha256(
        {"schema": "independent_builder_v8_transaction_probe_v1"}
    )
    (staging / "manifest.json").write_bytes(
        builder.canonical_json_bytes(manifest) + b"\n"
    )
    retained = builder._open_publication_parent(publication_parent)
    staging_identity = builder._named_directory_identity(
        retained.parent_fd, staging_name
    )
    monkeypatch.setattr(
        builder,
        "_exact_publication_source_hashes",
        lambda _context: {source: _sha256(source)},
    )
    transaction = builder._ClosedPublicationTransaction(
        context=object(),  # type: ignore[arg-type]
        retained=retained,
        staging=staging,
        staging_name=staging_name,
        staging_identity=staging_identity,
        destination=publication_parent / "dataset",
        expected_files=(
            {
                "path": "payload.bin",
                "byte_count": payload.stat().st_size,
                "file_sha256": _sha256(payload),
            },
        ),
        manifest=manifest,
    )
    return transaction, retained, {
        "source": source,
        "destination": publication_parent / "dataset",
    }


def _publish_to_final_boundary(transaction: Any, retained: Any) -> None:
    transaction.validate_before_rename()
    transaction.rename_owned()
    retained.refresh_after_owned_mutation()
    transaction.validate_after_rename()
    os.fsync(retained.parent_fd)


def test_frozen_candidate_rehash_and_complete_predecessor_closure() -> None:
    assert {path: _sha256(ROOT / path) for path in AMENDMENTS} == AMENDMENTS
    assert {path: _sha256(ROOT / path) for path in CANDIDATE} == CANDIDATE
    assert len(builder.FROZEN_PARENT_HASHES) == 69
    assert len(builder.REVIEWED_V4_SOURCES) == 9
    assert all(
        _sha256(ROOT / path) == digest
        for path, digest in builder.FROZEN_PARENT_HASHES.items()
    )
    assert all(
        _sha256(ROOT / path) == digest
        for path, digest in builder.REVIEWED_V4_SOURCES.items()
    )


def test_all_non_authority_v8_top_level_ast_is_v7_equivalent() -> None:
    v7 = _normalized_module(
        ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v7.py",
        from_v7=True,
    )
    v8 = _normalized_module(Path(builder.__file__), from_v7=False)
    permitted = {
        "AMENDMENT_AUTHOR",
        "AUDITOR_IMPLEMENTATION_AUTHOR",
        "BUILDER_IMPLEMENTATION_AUTHOR",
        "FROZEN_PARENT_HASHES",
        "REVIEW_AUTHORITY_FALSE_FIELDS",
        "V8_IDENTITY_REBINDING_AMENDMENT_PATH",
        "V8_SUCCESSOR_AMENDMENT_PATH",
        "_review_binding",
    }

    def retained(tree: ast.Module) -> list[str]:
        return [
            ast.dump(node, include_attributes=False)
            for node in tree.body
            if _top_level_key(node) not in permitted
        ]

    assert retained(v8) == retained(v7)
    assert set(builder._expected_review_authority("builder")) == {
        "builder_source_approved",
        *builder.REVIEW_AUTHORITY_FALSE_FIELDS,
    }
    assert builder._expected_review_authority("builder") == {
        "builder_source_approved": True,
        **{field: False for field in builder.REVIEW_AUTHORITY_FALSE_FIELDS},
    }
    v7_cli = (
        ROOT / "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v7.py"
    ).read_text(encoding="utf-8")
    v8_cli = (
        ROOT / "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v8.py"
    ).read_text(encoding="utf-8")
    assert v7_cli.replace("V7", "V8").replace("v7", "v8").rstrip() == v8_cli.rstrip()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source_map"].pop(),
        lambda value: value["source_map"].reverse(),
        lambda value: value["source_map"][0].__setitem__("path", "lewm/../escape.py"),
        lambda value: value["builder_review"].__setitem__("reviewer", "/root"),
        lambda value: value["builder_review"].__setitem__(
            "reviewer", builder.AUDITOR_IMPLEMENTATION_AUTHOR
        ),
        lambda value: value["auditor_review"].__setitem__(
            "reviewer", value["builder_review"]["reviewer"]
        ),
        lambda value: value.__setitem__("unreviewed_authority", True),
    ],
)
def test_incomplete_or_nonindependent_authority_rejects_before_target_open(
    mutate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _raw_by_role = _synthetic_authority()
    mutate(authority)
    authority = _hashed(authority)
    opened: list[Path] = []

    def forbidden(*_args: Any, **kwargs: Any) -> bytes:
        opened.append(kwargs["path"])
        raise AssertionError("invalid phase one reached a target opener")

    monkeypatch.setattr(builder, "_read_bound_regular_file", forbidden)
    with pytest.raises((PermissionError, builder.RawSupervisionBuildError)):
        builder._validate_authorization_phase_one(
            authority, authorization_file_sha256="a" * 64
        )
    assert opened == []


def test_complete_phase_one_precedes_exact_nine_role_open_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, raw_by_role = _synthetic_authority()
    phase_one = builder._validate_authorization_phase_one(
        authority, authorization_file_sha256="b" * 64
    )
    role_by_path = {path: role for role, path in builder.AUTHORIZED_ROLE_PATHS}
    opened_roles: list[str] = []

    def reader(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        relative = path.relative_to(repository_root).as_posix()
        role = role_by_path.get(relative)
        if role is not None:
            raw = raw_by_role[role]
            assert hashlib.sha256(raw).hexdigest() == expected_sha256
            opened_roles.append(role)
            return raw
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        return raw

    monkeypatch.setattr(builder, "_read_bound_regular_file", reader)
    receipt = builder._validate_authorization_phase_two(phase_one)
    assert opened_roles == [role for role, _path in builder.AUTHORIZED_ROLE_PATHS]
    assert receipt.authorization_file_sha256 == "b" * 64


def test_production_entry_and_worker_dispatch_have_no_legacy_or_injection_path() -> None:
    signature = inspect.signature(builder.execute_exact_build_v8)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    tree = ast.parse(Path(builder.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        "raw_supervision_builder_v7" in name
        or "raw_supervision_auditor" in name
        or "raw_supervision_builder_v8_test" in name
        for name in imported
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"__import__", "eval", "exec"}
        for node in ast.walk(tree)
    )
    submitted = {
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
        and node.args
        and isinstance(node.args[0], ast.Name)
    }
    assert submitted == {
        "_load_exact_scene_job",
        "_revalidate_exact_scene_sources",
        "_write_prepared_scene_job",
    }


def test_final_identity_boundary_detects_modify_restore_before_last_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction, retained, paths = _open_transaction(tmp_path, monkeypatch)
    original_identity = builder._named_directory_identity
    identity_calls = 0
    injected = False

    def inject(parent_fd: int, name: str) -> tuple[int, int]:
        nonlocal identity_calls, injected
        identity = original_identity(parent_fd, name)
        identity_calls += 1
        if identity_calls == 2:
            original = paths["source"].read_bytes()
            paths["source"].write_bytes(b"last-boundary mutation\n")
            paths["source"].write_bytes(original)
            injected = True
        return identity

    try:
        _publish_to_final_boundary(transaction, retained)
        monkeypatch.setattr(builder, "_named_directory_identity", inject)
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.require_final_quiet()
        assert injected
        assert transaction.poisoned
        assert paths["destination"].is_dir()
    finally:
        transaction.close()
        retained.close()


def test_clean_terminal_sequence_ends_in_second_identity_and_final_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction, retained, paths = _open_transaction(tmp_path, monkeypatch)
    phases: list[str] = []
    original = transaction._require_no_events

    def record(phase: str) -> None:
        phases.append(phase)
        original(phase)

    try:
        _publish_to_final_boundary(transaction, retained)
        monkeypatch.setattr(transaction, "_require_no_events", record)
        transaction.require_final_quiet()
        assert phases == [
            "post-rename parent fsync",
            "terminal source and published inventory",
            "final publication identity",
        ]
        assert paths["destination"].is_dir()
        assert not transaction.poisoned
    finally:
        transaction.close()
        retained.close()

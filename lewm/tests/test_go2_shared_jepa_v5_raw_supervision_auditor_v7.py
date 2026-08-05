from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v7 as auditor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _transaction_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    root = (tmp_path / "repository").absolute()
    publication_parent = root / "generated"
    dataset = publication_parent / "dataset"
    dataset.mkdir(parents=True)
    data = dataset / "data.bin"
    data_payload = b"bound-dataset-payload\n"
    _write(data, data_payload)
    manifest_payload = b'{"synthetic":true}\n'
    _write(dataset / "manifest.json", manifest_payload)

    authorization_path = root / "docs" / "authorization.json"
    authorization_sha256 = _write(authorization_path, b'{"synthetic":true}\n')
    plan_manifest = root / "metadata" / "manifest.json"
    plan_rows = root / "metadata" / "rows.jsonl"
    source_index = root / "metadata" / "source.json"
    plan_manifest_sha256 = _write(plan_manifest, b"manifest\n")
    plan_rows_sha256 = _write(plan_rows, b"rows\n")
    source_index_sha256 = _write(source_index, b"source\n")

    monkeypatch.setattr(auditor, "ROOT", root)
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(
        auditor, "CANONICAL_AUDIT_REPORT", publication_parent / "audit_v7.json"
    )
    monkeypatch.setattr(
        auditor,
        "CANONICAL_AUDIT_FAILURE",
        publication_parent / "audit_v7.failed.json",
    )
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(auditor, "FROZEN_V7_PREDECESSOR_SHA256", {})
    monkeypatch.setattr(auditor, "REVIEWED_V4_SOURCE_SHA256", {})
    monkeypatch.setattr(
        auditor,
        "DATASET_MANIFEST_RELATIVE_PATH",
        str(plan_manifest.relative_to(root)),
    )
    monkeypatch.setattr(
        auditor, "DATASET_ROWS_RELATIVE_PATH", str(plan_rows.relative_to(root))
    )
    monkeypatch.setattr(
        auditor, "SOURCE_INDEX_RELATIVE_PATH", str(source_index.relative_to(root))
    )
    monkeypatch.setattr(
        auditor, "DATASET_MANIFEST_FILE_SHA256", plan_manifest_sha256
    )
    monkeypatch.setattr(auditor, "DATASET_ROWS_FILE_SHA256", plan_rows_sha256)
    monkeypatch.setattr(auditor, "SOURCE_INDEX_FILE_SHA256", source_index_sha256)

    manifest = {
        "files": [
            {
                "path": "data.bin",
                "byte_count": len(data_payload),
                "file_sha256": hashlib.sha256(data_payload).hexdigest(),
            }
        ]
    }
    authorization = auditor.AcceptedAuthorizationV7(
        authorization_file_sha256=authorization_sha256,
        authorization_content_sha256="1" * 64,
        source_map_sha256="2" * 64,
        sources=(),
    )
    context = auditor._AuditPublicationContextV7(
        authorization=authorization,
        manifest=manifest,
        manifest_file_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        hashed_sources=(),
        parent_contracts=(),
    )
    result_core = {"schema": "synthetic_audit_v7", "verdict": "PASS"}
    result = {
        **result_core,
        "content_sha256": auditor.canonical_json_sha256(result_core),
    }
    retained = auditor._open_retained_directory_chain(publication_parent)
    name, descriptor, fingerprint, digest = auditor._stage_owned_audit_candidate(
        retained, result
    )
    transaction = auditor._ClosedAuditPublicationTransaction(
        context=context,
        retained=retained,
        candidate_name=name,
        candidate_descriptor=descriptor,
        candidate_fingerprint=fingerprint,
        candidate_sha256=digest,
    )
    return {
        "root": root,
        "publication_parent": publication_parent,
        "dataset": dataset,
        "data": data,
        "data_payload": data_payload,
        "result": result,
        "retained": retained,
        "candidate_name": name,
        "descriptor": descriptor,
        "transaction": transaction,
    }


def _close_fixture(value: dict[str, object], *, cleanup: bool = True) -> None:
    transaction = value["transaction"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    retained = value["retained"]
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    descriptor = value["descriptor"]
    assert isinstance(descriptor, int)
    transaction.close()
    if cleanup:
        auditor._cleanup_owned_audit_candidate(
            retained,
            candidate_name=str(value["candidate_name"]),
            candidate_descriptor=descriptor,
            renamed=transaction.renamed,
        )
    os.close(descriptor)
    retained.close()


def test_v7_fixed_authority_roles_and_builder_hashes() -> None:
    assert tuple(role for role, _path in auditor.SOURCE_ROLE_PATHS) == (
        "builder_source",
        "builder_cli",
        "builder_test",
        "builder_handoff",
        "builder_review",
        "auditor_source",
        "auditor_cli",
        "auditor_test",
        "auditor_review",
    )
    assert all("_v7" in path for _role, path in auditor.SOURCE_ROLE_PATHS)
    assert auditor.FROZEN_BUILDER_V7_ROLE_SHA256 == {
        "builder_source": "c79e68a2dcccb0fba937a9e6cd0ab778fc267b99473163c6c3c0bdbe6d1ac2ab",
        "builder_cli": "9fdecaac622f0e5c1022c6a6298557da0ad0d7effb4b350330536da177cbf432",
        "builder_test": "cb03351928d9d9849736a53b57d338cab10ae60bb14de062e3218e01901e99da",
        "builder_handoff": "b4fc01993b5d47e11f789192cfcf0d4e9a8ea5cce865ef0624cbce7e6379642d",
    }
    assert auditor.BUILDER_IMPLEMENTATION_AUTHOR == "/root/raw_v7_successor_author"
    assert (
        auditor.AUDITOR_IMPLEMENTATION_AUTHOR
        == "/root/raw_v7_successor_author/auditor_v7_author"
    )
    assert (
        auditor.AUTHORIZATION_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v7"
    )
    assert (
        auditor.REVIEW_BINDING_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v7"
    )
    assert (
        auditor.BUILDER_REVIEW_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_builder_v7_independent_review_v1"
    )
    assert (
        auditor.AUDITOR_REVIEW_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_independent_review_v1"
    )


def test_v7_predecessor_closure_matches_frozen_builder_v7_map() -> None:
    assert len(auditor.FROZEN_V7_PREDECESSOR_SHA256) == 55
    assert (
        auditor.canonical_json_sha256(auditor.FROZEN_V7_PREDECESSOR_SHA256)
        == "5b549b5fe3ea5eb61cea0c9b8320e804326229f66e5da6c1df952048d064bd3e"
    )
    assert auditor.FROZEN_V7_PREDECESSOR_SHA256[
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v7_"
        "authorization_successor_amendment_2026-07-13.md"
    ] == "ebeb552a89792b63f10c7d9ab5c9c9abd96d74d6ae7cf39f709f0657708798fc"
    assert auditor.FROZEN_V7_PREDECESSOR_SHA256[
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
        "independent_review_2026-07-13.json"
    ] == "55d50a38f0c7d23e4ff537b124db3b9f24a24ea5b30413ff6be1ac381870c163"
    assert auditor.FROZEN_V7_PREDECESSOR_SHA256[
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v6.py"
    ] == "cf67c993427950c147860f9afe0e7661b2cb6841ccec27a867868cc34c7c00b8"


def test_v7_report_namespaces_are_additive() -> None:
    assert auditor.CANONICAL_AUDIT_REPORT.name.endswith(".audit_v7.json")
    assert auditor.CANONICAL_AUDIT_FAILURE.name.endswith(".audit_v7.failed.json")
    assert auditor.AUDIT_SCHEMA == "lewm_go2_shared_jepa_v5_raw_supervision_audit_v7"
    assert (
        auditor.AUDIT_FAILURE_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_audit_failure_v7"
    )


def test_v7_exact_entry_is_fixed_keyword_only() -> None:
    signature = inspect.signature(auditor.execute_exact_audit_v7)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize("workers", [True, False, 0, 7, -1, 1.0, "1", None])
def test_v7_worker_boundary_rejects_non_exact_values(workers: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        auditor._strict_workers(workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("workers", range(1, 7))
def test_v7_worker_boundary_accepts_one_through_six(workers: int) -> None:
    assert auditor._strict_workers(workers) == workers


def test_absent_authority_reaches_no_payload_opener(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", Path("/absent/v7.json"))
    opened = False

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        nonlocal opened
        opened = True
        raise AssertionError("payload opener reached")

    monkeypatch.setattr(auditor, "_read_absolute_bound_payload", forbidden)
    with pytest.raises(PermissionError):
        auditor._require_exact_authority("a" * 64)
    assert opened is False


def test_clean_closed_transaction_publishes_exact_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    transaction = value["transaction"]
    retained = value["retained"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    transaction.validate_before_rename()
    transaction.rename_owned()
    transaction.validate_after_rename()
    os.fsync(retained.directory_fd)
    transaction.require_final_quiet()
    expected = auditor.canonical_json_bytes(value["result"]) + b"\n"
    assert auditor.CANONICAL_AUDIT_REPORT.read_bytes() == expected
    _close_fixture(value, cleanup=False)


def test_modify_then_restore_poisoned_without_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    data = value["data"]
    payload = value["data_payload"]
    assert isinstance(data, Path) and isinstance(payload, bytes)
    original = data.stat()
    data.write_bytes(b"X" * len(payload))
    data.write_bytes(payload)
    os.utime(data, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        value["transaction"].validate_before_rename()  # type: ignore[union-attr]
    assert not auditor.CANONICAL_AUDIT_REPORT.exists()
    _close_fixture(value)


def test_destination_race_preserves_foreign_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    foreign = b"foreign\n"
    auditor.CANONICAL_AUDIT_REPORT.write_bytes(foreign)
    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        value["transaction"].rename_owned()  # type: ignore[union-attr]
    assert auditor.CANONICAL_AUDIT_REPORT.read_bytes() == foreign
    _close_fixture(value)


def test_post_rename_ancestor_move_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    transaction = value["transaction"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    transaction.validate_before_rename()
    transaction.rename_owned()
    transaction.validate_after_rename()
    root = value["root"]
    assert isinstance(root, Path)
    moved = root.with_name(root.name + ".moved")
    root.rename(moved)
    try:
        with pytest.raises((auditor.RawSupervisionAuditError, FileNotFoundError)):
            transaction.require_final_quiet()
    finally:
        moved.rename(root)
    _close_fixture(value)


def test_source_has_one_transaction_wrapping_final_pass_and_rename() -> None:
    source_path = Path(auditor.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    exact = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_exact_audit_v7"
    )
    exact_source = ast.get_source_segment(source, exact)
    assert exact_source is not None
    assert exact_source.index("_ClosedAuditPublicationTransaction(") < exact_source.index(
        "_final_revalidate_authorized_audit_v7("
    )
    assert exact_source.index("_final_revalidate_authorized_audit_v7(") < exact_source.index(
        "transaction.rename_owned()"
    )
    assert "_ExclusiveAuditPublisher" not in source
    assert "go2_shared_jepa_v5_raw_supervision_auditor_v5 import" not in source
    imports = (
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not any(
        "go2_shared_jepa_v5_raw_supervision_builder" in module
        or "go2_shared_jepa_v5_raw_supervision_auditor_v6" in module
        for module in imports
    )


def test_v7_cli_imports_only_v7_exact_entry() -> None:
    cli = Path("scripts/audit_go2_shared_jepa_v5_raw_supervision_v7.py").read_text(
        encoding="utf-8"
    )
    assert "raw_supervision_auditor_v7" in cli
    assert "execute_exact_audit_v7" in cli
    assert "execute_exact_audit_v5" not in cli

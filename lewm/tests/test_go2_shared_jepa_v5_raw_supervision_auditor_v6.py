from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v6 as auditor


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
        auditor, "CANONICAL_AUDIT_REPORT", publication_parent / "audit_v6.json"
    )
    monkeypatch.setattr(
        auditor,
        "CANONICAL_AUDIT_FAILURE",
        publication_parent / "audit_v6.failed.json",
    )
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(auditor, "FROZEN_V6_PREDECESSOR_SHA256", {})
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
    authorization = auditor.AcceptedAuthorizationV6(
        authorization_file_sha256=authorization_sha256,
        authorization_content_sha256="1" * 64,
        source_map_sha256="2" * 64,
        sources=(),
    )
    context = auditor._AuditPublicationContextV6(
        authorization=authorization,
        manifest=manifest,
        manifest_file_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        hashed_sources=(),
        parent_contracts=(),
    )
    result_core = {"schema": "synthetic_audit_v6", "verdict": "PASS"}
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


def test_v6_fixed_authority_roles_and_builder_hashes() -> None:
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
    assert all("_v6" in path for _role, path in auditor.SOURCE_ROLE_PATHS)
    assert auditor.FROZEN_BUILDER_V6_ROLE_SHA256 == {
        "builder_source": "88c36063e257d9d163317abb15d7854f3da783e0ec15537da4c3d62b113740d7",
        "builder_cli": "089aca4882f4f574be7972914c12c05acabf1cd898bea6f59422bf07b94f828d",
        "builder_test": "acf5ca8cdd829d1c3c4ef44dbc4fe7e5d2f05a7dc7ec01662b60d9f27ececdd0",
        "builder_handoff": "d2cf130a9e2c902776327f6bd71a1b1f363a4dcfde6df0e2aba15edc3957e80b",
    }


def test_v6_exact_entry_is_fixed_keyword_only() -> None:
    signature = inspect.signature(auditor.execute_exact_audit_v6)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize("workers", [True, False, 0, 7, -1, 1.0, "1", None])
def test_v6_worker_boundary_rejects_non_exact_values(workers: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        auditor._strict_workers(workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("workers", range(1, 7))
def test_v6_worker_boundary_accepts_one_through_six(workers: int) -> None:
    assert auditor._strict_workers(workers) == workers


def test_absent_authority_reaches_no_payload_opener(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", Path("/absent/v6.json"))
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
        if isinstance(node, ast.FunctionDef) and node.name == "execute_exact_audit_v6"
    )
    exact_source = ast.get_source_segment(source, exact)
    assert exact_source is not None
    assert exact_source.index("_ClosedAuditPublicationTransaction(") < exact_source.index(
        "_final_revalidate_authorized_audit_v6("
    )
    assert exact_source.index("_final_revalidate_authorized_audit_v6(") < exact_source.index(
        "transaction.rename_owned()"
    )
    assert "_ExclusiveAuditPublisher" not in source
    assert "go2_shared_jepa_v5_raw_supervision_auditor_v5 import" not in source


def test_v6_cli_imports_only_v6_exact_entry() -> None:
    cli = Path("scripts/audit_go2_shared_jepa_v5_raw_supervision_v6.py").read_text(
        encoding="utf-8"
    )
    assert "raw_supervision_auditor_v6" in cli
    assert "execute_exact_audit_v6" in cli
    assert "execute_exact_audit_v5" not in cli

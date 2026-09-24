from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v8 as auditor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v7 as predecessor
from lewm.tests import (
    test_go2_shared_jepa_v5_raw_supervision_auditor_v7 as predecessor_tests,
)


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
        auditor, "CANONICAL_AUDIT_REPORT", publication_parent / "audit_v8.json"
    )
    monkeypatch.setattr(
        auditor,
        "CANONICAL_AUDIT_FAILURE",
        publication_parent / "audit_v8.failed.json",
    )
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(auditor, "FROZEN_V8_PREDECESSOR_SHA256", {})
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
    authorization = auditor.AcceptedAuthorizationV8(
        authorization_file_sha256=authorization_sha256,
        authorization_content_sha256="1" * 64,
        source_map_sha256="2" * 64,
        sources=(),
    )
    context = auditor._AuditPublicationContextV8(
        authorization=authorization,
        manifest=manifest,
        manifest_file_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        hashed_sources=(),
        parent_contracts=(),
    )
    result_core = {"schema": "synthetic_audit_v8", "verdict": "PASS"}
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


def _advance_to_terminal(value: dict[str, object]) -> None:
    transaction = value["transaction"]
    retained = value["retained"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    transaction.validate_before_rename()
    transaction.rename_owned()
    transaction.validate_after_rename()
    os.fsync(retained.directory_fd)


def _phase_one_payload() -> dict[str, object]:
    source_map = []
    for role, path in auditor.SOURCE_ROLE_PATHS:
        digest = auditor.FROZEN_BUILDER_V8_ROLE_SHA256.get(
            role, hashlib.sha256(role.encode("ascii")).hexdigest()
        )
        source_map.append({"role": role, "path": path, "sha256": digest})
    source_by_role = {row["role"]: row for row in source_map}

    def review_binding(
        *,
        kind: str,
        reviewer: str,
        implementation_author: str,
        schema: str,
        candidate_roles: tuple[str, ...],
    ) -> dict[str, object]:
        review = source_by_role[f"{kind}_review"]
        return {
            "schema": auditor.REVIEW_BINDING_SCHEMA,
            "review_schema": schema,
            "verdict": "PASS",
            "reviewer": reviewer,
            "implementation_author": implementation_author,
            "path": review["path"],
            "file_sha256": review["sha256"],
            "content_sha256": hashlib.sha256(
                f"{kind}-review".encode("ascii")
            ).hexdigest(),
            "candidate": [source_by_role[role] for role in candidate_roles],
        }

    core: dict[str, object] = {
        "schema": auditor.AUTHORIZATION_SCHEMA,
        "exact_build_authorized_after_independent_reviews": True,
        "builder_review": review_binding(
            kind="builder",
            reviewer="/root/builder_v8_reviewer",
            implementation_author=auditor.BUILDER_IMPLEMENTATION_AUTHOR,
            schema=auditor.BUILDER_REVIEW_SCHEMA,
            candidate_roles=auditor.BUILDER_CANDIDATE_ROLES,
        ),
        "auditor_review": review_binding(
            kind="auditor",
            reviewer="/root/auditor_v8_reviewer",
            implementation_author=auditor.AUDITOR_IMPLEMENTATION_AUTHOR,
            schema=auditor.AUDITOR_REVIEW_SCHEMA,
            candidate_roles=auditor.AUDITOR_CANDIDATE_ROLES,
        ),
        "source_map": source_map,
    }
    return {**core, "content_sha256": auditor.canonical_json_sha256(core)}


def test_v8_fixed_authority_roles_and_builder_hashes() -> None:
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
    assert all("_v8" in path for _role, path in auditor.SOURCE_ROLE_PATHS)
    assert auditor.FROZEN_BUILDER_V8_ROLE_SHA256 == {
        "builder_source": "f45533354c8b45b88f8eadb2126ec5eaf96fe1f57c21a9bfcd95a8855bfaaa35",
        "builder_cli": "f6471f1fa0ca7a13976f752a41ee9ddacbc76636e4d5fb0eee1ebf75bdaee72d",
        "builder_test": "fc1f0cf3fc18bdbd1393be6a514bc04459f943f39b438ced78ebee30e7c57d9a",
        "builder_handoff": "9f4898e3620ac87c9a0145be103c4fdf397f727fe37d9f6ca306a0f50916156b",
    }
    assert (
        auditor.BUILDER_IMPLEMENTATION_AUTHOR
        == "/root/raw_v7_successor_author/auditor_v7_author"
    )
    assert (
        auditor.AUDITOR_IMPLEMENTATION_AUTHOR
        == "/root/camera_v5_independent/camera_v7_pre_freeze_review/"
        "v7_review_artifact_schema"
    )
    assert (
        auditor.AUTHORIZATION_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v8"
    )
    assert (
        auditor.REVIEW_BINDING_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v8"
    )
    assert (
        auditor.BUILDER_REVIEW_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_builder_v8_independent_review_v1"
    )
    assert (
        auditor.AUDITOR_REVIEW_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v8_independent_review_v1"
    )


def test_v8_predecessor_closure_matches_frozen_builder_v8_map() -> None:
    assert len(auditor.FROZEN_V8_PREDECESSOR_SHA256) == 69
    assert (
        auditor.canonical_json_sha256(auditor.FROZEN_V8_PREDECESSOR_SHA256)
        == "79fe832122ed335188357a59bad8a031cc235449ef17e6e19ac78de9d5aff669"
    )
    assert auditor.FROZEN_V8_PREDECESSOR_SHA256[
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v8_"
        "terminal_quiet_successor_amendment_2026-07-13.md"
    ] == "054de82d8648cd6be7edff01b82d549ec916700ebffad51698d4c2041edc6c88"
    assert auditor.FROZEN_V8_PREDECESSOR_SHA256[
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_v8_existing_agent_"
        "identity_rebinding_amendment_2026-07-13.md"
    ] == "392745c80ca2c6e7a103cca4a55c3614cd2c988de9a379fba950b0087df41698"
    assert auditor.FROZEN_V8_PREDECESSOR_SHA256[
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
        "independent_review_2026-07-13.json"
    ] == "55d50a38f0c7d23e4ff537b124db3b9f24a24ea5b30413ff6be1ac381870c163"
    assert auditor.FROZEN_V8_PREDECESSOR_SHA256[
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v6.py"
    ] == "cf67c993427950c147860f9afe0e7661b2cb6841ccec27a867868cc34c7c00b8"


def test_v8_phase_one_accepts_exact_nine_rows_without_target_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        nonlocal opened
        opened = True
        raise AssertionError("phase one opened a mapped target")

    monkeypatch.setattr(auditor, "_read_absolute_bound_payload", forbidden)
    phase_one = auditor._validate_authorization_phase_one(
        _phase_one_payload(),
        authorization_file_sha256="a" * 64,
    )
    assert tuple(item.role for item in phase_one.sources) == auditor.SOURCE_ROLES
    assert opened is False


@pytest.mark.parametrize(
    "mutation",
    [
        "reordered_rows",
        "aliased_path",
        "amendment_author_reviewer",
        "other_implementation_reviewer",
        "same_reviewers",
    ],
)
def test_v8_phase_one_rejects_authority_identity_or_path_drift(
    mutation: str,
) -> None:
    payload = _phase_one_payload()
    source_map = payload["source_map"]
    assert isinstance(source_map, list)
    if mutation == "reordered_rows":
        source_map[0], source_map[1] = source_map[1], source_map[0]
    elif mutation == "aliased_path":
        assert isinstance(source_map[0], dict)
        source_map[0]["path"] = "./" + str(source_map[0]["path"])
    else:
        builder_review = payload["builder_review"]
        auditor_review = payload["auditor_review"]
        assert isinstance(builder_review, dict) and isinstance(auditor_review, dict)
        if mutation == "amendment_author_reviewer":
            builder_review["reviewer"] = "/root"
        elif mutation == "other_implementation_reviewer":
            builder_review["reviewer"] = auditor.AUDITOR_IMPLEMENTATION_AUTHOR
        elif mutation == "same_reviewers":
            auditor_review["reviewer"] = builder_review["reviewer"]
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = auditor.canonical_json_sha256(core)
    with pytest.raises(
        (auditor.RawSupervisionAuditError, PermissionError)
    ):
        auditor._validate_authorization_phase_one(
            payload,
            authorization_file_sha256="a" * 64,
        )


def test_v8_review_authority_is_source_only_and_no_retry() -> None:
    assert auditor._expected_review_authority("auditor") == {
        "auditor_source_approved": True,
        **{field: False for field in auditor.REVIEW_AUTHORITY_FALSE_FIELDS},
    }
    assert auditor._expected_review_authority("auditor")["retry_authorized"] is False


def test_v8_report_namespaces_are_additive() -> None:
    assert auditor.CANONICAL_AUDIT_REPORT.name.endswith(".audit_v8.json")
    assert auditor.CANONICAL_AUDIT_FAILURE.name.endswith(".audit_v8.failed.json")
    assert auditor.AUDIT_SCHEMA == "lewm_go2_shared_jepa_v5_raw_supervision_audit_v8"
    assert (
        auditor.AUDIT_FAILURE_SCHEMA
        == "lewm_go2_shared_jepa_v5_raw_supervision_audit_failure_v8"
    )


def test_v8_exact_entry_is_fixed_keyword_only() -> None:
    signature = inspect.signature(auditor.execute_exact_audit_v8)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize("workers", [True, False, 0, 7, -1, 1.0, "1", None])
def test_v8_worker_boundary_rejects_non_exact_values(workers: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        auditor._strict_workers(workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("workers", range(1, 7))
def test_v8_worker_boundary_accepts_one_through_six(workers: int) -> None:
    assert auditor._strict_workers(workers) == workers


def test_v8_one_and_six_worker_science_bytes_are_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest = "d" * 64
    arrays = tuple(
        np.zeros(shape, dtype=np.dtype(dtype))
        for _name, dtype, shape in auditor.ARRAY_LAYOUT
    )
    observed = auditor.StoredEndpointEvidence(
        endpoint_identity_sha256=digest,
        arrays=arrays,
        evidence_content_sha256="e" * 64,
        raster_content_sha256="f" * 64,
    )
    manifest = {"content_sha256": "c" * 64, "shards": [{"path": "one"}]}
    sample = [{"endpoint_identity_sha256": digest}]
    population = {"synthetic_population": 1}
    inputs = auditor.AuditInputs(
        plan=SimpleNamespace(pairs=(object(),)),
        inventory=SimpleNamespace(),
    )
    worker_calls: list[int] = []
    monkeypatch.setattr(auditor, "_require_exact_authority", lambda _digest: object())
    monkeypatch.setattr(
        auditor, "_require_real_directory", lambda *_args, **_kwargs: tmp_path
    )
    monkeypatch.setattr(auditor, "_parse_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        auditor, "_validate_root_file_inventory", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        auditor,
        "_validate_pair_and_endpoint_indexes",
        lambda *_args, **_kwargs: ((), {digest: {}}, {}),
    )
    monkeypatch.setattr(
        auditor, "_derive_population", lambda *_args, **_kwargs: population
    )
    monkeypatch.setattr(
        auditor, "_validate_frozen_population", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        auditor, "_validate_sample_precommit", lambda *_args, **_kwargs: sample
    )
    monkeypatch.setattr(
        auditor,
        "_validate_shards_parallel",
        lambda *_args, **_kwargs: {digest: observed},
    )

    def recompute(
        _sample: object,
        _endpoints: object,
        _inputs: object,
        workers: int,
        **_kwargs: object,
    ) -> dict[str, tuple[np.ndarray, ...]]:
        worker_calls.append(workers)
        return {digest: arrays}

    monkeypatch.setattr(auditor, "_exact_sample_recomputer", recompute)
    monkeypatch.setattr(
        auditor, "_read_absolute_bound_payload", lambda *_args, **_kwargs: b""
    )
    one = auditor._audit_fixed_dataset(
        authorization_sha256="a" * 64,
        expected_manifest_file_sha256="b" * 64,
        inputs=inputs,
        workers=1,
    )
    six = auditor._audit_fixed_dataset(
        authorization_sha256="a" * 64,
        expected_manifest_file_sha256="b" * 64,
        inputs=inputs,
        workers=6,
    )
    assert auditor.canonical_json_bytes(one) == auditor.canonical_json_bytes(six)
    assert worker_calls == [1, 6]


def test_absent_authority_reaches_no_payload_opener(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", Path("/absent/v8.json"))
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


def test_v8_terminal_sequence_has_two_drains_and_final_identity_pass() -> None:
    source = textwrap.dedent(
        inspect.getsource(
            auditor._ClosedAuditPublicationTransaction.require_final_quiet
        )
    )
    assert source.count("self._require_no_events(") == 2
    first_drain = source.index('self._require_no_events("post-rename parent fsync")')
    full_inventory = source.index("self._validate_bound_inventory(", first_drain)
    first_ancestry = source.index("self._retained.validate(", full_inventory)
    first_report = source.index(
        "self._validate_report_inventory_and_destination(", first_ancestry
    )
    second_drain = source.index(
        'self._require_no_events("terminal source and report validation")',
        first_report,
    )
    final_ancestry = source.index("self._retained.validate(", second_drain)
    final_report = source.index(
        "self._validate_report_inventory_and_destination(", final_ancestry
    )
    assert (
        first_drain
        < full_inventory
        < first_ancestry
        < first_report
        < second_drain
        < final_ancestry
        < final_report
    )
    assert "sleep(" not in source
    assert "while " not in source


def test_frozen_v7_block_reproducer_still_returns_success_after_final_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = predecessor_tests._transaction_fixture(monkeypatch, tmp_path)
    transaction = value["transaction"]
    retained = value["retained"]
    root = value["root"]
    assert isinstance(
        transaction, predecessor._ClosedAuditPublicationTransaction
    )
    assert isinstance(retained, predecessor._RetainedDirectoryChain)
    assert isinstance(root, Path)
    transaction.validate_before_rename()
    transaction.rename_owned()
    transaction.validate_after_rename()
    os.fsync(retained.directory_fd)
    moved = root.with_name(root.name + ".moved_after_v7_drain")
    original_drain = transaction._require_no_events

    def drain_then_move(phase: str) -> None:
        original_drain(phase)
        root.rename(moved)
        (root / "generated").mkdir(parents=True)

    monkeypatch.setattr(transaction, "_require_no_events", drain_then_move)
    try:
        transaction.require_final_quiet()
        assert not predecessor.CANONICAL_AUDIT_REPORT.exists()
    finally:
        (root / "generated").rmdir()
        root.rmdir()
        moved.rename(root)
        predecessor_tests._close_fixture(value, cleanup=False)


@pytest.mark.parametrize("drain_index", [0, 1])
@pytest.mark.parametrize("position", ["before", "during", "after"])
def test_v8_terminal_drain_boundaries_reject_report_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drain_index: int,
    position: str,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    report = auditor.CANONICAL_AUDIT_REPORT
    mutated = False

    def mutate() -> None:
        nonlocal mutated
        if not mutated:
            report.write_bytes(b"terminal report mutation\n")
            mutated = True

    if position == "during":
        original_read = transaction._read_events
        calls = 0

        def read_with_mutation(*, wait_milliseconds: int = 0):
            nonlocal calls
            if calls == drain_index:
                mutate()
            calls += 1
            return original_read(wait_milliseconds=wait_milliseconds)

        monkeypatch.setattr(transaction, "_read_events", read_with_mutation)
    else:
        original_drain = transaction._require_no_events
        calls = 0

        def drain_with_mutation(phase: str) -> None:
            nonlocal calls
            if calls == drain_index and position == "before":
                mutate()
            original_drain(phase)
            if calls == drain_index and position == "after":
                mutate()
            calls += 1

        monkeypatch.setattr(
            transaction, "_require_no_events", drain_with_mutation
        )

    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        transaction.require_final_quiet()
    assert mutated is True
    assert transaction._poison_reason is not None
    _close_fixture(value, cleanup=False)


@pytest.mark.parametrize("ancestry_index", [0, 1])
def test_v8_each_terminal_ancestry_pass_rejects_move_and_recreation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ancestry_index: int,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    retained = value["retained"]
    root = value["root"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    assert isinstance(root, Path)
    original_validate = retained.validate
    moved = root.with_name(root.name + f".moved_ancestry_{ancestry_index}")
    calls = 0

    def validate_with_move(*, allow_final_metadata_change: bool = False) -> None:
        nonlocal calls
        if calls == ancestry_index:
            root.rename(moved)
            (root / "generated").mkdir(parents=True)
        calls += 1
        original_validate(
            allow_final_metadata_change=allow_final_metadata_change
        )

    monkeypatch.setattr(retained, "validate", validate_with_move)
    try:
        with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
            transaction.require_final_quiet()
        assert transaction._poison_reason is not None
    finally:
        if root.exists():
            (root / "generated").rmdir()
            root.rmdir()
        moved.rename(root)
        _close_fixture(value, cleanup=False)


def test_v8_terminal_inventory_rejects_source_modify_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    data = value["data"]
    payload = value["data_payload"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(data, Path) and isinstance(payload, bytes)
    original_inventory = auditor._dataset_transaction_inventory
    injected = False

    def inventory_with_restore(*args: object, **kwargs: object):
        nonlocal injected
        if not injected:
            before = data.stat()
            data.write_bytes(b"X" * len(payload))
            data.write_bytes(payload)
            os.utime(data, ns=(before.st_atime_ns, before.st_mtime_ns))
            injected = True
        return original_inventory(*args, **kwargs)

    monkeypatch.setattr(
        auditor, "_dataset_transaction_inventory", inventory_with_restore
    )
    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        transaction.require_final_quiet()
    assert injected is True
    assert transaction._poison_reason is not None
    _close_fixture(value, cleanup=False)


def test_v8_terminal_report_modify_restore_is_poisoned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    report = auditor.CANONICAL_AUDIT_REPORT
    expected = report.read_bytes()
    original_drain = transaction._require_no_events
    injected = False

    def drain_then_restore(phase: str) -> None:
        nonlocal injected
        original_drain(phase)
        if not injected:
            before = report.stat()
            report.write_bytes(b"X" * len(expected))
            report.write_bytes(expected)
            os.utime(report, ns=(before.st_atime_ns, before.st_mtime_ns))
            injected = True

    monkeypatch.setattr(transaction, "_require_no_events", drain_then_restore)
    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        transaction.require_final_quiet()
    assert injected is True
    _close_fixture(value, cleanup=False)


@pytest.mark.parametrize("report_check_index", [0, 1])
def test_v8_each_report_hash_pass_rechecks_post_hash_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report_check_index: int,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    descriptor = value["descriptor"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(descriptor, int)
    report = auditor.CANONICAL_AUDIT_REPORT
    expected = report.read_bytes()
    original_hash = auditor._sha256_fd
    calls = 0

    def hash_then_restore(current_descriptor: int) -> str:
        nonlocal calls
        digest = original_hash(current_descriptor)
        if current_descriptor == descriptor:
            if calls == report_check_index:
                before = report.stat()
                report.write_bytes(b"X" * len(expected))
                report.write_bytes(expected)
                os.utime(report, ns=(before.st_atime_ns, before.st_mtime_ns))
            calls += 1
        return digest

    monkeypatch.setattr(auditor, "_sha256_fd", hash_then_restore)
    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        transaction.require_final_quiet()
    assert transaction._poison_reason is not None
    _close_fixture(value, cleanup=False)


@pytest.mark.parametrize("report_check_index", [0, 1])
def test_v8_each_report_identity_pass_rejects_move_then_recreation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report_check_index: int,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    retained = value["retained"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    report = auditor.CANONICAL_AUDIT_REPORT
    displaced = report.with_name(report.name + ".displaced")
    expected = report.read_bytes()
    original_lstat = auditor._lstat_optional_at
    calls = 0

    def lstat_with_recreation(parent_fd: int, name: str):
        nonlocal calls
        result = original_lstat(parent_fd, name)
        if (
            parent_fd == retained.directory_fd
            and name == str(value["candidate_name"])
        ):
            if calls == report_check_index:
                report.rename(displaced)
                report.write_bytes(expected)
            calls += 1
        return result

    monkeypatch.setattr(auditor, "_lstat_optional_at", lstat_with_recreation)
    try:
        with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
            transaction.require_final_quiet()
        assert transaction._poison_reason is not None
    finally:
        if displaced.exists():
            report.unlink()
            displaced.rename(report)
        _close_fixture(value, cleanup=False)


def test_v8_unrelated_ancestry_churn_is_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    root = value["root"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(root, Path)
    unrelated = root / "unrelated-ancestry-child.tmp"
    unrelated.write_bytes(b"unrelated\n")
    unrelated.unlink()
    transaction.require_final_quiet()
    _close_fixture(value, cleanup=False)


@pytest.mark.parametrize("protected_role", ["source", "publication"])
def test_v8_protected_named_churn_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    protected_role: str,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    _advance_to_terminal(value)
    transaction = value["transaction"]
    root = value["root"]
    publication_parent = value["publication_parent"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(root, Path) and isinstance(publication_parent, Path)
    parent = root if protected_role == "source" else publication_parent
    name = "metadata" if protected_role == "source" else "unrelated.tmp"
    original = parent / name
    moved = parent / f".{name}.moved"
    if original.exists():
        original.rename(moved)
        moved.rename(original)
    else:
        original.write_bytes(b"publication churn\n")
        original.unlink()
    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        transaction.require_final_quiet()
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


def test_candidate_modify_restore_poisoned_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    transaction = value["transaction"]
    descriptor = value["descriptor"]
    candidate_name = value["candidate_name"]
    publication_parent = value["publication_parent"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(descriptor, int)
    assert isinstance(candidate_name, str) and isinstance(publication_parent, Path)
    candidate = publication_parent / candidate_name
    payload = os.pread(descriptor, os.fstat(descriptor).st_size, 0)
    before = candidate.stat()
    os.pwrite(descriptor, b"X" * len(payload), 0)
    os.pwrite(descriptor, payload, 0)
    os.utime(candidate, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
        transaction.validate_before_rename()
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
        if isinstance(node, ast.FunctionDef) and node.name == "execute_exact_audit_v8"
    )
    exact_source = ast.get_source_segment(source, exact)
    assert exact_source is not None
    assert exact_source.index("_ClosedAuditPublicationTransaction(") < exact_source.index(
        "_final_revalidate_authorized_audit_v8("
    )
    assert exact_source.index("_final_revalidate_authorized_audit_v8(") < exact_source.index(
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


def test_v8_science_ast_is_mechanically_identical_to_v7() -> None:
    v7_source = Path(predecessor.__file__).read_text(encoding="utf-8")
    v8_source = Path(auditor.__file__).read_text(encoding="utf-8")

    def science_region(source: str) -> str:
        start = source.index("def _validate_integer_fields(")
        end = source.index("def _strict_canonical_json_object(", start)
        return source[start:end]

    normalized_v8 = re.sub(r"V8", "V7", science_region(v8_source))
    normalized_v8 = re.sub(r"v8", "v7", normalized_v8)
    assert ast.dump(ast.parse(normalized_v8), include_attributes=False) == ast.dump(
        ast.parse(science_region(v7_source)), include_attributes=False
    )


def test_v8_unmodified_transaction_methods_match_v7_ast() -> None:
    def methods(module: object) -> dict[str, str]:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        transaction = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_ClosedAuditPublicationTransaction"
        )
        return {
            node.name: ast.unparse(node)
            for node in transaction.body
            if isinstance(node, ast.FunctionDef)
        }

    v7_methods = methods(predecessor)
    v8_methods = methods(auditor)
    retained = {
        "_poison",
        "_add_watch",
        "_bind_directories",
        "_bind_source_leaves",
        "_bind_candidate",
        "_read_events",
        "_candidate_path",
        "validate_before_rename",
        "validate_after_rename",
        "close",
    }
    for name in retained:
        normalized = v8_methods[name].replace("V8", "V7").replace("v8", "v7")
        assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(
            ast.parse(v7_methods[name]), include_attributes=False
        )


def test_v8_production_source_has_no_hook_dynamic_import_or_legacy_exact_entry() -> None:
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "importlib" not in source
    assert "test_hook" not in source
    assert "monkeypatch" not in source
    for version in range(1, 8):
        assert f"execute_exact_audit_v{version}" not in source
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert not any(
        "go2_shared_jepa_v5_raw_supervision_auditor_v7" in module
        or "go2_shared_jepa_v5_raw_supervision_builder_v8" in module
        for module in imports
    )


def test_v8_cli_imports_only_v8_exact_entry() -> None:
    cli = Path("scripts/audit_go2_shared_jepa_v5_raw_supervision_v8.py").read_text(
        encoding="utf-8"
    )
    assert "raw_supervision_auditor_v8" in cli
    assert "execute_exact_audit_v8" in cli
    assert "execute_exact_audit_v5" not in cli

from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v8 as v8
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v9 as auditor
from lewm.tests import (
    test_go2_shared_jepa_v5_raw_supervision_auditor_v8 as v8_tests,
)
from lewm.tests import (
    test_go2_shared_jepa_v5_raw_supervision_auditor_v9 as author_tests,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CANDIDATE = (
    (
        "auditor_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v9.py",
        "ebe0c6a31cf027b8b0bc049257079a5e0ab0493b12aabeb96bf50f02990bbc14",
    ),
    (
        "auditor_cli",
        "scripts/audit_go2_shared_jepa_v5_raw_supervision_v9.py",
        "76f0b2b29eff8df6905fed142cc622eb0fa8024c397a3c7efb54e58cc36f67ba",
    ),
    (
        "auditor_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v9.py",
        "10951cc2e622281f72ec2a20114ccca184af7624a95fef4683c83dc6839992d1",
    ),
)
AMENDMENT = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v9_"
    "linearization_successor_amendment_2026-07-13.md",
    "6fba5de8d7f04d85bd87e084096ae269c3d3dd6368a6db0b0f8f149c1c5cf773",
)
HANDOFF = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_"
    "author_handoff_2026-07-13.md",
    "819d1857bf315f775f45c4a16db994f333d7174c5c20f5cb762f93d04b30a3a5",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _top_definitions(source: str) -> dict[str, ast.AST]:
    definitions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert len(definitions) == 106
    return definitions


def _methods(node: ast.AST) -> dict[str, ast.FunctionDef]:
    assert isinstance(node, ast.ClassDef)
    return {
        child.name: child
        for child in node.body
        if isinstance(child, ast.FunctionDef)
    }


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def test_independent_frozen_bindings_authority_and_83_row_closure() -> None:
    for _role, relative, expected in EXPECTED_CANDIDATE:
        raw = (ROOT / relative).read_bytes()
        raw.decode("ascii")
        assert hashlib.sha256(raw).hexdigest() == expected
    for relative, expected in (AMENDMENT, HANDOFF):
        raw = (ROOT / relative).read_bytes()
        raw.decode("ascii")
        assert hashlib.sha256(raw).hexdigest() == expected

    assert tuple(auditor.SOURCE_ROLE_PATHS[5:8]) == tuple(
        (role, relative) for role, relative, _digest in EXPECTED_CANDIDATE
    )
    assert len(auditor.SOURCE_ROLE_PATHS) == 9
    assert len(auditor.FROZEN_V9_PREDECESSOR_SHA256) == 83
    assert (
        auditor.canonical_json_sha256(auditor.FROZEN_V9_PREDECESSOR_SHA256)
        == "76823317704cb35ad3342cb27c03c218816da89b56c294897a7eddd651cdd83e"
    )
    assert auditor.FROZEN_V9_PREDECESSOR_SHA256[AMENDMENT[0]] == AMENDMENT[1]
    assert auditor.AUDITOR_IMPLEMENTATION_AUTHOR == (
        "/root/camera_v5_independent/camera_v7_pre_freeze_review/"
        "v7_review_artifact_schema"
    )
    assert auditor._expected_review_authority("auditor") == {
        "auditor_source_approved": True,
        **{field: False for field in auditor.REVIEW_AUTHORITY_FALSE_FIELDS},
    }


def test_independent_v9_delta_is_only_the_reviewed_terminal_successor() -> None:
    v8_source = (
        ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v8.py"
    ).read_text(encoding="utf-8")
    v9_source = Path(auditor.__file__).read_text(encoding="utf-8")
    normalized_v9 = v9_source.replace("V9", "V8").replace("v9", "v8")
    v8_definitions = _top_definitions(v8_source)
    v9_definitions = _top_definitions(normalized_v9)

    changed = {
        name
        for name in v8_definitions
        if _dump(v8_definitions[name]) != _dump(v9_definitions[name])
    }
    assert changed == {"_ClosedAuditPublicationTransaction"}

    v8_methods = _methods(v8_definitions["_ClosedAuditPublicationTransaction"])
    v9_methods = _methods(v9_definitions["_ClosedAuditPublicationTransaction"])
    assert set(v9_methods) - set(v8_methods) == {"_stat_absolute_report_leaf"}
    assert set(v8_methods) - set(v9_methods) == set()
    changed_methods = {
        name
        for name in v8_methods
        if _dump(v8_methods[name]) != _dump(v9_methods[name])
    }
    assert changed_methods == {
        "_validate_report_inventory_and_destination",
        "require_final_quiet",
    }

    science_start = v8_source.index("def _validate_integer_fields(")
    science_end = v8_source.index(
        "def _strict_canonical_json_object(", science_start
    )
    v9_science_start = normalized_v9.index("def _validate_integer_fields(")
    v9_science_end = normalized_v9.index(
        "def _strict_canonical_json_object(", v9_science_start
    )
    assert _dump(ast.parse(v8_source[science_start:science_end])) == _dump(
        ast.parse(normalized_v9[v9_science_start:v9_science_end])
    )


def test_independent_linearization_is_literal_last_observation() -> None:
    quiet_source = textwrap.dedent(
        inspect.getsource(
            auditor._ClosedAuditPublicationTransaction.require_final_quiet
        )
    )
    quiet = ast.parse(quiet_source).body[0]
    assert isinstance(quiet, ast.FunctionDef)
    drains = [
        node
        for node in ast.walk(quiet)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_require_no_events"
    ]
    assert [ast.literal_eval(call.args[0]) for call in drains] == [
        "post-rename parent fsync",
        "terminal source and report validation",
        "final publication linearization",
    ]
    final = quiet.body[-1]
    assert isinstance(final, ast.Expr)
    assert isinstance(final.value, ast.Call)
    assert isinstance(final.value.func, ast.Attribute)
    assert final.value.func.attr == "_require_no_events"
    assert ast.literal_eval(final.value.args[0]) == "final publication linearization"

    exact_source = textwrap.dedent(inspect.getsource(auditor.execute_exact_audit_v9))
    exact = ast.parse(exact_source).body[0]
    assert isinstance(exact, ast.FunctionDef)
    transaction_try = next(node for node in exact.body if isinstance(node, ast.Try))
    linearization = next(
        index
        for index, statement in enumerate(transaction_try.body)
        if ast.unparse(statement) == "transaction.require_final_quiet()"
    )
    assert [
        ast.unparse(statement)
        for statement in transaction_try.body[linearization + 1 :]
    ] == ["result = dict(prepared.result)"]
    assert [
        ast.unparse(statement)
        for statement in exact.body[exact.body.index(transaction_try) + 1 :]
    ] == [
        "if transaction is not None:\n    transaction.close()",
        "if candidate_descriptor is not None:\n    os.close(candidate_descriptor)",
        "retained.close()",
        "return result",
    ]

    walk_source = textwrap.dedent(
        inspect.getsource(
            auditor._ClosedAuditPublicationTransaction._stat_absolute_report_leaf
        )
    )
    assert "self._retained.descriptors[0]" in walk_source
    assert "self._retained.entries" in walk_source
    assert "_directory_open_flags()" in walk_source
    assert "dir_fd=parent_fd" in walk_source
    assert "follow_symlinks=False" in walk_source


def test_independent_v8_alias_gap_is_closed_by_literal_third_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = v8_tests._transaction_fixture(monkeypatch, tmp_path / "v8")
    v8_tests._advance_to_terminal(old)
    old_transaction = old["transaction"]
    old_root = old["root"]
    assert isinstance(old_transaction, v8._ClosedAuditPublicationTransaction)
    assert isinstance(old_root, Path)
    old_moved = old_root.with_name(old_root.name + ".moved_for_independent_v8")
    old_validate = old_transaction._validate_report_inventory_and_destination
    old_calls = 0

    def v8_move_and_alias(*, after_rename: bool) -> None:
        nonlocal old_calls
        if old_calls == 1:
            old_root.rename(old_moved)
            old_root.symlink_to(old_moved, target_is_directory=True)
        old_calls += 1
        old_validate(after_rename=after_rename)

    monkeypatch.setattr(
        old_transaction,
        "_validate_report_inventory_and_destination",
        v8_move_and_alias,
    )
    try:
        old_transaction.require_final_quiet()
        assert old_root.is_symlink()
        assert old_transaction._poison_reason is None
    finally:
        if old_root.is_symlink():
            old_root.unlink()
        if old_moved.exists():
            old_moved.rename(old_root)
        v8_tests._close_fixture(old, cleanup=False)

    current = author_tests._transaction_fixture(monkeypatch, tmp_path / "v9")
    author_tests._advance_to_terminal(current)
    transaction = current["transaction"]
    report = auditor.CANONICAL_AUDIT_REPORT
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    reads = 0
    original_read = transaction._read_events

    def mutate_during_third_read(*, wait_milliseconds: int = 0):
        nonlocal reads
        if reads == 2:
            report.write_bytes(b"mutation during publication linearization\n")
        reads += 1
        return original_read(wait_milliseconds=wait_milliseconds)

    monkeypatch.setattr(transaction, "_read_events", mutate_during_third_read)
    with pytest.raises(
        auditor.RawSupervisionAuditError,
        match="final publication linearization",
    ):
        transaction.require_final_quiet()
    assert reads == 3
    assert transaction._poison_reason is not None
    author_tests._close_fixture(current, cleanup=False)


@pytest.mark.parametrize("component", ["repository_root", "publication_parent"])
def test_independent_absolute_walk_rejects_real_directory_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    component: str,
) -> None:
    value = author_tests._transaction_fixture(monkeypatch, tmp_path)
    author_tests._advance_to_terminal(value)
    transaction = value["transaction"]
    target = value["root"] if component == "repository_root" else value[
        "publication_parent"
    ]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(target, Path)
    moved = target.with_name(target.name + f".moved_{component}")
    target.rename(moved)
    target.mkdir()
    try:
        with pytest.raises(
            auditor.RawSupervisionAuditError,
            match="absolute audit report ancestry changed",
        ):
            transaction._stat_absolute_report_leaf(after_rename=True)
        assert transaction._poison_reason is not None
    finally:
        target.rmdir()
        moved.rename(target)
        author_tests._close_fixture(value, cleanup=False)


def test_independent_cleanup_preserves_foreign_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = author_tests._transaction_fixture(monkeypatch, tmp_path)
    retained = value["retained"]
    descriptor = value["descriptor"]
    candidate_name = value["candidate_name"]
    publication_parent = value["publication_parent"]
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    assert isinstance(descriptor, int)
    assert isinstance(candidate_name, str)
    assert isinstance(publication_parent, Path)
    candidate = publication_parent / candidate_name
    displaced = publication_parent / (candidate_name + ".owned-displaced")
    candidate.rename(displaced)
    foreign = b"foreign replacement must survive cleanup\n"
    candidate.write_bytes(foreign)
    try:
        assert (
            auditor._cleanup_owned_audit_candidate(
                retained,
                candidate_name=candidate_name,
                candidate_descriptor=descriptor,
                renamed=False,
            )
            is False
        )
        assert candidate.read_bytes() == foreign
    finally:
        candidate.unlink()
        displaced.rename(candidate)
        author_tests._close_fixture(value)

"""Independent source-only QA for the frozen Raw Supervision Auditor V7.

All dynamic filesystem work is synthetic and rooted below ``tmp_path``.  This
suite never opens the canonical authorization, source payload, dataset, audit
output, RGB, checkpoint, G2, held-out, runtime, hardware, or production paths.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v7 as auditor
from lewm.tests import test_go2_shared_jepa_v5_raw_supervision_auditor_v7 as author_tests


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/coordinator_v2_qa"
FROZEN_CANDIDATE = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v7.py": (
        "3550917e36d1401f8ad9c895afcf591b3226b2e0c5a09f4ad427d0b04bb1490e"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v7.py": (
        "9940d35e4e33b628bf64c4947cb1f92a68e1413e20e63fd0b9080728a64f949e"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v7.py": (
        "6d123d39014fd9c3dc7b34d113e665861536010d79117a3004cb8ee1484e894f"
    ),
}
FROZEN_REVIEW_INPUTS = {
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_author_handoff_2026-07-13.md": (
        "1351a2641025735a3a96d50283a7119f2ce02f7c49578e656133b1c48a46fd21"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v7_"
    "authorization_successor_amendment_2026-07-13.md": (
        "ebeb552a89792b63f10c7d9ab5c9c9abd96d74d6ae7cf39f709f0657708798fc"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_"
    "independent_review_2026-07-13.json": (
        "85d1a111e10eaac865a80cebd97e771b39eaa47f6ebcf6ffe6716ed445a1ff46"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_frozen_candidate_and_review_inputs() -> None:
    expected = {**FROZEN_CANDIDATE, **FROZEN_REVIEW_INPUTS}
    assert {relative: _sha256(ROOT / relative) for relative in expected} == expected


def test_independent_v7_authority_and_api_are_narrow() -> None:
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
    signature = inspect.signature(auditor.execute_exact_audit_v7)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert auditor._expected_review_authority("auditor") == {
        "auditor_source_approved": True,
        **{field: False for field in auditor.REVIEW_AUTHORITY_FALSE_FIELDS},
    }


def test_block_terminal_quiet_has_no_post_drain_revalidation() -> None:
    source = textwrap.dedent(
        inspect.getsource(
            auditor._ClosedAuditPublicationTransaction.require_final_quiet
        )
    )
    tree = ast.parse(source)
    statements = tree.body[0].body
    event_drains = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_require_no_events"
    ]
    assert len(event_drains) == 1
    assert isinstance(statements[-1], ast.Expr)
    final_call = statements[-1].value
    assert isinstance(final_call, ast.Call)
    assert isinstance(final_call.func, ast.Attribute)
    assert final_call.func.attr == "_require_no_events"


def test_block_ancestor_move_after_terminal_drain_returns_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = author_tests._transaction_fixture(monkeypatch, tmp_path)
    transaction = value["transaction"]
    retained = value["retained"]
    root = value["root"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    assert isinstance(root, Path)

    transaction.validate_before_rename()
    transaction.rename_owned()
    transaction.validate_after_rename()
    os.fsync(retained.directory_fd)

    moved = root.with_name(root.name + ".moved_during_terminal_drain")
    original_drain = transaction._require_no_events
    injected = False

    def drain_then_move_ancestor(phase: str) -> None:
        nonlocal injected
        original_drain(phase)
        assert phase == "post-rename parent fsync"
        root.rename(moved)
        (root / "generated").mkdir(parents=True)
        injected = True

    monkeypatch.setattr(transaction, "_require_no_events", drain_then_move_ancestor)
    try:
        # The V7 contract requires this mutation to poison the transaction.
        # Frozen Auditor V7 instead returns success with no report at its
        # canonical path because its final drain is its final operation.
        transaction.require_final_quiet()
        assert injected is True
        assert not auditor.CANONICAL_AUDIT_REPORT.exists()
    finally:
        (root / "generated").rmdir()
        root.rmdir()
        moved.rename(root)
        author_tests._close_fixture(value, cleanup=False)

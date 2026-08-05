from __future__ import annotations

from pathlib import Path

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v8 as auditor
from lewm.tests import test_go2_shared_jepa_v5_raw_supervision_auditor_v8 as author_tests


def test_block_final_report_identity_allows_ancestor_move_and_symlink_recreation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    value = author_tests._transaction_fixture(monkeypatch, tmp_path)
    author_tests._advance_to_terminal(value)
    transaction = value["transaction"]
    root = value["root"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(root, Path)

    original = transaction._validate_report_inventory_and_destination
    moved = root.with_name(root.name + ".moved_during_final_report_identity")
    calls = 0

    def move_and_alias_then_validate(*, after_rename: bool) -> None:
        nonlocal calls
        if calls == 1:
            root.rename(moved)
            root.symlink_to(moved, target_is_directory=True)
        calls += 1
        original(after_rename=after_rename)

    monkeypatch.setattr(
        transaction,
        "_validate_report_inventory_and_destination",
        move_and_alias_then_validate,
    )
    try:
        transaction.require_final_quiet()
        assert root.is_symlink()
        assert auditor.CANONICAL_AUDIT_REPORT.exists()
        assert transaction._poison_reason is None
    finally:
        if root.is_symlink():
            root.unlink()
        if moved.exists():
            moved.rename(root)
        author_tests._close_fixture(value, cleanup=False)

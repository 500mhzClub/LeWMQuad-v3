"""Different-agent adversarial QA for frozen Raw Supervision Builder V6."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v6 as builder
from lewm.tests.test_go2_shared_jepa_v5_raw_supervision_builder_v6 import (
    _open_synthetic_transaction,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v6.py": (
        "88c36063e257d9d163317abb15d7854f3da783e0ec15537da4c3d62b113740d7"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v6.py": (
        "089aca4882f4f574be7972914c12c05acabf1cd898bea6f59422bf07b94f828d"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py": (
        "acf5ca8cdd829d1c3c4ef44dbc4fe7e5d2f05a7dc7ec01662b60d9f27ececdd0"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
    "author_handoff_2026-07-13.md": (
        "d2cf130a9e2c902776327f6bd71a1b1f363a4dcfde6df0e2aba15edc3957e80b"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_builder_v6_candidate_rehashes_exactly() -> None:
    assert {path: _sha256(ROOT / path) for path in FROZEN} == FROZEN


def test_final_quiet_rejects_publication_ancestor_move_and_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final close must still bind the canonical publication ancestry."""

    canonical_ancestor = tmp_path / "canonical"
    canonical_ancestor.mkdir()
    transaction, retained, paths = _open_synthetic_transaction(
        canonical_ancestor,
        monkeypatch,
    )
    moved_ancestor = tmp_path / "moved"
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        transaction.validate_after_rename()
        os.fsync(retained.parent_fd)

        canonical_ancestor.rename(moved_ancestor)
        canonical_ancestor.mkdir()
        (canonical_ancestor / "publication").mkdir()

        with pytest.raises(builder.RawSupervisionBuildError):
            transaction.require_final_quiet()
        assert not paths["destination"].exists()
        assert (moved_ancestor / "publication" / "dataset").is_dir()
    finally:
        transaction.close()
        retained.close()

"""Independent CPU-only review probes for Raw Supervision Builder V7."""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import pytest

from lewm.datasets import (
    go2_shared_jepa_v5_raw_supervision_builder_v6 as v6,
)
from lewm.datasets import (
    go2_shared_jepa_v5_raw_supervision_builder_v7 as v7,
)


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v7_"
    "authorization_successor_amendment_2026-07-13.md"
)
CANDIDATE = {
    AMENDMENT: "ebeb552a89792b63f10c7d9ab5c9c9abd96d74d6ae7cf39f709f0657708798fc",
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v7.py": (
        "c79e68a2dcccb0fba937a9e6cd0ab778fc267b99473163c6c3c0bdbe6d1ac2ab"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v7.py": (
        "9fdecaac622f0e5c1022c6a6298557da0ad0d7effb4b350330536da177cbf432"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v7.py": (
        "cb03351928d9d9849736a53b57d338cab10ae60bb14de062e3218e01901e99da"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_"
    "author_handoff_2026-07-13.md": (
        "b4fc01993b5d47e11f789192cfcf0d4e9a8ea5cce865ef0624cbce7e6379642d"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _open_transaction(
    module: Any,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, dict[str, Path]]:
    source_parent = root / "sources"
    publication_parent = root / "publication"
    source_parent.mkdir(parents=True)
    publication_parent.mkdir()
    source = source_parent / "source.bin"
    source.write_bytes(b"independent frozen source\n")
    staging_name = ".dataset.review-staging"
    staging = publication_parent / staging_name
    staging.mkdir(mode=0o700)
    payload = staging / "payload.bin"
    payload.write_bytes(b"independent staged payload\n")
    manifest = module._with_content_sha256(
        {"schema": "independent_builder_v7_transaction_probe_v1"}
    )
    (staging / "manifest.json").write_bytes(
        module.canonical_json_bytes(manifest) + b"\n"
    )
    retained = module._open_publication_parent(publication_parent)
    staging_identity = module._named_directory_identity(
        retained.parent_fd, staging_name
    )
    monkeypatch.setattr(
        module,
        "_exact_publication_source_hashes",
        lambda _context: {source: _sha256(source)},
    )
    try:
        transaction = module._ClosedPublicationTransaction(
            context=object(),
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
    except BaseException:
        retained.close()
        raise
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


def test_frozen_builder_v7_candidate_and_parent_closure_rehash() -> None:
    assert {path: _sha256(ROOT / path) for path in CANDIDATE} == CANDIDATE
    assert v7.FROZEN_PARENT_HASHES[AMENDMENT] == CANDIDATE[AMENDMENT]
    assert len(v7.FROZEN_PARENT_HASHES) == 55
    assert len(v7.REVIEWED_V4_SOURCES) == 9
    assert all(_sha256(ROOT / path) == digest for path, digest in v7.FROZEN_PARENT_HASHES.items())
    assert all(_sha256(ROOT / path) == digest for path, digest in v7.REVIEWED_V4_SOURCES.items())


def test_v6_false_success_is_real_and_v7_rejects_same_ancestor_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v6_root = tmp_path / "v6" / "canonical"
    v6_root.mkdir(parents=True)
    old, retained, paths = _open_transaction(v6, v6_root, monkeypatch)
    moved = tmp_path / "v6-moved"
    try:
        _publish_to_final_boundary(old, retained)
        v6_root.rename(moved)
        v6_root.mkdir()
        (v6_root / "publication").mkdir()
        old.require_final_quiet()
        assert not paths["destination"].exists()
        assert (moved / "publication" / "dataset").is_dir()
    finally:
        old.close()
        retained.close()

    v7_root = tmp_path / "v7" / "canonical"
    v7_root.mkdir(parents=True)
    new, retained, paths = _open_transaction(v7, v7_root, monkeypatch)
    moved = tmp_path / "v7-moved"
    try:
        _publish_to_final_boundary(new, retained)
        v7_root.rename(moved)
        v7_root.mkdir()
        (v7_root / "publication").mkdir()
        with pytest.raises(v7.RawSupervisionBuildError, match="poisoned"):
            new.require_final_quiet()
        assert not paths["destination"].exists()
        assert (moved / "publication" / "dataset").is_dir()
    finally:
        new.close()
        retained.close()


def test_v7_rejects_ancestor_move_between_terminal_inventory_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    transaction, retained, _paths = _open_transaction(v7, canonical, monkeypatch)
    moved = tmp_path / "moved-after-inventory"
    original = transaction._require_no_events
    injected = False

    def inject(phase: str) -> None:
        nonlocal injected
        original(phase)
        if phase == "terminal source and published inventory" and not injected:
            injected = True
            canonical.rename(moved)
            canonical.mkdir()
            (canonical / "publication").mkdir()

    try:
        _publish_to_final_boundary(transaction, retained)
        monkeypatch.setattr(transaction, "_require_no_events", inject)
        with pytest.raises(v7.RawSupervisionBuildError, match="poisoned"):
            transaction.require_final_quiet()
        assert injected
    finally:
        transaction.close()
        retained.close()


def test_v7_production_api_and_event_filter_are_closed() -> None:
    signature = inspect.signature(v7.execute_exact_build_v7)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in signature.parameters.values()
    )
    tree = ast.parse(Path(v7.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("raw_supervision_builder_v6" in name for name in imported)
    assert not any("go2_shared_jepa_v5_raw_supervision_builder_v7_test" in name for name in imported)
    assert v7._IN_ANCESTOR_MASK & v7._IN_MOVE_SELF
    assert v7._IN_ANCESTOR_MASK & v7._IN_DELETE_SELF
    assert v7._IN_ANCESTOR_MASK & v7._IN_ATTRIB
    assert v7._IN_ANCESTOR_MASK & v7._IN_UNMOUNT
    # IN_IGNORED is an output-only event emitted automatically when a watch is
    # removed. It belongs in the accepted event bits and fail-closed parser,
    # not in the inotify_add_watch subscription mask.
    assert v7._IN_EVENT_BITS & v7._IN_IGNORED
    read_events_source = inspect.getsource(v7._ClosedPublicationTransaction._read_events)
    assert "_IN_IGNORED | _IN_UNMOUNT" in read_events_source

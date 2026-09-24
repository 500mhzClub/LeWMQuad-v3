"""Independent quality assurance for raw-supervision metadata plan V4.

Only the frozen allowlisted metadata files and temporary synthetic trees may be
opened. Referenced source payloads and excluded/protected roles remain unopened.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v2 as v2
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v3 as v3
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v4 as v4


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INDEX = ROOT / v4.SOURCE_INDEX_RELATIVE_PATH
DATASET_MANIFEST = ROOT / v2._v1.DATASET_MANIFEST_RELATIVE_PATH
DATASET_ROWS = ROOT / v2._v1.DATASET_ROWS_RELATIVE_PATH
SIDECAR_MANIFEST = ROOT / v2._v1.SIDECAR_MANIFEST_RELATIVE_PATH
ROLE_FILES = {
    role: SIDECAR_MANIFEST.parent / f"{role}.jsonl"
    for role in v4.DEVELOPMENT_ROLES
}
EXPECTED_OPEN_COUNTS = Counter(
    {
        DATASET_MANIFEST: 1,
        DATASET_ROWS: 1,
        SIDECAR_MANIFEST: 1,
        ROLE_FILES["train"]: 2,
        ROLE_FILES["checkpoint_selection"]: 2,
        ROLE_FILES["probability_calibration"]: 2,
        SOURCE_INDEX: 1,
    }
)
EXPECTED_INVENTORY_HASHES = {
    "scene_role": "f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b",
    "frames": "7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d",
    "manifests": "2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5",
    "plans": "0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4",
    "summaries": "bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    nested_root: bool = False,
) -> tuple[Path, Path, bytes]:
    repo = tmp_path / "anchor" / "repo" if nested_root else tmp_path / "repo"
    source = repo / "index" / "source.jsonl"
    source.parent.mkdir(parents=True)
    payload = b'{"scene_id":"synthetic"}\n'
    source.write_bytes(payload)
    monkeypatch.setattr(v4, "SOURCE_INDEX_RELATIVE_PATH", "index/source.jsonl")
    monkeypatch.setattr(
        v4,
        "SOURCE_INDEX_FILE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return repo, source, payload


def _advance_directory_fingerprint(path: Path) -> None:
    before = path.stat(follow_symlinks=False)
    os.utime(
        path,
        ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000),
        follow_symlinks=False,
    )
    assert v4._file_fingerprint(path.stat(follow_symlinks=False)) != (
        v4._file_fingerprint(before)
    )


def test_v4_independent_freezes_exact_candidate_identities() -> None:
    expected = {
        ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v4.py": (
            "d6282a6ee561d34fbe20542f31acd8c7bee82badfa74d1d640930148a9951de2"
        ),
        ROOT / "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4.py": (
            "724f1c93023256015fe0d468c56591fab35512de79c1e0b0822e78bccdb4a0e0"
        ),
        ROOT
        / "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_author_handoff_2026-07-13.md": (
            "4753d83517a41d2e70e8f25d7cb03ad3709f2d798d1f9f39eea358a527c91415"
        ),
    }
    assert {path: _sha256(path) for path in expected} == expected


@pytest.mark.parametrize("target", ("source_parent", "root_ancestor"))
def test_v4_requires_complete_post_read_directory_fingerprint_continuity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    repo, source, _payload = _bind_index(
        tmp_path,
        monkeypatch,
        nested_root=target == "root_ancestor",
    )
    changed_directory = source.parent if target == "source_parent" else repo.parent
    original_read = os.read
    changed = False

    def change_fingerprint_then_read(descriptor: int, length: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            _advance_directory_fingerprint(changed_directory)
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "read", change_fingerprint_then_read)
    with pytest.raises(v4.RawSupervisionPlanError, match="directory.*changed"):
        v4._read_frozen_source_index(repo)
    assert changed


def test_v4_closes_transient_repo_ancestor_alias_before_first_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source, _payload = _bind_index(tmp_path, monkeypatch, nested_root=True)
    anchor = repo.parent
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def transient_alias(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == anchor.name and dir_fd is not None:
            replaced = True
            moved = tmp_path / "moved-anchor"
            anchor.rename(moved)
            anchor.symlink_to(moved, target_is_directory=True)
            try:
                return original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                anchor.unlink()
                moved.rename(anchor)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", transient_alias)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v4.RawSupervisionPlanError, match="changed during open"):
        v4._read_frozen_source_index(repo)
    assert replaced and reads == 0


def test_v4_closes_same_inode_relink_before_first_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch)
    before = source.stat(follow_symlinks=False)
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def relink(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == source.name and dir_fd is not None:
            replaced = True
            moved = tmp_path / "moved-source.jsonl"
            source.rename(moved)
            os.link(moved, source)
            moved.unlink()
            after = source.stat(follow_symlinks=False)
            assert v4._entry_identity(after) == v4._entry_identity(before)
            assert after.st_nlink == 1
            assert v4._file_fingerprint(after) != v4._file_fingerprint(before)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", relink)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v4.RawSupervisionPlanError, match="fingerprint changed"):
        v4._read_frozen_source_index(repo)
    assert replaced and reads == 0


@pytest.mark.parametrize("kind", ("directory", "fifo", "symlink", "hardlink"))
def test_v4_rejects_nonregular_or_linked_leaf_before_first_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch)
    source.unlink()
    if kind == "directory":
        source.mkdir()
    elif kind == "fifo":
        os.mkfifo(source)
    elif kind == "symlink":
        target = source.with_name("target.jsonl")
        target.write_bytes(b"target\n")
        source.symlink_to(target)
    else:
        target = source.with_name("target.jsonl")
        target.write_bytes(b"target\n")
        os.link(target, source)
    original_read = os.read
    reads = 0

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v4.RawSupervisionPlanError, match="regular|symlink|hard-link"):
        v4._read_frozen_source_index(repo)
    assert reads == 0


def test_v4_reconstructs_exact_science_and_only_ten_metadata_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original_path_open = Path.open
    original_os_open = os.open

    def traced_path_open(path: Path, *args: Any, **kwargs: Any):
        resolved = path.resolve(strict=True)
        assert resolved in EXPECTED_OPEN_COUNTS
        opened.append(resolved)
        return original_path_open(path, *args, **kwargs)

    def traced_os_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_os_open(path, flags, mode, dir_fd=dir_fd)
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            candidate = Path(os.fsdecode(os.fspath(path)))
            if not candidate.is_absolute() and dir_fd is not None:
                candidate = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / candidate
            resolved = candidate.resolve(strict=True)
            assert resolved in EXPECTED_OPEN_COUNTS
            opened.append(resolved)
        return descriptor

    monkeypatch.setattr(Path, "open", traced_path_open)
    monkeypatch.setattr(os, "open", traced_os_open)
    plan = v4.load_frozen_development_metadata(ROOT)
    inventory = v4.load_frozen_development_source_inventory(ROOT, plan)

    assert Counter(opened) == EXPECTED_OPEN_COUNTS
    assert sum(EXPECTED_OPEN_COUNTS.values()) == 10
    assert len(plan.pairs) == 5172
    assert plan.value["endpoint_instance_count"] == 10344
    assert len(plan.endpoints) == 9460
    assert len(inventory.records) == 88
    assert inventory.hashes == EXPECTED_INVENTORY_HASHES
    assert plan.value["content_sha256"] == (
        "8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3"
    )
    assert plan.value["ordered_pair_sha256"] == (
        "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
    )
    assert plan.value["ordered_endpoint_sha256"] == (
        "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
    )
    assert not any(plan.value["licenses"].values())
    assert all(
        value == 0
        for key, value in inventory.access_ledger.items()
        if key.endswith("_opens")
    )
    assert SIDECAR_MANIFEST.parent / "g2_evaluation.jsonl" not in opened


def test_v4_validates_exactly_704_selected_references_without_opening_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspected: list[str] = []
    original_validate = v2._validate_referenced_path

    def traced_validate(value: object, **kwargs: Any):
        inspected.append(str(value))
        return original_validate(value, **kwargs)

    monkeypatch.setattr(v2, "_validate_referenced_path", traced_validate)
    plan = v4.load_frozen_development_metadata(ROOT)
    inventory = v4.load_frozen_development_source_inventory(ROOT, plan)

    assert len(inspected) == 704
    assert len(set(inspected)) == 352
    assert all(Counter(inspected).values())
    assert set(Counter(inspected).values()) == {2}
    assert all("g2_evaluation" not in path for path in inspected)
    assert all(
        value == 0
        for key, value in inventory.access_ledger.items()
        if key.endswith("_opens")
    )

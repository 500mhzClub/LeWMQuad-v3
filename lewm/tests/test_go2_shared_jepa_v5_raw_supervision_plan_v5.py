"""Author closure for raw-supervision metadata plan V5.

Only frozen allowlisted metadata files and temporary synthetic trees may be
opened. Referenced source payloads and excluded/protected roles stay unopened.
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
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v5 as v5


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INDEX = ROOT / v5.SOURCE_INDEX_RELATIVE_PATH
DATASET_MANIFEST = ROOT / v2._v1.DATASET_MANIFEST_RELATIVE_PATH
DATASET_ROWS = ROOT / v2._v1.DATASET_ROWS_RELATIVE_PATH
SIDECAR_MANIFEST = ROOT / v2._v1.SIDECAR_MANIFEST_RELATIVE_PATH
ROLE_FILES = {
    role: SIDECAR_MANIFEST.parent / f"{role}.jsonl"
    for role in v5.DEVELOPMENT_ROLES
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
    monkeypatch.setattr(v5, "SOURCE_INDEX_RELATIVE_PATH", "index/source.jsonl")
    monkeypatch.setattr(
        v5,
        "SOURCE_INDEX_FILE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return repo, source, payload


def _change_directory_fingerprint(path: Path, mutation: str = "mtime") -> None:
    before = path.stat(follow_symlinks=False)
    identity = v5._entry_identity(before)
    if mutation == "mtime":
        os.utime(
            path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000),
            follow_symlinks=False,
        )
    elif mutation == "mode":
        os.chmod(path, stat.S_IMODE(before.st_mode) ^ stat.S_IWGRP)
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    after = path.stat(follow_symlinks=False)
    assert v5._entry_identity(after) == identity
    assert stat.S_IFMT(after.st_mode) == stat.S_IFMT(before.st_mode)
    assert v5._file_fingerprint(after) != v5._file_fingerprint(before)


def test_v5_freezes_v4_candidate_and_block_evidence() -> None:
    expected = {
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v4.py": (
            "d6282a6ee561d34fbe20542f31acd8c7bee82badfa74d1d640930148a9951de2"
        ),
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4.py": (
            "724f1c93023256015fe0d468c56591fab35512de79c1e0b0822e78bccdb4a0e0"
        ),
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_author_handoff_2026-07-13.md": (
            "4753d83517a41d2e70e8f25d7cb03ad3709f2d798d1f9f39eea358a527c91415"
        ),
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4_independent_qa.py": (
            "5e079be910f5633c01df6d9afc2967715515b27293cc09f279eb71f373c40f78"
        ),
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_independent_review_2026-07-13.md": (
            "46d44155916b53dd850274b1c8704d0feb62ed7c1bd05f28391b1ea83ded9757"
        ),
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_independent_review_block_2026-07-13.json": (
            "6897064fb3752b0d9552c0db9b8bd81372a3ba891ff3e98ce7174a84e9e6c2d8"
        ),
    }
    assert {
        relative: _sha256(ROOT / relative)
        for relative in expected
    } == expected


@pytest.mark.parametrize("target", ("source_parent", "root_ancestor", "repo_root"))
@pytest.mark.parametrize("mutation", ("mtime", "mode"))
def test_v5_rejects_v4_directory_fingerprint_mismatches_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation: str,
) -> None:
    repo, source, _payload = _bind_index(
        tmp_path,
        monkeypatch,
        nested_root=target == "root_ancestor",
    )
    changed_directory = {
        "source_parent": source.parent,
        "root_ancestor": repo.parent,
        "repo_root": repo,
    }[target]
    original_read = os.read
    changed = False

    def change_then_read(descriptor: int, length: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            _change_directory_fingerprint(changed_directory, mutation)
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "read", change_then_read)
    with pytest.raises(v5.RawSupervisionPlanError, match="directory.*changed"):
        v5._read_frozen_source_index(repo)
    assert changed


def test_v5_rejects_directory_change_at_pre_read_boundary_without_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch)
    original_revalidate = v5._revalidate_open_chain
    original_read = os.read
    validations = 0
    reads = 0

    def change_then_validate(**kwargs: Any) -> None:
        nonlocal validations
        validations += 1
        if validations == 1:
            _change_directory_fingerprint(source.parent)
        original_revalidate(**kwargs)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(v5, "_revalidate_open_chain", change_then_validate)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v5.RawSupervisionPlanError, match="directory.*changed"):
        v5._read_frozen_source_index(repo)
    assert validations == 1
    assert reads == 0


def test_v5_validates_complete_chain_immediately_around_all_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source, payload = _bind_index(tmp_path, monkeypatch, nested_root=True)
    original_revalidate = v5._revalidate_open_chain
    original_read = os.read
    events: list[str] = []
    validations: list[dict[str, Any]] = []

    def trace_validation(**kwargs: Any) -> None:
        events.append("validate")
        validations.append(dict(kwargs))
        original_revalidate(**kwargs)

    def trace_read(descriptor: int, length: int) -> bytes:
        events.append("read")
        return original_read(descriptor, length)

    monkeypatch.setattr(v5, "_revalidate_open_chain", trace_validation)
    monkeypatch.setattr(os, "read", trace_read)
    assert v5._read_frozen_source_index(repo) == payload
    assert events[0] == "validate"
    assert events[-1] == "validate"
    assert events.count("validate") == 2
    assert all(event == "read" for event in events[1:-1])
    for validation in validations:
        assert len(validation["anchor_fingerprint"]) == 7
        assert len(validation["root_fingerprint"]) == 7
        assert len(validation["leaf_fingerprint"]) == 7
        assert validation["root_directory_entries"]
        assert validation["directory_entries"]
        assert all(
            len(entry[3]) == 7
            for entry in (
                *validation["root_directory_entries"],
                *validation["directory_entries"],
            )
        )


def test_v5_source_index_uses_only_component_nofollow_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source, payload = _bind_index(tmp_path, monkeypatch, nested_root=True)
    calls: list[tuple[object, int | None, int]] = []
    original_open = os.open

    def traced_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        calls.append((path, dir_fd, flags))
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", traced_open)
    assert v5._read_frozen_source_index(repo) == payload
    assert tuple(path for path, _dir_fd, _flags in calls) == (
        Path(repo.anchor),
        *repo.parts[1:],
        "index",
        "source.jsonl",
    )
    assert calls[0][1] is None
    assert all(dir_fd is not None for _path, dir_fd, _flags in calls[1:])
    assert all(flags & os.O_NOFOLLOW for _path, _dir_fd, flags in calls)
    assert all(flags & os.O_DIRECTORY for _path, _dir_fd, flags in calls[:-1])
    assert calls[-1][2] & os.O_NONBLOCK
    assert not calls[-1][2] & os.O_DIRECTORY


def test_v5_rejects_transient_repo_ancestor_alias_before_first_read(
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
    with pytest.raises(v5.RawSupervisionPlanError, match="changed during open"):
        v5._read_frozen_source_index(repo)
    assert replaced
    assert reads == 0


def test_v5_revalidates_filesystem_root_fingerprint(tmp_path: Path) -> None:
    anchor = tmp_path / "filesystem-root"
    root = anchor / "repo"
    leaf = root / "source.jsonl"
    root.mkdir(parents=True)
    leaf.write_bytes(b"source\n")
    anchor_fd = os.open(anchor, v5._directory_flags())
    root_fd = os.open("repo", v5._directory_flags(), dir_fd=anchor_fd)
    leaf_fd = os.open("source.jsonl", v5._file_flags(), dir_fd=root_fd)
    try:
        anchor_fingerprint = v5._file_fingerprint(os.fstat(anchor_fd))
        root_fingerprint = v5._file_fingerprint(os.fstat(root_fd))
        leaf_fingerprint = v5._file_fingerprint(os.fstat(leaf_fd))
        arguments = {
            "filesystem_root": anchor,
            "anchor_fd": anchor_fd,
            "anchor_fingerprint": anchor_fingerprint,
            "root": root,
            "root_fd": root_fd,
            "root_fingerprint": root_fingerprint,
            "root_directory_entries": [
                (anchor_fd, "repo", root_fd, root_fingerprint)
            ],
            "directory_entries": [],
            "leaf_parent_fd": root_fd,
            "leaf_name": "source.jsonl",
            "leaf_fd": leaf_fd,
            "leaf_fingerprint": leaf_fingerprint,
        }
        v5._revalidate_open_chain(**arguments)
        _change_directory_fingerprint(anchor)
        with pytest.raises(
            v5.RawSupervisionPlanError,
            match="filesystem root directory changed",
        ):
            v5._revalidate_open_chain(**arguments)
    finally:
        os.close(leaf_fd)
        os.close(root_fd)
        os.close(anchor_fd)


def test_v5_rejects_same_inode_leaf_relink_before_first_read(
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
            assert v5._entry_identity(after) == v5._entry_identity(before)
            assert after.st_nlink == 1
            assert v5._file_fingerprint(after) != v5._file_fingerprint(before)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", relink)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v5.RawSupervisionPlanError, match="fingerprint changed"):
        v5._read_frozen_source_index(repo)
    assert replaced
    assert reads == 0


@pytest.mark.parametrize("kind", ("directory", "fifo", "symlink", "hardlink"))
def test_v5_rejects_nonregular_or_linked_leaf_before_first_read(
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
    with pytest.raises(
        v5.RawSupervisionPlanError,
        match="regular|symlink|hard-link",
    ):
        v5._read_frozen_source_index(repo)
    assert reads == 0


def test_v5_reconstructs_exact_science_with_only_ten_metadata_opens(
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
    plan = v5.load_frozen_development_metadata(ROOT)
    inventory = v5.load_frozen_development_source_inventory(ROOT, plan)

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


def test_v5_validates_704_references_without_opening_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspected: list[str] = []
    original_validate = v2._validate_referenced_path

    def traced_validate(value: object, **kwargs: Any):
        inspected.append(str(value))
        return original_validate(value, **kwargs)

    monkeypatch.setattr(v2, "_validate_referenced_path", traced_validate)
    plan = v5.load_frozen_development_metadata(ROOT)
    inventory = v5.load_frozen_development_source_inventory(ROOT, plan)

    assert len(inspected) == 704
    assert len(set(inspected)) == 352
    assert set(Counter(inspected).values()) == {2}
    assert all("g2_evaluation" not in path for path in inspected)
    assert all(
        value == 0
        for key, value in inventory.access_ledger.items()
        if key.endswith("_opens")
    )

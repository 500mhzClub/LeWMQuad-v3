"""Independent QA for the frozen raw-supervision metadata plan V5.

The real-data checks open only the frozen metadata allowlist.  All adversarial
continuity checks use temporary synthetic trees; referenced payloads are never
opened.
"""
from __future__ import annotations

from collections import Counter
import builtins
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
FROZEN_INPUTS = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v5.py": (
        "67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5.py": (
        "384af6e2b254ea98d32fd7f4798beafe429a4cd83fee6e2903d0d1e8c84f9636"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_"
    "author_handoff_2026-07-13.md": (
        "b362d26372f01e670a477dda5e7abb5e55370cc1d8d89052545afa229e7bba66"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4_"
    "independent_qa.py": (
        "5e079be910f5633c01df6d9afc2967715515b27293cc09f279eb71f373c40f78"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_"
    "independent_review_2026-07-13.md": (
        "46d44155916b53dd850274b1c8704d0feb62ed7c1bd05f28391b1ea83ded9757"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_index(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, bytes]:
    repo = base / "ancestor_a" / "ancestor_b" / "repo"
    source = repo / "catalog" / "nested" / "source.jsonl"
    source.parent.mkdir(parents=True)
    payload = b'{"scene_id":"synthetic"}\n'
    source.write_bytes(payload)
    monkeypatch.setattr(
        v5,
        "SOURCE_INDEX_RELATIVE_PATH",
        "catalog/nested/source.jsonl",
    )
    monkeypatch.setattr(
        v5,
        "SOURCE_INDEX_FILE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return repo, source, payload


def _advance_fingerprint(path: Path) -> None:
    before = path.stat(follow_symlinks=False)
    identity = v5._entry_identity(before)
    os.utime(
        path,
        ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000),
        follow_symlinks=False,
    )
    after = path.stat(follow_symlinks=False)
    assert v5._entry_identity(after) == identity
    assert stat.S_IFMT(after.st_mode) == stat.S_IFMT(before.st_mode)
    assert v5._file_fingerprint(after) != v5._file_fingerprint(before)


def _descriptor_path(descriptor: int) -> Path:
    return Path(os.readlink(f"/proc/self/fd/{descriptor}"))


def _fake_hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _synthetic_plan() -> v5.DevelopmentRawSupervisionPlan:
    return v5.DevelopmentRawSupervisionPlan(
        value={"licenses": {"raw_raycast_build_authorized": False}},
        pairs=(
            {
                "scene_id": "scene_a",
                "family": "family_a",
                "source_split": "train",
            },
        ),
        endpoints=(
            {
                "identity": {
                    "scene_id": "scene_a",
                    "dataset_role": "train",
                }
            },
        ),
    )


def _synthetic_source_row(repo: Path) -> dict[str, Any]:
    base = repo / "metadata" / "scene_a"
    return {
        "scene_id": "scene_a",
        "family": "family_a",
        "split": "train",
        "frames_jsonl_path": str(base / "frames.jsonl"),
        "scene_manifest_path": str(base / "manifest.json"),
        "render_plan_path": str(base / "plan.json"),
        "render_summary_path": str(base / "summary.json"),
        "hashes": {
            "frames_jsonl_file_sha256": _fake_hash("frames"),
            "scene_manifest_file_sha256": _fake_hash("manifest-file"),
            "scene_manifest_sha256": _fake_hash("manifest-content"),
            "render_plan_file_sha256": _fake_hash("plan"),
            "render_summary_file_sha256": _fake_hash("summary"),
        },
    }


def test_v5_independent_freezes_exact_inputs_and_v4_block_evidence() -> None:
    assert {
        relative: _sha256(ROOT / relative)
        for relative in FROZEN_INPUTS
    } == FROZEN_INPUTS


def test_v5_independent_retains_and_rechecks_the_complete_original_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, payload = _bind_index(tmp_path, monkeypatch)
    original_validate = v5._revalidate_open_chain
    original_read = os.read
    events: list[str] = []
    snapshots: list[dict[str, Any]] = []

    def validate(**kwargs: Any) -> None:
        events.append("validate")
        original_validate(**kwargs)
        snapshots.append(
            {
                "filesystem_root": kwargs["filesystem_root"],
                "anchor_fingerprint": kwargs["anchor_fingerprint"],
                "root": kwargs["root"],
                "root_fingerprint": kwargs["root_fingerprint"],
                "root_rows": tuple(
                    (_descriptor_path(row[2]), row[3])
                    for row in kwargs["root_directory_entries"]
                ),
                "source_rows": tuple(
                    (_descriptor_path(row[2]), row[3])
                    for row in kwargs["directory_entries"]
                ),
                "leaf": _descriptor_path(kwargs["leaf_fd"]),
                "leaf_fingerprint": kwargs["leaf_fingerprint"],
            }
        )

    def read(descriptor: int, length: int) -> bytes:
        events.append("read")
        return original_read(descriptor, length)

    monkeypatch.setattr(v5, "_revalidate_open_chain", validate)
    monkeypatch.setattr(os, "read", read)
    assert v5._read_frozen_source_index(repo) == payload

    assert events[0] == "validate"
    assert events[-1] == "validate"
    assert events.count("validate") == 2
    assert all(event == "read" for event in events[1:-1])
    assert snapshots[0] == snapshots[1]
    snapshot = snapshots[0]
    assert snapshot["filesystem_root"] == Path(repo.anchor)
    assert snapshot["root"] == repo
    assert snapshot["leaf"] == source
    assert all(
        len(fingerprint) == 7
        for fingerprint in (
            snapshot["anchor_fingerprint"],
            snapshot["root_fingerprint"],
            snapshot["leaf_fingerprint"],
            *(fingerprint for _path, fingerprint in snapshot["root_rows"]),
            *(fingerprint for _path, fingerprint in snapshot["source_rows"]),
        )
    )

    expected_root_paths = tuple(
        Path(repo.anchor).joinpath(*repo.parts[1 : index + 1])
        for index in range(1, len(repo.parts))
    )
    expected_source_paths = (repo / "catalog", repo / "catalog" / "nested")
    assert tuple(path for path, _fingerprint in snapshot["root_rows"]) == (
        expected_root_paths
    )
    assert tuple(path for path, _fingerprint in snapshot["source_rows"]) == (
        expected_source_paths
    )
    for path, fingerprint in (
        *snapshot["root_rows"],
        *snapshot["source_rows"],
        (source, snapshot["leaf_fingerprint"]),
    ):
        assert fingerprint == v5._file_fingerprint(
            path.stat(follow_symlinks=False)
        )


@pytest.mark.parametrize(
    "target",
    (
        "ancestor_a",
        "ancestor_b",
        "repo",
        "catalog",
        "nested",
        "leaf",
    ),
)
def test_v5_independent_rejects_each_owned_chain_fingerprint_change_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch)
    paths = {
        "ancestor_a": repo.parent.parent,
        "ancestor_b": repo.parent,
        "repo": repo,
        "catalog": repo / "catalog",
        "nested": source.parent,
        "leaf": source,
    }
    original_read = os.read
    changed = False

    def mutate_then_read(descriptor: int, length: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            _advance_fingerprint(paths[target])
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "read", mutate_then_read)
    with pytest.raises(v5.RawSupervisionPlanError, match="changed"):
        v5._read_frozen_source_index(repo)
    assert changed


@pytest.mark.parametrize("target", ("ancestor_a", "repo", "nested", "leaf"))
def test_v5_independent_rejects_chain_change_before_first_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch)
    paths = {
        "ancestor_a": repo.parent.parent,
        "repo": repo,
        "nested": source.parent,
        "leaf": source,
    }
    original_validate = v5._revalidate_open_chain
    original_read = os.read
    validations = 0
    reads = 0

    def mutate_then_validate(**kwargs: Any) -> None:
        nonlocal validations
        validations += 1
        if validations == 1:
            _advance_fingerprint(paths[target])
        original_validate(**kwargs)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(v5, "_revalidate_open_chain", mutate_then_validate)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v5.RawSupervisionPlanError, match="changed"):
        v5._read_frozen_source_index(repo)
    assert (validations, reads) == (1, 0)


def test_v5_independent_rechecks_filesystem_root_named_entry_and_descriptor(
    tmp_path: Path,
) -> None:
    anchor = tmp_path / "synthetic_root"
    repo = anchor / "repo"
    leaf = repo / "source.jsonl"
    repo.mkdir(parents=True)
    leaf.write_bytes(b"source\n")
    anchor_fd = os.open(anchor, v5._directory_flags())
    repo_fd = os.open("repo", v5._directory_flags(), dir_fd=anchor_fd)
    leaf_fd = os.open("source.jsonl", v5._file_flags(), dir_fd=repo_fd)
    try:
        anchor_fingerprint = v5._file_fingerprint(os.fstat(anchor_fd))
        repo_fingerprint = v5._file_fingerprint(os.fstat(repo_fd))
        leaf_fingerprint = v5._file_fingerprint(os.fstat(leaf_fd))
        arguments = {
            "filesystem_root": anchor,
            "anchor_fd": anchor_fd,
            "anchor_fingerprint": anchor_fingerprint,
            "root": repo,
            "root_fd": repo_fd,
            "root_fingerprint": repo_fingerprint,
            "root_directory_entries": (
                (anchor_fd, "repo", repo_fd, repo_fingerprint),
            ),
            "directory_entries": (),
            "leaf_parent_fd": repo_fd,
            "leaf_name": "source.jsonl",
            "leaf_fd": leaf_fd,
            "leaf_fingerprint": leaf_fingerprint,
        }
        v5._revalidate_open_chain(**arguments)
        _advance_fingerprint(anchor)
        with pytest.raises(
            v5.RawSupervisionPlanError,
            match="filesystem root directory changed",
        ):
            v5._revalidate_open_chain(**arguments)
    finally:
        os.close(leaf_fd)
        os.close(repo_fd)
        os.close(anchor_fd)


@pytest.mark.parametrize(
    "replacement",
    (
        "ancestor_symlink",
        "source_parent_symlink",
        "source_parent_identity",
        "leaf_hardlink",
        "leaf_same_inode_relink",
        "opened_leaf_hardlink",
    ),
)
def test_v5_independent_closes_published_v2_v3_continuity_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch)
    original_open = os.open
    original_read = os.read
    exercised = False
    reads = 0

    def replace_during_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal exercised
        candidate = Path(os.fsdecode(os.fspath(path)))
        if not exercised and replacement == "ancestor_symlink" and (
            candidate.name == "ancestor_a" and dir_fd is not None
        ):
            exercised = True
            target = repo.parent.parent
            moved = tmp_path / "moved_ancestor_a"
            target.rename(moved)
            target.symlink_to(moved, target_is_directory=True)
            try:
                return original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                target.unlink()
                moved.rename(target)
        if not exercised and replacement in {
            "source_parent_symlink",
            "source_parent_identity",
        } and candidate.name == "nested" and dir_fd is not None:
            exercised = True
            parent = source.parent
            moved = tmp_path / "moved_nested"
            parent.rename(moved)
            if replacement == "source_parent_symlink":
                parent.symlink_to(moved, target_is_directory=True)
            else:
                parent.mkdir()
                (parent / source.name).write_bytes(b'{"scene_id":"synthetic"}\n')
        if not exercised and replacement in {
            "leaf_hardlink",
            "leaf_same_inode_relink",
        } and candidate.name == source.name and dir_fd is not None:
            exercised = True
            moved = tmp_path / "moved_source.jsonl"
            source.rename(moved)
            os.link(moved, source)
            if replacement == "leaf_same_inode_relink":
                moved.unlink()
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if not exercised and replacement == "opened_leaf_hardlink" and (
            candidate.name == source.name and dir_fd is not None
        ):
            exercised = True
            os.link(source, tmp_path / "opened_leaf_alias.jsonl")
        return descriptor

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", replace_during_open)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v5.RawSupervisionPlanError):
        v5._read_frozen_source_index(repo)
    assert exercised
    assert reads == 0


@pytest.mark.parametrize("location", ("parent", "leaf"))
def test_v5_independent_closes_the_published_v1_symlink_escape(
    tmp_path: Path,
    location: str,
) -> None:
    repo = tmp_path / "repo"
    row = _synthetic_source_row(repo)
    source_fields = (
        "frames_jsonl_path",
        "scene_manifest_path",
        "render_plan_path",
        "render_summary_path",
    )
    for field in source_fields:
        path = Path(row[field])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic metadata fixture\n")

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_frame = outside / "frames.jsonl"
    outside_frame.write_bytes(b"outside synthetic fixture\n")
    if location == "parent":
        alias = repo / "metadata_alias"
        alias.symlink_to(outside, target_is_directory=True)
        row["frames_jsonl_path"] = str(alias / outside_frame.name)
    else:
        frame = Path(row["frames_jsonl_path"])
        frame.unlink()
        frame.symlink_to(outside_frame)

    with pytest.raises(v5.RawSupervisionPlanError, match="symlinked component"):
        v5.plan_development_source_inventory(
            _synthetic_plan(),
            [row],
            repo_root=repo,
            enforce_frozen_hashes=False,
        )


@pytest.mark.parametrize("kind", ("directory", "fifo", "symlink", "hardlink"))
def test_v5_independent_rejects_every_published_leaf_kind_before_read(
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
        target = tmp_path / "leaf_target.jsonl"
        target.write_bytes(b"target\n")
        source.symlink_to(target)
    else:
        target = tmp_path / "leaf_target.jsonl"
        target.write_bytes(b"target\n")
        os.link(target, source)

    original_read = os.read
    reads = 0

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v5.RawSupervisionPlanError):
        v5._read_frozen_source_index(repo)
    assert reads == 0


def test_v5_independent_reconstructs_science_with_exact_bounded_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    inspected: list[str] = []
    original_path_open = Path.open
    original_os_open = os.open
    original_builtin_open = builtins.open
    original_validate = v2._validate_referenced_path

    def record_regular(path: Any, *, dir_fd: int | None = None) -> None:
        candidate = Path(os.fsdecode(os.fspath(path)))
        if not candidate.is_absolute() and dir_fd is not None:
            candidate = _descriptor_path(dir_fd) / candidate
        resolved = candidate.resolve(strict=True)
        assert resolved in EXPECTED_OPEN_COUNTS
        opened.append(resolved)

    def traced_path_open(path: Path, *args: Any, **kwargs: Any):
        record_regular(path)
        return original_path_open(path, *args, **kwargs)

    def traced_builtin_open(path: Any, *args: Any, **kwargs: Any):
        record_regular(path)
        return original_builtin_open(path, *args, **kwargs)

    def traced_os_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_os_open(path, flags, mode, dir_fd=dir_fd)
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            record_regular(path, dir_fd=dir_fd)
        return descriptor

    def traced_validate(value: object, **kwargs: Any):
        inspected.append(str(value))
        return original_validate(value, **kwargs)

    monkeypatch.setattr(Path, "open", traced_path_open)
    monkeypatch.setattr(builtins, "open", traced_builtin_open)
    monkeypatch.setattr(os, "open", traced_os_open)
    monkeypatch.setattr(v2, "_validate_referenced_path", traced_validate)

    plan = v5.load_frozen_development_metadata(ROOT)
    inventory = v5.load_frozen_development_source_inventory(ROOT, plan)

    assert Counter(opened) == EXPECTED_OPEN_COUNTS
    assert sum(EXPECTED_OPEN_COUNTS.values()) == 10
    assert len(plan.pairs) == 5172
    assert plan.value["endpoint_instance_count"] == 10344
    assert len(plan.endpoints) == 9460
    assert len(inventory.records) == 88
    assert plan.value["content_sha256"] == (
        "8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3"
    )
    assert plan.value["ordered_pair_sha256"] == (
        "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
    )
    assert plan.value["ordered_endpoint_sha256"] == (
        "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
    )
    assert inventory.hashes == EXPECTED_INVENTORY_HASHES
    assert not any(plan.value["licenses"].values())
    assert len(inspected) == 704
    assert len(set(inspected)) == 352
    assert set(Counter(inspected).values()) == {2}
    assert all("g2_evaluation" not in path for path in inspected)
    assert all(
        value == 0
        for key, value in inventory.access_ledger.items()
        if key.endswith("_opens")
    )
    assert not any("g2_evaluation" in str(path) for path in opened)

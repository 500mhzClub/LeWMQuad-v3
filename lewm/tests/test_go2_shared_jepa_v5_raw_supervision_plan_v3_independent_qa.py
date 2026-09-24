"""Independent software-conformance QA for raw-supervision metadata plan V3.

Only frozen allowlisted metadata and temporary synthetic trees are opened.
Referenced source payloads and excluded G2 paths remain unopened.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan as v1
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v2 as v2
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v3 as v3


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INDEX = ROOT / v3.SOURCE_INDEX_RELATIVE_PATH
DATASET_MANIFEST = ROOT / v1.DATASET_MANIFEST_RELATIVE_PATH
DATASET_ROWS = ROOT / v1.DATASET_ROWS_RELATIVE_PATH
SIDECAR_MANIFEST = ROOT / v1.SIDECAR_MANIFEST_RELATIVE_PATH
ROLE_FILES = {
    role: SIDECAR_MANIFEST.parent / f"{role}.jsonl"
    for role in v3.DEVELOPMENT_ROLES
}
SOURCE_FIELDS = (
    "frames_jsonl_path",
    "scene_manifest_path",
    "render_plan_path",
    "render_summary_path",
)
ROLE_PAIR_COUNTS = {
    "train": 4262,
    "checkpoint_selection": 495,
    "probability_calibration": 415,
}
ROLE_ENDPOINT_COUNTS = {
    "train": 7777,
    "checkpoint_selection": 924,
    "probability_calibration": 759,
}
ROLE_SCENE_COUNTS = {
    "train": 72,
    "checkpoint_selection": 8,
    "probability_calibration": 8,
}
EXPECTED_INVENTORY_HASHES = {
    "scene_role": "f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b",
    "frames": "7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d",
    "manifests": "2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5",
    "plans": "0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4",
    "summaries": "bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a",
}
FROZEN_HASHES = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan.py": (
        "e7ab8727b0d93d3fd8f9e2a3ab5cfdc4f9199e18b8d0a7f5a1f7dc0b5dc0c18e"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan.py": (
        "e2b49e660292ff99a7794ff3c761f9563a9e182b2889aec3a4e94b835c4be56c"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v2.py": (
        "44641fb97d6172342a3129262c6a0047cae14048c5174a5ed9418420080e1def"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v2.py": (
        "263b91d04029df172c9c43c79f4d81c4dd887672ee88b9e80555c2f240dc3cd7"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v2_independent_qa.py": (
        "3d9a8203b4cfc7aa208b6b319932aaa5d912ac337d02fa436663855bbd090b0c"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v2_"
    "independent_review_2026-07-13.md": (
        "376a8a761b28502a8495b51554960c046e30f4dd1c7f7d7697d7e3c1b407c65a"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v3.py": (
        "0adc6bfa0ea76484f9491a2bbde68f072fd4b908ca8cda7b112c4a32fe481247"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v3.py": (
        "f1f0bff99f997e94677b542eb35c76332c8f79cf0ff88d5474011ed007f6aa78"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v3_"
    "author_handoff_2026-07-13.md": (
        "66f55b3489c35cbbd5413f4f223942463fad08607564f08994d3f1887bac8160"
    ),
}


def _bind_index(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *,
    nested_root: bool = False,
) -> tuple[Path, Path, bytes]:
    repo = base / "anchor" / "repo" if nested_root else base / "repo"
    source = repo / "index" / "source.jsonl"
    source.parent.mkdir(parents=True)
    payload = b'{"scene_id":"synthetic"}\n'
    source.write_bytes(payload)
    monkeypatch.setattr(module, "SOURCE_INDEX_RELATIVE_PATH", "index/source.jsonl")
    monkeypatch.setattr(
        module,
        "SOURCE_INDEX_FILE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return repo, source, payload


def _replace_published_v2_case(
    *,
    source: Path,
    base: Path,
    replacement: str,
) -> None:
    outside = base / f"outside-{replacement}"
    outside.mkdir()
    if replacement == "parent-symlink":
        parent = source.parent
        moved = outside / "moved-index"
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    else:
        moved = outside / "moved-source.jsonl"
        source.rename(moved)
        os.link(moved, source)


def test_v3_independent_freezes_exact_review_inputs() -> None:
    assert {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in FROZEN_HASHES
    } == FROZEN_HASHES


@pytest.mark.parametrize("replacement", ("parent-symlink", "leaf-hardlink"))
def test_v3_independent_reproduces_both_v2_continuity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    repo, source, payload = _bind_index(tmp_path, monkeypatch, v2)
    original_open = os.open
    replaced = False

    def replace_before_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and Path(path) == source:
            replaced = True
            _replace_published_v2_case(
                source=source,
                base=tmp_path,
                replacement=replacement,
            )
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_open)
    assert v2._read_frozen_source_index(repo) == payload
    assert replaced


@pytest.mark.parametrize("replacement", ("parent-symlink", "leaf-hardlink"))
def test_v3_independent_closes_both_published_v2_continuity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch, v3)
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_before_leaf_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == source.name and dir_fd is not None:
            replaced = True
            _replace_published_v2_case(
                source=source,
                base=tmp_path,
                replacement=replacement,
            )
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", replace_before_leaf_open)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v3.RawSupervisionPlanError):
        v3._read_frozen_source_index(repo)
    assert replaced and reads == 0


def test_v3_independent_requires_repo_root_open_continuity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source, _payload = _bind_index(
        tmp_path,
        monkeypatch,
        v3,
        nested_root=True,
    )
    anchor = repo.parent
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_root_ancestor_during_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and Path(path) == repo and dir_fd is None:
            replaced = True
            moved = tmp_path / "moved-anchor"
            anchor.rename(moved)
            anchor.symlink_to(moved, target_is_directory=True)
            assert repo.resolve(strict=True) != repo
            try:
                descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                anchor.unlink()
                moved.rename(anchor)
            return descriptor
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", replace_root_ancestor_during_open)
    monkeypatch.setattr(os, "read", count_read)
    rejected = False
    try:
        v3._read_frozen_source_index(repo)
    except v3.RawSupervisionPlanError:
        rejected = True
    assert {
        "replacement_exercised": replaced,
        "rejected": rejected,
        "read_calls": reads,
    } == {
        "replacement_exercised": True,
        "rejected": True,
        "read_calls": 0,
    }


def test_v3_independent_binds_full_preopen_leaf_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch, v3)
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0
    before = source.stat(follow_symlinks=False)
    before_fingerprint = v3._file_fingerprint(before)

    def relink_same_inode_before_open(
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
            relinked = source.stat(follow_symlinks=False)
            assert v3._entry_identity(relinked) == v3._entry_identity(before)
            assert relinked.st_nlink == 1
            assert v3._file_fingerprint(relinked) != before_fingerprint
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", relink_same_inode_before_open)
    monkeypatch.setattr(os, "read", count_read)
    rejected = False
    try:
        v3._read_frozen_source_index(repo)
    except v3.RawSupervisionPlanError:
        rejected = True
    assert {
        "replacement_exercised": replaced,
        "rejected": rejected,
        "read_calls": reads,
    } == {
        "replacement_exercised": True,
        "rejected": True,
        "read_calls": 0,
    }


def test_v3_independent_rejects_component_identity_replacement_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch, v3)
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_component(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == source.parent.name and dir_fd is not None:
            replaced = True
            source.parent.rename(tmp_path / "validated-index")
            source.parent.mkdir()
            (source.parent / source.name).write_bytes(b'{"scene_id":"synthetic"}\n')
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", replace_component)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v3.RawSupervisionPlanError, match="identity changed"):
        v3._read_frozen_source_index(repo)
    assert replaced and reads == 0


def test_v3_independent_rechecks_opened_leaf_link_count_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch, v3)
    original_open = os.open
    original_read = os.read
    linked = False
    reads = 0

    def add_link_after_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal linked
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if not linked and path == source.name and dir_fd is not None:
            linked = True
            os.link(source, tmp_path / "opened-leaf-alias.jsonl")
        return descriptor

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", add_link_after_open)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v3.RawSupervisionPlanError, match="hard-link alias"):
        v3._read_frozen_source_index(repo)
    assert linked and reads == 0


@pytest.mark.parametrize("kind", ("directory", "fifo", "symlink", "hardlink"))
def test_v3_independent_rejects_nonregular_and_linked_leaf_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    repo, source, _payload = _bind_index(tmp_path, monkeypatch, v3)
    source.unlink()
    if kind == "directory":
        source.mkdir()
    elif kind == "fifo":
        os.mkfifo(source)
    elif kind == "symlink":
        target = tmp_path / "target.jsonl"
        target.write_bytes(b"{}\n")
        source.symlink_to(target)
    else:
        target = tmp_path / "target.jsonl"
        target.write_bytes(b"{}\n")
        os.link(target, source)

    reads = 0
    original_read = os.read

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v3.RawSupervisionPlanError):
        v3._read_frozen_source_index(repo)
    assert reads == 0


def test_v3_independent_reconstructs_counts_hashes_and_closed_licenses() -> None:
    raw_rows = [json.loads(line) for line in DATASET_ROWS.read_bytes().splitlines()]
    development = [
        row for row in raw_rows if row["dataset_role"] in v3.DEVELOPMENT_ROLES
    ]
    role_pairs = Counter(row["dataset_role"] for row in development)
    role_scenes = {
        role: {row["scene_id"] for row in development if row["dataset_role"] == role}
        for role in v3.DEVELOPMENT_ROLES
    }
    endpoints: dict[tuple[Any, ...], str] = {}
    for row in development:
        for side in ("current", "next"):
            identity = (
                row["dataset_role"],
                row["scene_id"],
                row["episode_id"],
                row["env_index"],
                row[f"{side}_episode_step"],
                row[f"{side}_frame_index"],
                row[f"{side}_timestamp_ns"],
                row[f"{side}_image_sha256"],
            )
            endpoints.setdefault(identity, row["dataset_role"])

    assert dict(role_pairs) == ROLE_PAIR_COUNTS
    assert {role: len(scenes) for role, scenes in role_scenes.items()} == (
        ROLE_SCENE_COUNTS
    )
    assert Counter(endpoints.values()) == Counter(ROLE_ENDPOINT_COUNTS)
    assert (len(development), 2 * len(development), len(endpoints)) == (
        5172,
        10344,
        9460,
    )

    plan = v3.load_frozen_development_metadata(ROOT)
    inventory = v3.load_frozen_development_source_inventory(ROOT, plan)
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
        count == 0
        for key, count in inventory.access_ledger.items()
        if key.endswith("_opens")
    )


def test_v3_independent_opens_exact_metadata_allowlist_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original_path_open = Path.open
    original_os_open = os.open

    def traced_path_open(path: Path, *args: Any, **kwargs: Any):
        opened.append(path.resolve(strict=True))
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
            opened.append(candidate.resolve(strict=True))
        return descriptor

    monkeypatch.setattr(Path, "open", traced_path_open)
    monkeypatch.setattr(os, "open", traced_os_open)
    plan = v3.load_frozen_development_metadata(ROOT)
    inventory = v3.load_frozen_development_source_inventory(ROOT, plan)
    assert Counter(opened) == Counter(
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
    assert SIDECAR_MANIFEST.parent / "g2_evaluation.jsonl" not in opened
    referenced = {
        Path(record[section]["path"])
        for record in inventory.records
        for section in ("frames", "scene_manifest", "render_plan", "render_summary")
    }
    assert not referenced.intersection(opened)


def test_v3_independent_never_inspects_excluded_g2_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = v3.load_frozen_development_metadata(ROOT)
    rows = [json.loads(line) for line in SOURCE_INDEX.read_bytes().splitlines()]
    selected_scenes = {
        endpoint["identity"]["scene_id"] for endpoint in plan.endpoints
    }
    excluded_scenes = {row["scene_id"] for row in rows} - selected_scenes
    assert (len(selected_scenes), len(excluded_scenes)) == (88, 8)
    selected_paths = {
        row[field]
        for row in rows
        if row["scene_id"] in selected_scenes
        for field in SOURCE_FIELDS
    }
    forbidden_root = ROOT / ".generated" / "v5-v3-independent-forbidden-g2"
    for row in rows:
        if row["scene_id"] in excluded_scenes:
            for field in SOURCE_FIELDS:
                row[field] = str(forbidden_root / row["scene_id"] / field)

    validated: list[str] = []
    original = v2._validate_referenced_path

    def traced(value: object, **kwargs: Any):
        assert isinstance(value, str)
        assert not value.startswith(str(forbidden_root))
        validated.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(v2, "_validate_referenced_path", traced)
    inventory = v3.plan_development_source_inventory(plan, rows, repo_root=ROOT)
    assert len(inventory.records) == 88
    assert inventory.hashes == EXPECTED_INVENTORY_HASHES
    assert Counter(validated) == Counter({path: 2 for path in selected_paths})
    assert len(validated) == 88 * 4 * 2

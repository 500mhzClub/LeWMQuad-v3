from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v2 as v2
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v3 as v3


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INDEX = ROOT / v3.SOURCE_INDEX_RELATIVE_PATH
DATASET_MANIFEST = ROOT / v2._v1.DATASET_MANIFEST_RELATIVE_PATH
DATASET_ROWS = ROOT / v2._v1.DATASET_ROWS_RELATIVE_PATH
SIDECAR_MANIFEST = ROOT / v2._v1.SIDECAR_MANIFEST_RELATIVE_PATH
ROLE_FILES = {
    role: SIDECAR_MANIFEST.parent / f"{role}.jsonl"
    for role in v3.DEVELOPMENT_ROLES
}
EXPECTED_INVENTORY_HASHES = {
    "scene_role": "f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b",
    "frames": "7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d",
    "manifests": "2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5",
    "plans": "0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4",
    "summaries": "bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a",
}
SOURCE_FIELDS = (
    "frames_jsonl_path",
    "scene_manifest_path",
    "render_plan_path",
    "render_summary_path",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _synthetic_plan() -> v3.DevelopmentRawSupervisionPlan:
    return v3.DevelopmentRawSupervisionPlan(
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


def _source_row(repo: Path) -> dict[str, Any]:
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
            "frames_jsonl_file_sha256": _sha("frames"),
            "scene_manifest_file_sha256": _sha("manifest-file"),
            "scene_manifest_sha256": _sha("manifest-content"),
            "render_plan_file_sha256": _sha("plan"),
            "render_summary_file_sha256": _sha("summary"),
        },
    }


def _repo_with_sources(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    row = _source_row(repo)
    (repo / "metadata" / "scene_a").mkdir(parents=True)
    for field in SOURCE_FIELDS:
        Path(row[field]).write_bytes(b"fixture\n")
    return repo, row


def _bind_synthetic_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, bytes]:
    repo = tmp_path / "repo"
    source = repo / "index" / "source.jsonl"
    source.parent.mkdir(parents=True)
    payload = b'{"scene_id":"synthetic"}\n'
    source.write_bytes(payload)
    monkeypatch.setattr(v3, "SOURCE_INDEX_RELATIVE_PATH", "index/source.jsonl")
    monkeypatch.setattr(
        v3,
        "SOURCE_INDEX_FILE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return repo, source, payload


def test_v3_preserves_v1_v2_and_block_evidence_bytes() -> None:
    expected = {
        ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v2.py": (
            "44641fb97d6172342a3129262c6a0047cae14048c5174a5ed9418420080e1def"
        ),
        ROOT / "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v2.py": (
            "263b91d04029df172c9c43c79f4d81c4dd887672ee88b9e80555c2f240dc3cd7"
        ),
        ROOT
        / "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v2_author_handoff_2026-07-13.md": (
            "a6629b32f7ee2266a57d9f705d52d6b2136c391ef5eb135937c4195cbe9f24fb"
        ),
        ROOT
        / "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v2_independent_qa.py": (
            "3d9a8203b4cfc7aa208b6b319932aaa5d912ac337d02fa436663855bbd090b0c"
        ),
        ROOT
        / "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v2_independent_review_2026-07-13.md": (
            "376a8a761b28502a8495b51554960c046e30f4dd1c7f7d7697d7e3c1b407c65a"
        ),
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_v3_preserves_exact_scientific_identities_counts_and_licenses() -> None:
    v2_plan = v2.load_frozen_development_metadata(ROOT)
    v3_plan = v3.load_frozen_development_metadata(ROOT)
    v2_inventory = v2.load_frozen_development_source_inventory(ROOT, v2_plan)
    v3_inventory = v3.load_frozen_development_source_inventory(ROOT, v3_plan)
    assert v3_plan == v2_plan
    assert v3_inventory == v2_inventory
    assert len(v3_plan.pairs) == 5172
    assert v3_plan.value["endpoint_instance_count"] == 10344
    assert len(v3_plan.endpoints) == 9460
    assert len(v3_inventory.records) == 88
    assert v3_inventory.hashes == EXPECTED_INVENTORY_HASHES
    assert v3_plan.value["content_sha256"] == (
        "8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3"
    )
    assert v3_plan.value["ordered_pair_sha256"] == (
        "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
    )
    assert v3_plan.value["ordered_endpoint_sha256"] == (
        "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
    )
    assert not any(v3_plan.value["licenses"].values())


def test_v3_source_index_uses_relative_nofollow_descriptor_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source, payload = _bind_synthetic_index(tmp_path, monkeypatch)
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
    assert v3._read_frozen_source_index(repo) == payload
    assert calls[0][0] == repo and calls[0][1] is None
    assert calls[1][0] == "index" and calls[1][1] is not None
    assert calls[2][0] == "source.jsonl" and calls[2][1] is not None
    assert len(calls) == 3
    assert all(flags & os.O_NOFOLLOW for _path, _dir_fd, flags in calls)
    assert calls[0][2] & os.O_DIRECTORY
    assert calls[1][2] & os.O_DIRECTORY


@pytest.mark.parametrize("replacement", ("parent-symlink", "leaf-hardlink"))
def test_v3_rejects_v2_post_validation_replacements_before_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    repo, source, _payload = _bind_synthetic_index(tmp_path, monkeypatch)
    parent = source.parent
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_at_leaf_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == source.name and dir_fd is not None:
            replaced = True
            outside = tmp_path / f"outside-{replacement}"
            outside.mkdir()
            if replacement == "parent-symlink":
                moved = outside / "moved-index"
                parent.rename(moved)
                parent.symlink_to(moved, target_is_directory=True)
            else:
                moved = outside / "moved-source.jsonl"
                source.rename(moved)
                os.link(moved, source)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", replace_at_leaf_open)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v3.RawSupervisionPlanError, match="changed|hard-link"):
        v3._read_frozen_source_index(repo)
    assert replaced
    assert reads == 0


def test_v3_rejects_parent_replacement_before_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_synthetic_index(tmp_path, monkeypatch)
    parent = source.parent
    original_open = os.open
    replaced = False

    def replace_parent(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == parent.name and dir_fd is not None:
            replaced = True
            moved = tmp_path / "moved-index"
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_parent)
    with pytest.raises(v3.RawSupervisionPlanError, match="changed during open"):
        v3._read_frozen_source_index(repo)
    assert replaced


def test_v3_rejects_leaf_different_inode_substitution_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, payload = _bind_synthetic_index(tmp_path, monkeypatch)
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_leaf(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == source.name and dir_fd is not None:
            replaced = True
            source.rename(source.with_name("validated-source.jsonl"))
            source.write_bytes(payload)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", replace_leaf)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v3.RawSupervisionPlanError, match="identity changed"):
        v3._read_frozen_source_index(repo)
    assert replaced
    assert reads == 0


def test_v3_rejects_leaf_fifo_replacement_without_blocking_or_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_synthetic_index(tmp_path, monkeypatch)
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_leaf_with_fifo(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == source.name and dir_fd is not None:
            replaced = True
            source.unlink()
            os.mkfifo(source)
            assert flags & os.O_NONBLOCK
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "open", replace_leaf_with_fifo)
    monkeypatch.setattr(os, "read", count_read)
    with pytest.raises(v3.RawSupervisionPlanError, match="regular file"):
        v3._read_frozen_source_index(repo)
    assert replaced
    assert reads == 0


@pytest.mark.parametrize("kind", ("missing", "directory", "fifo", "symlink", "hardlink"))
def test_v3_rejects_static_source_index_aliases_and_nonregular_leaves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    repo, source, _payload = _bind_synthetic_index(tmp_path, monkeypatch)
    source.unlink()
    if kind == "directory":
        source.mkdir()
    elif kind == "fifo":
        os.mkfifo(source)
    elif kind == "symlink":
        target = tmp_path / "outside-index.jsonl"
        target.write_bytes(b'{}\n')
        source.symlink_to(target)
    elif kind == "hardlink":
        target = tmp_path / "outside-index.jsonl"
        target.write_bytes(b'{}\n')
        os.link(target, source)
    expected = "exist|regular|symlink|hard-link"
    with pytest.raises(v3.RawSupervisionPlanError, match=expected):
        v3._read_frozen_source_index(repo)


@pytest.mark.parametrize(
    "spelling",
    ("/absolute/index.jsonl", "index/../index/source.jsonl", "index//source.jsonl", "index/./source.jsonl"),
)
def test_v3_rejects_noncanonical_source_index_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    repo, _source, _payload = _bind_synthetic_index(tmp_path, monkeypatch)
    monkeypatch.setattr(v3, "SOURCE_INDEX_RELATIVE_PATH", spelling)
    with pytest.raises(v3.RawSupervisionPlanError, match="canonical and relative"):
        v3._read_frozen_source_index(repo)


@pytest.mark.parametrize("kind", ("parent-symlink", "leaf-symlink", "hardlink", "fifo", "missing"))
def test_v3_retains_v2_referenced_metadata_path_defenses(
    tmp_path: Path,
    kind: str,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    original = Path(row["frames_jsonl_path"])
    if kind == "parent-symlink":
        target = tmp_path / "outside-parent"
        target.mkdir()
        (target / original.name).write_bytes(b"outside\n")
        alias = repo / "parent-alias"
        alias.symlink_to(target, target_is_directory=True)
        row["frames_jsonl_path"] = str(alias / original.name)
    elif kind == "leaf-symlink":
        alias = original.with_name("leaf-alias.jsonl")
        alias.symlink_to(original)
        row["frames_jsonl_path"] = str(alias)
    elif kind == "hardlink":
        alias = original.with_name("hardlink.jsonl")
        os.link(original, alias)
        row["frames_jsonl_path"] = str(alias)
    else:
        replacement = original.with_name(kind)
        if kind == "fifo":
            os.mkfifo(replacement)
        row["frames_jsonl_path"] = str(replacement)
    with pytest.raises(v3.RawSupervisionPlanError):
        v3.plan_development_source_inventory(
            _synthetic_plan(),
            [row],
            repo_root=repo,
            enforce_frozen_hashes=False,
        )


def test_v3_exact_open_trace_remains_metadata_only(
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
    expected = Counter(
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
    assert Counter(opened) == expected
    assert SIDECAR_MANIFEST.parent / "g2_evaluation.jsonl" not in opened
    assert all(
        value == 0
        for key, value in inventory.access_ledger.items()
        if key.endswith("_opens")
    )


def test_v3_rejects_source_index_mutation_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source, _payload = _bind_synthetic_index(tmp_path, monkeypatch)
    original_read = os.read
    mutated = False

    def mutate_then_read(descriptor: int, length: int) -> bytes:
        nonlocal mutated
        if not mutated:
            mutated = True
            with source.open("ab") as handle:
                handle.write(b"mutation\n")
        return original_read(descriptor, length)

    monkeypatch.setattr(os, "read", mutate_then_read)
    with pytest.raises(v3.RawSupervisionPlanError, match="changed while read"):
        v3._read_frozen_source_index(repo)
    assert mutated

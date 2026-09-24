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

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan as v1
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INDEX = ROOT / v2.SOURCE_INDEX_RELATIVE_PATH


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _synthetic_plan() -> v2.DevelopmentRawSupervisionPlan:
    return v2.DevelopmentRawSupervisionPlan(
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
    base = repo / "metadata" / "scene_a"
    base.mkdir(parents=True)
    row = _source_row(repo)
    for field in (
        "frames_jsonl_path",
        "scene_manifest_path",
        "render_plan_path",
        "render_summary_path",
    ):
        Path(row[field]).touch()
    return repo, row


def _plan(row: dict[str, Any], repo: Path) -> v2.DevelopmentSourceInventory:
    return v2.plan_development_source_inventory(
        _synthetic_plan(),
        [row],
        repo_root=repo,
        enforce_frozen_hashes=False,
    )


def test_v2_preserves_every_frozen_v1_scientific_identity_and_count() -> None:
    v1_plan = v1.load_frozen_development_metadata(ROOT)
    v2_plan = v2.load_frozen_development_metadata(ROOT)
    assert v2_plan == v1_plan
    v1_inventory = v1.load_frozen_development_source_inventory(ROOT, v1_plan)
    v2_inventory = v2.load_frozen_development_source_inventory(ROOT, v2_plan)
    assert v2_inventory == v1_inventory
    assert len(v2_plan.pairs) == 5172
    assert v2_plan.value["endpoint_instance_count"] == 10344
    assert len(v2_plan.endpoints) == 9460
    assert len(v2_inventory.records) == 88
    assert v2_inventory.hashes == v2.SOURCE_INVENTORY_SHA256
    assert not any(v2_plan.value["licenses"].values())


def test_v2_frozen_loader_opens_only_the_exact_metadata_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original_path_open = Path.open
    original_os_open = os.open

    def traced_path_open(path: Path, *args: Any, **kwargs: Any):
        opened.append(path.resolve(strict=True))
        return original_path_open(path, *args, **kwargs)

    def traced_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
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
    plan = v2.load_frozen_development_metadata(ROOT)
    inventory = v2.load_frozen_development_source_inventory(ROOT, plan)

    sidecar_manifest = ROOT / v1.SIDECAR_MANIFEST_RELATIVE_PATH
    expected = Counter(
        {
            ROOT / v1.DATASET_MANIFEST_RELATIVE_PATH: 1,
            ROOT / v1.DATASET_ROWS_RELATIVE_PATH: 1,
            sidecar_manifest: 1,
            sidecar_manifest.parent / "train.jsonl": 2,
            sidecar_manifest.parent / "checkpoint_selection.jsonl": 2,
            sidecar_manifest.parent / "probability_calibration.jsonl": 2,
            SOURCE_INDEX: 1,
        }
    )
    assert Counter(opened) == expected
    assert sidecar_manifest.parent / "g2_evaluation.jsonl" not in opened
    assert all(
        value == 0
        for key, value in inventory.access_ledger.items()
        if key.endswith("_opens")
    )


def test_v2_inventory_reads_no_referenced_payload_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    referenced = {
        Path(row[field])
        for field in (
            "frames_jsonl_path",
            "scene_manifest_path",
            "render_plan_path",
            "render_summary_path",
        )
    }
    original_read_bytes = Path.read_bytes
    original_open = Path.open
    original_os_open = os.open

    def guarded_read_bytes(path: Path) -> bytes:
        assert path not in referenced, f"referenced payload read: {path}"
        return original_read_bytes(path)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        assert path not in referenced, f"referenced payload opened: {path}"
        return original_open(path, *args, **kwargs)

    def guarded_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        candidate = Path(os.fsdecode(os.fspath(path)))
        assert candidate not in referenced, f"referenced payload opened: {candidate}"
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    inventory = _plan(row, repo)
    assert len(inventory.records) == 1
    assert all(
        value == 0
        for key, value in inventory.access_ledger.items()
        if key.endswith("_opens")
    )


@pytest.mark.parametrize(
    "form",
    ("relative", "dot", "dotdot", "trailing", "double-root", "nul"),
)
def test_v2_rejects_noncanonical_or_nonabsolute_paths(
    tmp_path: Path,
    form: str,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    canonical = Path(row["frames_jsonl_path"])
    if form == "relative":
        changed = "metadata/scene_a/frames.jsonl"
    elif form == "dot":
        changed = str(canonical.parent) + "/./" + canonical.name
    elif form == "dotdot":
        changed = str(canonical.parent) + "/other/../" + canonical.name
    elif form == "double-root":
        changed = "/" + str(canonical)
    elif form == "nul":
        changed = str(canonical) + "\x00"
    else:
        changed = str(canonical) + "/"
    row["frames_jsonl_path"] = changed
    with pytest.raises(v2.RawSupervisionPlanError, match="canonical and absolute"):
        _plan(row, repo)


def test_v2_rejects_lexical_and_resolved_repository_escapes(tmp_path: Path) -> None:
    repo, row = _repo_with_sources(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "frames.jsonl"
    outside_file.touch()

    lexical = copy.deepcopy(row)
    lexical["frames_jsonl_path"] = str(outside_file)
    with pytest.raises(PermissionError, match="escapes the repository"):
        _plan(lexical, repo)

    alias = repo / "outside-alias"
    alias.symlink_to(outside, target_is_directory=True)
    resolved = copy.deepcopy(row)
    resolved["frames_jsonl_path"] = str(alias / "frames.jsonl")
    with pytest.raises(v2.RawSupervisionPlanError, match="symlinked component"):
        _plan(resolved, repo)


def test_v2_rejects_internal_parent_and_leaf_symlinks(tmp_path: Path) -> None:
    repo, row = _repo_with_sources(tmp_path)
    real_base = repo / "metadata" / "scene_a"
    parent_alias = repo / "metadata-alias"
    parent_alias.symlink_to(repo / "metadata", target_is_directory=True)
    changed_parent = copy.deepcopy(row)
    changed_parent["frames_jsonl_path"] = str(
        parent_alias / "scene_a" / "frames.jsonl"
    )
    with pytest.raises(v2.RawSupervisionPlanError, match="symlinked component"):
        _plan(changed_parent, repo)

    leaf_alias = real_base / "frames-alias.jsonl"
    leaf_alias.symlink_to(real_base / "frames.jsonl")
    changed_leaf = copy.deepcopy(row)
    changed_leaf["frames_jsonl_path"] = str(leaf_alias)
    with pytest.raises(v2.RawSupervisionPlanError, match="symlinked component"):
        _plan(changed_leaf, repo)


@pytest.mark.parametrize("form", ("missing", "directory", "fifo"))
def test_v2_rejects_missing_and_nonregular_referenced_paths(
    tmp_path: Path,
    form: str,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    replacement = repo / "metadata" / "scene_a" / f"invalid-{form}"
    if form == "directory":
        replacement.mkdir()
    elif form == "fifo":
        os.mkfifo(replacement)
    row["frames_jsonl_path"] = str(replacement)
    message = "must exist" if form == "missing" else "regular file"
    with pytest.raises(v2.RawSupervisionPlanError, match=message):
        _plan(row, repo)


def test_v2_rejects_hard_links_and_duplicate_file_aliases(tmp_path: Path) -> None:
    repo, row = _repo_with_sources(tmp_path)
    original = Path(row["frames_jsonl_path"])
    hard_link = original.with_name("frames-hard-link.jsonl")
    os.link(original, hard_link)
    changed = copy.deepcopy(row)
    changed["frames_jsonl_path"] = str(hard_link)
    with pytest.raises(v2.RawSupervisionPlanError, match="hard-link alias"):
        _plan(changed, repo)

    repo2, row2 = _repo_with_sources(tmp_path / "second")
    row2["render_plan_path"] = row2["frames_jsonl_path"]
    with pytest.raises(v2.RawSupervisionPlanError, match="unique file identities"):
        _plan(row2, repo2)


def test_v2_rejects_source_index_path_substitution_without_payload_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = v2.load_frozen_development_metadata(ROOT)
    rows = [json.loads(line) for line in SOURCE_INDEX.read_bytes().splitlines()]
    development_scene = str(plan.pairs[0]["scene_id"])
    row = next(item for item in rows if item["scene_id"] == development_scene)
    referenced = {
        Path(item[field])
        for item in rows
        for field in (
            "frames_jsonl_path",
            "scene_manifest_path",
            "render_plan_path",
            "render_summary_path",
        )
    }
    row["frames_jsonl_path"] = row["render_plan_path"]
    original_read_bytes = Path.read_bytes
    original_open = Path.open
    original_os_open = os.open

    def guarded_read_bytes(path: Path) -> bytes:
        assert path not in referenced
        return original_read_bytes(path)

    def guarded_path_open(path: Path, *args: Any, **kwargs: Any):
        assert path not in referenced
        return original_open(path, *args, **kwargs)

    def guarded_os_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        assert Path(os.fsdecode(os.fspath(path))) not in referenced
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    with pytest.raises(v2.RawSupervisionPlanError, match="inventory changed"):
        v2.plan_development_source_inventory(plan, rows, repo_root=ROOT)


def test_v2_rejects_noncanonical_or_aliased_repository_roots(tmp_path: Path) -> None:
    repo, row = _repo_with_sources(tmp_path)
    with pytest.raises(v2.RawSupervisionPlanError, match="canonical and absolute"):
        v2.plan_development_source_inventory(
            _synthetic_plan(),
            [row],
            repo_root=Path(str(repo) + "/../repo"),
            enforce_frozen_hashes=False,
        )
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(v2.RawSupervisionPlanError, match="path alias"):
        v2.plan_development_source_inventory(
            _synthetic_plan(),
            [row],
            repo_root=alias,
            enforce_frozen_hashes=False,
        )
    with pytest.raises(v2.RawSupervisionPlanError, match="path alias"):
        v2.load_frozen_development_metadata(alias)

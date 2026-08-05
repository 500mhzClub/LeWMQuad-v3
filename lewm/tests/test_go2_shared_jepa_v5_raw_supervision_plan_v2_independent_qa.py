"""Independent adversarial QA for the V5 raw-supervision metadata plan V2.

The candidate and its author evidence are frozen inputs.  This review opens
only the preregistered metadata allowlist and uses temporary synthetic trees
for path-boundary and source-index replacement probes.
"""
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
DATASET_MANIFEST = ROOT / v1.DATASET_MANIFEST_RELATIVE_PATH
DATASET_ROWS = ROOT / v1.DATASET_ROWS_RELATIVE_PATH
SIDECAR_MANIFEST = ROOT / v1.SIDECAR_MANIFEST_RELATIVE_PATH
ROLE_FILES = {
    role: SIDECAR_MANIFEST.parent / f"{role}.jsonl"
    for role in v1.DEVELOPMENT_ROLES
}

FROZEN_ARTIFACT_SHA256 = {
    ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan.py": (
        "e7ab8727b0d93d3fd8f9e2a3ab5cfdc4f9199e18b8d0a7f5a1f7dc0b5dc0c18e"
    ),
    ROOT / "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan.py": (
        "e2b49e660292ff99a7794ff3c761f9563a9e182b2889aec3a4e94b835c4be56c"
    ),
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v1_handoff_2026-07-13.md": (
        "557e6877f02ab61cf300177131735d0831304995dfe0b0f2482b0b5c91fc85fa"
    ),
    ROOT
    / "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_independent_access_review.py": (
        "b7180e901852e34cd412806aa9e8889c0da544b8ea83d3ba51f8efc663018bc6"
    ),
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v1_independent_review_2026-07-13.md": (
        "fcedb1efaffe4ca07141f7750188409c9f8f474231d2bb4f2db750f36f3f07b5"
    ),
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
}
EXPECTED_INVENTORY_HASHES = {
    "scene_role": "f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b",
    "frames": "7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d",
    "manifests": "2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5",
    "plans": "0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4",
    "summaries": "bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a",
}
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
SOURCE_FIELDS = (
    "frames_jsonl_path",
    "scene_manifest_path",
    "render_plan_path",
    "render_summary_path",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _synthetic_plan(
    scenes: tuple[str, ...] = ("scene_a",),
) -> v2.DevelopmentRawSupervisionPlan:
    return v2.DevelopmentRawSupervisionPlan(
        value={"licenses": {"raw_raycast_build_authorized": False}},
        pairs=tuple(
            {
                "scene_id": scene,
                "family": f"family_{scene}",
                "source_split": "train",
            }
            for scene in scenes
        ),
        endpoints=tuple(
            {
                "identity": {
                    "scene_id": scene,
                    "dataset_role": "train",
                }
            }
            for scene in scenes
        ),
    )


def _source_row(repo: Path, scene: str = "scene_a") -> dict[str, Any]:
    base = repo / "metadata" / scene
    return {
        "scene_id": scene,
        "family": f"family_{scene}",
        "split": "train",
        "frames_jsonl_path": str(base / "frames.jsonl"),
        "scene_manifest_path": str(base / "manifest.json"),
        "render_plan_path": str(base / "plan.json"),
        "render_summary_path": str(base / "summary.json"),
        "hashes": {
            "frames_jsonl_file_sha256": _sha(f"{scene}-frames"),
            "scene_manifest_file_sha256": _sha(f"{scene}-manifest-file"),
            "scene_manifest_sha256": _sha(f"{scene}-manifest-content"),
            "render_plan_file_sha256": _sha(f"{scene}-plan"),
            "render_summary_file_sha256": _sha(f"{scene}-summary"),
        },
    }


def _repo_with_sources(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    row = _source_row(repo)
    base = repo / "metadata" / "scene_a"
    base.mkdir(parents=True)
    for field in SOURCE_FIELDS:
        Path(row[field]).write_bytes(b"metadata-only-fixture\n")
    return repo, row


def _inventory(
    repo: Path,
    rows: list[dict[str, Any]],
    *,
    plan: v2.DevelopmentRawSupervisionPlan | None = None,
) -> v2.DevelopmentSourceInventory:
    return v2.plan_development_source_inventory(
        plan or _synthetic_plan(),
        rows,
        repo_root=repo,
        enforce_frozen_hashes=False,
    )


def _read_source_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SOURCE_INDEX.read_bytes().splitlines()]


def test_v2_qa_frozen_author_and_predecessor_hashes() -> None:
    for path, expected in FROZEN_ARTIFACT_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_v2_qa_reproduces_v1_block_and_closes_static_symlink_escape(
    tmp_path: Path,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_frame = outside / "frames.jsonl"
    outside_frame.write_bytes(b"outside\n")
    alias = repo / "metadata-alias"
    alias.symlink_to(outside, target_is_directory=True)
    row["frames_jsonl_path"] = str(alias / outside_frame.name)

    accepted = v1.plan_development_source_inventory(
        _synthetic_plan(),
        [row],
        repo_root=repo,
        enforce_frozen_hashes=False,
    )
    emitted = Path(accepted.records[0]["frames"]["path"])
    assert not emitted.resolve(strict=True).is_relative_to(repo)
    with pytest.raises(v2.RawSupervisionPlanError, match="symlinked component"):
        _inventory(repo, [row])


def test_v2_qa_independently_reconstructs_exact_population_and_inventory() -> None:
    raw_rows = [json.loads(line) for line in DATASET_ROWS.read_bytes().splitlines()]
    development = [
        row for row in raw_rows if row["dataset_role"] in v2.DEVELOPMENT_ROLES
    ]
    role_pairs = Counter(row["dataset_role"] for row in development)
    role_scenes = {
        role: {row["scene_id"] for row in development if row["dataset_role"] == role}
        for role in v2.DEVELOPMENT_ROLES
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
    assert {role: len(value) for role, value in role_scenes.items()} == (
        ROLE_SCENE_COUNTS
    )
    assert Counter(endpoints.values()) == Counter(ROLE_ENDPOINT_COUNTS)
    assert len(development) == 5172
    assert 2 * len(development) == 10344
    assert len(endpoints) == 9460

    plan = v2.load_frozen_development_metadata(ROOT)
    inventory = v2.load_frozen_development_source_inventory(ROOT, plan)
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

    source_by_scene = {row["scene_id"]: row for row in _read_source_rows()}
    scene_roles = {
        scene: role for role, scenes in role_scenes.items() for scene in scenes
    }
    scene_role_records: list[dict[str, str]] = []
    frames: list[dict[str, str]] = []
    manifests: list[dict[str, str]] = []
    plans: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    for scene in sorted(scene_roles):
        row = source_by_scene[scene]
        hashes = row["hashes"]
        scene_role_records.append({"scene_id": scene, "role": scene_roles[scene]})
        frames.append(
            {
                "scene_id": scene,
                "path": row["frames_jsonl_path"],
                "sha256": hashes["frames_jsonl_file_sha256"],
            }
        )
        manifests.append(
            {
                "scene_id": scene,
                "path": row["scene_manifest_path"],
                "file_sha256": hashes["scene_manifest_file_sha256"],
                "content_sha256": hashes["scene_manifest_sha256"],
            }
        )
        plans.append(
            {
                "scene_id": scene,
                "path": row["render_plan_path"],
                "sha256": hashes["render_plan_file_sha256"],
            }
        )
        summaries.append(
            {
                "scene_id": scene,
                "path": row["render_summary_path"],
                "sha256": hashes["render_summary_file_sha256"],
            }
        )
    assert {
        "scene_role": _canonical_sha256(scene_role_records),
        "frames": _canonical_sha256(frames),
        "manifests": _canonical_sha256(manifests),
        "plans": _canonical_sha256(plans),
        "summaries": _canonical_sha256(summaries),
    } == EXPECTED_INVENTORY_HASHES


@pytest.mark.parametrize(
    "spelling",
    ("relative", "dot", "dotdot", "duplicate-separator", "trailing", "double-root", "nul"),
)
def test_v2_qa_rejects_every_noncanonical_path_spelling(
    tmp_path: Path,
    spelling: str,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    path = Path(row["frames_jsonl_path"])
    if spelling == "relative":
        replacement = "metadata/scene_a/frames.jsonl"
    elif spelling == "dot":
        replacement = f"{path.parent}/./{path.name}"
    elif spelling == "dotdot":
        replacement = f"{path.parent}/unused/../{path.name}"
    elif spelling == "duplicate-separator":
        replacement = f"{path.parent}//{path.name}"
    elif spelling == "trailing":
        replacement = f"{path}/"
    elif spelling == "double-root":
        replacement = f"/{path}"
    elif spelling == "nul":
        replacement = f"{path}\x00"
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(spelling)
    row["frames_jsonl_path"] = replacement
    with pytest.raises(v2.RawSupervisionPlanError, match="canonical and absolute"):
        _inventory(repo, [row])


@pytest.mark.parametrize("location", ("first-parent", "second-parent", "leaf"))
@pytest.mark.parametrize("target_scope", ("inside", "outside"))
def test_v2_qa_rejects_every_parent_and_leaf_symlink(
    tmp_path: Path,
    location: str,
    target_scope: str,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target_root = repo / "real" if target_scope == "inside" else outside
    target_scene = target_root / "scene_a"
    target_scene.mkdir(parents=True)
    target_frame = target_scene / "frames.jsonl"
    target_frame.write_bytes(b"target\n")

    if location == "first-parent":
        alias = repo / "parent-alias"
        alias.symlink_to(target_root, target_is_directory=True)
        row["frames_jsonl_path"] = str(alias / "scene_a" / "frames.jsonl")
    elif location == "second-parent":
        alias_parent = repo / "metadata" / "scene-alias"
        alias_parent.symlink_to(target_scene, target_is_directory=True)
        row["frames_jsonl_path"] = str(alias_parent / "frames.jsonl")
    else:
        alias_leaf = repo / "metadata" / "scene_a" / "frames-alias.jsonl"
        alias_leaf.symlink_to(target_frame)
        row["frames_jsonl_path"] = str(alias_leaf)
    with pytest.raises(v2.RawSupervisionPlanError, match="symlinked component"):
        _inventory(repo, [row])


@pytest.mark.parametrize("kind", ("missing", "directory", "fifo"))
def test_v2_qa_rejects_missing_directory_and_fifo_leaves(
    tmp_path: Path,
    kind: str,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    replacement = repo / "metadata" / "scene_a" / f"invalid-{kind}"
    if kind == "directory":
        replacement.mkdir()
    elif kind == "fifo":
        os.mkfifo(replacement)
    row["frames_jsonl_path"] = str(replacement)
    expected = "must exist" if kind == "missing" else "regular file"
    with pytest.raises(v2.RawSupervisionPlanError, match=expected):
        _inventory(repo, [row])


def test_v2_qa_rejects_hardlinks_repeated_paths_and_inode_aliases(
    tmp_path: Path,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    original = Path(row["frames_jsonl_path"])
    alias = original.with_name("frames-hardlink.jsonl")
    os.link(original, alias)
    changed = copy.deepcopy(row)
    changed["frames_jsonl_path"] = str(alias)
    with pytest.raises(v2.RawSupervisionPlanError, match="hard-link alias"):
        _inventory(repo, [changed])

    repo2, row2 = _repo_with_sources(tmp_path / "duplicate")
    row2["render_plan_path"] = row2["frames_jsonl_path"]
    with pytest.raises(v2.RawSupervisionPlanError, match="unique file identities"):
        _inventory(repo2, [row2])


def test_v2_qa_rejects_repository_root_aliases_and_non_directories(
    tmp_path: Path,
) -> None:
    repo, row = _repo_with_sources(tmp_path)
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(v2.RawSupervisionPlanError, match="path alias"):
        _inventory(alias, [row])
    with pytest.raises(v2.RawSupervisionPlanError, match="canonical and absolute"):
        _inventory(Path("relative-repo"), [row])
    with pytest.raises(v2.RawSupervisionPlanError, match="canonical and absolute"):
        _inventory(Path(str(repo) + "/../repo"), [row])
    non_directory = tmp_path / "not-a-directory"
    non_directory.write_bytes(b"x")
    with pytest.raises(v2.RawSupervisionPlanError, match="directory"):
        _inventory(non_directory, [row])
    with pytest.raises(v2.RawSupervisionPlanError, match="must exist"):
        _inventory(tmp_path / "missing-root", [row])


def test_v2_qa_validates_only_88_retained_records_and_never_g2_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = v2.load_frozen_development_metadata(ROOT)
    rows = _read_source_rows()
    selected_scenes = {
        endpoint["identity"]["scene_id"] for endpoint in plan.endpoints
    }
    excluded_scenes = {row["scene_id"] for row in rows} - selected_scenes
    assert len(selected_scenes) == 88
    assert len(excluded_scenes) == 8

    selected_paths = {
        row[field]
        for row in rows
        if row["scene_id"] in selected_scenes
        for field in SOURCE_FIELDS
    }
    forbidden_root = ROOT / ".generated" / "v5-v2-qa-forbidden-g2"
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
    inventory = v2.plan_development_source_inventory(plan, rows, repo_root=ROOT)
    assert len(inventory.records) == 88
    assert inventory.hashes == EXPECTED_INVENTORY_HASHES
    assert Counter(validated) == Counter({path: 2 for path in selected_paths})
    assert len(validated) == 88 * 4 * 2


def test_v2_qa_opens_exact_metadata_allowlist_and_zero_forbidden_payloads(
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
    referenced = {
        Path(record[section]["path"])
        for record in inventory.records
        for section in ("frames", "scene_manifest", "render_plan", "render_summary")
    }
    assert not referenced.intersection(opened)
    assert all(
        count == 0
        for key, count in inventory.access_ledger.items()
        if key.endswith("_opens")
    )


def test_v2_qa_rejects_source_index_leaf_identity_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "index" / "source.jsonl"
    source.parent.mkdir(parents=True)
    payload = b'{}\n'
    source.write_bytes(payload)
    monkeypatch.setattr(v2, "SOURCE_INDEX_RELATIVE_PATH", "index/source.jsonl")
    monkeypatch.setattr(v2, "SOURCE_INDEX_FILE_SHA256", hashlib.sha256(payload).hexdigest())
    original_open = os.open
    swapped = False

    def replace_before_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        if not swapped and Path(path) == source:
            swapped = True
            source.rename(source.with_name("validated-source.jsonl"))
            source.write_bytes(payload)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(v2.RawSupervisionPlanError, match="identity changed"):
        v2._read_frozen_source_index(repo)


@pytest.mark.parametrize("replacement", ("parent-symlink", "leaf-hardlink"))
def test_v2_qa_source_index_open_is_atomic_against_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    """This is expected to fail on V2 and is the independent BLOCK evidence."""

    repo = tmp_path / "repo"
    parent = repo / "index"
    parent.mkdir(parents=True)
    source = parent / "source.jsonl"
    payload = b'{}\n'
    source.write_bytes(payload)
    monkeypatch.setattr(v2, "SOURCE_INDEX_RELATIVE_PATH", "index/source.jsonl")
    monkeypatch.setattr(v2, "SOURCE_INDEX_FILE_SHA256", hashlib.sha256(payload).hexdigest())
    original_open = os.open
    replaced = False

    def replace_before_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal replaced
        if not replaced and Path(path) == source:
            replaced = True
            outside = tmp_path / f"outside-{replacement}"
            outside.mkdir()
            if replacement == "parent-symlink":
                moved_parent = outside / "moved-index"
                parent.rename(moved_parent)
                parent.symlink_to(moved_parent, target_is_directory=True)
            else:
                moved_source = outside / "moved-source.jsonl"
                source.rename(moved_source)
                os.link(moved_source, source)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_open)
    try:
        v2._read_frozen_source_index(repo)
    except (PermissionError, v2.RawSupervisionPlanError):
        return
    metadata = source.stat(follow_symlinks=False)
    assert source.resolve(strict=True).is_relative_to(repo) and metadata.st_nlink == 1, (
        "V2 accepted a source-index path replaced after validation: "
        f"resolved={source.resolve(strict=True)}, nlink={metadata.st_nlink}"
    )

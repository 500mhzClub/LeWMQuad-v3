"""Independent correctness and access-boundary review for the V5 metadata plan.

The frozen source, author tests, and planning documents are review inputs.  This
suite reconstructs the frozen populations from metadata and uses only synthetic
fixtures for adversarial path and substitution probes.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import pytest

from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan import (
    DEVELOPMENT_ROLES,
    PRIMITIVE_VOCABULARY,
    DevelopmentRawSupervisionPlan,
    RawSupervisionPlanError,
    canonical_json_sha256,
    load_frozen_development_metadata,
    load_frozen_development_source_inventory,
    plan_development_raw_supervision,
    plan_development_source_inventory,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_MANIFEST = (
    ROOT
    / ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
    "dataset/dataset_manifest.json"
)
DATASET_ROWS = (
    ROOT
    / ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
    "dataset/rows.jsonl"
)
SIDECAR_MANIFEST = (
    ROOT / ".generated/go2_attitude_sidecar/dynamic_cartesian_v1/manifest.json"
)
SOURCE_INDEX = (
    ROOT
    / ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
    "source_index/go2_navigation_sources_v04.jsonl"
)
ROLE_FILES = {
    role: SIDECAR_MANIFEST.parent / f"{role}.jsonl" for role in DEVELOPMENT_ROLES
}

ARTIFACT_SHA256 = {
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
    / "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_preregistration_2026-07-13.md": (
        "07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb"
    ),
    ROOT
    / (
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_source_inventory_"
        "amendment_2026-07-13.md"
    ): (
        "39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3"
    ),
}
METADATA_SHA256 = {
    DATASET_MANIFEST: (
        "ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180"
    ),
    DATASET_ROWS: (
        "187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac"
    ),
    SIDECAR_MANIFEST: (
        "6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529"
    ),
    SOURCE_INDEX: (
        "11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c"
    ),
    ROLE_FILES["train"]: (
        "6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6"
    ),
    ROLE_FILES["checkpoint_selection"]: (
        "4ed434d04afc94b7b82050f5e9fafc900cc03c33a2d847f9784410f8f76f65de"
    ),
    ROLE_FILES["probability_calibration"]: (
        "3e5c10e6c15969eb30fbf38bbdb7b47d5fafe25bf14c5547f07ac609b79d91ae"
    ),
}
FULL_ROLE_COUNTS = {
    "train": 4262,
    "checkpoint_selection": 495,
    "probability_calibration": 415,
    "g2_evaluation": 469,
}
SCENE_COUNTS = {
    "train": 72,
    "checkpoint_selection": 8,
    "probability_calibration": 8,
}
UNIQUE_ENDPOINT_COUNTS = {
    "train": 7777,
    "checkpoint_selection": 924,
    "probability_calibration": 759,
}
INVENTORY_SHA256 = {
    "scene_role": "f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b",
    "frames": "7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d",
    "manifests": "2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5",
    "plans": "0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4",
    "summaries": "bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_bound(path: Path, opened: list[Path]) -> bytes:
    resolved = path.resolve(strict=True)
    assert resolved == path
    assert not path.is_symlink()
    assert stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == METADATA_SHA256[path]
    opened.append(path)
    return payload


def _parse_jsonl(payload: bytes) -> list[dict[str, Any]]:
    assert payload and payload.endswith(b"\n")
    rows = [json.loads(line) for line in payload.splitlines()]
    assert all(type(row) is dict for row in rows)
    return rows


@pytest.fixture(scope="module")
def independent_metadata() -> dict[str, Any]:
    opened: list[Path] = []
    dataset_manifest = json.loads(_read_bound(DATASET_MANIFEST, opened))
    rows = _parse_jsonl(_read_bound(DATASET_ROWS, opened))
    sidecar_manifest = json.loads(_read_bound(SIDECAR_MANIFEST, opened))
    sidecars = {
        role: _parse_jsonl(_read_bound(ROLE_FILES[role], opened))
        for role in DEVELOPMENT_ROLES
    }
    source_rows = _parse_jsonl(_read_bound(SOURCE_INDEX, opened))
    assert set(opened) == set(METADATA_SHA256)
    assert len(opened) == len(METADATA_SHA256) == 7
    return {
        "dataset_manifest": dataset_manifest,
        "rows": rows,
        "sidecar_manifest": sidecar_manifest,
        "sidecars": sidecars,
        "source_rows": source_rows,
        "opened": tuple(opened),
    }


def _independent_reconstruction(metadata: Mapping[str, Any]) -> dict[str, Any]:
    rows = metadata["rows"]
    sidecars = metadata["sidecars"]
    role_counts = Counter(row["dataset_role"] for row in rows)
    assert dict(role_counts) == FULL_ROLE_COUNTS
    assert len({row["global_row"] for row in rows}) == len(rows) == 5641
    assert (
        metadata["dataset_manifest"]["scene_roles"]["assignments_sha256"]
        == "016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02"
    )

    manifest = metadata["sidecar_manifest"]
    manifest_core = dict(manifest)
    manifest_hash = manifest_core.pop("content_sha256")
    assert _canonical_sha256(manifest_core) == manifest_hash
    assert set(manifest["roles"]) == {*DEVELOPMENT_ROLES, "g2_evaluation"}

    development = [row for row in rows if row["dataset_role"] in DEVELOPMENT_ROLES]
    scenes: dict[str, set[str]] = {role: set() for role in DEVELOPMENT_ROLES}
    endpoints: dict[str, tuple[Any, ...]] = {}
    endpoint_roles: Counter[str] = Counter()
    joined_rows = 0
    primitives: set[str] = set()
    for role in DEVELOPMENT_ROLES:
        role_rows = [row for row in development if row["dataset_role"] == role]
        sidecar_index = {row["global_row"]: row for row in sidecars[role]}
        assert len(sidecar_index) == len(sidecars[role]) == FULL_ROLE_COUNTS[role]
        assert set(sidecar_index) == {row["global_row"] for row in role_rows}
        for row in role_rows:
            scenes[role].add(row["scene_id"])
            primitives.add(row["primitive"])
            sidecar = sidecar_index[row["global_row"]]
            sidecar_core = dict(sidecar)
            declared = sidecar_core.pop("content_sha256")
            assert _canonical_sha256(sidecar_core) == declared
            row_identity = {
                "global_row": row["global_row"],
                "scene_id": row["scene_id"],
                "scene_id_sha256": hashlib.sha256(
                    row["scene_id"].encode("utf-8")
                ).hexdigest(),
                "dataset_role": role,
                "label_shard_row": row["label_shard_row"],
                "label_shard_sha256": row["label_shard_sha256"],
                "current_image_sha256": row["current_image_sha256"],
                "next_image_sha256": row["next_image_sha256"],
            }
            expected_join = {
                "global_row": row["global_row"],
                "dataset_role": role,
                "scene_id_sha256": row_identity["scene_id_sha256"],
                "frames_jsonl_sha256": row["frames_jsonl_sha256"],
                "env_index": row["env_index"],
                "current_frame_index": row["current_frame_index"],
                "next_frame_index": row["next_frame_index"],
                "current_timestamp_ns": row["current_timestamp_ns"],
                "next_timestamp_ns": row["next_timestamp_ns"],
                "row_identity_sha256": _canonical_sha256(row_identity),
            }
            assert all(sidecar[field] == value for field, value in expected_join.items())
            joined_rows += 1
            for side in ("current", "next"):
                identity = {
                    "dataset_role": role,
                    "scene_id": row["scene_id"],
                    "episode_id": row["episode_id"],
                    "env_index": row["env_index"],
                    "episode_step": row[f"{side}_episode_step"],
                    "frame_index": row[f"{side}_frame_index"],
                    "timestamp_ns": row[f"{side}_timestamp_ns"],
                    "image_sha256": row[f"{side}_image_sha256"],
                }
                identity_sha = _canonical_sha256(identity)
                attitude = sidecar[side]
                endpoint_metadata = (
                    row[f"{side}_image_path"],
                    row["frames_jsonl_sha256"],
                    row["scene_manifest_sha256"],
                    tuple(float(value) for value in attitude["base_quat_world_xyzw"]),
                    float(attitude["stored_base_yaw_rad"]),
                )
                if identity_sha in endpoints:
                    assert endpoints[identity_sha] == endpoint_metadata
                else:
                    endpoints[identity_sha] = endpoint_metadata
                    endpoint_roles[role] += 1

    assert joined_rows == len(development) == 5172
    assert 2 * joined_rows == 10344
    assert len(endpoints) == 9460
    assert dict(endpoint_roles) == UNIQUE_ENDPOINT_COUNTS
    assert {role: len(scenes[role]) for role in DEVELOPMENT_ROLES} == SCENE_COUNTS
    assert primitives == set(PRIMITIVE_VOCABULARY)

    paired_scene: dict[str, tuple[str, str, str]] = {}
    for row in development:
        identity = (row["dataset_role"], row["family"], row["source_split"])
        assert paired_scene.setdefault(row["scene_id"], identity) == identity
    source_rows = metadata["source_rows"]
    source_by_scene = {row["scene_id"]: row for row in source_rows}
    assert len(source_by_scene) == len(source_rows) == 96
    assert set(paired_scene) <= set(source_by_scene)
    assert len(set(source_by_scene) - set(paired_scene)) == 8

    scene_role_records = []
    frames = []
    manifests = []
    plans = []
    summaries = []
    for scene_id in sorted(paired_scene):
        role, family, source_split = paired_scene[scene_id]
        row = source_by_scene[scene_id]
        assert (row["family"], row["split"]) == (family, source_split)
        hashes = row["hashes"]
        for field in (
            "frames_jsonl_path",
            "scene_manifest_path",
            "render_plan_path",
            "render_summary_path",
        ):
            path = Path(row[field])
            assert path.is_absolute()
            assert str(path) == os.path.normpath(str(path))
            path.relative_to(ROOT)
        scene_role_records.append({"scene_id": scene_id, "role": role})
        frames.append(
            {
                "scene_id": scene_id,
                "path": row["frames_jsonl_path"],
                "sha256": hashes["frames_jsonl_file_sha256"],
            }
        )
        manifests.append(
            {
                "scene_id": scene_id,
                "path": row["scene_manifest_path"],
                "file_sha256": hashes["scene_manifest_file_sha256"],
                "content_sha256": hashes["scene_manifest_sha256"],
            }
        )
        plans.append(
            {
                "scene_id": scene_id,
                "path": row["render_plan_path"],
                "sha256": hashes["render_plan_file_sha256"],
            }
        )
        summaries.append(
            {
                "scene_id": scene_id,
                "path": row["render_summary_path"],
                "sha256": hashes["render_summary_file_sha256"],
            }
        )
    inventory_hashes = {
        "scene_role": _canonical_sha256(scene_role_records),
        "frames": _canonical_sha256(frames),
        "manifests": _canonical_sha256(manifests),
        "plans": _canonical_sha256(plans),
        "summaries": _canonical_sha256(summaries),
    }
    assert inventory_hashes == INVENTORY_SHA256
    return {
        "development_rows": development,
        "endpoint_count": len(endpoints),
        "inventory_hashes": inventory_hashes,
        "scene_count": len(paired_scene),
    }


def _synthetic_sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _synthetic_row(index: int, primitive: str) -> dict[str, Any]:
    return {
        "global_row": index,
        "dataset_role": "train",
        "scene_id": "scene_a",
        "family": "family_a",
        "episode_id": "episode_a",
        "env_index": 0,
        "reset_count": 1,
        "source_split": "train",
        "frames_jsonl_sha256": _synthetic_sha("frames"),
        "scene_manifest_sha256": _synthetic_sha("manifest"),
        "current_episode_step": index,
        "current_frame_index": index,
        "current_timestamp_ns": index * 100,
        "current_image_path": f"/synthetic/frame_{index:04d}.png",
        "current_image_sha256": _synthetic_sha(f"image-{index}"),
        "next_episode_step": index + 1,
        "next_frame_index": index + 1,
        "next_timestamp_ns": (index + 1) * 100,
        "next_image_path": f"/synthetic/frame_{index + 1:04d}.png",
        "next_image_sha256": _synthetic_sha(f"image-{index + 1}"),
        "primitive": primitive,
        "relative_se2_current_frame": [0.1, 0.0, 0.01],
        "label_shard_path": "/synthetic/labels.npz",
        "label_shard_sha256": _synthetic_sha("labels"),
        "label_shard_row": index,
    }


def _synthetic_sidecar(row: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "global_row": row["global_row"],
        "dataset_role": row["dataset_role"],
        "scene_id_sha256": hashlib.sha256(row["scene_id"].encode("utf-8")).hexdigest(),
        "frames_jsonl_sha256": row["frames_jsonl_sha256"],
        "env_index": row["env_index"],
        "current_frame_index": row["current_frame_index"],
        "next_frame_index": row["next_frame_index"],
        "current_timestamp_ns": row["current_timestamp_ns"],
        "next_timestamp_ns": row["next_timestamp_ns"],
        "row_identity_sha256": _canonical_sha256(
            {
                "global_row": row["global_row"],
                "scene_id": row["scene_id"],
                "scene_id_sha256": hashlib.sha256(
                    row["scene_id"].encode("utf-8")
                ).hexdigest(),
                "dataset_role": row["dataset_role"],
                "label_shard_row": row["label_shard_row"],
                "label_shard_sha256": row["label_shard_sha256"],
                "current_image_sha256": row["current_image_sha256"],
                "next_image_sha256": row["next_image_sha256"],
            }
        ),
        "current": {
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": float(row["current_frame_index"]) / 10.0,
        },
        "next": {
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": float(row["next_frame_index"]) / 10.0,
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _synthetic_inputs() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = [
        _synthetic_row(index, primitive)
        for index, primitive in enumerate(PRIMITIVE_VOCABULARY)
    ]
    sidecars = {role: [] for role in DEVELOPMENT_ROLES}
    sidecars["train"] = [_synthetic_sidecar(row) for row in rows]
    return rows, sidecars


def _synthetic_plan() -> DevelopmentRawSupervisionPlan:
    rows, sidecars = _synthetic_inputs()
    return plan_development_raw_supervision(
        rows,
        sidecars,
        input_bindings={"synthetic": True},
        access_ledger={"payload_opens": 0},
        enforce_frozen_counts=False,
    )


def _source_row(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "metadata/scene_a"
    return {
        "scene_id": "scene_a",
        "family": "family_a",
        "split": "train",
        "frames_jsonl_path": str(base / "frames.jsonl"),
        "scene_manifest_path": str(base / "manifest.json"),
        "render_plan_path": str(base / "plan.json"),
        "render_summary_path": str(base / "summary.json"),
        "hashes": {
            "frames_jsonl_file_sha256": _synthetic_sha("frames"),
            "scene_manifest_file_sha256": _synthetic_sha("manifest-file"),
            "scene_manifest_sha256": _synthetic_sha("manifest-content"),
            "render_plan_file_sha256": _synthetic_sha("plan"),
            "render_summary_file_sha256": _synthetic_sha("summary"),
        },
    }


def _rehash_sidecar(row: dict[str, Any]) -> None:
    core = dict(row)
    core.pop("content_sha256", None)
    row["content_sha256"] = canonical_json_sha256(core)


def test_independent_review_artifact_hashes_are_frozen() -> None:
    for path, expected in ARTIFACT_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_independent_reconstruction_matches_every_frozen_population(
    independent_metadata: Mapping[str, Any],
) -> None:
    result = _independent_reconstruction(independent_metadata)
    assert len(result["development_rows"]) == 5172
    assert 2 * len(result["development_rows"]) == 10344
    assert result["endpoint_count"] == 9460
    assert result["scene_count"] == 88
    assert result["inventory_hashes"] == INVENTORY_SHA256


def test_candidate_plan_matches_independent_identities_and_denies_authority() -> None:
    plan = load_frozen_development_metadata(ROOT)
    inventory = load_frozen_development_source_inventory(ROOT, plan)
    assert len(plan.pairs) == 5172
    assert plan.value["endpoint_instance_count"] == 10344
    assert len(plan.endpoints) == 9460
    assert plan.value["content_sha256"] == (
        "8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3"
    )
    assert plan.value["ordered_pair_sha256"] == (
        "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
    )
    assert plan.value["ordered_endpoint_sha256"] == (
        "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
    )
    assert len(inventory.records) == 88
    assert inventory.hashes == INVENTORY_SHA256
    assert not any(plan.value["licenses"].values())


def test_candidate_controlled_file_opens_are_exact_metadata_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_opens: list[Path] = []
    original_path_open = Path.open
    original_os_open = os.open

    def traced_path_open(path: Path, *args: Any, **kwargs: Any):
        file_opens.append(path.resolve(strict=True))
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
            raw = Path(os.fsdecode(os.fspath(path)))
            if not raw.is_absolute() and dir_fd is not None:
                raw = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / raw
            file_opens.append(raw.resolve(strict=True))
        return descriptor

    monkeypatch.setattr(Path, "open", traced_path_open)
    monkeypatch.setattr(os, "open", traced_os_open)
    plan = load_frozen_development_metadata(ROOT)
    load_frozen_development_source_inventory(ROOT, plan)

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
    assert Counter(file_opens) == expected
    assert SIDECAR_MANIFEST.parent / "g2_evaluation.jsonl" not in file_opens


def test_image_hash_only_collision_retains_two_full_endpoint_identities() -> None:
    rows, sidecars = _synthetic_inputs()
    rows[1]["current_image_sha256"] = rows[0]["current_image_sha256"]
    sidecars["train"] = [_synthetic_sidecar(row) for row in rows]
    plan = plan_development_raw_supervision(
        rows,
        sidecars,
        input_bindings={},
        access_ledger={},
        enforce_frozen_counts=False,
    )
    shared_image = rows[0]["current_image_sha256"]
    collided = [
        endpoint
        for endpoint in plan.endpoints
        if endpoint["identity"]["image_sha256"] == shared_image
    ]
    assert len(collided) == 2
    assert len({endpoint["identity_sha256"] for endpoint in collided}) == 2


def test_cross_role_and_cross_scene_sidecar_substitutions_reject() -> None:
    rows, sidecars = _synthetic_inputs()
    crossed_role = copy.deepcopy(sidecars)
    crossed_role["train"][0]["dataset_role"] = "checkpoint_selection"
    _rehash_sidecar(crossed_role["train"][0])
    with pytest.raises(RawSupervisionPlanError, match="another role"):
        plan_development_raw_supervision(
            rows,
            crossed_role,
            input_bindings={},
            access_ledger={},
            enforce_frozen_counts=False,
        )

    crossed_scene = copy.deepcopy(sidecars)
    crossed_scene["train"][0]["scene_id_sha256"] = hashlib.sha256(
        b"scene_b"
    ).hexdigest()
    _rehash_sidecar(crossed_scene["train"][0])
    with pytest.raises(RawSupervisionPlanError, match="sidecar join changed"):
        plan_development_raw_supervision(
            rows,
            crossed_scene,
            input_bindings={},
            access_ledger={},
            enforce_frozen_counts=False,
        )


def test_cross_scene_family_and_source_split_inventory_substitutions_reject(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    row = _source_row(repo)
    for field, replacement, message in (
        ("scene_id", "scene_b", "lacks a development scene"),
        ("family", "family_b", "family/split differs"),
        ("split", "validation", "family/split differs"),
    ):
        changed = copy.deepcopy(row)
        changed[field] = replacement
        with pytest.raises(RawSupervisionPlanError, match=message):
            plan_development_source_inventory(
                _synthetic_plan(),
                [changed],
                repo_root=repo,
                enforce_frozen_hashes=False,
            )


def test_sidecar_orphan_and_content_mutation_reject() -> None:
    rows, sidecars = _synthetic_inputs()
    orphaned = copy.deepcopy(sidecars)
    orphan = copy.deepcopy(orphaned["train"][-1])
    orphan["global_row"] = 999
    orphaned["train"].append(orphan)
    with pytest.raises(RawSupervisionPlanError, match="orphan row"):
        plan_development_raw_supervision(
            rows,
            orphaned,
            input_bindings={},
            access_ledger={},
            enforce_frozen_counts=False,
        )

    mutated = copy.deepcopy(sidecars)
    mutated["train"][0]["content_sha256"] = "0" * 64
    with pytest.raises(RawSupervisionPlanError, match="content hash changed"):
        plan_development_raw_supervision(
            rows,
            mutated,
            input_bindings={},
            access_ledger={},
            enforce_frozen_counts=False,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "repeated"))
def test_frozen_source_inventory_rejects_missing_extra_and_repeated_rows(
    independent_metadata: Mapping[str, Any],
    mutation: str,
) -> None:
    plan = load_frozen_development_metadata(ROOT)
    source_rows = copy.deepcopy(independent_metadata["source_rows"])
    development_scene = plan.pairs[0]["scene_id"]
    if mutation == "missing":
        source_rows = [row for row in source_rows if row["scene_id"] != development_scene]
    elif mutation == "extra":
        extra = copy.deepcopy(source_rows[0])
        extra["scene_id"] = "independent-extra-scene"
        source_rows.append(extra)
    else:
        source_rows.append(copy.deepcopy(source_rows[0]))
    with pytest.raises(RawSupervisionPlanError):
        plan_development_source_inventory(plan, source_rows, repo_root=ROOT)


def test_inventory_rejects_lexical_repository_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    row = _source_row(repo)
    row["frames_jsonl_path"] = str(outside / "frames.jsonl")
    with pytest.raises(PermissionError, match="escapes the repository"):
        plan_development_source_inventory(
            _synthetic_plan(),
            [row],
            repo_root=repo,
            enforce_frozen_hashes=False,
        )


def test_inventory_rejects_in_repository_symlink_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    alias = repo / "aliased-source"
    alias.symlink_to(outside, target_is_directory=True)
    row = _source_row(repo)
    row["frames_jsonl_path"] = str(alias / "frames.jsonl")
    try:
        inventory = plan_development_source_inventory(
            _synthetic_plan(),
            [row],
            repo_root=repo,
            enforce_frozen_hashes=False,
        )
    except (PermissionError, RawSupervisionPlanError):
        return
    accepted = Path(inventory.records[0]["frames"]["path"])
    assert accepted.resolve(strict=False).is_relative_to(repo.resolve()), (
        f"accepted in-repository path {accepted} resolves outside to "
        f"{accepted.resolve(strict=False)}"
    )

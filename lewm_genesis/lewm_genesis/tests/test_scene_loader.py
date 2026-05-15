"""Tests for ``lewm_genesis.scene_loader`` against the real acceptance corpus.

The acceptance corpus under ``.generated/scene_corpus/acceptance/`` is produced
by the existing scene-generator pipeline and committed to the workspace as
fixtures for downstream consumers. These tests skip cleanly when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lewm_genesis.scene_loader import (
    DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER,
    PhysicsTiming,
    ScenePack,
    camera_mount_from_platform,
    find_scene_dirs,
    iter_scene_packs,
    load_platform_manifest,
    load_scene_pack,
    physics_timing_from_platform,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_CORPUS = REPO_ROOT / ".generated" / "scene_corpus" / "acceptance"
PLATFORM_MANIFEST = REPO_ROOT / "config" / "go2_platform_manifest.yaml"


def _require_acceptance_corpus() -> Path:
    if not ACCEPTANCE_CORPUS.is_dir():
        pytest.skip(f"acceptance corpus not present at {ACCEPTANCE_CORPUS}")
    return ACCEPTANCE_CORPUS


def _require_platform_manifest() -> Path:
    if not PLATFORM_MANIFEST.is_file():
        pytest.skip(f"platform manifest not present at {PLATFORM_MANIFEST}")
    return PLATFORM_MANIFEST


# ---------------------------------------------------------------------------
# Platform-manifest parsing
# ---------------------------------------------------------------------------


def test_camera_mount_pulled_from_real_manifest():
    manifest = load_platform_manifest(_require_platform_manifest())
    mount = camera_mount_from_platform(manifest)
    assert mount.parent_link == "camera_link"
    assert mount.native_resolution == (640, 480)
    assert mount.training_resolution == (224, 224)
    assert mount.fov_axis == "horizontal"
    assert mount.fov_deg == pytest.approx(78.323)
    assert mount.near_m == pytest.approx(0.05)
    assert mount.far_m == pytest.approx(200.0)
    assert mount.encoding == "rgb8"
    # Mount is forward of base by ~0.326m per manifest.
    assert mount.xyz_body_m[0] == pytest.approx(0.326)


def test_physics_timing_pulled_from_real_manifest():
    manifest = load_platform_manifest(_require_platform_manifest())
    timing = physics_timing_from_platform(manifest)
    assert timing.physics_dt_s == pytest.approx(0.002)
    assert timing.policy_dt_s == pytest.approx(0.02)
    assert timing.command_dt_s == pytest.approx(0.10)
    assert timing.action_block_size == 5
    # 0.02 / 0.002 = 10 physics ticks per policy tick
    assert timing.policy_decimation == 10
    # 0.10 / 0.02 = 5 policy ticks per command-block tick
    assert timing.command_ticks_per_block == 5


def test_physics_timing_derived_properties_defend_against_zero():
    # The defaults are nonzero, but make sure the derived properties at least
    # don't return zero (rollout loops would divide by these).
    timing = PhysicsTiming(
        physics_dt_s=0.002, policy_dt_s=0.02, command_dt_s=0.10, action_block_size=5
    )
    assert timing.policy_decimation >= 1
    assert timing.command_ticks_per_block >= 1


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------


def test_find_scene_dirs_returns_sorted_unique_scenes():
    corpus = _require_acceptance_corpus()
    scenes = find_scene_dirs(corpus)
    assert scenes, "acceptance corpus should contain at least one scene"
    assert scenes == sorted(scenes)
    assert len({s.name for s in scenes}) == len(scenes)


def test_find_scene_dirs_filters_by_split():
    corpus = _require_acceptance_corpus()
    splits_present = {p.parents[1].name for p in find_scene_dirs(corpus)}
    a_split = next(iter(splits_present))
    filtered = find_scene_dirs(corpus, split=a_split)
    assert filtered
    for scene in filtered:
        assert scene.parents[1].name == a_split


def test_find_scene_dirs_filters_by_family():
    corpus = _require_acceptance_corpus()
    families_present = {p.parents[0].name for p in find_scene_dirs(corpus)}
    a_family = next(iter(families_present))
    filtered = find_scene_dirs(corpus, family=a_family)
    assert filtered
    for scene in filtered:
        assert scene.parents[0].name == a_family


# ---------------------------------------------------------------------------
# Scene pack loading
# ---------------------------------------------------------------------------


def test_load_scene_pack_reads_real_acceptance_scene():
    corpus = _require_acceptance_corpus()
    platform_path = _require_platform_manifest()
    scenes = find_scene_dirs(corpus)
    pack = load_scene_pack(
        scenes[0],
        platform_manifest=platform_path,
        workspace_root=REPO_ROOT,
    )
    assert isinstance(pack, ScenePack)
    assert pack.scene_id
    assert pack.family
    assert pack.split  # acceptance corpus always sets split
    assert pack.manifest_sha256
    assert pack.physics_seed >= 0
    assert pack.source_dir == scenes[0].resolve()
    # World bounds is a pair of (min_xy, max_xy).
    (xmin, ymin), (xmax, ymax) = pack.world_bounds_xy_m
    assert xmax > xmin
    assert ymax > ymin
    # Static objects are present for any non-empty scene family.
    assert pack.static_objects
    obj = pack.static_objects[0]
    assert len(obj.center_xyz_m) == 3
    assert len(obj.size_xyz_m) == 3


def test_load_scene_pack_resolves_go2_urdf_path():
    corpus = _require_acceptance_corpus()
    platform_path = _require_platform_manifest()
    scenes = find_scene_dirs(corpus)
    pack = load_scene_pack(
        scenes[0],
        platform_manifest=platform_path,
        workspace_root=REPO_ROOT,
    )
    assert pack.robot.urdf_path.is_file(), (
        f"resolved URDF should exist; got {pack.robot.urdf_path}"
    )
    assert pack.robot.foot_links_in_lewm_order == DEFAULT_GO2_FOOT_LINKS_LEWM_ORDER


def test_load_scene_pack_carries_spawn_pose():
    corpus = _require_acceptance_corpus()
    platform_path = _require_platform_manifest()
    scenes = find_scene_dirs(corpus)
    pack = load_scene_pack(
        scenes[0],
        platform_manifest=platform_path,
        workspace_root=REPO_ROOT,
    )
    assert len(pack.robot.spawn_xyz_m) == 3
    assert len(pack.robot.spawn_quat_wxyz) == 4
    # Spawn z should be roughly the upstream stance height (~0.375m).
    assert 0.2 < pack.robot.spawn_xyz_m[2] < 0.6


def test_load_scene_pack_caches_platform_when_dict_passed():
    corpus = _require_acceptance_corpus()
    platform_dict = load_platform_manifest(_require_platform_manifest())
    scenes = find_scene_dirs(corpus)
    pack = load_scene_pack(
        scenes[0],
        platform_manifest=platform_dict,
        workspace_root=REPO_ROOT,
    )
    assert pack.camera.parent_link == "camera_link"


def test_load_scene_pack_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene_pack(
            tmp_path,
            platform_manifest=_require_platform_manifest(),
            workspace_root=REPO_ROOT,
        )


def test_load_scene_pack_detects_manifest_inconsistency(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": "a",
                "family": "x",
                "split": "train",
                "difficulty_tier": "smoke",
                "manifest_sha256": "AAA",
                "physics_seed": 1,
                "topology_seed": 2,
                "visual_seed": 3,
            }
        )
    )
    (tmp_path / "genesis_scene.json").write_text(
        json.dumps(
            {
                "scene_id": "b",  # mismatched
                "family": "x",
                "manifest_sha256": "AAA",
                "physics_seed": 1,
                "world_bounds_xy_m": [[-1.0, -1.0], [1.0, 1.0]],
                "spawn": {"xyz_m": [0, 0, 0.4], "quat_wxyz": [1, 0, 0, 0]},
                "objects": [],
                "camera_constraints": {},
            }
        )
    )
    with pytest.raises(ValueError, match="scene_id"):
        load_scene_pack(
            tmp_path,
            platform_manifest=_require_platform_manifest(),
            workspace_root=REPO_ROOT,
        )


def test_iter_scene_packs_walks_corpus():
    corpus = _require_acceptance_corpus()
    platform_path = _require_platform_manifest()
    packs = list(
        iter_scene_packs(
            corpus,
            platform_manifest=platform_path,
            workspace_root=REPO_ROOT,
        )
    )
    assert packs
    # IDs should be unique across the corpus walk.
    assert len({p.scene_id for p in packs}) == len(packs)

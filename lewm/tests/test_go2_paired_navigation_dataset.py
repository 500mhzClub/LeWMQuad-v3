from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lewm.benchmarks.generalization_protocol import (
    SceneDisjointManifests,
    build_hashed_scene_role_commitment,
)
from lewm.datasets.go2_paired_navigation import (
    DatasetContractError,
    FREE_CLASS,
    ForbiddenSceneError,
    PrimitiveTransition,
    ProvenanceError,
    SceneRenderSource,
    V3SceneExclusions,
    build_paired_navigation_dataset,
    canonical_json_sha256,
    deterministic_family_role_split,
    deterministic_scene_split,
    load_scene_id_exclusions,
    occupancy_label_metrics,
    scene_id_sha256,
    select_primitive_transitions,
    sha256_file,
    verify_dataset_provenance,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
    manifest_sha256,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _empty_manifest(scene_id: str) -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family="unit_test",
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-10.0, -10.0), (10.0, 10.0)),
        spawn=SpawnSpec(xyz_m=(0.0, 0.0, 0.35), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _pose(x: float, y: float, yaw: float) -> tuple[dict, dict]:
    return (
        {
            "position": {"x": x, "y": y, "z": 0.35},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
    )


def _make_source(
    tmp_path: Path,
    scene_id: str = "training_scene",
    *,
    scene_manifest: SceneManifest | None = None,
) -> SceneRenderSource:
    root = tmp_path / scene_id
    root.mkdir()
    rgb_dir = root / "rgb"
    rgb_dir.mkdir()
    manifest = scene_manifest or _empty_manifest(scene_id)
    assert manifest.scene_id == scene_id
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest.to_dict())
    canonical_manifest_hash = manifest_sha256(manifest)

    frames: list[dict] = []
    rendered: list[dict] = []
    for step in range(1, 7):
        for env_index in (1, 0):  # Deliberately opposite the emitted stream order.
            frame_index = (step - 1) * 2 + env_index
            yaw = 0.0 if env_index == 0 else math.pi / 2.0
            x = (step - 1) * 0.1 if env_index == 0 else 0.0
            y = 0.0 if env_index == 0 else (step - 1) * 0.1
            base_pose, base_rpy = _pose(x, y, yaw)
            sequence_id = env_index if step <= 5 else 10 + env_index
            frame = {
                "frame_index": frame_index,
                "env_index": env_index,
                "timestamp_ns": step * 100_000_000,
                "timestamp_s": step * 0.1,
                "episode": {
                    "episode_id": 7,
                    "episode_step": step,
                    "reset_count": 1,
                    "manifest_sha256": canonical_manifest_hash,
                    "split": "train",
                },
                "base_pose_world": base_pose,
                "base_rpy_rad": base_rpy,
                # Planned direction is intentionally wrong. Rendered metadata
                # below is the camera pose that must control the label.
                "camera_pose_world": {
                    "position": [x, y, 0.40],
                    "lookat": [x - math.cos(yaw), y - math.sin(yaw), 0.40],
                    "up": [0.0, 0.0, 1.0],
                },
                "command_context": {
                    "sequence_id": sequence_id,
                    "primitive_name": "forward",
                    "block_size": 5,
                    "command_dt_s": 0.1,
                },
            }
            frames.append(frame)
            image_path = rgb_dir / f"frame_{frame_index:06d}_env_{env_index:02d}.png"
            image_path.write_bytes(f"rgb:{frame_index}:{env_index}".encode())
            rendered.append(
                {
                    "frame_index": frame_index,
                    "env_index": env_index,
                    "timestamp_ns": step * 100_000_000,
                    "camera_valid": True,
                    "rgb_path": str(image_path),
                    "camera_pose_world": {
                        "position": [x, y, 0.40],
                        "lookat": [x + math.cos(yaw), y + math.sin(yaw), 0.40],
                        "up": [0.0, 0.0, 1.0],
                    },
                }
            )

    frames_path = root / "frames.jsonl"
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in frames))
    rendered_path = root / "frames_rendered.jsonl"
    rendered_path.write_text("".join(json.dumps(row) + "\n" for row in rendered))
    plan_path = root / "render_replay_plan.json"
    _write_json(
        plan_path,
        {
            "schema": "lewm_render_replay_plan_v0",
            "scene_id": scene_id,
            "scene_family": manifest.family,
            "split": "train",
            "manifest_sha256": canonical_manifest_hash,
            "frames_jsonl": str(frames_path),
            "camera": {
                "fov_axis": "horizontal",
                "fov_deg": 78.323,
                "near_m": 0.05,
            },
        },
    )
    return SceneRenderSource(
        scene_id=scene_id,
        scene_manifest_path=manifest_path,
        render_plan_path=plan_path,
        family=manifest.family,
        frames_jsonl_path=frames_path,
        rendered_frames_jsonl_path=rendered_path,
    )


def _exclusions() -> V3SceneExclusions:
    return V3SceneExclusions(
        development_scene_id_sha256=frozenset({scene_id_sha256("v3_development")}),
        sealed_scene_id_sha256=frozenset({scene_id_sha256("v3_sealed")}),
    )


def _write_scene_commitments(path: Path, scene_ids: tuple[str, ...]) -> Path:
    path.write_text(
        "".join(scene_id_sha256(scene_id) + "\n" for scene_id in scene_ids)
    )
    return path


def _write_hashed_scene_roles(
    path: Path,
    *,
    benchmark_id: str,
    role_scene_ids: dict[str, tuple[str, ...]],
) -> Path:
    def entries(role: str) -> list[dict[str, str]]:
        return [
            {"scene_id": scene_id} for scene_id in role_scene_ids.get(role, ())
        ]

    payload = build_hashed_scene_role_commitment(
        SceneDisjointManifests(
            development={
                "benchmark_id": benchmark_id,
                "geometry_contract_sha256": "1" * 64,
                "train_scenes": entries("train"),
                "validation_scenes": entries("development"),
                "excluded_scenes": entries("excluded"),
            },
            sealed_test={
                "benchmark_id": benchmark_id,
                "commitment_sha256": "2" * 64,
                "scenes": entries("sealed_test"),
            },
        )
    )
    _write_json(path, payload)
    return path


def test_scene_split_is_deterministic_order_independent_and_scene_disjoint() -> None:
    scene_ids = [f"scene_{index:03d}" for index in range(200)]
    forward = deterministic_scene_split(scene_ids, validation_fraction=0.2, seed="fixed")
    reverse = deterministic_scene_split(
        reversed(scene_ids), validation_fraction=0.2, seed="fixed"
    )
    assert forward == reverse
    assert set(forward) == set(scene_ids)
    assert {split for split in forward.values()} == {"train", "validation"}
    assert all(split in {"train", "validation"} for split in forward.values())


def test_family_role_split_is_fixed_per_family_and_label_independent() -> None:
    scene_families = {
        f"{family}_{index}": family
        for family in ("alpha", "beta")
        for index in range(5)
    }
    first = deterministic_family_role_split(
        scene_families, role_scenes_per_family=1, seed="fixed"
    )
    second = deterministic_family_role_split(
        dict(reversed(list(scene_families.items()))),
        role_scenes_per_family=1,
        seed="fixed",
    )
    assert first == second
    for family in ("alpha", "beta"):
        roles = [first[scene] for scene, item in scene_families.items() if item == family]
        assert roles.count("checkpoint_selection") == 1
        assert roles.count("probability_calibration") == 1
        assert roles.count("g2_evaluation") == 1
        assert roles.count("train") == 2

    with pytest.raises(ValueError, match="at least 4"):
        deterministic_family_role_split(
            {f"short_{index}": "short" for index in range(3)},
            role_scenes_per_family=1,
            seed="fixed",
        )


def test_declared_source_family_is_verified_against_manifest(tmp_path: Path) -> None:
    source = replace(_make_source(tmp_path), family="wrong_family")
    contract = load_geometry_contract(verify_sources=False)
    with pytest.raises(DatasetContractError, match="source family.*does not match"):
        build_paired_navigation_dataset(
            sources=[source],
            output_dir=tmp_path / "dataset",
            geometry_contract=contract,
            v3_exclusions=_exclusions(),
            validation_fraction=0.0,
            max_transitions_per_scene=1,
        )


def test_direct_family_roles_persist_counts_commitments_and_strict_quota(
    tmp_path: Path,
) -> None:
    sources = [_make_source(tmp_path, f"role_scene_{index}") for index in range(4)]
    contract = load_geometry_contract(verify_sources=False)
    result = build_paired_navigation_dataset(
        sources=sources,
        output_dir=tmp_path / "dataset",
        geometry_contract=contract,
        v3_exclusions=_exclusions(),
        role_scenes_per_family=1,
        split_seed="direct-fixed",
        max_transitions_per_scene=2,
    )
    role_contract = result["scene_roles"]
    assert role_contract["scene_counts"] == {
        "train": 1,
        "checkpoint_selection": 1,
        "probability_calibration": 1,
        "g2_evaluation": 1,
    }
    assert set(role_contract["row_counts"].values()) == {2}
    assert role_contract["family_scene_counts"]["unit_test"] == (
        role_contract["scene_counts"]
    )
    assert role_contract["family_row_counts"]["unit_test"] == (
        role_contract["row_counts"]
    )
    assert role_contract["assignments_sha256"] == canonical_json_sha256(
        role_contract["assignments"]
    )
    rows = [json.loads(line) for line in (tmp_path / "dataset/rows.jsonl").read_text().splitlines()]
    assert all(row["dataset_role"] == role_contract["assignments"][row["scene_id"]] for row in rows)

    short_sources = [
        _make_source(tmp_path, f"short_role_scene_{index}") for index in range(4)
    ]
    with pytest.raises(DatasetContractError, match="fewer than the requested 3"):
        build_paired_navigation_dataset(
            sources=short_sources,
            output_dir=tmp_path / "short-dataset",
            geometry_contract=contract,
            v3_exclusions=_exclusions(),
            role_scenes_per_family=1,
            max_transitions_per_scene=3,
        )


def test_provenance_verifier_rejects_legacy_single_grid_schema(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "legacy_dataset_manifest.json"
    _write_json(
        manifest_path,
        {"schema": "lewm_go2_paired_navigation_dataset_v1"},
    )
    with pytest.raises(ProvenanceError, match="unsupported paired-navigation"):
        verify_dataset_provenance(manifest_path, verify_images=False)


def _selection_transition(index: int, primitive: str, env_index: int) -> PrimitiveTransition:
    current_pose, current_rpy = _pose(index * 0.1, float(env_index), 0.0)
    next_pose, next_rpy = _pose(index * 0.1 + 0.5, float(env_index), 0.0)
    episode = {
        "episode_id": env_index,
        "episode_step": index * 5 + 1,
        "reset_count": 1,
    }
    current = {
        "frame_index": index * 2,
        "env_index": env_index,
        "timestamp_ns": (index * 5 + 1) * 100_000_000,
        "episode": episode,
        "base_pose_world": current_pose,
        "base_rpy_rad": current_rpy,
    }
    nxt = {
        "frame_index": index * 2 + 1,
        "env_index": env_index,
        "timestamp_ns": (index * 5 + 6) * 100_000_000,
        "episode": {**episode, "episode_step": index * 5 + 6},
        "base_pose_world": next_pose,
        "base_rpy_rad": next_rpy,
    }
    return PrimitiveTransition(
        current=current,
        next=nxt,
        primitive=primitive,
        duration_s=0.5,
    )


def test_transition_cap_is_hash_deterministic_and_balances_strata() -> None:
    candidates = [
        _selection_transition(index, primitive, env)
        for index, (primitive, env) in enumerate(
            [
                ("forward", 0),
                ("forward", 0),
                ("forward", 1),
                ("forward", 1),
                ("yaw_left", 0),
                ("yaw_left", 0),
                ("yaw_left", 1),
                ("yaw_left", 1),
            ]
        )
    ]
    selected, metadata = select_primitive_transitions(
        candidates,
        scene_id="selection_scene",
        max_transitions=4,
        seed="fixed",
    )
    reversed_selected, _ = select_primitive_transitions(
        list(reversed(candidates)),
        scene_id="selection_scene",
        max_transitions=4,
        seed="fixed",
    )
    selected_ids = [item.current["frame_index"] for item in selected]
    reversed_ids = [item.current["frame_index"] for item in reversed_selected]
    assert selected_ids == reversed_ids
    assert {item.primitive for item in selected} == {"forward", "yaw_left"}
    assert {item.current["env_index"] for item in selected} == {0, 1}
    assert metadata["candidate_count"] == 8
    assert metadata["selected_count"] == 4
    assert metadata["stratum_count"] == 4


@pytest.mark.parametrize("forbidden_scene", ["v3_development", "v3_sealed"])
def test_v3_scene_is_rejected_before_any_scene_artifact_is_opened(
    tmp_path: Path, forbidden_scene: str
) -> None:
    missing = tmp_path / "must_not_be_opened"
    source = SceneRenderSource(
        scene_id=forbidden_scene,
        scene_manifest_path=missing,
        render_plan_path=missing,
    )
    contract = load_geometry_contract(verify_sources=False)
    with pytest.raises(ForbiddenSceneError):
        build_paired_navigation_dataset(
            sources=[source],
            output_dir=tmp_path / "output",
            geometry_contract=contract,
            v3_exclusions=_exclusions(),
        )


def test_mixed_newline_and_structured_commitments_reject_before_artifact_open(
    tmp_path: Path,
) -> None:
    raw_ids = {
        "v3_development": "heldout_scene_v3_dev_001",
        "v3_sealed": "heldout_scene_v3_sealed_001",
        "v4_development": "heldout_scene_v4_dev_001",
        "v4_sealed": "heldout_scene_v4_sealed_001",
    }
    v3_development = _write_scene_commitments(
        tmp_path / "v3-development.sha256", (raw_ids["v3_development"],)
    )
    v3_sealed = _write_scene_commitments(
        tmp_path / "v3-sealed.sha256", (raw_ids["v3_sealed"],)
    )
    v4_roles = _write_hashed_scene_roles(
        tmp_path / "v4-roles.json",
        benchmark_id="go2_generalization_v4",
        role_scene_ids={
            "train": ("v4_training",),
            "development": (raw_ids["v4_development"],),
            "sealed_test": (raw_ids["v4_sealed"],),
            "excluded": (),
        },
    )
    exclusions = load_scene_id_exclusions(
        (
            ("v3_development", v3_development),
            ("v3_sealed", v3_sealed),
            ("v4", v4_roles),
        )
    )
    assert [item.label for item in exclusions.sets] == [
        "v3_development",
        "v3_sealed",
        "v4.development",
        "v4.sealed_test",
    ]
    exclusions.assert_allowed("v4_training")

    missing = tmp_path / "must_not_be_opened"
    contract = load_geometry_contract(verify_sources=False)
    for forbidden_scene in raw_ids.values():
        source = SceneRenderSource(
            scene_id=forbidden_scene,
            scene_manifest_path=missing,
            render_plan_path=missing,
        )
        with pytest.raises(ForbiddenSceneError) as caught:
            build_paired_navigation_dataset(
                sources=[source],
                output_dir=tmp_path / f"output-{scene_id_sha256(forbidden_scene)}",
                geometry_contract=contract,
                scene_exclusions=exclusions,
            )
        assert forbidden_scene not in str(caught.value)

    metadata = exclusions.to_metadata()
    encoded = json.dumps(metadata, sort_keys=True)
    assert metadata["union_count"] == 4
    assert metadata["raw_forbidden_scene_ids_persisted"] is False
    assert metadata["sets"]["v4.development"]["source_role"] == "development"
    assert metadata["sets"]["v4.development"]["file_sha256"] == sha256_file(
        v4_roles
    )
    assert metadata["sets"]["v4.development"][
        "source_role_set_sha256"
    ] == canonical_json_sha256(
        {"scene_id_sha256": [scene_id_sha256(raw_ids["v4_development"])]}
    )
    assert all(forbidden not in encoded for forbidden in raw_ids.values())


def test_structured_commitment_tamper_is_rejected(tmp_path: Path) -> None:
    path = _write_hashed_scene_roles(
        tmp_path / "roles.json",
        benchmark_id="tamper_test",
        role_scene_ids={
            "train": ("train",),
            "development": ("development",),
            "sealed_test": ("sealed",),
            "excluded": (),
        },
    )
    payload = json.loads(path.read_text())
    payload["scene_id_sha256_by_role"]["development"] = ["f" * 64]
    _write_json(path, payload)
    with pytest.raises(ProvenanceError, match="content hash mismatch"):
        load_scene_id_exclusions((("v4", path),))


def test_interleaved_env_pairs_labels_odometry_and_provenance(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    contract = load_geometry_contract(verify_sources=False)
    output = tmp_path / "dataset"
    development_path = _write_scene_commitments(
        tmp_path / "development.sha256", ("unseen_development_scene",)
    )
    sealed_path = _write_scene_commitments(
        tmp_path / "sealed.sha256", ("unseen_sealed_scene",)
    )
    exclusions = load_scene_id_exclusions(
        (("v3_development", development_path), ("v3_sealed", sealed_path))
    )
    result = build_paired_navigation_dataset(
        sources=[source],
        output_dir=output,
        geometry_contract=contract,
        scene_exclusions=exclusions,
        validation_fraction=0.0,
        max_transitions_per_scene=2,
    )

    assert result["schema"] == "lewm_go2_paired_navigation_dataset_v2"
    assert result["row_count"] == 2
    assert result["scene_count"] == 1
    assert result["local_grid"]["forward_edge_range_m"] == [-1.0, 5.4]
    assert result["local_grid"]["left_edge_range_m"] == [-3.2, 3.2]
    assert result["local_grid"]["forward_center_range_m"] == [-0.95, 5.35]
    assert result["geometry_contract"]["sha256"] == contract.sha256
    assert result["label_semantics"]["target_occupancy_space"] == (
        "body_inflated_configuration_space"
    )
    assert result["label_semantics"]["visibility_occlusion_space"] == (
        "uninflated_physical_obstacle_occupancy"
    )
    assert result["label_semantics"]["configuration_inflation_radius_m"] == (
        contract.configuration_space.body_inflation_radius_m
    )
    assert result["label_semantics"]["visibility_inflation_radius_m"] == 0.0
    assert result["exclusions"]["sets"]["v3_development"][
        "file_sha256"
    ] == sha256_file(development_path)
    assert result["exclusions"]["sets"]["v3_sealed"][
        "commitment_set_sha256"
    ] == canonical_json_sha256([scene_id_sha256("unseen_sealed_scene")])
    manifest_text = (output / "dataset_manifest.json").read_text()
    assert "unseen_development_scene" not in manifest_text
    assert "unseen_sealed_scene" not in manifest_text

    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert {row["schema"] for row in rows} == {"lewm_go2_paired_navigation_row_v2"}
    assert [row["env_index"] for row in rows] == [0, 1]
    assert [row["current_episode_step"] for row in rows] == [1, 1]
    assert [row["next_episode_step"] for row in rows] == [6, 6]
    assert [row["transition_duration_s"] for row in rows] == pytest.approx([0.5, 0.5])
    assert [row["primitive"] for row in rows] == ["forward", "forward"]
    assert all(row["source_transition_configuration_validated"] for row in rows)
    np.testing.assert_allclose(
        [row["relative_se2_current_frame"] for row in rows],
        [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        atol=1e-6,
    )
    assert rows[0]["scene_manifest_sha256"] == manifest_sha256(
        _empty_manifest("training_scene")
    )
    assert rows[0]["frame_plan_sha256"] == sha256_file(source.render_plan_path)
    assert rows[0]["frames_jsonl_sha256"] == sha256_file(source.frames_jsonl_path)
    assert rows[0]["current_image_sha256"] == sha256_file(
        Path(rows[0]["current_image_path"])
    )

    shard_path = Path(rows[0]["label_shard_path"])
    with np.load(shard_path, allow_pickle=False) as shard:
        assert shard["current_labels"].shape == (2, 64, 64)
        assert shard["next_labels"].shape == (2, 64, 64)
        assert shard["current_supervision_mask"].all()
        assert shard["next_supervision_mask"].all()
        np.testing.assert_array_equal(
            shard["current_observed_mask"], shard["current_labels"] != 0
        )
        np.testing.assert_array_equal(
            shard["next_observed_mask"], shard["next_labels"] != 0
        )
        assert np.any(
            shard["current_supervision_mask"] & (shard["current_labels"] == 0)
        )
        np.testing.assert_allclose(
            shard["relative_se2_current_frame"],
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            atol=1e-6,
        )
        # Ahead is visible only if rendered camera metadata overrode the
        # deliberately backwards planned camera pose.
        forward_row = int(np.argmin(np.abs((-0.95 + np.arange(64) * 0.1) - 0.55)))
        center_col = int(np.argmin(np.abs((-3.15 + np.arange(64) * 0.1) - 0.05)))
        assert shard["current_labels"][0, forward_row, center_col] == FREE_CLASS

    checked = verify_dataset_provenance(output / "dataset_manifest.json")
    assert checked["image"] == 4
    assert checked["scene_manifest"] == 1
    assert checked["frame_plan"] == 1
    assert checked["frames_jsonl"] == 1
    assert checked["rendered_frames_jsonl"] == 1
    assert checked["shard"] == 1
    assert checked["exclusion_commitment"] == 2

    original_commitment = development_path.read_bytes()
    development_path.write_bytes(original_commitment + b"# tampered\n")
    with pytest.raises(ProvenanceError, match="exclusion_commitment hash mismatch"):
        verify_dataset_provenance(output / "dataset_manifest.json")
    development_path.write_bytes(original_commitment)

    tampered_image = Path(rows[0]["current_image_path"])
    tampered_image.write_bytes(tampered_image.read_bytes() + b"tampered")
    with pytest.raises(ProvenanceError, match="image hash mismatch"):
        verify_dataset_provenance(output / "dataset_manifest.json")


def test_known_metrics_exclude_unknown_truth_and_report_unknown_admission() -> None:
    target = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.uint8)
    prediction = np.asarray([[1, 1, 2], [0, 0, 1]], dtype=np.uint8)
    supervision = np.ones_like(target, dtype=bool)
    observed = target != 0
    metrics = occupancy_label_metrics(
        prediction,
        target,
        supervision_mask=supervision,
        observed_mask=observed,
    )
    assert metrics["known_cell_accuracy"] == pytest.approx(0.5)
    assert metrics["free_iou_on_known_truth"] == pytest.approx(1.0 / 3.0)
    assert metrics["occupied_iou_on_known_truth"] == pytest.approx(0.5)
    assert metrics["unknown_recall"] == pytest.approx(0.5)
    assert metrics["unknown_admission_rate_on_known_truth"] == pytest.approx(0.25)
    assert metrics["known_hallucination_rate_on_unknown_truth"] == pytest.approx(0.5)


def test_builder_rejects_transition_with_unsafe_intermediate_command_pose(
    tmp_path: Path,
) -> None:
    scene_id = "configuration_window_scene"
    wall = BoxObject(
        object_id="remote_wall",
        kind="wall",
        center_xyz_m=(5.0, 0.0, 0.5),
        size_xyz_m=(0.10, 1.0, 1.0),
        yaw_rad=0.0,
        material_id="wall",
    )
    base_manifest = _empty_manifest(scene_id)
    manifest = SceneManifest(**{**base_manifest.__dict__, "walls": (wall,)})
    source = _make_source(
        tmp_path,
        scene_id=scene_id,
        scene_manifest=manifest,
    )

    frames = [json.loads(line) for line in source.frames_jsonl_path.read_text().splitlines()]
    intermediate = next(
        frame
        for frame in frames
        if frame["env_index"] == 0 and frame["episode"]["episode_step"] == 3
    )
    # Endpoints remain at x=0.0 and x=0.5, but one frame inside the complete
    # command window is deliberately placed inside a remote wall. Endpoint-
    # only filtering would incorrectly retain this transition.
    intermediate["base_pose_world"]["position"]["x"] = 5.0
    source.frames_jsonl_path.write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames)
    )

    contract = load_geometry_contract(verify_sources=False)
    result = build_paired_navigation_dataset(
        sources=[source],
        output_dir=tmp_path / "configuration_filtered_dataset",
        geometry_contract=contract,
        v3_exclusions=_exclusions(),
        validation_fraction=0.0,
        max_transitions_per_scene=2,
    )

    assert result["row_count"] == 1
    rows = [
        json.loads(line)
        for line in Path(result["index"]["path"]).read_text().splitlines()
    ]
    assert [row["env_index"] for row in rows] == [1]
    validity = result["sources"][0]["selection"]["configuration_validity"]
    assert validity["screened_transition_count"] == 2
    assert validity["accepted_transition_count"] == 1
    assert validity["rejected_transition_count"] == 1
    assert validity["rejection_counts"] == {
        "transitions_rejected_negative_configuration_clearance": 1
    }
    assert validity["selected_rejection_count"] == 0
    assert result["stats"][
        "transitions_rejected_negative_configuration_clearance"
    ] == 1


def test_configuration_filter_rejects_segment_crossing_obstacle() -> None:
    import lewm.datasets.go2_paired_navigation as dataset_module

    scene_id = "configuration_segment_scene"
    wall = BoxObject(
        object_id="crossed_wall",
        kind="wall",
        center_xyz_m=(5.0, 0.0, 0.5),
        size_xyz_m=(0.10, 2.0, 1.0),
        yaw_rad=0.0,
        material_id="wall",
    )
    base_manifest = _empty_manifest(scene_id)
    manifest = SceneManifest(**{**base_manifest.__dict__, "walls": (wall,)})
    grid = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.47)
    left_pose, left_rpy = _pose(4.0, 0.0, 0.0)
    right_pose, right_rpy = _pose(6.0, 0.0, 0.0)
    left = {"base_pose_world": left_pose, "base_rpy_rad": left_rpy}
    right = {"base_pose_world": right_pose, "base_rpy_rad": right_rpy}
    transition = PrimitiveTransition(
        current=left,
        next=right,
        primitive="teleport_test",
        duration_s=0.1,
        frame_window=(left, right),
    )

    assert grid.configuration_clearance_m((4.0, 0.0)) > 0.0
    assert grid.configuration_clearance_m((6.0, 0.0)) > 0.0
    assert dataset_module._transition_configuration_rejection(
        transition, grid
    ) == "transitions_rejected_nonfree_configuration_segment"


def test_row_cap_is_applied_before_label_raycast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lewm.datasets.go2_paired_navigation as dataset_module

    source = _make_source(tmp_path, scene_id="capped_scene")
    contract = load_geometry_contract(verify_sources=False)
    calls = 0
    original = dataset_module.label_camera_visible_configuration_grid

    def counted_label(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert args[0].inflation_m == pytest.approx(
            contract.configuration_space.body_inflation_radius_m
        )
        assert kwargs["physical_visibility_grid"].inflation_m == 0.0
        assert kwargs["physical_visibility_grid"] is not args[0]
        return original(*args, **kwargs)

    monkeypatch.setattr(
        dataset_module, "label_camera_visible_configuration_grid", counted_label
    )
    result = build_paired_navigation_dataset(
        sources=[source],
        output_dir=tmp_path / "capped_dataset",
        geometry_contract=contract,
        v3_exclusions=_exclusions(),
        validation_fraction=0.0,
        max_transitions_per_scene=1,
        selection_seed="cap_smoke",
    )

    assert result["row_count"] == 1
    assert calls == 2
    assert result["row_selection"] == {
        "selection_seed": "cap_smoke",
        "max_transitions_per_scene": 1,
        "occurs_before_label_raycast": True,
        "occurs_after_configuration_validity_filter": True,
        "method": "hash_rank_within_primitive_env_episode_strata_then_round_robin",
    }
    selection = result["sources"][0]["selection"]
    assert selection["candidate_count"] == 2
    assert selection["selected_count"] == 1
    assert selection["stratum_count"] == 2
    assert selection["configuration_validity"] == {
        "screened_transition_count": 2,
        "accepted_transition_count": 2,
        "rejected_transition_count": 0,
        "rejection_counts": {},
        "selected_transition_count": 1,
        "selected_rejection_count": 0,
        "frame_scope": "complete_command_window_including_post_command_frame",
        "pose_test": "exact_configuration_clearance_m_gte_0",
        "segment_test": "raster_has_free_line_between_adjacent_frames",
    }

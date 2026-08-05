from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lewm.benchmarks import go2_perfect_camera_ray_field_audit as ray_audit
from lewm.datasets import go2_paired_navigation as labels_v3
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


ROOT = Path(__file__).resolve().parents[2]


def _box(
    object_id: str = "obstacle",
    *,
    center: tuple[float, float, float] = (2.0, 0.0, 0.5),
    size: tuple[float, float, float] = (0.5, 0.8, 1.0),
    yaw: float = 0.23,
) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind="obstacle",
        center_xyz_m=center,
        size_xyz_m=size,
        yaw_rad=yaw,
        material_id="wall",
    )


def _manifest(*, obstacles: tuple[BoxObject, ...] = ()) -> SceneManifest:
    return SceneManifest(
        scene_id="ray_audit_unit",
        family="open_obstacle_field",
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-5.0, -5.0), (7.0, 5.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=obstacles,
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _camera() -> labels_v3.CameraObservation:
    return labels_v3.CameraObservation(
        position_xyz_m=(0.326, 0.0, 0.393),
        lookat_xyz_m=(1.326, 0.0, 0.393),
        horizontal_fov_deg=78.323,
        vertical_fov_deg=62.8370386364,
        near_m=0.05,
        image_width_px=224,
        image_height_px=168,
        obstacle_ray_stride_px=2,
    )


def _grid(manifest: SceneManifest) -> InflatedOccupancyGrid:
    return InflatedOccupancyGrid(
        manifest,
        cell_size_m=0.05,
        inflation_m=0.0,
        treat_landmarks_as_obstacles=True,
        treat_distractors_as_obstacles=True,
    )


def _reconstruct(
    manifest: SceneManifest,
    *,
    rendered: tuple[BoxObject, ...],
    collision: tuple[BoxObject, ...],
) -> ray_audit.RayFieldRasterization:
    grid = _grid(manifest)
    return ray_audit.reconstruct_frame_from_perfect_rays(
        camera=ray_audit.CameraRaySpec.from_camera_observation(_camera()),
        rendered_obstacle_boxes=rendered,
        collision_obstacle_boxes=collision,
        base_xy_yaw=(0.0, 0.0, 0.17),
        physical_free_mask=grid.free_mask,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
    )


@pytest.mark.parametrize(
    "obstacles",
    [(), (_box(),), (_box(yaw=0.0, size=(0.08, 1.1, 0.7)),)],
)
def test_contract_assisted_reconstruction_is_bit_exact_to_v3(
    obstacles: tuple[BoxObject, ...],
) -> None:
    manifest = _manifest(obstacles=obstacles)
    grid = _grid(manifest)
    expected, supervision, _observed = (
        labels_v3._observable_physical_raster_and_output_labels(
            grid,
            rendered_obstacle_boxes=obstacles,
            collision_obstacle_boxes=obstacles,
            base_xy_yaw=(0.0, 0.0, 0.17),
            camera=_camera(),
            local_grid=labels_v3.DEFAULT_LOCAL_GRID,
        )
    )

    result = _reconstruct(manifest, rendered=obstacles, collision=obstacles)

    assert np.array_equal(result.contract_labels, expected)
    assert np.array_equal(supervision, np.ones((64, 64), dtype=bool))


def test_field_is_deterministic_and_detached_from_box_objects() -> None:
    obstacle = _box()
    manifest = _manifest(obstacles=(obstacle,))
    grid = _grid(manifest)
    output_x, output_y = ray_audit.output_world_centers(
        (0.0, 0.0, 0.0), ray_audit.OutputGridSpec()
    )
    window = ray_audit.physical_window_for_output(
        physical_free_mask=grid.free_mask,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
        output_world_x_m=output_x,
        output_world_y_m=output_y,
        output_cell_size_m=0.1,
    )
    kwargs = {
        "camera": ray_audit.CameraRaySpec.from_camera_observation(_camera()),
        "rendered_obstacle_boxes": (obstacle,),
        "physical_x_centers_m": window.x_centers_m,
        "physical_y_centers_m": window.y_centers_m,
        "physical_cell_size_m": window.cell_size_m,
    }

    first = ray_audit.build_perfect_camera_ray_field(**kwargs)
    second = ray_audit.build_perfect_camera_ray_field(**kwargs)

    assert first.content_sha256() == second.content_sha256()
    assert np.array_equal(first.ground_support_visible, second.ground_support_visible)
    assert np.array_equal(first.obstacle_first_hit_xy_m(), second.obstacle_first_hit_xy_m())
    assert not first.pixel_first_hit_distance_m.flags.writeable
    with pytest.raises(ValueError):
        first.pixel_first_hit_distance_m[0, 0] = 0.0


def test_ray_only_arm_exposes_non_camera_physical_free_dependency() -> None:
    manifest = _manifest()
    grid = _grid(manifest)
    full_free = np.array(grid.free_mask, dtype=bool, copy=True)
    origin_x, origin_y = grid.origin_xy
    ix = int(np.floor((1.5 - origin_x) / grid.cell_size_m))
    iy = int(np.floor((0.0 - origin_y) / grid.cell_size_m))
    full_free[ix - 1 : ix + 2, iy - 1 : iy + 2] = False

    result = ray_audit.reconstruct_frame_from_perfect_rays(
        camera=ray_audit.CameraRaySpec.from_camera_observation(_camera()),
        rendered_obstacle_boxes=(),
        collision_obstacle_boxes=(),
        base_xy_yaw=(0.0, 0.0, 0.0),
        physical_free_mask=full_free,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
    )

    assert np.any(result.contract_labels != result.ray_only_labels)
    difference = result.contract_labels != result.ray_only_labels
    assert np.all(result.contract_labels[difference] == ray_audit.UNKNOWN_CLASS)
    assert np.all(result.ray_only_labels[difference] == ray_audit.FREE_CLASS)


def test_collision_geometry_is_only_a_free_to_unknown_veto() -> None:
    manifest = _manifest()
    grid = _grid(manifest)
    collision_only = _box(center=(1.5, 0.0, 0.5), yaw=0.0)
    result = ray_audit.reconstruct_frame_from_perfect_rays(
        camera=ray_audit.CameraRaySpec.from_camera_observation(_camera()),
        rendered_obstacle_boxes=(),
        collision_obstacle_boxes=(collision_only,),
        base_xy_yaw=(0.0, 0.0, 0.0),
        physical_free_mask=grid.free_mask,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
    )

    changed = result.contract_pre_veto_labels != result.contract_labels
    assert np.any(changed)
    assert np.all(result.contract_pre_veto_labels[changed] == ray_audit.FREE_CLASS)
    assert np.all(result.contract_labels[changed] == ray_audit.UNKNOWN_CLASS)
    assert not np.any(result.contract_labels == ray_audit.OCCUPIED_CLASS)


def test_frame_audit_rejects_partial_supervision() -> None:
    result = _reconstruct(_manifest(), rendered=(), collision=())
    mask = np.ones((64, 64), dtype=bool)
    mask[0, 0] = False

    with pytest.raises(ValueError, match="full-grid"):
        ray_audit.audit_frame_labels(
            authoritative_labels=result.contract_labels,
            supervision_mask=mask,
            reconstruction=result,
            frame_key={"frame": 0},
        )


def test_exact_fit_summary_requires_320_unique_frames() -> None:
    base = {
        "schema": ray_audit.FRAME_AUDIT_SCHEMA,
        "frame_key": {"frame": 0},
        "authoritative_labels_sha256": "a" * 64,
        "contract_labels_sha256": "a" * 64,
        "ray_only_labels_sha256": "a" * 64,
        "contract_confusion_reference_rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "ray_only_confusion_reference_rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "contract_mismatch_cell_count": 0,
        "ray_only_mismatch_cell_count": 0,
    }
    reports = []
    for index in range(320):
        report = copy.deepcopy(base)
        report["frame_key"] = {"frame": index}
        reports.append(report)

    summary = ray_audit.summarize_exact_fit(reports)

    assert summary["frame_count"] == 320
    assert summary["cell_count"] == 1_310_720
    assert summary["contract_assisted"]["exact"] is True
    with pytest.raises(ValueError, match="exactly 320"):
        ray_audit.summarize_exact_fit(reports[:-1])
    reports[-1]["frame_key"] = {"frame": 0}
    with pytest.raises(ValueError, match="exactly unique"):
        ray_audit.summarize_exact_fit(reports)


def test_dry_run_cli_is_cpu_only_and_deterministic() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/audit_go2_perfect_camera_ray_field_fit.py"),
        "--dry-run",
    ]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    assert first.stdout == second.stdout
    assert '"dry_run": true' in first.stdout
    assert '"contract_parity": true' in first.stdout

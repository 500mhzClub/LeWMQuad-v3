from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lewm.benchmarks import go2_perfect_camera_ray_field_audit_v2 as ray_v2
from lewm.benchmarks import (
    go2_perfect_camera_ray_field_mismatch_decomposition as decomposition,
)
from lewm.datasets.go2_paired_navigation import CameraObservation
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> SceneManifest:
    return SceneManifest(
        scene_id="decomposition_unit",
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


def _grid() -> InflatedOccupancyGrid:
    return InflatedOccupancyGrid(
        _manifest(),
        cell_size_m=0.05,
        inflation_m=0.0,
        treat_landmarks_as_obstacles=True,
        treat_distractors_as_obstacles=True,
    )


def _camera() -> CameraObservation:
    return CameraObservation(
        position_xyz_m=(0.326, 0.0, 0.393),
        lookat_xyz_m=(1.326, 0.0, 0.393),
        up_xyz=(0.0, 0.0, 1.0),
        horizontal_fov_deg=78.323,
        vertical_fov_deg=62.8370386364,
        near_m=0.05,
        image_width_px=224,
        image_height_px=168,
        obstacle_ray_stride_px=2,
    )


def _camera_mapping(camera: CameraObservation) -> dict:
    return {
        "position_xyz_m": camera.position_xyz_m,
        "lookat_xyz_m": camera.lookat_xyz_m,
        "up_xyz": camera.up_xyz,
        "horizontal_fov_deg": camera.horizontal_fov_deg,
        "vertical_fov_deg": camera.vertical_fov_deg,
        "near_m": camera.near_m,
        "ground_plane_z_m": camera.ground_plane_z_m,
        "image_width_px": camera.image_width_px,
        "image_height_px": camera.image_height_px,
        "obstacle_ray_stride_px": camera.obstacle_ray_stride_px,
    }


def _box_mapping(
    *,
    y: float = 0.0,
    size_x: float = 0.02,
    size_y: float = 0.8,
    yaw: float = 0.0,
) -> dict[str, object]:
    return {
        "center_xyz_m": (2.0, y, 0.5),
        "size_xyz_m": (size_x, size_y, 1.0),
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "yaw_rad": yaw,
    }


def _output_grid() -> dict[str, object]:
    return {
        "rows": 64,
        "cols": 64,
        "cell_size_m": 0.1,
        "forward_min_edge_m": -1.0,
        "left_min_edge_m": -3.2,
    }


def _decompose_with_box(
    box: dict[str, object], *, base_yaw: float = 0.0
) -> dict:
    grid = _grid()
    camera = _camera()
    oriented = ray_v2.OrientedBox(**box)
    reconstruction = ray_v2.reconstruct_frame_from_perfect_rays(
        camera=ray_v2.CameraRaySpec.from_camera_observation(camera),
        rendered_obstacle_boxes=(oriented,),
        collision_obstacle_boxes=(oriented,),
        base_xy_yaw=(0.0, 0.0, base_yaw),
        physical_free_mask=grid.free_mask,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
    )
    return decomposition.decompose_frame(
        authoritative_labels=reconstruction.contract_labels,
        supervision_mask=np.ones((64, 64), dtype=bool),
        frame_key={"family": "open_obstacle_field", "frame": 0},
        camera=_camera_mapping(camera),
        rendered_boxes=(box,),
        collision_records=(
            {
                "box": box,
                "group": "obstacle",
                "kind": "obstacle",
                "object_id": "box",
                "rendered_index": 0,
            },
        ),
        base_xy_yaw=(0.0, 0.0, base_yaw),
        physical_free_mask=grid.free_mask,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
        world_bounds_xy_m=_manifest().world_bounds_xy_m,
        rendered_collision_parity_complete=True,
        output_grid=_output_grid(),
    )


def test_physical_causal_flags_and_precedence_are_explicit() -> None:
    point = np.asarray([0.0])
    inside = np.asarray([True])
    ramp = {
        "box": {
            **_box_mapping(),
            "center_xyz_m": (0.0, 0.0, 0.5),
        },
        "group": "obstacle",
        "kind": "ramp",
        "rendered_index": 0,
    }
    flags = decomposition._physical_blocker_flags(
        blocker_x=point,
        blocker_y=point,
        blocker_inside_grid=inside,
        physical_cell_size_m=0.05,
        world_bounds_xy_m=((-1.0, -1.0), (1.0, 1.0)),
        collision_records=(ramp,),
    )
    assert flags["terrain_surface"] is True
    assert decomposition._physical_category(flags) == "nonfree_terrain_or_surface"

    outside = decomposition._physical_blocker_flags(
        blocker_x=np.asarray([1.0]),
        blocker_y=point,
        blocker_inside_grid=inside,
        physical_cell_size_m=0.05,
        world_bounds_xy_m=((-1.0, -1.0), (1.0, 1.0)),
        collision_records=(),
    )
    assert outside["source_square_outside_world_bounds"] is True
    assert decomposition._physical_category(outside) == (
        "outside_domain_or_world_boundary"
    )


def test_outside_grid_physical_prior_mismatches_all_enter_boundary_bucket() -> None:
    camera = _camera()
    physical_free = np.ones((20, 20), dtype=bool)
    origin = (-0.5, -0.5)
    reconstruction = ray_v2.reconstruct_frame_from_perfect_rays(
        camera=ray_v2.CameraRaySpec.from_camera_observation(camera),
        rendered_obstacle_boxes=(),
        collision_obstacle_boxes=(),
        base_xy_yaw=(0.0, 0.0, 0.0),
        physical_free_mask=physical_free,
        physical_origin_xy_m=origin,
        physical_cell_size_m=0.05,
    )
    report = decomposition.decompose_frame(
        authoritative_labels=reconstruction.contract_labels,
        supervision_mask=np.ones((64, 64), dtype=bool),
        frame_key={"family": "open_obstacle_field", "frame": 0},
        camera=_camera_mapping(camera),
        rendered_boxes=(),
        collision_records=(),
        base_xy_yaw=(0.0, 0.0, 0.0),
        physical_free_mask=physical_free,
        physical_origin_xy_m=origin,
        physical_cell_size_m=0.05,
        world_bounds_xy_m=((-0.45, -0.45), (0.45, 0.45)),
        rendered_collision_parity_complete=True,
        output_grid=_output_grid(),
    )

    count = report["physical_prior_mismatch_cell_count"]
    assert count > 0
    assert report["physical_prior_categories"][
        "outside_domain_or_world_boundary"
    ]["count"] == count
    assert report["physical_prior_categories"]["residual"]["count"] == 0


def test_native_stride1_lattice_recovers_thin_surface_missed_by_stride2() -> None:
    report = _decompose_with_box(_box_mapping(y=-0.08, size_y=0.002))

    assert report["collision_veto_delta_cell_count"] == 2
    assert report["collision_veto_categories"][
        "recovered_by_native_stride1_lattice_absent_from_registered_stride2"
    ]["count"] == 1
    assert report["collision_veto_categories"][
        "no_native_pixel_first_surface_witness"
    ]["count"] == 1


def test_no_native_witness_bucket_is_qualified_and_exhaustive() -> None:
    report = _decompose_with_box(
        _box_mapping(y=0.1, size_x=0.45, size_y=0.8, yaw=0.21),
        base_yaw=0.17,
    )
    categories = report["collision_veto_categories"]

    assert sum(value["count"] for value in categories.values()) == report[
        "collision_veto_delta_cell_count"
    ]
    assert categories["no_native_pixel_first_surface_witness"]["count"] > 0


def test_decomposition_rejects_incomplete_rendered_collision_parity() -> None:
    grid = _grid()
    camera = _camera()
    with pytest.raises(ValueError, match="exact rendered/collision box parity"):
        decomposition.decompose_frame(
            authoritative_labels=np.zeros((64, 64), dtype=np.uint8),
            supervision_mask=np.ones((64, 64), dtype=bool),
            frame_key={"family": "open_obstacle_field", "frame": 0},
            camera=_camera_mapping(camera),
            rendered_boxes=(),
            collision_records=(),
            base_xy_yaw=(0.0, 0.0, 0.0),
            physical_free_mask=grid.free_mask,
            physical_origin_xy_m=grid.origin_xy,
            physical_cell_size_m=grid.cell_size_m,
            world_bounds_xy_m=_manifest().world_bounds_xy_m,
            rendered_collision_parity_complete=False,
            output_grid=_output_grid(),
        )


def test_decomposition_dry_run_is_deterministic_and_fit_is_untouched() -> None:
    command = [
        sys.executable,
        str(
            ROOT
            / "scripts/audit_go2_perfect_camera_ray_field_mismatch_decomposition.py"
        ),
        "--dry-run",
    ]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(first.stdout)

    assert first.stdout == second.stdout
    assert payload["dry_run"] is True
    assert payload["generated_fit_payload_opened"] is False
    assert payload["g2_or_holdout_payload_opened"] is False
    assert payload["physical_partition_count"] == payload[
        "physical_prior_mismatch_cell_count"
    ]
    assert payload["collision_partition_count"] == payload[
        "collision_veto_delta_cell_count"
    ]

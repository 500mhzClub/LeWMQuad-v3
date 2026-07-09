from __future__ import annotations

import numpy as np

from lewm.datasets.go2_paired_navigation import (
    CameraObservation,
    DEFAULT_LOCAL_GRID,
    FREE_CLASS,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
    label_camera_visible_configuration_grid,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


def _manifest_with_cross_corridor_wall() -> SceneManifest:
    wall = BoxObject(
        object_id="wall",
        kind="wall",
        center_xyz_m=(1.2, 0.0, 0.5),
        size_xyz_m=(0.10, 4.0, 1.0),
        yaw_rad=0.0,
        material_id="wall",
    )
    return SceneManifest(
        scene_id="label_test_scene",
        family="unit_test",
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-5.0, -5.0), (6.0, 5.0)),
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
        walls=(wall,),
    )


def _cell_nearest(forward_m: float, left_m: float) -> tuple[int, int]:
    row = int(np.argmin(np.abs(DEFAULT_LOCAL_GRID.forward_centers_m() - forward_m)))
    col = int(np.argmin(np.abs(DEFAULT_LOCAL_GRID.left_centers_m() - left_m)))
    return row, col


def test_default_grid_uses_explicit_edges_and_model_aligned_centers() -> None:
    metadata = DEFAULT_LOCAL_GRID.to_metadata()
    assert metadata["shape"] == [64, 64]
    assert metadata["cell_size_m"] == 0.10
    assert metadata["forward_edge_range_m"] == [-1.0, 5.4]
    assert metadata["left_edge_range_m"] == [-3.2, 3.2]
    assert metadata["forward_center_range_m"] == [-0.95, 5.35]
    assert metadata["left_center_range_m"] == [-3.15, 3.15]
    assert metadata["array_axes"] == {
        "row": "base_forward_increasing",
        "column": "base_left_increasing",
    }
    assert metadata["bounds_are"] == "cell_edges"


def test_first_physical_obstacle_is_occupied_and_cells_behind_are_unknown() -> None:
    manifest = _manifest_with_cross_corridor_wall()
    configuration = InflatedOccupancyGrid(
        manifest, cell_size_m=0.05, inflation_m=0.20
    )
    physical = InflatedOccupancyGrid(
        manifest, cell_size_m=0.05, inflation_m=0.0
    )
    labels, supervision, observed = label_camera_visible_configuration_grid(
        configuration,
        physical_visibility_grid=physical,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=CameraObservation(
            position_xyz_m=(0.326, 0.0, 0.40),
            lookat_xyz_m=(1.326, 0.0, 0.40),
            horizontal_fov_deg=78.323,
            near_m=0.05,
        ),
    )

    before = _cell_nearest(0.55, 0.05)
    behind = _cell_nearest(1.55, 0.05)
    outside_fov = _cell_nearest(0.05, 1.55)
    assert labels[before] == FREE_CLASS
    assert labels[behind] == UNKNOWN_CLASS
    assert labels[outside_fov] == UNKNOWN_CLASS

    center_col = _cell_nearest(0.0, 0.05)[1]
    occupied_rows = np.flatnonzero(labels[:, center_col] == OCCUPIED_CLASS)
    assert occupied_rows.size > 0
    occupied_forward = DEFAULT_LOCAL_GRID.forward_centers_m()[occupied_rows]
    assert np.any((occupied_forward >= 0.85) & (occupied_forward <= 1.15))
    assert supervision.all()
    np.testing.assert_array_equal(observed, labels != UNKNOWN_CLASS)


def test_recorded_camera_look_direction_not_base_yaw_controls_visibility() -> None:
    empty_manifest = _manifest_with_cross_corridor_wall()
    empty_manifest = SceneManifest(
        **{**empty_manifest.__dict__, "walls": ()},
    )
    configuration = InflatedOccupancyGrid(
        empty_manifest, cell_size_m=0.05, inflation_m=0.20
    )
    physical = InflatedOccupancyGrid(
        empty_manifest, cell_size_m=0.05, inflation_m=0.0
    )
    labels, supervision, observed = label_camera_visible_configuration_grid(
        configuration,
        physical_visibility_grid=physical,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=CameraObservation(
            position_xyz_m=(0.0, 0.0, 0.40),
            lookat_xyz_m=(0.0, 1.0, 0.40),
            horizontal_fov_deg=78.323,
            near_m=0.05,
        ),
    )
    assert labels[_cell_nearest(0.55, 0.05)] == UNKNOWN_CLASS
    assert labels[_cell_nearest(0.05, 0.55)] == FREE_CLASS
    assert supervision[_cell_nearest(0.55, 0.05)]
    assert not observed[_cell_nearest(0.55, 0.05)]


def test_uninflated_physical_grid_marks_obstacle_interior_occupied() -> None:
    physical = InflatedOccupancyGrid(
        _manifest_with_cross_corridor_wall(),
        cell_size_m=0.05,
        inflation_m=0.0,
    )
    assert not physical.is_free((1.2, 0.0))
    assert physical.is_free((0.8, 0.0))


def test_body_inflation_changes_target_class_but_not_visibility() -> None:
    manifest = _manifest_with_cross_corridor_wall()
    configuration = InflatedOccupancyGrid(
        manifest, cell_size_m=0.05, inflation_m=0.20
    )
    physical = InflatedOccupancyGrid(
        manifest, cell_size_m=0.05, inflation_m=0.0
    )
    labels, _, _ = label_camera_visible_configuration_grid(
        configuration,
        physical_visibility_grid=physical,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=CameraObservation(
            position_xyz_m=(0.326, 0.0, 0.40),
            lookat_xyz_m=(1.326, 0.0, 0.40),
            horizontal_fov_deg=78.323,
            near_m=0.05,
        ),
    )

    # The wall's physical front face is x=1.15, while its 0.20 m
    # configuration boundary begins at x=0.95. This cell is physically
    # visible and therefore supervised as an OCCUPIED navigation target.
    assert labels[_cell_nearest(1.05, 0.05)] == OCCUPIED_CLASS
    assert labels[_cell_nearest(1.55, 0.05)] == UNKNOWN_CLASS


def test_label_contract_rejects_inflated_visibility_grid() -> None:
    manifest = _manifest_with_cross_corridor_wall()
    configuration = InflatedOccupancyGrid(
        manifest, cell_size_m=0.05, inflation_m=0.20
    )
    with np.testing.assert_raises_regex(
        ValueError, "physical_visibility_grid must use zero obstacle inflation"
    ):
        label_camera_visible_configuration_grid(
            configuration,
            physical_visibility_grid=configuration,
            base_xy_yaw=(0.0, 0.0, 0.0),
            camera=CameraObservation(
                position_xyz_m=(0.326, 0.0, 0.40),
                lookat_xyz_m=(1.326, 0.0, 0.40),
                horizontal_fov_deg=78.323,
                near_m=0.05,
            ),
        )

from __future__ import annotations

import numpy as np

from lewm.datasets.go2_paired_navigation import (
    CameraObservation,
    DEFAULT_LOCAL_GRID,
    FREE_CLASS,
    LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
    label_camera_visible_configuration_grid,
    label_camera_visible_physical_grid,
    derive_configuration_labels_from_fused_physical_raster,
    observable_physical_labels_from_raster,
    post_memory_configuration_morphology_metadata,
    vertical_fov_from_horizontal,
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


def test_post_memory_configuration_morphology_is_exact_and_not_supervised() -> None:
    metadata = post_memory_configuration_morphology_metadata(
        radius_m=0.47,
        physical_cell_size_m=0.10,
    )
    assert metadata["schema"] == (
        "lewm_post_memory_configuration_morphology_v1"
    )
    assert metadata["radius_m"] == 0.47
    assert metadata["memory_cell_size_m"] == 0.10
    assert metadata["applied_during_per_frame_supervision"] is False
    assert metadata["support_is_pose_dependent"] is True
    assert metadata["support_contract_sha256"] == (
        "79ac1cb5e0c83d088b4df41eaa3789fd43c1b94e4470afc19342de0d6b519c1a"
    )


def test_post_memory_morphology_sees_fine_obstacles_without_false_occupied() -> None:
    physical_centers = np.arange(-1.025, 1.026, 0.05)
    output_centers = np.arange(-0.5, 0.501, 0.1)
    output_x, output_y = np.meshgrid(output_centers, output_centers, indexing="ij")
    physical = np.full(
        (physical_centers.size, physical_centers.size), FREE_CLASS, dtype=np.uint8
    )

    def lift() -> np.ndarray:
        return derive_configuration_labels_from_fused_physical_raster(
            physical,
            physical_x_centers_m=physical_centers,
            physical_y_centers_m=physical_centers,
            configuration_world_x_m=output_x,
            configuration_world_y_m=output_y,
            footprint_radius_m=0.47,
            physical_cell_size_m=0.05,
        )

    assert np.all(lift() == FREE_CLASS)
    center = int(np.argmin(np.abs(physical_centers - 0.025)))
    physical[center, center] = UNKNOWN_CLASS
    one_unknown = lift()
    assert np.count_nonzero(one_unknown == FREE_CLASS) == 42
    assert np.count_nonzero(one_unknown == OCCUPIED_CLASS) == 0
    assert np.count_nonzero(one_unknown == UNKNOWN_CLASS) == 79

    # This 0.05 m witness lies between the 0.10 m output centers. It must
    # still expand to occupied configuration cells.
    physical[center, center] = OCCUPIED_CLASS
    one_obstacle = lift()
    assert np.count_nonzero(one_obstacle == FREE_CLASS) == 42
    assert np.count_nonzero(one_obstacle == OCCUPIED_CLASS) == 69
    assert np.count_nonzero(one_obstacle == UNKNOWN_CLASS) == 10

    # A cell square that only touches the disc with its center outside 0.47 m
    # withholds FREE but is not strong enough to assert OCCUPIED.
    physical.fill(FREE_CLASS)
    outside = int(np.argmin(np.abs(physical_centers - 0.475)))
    origin = int(np.argmin(np.abs(physical_centers)))
    physical[outside, origin] = OCCUPIED_CLASS
    one_output = derive_configuration_labels_from_fused_physical_raster(
        physical,
        physical_x_centers_m=physical_centers,
        physical_y_centers_m=physical_centers,
        configuration_world_x_m=np.asarray([[0.0]]),
        configuration_world_y_m=np.asarray([[0.0]]),
        footprint_radius_m=0.47,
        physical_cell_size_m=0.05,
    )
    assert one_output[0, 0] == UNKNOWN_CLASS


def test_observable_physical_aggregation_keeps_morphology_out_of_frame_targets() -> None:
    physical_centers = np.arange(-0.225, 0.226, 0.05)
    output_centers = np.asarray([-0.1, 0.0, 0.1])
    output_x, output_y = np.meshgrid(output_centers, output_centers, indexing="ij")
    physical = np.full(
        (physical_centers.size, physical_centers.size), FREE_CLASS, dtype=np.uint8
    )

    def aggregate(witnesses: np.ndarray | None = None) -> np.ndarray:
        return observable_physical_labels_from_raster(
            physical,
            physical_x_centers_m=physical_centers,
            physical_y_centers_m=physical_centers,
            output_world_x_m=output_x,
            output_world_y_m=output_y,
            output_yaw_rad=0.0,
            physical_cell_size_m=0.05,
            output_cell_size_m=0.10,
            visible_obstacle_first_hit_xy_m=witnesses,
        )

    assert np.all(aggregate() == FREE_CLASS)
    source_index = int(np.argmin(np.abs(physical_centers - 0.025)))
    physical[source_index, source_index] = UNKNOWN_CLASS
    unknown = aggregate()
    assert np.count_nonzero(unknown == FREE_CLASS) == 5
    assert np.count_nonzero(unknown == UNKNOWN_CLASS) == 4

    # The exact first-hit point belongs to only the physical 0.10 m output
    # cell. It is not expanded by the 0.47 m robot radius in this frame target.
    occupied = aggregate(np.asarray([[0.025, 0.025]], dtype=np.float64))
    assert occupied[1, 1] == OCCUPIED_CLASS
    assert np.count_nonzero(occupied == OCCUPIED_CLASS) == 1
    assert np.count_nonzero(occupied == UNKNOWN_CLASS) == 3


def test_hidden_physical_obstacle_cannot_create_observable_configuration_occupied() -> None:
    obstacle = BoxObject(
        object_id="outside_fov_obstacle",
        kind="obstacle",
        center_xyz_m=(0.55, 0.60, 0.5),
        size_xyz_m=(0.10, 0.10, 1.0),
        yaw_rad=0.0,
        material_id="wall",
    )
    base_manifest = _manifest_with_cross_corridor_wall()
    manifest = SceneManifest(
        **{**base_manifest.__dict__, "walls": (), "obstacles": (obstacle,)},
    )
    configuration = InflatedOccupancyGrid(
        manifest, cell_size_m=0.05, inflation_m=0.47
    )
    physical = InflatedOccupancyGrid(
        manifest, cell_size_m=0.05, inflation_m=0.0
    )
    camera = CameraObservation(
        position_xyz_m=(0.0, 0.0, 0.40),
        lookat_xyz_m=(1.0, 0.0, 0.40),
        horizontal_fov_deg=78.323,
        near_m=0.05,
        vertical_fov_deg=vertical_fov_from_horizontal(
            78.323, image_width=640, image_height=480
        ),
    )
    target = _cell_nearest(0.55, 0.15)

    physical_labels, _, _ = label_camera_visible_physical_grid(
        physical,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=camera,
    )
    legacy_labels, _, _ = label_camera_visible_configuration_grid(
        configuration,
        physical_visibility_grid=physical,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=camera,
    )
    observable_labels, _, _ = label_camera_visible_configuration_grid(
        configuration,
        physical_visibility_grid=physical,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=camera,
        label_contract=LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
        obstacle_boxes=manifest.static_objects,
    )

    assert physical_labels[target] == FREE_CLASS
    assert legacy_labels[target] == OCCUPIED_CLASS
    assert observable_labels[target] == UNKNOWN_CLASS


def test_vertical_frustum_separates_free_floor_from_visible_obstacle_column() -> None:
    base_manifest = _manifest_with_cross_corridor_wall()
    empty_manifest = SceneManifest(**{**base_manifest.__dict__, "walls": ()})
    vertical_fov = vertical_fov_from_horizontal(
        78.323, image_width=640, image_height=480
    )
    camera = CameraObservation(
        position_xyz_m=(0.0, 0.0, 0.40),
        lookat_xyz_m=(1.0, 0.0, 0.40),
        up_xyz=(0.0, 0.0, 1.0),
        horizontal_fov_deg=78.323,
        vertical_fov_deg=vertical_fov,
        near_m=0.05,
    )
    empty_configuration = InflatedOccupancyGrid(
        empty_manifest, cell_size_m=0.05, inflation_m=0.47
    )
    empty_physical = InflatedOccupancyGrid(
        empty_manifest, cell_size_m=0.05, inflation_m=0.0
    )
    empty_labels, _, _ = label_camera_visible_configuration_grid(
        empty_configuration,
        physical_visibility_grid=empty_physical,
        obstacle_boxes=(),
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=camera,
        label_contract=LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
    )
    # Horizontally centered, but some footprint floor support is below the
    # 31.4-degree vertical half-FOV, so FREE is not observable.
    assert empty_labels[_cell_nearest(0.55, 0.05)] == UNKNOWN_CLASS

    obstacle = BoxObject(
        object_id="near_vertical_column",
        kind="obstacle",
        center_xyz_m=(0.55, 0.0, 0.5),
        size_xyz_m=(0.05, 0.05, 1.0),
        yaw_rad=0.0,
        material_id="wall",
    )
    obstacle_manifest = SceneManifest(
        **{**empty_manifest.__dict__, "obstacles": (obstacle,)}
    )
    obstacle_configuration = InflatedOccupancyGrid(
        obstacle_manifest, cell_size_m=0.05, inflation_m=0.47
    )
    obstacle_physical = InflatedOccupancyGrid(
        obstacle_manifest, cell_size_m=0.05, inflation_m=0.0
    )
    obstacle_labels, _, _ = label_camera_visible_configuration_grid(
        obstacle_configuration,
        physical_visibility_grid=obstacle_physical,
        obstacle_boxes=obstacle_manifest.static_objects,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera=camera,
        label_contract=LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
    )
    # The ground point is still below the vertical FOV, but the box column at
    # camera height is a direct 3D first hit and therefore valid OCCUPIED evidence.
    assert obstacle_labels[_cell_nearest(0.55, 0.05)] == OCCUPIED_CLASS
    # A nearby body center lies inside the old 0.47 m inflated obstacle region,
    # but per-frame v3 targets remain physical and do not dilate this hit.
    assert obstacle_labels[_cell_nearest(0.95, 0.05)] != OCCUPIED_CLASS


def test_first_hit_uses_full_roll_pitch_yaw_box_transform() -> None:
    import lewm.datasets.go2_paired_navigation as dataset_module

    box = BoxObject(
        object_id="tilted_box",
        kind="rough_ramp",
        center_xyz_m=(2.0, 0.0, 0.5),
        size_xyz_m=(0.8, 0.6, 0.2),
        roll_rad=0.25,
        pitch_rad=0.35,
        yaw_rad=0.20,
        material_id="ramp",
    )
    camera = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
    direction = np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64)
    entry = dataset_module._ray_box_entry_distances(camera, direction, box)[0]

    rotation = dataset_module._box_rotation_matrix(box)
    local_direction = direction[0] @ rotation
    half = 0.5 * np.asarray(box.size_xyz_m)
    local_center_distance = rotation.T @ (
        camera - np.asarray(box.center_xyz_m)
    )
    expected_low = -np.inf
    expected_high = np.inf
    for origin, component, extent in zip(
        local_center_distance, local_direction, half
    ):
        first = (-extent - origin) / component
        second = (extent - origin) / component
        expected_low = max(expected_low, min(first, second))
        expected_high = min(expected_high, max(first, second))
    assert expected_high >= expected_low > 0.0
    np.testing.assert_allclose(entry, expected_low, atol=1e-10)

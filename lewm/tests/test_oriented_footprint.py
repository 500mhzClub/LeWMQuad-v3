from __future__ import annotations

import math

import pytest

from lewm.planning.oriented_footprint import (
    AsymmetricFootprint,
    DirectionalSupportFootprint,
    ManifestDirectionalFootprintFeasibility,
    ManifestFootprintFeasibility,
    OrientedRectangle,
    Pose2D,
    convex_polygon_intersects_rectangle,
    oriented_rectangles_intersect,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    LightingSpec,
    SceneManifest,
    SpawnSpec,
    VisualRandomization,
)


def _box(
    object_id: str,
    *,
    kind: str,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_rad: float = 0.0,
) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind=kind,
        center_xyz_m=(center_xy[0], center_xy[1], 0.5),
        size_xyz_m=(size_xy[0], size_xy[1], 1.0),
        yaw_rad=yaw_rad,
        material_id=kind,
    )


def _manifest(
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-2.0, -2.0),
        (2.0, 2.0),
    ),
    walls: tuple[BoxObject, ...] = (),
    obstacles: tuple[BoxObject, ...] = (),
    landmarks: tuple[BoxObject, ...] = (),
    distractors: tuple[BoxObject, ...] = (),
) -> SceneManifest:
    visual_randomization = None
    if distractors:
        visual_randomization = VisualRandomization(
            material_overrides=(),
            lighting=LightingSpec(
                direction=(0.0, 0.0, -1.0),
                diffuse_rgb=(1.0, 1.0, 1.0),
                specular_rgb=(0.0, 0.0, 0.0),
                ambient_rgb=(0.2, 0.2, 0.2),
            ),
            distractor_objects=distractors,
        )
    return SceneManifest(
        scene_id="oriented_footprint_test",
        family="test",
        difficulty_tier="test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=bounds,
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=obstacles,
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=20.0,
            min_camera_clearance_m=0.1,
        ),
        walls=walls,
        visual_randomization=visual_randomization,
    )


def test_asymmetric_footprint_corners_rotate_about_base_reference() -> None:
    footprint = AsymmetricFootprint(
        forward_m=0.6,
        rear_m=0.2,
        left_m=0.3,
        right_m=0.1,
    )

    axis_aligned = sorted(footprint.corners_at(Pose2D(1.0, 2.0, 0.0)))
    quarter_turn = sorted(
        footprint.corners_at(Pose2D(1.0, 2.0, math.pi / 2.0))
    )

    for actual, expected in zip(
        axis_aligned,
        sorted(((1.6, 2.3), (0.8, 2.3), (0.8, 1.9), (1.6, 1.9))),
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(
        quarter_turn,
        sorted(((0.7, 2.6), (0.7, 1.8), (1.1, 1.8), (1.1, 2.6))),
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_sat_handles_yawed_manifest_box_and_counts_contact_as_collision() -> None:
    diagonal = OrientedRectangle(
        center_xy_m=(0.0, 0.0),
        half_extent_x_m=1.0,
        half_extent_y_m=0.1,
        yaw_rad=math.pi / 4.0,
    )
    on_diagonal = OrientedRectangle(
        center_xy_m=(0.7, 0.7),
        half_extent_x_m=0.1,
        half_extent_y_m=0.1,
        yaw_rad=0.0,
    )
    off_diagonal = OrientedRectangle(
        center_xy_m=(0.7, -0.7),
        half_extent_x_m=0.1,
        half_extent_y_m=0.1,
        yaw_rad=0.0,
    )
    contact_a = OrientedRectangle((0.0, 0.0), 0.2, 0.1, 0.0)
    contact_b = OrientedRectangle((0.3, 0.0), 0.1, 0.1, 0.0)

    assert oriented_rectangles_intersect(diagonal, on_diagonal)
    assert not oriented_rectangles_intersect(diagonal, off_diagonal)
    assert oriented_rectangles_intersect(contact_a, contact_b)


def test_checker_includes_walls_obstacles_landmarks_and_distractors() -> None:
    checker = ManifestFootprintFeasibility(
        _manifest(
            walls=(_box("wall", kind="wall", center_xy=(0.0, 0.0), size_xy=(0.1, 0.1)),),
            obstacles=(
                _box("obstacle", kind="obstacle", center_xy=(0.0, 0.0), size_xy=(0.1, 0.1)),
            ),
            landmarks=(
                _box("landmark", kind="landmark", center_xy=(0.0, 0.0), size_xy=(0.1, 0.1)),
            ),
            distractors=(
                _box("distractor", kind="distractor", center_xy=(0.0, 0.0), size_xy=(0.1, 0.1)),
            ),
        ),
        AsymmetricFootprint.with_half_width(
            forward_m=0.2,
            rear_m=0.2,
            half_width_m=0.1,
        ),
    )

    report = checker.pose_feasibility(Pose2D(0.0, 0.0, 0.0))

    assert not report.feasible
    assert report.inside_world_bounds
    assert report.colliding_object_ids == (
        "wall",
        "obstacle",
        "landmark",
        "distractor",
    )


def test_world_bounds_contain_entire_oriented_asymmetric_footprint() -> None:
    checker = ManifestFootprintFeasibility(
        _manifest(bounds=((-1.0, -1.0), (1.0, 1.0))),
        AsymmetricFootprint.with_half_width(
            forward_m=0.6,
            rear_m=0.2,
            half_width_m=0.1,
        ),
    )

    forward_outside = checker.pose_feasibility(Pose2D(0.5, 0.0, 0.0))
    reversed_inside = checker.pose_feasibility(Pose2D(0.5, 0.0, math.pi))

    assert not forward_outside.feasible
    assert not forward_outside.inside_world_bounds
    assert reversed_inside.feasible
    assert reversed_inside.inside_world_bounds


def test_translation_sweep_detects_thin_wall_between_safe_endpoints() -> None:
    wall = _box(
        "thin_wall",
        kind="wall",
        center_xy=(0.0, 0.0),
        size_xy=(0.02, 1.0),
    )
    checker = ManifestFootprintFeasibility(
        _manifest(walls=(wall,)),
        AsymmetricFootprint.with_half_width(
            forward_m=0.1,
            rear_m=0.1,
            half_width_m=0.1,
        ),
    )
    start = Pose2D(-1.0, 0.0, 0.0)
    end = Pose2D(1.0, 0.0, 0.0)

    report = checker.swept_pose_feasibility(
        start,
        end,
        maximum_corner_step_m=0.02,
    )

    assert checker.is_pose_feasible(start)
    assert checker.is_pose_feasible(end)
    assert not report.feasible
    assert report.first_infeasible_fraction is not None
    assert report.first_infeasible_pose is not None
    assert report.first_infeasible_pose.colliding_object_ids == ("thin_wall",)


def test_turn_sweep_detects_corner_collision_between_safe_endpoints() -> None:
    corner_obstacle = _box(
        "corner_obstacle",
        kind="obstacle",
        center_xy=(0.72, 0.72),
        size_xy=(0.08, 0.08),
    )
    checker = ManifestFootprintFeasibility(
        _manifest(obstacles=(corner_obstacle,)),
        AsymmetricFootprint.with_half_width(
            forward_m=1.0,
            rear_m=1.0,
            half_width_m=0.1,
        ),
    )
    start = Pose2D(0.0, 0.0, 0.0)
    end = Pose2D(0.0, 0.0, math.pi / 2.0)

    report = checker.swept_pose_feasibility(start, end)

    assert checker.is_pose_feasible(start)
    assert checker.is_pose_feasible(end)
    assert not report.feasible
    assert report.first_infeasible_pose is not None
    assert report.first_infeasible_pose.colliding_object_ids == ("corner_obstacle",)


def test_sweep_uses_shortest_yaw_arc_across_pi_boundary() -> None:
    checker = ManifestFootprintFeasibility(
        _manifest(),
        AsymmetricFootprint.with_half_width(
            forward_m=0.4,
            rear_m=0.2,
            half_width_m=0.2,
        ),
    )
    samples = checker.interpolated_sweep(
        Pose2D(0.0, 0.0, math.pi - 0.05),
        Pose2D(0.0, 0.0, -math.pi + 0.05),
        maximum_corner_step_m=0.02,
        maximum_yaw_step_rad=0.05,
    )

    assert len(samples) == 4
    assert all(abs(pose.yaw_rad) > 3.0 for _, pose in samples)
    assert samples[-1][1].yaw_rad == pytest.approx(-math.pi + 0.05)


def test_directional_support_planes_reconstruct_margin_expanded_square() -> None:
    footprint = DirectionalSupportFootprint.from_directional_support(
        {0: 0.4, 90: 0.2, 180: 0.3, 270: 0.1},
        margin_m=0.05,
    )

    for actual, expected in zip(
        sorted(footprint.vertices_xy_m),
        sorted(((-0.35, -0.15), (0.45, -0.15), (0.45, 0.25), (-0.35, 0.25))),
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert footprint.support_m(0) == pytest.approx(0.45)
    assert footprint.support_m(90) == pytest.approx(0.25)
    assert footprint.support_m(180) == pytest.approx(0.35)
    assert footprint.support_m(270) == pytest.approx(0.15)


def test_directional_support_intersection_does_not_invent_rectangle_corners() -> None:
    footprint = DirectionalSupportFootprint.from_directional_support(
        {
            0: 0.4,
            45: 0.42,
            90: 0.2,
            135: 0.35,
            180: 0.3,
            225: 0.32,
            270: 0.15,
            315: 0.36,
        }
    )

    assert footprint.support_m(0) == pytest.approx(0.4)
    assert footprint.support_m(45) == pytest.approx(0.42)
    assert footprint.maximum_vertex_radius_m < math.hypot(0.4, 0.2)


def test_polygon_rectangle_sat_handles_separation_and_contact() -> None:
    triangle = ((-0.2, -0.2), (0.3, -0.2), (0.0, 0.3))
    separated = OrientedRectangle((1.0, 0.0), 0.1, 0.1, math.pi / 4.0)
    contacting = OrientedRectangle((0.4, -0.2), 0.1, 0.1, 0.0)

    assert not convex_polygon_intersects_rectangle(triangle, separated)
    assert convex_polygon_intersects_rectangle(triangle, contacting)


def test_directional_checker_detects_mid_turn_corner_collision() -> None:
    checker = ManifestDirectionalFootprintFeasibility(
        _manifest(
            obstacles=(
                _box(
                    "turn_obstacle",
                    kind="obstacle",
                    center_xy=(0.55, 0.55),
                    size_xy=(0.08, 0.08),
                ),
            )
        ),
        DirectionalSupportFootprint.from_directional_support(
            {
                0: 0.8,
                45: 0.64,
                90: 0.1,
                135: 0.64,
                180: 0.8,
                225: 0.64,
                270: 0.1,
                315: 0.64,
            }
        ),
    )
    start = Pose2D(0.0, 0.0, 0.0)
    end = Pose2D(0.0, 0.0, math.pi / 2.0)

    report = checker.swept_pose_feasibility(start, end)

    assert checker.is_pose_feasible(start)
    assert checker.is_pose_feasible(end)
    assert not report.feasible
    assert report.first_infeasible_pose is not None
    assert report.first_infeasible_pose.colliding_object_ids == ("turn_obstacle",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward_m": 0.0, "rear_m": 0.2, "left_m": 0.1, "right_m": 0.1},
        {"forward_m": 0.2, "rear_m": -0.1, "left_m": 0.1, "right_m": 0.1},
        {"forward_m": math.nan, "rear_m": 0.2, "left_m": 0.1, "right_m": 0.1},
    ],
)
def test_footprint_rejects_nonpositive_or_nonfinite_extents(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AsymmetricFootprint(**kwargs)

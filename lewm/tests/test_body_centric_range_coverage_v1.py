import importlib.util
import math
from pathlib import Path
import time

import numpy as np

from lewm.safety import body_centric_range_coverage_v1 as subject


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _primitive(
    identity: str,
    kind: str,
    data: tuple[float, ...],
    position: tuple[float, float, float],
    *,
    geom_index: int = 0,
    link_index: int = 0,
    link_name: str = "base",
    quaternion: tuple[float, float, float, float] = IDENTITY,
) -> subject.RobotPrimitive:
    return subject.RobotPrimitive(
        identity=identity,
        kind=kind,
        data=data,
        position_xyz_m=position,
        quaternion_wxyz=quaternion,
        geom_index=geom_index,
        link_index=link_index,
        link_name=link_name,
    )


def _spec(
    identity: str,
    kind: str,
    data: tuple[float, ...],
    *,
    geom_index: int,
    link_index: int,
    link_name: str,
) -> subject.RobotPrimitiveSpec:
    return subject.RobotPrimitiveSpec(
        identity=identity,
        kind=kind,
        data=data,
        local_position_xyz_m=(0.0, 0.0, 0.0),
        local_quaternion_wxyz=IDENTITY,
        geom_index=geom_index,
        link_index=link_index,
        link_name=link_name,
    )


def test_sphere_and_finite_capsule_tangents_are_exact():
    sphere = subject.ray_sphere_distances(
        (-2.0, 1.0, 0.0),
        ((1.0, 0.0, 0.0),),
        (0.0, 0.0, 0.0),
        1.0,
    )
    capsule = subject.ray_capsule_distances(
        (-2.0, 1.0, 0.0),
        ((1.0, 0.0, 0.0),),
        (0.0, 0.0, 0.0),
        IDENTITY,
        1.0,
        2.0,
    )
    cap = subject.ray_capsule_distances(
        (0.0, 0.0, 3.0),
        ((0.0, 0.0, -1.0),),
        (0.0, 0.0, 0.0),
        IDENTITY,
        1.0,
        2.0,
    )
    assert sphere[0] == 2.0
    assert capsule[0] == 2.0
    assert cap[0] == 1.0


def test_origin_inside_robot_is_zero_distance_self_return():
    robot = _primitive("base_box", "box", (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), geom_index=4)
    wall = subject.OrientedBox("wall", (2.0, 0.0, 0.0), (0.1, 1.0, 1.0), object_index=2)
    hit = subject.first_hits(
        (0.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0),),
        environment_boxes=(wall,),
        robot_primitives=(robot,),
        near_m=0.05,
        far_m=10.0,
    )
    assert hit.raw_distance_m[0] == 0.0
    assert hit.hit_class[0] == subject.ROBOT_HIT
    assert hit.hit_identity[0] == "base_box"
    assert hit.hit_index[0] == 4
    assert hit.self_return[0]
    assert hit.self_occluded_within_range[0]
    assert hit.near_blind[0]
    assert not hit.valid_return[0]


def test_rotated_oriented_box_uses_full_3d_orientation():
    half_angle = math.pi / 4.0
    box = subject.OrientedBox(
        "rotated",
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 0.5),
        (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)),
    )
    distance = subject.ray_oriented_box_distances((-2.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), box)
    assert distance[0] == 1.0


def test_near_blind_hit_does_not_reveal_farther_wall():
    blind = subject.OrientedBox("blind", (0.03, 0.0, 0.0), (0.01, 0.1, 0.1), object_index=0)
    wall = subject.OrientedBox("wall", (1.0, 0.0, 0.0), (0.1, 1.0, 1.0), object_index=1)
    hit = subject.first_hits(
        (0.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0),),
        environment_boxes=(wall, blind),
        near_m=0.05,
        far_m=10.0,
    )
    assert hit.hit_identity[0] == "blind"
    assert hit.raw_distance_m[0] == 0.019999999999999997
    assert hit.near_blind[0]
    assert not hit.valid_return[0]
    assert np.isnan(hit.reported_distance_m[0])


def test_robot_self_occlusion_and_equal_distance_tie_are_conservative():
    blocker = _primitive("front_capsule", "capsule", (0.1, 0.4), (1.0, 0.0, 0.0), geom_index=7)
    wall = subject.OrientedBox("wall", (2.0, 0.0, 0.0), (0.1, 1.0, 1.0), object_index=3)
    occluded = subject.first_hits(
        (0.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0),),
        environment_boxes=(wall,),
        robot_primitives=(blocker,),
        near_m=0.05,
        far_m=10.0,
    )
    assert occluded.hit_class[0] == subject.ROBOT_HIT
    assert occluded.hit_identity[0] == "front_capsule"
    assert not occluded.environment_return[0]

    robot_tie = _primitive("tie_robot", "sphere", (0.5,), (1.5, 0.0, 0.0), geom_index=2)
    environment_tie = subject.OrientedBox("tie_wall", (1.1, 0.0, 0.0), (0.1, 1.0, 1.0), object_index=0)
    tied = subject.first_hits(
        (0.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0),),
        environment_boxes=(environment_tie,),
        robot_primitives=(robot_tie,),
        near_m=0.05,
        far_m=10.0,
    )
    assert tied.raw_distance_m[0] == 1.0
    assert tied.hit_class[0] == subject.ROBOT_HIT
    assert tied.hit_identity[0] == "tie_robot"


def test_scan_pattern_can_miss_geometry_seen_by_continuum_ray():
    obstacle = subject.OrientedBox("narrow", (2.0, 0.0, 0.0), (0.02, 0.02, 0.05), object_index=1)
    sparse = subject.spherical_directions_fru(np.radians(np.asarray([-2.0, 2.0])), np.asarray([0.0]))
    dense = subject.spherical_directions_fru(np.asarray([0.0]), np.asarray([0.0]))
    sparse_hit = subject.first_hits(
        (0.0, 0.0, 0.0),
        sparse,
        environment_boxes=(obstacle,),
        near_m=0.05,
        far_m=10.0,
    )
    dense_hit = subject.first_hits(
        (0.0, 0.0, 0.0),
        dense,
        environment_boxes=(obstacle,),
        near_m=0.05,
        far_m=10.0,
    )
    visibility = subject.continuum_target_visibility(
        (0.0, 0.0, 0.0),
        IDENTITY,
        (1.98, 0.0, 0.0),
        horizontal_fov_deg=(-180.0, 180.0),
        vertical_fov_deg=(-45.0, 45.0),
        near_m=0.05,
        far_m=10.0,
        environment_boxes=(obstacle,),
        target_object_identity="narrow",
    )
    assert sparse_hit.no_hit.tolist() == [True, True]
    assert dense_hit.environment_return[0]
    assert visibility.nominal_fov_inclusion
    assert visibility.direct_visibility_after_occlusion
    assert visibility.target_object_observable


def test_ground_is_first_surface_but_not_environment_clearance_return():
    direction = subject.spherical_directions_fru(np.asarray([0.0]), np.radians(np.asarray([-45.0])))
    hit = subject.first_hits(
        (0.0, 0.0, 1.0),
        direction,
        ground_z_m=0.0,
        near_m=0.05,
        far_m=10.0,
    )
    assert hit.hit_class[0] == subject.GROUND_HIT
    assert hit.ground_return[0]
    assert not hit.environment_return[0]
    assert np.isclose(hit.raw_distance_m[0], math.sqrt(2.0))


def test_transform_nlerp_uses_shortest_arc_and_series_is_exact():
    q0 = np.asarray([1.0, 0.0, 0.0, 0.0])
    q1 = -np.asarray([0.0, 0.0, 0.0, 1.0])
    position, quaternion = subject.interpolate_transform((0.0, 0.0, 0.0), q0, (2.0, 0.0, 0.0), q1, 0.5)
    assert np.array_equal(position, np.asarray([1.0, 0.0, 0.0]))
    assert np.isclose(abs(float(quaternion @ subject.normalize_quaternion_wxyz(q1))), math.sqrt(0.5))

    positions, quaternions = subject.interpolate_transform_series(
        np.asarray([0.0, 1.0]),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.asarray([q0, q1]),
        np.asarray([0.0, 0.5, 1.0]),
    )
    assert np.array_equal(positions[:, 0], np.asarray([0.0, 1.0, 2.0]))
    assert np.array_equal(quaternions[0], q0)
    assert np.array_equal(quaternions[-1], subject.normalize_quaternion_wxyz(q1))


def test_point_cloud_clearance_corrects_genesis_box_full_extents():
    box = _primitive("base", "box", (2.0, 4.0, 6.0), (0.0, 0.0, 0.0), geom_index=0)
    calf = _primitive(
        "FL_calf",
        "capsule",
        (0.1, 1.0),
        (4.0, 0.0, 0.0),
        geom_index=1,
        link_index=9,
        link_name="FL_calf",
    )
    points = np.asarray([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.2, 0.0, 0.0]])
    distances = subject.point_to_primitive_clearance(points[:2], box)
    reduced = subject.point_cloud_per_link_clearance(points, (calf, box))
    assert np.array_equal(distances, np.asarray([0.0, 0.5]))
    assert reduced.link_names == ("base", "FL_calf")
    assert np.isclose(reduced.minimum_clearance_m[0], 0.0)
    assert np.isclose(reduced.minimum_clearance_m[1], 0.1)
    assert reduced.responsible_geom_index.tolist() == [0, 1]


def test_continuum_visibility_reports_nominal_fov_and_self_occlusion():
    blocker = _primitive("base", "sphere", (0.2,), (1.0, 0.0, 0.0))
    result = subject.continuum_target_visibility(
        (0.0, 0.0, 0.0),
        IDENTITY,
        (2.0, 0.0, 0.0),
        horizontal_fov_deg=(-180.0, 180.0),
        vertical_fov_deg=(-45.0, 45.0),
        near_m=0.05,
        far_m=10.0,
        robot_primitives=(blocker,),
    )
    assert result.nominal_fov_inclusion
    assert result.self_occluded
    assert not result.direct_visibility_after_occlusion
    assert result.first_hit_identity == "base"


def test_fixture_receipt_is_byte_identical():
    first = subject.fixture_receipt()
    second = subject.fixture_receipt()
    first_bytes = subject.canonical_json_bytes(first)
    second_bytes = subject.canonical_json_bytes(second)
    assert first_bytes == second_bytes
    assert first["content_digest"] == second["content_digest"]
    assert first["content_digest"] == subject.canonical_digest(
        {key: value for key, value in first.items() if key != "content_digest"}
    )


def test_scene_api_retains_exact_object_link_and_geom_identity():
    robot = _primitive(
        "FL_calf_geom",
        "sphere",
        (0.25,),
        (1.0, 0.0, 0.0),
        geom_index=17,
        link_index=9,
        link_name="FL_calf",
    )
    wall = subject.OrientedBox("wall", (2.0, 0.0, 0.0), (0.1, 1.0, 1.0), object_index=5)
    result = subject.raycast_scene(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        0.05,
        10.0,
        (wall,),
        (robot,),
        False,
    )
    assert result.hit_kind.tolist() == ["ROBOT_SELF", "NO_HIT"]
    assert result.geom_identity.tolist() == ["FL_calf_geom", ""]
    assert result.geom_index.tolist() == [17, -1]
    assert result.link_name.tolist() == ["FL_calf", ""]
    assert result.link_index.tolist() == [9, -1]
    assert result.object_identity.tolist() == ["", ""]


def test_pose_composition_and_geom_instantiation_are_wxyz_fru():
    half = math.pi / 4.0
    qz90 = (math.cos(half), 0.0, 0.0, math.sin(half))
    spec = subject.RobotPrimitiveSpec(
        identity="geom",
        kind="box",
        data=(2.0, 4.0, 6.0),
        local_position_xyz_m=(1.0, 0.0, 0.0),
        local_quaternion_wxyz=IDENTITY,
        geom_index=3,
        link_index=2,
        link_name="trunk",
    )
    (world,) = subject.instantiate_geoms(
        (spec,),
        {2: ((2.0, 3.0, 4.0), qz90)},
    )
    assert np.allclose(world.position_xyz_m, (2.0, 4.0, 4.0), atol=1.0e-12)
    assert np.allclose(world.box_half_extents_xyz_m, (1.0, 2.0, 3.0), atol=0.0)
    transformed = subject.transform_points(((1.0, 0.0, 0.0),), world.position_xyz_m, world.quaternion_wxyz)
    recovered = subject.inverse_transform_points(transformed, world.position_xyz_m, world.quaternion_wxyz)
    assert np.allclose(recovered, ((1.0, 0.0, 0.0),), atol=1.0e-12)


def test_sparse_and_l2_scan_generators_freeze_order_frequency_and_timing():
    sparse = subject.generate_sparse_scan_pattern(
        azimuth_bins=2,
        vertical_channels_deg=(-10.0, 10.0),
        duration_s=0.1,
    )
    assert sparse.ray_count == 4
    assert np.array_equal(sparse.timestamps_s, np.asarray([0.0, 0.0, 0.05, 0.05]))
    assert np.array_equal(sparse.elevation_rad, np.radians(np.asarray([-10.0, 10.0, -10.0, 10.0])))

    l2 = subject.generate_l2_scan_pattern()
    again = subject.generate_l2_scan_pattern()
    assert l2.ray_count == 6400
    assert l2.timestamps_s[0] == 0.5 / 64_000.0
    assert l2.timestamps_s[-1] == 6399.5 / 64_000.0
    assert l2.parameters["azimuth_frequency_hz"] == 5.55
    assert l2.parameters["vertical_frequency_hz"] == 216.0
    assert l2.parameters["vertical_fov_deg"] == [-6.0, 90.0]
    assert l2.to_serializable()["ray_sha256"] == again.to_serializable()["ray_sha256"]
    phase = (216.0 * l2.timestamps_s[0]) % 1.0
    expected_elevation = -6.0 + 96.0 * (1.0 - abs(2.0 * phase - 1.0))
    assert np.isclose(np.degrees(l2.elevation_rad[0]), expected_elevation)


def test_closest_points_wrapper_excludes_invalid_returns():
    box = _primitive("base", "box", (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    reduced = subject.closest_points_to_scene(
        np.asarray([[1.1, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        (box,),
        valid_environment_return=np.asarray([False, True]),
    )
    assert reduced.responsible_point_index.tolist() == [0]
    assert np.isclose(reduced.minimum_clearance_m[0], 4.0)


def test_combined_named_fixture_gate_is_complete_and_byte_identical():
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics

    expected = {
        "clear full-body sweep",
        "front trunk contact",
        "side trunk contact",
        "front-limb contact",
        "rear-limb contact",
        "calf contact",
        "contact inside the near blind region",
        "contact hidden by robot self-occlusion",
        "contact between scan samples",
        "current-state visible geometry",
        "future-only visible geometry",
        "one safe successor action",
        "zero safe successor actions",
        "exact threshold tie",
        "correct abstention",
        "deterministic H3 route selection",
    }
    first = subject.run_fixtures(metrics_module=metrics)
    second = subject.run_fixtures(metrics_module=metrics)
    assert first["pass"]
    assert set(first["fixtures"]) == expected
    assert all(row["pass"] for row in first["fixtures"].values())
    assert all(row["pass"] for row in first["requirements"].values())
    assert subject.canonical_json_bytes(first) == subject.canonical_json_bytes(second)


def test_vectorized_transform_sampling_matches_scalar_short_arc_nlerp():
    timestamps = np.asarray([0.0, 0.05, 0.1])
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [1.0, 0.5, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    quaternions = np.asarray(
        [
            [IDENTITY, IDENTITY],
            [(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), IDENTITY],
            [(0.0, 0.0, 0.0, -1.0), (0.0, 1.0, 0.0, 0.0)],
        ]
    )
    query = np.asarray([0.0, 0.025, 0.05, 0.075, 0.1])
    sampled_position, sampled_quaternion = subject.interpolate_transform_series_vectorized(
        timestamps, positions, quaternions, query
    )
    for query_index, timestamp in enumerate(query):
        for geom_index in range(2):
            scalar_position, scalar_quaternion = subject.interpolate_transform_series(
                timestamps,
                positions[:, geom_index],
                quaternions[:, geom_index],
                np.asarray([timestamp]),
            )
            assert np.array_equal(sampled_position[query_index, geom_index], scalar_position[0])
            assert np.array_equal(sampled_quaternion[query_index, geom_index], scalar_quaternion[0])


def test_moving_scene_vectorization_matches_looped_first_hits_randomized():
    rng = np.random.default_rng(20260825)
    ray_count = 96
    specs = (
        _spec("sphere", "sphere", (0.25,), geom_index=7, link_index=2, link_name="hip"),
        _spec("capsule", "capsule", (0.12, 0.8), geom_index=3, link_index=4, link_name="calf"),
        _spec("box", "box", (0.6, 0.4, 0.2), geom_index=11, link_index=0, link_name="trunk"),
    )
    origins = rng.uniform((-1.0, -1.0, 0.05), (1.0, 1.0, 1.5), size=(ray_count, 3))
    directions = rng.normal(size=(ray_count, 3))
    positions = rng.uniform((-0.8, -0.8, 0.0), (1.4, 0.8, 1.2), size=(ray_count, len(specs), 3))
    quaternions = rng.normal(size=(ray_count, len(specs), 4))
    quaternions /= np.linalg.norm(quaternions, axis=2, keepdims=True)
    environment = (
        subject.OrientedBox("wall_a", (2.0, 0.0, 0.7), (0.2, 2.0, 0.7), object_index=4),
        subject.OrientedBox(
            "wall_b",
            (-1.5, 0.5, 0.6),
            (0.1, 0.8, 0.6),
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            object_index=2,
        ),
    )
    vectorized = subject.raycast_moving_scene(
        origins,
        directions,
        0.08,
        4.0,
        environment,
        specs,
        positions,
        quaternions,
        0.0,
    )
    for ray_index in range(ray_count):
        primitives = tuple(
            subject.RobotPrimitive(
                identity=spec.identity,
                kind=spec.kind,
                data=spec.data,
                position_xyz_m=tuple(positions[ray_index, spec_index]),
                quaternion_wxyz=tuple(quaternions[ray_index, spec_index]),
                geom_index=spec.geom_index,
                link_index=spec.link_index,
                link_name=spec.link_name,
            )
            for spec_index, spec in enumerate(specs)
        )
        scalar = subject.first_hits(
            origins[ray_index],
            directions[ray_index : ray_index + 1],
            environment_boxes=environment,
            robot_primitives=primitives,
            ground_z_m=0.0,
            near_m=0.08,
            far_m=4.0,
        )
        np.testing.assert_allclose(
            vectorized.raw_distance_m[ray_index], scalar.raw_distance_m[0], rtol=0.0, atol=2.0e-12
        )
        np.testing.assert_allclose(
            vectorized.distance_m[ray_index], scalar.reported_distance_m[0], rtol=0.0, atol=2.0e-12,
            equal_nan=True,
        )
        assert vectorized.hit_kind[ray_index] == subject.HIT_CLASS_NAMES[int(scalar.hit_class[0])]
        assert vectorized.near_blind[ray_index] == scalar.near_blind[0]
        assert vectorized.beyond_far[ray_index] == scalar.beyond_far[0]
        assert vectorized.no_hit[ray_index] == scalar.no_hit[0]
        assert vectorized.self_return[ray_index] == scalar.self_return[0]
        assert vectorized.free_to_far[ray_index] == scalar.free_to_far[0]
        if scalar.hit_class[0] == subject.ROBOT_HIT:
            winner = next(item for item in primitives if item.identity == scalar.hit_identity[0])
            assert vectorized.geom_identity[ray_index] == winner.identity
            assert vectorized.geom_index[ray_index] == winner.geom_index
            assert vectorized.link_name[ray_index] == winner.link_name
            assert vectorized.link_index[ray_index] == winner.link_index
        elif scalar.hit_class[0] in (subject.ENVIRONMENT_HIT, subject.GROUND_HIT):
            assert vectorized.object_identity[ray_index] == scalar.hit_identity[0]
            assert vectorized.object_index[ray_index] == scalar.hit_index[0]


def test_moving_scene_preserves_inside_near_and_robot_tie_semantics():
    specs = (
        _spec("inside", "box", (0.2, 0.2, 0.2), geom_index=4, link_index=0, link_name="trunk"),
        _spec("tie", "sphere", (0.5,), geom_index=2, link_index=1, link_name="head"),
    )
    origins = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    directions = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [1.5, 0.0, 1.0]],
        ]
    )
    quaternions = np.broadcast_to(np.asarray(IDENTITY), (2, 2, 4)).copy()
    wall = subject.OrientedBox("tie_wall", (1.1, 0.0, 1.0), (0.1, 1.0, 1.0), object_index=0)
    result = subject.raycast_moving_scene(
        origins, directions, 0.05, 10.0, (wall,), specs, positions, quaternions, False
    )
    assert result.raw_distance_m.tolist() == [0.0, 1.0]
    assert result.geom_identity.tolist() == ["inside", "tie"]
    assert result.near_blind.tolist() == [True, False]
    assert result.self_return.tolist() == [True, True]
    assert result.valid_return.tolist() == [False, True]


def test_trajectory_tree_clearance_matches_brute_force_exact_reduction():
    rng = np.random.default_rng(731)
    specs = (
        _spec("sphere", "sphere", (0.2,), geom_index=5, link_index=1, link_name="limb"),
        _spec("capsule", "capsule", (0.1, 0.8), geom_index=2, link_index=1, link_name="limb"),
        _spec("box", "box", (1.0, 0.5, 0.3), geom_index=8, link_index=0, link_name="trunk"),
    )
    steps = 4
    clouds = tuple(rng.normal(size=(150 + step, 3)) for step in range(steps))
    positions = rng.normal(scale=0.4, size=(steps, len(specs), 3))
    quaternions = rng.normal(size=(steps, len(specs), 4))
    quaternions /= np.linalg.norm(quaternions, axis=2, keepdims=True)
    accelerated = subject.trajectory_point_cloud_per_link_clearance(
        clouds, specs, positions, quaternions
    )
    brute_minimum = []
    brute_points = []
    brute_geoms = []
    for step in range(steps):
        primitives = tuple(
            subject.RobotPrimitive(
                identity=spec.identity,
                kind=spec.kind,
                data=spec.data,
                position_xyz_m=tuple(positions[step, spec_index]),
                quaternion_wxyz=tuple(quaternions[step, spec_index]),
                geom_index=spec.geom_index,
                link_index=spec.link_index,
                link_name=spec.link_name,
            )
            for spec_index, spec in enumerate(specs)
        )
        brute = subject.point_cloud_per_link_clearance(clouds[step], primitives)
        brute_minimum.append(brute.minimum_clearance_m)
        brute_points.append(brute.responsible_point_index)
        brute_geoms.append(brute.responsible_geom_index)
    np.testing.assert_allclose(accelerated.minimum_clearance_m, np.asarray(brute_minimum), rtol=0.0, atol=0.0)
    assert np.array_equal(accelerated.responsible_point_index, np.asarray(brute_points))
    assert np.array_equal(accelerated.responsible_geom_index, np.asarray(brute_geoms))


def test_trajectory_tree_refines_beyond_64_center_nearest_for_long_capsule():
    spec = _spec("long", "capsule", (0.1, 20.0), geom_index=1, link_index=3, link_name="calf")
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    distractors = np.column_stack((0.2 * np.cos(angles), 0.2 * np.sin(angles), np.zeros(64)))
    target = np.asarray([[0.0, 0.0, 10.1]])
    cloud = np.concatenate((distractors, target), axis=0)
    result = subject.trajectory_point_cloud_per_link_clearance(
        cloud,
        (spec,),
        np.asarray([[[0.0, 0.0, 0.0]]]),
        np.asarray([[IDENTITY]]),
    )
    assert np.isclose(result.minimum_clearance_m[0, 0], 0.0, atol=1.0e-12)
    assert result.responsible_point_index[0, 0] == 64
    assert result.responsible_geom_index[0, 0] == 1


def test_shared_trajectory_cloud_broadcasts_and_builds_one_tree(monkeypatch):
    real_tree = subject.cKDTree
    builds = []

    def counting_tree(*args, **kwargs):
        builds.append(1)
        return real_tree(*args, **kwargs)

    monkeypatch.setattr(subject, "cKDTree", counting_tree)
    spec = _spec("sphere", "sphere", (0.2,), geom_index=1, link_index=0, link_name="body")
    cloud = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = subject.trajectory_point_cloud_per_link_clearance(
        cloud,
        (spec,),
        np.asarray([[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        np.asarray([[IDENTITY], [IDENTITY], [IDENTITY]]),
    )
    assert len(builds) == 1
    assert result.minimum_clearance_m.shape == (3, 1)
    np.testing.assert_allclose(result.minimum_clearance_m[:, 0], [0.8, 0.3, -0.2], atol=0.0)


def _quaternion_from_local_z(axis):
    target = np.asarray(axis, dtype=np.float64)
    target /= np.linalg.norm(target)
    if target[2] < -1.0 + 1.0e-12:
        return (0.0, 1.0, 0.0, 0.0)
    value = np.concatenate(([1.0 + target[2]], np.cross((0.0, 0.0, 1.0), target)))
    value /= np.linalg.norm(value)
    return tuple(value)


def test_exact_sphere_obb_witness_uses_environment_surface_and_signed_clearance():
    box = subject.OrientedBox("wall", (1.0, 0.0, 0.0), (0.25, 0.5, 0.5), object_index=2)
    separated = _primitive("sphere", "sphere", (0.25,), (0.0, 0.0, 0.0))
    witness = subject.primitive_obb_closest_witness(separated, box)
    assert witness.signed_clearance_m == 0.5
    assert witness.separation_distance_m == 0.5
    assert not witness.intersects
    assert witness.primitive_point_world_xyz_m == (0.25, 0.0, 0.0)
    assert witness.environment_point_world_xyz_m == (0.75, 0.0, 0.0)

    overlapping = _primitive("sphere", "sphere", (0.25,), (0.70, 0.0, 0.0))
    overlap = subject.primitive_obb_closest_witness(overlapping, box)
    assert np.isclose(overlap.signed_clearance_m, -0.20, atol=1.0e-15)
    assert overlap.separation_distance_m == 0.0
    assert overlap.intersects
    assert overlap.environment_point_world_xyz_m == (0.75, 0.0, 0.0)


def test_exact_capsule_obb_witness_closes_center_target_counterexample():
    # This frozen counterexample is more than 5.5 cm closer than the OBB point
    # nearest the capsule centre.  It guards the complete segment--AABB solve.
    axis = (-0.8108304953853359, 0.41030836501145507, 0.41737387718303276)
    primitive = _primitive(
        "capsule",
        "capsule",
        (0.08, 0.8),
        (0.0, 0.0, 0.0),
        quaternion=_quaternion_from_local_z(axis),
    )
    box = subject.OrientedBox(
        "counterexample",
        (-0.6146150200467675, 0.1975836088600007, 0.4426929829476769),
        (0.26455115536856416, 0.37105419667681594, 0.063857506832471),
        object_index=9,
    )
    witness = subject.primitive_obb_closest_witness(primitive, box)
    assert np.isclose(witness.signed_clearance_m, 0.13344264798245276, atol=1.0e-14)
    assert witness.signed_clearance_m < 0.18924753793024923 - 0.05
    local_environment = np.asarray(witness.environment_point_world_xyz_m) - np.asarray(
        box.center_xyz_m
    )
    half = np.asarray(box.half_extents_xyz_m)
    assert np.all(np.abs(local_environment) <= half + 1.0e-12)
    assert np.any(np.isclose(np.abs(local_environment), half, rtol=0.0, atol=1.0e-12))


def test_capsule_axis_intersection_is_conservative_negative_and_surface_bound():
    capsule = _primitive("capsule", "capsule", (0.10, 1.0), (0.0, 0.0, 0.0))
    box = subject.OrientedBox("box", (0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    witness = subject.primitive_obb_closest_witness(capsule, box)
    assert witness.signed_clearance_m == -0.10
    assert witness.separation_distance_m == 0.0
    assert witness.intersects
    environment = np.asarray(witness.environment_point_world_xyz_m)
    assert np.any(np.isclose(np.abs(environment), 0.25, rtol=0.0, atol=0.0))


def test_exact_obb_obb_witness_includes_edge_edge_candidates_and_full_extents():
    primitive = _primitive(
        "robot_box",
        "box",
        (2.0, 0.2, 0.2),
        (0.0, 0.0, 0.0),
        quaternion=(
            0.5339459106344904,
            0.004501251360638093,
            -0.6897767337850111,
            -0.4889678525033882,
        ),
    )
    environment = subject.OrientedBox(
        "skew_box",
        (1.1889281211140412, 1.0743914672517265, -1.4915189034401397),
        (1.0, 0.1, 0.1),
        (
            0.3537081197067386,
            -0.7164098327747163,
            0.5335349598813651,
            0.2774670505512626,
        ),
    )
    witness = subject.primitive_obb_closest_witness(primitive, environment)
    assert witness.feature_pair == "EDGE_EDGE"
    assert np.isclose(witness.signed_clearance_m, 0.9098440541818642, atol=1.0e-14)
    assert witness.separation_distance_m == witness.signed_clearance_m


def test_intersecting_obb_uses_sat_signed_overlap_and_environment_surface():
    primitive = _primitive("robot_box", "box", (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    environment = subject.OrientedBox("wall", (1.5, 0.0, 0.0), (1.0, 1.0, 1.0))
    witness = subject.primitive_obb_closest_witness(primitive, environment)
    assert witness.intersects
    assert np.isclose(witness.signed_clearance_m, -0.5, atol=0.0)
    assert witness.separation_distance_m == 0.0
    point = np.asarray(witness.environment_point_world_xyz_m) - np.asarray(environment.center_xyz_m)
    assert np.any(np.isclose(np.abs(point), 1.0, rtol=0.0, atol=0.0))


def _yaw_quaternions(angles: np.ndarray) -> np.ndarray:
    output = np.zeros((len(angles), 4), dtype=np.float64)
    output[:, 0] = np.cos(angles * 0.5)
    output[:, 3] = np.sin(angles * 0.5)
    return output


def _random_quaternions(rng: np.random.Generator, count: int) -> np.ndarray:
    output = rng.normal(size=(count, 4))
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return output


def _evaluator_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluate_body_centric_range_coverage_qualification_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "body_centric_range_coverage_evaluator_test_subject", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batched_box_obb_separated_clearance_and_witnesses_are_exact_randomized():
    rng = np.random.default_rng(20260825)
    count = 192
    primitive_center = rng.uniform(-0.5, 0.5, size=(count, 3))
    primitive_extent = rng.uniform(0.05, 0.7, size=(count, 3))
    environment_half = rng.uniform(0.03, 0.6, size=(count, 3))
    environment_center = primitive_center + np.column_stack(
        (
            rng.uniform(2.0, 3.0, size=count),
            rng.uniform(-0.4, 0.4, size=count),
            rng.uniform(-0.4, 0.4, size=count),
        )
    )
    primitive_quaternion = _random_quaternions(rng, count)
    environment_quaternion = _yaw_quaternions(
        rng.uniform(-math.pi, math.pi, size=count)
    )
    batched = subject.batch_box_obb_closest_witness(
        primitive_center,
        primitive_quaternion,
        primitive_extent,
        environment_center,
        environment_quaternion,
        environment_half,
        chunk_size=47,
    )
    assert not np.any(batched.intersects)
    for index in range(count):
        scalar = subject.primitive_obb_closest_witness(
            _primitive(
                f"robot_{index}",
                "box",
                tuple(primitive_extent[index]),
                tuple(primitive_center[index]),
                geom_index=index,
                quaternion=tuple(primitive_quaternion[index]),
            ),
            subject.OrientedBox(
                f"environment_{index}",
                tuple(environment_center[index]),
                tuple(environment_half[index]),
                tuple(environment_quaternion[index]),
                index,
            ),
        )
        assert np.isclose(
            batched.signed_clearance_m[index],
            scalar.signed_clearance_m,
            rtol=0.0,
            atol=2.0e-14,
        )
        # Parallel features can have a continuum of equally exact witnesses,
        # so scalar and batch coordinates need not choose the same member.
        # The signed distance must match and both batch points must be bound to
        # their respective box surfaces.
        assert np.isclose(
            np.linalg.norm(
                batched.primitive_point_world_xyz_m[index]
                - batched.environment_point_world_xyz_m[index]
            ),
            scalar.signed_clearance_m,
            rtol=0.0,
            atol=2.0e-13,
        )
        primitive_rotation = subject.rotation_matrix_wxyz(
            primitive_quaternion[index]
        )
        primitive_local = (
            batched.primitive_point_world_xyz_m[index]
            - primitive_center[index]
        ) @ primitive_rotation
        primitive_half = primitive_extent[index] * 0.5
        assert np.all(np.abs(primitive_local) <= primitive_half + 2.0e-13)
        assert np.any(
            np.isclose(
                np.abs(primitive_local), primitive_half, rtol=0.0, atol=2.0e-13
            )
        )
        environment_rotation = subject.rotation_matrix_wxyz(
            environment_quaternion[index]
        )
        environment_local = (
            batched.environment_point_world_xyz_m[index]
            - environment_center[index]
        ) @ environment_rotation
        assert np.all(np.abs(environment_local) <= environment_half[index] + 2.0e-13)
        assert np.any(
            np.isclose(
                np.abs(environment_local),
                environment_half[index],
                rtol=0.0,
                atol=2.0e-13,
            )
        )


def test_batched_box_obb_intersection_signed_clearance_matches_scalar():
    rng = np.random.default_rng(731)
    count = 128
    primitive_center = rng.uniform(-0.1, 0.1, size=(count, 3))
    environment_center = primitive_center + rng.uniform(
        -0.08, 0.08, size=(count, 3)
    )
    primitive_extent = rng.uniform(0.3, 0.8, size=(count, 3))
    environment_half = rng.uniform(0.3, 0.8, size=(count, 3))
    primitive_quaternion = _random_quaternions(rng, count)
    environment_quaternion = _yaw_quaternions(
        rng.uniform(-math.pi, math.pi, size=count)
    )
    batched = subject.batch_box_obb_closest_witness(
        primitive_center,
        primitive_quaternion,
        primitive_extent,
        environment_center,
        environment_quaternion,
        environment_half,
    )
    assert np.all(batched.intersects)
    scalar_signed = []
    for index in range(count):
        scalar_signed.append(
            subject.primitive_obb_closest_witness(
                _primitive(
                    f"robot_{index}",
                    "box",
                    tuple(primitive_extent[index]),
                    tuple(primitive_center[index]),
                    geom_index=index,
                    quaternion=tuple(primitive_quaternion[index]),
                ),
                subject.OrientedBox(
                    f"environment_{index}",
                    tuple(environment_center[index]),
                    tuple(environment_half[index]),
                    tuple(environment_quaternion[index]),
                    index,
                ),
            ).signed_clearance_m
        )
    assert np.allclose(
        batched.signed_clearance_m,
        scalar_signed,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_batched_box_obb_has_a_bounded_speed_regression():
    rng = np.random.default_rng(99)
    count = 128
    primitive_center = rng.uniform(-0.5, 0.5, size=(count, 3))
    environment_center = primitive_center + np.asarray([2.5, 0.0, 0.0])
    primitive_extent = rng.uniform(0.05, 0.5, size=(count, 3))
    environment_half = rng.uniform(0.05, 0.5, size=(count, 3))
    primitive_quaternion = _yaw_quaternions(
        rng.uniform(-math.pi, math.pi, size=count)
    )
    environment_quaternion = _yaw_quaternions(
        rng.uniform(-math.pi, math.pi, size=count)
    )

    started = time.perf_counter()
    subject.batch_box_obb_closest_witness(
        primitive_center,
        primitive_quaternion,
        primitive_extent,
        environment_center,
        environment_quaternion,
        environment_half,
    )
    batch_s = time.perf_counter() - started
    started = time.perf_counter()
    for index in range(count):
        subject.primitive_obb_closest_witness(
            _primitive(
                f"robot_{index}",
                "box",
                tuple(primitive_extent[index]),
                tuple(primitive_center[index]),
                geom_index=index,
                quaternion=tuple(primitive_quaternion[index]),
            ),
            subject.OrientedBox(
                f"environment_{index}",
                tuple(environment_center[index]),
                tuple(environment_half[index]),
                tuple(environment_quaternion[index]),
                index,
            ),
        )
    scalar_s = time.perf_counter() - started
    assert batch_s < scalar_s * 0.5


def test_evaluator_box_candidate_pruning_is_exhaustive_equivalent():
    evaluator = _evaluator_module()
    rng = np.random.default_rng(4815)
    specs = [
        _spec(
            f"box_{index}",
            "box",
            tuple(rng.uniform(0.06, 0.35, size=3)),
            geom_index=index,
            link_index=index % 13,
            link_name=f"link_{index % 13}",
        )
        for index in range(27)
    ]
    positions = rng.uniform(
        np.asarray([-0.5, -0.4, 0.05]),
        np.asarray([0.5, 0.4, 0.6]),
        size=(50, 27, 3),
    )
    quaternions = _random_quaternions(rng, 50 * 27).reshape(50, 27, 4)
    boxes = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        radius = 0.25 if index == 0 else 0.8 + 0.15 * index
        boxes.append(
            subject.OrientedBox(
                f"environment_{index}",
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    0.3,
                ),
                (0.08 + 0.01 * (index % 2), 0.25, 0.3),
                (math.cos(angle * 0.5), 0.0, 0.0, math.sin(angle * 0.5)),
                index,
            )
        )
    pruned = evaluator.closest_scene_targets(
        positions, quaternions, specs, boxes
    )
    original_radii = evaluator.primitive_bounding_radii
    evaluator.primitive_bounding_radii = lambda _specs, indices: np.full(
        len(indices), 1.0e6, dtype=np.float64
    )
    try:
        exhaustive = evaluator.closest_scene_targets(
            positions, quaternions, specs, boxes
        )
    finally:
        evaluator.primitive_bounding_radii = original_radii
    for key in ("target_points", "object_index", "geom_index", "clearance_m"):
        assert np.array_equal(pruned[key], exhaustive[key])


def test_evaluator_deduplicates_exact_intersecting_scalar_fallbacks():
    evaluator = _evaluator_module()
    specs = [
        _spec(
            f"box_{index}",
            "box",
            (0.2, 0.2, 0.2),
            geom_index=index,
            link_index=index % 13,
            link_name=f"link_{index % 13}",
        )
        for index in range(27)
    ]
    positions = np.zeros((50, 27, 3), dtype=np.float64)
    quaternions = np.zeros((50, 27, 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    boxes = [
        subject.OrientedBox(
            "inner", (0.0, 0.0, 0.0), (0.6, 0.6, 0.6), object_index=0
        ),
        subject.OrientedBox(
            "outer", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), object_index=1
        ),
    ]
    scalar = subject.primitive_spec_at_geom_pose_obb_closest_witness
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return scalar(*args, **kwargs)

    subject.primitive_spec_at_geom_pose_obb_closest_witness = counted
    try:
        result = evaluator.closest_scene_targets(
            positions, quaternions, specs, boxes
        )
    finally:
        subject.primitive_spec_at_geom_pose_obb_closest_witness = scalar
    assert calls == 13
    assert np.all(result["object_index"] == 1)
    assert np.all(result["clearance_m"] == -2.1)


def test_spec_pose_wrappers_distinguish_geom_pose_from_link_pose():
    spec = subject.RobotPrimitiveSpec(
        identity="offset_sphere",
        kind="sphere",
        data=(0.1,),
        local_position_xyz_m=(0.5, 0.0, 0.0),
        local_quaternion_wxyz=IDENTITY,
        geom_index=0,
        link_index=0,
        link_name="base",
    )
    box = subject.OrientedBox("wall", (1.0, 0.0, 0.0), (0.1, 0.5, 0.5))
    from_link = subject.primitive_spec_at_link_pose_obb_closest_witness(
        spec, (0.0, 0.0, 0.0), IDENTITY, box
    )
    from_geom = subject.primitive_spec_at_geom_pose_obb_closest_witness(
        spec, (0.5, 0.0, 0.0), IDENTITY, box
    )
    assert from_link == from_geom
    assert np.isclose(from_link.signed_clearance_m, 0.3, atol=1.0e-15)

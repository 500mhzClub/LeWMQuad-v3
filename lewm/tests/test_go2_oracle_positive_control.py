from __future__ import annotations

from pathlib import Path

from lewm.benchmarks.go2_oracle_positive_control import (
    OracleConfig,
    Pose2D,
    _development_path_guard,
    reachable_component,
    run_scene,
    simulate_primitive,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm_genesis.lewm_contract import PrimitiveRegistry
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


REPO_ROOT = Path(__file__).resolve().parents[2]


def _box(
    object_id: str,
    kind: str,
    x: float,
    y: float,
    sx: float,
    sy: float,
    material_id: str,
) -> BoxObject:
    return BoxObject(
        object_id=object_id,
        kind=kind,
        center_xyz_m=(x, y, 0.44),
        size_xyz_m=(sx, sy, 0.88),
        yaw_rad=0.0,
        material_id=material_id,
    )


def _manifest(
    *,
    walls: tuple[BoxObject, ...] = (),
    landmarks: tuple[BoxObject, ...] | None = None,
) -> SceneManifest:
    if landmarks is None:
        landmarks = tuple(
            _box(
                f"landmark_{color}",
                "landmark",
                x,
                y,
                0.30,
                0.30,
                f"landmark_{color}",
            )
            for color, x, y in (
                ("red", -1.0, -1.0),
                ("blue", -1.0, 1.0),
                ("green", 1.0, -1.0),
                ("yellow", 1.0, 1.0),
            )
        )
    return SceneManifest(
        scene_id="oracle_toy",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(GraphNode(node_id=0, center_xy_m=(0.0, 0.0), width_m=1.0),),
        graph_edges=(),
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        walls=walls,
    )


def _registry() -> PrimitiveRegistry:
    return PrimitiveRegistry.from_yaml(REPO_ROOT / "config/go2_primitive_registry.yaml")


def _boundary_walls() -> tuple[BoxObject, ...]:
    return (
        _box("boundary_n", "wall", 0.0, 1.95, 4.0, 0.10, "wall"),
        _box("boundary_s", "wall", 0.0, -1.95, 4.0, 0.10, "wall"),
        _box("boundary_e", "wall", 1.95, 0.0, 0.10, 4.0, "wall"),
        _box("boundary_w", "wall", -1.95, 0.0, 0.10, 4.0, "wall"),
    )


def _geometry():
    return load_geometry_contract(
        REPO_ROOT / "config/go2_generalization_geometry_v1.json",
        repository_root=REPO_ROOT,
    )


def _geometry_v2():
    return load_geometry_contract(
        REPO_ROOT / "config/go2_generalization_geometry_v2.json",
        repository_root=REPO_ROOT,
    )


def test_reachable_component_matches_astar_corner_rule() -> None:
    wall_a = _box("wall_a", "wall", 0.0, 0.55, 0.70, 0.10, "wall")
    wall_b = _box("wall_b", "wall", 0.55, 0.0, 0.10, 0.70, "wall")
    grid = InflatedOccupancyGrid(
        _manifest(walls=(wall_a, wall_b), landmarks=()),
        cell_size_m=0.05,
        inflation_m=0.20,
    )
    start, component, _ = reachable_component(grid, (0.0, 0.0))
    assert start in component
    for neighbor in component:
        assert grid.free_mask[neighbor]


def test_primitive_execution_rejects_configuration_space_collision() -> None:
    wall = _box("front_wall", "wall", 0.28, 0.0, 0.10, 1.0, "wall")
    manifest = _manifest(walls=(wall,), landmarks=())
    grid = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.20)
    outcome = simulate_primitive(
        Pose2D(0.0, 0.0, 0.0),
        "forward_medium",
        _registry(),
        grid,
        OracleConfig.from_geometry_contract(_geometry()),
    )
    assert not outcome.completed
    assert outcome.blocked_reason == "inflated_center_grid"
    assert outcome.end_pose.x <= 0.051
    assert grid.is_free(outcome.end_pose.xy)


def test_open_scene_positive_control_claims_all_four() -> None:
    report = run_scene(
        _manifest(walls=_boundary_walls()),
        _registry(),
        OracleConfig.from_geometry_contract(
            _geometry(),
            max_ticks=1000,
            max_goal_ticks=220,
            coverage_resolution_m=0.60,
            coverage_visit_radius_m=0.38,
            coverage_completion_fraction=0.75,
        ),
        _geometry(),
    )
    assert report["all_beacons_claimed"]
    assert report["claimed_count"] == 4
    assert report["failure_class"] in {"success", "budget"}
    assert 0.0 <= report["normalized_coverage_auc"] <= 1.0
    assert report["collisions"] == 0


def test_geometry_v2_routes_through_shared_map_and_scores_directional_trajectory() -> None:
    geometry = _geometry_v2()
    report = run_scene(
        _manifest(walls=_boundary_walls()),
        _registry(),
        OracleConfig.from_geometry_contract(
            geometry,
            max_ticks=1000,
            max_goal_ticks=220,
            coverage_resolution_m=0.60,
            coverage_visit_radius_m=0.38,
            coverage_completion_fraction=0.75,
        ),
        geometry,
    )

    assert report["all_beacons_claimed"]
    assert report["route_planner"]["source"] == "OnlineBeliefMap.shortest_path"
    assert report["route_planner"]["queries"] > 0
    assert report["shared_map_agreement"]["online_topology_agrees"]
    assert report["shared_map_agreement"]["resolution_is_conservative"]
    assert report["spawn_snap_m"] == 0.0
    assert report["directional_policy"]["profile"] == "observed_max_plus_margin"
    assert report["directional_polygon_initial_pose_feasible"]
    assert report["directional_polygon_sweep_samples_evaluated"] > 0
    assert report["directional_polygon_collision_segments"] == 0
    assert report["strict_directional_safe"]
    assert report["trajectory_pose_count"] == len(
        report["actual_yaw_microstep_trajectory"]
    )
    assert len(report["trajectory_sha256"]) == 64


def test_repeated_colors_are_claimed_by_unique_object_id() -> None:
    landmarks = tuple(
        _box(
            f"landmark_{index:02d}_{color}",
            "landmark",
            x,
            y,
            0.30,
            0.30,
            f"landmark_{color}",
        )
        for index, (color, x, y) in enumerate(
            (
                ("red", -1.0, -1.0),
                ("blue", -1.0, 1.0),
                ("green", 1.0, -1.0),
                ("yellow", 1.0, 1.0),
                ("red", 0.0, -1.2),
                ("blue", 0.0, 1.2),
            )
        )
    )
    report = run_scene(
        _manifest(walls=_boundary_walls(), landmarks=landmarks),
        _registry(),
        OracleConfig.from_geometry_contract(
            _geometry(),
            max_ticks=1200,
            max_goal_ticks=240,
            coverage_resolution_m=0.60,
            coverage_visit_radius_m=0.38,
            coverage_completion_fraction=0.75,
        ),
        _geometry(),
    )
    assert report["all_beacons_claimed"]
    assert report["claimed_count"] == 6
    assert len(report["claimed_beacon_ids"]) == 6
    assert report["claimed_colors"].count("red") == 2
    assert report["claimed_colors"].count("blue") == 2


def test_missing_claim_anchor_is_scene_geometry() -> None:
    beacon = _box(
        "landmark_red",
        "landmark",
        1.0,
        1.0,
        0.30,
        0.30,
        "landmark_red",
    )
    walls = (
        _box("north", "wall", 1.0, 1.42, 1.0, 0.10, "wall"),
        _box("south", "wall", 1.0, 0.58, 1.0, 0.10, "wall"),
        _box("east", "wall", 1.42, 1.0, 0.10, 1.0, "wall"),
        _box("west", "wall", 0.58, 1.0, 0.10, 1.0, "wall"),
    )
    report = run_scene(
        _manifest(walls=walls, landmarks=(beacon,)),
        _registry(),
        OracleConfig.from_geometry_contract(
            _geometry(),
            max_ticks=100,
            claim_distance_m=0.55,
            preferred_standoff_m=0.50,
        ),
        _geometry(),
    )
    assert report["failure_class"] == "scene_geometry"
    assert report["claimed_count"] == 0
    assert any("no connected true-claim anchor" in item for item in report["geometry_failures"])


def test_cli_guard_rejects_sealed_paths() -> None:
    try:
        _development_path_guard(Path(".generated/sealed/final.json"), label="output")
    except ValueError:
        pass
    else:
        raise AssertionError("sealed output path was accepted")

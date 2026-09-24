from __future__ import annotations

import inspect
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lewm.benchmarks.go2_oracle_positive_control import (
    CANONICAL_PHYSICAL_CLAIM_REGRESSION_OUTPUT,
    OracleConfig,
    Pose2D,
    _development_path_guard,
    _generic_output_path_guard,
    _oracle_claim_attempt_id,
    bind_oracle_claim_task,
    merge_indexed_scene_reports,
    reachable_component,
    run_development_suite,
    run_scene,
    simulate_primitive,
)
from lewm.benchmarks.go2_physical_eligibility import (
    directional_footprint_semantic_sha256,
    policy_from_geometry_contract,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm.planning.oriented_footprint import DirectionalSupportFootprint
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
    claim_trace = report["canonical_physical_claim_trace"]
    assert len(claim_trace["controller_claim_attempts"]) == 4
    assert len(claim_trace["physical_claim_evaluations"]) == 4
    assert claim_trace["physical_claim_summary"]["credited_count"] == 4
    assert claim_trace["physical_claim_summary"]["all_targets_claimed"] is True
    assert len(
        {item["event_id"] for item in claim_trace["controller_claim_attempts"]}
    ) == 4
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


def test_generic_cli_cannot_publish_authoritative_claim_report() -> None:
    with pytest.raises(ValueError, match="reserved"):
        _generic_output_path_guard(CANONICAL_PHYSICAL_CLAIM_REGRESSION_OUTPUT)


def test_oracle_task_binding_and_attempt_identity_are_exact() -> None:
    manifest = _manifest(walls=_boundary_walls())
    binding = bind_oracle_claim_task(manifest)
    assert binding.task_object_ids == tuple(
        sorted(item.object_id for item in manifest.landmarks)
    )
    task_object_id = binding.task_object_ids[0]
    expected = hashlib.sha256(
        json.dumps(
            {
                "domain": "lewm-go2-oracle-claim-attempt-v1",
                "episode_id": binding.episode_id,
                "scene_id": manifest.scene_id,
                "task_object_id": task_object_id,
                "trace_id": binding.trace_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert _oracle_claim_attempt_id(
        trace_id=binding.trace_id,
        episode_id=binding.episode_id,
        scene_id=manifest.scene_id,
        task_object_id=task_object_id,
    ) == expected


def test_failed_terminals_produce_no_attempts_or_retries(monkeypatch) -> None:
    from lewm.benchmarks import go2_oracle_positive_control as oracle

    calls = []

    def fail_terminal(*args, **kwargs):
        calls.append((args, kwargs))
        return False, "synthetic_terminal_failure"

    monkeypatch.setattr(oracle, "drive_to_goal", fail_terminal)
    report = run_scene(
        _manifest(walls=_boundary_walls()),
        _registry(),
        OracleConfig.from_geometry_contract(_geometry(), max_ticks=1000),
        _geometry(),
    )
    trace = report["canonical_physical_claim_trace"]
    assert len(calls) == 4
    assert trace["controller_claim_attempts"] == []
    assert trace["physical_claim_evaluations"] == []
    assert len(report["follower_failures"]) == 4


def test_parallel_scene_reports_merge_in_manifest_order() -> None:
    scene_ids = ["scene_z", "scene_a", "scene_m"]
    completed = [
        (1, {"scene_id": "scene_a"}),
        (2, {"scene_id": "scene_m"}),
        (0, {"scene_id": "scene_z"}),
    ]
    assert merge_indexed_scene_reports(
        completed, expected_scene_ids=scene_ids
    ) == [
        {"scene_id": "scene_z"},
        {"scene_id": "scene_a"},
        {"scene_id": "scene_m"},
    ]
    with pytest.raises(ValueError, match="duplicated"):
        merge_indexed_scene_reports(
            [completed[0], completed[0], completed[2]],
            expected_scene_ids=scene_ids,
        )
    with pytest.raises(ValueError, match="scene identity"):
        merge_indexed_scene_reports(
            [(0, {"scene_id": "scene_a"}), completed[0], completed[1]],
            expected_scene_ids=scene_ids,
        )
    with pytest.raises(ValueError, match="outside"):
        merge_indexed_scene_reports(
            [completed[0], completed[1], (3, {"scene_id": "other"})],
            expected_scene_ids=scene_ids,
        )
    with pytest.raises(ValueError, match="incomplete"):
        merge_indexed_scene_reports(completed[:2], expected_scene_ids=scene_ids)


def test_two_worker_scene_execution_is_byte_identical_to_serial(
    tmp_path: Path,
) -> None:
    scene_ids = ["synthetic_z", "synthetic_a"]
    for scene_id in scene_ids:
        manifest = replace(
            _manifest(walls=_boundary_walls()),
            scene_id=scene_id,
        )
        scene_dir = tmp_path / "test_id" / "test" / scene_id
        scene_dir.mkdir(parents=True)
        (scene_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
    geometry = _geometry()
    config = OracleConfig.from_geometry_contract(
        geometry,
        max_ticks=20,
        max_goal_ticks=5,
    )
    kwargs = {
        "scene_corpus": tmp_path,
        "split": "test_id",
        "family": "test",
        "scene_ids": scene_ids,
        "registry": _registry(),
        "config": config,
        "geometry_contract": geometry,
    }
    serial = run_development_suite(**kwargs, workers=1)
    parallel = run_development_suite(**kwargs, workers=2)
    serial_scientific = dict(serial)
    parallel_scientific = dict(parallel)
    assert serial_scientific.pop("scene_execution") == {
        "kind": "serial",
        "worker_count": 1,
        "threads_per_worker": 1,
        "merge_order": "development_manifest_index",
        "worker_runtime_input_file_access": False,
    }
    assert parallel_scientific.pop("scene_execution") == {
        "kind": "spawn_process",
        "worker_count": 2,
        "threads_per_worker": 1,
        "merge_order": "development_manifest_index",
        "worker_runtime_input_file_access": False,
    }
    assert json.dumps(
        serial_scientific,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") == json.dumps(
        parallel_scientific,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def test_preloaded_directional_policy_is_reused_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.benchmarks import go2_oracle_positive_control as oracle

    geometry = _geometry_v2()
    policy = policy_from_geometry_contract(geometry, repository_root=REPO_ROOT)
    manifest = _manifest(walls=_boundary_walls())
    captured: list[object] = []

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("preloaded policy must suppress the suite loader")

    def fake_scene(job):
        index, scene, _registry_value, _config, _geometry_value, job_policy = job
        captured.append(job_policy)
        return index, {
            "scene_id": scene.scene_id,
            "failure_class": "budget",
            "all_beacons_claimed": False,
            "beacon_count": 4,
            "success": False,
            "claimed_count": 0,
            "normalized_coverage_auc": 0.0,
            "final_coverage_fraction": 0.0,
            "collisions": 0,
            "stalls": 0,
            "directional_polygon_collision_segments": 0,
            "strict_directional_safe": True,
            "route_planner": {"source": "OnlineBeliefMap.shortest_path"},
            "ticks": 0,
        }

    monkeypatch.setattr(oracle, "policy_from_geometry_contract", fail_loader)
    monkeypatch.setattr(oracle, "_run_indexed_development_scene", fake_scene)
    run_development_suite(
        scene_corpus=Path("unused"),
        split="development",
        family="test",
        scene_ids=[manifest.scene_id],
        registry=_registry(),
        config=OracleConfig.from_geometry_contract(geometry, max_ticks=1),
        geometry_contract=geometry,
        workers=1,
        preloaded_scene_manifests={manifest.scene_id: manifest},
        preloaded_directional_policy=policy,
    )
    assert captured == [policy]
    assert captured[0] is policy


def test_directional_policy_loader_reads_artifact_bytes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry_v2()
    source = geometry.source_artifacts["directional_footprint_policy"]
    expected_path = (REPO_ROOT / str(source["path"])).resolve()
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def counted_read_bytes(path: Path) -> bytes:
        if path.resolve() == expected_path:
            reads.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    policy_from_geometry_contract(geometry, repository_root=REPO_ROOT)
    assert reads == [expected_path]


def test_preloaded_directional_policy_must_match_geometry_contract() -> None:
    from lewm.benchmarks import go2_oracle_positive_control as oracle

    geometry = _geometry_v2()
    policy = policy_from_geometry_contract(geometry, repository_root=REPO_ROOT)
    with pytest.raises(ValueError, match="wrong exact type"):
        oracle._directional_policy_for_suite(geometry, object())
    with pytest.raises(ValueError, match="invalid field types"):
        oracle._directional_policy_for_suite(
            geometry,
            replace(policy, source_path="not-a-path"),
        )
    with pytest.raises(ValueError, match="content bytes do not match"):
        oracle._directional_policy_for_suite(
            geometry,
            replace(policy, content_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="only valid for geometry contract v2"):
        oracle._directional_policy_for_suite(_geometry(), policy)


def test_preloaded_policy_footprint_is_bound_to_verified_content() -> None:
    from lewm.benchmarks import go2_oracle_positive_control as oracle

    geometry = _geometry_v2()
    policy = policy_from_geometry_contract(geometry, repository_root=REPO_ROOT)
    original = policy.footprint
    radius = original.maximum_vertex_radius_m

    diamond = DirectionalSupportFootprint(
        vertices_xy_m=(
            (radius, 0.0),
            (0.0, radius),
            (-radius, 0.0),
            (0.0, -radius),
        ),
        support_angles_deg=original.support_angles_deg,
        support_values_m=original.support_values_m,
        margin_m=original.margin_m,
    )
    rotated = DirectionalSupportFootprint(
        vertices_xy_m=original.vertices_xy_m[1:] + original.vertices_xy_m[:1],
        support_angles_deg=original.support_angles_deg,
        support_values_m=original.support_values_m,
        margin_m=original.margin_m,
    )
    scaled = DirectionalSupportFootprint(
        vertices_xy_m=tuple(
            (x_m * 0.999, y_m * 0.999)
            for x_m, y_m in original.vertices_xy_m
        ),
        support_angles_deg=original.support_angles_deg,
        support_values_m=original.support_values_m,
        margin_m=original.margin_m,
    )
    for mutated in (diamond, rotated, scaled):
        forged = replace(
            policy,
            footprint=mutated,
            footprint_semantic_sha256=directional_footprint_semantic_sha256(
                mutated
            ),
        )
        with pytest.raises(ValueError, match="bound policy content"):
            oracle._directional_policy_for_suite(geometry, forged)

    malformed = object.__new__(DirectionalSupportFootprint)
    object.__setattr__(
        malformed,
        "vertices_xy_m",
        [list(vertex) for vertex in original.vertices_xy_m],
    )
    object.__setattr__(malformed, "support_angles_deg", original.support_angles_deg)
    object.__setattr__(malformed, "support_values_m", original.support_values_m)
    object.__setattr__(malformed, "margin_m", original.margin_m)
    with pytest.raises(ValueError, match="exact finite tuples"):
        oracle._directional_policy_for_suite(
            geometry,
            replace(policy, footprint=malformed),
        )


def test_oracle_source_has_no_private_claim_acceptance_or_per_tick_update() -> None:
    from lewm.benchmarks import go2_oracle_positive_control as oracle

    source = inspect.getsource(oracle)
    assert "def _true_claim" not in source
    assert "def update_claims" not in source
    assert ".update_claims(" not in source
    run_scene_source = inspect.getsource(oracle.run_scene)
    assert run_scene_source.index("claim_task_binding = bind_oracle_claim_task") < (
        run_scene_source.index("drive_to_goal(")
    )

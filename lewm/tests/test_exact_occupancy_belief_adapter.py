from __future__ import annotations

from pathlib import Path

import pytest

from lewm.benchmarks.go2_belief_map_positive_control import (
    _development_path_guard,
)
from lewm.planning.exact_occupancy_belief_adapter import (
    ExactOccupancyBeliefAdapter,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    GraphNode,
    SceneManifest,
    SpawnSpec,
)


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


def _manifest() -> SceneManifest:
    walls = (
        _box("boundary_n", "wall", 0.0, 1.95, 4.0, 0.10, "wall"),
        _box("boundary_s", "wall", 0.0, -1.95, 4.0, 0.10, "wall"),
        _box("boundary_e", "wall", 1.95, 0.0, 0.10, 4.0, "wall"),
        _box("boundary_w", "wall", -1.95, 0.0, 0.10, 4.0, "wall"),
    )
    landmark = _box(
        "landmark_00_landmark_red",
        "landmark",
        1.20,
        0.0,
        0.30,
        0.30,
        "landmark_red",
    )
    return SceneManifest(
        scene_id="belief_adapter_toy",
        family="test",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec(
            xyz_m=(-1.0, 0.0, 0.375),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(
            GraphNode(node_id=0, center_xy_m=(-1.0, 0.0), width_m=1.0),
            GraphNode(node_id=1, center_xy_m=(1.0, 0.0), width_m=1.0),
        ),
        graph_edges=(),
        obstacles=(),
        landmarks=(landmark,),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        walls=walls,
    )


def _geometry():
    return load_geometry_contract(
        REPO_ROOT / "config/go2_generalization_geometry_v1.json",
        repository_root=REPO_ROOT,
    )


def test_full_exact_load_matches_online_reference_and_routes_claim() -> None:
    adapter = ExactOccupancyBeliefAdapter(_manifest(), _geometry())

    agreement = adapter.load()
    route = adapter.connected_claim_route(_manifest().landmarks[0])

    assert agreement.online_topology_agrees
    assert agreement.component_symmetric_difference_cells == 0
    assert agreement.frontier_symmetric_difference_cells == 0
    assert agreement.map_frontier_cells == 0
    assert agreement.confirmed_free_cells > 0
    assert agreement.confirmed_occupied_cells > 0
    assert 0.0 <= agreement.resolution_jaccard <= 1.0
    assert agreement.resolution_is_conservative
    assert route is not None
    assert route.object_id == "landmark_00_landmark_red"
    assert route.path_cells > 1
    assert route.target_distance_m <= _geometry().visibility_and_claim.claim_radius_m
    assert route.oracle_endpoint_connected
    assert adapter.belief_map.shortest_path(
        adapter.spawn_cell,
        route.endpoint_cell,
    ) is not None


def test_partial_exact_load_has_matching_nontrivial_frontiers() -> None:
    adapter = ExactOccupancyBeliefAdapter(_manifest(), _geometry())
    spawn = adapter.spawn_cell
    observed = {
        cell
        for cell in adapter.all_online_cells
        if abs(cell[0] - spawn[0]) <= 3 and abs(cell[1] - spawn[1]) <= 3
    }

    agreement = adapter.load(observed)

    assert agreement.online_topology_agrees
    assert agreement.map_component_cells > 0
    assert agreement.map_frontier_cells > 0
    assert agreement.map_frontier_cells == agreement.online_reference_frontier_cells
    outside_free = next(
        cell
        for cell in adapter.all_online_cells - observed
        if adapter.online_traversable[cell]
    )
    assert adapter.belief_map.shortest_path(spawn, outside_free) is None


def test_adapter_rejects_observations_outside_scene_grid() -> None:
    adapter = ExactOccupancyBeliefAdapter(_manifest(), _geometry())

    try:
        adapter.load({(10_000, 10_000)})
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-grid exact evidence was accepted")


def test_shared_map_positive_control_rejects_sealed_paths() -> None:
    with pytest.raises(ValueError, match="development-only"):
        _development_path_guard(
            Path(".generated/sealed/shared_map.json"),
            label="output",
        )

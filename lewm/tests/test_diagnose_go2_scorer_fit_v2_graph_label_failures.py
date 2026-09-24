from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/diagnose_go2_scorer_fit_v2_graph_label_failures.py"
SPEC = importlib.util.spec_from_file_location("graph_label_diagnostic", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def _scene_graph(nodes, edges):
    import sys

    package_root = ROOT / "lewm_worlds"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from lewm_worlds.manifest import (
        CameraValidityConstraints,
        GraphEdge,
        GraphNode,
        SceneManifest,
        SpawnSpec,
    )
    from lewm_worlds.scene_graph import SceneGraph

    manifest = SceneManifest(
        scene_id="synthetic_graph_boundary",
        family="synthetic",
        difficulty_tier="test",
        topology_seed=0,
        visual_seed=0,
        physics_seed=0,
        world_bounds_xy_m=((-5.0, -5.0), (5.0, 5.0)),
        spawn=SpawnSpec((0.0, 0.0, 0.35), (1.0, 0.0, 0.0, 0.0)),
        graph_nodes=tuple(
            GraphNode(index, tuple(xy), 0.8) for index, xy in enumerate(nodes)
        ),
        graph_edges=tuple(GraphEdge(a, b, 0.8) for a, b in edges),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(0.1, 0.05, 200.0, 0.05),
    )
    return SceneGraph(manifest)


def test_frozen_camera_pose_inverse_round_trip() -> None:
    yaw = 0.37
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    base = np.asarray([1.25, -0.8, 0.36])
    camera = base + rotation @ D.CAMERA_MOUNT_XYZ_BODY_M
    forward = rotation[:, 0]
    up = rotation[:, 2]
    recovered = D.recover_base_pose(
        {
            "position": camera.tolist(),
            "lookat": (camera + forward).tolist(),
            "up": up.tolist(),
        }
    )
    assert recovered["position_world_xyz_m"] == pytest.approx(base, abs=1e-12)
    assert recovered["roll_pitch_yaw_world_rad"] == pytest.approx(
        [0.0, 0.0, yaw], abs=1e-12
    )


def test_sparse_graph_coverage_boundary_is_not_nearest_node_bug() -> None:
    graph = _scene_graph([(0.0, 0.0)], [])
    accepted = graph.locate((2.0, 0.0))
    refused = graph.locate((math.nextafter(2.0, math.inf), 0.0))
    assert accepted.cell_id == refused.cell_id == 0
    assert accepted.distance_m <= D.LOCATE_MAX_DISTANCE_M
    assert refused.distance_m > D.LOCATE_MAX_DISTANCE_M
    assert D.classify_final_evidence(
        {
            "locator_matches_bruteforce_nearest_node": True,
            "located_under_frozen_2m_rule": False,
            "geodesic_distance_m": None,
        }
    ) == ("OFF_NAVIGABLE_GRAPH_OUTCOME", "locate")


def test_blocked_transit_fixture_matches_independent_bfs_and_geodesic() -> None:
    import sys

    package_root = ROOT / "lewm_worlds"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from lewm.oracle.go2_branch_oracle_v1_2 import GeodesicField

    graph = _scene_graph([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], [(0, 1), (1, 2)])
    blocked = frozenset({1})
    field = GeodesicField(graph, 2, transit_blocked=blocked)
    assert graph.bfs_distance(0, 2) == 2
    assert graph.bfs_distance(0, 2, transit_blocked=blocked) is None
    assert D._independent_bfs_distance(graph, 0, 2, blocked) is None
    assert math.isinf(field.cell_distance(0))
    assert D.classify_final_evidence(
        {
            "locator_matches_bruteforce_nearest_node": True,
            "located_under_frozen_2m_rule": True,
            "geodesic_distance_m": None,
        }
    ) == ("LOCATABLE_GOAL_UNREACHABLE_OUTCOME", "bfs_distance/geodesic")


def test_audit_digest_validation_rejects_mutation() -> None:
    report = {
        "failure_inventory": [
            {"primary_category": "OFF_NAVIGABLE_GRAPH_OUTCOME"}
            for _ in range(4)
        ]
        + [
            {"primary_category": "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"}
            for _ in range(14)
        ],
        "counts": {
            "by_primary_category": {
                "INSUFFICIENT_TRACE_FOR_LABEL": 0,
                "LOCATABLE_GOAL_UNREACHABLE_OUTCOME": 14,
                "LOCATOR_IMPLEMENTATION_DEFECT": 0,
                "OFF_NAVIGABLE_GRAPH_OUTCOME": 4,
                "OTHER": 0,
            }
        },
    }
    report["audit_digest"] = D.audit_digest(report)
    D.validate_report(report)
    report["counts"]["by_primary_category"]["OFF_NAVIGABLE_GRAPH_OUTCOME"] = 5
    with pytest.raises(RuntimeError, match="audit digest"):
        D.validate_report(report)


def test_first_failure_range_does_not_assume_monotonic_unavailability() -> None:
    evidence = D._first_unavailable(
        [
            {
                "global_tick": None,
                "located_under_frozen_2m_rule": True,
                "geodesic_distance_m": 2.0,
            },
            {
                "global_tick": 4,
                "located_under_frozen_2m_rule": True,
                "geodesic_distance_m": 1.9,
            },
            {
                "global_tick": 9,
                "located_under_frozen_2m_rule": True,
                "geodesic_distance_m": None,
            },
        ]
    )
    assert evidence["last_observed_valid_block_endpoint_tick"] == 4
    assert evidence["possible_first_unavailable_tick_range_inclusive"] == [0, 9]
    assert evidence["exact_first_unavailable_tick"] is None

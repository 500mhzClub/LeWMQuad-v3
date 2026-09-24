from __future__ import annotations

import json

import numpy as np
import pytest

from lewm.planning.online_belief_map import (
    FEATURE_CHANNELS,
    BeliefMapConfig,
    CellState,
    OnlineBeliefMap,
)


def test_conflicting_occupancy_evidence_is_explicit_and_reversible() -> None:
    belief_map = OnlineBeliefMap()
    cell = (2, -1)

    assert belief_map.cell_state(cell) is CellState.UNKNOWN

    belief_map.fuse_occupied((cell,), confidence=2.0, tick=1)
    assert belief_map.cell_state(cell) is CellState.CONFIRMED_OCCUPIED

    belief_map.fuse_free((cell,), confidence=2.0, tick=2)
    assert belief_map.cell_state(cell) is CellState.CONFLICTED
    belief = belief_map.cell_belief(cell)
    assert belief.free_evidence == 2.0
    assert belief.occupied_evidence == 2.0
    assert belief.occupancy_log_odds == 0.0

    belief_map.fuse_free((cell,), confidence=2.0, tick=3)
    assert belief_map.cell_state(cell) is CellState.CONFIRMED_FREE
    assert belief_map.occupancy_probability(cell) < 0.5

    belief_map.fuse_occupied((cell,), confidence=4.0, tick=4)
    assert belief_map.cell_state(cell) is CellState.CONFIRMED_OCCUPIED


def test_traversal_clears_stale_perception_and_physical_blocks() -> None:
    belief_map = OnlineBeliefMap()
    cell = (1, 1)

    belief_map.fuse_occupied((cell,), confidence=4.0, tick=1)
    belief_map.record_physical_blocks((cell,), confidence=2.0, tick=2)
    assert belief_map.cell_state(cell) is CellState.CONFIRMED_OCCUPIED

    belief_map.record_traversal((cell,), tick=3)
    belief = belief_map.cell_belief(cell)
    assert belief_map.cell_state(cell) is CellState.CONFIRMED_FREE
    assert belief.physical_block_evidence == 0.0
    assert belief.occupied_evidence == 0.0
    assert belief.last_visited_tick == 3
    assert belief.visit_count == 1
    assert belief_map.visit_age_ticks(cell) == 0
    assert belief_map.visit_age_ticks(cell, at_tick=8) == 5

    belief_map.record_physical_blocks((cell,), tick=4)
    assert belief_map.cell_state(cell) is CellState.CONFIRMED_OCCUPIED

    belief_map.record_traversal((cell,), tick=5)
    assert belief_map.cell_state(cell) is CellState.CONFIRMED_FREE
    assert belief_map.cell_belief(cell).visit_count == 2


def test_ray_rasterization_is_four_connected_and_stops_at_endpoint() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))

    ray = belief_map.fuse_ray(
        (0.1, 0.1),
        (3.1, 3.1),
        endpoint_occupied=True,
        tick=1,
    )

    assert ray[0] == (0, 0)
    assert ray[-1] == (3, 3)
    assert all(
        abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1
        for left, right in zip(ray, ray[1:])
    )
    assert all(belief_map.is_confirmed_free(cell) for cell in ray[:-1])
    assert belief_map.is_confirmed_occupied(ray[-1])
    assert belief_map.shortest_path(ray[0], ray[-2]) == ray[:-1]
    assert belief_map.cell_state((4, 4)) is CellState.UNKNOWN


def test_connected_frontiers_and_routes_never_cross_unknown_space() -> None:
    belief_map = OnlineBeliefMap()
    corridor = ((0, 0), (1, 0), (2, 0))
    belief_map.record_traversal(corridor, tick=1)
    belief_map.record_traversal(((10, 10),), tick=1)
    belief_map.record_physical_blocks(
        (
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 1),
            (2, -1),
            (2, 1),
        ),
        tick=2,
    )

    assert belief_map.connected_confirmed_free((0, 0)) == frozenset(corridor)
    assert belief_map.frontier_cells((0, 0)) == ((2, 0),)
    assert belief_map.shortest_path((0, 0), (2, 0)) == corridor
    assert belief_map.shortest_path((0, 0), (3, 0)) is None
    assert belief_map.shortest_path((0, 0), (10, 10)) is None


def test_scene_aligned_origin_roundtrips_world_coordinates() -> None:
    belief_map = OnlineBeliefMap(
        BeliefMapConfig(cell_size_m=0.1, origin_xy_m=(-4.35, -3.25))
    )

    assert belief_map.cell_center((0, 0)) == pytest.approx((-4.30, -3.20))
    assert belief_map.world_to_cell((-4.30, -3.20)) == (0, 0)
    restored = OnlineBeliefMap.from_state_dict(
        json.loads(json.dumps(belief_map.state_dict()))
    )
    assert restored.config.origin_xy_m == (-4.35, -3.25)


def test_eight_connected_routes_reject_diagonal_corner_cutting() -> None:
    belief_map = OnlineBeliefMap(
        BeliefMapConfig(
            planning_connectivity=8,
            allow_diagonal_corner_cutting=False,
        )
    )
    belief_map.fuse_free(((0, 0), (1, 1)), confidence=2.0)

    assert belief_map.shortest_path((0, 0), (1, 1)) is None

    belief_map.fuse_free(((1, 0), (0, 1)), confidence=2.0)
    path = belief_map.shortest_path((0, 0), (1, 1))
    assert path == ((0, 0), (1, 1))
    assert belief_map.connected_confirmed_free((0, 0)) == frozenset(
        {(0, 0), (1, 0), (0, 1), (1, 1)}
    )


def test_target_observations_fuse_as_typed_gaussian_beliefs() -> None:
    belief_map = OnlineBeliefMap(
        BeliefMapConfig(target_covariance_floor=1e-6)
    )
    first = belief_map.fuse_target_observation(
        "blue",
        np.asarray([0.0, 0.0]),
        np.eye(2),
        confidence=0.8,
        tick=1,
    )
    second = belief_map.fuse_target_observation(
        "blue",
        np.asarray([2.0, 0.0]),
        np.eye(2),
        confidence=0.8,
        tick=2,
    )

    assert first.observation_count == 1
    assert second.observation_count == 2
    assert second.mean_xy == pytest.approx((1.0, 0.0))
    assert second.covariance[0][0] < first.covariance[0][0]
    assert second.confidence == pytest.approx(0.96)

    belief_map.mark_target_claimed("blue")
    assert belief_map.targets["blue"].claimed
    with pytest.raises(KeyError, match="unknown target"):
        belief_map.mark_target_claimed("red")


def test_state_dict_roundtrip_preserves_all_typed_state() -> None:
    config = BeliefMapConfig(cell_size_m=0.4, visit_age_horizon_ticks=100)
    belief_map = OnlineBeliefMap(config)
    belief_map.record_traversal(((0, 0), (1, 0)), tick=4)
    belief_map.set_pose(
        np.asarray([0.2, 0.3, 0.1]),
        np.diag([0.04, 0.09, 0.01]),
        tick=4,
        frame="onboard_odom",
    )
    belief_map.fuse_target_observation(
        "green",
        (1.0, 2.0),
        ((0.2, 0.01), (0.01, 0.3)),
        confidence=0.7,
        tick=4,
    )
    belief_map.mark_target_claimed("green")
    belief_map.record_physical_blocks(((2, 0),), confidence=1.5, tick=5)

    state = belief_map.state_dict()
    json.dumps(state)
    restored = OnlineBeliefMap.from_state_dict(state)

    assert restored.state_dict() == state
    assert restored.config == config
    assert restored.pose == belief_map.pose
    assert restored.targets == belief_map.targets

    loaded = OnlineBeliefMap()
    loaded.load_state_dict(state)
    assert loaded.state_dict() == state

    restored.record_traversal(((2, 0),), tick=6)
    assert restored.state_dict() != belief_map.state_dict()
    with pytest.raises(ValueError, match="schema"):
        OnlineBeliefMap.from_state_dict({**state, "schema": "wrong"})


def test_feature_export_has_stable_bounded_semantics() -> None:
    belief_map = OnlineBeliefMap(
        BeliefMapConfig(
            cell_size_m=1.0,
            visit_age_horizon_ticks=100,
            pose_uncertainty_scale_m=1.0,
        )
    )
    belief_map.record_traversal(((0, 0), (1, 0)), tick=10)
    belief_map.fuse_occupied(((-1, 0),), confidence=2.0, tick=10)
    belief_map.fuse_occupied(((0, 1),), confidence=2.0, tick=10)
    belief_map.fuse_free(((0, 1),), confidence=2.0, tick=10)
    belief_map.set_pose(
        (0.5, 0.5, 0.0),
        ((0.09, 0.0, 0.0), (0.0, 0.16, 0.0), (0.0, 0.0, 0.01)),
        tick=10,
    )
    belief_map.fuse_target_observation(
        "yellow",
        belief_map.cell_center((1, 0)),
        ((0.01, 0.0), (0.0, 0.01)),
        tick=10,
    )

    features = belief_map.export_features(
        (0, 0), size=5, at_tick=20, target_id="yellow"
    )
    center = features.row_col((0, 0))
    occupied = features.row_col((-1, 0))
    conflicted = features.row_col((0, 1))
    target = features.row_col((1, 0))
    unknown = features.row_col((2, 2))
    assert None not in (center, occupied, conflicted, target, unknown)

    assert features.values.shape == (len(FEATURE_CHANNELS), 5, 5)
    assert features.channel_names == FEATURE_CHANNELS
    assert float(features.values.min()) >= 0.0
    assert float(features.values.max()) <= 1.0
    assert features.channel("confirmed_free")[center] == 1.0
    assert features.channel("visited")[center] == 1.0
    assert features.channel("visit_age")[center] == pytest.approx(0.1)
    assert features.channel("confirmed_occupied")[occupied] == 1.0
    assert features.channel("conflicted")[conflicted] == 1.0
    assert features.channel("unknown")[unknown] == 1.0
    assert features.channel("target_belief")[target] > 0.99
    assert np.allclose(features.channel("pose_position_uncertainty"), 0.5)
    assert features.channel("frontier").sum() > 0.0


def test_updates_are_monotonic_and_unknown_cannot_be_exported_as_free() -> None:
    belief_map = OnlineBeliefMap()
    belief_map.fuse_free(((0, 0),), tick=3)
    with pytest.raises(ValueError, match="backward"):
        belief_map.fuse_occupied(((0, 0),), tick=2)

    features = belief_map.export_features((0, 0), size=3)
    unknown = features.row_col((1, 1))
    assert unknown is not None
    assert features.channel("unknown")[unknown] == 1.0
    assert features.channel("confirmed_free")[unknown] == 0.0
    assert belief_map.shortest_path((0, 0), (1, 1)) is None


def test_configuration_rejects_non_finite_or_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        BeliefMapConfig(log_odds_cap=float("nan"))
    with pytest.raises(ValueError, match="visit_age_horizon_ticks"):
        BeliefMapConfig(visit_age_horizon_ticks=True)
    with pytest.raises(ValueError, match="finite"):
        OnlineBeliefMap().world_to_cell((float("inf"), 0.0))

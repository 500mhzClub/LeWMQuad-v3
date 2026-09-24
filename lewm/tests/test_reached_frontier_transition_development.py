from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.continuous_connector_waypoint_development import propose as original_propose
from lewm.reached_frontier_waypoint_development import propose
from lewm.reached_frontier_state_development import ReachedFrontierState
from lewm.observed_floor_waypoint_development import centre
from lewm.reached_frontier_recent_qualified_controller_development import (
    ReachedFrontierRecentQualifiedController as New, ReachedFrontierMap)
from lewm.recent_qualified_direct_flow_controller_development import RecentQualifiedDirectFlowController as Old

NOW = 1_800_000_000
FLOOR = {(x, y) for x in range(5) for y in (-1, 0, 1)}
GOAL = [.7, .025]


def advance(state, position, *, floor=FLOOR, occupied=(), goal=GOAL, now=NOW):
    base = original_propose(floor, occupied, position, goal)
    return state.waypoint(floor, occupied, position, goal, base, mission_goal=goal, now_ns=now)


@pytest.mark.parametrize('floor,position,goal,occupied', [
    (FLOOR, centre((0, 0)), GOAL, ()),
    (FLOOR, centre((4, 0)), GOAL, ()),
    (FLOOR, centre((0, 0)), centre((4, 0)), ()),
    (FLOOR, centre((0, 0)), GOAL, {(0, 0)}),
    (set(), [0., 0.], GOAL, ()),
])
def test_empty_retirement_preserves_original_route_and_all_geometry(floor, position, goal, occupied):
    assert propose(floor, occupied, position, goal) == original_propose(floor, occupied, position, goal)


def test_before_same_cell_arrival_every_original_proposal_field_is_preserved():
    state = ReachedFrontierState()
    for index, cell in enumerate(((0, 0), (1, 0), (2, 0), (3, 0))):
        position = centre(cell); expected = original_propose(FLOOR, (), position, GOAL)
        assert advance(state, position, now=NOW+index*100_000_000) == expected
        assert state.retired == {} and state.last_receipt['reached_frontier_cell'] is None


def test_reached_frontier_changes_target_but_remains_usable_in_route():
    state = ReachedFrontierState(); floor = set(FLOOR)
    position = centre((4, 0)); old = original_propose(floor, (), position, GOAL)
    assert old['route_cells'] == [[4, 0]]
    new = advance(state, position, floor=floor)
    assert new['route_cells'][0] == [4, 0] and new['route_cells'][-1] != [4, 0]
    assert state.last_receipt['reached_frontier_cell'] == [4, 0] and (4, 0) in state.retired
    assert floor == FLOOR
    for key in ('start_clearance', 'connector_clearance', 'observed_floor_cells', 'occupied_cells',
            'nominal_radius_m', 'nominal_route_cells', 'footprint_coverage_established'):
        assert new[key] == old[key]


def test_retired_frontier_remains_excluded_without_new_local_evidence():
    state = ReachedFrontierState(); advance(state, centre((4, 0)))
    next_proposal = advance(state, centre((0, 0)), floor=FLOOR | {(30, 30)}, now=NOW+100_000_000)
    assert (4, 0) in state.retired and next_proposal['route_cells'][-1] != [4, 0]
    assert state.last_receipt['released_frontier_cells'] == []


def test_local_observed_classification_change_allows_reconsideration():
    state = ReachedFrontierState(); advance(state, centre((4, 0)))
    result = advance(state, centre((0, 0)), floor=FLOOR | {(5, 0)}, now=NOW+100_000_000)
    assert state.last_receipt['released_frontier_cells'] == [[4, 0]]
    assert (4, 0) not in state.retired and result['route_cells'][-1] == [5, 0]


def test_new_mission_goal_starts_new_frontier_search_without_mutating_map():
    state = ReachedFrontierState(); advance(state, centre((4, 0)))
    advance(state, centre((4, 0)), goal=[-.5, .025], now=NOW+100_000_000)
    assert state.last_receipt['mission_goal_changed'] and (4, 0) not in state.retired


def test_mission_goal_cell_is_never_removed_by_frontier_retirement():
    retired = set(FLOOR)
    goal = centre((4, 0)); position = centre((0, 0))
    assert propose(FLOOR, (), position, goal, retired_frontiers=retired) == original_propose(FLOOR, (), position, goal)
    state = ReachedFrontierState(); result = advance(state, goal, goal=goal)
    assert result['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL' and not state.retired


def test_all_visited_frontiers_request_view_without_turning_cells_into_obstacles():
    state = ReachedFrontierState(); floor = {(0, 0)}
    result = advance(state, centre((0, 0)), floor=floor)
    assert result['status'] == 'OBSERVED_COMPONENT_HAS_NO_UNVISITED_FRONTIER'
    assert result['route_cells'] == [] and result['occupied_cells'] == 0 and result['observed_floor_cells'] == 1
    assert state.last_receipt['retirement_is_obstacle_evidence'] is False
    assert state.last_receipt['retired_cells_remain_traversable'] is True


def test_retirement_does_not_override_occupied_start_constraint():
    result = propose(FLOOR, {(4, 0)}, centre((4, 0)), GOAL, retired_frontiers={(4, 0)})
    assert result['status'] == 'ADDITIONAL_VIEW_REQUIRED'
    assert not result['start_clearance']['nominal_disk_connector_clear']


def test_repeated_same_frame_is_idempotent_and_defensive():
    state = ReachedFrontierState(); position = centre((4, 0))
    first = advance(state, position); receipt = deepcopy(state.last_receipt)
    first['route_cells'].clear()
    second = advance(state, position)
    assert second['route_cells'] and state.last_receipt == receipt
    with pytest.raises(ValueError, match='same observation'):
        advance(state, centre((3, 0)))
    with pytest.raises(ValueError, match='clock'):
        advance(state, position, now=NOW-100_000_000)


def test_existing_observation_registration_contact_and_mission_pipeline_is_retained(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_measured_floor_transport_development import item
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    geometry = ArticulatedCollisionGeometry(URDF)
    options = dict(public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
        require_return_after_goal=True), navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    old, new = Old(None, geometry, **options), New(None, geometry, **options)
    assert New.observe is Old.observe and New.advance is Old.advance
    assert type(new.selector) is type(old.selector) and type(new.motion) is type(old.motion)
    assert type(new.mapper) is ReachedFrontierMap and new.memory is new.mapper.surface
    previous = None
    for frame in range(3):
        p, d, auxiliary, raw, now, image = item(frame, previous, narrow=frame == 1)
        results = []
        for controller in (old, new):
            controller.motion = SimpleNamespace(observe=lambda *args, **kwargs: raw)
            results.append(controller.observe(p, d, None, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image))
        baseline, candidate = results
        candidate.pop('reached_frontier_transition_enabled'); candidate.pop('last_frontier_transition_receipt')
        candidate['controller'] = baseline['controller']
        assert candidate == baseline and baseline['terminal'] is None
        assert old.mapper.floor == new.mapper.floor and old.mapper.occupied == new.mapper.occupied
        assert old.residual.snapshot() == new.residual.snapshot()
        previous = raw

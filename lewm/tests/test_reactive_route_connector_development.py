from copy import deepcopy
import math
import numpy as np
import pytest
from lewm.reactive_nominal_route_selection_development import ReactiveNominalRouteSelector
from lewm.reactive_connector_round_trip_controller_development import ReactiveConnectorRouteSelector, ReactiveConnectorRoundTripController
from lewm.reactive_nominal_round_trip_controller_development import ReactiveNominalRoundTripController
from lewm.reactive_route_connector_development import nearer_route_target
from lewm.tests.test_reactive_observed_route_development import mapper_fixture


def corner_mapper(*, conflict=False):
    heading = math.atan2(.225, .3)
    m, calls = mapper_fixture(heading=heading, conflict=conflict)
    m.surface.position = np.array([-.375, .3, 0.])
    route = [[-8, 6], [-8, 7], [-8, 8], [-7, 8], [-7, 9], [-6, 9], [-5, 9], [-4, 9], [-4, 10], [-3, 10], [-2, 10]]
    m.floor = {(x, y) for x in range(-15, 5) for y in range(1, 16)}
    m.occupied = {(0, 0)}
    m.waypoint = lambda goal, **kwargs: dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', route_cells=deepcopy(route))
    return m, calls


def test_clear_nearer_route_point_avoids_blocked_corner_chord_without_relaxing_radius():
    m, calls = corner_mapper(); floor = deepcopy(m.floor)
    old = ReactiveNominalRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    new = ReactiveConnectorRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    assert old['action'] is None and not old['measured_waypoint_connector']['nominal_disk_connector_clear']
    assert old['current_nominal_clearance']['nominal_disk_connector_clear']
    assert new['action'] == 'left_turn' and new['nearer_observed_route_target']
    assert new['selected_route_target_index'] < new['original_route_target_index']
    assert new['measured_waypoint_connector']['nominal_disk_connector_clear']
    assert new['measured_waypoint_connector']['radius_m'] == .45
    assert new['original_waypoint_selection'] == old and new['proposal'] == old['proposal'] and m.floor == floor
    assert all(c[:2] == ([0., 0.], 0.) for c in calls)
    assert not new['learned_model_used'] and not new['candidate_future_outcomes_evaluated']
    assert 'prediction' not in new and 'nominal_path_checks' not in new


@pytest.mark.parametrize('cause', ['current_surface', 'current_clearance', 'existing_turn', 'exact_goal', 'view_exhausted'])
def test_original_current_geometry_valid_action_and_mission_guards_are_preserved(cause):
    m, _ = corner_mapper(conflict=cause == 'current_surface')
    old = ReactiveNominalRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    if cause == 'current_clearance': old['current_nominal_clearance']['nominal_disk_connector_clear'] = False
    if cause == 'existing_turn': old['action'] = 'right_turn'
    if cause == 'exact_goal': old['intermediate_target_is_mission_goal'] = True
    if cause == 'view_exhausted': old['view_budget_exhausted'] = True
    assert nearer_route_target(old, m.surface.position, m.surface.rotation, m.floor, m.occupied) is old


def test_no_clear_earlier_point_preserves_infeasibility_and_bad_geometry_receipts_fail():
    m, _ = corner_mapper()
    old = ReactiveNominalRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    single = deepcopy(old); single['proposal']['route_cells'] = [old['proposal']['route_cells'][old['proposal']['route_cells'].index([-2, 10])]]
    assert nearer_route_target(single, m.surface.position, m.surface.rotation, m.floor, m.occupied) is single
    bad = deepcopy(old); bad['measured_waypoint_connector']['minimum_observed_cell_distance_m'] += .1
    with pytest.raises(ValueError): nearer_route_target(bad, m.surface.position, m.surface.rotation, m.floor, m.occupied)
    with pytest.raises(ValueError): nearer_route_target(old, m.surface.position, m.surface.rotation, m.floor, {(1, 0)})
    with pytest.raises(ValueError): nearer_route_target(old | {'native_state_used': True}, m.surface.position, m.surface.rotation, m.floor, m.occupied)


def test_unblocked_route_and_controller_state_contracts_are_unchanged():
    m, _ = mapper_fixture()
    a = ReactiveNominalRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    b = ReactiveConnectorRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(m, object(), now_ns=1)
    assert a == b
    assert ReactiveConnectorRoundTripController.observe is ReactiveNominalRoundTripController.observe
    assert ReactiveConnectorRoundTripController.advance is ReactiveNominalRoundTripController.advance
    c = ReactiveConnectorRoundTripController(object(), public_mission=dict(goal_initial_body_xy_m=[.2, 0.],
        return_initial_body_xy_m=[0., 0.], require_return_after_goal=True), navigation_ticks=40)
    assert not hasattr(c, 'model') and not hasattr(c, 'residual')
    r = c.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert c.advance({}, {}, now_ns=2)['terminal'] == r['terminal']

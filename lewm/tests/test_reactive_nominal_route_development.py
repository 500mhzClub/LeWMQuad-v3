import ast
from pathlib import Path
import pytest
from lewm.reactive_nominal_route_selection_development import ReactiveNominalRouteSelector
from lewm.reactive_nominal_round_trip_controller_development import ReactiveNominalRoundTripController
from lewm.reactive_observed_round_trip_controller_development import ReactiveObservedRoundTripController
from lewm.tests.test_reactive_observed_route_development import mapper_fixture


def test_unknown_connector_is_explicitly_permitted_without_inventing_free_space_or_predictions():
    mapper, calls = mapper_fixture(); mapper.floor.remove((4, 0))
    r = ReactiveNominalRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(mapper, object(), now_ns=1)
    assert r['action'] == 'forward' and r['unknown_connector_motion_permitted']
    assert r['unknown_waypoint_connector_cells'] == [[4, 0]] and (4, 0) not in mapper.floor
    assert not r['unobserved_space_certified'] and not r['candidate_future_outcomes_evaluated']
    assert calls[0][:2] == ([0., 0.], 0.)


@pytest.mark.parametrize('cause', ['occupied_current', 'occupied_connector', 'current_surface'])
def test_original_known_geometry_vetoes_remain(cause):
    mapper, _ = mapper_fixture(conflict=cause == 'current_surface')
    if cause == 'occupied_current': mapper.occupied.add((8, 0))
    if cause == 'occupied_connector': mapper.occupied.add((12, 0))
    r = ReactiveNominalRouteSelector(goal_initial_body_xy_m=[1., 0.]).choose(mapper, object(), now_ns=1)
    assert r['action'] is None and r['requested_command'] == [0., 0., 0.]


def test_source_scope_only_changes_connector_observation_gate_and_explicit_receipts():
    root = Path(__file__).resolve().parents[2]
    old = (root/'lewm/reactive_observed_route_selection_development.py').read_text()
    expected = old.replace('Non-predictive action rule over the same persistent observed floor route.',
        'Non-predictive nominal-route rule with explicit unknown start connectors.')
    expected = expected.replace('ReactiveObservedRouteSelector', 'ReactiveNominalRouteSelector')
    expected = expected.replace("elif connector['nominal_disk_connector_clear'] and not unknown:",
        "elif connector['nominal_disk_connector_clear']:")
    expected = expected.replace('current_geometry_checked=True, learned_model_used=False',
        "unknown_connector_motion_permitted=bool(action == 'forward' and unknown),\n            unobserved_space_certified=False, current_geometry_checked=True, learned_model_used=False")
    expected = expected.replace('turn_to_observed_waypoint_then_forward_on_measured_clear_connector',
        'turn_to_observed_waypoint_then_forward_on_nominal_connector_unknown_recorded')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((root/'lewm/reactive_nominal_route_selection_development.py').read_text()))
    assert ReactiveNominalRoundTripController.observe is ReactiveObservedRoundTripController.observe
    assert ReactiveNominalRoundTripController.advance is ReactiveObservedRoundTripController.advance

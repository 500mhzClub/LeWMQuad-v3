from copy import deepcopy
import numpy as np
import pytest
from lewm.nominal_clearance_reentry_development import reenter
from lewm.nominal_reentry_round_trip_controller_development import NominalReentryRoundTripController
from lewm.observed_round_trip_controller_development import ObservedRoundTripController
from lewm.tests.test_executed_horizon_final_goal_development import final_selection


def fixture(*, cells=((8, 0),), late_worsening=False, still=False):
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.; p[:, :, 4] = -10.
    p[:, :, 0] = .005*np.arange(1, 9)  # Other forecasts approach the obstacle.
    p[0, :, 0] = -.01*np.arange(1, 9)  # A predicted drifting hold cannot reset the wait.
    p[4, :, 0] = 0. if still else -.005*np.arange(1, 9)
    if late_worsening: p[4, 5, 0] = .001
    s = final_selection(p, cells); s['native_state_used'] = False
    s['phase_allowed_actions'] = ['hold', 'left_turn', 'right_turn']
    return s


def test_nonzero_reentry_retains_original_veto_and_all_raw_evidence():
    s = fixture(); original = deepcopy(s)
    assert s['action'] is None
    r = reenter(s, np.zeros(3), np.eye(3), [(8, 0)])
    assert r['action'] == 'left_turn' and r['requested_command'] == [0., 0., .45]
    assert s == original and r['recovery_is_nominal_policy_exception']
    for k in ('prediction', 'candidates', 'surface_checks', 'nominal_action_checks', 'nominal_path_checks', 'phase_allowed_actions'):
        assert r[k] == s[k]
    assert r['reentry_current_clearance']['minimum_observed_cell_distance_m'] == .4
    assert not r['nominal_path_checks'][4]['all_predicted_segments_nominally_clear']
    assert r['reentry_candidates'][4]['first_endpoint_clearance_gain_m'] == pytest.approx(.005)
    assert not r['reentry_candidates'][0]['eligible'] and not r['reentry_guaranteed']


@pytest.mark.parametrize('cause', ['later_worsening', 'surface', 'phase', 'zero_progress'])
def test_reentry_does_not_bypass_other_vetoes_or_accept_later_worsening(cause):
    s = fixture(late_worsening=cause == 'later_worsening', still=cause == 'zero_progress')
    if cause == 'surface': s['surface_checks'][4]['possible_intersection'] = True
    if cause == 'phase': s['phase_allowed_actions'] = ['hold', 'right_turn']
    assert reenter(s, np.zeros(3), np.eye(3), [(8, 0)]) is s


def test_only_an_already_violated_nonzero_current_clearance_can_activate():
    for cells in ([(10, 0)], [(0, 0)], []):
        s = fixture(cells=cells); s['action'] = None
        assert reenter(s, np.zeros(3), np.eye(3), cells) is s
    s = fixture(); s['action'] = 'right_turn'
    assert reenter(s, np.zeros(3), np.eye(3), [(8, 0)]) is s
    s['action'] = None; s['view_budget_exhausted'] = True
    assert reenter(s, np.zeros(3), np.eye(3), [(8, 0)]) is s


def test_changed_map_or_forged_forecast_receipt_rejected():
    s = fixture()
    with pytest.raises(ValueError): reenter(s, np.zeros(3), np.eye(3), [(7, 0)])
    s['nominal_path_checks'][4]['segments'][7]['minimum_observed_cell_distance_m'] = 2.
    with pytest.raises(ValueError): reenter(s, np.zeros(3), np.eye(3), [(8, 0)])
    s = fixture(); s['native_state_used'] = True
    with pytest.raises(ValueError): reenter(s, np.zeros(3), np.eye(3), [(8, 0)])


def test_observer_mission_budget_and_latched_stop_remain_inherited():
    assert NominalReentryRoundTripController.observe is ObservedRoundTripController.observe
    assert NominalReentryRoundTripController.advance is ObservedRoundTripController.advance
    c = NominalReentryRoundTripController(object(), object(), public_mission=dict(
        goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    r = c.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
    before = c.residual.snapshot()
    assert c.advance({}, {}, now_ns=2)['terminal'] == r['terminal']
    assert c.residual.snapshot() == before

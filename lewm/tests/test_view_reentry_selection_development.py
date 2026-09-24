from copy import deepcopy
import numpy as np
import pytest
from lewm.nominal_clearance_reentry_development import reenter
from lewm.view_reentry_selection_development import reenter_with_translation
from lewm.tests.test_executed_horizon_final_goal_development import final_selection


def fixture(*, cells=((8, 0),), later_worsening=False, gain=.01, turn_gain=None, forward_contact=-10.):
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.; p[:, :, 4] = -10.
    p[:, :, 0] = .005*np.arange(1, 9)
    p[0, :, 0] = -.1*np.arange(1, 9)  # A drifting hold cannot count as recovery.
    p[1, :, 0] = -gain*np.arange(1, 9)
    p[1, -1, 4] = forward_contact
    if turn_gain is not None: p[4, :, 0] = -turn_gain*np.arange(1, 9)
    if later_worsening: p[1, 7, 0] = .001
    s = final_selection(p, cells)
    s.update(native_state_used=False, mode='VIEW_ACQUISITION',
        phase_allowed_actions=['hold', 'left_turn', 'right_turn'])
    return s


def test_translation_recovers_when_turns_worsen_and_preserves_original_receipts():
    s = fixture(); original = deepcopy(s)
    assert reenter(s, np.zeros(3), np.eye(3), [(8, 0)]) is s
    r = reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)])
    assert r['action'] == 'forward' and r['requested_command'] == [.2, 0., 0.]
    assert r['selected_action_requires_view_phase_exception'] and not r['reentry_guaranteed']
    assert r['phase_admissible_candidates'] == 0 and r['reentry_admissible_candidates'] == 1
    assert s == original
    for k in ('prediction', 'candidates', 'surface_checks', 'nominal_action_checks', 'nominal_path_checks', 'phase_allowed_actions'):
        assert r[k] == original[k]
    c = r['reentry_candidates'][1]
    assert c['eligible'] and not c['phase_allowed'] and c['reentry_phase_allowed']
    assert c['first_endpoint_clearance_gain_m'] == pytest.approx(.01)
    assert not r['reentry_candidates'][0]['eligible']


@pytest.mark.parametrize('cause', ['later_worsening', 'surface', 'zero_gain', 'view_exhausted'])
def test_translation_does_not_bypass_other_recovery_requirements(cause):
    s = fixture(later_worsening=cause == 'later_worsening', gain=0. if cause == 'zero_gain' else .01)
    if cause == 'surface': s['surface_checks'][1]['possible_intersection'] = True
    if cause == 'view_exhausted': s['view_budget_exhausted'] = True
    assert reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)]) is s


def test_clear_radius_zero_clearance_and_valid_original_action_do_not_activate():
    for cells in ([(10, 0)], [(0, 0)], []):
        s = fixture(cells=cells); s['action'] = None
        assert reenter_with_translation(s, np.zeros(3), np.eye(3), cells) is s
    s = fixture(); s['action'] = 'left_turn'
    assert reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)]) is s


def test_original_waypoint_phase_is_unchanged_and_forged_receipts_fail():
    s = fixture(); s['mode'] = 'WAYPOINT'
    assert reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)]) == reenter(s, np.zeros(3), np.eye(3), [(8, 0)])
    for patch in ({'phase_allowed_actions': ['hold']}, {'native_state_used': True}):
        with pytest.raises(ValueError): reenter_with_translation(fixture() | patch, np.zeros(3), np.eye(3), [(8, 0)])
    with pytest.raises(ValueError): reenter_with_translation(fixture(), np.zeros(3), np.eye(3), [(7, 0)])
    s = fixture(); s['nominal_path_checks'][1]['segments'][7]['minimum_observed_cell_distance_m'] = 2.
    with pytest.raises(ValueError): reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)])


def test_controller_retains_observation_mission_budget_and_latched_failures():
    from lewm.view_reentry_round_trip_controller_development import ViewReentryRoundTripController
    from lewm.executed_waypoint_round_trip_controller_development import ExecutedWaypointRoundTripController
    assert ViewReentryRoundTripController.observe is ExecutedWaypointRoundTripController.observe
    assert ViewReentryRoundTripController.advance is ExecutedWaypointRoundTripController.advance
    c = ViewReentryRoundTripController(object(), object(), public_mission=dict(
        goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    r = c.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
    before = c.residual.snapshot()
    assert c.advance({}, {}, now_ns=2)['terminal'] == r['terminal']
    assert c.residual.snapshot() == before


def test_recovery_compares_translation_with_an_eligible_turn_and_keeps_full_contact_cost():
    s = fixture(turn_gain=.005)
    assert reenter(s, np.zeros(3), np.eye(3), [(8, 0)])['action'] == 'left_turn'
    assert reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)])['action'] == 'forward'
    s = fixture(turn_gain=.005, forward_contact=10.)
    r = reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)])
    assert r['action'] == 'left_turn' and not r['selected_action_requires_view_phase_exception']
    assert r['reentry_candidates'][1]['eligible'] and r['reentry_candidates'][1]['full_plan_contact_score'] > .99


@pytest.mark.parametrize('turn_gain', [None, .005])
def test_prefix_comparison_reconstructs_both_policies_and_rejects_forgery(turn_gain):
    from scripts.view_reentry_prefix_selection_development import compare_selection
    s = fixture(turn_gain=turn_gain)
    old = reenter(s, np.zeros(3), np.eye(3), [(8, 0)])
    new = reenter_with_translation(s, np.zeros(3), np.eye(3), [(8, 0)])
    compare_selection(old, new, np.zeros(3), np.eye(3), [(8, 0)])
    for target in ('old', 'new'):
        a, b = deepcopy(old), deepcopy(new)
        (a if target == 'old' else b)['nominal_path_checks'][1]['segments'][7]['minimum_observed_cell_distance_m'] = 2.
        with pytest.raises(ValueError): compare_selection(a, b, np.zeros(3), np.eye(3), [(8, 0)])
    forged = deepcopy(new); forged['action'] = 'hold'
    with pytest.raises(ValueError): compare_selection(old, forged, np.zeros(3), np.eye(3), [(8, 0)])

from copy import deepcopy
import numpy as np
import pytest
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.tests.test_eight_step_planning_development import checked
from lewm.eight_step_planning_development import plan
from lewm.online_executed_residual_development import OnlineExecutedResidual


def fixture():
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.; p[:, :, 4] = -10.
    p[1, :, 0] = np.linspace(.01, .16, 8)
    s = checked(p, ())
    s.update(intermediate_target_is_mission_goal=False, goal_body_xy_m=[.06, 0.], native_state_used=False)
    s = plan(s, np.zeros(3), np.eye(3), ())
    return s, OnlineExecutedResidual().snapshot()


def test_short_execution_can_approach_near_waypoint_despite_unexecuted_overshoot():
    s, receipt = fixture(); original = deepcopy(s)
    r = score_waypoint_execution(s, receipt)
    assert s['action'] == 'hold' and r['action'] == 'forward'
    assert r['candidates'][1]['executed_waypoint_distance_progress_m'] == pytest.approx(.01)
    assert r['scored_pose_horizon_ns'] == 100_000_000 and r['scored_contact_horizon_ns'] == 800_000_000
    for key in ('prediction', 'nominal_path_checks', 'nominal_action_checks', 'surface_checks', 'phase_allowed_actions'):
        assert r[key] == original[key]
    assert s == original and r['original_waypoint_candidates'] == s['candidates']
    assert not r['corrected_scoring_path_checked']


@pytest.mark.parametrize('veto', ['later_path', 'surface', 'phase', 'full_contact'])
def test_original_veto_and_full_plan_contact_still_prevent_forward(veto):
    s, receipt = fixture()
    if veto == 'later_path':
        s['nominal_path_checks'][1]['all_predicted_segments_nominally_clear'] = False
    elif veto == 'surface': s['surface_checks'][1]['possible_intersection'] = True
    elif veto == 'phase': s['phase_allowed_actions'] = ['hold', 'left_turn', 'right_turn']
    else: s['prediction'][1][-1][4] = 10.
    assert score_waypoint_execution(s, receipt)['action'] != 'forward'


def test_causal_residual_changes_only_score_and_rejects_future_or_privileged_inputs():
    s, receipt = fixture()
    receipt.update(frame=5, residual_source_ticks=[3], residual_available_ticks=[4], correction_xy_m=[.004, -.002])
    r = score_waypoint_execution(s, receipt)
    np.testing.assert_allclose(r['candidates'][1]['causal_scoring_body_xy_m'], [.006, .002])
    assert r['prediction'] == s['prediction']
    for patch in ({'residual_source_ticks': [5]}, {'residual_available_ticks': [6]}, {'native_outcomes_used': True},
                  {'command_integrated_pose_used': True}):
        with pytest.raises(ValueError): score_waypoint_execution(s, receipt | patch)


def test_final_goal_view_and_reentry_preserve_original_policy_and_no_feasible_remains_none():
    s, receipt = fixture()
    for patch in ({'intermediate_target_is_mission_goal': True}, {'mode': 'VIEW_ACQUISITION'}, {'nominal_clearance_reentry': True}):
        q = s | patch
        assert score_waypoint_execution(q, receipt) is q
    for r in s['nominal_path_checks']: r['all_predicted_segments_nominally_clear'] = False
    r = score_waypoint_execution(s, receipt)
    assert r['action'] is None and r['requested_command'] == [0., 0., 0.]


def test_mission_observer_and_latched_failures_are_inherited():
    from lewm.executed_waypoint_round_trip_controller_development import ExecutedWaypointRoundTripController
    from lewm.nominal_reentry_round_trip_controller_development import NominalReentryRoundTripController
    assert ExecutedWaypointRoundTripController.observe is NominalReentryRoundTripController.observe
    assert ExecutedWaypointRoundTripController.advance is NominalReentryRoundTripController.advance
    c = ExecutedWaypointRoundTripController(object(), object(), public_mission=dict(
        goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    r = c.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
    before = c.residual.snapshot()
    assert c.advance({}, {}, now_ns=2)['terminal'] == r['terminal']
    assert c.residual.snapshot() == before

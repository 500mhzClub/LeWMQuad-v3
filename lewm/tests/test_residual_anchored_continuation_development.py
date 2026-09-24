"""Real planner geometry distinguishes first-point and later-path corrections."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_residual_hold_feasibility_development import fixture as hold_fixture
from lewm.tests.test_residual_first_interval_feasibility_development import NOW
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.observation_horizon_predictive_selection_development import score_candidates
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.residual_hold_feasibility_development import reconsider_hold_feasibility
from lewm.residual_anchored_continuation_development import reconsider_anchored_continuation


def fixture(*, late_x=.01, expensive=False):
    old, receipt, mapper = hold_fixture(expensive=expensive)
    prediction = np.asarray(old['prediction']); prediction[1:,4,0] = late_x
    selection = score_candidates(prediction, goal_body_xy_m=[1.,0.], contact_penalty_m=1.2)
    selection.update(prediction=prediction.tolist(), first_prediction_horizon_ns=100_000_000,
        target_offsets_ns=list(range(100_000_000,800_000_001,100_000_000)),
        mode='WAYPOINT', phase_allowed_actions=list(ACTIONS), phase_admissible_candidates=6,
        intermediate_target_is_mission_goal=False)
    selection = filter_selection(selection, mapper.surface, object(), now_ns=NOW, persistent=True)
    selection = constrain(selection, mapper.surface.position, np.eye(3), mapper.occupied)
    selection = plan(selection, mapper.surface.position, np.eye(3), mapper.occupied)
    selection = score_waypoint_execution(selection, receipt)
    assert selection['action'] == 'hold'
    return selection, receipt, mapper


def apply(selection, receipt, mapper):
    return reconsider_anchored_continuation(selection, receipt, mapper, object(), now_ns=NOW)


def test_later_path_recovery_uses_original_short_score_and_raw_targets():
    s, r, m = fixture(); before = deepcopy(s)
    assert reconsider_hold_feasibility(s, r, m, object(), now_ns=NOW) is s
    result = apply(s, r, m); check = result['residual_anchored_continuation']
    assert result['action'] == 'forward' and s == before
    assert check['selected_utility_m'] > check['original_hold_utility_m']
    assert any(not x['nominal_disk_connector_clear'] for x in s['nominal_path_checks'][1]['segments'][2:])
    assert check['corrected_nominal_path_checks'][1]['all_predicted_segments_nominally_clear']
    raw = np.asarray(s['prediction'])[:,:,:2]
    corrected = np.asarray(check['corrected_body_xy_m'])
    np.testing.assert_array_equal(corrected[:,0], raw[:,0]-r['correction_xy_m'])
    np.testing.assert_allclose(np.diff(corrected,axis=1), np.diff(raw,axis=1), rtol=0, atol=1e-16)
    for key in ('prediction','candidates','surface_checks','nominal_action_checks','nominal_path_checks'):
        assert result[key] == s[key]
    assert not check['later_predicted_points_unchanged']
    assert not check['later_prediction_correction_calibrated']
    assert not check['physical_clearance_certified'] and not check['model_error_bound_applied']
    from lewm.tests.test_residual_first_interval_feasibility_development import fixture as residual_fixture
    memory = residual_fixture()[3]
    memory.remember(dict(tick=10,terminal=None,new_selection=result,requested_command=result['requested_command']))
    assert memory.pending['predicted_body_xy_m'] == s['prediction'][1][0][:2]
    assert memory.pending['predicted_body_xy_m'] != check['corrected_first_body_xy_m'][1]
    check['corrected_nominal_path_checks'][1]['segments'][0]['radius_m'] = 9.
    assert s == before


def test_first_point_recovery_has_exact_precedence():
    s, r, m = hold_fixture()
    old = reconsider_hold_feasibility(s,r,m,object(),now_ns=NOW)
    assert old['action'] == 'forward'
    assert apply(s,r,m) == old
    assert 'residual_anchored_continuation' not in old


@pytest.mark.parametrize('veto', ['later_segment','contact_cost','original_surface','corrected_surface','phase','tie'])
def test_every_existing_gate_still_applies(veto):
    s,r,m = fixture(late_x=.02 if veto=='later_segment' else .01, expensive=veto=='contact_cost')
    if veto=='original_surface':
        for x in s['surface_checks'][1:]: x['possible_intersection']=True
    elif veto=='corrected_surface': m.surface.block=True
    elif veto=='phase': s['phase_allowed_actions']=['hold']
    elif veto=='tie':
        for x in s['candidates'][1:]: x['utility_m']=s['candidates'][0]['utility_m']
    before=deepcopy(s)
    assert apply(s,r,m) is s and s==before


@pytest.mark.parametrize('patch', [dict(action=None),dict(action='forward'),dict(mode='VIEW_ACQUISITION'),
    dict(intermediate_target_is_mission_goal=True),dict(nominal_clearance_reentry=True),
    dict(view_budget_exhausted=True),dict(residual_first_interval_feasibility={'already_checked':True})])
def test_other_policies_unchanged(patch):
    s,r,m=fixture();s.update(patch)
    assert apply(s,r,m) is s


@pytest.mark.parametrize('fault', ['future','bias','path','score','clock','failed_map'])
def test_invalid_causal_and_geometry_evidence_rejects(fault):
    s,r,m=fixture()
    if fault=='future': r['residuals'][-1]['available_tick']=11
    elif fault=='bias': r['correction_xy_m'][0]=.2
    elif fault=='path': s['nominal_path_checks'][1]['segments'][4]['radius_m']=.4
    elif fault=='score': s['candidates'][1]['causal_scoring_body_xy_m']=[1.,0.]
    elif fault=='clock': m.surface.last_ns-=100_000_000
    else: m.failed=True
    with pytest.raises(ValueError): apply(s,r,m)


def test_empty_causal_history_and_current_collision_do_not_activate():
    s,r,m=fixture();m.surface.position[0]=.01
    assert apply(s,r,m) is s
    s,r,m=fixture()
    from lewm.tests.test_residual_first_interval_feasibility_development import fixture as residual_fixture
    memory=residual_fixture()[3];memory.history.clear()
    assert apply(s,memory.snapshot(),m) is s


def test_observation_state_and_failure_latch_are_inherited():
    from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController as New
    from lewm.residual_hold_feasibility_controller_development import ResidualHoldFeasibilityController as Old
    for name in ('observe','advance'): assert getattr(New,name) is getattr(Old,name)
    c=New(object(),object(),public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True),navigation_ticks=40,condition='jepa',variant='full',persistent=True)
    assert c.selector.residual is c.residual and c.memory is c.mapper.surface
    failed=c.observe({}, {}, {}, now_ns=1)
    assert failed['terminal']=='SENSOR_OR_MODEL_FAILURE' and failed['requested_command']==[0.,0.,0.]
    assert failed['residual_anchored_continuation_enabled'] is True

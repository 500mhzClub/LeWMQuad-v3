"""Original hold is retained unless a strictly better action passes every gate."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_residual_first_interval_feasibility_development import fixture as original_fixture, NOW
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.observation_horizon_predictive_selection_development import score_candidates
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.residual_hold_feasibility_development import reconsider_hold_feasibility


def fixture(*,late_collision=False,expensive=False):
    original,receipt,mapper,_=original_fixture()
    p=np.asarray(original['prediction']); p[0,0,0]=0.
    if late_collision: p[1:,4,0]=.02
    if expensive: p[1:,:,4]=10.
    s=score_candidates(p,goal_body_xy_m=[1.,0.],contact_penalty_m=1.2)
    s.update(prediction=p.tolist(),first_prediction_horizon_ns=100_000_000,
        target_offsets_ns=list(range(100_000_000,800_000_001,100_000_000)),
        mode='WAYPOINT',phase_allowed_actions=list(ACTIONS),phase_admissible_candidates=6,
        intermediate_target_is_mission_goal=False)
    s=filter_selection(s,mapper.surface,object(),now_ns=NOW,persistent=True)
    s=constrain(s,mapper.surface.position,np.eye(3),mapper.occupied)
    s=plan(s,mapper.surface.position,np.eye(3),mapper.occupied)
    s=score_waypoint_execution(s,receipt)
    assert s['action']=='hold' and s['nominal_path_checks'][0]['all_predicted_segments_nominally_clear']
    return s,receipt,mapper


def apply(s,receipt,mapper):
    return reconsider_hold_feasibility(s,receipt,mapper,object(),now_ns=NOW)


def test_only_first_point_feasibility_changes_and_original_utility_wins():
    s,receipt,mapper=fixture(); before=deepcopy(s); result=apply(s,receipt,mapper)
    assert result['action']=='forward' and s==before
    check=result['residual_hold_feasibility']
    assert check['original_action']=='hold' and check['selected_utility_m']>check['original_hold_utility_m']
    assert not s['nominal_path_checks'][1]['all_predicted_segments_nominally_clear']
    assert check['corrected_nominal_path_checks'][1]['all_predicted_segments_nominally_clear']
    for key in ('prediction','candidates','surface_checks','nominal_action_checks','nominal_path_checks'):
        assert result[key]==s[key]
    assert check['raw_predictions_remain_residual_targets'] and not check['physical_clearance_certified']
    assert check['original_surface_vetoes_preserved'] and not check['model_error_bound_applied']


@pytest.mark.parametrize('veto',['later_segment','contact_cost','original_surface','corrected_surface','phase','tie'])
def test_feasible_hold_is_not_forced_to_move(veto):
    s,receipt,mapper=fixture(late_collision=veto=='later_segment',expensive=veto=='contact_cost')
    if veto=='original_surface':
        for c in s['surface_checks'][1:]: c['possible_intersection']=True
    elif veto=='corrected_surface': mapper.surface.block=True
    elif veto=='phase': s['phase_allowed_actions']=['hold']
    elif veto=='tie':
        for c in s['candidates'][1:]: c['utility_m']=s['candidates'][0]['utility_m']
    before=deepcopy(s)
    assert apply(s,receipt,mapper) is s and s==before


@pytest.mark.parametrize('patch',[{'action':None},{'action':'forward'},{'mode':'VIEW_ACQUISITION'},
    {'intermediate_target_is_mission_goal':True},{'nominal_clearance_reentry':True},
    {'residual_first_interval_feasibility':{'already_checked':True}}])
def test_existing_no_action_recovery_and_other_policies_are_untouched(patch):
    s,receipt,mapper=fixture(); s.update(patch); assert apply(s,receipt,mapper) is s


@pytest.mark.parametrize('fault',['future','bias','original_hold_path','raw_path','score_position'])
def test_invalid_causal_or_original_gate_evidence_rejects(fault):
    s,receipt,mapper=fixture()
    if fault=='future': receipt['residuals'][-1]['available_tick']=11
    elif fault=='bias': receipt['correction_xy_m'][0]=.2
    elif fault=='original_hold_path': s['nominal_path_checks'][0]['all_predicted_segments_nominally_clear']=False
    elif fault=='raw_path': s['nominal_path_checks'][1]['segments'][0]['radius_m']=.4
    elif fault=='score_position': s['candidates'][1]['causal_scoring_body_xy_m']=[1.,0.]
    with pytest.raises(ValueError): apply(s,receipt,mapper)


def test_controller_retains_original_observation_and_state_owners():
    from lewm.residual_hold_feasibility_controller_development import ResidualHoldFeasibilityController as New
    from lewm.residual_first_interval_controller_development import ResidualFirstIntervalController as Old
    for name in ('observe','advance'): assert getattr(New,name) is getattr(Old,name)
    c=New(object(),object(),public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True),navigation_ticks=40,condition='jepa',variant='full',persistent=True)
    assert c.selector.residual is c.residual and c.memory is c.mapper.surface

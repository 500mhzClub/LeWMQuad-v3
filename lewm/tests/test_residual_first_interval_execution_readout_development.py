"""Native frame transform, censoring, and causal-veto integrity of fallback scoring."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_residual_first_interval_feasibility_development import fixture, apply
from lewm.residual_first_interval_execution_readout_development import summarize_execution


def data(*,blocked=False,partial=False):
    s,receipt,mapper,_=fixture(late_collision=blocked); s=apply(s,receipt,mapper)
    rows=[dict(tick=i,decision=dict(new_selection=None,terminal=None,failure=None)) for i in range(10)]
    rows.append(dict(tick=10,decision=dict(new_selection=s,terminal=None,failure=None,
        selected_action=s['action'],requested_command=s['requested_command'])))
    poses=np.zeros((1300,7)); poses[:,5:7]=np.sqrt(.5)  # 90-degree world yaw.
    poses[1299,1]=.004
    tape=[{} for _ in range(10)]+[dict(tick=10,pre_sample_index=1249,
        post_sample_index=1270 if partial else 1299,completed=not partial,requested_command=s['requested_command'])]
    return poses,tape,rows


@pytest.mark.parametrize('partial',[False,True])
def test_only_actual_selected_motion_with_correct_frame_and_censoring(partial):
    poses,tape,rows=data(partial=partial); before=deepcopy(rows)
    r=summarize_execution(poses,tape,rows)
    assert rows==before and r['observations']==11 and r['fallback_selected_intervals']==1
    assert r['completed_fallback_intervals']==int(not partial) and r['censored_fallback_intervals']==int(partial)
    entry=r['fallback_execution'][0]
    assert not entry['original_selected_path_nominally_clear']
    if partial:
        assert entry['native_body_xy_m'] is None and entry['corrected_xy_error_m'] is None
        assert r['corrected_forecast_mean_xy_error_m'] is None
    else:
        assert entry['native_body_xy_m']==pytest.approx([.004,0.])
        assert r['raw_forecast_mean_xy_error_m']==pytest.approx(.006)
        assert r['corrected_forecast_mean_xy_error_m']==pytest.approx(.004)
    assert not r['unexecuted_alternative_outcomes_inferred'] and not r['physical_clearance_certified']


def test_failed_fallback_attempt_is_retained_without_invented_execution():
    poses,tape,rows=data(blocked=True)
    rows[-1]['decision'].update(terminal='NO_FEASIBLE_ACTION',failure='synthetic failure')
    r=summarize_execution(poses,tape,rows)
    assert r['fallback_attempts']==[dict(tick=10,selected_action=None,eligible_actions=[])]
    assert not r['fallback_execution'] and r['first_terminal']['tick']==10
    assert r['first_terminal']['failure']=='synthetic failure'


@pytest.mark.parametrize('fault',['bias','future','surface','path','radius','selected','tape','endpoint','forecast','order'])
def test_bad_evidence_cannot_be_scored_as_executed_recovery(fault):
    poses,tape,rows=data(); d=rows[-1]['decision']; s=d['new_selection']; r=s['residual_first_interval_feasibility']
    if fault=='bias': r['correction_xy_m']=[.02,0.]
    elif fault=='future': s['causal_score_residual_receipt']['residuals'][-1]['available_tick']=11
    elif fault=='surface': s['surface_checks'][1]['possible_intersection']=True
    elif fault=='path': r['corrected_nominal_path_checks'][1]['all_predicted_segments_nominally_clear']=False
    elif fault=='radius': r['corrected_nominal_path_checks'][1]['segments'][0]['radius_m']=.4
    elif fault=='selected': d['selected_action']='hold'
    elif fault=='tape': tape[-1]['requested_command']=[0.,0.,0.]
    elif fault=='endpoint': tape[-1]['post_sample_index']=1298
    elif fault=='forecast': s['prediction'][1][0][0]=float('nan')
    elif fault=='order': rows.pop(5)
    with pytest.raises(ValueError): summarize_execution(poses,tape,rows)

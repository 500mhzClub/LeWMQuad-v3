import copy

import numpy as np
import pytest

from lewm.successive_choice_maze_development import trials
from lewm.successive_choice_metrics_development import reduce_trial,paired_reduction


def fixture():
    times=.002*np.arange(1,3001); pose=np.zeros((3000,7)); pose[:,1]=np.maximum(times-1.5,0)*.1
    pose[:,2]=.3; pose[:,5:]=np.sqrt(.5)
    raw={'timestamp_s':times,'base_pose_world':pose,'base_twist_world':np.zeros((3000,6)),
        'physics_contact':np.zeros(3000,dtype=bool)}
    tape=[{'pre_sample_index':749+i*50,'post_sample_index':799+i*50,'stage':'control' if i<40 else 'release',
        'decision_index':i//5 if i<40 else None} for i in range(45)]
    selections=[{'selected_action_index':1,'mean_motion_sin_cos':[[.05,0,0,1]]*5,
        'mean_contact_probability':[.1]*5} for _ in range(8)]
    return raw,tape,selections


def reduce(raw,tape,selections,**kwargs):
    return reduce_trial(raw,start_index=749,tape=tape,selections=selections,direction=[.8,0],
        branchable=kwargs.get('branchable',True),stop_reason=kwargs.get('stop_reason'),sensor_fault=kwargs.get('sensor_fault'))


def test_initial_frame_progress_and_all_executed_half_seconds():
    raw,tape,choices=fixture(); result=reduce(raw,tape,choices)
    assert result['complete_control_and_release'] and result['release_motion_pass']
    assert result['four_second_signed_displacement_m']==pytest.approx(.4)
    assert result['observed_lateral_control_displacement_m']==pytest.approx(0,abs=1e-12)
    assert len(result['executed_decision_errors'])==8
    for row in result['executed_decision_errors']:
        assert row['position_error_m']==pytest.approx(0,abs=1e-12) and row['contact_brier']==pytest.approx(.01)


def test_early_contact_censors_future_without_deleting_failure():
    raw,tape,choices=fixture(); raw={k:v[:851] for k,v in raw.items()}; raw['physics_contact'][-1]=True
    tape[2]['post_sample_index']=850
    result=reduce(raw,tape[:3],choices[:1],stop_reason='DISALLOWED_CONTACT')
    assert result['any_contact'] and result['four_second_signed_displacement_m'] is None
    assert result['observed_signed_control_displacement_m']>0
    error=result['executed_decision_errors'][0]
    assert error['label']['contact_valid'] and error['label']['contact_by_horizon']
    assert error['position_error_m'] is None and error['contact_brier']==pytest.approx(.81)


def test_fault_release_does_not_fill_missing_candidate_horizon():
    raw,tape,choices=fixture(); raw={k:v[:1100] for k,v in raw.items()}
    tape=tape[:7]
    for row in tape[2:]: row.update(stage='fault_release',decision_index=None)
    result=reduce(raw,tape,choices[:1],stop_reason='SENSOR_CONTRACT_FAILURE',sensor_fault='test')
    assert result['sensor_failure'] and result['release_complete']
    error=result['executed_decision_errors'][0]
    assert not error['label']['contact_valid'] and not error['label']['motion_valid']
    assert error['position_error_m'] is None and error['contact_brier'] is None


def test_prefix_failure_retained_and_no_progress_imputation():
    raw,_,_=fixture(); raw={k:v[:750] for k,v in raw.items()}
    result=reduce(raw,[],[],branchable=False,stop_reason='PREFIX_NOT_CROSSED')
    assert result['prefix_failure'] and result['observed_signed_control_displacement_m'] is None
    assert not result['complete_control_and_release']


def test_predispatch_fault_does_not_count_release_as_learned_motion():
    raw,tape,choices=fixture(); raw={k:v[:1000] for k,v in raw.items()}
    failed=copy.deepcopy(tape[0]); failed['post_sample_index']=failed['pre_sample_index']
    release=tape[:5]
    for row in release: row.update(stage='fault_release',decision_index=None)
    result=reduce(raw,[failed,*release],choices[:1],sensor_fault='pre-dispatch',stop_reason='SENSOR_CONTRACT_FAILURE')
    assert result['observed_control_duration_s']==0 and result['release_complete']
    assert result['executed_decision_errors'][0]['label'] is None


def test_paired_panel_failure_population_and_explicit_missing_progress():
    raw,tape,choices=fixture(); metrics=reduce(raw,tape,choices)
    rows=[r | {'metrics':copy.deepcopy(metrics)} for r in trials()]
    failed=next(r for r in rows if r['method']=='jepa_rollout')
    failed['metrics'].update(any_contact=True,complete_control_and_release=False,four_second_signed_displacement_m=None)
    result=paired_reduction(rows); comparison=result['paired_comparisons']['jepa_rollout_minus_supervised_rollout']
    assert comparison['any_contact']['mean_delta']==pytest.approx(1/24)
    assert comparison['completed_pair_progress']['omitted_intent_pairs']==1
    assert len(result['layout_methods'])==48
    with pytest.raises(ValueError,match='incomplete paired'): paired_reduction(rows[:-1])

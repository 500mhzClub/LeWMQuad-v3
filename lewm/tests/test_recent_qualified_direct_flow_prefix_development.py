"""Reject acausal reference receipts and never consume changed-command futures."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.recent_qualified_direct_flow_prefix_development import PrefixComparison


def row(frame,*,previous=None,attempt=False,qualified=True):
    now=1_500_000_000+frame*100_000_000
    visual=dict(status='CURRENT_VISUAL_POSE',current_pose=dict(frame=frame,measured_ns=now),
        continuity_evidence=dict(status='INITIAL_REFERENCE' if frame==0 else 'ANCHOR_MEASUREMENT'))
    old=dict(controller='direct_flow_floor_transport_controller_v1',
        requested_command=[0.,0.,0.],terminal=None,original_visual_evidence=visual,
        new_selection={'prediction':np.zeros((6,8,5)).tolist()},mission_receipt={'fixture':True})
    new=deepcopy(old);new.update(controller='recent_qualified_direct_flow_controller_v1',recent_qualified_anchor_enabled=True)
    receipt=dict(retained_from_frame=previous,retained_current_frame=frame,maximum_additional_references=1,
        original_reference_population_preserved_before_search=True,reference_retained_from_bridge=False,
        reference_retained_from_floor_transport=False,bridge_limit_unchanged=True,pose_uncertainty_calibrated=False,attempts=[])
    if attempt:
        receipt['attempts']=[dict(camera='auxiliary',current_frame=frame,reference_frame=frame-1,
            reference_measured_ns=now-100_000_000,reference_was_anchor_qualified=True,
            reference_is_immediately_previous=True,rigid_thresholds_unchanged=True,qualified=qualified,
            direct_flow_mode=False,failure=None if qualified else 'unchanged pair gate failed',
            original_reference_selection=dict(status='NO_QUALIFIED_REFERENCE',retained_references=1,
                attempts=[dict(reference_frame=0,status='REJECTED',reason='missing matches')]))]
    new['original_visual_evidence']['recent_qualified_anchor']=receipt
    return old,new


def warm():
    comparison=PrefixComparison()
    for i in range(2):
        old,new=row(i,previous=None if i==0 else i-1)
        result=comparison.compare(old,new,old['requested_command'],frame=i)
        assert result['complete_original_decision_exact'] and not result['stop']
    return comparison


@pytest.mark.parametrize('fault',[None,'prior_bridge','reference_time','reference_frame','conflict',
    'original_accepted','duplicate_reference','too_many_attempts','pair_failure','retained_bridge','forecast','input_command'])
def test_actual_causal_reference_extension_and_raw_banks(monkeypatch,fault):
    comparison=warm();old,new=row(2,previous=1,attempt=True)
    receipt=new['original_visual_evidence']['recent_qualified_anchor'];a=receipt['attempts'][0]
    if fault=='prior_bridge':comparison.last_qualified_frame=None
    elif fault=='reference_time':a['reference_measured_ns']+=1
    elif fault=='reference_frame':a['reference_frame']=0
    elif fault=='conflict':a['original_reference_selection']['status']='CONFLICTING_ALTERNATIVES'
    elif fault=='original_accepted':a['original_reference_selection']['attempts'][0]['status']='ACCEPTED_PRIMARY'
    elif fault=='duplicate_reference':a['original_reference_selection']['attempts'][0]['reference_frame']=1
    elif fault=='too_many_attempts':receipt['attempts']*=5
    elif fault=='pair_failure':a['failure']='failed fit'
    elif fault=='retained_bridge':new['original_visual_evidence']['continuity_evidence']['status']='MEASURED_INCREMENT_BRIDGE'
    elif fault=='forecast':new['new_selection']['prediction'][0][0][0]=1.
    elif fault=='input_command':old['requested_command']=[.1,0.,0.]
    if fault is None:
        new['mission_receipt']['different_observed_state']=True
        result=comparison.compare(old,new,[0.,0.,0.],frame=2)
        assert result['first_reference_attempt']==result['first_qualified_reference']==result['first_decision_difference']==2
        assert result['raw_model_forecasts_compared'] and not result['stop']
    else:
        with pytest.raises(ValueError):comparison.compare(old,new,[0.,0.,0.],frame=2)


def test_no_unannounced_state_difference_before_reference_attempt():
    comparison=warm();old,new=row(2,previous=1)
    new['mission_receipt']['changed']=True
    with pytest.raises(ValueError,match='before any extra reference attempt'):
        comparison.compare(old,new,[0.,0.,0.],frame=2)


@pytest.mark.parametrize('stop_kind',['command','original_terminal','candidate_terminal'])
def test_first_changed_command_or_either_terminal_forbids_following_observation(stop_kind):
    comparison=warm();old,new=row(2,previous=1,attempt=True)
    if stop_kind=='command':new['requested_command']=[.16,0.,.45]
    elif stop_kind=='original_terminal':old['terminal']='SENSOR_OR_MODEL_FAILURE'
    else:new['terminal']='SENSOR_OR_MODEL_FAILURE'
    result=comparison.compare(old,new,[0.,0.,0.],frame=2)
    assert result['stop'] and result['requested_command_changed']==(stop_kind=='command')
    old,new=row(3,previous=2)
    with pytest.raises(ValueError):comparison.compare(old,new,[0.,0.,0.],frame=3)


def test_rejected_extra_pair_is_not_reported_as_qualified():
    comparison=warm();old,new=row(2,previous=1,attempt=True,qualified=False)
    result=comparison.compare(old,new,[0.,0.,0.],frame=2)
    assert result['first_reference_attempt']==2 and result['first_qualified_reference'] is None
    assert result['extra_qualified_references']==0 and result['complete_original_decision_exact']


@pytest.mark.parametrize('side',['original','candidate'])
@pytest.mark.parametrize('flag',[True,False])
def test_height_intervention_field_is_not_admitted(side,flag):
    old,new=row(0)
    (old if side=='original' else new)['partial_floor_height_constraint_enabled']=flag
    with pytest.raises(ValueError):PrefixComparison().compare(old,new,[0.,0.,0.],frame=0)


def test_full_population_bound_includes_failed_observation_and_excludes_drain():
    from lewm.recent_qualified_direct_flow_prefix_development import MAX_FRAMES
    assert MAX_FRAMES==1207
    comparison=PrefixComparison()
    comparison.next_frame=1206
    old,new=row(1206)
    new['original_visual_evidence']['recent_qualified_anchor']['retained_from_frame']=None
    result=comparison.compare(old,new,[0.,0.,0.],frame=1206)
    assert not result['stop']
    old,new=row(1207,previous=1206)
    with pytest.raises(ValueError):comparison.compare(old,new,[0.,0.,0.],frame=1207)

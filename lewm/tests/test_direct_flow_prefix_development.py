"""Causal boundary: altered prior decisions and borrowed failure evidence reject."""
from copy import deepcopy
import pytest
from lewm.direct_flow_prefix_development import compare_step, BOUNDARY_FRAME


def rows(frame=10):
    old=dict(controller='measured_floor_transport_round_trip_controller_v1',tick=frame,
        terminal=None,failure=None,requested_command=[0.,0.,0.],evidence={'map_pose':'current'},
        new_selection={'action':None,'prediction':[[1.]]},mission_receipt={'state':'OUTBOUND'},
        original_visual_evidence={'status':'CURRENT_VISUAL_POSE','decision_ns':1_500_000_000+frame*100_000_000})
    new=deepcopy(old); new.update(controller='direct_flow_floor_transport_controller_v1',
        direct_corner_flow_missingness_fallback_enabled=True)
    return old,new


def boundary():
    old,new=rows(BOUNDARY_FRAME)
    old.update(tick=BOUNDARY_FRAME-1,terminal='SENSOR_OR_MODEL_FAILURE',failure='missing current pose',
        new_selection=None,evidence=None)
    old['original_visual_evidence'].update(status='VISUAL_TERMINAL_FAILURE',camera_selection={'original':'camera'},
        continuity_evidence={'original':'continuity'},reference_selection={'original':'references'})
    new['original_visual_evidence'].update(current_pose={'frame':BOUNDARY_FRAME},direct_corner_flow_fallback=dict(
        frame=BOUNDARY_FRAME,measured_ns=1_500_000_000+BOUNDARY_FRAME*100_000_000,
        original_camera_selection=deepcopy(old['original_visual_evidence']['camera_selection']),
        original_auxiliary_continuity=deepcopy(old['original_visual_evidence']['continuity_evidence']),
        original_reference_selection=deepcopy(old['original_visual_evidence']['reference_selection']),
        association_rule_changed=True,rigid_geometry_thresholds_unchanged=True,
        temporal_continuity_thresholds_unchanged=True,bridge_budget_unchanged=True,
        reference_or_pose_history_reset=False,accepted=True))
    return old,new


def test_complete_decisions_exact_before_boundary():
    old,new=rows()
    r=compare_step(old,new,[0.,0.,0.],frame=10)
    assert r['complete_original_decision_exact'] and r['raw_model_forecasts_compared'] and not r['stop']


@pytest.mark.parametrize('field,value',[('requested_command',[.2,0.,0.]),
    ('mission_receipt',{'state':'RETURN'}),('new_selection',{'action':None,'prediction':[[2.]]}),
    ('evidence',{'map_pose':'different'}),('terminal','SENSOR_OR_MODEL_FAILURE')])
def test_any_preboundary_controller_change_rejects(field,value):
    old,new=rows(); new[field]=value
    with pytest.raises(ValueError): compare_step(old,new,[0.,0.,0.],frame=10)


def test_old_request_must_match_physically_completed_tape():
    old,new=rows()
    with pytest.raises(ValueError): compare_step(old,new,[.2,0.,0.],frame=10)


def test_successful_current_boundary_stops_even_if_command_stays_zero():
    old,new=boundary(); r=compare_step(old,new,[0.,0.,0.],frame=BOUNDARY_FRAME)
    assert r['controller_recovered'] and r['stop'] and r['terminal_changed']
    assert not r['requested_command_changed'] and not r['raw_model_forecasts_compared']


@pytest.mark.parametrize('key',['original_camera_selection','original_auxiliary_continuity','original_reference_selection'])
def test_original_failure_evidence_cannot_be_replaced(key):
    old,new=boundary(); new['original_visual_evidence']['direct_corner_flow_fallback'][key]={}
    with pytest.raises(ValueError): compare_step(old,new,[0.,0.,0.],frame=BOUNDARY_FRAME)


def test_rejected_fallback_is_a_terminal_negative_not_a_recovery():
    old,new=boundary(); new.update(tick=BOUNDARY_FRAME-1,terminal='SENSOR_OR_MODEL_FAILURE',
        failure='bounded measured bridge exhausted',new_selection=None,evidence=None)
    new['original_visual_evidence'].update(status='VISUAL_TERMINAL_FAILURE',current_pose=None)
    new['original_visual_evidence']['direct_corner_flow_fallback']['accepted']=False
    r=compare_step(old,new,[0.,0.,0.],frame=BOUNDARY_FRAME)
    assert r['stop'] and r['fallback_attempted'] and not r['controller_recovered']


def test_no_fallback_requires_exact_original_failure():
    old,_=boundary(); new=deepcopy(old); new.update(controller='direct_flow_floor_transport_controller_v1',
        direct_corner_flow_missingness_fallback_enabled=True)
    r=compare_step(old,new,[0.,0.,0.],frame=BOUNDARY_FRAME)
    assert r['stop'] and r['complete_original_decision_exact'] and not r['controller_recovered']
    new['failure']='different'
    with pytest.raises(ValueError): compare_step(old,new,[0.,0.,0.],frame=BOUNDARY_FRAME)


@pytest.mark.parametrize('fault',['stale_pose','no_mapping','no_forecast','bad_command','unaccepted_fallback'])
def test_pose_alone_cannot_claim_full_controller_recovery(fault):
    old,new=boundary()
    if fault=='stale_pose': new['original_visual_evidence']['current_pose']['frame']-=1
    elif fault=='no_mapping': new['evidence']=None
    elif fault=='no_forecast': new['new_selection'].pop('prediction')
    elif fault=='bad_command': new['requested_command']=[.2,0.,0.]
    else: new['original_visual_evidence']['direct_corner_flow_fallback']['accepted']=False
    with pytest.raises(ValueError): compare_step(old,new,[0.,0.,0.],frame=BOUNDARY_FRAME)


def test_next_recorded_observation_is_outside_admitted_scope():
    old,new=rows(BOUNDARY_FRAME+1)
    with pytest.raises(ValueError): compare_step(old,new,[0.,0.,0.],frame=BOUNDARY_FRAME+1)

from copy import deepcopy
from functools import partial
import numpy as np
import pytest
from lewm.dual_camera_controller_prefix_comparison_development import compare_primary_decision
from lewm.dual_camera_settled_controller_development import DualCameraSettledController
from lewm.settled_boundary_round_trip_development import SettledBoundaryRoundTripController
from lewm.fast_gyro_development import FastGyroBuffer
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb


@pytest.fixture
def initial(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    monkeypatch.setattr(fixture,'visual',partial(visual,origin=1_500_000_000))
    p,d,a,_,now=packets()
    image=from_captured_rgb(p['image']['rgb'],a,p,measured_ns=now,available_ns=now,now_ns=now)
    gyro=FastGyroBuffer((0,0,0))
    for t in range(now-100_000_000,now+1,2_000_000):
        gyro.append(np.zeros(3),np.ones(3,bool),measured_ns=t,available_ns=t)
    f=gyro.packet(now_ns=now)
    kwargs=dict(public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True),navigation_ticks=100,condition='jepa',variant='full',persistent=True)
    old=SettledBoundaryRoundTripController(None,None,**kwargs)
    new=DualCameraSettledController(None,None,**kwargs)
    original=old.observe(p,d,f,auxiliary_depth=a,now_ns=now)
    candidate=new.observe(p,d,f,auxiliary_depth=a,auxiliary_rgb=image,now_ns=now)
    assert original['terminal'] is candidate['terminal'] is None
    return original,candidate,p,image,a,now


def test_real_initial_controller_decisions_equal_after_only_declared_metadata(initial):
    original,candidate,p,image,a,now=initial
    before=deepcopy(candidate)
    result=compare_primary_decision(original,candidate,p,image,a,now_ns=now)
    assert result['requested_command_exact'] and result['no_auxiliary_intervention']
    assert candidate == before


@pytest.mark.parametrize('fault',['command','mission','extra','binding','camera','registered_raw'])
def test_no_behavior_or_witness_difference_can_be_hidden_by_normalization(initial,fault):
    original,candidate,p,image,a,now=initial
    if fault=='command':candidate['requested_command']=[.2,0.,0.]
    if fault=='mission':candidate['mission_receipt']['phase']='RETURN'
    if fault=='extra':candidate['unrecognized_field']=True
    if fault=='binding':candidate['original_visual_evidence']['current_pose']['auxiliary_rgb_sha256']='0'*64
    if fault=='camera':candidate['original_visual_evidence']['camera_selection']['auxiliary_attempted']=True
    if fault=='registered_raw':candidate['evidence']['original_visual_evidence']['observer_variant']='other'
    with pytest.raises(ValueError):compare_primary_decision(original,candidate,p,image,a,now_ns=now)

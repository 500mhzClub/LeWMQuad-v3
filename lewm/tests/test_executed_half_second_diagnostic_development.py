import math
import numpy as np
import pytest
from lewm.executed_half_second_diagnostic_development import executed_outcome, error


def fixture():
    n=1000
    pose=np.zeros((n,7));pose[:,6]=1
    pose[749,3:]=[0,0,math.sin(math.pi/4),math.cos(math.pi/4)]
    pose[999,:3]=[0, .2, 0]
    pose[999,3:]=[0,0,math.sin((math.pi/2+.1)/2),math.cos((math.pi/2+.1)/2)]
    raw=dict(base_pose_world=pose,timestamp_s=(np.arange(n)+1)*.002,
        requested_command=np.zeros((n,3)),physics_contact=np.zeros(n,bool))
    tape=[dict(tick=i,completed=True,requested_command=[0.,0.,0.],
        pre_sample_index=749+50*i,post_sample_index=799+50*i) for i in range(5)]
    return raw,tape


def test_current_window_frame_and_horizon_contact():
    raw,tape=fixture();raw['physics_contact'][749]=True
    result=executed_outcome(raw,tape,tick=0,action='hold')
    np.testing.assert_allclose(result['xy_m'],[.2,0],atol=1e-15)
    assert result['yaw_rad']==pytest.approx(.1) and not result['contact']
    raw['physics_contact'][999]=True
    assert executed_outcome(raw,tape,tick=0,action='hold')['contact']


def test_interruption_and_mismatched_action_are_excluded():
    raw,tape=fixture();tape[-1]['completed']=False
    assert not executed_outcome(raw,tape,tick=0,action='hold')['eligible']
    tape[-1]['completed']=True;tape[-1]['requested_command']=[.2,0,0]
    assert not executed_outcome(raw,tape,tick=0,action='hold')['eligible']
    assert not executed_outcome(raw,tape[:4],tick=0,action='hold')['eligible']


def test_raw_command_mismatch_fails_and_angle_wrap_is_shortest():
    raw,tape=fixture();raw['requested_command'][900,0]=.2
    with pytest.raises(AssertionError):executed_outcome(raw,tape,tick=0,action='hold')
    o=dict(eligible=True,xy_m=[0,0],yaw_rad=math.pi-.01,contact=False)
    p=[0,0,math.sin(-math.pi+.01),math.cos(-math.pi+.01),0]
    assert error(p,o)['yaw_error_rad']==pytest.approx(.02)
    assert error([0,0,0,0,0],o)['yaw_error_rad'] is None

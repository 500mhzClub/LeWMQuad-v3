"""Complete summaries retain bridge endings and failed/unavailable observations."""
from copy import deepcopy

import pytest

from scripts.read_go2_joint_room_return_science_v1 import sensor_summary


def rows():
    result=[]
    for i,mode in enumerate(('INITIAL_REFERENCE','MEASURED_INCREMENT_BRIDGE','ANCHOR_MEASUREMENT','UNAVAILABLE')):
        e=dict(current_pose=None if mode=='UNAVAILABLE' else dict(frame=i,mode='joint'),
            continuity_evidence=dict(status=mode))
        d=dict(evidence=e,execution=dict(local_decision=None,mission_pulses=1,attempted_legs=1),
            stage='FORWARD_ONE',terminal='ROOM_RETURN_FAILED' if i==3 else None,
            reason='missing sensor' if i==3 else None,completed_stages=0)
        result.append(dict(tick=i,observation_index=i,evidence=e,decision=d))
    return result


def test_summary_retains_complete_failed_population_and_bridge_rejoin():
    r=rows();before=deepcopy(r);s=sensor_summary(r,4)
    assert r==before and s['frames']==4 and s['unavailable_frames']==[3]
    assert s['terminal']=='ROOM_RETURN_FAILED' and s['completed_stages']==0
    assert s['bridge_spans']==[dict(start_frame=1,end_frame=1,frames=1,following_frame=2,following_mode='ANCHOR_MEASUREMENT')]


@pytest.mark.parametrize('fault',['truncated','reordered','relabelled','different_input'])
def test_incomplete_or_misidentified_sensor_population_rejected(fault):
    r=rows()
    if fault=='truncated':r.pop()
    if fault=='reordered':r[1],r[2]=r[2],r[1]
    if fault=='relabelled':r[1]['evidence']['current_pose']['mode']='gyro'
    if fault=='different_input':r[1]['decision']['evidence']=None
    with pytest.raises(ValueError):sensor_summary(r,4)


def test_bridge_at_recording_end_is_not_claimed_rejoined():
    s=sensor_summary(rows()[:2],2)
    assert s['bridge_spans'][0]['following_mode']=='END_OF_RECORDING'
    assert s['bridge_spans'][0]['following_frame'] is None

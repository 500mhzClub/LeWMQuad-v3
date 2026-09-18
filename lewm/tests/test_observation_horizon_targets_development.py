from copy import deepcopy
import numpy as np
import pytest
import torch
from lewm.observation_horizon_targets_development import derive,verify_half_second_overlap
from lewm.observation_horizon_plan_development import plan,validate_plan


def raw(*,n=1400,contact_at=None):
    poses=np.zeros((n,7));poses[:,6]=1.;poses[:,0]=np.arange(n)*.0002
    events=np.zeros(n,dtype=bool)
    if contact_at is not None:events[contact_at]=True
    commands=np.zeros((n,3));commands[900:,0]=.2
    values=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=poses,
        physics_contact=events,requested_command=commands)
    return values,[dict(physical_sample_index=i) for i in range(749,n,50)]


def test_actual_short_clocks_and_shared_half_second_are_exact():
    r,c=raw();labels=derive(r,c,frame=3,commands=[[.2,0,0]]*8)
    assert [t['offset_ns'] for t in labels['targets']]==list(range(100_000_000,800_000_001,100_000_000))
    assert [t['future_observation_index'] for t in labels['targets']]==list(range(4,12))
    assert labels['targets'][0]['motion'][0]==pytest.approx(.01)
    old=[deepcopy(labels['targets'][4])]
    assert verify_half_second_overlap(labels,old)
    old[0]['motion'][0]+=.001
    with pytest.raises(ValueError,match='500-ms'):verify_half_second_overlap(labels,old)


def test_contact_absorbs_and_missing_acquisition_never_becomes_zero_motion():
    r,c=raw(n=1001,contact_at=975);labels=derive(r,c,frame=3,commands=[[.2,0,0]]*8)
    assert labels['targets'][0]['motion_valid']
    for t in labels['targets'][1:]:
        assert t['contact_valid'] and t['contact']==1.
        assert t['motion'] is None and not t['future_image_valid'] and t['future_observation_index'] is None
    r,c=raw(n=975);labels=derive(r,c,frame=3,commands=[[.2,0,0]]*8)
    assert not labels['targets'][1]['contact_valid'] and labels['targets'][1]['contact'] is None
    assert labels['targets'][1]['motion'] is None
    r,c=raw(contact_at=800)
    assert derive(r,c,frame=3,commands=[[.2,0,0]]*8)['reason']=='POST_CONTACT_CONTEXT'


def test_unknown_plan_padding_and_wrong_executed_command_are_rejected():
    blocks,valid=plan('forward',offset_ticks=35)
    active,offsets=validate_plan(blocks[None],valid[None],1)
    assert active.sum()==5 and offsets.tolist()==[[100_000_000,200_000_000,300_000_000,400_000_000,500_000_000,0,0,0]]
    blocks[7,0,0]=.1
    with pytest.raises(ValueError,match='unknown padding'):validate_plan(blocks[None],valid[None],1)
    with pytest.raises(ValueError):validate_plan(torch.zeros(1,8,5,3),torch.ones(1,8,5,dtype=torch.bool),1)
    r,c=raw();r['requested_command'][920,0]=0.
    with pytest.raises(AssertionError):derive(r,c,frame=3,commands=[[.2,0,0]]*8)

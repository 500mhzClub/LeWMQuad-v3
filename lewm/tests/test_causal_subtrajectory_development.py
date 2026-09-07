import math

import numpy as np
import pytest
import torch

from lewm.causal_subtrajectory_development import branch_windows,frame_lookup
from lewm.causal_subtrajectory_learning_development import remaining_plan,causal_history_tensors,AuditedSubtrajectoryDataset


def fixture(end=5_000_000_000,contact_ns=None):
    ns=np.arange(2_000_000,end+1,2_000_000,dtype=np.int64)
    raw={'timestamp_s':ns/1e9,'base_pose_world':np.tile([0.,0.,.3,0.,0.,0.,1.],(len(ns),1)),
        'physics_contact':ns==contact_ns}
    raw['base_pose_world'][:,0]=ns/1e9*.2
    frame_ns=list(range(700_000_000,end+1,100_000_000))
    if end not in frame_ns: frame_ns.append(end)
    frames=[{'image_ns':n,'decision_ns':n} for n in frame_ns]
    return raw,499,frames,frames.copy()


def test_full_windows_have_causal_history_remaining_plan_and_relative_targets():
    rows=branch_windows(*fixture())
    assert len(rows)==8
    assert [r['remaining_ticks'] for r in rows]==[40,35,30,25,20,15,10,5]
    assert rows[0]['history_observation_indices']==[0,1,2,3]
    for row in rows:
        assert sum(t['motion_valid'] for t in row['targets'])==row['remaining_ticks']//5
        for target in row['targets']:
            if target['motion_valid']:
                assert target['delta_xy_yaw_current_body']==pytest.approx([target['horizon_ns']/1e9*.2,0,0])
                assert target['contact_by_horizon'] is False
            else:
                assert target['contact_by_horizon'] is None
                assert target['future_observation_index'] is None


@pytest.mark.parametrize('contact_ns',[2_500_000_000,2_502_000_000])
def test_first_contact_censors_motion_but_absorbs_contact_within_remaining_plan(contact_ns):
    rows=branch_windows(*fixture(contact_ns,contact_ns))
    assert all(r['decision_ns']<contact_ns for r in rows)
    for row in rows:
        for target in row['targets']:
            at=row['decision_ns']+target['horizon_ns']
            if target['in_plan']:
                assert target['contact_valid']
                assert target['contact_by_horizon']==(at>=contact_ns)
                assert target['motion_valid']==(at<contact_ns)
            else: assert not target['contact_valid']


def test_noncontact_termination_is_censored_not_safe():
    rows=branch_windows(*fixture(2_502_000_000))
    assert len(rows)==4
    for row in rows:
        for target in row['targets']:
            if row['decision_ns']+target['horizon_ns']>2_502_000_000:
                assert not target['contact_valid'] and not target['motion_valid']


def test_relative_labels_use_full_current_tilt_and_translation_not_subtracted_xy():
    args=fixture(); raw=args[0]; angle=.4
    raw['base_pose_world'][:,3:]=[0,math.sin(angle/2),0,math.cos(angle/2)]
    raw['base_pose_world'][:,2]=.3+raw['timestamp_s']*.1
    target=branch_windows(*args)[2]['targets'][0]
    assert target['delta_xy_yaw_current_body']==pytest.approx([.1*math.cos(angle)-.05*math.sin(angle),0,0])


def test_missing_past_or_future_frame_cannot_be_padded():
    raw,start,frames,canonical=fixture()
    with pytest.raises(ValueError,match='past image'): branch_windows(raw,start,frames,canonical[1:])
    frames=[r for r in frames if r['image_ns']!=2_500_000_000]
    with pytest.raises(ValueError,match='lacks actual RGB'): branch_windows(raw,start,frames,canonical)


def test_canonical_only_at_initial_time_later_context_uses_own_indices():
    raw,start,frames,canonical=fixture()
    canonical=[{'image_ns':600_000_000,'decision_ns':600_000_000}]+canonical
    rows=branch_windows(raw,start,frames,canonical)
    assert rows[0]['history_observation_indices']==[1,2,3,4]
    assert rows[1]['history_observation_indices']==[5,6,7,8]


@pytest.mark.parametrize('frames',[
    [{'image_ns':1,'decision_ns':2}],
    [{'image_ns':1,'decision_ns':1}]*2,
    [{'image_ns':2,'decision_ns':2},{'image_ns':1,'decision_ns':1}],
    [{'image_ns':True,'decision_ns':True}],
])
def test_bad_frame_clocks_rejected(frames):
    with pytest.raises(ValueError): frame_lookup(frames)


def test_remaining_plan_reconstructs_from_current_past_and_masks_unknown_tail():
    row=remaining_plan([.2,0,.5],[-.2,0,-.5],5)
    assert row['known_action_blocks'][0,0].tolist()==pytest.approx([.05/.3,0,-.15/.5])
    assert row['known_action_valid'].sum()==5
    assert torch.count_nonzero(row['known_action_blocks'][1:])==0
    assert not row['known_action_valid'][1:].any()


@pytest.mark.parametrize('ticks',[0,1,41,True,5.0])
def test_invalid_plan_duration_rejected(ticks):
    with pytest.raises(ValueError): remaining_plan([0,0,0],[0,0,0],ticks)


def test_history_rejects_episode_mix_and_future_without_tensor_conversion():
    packets=[{'image':{'measured_ns':700_000_000+i*100_000_000},
        'sensor_state':{'decision_ns':700_000_000+i*100_000_000,'identity':(0,0,0)}} for i in range(4)]
    packets[0]['sensor_state']['identity']=(1,0,0)
    with pytest.raises(ValueError,match='episode'): causal_history_tensors(packets,1_000_000_000)
    packets[0]['sensor_state']['identity']=(0,0,0); packets[0]['image']['measured_ns']=1_100_000_000
    with pytest.raises(ValueError,match='clock'): causal_history_tensors(packets,1_000_000_000)


def test_loader_rejects_unapproved_root_before_access(tmp_path):
    with pytest.raises(ValueError,match='exact'): AuditedSubtrajectoryDataset(tmp_path/'sealed','train')

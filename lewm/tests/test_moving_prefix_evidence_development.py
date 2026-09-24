import hashlib
import json

import numpy as np
import pytest

from lewm.moving_prefix_evidence_development import reference_context,suffix_targets


def fixture():
    times=.002*np.arange(1,2001); pose=np.zeros((2000,7)); pose[:,0]=times*.2; pose[:,2]=.3; pose[:,6]=1
    raw={'timestamp_s':times,'base_pose_world':pose,'physics_contact':np.zeros(2000,dtype=bool)}
    frames=[{'image_ns':t*100_000_000,'decision_ns':t*100_000_000} for t in range(1,41)]
    return raw,frames


def test_six_known_targets_and_no_release_leakage():
    raw,frames=fixture(); result=suffix_targets(raw,249,1749,frames)
    assert result['decision_ns']==500_000_000 and result['history_observation_indices']==[1,2,3,4]
    assert all(t['motion_valid'] and t['contact_valid'] for t in result['targets'][:6])
    assert result['targets'][5]['delta_xy_yaw_current_body']==pytest.approx([.6,0,0])
    assert all(not t['in_plan'] and not t['motion_valid'] and not t['contact_valid'] and t['contact_by_horizon'] is None for t in result['targets'][6:])


def test_early_contact_absorbs_only_within_known_plan():
    raw,frames=fixture(); raw['physics_contact'][499]=True
    targets=suffix_targets(raw,249,499,frames)['targets']
    assert all(t['contact_by_horizon'] and not t['motion_valid'] for t in targets[:6])
    assert all(t['contact_by_horizon'] is None for t in targets[6:])


def test_post_suffix_contact_cannot_change_predictions_target_labels():
    raw,frames=fixture(); before=suffix_targets(raw,249,1749,frames)
    raw['physics_contact'][1750]=True
    assert suffix_targets(raw,249,1749,frames)==before


def test_noncontact_early_stop_and_missing_rgb_are_not_imputed():
    raw,frames=fixture(); targets=suffix_targets(raw,249,699,frames)['targets']
    assert targets[0]['motion_valid'] and all(not t['contact_valid'] for t in targets[1:])
    with pytest.raises(ValueError,match='actual RGB'): suffix_targets(raw,249,1749,frames[:7])
    raw['physics_contact'][249]=True
    with pytest.raises(ValueError,match='conditioning'): suffix_targets(raw,249,1749,frames)


def reference_fixture(tmp_path):
    times=.002*np.arange(1,701)
    raw={'timestamp_s':times,'physics_contact':np.zeros(700,dtype=bool),'position':np.zeros((700,3))}
    np.savez_compressed(tmp_path/'physics_trace.npz',**raw)
    ns=np.arange(1,15)*100_000_000
    np.savez_compressed(tmp_path/'policy_histories.npz',image_ns=ns,decision_ns=ns)
    camera=[{'physical_sample_index':(i+1)*50-1,'rgb_sha256':'0'*64} for i in range(14)]
    (tmp_path/'camera_audit.json').write_text(json.dumps(camera))
    member={'scene_id':'example-forward','layout_id':'example','data_role':'train','action_index':1,
        'prefix_terminal_sample_index':149,'artifact_sha256':{name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
            for name in ('physics_trace.npz','policy_histories.npz','camera_audit.json')}}
    window={'offset_ns':1_000_000_000,'scene_id':'example-forward','decision_ns':1_300_000_000,
        'history_observation_indices':[9,10,11,12]}
    return member,window


def test_reference_prefix_is_bound_through_the_moving_context(tmp_path):
    member,window=reference_fixture(tmp_path); value=reference_context(tmp_path,member,window)
    assert value['prefix_terminal_sample_index']==649 and value['branch_start_observation_index']==12
    assert value['prefix_binding']['physics_samples']==650 and value['prefix_binding']['timestamp_ns']==1_300_000_000


def test_reference_rejects_artifact_tampering(tmp_path):
    member,window=reference_fixture(tmp_path)
    (tmp_path/'camera_audit.json').write_text('[]')
    with pytest.raises(ValueError,match='binding'): reference_context(tmp_path,member,window)


def test_reference_rejects_wrong_conditioning_clock(tmp_path):
    member,window=reference_fixture(tmp_path); window['decision_ns']+=100_000_000
    with pytest.raises(ValueError,match='unavailable moving'): reference_context(tmp_path,member,window)

"""Moving contexts, remaining commands and independent post-stop modality masks."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lewm.geometry_progress_layout_family_development import assignments,candidate_commands
from lewm.geometry_progress_family_causal_windows_development import derive,materialize,remaining_candidate
from lewm.tests.test_independent_pulse_context_development import policy
from scripts.read_go2_geometry_progress_commands_v1 import native_horizons


def fixture(n=2900,contact_at=None):
    trial=next(iter(assignments()));pose=np.zeros((n,7));pose[:,6]=1.
    pose[:,0]=np.arange(n)*.0002;pose[:,1]=np.arange(n)*.0001
    contact=np.zeros(n,bool)
    if contact_at is not None:contact[contact_at:]=True
    raw=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=pose,physics_contact=contact)
    cameras=[dict(physical_sample_index=i) for i in range(749,n,50)]
    report=dict(trial=trial,**assignments()[trial],raw_sensor_reconstruction_pass=True,command_stop_replay_pass=True)
    return raw,cameras,report


def test_initial_labels_match_existing_audit_and_moving_motion_uses_current_origin():
    raw,cameras,r=fixture();rows=derive(raw,cameras,r);old=native_horizons(raw,cameras)
    assert len(rows)==8 and all(w['available'] for w in rows)
    for new,previous in zip(rows[0]['targets'],old['targets'],strict=True):
        assert {k:v for k,v in new.items() if k!='in_plan'}=={k:v for k,v in previous.items() if k!='status'}
    np.testing.assert_allclose(rows[4]['targets'][0]['motion'],[.05,.025,0.],atol=1e-12)
    assert rows[4]['decision_ns']==3_800_000_000 and rows[4]['history_observation_indices']==[20,21,22,23]


def test_motion_is_projected_into_current_body_frame():
    raw,cameras,r=fixture();raw['base_pose_world'][:,5:]=[np.sqrt(.5),np.sqrt(.5)]
    rows=derive(raw,cameras,r)
    np.testing.assert_allclose(rows[2]['targets'][0]['motion'],[.025,-.05,0.],atol=1e-12)


def test_all_requested_suffixes_are_exact_with_unknown_zero_padding():
    action=next(c['action'] for c in assignments().values() if c['action']=='left_arc')
    for offset in range(0,40,5):
        blocks,mask=remaining_candidate(action,offset)
        expected=torch.tensor(candidate_commands(action)[offset:])/torch.tensor([.3,1.,.5])
        assert torch.equal(blocks.reshape(40,3)[:40-offset],expected)
        assert mask.sum()==40-offset and not blocks[~mask].any()


def test_contact_keeps_known_event_targets_after_missing_future_and_absent_contexts():
    raw,cameras,r=fixture(1658,1657);rows=derive(raw,cameras,r)
    assert [x['available'] for x in rows]==[True]*4+[False]*4
    final=rows[3];assert final['remaining_ticks']==25
    assert all(t['contact_valid'] and t['contact']==1. and not t['motion_valid'] and not t['future_image_valid'] for t in final['targets'][:5])
    assert all(t['offset_ns']==0 and not t['contact_valid'] and t['contact'] is None for t in final['targets'][5:])
    assert all(x['targets'] is None for x in rows[4:])


def test_unknown_contact_without_an_event_stays_unknown():
    raw,cameras,r=fixture(1658);row=derive(raw,cameras,r)[3]
    assert all(not t['contact_valid'] and t['contact'] is None for t in row['targets'])


def test_moving_materialization_reads_own_past_and_actual_remaining_futures_only():
    raw,cameras,r=fixture();row=derive(raw,cameras,r)[6];calls=[]
    def packet(i):calls.append(i);return policy(i),None,None,None
    sample=materialize(SimpleNamespace(packet=packet),row)
    assert calls==[30,31,32,33,38,43]
    assert sample['inputs']['known_action_valid'].sum()==10
    assert sample['targets']['target_offsets_ns'].tolist()==[500_000_000,1_000_000_000,0,0,0,0,0,0]
    assert sample['targets']['future_valid'].sum()==2 and sample['targets']['motion'][2:].isnan().all()


@pytest.mark.parametrize('fault',['role','past_clock','future_clock','unknown_target','fake_motion','event_reversal'])
def test_misbound_or_fabricated_window_rejected(fault):
    raw,cameras,r=fixture(1658,1657);row=derive(raw,cameras,r)[3]
    def packet(i):return policy(i),None,None,None
    if fault=='role':row['data_role']='changed'
    if fault=='past_clock':row['history_observation_indices'][0]-=1
    if fault=='future_clock':
        raw,cameras,r=fixture();row=derive(raw,cameras,r)[3]
        def packet(i):return policy(i+1 if i>18 else i),None,None,None
    if fault=='unknown_target':row['targets'][-1].update(contact_valid=True,contact=1.)
    if fault=='fake_motion':row['targets'][0]['motion']=[0.,0.,0.]
    if fault=='event_reversal':row['targets'][1]['contact']=0.
    with pytest.raises(ValueError):materialize(SimpleNamespace(packet=packet),row)


def test_causal_readout_counts_preserve_planned_windows_and_role_denominators():
    from scripts.read_go2_geometry_progress_family_causal_v1 import counts
    raw,cameras,r=fixture(1658,1657);rows=[]
    for trial,c in assignments().items():rows.extend(derive(raw,cameras,r|dict(trial=trial,**c)))
    result=counts(rows)
    for role in result.values():
        assert role['planned_windows']==384 and role['available_windows']==192
        assert role['available_initial_windows']==48 and role['available_moving_windows']==144
        assert role['planned_known_horizon_slots']==1728
        assert role['contact_positive_without_motion']==role['contact_positive']
    with pytest.raises(ValueError,match='complete ordered'):counts(rows[:-1])
    rows[0]['data_role']='changed'
    with pytest.raises(ValueError,match='assignments'):counts(rows)

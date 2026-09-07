import hashlib
import json

import pytest
import torch

from lewm import moving_prefix_learning_data_development as data
from lewm.tests.test_relative_gyro_turn_development import initialized,append,packet


def example():
    buffer=initialized(); packets={80:packet(buffer,80)}
    for step in range(81,246):
        append(buffer,step,[0,0,.1])
        if step%5==0: packets[step]=packet(buffer,step)
    context=[packets[s] for s in (80,85,90,95)]; current=1_900_000_000
    targets=[]
    for i in range(8):
        valid=i<6
        targets.append({'horizon_ns':(i+1)*500_000_000,'in_plan':valid,'motion_valid':valid,'contact_valid':valid,
            'contact_by_horizon':False if valid else None,'delta_xy_yaw_current_body':[.1*(i+1),0,0] if valid else None,
            'future_observation_index':120+25*i if valid else None})
    window={'decision_ns':current,'remaining_ticks':30,'targets':targets}
    return window,context,packets


def test_switch_tensor_shapes_plan_and_unknown_tail():
    window,context,packets=example(); calls=[]
    def future(index): calls.append(index); return packets[index]
    value=data.switch_tensors(window,[-.2,0,0],context,future)
    assert value['observation_history']['rgb'].shape==(4,3,96,128)
    assert value['observation_history']['body'].shape==(4,20,63)
    assert value['known_action_valid'].sum()==30
    assert not value['known_action_valid'][6:].any() and not value['known_action_blocks'][6:].any()
    assert value['targets']['future_valid'].sum()==6 and torch.isnan(value['targets']['contact'][6:]).all()
    assert calls==[120,145,170,195,220,245]


def test_future_and_label_interventions_cannot_change_policy_history_or_plan():
    window,context,packets=example()
    original=data.switch_tensors(window,[.2,0,.5],context,packets.__getitem__)
    for index in (120,145,170,195,220,245): packets[index]['image']['rgb'][:]=255
    window['targets'][0]['delta_xy_yaw_current_body'][0]=99.
    changed=data.switch_tensors(window,[.2,0,.5],context,packets.__getitem__)
    for key in original['observation_history']: assert torch.equal(original['observation_history'][key],changed['observation_history'][key])
    assert torch.equal(original['known_action_blocks'],changed['known_action_blocks'])
    assert not torch.equal(original['targets']['future_observations']['rgb'],changed['targets']['future_observations']['rgb'])
    assert changed['targets']['motion'][0,0]==99.


def test_contact_censored_targets_never_read_future_packet():
    window,context,_=example()
    for t in window['targets'][:6]:
        t.update(motion_valid=False,contact_by_horizon=True,delta_xy_yaw_current_body=None,future_observation_index=None)
    def forbidden(index): raise AssertionError('unobserved future opened')
    value=data.switch_tensors(window,[0,0,0],context,forbidden)
    assert not value['targets']['future_valid'].any() and value['targets']['contact'][:6].sum()==6


@pytest.mark.parametrize('fault',['future_clock','context_privilege','future_identity','unknown_contact','nonboolean','nonfinite','duration'])
def test_invalid_inputs_and_targets_rejected(fault):
    window,context,packets=example()
    if fault=='future_clock': packets[120]['image']['measured_ns']+=1
    if fault=='context_privilege': context[0]['world_pose']=[0]*7
    if fault=='future_identity': packets[120]['sensor_state']['identity']=(1,0,0)
    if fault=='unknown_contact': window['targets'][7]['contact_by_horizon']=False
    if fault=='nonboolean': window['targets'][0]['contact_by_horizon']=0
    if fault=='nonfinite': window['targets'][0]['delta_xy_yaw_current_body'][0]=float('nan')
    if fault=='duration': window['remaining_ticks']=40
    with pytest.raises(ValueError): data.switch_tensors(window,[0,0,0],context,packets.__getitem__)


def test_unapproved_roots_and_roles_rejected_before_io(tmp_path):
    with pytest.raises(ValueError,match='exact'): data.AuditedMovingPrefixDataset(tmp_path,'train')
    with pytest.raises(ValueError,match='role'): data.AuditedMovingPrefixDataset(data.OUTPUT,'test')


def test_partial_audit_cannot_qualify_loader(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'OUTPUT',tmp_path)
    launch=tmp_path/'launch.json'; launch.write_text('{}')
    monkeypatch.setattr(data,'LAUNCH_SHA',hashlib.sha256(launch.read_bytes()).hexdigest())
    (tmp_path/'result.json').write_text(json.dumps({'status':'COMPLETE','completed_trials':384,'planned_trials':384}))
    (tmp_path/'raw_artifact_audit.json').write_text(json.dumps({'status':'PASS','audited_trials':16}))
    with pytest.raises(ValueError,match='full raw audit'): data.AuditedMovingPrefixDataset(tmp_path,'train')


def test_policy_leaf_rejects_replaced_or_symlinked_bytes(tmp_path):
    path=tmp_path/'policy.json'; path.write_text('{}'); digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert data.checked_policy_leaf(tmp_path,path.name,digest)==path
    path.write_text('[]')
    with pytest.raises(ValueError): data.checked_policy_leaf(tmp_path,path.name,digest)
    link=tmp_path/'linked.json'; link.symlink_to(path)
    with pytest.raises(ValueError): data.checked_policy_leaf(tmp_path,link.name,hashlib.sha256(path.read_bytes()).hexdigest())

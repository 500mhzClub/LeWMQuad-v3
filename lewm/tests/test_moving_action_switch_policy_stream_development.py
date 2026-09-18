from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.moving_action_switch_family_development import assignments
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView
from lewm.moving_action_switch_learning_sample_development import inference_inputs
from lewm.tests.test_moving_action_switch_learning_development import fixture
from lewm.tests.test_independent_pulse_context_development import policy
from scripts import moving_action_switch_policy_stream_development as mod
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def population():
    _,template,_=fixture()
    return [dict(deepcopy(template),trial=t,**c,prefix=dict(complete=True),outcome=dict(branch_available=True)) for t,c in assignments().items()]


def test_balanced_complete_training_only_schedule():
    view=MovingActionSwitchView(population());a=view.schedule(updates=1200,batch_size=6,seed=2026091401)
    counts=Counter(i for b in a['batches'] for i in b)
    assert set(counts)==set(view.indices('train')) and set(counts.values())=={100}
    for b in a['batches']:
        assert len({(view.reports[i]['cluster'],view.reports[i]['prefix_action']) for i in b})==1
        assert len({view.reports[i]['suffix_action'] for i in b})==6
    assert a==view.schedule(updates=1200,batch_size=6,seed=2026091401)
    assert a['batches']!=view.schedule(updates=1200,batch_size=6,seed=2026091402)['batches']
    view.reports[view.indices('train')[0]]['outcome']['branch_available']=False
    with pytest.raises(ValueError,match='disappear'):view.schedule(updates=1200,batch_size=6,seed=1)


def stream(monkeypatch,tmp_path):
    view=MovingActionSwitchView(population());i=view.indices('train')[0];row=view.reports[i]
    inputs=inference_inputs(SimpleNamespace(packet=lambda i:(policy(i),)),row)
    index=[dict(trial=r['trial'],materialized=False) for r in view.reports]
    index[i].update(materialized=True,history_sha256={k:fingerprint(v.numpy()) for k,v in inputs['observation_history'].items()},
        known_action_sha256=fingerprint(inputs['known_action_blocks'].numpy()),
        known_action_valid_sha256=fingerprint(inputs['known_action_valid'].numpy()))
    leaves=mod.policy_leaves(row,include_future=True);bindings={row['trial']+'/'+n:'0'*64 for n in leaves}
    reads=[];checks=[]
    monkeypatch.setattr(mod,'INPUT',tmp_path);monkeypatch.setattr(mod,'validate_root',lambda p:p)
    monkeypatch.setattr(mod,'verify_artifacts',lambda p,b:checks.append(set(b)))
    def load(p,i):reads.append(i);return policy(i)
    monkeypatch.setattr(mod,'load_route_observation',load)
    return mod.MovingActionSwitchPolicyStream(view,output=tmp_path,bindings=bindings,tensor_index=index),i,reads,checks


def test_inference_never_touches_future_labels_or_images(monkeypatch,tmp_path):
    s,i,reads,checks=stream(monkeypatch,tmp_path)
    class ForbiddenFuture:
        def __getitem__(self,key):pytest.fail('inference touched target labels')
    s.view.reports[i]['targets']=ForbiddenFuture()
    inputs=s.inference_batch([i],role='train')
    assert reads==[10,11,12,13] and len(checks)==2
    assert set(inputs)=={'observation_history','known_action_blocks','known_action_valid'}
    assert all(not any('native' in n or 'depth' in n for n in c) for c in checks)


def test_cache_returns_private_batches_and_transfer_rejection_latches(monkeypatch,tmp_path):
    s,i,reads,checks=stream(monkeypatch,tmp_path);a=s.training_batch([i])
    assert reads==[10,11,12,13,18] and len(checks)==2
    a['inputs']['observation_history']['rgb'].fill_(123.)
    b=s.training_batch([i]);assert not (b['inputs']['observation_history']['rgb']==123.).any()
    assert len(reads)==5 and b['targets']['contact'].sum()==6
    with pytest.raises(ValueError,match='role'):s.training_batch([s.view.indices('geometry_transfer')[0]])
    assert s.failed
    with pytest.raises(ValueError,match='latched'):s.inference_batch([i],role='train')


def test_missing_binding_and_tensor_mismatch_fail(monkeypatch,tmp_path):
    s,i,reads,_=stream(monkeypatch,tmp_path);s.bindings={}
    with pytest.raises(ValueError,match='bound'):s.inference_batch([i],role='train')
    assert reads==[]
    s,i,_,_=stream(monkeypatch,tmp_path);s.index[i]['known_action_sha256']='1'*64
    with pytest.raises(ValueError,match='differ'):s.inference_batch([i],role='train')

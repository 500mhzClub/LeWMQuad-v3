"""Inference cannot materialize future data; role and cache boundaries are tested."""
from types import SimpleNamespace
import pytest
from lewm.geometry_progress_layout_family_development import assignments
from lewm.geometry_progress_family_causal_windows_development import derive,materialize
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.tests.test_geometry_progress_family_learning_view_development import windows
from lewm.tests.test_geometry_progress_family_causal_windows_development import fixture
from lewm.tests.test_independent_pulse_context_development import policy
from scripts import geometry_progress_family_policy_stream_development as mod
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def stream(monkeypatch,tmp_path):
    rows=windows();trial=next(t for t,c in assignments().items() if c['data_role']=='train')
    raw,cameras,r=fixture(1658,1657);r.update(trial=trial,**assignments()[trial]);actual=derive(raw,cameras,r)
    selected=next(i for i,w in enumerate(rows) if w['trial']==trial)
    rows[selected:selected+8]=actual
    sample=materialize(SimpleNamespace(packet=lambda i:(policy(i),)),actual[0]);inputs=sample['inputs']
    index=[dict(window_id=w['window_id'],materialized=False) for w in rows]
    index[selected].update(materialized=True,history_sha256={k:fingerprint(v.numpy()) for k,v in inputs['observation_history'].items()},
        known_action_sha256=fingerprint(inputs['known_action_blocks'].numpy()),
        known_action_valid_sha256=fingerprint(inputs['known_action_valid'].numpy()))
    leaves=mod.policy_leaves(actual[0],include_future=True);bindings={trial+'/'+n:'0'*64 for n in leaves}
    reads=[];checks=[]
    monkeypatch.setattr(mod,'INPUT',tmp_path);monkeypatch.setattr(mod,'validate_root',lambda p:p)
    monkeypatch.setattr(mod,'verify_artifacts',lambda p,b:checks.append(set(b)))
    def load(p,i):reads.append(i);return policy(i)
    monkeypatch.setattr(mod,'load_route_observation',load)
    return mod.FamilyPolicyStream(FamilyWindowView(rows),output=tmp_path,bindings=bindings,tensor_index=index),selected,reads,checks


def test_inference_never_visits_future_targets_or_images(monkeypatch,tmp_path):
    s,i,reads,checks=stream(monkeypatch,tmp_path)
    class ForbiddenFuture:
        def __iter__(self):pytest.fail('inference touched future targets')
    s.view.windows[i]['targets']=ForbiddenFuture()
    inputs=s.inference_batch([i],role='train')
    assert reads==[0,1,2,3] and len(checks)==2
    assert set(inputs)=={'observation_history','known_action_blocks','known_action_valid'}
    assert all(not any('native' in n or 'depth' in n for n in c) for c in checks)


def test_training_cache_admission_is_bound_and_batches_cannot_mutate_cache(monkeypatch,tmp_path):
    s,i,reads,checks=stream(monkeypatch,tmp_path);a=s.training_batch([i])
    assert reads==[0,1,2,3,8,13,18] and len(checks)==2
    a['inputs']['observation_history']['rgb'].fill_(123.)
    b=s.training_batch([i]);assert not (b['inputs']['observation_history']['rgb']==123.).any()
    assert len(reads)==7 and len(checks)==2
    assert b['targets']['contact'].sum()==5


def test_transfer_indices_cannot_enter_training_and_failure_latches(monkeypatch,tmp_path):
    s,i,reads,_=stream(monkeypatch,tmp_path)
    transfer=s.view.indices('geometry_transfer',initial_only=True)[0]
    with pytest.raises(ValueError,match='role'):s.training_batch([transfer])
    assert reads==[] and s.failed
    with pytest.raises(ValueError,match='latched'):s.inference_batch([i],role='train')


def test_tensor_identity_mismatch_is_rejected(monkeypatch,tmp_path):
    s,i,_,_=stream(monkeypatch,tmp_path);s.index[i]['known_action_sha256']='1'*64
    with pytest.raises(ValueError,match='differ'):s.inference_batch([i],role='train')
    assert s.failed


def test_missing_policy_binding_rejected_before_load(monkeypatch,tmp_path):
    s,i,reads,_=stream(monkeypatch,tmp_path);s.bindings={}
    with pytest.raises(ValueError,match='bound'):s.inference_batch([i],role='train')
    assert reads==[]

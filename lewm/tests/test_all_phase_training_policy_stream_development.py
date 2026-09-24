"""Causal reader isolation, exact old plans and bounded immutable cache outputs."""
from copy import deepcopy
import torch
import pytest
from lewm.tests.test_all_phase_training_targets_development import fixture
from lewm.all_phase_training_targets_development import expand_trial
from lewm.observation_horizon_plan_development import plan as old_plan
from scripts import all_phase_training_policy_stream_development as stream
from lewm import causal_subtrajectory_learning_development as history_module
from lewm import observation_horizon_sample_development as target_module


def data(source='family'):
    old,raw,cameras=fixture(source)
    return expand_trial(old,raw,cameras)


def bindings():
    names=['policy_observations.json','policy_histories.npz']+[
        f'rgb_{i:04d}.png' for i in range(60)]
    return {s:{'fixture/'+n:'a'*64 for n in names} for s in ('family','switch')}


def mocks(monkeypatch,*,clock_at=None,identity_at=None):
    reads=[];checks=[]
    def load(directory,index):
        reads.append(index);now=1_500_000_000+100_000_000*index
        if index==clock_at:now+=1
        return dict(image=dict(measured_ns=now,index=index),
            sensor_state=dict(decision_ns=now,identity=(1,0,0) if index==identity_at else (0,0,0)))
    def tensors(packet):
        return {k:torch.tensor([float(packet['image']['index'])]) for k in ('rgb','body','control')}
    monkeypatch.setattr(stream,'load_route_observation',load)
    monkeypatch.setattr(stream,'verify_artifacts',lambda root,ids:checks.append((root,sorted(ids))))
    monkeypatch.setattr(history_module,'observation_tensors',tensors)
    monkeypatch.setattr(target_module,'observation_tensors',tensors)
    return reads,checks


@pytest.mark.parametrize('source',['family','switch'])
def test_every_offset_and_old_plan_normalization(source):
    rows=data(source)
    for offset,row in enumerate(rows):
        blocks,valid=stream.plan(row)
        assert blocks.shape==(8,1,3) and valid.shape==(8,1)
        assert valid.sum()==min(8,40-offset) and torch.count_nonzero(blocks[~valid])==0
        if offset%5==0:
            old,mask=old_plan(row['action'],offset_ticks=offset)
            assert torch.equal(blocks,old) and torch.equal(valid,mask)


@pytest.mark.parametrize('source',['family','switch'])
@pytest.mark.parametrize('offset',[0,1,2,3,4,35,36,39])
def test_actual_clock_checked_past_and_future_reader_scopes(monkeypatch,source,offset):
    reads,checks=mocks(monkeypatch);rows=data(source)
    obj=stream.AllPhaseTrainingStream(rows,bindings(),maximum_cache_bytes=0)
    row=rows[offset];frame=(3 if source=='family' else 13)+offset
    inputs=obj.materialize(offset,training=False)
    assert reads==list(range(frame-3,frame+1)) and obj.last_access['future_packet_indices']==[]
    assert set(inputs)=={'observation_history','known_action_blocks','known_action_valid'}
    reads.clear();sample=obj.materialize(offset,training=True)
    future=list(range(frame+1,frame+1+min(8,40-offset)))
    assert reads==list(range(frame-3,frame+1))+future
    assert obj.last_access['future_packet_indices']==future
    assert set(sample)=={'inputs','targets'}
    assert torch.equal(sample['inputs']['observation_history']['rgb'],inputs['observation_history']['rgb'])
    assert sample['targets']['future_valid'].sum()==len(future)
    assert len(checks)==4 and all('/physics_trace.npz' not in n for _,names in checks for n in names)


def test_inference_never_inspects_target_labels_or_future_indices(monkeypatch):
    reads,_=mocks(monkeypatch);obj=stream.AllPhaseTrainingStream(data(),bindings())
    class Forbidden:
        def __iter__(self):pytest.fail('inference inspected targets')
        def __getitem__(self,key):pytest.fail('inference inspected targets')
    obj.rows[0]['targets']=Forbidden()
    obj.materialize(0,training=False)
    assert reads==[0,1,2,3]


@pytest.mark.parametrize('fault',['past_clock','future_clock','future_identity','future_index',
    'past_index','command','role','unavailable','missing_binding','motion_mask','receipt','native_label_scope'])
def test_invalid_scope_clock_role_or_binding_latches_failure(monkeypatch,fault):
    reads,_=mocks(monkeypatch,clock_at=3 if fault=='past_clock' else 4 if fault=='future_clock' else None,
        identity_at=4 if fault=='future_identity' else None)
    obj=stream.AllPhaseTrainingStream(data(),bindings());row=obj.rows[0]
    if fault=='future_index':row['targets'][0]['future_observation_index']=20
    elif fault=='past_index':row['history_observation_indices'][-1]=4
    elif fault=='command':row['known_commands'][0][0]=.1
    elif fault=='role':row['data_role']='geometry_transfer'
    elif fault=='unavailable':row['available']=False
    elif fault=='missing_binding':del obj.bindings['family']['fixture/rgb_0000.png']
    elif fault=='motion_mask':row['targets'][0]['contact_valid']=False
    elif fault=='receipt':row['observation_horizon_receipt']['departure_ns']+=1
    elif fault=='native_label_scope':row['native_labels_are_target_only']=False
    with pytest.raises(ValueError):obj.materialize(0,training=True)
    assert obj.failed
    with pytest.raises(ValueError,match='latched'):obj.materialize(1,training=False)


def test_cache_budget_eviction_and_fresh_batch_mutation_isolation(monkeypatch):
    reads,_=mocks(monkeypatch);rows=data();b=bindings()
    probe=stream.AllPhaseTrainingStream(rows,b,maximum_cache_bytes=0)
    size=stream.tensor_bytes(probe.materialize(0,training=True))
    obj=stream.AllPhaseTrainingStream(rows,b,maximum_cache_bytes=size)
    first=obj.training_batch([0]);expected=first['targets']['future_observations']['rgb'].clone()
    first['targets']['future_observations']['rgb'].fill_(999)
    read_count=len(reads);second=obj.training_batch([0])
    assert len(reads)==read_count and torch.equal(second['targets']['future_observations']['rgb'],expected)
    obj.training_batch([1]);assert list(obj._cache)==[1]
    assert obj.cache_bytes<=size and obj.maximum_observed_cache_bytes<=size
    count=len(reads);obj.training_batch([0]);assert len(reads)>count


def test_cache_hit_still_rejects_changed_policy_artifact(monkeypatch):
    mocks(monkeypatch);obj=stream.AllPhaseTrainingStream(data(),bindings())
    obj.training_batch([0])
    def changed(*args):raise ValueError('policy artifact changed')
    monkeypatch.setattr(stream,'verify_artifacts',changed)
    with pytest.raises(ValueError,match='artifact changed'):obj.training_batch([0])
    assert obj.failed


def test_transfer_batches_are_not_materialized_by_training_stream(monkeypatch):
    reads,_=mocks(monkeypatch);obj=stream.AllPhaseTrainingStream(data(),bindings())
    with pytest.raises(ValueError,match='original transfer stream'):obj.inference_batch([0],role='geometry_transfer')
    assert reads==[]

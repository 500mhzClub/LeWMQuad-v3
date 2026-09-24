from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from lewm.augmented_family_switch_view_development import AugmentedFamilySwitchView
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView
from lewm.geometry_progress_family_causal_windows_development import remaining_candidate
from lewm.geometry_progress_pilot_development import timed_candidate
from lewm.observation_horizon_view_development import ObservationHorizonView
from lewm.observation_horizon_sample_development import inference_inputs,materialize_training
from lewm.tests.test_geometry_progress_family_learning_view_development import windows
from lewm.tests.test_moving_action_switch_policy_stream_development import population
from lewm.tests.test_independent_pulse_context_development import policy
from scripts.augmented_family_switch_stream_development import AugmentedFamilySwitchStream
from scripts import observation_horizon_policy_stream_development as mod


def fixture():
    old=windows()
    for r in old:
        if r['available']:
            r['targets']=[]
            for h in range(1,9):
                active=5*h<=40-r['offset_ticks']
                r['targets'].append(dict(in_plan=active,offset_ns=h*500_000_000 if active else 0,
                    motion_valid=active,motion=[0.,0.,0.] if active else None,contact_valid=active,contact=0. if active else None,
                    future_image_valid=active,future_observation_index=3+r['offset_ticks']+5*h if active else None))
    original=AugmentedFamilySwitchView(FamilyWindowView(old),MovingActionSwitchView(population()));rows=[]
    for r in original.rows:
        frame=3+r['offset_ticks'] if r['source']=='family' else 13
        known=min(8,40-r['offset_ticks']) if r['source']=='family' else 8
        targets=None
        if r['available']:
            targets=[]
            for h in range(1,9):
                if h<=known:
                    t=deepcopy(r['targets'][0]);t.update(in_plan=True,offset_ns=h*100_000_000)
                    if t['future_image_valid']:t['future_observation_index']=frame+h
                else:t=dict(in_plan=False,offset_ns=0,motion_valid=False,motion=None,contact_valid=False,contact=None,
                    future_image_valid=False,future_observation_index=None)
                targets.append(t)
        rows.append(deepcopy(r)|dict(targets=targets,shared_half_second_native_target_exact=r['available'],
            observation_horizon_receipt=dict(target_only=True,departure_tick=frame,departure_ns=1_500_000_000+frame*100_000_000,
                history_observation_indices=list(range(frame-3,frame+1)),target_cadence_ns=100_000_000,
                maximum_horizon_ns=800_000_000,available=r['available'],reason=None if r['available'] else r['reason'])))
    return original,rows


def old_inputs(row):
    blocks,valid=remaining_candidate(row['action'],row['offset_ticks']) if row['source']=='family' else timed_candidate(row['action'])
    history={k:torch.zeros(shape) for k,shape in dict(rgb=(4,3,96,128),body=(4,20,63),control=(4,15,7)).items()}
    return dict(observation_history=history,known_action_blocks=blocks,known_action_valid=valid)


def test_exact_original_roles_and_schedule_are_preserved():
    original,rows=fixture();view=ObservationHorizonView(original,rows)
    assert view.indices('train')==original.indices('train')
    assert view.schedule(updates=1200,batch_size=6,seed=2026091401)==original.schedule(updates=1200,batch_size=6,seed=2026091401)
    bad=deepcopy(rows);bad[0]['data_role']='geometry_transfer' if bad[0]['data_role']=='train' else 'train'
    with pytest.raises(ValueError,match='context'):ObservationHorizonView(original,bad)
    i=view.indices('train')[0];bad=deepcopy(rows);bad[i]['targets'][0]['offset_ns']=500_000_000
    with pytest.raises(ValueError,match='offsets'):ObservationHorizonView(original,bad)


def test_inference_never_opens_target_labels_or_future_reader(monkeypatch):
    original,rows=fixture();stub=object.__new__(AugmentedFamilySwitchStream);stub.view=original;calls=[]
    def batch(indices,*,role):
        calls.append((indices,role));inputs=old_inputs(original.rows[indices[0]])
        def expand(value):return {k:expand(v) for k,v in value.items()} if isinstance(value,dict) else value[None]
        return expand(inputs)
    stub.inference_batch=batch
    s=mod.ObservationHorizonPolicyStream(stub,rows);i=s.view.indices('geometry_transfer',source='switch')[0]
    class Forbidden:
        def __iter__(self):pytest.fail('inference touched future labels')
        def __getitem__(self,key):pytest.fail('inference touched future labels')
    s.view.rows[i]['targets']=Forbidden()
    monkeypatch.setattr(mod,'_PolicyReader',lambda *a,**k:pytest.fail('inference constructed future reader'))
    result=s.inference_batch([i],role='geometry_transfer')
    assert result['known_action_blocks'].shape==(1,8,1,3) and calls==[([i],'geometry_transfer')]
    with pytest.raises(ValueError,match='role'):s.training_batch([i])
    assert len(calls)==1 and s.failed


def test_only_actual_short_future_boundaries_enter_training():
    original,rows=fixture();i=original.indices('train',source='switch')[0];row=rows[i]
    inputs=inference_inputs(old_inputs(original.rows[i]),row);calls=[]
    def packet(index):calls.append(index);return (policy(index),)
    sample=materialize_training(SimpleNamespace(packet=packet),row,inputs)
    assert calls==list(range(14,22))
    assert sample['targets']['target_offsets_ns'].tolist()==list(range(100_000_000,800_000_001,100_000_000))
    bad=deepcopy(row);bad['targets'][0]['future_observation_index']=18
    with pytest.raises(ValueError,match='boundary'):materialize_training(SimpleNamespace(packet=packet),bad,inputs)

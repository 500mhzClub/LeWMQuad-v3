"""Synthetic integration of the new action bank with the existing JEPA model."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from lewm.geometry_progress_pilot_development import assignments
from lewm.geometry_progress_learning_sample_development import materialize
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.tests.test_independent_pulse_context_development import policy


def fixture():
    trial=next(iter(assignments()));calls=[]
    def packet(i):
        calls.append(i);return (policy(i),None,None,None)
    row=dict(trial=trial,**assignments()[trial],raw_sensor_reconstruction_pass=True,command_stop_replay_pass=True,
        targets=dict(target_only=True,departure_tick=3,departure_ns=1_800_000_000,
            history_observation_indices=[0,1,2,3],targets=[dict(offset_ns=i*500_000_000,
                motion_valid=i<=2,motion=[.02*i,.01,0.] if i<=2 else None,
                contact_valid=True,contact=0. if i<=2 else 1.,future_image_valid=i==1,
                future_observation_index=8 if i==1 else None) for i in range(1,9)]))
    return SimpleNamespace(packet=packet),row,calls


def test_actual_packet_masks_feed_existing_encoder_and_cumulative_rollout_without_fitting():
    reader,row,calls=fixture();sample=materialize(reader,row)
    assert calls==[0,1,2,3,8]
    assert set(sample['inputs'])=={'observation_history','known_action_blocks','known_action_valid'}
    t=sample['targets'];assert t['motion_valid'].sum()==2 and t['future_valid'].sum()==1
    assert t['contact_valid'].all() and t['contact'].sum()==6
    assert t['motion'][2:].isnan().all()
    batch=stack_samples([sample])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260909);model=CumulativePulseRGBBodyJEPA(32).eval()
    with torch.no_grad():out=model(**batch['inputs'])
    assert out['direct_outcomes'].shape==out['rollout_outcomes'].shape==(1,8,5)
    assert torch.isfinite(out['rollout_outcomes']).all()
    assert torch.equal(out['target_offsets_ns'],batch['targets']['target_offsets_ns'])
    # Outcomes and episode/geometry identifiers never enter model forward.
    assert not {'trial','geometry','action','outcome','native_pose','targets'} & set(batch['inputs'])


@pytest.mark.parametrize('fault',['future_clock','motion_filled','event_reversal','future_index','action_assignment'])
def test_fabricated_or_misbound_targets_rejected(fault):
    reader,row,_=fixture()
    if fault=='future_clock':reader.packet=lambda i:(policy(i+1) if i==8 else policy(i),None,None,None)
    if fault=='motion_filled':row['targets']['targets'][3]['motion']=[0.,0.,0.]
    if fault=='event_reversal':row['targets']['targets'][4]['contact']=0.
    if fault=='future_index':row['targets']['targets'][0]['future_observation_index']=9
    if fault=='action_assignment':row['action']='hold'
    with pytest.raises(ValueError):materialize(reader,row)

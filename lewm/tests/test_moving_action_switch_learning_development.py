from types import SimpleNamespace
import pytest
import torch
from lewm.moving_action_switch_family_development import assignments
from lewm.moving_action_switch_learning_sample_development import inference_inputs,materialize_training
from lewm.tests.test_independent_pulse_context_development import policy
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
from lewm.pulse_timed_dataset_development import stack_samples


def fixture(role='train'):
    trial=next(t for t,c in assignments().items() if c['data_role']==role);calls=[]
    def packet(i):calls.append(i);return (policy(i),)
    row=dict(trial=trial,**assignments()[trial],raw_sensor_reconstruction_pass=True,command_stop_replay_pass=True,
        targets=dict(target_only=True,departure_tick=13,departure_ns=2_800_000_000,history_observation_indices=[10,11,12,13],
            targets=[dict(offset_ns=i*500_000_000,motion_valid=i<=2,motion=[.02*i,.01,0.] if i<=2 else None,
                contact_valid=True,contact=0. if i<=2 else 1.,future_image_valid=i==1,
                future_observation_index=18 if i==1 else None) for i in range(1,9)]))
    return SimpleNamespace(packet=packet),row,calls


def test_past_only_inference_ignores_target_object():
    reader,row,calls=fixture('geometry_transfer')
    class ForbiddenTargets:
        def __getitem__(self,key):pytest.fail('inference touched targets')
    row['targets']=ForbiddenTargets();inputs=inference_inputs(reader,row)
    assert calls==[10,11,12,13]
    assert set(inputs)=={'observation_history','known_action_blocks','known_action_valid'}


def test_training_masks_and_existing_model_contract():
    reader,row,calls=fixture();sample=materialize_training(reader,row)
    assert calls==[10,11,12,13,18]
    assert sample['targets']['contact'].sum()==6 and sample['targets']['motion'][2:].isnan().all()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026091400);model=CumulativePulseRGBBodyJEPA(32).eval()
    with torch.inference_mode():out=model(**stack_samples([sample])['inputs'])
    assert out['direct_outcomes'].shape==out['rollout_outcomes'].shape==(1,8,5)
    assert torch.isfinite(out['rollout_outcomes']).all()
    assert torch.equal(out['target_offsets_ns'][0],sample['targets']['target_offsets_ns'])


def test_transfer_cannot_read_future_training_images():
    reader,row,calls=fixture('geometry_transfer')
    with pytest.raises(ValueError,match='training-only'):materialize_training(reader,row)
    assert calls==[]


@pytest.mark.parametrize('fault',['assignment','branch','future_index','censored_image','contact_reversal'])
def test_misbound_or_fabricated_targets_rejected(fault):
    reader,row,_=fixture()
    if fault=='assignment':row['cluster']='other'
    if fault=='branch':row['targets']['departure_tick']=3
    if fault=='future_index':row['targets']['targets'][0]['future_observation_index']=19
    if fault=='censored_image':row['targets']['targets'][2].update(future_image_valid=True,future_observation_index=28)
    if fault=='contact_reversal':row['targets']['targets'][4]['contact']=0.
    with pytest.raises(ValueError):materialize_training(reader,row)

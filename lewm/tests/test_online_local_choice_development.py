import copy
import hashlib

import numpy as np
import pytest
import torch

from lewm.causal_sensor_state import SensorContractError
from lewm.online_local_choice_development import OnlineLocalChoice,_read_bound,rank_ensemble
from lewm.simulated_body_observation_development import BodyObservationBuffer,IdealBodySensor,JOINT_NAMES


def packet(identity=(0,0,0)):
    buffer=BodyObservationBuffer(identity); sensor=IdealBodySensor()
    for tick in range(1,81):
        ns=tick*20_000_000
        values=sensor.sample(measured_ns=ns,quaternion_xyzw=[0,0,0,1],velocity_world=[0,0,0],
            angular_velocity_world=[0,0,0],joint_position=np.zeros(12),joint_velocity=np.zeros(12),joint_names=JOINT_NAMES)
        buffer.append_sensors(values,ns)
        if tick%5==0: buffer.append_applied_command([.25,0,0],ns)
    return buffer.packet(np.zeros((480,640,3),dtype=np.uint8),1_600_000_000)


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.marker=torch.nn.Parameter(torch.tensor(1.)); self.seen=[]
    def forward(self,observation,plans):
        self.seen.append((observation,plans))
        values=torch.zeros(5,8,5); values[...,3]=1; values[...,4]=-30; values[1,:,0]=.8
        return {'rollout_outcomes':values}


def policy(condition='supervised_rollout'):
    result=OnlineLocalChoice(condition,[FixedModel() for _ in range(3)],[{'test_identity':i} for i in range(3)])
    result.begin_episode((0,0,0)); return result


def test_five_prospective_plans_and_release_are_causal():
    adapter=policy(); result=adapter.select(packet(),[.8,0],now_ns=1_600_000_000)
    assert result['selected_action_index']==1
    assert np.asarray(result['requested_command_tape']).shape==(45,3)
    assert np.all(np.asarray(result['requested_command_tape'])[40:]==0)
    # First release slews .3 -> .05 -> 0, not an instantaneous command reset.
    assert result['expected_applied_command_tape'][40][0]==pytest.approx(.05)
    assert result['expected_applied_command_tape'][41]==[0,0,0]
    assert np.asarray(result['candidate_applied_plans']).shape==(5,40,3)
    for model in adapter.models:
        observation,plans=model.seen[0]
        assert set(observation)=={'rgb','body','control'} and plans.shape==(5,8,5,3)
        assert not model.training and not model.marker.requires_grad
    assert result['adapter_ms']>=result['inference_ms']>=0


def test_ensemble_averages_probabilities_not_logits_and_ties_are_stable():
    predictions=np.zeros((3,5,8,5)); predictions[...,3]=1
    predictions[0,:,:,4]=10; predictions[1:,:,:,4]=-1
    result=rank_ensemble(predictions,[0,0])
    expected=(1/(1+np.exp(-10))+2/(1+np.exp(1)))/3
    assert result['mean_contact_probability'][0][0]==pytest.approx(expected)
    assert result['selected_action_index']==0


@pytest.mark.parametrize('fault',['privilege','stale_packet','stale_image','missing_latest','reset','future','missing_history','off_clock'])
def test_contract_failures_do_not_issue_a_motion_tape(fault):
    adapter=policy(); value=packet(); now=1_600_000_000
    if fault=='privilege': value['simulator_pose']=[0]*7
    elif fault=='stale_packet': now+=100_000_000
    elif fault=='stale_image': value['image']['measured_ns']-=100_000_000; value['sensor_state']['image_ns']-=100_000_000
    elif fault=='missing_latest': value['sensor_state']['sensed']['gyro']['valid'][-1]=False; value['sensor_state']['sensed']['gyro']['values'][-1]=0
    elif fault=='reset': value['sensor_state']['identity']=(0,0,1)
    elif fault=='future': value['sensor_state']['sensed']['joints']['available_ns'][-1]+=1
    elif fault=='missing_history': value['sensor_state']['control']['applied_command']['valid'][0]=False; value['sensor_state']['control']['applied_command']['values'][0]=0
    elif fault=='off_clock': now+=1
    with pytest.raises((SensorContractError,ValueError)): adapter.select(value,[.8,0],now_ns=now)
    assert not adapter._selected


def test_one_shot_and_reset_identity():
    adapter=policy(); value=packet()
    adapter.select(value,[.8,0],now_ns=1_600_000_000)
    with pytest.raises(SensorContractError,match='one local choice'): adapter.select(value,[.8,0],now_ns=1_600_000_000)
    with pytest.raises(SensorContractError,match='already used'): adapter.begin_episode((0,0,0))
    adapter.begin_episode((0,0,1))
    value['sensor_state']['identity']=(0,0,1)
    assert adapter.select(value,[.8,0],now_ns=1_600_000_000)['episode_identity']==[0,0,1]


def test_always_stop_keeps_same_input_contract():
    adapter=OnlineLocalChoice.from_completed_study('always_stop'); adapter.begin_episode((0,0,0))
    result=adapter.select(packet(),[0,.8],now_ns=1_600_000_000)
    assert result['selected_action_index']==0 and result['member_predictions']==[]
    assert np.all(np.asarray(result['requested_command_tape'])==0)


@pytest.mark.parametrize('intent',[[1,0],[np.nan,0],[0],[0,0,0]])
def test_invalid_intents(intent):
    with pytest.raises(ValueError,match='intent'): policy().select(packet(),intent,now_ns=1_600_000_000)


def test_binding_rejects_wrong_bytes_and_symlinks(tmp_path):
    path=tmp_path/'synthetic.bin'; path.write_bytes(b'synthetic')
    expected=hashlib.sha256(b'synthetic').hexdigest()
    assert _read_bound(path,expected)==b'synthetic'
    with pytest.raises(ValueError,match='binding'): _read_bound(path,'0'*64)
    link=tmp_path/'link'; link.symlink_to(path)
    with pytest.raises(ValueError,match='symlink'): _read_bound(link,expected)

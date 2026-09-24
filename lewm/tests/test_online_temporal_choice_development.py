import numpy as np
import pytest
import torch
from torch import nn

from lewm.causal_sensor_state import SensorContractError
from lewm.online_temporal_choice_development import OnlineTemporalChoice,METHODS,rank_half_second_ensemble,_read_bound
from lewm.tests.test_relative_gyro_turn_development import initialized,append,packet


class StubModel(nn.Module):
    def encode_history(self,history):
        assert history['rgb'].shape==(5,4,3,96,128)
        return torch.zeros(5,128),None

    def direct(self,z,blocks):
        assert torch.count_nonzero(blocks[:,1:])==0
        result=torch.zeros(5,8,5); result[...,3]=1.; result[...,4]=-30.
        result[:,:,0]=blocks[:,:,0,0]*.15
        return result

    def forward(self,history,blocks,valid):
        assert valid[:,0].all() and not valid[:,1:].any()
        z,_=self.encode_history(history)
        return {'rollout_outcomes':self.direct(z,blocks)}


def stream():
    buffer=initialized([0,0,.1]); rows=[packet(buffer,80)]
    for tick in range(1,14):
        for step in range(80+(tick-1)*5+1,80+tick*5+1): append(buffer,step,[0,0,.1])
        value=packet(buffer,80+tick*5); value['image']['rgb'][:]=tick; rows.append(value)
    return rows


def adapter(method='supervised_direct'):
    count=0 if method=='always_stop' else 3
    value=OnlineTemporalChoice(method,[StubModel() for _ in range(count)],[{'seed':i} for i in range(count)])
    value.begin_episode((0,0,0)); return value


def begin(value,rows):
    for p in rows[:4]: value.observe(p,now_ns=p['image']['measured_ns'])
    ns=rows[3]['image']['measured_ns']; value.begin_control([.8,0],now_ns=ns)
    return ns


@pytest.mark.parametrize('method',list(METHODS))
def test_all_methods_emit_only_five_causal_commands_and_keep_head_identity(method):
    rows=stream(); value=adapter(method); ns=begin(value,rows)
    result=value.select(now_ns=ns)
    assert result['selected_action_index']==(0 if method=='always_stop' else 1)
    assert result['branch_ticks']==5 and result['horizon_ns']==500_000_000
    assert len(result['requested_command_tape'])==len(result['expected_applied_command_tape'])==5
    assert result['training_condition']==METHODS[method][0] and result['inference_head']==METHODS[method][1]
    assert [p['measured_ns'] for p in result['input_images']]==[r['image']['measured_ns'] for r in rows[:4]]
    if method!='always_stop': assert result['expected_applied_command_tape'][0]==pytest.approx([.25,0,0])


def test_repeated_choices_require_every_packet_and_transport_fixed_initial_direction():
    rows=stream(); value=adapter(); ns=begin(value,rows)
    first=value.select(now_ns=ns)
    for p in rows[4:9]: value.observe(p,now_ns=p['image']['measured_ns'])
    second=value.select(now_ns=ns+500_000_000)
    assert second['decision_index']==1 and second['orientation']['samples_integrated']==25
    assert second['direction_current_body_xy']==pytest.approx([.8*np.cos(.05),-.8*np.sin(.05)])
    assert first['input_tensor_sha256']['rgb']!=second['input_tensor_sha256']['rgb']


@pytest.mark.parametrize('fault',['repeat_choice','missing_packet','privilege'])
def test_fault_latches_and_cannot_silently_reuse_old_prediction(fault):
    rows=stream(); value=adapter(); ns=begin(value,rows); value.select(now_ns=ns)
    if fault=='repeat_choice':
        with pytest.raises(SensorContractError): value.select(now_ns=ns)
    elif fault=='missing_packet':
        with pytest.raises(SensorContractError): value.observe(rows[8],now_ns=rows[8]['image']['measured_ns'])
    else:
        rows[4]['world_pose']=np.zeros(7)
        with pytest.raises(SensorContractError): value.observe(rows[4],now_ns=rows[4]['image']['measured_ns'])
    with pytest.raises(SensorContractError): value.select(now_ns=ns+500_000_000)
    with pytest.raises(SensorContractError): value.observe(rows[9],now_ns=rows[9]['image']['measured_ns'])
    with pytest.raises(SensorContractError): value.begin_episode((0,0,0))


def test_ensemble_averages_contact_probabilities_not_logits():
    prediction=np.zeros((3,5,5)); prediction[...,4]=10.
    prediction[:,0,4]=[-20.,10.,10.]; prediction[:,1,4]=.3
    result=rank_half_second_ensemble(prediction,[.8,0])
    assert result['selected_action_index']==1
    assert result['mean_contact_probability'][0]==pytest.approx(np.mean(1/(1+np.exp(-prediction[:,0,4]))))


def test_incomplete_warmup_and_unbound_paths_rejected(tmp_path):
    value=adapter(); p=stream()[0]; value.observe(p,now_ns=p['image']['measured_ns'])
    with pytest.raises(SensorContractError): value.begin_control([.8,0],now_ns=p['image']['measured_ns'])
    with pytest.raises(ValueError,match='nonprotected'): _read_bound(tmp_path/'sealed'/'x', '0'*64)
    with pytest.raises(ValueError): OnlineTemporalChoice.from_completed_study('best_validation_seed')


def test_ensemble_population_cannot_be_reduced_to_a_chosen_seed():
    with pytest.raises(ValueError): OnlineTemporalChoice('jepa_direct',[StubModel()],[{'seed':1}])

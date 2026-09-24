"""Causal boundaries, command interruption and observation censoring."""
from copy import deepcopy
import pytest
import torch
from lewm.pulse_timed_observation_pairing_development import pulse_window,observation_pair_tensors
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT
from scripts.startup_raw_sensor_audit_development import read_json


def fixture():
    frames=[dict(image_ns=1_500_000_000+i*100_000_000,decision_ns=1_500_000_000+i*100_000_000) for i in range(50)]
    tape=[dict(tick=i,requested_command=[.2,0.,0.] if 3<=i<5 else [0.,0.,0.],
               completed=True,pre_sample_index=749+50*i,post_sample_index=799+50*i) for i in range(49)]
    args=dict(departure_tick=3,departure_ns=1_800_000_000,command=(.2,0.,0.),pulse_ticks=2)
    return frames,tape,args


def test_exact_partial_endpoint_and_target_command_is_not_in_prefix():
    f,t,a=fixture();t[25]['requested_command']=[0.,0.,.45]
    r=pulse_window(f,t,**a)
    assert r['history_observation_indices']==[0,1,2,3]
    assert [x['offset_ns'] for x in r['targets']]==[500000000,1000000000,1500000000,2000000000,2200000000,0,0,0]
    assert all(x['future_valid'] for x in r['targets'][:5])
    assert r['targets'][4]['future_observation_index']==25


@pytest.mark.parametrize('mode',['changed','partial','absent'])
def test_unexecuted_prefix_censors_only_affected_targets(mode):
    f,t,a=fixture()
    if mode=='changed':t[24]['requested_command']=[0.,0.,.45]
    elif mode=='partial':t[24]['completed']=False;t[24]['post_sample_index']-=1
    else:t=t[:24]
    r=pulse_window(f,t,**a)
    assert all(x['future_valid'] for x in r['targets'][:4])
    assert r['targets'][4]['reason']=='UNEXECUTED_PREFIX'
    assert r['targets'][4]['observation_available']


def test_missing_rgb_not_confused_with_missing_execution():
    f,t,a=fixture();f.pop(25);r=pulse_window(f,t,**a)
    assert r['targets'][4]['command_prefix_executed']
    assert r['targets'][4]['reason']=='MISSING_RAW_OBSERVATION'


def test_missing_past_is_retained_not_fabricated():
    f,t,a=fixture();f.pop(0);r=pulse_window(f,t,**a)
    assert not r['history_ready'] and r['history_observation_indices'][0] is None
    with pytest.raises(ValueError,match='past RGB'):observation_pair_tensors(None,r)


@pytest.mark.parametrize('mode',['clock','completed','interval','nan'])
def test_malformed_accounting_rejected(mode):
    f,t,a=fixture()
    if mode=='clock':a['departure_ns']+=1
    elif mode=='completed':t[3]['completed']=1
    elif mode=='interval':t[3]['post_sample_index']-=1
    else:t[3]['requested_command'][0]=float('nan')
    with pytest.raises(ValueError):pulse_window(f,t,**a)


def test_real_recorded_pulse_pairs_actual_images_and_keeps_inputs_separate():
    d=ROOT/'.generated/go2_coupled_room_return_v1_attempt_001/nominal_left'
    frames=read_json(d,'policy_observations.json')['frames'];tape=read_json(d,'command_tape.json')
    r=pulse_window(frames,tape,departure_tick=20,departure_ns=3_500_000_000,command=(0.,0.,.45),pulse_ticks=2)
    pair=observation_pair_tensors(IntentReturnRGBDReplay(d),r)
    assert set(pair['inputs'])=={'observation_history','known_action_blocks','known_action_valid'}
    assert pair['targets']['future_valid'].tolist()==[True]*5+[False]*3
    assert pair['targets']['target_offsets_ns'][4]==2_200_000_000
    original=deepcopy(pair['inputs']);pair['targets']['future_observations']['rgb'].fill_(0)
    assert torch.equal(original['observation_history']['rgb'],pair['inputs']['observation_history']['rgb'])
    broken=deepcopy(r);broken['targets'][4]['future_observation_index']+=1
    with pytest.raises(ValueError,match='timestamp'):observation_pair_tensors(IntentReturnRGBDReplay(d),broken)

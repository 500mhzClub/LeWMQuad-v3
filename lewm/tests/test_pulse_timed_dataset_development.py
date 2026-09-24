"""Split/coverage/schedule integrity plus actual recorded tensor joining."""
from copy import deepcopy
from collections import Counter
import pytest
import torch
from lewm.pulse_timed_dataset_development import PulseTimedDataset,decode_targets,stack_samples
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan,validate_timed_plan
from lewm.coupled_pulse_rollout_development import COMMANDS


def fixture():
    windows=[];targets=[];roles={}
    for episode in ('a','b'):
        roles[episode]=dict(layout_id='layout-'+episode,role='train')
        for action in range(6):
            command=COMMANDS[action//2];ticks=(2,5)[action%2]
            b,m=pulse_brake_plan(command,ticks);active,offsets=validate_timed_plan(b[None],m[None],1)
            w=dict(condition=episode,departure_tick=10+action,decision_ns=2_500_000_000+action*100_000_000,
                action_index=action,command=list(command),pulse_ticks=ticks,
                history_ready=True,history_observation_indices=[7+action,8+action,9+action,10+action],
                targets=[dict(offset_ns=int(o),future_valid=bool(v)) for o,v in zip(offsets[0],active[0])])
            t={k:w[k] for k in ('condition','departure_tick','decision_ns','action_index')}
            t.update(target_only=True,targets=[dict(offset_ns=r['offset_ns'],image_target_valid=r['future_valid'],
                motion_valid=r['future_valid'],contact_valid=r['future_valid'],
                motion=[.1,0.,0.] if r['future_valid'] else None,contact=0. if r['future_valid'] else None)
                for r in w['targets']])
            windows.append(w);targets.append(t)
    return windows,targets,roles


def test_exact_balanced_cycle_and_reproducible_shared_schedule():
    w,t,r=fixture();d=PulseTimedDataset(w,t,r)
    a=d.schedule('train',updates=12,batch_size=2,seed=73)
    assert a==d.schedule('train',updates=12,batch_size=2,seed=73)
    assert a['schedule_sha256']!=d.schedule('train',updates=12,batch_size=2,seed=74)['schedule_sha256']
    assert Counter(i for batch in a['batches'] for i in batch)=={i:2 for i in range(12)}
    assert not a['outcome_conditioned_sampling'] and not a['layout_geometry_verified_by_interface']


def test_targets_cannot_change_sample_schedule():
    w,t,r=fixture();first=PulseTimedDataset(w,t,r);altered=deepcopy(t)
    for row in altered:
        for target in row['targets']:
            if target['motion_valid']:target['motion']=[100.,-200.,3.]
    other=PulseTimedDataset(w,altered,r)
    assert first.schedule('train',updates=5,batch_size=3,seed=1)==other.schedule('train',updates=5,batch_size=3,seed=1)


def test_same_layout_cannot_cross_roles_even_with_different_episode_names():
    w,t,r=fixture();r['b']=dict(layout_id='layout-a',role='selection')
    with pytest.raises(ValueError,match='one layout'):PulseTimedDataset(w,t,r)
    r['b']['role']='train';d=PulseTimedDataset(w,t,r)
    assert len(d.coverage('train'))==1 and d.coverage('train')['layout-a']=={str(i):2 for i in range(6)}


def test_missing_cells_fail_strict_schedule_but_remain_reported():
    w,t,r=fixture();w=w[1:];t=t[1:];d=PulseTimedDataset(w,t,r)
    assert d.coverage('train')['layout-a']['0']==0
    with pytest.raises(ValueError,match='missing action'):d.schedule('train',updates=1,batch_size=2,seed=0)
    assert d.schedule('train',updates=1,batch_size=2,seed=0,require_all_actions=False)['batches']


def test_missing_history_retained_not_fabricated_and_empty_layout_not_hidden():
    w,t,r=fixture()
    for row in w[:6]:row['history_ready']=False;row['history_observation_indices'][0]=None
    d=PulseTimedDataset(w,t,r)
    assert len(d)==12 and len(d.excluded)==6 and set(d.coverage('train')['layout-a'].values())=={0}
    with pytest.raises(ValueError,match='no eligible'):d.schedule('train',updates=1,batch_size=2,seed=0,require_all_actions=False)


@pytest.mark.parametrize('mutation',['duplicate','missing','identity','action','time','role'])
def test_invalid_join_or_publisher_metadata_rejected(mutation):
    w,t,r=fixture()
    if mutation=='duplicate':t.append(deepcopy(t[0]))
    elif mutation=='missing':t.pop()
    elif mutation=='identity':t[0]['condition']='wrong'
    elif mutation=='action':w[0]['command']=[0.,0.,.45]
    elif mutation=='time':t[0]['targets'][4]['offset_ns']+=100_000_000
    else:r['a']['role']='sealed'
    with pytest.raises(ValueError):PulseTimedDataset(w,t,r)


def test_native_and_image_masks_remain_independent():
    w,t,r=fixture();t[0]['targets'][0].update(motion_valid=False,motion=None,contact=1.)
    w[0]['targets'][1]['future_valid']=False;t[0]['targets'][1]['image_target_valid']=False
    d=PulseTimedDataset(w,t,r);n=decode_targets(w[0],t[0])
    assert not n['motion_valid'][0] and n['contact_valid'][0] and n['contact'][0]==1
    assert n['motion_valid'][1] and not w[0]['targets'][1]['future_valid']
    assert len(d)==12


def test_constructor_copies_mutable_inputs():
    w,t,r=fixture();d=PulseTimedDataset(w,t,r);w[0]['command'][0]=100.;r['a']['role']='selection'
    assert d.windows[0]['command'][0]==.2 and d.episode_roles['a']['role']=='train'


def test_real_recorded_rows_match_existing_native_materializer():
    from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,OUTPUT
    from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
    from scripts.startup_raw_sensor_audit_development import read_json,read_npz
    from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
    from lewm.recorded_pulse_native_targets_development import RecordedPulseNativeTargets
    windows=read_json(OUTPUT,'windows.json');targets=read_json(LABELS,'targets.json')
    roles={c:dict(layout_id='same-four-wall-room',role='train') for c in {w['condition'] for w in windows}}
    d=PulseTimedDataset(windows,targets,roles)
    assert len(d)==185 and len(d.coverage('train'))==1
    readers={};samples=[]
    for c in ('nominal_left','lower_friction_left'):
        i=next(i for i,w in enumerate(windows) if w['condition']==c)
        readers[c]=IntentReturnRGBDReplay(INPUT/c);samples.append(d.sample(i,readers))
        expected=RecordedPulseNativeTargets(read_npz(INPUT/c,'physics_trace.npz')).labels(windows[i])
        for name in ('motion','motion_valid','contact','contact_valid','target_offsets_ns'):
            torch.testing.assert_close(samples[-1]['targets'][name],expected[name],equal_nan=True)
        samples[-1]['targets']['motion'].fill_(999.)
        torch.testing.assert_close(d.sample(i,readers)['targets']['motion'],expected['motion'],equal_nan=True)
    batch=stack_samples([d.sample(next(i for i,w in enumerate(windows) if w['condition']==c),readers)
        for c in readers])
    assert batch['inputs']['observation_history']['rgb'].shape==(2,4,3,96,128)
    assert set(batch['inputs'])=={'observation_history','known_action_blocks','known_action_valid'}

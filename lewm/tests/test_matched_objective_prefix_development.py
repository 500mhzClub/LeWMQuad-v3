"""Objective comparison preserves observations and never follows a changed action."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.matched_objective_prefix_development import compare_step, SHARED

CORRECTIONS={'jepa':[[0.,0.]]*8,'supervised_rollout':[[.01,0.]]*8}


def decision(condition,frame,*,change=False):
    d={k:{'observed_frame':frame} for k in SHARED}
    action='right_turn' if change else 'left_turn'
    request=[0.,0.,-.45 if change else .45] if frame>=3 else [0.,0.,0.]
    prediction=np.zeros((6,8,5)); prediction[:,:,3]=1.
    if condition=='supervised_rollout': prediction[:,:,0]=.01
    selection=None if frame<3 else dict(action=action,prediction=prediction.tolist(),head='rollout_outcomes',
        input_variant='full',model_prediction_corrected=True,translation_bias_training_only=True,
        translation_bias_xy_m=deepcopy(CORRECTIONS[condition]),first_prediction_horizon_ns=100_000_000,
        prediction_horizon_ns=800_000_000,target_offsets_ns=list(range(100_000_000,800_000_001,100_000_000)))
    d.update(controller='measured_floor_transport_round_trip_controller_v1',tick=frame,
        model_condition=condition,input_variant='full',memory_variant='persistent',
        terminal=None,failure=None,requested_command=request,new_selection=selection)
    return d


def compare(a,b,frame,original=None,actual=None):
    return compare_step(deepcopy(a) if original is None else original,a,b,
        a['requested_command'] if actual is None else actual,frame=frame,layout=1,corrections=CORRECTIONS)


def test_matched_warmup_and_predictions_do_not_alone_end_physical_prefix():
    for frame in range(4):
        a,b=decision('jepa',frame),decision('supervised_rollout',frame)
        r=compare(a,b,frame)
        assert not r['stop'] and r['shared_observed_state_exact']
        assert r['both_full_forecast_banks_present']==(frame==3)
        assert r['raw_prediction_changed']==(frame==3)


@pytest.mark.parametrize('field',SHARED)
def test_observed_state_cannot_be_changed_with_the_model(field):
    a,b=decision('jepa',3),decision('supervised_rollout',3)
    b[field]={'altered':True}
    with pytest.raises(ValueError,match='same current'): compare(a,b,3)


@pytest.mark.parametrize('fault',['head','bias','variant','clock','shape','nonfinite','original','tape','warmup'])
def test_confounding_and_changed_original_evidence_reject(fault):
    a,b=decision('jepa',3),decision('supervised_rollout',3); original=deepcopy(a); actual=deepcopy(a['requested_command'])
    if fault=='head': b['new_selection']['head']='direct_outcomes'
    elif fault=='bias': b['new_selection']['translation_bias_xy_m']=CORRECTIONS['jepa']
    elif fault=='variant': b['input_variant']='no_rgb'
    elif fault=='clock': b['new_selection']['first_prediction_horizon_ns']=500_000_000
    elif fault=='shape': b['new_selection']['prediction']=[[1.]]
    elif fault=='nonfinite': b['new_selection']['prediction'][0][0][0]=float('nan')
    elif fault=='original': original['quiet_intervals']={'altered':True}
    elif fault=='tape': actual=[.2,0.,0.]
    elif fault=='warmup':
        a,b=decision('jepa',0),decision('supervised_rollout',0); b['extra_policy']='changed'
        original=deepcopy(a); actual=a['requested_command']
    with pytest.raises(ValueError): compare(a,b,a['tick'],original,actual)


def test_changed_command_or_terminal_stops_the_prefix():
    a,b=decision('jepa',3),decision('supervised_rollout',3,change=True)
    assert compare(a,b,3)['stop']
    b['requested_command']=[0.,0.,0.]; b['terminal']='NO_FEASIBLE_ACTION'
    assert compare(a,b,3)['terminal_changed'] and compare(a,b,3)['stop']


def test_full_runner_stops_before_reading_the_next_recorded_observation(monkeypatch,tmp_path):
    from scripts import replay_go2_matched_objective_prefixes_v1 as runner
    monkeypatch.setattr(runner,'OUTPUT',tmp_path); monkeypatch.setattr(runner,'INPUT',tmp_path/'input')
    models=[SimpleNamespace(state_dict=lambda:{},parameters=lambda:[]) for _ in range(2)]
    monkeypatch.setattr(runner,'load_pair',lambda a:(models,[{'corrected_state_sha256':'state'}]*2))
    monkeypatch.setattr(runner,'state_digest',lambda d:'state')
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    observed=[]
    class Controller:
        def __init__(self,*args,condition,**kwargs): self.condition=condition; self.selector=SimpleNamespace(head='rollout_outcomes')
        def observe(self,policy,*args,**kwargs):
            i=policy['frame']; observed.append((self.condition,i))
            return decision(self.condition,i,change=self.condition=='supervised_rollout' and i==3)
    monkeypatch.setattr(runner,'MeasuredFloorTransportController',Controller)
    monkeypatch.setattr(runner,'IntentReturnRGBDReplay',lambda p:SimpleNamespace(frames=[None]*216,
        packet=lambda i:({'frame':i},{},{},1_500_000_000+i*100_000_000)))
    tape=[{'requested_command':decision('jepa',i)['requested_command'],'completed':True} for i in range(215)]
    monkeypatch.setattr(runner,'read_json',lambda p,n:tape if n=='command_tape.json' else [{}]*216)
    monkeypatch.setattr(runner,'packet',lambda *args,**kwargs:({},{}))
    monkeypatch.setattr(runner,'public_acquisition',lambda a:a)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**15))
    def rows(p):
        for i in range(4):
            yield dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=decision('jepa',i))
        pytest.fail('borrowed observation after changed command')
    monkeypatch.setattr(runner,'read_rows',rows)
    launch={'correction_admission':{'coefficients':{n:{'heads':{'rollout_outcomes':{'applied_bias_xy_m':CORRECTIONS[c]}}}
        for n,c in zip(runner.NAMES,runner.CONDITIONS,strict=True)}}}
    report=runner.replay_case(launch,1)
    assert report['frames']==4 and report['prior_commands_compared']==3
    assert report['first_prediction_difference']==report['first_requested_command_difference']==3
    assert len(observed)==8 and not report['following_recorded_observations_consumed']

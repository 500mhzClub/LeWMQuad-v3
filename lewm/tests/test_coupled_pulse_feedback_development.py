"""Controller contract tests and a model-matched synthetic plant, not physics."""
import math
from copy import deepcopy
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.sensor_anchored_goal_development import AnchoredGoal
from lewm.coupled_pulse_rollout_development import compose
from lewm.coupled_pulse_feedback_development import CoupledPulseServo
from lewm.coupled_room_return_development import CoupledPulseExecution,CoupledRoomReturn,RawCoupledPulseExecution
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_coupled_pulse_rollout_development import table
from scripts.fixed_nominal_pulse_table_development import load_fixed_table


def servo(delta=(.4,0),angle=0.):
    e,t=visual(0)
    return CoupledPulseServo(AnchoredGoal.from_observation(e,delta,angle,identity=(0,0,0),now_ns=t),table())


@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('empirical',[False,True])
def test_model_matched_full_room_return_with_signed_turns(sign,empirical):
    model_table=load_fixed_table() if empirical else table()
    model=CoupledRoomReturn(sign,model_table);state=np.zeros(3);increment=np.zeros(3)
    for tick in range(3601):
        e,t=visual(tick,*state);ex=model.runtime.executor.observe(e,now_ns=t)
        r=model.advance(dict(evidence=e,execution=ex,requested_command=ex['requested_command']),now_ns=t)
        if r['terminal']:break
        local=ex['local_decision']
        if local and 'new_pulse' in local['diagnostic']:
            effect=model_table.effects[local['diagnostic']['new_pulse']['action_index']]
            increment=(compose(state,effect.delta_xy_yaw)-state)/effect.ticks
        if any(r['requested_command']):state+=increment
    assert r['terminal']=='ROOM_RETURN_CANDIDATE',r
    assert len(model.completed)==7 and np.linalg.norm(state[:2])<=.06
    legs=model.runtime.executor.legs
    for index in (1,3):
        d=legs[index]['final_decision'];net=d['diagnostic']['unwrapped_yaw_rad']-d['anchor_yaw_rad']
        assert abs(net-sign*(math.pi/2 if index==1 else math.pi))<=.05
    assert not r['mission_complete'] and not model.snapshot()['home_verified']


def test_only_first_planned_pulse_then_real_braking_and_replanning():
    m=servo();rows=[]
    for tick in range(1,44):
        e,t=visual(tick);rows.append(m.step(e,now_ns=t))
    first=next(i for i,r in enumerate(rows) if 'new_pulse' in r['diagnostic'])
    count=rows[first]['diagnostic']['new_pulse']['ticks']
    assert first==19 and len(rows[first]['diagnostic']['rollout']['action_indices'])>1
    assert all(any(r['requested_command']) for r in rows[first:first+count])
    assert all(not any(r['requested_command']) for r in rows[first+count:first+count+20])
    assert sum('new_pulse' in r['diagnostic'] for r in rows)==(2 if first+count+20<len(rows) else 1)


@pytest.mark.parametrize('fault',['stale','missing','gap','yaw_jump','nan'])
def test_bad_sensor_input_latches_zero(fault):
    m=servo();e,t=visual(1)
    if fault=='stale':e['current_pose']['measured_ns']-=1
    if fault=='missing':e['current_pose']=None
    if fault=='gap':e,t=visual(2)
    if fault=='yaw_jump':e,t=visual(1,yaw=.21)
    if fault=='nan':e['current_pose']['position_initial_body_m'][0]=float('nan')
    r=m.step(e,now_ns=t)
    assert r['terminal']=='VISUAL_SERVO_FAILED' and not any(r['requested_command'])
    e,t=visual(3);assert not any(m.step(e,now_ns=t)['requested_command'])


def test_goal_candidate_needs_ten_new_quiet_intervals_not_prediction():
    m=servo(delta=(0,0))
    for tick in range(1,30):
        e,t=visual(tick);assert m.step(e,now_ns=t)['terminal'] is None
    e,t=visual(30);assert m.step(e,now_ns=t)['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'


def test_wrapped_opposite_half_turn_does_not_satisfy_signed_request():
    m=servo(delta=(0,0),angle=-math.pi)
    for tick in range(1,65):
        e,t=visual(tick,yaw=min(tick/32,1)*math.pi);r=m.step(e,now_ns=t)
        assert r['terminal']!='VISUAL_TARGET_SEQUENCE_COMPLETE'
    assert m.target_unwrapped_yaw==-math.pi and m.unwrapped_yaw==pytest.approx(math.pi)


def test_continuous_owner_substitutes_before_old_controller_can_step(monkeypatch):
    from lewm.anchored_pulse_servo_development import AnchoredPulseServo
    def forbidden(*a,**k):raise AssertionError('old controller must not step')
    monkeypatch.setattr(AnchoredPulseServo,'step',forbidden)
    m=CoupledPulseExecution(table());e,t=visual(0);m.observe(e,now_ns=t);m.begin([.4,0],0.,now_ns=t)
    assert isinstance(m.active,CoupledPulseServo)
    e,t=visual(1);assert m.observe(e,now_ns=t)['status']=='EXECUTING'
    m.fail('test',now_ns=t)
    with pytest.raises(SensorContractError):m.begin([0,0],0.,now_ns=t)


def test_outer_mission_budget_suppresses_child_pulse():
    m=CoupledPulseExecution(table(),maximum_pulses=1);e,t=visual(0);m.observe(e,now_ns=t);m.begin([.4,0],0.,now_ns=t)
    for tick in range(1,100):
        e,t=visual(tick);r=m.observe(e,now_ns=t)
        if r['status']=='FAILED':break
    assert r['status']=='FAILED' and not any(r['requested_command']) and m.pulses==1


def test_raw_input_rejection_latches_zero_and_keeps_single_owner():
    m=RawCoupledPulseExecution(table());r=m.observe({}, {}, {}, now_ns=1_600_000_000)
    assert r['execution']['status']=='FAILED' and not any(r['requested_command'])


def test_fixed_model_matches_bound_offline_nominal_table():
    from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest
    from scripts.startup_raw_sensor_audit_development import read_json
    p=ROOT/'.generated/go2_coupled_pulse_rollout_diagnostic_v1_attempt_001'
    assert digest(p/'result.json')=='500e0d973ce4b6d2f119af888557af0780dab84475b323f45df87ed275a4af47'
    expected=read_json(p,'result.json')['transfer']['table'];actual=load_fixed_table()
    for e,a in zip(expected,actual.effects,strict=True):
        assert tuple(e['command'])==a.command and e['pulse_ticks']==a.ticks
        np.testing.assert_allclose(np.array(e['mean_sensor_delta_xyz_yaw'])[[0,1,3]],a.delta_xy_yaw,rtol=0,atol=0)


@pytest.mark.parametrize('sign',[-1,1])
def test_native_winding_score_rejects_opposite_equivalent_orientation(sign):
    from scripts.audit_go2_coupled_room_return_v1 import score_winding
    e,t=visual(0,origin=1_500_000_000)
    g=AnchoredGoal.from_observation(e,[0,0],sign*math.pi,identity=(0,0,0),now_ns=t).snapshot()
    yaw=np.zeros(1750);yaw[749:1249]=np.linspace(0,sign*math.pi,500);yaw[1249:]=sign*math.pi
    assert score_winding(yaw,g,end=1749)['passed']
    assert not score_winding(-yaw,g,end=1749)['passed']
    yaw[1500]+=.051
    assert not score_winding(yaw,g,end=1749)['passed']


def test_new_actual_scene_starts_are_not_predecessor_relabeling():
    from lewm.coupled_room_return_scene_development import specification,pack,TRIALS
    from lewm.room_return_scene_development import specification as old
    for c in TRIALS:
        s=specification(c);before=old(c);p=pack(s)
        assert s['geometry']['spawn_se2_world']!=before['geometry']['spawn_se2_world']
        assert s['procedural_seed']!=before['procedural_seed'] and s['appearance_seed']!=before['appearance_seed']
        assert len(p.static_objects)==4 and p.robot.spawn_xyz_m[:2]==tuple(s['geometry']['spawn_se2_world'][:2])
        assert p.physics_randomization.floor_friction_mu==s['friction_mu']


def test_storage_fault_preserves_history_and_drives_terminal_zero():
    m=CoupledRoomReturn(1,table());e,t=visual(0);ex=m.runtime.executor.observe(e,now_ns=t)
    m.advance(dict(evidence=e,execution=ex,requested_command=ex['requested_command']),now_ns=t)
    m.runtime.executor.fail('STORAGE_RESERVE_STOP',now_ns=t+100_000_000)
    e,t=visual(1);ex=m.runtime.executor.observe(e,now_ns=t)
    r=m.advance(dict(evidence=e,execution=ex,requested_command=ex['requested_command']),now_ns=t)
    assert r['terminal']=='ROOM_RETURN_FAILED' and r['reason']=='STORAGE_RESERVE_STOP'
    assert not any(r['requested_command']) and len(m.runtime.executor.legs)==1

"""Scheduling and fault tests; synthetic plants are not physical return evidence."""
from copy import deepcopy
import math
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.room_return_pulse_development import RoomReturnPulse,STAGES,local_target
from lewm.room_return_scene_development import TRIALS,specification,pack
from lewm.raw_pulse_runtime_development import RawPulseExecution,RawPulseRouteRuntime
from lewm.sensor_anchored_goal_development import AnchoredGoal
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_continuous_pulse_execution_development import visual
from scripts.audit_go2_room_return_pulse_v1 import score_hold


@pytest.mark.parametrize('sign',[-1,1])
def test_complete_continuous_room_return_on_responsive_plant(sign):
    model=RoomReturnPulse(sign);x=y=yaw=0.;dispatches=[]
    for tick in range(3601):
        e,t=visual(tick,x,y,yaw);ex=model.runtime.executor.observe(e,now_ns=t)
        r=model.advance(dict(evidence=e,execution=ex,requested_command=ex['requested_command']),now_ns=t)
        if r['dispatch']:dispatches.append(r['dispatch'])
        if r['terminal']:break
        vx,_,w=r['requested_command'];x+=vx*.1*math.cos(yaw);y+=vx*.1*math.sin(yaw);yaw+=w*.1
    assert r['terminal']=='ROOM_RETURN_CANDIDATE',r
    assert [s['stage'] for s in model.completed]==list(STAGES)
    assert math.hypot(x,y)<=.06 and abs(math.atan2(math.sin(yaw),math.cos(yaw)))<=.05
    assert model.runtime.executor.start_ns==1_600_000_000
    assert not r['mission_complete'] and not model.snapshot()['home_verified']
    assert model.corner['position_initial_body_m'][0]>.3
    assert all(d['goal']['anchor_frame']>0 for d in dispatches[1:])


def test_stored_point_transform_handles_body_tilt_and_clipping():
    e,t=visual(0);R=rotation_increment([.1,-.15,.7]);e['current_pose']['rotation_initial_body_from_current_body']=R.tolist()
    delta,angle,final=local_target(e['current_pose'],[.1,.1],-.4)
    goal=AnchoredGoal.from_observation(e,delta,angle,identity=(0,0,0),now_ns=t)
    assert final and goal.target_xy==pytest.approx([.1,.1]) and goal.target_yaw_rad==pytest.approx(-.4)
    delta,_,final=local_target(e['current_pose'],[1.,1.],0.)
    assert not final and np.linalg.norm(delta)==pytest.approx(.4)
    e['current_pose']['rotation_initial_body_from_current_body']=rotation_increment([math.pi/2,0,0]).tolist()
    with pytest.raises(SensorContractError):local_target(e['current_pose'],[0,0],0.)


def test_partial_return_cannot_complete_stage_by_merely_finishing_local_lookahead():
    model=RoomReturnPulse(1);model.index=4;model.pending=dict(final=False,leg_index=0)
    e,t=visual(0);model.home=deepcopy(e['current_pose']);model.corner=deepcopy(model.home)
    model.corner['position_initial_body_m']=[1.,0.,0.]
    ex=model.runtime.executor.observe(e,now_ns=t)
    r=model.advance(dict(evidence=e,execution=ex,requested_command=[0,0,0]),now_ns=t)
    assert model.index==4 and not model.completed and r['dispatch']['final'] is False


def test_raw_input_failure_is_zero_and_cannot_start_new_leg():
    model=RawPulseExecution();r=model.observe({}, {}, {}, now_ns=1_600_000_000)
    assert r['execution']['status']=='FAILED' and r['requested_command']==[0,0,0]
    with pytest.raises(SensorContractError):model.executor.begin([.4,0],0.,now_ns=1_600_000_000)
    route=RawPulseRouteRuntime();r=route.observe({}, {}, {}, now_ns=1_600_000_000)
    assert r['route']['fault'] and r['requested_command']==[0,0,0]


def test_raw_route_uses_visual_observers_exact_gyro_stream(monkeypatch):
    runtime=RawPulseRouteRuntime();sentinel={'actual':'gyro'};seen=[]
    monkeypatch.setattr(runtime.motion,'observe',lambda *a,**k:{'actual':'evidence'})
    monkeypatch.setattr(runtime.motion.model.gyro,'_result',lambda:sentinel)
    def consume(p,a,e,**k):
        seen.append((a,e));return {'requested_command':[0,0,0]}
    monkeypatch.setattr(runtime.bridge,'observe',consume)
    runtime.observe({}, {}, {}, now_ns=1_600_000_000)
    assert seen==[(sentinel,{'actual':'evidence'})]


def test_raw_failure_reaches_room_terminal_and_native_stop_keeps_history():
    model=RoomReturnPulse(1);r=model.observe({}, {}, {}, now_ns=1_600_000_000)
    assert r['terminal']=='ROOM_RETURN_FAILED' and r['requested_command']==[0,0,0]
    model.finish_physical_stop('contact',now_ns=1_700_000_000)
    assert model.snapshot()['terminal']=='ROOM_RETURN_PHYSICAL_STOP' and not model.completed


def test_three_declared_trials_build_all_four_native_walls_and_fresh_starts():
    robots=[]
    for c in TRIALS:
        s=specification(c);p=pack(s);robots.append(p.robot)
        assert len(p.static_objects)==4 and {o.object_id for o in p.static_objects}=={'east','west','north','south'}
        assert p.physics_randomization.floor_friction_mu==(.15 if c.startswith('lower') else 1.)
        assert p.robot.spawn_xyz_m[:2]==tuple(s['geometry']['spawn_se2_world'][:2])
        bad=deepcopy(s);bad['geometry']['wall_boxes'][0]['centre_xyz'][0]+=.1
        with pytest.raises(ValueError):pack(bad)
    assert robots[0]==robots[2] and robots[0]!=robots[1]


def test_native_hold_checks_interior_motion_and_entire_interval():
    p=np.zeros((1300,3));yaw=np.zeros(1300);twist=np.zeros((1300,6))
    assert score_hold(p,yaw,twist,end=1299,target_xy=[0,0],target_yaw=0)['passed']
    p[1100,0]=.061
    assert not score_hold(p,yaw,twist,end=1299,target_xy=[0,0],target_yaw=0)['passed']
    p[:]=0.;yaw[1100]=.001
    assert not score_hold(p,yaw,twist,end=1299,target_xy=[0,0],target_yaw=0)['passed']
    with pytest.raises(ValueError):score_hold(p,yaw,twist,end=1248,target_xy=[0,0],target_yaw=0)

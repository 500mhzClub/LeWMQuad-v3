"""Synthetic actuator/pose fixtures, not physical or place-recognition evidence."""
from copy import deepcopy
import hashlib
import math
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.sensor_anchored_goal_development import AnchoredGoal
from lewm.anchored_pulse_servo_development import AnchoredPulseServo
from lewm.continuous_pulse_execution_development import ContinuousPulseExecution
from lewm.pulse_route_bridge_development import PulseRouteBridge
from lewm.tests.test_bounded_visual_servo_development import evidence
from lewm.tests.test_episodic_route_hypotheses_development import proposal
from lewm.tests.test_observed_traversal_controller_development import Stream


def visual(tick, x=0., y=0., yaw=0., *, origin=1_600_000_000, rgb='a'*64):
    now=origin+tick*100_000_000
    e=evidence(now,x=x,y=y,yaw=yaw)
    e['current_pose'].update(frame=tick,rgb_sha256=rgb,depth_sha256='b'*64)
    return e,now


def test_sensor_anchored_goal_transforms_once_and_copies_input():
    e,t=visual(7,2.,3.,math.pi/2)
    goal=AnchoredGoal.from_observation(e,[.4,0.],-.3,identity=(0,0,0),now_ns=t)
    assert goal.target_xy==pytest.approx((2.,3.4))
    assert goal.target_yaw_rad==pytest.approx(math.pi/2-.3)
    assert goal.anchor_frame==7 and goal.rgb_sha256=='a'*64
    e['current_pose']['position_initial_body_m'][0]=100.
    assert goal.anchor_position[0]==2. and goal.snapshot()['place_identity'] is None


@pytest.mark.parametrize('delta,yaw', [([.401,0],0),([float('nan'),0],0),([0,0,0],0),([0,0],True),([0,0],4),([0,0],float('nan'))])
def test_bad_goal_rejected(delta,yaw):
    e,t=visual(0)
    with pytest.raises(SensorContractError):AnchoredGoal.from_observation(e,delta,yaw,identity=(0,0,0),now_ns=t)


def test_arbitrary_global_anchor_is_not_old_global_excursion_limit():
    e,t=visual(0,4.,-3.,math.pi/2)
    goal=AnchoredGoal.from_observation(e,[.4,0],0,identity=(0,0,0),now_ns=t)
    servo=AnchoredPulseServo(goal)
    for i in range(21):
        e,t=visual(i,4.,-3.,math.pi/2);r=servo.step(e,now_ns=t)
    assert r['terminal'] is None and r['requested_command']==[.2,0,0]
    e,t=visual(21,4.,-2.51,math.pi/2)
    assert servo.step(e,now_ns=t)['reason']=='OBSERVED_EXCURSION_LIMIT'


def test_two_legs_and_signed_turns_preserve_visual_frame_and_global_history():
    model=ContinuousPulseExecution();x=y=yaw=0.;goals=[([.4,0],.3),([0,0],-math.pi/2),([.4,0],0.),([0,0],math.pi/2)]
    selected=0
    for tick in range(3601):
        e,t=visual(tick,x,y,yaw);r=model.observe(e,now_ns=t)
        assert r['status']!='FAILED',r
        if r['status']=='IDLE':
            if selected==len(goals):break
            d,w=goals[selected];model.begin(d,w,now_ns=t);selected+=1
        vx,_,w=r['requested_command'];x+=vx*.1*math.cos(yaw);y+=vx*.1*math.sin(yaw);yaw+=w*.1
    s=model.snapshot()
    assert len(s['legs'])==4 and all(a['status']=='LOCAL_GOAL_CANDIDATE' for a in s['legs'])
    assert s['start_ns']==1_600_000_000 and s['last_frame']==tick
    assert s['legs'][2]['goal']['anchor_position'][0]>.3
    assert s['legs'][2]['goal']['anchor_frame']>s['legs'][1]['goal']['anchor_frame']
    assert s['pulses']>4 and not s['mission_complete']
    s['legs'].clear();assert len(model.snapshot()['legs'])==4


@pytest.mark.parametrize('fault', ['gap','reset','episode','missing','stale','future','hash','nan','reflection'])
def test_fault_is_latched_across_attempts(fault):
    m=ContinuousPulseExecution();e,t=visual(0);m.observe(e,now_ns=t);m.begin([.4,0],0.,now_ns=t)
    e,t=visual(1)
    if fault=='gap':e,t=visual(2)
    if fault=='reset':e['current_pose']['frame']=0
    if fault=='episode':e['identity']=(1,0,0)
    if fault=='missing':e['current_pose']=None
    if fault=='stale':e['current_pose']['measured_ns']-=1
    if fault=='future':e['current_pose']['available_ns']+=1
    if fault=='hash':e['current_pose']['rgb_sha256']='unbound'
    if fault=='nan':e['current_pose']['position_initial_body_m'][0]=float('nan')
    if fault=='reflection':e['current_pose']['rotation_initial_body_from_current_body']=(-np.eye(3)).tolist()
    r=m.observe(e,now_ns=t)
    assert r['status']=='FAILED' and r['requested_command']==[0,0,0]
    e,t=visual(3);assert m.observe(e,now_ns=t)['status']=='FAILED'
    with pytest.raises(SensorContractError):m.begin([.4,0],0.,now_ns=t)
    assert m.snapshot()['legs'][0]['status']=='FAILED_EXECUTION'


def test_mission_time_budget_counts_idle_and_cannot_restart():
    m=ContinuousPulseExecution(maximum_ticks=3)
    for i in range(4):e,t=visual(i);r=m.observe(e,now_ns=t)
    assert r['status']=='FAILED' and 'time budget' in r['reason']
    with pytest.raises(SensorContractError):m.begin([.4,0],0.,now_ns=t)


def test_invalid_decision_clock_still_latches_and_missing_image_records_current_failure_time():
    m=ContinuousPulseExecution();e,t=visual(0);m.observe(e,now_ns=t);m.begin([.4,0],0.,now_ns=t)
    assert m.observe(e,now_ns=None)['status']=='FAILED'
    n=ContinuousPulseExecution();n.observe(e,now_ns=t);n.begin([.4,0],0.,now_ns=t)
    e,t=visual(1);e['current_pose']=None
    n.observe(e,now_ns=t)
    assert n.legs[-1]['finished_ns']==t and n.last_ns==t-100_000_000


def test_total_pulse_budget_prevents_command_dispatch():
    m=ContinuousPulseExecution(maximum_pulses=1);e,t=visual(0);m.observe(e,now_ns=t);m.begin([.4,0],0.,now_ns=t)
    commands=[]
    for i in range(1,100):
        e,t=visual(i);r=m.observe(e,now_ns=t);commands.append(r['requested_command'])
        if r['status']=='FAILED':break
    assert 'pulse budget' in r['reason'] and r['requested_command']==[0,0,0]
    assert sum(c[0]>.0 for c in commands)==5 and m.pulses==1


def test_leg_budget_not_cleared_after_success():
    m=ContinuousPulseExecution(maximum_legs=1);e,t=visual(0);m.observe(e,now_ns=t);m.begin([0,0],0.,now_ns=t)
    for i in range(1,35):e,t=visual(i);r=m.observe(e,now_ns=t)
    assert r['status']=='IDLE' and len(m.legs)==1
    with pytest.raises(SensorContractError):m.begin([0,0],0.,now_ns=t)
    assert m.fault is not None and m.legs[0]['status']=='LOCAL_GOAL_CANDIDATE'


class BridgeStream:
    def __init__(self):self.stream=Stream()
    def frame(self,tick,x=0.,y=0.,yaw=0.):
        p,_,t=self.stream.frame(tick)
        e,_=visual(tick,x,y,yaw,origin=t-tick*100_000_000,rgb=hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest())
        a=dict(decision_ns=t,start_ns=1_600_000_000,rotation_initial_body_from_current_body=e['current_pose']['rotation_initial_body_from_current_body'],
               samples_integrated=tick*50,gyro_rate_hz=500,integration='causal_midpoint',hardware_calibrated=False)
        return p,a,e,t


def test_live_bridge_records_actual_local_completion_not_place_or_edge():
    stream=BridgeStream();m=PulseRouteBridge();x=0.
    for i in range(200):
        p,a,e,t=stream.frame(i,x=x);r=m.observe(p,a,e,now_ns=t)
        assert r['fault'] is None,r
        if i==0:m.begin_branch(proposal(t,packet=p),now_ns=t)
        x+=r['requested_command'][0]*.1
        if i>0 and m.active_kind is None:break
    s=m.snapshot();assert len(s['memory']['attempts'])==1 and len(s['memory']['visits'])==2
    assert s['memory']['attempts'][0]['status']=='ARRIVAL_CANDIDATE'
    assert s['memory']['trusted_graph_edges']==0 and not s['home_verified'] and not s['mission_complete']
    assert s['memory']['return_intent']['kind']=='OBSERVE_RETURN_DIRECTION'


@pytest.mark.parametrize('fault',['rgb','rotation','reference','frame_reset','stale_branch','hash_branch'])
def test_bridge_fault_aborts_pending_or_idle_memory_without_fabricated_arrival(fault):
    stream=BridgeStream();m=PulseRouteBridge();p,a,e,t=stream.frame(0);m.observe(p,a,e,now_ns=t)
    if fault in ('stale_branch','hash_branch'):
        c=proposal(t,packet=p)
        if fault=='stale_branch':c['timestamp_ns']-=1
        else:c['observation_id']='c'*64+':proposal-0'
        with pytest.raises(SensorContractError):m.begin_branch(c,now_ns=t)
    else:
        m.begin_branch(proposal(t,packet=p),now_ns=t)
        p,a,e,t=stream.frame(1)
        if fault=='rgb':e['current_pose']['rgb_sha256']='c'*64
        if fault=='rotation':a['rotation_initial_body_from_current_body']=[[0,-1,0],[1,0,0],[0,0,1]]
        if fault=='reference':a['start_ns']-=100_000_000
        if fault=='frame_reset':e['current_pose']['frame']=0
        r=m.observe(p,a,e,now_ns=t);assert r['requested_command']==[0,0,0]
    s=m.snapshot();assert s['fault'] and s['memory']['phase']=='UNCERTAIN_AFTER_FAILURE'
    assert len(s['memory']['visits'])==1 and s['memory']['trusted_graph_edges']==0
    with pytest.raises(SensorContractError):m.begin_turn(.3,now_ns=t)


def test_turn_only_records_acquired_view_without_creating_route_edge():
    stream=BridgeStream();m=PulseRouteBridge();yaw=0.
    for i in range(300):
        p,a,e,t=stream.frame(i,yaw=yaw);r=m.observe(p,a,e,now_ns=t)
        assert r['fault'] is None,r
        if i==0:m.begin_turn(-math.pi/2,now_ns=t)
        yaw+=r['requested_command'][2]*.1
        if i>0 and m.active_kind is None:break
    s=m.snapshot();assert len(s['memory']['visits'])==1 and not s['memory']['attempts']
    assert len(s['memory']['visits'][0]['context_views'])==1 and not s['mission_complete']


def test_bridge_outward_and_memory_selected_return_preserve_all_attempts():
    stream=BridgeStream();m=PulseRouteBridge();x=y=yaw=0.;dispatched=0
    for tick in range(2000):
        p,a,e,t=stream.frame(tick,x,y,yaw);r=m.observe(p,a,e,now_ns=t)
        assert r['fault'] is None,r
        if m.active_kind is None:
            if dispatched==4:break
            if dispatched<2:
                angle=0. if dispatched==0 else math.pi/2;mode='OUTWARD'
            else:
                intent=m.memory.return_intent();d=intent['direction_initial_body']
                angle=math.atan2(math.sin(math.atan2(d[1],d[0])-yaw),math.cos(math.atan2(d[1],d[0])-yaw));mode='RETURN'
                choice=m.memory.choose_return(p,a,[proposal(t,angle,p)],now_ns=t)
                assert choice['kind']=='RETURN_CANDIDATE'
            m.begin_branch(proposal(t,angle,p),now_ns=t,mode=mode);dispatched+=1
        vx,_,w=r['requested_command'];x+=vx*.1*math.cos(yaw);y+=vx*.1*math.sin(yaw);yaw+=w*.1
    s=m.snapshot()
    assert dispatched==4 and len(s['memory']['attempts'])==4
    assert s['memory']['hypothesized_route_depth']==0 and s['memory']['return_intent']['kind']=='HOME_CANDIDATE'
    assert not s['memory']['home_verified'] and s['memory']['trusted_graph_edges']==0
    assert len(s['executor']['legs'])==4 and s['executor']['start_ns']==1_600_000_000


def test_new_session_declares_pulse_domain_and_keeps_old_validator_and_guard(monkeypatch):
    from scripts.pulse_mission_session_development import PulseMissionRGBDSession
    from scripts.fresh_maze_session_development import MissionRGBDSession,validate_command as old_validate
    from scripts.rgbd_session_development import RGBDSession
    calls=[]
    monkeypatch.setattr(RGBDSession,'command_tick',lambda self,command:calls.append(command))
    session=object.__new__(PulseMissionRGBDSession)
    session.command_tick([0.,0.,.45]);session.command_tick([0.,0.,-.45])
    assert calls==[[0.,0.,.45],[0.,0.,-.45]]
    for bad in ([0,0,.451],[.201,0,0],[0,.01,0],[-.2,0,0],[0,0,float('nan')]):
        with pytest.raises(ValueError):session.command_tick(bad)
    with pytest.raises(ValueError):old_validate([0,0,.45])
    assert PulseMissionRGBDSession._sample is MissionRGBDSession._sample

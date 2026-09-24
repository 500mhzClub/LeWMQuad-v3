from copy import deepcopy
import math
import numpy as np
import pytest
from lewm.pulse_feedback_servo_development import PulseFeedbackServo,RULES
from lewm.pulse_feedback_scene_development import TRIALS,specification,pack
from lewm.tests.test_bounded_visual_servo_development import evidence


def step(model,tick,x=0.,y=0.,yaw=0.):
    t=1_500_000_000+tick*100_000_000
    return model.step(evidence(t,x=x,y=y,yaw=yaw),now_ns=t)


def test_complete_feedback_on_synthetic_command_responsive_plant():
    model=PulseFeedbackServo();x=y=yaw=0.;rows=[]
    for tick in range(1001):
        r=step(model,tick,x,y,yaw);rows.append(r)
        if r['terminal']:break
        u,_,w=r['requested_command'];x+=u*.1*math.cos(yaw);y+=u*.1*math.sin(yaw);yaw+=w*.1
    assert r['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'
    assert np.linalg.norm([x-.4,y])<=.06 and abs(yaw-.3)<=.05
    assert all(s['requested_command']==[0.,0.,0.] for s in rows[-11:])
    assert all(s['phase']==4 for s in rows[-11:])
    assert r['pulse_count']>1 and not r['learned_predictor_used']


def test_no_timer_only_arrival_and_total_pulse_budget_is_finite():
    model=PulseFeedbackServo()
    for tick in range(1001):
        r=step(model,tick)
        if r['terminal']:break
    assert r['terminal']=='VISUAL_SERVO_FAILED' and r['reason']=='PULSE_LIMIT'
    assert r['pulse_count']==35 and r['requested_command']==[0.,0.,0.]


def test_five_drive_ticks_then_twenty_zero_intervals_before_next_choice():
    model=PulseFeedbackServo();rows=[step(model,i) for i in range(46)]
    assert all(r['requested_command']==[0.,0.,0.] for r in rows[:20])
    assert all(r['requested_command']==[.2,0.,0.] for r in rows[20:25])
    assert all(r['requested_command']==[0.,0.,0.] for r in rows[25:45])
    assert rows[45]['diagnostic']['new_pulse']['pulse_index']==2


def test_only_completed_forward_pulse_updates_forward_direction():
    model=PulseFeedbackServo()
    for tick in range(25):step(model,tick)
    for tick in range(25,46):r=step(model,tick,x=.05,y=.015)
    expected=math.atan2(.015,.05)
    assert r['forward_response_offset_rad']==pytest.approx(expected)
    assert r['diagnostic']['completed_action_response']['kind']=='forward'
    assert r['diagnostic']['new_pulse']['kind']=='turn'
    assert r['requested_command'][2]<0
    for tick in range(46,48):step(model,tick,x=.05,y=.015,yaw=-.1)
    for tick in range(48,69):r=step(model,tick,x=.05,y=.03,yaw=-.1)
    assert r['forward_response_offset_rad']==pytest.approx(expected)


def test_actual_final_tolerance_not_old_tighter_correction_margin():
    model=PulseFeedbackServo()
    for tick in range(31):r=step(model,tick,x=.4,yaw=.275)
    assert r['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE' and r['pulse_count']==0


def test_goal_escape_restarts_bounded_choice_and_quiet_verification():
    model=PulseFeedbackServo()
    for tick in range(26):step(model,tick,x=.4,yaw=.3)
    r=step(model,26,x=.4,yaw=.36)
    assert r['terminal'] is None and r['stage']=='pulse' and r['quiet_intervals']==0
    assert r['requested_command']==[0.,0.,-.45]
    assert r['remaining_pulse_ticks']==1


def test_position_escape_returns_to_approach_without_reset():
    model=PulseFeedbackServo()
    for tick in range(25):step(model,tick,x=.4,yaw=.3)
    r=step(model,25,x=.33,yaw=.3)
    assert r['task_stage']=='approach' and r['stage']=='pulse' and r['terminal'] is None
    assert model.start_ns==1_500_000_000


def test_motion_rejects_brake_settling_and_elapsed_time_remains_bounded():
    model=PulseFeedbackServo()
    for tick in range(41):r=step(model,tick,x=.01*(tick%2))
    assert r['reason']=='BRAKE_NOT_SETTLED'
    other=PulseFeedbackServo();other.start_ns=-100_000_000_000
    assert step(other,0)['reason']=='TIME_LIMIT'


@pytest.mark.parametrize('fault',['missing','stale','future','mode','episode','nan','reflection','clock','excursion'])
def test_input_faults_stop_and_latch(fault):
    model=PulseFeedbackServo();step(model,0);t=1_600_000_000;e=evidence(t)
    if fault=='missing':e['current_pose']=None
    elif fault=='stale':e['current_pose']['measured_ns']-=1
    elif fault=='future':e['current_pose']['available_ns']+=1
    elif fault=='mode':e['current_pose']['mode']='joint'
    elif fault=='episode':e['identity']=(1,0,0)
    elif fault=='nan':e['current_pose']['position_initial_body_m'][0]=np.nan
    elif fault=='reflection':e['current_pose']['rotation_initial_body_from_current_body']=(-np.eye(3)).tolist()
    elif fault=='excursion':e['current_pose']['position_initial_body_m'][0]=.481
    else:t+=100_000_000;e=evidence(t)
    r=model.step(e,now_ns=t)
    assert r['terminal']=='VISUAL_SERVO_FAILED' and r['requested_command']==[0.,0.,0.]
    assert model.step(evidence(t+100_000_000),now_ns=t+100_000_000)['terminal']=='VISUAL_SERVO_FAILED'


def test_three_nominal_starts_and_friction_challenge_are_real_pack_changes():
    robots=[]
    for trial in TRIALS:
        s=specification(trial);p=pack(s);x,y,yaw=s['geometry']['spawn_se2_world'];robots.append(p.robot)
        assert p.robot.spawn_xyz_m==(x,y,.375)
        assert p.robot.spawn_quat_wxyz[3]==pytest.approx(math.sin(yaw/2))
        assert p.physics_randomization.floor_friction_mu==(1. if trial.startswith('nominal') else .15)
        bad=deepcopy(s);bad['geometry']['spawn_se2_world'][0]=0.
        with pytest.raises(ValueError):pack(bad)
    assert robots[0]!=robots[1] and robots[1]!=robots[2] and robots[0]==robots[3]
    assert RULES['final_position_tolerance_m']==.06 and RULES['final_yaw_tolerance_rad']==.05

from copy import deepcopy
import numpy as np
import pytest

from lewm.goal_hold_visual_servo_development import GoalHoldVisualServo, RULES
from lewm.goal_hold_visual_servo_scene_development import specification, pack
from lewm.tests.test_bounded_visual_servo_development import evidence


def advance(model, tick, **pose):
    now=1_500_000_000+tick*100_000_000
    return model.step(evidence(now, **pose), now_ns=now)


def at_final_brake():
    model=GoalHoldVisualServo()
    advance(model,0,x=.4)
    for tick in range(1,11): row=advance(model,tick,x=.4)
    assert row['stage']=='turn'
    row=advance(model,11,x=.4,yaw=.3)
    assert row['stage']=='final_brake' and row['quiet_intervals']==0
    return model


def test_complete_whole_sequence_requires_ten_quiet_zero_intervals():
    model=at_final_brake()
    for tick in range(12,22):
        row=advance(model,tick,x=.4,yaw=.3)
        assert row['requested_command']==[0.,0.,0.]
        assert (row['terminal'] is not None)==(tick==21)
    assert row['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'
    assert row['corrections']==0 and not row['navigation_qualified']


def test_brake_drift_corrects_and_requires_new_verification():
    model=at_final_brake()
    for tick in range(12,17): advance(model,tick,x=.4,yaw=.3)
    row=advance(model,17,x=.4,yaw=.345)
    assert row['stage']=='corrective_turn' and row['corrections']==1
    assert row['requested_command'][2]<0 and row['quiet_intervals']==0
    row=advance(model,18,x=.4,yaw=.32)
    assert row['stage']=='corrective_turn'  # tighter arrival than original turn
    row=advance(model,19,x=.4,yaw=.31)
    assert row['stage']=='final_brake' and row['quiet_intervals']==0
    for tick in range(20,30): row=advance(model,tick,x=.4,yaw=.31)
    assert row['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'


def test_correction_limit_is_global_and_failure_latches():
    model=at_final_brake(); tick=12
    for count in range(1,4):
        row=advance(model,tick,x=.4,yaw=.36); tick+=1
        assert row['corrections']==count and row['stage']=='corrective_turn'
        advance(model,tick,x=.4,yaw=.3); tick+=1
    row=advance(model,tick,x=.4,yaw=.36)
    assert row['reason']=='CORRECTION_LIMIT' and row['requested_command']==[0.,0.,0.]
    assert advance(model,tick+1,x=.4,yaw=.3)['terminal']=='VISUAL_SERVO_FAILED'


@pytest.mark.parametrize('stage',['turn','corrective_turn','final_brake'])
def test_position_drift_is_failure_not_relaxed_or_restarted(stage):
    model=at_final_brake(); model.stage=stage
    row=advance(model,12,x=.461,yaw=.3)
    assert row['terminal']=='VISUAL_SERVO_FAILED' and row['requested_command']==[0.,0.,0.]


def test_motion_resets_quiet_count_and_brake_timeout_is_bounded():
    model=at_final_brake()
    for tick in range(12,52):
        row=advance(model,tick,x=.4+.01*(tick%2),yaw=.3)
    assert row['reason']=='BRAKE_NOT_SETTLED' and row['quiet_intervals']==0


@pytest.mark.parametrize('fault',['stale','episode','mode','nan','reflection','missing','future','clock_gap'])
def test_bad_input_fails_closed(fault):
    model=at_final_brake(); now=2_700_000_000; e=evidence(now,x=.4,yaw=.3)
    if fault=='stale': e['current_pose']['measured_ns']-=1
    elif fault=='episode': e['identity']=(1,0,0)
    elif fault=='mode': e['current_pose']['mode']='joint'
    elif fault=='nan': e['current_pose']['position_initial_body_m'][0]=np.nan
    elif fault=='reflection': e['current_pose']['rotation_initial_body_from_current_body']=(-np.eye(3)).tolist()
    elif fault=='future': e['current_pose']['available_ns']+=1
    elif fault=='clock_gap': now+=100_000_000; e=evidence(now,x=.4,yaw=.3)
    else: e['current_pose']=None
    row=model.step(e,now_ns=now)
    assert row['terminal']=='VISUAL_SERVO_FAILED' and row['requested_command']==[0.,0.,0.]


def test_time_limit_and_original_final_tolerances_unchanged():
    model=GoalHoldVisualServo()
    for tick in range(351): row=advance(model,tick)
    assert row['reason']=='TIME_LIMIT'
    assert RULES['final_position_tolerance_m']==.06 and RULES['final_yaw_tolerance_rad']==.05


def test_scene_changes_actual_pose_without_changing_condition_pair():
    a=specification('nominal'); b=specification('lower_friction'); pa,pb=pack(a),pack(b)
    assert pa.robot==pb.robot and pa.robot.spawn_xyz_m==(-.60,-.10,.375)
    assert pa.robot.spawn_quat_wxyz[3]==pytest.approx(np.sin(.03))
    assert a['geometry']==b['geometry']
    assert pa.physics_randomization.floor_friction_mu==1. and pb.physics_randomization.floor_friction_mu==.15
    changed=deepcopy(a); changed['geometry']['spawn_se2_world'][0]=0.
    with pytest.raises(ValueError): pack(changed)


@pytest.mark.parametrize('fault',['none','intersample_position','intersample_yaw','speed','yaw_rate','nan','short'])
def test_native_hold_scores_intersample_failures(fault):
    from scripts.audit_go2_goal_hold_visual_servo_v1 import score_native_hold
    p=np.tile([.4,0.,0.],(501,1)); yaw=np.full(501,.3); twist=np.zeros((501,6))
    if fault=='intersample_position': p[25,0]=.47
    elif fault=='intersample_yaw': yaw[25]=.36
    elif fault=='speed': twist[25,0]=.021
    elif fault=='yaw_rate': yaw[25]=.301
    elif fault=='nan': p[25,0]=np.nan
    elif fault=='short': p=p[:-1]
    if fault in ('nan','short'):
        with pytest.raises(ValueError): score_native_hold(p,yaw,twist,0,500)
    else:
        assert score_native_hold(p,yaw,twist,0,500)['passed']==(fault=='none')

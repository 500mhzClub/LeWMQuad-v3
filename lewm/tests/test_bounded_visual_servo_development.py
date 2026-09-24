from copy import deepcopy
import numpy as np
import pytest

from lewm.bounded_visual_servo_development import BoundedVisualServo,RULES
from lewm.bounded_visual_servo_scene_development import specification,pack


def evidence(t,x=0.,y=0.,yaw=0.):
    c,s=np.cos(yaw),np.sin(yaw)
    return dict(schema='visual_led_motion_evidence_development.v1',identity=(0,0,0),decision_ns=t,
        status='CURRENT_VISUAL_POSE',terminal_failure=None,current_pose=dict(mode='gyro',measured_ns=t,available_ns=t,
            position_initial_body_m=[x,y,0.],rotation_initial_body_from_current_body=[[c,-s,0],[s,c,0],[0,0,1.]]))


def test_command_depends_on_observed_error_not_elapsed_tape():
    t=1_500_000_000; a=BoundedVisualServo(); b=BoundedVisualServo()
    first=a.step(evidence(t),now_ns=t); arrived=b.step(evidence(t,x=.4),now_ns=t)
    assert first['requested_command']==[.1,0.,0.]
    assert arrived['requested_command']==[0.,0.,0.] and arrived['stage']=='forward_brake'
    turning=BoundedVisualServo().step(evidence(t,y=.1),now_ns=t)
    assert turning['requested_command'][2]<0 and not turning['native_pose_used']


def test_complete_sequence_needs_observed_settling_and_target_turn():
    model=BoundedVisualServo(); t=1_500_000_000
    row=model.step(evidence(t),now_ns=t); assert row['stage']=='forward'
    t+=100_000_000; row=model.step(evidence(t,x=.4),now_ns=t); assert row['stage']=='forward_brake'
    for _ in range(10):
        t+=100_000_000; row=model.step(evidence(t,x=.4),now_ns=t)
    assert row['stage']=='turn' and row['terminal'] is None
    t+=100_000_000; row=model.step(evidence(t,x=.4),now_ns=t); assert row['requested_command'][2]==.25
    t+=100_000_000; row=model.step(evidence(t,x=.4,yaw=.3),now_ns=t); assert row['stage']=='final_brake'
    for _ in range(10):
        t+=100_000_000; row=model.step(evidence(t,x=.4,yaw=.3),now_ns=t)
    assert row['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE' and row['requested_command']==[0.,0.,0.]
    assert not row['terrain_safety_qualified'] and not row['navigation_qualified']


@pytest.mark.parametrize('fault',['stale','episode','mode','nan','reflection','missing'])
def test_bad_visual_evidence_stops_and_latches(fault):
    model=BoundedVisualServo(); t=1_500_000_000; e=evidence(t)
    if fault=='stale': e['current_pose']['measured_ns']-=1
    elif fault=='episode': e['identity']=(1,0,0)
    elif fault=='mode': e['current_pose']['mode']='joint'
    elif fault=='nan': e['current_pose']['position_initial_body_m'][0]=np.nan
    elif fault=='reflection': e['current_pose']['rotation_initial_body_from_current_body']=(-np.eye(3)).tolist()
    else: e['current_pose']=None
    row=model.step(e,now_ns=t)
    assert row['terminal']=='VISUAL_SERVO_FAILED' and row['requested_command']==[0.,0.,0.]
    assert model.step(evidence(t+100_000_000),now_ns=t+100_000_000)['terminal']=='VISUAL_SERVO_FAILED'


def test_timeout_does_not_turn_stationary_failure_into_arrival():
    model=BoundedVisualServo()
    for i in range(RULES['maximum_ticks']+1):
        t=1_500_000_000+i*100_000_000; row=model.step(evidence(t),now_ns=t)
    assert row['terminal']=='VISUAL_SERVO_FAILED' and row['reason']=='TIME_LIMIT'


def test_new_spawn_is_real_pack_change_and_friction_pair_matches():
    a=specification('nominal'); b=specification('lower_friction'); pa,pb=pack(a),pack(b)
    assert pa.robot==pb.robot and pa.robot.spawn_xyz_m==(-.65,.15,.375)
    assert pa.robot.spawn_quat_wxyz[3]==pytest.approx(np.sin(.06))
    assert a['geometry']==b['geometry'] and a['geometry']['spawn_se2_world']!=[-.5,-.3,0.]
    assert pa.physics_randomization.floor_friction_mu==1. and pb.physics_randomization.floor_friction_mu==.15
    assert a['controlled_continuous_level_floor'] and a['hidden_robot_ideal_camera']
    changed=deepcopy(a); changed['geometry']['spawn_se2_world'][0]=0
    with pytest.raises(ValueError): pack(changed)

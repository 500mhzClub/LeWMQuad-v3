import math
import pytest
from lewm.goal_region_pulse_servo_development import GoalRegionPulseServo,RULES
from lewm.tests.test_pulse_feedback_servo_development import step


def test_actual_goal_region_is_not_blocked_by_tighter_approach_threshold():
    model=GoalRegionPulseServo()
    for tick in range(31):row=step(model,tick,x=.364,y=-.022,yaw=.268)
    assert row['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE' and row['pulse_count']==0
    assert RULES['final_position_tolerance_m']==.06 and RULES['final_yaw_tolerance_rad']==.05


def test_inside_position_region_targets_final_yaw_not_bearing_to_centre():
    model=GoalRegionPulseServo()
    for tick in range(21):row=step(model,tick,x=.365,y=.008,yaw=0.)
    assert row['task_stage']=='orient' and row['requested_command']==[0.,0.,.45]
    assert 'forward_steering_error_rad' not in row['diagnostic']


def test_outside_position_region_still_requires_position_correction():
    model=GoalRegionPulseServo()
    for tick in range(21):row=step(model,tick,x=.33,yaw=.3)
    assert row['task_stage']=='approach' and row['terminal'] is None


def test_complete_responsive_plant_with_same_final_hold():
    model=GoalRegionPulseServo();x=y=yaw=0.
    for tick in range(1001):
        row=step(model,tick,x=x,y=y,yaw=yaw)
        if row['terminal']:break
        u,_,w=row['requested_command'];x+=u*.1*math.cos(yaw);y+=u*.1*math.sin(yaw);yaw+=w*.1
    assert row['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'
    assert math.hypot(.4-x,y)<=.06 and abs(.3-yaw)<=.05 and row['quiet_intervals']==10

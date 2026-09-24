import copy
import math

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.simulated_body_observation_development import (
    JOINT_NAMES, IdealBodySensor, BodyObservationBuffer, validate_policy_packet,
)


def sample(sensor,time,**changes):
    values=dict(measured_ns=time,quaternion_xyzw=[0,0,0,1],velocity_world=[0,0,0],
        angular_velocity_world=[0,0,0],joint_position=np.arange(12)*.01,
        joint_velocity=np.zeros(12),joint_names=JOINT_NAMES)
    return sensor.sample(**(values | changes))


def test_stationary_sensor_has_upward_specific_force_and_invalid_first_acceleration():
    sensor=IdealBodySensor()
    first=sample(sensor,20_000_000)
    assert not first['specific_force'][1].any()
    second=sample(sensor,40_000_000)
    assert second['specific_force'][0]==pytest.approx([0,0,9.81])
    assert second['specific_force'][1].all()


def test_freefall_has_zero_specific_force():
    sensor=IdealBodySensor()
    sample(sensor,20_000_000)
    assert sample(sensor,40_000_000,velocity_world=[0,0,-9.81*.02])['specific_force'][0]==pytest.approx([0,0,0])


def test_body_frame_rotation_and_joint_order_are_preserved():
    sensor=IdealBodySensor()
    value=sample(sensor,20_000_000,quaternion_xyzw=[0,0,math.sin(math.pi/4),math.cos(math.pi/4)],
                 angular_velocity_world=[1,0,0])
    assert value['gyro'][0]==pytest.approx([0,-1,0],abs=1e-12)
    assert value['joints'][0][:12]==pytest.approx(np.arange(12)*.01)


@pytest.mark.parametrize('time',[20_000_000,39_000_000,60_000_000])
def test_bad_cadence_cannot_be_silently_differentiated(time):
    sensor=IdealBodySensor()
    sample(sensor,20_000_000)
    with pytest.raises(SensorContractError,match='cadence'):
        sample(sensor,time)


def test_wrong_joint_names_fail_before_using_values():
    with pytest.raises(SensorContractError,match='joint identity'):
        sample(IdealBodySensor(),20_000_000,joint_names=JOINT_NAMES[::-1])


def packet():
    buffer=BodyObservationBuffer((0,0,0))
    sensor=IdealBodySensor()
    for tick in range(1,26):
        buffer.append_sensors(sample(sensor,tick*20_000_000),tick*20_000_000)
        if tick%5==0:
            buffer.append_applied_command([0,0,0],tick*20_000_000)
    return buffer.packet(np.zeros((480,640,3),dtype=np.uint8),500_000_000)


def test_packet_contains_only_declared_causal_history_and_rgb():
    result=packet()
    validate_policy_packet(result)
    state=result['sensor_state']
    assert state['sensed']['gyro']['values'].shape==(20,3)
    assert state['sensed']['joints']['values'].shape==(20,24)
    assert state['sensed']['gyro']['measured_ns'].tolist()==list(range(120_000_000,500_000_001,20_000_000))
    assert state['control']['applied_command']['valid'].sum()==15


@pytest.mark.parametrize('fault',['pose','image_pose','future_contact','future_time','channel_alias','nonfinite_time','invalid_nan','duplicate','late_image'])
def test_policy_boundary_rejects_privilege_and_causality_corruption(fault):
    value=copy.deepcopy(packet())
    state=value['sensor_state']
    if fault=='pose': value['world_pose']=[0]*7
    elif fault=='image_pose': value['image']['world_from_optical']=np.eye(4)
    elif fault=='future_contact': state['sensed']['contact_outcome']=True
    elif fault=='future_time': state['sensed']['gyro']['available_ns'][-1]+=1
    elif fault=='channel_alias': state['sensed']['gyro']['channels']=('global_x','global_y','global_z')
    elif fault=='nonfinite_time': state['decision_ns']=float('nan')
    elif fault=='invalid_nan': state['control']['applied_command']['values'][0,0]=float('nan')
    elif fault=='duplicate': state['sensed']['gyro']['measured_ns'][-1]=state['sensed']['gyro']['measured_ns'][-2]
    elif fault=='late_image': value['image']['available_ns']+=1
    with pytest.raises(SensorContractError): validate_policy_packet(value)

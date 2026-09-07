import copy
import math

import numpy as np
import pytest

from lewm.relative_gyro_turn_development import RelativeGyroTurn,rotation_increment
from lewm.simulated_body_observation_development import BodyObservationBuffer


def append(buffer,step,gyro):
    ns=step*20_000_000
    buffer.append_sensors({'gyro':(np.array(gyro),np.ones(3,dtype=bool)),
        'specific_force':(np.array([0.,0.,9.81]),np.ones(3,dtype=bool)),
        'joints':(np.zeros(24),np.ones(24,dtype=bool))},ns)
    if step%5==0: buffer.append_applied_command([0.,0.,0.],ns)


def packet(buffer,step): return buffer.packet(np.zeros((480,640,3),dtype=np.uint8),step*20_000_000)


def initialized(gyro=(0.,0.,0.)):
    buffer=BodyObservationBuffer((0,0,0))
    for step in range(1,81): append(buffer,step,gyro)
    return buffer


def test_rotation_increment_handedness_and_orthogonality():
    rotation=rotation_increment([0,0,math.pi/2])
    assert rotation@[1,0,0]==pytest.approx([0,1,0],abs=1e-12)
    for vector in ([1e-9,2e-9,3e-9],[.1,-.2,.3],[0,0,0]):
        rotation=rotation_increment(vector)
        assert rotation.T@rotation==pytest.approx(np.eye(3),abs=1e-12)
        assert np.linalg.det(rotation)==pytest.approx(1.)


@pytest.mark.parametrize('target',[math.pi/2,-math.pi/2,math.pi])
def test_ideal_command_following_reaches_a_relative_turn_and_holds(target):
    buffer=initialized(); controller=RelativeGyroTurn()
    command=controller.begin(packet(buffer,80),target)['requested_command']
    for tick in range(1,121):
        for step in range(80+(tick-1)*5+1,80+tick*5+1): append(buffer,step,[0.,0.,command[2]])
        result=controller.step(packet(buffer,80+tick*5)); command=result['requested_command']
        assert command[:2]==[0.,0.] and abs(command[2])<=.35
        if result['status']=='COMPLETE': break
    assert result['status']=='COMPLETE' and abs(result['heading_error_rad'])<=.08
    assert controller.step(packet(buffer,80+tick*5))['requested_command']==[0,0,0]


def test_constant_body_rate_uses_all_new_samples_exactly_once():
    buffer=initialized((.1,.2,.3)); controller=RelativeGyroTurn()
    controller.begin(packet(buffer,80),1.)
    for tick in range(1,11):
        for step in range(80+(tick-1)*5+1,80+tick*5+1): append(buffer,step,[.1,.2,.3])
        controller.step(packet(buffer,80+tick*5))
    assert controller.rotation==pytest.approx(rotation_increment([.1,.2,.3]),abs=1e-12)


@pytest.mark.parametrize('fault',['identity','invalid','gap','rewritten','rewritten_validity','privilege'])
def test_bad_input_fails_closed_without_partial_orientation(fault):
    buffer=initialized(); controller=RelativeGyroTurn(); controller.begin(packet(buffer,80),1.)
    for step in range(81,86): append(buffer,step,[0.,0.,.2])
    value=packet(buffer,85)
    if fault=='identity': value['sensor_state']['identity']=(0,0,1)
    elif fault=='invalid': value['sensor_state']['sensed']['gyro']['valid'][-3]=False; value['sensor_state']['sensed']['gyro']['values'][-3]=0
    elif fault=='gap': value['sensor_state']['sensed']['gyro']['measured_ns'][-3]+=1
    elif fault=='rewritten': value['sensor_state']['sensed']['gyro']['values'][-6,2]=1.
    elif fault=='rewritten_validity': value['sensor_state']['sensed']['gyro']['valid'][-6]=False
    elif fault=='privilege': value['world_yaw']=0.
    before=controller.rotation.copy(); result=controller.step(value)
    assert result['status']=='FAILED_SENSOR' and result['requested_command']==[0,0,0]
    assert np.array_equal(controller.rotation,before)


def test_stationary_unresponsive_turn_times_out():
    buffer=initialized(); controller=RelativeGyroTurn(); controller.begin(packet(buffer,80),1.)
    for tick in range(1,121):
        for step in range(80+(tick-1)*5+1,80+tick*5+1): append(buffer,step,[0,0,0])
        result=controller.step(packet(buffer,80+tick*5))
    assert result['status']=='FAILED_TIMEOUT' and result['requested_command']==[0,0,0]


def test_entering_tolerance_at_deadline_does_not_extend_budget():
    buffer=initialized(); controller=RelativeGyroTurn(); controller.begin(packet(buffer,80),1.)
    controller.rotation=rotation_increment([0,0,1.]); controller.last_gyro=np.zeros(3)
    result=controller._command(controller.start_ns+12_000_000_000)
    assert result['status']=='FAILED_TIMEOUT'

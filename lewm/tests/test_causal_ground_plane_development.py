import math

import numpy as np
import pytest

from lewm.causal_ground_plane_development import CausalGroundPlane,foot_sphere_centres_body
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_relative_gyro_turn_development import initialized,append,packet


def test_zero_angle_robot_geometry_and_joint_order():
    feet=foot_sphere_centres_body(np.zeros(12))
    assert feet==pytest.approx(np.array([[.1914,.142,-.426],[.1914,-.142,-.426],[-.1954,.142,-.426],[-.1954,-.142,-.426]]))
    q=np.zeros(12); q[4]=math.pi/2; feet=foot_sphere_centres_body(q)
    assert feet[0]==pytest.approx([.1934-.426,.142,.002])
    assert feet[1]==pytest.approx([.1914,-.142,-.426])


def example():
    buffer=initialized(); p=packet(buffer,80)
    p['sensor_state']['sensed']['specific_force']['values'][:]=[0,0,9.81]
    p['sensor_state']['sensed']['joints']['values'][:,:12]=0.
    return buffer,p


def test_sensor_initialized_plane_and_pixel_projection():
    _,p=example(); g=CausalGroundPlane(); result=g.begin(p,now_ns=1_600_000_000)
    assert result['body_origin_height_m']==pytest.approx(.448) and not result['ground_plane_qualified']
    point=g.project_pixel(319.5,400,now_ns=1_600_000_000)
    assert point[2]==pytest.approx(-.448) and point[1]==pytest.approx(0) and point[0]>.326
    assert g.project_pixel(319.5,200,now_ns=1_600_000_000) is None


def test_full_relative_attitude_rotates_gravity_without_world_pose():
    buffer,p=example(); g=CausalGroundPlane(); g.begin(p,now_ns=1_600_000_000)
    for i in range(81,86): append(buffer,i,[0,.2,0])
    next_packet=packet(buffer,85); next_packet['sensor_state']['sensed']['joints']['values'][:,:12]=0.
    state=g.step(next_packet,now_ns=1_700_000_000)
    assert state['up_current_body'][0]<0 and np.linalg.norm(state['up_current_body'])==pytest.approx(1.)


@pytest.mark.parametrize('fault',['command','force','force_invalid','privilege'])
def test_bad_initial_ground_assumption_rejected(fault):
    _,p=example()
    if fault=='command': p['sensor_state']['control']['applied_command']['values'][-1,0]=.2
    if fault=='force': p['sensor_state']['sensed']['specific_force']['values'][:]=0
    if fault=='force_invalid':
        p['sensor_state']['sensed']['specific_force']['valid'][-1]=False
        p['sensor_state']['sensed']['specific_force']['values'][-1]=0
    if fault=='privilege': p['world_height']=.3
    g=CausalGroundPlane()
    with pytest.raises(SensorContractError): g.begin(p,now_ns=1_600_000_000)
    assert g.status=='FAILED_SENSOR'


def test_faulted_update_and_stale_query_cannot_reuse_plane():
    _,p=example(); g=CausalGroundPlane(); g.begin(p,now_ns=1_600_000_000)
    with pytest.raises(SensorContractError): g.snapshot(now_ns=1_700_000_000)
    with pytest.raises(SensorContractError): g.step(p,now_ns=1_600_000_000)
    with pytest.raises(SensorContractError): g.snapshot(now_ns=1_600_000_000)


@pytest.mark.parametrize('bad',[np.zeros(11),np.full(12,float('nan'))])
def test_bad_joint_geometry_input_rejected(bad):
    with pytest.raises(ValueError): foot_sphere_centres_body(bad)

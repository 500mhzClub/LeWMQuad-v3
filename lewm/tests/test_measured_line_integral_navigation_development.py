import math
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.measured_line_integral_navigation_development import (
    BoundedIntegralAlignment, MeasuredLineTraversal, MeasuredLineIntegralNavigation,
    approach_line_command)
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_observed_turn_region_development import frame
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def alignment_packet(tick, heading, rate=0.):
    now = 1_500_000_000+tick*100_000_000
    c, s = math.cos(heading), math.sin(heading)
    packet = {'sensor_state': {'decision_ns': now, 'sensed': {
        'gyro': {'values': np.array([[0., 0., rate]])}}}}
    attitude = {'decision_ns': now, 'rotation_initial_body_from_current_body':
                [[c,-s,0.], [s,c,0.], [0.,0.,1.]]}
    return packet, attitude, now


@pytest.mark.parametrize('target', [-.24, .24])
def test_integral_overcomes_synthetic_deadzone_with_original_acceptance(target):
    controller = BoundedIntegralAlignment([math.cos(target), math.sin(target), 0.])
    heading = rate = 0.
    for tick in range(121):
        p,a,now = alignment_packet(tick, heading, rate)
        row = controller.observe(p,a,now_ns=now)
        assert abs(row['requested_command'][2]) <= .35
        assert abs(row['integral_command_rad_s']) <= .12
        if controller.terminal: break
        command = row['requested_command'][2]
        # Synthetic stress plant, not a fitted or validated Go2 dynamics model.
        rate = math.copysign(max(0., abs(command)-.04), command)
        heading += rate*.1
    assert row['status'] == 'COMPLETE'
    assert abs(row['heading_error_rad']) <= .02
    assert row['requested_command'] == [0.,0.,0.]


def test_no_physical_response_still_times_out_and_integral_is_bounded():
    controller = BoundedIntegralAlignment([math.cos(.1), math.sin(.1), 0.])
    for tick in range(121):
        p,a,now = alignment_packet(tick,0.)
        row = controller.observe(p,a,now_ns=now)
        assert abs(row['integral_command_rad_s']) <= .12
    assert row['status'] == 'FAILED_TIMEOUT'
    assert row['requested_command'] == [0.,0.,0.]


def test_quiet_reset_sign_change_and_bad_clock():
    controller = BoundedIntegralAlignment([1.,0.,0.])
    for tick, heading in enumerate([-.1, -.1, .1, .01]):
        p,a,now = alignment_packet(tick,heading)
        row = controller.observe(p,a,now_ns=now)
        if tick == 2: assert row['integral_command_rad_s'] == pytest.approx(-.004)
    assert row['integral_command_rad_s'] == 0.
    assert row['requested_command'] == [0.,0.,0.]
    with pytest.raises(SensorContractError): controller.observe(p,a,now_ns=now)


@pytest.mark.parametrize('x', [0., 1., 1.55, 1.57, 2.])
def test_line_guidance_does_not_sharpen_near_endpoint(x):
    row = approach_line_command([x,.03,0.], np.eye(3), [0.,0.,0.], [1.,0.,0.], [0.,0.,1.])
    assert row['yaw_command_rad_s'] == pytest.approx(-1.5*math.atan(.03/.5))
    assert row['cross_track_m'] == pytest.approx(.03)


def test_line_is_rigid_rotation_invariant_and_rejects_missing_motion():
    rotation = np.array([[0.,-1.,0.], [1.,0.,0.], [0.,0.,1.]])
    a = approach_line_command([1.,.03,0.], np.eye(3), [0.,0.,0.], [1.,0.,0.], [0.,0.,1.])
    b = approach_line_command([-.03,1.,0.], rotation, [0.,0.,0.], [0.,1.,0.], [0.,0.,1.])
    assert a == b
    with pytest.raises(SensorContractError):
        approach_line_command(None, rotation, [0.,0.,0.], [0.,1.,0.], [0.,0.,1.])


def test_successor_child_retains_measured_arrival_and_zero_release():
    geometry=ArticulatedCollisionGeometry(URDF); stream=Stream()
    observations=SimpleNamespace(memory=SimpleNamespace(last_ns=None, position=np.zeros(3),
        rotation=np.eye(3), up_initial=np.array([0.,0.,1.])), relative=None)
    observations.target=lambda *a,**k: {'target_initial_body_m':[1.,0.,0.], 'kind':'SYNTHETIC_MEASURED_TARGET'}
    observations.clearance=lambda *a,**k: {'all_samples_supported':True, 'unknown_samples':0,
        'sample_points':100, 'future_gait_qualified':False, 'continuous_volume_qualified':False}
    child=MeasuredLineTraversal(geometry, observations)
    for tick in range(30):
        p,fast,now=stream.frame(tick); observations.memory.last_ns=now
        observations.memory.position=np.array([min(1.,max(0.,tick-3)*.1),0.,0.])
        observations.relative={'motion':{'translation_previous_body_m':[.1 if 3<tick<=13 else 0.,0.,0.]}}
        row=child.observe(p,fast,now_ns=now)
        if row['terminal']: break
    assert row['status']=='ARRIVAL_CANDIDATE'
    assert row['requested_command']==[0.,0.,0.]
    assert child.ledger.snapshot()['arrival']['qualified_arrival'] is False


def test_navigation_still_fails_closed_on_unobserved_translation():
    controller=MeasuredLineIntegralNavigation('measured_line_integral', ArticulatedCollisionGeometry(URDF), memory_arm='episodic')
    p,fast,now=Stream().frame(0); _,d,r,_=frame(Stream(),0)
    r['position_initial_body_m']=None
    with pytest.raises(SensorContractError): controller.observe_rgbd(p,fast,d,r,now_ns=now)
    assert controller.status=='FAILED_SENSOR'


def test_navigation_installs_new_operators_before_their_first_observation():
    controller=MeasuredLineIntegralNavigation('measured_line_integral', ArticulatedCollisionGeometry(URDF), memory_arm='episodic')
    depth_stream=Stream(); body_stream=Stream()
    for tick in range(19):
        p,d,r,now=frame(depth_stream,tick)
        _,fast,_=body_stream.frame(tick)
        controller.observe_rgbd(p,fast,d,r,now_ns=now)
    assert controller.stage=='TRAVERSE'
    assert isinstance(controller.child,MeasuredLineTraversal)
    assert controller.child.tick==-1 and controller.children[-1] is controller.child
    # Synthetic stage setup tests construction only, not physical traversal.
    controller.stage='HOLD_ALIGN'; controller.hold_since=now-1_500_000_000
    controller.selected={'direction_initial_body':[1.,0.,0.]}
    p,d,r,now=frame(depth_stream,19); _,fast,_=body_stream.frame(19)
    controller.observe_rgbd(p,fast,d,r,now_ns=now)
    assert isinstance(controller.alignment,BoundedIntegralAlignment)
    assert controller.alignment.start_ns is None

import math

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.observable_hold_navigation_development import HeadingServo, HoldingAlignment, ObservableHoldNavigation
from lewm.observable_approach_development import ObservableApproachTraversal
from lewm.tests.test_release_aware_navigation_development import packet
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_observed_turn_region_development import frame
from scripts.analyze_go2_ground_plane_development_v1 import URDF


@pytest.mark.parametrize('bias',[-.03,.03])
def test_alignment_transfers_active_integral_through_biased_hold(bias):
    target=.12; c=HoldingAlignment([math.cos(target),math.sin(target),0.])
    heading=rate=0.; completion=None
    for tick in range(121):
        p,a,now=packet(tick,heading,rate)
        row=c.observe(p,a,now_ns=now) if completion is None else c.servo.observe(p,a,now_ns=now)
        if row.get('status')=='COMPLETE': completion=tick
        assert abs(row['requested_command'][2])<=.35
        rate=row['requested_command'][2]+bias; heading+=rate*.1
        if completion is not None and tick>=completion+15: break
    assert completion is not None
    assert abs(target-heading)<.02
    assert row['holding_contract']=='active_measured_heading_not_zero_command_release'


def test_no_response_still_times_out_with_zero():
    c=HoldingAlignment([0.,1.,0.])
    for tick in range(121):
        p,a,now=packet(tick,0.); row=c.observe(p,a,now_ns=now)
    assert row['status']=='FAILED_TIMEOUT' and row['requested_command']==[0.,0.,0.]


def test_hold_clock_and_invalid_gyro_fail():
    c=HeadingServo([1.,0.,0.]); p,a,now=packet(0,0.); c.observe(p,a,now_ns=now)
    with pytest.raises(SensorContractError): c.observe(p,a,now_ns=now)
    p,a,now=packet(1,0.); p['sensor_state']['sensed']['gyro']['valid'][0,0]=False
    with pytest.raises(SensorContractError): c.observe(p,a,now_ns=now)


def test_wrapper_constructs_only_fresh_operators_and_retains_fault_zero():
    c=ObservableHoldNavigation('observable_hold',ArticulatedCollisionGeometry(URDF),memory_arm='episodic')
    ds=Stream(); bs=Stream()
    for tick in range(19):
        p,d,r,now=frame(ds,tick); _,fast,_=bs.frame(tick)
        c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert isinstance(c.child,ObservableApproachTraversal) and c.child.tick==-1
    c.stage='HOLD_ALIGN'; c.hold_since=now-1_500_000_000; c.selected={'direction_initial_body':[1.,0.,0.]}
    p,d,r,now=frame(ds,19); _,fast,_=bs.frame(19)
    c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert isinstance(c.alignment,HoldingAlignment) and c.alignment.start_ns is None
    p,d,r,now=frame(ds,20); _,fast,_=bs.frame(20); r['position_initial_body_m']=None
    with pytest.raises(SensorContractError): c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert c.status=='FAILED_SENSOR'

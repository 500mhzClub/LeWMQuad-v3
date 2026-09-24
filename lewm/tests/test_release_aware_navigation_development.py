import math

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.release_aware_navigation_development import ReleaseAwareAlignment, ReleaseAwareNavigation
from lewm.measured_line_integral_navigation_development import MeasuredLineTraversal
from lewm.tests.test_measured_line_integral_navigation_development import alignment_packet
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_observed_turn_region_development import frame
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def packet(tick, heading, rate=0.):
    p,a,now=alignment_packet(tick,heading,rate)
    p['sensor_state']['sensed']['gyro']['valid']=np.ones((1,3),bool)
    return p,a,now


def test_outer_tolerance_does_not_trigger_early_release_or_completion():
    c=ReleaseAwareAlignment([1.,0.,0.])
    for tick in range(8):
        p,a,now=packet(tick,-.019)
        row=c.observe(p,a,now_ns=now)
        assert row['phase']=='APPROACH' and row['status']=='ALIGNING'
        assert row['requested_command'][2]>0.


def test_inner_entry_needs_actual_release_intervals_and_unchanged_outer_dwell():
    c=ReleaseAwareAlignment([1.,0.,0.])
    for tick in range(6):
        p,a,now=packet(tick,-.004 if tick==0 else -.012)
        row=c.observe(p,a,now_ns=now)
        assert row['requested_command']==[0.,0.,0.]
        assert row['status']==('COMPLETE' if tick==5 else 'ALIGNING')
    assert row['release_attempts']==1 and row['heading_error_rad']==pytest.approx(.012)


@pytest.mark.parametrize('target,delay', [(.12,0),(-.12,0),(.12,2)])
def test_synthetic_deadzone_and_release_recoil_require_measured_settle(target,delay):
    c=ReleaseAwareAlignment([math.cos(target),math.sin(target),0.])
    heading=rate=previous=0.; pending=[0.]*delay
    releases=0
    for tick in range(121):
        p,a,now=packet(tick,heading,rate); row=c.observe(p,a,now_ns=now)
        if c.terminal: break
        command=row['requested_command'][2]
        pending.append(command); effective=pending.pop(0)
        rate=math.copysign(max(0.,abs(effective)-.04),effective)
        if effective==0. and previous!=0.:
            # Stress fixture, not a fitted Go2 release dynamics model.
            heading-=math.copysign(.008,previous); releases+=1
        heading+=.1*rate; previous=effective
        assert abs(command)<=.35 and abs(row['integral_command_rad_s'])<=.12
    assert row['status']=='COMPLETE'
    assert releases>=1 and abs(row['heading_error_rad'])<=.02
    assert row['requested_command']==[0.,0.,0.]


def test_large_release_error_recorrects_without_resetting_deadline():
    c=ReleaseAwareAlignment([1.,0.,0.])
    for tick in range(121):
        # Adversarial plant always jumps outside tolerance upon release.
        heading=-.004 if c.phase=='APPROACH' else -.05
        p,a,now=packet(tick,heading); row=c.observe(p,a,now_ns=now)
    assert row['status']=='FAILED_TIMEOUT' and row['requested_command']==[0.,0.,0.]
    assert c.start_ns==1_500_000_000 and c.release_attempts>1


def test_no_response_and_saturated_integral_still_fail():
    c=ReleaseAwareAlignment([0.,1.,0.])
    for tick in range(121):
        p,a,now=packet(tick,0.); row=c.observe(p,a,now_ns=now)
        assert row['integral_command_rad_s']==0.
    assert row['status']=='FAILED_TIMEOUT'


@pytest.mark.parametrize('bad', ['clock','gyro','rotation','nan'])
def test_invalid_observations_latch_fault(bad):
    c=ReleaseAwareAlignment([1.,0.,0.]); p,a,now=packet(0,0.)
    c.observe(p,a,now_ns=now); p,a,now=packet(1,0.)
    if bad=='clock': now+=100_000_000
    elif bad=='gyro': p['sensor_state']['sensed']['gyro']['valid'][0,0]=False
    elif bad=='rotation': a['rotation_initial_body_from_current_body'][0][0]=2.
    else: p['sensor_state']['sensed']['gyro']['values'][0,0]=np.nan
    with pytest.raises(SensorContractError): c.observe(p,a,now_ns=now)
    p,a,now=packet(1,0.)
    with pytest.raises(SensorContractError): c.observe(p,a,now_ns=now)


def test_full_wrapper_keeps_line_operator_and_installs_unstarted_release_alignment():
    c=ReleaseAwareNavigation('release_aware',ArticulatedCollisionGeometry(URDF),memory_arm='episodic')
    ds=Stream(); bs=Stream()
    for tick in range(19):
        p,d,r,now=frame(ds,tick); _,fast,_=bs.frame(tick)
        c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert isinstance(c.child,MeasuredLineTraversal) and c.child.tick==-1
    c.stage='HOLD_ALIGN'; c.hold_since=now-1_500_000_000
    c.selected={'direction_initial_body':[1.,0.,0.]}
    p,d,r,now=frame(ds,19); _,fast,_=bs.frame(19)
    c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert isinstance(c.alignment,ReleaseAwareAlignment) and c.alignment.start_ns is None
    p,d,r,now=frame(ds,20); _,fast,_=bs.frame(20)
    c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert c.alignment.start_ns==now

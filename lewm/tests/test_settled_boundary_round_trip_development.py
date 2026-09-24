from copy import deepcopy
import pytest
from lewm.settled_boundary_round_trip_development import SettledBoundaryRoundTripMission


def mission():
    return SettledBoundaryRoundTripMission(dict(goal_initial_body_xy_m=[.2,0.],
        return_initial_body_xy_m=[0.,0.],require_return_after_goal=True),navigation_ticks=60)


def step(m,i,p,command=(0.,0.,0.)):
    return m.advance(p,frame=i,now_ns=1_500_000_000+i*100_000_000,previous_requested_command=command)


def test_first_low_motion_observation_establishes_boundary_then_ten_intervals():
    m=mission()
    for i in range(3):step(m,i,[0.,0.,0.])
    step(m,3,[.18,0.,0.],(.2,0.,0.));step(m,4,[.19,0.,0.])
    r=step(m,5,[.19,0.,0.])
    assert r['quiet_intervals']==0 and r['observed_settling']['measured_motion_quiet']
    assert not r['observed_settling']['previous_boundary_measured_quiet']
    for i in range(6,15):
        r=step(m,i,[.19,0.,0.]);assert not r['arrivals']
        assert r['quiet_intervals']==i-5
    r=step(m,15,[.19,0.,0.])
    assert r['phase']=='RETURN' and r['arrivals'][0]['quiet_intervals']==10
    assert r['observed_settling']['both_boundaries_measured_quiet']


def test_motion_after_partial_dwell_requires_new_boundary():
    m=mission()
    for i in range(8):step(m,i,[.2,0.,0.])
    assert step(m,8,[.2,0.,.01])['quiet_intervals']==0
    assert step(m,9,[.2,0.,.01])['quiet_intervals']==0
    assert step(m,10,[.2,0.,.01])['quiet_intervals']==1


def test_return_dwell_cannot_reuse_outbound_window():
    m=mission()
    for i in range(13):r=step(m,i,[.2,0.,0.])
    assert r['phase']=='RETURN' and len(r['arrivals'])==1
    assert step(m,13,[0.,0.,0.],(.2,0.,0.))['quiet_intervals']==0
    assert step(m,14,[0.,0.,0.])['quiet_intervals']==0
    for i in range(15,25):r=step(m,i,[0.,0.,0.])
    assert r['terminal']=='OBSERVED_ROUND_TRIP_CANDIDATE' and len(r['arrivals'])==2


def test_failed_input_latches_with_no_future_settling_evidence():
    m=mission();step(m,0,[0.,0.,0.]);r=step(m,2,[.2,0.,0.])
    assert r['terminal']=='SENSOR_OR_MISSION_FAILURE'
    assert not m.previous_motion_quiet
    assert step(m,3,[.2,0.,0.])==r


def test_stationary_interior_motion_spike_is_explicitly_not_certified():
    # Equal sampled endpoints cannot reveal a between-frame excursion.
    m=mission();step(m,0,[.2,0.,0.]);r=step(m,1,[.2,0.,0.])
    assert r['observed_settling']['measured_motion_quiet']
    assert not r['observed_settling']['continuous_speed_bound']
    assert not r['verified_round_trip']


def test_controller_constructor_installs_boundary_mission_without_replacing_memory():
    from lewm.settled_boundary_round_trip_development import SettledBoundaryRoundTripController
    from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMap,LaterResolvedFloorMemory
    from lewm.overlap_retention_joint_observer_development import OverlapRetentionVisualLedMotion
    m=mission()
    c=SettledBoundaryRoundTripController(None,None,public_mission=dict(goal_initial_body_xy_m=m.outbound.tolist(),
        return_initial_body_xy_m=m.home.tolist(),require_return_after_goal=True),navigation_ticks=60,
        condition='jepa',variant='full',persistent=True)
    assert isinstance(c.mission,SettledBoundaryRoundTripMission)
    assert isinstance(c.mapper,LaterResolvedFloorMap) and isinstance(c.memory,LaterResolvedFloorMemory)
    assert c.memory is c.mapper.surface and isinstance(c.motion,OverlapRetentionVisualLedMotion)
    assert c.residual is c.selector.residual

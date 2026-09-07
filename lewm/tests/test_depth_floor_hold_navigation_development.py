from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_floor_hold_navigation_development import ObservedDepthFloor, DepthFloorHoldNavigation
from lewm.observable_approach_development import floor_view_standoff
from lewm.ground_projection_envelope_development import observe_ground_envelope
from lewm.tests.test_observable_approach_development import floor
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_observed_turn_region_development import frame
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def setup():
    p,_,now=Stream().frame(0); points,normals=floor(.32)
    regions=SimpleNamespace(memory=SimpleNamespace(last_ns=now),
        floor_view=floor_view_standoff(points,normals,[0.,0.,1.]))
    return p,regions,now


def test_measured_plane_accepts_real_nonzero_holding_history_without_rewriting_it():
    p,r,now=setup(); commands=p['sensor_state']['control']['applied_command']
    commands['values'][:,2]=.015; before=deepcopy(commands)
    state=ObservedDepthFloor(r).begin(p,now_ns=now)
    assert state['body_origin_height_m']==pytest.approx(.32)
    assert state['estimator_mode']=='current_observed_depth_floor'
    for key in commands: assert np.array_equal(commands[key],before[key])
    result=observe_ground_envelope(p,state,now_ns=now,height_radius=0.,angle_radius=0.)
    assert result['nominal_valid'].any() and not result['metric_clearance_qualified']


@pytest.mark.parametrize('bad',['missing','stale','unsupported','normal'])
def test_invalid_floor_is_not_replaced_by_height_or_zero_history(bad):
    p,r,now=setup()
    if bad=='missing': r.floor_view=None
    elif bad=='stale': r.memory.last_ns-=100_000_000
    elif bad=='unsupported': r.floor_view['floor_points']=1
    else: r.floor_view['floor_normal_body']=[0.,0.,2.]
    observer=ObservedDepthFloor(r)
    with pytest.raises(SensorContractError): observer.begin(p,now_ns=now)
    assert observer.status=='FAILED_SENSOR'


def test_adapter_only_installed_on_unstarted_scan_and_traversal():
    c=DepthFloorHoldNavigation('depth_floor_hold',ArticulatedCollisionGeometry(URDF),memory_arm='local_only')
    ds=Stream(); bs=Stream()
    for tick in range(19):
        p,d,r,now=frame(ds,tick); _,fast,_=bs.frame(tick)
        c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert isinstance(c.child.ground,ObservedDepthFloor) and c.child.ground.status=='NEW'
    c.stage='HOLD_SCAN'; c.hold_since=now-1_500_000_000
    p,d,r,now=frame(ds,19); _,fast,_=bs.frame(19)
    c.observe_rgbd(p,fast,d,r,now_ns=now)
    assert isinstance(c.ground,ObservedDepthFloor) and c.ground.status=='NEW'

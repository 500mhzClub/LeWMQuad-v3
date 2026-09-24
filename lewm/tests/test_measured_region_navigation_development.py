from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.measured_region_navigation_development import MeasuredRegionTraversal, RegionObservation, MeasuredRegionNavigation
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_observed_turn_region_development import frame
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def test_region_target_comes_from_observed_far_or_near_corner_not_fixed_travel():
    geometry=ArticulatedCollisionGeometry(URDF); obs=RegionObservation(geometry)
    p,d,r,now=frame(Stream(),0)
    obs.observe(p,d,r,now_ns=now)
    for role, sign in [('far_boundary',-1),('near_boundary',1)]:
        obs.corners=[{'point_initial_body_m':[2.,.7,0.],'approach_initial_body':[1.,0.,0.],
                      'longitudinal_role':role,'observed_ns':now,'side':'left'}]
        target=obs.target(p,now_ns=now)
        assert target['kind']=='OBSERVED_EXTERIOR_CORNER'
        assert target['target_body_m'][0]==pytest.approx(2.+sign*(target['nominal_radius_m']+.10))
        obs.corners[0]['point_initial_body_m'][0]+=.3
        shifted=obs.target(p,now_ns=now)
        assert shifted['target_body_m'][0]-target['target_body_m'][0]==pytest.approx(.3)


def test_measured_arrival_does_not_need_global_image_change_or_command_integration():
    geometry=ArticulatedCollisionGeometry(URDF); stream=Stream()
    observations=SimpleNamespace(memory=SimpleNamespace(last_ns=None,position=np.zeros(3),rotation=np.eye(3)),relative=None)
    observations.target=lambda *a,**k: {'target_initial_body_m':[1.,0.,0.],'kind':'SYNTHETIC_MEASURED_TARGET'}
    observations.clearance=lambda *a,**k: {'all_samples_supported':True,'unknown_samples':0,'sample_points':100,
                                        'future_gait_qualified':False,'continuous_volume_qualified':False}
    child=MeasuredRegionTraversal(geometry,observations)
    rows=[]
    for tick in range(30):
        p,fast,now=stream.frame(tick)
        observations.memory.last_ns=now
        observations.memory.position=np.array([min(1.,max(0.,tick-3)*.1),0.,0.])
        observations.relative={'motion':{'translation_previous_body_m':[.1 if 3<tick<=13 else 0.,0.,0.]}}
        row=child.observe(p,fast,now_ns=now); rows.append(row)
        if row['terminal']: break
    assert rows[-1]['status']=='ARRIVAL_CANDIDATE'
    assert rows[-1]['requested_command']==[0.,0.,0.]
    arrival=child.ledger.snapshot()['arrival']
    assert arrival['schema']=='sampled_nominal_turn_region_arrival.v1'
    assert 'floor_mask_change_fraction' not in arrival and 'command_progress_proxy_m' not in arrival
    assert not arrival['qualified_arrival']


def test_initial_selection_does_not_request_unobserved_yaw_alignment():
    controller=MeasuredRegionNavigation('measured_region',ArticulatedCollisionGeometry(URDF),memory_arm='episodic')
    controller._select({'direction_initial_body':[1.,0.,0.]},100)
    assert controller.stage=='HOLD_TRAVERSE'
    controller.stage='SCAN'; controller._select({'direction_initial_body':[0.,1.,0.]},200)
    assert controller.stage=='HOLD_ALIGN'


def test_measured_region_boundary_rejects_missing_observed_state_before_motion():
    controller=MeasuredRegionNavigation('measured_region',ArticulatedCollisionGeometry(URDF),memory_arm='episodic')
    stream=Stream(); p,fast,now=stream.frame(0); _,d,r,_=frame(Stream(),0)
    r['position_initial_body_m']=None
    with pytest.raises(SensorContractError): controller.observe_rgbd(p,fast,d,r,now_ns=now)
    assert controller.status=='FAILED_SENSOR'


def test_nominal_volume_cannot_fill_missing_joint_history_with_values():
    obs=RegionObservation(ArticulatedCollisionGeometry(URDF)); p,d,r,now=frame(Stream(),0)
    obs.observe(p,d,r,now_ns=now)
    p['sensor_state']['sensed']['joints']['valid'][0,0]=False
    with pytest.raises(SensorContractError): obs.volume(p)

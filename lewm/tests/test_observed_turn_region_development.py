from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_depth_observation_development import from_native_depth, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_local_surfaces_development import observe_local_surfaces
from lewm.observed_turn_region_development import RayEvidenceMemory, observed_corners, nominal_turn_volume, gravity_basis
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_observed_traversal_controller_development import Stream
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def frame(stream, tick, *, position=0., previous=0., floor=False, wall=2.):
    p, _, now = stream.frame(tick)
    native = np.full((480, 640), wall-position, np.float32)
    if floor:
        v = np.arange(480)+.5
        z = np.divide(.36, (v-240)/FOCAL, out=np.full(480, 200.), where=v > 240)
        native[:] = z[:, None]
    d = from_native_depth(native, p, measured_ns=now, available_ns=now, now_ns=now)
    relative = {'measured_ns': now, 'local_surfaces': observe_local_surfaces(d, p, now_ns=now),
        'relative_orientation': {'rotation_initial_body_from_current_body': np.eye(3).tolist(), 'decision_ns': now},
        'motion': None if tick == 0 else {'translation_previous_body_m': [position-previous, 0., 0.],
                                         'status': 'OBSERVED_TRANSLATION', 'rank': 3},
        'surface_points': 100, 'position_initial_body_m': [position, 0., 0.], 'position_is_observed_anchor': tick == 0,
        'arrival_verified': False, 'turn_clearance_qualified': False, 'scope': 'synthetic observed pose fixture'}
    return p, d, relative, now


def test_free_seen_points_unknown_back_volume_and_surface_conflict():
    m = RayEvidenceMemory(); p,d,r,now = frame(Stream(), 0); m.observe(p,d,r,now_ns=now)
    result = m.query([[1.,0.,0.],[-1.,0.,0.],[2.326,0.,0.],[3.,0.,0.]], np.zeros(4,bool), now_ns=now)
    assert result['free'].tolist() == [True,False,False,False]
    assert result['unknown_or_blocked'].tolist() == [False,True,True,True]
    assert result['contradictory_or_near_surface'].tolist() == [False,False,True,False]
    assert not result['continuous_volume_qualified'] and not result['future_gait_qualified']


def test_observed_motion_can_transport_old_free_rays_behind_current_camera():
    stream=Stream(); m=RayEvidenceMemory()
    for tick in range(11):
        p,d,r,now=frame(stream,tick,position=.1*tick,previous=.1*max(0,tick-1))
        m.observe(p,d,r,now_ns=now)
    result=m.query([[-.4,0.,0.]],np.zeros(1,bool),now_ns=now)
    assert result['all_samples_supported'] and len(m.frames)==11
    assert result['free'][0]


def test_new_observed_surface_overrides_old_free_evidence():
    stream=Stream(); m=RayEvidenceMemory()
    p,d,r,now=frame(stream,0); m.observe(p,d,r,now_ns=now)
    p,d,r,now=frame(stream,1,position=.1,wall=1.); m.observe(p,d,r,now_ns=now)
    result=m.query([[1.226,0.,0.]],np.zeros(1,bool),now_ns=now)
    assert result['contradictory_or_near_surface'][0] and not result['all_samples_supported']


def test_latest_stationary_view_can_contradict_old_keyframe():
    stream=Stream(); m=RayEvidenceMemory()
    p,d,r,now=frame(stream,0); m.observe(p,d,r,now_ns=now)
    p,d,r,now=frame(stream,1,wall=1.); m.observe(p,d,r,now_ns=now)
    assert len(m.frames)==1
    result=m.query([[1.326,0.,0.]],np.zeros(1,bool),now_ns=now)
    assert result['contradictory_or_near_surface'][0] and not result['all_samples_supported']


def test_floor_support_is_separate_from_free_and_never_allowed_for_torso():
    m=RayEvidenceMemory(); p,d,r,now=frame(Stream(),0,floor=True); m.observe(p,d,r,now_ns=now)
    result=m.query([[1.5,0.,-.317],[1.5,0.,-.317],[1.5,0.,-.42]],np.array([True,False,True]),now_ns=now)
    assert result['observed_ground_support'].tolist()==[True,False,False]
    assert not result['free'].any()


def test_missing_pose_or_rewritten_depth_fault_latches():
    stream=Stream(); m=RayEvidenceMemory(); p,d,r,now=frame(stream,0); m.observe(p,d,r,now_ns=now)
    p,d,r,now=frame(stream,1); r['position_initial_body_m']=None
    with pytest.raises(SensorContractError): m.observe(p,d,r,now_ns=now)
    with pytest.raises(SensorContractError): m.query([[1.,0.,0.]],np.zeros(1,bool),now_ns=now)
    m=RayEvidenceMemory(); p,d,r,now=frame(Stream(),0); d['depth_m'][0,0]=1.5
    with pytest.raises(SensorContractError): m.observe(p,d,r,now_ns=now)


def test_stationary_captures_do_not_evict_earliest_useful_view():
    stream=Stream(); m=RayEvidenceMemory()
    for tick in range(6):
        p,d,r,now=frame(stream,tick); m.observe(p,d,r,now_ns=now)
    assert len(m.frames)==1
    d['depth_m'][:]=.2
    assert m.frames[0]['depth'][0,0]==2.


def test_visible_corner_requires_two_nonparallel_nearby_supports():
    a={'normal_body_xy':[1.,0.],'offset_body_m':2.,'endpoints_body_xy_m':[[2.,1.5],[2.,.72]],
       'first_column':0,'last_column':30}
    b={'normal_body_xy':[0.,1.],'offset_body_m':.7,'endpoints_body_xy_m':[[2.03,.7],[3.,.7]],
       'first_column':32,'last_column':70}
    surface={'surface_segments':[a,b],'measured_ns':100,'sampled_columns':[30,32],
             'valid_columns':[True,True],'sampled_points_body_xy_m':[[2.,.72],[2.03,.7]]}
    corners=observed_corners(surface)
    assert len(corners)==1 and corners[0]['point_body_xy_m']==[2.,.7]
    assert not corners[0]['opening_extent_verified']
    assert corners[0]['exterior_side_corner'] and corners[0]['side']=='left'
    b['first_column']=100
    assert not observed_corners(surface)


def test_nominal_turn_profile_includes_all_shapes_but_does_not_claim_future_gait():
    model=ArticulatedCollisionGeometry(URDF)
    volume=nominal_turn_volume(model,np.zeros((1,12)),[0.,0.,1.])
    assert len(volume['points_body_m'])>100 and np.isfinite(volume['points_body_m']).all()
    support=model.supports(np.zeros(12),np.eye(3))
    assert min(b['height_m'] for b in volume['bands']) <= support['lower'][2]-.039
    assert max(b['height_m'] for b in volume['bands']) >= support['upper'][2]+.039
    assert not volume['future_gait_qualified'] and not volume['continuous_volume_qualified']
    np.testing.assert_allclose(gravity_basis([0.,0.,1.]),np.eye(3))

from copy import deepcopy

import cv2
import numpy as np
import pytest

from lewm.causal_depth_observation_development import FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.keyframe_rgbd_pose_development import (KeyframeHypotheses,KeyframeRGBDPose,
    point_radius,translation_radius,compose_radius,rotation_distance_bound,register)
from lewm.rgbd_correspondence_motion_development import project,solve_translation
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_rgbd_correspondence_motion_development import texture,packets,point_fixture


def test_direct_known_rotation_and_camera_lever_arm_match_previous_solver():
    a,b,ua,ub,R,t=point_fixture(); estimate,mask,_=register(a,b,ua,ub,R)
    np.testing.assert_allclose(estimate,t,atol=1e-12)
    np.testing.assert_allclose(estimate,solve_translation(a,b,ua,ub,R)['translation_previous_body_m'],atol=1e-12)
    assert mask.all()


def test_direct_reference_displacement_is_not_a_single_tick_increment():
    a,b,ua,ub,R,_=point_fixture(); t=np.array([.3,0,0]); b=(a-t)@R; ub,_=project(b)
    actual,_,_=register(a,b,ua,ub,R)
    np.testing.assert_allclose(actual,t,atol=1e-12)
    assert solve_translation(a,b,ua,ub,R)['translation_previous_body_m'] is None


@pytest.mark.parametrize('fault',['few','outliers','reprojection','coverage','rotation'])
def test_registration_rejects_unsupported_constraints(fault):
    a,b,ua,ub,R,_=point_fixture()
    if fault=='few': a,b,ua,ub=a[:3],b[:3],ua[:3],ub[:3]
    elif fault=='outliers': b[::2]+=[.1,.2,.3]
    elif fault=='reprojection': ub+=10
    elif fault=='coverage': ua[:]=80;ub[:]=80
    elif fault=='rotation': R*=2
    with pytest.raises(SensorContractError): register(a,b,ua,ub,R)


def test_point_error_box_corners_and_correlated_mean_are_enclosed():
    a,_,uv,_,_,_=point_fixture(); h=KeyframeHypotheses()
    radius=point_radius(a,uv,h); z=a[:,0]-.326
    for su in (-1,1):
        for sv in (-1,1):
            for sz in (-1,1):
                up=uv+[su*h.pixel_coordinate_error,sv*h.pixel_coordinate_error]; zp=z+sz*h.lifted_depth_error_m
                true=np.column_stack((zp+.326,-zp*(up[:,0]+.5-320)/FOCAL,-zp*(up[:,1]+.5-240)/FOCAL+.043))
                assert np.all(np.linalg.norm(true-a,axis=1)<=radius+1e-12)
    one=translation_radius(a,a,uv,uv,.01,h)
    repeated=translation_radius(np.repeat(a,4,axis=0),np.repeat(a,4,axis=0),np.repeat(uv,4,axis=0),np.repeat(uv,4,axis=0),.01,h)
    assert one==pytest.approx(repeated)  # Repeated/correlated data do not shrink the bound.


def test_composition_preserves_anchor_error_and_rotation_lever_arm():
    assert compose_radius(.07,.01,.02,[1.,0,0])==pytest.approx(.08+rotation_distance_bound(.02))
    assert compose_radius(.07,0.,0.,[0.,0,0])==.07
    R=rotation_increment([0,0,.02]); t=np.array([1.,0,0])
    assert np.linalg.norm(R@t-t)<=compose_radius(0,0,.02,t)+1e-12


def test_direct_static_registration_does_not_add_a_point_error_per_frame():
    rows=list(packets([texture()]*5)); model=KeyframeRGBDPose(KeyframeHypotheses(gyro_and_integration_error_rad_s=0.))
    outputs=[model.observe(*r[:3],now_ns=r[3]) for r in rows]
    assert len(model.nodes)==1 and all(not o['global_history_reset'] for o in outputs)
    radii=[o['conditional_global_position_radius_m'] for o in outputs[1:]]
    np.testing.assert_allclose(radii,np.full(4,radii[0]),atol=1e-12)
    assert radii[0]>0 and not outputs[-1]['navigation_qualified']


def test_promoted_keyframe_retains_global_pose_and_error():
    image=texture(); images=[cv2.warpAffine(image,np.float32([[1,0,i*3],[0,1,0]]),(640,480)) for i in range(4)]
    model=KeyframeRGBDPose(KeyframeHypotheses(promote_translation_m=.01,gyro_and_integration_error_rad_s=0.))
    outputs=[model.observe(*r[:3],now_ns=r[3]) for r in packets(images)]
    assert outputs[1]['promoted_keyframe'] and len(model.nodes)==4
    assert outputs[-1]['position_initial_body_m'][1]>.04
    assert outputs[-1]['conditional_global_position_radius_m']>outputs[1]['conditional_global_position_radius_m']
    assert [r['parent_frame'] for r in model.nodes]==[None,0,1,2]


@pytest.mark.parametrize('fault',['blank','clock','binding','privileged'])
def test_failure_is_terminal_without_reset_or_stale_pose(fault):
    rows=list(packets([texture()]*3)); model=KeyframeRGBDPose(); model.observe(*rows[0][:3],now_ns=rows[0][3])
    p,d,f,now=deepcopy(rows[1])
    if fault=='blank':
        import hashlib
        p['image']['rgb'][:]=128;d['rgb_sha256']=hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    elif fault=='clock': now+=100_000_000
    elif fault=='binding': p['image']['rgb'][0,0]^=255
    elif fault=='privileged': p['native_pose']=[0,0,0]
    with pytest.raises(SensorContractError): model.observe(p,d,f,now_ns=now)
    with pytest.raises(SensorContractError): model.observe(*rows[2][:3],now_ns=rows[2][3])
    assert model.failed and len(model.nodes)==1

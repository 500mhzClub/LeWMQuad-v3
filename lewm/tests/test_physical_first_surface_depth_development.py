"""Opaque visibility must not inherit the renderer's clipping blind spot."""
from dataclasses import replace
import numpy as np
import pytest
from lewm.physical_semantics import world_from_optical
from lewm.physical_first_surface_depth_development import expected_optical_depth,evaluate_visibility
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth as legacy
from lewm.longer_motion_collection_development import specification,pack
from lewm_genesis.near_field_rgbd_scene_development import floor_domain,RENDER_NEAR_M
from lewm_genesis.bounded_scene_builder_development import floor_domain as legacy_domain


def boxes():
    return [dict(wall_id=name,centre_xyz=[x,0.,6.],size_xyz=[.08,12.,12.],yaw_rad=0.)
        for name,x in (('front',.6),('rear',1.8))]


def transform(distance):return world_from_optical([.56-distance,0.,6.],[1.,0.,0.],[0.,0.,1.])


@pytest.mark.parametrize('distance',[.004,.006,.020,.049,.051,.199,.201,1.,4.9])
def test_first_opaque_surface_never_disappears_below_render_near(distance):
    ref=expected_optical_depth(boxes(),transform(distance))
    np.testing.assert_allclose(ref['expected_depth_m'],distance,rtol=0,atol=1e-14)
    assert (ref['object_index']==1).all() and ref['surface_interior'].all()
    actual=np.full((480,640),distance,np.float32)
    for near in (.005,.05):
        result=evaluate_visibility(actual,boxes(),transform(distance),render_near_m=near)
        assert result['within1mm']
        assert result['passes_sampled_physical_visibility']==(distance>near)


def test_legacy_reference_can_certify_seeing_through_near_wall_new_reference_rejects():
    T=transform(.02);actual=np.full((480,640),1.22,np.float32)
    old=legacy(boxes(),T,floor_z_m=0.)
    np.testing.assert_allclose(old['expected_depth_m'],1.22,atol=1e-12)
    result=evaluate_visibility(actual,boxes(),T,render_near_m=.05)
    assert not result['passes_sampled_physical_visibility']
    assert result['clipped_opaque_rays']==result['false_public_valid_near_rays']==4800
    assert result['maximum_error_m']>1.


@pytest.mark.parametrize('yaw',[np.pi/2,np.pi,-np.pi/2])
def test_reference_equivariant_under_scene_camera_rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    walls=boxes()
    for w in walls:w['centre_xyz']=(R@w['centre_xyz']).tolist();w['yaw_rad']=yaw
    T=transform(.02);T[:3,:]=R@T[:3,:]
    np.testing.assert_allclose(expected_optical_depth(walls,T)['expected_depth_m'],.02,atol=1e-14,rtol=0)


def test_parallel_nonintersecting_box_does_not_occlude():
    wall=boxes()[0];wall['centre_xyz'][1]=20
    ref=expected_optical_depth([wall],transform(.02))
    assert not (ref['object_index']==1).any()


@pytest.mark.parametrize('origin',[[.60,0,6],[.56,0,6],[0,0,0]])
def test_camera_in_solid_or_on_floor_rejected(origin):
    with pytest.raises(ValueError):expected_optical_depth(boxes(),world_from_optical(origin,[1,0,0],[0,0,1]))


def test_close_floor_is_not_discarded():
    T=world_from_optical([0,0,.02],[0,0,-1],[0,1,0])
    ref=expected_optical_depth([],T)
    np.testing.assert_allclose(ref['expected_depth_m'],.02,rtol=0,atol=1e-14)


@pytest.mark.parametrize('fault',['stride','rotation','size','near'])
def test_invalid_geometry_or_calibration_rejected(fault):
    T=transform(.02);walls=boxes()
    with pytest.raises(ValueError):
        if fault=='stride':expected_optical_depth(walls,T,stride=True)
        elif fault=='rotation':T[0,0]=3;expected_optical_depth(walls,T)
        elif fault=='size':walls[0]['size_xyz'][0]=0;expected_optical_depth(walls,T)
        else:evaluate_visibility(np.ones((480,640),np.float32),walls,T,render_near_m=np.nan)


def test_corrected_domain_retains_all_other_calibration_checks_and_original_pack():
    old=pack(specification('fit'));new=replace(old,camera=replace(old.camera,near_m=RENDER_NEAR_M))
    assert floor_domain(new)==legacy_domain(old)
    assert old.camera.near_m==.05 and new.camera.near_m==.005
    for field,value in (('near_m',.01),('far_m',100.),('fov_deg',80.)):
        with pytest.raises(ValueError):floor_domain(replace(new,camera=replace(new.camera,**{field:value})))


def test_new_reference_does_not_hide_bad_far_precision_or_missing_depth():
    for value in (4.902,np.nan,np.inf):
        result=evaluate_visibility(np.full((480,640),value,np.float32),boxes(),transform(4.9),render_near_m=.005)
        assert not result['passes_sampled_physical_visibility'] and not result['within1mm']

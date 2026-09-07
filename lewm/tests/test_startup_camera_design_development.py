import numpy as np
import pytest

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.startup_camera_design_development import (
    CANDIDATES,body_from_optical,checked_transform,floor_depth,classify_depth,project,footprint_visibility)


def down():
    T=body_from_optical('overhead_down90'); T[2,3]=.6
    return T


@pytest.mark.parametrize('name',[r[0] for r in CANDIDATES])
def test_fixed_proper_mount(name):
    checked_transform(body_from_optical(name))


def test_original_forward_mount():
    np.testing.assert_array_equal(body_from_optical('forward'),BODY_FROM_OPTICAL)
    with pytest.raises(ValueError): body_from_optical('adaptive')


@pytest.mark.parametrize('bad',[np.eye(3),np.full((4,4),np.nan),np.diag([1,1,-1,1]),np.diag([2,1,1,1])])
def test_invalid_transform(bad):
    with pytest.raises(ValueError): floor_depth(bad)


def test_floor_projection_pixel_centres():
    T=down(); depth,valid=floor_depth(T)
    assert valid.all(); np.testing.assert_allclose(depth,.6,atol=1e-14)
    uv,good=project(np.array([[0,0,0]]),T)
    assert good.all(); np.testing.assert_allclose(uv,[[319.5,239.5]],atol=1e-12)


def test_occlusion_missingness_and_background_wall():
    T=down(); d,_=floor_depth(T); visible=d.copy(); background=d.copy()
    visible[0,0]=.1; visible[0,1]=np.nan; visible[0,2]=0
    background[0,3]=.3; visible[0,3]=.3
    masks=classify_depth(visible,background,T)
    assert masks['self_occluded'][0,0] and masks['unavailable'][0,0]
    assert not masks['floor'][0,:4].any()
    assert not masks['self_occluded'][0,1:4].any()
    assert masks['other'][0,3]
    assert masks['floor'][1:].all()


def test_footprint_rectangle_and_self_occlusion():
    T=down(); d,_=floor_depth(T); masks=classify_depth(d,d,T)
    a=footprint_visibility([-.03,-.03,0],[.03,.03,.1],T,masks)
    assert a['complete_rectangle_floor'] and all(a['sampled_floor_visibility'])
    assert a['sampled_visibility_is_not_complete_coverage']
    masks['floor'][239:241,319:321]=False; masks['self_occluded'][239:241,319:321]=True
    b=footprint_visibility([-.03,-.03,0],[.03,.03,.1],T,masks)
    assert not b['complete_rectangle_floor'] and b['rectangle_self_occluded_pixels']==4
    assert not b['sampled_floor_visibility'][12]


def test_outside_view_not_coverage():
    T=down(); d,_=floor_depth(T); masks=classify_depth(d,d,T)
    a=footprint_visibility([10,10,0],[11,11,1],T,masks)
    assert not a['complete_frustum'] and not a['complete_rectangle_floor']
    assert not any(a['sampled_floor_visibility'])
    with pytest.raises(ValueError): classify_depth(d[:1],d,T)
    with pytest.raises(ValueError): footprint_visibility([1,0,0],[0,0,0],T,masks)

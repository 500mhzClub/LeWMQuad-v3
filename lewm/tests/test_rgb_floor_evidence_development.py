import math

import numpy as np
import pytest

from lewm.rgb_floor_evidence_development import palette_floor_mask,bottom_connected_envelope,observe_floor
from lewm.floor_visibility_reference_development import pixel_rays,box_depth,visible_floor,confusion
from lewm.tests.test_simulated_body_observation_development import packet


def camera():
    value=np.eye(4); value[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]; value[:3,3]=[0,0,.4]; return value


def test_rgb_palette_evidence_and_unknown_grayscale():
    rgb=np.full((480,640,3),80,dtype=np.uint8); rgb[300:]=[115,120,108]
    assert palette_floor_mask(rgb).sum()==180*640
    gray=np.repeat(rgb.mean(-1,keepdims=True).astype(np.uint8),3,axis=-1)
    assert not palette_floor_mask(gray).any()
    assert not palette_floor_mask(rgb[...,::-1].copy()).any()


def test_uint8_subtraction_cannot_wrap_blue_or_red_into_floor():
    for colour in ([255,0,0],[0,0,255],[0,255,0],[240,240,240]):
        assert not palette_floor_mask(np.broadcast_to(np.array(colour,dtype=np.uint8),(480,640,3))).any()


def test_envelope_requires_unbroken_bottom_floor_and_does_not_fill_occlusion():
    mask=np.zeros((480,640),dtype=bool); mask[300:]=True; mask[450,20]=False; mask[-5:,21]=False
    value=bottom_connected_envelope(mask)
    assert value['first_floor_row'][0]==300 and value['first_floor_row'][20]==451
    assert not value['valid_columns'][21] and value['first_floor_row'][21]==-1


def test_policy_boundary_current_clock_and_nonmetric_output():
    p=packet(); now=p['image']['measured_ns']; p['image']['rgb'][300:]=[115,120,108]
    value=observe_floor(p,now_ns=now)
    assert value['floor_evidence_mask'].any() and value['metric_clearance_qualified'] is False
    assert value['place_or_exit_identity'] is None
    with pytest.raises(ValueError): observe_floor(p,now_ns=now+1)
    p['world_pose']=[0]*7
    with pytest.raises(ValueError): observe_floor(p,now_ns=now)


def test_empty_scene_projects_floor_only_below_horizon():
    result=visible_floor(camera(),[])
    assert not result['visible_floor'][:30].any() and result['visible_floor'][30:].all()
    assert result['valid'].all()
    assert result['rows'].shape==(60,) and result['columns'].shape==(80,)


def test_visible_floor_is_occluded_by_wall_before_ground_intersection():
    box={'centre_xyz':[1.,0,.3],'size_xyz':[.08,3.,.6],'yaw_rad':0.}
    value=visible_floor(camera(),[box]); centre=40
    assert not value['visible_floor'][40,centre] and value['visible_floor'][-1,centre]


def test_wall_behind_camera_does_not_occlude_forward_ground():
    box={'centre_xyz':[-1.,0,.3],'size_xyz':[.08,3.,.6],'yaw_rad':0.}
    assert np.array_equal(visible_floor(camera(),[box])['visible_floor'],visible_floor(camera(),[])['visible_floor'])


def test_independent_box_slab_parallel_and_rotated_cases():
    origin=[0,0,.4]; rays=np.array([[1.,0,0],[0,1.,0],[-1.,0,0]])
    box={'centre_xyz':[1.,0,.4],'size_xyz':[.2,2.,1.],'yaw_rad':0.}
    depth,ambiguous=box_depth(origin,rays,box)
    assert depth[0]==pytest.approx(.9) and np.isinf(depth[1:]).all() and not ambiguous.any()
    box.update(size_xyz=[2.,.2,1.],yaw_rad=math.pi/2)
    assert box_depth(origin,rays,box)[0][0]==pytest.approx(.9)


def test_near_clip_inside_wall_is_explicitly_unknown():
    box={'centre_xyz':[0,0,.4],'size_xyz':[.2,2.,1.],'yaw_rad':0.}
    assert not visible_floor(camera(),[box])['valid'].any()


@pytest.mark.parametrize('bad',[True,0,7,481])
def test_bad_pixel_sampling_rejected(bad):
    with pytest.raises(ValueError): pixel_rays(stride=bad)


def test_reflected_or_invalid_camera_frame_rejected():
    value=camera(); value[:3,0]*=-1
    with pytest.raises(ValueError): visible_floor(value,[])


def test_pixel_accounting_preserves_false_evidence_and_missed_floor():
    pred=np.array([True,True,False,False,True]); truth=np.array([True,False,True,False,False]); valid=np.array([True]*4+[False])
    assert confusion(pred,truth,valid)=={'true_positive':1,'false_positive':1,'false_negative':1,'true_negative':1}

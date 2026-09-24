import numpy as np
import pytest
from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_first_surface_depth_development import expected_optical_depth
from lewm.raster_footprint_visibility_development import projected_boundary_mask,evaluate_footprint

def fixture():
    # Optical forward is +world x; right is -world y; down is -world z.
    T=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,.7],[0,0,0,1.]])
    # Foreground right-hand silhouette placed within a fraction of one pixel
    # of a sampled centre ray which instead strikes the background interior.
    # Use the far-x vertical edge: otherwise this slanted ray enters the
    # finite-thickness side face despite missing its near-x face.
    y_edge=-(324.5-320)/FOCAL*1.1
    boxes=[dict(wall_id='back',centre_xyz=[3.05,0,1.5],size_xyz=[.1,12,3],yaw_rad=0.),
           dict(wall_id='front',centre_xyz=[1.05,y_edge-.100001,.7],size_xyz=[.1,.2,1.4],yaw_rad=0.)]
    ref=expected_optical_depth(boxes,T)
    n=np.zeros((480,640),np.float32)
    n[np.ix_(ref['rows'],ref['columns'])]=ref['expected_depth_m']
    return boxes,T,n,ref

def test_foreground_silhouette_is_not_background_interior_certification():
    boxes,T,n,ref=fixture();_,mask=projected_boundary_mask(boxes,T)
    assert mask[30,40] and ref['object_names'][ref['object_index'][30,40]]=='back'
    n[ref['rows'][30],ref['columns'][40]]=1.
    r=evaluate_footprint(n,boxes,T,render_near_m=.005)
    assert not r['original_strict_score']['passes_sampled_physical_visibility']
    assert r['stable_interior_metric_pass'] and r['boundary_bad_rays']==1
    assert not r['qualification_granted'] and not r['boundary_pixels_certified']

def test_unrelated_interior_corruption_still_fails():
    boxes,T,n,ref=fixture();_,mask=projected_boundary_mask(boxes,T)
    select=ref['surface_interior']&np.isfinite(ref['expected_depth_m'])&(ref['expected_depth_m']<4.98)&~mask
    y,x=np.argwhere(select)[0];n[ref['rows'][y],ref['columns'][x]]+=.1
    r=evaluate_footprint(n,boxes,T,render_near_m=.005)
    assert not r['stable_interior_metric_pass'] and r['stable_interior_bad_rays']==1

def test_partition_and_mask_are_measurement_independent():
    boxes,T,n,ref=fixture();a=evaluate_footprint(n,boxes,T,render_near_m=.005)
    n[:]=np.nan;b=evaluate_footprint(n,boxes,T,render_near_m=.005)
    for key in ('original_compared_rays','stable_interior_rays','boundary_ambiguous_rays','all_projected_boundary_rays'):
        assert a[key]==b[key]
    assert a['original_compared_rays']==a['stable_interior_rays']+a['boundary_ambiguous_rays']
    assert not b['stable_interior_metric_pass']

def test_near_clipping_failure_never_disappears():
    boxes,T,n,ref=fixture()
    r=evaluate_footprint(n,boxes,T,render_near_m=4.)
    assert r['near_occlusion_failure'] and r['original_strict_score']['clipped_opaque_rays']>0

def test_input_order_does_not_change_boundary_mask():
    boxes,T,_,_=fixture()
    np.testing.assert_array_equal(projected_boundary_mask(boxes,T)[1],projected_boundary_mask(boxes[::-1],T)[1])

@pytest.mark.parametrize('bad',['empty','many','transform','stride'])
def test_reject_invalid_geometry(bad):
    boxes,T,_,_=fixture();stride=8
    if bad=='empty':boxes=[]
    elif bad=='many':boxes=boxes*65
    elif bad=='transform':T[0,0]=2
    else:stride=0
    with pytest.raises(ValueError):projected_boundary_mask(boxes,T,stride=stride)

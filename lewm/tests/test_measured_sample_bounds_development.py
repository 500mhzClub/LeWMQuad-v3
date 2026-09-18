import numpy as np
import pytest
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex


def test_empty_voxel_portion_is_not_a_measured_return_and_tangency_remains():
    index=MeasuredSampleBoundsIndex();index.insert([[.001,.001,.001]],{'frame':0})
    r=index.intersect([.01,.01,.01],[.02,.02,.02])
    assert r['whole_voxel_intersections']==1 and r['intersecting_voxels']==0
    assert not r['free_space_established'] and not r['motion_permitted']
    assert index.intersect([.001,.001,.001],[.001,.001,.001])['intersecting_voxels']==1


def test_later_mixed_returns_expand_bounds_without_losing_first_witness():
    index=MeasuredSampleBoundsIndex();index.insert([[.001,.001,.001]],{'frame':0,'source':'first'})
    index.insert([[.019,.019,.019],[.001,.001,.001]],{'frame':1,'source':'later'})
    r=index.intersect([.018,.018,.018],[.02,.02,.02])
    assert r['intersecting_voxels']==1 and r['first_bounds_sample_count']==3
    assert r['witness']=={'frame':0,'source':'first'} and r['first_bounds_latest_frame']==1
    assert r['first_bounds_m'][0][0]<=.001 and r['first_bounds_m'][1][0]>=.019


def test_negative_and_boundary_points_are_all_enclosed_and_counted():
    index=MeasuredSampleBoundsIndex();rng=np.random.default_rng(12)
    points=np.concatenate((rng.uniform(-.1,.1,(1000,3)),[[0,0,0],[.025,-.025,0]]))
    index.insert(points,{'frame':0})
    assert sum(index.sample_counts.values())==len(points)
    for p in points:
        k=tuple(np.floor(p/.025).astype(int));bounds=index.bounds[k]
        assert np.all(bounds[0]<=p) and np.all(bounds[1]>=p)
    for p in points[-2:]:assert index.intersect(p,p)['intersecting_voxels']>=1


def test_invalid_samples_and_inverted_queries_reject():
    index=MeasuredSampleBoundsIndex()
    with pytest.raises(ValueError):index.insert([[np.nan,0,0]],{'frame':0})
    with pytest.raises(ValueError):index.intersect([1,1,1],[0,0,0])
    assert not index.cells and not index.bounds


def test_sphere_excludes_box_corners_but_retains_tangent_points():
    index=MeasuredSampleBoundsIndex();index.insert([[.019,.019,0]],{'frame':0})
    assert index.intersect([-.02,-.02,-.02],[.02,.02,.02])['intersecting_voxels']==1
    assert index.intersect_sphere([0,0,0],.02)['intersecting_voxels']==0
    tangent=MeasuredSampleBoundsIndex();tangent.insert([[.02,0,0]],{'frame':0})
    assert tangent.intersect_sphere([0,0,0],.02)['intersecting_voxels']==1

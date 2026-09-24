from copy import deepcopy
import numpy as np
import pytest
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex


def indices(points):
    old,new=MeasuredSampleBoundsIndex(),SinglePassMeasuredSampleBoundsIndex()
    for frame,p in enumerate(points):
        witness=dict(frame=frame,proof={'values':[frame]})
        for index in (old,new):index.insert(p,witness)
    return old,new


def test_complete_random_queries_match_after_each_monotone_update():
    rng=np.random.default_rng(2026090901)
    old,new=indices([])
    for frame in range(3):
        points=rng.uniform(-.4,.4,(500,3));witness=dict(frame=frame,proof=[frame])
        for index in (old,new):index.insert(points,witness)
        assert old.cells==new.cells and old.sample_counts==new.sample_counts and old.latest_frames==new.latest_frames
        for key in old.bounds:assert old.bounds[key].tobytes()==new.bounds[key].tobytes()
        for i in range(80):
            center=rng.uniform(-.5,.5,3);radius=float(rng.uniform(.001,.5))
            extent=np.full(3,.002) if i%2 else rng.uniform(.02,.7,3)
            assert old.intersect(center-extent,center+extent)==new.intersect(center-extent,center+extent)
            assert old.intersect_sphere(center,radius)==new.intersect_sphere(center,radius)


@pytest.mark.parametrize('center', [[0.,0.,0.],[-.025,0.,.025],[49.,0.,0.]])
def test_closed_boundaries_and_sphere_tolerance_preserve_original_arithmetic(center):
    c=np.asarray(center);points=np.array([c,c+[.025,0.,0.],c+[-.025,0.,0.],c+[0.,.025,0.]])
    old,new=indices([points])
    for radius in (.025-2e-12,.025-1e-12,.025,.025+1e-12):
        assert old.intersect_sphere(c,radius)==new.intersect_sphere(c,radius)
    for p in points:
        assert old.intersect(p,p)==new.intersect(p,p)
        assert old.intersect(np.nextafter(p,-np.inf),np.nextafter(p,np.inf))==new.intersect(np.nextafter(p,-np.inf),np.nextafter(p,np.inf))


def test_empty_and_broad_only_cells_remain_unknown_without_clearance_claim():
    old,new=indices([])
    for index in (old,new):index.cells[(0,0,0)]={'frame':0}
    expected=old.intersect([0.,0.,0.],[.01,.01,.01])
    assert new.intersect([0.,0.,0.],[.01,.01,.01])==expected
    assert expected['status']=='UNKNOWN' and expected['whole_voxel_intersections']==1
    assert not expected['free_space_established'] and not expected['motion_permitted']


def test_lexicographic_witness_and_returned_values_do_not_alias_stored_evidence():
    old,new=indices([np.array([[.02,0.,0.],[-.02,0.,0.]]),np.array([[-.02,0.,0.]])])
    a,b=[index.intersect([-.03,-.01,-.01],[.03,.01,.01]) for index in (old,new)]
    assert a==b and b['first_cell']==[-1,0,0] and b['first_bounds_latest_frame']==1
    before=deepcopy(new.cells);bounds={k:v.tobytes() for k,v in new.bounds.items()}
    b['witness']['proof']['values'].append(999);b['first_bounds_m'][0][0]=999
    assert new.cells==before and {k:v.tobytes() for k,v in new.bounds.items()}==bounds


@pytest.mark.parametrize('lower,upper', [([1,0,0],[0,0,0]),([0,0],[1,1]),
    ([float('nan'),0,0],[1,1,1]),([-51,0,0],[1,1,1]),([0,0,0],[float('inf'),1,1])])
def test_invalid_box_queries_fail_with_the_original_exception(lower,upper):
    old,new=indices([])
    with pytest.raises(ValueError) as original:old.intersect(lower,upper)
    with pytest.raises(type(original.value),match=str(original.value)):new.intersect(lower,upper)

import numpy as np
import pytest
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.tests.test_observed_geometry_refinement_development import floor


def memory(depth=None,valid=None):
    if depth is None:depth,valid=floor()
    m=RetainedFloorPatches();m.append(depth,valid,np.eye(3),np.zeros(3),-.32,dict(frame=0,measured_ns=0))
    return m


def test_whole_foot_patch_and_retained_witness_without_support_claim():
    result=memory().coverage([[1.,0.]])[0]
    assert result['complete_nominal_foot_patch'] and result['coverage_witness']['witness']['frame']==0
    assert not result['ground_support_approved']


def test_one_invalid_interior_pixel_rejects_complete_patch():
    d,v=floor();r=memory(d,v).coverage([[1.,0.]])[0]['coverage_witness']['pixel_rectangle']
    x=(r[0][0]+r[1][0])//2;y=(r[0][1]+r[1][1])//2;d[y,x]=0;v[y,x]=False
    assert not memory(d,v).coverage([[1.,0.]])[0]['complete_nominal_foot_patch']


def test_unseen_and_wall_regions_are_not_covered():
    assert not memory().coverage([[0.,0.]])[0]['complete_nominal_foot_patch']
    assert not memory(np.ones((480,640),np.float32),np.ones((480,640),bool)).coverage([[1.,0.]])[0]['complete_nominal_foot_patch']


def test_history_does_not_forget_valid_old_patch_or_accept_gaps():
    m=memory();d=np.zeros((480,640),np.float32);v=np.zeros_like(d,bool)
    m.append(d,v,np.eye(3),np.zeros(3),-.32,dict(frame=1,measured_ns=100_000_000))
    assert m.coverage([[1.,0.]])[0]['coverage_witness']['witness']['frame']==0
    with pytest.raises(ValueError):m.append(d,v,np.eye(3),np.zeros(3),-.32,dict(frame=3,measured_ns=300_000_000))


def test_queries_are_batched_and_keep_independent_unknowns():
    m=memory();r=m.coverage([[1.,0.],[0.,0.],[1.2,.1]])
    assert [x['complete_nominal_foot_patch'] for x in r]==[True,False,True]
    assert r==[m.coverage([p])[0] for p in [[1.,0.],[0.,0.],[1.2,.1]]]

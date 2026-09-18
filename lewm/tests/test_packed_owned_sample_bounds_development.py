import numpy as np
import pytest
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex, grouped_voxels
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.joint_visual_surface_memory_development import VOXEL_M, MAX_VOXELS
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_batched_sample_bounds_development import assert_equal


def test_packing_preserves_signed_lexicographic_grouping_and_boundaries():
    rng=np.random.default_rng(2026091901)
    edge=np.array([[0.,-0.,.025],[50.,-50.,0.],[-50.,50.,0.],
        [np.nextafter(.025,-np.inf),np.nextafter(-.025,np.inf),0.],[-.025,.025,-.05]])
    for points in (np.empty((0,3)),edge,rng.uniform(-50.,50.,(19200,3))):
        expected=np.unique(np.floor(points/VOXEL_M).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
        for a,b in zip(grouped_voxels(points),expected,strict=True):np.testing.assert_array_equal(a,b)


def test_accumulated_exact_bounds_counts_queries_and_independent_storage():
    rng=np.random.default_rng(2026091902);a=MeasuredSampleBoundsIndex();b=PackedOwnedMeasuredSampleBoundsIndex()
    clouds=[np.empty((0,3)),rng.uniform(-.3,.3,(19200,3)),np.array([[0.,0.,0.],[.025,-.025,0.]])]
    clouds += [clouds[1][:4096],rng.uniform(-.3,.3,(8192,3))]
    for frame,p in enumerate(clouds):
        witness=dict(frame=frame,nested=[frame]);a.insert(p,witness);b.insert(p,witness);assert_equal(a,b)
        witness['nested'].append('caller mutation');assert_equal(a,b)
        for center in rng.uniform(-.4,.4,(10,3)):
            assert a.intersect(center-.04,center+.04)==b.intersect(center-.04,center+.04)
            assert a.intersect_sphere(center,.022)==b.intersect_sphere(center,.022)
        assert all(v.flags.owndata and v.base is None and v.shape==(2,3) for v in b.bounds.values())
    keys=list(b.cells);b.cells[keys[0]]['nested'].append('one-cell mutation')
    assert all('one-cell mutation' not in b.cells[k]['nested'] for k in keys[1:])


@pytest.mark.parametrize('p',[np.zeros((3,2)),np.zeros((19201,3)),[[np.nan,0.,0.]],
    [[np.inf,0.,0.]],[[50.0001,0.,0.]]])
def test_invalid_insert_has_same_exception_and_no_partial_evidence(p):
    a=MeasuredSampleBoundsIndex();b=PackedOwnedMeasuredSampleBoundsIndex()
    for index in (a,b):index.insert([[.1,.2,.3]],{'frame':0})
    for index in (a,b):
        with pytest.raises(SensorContractError):index.insert(p,{'frame':1})
    assert_equal(a,b)


def test_capacity_failure_does_not_evict_or_add_evidence():
    index=PackedOwnedMeasuredSampleBoundsIndex()
    index.cells={(i,0,0):{'frame':0} for i in range(MAX_VOXELS)}
    with pytest.raises(SensorContractError,match='capacity exhausted'):index.insert([[-.025,0.,0.]],{'frame':1})
    assert len(index.cells)==MAX_VOXELS and index.bounds=={} and index.sample_counts=={} and index.latest_frames=={}


def test_sparse_historical_cell_does_not_keep_old_batch_array_alive():
    index=PackedOwnedMeasuredSampleBoundsIndex()
    points=np.column_stack((np.arange(1000)*.025,np.zeros(1000),np.zeros(1000)))
    index.insert(points,{'frame':0});old=index.bounds[(0,0,0)]
    index.insert(points[1:],{'frame':1})
    assert index.bounds[(0,0,0)] is old and old.base is None and old.nbytes==48


def test_missing_witness_frame_preserves_original_partial_failure_semantics():
    a=MeasuredSampleBoundsIndex();b=PackedOwnedMeasuredSampleBoundsIndex()
    for index in (a,b):
        with pytest.raises(KeyError):index.insert([[.1,.2,.3],[.2,.3,.4]],{})
    assert_equal(a,b)

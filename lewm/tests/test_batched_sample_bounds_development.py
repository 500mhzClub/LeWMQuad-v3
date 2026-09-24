"""Compare the optimization with the frozen measured-evidence implementation."""
import numpy as np
import pytest

from lewm.batched_sample_bounds_development import BatchedMeasuredSampleBoundsIndex
from lewm.causal_sensor_state import SensorContractError
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex


def assert_equal(a,b):
    assert list(a.cells)==list(b.cells) and a.cells==b.cells
    assert list(a.bounds)==list(b.bounds)
    assert a.sample_counts==b.sample_counts and a.latest_frames==b.latest_frames
    for k in a.bounds:np.testing.assert_array_equal(a.bounds[k],b.bounds[k])


def test_accumulated_points_counts_queries_and_rounding_match_frozen():
    rng=np.random.default_rng(2026091701)
    a,b=MeasuredSampleBoundsIndex(),BatchedMeasuredSampleBoundsIndex()
    edge=np.array([[0.,-0.,.025],[.025,-.025,.05],[-.025,.025,-.05],
        [np.nextafter(.025,-np.inf),np.nextafter(-.025,np.inf),0.]])
    clouds=[np.empty((0,3)),edge,rng.uniform(-.3,.3,(19200,3)),edge,
        rng.uniform(-.3,.3,(8192,3))]
    for frame,points in enumerate(clouds):
        witness=dict(frame=frame,measured_ns=frame*100_000_000,
            rgb_sha256='a'*64,depth_sha256='b'*64,nested={'values':[frame]})
        a.insert(points,witness);b.insert(points,witness);assert_equal(a,b)
        witness['nested']['values'].append('caller mutation')
        assert_equal(a,b)
        for center in rng.uniform(-.4,.4,(12,3)):
            low,high=center-.04,center+.04
            assert a.intersect(low,high)==b.intersect(low,high)
            assert a.intersect_sphere(center,.022)==b.intersect_sphere(center,.022)
    keys=list(b.cells)
    b.cells[keys[0]]['nested']['values'].append('one witness mutation')
    assert all('one witness mutation' not in b.cells[k]['nested']['values'] for k in keys[1:])
    first,second=list(b.bounds)[:2];old=b.bounds[second].copy()
    b.bounds[first][0,0]=42.
    np.testing.assert_array_equal(b.bounds[second],old)


@pytest.mark.parametrize('points',[
    np.zeros((2,2)),np.zeros((19201,3)),np.array([[np.nan,0.,0.]]),
    np.array([[np.inf,0.,0.]]),np.array([[50.0001,0.,0.]]),
])
def test_invalid_input_fails_before_evidence_mutation(points):
    for cls in (MeasuredSampleBoundsIndex,BatchedMeasuredSampleBoundsIndex):
        index=cls();index.insert([[.1,.2,.3]],{'frame':0})
        before=(dict(index.cells),{k:v.copy() for k,v in index.bounds.items()},
            dict(index.sample_counts),dict(index.latest_frames))
        with pytest.raises(SensorContractError):index.insert(points,{'frame':1})
        assert index.cells==before[0] and index.sample_counts==before[2] and index.latest_frames==before[3]
        for k,v in before[1].items():np.testing.assert_array_equal(index.bounds[k],v)


def test_capacity_failure_keeps_existing_evidence():
    from lewm.joint_visual_surface_memory_development import MAX_VOXELS
    for cls in (MeasuredSampleBoundsIndex,BatchedMeasuredSampleBoundsIndex):
        index=cls();index.cells={(i,0,0):{'frame':0} for i in range(MAX_VOXELS)}
        with pytest.raises(SensorContractError,match='capacity exhausted'):
            index.insert([[-.025,0.,0.]],{'frame':1})
        assert len(index.cells)==MAX_VOXELS and index.bounds=={} and index.sample_counts=={} and index.latest_frames=={}

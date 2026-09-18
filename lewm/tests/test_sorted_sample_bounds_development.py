import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from lewm.sorted_sample_bounds_development import SortedSampleBoundsIndex


def same(a, b):
    assert a.cells == b.cells and list(a.cells) == list(b.cells)
    assert a.sample_counts == b.sample_counts and a.latest_frames == b.latest_frames
    assert a.bounds.keys() == b.bounds.keys()
    for k in a.bounds: assert a.bounds[k].tobytes() == b.bounds[k].tobytes()


def test_accumulated_bounds_counts_first_witness_and_queries_match():
    rng = np.random.default_rng(2026091309)
    a = SinglePassMeasuredSampleBoundsIndex(); b = SortedSampleBoundsIndex()
    for frame in range(6):
        p = rng.uniform(-.2, .2, (19200, 3))
        p[:4] = [[0., -.0, 0.], [-.0, 0., -.0], [.025, -.025, .05], [-50., 50., 0.]]
        witness = dict(frame=frame, nested=dict(value=[frame]))
        a.insert(p, witness); b.insert(p, witness); same(a, b)
        witness['nested']['value'][0] = -1
    for center in ([0, 0, 0], [.025, -.025, .05], [1., 1., 1.]):
        assert a.intersect_sphere(center, .03) == b.intersect_sphere(center, .03)
    b.insert(np.empty((0, 3)), {'frame': 6}); same(a, b)


def test_returned_cell_bounds_remain_independent_and_enclose_all_samples():
    p = np.array([[.001, .002, .003], [.004, .005, .006], [.03, .04, .049]])
    index = SortedSampleBoundsIndex(); index.insert(p, dict(frame=0))
    for point in p:
        bounds = index.bounds[tuple(np.floor(point/.025).astype(int))]
        assert (bounds[0] < point).all() and (bounds[1] > point).all()
    first, second = list(index.bounds.values())
    assert not np.shares_memory(first, second)


def test_invalid_points_leave_index_unchanged():
    index = SortedSampleBoundsIndex()
    for points in ([[np.nan, 0, 0]], [[51., 0, 0]], np.ones((19201, 3)), np.ones((3, 2))):
        with pytest.raises(SensorContractError): index.insert(points, dict(frame=0))
        assert not index.cells and not index.bounds

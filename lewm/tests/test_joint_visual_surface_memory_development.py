"""Unknown-space, evidence retention and no-shortcut route regressions."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_visual_surface_memory_development import SurfaceIndex, JointVisualSurfaceMemory


def test_empty_and_unobserved_boxes_never_become_free():
    index = SurfaceIndex()
    assert index.intersect([0, 0, 0], [1, 1, 1])['status'] == 'UNKNOWN'
    index.insert([[.51, .01, .01]], {'frame': 0})
    hit = index.intersect([.5, 0, 0], [.6, .1, .1])
    assert hit['status'] == 'POSSIBLE_SURFACE_INTERSECTION'
    assert not hit['motion_permitted'] and not hit['free_space_established']
    assert index.intersect([.6, 0, 0], [.7, .1, .1])['status'] == 'UNKNOWN'


def test_closed_negative_voxels_and_first_witness_survive_later_frames():
    index = SurfaceIndex(); witness = {'frame': 0}
    index.insert([[-.01, 0, 0]], witness); witness['frame'] = 100
    index.insert([[-.02, 0, 0], [2, 2, 2]], {'frame': 1})
    hit = index.intersect([0, 0, 0], [0, 0, 0])
    assert hit['first_cell'] == [-1, 0, 0] and hit['witness']['frame'] == 0
    hit['witness']['frame'] = 30
    assert index.intersect([0, 0, 0], [0, 0, 0])['witness']['frame'] == 0


def test_invalid_insertion_is_atomic_and_bad_queries_reject():
    index = SurfaceIndex(); index.insert([[1, 2, 3]], {'frame': 0})
    original = deepcopy(index.cells)
    for points in ([[np.nan, 0, 0]], [[51, 0, 0]], np.zeros((19201, 3))):
        with pytest.raises(SensorContractError): index.insert(points, {'frame': 1})
        assert index.cells == original
    with pytest.raises(SensorContractError): index.intersect([1, 0, 0], [0, 0, 0])


def test_backtrack_reverses_observed_loop_without_shortcut_and_requires_freshness():
    memory = JointVisualSurfaceMemory(identity=(0, 0, 0))
    memory.route = [dict(frame=i, position_initial_body_m=p) for i, p in enumerate(
        ([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]))]
    memory.last_ns = 500
    row = memory.backtrack(0, now_ns=500)
    assert [x['frame'] for x in row['targets']] == [3, 2, 1, 0]
    assert row['physical_execution_required'] and not row['motion_permitted']
    row['targets'][0]['frame'] = 99
    assert memory.route[3]['frame'] == 3
    with pytest.raises(SensorContractError): memory.backtrack(0, now_ns=501)
    memory.failed = True
    with pytest.raises(SensorContractError): memory.backtrack(0, now_ns=500)

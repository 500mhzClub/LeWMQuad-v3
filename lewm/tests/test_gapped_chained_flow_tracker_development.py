import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.chained_corner_flow_association_development import chained_points as original
from lewm.gapped_chained_flow_tracker_development import chained_points
from lewm.tests.test_chained_corner_flow_association_development import scene


def clocks(frames, gaps):
    now = frames[0][1]; result = [frames[0]]
    for i, row in enumerate(frames[1:]):
        now += gaps[i % len(gaps)]*100_000_000
        result.append((row[0], now, row[2]))
    return result


def test_moving_corner_endpoints_match_original_and_keep_real_clocks():
    frames = scene(8)
    expected, _ = original(frames)
    for rows in (frames, clocks(frames, (1, 2, 3))):
        actual, receipt = chained_points(rows)
        assert len(actual[0]) >= 12
        for a, b in zip(actual, expected, strict=True): np.testing.assert_array_equal(a, b)
        assert [s['current_measured_ns'] for s in receipt['steps']] == [r[1] for r in rows[1:]]
        assert receipt['image_timestamps_changed'] is False


@pytest.mark.parametrize('fault', ['occluded_image', 'invalid_depth'])
def test_lost_tracks_do_not_reappear_after_gap(fault):
    frames = clocks(scene(4, shift=0), (3,))
    if fault == 'occluded_image': frames[1][2].gray.fill(0)
    else: frames[1][2].depth = dict(depth_m=np.ones((480, 640)), valid=np.zeros((480, 640), bool))
    result, receipt = chained_points(frames)
    assert len(result[0]) == 0 and receipt['lost_tracks_reintroduced'] is False


@pytest.mark.parametrize('gaps,count', [((0,), 2), ((6,), 2), ((5,), 8)])
def test_camera_gap_and_elapsed_chain_limits(gaps, count):
    with pytest.raises(SensorContractError): chained_points(clocks(scene(count), gaps))

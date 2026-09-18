"""Measured image-chain association: geometry, loss, clock and ownership tests."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.chained_corner_flow_association_development import chained_points
from lewm.direct_corner_flow_association_development import tracked_points
from lewm.rgbd_correspondence_motion_development import lift


def scene(count=5, shift=8):
    rng = np.random.default_rng(2026091101)
    canvas = rng.integers(0, 256, (480, 1000), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
    depth = dict(depth_m=np.ones((480, 640), np.float64), valid=np.ones((480, 640), bool))
    frames = []
    points = [(float(x), float(y)) for y in (90, 180, 270, 360) for x in (200, 270, 340, 410, 480)]
    for i in range(count):
        f = SimpleNamespace(gray=canvas[:, i*shift:i*shift+640].copy(), depth=depth,
            keypoints=[cv2.KeyPoint(x, y, 8.) for x, y in points])
        frames.append((40+i, 5_500_000_000+i*100_000_000, f))
    return frames


def test_one_interval_equals_original_association():
    frames = scene(2)
    expected, _ = tracked_points(frames[0][2], frames[1][2])
    actual, receipt = chained_points(frames)
    assert len(actual[0]) >= 12
    for a, b in zip(actual, expected, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert receipt['intervals'] == 1 and receipt['pose_admitted'] is False


def test_large_total_shift_keeps_original_coordinates_and_endpoint_depth():
    frames = scene(13)
    snapshots = [(r[2].gray.tobytes(), r[2].depth['depth_m'].tobytes(), [k.pt for k in r[2].keypoints]) for r in frames]
    (a, b, ua, ub), receipt = chained_points(frames)
    assert len(a) >= 12
    np.testing.assert_allclose(ua-ub, np.tile([96., 0.], (len(ua), 1)), atol=.15, rtol=0)
    expected_a, valid_a = lift(frames[0][2].depth, ua)
    expected_b, valid_b = lift(frames[-1][2].depth, ub)
    assert valid_a.all() and valid_b.all()
    np.testing.assert_array_equal(a, expected_a)
    np.testing.assert_array_equal(b, expected_b)
    assert receipt['pose_increments_composed'] is False
    assert snapshots == [(r[2].gray.tobytes(), r[2].depth['depth_m'].tobytes(), [k.pt for k in r[2].keypoints]) for r in frames]
    repeated, check = chained_points(frames)
    assert check == receipt
    for x, y in zip((a, b, ua, ub), repeated, strict=True):
        assert x.tobytes() == y.tobytes()


def test_occluded_middle_frame_cannot_reseed_from_identical_endpoint():
    frames = scene(4, shift=0)
    frames[1][2].gray.fill(0)
    result, receipt = chained_points(frames)
    assert len(result[0]) == 0
    assert receipt['lost_tracks_reintroduced'] is False
    assert all(s['association']['counts']['valid_depth_pair'] == 0 for s in receipt['steps'])


def test_invalid_intermediate_depth_cannot_recover_at_endpoint():
    frames = scene(3, shift=0)
    frames[1][2].depth = dict(depth_m=np.ones((480, 640)), valid=np.zeros((480, 640), bool))
    values, _ = chained_points(frames)
    assert len(values[0]) == 0


@pytest.mark.parametrize('fault', ['empty', 'single', 'too_long', 'gap', 'clock', 'boolean_frame', 'bad_gray'])
def test_rejects_invalid_chain_even_when_no_tracks(fault):
    frames = scene(2)
    frames[0][2].keypoints = []
    if fault == 'empty': frames = []
    elif fault == 'single': frames = frames[:1]
    elif fault == 'too_long': frames = frames*17
    elif fault == 'gap': frames[1] = (42, frames[1][1], frames[1][2])
    elif fault == 'clock': frames[1] = (41, frames[1][1]+1, frames[1][2])
    elif fault == 'boolean_frame': frames[0] = (True, frames[0][1], frames[0][2])
    elif fault == 'bad_gray': frames[1][2].gray = np.zeros((480, 640), np.float64)
    with pytest.raises(SensorContractError): chained_points(frames)


def test_duplicate_reference_corner_does_not_add_evidence():
    frames = scene(2)
    expected, _ = chained_points(frames)
    frames[0][2].keypoints += frames[0][2].keypoints
    actual, _ = chained_points(frames)
    for a, b in zip(actual, expected, strict=True):
        assert a.tobytes() == b.tobytes()

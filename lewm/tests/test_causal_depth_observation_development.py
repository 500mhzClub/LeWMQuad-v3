from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import (from_native_depth, validate_depth, body_points,
    CausalDepthHistory, calibration_metadata, FOCAL)
from lewm.tests.test_observed_traversal_controller_development import Stream


def frame(stream=None, tick=0):
    p, _, now = (stream or Stream()).frame(tick)
    native = np.full((480, 640), 2., np.float32)
    depth = from_native_depth(native, p, measured_ns=now, available_ns=now, now_ns=now)
    return p, depth, now


def test_native_clip_missingness_and_copy_without_rgb_or_body_mutation():
    p, _, now = frame(); before = deepcopy(p)
    native = np.full((480, 640), 2., np.float32)
    native[0, :6] = [np.nan, np.inf, -.1, 0., .19, 200.]
    d = from_native_depth(native, p, measured_ns=now, available_ns=now, now_ns=now)
    assert not d['valid'][0, :6].any() and np.all(d['depth_m'][0, :6] == 0.)
    native[:] = 3.
    assert d['depth_m'][1, 1] == 2.
    np.testing.assert_array_equal(p['image']['rgb'], before['image']['rgb'])
    assert set(p) == {'image', 'sensor_state'}


@pytest.mark.parametrize('fault', ['future', 'stale', 'available_before_measured', 'episode', 'rgb', 'privilege',
    'calibration', 'units', 'hardware', 'shape', 'dtype', 'mask', 'invalid_nonzero', 'valid_nan', 'valid_out_of_range'])
def test_depth_contract_rejects_invalid_or_privileged_input(fault):
    p, d, now = frame()
    if fault == 'future': d['available_ns'] = now+1
    if fault == 'stale': d['measured_ns'] -= 100_000_000
    if fault == 'available_before_measured': d['available_ns'] -= 1
    if fault == 'episode': d['identity'] = (0, 0, 1)
    if fault == 'rgb': d['rgb_sha256'] = '0'*64
    if fault == 'privilege': d['world_pose'] = [0., 0., 0.]
    if fault == 'calibration': d['calibration_id'] = 'unknown'
    if fault == 'units': d['representation'] = 'euclidean_range_mm'
    if fault == 'hardware': d['hardware_calibrated'] = True
    if fault == 'shape': d['depth_m'] = d['depth_m'][:240]
    if fault == 'dtype': d['depth_m'] = d['depth_m'].astype(np.float64)
    if fault == 'mask': d['valid'] = d['valid'].astype(np.uint8)
    if fault == 'invalid_nonzero': d['valid'][0, 0] = False
    if fault == 'valid_nan': d['depth_m'][0, 0] = np.nan
    if fault == 'valid_out_of_range': d['depth_m'][0, 0] = 6.
    with pytest.raises(SensorContractError): validate_depth(d, p, now_ns=now)


def test_optical_depth_unprojection_uses_half_pixel_centres_and_body_mount():
    p, d, now = frame()
    cloud = body_points(d, p, now_ns=now, stride=8)
    np.testing.assert_allclose(cloud['points_body_m'][0, 0], [2.326, -2*(4.5-320)/FOCAL, .043-2*(4.5-240)/FOCAL])
    d['valid'][4, 4] = False; d['depth_m'][4, 4] = 0.
    cloud = body_points(d, p, now_ns=now)
    assert np.isnan(cloud['points_body_m'][0, 0]).all() and not cloud['environment_clearance_qualified']
    assert calibration_metadata()['hardware_calibrated'] is False


def test_four_frame_history_is_copied_and_gap_failure_latches():
    stream = Stream(); history = CausalDepthHistory()
    for tick in range(6):
        p, d, now = frame(stream, tick); history.push(d, p, now_ns=now)
    snapshot = history.snapshot(); assert len(snapshot) == 4
    snapshot[-1]['depth_m'][:] = 4.; d['depth_m'][:] = 3.
    assert history.snapshot()[-1]['depth_m'][0, 0] == 2.
    with pytest.raises(SensorContractError): history.push(d, p, now_ns=now)
    p, d, now = frame(stream, 6)
    with pytest.raises(SensorContractError): history.push(d, p, now_ns=now)

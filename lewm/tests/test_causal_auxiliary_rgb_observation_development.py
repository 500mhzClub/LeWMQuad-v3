from copy import deepcopy
import hashlib
import numpy as np
import pytest
from PIL import Image
from lewm.tests.test_causal_depth_observation_development import frame
from lewm.causal_sensor_state import SensorContractError
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth
from lewm.auxiliary_downward45_depth_geometry_development import CALIBRATION_ID
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb, validate_rgb
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition


def inputs():
    p, _, now = frame()
    native = np.full((480, 640), 2., np.float32)
    native[0, 0] = np.nan
    d = from_native_depth(native, p, measured_ns=now, available_ns=now, now_ns=now)
    rgb = np.full((480, 640, 3), 77, np.uint8)
    a = from_captured_rgb(rgb, d, p, measured_ns=now, available_ns=now, now_ns=now)
    return p, d, a, now, native, rgb


def test_pixels_are_owned_and_auxiliary_depth_missingness_is_preserved():
    p, d, a, now, _, rgb = inputs()
    before = deepcopy(d)
    rgb[:] = 12
    validate_rgb(a, d, p, now_ns=now)
    assert np.all(a['rgb'] == 77) and a['rgb'].flags.owndata
    assert not d['valid'][0, 0] and d['depth_m'][0, 0] == 0
    np.testing.assert_array_equal(d['depth_m'], before['depth_m'])
    assert a['calibration_id'] != p['image']['calibration_id']


@pytest.mark.parametrize('fault', ['future', 'stale', 'early', 'decision', 'episode',
    'primary', 'depth', 'pixels', 'shape', 'dtype', 'privilege', 'calibration',
    'depth_calibration', 'hardware', 'primary_depth'])
def test_rejects_misaligned_or_privileged_packet(fault):
    p, d, a, now, _, _ = inputs()
    if fault == 'future': a['available_ns'] = now+1
    if fault == 'stale': a['measured_ns'] -= 100_000_000
    if fault == 'early': a['available_ns'] = now-1
    if fault == 'decision': a['decision_ns'] = now-1
    if fault == 'episode': a['identity'] = (0, 0, 1)
    if fault == 'primary': a['primary_rgb_sha256'] = '0'*64
    if fault == 'depth': d['depth_m'][1, 1] = 3.
    if fault == 'pixels': a['rgb'][1, 1, 0] = 2
    if fault == 'shape': a['rgb'] = a['rgb'][:240]
    if fault == 'dtype': a['rgb'] = a['rgb'].astype(float)
    if fault == 'privilege': a['world_from_optical'] = np.eye(4)
    if fault == 'calibration': a['calibration_id'] = p['image']['calibration_id']
    if fault == 'depth_calibration': a['depth_calibration_id'] = 'primary'
    if fault == 'hardware': a['hardware_calibrated'] = True
    if fault == 'primary_depth': _, d, _ = frame()
    with pytest.raises(SensorContractError): validate_rgb(a, d, p, now_ns=now)


def test_replay_reads_only_public_arrays_and_rejects_wrong_capture(tmp_path):
    p, d, a, now, native, rgb = inputs()
    np.savez(tmp_path/'auxiliary_depth_0000.npz', native_optical_depth_m=native,
        depth_m=d['depth_m'], valid=d['valid'],
        diagnostic_segmentation=np.array([{'privileged': True}], dtype=object))
    Image.fromarray(rgb).save(tmp_path/'auxiliary_rgb_0000.png')
    row = dict(frame=0, measured_ns=now, calibration_id=CALIBRATION_ID,
        native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        rgb_sha256=a['rgb_sha256'], world_from_optical='not public')
    public = public_acquisition(row)
    image, depth = packet(tmp_path, 0, p, public, now_ns=now)
    np.testing.assert_array_equal(image['rgb'], rgb)
    np.testing.assert_array_equal(depth['valid'], d['valid'])
    for bad in [row, public|{'frame': True}, public|{'frame': 1},
                public|{'rgb_sha256': '0'*64}, public|{'native_depth_sha256': '0'*64},
                public|{'measured_ns': now-100_000_000}]:
        with pytest.raises(ValueError): packet(tmp_path, 0, p, bad, now_ns=now)
    for index in [-1, True, 1.5, 100000]:
        with pytest.raises(ValueError): packet(tmp_path, index, p, public, now_ns=now)

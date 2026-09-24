import json

import numpy as np
import pytest

from lewm.causal_depth_observation_development import SCHEMA, calibration_metadata, from_native_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.whole_task_rgb_dataset_development import load_whole_task_observation
from lewm.tests.test_causal_rgb_dataset_development import create


def fixture(directory):
    manifest, arrays = create(directory)
    manifest['schema'] = 'causal_rgb_body_routes_development.v1'
    (directory/'policy_observations.json').write_text(json.dumps(manifest))
    policy = load_whole_task_observation(directory, 0)
    now = policy['sensor_state']['decision_ns']
    depth = from_native_depth(np.full((480, 640), 2., np.float32), policy,
                              measured_ns=now, available_ns=now, now_ns=now)
    np.savez_compressed(directory/'depth_0000.npz', depth_m=depth['depth_m'], valid=depth['valid'])
    frame = {k: v for k, v in depth.items() if k not in ('depth_m', 'valid')}
    frame['depth_file'] = 'depth_0000.npz'
    result = {'schema': SCHEMA, 'calibration': calibration_metadata(), 'frames': [frame]}
    (directory/'depth_observations.json').write_text(json.dumps(result))
    return policy, depth, result


def test_policy_reader_needs_no_physics_world_geometry_or_native_depth_files(tmp_path):
    expected, depth, _ = fixture(tmp_path)
    for name in ('physics_trace.npz', 'camera_audit.json', 'static_objects.json', 'native_depth_0000.npz'):
        (tmp_path/name).write_bytes(b'not-readable-policy-evidence')
    p, d = load_rgbd_observation(tmp_path, 0)
    assert set(p) == set(expected) == {'image', 'sensor_state'}
    np.testing.assert_array_equal(p['image']['rgb'], expected['image']['rgb'])
    for key in depth:
        assert np.array_equal(d[key], depth[key]) if isinstance(depth[key], np.ndarray) else d[key] == depth[key]


@pytest.mark.parametrize('fault', ['future', 'path', 'privilege', 'calibration', 'population', 'rgb', 'tensor'])
def test_rgbd_reader_rejects_wrong_or_privileged_artifact(tmp_path, fault):
    _, _, manifest = fixture(tmp_path)
    if fault == 'future': manifest['frames'][0]['available_ns'] += 1
    if fault == 'path': manifest['frames'][0]['depth_file'] = '../depth_0000.npz'
    if fault == 'privilege': manifest['frames'][0]['target_coordinates'] = [1., 0.]
    if fault == 'calibration': manifest['calibration']['intrinsics'][0][0] += 1
    if fault == 'population': manifest['frames'].append(dict(manifest['frames'][0]))
    if fault == 'rgb': manifest['frames'][0]['rgb_sha256'] = '0'*64
    if fault == 'tensor': np.savez_compressed(tmp_path/'depth_0000.npz', geometry=np.ones(3))
    (tmp_path/'depth_observations.json').write_text(json.dumps(manifest))
    with pytest.raises(SensorContractError): load_rgbd_observation(tmp_path, 0)

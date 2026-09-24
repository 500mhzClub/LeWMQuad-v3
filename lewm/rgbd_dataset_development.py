"""Policy-only reader: RGB/body stays unchanged; depth is a separate modality."""
import json
from pathlib import Path

import numpy as np

from lewm.causal_rgb_dataset_development import _leaf, _protected
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import SCHEMA, calibration_metadata, validate_depth
from lewm.whole_task_rgb_dataset_development import load_whole_task_observation


def load_rgbd_observation(directory, index):
    directory = Path(directory).absolute()
    if _protected(directory) or _protected(directory.resolve()):
        raise SensorContractError('protected RGBD input forbidden')
    directory = directory.resolve()
    policy = load_whole_task_observation(directory, index)
    manifest = json.loads(_leaf(directory, 'depth_observations.json').read_text())
    if (set(manifest) != {'schema', 'calibration', 'frames'} or manifest['schema'] != SCHEMA
            or manifest['calibration'] != calibration_metadata()):
        raise SensorContractError('exact depth observation metadata required')
    frames = manifest['frames']
    rgb = json.loads(_leaf(directory, 'policy_observations.json').read_text())
    if not isinstance(frames, list) or len(frames) != len(rgb['frames']):
        raise SensorContractError('paired depth/RGB population required')
    for i, frame in enumerate(frames):
        if (set(frame) != {'depth_file', 'schema', 'calibration_id', 'identity', 'measured_ns', 'available_ns',
                          'decision_ns', 'rgb_sha256', 'representation', 'hardware_calibrated'}
                or frame['depth_file'] != f'depth_{i:04d}.npz'):
            raise SensorContractError('exact depth frame fields and paths required')
    result = dict(frames[index]); name = result.pop('depth_file')
    result['identity'] = tuple(result['identity'])
    with np.load(_leaf(directory, name), allow_pickle=False) as archive:
        if set(archive.files) != {'depth_m', 'valid'}:
            raise SensorContractError('unexpected depth tensor fields')
        result.update({k: archive[k] for k in archive.files})
    validate_depth(result, policy, now_ns=policy['sensor_state']['decision_ns'])
    return policy, result

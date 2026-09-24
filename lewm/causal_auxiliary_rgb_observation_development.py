"""Separate downward RGB tied to its public depth and primary observation.

This is an ideal simulated camera contract, not hardware calibration or a
tracking result. No native pose, object label or segmentation field is allowed.
"""
import hashlib
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _ns, _identity
from lewm.causal_depth_observation_development import rgb_digest
from lewm.auxiliary_downward45_depth_observation_development import (
    CALIBRATION_ID as DEPTH_CALIBRATION, validate_depth)

SCHEMA = 'causal_auxiliary_downward45_rgb_development.v1'
CALIBRATION_ID = 'go2_auxiliary_640x480_pitch45_mount035_000_008_rgb_v1'
FIELDS = {'schema', 'calibration_id', 'depth_calibration_id', 'identity',
          'measured_ns', 'available_ns', 'decision_ns', 'primary_rgb_sha256',
          'auxiliary_depth_sha256', 'rgb_sha256', 'rgb', 'hardware_calibrated'}


def depth_digest(depth):
    return hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()


def from_captured_rgb(rgb, depth, policy, *, measured_ns, available_ns, now_ns):
    validate_depth(depth, policy, now_ns=now_ns)
    if not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise SensorContractError('auxiliary RGB requires uint8 HxWx3 captured pixels')
    result = dict(schema=SCHEMA, calibration_id=CALIBRATION_ID,
        depth_calibration_id=DEPTH_CALIBRATION, identity=depth['identity'],
        measured_ns=measured_ns, available_ns=available_ns, decision_ns=now_ns,
        primary_rgb_sha256=rgb_digest(policy), auxiliary_depth_sha256=depth_digest(depth),
        rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), rgb=rgb.copy(),
        hardware_calibrated=False)
    validate_rgb(result, depth, policy, now_ns=now_ns)
    return result


def validate_rgb(image, depth, policy, *, now_ns):
    validate_depth(depth, policy, now_ns=now_ns)
    now = _ns(now_ns, 'auxiliary RGB decision')
    if not isinstance(image, dict) or set(image) != FIELDS:
        raise SensorContractError('exact auxiliary RGB fields required')
    if (image['schema'] != SCHEMA or image['calibration_id'] != CALIBRATION_ID
            or image['depth_calibration_id'] != DEPTH_CALIBRATION
            or image['hardware_calibrated'] is not False):
        raise SensorContractError('explicit unqualified downward RGB calibration required')
    measured = _ns(image['measured_ns'], 'auxiliary RGB measured')
    available = _ns(image['available_ns'], 'auxiliary RGB available')
    if (_ns(image['decision_ns'], 'auxiliary RGB decision') != now
            or measured != depth['measured_ns'] or not measured <= available <= now):
        raise SensorContractError('causal synchronized auxiliary RGB/depth clock required')
    if (_identity(image['identity']) != depth['identity']
            or image['primary_rgb_sha256'] != rgb_digest(policy)
            or image['auxiliary_depth_sha256'] != depth_digest(depth)):
        raise SensorContractError('auxiliary RGB episode, primary or depth binding mismatch')
    rgb = image['rgb']
    if (not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8
            or image['rgb_sha256'] != hashlib.sha256(rgb.tobytes()).hexdigest()):
        raise SensorContractError('exact captured auxiliary RGB pixels required')

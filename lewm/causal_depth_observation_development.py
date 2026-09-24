"""Separate ideal RGB-aligned optical-depth modality, not an RGB-only input.

Fixed development calibration. No world-frame pose, object labels or analytic
scene queries are accepted. Invalid rays remain unknown, never free space.
"""
from collections import deque
from copy import deepcopy
import hashlib
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError, _ns, _identity
from lewm.simulated_body_observation_development import validate_policy_packet

SCHEMA = 'causal_colocated_optical_depth_development.v1'
CALIBRATION_ID = 'go2_colocated_640x480_hfov78.323_optical_depth_0.2_to_5m_v1'
MIN_DEPTH_M, MAX_DEPTH_M = .20, 5.
HEIGHT, WIDTH = 480, 640
FOCAL = WIDTH/(2*math.tan(math.radians(78.323)/2))
INTRINSICS = ((FOCAL, 0., 320.), (0., FOCAL, 240.), (0., 0., 1.))
BODY_FROM_OPTICAL = ((0., 0., 1., .326), (-1., 0., 0., 0.), (0., -1., 0., .043), (0., 0., 0., 1.))
FIELDS = {'schema', 'calibration_id', 'identity', 'measured_ns', 'available_ns', 'decision_ns',
          'rgb_sha256', 'depth_m', 'valid', 'representation', 'hardware_calibrated'}


def rgb_digest(policy):
    return hashlib.sha256(policy['image']['rgb'].tobytes()).hexdigest()


def calibration_metadata():
    return {'calibration_id': CALIBRATION_ID, 'resolution_wh': [WIDTH, HEIGHT],
            'intrinsics': [list(r) for r in INTRINSICS],
            'body_from_optical': [list(r) for r in BODY_FROM_OPTICAL],
            'pixel_centres': 'column+0.5,row+0.5', 'representation': 'optical_axis_depth_m',
            'minimum_depth_m': MIN_DEPTH_M, 'maximum_depth_m': MAX_DEPTH_M,
            'hardware_calibrated': False, 'assumption': 'ideal_colocated_simulated_depth_zero_latency'}


def from_native_depth(native_depth, policy, *, measured_ns, available_ns, now_ns):
    """Acquisition adapter: convert only native invalid/out-of-range pixels to zero."""
    native = np.asarray(native_depth)
    if native.shape != (HEIGHT, WIDTH) or native.dtype != np.float32:
        raise SensorContractError('native depth requires float32 HxW optical metres')
    valid = np.isfinite(native) & (native >= MIN_DEPTH_M) & (native <= MAX_DEPTH_M)
    result = {'schema': SCHEMA, 'calibration_id': CALIBRATION_ID,
              'identity': policy['sensor_state']['identity'], 'measured_ns': measured_ns,
              'available_ns': available_ns, 'decision_ns': now_ns, 'rgb_sha256': rgb_digest(policy),
              'depth_m': np.where(valid, native, np.float32(0.)).copy(), 'valid': valid.copy(),
              'representation': 'optical_axis_depth_m', 'hardware_calibrated': False}
    validate_depth(result, policy, now_ns=now_ns)
    return result


def validate_depth(depth, policy, *, now_ns):
    validate_policy_packet(policy)
    now = _ns(now_ns, 'depth decision')
    if not isinstance(depth, dict) or set(depth) != FIELDS:
        raise SensorContractError('exact depth-only fields required')
    if (depth['schema'] != SCHEMA or depth['calibration_id'] != CALIBRATION_ID
            or depth['representation'] != 'optical_axis_depth_m' or depth['hardware_calibrated'] is not False):
        raise SensorContractError('explicit unqualified optical-depth calibration required')
    measured = _ns(depth['measured_ns'], 'depth measured')
    available = _ns(depth['available_ns'], 'depth available')
    if (_ns(depth['decision_ns'], 'depth decision') != now or policy['sensor_state']['decision_ns'] != now
            or measured != policy['image']['measured_ns'] or not measured <= available <= now
            or now-measured > 100_000_000):
        raise SensorContractError('causal current RGB-aligned depth clock required')
    if _identity(depth['identity']) != policy['sensor_state']['identity'] or depth['rgb_sha256'] != rgb_digest(policy):
        raise SensorContractError('depth episode or current RGB binding mismatch')
    values, valid = depth['depth_m'], depth['valid']
    if (not isinstance(values, np.ndarray) or not isinstance(valid, np.ndarray)
            or values.shape != (HEIGHT, WIDTH) or values.dtype != np.float32
            or valid.shape != values.shape or valid.dtype != bool or not np.isfinite(values).all()
            or np.any(values[~valid] != 0.) or np.any(values[valid] < MIN_DEPTH_M)
            or np.any(values[valid] > MAX_DEPTH_M)):
        raise SensorContractError('metric depth and explicit unknown-ray mask required')


class CausalDepthHistory:
    """Four acquired frames, fixed episode and 10-Hz cadence; faults latch."""
    def __init__(self):
        self.frames = deque(maxlen=4)
        self.identity = None
        self.last_ns = None
        self.failed = False

    def push(self, depth, policy, *, now_ns):
        if self.failed:
            raise SensorContractError('depth history fault latched')
        try:
            validate_depth(depth, policy, now_ns=now_ns)
            identity = _identity(depth['identity'])
            if self.identity is not None and identity != self.identity:
                raise SensorContractError('depth history episode changed')
            if self.last_ns is not None and depth['measured_ns']-self.last_ns != 100_000_000:
                raise SensorContractError('depth history gap or duplicate')
            self.frames.append(deepcopy(depth))
            self.identity, self.last_ns = identity, depth['measured_ns']
        except (ValueError, TypeError, KeyError) as error:
            self.failed = True
            raise SensorContractError('causal depth history failure; motion must stop') from error

    def snapshot(self):
        return deepcopy(list(self.frames))


def body_points(depth, policy, *, now_ns, stride=8):
    """Current measured surface points; no whole-volume/free-space certificate."""
    validate_depth(depth, policy, now_ns=now_ns)
    if type(stride) is not int or stride < 1 or HEIGHT % stride or WIDTH % stride:
        raise ValueError('native pixel-grid divisor required')
    rows = np.arange(stride//2, HEIGHT, stride)
    columns = np.arange(stride//2, WIDTH, stride)
    u, v = np.meshgrid(columns+.5, rows+.5)
    z = depth['depth_m'][np.ix_(rows, columns)]
    valid = depth['valid'][np.ix_(rows, columns)]
    optical = np.stack((z*(u-320)/FOCAL, z*(v-240)/FOCAL, z), axis=-1)
    transform = np.asarray(BODY_FROM_OPTICAL)
    points = optical@transform[:3, :3].T+transform[:3, 3]
    return {'points_body_m': np.where(valid[..., None], points, np.nan), 'valid': valid.copy(),
            'rows': rows, 'columns': columns, 'measured_ns': depth['measured_ns'],
            'calibration_id': CALIBRATION_ID, 'environment_clearance_qualified': False,
            'scope': 'observed surface only; invalid and unobserved rays remain unknown'}

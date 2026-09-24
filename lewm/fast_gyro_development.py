"""Separate causal500 Hz body-gyro channel; existing learned tensors unchanged."""
import copy

import numpy as np

from lewm.causal_sensor_state import CausalSensorBuffer, SensorSchema, SensorContractError, _identity, _ns
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.simulated_body_observation_development import validate_policy_packet

CALIBRATION = 'ideal-body-gyro-500hz-development-v1'
SCHEMA_ID = 'causal_fast_body_gyro_development.v1'
SCHEMA = SensorSchema('gyro', ('wx', 'wy', 'wz'), ('rad/s',) * 3, 51, 100_000_000, CALIBRATION)


class FastGyroBuffer:
    def __init__(self, identity):
        self.identity = _identity(identity)
        self.buffer = CausalSensorBuffer((SCHEMA,), capacity_per_sensor=256)
        self.buffer.begin_episode(self.identity)

    def append(self, values, valid, *, measured_ns, available_ns):
        self.buffer.append('gyro', values, valid, measured_ns=measured_ns, available_ns=available_ns,
                           identity=self.identity, calibration_id=CALIBRATION)

    def packet(self, *, now_ns):
        # Incomplete terminal/startup histories may be logged; consumers require
        # the complete validated interval and cannot treat padding as a sample.
        state = self.buffer.snapshot(image_ns=now_ns, decision_ns=now_ns,
                                     identity=self.identity, sensor_anchor='decision')
        return {'schema': SCHEMA_ID, 'identity': self.identity, 'decision_ns': now_ns,
                **state['sensed']['gyro']}


def validate_fast_packet(fast, slow, *, now_ns):
    validate_policy_packet(slow)
    now = _ns(now_ns, 'fast gyro decision')
    if (set(fast) != {'schema', 'identity', 'decision_ns', 'values', 'valid', 'measured_ns', 'available_ns',
                      'channels', 'units', 'calibration_id'}
            or fast['schema'] != SCHEMA_ID or fast['calibration_id'] != CALIBRATION
            or tuple(fast['channels']) != SCHEMA.channels or tuple(fast['units']) != SCHEMA.units):
        raise SensorContractError('strict declared fast gyro channel required')
    identity = _identity(fast['identity'])
    if (identity != _identity(slow['sensor_state']['identity']) or now % 100_000_000
            or _ns(fast['decision_ns'], 'fast packet clock') != now
            or slow['sensor_state']['decision_ns'] != now or slow['image']['measured_ns'] != now):
        raise SensorContractError('co-timed current same-episode fast/slow packets required')
    data, valid = np.asarray(fast['values']), np.asarray(fast['valid'])
    times, available = np.asarray(fast['measured_ns']), np.asarray(fast['available_ns'])
    if (data.shape != (51, 3) or valid.shape != data.shape or valid.dtype != bool or not valid.all()
            or not np.isfinite(data).all() or times.shape != (51,) or available.shape != times.shape
            or times.dtype.kind not in 'iu' or available.dtype.kind not in 'iu'
            or not np.array_equal(times, now - 100_000_000 + np.arange(51, dtype=np.int64) * 2_000_000)
            or np.any(available < times) or np.any(available > now) or np.any(np.diff(available) < 0)):
        raise SensorContractError('complete finite causal2-ms history required')
    gyro = slow['sensor_state']['sensed']['gyro']
    for i, time in enumerate(times):
        matches = np.flatnonzero(gyro['measured_ns'] == time)
        if len(matches) and (not gyro['valid'][matches[0]].all() or not np.array_equal(data[i], gyro['values'][matches[0]])):
            raise SensorContractError('shared fast/slow ideal gyro measurements disagree')
    if not np.array_equal(gyro['measured_ns'][-6:], times[::10]):
        raise SensorContractError('six co-timed slow gyro samples required')
    return identity


class FastRelativeOrientation:
    """Causal midpoint integration; no pose, geometry, bias or hardware claim."""
    def __init__(self):
        self.status = 'NEW'
        self.rotation = np.eye(3)
        self.identity = self.last_ns = self.start_ns = None
        self.previous = None
        self.samples_integrated = 0

    def _result(self):
        return {'rotation_initial_body_from_current_body': self.rotation.tolist(), 'decision_ns': self.last_ns,
                'start_ns': self.start_ns, 'samples_integrated': self.samples_integrated, 'gyro_rate_hz': 500,
                'integration': 'causal_midpoint', 'hardware_calibrated': False}

    def begin(self, slow, fast, *, now_ns):
        if self.status != 'NEW':
            raise SensorContractError('fresh high-rate orientation required')
        try:
            self.identity = validate_fast_packet(fast, slow, now_ns=now_ns)
            self.last_ns = self.start_ns = int(now_ns)
            self.previous = copy.deepcopy(fast)
            self.status = 'ACTIVE'
            return self._result()
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('fast orientation initialization failed') from error

    def step(self, slow, fast, *, now_ns):
        if self.status != 'ACTIVE':
            raise SensorContractError('active high-rate orientation required')
        try:
            identity = validate_fast_packet(fast, slow, now_ns=now_ns)
            if identity != self.identity or now_ns - self.last_ns != 100_000_000:
                raise SensorContractError('fast orientation reset or packet gap')
            for field in ('values', 'valid', 'measured_ns', 'available_ns'):
                if not np.array_equal(np.asarray(self.previous[field])[-1], np.asarray(fast[field])[0]):
                    raise SensorContractError('fast boundary measurement rewritten')
            rates = np.asarray(fast['values'])
            updated = self.rotation.copy()
            for previous, current in zip(rates[:-1], rates[1:], strict=True):
                updated = updated @ rotation_increment((previous + current) * .001)
            if not np.allclose(updated.T @ updated, np.eye(3), rtol=0, atol=1e-8) or abs(np.linalg.det(updated) - 1) > 1e-8:
                raise SensorContractError('invalid fast integrated rotation')
            self.rotation, self.last_ns = updated, int(now_ns)
            self.previous = copy.deepcopy(fast)
            self.samples_integrated += 50
            return self._result()
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('fast orientation update failed') from error

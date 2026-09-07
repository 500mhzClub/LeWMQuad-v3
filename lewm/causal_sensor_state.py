"""Development causal sensor history; independent of all frozen experiments.

Measurement time and availability time are different. A sample can enter a
decision only if it was measured by the declared anchor and arrived by selection.
This module does not discover logs, estimate odometry, or provide calibration.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np


class SensorContractError(ValueError):
    pass


def _ns(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not 0 <= value <= np.iinfo(np.int64).max):
        raise SensorContractError(f'{name} must be a nonnegative integer nanosecond timestamp')
    return int(value)


def _identity(value):
    if not isinstance(value, tuple) or len(value) != 3:
        raise SensorContractError('identity must be (environment, episode, reset)')
    return tuple(_ns(item, 'episode identity') for item in value)


@dataclass(frozen=True)
class SensorSchema:
    name: str
    channels: tuple[str, ...]
    units: tuple[str, ...]
    history_length: int
    max_age_ns: int
    calibration_id: str
    role: str = 'sensed'

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name
                or not isinstance(self.calibration_id, str) or not self.calibration_id
                or self.role not in ('sensed', 'control')):
            raise SensorContractError('schema needs name, calibration identity, and sensed/control role')
        if (not isinstance(self.channels, tuple) or not isinstance(self.units, tuple)
                or not self.channels or len(self.channels) != len(self.units)
                or len(set(self.channels)) != len(self.channels)
                or not all(isinstance(s, str) and s for s in self.channels + self.units)):
            raise SensorContractError('ordered unique channels and explicit units required')
        if _ns(self.history_length, 'history length') == 0:
            raise SensorContractError('history length must be positive')
        _ns(self.max_age_ns, 'maximum sample age')


class CausalSensorBuffer:
    """Bounded, explicit-episode storage for declared deployable modalities.

    Out-of-order measurement timestamps and duplicate samples are rejected, not
    silently reordered or overwritten. The adapter must resolve those cases before
    insertion. Unavailable/invalid channels never receive imputed confident values.
    """

    def __init__(self, schemas: tuple[SensorSchema, ...], *, capacity_per_sensor=256):
        if not schemas or len({s.name for s in schemas}) != len(schemas):
            raise SensorContractError('at least one uniquely named sensor is required')
        if _ns(capacity_per_sensor, 'capacity') < max(s.history_length for s in schemas):
            raise SensorContractError('capacity cannot be shorter than requested history')
        self.schemas = {schema.name: schema for schema in schemas}
        self._samples = {name: deque(maxlen=capacity_per_sensor) for name in self.schemas}
        self._episode = None

    def begin_episode(self, identity):
        identity = _identity(identity)
        if identity == self._episode:
            raise SensorContractError('duplicate episode reset; increment reset identity explicitly')
        self._episode = identity
        for samples in self._samples.values():
            samples.clear()

    def append(self, name, values, valid, *, measured_ns, available_ns, identity, calibration_id):
        if self._episode is None or _identity(identity) != self._episode:
            raise SensorContractError('sample belongs to an inactive environment/episode/reset')
        if name not in self.schemas:
            raise SensorContractError('undeclared sensor input')
        schema = self.schemas[name]
        if calibration_id != schema.calibration_id:
            raise SensorContractError('calibration identity mismatch')
        measured_ns, available_ns = _ns(measured_ns, 'measurement'), _ns(available_ns, 'availability')
        if available_ns < measured_ns:
            raise SensorContractError('availability precedes measurement; clocks must be aligned')
        try:
            data, mask = np.asarray(values, dtype=np.float64), np.asarray(valid)
        except (TypeError, ValueError) as exc:
            raise SensorContractError('sensor values must be numeric vectors') from exc
        if data.shape != (len(schema.channels),) or mask.shape != data.shape or mask.dtype != np.bool_:
            raise SensorContractError('sensor vector and explicit boolean validity shape mismatch')
        if not np.isfinite(data[mask]).all():
            raise SensorContractError('nonfinite measurement marked valid')
        samples = self._samples[name]
        if samples and (measured_ns <= samples[-1][0] or available_ns < samples[-1][1]):
            raise SensorContractError('duplicate or out-of-order sensor sample')
        samples.append((measured_ns, available_ns, np.where(mask, data, 0.0).copy(), mask.copy()))

    def snapshot(self, *, image_ns, decision_ns, identity, sensor_anchor='image'):
        """Build image-anchored training or decision-anchored online history.

        Explicit decision anchoring permits newer IMU/control samples after a
        delayed image, but never samples unavailable at selection. Consumers
        receive both times and must train with the same anchoring convention.
        """
        if self._episode is None or _identity(identity) != self._episode:
            raise SensorContractError('snapshot episode does not match active episode')
        image_ns, decision_ns = _ns(image_ns, 'image'), _ns(decision_ns, 'decision')
        if decision_ns < image_ns:
            raise SensorContractError('decision precedes its image')
        if sensor_anchor not in ('image', 'decision'):
            raise SensorContractError('sensor anchor must be image or decision')
        anchor_ns = image_ns if sensor_anchor == 'image' else decision_ns
        result = {'identity': self._episode, 'image_ns': image_ns, 'decision_ns': decision_ns,
                  'sensor_anchor': sensor_anchor, 'sensor_anchor_ns': anchor_ns,
                  'sensed': {}, 'control': {}}
        for name, schema in self.schemas.items():
            shape = (schema.history_length, len(schema.channels))
            values, valid = np.zeros(shape), np.zeros(shape, dtype=bool)
            measured = np.full(schema.history_length, -1, dtype=np.int64)
            available = np.full(schema.history_length, -1, dtype=np.int64)
            # Age is relative to the actual decision, not merely an old image.
            eligible = [sample for sample in self._samples[name]
                        if sample[0] <= anchor_ns and sample[1] <= decision_ns
                        and decision_ns - sample[0] <= schema.max_age_ns][-schema.history_length:]
            start = schema.history_length - len(eligible)
            for i, (mt, at, data, mask) in enumerate(eligible, start=start):
                measured[i], available[i], values[i], valid[i] = mt, at, data, mask
            result[schema.role][name] = {
                'values': values, 'valid': valid, 'measured_ns': measured, 'available_ns': available,
                'channels': schema.channels, 'units': schema.units, 'calibration_id': schema.calibration_id,
            }
        return result


def reorder_named_channels(values, valid, source_names, target_names):
    """Explicit name-based adapter, preserving missingness and channel identity.

    Useful for translating Unitree motor order into a model's declared joint
    order. It does not infer units or calibrate sensors. Unrequested source
    channels are omitted; missing target channels remain invalid zero-filled.
    """
    for names in (source_names, target_names):
        if (not isinstance(names, tuple) or not all(isinstance(n, str) and n for n in names)
                or len(set(names)) != len(names)):
            raise SensorContractError('channel names must be unique ordered tuples')
    try:
        data, mask = np.asarray(values, dtype=np.float64), np.asarray(valid)
    except (TypeError, ValueError) as exc:
        raise SensorContractError('channel data must be numeric') from exc
    if data.shape != (len(source_names),) or mask.shape != data.shape or mask.dtype != np.bool_:
        raise SensorContractError('source channel shape or validity mismatch')
    if not np.isfinite(data[mask]).all():
        raise SensorContractError('nonfinite channel marked valid')
    lookup = {name: index for index, name in enumerate(source_names)}
    output, output_valid = np.zeros(len(target_names)), np.zeros(len(target_names), dtype=bool)
    missing = []
    for index, name in enumerate(target_names):
        if name not in lookup:
            missing.append(name)
        elif mask[lookup[name]]:
            output[index], output_valid[index] = data[lookup[name]], True
    return {'values': output, 'valid': output_valid, 'missing_channels': tuple(missing)}


def build_decision_packet(buffer, rgb, *, image_ns, image_available_ns, decision_ns,
                          identity, camera_calibration_id, expected_calibration_id,
                          expected_rgb_shape, max_image_age_ns):
    """Fuse a received RGB frame with decision-time sensor history.

    No world pose, future frame, oracle target or candidate outcome is accepted.
    This defines the prospective online boundary, not a frozen model adapter.
    Clock alignment and calibration must already have been established by the
    acquisition system. A stale image fails explicitly; no blind-motion fallback
    is invented here.
    """
    image_ns = _ns(image_ns, 'image measurement')
    image_available_ns = _ns(image_available_ns, 'image availability')
    decision_ns = _ns(decision_ns, 'decision')
    max_age = _ns(max_image_age_ns, 'maximum image age')
    if not image_ns <= image_available_ns <= decision_ns:
        raise SensorContractError('image was unavailable or image clocks are inconsistent')
    if decision_ns - image_ns > max_age:
        raise SensorContractError('image is stale at action selection')
    if (not isinstance(expected_calibration_id, str) or not expected_calibration_id
            or camera_calibration_id != expected_calibration_id):
        raise SensorContractError('camera calibration identity mismatch')
    if (not isinstance(expected_rgb_shape, tuple) or len(expected_rgb_shape) != 3
            or expected_rgb_shape[2] != 3
            or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0
                   for n in expected_rgb_shape)):
        raise SensorContractError('explicit positive HWC RGB shape required')
    rgb = np.asarray(rgb)
    if rgb.shape != expected_rgb_shape or rgb.dtype != np.uint8:
        raise SensorContractError('RGB shape or uint8 encoding mismatch')
    return {
        'image': {'rgb': np.array(rgb, copy=True, order='C'), 'measured_ns': image_ns,
                  'available_ns': image_available_ns, 'calibration_id': camera_calibration_id},
        'sensor_state': buffer.snapshot(image_ns=image_ns, decision_ns=decision_ns,
                                         identity=identity, sensor_anchor='decision'),
    }

"""Fixed causal gravity-feedback hypotheses, not hardware attitude calibration.

Specific force is gravity plus acceleration in the body frame. Quiet/steady
history gates reduce some contamination but cannot identify constant unknown
acceleration. No output claims a calibrated normal, metric clearance, or yaw.
"""
import copy
import math

import numpy as np

from lewm.causal_ground_plane_development import CausalGroundPlane, foot_sphere_centres_body
from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.simulated_body_observation_development import validate_policy_packet

MODES = ('gyro_only', 'body_mean_feedback', 'transported_feedback')
TIME_CONSTANT_SECONDS = 2.


def force_in_current_frame(force, gyro):
    """Transport all20 actual force vectors before averaging, using gyro only."""
    force, gyro = np.asarray(force, dtype=float), np.asarray(gyro, dtype=float)
    if force.shape != (20, 3) or gyro.shape != (20, 3) or not np.isfinite(force).all() or not np.isfinite(gyro).all():
        raise SensorContractError('twenty finite co-timed force/gyro samples required')
    rotation = np.eye(3)
    rotated = [force[0]]
    for i in range(1, 20):
        rotation = rotation @ rotation_increment((gyro[i - 1] + gyro[i]) * .01)
        rotated.append(rotation @ force[i])
    return np.stack(rotated) @ rotation


def force_observation(packet, mode):
    if mode not in MODES:
        raise ValueError('fixed gravity estimator mode required')
    validate_policy_packet(packet)
    sensed = packet['sensor_state']['sensed']
    force, gyro = sensed['specific_force'], sensed['gyro']
    commands = packet['sensor_state']['control']['applied_command']
    if (not np.asarray(gyro['valid']).all() or not np.all(np.diff(gyro['measured_ns']) == 20_000_000)
            or not np.array_equal(force['measured_ns'], gyro['measured_ns'])):
        raise SensorContractError('complete co-timed regular gyro/force history required')
    result = {'accepted': False, 'reason': None, 'mean_force_current_body': None,
              'mean_magnitude_m_s2': None, 'residual_rms_m_s2': None,
              'acceleration_separation_qualified': False}
    if mode == 'gyro_only':
        return result | {'reason': 'baseline_no_force_feedback'}
    if not np.asarray(force['valid']).all():
        return result | {'reason': 'force_unavailable'}
    if (not np.asarray(commands['valid'])[-5:].all()
            or not np.all(np.diff(np.asarray(commands['measured_ns'])[-5:]) == 100_000_000)
            or commands['measured_ns'][-1] != packet['sensor_state']['decision_ns']):
        return result | {'reason': 'command_history_unavailable'}
    if np.any(np.ptp(np.asarray(commands['values'])[-5:], axis=0) > .05 + 1e-12):
        return result | {'reason': 'recent_command_change'}
    values = np.asarray(force['values'], dtype=float)
    if mode == 'transported_feedback':
        values = force_in_current_frame(values, gyro['values'])
    mean = values.mean(0)
    magnitude = float(np.linalg.norm(mean))
    residual = float(np.sqrt(np.mean(np.sum((values - mean) ** 2, axis=1))))
    result.update(mean_force_current_body=mean.tolist(), mean_magnitude_m_s2=magnitude, residual_rms_m_s2=residual)
    if abs(magnitude - 9.81) > .75:
        return result | {'reason': 'mean_force_not_near_gravity'}
    if residual > 3.:
        return result | {'reason': 'force_history_not_quiet'}
    return result | {'accepted': True, 'reason': 'conditional_quiet_steady_force_hypothesis'}


class CausalGravityFeedbackGround:
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError('fixed gravity estimator mode required')
        self.mode = mode
        self.baseline = CausalGroundPlane()
        self.status = 'NEW'
        self.last_ns = None
        self._state = self._up = self._rotation = self._history = None

    def _check_history(self, packet):
        current = {'force': packet['sensor_state']['sensed']['specific_force'],
                   'command': packet['sensor_state']['control']['applied_command']}
        if self._history is not None:
            for name, history in current.items():
                old = self._history[name]
                lookup = {int(t): i for i, t in enumerate(old['measured_ns']) if t >= 0}
                for i, time in enumerate(history['measured_ns']):
                    if int(time) not in lookup:
                        continue
                    for field in ('values', 'valid', 'available_ns'):
                        if not np.array_equal(np.asarray(history[field])[i], np.asarray(old[field])[lookup[int(time)]]):
                            raise SensorContractError('force/control history rewritten')
        return copy.deepcopy(current)

    def begin(self, packet, *, now_ns):
        if self.status != 'NEW':
            raise SensorContractError('fresh gravity-feedback estimator required')
        try:
            state = self.baseline.begin(packet, now_ns=now_ns)
            observation = force_observation(packet, self.mode)
            history = self._check_history(packet)
            self._up = np.asarray(state['up_current_body'])
            self._rotation = np.eye(3)
            self._state = state | {'estimator_mode': self.mode, 'feedback': observation,
                                   'feedback_applied': False, 'feedback_time_constant_s': TIME_CONSTANT_SECONDS}
            self._history = history
            self.last_ns = now_ns
            self.status = 'ACTIVE'
            return self.snapshot(now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('gravity feedback initialization failed') from error

    def step(self, packet, *, now_ns):
        if self.status != 'ACTIVE':
            raise SensorContractError('active gravity-feedback estimator required')
        try:
            state = self.baseline.step(packet, now_ns=now_ns)
            history = self._check_history(packet)
            observation = force_observation(packet, self.mode)
            rotation = np.asarray(self.baseline.orientation.snapshot(now_ns=now_ns)['rotation_initial_body_from_current_body'])
            up = rotation.T @ self._rotation @ self._up
            if observation['accepted']:
                observed = np.asarray(observation['mean_force_current_body']) / observation['mean_magnitude_m_s2']
                alpha = 1 - math.exp(-.1 / TIME_CONSTANT_SECONDS)
                up = (1 - alpha) * up + alpha * observed
                up /= np.linalg.norm(up)
            if self.mode == 'gyro_only':
                # Exact frozen baseline witness, avoiding accumulated regrouping
                # roundoff in a mathematically equivalent transport composition.
                up = np.asarray(state['up_current_body'])
            feet = foot_sphere_centres_body(packet['sensor_state']['sensed']['joints']['values'][-1, :12])
            heights = .022 - feet @ up
            height = float(heights.max())
            if not .1 <= height <= .6 or not np.isfinite(up).all() or abs(np.linalg.norm(up) - 1) > 1e-8:
                raise SensorContractError('gravity-feedback plane outside kinematic envelope')
            self._state = state | {'up_current_body': up.tolist(), 'body_origin_height_m': height,
                                   'per_foot_support_height_m': heights.tolist(), 'estimator_mode': self.mode,
                                   'feedback': observation, 'feedback_applied': observation['accepted'],
                                   'feedback_time_constant_s': TIME_CONSTANT_SECONDS,
                                   'assumptions': state['assumptions'] + ['feedback only conditionally treats quiet steady specific force as gravity']}
            self._history = history
            self._rotation, self._up, self.last_ns = rotation, up, now_ns
            return self.snapshot(now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('gravity feedback update failed') from error

    def snapshot(self, *, now_ns):
        now_ns = _ns(now_ns, 'gravity feedback query clock')
        if self.status != 'ACTIVE' or now_ns != self.last_ns:
            raise SensorContractError('fresh active gravity feedback required')
        return copy.deepcopy(self._state)

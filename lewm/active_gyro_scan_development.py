"""Bounded four-view-plus-return active scan using actual body gyro history."""
import copy
import math

import numpy as np

from lewm.causal_relative_orientation_development import CausalRelativeOrientation
from lewm.causal_sensor_state import SensorContractError


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ActiveGyroScan:
    """Four positive quarter-turn targets in one initial-body reference frame.

    Thirty-second global deadline,0.35 rad/s command limit,0.08 rad heading /
    0.1 rad/s measured-rate tolerance and0.3 s dwell. No translation/clearance
    guarantee; the physical executor must preserve native contact termination.
    """
    def __init__(self):
        self.orientation = CausalRelativeOrientation()
        self.status = 'NEW'
        self.start_ns = self.last_ns = self.stable_since = None
        self.target_index = 1
        self.views = []
        self._last = None

    def _decision(self, packet, attitude):
        now = packet['sensor_state']['decision_ns']
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        forward = rotation[:, 0]
        if np.linalg.norm(forward[:2]) < .2:
            raise SensorContractError('scan forward direction near vertical')
        heading = math.atan2(forward[1], forward[0])
        omega = rotation @ packet['sensor_state']['sensed']['gyro']['values'][-1]
        derivative = np.cross(omega, forward)
        rate = float((forward[0] * derivative[1] - forward[1] * derivative[0]) / (forward[0] ** 2 + forward[1] ** 2))
        error = wrap(self.target_index * math.pi / 2 - heading)
        view = None
        quiet = abs(error) <= .08 and abs(rate) <= .1
        if quiet:
            if self.stable_since is None:
                self.stable_since = now
            if now - self.stable_since >= 300_000_000:
                view = {'view_index': self.target_index, 'decision_ns': now, 'relative_heading_rad': heading,
                        'rotation_initial_body_from_current_body': rotation.tolist()}
                self.views.append(view)
                self.target_index += 1
                self.stable_since = None
                if self.target_index == 5:
                    self.status = 'COMPLETE'
        else:
            self.stable_since = None
        if now - self.start_ns >= 30_000_000_000 and self.status != 'COMPLETE':
            self.status = 'FAILED_TIMEOUT'
        if self.status == 'COMPLETE' or self.status.startswith('FAILED_'):
            command = [0., 0., 0.]
        elif view is not None:
            error = wrap(self.target_index * math.pi / 2 - heading)
            self.status = 'SCANNING'
            command = [0., 0., float(np.clip(1.5 * error, -.35, .35))]
        elif quiet:
            self.status = 'DWELLING'
            command = [0., 0., 0.]
        else:
            self.status = 'SCANNING'
            command = [0., 0., float(np.clip(1.5 * error, -.35, .35))]
        result = {'status': self.status, 'decision_ns': now, 'requested_command': command,
                  'target_index': self.target_index, 'relative_heading_rad': heading,
                  'heading_error_rad': error, 'projected_heading_rate_rad_s': rate,
                  'new_completed_view': view, 'completed_target_views': len(self.views) - 1,
                  'rotation_initial_body_from_current_body': rotation.tolist(),
                  'translation_compensated': False, 'metric_clearance_qualified': False}
        self._last = copy.deepcopy(result)
        self.last_ns = now
        return result

    def begin(self, packet, *, now_ns):
        if self.status != 'NEW':
            raise SensorContractError('fresh scan required')
        try:
            attitude = self.orientation.begin(packet, now_ns=now_ns)
            self.start_ns = now_ns
            self.views = [{'view_index': 0, 'decision_ns': now_ns, 'relative_heading_rad': 0.,
                           'rotation_initial_body_from_current_body': np.eye(3).tolist()}]
            self.status = 'SCANNING'
            return self._decision(packet, attitude)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('scan initialization failed') from error

    def step(self, packet, *, now_ns):
        if self.status not in ('SCANNING', 'DWELLING'):
            raise SensorContractError('active scan required; terminal action is explicit zero')
        try:
            attitude = self.orientation.step(packet, now_ns=now_ns)
            return self._decision(packet, attitude)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('scan sensor update failed') from error

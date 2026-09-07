"""Select an observed side branch; directions are relative, not map coordinates."""
import copy
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.rgb_exit_candidates_development import ExitCandidate


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def observed_branch(candidate, rotation, *, decision_ns):
    proposal = ExitCandidate(**candidate)
    rotation = np.asarray(rotation, dtype=float)
    if (proposal.timestamp_ns != decision_ns or rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-8)):
        raise SensorContractError('current proposal and relative rotation required')
    direction = rotation@np.array([math.cos(proposal.bearing_body_rad), math.sin(proposal.bearing_body_rad), 0.])
    if np.linalg.norm(direction[:2]) < .2:
        raise SensorContractError('near-vertical branch bearing')
    return {'candidate': copy.deepcopy(candidate), 'observed_ns': decision_ns,
            'direction_initial_body': direction.tolist(), 'translation_compensated': False,
            'qualified_exit': False, 'qualified_traversal': False}


def choose_side_branch(observations, incoming_direction, *, now_ns):
    incoming = np.asarray(incoming_direction, dtype=float)
    if (incoming.shape != (3,) or not np.isfinite(incoming).all()
            or np.linalg.norm(incoming[:2]) < .2 or type(now_ns) is not int):
        raise SensorContractError('finite observed incoming direction and clock required')
    heading = math.atan2(incoming[1], incoming[0])
    eligible = []
    for item in observations:
        if (item['qualified_exit'] is not False or item['qualified_traversal'] is not False
                or not 0 <= now_ns-item['observed_ns'] <= 30_000_000_000):
            raise SensorContractError('bounded-age unqualified scan observation required')
        direction = np.asarray(item['direction_initial_body'], dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction[:2]) < .2:
            raise SensorContractError('finite observed branch direction required')
        delta = wrap(math.atan2(direction[1], direction[0])-heading)
        if math.pi/4 <= abs(delta) <= 3*math.pi/4:
            # Fixed left-before-right exploration ordering. No expected fixture
            # exit, true place label or destination coordinate is consulted.
            key = (delta < 0, abs(abs(delta)-math.pi/2), -item['candidate']['support_points'], -item['observed_ns'])
            eligible.append((key, item, delta))
    if not eligible: return None
    _, item, delta = min(eligible, key=lambda row: row[0])
    return {**copy.deepcopy(item), 'selected_ns': now_ns, 'relative_to_incoming_rad': delta,
            'requires_fresh_forward_reobservation': True, 'place_identity': None}


class RelativeBearingAlignment:
    """Bounded alignment using the wrapper's uninterrupted gyro reference."""
    def __init__(self, direction_initial_body):
        self.direction = np.asarray(direction_initial_body, dtype=float).copy()
        if self.direction.shape != (3,) or not np.isfinite(self.direction).all() or np.linalg.norm(self.direction[:2]) < .2:
            raise SensorContractError('finite relative target bearing required')
        self.start_ns = self.last_ns = self.quiet_since = None
        self.terminal = False

    def observe(self, packet, attitude, *, now_ns):
        if self.terminal or (self.last_ns is not None and now_ns-self.last_ns != 100_000_000):
            raise SensorContractError('active consecutive alignment required')
        if attitude['decision_ns'] != now_ns or packet['sensor_state']['decision_ns'] != now_ns:
            raise SensorContractError('current causal attitude required')
        if self.start_ns is None: self.start_ns = now_ns
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        forward = rotation[:, 0]
        if np.linalg.norm(forward[:2]) < .2: raise SensorContractError('near-vertical body heading')
        heading = math.atan2(forward[1], forward[0])
        target = math.atan2(self.direction[1], self.direction[0])
        error = wrap(target-heading)
        omega = rotation@packet['sensor_state']['sensed']['gyro']['values'][-1]
        derivative = np.cross(omega, forward)
        rate = float((forward[0]*derivative[1]-forward[1]*derivative[0])/(forward[0]**2+forward[1]**2))
        quiet = abs(error) <= .08 and abs(rate) <= .1
        if quiet:
            if self.quiet_since is None: self.quiet_since = now_ns
        else: self.quiet_since = None
        complete = self.quiet_since is not None and now_ns-self.quiet_since >= 300_000_000
        status = 'COMPLETE' if complete else 'FAILED_TIMEOUT' if now_ns-self.start_ns >= 12_000_000_000 else 'ALIGNING'
        self.terminal = status != 'ALIGNING'
        self.last_ns = now_ns
        return {'status': status, 'decision_ns': now_ns, 'heading_error_rad': error,
                'projected_heading_rate_rad_s': rate,
                'requested_command': [0., 0., 0. if self.terminal or quiet else float(np.clip(1.5*error, -.35, .35))],
                'translation_compensated': False, 'clearance_qualified': False}

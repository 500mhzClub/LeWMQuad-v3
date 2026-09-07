"""Fresh observed initial bearing alignment before the frozen continuation.

This is a new, unqualified development intervention, not a retrofit to the
completed panel. A floor-extension bearing is not a corridor-center certificate.
"""
import copy
import hashlib
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import wrap, observed_branch
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.observed_continuation_development import ObservedContinuation
from lewm.rgb_exit_candidates_development import observe_exit_candidates


class FineInitialBearingAlignment:
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
        quiet = abs(error) <= .02 and abs(rate) <= .1
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


class InitiallyAlignedContinuation(ObservedContinuation):
    def __init__(self, method, geometry, template=None):
        super().__init__(method, geometry, template)
        self.stage = 'INITIAL_OBSERVE'
        self.initial_ground = CausalGravityFeedbackGround('transported_feedback')
        self.initial_bearing = self.initial_turn = None

    def _observe(self, packet, fast_packet, *, now_ns):
        if not self.stage.startswith('INITIAL_'):
            return super()._observe(packet, fast_packet, now_ns=now_ns)
        self.tick += 1
        self.history.push(packet, now_ns=now_ns)
        attitude = (self.orientation.begin(packet, fast_packet, now_ns=now_ns) if self.tick == 0
                    else self.orientation.step(packet, fast_packet, now_ns=now_ns))
        if self.start_ns is None: self.start_ns = now_ns
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        stage = self.stage
        command = [0., 0., 0.]
        proposal_rows = selected = turn = None
        if stage == 'INITIAL_OBSERVE':
            ground = (self.initial_ground.begin(packet, now_ns=now_ns) if self.tick == 0
                      else self.initial_ground.step(packet, now_ns=now_ns))
            if self.tick == 3:
                image_id = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()
                proposal_rows = observe_exit_candidates(packet, ground, now_ns=now_ns, observation_id=image_id)['candidate_rows']
                eligible = [row for row in proposal_rows if abs(row['bearing_body_rad']) <= .35]
                if not eligible:
                    self.status = 'FAILED_INITIAL_NO_EXIT'
                else:
                    candidate = min(eligible, key=lambda row: (abs(row['bearing_body_rad']), -row['support_points']))
                    self.initial_bearing = observed_branch(candidate, rotation, decision_ns=now_ns)
                    selected = copy.deepcopy(self.initial_bearing)
                    self.initial_turn = FineInitialBearingAlignment(self.initial_bearing['direction_initial_body'])
                    self.stage = 'INITIAL_ALIGN'
        elif stage == 'INITIAL_ALIGN':
            turn = self.initial_turn.observe(packet, attitude, now_ns=now_ns)
            command = turn['requested_command']
            if turn['status'] == 'COMPLETE':
                self.stage, self.hold_since = 'INITIAL_HOLD', now_ns
            elif turn['status'].startswith('FAILED_'):
                self.status = 'FAILED_INITIAL_ALIGNMENT_'+turn['status'].removeprefix('FAILED_')
        elif stage == 'INITIAL_HOLD':
            if now_ns-self.hold_since >= 1_500_000_000:
                self.stage = 'FIRST'
        else: raise SensorContractError('unknown initial alignment stage')
        terminal = self.status != 'RUNNING'
        if terminal: command = [0., 0., 0.]
        return {'status': self.status, 'stage': stage, 'next_stage': self.stage,
                'decision_ns': now_ns, 'tick': self.tick, 'terminal': terminal,
                'requested_command': command, 'child': None, 'scan': None, 'turn': turn,
                'selected_view_proposals': None, 'selected_side_branch': None,
                'initial_proposal_rows': proposal_rows, 'selected_initial_bearing': selected,
                'global_orientation': attitude, 'ledgers': self.ledgers(), 'trusted_graph_edges': 0,
                'scope': 'observed initial fine heading alignment only; no centering, clearance or hardware qualification'}


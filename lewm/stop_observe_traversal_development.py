"""Acquire stopped arrival evidence at the existing translation cap.

Only the cap branch changes: enter bounded braking, retaining every existing
arrival predicate and settling deadline. No coordinate/map input or trusted edge.
"""
import hashlib
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.rgb_exit_candidates_development import observe_exit_candidates
from lewm.rgb_floor_evidence_development import observe_floor
from lewm.observed_traversal_controller_development import ObservedTraversalController
from lewm.fixed_forward_traversal_development import FixedForwardChoice


class StopObserveTraversal(ObservedTraversalController):
    def __init__(self, method, geometry, template=None):
        if method == 'fixed_forward':
            super().__init__('always_stop', geometry)
            self.method = method
            self.adapter = FixedForwardChoice()
        else:
            super().__init__(method, geometry, template)

    def _observe(self, packet, fast_packet, *, now_ns):
        self.tick += 1
        initial = self.tick == 0
        attitude = (self.orientation.begin(packet, fast_packet, now_ns=now_ns) if initial
                    else self.orientation.step(packet, fast_packet, now_ns=now_ns))
        ground = self.ground.begin(packet, now_ns=now_ns) if initial else self.ground.step(packet, now_ns=now_ns)
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        if not initial:
            command = packet['sensor_state']['control']['applied_command']
            if command['measured_ns'][-1] != now_ns or not command['valid'][-1].all():
                raise SensorContractError('actual current applied command required for progress proxy')
            velocity = np.array([*command['values'][-1,:2],0.])
            self.position += .05*(self.rotation+rotation)@velocity
        self.rotation, self.clock = rotation, now_ns
        body = self.geometry.observe(packet, now_ns=now_ns)
        floor = observe_floor(packet, now_ns=now_ns)
        # Fixed native grid; no future buffer or geometry enters appearance.
        mask = floor['floor_evidence_mask'][4::8,4::8]
        observation_id = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()
        if self.adapter is not None: self.adapter.observe(packet, now_ns=now_ns)
        selection = candidate = arrival = None
        progress = required = novelty = None
        quiet = False
        command = [0.,0.,0.]
        if self.tick < 3:
            self.status = 'WARMUP'
        elif self.tick == 3:
            proposals = observe_exit_candidates(packet, ground, now_ns=now_ns, observation_id=observation_id)['candidate_rows']
            eligible = [c for c in proposals if abs(c['bearing_body_rad']) <= .35]
            if not eligible:
                self.status = 'FAILED_NO_EXIT'
            else:
                candidate = min(eligible, key=lambda c: (abs(c['bearing_body_rad']), -c['support_points']))
                bearing = candidate['bearing_body_rad']
                current = np.array([math.cos(bearing),math.sin(bearing),0.])
                self.direction = rotation @ current
                self.start_position = self.position.copy()
                self.reference_mask = mask.copy()
                self.start_ns = now_ns
                self.status = 'TRAVERSING'
                self.ledger.begin(observation_id=observation_id, decision_ns=now_ns, candidate=candidate)
                if self.adapter is not None: self.adapter.begin_control(.8*current[:2], now_ns=now_ns)
        if self.status in ('TRAVERSING','BRAKING'):
            progress = float((self.position-self.start_position) @ self.direction)
            projection = self.geometry.supports(packet['sensor_state']['sensed']['joints']['values'][-1,:12],
                                                (rotation.T@self.direction)[None])
            required = max(.8, projection['upper'][0]-projection['lower'][0]+.35)
            novelty = float(np.mean(mask != self.reference_mask))
            self.changes = (self.changes+[novelty])[-3:]
            gyro = packet['sensor_state']['sensed']['gyro']
            joints = packet['sensor_state']['sensed']['joints']
            quiet = bool(gyro['valid'][-5:].all() and joints['valid'][-5:,12:].all()
                         and np.max(np.linalg.norm(gyro['values'][-5:],axis=1)) <= .15
                         and np.sqrt(np.mean(joints['values'][-5:,12:]**2)) <= 1.)
            boundary = (now_ns-self.start_ns) % 500_000_000 == 0
            if self.status == 'TRAVERSING' and boundary:
                visual_change = len(self.changes) == 3 and min(self.changes) >= .10
                if progress >= required and visual_change:
                    self.status = 'BRAKING'
                    self.brake_ns = now_ns
                elif progress >= 1.4:
                    self.status = 'BRAKING'
                    self.brake_ns = now_ns
                elif now_ns-self.start_ns >= 12_000_000_000:
                    self.status = 'FAILED_TIMEOUT'
            if self.status == 'BRAKING':
                if quiet:
                    if self.quiet_since is None: self.quiet_since = now_ns
                else: self.quiet_since = None
                if (now_ns-self.brake_ns >= 500_000_000 and self.quiet_since is not None
                        and now_ns-self.quiet_since >= 300_000_000):
                    if progress < required:
                        self.status = 'FAILED_PROGRESS'
                    elif min(self.changes) >= .10:
                        self.status = 'ARRIVAL_CANDIDATE'
                        arrival = {'observation_id': observation_id, 'decision_ns': now_ns,
                                   'command_progress_proxy_m': progress, 'required_body_extent_plus_margin_m': required,
                                   'floor_mask_change_fraction': novelty, 'body_quiet_proxy': quiet,
                                   'place_identity': None, 'qualified_arrival': False}
                    else: self.status = 'FAILED_NO_VISUAL_CHANGE'
                elif now_ns-self.brake_ns >= 2_000_000_000: self.status = 'FAILED_SETTLING'
            if self.status == 'TRAVERSING':
                if self.method == 'directional_gait':
                    direction = rotation.T@self.direction
                    command = [.2,0.,float(np.clip(1.5*math.atan2(direction[1],direction[0]),-.35,.35))]
                elif self.method != 'always_stop':
                    if boundary:
                        selection = self.adapter.select(now_ns=now_ns)
                        self.held = selection['requested_command_tape'][0]
                    command = list(self.held)
            if self.status not in ('TRAVERSING','BRAKING'):
                self.ledger.finish(status=self.status, arrival=arrival)
        terminal = self.status.startswith('FAILED_') or self.status == 'ARRIVAL_CANDIDATE'
        return {'status': self.status, 'decision_ns': now_ns, 'tick': self.tick, 'terminal': terminal,
                'requested_command': command, 'selection': selection, 'selected_exit_candidate': candidate,
                'command_progress_proxy_m': progress, 'required_progress_m': required,
                'floor_mask_change_fraction': novelty, 'quiet_body_proxy': quiet,
                'body_extent_lower_m': body['lower'], 'body_extent_upper_m': body['upper'],
                'fast_orientation': attitude, 'ground_state': ground, 'ledger': self.ledger.snapshot(),
                'scope': 'provisional one-transition integration; no trusted place, edge, clearance or navigation qualification'}

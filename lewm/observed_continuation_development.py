"""Continuous two-traversal prototype with observed intervening side-branch scan.

No destination/map/true-motion inputs; all ledgers and arrivals stay provisional.
Global gyro orientation persists through traversals, holds, scan and alignment.
"""
import copy
import hashlib

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.fast_gyro_scan_development import FastGyroScan
from lewm.online_rgb_history_development import OnlineRGBHistory
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.rgb_exit_candidates_development import observe_exit_candidates
from lewm.observed_traversal_controller_development import ObservedTraversalController
from lewm.fixed_forward_traversal_development import FixedForwardTraversal
from lewm.continuation_branch_development import observed_branch, choose_side_branch, RelativeBearingAlignment

METHODS = ('fixed_forward', 'direct_direct', 'supervised_rollout', 'jepa_rollout')


class ObservedContinuation:
    def __init__(self, method, geometry, template=None):
        if method not in METHODS: raise ValueError('fixed continuation method required')
        self.method, self.geometry, self.template = method, geometry, template
        self.orientation = FastRelativeOrientation()
        self.history = OnlineRGBHistory()
        self.history.begin_episode((0, 0, 0))
        self.first = self._traversal()
        self.second = self.scan = self.ground = self.alignment = None
        self.stage = 'FIRST'
        self.status = 'RUNNING'
        self.start_ns = self.hold_since = None
        self.incoming = self.selected = None
        self.observations = []
        self.tick = -1

    def _traversal(self):
        return FixedForwardTraversal(self.geometry) if self.method == 'fixed_forward' else ObservedTraversalController(self.method, self.geometry, self.template)

    def ledgers(self):
        return [{'leg_index': i, 'record': child.ledger.snapshot()}
                for i, child in enumerate((self.first, self.second)) if child is not None]

    def finish_physical_stop(self):
        for child in (self.first, self.second):
            if child is not None and child.ledger.record is not None and child.ledger.record['status'] == 'PENDING':
                child.ledger.finish(status='PHYSICAL_STOP')
        self.status = 'PHYSICAL_STOP'

    def observe(self, packet, fast_packet, *, now_ns):
        if self.status != 'RUNNING': raise SensorContractError('active continuation required')
        try:
            return self._observe(packet, fast_packet, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.status = 'FAILED_SENSOR'
            for child in (self.first, self.second):
                if child is not None and child.ledger.record is not None and child.ledger.record['status'] == 'PENDING':
                    child.ledger.finish(status='FAILED_SENSOR')
            raise SensorContractError('observed continuation sensor/control failure') from error

    def _observe(self, packet, fast_packet, *, now_ns):
        self.tick += 1
        self.history.push(packet, now_ns=now_ns)
        attitude = (self.orientation.begin(packet, fast_packet, now_ns=now_ns) if self.tick == 0
                    else self.orientation.step(packet, fast_packet, now_ns=now_ns))
        if self.start_ns is None: self.start_ns = now_ns
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        stage = self.stage
        child_result = scan_result = turn_result = proposal_rows = selected_now = None
        command = [0., 0., 0.]
        if now_ns-self.start_ns >= 80_000_000_000:
            self.status = 'FAILED_GLOBAL_TIMEOUT'
            for child in (self.first, self.second):
                if child is not None and child.ledger.record is not None and child.ledger.record['status'] == 'PENDING':
                    child.ledger.finish(status='FAILED_TIMEOUT')
        elif stage in ('FIRST', 'SECOND'):
            child = self.first if stage == 'FIRST' else self.second
            child_result = child.observe(packet, fast_packet, now_ns=now_ns)
            command = child_result['requested_command']
            candidate = child_result['selected_exit_candidate']
            if candidate is not None:
                bearing = candidate['bearing_body_rad']
                direction = rotation@np.array([np.cos(bearing), np.sin(bearing), 0.])
                if stage == 'FIRST': self.incoming = direction
                else:
                    # A transported scan ray is only an alignment suggestion.
                    # The unchanged child must independently propose an exit
                    # from its fourth fresh, post-alignment RGB observation.
                    target = np.asarray(self.selected['direction_initial_body'])
                    agreement = float(direction@target/(np.linalg.norm(direction)*np.linalg.norm(target)))
                    if agreement < np.cos(.35):
                        self.status = 'FAILED_BRANCH_REOBSERVATION'
                        child.ledger.finish(status='FAILED_NO_VISUAL_CHANGE')
                        command = [0., 0., 0.]
            if self.status == 'RUNNING' and child_result['terminal']:
                if child_result['status'] != 'ARRIVAL_CANDIDATE':
                    self.status = 'FAILED_'+stage+'_'+child_result['status'].removeprefix('FAILED_')
                elif stage == 'FIRST':
                    self.stage, self.hold_since = 'HOLD_SCAN', now_ns
                else: self.status = 'COMPLETE_PROVISIONAL'
        elif stage in ('HOLD_SCAN', 'HOLD_ALIGN', 'HOLD_SECOND'):
            if now_ns-self.hold_since >= 1_500_000_000:
                if stage == 'HOLD_SCAN':
                    self.scan = FastGyroScan()
                    self.ground = CausalGravityFeedbackGround('transported_feedback')
                    self.stage = 'SCAN'
                elif stage == 'HOLD_ALIGN':
                    self.alignment = RelativeBearingAlignment(self.selected['direction_initial_body'])
                    self.stage = 'ALIGN'
                else:
                    self.second = self._traversal()
                    self.stage = 'SECOND'
        elif stage == 'SCAN':
            initial = self.scan.status == 'NEW'
            scan_result = (self.scan.begin(packet, fast_packet, now_ns=now_ns) if initial
                           else self.scan.step(packet, fast_packet, now_ns=now_ns))
            ground = self.ground.begin(packet, now_ns=now_ns) if initial else self.ground.step(packet, now_ns=now_ns)
            command = scan_result['requested_command']
            if initial or scan_result['new_completed_view'] is not None:
                image_id = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()
                proposal_rows = observe_exit_candidates(packet, ground, now_ns=now_ns, observation_id=image_id)['candidate_rows']
                self.observations.extend(observed_branch(row, rotation, decision_ns=now_ns) for row in proposal_rows)
            if scan_result['status'] == 'COMPLETE':
                self.selected = choose_side_branch(self.observations, self.incoming, now_ns=now_ns)
                selected_now = copy.deepcopy(self.selected)
                if self.selected is None: self.status = 'FAILED_NO_SIDE_BRANCH'
                else: self.stage, self.hold_since = 'HOLD_ALIGN', now_ns
            elif scan_result['status'].startswith('FAILED_'):
                self.status = 'FAILED_SCAN_'+scan_result['status'].removeprefix('FAILED_')
        elif stage == 'ALIGN':
            turn_result = self.alignment.observe(packet, attitude, now_ns=now_ns)
            command = turn_result['requested_command']
            if turn_result['status'] == 'COMPLETE': self.stage, self.hold_since = 'HOLD_SECOND', now_ns
            elif turn_result['status'].startswith('FAILED_'):
                self.status = 'FAILED_ALIGNMENT_'+turn_result['status'].removeprefix('FAILED_')
        else: raise SensorContractError('unknown continuation stage')
        terminal = self.status != 'RUNNING'
        if terminal: command = [0., 0., 0.]
        return {'status': self.status, 'stage': stage, 'next_stage': self.stage,
                'decision_ns': now_ns, 'tick': self.tick, 'terminal': terminal,
                'requested_command': command, 'child': child_result, 'scan': scan_result,
                'turn': turn_result, 'selected_view_proposals': proposal_rows,
                'selected_side_branch': selected_now, 'global_orientation': attitude,
                'ledgers': self.ledgers(), 'trusted_graph_edges': 0,
                'scope': 'continuous two-traversal development only; no place identity, safe scan, beacon/return or hardware qualification'}

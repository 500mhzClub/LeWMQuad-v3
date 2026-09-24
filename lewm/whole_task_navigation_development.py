"""Continuous RGB marker search and provisional route-based return.

No maze geometry, cell association, beacon coordinate, true motion or goal image
is accepted. This is a hybrid integration controller, not a learned navigation
policy. Local arrivals and HOME_CANDIDATE require independent physical evaluation.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import observed_branch, wrap
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.fast_gyro_scan_development import FastGyroScan
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.memory.episodic_route_hypotheses_development import EpisodicRouteHypotheses, current_view
from lewm.online_rgb_history_development import OnlineRGBHistory
from lewm.persistent_alignment_continuation_development import PersistentBearingAlignment
from lewm.rgb_exit_candidates_development import observe_exit_candidates
from lewm.rgb_marker_beacon_development import MarkerDiscovery
from lewm.stop_observe_traversal_development import StopObserveTraversal

MEMORY_ARMS = ('episodic', 'local_only')
MAX_SECONDS = 360
MAX_LEGS = 36


def direction_error(a, b):
    return abs(wrap(math.atan2(a[1], a[0])-math.atan2(b[1], b[0])))


def branch_order(item, incoming):
    """Left, forward, right, then reverse relative to latest departure direction."""
    d = item['direction_initial_body']
    delta = wrap(math.atan2(d[1], d[0])-math.atan2(incoming[1], incoming[0]))
    if math.pi/4 <= delta <= 3*math.pi/4:
        group, ideal = 0, math.pi/2
    elif abs(delta) < math.pi/4:
        group, ideal = 1, 0.
    elif -3*math.pi/4 <= delta <= -math.pi/4:
        group, ideal = 2, -math.pi/2
    else:
        group, ideal = 3, math.pi
    return group, abs(wrap(delta-ideal)), -item['candidate']['support_points'], -item['observed_ns']


class WholeTaskNavigation:
    def __init__(self, method, geometry, template=None, *, memory_arm):
        if method not in ('fixed_forward', 'direct_direct', 'supervised_rollout', 'jepa_rollout') or memory_arm not in MEMORY_ARMS:
            raise ValueError('explicit supported local method and memory arm required')
        self.method, self.geometry, self.template, self.memory_arm = method, geometry, template, memory_arm
        self.orientation = FastRelativeOrientation()
        self.history = OnlineRGBHistory(); self.history.begin_episode((0, 0, 0))
        self.marker = MarkerDiscovery()
        self.memory = EpisodicRouteHypotheses() if memory_arm == 'episodic' else None
        self.stage, self.status, self.mission = 'INITIAL_OBSERVE', 'RUNNING', 'EXPLORE'
        self.tick = -1
        self.start_ns = self.last_ns = self.hold_since = None
        self.ground = CausalGravityFeedbackGround('transported_feedback')
        self.scan = self.alignment = self.child = None
        self.children = []  # Audit ledgers only, not local-only routing input.
        self.incoming = np.array([1., 0., 0.])
        self.selected = None
        self.scan_views = []  # Cleared at each new scan in both arms.
        self.home_view = None
        self.home_quiet_since = None
        self.home_last_ns = None
        self.first_return = True
        self.leg_mission = None
        self.completed_legs = 0
        self.memory_digest = None

    def _memory_record(self):
        if self.memory is None:
            return None
        snapshot = self.memory.snapshot()
        self.memory_digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                                                       allow_nan=False).encode()).hexdigest()
        return self.memory_digest

    def _fail(self, status, now_ns):
        self.status = status
        if self.child is not None and self.child.ledger.record is not None and self.child.ledger.record['status'] == 'PENDING':
            self.child.ledger.finish(status='PHYSICAL_STOP' if status == 'PHYSICAL_STOP' else 'FAILED_SENSOR'
                                     if status == 'FAILED_SENSOR' else 'FAILED_NO_VISUAL_CHANGE'
                                     if status == 'FAILED_FRESH_EXIT_AGREEMENT' else 'FAILED_TIMEOUT')
        if self.memory is not None and self.memory._phase not in ('UNSTARTED', 'UNCERTAIN_AFTER_FAILURE'):
            self.memory.abort(now_ns=now_ns, status='PHYSICAL_STOP' if status == 'PHYSICAL_STOP' else
                              'FAILED_SENSOR' if status == 'FAILED_SENSOR' else 'FAILED_EXECUTION')
            self._memory_record()

    def finish_physical_stop(self, *, now_ns):
        self._fail('PHYSICAL_STOP', now_ns)

    def ledgers(self):
        return [{'leg_index': i, 'record': child.ledger.snapshot()} for i, child in enumerate(self.children)]

    def memory_snapshot(self):
        return self.memory.snapshot() if self.memory is not None else None

    def _select(self, selected, now_ns):
        self.selected = deepcopy(selected)
        self.stage, self.hold_since = 'HOLD_ALIGN', now_ns

    def _target_return_direction(self):
        if self.mission != 'RETURN':
            return None
        if self.memory is not None:
            intent = self.memory.return_intent()
            return intent.get('direction_initial_body')
        return (-self.incoming).tolist() if self.first_return else None

    def _home_appearance(self, packet, attitude, now_ns):
        if self.home_last_ns is not None and now_ns-self.home_last_ns != 100_000_000:
            self.home_quiet_since = None
        self.home_last_ns = now_ns
        view = current_view(packet, attitude, now_ns=now_ns)
        score = float(np.mean(np.abs(np.array(view.descriptor)-self.home_view.descriptor)))
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        heading = abs(math.atan2(rotation[1, 0], rotation[0, 0]))
        gyro = packet['sensor_state']['sensed']['gyro']
        joints = packet['sensor_state']['sensed']['joints']
        quiet = bool(gyro['valid'][-5:].all() and joints['valid'][-5:, 12:].all()
                     and np.max(np.linalg.norm(gyro['values'][-5:], axis=1)) <= .15
                     and np.sqrt(np.mean(joints['values'][-5:, 12:]**2)) <= 1.)
        eligible = score <= .02 and heading <= .2 and quiet
        if eligible:
            if self.home_quiet_since is None: self.home_quiet_since = now_ns
        else:
            self.home_quiet_since = None
        return {'rgb_block_mean_l1': score, 'relative_heading_error_rad': heading, 'quiet': quiet,
                'provisional_match': self.home_quiet_since is not None and now_ns-self.home_quiet_since >= 300_000_000,
                'home_verified': False}

    def observe(self, packet, fast_packet, *, now_ns):
        if self.status != 'RUNNING':
            raise SensorContractError('active whole-task controller required')
        try:
            return self._observe(packet, fast_packet, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            # Use the last valid decision time if the supplied clock is invalid.
            stamp = self.last_ns if self.last_ns is not None else 0
            self._fail('FAILED_SENSOR', stamp)
            raise SensorContractError('whole-task sensor/control failure; apply explicit zero') from error

    def _observe(self, packet, fast_packet, *, now_ns):
        self.tick += 1
        self.history.push(packet, now_ns=now_ns)
        attitude = (self.orientation.begin(packet, fast_packet, now_ns=now_ns) if self.tick == 0
                    else self.orientation.step(packet, fast_packet, now_ns=now_ns))
        self.last_ns = now_ns
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        marker = self.marker.observe(packet, now_ns=now_ns)
        if self.start_ns is None:
            self.start_ns = now_ns
            self.home_view = current_view(packet, attitude, now_ns=now_ns)
            if self.memory is not None:
                self.memory.start(packet, attitude, now_ns=now_ns); self._memory_record()
        mission_changed = marker['distinct_marker_count'] > 0 and self.mission == 'EXPLORE'
        if marker['distinct_marker_count']:
            self.mission = 'RETURN'
        canceled_unstarted_leg = False
        if mission_changed and self.stage not in ('SCAN', 'HOLD_SCAN'):
            executing = self.stage == 'TRAVERSE' and self.child.tick >= 3
            if not executing:
                if self.completed_legs == 0:
                    self.status = 'HOME_CANDIDATE_INITIAL_MARKER'
                else:
                    self.stage, self.hold_since = 'HOLD_SCAN', now_ns
                    canceled_unstarted_leg = True
        stage = self.stage
        child_result = scan_result = turn_result = proposals = selected_now = home = None
        scan_stop = None
        command = [0., 0., 0.]
        if self.status != 'RUNNING':
            pass
        elif now_ns-self.start_ns >= MAX_SECONDS*1_000_000_000:
            self._fail('FAILED_GLOBAL_TIMEOUT', now_ns)
        elif stage == 'INITIAL_OBSERVE':
            ground = (self.ground.begin(packet, now_ns=now_ns) if self.tick == 0
                      else self.ground.step(packet, now_ns=now_ns))
            if self.tick == 3:
                image_id = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()
                proposals = observe_exit_candidates(packet, ground, now_ns=now_ns, observation_id=image_id)['candidate_rows']
                eligible = [p for p in proposals if abs(p['bearing_body_rad']) <= .35]
                if not eligible:
                    self._fail('FAILED_INITIAL_NO_EXIT', now_ns)
                else:
                    candidate = min(eligible, key=lambda p: (abs(p['bearing_body_rad']), -p['support_points']))
                    selected_now = observed_branch(candidate, rotation, decision_ns=now_ns)
                    self._select(selected_now, now_ns)
                    if self.memory is not None:
                        self.memory.remember_view(packet, attitude, now_ns=now_ns); self._memory_record()
        elif stage in ('HOLD_ALIGN', 'HOLD_TRAVERSE', 'HOLD_SCAN'):
            if now_ns-self.hold_since >= 1_500_000_000:
                if stage == 'HOLD_ALIGN':
                    self.alignment = PersistentBearingAlignment(self.selected['direction_initial_body'])
                    self.stage = 'ALIGN'
                elif stage == 'HOLD_TRAVERSE':
                    if len(self.children) >= MAX_LEGS:
                        self._fail('FAILED_LEG_BUDGET', now_ns)
                    else:
                        self.child = StopObserveTraversal(self.method, self.geometry, self.template)
                        self.leg_mission = self.mission
                        self.children.append(self.child); self.stage = 'TRAVERSE'
                else:
                    self.scan = FastGyroScan()
                    self.ground = CausalGravityFeedbackGround('transported_feedback')
                    self.scan_views = []; self.stage = 'SCAN'
        elif stage == 'ALIGN':
            turn_result = self.alignment.observe(packet, attitude, now_ns=now_ns)
            command = turn_result['requested_command']
            if turn_result['status'] == 'COMPLETE':
                self.stage, self.hold_since = 'HOLD_TRAVERSE', now_ns
            elif turn_result['status'].startswith('FAILED_'):
                self._fail('FAILED_ALIGNMENT', now_ns)
        elif stage == 'TRAVERSE':
            child_result = self.child.observe(packet, fast_packet, now_ns=now_ns)
            command = child_result['requested_command']
            candidate = child_result['selected_exit_candidate']
            if candidate is not None:
                direction = observed_branch(candidate, rotation, decision_ns=now_ns)['direction_initial_body']
                if direction_error(direction, self.selected['direction_initial_body']) > .35:
                    self._fail('FAILED_FRESH_EXIT_AGREEMENT', now_ns)
                else:
                    self.incoming = np.array(direction)
                    if self.memory is not None:
                        mode = 'RETURN' if self.leg_mission == 'RETURN' else 'OUTWARD'
                        self.memory.begin(packet, attitude, candidate, now_ns=now_ns, mode=mode)
                        self._memory_record()
            if self.status == 'RUNNING' and child_result['terminal']:
                if child_result['status'] != 'ARRIVAL_CANDIDATE':
                    self._fail('FAILED_LOCAL_'+child_result['status'], now_ns)
                else:
                    if self.memory is not None:
                        self.memory.finish(packet, attitude, now_ns=now_ns, status='ARRIVAL_CANDIDATE')
                        self._memory_record()
                    self.completed_legs += 1
                    if self.leg_mission == 'RETURN':
                        self.first_return = False
                    if self.memory is not None and self.memory.return_intent()['kind'] == 'HOME_CANDIDATE':
                        self.status = 'HOME_CANDIDATE_ROUTE_HYPOTHESIS'
                    else:
                        self.stage, self.hold_since = 'HOLD_SCAN', now_ns
        elif stage == 'SCAN':
            initial = self.scan.status == 'NEW'
            scan_result = (self.scan.begin(packet, fast_packet, now_ns=now_ns) if initial
                           else self.scan.step(packet, fast_packet, now_ns=now_ns))
            ground = self.ground.begin(packet, now_ns=now_ns) if initial else self.ground.step(packet, now_ns=now_ns)
            command = scan_result['requested_command']
            acquired = initial or scan_result['new_completed_view'] is not None
            target = self._target_return_direction()
            if acquired:
                image_id = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()
                proposals = observe_exit_candidates(packet, ground, now_ns=now_ns, observation_id=image_id)['candidate_rows']
                current = [observed_branch(p, rotation, decision_ns=now_ns) for p in proposals]
                self.scan_views.extend(current)
                if self.memory is not None:
                    self.memory.remember_view(packet, attitude, now_ns=now_ns)
                if target is not None:
                    matching = [v for v in current if direction_error(v['direction_initial_body'], target) <= .35]
                    selected_now = min(matching, key=lambda v: direction_error(v['direction_initial_body'], target)) if matching else None
                else:
                    left = [v for v in current if branch_order(v, self.incoming)[0] == 0]
                    selected_now = min(left, key=lambda v: branch_order(v, self.incoming)) if left else None
                if selected_now is not None:
                    self._select(selected_now, now_ns); command = [0., 0., 0.]
                    scan_stop = {'reason': 'FRESH_RETURN_BEARING' if target is not None else 'FRESH_LEFT_BRANCH',
                                 'observed_ns': now_ns, 'full_circle_complete': scan_result['status'] == 'COMPLETE'}
                self._memory_record()
            if self.stage == 'SCAN' and scan_result['status'] == 'COMPLETE':
                eligible = [v for v in self.scan_views if 0 <= now_ns-v['observed_ns'] <= 30_000_000_000]
                if target is not None:
                    eligible = [v for v in eligible if direction_error(v['direction_initial_body'], target) <= .35]
                if not eligible:
                    self._fail('FAILED_NO_OBSERVED_RETURN' if target is not None else 'FAILED_NO_OBSERVED_BRANCH', now_ns)
                else:
                    selected_now = min(eligible, key=(lambda v: direction_error(v['direction_initial_body'], target))
                                       if target is not None else lambda v: branch_order(v, self.incoming))
                    self._select(selected_now, now_ns)
            elif scan_result['status'].startswith('FAILED_'):
                self._fail('FAILED_SCAN', now_ns)
        else:
            raise SensorContractError('unknown whole-task stage')
        if self.mission == 'RETURN' and stage != 'TRAVERSE' and self.status == 'RUNNING':
            home = self._home_appearance(packet, attitude, now_ns)
            if home['provisional_match']:
                self.status = 'HOME_CANDIDATE_RGB_REFERENCE'
        terminal = self.status != 'RUNNING'
        if terminal: command = [0., 0., 0.]
        return {'status': self.status, 'stage': stage, 'next_stage': self.stage, 'mission': self.mission,
                'decision_ns': now_ns, 'tick': self.tick, 'terminal': terminal, 'requested_command': command,
                'leg_index': len(self.children)-1, 'child': child_result, 'scan': scan_result, 'turn': turn_result,
                'selected_view_proposals': proposals, 'selected_branch': deepcopy(selected_now), 'scan_stop_evidence': scan_stop,
                'global_orientation': attitude, 'marker': marker, 'home_appearance': home,
                'memory_arm': self.memory_arm, 'memory_digest': self.memory_digest,
                'mission_changed': mission_changed, 'canceled_unstarted_leg': canceled_unstarted_leg,
                'completed_legs': self.completed_legs,
                'trusted_graph_edges': 0, 'home_verified': False,
                'scope': 'continuous development search and tentative return; all physical outcomes evaluated separately'}

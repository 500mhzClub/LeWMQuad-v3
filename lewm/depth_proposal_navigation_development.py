"""Measured-depth proposals throughout fused discovery, traversal and scanning.

The two copied decision methods differ ONLY in their proposal-provider calls;
AST equivalence tests enforce this. No global patching or predecessor edits.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import observed_branch
from lewm.depth_floor_hold_navigation_development import ObservedDepthFloor
from lewm.depth_supported_exit_candidates_development import observe_depth_exit_candidates
from lewm.fast_gyro_scan_development import FastGyroScan
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.memory.episodic_route_hypotheses_development import current_view
from lewm.persistent_alignment_continuation_development import PersistentBearingAlignment
from lewm.rgbd_fused_navigation_development import RGBDFusedNavigation, RGBDFusedTraversal, fused_forward_speed
from lewm.stop_observe_traversal_development import StopObserveTraversal
from lewm.whole_task_navigation_development import MAX_SECONDS, MAX_LEGS, direction_error, branch_order


class DepthProposalTraversal(RGBDFusedTraversal):
    def __init__(self, geometry, observations, navigator):
        super().__init__(geometry, observations)
        self.navigator = navigator

    def _fused_step(self, packet, *, now_ns):
        # Frozen measured-region rules copied with one explicit speed-source change.
        self.tick += 1; self.clock = now_ns
        ground = self.ground.begin(packet, now_ns=now_ns) if self.tick == 0 else self.ground.step(packet, now_ns=now_ns)
        obs = self.observations; memory = obs.memory
        if memory.last_ns != now_ns: raise SensorContractError('current measured region state required')
        image_id = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()
        candidate = arrival = clearance = None
        command = [0.,0.,0.]; distance = None
        gyro = packet['sensor_state']['sensed']['gyro']; joints = packet['sensor_state']['sensed']['joints']
        quiet = bool(gyro['valid'][-5:].all() and joints['valid'][-5:,12:].all()
                     and np.max(np.linalg.norm(gyro['values'][-5:],axis=1)) <= .15
                     and np.sqrt(np.mean(joints['values'][-5:,12:]**2)) <= 1.)
        if self.tick < 3: self.status = 'WARMUP'
        elif self.tick == 3:
            proposals = self.navigator._propose(packet,ground,now_ns=now_ns,observation_id=image_id)['candidate_rows']
            eligible = [c for c in proposals if abs(c['bearing_body_rad']) <= .35]
            self.target = obs.target(packet,now_ns=now_ns)
            if not eligible or self.target is None: self.status = 'FAILED_NO_MEASURED_TARGET'
            else:
                candidate = min(eligible,key=lambda c:(abs(c['bearing_body_rad']),-c['support_points']))
                self.ledger.begin(observation_id=image_id,decision_ns=now_ns,candidate=candidate)
                self.start_ns=now_ns; self.status='TRAVERSING'
        if self.status in ('TRAVERSING','BRAKING'):
            delta = memory.rotation.T@(np.asarray(self.target['target_initial_body_m'])-memory.position)
            distance = float(delta[0])
            speed = fused_forward_speed(memory, now_ns=now_ns)
            if self.status == 'TRAVERSING' and distance <= .5*speed+.03:
                self.status='BRAKING'; self.brake_ns=now_ns
            if now_ns-self.start_ns >= 30_000_000_000: self.status='FAILED_TIMEOUT'
            if self.status == 'BRAKING':
                if quiet:
                    if self.quiet_since is None: self.quiet_since=now_ns
                else: self.quiet_since=None
                if now_ns-self.brake_ns >= 500_000_000 and self.quiet_since is not None and now_ns-self.quiet_since >= 300_000_000:
                    clearance=obs.clearance(packet,now_ns=now_ns)
                    if distance < -.15: self.status='FAILED_TARGET_OVERSHOOT'
                    elif clearance['all_samples_supported'] and abs(distance) <= .20:
                        self.status='ARRIVAL_CANDIDATE'
                        arrival={'schema':'sampled_nominal_turn_region_arrival.v1','observation_id':image_id,
                            'decision_ns':now_ns,'target':deepcopy(self.target),'remaining_forward_m':distance,
                            'sampled_volume':clearance,'quiet_body_proxy':quiet,'qualified_arrival':False,'place_identity':None}
                    elif distance > .04:
                        # Bounded low-speed measured correction, not a larger
                        # fixed traversal distance or a relaxed clearance count.
                        self.status='TRAVERSING'; self.quiet_since=None
                    else: self.status='FAILED_UNOBSERVED_TURN_VOLUME'
                elif now_ns-self.brake_ns >= 2_000_000_000: self.status='FAILED_SETTLING'
            if self.status == 'TRAVERSING':
                bearing=math.atan2(delta[1],max(delta[0],.05))
                command=[.10 if distance < .25 else .20,0.,float(np.clip(1.5*bearing,-.25,.25))]
            if self.status not in ('TRAVERSING','BRAKING'):
                self.ledger.finish(status=self.status,arrival=arrival)
        terminal=self.status.startswith('FAILED_') or self.status=='ARRIVAL_CANDIDATE'
        return {'status':self.status,'decision_ns':now_ns,'tick':self.tick,'terminal':terminal,
            'requested_command':command,'selection':None,'selected_exit_candidate':candidate,
            'measured_target':deepcopy(self.target),'remaining_forward_m':distance,
            'quiet_body_proxy':quiet,'sampled_turn_volume':clearance,'ledger':self.ledger.snapshot(),
            'scope':'observed nominal observation region; no future-gait, trusted-place or hardware certificate'}


class DepthProposalNavigation(RGBDFusedNavigation):
    def __init__(self, geometry, *, memory_arm, prior, hypotheses):
        super().__init__(geometry, memory_arm=memory_arm, prior=prior, hypotheses=hypotheses)
        self._proposal_depth = None
        self.proposal_evidence = []

    def _propose(self, packet, ground_state, *, now_ns, observation_id):
        if (self._context is None or packet is not self._context[0]
                or self._proposal_depth is None or self.sensor_memory.last_ns != now_ns):
            raise SensorContractError('same synchronous depth proposal context required')
        result = observe_depth_exit_candidates(packet, self._proposal_depth, ground_state,
                                              now_ns=now_ns, observation_id=observation_id)
        self.proposal_evidence.append({k: deepcopy(result[k]) for k in (
            'decision_ns', 'candidate_rows', 'proposal_source', 'color_segmentation_used',
            'measured_floor_points', 'depth_sha256', 'body_ground_estimator', 'scope')})
        return result

    def observe_rgbd(self, packet, fast_packet, depth, *, now_ns):
        if self._proposal_depth is not None:
            raise SensorContractError('non-reentrant sensor consumer required')
        self._proposal_depth = depth
        self.proposal_evidence = []
        try:
            row = super().observe_rgbd(packet, fast_packet, depth, now_ns=now_ns)
            if self.stage == 'TRAVERSE' and not isinstance(self.child, DepthProposalTraversal):
                if self.child is None or self.child.tick != -1:
                    raise SensorContractError('only replace an unstarted depth-proposal traversal')
                self.child = DepthProposalTraversal(self.geometry, self.regions, self)
                self.child.ground = ObservedDepthFloor(self.regions)
                self.children[-1] = self.child
            row.update(local_controller='depth_proposal_fused_navigation_development_v1',
                       exit_proposal_evidence=deepcopy(self.proposal_evidence))
            return row
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.sensor_memory.failed = self.sensor_memory.rays.failed = True
            self._fail('FAILED_SENSOR', self.last_ns if self.last_ns is not None else 0)
            raise SensorContractError('depth-proposal navigation failed; apply explicit zero') from error
        finally:
            self._proposal_depth = None

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
                proposals = self._propose(packet, ground, now_ns=now_ns, observation_id=image_id)['candidate_rows']
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
                proposals = self._propose(packet, ground, now_ns=now_ns, observation_id=image_id)['candidate_rows']
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

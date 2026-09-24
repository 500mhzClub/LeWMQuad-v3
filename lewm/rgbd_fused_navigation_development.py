"""Sensor-owned fusion for the full discovery/return development controller.

No learned navigation or calibrated safety claim. Frozen predecessors remain
unchanged. Raw depth evidence is never replaced by a fused-motion lookalike.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_floor_hold_navigation_development import DepthFloorHoldNavigation, ObservedDepthFloor
from lewm.measured_line_integral_navigation_development import approach_line_command
from lewm.observable_approach_development import ObservableApproachRegions, ObservableApproachTraversal
from lewm.rgb_exit_candidates_development import observe_exit_candidates
from lewm.rgbd_inertial_ray_memory_development import RGBDInertialRayMemory


class PreparedRayView:
    """Synchronous read-through facade; the owner already integrated this frame."""
    def __init__(self, owner):
        self.owner = owner
        self.pending = None

    def stage(self, policy, depth, row):
        if self.pending is not None or self.owner.failed:
            raise SensorContractError('unconsumed or failed ray view')
        self.pending = (policy, depth, row)

    def observe(self, policy, depth, relative, *, now_ns):
        pending, self.pending = self.pending, None
        if (pending is None or policy is not pending[0] or depth is not pending[1]
                or relative is not pending[2]['depth_state']
                or now_ns != self.owner.last_ns or self.owner.failed):
            raise SensorContractError('exact current single-use region observation required')
        return deepcopy(pending[2]['ray_memory'])

    def query(self, points, roles, *, now_ns, backend='compiled'):
        return self.owner.query(points, roles, now_ns=now_ns, backend=backend)

    def __getattr__(self, name):
        if name not in ('position', 'rotation', 'up_initial', 'last_ns', 'latest_surface',
                        'frames', 'latest_frame', 'fusion', 'identity'):
            raise AttributeError(name)
        if self.owner.failed:
            raise SensorContractError('faulted fusion has no usable region state')
        return getattr(self.owner.rays, name)


class SharedAttitudeView:
    """Read the common gyro owner, optionally expressed in a scan-start frame.

    Re-anchoring an operator's attitude does not reset the global pose, gyro
    history, ray memory, prior sensitivity or uncertainty.
    """
    def __init__(self, navigator, *, local=False):
        self.navigator = navigator
        self.local = local
        self.last_ns = self.start_ns = self.anchor = self.samples_at_anchor = None

    def _read(self, policy, fast, now_ns, initial):
        context = self.navigator._context
        if (context is None or policy is not context[0] or fast is not context[1]
                or now_ns != context[2]['measured_ns'] or self.navigator.sensor_memory.failed
                or (initial and self.start_ns is not None)
                or (not initial and (self.last_ns is None or now_ns-self.last_ns != 100_000_000))):
            raise SensorContractError('current uninterrupted shared attitude required')
        attitude = deepcopy(context[2]['depth_state']['relative_orientation'])
        rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
        if initial:
            self.start_ns = now_ns
            self.anchor = rotation.copy()
            self.samples_at_anchor = attitude['samples_integrated']
        if self.local:
            attitude.update(rotation_initial_body_from_current_body=(self.anchor.T@rotation).tolist(),
                            start_ns=self.start_ns,
                            samples_integrated=attitude['samples_integrated']-self.samples_at_anchor)
        self.last_ns = now_ns
        return attitude

    def begin(self, policy, fast, *, now_ns):
        return self._read(policy, fast, now_ns, True)

    def step(self, policy, fast, *, now_ns):
        return self._read(policy, fast, now_ns, False)


def fused_forward_speed(memory, *, now_ns):
    """Conditional endpoint velocity in CURRENT body axes, not raw depth speed."""
    fusion = memory.fusion
    if (memory.last_ns != now_ns or fusion['measured_ns'] != now_ns
            or not fusion['usable_under_declared_proxy_budget']):
        raise SensorContractError('current admitted fusion required for approach speed')
    velocity = np.asarray(fusion['velocity_initial_body_m_s'], dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise SensorContractError('finite fused velocity required')
    return max(0., float((memory.rotation.T@velocity)[0]))


class RGBDFusedTraversal(ObservableApproachTraversal):
    """Same local rules, with explicitly fused braking velocity and line state."""

    def _observe(self, packet, fast_packet, *, now_ns):
        update = None
        if self.target is not None and self.status in ('TRAVERSING', 'BRAKING'):
            obs = self.observations
            memory = obs.memory
            limit = obs.limit(packet, now_ns=now_ns)
            value = limit['maximum_approach_m']
            remaining = float((memory.rotation.T@(np.asarray(self.target['target_initial_body_m'])-memory.position))[0])
            if value is not None and value < remaining-.01:
                if value < 0.:
                    raise SensorContractError('observed blocker inside required stopping/view region')
                self.target['target_initial_body_m'] = (memory.position+memory.rotation@np.array([value, 0., 0.])).tolist()
                self.target['kind'] = 'OBSERVED_BLOCKER_LIMITED_REGION'
                self.target['approach_limit'] = limit
                update = dict(decision_ns=now_ns, previous_remaining_m=remaining, new_remaining_m=value, limit=limit)
        result = self._fused_step(packet, now_ns=now_ns)
        memory = self.observations.memory
        if self.target is not None and self.approach_origin is None:
            self.approach_origin = memory.position.copy()
            self.approach_direction = memory.rotation[:, 0].copy()
        if result['status'] == 'TRAVERSING':
            guidance = approach_line_command(memory.position, memory.rotation, self.approach_origin,
                                            self.approach_direction, memory.up_initial)
            result['requested_command'][2] = guidance['yaw_command_rad_s']
            result['line_guidance'] = guidance
        result.update(approach_constraint_update=update,
                      local_controller='rgbd_fused_traversal_development_v1',
                      braking_velocity_source='CONDITIONAL_RGBD_INERTIAL_ENDPOINT_CURRENT_BODY',
                      braking_model_validated=False)
        return result


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
            proposals = observe_exit_candidates(packet,ground,now_ns=now_ns,observation_id=image_id)['candidate_rows']
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


class RGBDFusedNavigation(DepthFloorHoldNavigation):
    """Own raw sensor processing throughout one uninterrupted full mission."""
    def __init__(self, geometry, *, memory_arm, prior, hypotheses):
        super().__init__('depth_floor_hold', geometry, memory_arm=memory_arm)
        self.sensor_memory = RGBDInertialRayMemory(prior=prior, hypotheses=hypotheses)
        self.regions = ObservableApproachRegions(geometry)
        self.regions.memory = PreparedRayView(self.sensor_memory)
        self.ground = ObservedDepthFloor(self.regions)
        self._context = None
        self.orientation = SharedAttitudeView(self)

    def observe_rgbd(self, packet, fast_packet, depth, *, now_ns):
        if self.status != 'RUNNING':
            raise SensorContractError('terminal fused navigation cannot restart')
        try:
            fused = self.sensor_memory.observe(packet, depth, fast_packet, now_ns=now_ns)
            self._context = (packet, fast_packet, fused)
            self.regions.memory.stage(packet, depth, fused)
            row = super().observe_rgbd(packet, fast_packet, depth, fused['depth_state'], now_ns=now_ns)
            if self.regions.memory.pending is not None:
                raise SensorContractError('region did not consume current fused observation')
            if self.stage == 'TRAVERSE' and not isinstance(self.child, RGBDFusedTraversal):
                if self.child is None or self.child.tick != -1:
                    raise SensorContractError('only replace an unstarted fused traversal')
                self.child = RGBDFusedTraversal(self.geometry, self.regions)
                self.child.ground = ObservedDepthFloor(self.regions)
                self.children[-1] = self.child
            if self.stage == 'SCAN' and self.scan.status == 'NEW':
                self.scan.orientation = SharedAttitudeView(self, local=True)
            row.update(local_controller='rgbd_fused_navigation_development_v1',
                       sensor_fusion=deepcopy(fused['fusion']),
                       raw_depth_motion=deepcopy(fused['depth_state']['motion']),
                       point_motion=deepcopy(fused['point_state']['motion']),
                       learned_navigation_policy=False, navigation_qualified=False)
            return row
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.sensor_memory.failed = self.sensor_memory.rays.failed = True
            self.regions.memory.pending = None
            self._fail('FAILED_SENSOR', self.last_ns if self.last_ns is not None else 0)
            raise SensorContractError('fused navigation failed; apply explicit zero') from error
        finally:
            self._context = None

    def observe_stopping_tail(self, packet, fast_packet, depth, *, now_ns):
        """Ingest actual post-terminal sensors without restarting mission logic.

        A faulted estimator stays faulted. The executor must independently
        record the physical stopping tail even when this method refuses it.
        """
        if self.status == 'RUNNING':
            raise SensorContractError('stopping-tail ingestion requires terminal mission')
        row = self.sensor_memory.observe(packet, depth, fast_packet, now_ns=now_ns)
        return dict(measured_ns=now_ns, requested_command=[0., 0., 0.],
                    mission_status=self.status, sensor_fusion=deepcopy(row['fusion']),
                    controller_restarted=False, physical_rest_verified=False)

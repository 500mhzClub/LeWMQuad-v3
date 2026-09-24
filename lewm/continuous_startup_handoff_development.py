"""One observer/memory across startup, measured zero tail and navigation handoff.

READY_FOR_NAVIGATION_CONSUMER means state is available, not that an action is
clear or permitted. The eventual navigation consumer must separately establish
its action-dependent motion, clearance and ground-support conditions.
"""
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.startup_observation_turn_development import (
    StartupObservationTurn, MAX_BASE_SPEED_M_S, COMMAND_PLUS_STOP_HORIZON_S, PADDING_M)


class ContinuousStartupHandoff:
    """Consume each 100-ms RGBD/body/fast-gyro observation exactly once.

Startup retains the frozen controller. After its successful terminal the SAME
memory continues observing the stopping tail; the terminal controller is never
called again. No evaluator poses, contacts or scene geometry enter this API.
"""
    def __init__(self, geometry, *, velocity_prior, region_prior, admission):
        self._startup = StartupObservationTurn(geometry, velocity_prior=velocity_prior,
            region_prior=region_prior, admission=admission)
        self._relative = DepthRelativeState()
        self._memory = self._startup.memory
        self._region = region_prior
        self._last_ns = None; self._count = 0; self._tail_frames = 0
        self._tail_start_ns = None; self._last_request = None
        self._relative_row = None; self._evidence = None
        self.status = 'STARTUP'; self._last = None

    def _fail(self, now, error):
        self.status = 'FAILED_HANDOFF_SENSOR_OR_CONDITION'
        reasons = []
        while error is not None:
            reasons.append(str(error)); error = error.__cause__
        self._last = dict(status=self.status, measured_ns=now, terminal=True,
            requested_command=[0., 0., 0.], handoff_ready=False, failure_chain=reasons,
            navigation_qualified=False, contact_permitted=False)
        return deepcopy(self._last)

    def observe(self, policy, depth, fast, *, now_ns):
        if self.status.startswith('FAILED_'):
            raise SensorContractError('continuous handoff fault latched; no restart or stale snapshot')
        try:
            expected = self._region.anchor_ns if self._last_ns is None else self._last_ns+100_000_000
            if type(now_ns) is not int or now_ns != expected:
                raise SensorContractError('exact initial epoch and consecutive 100-ms observations required')
            relative = self._relative.observe(policy, depth, fast, now_ns=now_ns)
            # The recorded command is actuator acknowledgement, not motion evidence.
            if self._last_request is not None and self.status != 'READY_FOR_NAVIGATION_CONSUMER':
                applied = policy['sensor_state']['control']['applied_command']
                if (applied['measured_ns'][-1] != now_ns or not applied['valid'][-1].all()
                        or not np.allclose(applied['values'][-1], self._last_request, atol=1e-7, rtol=0)):
                    raise SensorContractError('startup/tail command acknowledgement differs from requested tick')
            startup_decision = None
            if self.status == 'STARTUP':
                startup_decision = self._startup.observe(policy, depth, relative, now_ns=now_ns)
                if startup_decision['status'].startswith('FAILED_'):
                    raise SensorContractError('startup terminal failure: '+str(startup_decision))
                requested = startup_decision['requested_command']
                if startup_decision['status'] == 'COMPLETE_OBSERVATION_TURN':
                    self.status = 'MEASURED_ZERO_TAIL'; self._tail_start_ns = now_ns
            else:
                self._memory.observe(policy, depth, relative, now_ns=now_ns)
                self._evidence = self._memory.query_current_primitives(now_ns=now_ns)
                if np.any(self._evidence['non_floor_conflict'] | self._evidence['floor_penetration']):
                    raise SensorContractError('observed primitive conflict during handoff/continuation')
                requested = None
                if self.status == 'MEASURED_ZERO_TAIL':
                    requested = [0., 0., 0.]
                    fusion = self._memory._rays.fusion
                    centre = np.asarray(fusion['position_initial_body_m'])
                    extent = (self._startup.radius+PADDING_M
                        +MAX_BASE_SPEED_M_S*COMMAND_PLUS_STOP_HORIZON_S+fusion['position_error_scale_m'])
                    through = now_ns+int(COMMAND_PLUS_STOP_HORIZON_S*1e9)
                    check = self._region.query([centre-extent], [centre+extent], [0.],
                        identity=self._memory._rays.identity, now_ns=through, observed_conflict=[False])
                    speed = float(np.linalg.norm(fusion['velocity_initial_body_m_s']))
                    if (not check['conditional_setup_non_floor_clearance'][0]
                            or speed+fusion['initial_velocity_prior_transport']['velocity_radius_m_s'] > MAX_BASE_SPEED_M_S):
                        raise SensorContractError('tail setup envelope/expiry or inferred speed condition failed')
                    self._tail_frames += 1
                    if self._tail_frames == 3:
                        # A completed duration is not stationary evidence. Check
                        # all 50 new fast-gyro samples in the final 100-ms window.
                        quiet = np.linalg.norm(np.asarray(fast['values'])[1:], axis=1).max() <= .1
                        if fusion['depth_rank'] != 3 or speed > .05 or not quiet:
                            raise SensorContractError('three-tick tail ended without quiet observable state')
                        self.status = 'READY_FOR_NAVIGATION_CONSUMER'
            self._last_ns = now_ns; self._count += 1; self._last_request = deepcopy(requested)
            self._relative_row = deepcopy(relative)
            fusion = self._memory._rays.fusion
            self._last = dict(status=self.status, measured_ns=now_ns, terminal=False,
                requested_command=deepcopy(requested), handoff_ready=self.status == 'READY_FOR_NAVIGATION_CONSUMER',
                startup_decision=deepcopy(startup_decision), consumed_observations=self._count,
                stopping_tail_observations=self._tail_frames, tail_start_ns=self._tail_start_ns,
                depth_rank=fusion['depth_rank'], combined_position_scale_m=fusion['position_error_scale_m'],
                navigation_qualified=False, contact_permitted=False, real_time_qualified=False)
            return deepcopy(self._last)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            return self._fail(now_ns, error)

    def relative_observation(self, *, now_ns):
        if self.status.startswith('FAILED_') or self._last_ns != now_ns or self._relative_row is None:
            raise SensorContractError('fresh nonfaulted relative observation required')
        return deepcopy(self._relative_row)

    def navigation_snapshot(self, *, now_ns):
        if self.status != 'READY_FOR_NAVIGATION_CONSUMER' or self._last_ns != now_ns:
            raise SensorContractError('fresh completed continuous handoff required')
        rays = self._memory._rays
        if rays.failed or self._memory.failed or self._relative.failed or rays.last_ns != now_ns:
            raise SensorContractError('same live nonfaulted observer and memory required')
        evidence = self._evidence
        ids = np.asarray(evidence['shape_ids'])
        fields = ('conditional_clearance', 'foot_contact_candidate', 'non_floor_conflict', 'floor_penetration')
        return deepcopy(dict(schema='continuous_startup_navigation_state_development.v1',
            measured_ns=now_ns, identity=rays.identity, initial_epoch_ns=self._region.anchor_ns,
            consumed_observations=self._count, gyro_intervals=self._relative.orientation.samples_integrated,
            fusion=rays.fusion, relative_observation=self._relative_row,
            retained_view_ns=[v['measured_ns'] for v in rays.frames], latest_view_ns=rays.latest_frame['measured_ns'],
            supplied_initial_nonfloor_region=asdict(self._region),
            supplied_region_active=self._region.anchor_ns <= now_ns <= self._region.valid_until_ns,
            observed_current_posture={k: ids[evidence[k]].tolist() for k in fields},
            evidence_roles=['SUPPLIED_INITIAL_NONFLOOR_REGION', 'SENSOR_OBSERVED_CURRENT_POSTURE'],
            current_posture_is_future_sweep=False, ground_support_permission=False,
            navigation_action_permitted=False, navigation_qualified=False, real_time_qualified=False))

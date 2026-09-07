"""Live packet-to-motion-to-visit handoff, not a place or clearance detector.

The owner must supply one continuously acquired VisualLedMotion stream and
FastRelativeOrientation stream. A branch bearing is not a measured endpoint:
the .4m lookahead is explicitly a local goal, not a corridor exit or place.
"""
from copy import deepcopy
import math
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import observed_branch
from lewm.continuous_pulse_execution_development import ContinuousPulseExecution
from lewm.memory.episodic_route_hypotheses_development import EpisodicRouteHypotheses, current_view
from lewm.sensor_anchored_goal_development import current_pose


class PulseRouteBridge:
    def __init__(self, *, identity=(0, 0, 0), **budgets):
        self.executor = ContinuousPulseExecution(identity=identity, **budgets)
        self.memory = EpisodicRouteHypotheses()
        self.packet = self.attitude = self.view = None
        self.active_kind = None
        self.started = False
        self.fault = None

    def _fail(self, reason, now_ns):
        if self.fault is None:
            self.fault = str(reason)
            self.executor.fail(self.fault, now_ns=now_ns)
            if self.started:
                self.memory.abort(now_ns=max(now_ns, self.view.timestamp_ns), status='FAILED_SENSOR')

    def observe(self, packet, attitude, evidence, *, now_ns):
        if self.fault is not None:
            return self._result(now_ns, None)
        result = None
        try:
            view = current_view(packet, attitude, now_ns=now_ns)
            _, R, pose = current_pose(evidence, identity=self.executor.identity, now_ns=now_ns)
            if (view.rgb_sha256 != pose['rgb_sha256'] or view.episode != self.executor.identity
                    or not np.allclose(R, np.asarray(view.rotation).reshape(3, 3), rtol=0, atol=1e-8)
                    or (self.view is not None and view.orientation_reference_ns != self.view.orientation_reference_ns)):
                raise SensorContractError('memory and motion must share acquired RGB, episode and uninterrupted orientation')
            result = self.executor.observe(evidence, now_ns=now_ns)
            if result['status'] == 'FAILED':
                raise SensorContractError(result['reason'])
            if view.orientation_reference_ns != self.executor.start_ns:
                raise SensorContractError('gyro reference must equal initial visual episode time')
            self.packet, self.attitude, self.view = deepcopy(packet), deepcopy(attitude), view
            if not self.started:
                self.memory.start(packet, attitude, now_ns=now_ns)
                self.started = True
            local = result['local_decision']
            if local is not None and local['terminal'] == 'VISUAL_TARGET_SEQUENCE_COMPLETE':
                if self.active_kind == 'branch':
                    self.memory.finish(packet, attitude, now_ns=now_ns, status='ARRIVAL_CANDIDATE')
                elif self.active_kind == 'turn':
                    self.memory.remember_view(packet, attitude, now_ns=now_ns)
                else:
                    raise SensorContractError('local completion without matching bridge attempt')
                self.active_kind = None
        except (ValueError, TypeError, KeyError, IndexError) as error:
            stamp=self.executor.last_ns or 0
            if type(now_ns) is int and now_ns>=stamp:stamp=now_ns
            self._fail('PULSE_ROUTE_INPUT_OR_EXECUTION_REJECTED: ' + str(error), stamp)
        return self._result(now_ns, result)

    def _ready(self, now_ns):
        if (self.fault is not None or not self.started or self.active_kind is not None
                or self.executor.active is not None or self.view.timestamp_ns != now_ns):
            raise SensorContractError('idle fault-free current bridge observation required')

    def begin_branch(self, candidate, *, now_ns, mode='OUTWARD', distance_m=.4):
        try:
            self._ready(now_ns)
            if isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)) or not 0 < distance_m <= .4:
                raise SensorContractError('positive bounded local lookahead required')
            branch = observed_branch(candidate, np.asarray(self.view.rotation).reshape(3, 3), decision_ns=now_ns)
            # Validate memory dispatch transactionally, including fresh RGB hash
            # and return-intent agreement, before mutating the live executor.
            candidate_memory = deepcopy(self.memory)
            attempt = candidate_memory.begin(self.packet, self.attitude, candidate, now_ns=now_ns, mode=mode)
            angle = branch['candidate']['bearing_body_rad']
            leg = self.executor.begin([distance_m*math.cos(angle), distance_m*math.sin(angle)], angle, now_ns=now_ns)
            self.memory = candidate_memory
            self.active_kind = 'branch'
            return dict(attempt_id=attempt, local_leg=leg, qualified_traversal=False,
                        metric_exit_position_observed=False, clearance_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self._fail('BRANCH_DISPATCH_REJECTED: '+str(error), self.executor.last_ns or 0)
            raise SensorContractError('branch dispatch rejected; bridge fault latched') from error

    def begin_turn(self, yaw_delta_rad, *, now_ns):
        try:
            self._ready(now_ns)
            leg = self.executor.begin([0., 0.], yaw_delta_rad, now_ns=now_ns)
            self.active_kind = 'turn'
            return leg
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self._fail('TURN_DISPATCH_REJECTED: '+str(error), self.executor.last_ns or 0)
            raise SensorContractError('turn dispatch rejected; bridge fault latched') from error

    def _result(self, now_ns, result):
        return dict(decision_ns=now_ns, fault=self.fault, execution=result,
                    requested_command=[0., 0., 0.] if self.fault is not None or result is None else result['requested_command'],
                    active_kind=self.active_kind, home_verified=False, mission_complete=False, navigation_qualified=False)

    def snapshot(self):
        return dict(executor=self.executor.snapshot(), memory=self.memory.snapshot(), fault=self.fault,
                    active_kind=self.active_kind, mission_complete=False, home_verified=False)

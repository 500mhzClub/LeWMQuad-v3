"""Multiple local motions, one visual frame and one fault/budget lifetime."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.sensor_anchored_goal_development import AnchoredGoal, current_pose
from lewm.anchored_pulse_servo_development import AnchoredPulseServo


class ContinuousPulseExecution:
    def __init__(self, *, identity=(0, 0, 0), maximum_ticks=3600, maximum_pulses=140, maximum_legs=36):
        if any(type(x) is not int or x <= 0 for x in (maximum_ticks, maximum_pulses, maximum_legs)):
            raise SensorContractError('positive fixed mission budgets required')
        self.identity = _identity(identity)
        self.maximum_ticks, self.maximum_pulses, self.maximum_legs = maximum_ticks, maximum_pulses, maximum_legs
        self.start_ns = self.last_ns = self.last_frame = self.evidence = None
        self.active = None
        self.fault = None
        self.pulses = 0
        self.legs = []

    def fail(self, reason, *, now_ns=None):
        if self.fault is None:
            self.fault = str(reason)
            if self.active is not None:
                stamp=self.last_ns if now_ns is None else now_ns
                self.legs[-1].update(status='FAILED_EXECUTION', finished_ns=stamp, reason=self.fault)

    def begin(self, displacement_body_xy, yaw_delta_rad, *, now_ns):
        if self.fault is not None:
            raise SensorContractError('mission fault latched; no next leg')
        try:
            if self.active is not None or self.last_ns != _ns(now_ns, 'dispatch clock') or self.evidence is None:
                raise SensorContractError('idle current observation required for dispatch')
            if len(self.legs) >= self.maximum_legs or self.pulses >= self.maximum_pulses:
                raise SensorContractError('mission leg/pulse budget exhausted')
            goal = AnchoredGoal.from_observation(self.evidence, displacement_body_xy, yaw_delta_rad,
                                                identity=self.identity, now_ns=now_ns)
            child = AnchoredPulseServo(goal)
            self.legs.append(dict(leg_index=len(self.legs), goal=goal.snapshot(), status='PENDING',
                                  started_ns=now_ns, finished_ns=None, place_identity=None))
            self.active = child
            return deepcopy(self.legs[-1])
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.fail('DISPATCH_REJECTED: ' + str(error))
            raise SensorContractError('dispatch rejected; mission fault latched') from error

    def observe(self, evidence, *, now_ns):
        now = self.last_ns if self.last_ns is not None else 0
        decision = None
        if self.fault is not None:
            return self._result(now, decision)
        try:
            now = _ns(now_ns, 'mission decision')
            p, R, pose = current_pose(evidence, identity=self.identity, now_ns=now)
            if self.last_ns is None:
                if pose['frame'] != 0 or not np.allclose(p, 0., rtol=0, atol=1e-12) or not np.allclose(R, np.eye(3), rtol=0, atol=1e-12):
                    raise SensorContractError('initial visual frame required; no midstream mission reset')
                self.start_ns = now
            elif now-self.last_ns != 100_000_000 or pose['frame'] != self.last_frame+1:
                raise SensorContractError('uninterrupted 10Hz visual frame sequence required')
            self.last_ns, self.last_frame = now, pose['frame']
            self.evidence = deepcopy(evidence)
            if (now-self.start_ns)//100_000_000 >= self.maximum_ticks:
                raise SensorContractError('mission time budget exhausted')
            if self.active is not None:
                before = self.active.pulses
                decision = self.active.step(evidence, now_ns=now)
                extra = self.active.pulses-before
                if self.pulses+extra > self.maximum_pulses:
                    raise SensorContractError('mission pulse budget exhausted before command dispatch')
                self.pulses += extra
                if decision['terminal'] == 'VISUAL_SERVO_FAILED':
                    self.fail(decision['reason'])
                elif decision['terminal'] == 'VISUAL_TARGET_SEQUENCE_COMPLETE':
                    self.legs[-1].update(status='LOCAL_GOAL_CANDIDATE', finished_ns=now,
                                         final_decision=deepcopy(decision))
                    self.active = None
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.fail('MISSION_INPUT_OR_CONTROL_REJECTED: ' + str(error), now_ns=max(now,self.last_ns or 0))
        return self._result(now, decision)

    def _result(self, now, decision):
        command = [0., 0., 0.] if self.fault is not None or decision is None else decision['requested_command']
        return dict(decision_ns=now, status='FAILED' if self.fault is not None else 'EXECUTING' if self.active is not None else 'IDLE',
                    reason=self.fault, requested_command=command, local_decision=deepcopy(decision),
                    mission_pulses=self.pulses, attempted_legs=len(self.legs),
                    completed_local_goals=sum(r['status']=='LOCAL_GOAL_CANDIDATE' for r in self.legs),
                    pose_reset=False, place_identity=None, mission_complete=False, navigation_qualified=False)

    def snapshot(self):
        return deepcopy(dict(identity=self.identity, start_ns=self.start_ns, last_ns=self.last_ns,
                             last_frame=self.last_frame, fault=self.fault, pulses=self.pulses, legs=self.legs,
                             mission_complete=False, navigation_qualified=False))

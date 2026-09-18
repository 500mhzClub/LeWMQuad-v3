"""Hold through rejected partial-floor measurements; resume on fresh valid poses."""
from copy import deepcopy
import numpy as np
from lewm.robust_height_floor_tracking_development import RobustHeightFloorRegistration
from lewm.stop_conditioned_settling_development import StopConditionedSettlingMission
from lewm.observed_round_trip_mission_development import WARMUP_TICKS

CONFLICT = 'current measured candidate conflicts with transported floor reference'
UNAVAILABLE = 'CURRENT_FLOOR_MEASUREMENT_REJECTED'


class ReacquiringFloorRegistration(RobustHeightFloorRegistration):
    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        try:
            return super().observe(policy,primary,auxiliary,raw,now_ns=now_ns)
        except ValueError as error:
            # Only this geometric residual rejection occurs after the original
            # packet, raw-pose, clock and partial-plane checks have succeeded.
            if str(error) != CONFLICT or self.anchor is None:
                raise
            frame = raw['current_pose']['frame']
            if frame != self.frame+1:
                raise
            # Consume this observation, without accepting or replacing a pose,
            # anchor, floor reference or threshold. The next frame is rechecked.
            self.frame = frame
            self.failed = False
            return dict(status=UNAVAILABLE,frame=frame,measured_ns=now_ns,
                current_pose=None,reason=CONFLICT,anchor_frame=self.anchor['current_pose']['frame'],
                rejected_measurement_used_for_mapping=False,
                rejected_measurement_used_for_arrival=False)


class ReacquiringFloorMission(StopConditionedSettlingMission):
    def unavailable(self, *, frame, now_ns, previous_requested_command):
        if self.terminal is not None:
            return deepcopy(self.last)
        command = np.asarray(previous_requested_command,float)
        if (type(frame) is not int or frame != self.frame+1
                or now_ns != 1_500_000_000+frame*100_000_000
                or command.shape != (3,) or not np.isfinite(command).all()):
            raise ValueError('consecutive missing observation and actual request required')
        self.frame = frame; self.now_ns = now_ns
        self.quiet = 0; self.was_within = False
        self.previous_visual_position = None
        self.previous_motion_quiet = False; self.previous_stopped_boundary = False
        if frame >= WARMUP_TICKS+self.navigation_ticks:
            self.terminal = 'MISSION_TICK_BUDGET_EXHAUSTED'
        self.last = dict(frame=frame,measured_ns=now_ns,phase=self.phase,phase_transition=None,
            active_goal_initial_body_xy_m=self.target().tolist(),observed_goal_distance_m=None,
            quiet_intervals=0,hold_required=True,arrival_confirmed_this_frame=False,
            arrivals=deepcopy(self.arrivals),terminal=self.terminal,failure=None,
            observation_status=UNAVAILABLE,observed_settling=None,
            native_state_used=False,verified_round_trip=False)
        return deepcopy(self.last)


class FloorReacquisitionRuntimeMixin:
    def __init__(self,*args,**kwargs):
        self.floor_waiting = False
        self.floor_valid_streak = 0
        self.floor_plan_minimum_ns = -1
        super().__init__(*args,**kwargs)
        old = self.mission
        self.mission = ReacquiringFloorMission(dict(goal_initial_body_xy_m=old.outbound.tolist(),
            return_initial_body_xy_m=old.home.tolist(),require_return_after_goal=old.require_return),
            navigation_ticks=old.navigation_ticks,arrival_radius_m=old.arrival_radius_m)

    def _register(self,item):
        # Same publication/mission path as ContinuousRoundTripRuntime, with an
        # explicit unavailable observation before any pose-dependent consumer.
        packet,raw = item
        evidence = self.registration.observe(packet.policy,packet.depth,packet.auxiliary_depth,raw,
            now_ns=packet.measured_ns)
        missing = evidence.get('status') == UNAVAILABLE
        with self.lock:
            command,previous = self.frame_request_history.pop(packet.frame)
        if missing:
            receipt = self.mission.unavailable(frame=packet.frame,now_ns=packet.measured_ns,
                previous_requested_command=command)
        else:
            pose = evidence['current_pose']
            receipt = self.mission.advance(np.asarray(pose['position_initial_body_m'],float),
                frame=packet.frame,now_ns=packet.measured_ns,previous_requested_command=command)
        published_ns = self.clock_ns()
        receipt = receipt | dict(preceding_20ms_requests=previous,published_ns=published_ns,
            consumed_pose_frame=None if missing else pose['frame'])
        with self.lock:
            if missing:
                self.floor_waiting = True; self.floor_valid_streak = 0
                self.floor_plan_minimum_ns = packet.measured_ns+400_000_000
                receipt['floor_rejection'] = evidence
            else:
                self.floor_valid_streak += 1
                if self.floor_valid_streak >= 4:
                    self.floor_waiting = False
            receipt['floor_reacquisition_hold'] = self.floor_waiting
            receipt['consecutive_accepted_floor_poses'] = self.floor_valid_streak
            self.mission_rows.append(receipt); self.mission_latest = receipt
            self.mission_terminal = receipt['terminal']
            self.goal = np.asarray(receipt['active_goal_initial_body_xy_m'],float)
            if receipt.get('phase_transition') is not None:
                self.mission_generation += 1; self.scan_target = None; self.scan_index = 0
            if self.floor_waiting or receipt['hold_required']:
                reason = 'FLOOR_REACQUISITION_HOLD' if self.floor_waiting else 'MISSION_SETTLING_OR_TERMINAL'
                for plan in self.plans:
                    self.rejected_windows[plan.observed_ns] = reason
                    self.commitment_ledger.veto(plan,published_ns)
        if missing:
            return
        if self.evidence_sink is not None:
            self.evidence_sink(packet.frame,raw,evidence)
        if packet.frame%4 == 0:
            self.queues['mapping'].put_nowait((packet,evidence))
            if packet.frame >= 4:
                self.queues['planning'].put_nowait((packet,evidence))

    def _plan(self,item):
        packet,_ = item
        with self.lock:
            if self.floor_waiting or packet.measured_ns < self.floor_plan_minimum_ns:
                self.planning.append(dict(frame=packet.frame,measured_ns=packet.measured_ns,
                    reason='FLOOR_REACQUISITION_HOLD'))
                return
        super()._plan(item)

    def _store_plan(self,plan,completed,prefix):
        # Called under the controller lock; also rejects work that was already
        # computing when a later floor observation became unavailable.
        if self.floor_waiting or plan.observed_ns < self.floor_plan_minimum_ns:
            self.planning[-1].update(committed=False,discard_reason='FLOOR_REACQUISITION_HOLD')
            return
        super()._store_plan(plan,completed,prefix)

    def _command_gate(self,result,now_ns):
        result = super()._command_gate(result,now_ns)
        if self.floor_waiting:
            return result | dict(requested_command=[0.,0.,0.],reason='FLOOR_REACQUISITION_HOLD')
        return result


def initialize_registration():
    from scripts.run_go2_live_gyro_height_floor_noise_development import initialize_registration as previous
    from lewm import process_registered_round_trip_development as process
    previous()
    process._registration = ReacquiringFloorRegistration()

"""Separate later-floor controller with measured settling before arrival claims."""
from copy import deepcopy
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.joint_floor_registered_evidence_development import current_joint_floor_registered_pose
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.auxiliary_depth_reobserve_goal_probe_development import NO_FEASIBLE, MAX_CONSECUTIVE_WAIT_COMMANDS
from lewm.later_floor_resolution_controller_development import LaterFloorResolutionRoundTripController
from lewm.measured_settling_round_trip_mission_development import MeasuredSettlingRoundTripMission


class MeasuredSettlingRoundTripController(LaterFloorResolutionRoundTripController):
    def __init__(self, model, geometry, *, public_mission, navigation_ticks, **kwargs):
        super().__init__(model, geometry, public_mission=public_mission, navigation_ticks=navigation_ticks, **kwargs)
        self.mission = MeasuredSettlingRoundTripMission(public_mission, navigation_ticks=navigation_ticks)

    def advance(self, policy, evidence, *, now_ns):
        if self.terminal is not None:
            return self._result([0., 0., 0.], None, None)
        self.infeasible_wait_active = False
        try:
            if type(now_ns) is not int or now_ns != 1_500_000_000+(self.tick+1)*100_000_000:
                raise SensorContractError('uninterrupted exact mission observation clock required')
            p, _, pose = current_joint_floor_registered_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
            if pose['frame'] != self.tick+1:
                raise SensorContractError('uninterrupted measured mission frame required')
            self.history.append(deepcopy(policy))
            history = causal_history_tensors(list(self.history), now_ns) if len(self.history) == 4 else None
            self.residual.observe(evidence, now_ns=now_ns)
            self.tick += 1; self.last_ns = now_ns
            self.mission_receipt = self.mission.advance(p, frame=self.tick, now_ns=now_ns,
                previous_requested_command=self.previous_command)
            mission = self.mission_receipt
            self.terminal = mission['terminal']; self.failure = mission['failure']
            self.quiet = mission.get('quiet_intervals', 0)
            # Only target-specific scanning state changes. Map, tracker, learned
            # history and residual memory stay alive across the return transition.
            self.selector.set_goal(mission['active_goal_initial_body_xy_m'])
            self.action = None; self.plan_offset = 0
            selection = None; requested = [0., 0., 0.]
            if not mission['hold_required']:
                selection = self.selector.choose(self.model, history, self.mapper, self.geometry, now_ns=now_ns)
                if selection['view_budget_exhausted']:
                    self.terminal = 'VIEW_BUDGET_EXHAUSTED'
                elif selection['action'] is None:
                    if 'prediction' not in selection:
                        raise ValueError('infeasibility requires a valid observed forecast')
                    self.infeasible_wait_count += 1
                    if self.infeasible_wait_count <= MAX_CONSECUTIVE_WAIT_COMMANDS:
                        self.infeasible_wait_active = True
                    else:
                        self.terminal = NO_FEASIBLE
                else:
                    if self.infeasible_wait_count: self.feasible_action_recoveries += 1
                    self.infeasible_wait_count = 0
                    self.action = selection['action']; self.plan_offset = 1
                    requested = list(candidate_commands(self.action)[0])
            self.previous_command = requested.copy()
            result = self._result(requested, selection, mission.get('observed_goal_distance_m'))
            if self.terminal is None: self.residual.remember(result)
            return result | dict(causal_residual_receipt=self.residual.snapshot())
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'; self.failure = str(error)
            self.previous_command = [0., 0., 0.]
            return self._result([0., 0., 0.], None, None)

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='measured_settling_round_trip_controller_v1',
            measured_settling_required_for_arrival=True)

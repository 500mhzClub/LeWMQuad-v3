"""Coordinate-only outbound/return control with persistent observed state.

This controller does not certify arrivals, physical backtracking, or clearance.
The executor must dispatch each request before acquiring the next observation.
"""
from copy import deepcopy
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.observed_floor_contact_development import ObservedFloorContactGoalProbe
from lewm.observed_round_trip_mission_development import ObservedRoundTripMission
from lewm.mission_target_eight_step_selection_development import MissionTargetEightStepSelector
from lewm.executed_horizon_final_goal_development import score_final_goal
from lewm.causal_residual_final_goal_development import correct_final_goal_score
from lewm.online_executed_residual_development import OnlineExecutedResidual
from lewm.auxiliary_depth_reobserve_goal_probe_development import (
    NO_FEASIBLE, MAX_CONSECUTIVE_WAIT_COMMANDS)


class RoundTripMissionSelector(MissionTargetEightStepSelector):
    def __init__(self, *, residual, **kwargs):
        super().__init__(**kwargs)
        self.residual = residual

    def choose(self, model, history, mapper, geometry, *, now_ns):
        receipt = self.residual.snapshot()
        if receipt['measured_ns'] != now_ns:
            raise ValueError('current observed residual state required')
        selection = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        return correct_final_goal_score(score_final_goal(selection), receipt)


class ObservedRoundTripController(ObservedFloorContactGoalProbe):
    def __init__(self, model, geometry, *, public_mission, navigation_ticks,
            condition, variant, persistent):
        self.mission = ObservedRoundTripMission(public_mission, navigation_ticks=navigation_ticks)
        self.mission_receipt = None
        super().__init__(model, geometry, condition=condition, variant=variant, persistent=persistent)
        self.residual = OnlineExecutedResidual()
        self.selector = RoundTripMissionSelector(residual=self.residual, condition=condition,
            variant=variant, goal_initial_body_xy_m=self.mission.target())

    def advance(self, policy, evidence, *, now_ns):
        if self.terminal is not None:
            return self._result([0., 0., 0.], None, None)
        self.infeasible_wait_active = False
        try:
            if type(now_ns) is not int or now_ns != 1_500_000_000+(self.tick+1)*100_000_000:
                raise SensorContractError('uninterrupted exact mission observation clock required')
            p, _, pose = current_joint_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
            if pose['frame'] != self.tick+1:
                raise SensorContractError('uninterrupted measured mission frame required')
            self.history.append(deepcopy(policy))
            history = causal_history_tensors(list(self.history), now_ns) if len(self.history) == 4 else None
            self.residual.observe(evidence, now_ns=now_ns)
            self.tick += 1; self.last_ns = now_ns
            self.mission_receipt = self.mission.advance(p[:2], frame=self.tick, now_ns=now_ns,
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
            controller='observed_round_trip_controller_v1',
            goal_initial_body_xy_m=self.mission.target().tolist(),
            mission_receipt=deepcopy(self.mission_receipt),
            causal_residual_receipt=self.residual.snapshot(),
            shared_navigation_budget_ticks=self.mission.navigation_ticks,
            mission_transition_retains_observed_state=True, verified_round_trip=False)

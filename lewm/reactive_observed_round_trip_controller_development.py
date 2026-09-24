"""Observed-route reactive control with shared sensors, memory and mission.

This baseline has no world model, forecast residual or predictive feasibility
filter. It retains the sensor admission, observed pose, map and arrival contract.
"""
from collections import deque
from copy import deepcopy
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.observed_floor_contact_development import ObservedFloorContactMap
from lewm.overlap_retention_joint_observer_development import OverlapRetentionVisualLedMotion
from lewm.observed_round_trip_mission_development import ObservedRoundTripMission
from lewm.reactive_observed_route_selection_development import ReactiveObservedRouteSelector
from lewm.auxiliary_downward45_depth_observation_development import CALIBRATION_ID
from lewm.auxiliary_depth_reobserve_goal_probe_development import MAX_CONSECUTIVE_WAIT_COMMANDS


class ReactiveObservedRoundTripController:
    def __init__(self, geometry, *, public_mission, navigation_ticks):
        self.geometry = geometry
        self.mapper = ObservedFloorContactMap(identity=(0, 0, 0)); self.memory = self.mapper.surface
        self.motion = OverlapRetentionVisualLedMotion(identity=(0, 0, 0))
        self.mission = ObservedRoundTripMission(public_mission, navigation_ticks=navigation_ticks)
        self.selector = ReactiveObservedRouteSelector(goal_initial_body_xy_m=self.mission.target())
        self.history = deque(maxlen=4); self.memory_receipt = None; self.mission_receipt = None
        self.tick = -1; self.last_ns = None; self.terminal = None; self.failure = None
        self.action = None; self.quiet = 0; self.previous_command = [0., 0., 0.]
        self.infeasible_wait_count = 0; self.infeasible_wait_active = False; self.feasible_action_recoveries = 0

    def observe(self, policy, depth, fast, *, now_ns, auxiliary_depth=None):
        evidence = None; self.memory_receipt = None
        try:
            if self.terminal is None:
                evidence = self.motion.observe(policy, depth, fast, now_ns=now_ns)
                self.memory_receipt = self.mapper.observe(policy, depth, evidence,
                    auxiliary_depth=auxiliary_depth, now_ns=now_ns)
            result = self.advance(policy, evidence, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'; self.failure = str(error)
            self.action = None; self.previous_command = [0., 0., 0.]
            result = self._result([0., 0., 0.], None, None)
        return result | dict(evidence=evidence)

    def advance(self, policy, evidence, *, now_ns):
        if self.terminal is not None: return self._result([0., 0., 0.], None, None)
        self.infeasible_wait_active = False
        try:
            if type(now_ns) is not int or now_ns != 1_500_000_000+(self.tick+1)*100_000_000:
                raise SensorContractError('uninterrupted exact mission observation clock required')
            p, _, pose = current_joint_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
            if pose['frame'] != self.tick+1: raise SensorContractError('uninterrupted measured mission frame required')
            self.history.append(deepcopy(policy))
            if len(self.history) == 4: causal_history_tensors(list(self.history), now_ns)
            self.tick += 1; self.last_ns = now_ns
            self.mission_receipt = self.mission.advance(p[:2], frame=self.tick, now_ns=now_ns,
                previous_requested_command=self.previous_command)
            mission = self.mission_receipt; self.terminal = mission['terminal']; self.failure = mission['failure']
            self.quiet = mission.get('quiet_intervals', 0)
            self.selector.set_goal(mission['active_goal_initial_body_xy_m'])
            self.action = None; selection = None; requested = [0., 0., 0.]
            if not mission['hold_required']:
                selection = self.selector.choose(self.mapper, self.geometry, now_ns=now_ns)
                if selection['view_budget_exhausted']:
                    self.terminal = 'VIEW_BUDGET_EXHAUSTED'
                elif selection['action'] is None:
                    if not selection['current_geometry_checked']:
                        raise ValueError('reactive infeasibility requires current measured geometry')
                    self.infeasible_wait_count += 1
                    if self.infeasible_wait_count <= MAX_CONSECUTIVE_WAIT_COMMANDS: self.infeasible_wait_active = True
                    else: self.terminal = 'NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY'
                else:
                    if self.infeasible_wait_count: self.feasible_action_recoveries += 1
                    self.infeasible_wait_count = 0; self.action = selection['action']
                    requested = list(selection['requested_command'])
            self.previous_command = requested.copy()
            return self._result(requested, selection, mission.get('observed_goal_distance_m'))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'; self.failure = str(error)
            self.action = None; self.previous_command = [0., 0., 0.]
            return self._result([0., 0., 0.], None, None)

    def _result(self, command, selection, distance):
        return dict(controller='reactive_observed_round_trip_controller_v1', planner_mode=self.selector.mode,
            model_condition=None, learned_model_used=False, candidate_future_outcomes_evaluated=False,
            input_variant='public_primary_rgbd_body_and_auxiliary_depth', memory_variant='persistent',
            memory_receipt=deepcopy(self.memory_receipt), mission_receipt=deepcopy(self.mission_receipt),
            requested_command=command, tick=self.tick, terminal=self.terminal, failure=self.failure,
            selected_action=self.action, plan_offset=int(self.action is not None), new_selection=selection,
            observed_goal_distance_m=distance, quiet_intervals=self.quiet,
            goal_initial_body_xy_m=self.mission.target().tolist(), native_state_used=False,
            auxiliary_depth_input_required=True, auxiliary_calibration_id=CALIBRATION_ID,
            auxiliary_floor_partition_receipt=deepcopy(self.memory.auxiliary_receipt) if self.memory_receipt is not None else None,
            infeasible_wait_active=self.infeasible_wait_active, consecutive_infeasible_observations=self.infeasible_wait_count,
            maximum_consecutive_wait_commands=MAX_CONSECUTIVE_WAIT_COMMANDS,
            feasible_action_recoveries=self.feasible_action_recoveries, shared_navigation_budget_ticks=self.mission.navigation_ticks,
            collision_contact_policy='observed_floor_contact_v1', ground_support_approved=False,
            experimental_motion_without_clearance_certificate=True, maximum_open_loop_command_ticks=1,
            mission_transition_retains_observed_state=True, verified_round_trip=False,
            predictive_surface_or_path_gates_applied=False, navigation_qualified=False, hardware_qualified=False)

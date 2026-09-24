"""Non-predictive route following with the current sensing, map and mission.

This is a method-level reactive comparator, not an isolated ranking ablation:
current geometry replaces predicted candidate feasibility and learned scores.
"""
from copy import deepcopy
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.auxiliary_depth_reobserve_goal_probe_development import MAX_CONSECUTIVE_WAIT_COMMANDS
from lewm.reactive_observed_round_trip_controller_development import ReactiveObservedRoundTripController
from lewm.reactive_connector_round_trip_controller_development import ReactiveConnectorRouteSelector
from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion, current_dual_camera_pose
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.measured_floor_transport_development import current_measured_floor_pose
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMap, MeasuredFloorTransportMission


class ReactiveFloorTransportController(ReactiveObservedRoundTripController):
    def __init__(self, geometry, *, public_mission, navigation_ticks):
        super().__init__(geometry, public_mission=public_mission, navigation_ticks=navigation_ticks)
        self.motion = DualCameraVisualMotion(identity=(0, 0, 0))
        self.registration = MeasuredFloorTransportRegistration(identity=(0, 0, 0))
        self.mapper = MeasuredFloorTransportMap(identity=(0, 0, 0)); self.memory = self.mapper.surface
        self.mission = MeasuredFloorTransportMission(public_mission, navigation_ticks=navigation_ticks)
        self.selector = ReactiveConnectorRouteSelector(goal_initial_body_xy_m=self.mission.target())

    def observe(self, policy, depth, fast, *, now_ns, auxiliary_depth=None, auxiliary_rgb=None):
        evidence = None; raw = None; self.memory_receipt = None
        try:
            if self.terminal is None:
                raw = self.motion.observe(policy, depth, fast, now_ns=now_ns,
                    auxiliary_depth=auxiliary_depth, auxiliary_rgb=auxiliary_rgb)
                current_dual_camera_pose(raw, policy, auxiliary_rgb, auxiliary_depth,
                    identity=(0, 0, 0), now_ns=now_ns)
                evidence = self.registration.observe(policy, depth, auxiliary_depth, raw, now_ns=now_ns)
                self.memory_receipt = self.mapper.observe(policy, depth, evidence,
                    auxiliary_depth=auxiliary_depth, now_ns=now_ns)
            result = self.advance(policy, evidence, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'; self.failure = str(error)
            self.action = None; self.previous_command = [0., 0., 0.]
            result = self._result([0., 0., 0.], None, None)
        return result | dict(evidence=evidence, original_visual_evidence=raw)

    def advance(self, policy, evidence, *, now_ns):
        if self.terminal is not None: return self._result([0., 0., 0.], None, None)
        self.infeasible_wait_active = False
        try:
            if type(now_ns) is not int or now_ns != 1_500_000_000+(self.tick+1)*100_000_000:
                raise SensorContractError('uninterrupted exact mission observation clock required')
            p, _, pose = current_measured_floor_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
            if pose['frame'] != self.tick+1: raise SensorContractError('uninterrupted measured mission frame required')
            self.history.append(deepcopy(policy))
            if len(self.history) == 4: causal_history_tensors(list(self.history), now_ns)
            self.tick += 1; self.last_ns = now_ns
            self.mission_receipt = self.mission.advance(p, frame=self.tick, now_ns=now_ns,
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
        return super()._result(command, selection, distance) | dict(
            controller='reactive_floor_transport_round_trip_controller_v1',
            input_variant='public_primary_rgbd_body_auxiliary_depth_and_rgb',
            additional_auxiliary_rgb_for_motion=True, floor_transport_during_missingness_enabled=True,
            measured_quiet_boundary_required_before_dwell=True,
            collision_contact_policy='later_complete_floor_evidence_nominal_feet_v1',
            later_measured_floor_contact_resolution_enabled=True,
            nearer_observed_route_target_policy_enabled=True,
            learned_model_used=False, candidate_future_outcomes_evaluated=False,
            predictive_surface_or_path_gates_applied=False, learned_residual_used=False,
            isolated_prediction_ranking_ablation=False, pose_uncertainty_calibrated=False)

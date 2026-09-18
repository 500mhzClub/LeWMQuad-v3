"""Current-geometry feedback using the learned planner's six command primitives.

This matches the action bank, sensing, memory and stopping machinery. It is
still a whole-method reactive comparison: desired-twist feedback replaces
forecast utility and predicted feasibility. No candidate pose is propagated.
"""
import math
import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.reactive_connector_round_trip_controller_development import ReactiveConnectorRouteSelector
from lewm.reactive_nominal_route_selection_development import HEADING_TOLERANCE_RAD
from lewm.stop_conditioned_comparators_development import StopConditionedReactiveController

COMMANDS = {action:tuple(candidate_commands(action)[0]) for action in ACTIONS}
MAX_FORWARD = .20
MAX_YAW = .45
HEADING_GAIN = MAX_YAW/HEADING_TOLERANCE_RAD


def six_action_selection(selection):
    if (selection.get('current_geometry_checked') is not True
            or selection.get('learned_model_used') is not False
            or selection.get('candidate_future_outcomes_evaluated') is not False
            or selection.get('command_integrated_pose_used') is not False):
        raise ValueError('original measured reactive selection required')
    geometry_clear = (selection['current_nominal_clearance']['nominal_disk_connector_clear']
        and not selection['current_surface_check']['possible_intersection'])
    error = selection['heading_error_rad']
    available = geometry_clear and not selection['view_budget_exhausted'] and error is not None
    if error is not None and not math.isfinite(error): raise ValueError('finite measured heading error required')
    translation_clear = bool(selection['mode'] == 'WAYPOINT'
        and selection.get('measured_waypoint_connector')
        and selection['measured_waypoint_connector']['nominal_disk_connector_clear'])
    desired = [0., 0., 0.]
    if available:
        desired[0] = MAX_FORWARD*max(0., math.cos(error)) if translation_clear else 0.
        desired[2] = float(np.clip(HEADING_GAIN*error, -MAX_YAW, MAX_YAW))
    candidates = []
    for action in ACTIONS:
        command = COMMANDS[action]
        eligible = bool(available and (command[0] == 0. or translation_clear))
        distance = ((command[0]-desired[0])/MAX_FORWARD)**2+((command[2]-desired[2])/MAX_YAW)**2
        candidates.append(dict(action=action, requested_command=list(command), eligible=eligible,
            normalized_command_distance_squared=float(distance)))
    feasible = [r for r in candidates if r['eligible']]
    chosen = min(feasible, key=lambda r:r['normalized_command_distance_squared']) if feasible else None
    command = [0., 0., 0.] if chosen is None else chosen['requested_command']
    return selection | dict(action=None if chosen is None else chosen['action'],
        requested_command=command, candidates=candidates, action_bank=list(ACTIONS),
        desired_instantaneous_command=desired, heading_gain_per_s=HEADING_GAIN,
        normalized_command_scales=[MAX_FORWARD, MAX_YAW],
        previous_three_action_choice=selection['action'],
        unknown_connector_motion_permitted=bool(command[0]>0 and selection.get('unknown_waypoint_connector_cells')),
        rule='nearest_six_action_primitive_to_current_waypoint_feedback',
        action_bank_matches_learned_planner=True, fully_nonpredictive_controller=True,
        matched_forecast_utility=False, matched_predictive_feasibility=False,
        retrospective_or_native_benefit_established=False)


class SixActionReactiveSelector(ReactiveConnectorRouteSelector):
    def choose(self, mapper, geometry, *, now_ns):
        return six_action_selection(super().choose(mapper, geometry, now_ns=now_ns))


class SixActionReactiveController(StopConditionedReactiveController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = SixActionReactiveSelector(goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='six_action_reactive_stop_conditioned_controller_v1',
            action_bank_matches_learned_planner=True, fully_nonpredictive_controller=True,
            reactive_is_whole_method_comparison=True)

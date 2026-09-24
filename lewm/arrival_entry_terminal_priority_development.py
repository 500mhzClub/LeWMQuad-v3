"""Keep heading guidance unless a terminal override predicts entering the goal."""
from copy import deepcopy
import numpy as np
from lewm.cached_fine_goal_route_development import CachedFineGoalRecoveryRuntime
from lewm.clearance_turn_recovery_development import choose
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.mission_coordinate_metric_development import position_distance


def require_arrival_entry(selection, prediction, *, arrival_radius_m):
    result = deepcopy(selection)
    priority = result.get('terminal_position_priority') or {}
    # Preserve subsequent hold/recovery selections and ordinary aligned movement.
    considered = bool(priority.get('changed')
        and result['action'] == priority.get('selected_action'))
    distance = None; changed = False
    if considered:
        i = ACTIONS.index(result['action'])
        endpoint = np.asarray(prediction)[i, 6, :2]
        distance = float(position_distance(endpoint-np.asarray(result['waypoint_body_xy_m']),
            result.get('position_metric_matrix')))
        if not np.isfinite(distance) or distance > arrival_radius_m:
            choose(result, ACTIONS.index(priority['original_action']))
            result['selection_objective'] = priority['original_selection_objective']
            changed = True
    result['arrival_entry_terminal_priority'] = dict(
        terminal_override_considered=considered, changed=changed,
        before_gate_action=selection['action'], selected_action=result['action'],
        predicted_translation_endpoint_distance_m=distance,
        required_arrival_radius_m=arrival_radius_m,
        scoring_endpoint_offset_ns=700_000_000,
        heading_guidance_restored_without_new_clearance_or_arrival_limits=True)
    return result


class ArrivalEntryTerminalPriorityRuntime(CachedFineGoalRecoveryRuntime):
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        if not self.terminal_position_approach:
            return result
        return require_arrival_entry(result, prediction,
            arrival_radius_m=self.mission.arrival_radius_m)

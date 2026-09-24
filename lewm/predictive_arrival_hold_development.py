"""Prefer stopping when the existing hold forecast predicts a quiet arrival.

This selects an existing action on its existing forecast. It neither declares
arrival nor changes the measured dwell, physical evaluation or action cadence.
"""
from copy import deepcopy
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.clearance_turn_recovery_development import choose
from lewm.fine_goal_route_development import FineGoalRouteRuntime
from lewm.mission_coordinate_metric_development import position_distance


def select_predicted_arrival_hold(selection, prediction, *, arrival_radius_m):
    if (selection.get('clearance_turn') or {}).get('active'):
        return selection
    hold = ACTIONS.index('hold')
    xy = np.asarray(prediction)[hold, -2:, :2]
    target = np.asarray(selection['waypoint_body_xy_m'])
    distances = position_distance(xy-target, selection.get('position_metric_matrix'))
    speed = float(np.linalg.norm(xy[1]-xy[0])/.1)
    clear = selection['memory_forecast_candidates'][hold]['nominal_predicted_path_clear']
    eligible = bool(clear and np.isfinite(xy).all()
        and np.all(distances <= arrival_radius_m) and speed <= .05)
    result = deepcopy(selection)
    original = result['action']
    if eligible:
        choose(result, hold)
        result['selection_objective'] = 'predicted_quiet_arrival_hold'
    result['predictive_arrival_hold'] = dict(eligible=eligible,
        original_action=original, selected_action=result['action'],
        changed=original != result['action'], predicted_terminal_distances_m=distances.tolist(),
        predicted_terminal_xy_speed_m_s=speed, observed_arrival_radius_m=arrival_radius_m,
        maximum_predicted_terminal_xy_speed_m_s=.05,
        existing_hold_forecast_used=True, measured_arrival_declared=False,
        measured_3d_settling_and_physical_dwell_unchanged=True)
    return result


class PredictiveArrivalHoldRuntime(FineGoalRouteRuntime):
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
        if self.terminal_position_approach:
            return select_predicted_arrival_hold(result, prediction,
                arrival_radius_m=self.mission.arrival_radius_m)
        return result

"""Instantaneous directional derivative of the existing waypoint objective."""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands


def angular_improvement_rate(error, error_rate):
    if abs(error) < 1e-12:
        return -abs(error_rate)
    if abs(abs(error)-math.pi) < 1e-12:
        return abs(error_rate)
    return -math.copysign(1.,error)*error_rate


def instantaneous_scores(goal, *, scan_error=None, pulse=False):
    goal = np.asarray(goal,float)
    if goal.shape!=(2,) or not np.isfinite(goal).all():
        raise ValueError('finite current waypoint required')
    if scan_error is not None and not math.isfinite(scan_error):
        raise ValueError('finite current view error required')
    radius = float(np.linalg.norm(goal))
    error = math.atan2(goal[1],goal[0]) if scan_error is None else scan_error
    scale = min(.35,radius) if scan_error is None else .35
    rows = []
    for action in ACTIONS:
        v,_,omega = candidate_commands(action)[0]
        duration = .1 if pulse and v else .4
        if scan_error is not None:
            position_rate = 0.
            error_rate = -omega
        elif radius > 1e-12:
            position_rate = v*math.cos(error)
            error_rate = v*math.sin(error)/radius-omega
        else:
            # Moving away from an already reached point increases distance.
            position_rate = -abs(v)
            error_rate = 0.
        alignment_rate = scale*angular_improvement_rate(error,error_rate)
        rows.append(dict(action=action,position_utility_m=duration*position_rate,
            alignment_utility_m=duration*alignment_rate,
            utility_m=duration*(position_rate+alignment_rate),command_duration_s=duration,
            eligible_for_view=not bool(v)))
    return rows


def replace_ranking(selection, *, pulse):
    result = deepcopy(selection)
    scan_error = result.get('scan_heading_error_rad')
    rows = instantaneous_scores(result['waypoint_body_xy_m'],scan_error=scan_error,pulse=pulse)
    result['instantaneous_ranking'] = dict(
        source='current_waypoint_cost_directional_derivative',rows=rows,
        forecast_ranked_action=result['action'],forecast_candidates=deepcopy(result['candidates']),
        forecast_scan_utilities=deepcopy(result.get('scan_utilities')),
        predictions_used_for_main_utilities=False,
        predicted_clearance_recovery_and_arrival_gates_retained=True,
        contact_score_disabled=True,full_online_rollout_ablation=False)
    for candidate,row in zip(result['candidates'],rows):
        assert candidate['action']==row['action']
        candidate['utility_m']=row['utility_m']
        candidate['position_contact_utility_m']=row['position_utility_m']
    eligible = [r for r in rows if scan_error is None or r['eligible_for_view']]
    if scan_error is not None:
        result['scan_utilities']=[dict(action=r['action'],utility_m=r['utility_m']) for r in eligible]
    preferred = max(eligible,key=lambda r:r['utility_m'])['action']
    result.update(action=preferred,action_index=ACTIONS.index(preferred),
        requested_command=candidate_commands(preferred)[0],
        selection_objective='instantaneous_distance_and_heading_cost_directional_derivative')
    result['instantaneous_ranking']['instantaneous_ranked_action']=preferred
    return result


class InstantaneousWaypointScoreMixin:
    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):
        revised = replace_ranking(selected,pulse=bool(self.planning_translation_pulse))
        # Correct action-conditioned forecasts still reach every existing
        # feasibility, recovery, arrival and planned stopping check.
        return super()._select_clear_prediction(revised,prediction,snapshot,position,rotation)

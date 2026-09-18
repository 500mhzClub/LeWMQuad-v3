"""Score model-predicted waypoint alignment over the actual commitment interval."""
import numpy as np
from lewm.delayed_action_planning_development import score_delayed_predictions
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.partial_height_round_trip_development import PartialHeightRoundTripRuntime
from lewm.mission_coordinate_metric_development import position_distance


def score_waypoint_alignment(prediction, goal_body_xy_m, **kwargs):
    result = score_delayed_predictions(prediction, goal_body_xy_m, **kwargs)
    p = np.asarray(prediction)
    goal = np.asarray(goal_body_xy_m, float)
    begin = kwargs.get('delay_ticks', 2)-1
    end = begin+kwargs.get('commit_ticks', 1)
    # Reuse the existing view-turn scale, capped by waypoint distance so
    # orientation cannot dominate an arbitrarily close positional target.
    scale = min(.35, float(position_distance(goal, kwargs.get('position_metric_matrix'))))
    errors = []
    for index in (begin, end):
        delta = goal-p[:, index, :2]
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        yaw = np.arctan2(p[:, index, 2], p[:, index, 3])
        error = np.arctan2(np.sin(bearing-yaw), np.cos(bearing-yaw))
        errors.append(np.abs(error))
    alignment = scale*(errors[0]-errors[1])
    for i, row in enumerate(result['candidates']):
        row['position_contact_utility_m'] = row['utility_m']
        row['predicted_alignment_progress_m'] = float(alignment[i])
        row['predicted_heading_error_at_commit_start_rad'] = float(errors[0][i])
        row['predicted_heading_error_at_commit_end_rad'] = float(errors[1][i])
        row['utility_m'] += float(alignment[i])
    selected = max(range(6), key=lambda i: result['candidates'][i]['utility_m'])
    action = result['candidates'][selected]['action']
    result.update(action=action, action_index=selected,
        requested_command=candidate_commands(action)[0],
        waypoint_body_xy_m=goal.tolist(), alignment_scale_m=scale,
        selection_objective='distance_and_heading_progress_minus_contact')
    return result


class WaypointAlignmentRoundTripRuntime(PartialHeightRoundTripRuntime):
    _score = staticmethod(score_waypoint_alignment)

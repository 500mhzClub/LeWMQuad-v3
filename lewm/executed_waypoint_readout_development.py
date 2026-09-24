"""Executed selected-waypoint forecast errors; native positions are evaluator-only."""
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.nominal_reentry_execution_readout_development import executed_motion


def waypoint_execution(poses, tape, rows):
    entries = []
    for row in rows:
        d = row['decision']; s = d['new_selection'] or {}
        if (not s.get('executed_waypoint_scoring') or s['action'] is None
                or d['terminal'] is not None):
            continue
        action = s['action']; i = ACTIONS.index(action)
        if (d['selected_action'] != action or d['requested_command'] != candidate_commands(action)[0]
                or s['mode'] != 'WAYPOINT' or s.get('intermediate_target_is_mission_goal', False)
                or s['scored_pose_horizon_ns'] != 100_000_000
                or not s['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']):
            raise ValueError('actual original-feasible intermediate-waypoint selection required')
        xy = np.asarray(s['candidates'][i]['causal_scoring_body_xy_m'], float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('finite causal scoring XY required')
        entries.append(dict(tick=row['tick'], action=action, requested_command=d['requested_command'],
            predicted_body_xy_m=s['prediction'][i][0][:2], causal_scoring_body_xy_m=xy.tolist(),
            original_waypoint_action=s['original_waypoint_action'],
            local_reranking_changed_action=s['original_waypoint_action'] != action,
            current_waypoint_distance_m=float(np.linalg.norm(s['goal_body_xy_m'])),
            observed_residual_samples=s['causal_score_residual_receipt']['observed_residual_samples']))
    records = executed_motion(poses, tape, entries)
    for record in records:
        record['causal_scoring_xy_error_m'] = (float(np.linalg.norm(
            np.asarray(record['causal_scoring_body_xy_m'])-record['native_body_xy_m']))
            if record['complete_100ms_execution'] else None)
    complete = [r for r in records if r['complete_100ms_execution']]
    return dict(records=records, completed_intervals=len(complete), censored_intervals=len(records)-len(complete),
        local_rerank_changes=sum(r['local_reranking_changed_action'] for r in records),
        raw_forecast_mean_xy_error_m=float(np.mean([r['forecast_xy_error_m'] for r in complete])) if complete else None,
        causal_scoring_mean_xy_error_m=float(np.mean([r['causal_scoring_xy_error_m'] for r in complete])) if complete else None,
        native_outcomes_used_for_policy=False, unexecuted_alternative_outcomes_inferred=False,
        original_controller_counterfactual_trajectory_inferred=False)

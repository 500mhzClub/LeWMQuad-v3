"""Causal 100 ms intermediate-waypoint value with original 800 ms constraints."""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from lewm.observation_horizon_waypoint_utility_development import potential, CONTACT_PENALTY_M


def score_waypoint_execution(selection, receipt):
    if ('prediction' not in selection or selection.get('mode') != 'WAYPOINT'
            or selection.get('intermediate_target_is_mission_goal', False)
            or selection.get('nominal_clearance_reentry', False)):
        return selection
    require_short_forecast(selection)
    p = np.asarray(selection['prediction'], float)
    bias = np.asarray(receipt['correction_xy_m'], float)
    if (p.shape != (6, 8, 5) or not np.isfinite(p).all()
            or np.any(np.hypot(p[:, :, 2], p[:, :, 3]) <= 1e-8)
            or (np.diff(p[:, :, 4], axis=1) < -1e-6).any()
            or selection.get('actual_commitment_horizon_ns') != 100_000_000
            or selection.get('path_constraint_horizon_ns') != 800_000_000
            or selection.get('scored_horizon_ns') != 800_000_000
            or selection.get('planned_segment_count') != 8
            or selection.get('native_state_used') is not False
            or [r['action'] for r in selection['candidates']] != list(ACTIONS)
            or [r['action'] for r in selection['nominal_path_checks']] != list(ACTIONS)
            or len(selection['surface_checks']) != 6
            or bias.shape != (2,) or not np.isfinite(bias).all()
            or receipt['native_outcomes_used'] or receipt['command_integrated_pose_used']
            or any(t >= receipt['frame'] for t in receipt['residual_source_ticks'])
            or any(t > receipt['frame'] for t in receipt['residual_available_ticks'])):
        raise ValueError('original checked forecast and strictly causal observed residual required')
    allowed = selection['phase_allowed_actions']
    if not allowed or any(a not in ACTIONS for a in allowed):
        raise ValueError('original phase allowance required')
    result = deepcopy(selection)
    goal = np.asarray(result['goal_body_xy_m'], float)
    initial, distance, alignment = potential(goal, 0.)
    for i, candidate in enumerate(result['candidates']):
        xy = p[i, 0, :2]-bias
        final, end_distance, end_alignment = potential(goal-xy, math.atan2(p[i, 0, 2], p[i, 0, 3]))
        contact = float(np.exp(-np.logaddexp(0., -p[i, -1, 4])))
        candidate.update(utility_m=initial-final-CONTACT_PENALTY_M*contact,
            causal_scoring_body_xy_m=xy.tolist(),
            executed_waypoint_distance_progress_m=distance-end_distance,
            executed_waypoint_alignment_progress_m=alignment-end_alignment,
            full_plan_contact_score=contact)
    feasible = [i for i, a in enumerate(ACTIONS) if a in allowed
        and not result['surface_checks'][i]['possible_intersection']
        and result['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
    chosen = max(feasible, key=lambda i: result['candidates'][i]['utility_m']) if feasible else None
    result.update(action=None if chosen is None else ACTIONS[chosen], action_index=chosen,
        requested_command=[0., 0., 0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        phase_admissible_candidates=len(feasible), scored_horizon_ns=100_000_000,
        scored_pose_horizon_ns=100_000_000, scored_contact_horizon_ns=800_000_000,
        score_contract='causal_executed_waypoint_potential_minus_full_plan_contact',
        executed_waypoint_scoring=True, causal_score_residual_receipt=deepcopy(receipt),
        original_waypoint_candidates=deepcopy(selection['candidates']),
        original_waypoint_action=selection['action'], original_waypoint_score_contract=selection['score_contract'],
        raw_forecasts_and_constraints_preserved=True, corrected_scoring_path_checked=False)
    return result

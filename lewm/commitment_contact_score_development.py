"""Prospective 100 ms contact cost; retain every original 800 ms veto."""
from copy import deepcopy
import math
import numpy as np
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.observation_horizon_waypoint_utility_development import potential, CONTACT_PENALTY_M


def score_commitment_contact(selection):
    if (not selection or 'prediction' not in selection or selection.get('mode') != 'WAYPOINT'
            or selection.get('intermediate_target_is_mission_goal', False)
            or selection.get('nominal_clearance_reentry', False)):
        return selection
    if (selection.get('score_contract') != 'causal_executed_waypoint_potential_minus_full_plan_contact'
            or selection.get('scored_pose_horizon_ns') != 100_000_000
            or selection.get('scored_contact_horizon_ns') != 800_000_000):
        raise ValueError('original executed-waypoint pose and full-plan contact contract required')
    # Reuse the frozen scorer's forecast/causality checks, also authenticating
    # every inherited score and receipt rather than accepting an arbitrary bank.
    prior = deepcopy(selection)
    prior.update(scored_horizon_ns=800_000_000,
        candidates=deepcopy(selection['original_waypoint_candidates']),
        action=selection['original_waypoint_action'], score_contract=selection['original_waypoint_score_contract'])
    if score_waypoint_execution(prior, selection['causal_score_residual_receipt']) != selection:
        raise ValueError('complete original executed-waypoint selection must reconstruct exactly')
    result = deepcopy(selection)
    p = np.asarray(selection['prediction'], float)
    bias = np.asarray(selection['causal_score_residual_receipt']['correction_xy_m'], float)
    goal = np.asarray(selection['goal_body_xy_m'], float)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError('finite current intermediate waypoint required')
    initial, _, _ = potential(goal, 0.)
    for i, candidate in enumerate(result['candidates']):
        final, _, _ = potential(goal-(p[i, 0, :2]-bias), math.atan2(p[i, 0, 2], p[i, 0, 3]))
        contact = float(np.exp(-np.logaddexp(0., -p[i, 0, 4])))
        candidate.update(utility_m=initial-final-CONTACT_PENALTY_M*contact,
            commitment_contact_score=contact)
    feasible = [i for i, action in enumerate(ACTIONS) if action in selection['phase_allowed_actions']
        and not selection['surface_checks'][i]['possible_intersection']
        and selection['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
    chosen = max(feasible, key=lambda i: result['candidates'][i]['utility_m']) if feasible else None
    result.update(action=None if chosen is None else ACTIONS[chosen], action_index=chosen,
        requested_command=[0., 0., 0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        phase_admissible_candidates=len(feasible), scored_contact_horizon_ns=100_000_000,
        score_contract='causal_executed_waypoint_potential_minus_commitment_contact',
        commitment_contact_scoring=True, original_full_contact_candidates=deepcopy(selection['candidates']),
        original_full_contact_action=selection['action'], contact_penalty_coefficient_m=CONTACT_PENALTY_M,
        contact_scores_calibrated=False)
    return result

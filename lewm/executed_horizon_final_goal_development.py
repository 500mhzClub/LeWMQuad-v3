"""Score the exact final target at execution time, retaining full-plan risk."""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from lewm.observation_horizon_waypoint_utility_development import potential, CONTACT_PENALTY_M
from lewm.exact_mission_target_goal_probe_development import (
    ExactMissionTargetEightStepSelector, ExactMissionTargetGoalProbe)


def score_final_goal(selection):
    """Rerank an already checked bank; never turn a veto into a feasible action.

    Only the original observed final-goal connector activates this score.
    Forecasts, contact penalty, phase allowance and all clearance checks retain
    their original values. Contact probabilities are uncalibrated scores.
    """
    if ('prediction' not in selection or selection.get('mode') != 'WAYPOINT'
            or not selection.get('intermediate_target_is_mission_goal', False)):
        return selection
    require_short_forecast(selection)
    p = np.asarray(selection['prediction'], float)
    if (selection.get('terminal_goal_target_evidence', {}).get('selected') is not True
            or selection.get('actual_commitment_horizon_ns') != 100_000_000
            or selection.get('path_constraint_horizon_ns') != 800_000_000
            or selection.get('planned_segment_count') != 8
            or selection.get('scored_horizon_ns') != 800_000_000
            or p.shape != (6, 8, 5) or not np.isfinite(p).all()
            or (np.diff(p[:, :, 4], axis=1) < -1e-6).any()
            or np.any(np.hypot(p[:, :, 2], p[:, :, 3]) <= 1e-8)
            or [r['action'] for r in selection['candidates']] != list(ACTIONS)
            or [r['action'] for r in selection['nominal_path_checks']] != list(ACTIONS)
            or len(selection['surface_checks']) != 6):
        raise ValueError('original fully checked exact final-goal forecast required')
    allowed = selection['phase_allowed_actions']
    if not allowed or any(a not in ACTIONS for a in allowed):
        raise ValueError('original phase allowance required')
    result = deepcopy(selection)
    result['eight_step_final_goal_candidates'] = deepcopy(selection['candidates'])
    result['eight_step_final_goal_action'] = selection['action']
    goal = np.asarray(result['goal_body_xy_m'], float)
    initial, distance, alignment = potential(goal, 0.)
    for i, row in enumerate(result['candidates']):
        dx, dy, sy, cy, _ = p[i, 0]
        final, end_distance, end_alignment = potential(goal - [dx, dy], math.atan2(sy, cy))
        contact = float(np.exp(-np.logaddexp(0., -p[i, -1, 4])))
        row.update(utility_m=initial-final-CONTACT_PENALTY_M*contact,
            executed_goal_distance_progress_m=distance-end_distance,
            executed_goal_alignment_progress_m=alignment-end_alignment,
            full_plan_contact_score=contact)
    feasible = [i for i, action in enumerate(ACTIONS) if action in allowed
        and not result['surface_checks'][i]['possible_intersection']
        and result['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
    chosen = max(feasible, key=lambda i: result['candidates'][i]['utility_m']) if feasible else None
    result.update(action=None if chosen is None else ACTIONS[chosen], action_index=chosen,
        requested_command=[0., 0., 0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        phase_admissible_candidates=len(feasible), scored_horizon_ns=100_000_000,
        scored_pose_horizon_ns=100_000_000, scored_contact_horizon_ns=800_000_000,
        score_contract='executed_horizon_exact_goal_potential_minus_full_plan_contact',
        final_goal_execution_horizon_scoring=True)
    return result


class ExecutedHorizonFinalGoalSelector(ExactMissionTargetEightStepSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        return score_final_goal(super().choose(model, history, mapper, geometry, now_ns=now_ns))


class ExecutedHorizonFinalGoalProbe(ExactMissionTargetGoalProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ExecutedHorizonFinalGoalSelector(
            condition=self.selector.condition, variant=self.selector.variant)

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='executed_horizon_final_goal_probe_v1')

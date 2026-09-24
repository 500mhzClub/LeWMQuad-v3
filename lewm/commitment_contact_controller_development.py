"""Separate contact-horizon experiment; no current runner adopts this controller.

Score the first 100 ms contact outcome with the existing 100 ms progress value.
Retain the full raw forecast and every original 800 ms geometry constraint.
Contact scores remain uncalibrated, and shorter-horizon risk is not a safety
guarantee. Native navigation evidence is required before claiming a benefit.
"""
import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from lewm.observation_horizon_waypoint_utility_development import CONTACT_PENALTY_M
from lewm.extended_return_budget_comparator_controllers_development import ExtendedReturnBudgetForecastSelector
from lewm.stop_conditioned_comparators_development import StopConditionedForecastController


CONTRACTS = {
    'causal_executed_waypoint_potential_minus_full_plan_contact',
    'causal_residual_exact_goal_potential_minus_full_plan_contact',
}


def score_commitment_contact(selection):
    if (selection.get('score_contract') not in CONTRACTS
            or selection.get('nominal_clearance_reentry', False)):
        return selection
    require_short_forecast(selection)
    prediction = np.asarray(selection['prediction'], float)
    if (prediction.shape != (6, 8, 5) or not np.isfinite(prediction).all()
            or selection['scored_pose_horizon_ns'] != 100_000_000
            or selection['scored_contact_horizon_ns'] != 800_000_000
            or selection['actual_commitment_horizon_ns'] != 100_000_000
            or selection['path_constraint_horizon_ns'] != 800_000_000
            or [c['action'] for c in selection['candidates']] != list(ACTIONS)
            or [c['action'] for c in selection['nominal_path_checks']] != list(ACTIONS)
            or len(selection['surface_checks']) != len(ACTIONS)):
        raise ValueError('existing execution score and complete original geometry constraints required')
    # Copy only the fields edited here; forecast and geometry evidence is untouched.
    result = dict(selection)
    result['candidates'] = [dict(c) for c in selection['candidates']]
    for i, candidate in enumerate(result['candidates']):
        full = float(np.exp(-np.logaddexp(0., -prediction[i, -1, 4])))
        first = float(np.exp(-np.logaddexp(0., -prediction[i, 0, 4])))
        previous = candidate['utility_m']
        candidate.update(utility_m=previous+CONTACT_PENALTY_M*(full-first),
            utility_before_contact_intervention_m=previous,
            scored_commitment_contact_score=first)
    feasible = [i for i, action in enumerate(ACTIONS)
        if action in selection['phase_allowed_actions']
        and not selection['surface_checks'][i]['possible_intersection']
        and selection['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
    chosen = max(feasible, key=lambda i: result['candidates'][i]['utility_m']) if feasible else None
    result.update(action=None if chosen is None else ACTIONS[chosen], action_index=chosen,
        requested_command=[0., 0., 0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        phase_admissible_candidates=len(feasible), scored_contact_horizon_ns=100_000_000,
        score_contract=selection['score_contract'].replace('full_plan_contact', 'commitment_contact'),
        contact_horizon_intervention=dict(previous_horizon_ns=800_000_000,
            selected_horizon_ns=100_000_000, original_action=selection['action'],
            original_score_contract=selection['score_contract'], original_geometry_filters_retained=True))
    return result


class CommitmentContactSelector(ExtendedReturnBudgetForecastSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        return score_commitment_contact(super().choose(model, history, mapper, geometry, now_ns=now_ns))


class CommitmentContactController(StopConditionedForecastController):
    def __init__(self, *args, forecast_source='frozen_world_model', **kwargs):
        super().__init__(*args, forecast_source=forecast_source, **kwargs)
        self.selector = CommitmentContactSelector(forecast_source=forecast_source,
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        result = super()._result(*args, **kwargs)
        return result | dict(controller=result['controller']+'_commitment_contact',
            commitment_contact_scoring_experiment=True)

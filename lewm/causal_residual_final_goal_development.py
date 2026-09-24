"""Causal sensor residuals adjust final-goal utility, not raw checked paths."""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.observation_horizon_waypoint_utility_development import potential, CONTACT_PENALTY_M
from lewm.executed_horizon_final_goal_development import (
    ExecutedHorizonFinalGoalSelector, ExecutedHorizonFinalGoalProbe)
from lewm.online_executed_residual_development import OnlineExecutedResidual


def correct_final_goal_score(selection, receipt):
    if not selection.get('final_goal_execution_horizon_scoring', False): return selection
    bias = np.asarray(receipt['correction_xy_m'], float)
    if (bias.shape != (2,) or not np.isfinite(bias).all()
            or receipt['native_outcomes_used'] or receipt['command_integrated_pose_used']
            or any(t >= receipt['frame'] for t in receipt['residual_source_ticks'])
            or any(t > receipt['frame'] for t in receipt['residual_available_ticks'])
            or selection['scored_pose_horizon_ns'] != 100_000_000
            or selection['scored_contact_horizon_ns'] != 800_000_000):
        raise ValueError('causal public residual and original final-goal scoring horizons required')
    r = deepcopy(selection); p = np.asarray(r['prediction'], float)
    goal = np.asarray(r['goal_body_xy_m'], float); initial, distance, alignment = potential(goal, 0.)
    r['uncorrected_final_goal_candidates'] = deepcopy(selection['candidates'])
    r['uncorrected_final_goal_action'] = selection['action']
    for i, row in enumerate(r['candidates']):
        xy = p[i, 0, :2]-bias
        yaw = math.atan2(p[i, 0, 2], p[i, 0, 3])
        final, end_distance, end_alignment = potential(goal-xy, yaw)
        contact = float(np.exp(-np.logaddexp(0., -p[i, -1, 4])))
        row.update(utility_m=initial-final-CONTACT_PENALTY_M*contact,
            causal_scoring_body_xy_m=xy.tolist(), causal_goal_distance_progress_m=distance-end_distance,
            causal_goal_alignment_progress_m=alignment-end_alignment)
    feasible = [i for i, a in enumerate(ACTIONS) if a in r['phase_allowed_actions']
        and not r['surface_checks'][i]['possible_intersection']
        and r['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
    chosen = max(feasible, key=lambda i: r['candidates'][i]['utility_m']) if feasible else None
    r.update(action=None if chosen is None else ACTIONS[chosen], action_index=chosen,
        requested_command=[0., 0., 0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        phase_admissible_candidates=len(feasible), score_contract='causal_residual_exact_goal_potential_minus_full_plan_contact',
        causal_final_goal_scoring=True, causal_score_residual_receipt=deepcopy(receipt),
        raw_forecasts_and_constraints_preserved=True, corrected_scoring_path_checked=False)
    return r


class CausalResidualFinalGoalSelector(ExecutedHorizonFinalGoalSelector):
    def __init__(self, *, residual, condition, variant):
        super().__init__(condition=condition, variant=variant)
        self.residual = residual

    def choose(self, model, history, mapper, geometry, *, now_ns):
        receipt = self.residual.snapshot()
        if receipt['measured_ns'] != now_ns: raise ValueError('current observed residual state required')
        return correct_final_goal_score(super().choose(model, history, mapper, geometry, now_ns=now_ns), receipt)


class CausalResidualFinalGoalProbe(ExecutedHorizonFinalGoalProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual = OnlineExecutedResidual()
        self.selector = CausalResidualFinalGoalSelector(residual=self.residual,
            condition=self.selector.condition, variant=self.selector.variant)

    def advance(self, policy, evidence, *, now_ns):
        if self.terminal is None: self.residual.observe(evidence, now_ns=now_ns)
        result = super().advance(policy, evidence, now_ns=now_ns)
        if self.terminal is None: self.residual.remember(result)
        return result | dict(causal_residual_receipt=self.residual.snapshot())

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='causal_residual_final_goal_probe_v1', causal_residual_receipt=self.residual.snapshot())

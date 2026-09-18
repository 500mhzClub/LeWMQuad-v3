"""One freshly checked turn after ten discretionary waypoint hold requests.

This is a prospective policy intervention, not a prediction correction. It
permits a lower-scored turn without relaxing any saved raw geometry veto.
"""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.auxiliary_depth_reobserve_goal_probe_development import MAX_CONSECUTIVE_WAIT_COMMANDS
from lewm.observation_horizon_predictive_selection_development import require_short_forecast


class HoldReorientation:
    def __init__(self):
        self.last_frame = None
        self.holds = 0

    def reset_goal(self):
        self.holds = 0

    def reconsider(self, selection, *, frame, now_ns):
        if (type(frame) is not int or frame < 3 or type(now_ns) is not int
                or now_ns != 1_500_000_000 + frame * 100_000_000
                or (self.last_frame is not None and frame <= self.last_frame)):
            raise ValueError('fresh increasing exact observation clock required')
        if self.last_frame is not None and frame != self.last_frame + 1:
            self.holds = 0  # Mission settling and skipped choices cannot accrue credit.
        self.last_frame = frame
        if (selection.get('action') != 'hold' or selection.get('mode') != 'WAYPOINT'
                or selection.get('view_budget_exhausted', False)
                or selection.get('intermediate_target_is_mission_goal', False)
                or selection.get('nominal_clearance_reentry', False)
                or any(selection.get(k) is not None for k in (
                    'residual_first_interval_feasibility', 'residual_hold_feasibility',
                    'residual_anchored_continuation'))):
            self.holds = 0
            return selection
        require_short_forecast(selection)
        prediction = np.asarray(selection['prediction'], float)
        if (selection.get('action_index') != 0 or selection.get('requested_command') != [0., 0., 0.]
                or selection.get('score_contract') != 'causal_executed_waypoint_potential_minus_full_plan_contact'
                or selection.get('executed_waypoint_scoring') is not True
                or selection.get('actual_commitment_horizon_ns') != 100_000_000
                or selection.get('path_constraint_horizon_ns') != 800_000_000
                or selection.get('native_state_used') is not False
                or prediction.shape != (6, 8, 5) or not np.isfinite(prediction).all()
                or [r['action'] for r in selection['candidates']] != list(ACTIONS)
                or len(selection['surface_checks']) != 6 or len(selection['nominal_path_checks']) != 6
                or selection['causal_score_residual_receipt']['frame'] != frame
                or selection['causal_score_residual_receipt']['measured_ns'] != now_ns):
            raise ValueError('current original scored six-action forecast and hold required')
        utilities = [r['utility_m'] for r in selection['candidates']]
        if any(not math.isfinite(u) for u in utilities):
            raise ValueError('finite original model utilities required')
        eligible = [i for i, a in enumerate(ACTIONS)
            if a in selection['phase_allowed_actions']
            and not selection['surface_checks'][i]['possible_intersection']
            and selection['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
        if 0 not in eligible or any(utilities[i] > utilities[0] for i in eligible):
            raise ValueError('original hold must win among original raw-feasible actions')
        prior = self.holds
        self.holds = min(prior + 1, MAX_CONSECUTIVE_WAIT_COMMANDS)
        turns = [i for i in eligible if ACTIONS[i] in ('left_turn', 'right_turn')]
        if prior < MAX_CONSECUTIVE_WAIT_COMMANDS or not turns:
            return selection
        chosen = max(turns, key=lambda i: utilities[i])
        result = deepcopy(selection)
        result.update(action=ACTIONS[chosen], action_index=chosen,
            requested_command=list(candidate_commands(ACTIONS[chosen])[0]),
            hold_reorientation=dict(frame=frame, measured_ns=now_ns,
                preceding_discretionary_holds=prior,
                maximum_preceding_holds=MAX_CONSECUTIVE_WAIT_COMMANDS,
                original_action='hold', selected_action=ACTIONS[chosen],
                eligible_turns=[ACTIONS[i] for i in turns],
                original_hold_utility_m=utilities[0], selected_utility_m=utilities[chosen],
                original_forecasts_scores_and_vetoes_preserved=True,
                higher_utility_than_hold_required=False, geometry_veto_relaxed=False,
                command_horizon_ns=100_000_000, next_observation_required=True,
                physical_clearance_certified=False, goal_achieved=False))
        self.holds = 0
        return result

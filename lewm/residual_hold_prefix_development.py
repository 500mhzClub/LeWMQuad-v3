"""Compare an unchanged residual controller through the first changed hold."""
from copy import deepcopy
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.residual_first_interval_feasibility_development import causal_correction

MAX_FRAMES = 3004
CHOICE = ('action', 'action_index', 'requested_command', 'phase_admissible_candidates')


def compare_step(original, candidate, actual_command, *, frame):
    if (type(frame) is not int or not 0 <= frame < MAX_FRAMES
            or original['tick'] != frame or candidate['tick'] != frame
            or original['controller'] != 'residual_first_interval_feasibility_controller_v1'
            or original['residual_first_interval_feasibility_fallback_enabled'] is not True
            or candidate['controller'] != 'residual_hold_feasibility_controller_v1'
            or candidate['residual_hold_feasibility_enabled'] is not True
            or original['requested_command'] != actual_command
            or candidate['failure'] is not None):
        raise ValueError('exact original residual controller and explicit nonfailed hold successor required')
    old = original['new_selection']; new = candidate['new_selection']
    changed = bool(new and 'residual_hold_feasibility' in new)
    restored = deepcopy(new)
    if changed:
        receipt = restored.pop('residual_hold_feasibility')
        if (not old or old['action'] != 'hold' or old['action_index'] != 0
                or 'residual_first_interval_feasibility' in old
                or old['mode'] != 'WAYPOINT' or old['intermediate_target_is_mission_goal']
                or old.get('nominal_clearance_reentry', False) or old['view_budget_exhausted']
                or original['terminal'] is not None or candidate['terminal'] is not None
                or receipt['original_action'] != 'hold' or receipt['frame'] != frame
                or receipt['measured_ns'] != 1_500_000_000+frame*100_000_000
                or receipt['selected_action'] != new['action'] or new['action'] in (None, 'hold')):
            raise ValueError('current feasible waypoint hold intervention required')
        for key in ('strictly_higher_original_utility_required', 'all_eight_segments_checked',
                'later_predicted_points_unchanged', 'yaw_and_contact_forecasts_unchanged',
                'original_forecasts_and_veto_receipts_preserved', 'original_surface_vetoes_preserved',
                'raw_predictions_remain_residual_targets', 'nominal_policy_exception'):
            if receipt[key] is not True: raise ValueError('declared unchanged forecast constraint required: '+key)
        for key in ('model_error_bound_applied', 'physical_clearance_certified', 'goal_achieved'):
            if receipt[key] is not False: raise ValueError('unsupported claim: '+key)
        bias = causal_correction(old['causal_score_residual_receipt'], now_ns=receipt['measured_ns'])
        points = np.asarray(old['prediction'], float)[:, 0, :2]-bias
        utilities = [r['utility_m'] for r in old['candidates']]
        eligible = [i for i,a in enumerate(ACTIONS) if a in old['phase_allowed_actions']
            and not old['surface_checks'][i]['possible_intersection']
            and not receipt['corrected_surface_checks'][i]['possible_intersection']
            and receipt['corrected_nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
        selected = new['action_index']
        if (not old['nominal_path_checks'][0]['all_predicted_segments_nominally_clear']
                or old['surface_checks'][0]['possible_intersection'] or 'hold' not in old['phase_allowed_actions']
                or receipt['correction_xy_m'] != bias.tolist()
                or receipt['residual_source_ticks'] != old['causal_score_residual_receipt']['residual_source_ticks']
                or receipt['corrected_first_body_xy_m'] != points.tolist()
                or receipt['correction_horizon_ns'] != 100_000_000
                or not receipt['current_nominal_clearance']['nominal_disk_connector_clear']
                or receipt['eligible_actions'] != [ACTIONS[i] for i in eligible]
                or not eligible or selected != max(eligible, key=lambda i:utilities[i])
                or new['action'] != ACTIONS[selected] or utilities[selected] <= utilities[0]
                or receipt['original_hold_utility_m'] != utilities[0]
                or receipt['selected_utility_m'] != utilities[selected]
                or receipt['original_phase_admissible_candidates'] != old['phase_admissible_candidates']
                or new['phase_admissible_candidates'] != len(eligible)
                or new['requested_command'] != candidate_commands(new['action'])[0]
                or candidate['requested_command'] != new['requested_command']
                or candidate['selected_action'] != new['action']):
            raise ValueError('strictly better original utility and complete corrected feasibility required')
        for key in CHOICE: restored[key] = deepcopy(old[key])
    if restored != old:
        raise ValueError('complete original raw forecast, utilities, vetoes and fallback must remain exact')
    normalized = deepcopy(candidate)
    normalized.pop('residual_hold_feasibility_enabled')
    normalized['controller'] = original['controller']; normalized['new_selection'] = restored
    if changed:
        # Both policies select a feasible menu action; wait, terminal, plan and
        # all observed/residual/mission state must therefore remain identical.
        for key in ('requested_command', 'selected_action'):
            normalized[key] = deepcopy(original[key])
    if normalized != original:
        raise ValueError('complete unchanged observed, mission, residual and execution state required')
    command_changed = candidate['requested_command'] != actual_command
    if changed != command_changed:
        raise ValueError('hold intervention must stop at its first changed physical request')
    return dict(hold_reconsideration_changed_action=changed, requested_command_changed=command_changed,
        raw_model_forecasts_compared=bool(old and 'prediction' in old),
        complete_original_selection_preserved=True, unchanged_observed_mission_and_residual_state_exact=True,
        terminal_changed=False)

"""Exact causal-state comparison for the declared first-interval intervention."""
from copy import deepcopy
from lewm.geometry_progress_pilot_development import candidate_commands

MAX_FRAMES = 504
CHOICE = ('action', 'action_index', 'requested_command', 'phase_admissible_candidates')
CONTROL = ('requested_command', 'terminal', 'selected_action', 'plan_offset',
           'infeasible_wait_active', 'consecutive_infeasible_observations', 'feasible_action_recoveries')


def compare_step(original, candidate, actual_command, *, frame, prior_policy_changed=False):
    if (type(frame) is not int or not 0 <= frame < MAX_FRAMES
            or original['tick'] != frame or candidate['tick'] != frame
            or original['requested_command'] != actual_command
            or original['controller'] != 'measured_floor_transport_round_trip_controller_v1'
            or candidate['controller'] != 'residual_first_interval_feasibility_controller_v1'
            or candidate['residual_first_interval_feasibility_fallback_enabled'] is not True
            or candidate['failure'] is not None):
        raise ValueError('exact current observed baseline and explicit nonfailed successor required')
    old = original['new_selection']; new = candidate['new_selection']
    attempted = bool(new and 'residual_first_interval_feasibility' in new)
    policy_changed = False
    restored = deepcopy(new)
    if attempted:
        receipt = restored.pop('residual_first_interval_feasibility')
        if (not old or old['action'] is not None or receipt['original_action'] is not None
                or receipt['frame'] != frame or receipt['measured_ns'] != 1_500_000_000+frame*100_000_000
                or receipt['selected_action'] != new['action']
                or receipt['raw_predictions_remain_residual_targets'] is not True
                or receipt['original_surface_vetoes_preserved'] is not True
                or receipt['all_eight_segments_checked'] is not True):
            raise ValueError('explicit current original-infeasibility fallback receipt required')
        policy_changed = old['action'] != new['action']
        for key in CHOICE: restored[key] = deepcopy(old[key])
    if restored != old:
        raise ValueError('complete original forecast, scoring and veto selection must be unchanged')
    normalized = deepcopy(candidate)
    normalized.pop('residual_first_interval_feasibility_fallback_enabled')
    normalized['controller'] = original['controller']
    normalized['new_selection'] = restored
    # A newly feasible hold can reset the wait counter without changing the
    # dispatched command. Retain those declared policy changes in the output.
    if prior_policy_changed or policy_changed:
        for key in CONTROL: normalized[key] = deepcopy(original[key])
        if candidate['terminal'] != original['terminal']:
            for d in (candidate, original):
                if d['causal_residual_receipt']['pending_forecast_tick'] not in (None, frame):
                    raise ValueError('only the current forecast may be pending at a terminal boundary')
            normalized['causal_residual_receipt']['pending_forecast_tick'] = original['causal_residual_receipt']['pending_forecast_tick']
    if normalized != original:
        raise ValueError('unchanged complete observed, mission and residual state required')
    if candidate['terminal'] is None and new and new.get('action') is not None:
        if candidate['requested_command'] != candidate_commands(new['action'])[0]:
            raise ValueError('actual candidate request must match its selected menu action')
    return dict(complete_original_selection_preserved=True,
        unchanged_observed_mission_and_residual_state_exact=True,
        raw_model_forecasts_compared=bool(old and 'prediction' in old),
        fallback_attempted=attempted, policy_action_changed=policy_changed,
        requested_command_changed=candidate['requested_command'] != actual_command,
        terminal_changed=candidate['terminal'] != original['terminal'])

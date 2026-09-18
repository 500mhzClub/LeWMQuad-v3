"""Exact full-controller comparison against the frozen saved selection boundary."""
from copy import deepcopy


def compare_step(old, new, actual_command, *, frame, expected_selection):
    if (type(frame) is not int or not 0 <= frame < 3014
            or old['tick'] != frame or new['tick'] != frame
            or old['controller'] != 'residual_anchored_continuation_controller_v1'
            or new['controller'] != 'hold_reorientation_controller_v1'
            or new['hold_reorientation_enabled'] is not True
            or old['requested_command'] != actual_command
            or old['terminal'] is not None or new['terminal'] is not None
            or old['failure'] is not None or new['failure'] is not None):
        raise ValueError('exact nonterminal original and prospective controller decisions required')
    a, b = old['new_selection'], new['new_selection']
    if b != expected_selection:
        raise ValueError('raw candidate selection must match the frozen saved-selection expectation')
    changed = bool(b and 'hold_reorientation' in b)
    restored = deepcopy(b)
    if changed:
        receipt = restored.pop('hold_reorientation')
        if (not a or a['action'] != 'hold' or a['requested_command'] != actual_command
                or b['action'] not in ('left_turn', 'right_turn')
                or b['requested_command'] != new['requested_command']
                or b['action'] != new['selected_action']
                or receipt['frame'] != frame
                or receipt['measured_ns'] != 1_500_000_000 + frame*100_000_000):
            raise ValueError('declared current hold-to-turn boundary required')
        for key in ('action', 'action_index', 'requested_command'):
            restored[key] = deepcopy(a[key])
    if restored != a:
        raise ValueError('complete original forecasts, scores, gates and selection evidence must remain exact')
    normalized = deepcopy(new)
    normalized.pop('hold_reorientation_enabled')
    normalized['controller'] = old['controller']
    normalized['new_selection'] = restored
    if changed:
        for key in ('requested_command', 'selected_action'):
            normalized[key] = deepcopy(old[key])
    if normalized != old:
        raise ValueError('complete observed, mission, residual and execution state must remain exact')
    command_changed = new['requested_command'] != actual_command
    if changed != command_changed:
        raise ValueError('declared intervention must be the first changed physical request')
    return dict(requested_command_changed=command_changed, terminal_changed=False,
        raw_model_forecasts_compared=bool(a and 'prediction' in a),
        complete_original_selection_preserved=True,
        candidate_matches_saved_selection_expectation=True,
        normalized_complete_decision_exact=True,
        unchanged_observed_mission_and_executed_residual_state_exact=True)

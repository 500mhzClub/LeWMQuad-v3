"""Non-predictive feedback retaining action reserve and terminal position priority."""
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.eligible_floor_registration_development import bind
from lewm.rollout_selection_off_development import select_current_clearance,RolloutSelectionOffMixin

ACTION_RESERVE_M=.03


def select_reserved_terminal(goal, *, scan_error=None, pulse=False, clearance_m=None):
    result=select_current_clearance(goal,scan_error=scan_error,pulse=pulse,clearance_m=clearance_m)
    for row in result['candidates']:
        required=.45+(ACTION_RESERVE_M if row['action']!='hold' else 0.)
        clear=clearance_m is None or clearance_m>required+1e-12
        row.update(required_current_clearance_m=required,current_reserve_clear=clear,
            eligible=bool(clear and (scan_error is None or row['eligible_for_view'])))
    eligible=[r for r in result['candidates'] if r['eligible']]
    preferred=max(eligible,key=lambda r:r['utility_m']) if eligible else result['candidates'][0]
    selected=preferred
    if pulse and scan_error is None:
        # The mission has no terminal heading requirement. Prefer positive
        # instantaneous distance reduction; use heading feedback if none exists.
        translations=[r for r in eligible if r['action'] in ('forward','left_arc','right_arc')
            and r['position_utility_m']>max(0.,preferred['position_utility_m'])]
        if translations:selected=max(translations,key=lambda r:r['position_utility_m'])
    action=selected['action']
    result.update(action=action,action_index=ACTIONS.index(action),
        requested_command=candidate_commands(action)[0],
        selection_objective='instantaneous_feedback_with_current_reserve_and_terminal_position_priority',
        current_action_reserve_clear=clearance_m is None or clearance_m>.48+1e-12,
        current_reserve_terminal=dict(action_reserve_m=ACTION_RESERVE_M,
            reserve_applies_to_translation_and_turns=True,
            terminal_position_priority_active=bool(pulse and scan_error is None),
            heading_preferred_action=preferred['action'],selected_action=action,
            position_priority_changed=action!=preferred['action'],
            terminal_priority_uses_instantaneous_distance_derivative=True,
            future_pose_or_arrival_projection_used=False))
    return result


class ReservedTerminalFeedbackMixin:
    _select_clear_prediction=bind(RolloutSelectionOffMixin._select_clear_prediction,
        select_current_clearance=select_reserved_terminal)

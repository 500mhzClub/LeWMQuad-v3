"""Separate changed frontier planning from unchanged observed evidence."""
from copy import deepcopy

SHARED = ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt',
    'observed_goal_distance_m', 'auxiliary_floor_partition_receipt')


def compare_step(old, new, actual_command, *, frame, frontier_previously_reached):
    if (type(frame) is not int or not 0 <= frame < 3014 or old['tick'] != frame or new['tick'] != frame
            or old['controller'] != 'recent_qualified_direct_flow_controller_v1'
            or new['controller'] != 'reached_frontier_recent_qualified_controller_v1'
            or new['reached_frontier_transition_enabled'] is not True
            or old['requested_command'] != actual_command):
        raise ValueError('exact original and prospective frontier observation required')
    for key in SHARED:
        if old[key] != new[key]: raise ValueError('unchanged observed evidence differs: '+key)
    residuals = [deepcopy(row['causal_residual_receipt']) for row in (old, new)]
    for receipt in residuals:
        pending = receipt.pop('pending_forecast_tick')
        if pending is not None and pending != frame: raise ValueError('only the current forecast may be pending')
    if residuals[0] != residuals[1]: raise ValueError('unchanged executed residual history differs')
    transition = new['last_frontier_transition_receipt']
    reached = bool(transition and transition['reached_frontier_cell'] is not None)
    if transition is not None:
        if (transition['frame'] != frame or transition['measured_ns'] != 1_500_000_000+100_000_000*frame
                or transition['native_state_used'] is not False
                or transition['retirement_is_obstacle_evidence'] is not False
                or transition['retired_cells_remain_traversable'] is not True):
            raise ValueError('current observation-grounded frontier receipt required')
    normalized = deepcopy(new)
    normalized.pop('reached_frontier_transition_enabled'); normalized.pop('last_frontier_transition_receipt')
    normalized['controller'] = old['controller']
    exact = normalized == old
    if not frontier_previously_reached and not reached and not exact:
        raise ValueError('complete decisions must remain exact before first reached frontier')
    a, b = old['new_selection'], new['new_selection']
    forecasts = bool(a and b and 'prediction' in a and 'prediction' in b)
    if forecasts and a['prediction'] != b['prediction']:
        raise ValueError('unchanged model and causal inputs must produce exact raw forecasts')
    return dict(observed_and_executed_residual_state_exact=True, raw_model_forecasts_compared=forecasts,
        raw_model_forecasts_exact=forecasts, normalized_complete_decision_exact=exact,
        frontier_reached_this_frame=reached, requested_command_changed=new['requested_command'] != actual_command,
        terminal_changed=new['terminal'] != old['terminal'])

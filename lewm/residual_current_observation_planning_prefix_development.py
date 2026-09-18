"""Exact unchanged causal state at a current-planning-map intervention boundary."""
from copy import deepcopy
from lewm.residual_current_observation_planning_controller_development import METADATA

SHARED = ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt',
    'observed_goal_distance_m', 'auxiliary_floor_partition_receipt')


def compare_step(original, candidate, actual_command, *, frame):
    if (type(frame) is not int or not 0 <= frame < 3014
            or original['tick'] != frame or candidate['tick'] != frame
            or original['controller'] != 'residual_anchored_continuation_controller_v1'
            or candidate['controller'] != 'residual_current_observation_planning_controller_v1'):
        raise ValueError('bounded original residual-planner observation required')
    for key, value in METADATA.items():
        if candidate[key] != value or type(candidate[key]) is not type(value):
            raise ValueError('exact planning-map intervention metadata required: '+key)
    for key in SHARED:
        if original[key] != candidate[key]:
            raise ValueError('unchanged observed state differs: '+key)
    if original['requested_command'] != actual_command:
        raise ValueError('original request must match actual dispatched command')
    residuals = [deepcopy(row['causal_residual_receipt']) for row in (original, candidate)]
    for receipt in residuals:
        pending = receipt.pop('pending_forecast_tick')
        if pending is not None and pending != frame:
            raise ValueError('only the current forecast may be pending')
    if residuals[0] != residuals[1]:
        raise ValueError('executed causal residual history differs')
    old, new = original['new_selection'], candidate['new_selection']
    both_predict = bool(old and new and 'prediction' in old and 'prediction' in new)
    if both_predict and old['prediction'] != new['prediction']:
        raise ValueError('same model and actual history must give identical raw forecasts')
    if new is not None:
        view = new['planning_map_receipt']
        if (view['frame'] != frame or view['measured_ns'] != 1_500_000_000+frame*100_000_000
                or view['planning_map_variant'] != 'current_paired_observation'
                or view['accumulated_planning_cells_queried'] is not False
                or view['persistent_contact_history_retained'] is not True):
            raise ValueError('current complete paired planning-map receipt required')
    return dict(unchanged_observed_and_executed_residual_state_exact=True,
        raw_model_forecasts_compared=both_predict, raw_model_forecasts_exact=both_predict,
        requested_command_changed=candidate['requested_command'] != actual_command,
        terminal_changed=candidate['terminal'] != original['terminal'])

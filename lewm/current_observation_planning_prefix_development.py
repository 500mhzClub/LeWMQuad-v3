"""Compare unchanged causal state while permitting the declared planning-map intervention."""
from copy import deepcopy
from lewm.measured_floor_transport_prefix_development import normalize_labels

METADATA = dict(planning_map_variant='current_paired_observation',
    accumulated_planning_cells_queried=False, selector_scan_state_retained=True,
    persistent_contact_history_retained=True, tracking_and_floor_anchor_history_retained=True,
    learned_temporal_history_and_residual_retained=True, mission_and_settling_state_retained=True,
    memoryless_controller=False)
SHARED = ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt',
    'observed_goal_distance_m', 'auxiliary_floor_partition_receipt')


def compare_step(original, candidate, actual_command, *, frame):
    if type(frame) is not int or not 0 <= frame < 64 or candidate['tick'] != frame or original['tick'] != frame:
        raise ValueError('bounded current causal observation required')
    if candidate['controller'] != 'current_observation_planning_round_trip_controller_v1':
        raise ValueError('explicit current-observation planning successor required')
    normalized = deepcopy(candidate)
    for key,value in METADATA.items():
        if normalized.pop(key) != value or type(candidate[key]) is not type(value):
            raise ValueError('exact planning-map intervention metadata required: '+key)
    normalized['controller'] = 'measured_floor_transport_round_trip_controller_v1'
    normalized = normalize_labels(normalized)
    for key in SHARED:
        if normalized[key] != original[key]: raise ValueError('unchanged observed state differs: '+key)
    if original['requested_command'] != actual_command:
        raise ValueError('original decision must equal the actual dispatched command')
    residuals = [deepcopy(d['causal_residual_receipt']) for d in (normalized,original)]
    for receipt in residuals:
        # This field records whether the current choice was remembered. It may
        # differ at a terminal intervention; prior executed residuals may not.
        pending = receipt.pop('pending_forecast_tick')
        if pending is not None and pending != frame:
            raise ValueError('only the current forecast can be pending')
    if residuals[0] != residuals[1]: raise ValueError('executed causal residual history differs')
    old = original['new_selection']; new = candidate['new_selection']
    both_predict = bool(old and new and 'prediction' in old and 'prediction' in new)
    if both_predict and old['prediction'] != new['prediction']:
        raise ValueError('same model and actual history must give the same raw forecasts')
    if new is not None:
        view = new['planning_map_receipt']
        if (view['frame'] != frame or view['measured_ns'] != 1_500_000_000+frame*100_000_000
                or view['planning_map_variant'] != 'current_paired_observation'
                or view['accumulated_planning_cells_queried'] is not False
                or view['persistent_contact_history_retained'] is not True):
            raise ValueError('actual current planning-map receipt required')
    return dict(unchanged_observed_and_executed_residual_state_exact=True,
        raw_model_forecasts_compared=both_predict, raw_model_forecasts_exact=both_predict,
        requested_command_changed=candidate['requested_command'] != actual_command,
        terminal_changed=candidate['terminal'] != original['terminal'])

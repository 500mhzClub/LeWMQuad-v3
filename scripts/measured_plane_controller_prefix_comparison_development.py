"""Prospective comparison boundary for the measured-plane controller composition.

Reproduce the entire original controller decision and the authenticated new
observer evidence. Permit changed estimates/scores only until the first changed
requested command or terminal result. Never consume its unexecuted outcome.
"""
import json

from lewm.geometry_progress_pilot_development import candidate_commands

ORIGINAL = 'residual_anchored_continuation_controller_v1'
CANDIDATE = 'measured_plane_residual_continuation_controller_v1'
MAX_FRAMES = 3838


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compare(original, candidate, recorded, observer, *, frame):
    if type(frame) is not int or not 0 <= frame < MAX_FRAMES:
        raise ValueError('bounded actual controller frame required')
    if canonical(original) != canonical(recorded):
        raise ValueError('complete original controller decision must reproduce')
    if (original['controller'] != ORIGINAL or candidate['controller'] != CANDIDATE
            or candidate['measured_plane_constrained_estimator'] is not True):
        raise ValueError('exact original and prospective controller definitions required')
    if (canonical(candidate['original_visual_evidence']) != canonical(observer['candidate'])
            or canonical(original['original_visual_evidence']) != canonical(observer['original'])):
        raise ValueError('complete authenticated paired observer history required')
    if canonical(candidate['evidence']) != canonical(observer['candidate_floor']):
        raise ValueError('unchanged candidate floor registration must reproduce observer proof')
    if original['terminal'] is None and original['tick'] != frame:
        raise ValueError('current original nonterminal controller frame required')
    if candidate['terminal'] is None:
        if candidate['failure'] is not None or candidate['tick'] != frame:
            raise ValueError('current admitted candidate controller frame required')
        selection = candidate['new_selection']
        if frame >= 3:
            if selection is None:
                mission = candidate['mission_receipt']
                if (mission['frame'] != frame or mission['hold_required'] is not True
                        or candidate['requested_command'] != [0.,0.,0.]):
                    raise ValueError('only an explicit current mission hold may omit a forecast')
            elif 'prediction' not in selection:
                raise ValueError('full trained action-conditioned forecast selection required')
            else:
                action = selection['action']
                requested = [0.,0.,0.] if action is None else list(candidate_commands(action)[0])
                if candidate['requested_command'] != requested:
                    raise ValueError('candidate command must follow its actual selected action')
    elif candidate['requested_command'] != [0.,0.,0.]:
        raise ValueError('candidate terminal must retain a zero command')
    a, b = original['new_selection'], candidate['new_selection']
    forecast_compared = bool(a and b and 'prediction' in a and 'prediction' in b)
    if forecast_compared:
        for name in ('prediction','head','input_variant','target_offsets_ns',
                'model_prediction_corrected','translation_bias_training_only','translation_bias_xy_m'):
            if canonical(a[name]) != canonical(b[name]):
                raise ValueError('same frozen model and public history must preserve raw forecasts: '+name)
    command_changed = canonical(original['requested_command']) != canonical(candidate['requested_command'])
    terminal_changed = original['terminal'] != candidate['terminal']
    stop = command_changed or terminal_changed or candidate['terminal'] is not None or frame == MAX_FRAMES-1
    return dict(frame=frame, complete_original_decision_exact=True,
        complete_candidate_observer_and_floor_exact=True, original_forecast_compared=forecast_compared,
        requested_command_changed=command_changed, terminal_changed=terminal_changed, stop=stop,
        stop_reason='FIRST_CHANGED_REQUEST_OR_TERMINAL' if command_changed or terminal_changed else
            'CANDIDATE_TERMINAL' if candidate['terminal'] is not None else
            'FIXED_OBSERVER_HISTORY_END' if frame == MAX_FRAMES-1 else None,
        changed_command_executed=False, following_unexecuted_outcome_consumed=False,
        navigation_recovered=False, model_or_memory_advantage_established=False)

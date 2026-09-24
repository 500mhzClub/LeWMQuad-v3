"""Full original-decision reproduction and first changed command boundary."""
import json

from lewm.geometry_progress_pilot_development import candidate_commands

ORIGINAL = 'measured_plane_residual_continuation_controller_v1'
CANDIDATE = 'measured_plane_chained_anchor_controller_v1'
FORECAST_FIELDS = ('prediction', 'head', 'input_variant', 'target_offsets_ns',
    'model_prediction_corrected', 'translation_bias_training_only', 'translation_bias_xy_m')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compare(original, candidate, recorded, *, frame, maximum_frames, model_calls):
    if (type(frame) is not int or type(maximum_frames) is not int
            or not 4 <= maximum_frames <= 4014 or not 0 <= frame < maximum_frames):
        raise ValueError('bounded current native controller observation required')
    if canonical(original) != canonical(recorded):
        raise ValueError('complete original sensor-to-command decision must reproduce')
    if (original['controller'] != ORIGINAL or candidate['controller'] != CANDIDATE
            or original['measured_plane_constrained_estimator'] is not True
            or candidate['measured_plane_constrained_estimator'] is not True
            or candidate['chained_anchor_reacquisition_enabled'] is not True
            or candidate['direct_corner_flow_missingness_fallback_enabled'] is not True):
        raise ValueError('exact original and combined candidate definitions required')
    if type(model_calls) is not list or len(model_calls) != 2 or any(type(v) is not int or v not in (0,1) for v in model_calls):
        raise ValueError('actual bounded per-controller model forward counts required')
    forecasts = []
    for index, decision in enumerate((original, candidate)):
        selection = decision['new_selection']
        forecast = bool(selection and 'prediction' in selection)
        if (forecast and model_calls[index] != 1
                or decision['terminal'] is None and forecast != bool(model_calls[index])):
            raise ValueError('saved forecast must match actual model invocation')
        forecasts.append(forecast)
        if decision['terminal'] is None:
            if decision['tick'] != frame or decision['failure'] is not None:
                raise ValueError('current nonterminal controller evidence required')
            if frame >= 3:
                if selection is None:
                    mission = decision['mission_receipt']
                    if (mission['frame'] != frame or mission['hold_required'] is not True
                            or decision['requested_command'] != [0.,0.,0.]):
                        raise ValueError('only current mission hold can omit a selection')
                elif not forecast:
                    raise ValueError('full learned forecast required for a selection')
                else:
                    action = selection['action']
                    command = [0.,0.,0.] if action is None else list(candidate_commands(action)[0])
                    if decision['requested_command'] != command:
                        raise ValueError('requested command must follow its selected action')
        elif decision['requested_command'] != [0.,0.,0.]:
            raise ValueError('terminal controller must retain zero command')
    common_forecast = all(forecasts)
    if common_forecast:
        for field in FORECAST_FIELDS:
            if canonical(original['new_selection'][field]) != canonical(candidate['new_selection'][field]):
                raise ValueError('unchanged public model history must preserve forecast: '+field)
    command_changed = original['requested_command'] != candidate['requested_command']
    terminal_changed = original['terminal'] != candidate['terminal']
    terminal = original['terminal'] is not None or candidate['terminal'] is not None
    reason = ('FIRST_CHANGED_REQUEST_OR_TERMINAL' if command_changed or terminal_changed else
        'MATCHED_TERMINAL' if terminal else 'RECORDED_HISTORY_END' if frame == maximum_frames-1 else None)
    visual = candidate.get('original_visual_evidence') or {}
    chained = visual.get('chained_anchor_fallback') or {}
    return dict(frame=frame, complete_original_decision_exact=True,
        actual_model_forward_calls=model_calls, original_forecast_compared=common_forecast,
        requested_command_changed=command_changed, terminal_changed=terminal_changed,
        candidate_anchor_reacquisition_recorded=chained.get('accepted') is True,
        candidate_terminal=candidate['terminal'], original_terminal=original['terminal'],
        stop=reason is not None, stop_reason=reason, changed_command_executed=False,
        following_changed_command_outcome_consumed=False, navigation_recovered=False)

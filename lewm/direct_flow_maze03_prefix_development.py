"""Exact original decisions until a fixed, previously recorded tracking failure."""
from copy import deepcopy
from lewm.geometry_progress_pilot_development import candidate_commands

BOUNDARY_FRAME = 264
MAX_FRAMES = BOUNDARY_FRAME+1


def compare_step(original, candidate, actual_command, *, frame):
    if (type(frame) is not int or not 0 <= frame < MAX_FRAMES
            or original['controller'] != 'measured_floor_transport_round_trip_controller_v1'
            or candidate['controller'] != 'direct_flow_floor_transport_controller_v1'
            or candidate['direct_corner_flow_missingness_fallback_enabled'] is not True
            or original['requested_command'] != actual_command):
        raise ValueError('fixed original episode and explicit successor required')
    normalized=deepcopy(candidate)
    normalized.pop('direct_corner_flow_missingness_fallback_enabled')
    normalized['controller']=original['controller']
    raw=candidate['original_visual_evidence']
    fallback=(raw or {}).get('direct_corner_flow_fallback')
    if frame < BOUNDARY_FRAME:
        if (original['tick'] != frame or candidate['tick'] != frame
                or original['terminal'] is not None or candidate['failure'] is not None
                or fallback is not None or normalized != original):
            raise ValueError('complete original decision and current state must match before failure')
        return dict(complete_original_decision_exact=True,
            raw_model_forecasts_compared=bool(original['new_selection'] and 'prediction' in original['new_selection']),
            boundary_reached=False, fallback_attempted=False, controller_recovered=False,
            requested_command_changed=False, terminal_changed=False, stop=False)
    prior=original['original_visual_evidence']
    now=1_500_000_000+frame*100_000_000
    if (original['tick'] != frame-1 or original['terminal'] != 'SENSOR_OR_MODEL_FAILURE'
            or prior['status'] != 'VISUAL_TERMINAL_FAILURE' or prior['decision_ns'] != now
            or raw['decision_ns'] != now):
        raise ValueError('exact recorded original tracking-failure boundary required')
    if fallback is None:
        if normalized != original:
            raise ValueError('without fallback the complete original failure must be reproduced')
    elif (fallback['frame'] != frame or fallback['measured_ns'] != now
            or fallback['original_camera_selection'] != prior['camera_selection']
            or fallback['original_auxiliary_continuity'] != prior['continuity_evidence']
            or fallback['original_reference_selection'] != prior['reference_selection']
            or fallback['association_rule_changed'] is not True
            or fallback['rigid_geometry_thresholds_unchanged'] is not True
            or fallback['temporal_continuity_thresholds_unchanged'] is not True
            or fallback['bridge_budget_unchanged'] is not True
            or fallback['reference_or_pose_history_reset'] is not False):
        raise ValueError('original failure evidence and explicit bounded association change required')
    recovered=candidate['failure'] is None and candidate['terminal'] is None
    if recovered:
        if (fallback is None or fallback['accepted'] is not True
                or candidate['tick'] != frame or raw['status'] != 'CURRENT_VISUAL_POSE'
                or raw['current_pose']['frame'] != frame or candidate['evidence'] is None
                or not candidate['new_selection'] or 'prediction' not in candidate['new_selection']):
            raise ValueError('current admitted pose, mapping and complete forecast selection required')
        action=candidate['new_selection']['action']
        expected=[0.,0.,0.] if action is None else candidate_commands(action)[0]
        if candidate['requested_command'] != expected:
            raise ValueError('new request must match current selected action')
    elif candidate['terminal'] is None or candidate['requested_command'] != [0.,0.,0.]:
        raise ValueError('unsuccessful boundary must retain a terminal zero command')
    return dict(complete_original_decision_exact=normalized == original,
        raw_model_forecasts_compared=False, boundary_reached=True,
        fallback_attempted=fallback is not None, controller_recovered=recovered,
        requested_command_changed=candidate['requested_command'] != actual_command,
        terminal_changed=candidate['terminal'] != original['terminal'], stop=True)

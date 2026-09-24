"""Compare a complete controller prefix at an authenticated anchor intervention.

The original observer is still live at this boundary. A candidate that admits
an anchored pose is not labelled recovery from an original controller failure.
"""
from lewm.direct_flow_residual_anchored_controller_development import CONTROLLER as ORIGINAL, FLAG as ORIGINAL_FLAG
from lewm.chained_anchor_residual_controller_development import CONTROLLER, FLAG
from lewm.geometry_progress_pilot_development import candidate_commands


def compare(original, candidate, actual, expected_visual, *, frame, boundary):
    if (type(frame) is not int or type(boundary) is not int or not 3 <= boundary < 864
            or not 0 <= frame <= boundary or original['controller'] != ORIGINAL
            or original.get(ORIGINAL_FLAG) is not True or candidate['controller'] != CONTROLLER
            or candidate.get(FLAG) is not True or candidate.get(ORIGINAL_FLAG) is not True
            or original['requested_command'] != actual):
        raise ValueError('exact original controller, executed command and bounded intervention required')
    if candidate['original_visual_evidence'] != expected_visual:
        raise ValueError('complete authenticated observer evidence required')
    if original['tick'] != frame or original['terminal'] is not None or original['failure'] is not None:
        raise ValueError('original live controller prefix must remain intact')
    normalized = dict(candidate)
    normalized.pop(FLAG)
    normalized['controller'] = ORIGINAL
    if frame < boundary:
        if normalized != original:
            raise ValueError('complete original decision and forecast must match before intervention')
        return dict(stop=False, complete_original_decision_exact=True,
            original_forecast_compared=bool(original['new_selection'] and 'prediction' in original['new_selection']),
            controller_admitted_reacquired_pose=False, requested_command_changed=False,
            original_controller_failed=False, navigation_recovered=False)
    old = original['original_visual_evidence']
    raw = candidate['original_visual_evidence']
    fallback = raw.get('chained_anchor_fallback') or {}
    if (old['status'] != 'CURRENT_VISUAL_POSE'
            or old['continuity_evidence']['status'] != 'MEASURED_INCREMENT_BRIDGE'
            or fallback.get('accepted') is not True
            or fallback['original_bridge_available'] is not True
            or fallback['original_continuity'] != old['continuity_evidence']
            or fallback['original_camera_selection'] != old['camera_selection']
            or fallback['original_direct_flow_fallback'] != old.get('direct_corner_flow_fallback')
            or raw['continuity_evidence']['status'] != 'ANCHOR_MEASUREMENT'):
        raise ValueError('exact original bridge and authenticated anchor reacquisition required')
    admitted = candidate['terminal'] is None and candidate['failure'] is None
    if admitted:
        selection = candidate['new_selection']
        if (candidate['tick'] != frame or candidate['evidence'] is None
                or raw['status'] != 'CURRENT_VISUAL_POSE' or not selection or 'prediction' not in selection):
            raise ValueError('registered current pose and full forecast selection required')
        action = selection['action']
        requested = [0., 0., 0.] if action is None else list(candidate_commands(action)[0])
        if candidate['requested_command'] != requested:
            raise ValueError('candidate request must follow its selected action')
    elif candidate['terminal'] is None or candidate['requested_command'] != [0., 0., 0.]:
        raise ValueError('rejected candidate must preserve a terminal zero command')
    return dict(stop=True, complete_original_decision_exact=False, original_forecast_compared=False,
        controller_admitted_reacquired_pose=admitted, requested_command_changed=candidate['requested_command'] != actual,
        original_controller_failed=False, navigation_recovered=False)

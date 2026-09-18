"""Exact tracking episode through its first floor rejection; no following tape."""
from copy import deepcopy
import json
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.partial_floor_height_development import SCHEMA, CONFLICT, current_partial_height_pose
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose

BOUNDARY_FRAME = 504
MAX_FRAMES = BOUNDARY_FRAME+1


def compare_step(original, candidate, actual_command, *, frame):
    if (type(frame) is not int or not 0 <= frame < MAX_FRAMES
            or original['controller'] != 'direct_flow_floor_transport_controller_v1'
            or candidate['controller'] != 'partial_height_direct_flow_controller_v1'
            or candidate['partial_floor_height_constraint_enabled'] is not True
            or original['requested_command'] != actual_command):
        raise ValueError('fixed original tracking episode and explicit height successor required')
    normalized = deepcopy(candidate)
    normalized.pop('partial_floor_height_constraint_enabled')
    normalized['controller'] = original['controller']
    if candidate['original_visual_evidence'] != original['original_visual_evidence']:
        raise ValueError('complete unchanged raw visual tracker output required')
    if frame < BOUNDARY_FRAME:
        if (original['tick'] != frame or original['terminal'] is not None
                or original['failure'] is not None or normalized != original):
            raise ValueError('every complete decision must match before the original floor failure')
        return dict(complete_original_decision_exact=True, boundary_reached=False, stop=False,
            raw_model_forecasts_compared=bool(original['new_selection'] and 'prediction' in original['new_selection']),
            controller_recovered=False, partial_height_admitted=False, requested_command_changed=False)
    now = 1_500_000_000+frame*100_000_000
    raw = original['original_visual_evidence']
    if (original['tick'] != frame-1 or original['terminal'] != 'SENSOR_OR_MODEL_FAILURE'
            or original['failure'] != CONFLICT or original['evidence'] is not None
            or actual_command != [0., 0., 0.] or raw['decision_ns'] != now
            or raw['current_pose']['frame'] != frame):
        raise ValueError('exact first recorded floor conflict required')
    evidence = candidate['evidence']
    admitted = evidence is not None and evidence['schema'] == SCHEMA
    recovered = candidate['terminal'] is None and candidate['failure'] is None
    if admitted:
        if (evidence['original_visual_evidence'] != raw or evidence['decision_ns'] != now
                or evidence['current_pose']['frame'] != frame):
            raise ValueError('same current raw visual witness must underlie the partial pose')
    if recovered:
        selection = candidate['new_selection']
        if (not admitted or candidate['tick'] != frame or not selection or 'prediction' not in selection):
            raise ValueError('full current pose and fresh learned selection required for recovery')
        action = selection['action']
        expected = [0., 0., 0.] if action is None else list(candidate_commands(action)[0])
        if candidate['requested_command'] != expected:
            raise ValueError('current selected action must determine the requested command')
    elif candidate['terminal'] is None or candidate['requested_command'] != [0., 0., 0.]:
        raise ValueError('unsuccessful boundary must retain terminal zero command')
    return dict(complete_original_decision_exact=normalized == original, boundary_reached=True, stop=True,
        raw_model_forecasts_compared=False, controller_recovered=recovered, partial_height_admitted=admitted,
        requested_command_changed=candidate['requested_command'] != actual_command)


def validate_live(live, recorded, comparison, policy, image, auxiliary, *, now_ns, prior_anchor):
    if json.loads(json.dumps(live, allow_nan=False)) != recorded:
        raise ValueError('exact serialization of live decision required')
    if comparison['boundary_reached']:
        current_dual_camera_pose(live['original_visual_evidence'], policy, image, auxiliary,
            identity=(0, 0, 0), now_ns=now_ns)
    if comparison['partial_height_admitted']:
        current_partial_height_pose(live['evidence'], identity=(0, 0, 0), now_ns=now_ns)
        if live['evidence']['partial_floor_height']['anchor'] != prior_anchor:
            raise ValueError('partial observation must retain the original preceding full anchor')

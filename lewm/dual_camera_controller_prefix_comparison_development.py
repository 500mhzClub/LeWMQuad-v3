"""Exact settled-controller prefix outside explicitly validated modality metadata."""
from copy import deepcopy
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
from lewm.joint_floor_registered_evidence_development import current_joint_floor_registered_pose


def compare_primary_decision(original, candidate, policy, image, auxiliary, *, now_ns):
    if (original['controller'] != 'settled_boundary_round_trip_controller_v1'
            or candidate['controller'] != 'dual_camera_settled_round_trip_controller_v1'
            or candidate['additional_auxiliary_rgb_for_motion'] is not True
            or original['terminal'] is not None or candidate['terminal'] is not None):
        raise ValueError('active original and declared dual-camera settled controllers required')
    raw = candidate['original_visual_evidence']
    _, _, pose = current_dual_camera_pose(raw, policy, image, auxiliary, identity=(0,0,0), now_ns=now_ns)
    registered = candidate['evidence']
    current_joint_floor_registered_pose(registered, identity=(0,0,0), now_ns=now_ns)
    if registered['original_visual_evidence'] != raw or raw['last_visual'] != raw['current_pose']:
        raise ValueError('same current raw evidence across every registered consumer required')
    expected_choice = (dict(selected_camera=None, auxiliary_attempted=False, initial_paired_reference=True)
        if pose['frame'] == 0 else dict(selected_camera='primary', auxiliary_attempted=False,
            primary_failure=None, primary_continuity=None, primary_reference_selection=None,
            cross_camera_disagreement_m=None, cross_camera_disagreement_rad=None,
            measurements_independent=False, thresholds_unchanged=True))
    if raw['camera_selection'] != expected_choice:
        raise ValueError('comparison stops before any auxiliary camera intervention')

    def strip_pose(value):
        for key in ('auxiliary_rgb_sha256','auxiliary_depth_sha256'):
            if value.pop(key) != pose[key]: raise ValueError('current auxiliary pose bindings differ')

    def strip_motion(value):
        # In-memory snapshots may share current_pose and last_visual; decoded
        # JSON does not. Normalize each explicit pose path independently.
        value['current_pose'] = deepcopy(value['current_pose'])
        value['last_visual'] = deepcopy(value['last_visual'])
        if value.pop('auxiliary_rgb_is_additional_modality') is not True:
            raise ValueError('explicit added auxiliary RGB modality required')
        if value.pop('camera_selection_current') is not True or value.pop('camera_selection') != expected_choice:
            raise ValueError('current original primary selection required')
        value.pop('auxiliary_feature_witness')
        if value['observer_variant'] != 'front_first_dual_camera_anchor_v1':
            raise ValueError('declared dual-camera observer required')
        value['observer_variant'] = 'accepted_half_feature_overlap_v1'
        for key, expected in [('auxiliary_rgb', image['calibration_id']), ('auxiliary_depth', auxiliary['calibration_id'])]:
            if value['calibration_ids'].pop(key) != expected: raise ValueError('fixed added calibration required')
        strip_pose(value['current_pose']); strip_pose(value['last_visual'])

    normalized = deepcopy(candidate)
    normalized['original_visual_evidence'] = deepcopy(candidate['original_visual_evidence'])
    normalized['evidence'] = deepcopy(candidate['evidence'])
    normalized.pop('additional_auxiliary_rgb_for_motion')
    normalized['controller'] = original['controller']
    strip_motion(normalized['original_visual_evidence'])
    strip_motion(normalized['evidence']['original_visual_evidence'])
    strip_pose(normalized['evidence']['current_pose'])
    if normalized != original:
        raise ValueError('complete primary controller decision differs outside declared auxiliary metadata')
    return dict(complete_decision_exact_outside_added_auxiliary_metadata=True,
        requested_command_exact=True, selected_camera='primary' if pose['frame'] else None,
        no_auxiliary_intervention=True, frame=pose['frame'])

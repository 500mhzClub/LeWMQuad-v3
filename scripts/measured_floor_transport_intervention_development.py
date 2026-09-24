"""Require the complete frozen causal prefix and its exact first transport witness."""
from lewm.joint_floor_registered_evidence_development import SCHEMA as FLOOR_SCHEMA
from lewm.measured_floor_transport_development import SCHEMA as TRANSPORT_SCHEMA
from lewm.measured_floor_transport_json_development import current_json_pose
from lewm.measured_floor_transport_prefix_development import normalize_labels
from scripts.maze_decision_stream_development import read_rows
from scripts.startup_raw_sensor_audit_development import read_json


def prefix_shape(report):
    if (report['status'] != 'MEASURED_FLOOR_TRANSPORT_PREFIX_COMPLETE'
            or type(report['frames']) is not int or report['frames'] != 1905
            or type(report['first_intervention_frame']) is not int or report['first_intervention_frame'] != 1904):
        raise ValueError('complete fixed1905-frame floor-transport prefix required')
    for key in ('complete_preintervention_decisions_exact_outside_validated_labels',
            'all_1904_prior_actual_commands_exact', 'raw_visual_evidence_exact_at_intervention',
            'current_transport_pose_available', 'active_return_at_intervention', 'model_state_unchanged',
            'public_input_arrays_unchanged', 'diagnosis_bindings_match_final_native_artifacts'):
        if report[key] is not True: raise ValueError('completed prefix invariant required: '+key)
    if report['following_recorded_observations_consumed'] is not False or report['native_execution'] is not False:
        raise ValueError('causal replay without following observations or native execution required')
    return 1905, 1904


def admit_intervention(directory, report):
    frames, changed = prefix_shape(report)
    witness = read_json(directory, 'intervention.json')
    if (witness['frame'] != changed or witness['current_pose_validated'] is not True
            or witness['first_intervention'] is not True or witness['following_recorded_observations_consumed'] is not False):
        raise ValueError('explicit first transported pose without following observations required')
    candidate = witness['candidate_decision']; original = witness['original_failed_decision']
    normalize_labels(candidate)
    if (candidate['terminal'] is not None or candidate['failure'] is not None
            or candidate['mission_receipt']['phase'] != 'RETURN'
            or candidate['requested_command'] != report['final_requested_command']
            or original['terminal'] != 'SENSOR_OR_MODEL_FAILURE' or original['evidence'] is not None
            or original['failure'] != 'admitted combined measured moments must reconstruct exactly'
            or candidate['original_visual_evidence'] != original['original_visual_evidence']):
        raise ValueError('same measured visual input and active RETURN intervention required')
    evidence = candidate['evidence']; now = 1_500_000_000+changed*100_000_000
    _, _, pose = current_json_pose(evidence, now_ns=now)
    if (evidence['schema'] != TRANSPORT_SCHEMA or pose['frame'] != changed
            or evidence['floor_transport']['correction']['anchor_frame'] != changed-1
            or evidence['floor_transport']['anchor']['current_pose']['frame'] != changed-1):
        raise ValueError('current transport from the last admitted1903 floor anchor required')
    count = 0; previous = None
    for i, row in enumerate(read_rows(directory)):
        if i >= frames or row['tick'] != i: raise ValueError('complete ordered bounded prefix stream required')
        decision = row['decision']; normalize_labels(decision)
        if (row['input_arrays_unchanged'] is not True or decision['terminal'] is not None or decision['failure'] is not None
                or decision['evidence']['current_pose']['frame'] != i):
            raise ValueError('every saved prefix decision requires active current evidence and unchanged inputs')
        if i < changed:
            if row['preintervention_complete_decision_exact'] is not True or decision['evidence']['schema'] != FLOOR_SCHEMA:
                raise ValueError('no earlier transport or changed decision is permitted')
            previous = decision
        elif (row['preintervention_complete_decision_exact'] is not False or decision != candidate
                or previous['evidence'] != evidence['floor_transport']['anchor']):
            raise ValueError('intervention must match complete stream and immediately preceding floor anchor')
        count += 1
    if count != frames: raise ValueError('complete1905-frame stream required')
    return dict(frames=count, first_intervention_frame=changed,
        saved_intervention_matches_summary_and_stream=True, anchor_matches_last_admitted_floor_pose=True,
        current_transport_pose_available=True, active_return_at_intervention=True,
        following_recorded_observations_consumed=False)

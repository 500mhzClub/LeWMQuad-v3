"""Fixed development-layout cohort and admission of completed native evidence."""
from copy import deepcopy

LAYOUTS = (1, 2, 3)
MODEL_STATE = '4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6'


def planned_cases(predecessor_case):
    name, index, variant, condition, model_name = predecessor_case
    if index != 0 or name != 'full_jepa_novel_maze_00':
        raise ValueError('fixed maze0 predecessor required')
    return [(f'full_jepa_novel_maze_{i:02d}', i, variant, condition, model_name) for i in LAYOUTS]


def admit_predecessor(result, audit, launch, predecessor_case):
    """Infrastructure validity is required; scientific failures remain failures."""
    if (result['status'] != 'MEASURED_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE'
            or len(result['conditions']) != 1):
        raise ValueError('completed floor-transport native result required')
    record = result['conditions'][0]
    if (record['case'] != predecessor_case[0] or record['layout_index'] != 0
            or record['status'] != 'MEASURED_FLOOR_TRANSPORT_MAZE_COLLECTED_AND_RAW_AUDITED'
            or launch['planned_case'] != list(predecessor_case)):
        raise ValueError('exact completed maze0 case required')
    for key in ('physical_and_public_prefix_exact', 'complete_candidate_decisions_match_prospective_prefix'):
        if record['prefix_comparison'][key] is not True:
            raise ValueError('prospective intervention evidence required: '+key)
    for key in ('raw_sensor_reconstruction_pass', 'raw_model_command_replay_pass',
            'raw_command_audit_pass', 'model_state_unchanged'):
        if audit[key] is not True:
            raise ValueError('complete native raw audit required: '+key)
    if (record['model_state_unchanged'] is not True
            or launch['prefix_report']['model_state_sha256'] != MODEL_STATE
            or launch['implementation_class'] != 'MeasuredFloorTransportController'
            or launch['renderer_capture_witnesses_enabled'] is not True
            or launch['additional_auxiliary_rgb_for_motion'] is not True):
        raise ValueError('unchanged assigned model, controller and acquisition required')
    for key in ('verified_round_trip', 'native_evaluation', 'strict_physical_visibility_pass',
            'hard_measurement_failed_frames', 'renderer_capture_audit'):
        if record[key] != audit[key]:
            raise ValueError('native result and saved audit disagree: '+key)
    return dict(predecessor_verified_round_trip=record['verified_round_trip'],
        predecessor_strict_physical_visibility_pass=record['strict_physical_visibility_pass'],
        model_state_sha256=MODEL_STATE, planned_cases=planned_cases(predecessor_case),
        predecessor_success_required=False, controller_selected_using_new_layout_outcomes=False)


def independent_scope(report, index):
    """Reuse physical calculations verbatim; declare the new experiment's scope."""
    if type(index) is not int or index not in LAYOUTS or report['layout_index'] != index:
        raise ValueError('one of the three fixed independent layouts required')
    result = deepcopy(report)
    result.update(independent_layout_development_execution=True, reused_development_layout=False)
    return result


def require_resources(resources, remaining_cases, *, reserve, collection, persistence):
    if type(remaining_cases) is not int or not 1 <= remaining_cases <= len(LAYOUTS):
        raise ValueError('remaining fixed cohort size required')
    required = reserve + remaining_cases * (collection + persistence)
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < required:
        raise ValueError('remaining independent cohort resource allowance unavailable')
    return dict(memory_admission_bytes=32*1024**3, required_free_bytes=required,
        remaining_cases=remaining_cases, native_scene_workers=1, os_resource_limits_enforced=False)

"""Predeclared reactive comparison on the same three independent layouts."""
from copy import deepcopy
from lewm.independent_floor_transport_study_development import LAYOUTS, MODEL_STATE

MATCHED_KEYS = ('robot_urdf_sha256', 'navigation_ticks', 'native_scene_workers',
    'opencv_threads', 'blas_threads', 'maximum_tasks_per_process',
    'physics_paused_during_compute', 'renderer_environment',
    'renderer_capture_witnesses_enabled', 'input_sha256', 'native_sha256',
    'native_scene_sha256', 'native_geometry_sha256', 'opencv_binary_sha256',
    'opencv_version', 'rules')
OUTCOME_KEYS = ('verified_round_trip', 'native_evaluation',
    'strict_physical_visibility_pass', 'hard_measurement_failed_frames', 'renderer_capture_audit')


def planned_cases():
    return [(f'reactive_floor_transport_novel_maze_{i:02d}', i) for i in LAYOUTS]


def merge_sources(*manifests):
    merged = {}
    for manifest in manifests:
        for name, sha in manifest.items():
            if name in merged and merged[name] != sha:
                raise ValueError('conflicting frozen source: '+name)
            merged[name] = sha
    return merged


def require_raw_audit(record, audit, *, learned):
    flags = ['raw_sensor_reconstruction_pass', 'raw_command_audit_pass',
        'raw_model_command_replay_pass' if learned else 'raw_controller_command_replay_pass']
    if learned: flags.append('model_state_unchanged')
    for key in flags:
        if audit[key] is not True: raise ValueError('completed raw audit required: '+key)
    if not learned and audit['high_level_world_model_used'] is not False:
        raise ValueError('reactive high-level model must be absent')
    for key in OUTCOME_KEYS:
        if record[key] != audit[key]: raise ValueError('saved outcome disagrees: '+key)


def admit_inputs(learned, learned_launch, learned_audits, reactive, reactive_launch, reactive_audit):
    if (learned['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_MAZES_V1_COMPLETE'
            or learned['all_fixed_cases_executed'] is not True
            or learned['original_case_order'] != list(LAYOUTS)
            or len(learned['conditions']) != len(LAYOUTS)
            or len(learned_audits) != len(LAYOUTS)
            or learned_launch['case_order'] != list(LAYOUTS)
            or learned_launch['model_state_sha256'] != MODEL_STATE
            or learned_launch['implementation_class'] != 'MeasuredFloorTransportController'
            or learned_launch['controller_or_model_changes_between_cases'] is not False
            or learned_launch['fresh_controller_and_memory_per_case'] is not True):
        raise ValueError('completed unchanged all-three-layout learned cohort required')
    cases = learned_launch['planned_cases']
    if len(cases) != len(LAYOUTS): raise ValueError('all planned learned cases required')
    for index, record, audit, case in zip(LAYOUTS, learned['conditions'], learned_audits, cases, strict=True):
        if (record['case'] != f'full_jepa_novel_maze_{index:02d}'
                or type(record['layout_index']) is not int or record['layout_index'] != index
                or audit['layout_index'] != index or case[:2] != [record['case'], index]
                or record['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
                or record['model_state_unchanged'] is not True):
            raise ValueError('ordered learned case and audit identity required')
        require_raw_audit(record, audit, learned=True)
    if (reactive['status'] != 'REACTIVE_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE'
            or len(reactive['conditions']) != 1
            or reactive_launch['implementation_class'] != 'ReactiveFloorTransportController'
            or reactive_launch['high_level_world_model_loaded'] is not False
            or reactive_launch['candidate_future_outcomes_evaluated'] is not False
            or reactive_launch['learned_residual_used'] is not False):
        raise ValueError('completed current reactive maze0 pilot required')
    record = reactive['conditions'][0]
    if (record['case'] != 'reactive_floor_transport_novel_maze_00' or record['layout_index'] != 0
            or reactive_audit['layout_index'] != 0
            or record['status'] != 'REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'):
        raise ValueError('exact current reactive pilot identity required')
    require_raw_audit(record, reactive_audit, learned=False)
    prefix = record['prefix_comparison']
    for key in ('physical_and_public_prefix_exact', 'shared_observed_state_exact',
            'all_preintervention_requested_commands_exact', 'complete_candidate_decisions_match_prospective_prefix'):
        if prefix[key] is not True: raise ValueError('reactive pilot prefix required: '+key)
    if prefix['common_prefix_frames'] != 4 or prefix['physical_prefix_samples'] != 900:
        raise ValueError('complete reactive pilot common prefix required')
    if not (learned['predecessor_result_sha256'] == reactive['learned_result_sha256']
            == reactive_launch['learned_result_sha256']):
        raise ValueError('same frozen maze0 learned predecessor required')
    for key in MATCHED_KEYS:
        if learned_launch[key] != reactive_launch[key]:
            raise ValueError('learned/reactive environment or budget differs: '+key)
    return dict(learned_layouts=list(LAYOUTS), reactive_layouts=list(LAYOUTS),
        matched_launch_fields=list(MATCHED_KEYS),
        learned_successes=sum(int(r['verified_round_trip']) for r in learned['conditions']),
        reactive_pilot_verified_round_trip=record['verified_round_trip'],
        predecessor_scientific_success_required=False, outcome_based_case_selection=False,
        isolated_prediction_ranking_ablation=False, predictive_feasibility_gates_matched=False)


def paired_outcomes(learned_records, reactive_records):
    if len(learned_records) != 3 or len(reactive_records) != 3:
        raise ValueError('both complete fixed cohorts required')
    pairs = []
    for index, old, new in zip(LAYOUTS, learned_records, reactive_records, strict=True):
        if (old['layout_index'] != index or new['layout_index'] != index
                or old['case'] != f'full_jepa_novel_maze_{index:02d}'
                or new['case'] != f'reactive_floor_transport_novel_maze_{index:02d}'
                or old['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
                or new['status'] != 'INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'):
            raise ValueError('ordered complete paired executions required')
        pairs.append(dict(layout_index=index, learned=deepcopy({k:old[k] for k in OUTCOME_KEYS}),
            reactive=deepcopy({k:new[k] for k in OUTCOME_KEYS})))
    return pairs

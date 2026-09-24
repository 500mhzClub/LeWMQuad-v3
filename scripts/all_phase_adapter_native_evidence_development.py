"""Bind the completed interface failures and exact six-model adapter replay."""
from scripts import run_go2_all_phase_residual_maze02_matched_native_v1 as original
from scripts import await_go2_all_phase_residual_maze02_native_v1 as waiter
from scripts import replay_go2_all_phase_planner_adapter_startup_v2 as prefix
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from lewm.independent_reactive_floor_transport_study_development import merge_sources

ORIGINAL_SHA = 'a08496e1d62ec6e00ffae3d85729cc7cf069c2fddcb60d421910c83d30556e80'
ORIGINAL_LAUNCH = '8260991892f61dd36d75ac001e2198215b19a2531f535bfc381f86c6282c46eb'
WAIT_SHA = '68b965638210893d5b3cd446eac70d432b7b9d9afd130c2ee64e4d9966b70da2'
PREFIX_SHA = '0d006a18920ba733f46f79fc27444c72038de8da3a4df08ff5af343c8f965daa'
PREFIX_LAUNCH = '5faf990e3a695a9a32dde7f161cac0a48230f16970cc3ceb419a199bcdac0567'


def prepared_sources(seeds):
    inherited = {}
    for root, sha in ((original.OUTPUT, ORIGINAL_SHA), (waiter.OUTPUT, WAIT_SHA), (prefix.OUTPUT, PREFIX_SHA)):
        verify_artifacts(root, {'result.json': sha})
        inherited = merge_sources(inherited, read_json(root, 'result.json')['source_sha256'])
    verify(inherited)
    sources = discover_sources(seeds, inherited); verify(sources)
    return sources


def require_prefix(result, states):
    if (result['status'] != 'ALL_PHASE_PLANNER_ADAPTER_STARTUP_COMPLETE'
            or result['models'] != 6 or result['original_frames_per_model'] != 4
            or result['first_terminal_difference'] != 3 or len(result['reports']) != 6
            or result['all_selected_forecasts_and_complete_original_outputs_exact'] is not True
            or result['raw_public_startup_used'] is not True
            or result['model_training'] is not False or result['native_execution'] is not False):
        raise ValueError('complete original six-model raw adapter prefix required')
    for case, report in zip(original.CASES, result['reports'], strict=True):
        if (report['case'] != case[0] or report['model_name'] != case[4]
                or report['model_state_sha256'] != states[case[4]] or report['frames'] != 4
                or report['first_terminal_difference'] != 3):
            raise ValueError('exact ordered assigned prefix models required')
        for flag in ('model_state_unchanged', 'complete_original_warmup_decisions_exact',
                'complete_original_decisions_reconstructed', 'no_recorded_observation_after_divergence_consumed',
                'observed_and_contact_state_unchanged', 'controller_class_and_all_planning_constraints_unchanged'):
            if report[flag] is not True: raise ValueError('incomplete prefix proof: '+flag)
        boundary = report['boundary']
        if (boundary['frame'] != 3 or boundary['old_failure'] != prefix.EXPECTED_FAILURE
                or boundary['original_terminal'] != 'SENSOR_OR_MODEL_FAILURE'
                or boundary['original_requested_command'] != [0., 0., 0.]
                or boundary['candidate_terminal'] is not None
                or boundary['candidate_requested_command'] == [0., 0., 0.]
                or boundary['all_original_expanded_forward_tensors_exact'] is not True
                or boundary['planner_forecast_matches_original_expanded_forward'] is not True
                or boundary['recorded_predecessor_forecasts_present'] is not False
                or report['command_executed'] is not False or report['navigation_verified'] is not False):
            raise ValueError('exact interface-only first planning transition required')


def admit(sources):
    result, launch, old_ids = completed(original.OUTPUT, ORIGINAL_SHA, ORIGINAL_LAUNCH, sources)
    waited, _, wait_ids = completed(waiter.OUTPUT, WAIT_SHA, prefix.QUEUE_SHA, sources)
    receipt = read_json(waiter.OUTPUT, 'input_completion.json')
    report = waiter.authenticate_completed(sources, receipt)
    if (waited['status'] != 'ALL_PHASE_RESIDUAL_MAZE02_NATIVE_WAIT_COMPLETE'
            or waited['report'] != report or report['native_result_sha256'] != ORIGINAL_SHA
            or result['measured_round_trip_successes'] != 0):
        raise ValueError('completed original negative cohort and waiter required')
    replay, replay_launch, prefix_ids = completed(prefix.OUTPUT, PREFIX_SHA, PREFIX_LAUNCH, sources)
    require_prefix(replay, launch['assigned_model_states'])
    if any(old_ids.get(n) != h for n, h in replay_launch['original_artifact_sha256'].items()):
        raise ValueError('adapter replay must bind the same completed original cohort')
    return dict(original_artifact_sha256=old_ids, waiter_artifact_sha256=wait_ids,
        prefix_artifact_sha256=prefix_ids, assigned_model_states=launch['assigned_model_states'],
        original_completion=report, complete_original_failures_retained=True,
        adapter_prefix_admitted=True)


def verify_admission(admission, sources):
    verify(sources)
    for root, ids, sha in ((original.OUTPUT, admission['original_artifact_sha256'], ORIGINAL_SHA),
            (waiter.OUTPUT, admission['waiter_artifact_sha256'], WAIT_SHA),
            (prefix.OUTPUT, admission['prefix_artifact_sha256'], PREFIX_SHA)):
        if ids.get('result.json') != sha: raise ValueError('fixed completed adapter evidence identities required')
        verify_artifacts(root, ids)
        result = read_json(root, 'result.json')
        if (any(ids.get(n) != h for n, h in result['artifact_sha256'].items())
                or any(sources.get(n) != h for n, h in result['source_sha256'].items())):
            raise ValueError('complete preserved input artifacts and sources required')
    if (admission['assigned_model_states'] != read_json(original.OUTPUT, 'launch.json')['assigned_model_states']
            or admission['complete_original_failures_retained'] is not True
            or admission['adapter_prefix_admitted'] is not True):
        raise ValueError('unchanged preassigned states and admitted adapter evidence required')
    require_prefix(read_json(prefix.OUTPUT, 'result.json'), admission['assigned_model_states'])

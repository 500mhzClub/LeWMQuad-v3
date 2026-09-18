"""Require completed frontier replay and the original six-case adapter batch."""
from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as batch
from scripts import replay_go2_reached_frontier_maze03_prefix_v1 as replay
from scripts.reached_frontier_native_prefix_development import admit_prefix
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from lewm.independent_reactive_floor_transport_study_development import merge_sources

BATCH_LAUNCH = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
PREFIX_LAUNCH = '28380c18617abc59c7019a90786204ae6b3281842b70ca9dcc0f8de0b93b755f'


def prepared_sources(seeds):
    sources = {}
    for root, sha in ((batch.OUTPUT, BATCH_LAUNCH), (replay.OUTPUT, PREFIX_LAUNCH)):
        verify_artifacts(root, {'launch.json': sha})
        sources = merge_sources(sources, read_json(root, 'launch.json')['source_sha256'])
    verify(sources); sources = discover_sources(seeds, sources); verify(sources)
    return sources


def require_batch(result, launch, records, audits, startups, readouts):
    if (result['status'] != 'ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE'
            or result['planner_interface_adapter_enabled'] is not True
            or result['frontier_transition_enabled'] is not False
            or launch['planner_interface_adapter_enabled'] is not True
            or launch['frontier_transition_enabled'] is not False
            or result['conditions'] != records
            or any(len(items) != 6 for items in (records, audits, startups, readouts))):
        raise ValueError('complete original six-case adapter experiment required')
    summary = batch.complete_cohort(records, audits)
    if any(result[k] != v for k, v in summary.items()):
        raise ValueError('complete original six-case outcomes must reconstruct')
    for case, record, startup, readout in zip(batch.CASES, records, startups, readouts, strict=True):
        if (record['startup_comparison'] != startup or record['readout'] != readout
                or record['model_state_sha256'] != launch['assigned_model_states'][case[4]]
                or any(result['artifact_sha256'].get(n) != h for n, h in record['artifact_sha256'].items())):
            raise ValueError('complete unchanged assigned worker evidence required')
        for flag in ('complete_candidate_decisions_match_prospective_prefix',
                'candidate_intervention_command_completed', 'all_selected_forecasts_match_prospective_prefix',
                'no_later_counterfactual_observations_used'):
            if startup[flag] is not True: raise ValueError('actual original adapter intervention required: '+flag)
        if startup['first_intervention_frame'] != 3:
            raise ValueError('original adapter first planning boundary required')
    return dict(all_six_adapter_cases_completed_and_raw_audited=True,
        measured_round_trip_successes=summary['measured_round_trip_successes'], scientific_success_required=False)


def admit(prefix_sha, batch_sha, sources):
    result, launch, batch_ids = completed(batch.OUTPUT, batch_sha, BATCH_LAUNCH, sources)
    batch.verify_inputs(launch)
    records = [read_json(batch.OUTPUT, c[0]+'_worker_terminal.json') for c in batch.CASES]
    audits = [read_json(batch.OUTPUT, c[0]+'_audit.json') for c in batch.CASES]
    startups = [read_json(batch.OUTPUT, c[0]+'_startup_comparison.json') for c in batch.CASES]
    readouts = [read_json(batch.OUTPUT, c[0]+'_readout.json') for c in batch.CASES]
    for c in batch.CASES:
        for suffix in ('_worker_terminal.json', '_audit.json', '_startup_comparison.json', '_readout.json', '_worker.log'):
            if c[0]+suffix not in batch_ids: raise ValueError('all original adapter worker outputs must be bound')
    batch_report = require_batch(result, launch, records, audits, startups, readouts)
    prefix, prefix_launch, prefix_ids = completed(replay.OUTPUT, prefix_sha, PREFIX_LAUNCH, sources)
    report = admit_prefix(replay.OUTPUT, prefix)
    _, original_launch, original_ids = completed(replay.original.OUTPUT, replay.INPUT_SHA, replay.INPUT_LAUNCH_SHA, sources)
    if prefix_launch['input_artifact_sha256'] != original_ids:
        raise ValueError('frontier replay must bind the complete original native input')
    original_admission = replay.authenticate_inputs(sources)
    if original_admission != prefix_launch['input_admission']:
        raise ValueError('unchanged complete original frontier replay admission required')
    return dict(prefix_result_sha256=prefix_sha, prefix_artifact_sha256=prefix_ids, prefix_report=report,
        adapter_batch_result_sha256=batch_sha, adapter_batch_artifact_sha256=batch_ids,
        adapter_batch_completion=batch_report, original_artifact_sha256=original_ids,
        original_admission=original_admission, correction_admission=original_launch['correction_admission'],
        complete_input_admission_performed=True)


def verify_bound(admission, sources):
    verify(sources)
    if admission['complete_input_admission_performed'] is not True:
        raise ValueError('complete native input admission required')
    for root, ids, sha in ((replay.OUTPUT, admission['prefix_artifact_sha256'], admission['prefix_result_sha256']),
            (batch.OUTPUT, admission['adapter_batch_artifact_sha256'], admission['adapter_batch_result_sha256']),
            (replay.original.OUTPUT, admission['original_artifact_sha256'], replay.INPUT_SHA)):
        if ids.get('result.json') != sha: raise ValueError('exact completed result binding required')
        verify_artifacts(root, ids); result = read_json(root, 'result.json')
        if (any(ids.get(n) != h for n, h in result['artifact_sha256'].items())
                or any(sources.get(n) != h for n, h in result['source_sha256'].items())):
            raise ValueError('all completed predecessor artifacts and sources required')
    if (admission['prefix_report'] != read_json(replay.OUTPUT, 'result.json')['report']
            or admission['correction_admission'] != read_json(replay.original.OUTPUT, 'launch.json')['correction_admission']):
        raise ValueError('unchanged original model admission and prospective prefix required')

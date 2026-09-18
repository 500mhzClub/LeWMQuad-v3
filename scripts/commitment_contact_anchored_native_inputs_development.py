"""Admit completed contact replay after the already scheduled hold native trial."""
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import await_go2_commitment_contact_anchored_raw_prefix_v1 as raw_wait
from scripts import await_go2_hold_reorientation_maze02_native_v1 as prior_wait
from scripts import replay_go2_commitment_contact_anchored_prefix_v1 as replay
from scripts.commitment_contact_anchored_native_prefix_development import admit_prefix
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify

RAW_WAIT_LAUNCH = 'ca3027a13c67b92e2e181d92a55377ae102fcfd0bd007544ba87cf5dea61c880'
PRIOR_WAIT_LAUNCH = 'e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6'


def prepared_sources(seeds):
    sources = {}
    for root, sha in ((raw_wait.OUTPUT, RAW_WAIT_LAUNCH), (prior_wait.OUTPUT, PRIOR_WAIT_LAUNCH)):
        verify_artifacts(root, {'launch.json': sha})
        sources = merge_sources(sources, read_json(root, 'launch.json')['source_sha256'])
    sources = discover_sources(seeds, sources); verify(sources)
    return sources


def require_waits(raw, prior):
    if (raw['status'] != 'COMMITMENT_CONTACT_ANCHORED_RAW_PREFIX_WAIT_V1_COMPLETE'
            or raw['native_execution'] is not False or raw['automatic_retry'] is not False
            or prior['status'] != 'HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or prior['automatic_retry'] is not False):
        raise ValueError('both original waiters must complete without replacement attempts')


def admit(raw_wait_sha, prior_wait_sha, sources):
    prior, _, prior_wait_ids = completed(prior_wait.OUTPUT, prior_wait_sha, PRIOR_WAIT_LAUNCH, sources)
    raw, _, raw_wait_ids = completed(raw_wait.OUTPUT, raw_wait_sha, RAW_WAIT_LAUNCH, sources)
    require_waits(raw, prior)
    prior_input = read_json(prior_wait.OUTPUT, 'input_completion.json')
    prior_report = prior_wait.authenticate_completed(sources, prior_input)
    if prior_report != prior['report'] or prior_report != read_json(prior_wait.OUTPUT, 'native_completion.json'):
        raise ValueError('complete original scheduled native handoff must reconstruct')
    native_sha = prior_report['native_result_sha256']; native_launch_sha = digest(prior_wait.native.OUTPUT/'launch.json')
    _, native_launch, native_ids = completed(prior_wait.native.OUTPUT, native_sha, native_launch_sha, sources)
    prior_wait.native.verify_inputs(native_launch, full=True)
    worker_sha = raw['report']['original_worker_terminal_sha256']
    if read_json(raw_wait.OUTPUT, 'input_completion.json') != {'original_worker_terminal_sha256': worker_sha}:
        raise ValueError('same original second adapter worker required')
    raw_report = raw_wait.authenticate_replay(worker_sha, sources)
    if raw_report != raw['report'] or raw_report != read_json(raw_wait.OUTPUT, 'replay_completion.json'):
        raise ValueError('complete original raw replay handoff must reconstruct')
    prefix_sha = raw_report['raw_replay_result_sha256']; prefix_launch_sha = digest(replay.OUTPUT/'launch.json')
    prefix, prefix_launch, prefix_ids = completed(replay.OUTPUT, prefix_sha, prefix_launch_sha, sources)
    original_admission = replay.admit_worker(worker_sha, sources)
    if original_admission != prefix_launch['input_admission']:
        raise ValueError('complete original second-case raw input admission changed')
    prefix_report = admit_prefix(replay.OUTPUT, prefix)
    batch = replay.original; batch_sha = native_launch['input_admission']['adapter_batch_result_sha256']
    batch_result, batch_launch, batch_ids = completed(batch.OUTPUT, batch_sha, replay.LAUNCH_SHA, sources)
    if (batch_result['status'] != 'ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE'
            or batch_result['all_fixed_cases_executed'] is not True
            or native_launch['input_admission']['all_six_adapter_cases_completed'] is not True
            or any(batch_ids.get(n) != h for n,h in original_admission['original_artifact_sha256'].items())):
        raise ValueError('same completed six-case batch must bind the original second worker')
    bindings = [dict(root=str(root), artifact_sha256=ids) for root,ids in (
        (raw_wait.OUTPUT, raw_wait_ids), (prior_wait.OUTPUT, prior_wait_ids),
        (replay.OUTPUT, prefix_ids), (prior_wait.native.OUTPUT, native_ids), (batch.OUTPUT, batch_ids))]
    return dict(raw_prefix_wait_result_sha256=raw_wait_sha, prior_native_wait_result_sha256=prior_wait_sha,
        prefix_result_sha256=prefix_sha, prefix_report=prefix_report,
        original_worker_terminal_sha256=worker_sha, original_worker_admission=original_admission,
        prior_native_result_sha256=native_sha, adapter_batch_result_sha256=batch_sha,
        correction_admission=batch_launch['input_admission']['correction_admission'],
        completed_artifact_bindings=bindings, scheduled_hold_native_completed_before_this_experiment=True,
        all_six_adapter_cases_completed=True, complete_input_admission_performed=True)


def verify_bound(admission, sources):
    verify(sources)
    if (admission['complete_input_admission_performed'] is not True
            or admission['scheduled_hold_native_completed_before_this_experiment'] is not True
            or admission['all_six_adapter_cases_completed'] is not True):
        raise ValueError('complete original replay and scheduled experiments required')
    expected = [(raw_wait.OUTPUT, admission['raw_prefix_wait_result_sha256']),
        (prior_wait.OUTPUT, admission['prior_native_wait_result_sha256']),
        (replay.OUTPUT, admission['prefix_result_sha256']),
        (prior_wait.native.OUTPUT, admission['prior_native_result_sha256']),
        (replay.original.OUTPUT, admission['adapter_batch_result_sha256'])]
    if len(admission['completed_artifact_bindings']) != len(expected):
        raise ValueError('all five exact completed artifact groups required')
    for binding, (root, sha) in zip(admission['completed_artifact_bindings'], expected, strict=True):
        ids = binding['artifact_sha256']
        if binding['root'] != str(root) or ids.get('result.json') != sha:
            raise ValueError('exact ordered predecessor root and result identity required')
        verify_artifacts(root, ids); result = read_json(root, 'result.json')
        if (any(ids.get(n) != h for n,h in result['artifact_sha256'].items())
                or any(sources.get(n) != h for n,h in result['source_sha256'].items())):
            raise ValueError('all complete predecessor artifacts and sources required')
    prior = read_json(replay.original.OUTPUT, 'launch.json')
    if (admission['prefix_report'] != read_json(replay.OUTPUT, 'result.json')['report']
            or admission['correction_admission'] != prior['input_admission']['correction_admission']):
        raise ValueError('unchanged original model and prospective report required')
    verify_artifacts(replay.original.OUTPUT, {replay.CASE[0]+'_worker_terminal.json': admission['original_worker_terminal_sha256']})

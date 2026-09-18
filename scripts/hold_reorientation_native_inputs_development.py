"""Admit completed CPU replay and prior scheduled frontier experiment."""
from pathlib import Path
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import await_go2_hold_reorientation_raw_prefix_v1 as raw_wait
from scripts import await_go2_reached_frontier_maze03_native_v1 as frontier_wait
from scripts import replay_go2_hold_reorientation_maze02_prefix_v1 as replay
from scripts.hold_reorientation_native_prefix_development import admit_prefix
from scripts.all_phase_residual_maze02_native_inputs_development import completed
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify

RAW_WAIT_LAUNCH = '969c9153d9bebbf7cc7ba24e56ed76a912edf33f0a13aeaacc243cdcd2862e2c'
FRONTIER_WAIT_LAUNCH = '68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74'
PREPARATION = 'docs/go2_hold_reorientation_native_preparation_verification_2026-09-10.json'
PREPARATION_SHA = '79e43d5cc571d6ccd7303a795a859627851ad1e932609f8e2e6c35dd31f5527e'


def prepared_sources(seeds):
    import json
    if digest(Path(PREPARATION)) != PREPARATION_SHA: raise ValueError('bound native source preparation required')
    sources = json.loads(Path(PREPARATION).read_text())['source_sha256']
    for root, sha in ((raw_wait.OUTPUT, RAW_WAIT_LAUNCH), (frontier_wait.OUTPUT, FRONTIER_WAIT_LAUNCH)):
        verify_artifacts(root, {'launch.json': sha})
        sources = merge_sources(sources, read_json(root, 'launch.json')['source_sha256'])
    sources = discover_sources((*seeds, PREPARATION), sources); verify(sources)
    return sources


def require_waits(raw, frontier):
    if (raw['status'] != 'HOLD_REORIENTATION_RAW_PREFIX_WAIT_V1_COMPLETE'
            or raw['native_execution'] is not False or raw['automatic_retry'] is not False
            or frontier['status'] != 'REACHED_FRONTIER_MAZE03_NATIVE_WAIT_COMPLETE'
            or frontier['automatic_retry'] is not False):
        raise ValueError('both original waiters must complete without replacement attempts')


def admit(raw_wait_sha, frontier_wait_sha, sources):
    front, _, front_wait_ids = completed(frontier_wait.OUTPUT, frontier_wait_sha, FRONTIER_WAIT_LAUNCH, sources)
    raw, _, raw_wait_ids = completed(raw_wait.OUTPUT, raw_wait_sha, RAW_WAIT_LAUNCH, sources)
    require_waits(raw, front)
    frontier_input = read_json(frontier_wait.OUTPUT, 'input_completion.json')
    frontier_report = frontier_wait.authenticate_completed(sources, frontier_input)
    if (frontier_report != front['report']
            or frontier_report != read_json(frontier_wait.OUTPUT, 'native_completion.json')):
        raise ValueError('complete original frontier handoff must reconstruct')
    native_sha = frontier_report['native_result_sha256']
    native_launch_sha = digest(frontier_wait.native.OUTPUT/'launch.json')
    native_result, native_launch, native_ids = completed(frontier_wait.native.OUTPUT, native_sha, native_launch_sha, sources)
    frontier_wait.native.verify_inputs(native_launch, full=True)
    worker_sha = raw['report']['original_worker_terminal_sha256']
    if read_json(raw_wait.OUTPUT, 'input_completion.json') != {'original_worker_terminal_sha256': worker_sha}:
        raise ValueError('same original first adapter worker required')
    raw_report = raw_wait.authenticate_replay(worker_sha, sources)
    if raw_report != raw['report'] or raw_report != read_json(raw_wait.OUTPUT, 'replay_completion.json'):
        raise ValueError('complete original raw replay handoff must reconstruct')
    prefix_sha = raw_report['raw_replay_result_sha256']; prefix_launch_sha = digest(replay.OUTPUT/'launch.json')
    prefix, prefix_launch, prefix_ids = completed(replay.OUTPUT, prefix_sha, prefix_launch_sha, sources)
    original_admission = replay.admit_worker(worker_sha, sources)
    if original_admission != prefix_launch['input_admission']:
        raise ValueError('complete original first-case raw input admission changed')
    prefix_report = admit_prefix(replay.OUTPUT, prefix)
    batch = replay.original
    batch_sha = frontier_input['adapter_batch_result_sha256']
    batch_result, batch_launch, batch_ids = completed(batch.OUTPUT, batch_sha, replay.LAUNCH_SHA, sources)
    if (native_launch['input_admission']['adapter_batch_result_sha256'] != batch_sha
            or native_result['adapter_batch_result_sha256'] != batch_sha
            or any(batch_ids.get(n) != h for n,h in original_admission['original_artifact_sha256'].items())):
        raise ValueError('same completed six-case batch must bind the original first worker')
    bindings = [dict(root=str(root), artifact_sha256=ids) for root,ids in (
        (raw_wait.OUTPUT, raw_wait_ids), (frontier_wait.OUTPUT, front_wait_ids),
        (replay.OUTPUT, prefix_ids), (frontier_wait.native.OUTPUT, native_ids), (batch.OUTPUT, batch_ids))]
    return dict(raw_prefix_wait_result_sha256=raw_wait_sha, frontier_wait_result_sha256=frontier_wait_sha,
        prefix_result_sha256=prefix_sha, prefix_report=prefix_report,
        original_worker_terminal_sha256=worker_sha, original_worker_admission=original_admission,
        frontier_native_result_sha256=native_sha, adapter_batch_result_sha256=batch_sha,
        correction_admission=batch_launch['input_admission']['correction_admission'],
        completed_artifact_bindings=bindings, scheduled_frontier_completed_before_this_experiment=True,
        all_six_adapter_cases_completed=True, complete_input_admission_performed=True)


def verify_bound(admission, sources):
    verify(sources)
    if (admission['complete_input_admission_performed'] is not True
            or admission['scheduled_frontier_completed_before_this_experiment'] is not True
            or admission['all_six_adapter_cases_completed'] is not True):
        raise ValueError('complete original replay and scheduled experiments required')
    expected = [(raw_wait.OUTPUT, admission['raw_prefix_wait_result_sha256']),
        (frontier_wait.OUTPUT, admission['frontier_wait_result_sha256']),
        (replay.OUTPUT, admission['prefix_result_sha256']),
        (frontier_wait.native.OUTPUT, admission['frontier_native_result_sha256']),
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

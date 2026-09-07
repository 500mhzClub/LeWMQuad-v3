"""Outside keeper for the fixed, memory-bounded independent tracking challenge.

No launch without the completed original study and genuine source-bound resource
review. No replacement unit, replay-only substitute, retry or unbounded fallback.
"""
import argparse

from scripts import independent_tracking_memory_supervision_development as memory
from scripts import run_go2_independent_tracking_challenge_v1 as challenge
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts

SOURCE = 'scripts/supervise_go2_independent_tracking_challenge_v1.py'
TEST = 'lewm/tests/test_independent_tracking_memory_supervision_development.py'


def supervise(definition_sha256, study_result_sha256):
    memory.outside_scope()
    # Metadata/source-only preflight outside the scope. Actual native collection,
    # RGB replay and raw scoring run only in the scoped parent and its children.
    d, completed = challenge.preflight(definition_sha256, study_result_sha256)
    validate_root(memory.OUTPUT, must_exist=False)
    memory.require(not memory.OUTPUT.exists() and not memory.OUTPUT.is_symlink(),
        'exclusive outside attempt; no retry/resume')
    memory.require_fresh_unit()
    store = memory.EvidenceStore()
    request_sha = store.save('request.json', memory.request(definition_sha256, study_result_sha256))
    terminal = dict(status='SUPERVISOR_INTERRUPTED_CHILD_STATE_UNVERIFIED',
        request_sha256=request_sha, definition_sha256=definition_sha256,
        unit=memory.UNIT, child_handle_terminal=False, retry_performed=False,
        navigation_qualified=False, goal_achieved=False)
    try:
        command = memory.service_command(challenge.PYTHON, challenge.SOURCE, challenge.ENVIRONMENT,
            definition_sha256, study_result_sha256, request_sha)
        evidence = memory.relay_process(command, store)
        terminal.update(evidence)
        terminal['status'] = 'SCOPED_COMMAND_FAILED'
        memory.require(evidence['systemd_run_returncode'] == 0 and evidence['log_complete'],
            'scoped command failed or diagnostic evidence truncated; no retry')
        challenge.verify_ordered_launch(d)
        verify_artifacts(memory.OUTPUT, {'request.json': request_sha, 'unit.log': evidence['log_sha256']})
        path = artifact_path(challenge.OUTPUT, 'challenge_result.json')
        memory.require(path.stat().st_size <= challenge.base.MAX_METADATA, 'bounded inner terminal report')
        result = challenge.base.read(challenge.OUTPUT, 'challenge_result.json')
        memory.require(result['status'] == 'NATIVE_TRACKING_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE'
            and result['definition_sha256'] == definition_sha256
            and 'launch.json' in result['output_sha256'],
            'successful exact challenge terminal report required, not exit zero alone')
        verify_artifacts(challenge.OUTPUT, result['output_sha256'])
        launch = challenge.base.read(challenge.OUTPUT, 'launch.json')
        memory.require(launch['definition'] == d and launch['completed_learning'] == completed
            and launch['memory_supervision']['request_sha256'] == request_sha
            and not (challenge.OUTPUT / 'failure.json').exists(), 'same bound supervised attempt required')
        memory.require(all(result[k] is False for k in ('full_challenge_pass',
            'independent_result_verification_complete', 'navigation_qualified', 'real_time_qualified', 'goal_achieved')),
            'collection completion is not scientific qualification')
        terminal.update(status='SCOPED_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE',
            challenge_result_sha256=memory.sha256(path), workload_completion_verified=True)
    except BaseException as error:
        terminal['error'] = f'{type(error).__name__}: {str(error)[:4096]}'
        store.save('terminal.json', terminal)
        raise
    store.save('terminal.json', terminal)
    return terminal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--definition-sha256', required=True)
    p.add_argument('--study-result-sha256', required=True)
    args = p.parse_args()
    supervise(args.definition_sha256, args.study_result_sha256)


if __name__ == '__main__':
    main()

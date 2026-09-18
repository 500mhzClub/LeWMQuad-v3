"""Require completion of the existing fixed queue before another native scene."""
from scripts.run_go2_prepared_native_queue_v1 import OUTPUT as QUEUE, SUPERVISED, JOBS, competitors
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import verify

QUEUE_LAUNCH_SHA = '651a5815275ecfdd20c5ff6ff7f4d8cdbe4b5c52bbcaf3ad124a6ac97ffd2fb7'


def admit_queue_result(result, launch):
    if (result['status'] != 'PREPARED_NATIVE_QUEUE_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256'].get('launch.json') != QUEUE_LAUNCH_SHA
            or result['automatic_retry'] is not False or len(result['completed']) != 4):
        raise ValueError('complete original fixed queue without retries required')
    for spec, record in zip((SUPERVISED, *JOBS), result['completed'], strict=True):
        if (record['output_root'] != str(QUEUE.parent/spec['output']) or record['cases'] != spec['cases']
                or record['all_raw_audits_pass'] is not True or record['all_physical_prefixes_pass'] is not True
                or record['original_verifier_reexecuted'] is not True or record['scientific_success_required'] is not False):
            raise ValueError('all four original ordered completion receipts required')
    return result['completed']


def verify_queue_launch(sources):
    verify_artifacts(QUEUE, {'launch.json': QUEUE_LAUNCH_SHA}); launch = read_json(QUEUE, 'launch.json')
    if any(sources.get(k) != v for k, v in launch['source_sha256'].items()):
        raise ValueError('existing queue sources must remain in the new frozen source union')
    verify(launch['source_sha256'])
    return launch


def verify_queue_completion(expected_sha, sources):
    launch = verify_queue_launch(sources)
    if (QUEUE/'failure.json').exists() or (QUEUE/'failure.json').is_symlink():
        raise ValueError('original queue failed; preserve evidence and do not bypass it')
    verify_artifacts(QUEUE, {'result.json': expected_sha}); result = read_json(QUEUE, 'result.json')
    verify_artifacts(QUEUE, result['artifact_sha256']); completed = admit_queue_result(result, launch)
    verify_artifacts(QUEUE, {'result.json': expected_sha})
    return dict(queue_launch_sha256=QUEUE_LAUNCH_SHA, queue_result_sha256=expected_sha,
        completed=completed, completion_receipts_and_bindings_verified=True,
        original_queue_verifiers_independently_reexecuted=False)


def require_native_idle():
    if competitors(): raise ValueError('existing native runner or worker still live; do not start another scene')

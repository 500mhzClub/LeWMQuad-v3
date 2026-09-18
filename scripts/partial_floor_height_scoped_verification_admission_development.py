"""Admit the completed paired verifier before using it in a fresh height pilot."""
import time
from scripts.scoped_verification_digest_development import verify_with_scoped_digests
from scripts.run_go2_partial_floor_height_maze01_pilot_v1 import verify_inputs as original_verify_inputs, PRIOR_SHA
from scripts.benchmark_go2_scoped_verification_digest_v1 import OUTPUT as BENCHMARK
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify as verify_sources
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

BENCHMARK_SHA = '137867773bfe6c6eb05a125a288012ff6017aa3134f4687d7bccdc7f99c02071'


def admit_benchmark():
    verify_artifacts(BENCHMARK, {'result.json':BENCHMARK_SHA}); result = read_json(BENCHMARK, 'result.json')
    verify_artifacts(BENCHMARK, result['artifact_sha256']); verify_sources(result['source_sha256'])
    launch = read_json(BENCHMARK, 'launch.json'); report = result['report']
    if (result['status'] != 'SCOPED_VERIFICATION_DIGEST_BENCHMARK_V1_COMPLETE'
            or result['model_loaded'] is not False or result['native_execution'] is not False
            or launch['target_result_sha256'] != PRIOR_SHA or report['pairs'] != 1
            or report['fixed_order'] != ['scoped', 'original']
            or report['both_verifiers_completed'] is not True or report['input_context_unchanged'] is not True
            or report['digest_results_reused_across_calls'] is not False
            or report['imported_verifier_globals_changed'] is not False):
        raise ValueError('completed matching original/scoped verification pair required')
    if len(report['rows']) != 2: raise ValueError('both completed verifier stages required')
    for mode, row in zip(('scoped', 'original'), report['rows'], strict=True):
        if (row != read_json(BENCHMARK, mode+'_verification.json') or row['mode'] != mode
                or row['verified'] is not True or row['input_context_unchanged'] is not True):
            raise ValueError('paired verifier stage differs from completed result')
    scope = report['rows'][0]['digest_scope']
    if (scope['every_cached_file_freshly_rehashed'] is not True
            or scope['cache_retained_after_call'] is not False
            or scope['imported_module_globals_mutated'] is not False
            or scope['original_verification_conditions_executed'] is not True
            or not 0 < scope['unique_files'] < scope['digest_requests']
            or scope['guarded_cache_hits'] != scope['digest_requests']-scope['unique_files']
            or scope['initial_hashed_bytes'] != scope['final_hashed_bytes']
            or report['rows'][1]['digest_scope'] is not None):
        raise ValueError('complete scoped digest identity and original control required')
    return result


def verify_inputs(launch):
    if launch['verification_benchmark_result_sha256'] != BENCHMARK_SHA:
        raise ValueError('exact completed verification benchmark required')
    admit_benchmark(); before = fingerprint(launch); start = time.perf_counter()
    result, report = verify_with_scoped_digests(original_verify_inputs, digest, launch)
    if result is not None or fingerprint(launch) != before:
        raise ValueError('unchanged launch and original verification result required')
    print('PARTIAL_HEIGHT_SCOPED_INPUTS_VERIFIED', time.perf_counter()-start, report, flush=True)

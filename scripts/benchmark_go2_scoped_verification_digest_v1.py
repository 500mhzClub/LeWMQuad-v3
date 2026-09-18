"""One fixed paired verification of the completed tracking native ancestry."""
import json
import time
from scripts.scoped_verification_digest_development import verify_with_scoped_digests
from scripts.run_go2_direct_flow_maze01_pilot_v1 import OUTPUT as INPUT, verify_inputs as verify_native
from scripts.diagnose_go2_direct_flow_maze01_floor_conflict_v1 import admit, CASE
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify as verify_sources
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES

OUTPUT = BASE/'go2_scoped_verification_digest_benchmark_v1_attempt_001'
PROTOCOL = 'docs/go2_scoped_verification_digest_benchmark_v1_2026-09-09.md'
INPUT_SHA = 'd6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de'
ALLOWANCE = 128*1024**2
MEMORY = 8*1024**3


def verify_target(context):
    verify_artifacts(INPUT, context['target_bindings'])
    verify_native(context['target_launch'])


def paired_verification(context, save):
    before = fingerprint(context); rows = []
    # Fixed order: scoped first, original second. One pair is not an order-
    # controlled performance estimate; both must actually finish successfully.
    for mode in ('scoped', 'original'):
        start = time.perf_counter()
        if mode == 'scoped':
            result, counts = verify_with_scoped_digests(verify_target, digest, context)
        else:
            result = verify_target(context); counts = None
        if result is not None or fingerprint(context) != before:
            raise ValueError('same verification result and unchanged input context required')
        row = dict(mode=mode, wall_s=time.perf_counter()-start, verified=True,
            input_context_unchanged=True, digest_scope=counts)
        save(row); rows.append(row)
    return dict(pairs=1, fixed_order=['scoped', 'original'], both_verifiers_completed=True,
        input_context_unchanged=True, rows=rows, digest_results_reused_across_calls=False,
        imported_verifier_globals_changed=False, controlled_speedup_established=False,
        implementation_adopted_by_existing_launchers=False, native_execution=False, model_loaded=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive verification benchmark required')
    verify_artifacts(INPUT, {'result.json':INPUT_SHA}); result = read_json(INPUT, 'result.json')
    ids = dict(result['artifact_sha256']); ids['result.json'] = INPUT_SHA; verify_artifacts(INPUT, ids)
    target = read_json(INPUT, 'launch.json'); admit(result, target, read_json(INPUT, CASE[0]+'_audit.json'))
    sources = discover_sources((PROTOCOL, 'scripts/benchmark_go2_scoped_verification_digest_v1.py',
        'lewm/tests/test_scoped_verification_digest_development.py',
        'lewm/tests/test_scoped_verification_digest_benchmark_development.py'), result['source_sha256'])
    verify_sources(sources); resources = hardware()
    if resources['memory_available_bytes'] < MEMORY or resources['artifact_free_bytes'] < RESERVE_BYTES+ALLOWANCE:
        raise ValueError('bounded verification resources unavailable')
    launch = dict(protocol=PROTOCOL, source_sha256=sources, target_root=str(INPUT), target_bindings=ids,
        target_result_sha256=INPUT_SHA, target_launch_sha256=ids['launch.json'],
        output_root=str(OUTPUT), hardware=resources, memory_admission_bytes=MEMORY,
        output_allowance_bytes=ALLOWANCE, workers=1, native_execution=False, model_loaded=False,
        fixed_order=['scoped', 'original'], cache_scope='one complete verification call',
        original_digest_on_first_request=True, fresh_final_sha256_for_every_cached_file=True)
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('SCOPED_VERIFICATION_BENCHMARK_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        def save(row):
            write_json(OUTPUT/(row['mode']+'_verification.json'), row)
            print('SCOPED_VERIFICATION_STAGE_COMPLETE', row['mode'], row['wall_s'], row['digest_scope'], flush=True)
        report = paired_verification(dict(target_bindings=ids, target_launch=target), save)
        verify_sources(sources); verify_artifacts(INPUT, ids)
        bindings = {name:digest(OUTPUT/name) for name in ('launch.json', 'scoped_verification.json', 'original_verification.json')}
        verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SCOPED_VERIFICATION_DIGEST_BENCHMARK_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report,
            wall_s=time.perf_counter()-started, hardware_after=hardware(),
            model_loaded=False, native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('SCOPED_VERIFICATION_BENCHMARK_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SCOPED_VERIFICATION_BENCHMARK_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()

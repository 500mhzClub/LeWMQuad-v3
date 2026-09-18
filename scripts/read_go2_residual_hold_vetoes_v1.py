"""Reconstruct why completed residual-maze decisions selected feasible holds."""
import time
from lewm.residual_hold_veto_readout_development import summarize_hold_vetoes
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.read_go2_residual_first_interval_maze_pilot_v1 import (
    INPUT, CASE, LEARNED, LEARNED_CASE, LEARNED_SHA, admit_results, verify_native, OUTPUT as READOUT)
from scripts.partial_floor_height_scoped_verification_admission_development import admit_benchmark, BENCHMARK_SHA
from scripts.scoped_verification_digest_development import verify_with_scoped_digests
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.maze_decision_stream_development import read_rows
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify as verify_sources
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_residual_hold_veto_readout_v1_attempt_001'
PROTOCOL = 'docs/go2_residual_hold_veto_readout_v1_2026-09-09.md'
INPUT_SHA = '55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466'
READOUT_SHA = 'aa1abf8ce0110dca271bc5f93fabf51a62bb41315fa5971afe16f87c25c59888'
ALLOWANCE = 128*1024**2


def verify_input_context(context):
    verify_artifacts(INPUT, context['native_bindings'])
    verify_native(context['native_launch'])
    verify_artifacts(READOUT, context['readout_bindings'])
    verify_artifacts(LEARNED, {'result.json':LEARNED_SHA})


def verified(context):
    before = fingerprint(context)
    result, counters = verify_with_scoped_digests(verify_input_context, digest, context)
    if result is not None or fingerprint(context) != before:
        raise ValueError('original verification result and unchanged context required')
    return counters


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive hold-veto readout required')
    benchmark = admit_benchmark()
    verify_artifacts(INPUT, {'result.json':INPUT_SHA}); result = read_json(INPUT, 'result.json')
    ids = dict(result['artifact_sha256']); ids['result.json'] = INPUT_SHA
    verify_artifacts(READOUT, {'result.json':READOUT_SHA}); readout = read_json(READOUT, 'result.json')
    if readout['native_result_sha256'] != INPUT_SHA or readout['status'] != 'RESIDUAL_FIRST_INTERVAL_MAZE_READOUT_V1_COMPLETE':
        raise ValueError('completed same native execution readout required')
    readout_ids = {'result.json':READOUT_SHA, 'launch.json':readout['launch_sha256']}
    native = read_json(INPUT, 'launch.json'); old = read_json(LEARNED, 'launch.json')
    admit_results(result, native, read_json(LEARNED, 'result.json'), old,
        read_json(INPUT, CASE[0]+'_audit.json'), read_json(LEARNED, LEARNED_CASE[0]+'_audit.json'))
    inherited = dict(result['source_sha256'])
    for source in (readout['source_sha256'], benchmark['source_sha256']):
        for name, sha in source.items():
            if name in inherited and inherited[name] != sha: raise ValueError('source binding conflict: '+name)
            inherited[name] = sha
    sources = discover_sources((PROTOCOL, 'scripts/read_go2_residual_hold_vetoes_v1.py',
        'lewm/tests/test_residual_hold_veto_readout_development.py'), inherited)
    context = dict(native_bindings=ids, native_launch=native, readout_bindings=readout_ids)
    verify_sources(sources); before = verified(context); resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+ALLOWANCE:
        raise ValueError('bounded hold-veto readout resources unavailable')
    launch = dict(protocol=PROTOCOL, source_sha256=sources, native_result_sha256=INPUT_SHA,
        readout_result_sha256=READOUT_SHA, native_bindings=ids, readout_bindings=readout_ids,
        verification_benchmark_result_sha256=BENCHMARK_SHA, verification_before=before,
        output_root=str(OUTPUT), hardware=resources, memory_admission_bytes=8*1024**3,
        output_allowance_bytes=ALLOWANCE, model_loaded=False, native_execution=False,
        scope='all completed original decisions; score and original veto reconstruction only')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('RESIDUAL_HOLD_VETO_READOUT_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        directory = INPUT/CASE[0]; tape = read_json(directory, 'command_tape.json')
        report = summarize_hold_vetoes(read_rows(directory), tape)
        if (report['observations'] != 3014 or report['selected_holds'] != 2643
                or report['first_terminal'] != dict(frame=3003, terminal='MISSION_TICK_BUDGET_EXHAUSTED')):
            raise ValueError('complete original population and terminal required')
        verify_sources(sources); after = verified(context); admit_benchmark()
        bindings = {'launch.json':digest(OUTPUT/'launch.json')}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='RESIDUAL_HOLD_VETO_READOUT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report, verification_after=after,
            native_result_sha256=INPUT_SHA, wall_s=time.perf_counter()-started,
            model_loaded=False, native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('RESIDUAL_HOLD_VETO_READOUT_COMPLETE', digest(OUTPUT/'result.json'),
            report['hold_reason_counts'], report['better_alternative_veto_counts'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_RESIDUAL_HOLD_VETO_READOUT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()

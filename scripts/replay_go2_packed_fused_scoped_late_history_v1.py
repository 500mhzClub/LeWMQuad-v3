"""Incremental packed-index comparison after the original fused replay ends."""
import argparse
import builtins
import json
import os
from pathlib import Path
import re
import time
from types import FunctionType, SimpleNamespace

import psutil
from lewm.fused_scoped_batched_controller_development import FusedScopedBatchedController
from lewm.packed_fused_scoped_controller_development import PackedFusedScopedController, normalize_to_fused
from scripts.packed_fused_state_development import normalized_state_tree, STATE_TYPE_PATHS
from scripts import replay_go2_fused_scoped_batched_late_history_v1 as previous
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/replay_go2_packed_fused_scoped_late_history_v1.py'
TEST = 'lewm/tests/test_packed_fused_scoped_replay_development.py'
PROTOCOL = 'docs/go2_packed_fused_scoped_late_history_v1_2026-09-11.md'
PREPARATION = 'docs/go2_packed_fused_scoped_controller_preparation_2026-09-11.json'
PREPARATION_SHA = 'a33aa0596da7a6cca9326ce722034c83932df95e1f8e1e717c9ac52d19c75108'
PREVIOUS_LAUNCH = '45bab28d786c146a5d2f66bf7d8d1d429f8a30e7b2007dfde0425be6ce47e6da'
PREVIOUS_OWNER = dict(pid=2780519, created=1789090638.22, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', previous.SOURCE])
OUTPUT = BASE/'go2_packed_fused_scoped_late_history_v1_attempt_001'


def prepared_sources():
    verify({PREPARATION: PREPARATION_SHA})
    preparation = json.loads((ROOT/PREPARATION).read_text())
    if preparation['status'] != 'PACKED_FUSED_SCOPED_COMPOSITION_SOURCE_CHECKED_NOT_RAW_REPLAYED':
        raise ValueError('exact packed-index composition preparation required')
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), preparation['source_sha256'])
    verify(sources)
    return sources


def previous_owner_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != previous.profile.paired.previous.BOOT:
        raise ValueError('same original fused replay boot required')
    try:
        process = psutil.Process(PREVIOUS_OWNER['pid'])
        if process.create_time() != PREVIOUS_OWNER['created'] or process.cmdline() != PREVIOUS_OWNER['command']:
            raise ValueError('original fused replay process identity changed')
    except psutil.NoSuchProcess:
        return
    raise ValueError('original fused replay still occupies full replay slot')


def require_completed(result, launch, rows, preceding, prior_rows, sources):
    report = result['report']
    if (result['status'] != 'FUSED_SCOPED_BATCHED_LATE_HISTORY_REPLAY_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or set(result['artifact_sha256']) != {'launch.json', 'comparison.jsonl'}
            or result['artifact_sha256']['launch.json'] != PREVIOUS_LAUNCH
            or launch['paired_result_sha256'] != previous.PAIRED_SHA
            or launch['profile_result_sha256'] != previous.PROFILE_SHA
            or result['sensing_scope'] != preceding['sensing_scope']
            or report['frames'] != 1428 or report['raw_model_forecast_comparisons'] != 1425
            or report['model_state_sha256'] != previous.profile.original.reference.MODEL_SHA
            or report['baseline'] != 'ScopedBatchedFootprintController'
            or report['candidate'] != 'FusedScopedBatchedController'
            or report['state_scope'] != ['memory', 'mapper.floor', 'mapper.occupied', 'residual', 'history']
            or report['normalized_state_type_paths'] != previous.profile.paired.STATE_TYPE_PATHS
            or report['observed_state_checks'] != preceding['report']['observed_state_checks']):
        raise ValueError('complete exact fused replay and unchanged original state identities required')
    for key in ('model_state_unchanged', 'complete_original_decisions_reconstructed',
                'complete_normalized_candidate_decisions_exact', 'public_input_arrays_unchanged',
                'alternating_execution_order', 'controller_observe_only_timed',
                'no_observation_1428_consumed', 'incremental_receipt_construction_comparison',
                'both_controllers_use_scoped_reuse_and_batched_patches'):
        if report[key] is not True:
            raise ValueError('complete fused replay invariant required: '+key)
    for key in ('incremental_reuse_comparison', 'imported_module_globals_mutated', 'profiling_enabled',
                'sensor_acquisition_timed', 'isolated_benchmark', 'native_execution',
                'real_time_qualified', 'navigation_qualified'):
        if report[key] is not False:
            raise ValueError('unchanged original diagnostic scope required: '+key)
    if result['native_execution'] is not False or result['goal_achieved'] is not False:
        raise ValueError('original diagnostic is not native goal completion')
    if len(rows) != 1428 or len(prior_rows) != 1428:
        raise ValueError('complete ordered fused and preceding histories required')
    for i,(row,prior) in enumerate(zip(rows,prior_rows,strict=True)):
        if (type(row['frame']) is not int or row['frame'] != i
                or row['public_input_sha256'] != prior['public_input_sha256']
                or row['original_decision_sha256'] != prior['original_decision_sha256']
                or row['baseline_decision_sha256'] != prior['candidate_decision_sha256']
                or type(row['candidate_decision_sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}',row['candidate_decision_sha256']) is None):
            raise ValueError('every original input and fused decision identity required')
        for key in ('candidate_normalized_decision_exact', 'complete_original_decision_reconstructed',
                    'public_input_arrays_unchanged'):
            if row[key] is not True:
                raise ValueError('all original fused comparisons must pass')
    if previous.profile.paired.previous.timing_summary(rows) != report['timing_windows']:
        raise ValueError('complete original fused timing population must reconstruct')


def completed_previous(result_sha, sources):
    previous_owner_ended()
    if type(result_sha) is not str or re.fullmatch('[0-9a-f]{64}',result_sha) is None:
        raise ValueError('explicit completed fused result SHA-256 required')
    if (previous.OUTPUT/'failure.json').exists() or (previous.OUTPUT/'failure.json').is_symlink():
        raise ValueError('preserve original fused failure')
    verify_artifacts(previous.OUTPUT, {'result.json': result_sha, 'launch.json': PREVIOUS_LAUNCH})
    result = read_json(previous.OUTPUT, 'result.json'); launch = read_json(previous.OUTPUT, 'launch.json')
    verify(result['source_sha256']); verify_artifacts(previous.OUTPUT, result['artifact_sha256'])
    preceding, raw_launch, prior_rows = previous.completed_reference(sources)
    rows = [json.loads(line) for line in (previous.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    require_completed(result, launch, rows, preceding, prior_rows, sources)
    return result, launch, rows, raw_launch


def normalize_candidate(decision):
    return previous.normalize_candidate(normalize_to_fused(decision))


def progress(*args, **kwargs):
    if args and args[0] == 'SCOPED_FOOTPRINT_PAIRED_FRAME':
        args = ('PACKED_FUSED_SCOPED_PAIRED_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    original = previous.profile.paired.previous.replay
    if original.__closure__ is not None:
        raise ValueError('closure-free original paired loop required')
    view = SimpleNamespace(**vars(previous.profile.paired.previous.profile))
    view.normalize_candidate = previous.normalize_candidate
    namespace = dict(original.__globals__, profile=view,
        FrozenFootprintAnchoredController=FusedScopedBatchedController,
        ScopedFootprintAnchoredController=PackedFusedScopedController,
        normalize_candidate=normalize_candidate, state_tree=normalized_state_tree, OUTPUT=OUTPUT, print=progress)
    function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    function.__kwdefaults__ = original.__kwdefaults__
    return function


def replay(rows, previous_report):
    report = isolated_replay()(rows)
    if report['observed_state_checks'] != previous_report['observed_state_checks']:
        raise ValueError('all seven original fused retained-state hashes required')
    return report | dict(baseline='FusedScopedBatchedController', candidate='PackedFusedScopedController',
        normalized_state_type_paths=STATE_TYPE_PATHS, incremental_reuse_comparison=False,
        incremental_packed_insertion_comparison=True, both_controllers_use_fused_scoped_batched_receipts=True,
        imported_module_globals_mutated=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    parser.add_argument('--fused-result-sha256'); args = parser.parse_args()
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k,v in env.items()) or previous.profile.original.cv2.ocl.useOpenCL():
        raise ValueError('assertions, single threads/hash and disabled OpenCL required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive packed/fused replay; no retry or resume')
    sources = prepared_sources()
    resources = previous.profile.original.reference.hardware(); previous.profile.paired.previous.resources_for(resources)
    if args.source_preflight_only:
        print('PACKED_FUSED_SCOPED_PREFLIGHT_PASS', len(sources), flush=True); return
    preceding = completed_previous(args.fused_result_sha256, sources)
    result, prior_launch, rows, raw_launch = preceding
    admission = previous.profile.bound_profile_inputs(raw_launch, sources)
    if admission != prior_launch['input_admission']:
        raise ValueError('exact original fused raw and model input bindings required')
    resources = previous.profile.original.reference.hardware(); previous.profile.paired.previous.resources_for(resources)
    verify(sources); previous_owner_ended(); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, fused_result_sha256=args.fused_result_sha256,
        fused_launch_sha256=PREVIOUS_LAUNCH, input_admission=admission, hardware=resources, environment=env,
        frames=1428, state_frames=previous.profile.paired.previous.STATE_FRAMES,
        normalized_state_type_paths=STATE_TYPE_PATHS, baseline='FusedScopedBatchedController',
        candidate='PackedFusedScopedController', native_execution=False, model_training=False))
    print('PACKED_FUSED_SCOPED_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    previous.profile.original.cv2.setNumThreads(1); previous.profile.original.torch.set_num_threads(1)
    previous.profile.original.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = replay(rows, result['report'])
        if (previous.profile.bound_profile_inputs(raw_launch, sources) != admission
                or completed_previous(args.fused_result_sha256, sources) != preceding):
            raise ValueError('original raw/model inputs or completed fused reference changed')
        ids = {n:digest(OUTPUT/n) for n in ('launch.json', 'comparison.jsonl')}
        verify(sources); verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='PACKED_FUSED_SCOPED_LATE_HISTORY_REPLAY_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, sensing_scope=result['sensing_scope'],
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('PACKED_FUSED_SCOPED_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_PACKED_FUSED_SCOPED_REPLAY_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()

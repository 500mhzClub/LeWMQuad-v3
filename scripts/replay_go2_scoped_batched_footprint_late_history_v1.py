"""Incremental batched-patch replay against the completed scoped-only pair."""
import argparse
import builtins
import json
import os
from pathlib import Path
import re
import time
from types import FunctionType, SimpleNamespace

import psutil
from lewm.scoped_footprint_anchored_controller_development import ScopedFootprintAnchoredController
from lewm.scoped_batched_footprint_controller_development import (
    ScopedBatchedFootprintController, normalize_to_scoped)
from scripts import replay_go2_scoped_footprint_late_history_v1 as previous
from scripts.replay_go2_batched_patch_anchored_prefix_v1 import normalized_state_tree
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/replay_go2_scoped_batched_footprint_late_history_v1.py'
TEST = 'lewm/tests/test_scoped_batched_footprint_late_history_replay_development.py'
PROTOCOL = 'docs/go2_scoped_batched_footprint_late_history_replay_v1_2026-09-10.md'
PREPARATION = 'docs/go2_scoped_batched_footprint_controller_preparation_2026-09-10.json'
PREPARATION_SHA = 'b8a9c8b332838e5422d67b9ba15daf0414c9999a51ee8b3b687b108acdc88e13'
OUTPUT = BASE/'go2_scoped_batched_footprint_late_history_v1_attempt_001'
PREVIOUS_LAUNCH_SHA = '13145419c4676be6329f49698fca9bae63eb27728184cd0b71fad1d2ee2d21dc'
PREVIOUS_PID = 2754886
PREVIOUS_CREATED = 1789077365.71
PREVIOUS_ARGV = ['.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', previous.SOURCE,
    '--profile-result-sha256', '07c096a8e6d83ac7676337993643e30dc35ebdec65758674f9ffb71c58b23c84']
STATE_TYPE_PATHS = ['memory.fields.patches.type', 'memory.fields.auxiliary_patches.type']


def prepared_sources():
    verify({PREPARATION: PREPARATION_SHA})
    preparation = json.loads((ROOT/PREPARATION).read_text())
    if preparation['status'] != 'SCOPED_BATCHED_FOOTPRINT_COMPOSITION_SOURCE_CHECKED_NOT_RAW_REPLAYED':
        raise ValueError('exact checked combined controller preparation required')
    verify(preparation['source_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), preparation['source_sha256'])
    verify(sources)
    return sources


def previous_owner_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != previous.BOOT:
        raise ValueError('original scoped replay boot identity required')
    try:
        process = psutil.Process(PREVIOUS_PID)
        if process.create_time() != PREVIOUS_CREATED or process.cmdline() != PREVIOUS_ARGV:
            raise ValueError('scoped replay PID identity changed')
    except psutil.NoSuchProcess:
        return
    raise ValueError('original scoped replay remains live')


def require_completed(result, launch, rows, sources):
    report = result['report']
    if (result['status'] != 'SCOPED_FOOTPRINT_LATE_HISTORY_PAIRED_REPLAY_V1_COMPLETE'
            or set(result['artifact_sha256']) != {'launch.json','comparison.jsonl'}
            or result['artifact_sha256']['launch.json'] != PREVIOUS_LAUNCH_SHA
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or launch['profile_result_sha256'] != PREVIOUS_ARGV[-1]
            or result['sensing_scope'] != previous.profile.sensing_scope()
            or report['frames'] != previous.FRAMES
            or report['raw_model_forecast_comparisons'] != previous.FRAMES-3
            or report['model_state_sha256'] != previous.profile.reference.MODEL_SHA
            or report['baseline'] != 'FrozenFootprintAnchoredController'
            or report['candidate'] != 'ScopedFootprintAnchoredController'
            or report['state_scope'] != ['memory','mapper.floor','mapper.occupied','residual','history']
            or report['normalized_state_type_paths'] != []):
        raise ValueError('exact complete original scoped replay and negative sensing scope required')
    for key in ('model_state_unchanged','complete_original_decisions_reconstructed',
            'complete_normalized_candidate_decisions_exact','public_input_arrays_unchanged',
            'incremental_reuse_comparison','alternating_execution_order','controller_observe_only_timed',
            'no_observation_1428_consumed'):
        if report[key] is not True:
            raise ValueError('complete original scoped replay invariant required: '+key)
    for key in ('profiling_enabled','sensor_acquisition_timed','isolated_benchmark','native_execution',
            'real_time_qualified','navigation_qualified'):
        if report[key] is not False:
            raise ValueError('original diagnostic scope required: '+key)
    if len(rows) != previous.FRAMES:
        raise ValueError('complete original scoped comparison population required')
    for i,row in enumerate(rows):
        if type(row['frame']) is not int or row['frame'] != i or any(row[k] is not True for k in (
                'candidate_normalized_decision_exact','complete_original_decision_reconstructed',
                'public_input_arrays_unchanged')):
            raise ValueError('ordered exact original scoped comparisons required')
        for key in ('public_input_sha256','original_decision_sha256','baseline_decision_sha256','candidate_decision_sha256'):
            if type(row[key]) is not str or re.fullmatch('[0-9a-f]{64}',row[key]) is None:
                raise ValueError('exact original scoped comparison identities required')
    if previous.timing_summary(rows) != report['timing_windows']:
        raise ValueError('all original scoped timing measurements must reconstruct')
    states = report['observed_state_checks']
    if ([r['frame'] for r in states] != list(previous.STATE_FRAMES)
            or any(r['retained_observed_state_equal'] is not True
                or type(r['state_sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}',r['state_sha256']) is None for r in states)):
        raise ValueError('all original scoped retained-state checkpoints required')


def completed_previous(result_sha, sources):
    previous_owner_ended()
    if type(result_sha) is not str or re.fullmatch('[0-9a-f]{64}',result_sha) is None:
        raise ValueError('explicit completed scoped replay SHA-256 required')
    if (previous.OUTPUT/'failure.json').exists() or (previous.OUTPUT/'failure.json').is_symlink():
        raise ValueError('original scoped replay failure must be preserved')
    verify_artifacts(previous.OUTPUT, {'result.json':result_sha,'launch.json':PREVIOUS_LAUNCH_SHA})
    result = read_json(previous.OUTPUT,'result.json'); launch = read_json(previous.OUTPUT,'launch.json')
    verify(result['source_sha256']); verify_artifacts(previous.OUTPUT,result['artifact_sha256'])
    rows = [json.loads(line) for line in (previous.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    require_completed(result,launch,rows,sources)
    return result, rows


def normalize_candidate(decision):
    return previous.normalize_candidate(normalize_to_scoped(decision))


def progress(*args, **kwargs):
    if args and args[0] == 'SCOPED_FOOTPRINT_PAIRED_FRAME':
        args = ('SCOPED_BATCHED_FOOTPRINT_PAIRED_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    """Keep the existing complete replay loop without mutating module globals."""
    original = previous.replay
    if original.__closure__ is not None:
        raise ValueError('closure-free exact paired replay required')
    profile = SimpleNamespace(**vars(previous.profile))
    profile.normalize_candidate = previous.normalize_candidate
    namespace = original.__globals__.copy()
    namespace.update(FrozenFootprintAnchoredController=ScopedFootprintAnchoredController,
        ScopedFootprintAnchoredController=ScopedBatchedFootprintController,
        normalize_candidate=normalize_candidate, state_tree=normalized_state_tree,
        profile=profile, OUTPUT=OUTPUT, print=progress)
    replay = FunctionType(original.__code__,namespace,original.__name__,original.__defaults__)
    replay.__kwdefaults__ = original.__kwdefaults__
    return replay


def replay(rows, previous_report):
    report = isolated_replay()(rows)
    if (report['observed_state_checks'] != previous_report['observed_state_checks']
            or report['baseline'] != 'FrozenFootprintAnchoredController'
            or report['candidate'] != 'ScopedFootprintAnchoredController'
            or report['normalized_state_type_paths'] != []):
        raise ValueError('same original replay contract and all prior scoped state hashes required')
    return report | dict(baseline='ScopedFootprintAnchoredController',
        candidate='ScopedBatchedFootprintController',normalized_state_type_paths=STATE_TYPE_PATHS,
        incremental_reuse_comparison=False,incremental_batching_comparison=True,
        both_controllers_use_scoped_reuse=True,imported_module_globals_mutated=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only',action='store_true')
    parser.add_argument('--scoped-result-sha256')
    args=parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    if any(os.environ.get(k) != v for k,v in dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0').items()):
        raise ValueError('fixed single-thread settings and hash seed required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive combined replay; no retry or resume')
    sources=prepared_sources();previous.resources_for(previous.profile.reference.hardware())
    if args.source_preflight_only:
        print('SCOPED_BATCHED_FOOTPRINT_SOURCE_PREFLIGHT_PASS',len(sources),flush=True);return
    preceding,rows=completed_previous(args.scoped_result_sha256,sources)
    print('SCOPED_BATCHED_FOOTPRINT_FULL_INPUT_ADMISSION_STARTED',flush=True)
    admission=previous.profile.reference.admit_worker(previous.profile.WORKER_SHA,sources)
    resources=previous.profile.reference.hardware();previous.resources_for(resources)
    verify(sources);previous_owner_ended();create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_admission=admission,
        scoped_result_sha256=args.scoped_result_sha256,scoped_launch_sha256=PREVIOUS_LAUNCH_SHA,
        sensing_scope=preceding['sensing_scope'],hardware=resources,frames=previous.FRAMES,
        state_frames=previous.STATE_FRAMES,normalized_state_type_paths=STATE_TYPE_PATHS,
        baseline='ScopedFootprintAnchoredController',candidate='ScopedBatchedFootprintController',
        incremental_batching_comparison=True,native_execution=False,model_training=False))
    print('SCOPED_BATCHED_FOOTPRINT_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    previous.cv2.setNumThreads(1);previous.torch.set_num_threads(1);previous.torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    try:
        report=replay(rows,preceding['report']);verify(sources)
        if (previous.profile.reference.admit_worker(previous.profile.WORKER_SHA,sources) != admission
                or completed_previous(args.scoped_result_sha256,sources) != (preceding,rows)):
            raise ValueError('complete original input admission or scoped reference changed')
        ids={n:digest(OUTPUT/n) for n in ('launch.json','comparison.jsonl')};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='SCOPED_BATCHED_FOOTPRINT_LATE_HISTORY_REPLAY_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,sensing_scope=preceding['sensing_scope'],
            wall_s=time.perf_counter()-start,native_execution=False,goal_achieved=False))
        print('SCOPED_BATCHED_FOOTPRINT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SCOPED_BATCHED_FOOTPRINT_REPLAY_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()

"""Profile the completed combined controller on the same old causal history."""
import argparse
import builtins
import json
import os
from pathlib import Path
import re
import time
from types import FunctionType

import psutil
from scripts import replay_go2_scoped_batched_footprint_late_history_v1 as paired
from scripts import profile_go2_frozen_footprint_late_history_v1 as original
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/profile_go2_scoped_batched_footprint_late_history_v1.py'
TEST = 'lewm/tests/test_scoped_batched_footprint_late_history_profile_development.py'
PROTOCOL = 'docs/go2_scoped_batched_footprint_late_history_profile_v1_2026-09-11.md'
PREPARATION = 'docs/go2_scoped_batched_footprint_late_history_replay_preparation_2026-09-10.json'
PREPARATION_SHA = '36b447e2ed4684e8a53b3980defb5fc3cfe7bb79eaee68297f3b0c1cfd1838fb'
OUTPUT = BASE/'go2_scoped_batched_footprint_late_history_profile_v1_attempt_001'
PAIRED_LAUNCH = 'b6452eac3ed27de8df47ca336e7be906727b1e98198616e172d4f9261cd95f63'
SCOPED_SHA = 'b7d80d3d98f9c4d0be92ab3a91d5d92f86d1ec3a5b78add2e078cb71a630670a'
PAIRED_OWNER = dict(pid=2766980,created=1789083663.49,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python','-B',paired.SOURCE,
    '--scoped-result-sha256',SCOPED_SHA])


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']; verify(inherited)
    sources = discover_sources((SOURCE,TEST,PROTOCOL,PREPARATION), inherited); verify(sources)
    return sources


def paired_owner_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != paired.previous.BOOT:
        raise ValueError('same original replay boot required')
    try:
        p = psutil.Process(PAIRED_OWNER['pid'])
        if p.create_time() != PAIRED_OWNER['created'] or p.cmdline() != PAIRED_OWNER['command']:
            raise ValueError('original paired replay process identity changed')
    except psutil.NoSuchProcess: return
    raise ValueError('original combined replay remains live; preserve its full-replay slot')


def require_complete(result, launch, rows, preceding, prior_rows, sources):
    report = result['report']
    if (result['status'] != 'SCOPED_BATCHED_FOOTPRINT_LATE_HISTORY_REPLAY_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or set(result['artifact_sha256']) != {'launch.json','comparison.jsonl'}
            or result['artifact_sha256']['launch.json'] != PAIRED_LAUNCH
            or launch['scoped_result_sha256'] != SCOPED_SHA
            or result['sensing_scope'] != preceding['sensing_scope']
            or report['frames'] != 1428 or report['raw_model_forecast_comparisons'] != 1425
            or report['model_state_sha256'] != original.reference.MODEL_SHA
            or report['baseline'] != 'ScopedFootprintAnchoredController'
            or report['candidate'] != 'ScopedBatchedFootprintController'
            or report['normalized_state_type_paths'] != paired.STATE_TYPE_PATHS
            or report['state_scope'] != ['memory','mapper.floor','mapper.occupied','residual','history']
            or report['observed_state_checks'] != preceding['report']['observed_state_checks']):
        raise ValueError('complete exact combined replay and original retained-state evidence required')
    for k in ('complete_original_decisions_reconstructed','complete_normalized_candidate_decisions_exact',
            'public_input_arrays_unchanged','model_state_unchanged','alternating_execution_order',
            'controller_observe_only_timed','no_observation_1428_consumed',
            'incremental_batching_comparison','both_controllers_use_scoped_reuse'):
        if report[k] is not True: raise ValueError('complete combined replay invariant required: '+k)
    for k in ('incremental_reuse_comparison','imported_module_globals_mutated','profiling_enabled',
            'sensor_acquisition_timed','isolated_benchmark','native_execution','real_time_qualified','navigation_qualified'):
        if report[k] is not False: raise ValueError('original diagnostic scope required: '+k)
    if result['native_execution'] is not False or result['goal_achieved'] is not False:
        raise ValueError('unchanged diagnostic-only scope required')
    if len(rows) != 1428 or len(prior_rows) != 1428:
        raise ValueError('complete paired histories required')
    for i,(row,prior) in enumerate(zip(rows,prior_rows,strict=True)):
        if (type(row['frame']) is not int or row['frame'] != i
                or row['public_input_sha256'] != prior['public_input_sha256']
                or row['original_decision_sha256'] != prior['original_decision_sha256']
                or row['baseline_decision_sha256'] != prior['candidate_decision_sha256']):
            raise ValueError('exact original input and baseline decision bindings required')
        for k in ('complete_original_decision_reconstructed','candidate_normalized_decision_exact','public_input_arrays_unchanged'):
            if row[k] is not True: raise ValueError('every paired decision must be exact')
        if re.fullmatch('[0-9a-f]{64}',row['candidate_decision_sha256']) is None:
            raise ValueError('exact combined candidate decision identity required')
    if paired.previous.timing_summary(rows) != report['timing_windows']:
        raise ValueError('all saved combined timing windows must reconstruct')


def admit_completed(result_sha, sources):
    paired_owner_ended()
    if (paired.OUTPUT/'failure.json').exists() or (paired.OUTPUT/'failure.json').is_symlink():
        raise ValueError('preserve original replay failure')
    verify_artifacts(paired.OUTPUT,{'result.json':result_sha,'launch.json':PAIRED_LAUNCH})
    result = read_json(paired.OUTPUT,'result.json'); launch = read_json(paired.OUTPUT,'launch.json')
    verify(result['source_sha256']); verify_artifacts(paired.OUTPUT,result['artifact_sha256'])
    preceding, prior = paired.completed_previous(SCOPED_SHA,sources)
    rows = [json.loads(line) for line in (paired.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    require_complete(result,launch,rows,preceding,prior,sources)
    paired_owner_ended()
    return result,launch,rows


def bound_profile_inputs(launch, sources):
    """Rehash actual old raw inputs and bound model inputs; no ancestry rerun."""
    native = original.reference.original; admission = launch['input_admission']
    if (admission['original_worker_terminal_sha256'] != original.WORKER_SHA
            or admission['original_case'] != original.reference.CASE[0]
            or admission['original_worker_complete_and_raw_audited'] is not True
            or admission['original_full_input_verifier_reexecuted'] is not True):
        raise ValueError('completed paired replay must bind original full admission')
    ids = admission['original_artifact_sha256']; verify_artifacts(native.OUTPUT,ids)
    native_launch = read_json(native.OUTPUT,'launch.json')
    if any(sources.get(n) != h for n,h in native_launch['source_sha256'].items()):
        raise ValueError('unchanged original native source required')
    record = read_json(native.OUTPUT,original.reference.CASE[0]+'_worker_terminal.json')
    report = read_json(native.OUTPUT,original.reference.CASE[0]+'_audit.json')
    original.reference.require_case(original.reference.CASE,record,report)
    if (record['model_state_sha256'] != original.reference.MODEL_SHA
            or any(ids.get(n) != h for n,h in record['artifact_sha256'].items())):
        raise ValueError('complete original worker data and unchanged model required')
    native.verify_inputs(native_launch,full=False)
    scope = original.sensing_scope(); verify(sources)
    return dict(original_artifact_sha256=ids,model_state_sha256=original.reference.MODEL_SHA,
        sensing_scope=scope,prior_completed_replay_full_input_admission_reused=True,
        complete_bound_raw_worker_artifacts_rehashed=True,bound_model_inputs_verified=True,
        full_training_ancestry_reexecuted=False)


def progress(*args,**kwargs):
    if args and args[0] == 'LATE_HISTORY_CONTROLLER_PROFILE_FRAME':
        args = ('SCOPED_BATCHED_CONTROLLER_PROFILE_FRAME',*args[1:])
    builtins.print(*args,**kwargs)


def isolated_replay():
    function = original.replay
    if function.__closure__ is not None: raise ValueError('closure-free original profile body required')
    clone = FunctionType(function.__code__,dict(function.__globals__,OUTPUT=OUTPUT,
        FrozenFootprintAnchoredController=paired.ScopedBatchedFootprintController,
        normalize_candidate=paired.normalize_candidate,print=progress),function.__name__,function.__defaults__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def compare_profile_rows(rows, expected):
    if len(rows) != len(expected) or len(rows) != original.FRAMES:
        raise ValueError('complete original 1428-observation profiled history required')
    for i,(row,prior) in enumerate(zip(rows,expected,strict=True)):
        if row['frame'] != i: raise ValueError('ordered profiled observations required')
        for k in ('public_input_sha256','original_decision_sha256','candidate_decision_sha256'):
            if row[k] != prior[k]: raise ValueError('profiling changed original combined input or decision')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-preflight-only',action='store_true')
    parser.add_argument('--paired-result-sha256');args=parser.parse_args()
    env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONHASHSEED='0',OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k,v in env.items()) or original.cv2.ocl.useOpenCL():
        raise ValueError('assertions, fixed CPU threads/hash and disabled OpenCL required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive profile; no retry or resume')
    sources=prepared_sources(); resources=original.reference.hardware();original.resources_for(resources)
    if args.source_preflight_only:
        print('SCOPED_BATCHED_PROFILE_SOURCE_PREFLIGHT_PASS',len(sources),flush=True);return
    result,launch,rows=admit_completed(args.paired_result_sha256,sources)
    admission=bound_profile_inputs(launch,sources)
    original.resources_for(original.reference.hardware());paired_owner_ended();create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,paired_result_sha256=args.paired_result_sha256,
        paired_launch_sha256=PAIRED_LAUNCH,input_admission=admission,environment=env,
        frames=original.FRAMES,windows=original.WINDOWS,hardware=resources,
        controller='ScopedBatchedFootprintController',native_execution=False,model_training=False))
    print('SCOPED_BATCHED_PROFILE_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    original.cv2.setNumThreads(1);original.torch.set_num_threads(1);original.torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    try:
        report=isolated_replay()()
        profiled=[json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
        compare_profile_rows(profiled,rows)
        if bound_profile_inputs(launch,sources) != admission or admit_completed(args.paired_result_sha256,sources) != (result,launch,rows):
            raise ValueError('original paired reference or bound profile inputs changed')
        names=['launch.json','comparison.jsonl']+[n+s for n in original.WINDOWS for s in ('.prof','.json')]
        ids={n:digest(OUTPUT/n) for n in names};verify_artifacts(OUTPUT,ids);verify(sources)
        report.update(controller='ScopedBatchedFootprintController',scoped_reuse_and_batched_patches_profiled=True,
            all_profiled_decisions_equal_completed_combined_replay=True,imported_module_globals_mutated=False)
        write_json(OUTPUT/'result.json',dict(status='SCOPED_BATCHED_FOOTPRINT_LATE_HISTORY_PROFILE_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,sensing_scope=admission['sensing_scope'],
            wall_s=time.perf_counter()-start,native_execution=False,goal_achieved=False))
        print('SCOPED_BATCHED_PROFILE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SCOPED_BATCHED_PROFILE_FAILURE',
            reason=repr(error),automatic_retry=False,evidence_preserved=True));raise


if __name__=='__main__':main()

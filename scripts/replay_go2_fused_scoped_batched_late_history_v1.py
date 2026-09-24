"""Pair fused receipts with the completed combined controller on old raw data."""
import argparse
import builtins
import json
import os
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

import psutil
from lewm.fused_scoped_batched_controller_development import (
    FusedScopedBatchedController, normalize_to_combined)
from lewm.scoped_batched_footprint_controller_development import ScopedBatchedFootprintController
from scripts import profile_go2_scoped_batched_footprint_late_history_v1 as profile
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/replay_go2_fused_scoped_batched_late_history_v1.py'
TEST = 'lewm/tests/test_fused_scoped_batched_replay_development.py'
COMPONENT_TEST = 'lewm/tests/test_fused_scoped_footprint_development.py'
PROTOCOL = 'docs/go2_fused_scoped_batched_late_history_v1_2026-09-11.md'
PROFILE_CHECK = 'docs/go2_scoped_batched_footprint_profile_completion_verification_2026-09-11.json'
PROFILE_CHECK_SHA = '100c32d448a47eeb6860af7a404325c1443cbef76cfc9fae5ef43743bed233df'
PROFILE_SHA = 'c1890e862509753457c1df2fca03555064dbfa1e678f93433572b0ba8f08363e'
PROFILE_LAUNCH = 'b0928142d3d2368bcaffe1fa093347c0f8207941d5c2e439ca85315a212a594f'
PAIRED_SHA = '5ebc45e317e217a39a144eacbb0029e3a57e59cf9713c27a1a0cf26c8d0cb5d0'
PROFILE_OWNER = dict(pid=2776882, created=1789088841.43, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', profile.SOURCE,
    '--paired-result-sha256', PAIRED_SHA])
OUTPUT = BASE/'go2_fused_scoped_batched_late_history_v1_attempt_001'


def prepared_sources():
    verify({PROFILE_CHECK: PROFILE_CHECK_SHA})
    check = json.loads((ROOT/PROFILE_CHECK).read_text())
    sources = discover_sources((SOURCE, TEST, COMPONENT_TEST, PROTOCOL, PROFILE_CHECK), check['source_sha256'])
    verify(sources)
    return sources


def profile_owner_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != profile.paired.previous.BOOT:
        raise ValueError('same original profile boot required')
    try:
        process = psutil.Process(PROFILE_OWNER['pid'])
        if process.create_time() != PROFILE_OWNER['created'] or process.cmdline() != PROFILE_OWNER['command']:
            raise ValueError('original profile process identity changed')
    except psutil.NoSuchProcess:
        return
    raise ValueError('original profile still occupies replay slot')


def completed_reference(sources):
    profile_owner_ended()
    if (profile.OUTPUT/'failure.json').exists():
        raise ValueError('original profile failure must be retained')
    verify_artifacts(profile.OUTPUT, {'result.json': PROFILE_SHA, 'launch.json': PROFILE_LAUNCH})
    result = read_json(profile.OUTPUT, 'result.json')
    launch = read_json(profile.OUTPUT, 'launch.json')
    check = json.loads((ROOT/PROFILE_CHECK).read_text())
    if (result['status'] != 'SCOPED_BATCHED_FOOTPRINT_LATE_HISTORY_PROFILE_V1_COMPLETE'
            or check['result_sha256'] != PROFILE_SHA or check['launch_sha256'] != PROFILE_LAUNCH
            or check['artifact_sha256'] != result['artifact_sha256']
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or launch['paired_result_sha256'] != PAIRED_SHA):
        raise ValueError('exact completed profile and checked output bindings required')
    verify(result['source_sha256']); verify_artifacts(profile.OUTPUT, result['artifact_sha256'])
    paired, paired_launch, rows = profile.admit_completed(PAIRED_SHA, sources)
    profiled = [json.loads(line) for line in (profile.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    profile.compare_profile_rows(profiled, rows)
    if result['sensing_scope'] != paired['sensing_scope']:
        raise ValueError('original negative sensing evidence must remain')
    return paired, paired_launch, rows


def normalize_candidate(decision):
    return profile.paired.normalize_candidate(normalize_to_combined(decision))


def progress(*args, **kwargs):
    if args and args[0] == 'SCOPED_FOOTPRINT_PAIRED_FRAME':
        args = ('FUSED_SCOPED_BATCHED_PAIRED_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    paired = profile.paired
    original = paired.previous.replay
    if original.__closure__ is not None:
        raise ValueError('closure-free original replay required')
    view = SimpleNamespace(**vars(paired.previous.profile))
    view.normalize_candidate = paired.normalize_candidate
    namespace = dict(original.__globals__, profile=view,
        FrozenFootprintAnchoredController=ScopedBatchedFootprintController,
        ScopedFootprintAnchoredController=FusedScopedBatchedController,
        normalize_candidate=normalize_candidate, state_tree=paired.normalized_state_tree,
        OUTPUT=OUTPUT, print=progress)
    function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    function.__kwdefaults__ = original.__kwdefaults__
    return function


def replay(rows, previous_report):
    report = isolated_replay()(rows)
    if report['observed_state_checks'] != previous_report['observed_state_checks']:
        raise ValueError('all seven original combined retained-state hashes required')
    return report | dict(baseline='ScopedBatchedFootprintController', candidate='FusedScopedBatchedController',
        normalized_state_type_paths=profile.paired.STATE_TYPE_PATHS,
        incremental_reuse_comparison=False, incremental_receipt_construction_comparison=True,
        both_controllers_use_scoped_reuse_and_batched_patches=True, imported_module_globals_mutated=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k,v in env.items()) or profile.original.cv2.ocl.useOpenCL():
        raise ValueError('assertions, single threads/hash and disabled OpenCL required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive fused replay; no retry or resume')
    sources = prepared_sources()
    resources = profile.original.reference.hardware(); profile.paired.previous.resources_for(resources)
    if args.source_preflight_only:
        print('FUSED_SCOPED_BATCHED_PREFLIGHT_PASS', len(sources), flush=True); return
    preceding, launch, rows = completed_reference(sources)
    admission = profile.bound_profile_inputs(launch, sources)
    resources = profile.original.reference.hardware(); profile.paired.previous.resources_for(resources)
    verify(sources); profile_owner_ended(); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, profile_result_sha256=PROFILE_SHA,
        paired_result_sha256=PAIRED_SHA, input_admission=admission, hardware=resources, environment=env,
        frames=1428, state_frames=profile.paired.previous.STATE_FRAMES,
        baseline='ScopedBatchedFootprintController', candidate='FusedScopedBatchedController',
        native_execution=False, model_training=False))
    print('FUSED_SCOPED_BATCHED_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    profile.original.cv2.setNumThreads(1); profile.original.torch.set_num_threads(1)
    profile.original.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = replay(rows, preceding['report'])
        if (profile.bound_profile_inputs(launch, sources) != admission
                or completed_reference(sources) != (preceding, launch, rows)):
            raise ValueError('original raw/model inputs or completed reference changed')
        ids = {n:digest(OUTPUT/n) for n in ('launch.json', 'comparison.jsonl')}
        verify(sources); verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='FUSED_SCOPED_BATCHED_LATE_HISTORY_REPLAY_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, sensing_scope=preceding['sensing_scope'],
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('FUSED_SCOPED_BATCHED_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_FUSED_SCOPED_BATCHED_REPLAY_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()

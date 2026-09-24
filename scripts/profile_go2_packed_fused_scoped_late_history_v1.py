"""Profile remaining packed/fused controller costs on the same causal history."""
import argparse
import builtins
import json
import os
import time
from types import FunctionType

from scripts import verify_go2_packed_fused_scoped_completion_v1 as completed
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

paired = completed.run
profile = paired.previous.profile
original = profile.original
SOURCE = 'scripts/profile_go2_packed_fused_scoped_late_history_v1.py'
TEST = 'lewm/tests/test_packed_fused_scoped_late_history_profile_development.py'
PROTOCOL = 'docs/go2_packed_fused_scoped_late_history_profile_v1_2026-09-11.md'
VERIFICATION = 'docs/go2_packed_fused_scoped_late_history_completion_verification_2026-09-11.json'
VERIFICATION_SHA = '13facf5948079eb6d26571b8d8977dacda3da77e9ab72b628575ad84a213411c'
PACKED_SHA = 'ddcb9719bd60b55d44865078773c7098357fc7db316f0c0987dd5690ae52d72f'
OUTPUT = BASE/'go2_packed_fused_scoped_late_history_profile_v1_attempt_001'


def prepared_sources():
    verify({VERIFICATION:VERIFICATION_SHA})
    witness = json.loads((ROOT/VERIFICATION).read_text())
    if (witness['status'] != 'PACKED_FUSED_SCOPED_LATE_HISTORY_COMPLETION_VERIFIED'
            or witness['report']['result_sha256'] != PACKED_SHA):
        raise ValueError('checked completed original packed replay required')
    sources = discover_sources((SOURCE, TEST, PROTOCOL, VERIFICATION), witness['source_sha256'])
    verify(sources)
    return sources


def admit_completed(sources):
    receipt = completed.completed(PACKED_SHA, sources)
    result = read_json(paired.OUTPUT, 'result.json'); launch = read_json(paired.OUTPUT, 'launch.json')
    rows = [json.loads(line) for line in (paired.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    # completed() has authenticated this exact preceding combined launch via
    # the original packed -> fused -> combined completion chain.
    verify_artifacts(profile.paired.OUTPUT, {'launch.json':profile.PAIRED_LAUNCH})
    raw_launch = read_json(profile.paired.OUTPUT, 'launch.json')
    return receipt, result, launch, rows, raw_launch


def progress(*args, **kwargs):
    if args and args[0] == 'LATE_HISTORY_CONTROLLER_PROFILE_FRAME':
        args = ('PACKED_FUSED_CONTROLLER_PROFILE_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    function = original.replay
    if function.__closure__ is not None: raise ValueError('closure-free original profiling body required')
    clone = FunctionType(function.__code__, function.__globals__ | dict(OUTPUT=OUTPUT,
        FrozenFootprintAnchoredController=paired.PackedFusedScopedController,
        normalize_candidate=paired.normalize_candidate, print=progress), function.__name__, function.__defaults__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def compare_profile_rows(rows, expected):
    if len(rows) != len(expected) or len(rows) != 1428:
        raise ValueError('complete original 1428-observation profiled history required')
    for i, (row, prior) in enumerate(zip(rows, expected, strict=True)):
        if type(row['frame']) is not int or row['frame'] != i:
            raise ValueError('ordered profiled observations required')
        for key in ('public_input_sha256', 'original_decision_sha256', 'candidate_decision_sha256'):
            if row[key] != prior[key]: raise ValueError('profiling changed original packed input or decision')
        for key in ('candidate_normalized_decision_exact', 'complete_original_decision_reconstructed', 'public_input_arrays_unchanged'):
            if row[key] is not True: raise ValueError('complete unchanged profiled observation required')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k, v in env.items()) or original.cv2.ocl.useOpenCL():
        raise ValueError('assertions, fixed threads/hash and disabled OpenCL required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive profile; no retry/resume')
    sources = prepared_sources(); resources = original.reference.hardware(); original.resources_for(resources)
    if args.source_preflight_only:
        print('PACKED_FUSED_PROFILE_SOURCE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            native_execution=False, output_created=False, full_input_admission_performed=False)), flush=True); return
    preceding = admit_completed(sources)
    receipt, result, launch, rows, raw_launch = preceding
    admission = profile.bound_profile_inputs(raw_launch, sources)
    if admission != launch['input_admission']:
        raise ValueError('same completed packed raw/model input admission required')
    original.resources_for(original.reference.hardware()); verify(sources)
    if completed.owner_live(completed.OWNER): raise ValueError('original packed replay still live')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, packed_result_sha256=PACKED_SHA,
        packed_launch_sha256=completed.LAUNCH_SHA, input_admission=admission, environment=env,
        frames=1428, windows=original.WINDOWS, hardware=resources,
        controller='PackedFusedScopedController', native_execution=False, model_training=False))
    print('PACKED_FUSED_PROFILE_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    original.cv2.setNumThreads(1); original.torch.set_num_threads(1); original.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = isolated_replay()()
        profiled = [json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
        compare_profile_rows(profiled, rows)
        if profile.bound_profile_inputs(raw_launch, sources) != admission or admit_completed(sources) != preceding:
            raise ValueError('bound raw/model inputs or completed packed reference changed')
        names = ['launch.json', 'comparison.jsonl']+[n+s for n in original.WINDOWS for s in ('.prof', '.json')]
        ids = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids); verify(sources)
        report.update(controller='PackedFusedScopedController', packed_fused_scoped_controller_profiled=True,
            all_profiled_decisions_equal_completed_packed_replay=True, imported_module_globals_mutated=False)
        write_json(OUTPUT/'result.json', dict(status='PACKED_FUSED_SCOPED_LATE_HISTORY_PROFILE_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, sensing_scope=admission['sensing_scope'],
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('PACKED_FUSED_PROFILE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_PACKED_FUSED_PROFILE_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise


if __name__ == '__main__': main()

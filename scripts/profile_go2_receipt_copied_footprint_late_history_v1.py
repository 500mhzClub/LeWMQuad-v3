"""Profile remaining receipt-copy/fused controller costs on the same causal history."""
import argparse
import builtins
import json
import os
import time
from types import FunctionType

from scripts import verify_go2_receipt_copied_footprint_completion_v1 as completed
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

paired = completed.run
profile = paired.profile.profile
original = paired.profile.original
SOURCE = 'scripts/profile_go2_receipt_copied_footprint_late_history_v1.py'
TEST = 'lewm/tests/test_receipt_copied_footprint_late_history_profile_development.py'
PROTOCOL = 'docs/go2_receipt_copied_footprint_late_history_profile_v1_2026-09-11.md'
VERIFICATION = 'docs/go2_receipt_copied_footprint_completion_verification_2026-09-11.json'
VERIFICATION_SHA = '3426bf9cab4431bd87b6451ce25d28d9904f348f467cc63e602ba1fabd0dd951'
COPIED_SHA = '3e82c140151b4e0976b65bd4b439c519f87a89e4f4df3cb9b2975df7324fe56d'
COPIED_LAUNCH_SHA = 'dc3d4c80ef00d1cf183414459ede26f7f24d1879c5829472bfeb44faa4697388'
OUTPUT = BASE/'go2_receipt_copied_footprint_late_history_profile_v1_attempt_001'


def prepared_sources():
    verify({VERIFICATION:VERIFICATION_SHA})
    witness = json.loads((ROOT/VERIFICATION).read_text())
    if (witness['status'] != 'RECEIPT_COPIED_FOOTPRINT_COMPLETION_VERIFIED'
            or witness['result_sha256'] != COPIED_SHA):
        raise ValueError('checked completed original receipt-copy replay required')
    sources = discover_sources((SOURCE, TEST, PROTOCOL, VERIFICATION), witness['source_sha256'])
    verify(sources)
    return sources


def capture_verification():
    """Reexecute the frozen checker without overwriting its original receipt."""
    receipts = []
    def capture(path, value):
        if path != completed.OUTPUT or receipts:
            raise ValueError('exactly one original completion receipt required')
        receipts.append(value)
    function = completed.main
    clone = FunctionType(function.__code__, function.__globals__ | dict(write_json=capture, print=lambda *a, **k:None),
        function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone()
    if len(receipts) != 1: raise ValueError('original completion checker did not produce its receipt')
    return receipts[0]


def admit_completed(sources):
    verify({VERIFICATION:VERIFICATION_SHA})
    witness = json.loads((ROOT/VERIFICATION).read_text())
    if any(sources.get(n) != h for n,h in witness['source_sha256'].items()):
        raise ValueError('same checked original receipt-copy source bindings required')
    actual = capture_verification()
    if {k:v for k,v in actual.items() if k != 'utc'} != {k:v for k,v in witness.items() if k != 'utc'}:
        raise ValueError('complete frozen receipt-copy verification must reconstruct')
    result = read_json(paired.OUTPUT, 'result.json'); launch = read_json(paired.OUTPUT, 'launch.json')
    rows = [json.loads(line) for line in (paired.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    verify_artifacts(profile.paired.OUTPUT, {'launch.json':profile.PAIRED_LAUNCH})
    raw_launch = read_json(profile.paired.OUTPUT, 'launch.json')
    return witness, result, launch, rows, raw_launch


def progress(*args, **kwargs):
    if args and args[0] == 'LATE_HISTORY_CONTROLLER_PROFILE_FRAME':
        args = ('RECEIPT_COPIED_FOOTPRINT_CONTROLLER_PROFILE_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    function = original.replay
    if function.__closure__ is not None: raise ValueError('closure-free original profiling body required')
    clone = FunctionType(function.__code__, function.__globals__ | dict(OUTPUT=OUTPUT,
        FrozenFootprintAnchoredController=paired.ReceiptCopiedFootprintController,
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
            if row[key] != prior[key]: raise ValueError('profiling changed original receipt-copy input or decision')
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
        print('RECEIPT_COPIED_FOOTPRINT_PROFILE_SOURCE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            native_execution=False, output_created=False, full_input_admission_performed=False)), flush=True); return
    preceding = admit_completed(sources)
    receipt, result, launch, rows, raw_launch = preceding
    admission = profile.bound_profile_inputs(raw_launch, sources)
    if admission != launch['input_admission']:
        raise ValueError('same completed receipt-copy raw/model input admission required')
    original.resources_for(original.reference.hardware()); verify(sources)
    if paired.owner_live(receipt['original_owner']): raise ValueError('original receipt-copy replay still live')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, copied_result_sha256=COPIED_SHA,
        copied_launch_sha256=COPIED_LAUNCH_SHA, input_admission=admission, environment=env,
        frames=1428, windows=original.WINDOWS, hardware=resources,
        controller='ReceiptCopiedFootprintController', native_execution=False, model_training=False))
    print('RECEIPT_COPIED_FOOTPRINT_PROFILE_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    original.cv2.setNumThreads(1); original.torch.set_num_threads(1); original.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = isolated_replay()()
        profiled = [json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
        compare_profile_rows(profiled, rows)
        if profile.bound_profile_inputs(raw_launch, sources) != admission or admit_completed(sources) != preceding:
            raise ValueError('bound raw/model inputs or completed receipt-copy reference changed')
        names = ['launch.json', 'comparison.jsonl']+[n+s for n in original.WINDOWS for s in ('.prof', '.json')]
        ids = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids); verify(sources)
        report.update(controller='ReceiptCopiedFootprintController', receipt_copied_footprint_controller_profiled=True,
            all_profiled_decisions_equal_completed_copied_replay=True, imported_module_globals_mutated=False)
        write_json(OUTPUT/'result.json', dict(status='RECEIPT_COPIED_FOOTPRINT_LATE_HISTORY_PROFILE_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, sensing_scope=admission['sensing_scope'],
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('RECEIPT_COPIED_FOOTPRINT_PROFILE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_RECEIPT_COPIED_FOOTPRINT_PROFILE_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise


if __name__ == '__main__': main()

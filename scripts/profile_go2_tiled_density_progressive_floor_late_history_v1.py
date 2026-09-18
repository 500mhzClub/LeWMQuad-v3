"""Profile the verified tiled controller on its complete original sensor history."""
import argparse
import builtins
import json
import os
import re
import time
from types import FunctionType, SimpleNamespace

from scripts import profile_go2_receipt_copied_footprint_late_history_v1 as predecessor
from scripts import verify_go2_tiled_density_progressive_floor_controller_completion_v1 as completed
from lewm.tiled_density_progressive_floor_controller_development import TiledDensityProgressiveFloorController
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

paired = completed.run
original = predecessor.original
SOURCE = 'scripts/profile_go2_tiled_density_progressive_floor_late_history_v1.py'
TEST = 'lewm/tests/test_tiled_density_progressive_floor_profile_development.py'
PROTOCOL = 'docs/go2_tiled_density_progressive_floor_profile_v1_2026-09-11.md'
PREPARATION = 'docs/go2_tiled_density_progressive_floor_controller_completion_preparation_2026-09-11.json'
PREPARATION_SHA = 'a2bbac6f70acf89bd32ab7b2c5775476693783974ca32bf697932c770497b8dc'
VERIFICATION = 'docs/go2_tiled_density_progressive_floor_controller_completion_verification_2026-09-11.json'
LAUNCH_SHA = '8ad6b00ce598b26b2058c9c904a51a908f2f63f4398b6eb5b46e8484b6ca45dc'
OUTPUT = BASE/'go2_tiled_density_progressive_floor_late_history_profile_v1_attempt_001'


def exact_sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('actual completed SHA256 binding required')
    return value


def prepared_sources():
    verify({PREPARATION: PREPARATION_SHA})
    preparation = json.loads((ROOT/PREPARATION).read_text())
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), preparation['source_sha256'])
    verify(sources)
    return sources


def verification_sources(sources, completion_sha, result_sha):
    exact_sha(completion_sha); exact_sha(result_sha)
    verify({VERIFICATION: completion_sha})
    witness = json.loads((ROOT/VERIFICATION).read_text())
    if (witness['status'] != 'TILED_DENSITY_PROGRESSIVE_FLOOR_COMPLETION_VERIFIED'
            or witness['result_sha256'] != result_sha
            or witness['artifact_sha256']['launch.json'] != LAUNCH_SHA):
        raise ValueError('verified original tiled replay required')
    for name, sha in witness['source_sha256'].items():
        if sources.get(name) != sha:
            raise ValueError('same checked original tiled source bindings required')
    return discover_sources((VERIFICATION,), sources)


def capture_verification(result_sha):
    """Run the exact frozen checker with an in-memory output and private CLI.

    Its exclusive destination is a new, absent capture path. Neither that path
    nor the existing verification receipt is written. Every admission, owner,
    raw/model, report, timing and source check in the checker still executes.
    """
    exact_sha(result_sha)
    destination = OUTPUT/'read_only_completion_capture.json'
    receipts = []

    def capture(path, value):
        if path != destination or receipts:
            raise ValueError('exactly one captured completion receipt required')
        receipts.append(value)

    class FixedArguments:
        def add_argument(self, *args, **kwargs):
            if args != ('--result-sha256',) or kwargs != {'required': True}:
                raise ValueError('unchanged frozen checker CLI required')

        def parse_args(self):
            return SimpleNamespace(result_sha256=result_sha)

    function = completed.main
    if function.__closure__ is not None:
        raise ValueError('closure-free frozen completion checker required')
    bindings = dict(OUTPUT=destination, write_json=capture,
        argparse=SimpleNamespace(ArgumentParser=FixedArguments), print=lambda *a, **k: None)
    clone = FunctionType(function.__code__, function.__globals__ | bindings,
        function.__name__, function.__defaults__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone()
    if len(receipts) != 1:
        raise ValueError('exactly one captured completion receipt required')
    return receipts[0]


def admit_completed(sources, completion_sha, result_sha):
    verification_sources(sources, completion_sha, result_sha)
    witness = json.loads((ROOT/VERIFICATION).read_text())
    actual = capture_verification(result_sha)
    if {k:v for k,v in actual.items() if k != 'utc'} != {k:v for k,v in witness.items() if k != 'utc'}:
        raise ValueError('complete frozen tiled verification must reconstruct')
    result = read_json(paired.OUTPUT, 'result.json')
    launch = read_json(paired.OUTPUT, 'launch.json')
    rows = [json.loads(line) for line in (paired.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    verify_artifacts(predecessor.profile.paired.OUTPUT, {'launch.json': predecessor.profile.PAIRED_LAUNCH})
    raw_launch = read_json(predecessor.profile.paired.OUTPUT, 'launch.json')
    return witness, result, launch, rows, raw_launch


def progress(*args, **kwargs):
    if args and args[0] == 'LATE_HISTORY_CONTROLLER_PROFILE_FRAME':
        args = ('TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    function = original.replay
    if function.__closure__ is not None:
        raise ValueError('closure-free original profiling body required')
    clone = FunctionType(function.__code__, function.__globals__ | dict(OUTPUT=OUTPUT,
        FrozenFootprintAnchoredController=TiledDensityProgressiveFloorController,
        normalize_candidate=paired.harness.normalize_candidate, print=progress),
        function.__name__, function.__defaults__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


compare_profile_rows = predecessor.compare_profile_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only', action='store_true')
    parser.add_argument('--completion-sha256')
    parser.add_argument('--result-sha256')
    args = parser.parse_args()
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k,v in env.items()) or original.cv2.ocl.useOpenCL():
        raise ValueError('assertions, fixed threads/hash and disabled OpenCL required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive profile; no retry/resume')
    sources = prepared_sources()
    resources = original.reference.hardware(); paired.original.resources_for(resources)
    if args.source_preflight_only:
        print('TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_SOURCE_PREFLIGHT', json.dumps(dict(
            source_count=len(sources), hardware=resources, native_execution=False,
            output_created=False, full_input_admission_performed=False)), flush=True)
        return
    sources = verification_sources(sources, args.completion_sha256, args.result_sha256)
    preceding = admit_completed(sources, args.completion_sha256, args.result_sha256)
    receipt, result, launch, rows, raw_launch = preceding
    admission = predecessor.profile.bound_profile_inputs(raw_launch, sources)
    if admission != launch['input_admission']:
        raise ValueError('same completed tiled raw/model input admission required')
    paired.original.resources_for(original.reference.hardware()); verify(sources)
    if paired.owner_live(receipt['original_owner']):
        raise ValueError('original tiled replay still live')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources,
        completed_result_sha256=args.result_sha256, completion_sha256=args.completion_sha256,
        completed_launch_sha256=LAUNCH_SHA, input_admission=admission, environment=env,
        frames=1428, windows=original.WINDOWS, hardware=resources,
        controller='TiledDensityProgressiveFloorController', native_execution=False, model_training=False))
    print('TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    original.cv2.setNumThreads(1); original.torch.set_num_threads(1)
    original.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = isolated_replay()()
        profiled = [json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
        compare_profile_rows(profiled, rows)
        if (predecessor.profile.bound_profile_inputs(raw_launch, sources) != admission
                or admit_completed(sources, args.completion_sha256, args.result_sha256) != preceding):
            raise ValueError('bound raw/model inputs or completed tiled reference changed')
        names = ['launch.json', 'comparison.jsonl']+[n+s for n in original.WINDOWS for s in ('.prof', '.json')]
        ids = {n:digest(OUTPUT/n) for n in names}; verify_artifacts(OUTPUT, ids); verify(sources)
        report.update(controller='TiledDensityProgressiveFloorController',
            tiled_density_progressive_floor_controller_profiled=True,
            all_profiled_decisions_equal_completed_tiled_replay=True, imported_module_globals_mutated=False)
        write_json(OUTPUT/'result.json', dict(status='TILED_DENSITY_PROGRESSIVE_FLOOR_LATE_HISTORY_PROFILE_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, sensing_scope=admission['sensing_scope'],
            wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()

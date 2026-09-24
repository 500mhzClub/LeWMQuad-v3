"""Paired raw original/combined-optimization replay with unchanged replay code."""
import argparse
import builtins
import json
from pathlib import Path
import time
from types import FunctionType

from lewm.reused_floor_mesh_controller_development import ReusedFloorMeshController, CONTROLLER, FLAG
from scripts import replay_go2_frozen_footprint_anchored_prefix_v1 as frozen
from scripts import replay_go2_receipt_copied_anchored_prefix_v1 as preceding
from scripts import profile_go2_frozen_footprint_controller_windows_v1 as optimized_profile
from scripts.navigation_artifact_root_development import create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json

profile = preceding.profile
OUTPUT = profile.reference.BASE/'go2_reused_floor_mesh_prefix_v1_attempt_001'
PROFILE_LAUNCH_SHA = '41b564dae8ec07a43c0c9d23068010b8f520692bb49465afbeb7ca216ba0b6d1'
PROFILE_RESULT_SHA = 'c636eb55c13f02624b73680295ab3f70d7faac00680d7e66dd820b870cfb9866'
PROFILE_VERIFICATION = Path('docs/go2_frozen_footprint_controller_profile_verification_2026-09-10.json')
PROFILE_VERIFICATION_SHA = 'bc432a67568521ee173057ecd6b0fda714c8281ca88480e203d4df439dafea56'
SOURCE = 'scripts/replay_go2_reused_floor_mesh_prefix_v1.py'
PROTOCOL = 'docs/go2_reused_floor_mesh_v1_2026-09-10.md'
TEST = 'lewm/tests/test_reused_floor_mesh_replay_development.py'
FRAMES = preceding.FRAMES
STATE_FRAMES = preceding.STATE_FRAMES
SEEDS = (SOURCE, PROTOCOL, TEST, 'lewm/tests/test_reused_floor_mesh_development.py',
    'lewm/tests/test_reused_floor_mesh_controller_development.py', str(PROFILE_VERIFICATION),
    'docs/go2_frozen_footprint_controller_profile_result_2026-09-10.md')


def normalize_candidate(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit combined mesh reuse and frozen footprint implementation required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = frozen.CONTROLLER
    return frozen.normalize_candidate(result)


def progress(*args, **kwargs):
    if args and args[0] == 'RECEIPT_COPIED_ANCHORED_RAW_FRAME':
        args = ('REUSED_FLOOR_MESH_RAW_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    original = preceding.replay
    if original.__closure__ is not None:
        raise ValueError('closure-free original paired replay required')
    namespace = original.__globals__.copy()
    namespace.update(ReceiptCopiedAnchoredController=ReusedFloorMeshController,
        normalize_candidate=normalize_candidate, OUTPUT=OUTPUT, print=progress)
    result = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def prepared_inputs():
    verify_artifacts(optimized_profile.OUTPUT, {'launch.json': PROFILE_LAUNCH_SHA})
    launch = read_json(optimized_profile.OUTPUT, 'launch.json')
    verify(launch['source_sha256'])
    completed, verification = optimized_profile.completed_inputs()
    if any(launch['source_sha256'].get(n) != h for n,h in verification['verification_source_sha256'].items()):
        raise ValueError('profile source closure must contain completed paired verification')
    if digest(PROFILE_VERIFICATION) != PROFILE_VERIFICATION_SHA:
        raise ValueError('exact completed profile verification required')
    witness = json.loads(PROFILE_VERIFICATION.read_text())
    if (witness['result_sha256'] != PROFILE_RESULT_SHA or witness['launch_sha256'] != PROFILE_LAUNCH_SHA
            or witness['verified_source_sha256'] != launch['source_sha256']):
        raise ValueError('profile verification identities and source closure must match')
    return launch


def completed_profile(result_sha, launch):
    if result_sha != PROFILE_RESULT_SHA:
        raise ValueError('exact independently checked completed profile identity required')
    root = optimized_profile.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('failed profiling attempt cannot authorize this replay')
    verify_artifacts(root, {'result.json': result_sha, 'launch.json': PROFILE_LAUNCH_SHA})
    result = read_json(root, 'result.json')
    names = {'launch.json', 'comparison.jsonl', 'early_navigation.prof', 'early_navigation.json',
        'repeated_hold.prof', 'repeated_hold.json'}
    if (result['status'] != 'FROZEN_FOOTPRINT_CONTROLLER_WINDOWS_PROFILE_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or set(result['artifact_sha256']) != names
            or result['artifact_sha256']['launch.json'] != PROFILE_LAUNCH_SHA
            or result['native_execution'] is not False or result['goal_achieved'] is not False):
        raise ValueError('exact complete non-native optimized profile required')
    report = result['report']
    expected = dict(frames=405, raw_model_forecast_comparisons=402,
        model_state_sha256=profile.reference.MODEL_SHA, model_state_unchanged=True,
        complete_original_decisions_reconstructed=True, complete_normalized_candidate_decisions_exact=True,
        no_observation_405_consumed=True, invocation_frozen_footprint_receipts=True,
        normalization_outside_profiled_region=True, native_execution=False, policy_changed=False)
    if any(type(report.get(k)) is not type(v) or report[k] != v for k,v in expected.items()):
        raise ValueError('complete exact profile scope required')
    verify(result['source_sha256']); verify_artifacts(root, result['artifact_sha256'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only', action='store_true')
    parser.add_argument('--profile-result-sha256')
    args = parser.parse_args()
    if not args.source_preflight_only and args.profile_result_sha256 is None:
        parser.error('--profile-result-sha256 is required for raw replay')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive combined optimization replay required')
    launch = prepared_inputs()
    sources = discover_sources(SEEDS, launch['source_sha256']); verify(sources)
    resources = profile.reference.hardware(); optimized_profile.resources_for(resources)
    if args.source_preflight_only:
        print('REUSED_FLOOR_MESH_SOURCE_PREFLIGHT_PASS', len(sources), flush=True)
        return
    completed = completed_profile(args.profile_result_sha256, launch)
    print('REUSED_FLOOR_MESH_INPUT_ADMISSION_STARTED', flush=True)
    admission = profile.reference.admit_worker(profile.WORKER_SHA, sources)
    resources = profile.reference.hardware(); optimized_profile.resources_for(resources)
    verify(sources); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources,
        optimized_profile_result_sha256=args.profile_result_sha256,
        optimized_profile_launch_sha256=PROFILE_LAUNCH_SHA,
        paired_frozen_footprint_result_sha256=optimized_profile.COMPLETED_SHA,
        input_admission=admission, frames=FRAMES, state_frames=STATE_FRAMES,
        protocol=PROTOCOL, hardware=resources, model_state_sha256=profile.reference.MODEL_SHA,
        native_execution=False, model_training=False, profiling_enabled=False,
        imported_module_globals_mutated=False, invocation_frozen_footprint_receipts=True,
        observation_local_floor_mesh_reuse=True, normalized_state_type_paths=[],
        comparison='original controller versus combined frozen-footprint and mesh-reuse controller',
        isolated_incremental_mesh_speedup_claimed=False))
    print('REUSED_FLOOR_MESH_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    preceding.cv2.setNumThreads(1); preceding.torch.set_num_threads(1)
    preceding.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = isolated_replay()(); verify(sources)
        if profile.reference.admit_worker(profile.WORKER_SHA, sources) != admission:
            raise ValueError('original full input admission changed')
        if completed_profile(args.profile_result_sha256, launch) != completed or prepared_inputs() != launch:
            raise ValueError('completed optimized profile inputs changed')
        ids = {name:digest(OUTPUT/name) for name in ('launch.json', 'comparison.jsonl')}
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='REUSED_FLOOR_MESH_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids,
            optimized_profile_result_sha256=args.profile_result_sha256,
            report=report, wall_s=time.perf_counter()-start, native_execution=False,
            isolated_incremental_mesh_speedup_claimed=False, goal_achieved=False))
        print('REUSED_FLOOR_MESH_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REUSED_FLOOR_MESH_PREFIX_FAILURE',
            reason=repr(error)))
        raise


if __name__ == '__main__':
    main()

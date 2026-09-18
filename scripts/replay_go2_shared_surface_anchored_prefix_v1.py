"""Paired raw replay for invocation-local surface receipt sharing using unchanged replay code."""
import argparse
import builtins
import time
from types import FunctionType
from lewm.shared_surface_anchored_controller_development import (
    SharedSurfaceAnchoredController, CONTROLLER, FLAG)
from scripts import replay_go2_batched_patch_anchored_prefix_v1 as completed_batch
from scripts import replay_go2_receipt_copied_anchored_prefix_v1 as preceding
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree
from scripts.navigation_artifact_root_development import create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json

profile = preceding.profile
OUTPUT = profile.reference.BASE/'go2_shared_surface_anchored_prefix_v1_attempt_001'
PRECEDING_SHA = '650fe137b41c91f8860ce59987c876d1c2591329ee2fc63ad056e63059e47419'
SOURCE = 'scripts/replay_go2_shared_surface_anchored_prefix_v1.py'
PROTOCOL = 'docs/go2_shared_surface_anchored_prefix_v1_2026-09-10.md'
TEST = 'lewm/tests/test_shared_surface_anchored_replay_development.py'
FRAMES = preceding.FRAMES
STATE_FRAMES = preceding.STATE_FRAMES


def normalize_candidate(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit shared surface receipt implementation required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = 'residual_anchored_continuation_controller_v1'
    return result


def progress(*args, **kwargs):
    if args and args[0] == 'RECEIPT_COPIED_ANCHORED_RAW_FRAME':
        args = ('SHARED_SURFACE_ANCHORED_RAW_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def isolated_replay():
    """Retain the completed paired replay's exact calculation code and globals."""
    original = preceding.replay
    if original.__closure__ is not None:
        raise ValueError('closure-free original paired replay required')
    namespace = original.__globals__.copy()
    namespace.update(ReceiptCopiedAnchoredController=SharedSurfaceAnchoredController,
        normalize_candidate=normalize_candidate,
        OUTPUT=OUTPUT, print=progress)
    result = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def preceding_inputs():
    verify_artifacts(completed_batch.OUTPUT, {'result.json': PRECEDING_SHA})
    result = read_json(completed_batch.OUTPUT, 'result.json')
    if result['status'] != 'BATCHED_PATCH_ANCHORED_PREFIX_V1_COMPLETE':
        raise ValueError('completed exact batched-query comparison required')
    verify(result['source_sha256']); verify_artifacts(completed_batch.OUTPUT, result['artifact_sha256'])
    completed_batch.preceding_inputs()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive shared surface receipt paired prefix required')
    predecessor = preceding_inputs()
    seeds = (SOURCE, PROTOCOL, TEST,
        'lewm/tests/test_shared_surface_anchored_development.py',
        'docs/go2_shared_surface_anchored_v1_2026-09-10.md',
        'docs/go2_batched_patch_anchored_prefix_result_2026-09-10.md',
        'docs/go2_batched_patch_anchored_prefix_verification_2026-09-10.json')
    sources = discover_sources(seeds, predecessor['source_sha256']); verify(sources)
    resources = profile.reference.hardware(); profile.resources_for(resources)
    if args.source_preflight_only:
        print('SHARED_SURFACE_ANCHORED_SOURCE_PREFLIGHT_PASS', len(sources), flush=True)
        return
    print('SHARED_SURFACE_ANCHORED_INPUT_ADMISSION_STARTED', flush=True)
    admission = profile.reference.admit_worker(profile.WORKER_SHA, sources)
    resources = profile.reference.hardware(); profile.resources_for(resources)
    verify(sources); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, preceding_result_sha256=PRECEDING_SHA,
        profile_result_sha256=preceding.PROFILE_SHA, input_admission=admission,
        frames=FRAMES, state_frames=STATE_FRAMES, protocol=PROTOCOL, hardware=resources,
        model_state_sha256=profile.reference.MODEL_SHA, native_execution=False, model_training=False,
        profiling_enabled=False, imported_module_globals_mutated=False,
        receipt_copy_optimization_enabled=False, batched_patch_queries_enabled=False,
        invocation_local_surface_receipt_sharing=True, normalized_state_type_paths=[]))
    print('SHARED_SURFACE_ANCHORED_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    preceding.cv2.setNumThreads(1); preceding.torch.set_num_threads(1)
    preceding.torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    try:
        report = isolated_replay()(); verify(sources)
        if profile.reference.admit_worker(profile.WORKER_SHA, sources) != admission:
            raise ValueError('original complete input admission changed')
        preceding_inputs()
        ids = {name: digest(OUTPUT/name) for name in ('launch.json', 'comparison.jsonl')}
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='SHARED_SURFACE_ANCHORED_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, preceding_result_sha256=PRECEDING_SHA,
            report=report, wall_s=time.perf_counter()-start, native_execution=False, goal_achieved=False))
        print('SHARED_SURFACE_ANCHORED_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SHARED_SURFACE_ANCHORED_PREFIX_FAILURE',
            reason=repr(error)))
        raise


if __name__ == '__main__':
    main()

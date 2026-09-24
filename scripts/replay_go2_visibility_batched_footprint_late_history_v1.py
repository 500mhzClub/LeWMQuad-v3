"""Compare empty visibility skipping with the completed receipt-copy controller."""
import argparse
import builtins
import json
import os
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

from lewm.visibility_batched_footprint_controller_development import VisibilityBatchedFootprintController, normalize_to_receipt_copied
from lewm.receipt_copied_footprint_development import ReceiptCopiedFootprintController
from scripts import profile_go2_receipt_copied_footprint_late_history_v1 as profile
from scripts.visibility_batched_footprint_state_development import normalized_state_tree, STATE_TYPE_PATHS
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE='scripts/replay_go2_visibility_batched_footprint_late_history_v1.py'
TEST='lewm/tests/test_visibility_batched_footprint_replay_development.py'
PROTOCOL='docs/go2_visibility_batched_footprint_late_history_v1_2026-09-11.md'
WITNESS='docs/go2_receipt_copied_profile_completion_verification_2026-09-11.json'
WITNESS_SHA='b94eb7781acd5a45a9ea93329e73f513a1d20fd27630f5497f9454c03d5a0119'
MICROBENCHMARK='docs/go2_visibility_batched_retained_patch_microbenchmark_2026-09-11.json'
MICROBENCHMARK_SHA='4b3bfb228073d51b10794e27ecdd4f34988094fe5d9cd9f698335f9c9af7cc74'
OUTPUT=BASE/'go2_visibility_batched_footprint_late_history_v1_attempt_001'
original=profile.paired.original


def prepared_sources():
    verify({WITNESS:WITNESS_SHA})
    witness=json.loads((ROOT/WITNESS).read_text())
    if witness['status'] != 'RECEIPT_COPIED_PROFILE_COMPLETION_VERIFIED':
        raise ValueError('completed original profile verification required')
    verify({MICROBENCHMARK:MICROBENCHMARK_SHA})
    benchmark=json.loads((ROOT/MICROBENCHMARK).read_text())
    if benchmark['status']!='VISIBILITY_BATCHED_RETAINED_PATCH_MICROBENCHMARK_COMPLETE':
        raise ValueError('completed exact visibility microbenchmark required')
    inherited=merge_sources(witness['source_sha256'],benchmark['source_sha256'])
    sources=discover_sources((SOURCE,TEST,PROTOCOL,WITNESS,MICROBENCHMARK,
        'lewm/tests/test_visibility_batched_footprint_controller_development.py'),inherited)
    verify(sources)
    return sources


def previous_owner_ended():
    verify({WITNESS:WITNESS_SHA})
    witness=json.loads((ROOT/WITNESS).read_text())
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != original.BOOT:
        raise ValueError('original execution boot required')
    if owner_live(witness['original_owner']):
        raise ValueError('original profile still occupies full replay slot')


def normalize_candidate(decision):
    return profile.paired.normalize_candidate(normalize_to_receipt_copied(decision))


def progress(*args,**kwargs):
    if args and args[0]=='SCOPED_FOOTPRINT_PAIRED_FRAME':
        args=('VISIBILITY_BATCHED_FOOTPRINT_PAIRED_FRAME',*args[1:])
    builtins.print(*args,**kwargs)


def isolated_replay():
    function=original.replay
    view=SimpleNamespace(**vars(original.profile))
    view.normalize_candidate=profile.paired.normalize_candidate
    namespace=dict(function.__globals__,profile=view,
        FrozenFootprintAnchoredController=ReceiptCopiedFootprintController,
        ScopedFootprintAnchoredController=VisibilityBatchedFootprintController,
        normalize_candidate=normalize_candidate,state_tree=normalized_state_tree,OUTPUT=OUTPUT,print=progress)
    clone=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__
    return clone


def replay(rows,prior_report):
    report=isolated_replay()(rows)
    if report['observed_state_checks'] != prior_report['observed_state_checks']:
        raise ValueError('all seven completed receipt-copy retained-state identities required')
    return report | dict(baseline='ReceiptCopiedFootprintController',candidate='VisibilityBatchedFootprintController',
        normalized_state_type_paths=STATE_TYPE_PATHS,incremental_reuse_comparison=False,
        incremental_empty_patch_visibility_skip_comparison=True,persistent_memory_type_unchanged=True,
        both_controllers_use_receipt_copied_packed_fused_queries=True,imported_module_globals_mutated=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-preflight-only',action='store_true')
    args=parser.parse_args()
    env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             PYTHONHASHSEED='0',OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k)!=v for k,v in env.items()) or profile.original.cv2.ocl.useOpenCL():
        raise ValueError('assertions, fixed threads/hash and disabled OpenCL required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive copy replay; no retry or resume')
    sources=prepared_sources();previous_owner_ended()
    resources=profile.original.reference.hardware();original.resources_for(resources)
    if args.source_preflight_only:
        print('VISIBILITY_BATCHED_FOOTPRINT_PREFLIGHT_PASS',len(sources),json.dumps(resources),flush=True);return
    preceding=profile.admit_completed(sources)
    _,result,launch,rows,raw_launch=preceding
    admission=profile.profile.bound_profile_inputs(raw_launch,sources)
    if admission!=launch['input_admission']:raise ValueError('same receipt-copy raw/model admission required')
    resources=profile.original.reference.hardware();original.resources_for(resources)
    verify(sources);previous_owner_ended();create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,copied_result_sha256=profile.COPIED_SHA,
        copied_launch_sha256=profile.COPIED_LAUNCH_SHA,input_admission=admission,
        hardware=resources,environment=env,frames=1428,state_frames=original.STATE_FRAMES,
        normalized_state_type_paths=STATE_TYPE_PATHS,baseline='ReceiptCopiedFootprintController',
        candidate='VisibilityBatchedFootprintController',native_execution=False,model_training=False))
    print('VISIBILITY_BATCHED_FOOTPRINT_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    profile.original.cv2.setNumThreads(1);profile.original.torch.set_num_threads(1)
    profile.original.torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    try:
        report=replay(rows,result['report'])
        if (profile.profile.bound_profile_inputs(raw_launch,sources)!=admission
                or profile.admit_completed(sources)!=preceding):
            raise ValueError('original raw/model inputs or completed receipt-copy reference changed')
        ids={n:digest(OUTPUT/n) for n in ('launch.json','comparison.jsonl')}
        verify(sources);verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='VISIBILITY_BATCHED_FOOTPRINT_LATE_HISTORY_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,sensing_scope=result['sensing_scope'],
            wall_s=time.perf_counter()-start,native_execution=False,goal_achieved=False))
        print('VISIBILITY_BATCHED_FOOTPRINT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_VISIBILITY_BATCHED_FOOTPRINT_FAILURE',
            reason=repr(error),automatic_retry=False,evidence_preserved=True))
        raise


if __name__=='__main__':main()

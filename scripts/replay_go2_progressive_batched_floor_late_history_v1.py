"""Compare progressive patch batching against the completed density-routed controller."""
import argparse
import builtins
import json
import os
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

from lewm.progressive_batched_floor_controller_development import ProgressiveBatchedFloorController, normalize_to_density_routed
from lewm.density_routed_floor_controller_development import DensityRoutedFloorController
from scripts import replay_go2_density_routed_floor_late_history_v1 as previous
from scripts import verify_go2_density_routed_floor_controller_completion_v1 as previous_check
from scripts.startup_raw_sensor_audit_development import read_json
from scripts import profile_go2_receipt_copied_footprint_late_history_v1 as profile
from scripts.progressive_batched_floor_state_development import normalized_state_tree, STATE_TYPE_PATHS
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE='scripts/replay_go2_progressive_batched_floor_late_history_v1.py'
TEST='lewm/tests/test_progressive_batched_floor_replay_development.py'
PROTOCOL='docs/go2_progressive_batched_floor_late_history_v1_2026-09-11.md'
WITNESS='docs/go2_density_routed_floor_controller_completion_verification_2026-09-11.json'
WITNESS_SHA='6cc04190aea13972ff2a6236dc510170d2cc458672d07df26345a743170ff370'
MICROBENCHMARK='docs/go2_progressive_batched_retained_patch_microbenchmark_2026-09-11.json'
MICROBENCHMARK_SHA='f3e5b7921fcf62a286129e6d0952626c3dd6132a5d9c49c08f0dc273691be82b'
OUTPUT=BASE/'go2_progressive_batched_floor_late_history_v1_attempt_001'
PREVIOUS_LAUNCH_SHA='cb9a6a86ad782bf813b78dcee3fb9fc9de1731423491b5a8cd1cd11621228100'
PREVIOUS_RESULT_SHA='cc1680b8922273d809fd2663c9f39be967638dfd9757858025c89b8b22cc2420'
REGISTRATION_EXECUTION='docs/go2_density_routed_floor_registration_execution_2026-09-11.json'
REGISTRATION_EXECUTION_SHA='4aca6e8a962d12beb5efbe6cfc2eb65bb3779cbfe052e0928fb3173f2610680f'
original=profile.paired.original


def prepared_sources():
    verify({WITNESS:WITNESS_SHA})
    witness=json.loads((ROOT/WITNESS).read_text())
    if witness['status'] != 'DENSITY_ROUTED_FLOOR_COMPLETION_VERIFIED':
        raise ValueError('completed original controller verification required')
    verify({MICROBENCHMARK:MICROBENCHMARK_SHA})
    benchmark=json.loads((ROOT/MICROBENCHMARK).read_text())
    if benchmark['status']!='PROGRESSIVE_BATCHED_RETAINED_PATCH_MICROBENCHMARK_COMPLETE':
        raise ValueError('completed progressive patch microbenchmark required')
    inherited=merge_sources(witness['source_sha256'],benchmark['source_sha256'])
    verify({REGISTRATION_EXECUTION:REGISTRATION_EXECUTION_SHA})
    sources=discover_sources((SOURCE,TEST,PROTOCOL,WITNESS,MICROBENCHMARK,
        'lewm/tests/test_progressive_batched_floor_controller_development.py',REGISTRATION_EXECUTION),inherited)
    verify(sources)
    return sources


def previous_owner_ended():
    verify({WITNESS:WITNESS_SHA})
    witness=json.loads((ROOT/WITNESS).read_text())
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != original.BOOT:
        raise ValueError('original execution boot required')
    if owner_live(witness['original_owner']):
        raise ValueError('previous controller still occupies full replay slot')
    verify({REGISTRATION_EXECUTION:REGISTRATION_EXECUTION_SHA})
    if owner_live(json.loads((ROOT/REGISTRATION_EXECUTION).read_text())['owner']):
        raise ValueError('registration replay still occupies full replay slot')


def admit_completed(sources):
    previous_owner_ended()
    fixed={'launch.json':PREVIOUS_LAUNCH_SHA,'result.json':PREVIOUS_RESULT_SHA}
    verify_artifacts(previous.OUTPUT,fixed)
    if (previous.OUTPUT/'failure.json').exists():raise ValueError('previous failure cannot be admitted as completion')
    result=read_json(previous.OUTPUT,'result.json');launch=read_json(previous.OUTPUT,'launch.json')
    if (result['status']!='DENSITY_ROUTED_FLOOR_LATE_HISTORY_V1_COMPLETE'
            or result['source_sha256']!=launch['source_sha256']
            or any(sources.get(n)!=h for n,h in result['source_sha256'].items())
            or set(result['artifact_sha256'])!={'launch.json','comparison.jsonl'}
            or result['artifact_sha256']['launch.json']!=PREVIOUS_LAUNCH_SHA
            or result['native_execution'] is not False or result['goal_achieved'] is not False):
        raise ValueError('complete fixed preceding controller replay required')
    verify_artifacts(previous.OUTPUT,result['artifact_sha256'])
    prior=previous.admit_completed(sources)
    _,prior_result,prior_launch,prior_rows,raw_launch=prior
    admission=profile.profile.bound_profile_inputs(raw_launch,sources)
    if admission!=launch['input_admission'] or admission!=prior_launch['input_admission']:
        raise ValueError('same original public raw and assigned model inputs required')
    rows=[json.loads(line) for line in (previous.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    timing=previous_check.check_rows(rows,prior_rows)
    expected=previous_check.expected_report(prior_result['report'],timing)
    if (previous_check.fingerprint(result['report'])!=previous_check.fingerprint(expected)
            or result['sensing_scope']!=prior_result['sensing_scope']):
        raise ValueError('complete original result and negative sensing scope must reconstruct')
    verify(sources);verify_artifacts(previous.OUTPUT,result['artifact_sha256']|fixed)
    return fixed,result,launch,rows,raw_launch


def normalize_candidate(decision):
    return previous.normalize_candidate(normalize_to_density_routed(decision))


def progress(*args,**kwargs):
    if args and args[0]=='SCOPED_FOOTPRINT_PAIRED_FRAME':
        args=('PROGRESSIVE_BATCHED_FLOOR_PAIRED_FRAME',*args[1:])
    builtins.print(*args,**kwargs)


def isolated_replay():
    function=original.replay
    view=SimpleNamespace(**vars(original.profile))
    view.normalize_candidate=previous.normalize_candidate
    namespace=dict(function.__globals__,profile=view,
        FrozenFootprintAnchoredController=DensityRoutedFloorController,
        ScopedFootprintAnchoredController=ProgressiveBatchedFloorController,
        normalize_candidate=normalize_candidate,state_tree=normalized_state_tree,OUTPUT=OUTPUT,print=progress)
    clone=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__
    return clone


def replay(rows,prior_report):
    report=isolated_replay()(rows)
    if report['observed_state_checks'] != prior_report['observed_state_checks']:
        raise ValueError('all seven completed density-routed retained-state identities required')
    return report | dict(baseline='DensityRoutedFloorController',candidate='ProgressiveBatchedFloorController',
        normalized_state_type_paths=STATE_TYPE_PATHS,incremental_reuse_comparison=False,
        incremental_progressive_patch_batching_comparison=True,persistent_memory_type_unchanged=True,
        both_controllers_use_density_routed_floor_registration_and_mapping=True,imported_module_globals_mutated=False)


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
        print('PROGRESSIVE_BATCHED_FLOOR_PREFLIGHT_PASS',len(sources),json.dumps(resources),flush=True);return
    preceding=admit_completed(sources)
    _,result,launch,rows,raw_launch=preceding
    admission=profile.profile.bound_profile_inputs(raw_launch,sources)
    if admission!=launch['input_admission']:raise ValueError('same receipt-copy raw/model admission required')
    resources=profile.original.reference.hardware();original.resources_for(resources)
    verify(sources);previous_owner_ended();create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,previous_result_sha256=PREVIOUS_RESULT_SHA,
        previous_launch_sha256=PREVIOUS_LAUNCH_SHA,progressive_patch_microbenchmark_sha256=MICROBENCHMARK_SHA,input_admission=admission,
        hardware=resources,environment=env,frames=1428,state_frames=original.STATE_FRAMES,
        normalized_state_type_paths=STATE_TYPE_PATHS,baseline='DensityRoutedFloorController',
        candidate='ProgressiveBatchedFloorController',native_execution=False,model_training=False))
    print('PROGRESSIVE_BATCHED_FLOOR_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    profile.original.cv2.setNumThreads(1);profile.original.torch.set_num_threads(1)
    profile.original.torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    try:
        report=replay(rows,result['report'])
        if (profile.profile.bound_profile_inputs(raw_launch,sources)!=admission
                or admit_completed(sources)!=preceding):
            raise ValueError('original raw/model inputs or completed receipt-copy reference changed')
        ids={n:digest(OUTPUT/n) for n in ('launch.json','comparison.jsonl')}
        verify(sources);verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='PROGRESSIVE_BATCHED_FLOOR_LATE_HISTORY_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,sensing_scope=result['sensing_scope'],
            wall_s=time.perf_counter()-start,native_execution=False,goal_achieved=False))
        print('PROGRESSIVE_BATCHED_FLOOR_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_PROGRESSIVE_BATCHED_FLOOR_FAILURE',
            reason=repr(error),automatic_retry=False,evidence_preserved=True))
        raise


if __name__=='__main__':main()

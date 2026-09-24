"""Compare primitive-leaf receipt freezing with the completed tiled controller."""
import argparse
import json
import os
from pathlib import Path
import re
import time

from scripts import replay_go2_tiled_density_progressive_floor_late_history_v1 as previous
from scripts import verify_go2_tiled_density_progressive_floor_controller_completion_v1 as previous_check
from scripts import atomic_leaf_freeze_tiled_replay_development as harness
from scripts.startup_raw_sensor_audit_development import read_json
from scripts import profile_go2_receipt_copied_footprint_late_history_v1 as profile
from scripts.progressive_batched_floor_state_development import normalized_state_tree, STATE_TYPE_PATHS
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

SOURCE='scripts/replay_go2_atomic_leaf_freeze_tiled_late_history_v1.py'
TEST='lewm/tests/test_atomic_leaf_freeze_tiled_admission_development.py'
PROTOCOL='docs/go2_atomic_leaf_freeze_tiled_late_history_v1_2026-09-11.md'
WITNESS='docs/go2_tiled_density_progressive_floor_controller_completion_verification_2026-09-11.json'
WITNESS_SHA='543ff2e72fc8f31b1f241f5f905212ca8d952348c3216ca6f73af7666b564556'
MICROBENCHMARK='docs/go2_atomic_leaf_fused_footprint_component_benchmark_2026-09-11.json'
MICROBENCHMARK_SHA='4d9bfd6302f01b82ad1da9c4bb116b2fb1447ce51dd13769418c082196799083'
OUTPUT=harness.OUTPUT
PREVIOUS_LAUNCH_SHA='8ad6b00ce598b26b2058c9c904a51a908f2f63f4398b6eb5b46e8484b6ca45dc'
PREVIOUS_RESULT_SHA='9f83c8e10db48c31918d386ba4867818a3e143625dde191e565aa6b796fb7307'
FREEZE_PREPARATION='docs/go2_atomic_leaf_freeze_tiled_replay_harness_preparation_2026-09-11.json'
FREEZE_PREPARATION_SHA='eebd974238e1df486a3df4d0854e20cb81bd4cb53ef941c68726320948c914a7'
original=profile.paired.original


def prepared_sources():
    if any(re.fullmatch('[0-9a-f]{64}', value) is None for value in (WITNESS_SHA, PREVIOUS_RESULT_SHA)):
        raise ValueError('actual completed tiled result and completion bindings required')
    verify({WITNESS:WITNESS_SHA})
    witness=json.loads((ROOT/WITNESS).read_text())
    if witness['status'] != 'TILED_DENSITY_PROGRESSIVE_FLOOR_COMPLETION_VERIFIED':
        raise ValueError('completed original controller verification required')
    verify({MICROBENCHMARK:MICROBENCHMARK_SHA})
    benchmark=json.loads((ROOT/MICROBENCHMARK).read_text())
    if benchmark['status']!='ATOMIC_LEAF_FUSED_FOOTPRINT_SYNTHETIC_COMPONENT_BENCHMARK_COMPLETE':
        raise ValueError('completed synthetic primitive-leaf component benchmark required')
    verify({FREEZE_PREPARATION:FREEZE_PREPARATION_SHA})
    preparation=json.loads((ROOT/FREEZE_PREPARATION).read_text())
    if preparation['status']!='ATOMIC_LEAF_FREEZE_TILED_REPLAY_HARNESS_PREPARED_NOT_LAUNCHED':
        raise ValueError('tested freeze-only controller and full replay harness required')
    inherited=merge_sources(witness['source_sha256'],benchmark['source_sha256'],preparation['source_sha256'])
    sources=discover_sources((SOURCE,TEST,PROTOCOL,WITNESS,MICROBENCHMARK,
        harness.TEST,FREEZE_PREPARATION),inherited)
    verify(sources)
    return sources


def previous_owner_ended():
    verify({WITNESS:WITNESS_SHA})
    witness=json.loads((ROOT/WITNESS).read_text())
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != original.BOOT:
        raise ValueError('original execution boot required')
    if owner_live(witness['original_owner']):
        raise ValueError('previous controller still occupies full replay slot')


def admit_completed(sources):
    previous_owner_ended()
    fixed={'launch.json':PREVIOUS_LAUNCH_SHA,'result.json':PREVIOUS_RESULT_SHA}
    verify_artifacts(previous.OUTPUT,fixed)
    if (previous.OUTPUT/'failure.json').exists():raise ValueError('previous failure cannot be admitted as completion')
    result=read_json(previous.OUTPUT,'result.json');launch=read_json(previous.OUTPUT,'launch.json')
    if (result['status']!='TILED_DENSITY_PROGRESSIVE_FLOOR_LATE_HISTORY_V1_COMPLETE'
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


replay=harness.replay


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
        print('ATOMIC_LEAF_FREEZE_TILED_PREFLIGHT_PASS',len(sources),json.dumps(resources),flush=True);return
    preceding=admit_completed(sources)
    _,result,launch,rows,raw_launch=preceding
    admission=profile.profile.bound_profile_inputs(raw_launch,sources)
    if admission!=launch['input_admission']:raise ValueError('same receipt-copy raw/model admission required')
    resources=profile.original.reference.hardware();original.resources_for(resources)
    verify(sources);previous_owner_ended();create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,previous_result_sha256=PREVIOUS_RESULT_SHA,
        previous_launch_sha256=PREVIOUS_LAUNCH_SHA,atomic_leaf_freeze_component_benchmark_sha256=MICROBENCHMARK_SHA,input_admission=admission,
        hardware=resources,environment=env,frames=1428,state_frames=original.STATE_FRAMES,
        normalized_state_type_paths=STATE_TYPE_PATHS,baseline='TiledDensityProgressiveFloorController',
        candidate='AtomicLeafFreezeTiledController',native_execution=False,model_training=False))
    print('ATOMIC_LEAF_FREEZE_TILED_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
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
        write_json(OUTPUT/'result.json',dict(status='ATOMIC_LEAF_FREEZE_TILED_LATE_HISTORY_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,sensing_scope=result['sensing_scope'],
            wall_s=time.perf_counter()-start,native_execution=False,goal_achieved=False))
        print('ATOMIC_LEAF_FREEZE_TILED_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ATOMIC_LEAF_FREEZE_TILED_FAILURE',
            reason=repr(error),automatic_retry=False,evidence_preserved=True))
        raise


if __name__=='__main__':main()

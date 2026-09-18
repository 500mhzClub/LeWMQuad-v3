"""Coarse phase durations with exact complete decisions on the frozen prefix."""
import argparse
import hashlib
import json
import time
import cv2
import torch
from lewm.controller_phase_timing_development import PhaseTiming, PhaseTimedLaterFloorController, model_forward_timing
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.replay_go2_later_floor_resolution_maze_prefix_v1 import (
    OUTPUT as PREFIX, INPUT, CASE, verify_inputs)

OUTPUT=BASE/'go2_controller_phase_timing_v1_attempt_001'
PROTOCOL='docs/go2_controller_phase_timing_v1_2026-09-09.md'
RESULT='a4e34e4ca8b72422f8fc6c1ca54b33a80c9821ff076fbc431d0575f0a1701fbe'


def verify_all(launch):
    verify_inputs(launch)
    verify_artifacts(PREFIX,launch['prefix_artifact_sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive phase timing diagnosis required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    verify_artifacts(PREFIX,{'result.json':RESULT});result=read_json(PREFIX,'result.json');report=result['report']
    assert result['status']=='LATER_FLOOR_RESOLUTION_PREFIX_COMPLETE'
    assert report['frames']==960 and report['first_requested_command_difference']==959
    assert report['final_terminal'] is None and report['stopped_before_unexecuted_outcome']
    ids={'result.json':RESULT,**result['artifact_sha256']};verify_artifacts(PREFIX,ids)
    old=read_json(PREFIX,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_controller_phase_timing_v1.py',
        'lewm/tests/test_controller_phase_timing_development.py'),result['source_sha256'])
    resources=hardware()
    launch=old|dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,
        prefix_artifact_sha256=ids,hardware=resources,maximum_frames=960,
        implementation_class='PhaseTimedLaterFloorController',native_execution=False,native_scene_workers=0,
        model_training=False,cpu_processes=1,numerical_threads=1,output_allowance_bytes=128*1024**2,
        minimum_available_ram_bytes=8*1024**3,os_resource_limits_enforced=False,
        concurrency_reason='one exact-prefix phase timing replay beside the existing single native scene',
        instrumentation_overhead_included=True,controlled_speed_comparison=False)
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=8*1024**3
    storage_ok=resources['artifact_free_bytes']>=RESERVE_BYTES+128*1024**2
    if args.preflight_only:
        print('CONTROLLER_PHASE_TIMING_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('bounded phase replay resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('CONTROLLER_PHASE_TIMING_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter()
    try:
        name,index,variant,condition,model_name=CASE
        model,c,v=load_assigned(launch['correction_admission'],model_name);assert(c,v)==(condition,variant)
        before=state_digest(model.state_dict());assert before==report['model_state_sha256']
        timing=PhaseTiming()
        controller=PhaseTimedLaterFloorController(model,ArticulatedCollisionGeometry(URDF),timing=timing,
            public_mission=public_mission(index),navigation_ticks=NAVIGATION_TICKS,persistent=True,condition=c,variant=v)
        directory=INPUT/name;reader=IntentReturnRGBDReplay(directory)
        acquisitions=read_json(directory,'auxiliary_camera_audit.json');frames=0
        with model_forward_timing(model,timing),(OUTPUT/'phase_timings.jsonl').open('x') as stream:
            for i,saved in enumerate(read_rows(PREFIX)):
                if i>=960 or saved['tick']!=i:raise ValueError('exact ordered bound prefix required')
                p,d,f,now=reader.packet(i)
                auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
                timing.reset();decision=controller.observe(p,d,f,now_ns=now,auxiliary_depth=auxiliary)
                phases=timing.snapshot();total=phases['controller.observe']['inclusive_ns']
                assert sum(row['exclusive_ns'] for row in phases.values())==total
                normalized=json.loads(json.dumps(decision,allow_nan=False))
                if normalized!=saved['decision']:
                    write_json(OUTPUT/'mismatch.json',dict(frame=i,decision=normalized))
                    raise ValueError('phase-instrumented complete decision differs at frame '+str(i))
                payload=json.dumps(normalized,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                stream.write(json.dumps(dict(frame=i,phases=phases,controller_wall_ms=total/1e6,
                    complete_decision_exact=True,decision_sha256=hashlib.sha256(payload).hexdigest()))+'\n');stream.flush()
                frames+=1
                if i%100==0:print('CONTROLLER_PHASE_TIMING_FRAME',i,flush=True)
        assert frames==960 and state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        assert not model._forward_hooks and not model._forward_pre_hooks
        verify_all(launch)
        write_json(OUTPUT/'result.json',dict(status='CONTROLLER_PHASE_TIMING_COMPLETE',frames=frames,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','phase_timings.jsonl')},
            complete_decisions_exact=True,model_state_unchanged=True,model_state_sha256=before,
            instrumentation_overhead_included=True,exclusive_time_partitions_controller_duration=True,
            model_hooks_removed=True,controlled_speed_comparison=False,wall_s=time.perf_counter()-started,
            hardware_after=hardware(),native_execution=False,model_training=False,
            navigation_qualified=False,real_time_qualified=False,goal_achieved=False))
        print('CONTROLLER_PHASE_TIMING_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='CONTROLLER_PHASE_TIMING_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()

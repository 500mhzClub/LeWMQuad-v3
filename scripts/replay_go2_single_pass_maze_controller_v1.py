"""Exact full ninth controller replay with single-pass measured-bound queries."""
import argparse
import json
import time
import cv2
import torch
from lewm.single_pass_later_floor_controller_development import SinglePassLaterFloorController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows, writer
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.replay_go2_later_floor_resolution_maze_prefix_v1 import CASE, FITS, CORRECTION

INPUT=BASE/'go2_later_floor_resolution_maze_pilot_v1_attempt_001'
BENCHMARK=BASE/'go2_single_pass_maze_queries_v1_attempt_001'
OUTPUT=BASE/'go2_single_pass_maze_controller_replay_v1_attempt_001'
PROTOCOL='docs/go2_single_pass_maze_controller_replay_v1_2026-09-09.md'
NATIVE_RESULT='3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755'
BENCHMARK_RESULT='570a3978c98b26d8c47d25a31f646c9278eae8221a86af2cd8ca6538e8991d79'


def verify_all(launch):
    verify(launch);verify_artifacts(INPUT,launch['native_input_sha256'])
    verify_artifacts(BENCHMARK,launch['benchmark_artifact_sha256'])
    admission=launch['correction_admission']
    verify_artifacts(FITS,admission['base_admission']['fit_artifact_sha256'])
    verify_artifacts(CORRECTION,admission['correction_artifact_sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive full controller replay required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    verify_artifacts(INPUT,{'result.json':NATIVE_RESULT});native=read_json(INPUT,'result.json')
    assert native['status']=='LATER_FLOOR_RESOLUTION_MAZE_PILOT_COMPLETE'
    ids={'result.json':NATIVE_RESULT,**native['artifact_sha256']}
    verify_artifacts(BENCHMARK,{'result.json':BENCHMARK_RESULT});benchmark=read_json(BENCHMARK,'result.json')
    assert benchmark['status']=='SINGLE_PASS_MAZE_QUERY_BENCHMARK_COMPLETE'
    bids={'result.json':BENCHMARK_RESULT,**benchmark['artifact_sha256']}
    old=read_json(INPUT,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_single_pass_maze_controller_v1.py',
        'lewm/tests/test_single_pass_later_floor_controller_development.py'),benchmark['source_sha256'])
    resources=hardware()
    launch=old|dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,native_input_sha256=ids,
        benchmark_artifact_sha256=bids,hardware=resources,frames=1881,
        implementation_class='SinglePassLaterFloorController',original_decision_labels_preserved=True,
        cpu_processes=1,numerical_threads=1,native_scene_workers=0,native_execution=False,model_training=False,
        minimum_available_ram_bytes=8*1024**3,output_allowance_bytes=1024**3,
        os_resource_limits_enforced=False,controlled_speed_comparison=False,native_adoption=False,
        concurrency_reason='one exact-performance replay beside the existing single native settled-boundary maze scene')
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=8*1024**3;storage_ok=resources['artifact_free_bytes']>=RESERVE_BYTES+1024**3
    if args.preflight_only:
        print('SINGLE_PASS_CONTROLLER_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('full controller replay resource admission failed')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('SINGLE_PASS_CONTROLLER_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);start=time.perf_counter()
    try:
        name,index,variant,condition,model_name=CASE
        model,c,v=load_assigned(launch['correction_admission'],model_name);assert(c,v)==(condition,variant)
        before=state_digest(model.state_dict());assert before==old['prefix_report']['model_state_sha256']
        controller=SinglePassLaterFloorController(model,ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index),navigation_ticks=NAVIGATION_TICKS,persistent=True,condition=c,variant=v)
        reader=IntentReturnRGBDReplay(INPUT/name);acquisitions=read_json(INPUT/name,'auxiliary_camera_audit.json')
        count=0;first_terminal=None
        with writer(OUTPUT) as append:
            for i,saved in enumerate(read_rows(INPUT/name)):
                if i>=1881 or saved['tick']!=i:raise ValueError('exact full recorded population required')
                p,d,f,now=reader.packet(i)
                auxiliary=packet(INPUT/name,i,p,public_acquisition(acquisitions[i]),now_ns=now)
                began=time.perf_counter_ns();candidate=controller.observe(p,d,f,now_ns=now,auxiliary_depth=auxiliary)
                elapsed=(time.perf_counter_ns()-began)/1e6
                normalized=json.loads(json.dumps(candidate,allow_nan=False))
                if normalized!=saved['decision']:
                    write_json(OUTPUT/'mismatch.json',dict(frame=i,decision=normalized))
                    raise ValueError('complete recorded decision differs at frame '+str(i))
                append(dict(tick=i,decision=normalized,complete_decision_exact=True,controller_wall_ms=elapsed));count+=1
                if first_terminal is None and normalized['terminal'] is not None:first_terminal=i
                if i%100==0:print('SINGLE_PASS_CONTROLLER_FRAME',i,flush=True)
        assert count==1881 and first_terminal==1870
        assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        verify_all(launch)
        write_json(OUTPUT/'result.json',dict(status='SINGLE_PASS_MAZE_CONTROLLER_REPLAY_COMPLETE',
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','context_decisions.jsonl.gz')},
            frames=count,first_terminal_frame=first_terminal,all_complete_decisions_exact=True,
            model_state_unchanged=True,model_state_sha256=before,original_failure_preserved=True,
            controlled_speed_comparison=False,native_adoption=False,native_execution=False,
            real_time_qualified=False,navigation_qualified=False,wall_s=time.perf_counter()-start,hardware_after=hardware()))
        print('SINGLE_PASS_CONTROLLER_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()

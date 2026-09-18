"""Fresh sensor/controller prefix with only declared settling differences."""
import argparse
import json
import time
import cv2
import torch
from lewm.settled_boundary_round_trip_development import SettledBoundaryRoundTripController
from lewm.settled_target_reset_prefix_comparison_development import compare_current
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows, writer
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts, artifact_path
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.replay_go2_later_floor_resolution_maze_prefix_v1 import CASE, FITS, CORRECTION
from scripts.replay_go2_measured_settling_mission_json_prefix_v1 import INPUT, OUTPUT as MISSION, BINDINGS

OUTPUT = BASE/'go2_settled_boundary_controller_prefix_v2_attempt_001'
PROTOCOL = 'docs/go2_settled_boundary_controller_prefix_v2_2026-09-09.md'
FAILED = BASE/'go2_settled_boundary_controller_prefix_v1_attempt_001'
FAILED_LAUNCH = 'cf320b97b50fd5e9d6369608d690147cdb1cce9bb15a19d54cfb89bc126f9f1e'
FAILED_RESULT = 'f7dedb15dabba46667624a63434b4ba0bcf01d7b294c501dc19a095bf5f138ef'

MISSION_RESULT = 'bf875306754c620d393d7a05496081c318c17d624726d2ae8bf348bab9caa053'


def verify_all(launch):
    verify(launch); verify_artifacts(INPUT, launch['prefix_input_sha256'])
    verify_artifacts(FAILED, launch['failed_attempt_artifact_sha256'])
    verify_artifacts(MISSION, launch['mission_artifact_sha256'])
    admission = launch['correction_admission']
    verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
    verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive fresh controller prefix required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    verify_artifacts(INPUT,BINDINGS);verify_artifacts(MISSION,{'result.json':MISSION_RESULT})
    result=read_json(MISSION,'result.json')
    assert result['frames']==1867 and result['first_mission_behavior_difference']==1866
    assert result['original_mission_receipts_exact'] and result['current_registered_pose_witnesses_validated']
    mids={'result.json':MISSION_RESULT,**result['artifact_sha256']};verify_artifacts(MISSION,mids)
    failed_ids={'launch.json':FAILED_LAUNCH,'failure.json':FAILED_RESULT}
    verify_artifacts(FAILED,failed_ids);failed=read_json(FAILED,'launch.json');verify(failed)
    failed_ids |= {n:digest(artifact_path(FAILED,n)) for n in ('mismatch.json','context_decisions.jsonl.gz')}
    assert read_json(FAILED,'mismatch.json')['frame']==1866
    old=read_json(INPUT,'launch.json');name,index,variant,condition,model_name=CASE
    names=['policy_observations.json','policy_histories.npz','depth_observations.json','fast_gyro_histories.npz',
        'auxiliary_camera_audit.json','command_tape.json']
    names += [n for i in range(1867) for n in (f'rgb_{i:04d}.png',f'depth_{i:04d}.npz',f'auxiliary_depth_{i:04d}.npz')]
    ids=BINDINGS|{name+'/'+n:digest(artifact_path(INPUT,name+'/'+n)) for n in names}
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_settled_boundary_controller_prefix_v2.py',
        'lewm/tests/test_settled_boundary_round_trip_development.py',
        'lewm/tests/test_settled_target_reset_prefix_comparison_development.py'),failed['source_sha256'])
    resources=hardware()
    launch=old|dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,prefix_input_sha256=ids,
        mission_artifact_sha256=mids,failed_attempt_artifact_sha256=failed_ids,hardware=resources,maximum_frames=1867,
        unchanged_controller_fresh_replay=True,declared_target_reset_mode_effect=True,
        native_audit_pending_at_launch=False,completed_native_audit_required_before_next_native=True,
        cpu_processes=1,numerical_threads=1,native_scene_workers=0,model_training=False,native_execution=False,
        minimum_available_ram_bytes=8*1024**3,output_allowance_bytes=1024**3,
        os_resource_limits_enforced=False,
        concurrency_reason='one fresh unchanged-controller replay beside the independent packed-owned controller replay')
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=8*1024**3
    storage_ok=resources['artifact_free_bytes']>=RESERVE_BYTES+1024**3
    if args.preflight_only:
        print('SETTLED_BOUNDARY_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('controller prefix resource admission failed')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('SETTLED_BOUNDARY_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);start=time.perf_counter()
    try:
        model,c,v=load_assigned(launch['correction_admission'],model_name);assert(c,v)==(condition,variant)
        before=state_digest(model.state_dict());assert before==old['prefix_report']['model_state_sha256']
        controller=SettledBoundaryRoundTripController(model,ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index),navigation_ticks=NAVIGATION_TICKS,persistent=True,condition=c,variant=v)
        directory=INPUT/name;reader=IntentReturnRGBDReplay(directory)
        acquisitions=read_json(directory,'auxiliary_camera_audit.json');tape=read_json(directory,'command_tape.json')
        frames=0;first_behavior=first_counter=None;last=None
        with writer(OUTPUT) as append:
            for i,saved in enumerate(read_rows(directory)):
                if i>=1867 or saved['tick']!=i:raise ValueError('bounded complete prefix required')
                p,d,f,now=reader.packet(i)
                auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
                candidate=json.loads(json.dumps(controller.observe(p,d,f,now_ns=now,auxiliary_depth=auxiliary),allow_nan=False))
                original=saved['decision']
                assert original['requested_command']==tape[i]['requested_command']
                try:comparison=compare_current(original,candidate,previous_candidate=last)
                except Exception:
                    write_json(OUTPUT/'mismatch.json',dict(frame=i,candidate=candidate));raise
                append(dict(tick=i,decision=candidate,comparison=comparison));frames+=1;last=candidate
                if comparison['quiet_counter_changed'] and first_counter is None:first_counter=i
                if i%100==0:print('SETTLED_BOUNDARY_PREFIX_FRAME',i,flush=True)
                if comparison['mission_behavior_differences']:
                    first_behavior=i;break
        assert frames==1867 and first_behavior==1866
        assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        verify_all(launch)
        write_json(OUTPUT/'result.json',dict(status='SETTLED_BOUNDARY_CONTROLLER_PREFIX_V2_COMPLETE',
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','context_decisions.jsonl.gz')},
            frames=frames,first_quiet_counter_difference=first_counter,first_mission_behavior_difference=first_behavior,
            final_mission_receipt=last['mission_receipt'],final_requested_command=last['requested_command'],
            complete_decisions_exact_outside_declared_mission_and_target_reset_fields=True,actual_requested_commands_exact=True,
            final_target_reset_mode_difference=comparison['delayed_target_reset_mode_difference'],
            original_failed_attempt_preserved=True,controller_implementation_unchanged=True,
            stopped_before_later_decisions=True,model_state_unchanged=True,model_state_sha256=before,
            native_audit_replaced=False,new_native_execution=False,navigation_qualified=False,
            wall_s=time.perf_counter()-start,hardware_after=hardware()))
        print('SETTLED_BOUNDARY_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()

"""New controller on closed tenth-episode bytes, stopping at first camera intervention."""
import argparse
import json
import time
import cv2
import torch
from lewm.dual_camera_settled_controller_development import DualCameraSettledController
from lewm.dual_camera_controller_prefix_comparison_development import compare_primary_decision
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, artifact_path, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.run_go2_settled_boundary_maze_pilot_v2 import OUTPUT as INPUT, CASE, verify_inputs as verify_native
from scripts.settled_boundary_maze_episode_development import artifacts as collection_artifacts
from scripts.replay_go2_dual_camera_registered_v1 import OUTPUT as REGISTERED, verify_all as verify_registered
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.maze_decision_stream_development import read_rows, writer

OUTPUT = BASE/'go2_dual_camera_controller_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_dual_camera_controller_prefix_v1_2026-09-09.md'
NATIVE_LAUNCH = 'eee7ddac0d0a5410806c7e16f6d5abc71ee2cf9ffb3772ed4f910333868de188'
COLLECTION_SHA = '2875fbdee5d5069de2ec3ef872f942e5a52b53d34eafccd88108cab45a3cd914'


def admit_registered(result):
    if (result['status'] != 'DUAL_CAMERA_REGISTERED_REPLAY_COMPLETE'
            or result['frames'] != 1881 or result['accepted_raw_poses'] != 1881
            or result['accepted_registered_poses'] != 1881 or result['exact_raw_observer_frames'] != 1881
            or result['exact_original_registered_prefix_frames'] != 1870
            or result['first_failure_frame'] is not None or result['all_frames_registered'] is not True):
        raise ValueError('completed full dual-camera motion/floor replay required')


def admit_collection(result):
    counts = dict(decisions=1883, rgbd_frames=1883, auxiliary_frames=1883, command_ticks=1882,
        completed_ticks=1882, physics_samples=94850, terminal_zero_ticks=10)
    if (result['status'] != 'SETTLED_BOUNDARY_MAZE_TERMINAL_AUDIT_REQUIRED'
            or result['layout_index'] != 0 or any(result[k] != v for k,v in counts.items())
            or result['physical_stop'] is not None or result['acquisition_stop'] is not None
            or result['schedule_terminal'] != 'SENSOR_OR_MODEL_FAILURE'):
        raise ValueError('complete fixed tenth collection required; native audit still separate')


def verify_all(launch):
    source_check(launch['source_sha256'])
    verify_artifacts(INPUT, launch['collected_input_sha256'])
    verify_native(read_json(INPUT, 'launch.json'))
    verify_artifacts(REGISTERED, launch['registered_artifact_sha256'])
    verify_registered(read_json(REGISTERED, 'launch.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registered-result-sha256',required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive controller prefix required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    name,index,variant,condition,model_name=CASE
    verify_artifacts(INPUT,{'launch.json':NATIVE_LAUNCH,name+'/result.json':COLLECTION_SHA})
    native=read_json(INPUT,'launch.json');collection=read_json(INPUT,name+'/result.json');admit_collection(collection)
    verify_artifacts(REGISTERED,{'result.json':args.registered_result_sha256})
    registered=read_json(REGISTERED,'result.json');admit_registered(registered)
    bindings={'launch.json':NATIVE_LAUNCH,name+'/result.json':COLLECTION_SHA}
    for n in collection_artifacts(index,collection):
        relative=name+'/'+n;bindings[relative]=digest(artifact_path(INPUT,relative))
    inherited=dict(native['source_sha256'])
    for path,h in registered['source_sha256'].items():
        if path in inherited and inherited[path]!=h:raise ValueError('frozen source witnesses disagree: '+path)
        inherited[path]=h
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_dual_camera_controller_prefix_v1.py',
        'lewm/tests/test_dual_camera_controller_prefix_comparison_development.py'),inherited)
    resources=hardware()
    launch=dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,
        collected_input_sha256=bindings,
        registered_artifact_sha256={'result.json':args.registered_result_sha256,**registered['artifact_sha256']},
        hardware=resources,maximum_frames=1883,correction_admission=native['correction_admission'],
        cpu_processes=1,numerical_threads=1,native_scene_workers=0,
        minimum_available_ram_bytes=8*1024**3,output_allowance_bytes=1024**3,
        os_resource_limits_enforced=False,native_execution=False,model_training=False,
        native_audit_validation_pending_for_this_prefix=True,
        completed_native_audit_and_matching_collection_bindings_required_before_next_native=True,
        concurrency_reason='closed-collection controller prefix beside existing native audit; no new scene')
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=8*1024**3
    storage_ok=resources['artifact_free_bytes']>=RESERVE_BYTES+1024**3
    if args.preflight_only:
        print('DUAL_CAMERA_CONTROLLER_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('controller prefix resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('DUAL_CAMERA_CONTROLLER_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);start=time.perf_counter()
    try:
        model,c,v=load_assigned(launch['correction_admission'],model_name);assert(c,v)==(condition,variant)
        before=state_digest(model.state_dict());assert before==native['prefix_report']['model_state_sha256']
        controller=DualCameraSettledController(model,ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index),navigation_ticks=NAVIGATION_TICKS,persistent=True,condition=c,variant=v)
        directory=INPUT/name;reader=IntentReturnRGBDReplay(directory)
        acquisitions=read_json(directory,'auxiliary_camera_audit.json');tape=read_json(directory,'command_tape.json')
        assert len(reader.frames)==1883 and len(tape)==1882
        exact_frames=0;intervention=None;last=None
        with writer(OUTPUT) as append:
            for i,saved in enumerate(read_rows(directory)):
                if i>=1883 or saved['tick']!=i:raise ValueError('bounded complete prefix required')
                p,d,f,now=reader.packet(i)
                image,auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
                candidate=json.loads(json.dumps(controller.observe(p,d,f,now_ns=now,
                    auxiliary_depth=auxiliary,auxiliary_rgb=image),allow_nan=False))
                original=saved['decision']
                if i<len(tape) and original['requested_command']!=tape[i]['requested_command']:
                    raise ValueError('recorded original command differs from actual tape')
                raw=candidate['original_visual_evidence'] or {};choice=raw.get('camera_selection') or {}
                if choice.get('auxiliary_attempted'):
                    intervention=dict(frame=i,original=original,candidate=candidate,
                        first_auxiliary_attempt=True,following_recorded_observations_consumed=False)
                    write_json(OUTPUT/'intervention.json',intervention)
                    append(dict(tick=i,decision=candidate,first_auxiliary_intervention=True));last=candidate;break
                try:comparison=compare_primary_decision(original,candidate,p,image,auxiliary,now_ns=now)
                except Exception:
                    write_json(OUTPUT/'mismatch.json',dict(frame=i,candidate=candidate));raise
                append(dict(tick=i,decision=candidate,comparison=comparison));exact_frames+=1
                if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size>1024**3:raise ValueError('prefix output allowance exceeded')
                if i%100==0:print('DUAL_CAMERA_CONTROLLER_PREFIX_FRAME',i,flush=True)
        if intervention is None:raise ValueError('declared auxiliary intervention not reached')
        assert exact_frames==intervention['frame']
        assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        verify_all(launch)
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json','context_decisions.jsonl.gz','intervention.json')}
        write_json(OUTPUT/'result.json',dict(status='DUAL_CAMERA_CONTROLLER_PREFIX_COMPLETE',
            source_sha256=sources,artifact_sha256=artifacts,frames=exact_frames+1,
            exact_primary_decision_frames=exact_frames,first_auxiliary_intervention_frame=intervention['frame'],
            all_preintervention_requested_commands_exact=True,
            complete_preintervention_decisions_exact_outside_added_modality_metadata=True,
            following_recorded_observations_consumed=False,model_state_unchanged=True,model_state_sha256=before,
            final_requested_command=last['requested_command'],final_terminal=last['terminal'],final_failure=last['failure'],
            final_registered_pose_available=last['evidence'] is not None,
            completed_native_audit_and_matching_collection_bindings_required_before_next_native=True,
            native_audit_replaced=False,native_execution=False,navigation_qualified=False,goal_achieved=False,
            wall_s=time.perf_counter()-start,hardware_after=hardware()))
        print('DUAL_CAMERA_CONTROLLER_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()

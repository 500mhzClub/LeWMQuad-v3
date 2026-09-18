"""Fixed-model shadow replay of the new primary sensor prefix, without aux input."""
import json
import time
import cv2
import numpy as np
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.training_bias_goal_probe_development import TrainingBiasGoalProbe
from lewm.recorded_command_prefix_comparison_development import compare
from lewm.physical_execution_development import rotation_xyzw
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.capture_go2_auxiliary_tilted_depth_prefix_integrity_v2 import OUTPUT as INPUT
from scripts.read_go2_auxiliary_tilted_depth_prefix_v1 import OUTPUT as READOUT
from scripts.run_go2_training_bias_goal_probe_v1 import OUTPUT as PRIOR,CORRECTION,FITS,CASES
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_visible_robot_primary_compatibility_v1_attempt_001'
PROTOCOL='docs/go2_visible_robot_primary_compatibility_v1_2026-09-08.md'
INPUT_SHA='e8de0873c71f50c5a01793c0464ba2014c1a7fe72f77daf47204eaeee1c8c38a'
READOUT_SHA='f381538cf2f67afa6e654494027409178905ce625ef48f69a35366d26c912cb1'
PRIOR_SHA='5e48672a074d2086d734d02844af499d49d0a57571b9148b433b65ad2bedcfb5'


def replay(admission,model_name,reader):
    model,condition,variant=load_assigned(admission,model_name)
    before=state_digest(model.state_dict());controller=TrainingBiasGoalProbe(model,ArticulatedCollisionGeometry(URDF),
        condition=condition,variant=variant,persistent=True)
    rows=[]
    for i in range(len(reader.frames)):
        p,d,f,now=reader.packet(i);rows.append(controller.observe(p,d,f,now_ns=now))
    assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
    return json.loads(json.dumps(rows)),before


def main():
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive compatibility replay required')
    bound=[]
    for root,sha,status in ((INPUT,INPUT_SHA,'AUXILIARY_TILTED_DEPTH_PREFIX_INTEGRITY_V2_COMPLETE'),
            (READOUT,READOUT_SHA,'AUXILIARY_TILTED_DEPTH_PREFIX_READOUT_COMPLETE'),
            (PRIOR,PRIOR_SHA,'TRAINING_BIAS_GOAL_PROBE_COMPLETE')):
        verify_artifacts(root,{'result.json':sha});result=read_json(root,'result.json');assert result['status']==status
        ids={'result.json':sha,**result.get('artifact_sha256',{})}
        if 'launch_sha256' in result:ids['launch.json']=result['launch_sha256']
        verify_artifacts(root,ids);bound.append((root,ids,result))
    original=read_json(PRIOR,'launch.json');admission=original['correction_admission'];old=read_json(INPUT,'launch.json');verify(old);verify(original)
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_visible_robot_primary_compatibility_v1.py',
        'lewm/tests/test_recorded_command_prefix_comparison_development.py',
        'docs/go2_auxiliary_tilted_depth_prefix_result_2026-09-08.md'),bound[1][2]['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('bounded compatibility replay resource allowance unavailable')
    launch=old|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=resources,
        compatibility_inputs={str(p):ids for p,ids,_ in bound},correction_admission=admission,
        assigned_cases=[list(c) for c in CASES],native_execution=False,model_training=False,
        auxiliary_depth_used=False,shadow_replay_only=True,complete_replay_passes_per_model=2)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('VISIBLE_ROBOT_PRIMARY_COMPATIBILITY_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);start=time.perf_counter()
    try:
        reader=IntentReturnRGBDReplay(INPUT/'sensor_prefix');assert len(reader.frames)==20
        tape=read_json(INPUT/'sensor_prefix','command_tape.json');assert len(tape)==19
        with np.load(INPUT/'sensor_prefix/physics_trace.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        actual_pose=raw['base_pose_world'];R0=rotation_xyzw(actual_pose[749,3:]);reports=[];names=['launch.json']
        for case,trial,variant,condition,model_name in CASES:
            reference=[r['decision'] for r in read_json(PRIOR/case,'context_decisions.json')[:20]]
            with np.load(PRIOR/case/'physics_trace.npz',allow_pickle=False) as z:
                assert set(z.files)==set(raw)
                for k,v in raw.items():assert np.array_equal(v,z[k][:len(v)]),(case,k,'exact physical prefix')
            records,state=replay(admission,model_name,reader);again,state2=replay(admission,model_name,reader)
            assert records==again and state==state2,'independent fresh model/controller replay must be exact'
            name=model_name+'_decisions.json';write_json(OUTPUT/name,records);names.append(name)
            comparison=compare(reference,records,[t['requested_command'] for t in tape]);pose_rows=[]
            for i,r in enumerate(records):
                e=r['evidence'];pose=e.get('current_pose') if e else None;old_pose=reference[i]['evidence']['current_pose']
                measured=(R0.T@(actual_pose[749+50*i,:3]-actual_pose[749,:3]))[:2]
                pose_rows.append(dict(frame=i,controller_failure=r['failure'],terminal=r['terminal'],
                    observer_terminal_failure=e.get('terminal_failure') if e else None,current_pose_present=pose is not None,
                    observed_xy_error_m=None if pose is None else float(np.linalg.norm(np.asarray(pose['position_initial_body_m'])[:2]-measured)),
                    predecessor_xy_error_m=float(np.linalg.norm(np.asarray(old_pose['position_initial_body_m'])[:2]-measured))))
            forecasts=[]
            for i in range(comparison['common_executed_prefix_observations']):
                a=(reference[i]['new_selection'] or {}).get('prediction');b=(records[i]['new_selection'] or {}).get('prediction')
                if a is not None and b is not None:
                    delta=np.abs(np.asarray(a)-np.asarray(b));forecasts.append(dict(frame=i,maximum_xy_difference_m=float(delta[...,:2].max()),
                        maximum_yaw_channel_difference=float(delta[...,2:4].max()),maximum_contact_logit_difference=float(delta[...,4].max())))
            reports.append(dict(case=case,model=model_name,model_state_sha256=state,model_state_unchanged=True,
                exact_fresh_replay_passes=2,command_comparison=comparison,pose_checks=pose_rows,
                common_prefix_forecast_differences=forecasts,
                observer_and_controller_admitted_all_frames=all(r['current_pose_present'] and r['controller_failure'] is None
                    and r['observer_terminal_failure'] is None for r in pose_rows)))
            print('PRIMARY_COMPATIBILITY_MODEL',model_name,comparison,flush=True)
        verify(launch)
        for root,ids,_ in bound:verify_artifacts(root,ids)
        verify_artifacts(CORRECTION,admission['correction_artifact_sha256']);verify_artifacts(FITS,admission['base_admission']['fit_artifact_sha256'])
        write_json(OUTPUT/'result.json',dict(status='VISIBLE_ROBOT_PRIMARY_COMPATIBILITY_COMPLETE',
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in names},conditions=reports,
            wall_s=time.perf_counter()-start,hardware_after=hardware(),native_execution=False,model_training=False,
            auxiliary_depth_used=False,shadow_replay_only=True,prospective_closed_loop=False,
            full_mission_compatibility_established=False,navigation_qualified=False,goal_achieved=False))
        print('VISIBLE_ROBOT_PRIMARY_COMPATIBILITY_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_PRIMARY_COMPATIBILITY_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()

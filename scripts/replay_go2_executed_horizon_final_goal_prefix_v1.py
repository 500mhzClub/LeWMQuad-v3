"""Fresh reobservation controllers, bounded at their first unexecuted command."""
import json
import time
import cv2
import torch
from lewm.executed_horizon_final_goal_development import ExecutedHorizonFinalGoalProbe,score_final_goal
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.auxiliary_downward45_packet_replay_development import packet,public_acquisition
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_exact_mission_target_goal_probe_v1 import OUTPUT as INPUT,CASES,CORRECTION,FITS
from scripts.read_go2_exact_mission_target_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json

OUTPUT=BASE/'go2_executed_horizon_final_goal_prefix_v1_attempt_001'
PROTOCOL='docs/go2_executed_horizon_final_goal_prefix_v1_2026-09-08.md'
INPUT_SHA='d9e0cef6a7e66459a5cc6d21cc6cf6e638666a33a12cdcfb8b2364f1477bd125'
READOUT_SHA='d13c20027d16f48e146a542c443af61e205e02420bcf27e7e7cfa2ace7d61a43'


def replay(admission,case):
    name,trial,variant,condition,model_name=case;directory=INPUT/name
    model,c,v=load_assigned(admission,model_name);assert (c,v)==(condition,variant)
    before=state_digest(model.state_dict())
    controller=ExecutedHorizonFinalGoalProbe(model,ArticulatedCollisionGeometry(URDF),condition=c,variant=v,persistent=True)
    reader=IntentReturnRGBDReplay(directory);original=read_json(directory,'context_decisions.json')
    acquisitions=read_json(directory,'auxiliary_camera_audit.json');tape=read_json(directory,'command_tape.json')
    assert len(reader.frames)==len(original)==len(acquisitions)==len(tape)+1
    records=[];first_terminal=None;first_command=None
    for i in range(len(reader.frames)):
        p,d,f,now=reader.packet(i);aux=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
        r=json.loads(json.dumps(controller.observe(p,d,f,now_ns=now,auxiliary_depth=aux)))
        old=original[i]['decision']
        for key in ('evidence','memory_receipt','observed_goal_distance_m','auxiliary_floor_partition_receipt'):
            assert r[key]==old[key],('unchanged causal observation and map',i,key)
        a,b=old['new_selection'],r['new_selection']
        assert (a is None)==(b is None)
        if a is not None:
            assert b==score_final_goal(a),('exact final-goal score transformation',i)
        if a is not None and 'prediction' in a:
            assert a['prediction']==b['prediction'],('unchanged learned forecasts',i)
            assert a['nominal_action_checks']==b['nominal_action_checks']
            assert a['nominal_path_checks']==b['nominal_path_checks']
            for old_check,new_check in zip(a['surface_checks'],b['surface_checks'],strict=True):
                assert old_check==new_check,('unchanged surface checks',i)
        if old['terminal']!=r['terminal']:first_terminal=i
        if i<len(tape) and r['requested_command']!=tape[i]['requested_command']:first_command=i
        records.append(r)
        if first_terminal is not None or first_command is not None:break
    assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
    return dict(decisions=records,model_state_sha256=before,first_terminal_policy_difference=first_terminal,
        first_requested_command_difference=first_command,
        replay_stopped_at_first_command_or_terminal_intervention=first_command is not None or first_terminal is not None,
        recorded_tape_bound_reached=len(records)==len(reader.frames),
        causal_observations_maps_forecasts_and_nominal_checks_exact=True,surface_checks_exact=True,only_final_goal_pose_scoring_horizon_changed=True,
        unexecuted_outcomes_inferred=False)


def main():
    if not __debug__:raise ValueError('assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive replay required')
    bound=[]
    for root,sha,status in ((INPUT,INPUT_SHA,'EXACT_MISSION_TARGET_GOAL_PROBE_COMPLETE'),
            (READOUT,READOUT_SHA,'EXACT_MISSION_TARGET_GOAL_READOUT_COMPLETE')):
        verify_artifacts(root,{'result.json':sha});r=read_json(root,'result.json');assert r['status']==status
        ids={'result.json':sha,**r.get('artifact_sha256',{})}
        if 'launch_sha256' in r:ids['launch.json']=r['launch_sha256']
        verify_artifacts(root,ids);bound.append((root,ids,r))
    old=read_json(INPUT,'launch.json');verify(old);admission=old['correction_admission']
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_executed_horizon_final_goal_prefix_v1.py',
        'lewm/tests/test_executed_horizon_final_goal_development.py',
        'docs/go2_exact_mission_target_goal_probe_result_2026-09-08.md'),bound[-1][2]['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('replay resource allowance unavailable')
    launch=old|dict(source_sha256=sources,output_root=str(OUTPUT),protocol=PROTOCOL,hardware=resources,
        replay_input_bindings={str(p):ids for p,ids,_ in bound},native_execution=False,model_training=False,
        shadow_replay_only=True,complete_fresh_replays_per_model=2,workers=1)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    print('EXECUTED_HORIZON_FINAL_GOAL_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        reports=[];names=['launch.json']
        for case in CASES:
            first=replay(admission,case);second=replay(admission,case);assert first==second,'exact fresh replay required'
            name=case[4]+'_decisions.json';write_json(OUTPUT/name,first);names.append(name)
            rows=first['decisions']
            report={k:v for k,v in first.items() if k!='decisions'}|dict(case=case[0],model=case[4],
                exact_fresh_replay_passes=2,model_state_unchanged=True,frames=len(rows),
                wait_ticks=[r['tick'] for r in rows if r['infeasible_wait_active']],
                final_requested_command=rows[-1]['requested_command'],final_terminal=rows[-1]['terminal'],
                final_recovery_count=rows[-1]['feasible_action_recoveries'],
                changed_score_ticks=[i for i,r in enumerate(rows)
                    if (r['new_selection'] or {}).get('final_goal_execution_horizon_scoring')],
                controller_failures=[dict(frame=i,failure=r['failure']) for i,r in enumerate(rows) if r['failure']])
            reports.append(report);print('EXECUTED_HORIZON_FINAL_GOAL_MODEL',report,flush=True)
        verify(launch)
        for root,ids,_ in bound:verify_artifacts(root,ids)
        verify_artifacts(CORRECTION,admission['correction_artifact_sha256']);verify_artifacts(FITS,admission['base_admission']['fit_artifact_sha256'])
        bindings={n:digest(OUTPUT/n) for n in names};verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='EXECUTED_HORIZON_FINAL_GOAL_PREFIX_COMPLETE',source_sha256=sources,
            artifact_sha256=bindings,conditions=reports,wall_s=time.perf_counter()-started,hardware_after=hardware(),
            native_execution=False,model_training=False,shadow_replay_only=True,prospective_recovery_verified=False,
            navigation_qualified=False,goal_achieved=False))
        print('EXECUTED_HORIZON_FINAL_GOAL_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_EXECUTED_HORIZON_FINAL_GOAL_PREFIX_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()


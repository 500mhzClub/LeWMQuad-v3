"""Fresh reobservation controllers, bounded at their first unexecuted command."""
import json
import time
import cv2
import torch
from lewm.auxiliary_depth_reobserve_goal_probe_development import AuxiliaryDepthReobserveGoalProbe
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.auxiliary_depth_packet_replay_development import packet,public_acquisition
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_auxiliary_depth_goal_probe_v1 import OUTPUT as INPUT,CASES,CORRECTION,FITS
from scripts.read_go2_auxiliary_depth_goal_probe_v1 import OUTPUT as READOUT
from scripts.diagnose_go2_auxiliary_depth_executed_motion_v1 import OUTPUT as DIAGNOSTIC,INPUT_SHA,READOUT_SHA
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json

OUTPUT=BASE/'go2_auxiliary_depth_reobserve_prefix_v1_attempt_001'
PROTOCOL='docs/go2_auxiliary_depth_reobserve_prefix_v1_2026-09-08.md'
DIAGNOSTIC_SHA='8e04661518182c57f897fb60ef39f5bcefc52712ed9eede6c5e0555145436d59'


def replay(admission,case):
    name,trial,variant,condition,model_name=case;directory=INPUT/name
    model,c,v=load_assigned(admission,model_name);assert (c,v)==(condition,variant)
    before=state_digest(model.state_dict())
    controller=AuxiliaryDepthReobserveGoalProbe(model,ArticulatedCollisionGeometry(URDF),condition=c,variant=v,persistent=True)
    reader=IntentReturnRGBDReplay(directory);original=read_json(directory,'context_decisions.json')
    acquisitions=read_json(directory,'auxiliary_camera_audit.json');tape=read_json(directory,'command_tape.json')
    assert len(reader.frames)==len(original)==len(acquisitions)==len(tape)+1
    records=[];first_terminal=None;first_command=None
    for i in range(len(reader.frames)):
        p,d,f,now=reader.packet(i);aux=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
        r=json.loads(json.dumps(controller.observe(p,d,f,now_ns=now,auxiliary_depth=aux)))
        old=original[i]['decision']
        if first_terminal is None and old['terminal']!=r['terminal']:first_terminal=i
        if first_terminal is None:
            assert {k:r[k] for k in old if k!='controller'}=={k:v for k,v in old.items() if k!='controller'},('preintervention identity',i)
        records.append(r)
        if i<len(tape) and r['requested_command']!=tape[i]['requested_command']:
            first_command=i;break
    assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
    return dict(decisions=records,model_state_sha256=before,first_terminal_policy_difference=first_terminal,
        first_requested_command_difference=first_command,
        replay_stopped_before_executing_changed_command=first_command is not None,
        recorded_tape_bound_reached=first_command is None,
        original_decisions_exact_before_terminal_intervention=True,
        unexecuted_outcomes_inferred=False)


def main():
    if not __debug__:raise ValueError('assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive replay required')
    bound=[]
    for root,sha,status in ((INPUT,INPUT_SHA,'AUXILIARY_DEPTH_GOAL_PROBE_COMPLETE'),
            (READOUT,READOUT_SHA,'AUXILIARY_DEPTH_GOAL_READOUT_COMPLETE'),
            (DIAGNOSTIC,DIAGNOSTIC_SHA,'AUXILIARY_DEPTH_EXECUTED_MOTION_DIAGNOSIS_COMPLETE')):
        verify_artifacts(root,{'result.json':sha});r=read_json(root,'result.json');assert r['status']==status
        ids={'result.json':sha,**r.get('artifact_sha256',{})}
        if 'launch_sha256' in r:ids['launch.json']=r['launch_sha256']
        verify_artifacts(root,ids);bound.append((root,ids,r))
    old=read_json(INPUT,'launch.json');verify(old);admission=old['correction_admission']
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_auxiliary_depth_reobserve_prefix_v1.py',
        'lewm/tests/test_auxiliary_depth_reobserve_goal_probe_development.py',
        'docs/go2_auxiliary_depth_executed_motion_result_2026-09-08.md'),bound[-1][2]['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('replay resource allowance unavailable')
    launch=old|dict(source_sha256=sources,output_root=str(OUTPUT),protocol=PROTOCOL,hardware=resources,
        replay_input_bindings={str(p):ids for p,ids,_ in bound},native_execution=False,model_training=False,
        shadow_replay_only=True,complete_fresh_replays_per_model=2,workers=1)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    print('AUXILIARY_REOBSERVE_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
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
                controller_failures=[dict(frame=i,failure=r['failure']) for i,r in enumerate(rows) if r['failure']])
            reports.append(report);print('AUXILIARY_REOBSERVE_MODEL',report,flush=True)
        verify(launch)
        for root,ids,_ in bound:verify_artifacts(root,ids)
        verify_artifacts(CORRECTION,admission['correction_artifact_sha256']);verify_artifacts(FITS,admission['base_admission']['fit_artifact_sha256'])
        bindings={n:digest(OUTPUT/n) for n in names};verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='AUXILIARY_DEPTH_REOBSERVE_PREFIX_COMPLETE',source_sha256=sources,
            artifact_sha256=bindings,conditions=reports,wall_s=time.perf_counter()-started,hardware_after=hardware(),
            native_execution=False,model_training=False,shadow_replay_only=True,prospective_recovery_verified=False,
            navigation_qualified=False,goal_achieved=False))
        print('AUXILIARY_REOBSERVE_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUXILIARY_REOBSERVE_PREFIX_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()

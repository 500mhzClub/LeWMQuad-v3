"""Prospective later-measurement contact interpretation on an audited trajectory."""
import argparse
import json
import time
import cv2
import torch
from lewm.later_floor_resolution_controller_development import LaterFloorResolutionRoundTripController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows, writer
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_joint_floor_registered_maze_pilot_v1 import OUTPUT as INPUT, CASE, FITS, CORRECTION

OUTPUT = BASE/'go2_later_floor_resolution_maze_prefix_v1_attempt_001'
READOUT = BASE/'go2_joint_floor_registered_maze_readout_v1_attempt_001'
CACHE = BASE/'go2_frame_cached_floor_prefix_v1_attempt_001'
DIAGNOSIS = BASE/'go2_retained_floor_resolution_diagnosis_v1_attempt_001'
PROTOCOL = 'docs/go2_later_floor_resolution_maze_prefix_v1_2026-09-09.md'
RESULTS = {
    INPUT: 'f7a6ef564d6e2bfac8b259e2ddc1988e5b357d8be7f180512f8b00393bbdfe41',
    READOUT: '5665f51fbb5315d2467494b295ef3c8ba4f4eecacb1629c5b70bdd6b78ae0390',
    CACHE: '633a16730506480011d9a00a4a76c63daa9c62c19eaf39695a3376abfc97e14b',
    DIAGNOSIS: '3d9a5d3109eff7fb9f8e38acf347ac13a631ef0e643679883eeed3db468b9912',
}


def compare_current(old, candidate):
    if candidate['failure'] is not None: raise ValueError('candidate admission failure: '+str(candidate['failure']))
    for field in ('evidence','original_visual_evidence','memory_receipt','mission_receipt'):
        if candidate[field] != old[field]: raise ValueError('original observation/map/mission changed: '+field)
    a, b = old['new_selection'], candidate['new_selection']
    if a and b and 'prediction' in a and 'prediction' in b:
        if a['prediction'] != b['prediction']: raise ValueError('original raw forecasts changed')
        if len(a['surface_checks']) != len(b['surface_checks']): raise ValueError('same ordered candidate contact queries required')
        for x, y in zip(a['surface_checks'], b['surface_checks'], strict=True):
            if y['original_contact_check_before_later_floor_resolution'] != x:
                raise ValueError('original contact query changed')
        if 'nominal_path_checks' in a and 'nominal_path_checks' in b and a['nominal_path_checks'] != b['nominal_path_checks']:
            raise ValueError('original full nominal path checks changed')
    return dict(command_changed=old['requested_command'] != candidate['requested_command'],
        terminal_changed=old['terminal'] != candidate['terminal'])


def verify_inputs(launch):
    verify(launch)
    for root, bindings in launch['replay_input_bindings'].items(): verify_artifacts(root, bindings)
    admission = launch['correction_admission']
    verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
    verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive prospective contact interpretation required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    inputs = {}; bindings = {}; inherited = {}
    for root, sha in RESULTS.items():
        verify_artifacts(root, {'result.json': sha}); result = read_json(root,'result.json'); inputs[root] = result
        ids = {'result.json':sha, **result.get('artifact_sha256',{})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root,ids); bindings[str(root)] = ids
        for path,h in result['source_sha256'].items():
            if path in inherited and inherited[path] != h: raise ValueError('incompatible frozen input source identities')
            inherited[path] = h
    assert inputs[INPUT]['status'] == 'JOINT_FLOOR_REGISTERED_MAZE_PILOT_COMPLETE'
    assert inputs[READOUT]['native_result_sha256'] == RESULTS[INPUT]
    assert inputs[CACHE]['complete_decisions_exact'] and inputs[CACHE]['model_state_unchanged']
    assert inputs[DIAGNOSIS]['status'] == 'RETAINED_FLOOR_RESOLUTION_DIAGNOSIS_COMPLETE'
    audit = read_json(INPUT,CASE[0]+'_audit.json')
    for key in ('raw_sensor_reconstruction_pass','raw_model_command_replay_pass','raw_command_audit_pass','model_state_unchanged'):
        assert audit[key] is True
    old = read_json(INPUT,'launch.json')
    sources = discover_sources((PROTOCOL,'scripts/replay_go2_later_floor_resolution_maze_prefix_v1.py',
        'lewm/tests/test_later_floor_evidence_development.py',
        'lewm/tests/test_later_floor_resolution_prefix_development.py',
        'docs/go2_joint_floor_registered_maze_pilot_result_2026-09-09.md',
        'docs/go2_retained_floor_resolution_diagnosis_result_2026-09-09.md',
        'docs/go2_frame_cached_floor_prefix_result_2026-09-09.md'),inherited)
    resources = hardware()
    launch = old | dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,
        replay_input_bindings=bindings,hardware=resources,completed_predecessor_raw_audit_reused=True,
        predecessor_controller_rerun=False,implementation_class='LaterFloorResolutionRoundTripController',
        native_execution=False,model_training=False,native_scene_workers=0,cpu_processes=1,
        numerical_threads=1,minimum_available_ram_bytes=8*1024**3,output_allowance_bytes=1024**3,
        concurrency_reason='one fresh candidate; completed predecessor raw audit reused',
        os_resource_limits_enforced=False,contact_interpretation_is_declared_intervention=True)
    verify_inputs(launch)
    memory_ok = resources['memory_available_bytes'] >= 8*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+1024**3
    if args.preflight_only:
        print('LATER_FLOOR_RESOLUTION_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True); return
    if not memory_ok or not storage_ok: raise ValueError('prospective replay resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('LATER_FLOOR_RESOLUTION_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter()
    try:
        name,index,variant,condition,model_name=CASE
        model,c,v=load_assigned(launch['correction_admission'],model_name);assert(c,v)==(condition,variant)
        before=state_digest(model.state_dict());assert before==inputs[CACHE]['model_state_sha256']
        controller=LaterFloorResolutionRoundTripController(model,ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index),navigation_ticks=NAVIGATION_TICKS,persistent=True,condition=c,variant=v)
        directory=INPUT/name;reader=IntentReturnRGBDReplay(directory)
        acquisitions=read_json(directory,'auxiliary_camera_audit.json');tape=read_json(directory,'command_tape.json')
        frames=0;first_command=first_terminal=first_resolution=None;last=prior=None
        with writer(OUTPUT) as append:
            for i, saved in enumerate(read_rows(directory)):
                assert saved['tick']==i
                p,d,f,now=reader.packet(i)
                auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
                prior=saved['decision']
                last=json.loads(json.dumps(controller.observe(p,d,f,now_ns=now,auxiliary_depth=auxiliary),allow_nan=False))
                try: change=compare_current(prior,last)
                except (ValueError,TypeError,KeyError,IndexError) as error:
                    write_json(OUTPUT/'terminal_decision_diagnostic.json',dict(tick=i,candidate_decision=last,reason=repr(error)))
                    raise
                if i<len(tape) and prior['requested_command']!=tape[i]['requested_command']:
                    raise ValueError('original request differs from executed tape')
                selection=last['new_selection'] or {}
                resolved=sum(q['resolved_intersections'] for s in selection.get('surface_checks',[])
                    for q in s.get('later_floor_contact_resolution',[]))
                if resolved and first_resolution is None:first_resolution=i
                append(dict(tick=i,decision=last,comparison=change,completed_predecessor_raw_audit_reused=True))
                frames+=1
                if change['command_changed']:first_command=i
                if change['terminal_changed']:first_terminal=i
                if i%100==0:print('LATER_FLOOR_RESOLUTION_FRAME',i,flush=True)
                if first_command is not None or first_terminal is not None:break
        assert frames and state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
        verify_inputs(launch)
        report=dict(frames=frames,first_requested_command_difference=first_command,
            first_terminal_policy_difference=first_terminal,first_any_candidate_contact_resolution=first_resolution,
            final_requested_command=last['requested_command'],prior_requested_command=prior['requested_command'],
            final_terminal=last['terminal'],changed_selection=last['new_selection'],
            completed_predecessor_raw_audit_reused=True,predecessor_controller_rerun=False,
            original_visual_pose_map_mission_exact=True,raw_forecasts_and_original_contact_queries_exact=True,
            full_nominal_path_checks_unchanged=True,contact_interpretation_is_declared_intervention=True,
            model_state_unchanged=True,model_state_sha256=before,stopped_before_unexecuted_outcome=True,
            unexecuted_outcomes_inferred=False,new_native_navigation_verified=False)
        write_json(OUTPUT/'result.json',dict(status='LATER_FLOOR_RESOLUTION_PREFIX_COMPLETE',report=report,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','context_decisions.jsonl.gz')},
            wall_s=time.perf_counter()-started,hardware_after=hardware(),native_execution=False,model_training=False,
            navigation_qualified=False,goal_achieved=False))
        print('LATER_FLOOR_RESOLUTION_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='LATER_FLOOR_RESOLUTION_PREFIX_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()

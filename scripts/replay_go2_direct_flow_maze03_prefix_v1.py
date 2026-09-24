"""Fresh unchanged controller through the fixed maze3 failed observation; no following tape."""
import argparse
from itertools import islice
import json
import shutil
import time
import cv2
import torch
from lewm.direct_flow_floor_transport_controller_development import DirectFlowFloorTransportController
from lewm.direct_flow_maze03_prefix_development import compare_step, MAX_FRAMES, BOUNDARY_FRAME
from lewm.direct_flow_live_replay_validation_development import validate_live
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.run_go2_independent_floor_transport_mazes_v1 import OUTPUT as INPUT, verify_inputs as verify_cohort
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_direct_flow_maze03_prefix_v1_attempt_001'
PROTOCOL='docs/go2_direct_flow_maze03_prefix_v1_2026-09-09.md'
INPUT_SHA='a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720'
CASE=('full_jepa_novel_maze_03',3,'full','jepa','seed_2026091001_full_jepa')
MAX_OUTPUT_BYTES=256*1024**2
PRIOR=BASE/'go2_direct_flow_maze01_prefix_v2_attempt_001'
PRIOR_SHA='345b54f1b3c647be516040f2b8bf03b4ab3c709b9737dc0bdf7c78bedcacdbe3'
FAILED=BASE/'go2_direct_flow_maze01_prefix_v1_attempt_001'
FAILED_BINDINGS={
    'launch.json':'4d9900f40d39eb40d7aa1740b99838756b7dc48cb595fafacae5c8b01fc72d2f',
    'context_decisions.jsonl.gz':'d4d90b933d07f18ce86d3dac86763d002a6b0f1e3fccdc4bd79801fadc656521',
    'failure.json':'3eae1b519549f7762a3d353cf430ee9856863b9dabf6a233c186aea6a66c87e5',
}


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA})
    verify_artifacts(PRIOR,read_json(PRIOR,'result.json')['artifact_sha256'])
    verify(read_json(PRIOR,'launch.json'))
    verify_artifacts(FAILED,FAILED_BINDINGS)
    verify(read_json(FAILED,'launch.json'))
    verify_artifacts(INPUT,launch['replay_input_bindings'])
    verify_cohort(read_json(INPUT,'launch.json'))


def admit(result,audit,old):
    if (result['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_MAZES_V1_COMPLETE'
            or result['all_fixed_cases_executed'] is not True
            or [r['layout_index'] for r in result['conditions']] != [1,2,3]
            or old['model_state_sha256'] != MODEL_STATE or list(CASE) not in old['planned_cases']
            or old['implementation_class'] != 'MeasuredFloorTransportController'):
        raise ValueError('complete fixed cohort and original assigned controller required')
    record=result['conditions'][2]
    if (record['case'] != CASE[0] or record['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
            or audit['layout_index'] != 3 or audit['verified_round_trip'] != record['verified_round_trip']):
        raise ValueError('completed maze3 raw-audited population required')
    for key in ('raw_sensor_reconstruction_pass','additional_auxiliary_rgb_reconstructed',
                'raw_model_command_replay_pass','raw_command_audit_pass','model_state_unchanged'):
        if audit[key] is not True: raise ValueError('completed raw audit required: '+key)


def replay(launch):
    model,condition,variant=load_assigned(launch['correction_admission'],CASE[4])
    before=state_digest(model.state_dict())
    if (condition,variant) != (CASE[3],CASE[2]) or before != MODEL_STATE:
        raise ValueError('exact original assigned model required')
    controller=DirectFlowFloorTransportController(model,ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS,public_mission=public_mission(3),
        condition=condition,variant=variant,persistent=True)
    directory=INPUT/CASE[0]; reader=IntentReturnRGBDReplay(directory)
    acquisitions=read_json(directory,'auxiliary_camera_audit.json'); tape=read_json(directory,'command_tape.json')
    if not len(reader.frames)==len(acquisitions)==len(tape)+1 or len(tape)<MAX_FRAMES:
        raise ValueError('complete paired observations and actual command population required')
    frames=forecasts=exact_decisions=0; last=check=old=None
    with writer(OUTPUT) as append:
        for i,original in enumerate(islice(read_rows(directory),MAX_FRAMES)):
            if original['tick'] != i or original['pre_sample_index'] != 749+50*i or original['observation_index'] != i:
                raise ValueError('ordered actual observation endpoints required')
            if shutil.disk_usage(BASE).free<RESERVE_BYTES+MAX_OUTPUT_BYTES:
                raise ValueError('replay storage reserve unavailable')
            policy,depth,fast,now=reader.packet(i)
            image,auxiliary=packet(directory,i,policy,public_acquisition(acquisitions[i]),now_ns=now)
            inputs=fingerprint((policy,depth,fast,auxiliary,image,now))
            live=controller.observe(policy,depth,fast,now_ns=now,
                auxiliary_depth=auxiliary,auxiliary_rgb=image)
            last=json.loads(json.dumps(live,allow_nan=False))
            if fingerprint((policy,depth,fast,auxiliary,image,now)) != inputs:
                raise ValueError('controller mutated public input arrays')
            if not tape[i]['completed']: raise ValueError('original command was not completely dispatched')
            old=original['decision']
            # Persist every candidate, including an unexpected comparator failure.
            try:
                check=compare_step(old,last,tape[i]['requested_command'],frame=i)
                validate_live(live,last,check,policy,image,auxiliary,now_ns=now)
            except Exception as error:
                append(dict(tick=i,decision=last,comparison_failure=repr(error),
                    original_requested_command=tape[i]['requested_command']))
                raise
            forecasts+=int(check['raw_model_forecasts_compared'])
            exact_decisions+=int(check['complete_original_decision_exact'])
            append(dict(tick=i,decision=last,comparison=check,
                original_requested_command=tape[i]['requested_command'],public_input_arrays_unchanged=True))
            frames+=1
            if (OUTPUT/DECISIONS).stat().st_size>MAX_OUTPUT_BYTES//2:
                raise ValueError('compressed replay output headroom exceeded')
            if i%32==0: print('DIRECT_FLOW_MAZE03_PREFIX_FRAME',i,flush=True)
            if check['stop']: break
    if frames != MAX_FRAMES or not check or not check['boundary_reached']:
        raise ValueError('complete fixed prefix through original failed observation required')
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged weights and absent gradients required')
    return dict(case=CASE[0],layout_index=3,frames=frames,maximum_frames=MAX_FRAMES,boundary_frame=BOUNDARY_FRAME,
        exact_original_decisions=exact_decisions,raw_model_forecast_comparisons=forecasts,
        original_actual_commands_before_intervention_exact=True,
        prior_commands_compared=BOUNDARY_FRAME,boundary_comparison=check,
        final_requested_command=last['requested_command'],prior_requested_command=old['requested_command'],
        final_terminal=last['terminal'],final_failure=last['failure'],
        fallback_receipt=last['original_visual_evidence'].get('direct_corner_flow_fallback'),
        final_visual_status=last['original_visual_evidence']['status'],
        full_controller_recovered_at_boundary=check['controller_recovered'],
        stopped_at_original_failed_observation=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,model_state_sha256=before,model_state_unchanged=True,
        unexecuted_outcomes_inferred=False,native_execution=False,navigation_verified=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight-only',action='store_true'); args=parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive direct-flow prefix required')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA}); result=read_json(INPUT,'result.json')
    ids={'result.json':INPUT_SHA,**result['artifact_sha256']}; verify_artifacts(INPUT,ids)
    old=read_json(INPUT,'launch.json'); admit(result,read_json(INPUT,CASE[0]+'_audit.json'),old)
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_direct_flow_maze03_prefix_v1.py',
        'lewm/tests/test_direct_flow_maze03_prefix_development.py',
        'scripts/replay_go2_direct_flow_maze01_prefix_v1.py',
        'docs/go2_direct_flow_maze01_prefix_v1_2026-09-09.md',
        'lewm/tests/test_direct_flow_live_replay_validation_development.py',
        'docs/go2_direct_flow_maze01_prefix_v1_failure_2026-09-09.md',
        'lewm/tests/test_direct_flow_prefix_development.py','lewm/tests/test_direct_flow_dual_camera_pose_development.py',
        'lewm/tests/test_direct_corner_flow_association_development.py',
        'docs/go2_direct_corner_flow_failed_pairs_result_2026-09-09.md'),old['source_sha256'])
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA})
    for name,sha in read_json(PRIOR,'result.json')['source_sha256'].items():
        if name in sources and sources[name]!=sha: raise ValueError('unchanged completed maze1 implementation required: '+name)
        sources[name]=sha
    keys=('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
          'opencv_binary_sha256','opencv_version','rules')
    launch={k:old[k] for k in keys}
    launch.update(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),replay_input_bindings=ids,
        predecessor_failure_bindings=FAILED_BINDINGS,completed_maze1_prefix_result_sha256=PRIOR_SHA,
        same_tracking_implementation_as_completed_maze1=True,live_sensor_contracts_use_unserialized_decision=True,
        correction_admission=old['correction_admission'],model_state_sha256=MODEL_STATE,planned_case=list(CASE),
        implementation_class='DirectFlowFloorTransportController',native_execution=False,model_loaded=True,
        model_training=False,shadow_replay_only=True,replay_workers=1,maximum_frames=MAX_FRAMES,
        opencv_threads=1,blas_threads=1,output_allowance_bytes=MAX_OUTPUT_BYTES,memory_admission_bytes=8*1024**3,
        minimum_free_bytes=RESERVE_BYTES,concurrency_reason='one CPU replay alongside the existing sole native scene and paired CPU benchmark',
        input_scope='completed development maze3 through its original tracking failure; no following observation')
    verify_inputs(launch); resources=hardware(); launch['hardware']=resources
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('bounded direct-flow replay resources unavailable')
    if args.preflight_only:
        print('DIRECT_FLOW_MAZE03_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            input_and_source_bindings_verified=True,output_created=False,native_execution=False)),flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch); started=time.perf_counter()
    print('DIRECT_FLOW_MAZE03_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        report=replay(launch); verify_inputs(launch)
        bindings={n:digest(OUTPUT/n) for n in ('launch.json',DECISIONS)}; verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='DIRECT_FLOW_MAZE03_PREFIX_V1_COMPLETE',source_sha256=sources,
            artifact_sha256=bindings,report=report,hardware_after=hardware(),wall_s=time.perf_counter()-started,
            model_loaded=True,model_training=False,native_execution=False,shadow_replay_only=True,
            navigation_qualified=False,goal_achieved=False))
        print('DIRECT_FLOW_MAZE03_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),
            {k:v for k,v in report.items() if k != 'fallback_receipt'},flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DIRECT_FLOW_MAZE03_PREFIX_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__': main()

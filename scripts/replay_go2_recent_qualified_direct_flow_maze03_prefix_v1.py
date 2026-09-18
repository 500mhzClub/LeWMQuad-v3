"""Isolated reference-retention replay on the completed direct-flow maze3 run."""
import argparse
from itertools import islice
import json
import re
import shutil
import time
import cv2
import torch
from lewm.recent_qualified_direct_flow_controller_development import RecentQualifiedDirectFlowController
from lewm.recent_qualified_direct_flow_prefix_development import PrefixComparison,MAX_FRAMES
from lewm.measured_floor_transport_development import current_measured_floor_pose
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit
from lewm.novel_maze_round_trip_scene_development import public_mission,specification
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS,RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.run_go2_direct_flow_maze03_pilot_v1 import OUTPUT as INPUT,CASE,verify_inputs as verify_native_context
from scripts.partial_floor_height_scoped_verification_admission_development import admit_benchmark,BENCHMARK_SHA
from scripts.scoped_verification_digest_development import verify_with_scoped_digests
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet,public_acquisition
from scripts.maze_decision_stream_development import read_rows,writer,NAME as DECISIONS
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_recent_qualified_direct_flow_maze03_prefix_v1_attempt_001'
PROTOCOL='docs/go2_recent_qualified_direct_flow_maze03_prefix_v1_2026-09-10.md'
INPUT_LAUNCH_SHA='48493379a21f2691a82195873b62ed101e927d3667d36493e4e6aa90d5aa3f28'
INPUT_BINDINGS={
    'launch.json':INPUT_LAUNCH_SHA,
    CASE[0]+'/result.json':'659f3aa6a75cb29ce72847308e89abf691e703e81739ce07fd6448d9b2e4cd23',
    CASE[0]+'/context_decisions.jsonl.gz':'fd55017ee0e2463aa4d443508047c586e27d253c9561a8d5811b9bab1042ea36',
    CASE[0]+'/command_tape.json':'23f07bd793d166e1ff8e6ad1863b3c7cfe9939196865b1466d9aa364ce2f4afd',
}
MAX_OUTPUT_BYTES=2*1024**3
MEMORY_BYTES=8*1024**3


def admit_native(result,launch,audit):
    if (result['status']!='DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE'
            or len(result['conditions'])!=1 or launch['planned_case']!=list(CASE)
            or launch['implementation_class']!='DirectFlowFloorTransportController'
            or launch['scene_specification']!=specification(3) or launch['public_mission']!=public_mission(3)
            or launch['prefix_report']['model_state_sha256']!=MODEL_STATE
            or result['source_sha256']!=launch['source_sha256']
            or result['model_training'] is not False
            or result['learned_cohort_result_sha256']!='a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720'):
        raise ValueError('completed exact direct-flow maze3 input required')
    for name,sha in INPUT_BINDINGS.items():
        if result['artifact_sha256'].get(name)!=sha:
            raise ValueError('fixed launch and closed collection identities required: '+name)
    record=result['conditions'][0]
    if (record['case']!=CASE[0] or record['layout_index']!=3 or audit['layout_index']!=3
            or record['status']!='DIRECT_FLOW_MAZE03_COLLECTED_AND_RAW_AUDITED'
            or record['model_state_unchanged'] is not True
            or record['collection']['decisions']!=1217
            or record['collection']['completed_ticks']!=1216
            or record['collection']['physics_samples']!=61550):
        raise ValueError('complete original maze3 population and raw audit required')
    require_raw_audit(record,audit,learned=True)
    expected=dict(common_prefix_frames=265,first_intervention_frame=264,physical_prefix_samples=13950,
        physical_and_public_prefix_exact=True,all_preintervention_observed_state_exact=True,
        all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,raw_model_forecast_comparisons=261,
        all_compared_raw_model_forecasts_exact=True,candidate_intervention_command_completed=True,
        original_intervention_command=[0.,0.,0.],candidate_intervention_command=[0.,0.,.45],
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
    for key,value in expected.items():
        if record['prefix_comparison'][key]!=value or type(record['prefix_comparison'][key]) is not type(value):
            raise ValueError('complete original physical intervention required: '+key)


def verify_input_context(launch):
    verify(launch);verify_artifacts(INPUT,launch['replay_input_bindings'])
    old=read_json(INPUT,'launch.json')
    admit_native(read_json(INPUT,'result.json'),old,read_json(INPUT,CASE[0]+'_audit.json'))
    verify_native_context(old)
    if launch['correction_admission']!=old['correction_admission']:
        raise ValueError('unchanged assigned learned model admission required')


def verify_inputs(launch):
    sha=launch.get('native_result_sha256')
    if (type(sha) is not str or re.fullmatch('[0-9a-f]{64}',sha) is None
            or launch['replay_input_bindings'].get('result.json')!=sha
            or any(launch['replay_input_bindings'].get(n)!=h for n,h in INPUT_BINDINGS.items())
            or launch['verification_benchmark_result_sha256']!=BENCHMARK_SHA):
        raise ValueError('bound completed native, fixed collection and scoped benchmark identities required')
    admit_benchmark();before=fingerprint(launch)
    result,counters=verify_with_scoped_digests(verify_input_context,digest,launch)
    if result is not None or fingerprint(launch)!=before:raise ValueError('unchanged original verifier context required')
    print('RECENT_QUALIFIED_DIRECT_FLOW_INPUTS_VERIFIED',counters,flush=True)


def replay(launch):
    model,condition,variant=load_assigned(launch['correction_admission'],CASE[4]);before=state_digest(model.state_dict())
    if (condition,variant)!=(CASE[3],CASE[2]) or before!=MODEL_STATE:raise ValueError('same assigned learned model required')
    controller=RecentQualifiedDirectFlowController(model,ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS,public_mission=public_mission(3),condition=condition,variant=variant,persistent=True)
    directory=INPUT/CASE[0];reader=IntentReturnRGBDReplay(directory)
    acquisitions=read_json(directory,'auxiliary_camera_audit.json');tape=read_json(directory,'command_tape.json')
    if len(reader.frames)!=1217 or len(acquisitions)!=1217 or len(tape)!=1216:
        raise ValueError('complete original direct-flow maze3 population required')
    comparator=PrefixComparison();frames=forecasts=exact=attempts=qualified=0;last=check=None
    with writer(OUTPUT) as append:
        for i,original in enumerate(islice(read_rows(directory),MAX_FRAMES)):
            if original['tick']!=i or original['observation_index']!=i or original['pre_sample_index']!=749+50*i:
                raise ValueError('ordered actual observation endpoints required')
            if shutil.disk_usage(BASE).free<RESERVE_BYTES+MAX_OUTPUT_BYTES:raise ValueError('replay storage reserve unavailable')
            p,d,f,now=reader.packet(i);image,aux=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
            public=fingerprint((p,d,f,image,aux,now))
            live=controller.observe(p,d,f,now_ns=now,auxiliary_depth=aux,auxiliary_rgb=image)
            last=json.loads(json.dumps(live,allow_nan=False))
            try:
                if fingerprint((p,d,f,image,aux,now))!=public:raise ValueError('controller mutated public arrays')
                if tape[i]['completed'] is not True:raise ValueError('completed original requested command required')
                check=comparator.compare(original['decision'],last,tape[i]['requested_command'],frame=i)
                raw=live['original_visual_evidence']
                if raw['status']=='CURRENT_VISUAL_POSE':
                    current_dual_camera_pose(raw,p,image,aux,identity=(0,0,0),now_ns=now)
                if live['terminal'] is None:current_measured_floor_pose(live['evidence'],identity=(0,0,0),now_ns=now)
            except Exception as error:
                append(dict(tick=i,decision=last,comparison_failure=repr(error),original_requested_command=tape[i]['requested_command']))
                raise
            append(dict(tick=i,decision=last,comparison=check,original_requested_command=tape[i]['requested_command'],
                public_input_arrays_unchanged=True))
            frames+=1;forecasts+=int(check['raw_model_forecasts_compared']);exact+=int(check['complete_original_decision_exact'])
            attempts+=check['extra_reference_attempts'];qualified+=check['extra_qualified_references']
            if (OUTPUT/DECISIONS).stat().st_size>MAX_OUTPUT_BYTES//2:raise ValueError('compressed output headroom exceeded')
            if i%32==0:print('RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PREFIX_FRAME',i,flush=True)
            if check['stop']:break
    if not frames or not check['stop']:raise ValueError('complete prefix until intervention or terminal required')
    if state_digest(model.state_dict())!=before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged weights and absent gradients required')
    return dict(case=CASE[0],layout_index=3,frames=frames,maximum_frames=MAX_FRAMES,exact_original_decisions=exact,
        raw_model_forecast_comparisons=forecasts,extra_reference_attempts=attempts,extra_qualified_references=qualified,
        first_reference_attempt=comparator.first_reference_attempt,first_qualified_reference=comparator.first_qualified_reference,
        first_decision_difference=comparator.first_decision_difference,first_requested_command_difference=comparator.first_command_difference,
        final_requested_command=last['requested_command'],prior_requested_command=original['decision']['requested_command'],
        final_terminal=last['terminal'],prior_terminal=original['decision']['terminal'],final_failure=last['failure'],
        boundary_comparison=check,original_actual_commands_before_intervention_exact=True,
        stopped_at_first_changed_command_or_either_terminal=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,model_state_sha256=before,model_state_unchanged=True,
        unexecuted_outcomes_inferred=False,native_execution=False,navigation_verified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-result-sha256',required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive recent-qualified-anchor prefix required')
    if re.fullmatch('[0-9a-f]{64}',args.native_result_sha256) is None:raise ValueError('explicit completed native result SHA256 required')
    input_sha=args.native_result_sha256
    verify_artifacts(INPUT,INPUT_BINDINGS)
    verify_artifacts(INPUT,{'result.json':input_sha});result=read_json(INPUT,'result.json')
    ids=dict(result['artifact_sha256']);ids['result.json']=input_sha;verify_artifacts(INPUT,ids)
    old=read_json(INPUT,'launch.json');admit_native(result,old,read_json(INPUT,CASE[0]+'_audit.json'))
    benchmark=admit_benchmark();inherited=dict(result['source_sha256'])
    for name,sha in benchmark['source_sha256'].items():
        if name in inherited and inherited[name]!=sha:raise ValueError('benchmark source conflict: '+name)
        inherited[name]=sha
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_recent_qualified_direct_flow_maze03_prefix_v1.py',
        'lewm/tests/test_recent_qualified_anchor_development.py','lewm/tests/test_recent_qualified_direct_flow_prefix_development.py',
        'lewm/tests/test_recent_qualified_direct_flow_replay_development.py',
        'lewm/tests/test_recent_qualified_direct_flow_isolation_development.py'),inherited)
    keys=('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')
    launch={k:old[k] for k in keys}
    launch.update(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),replay_input_bindings=ids,
        native_result_sha256=input_sha,verification_benchmark_result_sha256=BENCHMARK_SHA,
        correction_admission=old['correction_admission'],model_state_sha256=MODEL_STATE,planned_case=list(CASE),
        implementation_class='RecentQualifiedDirectFlowController',native_execution=False,model_loaded=True,model_training=False,
        shadow_replay_only=True,replay_workers=1,native_scene_workers=0,maximum_frames=MAX_FRAMES,
        opencv_threads=1,blas_threads=1,output_allowance_bytes=MAX_OUTPUT_BYTES,memory_admission_bytes=MEMORY_BYTES,
        concurrent_native_allowance_bytes=32*1024**3,minimum_free_bytes=RESERVE_BYTES,
        input_scope='completed direct-flow maze3; stop before any changed-command future or either terminal',
        native_input_launch_sha256=INPUT_LAUNCH_SHA,partial_floor_height_change_included=False,
        recent_qualified_anchor_enabled=True,bridge_limit_unchanged=True,raw_registration_thresholds_unchanged=True,
        native_pose_used=False,pose_uncertainty_calibrated=False)
    verify_inputs(launch);resources=hardware();launch['hardware']=resources
    if resources['memory_available_bytes']<MEMORY_BYTES+32*1024**3 or resources['artifact_free_bytes']<RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('replay and concurrent native resource allowances unavailable')
    if args.preflight_only:
        print('RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            input_and_source_bindings_verified=True,output_created=False,native_execution=False)),flush=True);return
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    print('RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        report=replay(launch);verify_inputs(launch)
        bindings={n:digest(OUTPUT/n) for n in ('launch.json',DECISIONS)};verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PREFIX_V1_COMPLETE',source_sha256=sources,
            artifact_sha256=bindings,report=report,native_result_sha256=input_sha,hardware_after=hardware(),
            wall_s=time.perf_counter()-started,model_loaded=True,model_training=False,native_execution=False,
            shadow_replay_only=True,navigation_qualified=False,goal_achieved=False))
        print('RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),report,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PREFIX_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()

"""Compare completed maze2 executions and score actual residual-fallback motion."""
import argparse
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.independent_reactive_floor_transport_study_development import MATCHED_KEYS, require_raw_audit
from lewm.residual_first_interval_execution_readout_development import summarize_execution
from scripts.run_go2_residual_first_interval_maze_pilot_v1 import (
    OUTPUT as INPUT, CASE, LEARNED, LEARNED_CASE, verify_inputs as verify_native, admit_cohort)
from scripts.maze_decision_stream_development import read_rows
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_residual_first_interval_maze_readout_v1_attempt_001'
PROTOCOL='docs/go2_residual_first_interval_maze_readout_v1_2026-09-09.md'
LEARNED_SHA='a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720'


def admit_results(current,launch,learned,old,current_audit,old_audit):
    admit_cohort(learned,old_audit,old)
    if (current['status']!='RESIDUAL_FIRST_INTERVAL_MAZE_PILOT_V1_COMPLETE'
            or len(current['conditions'])!=1 or current['learned_cohort_result_sha256']!=LEARNED_SHA
            or launch['planned_case']!=list(CASE)
            or launch['implementation_class']!='ResidualFirstIntervalController'
            or launch['prefix_report']['model_state_sha256']!=MODEL_STATE
            or launch['correction_admission']!=old['correction_admission']
            or launch['scene_specification']!=old['scene_specifications'][1]
            or launch['public_mission']!=old['public_missions'][1]):
        raise ValueError('completed same-model same-maze residual comparison required')
    new=current['conditions'][0]; prior=learned['conditions'][1]
    if (new['case']!=CASE[0] or new['layout_index']!=2
            or new['status']!='RESIDUAL_FIRST_INTERVAL_COLLECTED_AND_RAW_AUDITED'
            or new['model_state_unchanged'] is not True or current_audit['layout_index']!=2):
        raise ValueError('complete exact maze2 audit required')
    require_raw_audit(new,current_audit,learned=True); require_raw_audit(prior,old_audit,learned=True)
    expected=dict(common_prefix_frames=464,first_intervention_frame=463,physical_prefix_samples=23900,
        physical_and_public_prefix_exact=True,shared_observed_mission_and_residual_state_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=461,all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=[0.,0.,0.],candidate_intervention_command=[.16,0.,-.45],
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
    for key,value in expected.items():
        if new['prefix_comparison'][key]!=value: raise ValueError('complete actual intervention required: '+key)
    for key in MATCHED_KEYS:
        if launch[key]!=old[key]: raise ValueError('matched execution setting differs: '+key)
    return dict(layout_index=2,model_state_sha256=MODEL_STATE,matched_launch_fields=list(MATCHED_KEYS),
        physical_prefix_samples=23900,first_intervention_frame=463,
        original_outcomes_unchanged=True,reused_development_layout=True,paired_failure_retained=True)


def summarize(root,record,audit):
    directory=root/record['case']
    with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
        poses=archive['base_pose_world']; stamps=archive['timestamp_s']
    tape=read_json(directory,'command_tape.json')
    execution=summarize_execution(poses,tape,read_rows(directory))
    local=(poses[749:,:3]-poses[749,:3])@rotation_xyzw(poses[749,3:])
    goal=public_mission(2)['goal_initial_body_xy_m']; distances=np.linalg.norm(local[:,:2]-goal,axis=1)
    return dict(case=record['case'],layout_index=2,collection=record['collection'],
        verified_round_trip=audit['verified_round_trip'],native_evaluation=audit['native_evaluation'],
        strict_physical_visibility_pass=audit['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=audit['hard_measurement_failed_frames'],
        renderer_capture_audit=audit['renderer_capture_audit'],
        raw_sensor_reconstruction_pass=audit['raw_sensor_reconstruction_pass'],
        raw_model_command_replay_pass=audit['raw_model_command_replay_pass'],
        raw_command_audit_pass=audit['raw_command_audit_pass'],model_state_unchanged=audit['model_state_unchanged'],
        minimum_native_outbound_goal_distance_m=float(distances.min()),
        terminal_native_outbound_goal_distance_m=float(distances[-1]),
        terminal_native_initial_xy_m=local[-1,:2].tolist(),
        native_xy_path_length_m=float(np.linalg.norm(np.diff(local[:,:2],axis=0),axis=1).sum()),
        simulated_duration_after_initial_observation_s=float(stamps[-1]-stamps[749]),
        maximum_observed_pose_xy_error_m=max(audit['observed_pose_xy_errors_m'],default=None),
        observed_arrival_transitions=audit['observed_arrival_transitions'],execution=execution,
        timing={k:timing(audit[k]) for k in ('observation_and_control_wall_ms','iteration_with_command_wall_ms',
            'iteration_with_receipt_wall_ms','decision_receipt_write_wall_ms')},
        native_pose_is_evaluator_only=True,physical_clearance_certified=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--native-result-sha256',required=True); args=parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive residual outcome readout required')
    verify_artifacts(INPUT,{'result.json':args.native_result_sha256}); current=read_json(INPUT,'result.json')
    current_ids={'result.json':args.native_result_sha256,**current['artifact_sha256']}
    verify_artifacts(INPUT,current_ids); native_launch=read_json(INPUT,'launch.json'); verify_native(native_launch)
    verify_artifacts(LEARNED,{'result.json':LEARNED_SHA}); learned=read_json(LEARNED,'result.json')
    old=read_json(LEARNED,'launch.json')
    audits=[read_json(root,name+'_audit.json') for root,name in ((LEARNED,LEARNED_CASE[0]),(INPUT,CASE[0]))]
    admission=admit_results(current,native_launch,learned,old,audits[1],audits[0])
    sources=discover_sources((PROTOCOL,'scripts/read_go2_residual_first_interval_maze_pilot_v1.py',
        'lewm/tests/test_residual_first_interval_execution_readout_development.py',
        'lewm/tests/test_residual_first_interval_readout_admission_development.py'),current['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<RESERVE_BYTES+128*1024**2:
        raise ValueError('bounded residual outcome readout resources unavailable')
    launch=native_launch|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),native_artifact_sha256=current_ids,
        learned_artifact_sha256=native_launch['learned_artifact_sha256'],paired_comparison_admission=admission,
        hardware=resources,native_execution=False,model_loaded=False,model_training=False,
        native_scene_workers=0,memory_admission_bytes=8*1024**3,maximum_readout_bytes=128*1024**2)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch)
    try:
        summaries=[summarize(root,record,audit) for root,record,audit in zip((LEARNED,INPUT),
            (learned['conditions'][1],current['conditions'][0]),audits,strict=True)]
        if summaries[0]['execution']['fallback_attempts']: raise ValueError('original policy cannot contain successor fallback')
        if not summaries[1]['execution']['fallback_attempts'] or summaries[1]['execution']['fallback_attempts'][0]['tick']!=463:
            raise ValueError('actual first fallback must match the admitted intervention')
        verify(launch); verify_native(native_launch); verify_artifacts(INPUT,current_ids)
        verify_artifacts(LEARNED,{'result.json':LEARNED_SHA})
        write_json(OUTPUT/'result.json',dict(status='RESIDUAL_FIRST_INTERVAL_MAZE_READOUT_V1_COMPLETE',
            native_result_sha256=args.native_result_sha256,learned_cohort_result_sha256=LEARNED_SHA,
            launch_sha256=digest(OUTPUT/'launch.json'),source_sha256=sources,conditions=summaries,
            paired_comparison_admission=admission,original_outcomes_unchanged=True,new_independent_layout_executions=0,
            native_execution=False,model_loaded=False,model_training=False,physical_clearance_certified=False,
            navigation_qualified=False,hardware_qualified=False,real_time_qualified=False,goal_achieved=False))
        print('RESIDUAL_FIRST_INTERVAL_MAZE_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RESIDUAL_FIRST_INTERVAL_READOUT_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()

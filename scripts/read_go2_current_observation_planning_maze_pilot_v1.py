"""Read actual dual-camera navigation with unchanged physical outcome calculations."""
import argparse
from collections import Counter
from lewm.nominal_reentry_execution_readout_development import collect_entries, executed_motion
from lewm.executed_waypoint_readout_development import waypoint_execution
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.run_go2_current_observation_planning_maze_pilot_v1 import OUTPUT as INPUT, CASE, LEARNED, verify_inputs as verify_native
from scripts.read_go2_measured_floor_transport_maze_pilot_v1 import OUTPUT as LEARNED_READOUT
from lewm.planning_memory_method_comparison_development import admit_comparison, compare_methods
from scripts.maze_decision_stream_development import read_rows
from scripts.read_go2_causal_residual_final_goal_probe_v1 import constraint_summary
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from lewm.later_floor_resolution_execution_readout_development import later_floor_resolution_execution
from lewm.measured_floor_transport_readout_development import registered_pose_accuracy

from scripts.dual_camera_intervention_witness_development import camera_execution

OUTPUT = BASE/'go2_current_observation_planning_maze_readout_v1_attempt_001'
PROTOCOL = 'docs/go2_current_observation_planning_maze_readout_v1_2026-09-09.md'


def view_translation_execution(records):
    selected = []
    for r in records:
        c = r['recovery_candidate']
        if not c.get('requires_view_phase_exception', False): continue
        if (r['action'] not in ('forward', 'left_arc', 'right_arc')
                or c['phase_allowed'] or not c['reentry_phase_allowed']):
            raise ValueError('explicit translating view-recovery execution required')
        selected.append(r)
    return selected


def summarize(record):
    name = record['case']; directory = INPUT/name
    report = read_json(INPUT, name+'_audit.json'); collection = record['collection']
    traces = []; first_failure = first_infeasible = last_infeasible = None; modes = Counter(); targets = Counter()
    for row in read_rows(directory):
        d = row['decision']; s = d['new_selection'] or {}; m = d['mission_receipt'] or {}
        memory = d['memory_receipt'] or {}
        if d['failure'] and first_failure is None:
            first_failure = dict(tick=row['tick'], failure=d['failure'], evidence=d['evidence'])
        if 'prediction' in s and s['action'] is None:
            last_infeasible = dict(tick=row['tick'], candidates=constraint_summary(s), proposal=s['proposal'])
            if first_infeasible is None: first_infeasible = last_infeasible
        if s: modes[s['mode']] += 1; targets[s['proposal']['status']] += 1
        traces.append(dict(tick=row['tick'], terminal=d['terminal'], requested_command=d['requested_command'],
            mission_phase=m.get('phase'), active_goal_initial_body_xy_m=m.get('active_goal_initial_body_xy_m'),
            observed_goal_distance_m=d['observed_goal_distance_m'], quiet_intervals=d['quiet_intervals'],
            selected_action=d['selected_action'], mode=s.get('mode'), proposal_status=s.get('proposal', {}).get('status'),
            waypoint_map_xy_m=s.get('waypoint_map_xy_m'), active_infeasible_wait=d['infeasible_wait_active'],
            retained_floor_cells=memory.get('retained_observed_floor_cells'),
            retained_occupied_cells=memory.get('retained_occupied_cells'),
            current_planning_floor_cells=s.get('planning_map_receipt', {}).get('current_floor_cells'),
            current_planning_occupied_cells=s.get('planning_map_receipt', {}).get('current_occupied_cells'),
            causal_final_goal_scoring=s.get('causal_final_goal_scoring', False)))
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
        poses = archive['base_pose_world']; stamps = archive['timestamp_s']
    recovery = collect_entries(read_rows(directory))
    motion = executed_motion(poses, read_json(directory, 'command_tape.json'), recovery['entries'])
    waypoint = waypoint_execution(poses, read_json(directory, 'command_tape.json'), read_rows(directory))
    local = (poses[749:, :3]-poses[749, :3])@rotation_xyzw(poses[749, 3:])
    mission = public_mission(record['layout_index'])
    distances = np.linalg.norm(local[:, :2]-mission['goal_initial_body_xy_m'], axis=1)
    native_trace = [dict(frame=i, sample_index=749+50*i, native_initial_xy_m=local[50*i, :2].tolist(),
        native_outbound_goal_distance_m=float(distances[50*i])) for i in range(len(traces))]
    return dict(case=name, layout_index=record['layout_index'], collection=collection,
        verified_round_trip=report['verified_round_trip'], reused_development_layout=True,
        renderer_capture_audit=report['renderer_capture_audit'], measured_floor_transport_enabled=True,
        planning_map_variant='current_paired_observation', persistent_contact_history_retained=True, selector_scan_state_retained=True,
        prefix_comparison=record.get('prefix_comparison'), reentry_execution=motion, waypoint_execution=waypoint,
        view_translation_execution=view_translation_execution(motion),
        dual_camera_execution=camera_execution(read_rows(directory)),
        registered_pose_accuracy=registered_pose_accuracy(poses, read_rows(directory)),
        later_floor_resolution_execution=later_floor_resolution_execution(poses, read_json(directory, 'command_tape.json'), read_rows(directory)),
        observed_nominal_reentry_transitions=recovery['observed_nominal_reentry_transitions'],
        unresolved_reentry_start_tick=recovery['unresolved_reentry_start_tick'], native_evaluation=report['native_evaluation'],
        raw_sensor_reconstruction_pass=report['raw_sensor_reconstruction_pass'],
        raw_model_command_replay_pass=report['raw_model_command_replay_pass'],
        raw_command_audit_pass=report['raw_command_audit_pass'], model_state_unchanged=report['model_state_unchanged'],
        strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
        minimum_native_outbound_goal_distance_m=float(distances.min()),
        terminal_native_outbound_goal_distance_m=float(distances[-1]),
        terminal_native_initial_xy_m=local[-1, :2].tolist(),
        native_xy_path_length_m=float(np.linalg.norm(np.diff(local[:, :2], axis=0), axis=1).sum()),
        simulated_duration_after_initial_observation_s=float(stamps[-1]-stamps[749]),
        maximum_observed_pose_xy_error_m=max(report['observed_pose_xy_errors_m'], default=None),
        selected_actions=report['selected_actions'], selection_modes=dict(modes), proposal_statuses=dict(targets),
        first_failure=first_failure, first_infeasible=first_infeasible, last_infeasible=last_infeasible, decision_trace=traces,
        native_trace_evaluator_only=native_trace, observed_arrival_transitions=report['observed_arrival_transitions'],
        timing={k: timing(report[k]) for k in ('observation_and_control_wall_ms', 'iteration_with_command_wall_ms',
            'iteration_with_receipt_wall_ms', 'decision_receipt_write_wall_ms')},
        auxiliary_frames=len(report['auxiliary_sensor_audit']),
        auxiliary_robot_occluded_frames=[r['frame'] for r in report['auxiliary_sensor_audit'] if r['robot_pixels']],
        matched_baseline_available=True, model_or_controller_selection_performed=False,
        physical_return_proves_memory_advantage=False, navigation_qualified=False, hardware_qualified=False,
        real_time_qualified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native-result-sha256',required=True)
    parser.add_argument('--learned-readout-sha256',required=True); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive paired planning-memory readout required')
    verify_artifacts(INPUT,{'result.json':args.native_result_sha256}); current = read_json(INPUT,'result.json')
    current_ids = dict(current['artifact_sha256']); current_ids['result.json'] = args.native_result_sha256
    verify_artifacts(INPUT,current_ids); launch = read_json(INPUT,'launch.json'); verify_native(launch)
    verify_artifacts(LEARNED_READOUT,{'result.json':args.learned_readout_sha256})
    baseline = read_json(LEARNED_READOUT,'result.json')
    baseline_ids = {'result.json':args.learned_readout_sha256,'launch.json':baseline['launch_sha256']}
    verify_artifacts(LEARNED_READOUT,baseline_ids); verify(read_json(LEARNED_READOUT,'launch.json'))
    verify_artifacts(LEARNED,launch['learned_artifact_sha256']); baseline_launch = read_json(LEARNED,'launch.json')
    admission = admit_comparison(current,launch,baseline,baseline_launch)
    inherited = dict(current['source_sha256'])
    for name,sha in baseline['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('conflicting frozen paired source: '+name)
        inherited[name] = sha
    sources = discover_sources((PROTOCOL,'scripts/read_go2_current_observation_planning_maze_pilot_v1.py',
        'lewm/tests/test_planning_memory_method_comparison_development.py',
        'lewm/tests/test_measured_floor_transport_readout_development.py'),inherited)
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+128*1024**2:
        raise ValueError('bounded paired planning-memory readout resources unavailable')
    readout_launch = launch | dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),
        planning_native_artifact_sha256=current_ids,learned_readout_artifact_sha256=baseline_ids,
        paired_comparison_admission=admission,hardware=resources,native_execution=False,
        high_level_world_model_loaded=False,model_training=False)
    verify(readout_launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json',readout_launch)
    try:
        report = summarize(current['conditions'][0]); comparison = compare_methods(report,baseline['conditions'][0],admission)
        verify(readout_launch); verify_native(launch); verify_artifacts(INPUT,current_ids)
        verify_artifacts(LEARNED_READOUT,baseline_ids); verify_artifacts(LEARNED,launch['learned_artifact_sha256'])
        write_json(OUTPUT/'result.json',dict(status='CURRENT_OBSERVATION_PLANNING_MAZE_READOUT_V1_COMPLETE',
            native_result_sha256=args.native_result_sha256,learned_readout_result_sha256=args.learned_readout_sha256,
            learned_native_result_sha256=baseline['native_result_sha256'],launch_sha256=digest(OUTPUT/'launch.json'),
            source_sha256=sources,conditions=[report],planning_map_comparison=comparison,
            native_execution=False,original_outcome_unchanged=True,new_independent_layout_executions=0,
            reused_development_layout=True,measured_round_trip_successes=int(report['verified_round_trip']),
            memory_advantage_established=False,memoryless_comparator=False,
            navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('CURRENT_OBSERVATION_PLANNING_MAZE_READOUT_V1_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_CURRENT_OBSERVATION_PLANNING_READOUT_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()


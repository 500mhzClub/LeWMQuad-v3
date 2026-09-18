"""Read completed reactive navigation and describe the matched-scene policy comparison."""
import argparse
from collections import Counter
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.run_go2_reactive_nominal_maze_pilot_v1 import OUTPUT as INPUT, CASE, WAYPOINT_READOUT, WAYPOINT_READOUT_SHA
from scripts.maze_decision_stream_development import read_rows
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_reactive_nominal_maze_readout_v1_attempt_001'
PROTOCOL = 'docs/go2_reactive_nominal_maze_readout_v1_2026-09-08.md'



def constraint_summary(selection):
    return dict(current_nominal_clearance=selection.get('current_nominal_clearance'),
        current_surface_intersection=selection.get('current_surface_check', {}).get('possible_intersection'),
        measured_waypoint_connector=selection.get('measured_waypoint_connector'),
        unknown_waypoint_connector_cells=selection.get('unknown_waypoint_connector_cells'),
        heading_error_rad=selection.get('heading_error_rad'),
        view_budget_exhausted=selection.get('view_budget_exhausted'),
        candidate_future_outcomes_evaluated=False)


def compare(reactive, predictive):
    if reactive['layout_index'] != 0 or predictive['layout_index'] != 0:
        raise ValueError('the same bound development maze 0 is required')
    def metrics(record):
        evaluation = record['native_evaluation']
        path = evaluation['outbound_traversal']
        return dict(case=record['case'], verified_round_trip=record['verified_round_trip'],
            observed_arrivals=len(record['observed_arrival_transitions']),
            native_arrival_windows=len(evaluation['arrival_windows']),
            outbound_edge_crossings=len(path['crossings']),
            outbound_loop_erased_cells=path['loop_erased_cells'],
            native_xy_path_length_m=record['native_xy_path_length_m'],
            minimum_native_outbound_goal_distance_m=record['minimum_native_outbound_goal_distance_m'],
            terminal_native_outbound_goal_distance_m=record['terminal_native_outbound_goal_distance_m'],
            schedule_terminal=record['collection']['schedule_terminal'],
            physical_stop=record['collection']['physical_stop'], acquisition_stop=record['collection']['acquisition_stop'],
            selected_actions=record['selected_actions'], timing=record['timing'])
    return dict(reactive=metrics(reactive), predictive=metrics(predictive),
        layout_index=0, reused_development_layout=True,
        same_scene_sensor_actuator_mission_budget=True, predictive_geometry_gates_matched=False,
        persistent_observed_memory_in_both=True, ranking_only_ablation=False,
        single_layout_controller_comparison=True, learned_planning_advantage_established=False,
        memory_advantage_established=False, independent_layout_generalization_established=False)


def summarize(record):
    name = record['case']; directory = INPUT/name
    report = read_json(INPUT, name+'_audit.json'); collection = record['collection']
    traces = []; first_failure = first_infeasible = last_infeasible = None; modes = Counter(); targets = Counter()
    for row in read_rows(directory):
        d = row['decision']; s = d['new_selection'] or {}; m = d['mission_receipt'] or {}
        memory = d['memory_receipt'] or {}
        if d['failure'] and first_failure is None:
            first_failure = dict(tick=row['tick'], failure=d['failure'], evidence=d['evidence'])
        if s and s['action'] is None:
            last_infeasible = dict(tick=row['tick'], constraints=constraint_summary(s), proposal=s['proposal'])
            if first_infeasible is None: first_infeasible = last_infeasible
        if s: modes[s['mode']] += 1; targets[s['proposal']['status']] += 1
        traces.append(dict(tick=row['tick'], terminal=d['terminal'], requested_command=d['requested_command'],
            mission_phase=m.get('phase'), active_goal_initial_body_xy_m=m.get('active_goal_initial_body_xy_m'),
            observed_goal_distance_m=d['observed_goal_distance_m'], quiet_intervals=d['quiet_intervals'],
            selected_action=d['selected_action'], mode=s.get('mode'), proposal_status=s.get('proposal', {}).get('status'),
            waypoint_map_xy_m=s.get('waypoint_map_xy_m'), active_infeasible_wait=d['infeasible_wait_active'],
            retained_floor_cells=memory.get('retained_observed_floor_cells'),
            retained_occupied_cells=memory.get('retained_occupied_cells'),
            heading_error_rad=s.get('heading_error_rad'),
            unknown_connector_motion_permitted=s.get('unknown_connector_motion_permitted', False)))
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
        poses = archive['base_pose_world']; stamps = archive['timestamp_s']
    local = (poses[749:, :3]-poses[749, :3])@rotation_xyzw(poses[749, 3:])
    mission = public_mission(record['layout_index'])
    distances = np.linalg.norm(local[:, :2]-mission['goal_initial_body_xy_m'], axis=1)
    native_trace = [dict(frame=i, sample_index=749+50*i, native_initial_xy_m=local[50*i, :2].tolist(),
        native_outbound_goal_distance_m=float(distances[50*i])) for i in range(len(traces))]
    return dict(case=name, layout_index=record['layout_index'], collection=collection,
        verified_round_trip=report['verified_round_trip'], native_evaluation=report['native_evaluation'],
        prefix_comparisons=record.get('prefix_comparisons'),
        raw_sensor_reconstruction_pass=report['raw_sensor_reconstruction_pass'],
        raw_controller_command_replay_pass=report['raw_controller_command_replay_pass'],
        raw_command_audit_pass=report['raw_command_audit_pass'], high_level_world_model_used=False,
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
        broader_baseline_cohort_completed=False, model_or_controller_selection_performed=False,
        physical_return_proves_memory_advantage=False, navigation_qualified=False, hardware_qualified=False,
        real_time_qualified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native-result-sha256', required=True); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completed-maze readout required')
    verify_artifacts(INPUT, {'result.json': args.native_result_sha256}); result = read_json(INPUT, 'result.json')
    assert result['status'] == 'REACTIVE_NOMINAL_MAZE_PILOT_COMPLETE' and len(result['conditions']) == 1
    assert result['conditions'][0]['case'] == CASE
    assert all(r['physical_and_public_prefix_exact'] for r in result['conditions'][0]['prefix_comparisons'].values())
    verify_artifacts(WAYPOINT_READOUT, {'result.json': WAYPOINT_READOUT_SHA})
    waypoint = read_json(WAYPOINT_READOUT, 'result.json')
    assert waypoint['status'] == 'EXECUTED_WAYPOINT_MAZE_READOUT_COMPLETE'
    bindings = {'result.json': args.native_result_sha256, **result['artifact_sha256']}; verify_artifacts(INPUT, bindings)
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/read_go2_reactive_nominal_maze_pilot_v1.py',
        'lewm/tests/test_reactive_nominal_maze_readout_development.py'), old['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+128*1024**2:
        raise ValueError('bounded readout resources unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT),
        input_artifact_sha256=bindings, waypoint_readout_result_sha256=WAYPOINT_READOUT_SHA,
        hardware=resources, native_execution=False, model_training=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    try:
        report = summarize(result['conditions'][0])
        verify(launch); verify_artifacts(INPUT, bindings)
        verify_artifacts(WAYPOINT_READOUT, {'result.json': WAYPOINT_READOUT_SHA})
        write_json(OUTPUT/'result.json', dict(status='REACTIVE_NOMINAL_MAZE_READOUT_COMPLETE',
            native_result_sha256=args.native_result_sha256, launch_sha256=digest(OUTPUT/'launch.json'),
            source_sha256=sources, conditions=[report], native_execution=False,
            matched_scene_comparison=compare(report, waypoint['conditions'][0]),
            original_outcome_unchanged=True, new_independent_layout_executions=0, reused_development_layout=True, measured_round_trip_successes=int(report['verified_round_trip']),
            matched_scene_baseline_execution_completed=True, broader_baseline_cohort_completed=False, navigation_qualified=False, goal_achieved=False))
        print('REACTIVE_NOMINAL_MAZE_READOUT_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACTIVE_NOMINAL_MAZE_READOUT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()

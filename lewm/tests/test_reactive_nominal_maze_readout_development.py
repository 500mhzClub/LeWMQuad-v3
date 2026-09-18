import numpy as np
import pytest
from lewm.novel_maze_round_trip_scene_development import public_mission
from scripts import read_go2_reactive_nominal_maze_pilot_v1 as module


def test_near_native_endpoint_cannot_relabel_failed_round_trip(tmp_path, monkeypatch):
    name = 'synthetic'; directory = tmp_path/name; directory.mkdir()
    target = np.asarray(public_mission(0)['goal_initial_body_xy_m'])
    poses = np.zeros((900, 7)); poses[:, 6] = 1.
    # Deliberately impossible trajectory: readout must preserve the supplied
    # failed audit, never promote a descriptive zero-distance minimum.
    poses[750:850, :2] = target
    np.savez(directory/'physics_trace.npz', base_pose_world=poses, timestamp_s=np.arange(1, 901)*.002)
    rows = [dict(tick=i, decision=dict(new_selection=None, failure=None,
        mission_receipt=dict(phase='OUTBOUND', active_goal_initial_body_xy_m=target.tolist()),
        memory_receipt=None, terminal='MISSION_TICK_BUDGET_EXHAUSTED', requested_command=[0., 0., 0.],
        observed_goal_distance_m=None, quiet_intervals=0, selected_action=None, infeasible_wait_active=False))
        for i in range(4)]
    report = dict(verified_round_trip=False, native_evaluation={'native_round_trip_candidate_pass': False},
        raw_sensor_reconstruction_pass=True, raw_controller_command_replay_pass=True, raw_command_audit_pass=True,
        high_level_world_model_used=False, strict_physical_visibility_pass=True, hard_measurement_failed_frames=[],
        observed_pose_xy_errors_m=[], selected_actions={}, observed_arrival_transitions=[], auxiliary_sensor_audit=[])
    for key in ('observation_and_control_wall_ms', 'iteration_with_command_wall_ms',
            'iteration_with_receipt_wall_ms', 'decision_receipt_write_wall_ms'):
        report[key] = [800., 900.]
    monkeypatch.setattr(module, 'INPUT', tmp_path)
    monkeypatch.setattr(module, 'read_rows', lambda directory: iter(rows))
    monkeypatch.setattr(module, 'read_json', lambda *a: report)
    result = module.summarize(dict(case=name, layout_index=0, collection={'synthetic': True}))
    assert result['minimum_native_outbound_goal_distance_m'] == 0.
    assert result['terminal_native_outbound_goal_distance_m'] == pytest.approx(np.linalg.norm(target))
    assert not result['verified_round_trip'] and not result['native_evaluation']['native_round_trip_candidate_pass']
    assert not result['broader_baseline_cohort_completed'] and not result['real_time_qualified']
    assert not result['physical_return_proves_memory_advantage']
    assert result['native_trace_evaluator_only'][1]['native_initial_xy_m'] == target.tolist()
    # A recovered early infeasibility must not replace the terminal constraint
    # witness in a diagnosis; retain both the first and the latest event.
    monkeypatch.setattr(module, 'constraint_summary', lambda selection: selection['current_nominal_clearance'])
    for i in (1, 2):
        rows[i]['decision']['new_selection'] = dict(current_nominal_clearance=[i], action=None, mode='WAYPOINT',
            proposal={'status': 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER'})
    result = module.summarize(dict(case=name, layout_index=0, collection={'synthetic': True}))
    assert result['first_infeasible']['tick'] == 1 and result['last_infeasible']['tick'] == 2
    assert result['last_infeasible']['constraints'] == [2]


def test_paired_readout_preserves_individual_outcomes_and_does_not_infer_attribution():
    def record(name, success):
        return dict(case=name, layout_index=0, verified_round_trip=success,
            native_evaluation=dict(arrival_windows=[{}]*int(success)*2,
                outbound_traversal=dict(crossings=[{}]*3, loop_erased_cells=[[-1, 0], [0, 0], [0, 1], [1, 1]])),
            observed_arrival_transitions=[{}]*int(success)*2, native_xy_path_length_m=4.9,
            minimum_native_outbound_goal_distance_m=2.85, terminal_native_outbound_goal_distance_m=2.86,
            collection=dict(schedule_terminal='SYNTHETIC', physical_stop=None, acquisition_stop=None),
            selected_actions={'left_arc': 4}, timing={})
    a, b = record('reactive', False), record('predictive', True)
    r = module.compare(a, b)
    assert not r['reactive']['verified_round_trip'] and r['predictive']['verified_round_trip']
    assert r['reactive']['native_arrival_windows'] == 0 and r['predictive']['native_arrival_windows'] == 2
    assert r['reactive']['outbound_edge_crossings'] == 3
    for key in ('predictive_geometry_gates_matched', 'ranking_only_ablation',
            'learned_planning_advantage_established', 'memory_advantage_established',
            'independent_layout_generalization_established'):
        assert not r[key]
    with pytest.raises(ValueError): module.compare(a, b | dict(layout_index=1))

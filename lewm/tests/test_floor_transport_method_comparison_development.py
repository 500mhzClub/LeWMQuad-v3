"""Reject mismatched experiments and preserve physical versus verified outcomes."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest
from lewm.floor_transport_method_comparison_development import (
    MATCHED_LAUNCH_KEYS, admit_comparison, compare_methods)


def evidence():
    settings = {k:{'fixture':k} for k in MATCHED_LAUNCH_KEYS}
    native_prefix = dict(physical_and_public_prefix_exact=True, shared_observed_state_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        common_prefix_frames=4, physical_prefix_samples=900)
    reactive = dict(status='REACTIVE_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE', learned_result_sha256='a'*64,
        conditions=[dict(case='reactive_floor_transport_novel_maze_00',layout_index=0,
            status='REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED',prefix_comparison=native_prefix)])
    rlaunch = deepcopy(settings) | dict(learned_result_sha256='a'*64,
        implementation_class='ReactiveFloorTransportController', high_level_world_model_loaded=False,
        candidate_future_outcomes_evaluated=False, learned_residual_used=False)
    predictive = dict(status='MEASURED_FLOOR_TRANSPORT_MAZE_READOUT_V1_COMPLETE',
        native_result_sha256='a'*64, original_outcome_unchanged=True,
        conditions=[dict(case='full_jepa_novel_maze_00',layout_index=0)])
    plaunch = deepcopy(settings) | dict(implementation_class='MeasuredFloorTransportController')
    return reactive,rlaunch,predictive,plaunch


@pytest.mark.parametrize('fault', [None,'result_identity','launch_identity','incomplete','wrong_case',
    'prefix','short_physics','learned_in_reactive','scene_specification','public_mission',
    'robot_urdf_sha256','navigation_ticks','renderer_environment','native_geometry_sha256','opencv_threads'])
def test_only_exact_completed_method_pair_and_settings_admitted(fault):
    reactive,rlaunch,predictive,plaunch = evidence()
    if fault == 'result_identity': predictive['native_result_sha256'] = 'b'*64
    elif fault == 'launch_identity': rlaunch['learned_result_sha256'] = 'b'*64
    elif fault == 'incomplete': reactive['status'] = 'RUNNING'
    elif fault == 'wrong_case': predictive['conditions'][0]['case'] = 'old_learned_pilot'
    elif fault == 'prefix': reactive['conditions'][0]['prefix_comparison']['shared_observed_state_exact'] = False
    elif fault == 'short_physics': reactive['conditions'][0]['prefix_comparison']['physical_prefix_samples'] = 899
    elif fault == 'learned_in_reactive': rlaunch['high_level_world_model_loaded'] = True
    elif fault is not None: rlaunch[fault] = {'different':True}
    if fault is None:
        result = admit_comparison(reactive,rlaunch,predictive,plaunch)
        assert result['paired_learned_native_result_sha256'] == 'a'*64
        assert not result['predictive_feasibility_gates_matched']
    else:
        with pytest.raises(ValueError): admit_comparison(reactive,rlaunch,predictive,plaunch)


def record(name, *, physical_candidate=False, returning=False):
    return dict(case=name,layout_index=0,verified_round_trip=False,
        native_evaluation=dict(native_round_trip_candidate_pass=physical_candidate,
            arrival_windows=[{'phase':'OUTBOUND','native_one_second_arrival_and_quiet_pass':True}],
            outbound_traversal={'loop_erased_cells':[[0,0],[1,0]]},
            return_traversal={'loop_erased_cells':[[1,0],[0,0]]} if returning else None,
            physically_retraced_outbound_route=returning, terminal_native_quiet_pass=returning),
        observed_arrival_transitions=[{'tick':100}], strict_physical_visibility_pass=False,
        hard_measurement_failed_frames=[17],native_xy_path_length_m=2.,
        minimum_native_outbound_goal_distance_m=.03,terminal_native_outbound_goal_distance_m=.1,
        terminal_native_initial_xy_m=[.1,.2],simulated_duration_after_initial_observation_s=30.,
        collection=dict(schedule_terminal='FIXTURE_NEGATIVE',physical_stop=None,acquisition_stop=None),
        selected_actions={'forward':10},timing={'iteration_with_receipt_wall_ms':{'median':1200.}})


def test_absent_return_and_failed_visibility_remain_distinct_from_physical_success():
    reactive = record('reactive')
    learned = record('learned',physical_candidate=True,returning=True)
    original = deepcopy((reactive,learned))
    result = compare_methods(reactive,learned,{'fixture':True})
    assert result['reactive']['return_traversal'] is None
    assert result['predictive']['native_round_trip_candidate_pass'] is True
    assert result['predictive']['verified_round_trip'] is False
    assert result['predictive']['strict_physical_visibility_pass'] is False
    for key in ('reactive','predictive'):
        assert result[key]['native_arrival_windows'][0]['native_one_second_arrival_and_quiet_pass']
        assert result[key]['hard_measurement_failed_frames'] == [17]
        assert result[key]['timing']['iteration_with_receipt_wall_ms']['median'] == 1200.
    assert not result['learned_planning_advantage_established'] and not result['memory_advantage_established']
    result['reactive']['selected_actions']['forward'] = 0
    result['predictive']['outbound_traversal']['loop_erased_cells'].append([2,0])
    assert (reactive,learned) == original


def test_readout_preserves_original_reactive_actual_motion_and_terminal_calculations():
    def function(path,name):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name == name)
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,node):
            node.keywords = [k for k in node.keywords if k.arg not in
                ('renderer_capture_audit','registered_pose_accuracy','dual_camera_execution')]
            for k in node.keywords:
                if k.arg == 'prefix_comparison':
                    k.arg = 'prefix_comparisons'; k.value = ast.parse("record.get('prefix_comparisons')",mode='eval').body
            return self.generic_visit(node)
    for name in ('constraint_summary','summarize'):
        old = function('scripts/read_go2_reactive_connector_maze_pilot_v1.py',name)
        new = function('scripts/read_go2_reactive_floor_transport_maze_pilot_v1.py',name)
        assert ast.dump(old) == ast.dump(Normalize().visit(new))

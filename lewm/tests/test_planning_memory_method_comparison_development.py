"""Require matched completed experiments and retain strict versus physical results."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest
from lewm.planning_memory_method_comparison_development import (
    MATCHED_LAUNCH_KEYS,MODEL_STATE,admit_comparison,compare_methods)
from lewm.tests.test_floor_transport_method_comparison_development import record


def evidence():
    settings={k:{'fixture':k} for k in MATCHED_LAUNCH_KEYS}
    prefix=dict(physical_and_public_prefix_exact=True,shared_observed_and_executed_residual_state_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        all_compared_raw_model_forecasts_exact=True,common_prefix_frames=11,physical_prefix_samples=1250,
        raw_model_forecast_comparisons=8)
    current=dict(status='CURRENT_OBSERVATION_PLANNING_MAZE_PILOT_V1_COMPLETE',learned_result_sha256='a'*64,
        conditions=[dict(case='full_jepa_current_observation_planning_maze_00',layout_index=0,
            status='CURRENT_OBSERVATION_PLANNING_COLLECTED_AND_RAW_AUDITED',model_state_unchanged=True,prefix_comparison=prefix)])
    cl=deepcopy(settings)|dict(learned_result_sha256='a'*64,implementation_class='CurrentObservationPlanningController',
        planning_map_variant='current_paired_observation',accumulated_planning_cells_queried=False,memoryless_controller=False,
        prefix_report={'model_state_sha256':MODEL_STATE},persistent_contact_history_retained=True,
        tracking_and_floor_anchor_history_retained=True,learned_temporal_history_and_residual_retained=True,
        mission_and_settling_state_retained=True,selector_scan_state_retained=True)
    baseline=dict(status='MEASURED_FLOOR_TRANSPORT_MAZE_READOUT_V1_COMPLETE',native_result_sha256='a'*64,
        original_outcome_unchanged=True,conditions=[dict(case='full_jepa_novel_maze_00',layout_index=0,
            raw_sensor_reconstruction_pass=True,raw_model_command_replay_pass=True,raw_command_audit_pass=True,model_state_unchanged=True)])
    bl=deepcopy(settings)|dict(implementation_class='MeasuredFloorTransportController',prefix_report={'model_state_sha256':MODEL_STATE})
    return current,cl,baseline,bl


@pytest.mark.parametrize('fault',[None,'incomplete','result_identity','case','model','old_model','baseline_raw',
    'prefix','short_physics','forecast_count','contact','scan','memoryless','scene_specification','public_mission',
    'renderer_environment','navigation_ticks','opencv_threads'])
def test_only_exact_completed_pair_with_same_non_map_state_is_admitted(fault):
    current,cl,baseline,bl=evidence()
    if fault=='incomplete':current['status']='RUNNING'
    elif fault=='result_identity':baseline['native_result_sha256']='b'*64
    elif fault=='case':current['conditions'][0]['case']='different'
    elif fault=='model':cl['prefix_report']['model_state_sha256']='b'*64
    elif fault=='old_model':bl['prefix_report']['model_state_sha256']='b'*64
    elif fault=='baseline_raw':baseline['conditions'][0]['raw_model_command_replay_pass']=False
    elif fault=='prefix':current['conditions'][0]['prefix_comparison']['shared_observed_and_executed_residual_state_exact']=False
    elif fault=='short_physics':current['conditions'][0]['prefix_comparison']['physical_prefix_samples']=1249
    elif fault=='forecast_count':current['conditions'][0]['prefix_comparison']['raw_model_forecast_comparisons']=7
    elif fault=='contact':cl['persistent_contact_history_retained']=False
    elif fault=='scan':cl['selector_scan_state_retained']=False
    elif fault=='memoryless':cl['memoryless_controller']=True
    elif fault is not None:cl[fault]={'changed':True}
    if fault is None:
        result=admit_comparison(current,cl,baseline,bl)
        assert result['planning_map_persistence_is_declared_intervention']
        assert result['model_state_sha256']==MODEL_STATE and not result['fully_memoryless_comparator']
    else:
        with pytest.raises(ValueError):admit_comparison(current,cl,baseline,bl)


def test_physical_candidate_visibility_failure_and_absent_return_are_not_relabelled():
    current=record('current',physical_candidate=True,returning=True);baseline=record('baseline')
    before=deepcopy((current,baseline));r=compare_methods(current,baseline,{'fixture':True})
    assert r['current_observation_planning']['native_round_trip_candidate_pass'] is True
    assert r['current_observation_planning']['verified_round_trip'] is False
    assert r['persistent_planning']['return_traversal'] is None
    assert r['current_observation_planning']['hard_measurement_failed_frames']==[17]
    assert r['current_observation_planning']['timing']['iteration_with_receipt_wall_ms']['median']==1200.
    assert not r['memory_advantage_established'] and not r['memoryless_comparator']
    assert r['persistent_contact_history_in_both'] and r['selector_scan_state_in_both']
    r['current_observation_planning']['native_arrival_windows'].clear()
    assert (current,baseline)==before


def test_readout_preserves_baseline_motion_and_outcome_calculations():
    def function(path,name):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,n):
            n.keywords=[k for k in n.keywords if k.arg not in ('planning_map_variant','persistent_contact_history_retained',
                'selector_scan_state_retained','current_planning_floor_cells','current_planning_occupied_cells')]
            for k in n.keywords:
                if k.arg=='matched_baseline_available':k.value=ast.Constant(False)
            return self.generic_visit(n)
    for name in ('view_translation_execution','summarize'):
        old=function('scripts/read_go2_measured_floor_transport_maze_pilot_v1.py',name)
        new=function('scripts/read_go2_current_observation_planning_maze_pilot_v1.py',name)
        assert ast.dump(old)==ast.dump(Normalize().visit(new))

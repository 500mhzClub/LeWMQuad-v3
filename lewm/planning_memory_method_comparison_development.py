"""Match native planning-map experiments and preserve each physical outcome."""
from copy import deepcopy
from lewm.floor_transport_method_comparison_development import MATCHED_LAUNCH_KEYS, metrics
from lewm.independent_floor_transport_study_development import MODEL_STATE


def admit_comparison(current, current_launch, baseline_readout, baseline_launch):
    if (current['status'] != 'CURRENT_OBSERVATION_PLANNING_MAZE_PILOT_V1_COMPLETE'
            or len(current['conditions']) != 1
            or baseline_readout['status'] != 'MEASURED_FLOOR_TRANSPORT_MAZE_READOUT_V1_COMPLETE'
            or len(baseline_readout['conditions']) != 1
            or baseline_readout['original_outcome_unchanged'] is not True):
        raise ValueError('completed current-planning native and unchanged baseline readout required')
    if not current['learned_result_sha256'] == current_launch['learned_result_sha256'] == baseline_readout['native_result_sha256']:
        raise ValueError('exact paired learned native result required')
    cr = current['conditions'][0]; br = baseline_readout['conditions'][0]
    if (cr['case'] != 'full_jepa_current_observation_planning_maze_00' or cr['layout_index'] != 0
            or cr['status'] != 'CURRENT_OBSERVATION_PLANNING_COLLECTED_AND_RAW_AUDITED'
            or cr['model_state_unchanged'] is not True
            or br['case'] != 'full_jepa_novel_maze_00' or br['layout_index'] != 0):
        raise ValueError('fixed paired learned maze0 methods required')
    for key in ('raw_sensor_reconstruction_pass','raw_model_command_replay_pass','raw_command_audit_pass','model_state_unchanged'):
        if br[key] is not True: raise ValueError('completed baseline raw evidence required: '+key)
    prefix = cr['prefix_comparison']
    for key in ('physical_and_public_prefix_exact','shared_observed_and_executed_residual_state_exact',
            'all_preintervention_requested_commands_exact','complete_candidate_decisions_match_prospective_prefix',
            'all_compared_raw_model_forecasts_exact'):
        if prefix[key] is not True: raise ValueError('actual planning-memory native prefix required: '+key)
    if (prefix['common_prefix_frames'] != 11 or prefix['physical_prefix_samples'] != 1250
            or prefix['raw_model_forecast_comparisons'] != 8):
        raise ValueError('complete eleven-observation/eight-forecast physical prefix required')
    for key in MATCHED_LAUNCH_KEYS:
        if current_launch[key] != baseline_launch[key]: raise ValueError('matched setting differs: '+key)
    if (current_launch['implementation_class'] != 'CurrentObservationPlanningController'
            or baseline_launch['implementation_class'] != 'MeasuredFloorTransportController'
            or current_launch['planning_map_variant'] != 'current_paired_observation'
            or current_launch['accumulated_planning_cells_queried'] is not False
            or current_launch['memoryless_controller'] is not False
            or current_launch['prefix_report']['model_state_sha256'] != MODEL_STATE
            or baseline_launch['prefix_report']['model_state_sha256'] != MODEL_STATE):
        raise ValueError('explicit planning-map intervention with unchanged assigned weights required')
    for key in ('persistent_contact_history_retained','tracking_and_floor_anchor_history_retained',
            'learned_temporal_history_and_residual_retained','mission_and_settling_state_retained','selector_scan_state_retained'):
        if current_launch[key] is not True: raise ValueError('retained non-map temporal state required: '+key)
    return dict(paired_learned_native_result_sha256=current['learned_result_sha256'],
        matched_launch_fields=list(MATCHED_LAUNCH_KEYS),model_state_sha256=MODEL_STATE,
        physical_and_public_prefix_exact=True,same_observed_contact_and_executed_residual_state_at_intervention=True,
        raw_model_forecasts_exact_at_intervention=True,planning_map_persistence_is_declared_intervention=True,
        fully_memoryless_comparator=False)


def compare_methods(current, baseline, admission):
    if current['layout_index'] != 0 or baseline['layout_index'] != 0:
        raise ValueError('paired reused maze0 required')
    return dict(current_observation_planning=metrics(current),persistent_planning=metrics(baseline),
        admission=deepcopy(admission),layout_index=0,reused_development_layout=True,independent_layout_executions=0,
        persistent_contact_history_in_both=True,tracking_and_floor_anchor_history_in_both=True,
        learned_temporal_history_and_residual_in_both=True,mission_and_settling_state_in_both=True,
        selector_scan_state_in_both=True,memoryless_comparator=False,
        memory_advantage_established=False,jepa_training_advantage_established=False,
        statistical_reliability_established=False,independent_layout_generalization_established=False,
        real_time_qualified=False,hardware_qualified=False,original_outcomes_unchanged=True)

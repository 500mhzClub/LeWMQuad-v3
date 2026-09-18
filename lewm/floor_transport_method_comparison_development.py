"""Paired native method comparison, preserving failed and absent return evidence."""
from copy import deepcopy

MATCHED_LAUNCH_KEYS = ('scene_specification', 'public_mission', 'robot_urdf_sha256',
    'navigation_ticks', 'shared_outbound_return_budget', 'native_scene_workers',
    'opencv_threads', 'blas_threads', 'maximum_tasks_per_process',
    'minimum_free_bytes', 'planned_collection_allowance_bytes', 'persistence_headroom_bytes',
    'physics_paused_during_compute', 'renderer_environment', 'renderer_capture_witnesses_enabled',
    'input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
    'opencv_binary_sha256', 'opencv_version', 'rules')


def admit_comparison(reactive, reactive_launch, predictive_readout, predictive_launch):
    if (reactive['status'] != 'REACTIVE_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE'
            or len(reactive['conditions']) != 1
            or predictive_readout['status'] != 'MEASURED_FLOOR_TRANSPORT_MAZE_READOUT_V1_COMPLETE'
            or len(predictive_readout['conditions']) != 1
            or predictive_readout['original_outcome_unchanged'] is not True):
        raise ValueError('one completed current reactive native and unchanged learned readout required')
    if not (reactive['learned_result_sha256'] == reactive_launch['learned_result_sha256']
            == predictive_readout['native_result_sha256']):
        raise ValueError('the exact paired learned native result is required')
    rr = reactive['conditions'][0]; pr = predictive_readout['conditions'][0]
    if (rr['case'] != 'reactive_floor_transport_novel_maze_00' or rr['layout_index'] != 0
            or rr['status'] != 'REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
            or pr['case'] != 'full_jepa_novel_maze_00' or pr['layout_index'] != 0):
        raise ValueError('fixed paired maze0 cases required')
    for key in ('physical_and_public_prefix_exact', 'shared_observed_state_exact',
            'all_preintervention_requested_commands_exact', 'complete_candidate_decisions_match_prospective_prefix'):
        if rr['prefix_comparison'][key] is not True:
            raise ValueError('actual matched native prefix required: '+key)
    if (rr['prefix_comparison']['common_prefix_frames'] != 4
            or rr['prefix_comparison']['physical_prefix_samples'] != 900):
        raise ValueError('complete four-observation native intervention required')
    for key in MATCHED_LAUNCH_KEYS:
        if reactive_launch[key] != predictive_launch[key]:
            raise ValueError('matched scene/sensor/actuator/mission setting differs: '+key)
    if (reactive_launch['implementation_class'] != 'ReactiveFloorTransportController'
            or predictive_launch['implementation_class'] != 'MeasuredFloorTransportController'
            or reactive_launch['high_level_world_model_loaded'] is not False
            or reactive_launch['candidate_future_outcomes_evaluated'] is not False
            or reactive_launch['learned_residual_used'] is not False):
        raise ValueError('explicit current reactive and learned method definitions required')
    return dict(paired_learned_native_result_sha256=reactive['learned_result_sha256'],
        matched_launch_fields=list(MATCHED_LAUNCH_KEYS), physical_and_public_prefix_exact=True,
        same_observer_map_and_settling_state_at_intervention=True,
        predictive_feasibility_gates_matched=False, isolated_prediction_ranking_ablation=False)


def metrics(record):
    evaluation = record['native_evaluation']
    return deepcopy(dict(case=record['case'], verified_round_trip=record['verified_round_trip'],
        native_round_trip_candidate_pass=evaluation['native_round_trip_candidate_pass'],
        observed_arrivals=len(record['observed_arrival_transitions']),
        native_arrival_windows=evaluation['arrival_windows'],
        outbound_traversal=evaluation['outbound_traversal'], return_traversal=evaluation['return_traversal'],
        physically_retraced_outbound_route=evaluation['physically_retraced_outbound_route'],
        terminal_native_quiet_pass=evaluation['terminal_native_quiet_pass'],
        strict_physical_visibility_pass=record['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=record['hard_measurement_failed_frames'],
        native_xy_path_length_m=record['native_xy_path_length_m'],
        minimum_native_outbound_goal_distance_m=record['minimum_native_outbound_goal_distance_m'],
        terminal_native_outbound_goal_distance_m=record['terminal_native_outbound_goal_distance_m'],
        terminal_native_initial_xy_m=record['terminal_native_initial_xy_m'],
        simulated_duration_after_initial_observation_s=record['simulated_duration_after_initial_observation_s'],
        schedule_terminal=record['collection']['schedule_terminal'],
        physical_stop=record['collection']['physical_stop'], acquisition_stop=record['collection']['acquisition_stop'],
        selected_actions=record['selected_actions'], timing=record['timing']))


def compare_methods(reactive, predictive, admission):
    if reactive['layout_index'] != 0 or predictive['layout_index'] != 0:
        raise ValueError('paired reused development maze0 required')
    return dict(reactive=metrics(reactive), predictive=metrics(predictive), admission=deepcopy(admission),
        layout_index=0, reused_development_layout=True, independent_layout_executions=0,
        persistent_observed_memory_in_both=True, predictive_feasibility_gates_matched=False,
        isolated_prediction_ranking_ablation=False, single_layout_method_comparison=True,
        jepa_training_advantage_established=False, learned_planning_advantage_established=False,
        memory_advantage_established=False, statistical_reliability_established=False,
        independent_layout_generalization_established=False, real_time_qualified=False,
        hardware_qualified=False, original_outcomes_unchanged=True)

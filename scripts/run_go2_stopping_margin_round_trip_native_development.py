"""Prospective round trip with earlier braking and measured post-veto viewing."""
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from lewm.eligible_floor_registration_development import bind
from lewm.feature_budget_100_tracker_development import initialize_pose_100
from lewm.independent_depth_process_development import initialize_obstacles,obstacles_ready
from lewm.process_registered_round_trip_development import initialize_registration,registration_ready
from lewm.stopping_margin_dispatch_development import StoppingMarginRoundTripRuntime,STOPPING_ALLOWANCE_S
from lewm.veto_view_round_trip_development import VetoViewRoundTripRuntime
from lewm.fine_obstacle_round_trip_development import FineObstacleRoundTripRuntime,initialize_fine_obstacles
from lewm.partial_height_round_trip_development import (
    PartialHeightRoundTripRuntime,initialize_registration as initialize_partial_registration,
    initialize_mapping as initialize_partial_mapping,initialize_obstacles as initialize_partial_obstacles)
from lewm.waypoint_alignment_planning_development import WaypointAlignmentRoundTripRuntime
from lewm.optional_plane_refinement_development import initialize_optional_pose,initialize_optional_pose_150
from lewm.continuous_start_connector_development import ContinuousStartConnectorRuntime
from lewm.fine_stored_obstacle_routing_development import FineStoredObstacleRuntime,initialize_fine_mapping
from lewm.frontier_visit_runtime_development import FrontierVisitRuntime
from lewm.memory_forecast_clearance_development import MemoryForecastClearanceRuntime,TranslationReserveRuntime
from lewm.clearance_lookahead_development import ClearanceLookaheadRuntime,ReserveRecoveryLookaheadRuntime,StandOffFrontierRuntime
from lewm.closed_loop_motion_residual_development import MotionResidualRuntime,PanoramicMotionResidualRuntime,FIT_SHA256
from lewm.initial_panorama_development import InitialSurveyRuntime
from lewm.continuous_reactive_runtime_development import ContinuousReactiveRuntime
from lewm.clearance_preferred_reactive_runtime_development import ClearancePreferredReactiveRuntime
from lewm.clearance_turn_recovery_development import ClearanceTurnRecoveryRuntime,StepwiseClearanceTurnRecoveryRuntime
from lewm.clearance_preferred_route_development import ClearancePreferredTurnRecoveryRuntime
from lewm.terminal_position_priority_development import TerminalPositionPriorityRuntime,ProgressRejoiningTerminalRuntime
from lewm.fine_goal_route_development import FineGoalRouteRuntime
from lewm.predictive_arrival_hold_development import PredictiveArrivalHoldRuntime
from lewm.terminal_translation_pulse_development import TerminalTranslationPulseRuntime
from lewm import plane_consensus_tracker_development as plane_consensus
from lewm import pair_local_plane_consensus_development as pair_local_plane
from lewm import gyro_consensus_visual_motion_development as gyro_consensus_motion
from lewm import reused_flow_gyro_visual_motion_development as reused_flow_motion
from lewm import orthonormal_gyro_visual_motion_development as orthonormal_gyro_motion
from lewm.conditioned_support_tracker_development import initialize_conditioned_support_pose
from lewm.conditioned_support_150_tracker_development import initialize_conditioned_support_pose_150
from lewm.joint_camera_anchor_tracker_development import initialize_joint_camera_pose
from lewm import two_cm_floor_extent_development as floor_extent
from scripts.in_memory_paired_camera_session_development import InMemoryPairedCameraSession,RawDepthPairedCameraSession,ARCHIVE_WORKERS
from scripts import run_go2_paced_native_prefix_development as source

NAVIGATION_TICKS=4800
LAYOUT_INDEX=6
MODEL_ASSIGNMENT='seed_2026091001_full_jepa'
USE_CLEARANCE_TURN_RECOVERY=True
USE_STEPWISE_TURN_RECOVERY=True
USE_CLEARANCE_PREFERRED_ROUTE=True
USE_TERMINAL_POSITION_PRIORITY=True
USE_PROGRESS_REJOINING=True
USE_GYRO_CONSENSUS_POSE=True
USE_REUSED_CHAIN_FLOW=True
USE_ORTHONORMAL_GYRO=True
USE_FINE_GOAL_ROUTE=True
USE_PREDICTIVE_ARRIVAL_HOLD=True
USE_TERMINAL_TRANSLATION_PULSE=True
USE_RAW_DEPTH_ARCHIVE=True
OUTPUT=source.BASE/'go2_raw_depth_terminal_pulse_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        predictive=MODEL_ASSIGNMENT!='reactive'
        value=value|dict(host_real_time_execution_claimed=False,in_memory_camera_packets=True,
            camera_archive_lossless_compression_level=1,
            camera_archive_workers=ARCHIVE_WORKERS,
            terminal_position_priority=USE_TERMINAL_POSITION_PRIORITY,
            turn_recovery_releases_for_full_reserve_progress=USE_PROGRESS_REJOINING,
            terminal_position_priority_radius_m=.10 if USE_TERMINAL_POSITION_PRIORITY else None,
            final_mission_heading_required=False,
            clearance_preferred_routing=USE_CLEARANCE_PREFERRED_ROUTE,
            routing_soft_preferred_clearance_m=.60 if USE_CLEARANCE_PREFERRED_ROUTE else None,
            routing_clearance_cost_weight=2. if USE_CLEARANCE_PREFERRED_ROUTE else None,
            routing_entry_target_and_traversable_graph_unchanged=not USE_FINE_GOAL_ROUTE,
            fine_observed_goal_connectivity_fallback=USE_FINE_GOAL_ROUTE,
            fine_goal_fallback_changes_action_clearance_checks=False,
            turn_prediction_error_reserve_m=.03 if USE_CLEARANCE_TURN_RECOVERY else 0.,
            clear_alternative_turn_direction_latched=USE_CLEARANCE_TURN_RECOVERY,
            stepwise_turn_reserve_recovery=USE_STEPWISE_TURN_RECOVERY,
            blocked_translation_can_request_alternative_turn=USE_STEPWISE_TURN_RECOVERY,
            blocked_latch_can_switch_to_full_reserve_alternative=USE_STEPWISE_TURN_RECOVERY,
            recovery_minimum_gain_m=.001 if USE_STEPWISE_TURN_RECOVERY else None,
            recovery_minimum_fraction_of_deficit=.1 if USE_STEPWISE_TURN_RECOVERY else None,
            learned_model_used=predictive,instantaneous_waypoint_feedback=not predictive,
            same_continuous_perception_memory_and_mission=not USE_GYRO_CONSENSUS_POSE,
            perception_changed_from_fixed_transfer_controller=USE_GYRO_CONSENSUS_POSE,
            gyro_conditioned_image_consensus_estimator=USE_GYRO_CONSENSUS_POSE,
            exact_chained_image_link_reuse=USE_REUSED_CHAIN_FLOW,
            gyro_rotation_reorthogonalized_each_camera_interval=USE_ORTHONORMAL_GYRO,
            predictive_terminal_arrival_hold=USE_PREDICTIVE_ARRIVAL_HOLD,
            terminal_translation_pulses=USE_TERMINAL_TRANSLATION_PULSE,
            terminal_translation_command_duration_ns=100_000_000 if USE_TERMINAL_TRANSLATION_PULSE else 400_000_000,
            maximum_command_duration_ns=400_000_000,
            native_depth_only_archive=USE_RAW_DEPTH_ARCHIVE,
            derived_depth_capture_digest_retained=USE_RAW_DEPTH_ARCHIVE,
            timed_live_camera_packets_unchanged=True,
            predictive_hold_terminal_xy_speed_limit_m_s=.05 if USE_PREDICTIVE_ARRIVAL_HOLD else None,
            gyro_role='rotation_estimator' if USE_GYRO_CONSENSUS_POSE else 'consistency_monitor_only',
            gyro_bias_estimated=False,gyro_noise_model_validated=False,
            independent_current_depth_process=True,independent_registration_process=True,
            visual_tracking_in_main_process=False,vectorized_same_connector_geometry=True,
            simulator_lag_forces_zero=False,simulation_release_resolution_ns=2_000_000,
            corner_budget_per_camera=150,command_duration_ns=None if USE_TERMINAL_TRANSLATION_PULSE else 400_000_000,
            actual_prefix_checked_before_dispatch=True,full_mission_implemented=True,
            measured_round_trip_enabled=True,navigation_tick_budget=NAVIGATION_TICKS,
            observed_arrival_radius_m=.02,physical_arrival_requirement_m=.04,
            arrival_pose_error_reserve_calibrated=False,exact_public_goal_route_endpoint=True,
            stopping_allowance_s=STOPPING_ALLOWANCE_S,observation_age_added_to_translation_projection=True,
            stopping_distance_bound_calibrated=False,
            translation_veto_requests_measured_new_view=True,
            current_obstacle_cell_m=.01,current_obstacle_crop_half_extent_m=1.,
            nominal_footprint_radius_m=.45,
            explicit_partial_floor_height_observations=True,
            partial_height_normal_measured_from_current_points=False,
            waypoint_predicted_alignment_progress=predictive,waypoint_alignment_scale_cap_m=.35 if predictive else None,
            optional_plane_refinement_preserves_qualified_image_fit=True,
            routing_start_connector_continuous_nominal_disk=True,
            stored_obstacle_cell_m=.01,coarse_interior_route_inflation_unchanged=not USE_FINE_GOAL_ROUTE,
            measured_frontier_visit_completion=True,frontier_arrival_radius_m=None,
            completed_frontier_views_are_physical_arrivals=False,
            stored_map_filters_model_forecast_paths=predictive,stored_map_forecast_horizon_ns=800_000_000 if predictive else None,
            translation_prediction_error_reserve_m=.03 if predictive else None,prediction_error_bound_calibrated=False,
            stored_obstacle_checked_route_lookahead=True,lookahead_clearance_cap_m=.48,
            lookahead_preserves_available_start_clearance=True,
            exact_clearance_aabb_lower_bound_pruning=False,exact_spatial_index_clearance_query=True,
            cached_obstacle_geometry=True,cached_bitmap_obstacle_inflation=True,
            reserve_recovery_requires_no_further_encroachment=predictive,
            reserve_recovery_requires_full_reserve_by_commit_end=predictive and not USE_STEPWISE_TURN_RECOVERY,
            reserve_recovery_nominal_footprint_unchanged=predictive,
            frontier_view_standoff_route_m=.50,
            frontier_view_heading_from_actual_viewpoint=True,
            frontier_panorama=True,frontier_panorama_heading_stages=9,
            frontier_panorama_step_radians=0.7853981633974483,
            initial_panorama_before_translating_route=True,initial_panorama_heading_stages=9,
            joint_camera_retained_anchor_fallback=True,camera_specific_reprojection=True,
            joint_camera_previous_frame_fallback=True,
            floor_minimum_second_extent_m=.02,original_floor_extent_gate_changed=True,
            robust_plane_image_consensus=True,all_original_image_inliers_required=False,
            floor_height_conflict_rejection_scope='image_reference_pair',
            conflicting_image_pose_admitted=False,
            original_pair_acceptance_checks_unchanged=not USE_GYRO_CONSENSUS_POSE,
            height_conflict_requires_new_consistent_fit=True,
            floor_all_point_residual_limit_m=.003,floor_noise_uncertainty_calibrated=False,
            closed_loop_motion_residual_fit_sha256=FIT_SHA256 if predictive else None,
            residual_uses_four_causal_registered_poses=predictive,
            residual_neural_weights_changed=False,residual_yaw_and_contact_changed=False,
            image_bin_count_gate_used=False,strict_inlier_majority_required=True,
            original_measured_3d_conditioning_required=True,
            extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                __file__,'scripts/in_memory_paired_camera_session_development.py',
                'scripts/raw_depth_archive_development.py',
                'scripts/in_memory_public_replay_development.py',
                'lewm/feature_budget_100_tracker_development.py','lewm/independent_depth_obstacle_development.py',
                'lewm/independent_depth_runtime_development.py','lewm/independent_depth_process_development.py',
                'lewm/measured_latency_simulation_development.py','lewm/continuous_commitment_runtime_development.py',
                'lewm/continuous_commitment_ledger_development.py','lewm/continuous_round_trip_runtime_development.py',
                'lewm/process_registered_round_trip_development.py','lewm/vectorized_connector_routing_development.py',
                'lewm/stopping_margin_dispatch_development.py','lewm/veto_view_round_trip_development.py',
                'lewm/fine_obstacle_round_trip_development.py','lewm/partial_floor_height_development.py',
                'lewm/partial_height_round_trip_development.py',
                'lewm/waypoint_alignment_planning_development.py',
                'lewm/measured_plane_dual_camera_pose_development.py',
                'lewm/optional_plane_refinement_development.py',
                'lewm/observed_floor_waypoint_development.py',
                'lewm/observed_round_trip_mission_development.py',
                'lewm/observed_geometry_refinement_development.py',
                'lewm/continuous_start_connector_development.py',
                'lewm/multirate_routing_map_development.py',
                'lewm/fine_stored_obstacle_routing_development.py',
                'lewm/frontier_visit_runtime_development.py',
                'lewm/memory_forecast_clearance_development.py',
                'lewm/clearance_lookahead_development.py',
                'lewm/closed_loop_motion_residual_development.py',
                'lewm/initial_panorama_development.py',
                'lewm/continuous_reactive_selection_development.py',
                'lewm/continuous_reactive_runtime_development.py',
                'lewm/clearance_preferred_reactive_runtime_development.py',
                'lewm/clearance_turn_recovery_development.py',
                'lewm/clearance_preferred_route_development.py',
                'lewm/terminal_position_priority_development.py',
                'lewm/fine_goal_route_development.py',
                'lewm/predictive_arrival_hold_development.py',
                'lewm/terminal_translation_pulse_development.py',
                'lewm/continuous_commitment_ledger_development.py',
                'lewm/paced_multirate_controller_development.py',
                'scripts/fit_closed_loop_motion_residual_development.py',
                'lewm/development_support_tracker_development.py',
                'lewm/conditioned_support_tracker_development.py',
                'lewm/joint_camera_registration_development.py',
                'lewm/joint_camera_anchor_tracker_development.py',
                'lewm/joint_measured_floor_plane_development.py',
                'lewm/two_cm_floor_extent_development.py',
                'lewm/plane_consensus_tracker_development.py',
                'lewm/pair_local_plane_consensus_development.py',
                'lewm/gyro_conditioned_pair_pose_development.py',
                'lewm/gyro_consensus_pair_pose_development.py',
                'lewm/gyro_consensus_visual_motion_development.py',
                'lewm/reused_flow_gyro_visual_motion_development.py',
                'lewm/orthonormal_gyro_visual_motion_development.py',
                'lewm/chained_flow_memo_development.py',
                'lewm/batched_patch_tracker_development.py',
                'lewm/batched_patch_agreement_development.py',
                'lewm/chained_corner_flow_association_development.py',
                'lewm/joint_sensor_anchored_goal_development.py',
                'lewm/joint_floor_registered_evidence_development.py',
                'lewm/measured_floor_transport_development.py',
                'lewm/measured_floor_transport_registration_development.py',
                'lewm/fresh_obstacle_dispatch_development.py',
                'lewm/dual_camera_visual_motion_development.py',
                'lewm/dual_camera_anchor_pose_development.py',
                'lewm/feature_budget_150_tracker_development.py',
                'lewm/conditioned_support_150_tracker_development.py')})
    _write(name,value)


if __name__=='__main__':
    floor_extent.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=floor_extent.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        if MODEL_ASSIGNMENT=='reactive' and USE_CLEARANCE_TURN_RECOVERY:
            raise ValueError('forecast-based turn recovery is a separate learned treatment')
        if USE_CLEARANCE_PREFERRED_ROUTE and MODEL_ASSIGNMENT!='reactive' and not USE_STEPWISE_TURN_RECOVERY:
            raise ValueError('this routing treatment extends the stepwise learned arm')
        if USE_TERMINAL_POSITION_PRIORITY and (MODEL_ASSIGNMENT=='reactive' or not USE_CLEARANCE_PREFERRED_ROUTE):
            raise ValueError('terminal position priority extends the preferred learned arm')
        if USE_PROGRESS_REJOINING and not USE_TERMINAL_POSITION_PRIORITY:
            raise ValueError('progress rejoining extends the terminal-position learned arm')
        if USE_FINE_GOAL_ROUTE and (MODEL_ASSIGNMENT=='reactive' or not USE_PROGRESS_REJOINING):
            raise ValueError('fine goal routing extends the progress-rejoining learned arm')
        if USE_REUSED_CHAIN_FLOW and not USE_GYRO_CONSENSUS_POSE:
            raise ValueError('this image-link reuse treatment extends gyro-consensus perception')
        if USE_ORTHONORMAL_GYRO and not USE_REUSED_CHAIN_FLOW:
            raise ValueError('orthonormal gyro extends the reused-flow gyro estimator')
        if USE_PREDICTIVE_ARRIVAL_HOLD and not USE_FINE_GOAL_ROUTE:
            raise ValueError('predictive arrival hold extends fine-goal navigation')
        if USE_TERMINAL_TRANSLATION_PULSE and not USE_PREDICTIVE_ARRIVAL_HOLD:
            raise ValueError('terminal pulses extend the predictive-arrival-hold runtime')
        runtime_type=((ClearancePreferredReactiveRuntime if USE_CLEARANCE_PREFERRED_ROUTE else ContinuousReactiveRuntime)
            if MODEL_ASSIGNMENT=='reactive' else
            TerminalTranslationPulseRuntime if USE_TERMINAL_TRANSLATION_PULSE else
            PredictiveArrivalHoldRuntime if USE_PREDICTIVE_ARRIVAL_HOLD else
            FineGoalRouteRuntime if USE_FINE_GOAL_ROUTE else
            ProgressRejoiningTerminalRuntime if USE_PROGRESS_REJOINING else
            TerminalPositionPriorityRuntime if USE_TERMINAL_POSITION_PRIORITY else
            ClearancePreferredTurnRecoveryRuntime if USE_CLEARANCE_PREFERRED_ROUTE else
            StepwiseClearanceTurnRecoveryRuntime if USE_STEPWISE_TURN_RECOVERY else
            ClearanceTurnRecoveryRuntime if USE_CLEARANCE_TURN_RECOVERY else InitialSurveyRuntime)
        def runtime(*args,**kwargs):return runtime_type(*args,registration_executor=executor,
            navigation_ticks=NAVIGATION_TICKS,arrival_radius_m=.02,**kwargs)
        bind(source.main,OUTPUT=OUTPUT,COUNT=NAVIGATION_TICKS+14,LAYOUT_INDEX=LAYOUT_INDEX,
            MODEL_ASSIGNMENT=MODEL_ASSIGNMENT,
            PacedNativeSession=RawDepthPairedCameraSession if USE_RAW_DEPTH_ARCHIVE else InMemoryPairedCameraSession,write=write,
            CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=orthonormal_gyro_motion.initialize_pose if USE_ORTHONORMAL_GYRO else
                reused_flow_motion.initialize_pose if USE_REUSED_CHAIN_FLOW else
                gyro_consensus_motion.initialize_pose if USE_GYRO_CONSENSUS_POSE else pair_local_plane.initialize_pose,
            MEASURED_RUNTIME_CLASS=runtime,OBSTACLE_INITIALIZER=floor_extent.initialize_obstacles,OBSTACLE_READY=obstacles_ready,
            initialize_mapping=floor_extent.initialize_mapping)()

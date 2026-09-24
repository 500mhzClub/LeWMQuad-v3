"""Prospective stable-reference trial using the existing compiled tracker."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.cached_fine_goal_route_development import CachedFineGoalRecoveryRuntime
from lewm.pulsed_fine_goal_reactive_development import PulsedFineGoalReactiveRuntime
from lewm.process_registered_round_trip_development import registration_ready
from lewm.independent_depth_process_development import obstacles_ready
from scripts import run_go2_stopping_margin_round_trip_native_development as configuration
from scripts.in_memory_paired_camera_session_development import LzmaRawDepthPairedCameraSession

source = configuration.source
floor = configuration.floor_extent
OUTPUT = None
REACTIVE = False
ARRIVAL_ENTRY_PRIORITY = False


def initialize_pose():
    import cv2
    import torch
    from lewm.stable_gyro_reference_development import CompiledFloorStableGyroReferenceMotion
    from lewm import process_mapped_runtime_development as process
    floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = CompiledFloorStableGyroReferenceMotion(activation_frame=0)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='stable_reference_selection',
            terminal_priority_requires_predicted_arrival=ARRIVAL_ENTRY_PRIORITY,
            output_volume='RecoveryStorage',
            reference_retention_replay_treatment_used=True,
            stable_reference_selection=True, stable_reference_activation_frame=0,
            maximum_retained_references=8, low_overlap_reference_refresh_preserved=True,
            compiled_floor_candidate_predicates=True,
            floor_kernel_warmup_before_timed_capture=True,
            exact_fine_goal_segment_cache=True,
            fine_goal_segment_cache_entries_per_geometry=16384,
            fine_goal_segment_cache_obstacle_sets=2,
            camera_archive_compression='ZIP_LZMA', camera_archive_lossless_compression_level=None,
            camera_archive_workers=12, hold_relative_clearance_recovery=not REACTIVE,
            candidate_future_outcomes_evaluated=not REACTIVE,
            world_model_and_motion_residual_used=not REACTIVE,
            comparison='complete_predictive_selection_versus_instantaneous_reactive_selection',
            jepa_specific_advantage_established=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/stable_gyro_reference_development.py',
                    'lewm/jit_floor_candidates_development.py',
                    'lewm/jit_floor_gyro_visual_motion_development.py',
                    'lewm/cached_fine_goal_route_development.py',
                    'lewm/hold_relative_clearance_recovery_development.py',
                    'lewm/pulsed_fine_goal_reactive_development.py')})
        if ARRIVAL_ENTRY_PRIORITY:
            path = 'lewm/arrival_entry_terminal_priority_development.py'
            value['extra_sources'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    bind(source.write, OUTPUT=OUTPUT)(name, value)


def main():
    global OUTPUT, REACTIVE, ARRIVAL_ENTRY_PRIORITY
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(8), default=4)
    parser.add_argument('--reactive', action='store_true')
    parser.add_argument('--arrival-entry-priority', action='store_true')
    args = parser.parse_args(); REACTIVE = args.reactive
    ARRIVAL_ENTRY_PRIORITY = args.arrival_entry_priority
    if REACTIVE and ARRIVAL_ENTRY_PRIORITY:
        parser.error('the terminal prediction override applies to the learned selector')
    arm = 'reactive' if REACTIVE else 'learned'
    prefix = 'arrival_entry_priority_' if ARRIVAL_ENTRY_PRIORITY else ''
    OUTPUT = source.BASE/f'go2_{prefix}stable_reference_jit_floor_cached_fine_goal_lzma_pulse_{arm}_round_trip_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    source.validate_root(OUTPUT, must_exist=False)
    assignment = 'reactive' if REACTIVE else configuration.MODEL_ASSIGNMENT
    writer = bind(configuration.write, _write=write, MODEL_ASSIGNMENT=assignment,
        USE_CLEARANCE_TURN_RECOVERY=not REACTIVE, USE_STEPWISE_TURN_RECOVERY=not REACTIVE,
        USE_TERMINAL_POSITION_PRIORITY=not REACTIVE, USE_PROGRESS_REJOINING=not REACTIVE,
        USE_PREDICTIVE_ARRIVAL_HOLD=not REACTIVE)
    floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        runtime_type = PulsedFineGoalReactiveRuntime if REACTIVE else CachedFineGoalRecoveryRuntime
        if ARRIVAL_ENTRY_PRIORITY:
            from lewm.arrival_entry_terminal_priority_development import ArrivalEntryTerminalPriorityRuntime
            runtime_type = ArrivalEntryTerminalPriorityRuntime
        def runtime(*args, **kwargs):
            return runtime_type(*args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)
        bind(source.main, OUTPUT=OUTPUT, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=LzmaRawDepthPairedCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=floor.initialize_obstacles, OBSTACLE_READY=obstacles_ready,
            initialize_mapping=floor.initialize_mapping)()


if __name__ == '__main__':
    main()

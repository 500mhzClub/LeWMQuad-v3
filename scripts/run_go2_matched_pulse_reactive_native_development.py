"""Prospective reactive comparison with the current camera, routing and timing."""
import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from lewm.eligible_floor_registration_development import bind
from lewm.pulsed_fine_goal_reactive_development import PulsedFineGoalReactiveRuntime
from lewm.orthonormal_gyro_visual_motion_development import initialize_pose
from lewm.jit_floor_gyro_visual_motion_development import initialize_pose as initialize_jit_floor_pose
from lewm.process_registered_round_trip_development import registration_ready
from lewm.independent_depth_process_development import obstacles_ready
from scripts import navigation_artifact_root_development as artifact
from scripts import run_go2_stopping_margin_round_trip_native_development as configuration
from scripts.in_memory_paired_camera_session_development import LzmaRawDepthPairedCameraSession

source = configuration.source
floor = configuration.floor_extent
OUTPUT_BASE = Path('/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
OUTPUT = None
USE_JIT_FLOOR = False


def output_write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='matched_current_pulse_reactive_comparison',
            output_volume='workspace', model_input_volume=None,
            compiled_floor_candidate_predicates=USE_JIT_FLOOR,
            floor_kernel_warmup_before_timed_capture=USE_JIT_FLOOR,
            exact_fine_goal_segment_cache=True,
            fine_goal_segment_cache_entries_per_geometry=16384,
            fine_goal_segment_cache_obstacle_sets=2,
            camera_archive_compression='ZIP_LZMA', camera_archive_lossless_compression_level=None,
            camera_archive_workers=12,
            same_current_gyro_perception_observed_memory_and_goal_routing=True,
            terminal_translation_pulse_radius_m=.10,
            terminal_position_priority=False, predictive_terminal_arrival_hold=False,
            hold_relative_clearance_recovery=False, candidate_future_outcomes_evaluated=False,
            world_model_and_motion_residual_used=False,
            comparison='complete_predictive_selection_versus_instantaneous_reactive_selection',
            jepa_specific_advantage_established=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/pulsed_fine_goal_reactive_development.py',
                    'lewm/cached_fine_goal_route_development.py',
                    'lewm/jit_floor_candidates_development.py',
                    'lewm/jit_floor_gyro_visual_motion_development.py')})
    bind(source.write, OUTPUT=OUTPUT)(name, value)


write = bind(configuration.write, MODEL_ASSIGNMENT='reactive',
    USE_CLEARANCE_TURN_RECOVERY=False, USE_STEPWISE_TURN_RECOVERY=False,
    USE_TERMINAL_POSITION_PRIORITY=False, USE_PROGRESS_REJOINING=False,
    USE_PREDICTIVE_ARRIVAL_HOLD=False, _write=output_write)


def main():
    global OUTPUT, USE_JIT_FLOOR
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(8), default=4)
    parser.add_argument('--jit-floor', action='store_true')
    args = parser.parse_args()
    USE_JIT_FLOOR = args.jit_floor
    prefix = 'jit_floor_' if USE_JIT_FLOOR else ''
    OUTPUT = OUTPUT_BASE/f'go2_{prefix}cached_fine_goal_lzma_pulse_reactive_round_trip_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    validate_output = bind(artifact.validate_root, BASE=OUTPUT_BASE)
    validate_output(OUTPUT, must_exist=False)
    OUTPUT_BASE.mkdir(exist_ok=True)
    floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        def runtime(*args, **kwargs):
            return PulsedFineGoalReactiveRuntime(*args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)
        bind(source.main, OUTPUT=OUTPUT, validate_root=validate_output,
            COUNT=4814, LAYOUT_INDEX=args.layout_index, MODEL_ASSIGNMENT='reactive',
            PacedNativeSession=LzmaRawDepthPairedCameraSession, write=write,
            CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_jit_floor_pose if USE_JIT_FLOOR else initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=floor.initialize_obstacles, OBSTACLE_READY=obstacles_ready,
            initialize_mapping=floor.initialize_mapping)()


if __name__ == '__main__':
    main()

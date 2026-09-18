"""Prospective clearance recovery trial on the workspace storage volume."""
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from lewm.eligible_floor_registration_development import bind
from lewm.hold_relative_clearance_recovery_development import HoldRelativeClearanceRecoveryRuntime
from lewm.cached_fine_goal_route_development import CachedFineGoalRecoveryRuntime
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
LAYOUT_INDEX = 4
USE_CACHED_FINE_GOAL = False
USE_LZMA_ARCHIVE = False
USE_JIT_FLOOR = False
OUTPUT = OUTPUT_BASE/f'go2_hold_relative_recovery_pulse_round_trip_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001'


def output_write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='hold_relative_clearance_recovery',
            output_volume='workspace', model_input_volume='RecoveryStorage',
            reference_retention_replay_treatment_used=False,
            compiled_floor_candidate_predicates=USE_JIT_FLOOR,
            floor_kernel_warmup_before_timed_capture=USE_JIT_FLOOR,
            exact_fine_goal_segment_cache=USE_CACHED_FINE_GOAL,
            fine_goal_segment_cache_entries_per_geometry=16384 if USE_CACHED_FINE_GOAL else 0,
            fine_goal_segment_cache_obstacle_sets=2 if USE_CACHED_FINE_GOAL else 0,
            camera_archive_compression='ZIP_LZMA' if USE_LZMA_ARCHIVE else 'ZIP_DEFLATED',
            camera_archive_lossless_compression_level=None if USE_LZMA_ARCHIVE else 1,
            camera_archive_workers=12 if USE_LZMA_ARCHIVE else configuration.ARCHIVE_WORKERS,
            hold_relative_clearance_recovery=True,
            recovery_requires_minimum_clearance_no_worse_than_hold=True,
            recovery_requires_endpoint_gain_over_hold=True,
            recovery_nominal_footprint_radius_m=.45,
            recovery_minimum_gain_m=.001, recovery_minimum_fraction_of_deficit=.1,
            reserve_recovery_requires_no_further_encroachment=False,
            recovery_comparison_baseline='hold_forecast_minimum_and_endpoint',
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/hold_relative_clearance_recovery_development.py',
                    'lewm/cached_fine_goal_route_development.py',
                    'lewm/jit_floor_candidates_development.py',
                    'lewm/jit_floor_gyro_visual_motion_development.py')})
    bind(source.write, OUTPUT=OUTPUT)(name, value)


write = bind(configuration.write, _write=output_write)


def main():
    global LAYOUT_INDEX, OUTPUT, USE_CACHED_FINE_GOAL, USE_LZMA_ARCHIVE, USE_JIT_FLOOR
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(8), default=4)
    parser.add_argument('--cached-fine-goal', action='store_true')
    parser.add_argument('--lzma-archive', action='store_true')
    parser.add_argument('--jit-floor', action='store_true')
    args = parser.parse_args()
    LAYOUT_INDEX = args.layout_index
    USE_CACHED_FINE_GOAL = args.cached_fine_goal
    USE_LZMA_ARCHIVE = args.lzma_archive
    USE_JIT_FLOOR = args.jit_floor
    prefix = ('jit_floor_' if USE_JIT_FLOOR else '') + ('cached_fine_goal_' if USE_CACHED_FINE_GOAL else '') + ('lzma_' if USE_LZMA_ARCHIVE else '')
    OUTPUT = OUTPUT_BASE/f'go2_{prefix}hold_relative_recovery_pulse_round_trip_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001'
    validate_output = bind(artifact.validate_root, BASE=OUTPUT_BASE)
    validate_output(OUTPUT, must_exist=False)
    OUTPUT_BASE.mkdir(exist_ok=True)
    floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        runtime_type = CachedFineGoalRecoveryRuntime if USE_CACHED_FINE_GOAL else HoldRelativeClearanceRecoveryRuntime
        def runtime(*args, **kwargs):
            return runtime_type(*args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)
        bind(source.main, OUTPUT=OUTPUT, validate_root=validate_output,
            COUNT=4814, LAYOUT_INDEX=LAYOUT_INDEX, MODEL_ASSIGNMENT=configuration.MODEL_ASSIGNMENT,
            PacedNativeSession=LzmaRawDepthPairedCameraSession if USE_LZMA_ARCHIVE else configuration.RawDepthPairedCameraSession, write=write,
            CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_jit_floor_pose if USE_JIT_FLOOR else initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=floor.initialize_obstacles, OBSTACLE_READY=obstacles_ready,
            initialize_mapping=floor.initialize_mapping)()


if __name__ == '__main__':
    main()

"""Short closed-loop terminal-control study; not an unseen-maze result."""
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from lewm.eligible_floor_registration_development import bind
from lewm.terminal_translation_pulse_development import TerminalTranslationPulseRuntime
from lewm.orthonormal_gyro_visual_motion_development import initialize_pose
from lewm.process_registered_round_trip_development import registration_ready
from lewm.independent_depth_process_development import obstacles_ready
from scripts import run_go2_stopping_margin_round_trip_native_development as configuration

source = configuration.source
floor = configuration.floor_extent
NAVIGATION_TICKS = 900
ARM = 'pulse'
GOAL_X_M = .09
OUTPUT = source.BASE/'go2_terminal_100ms_pulse_local_round_trip_layout06_v1_attempt_001'


def mission(index):
    if index != 6: raise ValueError('fixed local control-study scene required')
    return dict(goal_initial_body_xy_m=[GOAL_X_M, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True)


def output_write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='local_terminal_translation_pulse_control',
            local_control_arm=ARM,
            unseen_maze_navigation_result=False, terminal_translation_pulses=ARM=='pulse',
            terminal_translation_command_duration_ns=100_000_000 if ARM=='pulse' else 400_000_000,
            command_duration_ns=None if ARM=='pulse' else 400_000_000, maximum_command_duration_ns=400_000_000,
            planning_cadence_ns=400_000_000, scoring_endpoint_offset_ns=700_000_000,
            short_pulse_frozen_residual_accuracy_established=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/terminal_translation_pulse_development.py',
                    'lewm/continuous_commitment_ledger_development.py',
                    'lewm/delayed_action_planning_development.py')})
    bind(source.write, OUTPUT=OUTPUT)(name, value)


write = bind(configuration.write, NAVIGATION_TICKS=NAVIGATION_TICKS, _write=output_write)


def main():
    global ARM, OUTPUT, GOAL_X_M
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('pulse', 'standard'), default='pulse')
    parser.add_argument('--goal-x-m', type=float, choices=(.09, .65), default=.65)
    args = parser.parse_args(); ARM = args.arm; GOAL_X_M = args.goal_x_m
    treatment = '100ms_pulse' if ARM=='pulse' else 'standard400ms'
    goal_tag = '_goal065' if GOAL_X_M==.65 else ''
    OUTPUT = source.BASE/f'go2_terminal_{treatment}{goal_tag}_local_round_trip_layout06_v1_attempt_001'
    runtime_type = TerminalTranslationPulseRuntime if ARM=='pulse' else configuration.PredictiveArrivalHoldRuntime
    floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'), initializer=floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        def runtime(*args, **kwargs):
            return runtime_type(*args, registration_executor=executor,
                navigation_ticks=NAVIGATION_TICKS, arrival_radius_m=.02, **kwargs)
        bind(source.main, OUTPUT=OUTPUT, COUNT=NAVIGATION_TICKS+14, LAYOUT_INDEX=6,
            MODEL_ASSIGNMENT=configuration.MODEL_ASSIGNMENT, public_mission=mission,
            PacedNativeSession=configuration.RawDepthPairedCameraSession if configuration.USE_RAW_DEPTH_ARCHIVE else configuration.InMemoryPairedCameraSession, write=write,
            CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=floor.initialize_obstacles, OBSTACLE_READY=obstacles_ready,
            initialize_mapping=floor.initialize_mapping)()


if __name__ == '__main__': main()

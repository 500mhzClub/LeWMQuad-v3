"""Run the fixed successful controller on four new development mazes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
from pathlib import Path

from lewm.arrival_entry_terminal_priority_development import ArrivalEntryTerminalPriorityRuntime
from lewm.eligible_floor_registration_development import bind
from lewm import fresh_stable_reference_layouts_development as layouts
from lewm.pulsed_fine_goal_reactive_development import PulsedFineGoalReactiveRuntime
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_stable_reference_native_development as stable
from scripts.independent_round_trip_session_development import IndependentRoundTripPhysicalInit
from scripts.in_memory_paired_camera_session_development import LzmaRawDepthPairedCameraSession


class FreshPhysicalInit(IndependentRoundTripPhysicalInit):
    __init__ = bind(IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    """Keep all acquisition wrappers; replace only the scene initializer."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--reactive', action='store_true')
    args = parser.parse_args()
    arm = 'reactive' if args.reactive else 'learned'
    output = stable.source.BASE/f'go2_fresh_stable_reference_{arm}_round_trip_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve previous fresh-maze attempt')
    assignment = 'reactive' if args.reactive else stable.configuration.MODEL_ASSIGNMENT
    stable_writer = bind(stable.write, OUTPUT=output,
        REACTIVE=args.reactive, ARRIVAL_ENTRY_PRIORITY=not args.reactive)

    def write(name, value):
        if name == 'launch.json':
            value = value | dict(fresh_fixed_controller_transfer=True,
                fresh_layout_inventory=layouts.build_inventory(),
                extra_sources=value['extra_sources'] | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in (__file__, 'lewm/fresh_stable_reference_layouts_development.py',
                        'lewm/independent_round_trip_layouts_development.py',
                        'scripts/independent_round_trip_session_development.py')})
        stable_writer(name, value)

    writer = bind(stable.configuration.write, _write=write, MODEL_ASSIGNMENT=assignment,
        USE_CLEARANCE_TURN_RECOVERY=not args.reactive,
        USE_STEPWISE_TURN_RECOVERY=not args.reactive,
        USE_TERMINAL_POSITION_PRIORITY=not args.reactive,
        USE_PROGRESS_REJOINING=not args.reactive,
        USE_PREDICTIVE_ARRIVAL_HOLD=not args.reactive)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        runtime_type = PulsedFineGoalReactiveRuntime if args.reactive else ArrivalEntryTerminalPriorityRuntime

        def runtime(*args, **kwargs):
            return runtime_type(*args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=FreshCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__':
    main()

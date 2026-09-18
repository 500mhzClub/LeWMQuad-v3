"""Prospective recovery-release trial on the same four development mazes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.full_reserve_heading_release_development import FullReserveHeadingReleaseRuntime
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_fresh_stable_reference_native_development as fresh

stable = fresh.stable
layouts = fresh.layouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    output = stable.source.BASE/f'go2_heading_release_fresh_learned_round_trip_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve prior recovery-release attempt')
    assignment = stable.configuration.MODEL_ASSIGNMENT
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=False, ARRIVAL_ENTRY_PRIORITY=True)

    def write(name, value):
        if name == 'launch.json':
            value = value | dict(full_reserve_heading_release=True,
                terminal_approach_excluded_from_heading_release=True,
                fresh_layout_inventory=layouts.build_inventory(),
                baseline_root_name=f'go2_fresh_stable_reference_learned_round_trip_native_layout{args.layout_index:02d}_4800_v1_attempt_001',
                extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in (__file__, fresh.__file__,
                        'lewm/full_reserve_heading_release_development.py',
                        'lewm/fresh_stable_reference_layouts_development.py',
                        'lewm/independent_round_trip_layouts_development.py',
                        'scripts/independent_round_trip_session_development.py')})
        stable_writer(name, value)

    writer = bind(stable.configuration.write, _write=write, MODEL_ASSIGNMENT=assignment,
        USE_CLEARANCE_TURN_RECOVERY=True, USE_STEPWISE_TURN_RECOVERY=True,
        USE_TERMINAL_POSITION_PRIORITY=True, USE_PROGRESS_REJOINING=True,
        USE_PREDICTIVE_ARRIVAL_HOLD=True)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            return FullReserveHeadingReleaseRuntime(*args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=fresh.FreshCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__': main()

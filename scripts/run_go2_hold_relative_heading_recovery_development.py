"""Prospective pure-turn recovery against the completed compact directed trials."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.hold_relative_heading_recovery_development import HoldRelativeHeadingRecoveryMixin
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_nearby_panorama_comparison_development as prior


class HeadingRecoveryRuntime(HoldRelativeHeadingRecoveryMixin, prior.DirectedRevisitRuntime):
    pass


def annotate_write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='hold_relative_heading_recovery_native_development',
            comparison_condition='heading_recovery', hold_relative_heading_recovery=True,
            baseline_root_name=f'go2_nearby_panorama_directed_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/hold_relative_heading_recovery_development.py')})
    RAW_WRITE(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != prior.CPU_GROUPS[args.layout_index]:
        raise ValueError('launch with the assigned layout CPU group')
    stable = prior.prior.fresh.stable; layouts = prior.prior.fresh.layouts
    assignment = 'seed_2026091001_full_jepa'
    output = stable.source.BASE/f'go2_hold_relative_heading_recovery_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve previous heading-recovery trial')
    correction_root, correction_hash = prior.prior.FITS['jepa']
    annotated = bind(annotate_write, LAYOUT_INDEX=args.layout_index,
        RAW_WRITE=bind(stable.source.write, OUTPUT=output))
    final_write = bind(prior.finish_write, CORRECTION_HASH=correction_hash,
        CORRECTION_ROOT=correction_root, LAYOUT_INDEX=args.layout_index,
        CONDITION='directed', RAW_WRITE=annotated)
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=False,
        ARRIVAL_ENTRY_PRIORITY=True, source=SimpleNamespace(write=final_write))
    writer = bind(stable.configuration.write, _write=stable_writer,
        MODEL_ASSIGNMENT=assignment, USE_CLEARANCE_TURN_RECOVERY=True,
        USE_STEPWISE_TURN_RECOVERY=True, USE_TERMINAL_POSITION_PRIORITY=True,
        USE_PROGRESS_REJOINING=True, USE_PREDICTIVE_ARRIVAL_HOLD=True)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'jepa':
                raise ValueError('fixed JEPA model required')
            return HeadingRecoveryRuntime(*runtime_args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=prior.CompactFreshCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__':
    main()

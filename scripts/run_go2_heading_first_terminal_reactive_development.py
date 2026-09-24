"""Prospective stronger reactive control on the fixed four development mazes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.heading_first_terminal_reactive_development import HeadingFirstTerminalReactiveMixin
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_post_repeatability_transfer_development as original


class HeadingFirstReactiveRuntime(HeadingFirstTerminalReactiveMixin, original.TransferReactiveRuntime):
    pass


def annotate_write(name, value):
    if name == 'launch.json':
        reference_name = f'go2_post_repeatability_transfer_reactive_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001'
        reference = json.loads((original.stable.source.BASE/reference_name/'launch.json').read_text())
        changed = [k for k in reference.keys() | value.keys()
            if k != 'owner' and reference.get(k) != value.get(k)]
        if changed:
            raise ValueError(f'original reactive configuration changed: {changed}')
        value = value | dict(experiment='heading_first_terminal_reactive_development_v1',
            comparison_condition='heading_first_reactive',
            new_independent_development_layout=False,
            reference_root_name=reference_name,
            heading_first_terminal_control=True, terminal_heading_tolerance_rad=.1,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/heading_first_terminal_reactive_development.py')})
    RAW_WRITE(name, value)


def finish_write(name, value):
    annotated = bind(annotate_write, LAYOUT_INDEX=LAYOUT_INDEX, RAW_WRITE=RAW_WRITE)
    bind(original.finish_write, LAYOUT_INDEX=LAYOUT_INDEX, ARM='reactive',
        RAW_WRITE=annotated)(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    i = parser.parse_args().layout_index
    if sorted(os.sched_getaffinity(0)) != original.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(original.INVENTORY.read_bytes()).hexdigest() != original.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    stable = original.stable
    output = stable.source.BASE/f'go2_heading_first_terminal_reactive_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve completed or partial stronger-control trial')
    writer = bind(original.make_writer, finish_write=finish_write)(output, i, 'reactive')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            if kwargs.get('condition') != 'reactive':
                raise ValueError('model-free reactive assignment required')
            return HeadingFirstReactiveRuntime(*args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=original.layouts.specification, public_mission=original.layouts.public_mission,
            MODEL_ASSIGNMENT='reactive', PacedNativeSession=original.TransferCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__':
    main()

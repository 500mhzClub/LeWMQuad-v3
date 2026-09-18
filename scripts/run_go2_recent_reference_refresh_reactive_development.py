"""Test accepted-anchor reference refresh with the fixed stronger reactive control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_heading_first_terminal_reactive_development as original


def initialize_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    from lewm.recent_anchored_reference_refresh_development import RecentAnchoredReferenceRefreshMotion
    original.original.stable.floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = RecentAnchoredReferenceRefreshMotion(activation_frame=0)


def annotate_write(name, value):
    if name == 'launch.json':
        reference_name = f'go2_heading_first_terminal_reactive_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001'
        reference = json.loads((original.original.stable.source.BASE/reference_name/'launch.json').read_text())
        changed = [k for k in reference.keys() | value.keys()
            if k != 'owner' and reference.get(k) != value.get(k)]
        if changed:
            raise ValueError(f'original stronger reactive configuration changed: {changed}')
        value = value | dict(experiment='recent_anchored_reference_refresh_reactive_v1',
            comparison_condition='recent_reference_refresh_reactive',
            tracker_control_root_name=reference_name,
            recent_reference_refresh_from_accepted_anchor=True,
            maximum_recent_reference_age_ns=400_000_000,
            bridge_measurements_promoted=False,
            bridge_budget_and_measurement_acceptance_rules_unchanged=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/recent_anchored_reference_refresh_development.py')})
    RAW_WRITE(name, value)


def finish_write(name, value):
    annotated = bind(annotate_write, LAYOUT_INDEX=LAYOUT_INDEX, RAW_WRITE=RAW_WRITE)
    bind(original.finish_write, LAYOUT_INDEX=LAYOUT_INDEX, RAW_WRITE=annotated)(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    i = parser.parse_args().layout_index
    transfer = original.original
    if sorted(os.sched_getaffinity(0)) != transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest() != transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    stable = transfer.stable
    output = stable.source.BASE/f'go2_recent_reference_refresh_reactive_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve completed or partial reference-refresh trial')
    writer = bind(transfer.make_writer, finish_write=finish_write)(output, i, 'reactive')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            if kwargs.get('condition') != 'reactive':
                raise ValueError('fixed model-free stronger reactive assignment required')
            return original.HeadingFirstReactiveRuntime(*args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=transfer.layouts.specification, public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='reactive', PacedNativeSession=transfer.TransferCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__':
    main()

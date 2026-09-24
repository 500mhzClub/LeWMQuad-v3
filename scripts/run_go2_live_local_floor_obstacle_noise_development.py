"""Four prospective noisy navigation trials with locally estimated obstacle floors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.current_pair_routing_memory_development import initialize_mapping
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_live_local_feature_depth_noise_development as original

transfer = original.transfer


def initialize_obstacles():
    from lewm import independent_depth_process_development as process
    from lewm.local_floor_independent_obstacles_development import LocalFloorIndependentObstacles
    from lewm.jit_floor_candidates_development import warmup
    transfer.stable.floor.configure()
    process.initialize_obstacles()
    warmup()
    process._observer = LocalFloorIndependentObstacles()


def annotate_write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='live_local_floor_obstacle_noise_development_v1',
            comparison_condition='local_obstacle_floor_depth_noise_2mm',
            reference_root_name=f'go2_live_local_feature_depth_noise_2mm_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            independent_obstacle_observer='LocalFloorIndependentObstacles',
            independent_floor_candidate_depth_source='local_inverse_depth_5x5',
            independent_floor_candidates_are_raw_pixel_depth=False,
            mapping_and_independent_obstacle_algorithms_changed=True,
            mapping_algorithm_changed=False, independent_obstacle_floor_algorithm_changed=True,
            obstacle_points_use_original_depth=True,
            independent_floor_acceptance_thresholds_changed=False,
            new_independent_development_layout=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/local_floor_independent_obstacles_development.py')})
    RAW_WRITE(name, value)


def finish_write(name, value):
    annotated = bind(annotate_write, LAYOUT_INDEX=LAYOUT_INDEX, RAW_WRITE=RAW_WRITE)
    bind(original.finish_write, LAYOUT_INDEX=LAYOUT_INDEX, SIGMA_MM=2,
        RAW_WRITE=annotated)(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    args = parser.parse_args(); i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest() != transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    stable = transfer.stable
    output = stable.source.BASE/f'go2_live_local_floor_obstacle_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve completed or partial local-floor trial')
    writer = bind(transfer.make_writer, finish_write=finish_write)(output, i, 'learned')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=original.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'jepa':
                raise ValueError('fixed learned model assignment required')
            return original.RoutingMemoryRuntime(*runtime_args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=transfer.layouts.specification, public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_jepa',
            PacedNativeSession=partial(original.NoisyTransferSession,
                noise_layout_index=i, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=original.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready, initialize_mapping=initialize_mapping)()


if __name__ == '__main__':
    main()

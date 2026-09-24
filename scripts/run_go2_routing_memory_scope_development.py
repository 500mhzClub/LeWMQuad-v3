"""Paired routing-memory ablation with identical captured-map overhead."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.current_pair_routing_memory_development import (
    RoutingMemoryScopeMixin, initialize_mapping)
from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_post_repeatability_transfer_development as original

SCOPES = ('persistent', 'latest_mapped_pair')


class RoutingMemoryRuntime(RoutingMemoryScopeMixin, original.original.HeadingRecoveryRuntime):
    pass


def annotate_write(name, value):
    if name == 'launch.json':
        reference_name = f'go2_post_repeatability_transfer_learned_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001'
        reference = json.loads((original.stable.source.BASE/reference_name/'launch.json').read_text())
        changed = [k for k in reference.keys() | value.keys()
            if k != 'owner' and reference.get(k) != value.get(k)]
        if changed:
            raise ValueError(f'original learned configuration changed: {changed}')
        value = value | dict(experiment='routing_memory_scope_development_v1',
            comparison_condition=SCOPE, routing_memory_scope=SCOPE,
            new_independent_development_layout=False,
            reference_root_name=reference_name,
            current_pair_capture_enabled_in_both_arms=True,
            accumulated_action_clearance_preserved=True,
            other_tracking_model_mission_and_frontier_state_retained=True,
            memoryless_controller=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/current_pair_routing_memory_development.py')})
    RAW_WRITE(name, value)


def finish_write(name, value):
    annotated = bind(annotate_write, LAYOUT_INDEX=LAYOUT_INDEX, SCOPE=SCOPE,
        RAW_WRITE=RAW_WRITE)
    bind(original.finish_write, LAYOUT_INDEX=LAYOUT_INDEX, ARM='learned',
        RAW_WRITE=annotated)(name, value)


def make_writer(output, layout_index, scope):
    if scope not in SCOPES:
        raise ValueError('fixed routing-memory condition required')
    return bind(original.make_writer, finish_write=bind(finish_write, SCOPE=scope))(
        output, layout_index, 'learned')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--scope', choices=SCOPES, required=True)
    args = parser.parse_args()
    i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != original.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(original.INVENTORY.read_bytes()).hexdigest() != original.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    stable = original.stable
    output = stable.source.BASE/f'go2_routing_memory_{args.scope}_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve completed or partial routing-memory trial')
    writer = make_writer(output, i, args.scope)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'jepa':
                raise ValueError('fixed learned model assignment required')
            controller = RoutingMemoryRuntime(*runtime_args,
                registration_executor=executor, navigation_ticks=4800,
                arrival_radius_m=.02, **kwargs)
            controller.routing_memory_scope = args.scope
            return controller

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=original.layouts.specification, public_mission=original.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_jepa',
            PacedNativeSession=original.TransferCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=initialize_mapping)()


if __name__ == '__main__':
    main()

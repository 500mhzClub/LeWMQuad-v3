"""Frozen training-method comparison with the current noisy perception pipeline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.matched_motion_residual_runtime_development import FITS
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_current_plane_coverage_remaining_noise_development as learned


def initialize_gyro_coherent_pose():
    from lewm import process_mapped_runtime_development as process
    from lewm.gyro_coherent_floor_tracking_development import GyroCoherentFloorMotion
    learned.initialize_pose()
    process._motion = GyroCoherentFloorMotion()


def annotate_gyro_coherent(name, value):
    if name == 'launch.json':
        condition = value['training_condition']
        value = value | dict(experiment='gyro_coherent_floor_live_development_v1',
            comparison='perception_revision_after_completed_training_comparison',
            reference_root_name=f'go2_current_plane_matched_training_{condition}_noise_2mm_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            tracker='GyroCoherentFloorMotion', gyro_coherent_paired_floor_constraint=True,
            local_depth_estimator_and_pose_acceptance_unchanged=False,
            sensor_cache_mapping_and_noise_matched_to_learned=False,
            image_consensus_and_motion_thresholds_unchanged=True,
            mapping_and_independent_obstacle_algorithms_changed=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in ('lewm/gyro_coherent_floor_tracking_development.py',
                    'lewm/gyro_coherent_floor_constraint_development.py',
                    'lewm/gyro_consensus_pair_pose_development.py',
                    'lewm/gyro_conditioned_pair_pose_development.py')})
    RAW_WRITE(name, value)


def annotate_write(name, value):
    if name == 'launch.json':
        root, identity = FITS[CONDITION]
        value = value | dict(
            experiment='current_plane_matched_training_noise_development_v1',
            comparison='frozen_training_methods_with_matched_visual_motion_corrections',
            comparison_condition=CONDITION, training_condition=CONDITION,
            model_assignment=f'seed_2026091001_full_{CONDITION}',
            closed_loop_motion_residual_fit_sha256=identity,
            motion_residual_correction_root=root,
            reference_root_name=f'go2_current_plane_coverage_noise_2mm_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            mapping_algorithm_changed=False, mapping_floor_classifier_changed=False,
            mapping_and_independent_obstacle_algorithms_changed=False,
            independent_obstacle_floor_algorithm_changed=False,
            comparison_also_changes_retained_depth_cache=False,
            sensor_cache_mapping_and_noise_matched_to_learned=True,
            shared_planner_and_recovery_across_training_conditions=True,
            new_independent_development_layout=False,
            extra_sources=value['extra_sources'] | {
                __file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    RAW_WRITE(name, value)


def make_writer(output, layout_index, condition, *, gyro_coherent_floor=False):
    transfer = learned.previous.original.transfer
    stable = transfer.stable
    raw_write = bind(stable.source.write, OUTPUT=output)
    if gyro_coherent_floor:
        raw_write = bind(annotate_gyro_coherent, LAYOUT_INDEX=layout_index, RAW_WRITE=raw_write)
    final = bind(annotate_write, CONDITION=condition, LAYOUT_INDEX=layout_index,
        RAW_WRITE=raw_write)
    sensor = bind(learned.finish_write, LAYOUT_INDEX=layout_index, RAW_WRITE=final)
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=False,
        ARRIVAL_ENTRY_PRIORITY=True, source=SimpleNamespace(write=sensor))
    return bind(stable.configuration.write, _write=stable_writer,
        MODEL_ASSIGNMENT=f'seed_2026091001_full_{condition}',
        USE_CLEARANCE_TURN_RECOVERY=True, USE_STEPWISE_TURN_RECOVERY=True,
        USE_TERMINAL_POSITION_PRIORITY=True, USE_PROGRESS_REJOINING=True,
        USE_PREDICTIVE_ARRIVAL_HOLD=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--condition', choices=tuple(FITS), required=True)
    parser.add_argument('--gyro-coherent-floor', action='store_true',
        help='separate prospective perception revision; preserves completed comparison roots')
    args = parser.parse_args()
    i = args.layout_index
    gyro = learned.previous
    noisy = gyro.original
    transfer = noisy.transfer
    stable = transfer.stable
    if sorted(os.sched_getaffinity(0)) != transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest() != transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    prefix = 'gyro_coherent_floor' if args.gyro_coherent_floor else 'current_plane_matched_training'
    output = stable.source.BASE / f'go2_{prefix}_{args.condition}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve prior training-control trial')
    writer = make_writer(output, i, args.condition, gyro_coherent_floor=args.gyro_coherent_floor)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != args.condition:
                raise ValueError('model condition must match assigned frozen correction')
            return noisy.RoutingMemoryRuntime(*runtime_args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=transfer.layouts.specification, public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT=f'seed_2026091001_full_{args.condition}',
            PacedNativeSession=partial(noisy.NoisyTransferSession, noise_layout_index=i, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_gyro_coherent_pose if args.gyro_coherent_floor else learned.initialize_pose,
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=gyro.initialize_obstacles, OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=learned.initialize_mapping)()


if __name__ == '__main__':
    main()

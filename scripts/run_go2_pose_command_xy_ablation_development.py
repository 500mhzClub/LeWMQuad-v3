"""Paired live XY-source ablation on four already-exposed development layouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.pose_command_xy_control_development import PoseCommandXYSourceMixin, FIT_SHA256
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_post_training_transfer_noise_development as cohort

SOURCES = ('learned', 'pose_command')
BASE_CONDITION = 'supervised_rollout'


class XYSourceRuntime(PoseCommandXYSourceMixin, cohort.noisy.RoutingMemoryRuntime):
    pass


def annotate_xy(name, value):
    if name == 'launch.json':
        value = value | dict(
            experiment='pose_command_xy_source_ablation_development_v1',
            comparison='learned_vs_pose_command_xy_with_learned_yaw_and_contact',
            comparison_condition=XY_SOURCE, forecast_xy_source=XY_SOURCE,
            reference_training_condition=BASE_CONDITION,
            planned_conditions=list(SOURCES), planned_layout_count=4,
            new_independent_development_layout=False,
            layouts_previously_exposed_by_fixed_four_controller_cohort=True,
            both_xy_alternatives_computed_in_both_arms=True,
            pose_command_fit_sha256=FIT_SHA256,
            learned_yaw_and_contact_retained=True,
            downstream_planner_and_recovery_unchanged=True,
            reactive_recovery_rules_differ_from_predictive=False,
            fully_model_free_controller=False,
            extra_sources=value['extra_sources'] | {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/pose_command_xy_control_development.py',
                    'scripts/fit_pose_command_motion_control_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit = bind(annotate_xy, XY_SOURCE=XY_SOURCE, RAW_WRITE=RAW_WRITE)
    bind(cohort.annotate, CONDITION=BASE_CONDITION, RAW_WRITE=emit)(name, value)


def make_writer(output, index, xy_source):
    return bind(cohort.make_writer,
        annotate=bind(annotate, XY_SOURCE=xy_source))(output, index, BASE_CONDITION)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--xy-source', choices=SOURCES, required=True)
    args = parser.parse_args(); i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(cohort.INVENTORY.read_bytes()).hexdigest() != cohort.INVENTORY_SHA256:
        raise ValueError('fixed layout inventory required')
    output = cohort.stable.source.BASE / (
        f'go2_pose_command_xy_ablation_{BASE_CONDITION}_{args.xy_source}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001')
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve every assigned outcome')
    writer = make_writer(output, i, args.xy_source)
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != BASE_CONDITION:
                raise ValueError('fixed supervised yaw/contact and reference XY required')
            return XYSourceRuntime(*runtime_args, forecast_xy_source=args.xy_source,
                registration_executor=executor, navigation_ticks=4800,
                arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=cohort.layouts.specification,
            public_mission=cohort.layouts.public_mission,
            MODEL_ASSIGNMENT=f'seed_2026091001_full_{BASE_CONDITION}',
            PacedNativeSession=partial(cohort.PostTrainingCameraSession,
                noise_layout_index=i, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=cohort.training.initialize_gyro_coherent_pose,
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

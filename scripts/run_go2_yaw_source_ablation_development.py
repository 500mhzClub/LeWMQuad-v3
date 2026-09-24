"""Four fixed native yaw-source assignments with repaired frontier exploration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.arrival_conditioned_frontier_views_development import ArrivalConditionedFrontierMixin
from lewm.yaw_source_ablation_development import YawSourceMixin, SOURCES
from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_contact_score_ablation_development as pilot

cohort = pilot.cohort


class YawRuntime(YawSourceMixin, ArrivalConditionedFrontierMixin, pilot.ContactScoreRuntime):
    pass


def annotate_yaw(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='yaw_source_ablation_arrival_frontier_v1',
            comparison='learned_vs_command_yaw_with_pose_command_xy_and_disabled_contact',
            comparison_condition=YAW_SOURCE, forecast_yaw_source=YAW_SOURCE,
            planned_conditions=list(SOURCES), planned_layout_indices=[0,1],
            planned_layout_count=2, planned_native_assignments=4,
            both_yaw_alternatives_computed_in_both_arms=True,
            learned_yaw_retained=YAW_SOURCE=='learned', learned_yaw_and_contact_retained=False,
            neural_outcomes_used_for_scoring=YAW_SOURCE=='learned',
            pose_command_xy_remains_a_fitted_model=True,
            xy_yaw_and_physical_guards_unchanged=YAW_SOURCE=='learned',
            xy_contact_and_physical_guards_unchanged=True,
            frontier_exclusion_requires_observed_pose_arrival=True,
            frontier_exclusion_arrival_radius_m=.10,
            arrival_panorama_view_start_position_recorded=True,
            earlier_frontier_probe_outcomes_preserved=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/yaw_source_ablation_development.py',
                    'lewm/commanded_planar_motion_development.py',
                    'lewm/arrival_conditioned_frontier_views_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    final = bind(annotate_yaw, YAW_SOURCE=YAW_SOURCE, RAW_WRITE=RAW_WRITE)
    bind(pilot.annotate, MODE='disabled', RAW_WRITE=final)(name, value)


def make_writer(output, index, source):
    return bind(cohort.make_writer, annotate=bind(annotate, YAW_SOURCE=source))(
        output, index, 'supervised_rollout')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    parser.add_argument('--yaw-source', choices=SOURCES, required=True)
    args = parser.parse_args(); i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned physical CPU group required')
    output = cohort.stable.source.BASE / (
        f'go2_yaw_source_ablation_{args.yaw_source}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001')
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve every fixed yaw-source assignment')
    if hashlib.sha256(cohort.INVENTORY.read_bytes()).hexdigest() != cohort.INVENTORY_SHA256:
        raise ValueError('same development layout inventory required')
    writer = make_writer(output, i, args.yaw_source); cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'supervised_rollout': raise ValueError('same frozen model required')
            return YawRuntime(*runtime_args, forecast_yaw_source=args.yaw_source,
                contact_score_mode='disabled', forecast_xy_source='pose_command',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=cohort.layouts.specification, public_mission=cohort.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(cohort.PostTrainingCameraSession, noise_layout_index=i, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=cohort.training.initialize_gyro_coherent_pose,
            MEASURED_RUNTIME_CLASS=runtime, OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready, initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

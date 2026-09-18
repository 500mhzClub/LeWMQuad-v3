"""Disabled-contact follow-up after repairing arrival-panorama start metadata."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.arrival_conditioned_frontier_views_development import ArrivalConditionedFrontierMixin
from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_contact_score_ablation_development as pilot

cohort = pilot.cohort


class FrontierRuntime(ArrivalConditionedFrontierMixin, pilot.ContactScoreRuntime):
    pass


def annotate_frontier(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='arrival_conditioned_frontier_view_start_repair_v1',
            comparison='disabled_contact_frontier_probe_after_view_start_metadata_repair',
            planned_conditions=['disabled'],
            planned_layout_indices=[0], planned_layout_count=1, planned_native_assignments=1,
            arrival_panorama_view_start_position_recorded=True,
            predecessor_native_failure_preserved=True,
            frontier_exclusion_requires_observed_pose_arrival=True,
            frontier_exclusion_arrival_radius_m=.10,
            distant_completed_view_followed_by_observed_route_approach=True,
            previously_exposed_failure_diagnosis_layout=True,
            earlier_contact_pilot_outcomes_preserved=True,
            paired_original_policy_rerun=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/arrival_conditioned_frontier_views_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    final = bind(annotate_frontier, RAW_WRITE=RAW_WRITE)
    bind(pilot.annotate, MODE=MODE, RAW_WRITE=final)(name, value)


def make_writer(output, mode):
    return bind(cohort.make_writer, annotate=bind(annotate, MODE=mode))(
        output, 0, 'supervised_rollout')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contact-score', choices=('disabled',), required=True)
    args = parser.parse_args(); mode = args.contact_score
    # The inherited layout-0 writer binds group 0; execute these probes sequentially.
    group = 0
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[group]:
        raise ValueError('assigned physical CPU group required')
    # Keep startup failure 001 and native panorama-metadata failure 002 intact.
    attempt = '003'
    output = cohort.stable.source.BASE / (
        f'go2_arrival_conditioned_frontier_contact_{mode}_noise_2mm_native_layout00_4800_v1_attempt_{attempt}')
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve both prospective probe outcomes')
    if hashlib.sha256(cohort.INVENTORY.read_bytes()).hexdigest() != cohort.INVENTORY_SHA256:
        raise ValueError('same development layout inventory required')
    writer = make_writer(output, mode); cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'supervised_rollout': raise ValueError('same frozen model required')
            return FrontierRuntime(*runtime_args, contact_score_mode=mode,
                forecast_xy_source='pose_command', registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=0,
            specification=cohort.layouts.specification, public_mission=cohort.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(cohort.PostTrainingCameraSession, noise_layout_index=0, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=cohort.training.initialize_gyro_coherent_pose,
            MEASURED_RUNTIME_CLASS=runtime, OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready, initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

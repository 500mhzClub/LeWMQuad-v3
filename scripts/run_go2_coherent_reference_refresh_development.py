"""One live follow-up of the diagnosed learned-yaw layout-1 tracking failure."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_yaw_source_ablation_development as yaw

cohort = yaw.cohort
ROOT_NAME = 'go2_coherent_reference_refresh_learned_yaw_noise_2mm_native_layout01_4800_v1_attempt_001'
REFERENCE = 'go2_yaw_source_ablation_learned_noise_2mm_native_layout01_4800_v1_attempt_001'


def initialize_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    from lewm.coherent_reference_refresh_development import CoherentReferenceRefreshMotion
    cohort.stable.floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = CoherentReferenceRefreshMotion()


def annotate_refresh(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='coherent_reference_refresh_learned_yaw_v1',
            comparison='accepted_anchor_reference_refresh_followup',
            comparison_condition='coherent_reference_refresh', reference_root_name=REFERENCE,
            planned_conditions=['coherent_reference_refresh'], planned_layout_indices=[1],
            planned_layout_count=1, planned_native_assignments=1,
            recent_reference_refresh_from_accepted_anchor=True,
            maximum_recent_reference_age_ns=400_000_000,
            bridge_measurements_promoted=False,
            bridge_budget_and_measurement_acceptance_rules_unchanged=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/coherent_reference_refresh_development.py',
                    'lewm/recent_anchored_reference_refresh_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(yaw.annotate, YAW_SOURCE='learned',
        RAW_WRITE=bind(annotate_refresh, RAW_WRITE=RAW_WRITE))(name, value)


def make_writer(output):
    return bind(cohort.make_writer, annotate=annotate)(output, 1, 'supervised_rollout')


def main():
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('layout-1 physical CPU group required')
    output = cohort.stable.source.BASE / ROOT_NAME
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve this fixed follow-up and all predecessors')
    if hashlib.sha256(cohort.INVENTORY.read_bytes()).hexdigest() != cohort.INVENTORY_SHA256:
        raise ValueError('same development inventory required')
    writer = make_writer(output); cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            if kwargs.get('condition') != 'supervised_rollout':
                raise ValueError('same frozen supervised model required')
            return yaw.YawRuntime(*args, forecast_yaw_source='learned',
                contact_score_mode='disabled', forecast_xy_source='pose_command',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=1,
            specification=cohort.layouts.specification, public_mission=cohort.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(cohort.PostTrainingCameraSession, noise_layout_index=1, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

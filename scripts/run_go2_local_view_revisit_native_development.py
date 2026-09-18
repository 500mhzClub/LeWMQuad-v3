"""One exposed learned maze-3 navigation test of retained local image references."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_committed_camera_view_learned_followups_development as views

cohort = views.cohort
combined = views.combined
ROOT = 'go2_local_view_revisit_learned_noise_2mm_native_layout03_4800_v1_attempt_001'
REFERENCE = views.ROOT.format(index=3)


def initialize_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    from lewm.local_view_revisit_tracking_development import LocalViewRevisitMotion
    cohort.stable.floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = LocalViewRevisitMotion()


def annotate_revisit(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='local_view_revisit_native_v1',
            comparison='retained_local_view_references_with_unchanged_navigation',
            comparison_condition='local_view_revisit', reference_root_name=REFERENCE,
            planned_conditions=['local_view_revisit'], planned_layout_indices=[3],
            planned_layout_count=1, planned_native_assignments=1,
            tracker='LocalViewRevisitMotion', local_view_reference_bank=True,
            maximum_extra_local_view_references=8,
            original_pair_and_continuity_acceptance_unchanged=True,
            floor_reacquisition_enabled=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/local_view_revisit_tracking_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit = bind(annotate_revisit, RAW_WRITE=RAW_WRITE)
    bind(views.annotate, INDEX=3, RAW_WRITE=emit)(name, value)


def main():
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('original maze-3 CPU group required')
    base = cohort.stable.source.BASE
    if not (base/REFERENCE/'live_navigation_summary_v1.json').is_file():
        raise ValueError('evaluate the completed committed-view predecessor first')
    output = base/ROOT
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve this single follow-up outcome')
    writer = bind(cohort.make_writer, annotate=annotate)(output, 3, 'supervised_rollout')
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            return views.CommittedViewRuntime(*args, motion_prediction_source='learned',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=3,
            specification=combined.layouts.specification, public_mission=combined.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(combined.FreshCameraSession, noise_layout_index=3, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

"""One exposed maze-2 test of recovery from transient floor rejection."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.floor_reacquisition_development import FloorReacquisitionRuntimeMixin, initialize_registration
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_combined_perception_motion_development as combined

cohort = combined.cohort
ROOT = 'go2_combined_floor_reacquisition_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001'
REFERENCE = 'go2_combined_perception_motion_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001'


class ReacquiringRuntime(FloorReacquisitionRuntimeMixin, combined.MotionRuntime):
    pass


def annotate_recovery(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='combined_floor_reacquisition_v1',
            comparison='hold_and_reacquire_after_partial_floor_conflict',
            comparison_condition='floor_reacquisition', reference_root_name=REFERENCE,
            planned_conditions=['floor_reacquisition'], planned_layout_indices=[2],
            planned_layout_count=1, planned_native_assignments=1,
            fixed_first_source_by_layout=None, new_independent_development_layout=False,
            exposed_development_layout=True, layout_novelty_scope='exposed_combined_comparison_layout_2',
            registration='ReacquiringFloorRegistration', floor_reacquisition_enabled=True,
            temporary_floor_conflict_is_missing_observation=True,
            consecutive_accepted_poses_before_resuming_planning=4,
            rejected_floor_pose_used_by_map_or_mission=False,
            floor_rejection_resets_arrival_dwell=True,
            floor_rejection_cancels_pending_commands=True,
            global_budget_continues_during_missingness=True,
            committed_camera_view_turn=False, local_view_reference_bank=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/floor_reacquisition_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit = bind(annotate_recovery, RAW_WRITE=RAW_WRITE)
    bind(combined.annotate, SOURCE='pose_command', RAW_WRITE=emit)(name, value)


def main():
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[0]:
        raise ValueError('original maze-2 CPU group required')
    base = cohort.stable.source.BASE
    if not (base/REFERENCE/'floor_reacquisition_replay_v1/result.json').is_file():
        raise ValueError('complete recorded transient-conflict replay first')
    output = base/ROOT
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve this single follow-up outcome')
    writer = bind(cohort.make_writer, annotate=annotate)(output, 2, 'supervised_rollout')
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            return ReacquiringRuntime(*args, motion_prediction_source='pose_command',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=2,
            specification=combined.layouts.specification, public_mission=combined.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(combined.FreshCameraSession, noise_layout_index=2, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=combined.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

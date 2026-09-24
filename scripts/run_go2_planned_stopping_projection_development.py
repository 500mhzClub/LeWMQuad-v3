"""One learned maze-3 test of anticipating dispatch stopping projections."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.planned_stopping_projection_development import PlannedStoppingProjectionMixin
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_local_view_revisit_native_development as previous

cohort = previous.cohort
combined = previous.combined
ROOT = 'go2_planned_stopping_projection_learned_noise_2mm_native_layout03_4800_v1_attempt_001'
REFERENCE = previous.ROOT


class StoppingAwareRuntime(PlannedStoppingProjectionMixin, previous.views.CommittedViewRuntime):
    pass


def annotate_stopping(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='planned_stopping_projection_native_v1',
            comparison='anticipate_dispatch_stopping_projection_on_stored_obstacles',
            comparison_condition='planned_stopping_projection', reference_root_name=REFERENCE,
            planned_conditions=['planned_stopping_projection'], planned_layout_indices=[3],
            planned_layout_count=1, planned_native_assignments=1,
            planned_stopping_projection=True, dispatch_stopping_allowance_s=.5,
            anticipated_observation_age_maximum_s=.2,
            blocked_planned_translation_replaced_by_clear_turn_or_hold=True,
            actual_dispatch_guards_unchanged=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/planned_stopping_projection_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(previous.annotate, RAW_WRITE=bind(annotate_stopping, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('original maze-3 CPU group required')
    base = cohort.stable.source.BASE
    if not (base/REFERENCE/'planned_stopping_projection_replay_v1/result.json').is_file():
        raise ValueError('complete the saved stopping-projection comparison first')
    output = base/ROOT
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve this single follow-up outcome')
    writer = bind(cohort.make_writer, annotate=annotate)(output, 3, 'supervised_rollout')
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            return StoppingAwareRuntime(*args, motion_prediction_source='learned',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=3,
            specification=combined.layouts.specification, public_mission=combined.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(combined.FreshCameraSession, noise_layout_index=3, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=previous.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

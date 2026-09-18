"""Test prompt visual-recovery cancellation on the exposed replication failure."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.visual_recovery_dispatch_hold_development import VisualRecoveryDispatchHoldRuntime
from lewm.initial_panorama_development import InitialSurveyMixin
from lewm.recovery_limited_initial_survey_development import RecoveryLimitedInitialSurveyMixin
from scripts import run_go2_nogil_navigation_replication_development as previous

ROOT = 'go2_nogil_recovery_hold_jepa_noise_2mm_native_layout00_4800_v1_attempt_001'
SOURCES = (__file__, 'lewm/visual_recovery_dispatch_hold_development.py',
    'lewm/recovery_limited_initial_survey_development.py')


class ProfiledRecoveryHoldRuntime(VisualRecoveryDispatchHoldRuntime,
        previous.LivePlanningStageProfileRuntime):
    pass


class RecoveryLimitedSurveyRuntime(ProfiledRecoveryHoldRuntime,
        RecoveryLimitedInitialSurveyMixin, InitialSurveyMixin):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--limit-initial-survey', action='store_true')
    parser.add_argument('--arm', choices=('jepa', 'supervised_rollout'), default='jepa')
    args = parser.parse_args()
    if args.arm != 'jepa' and not args.limit_initial_survey:
        raise ValueError('supervised follow-up uses the combined survey treatment')
    study = previous.study
    root = (f'go2_nogil_recovery_limited_survey_{args.arm}_noise_2mm_native_layout00_4800_v1_attempt_001'
        if args.limit_initial_survey else ROOT)
    runtime_class = RecoveryLimitedSurveyRuntime if args.limit_initial_survey else ProfiledRecoveryHoldRuntime
    output = study.BASE/root
    if args.evaluate:
        selected = SimpleNamespace(**(vars(study) | dict(ROOT=root,
            ASSIGNMENTS=((0, args.arm),), PLAN=previous.PLAN)))
        return bind(previous.baseline.evaluation.evaluate, study=selected)(1)
    if output.exists():
        raise ValueError('preserve the exposed development attempt')
    frozen = json.loads(previous.PLAN.read_text())
    sources = previous.source_hashes()
    if sources != frozen['source_sha256']:
        raise ValueError('retain the completed replication sources')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[0]:
        raise ValueError('same layout-0 CPU allocation required')
    if sys.getswitchinterval() != .005 or shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('original scheduling and four GiB headroom required')
    reference_name = previous.ROOT.format(index=0, arm=args.arm)
    reference = json.loads((study.BASE/reference_name/'launch.json').read_text())
    sources |= {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}
    resources = previous.baseline.hardware()
    runtimes = []

    def write(name, value):
        if name == 'launch.json':
            value = reference | value | dict(experiment='nogil_visual_recovery_hold_v1',
                hardware=resources, reference_root_name=reference_name,
                comparison_condition='cancel_pre_trigger_commands_on_recovery_publication',
                actual_runtime_class=runtime_class.__name__, extra_sources=sources,
                fixed_sequential_assignments=[[0, args.arm]], planned_native_assignments=1,
                prospective_plan='docs/go2_nogil_recovery_hold_2026-09-16.md',
                new_independent_development_layout=False, exposed_development_layout=True,
                controller_changed=True, world_model_changed=False,
                completed_replication_plan_sha256=reference['prospective_plan_sha256'])
            if args.limit_initial_survey:
                value.update(experiment='nogil_recovery_limited_initial_survey_v1',
                    comparison_condition='defer_interrupted_initial_survey_with_prompt_recovery_hold',
                    prospective_plan='docs/go2_recovery_limited_initial_survey_2026-09-16.md',
                    prompt_hold_diagnostic_root_name=ROOT,
                    initial_survey_deferral_on_camera_time_recovery=True,
                    incomplete_survey_is_not_marked_complete=True)
            value.pop('prospective_plan_sha256', None)
        bind(study.source.write, OUTPUT=output)(name, value)
        if name == 'requests.json' and runtimes:
            for filename, record in (
                ('visual_dispatch_events.json', runtimes[0].visual_dispatch_events),
                ('planning_latency_stress.json', runtimes[0].clock_ns.rows),
                ('live_planning_profile.json', runtimes[0].plan_profile_rows)):
                bind(study.source.write, OUTPUT=output)(filename, record)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(previous.baseline.registration_ready).result()

        def runtime(model, **kwargs):
            kwargs['clock_ns'] = previous.PlanningLatencyClock(kwargs['clock_ns'], 20_000_000)
            result = runtime_class(model, prediction_source='neural',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02,
                **kwargs)
            runtimes.append(result)
            return result

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=0,
            specification=previous.layouts.specification,
            public_mission=previous.layouts.public_mission,
            MODEL_ASSIGNMENT=args.arm, MODEL_LOADER=bind(study.load_model, PLAN=previous.PLAN),
            PacedNativeSession=partial(previous.FreshCameraSession,
                noise_layout_index=0, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(previous.baseline.initialize_pose, str(output)),
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()

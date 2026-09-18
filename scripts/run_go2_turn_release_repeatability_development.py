"""Four balanced repeated missions with matched heading-release computation."""
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
from lewm.full_reserve_heading_release_development import FullReserveHeadingReleaseRuntime
from lewm.matched_heading_release_development import MatchedHeadingReleaseMixin
from scripts import run_go2_nogil_navigation_replication_development as previous
from scripts import run_go2_recovery_survey_transfer_development as transfer

study = previous.study
PLAN = Path('docs/go2_turn_release_repeatability_plan_2026-09-16.json')
CASES = tuple(dict(name=name, reference='go2_recovery_survey_transfer_limited_survey_jepa_noise_2mm_native_layout01_4800_v1_attempt_001', repaired=True, suppress=suppress)
    for name, suppress in (('enabled_rep1', False), ('disabled_rep1', True),
                           ('disabled_rep2', True), ('enabled_rep2', False)))
ROOT = 'go2_turn_release_repeatability_{name}_jepa_noise_2mm_native_layout01_4800_v1_attempt_001'


class MatchedReleaseRuntime(transfer.RecoveryLimitedSurveyRuntime,
        MatchedHeadingReleaseMixin, FullReserveHeadingReleaseRuntime):
    def __init__(self, *args, suppress_early_heading_release, **kwargs):
        self.suppress_early_heading_release = suppress_early_heading_release
        super().__init__(*args, **kwargs)


def source_hashes():
    return transfer.source_hashes() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__, 'lewm/matched_heading_release_development.py')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 5))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.assignment is not None or args.evaluate:
            raise ValueError('prepare separately')
        for source, frozen in ((previous, previous.PLAN), (transfer, transfer.PLAN)):
            if source.source_hashes() != json.loads(frozen.read_text())['source_sha256']:
                raise ValueError('completed reference runtime must remain unchanged')
        for cls in (MatchedReleaseRuntime,):
            mro = cls.__mro__
            assert mro.index(MatchedHeadingReleaseMixin)+1 == mro.index(FullReserveHeadingReleaseRuntime)
        references = {c['name']: json.loads((study.BASE/c['reference']/'launch.json').read_text()) for c in CASES}
        plan = dict(schema='turn_release_repeatability_plan.v1', cases=CASES,
            source_sha256=source_hashes(), references=references,
            fixed_before_first_repetition=True, exposed_development_layouts=True,
            final_evaluation=False, hardware_validated=False, candidate_count=6,
            navigation_ticks=4800, planning_extra_ns=20_000_000,
            only_intervention='enabled versus disabled early heading release through identical wrapper',
            condition_order='enabled,disabled,disabled,enabled', repetitions_per_condition=2,
            preserve_every_outcome=True, native_jobs_sequential=True,
            primary_outcome='physically verified goal-and-home round trip without contact')
        with PLAN.open('x') as f:
            json.dump(plan, f, indent=2); f.write('\n')
        print('PREPARED four balanced exposed-maze repetitions', flush=True)
        return
    if args.assignment is None:
        raise ValueError('fixed assignment required')
    case = CASES[args.assignment-1]
    output = study.BASE/ROOT.format(name=case['name'])
    model_plan = transfer.PLAN if case['repaired'] else previous.PLAN
    if args.evaluate:
        selected = SimpleNamespace(**(vars(study) | dict(ROOT=output.name,
            ASSIGNMENTS=((1, 'jepa'),), PLAN=model_plan)))
        return bind(previous.baseline.evaluation.evaluate, study=selected)(1)
    plan = json.loads(PLAN.read_text())
    if plan['source_sha256'] != source_hashes() or plan['cases'] != list(CASES):
        raise ValueError('preserve frozen ablation sources and cases')
    if output.exists():
        raise ValueError('preserve the assigned attempt')
    if args.assignment > 1 and not (study.BASE/ROOT.format(name=CASES[args.assignment-2]['name'])/'short_pulse_navigation_evaluation_v1.json').is_file():
        raise ValueError('evaluate first assignment before the second')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('same per-layout CPU group required')
    if sys.getswitchinterval() != .005 or shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('original scheduling and four GiB headroom required')
    reference = plan['references'][case['name']]
    resources = previous.baseline.hardware()
    layouts = transfer.layouts if case['repaired'] else previous.layouts
    camera = transfer.FreshCameraSession if case['repaired'] else previous.FreshCameraSession
    runtime_class = MatchedReleaseRuntime
    runtimes = []

    def write(name, value):
        if name == 'launch.json':
            value = reference | value | dict(experiment='turn_release_repeatability_v1',
                hardware=resources, reference_root_name=case['reference'],
                comparison_condition='disabled' if case['suppress'] else 'enabled',
                actual_runtime_class=runtime_class.__name__, extra_sources=plan['source_sha256'],
                fixed_sequential_assignments=[c['name'] for c in CASES], planned_native_assignments=4,
                prospective_plan=str(PLAN), prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                new_independent_development_layout=False, exposed_development_layout=True,
                controller_changed=True, world_model_changed=False, candidate_count=6,
                early_preferred_heading_release_enabled=not case['suppress'],
                heading_release_computation_wrapper_matched=True,
                original_clearance_and_stopping_checks_retained=True,
                translation_progress_release_retained=True,
                final_evaluation=False, hardware_validated=False)
        bind(study.source.write, OUTPUT=output)(name, value)
        if name == 'requests.json' and runtimes:
            for filename, record in (
                ('visual_dispatch_events.json', getattr(runtimes[0], 'visual_dispatch_events', [])),
                ('planning_latency_stress.json', runtimes[0].clock_ns.rows),
                ('live_planning_profile.json', runtimes[0].plan_profile_rows)):
                bind(study.source.write, OUTPUT=output)(filename, record)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(previous.baseline.registration_ready).result()

        def runtime(model, **kwargs):
            kwargs['clock_ns'] = previous.PlanningLatencyClock(kwargs['clock_ns'], 20_000_000)
            result = runtime_class(model, prediction_source='neural', registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02,
                suppress_early_heading_release=case['suppress'], **kwargs)
            runtimes.append(result)
            return result

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=1,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT='jepa', MODEL_LOADER=bind(study.load_model, PLAN=model_plan),
            PacedNativeSession=partial(camera, noise_layout_index=1, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(previous.baseline.initialize_pose, str(output)),
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()


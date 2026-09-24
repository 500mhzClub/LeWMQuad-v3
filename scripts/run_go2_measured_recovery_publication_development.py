"""One exposed-maze mission with the recovery-publication lock fix."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.measured_recovery_publication_development import MeasuredRecoveryPublicationMixin
from scripts import run_go2_recovery_repaired_navigation_replication_development as previous

study = previous.study
ROOT = 'go2_measured_recovery_publication_instantaneous_noise_2mm_native_layout00_4800_v1_attempt_001'
REFERENCE = previous.ROOT.format(arm='instantaneous', index=0)
PLAN = Path('docs/go2_measured_recovery_publication_plan_2026-09-16.json')


class FixedPublicationRuntime(MeasuredRecoveryPublicationMixin, previous.InstantaneousRuntime):
    pass


def source_hashes():
    return previous.source_hashes() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__, 'lewm/measured_recovery_publication_development.py')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    output = study.BASE/ROOT
    if args.prepare:
        if args.evaluate:
            raise ValueError('prepare separately')
        frozen = json.loads(previous.PLAN.read_text())
        if previous.source_hashes() != frozen['source_sha256']:
            raise ValueError('preserve the interrupted cohort sources')
        plan = dict(schema='measured_recovery_publication_plan.v1',
            reference_root=REFERENCE, assignments=[[0, 'instantaneous']],
            models=frozen['models'], training_seed=frozen['training_seed'],
            command_history_fit_sha256=frozen['command_history_fit_sha256'],
            source_sha256=source_hashes(), fixed_before_execution=True,
            intervention='wait for measured publication time outside the controller request lock',
            planned_native_assignments=1, exposed_development_layout=True,
            prior_interrupted_outcome_preserved=True, candidate_count=6,
            navigation_ticks=4800, planning_extra_ns=20_000_000,
            native_jobs_sequential=True, model_changed=False,
            final_evaluation=False, hardware_validated=False,
            primary_outcome='complete bounded execution without a publication deadlock',
            navigation_outcome_reported_separately=True)
        with PLAN.open('x') as f:
            json.dump(plan, f, indent=2); f.write('\n')
        print('PREPARED one exposed-maze publication-fix mission', flush=True)
        return
    if args.evaluate:
        selected = SimpleNamespace(**(vars(study) | dict(ROOT=ROOT,
            ASSIGNMENTS=((0, 'instantaneous'),), PLAN=PLAN)))
        previous.baseline.verify_selector(output, 'instantaneous')
        return bind(previous.baseline.evaluation.evaluate, study=selected,
            save_or_read=previous.baseline.save_receipt)(1)
    plan = json.loads(PLAN.read_text())
    if source_hashes() != plan['source_sha256'] or output.exists():
        raise ValueError('retain fixed sources and preserve the assigned attempt')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[0]:
        raise ValueError('same layout-0 CPU group required')
    if shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    reference = json.loads((study.BASE/REFERENCE/'launch.json').read_text())
    resources = previous.baseline.hardware()
    runtimes = []

    def write(name, value):
        if name == 'launch.json':
            value = reference | value | dict(experiment='measured_recovery_publication_v1',
                hardware=resources, reference_root_name=REFERENCE,
                comparison_condition='publication_clock_wait_outside_request_lock',
                actual_runtime_class=FixedPublicationRuntime.__name__,
                extra_sources=plan['source_sha256'], prospective_plan=str(PLAN),
                prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                fixed_sequential_assignments=plan['assignments'], planned_native_assignments=1,
                planned_layout_count=1, new_independent_development_layout=False,
                layout_novelty_scope='exposed interrupted-cohort layout 0',
                exposed_development_layout=True, world_model_changed=False,
                recovery_publication_clock_wait_outside_request_lock=True,
                final_evaluation=False, hardware_validated=False)
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
            result = FixedPublicationRuntime(model, prediction_source='neural',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)
            runtimes.append(result)
            return result

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=0,
            specification=previous.layouts.specification, public_mission=previous.layouts.public_mission,
            MODEL_ASSIGNMENT='instantaneous', MODEL_LOADER=bind(study.load_model, PLAN=PLAN),
            PacedNativeSession=partial(previous.FreshCameraSession, noise_layout_index=0, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(previous.baseline.initialize_pose, str(output)),
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()

"""Four fixed fresh-maze missions: original versus recovery-limited JEPA control."""
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
from lewm import recovery_limited_survey_transfer_layouts_development as layouts
from scripts import run_go2_nogil_navigation_replication_development as previous
from scripts.run_go2_nogil_recovery_hold_development import RecoveryLimitedSurveyRuntime

study = previous.study
PLAN = Path('docs/go2_recovery_survey_transfer_plan_2026-09-16.json')
ROOT = 'go2_recovery_survey_transfer_{condition}_jepa_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
ASSIGNMENTS = ((0, 'original'), (0, 'limited_survey'), (1, 'limited_survey'), (1, 'original'))
EXTRA_SOURCES = (__file__, 'lewm/recovery_limited_survey_transfer_layouts_development.py',
    'scripts/run_go2_nogil_recovery_hold_development.py',
    'lewm/visual_recovery_dispatch_hold_development.py',
    'lewm/recovery_limited_initial_survey_development.py')


class FreshPhysicalInit(study.cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(previous.NogilDrawingMixin, study.cohort.LiveDepthNoiseMixin,
        study.cohort.CompactDepthRetentionMixin,
        study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


def source_hashes():
    return previous.source_hashes() | {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in EXTRA_SOURCES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 5))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.assignment is not None or args.evaluate:
            raise ValueError('prepare before missions')
        frozen = json.loads(previous.PLAN.read_text())
        if previous.source_hashes() != frozen['source_sha256']:
            raise ValueError('original controller must match completed replication')
        plan = dict(schema='recovery_survey_transfer_plan.v1', assignments=ASSIGNMENTS,
            models=frozen['models'], training_seed=frozen['training_seed'],
            command_history_fit_sha256=frozen['command_history_fit_sha256'],
            inventory=layouts.build_inventory(), source_sha256=source_hashes(),
            fixed_before_first_navigation=True, final_evaluation=False,
            primary_outcome='physically verified goal-and-home round trip without contact',
            comparison='original versus prompt cancellation plus recovery-limited initial survey',
            paired_condition_order_reversed_on_second_layout=True,
            model_arm='jepa', model_changed=False, preserve_every_failure=True,
            navigation_ticks=4800, planning_extra_ns=20_000_000,
            candidate_count=6, native_jobs_sequential=True,
            limitations=['two new mazes from the existing family', 'one execution per condition/layout',
                'one neural training seed', 'measured simulation, not hardware qualification'])
        with PLAN.open('x') as stream:
            json.dump(plan, stream, indent=2); stream.write('\n')
        print('PREPARED', len(ASSIGNMENTS), 'fixed missions', flush=True)
        return
    if args.assignment is None:
        raise ValueError('fixed assignment required')
    index, condition = ASSIGNMENTS[args.assignment-1]
    output = study.BASE/ROOT.format(index=index, condition=condition)
    if args.evaluate:
        selected = SimpleNamespace(**(vars(study) | dict(ROOT=output.name,
            ASSIGNMENTS=((index, 'jepa'),), PLAN=PLAN)))
        return bind(previous.baseline.evaluation.evaluate, study=selected)(1)
    plan = json.loads(PLAN.read_text())
    if (plan['assignments'] != [list(a) for a in ASSIGNMENTS]
            or plan['source_sha256'] != source_hashes()
            or plan['inventory'] != layouts.build_inventory()):
        raise ValueError('fixed sources, assignments and layout inventory required')
    if args.assignment > 1:
        i, c = ASSIGNMENTS[args.assignment-2]
        if not (study.BASE/ROOT.format(index=i, condition=c)/'short_pulse_navigation_evaluation_v1.json').is_file():
            raise ValueError('evaluate preceding assignment before launching next')
    if output.exists():
        raise ValueError('preserve prospective attempt')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[index % 2]:
        raise ValueError('fixed per-layout CPU allocation required')
    if sys.getswitchinterval() != .005 or shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('original scheduling and four GiB headroom required')
    runtime_class = (previous.LivePlanningStageProfileRuntime if condition == 'original'
        else RecoveryLimitedSurveyRuntime)
    model = plan['models']['jepa']
    resources = previous.baseline.hardware()
    runtimes = []

    def write(name, value):
        if name == 'launch.json':
            value = value | dict(experiment='recovery_survey_transfer_v1', study_arm='jepa',
                comparison_condition=condition, model_assignment='jepa', training_condition='jepa',
                hardware=resources, fresh_layout_inventory=plan['inventory'],
                prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                fixed_sequential_assignments=plan['assignments'], planned_native_assignments=4,
                actual_runtime_class=runtime_class.__name__, extra_sources=plan['source_sha256'],
                neural_snapshot=dict(filename=model['filename'], sha256=model['sha256'],
                    model_state_sha256=model['model_sha256']), training_seed=plan['training_seed'],
                command_history_fit_sha256=plan['command_history_fit_sha256'],
                external_neural_motion_correction=False, contact_score_mode='disabled',
                new_independent_development_layout=True,
                layout_novelty_scope='distinct_from_explicit_92_layout_registry',
                sensor_noise_sigma_mm=2, gyro_noise_model='ideal', navigation_tick_budget=4800,
                observed_arrival_radius_m=.02, physical_arrival_requirement_m=.04,
                nominal_footprint_radius_m=.45, maximum_observation_age_ns=250_000_000,
                planning_extra_ns=20_000_000, renderer_drawing_releases_gil=True,
                inference_shared_history_encoding=False, parent_python_switch_interval_s=.005,
                persistent_routing_memory=True, every_camera_frame_tracked=True,
                initial_survey_deferral_enabled=condition=='limited_survey',
                prompt_visual_recovery_cancellation_enabled=condition=='limited_survey',
                world_model_changed=False, candidate_count=6,
                parallel_analysis_during_mission=False, final_evaluation=False, hardware_validated=False)
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
            result = runtime_class(model, prediction_source='neural',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02,
                **kwargs)
            runtimes.append(result)
            return result

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT='jepa', MODEL_LOADER=bind(study.load_model, PLAN=PLAN),
            PacedNativeSession=partial(FreshCameraSession, noise_layout_index=index, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(previous.baseline.initialize_pose, str(output)),
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()

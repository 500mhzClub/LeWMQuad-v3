"""Twenty fresh-maze missions with shared startup recovery and matched baselines."""
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

from lewm import recovery_repaired_navigation_replication_layouts_development as layouts
from lewm.eligible_floor_registration_development import bind
from scripts.run_go2_nogil_recovery_hold_development import RecoveryLimitedSurveyRuntime
from lewm.persistent_visual_baselines_development import ComputedForecastsUnusedMixin, ReactiveFeedbackMixin
from lewm.instantaneous_waypoint_score_development import InstantaneousWaypointScoreMixin
from lewm.planning_latency_stress_development import PlanningLatencyClock
from scripts import run_go2_persistent_visual_baselines_development as baseline
from scripts.nogil_drawing_session_development import NogilDrawingMixin

study = baseline.study
BASE = study.BASE
ROOT = 'go2_recovery_repaired_replication_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
PLAN = Path('docs/go2_recovery_repaired_navigation_replication_plan_2026-09-16.json')
INVENTORY = Path('docs/go2_recovery_repaired_navigation_replication_layout_inventory_2026-09-16.json')
ARMS = ('jepa', 'supervised_rollout', 'pose_command', 'instantaneous', 'reactive_feedback')
ASSIGNMENTS = tuple((i, a) for i in range(4) for a in ARMS[i:]+ARMS[:i])
SOURCES = tuple(dict.fromkeys((*baseline.SOURCES, __file__,
    'lewm/recovery_repaired_navigation_replication_layouts_development.py',
    'scripts/nogil_drawing_session_development.py',
    'lewm_genesis/lewm_genesis/nogil_readback_development.py',
    'lewm/live_planning_profile_development.py',
    'lewm/live_planning_stage_profile_development.py',
    'lewm/planning_latency_stress_development.py',
    'scripts/run_go2_short_pulse_navigation_development.py',
    'scripts/evaluate_go2_short_pulse_navigation_development.py',
    'scripts/run_go2_nogil_recovery_hold_development.py',
    'scripts/run_go2_nogil_navigation_replication_development.py',
    'lewm/nogil_navigation_replication_layouts_development.py',
    'lewm/recovery_limited_survey_transfer_layouts_development.py',
    'lewm/visual_recovery_dispatch_hold_development.py',
    'lewm/recovery_limited_initial_survey_development.py')))


class FreshPhysicalInit(study.cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(NogilDrawingMixin, study.cohort.LiveDepthNoiseMixin,
        study.cohort.CompactDepthRetentionMixin,
        study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


class InstantaneousRuntime(InstantaneousWaypointScoreMixin, RecoveryLimitedSurveyRuntime):
    pass


class ReactiveRuntime(ComputedForecastsUnusedMixin, ReactiveFeedbackMixin, RecoveryLimitedSurveyRuntime):
    pass


RUNTIMES = dict(jepa=RecoveryLimitedSurveyRuntime,
    supervised_rollout=RecoveryLimitedSurveyRuntime, pose_command=RecoveryLimitedSurveyRuntime,
    instantaneous=InstantaneousRuntime, reactive_feedback=ReactiveRuntime)


def source_hashes():
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}


def prepare():
    if PLAN.exists() or INVENTORY.exists():
        raise ValueError('preserve the prospective inventory and plan')
    frozen = json.loads(Path('docs/go2_nogil_navigation_replication_plan_2026-09-16.json').read_text())
    sources = source_hashes()
    if any(sources[p] != frozen['source_sha256'][p] for p in baseline.CONTROLLER_SOURCES):
        raise ValueError('retain the completed comparison controller')
    inventory = layouts.build_inventory()
    with INVENTORY.open('x') as stream:
        json.dump(inventory, stream, indent=2); stream.write('\n')
    plan = dict(schema='recovery_repaired_navigation_replication_plan.v1', assignments=ASSIGNMENTS,
        models=frozen['models'], training_seed=frozen['training_seed'],
        command_history_fit_sha256=frozen['command_history_fit_sha256'],
        inventory_sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest(), source_sha256=sources,
        fixed_before_first_navigation=True, model_selection_from_new_maze_outcomes=False,
        final_evaluation=False, broader_environment_type_tests_deferred=True,
        primary_outcome='physically verified goal-and-home round trip with no disallowed contact',
        secondary_outcomes=['goal arrival', 'physical contacts', 'tracking loss',
            'simulated completion time', 'fraction of plans on time', 'executed-window forecast errors'],
        comparisons=['JEPA versus same-data supervised prediction',
            'learned prediction versus fitted pose-command prediction',
            'supervised predictive ranking versus instantaneous ranking with predictive guards',
            'supervised predictive controller versus reactive feedback without forecast selection'],
        navigation_ticks=4800, planning_extra_ns=20_000_000,
        renderer_drawing_releases_gil=True, original_inference_batch=True,
        initial_survey_deferral_enabled=True, prompt_visual_recovery_cancellation_enabled=True,
        early_heading_release_unchanged=True, candidate_count=6,
        depth_noise_sigma_mm=2, gyro_noise_model='ideal',
        measured_simulation_not_real_time=True, native_jobs_run_sequentially=True,
        preserve_all_failures=True, controller_changes_during_batch=False,
        limitations=['four new mazes from the existing maze family', 'one execution per arm/layout',
            'one neural training seed', 'reactive arm is a controller-package comparison',
            'instantaneous ranking retains predictive guards', 'sensing and timing not deployment validated'])
    with PLAN.open('x') as stream:
        json.dump(plan, stream, indent=2); stream.write('\n')
    print('PREPARED', len(ASSIGNMENTS), 'fixed fresh-maze assignments', flush=True)


def evaluate(number):
    index, arm = ASSIGNMENTS[number-1]
    output = BASE/ROOT.format(index=index, arm=arm)
    selected = SimpleNamespace(**(vars(study) | dict(ROOT=output.name,
        ASSIGNMENTS=((index, arm),), PLAN=PLAN)))
    options = dict(study=selected)
    if arm in ('instantaneous', 'reactive_feedback'):
        baseline.verify_selector(output, arm)
        options['save_or_read'] = baseline.save_receipt
    return bind(baseline.evaluation.evaluate, **options)(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 21))
    args = parser.parse_args()
    if args.prepare:
        if args.assignment is not None or args.evaluate:
            raise ValueError('prepare separately')
        return prepare()
    if args.assignment is None:
        raise ValueError('fixed assignment required')
    if args.evaluate:
        return evaluate(args.assignment)
    index, arm = ASSIGNMENTS[args.assignment-1]
    output = BASE/ROOT.format(index=index, arm=arm)
    plan = json.loads(PLAN.read_text())
    if plan['assignments'] != [list(a) for a in ASSIGNMENTS] or source_hashes() != plan['source_sha256']:
        raise ValueError('fixed assignment order and source required')
    if hashlib.sha256(INVENTORY.read_bytes()).hexdigest() != plan['inventory_sha256']:
        raise ValueError('fixed layout inventory required')
    if json.loads(INVENTORY.read_text()) != layouts.build_inventory():
        raise ValueError('source layouts must match inventory')
    if args.assignment > 1:
        i, a = ASSIGNMENTS[args.assignment-2]
        if not (BASE/ROOT.format(index=i, arm=a)/'short_pulse_navigation_evaluation_v1.json').is_file():
            raise ValueError('evaluate preceding assignment first')
    if output.exists():
        raise ValueError('preserve the prospective attempt')
    if shutil.disk_usage(BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[index % 2]:
        raise ValueError('fixed per-layout CPU allocation required')
    if sys.getswitchinterval() != .005:
        raise ValueError('retain original parent switch interval')
    model_condition = study.model_condition(arm)
    model = plan['models'][model_condition]
    resources = baseline.hardware()
    runtimes = []

    def write(name, value):
        if name == 'launch.json':
            value = value | dict(experiment='recovery_repaired_navigation_replication_v1', study_arm=arm,
                comparison_condition=arm, model_assignment=arm, training_condition=model_condition,
                hardware=resources, fresh_layout_inventory=layouts.build_inventory(),
                frozen_layout_inventory_sha256=plan['inventory_sha256'],
                prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                fixed_sequential_assignments=plan['assignments'],
                new_independent_development_layout=True,
                layout_novelty_scope='distinct_from_explicit_94_layout_registry',
                planned_layout_count=4, planned_native_assignments=20,
                actual_runtime_class=RUNTIMES[arm].__name__, extra_sources=source_hashes(),
                neural_snapshot=dict(filename=model['filename'], sha256=model['sha256'],
                    model_state_sha256=model['model_sha256']), training_seed=plan['training_seed'],
                command_history_fit_sha256=plan['command_history_fit_sha256'],
                external_neural_motion_correction=False, contact_score_mode='disabled',
                sensor_noise_sigma_mm=2, gyro_noise_model='ideal', navigation_tick_budget=4800,
                observed_arrival_radius_m=.02, physical_arrival_requirement_m=.04,
                nominal_footprint_radius_m=.45, maximum_observation_age_ns=250_000_000,
                age_charged_to_stopping_projection=True, planning_extra_ns=20_000_000,
                renderer_drawing_releases_gil=True, inference_shared_history_encoding=False,
                parent_python_switch_interval_s=sys.getswitchinterval(),
                persistent_routing_memory=True, every_camera_frame_tracked=True,
                initial_survey_deferral_enabled=True, prompt_visual_recovery_cancellation_enabled=True,
                early_heading_release_unchanged=True, candidate_count=6,
                pose_tracker_class='CadencedViewRevisitMotion', old_view_revisit_period_frames=4,
                neural_reference_is_unused_for_control=arm in ('pose_command', 'reactive_feedback'),
                main_action_score_uses_predicted_outcomes=arm not in ('instantaneous', 'reactive_feedback'),
                predictive_guards_retained=arm!='reactive_feedback',
                model_forecasts_computed_for_workload_control=True, total_computation_identical=False,
                actual_dispatch_projection_retained=True, parallel_analysis_during_mission=False,
                final_evaluation=False, hardware_validated=False)
        bind(study.source.write, OUTPUT=output)(name, value)
        if name == 'requests.json' and runtimes:
            bind(study.source.write, OUTPUT=output)('visual_dispatch_events.json', runtimes[0].visual_dispatch_events)
            bind(study.source.write, OUTPUT=output)('planning_latency_stress.json', runtimes[0].clock_ns.rows)
            bind(study.source.write, OUTPUT=output)('live_planning_profile.json', runtimes[0].plan_profile_rows)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(baseline.registration_ready).result()

        def runtime(model, **kwargs):
            kwargs['clock_ns'] = PlanningLatencyClock(kwargs['clock_ns'], 20_000_000)
            result = RUNTIMES[arm](model, prediction_source='pose_command' if arm=='pose_command' else 'neural',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)
            runtimes.append(result)
            return result

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=arm, MODEL_LOADER=bind(study.load_model, PLAN=PLAN),
            PacedNativeSession=partial(FreshCameraSession, noise_layout_index=index, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(baseline.initialize_pose, str(output)), MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()

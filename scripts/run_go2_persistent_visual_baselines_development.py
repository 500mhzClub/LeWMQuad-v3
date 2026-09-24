"""Three fixed prediction controls with the current persistent visual controller."""
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

from lewm import persistent_visual_transfer_layouts_development as layouts
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts.run_go2_obstacle_grouping_navigation_development import study
from lewm.persistent_visual_baselines_development import RUNTIMES, select_reactive_terminal
from lewm.instantaneous_waypoint_score_development import instantaneous_scores
from lewm.current_reserve_terminal_feedback_development import select_reserved_terminal
from lewm.cadenced_view_revisit_tracking_development import CadencedViewRevisitMotion, REVISIT_PERIOD_FRAMES
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation

ROOT = 'go2_persistent_visual_baselines_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
PLAN = Path('docs/go2_persistent_visual_baselines_plan_2026-09-16.json')
INVENTORY = Path('docs/go2_persistent_visual_transfer_layout_inventory_2026-09-16.json')
ARMS = ('instantaneous', 'reserved_off', 'reactive_feedback')
ASSIGNMENTS = tuple((i, a) for i in range(4) for a in ARMS[i % 3:]+ARMS[:i % 3])
CONTROLLER_SOURCES = (
    'lewm/short_pulse_navigation_runtime_development.py',
    'lewm/pipeline_age_dispatch_development.py',
    'lewm/visual_support_recovery_development.py',
    'lewm/prefix_aware_terminal_approach_development.py',
    'lewm/framewise_visual_support_recovery_development.py',
    'lewm/persistent_local_visual_recovery_development.py',
    'lewm/cadenced_view_revisit_tracking_development.py')
SOURCES = (*CONTROLLER_SOURCES, __file__,
    'lewm/persistent_visual_transfer_layouts_development.py',
    'scripts/run_go2_paced_native_prefix_development.py',
    'lewm/persistent_visual_baselines_development.py',
    'lewm/instantaneous_waypoint_score_development.py',
    'lewm/current_reserve_terminal_feedback_development.py',
    'lewm/rollout_selection_off_development.py',
    'lewm/continuous_reactive_selection_development.py',
    'lewm/heading_first_terminal_reactive_development.py')


class FreshPhysicalInit(study.cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(study.cohort.LiveDepthNoiseMixin, study.cohort.CompactDepthRetentionMixin,
        study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


def source_hashes():
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}


def prepare():
    if PLAN.exists():
        raise ValueError('preserve the fixed baseline plan')
    frozen = json.loads(Path('docs/go2_persistent_visual_transfer_plan_2026-09-16.json').read_text())
    for path in CONTROLLER_SOURCES:
        if source_hashes()[path] != frozen['source_sha256'][path]:
            raise ValueError('retain the completed comparison controller')
    if json.loads(INVENTORY.read_text()) != layouts.build_inventory():
        raise ValueError('same four locked layouts required')
    plan = frozen | dict(schema='persistent_visual_baselines_plan.v1',
        assignments=ASSIGNMENTS, source_sha256=source_hashes(),
        fixed_before_first_navigation=False, fixed_before_first_baseline_navigation=True,
        comparison='instantaneous ranking with predictive guards; current-reserve feedback without forecasts; established reactive heading feedback without forecasts',
        computations_matched='same supervised model and alternative forecasts computed; selector costs differ',
        limitations=['same four exposed development layouts', 'one execution per condition/layout',
            'controller developed with supervised model', 'off controls replace predictive feasibility/recovery rules',
            'reactive feedback uses nominal clearance rather than the 3-cm reserve',
            'current-reserve feedback has no predictive escape from reserve deficits',
            'measured simulation, not real time', 'synthetic depth noise and ideal gyro'],
        broader_environment_type_tests_deferred=True)
    with PLAN.open('x') as stream:
        json.dump(plan, stream, indent=2); stream.write('\n')
    print('PREPARED', len(ASSIGNMENTS), 'fixed assignments', flush=True)


def initialize_pose(output):
    from lewm import process_mapped_runtime_development as process
    study.previous.reference.initialize_pose()
    process._motion = CadencedViewRevisitMotion()
    receipt = dict(pid=os.getpid(), motion_class=type(process._motion).__name__,
        pose_class=type(process._motion.model).__name__, old_view_revisit_period_frames=REVISIT_PERIOD_FRAMES)
    with (Path(output)/'pose_worker_identity.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)


def verify_selector(output, arm):
    plans = [p for p in json.loads((output/'planning.json').read_text()) if 'selection' in p]
    if not plans:
        raise ValueError('recorded decisions required')
    for p in plans:
        s = p['selection']; pulse = s['terminal_translation_pulse']['enabled']
        if p['action'] != s['action']:
            raise ValueError('recorded decision/action mismatch')
        if arm == 'instantaneous':
            expected = instantaneous_scores(s['waypoint_body_xy_m'],
                scan_error=s.get('scan_heading_error_rad'), pulse=pulse)
            witness = s['instantaneous_ranking']
            if witness['rows'] != expected or witness['predictions_used_for_main_utilities']:
                raise ValueError('main ranking must use the current observation')
            if any(c['utility_m'] != e['utility_m'] or c['position_contact_utility_m'] != e['position_utility_m']
                    for c, e in zip(s['candidates'], expected)):
                raise ValueError('candidate utilities differ')
            if 'memory_forecast_candidates' not in s or 'planned_stopping_projection' not in s:
                raise ValueError('predictive guards must remain')
        else:
            if arm == 'reserved_off':
                expected = select_reserved_terminal(s['waypoint_body_xy_m'],
                    scan_error=s['scan_heading_error_rad'], pulse=pulse,
                    clearance_m=s['current_stored_clearance_m'])
            else:
                expected = select_reactive_terminal(s['waypoint_body_xy_m'],
                    scan_error=s['scan_heading_error_rad'], pulse=pulse,
                    clearance_m=s['current_stored_clearance_m'], arrival_radius_m=.02)
            # Outer wrappers add pulse/routing/input receipts. Compare all
            # selector fields, and the pulse fields they preserve, explicitly.
            for key, value in expected.items():
                if isinstance(value, dict):
                    if any(s[key].get(k) != v for k, v in value.items()):
                        raise ValueError('recorded selector receipt differs: '+key)
                elif s.get(key) != value:
                    raise ValueError('recorded selector differs: '+key)
            if any(k in s for k in ('memory_forecast_candidates', 'predictive_arrival_hold',
                    'arrival_entry_terminal_priority', 'instantaneous_ranking')):
                raise ValueError('off control retained a predictive selection consumer')
    receipt = dict(arm=arm,selected_plans=len(plans),actual_selection_treatment_verified=True,
        predictive_guards_retained=arm=='instantaneous',
        model_computed_for_workload_control=True, total_computation_identical=False,
        actual_dispatch_projection_retained=True, comparison_is_controller_package=True)
    (output/'actual_baseline_selection_v1.json').write_text(json.dumps(receipt, indent=2)+'\n')


def save_receipt(root, name, value):
    if name == 'actual_controller_treatment_v1.json':
        value = value | dict(neural_reference_unused_by_control=value['arm']!='instantaneous',
            model_forecasts_computed_for_workload_control=True,
            actual_forecast_based_selection_disabled=value['arm']!='instantaneous')
    return evaluation.save_or_read(root, name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 13))
    args = parser.parse_args()
    if args.prepare:
        if args.assignment is not None or args.evaluate:
            raise ValueError('prepare separately from execution')
        return prepare()
    if args.assignment is None:
        raise ValueError('fixed assignment required')
    index, arm = ASSIGNMENTS[args.assignment-1]
    output = study.BASE/ROOT.format(index=index, arm=arm)
    plan = json.loads(PLAN.read_text())
    if plan['assignments'] != [list(a) for a in ASSIGNMENTS]:
        raise ValueError('fixed assignment order required')
    if args.evaluate:
        verify_selector(output, arm)
        selected = SimpleNamespace(**(vars(study) | {
            'ROOT': output.name, 'ASSIGNMENTS': ((index, arm),), 'PLAN': PLAN}))
        identity = json.loads((output/'pose_worker_identity.json').read_text())
        if identity['pose_class'] != 'CadencedViewRevisitPose' or identity['old_view_revisit_period_frames'] != 4:
            raise ValueError('actual prospective pose worker required')
        result = bind(evaluation.evaluate, study=selected, save_or_read=save_receipt)(1)
        requests = json.loads((output/'requests.json').read_text())
        passed = [r for r in requests if r['reason'] == 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED']
        if not passed or any(r.get('observation_age_limit_ns') != 250_000_000 for r in passed):
            raise ValueError('actual accepted dispatch must use the prospective age bound')
        for row in passed:
            age = row['now_ns']-row['observation_measured_ns']
            if not 0 <= age <= 250_000_000:
                raise ValueError('accepted observation exceeds the age bound')
            if any(row['requested_command'][:2]) and row['observation_age_allowance_s'] != age/1e9:
                raise ValueError('translation must charge actual age to stopping distance')
        treatment = dict(actual_dispatch_age_treatment_verified=True,
            accepted_requests=len(passed),
            accepted_requests_older_than_200ms=sum(r['observation_age_ns'] > 200_000_000 for r in passed),
            maximum_accepted_age_ns=max(r['observation_age_ns'] for r in passed),
            actual_translation_age_charged=True, stopping_bound_calibrated=False)
        (output/'actual_pipeline_age_treatment_v1.json').write_text(json.dumps(treatment, indent=2)+'\n')
        print(json.dumps(result | treatment, indent=2))
        return
    if args.assignment > 1:
        previous_index, previous_arm = ASSIGNMENTS[args.assignment-2]
        previous = study.BASE/ROOT.format(index=previous_index, arm=previous_arm)
        if not (previous/'short_pulse_navigation_evaluation_v1.json').is_file():
            raise ValueError('evaluate the preceding completed assignment first')
    if output.exists():
        raise ValueError('preserve this prospective attempt')
    if source_hashes() != plan['source_sha256']:
        raise ValueError('fixed controller and launcher required throughout this batch')
    if hashlib.sha256(INVENTORY.read_bytes()).hexdigest() != plan['inventory_sha256']:
        raise ValueError('fixed layout inventory required')
    if json.loads(INVENTORY.read_text()) != layouts.build_inventory():
        raise ValueError('source layouts must match the fixed inventory')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[index % 2]:
        raise ValueError('fixed per-layout CPU allocation required')
    if shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    before = sys.getswitchinterval()
    if before != .005:
        raise ValueError('reference parent switch interval required')
    resources = hardware()
    model = plan['models']['supervised_rollout']

    def write(name, value):
        if name == 'launch.json':
            value = value | dict(experiment='persistent_visual_baselines_v1',
                study_arm=arm, comparison_condition=arm,
                hardware=resources, fixed_sequential_assignments=plan['assignments'],
                fresh_layout_inventory=layouts.build_inventory(),
                frozen_layout_inventory_sha256=plan['inventory_sha256'],
                prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                new_independent_development_layout=False,
                layout_novelty_scope='same four exposed layouts as completed sixteen-model comparison',
                planned_layout_count=4, planned_native_assignments=12,
                neural_reference_is_unused_for_control=arm!='instantaneous',
                all_alternative_forecasts_computed_in_both_arms=True,
                recovery_updated_every_camera_frame=True,
                recovery_evidence_bound_to_planning_observation=True,
                local_visual_reference_independent_of_mission_leg=True,
                terminal_radius_rule='0.10m + sum(norm(queued_xy_command))*0.1s',
                measured_arrival_rules_changed=False, terminal_command_duration_changed=False,
                low_feature_threshold=48, strong_feature_threshold=96,
                maximum_reference_view_age_ns=None, maximum_reference_view_distance_m=.20,
                feature_count_is_calibrated_confidence=False,
                pose_tracker_class='CadencedViewRevisitMotion', old_view_revisit_period_frames=4,
                every_camera_frame_tracked=True, reference_acceptance_limits_changed=False,
                exposed_development_layout=True,
                parent_python_switch_interval_before_s=before,
                parent_python_switch_interval_s=sys.getswitchinterval(),
                spawned_worker_switch_interval_changed=False,
                independent_obstacle_observer='GyroConditionedAuxiliaryObstacles',
                neural_snapshot=dict(filename=model['filename'], sha256=model['sha256'],
                    model_state_sha256=model['model_sha256']), training_seed=plan['training_seed'],
                command_history_fit_sha256=plan['command_history_fit_sha256'],
                external_neural_motion_correction=False, contact_score_mode='disabled',
                sensor_noise_sigma_mm=2, gyro_noise_model='ideal',
                navigation_tick_budget=4800, observed_arrival_radius_m=.02,
                physical_arrival_requirement_m=.04, nominal_footprint_radius_m=.45,
                terminal_translation_pulses=True,
                terminal_translation_command_duration_ns=100_000_000,
                observation_age_limit_changed=False, maximum_observation_age_ns=250_000_000,
                reference_maximum_observation_age_ns=250_000_000, world_model_changed=False,
                age_charged_to_stopping_projection=True, other_dispatch_guards_changed=False,
                parallel_analysis_during_mission=False, final_evaluation=False,
                hardware_validated=False, extra_sources=source_hashes(),
                main_action_score_uses_predicted_outcomes=False,
                predictive_clearance_recovery_arrival_and_stopping_retained=arm=='instantaneous',
                full_learned_rollout_selection_disabled=arm!='instantaneous',
                model_forecasts_computed_for_workload_control=True, total_computation_identical=False,
                actual_dispatch_projection_retained=True, perception_and_measured_view_recovery_shared=True,
                actual_runtime_class=RUNTIMES[arm].__name__,
                turn_prediction_error_reserve_m=.03 if arm=='instantaneous' else 0.,
                translation_prediction_error_reserve_m=.03 if arm=='instantaneous' else 0.,
                current_action_reserve_m=.03 if arm=='reserved_off' else 0.,
                planned_stopping_projection=arm=='instantaneous',
                predictive_terminal_arrival_hold=arm=='instantaneous',
                terminal_feedback='measured_heading_then_forward_pulse' if arm=='reactive_feedback' else arm)
        bind(study.source.write, OUTPUT=output)(name, value)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(model, **kwargs):
            return RUNTIMES[arm](model, prediction_source='neural',
                registration_executor=executor, navigation_ticks=4800,
                arrival_radius_m=.02, **kwargs)

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=arm, MODEL_LOADER=bind(study.load_model, PLAN=PLAN),
            PacedNativeSession=partial(FreshCameraSession, noise_layout_index=index, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(initialize_pose, str(output)), MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()

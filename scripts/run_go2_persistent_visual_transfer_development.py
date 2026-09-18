"""Fixed learned/pose-command comparison on four prospective development mazes."""
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
from lewm.persistent_local_visual_recovery_development import PersistentLocalVisualRuntime
from lewm.cadenced_view_revisit_tracking_development import CadencedViewRevisitMotion, REVISIT_PERIOD_FRAMES
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation

ROOT = 'go2_persistent_visual_transfer_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
PLAN = Path('docs/go2_persistent_visual_transfer_plan_2026-09-16.json')
INVENTORY = Path('docs/go2_persistent_visual_transfer_layout_inventory_2026-09-16.json')
ASSIGNMENTS = tuple((i, a) for i in range(4) for a in
    (('supervised_rollout', 'pose_command') if i % 2 == 0 else ('pose_command', 'supervised_rollout')))
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
    'scripts/run_go2_paced_native_prefix_development.py')


class FreshPhysicalInit(study.cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(study.cohort.LiveDepthNoiseMixin, study.cohort.CompactDepthRetentionMixin,
        study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


def source_hashes():
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}


def prepare():
    if PLAN.exists() or INVENTORY.exists():
        raise ValueError('preserve the prospective plan and inventory')
    frozen = json.loads(study.PLAN.read_text())
    for i in (0, 1):
        root = study.BASE/f'go2_persistent_local_visual_supervised_rollout_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
        if not json.loads((root/'short_pulse_navigation_evaluation_v1.json').read_text())['round_trip']:
            raise ValueError('completed two-layout predecessor required')
        hashes = json.loads((root/'launch.json').read_text())['extra_sources']
        if any(source_hashes()[p] != hashes[p] for p in CONTROLLER_SOURCES):
            raise ValueError('hold the successful controller fixed')
    inventory = layouts.build_inventory()
    encoded = json.dumps(inventory, indent=2)+'\n'
    with INVENTORY.open('x') as stream: stream.write(encoded)
    plan = dict(schema='persistent_visual_transfer_plan.v1', assignments=ASSIGNMENTS,
        models=frozen['models'], training_seed=frozen['training_seed'],
        command_history_fit_sha256=frozen['command_history_fit_sha256'],
        source_sha256=source_hashes(), inventory_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        navigation_ticks=4800, depth_noise_sigma_mm=2, gyro_noise_model='ideal',
        model_selection_from_new_maze_outcomes=False, fixed_before_first_navigation=True,
        comparison='same controller, raw supervised neural vs fitted pose-command XY and command yaw',
        computations_matched='all alternative forecasts computed in both arms; same supervised model loaded',
        failures_retained=True, executions_per_condition_layout=1, independent_layout_count=4,
        final_evaluation=False, hardware_validated=False,
        limitations=['one training seed', 'same procedural maze family', 'ideal gyro',
            'not a prediction-on/off or JEPA attribution experiment', 'measured simulation, not real time'])
    with PLAN.open('x') as stream: json.dump(plan, stream, indent=2); stream.write('\n')
    print('PREPARED', len(ASSIGNMENTS), 'fixed assignments', flush=True)


def initialize_pose(output):
    from lewm import process_mapped_runtime_development as process
    study.previous.reference.initialize_pose()
    process._motion = CadencedViewRevisitMotion()
    receipt = dict(pid=os.getpid(), motion_class=type(process._motion).__name__,
        pose_class=type(process._motion.model).__name__, old_view_revisit_period_frames=REVISIT_PERIOD_FRAMES)
    with (Path(output)/'pose_worker_identity.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 9))
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
        selected = SimpleNamespace(**(vars(study) | {
            'ROOT': output.name, 'ASSIGNMENTS': ((index, arm),), 'PLAN': PLAN}))
        identity = json.loads((output/'pose_worker_identity.json').read_text())
        if identity['pose_class'] != 'CadencedViewRevisitPose' or identity['old_view_revisit_period_frames'] != 4:
            raise ValueError('actual prospective pose worker required')
        result = bind(evaluation.evaluate, study=selected)(1)
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
            value = value | dict(experiment='persistent_visual_transfer_v1',
                study_arm=arm, comparison_condition=arm,
                hardware=resources, fixed_sequential_assignments=plan['assignments'],
                fresh_layout_inventory=layouts.build_inventory(),
                frozen_layout_inventory_sha256=plan['inventory_sha256'],
                prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                new_independent_development_layout=True,
                layout_novelty_scope='distinct_from_explicit_84_layout_registry',
                planned_layout_count=4, planned_native_assignments=8,
                neural_reference_is_unused_for_control=arm=='pose_command',
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
                exposed_development_layout=False,
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
                hardware_validated=False, extra_sources=source_hashes())
        bind(study.source.write, OUTPUT=output)(name, value)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(model, **kwargs):
            return PersistentLocalVisualRuntime(model, prediction_source='pose_command' if arm=='pose_command' else 'neural',
                registration_executor=executor, navigation_ticks=4800,
                arrival_radius_m=.02, **kwargs)

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=arm, MODEL_LOADER=study.load_model,
            PacedNativeSession=partial(FreshCameraSession, noise_layout_index=index, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(initialize_pose, str(output)), MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()

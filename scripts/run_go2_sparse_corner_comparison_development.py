"""Five fixed treatments on unexecuted transfer layout zero."""
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
from lewm.axis_aligned_fine_connectivity_development import warmup
from lewm.projected_polygon_floor_coverage_development import initialize_mapping
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.sparse_corner_comparison_development import RUNTIMES
from scripts import run_go2_sparse_corner_completion_development as pilot
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation
from scripts import run_go2_persistent_visual_baselines_development as baselines
from scripts.read_go2_interrupted_view_replan_development import physical_return_edges
from scripts.navigation_artifact_root_development import validate_root

core = pilot.previous
study = core.study
native = pilot.native
BASE = pilot.BASE
PLAN = Path('docs/go2_sparse_corner_comparison_plan_2026-09-17.json')
INVENTORY = pilot.transfer.INVENTORY
ARMS = ('jepa', 'supervised_rollout', 'command_history', 'reactive_feedback',
    'jepa_no_route_turn_memory')
LAYOUT_INDEX = 0


def root_name(number):
    return f'go2_sparse_corner_comparison_{number:02d}_{ARMS[number-1]}_noise_2mm_native_layout00_4800_v1_attempt_001'


def sources():
    return pilot.sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'lewm/sparse_corner_comparison_development.py',
        'scripts/run_go2_sparse_corner_comparison_development.py')}


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def prepare():
    assert not PLAN.exists() and not any((BASE/root_name(n)).exists() for n in range(1, 6))
    frozen = json.loads(pilot.PLAN.read_text())
    assert pilot.sources() == frozen['source_sha256']
    outcome = json.loads((BASE/pilot.ROOT/'sparse_corner_completion_navigation_readout_v2.json').read_text())
    assert outcome['navigation']['round_trip'] and not outcome['pipeline_faults']
    models = json.loads(core.PLAN.read_text())['models']
    records = {arm: models['supervised_rollout' if arm == 'supervised_rollout' else
        'command_history' if arm == 'command_history' else 'jepa'] for arm in ARMS}
    plan = frozen | dict(schema='sparse_corner_comparison_plan.v1',
        assignments=[[LAYOUT_INDEX, arm] for arm in ARMS],
        root_names=[root_name(n) for n in range(1, 6)], models=records,
        source_sha256=sources(), planned_native_assignments=5,
        actual_runtime_classes={a: RUNTIMES[a].__name__ for a in ARMS},
        fresh_layout_indices=[0], unused_inventory_layout_indices=[],
        excluded_exposed_layout_indices=[1], reference_root=str(BASE/pilot.ROOT),
        intervention='fixed model, reactive-package and interrupted-route-turn-memory controls on one unused maze',
        new_independent_development_layout=True, exposed_development_layout=False,
        fixed_before_first_navigation_on_layout=True, controller_changes_during_batch=False,
        model_changes_during_batch=False, training_seed_count=1,
        cpu_group=[*range(8, 16), *range(24, 32)],
        native_jobs_run_sequentially=True, parallel_heavy_work_during_mission=False,
        comparisons=['JEPA versus same-data supervised prediction',
            'learned versus fitted command-history prediction with shared forecast consumers',
            'JEPA planner versus current-waypoint reactive feedback without forecast consumers',
            'JEPA with versus without interrupted-route-turn memory'],
        reactive_model_computed_for_workload_control=True,
        reactive_comparison_is_controller_package=True,
        memory_ablation_scope='interrupted route-turn memory only; observed map and visual recovery memory retained',
        primary_outcome='physically verified goal-and-home round trip without disallowed contact',
        secondary_outcomes=['tracking', 'contacts', 'physical backtracking', 'deadlines',
            'executed-window forecast errors', 'exercised route-turn memory'],
        preserve_every_failure=True, stop_on_scientific_failure=False,
        no_extra_repetitions_to_obtain_success=True,
        limitations=['one fresh same-family maze and one run per treatment',
            'one training seed', 'reactive baseline replaces predictive feasibility/recovery rules',
            'memory ablation is local, not full persistent-map removal',
            'measured simulation, 2mm depth noise and ideal gyro; no hardware validation'])
    plan.pop('actual_runtime_class', None)
    save(PLAN, plan)
    print('PREPARED', len(ARMS), 'fixed fresh-maze treatments', flush=True)


def evaluate(number):
    arm = ARMS[number-1]; root = BASE/root_name(number)
    selected = SimpleNamespace(BASE=BASE, ROOT=root.name, PLAN=PLAN,
        ASSIGNMENTS=((LAYOUT_INDEX, arm),))
    options = dict(study=selected,
        xy=bind(evaluation.xy, validate_root=bind(validate_root, BASE=BASE)))
    if arm == 'reactive_feedback':
        baselines.verify_selector(root, arm)
        options['save_or_read'] = baselines.save_receipt
    navigation = bind(evaluation.evaluate, **options)(1)
    plans = [p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p]
    exercised = [p for p in plans if p['selection'].get('visual_route_turn_memory', {}).get('active')]
    if arm in ('jepa_no_route_turn_memory', 'reactive_feedback'):
        assert not exercised
    identity = json.loads((root/'pose_worker_identity.json').read_text())
    assert identity['pose_class'] == 'SparseCornerCompletionPose'
    result = dict(schema='sparse_corner_comparison_navigation_readout.v1', assignment=number,
        arm=arm, navigation=navigation, physical_backtracking=physical_return_edges(root),
        pipeline_faults=json.loads((root/'pipeline_faults.json').read_text()),
        route_turn_memory_plans=len(exercised), new_independent_development_layout=True,
        exposed_layout=False, repeatability_or_causal_advantage_established=False)
    save(root/'sparse_corner_comparison_navigation_readout_v1.json', result)
    print('COMPARISON_EVALUATED', number, arm, navigation['round_trip'], flush=True)


def run(number):
    arm = ARMS[number-1]; output = BASE/root_name(number)
    plan = json.loads(PLAN.read_text())
    assert not output.exists() and sources() == plan['source_sha256']
    assert plan['assignments'] == [[LAYOUT_INDEX, a] for a in ARMS]
    if number > 1:
        assert (BASE/root_name(number-1)/'sparse_corner_comparison_navigation_readout_v1.json').exists()
    assert hashlib.sha256(INVENTORY.read_bytes()).hexdigest() == plan['inventory_sha256']
    assert sorted(os.sched_getaffinity(0)) == plan['cpu_group']
    if shutil.disk_usage(BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    warmup()
    hardware = native.baseline.hardware()
    reference = json.loads((BASE/pilot.ROOT/'launch.json').read_text())
    record = plan['models'][arm]; feature_arm = record['feature_arm']; runtimes = []

    def load_model(assignment):
        assert assignment == arm
        model = core.readout.load_readout(feature_arm)
        assert state_digest(model.state_dict()) == record['model_sha256']
        return model, 'supervised_rollout' if feature_arm == 'supervised_rollout' else 'jepa', 'full'

    def write(name, value):
        if name == 'launch.json':
            value = reference | value | dict(experiment='sparse_corner_comparison_v1',
                study_arm=arm, comparison_condition=arm, model_assignment=arm,
                training_condition=feature_arm, model_feature_condition=feature_arm,
                neural_snapshot=dict(format='frozen_parent_plus_ridge_motion_readout',
                    filename=str(core.readout.OUTPUT/feature_arm/'readout.npz'),
                    sha256=record['readout_sha256'], parent_model_state_sha256=record['parent_model_sha256'],
                    model_state_sha256=record['model_sha256']),
                hardware=hardware, batch_assignment=number, fixed_sequential_assignments=plan['assignments'],
                prospective_plan=str(PLAN), prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                extra_sources=plan['source_sha256'], actual_runtime_class=RUNTIMES[arm].__name__,
                planned_native_assignments=5, planned_layout_count=1,
                fresh_layout_inventory=json.loads(INVENTORY.read_text()),
                frozen_layout_inventory_sha256=plan['inventory_sha256'],
                layout_novelty_scope='unexecuted transfer inventory layout zero',
                exposed_development_layout=False, new_independent_development_layout=True,
                neural_reference_is_unused_for_control=arm in ('command_history', 'reactive_feedback'),
                route_turn_memory_enabled=arm not in ('jepa_no_route_turn_memory', 'reactive_feedback'),
                full_learned_rollout_selection_disabled=arm == 'reactive_feedback',
                reactive_comparison_is_controller_package=arm == 'reactive_feedback',
                reference_root_path=str(BASE/pilot.ROOT), reference_root_name=pilot.ROOT,
                output_base=str(BASE), model_input_base=str(core.readout.OUTPUT),
                final_evaluation=False, hardware_validated=False)
        bind(study.source.write, OUTPUT=output)(name, value)
        if name == 'requests.json' and runtimes:
            for filename, rows in (('visual_dispatch_events.json', runtimes[0].visual_dispatch_events),
                    ('planning_latency_stress.json', runtimes[0].clock_ns.rows),
                    ('live_planning_profile.json', runtimes[0].plan_profile_rows)):
                bind(study.source.write, OUTPUT=output)(filename, rows)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(native.baseline.registration_ready).result()
        def runtime(model, **kwargs):
            kwargs['clock_ns'] = native.PlanningLatencyClock(kwargs['clock_ns'], 20_000_000)
            instance = RUNTIMES[arm](model,
                prediction_source='command_history' if arm == 'command_history' else 'neural',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)
            runtimes.append(instance); return instance
        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=LAYOUT_INDEX,
            validate_root=bind(validate_root, BASE=BASE), specification=native.layouts.specification,
            public_mission=native.layouts.public_mission, MODEL_ASSIGNMENT=arm, MODEL_LOADER=load_model,
            PacedNativeSession=partial(native.FreshCameraSession, noise_layout_index=LAYOUT_INDEX, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(native.baseline.initialize_pose, str(output)),
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready, initialize_mapping=initialize_mapping)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 6))
    args = parser.parse_args()
    if args.prepare and args.assignment is None and not args.evaluate:
        prepare()
    elif not args.prepare and args.assignment is not None:
        evaluate(args.assignment) if args.evaluate else run(args.assignment)
    else:
        raise ValueError('prepare or run/evaluate one fixed assignment')

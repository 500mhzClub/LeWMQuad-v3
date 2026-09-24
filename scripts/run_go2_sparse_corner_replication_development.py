"""Frozen five-treatment comparison on three additional prospective mazes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil

from lewm.eligible_floor_registration_development import bind
from lewm import sparse_corner_replication_layouts_development as layouts
from scripts import run_go2_sparse_corner_comparison_development as previous

BASE = previous.BASE
PLAN = Path('docs/go2_sparse_corner_replication_plan_2026-09-17.json')
INVENTORY = Path('docs/go2_sparse_corner_replication_layout_inventory_2026-09-17.json')
ORDERS = (previous.ARMS, previous.ARMS[2:] + previous.ARMS[:2],
    previous.ARMS[4:] + previous.ARMS[:4])
ASSIGNMENTS = tuple((index, arm) for index, order in enumerate(ORDERS) for arm in order)
ARMS = tuple(arm for _, arm in ASSIGNMENTS)
READOUT = 'sparse_corner_comparison_navigation_readout_v1.json'


class FreshPhysicalInit(previous.native.FreshPhysicalInit):
    __init__ = bind(previous.native.study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(previous.native.NogilDrawingMixin,
        previous.native.study.cohort.LiveDepthNoiseMixin,
        previous.native.study.cohort.CompactDepthRetentionMixin,
        previous.native.study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


def root_name(number):
    index, arm = ASSIGNMENTS[number-1]
    return f'go2_sparse_corner_replication_{number:02d}_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


def sources():
    return previous.sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'lewm/sparse_corner_replication_layouts_development.py',
        'scripts/run_go2_sparse_corner_replication_development.py')}


def prepare():
    assert not PLAN.exists() and not INVENTORY.exists()
    assert not any((BASE/root_name(n)).exists() for n in range(1, 16))
    frozen = json.loads(previous.PLAN.read_text())
    assert previous.sources() == frozen['source_sha256']
    for number in range(1, 6):
        result = json.loads((BASE/previous.root_name(number)/READOUT).read_text())
        assert result['navigation']['round_trip'] and not result['pipeline_faults']
    previous.save(INVENTORY, layouts.build_inventory())
    plan = frozen | dict(schema='sparse_corner_replication_plan.v1',
        assignments=ASSIGNMENTS, root_names=[root_name(n) for n in range(1, 16)],
        source_sha256=sources(), inventory_sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        planned_native_assignments=15, planned_layout_count=3,
        fresh_layout_indices=[0, 1, 2], excluded_exposed_layout_indices=[],
        reference_root=str(BASE/previous.root_name(1)),
        intervention='replicate frozen five-controller comparison on three new maze geometries',
        order_varied_by_layout=True, controller_unchanged_from_repeatability_batch=True,
        retain_full_batch_depth_until_analysis=False,
        retention='Keep every failure in full. Completed successful depth eligible after per-run analysis; keep current full comparison reference until batch review.',
        limitations=['three fresh same-family mazes and one run per treatment per maze',
            'one training seed; three cyclic orders do not fully balance five treatments',
            'reactive baseline compares controller packages',
            'memory ablation removes local route-turn memory only',
            'asynchronous trajectories and deadline fractions differ',
            'ideal gyro and 2mm synthetic depth noise; measured simulation, no hardware validation'])
    previous.save(PLAN, plan)
    print('PREPARED 15 fixed assignments on three fresh mazes', flush=True)


def evaluate(number):
    index, _ = ASSIGNMENTS[number-1]
    bind(previous.evaluate, BASE=BASE, PLAN=PLAN, ARMS=ARMS,
        LAYOUT_INDEX=index, root_name=root_name)(number)


def run(number):
    index, arm = ASSIGNMENTS[number-1]
    output = BASE/root_name(number)
    plan = json.loads(PLAN.read_text())
    assert not output.exists() and sources() == plan['source_sha256']
    assert plan['assignments'] == [list(row) for row in ASSIGNMENTS]
    if number > 1:
        assert (BASE/root_name(number-1)/READOUT).exists()
    assert hashlib.sha256(INVENTORY.read_bytes()).hexdigest() == plan['inventory_sha256']
    assert sorted(os.sched_getaffinity(0)) == plan['cpu_group']
    if shutil.disk_usage(BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    previous.warmup()
    hardware = previous.native.baseline.hardware()
    reference = json.loads((BASE/previous.root_name(1)/'launch.json').read_text())
    record = plan['models'][arm]
    feature_arm = record['feature_arm']
    runtimes = []

    def load_model(assignment):
        assert assignment == arm
        model = previous.core.readout.load_readout(feature_arm)
        assert previous.state_digest(model.state_dict()) == record['model_sha256']
        return model, 'supervised_rollout' if feature_arm == 'supervised_rollout' else 'jepa', 'full'

    def write(name, value):
        if name == 'launch.json':
            value = reference | value | dict(experiment='sparse_corner_replication_v1',
                study_arm=arm, comparison_condition=arm, model_assignment=arm,
                training_condition=feature_arm, model_feature_condition=feature_arm,
                neural_snapshot=dict(format='frozen_parent_plus_ridge_motion_readout',
                    filename=str(previous.core.readout.OUTPUT/feature_arm/'readout.npz'),
                    sha256=record['readout_sha256'], parent_model_state_sha256=record['parent_model_sha256'],
                    model_state_sha256=record['model_sha256']),
                hardware=hardware, batch_assignment=number, fixed_sequential_assignments=plan['assignments'],
                prospective_plan=str(PLAN), prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                extra_sources=plan['source_sha256'], actual_runtime_class=previous.RUNTIMES[arm].__name__,
                planned_native_assignments=15, planned_layout_count=3,
                fresh_layout_inventory=json.loads(INVENTORY.read_text()),
                frozen_layout_inventory_sha256=plan['inventory_sha256'],
                layout_novelty_scope='three new mazes disjoint from the 100-layout development registry',
                exposed_development_layout=False, new_independent_development_layout=True,
                controller_unchanged_from_repeatability_batch=True,
                neural_reference_is_unused_for_control=arm in ('command_history', 'reactive_feedback'),
                route_turn_memory_enabled=arm not in ('jepa_no_route_turn_memory', 'reactive_feedback'),
                full_learned_rollout_selection_disabled=arm == 'reactive_feedback',
                reactive_comparison_is_controller_package=arm == 'reactive_feedback',
                reference_root_path=str(BASE/previous.root_name(1)), reference_root_name=previous.root_name(1),
                output_base=str(BASE), model_input_base=str(previous.core.readout.OUTPUT),
                final_evaluation=False, hardware_validated=False)
        bind(previous.study.source.write, OUTPUT=output)(name, value)
        if name == 'requests.json' and runtimes:
            for filename, rows in (('visual_dispatch_events.json', runtimes[0].visual_dispatch_events),
                    ('planning_latency_stress.json', runtimes[0].clock_ns.rows),
                    ('live_planning_profile.json', runtimes[0].plan_profile_rows)):
                bind(previous.study.source.write, OUTPUT=output)(filename, rows)

    previous.study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=previous.study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(previous.native.baseline.registration_ready).result()
        def runtime(model, **kwargs):
            kwargs['clock_ns'] = previous.native.PlanningLatencyClock(kwargs['clock_ns'], 20_000_000)
            instance = previous.RUNTIMES[arm](model,
                prediction_source='command_history' if arm == 'command_history' else 'neural',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)
            runtimes.append(instance)
            return instance
        bind(previous.study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=index,
            validate_root=bind(previous.validate_root, BASE=BASE), specification=layouts.specification,
            public_mission=layouts.public_mission, MODEL_ASSIGNMENT=arm, MODEL_LOADER=load_model,
            PacedNativeSession=partial(FreshCameraSession, noise_layout_index=index, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(previous.native.baseline.initialize_pose, str(output)),
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=previous.study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=previous.study.cohort.stable.obstacles_ready,
            initialize_mapping=previous.initialize_mapping)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 16))
    args = parser.parse_args()
    if args.prepare and args.assignment is None and not args.evaluate:
        prepare()
    elif not args.prepare and args.assignment is not None:
        evaluate(args.assignment) if args.evaluate else run(args.assignment)
    else:
        raise ValueError('prepare or run/evaluate one fixed assignment')

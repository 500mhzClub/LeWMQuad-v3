"""Two fixed model treatments on one fresh maze, using the current controller."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from lewm import route_turn_memory_transfer_layouts_development as layouts
from lewm.axis_aligned_fine_connectivity_development import AxisFineConnectivityMixin, warmup
from lewm.eligible_floor_registration_development import bind
from lewm.projected_polygon_floor_coverage_development import initialize_mapping
from scripts import run_go2_interrupted_route_turn_memory_development as pilot

previous = pilot.previous
BASE = pilot.BASE
PLAN = Path('docs/go2_route_turn_memory_transfer_plan_2026-09-17.json')
INVENTORY = Path('docs/go2_route_turn_memory_transfer_layout_inventory_2026-09-17.json')
ARMS = ('jepa', 'supervised_rollout')
RAW_WRITE = previous.study.source.write
OUTPUT = None


class FreshPhysicalInit(previous.native.FreshPhysicalInit):
    __init__ = bind(previous.native.study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(previous.native.NogilDrawingMixin,
        previous.native.study.cohort.LiveDepthNoiseMixin,
        previous.native.study.cohort.CompactDepthRetentionMixin,
        previous.native.study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


class TransferRuntime(AxisFineConnectivityMixin, pilot.RouteTurnMemoryRuntime):
    pass


native = SimpleNamespace(**(vars(previous.native) | dict(layouts=layouts,
    INVENTORY=INVENTORY, FreshCameraSession=FreshCameraSession)))


def root_name(number):
    return f'go2_route_turn_memory_transfer_{number:02d}_{ARMS[number-1]}_noise_2mm_native_layout01_4800_v1_attempt_001'


def sources():
    return pilot.sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'scripts/run_go2_route_turn_memory_transfer_development.py',
        'lewm/route_turn_memory_transfer_layouts_development.py',
        'lewm/axis_aligned_fine_connectivity_development.py')}


def write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='route_turn_memory_transfer_v1',
            actual_mapping_class='ProjectedPolygonFloorRoutingMap',
            projected_polygon_floor_coverage=True, route_turn_memory_enabled=True,
            axis_routing_acceleration_enabled=True, planned_native_assignments=2,
            fresh_layout_inventory=layouts.build_inventory(),
            controller_unchanged_from_repeatability_batch=False,
            reference_root_path=str(BASE / pilot.ROOT), reference_root_name=pilot.ROOT,
            world_model_changed=False, models_are_same_frozen_readouts=True,
            exposed_development_layout=False, new_independent_development_layout=True,
            final_evaluation=False, hardware_validated=False)
    bind(RAW_WRITE, OUTPUT=OUTPUT)(name, value)


def prepare():
    assert not PLAN.exists() and not INVENTORY.exists()
    assert not any((BASE / root_name(n)).exists() for n in (1, 2))
    assert pilot.sources() == json.loads(pilot.PLAN.read_text())['source_sha256']
    result = json.loads((BASE / pilot.ROOT / 'frozen_readout_navigation_readout_v1.json').read_text())
    assert result['navigation']['round_trip'] and not result['pipeline_faults']
    verification = json.loads((BASE / pilot.batch.root_name(1) / 'fine_goal_routing_profile_v1' /
        'axis_geometry_verification_v1.json').read_text())
    assert verification['all_four_saved_query_results_identical']
    for p, digest in verification['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    inventory = layouts.build_inventory()
    previous.previous.save(INVENTORY, inventory)
    frozen = json.loads(pilot.batch.PLAN.read_text())
    plan = frozen | dict(schema='route_turn_memory_transfer_plan.v1',
        assignments=[[1, a] for a in ARMS], root_names=[root_name(n) for n in (1, 2)],
        source_sha256=sources(), inventory_sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        planned_native_assignments=2, actual_runtime_class=TransferRuntime.__name__,
        reference_root=str(BASE / pilot.ROOT),
        intervention='fixed JEPA versus supervised readouts with interrupted route-turn memory and exact axis graph distances on fresh maze',
        axis_routing_acceleration_enabled=True,
        fresh_layout_indices=[1], unused_inventory_layout_indices=[0],
        exposed_development_layout=False, new_independent_development_layout=True,
        fixed_before_first_execution=True, controller_changes_during_batch=False,
        model_changes_during_batch=False, no_extra_repetitions_to_obtain_success=True,
        stop_on_scientific_failure=False, parallel_heavy_work_during_mission=False,
        preserve_every_failure=True, retain_full_batch_depth_until_analysis=True,
        unchanged=['frozen readouts', 'six candidates', 'polygon floor and obstacle geometry',
            'route-turn memory from completed pilot', 'tracking and view thresholds',
            'reserve, coverage, stopping and dispatch checks', '2mm depth noise and ideal gyro',
            '4800-tick budget', 'CPU group and planning deadlines'],
        limitations=['one new maze and one execution per model', 'one training seed',
            'no causal memory ablation in this pair', 'axis acceleration first used live here',
            'no final benchmark, realistic gyro validation or hardware evidence'])
    previous.previous.save(PLAN, plan)
    print('PREPARED fixed fresh-maze pair: JEPA, supervised', flush=True)


def run(number):
    warmup()
    original_mapper = previous.study.cohort.learned.initialize_mapping
    original_writer = previous.study.source.write
    previous.study.cohort.learned.initialize_mapping = initialize_mapping
    previous.study.source.write = write
    try:
        bind(previous.run, PLAN=PLAN, ARMS=ARMS, root_name=root_name, source_hashes=sources,
            Runtime=TransferRuntime, native=native)(number)
    finally:
        previous.study.cohort.learned.initialize_mapping = original_mapper
        previous.study.source.write = original_writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--assignment', type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.prepare and args.assignment is None and not args.evaluate:
        prepare()
    elif not args.prepare and args.assignment is not None:
        if args.evaluate:
            bind(previous.evaluate, PLAN=PLAN, ARMS=ARMS, root_name=root_name)(args.assignment)
        else:
            run(args.assignment)
    else:
        raise ValueError('prepare or run/evaluate one fixed assignment')

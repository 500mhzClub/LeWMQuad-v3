"""One prospective JEPA mission testing memory of weak route-turn directions."""
import argparse
import hashlib
import json
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.interrupted_route_turn_memory_development import InterruptedRouteTurnMemoryMixin
from lewm.projected_polygon_floor_coverage_development import initialize_mapping
from scripts import run_go2_polygon_floor_repeatability_development as batch

previous = batch.previous
BASE = batch.BASE
ROOT = 'go2_interrupted_route_turn_memory_jepa_noise_2mm_native_layout01_4800_v1_attempt_001'
PLAN = Path('docs/go2_interrupted_route_turn_memory_plan_2026-09-17.json')
RAW_WRITE = previous.study.source.write
OUTPUT = None


class RouteTurnMemoryRuntime(InterruptedRouteTurnMemoryMixin, previous.Runtime):
    pass


def root_name(number):
    assert number == 1
    return ROOT


def sources():
    return batch.sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'scripts/run_go2_interrupted_route_turn_memory_development.py',
        'lewm/interrupted_route_turn_memory_development.py',
        'lewm/tests/test_interrupted_route_turn_memory_development.py')}


def write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='interrupted_route_turn_memory_v1',
            actual_mapping_class='ProjectedPolygonFloorRoutingMap',
            projected_polygon_floor_coverage=True, route_turn_memory_enabled=True,
            axis_routing_acceleration_enabled=False,
            controller_unchanged_from_repeatability_batch=False,
            reference_root_path=str(BASE / batch.root_name(4)),
            reference_root_name=batch.root_name(4), world_model_changed=False,
            planned_native_assignments=1, models_are_same_frozen_readouts=True)
    bind(RAW_WRITE, OUTPUT=OUTPUT)(name, value)


def prepare():
    assert not (BASE / ROOT).exists() and not PLAN.exists()
    frozen = json.loads(batch.PLAN.read_text())
    assert batch.sources() == frozen['source_sha256']
    for n in range(1, 5):
        assert (BASE / batch.root_name(n) / 'frozen_readout_navigation_readout_v1.json').exists()
    activation = json.loads((BASE / batch.root_name(4) /
        'interrupted_route_turn_saved_activation_v1.json').read_text())
    assert activation['proposed_action'] == 'left_turn'
    assert activation['source_sha256'] == sources()['lewm/interrupted_route_turn_memory_development.py']
    plan = frozen | dict(schema='interrupted_route_turn_memory_plan.v1',
        assignments=[[1, 'jepa']], root_names=[ROOT], models={'jepa': frozen['models']['jepa']},
        source_sha256=sources(), planned_native_assignments=1,
        reference_root=str(BASE / batch.root_name(4)),
        actual_runtime_class=RouteTurnMemoryRuntime.__name__,
        intervention='remember locally observed weak route-turn direction and try the other forecast-clear direction after recovery',
        axis_routing_acceleration_enabled=False, focused_tests_passed=3,
        saved_first_changed_frame=activation['frame'],
        fixed_before_first_execution=True, no_extra_repetitions_to_obtain_success=True,
        unchanged=['frozen JEPA readout', 'six candidates', 'polygon floor and obstacle geometry',
            'tracking and visual-support thresholds', 'reserve, coverage, stopping and dispatch checks',
            'route-search implementation', '2mm depth noise and ideal gyro',
            '4800-tick budget', 'CPU group and planning deadlines'],
        limitations=['one exposed-maze pilot', 'no causal JEPA contribution established',
            'both turn directions may lose visual support', 'no hardware validation'])
    previous.previous.save(PLAN, plan)
    print('PREPARED one interrupted-route-turn memory pilot', flush=True)


def run():
    original_mapper = previous.study.cohort.learned.initialize_mapping
    original_writer = previous.study.source.write
    previous.study.cohort.learned.initialize_mapping = initialize_mapping
    previous.study.source.write = write
    try:
        bind(previous.run, PLAN=PLAN, ARMS=('jepa',), root_name=root_name,
            source_hashes=sources, Runtime=RouteTurnMemoryRuntime)(1)
    finally:
        previous.study.cohort.learned.initialize_mapping = original_mapper
        previous.study.source.write = original_writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.evaluate:
        bind(previous.evaluate, PLAN=PLAN, ARMS=('jepa',), root_name=root_name)(1)
    else:
        run()

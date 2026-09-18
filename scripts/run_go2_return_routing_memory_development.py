"""Fixed paired JEPA experiment: accumulated routing evidence on the return leg."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.return_routing_memory_development import RUNTIMES
from lewm import return_routing_memory_layouts_development as layouts
from scripts import run_go2_sparse_corner_replication_development as previous

BASE = previous.BASE
PLAN = Path('docs/go2_return_routing_memory_plan_2026-09-17.json')
INVENTORY = Path('docs/go2_return_routing_memory_layout_inventory_2026-09-17.json')
ASSIGNMENTS = ((0, 'persistent_return'), (0, 'current_pair_return'),
    (1, 'current_pair_return'), (1, 'persistent_return'))
ARMS = tuple(arm for _, arm in ASSIGNMENTS)
REFERENCE = previous.root_name(9)
READOUT = 'return_routing_memory_navigation_readout_v1.json'
RAW_WRITE = previous.previous.study.source.write


class FreshPhysicalInit(previous.FreshPhysicalInit):
    __init__ = bind(previous.previous.native.study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(previous.previous.native.NogilDrawingMixin,
        previous.previous.study.cohort.LiveDepthNoiseMixin,
        previous.previous.study.cohort.CompactDepthRetentionMixin,
        previous.previous.study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


def root_name(number):
    index, arm = ASSIGNMENTS[number - 1]
    return f'go2_return_routing_memory_{number:02d}_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


def sources():
    return previous.sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'lewm/selected_route_turn_memory_development.py',
        'lewm/return_routing_memory_development.py',
        'lewm/return_routing_memory_layouts_development.py',
        'lewm/tests/test_return_routing_memory_development.py',
        'scripts/run_go2_return_routing_memory_development.py')}


def prepare():
    assert not PLAN.exists() and not INVENTORY.exists()
    assert not any((BASE / root_name(n)).exists() for n in range(1, 5))
    old = json.loads(previous.PLAN.read_text())
    assert previous.sources() == old['source_sha256']
    previous.previous.save(INVENTORY, layouts.build_inventory())
    plan = old | dict(schema='return_routing_memory_plan.v1',
        assignments=ASSIGNMENTS, root_names=[root_name(n) for n in range(1, 5)],
        planned_native_assignments=4, planned_layout_count=2, fresh_layout_indices=[0, 1],
        source_sha256=sources(), inventory_sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        models={arm: old['models']['jepa'] for arm in RUNTIMES},
        reference_root=str(BASE / REFERENCE), controller_unchanged_from_repeatability_batch=False,
        actual_runtime_classes={arm: cls.__name__ for arm, cls in RUNTIMES.items()},
        intervention='use latest mapped pair for return routing only; identical persistent outbound routing',
        selected_turn_memory_correction_in_both_arms=True,
        accumulated_action_clearance_preserved=True,
        current_depth_dispatch_and_other_temporal_state_preserved=True,
        historical_routing_memory_result='persistent 4/4 versus latest-pair 0/4 on older development revisits; return role not isolated',
        no_extra_attempts_to_obtain_success=True,
        retention='Keep each new pair full until evaluated together; preserve failures and unresolved inputs. Completed success depth eligible under existing policy.',
        limitations=['two new same-family layouts, one execution per condition per layout',
            'return-only routing memory, not fully memoryless or model-internal-memory ablation',
            'asynchronous outbound trajectories may differ before treatment activation',
            'ideal gyro, 2mm depth noise, measured simulation; no hardware validation'])
    previous.previous.save(PLAN, plan)
    print('PREPARED four fixed return-memory assignments', flush=True)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='return_routing_memory_v1',
            planned_native_assignments=4, planned_layout_count=2,
            layout_novelty_scope='two prospective mazes excluding the 103-layout development registry',
            controller_unchanged_from_repeatability_batch=False,
            selected_route_turn_memory_enabled=True,
            routing_memory_ablation_phase='RETURN_ONLY',
            assigned_return_routing_scope=RUNTIMES[value['study_arm']].return_scope,
            accumulated_action_clearance_preserved=True,
            reference_root_path=str(BASE / REFERENCE), reference_root_name=REFERENCE)
    bind(RAW_WRITE, OUTPUT=OUTPUT)(name, value)


def run(number):
    core = previous.previous
    writer = bind(write, OUTPUT=BASE / root_name(number))
    source = SimpleNamespace(**(vars(core.study.source) | dict(write=writer)))
    study = SimpleNamespace(**(vars(core.study) | dict(source=source)))
    adapted = SimpleNamespace(**(vars(core) | dict(study=study, RUNTIMES=RUNTIMES,
        root_name=lambda number: REFERENCE)))
    bind(previous.run, BASE=BASE, PLAN=PLAN, INVENTORY=INVENTORY,
        ASSIGNMENTS=ASSIGNMENTS, root_name=root_name, sources=sources,
        previous=adapted, layouts=layouts, FreshCameraSession=FreshCameraSession,
        READOUT=READOUT)(number)


def evaluate(number):
    index, arm = ASSIGNMENTS[number - 1]
    root = BASE / root_name(number)
    plans = [p for p in json.loads((root / 'planning.json').read_text()) if 'selection' in p]
    scope_rows = []
    for p in plans:
        receipt = p['selection']['return_routing_memory_treatment']
        scope = p['selection']['routing_memory_scope']
        generation = receipt['planned_mission_generation']
        expected = RUNTIMES[arm].return_scope if generation > 0 else 'persistent'
        assert scope['condition'] == expected
        assert scope['accumulated_action_clearance_preserved']
        assert receipt['return_scope'] == RUNTIMES[arm].return_scope
        scope_rows.append(dict(frame=p['frame'], generation=generation,
            condition=scope['condition'], on_time=p['on_time'],
            routing_floor_cells=scope['routing_floor_cells'],
            retained_floor_cells=scope['retained_floor_cells'],
            routing_fine_obstacle_cells=scope['routing_fine_obstacle_cells'],
            retained_fine_obstacle_cells=scope['retained_fine_obstacle_cells']))
    memory = dict(schema='return_routing_memory_execution.v1',
        scope_verified_on_every_selected_plan=True,
        selected_plans=len(scope_rows),
        outbound_plans=sum(r['generation'] == 0 for r in scope_rows),
        return_plans=sum(r['generation'] > 0 for r in scope_rows),
        return_plans_with_reduced_floor=sum(r['generation'] > 0 and
            r['routing_floor_cells'] < r['retained_floor_cells'] for r in scope_rows),
        return_plans_with_reduced_obstacles=sum(r['generation'] > 0 and
            r['routing_fine_obstacle_cells'] < r['retained_fine_obstacle_cells'] for r in scope_rows),
        rows=scope_rows, accumulated_action_clearance_preserved=True,
        fully_memoryless_controller=False)
    previous.previous.save(root / 'return_routing_memory_execution_v1.json', memory)

    def save(path, value):
        if path.name == 'sparse_corner_comparison_navigation_readout_v1.json':
            path = path.with_name(READOUT)
            value = value | dict(schema='return_routing_memory_navigation_readout.v1',
                return_memory_execution={k: v for k, v in memory.items() if k != 'rows'})
        previous.previous.save(path, value)
    bind(previous.previous.evaluate, BASE=BASE, PLAN=PLAN, ARMS=ARMS,
        LAYOUT_INDEX=index, root_name=root_name, save=save)(number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--assignment', type=int, choices=range(1, 5))
    args = parser.parse_args()
    if args.prepare and args.assignment is None and not args.evaluate:
        prepare()
    elif not args.prepare and args.assignment is not None:
        evaluate(args.assignment) if args.evaluate else run(args.assignment)
    else:
        raise ValueError('prepare or run/evaluate one fixed assignment')

"""One exposed-layout native test of memory after clearance selection."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.selected_route_turn_memory_development import SelectedRouteTurnMemoryMixin
from scripts import run_go2_sparse_corner_replication_development as previous

BASE = previous.BASE
PLAN = Path('docs/go2_selected_route_turn_memory_pilot_attempt02_plan_2026-09-17.json')
ROOT = 'go2_selected_route_turn_memory_command_history_noise_2mm_native_layout01_4800_v1_attempt_002'
REFERENCE = previous.root_name(6)
ARM = 'command_history'
READOUT = 'selected_route_turn_memory_navigation_readout_v1.json'


class SelectedTurnMemoryRuntime(SelectedRouteTurnMemoryMixin, previous.previous.RUNTIMES[ARM]):
    pass


def root_name(number):
    assert number == 1
    return ROOT


def sources():
    return previous.sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'lewm/selected_route_turn_memory_development.py',
        'lewm/tests/test_selected_route_turn_memory_development.py',
        'scripts/run_go2_selected_route_turn_memory_pilot_development.py')}


def prepare():
    assert not (BASE / ROOT).exists()
    plan = json.loads(previous.PLAN.read_text())
    assert previous.sources() == plan['source_sha256']
    probe = json.loads((BASE / REFERENCE / 'turn_memory_selected_turn_probe_v1.json').read_text())
    assert probe['all_recorded_actions_and_memory_flags_matched']
    assert len(probe['isolated_recorded_state_probes']) == 45
    result = plan | dict(schema='selected_route_turn_memory_pilot_plan.v1',
        assignments=[[1, ARM]], root_names=[ROOT], planned_native_assignments=1,
        planned_layout_count=1, source_sha256=sources(), reference_root=str(BASE / REFERENCE),
        exposed_development_layout=True, new_independent_development_layout=False,
        fresh_layout_indices=[], controller_unchanged_from_repeatability_batch=False,
        intervention='match interrupted turns after clearance selection, retaining every parent gate',
        actual_runtime_class='SelectedTurnMemoryRuntime',
        unchanged=['tracker and pose gates', 'frozen command-history prediction and unused neural workload',
            'map and routing', 'six candidate actions', 'visual recovery and clearance',
            '2mm depth noise, ideal gyro, 4800 ticks, CPU affinity and deadlines'],
        limitations=['single exposed-layout developmental intervention',
            'asynchronous trajectories differ; no isolated causal success claim',
            'command-history control pilot, not JEPA superiority or independent generalization',
            'measured simulation; no hardware validation'],
        no_extra_attempts_to_obtain_success=True)
    result['predecessor_launch_failure'] = dict(
        plan='docs/go2_selected_route_turn_memory_pilot_plan_2026-09-17.json',
        native_navigation_started=False, fix='closure-free launch writer adapter')
    previous.previous.save(PLAN, result)
    print('PREPARED one exposed-layout selected-turn-memory pilot', flush=True)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='selected_route_turn_memory_pilot_v1',
            planned_native_assignments=1, planned_layout_count=1,
            exposed_development_layout=True, new_independent_development_layout=False,
            layout_novelty_scope='previously exposed replication layout one',
            controller_unchanged_from_repeatability_batch=False,
            selected_route_turn_memory_enabled=True,
            reference_root_path=str(BASE / REFERENCE), reference_root_name=REFERENCE)
    bind(previous.previous.study.source.write, OUTPUT=BASE / ROOT)(name, value)


def run():
    core = previous.previous

    # Reuse the unchanged physical runner with explicit local dependencies.
    # No original module globals or frozen experiment files are modified.
    source = SimpleNamespace(**(vars(core.study.source) | dict(write=write)))
    study = SimpleNamespace(**(vars(core.study) | dict(source=source)))
    adapted = SimpleNamespace(**(vars(core) | dict(study=study,
        RUNTIMES={ARM: SelectedTurnMemoryRuntime}, root_name=lambda number: REFERENCE)))
    bind(previous.run, BASE=BASE, PLAN=PLAN, ASSIGNMENTS=((1, ARM),),
        root_name=root_name, sources=sources, previous=adapted)(1)


def evaluate():
    def save(path, value):
        if path.name == 'sparse_corner_comparison_navigation_readout_v1.json':
            path = path.with_name(READOUT)
            value = value | dict(schema='selected_route_turn_memory_navigation_readout.v1',
                new_independent_development_layout=False, exposed_layout=True)
        previous.previous.save(path, value)
    bind(previous.previous.evaluate, BASE=BASE, PLAN=PLAN, ARMS=(ARM,),
        LAYOUT_INDEX=1, root_name=root_name, save=save)(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else evaluate() if args.evaluate else run()

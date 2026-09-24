"""Fixed prospective four-layout comparison using the existing dense runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts import train_go2_horizon_dense_predictor_development as fit
from scripts import train_go2_full_heading_readout_development as head

PLAN = Path('docs/go2_dense_world_model_maze_execution_plan_2026-09-22.json')
INVENTORY = Path('docs/go2_dense_world_model_maze_inventory_2026-09-18.json')
BASE = fit.OUTPUT.parent
ARMS = ('action', 'no_future_action', 'command_history', 'reactive_feedback')
ASSIGNMENTS = tuple((index, arm) for index in range(4)
    for arm in ARMS[index:] + ARMS[:index])
SOURCES = (__file__, 'scripts/run_go2_dense_horizon_navigation_development.py',
    'scripts/read_go2_dense_horizon_navigation_development.py',
    'lewm/dense_horizon_navigation_development.py',
    'lewm/dense_world_model_maze_layouts_development.py')


def root_name(index, arm):
    return f'go2_dense_world_model_maze_layout{index:02d}_{arm}_mixed_data_v1_attempt_001'


def identities():
    return dict(sources={str(p): fit.digest(p) for p in SOURCES},
        inventory=fit.digest(INVENTORY),
        checkpoints={arm: fit.digest(fit.OUTPUT/f'{arm}_final.pt') for arm in fit.ARMS},
        readout=fit.digest(head.OUTPUT/'mixed_data_final.pt'))


def prepare():
    assert not PLAN.exists()
    assert not any((BASE/root_name(*assignment)).exists() for assignment in ASSIGNMENTS)
    for name in ('go2_dense_horizon_untimed_action_mixed_data_readout_exposed_maze_full_v1_attempt_001',
            'go2_dense_horizon_untimed_reactive_feedback_exposed_maze_full_v1_attempt_001'):
        result = json.loads((BASE/name/'dense_navigation_readout.json').read_text())
        assert result['physical']['round_trip_arrival_checks_passed']
        assert not result['physical']['disallowed_contact_samples'] and not result['pipeline_faults']
    plan = dict(assignments=ASSIGNMENTS, identities=identities(),
        readout_arm='mixed_data', depth_retention='rgb_only', navigation_ticks=4800,
        timing='untimed synchronous simulation', depth_noise_sigma_mm=2, gyro='ideal simulator gyro',
        checkpoint_selection='fixed from exposed development results before prospective execution',
        primary_outcome='native-verified outbound and return arrivals without disallowed contact',
        source_or_model_changes_during_cohort=False, extra_repetitions_to_obtain_success=False,
        stop_on_navigation_failure=False, preserve_all_outcomes=True,
        execution='sequential GPU/native jobs; cyclic arm order by layout',
        parallelism='one GPU serves encoder and predictor; heavy concurrent native/GPU jobs deferred',
        retention='RGB, commands, physics, perception, depth hashes and outcomes for every run; no raw depth arrays',
        limitations=['four same-family development layouts; one run per arm and layout',
            'action/blind comparison tests future-action conditioning, not JEPA objective causality',
            'reactive comparison replaces a controller package',
            'persistent-memory and JEPA-training contributions need additional experiments',
            'not real-time, hardware validated, cross-family or sealed final evaluation'])
    fit.save(PLAN, plan)
    print('PREPARED_PROSPECTIVE_COMPARISON', len(ASSIGNMENTS), flush=True)


def run(number):
    plan = json.loads(PLAN.read_text())
    assert plan['assignments'] == [list(x) for x in ASSIGNMENTS]
    assert plan['identities'] == identities(), 'fixed cohort identities changed'
    index, arm = ASSIGNMENTS[number-1]
    root = BASE/root_name(index, arm)
    if root.exists():
        raise ValueError('preserve existing assignment; inspect its terminal state, never overwrite')
    command = [sys.executable, 'scripts/run_go2_dense_horizon_navigation_development.py',
        '--full-mission', '--arm', arm, '--readout-arm', 'mixed_data',
        '--depth-retention', 'rgb_only', '--prospective-layout', str(index)]
    print('PROSPECTIVE_ASSIGNMENT_START', number, index, arm, flush=True)
    owner = subprocess.run(command)
    if (root/'result.json').exists() or (root/'failure.json').exists():
        subprocess.run([sys.executable, 'scripts/read_go2_dense_horizon_navigation_development.py',
            '--root-name', root.name], check=True)
    if owner.returncode:
        raise RuntimeError(f'Native owner exited {owner.returncode}; preserved assignment {number}')
    result = json.loads((root/'dense_navigation_readout.json').read_text())
    assert result['new_independent_development_layout'] and result['layout_index'] == index
    print('PROSPECTIVE_ASSIGNMENT_COMPLETE', number, index, arm,
        result['physical']['round_trip_arrival_checks_passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--assignment', type=int, choices=range(1, 17))
    group.add_argument('--run-all', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.assignment is not None:
        run(args.assignment)
    else:
        for number, assignment in enumerate(ASSIGNMENTS, 1):
            root = BASE/root_name(*assignment)
            if (root/'dense_navigation_readout.json').exists() and not (root/'failure.json').exists():
                print('PRESERVED_COMPLETED_ASSIGNMENT', number, flush=True)
                continue
            run(number)

"""Mission entry admitting the stage-2 patch layout sets (development; Andrew, 5 October 2026).

Wraps scripts/run_go2_dev_mission_development.py (unchanged). It adds one loader, for the registered stage-2 sets
(scripts/register_go2_stage2_patch_sets_development.py; registry <capability root>/stage2_sets_v1_registry.json):
- stage2_fit: mazes 0-23;
- stage2_heldout: mazes 24-29;
- stage2_eval: mazes 30-49.
Each maze has one registered episode. Files are hash-verified against the registry, and the role must match the set
name. Every other set goes to the unchanged loader. The arguments and main() are the unchanged entry's.
"""
import argparse
import json
from pathlib import Path

from scripts import run_go2_dev_mission_development as dev

STAGE2 = dict(stage2_fit=range(0, 24), stage2_heldout=range(24, 30), stage2_eval=range(30, 50))
REGISTRY = 'stage2_sets_v1_registry.json'
_original_loader_for = dev.loader_for


def stage2_inputs(set_name):
    def inputs(root, maze, episode):
        if maze not in STAGE2[set_name]:
            raise ValueError(f'{set_name} admits mazes {STAGE2[set_name].start}-{STAGE2[set_name].stop-1}')
        registry = json.loads((Path(root)/REGISTRY).read_text())
        entry = next(e for e in registry['entries'] if e['maze_id'] == maze)
        if entry['role'] != set_name or entry['episode'] != episode:
            raise ValueError(f'maze {maze} is registered as {entry["role"]} episode {entry["episode"]}')
        paths = {}
        for key, record in (('maze', entry['maze']), ('episode', entry['episodes'][0])):
            path = Path(record['path'])
            assert path.parent.name == set_name and dev.sha(path) == record['sha256'], path
            paths[key] = path
        packet = json.loads(paths['episode'].read_text())
        original = json.loads(paths['maze'].read_text())
        assert packet['role'] == original['data_role'] == set_name
        spec = original | dict(procedural_seed=packet['simulation_seed'],
                               geometry=original['geometry'] | dict(spawn_se2_world=packet['home_se2_world']))
        return spec, packet
    return inputs


def loader_for(set_name, maze, allow_final_round):
    if set_name in STAGE2:
        return stage2_inputs(set_name)
    return _original_loader_for(set_name, maze, allow_final_round)


dev.loader_for = loader_for  # module global, read by dev.main


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--controller', required=True, choices=('C0', 'C1', 'C2', 'C3', 'C4'))
    p.add_argument('--set', required=True, choices=('dev_tune', 'validation', 'fresh_check', 'round', 'prelim_test', *STAGE2))
    p.add_argument('--maze', type=int, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--assignment', required=True)
    p.add_argument('--fixes', default='')
    p.add_argument('--recovery', choices=('on', 'off'))
    p.add_argument('--degrade')
    p.add_argument('--margin', choices=('p95', 'p99'))
    p.add_argument('--pessimistic-unknown', action='store_true')
    p.add_argument('--c3-decoder')
    p.add_argument('--c4-weights')
    p.add_argument('--allow-final-round', action='store_true')
    a = p.parse_args()
    fixes = [f for f in a.fixes.split(',') if f]
    if a.recovery and fixes:
        p.error('--recovery and --fixes are exclusive')
    recovery = a.recovery or (None if fixes else dev.DEFAULT_RECOVERY)
    if recovery:
        fixes = dev.fixes_for(recovery)
    dev.main(a.controller, a.set, a.maze, a.episode, a.assignment, fixes, a.c3_decoder, a.c4_weights, a.allow_final_round,
             recovery, a.degrade, a.margin, a.pessimistic_unknown)

"""Register fresh mazes for the approved C3 readout fix (Andrew Knowles, 29 September 2026).

Roles, fixed before construction, disjoint from every existing set (development,
validation, sealed test, the eight audit layouts, and all prior registries including
the maze-view training and transfer layouts):
  - accepted layouts 0-9:   fresh_check, one episode each (closed-loop check of C1, C3-v2, refit C4);
  - accepted layouts 10-13: rest_turn_recording, fit split (training-role recordings);
  - accepted layouts 14-15: rest_turn_recording, held-out split (offline acceptance only).
The capability inventory (including sealed layouts) is reconstructed in memory only to
exclude its graphs. It is verified against the registry by file hash alone and is never
written, printed or returned.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil

from lewm import decision_headroom_json_v42_development as output
from lewm import independent_round_trip_layouts_development as generator
from lewm.eligible_floor_registration_development import bind
from scripts import generate_go2_navigation_capability_sets_development as capability

REPO = Path(__file__).resolve().parents[1]
CONSTRUCTION_SEED = 2026092901
LAYOUT_COUNT = 16
SEEDS = dict(physics_seed_base=2026113000, appearance_seed_base=2026113100, episode_seed_base=2026113200)
ROOT_NAME = 'c3v2_sets_v1'


def role(index):
    if index < 10:
        return 'fresh_check'
    return 'rest_turn_recording'


def split(index):
    return None if index < 10 else ('fit' if index < 14 else 'heldout')


def capability_inventory(protocol):
    def make_spec(index, links, identity, candidate_index):
        spec = bind(generator.make_spec, PHYSICS_SEED_BASE=protocol['generator']['physics_seed_base'],
                    APPEARANCE_SEED_BASE=protocol['generator']['appearance_seed_base'])(index, links, identity, candidate_index)
        return spec | dict(scene_id=f'navigation-capability-v1-{index:02d}', family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY',
                           data_role='dev_tune' if index < 10 else 'validation' if index < 30 else 'sealed_test')
    return bind(generator.build_inventory, prior_graphs=capability.prior_graphs, make_spec=make_spec,
                LAYOUT_COUNT=90, CONSTRUCTION_SEED=protocol['generator']['construction_seed'])()


def main():
    protocol = json.loads(capability.PROTOCOL.read_text())
    assert capability.digest(capability.PROTOCOL.read_bytes()) == capability.PROTOCOL_SHA
    base = Path(protocol['output_root'])
    registry = json.loads((base/'registry.json').read_text())
    root = base/ROOT_NAME
    if shutil.disk_usage(base).free < protocol['caps']['recovery_reserve_bytes']+64*1024**2:
        raise RuntimeError('filesystem reserve')
    root.mkdir(exist_ok=False)
    output.install(root)
    # 1. In-memory reconstruction of the 90 capability layouts, verified by hash only.
    inventory = capability_inventory(protocol)
    expected = {e['maze_id']: e['maze']['sha256'] for e in registry['entries']}
    verified = sum(hashlib.sha256((output.dumps(s, indent=2)+'\n').encode()).hexdigest() == expected[s['layout_index']]
                   for s in inventory['layouts'])
    if verified != 90:
        raise ValueError(f'capability reconstruction does not match the registry ({verified}/90); stop')
    excluded = capability.prior_graphs() + [dict(name=s['scene_id'], edges=s['evaluation_layout']['edges'])
                                            for s in inventory['layouts']]
    del inventory

    # 2. New construction with every prior graph excluded.
    def make_spec(index, links, identity, candidate_index):
        spec = bind(generator.make_spec, PHYSICS_SEED_BASE=SEEDS['physics_seed_base'],
                    APPEARANCE_SEED_BASE=SEEDS['appearance_seed_base'])(index, links, identity, candidate_index)
        return spec | dict(scene_id=f'navigation-c3v2-{index:02d}', family='NAVIGATION_CAPABILITY_SAME_MAZE_FAMILY',
                           data_role=role(index), recording_split=split(index))
    new = bind(generator.build_inventory, prior_graphs=lambda: excluded, make_spec=make_spec,
               LAYOUT_COUNT=LAYOUT_COUNT, CONSTRUCTION_SEED=CONSTRUCTION_SEED)()
    episode_protocol = copy.deepcopy(protocol)
    episode_protocol['generator'].update(episode_seed_base=SEEDS['episode_seed_base'], physics_seed_base=SEEDS['physics_seed_base'])
    entries = []
    for spec in new['layouts']:
        index = spec['layout_index']
        folder = root/'sets'/spec['data_role']
        row = dict(maze_id=index, role=spec['data_role'], split=spec['recording_split'],
                   maze=capability.write(folder/f'maze_{index:02d}.json', spec), episodes=[])
        if spec['data_role'] == 'fresh_check':
            packet = capability.episode(spec, 0, episode_protocol)
            row['episodes'].append(capability.write(folder/f'episode_{index:02d}_0.json', packet))
            row['shortest_outbound_m'] = packet['shortest_outbound_m']
        entries.append(row)
    result = dict(schema='navigation_c3v2_set_registry.v1', construction_seed=CONSTRUCTION_SEED, seeds=SEEDS,
        counts=dict(fresh_check=10, rest_turn_recording_fit=4, rest_turn_recording_heldout=2),
        capability_reconstruction_hash_verified=verified, excluded_prior_graphs=len(excluded),
        candidates_examined=new['candidates_examined'], rejection_counts=new['rejection_counts'],
        disjoint_from_all_prior_sets=True, sealed_contents_written_or_displayed=False,
        physics_executed=False, rendering_performed=False, entries=entries)
    receipt = capability.write(root/'registry.json', result)
    print(json.dumps(dict(status='REGISTERED', registry=receipt, counts=result['counts'],
        candidates_examined=result['candidates_examined'], rejection_counts=result['rejection_counts'],
        check_shortest_outbound_m=[round(e['shortest_outbound_m'], 2) for e in entries if 'shortest_outbound_m' in e])))


if __name__ == '__main__':
    main()
